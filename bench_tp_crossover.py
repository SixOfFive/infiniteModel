#!/usr/bin/env python3
"""TP-vs-pipeline crossover benchmark harness for InfiniteModel.

WHAT THIS MEASURES
------------------
Now that heterogeneous tensor parallelism works (#68), the open question is *where*
tensor-parallel placement (tp=N: every layer split across N GPU nodes, all-reduce mesh)
starts to beat plain pipeline placement (layer-split across the fleet) — and how that
crossover moves with model size. Small models decode fastest collapsed onto one box
(pipeline auto); big models that spill to CPU RAM (or don't fit a single GPU) may decode
faster spread tensor-parallel across GPUs. This script sweeps a configurable model list
through several placement variants, times a fixed short generation for each, and emits a
table you can read the crossover off of.

For each (model, variant) cell it:
  1. unloads everything (clean slate),
  2. POST /load with the variant's params (pipeline auto, or tp=2 / tp=4, +quant),
  3. reads /status to capture the realized placement (hosts, layer split, plan basis),
  4. POST /api/generate (stream=false) with a fixed prompt + num_predict, twice:
     a short warm-up generation (page-in weights / build KV) then the timed run,
  5. records decode tok/s (eval_count / eval_duration), load time, and placement basis,
  6. unloads the model before the next cell.
Results are written as JSON (full detail, machine-readable) plus a printed markdown
summary table grouped by model so the per-variant tok/s sit side by side.

METHODOLOGY NOTES / CAVEATS
---------------------------
- tok/s is DECODE throughput: eval_count / eval_duration (the controller reports both,
  in nanoseconds, from /api/generate with stream=false). Prompt eval is excluded — this
  is a decode-latency comparison, which is what TP-vs-pipeline actually trades on.
- A warm-up generation runs first and is DISCARDED so the timed run isn't paying for
  first-token weight fault-in / KV allocation. Disable with --no-warmup.
- A variant is SKIPPED (not failed) when the fleet can't satisfy it — e.g. tp=4 with
  fewer than 4 GPU nodes, or a model that won't fit under that placement. /load returns
  ok:false with an error; that error is recorded so the table shows WHY a cell is blank.
- num_predict is small by default (64) to keep a full sweep quick; raise it (--tokens)
  for steadier numbers on fast models. Decode tok/s is fairly num_predict-insensitive
  once past the warm-up, but more tokens => less noise.
- This does ONE load per cell (no repeats across loads). For load-time noise, re-run the
  whole sweep; the JSON is timestamped so runs don't clobber each other.
- DESTRUCTIVE to controller STATE: it loads and unloads models on the live fleet. It does
  NOT change config, restart anything, or touch weights on disk. Run it when the fleet is
  otherwise idle — a concurrent load will perturb the timings (and fight for memory).

HOW TO RUN
----------
  # default models, pipeline-auto + tp=2 + tp=4, against the default controller:
  python bench_tp_crossover.py

  # explicit controller + a custom model list + bigger generation:
  python bench_tp_crossover.py --host <controller-host> --port 21434 \
      --models qwen2.5-7b llama-3.1-8b deepseek-70b --tokens 128

  # only compare pipeline vs tp=2, int4 quant, write results next to a tag:
  python bench_tp_crossover.py --variants auto tp2 --quant int4 --tag nightly

Output: writes  bench_tp_crossover_<tag>_<UTCstamp>.json  and prints the markdown table.
Stdlib only (urllib/json/time/argparse) — no third-party deps, runs anywhere python does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Defaults. The controller default mirrors the live fleet (BEAST:21434); override
# with --host/--port. Variants map a short name -> the /load query params that
# realize that placement. "auto" is plain pipeline placement (the planner's
# GPU-first default); "tpN" forces a tensor-parallel group of size N (tp>1
# overrides mode in the controller).
# ----------------------------------------------------------------------------
import wire as _wire
_dcfg = _wire.load_config()
DEFAULT_HOST = _dcfg["controller_host"]   # from config.json
DEFAULT_PORT = _dcfg["http_port"]
DEFAULT_MODELS = ["qwen2.5-7b", "llama-3.1-8b", "deepseek-70b"]
DEFAULT_VARIANTS = ["auto", "tp2", "tp4"]
DEFAULT_TOKENS = 64
DEFAULT_PROMPT = (
    "In one paragraph, explain how tensor parallelism splits a transformer layer "
    "across multiple machines and why it can lower decode latency for large models."
)

# variant name -> /load params (besides model/ctx/quant which are filled per-run).
VARIANT_PARAMS = {
    "auto": {"mode": "auto", "tp": 1},
    "single": {"mode": "single", "tp": 1},
    "distribute": {"mode": "distribute", "tp": 1},
    "tp2": {"tp": 2},
    "tp4": {"tp": 4},
    "tp6": {"tp": 6},
    "tp8": {"tp": 8},
}

NS_PER_S = 1_000_000_000


# ----------------------------------------------------------------------------
# Tiny HTTP layer (stdlib urllib). The controller takes /load and /unload as
# QUERY params (not a JSON body) and /api/generate as a JSON body; we mirror that.
# ----------------------------------------------------------------------------
def _url(base: str, path: str, params: dict | None = None) -> str:
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return url


def _http(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> tuple[int, dict]:
    """Issue one request. Returns (status_code, parsed-json-or-{}). Network/HTTP
    errors are turned into (status, {"error": ...}) so the caller can record a
    blocked cell instead of crashing the whole sweep."""
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, (json.loads(raw) if raw else {})
            except json.JSONDecodeError:
                return resp.status, {"error": f"non-JSON response: {raw[:200]}"}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        try:
            return exc.code, (json.loads(raw) if raw else {"error": f"HTTP {exc.code}"})
        except json.JSONDecodeError:
            return exc.code, {"error": f"HTTP {exc.code}: {raw[:200]}"}
    except (urllib.error.URLError, OSError) as exc:
        return 0, {"error": f"connection failed: {exc}"}


class Controller:
    """Thin client over the controller HTTP API used by the harness."""

    def __init__(self, host: str, port: int, gen_timeout: float, load_timeout: float):
        self.base = f"http://{host}:{port}"
        self.gen_timeout = gen_timeout
        self.load_timeout = load_timeout

    def status(self) -> dict:
        _, body = _http("GET", _url(self.base, "/status"), timeout=30.0)
        return body

    def unload(self, model: str) -> dict:
        _, body = _http("POST", _url(self.base, "/unload", {"model": model}),
                        timeout=self.load_timeout)
        return body

    def loaded_keys(self) -> set:
        return {m.get("internal_name") for m in (self.status().get("models") or [])
                if m.get("loaded")}

    def model_card(self, model: str) -> dict:
        for m in self.status().get("models") or []:
            if m.get("internal_name") == model:
                return m
        return {}

    def safe_unload(self, model: str, protect: set) -> None:
        """Unload ONE model by name unless it's protected (e.g. qwen3-4b). NEVER
        unload-all — a coexisting protected model must survive the whole sweep."""
        if model in protect:
            return
        self.unload(model)
        for _ in range(60):                      # wait until the card flips loaded=false
            if model not in self.loaded_keys():
                return
            time.sleep(2)

    def load(self, model: str, ctx: int, quant: str, params: dict) -> tuple[int, dict]:
        q = {"model": model, "quant": quant}
        if ctx > 0:
            q["ctx"] = ctx
        q.update(params)
        return _http("POST", _url(self.base, "/load", q), timeout=self.load_timeout)

    def generate(self, model: str, prompt: str, num_predict: int) -> tuple[int, dict]:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,   # single JSON response carrying eval_count + eval_duration
            "options": {"num_predict": int(num_predict), "temperature": 0.0},
        }
        return _http("POST", _url(self.base, "/api/generate"), body=body,
                     timeout=self.gen_timeout)


# ----------------------------------------------------------------------------
# Status / placement readers. After a load, /status carries loaded_models[] with
# the realized plan (stages -> hosts + layer ranges, plan_basis, quant). We pull
# the entry matching the model we just loaded.
# ----------------------------------------------------------------------------
def _find_loaded(status: dict, model: str) -> dict | None:
    for lm in status.get("loaded_models") or []:
        # match on the friendly name OR the base/target the controller resolved to.
        if model in (lm.get("friendly"), lm.get("base"), lm.get("target")):
            return lm
    # fall back to the single "loaded" panel if present.
    lm = status.get("loaded")
    if lm and model in (lm.get("friendly"), lm.get("base"), lm.get("target")):
        return lm
    return None


def _placement_summary(lm: dict | None) -> dict:
    """Compress a loaded-model entry's stages into a compact placement record."""
    if not lm:
        return {"basis": "", "hosts": [], "num_stages": 0, "stages": []}
    stages = lm.get("stages") or []
    hosts = [s.get("hostname", "?") for s in stages]
    compact = [
        {
            "host": s.get("hostname", "?"),
            "layers": f"{s.get('layer_start')}-{s.get('layer_end')}",
            "gpu_gb": s.get("gpu_gb", 0),
            "est_gb": s.get("est_gb", 0),
        }
        for s in stages
    ]
    return {
        "basis": lm.get("plan_basis", ""),
        "quant": lm.get("quant", ""),
        "ctx": lm.get("ctx", 0),
        "hosts": hosts,
        "num_stages": len(stages),
        "stages": compact,
    }


def _decode_toks(gen: dict) -> float | None:
    """Decode tok/s from an /api/generate response (eval_count / eval_duration[ns])."""
    n = gen.get("eval_count")
    dur_ns = gen.get("eval_duration")
    if not n or not dur_ns:
        return None
    return round(n / (dur_ns / NS_PER_S), 3)


# ----------------------------------------------------------------------------
# Sweep.
# ----------------------------------------------------------------------------
def run_cell(ctrl: Controller, model: str, variant: str, args) -> dict:
    """Load one (model, variant), time a generation, unload. Returns a result row."""
    protect = set(args.protect)
    params = dict(VARIANT_PARAMS[variant])
    if args.cpu_only:                      # force RAM placement for an apples-to-apples
        params["cpu_only"] = True          # CPU pipeline-vs-TP crossover (no VRAM)
    row: dict = {
        "model": model,
        "variant": variant,
        "params": params,
        "quant": args.quant,
        "ctx": args.ctx,
        "status": "pending",
    }
    print(f"  [{variant}] loading {model} (quant={args.quant}, "
          f"params={params}) ...", flush=True)

    # Clear ONLY this cell's model (never unload-all — a protected coexisting model
    # such as qwen3-4b must stay up for the whole sweep).
    ctrl.safe_unload(model, protect)

    t0 = time.perf_counter()
    code, lr = ctrl.load(model, args.ctx, args.quant, params)
    load_s = round(time.perf_counter() - t0, 3)
    row["load_s"] = load_s
    row["load_http"] = code

    if not (isinstance(lr, dict) and lr.get("ok")):
        # Fleet can't satisfy this variant (invalid tp, won't fit, etc.) -> skip.
        err = (lr or {}).get("error", f"load failed (HTTP {code})")
        row["status"] = "skipped"
        row["error"] = err
        print(f"    skipped: {err}", flush=True)
        ctrl.safe_unload(model, protect)
        return row

    row["load_mode"] = lr.get("mode", "")
    row["load_warnings"] = lr.get("warnings", [])

    # Authoritative placement = the /load response's stages (hostnames + layer ranges).
    # /status node rows are ambiguous under coexistence (a node hosting both this model
    # and the protected one shows only one assignment).
    stages = lr.get("stages") or []
    hosts = [s.get("hostname", "?") for s in stages]
    row["placement"] = {"mode": lr.get("mode", ""), "n_stages": len(stages),
                        "hosts": hosts, "stages": stages}
    print(f"    placed: mode={lr.get('mode')!r} stages={len(stages)} hosts={hosts}",
          flush=True)

    # Warm-up (discarded): pay first-token weight fault-in / KV alloc here.
    if not args.no_warmup:
        wc, _ = ctrl.generate(model, args.prompt, num_predict=min(8, args.tokens))
        if wc not in (200, 0):
            print(f"    warm-up HTTP {wc} (continuing)", flush=True)

    # Timed runs (two), keep the best — decode tok/s is noisy on a shared CPU fleet.
    best = None
    for i in range(args.repeats):
        gc, gen = ctrl.generate(model, args.prompt, num_predict=args.tokens)
        row["gen_http"] = gc
        if gc != 200 or "error" in (gen or {}):
            err = (gen or {}).get("error", f"generate failed (HTTP {gc})")
            row["status"] = "gen_error"
            row["error"] = err
            print(f"    generate error: {err}", flush=True)
            ctrl.safe_unload(model, protect)
            return row
        t = _decode_toks(gen)
        if t and (best is None or t > best["tok_s"]):
            best = {"tok_s": t, "gen": gen}
        print(f"    run{i+1}: {t} tok/s ({gen.get('eval_count')} tok)", flush=True)

    gen = best["gen"]
    row["tok_s"] = best["tok_s"]
    row["eval_count"] = gen.get("eval_count")
    row["eval_duration_ms"] = round((gen.get("eval_duration") or 0) / 1e6, 1)
    row["prompt_eval_count"] = gen.get("prompt_eval_count")
    row["total_duration_ms"] = round((gen.get("total_duration") or 0) / 1e6, 1)
    # Cross-check against the controller's own smoothed decode tok/s (model.last_tok_s).
    card = ctrl.model_card(model)
    row["status_tok_s"] = card.get("tok_s")
    row["status_ema_tok_s"] = card.get("ema_tok_s")
    row["size_gb"] = card.get("size_gb")
    row["status"] = "ok"
    print(f"    BEST {row['tok_s']} tok/s (status={card.get('tok_s')}), load {load_s}s",
          flush=True)

    ctrl.safe_unload(model, protect)
    return row


def run_sweep(ctrl: Controller, args, save=None) -> list[dict]:
    rows: list[dict] = []
    for model in args.models:
        print(f"\n== {model} ==", flush=True)
        for variant in args.variants:
            rows.append(run_cell(ctrl, model, variant, args))
            if save:                       # persist after every cell (kill-safe)
                save(rows)
    # Do NOT unload-all: protected models (e.g. qwen3-4b) must survive the sweep;
    # each cell already cleared its own (non-protected) model.
    return rows


# ----------------------------------------------------------------------------
# Reporting.
# ----------------------------------------------------------------------------
def _cell_text(row: dict) -> str:
    if row["status"] == "ok":
        return f"{row['tok_s']} ({row.get('load_s', '?')}s)"
    if row["status"] == "skipped":
        return "skip"
    return "ERR"


def markdown_summary(rows: list[dict], args, meta: dict) -> str:
    """Per-model table: rows=models, cols=variants, cell=tok/s (load_s). Plus a
    crossover note flagging, per model, which variant won on decode tok/s."""
    variants = args.variants
    by_model: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["variant"]] = r

    out: list[str] = []
    out.append(f"# TP-vs-pipeline crossover — {meta['timestamp']}")
    out.append("")
    out.append(f"Controller `{meta['controller']}` · quant `{args.quant}` · "
               f"ctx `{args.ctx or 'native'}` · {args.tokens} tok/run · "
               f"warmup `{not args.no_warmup}`")
    out.append("")
    out.append("Cells are decode **tok/s** (load seconds). `skip` = fleet couldn't "
               "place that variant; `ERR` = generate failed.")
    out.append("")

    header = "| model | " + " | ".join(variants) + " | best |"
    sep = "|" + "---|" * (len(variants) + 2)
    out.append(header)
    out.append(sep)
    for model in args.models:
        cells = by_model.get(model, {})
        line = [model]
        best_v, best_t = None, -1.0
        for v in variants:
            r = cells.get(v)
            line.append(_cell_text(r) if r else "-")
            if r and r["status"] == "ok" and (r.get("tok_s") or -1) > best_t:
                best_t, best_v = r["tok_s"], v
        line.append(f"**{best_v}** ({best_t})" if best_v else "-")
        out.append("| " + " | ".join(line) + " |")
    out.append("")

    # Crossover read-out: where does a tpN variant beat pipeline auto?
    out.append("## Crossover (tpN vs pipeline `auto`)")
    out.append("")
    tp_variants = [v for v in variants if v.startswith("tp")]
    if "auto" not in variants or not tp_variants:
        out.append("_(need both `auto` and a `tpN` variant in the sweep to compute)_")
    else:
        for model in args.models:
            cells = by_model.get(model, {})
            base = cells.get("auto")
            if not base or base["status"] != "ok":
                out.append(f"- **{model}**: no pipeline-auto baseline (status "
                           f"{base['status'] if base else 'missing'})")
                continue
            bt = base["tok_s"] or 0.0
            parts = []
            for v in tp_variants:
                r = cells.get(v)
                if not r or r["status"] != "ok" or not r.get("tok_s"):
                    parts.append(f"{v}=n/a")
                    continue
                ratio = (r["tok_s"] / bt) if bt else 0.0
                verdict = "WINS" if ratio > 1.0 else "loses"
                parts.append(f"{v}={r['tok_s']} ({ratio:.2f}x {verdict})")
            out.append(f"- **{model}** (auto={bt} tok/s): " + ", ".join(parts))
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark where tensor-parallel beats pipeline placement, by model size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Loads/unloads models on the LIVE controller — run when the fleet is idle.",
    )
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"controller host (default {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"controller port (default {DEFAULT_PORT})")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="model names to sweep (controller-resolvable)")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    choices=sorted(VARIANT_PARAMS.keys()),
                    help="placement variants to compare (default: auto tp2 tp4)")
    ap.add_argument("--quant", default="none", choices=["none", "int8", "int4"],
                    help="quant for every load (default none/bf16)")
    ap.add_argument("--ctx", type=int, default=0,
                    help="context length per load (0 = model's native, default)")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                    help=f"tokens to generate per timed run (default {DEFAULT_TOKENS})")
    ap.add_argument("--repeats", type=int, default=2,
                    help="timed runs per cell, best kept (default 2)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="generation prompt")
    ap.add_argument("--cpu-only", action="store_true",
                    help="force every load to RAM (cpu_only=true) — clean CPU "
                         "pipeline-vs-TP crossover with no VRAM in play")
    ap.add_argument("--protect", nargs="*", default=["qwen3-4b"],
                    help="models that must NEVER be unloaded (default: qwen3-4b)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the discarded warm-up generation before each timed run")
    ap.add_argument("--gen-timeout", type=float, default=600.0,
                    help="per-generate HTTP timeout seconds (default 600)")
    ap.add_argument("--load-timeout", type=float, default=1800.0,
                    help="per-load/unload HTTP timeout seconds (default 1800)")
    ap.add_argument("--tag", default="run", help="tag embedded in the output filename")
    ap.add_argument("--out", default="",
                    help="explicit JSON output path (default auto-named with tag + UTC stamp)")
    args = ap.parse_args(argv)

    ctrl = Controller(args.host, args.port, args.gen_timeout, args.load_timeout)

    # Sanity: is the controller reachable before we start mutating its state?
    st = ctrl.status()
    if not st or "error" in st:
        print(f"ERROR: controller {args.host}:{args.port} not reachable: "
              f"{(st or {}).get('error', 'no /status')}", file=sys.stderr)
        return 2
    gpu_nodes = sum(1 for n in (st.get("nodes") or []) if n.get("vram_total_gb", 0) > 0)
    print(f"controller {args.host}:{args.port} up — "
          f"{len(st.get('nodes') or [])} nodes ({gpu_nodes} with GPU). "
          f"Sweeping {len(args.models)} models x {len(args.variants)} variants.")

    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or f"bench_tp_crossover_{args.tag}_{stamp}.json"
    meta = {
        "controller": f"{args.host}:{args.port}",
        "timestamp": started.strftime("%Y-%m-%d %H:%M:%SZ"),
        "models": args.models,
        "variants": args.variants,
        "quant": args.quant,
        "ctx": args.ctx,
        "tokens": args.tokens,
        "repeats": args.repeats,
        "cpu_only": args.cpu_only,
        "protect": args.protect,
        "warmup": not args.no_warmup,
        "gpu_nodes": gpu_nodes,
        "total_nodes": len(st.get("nodes") or []),
    }

    def save(rows):                        # kill-safe incremental write after each cell
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "results": rows}, fh, indent=2)

    rows = run_sweep(ctrl, args, save=save)
    finished = datetime.now(timezone.utc)
    meta["finished"] = finished.strftime("%Y-%m-%d %H:%M:%SZ")
    meta["duration_s"] = round((finished - started).total_seconds(), 1)
    summary_md = markdown_summary(rows, args, meta)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": rows, "summary_md": summary_md},
                  fh, indent=2)

    print("\n" + summary_md)
    print(f"\nwrote {out_path}  ({len(rows)} cells, {meta['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
