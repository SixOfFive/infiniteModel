"""#40 tensor-parallel crossover benchmark.

Measures SINGLE-STREAM tok/s for a model under: consolidate (1 node), GPU TP (tp=2),
and CPU TP (tp=2 / tp=4). The question: does slicing a model across nodes ever beat
running it whole on one node, on this fabric? (Pipeline-splitting never wins single-stream;
TP can, when each node's saved memory-bandwidth/compute outweighs the per-layer all-reduce.)

Each config: load -> warm 1 req -> time N reqs (best of) -> unload. A config that can't fit
(load-full-then-shard needs the FULL model per rank) is recorded N/A with the reason.
stdlib only. Usage: python tp_bench.py
"""
import json, time, urllib.request, urllib.parse

import wire
_cfg = wire.load_config()
BASE = f"http://{_cfg['controller_host']}:{_cfg['http_port']}"   # from config.json
NTOK = 64
RUNS = 2   # best-of (single-stream is what we measure, so just take the fastest clean run)
PROMPT = "Write a short technical paragraph explaining how a CPU cache hierarchy works."

# (label, params for /load). tp=1 => consolidate (no TP). cpu_only routes to RAM.
CONFIGS = [
    # --- GPU: does TP beat one GPU for a model that fits one GPU? (expect: no) ---
    ("1.5b GPU consolidate",  {"model": "qwen2.5-1.5b", "ctx": 2048, "mode": "single"}),
    ("1.5b GPU tp=2",         {"model": "qwen2.5-1.5b", "ctx": 2048, "tp": 2}),
    # --- CPU: the user's question — slice a model across CPUs for single-stream speedup ---
    ("1.5b CPU consolidate",  {"model": "qwen2.5-1.5b", "ctx": 2048, "cpu_only": "true"}),
    ("1.5b CPU tp=2",         {"model": "qwen2.5-1.5b", "ctx": 2048, "tp": 2, "cpu_only": "true"}),
    ("1.5b CPU tp=4",         {"model": "qwen2.5-1.5b", "ctx": 2048, "tp": 4, "cpu_only": "true"}),
    # --- bigger model (more RAM-bandwidth-bound) where CPU TP should help most, if ever ---
    ("7b CPU consolidate",    {"model": "qwen2.5-7b", "ctx": 2048, "cpu_only": "true", "quant": "int8"}),
    ("7b CPU tp=2",           {"model": "qwen2.5-7b", "ctx": 2048, "tp": 2, "cpu_only": "true", "quant": "int8"}),
    ("7b GPU consolidate",    {"model": "qwen2.5-7b", "ctx": 2048, "mode": "single", "quant": "int8"}),
    ("7b GPU tp=2",           {"model": "qwen2.5-7b", "ctx": 2048, "tp": 2, "quant": "int8"}),
]


def post(path, timeout=1800, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load(params):
    return post("/load?" + urllib.parse.urlencode(params), timeout=1800)


def unload(model):
    try:
        post("/unload?" + urllib.parse.urlencode({"model": model}), timeout=180)
    except Exception:
        pass


def one_req(model):
    body = {"model": model, "prompt": PROMPT, "stream": False,
            "options": {"num_predict": NTOK, "temperature": 0}}
    t0 = time.time()
    j = post("/api/generate", timeout=900, body=body)
    dt = time.time() - t0
    n = int(j.get("eval_count") or 0)
    # prefer the server's eval timing if present (excludes prompt eval / network)
    ed = j.get("eval_duration")
    srv = (n / (ed / 1e9)) if ed else None
    return n, dt, (n / dt if dt else 0), srv, j.get("error")


def placements(model):
    d = json.loads(urllib.request.urlopen(BASE + "/status", timeout=10).read())
    cards = [m for m in d.get("cluster", {}).get("loaded_models", [])
             if (m.get("base") or m.get("friendly")) == model]
    hosts = []
    for m in cards:
        hosts += [s.get("hostname") for s in m.get("stages", [])]
    # for TP, all ranks share one stage card; read node assignments instead
    nodes = [n["hostname"] for n in d["nodes"] if n.get("stage") is not None]
    return nodes or hosts


def run():
    print(f"#40 TP crossover  num_predict={NTOK}  best-of-{RUNS}\n")
    results = []
    for label, params in CONFIGS:
        model = params["model"]
        unload(model)
        try:
            r = load(dict(params))
        except Exception as e:
            msg = str(e)
            print(f"  {label:<22} N/A (load failed: {msg[:80]})")
            results.append((label, None, msg[:60])); continue
        if not r.get("ok"):
            print(f"  {label:<22} N/A ({r.get('error','?')[:80]})")
            results.append((label, None, str(r.get("error"))[:60])); continue
        hosts = placements(model)
        try:
            one_req(model)  # warm
            best = max(one_req(model) for _ in range(RUNS))  # best by tok/s (idx2)
            n, dt, wallrate, srv, err = best
            rate = srv or wallrate
            tag = "srv" if srv else "wall"
            print(f"  {label:<22} {rate:6.1f} tok/s ({tag})  hosts={hosts}")
            results.append((label, rate, hosts))
        except Exception as e:
            print(f"  {label:<22} N/A (gen failed: {str(e)[:70]})")
            results.append((label, None, str(e)[:60]))
        unload(model)

    print("\n=== CROSSOVER SUMMARY ===")
    def get(lbl):
        return next((r for (l, r, _) in results if l == lbl and r), None)
    for fam, base, variants in [
        ("1.5b GPU", "1.5b GPU consolidate", ["1.5b GPU tp=2"]),
        ("1.5b CPU", "1.5b CPU consolidate", ["1.5b CPU tp=2", "1.5b CPU tp=4"]),
        ("7b CPU",   "7b CPU consolidate",   ["7b CPU tp=2"]),
        ("7b GPU",   "7b GPU consolidate",   ["7b GPU tp=2"]),
    ]:
        b = get(base)
        if not b:
            print(f"  {fam}: baseline N/A"); continue
        for v in variants:
            r = get(v)
            if r:
                print(f"  {fam}: {v.split()[-1]} {r:.1f} vs consolidate {b:.1f}  -> "
                      f"{r/b:.2f}x ({'TP WINS' if r > b else 'consolidate wins'})")
            else:
                print(f"  {fam}: {v} N/A")


if __name__ == "__main__":
    run()
