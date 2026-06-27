"""Consolidation thesis test: pipeline-parallel is for FIT, not single-stream
speed. Load 7B two ways and time plain greedy decode each way:
  - spread:      across the whole pool (consolidate=false)
  - consolidated: fewest strongest nodes that fit (consolidate=true) -> BEAST alone
Reports tok/s from Ollama's own eval_count/eval_duration.
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:11434"
MODEL = "qwen2.5-7b"
PROMPT = "Explain in detail how a modern CPU executes instructions out of order."
MAX = 48


def load(consolidate):
    url = f"{BASE}/load?model={MODEL}&ctx=2048&consolidate={'true' if consolidate else 'false'}"
    with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=1200) as r:
        o = json.loads(r.read())
    stages = [(s["hostname"], s["num_layers"]) for s in o.get("stages", [])]
    return o.get("ok"), stages


def gen(max_new=MAX):
    body = {"model": MODEL, "prompt": PROMPT, "stream": False,
            "options": {"temperature": 0, "num_predict": max_new}}
    req = urllib.request.Request(BASE + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        o = json.loads(r.read())
    n = o.get("eval_count", 0)
    dur = (o.get("eval_duration", 0) or 1) / 1e9
    return n, n / dur, o.get("response", "")


def run(label, consolidate):
    print(f"\n== {label} ==", flush=True)
    ok, stages = load(consolidate)
    print(f"load ok={ok}  stages={stages}", flush=True)
    gen(8)                                  # warm-up
    n, ts, txt = gen()
    print(f"plain decode: {n} tok @ {ts:.2f} tok/s", flush=True)
    return ts, stages, txt


ts_spread, st_spread, txt_s = run("SPREAD (whole pool)", False)
ts_cons, st_cons, txt_c = run("CONSOLIDATED (fewest strongest)", True)

print("\n== RESULT ==", flush=True)
print(f"spread       : {ts_spread:.2f} tok/s  across {len(st_spread)} nodes {st_spread}", flush=True)
print(f"consolidated : {ts_cons:.2f} tok/s  across {len(st_cons)} node(s) {st_cons}", flush=True)
if ts_spread:
    print(f"speedup      : {ts_cons/ts_spread:.2f}x", flush=True)
print(f"outputs identical: {txt_s == txt_c}", flush=True)
