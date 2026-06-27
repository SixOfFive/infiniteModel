"""Load a big (int4) model on the cluster and verify it end-to-end.

Registers a clean quant-agnostic name (via /add_model?name=), loads at the given quant
(ctx auto-fits), then reports placement, resident footprint, the context it settled on, and
runs a few generations for coherence + tok/s. Reusable for MiniMax-M2 and GLM-4.6.

Usage: python load_verify.py <hf_id> <clean_name> [quant=int4] [ctx=0]
  e.g. python load_verify.py ModelCloud/MiniMax-M2-BF16 minimax-m2 int4
       python load_verify.py zai-org/GLM-4.6 glm-4.6 int4
"""
import sys, json, time, urllib.request, urllib.parse

import wire
_cfg = wire.load_config()
BASE = f"http://{_cfg['controller_host']}:{_cfg['http_port']}"   # from config.json
HF    = sys.argv[1] if len(sys.argv) > 1 else "ModelCloud/MiniMax-M2-BF16"
NAME  = sys.argv[2] if len(sys.argv) > 2 else "minimax-m2"
QUANT = sys.argv[3] if len(sys.argv) > 3 else "int4"
CTX   = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 0 => native training ctx (auto-fits)

PROMPTS = [
    "In one sentence, what is a Mixture-of-Experts model?",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three differences between TCP and UDP.",
]


def post(path, timeout=3600, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path, timeout=15):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def card(name):
    d = get("/status")
    cs = [m for m in d.get("cluster", {}).get("loaded_models", [])
          if (m.get("base") or m.get("friendly")) == name]
    return cs[0] if cs else None


def main():
    print(f"=== load_verify: {HF} as '{NAME}' quant={QUANT} ctx={CTX or 'native'} ===")
    print("controller:", get("/status")["controller"]["version"])

    # 1) register the clean name (idempotent; instant if HF id already cached)
    r = post("/add_model?" + urllib.parse.urlencode({"model": HF, "name": NAME}), timeout=600)
    print("register:", r.get("ok"), r.get("friendly"), "status=", r.get("status"))

    # 2) load at the requested quant (ctx auto-fits if the native ctx overcommits)
    print("loading (this can take a while for a 100GB+ int4 build)...")
    t0 = time.time()
    lr = post("/load?" + urllib.parse.urlencode({"model": NAME, "quant": QUANT, "ctx": CTX}),
              timeout=7200)
    print(f"load ok={lr.get('ok')} mode={lr.get('mode')} quant={lr.get('quant')} "
          f"ctx={lr.get('ctx')} in {time.time()-t0:.0f}s")
    if not lr.get("ok"):
        print("LOAD FAILED:", lr.get("error")); return

    # 3) placement + footprint
    c = card(NAME)
    if c:
        hosts = [(s["hostname"], f"L{s['layer_start']}-{s['layer_end']}") for s in c.get("stages", [])]
        print(f"  resident: ctx={c.get('ctx')} kv_reserved={c.get('kv_reserved_gb')}GB "
              f"weights: vram={c.get('vram_used_gb')}GB ram={c.get('ram_used_gb')}GB "
              f"size_gb(spec)={c.get('size_gb')} quant={c.get('quant')} params={c.get('params')}")
        print(f"  stages ({len(hosts)}): {hosts}")
    print("  pool:", {k: get('/status')['pool'].get(k) for k in ('used_gb', 'free_gb', 'usable_gb')})

    # 4) generate — coherence + tok/s
    for p in PROMPTS:
        t0 = time.time()
        g = post("/api/generate", timeout=900, body={
            "model": NAME, "prompt": p, "stream": False,
            "options": {"num_predict": 80, "temperature": 0}})
        dt = time.time() - t0
        n = int(g.get("eval_count") or 0)
        ed = g.get("eval_duration")
        rate = (n / (ed / 1e9)) if ed else (n / dt if dt else 0)
        print(f"\n  Q: {p}\n  A: {repr(g.get('response',''))[:240]}")
        print(f"  [{n} tok, {rate:.1f} tok/s, wall {dt:.1f}s, err={g.get('error')}]")

    print("\n=== DONE ===  client model name:", NAME)


if __name__ == "__main__":
    main()
