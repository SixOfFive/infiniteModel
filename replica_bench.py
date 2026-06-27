"""#39 data-parallel replication — real-fleet throughput benchmark.

Loads a SMALL model first as 1 copy, then as N disjoint copies, and fires the same
burst of concurrent requests at each. With 1 copy all requests serialize on the single
replica's lock (aggregate ~= single-stream rate); with N copies up to N run at once
(aggregate ~= N x). Also polls /status mid-burst to prove requests are distributed
across replicas (per-replica `active`). stdlib only.

Usage: python replica_bench.py [model] [replicas] [concurrency] [num_predict]
"""
import sys, json, time, threading, urllib.request, urllib.parse

import wire
_cfg = wire.load_config()
BASE = f"http://{_cfg['controller_host']}:{_cfg['http_port']}"   # from config.json
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-1.5b"
NREP  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CONC  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
NTOK  = int(sys.argv[4]) if len(sys.argv) > 4 else 128
PROMPT = ("Explain, in a few sentences, why distributing a small model across many "
          "machines does not by itself make a single response faster.")


def _post(path, timeout=900, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(path, timeout=15):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load(replicas):
    q = urllib.parse.urlencode({"model": MODEL, "ctx": 2048, "replicas": replicas})
    return _post("/load?" + q, timeout=1800)


def unload():
    try:
        _post("/unload?" + urllib.parse.urlencode({"model": MODEL}), timeout=120)
    except Exception as e:
        print("  (unload:", e, ")")


def gen(i, results):
    body = {"model": MODEL, "prompt": PROMPT, "stream": False,
            "options": {"num_predict": NTOK, "temperature": 0}}
    t0 = time.time()
    try:
        j = _post("/api/generate", timeout=900, body=body)
        results[i] = {"tokens": int(j.get("eval_count") or 0),
                      "dt": time.time() - t0, "err": j.get("error")}
    except Exception as e:
        results[i] = {"tokens": 0, "dt": time.time() - t0, "err": repr(e)}


def poll_active(stop, peak):
    """Track the peak number of replicas decoding simultaneously (per-replica active>0)."""
    while not stop.is_set():
        try:
            d = _get("/status", timeout=6)
            cards = [m for m in d.get("cluster", {}).get("loaded_models", [])
                     if (m.get("base") or m.get("friendly")) == MODEL]
            busy = sum(1 for m in cards if (m.get("active") or 0) > 0)
            peak[0] = max(peak[0], busy)
        except Exception:
            pass
        time.sleep(0.25)


def burst(label):
    results = {}
    stop = threading.Event(); peak = [0]
    poller = threading.Thread(target=poll_active, args=(stop, peak), daemon=True)
    poller.start()
    t0 = time.time()
    threads = [threading.Thread(target=gen, args=(i, results)) for i in range(CONC)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    stop.set(); poller.join(timeout=2)
    toks = sum(r["tokens"] for r in results.values())
    ok = sum(1 for r in results.values() if not r["err"] and r["tokens"] > 0)
    errs = [r["err"] for r in results.values() if r["err"]]
    agg = toks / wall if wall else 0
    print(f"  [{label}] {ok}/{CONC} ok  tokens={toks}  wall={wall:.1f}s  "
          f"AGG={agg:.1f} tok/s  peak_concurrent_replicas={peak[0]}")
    if errs:
        print(f"     errors: {errs[:3]}")
    return agg, peak[0]


def placements():
    d = _get("/status")
    cards = sorted([m for m in d.get("cluster", {}).get("loaded_models", [])
                    if (m.get("base") or m.get("friendly")) == MODEL],
                   key=lambda m: m.get("replica_idx", 0))
    for m in cards:
        hosts = [s.get("hostname") for s in m.get("stages", [])]
        print(f"     replica {m.get('replica_idx')}: key={m['friendly']:<18} hosts={hosts}")
    return cards


if __name__ == "__main__":
    print(f"model={MODEL} concurrency={CONC} num_predict={NTOK} replicas-under-test={NREP}")
    print("raising queue_depth so admission isn't the limiter...")
    _post("/config?" + urllib.parse.urlencode({"queue_depth": max(16, CONC * 2)}), timeout=30)

    print(f"\n[1] load 1 copy ...")
    print("   ", load(1).get("ok"))
    placements()
    burst("warmup"); a1, _ = burst("1 copy")

    print(f"\n[{NREP}] load {NREP} disjoint copies ...")
    r = load(NREP)
    print("    ok=", r.get("ok"), "replicas=", r.get("replicas"))
    cards = placements()
    # disjoint check
    allhosts = [s.get("hostname") for m in cards for s in m.get("stages", [])]
    disjoint = len(allhosts) == len(set(allhosts))
    print(f"    disjoint placement: {disjoint}")
    burst("warmup"); aN, peak = burst(f"{NREP} copies")

    print(f"\n=== RESULT ===")
    print(f"  1 copy : {a1:.1f} tok/s aggregate")
    print(f"  {NREP} copies: {aN:.1f} tok/s aggregate")
    print(f"  speedup: {aN/a1:.2f}x   (ideal ~{min(NREP, CONC)}x)   "
          f"peak concurrent replicas during burst: {peak}")
    unload()
