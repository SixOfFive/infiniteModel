"""Head-to-head: M2e plain decode vs M3 speculative decode on the loaded 7B target.

Same greedy prompt both ways. Greedy is required for the speculative path to engage
(and it makes the two outputs directly comparable -- they must be identical).
Reports tok/s from Ollama's own eval_count / eval_duration so timing excludes
prompt prefill and HTTP overhead.
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:11434"
MODEL = "qwen2.5-7b"
PROMPT = "List the first eight prime numbers and then explain in two sentences why two is the only even prime."
MAX = 48


def gen(speculative, max_new=MAX):
    body = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_new, "speculative": speculative},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    n = out.get("eval_count", 0)
    dur_ns = out.get("eval_duration", 0) or 1
    toks_s = n / (dur_ns / 1e9)
    return out.get("response", ""), n, toks_s, wall


print("== warm-up (plain, 8 tok) ==", flush=True)
gen(False, 8)

print("\n== M2e PLAIN decode ==", flush=True)
txt_p, n_p, ts_p, wall_p = gen(False)
print(f"tokens={n_p}  {ts_p:.2f} tok/s  wall={wall_p:.1f}s", flush=True)

print("\n== M3 SPECULATIVE decode (0.5B draft) ==", flush=True)
txt_s, n_s, ts_s, wall_s = gen(True)
print(f"tokens={n_s}  {ts_s:.2f} tok/s  wall={wall_s:.1f}s", flush=True)

print("\n== RESULT ==", flush=True)
print(f"plain        : {ts_p:.2f} tok/s", flush=True)
print(f"speculative  : {ts_s:.2f} tok/s", flush=True)
speedup = ts_s / ts_p if ts_p else 0
verdict = "SPECULATIVE WINS" if speedup > 1.0 else "plain wins"
print(f"speedup      : {speedup:.2f}x  ->  {verdict}", flush=True)
print(f"outputs identical: {txt_p == txt_s}", flush=True)
if txt_p != txt_s:
    print("--- plain ---\n" + txt_p, flush=True)
    print("--- spec  ---\n" + txt_s, flush=True)
else:
    print("--- output ---\n" + txt_p, flush=True)
