"""Fire a sustained generation and poll /status during it to see live per-node
net rates + API metrics. Settles whether the dashboard net display is broken or
just genuinely idle when observed."""
import json, threading, time, urllib.request

BASE = "http://127.0.0.1:11434"


def gen():
    body = {"model": "qwen2.5-7b", "prompt": "Write a long detailed paragraph about the history of computing.",
            "stream": False, "options": {"temperature": 0, "num_predict": 60}}
    req = urllib.request.Request(BASE + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=600).read()
    except Exception as e:
        print("gen err:", e, flush=True)


t = threading.Thread(target=gen, daemon=True)
t.start()
print("gen started; polling /status every 2s ...", flush=True)
for i in range(30):
    if not t.is_alive():
        print("[gen finished]", flush=True)
        break
    try:
        o = json.loads(urllib.request.urlopen(BASE + "/status", timeout=5).read())
        m = o.get("metrics", {})
        rows = []
        for n in o.get("nodes", []):
            rows.append(f"{n.get('hostname','?')[:9]:>9} in={n.get('net_in_bps',0)/1024:6.1f}KB/s out={n.get('net_out_bps',0)/1024:7.1f}KB/s")
        print(f"t+{i*2:>3}s tok/s={m.get('tokens_per_s',0):.2f} "
              f"api_in={m.get('api_in_bps',0)/1024:.1f}KB/s api_out={m.get('api_out_bps',0)/1024:.1f}KB/s",
              flush=True)
        for r in rows:
            print("        " + r, flush=True)
    except Exception as e:
        print(f"t+{i*2}s status err:", e, flush=True)
    time.sleep(2)
t.join(timeout=120)
print("done", flush=True)
