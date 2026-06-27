"""#22 inc 5d: end-to-end distributed AUDIO -> text test.
Sends a speech WAV to the controller's Anthropic Messages API and prints the reply.
Pure stdlib (base64/json/urllib) — no torch needed (runs under the broken-CUDA python)."""
import base64
import json
import sys
import time
import urllib.request

import os
import wire
_cfg = wire.load_config()
CTRL = f"http://{_cfg['controller_host']}:{_cfg['http_port']}/v1/messages"   # from config.json
WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_speech.wav")

question = sys.argv[1] if len(sys.argv) > 1 else "Transcribe the speech in this audio, word for word."
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 300

with open(WAV, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

body = {
    "model": "qwen2.5-omni-7b",
    "max_tokens": max_tokens,
    "messages": [
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            {"type": "text", "text": question},
        ]},
    ],
}

req = urllib.request.Request(CTRL, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:800])
    sys.exit(1)
dt = time.time() - t0

# Anthropic response: content is a list of blocks
text = ""
for blk in resp.get("content", []):
    if blk.get("type") == "text":
        text += blk.get("text", "")
print(f"=== reply ({dt:.1f}s, stop={resp.get('stop_reason')}, "
      f"usage={resp.get('usage')}) ===")
print(text if text else json.dumps(resp)[:1000])
