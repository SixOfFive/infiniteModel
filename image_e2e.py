"""#22 Omni vision: end-to-end distributed IMAGE -> text test.
Generates a test PNG (solid color or a left/right split) and sends it to the controller's
Anthropic Messages API. Pure stdlib + PIL (no torch). Usage:
    python image_e2e.py <mode> "<question>" [max_tokens]
  mode = a color name (red/blue/green/...) for a solid image, or "split" for left-red/right-blue.
"""
import base64
import io
import json
import sys
import time
import urllib.request

from PIL import Image

import wire
_cfg = wire.load_config()
CTRL = f"http://{_cfg['controller_host']}:{_cfg['http_port']}/v1/messages"   # from config.json

mode = sys.argv[1] if len(sys.argv) > 1 else "red"
question = sys.argv[2] if len(sys.argv) > 2 else "What color is this image? Answer in one word."
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 300

COLORS = {"red": (220, 30, 30), "blue": (30, 30, 220), "green": (30, 180, 60),
          "yellow": (230, 220, 40), "purple": (150, 40, 200), "orange": (240, 140, 20)}

if mode == "split":
    img = Image.new("RGB", (448, 224), COLORS["red"])
    right = Image.new("RGB", (224, 224), COLORS["blue"])
    img.paste(right, (224, 0))
else:
    img = Image.new("RGB", (336, 336), COLORS.get(mode, (220, 30, 30)))

buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

body = {
    "model": "qwen2.5-omni-7b",
    "max_tokens": max_tokens,
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
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

text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
print(f"=== [{mode}] reply ({dt:.1f}s, stop={resp.get('stop_reason')}, usage={resp.get('usage')}) ===")
print(text if text else json.dumps(resp)[:1000])
