#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel Android worker — in-guest setup.
#  Run INSIDE the proot Debian/Ubuntu guest, from this android/ folder:
#      bash setup.sh
#  Builds a venv and installs the worker deps (CPU aarch64). You are root in
#  the proot guest, so no sudo is needed and nothing touches the host Android.
#
#  IMPORTANT: torch is installed from the PyTorch CPU index. The DEFAULT PyPI
#  aarch64 torch wheel is the CUDA/SBSA build (it drags in multi-GB nvidia-*
#  packages, useless on a tablet) — the CPU index serves the CPU-only wheel.
# ===========================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "== apt: python3 + venv + pip =="
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip

echo "== venv: $HERE/.venv (fresh) =="
rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q

echo "== pip: torch — CPU aarch64 wheel from the PyTorch CPU index =="
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

echo "== pip: transformers stack (PyPI) =="
pip install transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 numpy==2.4.6 psutil==7.2.2

echo "== verify =="
python - <<'PY'
import torch, transformers, safetensors, numpy, psutil
print("  OK  torch", torch.__version__, "| transformers", transformers.__version__,
      "| numpy", numpy.__version__, "| cuda", torch.cuda.is_available())
PY

echo
echo "READY."
echo "  1) keep the device awake — run in TERMUX (not here):   termux-wake-lock"
echo "  2) start the worker (under tmux so it survives):        bash start-client.sh --name tablet"
