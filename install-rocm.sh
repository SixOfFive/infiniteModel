#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel — AMD GPU (ROCm) worker/standalone installer  (Linux)
#
#  Builds a venv with a PyTorch+ROCm runtime matched to your AMD GPU arch, plus
#  the app deps, so a ROCm node runs 1:1 with a CUDA node. See docs/ROCM.md.
#
#  Usage:
#    ./install-rocm.sh                 # default arch gfx1151 (Strix Halo / Ryzen AI Max)
#    ./install-rocm.sh gfx110X-dgpu    # RX 7000 / W7000 (RDNA3 dGPU)
#    ./install-rocm.sh <arch>          # any index under rocm.nightlies.amd.com/v2/
#    VENV=~/imenv ./install-rocm.sh    # override venv location (default: ./.venv)
#
#  Nothing is installed system-wide except (optionally) python3-venv/git via apt and
#  your render/video group membership — the ROCm runtime ships inside the venv as pip
#  packages (rocm-sdk-*). The amdgpu kernel driver is in-tree on modern kernels.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-gfx1151}"
VENV="${VENV:-$(pwd)/.venv}"
INDEX="https://rocm.nightlies.amd.com/v2/${ARCH}/"

echo "==========================================================================="
echo " InfiniteModel ROCm installer  —  arch=${ARCH}"
echo " venv=${VENV}"
echo " torch index=${INDEX}"
echo "==========================================================================="

# --- 0) GPU device access (best-effort; needs sudo, reconnect to apply) -------
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "[0/4] adding $USER to render+video groups (sudo) — RECONNECT after this run"
  sudo usermod -aG render,video "$USER" || \
    echo "[warn] could not add groups; do it manually: sudo usermod -aG render,video $USER"
fi

# --- 1) python + venv + Triton build tools (C compiler + headers) -----------
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then echo "[ERROR] no python3 found"; exit 1; fi
# Triton JIT-compiles its launcher stubs at runtime, so the ROCm int4 kernel needs a host
# C compiler + Python headers. Best-effort (skipped if you lack sudo).
if ! "$PY" -m venv --help >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
  echo "[1/4] installing build tools (python3-venv, python3-dev, gcc, git) via sudo"
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev gcc git || true
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip -q

# --- 2) torch + matched ROCm runtime (arch-specific TheRock wheels) ----------
echo "[2/4] installing torch + ROCm runtime for ${ARCH} (this is a large download) ..."
"$VPY" -m pip install --pre torch --index-url "$INDEX"

# --- 3) app deps (same pins as the CUDA fleet) -------------------------------
echo "[3/4] installing app deps ..."
"$VPY" -m pip install transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
                      numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn

# --- 4) verify ---------------------------------------------------------------
echo "[4/4] verifying GPU is usable by torch ..."
"$VPY" - <<'PYEOF'
import torch
print("  torch", torch.__version__, "| hip", torch.version.hip)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False — check render group / driver"
print("  device:", torch.cuda.get_device_name(0),
      "| arch:", getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?"))
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
(a @ a).sum().item(); torch.cuda.synchronize()
print("  matmul on GPU: OK")
PYEOF

echo
echo "==========================================================================="
echo " READY. Launch (standalone controller + GPU worker on this box):"
echo
echo "   IM_ALLOW_NO_MODELS=1 setsid $VPY server.py >~/controller.log 2>&1 </dev/null &"
echo "   setsid $VPY client.py --controller 127.0.0.1 --device cpu+gpu >~/worker.log 2>&1 </dev/null &"
echo
echo " Or join an existing fleet (worker only):"
echo "   $VPY client.py --controller <controller-ip> --device cpu+gpu"
echo " See docs/ROCM.md for details."
echo "==========================================================================="
