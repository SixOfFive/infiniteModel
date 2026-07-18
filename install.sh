#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel — portable WORKER installer (Linux)
#
#  Builds a self-contained worker environment under ./install/ from the
#  pre-downloaded bundle. NOTHING is installed into the OS (no apt, no sudo).
#  Re-runnable. Designed to run straight off a USB stick / copied folder,
#  offline.
#
#  Prereq: the bundle was populated once with  install/prepare.sh  (online).
# ===========================================================================
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
INS="$ROOT/install"
VENV="$INS/.venv-linux"
WHEELS="$INS/wheels/linux"
REQ="$INS/requirements-client.txt"

echo "==========================================================================="
echo " InfiniteModel portable worker installer (Linux)"
echo " Repo: $ROOT"
echo "==========================================================================="

# --- 1) locate a Python: bundled standalone -> system python3 ----------------
PY=""
RT="$INS/runtime/linux/python"
if [ -x "$RT/bin/python3" ]; then
  PY="$RT/bin/python3"
elif [ -f "$INS/python/cpython-3.13-linux.tar.gz" ]; then
  echo "[1/5] extracting bundled Python ..."
  mkdir -p "$INS/runtime/linux"
  tar -xf "$INS/python/cpython-3.13-linux.tar.gz" -C "$INS/runtime/linux"
  [ -x "$RT/bin/python3" ] && PY="$RT/bin/python3"
fi
if [ -z "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ]; then
  echo "[ERROR] No Python found and no bundled Python in install/python/."
  echo "        Run install/prepare.sh on an internet-connected box first,"
  echo "        or install python3 (+python3-venv) on this machine."
  exit 1
fi
echo "[1/5] Python: $PY"

# --- 2) create the venv (offline) --------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[2/5] creating venv at install/.venv-linux ..."
  if ! "$PY" -m venv "$VENV"; then
    echo "[ERROR] venv creation failed."
    echo "        With a system python3 you may need: sudo apt-get install python3-venv"
    echo "        (or use the bundled Python: run install/prepare.sh, then re-run this)."
    exit 1
  fi
else
  echo "[2/5] venv already present - reusing"
fi
VPY="$VENV/bin/python"

# --- 3) install worker deps: offline from bundle, else online fallback --------
echo "[3/5] installing worker deps (offline from install/wheels/linux) ..."
if ! "$VPY" -m pip install --no-index --find-links "$WHEELS" -r "$REQ"; then
  echo
  echo "[warn] offline install incomplete - falling back to ONLINE install ..."
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
  if ! "$VPY" -m pip install transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 numpy==2.4.6 psutil==7.2.2; then
    echo "[ERROR] dependency install failed (offline AND online)."
    exit 1
  fi
fi

# --- 4) verify ---------------------------------------------------------------
echo "[4/5] verifying imports ..."
if ! "$VPY" -c "import torch,transformers,safetensors,huggingface_hub,numpy,psutil;print('  torch',torch.__version__,'| transformers',transformers.__version__,'| numpy',numpy.__version__)"; then
  echo "[ERROR] dependency verification failed."
  exit 1
fi

# --- 5) done -----------------------------------------------------------------
chmod +x "$INS/start-client.sh" 2>/dev/null || true
echo "[5/5] ready."
echo
echo "==========================================================================="
echo " READY — the worker environment is built (nothing installed into the OS)."
echo
echo " START THE WORKER:"
echo "     ./install/start-client.sh"
echo
echo " Useful variants:"
echo "     ./install/start-client.sh --device cpu          (force CPU)"
echo "     ./install/start-client.sh --controller <ip>     (other controller)"
echo "     ./install/start-client.sh --name <label>        (override hostname)"
echo
echo " Default controller: 192.168.15.103:50100  (edit start-client.sh to change)"
echo "==========================================================================="
