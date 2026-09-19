#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel — installer (bootstrap-on-install)
#
#  Builds a Python venv and installs the InfiniteModel wheel into it, choosing
#  the torch build for THIS machine (torch is not in the wheel — it is hardware-
#  and version-specific). Optional model backends are opt-in via --extras.
#
#  The code itself is materialised into a writable app home on first launch
#  (default ~/.local/share/infinitemodel), not into the venv — see the launcher.
#
#  Examples:
#     ./bootstrap.sh --cpu  --extras controller,worker
#     ./bootstrap.sh --cuda cu128 --extras worker,vision,stt,t2i
#     ./bootstrap.sh --cuda cu128 --extras worker,tts        # Kokoro TTS
#     ./bootstrap.sh --rocm gfx1151 --extras worker          # see docs/ROCM.md
#     ./bootstrap.sh --cpu  --extras controller  --no-torch  # controller-only
# ===========================================================================
set -euo pipefail

FLAVOR=""              # cpu | cuNNN | rocm
ROCM_ARCH=""
EXTRAS=""
WITH_ACESTEP=0
NO_TORCH=0
PREFIX="${INFINITEMODEL_HOME:-$HOME/.local/share/infinitemodel}"
BINDIR="$HOME/.local/bin"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)          FLAVOR="cpu" ;;
    --cuda)         FLAVOR="$2"; shift ;;      # e.g. cu128, cu126
    --rocm)         FLAVOR="rocm"; ROCM_ARCH="${2:-gfx1151}"; shift ;;
    --extras)       EXTRAS="$2"; shift ;;
    --with-acestep) WITH_ACESTEP=1 ;;
    --no-torch)     NO_TORCH=1 ;;
    --prefix)       PREFIX="$2"; shift ;;
    --bindir)       BINDIR="$2"; shift ;;
    -h|--help)      usage 0 ;;
    *) echo "[ERROR] unknown option: $1"; usage 1 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "$0")" && pwd)"
WHEEL="$(ls "$HERE"/infinitemodel-*.whl 2>/dev/null | head -1)"
[ -f "$WHEEL" ] || { echo "[ERROR] no infinitemodel wheel next to bootstrap.sh"; exit 1; }
VENV="$PREFIX/venv"

echo "==========================================================================="
echo " InfiniteModel installer"
echo "   wheel   : $(basename "$WHEEL")"
echo "   venv    : $VENV"
echo "   torch   : ${NO_TORCH:+(skipped)}${FLAVOR:-<none — pass --cpu/--cuda/--rocm>}"
echo "   extras  : ${EXTRAS:-<none>}"
echo "==========================================================================="

# --- python + venv ----------------------------------------------------------
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "[ERROR] no python3 on PATH"; exit 1; }
if [ ! -x "$VENV/bin/python" ]; then
  echo "[1/5] creating venv ..."
  "$PY" -m venv "$VENV" || { echo "[ERROR] venv creation failed (need python3-venv?)"; exit 1; }
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip >/dev/null

# --- torch (flavor-specific, first) -----------------------------------------
if [ "$NO_TORCH" -eq 0 ]; then
  echo "[2/5] installing torch ($FLAVOR) ..."
  case "$FLAVOR" in
    cpu)   "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu ;;
    cu*)   "$VPY" -m pip install torch --index-url "https://download.pytorch.org/whl/$FLAVOR" ;;
    rocm)  echo "    ROCm ($ROCM_ARCH): torch must come from AMD's arch-matched TheRock"
           echo "    wheels, not the generic index. See docs/ROCM.md — this installer"
           echo "    does not guess the ROCm wheel URL. Install torch into $VENV first,"
           echo "    then re-run with --no-torch."
           exit 1 ;;
    "")    echo "[ERROR] pick a torch flavor: --cpu | --cuda cuNNN | --rocm gfxNNNN"
           echo "        (or --no-torch for a controller-only box)"; exit 1 ;;
    *)     echo "[ERROR] unrecognised torch flavor: $FLAVOR"; exit 1 ;;
  esac
else
  echo "[2/5] torch: skipped (--no-torch)"
fi

# --- the wheel + chosen extras ----------------------------------------------
echo "[3/5] installing infinitemodel${EXTRAS:+[$EXTRAS]} ..."
if [ -n "$EXTRAS" ]; then
  "$VPY" -m pip install "${WHEEL}[${EXTRAS}]"
else
  "$VPY" -m pip install "$WHEEL"
fi

# --- special backends the extras cannot express cleanly ---------------------
case ",$EXTRAS," in
  *,tts,*)
    echo "[4/5] Kokoro TTS: installing kokoro+misaki with --no-deps ..."
    "$VPY" -m pip install --no-deps kokoro misaki
    echo "    NOTE: TTS also needs the system 'espeak-ng' library:"
    echo "          Debian/Ubuntu:  sudo apt-get install espeak-ng"
    echo "          Fedora/RHEL:    sudo dnf install espeak-ng" ;;
  *) echo "[4/5] no special-backend post-steps" ;;
esac

if [ "$WITH_ACESTEP" -eq 1 ]; then
  echo "    ACE-Step (t2a) is NOT auto-installed: it needs a --no-deps source"
  echo "    install under a constraints file (a plain 'pip install acestep' breaks"
  echo "    every LLM on the node), torchaudio (no ROCm build), and a bf16-capable"
  echo "    Ampere+ GPU. Follow docs/T2A.md exactly, installing into: $VENV"
fi

# --- launchers on PATH ------------------------------------------------------
echo "[5/5] linking launchers into $BINDIR ..."
mkdir -p "$BINDIR"
ln -sf "$VENV/bin/infinitemodel-controller" "$BINDIR/infinitemodel-controller"
ln -sf "$VENV/bin/infinitemodel-worker"     "$BINDIR/infinitemodel-worker"

echo
echo "==========================================================================="
echo " READY."
echo "   Controller:  infinitemodel-controller      (dashboard on :21434)"
echo "   Worker:      infinitemodel-worker           (auto-discovers controller)"
echo "   App home:    $PREFIX   (code + state; self-updates land here)"
case ":$PATH:" in *":$BINDIR:"*) ;; *) echo "   NOTE: add $BINDIR to your PATH.";; esac
echo "==========================================================================="
