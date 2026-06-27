#!/usr/bin/env bash
# ===========================================================================
#  PREPARE the portable worker bundle (run ONCE, on an internet-connected box).
#  Downloads standalone Python (win+linux) + the worker wheel closure into
#  install/python/ and install/wheels/. Nothing is installed into the OS.
#
#    ./prepare.sh            both platforms
#    ./prepare.sh --linux    Linux wheels only (still grabs both Pythons)
#    ./prepare.sh --force    re-download everything
# ===========================================================================
set -e
cd "$(dirname "$0")"
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "[ERROR] Python 3 with pip is required to PREPARE the bundle."
  exit 1
fi
echo "Using $PY"
exec "$PY" _fetch.py "$@"
