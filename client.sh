#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel WORKER (Linux)  -  connects to the BEAST controller
#
#  Usage:
#    ./client.sh                    CPU worker (uses ./.venv/bin/python if present)
#    ./client.sh --device cpu+gpu   use a local GPU, spill overflow to CPU
#    ./client.sh --name work        override the reported hostname
#
#  Cleanup is OFF by default; pass --clean only to purge cached models/chunks.
#  Extra client.py flags pass through ("$@"): --data-port, --ram, --name ...
#  To run detached on a worker:
#    setsid ./client.sh </dev/null >client.log 2>&1 &
# ===========================================================================
set -e
cd "$(dirname "$0")"
PY=./.venv/bin/python
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
exec "$PY" client.py "$@"   # controller host/port default from config.json (override: --controller HOST)
