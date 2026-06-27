#!/usr/bin/env bash
# ===========================================================================
#  Start the InfiniteModel WORKER using the offline venv built by install.sh.
#  Self-update aware: client.py exits 42 when it pulls new code -> relaunch.
#
#    ./start-client.sh                       CPU+GPU (auto-falls-back to CPU)
#    ./start-client.sh --device cpu          force CPU (silences GPU notice)
#    ./start-client.sh --controller 10.0.0.5 point at a different controller
#    ./start-client.sh --name mybox          override reported hostname
#
#  Detached:  setsid ./start-client.sh </dev/null >worker.log 2>&1 &
# ===========================================================================
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv-linux/bin/python"
if [ ! -x "$PY" ]; then
  echo "[!] Worker env not built yet. Run ./install.sh in the parent folder first."
  exit 1
fi
cd "$HERE/.."
code=0
while true; do
  set +e
  "$PY" client.py "$@"   # controller host/port default from config.json (override: --controller HOST)
  code=$?
  set -e
  if [ "$code" = "42" ]; then
    echo "[update] new code pulled - relaunching ..."
    continue
  fi
  break
done
exit "$code"
