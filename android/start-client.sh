#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel Android worker — launcher.
#  Run INSIDE the proot guest, from this android/ folder (after setup.sh):
#      bash start-client.sh --name tablet
#
#  Defaults: controller 192.168.15.38:50100, --device cpu (no CUDA on Android),
#  --ram "android-tablet" (dmidecode/root aren't available in proot — harmless).
#  Extra flags pass straight through, e.g.:
#      bash start-client.sh --name tablet --controller 192.168.1.50
#      bash start-client.sh --name tablet --os-reserve-gb 3
#
#  Tip: run under tmux so it survives the terminal closing:
#      tmux new -s im   ->   bash start-client.sh --name tablet   ->   Ctrl-b d
#  And in TERMUX (host) keep the CPU awake:   termux-wake-lock
# ===========================================================================
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  echo "[!] worker env not built yet — run:  bash setup.sh"
  exit 1
fi
PY="$HERE/.venv/bin/python"
code=0
while true; do
  set +e
  "$PY" client.py --controller 192.168.15.38 --control-port 50100 \
       --device cpu --ram "android-tablet" "$@"
  code=$?
  set -e
  if [ "$code" = "42" ]; then
    echo "[update] new code pulled - relaunching ..."
    continue
  fi
  break
done
exit "$code"
