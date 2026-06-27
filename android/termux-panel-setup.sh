#!/data/data/com.termux/files/usr/bin/bash
# InfiniteModel tablet: live bandwidth panel layout for Termux. Idempotent.
# Run once inside Termux:   bash /data/local/tmp/termux-panel-setup.sh
# Expects traffic_panel.py staged at /data/local/tmp/traffic_panel.py.
#
# DEFAULT = bandwidth panel only (a shell on top, the live FLEET BANDWIDTH panel on the
# bottom). The fleet WORKER is NOT auto-started: on this Wi-Fi aarch64 tablet the worker's
# control connection churns reconnects (every ~2s -> re-register spam on the controller),
# so it's opt-in via the `worker` command, run in the foreground (Ctrl-C to stop).
set -e
echo "[IM] installing deps (tmux, python, procps)..."
pkg install -y tmux python procps >/dev/null 2>&1 || true

# proot-distro hard-references Termux python3.12; a pkg bump to 3.13 breaks it. Shim it.
ln -sf python3.13 "$PREFIX/bin/python3.12" 2>/dev/null || true

mkdir -p "$HOME/.im"
if [ -f /data/local/tmp/traffic_panel.py ]; then
  cp /data/local/tmp/traffic_panel.py "$HOME/.im/traffic_panel.py"
  echo "[IM] panel deployed -> ~/.im/traffic_panel.py"
fi
rm -f "$HOME/.im/panel_state.json" 2>/dev/null    # old lifetime MAX/XFER state no longer used

mkdir -p "$HOME/.termux"
grep -q 'allow-external-apps' "$HOME/.termux/termux.properties" 2>/dev/null \
  || echo 'allow-external-apps=true' >> "$HOME/.termux/termux.properties"

BRC="$HOME/.bashrc"
if [ -f "$BRC" ] && grep -q 'IM-AUTOSTART' "$BRC"; then
  sed -i '/# IM-AUTOSTART BEGIN/,/# IM-AUTOSTART END/d' "$BRC"
fi

cat >> "$BRC" <<'EOF'
# IM-AUTOSTART BEGIN -- InfiniteModel live bandwidth panel (shell on top, panel on bottom).
_im_kill() {
  tmux kill-server 2>/dev/null
  pkill -9 -f 'client.py'                 2>/dev/null
  pkill -9 -f 'start-client.sh'           2>/dev/null
  pkill -9 -f 'proot-distro login debian' 2>/dev/null
  pkill -9 -f 'installed-rootfs/debian'   2>/dev/null
}
_im_panel() {
  ln -sf python3.13 /data/data/com.termux/files/usr/bin/python3.12 2>/dev/null
  # FULL-SCREEN bandwidth panel (dedicated display). For a shell, open a new Termux session.
  tmux new-session -d -s im "while true; do /data/data/com.termux/files/usr/bin/python3 /data/data/com.termux/files/home/.im/traffic_panel.py; sleep 2; done"
  tmux set-option -t im status off 2>/dev/null    # clean full-screen (hide the status bar)
}
# `panel` = (re)build the shell+bandwidth layout cleanly.
panel() {
  if [ -n "$TMUX" ]; then echo '[IM] already in the layout -- Ctrl-b d to detach first'; return 1; fi
  _im_kill; sleep 1; _im_panel; tmux attach -t im
}
# `worker` = OPT-IN, one foreground attempt of the fleet worker. It is NOT looped, but note
# the client itself may churn reconnects on this tablet's Wi-Fi link -- Ctrl-C to stop it.
worker() {
  ln -sf python3.13 /data/data/com.termux/files/usr/bin/python3.12 2>/dev/null
  echo '[IM] one-shot fleet worker (foreground). Ctrl-C to stop. (May reconnect-churn on Wi-Fi.)'
  termux-wake-lock >/dev/null 2>&1 || true
  proot-distro login debian -- bash -lc 'cd /root/android && bash start-client.sh --name tablet'
  echo '[IM] worker stopped.'
}
# New Termux session: run my pushed hook (if any), else attach/build the panel layout. No worker.
if [ -z "$TMUX" ] && [ -z "$IM_STARTED" ]; then
  export IM_STARTED=1
  if [ -f /data/local/tmp/im_boot.sh ]; then
    bash /data/local/tmp/im_boot.sh
  elif tmux has-session -t im 2>/dev/null; then
    tmux attach -t im
  else
    _im_kill; sleep 1; _im_panel; tmux attach -t im
  fi
fi
# IM-AUTOSTART END
EOF

echo "[IM] panel layout installed in ~/.bashrc"
echo "[IM] DONE. Open Termux -> shell on top, live bandwidth on bottom. ('panel' rebuilds it; 'worker' = opt-in fleet worker.)"
