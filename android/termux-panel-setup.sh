#!/data/data/com.termux/files/usr/bin/bash
# InfiniteModel tablet: live bandwidth panel (FOREGROUND) + fleet worker (BACKGROUND). Idempotent.
# Run inside Termux:   bash /data/local/tmp/termux-panel-setup.sh
# Expects traffic_panel.py staged at /data/local/tmp/traffic_panel.py.
#
# On each Termux launch this starts the fleet WORKER detached in the background (tmux 'wrk') and
# shows the live FLEET BANDWIDTH panel in the foreground (tmux 'im'). Note: on this Wi-Fi aarch64
# tablet the worker can churn reconnects / thrash RAM — the panel's TABLET line shows cpu+mem so
# you can watch the load. Commands: `panel` rebuilds the display; `startworker` / `stopworker`
# control the background worker; `worker` attaches to its log.
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
# IM-AUTOSTART BEGIN -- InfiniteModel: bandwidth panel (foreground) + fleet worker (background).
_im_shim() { ln -sf python3.13 /data/data/com.termux/files/usr/bin/python3.12 2>/dev/null; }
# Fleet WORKER, detached in the background (tmux 'wrk'). Idempotent (won't double-start).
_im_worker() {
  _im_shim
  termux-wake-lock >/dev/null 2>&1 || true
  tmux has-session -t wrk 2>/dev/null || \
    tmux new-session -d -s wrk "proot-distro login debian -- bash -lc 'cd /root/android && bash start-client.sh --name tablet'"
}
# Bandwidth PANEL, detached (tmux 'im'), full-screen. Idempotent.
_im_panel() {
  _im_shim
  if ! tmux has-session -t im 2>/dev/null; then
    tmux new-session -d -s im "while true; do /data/data/com.termux/files/usr/bin/python3 /data/data/com.termux/files/home/.im/traffic_panel.py; sleep 2; done"
    tmux set-option -t im status off 2>/dev/null
  fi
}
# `panel` = rebuild + show the bandwidth display (leaves the background worker running).
panel()       { if [ -n "$TMUX" ]; then echo '[IM] detach first: Ctrl-b d'; return 1; fi; tmux kill-session -t im 2>/dev/null; sleep 1; _im_panel; tmux attach -t im; }
# `worker` = attach to the background worker's log (Ctrl-b d to detach).
worker()      { if tmux has-session -t wrk 2>/dev/null; then tmux attach -t wrk; else echo '[IM] no background worker -- run: startworker'; fi; }
startworker() { _im_worker; echo '[IM] background worker running (tmux wrk).'; }
stopworker()  { tmux kill-session -t wrk 2>/dev/null; pkill -9 -f client.py 2>/dev/null; pkill -9 -f start-client.sh 2>/dev/null; pkill -9 -f proot 2>/dev/null; echo '[IM] background worker stopped.'; }
# New Termux session: start the worker in the BACKGROUND, then show the panel in the FOREGROUND.
if [ -z "$TMUX" ] && [ -z "$IM_STARTED" ]; then
  export IM_STARTED=1
  if [ -f /data/local/tmp/im_boot.sh ]; then
    bash /data/local/tmp/im_boot.sh
  else
    _im_worker
    _im_panel
    tmux attach -t im
  fi
fi
# IM-AUTOSTART END
EOF

# Termux:Boot -- start the worker + panel ON DEVICE BOOT (headless). Requires the Termux:Boot
# add-on installed. The worker then serves from boot without opening the app; the panel runs in
# the background and becomes visible the moment you open Termux (Android can't foreground an app's
# UI on boot). Replaces the old start-infinitemodel.sh boot script.
mkdir -p "$HOME/.termux/boot"
rm -f "$HOME/.termux/boot/start-infinitemodel.sh" "$HOME/.termux/boot/start-infinitemodel.sh.disabled"
cat > "$HOME/.termux/boot/im-autostart.sh" <<'BOOT'
#!/data/data/com.termux/files/usr/bin/bash
ln -sf python3.13 /data/data/com.termux/files/usr/bin/python3.12 2>/dev/null
termux-wake-lock >/dev/null 2>&1 || true
sleep 25   # let Wi-Fi associate before the worker dials the controller
tmux has-session -t wrk 2>/dev/null || \
  tmux new-session -d -s wrk "proot-distro login debian -- bash -lc 'cd /root/android && bash start-client.sh --name tablet'"
if ! tmux has-session -t im 2>/dev/null; then
  tmux new-session -d -s im "while true; do /data/data/com.termux/files/usr/bin/python3 /data/data/com.termux/files/home/.im/traffic_panel.py; sleep 2; done"
  tmux set-option -t im status off 2>/dev/null
fi
BOOT
chmod +x "$HOME/.termux/boot/im-autostart.sh"

echo "[IM] installed in ~/.bashrc: worker (background) + panel (foreground)."
echo "[IM] installed Termux:Boot script -> worker + panel start on device boot (panel shows on app open)."
echo "[IM] DONE. Open Termux -> the worker starts in the background, the live panel in the foreground."
echo "[IM]   panel=rebuild display · worker=attach worker log · startworker/stopworker=control"
