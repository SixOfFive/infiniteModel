#!/data/data/com.termux/files/usr/bin/bash
# ===========================================================================
#  InfiniteModel Android worker — Termux-side bootstrap (run in TERMUX, not in
#  proot). One shot: installs proot-distro + a Debian guest, deploys the worker
#  into the guest, runs the in-guest setup (venv + CPU aarch64 deps), and
#  creates a tap-to-launch shortcut (Termux:Widget) + a boot script
#  (Termux:Boot). Idempotent — safe to re-run.
#
#  Expects client.py, wire.py, config.json, requirements-android.txt, setup.sh
#  and start-client.sh to sit next to this script.
# ===========================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "=== IM TERMUX BOOTSTRAP  (files: $HERE) ==="
termux-wake-lock 2>/dev/null || true

echo "=== [1/4] Termux: proot-distro ==="
pkg install -y proot-distro

echo "=== [2/4] install Debian guest (idempotent) ==="
proot-distro install debian || echo "(debian already installed - continuing)"

echo "=== [3/4] deploy worker + in-guest setup (venv + deps; LONG, torch ~200MB) ==="
# Bind the staged files into the guest and copy them to the guest's REAL
# /root/android, then run setup there. Binding avoids guessing the rootfs path.
proot-distro login debian --bind "$HERE":/mnt/im -- bash -c '
  set -e
  mkdir -p /root/android
  cp /mnt/im/client.py /mnt/im/wire.py /mnt/im/config.json \
     /mnt/im/requirements-android.txt \
     /mnt/im/setup.sh /mnt/im/start-client.sh /root/android/
  echo "deployed to guest /root/android:"; ls -la /root/android
  bash /root/android/setup.sh
'

echo "=== [4/4] launcher shortcut + boot script ==="
mkdir -p ~/.shortcuts ~/.termux/boot
cat > ~/.shortcuts/InfiniteModel-Worker.sh <<'SH'
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
proot-distro login debian -- bash -lc 'cd /root/android && bash start-client.sh --name tablet'
SH
chmod +x ~/.shortcuts/InfiniteModel-Worker.sh
cp ~/.shortcuts/InfiniteModel-Worker.sh ~/.termux/boot/start-infinitemodel.sh
chmod +x ~/.termux/boot/start-infinitemodel.sh

echo
echo "=== IM BOOTSTRAP DONE ==="
echo "ICON     : install 'Termux:Widget', add its home-screen widget, tap 'InfiniteModel-Worker'."
echo "ON BOOT  : install 'Termux:Boot' (boot script already in ~/.termux/boot/)."
echo "RUN NOW  : bash ~/.shortcuts/InfiniteModel-Worker.sh"
