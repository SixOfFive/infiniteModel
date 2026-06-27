#!/usr/bin/env bash
# Persistent systemd unit for the Steam Deck worker (run as root via sudo -S).
# Replaces the transient `systemd-run infinitemodel-deck` (lost on reboot).
set -euo pipefail
RUN_USER="${RUN_USER:-deck}"             # SteamOS service account (run as root via sudo)
REPO="${IM_REPO:-/home/$RUN_USER/infinitemodel}"
PY="$REPO/.venv/bin/python"

# stop the transient unit + any stray detached worker
systemctl stop infinitemodel-deck 2>/dev/null || true
pkill -f "client.py" 2>/dev/null || true
sleep 1

cat > /etc/systemd/system/infinitemodel-worker.service <<EOF
[Unit]
Description=InfiniteModel worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Environment=HOME=/home/$RUN_USER
WorkingDirectory=$REPO
ExecStart=$PY $REPO/client.py --name steamdeck --ram 4x-LPDDR5-5500 --no-clean
Restart=always
RestartSec=5
OOMScoreAdjust=800
Nice=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable infinitemodel-worker.service
systemctl restart infinitemodel-worker.service
sleep 4
echo "ACTIVE: $(systemctl is-active infinitemodel-worker.service) | ENABLED: $(systemctl is-enabled infinitemodel-worker.service)"
journalctl -u infinitemodel-worker.service -n 6 --no-pager | tail -6
echo DONE-steamdeck
