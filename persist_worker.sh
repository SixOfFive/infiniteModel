#!/usr/bin/env bash
# Make an ALREADY-DEPLOYED InfiniteModel worker reboot-persistent via systemd.
# Does NOT touch the repo or venv (legacy nodes are file-copy, not git clones;
# theocomp has CUDA torch we must not clobber). Just installs + starts the unit.
# Usage: persist_worker.sh "<client.py extra args>"
#   e.g. persist_worker.sh "--name theocomp --device cpu+gpu --attn sdpa"
set -euo pipefail
RUN_USER="${RUN_USER:-$(id -un)}"        # service account (no baked username)
REPO="${IM_REPO:-$HOME/infinitemodel}"
EXTRA_ARGS="${1:---device cpu --attn sdpa}"   # controller host/port now come from config.json
PY="$REPO/.venv/bin/python"

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY"; exit 1; }
[ -f "$REPO/client.py" ] || { echo "FATAL: no client.py at $REPO"; exit 1; }
echo "== HOST $(hostname) | torch $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null) =="

# Stop any detached/old worker so it doesn't double-bind the data port
pkill -f "client.py" 2>/dev/null && echo "(stopped existing detached worker)" || echo "(no existing worker running)"
sleep 1

# Memory safety (m4c25+): cap the worker's RAM and FORBID swap, so an over-assignment is a CLEAN
# OOM-kill (node drops -> controller replans/auto-recovers) instead of swap-thrashing/freezing the
# host. Sized total-15% (min 4 GB headroom for OS/desktop/VMs). Existing nodes also have this via a
# `systemctl set-property` drop-in (applied live, no restart); baking it here covers fresh provisions.
TOT_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
HR_GB=$(( TOT_GB*15/100 )); [ "$HR_GB" -lt 4 ] && HR_GB=4
MEM_MAX_GB=$(( TOT_GB - HR_GB )); [ "$MEM_MAX_GB" -lt 2 ] && MEM_MAX_GB=2
echo "== memory cap: total ${TOT_GB}G -> MemoryMax=${MEM_MAX_GB}G + MemorySwapMax=0 =="
echo "== installing systemd unit (args: $EXTRA_ARGS) =="
sudo tee /etc/systemd/system/infinitemodel-worker.service >/dev/null <<EOF
[Unit]
Description=InfiniteModel worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO
ExecStart=$PY $REPO/client.py $EXTRA_ARGS
Restart=always
RestartSec=5
OOMScoreAdjust=800
Nice=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable -q infinitemodel-worker.service
sudo systemctl restart infinitemodel-worker.service
sleep 4
echo "== ACTIVE: $(systemctl is-active infinitemodel-worker.service) | ENABLED: $(systemctl is-enabled infinitemodel-worker.service) =="
journalctl -u infinitemodel-worker.service -n 6 --no-pager | tail -6
echo "== DONE $(hostname) =="
