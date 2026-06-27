#!/usr/bin/env bash
# InfiniteModel worker provisioner. Idempotent: safe to re-run. Run as the worker's service user
# (with passwordless sudo). Config via env, with safe internal-LAN defaults:
#   GITLAB_TOKEN  (REQUIRED) — LAN GitLab read token; NOT stored in the repo (#public-release)
#   GITLAB_HOST / GITLAB_PROJECT / IM_REPO / RUN_USER — optional overrides
# e.g.:  GITLAB_TOKEN=<lan-gitlab-read-token> ./provision_worker.sh
set -euo pipefail
RUN_USER="${RUN_USER:-$(id -un)}"
REPO="${IM_REPO:-$HOME/infinitemodel}"
GITLAB_HOST="${GITLAB_HOST:-192.168.15.23}"
GITLAB_PROJECT="${GITLAB_PROJECT:-sixoffive/infinitemodel}"
: "${GITLAB_TOKEN:?set GITLAB_TOKEN env var to the LAN GitLab read token (not stored in the repo)}"
GIT_URL="http://oauth2:${GITLAB_TOKEN}@${GITLAB_HOST}/${GITLAB_PROJECT}.git"
PY="$REPO/.venv/bin/python"

echo "== HOST $(hostname) =="

# 1) OS deps (only if missing)
need_apt=0
command -v git  >/dev/null 2>&1 || need_apt=1
command -v pip3 >/dev/null 2>&1 || need_apt=1
python3 -c 'import ensurepip, venv' >/dev/null 2>&1 || need_apt=1
if [ "$need_apt" = 1 ]; then
  echo "== apt: installing git python3-venv python3-pip =="
  sudo apt-get update -qq || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3-venv python3-pip
fi

# 2) repo (clone or fast-forward)
if [ -d "$REPO/.git" ]; then
  echo "== git: updating $REPO =="
  git -C "$REPO" fetch -q origin
  git -C "$REPO" reset -q --hard origin/main
else
  echo "== git: cloning -> $REPO =="
  git clone -q "$GIT_URL" "$REPO"
fi

# 3) venv + deps pinned to the proven fleet versions
if [ ! -x "$PY" ]; then
  echo "== venv: creating =="
  python3 -m venv "$REPO/.venv"
fi
"$PY" -m pip install -q --upgrade pip
echo "== pip: torch 2.12.0 (CPU wheel) =="
"$PY" -m pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
echo "== pip: transformers/safetensors/hub/numpy/psutil =="
"$PY" -m pip install -q "transformers==5.12.1" "safetensors==0.8.0" "huggingface_hub==1.19.0" "numpy==2.4.6" "psutil==7.2.2"
echo "== pip: einops (required by some models' trust_remote_code, e.g. nomic-embed-text) =="
"$PY" -m pip install -q einops

# 4) systemd unit: persist across reboot, auto-restart, and make this worker the
#    preferred OOM victim so a memory crunch never takes a production VM/CT.
echo "== systemd: installing unit =="
sudo tee /etc/systemd/system/infinitemodel-worker.service >/dev/null <<EOF
[Unit]
Description=InfiniteModel worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO
ExecStart=$PY $REPO/client.py --device cpu --attn sdpa
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
echo "== ACTIVE: $(systemctl is-active infinitemodel-worker.service) =="
echo "== recent log =="
journalctl -u infinitemodel-worker.service -n 12 --no-pager | tail -12
echo "== versions =="
"$PY" -c 'import torch,transformers,sys; print("py",sys.version.split()[0],"torch",torch.__version__,"tf",transformers.__version__)'
echo "== DONE $(hostname) =="
