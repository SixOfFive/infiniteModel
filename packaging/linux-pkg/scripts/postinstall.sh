#!/bin/sh
# InfiniteModel — post-install (deb: configure / rpm: %post).
# Idempotent and portable across dpkg and rpm. Creates the service account and
# writable state dir, registers the units, and points the admin at the one
# manual step (building the venv). It deliberately does NOT build the venv or
# start services: the venv install is hardware-specific (CPU/CUDA/ROCm) and can
# pull gigabytes of torch — wrong for a package transaction.
set -e

USER_NAME=infinitemodel
HOME_DIR=/var/lib/infinitemodel

# --- service account (system user, no login, home = state dir) --------------
if ! getent group "$USER_NAME" >/dev/null 2>&1; then
    groupadd --system "$USER_NAME" 2>/dev/null || addgroup --system "$USER_NAME" 2>/dev/null || true
fi
if ! getent passwd "$USER_NAME" >/dev/null 2>&1; then
    useradd --system --gid "$USER_NAME" --home-dir "$HOME_DIR" \
            --no-create-home --shell /usr/sbin/nologin "$USER_NAME" 2>/dev/null \
    || adduser --system --ingroup "$USER_NAME" --home "$HOME_DIR" \
            --no-create-home --disabled-login "$USER_NAME" 2>/dev/null || true
fi

# --- writable app home ------------------------------------------------------
mkdir -p "$HOME_DIR"
chown "$USER_NAME":"$USER_NAME" "$HOME_DIR"
chmod 0750 "$HOME_DIR"

# --- systemd ----------------------------------------------------------------
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload || true
fi

cat <<'EOF'

InfiniteModel installed. One manual step remains — build the runtime venv for
this machine's hardware and chosen backends (torch is not bundled):

    sudo infinitemodel-setup --role worker --cuda cu128 --extras worker,vision
    # or a CPU box:      sudo infinitemodel-setup --role both --cpu --extras controller,worker
    # or controller only: sudo infinitemodel-setup --role controller --no-torch --extras controller

Then enable a service:
    sudo systemctl enable --now infinitemodel-controller   # or infinitemodel-worker

Run `infinitemodel-setup --help` for backend options (tts, stt, t2i, music,
kimi) and ACE-Step (--with-acestep). State + models live in /var/lib/infinitemodel.
EOF
exit 0
