#!/bin/sh
# InfiniteModel — post-remove (deb: remove/purge / rpm: %postun).
# The venv under /opt is generated (rebuildable by infinitemodel-setup), so it
# is removed on uninstall. The app home /var/lib/infinitemodel holds STATE and
# downloaded MODELS — it is preserved on a plain remove and deleted only on a
# deb `purge`, so an accidental remove never destroys gigabytes of models.
# rpm has no purge concept; on rpm the home is always preserved (delete by hand).
set -e

# generated venv + PATH symlinks — safe to drop on any removal (not upgrade)
case "${1:-}" in
    upgrade|1) exit 0 ;;                          # upgrade — keep everything
esac
rm -rf /opt/infinitemodel/venv 2>/dev/null || true
rm -f /usr/bin/infinitemodel-controller /usr/bin/infinitemodel-worker 2>/dev/null || true

if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload 2>/dev/null || true
fi

# deb purge only: also remove state/models and the service account.
if [ "${1:-}" = "purge" ]; then
    rm -rf /var/lib/infinitemodel 2>/dev/null || true
    if getent passwd infinitemodel >/dev/null 2>&1; then
        userdel infinitemodel 2>/dev/null || deluser --system infinitemodel 2>/dev/null || true
    fi
fi
exit 0
