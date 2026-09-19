#!/bin/sh
# InfiniteModel — pre-remove (deb: remove/upgrade / rpm: %preun).
# Stop and disable the services so nothing is left running against files that
# are about to disappear. Guarded so an upgrade (which re-runs postinstall)
# doesn't leave a disabled service — we only stop here; enable state is the
# admin's to keep. On dpkg, $1 is "upgrade" during an upgrade; on rpm, $1 is
# the remaining-version count (1 = upgrade, 0 = final removal).
set -e

if command -v systemctl >/dev/null 2>&1; then
    for svc in infinitemodel-controller infinitemodel-worker; do
        systemctl stop "$svc" 2>/dev/null || true
        # Only disable on a true removal, not an upgrade.
        case "${1:-}" in
            upgrade|1) : ;;                       # upgrade — keep enable state
            *) systemctl disable "$svc" 2>/dev/null || true ;;
        esac
    done
fi
exit 0
