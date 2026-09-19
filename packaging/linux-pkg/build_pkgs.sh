#!/usr/bin/env bash
# ===========================================================================
#  build_pkgs.sh — build BOTH .deb and .rpm from nfpm.yaml (the canonical spec).
#
#  Requires `nfpm` (https://nfpm.goreleaser.com) — a single static binary, no
#  root, no rpmbuild. Builds the wheel first if missing. On a box without nfpm,
#  use build_deb.sh for the .deb and build the .rpm on an nfpm/rpm-capable box.
#  Output: dist/infinitemodel_<ver>_all.deb  and  dist/infinitemodel-<ver>.noarch.rpm
# ===========================================================================
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PKG/../.." && pwd)"
DIST="$ROOT/dist"
export MAINTAINER="${MAINTAINER:-sixoffive <sixoffive@users.noreply.github.com>}"
export VERSION="$(grep -oE 'VERSION = "[^"]+"' "$ROOT/server.py" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VERSION" ] || { echo "[ERROR] could not read VERSION from server.py"; exit 1; }

if ! command -v nfpm >/dev/null 2>&1; then
  cat >&2 <<EOF
[ERROR] nfpm not found. Install it (single binary, no root), e.g.:
    go install github.com/goreleaser/nfpm/v2/cmd/nfpm@latest
    # or download the release binary from https://github.com/goreleaser/nfpm/releases
Then re-run. For the .deb only, build_deb.sh needs no extra tooling.
EOF
  exit 1
fi

if [ ! -f "$DIST/infinitemodel-$VERSION-py3-none-any.whl" ]; then
  echo "== wheel missing — building via packaging/build.sh =="
  "$ROOT/packaging/build.sh"
fi

mkdir -p "$DIST"
echo "== nfpm: building .deb and .rpm (infinitemodel $VERSION) =="
( cd "$PKG" && nfpm pkg -f nfpm.yaml -p deb -t "$DIST/" && nfpm pkg -f nfpm.yaml -p rpm -t "$DIST/" )

echo "== done =="
ls -lh "$DIST"/infinitemodel*_all.deb "$DIST"/infinitemodel-*.noarch.rpm 2>/dev/null | sed 's/^/   /'
