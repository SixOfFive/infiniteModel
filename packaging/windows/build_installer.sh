#!/usr/bin/env bash
# ===========================================================================
#  build_installer.sh - build the Windows NSIS installer .exe.
#
#  Runs `makensis` (Linux-native — `sudo apt install nsis` on Debian/Ubuntu,
#  `sudo dnf install mingw32-nsis` on Fedora; no Windows or Wine needed). Builds
#  the wheel first if it is missing. Output: dist/infinitemodel-<ver>-setup.exe
# ===========================================================================
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"      # packaging/windows
ROOT="$(cd "$PKG/../.." && pwd)"          # repo root
DIST="$ROOT/dist"

VER="$(grep -oE 'VERSION = "[^"]+"' "$ROOT/server.py" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VER" ] || { echo "[ERROR] could not read VERSION from server.py"; exit 1; }
WHEEL="dist/infinitemodel-$VER-py3-none-any.whl"

if [ ! -f "$ROOT/$WHEEL" ]; then
  echo "== wheel missing - building via packaging/build.sh =="
  "$ROOT/packaging/build.sh"
fi

if ! command -v makensis >/dev/null 2>&1; then
  echo "[ERROR] makensis not found. Install NSIS:"
  echo "        Debian/Ubuntu:  sudo apt install nsis"
  echo "        Fedora:         sudo dnf install mingw32-nsis"
  exit 1
fi

echo "== makensis: building installer (infinitemodel $VER) =="
# NSIS resolves relative File/LicenseData/OutFile paths against the .nsi's own
# directory, so pass everything absolute.
mkdir -p "$DIST"
makensis -V2 \
    -DAPPVERSION="$VER" \
    -DWHEEL="$ROOT/$WHEEL" \
    -DSRCDIR="$PKG" \
    -DLICENSEFILE="$ROOT/LICENSE" \
    -DOUTFILE="$DIST/infinitemodel-$VER-setup.exe" \
    "$PKG/win-installer.nsi"

echo "== done =="
ls -lh "$DIST/infinitemodel-$VER-setup.exe" | sed 's/^/   /'
