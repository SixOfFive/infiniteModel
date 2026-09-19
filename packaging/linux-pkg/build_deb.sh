#!/usr/bin/env bash
# ===========================================================================
#  build_deb.sh — build the .deb natively with dpkg-deb (no nfpm/root needed).
#
#  Mirrors nfpm.yaml (which is the canonical spec and also emits the .rpm). Use
#  this on a box that has dpkg-deb but not nfpm. Builds the wheel first if it is
#  missing. Output: dist/infinitemodel_<ver>_all.deb
# ===========================================================================
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"          # packaging/linux-pkg
ROOT="$(cd "$PKG/../.." && pwd)"              # repo root
DIST="$ROOT/dist"
MAINTAINER="${MAINTAINER:-sixoffive <sixoffive@users.noreply.github.com>}"

VER="$(grep -oE 'VERSION = "[^"]+"' "$ROOT/server.py" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VER" ] || { echo "[ERROR] could not read VERSION from server.py"; exit 1; }
WHEEL="$DIST/infinitemodel-$VER-py3-none-any.whl"

if [ ! -f "$WHEEL" ]; then
  echo "== wheel missing — building it via packaging/build.sh =="
  "$ROOT/packaging/build.sh"
fi
[ -f "$WHEEL" ] || { echo "[ERROR] wheel not found: $WHEEL"; exit 1; }

echo "== staging deb tree (infinitemodel $VER, arch all) =="
# Stage on LOCAL disk, not in dist/: the repo often lives on a CIFS/SMB mount
# that cannot create the /usr/bin symlink (Input/output error). Only the final
# .deb (a plain ar archive) is written back to dist/.
STAGE_BASE="$(mktemp -d "${TMPDIR:-/tmp}/infinitemodel-deb.XXXXXX")"
trap 'rm -rf "$STAGE_BASE"' EXIT
FR="$STAGE_BASE/infinitemodel_${VER}_all"
mkdir -p "$FR"

install -D -m 0644 "$WHEEL"                              "$FR/opt/infinitemodel/wheel/$(basename "$WHEEL")"
install -D -m 0755 "$PKG/infinitemodel-setup"           "$FR/opt/infinitemodel/bin/infinitemodel-setup"
install -D -m 0644 "$PKG/systemd/infinitemodel-controller.service" "$FR/usr/lib/systemd/system/infinitemodel-controller.service"
install -D -m 0644 "$PKG/systemd/infinitemodel-worker.service"     "$FR/usr/lib/systemd/system/infinitemodel-worker.service"
install -D -m 0644 "$ROOT/LICENSE"                      "$FR/usr/share/doc/infinitemodel/LICENSE"
install -D -m 0644 "$PKG/README.md"                     "$FR/usr/share/doc/infinitemodel/README.md"
# PATH symlink for the setup CLI
mkdir -p "$FR/usr/bin"
ln -sf /opt/infinitemodel/bin/infinitemodel-setup       "$FR/usr/bin/infinitemodel-setup"

# --- control + maintainer scripts -------------------------------------------
mkdir -p "$FR/DEBIAN"
# Installed-Size in KiB (dpkg convention)
ISIZE="$(du -k -s "$FR" | cut -f1)"
cat > "$FR/DEBIAN/control" <<EOF
Package: infinitemodel
Version: $VER
Architecture: all
Maintainer: $MAINTAINER
Installed-Size: $ISIZE
Depends: python3, python3-venv
Recommends: espeak-ng, libsndfile1
Section: science
Priority: optional
Homepage: https://github.com/SixOfFive/infiniteModel
Description: Distributed LLM and multimodal inference across mixed CPU/GPU nodes
 Pipeline-parallel serving of large language models plus vision, speech-to-text,
 text-to-speech, text-to-image and music backends across a fleet of mixed
 hardware.
 .
 This package installs the code and systemd units; run infinitemodel-setup to
 build the hardware-specific Python venv (torch, chosen for CPU/CUDA/ROCm, is
 not bundled).
EOF

install -m 0755 "$PKG/scripts/postinstall.sh" "$FR/DEBIAN/postinst"
install -m 0755 "$PKG/scripts/preremove.sh"   "$FR/DEBIAN/prerm"
install -m 0755 "$PKG/scripts/postremove.sh"  "$FR/DEBIAN/postrm"

echo "== building .deb =="
DEB="$DIST/infinitemodel_${VER}_all.deb"
STAGED_DEB="$STAGE_BASE/infinitemodel_${VER}_all.deb"
dpkg-deb --root-owner-group --build "$FR" "$STAGED_DEB" >/dev/null
mkdir -p "$DIST"; cp "$STAGED_DEB" "$DEB"

echo "== done =="
echo "   $DEB"
command -v lintian >/dev/null 2>&1 && { echo "== lintian =="; lintian "$DEB" || true; }
echo
echo "Install:  sudo apt install $DEB   (or: sudo dpkg -i $DEB && sudo apt-get -f install)"
echo "Then:     sudo infinitemodel-setup --role both --cpu --extras controller,worker"
