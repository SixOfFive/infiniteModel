#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel — package builder (foundation: sdist + wheel + installer tgz)
#
#  Assembles a CLEAN staging tree from packaging/runtime-manifest.txt (an
#  allowlist — see that file), stamps the version from server.py's VERSION
#  constant, refuses to ship secrets/dev cruft, and builds:
#     dist/infinitemodel-<ver>.tar.gz              (pip-installable sdist)
#     dist/infinitemodel-<ver>-py3-none-any.whl    (wheel)
#     dist/infinitemodel-<ver>-installer.tar.gz    (wheel + bootstrap.sh + docs)
#
#  Run on an internet-connected build box (needs the `build` frontend).
#  Usage:  packaging/build.sh
# ===========================================================================
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$PKG/.." && pwd)"
DIST="$REPO/dist"
STAGE="$DIST/_stage"
PAYLOAD="$STAGE/src/infinitemodel/_payload"
MANIFEST="$PKG/runtime-manifest.txt"

echo "== InfiniteModel package build =="
echo "   repo:  $REPO"

# --- version: single source of truth is server.py's VERSION -----------------
VER="$(grep -oE 'VERSION = "[^"]+"' "$REPO/server.py" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
[ -n "$VER" ] || { echo "[ERROR] could not read VERSION from server.py"; exit 1; }
echo "   version: $VER (from server.py)"

# --- clean staging ----------------------------------------------------------
rm -rf "$STAGE"
mkdir -p "$PAYLOAD"
cp -r "$PKG/src" "$STAGE/"          # launcher package (src/infinitemodel/*)
cp "$PKG/pyproject.toml" "$STAGE/"
cp "$PKG/README.md" "$STAGE/" 2>/dev/null || echo "# InfiniteModel" > "$STAGE/README.md"
cp "$REPO/LICENSE" "$STAGE/" 2>/dev/null || true

# stamp __version__ from server.py
sed -i -E "s/^__version__ = \".*\"/__version__ = \"$VER\"/" \
    "$STAGE/src/infinitemodel/__init__.py"

# --- copy payload strictly from the manifest allowlist ----------------------
echo "== staging payload from manifest =="
n=0
while IFS= read -r line; do
  f="${line%%#*}"; f="$(echo "$f" | xargs)"   # strip comments + whitespace
  [ -z "$f" ] && continue
  [ -f "$REPO/$f" ] || { echo "[ERROR] manifest entry missing in repo: $f"; exit 1; }
  cp "$REPO/$f" "$PAYLOAD/$f"
  n=$((n+1))
done < "$MANIFEST"
# seed an EMPTY custom_models.json (never copy the repo's personal one)
printf '{}\n' > "$PAYLOAD/custom_models.json"
echo "   staged $n manifest files + seeded custom_models.json"

# --- safety guard: nothing forbidden slipped into the payload ---------------
echo "== safety scan of payload =="
BAD=0
for pat in 'hf_token' 'im_*.json' 'scratch_*' 'test_*' 'bench_*' '*.wav' '*.log' '*_history.json'; do
  if compgen -G "$PAYLOAD/$pat" > /dev/null; then
    echo "[ERROR] forbidden file(s) matching '$pat' in payload"; BAD=1
  fi
done
# token-shaped secret content check
if grep -rIlE 'hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----' "$PAYLOAD" 2>/dev/null; then
  echo "[ERROR] secret-shaped content found in payload"; BAD=1
fi
[ "$BAD" -eq 0 ] || { echo "[ABORT] payload failed the safety scan"; exit 1; }
echo "   clean."

# --- build sdist + wheel ----------------------------------------------------
echo "== building sdist + wheel =="
if ! python3 -c "import build" 2>/dev/null; then
  echo "[ERROR] the 'build' frontend is missing. Install it on the build box:"
  echo "        python3 -m pip install build"
  exit 1
fi
( cd "$STAGE" && python3 -m build --sdist --wheel --outdir "$DIST" )

WHEEL="$(ls -t "$DIST"/infinitemodel-"$VER"-*.whl | head -1)"
[ -f "$WHEEL" ] || { echo "[ERROR] wheel not produced"; exit 1; }

# --- assemble the self-contained installer tarball --------------------------
echo "== assembling installer tarball =="
ITMP="$DIST/_installer/infinitemodel-$VER"
rm -rf "$DIST/_installer"; mkdir -p "$ITMP"
cp "$WHEEL" "$ITMP/"
cp "$PKG/bootstrap.sh" "$ITMP/"; chmod +x "$ITMP/bootstrap.sh"
cp "$PKG/README.md" "$ITMP/" 2>/dev/null || true
cp "$REPO/LICENSE" "$ITMP/" 2>/dev/null || true
( cd "$DIST/_installer" && tar -czf "$DIST/infinitemodel-$VER-installer.tar.gz" "infinitemodel-$VER" )
rm -rf "$DIST/_installer"

echo
echo "== done =="
ls -lh "$DIST"/infinitemodel-"$VER"* | sed 's/^/   /'
echo
echo "Install (any Linux):"
echo "   tar -xzf dist/infinitemodel-$VER-installer.tar.gz"
echo "   cd infinitemodel-$VER && ./bootstrap.sh --cpu --extras controller,worker"
