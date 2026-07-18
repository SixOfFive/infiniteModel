#!/usr/bin/env python3
"""Populate the portable InfiniteModel worker bundle under ./install/.

Run this ONCE on an internet-connected box (it is the only step that needs the
network). It downloads, for BOTH Windows and Linux:

  * a standalone CPython 3.13 runtime (python-build-standalone, install_only)
      -> install/python/cpython-3.13-{windows,linux}.tar.gz
  * the full wheel closure for the worker deps (torch CPU + transformers stack)
      -> install/wheels/{win,linux}/*.whl

Nothing is installed into the OS. The wheels are cross-downloaded with explicit
cp313 tags, so you can prepare both platforms from a single machine. The native
platform is always reliable; the other is best-effort (if a cross-download of a
binary wheel fails, prepare the missing OS by running this on that OS).

Usage:
    python _fetch.py            # both platforms (default)
    python _fetch.py --win      # Windows wheels + both Pythons
    python _fetch.py --linux    # Linux wheels + both Pythons
    python _fetch.py --skip-python   # wheels only
    python _fetch.py --skip-wheels   # Python runtimes only
    python _fetch.py --force    # re-download even if files exist
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.join(HERE, "python")
WHEELS = os.path.join(HERE, "wheels")

# --- pinned to the proven fleet versions (provision_worker.sh) ----------------
PYVER = "3.13"
ABI = "cp313"
IMPL = "cp"
TORCH = "torch==2.13.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
REST = [
    "transformers==5.12.1",
    "safetensors==0.8.0",
    "huggingface_hub==1.19.0",
    "numpy==2.4.6",
    "psutil==7.2.2",
]
# Multiple linux platform tags widen wheel matching (manylinux_2_28 carries the
# torch CPU wheel; 2_17/2014 carry most of the rest).
PLATFORMS = {
    "win": ["win_amd64"],
    "linux": ["manylinux_2_28_x86_64", "manylinux2014_x86_64", "manylinux_2_17_x86_64"],
}
PB_API = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
PY_ASSET_RE = {
    "windows": re.compile(r"^cpython-(3\.13\.\d+)\+.*-x86_64-pc-windows-msvc-install_only\.tar\.gz$"),
    "linux": re.compile(r"^cpython-(3\.13\.\d+)\+.*-x86_64-unknown-linux-gnu-install_only\.tar\.gz$"),
}
UA = {"User-Agent": "infinitemodel-prepare/1.0"}


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


def download(url: str, dest: str, force: bool = False) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        print(f"  [skip] {os.path.basename(dest)} ({_human(os.path.getsize(dest))} already present)")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  [get ] {os.path.basename(dest)} <- {url}")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            mark = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got - mark >= (25 << 20):  # log every ~25 MB
                    mark = got
                    pct = f" {100*got//total}%" if total else ""
                    print(f"        {_human(got)}{pct}")
        os.replace(tmp, dest)
        print(f"  [ok  ] {os.path.basename(dest)} ({_human(os.path.getsize(dest))})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {os.path.basename(dest)}: {exc}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def resolve_python_assets() -> dict:
    print("== resolving standalone Python (python-build-standalone latest) ==")
    try:
        req = urllib.request.Request(PB_API, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            rel = json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] GitHub API: {exc}")
        return {}
    found: dict = {}
    for asset in rel.get("assets", []):
        name = asset.get("name", "")
        for osk, rx in PY_ASSET_RE.items():
            m = rx.match(name)
            if m:
                ver = tuple(int(x) for x in m.group(1).split("."))
                if osk not in found or ver > found[osk][0]:
                    found[osk] = (ver, asset.get("browser_download_url"), name)
    for osk, (ver, _url, name) in found.items():
        print(f"  {osk:7s} -> {name}")
    return {k: (v[1]) for k, v in found.items()}


def fetch_pythons(force: bool) -> dict:
    assets = resolve_python_assets()
    status = {}
    targets = {"windows": "cpython-3.13-windows.tar.gz", "linux": "cpython-3.13-linux.tar.gz"}
    for osk, fname in targets.items():
        url = assets.get(osk)
        if not url:
            print(f"  [skip] no {osk} asset resolved")
            status[osk] = False
            continue
        status[osk] = download(url, os.path.join(PYDIR, fname), force=force)
    return status


def pip_download(os_name: str) -> bool:
    outdir = os.path.join(WHEELS, os_name)
    os.makedirs(outdir, exist_ok=True)
    base = [
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:",
        "--python-version", PYVER,
        "--implementation", IMPL,
        "--abi", ABI,
        "-d", outdir,
    ]
    for plat in PLATFORMS[os_name]:
        base += ["--platform", plat]

    ok = True
    print(f"== wheels [{os_name}] pass 1/2: torch CPU (from {TORCH_INDEX}) ==")
    rc = subprocess.run(base + ["--index-url", TORCH_INDEX, TORCH]).returncode
    if rc != 0:
        print(f"  [WARN] torch pass for {os_name} failed (rc={rc})")
        ok = False
    print(f"== wheels [{os_name}] pass 2/2: transformers stack (from PyPI) ==")
    rc = subprocess.run(base + REST).returncode
    if rc != 0:
        print(f"  [WARN] transformers-stack pass for {os_name} failed (rc={rc})")
        ok = False
    return ok


def write_manifest() -> None:
    lines = ["InfiniteModel portable worker bundle — manifest", ""]
    grand = 0
    for label, path in (("python", PYDIR), ("wheels/win", os.path.join(WHEELS, "win")),
                        ("wheels/linux", os.path.join(WHEELS, "linux"))):
        if not os.path.isdir(path):
            lines.append(f"[{label}] (none)")
            continue
        sub = 0
        names = sorted(os.listdir(path))
        lines.append(f"[{label}] {len(names)} file(s)")
        for n in names:
            fp = os.path.join(path, n)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                sub += sz
                grand += sz
                lines.append(f"    {n}  ({_human(sz)})")
        lines.append(f"    -- subtotal {_human(sub)}")
        lines.append("")
    lines.append(f"TOTAL: {_human(grand)}")
    out = os.path.join(HERE, "manifest.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n(manifest written to {out})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate the portable worker bundle.")
    ap.add_argument("--win", action="store_true", help="prepare Windows wheels")
    ap.add_argument("--linux", action="store_true", help="prepare Linux wheels")
    ap.add_argument("--skip-python", action="store_true", help="don't fetch the Python runtimes")
    ap.add_argument("--skip-wheels", action="store_true", help="don't fetch wheels")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    do_win = args.win or not (args.win or args.linux)
    do_linux = args.linux or not (args.win or args.linux)

    print("=" * 70)
    print("InfiniteModel worker bundle — fetch")
    print(f"  python runtimes : {'skip' if args.skip_python else 'both (win+linux)'}")
    print(f"  wheels          : {'skip' if args.skip_wheels else (('win ' if do_win else '') + ('linux' if do_linux else '')).strip()}")
    print(f"  driver python   : {sys.version.split()[0]} ({sys.executable})")
    print("=" * 70)

    failures = []
    if not args.skip_python:
        st = fetch_pythons(args.force)
        for osk, okk in st.items():
            if not okk:
                failures.append(f"python:{osk}")

    if not args.skip_wheels:
        if do_win and not pip_download("win"):
            failures.append("wheels:win")
        if do_linux and not pip_download("linux"):
            failures.append("wheels:linux")

    print("\n" + "=" * 70)
    write_manifest()
    print("=" * 70)
    if failures:
        print(f"[!] incomplete: {', '.join(failures)}")
        print("    Native-platform downloads are reliable; for a failed *other*-OS")
        print("    wheel set, re-run this script ON that OS to complete it.")
        return 1
    print("[OK] bundle ready. Copy the whole repo folder to USB; run install.bat")
    print("     (Windows) or ./install.sh (Linux) on the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
