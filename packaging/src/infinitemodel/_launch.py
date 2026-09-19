"""Launcher for the InfiniteModel controller and worker console scripts.

WHY A LAUNCHER INSTEAD OF PLAIN ENTRY POINTS
--------------------------------------------
The runtime treats ``os.path.dirname(os.path.abspath(__file__))`` as its
*writable root*: it reads/writes ``node_config.json``, ``engine_config.json``,
download/history state, the ``models/`` and ``cache/`` trees there, and its
self-update rewrites its own ``.py`` files in place. A site-packages directory
is the wrong home for that (usually read-only; rewriting installed code is
hostile). So we ship the runtime as *payload data* inside this package and, at
launch, materialise it into a per-user writable directory, then exec the venv's
Python on ``<home>/server.py`` (or ``client.py``). Running a script file puts
its own directory on ``sys.path[0]``, so ``import wire`` etc. and every
``dirname(__file__)`` path resolve to the app home exactly as a git checkout
would — no source edits to the runtime were needed.

APP HOME resolution (first hit wins):
  1. $INFINITEMODEL_HOME
  2. $XDG_DATA_HOME/infinitemodel   (if XDG_DATA_HOME set)
  3. ~/.local/share/infinitemodel

DRY RUN
-------
Set $INFINITEMODEL_LAUNCH_DRYRUN=1 to sync the payload and print the resolved
interpreter, target script, app home and argv, then exit 0 without exec'ing.
Used by the packaging smoke test to prove the plumbing without importing torch.
"""
from __future__ import annotations

import os
import shutil
import sys
from importlib import resources
from pathlib import Path

from . import __version__

_STAMP = ".payload-version"


def _app_home() -> Path:
    env = os.environ.get("INFINITEMODEL_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "infinitemodel"
    return Path.home() / ".local" / "share" / "infinitemodel"


def _sync_payload(home: Path) -> None:
    """Copy bundled payload into `home` when missing or version-mismatched.

    Idempotent. Never deletes files the runtime created (state, models, cache,
    self-updated modules) — we only overwrite our own shipped files, and only
    when the stamped payload version differs from this package's version.
    """
    home.mkdir(parents=True, exist_ok=True)
    stamp = home / _STAMP
    current = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else ""
    if current == __version__ and (home / "server.py").exists():
        return  # already materialised at this version

    payload = resources.files(__package__).joinpath("_payload")
    for entry in payload.iterdir():
        name = entry.name
        # Ship only the flat runtime files. Skip anything else that may appear
        # next to them in site-packages — notably the __pycache__/ dir and .pyc
        # files pip byte-compiles from the payload .py (they are data to us,
        # never imported from here, so their bytecode is useless).
        if not (name.endswith(".py") or name.endswith(".json")):
            continue
        with resources.as_file(entry) as src:
            dst = home / name
            # custom_models.json is user-editable state: seed once, never clobber.
            if name == "custom_models.json" and dst.exists():
                continue
            shutil.copyfile(src, dst)
    stamp.write_text(__version__ + "\n", encoding="utf-8")


def _run(script: str) -> None:
    home = _app_home()
    _sync_payload(home)
    target = str(home / script)
    argv = [sys.executable, target, *sys.argv[1:]]

    if os.environ.get("INFINITEMODEL_LAUNCH_DRYRUN") == "1":
        print(f"infinitemodel {__version__} (dry run)")
        print(f"  interpreter : {sys.executable}")
        print(f"  script      : {target}")
        print(f"  app home    : {home}")
        print(f"  exists      : {os.path.exists(target)}")
        print(f"  argv        : {argv}")
        return

    # Run in the app home so any cwd-relative behaviour also lands there.
    os.chdir(home)

    # Supervisor loop. server.py / client.py exit with code 42 when they
    # self-update their own files in `home`; relaunch to pick up the new code.
    # Any other exit code is final and is propagated to our caller. We run a
    # CHILD process rather than os.execv on purpose: os.execv cannot propagate
    # the child's exit code to a parent launcher on Windows, which would break
    # the 42-relaunch and the .cmd / systemd supervisors layered on it. On a
    # clean service stop the manager kills the whole process group, so the
    # child receives the signal directly — no forwarding needed here.
    import subprocess
    while True:
        rc = subprocess.run(argv).returncode
        if rc == 42:
            continue
        sys.exit(rc)


def controller() -> None:
    """Entry point: infinitemodel-controller — runs server.py (the controller)."""
    _run("server.py")


def worker() -> None:
    """Entry point: infinitemodel-worker — runs client.py (a compute worker)."""
    _run("client.py")
