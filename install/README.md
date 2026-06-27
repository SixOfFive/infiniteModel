# Portable InfiniteModel worker — USB / offline install

This folder makes the InfiniteModel **worker** (`client.py`) runnable on any
Windows or Linux x86‑64 box **without installing anything into the OS** — no
admin, no `apt`, no system Python required. Everything lives inside the repo
folder, so you can copy it to a USB stick and run it anywhere.

## Two phases

| Phase | Script | Runs where | Needs internet? |
|-------|--------|-----------|-----------------|
| **Prepare** (once) | `install/prepare.bat` / `install/prepare.sh` | a build box | **yes** — downloads everything |
| **Install** (per target) | `install.bat` / `install.sh` (repo root) | the target box / USB | **no** (offline from the bundle) |

### 1. Prepare the bundle (once, online)
On any machine with Python 3 + internet:

```
install\prepare.bat          (Windows)
./install/prepare.sh         (Linux)
```

This downloads, for **both** Windows and Linux:
- a standalone **CPython 3.13** runtime → `install/python/cpython-3.13-{windows,linux}.tar.gz`
- the full **wheel closure** for the worker deps → `install/wheels/{win,linux}/*.whl`
  (torch 2.12.0 **CPU**, transformers 5.12.1, safetensors 0.8.0,
  huggingface_hub 1.19.0, numpy 2.4.6, psutil 7.2.2 — pinned to the proven
  fleet versions).

Wheels are cross‑downloaded with explicit `cp313` tags, so you can prepare both
platforms from one machine. The **native** platform is always reliable; if a
cross‑download of the *other* OS's binary wheels fails, just run `prepare` on
that OS to complete it (it merges into the same bundle).

Then **copy the whole repo folder** to the USB stick. (Use a file copy, not
`git clone` — the bundled binaries are git‑ignored on purpose so they don't
bloat the repo.)

### 2. Install on the target (offline)
Plug the USB into the target box and run, from the repo root:

```
install.bat                  (Windows)
./install.sh                 (Linux)
```

Each installer:
1. extracts the bundled Python (or falls back to a system Python if present),
2. builds an offline venv (`install/.venv-win` or `install/.venv-linux`),
3. `pip install`s the worker deps from the bundled wheels (no internet),
4. verifies `torch`/`transformers`/`safetensors`/… import, and
5. prints the exact command to start the worker.

### 3. Start the worker
```
install\start-client.bat            (Windows)
./install/start-client.sh           (Linux)
```
Flags pass straight through to `client.py`, e.g.:
```
install\start-client.bat --device cpu          # force CPU (no GPU notice)
./install/start-client.sh --controller 10.0.0.5  # different controller
./install/start-client.sh --name fieldbox        # override hostname
```
Default controller is **192.168.15.103:50100** (BEAST). Edit the `start-client.*`
launcher to change the default, or override per‑run with `--controller`.

## Why a venv is built on the target instead of shipped pre‑built
A Python venv bakes **absolute paths** (drive letter / mount point) into
`pyvenv.cfg` and its launchers, so a venv built here would break when the USB is
mounted elsewhere. Instead the heavy part — every download and compile‑free
wheel — is staged ahead of time, and `install.*` assembles the venv from those
wheels **offline in ~1 minute**. Same outcome, but correct on any machine.

## GPU note
The bundle ships the **CPU** torch build (portable, no CUDA/driver needed). On a
box with an NVIDIA GPU the worker still runs on CPU with this build; to use the
GPU, install a CUDA torch build into the venv instead (that download is large
and driver‑specific, so it's intentionally not bundled).

## What's git‑ignored vs committed
Committed (source): `install.bat`, `install.sh`, and everything in this folder
*except* the downloaded/generated artifacts. Ignored (see `install/.gitignore`):
`python/`, `wheels/`, `runtime/`, `.venv-win/`, `.venv-linux/`, `manifest.txt`.
