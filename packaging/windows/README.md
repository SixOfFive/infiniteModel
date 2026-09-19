# InfiniteModel - Windows installer

A per-user NSIS installer (no admin) that installs the code + launchers and
builds a hardware-specific venv, reusing the packaging foundation (the wheel +
`../pyproject.toml` extras). Same model as the Linux packages: torch is not
bundled; the installer's post-step (`bootstrap.ps1`) installs the torch build
you pick plus the backends you tick.

## What the installer does

- Installs into `%LOCALAPPDATA%\Programs\InfiniteModel`: the wheel (under
  `wheel\`), `bootstrap.ps1`, the two `.cmd` launchers, README, LICENSE.
- **Components page:** roles (Controller, Worker) and optional backends (Vision,
  STT, TTS, T2I, MusicGen, Kimi-Linear, ACE-Step) - each maps to a pip extra.
- **Compute page:** radio for NVIDIA CUDA 12.8 / CUDA 12.6 / CPU-only. ACE-Step
  is skipped on CPU (it needs an NVIDIA Ampere+ GPU) with a warning.
- Runs `bootstrap.ps1 -Flavor <cuNNN|cpu> -Extras <ticked>` to build
  `...\InfiniteModel\venv`, install torch + `wheel[extras]`, and (for TTS) add
  `kokoro`/`misaki` `--no-deps`.
- Creates Start Menu shortcuts to the launchers. The launchers set
  `INFINITEMODEL_HOME=%LOCALAPPDATA%\InfiniteModel` (the writable app home where
  state, models and self-updates live) and run the venv console scripts, which
  self-supervise (relaunch on the exit-42 self-update).

Prerequisite: **Python 3.10+** on PATH (the `py` launcher or `python`). The
installer's setup step reports clearly if it is missing
(`winget install Python.Python.3.12`). Bundling a standalone CPython is a
possible future enhancement.

## Build the installer (on Linux - no Windows/Wine needed)

```bash
sudo apt install nsis            # Debian/Ubuntu  (Fedora: sudo dnf install mingw32-nsis)
packaging/windows/build_installer.sh
# -> dist/infinitemodel-<ver>-setup.exe
```

`build_installer.sh` reads the version from `server.py`, builds the wheel first
if needed, and invokes `makensis` with the right defines.

## Files

- `win-installer.nsi` - the NSIS installer script (MUI2; components + compute page).
- `bootstrap.ps1` - the venv builder run post-install (Windows twin of `../bootstrap.sh`).
- `infinitemodel-controller.cmd`, `infinitemodel-worker.cmd` - Start Menu launchers.
- `build_installer.sh` - builds the `.exe` with `makensis`.
