# InfiniteModel — Linux packages (.deb / .rpm)

System packages that install the code + systemd units, then let the admin build
a hardware-specific venv. Built on the packaging foundation (the wheel +
`../pyproject.toml` extras); see `../README.md` for the design.

## What the package does

- Installs the **wheel** to `/opt/infinitemodel/wheel/`, the `infinitemodel-setup`
  CLI to `/opt/infinitemodel/bin` (symlinked onto PATH), and two systemd units
  (`infinitemodel-controller`, `infinitemodel-worker`) to
  `/usr/lib/systemd/system/`.
- Post-install creates the `infinitemodel` system user and the writable app home
  `/var/lib/infinitemodel` (state, models, cache; self-update writes here). It
  does **not** build the venv or start services — that step is hardware-specific
  and can pull gigabytes of torch, so it is a deliberate manual step.
- `sudo infinitemodel-setup ...` builds `/opt/infinitemodel/venv`, installs the
  right torch (`--cpu` / `--cuda cuNNN` / `--rocm gfxNNNN`) then the wheel with
  chosen `--extras`, wires the launchers, and can `--start` the services.

Why not bundle torch or run it in the package's post-install: distro package
managers can't know the box's accelerator, the versions differ per backend, and
a multi-GB download inside a dpkg/rpm transaction is fragile. See the CHANGELOG
`#packaging` entry.

## Install

```bash
# Debian/Ubuntu
sudo apt install ./infinitemodel_<ver>_all.deb
# Fedora/RHEL/openSUSE
sudo dnf install ./infinitemodel-<ver>.noarch.rpm

# then, once, for this machine:
sudo infinitemodel-setup --role worker --cuda cu128 --extras worker,vision,stt --start
sudo infinitemodel-setup --role both   --cpu         --extras controller,worker
sudo infinitemodel-setup --role controller --no-torch --extras controller
```

`espeak-ng` (tts) and `libsndfile1` (stt/audio/music) are Recommends on deb;
install them when you enable those backends.

## Build

Canonical spec is `nfpm.yaml` — one file, both formats:

```bash
packaging/linux-pkg/build_pkgs.sh      # needs `nfpm`; builds .deb + .rpm
```

No `nfpm` on the build box? The `.deb` builds natively with `dpkg-deb`:

```bash
packaging/linux-pkg/build_deb.sh       # .deb only, no extra tooling
```

Both read the version from `server.py`'s `VERSION` and build the wheel first if
it is missing. Keep `build_deb.sh` in sync with `nfpm.yaml` (the yaml leads).

## Files

- `nfpm.yaml` — canonical package spec (deb + rpm).
- `build_pkgs.sh` — nfpm build (both formats).
- `build_deb.sh` — native dpkg-deb build (.deb only, no nfpm).
- `infinitemodel-setup` — the venv builder run after install.
- `systemd/*.service` — controller and worker units.
- `scripts/{postinstall,preremove,postremove}.sh` — maintainer scripts (shared
  by both formats).
