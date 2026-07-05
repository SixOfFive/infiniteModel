# NVIDIA on Proxmox VE 9 / Debian 13 (Trixie) — bare-metal host driver for InfiniteModel

A repeatable playbook to make a **Proxmox VE 9.x** host's NVIDIA GPU usable by an InfiniteModel
worker running **directly on the host** (bare metal, same pattern as the CPU workers). The result is a
CUDA-capable host where `nvidia-smi` is green and `torch.cuda.is_available()` is `True` in the iM venv.

> **Scope.** Driver on the **host**; `im-worker` runs on the host. This is **mutually exclusive** with
> passing the same card through to a VM/LXC via vfio — a card is either bound by the host driver *or*
> handed to a guest, not both. For the passthrough variant, don't follow this doc.

> **You almost never need to hand-"backport" anything.** Debian 13 already ships the driver in
> `non-free`, and `trixie-backports` carries a newer build. The genuine obstacle on Proxmox 9 is
> **kernel-vs-DKMS compatibility**, because PVE ships very new kernels. That's what §2 is about.

---

## TL;DR

```bash
# 0. know your GPU generation (Turing+ => open modules; pre-Turing => proprietary)
lspci -nn | grep -i nvidia

# 1. pin a driver-friendly kernel (6.14 builds; 6.17/7.0 currently do NOT)
apt install proxmox-kernel-6.14 proxmox-headers-6.14
proxmox-boot-tool kernel pin 6.14.11-4-pve        # use the exact version you installed
reboot                                            # verify: uname -r  -> 6.14.x

# 2. repos: contrib/non-free/non-free-firmware + backports (see §3 for the sources snippet)
apt update

# 3. driver (Turing RTX20 or newer -> open modules):
apt install -t trixie-backports nvidia-open-kernel-dkms nvidia-driver
#    pre-Turing (Pascal GTX10 / Volta / Maxwell):
#    apt install -t trixie-backports nvidia-kernel-dkms nvidia-driver

# 4. Secure Boot? enroll the MOK (skip if SB is off):
mokutil --sb-state
mokutil --import /var/lib/dkms/mok.pub            # set a password, reboot, enroll at the blue screen

reboot
nvidia-smi                                        # green table = kernel module loaded + userspace OK
systemctl enable --now nvidia-persistenced        # headless latency + stable clocks
```

---

## 0. Confirm the GPU and its generation

```bash
lspci -nn | grep -i nvidia
```

The generation decides the kernel-module flavor (per the vault guide *"The NVIDIA driver stack on
Debian: open vs proprietary"*):

| GPU generation | Kernel module | Debian package |
|---|---|---|
| Turing (RTX 20 / GTX 16), Ampere (RTX 30), Ada (RTX 40), Hopper/Blackwell | **Open modules** (recommended / mandatory on newest) | `nvidia-open-kernel-dkms` |
| Pascal (GTX 10), Volta, Maxwell and older | **Proprietary** (open modules don't support these) | `nvidia-kernel-dkms` |

One driver serves every NVIDIA card in the box; the oldest card constrains the branch. Don't mix a
pre-Turing card and a Blackwell card in the same host — there's no single config that drives both.

---

## 1. The Proxmox-9 trap: fix the KERNEL before the driver

This is the step that makes PVE different from plain Debian. Proxmox ships its own, very new kernels
(`proxmox-kernel-*`), and the Debian-packaged **550** driver's DKMS build is kernel-sensitive:

- **Kernel 6.14** (PVE 9.0's default): the packaged 550 driver builds and runs. ✅
- **Kernel 6.17 / 7.0** (later PVE 9.x): DKMS build failures are widely reported. ⚠

So on PVE 9, **install and pin the 6.14 kernel** for the GPU host:

```bash
apt update
apt install proxmox-kernel-6.14 proxmox-headers-6.14
proxmox-boot-tool kernel list                     # see what's installed
proxmox-boot-tool kernel pin 6.14.11-4-pve        # <-- exact version from the list
reboot
uname -r                                           # must show 6.14.x before continuing
```

- `proxmox-boot-tool kernel pin` makes 6.14 the default boot entry and holds it across
  `apt full-upgrade`, so a later kernel bump can't silently pull you onto a version the driver won't
  build against.
- **Keep a fallback kernel installed** — don't prune. If a DKMS build ever fails, you want to boot the
  known-good kernel from GRUB's *Advanced options* rather than a headless black box.

> **Alternative if you must run a newer kernel** (e.g. hardware that needs 6.17+): skip the pin and use
> a newer NVIDIA branch from NVIDIA's CUDA/`cuda-drivers` repo (open modules) instead of the Debian
> 550. It's fiddlier and still kernel-sensitive (580.x has reported issues on 7.0), so **pinning 6.14 +
> packaged 550 is the recommended, lowest-friction path** for an iM GPU worker. Driver 550 gives CUDA
> 12.4, which is enough for current PyTorch — you gain nothing for inference by chasing 580.

---

## 2. Enable the repos (Debian components + backports)

Proxmox 9 is Debian 13 underneath and uses Debian's package repos for this. Enable `contrib`,
`non-free`, `non-free-firmware`, and `trixie-backports`. Add a file
`/etc/apt/sources.list.d/nonfree.sources`:

```
Types: deb
URIs: http://deb.debian.org/debian
Suites: trixie trixie-updates trixie-backports
Components: contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
```

```bash
apt update
```

`non-free-firmware` matters: the modern driver's GSP firmware (`firmware-misc-nonfree`) lives there.

---

## 3. Kernel headers — the #1 PVE gotcha

DKMS builds the module against **the running kernel's headers**. On Proxmox that is **not**
`linux-headers-amd64` — it's the PVE header package:

```bash
apt install proxmox-headers-$(uname -r)           # e.g. proxmox-headers-6.14.11-4-pve
# or the metapackage that tracks the default kernel:
apt install proxmox-default-headers
```

Installing `linux-headers-*` instead of `proxmox-headers-*` is the most common reason `nvidia-smi`
comes up empty after a clean-looking install — the module built against the wrong tree, or didn't build.

---

## 4. Install the driver

Pick the line matching your GPU generation from §0 (both pulled from backports for the newer 550 build):

```bash
# Turing (RTX 20 / GTX 16) or newer -> open kernel modules:
apt install -t trixie-backports nvidia-open-kernel-dkms nvidia-driver

# Pascal (GTX 10) / Volta / Maxwell -> proprietary kernel module:
apt install -t trixie-backports nvidia-kernel-dkms nvidia-driver
```

- Installing the driver **blacklists nouveau automatically** and rebuilds the initramfs.
- DKMS compiles the module now; watch the apt output for a **successful build** — do **not** reboot a
  remote host until you've confirmed it. Cross-check:

  ```bash
  dkms status                                       # nvidia/<ver>, <kernel>: installed
  ```

- Verify the exact package names on your box if apt complains (branch names drift):
  `apt-cache search '^nvidia-.*(open|kernel|driver)'`.

---

## 5. Secure Boot (MOK enrollment)

A DKMS-built module is unsigned; Secure Boot refuses to load unsigned modules, so the module builds
fine and then silently won't load — same missing-GPU symptom as a failed build.

```bash
mokutil --sb-state                                 # "SecureBoot enabled" or "disabled"
```

- **Disabled** → nothing to do.
- **Enabled** → enroll the DKMS Machine Owner Key:

  ```bash
  mokutil --import /var/lib/dkms/mok.pub            # set a one-time password
  reboot
  # At the blue MOK Manager screen: Enroll MOK -> Continue -> enter the password -> reboot
  ```

Most headless PVE hosts run with Secure Boot off; check rather than assume.

---

## 6. Reboot, verify the driver, enable persistence

```bash
reboot
nvidia-smi                                         # populated table, Driver 550.x, CUDA 12.4
systemctl enable --now nvidia-persistenced         # keep driver warm on a headless node
nvidia-smi -q | grep "Persistence Mode"            # -> Enabled
```

`nvidia-persistenced` avoids the multi-second re-init the driver otherwise pays on every cold job when
no client holds the GPU — a real latency win on an intermittent inference node.

---

## 7. Make InfiniteModel use the GPU

The host now has a CUDA driver; the iM worker needs a **CUDA build of PyTorch** in its venv (not the
CPU wheel):

```bash
# in the worker venv (e.g. /root/imenv or ~/infinitemodel/.venv)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- Expect `cuda` non-`None`, `is_available() == True`, and the card's name.
- **Match the wheel to the driver:** driver 550 → CUDA ≤ 12.4 → install a `cu124` (or `cu121`) PyTorch
  build. If `nvidia-smi` works but `torch.cuda.is_available()` is `False`, the venv has a CPU-only
  torch — reinstall the CUDA wheel.
- Restart the worker so it re-probes: iM auto-detects CUDA and the node joins the fleet as a **GPU
  node**. Confirm from the controller `/status` that the node reports a GPU with VRAM.
- Optional 24×7 hygiene: cap power for thermals/efficiency, e.g. `nvidia-smi -pl <watts>` (persist via
  a small systemd unit if you want it across reboots).
- **If this node also runs the CONTROLLER** (co-located controller+worker, as the main box does): the
  same venv additionally needs the controller stack — `fastapi uvicorn` plus the CONTROLLER-side
  multimodal deps `pillow` (images) and `soundfile` (audio-in), per the
  [README install section](../README.md#installation). ⚠ Miss `pillow` and EVERY vision request silently
  `ImportError`s ("No module named 'PIL'"), and because `transformers` caches the availability check at
  import you must **restart the controller** after installing (not just install). torchvision is NOT
  needed — every image processor has a pure-PIL backend, and it risks pulling a mismatched `torch`.

---

## 8. Ongoing: survive kernel upgrades

The whole reason for §1's pin: every PVE kernel change re-triggers a DKMS rebuild, and the packaged
550 driver only builds on 6.14.

- Leave the **6.14 pin in place** on GPU hosts. `apt full-upgrade` won't move you off it.
- If you ever unpin/upgrade the kernel, **confirm `dkms status` shows the module built for the new
  kernel BEFORE rebooting** a remote host. Keep the previous kernel installed as an escape hatch.
- Recovery if you're ever black-boxed: boot the old kernel from GRUB *Advanced options*, or at GRUB
  press `e` and append `nomodeset` to the `linux` line to get a basic console, then fix DKMS/headers.

---

## Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi`: *No devices were found* / command fails | DKMS didn't build, or built against wrong headers | `apt install proxmox-headers-$(uname -r)`; `dkms autoinstall`; check `dkms status` |
| Built cleanly but module won't load (still no GPU) | Secure Boot, MOK not enrolled | `mokutil --import /var/lib/dkms/mok.pub`, reboot, enroll — or disable Secure Boot |
| DKMS build **errors** on the running kernel | Kernel too new for 550 (6.17 / 7.0) | Pin 6.14 (§1); or move to NVIDIA's newer `cuda-drivers` branch |
| `nvidia-smi` green but `torch.cuda.is_available()` False | CPU-only torch in the venv | Reinstall a CUDA (`cu124`) PyTorch wheel |
| GPU vanished after a routine `apt full-upgrade` + reboot | Kernel bumped, DKMS silently failed | Boot old kernel; reinstall matching `proxmox-headers`; re-pin 6.14 |
| Want the GPU in a VM later | Host driver bound the card | This doc & vfio passthrough are mutually exclusive — unbind the host driver first |

---

## References

- Vault guide: *The NVIDIA driver stack on Debian: open vs proprietary* (open-vs-proprietary model,
  DKMS, MOK, persistence mode, recovery).
- Debian package: [`nvidia-open-kernel-dkms` in trixie](https://packages.debian.org/trixie/nvidia-open-kernel-dkms),
  [`nvidia-kernel-dkms` in trixie-backports](https://packages.debian.org/stable-backports/kernel/nvidia-kernel-dkms).
- [Debian Wiki — NvidiaGraphicsDrivers](https://wiki.debian.org/NvidiaGraphicsDrivers).
- [Debian 13 NVIDIA Driver Installation — LinuxConfig](https://linuxconfig.org/debian-13-nvidia-driver-installation).
- Proxmox forum (kernel-vs-driver compatibility on PVE 9):
  [NVIDIA DKMS fails on PVE 9 / kernel 6.17.4](https://forum.proxmox.com/threads/nvidia-dkms-fails-on-proxmox-ve-9-kernel-6-17-4-quadro-p2000.178353/),
  [NVIDIA 580.x + kernel 7.0 incompatibility on Trixie](https://forum.proxmox.com/threads/nvidia-580-x-kernel-7-0-incompatibility-on-trixie.183421/).
