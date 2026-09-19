# Node: RTX 4070 Ti SUPER — `beast` (GPU worker)

> **This doc describes `beast` as it is now: a Linux (Proxmox VE 9 / Debian 13) GPU worker.**
> That is a deliberate choice between the two things this file used to be. Until now the title
> said "beast", the role line said "GPU worker" — and every command below was the Windows
> controller+worker recipe from the box's previous life, down to `D:\infinitemodel` and
> `start_worker.bat`. A reader following it on the current machine would have failed at the
> first line. The Windows recipe was **not** deleted (it is the fleet's only Windows+NVIDIA
> writeup, and ACCELERATION.md's Windows Triton measurements were taken on *this* card): it
> now lives in [Appendix A](#appendix-a--the-windows-era-recipe-historical), clearly marked as
> history. §1–§6 are beast's own facts. `docs/nodes/README.md` already lists this doc as
> `beast` / Linux (Proxmox VE) / GPU worker, so this is the reading that keeps the index honest.

The host driver setup is a separate playbook: [../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md)
(kernel pin, DKMS, MOK, persistence). This doc assumes `nvidia-smi` is already green.

`beast` is a **GPU worker only** — the fleet controller moved to a dedicated host. Note the
distinction that trips people up: the controller *VM* still runs **on this physical box** (per
the 2026-08-14 CHANGELOG entry, VM 116 `iM` migrated onto the beast PVE host, which also
exports the model weights), but the *worker process* documented here does not host it and
`beast` is not the address you point workers at.

See also: [../ACCELERATION.md](../ACCELERATION.md) (int4 decode kernel matrix, the fused-MoE
opt-in, prefill chunking, CUDA-graph decode) and [../ROCM.md](../ROCM.md) (AMD recipe — not
this box, but the cross-platform reference).

---

## 1. Overview

| | |
|---|---|
| **Host** | `beast` |
| **GPU** | RTX 4070 Ti SUPER (GPU 0) — Ada Lovelace, **sm_89**. beast also has a second card, an **RTX 3060 (GPU 1, 12 GB, Ampere sm_86)**, served by its own worker — see [§3.1](#31-second-worker-for-the-rtx-3060-gpu-1) and [3060.md](3060.md) |
| **VRAM** | 16 GB (4070 Ti SUPER) + 12 GB (3060) |
| **Mem bandwidth** | ~672 GB/s (per ACCELERATION.md sweep) |
| **OS** | **Proxmox VE 9 / Debian 13 (trixie)** — the worker runs bare-metal on the PVE host, not in a guest ([../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md)) |
| **Role** | **GPU worker.** Not the controller — but the controller VM and the model store live on this same physical host |
| **Install dir** | `/root/infinitemodel` (venv `/root/imenv`) |
| **Worker unit** | `im-worker.service` (GPU 0, 4070 Ti SUPER) · `im-worker-3060.service` (GPU 1, 3060) |

> The install dir / venv / unit name are the ones the [../T2A.md](../T2A.md) enablement recipe
> uses for this box (`/root/imenv/bin/pip`, `systemctl restart im-worker`) and the ones the
> 2026-08-14 CHANGELOG entry names for the model mount (`/root/infinitemodel/models`). They are
> this box's actual layout, not a generic suggestion — a fresh install elsewhere can put them
> anywhere.

---

## 2. Install

**Driver first.** Ada is Turing-or-newer, so it takes the **open** kernel modules
(`nvidia-open-kernel-dkms`). Do not improvise this on Proxmox: the packaged 550 driver's DKMS
build only succeeds on kernel **6.14**, and PVE 9 ships newer kernels that break it. Follow
[../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md) §1–§6 (pin 6.14, install
`proxmox-headers-$(uname -r)` — *not* `linux-headers-*`, the single most common reason
`nvidia-smi` comes up empty — then the driver, then `nvidia-persistenced`).

**Python + torch.** Standard upstream CUDA wheel; the TheRock/ROCm indexes in ROCM.md do
**not** apply to this card.

```bash
python3 -m venv /root/imenv

# torch — CUDA 12.8 build. This box runs 2.11.0+cu128 (the version T2A.md pins into the
# ACE-Step constraints file so nothing can move it out from under the LLM stack).
/root/imenv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# App deps — the same pinned versions as the rest of the fleet (from ROCM.md's dep list):
/root/imenv/bin/pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

> (verify) **cu128 vs the driver branch.** PROXMOX9_NVIDIA.md §7 advises matching the wheel to
> the driver — 550 exposes CUDA 12.4, so it says install `cu124`/`cu121`. This box nonetheless
> runs `cu128` (T2A.md), which works because a CUDA 12.x runtime is minor-version-compatible
> with a 12.4 driver. Both statements are in the repo and they read as contradictory; the
> reconciliation above is **not** written down anywhere in it. Treat cu128-on-550 as *what this
> box does*, not as advice, until someone confirms the compat path deliberately.

**Fused-MoE opt-in — no extra toolchain on Linux.** Triton ships inside the Linux torch wheel;
all it needs is a C compiler and Python headers for the launcher stubs it JIT-compiles:

```bash
apt install -y gcc python3-dev
```

That is the whole story here. The MSVC Build Tools / CUDA-Toolkit-`ptxas` / `triton-windows`
dance in ACCELERATION.md's "Windows + NVIDIA" section (and in Appendix A below) exists only
because Windows has no Triton in the wheel — it is irrelevant on this install.

Ada (sm_89) is Ampere-or-newer, so the bf16 fused kernel compiles here.

---

## 3. Run the worker

```bash
cd /root/infinitemodel
/root/imenv/bin/python client.py --device cpu+gpu
```

- **No `--controller` flag.** `config.json` ships `controller_host: "auto"`, so the worker finds
  the controller by UDP broadcast on `discovery_port` (50099) and retries every 30 s until one
  answers. Discovery is the single lever: a future controller move needs zero edits here. Pass
  `--controller <ip>` only for something broadcast cannot cross (subnet/VLAN/VPN) — and note an
  explicit flag **wins over `config.json`**, so a stale one pins the worker to a dead address.
- `--device cpu+gpu` lets the box offer its GPU *and* CPU spill; `--device gpu` keeps it
  GPU-only. Match whatever the unit's `ExecStart` says.
- **Don't** edit `config.json` to point the worker — it is in `EXTRA_UPDATE_FILES` and a
  self-update reverts it.

**Persistence — `im-worker.service`.** A system unit (this box runs as root), enabled so it
comes back after a reboot. To enable the **fused-MoE tier**, add a drop-in rather than editing
the unit:

```ini
# /etc/systemd/system/im-worker.service.d/fused-moe.conf
[Service]
Environment=INFINITEMODEL_CUDA_FUSED_MOE=1
```

```bash
systemctl daemon-reload && systemctl restart im-worker
```

The fused forward installs at **load time**, so a restart alone is not enough — reload the model
too, or the running resident keeps the old path.

Run **one worker per GPU** — beast has two cards, so it runs two workers; see
[§3.1](#31-second-worker-for-the-rtx-3060-gpu-1). The registration collision is by *hostname*, so
a second worker on the same box needs a distinct `--name`.

> **Confirmed `ExecStart` (2026-09-19).** The primary unit runs
> `/root/imenv/bin/python client.py --device cpu+gpu` from `WorkingDirectory=/root/infinitemodel`
> with `Restart=always` — **no** `--name`, `--attn`, or legacy `--controller` flag (discovery is
> in use). This resolves the older "(verify) the exact ExecStart" note.

### 3.1 Second worker for the RTX 3060 (GPU 1)

beast has **two** GPUs, but a worker process serves exactly **one** — `worker_hw.detect_device()`
returns `torch.cuda.current_device()` (GPU 0) and `_gpu_mem_gb()` hardcodes device 0. So the
primary `im-worker` uses only the 4070 Ti SUPER; the 3060 sits idle until a **second worker** is
pinned to it. Established 2026-09-19 — beast is the fleet's first 2-worker host. The second unit:

```ini
# /etc/systemd/system/im-worker-3060.service
[Service]
WorkingDirectory=/root/infinitemodel
Environment=CUDA_VISIBLE_DEVICES=1
ExecStart=/root/imenv/bin/python client.py --device gpu --name beast-3060 --data-port 50201
Restart=always
RestartSec=5
```

Four settings are load-bearing:

- **`CUDA_VISIBLE_DEVICES=1`** pins the process to the physical 3060, which torch then sees as its
  *own* `cuda:0`. Both workers therefore print `cuda:0` — confirm the pin against the GPU **UUID**
  (`nvidia-smi --query-compute-apps=gpu_uuid,pid`), not the device string.
- **`--name beast-3060`** gives it a distinct fleet identity. The collision is by hostname, so a
  co-located second worker **must** rename or the two fight over registration.
- **`--data-port 50201`** is **mandatory**. The worker binds a *fixed* local data-plane port
  `50200` (the `--data-port` default) for inter-stage tensors; a second worker left on 50200 fails
  to bind. Give each co-located worker its own port.
- **`--device gpu`** (not `cpu+gpu`) keeps the second worker from re-advertising beast's single
  ~125 GB RAM pool as a second CPU-worker pool — the primary already offers CPU spill.

`systemctl enable --now im-worker-3060` starts it and persists it across reboot. Result: node
`beast-3060`, GPU 1 RTX 3060 (~11.6 GB usable), device-mode `gpu`. The 3060 tier guidance (int4; a
few layers in a split, or a small model) is in [3060.md](3060.md).

**Caveats — beast is the first 2-worker box, so these are unexercised.** Both workers run the
self-updater against the same `/root/infinitemodel/client.py` (a self-update could restart them
over each other); and TP rendezvous root-port assignment with two co-located workers is untested.

---

## 4. Optimal settings

**Quant: int4.** On NVIDIA, int4 dense decode uses torch's fused tinygemm
(`_weight_int4pack_mm`) and actually **beats bf16** (int4 7B 3.25 → 21.13 tok/s in the fleet
bench). With 16 GB, int4 is also what lets useful models fit — prefer it.

**Context:** prefill chunking is on by default (`INFINITEMODEL_PREFILL_CHUNK=2048`) and is
math-identical; leave it. Lower it (512–1024) only for very long contexts on a memory-tight
load. Decode is unaffected.

**Placement / what fits (16 GB):** this is a mid VRAM tier. Budget conservatively — the
controller VM and the model store share this physical host, so "free" RAM here is not all
yours, and InfiniteModel places stages against *physically free* VRAM while reserving each
layer's full-ctx KV. Prefer **sparse MoE with a low active-param ratio** (e.g. ~3B active) over
a big dense model: decode tok/s ≈ bandwidth ÷ active bytes-per-token, so a 30–35B-A3B MoE is far
faster than a dense model of similar footprint. For anything larger than fits in 16 GB,
**distribute it across the fleet** rather than forcing it onto this card (see Gotchas).

**The fused-MoE opt-in:** worth it for MoE models, where routed experts are a big share of
decode — ACCELERATION.md measured the expert-GEMM microbench at **24–34×** over the bf16-remat
default on this card, and the self-check went `-> ACTIVE` on all layers with coherent output.
On **dense** models the gain is small (tinygemm already dominates decode). Autotune note:
`num_stages=3` buys ~1.18× on narrow-N shapes (qwen3-a3b); the kernel sits at ~35–48% of the
card's ~672 GB/s peak. So enable it for MoE workloads; it's optional polish for dense.

> Those numbers were measured on this card **under Windows** (ACCELERATION.md's NVIDIA table
> says so explicitly). The kernel is the same and Linux is the easier Triton target, so they
> should hold or improve — but they have not been re-run since the rebuild. Re-bench before
> quoting them as Linux figures.

**Avoid:**
- Full-loading a big model on this box (see Gotchas).
- Treating 16 GB as a solo host for big models — pipeline them across the fleet instead.
- Split-K for the MoE GEMV — tested and **rejected** on this card (0.66–0.85×, slower).

---

## 5. Verify

1. **Driver and torch both see the GPU:**
   ```bash
   nvidia-smi
   /root/imenv/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   `nvidia-smi` green but `is_available()` False means the venv has a CPU-only torch — the
   single most common failure after a driver install (PROXMOX9_NVIDIA.md's troubleshooting
   table). (verify — the exact device-name string.)

2. **Worker registered:** the controller dashboard at `http://<controller>:21434` should list
   this node with its GPU and VRAM, or check the ring log:
   ```bash
   curl "http://<controller>:21434/logs?node=beast"
   ```

3. **Generation test:** load a small int4 model and generate through the controller's
   Ollama-compatible API:
   ```bash
   curl http://<controller>:21434/api/generate -d '{"model":"<model>","prompt":"The capital of France is"}'
   ```
   A coherent completion confirms the path end-to-end.

4. **MoE opt-in engaged (if enabled):** load a MoE model and look for
   `[int4] fused-MoE self-check ... -> ACTIVE` in the worker log
   (`GET /logs?node=beast`). `-> fallback` means it self-checked out or `gcc`/headers are
   missing, and the safe bf16-remat path is in use (still correct, just not accelerated).

---

## 6. Gotchas

- **Don't full-load a big model on this box.** The rule predates the controller move — a 67 GB
  `from_pretrained` crashed the controller back when the controller ran here — but it outlives
  that reason: a full load pulls every weight into *this* box's RAM instead of letting the
  controller place stages, and the controller VM still lives on this same physical host. Big
  models get **distributed**; this card contributes its 16 GB to a split.
- **A kernel upgrade can silently take the GPU away.** Every PVE kernel change re-triggers a
  DKMS rebuild and the packaged 550 driver only builds on 6.14 — which is why 6.14 is pinned.
  If you ever unpin, confirm `dkms status` shows the module built for the new kernel **before**
  rebooting; a remote reboot into a failed build is a headless black box. Recovery and the full
  matrix are in [../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md) §8.
- **One worker per *GPU*, each with a distinct `--name`.** The registration collision is by
  *hostname*, not by box: two workers both named `beast` fight over it. beast runs two workers —
  GPU 0 (4070 Ti SUPER) is `beast`, GPU 1 (3060) is `beast-3060` — see
  [§3.1](#31-second-worker-for-the-rtx-3060-gpu-1). A co-located second worker also needs its own
  `--data-port` (the default `50200` is a fixed bind).
- **Perf-only kernel changes carry no VERSION bump.** A self-update *stages* the new code but
  does **not** auto-restart the worker — `systemctl restart im-worker` (or the controller's
  `POST /restart?workers=1`) to pick up a kernel change.
- **Env changes apply at load time.** After setting or clearing `INFINITEMODEL_CUDA_FUSED_MOE`
  (or any switch), restart the worker **and reload the model** — the fused forward installs
  during load, so a resident that was loaded under the old setting keeps it.
- **`INFINITEMODEL_NO_FUSED_MOE=1`** is the kill switch — forces the bf16-remat expert path
  everywhere; use it to A/B the fused kernel.
- **`INFINITEMODEL_CUDA_GRAPH=<ctx>` — off by default, and an unproven win *here*.** This is a
  shipped, guarded integration, not a science experiment: the first decode captures
  `model.forward` over a StaticCache mirror, the second replays at a new position and
  self-checks against the eager DynamicCache decode, and on mismatch it **latches off
  permanently** with serving byte-identical to eager. So enabling it cannot corrupt output — the
  realistic risk is that it buys nothing. Why nothing: ACCELERATION.md's headline **5.56×** is a
  16-layer, compute-only probe on this exact card that deliberately **excludes the per-hop TCP
  transport**, and it only applies to single-node, standard-attention, uniform-CUDA models —
  neither describes the distributed splits this worker usually carries. (The older warning here,
  "not yet validated/safe", was wrong on both counts and is why this entry is long.) If you try
  it: set `<ctx>` to the serving ctx, restart the worker, reload, and confirm
  `[cudagraph] decode ACTIVE` in `GET /logs?node=beast` — anything else means it latched off.
- **Prefill OOM is not a concern here** the way it is on the AMD iGPU — NVIDIA has the
  mem-efficient SDPA backend; chunking is on by default regardless.

---

## Appendix A — the Windows-era recipe (historical)

**Do not run this on `beast`.** The box was wiped from Windows and rebuilt on Debian/Proxmox;
`D:\infinitemodel`, `start_worker.bat`, and the WDDM caveats below describe a machine that no
longer exists. It is kept for two reasons: it is the fleet's only Windows + NVIDIA worker
recipe, and ACCELERATION.md's Windows Triton numbers (fused MoE `-> ACTIVE`, the 24–34×
expert-GEMM microbench) were measured on this card **in this configuration** — so the recipe is
what those numbers were taken under.

**Python:** CPython 3.14 (the `triton-windows` 3.7.1 wheel was validated against 3.14).

**torch:** `pip install --user torch --index-url https://download.pytorch.org/whl/cu128`.

**Advanced MoE tier — the Windows-only toolchain.** Triton is not in the Windows torch wheel,
so all three of these are needed or the kernel self-checks out and falls back to bf16-remat:

1. **Visual Studio Build Tools** with "Desktop development with C++" (provides `cl.exe`).
2. **CUDA Toolkit 12.8 (toolkit only, no driver)** — provides `ptxas` + `cuda.lib`, which the
   pip `nvidia-cuda-*` wheels do not ship on Windows:
   ```bat
   cuda_12.8.x_windows_network.exe -s nvcc_12.8 cudart_12.8 cuobjdump_12.8 nvdisasm_12.8
   ```
3. **triton-windows** (the woct0rdho fork): `pip install --user triton-windows`.

**Worker launch** — `start_worker.bat`:

```bat
set "CC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "INFINITEMODEL_CUDA_FUSED_MOE=1"
cd /d D:\infinitemodel
python client.py --controller <controller-ip> --control-port 50100
```

**The Windows gotcha worth remembering even off Windows — WDDM / session-0 invisibility.** A
consumer GeForce under the WDDM driver is **not visible from session 0**, so a Windows *service*
or a "run whether logged on or not" scheduled task sees no GPU at all. Persistence had to be an
interactive-desktop launch (`shell:startup`). This is the constraint that makes a Windows GPU
worker structurally worse than a Linux one, and a large part of why the box moved.

---

_For anything not covered here, defer to [../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md) (host
driver), [../ACCELERATION.md](../ACCELERATION.md) (kernels, env switches) and
[../ROCM.md](../ROCM.md) (AMD reference)._
