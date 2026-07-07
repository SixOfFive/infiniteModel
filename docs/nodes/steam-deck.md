# Node: steamdeck (Steam Deck — CPU worker)

A Valve Steam Deck running as an InfiniteModel **CPU worker**. By default the GPU is **not**
used — the Deck contributes its CPU + RAM to the fleet's capacity pool. ROCm on the Deck's
iGPU is an unverified research item; see [ROCm on the Deck](#rocm-on-the-deck--research-only-verify).

See also: [../ACCELERATION.md](../ACCELERATION.md) (int4-decode kernel matrix, CPU path),
[../ROCM.md](../ROCM.md) (the validated RDNA ROCm recipe — for gfx1151, **not** the Deck).

## 1. Overview

| | |
|---|---|
| **Host** | `steamdeck` |
| **APU** | gfx1033 "Van Gogh" — RDNA2, 8 CU iGPU |
| **VRAM** | small UMA carve-out (shared with system RAM; APU unified memory) |
| **OS** | SteamOS (Arch-based, **immutable root filesystem**) |
| **Role** | **CPU worker** — GPU not used by default; capacity (not single-stream speed) |
| **Device flag** | `--device cpu` |

Per [../ACCELERATION.md](../ACCELERATION.md), CPU/RAM nodes exist for **capacity** — fitting
models too big for the GPU pool — **not** single-stream speed. CPU decode is bound by DDR
bandwidth (~50–90 GB/s) and is fundamentally ~5–10× slower than a GPU regardless of kernel.
None of the Triton fast-path kernels (fused MoE, split-K dense) run on a CPU worker.

## 2. Install

SteamOS's root fs is **read-only/immutable**, so do **not** try to `pacman -S` system
packages or write outside `$HOME`. Everything below lives in your home directory.

Python 3 is present on SteamOS. Create a venv under `$HOME` and install the **CPU** torch
build (no CUDA/ROCm wheel — plain CPU torch):

```bash
# venv in $HOME (writable on SteamOS)
python3 -m venv ~/imenv

# CPU-only PyTorch
~/imenv/bin/python -m pip install --upgrade pip
~/imenv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# App deps — same pinned versions as the rest of the fleet (from ROCM.md step 4)
~/imenv/bin/python -m pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

> The `torch ... /whl/cpu` index is the standard CPU wheel; the arch-specific ROCm
> indexes in [../ROCM.md](../ROCM.md) (e.g. `rocm.nightlies.amd.com/v2/gfx1151/`) are for
> RDNA3.5 GPU workers and do **not** apply to the Deck's CPU role. Exact Python version on
> SteamOS — `python3 --version` on the box (verify).

Clone the repo into `$HOME` as well:

```bash
git clone https://github.com/SixOfFive/infiniteModel ~/infinitemodel
```

## 3. Run the worker

Launch `client.py` as a **CPU** worker pointed at the fleet controller (BEAST `:21434` —
control port `50100`). The Deck does **not** run a controller; it only joins an existing
fleet. The `--controller` flag overrides `config.json`'s `controller_host` (don't edit
`config.json` — it's in `EXTRA_UPDATE_FILES` and a self-update would revert it; per
[../ROCM.md](../ROCM.md)):

```bash
cd ~/infinitemodel
setsid ~/imenv/bin/python client.py --controller <controller-ip> --device cpu \
    >~/worker.log 2>&1 </dev/null &
```

`<controller-ip>` is the BEAST controller's LAN address. `--device cpu` keeps everything
off the iGPU.

> The `--control-port 50100` flag form is shown for NVIDIA boxes in
> [../ACCELERATION.md](../ACCELERATION.md); for joining the fleet, `--controller <ip>` is the
> load-bearing flag and the control port defaults to the fleet's `50100`. Pass
> `--control-port 50100` explicitly if your fleet uses a non-default port (verify).

### Persistence

**As deployed (verified on the box 2026-07-07):** the worker runs as a **system** unit,
`/etc/systemd/system/infinitemodel-worker.service` (enabled, `WantedBy=multi-user.target`),
with the venv at `~/infinitemodel/.venv` — **not** `~/imenv`, and **not** a
`systemctl --user` unit:

```ini
[Unit]
Description=InfiniteModel worker -> controller <old-controller-ip>
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=deck
Environment=HOME=/home/deck
WorkingDirectory=/home/deck/infinitemodel
ExecStart=/home/deck/infinitemodel/.venv/bin/python /home/deck/infinitemodel/client.py --controller <controller-ip> --control-port 50100 --name steamdeck --ram 4x-LPDDR5-5500 --no-clean
Restart=always
RestartSec=5
OOMScoreAdjust=800
Nice=5

[Install]
WantedBy=multi-user.target
```

(The unit Description may still name a previous controller IP; the load-bearing
`--controller <controller-ip>` in ExecStart is what counts.) Restart with
`sudo systemctl restart infinitemodel-worker.service`.

> A system unit under `/etc` survives SteamOS updates in practice (the `/etc` overlay is
> writable and persistent), but the original `systemctl --user` + `loginctl enable-linger`
> approach is the no-`sudo` alternative if this box is ever rebuilt.

## 4. Optimal settings

This is a **CPU/capacity** node — tune for fit, not speed.

- **Quant:** **int4** (smallest footprint; ¼ the bytes read per token). On CPU, dense int4
  uses torch's CPU tinygemm (`_weight_int4pack_mm_for_cpu`) when present, else a
  dequant→fp32 GEMM; MoE experts always bf16-rematerialize per expert
  ([../ACCELERATION.md](../ACCELERATION.md), CPU section).
- **Placement:** let the controller place stages here only when the GPU pool is full — the
  Deck is a **spillover/capacity** stage, not a primary compute stage. Don't pin a hot model
  to it.
- **Context:** keep modest — KV cache lives in the (small, shared) Deck RAM. Long contexts
  on a memory-tight box are a real risk; if a long prefill OOMs, lower
  `INFINITEMODEL_PREFILL_CHUNK` (default 2048 → e.g. 512) per
  [../ACCELERATION.md](../ACCELERATION.md). The chunked-prefill default is already ON.
- **Models that fit/run well:** small sparse **MoE with low active-param ratio** (~3B active)
  decode fastest per byte read — prefer those over big dense models
  ([../ACCELERATION.md](../ACCELERATION.md), "Choosing a fast model"). The Deck is best as a
  small slice of a larger pipeline.
- **Avoid:** running it as the only/primary node for a model; big dense models (every param
  read per token → very slow on DDR bandwidth); expecting GPU-class tok/s — none of the
  Triton kernels help a CPU worker.

## 5. Verify

1. **CPU torch sanity** (no GPU expected):
   ```bash
   ~/imenv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available()); \
   a=torch.randn(2048,2048); (a@a).sum().item(); print('matmul OK')"
   ```
   Expect `cuda avail False` and `matmul OK` (CPU worker).
2. **Registration:** confirm the node appears in the fleet — controller dashboard at
   `http://<controller-ip>:21434`, or check the worker log on the controller:
   `GET /logs?node=steamdeck` (per [../ACCELERATION.md](../ACCELERATION.md) verify step).
3. **Gen test:** with the Deck placed in a pipeline stage, run a short generation against the
   controller API and confirm coherent output (e.g. the fleet's standard "capital of France
   is Paris" smoke test). Watch `~/worker.log` for errors.

## 6. Gotchas

- **Immutable root fs.** SteamOS's root is read-only — keep the venv, repo, and unit file in
  `$HOME`. Do not rely on system `pacman` packages; a SteamOS update can wipe anything
  written outside `$HOME` (and `pacman` modifications to the base image).
- **CPU kernels don't accelerate.** Per [../ACCELERATION.md](../ACCELERATION.md): the Triton
  fused-MoE and split-K dense kernels are **GPU-only**; `INFINITEMODEL_CUDA_FUSED_MOE` has no
  effect on a CPU worker. Don't set GPU env switches here.
- **Long-prompt prefill memory.** The Deck has little RAM headroom; a long prefill can blow
  memory. `INFINITEMODEL_PREFILL_CHUNK` (default 2048) caps it — lower to 512 on this box if
  needed ([../ACCELERATION.md](../ACCELERATION.md)).
- **One worker per box.** Run exactly one `client.py` per hostname — two workers sharing a
  hostname fight over the controller registration ([../ACCELERATION.md](../ACCELERATION.md)).
- **Capacity, not speed.** Expect single-digit tok/s at best for its stage; that's expected
  for a CPU node and is by design.

### ROCm on the Deck — research only (verify)

Using the gfx1033 iGPU is **experimental and unverified**. Treat every step below as
**(verify)** — none of it is validated on the Deck:

- **gfx1033 is not a TheRock-supported target (verify).** The validated RDNA ROCm recipe in
  [../ROCM.md](../ROCM.md) targets **gfx1151** (Strix Halo, RDNA3.5) via AMD's arch-specific
  "TheRock" wheel index (`https://rocm.nightlies.amd.com/v2/gfx1151/`). The Deck's **gfx1033**
  (Van Gogh, RDNA2) is **not** published as a TheRock arch index there — discover available
  arches under `https://rocm.nightlies.amd.com/v2/`, but do not assume gfx1033 is present
  (verify). Forcing `HSA_OVERRIDE_GFX_VERSION` is explicitly discouraged in
  [../ROCM.md](../ROCM.md) (it pushes foreign code objects onto the GPU and tends to fail with
  "no kernel image").
- **Immutable fs complicates the runtime (verify).** [../ROCM.md](../ROCM.md) relies on
  pip-installed userspace (TheRock `rocm-sdk-*` packages, all inside the venv) plus an
  in-tree amdgpu kernel driver and `render,video` group membership (`sudo usermod -aG
  render,video "$USER"`). On SteamOS's immutable root, both the group change and any system
  driver bits need a writable overlay or a **distrobox** container — none of this is set up or
  tested on the Deck (verify).
- **Triton needs a C toolchain (verify).** The RDNA int4 Triton kernel JIT-compiles launcher
  stubs and requires `gcc` + Python headers ([../ROCM.md](../ROCM.md)). SteamOS does not ship
  these on the immutable root; they'd have to come via distrobox (verify).
- **Even if it ran, payoff is small (verify).** 8 CU RDNA2 with a tiny UMA carve-out is far
  weaker than the gfx1151 results in [../ROCM.md](../ROCM.md)/[../ACCELERATION.md](../ACCELERATION.md).
  Until validated, the Deck stays a **CPU worker** (Section 3).
