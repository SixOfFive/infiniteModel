# Node: om3nbox — AMD Strix Halo (gfx1151)

> **This is an operational pointer, not the full recipe.** The complete, step-by-step
> ROCm install + launch + kernel detail lives in **[../ROCM.md](../ROCM.md)**, and the
> cross-platform int4-decode kernel matrix is in **[../ACCELERATION.md](../ACCELERATION.md)**.
> This page covers only what is specific to *this box*; for anything not below, follow ROCM.md.

## 1. Overview

| | |
|---|---|
| Host | **om3nbox** (hostname `InferenceEngine`) |
| SoC | Ryzen AI Max+ 395 — **`gfx1151`** Strix Halo APU (RDNA 3.5 iGPU, "Radeon 8060S") |
| Memory | ~60 GB **unified** GPU-accessible pool (`mem_get_info` reports the pool, not all 128 GB system RAM) |
| OS | Ubuntu |
| Device string | `cuda:0` (HIP masquerade — ROCm presents the iGPU through the CUDA API, see [../ROCM.md](../ROCM.md)) |
| Role | **Standalone controller + worker on one box** — it runs its own controller at `:21434` and a local worker pointed at `127.0.0.1`. It is *not* a BEAST fleet node. |
| Reachability | Tailscale `100.94.43.14:21434` |

Restart is **coupled**: cycling the box (or the `/restart` path) cycles **both** the
controller and the worker together — there is no way to bounce one without the other here.

## 2. Install

Follow **[../ROCM.md § TL;DR](../ROCM.md#tldr)** verbatim. The load-bearing point for this
arch: **use AMD's "TheRock" `gfx1151` wheel index, not the generic ROCm wheel** — the
generic `download.pytorch.org/whl/rocm7.0` runtime **segfaults in `libhsa-runtime64.so`**
on the first kernel dispatch on this kernel. From ROCM.md:

```bash
# GPU access (one-time, then reconnect so groups apply)
sudo usermod -aG render,video "$USER"

# venv + Triton JIT build deps (the int4 kernel needs a C compiler + Python headers)
sudo apt-get install -y python3-venv python3-dev gcc git
python3 -m venv ~/imenv

# PyTorch — arch-specific TheRock wheels for gfx1151 (bundles a matched ROCm runtime)
~/imenv/bin/python -m pip install --pre torch \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# App deps (same pins as the CUDA fleet)
~/imenv/bin/python -m pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

There is a helper: `./install-rocm.sh gfx1151` (see ROCM.md / the script header).
You do **not** need a system ROCm or `amdgpu-dkms` install — the amdgpu kernel driver is
in-tree and TheRock pip packages provide all userspace. Do **not** set
`HSA_OVERRIDE_GFX_VERSION` (it forces foreign code objects and fails with "no kernel image"
on RDNA3.5).

## 3. Run the controller + worker

This box runs **both** roles (standalone). The grounded launch from
[../ROCM.md § Launch](../ROCM.md#launch):

```bash
cd ~/infinitemodel

# Controller — binds 0.0.0.0:21434 (dashboard + Ollama/OpenAI/Anthropic APIs),
# control 50100, data 50101. IM_ALLOW_NO_MODELS=1 lets a fresh box start empty.
IM_ALLOW_NO_MODELS=1 setsid ~/imenv/bin/python server.py \
    >~/controller.log 2>&1 </dev/null &

# Worker — GPU + CPU spill, pointed at the LOCAL controller.
cd ~/infinitemodel
setsid ~/imenv/bin/python client.py --controller 127.0.0.1 --device cpu+gpu \
    >~/worker.log 2>&1 </dev/null &
```

`--controller 127.0.0.1` overrides `config.json`'s `controller_host` — **don't edit
config.json** (it's in `EXTRA_UPDATE_FILES`; a self-update would revert the edit).

**Persistence (used here):** `systemctl --user` units — **`im-controller`** and
**`im-worker`**. Enable lingering once so user services start at boot:

```bash
loginctl enable-linger "$USER"
systemctl --user enable --now im-controller im-worker
```

No special environment variables are required for ROCm. The fast int4 path on RDNA is
**automatic** — no `INFINITEMODEL_CUDA_FUSED_MOE` needed (that switch is the NVIDIA-Linux
opt-in; on ROCm the fused MoE kernel is already on; see [../ACCELERATION.md](../ACCELERATION.md)).

**Restart (coupled):** because controller + worker are one unit on this box, restart both:

```bash
systemctl --user restart im-controller im-worker
```

A `POST /restart` against the controller likewise cycles the coupled pair. (verify the
exact `/restart` query params against the controller you are hitting — ROCM.md does not
document them for this box.)

## 4. Optimal settings (this VRAM tier)

- **Quant:** **int4** is the default and the only fast path on RDNA — the fused-int4 GEMM
  is CDNA2+-only, so InfiniteModel substitutes its own **Triton w4a16 kernel** (dense +
  MoE experts) **+ split-K** for the batch-1 decode GEMV (split-K is RDNA-specific). All
  automatic, self-checked, bit-identical or it falls back. **bf16** is also fine when the
  model comfortably fits the ~60 GB unified pool — on an APU there's no dequant tax (plain
  rocBLAS GEMM, no Triton).
- **Placement:** single APU = one shared memory pool, so use **all-GPU** placement when the
  model fits; CPU spill (`cpu+gpu`) only when it doesn't. There is **NO tensor parallelism**
  here — TP needs multiple memory pools/devices, and a single APU has one. Do not try to TP
  across it.
- **Model choice:** decode is **memory-bandwidth-bound** (~150 GB/s realized, ~80% GPU-busy
  → tok/s tracks effective GB/s, not FLOPs). Prefer a **sparse MoE with a low active-param
  ratio** over a big dense model of similar quality. Per [../ACCELERATION.md](../ACCELERATION.md),
  `qwen3.6-35b-a3b` int4 (all-GPU) reaches **~15.4 tok/s** with the full kernel stack
  (split-K dense + fused grouped MoE).
- **What to avoid:** big **dense** models — a dense 70B int4 *fits* 60 GB but decodes at
  ~1.5–3 tok/s (every param read per token); use spec-decode (dense draft) if you must run
  one. **`gpt-oss` MXFP4** is unsupported (needs a dequant path). For long prefills on this
  memory-tight box, lower `INFINITEMODEL_PREFILL_CHUNK` (e.g. 512–1024) — see Gotchas.

## 5. Verify

```bash
# (a) torch sees the iGPU + a matmul runs without segfault (the TheRock-vs-generic test):
~/imenv/bin/python -c "import torch; \
print('hip', torch.version.hip, '| avail', torch.cuda.is_available(), \
'|', torch.cuda.get_device_name(0)); \
a=torch.randn(2048,2048,device='cuda'); (a@a).sum().item(); print('matmul OK')"
# must print 'cuda.is_available: True' and 'matmul OK'

# (b) int4 kernel sanity / benchmark (validates the Triton w4a16 path):
~/imenv/bin/python bench_w4a16.py

# (c) controller + worker up and registered:
curl -s http://127.0.0.1:21434/status        # worker should be registered
curl -s http://127.0.0.1:21434/logs?node=InferenceEngine   # worker log via heartbeat

# (d) end-to-end generation (substitute a model you've pulled):
curl -s http://127.0.0.1:21434/api/generate \
  -d '{"model":"qwen2.5:0.5b","prompt":"The capital of France is","stream":false}'
```

Reference validation (ROCM.md): `qwen2.5:0.5b` int4 on this iGPU generated *"The capital of
France is Paris."* For int4 engagement, watch the worker log for the kernel taking the
Triton path rather than the naive fallback.

## 6. Gotchas (device-specific)

- **TheRock wheels, not generic ROCm.** `cuda.is_available()` `True` + a **segfault in
  `libhsa-runtime64.so`** on first GPU op = generic-wheel runtime mismatched to the kernel
  → switch to the `gfx1151` TheRock index. `hipErrorInvalidDeviceFunction` /
  `hipErrorNoBinaryForGpu` = wheel lacks kernels for the arch → same fix. ([../ROCM.md](../ROCM.md))
- **Triton needs a C compiler.** No `gcc` + `python3-dev` → `RuntimeError: Failed to find
  C compiler` and int4 **silently falls back** to the correct-but-slow naive path
  (~13–18× slower). Install the build deps.
- **The decode kernel gap vs llama.cpp.** Even with the full kernel stack, int4 decode on
  this iGPU is **~13–18× slower than llama.cpp's Vulkan backend on the *same* gfx1151
  silicon** — a kernel-maturity gap, not a hardware ceiling. Set throughput expectations
  accordingly (e.g. 35B-A3B ~5.4 tok/s per-expert / ~15.4 tok/s full stack here vs
  70–100 tok/s under llama.cpp Vulkan).
- **NO tensor parallelism on a single APU.** One shared unified-memory pool → TP has nothing
  to split across. Use pipeline / all-GPU placement only.
- **Long-prompt prefill OOM.** A long prompt to a standard-attention model can fail with a
  huge single CUDA alloc (e.g. *"tried to allocate 43.06 GiB"*) inside
  `sdpa_attention_forward` — the explicit additive mask disables SDPA's flash backend and
  gfx1151 has no mem-efficient backend, so it materializes the full `[1,H,q,total]` score
  tensor. Fix ships on by default: **chunked prefill** `INFINITEMODEL_PREFILL_CHUNK`
  (default 2048); lower to `512` for very long contexts on this memory-tight box, `0`
  disables. ([../ACCELERATION.md](../ACCELERATION.md))
- **Coupled restart.** Controller + worker cycle together here — there is no single-role
  bounce. Rapid restarts can leave a transient **first-load wedge at 0/N** on the first
  model load; a clean restart + warmup clears it (this is *not* a NIC/model bug).
- **Cosmetic ROCr warning.** `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or
  directory` is just the device-name lookup table — ignore it.
- **GPU% / `/gpudiag`.** `torch.cuda.utilization()` needs `amdsmi` (not shipped by TheRock);
  the worker falls back to the bundled `rocm-smi` so dashboard GPU% stays 1:1. Controller
  `/gpudiag` per-process listing shells out to `nvidia-smi` and returns nothing on this AMD
  box (VRAM totals/used still report via torch).
