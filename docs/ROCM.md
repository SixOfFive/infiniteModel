# Running InfiniteModel on AMD GPUs (ROCm)

InfiniteModel runs on AMD GPUs with **no code changes** to the inference path: PyTorch's
ROCm build presents AMD GPUs through the CUDA API (HIP masquerade), so every
`torch.cuda.*` call, the `cuda:N` device strings, the pipeline transport, the int4
quantizer, and the load/generate flow all work exactly as on NVIDIA. A ROCm node
registers, gets pipeline stages, and serves tokens identically to a CUDA node — it is
**1:1**. This doc captures the one thing that is *not* automatic: getting a working
PyTorch+ROCm runtime onto the box.

Validated end-to-end on **AMD Strix Halo** (Ryzen AI Max, `gfx1151` / "Radeon 8060S",
RDNA 3.5 iGPU, 128 GB unified memory), Ubuntu 26.04, Python 3.14:
`qwen2.5:0.5b` int4 loaded onto the iGPU and generated *"The capital of France is Paris."*
The same recipe applies to any ROCm-supported AMD GPU — substitute the GPU's arch (see
[Choosing the wheel](#choosing-the-wheel)).

---

## TL;DR

```bash
# 1) GPU device access (one-time, needs sudo) — add yourself to the render+video groups,
#    then reconnect so the new groups apply.
sudo usermod -aG render,video "$USER"

# 2) Python venv (3.13/3.14). On Debian/Ubuntu you may need the venv package first:
#    sudo apt-get install -y python3-venv git
python3 -m venv ~/imenv

# 3) PyTorch for your AMD GPU. For Strix Halo (gfx1151) use AMD's arch-specific
#    "TheRock" wheels — they bundle a matched ROCm runtime as pip packages:
~/imenv/bin/python -m pip install --pre torch \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# 4) App deps (same versions as the CUDA fleet):
~/imenv/bin/python -m pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn

# 5) Sanity check: must print "cuda.is_available: True" and run a matmul with no crash.
~/imenv/bin/python -c "import torch; \
print('hip', torch.version.hip, '| avail', torch.cuda.is_available(), \
'|', torch.cuda.get_device_name(0)); \
a=torch.randn(2048,2048,device='cuda'); (a@a).sum().item(); print('matmul OK')"
```

Then launch the controller + worker (see [Launch](#launch)). Or use the helper:
`./install-rocm.sh gfx1151` (see the script header for options).

---

## Choosing the wheel

The **generic** PyTorch ROCm wheels from `download.pytorch.org/whl/rocm7.0` *list*
`gfx1151` in `torch.cuda.get_arch_list()`, but on a bleeding-edge kernel (e.g. Ubuntu
26.04, kernel 7.0) the **ROCm runtime they bundle (`libhsa-runtime64.so`) segfaults on
the first kernel dispatch** (null deref while loading the code object). The fix is to use
a runtime that matches both the kernel and the exact GPU arch. AMD's "TheRock" project
publishes **arch-specific** wheel indexes that package the ROCm runtime itself as pip
packages (`rocm-sdk-core`, `rocm-sdk-libraries-<arch>`), so torch + a matched runtime all
land inside the venv — nothing is installed system-wide:

| GPU family | arch | index URL |
|---|---|---|
| Strix Halo (Ryzen AI Max) | `gfx1151` | `https://rocm.nightlies.amd.com/v2/gfx1151/` |
| RX 7000 / W7000 (RDNA3) | `gfx110x` | `https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/` |
| MI200/MI300 (CDNA2/3) | — | the generic `download.pytorch.org/whl/rocm7.x` wheels are fine |

Discover available arch indexes under `https://rocm.nightlies.amd.com/v2/`.

> **Rule of thumb:** if `cuda.is_available()` is `True` but the first GPU op crashes with
> `hipErrorInvalidDeviceFunction` ("invalid device function") or
> `hipErrorNoBinaryForGpu` ("no kernel image"), the wheel lacks kernels for your arch —
> switch to the arch-specific TheRock index. If `cuda.is_available()` is `True`, a matmul
> succeeds, but you get a **segfault in `libhsa-runtime64.so`**, the *runtime* is
> mismatched to your kernel — also switch to TheRock (its runtime is built to match).

There is **no need** for `HSA_OVERRIDE_GFX_VERSION` with the right wheel; the override
forces foreign code objects onto your GPU and tends to fail with "no kernel image" on
RDNA3.5.

---

## Launch

A ROCm box can run a **standalone** controller + worker (client → its own controller):

```bash
cd ~/infinitemodel

# Controller — binds 0.0.0.0:21434 (dashboard/Ollama/OpenAI/Anthropic APIs),
# control 50100, data 50101. IM_ALLOW_NO_MODELS=1 lets a fresh box start with an
# empty models/ dir (pull models on demand).
IM_ALLOW_NO_MODELS=1 setsid ~/imenv/bin/python server.py \
    >~/controller.log 2>&1 </dev/null &

# Worker — GPU + CPU spill, pointed at the LOCAL controller. --controller overrides
# config.json's controller_host (don't edit config.json: it's in EXTRA_UPDATE_FILES and
# a self-update would revert it).
cd ~/infinitemodel
setsid ~/imenv/bin/python client.py --controller 127.0.0.1 --device cpu+gpu \
    >~/worker.log 2>&1 </dev/null &
```

To join an existing fleet instead, drop `IM_ALLOW_NO_MODELS`/the local controller and
launch only the worker with `--controller <controller-ip>`.

For persistence across reboot, run both under `systemctl --user` units (enable
`loginctl enable-linger "$USER"` once so user services start at boot). No special
environment variables are required for ROCm.

---

## What's the same, what differs

**Same (1:1 with CUDA):** device detection (`cuda:0`), VRAM sizing (`mem_get_info` — on a
unified-memory APU this reports the GPU-accessible pool, e.g. ~60 GB of 128 GB), weight
streaming, placement, pipeline transport, generation, the `/load` `/api/*` `/v1/*`
surface, and the dashboard.

**Differs (handled, no action needed):**

- **Fused int4 GEMM** (`torch._weight_int4pack_mm`) is implemented only for **CDNA2+**
  (MI200/MI300) on ROCm, not RDNA (gfx11xx). On RDNA the worker's `prepare_fused()`
  self-check sees the `... only supported on AMD gpu arch >= CDNA2` error and silently
  falls back to the **naive int4 dequant** path — correct results, same behavior as a
  pre-sm80 CUDA card, just without the fused-kernel decode speedup. int4 still works.
- **GPU utilization %** — `torch.cuda.utilization()` needs the `amdsmi` python binding,
  which the TheRock wheels don't ship. The worker heartbeat falls back to the bundled
  `rocm-smi` CLI (`_rocm_gpu_util()` in `client.py`, ROCm-guarded), so the dashboard's
  GPU% stays 1:1.
- **`/gpudiag`** per-process GPU list shells out to `nvidia-smi` (controller-side); on an
  AMD controller it simply returns nothing. VRAM totals/used still report via torch. (A
  `rocm-smi` per-process fallback is a possible future nicety.)

## Notes / gotchas

- `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` printed by ROCr is
  **cosmetic** (device-name lookup table); ignore it.
- Add the user to **`render`** (for `/dev/kfd` + `/dev/dri/renderD*`) and **`video`**;
  group changes apply on the next login/SSH session, not the current one.
- The amdgpu **kernel driver is in-tree** on modern kernels — you do *not* need a
  system ROCm/`amdgpu-dkms` install; the TheRock pip packages provide all userspace.
