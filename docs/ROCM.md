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

# 2) Python venv + build tools. The Triton int4 kernel JIT-compiles launcher stubs, so it
#    needs a C compiler + Python headers. On Debian/Ubuntu:
#    sudo apt-get install -y python3-venv python3-dev gcc git
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

## Fast int4 on RDNA — the Triton w4a16 kernel

torch's fused int4 GEMM (`torch._weight_int4pack_mm`) is **CDNA2+-only** on ROCm
(MI200/MI300). On RDNA (gfx11xx) it's unavailable, so the naive int4 path
**rematerializes the whole bf16 weight every token** — correct, but GPU-compute-bound and
slow (a 35B-A3B managed ~2 tok/s on Strix Halo). InfiniteModel ships a **Triton w4a16
kernel** that reads the packed int4 weight and dequantizes *inside* the GEMM, in the
worker's exact group-wise asymmetric format (byte j → col 2j low / 2j+1 high nibble;
`w=(q−zero)*scale` per 128-group). It is **bit-identical** to the naive path
(self-checked, rel<0.05, else it falls back) and **ROCm-only** — NVIDIA and CPU keep the
torch tinygemm / naive paths untouched.

Two integration points in `client.py`:

- **Dense linears** (`QuantLinear4.prepare_fused`) — a Triton backend (`_w4a16_triton_op`)
  taken on `cuda + torch.version.hip`. **5–20× faster decode** (validated by
  `bench_w4a16.py`: 14× at 4096², 20× at 5120²).
- **MoE routed experts** (`Packed4Tensor3D.__getitem__`) — returns a `torch.Tensor`
  subclass (`_W4A16Weight`) that intercepts `F.linear` via `__torch_function__` and routes
  to the kernel (and materializes to bf16 for any other op). The `qwen3_5_moe` host calls
  `F.linear(state, gate_up_proj[e])`, so this fuses the per-expert GEMMs that dominate an
  A3B model's decode.

**Measured — AMD Strix Halo gfx1151, `qwen3.6-35b-a3b` int4, all-GPU:**

| path | decode tok/s | vs naive |
|---|---|---|
| naive int4 (no kernel) | 2.08 | 1.0× |
| + Triton on dense linears | 3.5 | 1.7× |
| + per-expert Triton on MoE | 5.42 | 2.6× |
| + **fused grouped** MoE kernel (`_w4a16_moe_op`) | 10.8 | 5.2× |
| + **split-K** dense GEMV (M=1 decode) | **15.4** | **7.4×** |

The last two rows are the newer decode path: a **single grouped Triton launch** over all `top_k`
experts (replacing the per-expert subclass loop), and a **split-K** GEMV that gives the batch-1 dense
matmul enough programs to saturate the iGPU's bus. Both are ROCm-automatic + self-checked; the MoE
kernel is also `@triton.autotune`d per shape. Full cross-platform detail, the NVIDIA opt-in, and the
remaining optimization headroom (HIP-graph capture, AOTriton flash-attn; note split-K — which won on the
*dense* GEMV — was tested and **rejected** for the MoE GEMV, it's already occupied) live in
**[ACCELERATION.md](ACCELERATION.md)**.

Dense models gain the most — every linear is fused, not just the ~40% dense slice of an
MoE — so a dense 7–14B int4 decodes many× faster than its naive path. On an APU, **bf16**
is also a fine choice when the model fits in the unified pool (no dequant tax: a plain
rocBLAS GEMM, no Triton needed).

Verify/benchmark on any AMD box: `~/imenv/bin/python bench_w4a16.py`.

> **Requirement:** Triton JIT-compiles launcher stubs, so the box needs a host **C
> compiler + Python headers** (`sudo apt-get install -y gcc python3-dev`). Without them
> you'll see `RuntimeError: Failed to find C compiler` and int4 silently falls back to the
> (correct but slow) naive path.

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

- **int4 GEMM** — torch's fused int4 op is CDNA2+-only, so on RDNA InfiniteModel uses its
  own **Triton w4a16 kernel** for both dense linears and MoE experts (bit-identical,
  self-checked, ROCm-only). See [Fast int4 on RDNA](#fast-int4-on-rdna--the-triton-w4a16-kernel).
  Needs a host C compiler (`gcc` + `python3-dev`) for Triton's JIT; falls back to the
  correct-but-slow naive int4 path if absent.
- **GPU utilization %** — `torch.cuda.utilization()` needs the `amdsmi` python binding,
  which the TheRock wheels don't ship. The worker heartbeat falls back to the bundled
  `rocm-smi` CLI (`_rocm_gpu_util()` in `client.py`, ROCm-guarded), so the dashboard's
  GPU% stays 1:1.
- **`/gpudiag`** per-process GPU list shells out to `nvidia-smi` (controller-side); on an
  AMD controller it simply returns nothing. VRAM totals/used still report via torch. (A
  `rocm-smi` per-process fallback is a possible future nicety.)

## Notes / gotchas

- **Allocator pool on unified memory (recommended: `expandable_segments`).** torch's caching
  allocator holds freed VRAM inside the process, and on an APU the device-level "used" counter
  (GTT) counts that vacant pool — so an idle worker can *look* like it's holding tens of GB.
  Two layers handle this: workers report their reusable pool over the heartbeat and the
  controller's planner **credits it back** automatically (no phantom memory pressure); and
  setting `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` in the worker's environment makes
  freed segments actually return to the OS — measured **zero decode cost** on gfx1151. Set it
  in the worker unit's environment and restart the worker.
- `/load?kv_offload=1` is **force-disabled on ROCm** workers (it garbles output on gfx1151);
  the load proceeds with KV on-device.
- **Text-to-image on gfx1151:** int4-quantizing the DiT is *slower* than bf16 here (the dequant
  tax outweighs the bandwidth savings at diffusion's large batch shapes) — int4 is the
  *capacity* recipe for fitting the DiT, not a speed win. Use bf16 (or the offload mode) when
  it fits. See [T2I.md](T2I.md).
- **Text-to-music (ACE-Step / t2a) is NOT supported on ROCm — by design.** A ROCm worker is left
  `can_t2a=False` on purpose so the planner never routes music to it. Three walls: ACE-Step
  hard-depends on `torchaudio`, and the ROCm/TheRock builds ship **no `torchaudio` matching that
  torch ABI** (installing one from PyPI/CUDA breaks the venv, LLM stack included); its DiT + DCAE
  kernels JIT through **MIOpen**, which is flaky-to-broken on gfx1151 (the same wall that pushes
  **Kokoro TTS** to CPU here); and there is **no CPU fallback** — ACE-Step's diffusion does not run
  usably on CPU. Serve music from a **CUDA** GPU instead, locally or anywhere in the pool via
  `#media-anywhere`. Full rationale in [T2A.md](T2A.md).
- **CUDA-graph decode is broken on TheRock rocm7.13** — capture/replay run but replay computes
  wrong logits; the first-decode self-check auto-disables it (serving stays eager and correct).
  Leave `INFINITEMODEL_CUDA_GRAPH` unset on ROCm; details in
  [ACCELERATION.md](ACCELERATION.md).
- `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` printed by ROCr is
  **cosmetic** (device-name lookup table); ignore it.
- Add the user to **`render`** (for `/dev/kfd` + `/dev/dri/renderD*`) and **`video`**;
  group changes apply on the next login/SSH session, not the current one.
- The amdgpu **kernel driver is in-tree** on modern kernels — you do *not* need a
  system ROCm/`amdgpu-dkms` install; the TheRock pip packages provide all userspace.
- **Long-prompt prefill OOM (gfx1151):** a long prompt to a standard-attention model can
  fail with a huge single CUDA allocation (e.g. *"tried to allocate 43.06 GiB"*) inside
  `sdpa_attention_forward`. Cause: the shard passes HF layers an explicit additive float
  mask, which disables SDPA's flash backend; gfx1151 has no mem-efficient backend either,
  so SDPA falls back to the **math** backend and materializes the full `[1, H, q, total]`
  score tensor (`O(H·q²)`). The fix ships on by default — **chunked prefill**
  (`INFINITEMODEL_PREFILL_CHUNK`, default 2048) caps peak score memory to `O(H·C·q)`; see
  [ACCELERATION.md](ACCELERATION.md). For very long contexts on a memory-tight box, lower it
  (e.g. `512`); `0` disables it.
