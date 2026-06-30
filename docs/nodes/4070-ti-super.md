# Node: RTX 4070 Ti SUPER — `beast` (controller + GPU worker)

The fleet's primary box. It runs **both** the InfiniteModel **controller** (`:21434`,
serving the Ollama + OpenAI + Anthropic APIs from `D:\infinitemodel`) **and** a GPU
worker on the same machine.

See also: [../ACCELERATION.md](../ACCELERATION.md) (int4 decode kernel matrix, the
Windows + NVIDIA MoE setup, prefill chunking) and [../ROCM.md](../ROCM.md) (AMD recipe —
not this box, but the cross-platform reference).

---

## 1. Overview

| | |
|---|---|
| **Host** | `beast` |
| **GPU** | RTX 4070 Ti SUPER — Ada Lovelace, **sm_89** |
| **VRAM** | 16 GB |
| **Mem bandwidth** | ~672 GB/s (per ACCELERATION.md sweep) |
| **OS** | Windows 11 |
| **Role** | **Controller** (`0.0.0.0:21434`, dashboard + Ollama/OpenAI/Anthropic APIs) **and** a GPU worker — both on this box |
| **Install dir** | `D:\infinitemodel` |

Because this box co-hosts the controller, treat its GPU as a *contributing* worker, not a
solo host — see Gotchas.

---

## 2. Install

**Python:** CPython 3.14 is the validated interpreter for the Windows MoE toolchain in
ACCELERATION.md (the `triton-windows` 3.7.1 wheel was validated against 3.14). Use a
3.14 user environment.

**torch — Ada / sm_89 → standard CUDA wheel.** This is an NVIDIA card, so use the normal
upstream CUDA build of PyTorch (the TheRock/ROCm indexes in ROCM.md do **not** apply here).
Install the CUDA 12.8 build to line up with the CUDA Toolkit used for the MoE opt-in:

```bat
pip install --user torch --index-url https://download.pytorch.org/whl/cu128
```

> (verify) the exact torch version pin — ACCELERATION.md/ROCM.md don't pin a torch
> version for NVIDIA-Windows; cu128 is chosen to match the CUDA Toolkit 12.8 used below.

**App deps** — same versions the fleet uses (from ROCM.md's dep list):

```bat
pip install --user transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

**Optional — advanced MoE tier (fused Triton expert kernel).** Only needed if you want the
opt-in fused routed-expert kernel (see §4). One-time toolchain, per ACCELERATION.md
"Windows + NVIDIA":

1. **Visual Studio Build Tools** with the "Desktop development with C++" workload (provides
   `cl.exe`). Note its full path, e.g.
   `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe`.
2. **CUDA Toolkit 12.8 (toolkit only, no driver)** — provides `ptxas` + `cuda.lib`, which
   the pip `nvidia-cuda-*` wheels do not ship on Windows:
   ```bat
   cuda_12.8.x_windows_network.exe -s nvcc_12.8 cudart_12.8 cuobjdump_12.8 nvdisasm_12.8
   ```
   Lands at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`.
3. **triton-windows** (the woct0rdho fork; pick the wheel matching your Python):
   ```bat
   pip install --user triton-windows
   ```

Ada (sm_89) is Ampere-or-newer, so the bf16 fused kernel compiles here — the MoE opt-in is
genuinely usable on this card (validated in ACCELERATION.md).

---

## 3. Run the worker

This box runs the **controller** and a **worker**. The controller is launched from
`D:\infinitemodel` (`server.py`); the worker is launched via `start_worker.bat`.

**Worker launch** — `D:\infinitemodel\start_worker.bat` (form from ACCELERATION.md). The
controller is local, so point the worker at this box. Include the MoE toolchain env vars
only if you set up §2's optional tier:

```bat
set "CC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "INFINITEMODEL_CUDA_FUSED_MOE=1"
cd /d D:\infinitemodel
python client.py --controller <controller-ip> --control-port 50100
```

- `--controller` — the controller IP. On this box the controller is local; use the box's
  own LAN IP (or `127.0.0.1`).
- `--control-port 50100` — control plane port (matches ACCELERATION.md's example).
- `--device` — (verify) ACCELERATION.md's Windows example omits `--device`; the ROCm doc
  uses `--device cpu+gpu` for GPU + CPU spill. Use `--device cpu+gpu` to let the box's GPU
  plus CPU RAM both contribute, or `--device gpu` to keep it GPU-only.

If you are **not** using the fused-MoE opt-in, drop the `CC` / `CUDA_PATH` /
`INFINITEMODEL_CUDA_FUSED_MOE` lines — int4 **dense** decode uses torch tinygemm and is
fast with no extra toolchain. (Note: per ACCELERATION.md `INFINITEMODEL_CUDA_FUSED_MOE` is
documented as a *Linux*-only opt-in, but ACCELERATION.md also validates the fused kernel
building and going `-> ACTIVE` on this exact card under **Windows** — so it is effective
here.)

**Persistence — interactive session only.** This is a consumer GeForce card under the
**WDDM** driver, which is **not visible from session 0**. Do **not** install the worker as
a Windows *service* or a "run whether logged on or not" scheduled task — those run in
session 0 and the GPU will be invisible. Launch `start_worker.bat` in the **logged-on
interactive desktop session**:

- run the batch file from a normal `cmd`/desktop shortcut, **or**
- drop a shortcut to it in `shell:startup` so it starts at logon.

Run **exactly one** worker per box (two workers sharing a hostname fight over controller
registration).

---

## 4. Optimal settings

**Quant: int4.** On NVIDIA, int4 dense decode uses torch's fused tinygemm
(`_weight_int4pack_mm`) and actually **beats bf16** (int4 7B 3.25 → 21.13 tok/s in the
fleet bench). With 16 GB, int4 is also what lets useful models fit — prefer it.

**Context:** prefill chunking is on by default (`INFINITEMODEL_PREFILL_CHUNK=2048`) and is
math-identical; leave it. Lower it (512–1024) only for very long contexts on a
memory-tight load. Decode is unaffected.

**Placement / what fits (16 GB):** this is a mid VRAM tier. With the controller co-resident
on the box, budget conservatively — InfiniteModel places stages against *physically free*
VRAM and reserves each layer's full-ctx KV. Prefer **sparse MoE with a low active-param
ratio** (e.g. ~3B active) over a big dense model: decode tok/s ≈ bandwidth ÷ active
bytes-per-token, so a 30–35B-A3B MoE is far faster than a dense model of similar footprint.
For anything larger than fits in 16 GB, **distribute it across the fleet** rather than
forcing it onto this card (see Gotchas).

**The fused-MoE opt-in (advanced tier):** worth it for MoE models, where routed experts are
a big share of decode — ACCELERATION.md measured the expert-GEMM microbench at **24–34×**
over the bf16-remat default on this card, and the self-check went `-> ACTIVE` on all layers
with coherent output. On **dense** models the gain is small (tinygemm already dominates
decode). Autotune note: `num_stages=3` buys ~1.18× on narrow-N shapes (qwen3-a3b);
the kernel sits at ~35–48% of the card's ~672 GB/s peak. So enable it for MoE workloads;
it's optional polish for dense.

**Avoid:**
- Full-loading a model on this box (see Gotchas — it can crash the controller).
- Treating 16 GB as a solo host for big models — pipeline them across the fleet instead.
- Split-K for the MoE GEMV — tested and **rejected** on this card (0.66–0.85×, slower).

---

## 5. Verify

1. **torch sees the GPU:**
   ```bat
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Expect `True  NVIDIA GeForce RTX 4070 Ti SUPER`. (verify — exact device-name string.)

2. **Worker registered:** the controller dashboard at `http://<beast>:21434` should list
   this node, or check the controller ring log:
   ```
   curl http://<beast>:21434/logs?node=beast
   ```

3. **Generation test:** load a small int4 model and generate via the Ollama-compatible API
   on the controller, e.g.:
   ```bat
   curl http://<beast>:21434/api/generate -d "{\"model\":\"<model>\",\"prompt\":\"The capital of France is\"}"
   ```
   A coherent completion confirms the path end-to-end.

4. **MoE opt-in engaged (if enabled):** load a MoE model and look for
   `[int4] fused-MoE self-check ... -> ACTIVE` in the worker log
   (`GET /logs?node=beast`). `-> fallback` means it self-checked out or the toolchain is
   incomplete, and the safe bf16-remat path is in use (still correct, just not accelerated).

---

## 6. Gotchas

- **NEVER full-load a model on this box.** It co-hosts the controller; a 67 GB
  `from_pretrained` crashed the controller once. Big-model compute must be **distributed**
  across the fleet — let the controller place stages, don't pull a whole model onto this
  card.
- **WDDM / session-0 invisibility.** The consumer GPU is invisible to a Windows service or
  "run whether logged on or not" scheduled task. Run the worker in the **logged-on
  interactive session** only (`start_worker.bat` via desktop/`shell:startup`). (Datacenter
  cards in TCC mode don't have this limit — this one isn't one.)
- **One worker per box.** Two workers sharing the `beast` hostname fight over controller
  registration.
- **Perf-only kernel changes carry no VERSION bump.** A self-update *stages* the new code
  but does **not** auto-restart the worker — **relaunch `start_worker.bat`** to pick up a
  kernel change.
- **MoE toolchain must be complete.** Missing `cl.exe`, `ptxas` (CUDA Toolkit 12.8), or
  `triton-windows` → the fused kernel self-checks out and silently falls back to bf16-remat
  (correct but unaccelerated). Verify with the `-> ACTIVE` log line above.
- **`INFINITEMODEL_NO_FUSED_MOE=1`** is the kill switch — forces the bf16-remat expert path
  everywhere; use it to A/B the fused kernel.
- **`INFINITEMODEL_CUDA_GRAPH`** — ⚠ experimental, leave **unset** (per ACCELERATION.md the
  in-worker integration is not yet validated/safe).
- **Prefill OOM is not a concern here** the way it is on the AMD iGPU — NVIDIA has the
  mem-efficient SDPA backend; chunking is on by default regardless.

---

_For anything not covered here, defer to [../ACCELERATION.md](../ACCELERATION.md) (kernels,
env switches, the Windows + NVIDIA setup) and [../ROCM.md](../ROCM.md) (AMD reference)._
