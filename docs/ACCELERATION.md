# int4 decode acceleration — by platform

InfiniteModel quantizes weights to **int4** (group-wise asymmetric) at load time. How the *decode*
GEMMs (the per-token, batch-1 matmuls that dominate generation speed) are executed depends on the
worker's GPU **and OS**, because the fast paths come from different places on each. This page is the
map; for AMD setup specifics see **[ROCM.md](ROCM.md)**.

There are two hot matmul classes per layer:

- **Dense linears** — attention `q/k/v/o`, the shared expert, etc. (`QuantLinear4`, a 2D weight).
- **Routed MoE experts** — the per-token top-k experts of a sparse-MoE block (`Packed4Tensor3D`, a
  fused 3D `[E, out, in]` weight). ~90% of a big MoE's parameters, but only `top_k` of `E` run per token.

## Support matrix

| Platform | Dense int4 decode | Routed MoE experts | Enabled |
|---|---|---|---|
| **NVIDIA · Windows** | torch **tinygemm** (`_weight_int4pack_mm`) — fast | bf16 rematerialize per expert (portable) | automatic (default) |
| **NVIDIA · Linux** | torch **tinygemm** — fast | **fused Triton w4a16** (opt-in, **Ampere/sm_80+**) *or* bf16 remat (default) | dense auto; experts **opt-in** |
| **AMD ROCm · RDNA** (e.g. Strix Halo gfx1151) | **Triton w4a16 + split-K** GEMV | **fused Triton w4a16** | automatic (only fast path here) |
| **CPU** | tinygemm-cpu (`_weight_int4pack_mm_for_cpu`) | bf16 rematerialize per expert | automatic (default) |

### Why the split

- **Dense:** on NVIDIA and CPU, torch ships a tuned fused int4 GEMM (`_weight_int4pack_mm`) — this is
  already the fast batch-1 path, and it's why int4 beats bf16 on the fleet. On **ROCm/RDNA** that kernel
  is **CDNA2+-only** (unavailable on RDNA), so InfiniteModel substitutes a Triton **w4a16** kernel, plus
  a **split-K** variant for the M=1 decode GEMV (the plain kernel launches too few programs at batch-1 to
  saturate the iGPU's memory bus). Split-K is therefore ROCm-specific — NVIDIA doesn't need it.
- **Routed MoE experts:** there is **no** torch fused int4 path for the 3D expert tensor on any platform,
  so the *default* everywhere except RDNA is to rematerialize each routed expert's full bf16 weight per
  token (correct, but extra memory traffic + a per-expert Python loop). InfiniteModel has a **fused
  grouped Triton kernel** that runs all `top_k` experts' int4 GEMVs in one launch. It's **automatic on
  ROCm** and an **opt-in upgrade on Linux+NVIDIA**.

### Why MoE fusion is opt-in on NVIDIA (the Windows split)

The fused expert kernel is **Triton**, which is mature on Linux but **unreliable to build on Windows**.
Most NVIDIA fleet boxes are Windows, so enabling it by default would risk build failures / log noise on
the majority of nodes (it would safely fall back, but it's not worth the churn). Instead:

- **Default (all OSes):** the portable bf16-rematerialize path — no Triton needed, works on Windows.
- **Opt-in (Linux + NVIDIA Ampere/sm_80+):** set `INFINITEMODEL_CUDA_FUSED_MOE=1` on the worker to route
  routed experts through the fused Triton kernel. This is the **advanced acceleration tier**, available on
  Linux only **and only on Ampere-or-newer GPUs** — the kernel uses bf16, which pre-Ampere cards
  (Pascal/Turing, < sm_80) cannot compile (`ptxas: Feature '.bf16' requires .target sm_80 or higher`).
  On an unsupported card the self-check catches the compile error and **falls back to the default**, so
  enabling it is harmless there — just ineffective.

This is the one place where InfiniteModel deliberately offers more on Linux than on Windows for the same
NVIDIA hardware.

**Validated (2026-06-30):** RTX 3060 (Ampere sm_86) — kernel builds, correct (rel 0.006), **3.9× on the
routed-expert GEMM** (microbench) and **end-to-end in the live worker: `olmoe:1b-7b` int4 8.5 → ~31.5
tok/s (~3.7×)** with the opt-in on. RTX 4070 Ti SUPER (Ada sm_89, **Windows**) — Triton builds on Windows
too; **end-to-end in the live worker** (`olmoe:1b-7b` int4, 10 MoE layers placed on the card): the
self-check went `-> ACTIVE` on **all 10 layers** (rel 0.0055–0.0061, max-element 0.005–0.013) with coherent
output. Quadro P620 (Pascal sm_61) — Triton bf16 won't compile → correct automatic fallback to the
default path (no speedup, no breakage).

**Kernel autotuning.** The fused MoE-expert GEMV is `@triton.autotune`d over `(BN, num_warps, num_stages)`
keyed on `(B, N, K)`, so it picks the best tile per expert shape on each GPU. The config set is lean (the
measured winners) to bound first-decode JIT cost. Sweep on the 4070 Ti SUPER (`bench_moe_w4a16.py`):
`num_stages=3` buys **~1.18×** on narrow-N shapes (qwen3-a3b expert GEMM) and is ~neutral on wider ones
(olmoe). It's a small end-to-end gain — on NVIDIA the dense GEMMs (tinygemm) dominate decode, and even the
best MoE config sits at only **~35–48% of the card's ~672 GB/s peak**. The remaining ~2× is *not*
occupancy — a **split-K variant was prototyped and measured slower** (0.66–0.85× — see "Further
optimization headroom"), because the kernel's `B = top_k` grid dimension already supplies enough programs
at decode. The gap is the memory-access pattern (the per-expert weight tile is strided across `N`) plus
the on-the-fly dequant, which only a re-pack / re-tile would address. Tuning is therefore optional polish
here, not load-bearing as it was on RDNA (where no vendor int4 GEMM exists at all).

> **Triton version note:** the kernels resolve `triton`/`tl` from **module globals** (not a local import
> inside the builder), because triton 3.2 — unlike 3.7 — does not capture them as closure freevars and
> would otherwise fail to compile with `NameError: tl is not defined`. Keep the import at module scope.

## Environment switches (worker)

| Variable | Effect |
|---|---|
| `INFINITEMODEL_CUDA_FUSED_MOE=1` | **Linux + NVIDIA Ampere (sm_80+) opt-in:** use the fused Triton MoE-expert kernel instead of bf16 remat. No effect on ROCm (already on); on pre-Ampere NVIDIA the bf16 kernel won't compile and the self-check falls back to the default. |
| `INFINITEMODEL_NO_FUSED_MOE=1` | Kill switch: force the bf16-remat (default) expert path everywhere, incl. ROCm. Use to A/B the fused kernel on/off. |
| `INFINITEMODEL_CUDA_GRAPH=<ctx>` | Opt-in CUDA-graph decode (single-node, standard-attention, uniform-CUDA models; `<ctx>` sizes the StaticCache mirror — set it to the serving ctx). Copy-handoff design: prefill/verify stay on the proven eager DynamicCache path; the first decode captures `model.forward` over a StaticCache mirror, the second **replays at a new position** and self-checks against the eager DynamicCache decode — activates only on a match (`[cudagraph] decode ACTIVE`), else latches off permanently and serving stays eager (byte-identical). Default-OFF (inert without this var). **HIP/ROCm gfx1151: tested 2026-07-08 (TheRock torch 2.12.0a0+rocm7.13, HIP 7.13.60980, llama-3.3-70b int4) — capture and replay execute without error, but the replayed logits are ~71–74% off (`rel=0.740`/`0.714`, two independent loads); the self-check DISABLES it every time. Leave unset on ROCm until a TheRock/HIP build replays faithfully; NVIDIA is the intended target.** |
| `INFINITEMODEL_PREFILL_CHUNK=<tokens>` | **Default 2048 (ON).** Processes a long-prompt **prefill** in query-chunks of this many tokens so SDPA never materializes the full `[1, H, q, total]` attention-score tensor. An explicit additive mask disables SDPA's flash backend; on a device without the mem-efficient backend (**ROCm gfx1151**, the CPU math path) SDPA otherwise falls to the math backend and allocates the whole `O(H·q²)` matrix → OOM on long prompts (the 43 GiB single-alloc seen on the Strix Halo). Chunking caps peak score memory to `O(H·C·q)`. Math-identical to the unchunked pass (validated); **standard-attention models only** (per-type/hybrid/omni keep the full pass); decode is unaffected. Set `0` to disable (single full pass); lower it (512–1024) for very long contexts on memory-tight boxes. |

Set them in the worker's environment (e.g. a systemd `--user` unit `Environment=` line, or the shell
that launches `client.py`), then restart the worker and reload the model — the fused forward installs at
load time.

## Windows + NVIDIA — enabling the advanced MoE tier (setup & requirements)

The fused MoE opt-in needs a working **Triton-on-Windows** build toolchain (the kernel JIT-compiles
through MSVC + `ptxas`). This is the only setup beyond the normal worker deps. Validated on the RTX
4070 Ti SUPER (Ada, sm_89).

**Requirements (one-time, on the worker box):**

1. **Visual Studio Build Tools** (the "Desktop development with C++" workload) — provides `cl.exe`. Note
   its full path, e.g. `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe`.
2. **CUDA Toolkit** (toolkit-only, *no driver*) — provides `ptxas` + `cuda.lib`, which the pip
   `nvidia-cuda-*` wheels do **not** ship on Windows. Install just the compiler pieces, e.g.
   `cuda_12.8.x_windows_network.exe -s nvcc_12.8 cudart_12.8 cuobjdump_12.8 nvdisasm_12.8` → lands at
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`.
3. **triton-windows**: `pip install --user triton-windows` (the woct0rdho fork; pick the wheel matching
   your Python — validated with 3.7.1 on CPython 3.14).
4. GPU must be **Ampere or newer (sm_80+)** — the kernel is bf16; Pascal/Turing can't compile it.

**Worker launch** (no `vcvars` needed — Triton auto-detects the MSVC/SDK paths, and `cl` finds its
sibling DLLs from the full path). A `start_worker.bat`:

```bat
set "CC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "INFINITEMODEL_CUDA_FUSED_MOE=1"
cd /d D:\infinitemodel
python client.py --controller <controller-ip> --control-port 50100
```

⚠ **WDDM / consumer-GPU gotcha — run the worker in the interactive session.** A consumer GeForce/Quadro
card under the WDDM driver is **not visible to a Windows *service* or a "run whether logged on or not"
scheduled task** (session 0). The worker must run in the **logged-on interactive desktop session** —
launch it from a normal `cmd`/shortcut or `shell:startup`, *not* a service/Task Scheduler entry.
(Datacenter cards in TCC mode don't have this limit.)

**Verify it engaged:** load a MoE and look for `[int4] fused-MoE self-check ... -> ACTIVE` in the worker
log (controller `GET /logs?node=<host>`). `-> fallback` means it self-checked out (or the toolchain is
incomplete) and the safe bf16-remat path is in use.

> Perf-only changes (these kernels) carry **no VERSION bump**, so a self-update *stages* the new code but
> does **not** auto-restart the worker — relaunch the batch file to pick up a kernel change. Run **exactly
> one** worker per box (two workers sharing a hostname fight over the controller registration).

## Safety

Whenever the fused MoE kernel installs, it runs a **one-time self-check** on the first decode: it compares
its output against the genuine bf16 reference for that exact module and only activates if the relative
error is within decode tolerance (mean `rel < 0.03`, worst-element `< 0.1`); otherwise it logs and falls
back to the per-expert path. Every call also falls back on any runtime exception. So enabling the opt-in
is safe — a build failure, numeric mismatch, or unsupported arch degrades to the default, never to wrong
output. Look for `[int4] fused-MoE self-check ... -> ACTIVE` (or `-> fallback`) in the worker log.

## The optimization stack & measured speedups

int4 decode speed is built from several layers; each targets a different bottleneck on a different
platform. This table is the whole story — what each one does, where it runs, and the measured gain.

| # | Optimization | What it does | Runs on | Measured gain |
|---|---|---|---|---|
| 1 | **int4 group-wise quant** | weights → ~4.25-bit (asymmetric, 128-group) → ¼ the bytes read per token | all | enables fit; the baseline everything below builds on |
| 2 | **Fused dense int4 GEMM** (torch tinygemm `_weight_int4pack_mm`) | dequantizes *inside* the GEMM — no bf16 rematerialize | NVIDIA, CPU, CDNA2+ | int4 7B **3.25 → 21.13 tok/s**; int4 now beats bf16 |
| 3 | **Triton w4a16 dense** | hand-rolled substitute where tinygemm is absent | AMD RDNA only | **5–20×** over naive (14× @4096², 20× @5120²) |
| 4 | **Split-K dense GEMV** (M=1 decode) | splits K across more programs to saturate the bus at batch-1 | AMD RDNA only | **3.5–3.9×** on the dense GEMV |
| 4b | **DRAM channel de-aliasing** (#dram-dealias, padded qweight rows) | when the packed row stride (K/2 bytes) is an even multiple of 64B — worst case a power of two, e.g. K=8192 → 4096B — every row maps to the same DRAM channels/banks and any matrix too big for the 32MB MALL collapses (17–67 GB/s); `prepare_fused` re-allocates rows on an **odd** multiple of 64B (+64B/row, no kernel change) and the GEMV autotune space gains the `BN=64/warps=16` family these shapes want | AMD RDNA only | worst GEMV (28672×8192) **17 → 175 GB/s**; **llama-3.3-70b e2e 0.61 → 1.73 tok/s (2.8×)**; the 32B/5120-dim shapes got faster too (never worse) |
| 4c | **MoE expert-row de-aliasing — measured** (#dram-dealias MoE, `Packed4Tensor3D.prepare_fused`) | within-expert rows sit `K_pad/2` bytes apart in the `[E,N,rs]` stack and a layer's experts (134–280 MB) blow past the MALL — but the response is **not** the dense rule (qwen3.6's pow-2 strides run *faster unpadded*; padding halves them), so the loader **times** the fused op unpadded vs row-padded (+64B/row strided view) per `(E,N,rs)` on DRAM-cold expert subsets and keeps the winner; `sqn` joins the autotune key so the variants tune apart (side effect: MoE decode autotune moves to load time) | AMD RDNA only | gemma-4-26b gate_up GEMV **63.7 → 187.5 GB/s (2.9×)**, per-token expert kernels **9.62 → 4.45 ms**; qwen3.6-35b measured & kept unpadded (unchanged, by design) |
| 5 | **Fused grouped MoE GEMV** (`_w4a16_moe_op`) | all `top_k` experts' int4 GEMVs in ONE launch (kills the per-expert Python loop) | ROCm (auto), NVIDIA (opt-in) | **~3.7×** on the expert GEMM (per-platform below) |
| 6 | **MoE autotune** `(BN, warps, stages)` | best tile per `(B,N,K)` per GPU | ROCm, NVIDIA | up to **1.18×** (shape-dependent) |
| 7 | **Serve-from-cache pre-packed int4** | streams pre-quantized layers; skips load-time re-quant | all | faster *loads* (not decode) |
| 8 | **Speculative decoding** (dense draft) | draft + batched verify | dense models (opt-in) | ~**1.5–2×** on a dense 70B |

### AMD Strix Halo (gfx1151 iGPU, ~150 GB/s realized) — `qwen3.6-35b-a3b` int4

| Build | tok/s | vs naive |
|---|---|---|
| naive int4 (bf16 rematerialize, no kernel) | 2.08 | 1.0× |
| + Triton w4a16 on dense linears (#3) | 3.5 | 1.7× |
| + per-expert w4a16 on MoE (#5, subclass form) | 5.42 | 2.6× |
| + **fused grouped** MoE kernel (#5) | 10.8 | 5.2× |
| + **split-K** dense GEMV (#4) | **15.4** | **7.4×** |

Decode here is **memory-bandwidth-bound** (~80% GPU-busy), so tok/s tracks effective GB/s, not FLOPs.
RDNA needs *all* of these — no vendor int4 GEMM exists on it at all.

**Big dense models (llama-3.3-70b class, hidden 8192):** these dims hit the #4b DRAM-aliasing
pathology (measured 0.61 tok/s before the fix, same box). With the fix, a **clean** box measures
**3.5 tok/s / 284 ms-token steady-state** (2026-07-08: 3.48–3.52 over three 128-tok runs across two
fresh loads, exclusive box) — ~78% of the ~4.5 tok/s pure-kernel ceiling, i.e. real eager overhead
is only ~60 ms/token. Two earlier numbers are hereby corrected: the **1.73 tok/s** first recorded
for the fix was depressed ~2× by contaminated first-token autotune picks under concurrent fleet
traffic (the caveat noted at the time — always bench after a fresh worker restart with the box
exclusive), and the ~362 ms/token launch-schedule sim does **not** reproduce on a clean box.
The graph-decode lever was tested and is **broken on this HIP stack** (2026-07-08, TheRock torch
2.12.0a0+rocm7.13.0a20260411 / HIP 7.13.60980): `INFINITEMODEL_CUDA_GRAPH=8192` captures and
replays without error, but replayed logits come back ~71–74% off (rel=0.740/0.714, two independent
loads) — the first-decode self-check catches it (`[cudagraph] replay self-check vs DynamicCache
rel=0.714 -> DISABLED (eager)`) and serving stays eager and coherent. Retest on a newer TheRock/HIP.

### NVIDIA — the MoE opt-in (dense is already tinygemm-fast)

| GPU | What was measured |
|---|---|
| **RTX 3060** (Ampere sm_86, Linux) | fused MoE opt-in: **3.9×** on the expert GEMM (microbench) and **end-to-end `olmoe:1b-7b` int4 8.5 → ~31.5 tok/s (~3.7×)** |
| **RTX 4070 Ti SUPER** (Ada sm_89, Windows) | Triton builds on Windows; fused MoE **self-check ACTIVE on all layers**, coherent gen. Expert-GEMM microbench **24–34×** over the bf16-remat default. Autotune: `num_stages=3` → **1.18×** on qwen3-a3b shapes; kernel sits at **~35–48%** of the card's ~672 GB/s peak |
| **Quadro P620** (Pascal sm_61) | bf16 Triton won't compile (< sm_80) → safe automatic fallback (no speedup, no breakage) |

On NVIDIA the **dense** GEMMs already use tinygemm (#2), so they're fast without Triton; the opt-in only
upgrades the **routed-expert** GEMV. End-to-end gain is therefore smaller than on RDNA, and largest for
MoE models where the routed experts are a big share of decode.

## Further optimization headroom

These kernels are good but not the ceiling. Ranked by likely payoff:

1. **CUDA/HIP-graph capture of the decode step — the big one (measured; opt-in landed).** Now unblocked: fusing the
   experts removed the data-dependent per-expert Python loop, so the per-token graph is static and
   capturable. A faithful 16-layer batch-1 decode probe on the 4070 Ti SUPER (norms + q/k/v/o linears +
   rotary + SDPA over a fixed KV + router top-k + the real fused int4 MoE) measured **eager 15.9 → graph
   2.9 ms/token = 5.56×**, i.e. **~82% of batch-1 compute-region time is pure launch/dispatch overhead**
   (~240 tiny kernel launches/token, GPU mostly idle between them). This is by far the largest remaining
   decode lever. The **correctness mechanism is proven**: a fixed-size KV buffer + position-driven causal
   mask + `index_copy_` write-at-position + rotary-from-static-position, captured and replayed per token,
   produced a **token-for-token identical** 24-step autoregressive sequence vs eager (4070, synthetic
   weights). **Caveats for the real end-to-end win:** the probe is compute-only — it excludes the
   per-hop loopback-TCP transport (not capturable), so a *distributed* model gains less; and the
   integration is substantial — it needs a **fixed-size KV buffer (HF `StaticCache` semantics) with the
   position as a captured tensor** (a growing/dynamic KV breaks capture), capture-once-replay-many with
   the prefill KV refilled per sequence, an opt-in flag + first-decode self-check vs eager + fallback,
   and the transport left outside the graph. Standard-attention single-GPU models are the clean first
   target; hybrid (Gated-DeltaNet) state, multimodal, and spec-decode are harder. **Status: landed as an
   opt-in** (`INFINITEMODEL_CUDA_GRAPH=<ctx>`, default off — see Environment switches). Single-node
   standard-attention only; a persistent StaticCache + captured per-token graph + first-decode self-check
   vs eager (permanent fallback on mismatch). To activate on a worker: add the env var to its launch
   (e.g. `start_worker.bat`), restart, load an eligible model, and confirm `[cudagraph] decode ACTIVE`
   in the log. **HIP/ROCm status (tested 2026-07-08, gfx1151 / TheRock torch 2.12.0a0+rocm7.13, HIP
   7.13.60980, llama-3.3-70b int4):** `torch.cuda.CUDAGraph` capture and replay execute without raising,
   but the replay computes wrong logits — self-check rel≈0.71–0.74 across two independent loads → auto
   `DISABLED (eager)` both times. The fallback is safe (coherent serving, no crash), but there is **no
   win available on ROCm** until a TheRock/HIP build replays faithfully; NVIDIA remains the proven
   target. Note the eager overhead this lever buys back is also smaller than first simmed: a clean
   llama-70b box decodes at 284 ms/token vs the ~222 ms kernel floor (~1.28× headroom, not 2.6×).
2. **AOTriton flash-attention on ROCm.** SDPA currently runs the slow MATH path on RDNA; AOTriton's flash
   kernel is gated behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Attention is a real slice of decode
   on the hybrid (Gated-DeltaNet + SDPA) models.
3. **Single-node in-process transport.** Adjacent pipeline stages on the *same* box still hand off over
   loopback TCP; an in-process path removes that per-token round-trip.
4. **Dense-kernel autotune on RDNA.** The RDNA dense Triton kernel still uses fixed tiles; the AMD iGPU is
   ~5× behind llama.cpp/Vulkan on the same silicon — the dense kernel plus (1)+(2) is where that gap lives.
5. **MoE weight re-pack for coalescing (hard, uncertain).** The fused MoE GEMV reaches ~35–48% of peak;
   the gap is the strided per-expert weight tile (laid out `[E, N, K/2]`, read strided across `N`), not
   occupancy. A transposed/tiled pack could coalesce it, but it's a format change touching the packer,
   shard cache, and kernel — large surface, uncertain win, and decode on NVIDIA is dominated by the dense
   tinygemm path anyway. Not currently worth it.

> **Tested and rejected — split-K for the MoE GEMV.** Splitting the K reduction across programs (the trick
> that won 3.5–3.9× on the *dense* M=1 GEMV, #4) was prototyped and benched on the 4070 Ti SUPER:
> **0.66× (olmoe) / 0.85× (qwen3-a3b) — slower.** Unlike the dense M=1 case (only ~`cdiv(N,128)`≈16
> programs, badly under-occupied), the MoE kernel's grid is `(B, cdiv(N,BN))` where `B = top_k` already
> multiplies the program count ~8× (≈128 programs ≈ 2 waves), so there's no occupancy to recover and the
> extra fp32 atomic-add contention on the shared output makes it a net loss. Generalizes to RDNA (fewer
> CUs → even more waves → even less starved). Kept here so it isn't re-attempted.

None is required for correctness; each is a throughput lever with diminishing returns on NVIDIA (where
dense is already fast) and larger upside on the bandwidth-starved AMD iGPU.

## CPU / RAM workers — do these kernels help? No.

**The Triton kernels (fused MoE, split-K dense) are GPU-only**, and torch tinygemm is GPU/CPU *vendor*
code — so none of this kernel work runs on a CPU worker. The CPU path is entirely separate:

- **Dense int4 on CPU** uses torch's own **CPU tinygemm** (`_weight_int4pack_mm_for_cpu`) when present;
  otherwise it falls back to **dequant→fp32 GEMM** (at batch-1 the int4 unpack is paid regardless, so the
  fp32 weight is "free" and the fp32 GEMM is the faster CPU path).
- **MoE experts on CPU** always **bf16-rematerialize** per expert — the fused Triton kernel's self-check
  returns `False` off-`cuda`, so a CPU worker never takes it.

So adding split-K, the fused MoE, or the autotune does **nothing** for CPU/RAM decode — and it wouldn't
help much even if ported: CPU decode is bound by **DDR bandwidth (~50–90 GB/s, vs 150–670 GB/s on a GPU)**
plus CPU compute, so it's fundamentally ~5–10× slower regardless of kernel. The fleet's CPU/RAM nodes
exist for **capacity** — fitting models too big for the GPU pool — not single-stream speed; CPU
tensor-parallel never beats pipelining onto a GPU here. A fused CPU-int4 MoE path is *possible* but
low-payoff, so it's intentionally not built.

## Choosing a fast model on bandwidth-limited GPUs

Decode tok/s ≈ memory bandwidth ÷ **active** bytes-read-per-token. Prefer a **sparse MoE with a low
active-param ratio** (e.g. ~3B active) over a big dense model of similar quality — the dense model reads
far more weight bytes per token. A dense 70B at int4 *fits* a 60 GB GPU but decodes slowly (~all params
read every token); a 30–35B-A3B MoE is much faster for the same memory footprint.
