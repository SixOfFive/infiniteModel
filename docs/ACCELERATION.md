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
best MoE config sits at only **~35–48% of the card's ~672 GB/s peak**. The remaining ~2× is *structural*
(the serial-K GEMV + on-the-fly dequant), reachable only by a **split-K rewrite** of this kernel (the same
trick used for the ROCm *dense* GEMV) — not by these three knobs. Tuning is therefore optional polish here,
not load-bearing as it was on RDNA (where no vendor int4 GEMM exists at all).

> **Triton version note:** the kernels resolve `triton`/`tl` from **module globals** (not a local import
> inside the builder), because triton 3.2 — unlike 3.7 — does not capture them as closure freevars and
> would otherwise fail to compile with `NameError: tl is not defined`. Keep the import at module scope.

## Environment switches (worker)

| Variable | Effect |
|---|---|
| `INFINITEMODEL_CUDA_FUSED_MOE=1` | **Linux + NVIDIA Ampere (sm_80+) opt-in:** use the fused Triton MoE-expert kernel instead of bf16 remat. No effect on ROCm (already on); on pre-Ampere NVIDIA the bf16 kernel won't compile and the self-check falls back to the default. |
| `INFINITEMODEL_NO_FUSED_MOE=1` | Kill switch: force the bf16-remat (default) expert path everywhere, incl. ROCm. Use to A/B the fused kernel on/off. |

Set them in the worker's environment (e.g. a systemd `--user` unit `Environment=` line, or the shell
that launches `client.py`), then restart the worker and reload the model — the fused forward installs at
load time.

## Safety

Whenever the fused MoE kernel installs, it runs a **one-time self-check** on the first decode: it compares
its output against the genuine bf16 reference for that exact module and only activates if the relative
error is within decode tolerance (mean `rel < 0.03`, worst-element `< 0.1`); otherwise it logs and falls
back to the per-expert path. Every call also falls back on any runtime exception. So enabling the opt-in
is safe — a build failure, numeric mismatch, or unsupported arch degrades to the default, never to wrong
output. Look for `[int4] fused-MoE self-check ... -> ACTIVE` (or `-> fallback`) in the worker log.

## Measured

On AMD Strix Halo (gfx1151, ~150 GB/s realized iGPU bandwidth), `qwen3.6-35b-a3b` int4 decode:

| Build | tok/s |
|---|---|
| baseline (per-expert + tl.dot dense) | 5.4 |
| + fused MoE experts | 10.8 |
| + split-K dense GEMV | 15.4 |

Decode on this iGPU is **memory-bandwidth-bound** (~80% GPU-busy), so absolute speed tracks effective
GB/s, not FLOPs. The same kernels on NVIDIA (much higher GDDR bandwidth) make the routed-expert decode
proportionally cheaper than the bf16-remat default — that's the win the Linux opt-in unlocks.

## Choosing a fast model on bandwidth-limited GPUs

Decode tok/s ≈ memory bandwidth ÷ **active** bytes-read-per-token. Prefer a **sparse MoE with a low
active-param ratio** (e.g. ~3B active) over a big dense model of similar quality — the dense model reads
far more weight bytes per token. A dense 70B at int4 *fits* a 60 GB GPU but decodes slowly (~all params
read every token); a 30–35B-A3B MoE is much faster for the same memory footprint.
