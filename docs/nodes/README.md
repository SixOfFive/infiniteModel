# Per-device node setup

Practical, per-device setup guides for joining a machine to an InfiniteModel fleet as a
**worker** (or, for `beast`, as the controller + worker). Each guide covers install → run
the worker → persistence → optimal quant/ctx/placement for that VRAM tier → verify →
gotchas, grounded in the cross-platform references:

- [../ACCELERATION.md](../ACCELERATION.md) — int4 decode kernel matrix, the Windows/Linux
  NVIDIA + ROCm setup, the `INFINITEMODEL_CUDA_FUSED_MOE` MoE opt-in, prefill chunking.
- [../ROCM.md](../ROCM.md) — the full gfx1151 Strix Halo (AMD ROCm) recipe.

## Pick your node

| Doc | Host(s) | GPU / SoC | Arch | VRAM | OS | Role |
|---|---|---|---|---|---|---|
| [4070-ti-super.md](4070-ti-super.md) | `beast` | RTX 4070 Ti SUPER | Ada `sm_89` | 16 GB | Windows 11 | **controller** + GPU worker |
| [3060.md](3060.md) | `theocomp`, `mobile` | RTX 3060 / 3060 Laptop | Ampere `sm_86` | 12 GB / 6 GB | Linux / Windows | GPU worker |
| [p620.md](p620.md) | `work` | Quadro P620 | Pascal `sm_61` | ~4 GB | Linux | small GPU helper |
| [strix-halo.md](strix-halo.md) | `om3nbox` | Ryzen AI Max+ 395 (gfx1151) | RDNA3.5 (ROCm) | ~60 GB unified | Ubuntu | standalone controller+worker |
| [steam-deck.md](steam-deck.md) | `steamdeck` | Van Gogh (gfx1033) | RDNA2 | small UMA | SteamOS | CPU worker (ROCm experimental) |
| [cpu-worker.md](cpu-worker.md) | `nuc01-04`, `mini05`, `prodesk`, `zippy`, `tablet` | — (CPU/RAM) | — | — | Linux | CPU worker |

## Fleet shape, in one line

A **controller** (on `beast`, `:21434`) splits one model's transformer layers across
worker nodes over plain TCP and serves the Ollama + OpenAI + Anthropic APIs. GPU workers
contribute VRAM (run layers fast); CPU/RAM workers add **capacity** (fit bigger models when
distributed), not single-stream speed. `om3nbox` is a separate, self-contained
controller+worker on its own ~60 GB APU. Decode speed tracks a model's **active** params
(favor low-active-ratio MoE); total params track what **fits**.

> Compute-capability rule of thumb: the fused Triton MoE kernel needs **NVIDIA Ampere
> (`sm_80`) or newer** — it's an opt-in win on the 3060/4070, auto-disabled (safe fallback)
> on the Pascal P620, and the gfx1151 APU uses its own ROCm Triton path. See the per-device
> docs + ../ACCELERATION.md.
