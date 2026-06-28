# InfiniteModel changelog

A capability-level summary of how the engine came together. (The original repo tracked changes at
per-commit granularity in `server.py` / `client.py` `VERSION` tags; this public history starts from a
single squashed commit, so the detail below is grouped by milestone rather than by commit.)

## Distributed core
- Node registry + heartbeat + capability probe; live dashboard; RAM/VRAM-weighted partition planner.
- **Pipeline parallelism** over a hand-rolled plain-TCP transport (Windows + Linux): each worker holds
  a contiguous layer range; weights are **chunk-served** from the controller straight into worker RAM
  (no model on worker disk, no temp files).
- Incremental **KV-cache decode** (prefill-once, flat per-token cost); networked generation with full
  Ollama / OpenAI / Anthropic-compatible API surface.
- **Tensor parallelism** within a stage — capacity-proportional, GPU+CPU mixed meshes, KV-head
  replication; in-place reconfigure between pipeline and TP.
- **Speculative decoding** (opt-in, greedy-exact) — draft proposes K tokens, the pipeline verifies in
  one traversal; wins on big/distributed targets (measured). A checkpoint-MTP (nextn) *self*-draft
  for Qwen3.6 was built and the forward validated (~84-88% accept), but shelved: the hybrid
  Gated-DeltaNet trunk can't roll back its recurrent state on reject (not bit-exact) and a 2-token
  verify costs ~2x on the compute-bound GPU pipeline (no wall-clock win). Code kept, gated off.

## Memory, quantization & the shard cache
- **int4** (group-wise asymmetric, fused tinygemm GEMM) and **int8** (per-channel) load-time quant;
  serve-time dequant of **fp8** and **nvfp4** checkpoints.
- GPU-first placement that always fits (spill to CPU/RAM), full-context KV pre-reservation, coexistence
  budgets, and OOM-safe replans (cgroup caps, honest transient accounting). Placement MODES: `auto`
  (GPU-first, fewest nodes — best decode latency), `single`, `gpu-spread` (fill every GPU then spill to
  CPU), **`all-gpu`** (a stage on EVERY GPU, NOTHING on CPU — proportional across the GPU subset so each
  card carries >=1 layer; fails cleanly if the model won't fit GPU VRAM alone), `distribute`, `spread`,
  and `proportional`. `all-gpu` trades extra pipeline hops (per-token decode latency) for using all VRAM
  to avoid a CPU spill and to share prefill compute across cards.
- **Pre-compiled shard cache** — the controller quantizes a model once to `_shards/<quant>/`; loads
  then serve small **pre-packed** int4/int8 layers (skip the bf16 stream + re-quantize). Covers dense,
  fused-3D MoE, per-expert MoE fused at compile (Mixtral/OLMoE), and **non-fused per-expert MoE**
  (MiniMax-M2 — experts stay 2D Linears, int4-packed individually) — bit-identical to a cold load.
  Each cache unit's source tensors are read in **on-disk offset order** so a spinning weights drive
  reads sequentially (readahead) instead of seeking per tensor — large win for many-tiny-tensor MoE
  layers (read dominates compile time: e.g. MiniMax-M2 ~150 s read vs ~7 s pack per layer).
  **fp8/nvfp4-source MoE** compiles too: compressed-tensors quantizes per-expert `Linear`s, so each
  expert is a 2D `weight_packed` dequantized to bf16 by the same path dense fp8/nvfp4 uses, then
  fused-3D or packed per-expert on bf16 (only an exotic fused-3D *quantized* expert is unsupported).
- **Distributed packing** (exo-inspired) — the per-layer pack fans out across the fleet's idle CPUs:
  each worker fetches a layer's bf16, packs it with the *shared* packer (and, for per-expert MoE, fuses
  to 3D against a meta skeleton it rebuilds from the model config), and posts it back. Bit-identical to
  a single-box compile by construction (the same shared fuse + pack code), proven per-layer by a byte
  comparison, with automatic local fallback on any worker failure.

## Models
- **GGUF ingestion**: a model that ships weights only as a llama.cpp **`.gguf`** is normalized to a
  standard safetensors checkpoint ONCE at add/download time (`transformers` GGUF loader dequantizes →
  bf16 → `save_pretrained`), after which it is an ordinary model — chunk-streamed, int4/int8
  shard-cached, and run on the distributed pipeline with no GGUF awareness downstream (same idea as the
  fp8/nvfp4 source path). The heavy `from_pretrained` runs in a **subprocess** (`gguf_convert.py`) so it
  can OOM without taking down the controller box it co-hosts. Add via `/add_model?...&gguf_file=<one
  quant>.gguf` or the dashboard's optional GGUF field. Covers the architectures the GGUF loader supports
  (Llama/Qwen2/Mistral/Gemma/…); single-file quants only (split `NNNNN-of-NNNNN.gguf` is rejected with
  guidance); one quant per repo. Unlocks the large pool of GGUF-only community models.
- **MoE**: fused + non-fused experts; optional intra-layer offload (attention on GPU, routed experts in
  CPU RAM). Loaded + validated across Mixtral, OLMoE, Qwen3-MoE / Qwen3.6-A3B, MiniMax-M2.
- **Multimodal**: distributed vision + audio (Qwen2.5-Omni) — image/audio → text, 3D mRoPE positions.
- Hybrid architectures (Gated-DeltaNet + mRoPE), multimodal text-config models, and a range of dense
  decoders (Qwen2.5/3, Llama, Mistral/Devstral, DeepSeek).

## Multi-model & ops
- N models resident at once, per-node sharing, concurrency + queueing, auto-load/unload, same-model
  replication + data-parallel routing.
- Robust loads: survive a worker drop mid-load (replan on survivors), free partial shards on failure,
  scaled timeouts, gentler restarts; auto-recover resident models when a worker reconnects.
- **Stale-KV self-heal + crash-proof attention:** a worker stage's causal mask is sized for
  `cache_start + q`, but SDPA takes the real kv-dim from the cache (`past + q`); they desync — and
  crash with "expanded size N must match M" — when a generation reclaimed by the gen-stall watchdog
  (or a disconnecting client) leaves an UNCANCELLABLE forward running in a thread that keeps mutating
  the shared cache concurrently with a fresh forward. Two layers make this impossible: (1) forwards on
  a shard are SERIALIZED (a non-blocking per-shard guard — a racing new forward fails fast and the
  controller re-prefills, rather than concurrently corrupting the cache; lazily initialized so
  cache-served shards are covered); (2) a new sequence (`cache_start == 0`) unconditionally rebuilds
  the cache. A reclaimed generation can no longer corrupt the next one. (A defensive per-decode KV
  length "reconcile" was tried and reverted — `DynamicCache.get_seq_length()` inspects layer 0, which a
  mid/tail pipeline stage doesn't own, so it false-tripped on every multi-stage decode.)
- **Wedged-gen auto-recovery:** a distributed generation whose mid-pipeline hop dies never gets an
  error frame upstream (the data chain is one-way), so it used to sit ACTIVE at 0 tok/s until the
  600s timeout and needed a manual client restart. Two fixes: the gen-stall watchdog now (a) cancels
  the REAL streaming body-pump task (the cancel handle had been the route task, which returns
  immediately for a streaming response → the cancel was a no-op), and (b) fails the model's leaked
  controller-side pending futures so the orphaned `_send` returns at once. The model reclaims its slot
  and unblocks the queue on its own. (Hardened after an adversarial audit: the freed orphan's `finally`
  decrement is floored at 0 so it can't drive `active` negative after the watchdog zeroed it; and the
  leaked-future fail is skipped for data-parallel *replicated* models — `pending` is keyed by target_id,
  shared across replicas, so a blanket fail would also kill a healthy sibling's request, which instead
  relies on the per-request cancel.) Verified live: `/cancel` aborts a streaming gen at 6/400 tokens and
  the slot frees + the model serves again immediately (the same handle the watchdog uses).
- **Idle-pipeline self-heal:** every data-plane hop is fresh-reconnected at each generation's prefill
  if it has been idle (an idle TCP socket can go silently half-open — the write succeeds but the bytes
  never arrive — which otherwise stalls the first request after an idle gap until the generation
  timeout). Both the controller's connection to stage 0 and each worker's next-hop are freshened, so a
  model that sat loaded-but-unused replies immediately instead of appearing wedged.
- Observability: placement preview, per-load progress/ETA, fleet CPU/GPU/RAM + throughput + bandwidth,
  curl-able fleet logs; idle-gated multi-file self-update. **Per-model context history** — the model
  detail popup's "tokens in/out" rows are click-through to a scrollable view of the ACTUAL prompts
  sent and text generated (`GET /history`); captured as token ids (decoded lazily, off the hot path),
  kept to the most-recent N requests, and cleared when the model unloads. A managed reload (reconfigure to/from
  tensor-parallel) shows live layer progress on its own card (folded in from the in-flight load) rather
  than a progress-less "re-streaming weights" placeholder beside a duplicate load card.
- **TP mesh keepalive:** the tensor-parallel all-reduce mesh used to work for one generation then
  stall ("peer rank stalled or closed") after a short idle gap between requests — an idle mesh socket
  going silently half-open. Rank 0 now pings the peers (a tiny round-trip that keeps both directions
  warm) whenever the mesh has been idle a few seconds, so TP stays alive across idle periods instead
  of needing a reload after the first request.

## Public release
- Central `config.json` (all hosts/ports + the self-update source; no addresses baked into code);
  credentials and internal-only artifacts scrubbed for open source.
- **Self-update pulls from the public GitHub repo's raw endpoint** (`update_repo`/`update_branch`) — no
  token of any kind, on the controller or any worker. `provision_worker.sh` clones from public GitHub.
