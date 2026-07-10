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
  serve-time dequant of **fp8** and **nvfp4** checkpoints. Selecting **int8 on a MoE auto-downgrades to
  int4** (with a loud log line): the int8 path only quantizes 2D Linears, so a MoE's fused-3D routed
  experts would otherwise stay bf16 → a near-bf16 footprint (OOM/CPU-spill); int4 packs the experts.
- **int2 (#int2) — a 2-bit CAPACITY tier.** `quant=int2` on `/load`/`/reconfigure`/auto-load config:
  group-wise asymmetric 2-bit (4 values/byte, group 64 — finer than int4's 128 because 2-bit RTN
  needs it), ~2.5 bits/weight effective (~1/6 of bf16); head/embed/norms/router stay bf16 exactly
  like int4. The int4 architecture cloned end-to-end: naive dequant path everywhere (CPU big-M gets
  the fp32-GEMM treatment), a **Triton w2a16** batch + split-K-GEMV kernel (same autotune space and
  dram-dealias row pad as w4a16) as the fused decode path on **both CUDA and ROCm** (int2 has no torch
  tinygemm; no-triton workers self-gate to naive), self-checked vs naive at placement with automatic
  fallback (`IM_FUSED_INT2=0` kill-switch). **Shard cache included**: `_shards/int2/` compiles via the
  same shared bit-identical packer (`pack_linear_int2` == the worker's `_quantize_linear2` by
  construction), cache-on-first-load fires for int2 loads, serve-from-cache installs QuantLinear2
  holders directly. **Dense models only**: int2 on a MoE auto-downgrades to int4 (no 2-bit 3D-expert
  packer/kernel), mirroring the int8-on-MoE rule; MoE cache compiles reject non-int4 as before.
  Planner/status size the tier at 0.2× layer weights (`for_quant`), `/status` quant_gb/quant_fits
  carry an int2 entry, and the dashboard's Load + auto-load-default selects offer it. The shipped
  **auto-load default remains int4** — int2 is an explicit operator choice.
  **Measured quality verdict (2026-07-10, qwen2.5 0.5B + 7B, greedy):** plain round-to-nearest at
  2 bits **collapses the model** (token salad) — and stays collapsed at group 32/16, with per-group
  MSE-optimal clip search, and under mixed-tier salvage (down/o_proj + edge layers at int4 —
  grammatical but meaningless at best). This matches the literature: RTN-2bit is broken at any
  scale; 2-bit needs a **GPTQ-class calibrated packer** to be usable. The infrastructure shipped
  here is deliberately packer-agnostic — a calibrated packer emits the SAME qweight/scale/zero
  format through the same kernels, cache layout and serve path (a packer-only follow-up;
  `packer_hash` in the cache manifest auto-invalidates stale int2 caches when it lands). Until
  then int2 is machinery-complete but NOT usable for real serving.
- **Hybrid models reserve KV only on their attention layers:** a Gated-DeltaNet hybrid (qwen3-next /
  qwen3.6) grows a full-context KV only on its `full_attention` layers (the linear-attn layers keep a
  small fixed recurrent state). KV reservation — both the GPU placement budget and the pre-alloc probe,
  which mirror each other — now charges full-ctx KV only on the KV-holding layers, so more of a hybrid
  fits per card. Conservative (an unknown layer reserves full KV); dense models are bit-identical.
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
- **Compile-on-first-load** — an int4 load with no shard cache yet BUILDS the cache first (via the
  deprioritized `/compile_shards` subprocess, so the GIL-heavy quantize never starves the event loop /
  drops live-gen logits), then serves the small pre-packed layers — so the FIRST load *persists* the
  cache rather than just re-quantizing in memory and re-doing it next time. One shared
  `engine._precompile_int4` covers BOTH the explicit `/load` (`precompile=1`, default) AND the auto-load
  path (a serving request to a not-yet-resident model); no-op when a cache exists / quant≠int4 / tp>1,
  non-fatal (any failure falls through to the cold on-the-fly load).

## Models
- **Model aliases shown in the UI**: a registry alias (e.g. `qwen2.5:14b` → `qwen2.5:14b-instruct`,
  via `MODEL_ALIASES`) is now surfaced as an "alias: …" line under the model's primary name — in the
  models list, each loaded-model card, and the model detail modal — so it's obvious which alternate
  names resolve to a given model. (`_aliases_for` reverse-maps `MODEL_ALIASES`; rendered in Ollama
  `family:size` form.)
- **Delete is a complete removal** (`/delete`): it purges the model's on-disk cache (its
  `models/<name>/` incl. the `_shards/<quant>/` pre-quant caches *and* the HF-cache duplicate) AND its
  whole registry footprint — every registered name that resolves to the same repo (so re-registered
  alias names can't dangle on now-missing files), its GGUF mark, and any built-in `MODEL_ALIASES` entry
  pointing at it. The model also **leaves the list entirely** (no stale "download" button): custom
  models drop from `custom_models.json`, and a deleted **built-in** is persisted to a `deleted_models.json`
  hide-set and filtered out after `MODELS` is seeded on startup (re-`/add_model` un-hides it). Delete ==
  forget + purge files + hide; refuses if any of those names is loaded or downloading. (`/forget`
  remains the opposite trade-off: unregister but keep the files.)
- **Mistral3 / Pixtral distributed vision** (validated end-to-end on Devstral, 2026-06-29): the
  controller-side vision encoder handles Pixtral's split tower (`vision_tower` + a separate
  `multi_modal_projector`, both materialized from the checkpoint's RAW key prefixes — Mistral3 stores
  them un-`model.`-wrapped — with the 24B text model left on meta), drives `get_image_features(pixel_values,
  image_sizes)` at the merged patch grid, and splices per-image embeds at the `[IMG]` (id 10) placeholders
  with plain 1D positions. Pixtral's 2D rotary table is rebuilt via the module's own rope-init (the generic
  1D materializer would corrupt it). Two integration fixes were needed: (1) Mistral ships its chat template
  as a standalone `chat_template.jinja` (not inside `tokenizer_config.json`), so the model download now
  pulls `*.jinja` and tops it up for already-present models — without it the tokenizer had no template, the
  prompt fell back to a flat `user:/assistant:` form, and the model degenerated; with it the native
  `<s>[INST][IMG]…[/INST]` renders. (2) the serving path injects the image placeholder for any tokenizer
  whose template emits none. devstral image→text: *"The image contains a red circle and a blue square."*
  Covers Devstral / Ministral. **Row structure (#150):** the `[IMG]` run now carries the trained Pixtral
  layout — `[IMG]×W` per patch row followed by `[IMG_BREAK]`, the last row closed with `[IMG_END]`
  (ids resolved from the tokenizer, verified by round-trip) — instead of a flat run, so the LM sees where
  each patch row ends. The per-image `(rows, cols)` grid is derived from `image_sizes` at the same merged
  cell the processor used; image embeds still splice only into the `[IMG]` slots (break/end keep their own
  embeddings), and any image whose grid doesn't match its token count falls back to the flat run.
- **Gemma 4 unified vision** (#143, validated end-to-end on gemma-4:12b-it, 2026-07-03): the
  encoder-free arch — no vision tower at all; `model.embed_vision` (LN → Dense → +factorized-2D-posemb
  → RMSNorm → Linear) projects raw merged pixel patches straight into LM space. The HF image processor
  hard-requires torchvision (which would clobber the pinned ROCm/CUDA torch), so preprocessing is a
  pure-PIL/torch reimplementation of the exact algorithm: aspect-ratio-preserving resize to a multiple
  of `pool*patch`=48 px (`F.interpolate` bicubic+antialias ≡ `tvF.resize`), 16 px teacher patchify,
  3×3 `patches_merge` into ≤280 model patches of 6912 values, pad with (-1,-1) positions. The raw
  safetensors keys are stored RENAMED (`vision_embedder.*`, un-nested projection) — the loader applies
  transformers' WeightRenaming table during collection. `get_image_features(pixel_values,
  image_position_ids)` returns padding-stripped LM-ready embeds; each template-rendered `<|image|>` is
  bracketed `boi + n×image + eoi` (processor parity) then expanded to its REAL per-image count and
  spliced with plain 1D positions. Multi-image attribution exact; works on all three APIs. Image-span
  bidirectional attention is now honored (see "Gemma 4 bidirectional image-span attention" below; was
  previously causal-first). Side-fix: gemma-4's `chat_template.jinja` was missing from the model
  dir, so even TEXT prompts had been served through the flat fallback — with it in place the native
  `<|turn>` form renders (and `<turn|>`=106 was already a registered stop).
- **Gemma 4 tower vision** (31b-it / 26b-a4b-it, model_type `gemma4`; validated end-to-end int4 on the
  CUDA fleet + bf16 on the ROCm box, 2026-07-05): unlike the 12b unified path above, the tower variants
  carry a REAL `Gemma4VisionModel` ViT (`vision_tower`: patch-embed → 27-layer encoder → 3×3 pooler)
  plus a SEPARATE `embed_vision` projector — the Mistral3 tower+projector shape, but the projector is
  `embed_vision` (not `multi_modal_projector`) and the checkpoint keys need NO rename. Because the 3×3
  pooling happens INSIDE the tower (after the encoder), it consumes UNMERGED 768-d (16×16×3) teacher
  patches — the unified path's pre-merged 6912-d preprocess is NOT reusable — so it drives the real
  `Gemma4ImageProcessor`, whose pure-PIL variant is torchvision-free (runs on the ROCm box too).
  `get_image_features(pixel_values, image_position_ids).pooler_output` is padding-stripped and LM-ready,
  spliced with 1D positions + boi/eoi wrap exactly like the unified path. Rotary subtlety: `gemma4_vision`
  builds a 1D `inv_freq[18]` at **θ=100** with a `head_dim//2` spatial split (not the θ=1e4 default), so
  the meta-tensor materializer now rebuilds it via the module's own `compute_default_rope_parameters`
  when a rotary module exposes one (Qwen's θ=1e4 vision path is byte-identical). Image-span bidirectional
  attention now honored, same as the unified path (see the dedicated entry below).
- **Gemma 4 unified audio** (#144, speech→text): the audio analog of the encoder-free vision path,
  equally torchvision-free and mel-free — each frame of `audio_samples_per_token`=640 **raw** waveform
  samples (40 ms @16 kHz) is one soft token, and `model.embed_audio` (a scale-free RMSNorm → a single
  `Linear` 640→text-hidden) projects them straight into LM space. The HF feature extractor is a trivial
  reshape, reimplemented directly (zero-pad each waveform to a multiple of 640, frame, batch-pad with a
  bool mask); the model is meta-built and only `model.embed_audio` (one tensor) is materialized, then
  `get_audio_features(input_features, input_features_mask)` runs with no downsampling so its output
  aligns 1:1 with the mask. Each `<audio_soft_token>` (258881) run is bracketed `boa`/`eoa` and expanded
  to the real per-clip frame count, spliced with plain 1D positions. Clips beyond `audio_seq_length`
  (750 tokens ≈ 30 s, the model's documented cap) are truncated with a logged warning (never silently).
- **Gemma 4 per-type attention masks** (2026-07-03): the per-type serving path (`layer_types`
  sliding/full + per-type rotary) was handing EVERY layer a single full-causal mask, so
  `sliding_attention` layers attended the whole context instead of only the last `sliding_window`
  (1024) keys — diverging from the reference once a prompt or generation crosses the window
  (single-node and distributed alike; latent below 1024 tokens). Both forward paths (`_forward_impl`
  and `_forward_uniform_eager`, prefill + decode) now build a windowed causal mask for sliding layers
  and the plain causal mask for full layers — validated **bit-exact (0.0)** against the HF
  `Gemma4TextModel` reference across lengths. Also: the head now applies `final_logit_softcapping`
  (±30; monotonic so greedy is unchanged, corrects temperature/top-p sampling parity), and the
  KV-reserve probe sizes each layer from its OWN attention geometry — gemma-4's full-attn layers are
  `global_head_dim`(512)/`num_global_key_value_heads`(1), not the uniform `head_dim`(256)×8, so the
  old probe over-reserved them ~4× and could false-OOM a tight stage into a needless replan. Root-cause
  note: the pipeline SPLIT itself is bit-exact (`num_kv_shared_layers=0`, per-stage rotary indexing by
  global layer index is correct) — a controlled offline harness proved single-node ≡ 2-stage; the
  reported "distributed-only garble" was the sliding-mask error (which also hits single-node past the
  window) compounded by fleet-contention hop-death, not a stage-boundary bug.
- **Gemma 4 bidirectional image-span attention** (2026-07-05): with `use_bidirectional_attention='vision'`
  (the 12b unified text config's default; the tower checkpoints set it too) the reference lets the soft
  tokens of each image attend **bidirectionally within their own block** — the pipeline had shipped
  causal-first, the one remaining vision-quality gap (location precision), flagged twice across prior
  handoffs. Now honored: the controller derives each image's contiguous soft-token run from the mm splice
  positions and rides them down the pipeline in the frame header (`bidir_spans`), exactly like
  `position_ids`, so EVERY stage rebuilds the same mask (TP peers get them via the broadcast tuple).
  `_causal_addmask` OR's a **blockwise overlay** (two positions attend iff they share one image run) onto
  BOTH the full and the sliding-window causal masks — bit-identical to HF's
  `or_masks(base, blockwise_overlay(get_block_sequence_ids_for_mask(mm_token_type_ids)))` (validated by an
  offline parity harness across single/two/edge/all-image layouts × windows {∞,4,1024}, all MATCH).
  Prefill-only (a decoded text token is block −1 → no change), gated on the text config's flag (every
  non-bidir model byte-identical), and chunked prefill is disabled while active so an image never straddles
  a chunk boundary. **Validated end-to-end on om3nbox** (gemma-4-26b-a4b-it int4, 2026-07-05): red-bg /
  white-circle image → "there is a white circle. The background color is red" — 256 image tokens spliced,
  clean stop, zero mask/shard errors.
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
- **Multimodal**: distributed vision + audio (Qwen2.5-Omni) — image/audio → text, 3D mRoPE positions;
  speech-out via the controller-side Talker + Token2Wav (`/v1/audio/speech`). Omni's checkpoint declares
  `architectures=["Qwen2_5OmniModel"]` (a bare `*Model` name), which the conservative encoder heuristic
  mis-read as an embedding model and routed to a single-node `AutoModel` build transformers can't
  construct; composite generative checkpoints (thinker/talker/token2wav/text/vision/audio sub-configs)
  are now excluded from that heuristic, keeping Omni on the pipeline Thinker path (re-validated end-to-end
  — load, text, and WAV — under transformers 5.12.1).
- Hybrid architectures (Gated-DeltaNet + mRoPE), multimodal text-config models, and a range of dense
  decoders (Qwen2.5/3, Llama, Mistral/Devstral, DeepSeek).

## Multi-model & ops
- N models resident at once, per-node sharing, concurrency + queueing, auto-load/unload, same-model
  replication + data-parallel routing.
- **Silent-wedge hardening (the beast kernel-panic postmortem, 2026-07-10).** A poisoned 30h-old
  worker process turned every distributed vision prefill into a silent 240s gen-stall reclaim —
  37 wedges in 5.5h, each client retry re-wedging, and the accumulated pathological load fed a
  host kernel panic (netconsole-captured NULL-deref; no GPU Xid — nvidia/UVM software state, not
  hardware). The worker's stage exception ("F.embedding got CUDABFloat16" — the mm companion frame
  consumed as forward input) never reached the controller: the data-plane error frame rides the
  one-way stage chain a stale hop can eat. Four fixes so one sick worker can never again become an
  hours-long wedge storm: (1) **#stage-error-ctrl** — every stage COMPUTE exception is also
  mirrored over the heartbeat-kept control link; the controller fails the request's future
  immediately (fast causal 500 instead of a blind stall) and logs every arrival, matched or not;
  (2) **#mm-pairing** — the prefill ids frame declares its multimodal companion (`hdr["mm"]`):
  declared-but-missing fails loud (never run a vision prefill unspliced), undeclared never claims
  (a leaked companion + controller-restart req_id collision can't splice stale image embeds into an
  unrelated prompt), and staged companions expire after 10 min; (3) **#stage0-dtype-guard** — a
  first-stage prefill frame carrying floating-point data is classified at the door as a mispaired
  mm/ids or misrouted hidden frame; (4) **#wedge-quarantine** — `wedge_reload_n` (default 3,
  `/config`-tunable, 0=off) gen-stall reclaims of the same model within 15 min trigger an automatic
  fresh re-place (reconfigure: new shards + new data conns, rollback-safe, serialized with the
  juggler) — the demonstrated cure for poisoned pipeline state. Ops note: worker files apply on
  fetch but worker PROCESSES only pick them up on restart — deploys that change worker code need
  `POST /restart?workers=1` after the `/update` (a periodic fleet-wide worker restart is also the
  cheap hygiene against long-lived-process state poisoning).
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
- **Per-load KV-cache placement + per-model default temperature.** `/load?kv_offload=1` (or the
  Load dialog's "KV cache: System RAM" option) rests the KV cache in system RAM — transformers 5.x
  `DynamicCache(offloading=True)`, per-layer side-stream prefetch — so the VRAM the full-ctx KV
  would reserve goes to model LAYERS instead (long context on small cards, at a decode-speed
  cost). The worker stops reserving per-layer KV against VRAM, probes the reservation against RAM,
  and reports `gpu_kv_bytes=0` so the multi-model coexistence reserve stays honest; cudagraph
  decode is gated off; mutually exclusive with `kv_quant`. CUDA-only: on ROCm/HIP the offloaded
  prefetch was live-validated GARBLING decode (nondeterministic at temperature 0 — a side-stream
  race in the TheRock stack) and an APU's "VRAM" is unified RAM anyway, so HIP falls back loudly
  to the plain on-device cache. `/load?temperature=0.7` stores a per-model DEFAULT sampling
  temperature (0-2), used only when a request sends none — explicit request values, including an
  explicit 0, always win; applied across the Ollama/OpenAI/Anthropic serve paths and badged on
  the model card. **min-p sampling** rides the same paths: applied after temperature and before
  top-p, it keeps only tokens with `p >= min_p * p_max` and renormalizes — per-request (Ollama
  `options.min_p`, OpenAI/Anthropic top-level `min_p`) or as a per-model default via
  `/load?min_p=` (0-1, badged `mp=` on the card). Both defaults are runtime-mutable on a LOADED
  model with **`POST /model_config?model=...&temperature=...&min_p=...`** (absent = keep, empty
  string = clear, applies to all replicas), surfaced as a "Runtime settings" panel in the
  model-detail modal — no reload needed to tune a resident model's sampling.
- **The full sampling-knob family (#runtime-knobs).** `top_k` (post-min-p top-k filter),
  `repeat_penalty` + `repeat_last_n` (llama.cpp multiplicative penalty over the last-N window of
  prompt+output; -1 = whole context), `presence_penalty` / `frequency_penalty` (OpenAI additive,
  output-only), and `seed` (reproducible sampling via a fresh per-request `torch.Generator` —
  concurrency-safe, never touches the global RNG; negative = the llama.cpp/Ollama "random"
  sentinel = unset) — per-request on all three APIs (Ollama `options.*` / OpenAI+Anthropic
  top-level; `repetition_penalty` accepted as the vLLM/HF alias in either location). Penalties
  apply to the logits pre-argmax, so they steer greedy decode too; the speculative path is
  greedy-only and ignores them by design. Every knob — plus `top_p` and a default `num_predict`
  for requests that send no length cap — is also a runtime-mutable per-model default on
  `POST /model_config`, stored in one `sampling_defaults` dict, reported on `/status`, and
  editable in the dashboard's Runtime settings panel (10 fields with suggested-value dropdowns;
  empty = unset; Apply sends the whole panel state). All knob values are coerced at PARSE time so
  a malformed value fails as a clean pre-stream 400 — never a post-stream empty-200 (the
  cold-contract rule); the stored seed is capped at 2^53-1 so it round-trips JSON/JS float64
  losslessly (per-request seeds go to int64 max).
- **Connections panel (#connections).** The dashboard's models page gains a bottom section
  listing every connected client (by IP): connected-for, idle-for (an active stream is never
  "idle" — activity is stamped per chunk), REAL bytes in/out counted at the ASGI layer (streamed
  responses grow the counter live; worker `/weights` slice-pulls are excluded), token totals
  in/out, request count, what the client is using or loading RIGHT NOW (in-flight join + the
  load card's `requested_by`), and a **Terminate** button — `POST /terminate?ip=` cancels every
  in-flight request from that client. Browser tabs that only watch the dashboard are chipped
  "dashboard"; a row is a real API client only once it hits a generation/embedding endpoint.
  X-Forwarded-For is charset-validated on BOTH derivation paths before it becomes a client key
  (it renders in HTML and an onclick — arbitrary header text would be an XSS vector).
- **Idle unload (#idle-unload).** New engine setting (`/config?idle_unload_m=`, dashboard
  "Idle unload"): a model that served NO requests for N minutes is unloaded automatically.
  Default 0 = the long-standing behavior — every model stays loaded forever. Judged GROUP-wise
  across data-parallel replicas (unload(base) cascades, and the base carries last_used while the
  routed replica carries active/last_token_ts — judging one key alone could reap a group whose
  sibling is mid-decode); models with an active or queued request, a held per-model lock
  (embeddings), or either lifecycle pin (persist_models / no_unload_models) are never idle-unloaded, and the speech thinker
  stamps per-step progress so long TTS runs aren't reaped. Replaces the old hidden coupling
  where the LRU auto-unload checkbox also enabled a hardcoded 60-min idle unload. Ollama
  `/api/ps` `expires_at` is now honest: last activity + the idle window when the knob is on.
  The knob is clamped to a finite [0, ~1 year] (an `inf` would persist and 500 /status +
  /api/ps).
- **Honest RAM/CPU weight split (#real-stats).** `ram_used_gb` / `cpu_frac` (and the load-time
  "X% of weights on CPU" warning) were `spec_estimate − measured_gpu_bytes`; the spec's formulaic
  int4 estimate overshoots real packed MoE size ~10%, fabricating a phantom "1.9 GB RAM / 10.6%
  CPU" on a fully-GPU-resident Qwen3-30B-A3B (verified per-tensor: everything on cuda). The
  worker has always reported its MEASURED post-quant weight bytes in the load result — the stage
  now carries it and both numbers are computed measured-vs-measured (spec fallback only for
  workers that predate the field). The model-detail placement row also now reads
  "on GPU x of <node total> VRAM" instead of a bare "GPU x GB" that looked like a device spec. triton's `Autotuner.run()` keeps the
  call's args in unsynchronized instance state (`self.nargs`, set on entry / `None` on exit) and the
  int4 w4a16 kernels (dense GEMV + fused MoE) are process-wide singletons shared by every shard — so
  with TWO models resident, any decode that autotune-benchmarks a NEW shape key while the other model
  decodes crashed (`TypeError: 'NoneType' object is not a mapping` in `autotuner._bench`),
  deterministically. Fixed three ways: (1) `Autotuner.run` is serialized behind one process-wide RLock
  (a lock acquire per launch — negligible vs ms-scale decodes; during a bench window other int4
  launches briefly wait instead of crashing); (2) the lazy kernel **builders** are built under a lock
  with their tried-flag set only AFTER the op is final (a racing shard-install could previously capture
  a permanent naive 5-20x-slower fallback mid-build); (3) the expert tensor-subclass is single-built
  under the same lock.
- **Wedged-gen auto-recovery:** a distributed generation whose mid-pipeline hop dies never gets an
  error frame upstream (the data chain is one-way), so it used to sit ACTIVE at 0 tok/s until the
  600s timeout and needed a manual client restart. Two fixes: the gen-stall watchdog now (a) cancels
  the REAL streaming body-pump task (the cancel handle had been the route task, which returns
  immediately for a streaming response → the cancel was a no-op), and (b) fails the model's leaked
  controller-side pending futures so the orphaned `_send` returns at once. The model reclaims its slot
  and unblocks the queue on its own. (Hardened after an adversarial audit: the freed orphan's `finally`
  decrement is floored at 0 so it can't drive `active` negative after the watchdog zeroed it.) Verified
  live: `/cancel` aborts a streaming gen at 6/400 tokens and the slot frees + the model serves again
  immediately (the same handle the watchdog uses). Recovery is **replica-precise**: a parallel
  `pending_friendly` map keys each in-flight request by the UNIQUE replica it was routed to (not the
  `target_id` every replica of a base SHARES), so both the watchdog AND `invalidate_model` (a node
  leaving mid-pipeline) fail ONLY the dead replica's leaked futures and never a healthy sibling's — so
  a data-parallel model's stalled request now gets the same fast future-fail as a single-copy one
  instead of hanging out the ~600s timeout (this supersedes the earlier replicated-SKIP).
- **Concurrent-load isolation:** each control link keys its in-flight load/unload futures by model_id
  (a dict) instead of one shared future, and the worker echoes model_id in its ready/error reply. Two
  models loading onto stages of the SAME node concurrently no longer cross-resolve each other's load
  (which mis-counted VRAM / reported "ready" early and hung the loser to its multi-minute timeout); a
  sole-pending fallback keeps an old worker build working through a rolling deploy.
- **Early architecture guard:** an exotic/unsupported model now fails at load-plan time with a clean
  "unsupported architecture 'X'" instead of a cryptic meta-tensor crash deep in the streamed worker
  build. The controller checks the config RESOLVES via `AutoConfig` (a registered model_type) — it does
  NOT attempt a full model build, so natively-registered archs the worker hand-builds via a special path
  (e.g. Qwen2.5-Omni) still pass; trust_remote_code models (auto_map) pass through too (the worker fetches
  their .py via `/modelcode`), so no known-good model is rejected.
- **Complete Ollama + OpenAI API surface:** alongside the Ollama routes (`/api/tags`, `/api/chat`,
  `/api/generate`, `/api/show`, `/api/ps`, `/api/embed`+`/api/embeddings`, `/api/version`, `/api/pull`,
  `/api/delete`) and the Anthropic Messages API (`/v1/messages` — the Claude Code backend), the OpenAI
  surface is now complete: `/v1/chat/completions`, **`/v1/completions`** (legacy text completion —
  `text_completion` objects, SSE + `[DONE]`, prompt string-or-array), `/v1/models` + **`/v1/models/{id}`**
  (retrieve), `/v1/embeddings`, `/v1/audio/speech`. An unknown model returns **HTTP 404** with the
  dialect-correct shape — Ollama `{"error":"model 'X' not found"}`, OpenAI
  `{"error":{message,type,code:"model_not_found"}}` (was a bare 400) — and OpenAI endpoints default
  `stream` to FALSE when omitted (single JSON, per the OpenAI spec) while Ollama keeps its
  stream-when-omitted default.
- **Fast dead-hop recovery:** a mid-pipeline hop dying *during* a generation used to leave the request
  blocked until the gen-stall watchdog (~240s) or GEN_TIMEOUT (~600s) reclaimed it — the one-way data
  chain delivers no upstream error frame. Now, when a worker's forward to its next hop fails even after
  its own reconnect-retry (a genuine transport death — gated strictly on connection-type exceptions so a
  stage *compute* error never trips it), the worker pushes an unsolicited `hop_error` control frame up
  its (separate) control link; the controller fails ONLY that request's pending future at once
  (idempotent, replica-precise by `req_id`), reclaiming the slot in well under a second. The send reuses
  the control writer's lock+framing so it can't corrupt a heartbeat, doesn't double-decrement `active`
  (the resumed generate() does that), and resets `last_token_ts` so the watchdog doesn't double-act —
  falling back to the watchdog only if the control link is mid-reconnect.
- **Active-decode stall reclaim:** `hop_error` can't catch a *buffered-write deadlock* — a downstream
  hop dies, the upstream's small forward write buffers "successfully" (no error raised), and the one-way
  pipeline then deadlocks so the upstream never writes again to surface the failure. A **second, shorter
  watchdog threshold** (`gen_stall_decode_s`, default 60s) now covers it: it applies ONLY once a
  generation has produced its first token (it's *decoding*, tracked via `gen_started_ts` vs
  `last_token_ts`), so a streaming gen that goes silent is reclaimed in ~60s instead of ~240s. Cold
  prefill keeps the conservative 240s `gen_stall_s` (a slow big-model first-token wait is never
  false-killed — 60s is far longer than any healthy per-token decode, even heavy CPU spill). Both
  thresholds are `/config`-tunable. (Found by an isolated fault-injection test: a worker *crash*
  recovers in ~2.6s via the control-link drop, but a pure *data-plane* partition needed this.)
- **Idle-pipeline self-heal:** every data-plane hop is fresh-reconnected at each generation's prefill
  if it has been idle (an idle TCP socket can go silently half-open — the write succeeds but the bytes
  never arrive — which otherwise stalls the first request after an idle gap until the generation
  timeout). Both the controller's connection to stage 0 and each worker's next-hop are freshened, so a
  model that sat loaded-but-unused replies immediately instead of appearing wedged.
- Observability: placement preview, per-load progress/ETA, **live download speed + ETA** (a rolling
  ~30s byte-rate over the HF-cache pull → remaining/rate, surfaced per model card while a pull runs),
  fleet CPU/GPU/RAM + throughput + bandwidth, curl-able fleet logs; idle-gated multi-file self-update. **Per-model context history** — the model
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
- **One-click int4 precache from the models list:** every on-disk model without an int4 shard cache
  shows a clickable `⚡ int4` chip (on registered AND loaded rows) — hover for what the compile costs
  (estimated cache size on disk, source dtype, controller free-disk check) and what it buys (int4
  loads then serve from cache instantly); click fires the same `/compile_shards` as the detail
  modal's Precache button. Compiling rows show live progress (`done/total · elapsed · ETA` + bar)
  instead of a static "compiling…". Embedding encoders never show the chip (their serve path is a
  whole-model float32 load that doesn't read shard caches), and uncached models no longer display a
  misleading "cache ready".
- **Endpoint weather — contention is survivable and honestly retryable.** Under GPU contention,
  healthy-but-slow prefills used to be reclaimed by the gen-stall watchdog at the threshold (~4 min),
  and every client retry re-entered the same slow prefill and died again — a fan-out harness measured
  a 21% run-abort rate, all from this class. Three-part fix: **(1) prefill-progress liveness** —
  workers already stamp per-layer forward progress for their local watchdog; that signal now rides
  the existing heartbeat (`fwd_progress`, request-id-attributed so an orphaned forward can never
  shield a live generation), and the controller watchdog's PREFILL branch treats advancing progress
  as liveness. True wedges (no layer completed for `gen_stall_s`) still reclaim on the old schedule;
  decode stall detection is unchanged (tokens only). **(2) Adaptive prefill wait** — the controller's
  per-frame generation timeout no longer hard-kills a prefill at 600 s: the wait extends in slices
  while worker progress advances (absolute 1 h ceiling as the backstop). **(3) Retryable errors** —
  contention-class failures (watchdog reclaim, dropped data-plane sockets, hop timeouts, a shard
  held by an orphaned forward, node-drop recovery races) now return `503 + Retry-After`
  (Ollama/OpenAI) or `529 overloaded_error` (Anthropic) instead of bare 500s, and a watchdog-reclaimed
  in-flight request gets a clean retryable response instead of an aborted socket. User-initiated
  `/cancel` and `/terminate` keep their kill semantics (never invite a retry). **Streaming paths**
  get the same honesty: a mid-stream reclaim or contention failure emits a typed TERMINAL error
  frame in each protocol's idiom — Ollama `done_reason:"error"` + `retryable:true`, OpenAI a
  `{"error":…}` object (no longer a clean `finish_reason:"stop"` that presented a truncated answer
  as complete — the worst of the pre-fix cases), Anthropic an `overloaded_error` event — rather than
  a silently truncated stream; a genuine client disconnect or user cancel still drops the connection.
- **Idle-unload accepts `-1` as "keep forever":** the Ollama-style sentinel round-trips (saves and
  displays as -1) instead of silently resetting to 0; -1 and 0 mean the same thing — the reaper is
  off and `/api/ps` reports effectively-never expiry.
- **Lifecycle pins + the juggler (hitless VRAM promotion).** Two independent per-model pins on the
  model-detail modal plus one global control. **Autoload on restart** (`persist_models`, previously
  API-only, now a checkbox) re-streams a model to its workers on controller startup so a resident
  model survives a restart/redeploy. **Do not auto-unload** (`no_unload_models`, `/config?no_unload=`)
  is an absolute veto: the model is never reclaimed by idle-unload *or* by LRU eviction — a new load
  that can't otherwise fit FAILS rather than displacing it (distinct from persist, which survives a
  restart but stays evictable under memory pressure). **Juggler** (`/config?juggler=`, off by default)
  turns a model that auto-loaded HYBRID (weights split GPU+RAM under memory pressure) back into a
  full-GPU model once room frees: on a ~60 s sweep (and right after an idle-unload frees VRAM) it picks
  the hottest resident hybrid *that a VRAM-first planner dry-run says now fully fits on GPU* — skipping
  embeddings and any hybrid too big to fit, so a bigger hot one never blocks a smaller promotable one —
  then — only if that model is momentarily IDLE (a busy/backlogged one is skipped, not stalled: engaging
  the barrier and draining it could hold a slow model's clients for minutes; a later sweep catches it at
  a gap) — does a **hitless** swap: a per-model barrier (checked at the top of `generate()` before the
  request takes a queue slot, so it's race-tight) holds new requests while `reconfigure` re-places it
  VRAM-first (atomic, with rollback), then releases — so the client's open connection just pauses across
  the ~10-20 s re-place, no reconnect. The juggler is exempt
  from the do-not-auto-unload veto BY DESIGN: it may promote a pinned hybrid too, because a promotion
  is a reload-into-a-better-placement, not a removal — and it restores the model if a rare
  double-failure ever evicts it, so the pin's "always resident" contract still holds. **Autostart
  delay** (`autostart_delay_s`, default 60 s) makes the startup reload of persisted models wait at
  least that long — on top of the fleet-settle wait — so API clients reconnect before the controller
  gets busy streaming weights.

## Public release
- Central `config.json` (all hosts/ports + the self-update source; no addresses baked into code);
  credentials and internal-only artifacts scrubbed for open source.
- **Self-update pulls from the public GitHub repo's raw endpoint** (`update_repo`/`update_branch`) — no
  token of any kind, on the controller or any worker. `provision_worker.sh` clones from public GitHub.
- **Deploy guardrails:** each self-update file fetch is bounded-retried with backoff, so a freshly-added
  module that hasn't propagated on the raw CDN yet gets time to sync instead of aborting the whole cycle
  and leaving the fleet under-deployed (the apply stays atomic — all files or none). The auto-RESTART is
  gated on a **VERSION bump** in the primary file: a same-VERSION doc/comment commit stages to disk
  WITHOUT bouncing the fleet, so a casual push no longer reboots the cluster. The forced dashboard
  "Update + restart" always restarts regardless. (Stale "GitLab" self-update wording corrected to GitHub
  throughout.)
- **Deploy verification (`GET /code_manifest`):** the raw-CDN edge lags a push *per controller*, so a
  forced `/update` can pull a stale file on one box while the CDN looks fresh from elsewhere. This route
  reports the on-disk `sha1(12)`/size/mtime of every self-update file plus the running `VERSION`/`CODE_DATE`
  (and `?grep=<marker>` reports per-file whether a marker is present on disk) — so a deploy verifies the
  bytes actually landed with one HTTP call instead of SSH-ing in to grep.
- **Multimodal backend self-heal:** transformers memoizes its PIL/soundfile/torchvision availability at
  import, so a dep `pip install`ed *after* the controller started stayed invisible (vision kept
  ImportError-ing) until a full restart — the trap the `.38` Proxmox rebuild hit (venv had torch, not
  Pillow). The controller now re-probes and busts that cache at startup, lazily per image/audio request
  (throttled), and on demand via `POST /refresh_backends` — a freshly-installed backend goes live with no
  restart.

## Code organization (context-economy refactor)
- The controller and worker grew into multi-thousand-line files that were costly to read/edit. They are
  being split along seams the *callers* don't see — **zero public-API change** — so each subsystem fits a
  reader's (and an editor's) context window. The multi-file self-update built earlier already keeps any
  number of sibling modules in lock-step across the fleet (each in `EXTRA_UPDATE_FILES`, each imported
  through a pull-once **convergence bridge** so an old checkout self-heals on the deploy that introduces it).
- **Engine split (m4c152):** `server.py`'s `Engine` class (~2.5k lines, 47 methods) was relocated
  *verbatim* into three mixin modules — `engine_load.py` (load/placement/TP/reconfigure), `engine_gen.py`
  (prefill/decode/spec/MTP), `engine_lifecycle.py` (data-plane/recovery/replicas/unload) — recomposed as
  `class Engine(EngineLoadMixin, EngineGenMixin, EngineLifecycleMixin)`. Method bodies are byte-identical;
  only `__init__` and `generate_speech` (which rebinds the `ENCODING` idle-gate global) stay on the shell.
  A new `state.py` registry publishes the controller's namespace and injects it into the relocated modules
  at startup (`state.publish`/`state.bind`), so the moved bodies resolve their former module globals
  without a circular `import server`. server.py dropped from ~9090 to ~6790 lines.
- **Route split (m4c153):** `build_app`'s 73 HTTP routes (~2.5k lines, all defined inline) — 57 of them
  relocated *verbatim* into four `register_*(app)` modules: `routes_dashboard.py` (dashboard/status/
  graphs/plan/logs/config), `routes_lifecycle.py` (load/unload/compile/reconfigure/restart/weights),
  `routes_api.py` (Ollama+OpenAI+Anthropic inference + model-info), `routes_diag.py` (vision/audio/probe
  test endpoints). `build_app()` calls `register(app)` on each. The 15 routes that rebind a runtime global
  (download/add_model/forget/nodeconfig) or use a build_app-local helper (embed/delete) stay in build_app
  — avoiding the publish/bind stale-snapshot trap. Route bodies byte-identical; globals injected via
  `state.bind`. server.py dropped to ~4350 lines (from ~9090 at the start of the refactor).
- **Worker split (m4c153):** the worker's `Shard` (~1260 lines) and `Worker` (~760) classes split the same
  way — `shard_build.py` (placement / streaming weight-load / from_*), `shard_forward.py` (forward path),
  `worker_load.py` (build/load/pack/unload/TP), `worker_net.py` (next-hop connect/send + data-plane). Shells
  keep `__init__` (+ `Shard._finalize_placement`, which reads the rebound `_CPU_FP32_GEMM` so must read it
  live). `state.py` is now shared by controller and worker (in both EXTRA_UPDATE_FILES); the worker publishes/
  binds at module load. client.py dropped from ~4570 to ~2820 lines. Across the whole refactor the two giants
  went from 9090 + 4570 ≈ 13.7k lines to ~4350 + ~2820 ≈ 7.2k, the rest living in focused 200–1200-line modules.
- **Serving layer split (m4c154):** the request-serving functions `_serve` (Ollama/OpenAI generate+chat),
  `_serve_anthropic` (Claude Code backend), `_count_tokens_anthropic` (+ `_serve`'s private `_prepare`/
  `_ka_is_unload`) moved verbatim into `serving.py`. server.py back-imports the three entry points so the
  already-relocated `routes_api` resolves them through the published namespace; `state.bind(serving)` makes
  their bodies resolve server globals. server.py → ~3730 lines.
- **Status layer split (m4c155):** the read-only status builders `build_status` (the big /status + dashboard
  payload), `_model_entry`, `_loading_view`, `_tag_entry` moved verbatim into `status.py`; server.py
  back-imports `build_status`/`_tag_entry` (called by routes_dashboard/routes_api). Prerequisite fix:
  `load_download_state()` now mutates `DOWNLOAD_STATE` **in place** (`clear()`+`update()`) instead of
  rebinding it, preserving object identity so the `state.publish` snapshot stays live for the relocated
  `_model_entry` (and removing the last `DOWNLOAD_STATE` rebind footgun). server.py → ~3380 lines. The
  history/metrics block was analysed for extraction too but deliberately **left in server.py** — it's
  movable but needs ~16 back-imports (server would re-import almost the whole API) plus the
  `graphs.set_history_sources` identity invariant, i.e. a line-count move with little real decoupling.
- **Multimodal-adapter dedup (#147):** the encoder-free Gemma-4 vision (#143) and audio (#144) loaders
  had grown a duplicated "meta-build the model, then materialize only the multimodal submodule(s) from
  the raw safetensors" loop. That loop — read raw keys per `(submodule, prefix)`, apply per-arch stored-name
  renames before matching, try both the qualified prefix and the `model.`-stripped candidate (Mistral3),
  assign-load, materialize meta buffers, move to device — is now the single `_materialize_submodules(...)`
  helper that both `_load_vision_encoder` (all image arches: Omni / Qwen-VL / Mistral3 / gemma4 / standard)
  and `_load_gemma4_audio_encoder` call. Behaviour is byte-for-byte the pre-refactor vision loop
  (re-validated end-to-end on gemma-4 vision, gemma-4 audio, and Mistral3 split-tower vision).
- **Code-split round 2, increments 1-3 + deploy enablers (2026-07-06):** continuing the m4c152-155
  context-economy refactor with the same contract (byte-identical relocation, `state.bind` globals,
  convergence bridge, `EXTRA_UPDATE_FILES` sync). New controller-only leaves: **`control_plane.py`**
  (~500 lines: control-frame IO, `ControlLink`, the resilient TCP listener, `handle_control`,
  `reaper_loop`, `gen_stall_watchdog` — carries its own stdlib imports because `@dataclass` executes at
  import, before `state.bind`) and **`serving_anthropic.py`** (~450 lines: the `/v1/messages` Anthropic
  engine + `_count_tokens_anthropic`, where all recent vision/audio serve-path edits land; shared
  helpers stay in serving.py, imported leaf-to-leaf). The embed trio (`_serve_embed` + its 3 routes)
  folded into the existing `routes_api.py`. server.py 4,078 → 3,539; serving.py 1,434 → 1,000.
  **Deploy enablers:** `/code_manifest` now also reports `client.py` + the WORKER-side
  `EXTRA_UPDATE_FILES` (regex-extracted from client.py's source — worker deploys bump no controller
  VERSION, so this is their only HTTP-visible ground truth), and the worker convergence bridge
  bounded-retries then **exits 42** on failure so a raw-CDN 404 on a freshly-added module is a bounded
  crash-loop instead of a permanently dead Windows worker (`client.bat` relaunches only on 42). New
  controller modules deploy **two-phase**: module committed+pushed first, pre-staged on every
  controller (`git checkout origin/main -- <mod>.py`), then the server.py that imports it — the
  bridge is fetch-once, and a single commit can race the idle self-updater into a bridge-404
  restart loop.
- **Code-split round 2, increments 4-6 (2026-07-06):** the persistence loaders (`load_node_config` /
  `load_custom_models` / `load_deleted_models`) now mutate their dicts/set **in place** instead of
  rebinding — `main()` publishes the namespace *before* running them, so a rebind stranded every bound
  leaf module on the pre-load empty objects (latent staleness; the m4c155 `DOWNLOAD_STATE` fix,
  generalized). That unblocked **`downloads.py`** (~455 lines: `_pull_repo_interruptible`,
  `_start_download`/`_do_delete`, and the `/download*`, `/add_model`, `/delete`, `/forget`,
  `/api/pull`, `/api/delete` routes — the `DOWNLOAD_*`/`ENCODING` *definitions* stay in server.py where
  the self-update idle lambda live-reads them; the module header documents that invariant) and
  **`routes_shards.py`** (~900 lines merged out of routes_lifecycle.py: shard-cache/packing control
  routes + the worker-facing `/weights` `/weights_tp` `/experts` data plane + the parked `/mtp_probe`
  `/modelcode` debug pair; one module instead of two halves the fleet-sync surface on the route group
  whose convergence-window failure would break every model load). `/nodeconfig` + `/nodeconfig_all`
  landed in routes_api.py (tier config, not downloads). routes_lifecycle.py keeps the true lifecycle
  group (1,341 → 458). Cumulative round-2 effect: **server.py 4,078 → 3,121** · serving.py 1,434 →
  1,000 · routes_lifecycle.py 1,341 → 458; new leaves: control_plane, serving_anthropic, downloads,
  routes_shards. Validated per increment on om3nbox (incl. a real int4 load streaming through the
  relocated `/weights`) and on the production controller (12 nodes re-registered clean).
- **Code-split round 2, increments 7-8 — the worker side (2026-07-06):** first client.py splits under
  the same contract, deployed via VERSION-gated rolling worker self-update (fleet converged in ~3 min,
  zero dropped workers — the exit-42 bridge enabler held). **`worker_hw.py`** (~450 lines: memory/GC,
  capability probes, the read-only route detectors, RAM-module detection, `build_registration`, startup
  cleanup — `_ROUTE_SRC`/`_local_addr` stay in client.py, the live rebind pair). **`worker_update.py`**
  (~265 lines: the self-update machinery + fwd-watchdog + console panel; `EXTRA_UPDATE_FILES` stays in
  client.py, the primary file every worker refreshes). `EmbeddingModel` + `_build_with_autodeps` and
  the HF-local weight helpers moved into the EXISTING `worker_load.py` beside their only call sites
  (zero new fleet-sync surface). client.py 3,699 → 2,923.
- **Code-split round 2, increment 9 — `shard_compile.py` (2026-07-06):** the shard-cache compile/pack
  family (PACKER_VERSION/`_packer_tag`, `pack_linear_int4/_3d/int8`, `pack_unit_tensors`,
  `_shard_cache_root`, `_quant_scope`, `_sha256_file`, `compile_shards`, `verify_shard_cache`,
  `shard_cache_status`, `cache_unit_path`) moved out of shards.py into a **SHARED** leaf (both fleets'
  `EXTRA_UPDATE_FILES`), leaving shards.py a pure weight-serving/streaming layer (1,321 → 872).
  Bind-free by requirement — the `/compile_shards` subprocess imports it in a fresh interpreter — with
  the shared read/dequant/skeleton helpers (and `INT4_GROUP`, a def-time default arg) imported *from*
  shards. Every consumer repointed, including the three a naive grep misses: engine_load's aliased
  `import shards as _sh` (whose failure the non-fatal precompile try/except would have swallowed into
  a silent fleet-wide cache-on-first-load regression), the `/compile_shards` subprocess code-string,
  and the worker remote-pack handler (shipped atomically with the client VERSION bump). Validated the
  hard way: cache deleted and recompiled through the new path — **combined sha1 of all 26 units
  bit-identical to the pre-split cache** — plus a live `/pack_probe` (worker packs via the relocated
  shared packer: `byte_identical: true`). client.py 2,923 → 2,924 wiring net; fleet on m4c186.
- **gfx1151 int4 GEMV DRAM de-aliasing + 70B-shape autotune coverage (#dram-dealias, 2026-07-07):**
  `llama-3.3:70b` int4 decoded at 0.61 tok/s on Strix Halo while dense-32B hit 5.28 — a 4x per-BYTE
  gap, fully reproduced in an isolated kernel bench (per-shape times predicted 0.73 vs observed 0.61).
  Root cause: the split-K w4a16 GEMV walks `qweight` along N with a row stride of K/2 bytes; at the
  70B dims (K=8192 -> a 4096B power-of-two stride) every row maps to the same DRAM channels/banks,
  and any matrix too big for the 32MB MALL collapses to 17-67 GB/s (the 33MB q/o just overflows it;
  the 4MB k/v stays cached and fast — why only big-K dense models ever showed this). Two-part fix,
  no kernel change: (1) `prepare_fused` re-allocates the packed rows on an ODD multiple of 64B
  (kernels already read via `qweight.stride(0)`; +64B/row ≈ 1%, and the aligned 32B shapes got
  FASTER too); (2) the GEMV autotune space adds the `BN=64 / num_warps=16` family the de-aliased
  70B shapes want (28672x8192 gate/up: 1.94ms -> 0.67ms). Matrix-bench ceilings: 70B 0.75 -> 4.50
  tok/s, 32B 6.66 -> 10.81 tok/s.
- **gfx1151 fused-MoE expert-row de-aliasing — MEASURED per shape (#dram-dealias MoE, m4c188,
  2026-07-07):** the fused grouped MoE GEMV has the same exposure as the dense one — within-expert
  rows sit K_pad/2 bytes apart in the contiguous `[E, N, rs]` `Packed4Tensor3D`, and a layer's
  expert stack (134-280 MB) is far past the 32MB MALL, so decode reads are DRAM-cold. But the
  isolated-shape bench (bench_moe_dealias, the dense fix's methodology with a fresh random expert
  subset per call) showed the response is NOT the dense static rule: gemma-4-26b's gate_up
  (rs=1408B, even*64) collapses to 63.7 GB/s and row-padding restores 187.5 GB/s (2.9x; per-token
  expert kernels 9.62 -> 4.45 ms), yet qwen3.6-35b's power-of-two shapes (rs=1024B/256B) run ~96
  GB/s unpadded and padding HALVES them — an even-multiple pad control shows the same, so it's not
  the odd/even story dense followed. Fix accordingly: `Packed4Tensor3D.prepare_fused` (same
  post-placement sweep as the dense pad, ROCm-only) TIMES the production op unpadded vs row-padded
  (`[E,N,rs+64]` buffer kept as the `[:,:,:rs]` view) on DRAM-cold subsets and keeps the winner
  (>=15% to pad; decision cached per (E,N,rs) so 30 layers pay one bench; kernels read via
  `.stride()` so the strided view needs no kernel change). `sqn` joins the MoE autotune key so the
  two variants tune apart (side effect: MoE decode autotune now happens at load, not first decode),
  and the de-aliased gemma gate_up's preferred `BN=256/SPLITK=4/w8` config joins the space (+8%).
  Shard caches stay bit-identical — pad at load, never at pack time. gpt-oss is naturally odd-row
  (rs=1472B) and skips untouched. LIVE-VALIDATED on om3nbox — and the live tensors INVERTED the
  synthetic verdicts, proving the measured design necessary: the collapse is ALLOCATION-dependent
  (physical-page bits in the channel hash), not a shape rule. Live decisions: qwen3.6-35b gate_up
  PAD (0.088 -> 0.052 ms, 1.7x; e2e 15.75 -> 16.34 tok/s clean A/B), its down + both gemma shapes
  keep-unpadded (gemma's live tensors never collapsed; e2e ~20.5 tok/s unchanged). fused-MoE
  self-checks all `-> ACTIVE` (rel ~0.006), gemma `/api/chat` coherent, qwen output text identical
  to baseline at temp 0.
- **Code-split round 2, increment 10 — `worker_quant.py`, client.py's flagship (m4c189, 2026-07-08):**
  the whole quant/kernel family (~1,660 lines) relocated out of client.py into a SELF-CONTAINED worker
  leaf (shard_compile precedent — deliberately NOT in `state.bind`): the guarded module-level triton/tl
  import + the #triton-race Autotuner patch, the CPU fp32-GEMM family (flags + `tune_cpu_threads` +
  `_accelerate_cpu_linears`), the int8/int4 cores (QuantLinear/QuantLinear4 incl. `prepare_fused` and
  both #dram-dealias paths), the w4a16 triton kernels (dense + fused-MoE + expert tensor-subclass),
  the packers (`Packed4Tensor3D`, `_pack4_expert`/`_pack4_3d`), fused-MoE + gpt-oss installs, the MoE
  offload bridge, per-expert/streamed builds, meta-expert detectors, and `_assign_meta_from_sd`. All
  relocated bodies byte-identical (verified against git HEAD). worker_quant.py is now the CANONICAL
  home of the runtime-rebound flag family (`_CPU_FP32_GEMM`/`_CPU_FP32_MIN_ROWS`/`_CPU_BF16_GEMM_OK`/
  `_FUSED_INT4`): client.py back-imports only the functions/classes (never the flags) before
  `state.publish` so shard_build's bind-injected bare names keep resolving, and the two flagged
  NOT-byte-identical edits make every flag access a live module attribute — `main()`'s `--no-cpu-fp32`
  sets `worker_quant._CPU_FP32_GEMM = False` (was `global`), and `Shard._finalize_placement` reads
  `worker_quant._CPU_FP32_GEMM` / `._FUSED_INT4` (a from-import copy would freeze the pre-main values
  and silently ignore the CLI flag / the aarch64 FEAT_BF16 crash-guard rebind). Zero leftover
  flag/triton/tl definitions remain in client.py (they would re-enter the publish snapshot and stomp
  the live values). worker_quant.py joins `EXTRA_UPDATE_FILES` + the convergence-bridge tuple;
  bit-identity doc contracts in shard_compile.py/shards.py/bench_moe_w4a16.py repointed.
  client.py 3,032 → 1,346.
- **Code-split round 2, increment 11 — `media_encode.py` + EngineSpeechMixin, server.py's flagship
  (2026-07-08):** the media/speech encode family relocated verbatim into a new bound controller leaf:
  `_encode_images`, `_encode_audio_gemma4`, `_encode_audio`, the #P6 speech-out group (`_SPEECH_CACHE`/
  `_SPEECH_MAT`/`_ensure_spk_dict`/`_materialize_from_prefix`/`SPEECH_DEVICE`/`_load_speech_components`),
  and `Engine.generate_speech` as **EngineSpeechMixin** (Engine now composes four mixins; only `__init__`
  remains on the shell). **`ENCODING`'s canonical home moved with them** — this supersedes the "stays in
  server.py" rationale recorded at m4c152/Inc 5: all FOUR `global ENCODING` mutators live in
  media_encode.py, the self-updater's idle lambda reads `media_encode.ENCODING` as a live module
  attribute, and ENCODING is never back-imported or published (an int snapshot would freeze the idle
  gate open — the original "ENCODING hazard", now closed by moving definition + mutators as one unit;
  state.py's SAFETY NOTE documents both valid patterns). Relocated bodies byte-identical (verified
  against git HEAD); the hazard comments in state.py/multimodal.py/model_store.py/downloads.py
  repointed. server.py 3,152 → 2,551.
