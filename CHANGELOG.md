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
  Covers Devstral / Ministral. (`[IMG_BREAK]`/`[IMG_END]` row-structure tokens are still flat — a tracked
  refinement.)
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
  spliced with plain 1D positions. Multi-image attribution exact; works on all three APIs. Known gap:
  the reference runs bidirectional attention across each image block — we ship causal-first (minor
  location imprecision observed). Side-fix: gemma-4's `chat_template.jinja` was missing from the model
  dir, so even TEXT prompts had been served through the flat fallback — with it in place the native
  `<|turn>` form renders (and `<turn|>`=106 was already a registered stop).
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
  (embeddings), or a 📌 pin (persist_models) are never idle-unloaded, and the speech thinker
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
