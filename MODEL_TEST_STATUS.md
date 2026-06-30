# Model test status

Tracks which registered models have been **validated** on this fleet and to what depth. Update this as
new tests run. (Seeded from session testing + decision history; correct any miscategorized rows.)

**Legend**
- ✅ **Fully tested** — loads **and** produces coherent output, validated end-to-end (quant/path noted).
- 🟡 **Partially tested** — a capability is proven, but not full end-to-end, or there's a known caveat.
- ⬜ **Unverified** — registered but not validated here (no recent test evidence).

| Model | Status | Validated (quant / path) | Notes |
|---|---|---|---|
| `qwen3:4b` | ✅ | int4, bf16 — load + generate | Primary always-resident model; default test load is `int4 ctx=8192 auto`. |
| `qwen2.5:0.5b` | ✅ | int4 (shard-cache serve) — generate | Dense; serve-from-cache validated (gens "Paris"). |
| `qwen2.5:1.5b` | ✅ | bf16 + TP — generate | Used to validate TP mesh keepalive (tp2, gens over long idle). |
| `qwen2.5:7b` | ✅ | int4, bf16 — generate | int4 fused tinygemm ~21 tok/s; TP tested. |
| `qwen2.5:14b` (`-instruct`) | ✅ | bf16, int4 — load + generate | Coexistence + gen-stall watchdog validated (400-tok gen). |
| `ministral-3:14b` | ✅ text + **vision** | int4 — load + generate; **vision image→text** | #117 fixed prefill hang (beast→theocomp hop). **Pixtral vision validated END-TO-END (m4c161, 2026-06-29)** via the same Mistral3 path as devstral (chat_template.jinja auto-topped-up on load); image→text answered correctly: "The image contains a red circle and a blue square." |
| `devstral-small-2:24b` | ✅ text + **vision** | int4 — load + generate; **vision image→text** | #83/#89 fixed Mistral3 garbage + immediate-EOS. **Pixtral vision validated END-TO-END (m4c159–161, 2026-06-29)**: split tower (`vision_tower.*` + `multi_modal_projector.*`) materialized from the RAW checkpoint prefixes, 2D rotary rebuilt via the module's own rope-init, `get_image_features(pixel_values, image_sizes)` at the merged 28px grid, embeds spliced at `[IMG]` (id 10) with plain 1D positions. Needed `chat_template.jinja` topped-up into the model dir (Mistral ships it as a separate file, not in tokenizer_config) so the native `<s>[INST][IMG]…[/INST]` renders. image→text answered correctly: "The image contains a red circle and a blue square." |
| `deepseek-r1-distill-llama:70b` | ✅ | int4 — generate | Dense Llama; int4 decode tok/s re-measured (#73). |
| `mixtral:8x7b` | ✅ | int4 — load + generate; **int4 shard cache** | Fused-3D MoE (per-expert ckpt fused at build); non-fused-compile validated. |
| `olmoe:1b-7b` | ✅ | int4 — load + generate; **int4 shard cache** | Per-expert ckpt fused to 3D; serve-from-cache gens "Paris". |
| `qwen3.6-35b-a3b` | ✅ text + **vision** | int4 — load + generate; **int4 shard cache** (cached==cold); **vision image→text** | qwen3_5_moe hybrid (Gated-DeltaNet). MTP self-spec investigated → not viable (#91). **Vision validated END-TO-END on the existing arch-general path with NO code change** (2026-06-28): `/vision_test` → [100,2048] merged tokens (Qwen3_5MoeVisionPatchMerger, image_token 248056); image→text correctly named shapes + **left/right positions** + colors (red circle / blue square), confirming the interleaved-mRoPE positions (`mrope_section [11,11,10]`, `partial_rotary_factor 0.25`) are correct via `_mrope_position_ids` + the worker's own rotary. 27b-nvfp4 sibling expected to work identically (same path). |
| `qwen2.5-omni:7b` | ✅ | distributed multimodal — image+audio→text, speech-out | #22/#35/#36/#37 (vision, audio-in, Talker/token2wav speech-out). |
| `nomic-embed-text` | ✅ | embeddings (`/api/embed`) | Encoder, not a causal-LM (#81). |
| `nvfp4-moe-e2e` (`nm-testing/nvfp4_moe-e2e`) | ✅ | int4 (from nvfp4 source) — **compile + serve-from-cache + generate** | Qwen3-MoE 128-expert per-expert nvfp4 (16.86 GB) → 15.98 GB int4 cache → gens coherently. Validated the per-expert nvfp4 MoE compile fix (m4c132). |
| `qwen2.5-0.5b-gguf` (`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, `q4_k_m`) | ✅ | **GGUF → safetensors normalize** → int4 — load + generate | Validates GGUF ingestion (m4c137–m4c144): `.gguf` dequantized to a safetensors checkpoint at add-time, then served as an ordinary int4 model (gens "France's capital is Paris."). |
| `qwen3.6-27b-nvfp4` | ✅ (text) | int4 (from nvfp4 source) — load + **generate** | Reasoning model (emits `<think>`). Loaded int4 auto (CPU-spilled alongside resident models) + **coherent gen confirmed** (m4c149: "The capital of France is Paris."). Multimodal path untested. |
| `minimax-m2-bf16` | 🟡 | int4 **compile fix proven** (gate clears, packs correctly) | Non-fused per-expert MoE (#119). **Never loaded/generated end-to-end** — CPU-bound/unusable here (<0.3 tok/s, 110 GB int4 vs 33.6 GB fleet VRAM); full compile+load deferred. |
| `qwythos:9b-abliterated` | ✅ | int4 — load + **generate** | qwen3_5; loaded int4 auto (ctx 16384) + **coherent gen confirmed** (m4c149: "The capital of France is Paris."). |
| `qwen2.5-coder:32b` | ✅ | int4 — load + **generate** | Verified 2026-06-30 (BEAST fleet sweep): loaded int4 ctx 16384, "The capital of France is Paris." (CPU-spilled, ~1 tok/s). |
| `nemotron:70b` | ⬜ | — | LOAD-FAIL 2026-06-30: "no capable nodes left after exclusions" after ~17 min of retries (placement/node-flakiness, not coherence — retry on a stable fleet). |
| `kimi-dev:72b` | ⬜ | — | LOAD-FAIL 2026-06-30: too big for the pool at int4 (needs ~56.9 GB, pool offers 46.8 GB usable; won't fit even at ctx 1024). Needs CPU-spill `distribute` mode, more nodes, or a smaller quant. |
| `coneml-348m-alpha-polish900` | ⬜ | — | Custom small model; not validated here. |
| `gemma-4:12b-it` | ✅ (text) | int4 — load + **generate** | `gemma4_unified` (no PLE, no KV-sharing). **Text validated (m4c163–164, 2026-06-29):** "The capital of France is Paris…". Needed two worker fixes — materialize Gemma's `embed_scale` buffer (else meta-guard trips at load), and **per-attention-type rotary** (sliding/full interleave: build cos/sin per layer_type + `shared_kv_states={}`, else `None_inv_freq` at gen). **Stop-token FIXED (#stop-eos, 2026-06-30):** `<end_of_turn>` (id 106) is now registered as a stop in `_eos_ids` — verified on BEAST: finish=stop, no `thought`/end-of-turn leak, no streaming repeat, and 4 concurrent gens each return clean non-empty content. Vision (unified/encoder-free) + audio untested. |
| `gemma-4:31b-it` | ✅ (text) | int4 — load + **generate** (multi-stage) | `gemma4` dense (60L, h5376), `gemma4_vision` tower. **Text validated (2026-06-29):** "The capital of France is Paris." — worked on the existing m4c163–164 fixes (embed_scale + per-type rotary) with no new code, including the multi-stage split (num_kv_shared_layers=0 → no cross-shard KV dependency). Stop-token tail FIXED (#stop-eos, 2026-06-30) — same `<end_of_turn>` stop registration as 12B. Vision untested. |
| `gemma-4-26b-a4b-it` | ✅ (text) | int4 **shard cache** (serve-from-cache) — load + **generate** | `gemma4` **MoE (128 experts, 30 layers)** + `gemma4_vision` tower. No PLE/KV-sharing. **Text validated (m4c167–169, 2026-06-29)** via `/api/chat`: "The capital of France is Paris." Needed, beyond the dense 12B/31B fixes (embed_scale + per-type rotary), TWO MoE-specific fixes: **(1)** materialize the router's `scale`/`per_expert_scale` nn.Parameters (init-ones, ABSENT from the checkpoint → meta-guard tripped); **(2)** NEVER int4-quantize the router gate (`Gemma4TextRouter.proj` is a plain nn.Linear → quantizing the 128-expert gate corrupted top-k routing → garbage; excluded in `_quant_scope` + `_quantize_int4_`, cache recompiled). Distributed compile via `/compile_dist` HUNG (worker per-expert pack future never returned); built the cache with the proven local `/compile_shards` instead. Stop-token tail FIXED (#stop-eos, 2026-06-30) — `<end_of_turn>` now a stop. Vision untested. |

## How to update
When a model is tested, set its row to ✅/🟡, note the **quant + path** (e.g. `int4 serve-from-cache`,
`bf16 distributed`, `embeddings`) and a one-line evidence note. "Fully tested" requires an actual
**load + coherent generation** (or the model's equivalent output, e.g. embeddings), not just a
successful load or compile. Keep the legend honest — a compiled int4 shard cache alone is 🟡 until a
cache-backed load generates correctly.
