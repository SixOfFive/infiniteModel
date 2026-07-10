# ∞ InfiniteModel

**Run a large language model that's too big for any single machine — by splitting it across the
computers you already have.**

InfiniteModel is a from-scratch **distributed LLM inference engine**. One **controller** machine
pools the memory and compute of a fleet of **worker** machines (any mix of GPU and CPU boxes,
Windows or Linux) and splits a single transformer model's layers across them over a hand-rolled
plain-TCP transport. It's built on Hugging Face `transformers` + plain PyTorch — no vLLM, TGI, Ray,
or `torch.distributed` — so the **same code runs on Windows and Linux**.

It speaks **Ollama-, OpenAI-, and Anthropic-compatible** HTTP APIs, so tools you already use
(Ollama clients, OpenAI SDKs, Claude Code) can point at the cluster unchanged, plus a live web
dashboard.

![InfiniteModel dashboard — three models resident across a 12-node fleet](docs/dashboard.png)

> Personal research project — expect rough edges. Hugging Face **safetensors** models only (no GGUF).

> **A note from the author:** I'm fairly new to actually *using* git, so please go easy on me if the
> history or workflow isn't textbook — I'm still learning the etiquette. I built this for my own
> homelab and figured there might be demand for something like it out there, so I'm putting it up in
> case it's useful to someone else. Issues, suggestions, and patient corrections are all welcome.

---

## What it does

- **Fits big models by pooling machines.** Pipeline (layer-split) parallelism over plain TCP: each
  worker holds a contiguous block of layers; add machines to fit bigger models.
- **Goes faster where it can.** Tensor parallelism within a stage (capacity-proportional, GPU+CPU
  mixed meshes) and opt-in speculative decoding.
- **Quantization.** int4 (group-wise, fused tinygemm GEMM), int8 (per-channel), and **int2**
  (group-wise ~2.5-bit, group 64 — a *capacity* tier for dense models that won't fit at int4;
  MoE auto-downgrades to int4. ⚠ the current round-to-nearest packer **collapses model quality
  at 2 bits** — the tier is complete infrastructure awaiting a GPTQ-class calibrated packer,
  which slots into the same format/kernels/cache) at load time;
  serves fp8 and nvfp4 checkpoints by dequantizing on the fly. Decode-kernel acceleration is
  **platform-tiered** — torch tinygemm int4 on NVIDIA/CPU, a Triton w4a16 + split-K kernel on **AMD
  GPUs (ROCm/RDNA)**, a Triton **w2a16** kernel for int2 on both GPU stacks, and an **opt-in fused
  MoE-expert kernel on Linux+NVIDIA**
  ([docs/ACCELERATION.md](docs/ACCELERATION.md)). Runs on AMD **1:1 with CUDA via HIP**
  ([docs/ROCM.md](docs/ROCM.md)).
- **Pre-compiled shard cache.** The controller quantizes a model once to `_shards/<quant>/`, so later
  loads stream small **pre-packed** int4/int2/int8 layers instead of bf16 + re-quantizing — for dense
  models *and* MoE (int4: fused-3D and per-expert Mixtral/OLMoE), bit-identical to a cold load. Any model
  without an int4 cache shows a one-click **`⚡ int4` compile badge** on the models page (hover for
  the estimated on-disk size and free-disk check); compiles run in a background subprocess with live
  progress on the model's row.
- **MoE & multimodal.** Mixture-of-Experts (incl. attention-on-GPU / experts-in-CPU-RAM offload), and
  distributed vision + audio (Qwen2.5-Omni, Qwen2.5-VL, Qwen3.6, Mistral3): image/audio → text.
- **Tool calling.** Native `tools` on all three chat APIs — Ollama `/api/chat` (`tool_calls` with
  object args), OpenAI `/v1/chat/completions` (`tool_calls` + `finish_reason:"tool_calls"`), and
  Anthropic `/v1/messages` (`tool_use` blocks) — streaming and non-streaming, including the full
  reply loop (assistant `tool_calls` turns + `role:"tool"` results in either dialect's shape).
  Tool defs render through the model's chat template (text-instruction fallback for templates
  without native tool support); `tool_choice` honored (`none` / forced function best-effort);
  per-model support surfaces as a `tools` capability in `/api/show`.
- **API compatibility extras.** JSON mode (Ollama `format:"json"`/schema, OpenAI
  `response_format` — best-effort instruction + fence-stripping), OpenAI text-part list content,
  Ollama-native per-message `images:[b64]` and `/api/generate` top-level `images`.
- **Multi-model & ops.** N models resident at once, node-sharing, concurrency + queueing,
  auto-load/unload, a live dashboard (placement preview, per-load progress, fleet memory/throughput,
  bandwidth), curl-able fleet logs, and idle-gated self-update. **Idle unload** (settings page /
  `/config?idle_unload_m=`): unload any model that served no requests for N minutes — default 0 =
  every model stays loaded forever (`-1`, the Ollama-style spelling, is accepted and saves as -1
  with the same meaning); models with an active or queued request — and either **per-model lifecycle
  pin** — are never idle-unloaded. Those pins live on the model-detail modal: **Autoload on restart**
  re-streams a model to its workers on controller startup so it survives a restart/redeploy, and
  **Do not auto-unload** is an absolute veto (never reclaimed by idle-unload *or* by LRU eviction — a
  new load that can't otherwise fit fails instead). **Juggler** (settings page, opt-in): on a ~60 s
  sweep — and right after an idle-unload frees VRAM — the hottest model still running split across
  GPU+RAM *that would now fit entirely on GPU* — and is momentarily idle — is *promoted* to VRAM-only
  by a **hitless** re-place: new requests briefly pause on their still-open connection (no reconnect)
  while it re-places VRAM-first, then resume on the faster copy. A busy model is skipped (a later sweep
  catches it at a gap) rather than stalled; embeddings and models too big for GPU are skipped; a
  do-not-auto-unload model is promoted too, since the reload is a better placement, never a removal
  (and it's never left unloaded).
  **Autostart delay** (`autostart_delay_s`, default 60 s) holds the startup reload of persisted
  models that long so clients reconnect first. **Honest overload behavior:** under GPU contention the endpoint degrades into
  *retryable* backpressure, not failures — slow-but-advancing prefills are never reclaimed as wedged
  (workers report per-layer forward progress over their heartbeat), the prefill wait extends while
  progress advances, and contention-class failures return `503 + Retry-After` (Ollama/OpenAI) or
  `529 overloaded_error` (Anthropic) instead of bare 500s or dropped sockets. Per-load knobs: KV-cache placement
  (**GPU or system RAM** — offloading frees the VRAM KV reserve for model layers, for long context
  on small cards) and **per-model default temperature + min-p** (used when a request doesn't send
  its own). **Min-p sampling** is supported per-request too (Ollama `options.min_p`, top-level
  `min_p` on the OpenAI/Anthropic endpoints): tokens below `min_p` × the top token's probability
  are dropped — a confidence-adaptive floor that pairs well with high temperature.
- **The full sampling-knob family, per-request and runtime-tunable.** `top_p`, `top_k`,
  `repeat_penalty` (+ `repeat_last_n` window), `presence_penalty`, `frequency_penalty` and
  `seed` (reproducible sampling) work per-request on all three APIs (Ollama `options.*`;
  top-level on OpenAI/Anthropic, `repetition_penalty` accepted as an alias). Every knob — plus
  a default `num_predict` for requests that send no length cap — is also a **runtime-mutable
  per-model default**: `POST /model_config?model=...&top_k=40&repeat_penalty=1.1...` applies
  instantly to a loaded model (empty string clears; explicit request values always win), and the
  dashboard's model-detail **Runtime settings** panel edits all of them with suggested-value
  dropdowns.

## How it works

```
            ┌──────────── controller (server.py) ────────────┐
 client ──▶ │  HTTP API (Ollama / OpenAI / Anthropic) + UI    │
            │  holds the weights · planner splits into stages │
            └──────┬────────────────┬────────────────┬────────┘
           control │           data │ (plain TCP)     │
                   ▼                ▼                  ▼
             worker (client.py) … worker … worker            (GPU and/or CPU)
             layers [0,a)         [a,b)     [b,L)
```

The controller downloads the model once, plans placement (GPU-first, spill to CPU/RAM; or
tensor-parallel), and streams each stage's weights to its worker straight into RAM (workers keep no
model on disk). Generation flows around the ring `controller → stage0 → … → head → controller`.

## Project layout

Each role keeps a single entry point — **`server.py`** (controller) and **`client.py`** (worker) — but
the bulk of each is split into focused sibling modules, so any one subsystem fits a reader's (and an
editor's) context window. This is an internal refactor with **zero public-API change**; the fleet's
multi-file self-update keeps every module in lock-step across machines.

**Controller** — `server.py` is the `Engine` + FastAPI `build_app()` shell that wires the modules
together (via `state.py`):

- `engine_load.py` · `engine_gen.py` · `engine_lifecycle.py` — the `Engine` mixins: load / placement / TP, prefill / decode / speculative, and data-plane / recovery / unload.
- `routes_dashboard.py` · `routes_lifecycle.py` · `routes_api.py` · `routes_diag.py` — HTTP routes (UI + status, load / unload / compile, the inference APIs, multimodal test endpoints).
- `serving.py` — request serving (Ollama / OpenAI / Anthropic generate + chat); `status.py` — the `/status` + dashboard payload builders.
- `placement.py` — partition planner; `shards.py` — shard-cache compile / quant / dequant; `model_store.py` — model download / measure / storage.
- `formats.py` — prompt/response + tool-call formatting; `multimodal.py` — vision / audio / speech encoders; `graphs.py` + `dashboard_html.py` — the dashboard; `gguf_convert.py` — GGUF→safetensors (subprocess).

**Worker** — `client.py` is the `Shard` + `Worker` shell:

- `shard_build.py` · `shard_forward.py` — the `Shard` mixins: placement / streaming weight-load, and the forward path.
- `worker_load.py` · `worker_net.py` — the `Worker` mixins: build / load / pack / unload / TP, and next-hop connect / send + data-plane.

**Shared:** `state.py` (a namespace registry so relocated modules resolve their former globals without a circular `import server`), `wire.py` (plain-TCP transport primitives), and `config.json` (hosts / ports + self-update source).

---

## Installation

**The server and the worker need different dependencies** — install only what each machine's role
requires. Both need Python **3.13**; a CUDA GPU is optional (CPU-only workers are fully supported).

Pinned, proven versions live in [`requirements.txt`](requirements.txt) (server) and
[`install/requirements-client.txt`](install/requirements-client.txt) (worker).

### Server (controller) — `server.py`

The controller serves the HTTP API + dashboard, stores the weights, and compiles/quantizes shards,
so it needs the web framework **and** the model stack:

```bash
# core
pip install fastapi uvicorn torch transformers safetensors huggingface_hub numpy psutil

# optional — CONTROLLER-side only, and only if you serve multimodal models
pip install pillow          # images (ALL vision models) — required for ANY image input
pip install soundfile       # audio-in: WAV/FLAC/OGG (also needs the libsndfile system lib)
pip install librosa         # audio-in: adds mp3 + high-quality resample (heavier — pulls numba)
```

- **Multimodal deps are controller-side** — images/audio are decoded + preprocessed on the controller,
  not the workers — and **`transformers` caches the "is Pillow / torchvision available?" check at
  import**, so install these *before* starting the controller (or **restart it** after, else vision
  silently `ImportError`s even once the package is present). Vision needs **no torchvision**: every image
  processor here has a pure-PIL backend, and installing torchvision risks pip pulling a `torch` that
  doesn't match your node's pinned build. `soundfile` covers WAV in/out; without any of these, audio-in
  still handles PCM WAV via the stdlib `wave` fallback and speech-out (TTS) writes WAV the same way.
- Run the controller on the machine with the most disk + RAM (it holds every model's weights).
- `torch`: install the build matching that box — the default **CUDA** wheel on an NVIDIA box, or the
  **CPU** wheel otherwise (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). On an
  **AMD** box, ROCm is a **separate setup** — see **[docs/ROCM.md](docs/ROCM.md)**, not these wheels.

### Worker (client) — `client.py`

A worker only executes layers and talks to the controller over TCP — it runs **no HTTP server**, so
it does **not** need `fastapi`/`uvicorn` (a much lighter, different dependency set). **Pick the one
path that matches the worker's hardware:**

**① NVIDIA GPU or CPU worker** (the CUDA/CPU path):

```bash
# core — NVIDIA: the default CUDA torch wheel; CPU-only: add the cpu index-url to torch (below)
pip install torch transformers safetensors huggingface_hub numpy psutil
pip install einops                 # some models' trust_remote_code (e.g. nomic-embed-text)
pip install nvidia-ml-py           # optional, NVIDIA GPU nodes only (VRAM reporting)
```

- **CPU-only:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **NVIDIA GPU:** the default CUDA `torch` wheel (above). Dense int4 decode is fast out of the box
  (torch tinygemm). The **fused-MoE acceleration tier** for routed experts is an *optional opt-in*
  (Ampere/sm_80+) — extra build toolchain + `INFINITEMODEL_CUDA_FUSED_MOE=1`; recipe (incl. the
  Windows + CUDA-Toolkit + triton-windows setup and the WDDM interactive-session caveat) in
  **[docs/ACCELERATION.md](docs/ACCELERATION.md)**.

**② AMD GPU (ROCm) worker — a SEPARATE setup; do _not_ use the CUDA/CPU wheels above.** InfiniteModel
runs **1:1 with CUDA** on AMD via PyTorch's HIP (the device stays `cuda:N` and the inference code is
unchanged), but it needs a ROCm `torch` matched to your GPU arch — for **Strix Halo / RDNA** use AMD's
arch-specific *TheRock* wheels (the generic ROCm wheels can crash on new chips/kernels), plus a Triton
w4a16 int4 kernel for fast int4 decode on RDNA. The helper **[`install-rocm.sh`](install-rocm.sh)**
builds the entire venv in one step. **Full, self-contained guide → [docs/ROCM.md](docs/ROCM.md).**

- **Offline / pinned install (NVIDIA/CPU):** [`install/`](install/) has `install.sh` / `install.bat` that build a
  self-contained venv from `install/requirements-client.txt` (drop your own wheels into
  `install/wheels/` for a fully offline build).

---

## Usage

**1. Configure** `config.json` (the single source of truth for hosts/ports — built-in defaults apply
if it's absent):

```json
{ "controller_host": "10.0.0.5", "http_port": 21434, "control_port": 50100, "data_port": 50101 }
```

**2. Start the controller** (on the box that holds the weights):

```bash
python server.py            # Windows: server.bat
```

Open the dashboard at `http://<controller>:21434/`.

**3. Start a worker on each other machine:**

```bash
./client.sh --device cpu+gpu          # Linux  (Windows: client.bat)
./client.sh --controller 10.0.0.5     # override the controller if it's not in config.json
```

Each worker registers within a couple seconds and appears on the dashboard.

**4. Load a model across the fleet and use it** — from the dashboard, or via the API:

```bash
# plan + distribute (quant optional: int4 / int8)
curl -X POST "http://<controller>:21434/load?model=qwen2.5-0.5b&ctx=2048&quant=int4"

# generate (Ollama-style; the first request for a model also auto-loads it)
curl -X POST http://<controller>:21434/api/generate \
     -d '{"model":"qwen2.5-0.5b","prompt":"The capital of France is","stream":false}'
```

**API surface:** Ollama (`/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/pull`, …),
OpenAI (`/v1/chat/completions`, `/v1/models`), and Anthropic (`/v1/messages`). Point existing tooling
at `http://<controller>:21434`.

## Configuration & secrets

- **`config.json`** — cluster hosts/ports + self-update source (`update_repo`/`update_branch`); the one
  place to edit, no addresses baked into code.
- **`hf_token.txt`** or `$HF_TOKEN` — Hugging Face token for gated/authenticated pulls (gitignored).

Self-update pulls module sources from the public GitHub repo's raw endpoint — **no token needed**. No
secrets are stored in the source.

## Acknowledgments

Inspired in spirit by [exo](https://github.com/exo-explore/exo) and the broader idea of pooling the
everyday machines you already own to run models no single one could hold. InfiniteModel is an
independent, from-scratch implementation — its own plain-TCP pipeline/tensor-parallel transport,
planner, and quantization, no exo code — but exo helped convince me this was worth building.

## License

[MIT](LICENSE)
