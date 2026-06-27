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

> Personal research project — expect rough edges. Hugging Face **safetensors** models only (no GGUF).

---

## What it does

- **Fits big models by pooling machines.** Pipeline (layer-split) parallelism over plain TCP: each
  worker holds a contiguous block of layers; add machines to fit bigger models.
- **Goes faster where it can.** Tensor parallelism within a stage (capacity-proportional, GPU+CPU
  mixed meshes) and opt-in speculative decoding.
- **Quantization.** int4 (group-wise, fused tinygemm GEMM) and int8 (per-channel) at load time;
  serves fp8 and nvfp4 checkpoints by dequantizing on the fly.
- **Pre-compiled shard cache.** The controller quantizes a model once to `_shards/<quant>/`, so later
  loads stream small **pre-packed** int4/int8 layers instead of bf16 + re-quantizing — for dense
  models *and* MoE (fused-3D and per-expert Mixtral/OLMoE), bit-identical to a cold load.
- **MoE & multimodal.** Mixture-of-Experts (incl. attention-on-GPU / experts-in-CPU-RAM offload), and
  distributed vision + audio (Qwen2.5-Omni): image/audio → text.
- **Multi-model & ops.** N models resident at once, node-sharing, concurrency + queueing,
  auto-load/unload, a live dashboard (placement preview, per-load progress, fleet memory/throughput,
  bandwidth), curl-able fleet logs, and idle-gated self-update.

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

# optional — only if you serve multimodal models (vision / audio)
pip install pillow librosa soundfile
```

- Run the controller on the machine with the most disk + RAM (it holds every model's weights).
- `torch`: install the build matching that box — CUDA wheel if it has a GPU, otherwise the CPU wheel:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### Worker (client) — `client.py`

A worker only executes layers and talks to the controller over TCP — it runs **no HTTP server**, so
it does **not** need `fastapi`/`uvicorn` (a much lighter, different dependency set):

```bash
# core
pip install torch transformers safetensors huggingface_hub numpy psutil

# needed by some models' trust_remote_code (e.g. nomic-embed-text); optional VRAM reporting on CUDA nodes
pip install einops
pip install nvidia-ml-py            # optional, GPU nodes only
```

- **CPU-only worker:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **GPU worker:** install the default CUDA `torch` wheel instead.
- **Offline / pinned install:** [`install/`](install/) has `install.sh` / `install.bat` that build a
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

- **`config.json`** — cluster hosts/ports (the one place to edit; no addresses are baked into code).
- **`hf_token.txt`** or `$HF_TOKEN` — Hugging Face token for gated/authenticated pulls (gitignored).
- **`gitlab_token.txt`** or `$GITLAB_TOKEN` — optional read token if you self-update from a private
  git mirror (gitignored).

No secrets are stored in the source.

## License

[MIT](LICENSE)
