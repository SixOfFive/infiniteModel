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
- **Quantization.** int4 (group-wise, fused tinygemm GEMM) and int8 (per-channel) at load time;
  serves fp8 and nvfp4 checkpoints by dequantizing on the fly. Runs on **AMD GPUs (ROCm)** too —
  1:1 with CUDA via HIP, with a Triton w4a16 kernel for fast int4 on RDNA ([docs/ROCM.md](docs/ROCM.md)).
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
- **AMD GPU (ROCm) worker:** fully supported — InfiniteModel runs **1:1 with CUDA** via PyTorch's HIP
  (device stays `cuda:N`, same code path). Install a ROCm `torch` matched to your GPU arch — for AMD
  **Strix Halo / RDNA** use AMD's arch-specific *TheRock* wheels (the generic ROCm wheels can crash on
  new chips/kernels). The helper [`install-rocm.sh`](install-rocm.sh) builds the whole venv, and a
  **Triton w4a16 int4 kernel** keeps int4 decode fast on RDNA. Full guide: **[docs/ROCM.md](docs/ROCM.md)**.
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
