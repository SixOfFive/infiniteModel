# ∞ InfiniteModel

A from-scratch **distributed LLM inference** engine: a single **controller** splits one
transformer model's layers across a **heterogeneous fleet** of machines (any mix of GPU and
CPU nodes) over a hand-rolled plain-TCP transport — so you can run a model too big for any one
box. Built on Hugging Face `transformers` + plain PyTorch (no vLLM/TGI/Ray/`torch.distributed`).

It exposes **Ollama-, OpenAI-, and Anthropic-compatible** HTTP APIs, so existing clients
(Ollama tooling, OpenAI SDKs, Claude Code) can talk to the cluster unchanged, plus a live
dashboard.

> Personal research project — expect rough edges. Hugging Face **safetensors** models only (no GGUF).

## Why from-scratch + Hugging Face?

`torch.distributed` / PiPPy / TensorPipe are effectively Unix-only — that's what blocks HF
distribution on Windows. By rolling our own plain-TCP transport, the **same code runs on Windows
and Linux**; only the cross-node wire is custom, while per-node compute stays standard `torch`.
What crosses the wire per token is tiny (~KBs of hidden state), so **1 GbE is not the bottleneck —
per-token sync latency is.** Use wired links.

## How it splits

- **Pipeline parallel (to FIT):** each node holds a contiguous block of layers; pools memory so a
  big model fits. Always works — any node count, heterogeneous, cross-platform.
- **Tensor parallel (to go FASTER):** shard each stage's matmuls across nodes (capacity-proportional,
  GPU+CPU mixed). Real speedup, but capped by per-token all-reduce latency — so it's measured, not assumed.
- **Speculative decoding (opt-in):** a small draft model proposes K tokens; the pipeline verifies them
  in one traversal — cuts the network-bound traversals on big/distributed targets.

## Features

- **Quantization:** int4 (group-wise, fused tinygemm GEMM) + int8 (per-channel); serves fp8- and
  nvfp4-checkpoints by dequantizing at serve time.
- **Pre-compiled shard cache:** the controller quantizes a model once to `_shards/<quant>/`; loads
  then stream small **pre-packed** int4/int8 layers instead of bf16 + re-quantize — including **MoE**
  (fused-3D *and* per-expert Mixtral/OLMoE, fused at compile, bit-identical to a cold load).
- **MoE:** fused + non-fused experts; optional intra-layer offload (attention on GPU, routed experts in CPU RAM).
- **Multi-model:** N models resident at once, node-sharing, concurrency + queueing, auto-load/unload.
- **Multimodal:** distributed vision + audio (Qwen2.5-Omni) — image/audio → text.
- **Ops:** live dashboard (placement preview, per-load progress, fleet memory/throughput, bandwidth),
  curl-able fleet logs, idle-gated self-update, RAM/VRAM safety (OOM-clean replans).

## Architecture

```
            ┌──────────── controller (server.py) ────────────┐
 client ──▶ │  HTTP API (Ollama / OpenAI / Anthropic) + UI    │
            │  planner → splits the model into layer stages   │
            └──────┬────────────────┬────────────────┬────────┘
           control │           data │ (plain TCP)     │
                   ▼                ▼                  ▼
             worker (client.py) … worker … worker            (GPU and/or CPU)
             layers [0,a)         [a,b)     [b,L)
```

The controller plans placement (GPU-first, spill to CPU/RAM; or tensor-parallel), streams each
stage's weights to its worker (straight into RAM — no temp files), then drives generation across the
ring `controller → stage0 → … → head → controller`. A node leaving mid-run triggers a clean replan.

## Quick start

1. **Configure** `config.json` — the single source of truth for hosts/ports (built-in defaults apply if absent):
   ```json
   { "controller_host": "10.0.0.5", "http_port": 21434, "control_port": 50100, "data_port": 50101 }
   ```
2. **Controller** (the box that holds the weights & serves the API):
   ```
   python server.py            # Windows: server.bat
   ```
3. **Each worker:**
   ```
   ./client.sh --device cpu+gpu          # Linux  (Windows: client.bat)
   ./client.sh --controller 10.0.0.5     # override the controller if not in config.json
   ```
   For a pinned, self-contained install see [`install/`](install/).
4. **Use it** — open `http://<controller>:21434/` for the dashboard, or point any Ollama/OpenAI/Anthropic client at that URL:
   ```
   curl -X POST "http://<controller>:21434/load?model=qwen2.5-0.5b&ctx=2048&quant=int4"
   curl -X POST  http://<controller>:21434/api/generate \
        -d '{"model":"qwen2.5-0.5b","prompt":"The capital of France is","stream":false}'
   ```
   `load` distributes the model across the fleet; the first request for a model also auto-loads it.

## Configuration & secrets

- **`config.json`** — cluster hosts/ports (the one place to edit; no addresses are baked into code).
- **`hf_token.txt`** or `$HF_TOKEN` — Hugging Face token for gated/authenticated pulls (gitignored).
- **`gitlab_token.txt`** or `$GITLAB_TOKEN` — optional read token if you self-update from a private git mirror (gitignored).

No secrets are stored in the source.

## Requirements

Python 3.13, PyTorch 2.12, `transformers` 5.x, `safetensors`, `huggingface_hub`, `psutil`, plus
`fastapi`/`uvicorn` on the controller (see [`requirements.txt`](requirements.txt)). A CUDA GPU is
optional — CPU-only nodes work and the GPU path degrades gracefully to CPU.

## License

[MIT](LICENSE)
