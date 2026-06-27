# ∞ InfiniteModel

Split a single LLM across several home machines so you can run a model too big
for any one box — and make that distributed run as fast as it can honestly go.

One **controller** machine exposes an **Ollama-compatible API** and a live
**dashboard**. **Worker** machines connect to it, contribute their RAM, and the
controller pools that memory to fit the model. As more workers join, the pool
grows (bigger models) and — where the network allows — speed improves.

> **CPU-only fleet** for now (no usable GPUs). The GPU code path exists and
> degrades gracefully to CPU; nothing here requires CUDA.

## The engine, honestly

Splitting buys **capacity** or **speed** depending on *how* you split:

- **Pipeline parallel (to FIT):** each node holds a contiguous block of layers.
  Pools RAM so a big model fits. Runs sequentially — adds capacity, not speed.
  Always works, any node count, heterogeneous, cross-platform.
- **Tensor parallel (to go FASTER):** extra nodes shard each stage's matmuls and
  compute in parallel. Real speedup, **but capped at ~2–4 nodes per stage** before
  the per-token all-reduce latency dominates — and pure-Python-over-TCP on CPU may
  not beat the baseline at all. We build it, then **measure** before claiming a win.
- **Speculative decoding (the reliable CPU speed lever):** a small draft model
  proposes K tokens; the big pipeline verifies all K in one pass. Cuts the number
  of expensive forward passes instead of fighting network overhead.

What crosses the wire per token is tiny (~10 KB hidden state), so **1 GbE
bandwidth is not the bottleneck — per-token sync latency is.** Use **wired**
links only; Wi-Fi jitter kills it.

Why from-scratch + HuggingFace? Because `torch.distributed.rpc` / PiPPy /
TensorPipe are Unix-only — they're what blocks HF distribution on Windows. By
rolling our own plain-TCP transport we run the **same code on Windows and Linux**;
only the cross-node wire is custom, while per-node compute stays standard `torch`.

## Roadmap

| Milestone | What lands | Status |
|-----------|-----------|--------|
| **M1**  | Node registry, heartbeat, CPU/RAM reporting, live dashboard | ✅ |
| **M2a** | RAM-weighted partition planner (`/plan`, `--self-test-plan`) | ✅ |
| **M2b** | Worker partial model load (owns only its layers; rest stay on `meta`) | ✅ |
| **M2c** | Networked pipeline generation + full Ollama API | ✅ |
| **M2d** | Chunk serving (RAM-bound workers, no model on disk) + throughput/traffic metrics | ✅ |
| **M2e** | Incremental KV-cache decode (prefill-once, O(n) not O(n²)) | ✅ |
| **M3**  | Speculative decoding — greedy-exact, **opt-in** (`options.speculative=true`) | ✅* |
| **M4**  | Tensor-parallel within stages (node-scaling speed) → measure vs M2/M3 | ☐ |
| **M5**  | Dynamic re-plan on join/leave; KV-cache quantization (q8_0) for longer context | ☐ |

> **\*M3 is implemented and bit-exact** (a draft model on the controller proposes K
> tokens; the pipeline verifies them in one traversal; KV cache rolls back on
> mismatch). It's **opt-in and conditional** — and we measured the crossover on a
> real fleet:
>
> | Target (5-node CPU fleet) | M2e plain | M3 speculative | Verdict |
> |---|---|---|---|
> | 1.5B (+0.5B draft, 2 nodes) | 5.44 tok/s | 2.25 tok/s | plain wins |
> | **7B (+0.5B draft, 5 nodes)** | **1.08 tok/s** | **1.18 tok/s** | **spec wins ~1.09×** |
>
> On small targets the per-traversal cost is low, so the local draft's K forward
> passes + the re-establish traversal dominate. The win appears once the target's
> per-token pipeline traversal cost outgrows the draft cost (big model / many nodes)
> and widens from there — exactly the predicted crossover. So spec stays opt-in and
> never replaces the fast M2e default. (The ONE LAW's "measure before claiming,"
> realized.)

> **Decode is incremental** (M2e): each stage keeps a per-generation KV cache, so a
> decode step processes only the new token. Measured on a 5-node CPU fleet: 0.5B went
> from ~1.6 tok/s (O(n²) repeated-prefill) to ~5.6 tok/s with flat per-token time.
> Output is bit-identical to HF pure-greedy. Next speed lever is **M3 speculative
> decoding** (a small draft model proposes K tokens, the pipeline verifies them in one
> traversal — cuts the network-bound traversals that now dominate).

A node leaving mid-run tears the cluster down and requires a reload — by design;
membership is frozen at model-load time (no fragile mid-inference re-sharding).

**Weights are chunk-served (M2d).** The controller downloads the full model once
and serves each worker only its layer tensors (`/weights`); the worker loads them
straight into RAM. Workers keep **no model on disk**, so the model-size ceiling is
`min(RAM pool to run it, controller disk to hold it)` — never the smallest worker
drive. Workers also purge stale model/chunk caches on startup. The dashboard shows
the largest servable model, live tokens/s + API bytes in/out, and per-node
network traffic in/out (10 s rolling).

> **Don't run a cleaning worker on the controller box.** A `client.py` started on
> the controller machine without `--no-clean` will purge the controller's HF cache
> on startup and wipe every downloaded model. The controller itself never deletes.

**Model store + network metering.** The controller is the single model store and
**never auto-purges** — download several models, switch between them, and come back
to an older one without re-downloading. Only models whose weights are actually
present are reported as *available* (`/api/tags`, `/v1/models`, `/api/show`, and the
dashboard) — a model that isn't downloaded can't be distributed, so it isn't listed.
Manage models from the dashboard (or `POST /download?model=` / `POST /delete?model=`
/ Ollama `/api/pull` + `/api/delete`); deleting a loaded model is refused. **Network
traffic is metered by the controller itself** — it counts the bytes on its own
sockets (control plane, the frame to stage 0, the logits from the head, and weight
serving) rather than trusting client self-reports. Because the data path is a ring
`controller → stage0 → … → head → controller`, the controller is only physically on
the first and last hops; mid-pipeline stages exchange hidden states node-to-node,
off the controller, so during decode they show only their control-plane bytes here.

## Quick start

**Controller** (the machine the Ollama API is served from):

```powershell
pip install fastapi uvicorn torch transformers safetensors huggingface_hub
python server.py                 # dashboard + Ollama API :11434, control :50100, data :50101
```

Open the dashboard at `http://<controller>:11434/` — it has a load/unload +
test-generate panel.

**Each worker:**

```bash
pip install psutil torch transformers safetensors huggingface_hub  # torch CPU: --index-url https://download.pytorch.org/whl/cpu
python client.py --controller <controller-ip>
```

It registers within ~2 s and appears on the dashboard. Then load a model across
the pool and use it like any Ollama endpoint:

```bash
curl -X POST "http://<controller>:11434/load?model=qwen2.5-0.5b&ctx=2048"   # plan + distribute
curl -X POST http://<controller>:11434/api/generate \
     -d '{"model":"qwen2.5-0.5b","prompt":"The capital of France is","stream":false}'
```

`load` downloads the model on each worker (HF cache) and loads only that node's
layer slice. The first `/api/generate` for a model also auto-loads it.

### Ollama API surface

`/api/version`, `/api/tags`, `/api/show`, `/api/ps`, `/api/generate`, `/api/chat`
(streaming NDJSON + non-stream, full duration/eval fields), plus OpenAI-compat
`/v1/models` and `/v1/chat/completions`. Point existing Ollama tooling at the port.

### Useful flags

```
server.py  --host 0.0.0.0 --http-port 11434 --control-port 50100 --data-port 50101 --os-reserve-gb 2.0
           --self-test-plan --fleet 16,8,16,32     # offline planner demo, no cluster needed
client.py  --controller HOST --control-port 50100 --data-port 50200 --os-reserve-gb 2.0 --name LABEL
           --self-test-load --model Qwen/Qwen2.5-0.5B-Instruct   # local load/correctness check
```

## Layout

```
server.py         controller: registry + planner + control/data plane + Ollama API + dashboard
client.py         worker: capability probe + registry + partial model load + stage execution
requirements.txt  dependency notes (install per-role)
```

## Models

Configured target/draft pairs (draft + target share a tokenizer, as speculative
decoding requires):

- `qwen2.5-0.5b`, `qwen2.5-1.5b` — fast proof/iteration
- `qwen2.5-7b` — fits ~3 home boxes' pooled RAM
- `qwen2.5-coder-32b` — the real ~24 GB+ goal (needs a larger pool)

Edit the `MODELS` map in `server.py` to add more (Llama / Qwen2.5 / Mistral-family
decoders are supported first).
