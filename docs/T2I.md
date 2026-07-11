# Text-to-image serving

InfiniteModel serves **diffusers-layout** text-to-image checkpoints (validated: **Qwen-Image**,
a 20B MMDiT — ~54 GB bf16 on disk) through the OpenAI Images API, on the same fleet that serves
the LLMs. This page is the full guide: getting the model, the two serve modes, the API contract,
and the operational behavior around renders.

---

## Architecture (what actually runs where)

Unlike LLM serving, a t2i pipeline is **not** layer-split across the fleet. The whole pipeline
runs on **one controller-co-located GPU worker** (co-location is by hostname match — the worker
process on the controller's own box):

- **DiT (the diffusion transformer)** — on the GPU, in one of the two modes below.
- **Text encoder** (Qwen-Image's is Qwen2.5-VL-7B, ~16 GB) — on the worker's **CPU**,
  encode-once per request; it never competes for VRAM.
- **VAE decode** — tiled on the GPU, with a CPU fallback if the decode tile OOMs.
- Renders report **per-step progress** over the control link — the dashboard model card shows
  `rendering step i/n` live.

## Getting the model

Diffusers repos are first-class downloads: **+ Add model** on the dashboard with the HF repo id
(e.g. `Qwen/Qwen-Image`) pulls the full multi-folder layout with normal download progress. The
models page shows a **🖼 t2i** badge; a load is refused until the download is verifiably
complete (`_diffusers_complete`), so a half-pulled pipeline can't be served.

**Worker dep:** the serving worker's venv needs `pip install diffusers` — plus `accelerate` for
the offload mode. (Controller boxes need neither unless they also host the serving worker.)

## The two serve modes

| | **GPU-resident** (default) | **Offload** (`t2i_offload=1`) |
|---|---|---|
| DiT placement | on the GPU, **mixed-edge int4** — first/last blocks bf16, the rest int4 (≈ bf16 quality; pure-RTN int4 visibly drifts) | **bf16 in system RAM**; blocks stream to the GPU per forward (`accelerate` sequential offload) |
| VRAM needed | the quantized DiT (~13.5 GB for Qwen-Image) | **~4 GB** of transients, 0 GB resident |
| RAM needed | modest | the full bf16 DiT (~41 GB) + headroom |
| Evictions | may evict idle residents to fit | **never evicts** — refuses instead if the ~4 GB isn't free |
| Quality | ≈ bf16 (mixed-edge recipe) | bf16 — reference quality |
| Speed | competitive when the GPU has the bandwidth | ~25 s/step at 1024² measured on a 16 GB RTX-class card — **faster there than a big-APU GPU-resident int4** (~34 s/step), because per-step time is bandwidth-bound |

Rule of thumb: a card that can *hold* the quantized DiT can use either; a card that can't (or a
box whose VRAM is busy with resident LLMs) uses **offload** and coexists with everything. On a
unified-memory APU (gfx1151), int4 is the *capacity* recipe, not a speed win — see
[ROCM.md](ROCM.md).

## Loading

- **Dashboard:** the Load 🖼 button opens a confirm dialog with both modes.
- **API:** `POST /load?model=qwen-image` (GPU-resident) or
  `POST /load?model=qwen-image&t2i_offload=1` (offload). `force=1` applies as usual.
- Requests to the images endpoint **auto-load** a registered-but-cold image model, like the chat
  endpoints do (offload mode is a load-time choice, so auto-loads use GPU-resident).

## The API — `POST /v1/images/generations`

OpenAI Images shape, plus extensions:

```bash
curl -X POST http://<controller>:21434/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen-image",
        "prompt": "a corner store at dusk, neon sign reading OPEN 24 HOURS",
        "size": "1024x1024",
        "steps": 20,
        "seed": 42
      }'
```

| Field | Default | Notes |
|---|---|---|
| `model` | — | may be **omitted** when exactly one image model is loaded |
| `prompt` | required | |
| `size` | `1024x1024` | `"WxH"` |
| `n` | 1 | 1–4 images; with `seed` set, image *i* uses `seed+i` |
| `steps` / `num_inference_steps` | 20 | clamped 1–100 |
| `cfg` / `true_cfg_scale` / `guidance_scale` | 4.0 | |
| `negative_prompt` | — | extension |
| `seed` | random | extension |

Response: `{ "created": ..., "data": [ { "b64_json": "<png>" }, ... ] }` — base64 PNGs only
(no URL hosting). Renders for one model are serialized (per-model lock); a render is
minutes-scale (steps × per-step time), so budget client timeouts accordingly. Errors follow the
OpenAI error shape (`400` bad request, `404` unknown model, `503` load failure, `504` render
timeout).

The dashboard's model-detail modal has a **Generate** panel with the same knobs.

## Operational behavior

- **Renders block restarts and deploys.** `POST /update` and `POST /restart` return `409` while
  a render is in flight (a forced update once orphaned a finished render into a broken pipe);
  `force=1` overrides and aborts the render. The workers' idle self-update also waits for live
  renders.
- **Renders count as activity** — a rendering model is never idle-unloaded mid-render, and the
  generation watchdog knows a render's step progress (a slow render is not a wedge).
- **t2i loads participate in placement reservations** both directions: a t2i load reserves its
  RAM/VRAM in the ledger so concurrent LLM auto-loads can't race it into an OOM, and it
  subtracts other in-flight loads' reservations before picking its node.
- **Unload actually frees the DiT** (GPU storages emptied in place); an unload issued mid-render
  defers until the render's final step completes, then frees.

## Limitations (v1)

- One co-located GPU node — the DiT is not distributed across the fleet.
- `response_format` URL hosting is not implemented (`b64_json` only).
- Validated on the Qwen-Image pipeline layout; other diffusers pipelines may need adapter work.
- No int4 shard cache for the DiT yet — a GPU-resident int4 load re-quantizes at load time
  (offload mode skips quantization entirely).
