# Text-to-speech serving

InfiniteModel serves a dedicated **text-to-speech** engine — **Kokoro-82M** (a StyleTTS2
checkpoint, Apache-2.0, ~82M params / ~0.3 GB, 54 voices) — through the OpenAI Speech API, on
the same fleet that serves the LLMs. This page is the full guide: getting the model, how it runs,
the API contract, and the operational behavior around synthesis.

> **Why a dedicated TTS engine and not the Omni Talker?** The distributed Qwen2.5-Omni path can
> also emit speech via `/v1/audio/speech`, but its Talker/Token2Wav output is intrinsically choppy
> on this checkpoint (reproduced with HF-native `transformers` too — it is the checkpoint, not a
> serving bug). Kokoro is a small, purpose-built TTS model whose output is clean, so it is the
> recommended speech path. The Omni Talker route still exists as a fallback for callers that request
> an Omni model by name — see [the routing note](#the-api--post-v1audiospeech).

---

## Architecture (what actually runs where)

Like text-to-image, TTS is **not** layer-split across the fleet — the whole model runs on **one**
worker. Since `#media-anywhere` (2026-08-14) that worker no longer has to be co-located with the
controller: any node advertising the `can_tts` runtime (the full Kokoro stack: `kokoro` + `misaki`
+ `espeakng_loader` + `soundfile`) **and** the `mediab64` wire cap is eligible, and returns the WAV
as base64 over the control link rather than as a shared-filesystem path.
Verified live on the `.45` pool: `kokoro` placed on **remote** beast, `POST /v1/audio/speech` →
558 044 bytes of valid 24 kHz mono PCM, with no shared filesystem in the path.
When the chosen node IS local the original path-return is used, unchanged:

- **Kokoro `KModel`** — on the GPU when it compiles there, else CPU (see the fallback below).
  It is driven **directly**, not through Kokoro's `KPipeline`.
- **Grapheme→phoneme (G2P)** — `misaki.espeak.EspeakFallback`, backed by the pip-bundled
  `espeakng-loader` binary (no system `espeak-ng` needed). English text → phoneme string.
- **Voice style vector** — each of the 54 `voices/<name>.pt` packs is a per-length table of
  256-d style vectors; the leaf picks `voices/<name>.pt[len(phonemes)-1]` per chunk.
- Long inputs are split into sentences and phoneme chunks of **≤508** frames (KModel's context is
  ~512 incl. 2 bos/eos), synthesized in order, and concatenated. Progress mirrors back per chunk
  over the control link — the dashboard model card shows `synthesizing chunk i/n` live.

**Spacy-free by design.** Kokoro's `KPipeline` pulls `misaki.en → spacy → thinc → blis`, and
`blis` has no wheel for Python 3.13/3.14 and won't Cython-build on the fleet. So the leaf installs
`kokoro`/`misaki` **`--no-deps`**, injects a harmless `sys.modules['spacy']` stub so `import
kokoro` completes, and phonemizes through EspeakFallback. Do **not** "fix" this by installing the
full `kokoro` dependency tree — it will fail to build.

**GPU→CPU auto-fallback.** A GPU warmup that proves the card cannot execute this torch build
transparently **re-builds the model on CPU**. At 82M params CPU synthesis is ~2× realtime, so the
fallback is invisible in practice; a supported NVIDIA box (beast) runs on the GPU at ~4× realtime.
Two distinct families trip it, and both route through the one shared predicate
`worker_hw.gpu_exec_unsupported` (`#gpu-exec-fallback`) — also used by `worker_stt` and
`worker_t2music`:

- **ROCm JIT failure.** On gfx1151 (Strix Halo / om3nbox) MIOpen JIT-fails to compile Kokoro's
  LSTM dropout kernel (`MIOpenDropoutHIP.cpp: '<utility>' file not found` — a TheRock ROCm-7.13
  bug).
- **CUDA card older than the wheel.** `amdcomp`'s GTX 1070 is `sm_61`, and torch **`+cu128`**
  ships cubins for `sm_75`+ only. `torch.cuda.is_available()` still returns **True**, and
  `.to("cuda")` still succeeds (moving tensors needs no kernel), so the mismatch only surfaces
  when a kernel actually launches — at the warmup, as `CUDA error: no kernel image is available
  for execution on the device`, or cuDNN's `not compatible with devices with SM < 7.5` for the
  LSTM. The card itself is fine (Ollama serves on it happily with its own Pascal-capable build);
  it is purely a **torch-wheel** mismatch, and the fix is the wheel, not the card — see below.

**Fixing the wheel rather than living on the fallback (amdcomp, 2026-09-12).** The same torch
version is published against several CUDA runtimes, so a card below the default wheel's floor does
**not** require a version downgrade. amdcomp moved from `2.11.0+cu128` to **`2.11.0+cu126`** —
identical version, so `transformers` / `torchvision 0.26.0` / `torchaudio 2.11.0` / `triton 3.6.0`
all keep their existing pairings — and `cu126` still carries `sm_61`:

```bash
pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 torchaudio==2.11.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
```

Verified afterwards with a **real kernel launch**, which is the only probe that means anything
here: fp32 matmul, bf16 matmul, a cuDNN LSTM, and a Triton JIT kernel (exact, max err 0.0 — so the
fused w4a16/w8a16 int-quant path is intact). Kokoro then loads `ready on cuda` with **no fallback**
and runs at **RTF 0.04 (~25× realtime), 4.3× faster than the same box's CPU path** — so on a
discrete NVIDIA card the GPU is a clear win, unlike the gfx1151 APU case where CPU measured
faster. **Check `nvidia-smi` for other tenants first:** Ollama holds ~5.6 GB of this card's 8 GB
with its own Pascal build, which is what keeps larger iM models off the GPU here — that is a VRAM
contention limit, not a capability one.

> Until 2026-09-12 this predicate was three copy-pasted substring lists, all containing **only**
> the ROCm markers — so the fallback was **inert on the CUDA side** and a too-old card hard-failed
> the load instead of falling back. `scratch_gpu_exec_fallback_test.py` now locks down both
> families, the negative control (unrelated errors must still raise), and the parity property
> that no leaf re-spells the list.

## Getting the model

**+ Add model** on the dashboard with `hexgrad/Kokoro-82M` pulls the full layout —
`config.json` + `kokoro-v1_0.pth` + all 54 `voices/*.pt` — and it is immediately loadable. (The
downloader pulls `.pth`/`.pt` files for any repo that ships **no safetensors**, so weight-only and
voice-pack repos download completely instead of grabbing just the config.) The models page shows a
**🔊 tts** badge.

**Worker deps** (on the serving worker's venv — co-located *or* remote):

```bash
pip install --no-deps kokoro misaki
pip install loguru espeakng-loader phonemizer-fork num2words regex scipy soundfile
```

`KModel` itself only needs `torch` + `transformers` + `scipy` + `numpy`; the rest is the G2P
front-end. Controller boxes need none of this unless they also host the serving worker.
Those four packages are not incidental: `can_tts` probes `kokoro` **and** `misaki` **and**
`espeakng_loader` **and** `soundfile` (`worker_hw.py:500` — all four, because missing any one
ImportErrors at *load* rather than at registration), so installing them on a node is precisely
what makes that node an eligible placement candidate.

> **Controller HF-cache gotcha.** The controller sets `HF_HOME=<repo>/cache/huggingface` (not the
> default `~/.cache`). Acquire the model through **`/add_model`** / the dashboard — a manual
> `snapshot_download` on the box lands in the default cache the controller can't see. If you must
> download by hand, set `HF_HOME=<repo>/cache/huggingface` first.

## Loading

- **Dashboard:** the Load button on the Kokoro row hands the choice to the controller, which takes
  the first eligible node with VRAM and falls back to a CPU node otherwise
  (`engine_load.py:2399` builds the pool, `:2410` picks). Since `#media-anywhere` that pool is
  every `can_tts` + `mediab64` node, so the pick is routinely a *remote* box.
- **API:** `POST /load?model=kokoro`. `force=1` applies as usual.
- **Pinning the machine:** the Kokoro row has no "which machine runs it" select — the media node
  picker (8be766b) covered the t2i / t2a / t2music dialogs only. The backend honours a pin anyway:
  `POST /load?model=kokoro&node=<hostname>` threads through as `pin_host`
  (`routes_lifecycle.py:267` → `engine_load.py:652`) into `_place_filter`
  (`engine_load.py:2405`), which fails the load with a named error rather than silently placing
  elsewhere if that host isn't eligible.
- Requests to the speech endpoint **auto-load** a registered-but-cold Kokoro model, like the chat
  and images endpoints do.

It loads at ~0.3 GB. The juggler and the int4/int2 compile paths skip it (it is a single-node media
model, not a distributable LLM — there is nothing to promote or quantize).

## The API — `POST /v1/audio/speech`

OpenAI Speech shape:

```bash
curl -X POST http://<controller>:21434/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "kokoro",
        "input": "The quick brown fox jumps over the lazy dog.",
        "voice": "af_heart"
      }' --output speech.wav
```

| Field | Default | Notes |
|---|---|---|
| `model` | — | route to Kokoro by name; any other model (or none) falls through to the Omni Talker path |
| `input` | required | the text to speak |
| `voice` | `af_heart` | a Kokoro voice id (contains `_`, e.g. `am_michael`) is passed through; a bare OpenAI name is mapped (below); unknown → `af_heart` |
| `speed` | `1.0` | playback rate (extension) |
| `response_format` | `wav` | `wav` or `pcm` |

**OpenAI voice-name → Kokoro voice map** (so an OpenAI SDK's `voice="nova"` just works):

| OpenAI | Kokoro | | OpenAI | Kokoro |
|---|---|---|---|---|
| `alloy` | `af_alloy` | | `sage` | `af_sarah` |
| `echo` | `am_echo` | | `ash` | `am_adam` |
| `fable` | `bm_fable` | | `ballad` | `bm_george` |
| `onyx` | `am_onyx` | | `verse` | `am_michael` |
| `nova` | `af_nova` | | *(other/empty)* | `af_heart` |
| `shimmer` | `af_bella` | | | |
| `coral` | `af_kore` | | | |

The full 54-voice list is on the model-detail modal (expandable) and in `media_info` (see below).
Voice-id prefixes follow Kokoro's convention: `a`=American / `b`=British English, `f`=female /
`m`=male.

Response: a `24000 Hz` mono WAV (or raw PCM). Synthesis for one model is serialized (per-model
lock). Errors follow the OpenAI error shape (`400` empty input, `404` unknown/failed-load model,
`503` speech queue full).

The dashboard's model-detail modal exposes the same knobs in a small speak panel.

## Model-detail view (`media_info`)

Clicking a media model (tts / t2i / t2a) on the models page shows a media-appropriate Operational
block instead of the all-zeros LLM layout. For Kokoro the worker's `media_info()` reports:

```json
{ "kind": "tts", "engine": "kokoro", "device": "cuda:0", "sample_rate": 24000,
  "n_voices": 54, "voices": ["af_heart", "am_michael", ...],
  "default_voice": "af_heart", "params": 81763410, "loaded_bytes": 327053640 }
```

The dashboard renders: type, device (GPU/CPU), parameters, weight size (VRAM or RAM), sample rate,
the expandable voice list, default voice, **last-synthesis N× realtime** (RTF), request count, and
uptime. The block is generic across t2i / t2a / tts.

## Operational behavior

- **Synthesis counts as activity** — a synthesizing model is never idle-unloaded mid-run, and the
  generation watchdog knows a run's per-chunk progress (a slow synthesis is not a wedge).
- **Auto-fallback is silent** — if the GPU can't compile Kokoro's kernels the model loads on CPU;
  the load reply and `media_info` report the real device so the dashboard shows `CPU`.
- **Unload frees the model** on whichever worker is serving it, like any other resident.

## Limitations (v1)

- One worker — Kokoro is not distributed across the fleet (it is tiny; it doesn't need to be).
  That worker no longer has to be the controller's: `#media-anywhere` (5c2613a) made any
  `can_tts` + `mediab64` node eligible (`engine_load.py:2399`), and kokoro runs on remote beast
  from the `.45` controller today. "One worker" is the limitation; "one *local* worker" is not.
- English G2P only (EspeakFallback / `misaki.espeak`); other languages would need the language's
  misaki front-end, which pulls the spacy stack this leaf deliberately avoids.
- WAV / PCM out only (no MP3/Opus encode).
- The Omni Talker fallback remains available but is choppy on the current checkpoint — prefer
  Kokoro for speech.
