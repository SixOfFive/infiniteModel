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

Like text-to-image, TTS is **not** layer-split across the fleet. The whole model runs on **one
controller-co-located worker** (co-location is by hostname match — the worker process on the
controller's own box, so the finished WAV is written to a shared filesystem and handed back as a
local path, no transfer):

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

**GPU→CPU auto-fallback.** On gfx1151 (Strix Halo / om3nbox) MIOpen JIT-fails to compile Kokoro's
LSTM dropout kernel (`MIOpenDropoutHIP.cpp: '<utility>' file not found` — a TheRock ROCm-7.13
bug). A GPU warmup that raises a HIP/MIOpen compile error transparently **re-builds the model on
CPU**. At 82M params CPU synthesis is ~2× realtime, so the fallback is invisible in practice; an
NVIDIA box (beast) runs on the GPU at ~4× realtime.

## Getting the model

**+ Add model** on the dashboard with `hexgrad/Kokoro-82M` pulls the full layout —
`config.json` + `kokoro-v1_0.pth` + all 54 `voices/*.pt` — and it is immediately loadable. (The
downloader pulls `.pth`/`.pt` files for any repo that ships **no safetensors**, so weight-only and
voice-pack repos download completely instead of grabbing just the config.) The models page shows a
**🔊 tts** badge.

**Worker deps** (on the co-located serving worker's venv):

```bash
pip install --no-deps kokoro misaki
pip install loguru espeakng-loader phonemizer-fork num2words regex scipy soundfile
```

`KModel` itself only needs `torch` + `transformers` + `scipy` + `numpy`; the rest is the G2P
front-end. Controller boxes need none of this unless they also host the serving worker.

> **Controller HF-cache gotcha.** The controller sets `HF_HOME=<repo>/cache/huggingface` (not the
> default `~/.cache`). Acquire the model through **`/add_model`** / the dashboard — a manual
> `snapshot_download` on the box lands in the default cache the controller can't see. If you must
> download by hand, set `HF_HOME=<repo>/cache/huggingface` first.

## Loading

- **Dashboard:** the Load button on the Kokoro row loads it onto the controller-co-located worker
  (GPU-preferred, CPU fallback).
- **API:** `POST /load?model=kokoro`. `force=1` applies as usual.
- Requests to the speech endpoint **auto-load** a registered-but-cold Kokoro model, like the chat
  and images endpoints do.

It loads at ~0.3 GB. The juggler and the int4/int2 compile paths skip it (it is a co-located media
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
- **Unload frees the model** on the co-located worker like any other resident.

## Limitations (v1)

- One co-located worker — Kokoro is not distributed across the fleet (it is tiny; it doesn't need
  to be).
- English G2P only (EspeakFallback / `misaki.espeak`); other languages would need the language's
  misaki front-end, which pulls the spacy stack this leaf deliberately avoids.
- WAV / PCM out only (no MP3/Opus encode).
- The Omni Talker fallback remains available but is choppy on the current checkpoint — prefer
  Kokoro for speech.
