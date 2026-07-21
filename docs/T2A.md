# Text-to-music serving (ACE-Step)

InfiniteModel can serve **text-to-music** — **ACE-Step v1 3.5B** (a diffusion music generator:
genre/style tags in, an instrumental or vocal WAV out) — through an OpenAI-style
`POST /v1/audio/music` endpoint, on the same fleet that serves the LLMs.

> **This is an OPTIONAL component.** t2a needs the heavy `acestep` package (plus `diffusers`,
> `librosa`, `spacy`, `soundfile`, …) installed on the *serving* worker's Python env — it is **not**
> a core InfiniteModel dependency. A fleet with no acestep-capable worker simply has no music model;
> everything else (chat, embeddings, images, speech) is unaffected. Install it on as many or as few
> boxes as you like (see [Enabling t2a on a box](#enabling-t2a-on-a-box)).

---

## Architecture (what runs where)

Like text-to-image and TTS, ACE-Step is **not** layer-split across the fleet — the whole ~3.5B
pipeline (`ace_step_transformer` DiT + `music_dcae_f8c8` + `music_vocoder` + `umt5-base`) runs on
**one GPU worker**. As of **#media-anywhere (2026-07-17)** that worker no longer has to be the
controller's own box:

- **Placement** picks the controller-co-located GPU **or any remote GPU whose worker advertises
  the acestep runtime** (`can_t2a`, an import-free `find_spec("acestep")` probe reported at
  registration and shown per node in `/status`). Co-located is preferred when it fits (no transfer);
  otherwise the most-free capable GPU wins. So a full co-located card no longer blocks music if an
  idle remote GPU can host it.
- **Model delivery** needs no shared filesystem: a remote worker fetches the checkpoint itself via
  `snapshot_download(<hf-repo-id>)` — the registry target must be a public HF repo id (it is for
  `ACE-Step/ACE-Step-v1-3.5B`). A co-located worker keeps the fast local-disk path.
- **Result delivery** needs no shared filesystem either: the finished WAV returns as **base64 over
  the control link** (the controller decodes it off the event loop). A max-length clip (240 s) is
  tens of MB, so the controller's control-reader line limit is raised to 128 MB; the data plane is
  frame-based (`readexactly`) and unaffected.

Renders run on their **own CUDA/HIP side stream** (#gpu-share), so a music render doesn't starve a
co-resident LLM's decode to 0 tok/s.

## Getting the model

**+ Add model** on the dashboard with `ACE-Step/ACE-Step-v1-3.5B` pulls the 4-subfolder diffusers
layout onto the controller. Remote acestep-capable workers fetch it on first use via
`snapshot_download` (into their own HF cache — a ~10 GB one-time pull per worker). The models page
shows a **🎵 t2a** badge.

## Enabling t2a on a box

t2a serves on any GPU worker whose venv can `import acestep`. ACE-Step pins **old** deps
(`transformers==4.50.0`, etc.) that would downgrade — and break — the LLM-serving stack, so install
it **`--no-deps` under a constraints file** that pins the versions the rest of the fleet runs. This
is the exact recipe validated on beast (CUDA) and amdcomp (CUDA); adjust versions to match *your*
fleet's `torch`/`transformers`:

```bash
# 1) the ACE-Step source (editable, --no-deps below keeps its old pins from downgrading you)
git clone https://github.com/ace-step/ACE-Step.git /root/acestep-repo

# 2) pin the versions the fleet already runs so nothing here can move them.
#    numpy 2.4.6 (NOT 2.5.x) is deliberate — numba's py3.14 wheel needs numpy<2.5; see gotchas.
cat > /root/acestep-constraints.txt <<'EOF'
torch==2.11.0+cu128
transformers==5.12.1
numpy==2.4.6
EOF

# 3) torch companions matching your torch build (from the same index as torch)
/root/imenv/bin/pip install torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 4) acestep itself, NO deps (its requirements.txt would downgrade transformers)
/root/imenv/bin/pip install --no-deps -e /root/acestep-repo

# 5) acestep's runtime deps, at versions compatible with the newer torch/transformers,
#    constrained so the two critical libs above can't be moved
/root/imenv/bin/pip install -c /root/acestep-constraints.txt \
  accelerate diffusers librosa py3langid pypinyin soundfile spacy datasets \
  pytorch-lightning peft loguru matplotlib num2words tensorboard tensorboardX \
  cutlet fugashi unidic-lite hangul-romanize

# 6) verify the LLM stack is UNTOUCHED and acestep imports
/root/imenv/bin/python -c "import torch,transformers; print(torch.__version__, transformers.__version__)"
/root/imenv/bin/python -c "import acestep; print('acestep OK')"
```

Then restart that worker (`systemctl restart im-worker`, or the controller's per-node ↻ /
`POST /restart_node?node=<host>`). On reconnect it advertises `can_t2a=True` and becomes a valid
music-placement target. Verify with `/status` → the node's `can_t2a` flag.

### Why ACE-Step is **not** implemented on ROCm (AMD)

t2a is deliberately **CUDA-only**. A ROCm box (Strix Halo / gfx1151) is left `can_t2a=False` **on
purpose**, so the placement planner never routes music to it. Three independent walls, any one of
which is disqualifying:

1. **No `torchaudio` wheel matching the ROCm torch ABI.** ACE-Step hard-depends on `torchaudio` for
   audio I/O, but AMD's arch-specific builds (TheRock gfx1151 wheels, and ROCm nightlies generally)
   ship `torch` with **no matching `torchaudio`**. Installing one from PyPI or the CUDA index pulls a
   build compiled against a different torch ABI, which breaks the venv — *including the LLM stack the
   box is actually there to serve*. There is no supported combination.
2. **MIOpen-JIT unreliability on the diffusion kernels.** ACE-Step's DiT + DCAE decoder lean on
   conv/attention paths that JIT through MIOpen on RDNA. On gfx1151 those are flaky-to-broken — the
   same wall that already forces the **Kokoro TTS** leaf to fall back to CPU on that box. A render is
   dozens of diffusion steps deep, so a mid-render MIOpen failure wastes minutes of work.
3. **There is no CPU fallback to retreat to.** ACE-Step's diffusion does not run usably on CPU (see
   *CPU-only* below — measured: zero diffusion steps in 10+ minutes). So on a ROCm box there is no
   working path *at all*: not the iGPU, not the CPU.

**What to do instead:** serve music from a **CUDA GPU** — co-located, or anywhere in the pool via
`#media-anywhere`. That is exactly why the remote path exists: a ROCm-only controller keeps
`can_t2a=False` locally and renders on a CUDA worker elsewhere in its pool. If a pool has **no** CUDA
acestep worker, it cannot serve t2a at all — that is a hardware gap, not a config one.

## Install gotchas & troubleshooting

Real walls hit bringing this up on beast, amdcomp, and furnace, with their fixes:

- **acestep's `requirements.txt` will downgrade your LLM stack.** It pins `transformers==4.50.0`
  (plus old `torch`, `spacy==3.8.4`, `soundfile==0.13.1`, `pytorch_lightning==2.5.1`, …). A plain
  `pip install acestep` drags transformers back and breaks every LLM on that worker — hence the
  `--no-deps -e` + constraints recipe above. After installing, re-check that `transformers.__version__`
  is unchanged. The `ace-step 0.2.0 requires transformers==4.50.0, but you have 5.x (incompatible)`
  pip warnings are **expected and harmless** — acestep has no real transformers-5.x API break, only
  stale pins.

- **Python 3.14 + numba/numpy collision** (hit on furnace). librosa needs numba; numba's only
  py3.14 wheel is **`numba 0.66.0 (cp314)`, which requires `numpy<2.5`**. If your constraints pin
  `numpy==2.5.1` (furnace's stock), pip can't satisfy both, silently falls back to an *older* numba
  with no py3.14 wheel, tries to build it from source, and **fails — rolling back the ENTIRE dep
  install** (so `import acestep` then dies on a missing `loguru`/`librosa`). **Fix: pin
  `numpy==2.4.6`** (what beast/amdcomp already run; torch 2.11 is fine with it). Symptom to
  recognize: `RuntimeError: Cannot install on Python version 3.14.4; only versions >=3.10,<3.14 are
  supported` + `ERROR: Failed to build 'numba'`.

- **torchaudio / torchvision must come from the torch index, not PyPI.** They're `+cuXXX` builds
  tied to your exact torch; install with `--index-url https://download.pytorch.org/whl/cuXXX`
  (py3.14 cu128 wheels do exist). Plain PyPI gives a mismatched/CPU build or none.

- **The torchaudio.save → torchcodec crash is already handled.** ACE-Step calls
  `torchaudio.save(..., backend="soundfile")`, but torchaudio 2.11 routes `.save` through torchcodec
  (not installed). `worker_t2a.py` monkeypatches `torchaudio.save` to write via `soundfile` — no
  action needed; don't "fix" it by installing torchcodec.

- **spacy pulls blis/thinc (C builds).** They built fine on py3.13 (amdcomp) and py3.14 (furnace,
  `blis-1.3.3`), but on some Python/OS combos blis has no wheel and won't Cython-build (the same wall
  the Kokoro TTS leaf hits). If blis fails, the interpreter is too new for that spacy — use an older
  Python for the worker, or (last resort) stub spacy the way the Kokoro leaf does.

- **Deploy order: controllers BEFORE workers.** A worker on the new `worker_load.py` returns the WAV
  as base64 in `t2a_done`; an old controller still runs a 64 KB-capped control-reader, and that
  multi-MB line overruns it → the link resets and every model on that node is invalidated (a churn
  loop, not one failed render). Update/restart the controller onto the `#media-anywhere` code first,
  then the workers. (Workers don't run pushed code until they restart, so it's not a time-bomb —
  just never restart a worker ahead of its controller.)

- **A box with acestep can still show `can_t2a=False`** until its worker restarts onto the
  `worker_hw.py` that carries the probe. So a co-located worker that HAS acestep but hasn't been
  restarted is deprioritized behind a remote box that DID advertise `can_t2a` — it still serves as a
  last-resort candidate, but to make it preferred again, restart its worker. Check `/status` → each
  node's `can_t2a`.

- **Remote serving needs a public HF repo id.** A remote worker `snapshot_download`s the checkpoint
  by its registry target; `ACE-Step/ACE-Step-v1-3.5B` works. A locally-added / non-HF checkpoint has
  nothing to fetch → it serves only on a co-located worker (local disk); the error says so.

- **Register on each controller** with `POST /add_model?model=ACE-Step/ACE-Step-v1-3.5B&name=ace-step`
  (or **+ Add model**). This also downloads it to the controller for correct load-time sizing; the
  remote worker still fetches its own copy.

## Loading

- **Auto-load:** a `POST /v1/audio/music` for a registered-but-cold ace-step model loads it on
  demand (like chat/images/speech).
- **Dashboard / API:** the Load button, or `POST /load?model=ace-step`. `force=1` as usual.
- **Offload is the DEFAULT (`#t2a-offload-default`).** ACE-Step loads with its components **resident
  in RAM**, hopping the ~6.6 GB DiT onto the GPU only for the duration of each render: **0 GB
  resident VRAM**, ~8 GB *transient* VRAM during a render, ~12 GB RAM. So it cannot OOM a card on
  load, never evicts a co-resident model, and leaves VRAM free for LLMs — while renders still run on
  the GPU at full speed (seconds, not minutes). This is the standing default on every controller.
- **GPU-resident** (bf16, holds ~10 GB VRAM, marginally faster — no per-render weight hop) is now
  opt-**out**: `POST /config?t2a_offload_default=0` (persisted). Only worth it on a card with VRAM to
  spare.
- Offload still needs ~8 GB **free** VRAM at render time for the hop. A genuinely full card returns a
  clean `503` for that render — it never OOMs, and it never holds VRAM at rest.
- int4 is **not** a t2a tier in M1 (bf16 only).

### CPU-only (`cpu_only=1`) — plumbed, but ACE-Step does not render on CPU

`POST /load?model=ace-step&cpu_only=1` places the whole pipeline in RAM with `device='cpu'` and no
GPU at all (basis `t2a: single-node (CPU)`). The path works end-to-end — it loads cleanly in ~4 s,
0 GB GPU, no OOM — but **ACE-Step's diffusion does not actually execute on CPU**: measured on a
32-core box, a short render sat at ~8–14 % CPU with **zero diffusion steps** for 10+ minutes and
never produced audio. Treat CPU-only as a diagnostic curiosity, **not** a serving mode.

ACE-Step also has no cpu-force flag of its own — its `__init__` sets `self.device = cuda:0` whenever
`torch.cuda.is_available()`, and `load_checkpoint()` moves every component with `.to(self.device)`.
So `worker_t2a.py` must override `pipe.device` **before** `load_checkpoint()` for the request to be
honoured at all; otherwise a `cpu_only` load silently grabs the GPU (and OOMs if it's full).

> **Gotcha — `cpu_frac=1.0` does NOT mean "running on the CPU."** Both **offload** and **cpu_only**
> report `vram_used=0` and `cpu_frac=1.0`, because that stat describes **where the weights sit, not
> where the compute runs**. Offload renders on the GPU in seconds; true CPU never finishes. Tell them
> apart by the **basis string** — `t2a: single-node` (+ load line `…, offload`) vs
> `t2a: single-node (CPU)` (+ load line `…, 0.00 GB GPU, resident`) — or simply by render time.

## The API — `POST /v1/audio/music`

```bash
curl -X POST http://<controller>:21434/v1/audio/music \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "ace-step",
        "prompt": "warm lo-fi hip-hop, mellow rhodes, vinyl crackle, 80 bpm",
        "duration": 30,
        "steps": 60
      }' --output music.wav
```

| Field | Default | Notes |
|---|---|---|
| `model` | — | a registered ACE-Step music checkpoint |
| `prompt` (or `input`) | required | genre / style / instrument tags driving the generation |
| `lyrics` | `""` | optional lyric conditioning for vocals |
| `duration` | `30` | seconds, clamped to `3`–`240` |
| `steps` | `60` | diffusion steps |
| `guidance` | `15` | APG guidance scale |
| `seed` | random | integer for reproducibility |
| `response_format` | `wav` | WAV bytes returned |

Per-render progress mirrors back as `t2a_step` — the dashboard card shows `rendering … step i/n`
live. Generation is serialized per model (per-model lock). Errors: `400` bad request, `404`
unknown model, `503` at capacity / no capable GPU.

## Model-detail view (`media_info`)

Clicking the ace-step row on the models page shows the media Operational block (shared with
t2i/tts): type, device (GPU/CPU), parameters, weight size, sample rate, **last-render N× realtime**
(RTF), request count, and uptime.

## Operational behavior

- **A render counts as activity** — a rendering model is never idle-unloaded mid-run, and the
  generation watchdog knows its per-step progress (a slow render is not a wedge).
- **Renders share the GPU** with any co-resident LLM via a side stream (#gpu-share) — decode keeps
  flowing during a render instead of stalling to 0 tok/s.
- **Unload frees the model** on its worker like any other resident (media pipelines get an explicit
  VRAM-release pass since they have none of the shard tensor attrs).

## Limitations (v1 / M1)

- One GPU worker per model — ACE-Step is not layer-split across the fleet (its diffusion forward
  isn't a transformer pipeline). It fits a single ~12 GB card, so it doesn't need to be split.
- bf16 only (no int4/int2 tier for t2a in M1).
- WAV out only (no MP3/Opus encode).
- A remote worker can only fetch the checkpoint if the registry target is a **public HF repo id**;
  a locally-added, non-HF checkpoint serves only on a co-located worker (local disk).
- Each acestep-capable worker keeps its own ~10 GB copy of the checkpoint (no cross-worker dedup).
