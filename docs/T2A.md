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

# 2) pin the versions the fleet already runs so nothing here can move them
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

> **ROCm boxes** (Strix Halo / gfx1151): install the ROCm `torch`/`torchaudio`/`torchvision` wheels
> instead of the `+cu128` ones (see [ROCM.md](ROCM.md)); the `--no-deps`-plus-constraints shape is
> the same. A GPU that can't JIT-compile a kernel falls back per the engine's usual media path.

## Loading

- **Auto-load:** a `POST /v1/audio/music` for a registered-but-cold ace-step model loads it on
  demand (like chat/images/speech).
- **Dashboard / API:** the Load button, or `POST /load?model=ace-step`. `force=1` as usual.
- **Resident vs offload:** GPU-resident bf16 needs ~10 GB VRAM (fastest). If the chosen box can't
  spare that, pass `t2i_offload=1` (or let the auto-fallback take it): the components live in RAM and
  the ~6.6 GB DiT hops to the GPU per render — ~8 GB transient VRAM + ~12 GB RAM, and it never
  evicts a co-resident model. int4 is **not** a t2a tier in M1 (bf16 only).

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
