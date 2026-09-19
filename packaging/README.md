# InfiniteModel packaging

Turns the run-from-a-folder InfiniteModel project into distributable packages.
This directory is the **foundation** layer: a `pyproject.toml` with per-backend
extras, a launcher, a build script, and a bootstrap installer. The `.deb`/`.rpm`
and Windows installers (to come) all reuse the wheel + `bootstrap.sh` logic
below, so there is one source of truth for dependencies and layout.

## Design in one paragraph

InfiniteModel treats its own directory (`dirname(__file__)`) as a **writable
root** — it stores state there and its self-update rewrites its own `.py` files
in place. So the wheel does **not** install the runtime into site-packages.
Instead the ~45-module runtime ships as *payload data* inside the wheel and is
copied into a writable app home (default `~/.local/share/infinitemodel`) on
first launch; the console scripts then `exec` the venv's Python on
`<home>/server.py` / `client.py`. The venv holds only dependencies. `torch` is
never in the wheel — its build is hardware-specific and `bootstrap.sh` installs
the right flavour first.

## Backends and their extras

| Extra        | Backend / role            | Pulls in |
|--------------|---------------------------|----------|
| `controller` | server + dashboard        | fastapi, uvicorn, python-multipart, pillow |
| `worker`     | LLM execution             | transformers, safetensors, huggingface_hub, numpy (+ torch, installed by bootstrap) |
| `vision`     | image inputs              | pillow |
| `stt`        | Whisper transcription     | python-multipart, soundfile |
| `audio-in`   | WAV/FLAC/OGG decode       | soundfile |
| `music`      | MusicGen text-to-music    | transformers, soundfile |
| `t2i`        | text-to-image             | diffusers, accelerate |
| `tts`        | Kokoro TTS                | loguru, espeakng-loader, phonemizer-fork, num2words, regex, scipy, soundfile (+ `kokoro`/`misaki` via `--no-deps`, + system `espeak-ng`) |
| `kimi`       | Kimi-Linear arch          | fla-core==0.4.0, triton |
| *(none)*     | **ACE-Step (t2a)**        | source install + torchaudio + Ampere+ GPU — `--with-acestep`, see `docs/T2A.md` |

## Build (on an internet-connected box)

```bash
python3 -m pip install build      # one-time
packaging/build.sh
# -> dist/infinitemodel-<ver>.tar.gz            (sdist)
# -> dist/infinitemodel-<ver>-py3-none-any.whl  (wheel)
# -> dist/infinitemodel-<ver>-installer.tar.gz  (wheel + bootstrap.sh)
```

The version comes from `server.py`'s `VERSION` constant. `build.sh` copies only
the files in `runtime-manifest.txt` (an allowlist — no secrets, no dev cruft)
and runs a safety scan before building.

## Install (any Linux flavour)

```bash
tar -xzf infinitemodel-<ver>-installer.tar.gz
cd infinitemodel-<ver>
./bootstrap.sh --cuda cu128 --extras worker,vision,stt     # GPU worker
./bootstrap.sh --cpu --extras controller,worker            # CPU box
./bootstrap.sh --cpu --extras controller --no-torch        # controller only
```

Then: `infinitemodel-controller` (dashboard on :21434) or `infinitemodel-worker`
(auto-discovers the controller by UDP broadcast). Override the app home with
`INFINITEMODEL_HOME` or `--prefix`.

## CI / releasing

`.github/workflows/build-packages.yml` builds all four artifacts (wheel/sdist +
installer tarball, `.deb`, `.rpm`, Windows `.exe`) in one `ubuntu-latest` job.
It runs on a PR that touches `packaging/`, on manual dispatch, and on a `v*` tag
— where it also publishes a GitHub Release with the artifacts attached.

To cut a release: bump `VERSION` in `server.py`, then push a matching tag (the
workflow fails fast if the tag and `server.py` disagree):

```bash
git tag v0.3.44 && git push origin v0.3.44
```

## Files here

- `pyproject.toml` — package metadata, extras, entry points.
- `src/infinitemodel/` — the launcher package (`_launch.py`).
- `runtime-manifest.txt` — the authoritative allowlist of shipped runtime files.
- `tools/closure.py` — recomputes that allowlist from the import graph.
- `build.sh` — assembles the staging tree and builds the artifacts.
- `bootstrap.sh` — the end-user installer shipped inside the tarball.
- `linux-pkg/`, `windows/` — the `.deb`/`.rpm` and Windows installer layers.
