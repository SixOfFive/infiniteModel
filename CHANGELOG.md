# InfiniteModel changelog

A capability-level summary of how the engine came together. (The original repo tracked changes at
per-commit granularity in `server.py` / `client.py` `VERSION` tags; this public history starts from a
single squashed commit, so the detail below is grouped by milestone rather than by commit.)

## 2026-09-22 (latest) — feat: `#t2i-qwen21` — serve Qwen-Image-2.1 on the t2i path

### Added (`worker_t2i.py`, `server.py`/`client.py` `VERSION`)

- **`T2IPipeline` now serves Qwen-Image-2.1 alongside Qwen-Image v1, chosen from the checkpoint's
  own `model_index.json` `_class_name`.** 2.1 (`QwenImage21Pipeline`) differs from v1 in the
  diffusers pipeline/transformer/VAE class (`QwenImage21Pipeline` / `QwenImage21Transformer2DModel`
  / `AutoencoderKLQwenImage21`), the text encoder (a **full Qwen3-VL**,
  `Qwen3VLForConditionalGeneration`, vs Qwen2.5-VL), and the prompt tokenizer subfolder (a Qwen3-VL
  `processor/` vs a plain `tokenizer/`). The `_is21` branch imports the right classes, loads the
  processor, and hands it to **both** pipeline views (2.1's `__init__` derives `_drop_idx` from the
  processor; the render view keeps it — it is not a torch module, so it does not pull
  `_execution_device` onto CPU the way the text encoder would). The encoder/render two-view split,
  the CPU text encoder, the offload/int4 recipes and the `_decode` tail are unchanged and shared.
- **Geometry snap is now derived from the render pipeline's `vae_scale_factor`** (v1 → multiple of
  16, 2.1 → multiple of 32) instead of a hardcoded 16, so 2.1's coarser latent grid is respected.
- **2.1's classes require a `diffusers` built from git `main`** (no released `diffusers` — 0.40.0
  included — ships `QwenImage21*`); the import error names that. transformers 5.12 already carries
  Qwen3-VL. Registered as `qwen-image-2.1` → `Qwen/Qwen-Image-2.1`; first bring-up serves bf16 RAM
  **offload** on beast (16 GB 4070 Ti Super + 125 GB RAM; the 17.5 GB Qwen3-VL encoder rides CPU RAM).
- **Ops note:** before this deploy, the live controller (`iM:/root/infinitemodel`) was a stale git
  ancestor with the running file-set applied on top uncommitted; it was fast-forwarded to
  `origin/main` (`3be9148`, gitignored runtime catalog preserved) so the repos are reconciled.

## 2026-09-19 — docs: `#packaging` — README documents the packaged install + release

### Changed (`README.md`)

- **`## Installation` now leads with a "Packaged install (recommended)" subsection.** Points at the
  GitHub Release's `.deb` / `.rpm` / portable `.tar.gz` / Windows installer, the `infinitemodel-setup`
  (deb/rpm) and `bootstrap.sh` (tarball) venv step, and the backend extras
  (`vision/stt/tts/t2i/music/kimi` + ACE-Step). The existing per-role pip commands are reframed as
  "From source (manual pip)", and it links to `packaging/` for build details. No behaviour change.

## 2026-09-19 — ci: `#packaging` — GitHub Actions build for all four artifacts

### Added (`.github/workflows/build-packages.yml`)

- **One `ubuntu-latest` job builds every distributable off the shared wheel:** the wheel/sdist +
  installer tarball (`packaging/build.sh`), `.deb` + `.rpm` (`build_pkgs.sh` with a pinned,
  checksum-verified nfpm 2.47.0), and the Windows `.exe` (`build_installer.sh` with apt `nsis`). Runs on
  a PR touching `packaging/`, on manual dispatch, and on a `v*` tag. The version is read from
  `server.py`'s `VERSION`; a tag build **fails fast if the tag ≠ VERSION**. On a tag it uploads the
  artifacts and publishes a **GitHub Release** (`contents: write`, via `gh`); pushing to `main` does not
  build (tag to release). Cut a release with `git tag v0.3.44 && git push origin v0.3.44`. Validated the
  YAML locally (pyyaml); the three build scripts it orchestrates are each already proven on MOBILE with
  the same tools (nfpm 2.47.0, makensis). Not yet exercised on GitHub — needs a tag/PR to run.

## 2026-09-19 — fix: `#packaging` — infinitemodel-setup no longer starts services without --start; live .deb install test

### Fixed (`packaging/linux-pkg/infinitemodel-setup`)

- **`enable_svc` used `${DO_START:+--now}`, which expands to `--now` even when `DO_START=0`** — the
  `:+` operator tests set-and-non-empty, not truthiness, and `"0"` is non-empty. So
  `infinitemodel-setup --role controller` (no `--start`) both **enabled AND started** the service.
  Replaced with an explicit `[ "$DO_START" = "1" ]` test: selecting a role enables its unit (starts on
  boot) but only `--start` starts it immediately. Found by the live install test below; deb + rpm
  rebuilt with the fix.

### Verified — live `dpkg -i` install test on MOBILE (Debian 13)

- `sudo dpkg -i` installs cleanly; postinst creates the `infinitemodel` system user + `/var/lib/
  infinitemodel` (0750, owned) and registers both units (`dpkg -L` / `getent` / `systemctl
  list-unit-files` all correct). `infinitemodel-setup --role controller --no-torch --extras controller`
  built `/opt/infinitemodel/venv` and pip-installed the controller stack (fastapi/uvicorn/pillow/
  python-multipart) with no torch pulled. Purged cleanly afterwards; box returned to its prior state.
- **⚠ Gotcha (a deployment note, not a package defect): unit-name collision on a node that already runs
  a worker.** MOBILE is itself a fleet worker via a hand-rolled `/etc/systemd/system/
  infinitemodel-worker.service`; the package uses the same unit name. The install is safe (the
  package's unit lands in `/usr/lib/systemd/system`, shadowed by the `/etc` copy, so the running worker
  is untouched), but `apt remove` / `dpkg -r` would run the package prerm's `systemctl stop/disable
  infinitemodel-worker` and thus stop the hand-rolled worker too. On a normal target (no pre-existing
  same-named unit) that is correct; when co-installing on a node that already hand-rolls
  `infinitemodel-worker.service`, stop/rename that unit first. For this test the package was purged with
  its maintainer scripts neutralised, so the live worker was never touched.

## 2026-09-19 — feat: `#packaging` — Windows installer (NSIS) + launcher self-update supervisor

### Added (`packaging/windows/` — Windows installer)

- **A per-user NSIS installer built on Linux** (no Windows/Wine — `makensis`, from Debian's `nsis`
  package). `win-installer.nsi` (MUI2) installs into `%LOCALAPPDATA%\Programs\InfiniteModel`: the
  wheel, `bootstrap.ps1`, the two `.cmd` launchers, README, LICENSE. **Components page** = roles
  (Controller, Worker) + optional backends (Vision, STT, TTS, T2I, MusicGen, Kimi-Linear, ACE-Step),
  each mapping to a pip extra; a **custom compute page** picks CUDA 12.8 / 12.6 / CPU. ACE-Step warns +
  is skipped on the CPU build (needs an NVIDIA Ampere+ GPU). Post-install runs
  `bootstrap.ps1 -Flavor <…> -Extras <ticked>` to build the venv (torch flavour → `wheel[extras]`,
  Kokoro `kokoro`/`misaki` `--no-deps`), then drops Start Menu shortcuts. The `.cmd` launchers set
  `INFINITEMODEL_HOME=%LOCALAPPDATA%\InfiniteModel` (writable app home) and run the venv console scripts.
- **`bootstrap.ps1`** — the Windows twin of `packaging/bootstrap.sh` (find Python 3.10+, venv, torch by
  index-url, `wheel[extras]`, tts/acestep handling). Prerequisite: Python 3.10+ on PATH (installer
  reports clearly if missing). Bundling a standalone CPython is a possible later enhancement.
- **Built + validated** on MOBILE: `build_installer.sh` → `dist/infinitemodel-0.3.44-setup.exe` (1.1 MiB),
  which `file` confirms is a "Nullsoft Installer self-extracting archive" and `7z` shows packs the wheel
  + both launchers + `bootstrap.ps1` + docs. **Not validated on Linux** (needs a real Windows run): the
  install-time component→extras mapping, the compute page, and `bootstrap.ps1` itself (no `pwsh` on this
  box, so it was reviewed by hand, not interpreter-checked).

### Changed (`packaging/src/infinitemodel/_launch.py` — launcher is now a self-update supervisor)

- **The console-script launcher runs `server.py`/`client.py` as a CHILD in a loop** — relaunching on
  exit 42 (the runtime's self-update signal) and propagating any other exit code — instead of
  `os.execv`. `os.execv` cannot propagate a child's exit code to a parent `.cmd` on Windows, which
  would silently break the 42-relaunch that the Windows launchers and the existing `.bat` supervisors
  depend on. The loop also makes interactive (tgz) self-update work and lets the **systemd** service
  ride through a self-update without a restart (only a real crash trips `Restart=on-failure`). Verified
  the 42→relaunch→final-exit-code path with a stub. The wheel — and thus the .deb/.rpm/.exe/installer
  tarball — were rebuilt on the new launcher so all artifacts match.

## 2026-09-19 — feat: `#packaging` — .deb/.rpm system packages (systemd + venv-on-setup)

### Added (`packaging/linux-pkg/` — Debian/RPM packages)

- **A `.deb` (built) and `.rpm` (spec ready) that install the code + systemd units and defer the
  hardware-specific venv to a one-time `infinitemodel-setup`.** Layout: wheel → `/opt/infinitemodel/
  wheel/`, `infinitemodel-setup` → `/opt/infinitemodel/bin` (symlinked to `/usr/bin`), units →
  `/usr/lib/systemd/system/{infinitemodel-controller,infinitemodel-worker}.service`. Post-install
  creates the `infinitemodel` system user and the writable app home `/var/lib/infinitemodel` (the
  launcher's `INFINITEMODEL_HOME`; self-update writes there), reloads systemd, and prints the next
  step. It deliberately does **not** build the venv or start services — the venv install is
  CPU/CUDA/ROCm-specific and can pull gigabytes of torch, which is wrong inside a dpkg/rpm transaction.
- **`infinitemodel-setup`** (run as root after install) builds `/opt/infinitemodel/venv`, installs the
  torch flavour (`--cpu`/`--cuda cuNNN`/`--rocm`), then the packaged wheel with chosen `--extras`,
  handles the Kokoro-TTS `--no-deps` step and the ACE-Step guidance, symlinks the console scripts onto
  PATH, and can `--start` the chosen services. Same venv-build mechanism proven for `bootstrap.sh`.
- **One canonical spec, two emitters.** `nfpm.yaml` produces BOTH formats (`build_pkgs.sh`; needs the
  `nfpm` binary — no root, no rpmbuild). For boxes without nfpm, `build_deb.sh` builds the `.deb`
  natively with `dpkg-deb` (staged on local disk, because the CIFS repo mount cannot create the
  `/usr/bin` symlink — Input/output error). Per-format deps: deb `python3, python3-venv` (Recommends
  `espeak-ng`, `libsndfile1`); rpm `python3` (venv is in-stdlib there). Maintainer scripts are shared
  and portable across dpkg/rpm arg conventions; a plain remove keeps `/var/lib` state and models, deb
  `purge` removes them and the service account.
- **Both formats built + validated** on MOBILE. `.deb` via `build_deb.sh` (native `dpkg-deb`):
  `dpkg-deb -I/-c` confirm control metadata, layout, the preserved `/usr/bin` symlink, and
  postinst/prerm/postrm. `.deb` **and** `.rpm` via `build_pkgs.sh` + `nfpm` v2.47.0 (single static
  binary, checksum-verified download): `dist/infinitemodel_0.3.44_all.deb` (897 KiB) and
  `dist/infinitemodel-0.3.44-1.noarch.rpm` (901 KiB), both `all`/`noarch`, both installing the same 7
  paths (rpm payload cpio-listed to confirm; lead magic + gzip payload valid). Every shell script
  passes `bash -n`/`sh -n`; `systemd-analyze verify` passes (only the expected "venv not built yet"
  note). One nfpm gotcha handled: it does **not** expand `${VERSION}`/`${MAINTAINER}` in `contents`
  globs, so `build_pkgs.sh` sed-renders the yaml first. **Not yet done:** a root install test on live
  hosts (`dpkg -i` on Debian, `dnf install` on Fedora) — only build-time validation was run here.

## 2026-09-19 — feat: packaging foundation — pip extras, wheel, and a bootstrap installer (`#packaging`)

### Added (`packaging/` — distributable packages, foundation layer)

- **The run-from-a-folder project can now be built into a wheel + a self-contained installer tarball**
  — the shared base every future `.deb`/`.rpm`/Windows installer will reuse, so there is one source of
  truth for dependencies and layout. New `packaging/` dir: `pyproject.toml` (metadata + per-backend
  extras + console entry points), a launcher package (`src/infinitemodel/_launch.py`),
  `runtime-manifest.txt` (the shipped-file allowlist), `tools/closure.py` (recomputes it from the import
  graph), `build.sh`, `bootstrap.sh`, `README.md`.
- **Why a launcher instead of installing the modules into site-packages.** The runtime treats
  `dirname(__file__)` as a *writable root* — it stores `node_config.json`/`engine_config.json`/history/
  `models/`/`cache/` there, and self-update *rewrites its own `.py` files in place* (e.g. `server.py`
  overwrites `control_plane.py`). site-packages is the wrong home for that. So the ~45-module runtime
  ships as **payload data** inside the wheel and is materialised into a writable app home
  (`$INFINITEMODEL_HOME`, default `~/.local/share/infinitemodel`) on first launch; the
  `infinitemodel-controller`/`-worker` console scripts then `exec` the venv Python on
  `<home>/server.py`/`client.py`. Running a script file puts its dir on `sys.path[0]`, so `import wire`,
  the subprocess converters, and every `dirname(__file__)` path resolve to the app home exactly as a
  git checkout would — **zero source edits to the runtime**.
- **torch is deliberately absent from the wheel.** Its build is hardware/version-specific (CPU /
  CUDA cuNNN / ROCm TheRock), so `bootstrap.sh` installs the matching flavour first (`--cpu` /
  `--cuda cu128` / `--rocm gfxNNNN`), then `pip install`s `infinitemodel[extras]`. Optional model
  backends map to extras: `controller worker vision stt audio-in music t2i tts kimi`. The two "dirty"
  backends can't be plain extras and are handled explicitly: **Kokoro TTS** installs `kokoro`/`misaki`
  `--no-deps` (+ the system `espeak-ng` lib); **ACE-Step (t2a)** is gated behind `--with-acestep` and
  points at `docs/T2A.md` (source install under a constraints file — a plain `pip install acestep`
  drags transformers back and breaks every LLM on the node — plus torchaudio and a bf16-capable
  Ampere+ GPU).
- **Public-safety allowlist.** `build.sh` copies only the files in `runtime-manifest.txt` — the static
  import closure of `server.py`+`client.py` over the repo's own modules, plus the two
  subprocess-invoked converters (`gguf_convert.py`, `mxfp4_convert.py`) — and runs a secret/cruft scan
  before building. The 42 `scratch_*/test_*/bench_*` dev files, the `im_*.json` fleet presets, and the
  untracked `hf_token.txt` are all excluded; `config.json` ships as the public default
  (`controller_host: auto`, no secrets) and `custom_models.json` is seeded empty. Verified: the built
  wheel has **0** forbidden entries.
- **Proven end-to-end** on MOBILE: `build.sh` → `infinitemodel-0.3.44` sdist + wheel + installer
  tarball (version stamped from `server.py`'s `VERSION`); a clean-venv `pip install` of the wheel yields
  both console scripts; a dry-run launch materialises the 48-file payload into the app home and targets
  `<home>/server.py`/`client.py` with argv passthrough; the synced torch-free modules (`state`, `wire`)
  import from the home. Caught and fixed one bug in the process — pip byte-compiles the payload `.py`,
  creating a `__pycache__/` dir the sync must skip (now copies only the shipped `*.py`/`*.json`).
- **Scope / not yet done:** this is the foundation only. The `.deb`/`.rpm` (thin package → `/opt`,
  systemd units, postinst venv build) and the Windows installer (Inno Setup, component checkboxes incl.
  ACE-Step behind the Ampere+ gate) build on this wheel + bootstrap and are the next step. A full
  torch-backed end-to-end launch (a real controller boot) was **not** run here — only the
  packaging/launch plumbing was exercised, deliberately avoiding a multi-GB torch pull.

## 2026-09-19 — docs: beast is the fleet's first 2-worker (2-GPU) host (`#beast-2gpu`)

### Changed (docs — node guides)

- **`docs/nodes/4070-ti-super.md` + `docs/nodes/3060.md` now document beast running two workers,
  one per GPU.** A worker process serves exactly one GPU (`worker_hw.detect_device()` returns
  `torch.cuda.current_device()` = GPU 0; `_gpu_mem_gb()` hardcodes device 0), so the single
  `im-worker` used only the 4070 Ti SUPER (GPU 0) and the RTX 3060 (GPU 1) sat idle — the
  controller saw beast as one GPU. The 3060 was brought online as a co-located second worker
  `im-worker-3060.service` (`CUDA_VISIBLE_DEVICES=1`, `--name beast-3060`, `--data-port 50201`,
  `--device gpu`), registering as node `beast-3060` / GPU 1 (~11.6 GB). New **§3.1** in the 4070
  doc carries the recipe and the four load-bearing settings: the CVD pin (both workers still print
  `cuda:0` — verify by GPU UUID); `--name` to dodge the by-hostname registration collision;
  `--data-port` because the `50200` data-plane bind is fixed; `--device gpu` so the second worker
  doesn't re-advertise beast's shared ~125 GB RAM. The §6 "one worker per box" gotcha is corrected
  to "one worker per GPU", the overview table lists both cards + units, and the stale "(verify)
  exact ExecStart" note is resolved with the confirmed primary launch line (no `--name` /
  `--controller` — discovery). 3060 doc: two → three fleet 3060s + a beast row. Docs-only, no code
  change. Flagged as unexercised (beast is the first 2-worker box): both workers self-update the
  same `client.py`, and TP root-port assignment with two co-located workers.

## 2026-09-18 — int4 DiT tier for ACE-Step music (`#t2a-int4`, M2) — controller `server.py` 0.3.43

### Fixed (`#t2a-rss-leak` — worker RSS climbed each t2a load/unload cycle)

- **`T2APipeline._free_now` emptied only CUDA tensors, so the OFFLOAD pipeline's CPU-resident
  bytes leaked.** ACE-Step runs `cpu_offload` — the DiT/DCAE/UMT5 live in CPU RAM (~2.7 GB int4)
  and only the DiT hops to the GPU per render — so at unload the bulk of the pipeline is CPU, not
  VRAM. The unload pass reset just the `cuda` storages; the CPU storages stayed allocated (held by
  some lingering ref) until the pipeline object was itself GC'd. `_unload_model`'s `malloc_trim`
  runs *before* that, so it found nothing free to return to the OS, and the **deferred** post-render
  free (`generate()`'s `finally`) never re-enters `_unload_model` at all — no trim on that path. The
  worker's RSS grew ~2–3 GB per load/unload cycle (observed to 4.9 GB) and starved MOBILE's free RAM
  until a restart. `_free_now` now empties **both** cuda and cpu storages in place (releasing the
  bytes regardless of who still references the module) and trims the glibc arena itself, so the RAM
  returns to the OS on both the immediate and deferred paths. `malloc_trim(0)` only reclaims free
  arena, so it can't disturb another model resident in the same worker.

### Fixed (fleet-serve: thread `quant` through the load handoff)

- **The t2a load message + worker construction both hardcoded `quant="none"`**, so a fleet
  `POST /load?model=ace-step&quant=int4` accepted int4 in placement but always loaded bf16.
  `engine_load` now sends the real `quant` in the `{"type":"load","kind":"t2a",…}` message, and
  `worker_load` passes `a.get("quant")` into `T2APipeline` (was `"none"`). Verified: a fleet int4
  load now quantizes (worker logs "int4 DiT quantized during load") and rests at ~2.7 GB.
- **`/status` now reports the real t2a quant** — `_load_t2a_locked` recorded the `LoadedModel`
  with a hardcoded `quant="none"`; it now uses the actual quant, so an int4 load shows `int4`.


### Changed (fleet-serve fit — memory-efficient int4 load)

- **The int4 DiT is now quantized DURING load, so the fleet load fits a RAM-constrained 3060.**
  ACE-Step's `load_checkpoint` loads the DiT first (via `from_pretrained`, which mmaps the
  safetensors) and the small DCAE/UMT5 after. `worker_t2a` scopes a patch of
  `ACEStepTransformer2DModel.from_pretrained` that int4s the DiT before it returns — so the bf16
  weights stay reclaimable page cache and only the int4 output is committed. **Measured load RAM
  peak dropped from ~7.7 GB (whole bf16 pipeline) to ~1.9 GB.** A post-hoc safety net re-quantizes
  if the patch ever fails to fire.
- **int4-aware offload RAM budget in `_load_t2a_locked`** (`_offload_ram_gb` = `all_b×0.5 + 2` for
  int4 vs `all_b + 4` for bf16). The old bf16 budget (~11.7 GB) was falsely refusing a 7 GB-free
  3060 for a load that actually needs ~3 GB.

### Added

- **ACE-Step music now fits a bf16-capable 6 GB GPU (e.g. an RTX 3060) via an int4 DiT** — so the
  fleet can serve music without the big cards (beast's 4070 Ti Super / furnace's 5090). The DiT is
  quantized in place after load with the project's own `QuantLinear4` (group-wise int4), covering
  BOTH the attention/embedder `nn.Linear` **and** the 1x1 `nn.Conv1d` that ACE-Step's `GLUMBConv`
  FF is built from — a 1x1 conv is a pointwise Linear over channels, and the FF (mlp_ratio 4) is
  the majority of each block, so quantizing only the linears would miss the fit. Left bf16: the
  depthwise conv (k=3), the Conv2d patch-embed, norms, embeddings. `prepare_fused` is deliberately
  skipped — ACE-Step's `cpu_offload` hops modules CPU↔GPU per render, and `QuantLinear4`'s naive
  dequant path is device-agnostic; the int4-packed weights still cut resident + per-render transfer
  VRAM ~4x.
- **Measured on MOBILE's RTX 3060 (sm_86, 5.79 GB):** pipeline loads at **2.68 GB** (from ~8 GB
  bf16), **0 GB resident** under offload, and a 10 s render **peaks at 2.09 GB** (vs the ~6.7–7 GB
  bf16 peak) — valid 48 kHz stereo audio. int4 targets the bf16-capable-but-small tier only; it does
  NOT help a pre-Ampere card (int4 still computes in bf16), and placement still gates t2a on
  compute cap ≥ (8, 0) per `#t2a-bf16-gate`.

### Changed

- **`worker_t2a.T2APipeline`** honors `quant="int4"` (was bf16-only); **`engine_load._load_t2a_locked`**
  accepts `int4` as a t2a tier and sizes the offload VRAM gate for it (`_T2A_OFFLOAD_VRAM_GB` 4.0 for
  int4 vs 8.0 for bf16). bf16 stays the better-quality default on cards with room; int4 is opt-in for
  the small-GPU tier. `loaded_params` is now captured before quant (int4 turns weights into buffers).
- **`scratch_t2a_int4_test.py`** exercises the shipped `_quantize_dit_int4` on a synthetic
  ACE-Step-shaped block: the 1x1-conv FF + attention `nn.Linear` become int4 (`QuantLinear4`), the
  depthwise conv (k=3, groups=C) stays bf16, forward parity holds within int4 error (~6.6%), weights
  shrink ~3.7x. Skips cleanly where torch is absent (the controller box).

## 2026-09-18 — `can_t2a` now requires bf16 hardware, not just the package (`#t2a-bf16-gate`)

### Fixed

- **A GPU too old for bf16 no longer advertises ACE-Step music capability — so a music request
  fails cleanly at PLACEMENT instead of crashing mid-render.** Workers advertised `can_t2a` purely
  from `find_spec("acestep")` succeeding. `amdcomp`'s **GTX 1070 (Pascal, `sm_61`)** imports
  `acestep` fine and reported `can_t2a=True`, so the controller placed an ACE-Step load onto it —
  and it loaded, streamed the DiT to the GPU, then **crashed ~54 s into the render** with cuDNN's
  `RuntimeError('GET was unable to find an engine to execute this computation')`. Root cause:
  ACE-Step's M1 pipeline is **bf16-only**, and bf16 execution needs an **Ampere+ GPU (CUDA compute
  capability ≥ (8, 0))** — the same threshold `worker_quant` gates the tinygemm int4 path on and
  `perf_profile.classify_device` splits CUDA_MODERN / CUDA_LEGACY on. Pascal (`sm_61`) and Turing
  (`sm_75`) have no bf16 engine. (VRAM was a red herring — freeing the card and lowering the offload
  gate got it to render, and it still died on the missing bf16 kernel.)

  - **Worker (`worker_hw.build_registration`):** `can_t2a` is now `find_spec("acestep")` **and** a
    bf16-capable CUDA GPU (`_t2a_bf16_capable`, capability ≥ (8, 0)). The CUDA capability is read
    once and reused for the existing `compute_cap` report. ROCm was already `can_t2a=False` by not
    installing `acestep` (torchaudio ABI clash); this closes the CUDA-Pascal hole. A pre-Ampere
    card now reports `can_t2a=False`.
  - **Controller backstop (`engine_load._load_t2a_locked`):** the t2a candidate filters and ranking
    now route through `_t2a_capable(node)`, which rejects a `can_t2a` node whose reported
    `compute_cap` is **known and < (8, 0)** — catching an OLD worker still advertising `can_t2a`
    from before this gate (self-update convergence window). An **unknown** capability (a worker
    predating `#sm-probe`) still trusts `can_t2a`, the same "never slander an absent field" rule
    `classify_device` uses.
  - Test `scratch_t2a_bf16_gate_test.py` exercises the real shipped predicates (the worker helper by
    import, the controller predicate by AST-extracting and executing the actual function) — Pascal
    `[6,1]` / Turing `[7,5]` rejected, Ampere `[8,6]`+ accepted, unknown-cap trusts `can_t2a` — and
    asserts the placement filters route through `_t2a_capable` (not the raw flag) so the gate can't
    be silently dropped, mirroring the `#node-optout` parity test.

## 2026-09-12 — the GPU→CPU media fallback was inert on CUDA (`#gpu-exec-fallback`)

### Fixed

- **A GPU too OLD for the installed torch wheel now falls back to CPU instead of hard-failing the
  load.** Brought up Kokoro TTS on `amdcomp` (GTX 1070, `sm_61`) and found its GPU cannot execute
  torch 2.11+cu128 at all — that wheel ships cubins for `sm_75`+ only. The trap is that
  `torch.cuda.is_available()` still returns **True**, and `.to("cuda")` still succeeds, because
  moving tensors launches no kernel; the mismatch only appears when a kernel actually runs. So the
  media leaves picked `cuda`, and the warmup raised `CUDA error: no kernel image is available for
  execution on the device` (or, for Kokoro's LSTM, cuDNN's `not compatible with devices with SM <
  7.5`). The card is healthy — Ollama serves on it with its own Pascal-capable build.

  `worker_tts` / `worker_stt` / `worker_t2music` each already had a GPU→CPU fallback for exactly
  this situation, and all three were **inert here**: each carried its own copy-pasted
  `hip = any(s in msg for s in (...))` substring list, and every copy listed only ROCm markers
  (`MIOpen` / `HIPRTC` / `hipErrorNoBinaryForGpu` / `Code object build failed`), because gfx1151
  was the only box that had ever needed it. A CUDA error matched none of them, so the branch was
  skipped and the exception **re-raised** — the load failed rather than falling back. Three
  copies, all wrong the same way, each passing its own per-leaf test.

### Changed

- **One canonical predicate, `worker_hw.gpu_exec_unsupported(exc, include_oom=False)`.** All three
  inline copies deleted and routed through it; it matches both families (ROCm JIT failure *and* a
  CUDA card below the wheel's minimum compute capability). `include_oom` keeps `worker_t2music`'s
  extra "out of memory" marker opt-in — for MusicGen a GPU that can't *fit* the model is also a
  CPU case, but for TTS/STT an OOM is a placement bug the controller must still see, not something
  to silently absorb onto the CPU. `worker_hw` was already in `EXTRA_UPDATE_FILES`, so this adds
  no new module for the workers to fetch.
- **`scratch_gpu_exec_fallback_test.py`** asserts behaviour *and* the structural parity property:
  the real CUDA strings observed on amdcomp match, the ROCm ones still match, unrelated exceptions
  still raise (the negative control — a predicate that returned `True` for everything would
  "fix" the bug while swallowing every genuine load error), OOM is opt-in, the definition exists
  exactly once, and no leaf re-spells the list. Verified live by mutation: removing the CUDA
  markers, forcing the predicate to `True`, and re-adding an inline copy to a leaf each fail it.

### Notes

- `amdcomp` was missing `kokoro`, `misaki`, `espeakng_loader` and `phonemizer-fork` in
  `/root/imenv` — installed per `docs/TTS.md`, which is what flips `can_tts` for placement.
- **amdcomp moved to `torch 2.11.0+cu126` and now runs Kokoro on the GPU.** The fallback above is
  what made the node work *at all*; this is the separate fix for the underlying wheel. A card
  below the default wheel's compute floor does **not** need a torch *downgrade* — the same version
  is published against several CUDA runtimes, and `cu126` still ships `sm_61`. Moving
  `2.11.0+cu128` → `2.11.0+cu126` (plus `torchvision 0.26.0+cu126` / `torchaudio 2.11.0+cu126`)
  keeps the version, so `transformers 5.12.1` and `triton 3.6.0` keep their existing pairings.
  Re-verified by **real kernel launch** — fp32 matmul, bf16 matmul, cuDNN LSTM, and a Triton JIT
  kernel (exact, max err 0.0, so the fused int-quant kernels are intact) — plus a live embedding
  request through the node. Kokoro now loads `ready on cuda` with **no fallback** at **RTF 0.04
  (~25× realtime), 4.3× faster than the same box's CPU path**. Worth recording because the
  gfx1151 precedent pointed the other way: there, GPU Kokoro measured *slower* than CPU, so
  "82M params won't benefit from a GPU" does not generalise off that APU.
- Remaining limit on that node is **VRAM contention, not capability**: Ollama holds ~5.6 GB of the
  card's 8 GB with its own Pascal build, so iM fits Kokoro (0.3 GB) but a 0.5B int4 LLM still
  plans onto CPU there.

## 2026-08-21 — Bazarr-compatible Whisper ASR (`#bazarr-asr`) (0.3.38 / 0.3.34)

### Added

- **`POST /asr` and `POST /detect-language` at the server root** — the
  `ahmetoner/whisper-asr-webservice` API that Bazarr's built-in Whisper provider speaks. Bazarr
  uses it as the last-resort path to *generate* subtitles when none can be downloaded; it is given
  a base URL and appends the paths itself, so they cannot live under `/v1`. Both are adapters over
  the same Whisper leaf `/v1/audio/transcriptions` already uses.

  This was **not** a pure routing job. The STT worker ran `model.generate` per 30 s window and
  concatenated *text* — there were no timestamps anywhere, and SRT is nothing but timestamps. So:

  - `WhisperPipeline.transcribe_segments()` keeps the existing 30 s windowing (deliberately — that
    long-form behaviour is already proven) and shifts each window's timestamps by its absolute
    offset. Timestamps are recovered via the tokenizer's `output_offsets` API, falling back to
    parsing the literal `<|0.00|>` timestamp tokens, which are part of Whisper's vocabulary rather
    than an API and so survive transformers version drift. Which path ran is **logged** — a silent
    fallback to whole-window timings still produces valid SRT, so without that line there is no way
    to tell real per-utterance cues from 30 s blocks.
  - `WhisperPipeline.detect_language()` runs the language head on the first 30 s only, which is why
    `/detect-language` is near-instant next to `/asr`.
  - The worker request grew a `mode` field (`text` | `segments` | `detect`). Absent — i.e. every
    pre-existing caller — means `text`, the original path, byte-identical.

  **The `encode=false` gotcha.** Bazarr pre-decodes with ffmpeg and sends **headerless** s16le /
  mono / 16 kHz PCM — no container to sniff. The controller prepends a 44-byte RIFF/WAVE header, so
  the worker's decode path is completely unchanged rather than learning a second input shape.
  `encode=true` is also supported (ffmpeg-decoded when ffmpeg is present) purely so the endpoint
  can be exercised with an ordinary media file.

  If the worker answers without segments — an out-of-date node during a rollout — `/asr` returns
  **503 rather than a subtitle with invented timings**. Bazarr caches what it receives and stops
  retrying, so a wrong subtitle file is permanent while a 503 is recoverable.

  `scratch_bazarr_asr_test.py` decodes the generated WAV header back with the stdlib `wave` module
  and compares sample-for-sample (a wrong sample rate would still transcribe — at the wrong speed —
  so it must be caught structurally), and checks the SRT is monotonic, gap-free in numbering, and
  free of zero-length cues. Both properties were verified by sabotage: a 44.1 kHz header and a
  dropped monotonic clamp each fail the test.

## 2026-08-21 — dashboard pages were served with NO cache headers at all (0.3.37 / 0.3.33)

### Fixed

- **`#dash-nocache`: an open browser tab kept running dashboard JS the server had already
  replaced.** Every dashboard page was returned as a **bare string**, so the response carried no
  `Cache-Control`, no `ETag` and no `Last-Modified`. Given neither a directive nor a validator, a
  browser falls back to **heuristic caching** and may reuse its copy indefinitely without ever
  revalidating.

  The effect is nastier than a stale page: a controller self-update ships new JS, the wire serves
  the new page, and the operator — looking at a tab opened before the update — sees the **old**
  behaviour. That is indistinguishable from *"your fix does not work"*, and it cost exactly that:
  a chat still rendering `"(no output)"` from JS that had already been replaced on the server, with
  the previous fix wrongly suspected.

  All **six** HTML routes (the five in `routes_dashboard` plus `/controllers` in `routes_peers`) now
  serve `no-store, no-cache, must-revalidate, max-age=0` from a header set defined **once** in
  `dashboard_html.NOCACHE_HEADERS`, reaching both route modules by the same
  import + `state.publish`/`bind` path the HTML constants already take. The pages are ~12 KB Python
  constants that change on every update, so there is nothing to gain by caching them and a
  correctness bug to lose.

  `scratch_dash_nocache_test.py` asserts the structural property — one definition, it still
  contains `no-store`, and every route declaring `response_class=HTMLResponse` returns through it —
  so a page added later as a bare `return SOME_HTML` fails the test.

  ⚠ The first version of that test was **inert**: it substring-matched the function body, and the
  explanatory comment above the return contained `NOCACHE_HEADERS`, so reverting the return to a
  bare string still passed. A negative control caught it, and the check now inspects the actual
  return *expression* via AST. Worth recording because it is the same defect class the test exists
  to prevent.

## 2026-08-21 — the dashboard chat threw away reasoning and showed "(no output)" (0.3.36 / 0.3.33)

### Fixed

- **A reasoning model's whole turn could vanish into "(no output)".** Reported against
  `qwen3.8-27b`: two ordinary questions, two blank replies. Nothing was wrong with the model, the
  KV cache or `kv_quant` — reproduced on the API, the turn was healthy: `finish_reason=length`,
  **0 chars of content and 847 chars of reasoning**. Three things combined:

  1. `/api/chat` streams a thinking model's scratchpad on a **separate channel**
     (`message.thinking`, serving.py). The dashboard chat read only `message.content` and
     **discarded the thinking entirely** — the information was already arriving and was dropped.
  2. The chat hardcoded `num_predict: 512`. That ceiling covers the *whole* turn, reasoning
     included, and qwen3.8-27b spends roughly 850 characters of scratchpad before its first
     content token — so it hit the cap having emitted no answer at all.
  3. The renderer then did `content || '(no output)'`, which reads as a dead model rather than a
     model that worked and had its work thrown away.

  Now the reasoning channel is accumulated and rendered (dimmed, italic, labelled, kept out of the
  message history posted back next turn), the cap is 4096, and a turn that ends with no content
  says so explicitly and points at the Max-tokens knob instead of showing a bare "(no output)".

  **A `reasoning` toggle now sits beside the model picker, defaulting to OFF.** Measured while
  verifying the fix: with thinking on, *"how to calculate pi?"* ran for **over ten minutes** on
  qwen3.8-27b at ~7 tok/s without finishing, because the scratchpad and the answer share one token
  budget. Off is what a smoke-test chat wants; on renders the scratchpad above the reply. `think`
  is sent explicitly either way — on `/api/chat` a reasoning model defaults to thinking ON, so
  omitting the key is not the same as asking for a direct answer.

  The request also posts only `role`+`content`: the message objects carry render-state (`think`,
  `capped`) that has no place on the wire. The server ignores unknown message keys today, but the
  UI should not depend on that, and a scratchpad must never be replayed as conversation.

  Note for API callers: the endpoint's own default is `num_predict` from the fleet config, falling
  back to **256**. That is fine for a normal model and too small for a reasoning one — a
  no-`max_tokens` request to a thinking model can legitimately return empty content with
  `finish_reason=length` and the text in `reasoning_content`. Deliberately not changed here: it is
  an operator-visible default on the Config tab, and raising it globally would change behaviour for
  every model to fix a case specific to reasoning ones.

## 2026-08-18 — kv_quant now works on hybrid archs (0.3.34 / 0.3.33)

### Added

- **`kv_quant` on hybrid architectures — `#172-hybrid`.** A hybrid arch was refused a TurboQuant
  cache outright, on the reasoning that linear-attention layers hold recurrent state rather than
  K/V. That is true of *those layers*, but not of the arch: **Qwen3.5/3.6/3.8 is 48 linear +
  16 `full_attention` of 64**, and those 16 grow an ordinary ctx-scaled K/V that packs exactly like
  a dense model's. On `qwen3.8-27b` at its native 262144 ctx that was **~16 GB of bf16 KV left on
  the table** — measured, by differencing `/plan` at two contexts: 14.0 GB across 229,376 tokens =
  64 KB/token, i.e. the 16-layer figure (all-64 funding would be 256 KB/token).

  Now the full-attention layers pack and everything else keeps its existing slot.

  **Two layer kinds are deliberately NOT packed**, and conflating them is how this goes wrong:

  - *linear-attention* — fixed-size conv + recurrent state, no K/V at all.
  - *sliding-window* — real K/V, but **bounded** by its own transformers cache class.
    `_TurboQuantLayer` extends the **unbounded** `DynamicLayer`, so swapping one into a sliding slot
    would drop the window bound *and* reserve the smaller packed figure — an under-reserve, which is
    a mid-decode OOM rather than a clean refusal. This is the same mistake that once funded
    gpt-oss's 12-of-24 and Gemma-4's 20-of-24 sliding layers at zero bytes.

  Which layers pack is decided in exactly one place, `Shard._kv_quant_layer_mask`, read by both the
  cache builder (`shard_forward._make_hybrid_kv_cache`) and the reservation (`shard_build`), so the
  cache and the bytes reserved for it cannot disagree. The controller mirrors it with a third
  predicate, `ModelSpec.turboquant_layers` — distinct from `is_hybrid` ("deviates from uniform full
  attention?") and `full_attention_layers` ("grows ctx-scaled K/V?", where sliding **counts**),
  because "does it get PACKED?" is a third question with a different answer for sliding layers.
  `for_kv_quant` now blends: packed layers at the bit-packed figure, the rest at bf16.

  `kv_quant='none'` returns exactly `DynamicCache(config=cfg)` and dense models are bit-identical,
  so every previously-working load is unchanged. Any failure in the swap returns a **fresh** plain
  hybrid cache — fresh because the swap mutates in place, so a half-swapped cache must be discarded.

  The fla/kda archs (Kimi-Linear) still refuse: they serve from `_make_linattn_kv`, which is never a
  quantized cache. The controller detects them as `kv_layer_frac < 1.0` with no `layer_types`.

  Two tests. `scratch_kv_quant_hybrid_mask_test.py` calls the **real** worker method against a stub
  Shard (shard_build imports without torch) and compares it with the **real** controller property
  across four archs, four shard offsets, the fla/kda case and an unknown layer kind — a
  transcription of either rule would have proved nothing. `scratch_kv_quant_parity_test.py` gained
  the blend, including a guard that a hybrid must **not** shrink like a dense model (which would
  mean its sliding/linear layers had been packed).

## 2026-08-18 — the workers get #sha-pin, and the byte checks the controller already had (0.3.33 / 0.3.32)

### Fixed

- **`#sha-pin` on the worker updater.** `worker_update.py` fetched from the branch ref, so it could
  stage a set mixed from two pushes exactly as the controller could. It now resolves the branch tip
  once per cycle and fetches every file at that commit. Branch fallback is kept and logged, because
  a worker whose network reaches `raw.githubusercontent` but not `api.github.com` must still be
  able to update at all.

- **The worker never verified the bytes it was about to write.** The controller has parsed fetched
  files since its own truncation incidents; the worker did not, and it is worse off — a `.py` that
  does not parse puts the box in a supervisor crash-loop with no local operator, and a wedged
  worker is far more awkward to reach than a wedged controller. Changed `.py` files are now
  `ast.parse`d before anything lands, and the write is staged to `.new`, `fsync`ed and **read back**
  before the replace pass, matching the controller's short-write guard.

- **`restart` + `update:true` exits unconditionally** — the same shape as the controller bug fixed
  in 0.3.33, where a forced update restarted onto a set the version gate had declined to activate.
  It is safe on the worker *because* the cycle now fetches one commit, so a staged set is coherent
  by construction and coming back on it is precisely what "restart + update" asked for. Documented
  at the call site rather than papered over. Residual, deliberate: on the branch fallback a mix is
  still possible, but the `ast.parse` above refuses the common corruption instead of restarting
  into it.

  Verified the new helpers actually resolve through `state.bind()` and hit the network — a
  `load_config` that failed to bind would have made the pin fall back to the branch **silently**,
  leaving the whole change inert behind its own fallback.

## 2026-08-18 — the restart gate was INERT on the path it was written for (0.3.33 / 0.3.31)

### Fixed

- **0.3.30's "never restart onto an unbumped primary" gate did nothing on a forced `/update`.**
  Found while trying to prove `#sha-pin` fires in production, which is the only reason it was
  found at all.

  `routes_lifecycle`'s forced path ends in an unconditional `os._exit(42)` and decides whether to
  skip it by scanning ACTIVITY for `"update REFUSED: "` — resting on an assumption that was true
  when it was written: *the updater `exit(42)`s itself the moment it installs anything, so merely
  returning means nothing landed.* Staging-without-restarting broke that assumption. The gate
  announced its decision with `print()`, which that scan cannot see, so the route found no refusal
  and **restarted the box onto the very set the gate had just declined to activate**.

  The guard was therefore inert on precisely the path that caused the incident it was written for,
  while its unit test passed — because that test exercised `_self_update_check` in isolation and
  never asked whether the *caller* could see the decision. The test now asserts the contract: a
  no-restart outcome must be visible to the forced-update route. Mutation-checked by reverting to
  `print()`, which fails only that new assertion while restart/staging/pinning still look correct.

  Route wording also corrected — it claimed "nothing installed", which since `#mixed-set` can be
  false: files may have been staged and simply not activated.

### Added

- **`last_update` now survives the restart.** As first shipped it was blank exactly when wanted: a
  successful deploy ends in `os._exit(42)` *inside* the update cycle, so the process holding the
  in-memory record is the one that dies, and every fresh controller reported `mode: "never"`. The
  only cycles it could describe were the ones that changed nothing. Persisted to
  `.last_update.json` (gitignored) and reloaded at startup, so it answers the question that
  actually matters: *which commit is the code I am running now from?*

## 2026-08-18 — the self-update fetches ONE commit, and says which (0.3.32 / 0.3.31)

### Added

- **`GET /code_manifest` now reports `last_update`** — `{ref, mode, at}`: the commit the last
  self-update cycle fetched from, whether it was `pinned` to a SHA or fell back to the `branch`
  ref, and when (UTC, matching `#utc-logs`).

  Added because verifying `#sha-pin` in production turned up a real observability gap: the updater
  prints its progress to stdout and **`GET /logs` does not capture stdout** — *zero* `[update]`
  lines ever reach it. So there was no HTTP-visible answer to "which commit did this box pull?",
  which is exactly the question the om3nbox incident hours earlier had to answer by hand-diffing
  per-file `sha1`s. `mode: "branch"` is now the visible signal that a box's cycle could still have
  seen per-file CDN skew.

  The record is a dict **mutated in place, never rebound** — `routes_diag` receives it by reference
  through `state.bind()`, so rebinding would leave that leaf holding a stale object. Verified by
  binding it and asserting a mutation propagates, rather than assuming it does.

## 2026-08-18 — the self-update now fetches ONE commit, not "whatever the CDN has" (0.3.31 / 0.3.31)

### Fixed

- **`#sha-pin`: a self-update cycle now pins every fetch to a single commit SHA.** 0.3.30 stopped a
  half-propagated set from being *restarted into*, but it could still be *staged* — and staged
  files load on the next restart from any cause, so the hazard was deferred rather than removed.

  `raw.githubusercontent` serves a **commit SHA** as a ref just as happily as a branch, and a
  SHA-pinned URL is immutable and content-addressed. So the updater resolves the branch tip once
  per cycle (unauthenticated GitHub API — the repo is public, no token in the source; ~4 calls/hour
  against a 60/hour limit) and fetches every file at *that* SHA. Per-file skew is then not
  detected-and-refused, it is **impossible**: every file names the same commit.

  Falling back to the branch ref when the SHA cannot be resolved is deliberate. The API is a new
  dependency and must not be able to block deploys by itself (rate limit, outage, a mirror serving
  raw but not the API). The fallback is exactly the old behaviour, still protected by 0.3.30's
  VERSION-bump restart gate — weaker, but not a regression, and it **says so in the log** instead
  of silently degrading.

  Verified end-to-end against real GitHub rather than assumed, because a resolver that always
  returned `None` would leave the feature inert behind its own fallback and look identical in a
  code review: the resolved SHA matches `git rev-parse origin/main` exactly, and the SHA-pinned
  fetch returns the expected bytes.

  `wire.repo_raw_url()` was deliberately **not** given a `ref` parameter. `wire.py` is shared with
  `client.py` and ships in both update lists, so widening its signature would let a new caller meet
  an old `wire.py` on a half-updated box — precisely the cross-file skew being eliminated. The
  pinned template is built controller-side instead.

  The self-update test now asserts every fetch in a cycle carried the same resolved SHA, recorded
  **eagerly per call** — the restart path ends in `os._exit(42)`, which runs no `finally` and no
  `atexit`, so a deferred write is lost in exactly the scenario that matters most. Mutation-checked:
  dropping the pin fails the assertion while leaving restart/staging correct.

- **Still open, and now the only remaining half:** the WORKER updater (`worker_update.py`) fetches
  from the branch ref the same way. Its restart is already correctly VERSION-gated with no `force`
  override, so it cannot restart into a mix — but it can still stage one. Same `#sha-pin` fix
  applies; deliberately not bundled here so the controller change can be verified in production
  first rather than moving 14 workers on the same push.

## 2026-08-18 — a forced update could restart onto a half-propagated set (0.3.30 / 0.3.31)

### Fixed

- **`POST /update?force=1` could restart a controller onto a file set that never existed as a
  commit.** `raw.githubusercontent` propagates a push **per file**, so a fetch minutes later can
  return a NEW extra beside a STALE primary. Every guard already in `_self_update_check` passed
  that case — the primary was not *older*, it was **equal**; the bytes parsed; the staged copies
  read back clean — and `force` then restarted anyway, because the restart rule keyed off `force`
  rather than off what actually came down.

  **Hit live while deploying 0.3.29.** `om3nbox` sits at a different physical site behind its own
  WAN edge, and I had polled *MOBILE's* CDN edge for the marker before triggering the deploy — not
  om3nbox's. It fetched the 0.3.29 `routes_dashboard.py` and `engine_load.py`, both of which call
  the new `ModelSpec.for_kv_quant`, beside the 0.3.28 `placement.py` that does not **define** it.
  The controller came back up 500-ing `AttributeError: 'ModelSpec' object has no attribute
  'for_kv_quant'` on every `/plan` and every LLM load. It recovered on the third retry once its
  edge caught up, but it should never have restarted into that state.

  A VERSION bump is the only evidence available at that point that the fetched files are one
  commit, so it now gates the restart **unconditionally** — `force` no longer overrides it.

  Staging deliberately still happens on an unbumped primary: writing the newer extras is how a
  half-updated box **converges** when the lagging files arrive. The first fix attempted here
  refused to write them at all, which is worse — it strands the box in exactly the inconsistent
  state being escaped and needs a fresh VERSION bump to escape it. Cost of the change: a commit
  that edits an extra without bumping VERSION no longer deploys on its own. Every deploy in this
  project bumps VERSION, so that is a non-event.

  Residual risk, stated rather than papered over: the staged files are on disk, so a restart from
  some *other* cause before the primary catches up still loads the mix. Closing that needs per-file
  commit identity — a manifest of expected hashes carried in the primary — which is a design
  change, not a guard. What is fixed is that a routine deploy no longer *causes* that restart.

  `scratch_selfupdate_mixedset_test.py` drives the **real** `_self_update_check` against a
  throwaway copy of the tree (so `os.path.dirname(__file__)` can never be the live repo), with the
  fetcher stubbed per scenario, and reads the decision off the process exit code (42 = restart):
  mixed-set forced → stage but do **not** restart; mixed-set unforced → unchanged; **clean deploy
  → still restarts** (the case that keeps the fix honest); downgrade → still refused, nothing
  written. Verified non-vacuous by restoring the old condition, which fails the first scenario with
  `rc=42`.

## 2026-08-17 — the Preview and the live load planner disagreed about KV (0.3.29 / 0.3.31)

### Fixed

- **`/plan` sized a TurboQuant load's KV the way the worker really reserves it; the LIVE load
  planner still sized the same load at bf16.** `engine_load` baked only `kv_slots` into the spec
  (`for_kv_slots`), never the packed-KV ratio, so in the narrow band where packed KV fits and bf16
  does not, the Preview said *"fits"* and `POST /load` then raised `CapacityError` on identical
  numbers — while the worker's `kv_reserve_probe` reserved the packed figure and would have been
  fine. Two planners, one model, three different answers. The gap was flagged in a code comment
  when the Preview was fixed (*"the fix is a shared `ModelSpec.for_kv_quant` applied at BOTH
  sites"*) and is now closed.

  Both planners call one **`ModelSpec.for_kv_quant()`**, which returns the re-sized spec *and* a
  note when a preset was refused. The ratio is baked into `kv_layer_frac` because that is the one
  knob scaling `kv_bytes_per_layer` for every controller-side consumer at once, so a fit and the
  figures printed beside it cannot disagree. Applied at the live planner, at `/plan`, and at the
  **adopt** path — an adopted model whose workers hold a packed cache was having its coexistence
  reserve over-counted at bf16 for the life of the adoption.

- **The hybrid gate lives inside that helper, not in its callers.** A hybrid / sliding-attention
  arch never builds a TurboQuant cache, so sizing one claims a fit the worker then reserves at
  bf16 and OOMs on. `/plan` owned the only copy of that rule and implemented it by **re-reading
  `config.json` off disk** — because when it was written, `ModelSpec` carried no arch facts. It
  does now (`layer_types`, from the same-day `#perf-facts` fix), so the re-read is gone and the
  gate is `spec.is_hybrid`. Leaving the gate to callers is what produced the split above: the
  caller holding the rule shrinks and the caller lacking it does not, which is worse than neither
  shrinking. Net **−64 lines** in `routes_dashboard.py`.

  Safe because every one of the 10 hard-coded `MODEL_SPECS` entries is a dense/full-attention arch
  (Qwen2.5, Llama, Mixtral, OLMoE, Qwen3.6-MoE) — every genuine hybrid is a user-added model whose
  spec comes from `_spec_from_config`, which populates `layer_types`. Checked rather than assumed.

  `scratch_kv_quant_parity_test.py` proves the collapse was **lossless** rather than merely
  plausible: it compares the helper against the old inline formula over **45** (arch × preset)
  combinations, with the old formula **pinned to HEAD's actual bytes** — it greps the pre-change
  expression out of `git show HEAD:routes_dashboard.py` and fails if what it re-implements is not
  literally what shipped, so updating the expectation to match a changed helper cannot make it
  pass. Verified non-vacuous by mutation: perturbing the ratio, removing the hybrid gate, and
  making `/plan` stop calling the helper each fail it. The pre-existing `scratch_perf_facts_test`
  independently confirms `spec.is_hybrid` equals the old config-driven gate across gemma4-sliding,
  qwen3-next-linear, kimi-kda and a nested VLM config.

## 2026-08-17 — a disabled node still got work, and a refusal blamed the wrong resource (0.3.28 / 0.3.31)

### Fixed

- **A node with BOTH memory tiers disabled was still receiving work.** Turning RAM *and* VRAM off
  for a host in the node config is not a placement hint — it is the operator declaring that host
  off limits. That rule existed as **six inline copies** of `(n.ram_enabled or n.vram_enabled)` in
  the media leaves of `engine_load.py`, and the two placement paths written *after* those copies
  never got one at all:

  - `media_encode.py` — the vision-tower encode picker. It selects a node, streams the tower
    weights to it and runs the encode there, and it sorts candidates by **most free VRAM** — so an
    off-limits box was not merely eligible, it was usually the one that **won**.
  - `routes_shards.py` — the `/compile_shards` candidate list. A distributed pack streams the
    source weights to the worker and quantizes them in *its* RAM, which is the resource being
    withheld.

  Fixed by collapsing every copy onto a single definition, `Node.placement_enabled` (server.py),
  applied once for all callers inside `_place_filter`. Added `engine_load._load_link` as a
  **dispatch-side backstop**: every load already reached for `self.links.get(node.node_id)` and
  raised on `None`, so the check went in beside the lookup that all 8 load dispatches share
  (5 media leaves + embedding + the pipeline LLM path in `_load_impl` + the TP mesh in
  `_load_tp_locked`). The primary gate is still the filter; the backstop exists because a filter is
  something a caller must *remember* to apply, which is precisely what the two paths above did not.

  Scope is placement only. Unload, teardown, restart, reaping, vram-trim and generation against a
  model **already resident** on the node deliberately keep their plain `links.get` — disabling a
  node has to let it *drain*, and refusing teardown would strand whatever is on it.

  The pipeline and TP paths were not reachable in practice (`eff_ram_gb`/`eff_vram_gb` both return
  `0.0` for a disabled tier, so the node offers no capacity and never wins a stage) — but that is a
  *budget* argument that holds only for planners which size against `eff_*`, and says nothing about
  a pin, an adopt, or the next planner. They are now explicit.

  `scratch_node_optout_parity_test.py` asserts the **structural** property rather than per-path
  behaviour: exactly one definition of the rule, no file re-spelling it inline, every `"type":
  "load"` dispatch bound through `_load_link`, teardown *not* bound through it, and the two later
  pickers consulting it. Written this way because a per-path test passes happily while the paths
  disagree — which is the state this file was already in. It found the `_load_impl` and
  `_load_tp_locked` sites on its first run.

  Verified live against the deployed controller with `beast` genuinely opted out: an LLM pinned
  there is refused, and so is a media load. That first live run also showed the fix was still one
  step short — the media leaves were *also* still filtering on `placement_enabled` themselves, so
  an opted-out node never reached `_place_filter` and the pin error could only say *"not an
  eligible node"* instead of naming the opt-out. Those redundant leaf copies are now removed, which
  is what the "collapse to one helper" was supposed to mean in the first place: `_place_filter` is
  the sole applier for the media paths, and the parity test gained an assertion that it still
  applies the rule, since dropping that one call would now silently disarm all five leaves at once.

- **The juggler and the `#load-faster` ⬆ re-place could still promote onto an opted-out node.**
  Found by a 110-agent adversarial sweep of every controller→node dispatch and candidate-selection
  site, run because a missed path is the whole failure mode here. `_node_live_free_vram_gb` is the
  documented *single source of truth* for placeable VRAM and reads the heartbeat's raw
  `vram_total - vram_used` — deliberately **not** `usable_vram_gb`, so that freed VRAM actually
  moves the number. It never consulted `vram_enabled`. The load planner survives that because it
  takes `min(tracked, live)` and its tracked side is tier-aware, but `_juggle_would_fit_vram` and
  `_plan_vram_first` pair this figure with `eff_ram_gb` and nothing else: on a both-tiers-off node
  that yielded `ram=0.0` correctly and `fv=`*live free VRAM* incorrectly, so the node still offered
  capacity. Fixed in the helper (0.0 when the VRAM tier is off) rather than at the two callers —
  the other two callers were already tier-filtered and are unaffected.

- **The opt-out refusal could fire *after* the embedding path destroyed the resident model.**
  `_load_embedding_locked` validates, then unloads the resident, then selects, then dispatches —
  and `_embed_candidates` exists precisely so the pre-unload validation and post-unload selection
  cannot disagree. Enforcing the opt-out only at the dispatch added a rejection reason that the
  validator did not know, so pinning an embedding reload at an opted-out node would have passed
  validation, destroyed the working copy, and *then* refused. The check now lives in
  `_embed_candidates`, so it fails before anything is torn down. Both this and the item above are
  asserted by the parity test, which cannot be satisfied by a dispatch-side guard alone.

- **An ACE-Step refusal named VRAM when RAM was the constraint.** The RAM-offload recipe has *two*
  independent budgets — transient VRAM for the DiT hop, and system RAM for the resting weights —
  but every failure message quoted `_need_gb()`, which is the **VRAM** figure. So a node that
  passed the VRAM test and failed the RAM one was reported as *"no GPU has ~8.0 GB free for the
  music model"*. Observed on `amdcomp`: 11.32 GB of free VRAM on an idle GPU, refused because it
  was **0.13 GB short of system RAM** (two KVM guests and Jellyfin held ~20 GB of its 31 GB). The
  message pointed at the one resource that was not the problem, and read as a phantom shortage.

  Each candidate now records which budget it actually missed, and the failure reports the binding
  resource with need-vs-have per node, sorted by **smallest gap** — the near-miss node is the one
  worth freeing 200 MB on. No safety margin was changed: the shortfall was real, only misreported.

## 2026-08-17 — three arch facts reached the perf resolver DEAD (0.3.25 / 0.3.31)

### Fixed

- **`perf_profile.resolve()` was told nothing true about the architecture.** All three of its arch
  facts were structurally always their default at **both** call sites (`engine_load`'s
  `_perf_auto_knobs`, `routes_dashboard`'s `/optimize_knobs`):

  - `is_hybrid` and `is_multimodal` were read as `getattr(spec, "layer_types", None)` /
    `getattr(spec, "is_multimodal", False)` against a `ModelSpec` dataclass that declared
    **neither field**. Nothing anywhere attached one, so both were always `None`/`False`.
  - `full_attention_layers` was never passed by **any** caller, so a hybrid's KV was sized over
    every layer including the ones holding fixed-size recurrent state.

  The resolver's *"multimodal/hybrid arch → clamp `kv_slots` to 1"* branch could therefore never
  fire, and `kv_slots` is an **applied** knob. Proven live before the fix, against the deployed
  controller: `qwen2.5-vl:7b` resolved to **`kv_slots=3`** with the rationale *"keeps interleaved
  conversations' prefixes warm"* — byte-identical treatment to the plain-text `qwen2.5:7b-instruct`
  — reserving 3× full-ctx KV to buy prefix reuse that `#prefix-kv` gates off for VLMs. The defect
  stayed invisible because the big hybrids clamp to 1 for an unrelated reason (no spare VRAM);
  only a *small* multimodal model exposes it.

  Fixed by giving `ModelSpec` the facts as real fields (`layer_types`, `is_multimodal`), populated
  in `model_store._spec_from_config` from the config it already parses, and exposing `is_hybrid` /
  `full_attention_layers` as **single-implementation** properties both call sites read.

  `is_hybrid` and `full_attention_layers` answer different questions and deliberately use different
  predicates — conflating them is the bug the pair exists to prevent. A **sliding-window** layer is
  hybrid (no prefix reuse) but still grows a ctx-scaled KV, so it counts toward
  `full_attention_layers`; charging those zero is precisely what funded gpt-oss's 12-of-24 and
  Gemma-4's 20-of-24 sliding layers at 0 bytes earlier this session. Only genuinely KV-less kinds
  are excluded, and an **unrecognised** kind counts — a new arch must over-reserve (a refused
  placement), never under-reserve (a decode OOM).

- **`kv_layer_frac` was derived only from `linear_attn_config.kda_layers`** — the `layer_types`
  branch of the same rule did not exist. So every hybrid that declares itself the transformers
  way (qwen3-next, qwen3.6-35b-a3b, Gemma-4, gpt-oss) planned KV over **every** layer while the
  worker's `_kv_layer_mask` reserved only the layers that really grow one. Measured on the real
  Qwen3.6-35B-A3B checkpoint: `layer_types` is **30 linear_attention + 10 full_attention of 40**,
  so the controller sized KV at 4× the worker's reservation and refused placements that fit.

- **A hard-coded `MODEL_SPECS` entry short-circuits the config**, so a fact missing there is
  missing for good — `resolve_spec` returns `MODEL_SPECS[id]` without ever reading `config.json`.
  The `Qwen/Qwen3.6-35B-A3B` entry declared neither fact and was wrong about both; its own comment
  said *"the checkpoint is multimodal"* while the spec said otherwise. Corrected **from the real
  checkpoint**, not from the comment: `kv_layer_frac=10/40`, `is_multimodal=True`. Encoded as the
  fraction rather than a fabricated per-layer `layer_types` order, because only the counts are
  consumed and inventing an order would be data nobody measured.

- **The first draft of this fix introduced the exact drift it was written to prevent**, and the
  adversarial verifier caught it. The controller's KV-less predicate started as a frozenset naming
  `mamba`/`recurrent`/`gated_deltanet`/`short_conv`/`conv` as well — **six of seven strings the
  worker's `shard_build._is_linear_attn_type` does not recognise**, since that gate is
  `"linear" in layer_type.lower()` and its docstring states it is *"deliberately NARROW"* and kept
  substring-based *"so the two sniffs cannot drift apart."* The planner would have funded such a
  layer at **zero** while the worker reserved a full ctx-scaled KV — a decode-time OOM. Replaced
  with the worker's predicate character-for-character. The controller may over-reserve relative to
  the worker; it must never under-reserve.

- **`/plan`'s kv_quant hybrid gate was left duplicated, deliberately.** It reads `config.json`
  directly rather than `spec.is_hybrid` because it must stay conservative when the config is
  *unreadable* — a case where the spec falls back to the built-in dense table and would look
  non-hybrid. That is not a pure dedup, so it was not done; instead the new test asserts the two
  implementations **agree** on every config both can read.

### Added

- **`scratch_perf_facts_test.py` — a regression gate for the whole bug class, not just this bug.**
  It deliberately does *not* test `resolve()`: a test that calls the resolver passes the flag
  itself and so can never see a dead input. It tests the **wiring** — (1) an AST scan failing any
  `getattr(spec, X, default)` naming a non-field of `ModelSpec`; (2) a coverage gate requiring every
  arch-fact parameter `resolve()` declares to be passed by every call site, which is what catches a
  fact dead by **omission** (`full_attention_layers`) that no getattr scan can see; (3) behaviour
  across dense / sliding / linear / kda / unknown-kind shapes; (4) the `/plan`-gate parity check;
  (5) `is_multimodal` detection; (6) `kv_layer_frac` derived from `layer_types`; (7) the built-in
  table's arch facts, with every other entry asserted still dense so planning is bit-identical;
  and (8) the parity check that caught the frozenset defect above — it **extracts the worker's
  `_is_linear_attn_type` from `shard_build.py` by AST and runs both predicates over a shared
  vocabulary of layer-type spellings**, rather than asserting a hand-copied expectation, so the
  two cannot drift without this failing. 63 checks, no torch, no fleet.

## 2026-08-17 — int8 caches are SERVED, not just compiled (0.3.24 / 0.3.31)

### Added

- **int8 shard caches are now selected at load.** They have been *compilable* since 0.3.20, but
  `use_cache` was gated to int4/int2 on **both** the controller and the worker — so nothing ever
  selected one, the load re-quantized anyway, and the cache sat on disk as dead weight.

  Every hop was traced before either gate opened, because this session shipped three fixes that did
  nothing when a value was dropped at an unchecked boundary: compile → manifest → verify → transport
  → gate → install. The worker gate mattered independently: without it the controller's
  `cache: "int8"` frame died at the worker boundary and `use_cache` could never be true.

  **Scoped to DENSE models, and the scope is load-bearing.** An int8 MoE cache cannot exist today
  (all four pack entry points refuse it), but the shape one *would* have is the single failure
  `_install_cached` cannot detect: `pack_unit_tensors` gates `is_expert3d` to int4, so routed
  experts would be written **bf16 passthrough**, install cleanly as plain Parameters, trip neither
  the 3D refusal nor the meta guard, and bring the shard up with bf16 experts against an int8 plan.

  The eligibility rule lives in **one** place (`_cache_serve_tier`), computed once above the retry
  loop so its two read sites cannot drift; the worker deliberately keeps no copy, since it cannot
  see the checkpoint's weight map and a second copy is exactly what drifts.

### Fixed

- **`docs/GGUF.md` described the behaviour this session replaced** — "rejects a split GGUF early"
  and "single-file quants only" were both inverted. Rewritten from current source, with the one
  genuinely open unknown kept as a **caveat** rather than papered over: whether the fleet's
  installed transformers can read a split set end to end was never determined.
- `routes_shards`' int8-MoE refusal wording was half-false, and `/pack_probe` lacked the up-front
  gate `/compile_dist` has — an int8 MoE probe failed as a 504 that reads like a node fault.
- **`FWD_FAIL_BENIGN` classified a benign superseded forward by substring-matching an exception
  class name.** Renaming `_ForwardSuperseded` would silently disable it, and routine orphan cleanup
  would count as node faults — feeding the wedge detector added earlier today, which can escalate to
  restarting a healthy node holding a resident model.
- `android/shards.py` carried the unfixed `_is_tied` plain-key probe, which serves the **embedding
  matrix as the LM head** for a quantized-head checkpoint.

## 2026-08-17 — a fix that was inert, and a rule that re-forked within hours (0.3.23 / 0.3.30)

Three of these four are consequences of **this session's own changes**, found by chasing the
follow-ups the waves generated rather than by anything failing.

### Fixed

- **`#sm-probe` was INERT.** The worker reported `compute_cap` and the dashboard read it, but
  `server.py` had no such field — `Registry.add` whitelist-parses registration, so the key was
  dropped at the registry boundary and every consumer's `getattr` fell to `None`.
  `perf_profile` returns `CUDA_LEGACY` only when handed a capability, so that class stayed
  unreachable and a pre-Ampere card kept classifying `CUDA_MODERN` — while failing worker_quant's
  `>= (8,0)` gate and silently running the naive int4 path at **5–20× slower**.

  Two links of a three-link chain were built and the third did nothing. Now carried at all three
  sites, and the **load-time** resolver passes `capability` too (only the dry-run path did).
  Absent stays UNKNOWN → CUDA_MODERN; absent must never read as old.
  **Verified live**: beast `[8, 9]`, Furnace `[12, 0]`.

- **The MoE tier rule had already re-forked.** `POST /load` kept a 32-line inline copy while the
  P2 wave added `_moe_tier_downgrade` to engine_load — the same per-branch duplication that
  produced this session's router-corruption bug, recreated within hours of fixing it. Collapsed,
  after proving the two bodies agreed across the full cross-product of inputs.

- **`load_faster` mapped `kv_quant 'none' -> ''`**, and `load()` reads empty as "inherit the global
  default" — so a model explicitly resident at `kv_quant=none` came back from a hitless re-place
  with a different KV cache, from an operation that promises to change only placement.

- **`android/` carried byte-identical copies of both P0s** (unguarded int8 router walk; two-state
  KV mask, worse there — its non-full-attention arm charged a literal `0`). Latent in that tree,
  but latent is exactly how the main-tree router bug survived for months.

### Known-open

int8 shard caches are compiled but never **served** — `use_cache` is int4/int2-only at three sites;
`_install_cached` grew its int8 branch here, the two gate sites remain. A cache installed but not
selected is harmless; the reverse is not.

## 2026-08-17 — two silent-corruption P0s, found by auditing this session's own work (0.3.21 / 0.3.28)

### Fixed

- **int8 quantized a MoE's router/gate; int4 and int2 skip it.** The "router always stays bf16"
  invariant (`docs/ACCELERATION.md`) was implemented as a **per-tier copy**, and int8's copy never
  had it. Quantizing a router corrupts top-k **expert selection** — the `#bare-linear-router`
  failure mode (`4dc57e6`): loads clean, answers once correctly, then degenerates.

  It was latent only because `routes_lifecycle` downgraded int8-on-MoE to int4. **Removing that
  downgrade earlier in this same session made it live** — the int8 MoE load demonstrated as a
  success an hour earlier was very likely running a quantized router.

  Fixed by collapsing three per-tier walks into one shared `_quantize_linears_` carrying the
  exclusion **once**, so a fourth tier cannot reintroduce the hole. Verified live: after the fix
  the worker reports `48x (128, 2048) torch.bfloat16` for the router while experts stay int8.

- **`_kv_layer_mask` reserved ZERO KV bytes for every sliding-window layer.** It read "not
  full_attention" as "holds no K/V", which is true only of *linear* attention. A `sliding_attention`
  layer is ordinary softmax with a **bounded** K/V. gpt-oss (12 of 24 layers) and **Gemma-4 (20 of
  24)** were funded at literally zero bytes — a silent under-reservation, and it desynchronised the
  worker from a controller that sizes KV for every layer.

- A gpt-oss compile guard that could never fire (it tested MXFP4 on a directory normalized to bf16
  at add time); `_is_tied` serving the embedding matrix as the LM head for quantized-head
  checkpoints; a refused/failed self-update still calling `os._exit(42)` after the route returned
  `ok`; `#restart-stale` invalidating a co-hosted worker's healthy models; node-restart recovery
  dropping `kv_slots`/`head_quant`/`tp`.

### A test had to be corrected, not merely added

`scratch_moe_int8_pack_test` asserted `2D router gate -> QuantLinear` — it encoded the **defect** as
the expected result and would have passed forever while routing was corrupted. A test that asserts
current behaviour proves nothing about whether that behaviour is right. The decisive new assertion
is **tier parity**: the skipped set must be identical across int4/int8/int2 and equal exactly the
routers.

## 2026-08-17 — MoE finally has a tier above int4 (server 0.3.20 / client 0.3.27)

### Added

- **int8 3D routed-expert packer.** Until now **no MoE could use any tier but int4** —
  `routes_lifecycle` downgraded int8/int2 loudly because no 3D packer existed. With w8a16 making
  int8 fast (2.83 → 25.00 tok/s on gfx1151), the whole MoE fleet was locked out of the
  best-quality tier.

  Bit-identity is **structural, not asserted**: `_pack8_expert` is now the single owner of the
  int8 arithmetic, so "the 3D pack equals the 2D packer per expert" is a tautology. An AST
  comparison confirms the dense int8 path is byte-for-byte unchanged, so no existing int8 cache
  is invalidated.

  **Verified live**, not just unit-tested: `qwen3-30b-a3b-instruct` loaded at `quant=int8` on
  om3nbox — `48x (128, 2048, 768) torch.int8 = 9.00 GB`, i.e. the 128-expert 3D tensors really
  are int8 — and generates coherently.

  Two hard refusals rather than a wrong answer: gpt-oss experts at int8 (IN-major transpose-packed
  weights need the fused w4a16 forward) and meta experts (int4-only streaming). gpt-oss at int8
  previously "worked" by silently leaving experts bf16 while the planner sized it 0.5× — it now
  fails *before* placement instead of OOMing mid-load.

- **gpt-oss IN-major layout** for `pack_linear_int4_3d`, transposing per expert so the peak is one
  expert rather than the whole tensor (#61: a whole-tensor `.float()` once spiked ~14 GB). The
  `pack_unit_tensors` guard deliberately **stays** — the packer is correct now but the serve side
  still cannot consume such a cache.
- **NVFP4 tensor-parallel** load (was a bare `NotImplementedError`): ranks bit-identical to the
  bf16 serve, with three refusals for the cases that cannot be made certainly correct.
- **Split multi-part GGUF ingestion** — accepted only as a complete contiguous 1..N series; an
  incomplete set fails loud at download time rather than producing a corrupt conversion.

### Verification note

All four suites were **executed on real torch** (om3nbox / gfx1151), not merely written — the
controller box has no torch, so the code that produced them could not run them.

## 2026-08-16 — backlog batch: O(N²) segmentation, self-update manifest, toolchain probe (server 0.3.18 / client 0.3.25)

The 105-item audit was re-verified against HEAD first: **16 items had already been closed** by the
preceding commits and **23 were dropped** as refuted or self-defeating, leaving 43 open-code items.
This is the surviving small/medium work, implemented across seven disjoint file sets in parallel.

### Fixed

- **Streaming tool-segmentation was O(N²)** — `_segment_tools` re-ran over the WHOLE accumulated
  string every token, on the tools path, i.e. exactly the coding-agent path. Now incremental,
  exploiting the invariant its own docstring already claimed ("prefix-stable"). Equivalence is
  tested, not asserted: OLD vs NEW over the same stream at multiple chunk sizes, compared at every
  step. **9.2× at 500 tokens, 19.4× at 2000, 39.3× at 4000.**
- **Mixed audio+image on `/v1/messages` silently dropped the images** and returned a 200 computed
  from partial input — the worst failure shape, since the caller cannot detect, retry or fall back.
  Now a 400 naming both modalities, refused before any tokenize/render/encode work.
- `/gpudiag`'s rocm-smi branch labelled a byte count as `used_mib`.

### Added

- **`#cc-probe`** — `has_triton` / `has_cc` per node in `/status`. A worker missing gcc or
  `Python.h` silently falls back to the naive int4 path (measured 2.08 → 15.4 tok/s, **7.4×**, on
  gfx1151). It logged locally and surfaced nowhere. Both default TRUE so a pre-feature worker is
  never slandered, and neither gates `can_infer` nor feeds placement.
- **`#newmodule-2cycle`** — the self-update fetch list now comes from the FETCHED `server.py`
  unioned with the running one, so a newly-added module converges in one cycle. Read with
  `ast.literal_eval`, never exec'd: this runs on CDN bytes *before* the update's own gates have
  decided we want that copy. Matches `AnnAssign` as well as `Assign` — the real declaration is
  annotated, and an Assign-only match would have silently always returned `[]`.
- Dead `_materialize_from_prefix` removed from `media_encode.py`.

### Documentation

Corrected against the source, not from memory: remaining co-location claims in T2I/TTS; the
CUDA-graph retest advice in ACCELERATION/ROCM (which omitted that `db47ae1` made the env var inert
on ROCm, so a retest also needs the `shard_forward` guard lifted); and six `docs/nodes/` files
carrying Windows-era facts, retired hosts and bare `(verify)` placeholders — each either resolved
with a citation or marked "unverified — needs the box" with the command to run.

### Verified live

Both pools on 0.3.18 / 0.3.25 (14 workers). Tool-calling returns `finish_reason=tool_calls` with
correct args and no reasoning leak; `has_triton`/`has_cc` visible per node; zero errors in either
controller's log.

## 2026-08-14 — pick which machine runs a media model (server 0.3.17)

### Added

- **The media Load dialogs (t2a / t2i / t2music) now let you choose the node.** The backend could
  already do it — `#media-pin` taught all five media leaves to honour `node=`, and `can_t2a` /
  `can_t2i` had been advertised for a while — but the dialogs never offered the choice, so
  placement was always whatever the heuristic picked. Verified backend-first:
  `POST /load?model=ace-step&node=amdcomp&t2i_offload=1` → `mode: pin:amdcomp/gpu`, resident in 20 s.

  The select is built from nodes that ACTUALLY advertise the matching capability, so a machine that
  cannot serve the model is never offered rather than offered and failing at load. Options show
  free (and in-use) VRAM and sort by headroom. Default "Auto" sends no `node=`, so behaviour is
  unchanged for anyone who ignores it.

  Three details that would each have been a bug: the buttons used to `closeOv()` **before** the
  handler ran, which would have torn `#m-node` out of the DOM and made every pick silently do
  nothing; t2i's auto-spill retry now carries the same pin, because spilling to RAM is a
  placement-*within*-node decision and silently relocating would contradict an explicit pick; and
  with no capable node the dialog explains itself instead of rendering an empty dropdown.

  Verified live: ace-step loaded on **amdcomp**, survived a hitless controller update still on
  amdcomp, and rendered an 8 s clip — HTTP 200, 1.53 MB, valid 44.1 kHz stereo WAV, 40 s wall.

### Fixed

- Three media dialogs still told the operator these models run only on "the GPU sharing the
  controller box", refuted by `#media-anywhere`.

## 2026-08-14 — Qwen3.8-27B, a reasoning leak, and four quantization gaps (server 0.3.16 / client 0.3.24)

### Added

- **`Qwen/Qwen3.8-27B` serves on om3nbox**, the day it was published. It needed **no architecture
  work**: despite the version jump it is `Qwen3_5ForConditionalGeneration` / `qwen3_5` with exactly
  the same shape as `qwen3.6-27b` — 64 layers, hidden 5120, `head_dim` 256, vocab 248320,
  `attn_output_gate`, interleaved mRoPE `[11,11,10]`, a vision tower, and the same hybrid
  `layer_types` (48 `linear_attention` + 16 `full_attention`). A retrained checkpoint, not a new
  architecture. Confirming that BEFORE writing any code was the whole job.

  55.59 GB / 18 shards in ~15 min; loaded first try, single stage, 64/64 layers GPU, 16.79 GB VRAM
  — byte-for-byte the same footprint as its 3.6 sibling. **Decode 8.91 tok/s** (112.2 ms/tok),
  measured by differencing n=32 against n=128 so prefill cancels; the naive
  `eval_count/eval_duration` reads 8.86 because `eval_duration` includes prefill. Text, reasoning
  (bat-and-ball → **$0.05**), tools and vision all validated.

### Fixed

- **Reasoning leaked into `content` unless the request happened to send tools.** Found on Qwen3.8
  but never model-specific — it hit EVERY model whose chat template opens `<think>`, on both the
  OpenAI and Ollama endpoints. `starts_in_think` was computed inside `if tools_req:`, yet whether
  the template opened a thought is a property of the PROMPT. Callers who defined no tool received
  the entire chain-of-thought terminated by a bare, unpaired `</think>` — malformed, not merely
  untidy. `/v1/messages` was always correct because `serving_anthropic.py` computed it
  unconditionally, which is why the qwen3.6 reasoning work never surfaced it.

  Confirmed by experiment, not by reading: the identical prompt plus one unused dummy tool returned
  a clean answer. Reasoning now goes to `reasoning_content` (OpenAI) / `thinking` (Ollama), with a
  streaming gate that needed **three** states — releasing on `</think>` and stripping the blank
  lines after it in one step only works when both land in the same chunk, and at one token per
  piece the closer arrives alone. Verified identical output at chunk sizes 1..59, 200 and 1000.

- **The int4, int2 and routed-expert fused self-checks only ever probed M=8.** `_op` dispatches on
  M: `M==1` is a split-K GEMV that atomic-adds into a zeroed accumulator, everything else is a
  `tl.dot` GEMM. They share no code — so the check validated the PREFILL kernel and said nothing
  about the kernel that runs on every decoded token, which is also the one carrying the
  `reset_to_zero` hazard (a stale accumulator shows up as wrong logits, not as an error). The
  routed-expert case was the worst: at decode a MoE routes ONE token per active expert, so its
  per-expert GEMM is M=1 on every token. All three now probe M in (1, 8). Live on gfx1151 after
  deploy: `[int4] triton w4a16 kernel active` — which now means both regimes passed.

- **`/compile_shards` would have written a silently TRANSPOSED cache for gpt-oss.** Its fused
  experts are IN-major — `_dequant_mxfp4_to_bf16` returns `[E, in, out]` — while
  `pack_linear_int4_3d` packs `[out, in]`. The compile path already refuses int8 MoE, fp8/nvfp4
  fused MoE and skeleton-less per-expert MoE under a stated "never a silent wrong cache" rule, but
  MXFP4 is neither fp8 nor nvfp4 and passed all three. The worker transposes correctly, so cold
  loads were right and only the cache would have been wrong — it would load, pass its own sha
  verification, and generate garbage. Now refused with a message pointing at the working path.

- **`kv_offload` enabled offloaded KV when ANY layer was on cuda.** `DynamicCache(offloading=True)`
  prefetches every layer's K/V to "the" compute device — it assumes one. On a mixed shard (the
  ordinary shape whenever a model doesn't fit) CPU layers were offloaded too. Now requires ALL, with
  the empty case excluded explicitly since `all([])` is vacuously true.

- **`perf_profile`'s section 8 silently overwrote the `head_quant` advice** added in `3d0be71`.
  `take()` is last-wins, so a second call for the same knob erases the first. The stale line
  predated the #8 measurement and proposed **int4** on ROCm — the value measured at 92.5% of the
  damage the entire int4 body does, and refused outright by `client.py`. So ⚡ Optimized settings
  offered ROCm a load the loader rejects, and erased the int8 suggestion everywhere else, leaving
  the feature 100% inert. A gate now walks `resolve()`'s AST and fails on any knob taken twice.

### Added — operations

- **`INFINITEMODEL_TTS_CPU=1`** forces Kokoro to CPU, matching the existing STT and T2MUSIC levers.
  tts was the only one of the three without it and the one that most needs it on gfx1151.
- **`/gpudiag` falls back to `rocm-smi`** when nvidia-smi is absent — it returned nothing but a
  FileNotFoundError on om3nbox, the Strix Halo box where "who is holding the GPU" is asked most.
  The CLI lives INSIDE the venv on `sys.prefix/bin`, so `shutil.which` alone never finds it.

### Security

- **13 of the 14 `INFINITEMODEL_SESSION_HANDOFF_*.md` files contained cleartext SSH passwords** for
  the fleet, and were untracked but NOT gitignored — one `git add -A` away from a permanent public
  push. Verified nothing had leaked (`git log --all -S<secret>` → 0 commits; absent from HEAD), then
  ignored as a class. The guard that should have caught this was inert twice over: `.git/hooks/` is
  not cloned, and that one copy was mode 0644 so git never ran it. The hook now lives at
  `hooks/pre-push` tracked as 100755, enabled with `git config core.hooksPath hooks`, and catches
  cleartext passwords by SHAPE as well as forge tokens — without hardcoding the literals, since the
  hook itself is public. Tested: blocks to github.com, allows to LAN, no false positive on `$PW`.

## 2026-08-14 — `#media-anywhere` for `t2i` and `tts` (server 0.3.15 / client 0.3.23)

### Fixed

- **`t2i` and `tts` could not load at all on a controller with no worker on its own box.** That is
  the normal shape for a controller running in its own VM — `iM` at `.45` — so `/load?model=kokoro`
  there failed with *"no controller-co-located worker … v1 serves speech models only on a worker
  sharing the controller's box"*, however much idle GPU the fleet had. `t2a`, `stt` and `t2music`
  had already been freed of this; `t2i` and `tts` were the last two.

  Being co-located-only was never one property — it was **three**, and these two leaves were
  missing all three while the other leaves had all three:

  | | `t2i` / `tts` before | how the fix works |
  |---|---|---|
  | **weights** | worker reads the controller's `model_dir` off a shared FS | remote worker `snapshot_download`s the repo itself, exactly as `t2a`/`stt`/`t2music` do |
  | **result** | worker writes a PNG/WAV and returns a **path** the controller `open()`s | returns the **bytes** as base64 over the control link |
  | **placement** | candidate predicate accepted only co-located nodes | accepts a remote node advertising the runtime **and** the new `mediab64` wire cap |

  The `can_t2i` flag needed for step 3 had existed all along — `worker_hw`'s own comment says it is
  advertised *"so the controller can place a t2a/t2i model on ANY capable GPU, not only the
  co-located box"* — the loader simply never consulted it. `can_tts` is new (probes
  `kokoro`+`misaki`+`espeakng_loader`+`soundfile`; missing any one `ImportError`s at load rather
  than at registration, so all four are probed).

### The co-located path is deliberately untouched

om3nbox serves `qwen-image` through the path-returning branch today, and a worker can update
before its controller does. So the worker switches to base64 **only when the controller asks**
(`inline` in the request), and the controller only asks when it placed the model on a node that is
genuinely remote *and* advertises `mediab64`. Every version pairing therefore works: an old worker
ignores `inline` and is never chosen for a remote placement; an old controller never sets it and
keeps the path. The legacy reply dict is byte-identical to before.

Gating on the wire cap **as well as** the runtime is the load-bearing part. An older worker
advertises `can_t2i` quite happily but would still reply with a path the controller cannot open —
and since a media model loads first and renders later, that failure would surface only on the
first request, with several GB already resident on the wrong node.

### Verification

- Two AST checks over `engine_load.py`: validated-before-destroyed still holds across **all six**
  single-node load paths, and neither `t2i` nor `tts` retains a raw `_LOCAL_IPS` co-location test.
- The placement gate was extracted from the shipped source by AST and exercised against stub
  nodes — **12/12**, including the case that matters: *remote node advertising the runtime but on
  an old worker → refused*, and *unknown node → legacy path* (the conservative answer).
- Full `py_compile` sweep plus `node --check` on all six embedded dashboard script blocks.

### Verified live on the fleet

Deployed and exercised end-to-end on the `.45` pool (controller 0.3.15, 11 workers 0.3.23):

- capabilities converged — `mediab64` on **11/11** workers, `can_t2i` on amdcomp + beast,
  `can_tts` on beast (the only node carrying the whole Kokoro stack; amdcomp has `diffusers`
  for t2i but not Kokoro).
- `POST /load?model=kokoro` → `kokoro READY (tts/kokoro, beast, cuda)`, placed on a **remote**
  node. That exact request previously failed outright with *"no controller-co-located worker"*.
- `POST /v1/audio/speech` → HTTP 200, **558 044 bytes**, a valid 24 kHz mono 16-bit PCM WAV of
  **11.62 s**, returned as base64 over the control link with no shared filesystem anywhere in
  the path.

⚠️ **t2i's remote path is implemented but not yet exercised on hardware.** It shares the placement
gate and the result transport with tts (both verified above), but its *weights* bridge — the
`transformer/`-subdir probe that decides whether to `snapshot_download` — is specific to t2i and
untested. The only registered t2i model is `qwen-image` at ~54 GB, so testing it is a real load,
not a quick check.

### Deploy note

Plain `POST /update` restarted the controller **on old code**: at that moment `raw.githubusercontent`
had propagated `wire.py` and `client.py` but not yet `server.py`, and the CDN lag is **per file**.
The symptom is quiet — the controller comes back healthy at the old version, with
`/code_manifest?grep=` showing `grep_hit: false` and an unchanged mtime. Poll the marker on
**every** file being deployed, not a representative one, and use `?hitless=1` for controller-side
changes (plain `/update` is the worker path).

## 2026-08-14 — infrastructure: mini05 retired as a models source (beast only)

Not a code change — controller-host configuration, recorded here because it changes boot config
(`/etc/fstab`) and removes an operational lever that older runbooks still describe.

### Changed

- **`mini05` (`10.10.100.35`) is no longer a models source for the controller.** Its export
  `/mnt/sda1/external/iM-models` is being deleted and the disk reclaimed by another project, so a
  failover to it would mount an empty or *foreign* filesystem at `/root/infinitemodel/models` and
  the controller would come up serving nothing. **beast (`10.10.100.38`) is the only source.**

  `/usr/local/sbin/im-models-source` on iM: the `mini05` and `auto` subcommands now **hard-refuse
  (exit 2)** with an explanation rather than being deleted, so a stale runbook or an old habit gets
  a reason instead of a bare `usage:` line. `status` no longer probes or advertises mini05. The
  `beast` arm and the `switch()` body are byte-for-byte unchanged.

- **The fallback had already stopped meaning anything.** It existed because *"beast gets shut down
  on hot days — then iM runs from mini05 alone."* But VM 116 (`iM`) has since migrated onto beast
  (`/etc/pve/nodes/beast/qemu-server/116.conf`, `local-zfs:vm-116-disk-0`), so beast being off
  takes the **controller** down with it. Storage cannot fail over from a host the VM dies with.
  Retiring it removes a false sense of redundancy, not real redundancy. Consequence accepted
  deliberately: controller VM and weights are now a single point of failure on beast, and the
  models tree is replicated nowhere — beast down means *fix beast*, not re-point the mount.

- **Three stale `fstab` backups on iM neutralized.** `fstab.bak-nfsv3` (models **and** cache),
  `fstab.bak-cache38-20260802-185834` (cache) and `fstab.jtbak` (models) all still carried live
  `.35` lines; a blind `cp` restore would have silently re-pointed the mount. Each such line is now
  prefixed `# RETIRED-mini05-20260814` — commented, **not deleted**, so the original text survives
  verbatim — and each file carries a header saying why. When retiring a source, grep the backups,
  not just the live file.

### Fixed — `switch()` destroyed before it validated

Recorded first as "noted, not fixed", then fixed in the same session once the failure was
demonstrated rather than merely suspected.

`switch()` stopped `im-controller`, unmounted, and rewrote `/etc/fstab` **before** it knew the new
mount worked. A failure therefore left fstab pointing at a source that does not work — and that
**survives a reboot**, so the box comes back with no models. Same destroy-before-validate shape as
the `#embed-pin` bug fixed earlier today: *a path that previously could not fail acquires a new
failure mode, and nobody re-checks what it already tore down.*

Now: a **read-only probe-mount at a scratch path** confirms the source both mounts *and* is
non-empty before anything is touched, and `/etc/fstab` is snapshotted so a late `mount` failure
**rolls back and restarts the controller** instead of leaving the box wedged. Probe options are
deliberately not the fstab options — `soft` with a 5 s `timeo` so a dead server fails the probe in
seconds rather than hanging forever on `hard`, and `_netdev`/`nofail` are fstab-only.

Verified in a sandbox that stubs `mount`/`umount`/`mountpoint`/`systemctl` and redirects fstab and
the mountpoint into a temp dir, so the destructive paths run without touching a real box —
**16/16 assertions across four scenarios**: happy path, source unreachable, source mounts but is
empty, and mount fails late. The two failure cases assert the property the old code violated:
fstab byte-identical and the controller never stopped.

A differential run of the *same* harness against the previous version is what justified the
change — on a late mount failure it leaves `fstab: MODIFIED` and `service: LEFT STOPPED`, where
the new one reports `fstab: INTACT (rolled back)` and `service: restarted`. The live switch was
**not** executed on the controller; only the read-only `status` path was run there.

### Verification

`im-models-source status` → `mounted : beast`, `service : running`, 35 model dirs. `mini05` and
`auto` both exit 2 with the mount, fstab and service **provably untouched afterwards** (`findmnt`
still beast, zero mini05 lines in fstab, `im-controller` `active`). Controller HTTP healthy: `iM`
v0.3.14, 40 models visible, 11 nodes. No timer, cron or autofs entry ever referenced mini05 — the
only path back was a human running the command, which is now closed.

### Undo

Previous script preserved at `/usr/local/sbin/im-models-source.bak-premini05retire-20260814`;
restoring it re-arms the `mini05` and `auto` arms. Per the above, those would now point at a
deleted export, so this is a rollback of last resort. The commented `.35` lines in the three fstab
backups can be reactivated by removing the `# RETIRED-mini05-20260814 ` prefix.

## 2026-08-14 — `#media-pin`: the same defect in all five media leaves

### Fixed

- **`node=` was ignored by `t2i`, `t2a`, `tts`, `stt` and `t2music` too.** The audit flagged after
  `#embed-pin`, and it found the identical defect five more times. Each leaf branches out of
  `_load_impl` before the node-filtering loop and picks from a bare capability comprehension; none
  ever *received* `pin_host` or `exclude_nodes`, and none consulted `_peer_claimed_host`. So
  `/load?model=kokoro&node=X` went wherever the heuristic pointed, two replicas could share a node,
  and a host another controller already claimed could be double-booked (`#federation` Phase 5).

  Same root cause each time: written as "find a capable node" and never revisited as placement grew
  pin / replica / federation semantics. One shared `_place_filter()` now applies all three
  constraints to an already-built candidate list **without touching each leaf's own preference
  ordering** — `can_X` first, then co-location, then VRAM stays each leaf's business. An
  unsatisfiable pin raises naming the survivors.

- **…and none of them could safely refuse, because all five unload before choosing.** Exactly the
  trap the `#embed-pin` fix hit. Every candidate predicate here is *static* (capability + tier
  toggles + co-location); only the *ranking* wants the VRAM the unload frees. So `tts`/`stt`/
  `t2music` have their candidate build hoisted above the unload and filtered there (selection uses
  static attributes only and stays put), while `t2i`/`t2a` run a validating `_place_filter` on the
  same static predicate before the unload and let their retry loop re-derive and re-filter for the
  VRAM-aware ranking — the loop must keep its own build, since it re-evaluates after eviction.

### Verification

Checked statically rather than hoped at, after two NameError-class slips earlier in the day:

- an AST check that no precheck references a local bound **later** in its own function. It caught a
  real one — `t2a`'s precheck called `_is_colo` 13 lines before its `def`, an `UnboundLocalError`
  at runtime and completely invisible to `py_compile`. Inlined the test.
- an AST check that across **all six** single-node load paths the first `_place_filter` /
  `_embed_candidates` call precedes the first `_unload_model_locked`, so validated-before-destroyed
  is machine-checked rather than asserted in a comment.

Then end to end on a real media leaf (kokoro on om3nbox): pin honoured → `InferenceEngine`; bogus
pin → refused with `candidates: InferenceEngine`; **the resident model survived the refusal** and
still served a 102 KB WAV. The refusal case deliberately used a nonexistent node name rather than
pinning at `furnace`, so a wrong filter could not have placed anything on an off-limits card.

## 2026-08-14 (latest) — `#embed-pin` and `#utc-logs`

### Fixed

- **`#embed-pin` — `/load?model=<embedder>&node=X` silently ignored the pin.** The embedding branch
  leaves `_load_impl` *before* the node-filtering loop the pipeline path runs, and `pin_host` was
  never passed to `_load_embedding_locked` at all; the node choice was a bare
  `alive_sorted()[0] that has a GPU`. So an explicit pin went to whichever node sorted first — seen
  live as `node=work` landing on `amdcomp`, with the response reporting `mode="pin:work/gpu"` while
  `stages[0].hostname` said otherwise. `work` sorts second-to-last among candidates, so the pin
  could never have been honoured by accident.

  Two more constraints were dropped on the same line: **`exclude_nodes`** (two replicas of one
  embedder would land on the *same* node, silently breaking the disjoint-copies contract that makes
  replicas add a concurrent decode slot) and **peer claims** (double-booking a node another
  controller owns is the exact double-reservation that OOMs it, `#federation` Phase 5). All three
  are now applied, and **a pin that cannot be honoured raises**, naming the candidates, rather than
  quietly placing elsewhere — a wrong-but-working placement is invisible downstream.

  Root cause worth naming: the slim loader was written as "pick a capable node" and never revisited
  as placement grew pin / replica / federation semantics. Being the simpler path did not make those
  constraints optional.

- **…and the regression that fix caused, caught by its own test.** `_load_embedding_locked` drops
  the resident copy *before* choosing a node, so making an unsatisfiable pin raise turned a typo'd
  node name into "unload the working embedder, then error" — `/api/embed` returned 0 dims with
  nothing resident. Previously this path could not fail, so the ordering never mattered. The filter
  is now one helper, `_embed_candidates()`, called twice: once **before** the unload purely to
  validate (fail with the copy still serving), once after to select (so the choice sees freed
  VRAM). Sharing the helper is the point — two hand-written copies could drift, and a load that
  passed validation then found no node would hit exactly the failure being fixed.

  ⚠ **General shape: when adding a refusal to a path that previously could not fail, check what
  that path has already destroyed by the time it refuses.**

- **`#utc-logs` — every node stamped its log lines in LOCAL time.** `time.strftime` with no `tm`
  argument uses the node's own timezone, so `GET /logs` across the fleet returned streams hours
  apart: the `.45` LXC workers run local, om3nbox runs UTC. Correlating one load across two nodes
  read as ~8 hours of stale logging when both streams were current. Now UTC everywhere with a
  trailing `Z`, so the format is self-describing rather than a silent clock jump for anyone reading
  old and new lines together.

  Four stamp sites kept in sync — `server.py`, `client.py`, `worker_quant.py` (code-split copy of
  the same shim) and `multimodal._vlog`, which writes the crash-survival vision log and would
  otherwise have been the one local-time stream left in an otherwise-UTC file. Safe to change:
  nothing parses the prefix (the `/logs` ring buffer is display-only). Verified live — controller,
  a `.45` worker and an om3nbox worker all stamping within seconds of `date -u`.

## 2026-08-14 (later) — `#head-quant`: an int4 body with an int8 head

### Added

- **`POST /load?head_quant=int8`** — sets the `lm_head`'s precision independently of the body.
  This is the trade #8 identified but could not take. On Qwen2.5-7B the head is **22% of
  everything read per decoded token** — the largest single tensor per token — and #8 measured
  int4 on it at +0.0389 nats / 84.7% top-1 (**92.5% of the damage the whole int4 body does**,
  rejected) against int8 at **+0.0050 nats / 98.4% top-1**, roughly 8x cheaper than the body's own
  cost. int8 was the right answer and was unusable until the w8a16 kernel existed.

  Measured on om3nbox (gfx1151), Qwen2.5-7B, ctx 4096, first run discarded as Triton autotune:

  | load | decode | prefill | VRAM |
  |---|---|---|---|
  | `quant=int4` | 32.06 tok/s | 827 tok/s | 5.37 GB |
  | `quant=int4&head_quant=int8` | **35.22 tok/s** | 833 tok/s | **4.87 GB** |

  **+9.9% decode** — matching the byte arithmetic (4.60 → 4.09 GB/token) almost exactly — with
  **prefill unchanged** and **0.5 GB less VRAM**. The census moves exactly as it should:
  `qlinears` 196 → 197, `qweight` 3.04 → 3.55 GB (+0.51 int8 head), `bf16params` 2.03 → 1.02 GB
  (only the embedding left in bf16).

  Plumbed like `#moe-offload` (`/load` → `engine.load` → `_load_impl` → install directive →
  `worker_load` → `Shard._head_quant`), including through `replicate()` so two copies of one model
  cannot answer with different head precision depending on routing, and through the `#load-faster`
  re-place snapshot so a rebuild does not silently drop it.

  Applied in `client._finalize_placement` rather than at build time: that is after final device
  placement and before both the CPU-linear wrapper and the `prepare_fused` sweep, so the new
  `QuantLinear` is never wrapped as a native Linear, the sweep binds w8a16 automatically, and
  **cold loads and serve-from-cache loads take the identical path** — no cache format change, no
  manifest change, no new packed tier on disk. The head is quantized **on CPU** whatever its final
  device (the intermediates are a couple of GB for a 152k-row head — enough to OOM a tight GPU at
  the end of a load, for a result half the size of its input), which also frees the bf16 copy's
  VRAM as a side effect. The planner budgeted a bf16 head, so a packed head uses *less* than
  reserved — the safe direction, no placement change needed.

- **Four refusals instead of silent no-ops**, each a case where the flag would otherwise look like
  it worked: head on **CPU** (the kernel is GPU-only, and on CPU an int8 head is *slower* than
  bf16 — quantizing there is a pessimization); **tied embeddings** (`#tied-dedup` already points
  `head.weight` at `embed_tokens.weight`, so packing adds a copy rather than replacing one); head
  **already packed** by the int8 tier; and `head_quant=int4`, which raises citing the measurement.
  Any failure leaves the head exactly as built — a head pack must never cost a load.

- **`perf_profile` advises it, and deliberately never applies it.** This is the one knob in the
  resolver that changes *output*, so it stays an explicit operator choice; the rationale line
  carries the measured numbers and says why it was not taken automatically. The not-applicable
  branches name the specific reason (tied / not int4 body / no GPU).

### Fixed

- **`#load-faster` silently dropped `moe_offload` on re-place.** Found while copying the snapshot
  pattern: the snapshot READ `getattr(m, "moe_offload", False)`, but `moe_offload` is not a
  `LoadedModel` field and nothing ever assigned it — so the read always returned `False` and a
  re-placed MoE model lost its expert-offload split. Both `moe_offload` and `head_quant` are now
  recorded on the instance at construction.

- **`[head-quant] FAILED (NameError: name 'torch' is not defined')`** — caught live on the first
  deploy. `client.py` has no module-level `torch` in that scope; `Shard` carries `self.torch`. The
  load completed correctly with a bf16 head, which is the guard doing its job, but the feature was
  a silent no-op until fixed.

## 2026-08-14 — w8a16: the int8 tier stops being an anti-speed tier

### Added

- **Triton w8a16 int8 GEMM + split-K GEMV (`worker_quant`).** int8 weight-only quant shipped as a
  pure *memory* tier and was actively **anti-speed**: `QuantLinear.forward` did
  `qweight.to(x.dtype) * scale` then `F.linear`, so one decode step read the int8 weight, **wrote a
  full bf16 copy of it, and read that copy back** — roughly 2.5x the traffic of simply keeping the
  weight in bf16. That is why the #8 investigation found int8 to be the quality-safe head quant and
  still unusable: the kernel did not exist.

  Measured end to end on om3nbox (gfx1151), Qwen2.5-7B, ctx 4096, distinct prompts per run so
  `#prefix-kv` cannot fake a prefill number, first run discarded as Triton autotune:

  | tier | decode | prefill |
  |---|---|---|
  | int8, no kernel *(before)* | 2.83 tok/s | — |
  | **int8 + w8a16** | **25.00 tok/s** | **1180 tok/s** |
  | int4 *(existing tier)* | 32.06 tok/s | 827 tok/s |

  **8.8x on decode.** int8 went from unusable — ~11x slower than int4 on the same box — to
  competitive with int4, while being far more accurate than it (#8 measured int8 at +0.0050 nats /
  98.4% top-1 agreement, against int4's +0.0421 / 88.1%). The int8 tier is now a real choice
  rather than a memory-only fallback. int8 prefill also came out **43% faster than int4's**, which
  is a dequant-once+BLAS path already tuned for this box.

  The int8 format makes the kernel both simpler and *more accurate* than its int4 twin: because
  the scale is **per output row** it factors straight out of the K loop, so the inner loop is a
  plain dot product and the scale is applied once after an fp32 accumulation. `w4a16` must
  dequantize to bf16 *inside* its loop; this one never does. int8 values (|q| ≤ 127) are exactly
  representable in bf16's 8 mantissa bits, so converting `q` to feed `tl.dot` loses nothing.

  Split-K is what makes decode fast: at M=1 a `tl.dot` kernel launches only ~`cdiv(N,BN)` programs,
  far too few to hide memory latency on an iGPU. Splitting K and atomic-adding fp32 partials grows
  the grid ~SPLITKx. Scaling a *partial* is exact **only** because the scale is per-row —
  `sum_i(acc_i·s) == s·sum_i(acc_i)`; a group-wise scale could not be hoisted this way.

### Changed

- **The kernel is gated per M-REGIME, not bound globally.** The first cut benchmarked M=1 and then
  used the kernel at every M, quietly assuming the decode result carries to prefill. It does not:
  at large M the GEMM is compute-bound and the naive path's one-off dequant amortizes over every
  row, so dequant-once + a vendor BLAS can beat reading int8 inside a `tl.dot` — the same crossover
  the int4 tier already handles with `_naive_m_min`. `prepare_fused` now benchmarks **both**
  regimes (M=1 and M=256) and records `_w8_m_max`; the chosen scope is logged. On gfx1151 the
  kernel wins both (`26.86x at decode, 1.71x at M=256 -> decode+prefill`), so nothing changes
  there — but that is a measurement about one device, which is precisely why the gate exists.

- **`prepare_fused` decides by BENCHMARK, not by self-check alone.** "Reads fewer bytes" does not
  imply "faster": on a discrete NVIDIA card cuBLAS bf16 is fast enough that dequant-once can win,
  which is how torch's own `_weight_int8pack_mm` measured 2-5x *slower* than bf16 `F.linear` on
  CUDA. A device where the kernel loses keeps the naive path and says so in the log.

- **`[int4-vram]` census label `QuantLinear4=` → `qlinears=`.** The counter is "any module with a
  `qweight` buffer", so an int8 7B logged `QuantLinear4=197` on a shard holding exactly zero int4
  linears. (197 = 28 layers x 7 + the `lm_head`, confirming int8 quantizes the head;
  `qweight=6.58GB` is the int8 bytes, `bf16params=1.02GB` is the embedding, bf16 by design.)

### Verification

- `scratch_w8a16_test.py` on an RTX 3060: every shape passing at M ∈ {1,2,4,16,129,512} with rel
  err ~0.0017 (bf16 output rounding — the reference is bf16 too) against
  `F.linear(x, q.bf16 * scale)`. Shapes include Qwen2.5-7B's `lm_head` (152064x3584) and two
  **deliberately unaligned** cases (1000x999, 17x130) whose dims are not multiples of any BN/BK in
  the autotune space, so masking bugs cannot hide behind tidy dimensions. The GEMV is called four
  times against a fixed reference — what a missing `reset_to_zero` would fail.
- On the actual gfx1151 device at load: **zero self-check failures and zero fallbacks across all
  197 quantized linears** of the 7B, kernel bound to all of them including the `lm_head`.
- ⚠ `reset_to_zero` on the GEMV is load-bearing — it atomic-adds, so autotune's timing reruns
  would otherwise accumulate into the same buffer and corrupt the first call per (N,K). The int4
  twin shipped without it and the bug hid behind its load-time self-check absorbing the corrupted
  launch. That is why this one self-checks at **M=1 and M=4**: the two kernels are different code
  paths and an M=4-only check cannot see a broken decode path at all.

## 2026-08-13 (latest) — #8 int4 `lm_head` MEASURED AND REJECTED

### Not changed (deliberately)

- **The `lm_head` stays bf16 on every quant tier.** #8 proposed packing it int4 on ROCm, where
  decode is bandwidth-saturated (~140 GB/s measured) and the head is **1.02 GB of the ~4.6 GB read
  per decoded token — 22%**. The speed case was sound and the implementation was small (the
  `prepare_fused` sweep already covers `self.head`, so a `QuantLinear4` head picks up the shipped
  Triton `_ksk` GEMV with no new kernel). It was built, measured, and **thrown away**, because the
  quality cost is not affordable.

  Measured on Qwen2.5-7B-Instruct (a real, untied head), 750 next-token predictions of held-out
  prose, using the project's OWN `pack_linear_int4` round-tripped through `worker_quant`'s exact
  `_dequant` convention:

  | variant | ppl | ΔNLL | top-1 vs ref | KL |
  |---|---|---|---|---|
  | bf16 body + bf16 head (reference) | 10.72 | — | 100% | 0 |
  | bf16 body + **int4 head** | 11.23 | +0.046 | 84.3% | 0.060 |
  | int4 body + bf16 head (**today**) | 11.18 | +0.042 | 88.1% | 0.070 |
  | int4 body + int4 head (proposal) | 11.63 | +0.081 | 80.5% | 0.124 |

  **One head tensor costs as much as all 196 decoder tensors combined** — the head's marginal
  ΔNLL (+0.0389) is **92.5% of the damage already shipped and accepted** (+0.0421), and it changes
  **15.3% of greedy tokens**. The trade on offer was: double the total quantization damage to buy
  16.2% decode. Declined.

- **A narrower group does not rescue it.** Group size is nearly free in bandwidth terms (g128 →
  g16 costs only 2.4 points of the win, 16.2% → 13.8%), so it was the obvious lever. Swept on the
  same trunk: KL falls cleanly and monotonically (0.0599 → 0.0495 → 0.0424 → **0.0241** at g16) and
  top-1 agreement rises (84.7% → 90.9%), but even g16 still flips ~9% of greedy tokens at ~+0.042
  nats — still comparable to the entire body. ⚠ **ΔNLL is NOT monotonic across the sweep** (g64
  worse than g128, g16 worse than g32): at 750 predictions the per-variant NLL is too noisy to rank
  group sizes against each other. KL and top-1 agreement are per-token paired and far less noisy —
  read those columns, not ΔNLL, for the trend.

- **int8 is the quality-safe head quant and is a SPEED PESSIMIZATION here.** Per-row int8 costs
  only **+0.0050 nats** with **98.4%** top-1 agreement (KL 0.0023 — ~25x less damage than
  int4-g128). But `QuantLinear.forward` materializes the full bf16 weight every call, so an int8
  head reads 0.55 GB and then writes+reads 1.09 GB: strictly worse than leaving it bf16. There is
  no w8a16 kernel in the tree, only w4a16.

  **This is the real opportunity, and it needs a kernel, not a config change:** a w8a16 Triton
  GEMV would make the head 0.51 GB (an ~11% decode win on gfx1151) at a quality cost of +0.005
  nats — essentially free. That is the shape #8 should take if it is ever revisited.

### Added

- **`scratch_head_int4_quality.py`** — the gate this needed, kept because the question recurs.
  Compares bf16/int4 body × bf16/int4 head over real text and reports NLL, perplexity, greedy
  agreement and KL, framed against the body's already-accepted damage so the number is judgeable
  rather than arbitrary. `--sweep` walks group sizes, `--sweep-only` does it on one trunk pass,
  and `--werr` is a seconds-long weights-only screen (per-tensor round-trip error, kurtosis,
  max/p99.9) needing no model execution. Runs on CPU on any worker venv.

  ⚠ **The `--werr` screen predicted the OPPOSITE of the truth, and that is the durable lesson.**
  The 7B head round-trips at 0.1189 mean relative error — squarely inside the body's 0.1097-0.1303
  band, with *lower* kurtosis (13.9) than `L0.gate_proj` (131). By weight statistics it is an
  ordinary tensor. Yet it alone does as much damage as the whole body. The reason is structural:
  body errors pass through later layers, residual adds and RMS norms that partially cancel and
  re-normalise them, while the head's error lands directly in the logits with nothing downstream.
  **Position in the network dominates weight statistics** — "this tensor quantizes like the others"
  is not evidence that quantizing it is safe. (The router-gate exclusion is the same lesson.)

## 2026-08-13 (latest) — #ntpen: penalised requests keep the reduced wire

### Changed

- **#ntpen — `repeat_penalty` no longer disables `#logits-diet`.** The diet ships the head's
  ANSWER (a greedy token id, or the top-K candidates) instead of the full ~304 KB vocab row, every
  decoded token. But repetition penalties need arbitrary-id access to that row, and only the
  controller had the token history — so `_decode_plain` simply turned the diet OFF whenever
  `repeat_penalty` / `presence_penalty` / `frequency_penalty` was set. **Every Ollama client sends
  `repeat_penalty` by default**, which made the diet dead code for the common case: the wire paid
  ~304 KB/token to save an id window that is ~400 bytes.

  The fix inverts it — the history window goes to the head, not the row to the controller. The
  controller reduces the two windows to plain id lists (`rp_ids`; `fp_ids`+`fp_cnt`) and rides them
  on the frame header; `shard_forward._diet_penalize` applies the exact same arithmetic as
  `engine_gen._penalized` on the head's device, **before** the argmax/top-K. Penalising before the
  top-K is load-bearing, not stylistic: penalties reorder the distribution, so a top-K taken from
  the unpenalised row would be the wrong candidate *set*, not merely mis-scored.

  Measured on `qwen2.5-1.5b-instruct` int4, one RTX 3060 (single-stage, so the head reply is the
  only wire cost), 400-token greedy generations, conditions ALTERNATED within one run so fleet
  drift cannot masquerade as the effect — **before: penalised 52.59 tok/s vs unpenalised 56.94, a
  7.6% penalty for having penalties on** (ranges 52.09-53.30 and 56.02-57.82 — non-overlapping).
  Directive size at the default `repeat_last_n=64` is **319 bytes of header vs 303,872 bytes of
  bf16 row**; at `repeat_last_n=-1` (whole history) it is still only 23 KB on a 4096-token prompt.

- **Mixed-version safety, three independent guards** — because unlike every other wire cap, this
  one's failure mode is silent WRONG OUTPUT. A node that accepts the directive and ignores it
  returns a perfectly well-formed top-K of the *unpenalised* row; nothing downstream can tell.
  So: (1) the new `ntpen` cap, all-or-nothing across the chain, same doctrine as `ntdiet`;
  (2) **new mode names** `argmaxpen`/`topkpen` rather than an extra key beside `argmax` — a
  worker of `ntdiet` vintage gates on the mode string, so an unknown mode makes it reply the full
  row, whereas an ignored extra key beside a *known* mode would have answered unpenalised. The
  directive is also refused when `nt_pen` is missing, which is exactly what such a node's
  intermediate stage does to it (it rebuilds the next-hop header from a hardcoded `nt_*` tuple);
  (3) `shard._pen_capable`, for a stale `shard_forward.py` under a fresh `worker_net.py`. Every
  one of those degrades to the full row, which the controller still accepts and penalises itself —
  so the diet can only ever lose bandwidth, never correctness.

- **Equivalence validated before deploy, not after.** `scratch_ntpen_test.py` compares the head-side
  reduction against the controller's legacy `-inf`-mask + `_penalized` path over 720 adversarial
  trials — fp32/bf16/fp16 x {512, 2048, 32000} vocab x {argmax, topk} — with planted ties, logits
  straddling zero (where `repeat_penalty`'s divide/multiply branch flips) and a beyond-clip token
  planted as the global max (the `#21` case the clip mask exists for). argmax must be bit-exact;
  top-K must match the controller's penalised row elementwise. 720/720.

## 2026-08-13 (later) — #cuda-large-m, the ⚡ Optimized-settings button, perf-auto fixes

### Changed

- **#cuda-large-m — CUDA int4 stopped running a batch-1 kernel at prefill.** `prepare_fused` set
  `_naive_m_min` only on the ROCm branch and freed `qweight` on CUDA the moment the fused pack was
  built, so `_dequant` — and with it the whole dequant-once+cuBLAS fall-through — was
  **structurally unreachable at any M**. ROCm measured that the fused kernel loses at prefill row
  counts and shipped a fallback; CUDA never made the comparison possible. Measured on an RTX 3060
  (sm86, 2 MB L2 chosen so a 32 MB weight cannot sit in cache and report bandwidth it never paid
  for), M=2048, real Qwen2.5-7B shapes, **with the int4 packing verified against the naive dequant
  to rel=0.0025 before any timing was taken**: q/o 8.51→3.16 ms (2.69x), k/v 1.18→0.47 (2.54x),
  gate/up 44.42→16.21 (2.74x), down 45.96→15.92 (2.89x) — ≈154 ms/layer fused vs ≈56 ms naive,
  **~2.8 s per 2048-token prefill chunk** across 28 layers. Keeping `qweight` costs the packed bytes
  a second time (mat2 is the same size and its interleaved layout has no `_dequant`), so the
  **worker decides per-linear from its own `mem_get_info`**: retain while the device has room for 2x
  the tensor plus 1 GB slack, stop when it does not. A partially-armed shard is fine and strictly
  better than all-or-nothing. `IM_KEEP_QWEIGHT=1/0` forces it. Decode untouched (`_m_bucket(1)==1`
  can never reach the threshold); numerics unchanged (the fall-through *is* the self-check's own
  reference path). Verified live: `fallback ARMED (naive at row-bucket>=512)`, `qweight=0.61GB`.

### Added

- **"⚡ Optimized settings" in the Load screen** + a visible **KV slots** control (the dialog had
  none, so the knob measured to matter most for conversational traffic could only be set by editing
  a URL). New `GET /optimize_knobs` is a pure dry-run — it resolves nothing on the engine and loads
  nothing — returning the recommended knobs **plus the reason for each**; the button fills the form,
  prints a `changed: X → Y` summary and the full rationale, and stops there so the operator can
  override anything before pressing Load. Showing the reasons is the design point: several right
  answers are counter-intuitive and only defensible by measurement.

### Fixed

- **perf-auto no longer fails silently.** The resolver was wrapped in `contextlib.suppress`, which
  made "it threw" indistinguishable from "it decided nothing" — and cost two deploy cycles to
  notice. Now a loud `try/except` with an activity line, plus an else-branch explaining a skip.
  This is what surfaced the `ModuleNotFoundError` below.
- **perf-auto sized `kv_slots` against the wrong node.** It picked the roomiest node in the fleet
  even for a pinned load (observed: a `work`-pinned load sized against `amdcomp`). It fit by luck,
  but could fund `C x` full-ctx KV the target cannot hold — exactly the CPU spill the conservative
  sizing exists to prevent. Now filters candidates by `pin_host`.
- **perf-auto called MoE models dense.** It read `spec.is_moe`; `ModelSpec` has no such field, so
  `getattr(..., False)` turned "cannot tell" into a confident wrong claim (`qwen3.6-35b-a3b` was
  reported as "dense model — no expert tier to offload"). MoE-ness is only established worker-side,
  so it is now **inferred** via `perf_profile.looks_moe()` from measured-vs-dense per-layer bytes,
  and "not downloaded" is documented as *unknown*, not dense.

### Operational notes learned the hard way

- **Adding a file to `EXTRA_UPDATE_FILES` needs TWO `/update`s.** The update runs the *old*
  `server.py`, whose manifest predates the new module — om3nbox failed with
  `ModuleNotFoundError: No module named 'perf_profile'` until a second update.
- **Worker-side changes need an explicit `POST /restart_node`.** `/update` reports
  `worker_restart: false` (VERSION-gated): workers fetch the file but keep running old code.
- **`POST /update` unloads every resident model and `/api/embed` does NOT autoload** — reload
  embedding models by hand afterwards or dependent services take 503s.
- **Never benchmark the first generation after a ROCm int4 load** — Triton autotune measured
  5.68 tok/s against a 33 tok/s steady state.

## 2026-08-13 — #honest-durations, #prefix-min-128, #perf-auto, #large-m-descend

A measurement-driven performance pass. Its most useful output was **negative**: three
plausible-sounding optimizations (CUDA graphs, lm_head int8 on CUDA, `torch.compile`) were
implemented, benchmarked, and **reverted** because each measured *slower*. The wins that survived
are below, and none of them is in decode — a paired interleaved benchmark showed iM's decode is
already within ~12% of raw HuggingFace running the identical modules on the identical GPU, so the
time was never there to reclaim.

### Fixed

- **#honest-durations — `eval_duration` no longer includes prefill.** Every API emit site shipped
  `prompt_eval_duration: 0` and `eval_duration = total_duration`, i.e. queue + prefill + decode.
  Ollama reports the two *separately* (its `eval_duration` is decode only), so any client comparing
  the two APIs was comparing `tokens/(queue+prefill+decode)` against `tokens/decode`. Measured on one
  card and one model, varying only prompt length: iM under-reported its own decode rate by **1.10x at
  P=37, 1.47x at P=1959 and 2.47x at P=4196** — the decode rate was flat throughout; only the
  reported number collapsed. `engine_gen` now stamps `t_gen0` (after the slot is held, so queue wait
  is excluded) and `t_first_tok` on the INFLIGHT record; `serving._split_durations` derives the two
  fields from them. `total_duration` keeps its meaning, so `total - prompt_eval - eval` is the queue
  wait — the same shape Ollama has. Degrades to today's behaviour when either stamp is missing.
  *This also fixes a latent divide-by-zero in clients that compute `prompt_eval_count /
  prompt_eval_duration`, and means the repo's own harnesses (`bench_tp_crossover.py`, `tp_bench.py`,
  `load_verify.py`) stop reading a prefill-contaminated field as "decode tok/s".*
- **#prefix-min-128 — prompt-prefix reuse was inert.** `INFINITEMODEL_PREFIX_MIN` defaulted to 1024
  tokens while the mean real prompt on the live replica is 353, giving a **measured 0% hit rate over
  1816 served requests** (a 30.5-minute log window with 198 first-token lines and zero `prefix-kv`
  entries). Reproduced on an idle card: a 2594-token prompt HITS (1814 ms vs 6499 ms cold) while the
  identical test at 385 tokens MISSES. Default lowered to 128; the `>=16` clamp already provided the
  typo protection the 1024 value was justified by, and a miss costs at most one crop round-trip.

### Added

- **#perf-auto — setup-aware knob resolution at load time** (`perf_profile.py`, new). A pure,
  torch-free decision table that takes the detected setup (device class, live-free VRAM, model shape,
  MoE/hybrid/multimodal, draft availability, ctx) and returns resolved knobs *plus a rationale line
  per decision*, which is printed into the load log. Device class is the axis that matters most
  because it selects the int4 kernel: CUDA sm80+ reaches torch tinygemm, ROCm reaches the project's
  Triton w4a16, and **CUDA sm<80 reaches neither** — there `QuantLinear4` silently falls back to
  rematerializing the whole bf16 weight every forward, so int4 is a memory tier and never a speed
  tier. Explicit per-load values always win; `/config?perf_auto=0` disables it. Only knobs backed by
  a measurement are *applied* today (see below) — the rest are logged as advice, deliberately, since
  this pass established that untested tuning reverses under measurement more often than not.
- **`kv_slots` is now auto-raised when it provably fits** (`_perf_auto_kv_slots`). `#prefix-kv` keeps
  one prefix record *per slot*, so at `kv_slots=1` a single interleaved request — a second client, an
  agent side-query, a keep-alive probe — evicts it and every later turn re-prefills the whole
  conversation. Measured with a 2594-token prompt: `C=1` interleaved **6506 ms (MISS)** vs `C=3`
  **1665 ms (HIT)** = **3.9x**. `_SlotLease` already routed each request to the free slot with the
  longest common prefix, so `C>1` upgrades a depth-1 record into an associative prefix cache with no
  new machinery. Sized conservatively: raised only when quantized weights + `C x` full-ctx KV fit one
  node's live-free VRAM with 25% slack, because overshooting pushes layers onto CPU (far worse than a
  prefix miss) or fails the load outright. **First live validation of `kv_slots>1`**: three concurrent
  conversations produced bit-identical greedy output to the sequential baseline.

### Changed

- **#large-m-descend — prefill stopped running the decode-tuned int4 kernel.** The fused-vs-naive
  dispatch threshold was hard-coded to `_m_bucket(INFINITEMODEL_PREFILL_CHUNK)` = 2048, so every
  prompt below ~2048 tokens kept the decode-tuned Triton kernel at prefill — and the live mean prompt
  is 353 tokens, meaning essentially *all* production prefill used the wrong kernel. That kernel's
  grid re-reads the whole packed weight `cdiv(M,16)` times, so its cost is linear in M and can be
  extrapolated from the single measurement already taken; the naive side is then timed for real at
  each descending bucket (plain BLAS, no JIT, so **no new Triton shapes are probed** — the constraint
  that forced the one-point bench originally). Grounded against the live bench lines, the crossover
  lands near M=256 rather than 2048. Off-switch `IM_LARGE_M_NAIVE=0` unchanged; numerics unchanged
  (the fall-through *is* the self-check's own reference path).

## Recent — VRAM-accounting hygiene + KV-quant in the Load UI

- **Media render errors are now legible** — a t2a / t2music / t2i / tts render that failed *after* the
  model was already resident used to `print()` to controller stdout and return a **content-free 500**,
  so a caller saw only "500 Internal Server Error" and nothing landed in `/logs`. New
  `_media_render_error` helper (routes_api.py), wired into the music, images and speech render paths:
  it writes the failure to the **activity log**, and reports a CUDA/HIP **out-of-memory** *explicitly*
  as `insufficient_vram` (`code: out_of_vram`, still HTTP 500) with the raw GPU detail and operator
  guidance (free the pool / use a smaller model) — the #1 media render failure, since an offloaded
  model hops onto a GPU per render and a card full of co-resident LLMs has no transient room. iM never
  evicts a running model or grows VRAM; this makes the *cause* visible instead of a naked 500.
  Reporting-only — placement / OOM behaviour is unchanged.
- **#reservation-reconcile** — the controller's in-flight-load ledger (`engine._reservations`) is the
  planned per-node VRAM/RAM that *every* placement subtracts (`_reserved_bytes`) so two concurrent
  loads never over-provision a node. Each entry is set at load start and popped in the `load()`
  finally; a load cancelled/killed on a path that skips that finally — or a sub-load (replica/TP/media)
  whose key the top-level wrapper never pops — **leaked** the entry, so placement kept subtracting VRAM
  no worker actually held and a node read **"no room"** until the controller was *restarted* and the
  in-memory dict cleared (the "workers appear to hold fake VRAM; a restart frees it" symptom). A new
  20 s controller sweep (`control_plane.reservation_reconcile_loop`) does what the restart did, **live**:
  it frees any reservation whose owning load is provably over (its `_loading_tasks` handle finished) or
  that has out-lived a TTL with no live owner, returning that phantom budget without a restart. It
  **never** touches a reservation whose load task is still running (a genuine in-flight load), clears
  the stale "loading" card, wakes anyone queued behind the ghost load, and logs each drop with the GB
  it returns. Controller-only, deploys hitless. Tunable (default on): `reserve_reconcile`,
  `reserve_ttl_s` (600 s). (The **worker**-side half — an on-demand `empty_cache` endpoint plus
  `expandable_segments:True` to stop allocator *fragmentation* at the source — is proposed, not yet
  built.)
- **KV quant in the Load screen** — the dashboard Load dialog now exposes a **KV quant** selector
  (`none / turbo4 / turbo3 / turbo2`) beside the KV-cache-location control, forwarding `kv_quant` to
  `/load` (previously API-only). turbo4 is near-lossless and keeps a big context **on-GPU** without
  spilling weights — the right lever (over `kv_offload`, which shuffles KV over PCIe every token) for
  fitting full context on a single discrete GPU at 0 % CPU. (The "Preview fit" estimate still sizes KV
  at bf16, so it under-counts what turbo saves — a follow-up.)

## Release 0.3.6 — text-to-music (MusicGen)

- **#t2music-serve** — InfiniteModel now generates music from a text prompt with **MusicGen**
  (Meta), a second music engine deliberately chosen for a DIFFERENT architecture than ACE-Step
  (`#t2a-serve`): MusicGen is an **autoregressive transformer** over discrete EnCodec audio tokens,
  not latent diffusion. That is exactly what lets it run where ACE-Step can't — it ships inside
  `transformers`, needs **no `torchaudio`** (soundfile writes the WAV), its heavy compute is the
  transformer decode (MIOpen-free), and its only conv is EnCodec's one-shot decode (a bounded
  JIT-once tax like Whisper's encoder). So it serves on **AMD (ROCm) / NVIDIA (CUDA) / CPU**, with a
  real GPU→CPU fallback. New single-node media leaf `worker_t2music.py` (`MusicGenPipeline`,
  kind `t2music`); the prompt travels the control link and the finished WAV returns as base64
  (**#media-anywhere**, so any capable worker serves it). Exposed at `POST /v1/audio/music` — now
  polymorphic, dispatching to MusicGen or ACE-Step by the loaded model's type — with knobs
  `duration / guidance / temperature / top_k / seed`. Detected by config `model_type: musicgen`
  (or `musicgen_melody`); served WHOLE from its HF-cache snapshot like Kokoro (MusicGen ships
  `pytorch_model.bin`, no safetensors); badged `t2music`; placement honors the per-node tier opt-out
  (audit #28), keeping music off benched/off-limits nodes. The model-detail page gains a **Generate
  music** panel — prompt + controls, an inline `<audio>` player, a WAV download, and a rich info
  block (variant, backend, codec, token rate, device). Verified live on gfx1151 (om3nbox): medium
  renders ~2× realtime warm after the one-time EnCodec JIT. Sizes: `musicgen-small` 300M,
  `-medium` 1.5B, `-large` 3.3B, `-melody` 1.5B.
- **Add-model `.bin` fix** — `+ Add model` / `POST /add_model` now fetches `pytorch_model.bin` for
  repos that ship **no** safetensors (gated on that, so ordinary checkpoints never pull a redundant
  `.bin`) — a MusicGen model now downloads its weights instead of just `config.json`.
- **Duration cap (0.3.7)** — MusicGen's audio-token decoder has a fixed position table
  (`decoder.max_position_embeddings` = 2048 ≈ 40 s at 50 Hz). Requesting more (e.g. 45 s) indexed
  **past** it → a **CUDA device-side assert that wedges the worker** (its CUDA context is then
  corrupt until restart), or a CPU crash. Renders now hard-clamp `max_new_tokens` **and** the
  requested duration to the model's real ceiling, and the dashboard exposes that ceiling as the
  duration input's `max`.

## Release 0.3.2 — speech-to-text (Whisper)

- **#stt-serve** — InfiniteModel now transcribes speech. A Whisper checkpoint
  (`WhisperForConditionalGeneration`) loads as a single-node **media leaf** — the ASR sibling of
  Kokoro TTS (`#tts-serve`) and ACE-Step music (`#t2a-serve`) — and never touches the decoder-only
  pipeline (Whisper is a small encoder-decoder seq2seq model). New OpenAI-compatible endpoints
  `POST /v1/audio/transcriptions` and `/v1/audio/translations` (multipart `file`, or raw audio in the
  request body with `?model=` for a python-multipart-free path; `response_format=text` returns bare
  text). The worker decodes the audio (soundfile → 16 kHz mono, chunked at Whisper's 30 s window) and
  returns the transcript over the control link, so an STT model works **#media-anywhere** on any
  capable worker (a remote one `snapshot_download`s the checkpoint itself). Detected by config
  `model_type: whisper`; badged `stt`; the worker control reader was widened to carry the audio frame.
  Placement honors the per-node tier opt-out like the t2a filter (audit #28): a node with **both**
  tiers disabled in `NODE_CONFIG` (a deliberately benched box — e.g. an off-limits GPU) never
  receives a transcription, even if it advertises `can_stt`. Device: CUDA runs on the GPU (fast,
  warms on load). gfx1151/ROCm also runs on the GPU — but its first GPU inference JIT-compiles the
  MIOpen conv kernels (~8 min, uncached on TheRock 7.13), so the load is slow ONCE per worker while
  transcription is then GPU-fast (RTF ~1–3×); the warmup is deferred off the load so it doesn't race
  a restart. `INFINITEMODEL_STT_CPU=1` forces the instant-load / ~30×-realtime CPU path instead.

## Release 0.3.1 — multi-controller federation

First version tagged with a semantic version (earlier builds used internal `0.2-m4cNNN` deploy
counters). Headline: a fleet can now run **more than one controller**, and they cooperate.

- **#federation** — controllers discover each other over the existing UDP discovery channel, gossip
  inventory, and can **borrow** a model a peer has resident (a request for a model not loaded here is
  proxied to the peer that has it — one copy of the weights, either controller as the front door).
- **#unified-fleet** — either controller renders the *whole* fleet (its own nodes + models plus its
  peers'), with the owner's real cards/graphs, and can drive load/unload anywhere by federating to
  the owner.
- **Exclusive node ownership + `/peer_handoff`** — a node (and its live shards) belongs to one
  controller at a time and can be handed to another with no reload; `node=*` moves the whole fleet.
- **Controller failover** — if a controller dies, its workers re-home to a surviving controller
  (`INFINITEMODEL_DISCOVERY_RESPOND=standby`) which adopts their resident models with no reload.
  Live-verified end to end.
- **Peer model pull** — copy a model's weights controller-to-controller instead of from HuggingFace
  (resumable; the whole on-disk catalogue is pullable, not just resident models).
- **Zero-config discovery** — `controller_host: "auto"` is the default; workers find the controller
  by broadcast, so a fresh clone runs with no config edit.

## Distributed core
- Node registry + heartbeat + capability probe; live dashboard; RAM/VRAM-weighted partition planner.
- **Pipeline parallelism** over a hand-rolled plain-TCP transport (Windows + Linux): each worker holds
  a contiguous layer range; weights are **chunk-served** from the controller straight into worker RAM
  (no model on worker disk, no temp files).
- Incremental **KV-cache decode** (prefill-once, flat per-token cost); networked generation with full
  Ollama / OpenAI / Anthropic-compatible API surface.
- **Tensor parallelism** within a stage — capacity-proportional, GPU+CPU mixed meshes, KV-head
  replication; in-place reconfigure between pipeline and TP.
- **Speculative decoding** (opt-in, greedy-exact) — draft proposes K tokens, the pipeline verifies in
  one traversal; wins on big/distributed targets (measured). A checkpoint-MTP (nextn) *self*-draft
  for Qwen3.6 was built and the forward validated (~84-88% accept), but shelved: the hybrid
  Gated-DeltaNet trunk can't roll back its recurrent state on reject (not bit-exact) and a 2-token
  verify costs ~2x on the compute-bound GPU pipeline (no wall-clock win). Code kept, gated off.
- **#loopback-nexthop — LAN-visible next-hop wiring (2026-07-10).** A worker co-located with the
  controller advertises a **loopback** data endpoint (fastest for the controller's own stage-0
  dials) — but handed verbatim to a **remote** stage as its next pipeline hop (or TP mesh root),
  `127.0.0.1:<data_port>` made that remote stage dial **itself**: every stage output looped
  straight back into its own input (stage 0 then ate its own bf16 hidden state as "token ids"),
  and even the data-plane error frames cycled forever on the self-hop — the engine of a silent
  wedge storm that only struck placements with a mid-chain hop *into* the controller's box (rare:
  the planner usually seats that node first, so it masqueraded as intermittent "worker state
  poisoning" for a day). Fixed at wiring time (`Engine._lan_visible_host`, applied to pipeline
  `next_host` + TP `tp_root_host`): a loopback next-hop for a remote receiver is translated to the
  controller's address as that receiver already reaches it (its control-link sockname; fallback:
  first LAN IP); a receiver on the controller's own box keeps the loopback (correct and fastest).
  Caught within minutes by the silent-wedge hardening below (the dtype door-guard named the
  looped frame; the control-link stage_error delivered it) — the two fixes together close both
  the cause and the blindness.
- **#load-default-quant — `/load` without a quant defaults to int4, not bf16 (2026-07-11).**
  The `/load` endpoint hardcoded `quant="none"` (bf16) as its default, so any API caller that
  omitted the quant loaded a full-size bf16 copy — on a shared box a 30B MoE became ~57 GB that
  spilled onto CPU and evicted its neighbours. That default was inconsistent with every other
  path: the dashboard load dialog defaults to int4, and auto-loads use `autoload_quant` (int4).
  An unspecified quant now inherits `autoload_quant` (normally int4); an explicit `quant=none`
  still loads bf16 on purpose.
- **#reap-close-link — reaped nodes' surviving control links get closed (2026-07-11).** A
  heartbeat-timeout reap only deleted the registry entry; if the worker's TCP connection
  *survived* the network blip that caused the missed heartbeats (half-open, or fully healed),
  the worker kept heartbeating into a socket whose node id no longer existed — and since
  registration only happens on a fresh connect, it stayed an invisible zombie forever. A
  morning LAN blip (2026-07-11 ~07:43) demonstrated it at scale: seven nodes reaped, three
  reconnected on their own (their sockets broke), and four — prodesk / steamdeck / work /
  zippy — sat orphaned for hours with healthy worker processes heartbeating on live sockets
  the controller ignored. Two-sided fix: the reaper now **closes the reaped node's control
  link** (the handler tears down; the worker's reconnect loop re-registers in seconds), and a
  heartbeat arriving for an **unregistered node id** drops the link as a belt (covers reap
  races and stale #77 duplicate connections).

## Memory, quantization & the shard cache
- **int4** (group-wise asymmetric, fused tinygemm GEMM) and **int8** (per-channel) load-time quant;
  serve-time dequant of **fp8** and **nvfp4** checkpoints. Selecting **int8 on a MoE auto-downgrades to
  int4** (with a loud log line): the int8 path only quantizes 2D Linears, so a MoE's fused-3D routed
  experts would otherwise stay bf16 → a near-bf16 footprint (OOM/CPU-spill); int4 packs the experts.
- **int2 (#int2) — a 2-bit CAPACITY tier.** `quant=int2` on `/load`/`/reconfigure`/auto-load config:
  group-wise asymmetric 2-bit (4 values/byte, group 64 — finer than int4's 128 because 2-bit RTN
  needs it), ~2.5 bits/weight effective (~1/6 of bf16); head/embed/norms/router stay bf16 exactly
  like int4. The int4 architecture cloned end-to-end: naive dequant path everywhere (CPU big-M gets
  the fp32-GEMM treatment), a **Triton w2a16** batch + split-K-GEMV kernel (same autotune space and
  dram-dealias row pad as w4a16) as the fused decode path on **both CUDA and ROCm** (int2 has no torch
  tinygemm; no-triton workers self-gate to naive), self-checked vs naive at placement with automatic
  fallback (`IM_FUSED_INT2=0` kill-switch). **Shard cache included**: `_shards/int2/` compiles via the
  same shared bit-identical packer (`pack_linear_int2` == the worker's `_quantize_linear2` by
  construction), cache-on-first-load fires for int2 loads, serve-from-cache installs QuantLinear2
  holders directly. **Dense models only**: int2 on a MoE auto-downgrades to int4 (no 2-bit 3D-expert
  packer/kernel), mirroring the int8-on-MoE rule; MoE cache compiles reject non-int4 as before.
  Planner/status size the tier at 0.2× layer weights (`for_quant`), `/status` quant_gb/quant_fits
  carry an int2 entry, and the dashboard's Load + auto-load-default selects offer it. The shipped
  **auto-load default remains int4** — int2 is an explicit operator choice.
  **Measured quality verdict (2026-07-10, qwen2.5 0.5B + 7B, greedy):** plain round-to-nearest at
  2 bits **collapses the model** (token salad) — and stays collapsed at group 32/16, with per-group
  MSE-optimal clip search, and under mixed-tier salvage (down/o_proj + edge layers at int4 —
  grammatical but meaningless at best). This matches the literature: RTN-2bit is broken at any
  scale; 2-bit needs a **GPTQ-class calibrated packer** to be usable. The infrastructure shipped
  here is deliberately packer-agnostic — a calibrated packer emits the SAME qweight/scale/zero
  format through the same kernels, cache layout and serve path (a packer-only follow-up;
  `packer_hash` in the cache manifest auto-invalidates stale int2 caches when it lands). Until
  then int2 is machinery-complete but NOT usable for real serving.
- **int2 GPTQ-calibrated packer (#38, 2026-07-11) — the follow-up above, landed.** `gptq_pack.py`
  replaces the int2 compile with sequential per-layer **GPTQ**: Hessians `E[x xᵀ]` estimated per
  Linear from real forwards over a bundled offline corpus (`calib_corpus.txt`: public-domain novel
  + RFC 9110 + this repo's own Python; 32×512 tokens default, `INFINITEMODEL_GPTQ_SAMPLES/_SEQLEN/
  _PERCDAMP/_GRID` to tune), Cholesky error compensation column-by-column, group scale/zero by MSE
  shrink search, intra-layer stage order (qkv→o→gate/up→down, each stage seeing earlier stages
  already quantized), and each layer's QUANTIZED outputs feeding the next layer's calibration.
  Output format is byte-compatible with the RTN packer (same crumbs/scale/zero/g64), so
  QuantLinear2, the w2a16 kernels and serve-from-cache run it unchanged. `packer_hash` bumps to
  `v2-g<G>-int2-gptq` — v1 RTN caches fail verify with "recompile", and an int2 **load** without a
  valid v2 cache now FAILS LOUD with the compile instruction instead of silently falling back to
  the RTN cold path (which is salad). int2 stays explicit-compile (never auto-built on first load);
  /pack_probe + /compile_dist reject int2 (layer N's calibration needs layer N-1's quantized
  outputs — inherently sequential; compile subprocess uses the local GPU when present).
  **Measured**: synthetic activation MSE 25× lower than RTN; qwen2.5-0.5B/7B lift from token salad
  to grammatical, fact-retrieving output ("The capital of France is Paris") that still loops/
  degrades on open prose — consistent with GPTQ-2bit literature at small scale. The tier's real
  audience stays BIG dense models that otherwise cannot fit (a 70B at ~19 GB); at 7B-and-below,
  int4 exists and is strictly better.
- **Hybrid models reserve KV only on their attention layers:** a Gated-DeltaNet hybrid (qwen3-next /
  qwen3.6) grows a full-context KV only on its `full_attention` layers (the linear-attn layers keep a
  small fixed recurrent state). KV reservation — both the GPU placement budget and the pre-alloc probe,
  which mirror each other — now charges full-ctx KV only on the KV-holding layers, so more of a hybrid
  fits per card. Conservative (an unknown layer reserves full KV); dense models are bit-identical.
- GPU-first placement that always fits (spill to CPU/RAM), full-context KV pre-reservation, coexistence
  budgets, and OOM-safe replans (cgroup caps, honest transient accounting). Placement MODES: `auto`
  (GPU-first, fewest nodes — best decode latency), `single`, `gpu-spread` (fill every GPU then spill to
  CPU), **`all-gpu`** (a stage on EVERY GPU, NOTHING on CPU — proportional across the GPU subset so each
  card carries >=1 layer; fails cleanly if the model won't fit GPU VRAM alone), `distribute`, `spread`,
  and `proportional`. `all-gpu` trades extra pipeline hops (per-token decode latency) for using all VRAM
  to avoid a CPU spill and to share prefill compute across cards.
- **Pre-compiled shard cache** — the controller quantizes a model once to `_shards/<quant>/`; loads
  then serve small **pre-packed** int4/int8 layers (skip the bf16 stream + re-quantize). Covers dense,
  fused-3D MoE, per-expert MoE fused at compile (Mixtral/OLMoE), and **non-fused per-expert MoE**
  (MiniMax-M2 — experts stay 2D Linears, int4-packed individually) — bit-identical to a cold load.
  Each cache unit's source tensors are read in **on-disk offset order** so a spinning weights drive
  reads sequentially (readahead) instead of seeking per tensor — large win for many-tiny-tensor MoE
  layers (read dominates compile time: e.g. MiniMax-M2 ~150 s read vs ~7 s pack per layer).
  **fp8/nvfp4-source MoE** compiles too: compressed-tensors quantizes per-expert `Linear`s, so each
  expert is a 2D `weight_packed` dequantized to bf16 by the same path dense fp8/nvfp4 uses, then
  fused-3D or packed per-expert on bf16 (only an exotic fused-3D *quantized* expert is unsupported).
- **Distributed packing** (exo-inspired) — the per-layer pack fans out across the fleet's idle CPUs:
  each worker fetches a layer's bf16, packs it with the *shared* packer (and, for per-expert MoE, fuses
  to 3D against a meta skeleton it rebuilds from the model config), and posts it back. Bit-identical to
  a single-box compile by construction (the same shared fuse + pack code), proven per-layer by a byte
  comparison, with automatic local fallback on any worker failure.
- **Compile-on-first-load** — an int4 load with no shard cache yet BUILDS the cache first (via the
  deprioritized `/compile_shards` subprocess, so the GIL-heavy quantize never starves the event loop /
  drops live-gen logits), then serves the small pre-packed layers — so the FIRST load *persists* the
  cache rather than just re-quantizing in memory and re-doing it next time. One shared
  `engine._precompile_int4` covers BOTH the explicit `/load` (`precompile=1`, default) AND the auto-load
  path (a serving request to a not-yet-resident model); no-op when a cache exists / quant≠int4 / tp>1,
  non-fatal (any failure falls through to the cold on-the-fly load).

## Models
- **Text-to-image serving v1 (#t2i-serve, 2026-07-11): the fleet renders images.** A downloaded
  diffusers checkpoint (Qwen-Image, 20B MMDiT) now loads like any model and serves
  `POST /v1/images/generations` (OpenAI images shape + `negative_prompt` / `steps` / `cfg` / `seed`,
  `b64_json` out, auto-load) plus a Generate panel in the model's dashboard modal with a live
  "rendering step i/n" on its card. v1 places the whole pipeline on ONE GPU worker **co-located
  with the controller** (hostname match; shared filesystem — the model dir is read in place and the
  PNG handed back as a local path): the DiT is quantized at load with the fleet's own RTN int4 g128
  packer in the gate-tested **mixed-edge** recipe (first + last `edge` transformer blocks kept bf16
  ≈ bf16 quality at ~13.5 GB weights; `edge` 2→1 fallback on tight VRAM), the Qwen2.5-VL-7B text
  encoder runs on worker **CPU** bf16 (encode-once per request), and the VAE decodes tiled on GPU
  with an exact CPU fallback on OOM. Requests ride the existing control link (`t2i_gen` →
  per-step `t2i_step` progress → `t2i_done`); placement respects live-free VRAM with LRU eviction;
  text-gen on a t2i model refuses with the images-endpoint hint; the juggler skips t2i models (the
  CPU text encoder is designed placement, not a hybrid to "promote"). Needs `diffusers` in the
  worker venv. Integration fixes that made it real: **split encoder/render pipeline views** (one
  view holding the CPU TE poisons diffusers' `_execution_device` → 'mat1 is on cpu'), tier-string →
  torch device normalization, a one-refresh retry on stale post-update heartbeats, and the
  **fwd-watchdog defers its exit(42) relaunch while a render is active** — a co-resident text
  forward stalling under a render's GPU+CPU saturation is contention, not a poisoned forward
  (observed: the relaunch killed a healthy render at step 9/20). Post-ship hardening from live
  incidents: **unload actually frees the DiT's VRAM** (`T2IPipeline.release_vram` empties GPU
  storages in place — the generic shard release walks attrs a t2i pipeline doesn't have, so ~12 GB
  stayed pinned on ROCm; render-safe: an unload during a live render defers the free to the
  render's end), and **live renders block disruptive lifecycle ops** — `/update` and `/restart`
  refuse while a render is in flight (a forced update mid-render orphaned a finished PNG into a
  broken pipe, observed) and the worker's idle-gated self-update waits for it; `force=1` overrides.
- **Text-to-speech serving (#tts-serve, 2026-07-15): a dedicated TTS engine, Kokoro-82M.** A
  purpose-built speech model (StyleTTS2, Apache-2.0, ~82M params / ~0.3 GB, 54 voices) now serves
  `POST /v1/audio/speech` (OpenAI Speech shape; `voice` passes a Kokoro id through or maps an
  OpenAI name — `nova → af_nova` etc. — to a speaker; `speed`; `wav`/`pcm` out) on ONE
  controller-co-located worker, the same single-node media pattern as t2i (`tts_gen` → per-chunk
  `tts_step` → `tts_done` over the control link; result written to the shared filesystem). Loads
  at ~0.3 GB, auto-loads on a cold speech request, and skips the juggler / int4-int2 compile paths
  (nothing to promote or quantize). This **replaces the Qwen2.5-Omni Talker as the recommended
  speech path** — Omni's Token2Wav output is intrinsically choppy on that checkpoint (reproduced
  under HF-native transformers too), so the Omni checkpoint was retired from the speech role and
  `/v1/audio/speech` now routes a Kokoro model to the KokoroPipeline, falling through to the Omni
  path only when a caller names an Omni model. Two bring-up specifics baked in: (1) **spacy-free** —
  Kokoro's `KPipeline` pulls `misaki.en → spacy → thinc → blis` and blis won't build on py3.13/3.14,
  so the leaf installs `kokoro`/`misaki` `--no-deps`, drives `KModel` directly, phonemizes via
  `misaki.espeak.EspeakFallback` (pip-bundled `espeakng-loader`, no system espeak-ng), and stubs
  `sys.modules['spacy']` so the import chain completes; (2) **GPU→CPU auto-fallback** — on gfx1151
  MIOpen JIT-fails Kokoro's LSTM kernel, so a GPU warmup that raises a HIP/MIOpen compile error
  transparently rebuilds the model on CPU (82M params → ~2× realtime on CPU; ~4× on an NVIDIA GPU).
  `+ Add model` with `hexgrad/Kokoro-82M` downloads it complete (see the `.pth`/`.pt` fix below),
  and the models page badges it **🔊 tts**. Full guide → [docs/TTS.md](docs/TTS.md).
- **Weight-only repos download completely (#tts-serve, 2026-07-15).** `+ Add model` / `/add_model`
  now pull `.pth`/`.pt` files for any repo that ships **no safetensors** (previously such a repo
  grabbed only `config.json`, so a Kokoro-style checkpoint + its `voices/` pack arrived empty). The
  weight-total measurement (`_hf_total_bytes`) falls back to `.pth`/`.pt` the same way, so the size
  and download-% denominator are honest for weight-only and voice-pack repos. (Follow-up `2b86ad7`:
  the models-page **on-disk size** walk got the same `.pth`/`.pt` fallback so the Kokoro row shows a
  real size instead of a blank "on disk", and the one-click **⚡int4 compile badge** now excludes
  `tts`/`t2a` media models — their leaves never read the shard cache — matching the embedding/t2i
  exclusions.)
- **Media-model detail view (#tts-serve, 2026-07-15).** Clicking a media model (tts / t2i / t2a) on
  the models page now shows a media-appropriate Operational block instead of the LLM layout's zeros.
  The worker's `media_info()` (device, params, weight bytes, sample rate, voice list, default voice)
  rides the load reply; `/status` exposes a `media` block (device derived from stage GPU placement,
  weight size from the worker's `loaded_bytes` since a media ModelSpec has dummy dims, last-run RTF)
  and the dashboard `detailLive` branches on it — type / device / parameters / weights / VRAM|RAM /
  sample rate / expandable voice list / default voice / last-synthesis N× realtime / requests / uptime.
- **Diffusers-layout repos are first-class downloads (#t2i, 2026-07-10).** A multi-component
  image-generation checkpoint (`model_index.json` + `transformer/`/`text_encoder/`/`vae/`/`tokenizer/`
  subfolders — Qwen-Image class) now flows through the normal `/add_model` → background pull →
  dashboard progress card → migrate-to-`models/` lifecycle like any flat LLM repo. Completeness is
  diffusers-aware (`_diffusers_complete`: every component subdir with a `config.json` must hold
  weights, sharded sets verified per-prefix against their `-of-N` count *and* their index's
  `weight_map`; conservative — partial pulls never migrate or report ready); the cache→`models/`
  migration walks recursively preserving the component tree (it was top-level-only — subfolders were
  silently dropped) and now also carries `.py`/`.jinja`/`.txt`/`.model` sidecars; the pull and
  progress-total extension sets were widened in lockstep so tokenizer files (`merges.txt`,
  sentencepiece) arrive and the % denominator matches reality. Status badges the model **🖼 t2i**
  (and no longer freezes an empty badge set computed before a model finished downloading), sizes it
  by recursive safetensors sum, `/api/show` reports `capabilities: ["t2i"]`, the dashboard shows
  "pipeline pending" instead of a Load button, and `engine.load` refuses with the real reason
  instead of "unknown model". The **serving pipeline for these checkpoints is a separate, pending
  feature** — this milestone makes acquisition/registry/UI treat them properly.
- **Model aliases shown in the UI**: a registry alias (e.g. `qwen2.5:14b` → `qwen2.5:14b-instruct`,
  via `MODEL_ALIASES`) is now surfaced as an "alias: …" line under the model's primary name — in the
  models list, each loaded-model card, and the model detail modal — so it's obvious which alternate
  names resolve to a given model. (`_aliases_for` reverse-maps `MODEL_ALIASES`; rendered in Ollama
  `family:size` form.)
- **Delete is a complete removal** (`/delete`): it purges the model's on-disk cache (its
  `models/<name>/` incl. the `_shards/<quant>/` pre-quant caches *and* the HF-cache duplicate) AND its
  whole registry footprint — every registered name that resolves to the same repo (so re-registered
  alias names can't dangle on now-missing files), its GGUF mark, and any built-in `MODEL_ALIASES` entry
  pointing at it. The model also **leaves the list entirely** (no stale "download" button): custom
  models drop from `custom_models.json`, and a deleted **built-in** is persisted to a `deleted_models.json`
  hide-set and filtered out after `MODELS` is seeded on startup (re-`/add_model` un-hides it). Delete ==
  forget + purge files + hide; refuses if any of those names is loaded or downloading. (`/forget`
  remains the opposite trade-off: unregister but keep the files.)
- **Mistral3 / Pixtral distributed vision** (validated end-to-end on Devstral, 2026-06-29): the
  controller-side vision encoder handles Pixtral's split tower (`vision_tower` + a separate
  `multi_modal_projector`, both materialized from the checkpoint's RAW key prefixes — Mistral3 stores
  them un-`model.`-wrapped — with the 24B text model left on meta), drives `get_image_features(pixel_values,
  image_sizes)` at the merged patch grid, and splices per-image embeds at the `[IMG]` (id 10) placeholders
  with plain 1D positions. Pixtral's 2D rotary table is rebuilt via the module's own rope-init (the generic
  1D materializer would corrupt it). Two integration fixes were needed: (1) Mistral ships its chat template
  as a standalone `chat_template.jinja` (not inside `tokenizer_config.json`), so the model download now
  pulls `*.jinja` and tops it up for already-present models — without it the tokenizer had no template, the
  prompt fell back to a flat `user:/assistant:` form, and the model degenerated; with it the native
  `<s>[INST][IMG]…[/INST]` renders. (2) the serving path injects the image placeholder for any tokenizer
  whose template emits none. devstral image→text: *"The image contains a red circle and a blue square."*
  Covers Devstral / Ministral. **Row structure (#150):** the `[IMG]` run now carries the trained Pixtral
  layout — `[IMG]×W` per patch row followed by `[IMG_BREAK]`, the last row closed with `[IMG_END]`
  (ids resolved from the tokenizer, verified by round-trip) — instead of a flat run, so the LM sees where
  each patch row ends. The per-image `(rows, cols)` grid is derived from `image_sizes` at the same merged
  cell the processor used; image embeds still splice only into the `[IMG]` slots (break/end keep their own
  embeddings), and any image whose grid doesn't match its token count falls back to the flat run.
- **Gemma 4 unified vision** (#143, validated end-to-end on gemma-4:12b-it, 2026-07-03): the
  encoder-free arch — no vision tower at all; `model.embed_vision` (LN → Dense → +factorized-2D-posemb
  → RMSNorm → Linear) projects raw merged pixel patches straight into LM space. The HF image processor
  hard-requires torchvision (which would clobber the pinned ROCm/CUDA torch), so preprocessing is a
  pure-PIL/torch reimplementation of the exact algorithm: aspect-ratio-preserving resize to a multiple
  of `pool*patch`=48 px (`F.interpolate` bicubic+antialias ≡ `tvF.resize`), 16 px teacher patchify,
  3×3 `patches_merge` into ≤280 model patches of 6912 values, pad with (-1,-1) positions. The raw
  safetensors keys are stored RENAMED (`vision_embedder.*`, un-nested projection) — the loader applies
  transformers' WeightRenaming table during collection. `get_image_features(pixel_values,
  image_position_ids)` returns padding-stripped LM-ready embeds; each template-rendered `<|image|>` is
  bracketed `boi + n×image + eoi` (processor parity) then expanded to its REAL per-image count and
  spliced with plain 1D positions. Multi-image attribution exact; works on all three APIs. Image-span
  bidirectional attention is now honored (see "Gemma 4 bidirectional image-span attention" below; was
  previously causal-first). Side-fix: gemma-4's `chat_template.jinja` was missing from the model
  dir, so even TEXT prompts had been served through the flat fallback — with it in place the native
  `<|turn>` form renders (and `<turn|>`=106 was already a registered stop).
- **Gemma 4 tower vision** (31b-it / 26b-a4b-it, model_type `gemma4`; validated end-to-end int4 on the
  CUDA fleet + bf16 on the ROCm box, 2026-07-05): unlike the 12b unified path above, the tower variants
  carry a REAL `Gemma4VisionModel` ViT (`vision_tower`: patch-embed → 27-layer encoder → 3×3 pooler)
  plus a SEPARATE `embed_vision` projector — the Mistral3 tower+projector shape, but the projector is
  `embed_vision` (not `multi_modal_projector`) and the checkpoint keys need NO rename. Because the 3×3
  pooling happens INSIDE the tower (after the encoder), it consumes UNMERGED 768-d (16×16×3) teacher
  patches — the unified path's pre-merged 6912-d preprocess is NOT reusable — so it drives the real
  `Gemma4ImageProcessor`, whose pure-PIL variant is torchvision-free (runs on the ROCm box too).
  `get_image_features(pixel_values, image_position_ids).pooler_output` is padding-stripped and LM-ready,
  spliced with 1D positions + boi/eoi wrap exactly like the unified path. Rotary subtlety: `gemma4_vision`
  builds a 1D `inv_freq[18]` at **θ=100** with a `head_dim//2` spatial split (not the θ=1e4 default), so
  the meta-tensor materializer now rebuilds it via the module's own `compute_default_rope_parameters`
  when a rotary module exposes one (Qwen's θ=1e4 vision path is byte-identical). Image-span bidirectional
  attention now honored, same as the unified path (see the dedicated entry below).
- **Gemma 4 unified audio** (#144, speech→text): the audio analog of the encoder-free vision path,
  equally torchvision-free and mel-free — each frame of `audio_samples_per_token`=640 **raw** waveform
  samples (40 ms @16 kHz) is one soft token, and `model.embed_audio` (a scale-free RMSNorm → a single
  `Linear` 640→text-hidden) projects them straight into LM space. The HF feature extractor is a trivial
  reshape, reimplemented directly (zero-pad each waveform to a multiple of 640, frame, batch-pad with a
  bool mask); the model is meta-built and only `model.embed_audio` (one tensor) is materialized, then
  `get_audio_features(input_features, input_features_mask)` runs with no downsampling so its output
  aligns 1:1 with the mask. Each `<audio_soft_token>` (258881) run is bracketed `boa`/`eoa` and expanded
  to the real per-clip frame count, spliced with plain 1D positions. Clips beyond `audio_seq_length`
  (750 tokens ≈ 30 s, the model's documented cap) are truncated with a logged warning (never silently).
- **Gemma 4 per-type attention masks** (2026-07-03): the per-type serving path (`layer_types`
  sliding/full + per-type rotary) was handing EVERY layer a single full-causal mask, so
  `sliding_attention` layers attended the whole context instead of only the last `sliding_window`
  (1024) keys — diverging from the reference once a prompt or generation crosses the window
  (single-node and distributed alike; latent below 1024 tokens). Both forward paths (`_forward_impl`
  and `_forward_uniform_eager`, prefill + decode) now build a windowed causal mask for sliding layers
  and the plain causal mask for full layers — validated **bit-exact (0.0)** against the HF
  `Gemma4TextModel` reference across lengths. Also: the head now applies `final_logit_softcapping`
  (±30; monotonic so greedy is unchanged, corrects temperature/top-p sampling parity), and the
  KV-reserve probe sizes each layer from its OWN attention geometry — gemma-4's full-attn layers are
  `global_head_dim`(512)/`num_global_key_value_heads`(1), not the uniform `head_dim`(256)×8, so the
  old probe over-reserved them ~4× and could false-OOM a tight stage into a needless replan. Root-cause
  note: the pipeline SPLIT itself is bit-exact (`num_kv_shared_layers=0`, per-stage rotary indexing by
  global layer index is correct) — a controlled offline harness proved single-node ≡ 2-stage; the
  reported "distributed-only garble" was the sliding-mask error (which also hits single-node past the
  window) compounded by fleet-contention hop-death, not a stage-boundary bug.
- **Gemma 4 bidirectional image-span attention** (2026-07-05): with `use_bidirectional_attention='vision'`
  (the 12b unified text config's default; the tower checkpoints set it too) the reference lets the soft
  tokens of each image attend **bidirectionally within their own block** — the pipeline had shipped
  causal-first, the one remaining vision-quality gap (location precision), flagged twice across prior
  handoffs. Now honored: the controller derives each image's contiguous soft-token run from the mm splice
  positions and rides them down the pipeline in the frame header (`bidir_spans`), exactly like
  `position_ids`, so EVERY stage rebuilds the same mask (TP peers get them via the broadcast tuple).
  `_causal_addmask` OR's a **blockwise overlay** (two positions attend iff they share one image run) onto
  BOTH the full and the sliding-window causal masks — bit-identical to HF's
  `or_masks(base, blockwise_overlay(get_block_sequence_ids_for_mask(mm_token_type_ids)))` (validated by an
  offline parity harness across single/two/edge/all-image layouts × windows {∞,4,1024}, all MATCH).
  Prefill-only (a decoded text token is block −1 → no change), gated on the text config's flag (every
  non-bidir model byte-identical), and chunked prefill is disabled while active so an image never straddles
  a chunk boundary. **Validated end-to-end on om3nbox** (gemma-4-26b-a4b-it int4, 2026-07-05): red-bg /
  white-circle image → "there is a white circle. The background color is red" — 256 image tokens spliced,
  clean stop, zero mask/shard errors.
- **GGUF ingestion**: a model that ships weights only as a llama.cpp **`.gguf`** is normalized to a
  standard safetensors checkpoint ONCE at add/download time (`transformers` GGUF loader dequantizes →
  bf16 → `save_pretrained`), after which it is an ordinary model — chunk-streamed, int4/int8
  shard-cached, and run on the distributed pipeline with no GGUF awareness downstream (same idea as the
  fp8/nvfp4 source path). The heavy `from_pretrained` runs in a **subprocess** (`gguf_convert.py`) so it
  can OOM without taking down the controller box it co-hosts. Add via `/add_model?...&gguf_file=<one
  quant>.gguf` or the dashboard's optional GGUF field. Covers the architectures the GGUF loader supports
  (Llama/Qwen2/Mistral/Gemma/…); single-file quants only (split `NNNNN-of-NNNNN.gguf` is rejected with
  guidance); one quant per repo. Unlocks the large pool of GGUF-only community models.
- **MoE**: fused + non-fused experts; optional intra-layer offload (attention on GPU, routed experts in
  CPU RAM). Loaded + validated across Mixtral, OLMoE, Qwen3-MoE / Qwen3.6-A3B, MiniMax-M2.
- **Multimodal**: distributed vision + audio (Qwen2.5-Omni) — image/audio → text, 3D mRoPE positions.
  Omni's checkpoint declares `architectures=["Qwen2_5OmniModel"]` (a bare `*Model` name), which the
  conservative encoder heuristic mis-read as an embedding model and routed to a single-node `AutoModel`
  build transformers can't construct; composite generative checkpoints
  (thinker/talker/token2wav/text/vision/audio sub-configs) are now excluded from that heuristic, keeping
  Omni on the pipeline Thinker path (re-validated end-to-end — load, text, and vision — under
  transformers 5.12.1). **Speech-out is now the dedicated Kokoro engine (see Text-to-speech, above):**
  Omni's Talker + Token2Wav path works but is intrinsically choppy on that checkpoint (reproduced under
  HF-native transformers 5.12 and 4.52 — checkpoint, not a serving bug), so it was retired from the
  speech role; `/v1/audio/speech` prefers Kokoro and only falls through to Omni when a caller names an
  Omni model.
- Hybrid architectures (Gated-DeltaNet + mRoPE), multimodal text-config models, and a range of dense
  decoders (Qwen2.5/3, Llama, Mistral/Devstral, DeepSeek).

## Multi-model & ops
- **Music (t2a/ACE-Step) serves on ANY capable GPU, not just the co-located box (#media-anywhere,
  2026-07-17).** v1 t2a could only run on a worker sharing the controller's filesystem — it read
  the checkpoint from a local path and handed the WAV back as a local path — so music was locked
  to the one hand-configured box (beast), and an autoload just *errored* even with an idle remote
  GPU (amdcomp, 11 GB free) sitting right there. Decoupled from co-location: (1) workers advertise
  `can_t2a`/`can_t2i` (an import-free `find_spec` probe) in registration, stored on the Node and
  shown in `/status`; (2) placement now considers the co-located GPU **or any remote GPU whose
  worker advertised the acestep runtime**, preferring co-located when it fits (no transfer),
  otherwise the most-free capable box; (3) a remote worker fetches the checkpoint itself via
  `snapshot_download` (the same mechanism the embedding path uses — no shared FS, no controller
  streaming) with a clear error if the registry target isn't a public HF repo; (4) the finished
  WAV returns as **base64 over the control link** (the controller decodes it off the event loop,
  else falls back to the legacy local-path read). Because a `t2a_done` now carries multi-MB audio
  through the line-framed control link, the controller's accept-bridge reader limit is raised to
  128 MB (data-plane readers use `readexactly`, unaffected). **Deploy order matters: controllers
  before workers** — a new worker's base64 `t2a_done` would overrun an old 64 KB-reader controller.
  Adversarially reviewed (protocol/deploy-skew + placement/fetch) before ship. (Kokoro TTS and
  t2i can be mirrored onto the same path next; `can_t2i` is already advertised.)
- **Hitless controller restart — shard adoption (#adopt, 2026-07-16).** A controller-only restart
  no longer reloads models. Workers KEEP their loaded shards when the control link drops (gated on
  the register ack's `adopt: true` capability flag, so mixed code versions degrade to the old
  drop-on-disconnect in both directions) and re-register with a `loaded` inventory: the ORIGINAL
  load message each model was sent with (kind, layer range, ctx, quant, KV flags — the complete
  recipe, retained worker-side since the load) plus live gpu/loaded byte counts. The relaunched
  controller REBUILDS each model's resident state from those recipes — spec/tokenizer/eos re-derive
  from disk, stage plan from the assignments, a fresh stage0 dial — and the inter-worker data plane
  self-heals lazily at prefill (the existing #stage0-stale-reconnect freshening). Coverage is
  strict (every pipeline stage present + contiguous 0→N layers, else no adoption); TP models are
  not adopted; kept shards that never assemble are freed by a 90 s grace sweep so nothing pins
  worker memory invisibly; an auto-load racing a pending adoption waits ~10 s for it first;
  spec-decode drafts (controller-local) are not re-attached — reload to restore. Restart semantics
  split three ways on the Config page: **Restart controller** (`/restart?workers=0`, hitless via
  adoption), **Restart fleet** (`/restart?workers=1&controller=0` — NEW: workers only, the
  controller stays up and link-death invalidation cleans up the dropped models' state), and
  **Restart all** (`/restart?workers=1`, the old full reset). **Live-validated 2026-07-16** on both
  controllers — single-stage, multi-stage (a 14B split beast+amdcomp), embedding, and **t2i-media**
  (qwen-image, which rendered after adoption) — which surfaced one fix (`25bc53f`): a target
  registered under BOTH an alias and its canonical key was adopted under the *alias*, making the row
  unaddressable (`resolve_model_name` maps aliases away, so unload/load by name hit the canonical
  entry and a later load built a doppelgänger); adoption now re-keys under the canonical key.
- **Deploy cadence: 15-minute auto-poll + fleet-wide-immediate forced update (#fleet-update,
  2026-07-16).** The automatic idle self-update poll went 2 min → **15 min** on both controller and
  workers (a background safety net, not the deploy path). The forced **`POST /update`** ("Update +
  deploy") is now the immediate path fleet-wide: alongside the existing unload+free, it pushes a
  `self_update` command to every worker (stage files NOW; restart only on VERSION bump — the same
  rule as the poll), and `/update?workers=1` sends `restart+update` so each worker stages the new
  files *before* its exit(42), relaunching straight onto fresh code instead of waiting out the poll.
- **Renders share the GPU with LLM decode (#gpu-share, 2026-07-16).** A t2i/t2a render used to
  starve a co-resident text model to ~0 tok/s for its whole runtime: both submitted to the ONE
  default CUDA/HIP stream (strict FIFO), so the DiT's minutes-deep kernel backlog sat in front of
  every tiny decode kernel. Renders now run on their OWN side stream — the hardware scheduler
  interleaves the two queues at kernel boundaries, so decode keeps flowing while the render fills
  the gaps (true parallel sharing; no locks, no turn-taking). CPU devices / stream-creation
  failure degrade to the old inline behavior; streams are drained before results are read on CPU.
- **Auto-load can't OOM a render off the box (#render-oom-guard, 2026-07-17).** On om3nbox (Strix
  Halo, unified RAM=VRAM), a client repeatedly auto-requested `qwen2.5-coder:32b` — 32B, ~20 GB of
  weights that *"won't fit GPU → run on CPU, <0.3 tok/s."* Once the resident models idled, the
  planner stopped refusing it and instead auto-fit ctx down onto a CPU-spill placement; mmap-loading
  20 GB of weights beside the resident qwen-image + qwen3-30b + 14b exhausted the unified pool and
  **OOM-killed the co-located worker mid-load**, dropping all four co-resident models — a live image
  render among them → HTTP 500 to the render client (crash-looped 3+ times in ~20 min). Fix: a
  **request-triggered auto-load** (the new `auto=True` path, set only from `_autoload_shared`) whose
  weights would run **mostly on CPU** (`cpu_weight_frac > 0.5`, the SEVERE band) **on the
  controller's own co-located box while other models are resident there** is now refused — a terminal
  `CapacityError` → `503 at_capacity` to that one request — instead of proceeding into the
  over-commit. The load never reaches the mmap step (the gate sits after the plan/assess converge,
  before dispatch). Scoped so it never touches an explicit `/load` (`auto=False`), an idle box (no
  residents to endanger), a non-co-located capacity node (e.g. the dell CPU worker), or a placement
  that fits GPU (low `cpu_weight_frac`); media loads take a separate path entirely and are never
  gated. The useless <0.3 tok/s auto-placement it blocks was never worth serving anyway; load such a
  model explicitly to force it. (Adversarially reviewed end-to-end before ship.)
- **ACE-Step defaults to RAM-offload — 0 resident VRAM (#t2a-offload-default, 2026-07-20).** A t2a
  load previously defaulted to **GPU-resident** bf16 (holding ~10 GB VRAM) and only offloaded once the
  card was already full (#t2a-offload-fallback below); on a card with room, music sat on precious VRAM
  the LLMs wanted. Offload is now the **standing default on every controller**: components stay
  resident in RAM and the ~6.6 GB DiT hops to the GPU only for each render — **0 GB resident VRAM**,
  ~8 GB transient during a render, cannot OOM a card on load, never evicts a co-resident model, and
  renders still run on the GPU at full speed. GPU-resident became opt-*out* via a new persisted knob,
  `POST /config?t2a_offload_default=0`. Controller-only (the worker already honoured `t2a_offload` in
  the load message), so it deployed hitlessly to both controllers.
- **CPU-only t2a: plumbed, measured, documented as non-viable (#t2a-cpu, 2026-07-20).** Added opt-in
  `POST /load?model=<ace-step>&cpu_only=1` — the whole pipeline in RAM with `device='cpu'`, budgeted
  against RAM only, no GPU/offload fallback, basis `t2a: single-node (CPU)` — plus the worker-side
  override that makes it real: ACE-Step has **no cpu-force flag** (its `__init__` grabs `cuda:0`
  whenever a GPU is visible and `load_checkpoint()` moves every component with `.to(self.device)`), so
  `worker_t2a.py` now overrides `pipe.device` **before** `load_checkpoint()`. Without that, a
  `cpu_only` load silently loaded onto the GPU and OOM'd a full card. Result: it **loads** cleanly on
  CPU (~4 s, 0 GB GPU) but **does not render** — on a 32-core box a short clip sat at ~8–14 % CPU with
  **zero** diffusion steps for 10+ minutes. Kept opt-in and documented as a diagnostic curiosity, not
  a serving mode. Also documents the trap that **`cpu_frac=1.0` does not mean "running on the CPU"**:
  offload and cpu_only report identical `vram_used=0 / cpu_frac=1.0` (the stat describes where the
  *weights* sit, not where the *compute* runs) — only the basis string or the render time separates
  them. `docs/T2A.md` additionally now carries the full rationale for **why t2a is CUDA-only and not
  implemented on ROCm** (no `torchaudio` matching the ROCm torch ABI, MIOpen-JIT-unreliable diffusion
  kernels on gfx1151, and no CPU fallback to retreat to), cross-linked from `docs/ROCM.md`.
- **t2a/music auto-falls-back to RAM offload instead of failing (#t2a-offload-fallback,
  2026-07-17).** An ACE-Step music auto-load (`/v1/audio/music`) that couldn't get its ~10 GB of
  GPU-resident VRAM — because the co-located card was full of **un-evictable** residents (a pinned
  model + a busy one), leaving only the idle vision/TTS models to evict — used to **raise** (and the
  route surfaced it as a misleading 404, so the pipeline "shipped without music"). ACE-Step already
  has an **offload** recipe (components in RAM, the ~6.6 GB DiT hopped to the GPU per render: ~8 GB
  transient VRAM + RAM for the weights, and it **never evicts**) — it was just opt-in (`t2i_offload=1`)
  with no automatic fallback. Now, when the bf16 GPU-resident placement can't fit and nothing is
  evictable, the loader **flips to offload and retries** rather than failing. Resident stays the fast
  default (it still evicts idle LRU to try resident first); offload is the last resort, and since it
  is M1's proven serving mode it always beats a hard failure. (The symmetric gap in the t2i image
  loader is known and left as-is — qwen-image is deliberately served on the om3nbox pool.)
- **Per-node restart with in-use recovery (#node-restart, 2026-07-16).** Every node row on the
  models page gains an ↻ — `POST /restart_node?node=<hostname|id>` restarts JUST that worker
  process (exit 42 → supervisor relaunch), the per-node fresh start that clears whatever VRAM/RAM
  it holds without touching the controller or the rest of the fleet. Models with a stage on the
  node drop, then split by usage: **in use** (serving/queued, or used in the last 10 min) →
  auto-RECOVERED — once the invalidation lands, a background task re-loads them with their
  original ctx/quant/KV knobs and the planner re-places onto whatever capacity is up (other
  nodes' GPU/CPU; the restarted node usually rejoins in seconds and competes again); **idle** →
  the invalidation frees their surviving stages on the other nodes too, so they cost nothing
  anywhere and re-auto-load on demand. The response + toast list both sets.
- N models resident at once, per-node sharing, concurrency + queueing, auto-load/unload, same-model
  replication + data-parallel routing.
- **Silent-wedge hardening (the beast kernel-panic postmortem, 2026-07-10).** A poisoned 30h-old
  worker process turned every distributed vision prefill into a silent 240s gen-stall reclaim —
  37 wedges in 5.5h, each client retry re-wedging, and the accumulated pathological load fed a
  host kernel panic (netconsole-captured NULL-deref; no GPU Xid — nvidia/UVM software state, not
  hardware). The worker's stage exception ("F.embedding got CUDABFloat16" — the mm companion frame
  consumed as forward input) never reached the controller: the data-plane error frame rides the
  one-way stage chain a stale hop can eat. Four fixes so one sick worker can never again become an
  hours-long wedge storm: (1) **#stage-error-ctrl** — every stage COMPUTE exception is also
  mirrored over the heartbeat-kept control link; the controller fails the request's future
  immediately (fast causal 500 instead of a blind stall) and logs every arrival, matched or not;
  (2) **#mm-pairing** — the prefill ids frame declares its multimodal companion (`hdr["mm"]`):
  declared-but-missing fails loud (never run a vision prefill unspliced), undeclared never claims
  (a leaked companion + controller-restart req_id collision can't splice stale image embeds into an
  unrelated prompt), and staged companions expire after 10 min; (3) **#stage0-dtype-guard** — a
  first-stage prefill frame carrying floating-point data is classified at the door as a mispaired
  mm/ids or misrouted hidden frame; (4) **#wedge-quarantine** — `wedge_reload_n` (default 3,
  `/config`-tunable, 0=off) gen-stall reclaims of the same model within 15 min trigger an automatic
  fresh re-place (reconfigure: new shards + new data conns, rollback-safe, serialized with the
  juggler) — the demonstrated cure for poisoned pipeline state. Ops note: worker files apply on
  fetch but worker PROCESSES only pick them up on restart — deploys that change worker code need
  `POST /restart?workers=1` after the `/update` (a periodic fleet-wide worker restart is also the
  cheap hygiene against long-lived-process state poisoning).
- Robust loads: survive a worker drop mid-load (replan on survivors), free partial shards on failure,
  scaled timeouts, gentler restarts; auto-recover resident models when a worker reconnects.
- **Stale-KV self-heal + crash-proof attention:** a worker stage's causal mask is sized for
  `cache_start + q`, but SDPA takes the real kv-dim from the cache (`past + q`); they desync — and
  crash with "expanded size N must match M" — when a generation reclaimed by the gen-stall watchdog
  (or a disconnecting client) leaves an UNCANCELLABLE forward running in a thread that keeps mutating
  the shared cache concurrently with a fresh forward. Two layers make this impossible: (1) forwards on
  a shard are SERIALIZED (a non-blocking per-shard guard — a racing new forward fails fast and the
  controller re-prefills, rather than concurrently corrupting the cache; lazily initialized so
  cache-served shards are covered); (2) a new sequence (`cache_start == 0`) unconditionally rebuilds
  the cache. A reclaimed generation can no longer corrupt the next one. (A defensive per-decode KV
  length "reconcile" was tried and reverted — `DynamicCache.get_seq_length()` inspects layer 0, which a
  mid/tail pipeline stage doesn't own, so it false-tripped on every multi-stage decode.)
- **Per-load KV-cache placement + per-model default temperature.** `/load?kv_offload=1` (or the
  Load dialog's "KV cache: System RAM" option) rests the KV cache in system RAM — transformers 5.x
  `DynamicCache(offloading=True)`, per-layer side-stream prefetch — so the VRAM the full-ctx KV
  would reserve goes to model LAYERS instead (long context on small cards, at a decode-speed
  cost). The worker stops reserving per-layer KV against VRAM, probes the reservation against RAM,
  and reports `gpu_kv_bytes=0` so the multi-model coexistence reserve stays honest; cudagraph
  decode is gated off; mutually exclusive with `kv_quant`. CUDA-only: on ROCm/HIP the offloaded
  prefetch was live-validated GARBLING decode (nondeterministic at temperature 0 — a side-stream
  race in the TheRock stack) and an APU's "VRAM" is unified RAM anyway, so HIP falls back loudly
  to the plain on-device cache. `/load?temperature=0.7` stores a per-model DEFAULT sampling
  temperature (0-2), used only when a request sends none — explicit request values, including an
  explicit 0, always win; applied across the Ollama/OpenAI/Anthropic serve paths and badged on
  the model card. **min-p sampling** rides the same paths: applied after temperature and before
  top-p, it keeps only tokens with `p >= min_p * p_max` and renormalizes — per-request (Ollama
  `options.min_p`, OpenAI/Anthropic top-level `min_p`) or as a per-model default via
  `/load?min_p=` (0-1, badged `mp=` on the card). Both defaults are runtime-mutable on a LOADED
  model with **`POST /model_config?model=...&temperature=...&min_p=...`** (absent = keep, empty
  string = clear, applies to all replicas), surfaced as a "Runtime settings" panel in the
  model-detail modal — no reload needed to tune a resident model's sampling.
- **The full sampling-knob family (#runtime-knobs).** `top_k` (post-min-p top-k filter),
  `repeat_penalty` + `repeat_last_n` (llama.cpp multiplicative penalty over the last-N window of
  prompt+output; -1 = whole context), `presence_penalty` / `frequency_penalty` (OpenAI additive,
  output-only), and `seed` (reproducible sampling via a fresh per-request `torch.Generator` —
  concurrency-safe, never touches the global RNG; negative = the llama.cpp/Ollama "random"
  sentinel = unset) — per-request on all three APIs (Ollama `options.*` / OpenAI+Anthropic
  top-level; `repetition_penalty` accepted as the vLLM/HF alias in either location). Penalties
  apply to the logits pre-argmax, so they steer greedy decode too; the speculative path is
  greedy-only and ignores them by design. Every knob — plus `top_p` and a default `num_predict`
  for requests that send no length cap — is also a runtime-mutable per-model default on
  `POST /model_config`, stored in one `sampling_defaults` dict, reported on `/status`, and
  editable in the dashboard's Runtime settings panel (10 fields with suggested-value dropdowns;
  empty = unset; Apply sends the whole panel state). All knob values are coerced at PARSE time so
  a malformed value fails as a clean pre-stream 400 — never a post-stream empty-200 (the
  cold-contract rule); the stored seed is capped at 2^53-1 so it round-trips JSON/JS float64
  losslessly (per-request seeds go to int64 max).
- **Connections panel (#connections).** The dashboard's models page gains a bottom section
  listing every connected client (by IP): connected-for, idle-for (an active stream is never
  "idle" — activity is stamped per chunk), REAL bytes in/out counted at the ASGI layer (streamed
  responses grow the counter live; worker `/weights` slice-pulls are excluded), token totals
  in/out, request count, what the client is using or loading RIGHT NOW (in-flight join + the
  load card's `requested_by`), and a **Terminate** button — `POST /terminate?ip=` cancels every
  in-flight request from that client. Browser tabs that only watch the dashboard are chipped
  "dashboard"; a row is a real API client only once it hits a generation/embedding endpoint.
  X-Forwarded-For is charset-validated on BOTH derivation paths before it becomes a client key
  (it renders in HTML and an onclick — arbitrary header text would be an XSS vector).
- **Idle unload (#idle-unload).** New engine setting (`/config?idle_unload_m=`, dashboard
  "Idle unload"): a model that served NO requests for N minutes is unloaded automatically.
  Default 0 = the long-standing behavior — every model stays loaded forever. Judged GROUP-wise
  across data-parallel replicas (unload(base) cascades, and the base carries last_used while the
  routed replica carries active/last_token_ts — judging one key alone could reap a group whose
  sibling is mid-decode); models with an active or queued request, a held per-model lock
  (embeddings), or either lifecycle pin (persist_models / no_unload_models) are never idle-unloaded, and the speech thinker
  stamps per-step progress so long TTS runs aren't reaped. Replaces the old hidden coupling
  where the LRU auto-unload checkbox also enabled a hardcoded 60-min idle unload. Ollama
  `/api/ps` `expires_at` is now honest: last activity + the idle window when the knob is on.
  The knob is clamped to a finite [0, ~1 year] (an `inf` would persist and 500 /status +
  /api/ps).
- **Honest RAM/CPU weight split (#real-stats).** `ram_used_gb` / `cpu_frac` (and the load-time
  "X% of weights on CPU" warning) were `spec_estimate − measured_gpu_bytes`; the spec's formulaic
  int4 estimate overshoots real packed MoE size ~10%, fabricating a phantom "1.9 GB RAM / 10.6%
  CPU" on a fully-GPU-resident Qwen3-30B-A3B (verified per-tensor: everything on cuda). The
  worker has always reported its MEASURED post-quant weight bytes in the load result — the stage
  now carries it and both numbers are computed measured-vs-measured (spec fallback only for
  workers that predate the field). The model-detail placement row also now reads
  "on GPU x of <node total> VRAM" instead of a bare "GPU x GB" that looked like a device spec. triton's `Autotuner.run()` keeps the
  call's args in unsynchronized instance state (`self.nargs`, set on entry / `None` on exit) and the
  int4 w4a16 kernels (dense GEMV + fused MoE) are process-wide singletons shared by every shard — so
  with TWO models resident, any decode that autotune-benchmarks a NEW shape key while the other model
  decodes crashed (`TypeError: 'NoneType' object is not a mapping` in `autotuner._bench`),
  deterministically. Fixed three ways: (1) `Autotuner.run` is serialized behind one process-wide RLock
  (a lock acquire per launch — negligible vs ms-scale decodes; during a bench window other int4
  launches briefly wait instead of crashing); (2) the lazy kernel **builders** are built under a lock
  with their tried-flag set only AFTER the op is final (a racing shard-install could previously capture
  a permanent naive 5-20x-slower fallback mid-build); (3) the expert tensor-subclass is single-built
  under the same lock.
- **Wedged-gen auto-recovery:** a distributed generation whose mid-pipeline hop dies never gets an
  error frame upstream (the data chain is one-way), so it used to sit ACTIVE at 0 tok/s until the
  600s timeout and needed a manual client restart. Two fixes: the gen-stall watchdog now (a) cancels
  the REAL streaming body-pump task (the cancel handle had been the route task, which returns
  immediately for a streaming response → the cancel was a no-op), and (b) fails the model's leaked
  controller-side pending futures so the orphaned `_send` returns at once. The model reclaims its slot
  and unblocks the queue on its own. (Hardened after an adversarial audit: the freed orphan's `finally`
  decrement is floored at 0 so it can't drive `active` negative after the watchdog zeroed it.) Verified
  live: `/cancel` aborts a streaming gen at 6/400 tokens and the slot frees + the model serves again
  immediately (the same handle the watchdog uses). Recovery is **replica-precise**: a parallel
  `pending_friendly` map keys each in-flight request by the UNIQUE replica it was routed to (not the
  `target_id` every replica of a base SHARES), so both the watchdog AND `invalidate_model` (a node
  leaving mid-pipeline) fail ONLY the dead replica's leaked futures and never a healthy sibling's — so
  a data-parallel model's stalled request now gets the same fast future-fail as a single-copy one
  instead of hanging out the ~600s timeout (this supersedes the earlier replicated-SKIP).
- **Concurrent-load isolation:** each control link keys its in-flight load/unload futures by model_id
  (a dict) instead of one shared future, and the worker echoes model_id in its ready/error reply. Two
  models loading onto stages of the SAME node concurrently no longer cross-resolve each other's load
  (which mis-counted VRAM / reported "ready" early and hung the loser to its multi-minute timeout); a
  sole-pending fallback keeps an old worker build working through a rolling deploy.
- **Early architecture guard:** an exotic/unsupported model now fails at load-plan time with a clean
  "unsupported architecture 'X'" instead of a cryptic meta-tensor crash deep in the streamed worker
  build. The controller checks the config RESOLVES via `AutoConfig` (a registered model_type) — it does
  NOT attempt a full model build, so natively-registered archs the worker hand-builds via a special path
  (e.g. Qwen2.5-Omni) still pass; trust_remote_code models (auto_map) pass through too (the worker fetches
  their .py via `/modelcode`), so no known-good model is rejected.
- **Complete Ollama + OpenAI API surface:** alongside the Ollama routes (`/api/tags`, `/api/chat`,
  `/api/generate`, `/api/show`, `/api/ps`, `/api/embed`+`/api/embeddings`, `/api/version`, `/api/pull`,
  `/api/delete`) and the Anthropic Messages API (`/v1/messages` — the Claude Code backend), the OpenAI
  surface is now complete: `/v1/chat/completions`, **`/v1/completions`** (legacy text completion —
  `text_completion` objects, SSE + `[DONE]`, prompt string-or-array), `/v1/models` + **`/v1/models/{id}`**
  (retrieve), `/v1/embeddings`, `/v1/audio/speech`. An unknown model returns **HTTP 404** with the
  dialect-correct shape — Ollama `{"error":"model 'X' not found"}`, OpenAI
  `{"error":{message,type,code:"model_not_found"}}` (was a bare 400) — and OpenAI endpoints default
  `stream` to FALSE when omitted (single JSON, per the OpenAI spec) while Ollama keeps its
  stream-when-omitted default.
- **Fast dead-hop recovery:** a mid-pipeline hop dying *during* a generation used to leave the request
  blocked until the gen-stall watchdog (~240s) or GEN_TIMEOUT (~600s) reclaimed it — the one-way data
  chain delivers no upstream error frame. Now, when a worker's forward to its next hop fails even after
  its own reconnect-retry (a genuine transport death — gated strictly on connection-type exceptions so a
  stage *compute* error never trips it), the worker pushes an unsolicited `hop_error` control frame up
  its (separate) control link; the controller fails ONLY that request's pending future at once
  (idempotent, replica-precise by `req_id`), reclaiming the slot in well under a second. The send reuses
  the control writer's lock+framing so it can't corrupt a heartbeat, doesn't double-decrement `active`
  (the resumed generate() does that), and resets `last_token_ts` so the watchdog doesn't double-act —
  falling back to the watchdog only if the control link is mid-reconnect.
- **Active-decode stall reclaim:** `hop_error` can't catch a *buffered-write deadlock* — a downstream
  hop dies, the upstream's small forward write buffers "successfully" (no error raised), and the one-way
  pipeline then deadlocks so the upstream never writes again to surface the failure. A **second, shorter
  watchdog threshold** (`gen_stall_decode_s`, default 60s) now covers it: it applies ONLY once a
  generation has produced its first token (it's *decoding*, tracked via `gen_started_ts` vs
  `last_token_ts`), so a streaming gen that goes silent is reclaimed in ~60s instead of ~240s. Cold
  prefill keeps the conservative 240s `gen_stall_s` (a slow big-model first-token wait is never
  false-killed — 60s is far longer than any healthy per-token decode, even heavy CPU spill). Both
  thresholds are `/config`-tunable. (Found by an isolated fault-injection test: a worker *crash*
  recovers in ~2.6s via the control-link drop, but a pure *data-plane* partition needed this.)
- **Idle-pipeline self-heal:** every data-plane hop is fresh-reconnected at each generation's prefill
  if it has been idle (an idle TCP socket can go silently half-open — the write succeeds but the bytes
  never arrive — which otherwise stalls the first request after an idle gap until the generation
  timeout). Both the controller's connection to stage 0 and each worker's next-hop are freshened, so a
  model that sat loaded-but-unused replies immediately instead of appearing wedged.
- Observability: placement preview, per-load progress/ETA, **live download speed + ETA** (a rolling
  ~30s byte-rate over the HF-cache pull → remaining/rate, surfaced per model card while a pull runs),
  fleet CPU/GPU/RAM + throughput + bandwidth, curl-able fleet logs; idle-gated multi-file self-update. **Per-model context history** — the model
  detail popup's "tokens in/out" rows are click-through to a scrollable view of the ACTUAL prompts
  sent and text generated (`GET /history`); captured as token ids (decoded lazily, off the hot path),
  kept to the most-recent N requests, and cleared when the model unloads. A managed reload (reconfigure to/from
  tensor-parallel) shows live layer progress on its own card (folded in from the in-flight load) rather
  than a progress-less "re-streaming weights" placeholder beside a duplicate load card.
- **TP mesh keepalive:** the tensor-parallel all-reduce mesh used to work for one generation then
  stall ("peer rank stalled or closed") after a short idle gap between requests — an idle mesh socket
  going silently half-open. Rank 0 now pings the peers (a tiny round-trip that keeps both directions
  warm) whenever the mesh has been idle a few seconds, so TP stays alive across idle periods instead
  of needing a reload after the first request.
- **One-click int4 precache from the models list:** every on-disk model without an int4 shard cache
  shows a clickable `⚡ int4` chip (on registered AND loaded rows) — hover for what the compile costs
  (estimated cache size on disk, source dtype, controller free-disk check) and what it buys (int4
  loads then serve from cache instantly); click fires the same `/compile_shards` as the detail
  modal's Precache button. Compiling rows show live progress (`done/total · elapsed · ETA` + bar)
  instead of a static "compiling…". Embedding encoders never show the chip (their serve path is a
  whole-model float32 load that doesn't read shard caches), and uncached models no longer display a
  misleading "cache ready".
- **Endpoint weather — contention is survivable and honestly retryable.** Under GPU contention,
  healthy-but-slow prefills used to be reclaimed by the gen-stall watchdog at the threshold (~4 min),
  and every client retry re-entered the same slow prefill and died again — a fan-out harness measured
  a 21% run-abort rate, all from this class. Three-part fix: **(1) prefill-progress liveness** —
  workers already stamp per-layer forward progress for their local watchdog; that signal now rides
  the existing heartbeat (`fwd_progress`, request-id-attributed so an orphaned forward can never
  shield a live generation), and the controller watchdog's PREFILL branch treats advancing progress
  as liveness. True wedges (no layer completed for `gen_stall_s`) still reclaim on the old schedule;
  decode stall detection is unchanged (tokens only). **(2) Adaptive prefill wait** — the controller's
  per-frame generation timeout no longer hard-kills a prefill at 600 s: the wait extends in slices
  while worker progress advances (absolute 1 h ceiling as the backstop). **(3) Retryable errors** —
  contention-class failures (watchdog reclaim, dropped data-plane sockets, hop timeouts, a shard
  held by an orphaned forward, node-drop recovery races) now return `503 + Retry-After`
  (Ollama/OpenAI) or `529 overloaded_error` (Anthropic) instead of bare 500s, and a watchdog-reclaimed
  in-flight request gets a clean retryable response instead of an aborted socket. User-initiated
  `/cancel` and `/terminate` keep their kill semantics (never invite a retry). **Streaming paths**
  get the same honesty: a mid-stream reclaim or contention failure emits a typed TERMINAL error
  frame in each protocol's idiom — Ollama `done_reason:"error"` + `retryable:true`, OpenAI a
  `{"error":…}` object (no longer a clean `finish_reason:"stop"` that presented a truncated answer
  as complete — the worst of the pre-fix cases), Anthropic an `overloaded_error` event — rather than
  a silently truncated stream; a genuine client disconnect or user cancel still drops the connection.
- **#at-capacity + #autoload-herd — a cold model at a full cap answers honestly, and concurrent
  auto-loads share one load (2026-07-10).** With `auto_load` on but `max_loaded` reached and
  `auto_unload` OFF, every request for a cold model failed at the resident-cap check — and serving
  mapped that to `503 + Retry-After: 3`, a promise no retry could ever keep (a probe measured an
  honest client retrying 25× over 90 s, forever-looping). Capacity failures are now a typed
  `CapacityError` whose `terminal` flag distinguishes the two shapes at all three raise sites
  (resident cap, no-room, won't-fit-even-at-minimum-ctx): *retryable* = eviction is possible but
  blocked right now (residents busy serving) → unchanged `503 + Retry-After: 3`; *terminal* =
  no automatic recovery exists (auto-unload off, or every resident `no_unload`-pinned) → `503`
  with `code`/`state` **`at_capacity`** and **no Retry-After**, on every surface (OpenAI, Ollama,
  Anthropic, embeddings). The auto-load's one-shot bf16 fallback is skipped on a CapacityError
  (bf16 is strictly bigger — it can only fail the same way). Two adjacent fixes in the same pass:
  the **Anthropic path returned `404 not_found_error` for any load failure** (a capacity problem
  looked like a nonexistent model to Claude Code) — now typed 503s per terminality, embedding-
  misuse a 400; and **concurrent requests for the same cold model now await ONE shared load task**
  (`#autoload-herd`) — previously each duplicate queued behind the engine lock and, on acquiring
  it, found the model resident and *reloaded* it (serial unload+reload churn that could kill the
  first request's generation; measured fixed: 3 concurrent cold requests → exactly one load, all
  three served). `/api/ps` alias-echo rows (a loaded model re-listed under its alias names) now
  carry `alias_of: <canonical>` so clients counting real instances can filter them — the admission
  cap never counted echoes.
- **Idle-unload accepts `-1` as "keep forever":** the Ollama-style sentinel round-trips (saves and
  displays as -1) instead of silently resetting to 0; -1 and 0 mean the same thing — the reaper is
  off and `/api/ps` reports effectively-never expiry.
- **Lifecycle pins + the juggler (hitless VRAM promotion).** Two independent per-model pins on the
  model-detail modal plus one global control. **Autoload on restart** (`persist_models`, previously
  API-only, now a checkbox) re-streams a model to its workers on controller startup so a resident
  model survives a restart/redeploy. **Do not auto-unload** (`no_unload_models`, `/config?no_unload=`)
  is an absolute veto: the model is never reclaimed by idle-unload *or* by LRU eviction — a new load
  that can't otherwise fit FAILS rather than displacing it (distinct from persist, which survives a
  restart but stays evictable under memory pressure). **Juggler** (`/config?juggler=`, off by default)
  turns a model that auto-loaded HYBRID (weights split GPU+RAM under memory pressure) back into a
  full-GPU model once room frees: on a ~60 s sweep (and right after an idle-unload frees VRAM) it picks
  the hottest resident hybrid *that a VRAM-first planner dry-run says now fully fits on GPU* — skipping
  embeddings and any hybrid too big to fit, so a bigger hot one never blocks a smaller promotable one —
  then — only if that model is momentarily IDLE (a busy/backlogged one is skipped, not stalled: engaging
  the barrier and draining it could hold a slow model's clients for minutes; a later sweep catches it at
  a gap) — does a **hitless** swap: a per-model barrier (checked at the top of `generate()` before the
  request takes a queue slot, so it's race-tight) holds new requests while `reconfigure` re-places it
  VRAM-first (atomic, with rollback), then releases — so the client's open connection just pauses across
  the ~10-20 s re-place, no reconnect. The juggler is exempt
  from the do-not-auto-unload veto BY DESIGN: it may promote a pinned hybrid too, because a promotion
  is a reload-into-a-better-placement, not a removal — and it restores the model if a rare
  double-failure ever evicts it, so the pin's "always resident" contract still holds. **Autostart
  delay** (`autostart_delay_s`, default 60 s) makes the startup reload of persisted models wait at
  least that long — on top of the fleet-settle wait — so API clients reconnect before the controller
  gets busy streaming weights.
- **#juggler-live-free — the juggler now measures free VRAM the way the load planner does
  (2026-07-11).** Its promotion fit-check and its anti-churn guard had budgeted against
  `usable_vram_gb` (= `vram_total − a static reserve`) — a per-node *capacity* ceiling that ignores
  resident models and never moves when VRAM frees. Two failures fell out of that: the fit-check
  thought a node was free when a co-resident occupied it, so it fired a disruptive re-place that
  could only land hybrid again; and the anti-churn record ("won't retry until the fleet frees more
  VRAM") compared two copies of that static number, so once a model latched it *never* retried —
  a hybrid model stayed split GPU+RAM forever even after a co-resident idle-unloaded and freed the
  whole GPU (observed: a 4B model pinned 22%-on-CPU on a 12 GB card while a 16 GB card sat empty).
  Fix: one shared `_node_live_free_vram_gb` helper — heartbeat `vram_total − vram_used` + the
  worker's reusable allocator pool + the model's own reclaimable bytes − other in-flight
  reservations — is now the single basis for the load planner's weights budget, the juggler's
  fit-check, and the anti-churn measure. The guard now actually clears when VRAM frees, so a freed
  GPU triggers relocation on the next sweep.

- **#load-faster — one-click "load faster" placement upgrade (2026-07-19).** Under each loaded text
  model the dashboard shows a **⬆ load faster** badge whenever a strictly faster placement is
  achievable with currently-free fleet VRAM — a CPU-spilled/hybrid model that would now fit
  VRAM-resident, or a multi-node pipeline split that would now consolidate onto fewer nodes. Hover
  shows *from → to*; one click (no confirm) **drains the in-flight reply** (waits ~2 min for the
  current generation to finish, then forces), then re-places the model VRAM-first through the same
  hitless #juggler barrier (parked clients pause and ride onto the fresh copy — no reconnect),
  preserving full config (ctx/quant/tp/kv_quant/kv_offload/sampling defaults) and rolling back to a
  working copy if the faster layout no longer fits. Detection (`engine._upgrade_for`, throttled ~30 s
  off `/status`) reuses the juggler's live-free-VRAM fit-check, so the badge never promises a placement
  a real re-place can't reach. Complements the auto-juggler (which only promotes *idle* hybrids): the
  badge's value is **busy** models (drain-then-swap) and **node consolidation**. Kept self-contained
  (its own atomic reload + rollback) so it can't destabilize the auto-juggler / wedge self-heal.
  `POST /load_faster?model=<name>`.
- **Serving & placement correctness (2026-07-18/19).** Three fixes, deployed hitlessly to both
  controllers: (1) **Qwen3 `enable_thinking` is now reachable on the OpenAI/Ollama endpoints** — it
  was silently ignored there (only the Anthropic `/v1/messages` path mapped it, and only the
  `/no_think` prompt soft-switch worked); `serving.py` now honors `chat_template_kwargs.enable_thinking`
  (vLLM), Ollama `think`, or top-level `enable_thinking`, threaded into the template only when it
  supports the switch (default unchanged). (2) **Cache-on-first-load precompile now skips embedding
  models** — `_precompile_int4` excluded media checkpoints but not encoders, so every auto-load /
  re-adopt of a persist embedding (nomic-embed) self-POSTed an int4 `/compile_shards` that always
  failed `KeyError 'model.embed_tokens.weight'`, spamming recurring 400s; it now classifies the
  on-disk config and skips. (3) **A load never over-reserves context** — an explicit or `autoload_ctx`
  value is clamped **down** to the model's training context (`max_position_embeddings`), so a
  2k-trained model (e.g. nomic) never gets a 4k KV window; a smaller request is honored unchanged.

## Public release
- Central `config.json` (all hosts/ports + the self-update source; no addresses baked into code);
  credentials and internal-only artifacts scrubbed for open source.
- **Self-update pulls from the public GitHub repo's raw endpoint** (`update_repo`/`update_branch`) — no
  token of any kind, on the controller or any worker. `provision_worker.sh` clones from public GitHub.
- **Deploy guardrails:** each self-update file fetch is bounded-retried with backoff, so a freshly-added
  module that hasn't propagated on the raw CDN yet gets time to sync instead of aborting the whole cycle
  and leaving the fleet under-deployed (the apply stays atomic — all files or none). The auto-RESTART is
  gated on a **VERSION bump** in the primary file: a same-VERSION doc/comment commit stages to disk
  WITHOUT bouncing the fleet, so a casual push no longer reboots the cluster. The forced dashboard
  "Update + restart" always restarts regardless. (Stale "GitLab" self-update wording corrected to GitHub
  throughout.)
- **Deploy verification (`GET /code_manifest`):** the raw-CDN edge lags a push *per controller*, so a
  forced `/update` can pull a stale file on one box while the CDN looks fresh from elsewhere. This route
  reports the on-disk `sha1(12)`/size/mtime of every self-update file plus the running `VERSION`/`CODE_DATE`
  (and `?grep=<marker>` reports per-file whether a marker is present on disk) — so a deploy verifies the
  bytes actually landed with one HTTP call instead of SSH-ing in to grep.
- **Multimodal backend self-heal:** transformers memoizes its PIL/soundfile/torchvision availability at
  import, so a dep `pip install`ed *after* the controller started stayed invisible (vision kept
  ImportError-ing) until a full restart — the trap the `.38` Proxmox rebuild hit (venv had torch, not
  Pillow). The controller now re-probes and busts that cache at startup, lazily per image/audio request
  (throttled), and on demand via `POST /refresh_backends` — a freshly-installed backend goes live with no
  restart.

## Code organization (context-economy refactor)
- The controller and worker grew into multi-thousand-line files that were costly to read/edit. They are
  being split along seams the *callers* don't see — **zero public-API change** — so each subsystem fits a
  reader's (and an editor's) context window. The multi-file self-update built earlier already keeps any
  number of sibling modules in lock-step across the fleet (each in `EXTRA_UPDATE_FILES`, each imported
  through a pull-once **convergence bridge** so an old checkout self-heals on the deploy that introduces it).
- **Engine split (m4c152):** `server.py`'s `Engine` class (~2.5k lines, 47 methods) was relocated
  *verbatim* into three mixin modules — `engine_load.py` (load/placement/TP/reconfigure), `engine_gen.py`
  (prefill/decode/spec/MTP), `engine_lifecycle.py` (data-plane/recovery/replicas/unload) — recomposed as
  `class Engine(EngineLoadMixin, EngineGenMixin, EngineLifecycleMixin)`. Method bodies are byte-identical;
  only `__init__` and `generate_speech` (which rebinds the `ENCODING` idle-gate global) stay on the shell.
  A new `state.py` registry publishes the controller's namespace and injects it into the relocated modules
  at startup (`state.publish`/`state.bind`), so the moved bodies resolve their former module globals
  without a circular `import server`. server.py dropped from ~9090 to ~6790 lines.
- **Route split (m4c153):** `build_app`'s 73 HTTP routes (~2.5k lines, all defined inline) — 57 of them
  relocated *verbatim* into four `register_*(app)` modules: `routes_dashboard.py` (dashboard/status/
  graphs/plan/logs/config), `routes_lifecycle.py` (load/unload/compile/reconfigure/restart/weights),
  `routes_api.py` (Ollama+OpenAI+Anthropic inference + model-info), `routes_diag.py` (vision/audio/probe
  test endpoints). `build_app()` calls `register(app)` on each. The 15 routes that rebind a runtime global
  (download/add_model/forget/nodeconfig) or use a build_app-local helper (embed/delete) stay in build_app
  — avoiding the publish/bind stale-snapshot trap. Route bodies byte-identical; globals injected via
  `state.bind`. server.py dropped to ~4350 lines (from ~9090 at the start of the refactor).
- **Worker split (m4c153):** the worker's `Shard` (~1260 lines) and `Worker` (~760) classes split the same
  way — `shard_build.py` (placement / streaming weight-load / from_*), `shard_forward.py` (forward path),
  `worker_load.py` (build/load/pack/unload/TP), `worker_net.py` (next-hop connect/send + data-plane). Shells
  keep `__init__` (+ `Shard._finalize_placement`, which reads the rebound `_CPU_FP32_GEMM` so must read it
  live). `state.py` is now shared by controller and worker (in both EXTRA_UPDATE_FILES); the worker publishes/
  binds at module load. client.py dropped from ~4570 to ~2820 lines. Across the whole refactor the two giants
  went from 9090 + 4570 ≈ 13.7k lines to ~4350 + ~2820 ≈ 7.2k, the rest living in focused 200–1200-line modules.
- **Serving layer split (m4c154):** the request-serving functions `_serve` (Ollama/OpenAI generate+chat),
  `_serve_anthropic` (Claude Code backend), `_count_tokens_anthropic` (+ `_serve`'s private `_prepare`/
  `_ka_is_unload`) moved verbatim into `serving.py`. server.py back-imports the three entry points so the
  already-relocated `routes_api` resolves them through the published namespace; `state.bind(serving)` makes
  their bodies resolve server globals. server.py → ~3730 lines.
- **Status layer split (m4c155):** the read-only status builders `build_status` (the big /status + dashboard
  payload), `_model_entry`, `_loading_view`, `_tag_entry` moved verbatim into `status.py`; server.py
  back-imports `build_status`/`_tag_entry` (called by routes_dashboard/routes_api). Prerequisite fix:
  `load_download_state()` now mutates `DOWNLOAD_STATE` **in place** (`clear()`+`update()`) instead of
  rebinding it, preserving object identity so the `state.publish` snapshot stays live for the relocated
  `_model_entry` (and removing the last `DOWNLOAD_STATE` rebind footgun). server.py → ~3380 lines. The
  history/metrics block was analysed for extraction too but deliberately **left in server.py** — it's
  movable but needs ~16 back-imports (server would re-import almost the whole API) plus the
  `graphs.set_history_sources` identity invariant, i.e. a line-count move with little real decoupling.
- **Multimodal-adapter dedup (#147):** the encoder-free Gemma-4 vision (#143) and audio (#144) loaders
  had grown a duplicated "meta-build the model, then materialize only the multimodal submodule(s) from
  the raw safetensors" loop. That loop — read raw keys per `(submodule, prefix)`, apply per-arch stored-name
  renames before matching, try both the qualified prefix and the `model.`-stripped candidate (Mistral3),
  assign-load, materialize meta buffers, move to device — is now the single `_materialize_submodules(...)`
  helper that both `_load_vision_encoder` (all image arches: Omni / Qwen-VL / Mistral3 / gemma4 / standard)
  and `_load_gemma4_audio_encoder` call. Behaviour is byte-for-byte the pre-refactor vision loop
  (re-validated end-to-end on gemma-4 vision, gemma-4 audio, and Mistral3 split-tower vision).
- **Code-split round 2, increments 1-3 + deploy enablers (2026-07-06):** continuing the m4c152-155
  context-economy refactor with the same contract (byte-identical relocation, `state.bind` globals,
  convergence bridge, `EXTRA_UPDATE_FILES` sync). New controller-only leaves: **`control_plane.py`**
  (~500 lines: control-frame IO, `ControlLink`, the resilient TCP listener, `handle_control`,
  `reaper_loop`, `gen_stall_watchdog` — carries its own stdlib imports because `@dataclass` executes at
  import, before `state.bind`) and **`serving_anthropic.py`** (~450 lines: the `/v1/messages` Anthropic
  engine + `_count_tokens_anthropic`, where all recent vision/audio serve-path edits land; shared
  helpers stay in serving.py, imported leaf-to-leaf). The embed trio (`_serve_embed` + its 3 routes)
  folded into the existing `routes_api.py`. server.py 4,078 → 3,539; serving.py 1,434 → 1,000.
  **Deploy enablers:** `/code_manifest` now also reports `client.py` + the WORKER-side
  `EXTRA_UPDATE_FILES` (regex-extracted from client.py's source — worker deploys bump no controller
  VERSION, so this is their only HTTP-visible ground truth), and the worker convergence bridge
  bounded-retries then **exits 42** on failure so a raw-CDN 404 on a freshly-added module is a bounded
  crash-loop instead of a permanently dead Windows worker (`client.bat` relaunches only on 42). New
  controller modules deploy **two-phase**: module committed+pushed first, pre-staged on every
  controller (`git checkout origin/main -- <mod>.py`), then the server.py that imports it — the
  bridge is fetch-once, and a single commit can race the idle self-updater into a bridge-404
  restart loop.
- **Code-split round 2, increments 4-6 (2026-07-06):** the persistence loaders (`load_node_config` /
  `load_custom_models` / `load_deleted_models`) now mutate their dicts/set **in place** instead of
  rebinding — `main()` publishes the namespace *before* running them, so a rebind stranded every bound
  leaf module on the pre-load empty objects (latent staleness; the m4c155 `DOWNLOAD_STATE` fix,
  generalized). That unblocked **`downloads.py`** (~455 lines: `_pull_repo_interruptible`,
  `_start_download`/`_do_delete`, and the `/download*`, `/add_model`, `/delete`, `/forget`,
  `/api/pull`, `/api/delete` routes — the `DOWNLOAD_*`/`ENCODING` *definitions* stay in server.py where
  the self-update idle lambda live-reads them; the module header documents that invariant) and
  **`routes_shards.py`** (~900 lines merged out of routes_lifecycle.py: shard-cache/packing control
  routes + the worker-facing `/weights` `/weights_tp` `/experts` data plane + the parked `/mtp_probe`
  `/modelcode` debug pair; one module instead of two halves the fleet-sync surface on the route group
  whose convergence-window failure would break every model load). `/nodeconfig` + `/nodeconfig_all`
  landed in routes_api.py (tier config, not downloads). routes_lifecycle.py keeps the true lifecycle
  group (1,341 → 458). Cumulative round-2 effect: **server.py 4,078 → 3,121** · serving.py 1,434 →
  1,000 · routes_lifecycle.py 1,341 → 458; new leaves: control_plane, serving_anthropic, downloads,
  routes_shards. Validated per increment on om3nbox (incl. a real int4 load streaming through the
  relocated `/weights`) and on the production controller (12 nodes re-registered clean).
- **Code-split round 2, increments 7-8 — the worker side (2026-07-06):** first client.py splits under
  the same contract, deployed via VERSION-gated rolling worker self-update (fleet converged in ~3 min,
  zero dropped workers — the exit-42 bridge enabler held). **`worker_hw.py`** (~450 lines: memory/GC,
  capability probes, the read-only route detectors, RAM-module detection, `build_registration`, startup
  cleanup — `_ROUTE_SRC`/`_local_addr` stay in client.py, the live rebind pair). **`worker_update.py`**
  (~265 lines: the self-update machinery + fwd-watchdog + console panel; `EXTRA_UPDATE_FILES` stays in
  client.py, the primary file every worker refreshes). `EmbeddingModel` + `_build_with_autodeps` and
  the HF-local weight helpers moved into the EXISTING `worker_load.py` beside their only call sites
  (zero new fleet-sync surface). client.py 3,699 → 2,923.
- **Code-split round 2, increment 9 — `shard_compile.py` (2026-07-06):** the shard-cache compile/pack
  family (PACKER_VERSION/`_packer_tag`, `pack_linear_int4/_3d/int8`, `pack_unit_tensors`,
  `_shard_cache_root`, `_quant_scope`, `_sha256_file`, `compile_shards`, `verify_shard_cache`,
  `shard_cache_status`, `cache_unit_path`) moved out of shards.py into a **SHARED** leaf (both fleets'
  `EXTRA_UPDATE_FILES`), leaving shards.py a pure weight-serving/streaming layer (1,321 → 872).
  Bind-free by requirement — the `/compile_shards` subprocess imports it in a fresh interpreter — with
  the shared read/dequant/skeleton helpers (and `INT4_GROUP`, a def-time default arg) imported *from*
  shards. Every consumer repointed, including the three a naive grep misses: engine_load's aliased
  `import shards as _sh` (whose failure the non-fatal precompile try/except would have swallowed into
  a silent fleet-wide cache-on-first-load regression), the `/compile_shards` subprocess code-string,
  and the worker remote-pack handler (shipped atomically with the client VERSION bump). Validated the
  hard way: cache deleted and recompiled through the new path — **combined sha1 of all 26 units
  bit-identical to the pre-split cache** — plus a live `/pack_probe` (worker packs via the relocated
  shared packer: `byte_identical: true`). client.py 2,923 → 2,924 wiring net; fleet on m4c186.
- **gfx1151 int4 GEMV DRAM de-aliasing + 70B-shape autotune coverage (#dram-dealias, 2026-07-07):**
  `llama-3.3:70b` int4 decoded at 0.61 tok/s on Strix Halo while dense-32B hit 5.28 — a 4x per-BYTE
  gap, fully reproduced in an isolated kernel bench (per-shape times predicted 0.73 vs observed 0.61).
  Root cause: the split-K w4a16 GEMV walks `qweight` along N with a row stride of K/2 bytes; at the
  70B dims (K=8192 -> a 4096B power-of-two stride) every row maps to the same DRAM channels/banks,
  and any matrix too big for the 32MB MALL collapses to 17-67 GB/s (the 33MB q/o just overflows it;
  the 4MB k/v stays cached and fast — why only big-K dense models ever showed this). Two-part fix,
  no kernel change: (1) `prepare_fused` re-allocates the packed rows on an ODD multiple of 64B
  (kernels already read via `qweight.stride(0)`; +64B/row ≈ 1%, and the aligned 32B shapes got
  FASTER too); (2) the GEMV autotune space adds the `BN=64 / num_warps=16` family the de-aliased
  70B shapes want (28672x8192 gate/up: 1.94ms -> 0.67ms). Matrix-bench ceilings: 70B 0.75 -> 4.50
  tok/s, 32B 6.66 -> 10.81 tok/s.
- **gfx1151 fused-MoE expert-row de-aliasing — MEASURED per shape (#dram-dealias MoE, m4c188,
  2026-07-07):** the fused grouped MoE GEMV has the same exposure as the dense one — within-expert
  rows sit K_pad/2 bytes apart in the contiguous `[E, N, rs]` `Packed4Tensor3D`, and a layer's
  expert stack (134-280 MB) is far past the 32MB MALL, so decode reads are DRAM-cold. But the
  isolated-shape bench (bench_moe_dealias, the dense fix's methodology with a fresh random expert
  subset per call) showed the response is NOT the dense static rule: gemma-4-26b's gate_up
  (rs=1408B, even*64) collapses to 63.7 GB/s and row-padding restores 187.5 GB/s (2.9x; per-token
  expert kernels 9.62 -> 4.45 ms), yet qwen3.6-35b's power-of-two shapes (rs=1024B/256B) run ~96
  GB/s unpadded and padding HALVES them — an even-multiple pad control shows the same, so it's not
  the odd/even story dense followed. Fix accordingly: `Packed4Tensor3D.prepare_fused` (same
  post-placement sweep as the dense pad, ROCm-only) TIMES the production op unpadded vs row-padded
  (`[E,N,rs+64]` buffer kept as the `[:,:,:rs]` view) on DRAM-cold subsets and keeps the winner
  (>=15% to pad; decision cached per (E,N,rs) so 30 layers pay one bench; kernels read via
  `.stride()` so the strided view needs no kernel change). `sqn` joins the MoE autotune key so the
  two variants tune apart (side effect: MoE decode autotune now happens at load, not first decode),
  and the de-aliased gemma gate_up's preferred `BN=256/SPLITK=4/w8` config joins the space (+8%).
  Shard caches stay bit-identical — pad at load, never at pack time. gpt-oss is naturally odd-row
  (rs=1472B) and skips untouched. LIVE-VALIDATED on om3nbox — and the live tensors INVERTED the
  synthetic verdicts, proving the measured design necessary: the collapse is ALLOCATION-dependent
  (physical-page bits in the channel hash), not a shape rule. Live decisions: qwen3.6-35b gate_up
  PAD (0.088 -> 0.052 ms, 1.7x; e2e 15.75 -> 16.34 tok/s clean A/B), its down + both gemma shapes
  keep-unpadded (gemma's live tensors never collapsed; e2e ~20.5 tok/s unchanged). fused-MoE
  self-checks all `-> ACTIVE` (rel ~0.006), gemma `/api/chat` coherent, qwen output text identical
  to baseline at temp 0. **Unload leak fixed (#39, 2026-07-11):** the pad registers a VIEW
  (`buf[:, :rs]`) as the buffer, and the unload storage-release emptied only the view — the padded
  BASE's full storage stayed alive through the C-level `._base` reference (invisible to gc
  referrers; measured ~10 GB surviving a qwen3-30b unload: 48 padded expert stacks + dense pads).
  `_release_shard_vram` (and the t2i release) now empty `t._base` before `t.data`; verified live —
  an unload leaves 0.03 GB allocated (was 10.04). A `[vram-live]` gc diagnostic stays armed in
  worker_hw, printing the live-tensor groups whenever an unload leaves >2 GB allocated.
  Separately, the om3nbox worker runs the allocator with `expandable_segments:True` (systemd
  drop-in; A/B: decode 17.7 → 18.2-18.7 tok/s, coherent, pool returns to the OS instead of
  accumulating fragmentation).
- **t2i OFFLOAD mode (#t2i-offload, 2026-07-11) — render on a card that can't hold the DiT.**
  `/load?model=qwen-image&t2i_offload=1` (or the Load 🖼 dialog's "offloaded" button): the bf16
  pipeline loads into system RAM and accelerate's sequential offload streams each block to the
  controller-co-located GPU just-in-time per forward — VRAM peak is transients (~blocks +
  activations + VAE), so the card's resident models STAY. Placement needs only ~4 GB free VRAM
  plus RAM for the weights and NEVER evicts (it fails with the requirement instead); bf16-only
  (the int4 fused kernels are prepared per-device and don't survive block hopping) — which also
  makes it the REFERENCE-quality path. Measured on beast (4070TiS 16 GB, resident models loaded):
  first render 20 steps @1024² in **510 s (~25.5 s/step — faster than om3nbox's GPU-resident
  int4 at ~34)**, sign text exact, **0 GB GPU resident, mid-render VRAM 0.24/15.57 GB**, load 2 s
  from page cache (~8 min cold from the weights disk; the t2i load reply wait is 20 min for it).
  Getting there hardened placement concurrency for ALL t2i loads: they now register in the
  reservation ledger (concurrent auto-loads budget around the multi-minute build) AND subtract
  other in-flight loads' reservations (a cache-served auto-load planned seconds earlier was
  streaming toward a full card — both directions of the same race, both observed live as OOMs).
- **Code-split round 2, increment 10 — `worker_quant.py`, client.py's flagship (m4c189, 2026-07-08):**
  the whole quant/kernel family (~1,660 lines) relocated out of client.py into a SELF-CONTAINED worker
  leaf (shard_compile precedent — deliberately NOT in `state.bind`): the guarded module-level triton/tl
  import + the #triton-race Autotuner patch, the CPU fp32-GEMM family (flags + `tune_cpu_threads` +
  `_accelerate_cpu_linears`), the int8/int4 cores (QuantLinear/QuantLinear4 incl. `prepare_fused` and
  both #dram-dealias paths), the w4a16 triton kernels (dense + fused-MoE + expert tensor-subclass),
  the packers (`Packed4Tensor3D`, `_pack4_expert`/`_pack4_3d`), fused-MoE + gpt-oss installs, the MoE
  offload bridge, per-expert/streamed builds, meta-expert detectors, and `_assign_meta_from_sd`. All
  relocated bodies byte-identical (verified against git HEAD). worker_quant.py is now the CANONICAL
  home of the runtime-rebound flag family (`_CPU_FP32_GEMM`/`_CPU_FP32_MIN_ROWS`/`_CPU_BF16_GEMM_OK`/
  `_FUSED_INT4`): client.py back-imports only the functions/classes (never the flags) before
  `state.publish` so shard_build's bind-injected bare names keep resolving, and the two flagged
  NOT-byte-identical edits make every flag access a live module attribute — `main()`'s `--no-cpu-fp32`
  sets `worker_quant._CPU_FP32_GEMM = False` (was `global`), and `Shard._finalize_placement` reads
  `worker_quant._CPU_FP32_GEMM` / `._FUSED_INT4` (a from-import copy would freeze the pre-main values
  and silently ignore the CLI flag / the aarch64 FEAT_BF16 crash-guard rebind). Zero leftover
  flag/triton/tl definitions remain in client.py (they would re-enter the publish snapshot and stomp
  the live values). worker_quant.py joins `EXTRA_UPDATE_FILES` + the convergence-bridge tuple;
  bit-identity doc contracts in shard_compile.py/shards.py/bench_moe_w4a16.py repointed.
  client.py 3,032 → 1,346.
- **Code-split round 2, increment 11 — `media_encode.py` + EngineSpeechMixin, server.py's flagship
  (2026-07-08):** the media/speech encode family relocated verbatim into a new bound controller leaf:
  `_encode_images`, `_encode_audio_gemma4`, `_encode_audio`, the #P6 speech-out group (`_SPEECH_CACHE`/
  `_SPEECH_MAT`/`_ensure_spk_dict`/`_materialize_from_prefix`/`SPEECH_DEVICE`/`_load_speech_components`),
  and `Engine.generate_speech` as **EngineSpeechMixin** (Engine now composes four mixins; only `__init__`
  remains on the shell). **`ENCODING`'s canonical home moved with them** — this supersedes the "stays in
  server.py" rationale recorded at m4c152/Inc 5: all FOUR `global ENCODING` mutators live in
  media_encode.py, the self-updater's idle lambda reads `media_encode.ENCODING` as a live module
  attribute, and ENCODING is never back-imported or published (an int snapshot would freeze the idle
  gate open — the original "ENCODING hazard", now closed by moving definition + mutators as one unit;
  state.py's SAFETY NOTE documents both valid patterns). Relocated bodies byte-identical (verified
  against git HEAD); the hazard comments in state.py/multimodal.py/model_store.py/downloads.py
  repointed. server.py 3,152 → 2,551.
