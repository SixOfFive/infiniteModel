#!/usr/bin/env python3
"""
InfiniteModel — multimodal (vision + audio + speech) encoder helpers (server-only leaf module).

Extracted from server.py (#38, step B) to shrink that file. These are the #22 distributed-Omni
controller-side helpers: image/audio decode + collect, the meta-load + tower-materialize encoders
(vision tower / Omni audio tower), the meta-tensor materializer, processor/feature-extractor caches,
placeholder/position helpers, and the audio-response encoder + speaker resolver.

They are SELF-CONTAINED: none touch controller state (engine, registry, MODELS, METRICS, ENCODING,
app routes, …). The ONE controller dependency — resolving a model's on-disk weights dir — is supplied
by DEPENDENCY INJECTION: server.py calls ``set_model_dir_resolver(_controller_model_dir)`` once at
import time, and the encoders call ``_MODEL_DIR_FN(target_id)`` instead of importing it back (no
back-import of server -> no import cycle).

This is a controller-only leaf module: it must NEVER ``import server``. It is listed in server.py's
EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync across the fleet, and server.py
imports its symbols back via a convergence-bridge import.

NOTE (what STAYED in server.py and why): the public encode entry points ``_encode_images`` /
``_encode_audio`` / ``_load_speech_components`` and the whole speech-out group (``_SPEECH_CACHE`` /
``_SPEECH_MAT`` / ``_ensure_spk_dict`` / ``_materialize_from_prefix``) were NOT moved — they mutate
the ``ENCODING`` idle-gate counter (read by the self-updater's idle lambda in server.py; a moved
``global ENCODING`` would bind to THIS module and silently decouple the gate) and/or use server-only
globals (MODELS_DIR / _safe_name / HF_TOKEN / shutil). They call the helpers below, which resolve
through server.py's convergence-bridge import — so leaving them behind is correct and cycle-free.
"""
from __future__ import annotations

import contextlib
import os
import time

GB = 1024 ** 3
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Dependency injection: controller model-dir resolver (no back-import of server)
# ---------------------------------------------------------------------------
_MODEL_DIR_FN = None
_LOCAL_DIR_FN = None


def set_model_dir_resolver(fn):
    """server.py injects its ``_controller_model_dir`` here at import time, so the encoders
    below can resolve a model's on-disk weights dir WITHOUT importing server (no import cycle)."""
    global _MODEL_DIR_FN
    _MODEL_DIR_FN = fn


def set_local_dir_resolver(fn):
    """server.py injects ``_local_model_dir`` — returns a model's on-disk dir IF already present,
    without downloading/converting. Used by _get_tokenizer to load the tokenizer from the local dir
    (which always carries a usable tokenizer.json, incl. GGUF-normalized models) rather than the HF
    repo id (a GGUF-only repo has no tokenizer.json -> from_pretrained would need sentencepiece)."""
    global _LOCAL_DIR_FN
    _LOCAL_DIR_FN = fn


_TOK_CACHE: dict = {}   # target_id -> tokenizer; the build is slow (minutes) for big


def _get_tokenizer(target_id: str):
    """Load a model's tokenizer, cached by target_id. AutoTokenizer.from_pretrained can
    take minutes for large models (the slow 'finalization' after a load); caching makes
    every RELOAD of the same model (ctx change, dirty re-plan, unload+reload) instant."""
    tok = _TOK_CACHE.get(target_id)
    if tok is None:
        import os
        from transformers import AutoTokenizer
        # Source order: prefer the LOCAL model dir if it already holds a tokenizer (a GGUF-normalized
        # model — and any downloaded model — saves tokenizer.json there), because the HF repo id may
        # have no usable tokenizer (a GGUF-only repo ships .gguf, so from_pretrained(repo) would try to
        # build a slow tokenizer and fail without sentencepiece/tiktoken). Fall back to the repo id.
        sources = []
        if _LOCAL_DIR_FN is not None:
            try:
                d = _LOCAL_DIR_FN(target_id)
                if d and (os.path.exists(os.path.join(d, "tokenizer.json"))
                          or os.path.exists(os.path.join(d, "tokenizer_config.json"))):
                    sources.append(d)
            except Exception:
                pass
        sources.append(target_id)
        # trust_remote_code: harmless for the existing models, required for nomic's BERT-style
        # tokenizer (custom tokenization code shipped in the repo).
        # #devstral-eos: Mistral-Small-3.1-derived tokenizers (Devstral, Ministral) ship a broken
        # pretokenizer regex; transformers warns "set fix_mistral_regex=True" and WITHOUT it the
        # prompt mis-tokenizes -> the model can emit EOS (id=2) immediately = 0 output tokens. Pass
        # the flag when the tokenizer accepts it; fall back cleanly for tokenizers that don't.
        last = None
        for src in sources:
            for _kw in (dict(trust_remote_code=True, fix_mistral_regex=True),
                        dict(trust_remote_code=True)):
                try:
                    tok = AutoTokenizer.from_pretrained(src, **_kw)
                    break
                except Exception as exc:
                    last = exc
                    tok = None
            if tok is not None:
                break
        if tok is None:
            raise last if last is not None else RuntimeError(f"no tokenizer for {target_id}")
        _TOK_CACHE[target_id] = tok
    return tok


# --- #22 distributed-Omni VISION: controller-side image input + processor ---------------
# The encoder runs on the controller (per the chosen design): decode image content blocks,
# run the model's processor to expand <|image_pad|> placeholders + produce pixel_values, then
# (next increment) run model.model.visual via get_image_features -> embeds spliced at stage 0.
_PROCESSOR_CACHE: dict = {}   # target_id -> AutoProcessor (use_fast=False: PIL-only, no torchvision)


def _get_processor(target_id: str):
    """Cached image/text processor. use_fast=False keeps it on the PIL-only slow path so we
    NEVER need torchvision (whose default-PyPI install clobbers the CUDA torch on Windows)."""
    p = _PROCESSOR_CACHE.get(target_id)
    if p is None:
        from transformers import AutoProcessor
        p = AutoProcessor.from_pretrained(target_id, use_fast=False)
        _PROCESSOR_CACHE[target_id] = p
    return p


def _decode_image(block: dict):
    """An Anthropic/OpenAI image content block -> a PIL RGB image (or None). Handles
    Anthropic {type:image, source:{type:base64|url,...}} and OpenAI {type:image_url,
    image_url:{url}} incl. data: URLs. PIL is required (Pillow); no torchvision."""
    import base64, io, urllib.request
    from PIL import Image
    data = None
    t = block.get("type")
    if t == "image":
        src = block.get("source") or {}
        if src.get("type") == "base64":
            data = base64.b64decode(src.get("data", ""))
        elif src.get("type") in ("url", "image"):
            with urllib.request.urlopen(src.get("url", ""), timeout=20) as r:
                data = r.read()
    elif t == "image_url":
        u = (block.get("image_url") or {}).get("url", "") if isinstance(block.get("image_url"), dict) else block.get("image_url", "")
        if u.startswith("data:"):
            data = base64.b64decode(u.split(",", 1)[1])
        elif u:
            with urllib.request.urlopen(u, timeout=20) as r:
                data = r.read()
    if not data:
        return None
    return Image.open(io.BytesIO(data)).convert("RGB")


def _collect_images(messages) -> list:
    """Pull every image (in order) out of an Anthropic message list -> [PIL.Image]."""
    imgs = []
    for m in (messages or []):
        c = m.get("content")
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") in ("image", "image_url"):
                    with contextlib.suppress(Exception):
                        im = _decode_image(blk)
                        if im is not None:
                            imgs.append(im)
    return imgs


def _audio_bytes_to_waveform(data: bytes, sr: int = 16000):
    """Decode arbitrary audio bytes -> mono float32 @ sr Hz. Try librosa (handles
    mp3/wav/flac + resample), then soundfile (+ linear resample), then stdlib `wave`
    (PCM WAV only). Returns a 1-D numpy float32 array."""
    import io
    import numpy as np
    # 1) librosa: broadest format + sample-rate support
    try:
        import librosa
        y, _ = librosa.load(io.BytesIO(data), sr=sr, mono=True)
        return np.asarray(y, dtype=np.float32)
    except Exception:
        pass
    # 2) soundfile (wav/flac/ogg): manual linear resample if needed
    try:
        import soundfile as sf
        y, in_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if in_sr != sr and len(y) > 1:
            n = int(round(len(y) * sr / in_sr))
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        return np.asarray(y, dtype=np.float32)
    except Exception:
        pass
    # 3) stdlib wave: PCM WAV only (the format our own /audio_e2e test emits). Wrapped in
    # try/except like the paths above so a corrupt/truncated WAV falls through to a clean
    # RuntimeError instead of an unhandled wave.Error/struct.error.
    try:
        import wave
        with wave.open(io.BytesIO(data), "rb") as w:
            in_sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        y = np.frombuffer(raw, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max)
        if nch > 1:
            # truncate to a whole number of frames before reshape — a truncated WAV can
            # leave len(y) not divisible by nch (ValueError on reshape otherwise).
            valid = (len(y) // nch) * nch
            y = y[:valid].reshape(-1, nch).mean(axis=1)
        if in_sr != sr and len(y) > 1:
            n = int(round(len(y) * sr / in_sr))
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        return np.asarray(y, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"audio decode failed (librosa/soundfile/wave all failed): "
                           f"{type(exc).__name__}: {exc}") from exc


def _decode_audio(block: dict):
    """An audio content block -> 1-D float32 mono waveform @16 kHz (or None). Supports
    OpenAI {type:input_audio, input_audio:{data:b64, format}} and a generic Anthropic-style
    {type:audio, source:{type:base64|url, data|url}} / {type:audio_url, audio_url:{url}}
    incl. data: URLs."""
    import base64, urllib.request
    data = None
    t = block.get("type")
    if t == "input_audio":
        ia = block.get("input_audio") or {}
        d = ia.get("data")
        if d:
            data = base64.b64decode(d)
    elif t == "audio_url":
        u = (block.get("audio_url") or {}).get("url", "") if isinstance(block.get("audio_url"), dict) \
            else block.get("audio_url", "")
        if u.startswith("data:"):
            # a well-formed data: URL is "data:<mime>;base64,<payload>"; guard the comma
            # split so a malformed header (no comma) returns None instead of IndexError.
            parts = u.split(",", 1)
            data = base64.b64decode(parts[1]) if len(parts) > 1 else None
        elif u:
            with urllib.request.urlopen(u, timeout=20) as r:
                data = r.read()
    elif t == "audio":
        src = block.get("source") or {}
        if src.get("type") == "base64":
            data = base64.b64decode(src.get("data", ""))
        elif src.get("type") in ("url", "audio"):
            with urllib.request.urlopen(src.get("url", ""), timeout=20) as r:
                data = r.read()
    if not data:
        return None
    return _audio_bytes_to_waveform(data)


def _collect_audio(messages) -> list:
    """Pull every audio clip (in order) out of an Anthropic message list -> [np.float32].
    A clip that fails to decode is dropped but LOGGED with its index (so a multi-clip
    request that silently falls back to text-only downstream can be diagnosed — the
    placeholder count won't match the decoded-clip count if one was dropped)."""
    auds = []
    idx = -1
    for m in (messages or []):
        c = m.get("content")
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") in ("audio", "audio_url", "input_audio"):
                    idx += 1
                    try:
                        wav = _decode_audio(blk)
                    except Exception as exc:
                        print(f"[audio] dropped clip #{idx} (decode failed: "
                              f"{type(exc).__name__}: {str(exc)[:120]})")
                        continue
                    if wav is not None and len(wav):
                        auds.append(wav)
                    else:
                        print(f"[audio] dropped clip #{idx} (empty/undecodable block)")
    return auds


_VISION_CACHE: dict = {}   # target_id -> (model_with_only_visual_materialized, device)
_VISION_MAT: dict = {}     # target_id -> [(name, shape, how)]  (diagnostics from materialize)
_VISION_LOG = os.path.join(_PROJECT_DIR, "vision_diag.log")


def _vlog(msg: str) -> None:
    """Phase log that SURVIVES a process crash. The vision encode has hard-crash-restarted
    the controller (uncatchable native fault), so console prints are lost on restart. Append
    each phase to a file FIRST (flush+fsync) so /vision_log can report the last step reached
    AFTER the relaunch — pinpointing exactly which op killed the process."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with contextlib.suppress(Exception):
        with open(_VISION_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _recompute_rotary(mod, dev) -> bool:
    """Re-derive a rotary module's computed `inv_freq` (a non-persistent buffer, so it lands on
    meta after the assign-load) on `dev` using the MODULE'S OWN rope-init — so arch-specific
    layouts are exact: Pixtral's 2D per-patch frequency TABLE [positions, dim], Qwen's 1D vector,
    Gemma's theta. A flat one-size formula (or zero-fill) corrupts anything but plain 1D RoPE.
    Returns True iff it rebuilt inv_freq. Used by _materialize_meta_tensors for the 2D case."""
    import torch
    cfg = getattr(mod, "config", None)
    if cfg is None:
        return False
    fn = (getattr(mod, "rope_init_fn", None)
          or getattr(mod, "compute_default_rope_parameters", None))
    if fn is None:                          # fall back to the transformers registry by rope_type
        with contextlib.suppress(Exception):
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            rt = (getattr(mod, "rope_type", None)
                  or getattr(cfg, "rope_type", None) or "default")
            fn = ROPE_INIT_FUNCTIONS.get(rt)
    if fn is None:
        return False
    inv_freq, scaling = None, None
    for call in (lambda: fn(cfg, dev), lambda: fn(cfg, device=dev), lambda: fn(cfg)):
        try:
            res = call()
        except TypeError:
            continue                        # wrong arity — try the next call shape
        except Exception:
            return False
        if isinstance(res, tuple):
            inv_freq = res[0]
            scaling = res[1] if len(res) > 1 else None
        else:
            inv_freq = res
        break
    if not isinstance(inv_freq, torch.Tensor):
        return False
    mod.register_buffer("inv_freq", inv_freq.to(dev), persistent=False)
    if hasattr(mod, "original_inv_freq"):
        mod.original_inv_freq = inv_freq.detach().clone().to(dev)
    if scaling is not None and hasattr(mod, "attention_scaling"):
        with contextlib.suppress(Exception):
            mod.attention_scaling = float(scaling)
    return True


def _materialize_meta_tensors(module, dev: str) -> list:
    """After a partial `load_state_dict(..., assign=True)`, the weights present in the
    safetensors are real but COMPUTED non-persistent buffers (e.g. rotary `inv_freq`,
    registered persistent=False) stay on the meta device — they were built by __init__
    under `torch.device('meta')`, so `.to(dev)` on the module dies ("can't copy out of
    meta tensor"). Give every leftover meta tensor real storage on `dev`:
      * `*inv_freq` buffers -> recompute from shape (Qwen vision rotary uses theta=10000),
      * any other meta buffer -> zeros (harmless: dropout masks, cached sizes, etc.),
      * any meta PARAM -> zeros + flagged 'MISSING_WEIGHT' (a real red flag if it appears).
    Returns a diagnostic list so /vision_test can surface exactly what was synthesized."""
    import torch
    report = []
    for mod in module.modules():
        # Pixtral-style 2D positional rotary table (and any rotary module exposing its own rope
        # init): rebuild inv_freq on `dev` via the MODULE'S OWN function so the arch layout/theta
        # is exact. Gated on ndim>=2 so Qwen's validated 1D inv_freq path below is untouched.
        if any(("inv_freq" in n and b is not None and b.device.type == "meta" and b.ndim >= 2)
               for n, b in mod._buffers.items()):
            if _recompute_rotary(mod, dev):
                report.append((f"{type(mod).__name__}.inv_freq", [], "rope_init_fn(module 2D)"))
        for name, buf in list(mod._buffers.items()):
            if buf is None or buf.device.type != "meta":
                continue
            shape, dtype = buf.shape, buf.dtype
            if "inv_freq" in name and buf.ndim == 1 and shape[0] > 0:
                dim = shape[0] * 2
                new = (1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                                       device=dev) / dim)))
                new = new.to(dtype if dtype.is_floating_point else torch.float32)
                how = "inv_freq(theta=1e4)"
            elif "positional_embedding" in name and buf.ndim == 2 and shape[0] > 0 and shape[1] > 1:
                # Whisper-style sinusoidal positional table (Qwen2.5-Omni audio encoder's
                # SinusoidsPositionEmbedding): a non-persistent buffer, so it's NOT in the
                # safetensors and lands on meta. Zero-filling it (the generic branch below)
                # strips ALL positional information from the audio encoder -> garbage audio
                # features. Recompute the real sinusoids: cat([sin, cos], dim=1), theta=1e4.
                import math
                length, channels = int(shape[0]), int(shape[1])
                half = channels // 2
                log_ti = math.log(10000.0) / max(1, (half - 1))
                inv = torch.exp(-log_ti * torch.arange(half, dtype=torch.float32, device=dev))
                scaled = torch.arange(length, dtype=torch.float32, device=dev)[:, None] * inv[None, :]
                new = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
                new = new.to(dtype if dtype.is_floating_point else torch.float32)
                how = "whisper_sinusoids(theta=1e4)"
            else:
                new = torch.zeros(shape, dtype=dtype, device=dev)
                # A >=2D inv_freq reaching here means _recompute_rotary did NOT rebuild it (unknown
                # rope arch / missing config). Zero-fill would SILENTLY strip all vision positions
                # (cos=1/sin=0) -> garbage embeds with no error. Flag it so the missing-list below
                # surfaces it in /vision_test rather than corrupting silently. (Never fires on the
                # in-scope Qwen/Pixtral paths.)
                how = "zeros[MISSING_ROTARY]" if "inv_freq" in name else "zeros"
            mod._buffers[name] = new
            report.append((name, list(shape), how))
        for name, p in list(mod._parameters.items()):
            if p is None or p.device.type != "meta":
                continue
            mod._parameters[name] = torch.nn.Parameter(
                torch.zeros(p.shape, dtype=p.dtype, device=dev), requires_grad=False)
            report.append((name, list(p.shape), "zeros[MISSING_WEIGHT]"))
    return report


VISION_DEVICE = os.environ.get("INFINITEMODEL_VISION_DEVICE", "cpu").strip().lower()


def _pick_vision_device() -> str:
    """Where to run the vision tower. DEFAULT = CPU: the controller is SHARED infra (beast),
    and running the vision forward on its GPU hard-crash-restarted the whole controller even
    with the GPU nearly empty (native CUDA crash under beast's torch build) — unacceptable
    for a shared box. The tower is small, so CPU is reliable and fast enough for a one-time
    encode. Set INFINITEMODEL_VISION_DEVICE=auto to opt back into 'GPU when >3 GB free'."""
    if VISION_DEVICE in ("cpu", ""):
        return "cpu"
    if VISION_DEVICE.startswith("cuda"):
        return VISION_DEVICE
    # "auto"
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            if free > 3 * GB:
                return "cuda:0"
    return "cpu"


def _resolve_visual(model):
    """The vision tower module + its weight-prefix, handling both layouts:
      * standard image-text models (Qwen3.6-35B): model.model.visual  ('model.visual.')
      * Qwen2.5-Omni: model.thinker.visual                            ('thinker.visual.')
    Returns (visual_module, weight_prefix)."""
    th = getattr(model, "thinker", None)
    if th is not None and getattr(th, "visual", None) is not None:
        return th.visual, "thinker.visual."
    return model.model.visual, "model.visual."


def _visual_modules(model):
    """The vision submodule(s) to materialize, as (submodule, full_weight_prefix) pairs — a LIST
    so an arch with a SPLIT tower materializes every piece:
      * Qwen2.5-Omni:        thinker.visual                         ('thinker.visual.')
      * standard image-text: model.model.visual                    ('model.visual.')
      * Mistral3 / Pixtral / Llava-style: model.model.vision_tower PLUS the SEPARATE
                             model.model.multi_modal_projector      (two prefixes)
    The projector is its own top-level module in these arches (not nested in the tower), so it
    must be loaded + materialized too or get_image_features projects through meta weights."""
    th = getattr(model, "thinker", None)
    if th is not None and getattr(th, "visual", None) is not None:
        return [(th.visual, "thinker.visual.")]
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "vision_tower", None) is not None \
            and getattr(inner, "multi_modal_projector", None) is not None:
        return [(inner.vision_tower, "model.vision_tower."),
                (inner.multi_modal_projector, "model.multi_modal_projector.")]
    return [(model.model.visual, "model.visual.")]


def _vision_cfg_and_token(model):
    """(vision_config, image_token_id) handling Omni's nesting (vision_config + image_token
    live under thinker_config; image_token_id is NULL at the top Omni config, like audio)."""
    cfg = model.config
    base = getattr(cfg, "thinker_config", None) or cfg
    vcfg = getattr(base, "vision_config", None) or getattr(cfg, "vision_config", None)
    itid = (getattr(base, "image_token_index", None) or getattr(base, "image_token_id", None)
            or getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None))
    return vcfg, itid


def _load_vision_encoder(target_id: str):
    """Meta-load the full multimodal model (ZERO memory — text LM stays on meta) and
    materialize ONLY the vision tower from the safetensors, so we can run the tower without
    loading the big text model. Handles standard layout (model.model.visual) AND Qwen2.5-Omni
    (model.thinker.visual, built via AutoModelForTextToWaveform). Cached per target_id."""
    cached = _VISION_CACHE.get(target_id)
    if cached is not None:
        return cached
    import torch, glob
    from transformers import AutoConfig
    from safetensors import safe_open
    t0 = time.time()
    _vlog(f"[vision] load START {target_id}")
    cfg = AutoConfig.from_pretrained(target_id)
    is_omni = getattr(cfg, "thinker_config", None) is not None
    _vlog(f"[vision] config loaded ({'Omni' if is_omni else 'standard'}); meta-building model ...")
    with torch.device("meta"):
        if is_omni:
            from transformers import AutoModelForTextToWaveform
            model = AutoModelForTextToWaveform.from_config(cfg)
        else:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_config(cfg)
    model.eval()
    t_meta = time.time()
    _vlog(f"[vision] meta-built {type(model).__name__} in {t_meta - t0:.1f}s")
    mods = _visual_modules(model)
    model_dir = _MODEL_DIR_FN(target_id)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    dev = _pick_vision_device()
    _vlog(f"[vision] {len(mods)} vision submodule(s) to materialize "
          f"({', '.join(p for _, p in mods)}); scanning {len(files)} shard(s); device={dev}")
    total, mat_all = 0, []
    for submod, prefix in mods:
        sd = {}
        for fn in files:
            with safe_open(fn, framework="pt") as fh:
                hits = [k for k in fh.keys() if k.startswith(prefix)]
                for k in hits:
                    sd[k[len(prefix):]] = fh.get_tensor(k)
        if not sd:
            raise RuntimeError(f"no '{prefix}*' weights found in {model_dir}")
        # Computed non-persistent buffers (rotary inv_freq) stay on meta after the assign-load;
        # _materialize_meta_tensors gives them real storage on `dev` (arch-correct for 2D rope)
        # so the .to(dev) below is safe.
        submod.load_state_dict(sd, strict=False, assign=True)
        mat = _materialize_meta_tensors(submod, dev)
        submod.to(dev)
        total += len(sd)
        mat_all += mat
        _vlog(f"[vision]   '{prefix}': {len(sd)} tensors assign-loaded + moved to {dev}; "
              f"materialized {len(mat)} meta tensor(s)")
    _VISION_MAT[target_id] = mat_all
    missing = [m for m in mat_all if "MISSING_" in m[2]]   # MISSING_WEIGHT or MISSING_ROTARY
    _vlog(f"[vision] encoder READY {target_id}: {total} tensors across {len(mods)} "
          f"submodule(s) on {dev}; materialized {len(mat_all)}"
          + (f"; WARNING {len(missing)} missing: {[m[0] for m in missing][:5]}"
             if missing else ""))
    _VISION_CACHE[target_id] = (model, dev)
    return model, dev


_IMGPROC_CACHE: dict = {}   # target_id -> AutoImageProcessor (image-only, no video processor)


def _get_image_processor(target_id: str):
    """The IMAGE processor only (not the bundled AutoProcessor, which also pulls a VIDEO
    processor that hard-requires torchvision). use_fast=False keeps it on the PIL/numpy slow
    path so we need neither torchvision nor the video processor — Pillow alone."""
    p = _IMGPROC_CACHE.get(target_id)
    if p is None:
        from transformers import AutoImageProcessor
        p = AutoImageProcessor.from_pretrained(target_id, use_fast=False)
        _IMGPROC_CACHE[target_id] = p
    return p


def _as_feature_tensor(feats):
    """Vision towers / get_image_features may return a bare tensor, a ModelOutput
    (BaseModelOutputWithPooling, etc.), or a list/tuple. Normalize to the per-token
    hidden-state tensor we splice at image-placeholder positions."""
    import torch
    if isinstance(feats, torch.Tensor):
        return feats
    if isinstance(feats, (list, tuple)) and feats:
        return _as_feature_tensor(feats[0])
    for attr in ("last_hidden_state", "image_embeds", "image_features",
                 "hidden_states", "pooler_output"):
        v = getattr(feats, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if hasattr(feats, "values"):
        for v in feats.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"can't extract a tensor from {type(feats).__name__}")


def _pick_merged_embeds(feats, out_hidden):
    """The visual tower returns the LM-READY MERGED tokens [prod(grid)/merge^2, out_hidden]
    in `pooler_output`, while `last_hidden_state` is the larger PRE-merge ViT backbone
    [patches, vision_hidden]. Pick the candidate whose hidden dim == out_hidden so we splice
    the merged tokens (right count + right width = text hidden), not the backbone."""
    import torch
    if isinstance(feats, torch.Tensor):
        return feats
    cands = []
    for attr in ("pooler_output", "image_embeds", "image_features",
                 "last_hidden_state", "hidden_states"):
        v = getattr(feats, attr, None)
        if isinstance(v, torch.Tensor):
            cands.append(v)
    if not cands and hasattr(feats, "values"):
        cands = [v for v in feats.values() if isinstance(v, torch.Tensor)]
    if out_hidden:
        for c in cands:
            if c.shape[-1] == out_hidden:
                return c
    return cands[0] if cands else _as_feature_tensor(feats)


# ===================== #22 inc 5b: AUDIO encoder (Qwen2.5-Omni) ======================
# Mirror of the vision encoder, but for the Omni audio tower. The full Omni model is
# meta-built (zero memory; talker + token2wav + text LM all stay on meta) and ONLY
# `thinker.audio_tower` (a Whisper-derived Qwen2_5OmniAudioEncoder) is materialized from
# the safetensors. We then drive `thinker.get_audio_features(input_features,
# feature_attention_mask)` to produce per-audio-token embeds [tokens, text_hidden] that
# splice into stage-0's embed output at the <|AUDIO|> placeholder positions (reusing the
# inc-3 'mm' transport). The audio_tower output is ALREADY projected to the text hidden
# size, so — like vision's merged tokens — these are LM-ready.
_AUDIO_CACHE: dict = {}   # target_id -> (model_with_only_audio_tower_materialized, device)
_AUDIO_MAT: dict = {}     # target_id -> [(name, shape, how)]  (materialize diagnostics)

AUDIO_DEVICE = os.environ.get("INFINITEMODEL_AUDIO_DEVICE", "cpu").strip().lower()


def _pick_audio_device() -> str:
    """Where to run the audio tower. DEFAULT = CPU (same reasoning as the vision tower:
    the controller is shared infra and the tower is small). INFINITEMODEL_AUDIO_DEVICE=auto
    opts into 'GPU when >3 GB free'."""
    if AUDIO_DEVICE in ("cpu", ""):
        return "cpu"
    if AUDIO_DEVICE.startswith("cuda"):
        return AUDIO_DEVICE
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            if free > 3 * GB:
                return "cuda:0"
    return "cpu"


def _load_audio_encoder(target_id: str):
    """Meta-load the full Omni model and materialize ONLY thinker.audio_tower from the
    safetensors (keys 'thinker.audio_tower.*'), so we can run get_audio_features without
    loading the text LM / talker / token2wav. Cached per target_id. Returns (model, device)
    where model is the Qwen2_5OmniForConditionalGeneration with a live `.thinker`."""
    cached = _AUDIO_CACHE.get(target_id)
    if cached is not None:
        return cached
    import torch, glob
    from transformers import AutoConfig, AutoModelForTextToWaveform
    from safetensors import safe_open
    t0 = time.time()
    _vlog(f"[audio] load START {target_id}")
    cfg = AutoConfig.from_pretrained(target_id)
    _vlog("[audio] config loaded; meta-building full Omni model ...")
    with torch.device("meta"):
        model = AutoModelForTextToWaveform.from_config(cfg)
    model.eval()
    t_meta = time.time()
    _vlog(f"[audio] meta-built {type(model).__name__} in {t_meta - t0:.1f}s")
    thinker = getattr(model, "thinker", model)
    tower = getattr(thinker, "audio_tower", None)
    if tower is None:
        raise RuntimeError(f"{type(model).__name__} has no thinker.audio_tower")
    model_dir = _MODEL_DIR_FN(target_id)
    prefix = "thinker.audio_tower."
    sd = {}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    _vlog(f"[audio] scanning {len(files)} safetensors shard(s) for '{prefix}*' ...")
    for fi, fn in enumerate(files):
        with safe_open(fn, framework="pt") as fh:
            hits = [k for k in fh.keys() if k.startswith(prefix)]
            for k in hits:
                sd[k[len(prefix):]] = fh.get_tensor(k)
        if hits:
            _vlog(f"[audio]   shard {fi+1}/{len(files)}: +{len(hits)} audio tensors "
                  f"(total {len(sd)})")
    if not sd:
        raise RuntimeError(f"no '{prefix}*' weights found in {model_dir}")
    t_read = time.time()
    _vlog(f"[audio] read {len(sd)} audio tensors in {t_read - t_meta:.1f}s; assign-loading ...")
    tower.load_state_dict(sd, strict=False, assign=True)
    dev = _pick_audio_device()
    _vlog(f"[audio] assign-loaded; materializing meta buffers; target device={dev}")
    mat = _materialize_meta_tensors(tower, dev)
    _AUDIO_MAT[target_id] = mat
    _vlog(f"[audio] materialized {len(mat)} meta tensor(s); moving tower to {dev} ...")
    tower.to(dev)
    missing = [m for m in mat if "MISSING_WEIGHT" in m[2]]
    _vlog(f"[audio] encoder READY {target_id}: {len(sd)} tensors on {dev}; "
          f"materialized {len(mat)}"
          + (f"; WARNING {len(missing)} missing: {[m[0] for m in missing][:5]}"
             if missing else ""))
    _AUDIO_CACHE[target_id] = (model, dev)
    return model, dev


_AUDIOFE_CACHE: dict = {}   # target_id -> WhisperFeatureExtractor


def _get_audio_feature_extractor(target_id: str):
    """The audio FEATURE EXTRACTOR only (WhisperFeatureExtractor) — NOT the bundled
    Qwen2_5OmniProcessor, which also pulls an image + video processor (video hard-requires
    torchvision, exactly the trap the vision path hit). AutoFeatureExtractor loads only the
    mel/log-spectrogram front-end we need."""
    fe = _AUDIOFE_CACHE.get(target_id)
    if fe is None:
        from transformers import AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained(target_id)
        _AUDIOFE_CACHE[target_id] = fe
    return fe


def _omni_audio_token_id(model) -> int | None:
    """audio_token_id is NULL at the TOP Omni config (same as image_token was); it lives in
    thinker_config. Read it from the built model's config."""
    cfg = getattr(model, "config", None)
    tcfg = getattr(cfg, "thinker_config", None) or cfg
    for src in (tcfg, cfg):
        if src is None:
            continue
        for nm in ("audio_token_index", "audio_token_id"):
            v = getattr(src, nm, None)
            if v is not None:
                return int(v)
    return None


def _audio_out_lengths(feature_lens):
    """Per-audio <|AUDIO|> token counts — the AUTHORITATIVE Qwen2.5-Omni PROCESSOR formula
    (transformers processing_qwen2_5_omni.py), so each placeholder expands to exactly the
    number of embeds the tower will emit for that clip (inc 5c):
        input_lengths = (feat_len - 1)//2 + 1     # audio_tower conv, stride 2
        audio_tokens  = (input_lengths - 2)//2 + 1 # encoder avg-pool, stride 2
    NOTE: the tower's own _get_feat_extract_output_lengths only does the FIRST step (the
    conv output), which OVER-counts 2x — the encoder's pooling halves it again. `feature_lens`
    is a 1-D tensor of per-audio feature_attention_mask sums. Returns a list[int]."""
    counts = []
    for L in feature_lens.tolist():
        L = int(L)
        input_lengths = (L - 1) // 2 + 1
        counts.append((input_lengths - 2) // 2 + 1)
    return counts


_OPENAI_VOICE_MAP = {   # OpenAI voice names -> our Qwen2.5-Omni speakers (male->Ethan else Chelsie)
    "echo": "Ethan", "onyx": "Ethan", "ash": "Ethan", "ballad": "Ethan", "verse": "Ethan",
    "alloy": "Chelsie", "fable": "Chelsie", "nova": "Chelsie", "shimmer": "Chelsie",
    "coral": "Chelsie", "sage": "Chelsie"}


def _resolve_speaker(voice: str, speaker_map: dict) -> str:
    """Map a requested voice to an available speaker. Accepts our names (Chelsie/Ethan)
    directly, else maps OpenAI voice names, else defaults to Chelsie."""
    if voice in speaker_map:
        return voice
    m = _OPENAI_VOICE_MAP.get((voice or "").lower())
    if m and m in speaker_map:
        return m
    return "Chelsie" if "Chelsie" in speaker_map else next(iter(speaker_map))


def _encode_audio_response(wav, fmt: str):
    """Waveform tensor [-1,1] @24kHz -> (bytes, media_type). Native: wav (PCM16) + pcm (raw).
    Unsupported formats fall back to wav (clients generally accept it)."""
    import io
    import wave
    import numpy as np
    a = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    if (fmt or "").lower() == "pcm":
        return pcm.tobytes(), "audio/pcm"
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue(), "audio/wav"
