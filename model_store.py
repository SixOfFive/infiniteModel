#!/usr/bin/env python3
"""
InfiniteModel — controller model STORAGE + download/measure helpers (server-only leaf module).

Extracted from server.py (#38, step E) to shrink that file. These manage the controller's
on-disk model store (the single source of weights the workers fetch from): resolving/populating
``models/<org--name>/`` from the HF cache or a fresh pull, measuring real safetensors byte
breakdowns for the planner (MoE-correct), building a ModelSpec from a downloaded config.json,
the Ollama-style name normalization, and the HF-cache bookkeeping (size, purge, gc, delete) +
the ready/local-dir caches.

They are SELF-CONTAINED: stdlib + huggingface_hub + ``placement.ModelSpec`` + three pure
safetensors helpers from ``shards`` (``_weight_map`` / ``_text_prefix`` / ``_head_key``). The ONE
controller dependency — the HF read token — is supplied by DEPENDENCY INJECTION: server.py calls
``set_hf_token_provider(lambda: HF_TOKEN)`` once at import time, and the pull/HfApi helpers read
``_HF_TOKEN_FN()`` instead of importing it back (no back-import of server -> no import cycle).

This is a controller-only leaf module: it must NEVER ``import server``. It is listed in server.py's
EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync across the fleet, and server.py
imports its symbols back via a convergence-bridge import.

NOTE (what STAYED in server.py and why):
  * The whole DOWNLOAD-STATE group — the globals ``DOWNLOADING`` / ``DOWNLOAD_PROGRESS`` /
    ``DOWNLOAD_ERROR`` / ``DOWNLOAD_CONTROL`` / ``DOWNLOAD_STATE`` / ``DOWNLOAD_EPOCH`` /
    ``DOWNLOAD_STATE_PATH`` plus ``load_download_state`` / ``save_download_state`` and the
    interruptible pull ``_pull_repo_interruptible`` — were NOT moved. They are mutated IN-PLACE
    *and rebound* (``global DOWNLOAD_STATE``) by server.py's FastAPI download routes and read by
    the self-updater's idle lambda (``not DOWNLOADING``). A moved ``global`` rebind would bind to
    THIS module and silently decouple from server's name (the historical ``ENCODING`` hazard from
    the multimodal split — since Inc 11 ENCODING itself lives in media_encode.py WITH its mutators,
    the other valid pattern), so the DOWNLOAD-STATE group stays in server.py.
  * ``resolve_spec`` / ``resolve_model_name`` / ``_ollama_name`` / ``_split_family_size`` stay in
    server.py: they use the MODELS registry / MODEL_ALIASES (server globals). They call the moved
    ``_spec_from_config`` / ``_local_model_dir`` / ``_normalize_model_request`` / ``_friendly_from_hf``
    through server's convergence-bridge import.
"""
from __future__ import annotations

import contextlib
import dataclasses
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from typing import Optional

from placement import ModelSpec
# Three PURE safetensors-header helpers (no server state) live in shards.py; the storage
# helpers below use them to inspect a downloaded model's weight map / text prefix / head key.
from shards import _weight_map, _text_prefix, _head_key

GB = 1024 ** 3
# A model's training context falls back to this when its config.json lacks a usable
# max_position_embeddings (mirrors server.DEFAULT_CTX — a plain constant, not shared state).
DEFAULT_CTX = 8192


# ---------------------------------------------------------------------------
# Dependency injection: HF read token (no back-import of server)
# ---------------------------------------------------------------------------
# server.py injects its loaded HF_TOKEN here at import time so the authenticated pulls below
# can reach gated/rate-limited repos WITHOUT importing server (no import cycle). Until the
# setter runs, the provider returns None -> anonymous pulls (still work for open repos).
_HF_TOKEN_FN = lambda: None   # noqa: E731


def set_hf_token_provider(fn) -> None:
    """server.py injects ``lambda: HF_TOKEN`` here at import time so the authenticated pulls
    below read the live token WITHOUT importing server (no import cycle)."""
    global _HF_TOKEN_FN
    _HF_TOKEN_FN = fn


# GGUF source registry: a model whose weights ship ONLY as a llama.cpp .gguf file is normalized to
# safetensors once (see convert_gguf_to_model_dir). server.py injects a lookup so _controller_model_dir
# can recognize a GGUF target (the model_id == HF repo) and route it to conversion instead of the
# safetensors snapshot path. Returns the chosen .gguf filename for a GGUF model_id, else None.
_GGUF_FN = lambda _model_id: None   # noqa: E731


def set_gguf_provider(fn) -> None:
    """server.py injects ``lambda repo: GGUF_FILES.get(repo)`` so the acquisition path knows which
    targets are GGUF-sourced (and which quant file to fetch) WITHOUT importing server."""
    global _GGUF_FN
    _GGUF_FN = fn


def convert_gguf_to_model_dir(repo_id: str, gguf_file: str, model_id: str) -> str:
    """Normalize a GGUF model to a standard safetensors checkpoint under models/<name>/, ONCE.
    Idempotent: returns the existing dir if already converted. The heavy from_pretrained (which
    fully materializes the model in RAM) runs in a SUBPROCESS (gguf_convert.py) so it can OOM
    without taking down the controller box it co-hosts (the #never-full-load-on-controller lesson).
    After this, the model is an ordinary safetensors model — streamed, int4/int8-cached, and run
    on the pipeline with no GGUF awareness anywhere downstream."""
    local = os.path.join(MODELS_DIR, _safe_name(model_id))
    if _dir_has_model(local):
        return local
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_convert.py")
    if not os.path.exists(script):
        raise RuntimeError("gguf_convert.py not present on this node — self-update may not have synced it yet")
    env = dict(os.environ)
    _tok = _HF_TOKEN_FN()
    if _tok:
        env["HF_TOKEN"] = _tok          # pass via env, never argv (process listings leak args)
    os.makedirs(local, exist_ok=True)
    print(f"[model] GGUF -> safetensors: {repo_id} :: {gguf_file} (subprocess)", flush=True)
    proc = subprocess.run([sys.executable, script, repo_id, gguf_file, local],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        # a failed convert can leave a model dir with no usable tokenizer — wipe it so it isn't
        # mistaken for a complete model (and so a retry starts clean), regardless of _dir_has_model.
        shutil.rmtree(local, ignore_errors=True)
        raise RuntimeError(f"GGUF conversion failed ({repo_id} :: {gguf_file}): {tail}")
    # surface the converter's own progress lines (dep installs, which tokenizer path won) on success
    for _ln in (proc.stdout or "").splitlines()[-6:]:
        if _ln.strip():
            print(f"[gguf] {_ln.rstrip()}", flush=True)
    print(f"[model] GGUF conversion complete -> {local}", flush=True)
    return local


def _is_mxfp4_dir(d: str) -> bool:
    """True if the checkpoint at `d` is MXFP4-quantized (gpt-oss) — read from config.json."""
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
            qc = (json.load(fh) or {}).get("quantization_config") or {}
        return "mxfp4" in str(qc.get("quant_method", "")).lower()
    except Exception:
        return False


def convert_mxfp4_to_model_dir(src_dir: str, local: str, model_id: str) -> str:
    """Normalize an MXFP4 checkpoint (gpt-oss) to plain bf16 safetensors under models/<name>/, ONCE.
    The dequant runs in a SUBPROCESS (mxfp4_convert.py) that STREAMS one source file at a time, so it
    never materializes the full ~42 GB model in RAM — safe on a co-hosted box that is also serving
    (unlike a from_pretrained(dequantize)+save, which would OOM). After this it is an ordinary bf16
    model: chunk-streamed, run on the pipeline (transformers' real GptOss modules handle attention
    sinks + clamped SwiGLU natively). Mirrors convert_gguf_to_model_dir."""
    if _dir_has_model(local) and not _is_mxfp4_dir(local):
        return local                                 # already normalized
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp4_convert.py")
    if not os.path.exists(script):
        raise RuntimeError("mxfp4_convert.py not present on this node — self-update may not have synced it yet")
    env = dict(os.environ)
    _tok = _HF_TOKEN_FN()
    if _tok:
        env["HF_TOKEN"] = _tok          # pass via env, never argv
    os.makedirs(local, exist_ok=True)
    print(f"[model] MXFP4 -> bf16: {model_id} (streaming subprocess)", flush=True)
    proc = subprocess.run([sys.executable, script, src_dir, local],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        shutil.rmtree(local, ignore_errors=True)     # don't leave a half-converted dir
        raise RuntimeError(f"MXFP4 conversion failed ({model_id}): {tail}")
    for _ln in (proc.stdout or "").splitlines()[-6:]:
        if _ln.strip():
            print(f"[mxfp4] {_ln.rstrip()}", flush=True)
    print(f"[model] MXFP4 conversion complete -> {local}", flush=True)
    return local


# ---------------------------------------------------------------------------
# M2d: controller-side model storage
# ---------------------------------------------------------------------------
# The controller is the single source of model weights: it downloads the full model once and
# serves each worker only its layer tensors over HTTP, which the worker loads straight into RAM.
# Workers keep NO model on disk, so the smallest disk no longer caps model size — only the
# controller's disk does.
#
# Models live under <project>/models/<org--name>/ as real .safetensors + .json (no HF symlinks,
# so the controller never needs admin to write them, and you can browse them directly). Existing
# HF-cache downloads are migrated by MOVE (never re-downloaded).
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def _is_diffusers_dir(d: str) -> bool:
    """A diffusers multi-component checkpoint (model_index.json at the root): an image/video
    pipeline repo (e.g. Qwen-Image = transformer/ + text_encoder/ + vae/ + …), NOT a flat
    transformers LM. Weights live in component SUBFOLDERS, so every flat-layout assumption
    (root config.json, root safetensors) is wrong for these. (#t2i)"""
    return os.path.exists(os.path.join(d, "model_index.json"))


def _diffusers_complete(d: str) -> bool:
    """Completeness for a diffusers-layout dir (#t2i): every weight-bearing component subdir
    (marked by its config.json) holds at least one .safetensors; sharded components
    ('-0000X-of-0000N') have all N shards AND their index; every safetensors index that
    exists has all referenced files present. Conservative: any doubt -> incomplete, so a
    partial pull is never migrated/reported ready (mirrors the flat check's contract)."""
    try:
        found_weights = False
        for sub in sorted(os.listdir(d)):
            sd = os.path.join(d, sub)
            if not os.path.isdir(sd) or not os.path.exists(os.path.join(sd, "config.json")):
                continue                       # tokenizer/ (tokenizer_config) & scheduler/ carry no weights
            entries = os.listdir(sd)
            shards = [f for f in entries if f.endswith(".safetensors")]
            if not shards:
                return False                   # component config pulled but weights not yet
            found_weights = True
            sharded = [f for f in shards if re.search(r"-of-\d+", f)]
            # group by prefix so a dir holding several sharded weight SETS (e.g. an extra fp16
            # variant) counts each set against its own -of-N, not a pooled total
            groups: dict = {}
            for f in sharded:
                groups.setdefault(re.sub(r"-\d+-of-\d+", "", f), []).append(f)
            for fl in groups.values():
                m = re.search(r"-of-(\d+)", fl[0])
                if m and len(fl) != int(m.group(1)):
                    return False               # e.g. 3 of 9 transformer shards on disk so far
            idx = [f for f in entries if f.endswith(".safetensors.index.json")]
            if sharded and not idx:
                return False                   # shards present but index (ships last) not yet
            for f in idx:
                with open(os.path.join(sd, f), encoding="utf-8") as fh:
                    wm = json.load(fh).get("weight_map") or {}
                if not all(os.path.exists(os.path.join(sd, p)) for p in set(wm.values())):
                    return False
        return found_weights
    except Exception:
        return False


def _dir_has_model(d: str) -> bool:
    """True if dir holds a complete model: config.json + every shard in the index
    (flat transformers layout), or a complete diffusers component tree (#t2i)."""
    try:
        if _is_diffusers_dir(d):
            return _diffusers_complete(d)
        return (os.path.exists(os.path.join(d, "config.json"))
                and all(os.path.exists(p) for p in set(_weight_map(d).values())))
    except Exception:
        return False


def _ensure_template_files(model_id: str, local: str) -> None:
    """Top-up small template files (chat_template.jinja) that an earlier pull — which fetched only
    *.safetensors + *.json — missed, into an already-present model dir. Mistral3 / Devstral /
    Ministral ship their chat template as chat_template.jinja (NOT inside tokenizer_config.json), so
    without it AutoTokenizer loads NO template and rendering falls back to a flat prompt that breaks
    vision (the model never sees the native [INST][IMG]...[/INST]). Best-effort + silent: a model
    that has no such file in its repo (or an offline box) is simply left as-is."""
    if _is_diffusers_dir(local):
        return                       # image pipelines have no chat template (#t2i)
    aux = ["chat_template.jinja"]
    missing = [f for f in aux if not os.path.exists(os.path.join(local, f))]
    if not missing:
        return
    with contextlib.suppress(Exception):
        from huggingface_hub import hf_hub_download
        for f in missing:
            with contextlib.suppress(Exception):
                p = hf_hub_download(model_id, f, token=_HF_TOKEN_FN())
                shutil.copy2(p, os.path.join(local, f))
                print(f"[model] topped up {f} for {model_id}", flush=True)


def _controller_model_dir(model_id: str) -> str:
    """Resolve a model's dir under models/<name>/ (plain files). If already there,
    return it with ZERO network. Otherwise populate it ONCE — copying from the HF
    cache when present (no re-download; symlinks dereferenced to real files), else
    downloading — then it lives in models/ for good. Called on every /weights +
    /modelmeta fetch, so the already-present fast-path is just a dir check."""
    local = os.path.join(MODELS_DIR, _safe_name(model_id))
    if _dir_has_model(local):
        _ensure_template_files(model_id, local)   # self-heal chat_template.jinja missed by an
        return local                              # earlier *.safetensors+*.json-only pull
    # GGUF-sourced model: there are no safetensors to pull — normalize the .gguf to a safetensors
    # checkpoint into `local` (subprocess), then it's an ordinary model from here on.
    _gf = _GGUF_FN(model_id)
    if _gf:
        return convert_gguf_to_model_dir(model_id, _gf, model_id)
    from huggingface_hub import snapshot_download
    # *.jinja: chat_template.jinja (Mistral3 etc.); *.txt/*.model: tokenizer files (merges.txt,
    # sentencepiece) that diffusers repos keep under tokenizer/ (#t2i); *.py: trust_remote_code.
    patterns = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.py"]
    src = None
    try:
        cached = snapshot_download(model_id, allow_patterns=patterns, local_files_only=True)
        if _dir_has_model(cached):        # ONLY trust a COMPLETE cached snapshot —
            src = cached                  # a partial cache (prior aborted pull) would
    except Exception:                     # otherwise be copied incomplete and never
        src = None                        # become ready, with no error surfaced.
    if src is None:                       # nothing cached, or the cache is partial/broken
        src = snapshot_download(model_id, allow_patterns=patterns,
                                token=_HF_TOKEN_FN())   # authenticated pull (resumes partials)
    # MXFP4-quantized source (gpt-oss): normalize the MXFP4 experts to plain bf16 ONCE (bounded-memory
    # streaming subprocess) INTO `local` — then it's an ordinary bf16 model downstream. Mirrors GGUF.
    if _is_mxfp4_dir(src):
        return convert_mxfp4_to_model_dir(src, local, model_id)
    os.makedirs(local, exist_ok=True)
    # MOVE the cache blobs into models/ instead of COPYING them: on the same drive a move is
    # an instant rename — no 2x read+write (brutal on a USB/spinning disk), and no half-written
    # shard if interrupted. Across drives shutil.move falls back to copy+delete. The cache
    # snapshot holds symlinks -> blobs, so resolve realpath and move the blob; the now-dangling
    # snapshot is cleaned by the purge below. RECURSIVE (#t2i): diffusers repos keep their
    # weights in component subfolders (transformer/, text_encoder/, vae/, tokenizer/) — the
    # relative tree is preserved under models/<name>/ so the pipeline layout survives.
    _mig_ext = (".safetensors", ".json", ".jinja", ".txt", ".model", ".py")
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for fn in files:
            if not fn.endswith(_mig_ext):
                continue
            dst = os.path.join(local, fn) if rel == "." else os.path.join(local, rel, fn)
            real = os.path.realpath(os.path.join(root, fn))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                if not os.path.exists(real):
                    continue                   # blob already moved here on a prior run
                with contextlib.suppress(OSError):
                    os.remove(dst)             # replace a partial prior copy with the real blob
            if os.path.exists(real):
                shutil.move(real, dst)         # same-drive: instant rename; cross-drive: copy+del
    # Migration done: models/ is now the source of truth, so the HF-cache copy is a pure
    # duplicate (~2x the model on disk). Drop it once the copy is verified complete.
    if _dir_has_model(local):
        with contextlib.suppress(Exception):
            freed = _purge_hf_cache(model_id)
            if freed:
                print(f"[model] migrated {model_id} -> models/; freed "
                      f"{freed / GB:.1f} GB from the HF cache")
    return local


def _train_ctx_from_dir(model_dir: str, spec: "Optional[ModelSpec]" = None) -> int:
    """A model's native training context = its config.json max_position_embeddings.
    Used so a load defaults to the model's own context window (32k, 128k, 1M, …)
    instead of a fixed default. Falls back to the spec's max_ctx, then DEFAULT_CTX.
    NOTE: KV-cache memory scales with context, so a very large window reserves a lot
    of pool — the planner accounts for it and a load that can't fit fails cleanly."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            n = int(json.load(fh).get("max_position_embeddings") or 0)
        if n > 0:
            return n
    except Exception:
        pass
    return spec.max_ctx if spec else DEFAULT_CTX


_MEAS_CACHE: dict[str, Optional[dict]] = {}   # model_dir -> measured byte breakdown


def measure_model_weights(model_dir: str) -> Optional[dict]:
    """Measure REAL weight bytes per component by reading only the safetensors
    headers (fast — no tensor data). Generalises the planner to any architecture:
    MoE layers (router + N experts) are measured directly instead of guessed from a
    dense formula. Returns {layer_w_avg, embed, head, norm, total, n_layers} or None.
    Each tensor's size = data_offsets[1]-data_offsets[0] (exact on-disk bytes)."""
    hit = _MEAS_CACHE.get(model_dir)
    if hit is not None or model_dir in _MEAS_CACHE:
        return hit
    layer_bytes: dict[int, int] = {}
    embed = head = norm = 0
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        _MEAS_CACHE[model_dir] = None
        return None
    try:
        sizes: dict[str, int] = {}
        shapes: dict[str, list] = {}      # for the REAL param count (sum prod(shape)), not bytes/2
        dtypes: dict[str, str] = {}       # on-disk dtype per tensor (F32/BF16/F16/...) -> source dtype
        for f in files:
            with open(f, "rb") as fh:
                hlen = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(hlen))
            for name, meta in hdr.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                off = meta.get("data_offsets")
                if off and len(off) == 2:
                    sizes[name] = int(off[1]) - int(off[0])
                    shapes[name] = meta.get("shape") or []
                    dtypes[name] = meta.get("dtype") or ""
        # Scope to the TEXT decoder only. Composite checkpoints carry non-text towers with
        # their OWN '.layers.' (Qwen2.5-Omni: talker.model.layers, audio_tower.layers; the
        # Thinker text LM is thinker.model.layers) — counting those would inflate n_layers
        # and the per-layer size. _text_prefix picks the text root; everything else ignored.
        tp = _text_prefix(sizes)                 # 'thinker.model.' | 'model.language_model.' | 'model.'
        hk = _head_key(sizes, tp)
        params = 0                               # REAL param count of the served text LM (sum prod(shape))
        dcount: dict[str, int] = {}              # dtype histogram over those tensors -> dominant src dtype
        # NVFP4 (compressed-tensors) checkpoints store each quantized Linear as a uint8
        # '*.weight_packed' (2 FP4 codes/byte, shape [out, in//2]) + '*.weight_scale' (fp8) +
        # '*.weight_global_scale' (f32). The controller SERVES these DEQUANTIZED to bf16 and DROPS
        # the scale sidecars, so size the SERVED model: count weight_packed at its bf16-served size
        # (uint8 -> x4: 2 codes/byte * 2 bytes/elem) and SKIP the scales — else the planner sees the
        # ~4-bit on-disk size, for_quant() rescales an already-quantized number, and GPU placement
        # mis-plans (#96). Detected by any '*.weight_packed' present.
        is_packed = any(n.endswith(".weight_packed") for n in sizes)
        def _acc(nm: str) -> None:
            nonlocal params
            n = 1
            for s in (shapes.get(nm) or []):
                n *= s
            if is_packed and nm.endswith(".weight_packed"):
                n *= 2          # packed dim is in//2 -> real served param count doubles
            params += n
            dt = dtypes.get(nm)
            if dt:
                dcount[dt] = dcount.get(dt, 0) + 1
        for name, b in sizes.items():
            if is_packed:
                if "scale" in name.rsplit(".", 1)[-1]:
                    continue                     # nvfp4 scale sidecar — dropped at serve, exclude from sizing
                if name.endswith(".weight_packed"):
                    b *= 4                       # uint8 packed -> bf16-served bytes (x4)
            if name.startswith(f"{tp}layers."):
                try:
                    i = int(name.split(".layers.")[1].split(".")[0])
                except (ValueError, IndexError):
                    continue
                layer_bytes[i] = layer_bytes.get(i, 0) + b
                _acc(name)
            elif name == f"{tp}embed_tokens.weight":
                embed += b
                _acc(name)
            elif name == hk:
                head += b
                _acc(name)
            elif name == f"{tp}norm.weight":
                norm += b
                _acc(name)
    except Exception as exc:
        print(f"[measure] header read failed for {model_dir}: {exc!r}")
        _MEAS_CACHE[model_dir] = None
        return None
    if not layer_bytes:
        _MEAS_CACHE[model_dir] = None
        return None
    total_layer = sum(layer_bytes.values())
    out = {"layer_w_avg": total_layer // len(layer_bytes),
           "embed": embed, "head": head, "norm": norm,
           "total": total_layer + embed + head + norm,
           "n_layers": max(layer_bytes) + 1,
           "params": params,                                       # exact: sum prod(shape), dtype-agnostic
           # nvfp4 is SERVED as bf16 (dequant), so report BF16 as the served dtype regardless of the
           # on-disk U8/F8 packing (drives load_dtype + the size/quant display); else dominant on-disk.
           "dtype": ("BF16" if is_packed else (max(dcount, key=dcount.get) if dcount else None))}
    _MEAS_CACHE[model_dir] = out
    return out


def spec_with_measurements(spec: ModelSpec, model_dir: str) -> ModelSpec:
    """Return a copy of spec whose weight-byte properties reflect the model's real
    safetensors sizes (so MoE / any arch plans correctly). Falls back to spec if the
    files can't be measured."""
    m = measure_model_weights(model_dir)
    if not m:
        return spec
    return dataclasses.replace(spec, meas_layer_w=m["layer_w_avg"], meas_embed=m["embed"],
                               meas_head=m["head"], meas_norm=m["norm"],
                               meas_params=m.get("params"), src_dtype=m.get("dtype"))


_LOCAL_DIR_CACHE: dict = {}   # target_id -> local snapshot dir (only hits cached; cleared on delete)


def _local_model_dir(target_id: str):
    """Local snapshot dir for an ALREADY-downloaded model (models/ or the HF cache), or
    None. NEVER downloads (local_files_only) — safe to call from /status."""
    hit = _LOCAL_DIR_CACHE.get(target_id)
    if hit:
        return hit
    result = None
    local = os.path.join(MODELS_DIR, _safe_name(target_id))
    if _dir_has_model(local):
        result = local
    else:
        try:
            from huggingface_hub import snapshot_download
            d = snapshot_download(target_id,
                                  allow_patterns=["*.safetensors", "*.json", "*.jinja",
                                                  "*.txt", "*.model", "*.py"],
                                  local_files_only=True)
            if _dir_has_model(d):
                result = d
        except Exception:
            result = None
    if result:                       # cache only hits — the dir is stable once present
        _LOCAL_DIR_CACHE[target_id] = result
    return result


def _display_weight_bytes(target_id: str, spec: ModelSpec) -> int:
    """Weight bytes for DISPLAY/sizing: the REAL measured safetensors total once the model
    is downloaded (correct for MoE — the dense formula under-counts N experts, e.g. it
    estimates ~3.5 GB for the 66 GB Qwen3.6-35B-A3B), else the spec's formula estimate.
    Measurement is cached by dir (measure_model_weights / _MEAS_CACHE)."""
    d = _local_model_dir(target_id)
    if d:
        m = measure_model_weights(d)
        if m and m.get("total"):
            return int(m["total"])
    return spec.total_weight_bytes


def _tree_weight_bytes(d: str) -> int:
    """Recursive on-disk sum of every *.safetensors under a model dir. The size fallback for
    layouts measure_model_weights can't parse — diffusers repos (#t2i) keep weights in
    component subfolders, so the flat top-level glob sees nothing."""
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith(".safetensors"):
                with contextlib.suppress(OSError):
                    total += os.path.getsize(os.path.join(root, f))
    return total


def _friendly_from_hf(hf_id: str) -> str:
    """Derive a friendly REGISTRY KEY from an HF id: 'deepseek-ai/DeepSeek-R1-Distill-
    Llama-70B' -> 'deepseek-r1-distill-llama-70b'. The key stays in dash form (matching
    the built-ins + the on-disk custom_models.json) and is colon-free, so it is safe as a
    dict key, a URL query param, and a filename component. The Ollama 'family:size' display
    form is produced on demand by _ollama_name(); _normalize_model_request() bridges both
    forms back to this key, so 'qwen3:4b' and 'qwen3-4b' resolve to the same model."""
    base = hf_id.split("/")[-1].lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-._")
    return base or hf_id.lower().replace("/", "-")


def _normalize_model_request(name: str) -> str:
    """Collapse any equivalent client-facing form of a model name to its canonical
    dash-form registry key, so 'qwen3:4b', 'qwen3-4b', 'qwen3-4b:latest' and
    'qwen3:4b:latest' all map to 'qwen3-4b'. Lowercases, strips a trailing ':latest',
    then turns the size-boundary ':' into '-'. Does NOT validate existence (callers do)
    and leaves raw HF ids ('org/name') untouched."""
    n = (name or "").strip().lower()
    if "/" in n:                                   # a raw HF id — never rewrite it
        return n
    # strip a trailing ':latest' tag (possibly stacked after a size tag: 'qwen3:4b:latest')
    while n.endswith(":latest"):
        n = n[: -len(":latest")]
    if ":" not in n:
        return n
    head, _, tail = n.partition(":")               # 'qwen3' : '4b' (or '4b-instruct')
    return f"{head}-{tail}" if tail else head


def _spec_from_config(model_dir: str, name: str) -> Optional[ModelSpec]:
    """Build a ModelSpec from a downloaded model's config.json so arbitrary (user-added)
    models can be planned/placed. Standard transformer fields with fallbacks; multimodal
    composites use their text_config. The planner MEASURES real per-layer bytes at load,
    so this only needs the dims right (esp. layers/kv-heads/head_dim for KV math)."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            c = json.load(fh)
    except Exception:
        return None
    tc = c.get("text_config")
    if tc is None:                                 # Qwen2.5-Omni nests text dims deeper
        th = c.get("thinker_config")
        if isinstance(th, dict):
            tc = th.get("text_config") or th       # thinker_config.text_config (the Thinker LM)
    if isinstance(tc, dict):                        # composite/multimodal -> text dims win
        merged = dict(c); merged.update(tc); c = merged

    def gi(*keys, default=0):
        for k in keys:
            v = c.get(k)
            if isinstance(v, (int, float)) and v:
                return int(v)
        return default
    hidden = gi("hidden_size", "n_embd", "d_model")
    layers = gi("num_hidden_layers", "n_layer", "num_layers")
    heads = gi("num_attention_heads", "n_head", "num_heads")
    if not (hidden and layers and heads and gi("vocab_size")):
        return None
    kv = gi("num_key_value_heads", default=heads) or heads
    head_dim = gi("head_dim", default=(hidden // heads if heads else 0))
    inter = gi("intermediate_size", "moe_intermediate_size", "ffn_dim", "n_inner",
               default=4 * hidden)
    vocab = gi("vocab_size")
    max_ctx = gi("max_position_embeddings", "n_positions", "max_seq_len",
                 default=DEFAULT_CTX) or DEFAULT_CTX
    # Encoder/sentence-embedding detection (CONSERVATIVE): a known encoder model_type, OR an
    # architectures list that is ALL plain encoder *Model heads (no ForCausalLM / ForConditional
    # Generation / LMHead). A causal model like Qwen (Qwen2ForCausalLM) yields False.
    _archs = c.get("architectures") or []
    _mt = str(c.get("model_type") or "").lower()
    # A COMPOSITE generative multimodal checkpoint (Qwen2.5-Omni, a VLM, …) is NOT an encoder even
    # when its TOP-LEVEL architecture is a bare '*Model' name: Qwen2.5-Omni declares
    # architectures=["Qwen2_5OmniModel"], which otherwise trips the *Model heuristic below and mis-
    # flags it is_embedding -> the single-node AutoModel embedding load, which transformers CANNOT
    # build (Qwen2_5OmniConfig has no AutoModel mapping) -> the load hard-fails. Such configs nest a
    # generative sub-config (thinker/talker/token2wav/text_config) or a modality tower (vision/audio);
    # a flat encoder (BERT / nomic) nests NONE of these. Excluding composites keeps Omni on the
    # pipeline Thinker path (worker hand-builds Qwen2_5OmniThinkerTextModel), which /v1/audio/speech
    # then drives via the controller-side Talker + Token2Wav. (These markers survive the text_config
    # merge above: it only overwrites text-dim keys, never the sub-config keys themselves.)
    _composite = any(c.get(k) is not None for k in
                     ("thinker_config", "talker_config", "token2wav_config",
                      "text_config", "vision_config", "audio_config"))
    is_embedding = (not _composite
                    and (_mt in {"nomic_bert", "bert", "roberta", "xlm-roberta", "mpnet", "new"}
                         or (bool(_archs) and all(("ForCausalLM" not in a and "ForConditionalGeneration" not in a
                                                   and "LMHead" not in a) for a in _archs)
                             and any(a.endswith("Model") for a in _archs))))
    return ModelSpec(name, hidden, layers, heads, kv, head_dim, inter, vocab,
                     tie_embeddings=bool(c.get("tie_word_embeddings", False)),
                     arch=str(c.get("model_type") or "llama"),
                     attn_bias=bool(c.get("attention_bias", c.get("qkv_bias", False))),
                     max_ctx=max_ctx, is_embedding=is_embedding)


# ---------------------------------------------------------------------------
# HF-cache bookkeeping: size, ready-state, purge, gc, delete (no download state)
# ---------------------------------------------------------------------------
# The controller is the single source of weights and NEVER auto-purges; models are kept until
# explicitly deleted. Only models fully present on disk are reported as available.
_READY_CACHE: dict[str, tuple[float, bool]] = {}


def _hf_total_bytes(repo_id: str) -> int:
    """Total download size (safetensors + json) for a repo, from the HF API. 0 on failure
    (offline / gated / older hub) -> the dashboard then shows bytes-so-far without a %."""
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=_HF_TOKEN_FN()).model_info(repo_id, files_metadata=True)
        # extension set MUST mirror _pull_repo_interruptible's wanted list, or the progress
        # %/ETA denominator drifts from what is actually pulled
        return sum(int(s.size or 0) for s in (info.siblings or [])
                   if s.rfilename.endswith((".safetensors", ".json", ".jinja",
                                            ".txt", ".model", ".py")))
    except Exception:
        return 0


def _hf_cache_bytes(repo_id: str) -> int:
    """Bytes on disk in the HF cache for a repo right now (incl. .incomplete partials),
    so a download in flight reads as a growing number."""
    try:
        from huggingface_hub import constants
        base = os.path.join(constants.HF_HUB_CACHE,
                            "models--" + repo_id.replace("/", "--"))
    except Exception:
        return 0
    # Count only the canonical blobs/ store. snapshots/<rev>/ holds symlinks (Linux) or
    # copies (Windows, no symlink priv) of the SAME files, so walking the whole cache dir
    # and letting os.path.getsize follow the links double-counts every shard -> the
    # download bar overshoots 100% (seen: 85/67 GiB == 127%). blobs/ has each file exactly
    # once, including the .incomplete partials of an in-flight download.
    blobs = os.path.join(base, "blobs")
    walk_root = blobs if os.path.isdir(blobs) else base
    total = 0
    for root, _dirs, files in os.walk(walk_root):
        for f in files:
            p = os.path.join(root, f)
            try:
                if os.path.islink(p):
                    continue            # never follow a symlink's target (double-count guard)
                total += os.path.getsize(p)
            except OSError:
                pass
    return total


def model_ready(target_id: str, ttl: float = 3.0) -> bool:
    """True iff the model is fully on disk — in models/ OR still in the HF cache
    (a load migrates cache->models/). No network. Cached (the dashboard polls)."""
    now = time.time()
    hit = _READY_CACHE.get(target_id)
    if hit and now - hit[0] < ttl:
        return hit[1]
    ready = _dir_has_model(os.path.join(MODELS_DIR, _safe_name(target_id)))
    if not ready:
        try:
            from huggingface_hub import snapshot_download
            d = snapshot_download(target_id,
                                  allow_patterns=["*.safetensors", "*.json", "*.jinja",
                                                  "*.txt", "*.model", "*.py"],
                                  local_files_only=True)
            ready = _dir_has_model(d)
        except Exception:
            ready = False
    _READY_CACHE[target_id] = (now, ready)
    return ready


def _invalidate_ready_cache(target_id: str) -> None:
    _READY_CACHE.pop(target_id, None)
    _LOCAL_DIR_CACHE.pop(target_id, None)   # re-resolve the dir after a download/delete


def _purge_hf_cache(target_id: str) -> int:
    """Delete ONLY the HF-cache copy of a model (NOT the models/ copy). Returns bytes
    freed. Once a model is migrated to models/ (the source of truth the workers fetch
    from), its HF-cache copy is a pure duplicate; this reclaims that space."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        hashes = [rev.commit_hash for repo in cache.repos if repo.repo_id == target_id
                  for rev in repo.revisions]
        if not hashes:
            return 0
        strat = cache.delete_revisions(*hashes)
        freed = int(getattr(strat, "expected_freed_size", 0) or 0)
        strat.execute()
        return freed
    except Exception as exc:
        print(f"[cache] purge failed for {target_id}: {exc!r}")
        return 0


def gc_redundant_cache() -> dict:
    """Reclaim disk by deleting the HF-cache copy of every model that is ALSO complete
    in models/ (a pure duplicate). Models present ONLY in the cache (downloaded but
    never loaded → not yet migrated) are KEPT — deleting those would lose the only copy."""
    removed, freed = [], 0
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
    except Exception as exc:
        return {"ok": False, "error": str(exc), "removed": [], "freed_gb": 0.0}
    for repo in cache.repos:
        if getattr(repo, "repo_type", "model") != "model":
            continue
        local = os.path.join(MODELS_DIR, _safe_name(repo.repo_id))
        if _dir_has_model(local):                # migrated -> cache copy is redundant
            f = _purge_hf_cache(repo.repo_id)
            if f:
                freed += f
                removed.append({"model": repo.repo_id, "freed_gb": round(f / GB, 2)})
                _invalidate_ready_cache(repo.repo_id)
    return {"ok": True, "removed": removed, "freed_gb": round(freed / GB, 2)}


def delete_model_cache(target_id: str) -> bool:
    """Delete a model from the controller — the models/ folder AND any HF-cache
    copy. Returns True if anything was removed. Caller ensures it isn't loaded."""
    removed = False
    local = os.path.join(MODELS_DIR, _safe_name(target_id))
    if os.path.isdir(local):
        shutil.rmtree(local, ignore_errors=True)
        removed = True
    if _purge_hf_cache(target_id):
        removed = True
    _invalidate_ready_cache(target_id)
    return removed
