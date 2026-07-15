#!/usr/bin/env python3
"""
InfiniteModel — distributed LLM inference controller (server + dashboard + Ollama API).

A single controller machine that:
  - accepts worker nodes over a TCP control plane and tracks their CPU/RAM,
  - partitions a model across nodes' RAM (pipeline-parallel, RAM-weighted),
  - drives a networked decode pipeline and serves it through an Ollama-compatible
    HTTP API (so existing Ollama tooling/monitoring works against this port),
  - serves a live dashboard of the cluster.

Engine roadmap (README): pipeline-parallel to FIT, tensor-parallel to go FASTER,
speculative decoding as the reliable CPU speed lever.
  M1  — registry + heartbeat + dashboard
  M2a — RAM-weighted partition planner
  M2b — worker partial model load
  M2c — networked pipeline generation (this file) + full Ollama API
        (prefill-per-token; incremental KV-cache decode is M2d)

Run:
    python server.py                          # HTTP/Ollama :11434, control :50100
    python server.py --host 0.0.0.0 --http-port 11434 --control-port 50100
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional

# Keep the Hugging Face download cache INSIDE the project, not under the user account's
# ~/.cache (which we keep clean). MUST run before transformers/huggingface_hub are first
# imported — they read these env vars at import time. HF_HOME parents the hub/ cache, the
# token file, etc. Override with INFINITEMODEL_HF_HOME if you want it on another drive.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.environ.get("INFINITEMODEL_HF_HOME") or os.path.join(_PROJECT_DIR, "cache", "huggingface")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                   Response, StreamingResponse)
    from starlette.background import BackgroundTask
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing controller deps. Install with:\n"
        "    pip install fastapi uvicorn\n"
        f"(import error: {exc})"
    )

VERSION = "0.2-m4c168"  # version tag only; full changelog -> CHANGELOG.md
OLLAMA_API_VERSION = "0.5.4"   # version string reported on /api/version for tool compat
GB = 1024 ** 3


# Every console line is date/time-stamped so an unexpected event in the log can be
# correlated after the fact. Shadows the builtin print for THIS module only (uvicorn
# uses logging; workers have their own). log_activity()'s console echo and all the
# [load]/[+]/[!]/[load] FAILED lines pick this up automatically.
import builtins as _builtins
def print(*args, **kwargs):  # noqa: A001 — intentional builtin shadow for timestamping
    _builtins.print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)


# --- Self-update: poll GitHub for a newer server.py; when idle (no model loaded),
# swap it in and exit(42) so the supervisor (server.bat loop on Windows / systemd)
# relaunches the new code. Workers reconnect automatically. ---
import wire   # shared: cluster config (load_config) + self-update source URL. wire.py is a core file
             # present in every checkout and kept in sync via EXTRA_UPDATE_FILES.
SELF_UPDATE_POLL_S = 120   # poll the repo every 2 minutes (fast deploys; idle-gated)
SELF_UPDATE_FETCH_TRIES = 4      # #3: bounded retry per file within a cycle (CDN propagation lag on a
SELF_UPDATE_FETCH_BACKOFF_S = 8  # freshly-added module 404s on raw.githubusercontent until it syncs)


def _extract_version(blob: bytes) -> str:
    # #4: regex the `VERSION = "0.2-m4cNNN"` constant out of a fetched server.py/client.py so the
    # self-updater restarts ONLY on a real VERSION bump, not on any byte diff (doc/comment commits).
    import re
    try:
        m = re.search(rb'^VERSION\s*=\s*["\']([^"\']+)["\']', blob, re.MULTILINE)
        return m.group(1).decode("utf-8", "replace") if m else ""
    except Exception:
        return ""


def _ver_ordinal(v: str):
    """#no-downgrade: natural-sort key for VERSION tags ('0.2-m4c177') — digit runs compare
    NUMERICALLY, alpha runs lexicographically, so m4c9 < m4c176 (string compare gets this wrong)
    and m4bz < m4c0. Type-tagged tuples keep int/str comparisons well-defined at any divergence."""
    import re
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.findall(r"\d+|\D+", v or "")]


def _fetch_repo_file(fname: str):
    # Self-update fetches each file's latest bytes from the PUBLIC GitHub repo's raw endpoint
    # (wire.repo_raw_url, owner/branch from config.json) — NO auth/token, so no secret is in the source
    # (#public-release). Any failure -> returns None (fail-closed; the node keeps running current code).
    import urllib.request
    try:
        with urllib.request.urlopen(wire.repo_raw_url().format(f=fname), timeout=30) as r:
            return r.read()
    except Exception:
        return None


# Extra repo files (besides the primary entry point) to keep in sync on self-update. Extracted
# modules go here; a client+server SHARED module (wire.py) is listed in BOTH server.py + client.py.
EXTRA_UPDATE_FILES: list[str] = ["worker_t2i.py",   # #t2i-serve: worker diffusion engine
                                 "worker_tts.py",   # #tts-serve: worker Kokoro speech engine
                                 "worker_t2a.py",   # #t2a-serve: worker ACE-Step music engine
                                 "gptq_pack.py",       # #38: calibrated int2 compile (controller-only)
                                 "calib_corpus.txt",   # #38: its bundled calibration text
                                 "wire.py", "dashboard_html.py", "placement.py", "shards.py",
                                 "shard_compile.py",   # code-split Inc 9: SHARED compile/pack family
                                 "formats.py", "multimodal.py", "graphs.py", "model_store.py",
                                 "mtp_core.py",   # #91 MTP head forward (controller-only import)
                                 "gguf_convert.py",  # GGUF->safetensors converter (subprocess)
                                 "mxfp4_convert.py",  # MXFP4(gpt-oss)->bf16 converter (subprocess)
                                 "kv_quant.py",       # TurboQuant KV-cache quantizer (#172)
                                 # m4c152 code-split: shared-state registry + relocated Engine mixins
                                 "state.py", "engine_load.py", "engine_gen.py", "engine_lifecycle.py",
                                 "control_plane.py",   # code-split Inc 2: worker-facing control plane
                                 # m4c153 code-split: relocated build_app routes
                                 "routes_dashboard.py", "routes_lifecycle.py", "routes_api.py", "routes_diag.py",
                                 "routes_shards.py",   # code-split Inc 6: shard/pack/weights routes
                                 "serving.py", "status.py",   # m4c154/155 code-split: serving + status-building layers
                                 "serving_anthropic.py",   # code-split Inc 3: Anthropic Messages engine
                                 "downloads.py",   # code-split Inc 5: download/registry lifecycle
                                 "media_encode.py",   # code-split Inc 11: encode/speech family + ENCODING home
                                 "config.json"]   # central cluster config — synced like a module


def _code_date() -> str:
    """Newest mtime across the running build's self-update file set, as YYYY-MM-DD — the
    honest "when did this controller's code last change" stamp for the dashboard header
    (the VERSION string is a false tell: controller-only changes never bump it). Computed
    ONCE at import on purpose: the idle self-updater rewrites files on disk WITHOUT
    restarting for controller-only fetches, so a live mtime read would date code NEWER
    than what this process is actually running."""
    base = os.path.dirname(os.path.abspath(__file__))
    ts = []
    for f in ["server.py"] + EXTRA_UPDATE_FILES:
        with contextlib.suppress(Exception):
            ts.append(os.path.getmtime(os.path.join(base, f)))
    return time.strftime("%Y-%m-%d", time.localtime(max(ts))) if ts else ""


CODE_DATE = _code_date()


def _self_update_check(fname: str, is_idle, force: bool = False) -> None:
    """Multi-file self-update: fetch the primary file + EXTRA_UPDATE_FILES, and if ANY changed
    (and we're idle, OR force=True) stage ALL changed files together. RESTART only when the fetched
    primary-file VERSION differs from the running VERSION (#4: a same-VERSION doc/comment commit must
    NOT bounce the fleet) — a forced update always restarts. Each fetch is bounded-retried with
    backoff so a CDN-propagation 404 on a freshly-added file (#3) gets time to sync; if a file STILL
    won't fetch, abort THIS cycle (never apply a half-updated set) and retry next poll. force=True is
    the dashboard/API 'Update' button: swap NOW without waiting for idle (the caller has already
    unloaded models + told workers to free RAM)."""
    here = os.path.dirname(os.path.abspath(__file__))
    files = [fname] + [f for f in EXTRA_UPDATE_FILES if f != fname]
    fetched: dict = {}
    for fn in files:
        remote = None
        for attempt in range(SELF_UPDATE_FETCH_TRIES):   # #3: retry — give the raw CDN time to propagate
            remote = _fetch_repo_file(fn)
            if remote is not None and len(remote) >= 5:
                break
            if attempt + 1 < SELF_UPDATE_FETCH_TRIES:
                time.sleep(SELF_UPDATE_FETCH_BACKOFF_S * (attempt + 1))  # runs in a thread; safe to block
        if remote is None or len(remote) < 5:    # still failing -> abort THIS cycle (stay consistent)
            print(f"[update] {fn} not fetchable (404/transient on raw CDN) - aborting cycle, retry next poll")
            return
        fetched[fn] = remote
    changed = []
    for fn, remote in fetched.items():
        path = os.path.join(here, fn)
        try:
            with open(path, "rb") as fh:
                local = fh.read()
        except FileNotFoundError:
            local = b""                          # a not-yet-present module counts as changed
        except Exception:
            return
        if remote.replace(b"\r\n", b"\n") != local.replace(b"\r\n", b"\n"):
            changed.append(fn)
    if not changed:
        return
    if not force and not is_idle():
        print(f"[update] {changed} newer on repo - deferring (load/download/encode in progress)")
        return
    # #4: only RESTART on a VERSION bump in the primary file. Stage same-VERSION content changes to disk
    # (atomic, picked up on the next natural restart) but don't bounce the fleet for a doc/comment commit.
    remote_ver = _extract_version(fetched.get(fname, b""))
    # #no-downgrade: after a git push the raw CDN lags per-file by 1-5 min, so a box running code
    # AHEAD of the CDN (pscp deploy, or restarting mid-propagation) used to see "content differs" ->
    # overwrite its files with the STALE repo copy and restart into a DOWNGRADE (live hit: worker
    # m4c177 -> m4c176 minutes after the m4c177 push). The AUTOMATIC loop refuses to apply a remote
    # whose primary VERSION is strictly OLDER than the running one (skips the writes too, so stale
    # extras never land on disk); a FORCED /update (explicit operator intent — e.g. a deliberate
    # rollback commit) still applies it, loudly labeled.
    if remote_ver and _ver_ordinal(remote_ver) < _ver_ordinal(VERSION):
        if not force:
            print(f"[update] repo VERSION {remote_ver} is OLDER than running {VERSION} - "
                  f"ignoring (CDN lag / local ahead); re-check next poll")
            return
        print(f"[update] FORCED DOWNGRADE {VERSION} -> {remote_ver} (operator /update)")
    version_bumped = bool(remote_ver) and remote_ver != VERSION
    for fn in changed:                           # write all .new first, then atomic-replace each
        path = os.path.join(here, fn)
        tmp = path + ".new"
        with open(tmp, "wb") as fh:
            fh.write(fetched[fn])
        os.replace(tmp, path)
    if not force and not version_bumped:
        print(f"[update] {changed} staged on disk (VERSION {VERSION} unchanged) - NOT restarting (#4)")
        return
    print(f"[update] {changed} newer on repo (VERSION {VERSION} -> {remote_ver or '?'}) - restarting")
    os._exit(42)                                 # supervisor relaunches on the new code


async def _self_update_loop(fname: str, is_idle) -> None:
    while True:
        await asyncio.sleep(SELF_UPDATE_POLL_S)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(_self_update_check, fname, is_idle)


# ---------------------------------------------------------------------------
# Hugging Face auth — load a read token so model pulls are authenticated
# ---------------------------------------------------------------------------
def _load_hf_token() -> Optional[str]:
    """Return a Hugging Face access token, checking (in order) the HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN env vars, then a gitignored ``hf_token.txt`` beside this
    file. Export it back into the environment so every huggingface_hub call
    (snapshot_download, HfApi, ...) authenticates: anonymous pulls get rate-limited
    and cannot reach gated repos. The token lives only in the env or the gitignored
    file — never in the tree."""
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_token.txt")
        try:
            with open(path, encoding="utf-8") as fh:
                tok = fh.read().strip()
        except OSError:
            tok = ""
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    return tok or None


HF_TOKEN = _load_hf_token()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Friendly name -> (target HF id, draft HF id for speculative decoding).
# The draft and target MUST share the same tokenizer/vocab => same model family.
MODELS: dict[str, tuple[str, str]] = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-coder-32b": (
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ),
    # Llama arch (dense). Draft == target -> no speculative decode (no small
    # llama draft registered; spec needs a same-tokenizer draft anyway).
    "nemotron-70b": (
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    ),
    # Llama-3.3-70B arch (dense) — R1 reasoning distilled onto Llama-3.3-70B-Instruct.
    # Standard attention, so it runs on the existing pipeline (unlike the hybrid 35B).
    "deepseek-r1-distill-llama-70b": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ),
    # Llama-3.3-70B-Instruct (dense), via the unsloth mirror (the official meta-llama repo is
    # GATED and the fleet's HF account isn't on its authorized list). Draft = Llama-3.2-1B
    # (SAME 128256-token Llama-3 vocab/tokenizer) -> the load auto-attaches it controller-side
    # and greedy requests with options.speculative=true get draft-K/verify-once decode — the
    # one lever that beats the bandwidth ceiling on a 37.8 GB/token dense sweep (#spec).
    # Promoted to a built-in PAIR because custom /add_model entries register draft==target
    # (spec off); the built-in is seeded before custom_models.json merges, so it wins.
    "llama-3.3-70b": (
        "unsloth/Llama-3.3-70B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct",   # 3B: 1B accept measured 42% (parity); 3B lifts it
    ),
    # MoE (Mixtral 8x7B): ~47B params resident (~94 GB bf16) but only ~13B active per
    # token -> ideal for the RAM-rich fleet. Per-layer expert bytes are MEASURED from
    # the safetensors headers at load (the dense formula can't see the 8 experts).
    "mixtral-8x7b": (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
    # MoE (OLMoE 1B-7B): fully OPEN (Apache-2.0, allenai — not gated, unlike Mixtral), 64
    # experts (8 active), ~14 GB bf16. The open alternative for exercising the measured
    # per-layer expert sizing without the gated-download hassle.
    "olmoe-1b-7b": (
        "allenai/OLMoE-1B-7B-0924-Instruct",
        "allenai/OLMoE-1B-7B-0924-Instruct",
    ),
    # MoE (Qwen3.6-35B-A3B): 256 experts / 8 active, ~67 GB bf16, native ctx 262144. MULTIMODAL
    # checkpoint — its TEXT weights are nested under model.language_model.* (+ model.visual.* /
    # mtp.* we ignore); the client remaps language_model.* -> model.* at load. Big -> spills
    # across the fleet (slow decode); ALWAYS load with an explicit small ctx (256K KV is huge).
    "qwen3.6-35b-a3b": (
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B",
    ),
    # Encoder / sentence-embedding (nomic_bert, ~140M params). Served by the single-node
    # embedding path (no pipeline/TP/KV/lm_head); draft "" since there's no decode.
    "nomic-embed-text": ("nomic-ai/nomic-embed-text-v1.5", ""),
}

HEARTBEAT_TIMEOUT_S = 45.0   # generous: building a multi-GB shard (mmap+fuse+quantize) can
                             # block a worker's heartbeat for tens of seconds; 15s falsely
                             # reaped busy workers mid-load -> spurious model reloads.
# A node actively SERVING a shard gets a much longer grace before being reaped (#47): under
# heavy load (e.g. a big download on the controller box + CPU inference) a serving worker can
# miss the 45s heartbeat without being dead, and a false reap tears the model down and churns a
# reload (leaking the old shard's RAM). Only reap a serving node if it's been silent this long.
SERVING_GRACE_S = 180.0
REAPER_INTERVAL_S = 3.0
GEN_TIMEOUT_S = 600.0   # max wait for ONE token's logits before failing fast. Generous so a slow
                        # CPU big-model prefill/first-token (e.g. 70B int4 on CPU, minutes to first
                        # token) completes instead of tripping a false TimeoutError mid-generation.
GEN_STALL_S = 240.0     # #gen-stall-watchdog: a model active>0 that has produced NO token for this
                        # long is WEDGED (dead pipeline hop -> 0 tokens + idle data plane). The
                        # watchdog cancels its in-flight request(s) + reclaims the leaked active slot.
                        # > worst-case legit first-token wait (big CPU prefill) so it never false-fires.
GEN_STALL_DECODE_S = 60.0  # #active-decode-stall: a SHORTER stall threshold that applies ONLY once a gen
                        # has produced its first token (it's DECODING, not in cold prefill). A decoding
                        # gen that goes silent this long = a wedged mid-pipeline hop (e.g. the buffered-
                        # write deadlock the hop_error channel can't catch). Cold prefill keeps GEN_STALL_S;
                        # 60s >> any healthy per-token decode time (even heavy CPU spill), so no false-fire.
                        # 0 disables it (fall back to GEN_STALL_S for decode too).
REQUEUE_GRACE_S = 12.0  # #stage0-stale-reconnect: when a head's return data-conn closes, wait this
                        # long before dooming the requests it was serving — a head that just FRESHENED
                        # its return socket (drop+reconnect at prefill) re-delivers their logits on the
                        # new conn within its stage compute; only a genuinely dead head stays pending.
                        # #70b-long-prefill: this is now the SLICE of a progress-extended loop — grace
                        # keeps renewing while the owning model reports advancing per-layer forward
                        # progress (see engine_lifecycle._grace_doom), so a multi-minute big-model
                        # prefill is never doomed mid-flight by a flat timer.
STAGE0_STALE_S = 5.0    # #stage0-stale-reconnect: if the controller hasn't pushed a frame to a model's
                        # stage 0 for this long, its stage0_writer may have gone silently half-open while
                        # idle -> rebuild it FRESH before the next generation's prefill (a connect is ~ms;
                        # the alternative is a ~600s GEN_TIMEOUT hang the watchdog only papers over). Short
                        # enough to catch idle-between-requests; only checked at generate START (never
                        # per decode token), so a slow model's multi-second inter-token gaps never trip it.
SPEC_K = 4              # speculative decode: draft this many tokens per verify

# --- Per-node tier config (persisted): enable/disable each node's CPU/RAM and
# GPU/VRAM contribution. Keyed by HOSTNAME (stable across reconnects). Both tiers
# default to enabled; survives controller restarts via node_config.json.
NODE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_config.json")
NODE_CONFIG: dict[str, dict] = {}


def load_node_config() -> None:
    # In-place (NEVER rebind; code-split Inc 4): main() runs state.publish() BEFORE this loader,
    # so a rebind would strand every bound leaf module on the pre-load EMPTY dict (see state.py;
    # m4c155 DOWNLOAD_STATE precedent). clear()+update() keeps the published identity live.
    try:
        with open(NODE_CONFIG_PATH, encoding="utf-8") as fh:
            _cfg = json.load(fh)
    except Exception:
        _cfg = {}
    NODE_CONFIG.clear()
    NODE_CONFIG.update(_cfg)


def save_node_config() -> None:
    try:
        with open(NODE_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(NODE_CONFIG, fh, indent=2)
    except Exception as exc:
        print(f"[cfg] could not save node config: {exc!r}")


# --- User-added models (persisted): friendly -> HF id. Entered in the dashboard
# ("download any HF model"); merged into MODELS at startup so they behave exactly
# like the built-ins (list, download, load). Spec is built from config.json on demand.
CUSTOM_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_models.json")
CUSTOM_MODELS: dict[str, str] = {}
# GGUF-sourced models: HF target (repo) -> the chosen single .gguf filename. A model in here has no
# safetensors upstream; its weights are normalized to safetensors ONCE at acquisition (model_store.
# convert_gguf_to_model_dir) and it behaves like any other model thereafter. Persisted separately so
# the existing custom_models.json format (friendly->repo) is untouched. Keyed by the TARGET repo
# because that's what _controller_model_dir (the acquisition point) receives.
GGUF_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_gguf.json")
GGUF_FILES: dict[str, str] = {}


def load_custom_models() -> None:
    # In-place (NEVER rebind; code-split Inc 4) — same rationale as load_node_config: the
    # published CUSTOM_MODELS/GGUF_FILES identities must stay live for bound leaf modules.
    try:
        with open(CUSTOM_MODELS_PATH, encoding="utf-8") as fh:
            _cm = json.load(fh)
    except Exception:
        _cm = {}
    CUSTOM_MODELS.clear()
    CUSTOM_MODELS.update(_cm)
    for friendly, hf in CUSTOM_MODELS.items():
        MODELS.setdefault(friendly, (hf, hf))   # draft = target (no speculative)
    try:
        with open(GGUF_MODELS_PATH, encoding="utf-8") as fh:
            _gf = json.load(fh)
    except Exception:
        _gf = {}
    GGUF_FILES.clear()
    GGUF_FILES.update(_gf)


def save_custom_models() -> None:
    with contextlib.suppress(Exception):
        with open(GGUF_MODELS_PATH, "w", encoding="utf-8") as fh:
            json.dump(GGUF_FILES, fh, indent=2)   # keep the GGUF source map in lockstep
    try:
        with open(CUSTOM_MODELS_PATH, "w", encoding="utf-8") as fh:
            json.dump(CUSTOM_MODELS, fh, indent=2)
    except Exception as exc:
        print(f"[cfg] could not save custom models: {exc!r}")


# Models the user has DELETED (a full removal: cache + registry + aliases). For CUSTOM models the
# removal already persists by dropping them from custom_models.json — but a BUILT-IN re-seeds into
# MODELS from the literal above on every startup, so to keep a deleted built-in OUT of the list
# (no stale "download" button for a model the user removed) we persist its friendly name here and
# filter it after MODELS is seeded. Re-adding via /add_model un-hides it.
DELETED_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deleted_models.json")
DELETED_MODELS: set[str] = set()


def load_deleted_models() -> None:
    """Load the deleted-model hide-set and drop any matching entries from MODELS. MUST run AFTER
    MODELS is seeded with built-ins AND load_custom_models() has merged customs, so a hidden name
    is filtered no matter which source re-introduced it."""
    # In-place (NEVER rebind; code-split Inc 4) — published set identity stays live for bound leaves.
    try:
        with open(DELETED_MODELS_PATH, encoding="utf-8") as fh:
            _dm = set(json.load(fh))
    except Exception:
        _dm = set()
    DELETED_MODELS.clear()
    DELETED_MODELS.update(_dm)
    for friendly in DELETED_MODELS:
        MODELS.pop(friendly, None)


def save_deleted_models() -> None:
    with contextlib.suppress(Exception):
        with open(DELETED_MODELS_PATH, "w", encoding="utf-8") as fh:
            json.dump(sorted(DELETED_MODELS), fh, indent=2)


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------

@dataclass
class Node:
    node_id: str
    hostname: str
    os: str
    device: str
    device_name: str
    total_mem_gb: float
    usable_mem_gb: float
    data_host: str
    data_port: int
    connected_at: float
    last_heartbeat: float
    free_mem_gb: float = 0.0
    free_disk_gb: float = 0.0
    cpu_percent: float = 0.0
    proc_rss_gb: float = 0.0   # RAM this worker's python process holds (heartbeat); for
    #                            the "engine memory" (red) split on the pool bar
    net_in_bps: float = 0.0
    net_out_bps: float = 0.0
    peer_bytes: dict = field(default_factory=dict)  # worker-reported per-peer data bytes (bandwidth)
    client_version: str = ""
    wire: bool = False   # worker imported wire.py (vs the inline fallback) — code-split verify
    ram: str = ""   # e.g. "2x DDR4-2666" / "4x LPDDR5-5500"
    # False once a node proves it can't run a stage (e.g. no torch installed) —
    # such a node is kept visible but excluded from load planning so one
    # half-provisioned box can't break every load. Reset on reconnect.
    can_infer: bool = True
    incapable_reason: str = ""
    # GPU memory (worker-reported; the controller can't see a worker's VRAM).
    # Non-zero only when the worker runs on a GPU (--device gpu/cpu+gpu).
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    # #vram-reusable: the worker process's VACANT torch allocator pool (reserved - allocated).
    # Device counters (mem_get_info / GTT) report it as USED, but any new torch allocation in
    # that worker reuses it first — so it is FREE for the planner's purposes (a fresh load lands
    # into it). Only fully returns to the OS on worker restart (ROCm pool fragmentation).
    vram_reusable_gb: float = 0.0
    gpu_util: float = 0.0      # GPU compute utilization % (worker heartbeat, GPU nodes only)
    cores: int = 0             # logical CPU cores (registration) — capacity weight for load
    # --- pipeline assignment (set on load); reserved tp_* for the M4 grid ---
    stage: Optional[int] = None
    tp_rank: Optional[int] = None
    tp_size: int = 1
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    shard_gpu_bytes: int = 0   # bytes this node's loaded stage placed on its GPU
    load_state: str = "idle"   # "idle"|"loading"|"ready" — per-shard load progress (dashboard red->green)

    @property
    def age(self) -> float:
        return time.time() - self.last_heartbeat

    @property
    def alive(self) -> bool:
        return self.age <= HEARTBEAT_TIMEOUT_S

    @property
    def usable_vram_gb(self) -> float:
        """Raw VRAM available to hold weights (total minus CUDA-context/display reserve)."""
        return max(0.0, self.vram_total_gb - VRAM_RESERVE_GB) if self.vram_total_gb > 0 else 0.0

    @property
    def ram_enabled(self) -> bool:
        return NODE_CONFIG.get(self.hostname, {}).get("ram", True)

    @property
    def vram_enabled(self) -> bool:
        return NODE_CONFIG.get(self.hostname, {}).get("vram", True)

    @property
    def eff_ram_gb(self) -> float:
        # Budget by what's ACTUALLY free/committable now (free + reclaimable cache, from the
        # latest heartbeat) rather than total RAM. Sizing by total over-commits on a busy box
        # -> Windows paging-file / commit-limit error 1455 at load. Capped at total-reserve;
        # never below 0. RAM_SAFETY_GB leaves headroom for the load transient + heartbeat drift.
        if not self.ram_enabled:
            return 0.0
        # Adaptive safety: a flat RAM_SAFETY_GB bigger than a tiny node's whole free RAM (e.g. a
        # 4 GB Android tablet with ~2 GB free) would clamp eff_ram to 0 and bar it from EVERY
        # load. Reserve the smaller of RAM_SAFETY_GB or ~40% of free RAM — so small nodes still
        # offer a usable slice, while any node with >= ~7.5 GB free keeps the full flat margin
        # (free*0.4 >= RAM_SAFETY_GB there, so min() picks the flat 3 GB; behaviour unchanged).
        reserve = min(RAM_SAFETY_GB, self.free_mem_gb * 0.4)
        free_budget = self.free_mem_gb - reserve
        return max(0.0, min(self.usable_mem_gb, free_budget))

    @property
    def eff_vram_gb(self) -> float:
        return self.usable_vram_gb if self.vram_enabled else 0.0

    def free_vram_after_resident_gb(self, committed_vram_bytes: int) -> float:
        """VRAM still placeable here after the bytes already committed to RESIDENT models'
        shards on this node — used for node-sharing so a 2nd model is budgeted against the
        VRAM left, not the empty-GPU figure. (RAM needs no equivalent: eff_ram_gb is already
        live via free_mem_gb, which has dropped for resident models.)"""
        return max(0.0, self.eff_vram_gb - committed_vram_bytes / GB)

    @property
    def usable_total_gb(self) -> float:
        """Memory the controller will actually use here = enabled RAM + enabled VRAM.
        A cpu+gpu worker holds GPU layers in VRAM and spills the rest to RAM; a
        tier toggled off in the dashboard contributes 0 (and is excluded if both
        are off)."""
        return self.eff_ram_gb + self.eff_vram_gb

    def load_device(self) -> str:
        """Device directive sent to this node's worker, from its tier config.
        Both tiers on -> "" (NO override: the worker uses its own --device default,
        e.g. cpu+gpu); GPU-only (RAM off) -> 'gpu'; VRAM off -> 'cpu'. Only an
        explicitly-disabled tier forces the device."""
        if self.vram_enabled and self.ram_enabled:
            return ""
        if self.vram_enabled and not self.ram_enabled:
            return "gpu"
        return "cpu"

    def clear_assignment(self) -> None:
        self.stage = self.tp_rank = self.layer_start = self.layer_end = None
        self.tp_size = 1
        self.shard_gpu_bytes = 0
        self.load_state = "idle"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "hostname": self.hostname, "os": self.os,
            "device": self.device, "device_name": self.device_name,
            "total_mem_gb": round(self.total_mem_gb, 2),
            "usable_mem_gb": round(self.usable_mem_gb, 2),
            "usable_total_gb": round(self.usable_total_gb, 2),  # RAM + usable VRAM
            "free_mem_gb": round(self.free_mem_gb, 2),
            "free_disk_gb": round(self.free_disk_gb, 2),
            "cpu_percent": round(self.cpu_percent, 1),
            "proc_rss_gb": round(self.proc_rss_gb, 2),
            "net_in_bps": round(self.net_in_bps),
            "net_out_bps": round(self.net_out_bps),
            "ram": self.ram,
            "data_host": self.data_host, "data_port": self.data_port,
            "client_version": self.client_version, "wire": self.wire,
            "age_s": round(self.age, 1), "alive": self.alive,
            "can_infer": self.can_infer, "incapable_reason": self.incapable_reason,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_used_gb": round(self.vram_used_gb, 2),
            "vram_reusable_gb": round(self.vram_reusable_gb, 2),   # #vram-reusable: vacant pool
            "gpu_util": round(self.gpu_util, 1),
            "cores": self.cores,
            "usable_vram_gb": round(self.usable_vram_gb, 2),
            "has_gpu": self.vram_total_gb > 0,
            "ram_enabled": self.ram_enabled, "vram_enabled": self.vram_enabled,
            "stage": self.stage, "tp_rank": self.tp_rank, "tp_size": self.tp_size,
            "layer_start": self.layer_start, "layer_end": self.layer_end,
            "load_state": self.load_state,
        }


class Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._lock = asyncio.Lock()
        self._counter = 0
        self.dirty = False  # membership changed -> a loaded model must re-plan

    async def add(self, reg: dict, peer_host: str) -> Node:
        async with self._lock:
            self._counter += 1
            node_id = f"n{self._counter}"
            now = time.time()
            node = Node(
                node_id=node_id, hostname=reg.get("hostname", "?"),
                os=reg.get("os", "?"), device=reg.get("device", "cpu"),
                device_name=reg.get("device_name", ""),
                total_mem_gb=float(reg.get("total_mem_gb", 0.0)),
                usable_mem_gb=float(reg.get("usable_mem_gb", 0.0)),
                data_host=reg.get("data_host") or peer_host,
                data_port=int(reg.get("data_port", 0)),
                connected_at=now, last_heartbeat=now,
                free_mem_gb=float(reg.get("usable_mem_gb", 0.0)),
                free_disk_gb=float(reg.get("free_disk_gb", 0.0)),
                client_version=reg.get("client_version", ""),
                wire=bool(reg.get("wire", False)),
                ram=reg.get("ram", ""),
                vram_total_gb=float(reg.get("vram_total_gb", 0.0)),
                cores=int(reg.get("cores", 0)),
            )
            self._nodes[node_id] = node
            # NOTE: a node JOINING is just added capacity — it must NOT force resident models
            # to reload (that caused a worker reconnect/flap to re-stream the whole 35B). New
            # nodes are simply available for the NEXT load. Only node loss reloads, and only
            # the models that used the lost node (via invalidate_model on remove/reap).
            return node

    async def find_stale_dupes(self, fresh: Node) -> list[Node]:
        """Prior node entries that are the SAME physical worker as the just-registered `fresh`
        — i.e. the worker RESTARTED and re-registered before its old control link was noticed
        dropped (idle Windows sockets can go half-open silently; the reaper is up to
        SERVING_GRACE_S=180s away). Identity = the worker's data endpoint (data_host:data_port,
        how the controller dials it), which a restart re-uses, plus a matching hostname. Excludes
        `fresh` itself. A new node simply JOINING (different machine) won't match, so a healthy
        node is never flagged."""
        async with self._lock:
            return [n for nid, n in self._nodes.items()
                    if nid != fresh.node_id
                    and n.hostname == fresh.hostname
                    and n.data_host == fresh.data_host
                    and n.data_port == fresh.data_port
                    and fresh.data_port != 0]

    async def remove(self, node_id: str) -> None:
        async with self._lock:
            if self._nodes.pop(node_id, None) is not None:
                self.dirty = True

    async def heartbeat(self, node_id: str, free_mem_gb: float, cpu_percent: float,
                        free_disk_gb: float = 0.0) -> bool:
        """Returns False when node_id is no longer registered (reaped while its socket
        stayed up) so the control handler can drop the link and force a re-register."""
        async with self._lock:
            n = self._nodes.get(node_id)
            if n:
                n.last_heartbeat = time.time()
                n.free_mem_gb = free_mem_gb
                n.cpu_percent = cpu_percent
                if free_disk_gb:
                    n.free_disk_gb = free_disk_gb
            return n is not None

    async def reap_dead(self) -> list[Node]:
        async with self._lock:
            # A node holding a model shard (n.stage set) gets SERVING_GRACE_S before reaping —
            # a busy serving box can miss the 45s heartbeat without being dead, and a false reap
            # churns a model reload (and leaks the old shard's RAM). Idle nodes use the short
            # timeout. (#47)
            # ALSO grant the long grace to a node that is the target of an IN-FLIGHT load (from
            # engine.loadings) — a node-shared box building a shard can have its n.stage scalar
            # transiently cleared by a CO-RESIDENT model's unload during the released-lock gather
            # (node-sharing makes these scalars last-writer-wins), and a quiet-but-healthy builder
            # must not be falsely reaped then (#parallel-load shared-node reaper edge).
            _eng = globals().get("engine")
            loading_nodes: set = set()
            if _eng is not None:
                for _c in getattr(_eng, "loadings", {}).values():
                    loading_nodes.update(_c.get("node_ids") or [])
            def _reapable(n: "Node") -> bool:
                grace = (SERVING_GRACE_S if (n.stage is not None or n.node_id in loading_nodes)
                         else HEARTBEAT_TIMEOUT_S)
                return n.age > grace
            dead = [n for nid, n in list(self._nodes.items()) if _reapable(n)]
            for n in dead:
                del self._nodes[n.node_id]
            if dead:
                self.dirty = True
            return dead

    def alive_sorted(self) -> list[Node]:
        return sorted((n for n in self._nodes.values() if n.alive),
                      key=lambda x: int(x.node_id[1:]))


# ---------------------------------------------------------------------------
# Model specs + partition planner (M2a)
# ---------------------------------------------------------------------------

# Dashboard "how to run" modes -> (consolidate, prefer_vram) planner knobs.
LOAD_MODES: dict[str, tuple] = {
    "auto":       (True,  True),   # GPU-VRAM-first, fewest nodes (default; best latency)
    "single":     (True,  False),  # fewest nodes by RAM+VRAM (collapses to one box if it fits)
    "gpu-spread": (False, True),   # fill every GPU's VRAM, spill across nodes
    "all-gpu":    (False, False),  # #all-gpu: a stage on EVERY GPU, NOTHING on CPU (proportional
    #                                across the GPU subset). prefer_vram off so it doesn't spill;
    #                                gpu_spread=(mode=="all-gpu") flips the GPU-only filter on.
    "distribute": (False, False),  # spread across the WHOLE fleet (CPUs + GPUs)
    "spread":     (False, False),  # like distribute, but FORCE a stage on every capable node
    "proportional": (False, False),  # #78: layers across EVERY capable node PROPORTIONAL to its
    #                                  capacity (Hamilton apportionment) — for a big int4 MoE
    #                                  (MiniMax-M2) too big for the GPU-first subset.
}


class CapacityError(RuntimeError):
    """#at-capacity: a load was rejected for lack of room (resident-model cap or no placement).
    `terminal` picks the serving contract: False = eviction is possible but blocked RIGHT NOW
    (residents busy serving) — genuinely retryable, serve 503 + Retry-After; True = no automatic
    recovery exists (auto-unload off, or every resident no_unload-pinned) so a retry can NEVER
    succeed until an operator unloads/frees something — serve 503 + code "at_capacity" and NO
    Retry-After. A Retry-After on a terminal condition is a promise the server can't keep: an
    honest client that honors it retries forever (the om3nbox 25×503-over-90s transcript)."""
    def __init__(self, msg: str, terminal: bool = False):
        super().__init__(msg)
        self.terminal = terminal


FRAMEWORK_OVERHEAD_GB = 1.0
WEIGHT_DTYPE_BYTES = 2
KV_DTYPE_BYTES = 2
DEFAULT_CTX = 8192
VRAM_RESERVE_GB = 1.0   # leave this much VRAM per GPU for CUDA context/display
RAM_SAFETY_GB = 3.0     # headroom kept below a node's live free RAM (load transient + drift)


def _local_ipv4s() -> set:
    """This controller's own IPv4 addresses (+ loopback) — for co-located-worker detection."""
    ips = {"127.0.0.1"}
    try:
        import psutil   # not a module-level import here; absent psutil degrades to loopback-only
        for _n, _alist in psutil.net_if_addrs().items():
            for _a in _alist:
                if _a.family == socket.AF_INET and _a.address:
                    ips.add(_a.address)
    except Exception:
        pass
    return ips


_LOCAL_IPS = _local_ipv4s()


def _dial_host(host: str) -> str:
    """Dial a CO-LOCATED worker over loopback. A worker whose data_host is one of THIS machine's
    own IPs is local; dialing our own EXTERNAL IP throws WinError 64/1225 on Windows during any
    NIC/restart blip (the beast worker churn). 127.0.0.1 is robust + fastest. Remote workers
    (different IP) are returned unchanged so TP/pipeline reach them normally."""
    return "127.0.0.1" if host in _LOCAL_IPS else host
# Per-node build-transient reserve = factor x ONE layer's bf16 bytes (streaming load needs that
# much FREE during a layer's build, on top of the resident shard, or it OOMs). Since m4c25 EVERY
# node streams each layer straight into RAM bytes -> st_load heap tensors (no /dev/shm tmpfs), so
# the transient is ~2x one layer (bytes buffer + deserialized tensors) + margin on ALL OSes — the
# old Linux "tmpfs ~1x" assumption no longer holds and would UNDER-budget. A node lacking the
# headroom is excluded from the plan.
LOAD_TRANSIENT_RAM = 2.3   # in-RAM stream path (bytes + deserialize), all OSes (m4c25)
# #62: with per-expert FETCH streaming, an int4 MoE layer is NOT fetched as a whole ~7 GB bf16
# blob — only the small (skip_experts) layer blob (~tens of MB) + one ~256 MB expert chunk are
# transient at a time. So the per-layer build transient is bounded regardless of layer size. Cap
# the int4 reserve at this ceiling so big-MoE int4 loads spread across the WHOLE fleet (small nodes
# qualify) instead of piling many layers onto a few big nodes (which over-committed theocomp -> OOM).
STREAM_EXPERT_RESERVE_GB = 1.5
# #2-prealloc fix #1: reserve an extra per-node VRAM floor during PLANNING for the runtime
# overhead the static weight+KV estimate misses (decode activation buffers, allocator
# fragmentation). Without it a thin-headroom GPU node (e.g. a 6 GB laptop given a stage with
# ~0.7 GB to spare) passes the load then OOMs mid-decode and drops its data connection.
PLAN_VRAM_FLOOR_GB = float(os.environ.get("INFINITEMODEL_VRAM_FLOOR_GB", "2.0"))
# #78: the controller box ALSO reads the FULL bf16 from disk and streams it to every worker, so its
# OS file-cache + serving buffers + the controller process all want RAM *while* its co-located worker
# is building its shard. The planner used to hand the co-located worker the box's whole usable RAM,
# over-committing it -> the worker OOM-drops mid-load (the beast minimax crash: 93 GB balloon on
# 125.8 GB left too little for the 426 GB serve, died ~5/44 layers in). Reserve this much RAM on the
# CONTROLLER's co-located worker (data_host in _LOCAL_IPS) so serving + build don't collide. 0 disables.
CONTROLLER_RAM_RESERVE_GB = float(os.environ.get("INFINITEMODEL_CONTROLLER_RAM_RESERVE_GB", "20.0"))
# When a model's requested/training ctx won't fit the pool alongside its weights, the load
# auto-reduces ctx to the largest value that fits (binary-searched on the planner) rather than
# over-committing into an OOM. This is the floor: if not even CTX_AUTOFIT_FLOOR tokens fit, the
# weights themselves exceed the pool and the load fails with a clear error.
CTX_AUTOFIT_FLOOR = int(os.environ.get("INFINITEMODEL_CTX_FLOOR", "1024"))
# #76: when a model is too big for the fleet's VRAM (weights spill to CPU -> already slow) AND the
# user took the DEFAULT ctx, don't also pre-allocate a giant full-ctx KV buffer in RAM — that's what
# turned deepseek-70b's native 128K default into a generate hang (8192 ran, 128K timed out). Cap the
# AUTO ctx to this sane interactive default; an explicit ctx is always honored. Override via env.
AUTO_CTX_SLOW_CAP = int(os.environ.get("INFINITEMODEL_AUTO_CTX_SLOW_CAP", "16384"))
MAX_LOADED_MODELS = int(os.environ.get("INFINITEMODEL_MAX_LOADED", "4"))  # default safety cap
DEFAULT_QUEUE_DEPTH = 16  # waiters allowed per model beyond the one in the slot. Generation is
#                           serialized per model (model.lock), so a waiter just holds its connection
#                           until its turn — the gen-stall watchdog guards true wedges. Was 2 (cap 3),
#                           which 503'd a quorum/agent client that fans out 5-6+ concurrent requests to
#                           ONE model (#queue-depth). 16 → cap 17/model absorbs that; overflow now
#                           returns a RETRYABLE 429+Retry-After (not 503). Tunable live via /config
#                           queue_depth. (Ollama's OLLAMA_MAX_QUEUE defaults to 512.)

# --- #ctx-history: per-loaded-model rolling capture of the ACTUAL context in/out, for the dashboard
# model-detail popup. Stores TOKEN IDS (cheap at capture — the hot path never detokenizes; /history
# decodes lazily on click) keyed by friendly name; capped to the most-recent N requests AND a token
# budget so huge Claude-Code contexts can't grow unbounded. Cleared when the model unloads (the
# history is only meaningful while the model is resident).
REQUEST_HISTORY: dict = {}
HISTORY_KEEP = int(os.environ.get("IM_HISTORY_KEEP", "30") or 30)            # max requests kept/model
HISTORY_TOK_BUDGET = int(os.environ.get("IM_HISTORY_TOK_BUDGET", "1500000") or 1500000)  # total tok cap


def _record_ctx_history(friendly: str, in_ids, out_ids, tok_in: int, tok_out: int) -> None:
    """Append one request's input+output token ids to the model's rolling history. Best-effort —
    never raises into the decode path. Oldest entries drop past HISTORY_KEEP or the token budget."""
    try:
        if HISTORY_KEEP <= 0:
            return
        dq = REQUEST_HISTORY.setdefault(friendly, [])
        dq.append({"ts": int(time.time() * 1000), "tok_in": int(tok_in), "tok_out": int(tok_out),
                   "in_ids": list(in_ids), "out_ids": list(out_ids)})
        while len(dq) > HISTORY_KEEP:
            dq.pop(0)
        tot = sum(e["tok_in"] + e["tok_out"] for e in dq)
        while len(dq) > 1 and tot > HISTORY_TOK_BUDGET:
            e = dq.pop(0)
            tot -= (e["tok_in"] + e["tok_out"])
    except Exception:
        pass

# --- Engine config (persisted; runtime-tunable from the dashboard) ---
#   max_loaded   : cap on how many models stay resident at once.
#   auto_unload  : when a new load won't fit (or the cap is hit), evict IDLE models
#                  (no active/queued requests), LRU-first, to make room. A model that is
#                  actively serving is NEVER evicted — the load fails instead.
#   queue_depth  : max requests allowed WAITING per model (1 running slot + queue_depth
#                  queued); a request arriving when the queue is full is rejected (503).
ENGINE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_config.json")
ENGINE_CONFIG: dict = {"max_loaded": MAX_LOADED_MODELS, "auto_unload": False,
                       "queue_depth": DEFAULT_QUEUE_DEPTH,
                       # #autoload-smallest: quant an AUTO-LOADED (requested-but-not-resident) model
                       # defaults to — the SMALLEST that fits the common case. int4 is ~1/4 the bf16
                       # memory, fits more nodes, and serves PRE-PACKED when a shard cache exists, so a
                       # request never streams the full bf16 just to serve. int4|int8|none; on int4/int8
                       # failure ensure_loaded falls back ONCE to bf16 ("int4 in almost all cases").
                       "autoload_quant": "int4",
                       # #auto-defaults: the context length + placement mode an auto-load (and the
                       # dashboard's per-model Load button) uses. ctx default 8k (a sane working window
                       # that keeps KV modest); mode 'auto' (GPU-first, fewest nodes). A request with its
                       # own ctx>0 overrides autoload_ctx. Configurable via /config + the dashboard.
                       "autoload_ctx": DEFAULT_CTX,
                       "autoload_mode": "auto",
                       # #kv-quant (TurboQuant): KV-cache quantization preset for a load. "none" = bf16
                       # KV (default, unchanged). "turbo3"/"turbo4"/"turbo2" = TurboQuant K/V at 3/4/2
                       # bits (random-rotation + Lloyd-Max scalar quant; see kv_quant.py) — stores KV at
                       # ~3 bits/coord at near-FP quality to relieve the KV-VRAM ctx cap (#76). Plumbed
                       # to the worker shard at load; the quantized cache (TurboQuantCache) activates in
                       # shard_forward. (#172)
                       "kv_quant": "none",
                       # #kv-offload: default for loads that don't pass kv_offload= — KV cache in
                       # system RAM (transformers OffloadedCache, per-layer prefetch) instead of
                       # VRAM. Frees the GPU KV reserve for model layers (long-ctx on small cards)
                       # at a decode-speed cost. Per-load kv_offload=1 overrides. Exclusive with
                       # kv_quant (offload wins nothing; kv_quant takes precedence when both set).
                       "kv_offload": False,
                       # #vram-weights-first: budget a NEW model's WEIGHTS against PHYSICALLY-free VRAM
                       # (live: total - actually-used), letting them use resident models' RESERVED-but-
                       # unfaulted full-ctx KV headroom — so a model lands on GPU when VRAM is physically
                       # free, instead of spilling weights to CPU because another model's reserved KV
                       # "owns" that VRAM. Each model still reserves its OWN KV. Trade-off: if multiple
                       # coexisting models all grow long contexts at once, a resident model's KV growth
                       # can be VRAM-starved (slower/clamped) rather than guaranteed. Set False to restore
                       # the conservative #95 reservation (weights spill before a resident model's KV).
                       "vram_weights_first": True,
                       # #gen-stall-watchdog: seconds a model may show active>0 with NO token produced
                       # before the watchdog declares it wedged and reclaims the slot. 0 disables it.
                       "gen_stall_s": GEN_STALL_S,
                       # #active-decode-stall: tighter stall threshold AFTER the first token (decode phase)
                       "gen_stall_decode_s": GEN_STALL_DECODE_S,
                       # #77 persistence: models to AUTO-RELOAD on controller startup (survives a
                       # restart/crash/deploy). reg_key -> {"ctx", "quant"}. Workers drop their shards
                       # when the controller link drops, so recovery = re-stream on startup (after the
                       # fleet settles, so GPU models land on the GPU, not CPU). Opt-in (default empty);
                       # set via /config?persist=<model> / the dashboard 📌 toggle.
                       "persist_models": {},
                       # #no-unload: per-model ABSOLUTE do-not-auto-unload veto (reg_key -> true).
                       # A model in this set is NEVER auto-unloaded — not by idle_unload_m, and not by
                       # LRU eviction to make room for a new load (a load that can't otherwise fit
                       # FAILS instead). It WINS over every automatic-REMOVAL path. The juggler is the
                       # ONE deliberate exception: it may still RELOAD a pinned model to PROMOTE it to a
                       # faster VRAM-only placement — a reload-in-a-better-way, NOT a removal (and it
                       # restores the model if a rare double-failure evicts it, so "never auto-unloaded"
                       # still holds). Independent of persist_models (survives a restart but stays
                       # evictable under pressure).
                       # Opt-in (default empty); set via /config?no_unload=<model> / the dashboard
                       # per-model checkbox.
                       "no_unload_models": {},
                       # #juggler: promote a resident model running split GPU+RAM (hybrid) to VRAM-only
                       # once the fleet has room. Fires on TWO triggers — right after an idle-unload
                       # frees VRAM, AND a periodic ~60s sweep (juggle_sweep_s) so it also acts when
                       # VRAM frees for any OTHER reason (a manual unload, a shrinking KV, an earlier
                       # promotion). Picks the hottest hybrid that would now fit ENTIRELY on GPU (skips
                       # embeddings and anything that can't fit) and, only when that model is momentarily
                       # IDLE (a busy one is skipped, not stalled — a later sweep catches it at a gap),
                       # does a HITLESS swap: a per-model barrier holds new requests while reconfigure
                       # re-places it VRAM-first — so the client's open connection just pauses across
                       # the re-place (no reconnect). One promotion per pass, serialized. Lets models
                       # auto-load hybrid under pressure yet migrate to full-GPU speed as the hot one
                       # out-competes the others for VRAM. Opt-in (default off).
                       "juggler": False,
                       # #autostart-delay: minimum wall-clock seconds _persist_reload waits on
                       # startup before it begins reloading persisted models — ON TOP of waiting for
                       # the worker fleet to settle — so API clients have time to (re)connect before
                       # the controller gets busy streaming weights. 0 = no extra wait. Default 60.
                       "autostart_delay_s": 60.0,
                       # #wedge-quarantine: after this many gen-stall-watchdog reclaims of the SAME
                       # model within 15 min, force a fresh re-place of it (reconfigure: new shards +
                       # new data conns, rollback-safe) instead of letting client retries re-wedge it
                       # forever — the 2026-07-09 beast wedge-storm (37 VL prefill wedges in 5.5h fed
                       # a kernel panic) self-heals after ~3 wedges. 0 disables the self-heal.
                       "wedge_reload_n": 3,
                       # #idle-unload: unload a model after this many minutes with NO requests.
                       # 0 (default) = loaded forever. Independent of auto_unload (that knob is
                       # about evicting to make ROOM for a new load; this one reclaims memory on a
                       # quiet fleet). Pinned (persist_models) models and models with an active or
                       # queued request are never idle-unloaded.
                       "idle_unload_m": 0.0}
# A loaded model stays resident FOREVER by default: a request never unloads it, and a new load
# that doesn't fit simply fails (unload one first) unless auto_unload lets it LRU-evict an idle
# model for room. Time-BASED reclaim is the separate idle_unload_m knob above (0 = off).

# #logs: per-node log tails relayed by workers on their heartbeat (node_id -> list[str], trimmed).
# Served via GET /logs?node=<host|node_id>; the controller's OWN log ring lives in wire (tail_logs).
NODE_LOGS: dict = {}
NODE_LOGS_MAX = 4000


# --- In-flight request registry (slots + queue observability) -------------------
# 1 SLOT per model = the per-model lock (only one request generates at a time);
# everything else WAITS in that model's queue. We track each request here so the
# dashboard can show, per slot and per queue entry: client IP, the model wanted,
# and how long it has been running / waiting.
INFLIGHT: dict = {}        # id -> {id, ip, model, state: queued|running, enqueued, started}
_INFLIGHT_SEQ = 0


def _inflight_admit(ip: str, model: str, slots: int = 1):
    """Register a request for `model`. `slots` = concurrent running slots available (one per
    resident replica — #39 data-parallel; 1 for a single-placement model). Admit if a running
    slot is free, else queue it if the queue isn't full (>= queue_depth waiting); otherwise
    return None (caller rejects with 503). Running concurrency is still enforced downstream by
    each replica's per-model lock — this gate only bounds how much can pile up per model."""
    global _INFLIGHT_SEQ
    depth = int(ENGINE_CONFIG.get("queue_depth", DEFAULT_QUEUE_DEPTH))
    running = sum(1 for r in INFLIGHT.values()
                  if r["model"] == model and r["state"] == "running")
    queued = sum(1 for r in INFLIGHT.values()
                 if r["model"] == model and r["state"] == "queued")
    if running >= max(1, slots) and queued >= depth:
        return None
    _INFLIGHT_SEQ += 1
    # capture the running request task so /cancel (#48) can abort it (incl. one wedged in a load)
    try:
        _task = asyncio.current_task()
    except Exception:
        _task = None
    rec = {"id": _INFLIGHT_SEQ, "ip": ip or "?", "model": model,
           "state": "queued", "enqueued": time.time(), "started": None,
           "cancel": False, "task": _task}
    INFLIGHT[rec["id"]] = rec
    return rec


def _inflight_start(rec) -> None:
    """Mark a request as occupying its model's slot (the lock was acquired)."""
    if rec is not None:
        rec["state"] = "running"
        rec["started"] = time.time()


def _inflight_release(rec) -> None:
    if rec is not None:
        INFLIGHT.pop(rec["id"], None)


# --- Per-client connection registry (#connections) ------------------------------
# Every HTTP client (keyed by IP) gets a live accounting row for the dashboard's
# Connections panel: first/last activity, REAL bytes on the wire both directions
# (counted at the ASGI layer so streamed responses grow the counter chunk-by-chunk),
# token totals (bumped by the serving paths at request end), and the last model
# touched. In-memory only (resets with the controller). /weights is EXCLUDED —
# those are worker slice-pulls during a load, not client traffic (they would dwarf
# every real client with tens of GB of transfer).
CLIENTS: dict = {}
CLIENTS_MAX = 64
# X-Forwarded-For values must LOOK like an IP before they become a CLIENTS key (they render
# in dashboard HTML + a /terminate onclick — arbitrary header text would be an XSS vector)
_IP_RE = re.compile(r"^[0-9A-Fa-f.:]{2,45}$")
# generation/embedding endpoints — hitting one marks the row as a real API client
# (vs a browser that only watches the dashboard; the dashboard itself calls /api/show
# and /api/ps, so "starts with /api/" would mislabel it)
_CLIENT_API_PATHS = ("/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
                     "/v1/chat/completions", "/v1/completions", "/v1/messages",
                     "/v1/embeddings", "/v1/audio")


def _client_row(ip: str) -> dict:
    row = CLIENTS.get(ip)
    if row is None:
        if len(CLIENTS) >= CLIENTS_MAX:      # trim the longest-idle row
            oldest = min(CLIENTS.values(), key=lambda r: r["last_seen"])
            CLIENTS.pop(oldest["ip"], None)
        now = time.time()
        row = CLIENTS[ip] = {"ip": ip, "first_seen": now, "last_seen": now,
                             "bytes_in": 0, "bytes_out": 0, "tok_in": 0, "tok_out": 0,
                             "reqs": 0, "api": False, "last_model": None}
    return row


def _client_tokens(ip: str, tok_in: int = 0, tok_out: int = 0, model: str = "") -> None:
    """Serving paths report a finished request's token counts (engine.generate's finally +
    _serve_embed) so the Connections panel shows real per-client token totals."""
    try:
        row = _client_row(ip or "?")
        row["tok_in"] += int(tok_in or 0)
        row["tok_out"] += int(tok_out or 0)
        if model:
            row["last_model"] = model
    except Exception:
        pass


class _ClientAccounting:
    """#connections: count REAL bytes per client at the ASGI layer. An
    @app.middleware('http') only sees Response objects — a StreamingResponse's bytes
    are invisible there — while wrapping receive/send counts every chunk as it crosses
    the wire, so a minutes-long SSE stream shows a live, growing bytes_out. last_seen
    is stamped on every chunk, so an ACTIVE stream is never 'idle'."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "") or ""
        if path.startswith("/weights"):       # worker slice-pulls, not client traffic
            return await self.app(scope, receive, send)
        ip = None
        try:                                   # honor X-Forwarded-For like _client_ip
            for k, v in scope.get("headers") or []:
                if k == b"x-forwarded-for":
                    ip = v.decode("latin-1").split(",")[0].strip()
                    break
        except Exception:
            pass
        # X-Forwarded-For is CLIENT-CONTROLLED text that ends up in the dashboard's HTML and
        # in a /terminate onclick — accept only something that LOOKS like an IP (v4/v6 charset,
        # bounded length); anything else falls back to the socket address.
        if ip and not _IP_RE.match(ip):
            ip = None
        if not ip:
            ip = (scope.get("client") or ("?",))[0] or "?"
        row = _client_row(ip)
        row["reqs"] += 1
        row["last_seen"] = time.time()
        if not row["api"] and path.startswith(_CLIENT_API_PATHS):
            row["api"] = True

        async def _recv():
            msg = await receive()
            if msg.get("type") == "http.request":
                row["bytes_in"] += len(msg.get("body") or b"")
                row["last_seen"] = time.time()
            return msg

        async def _send(msg):
            if msg.get("type") == "http.response.body":
                row["bytes_out"] += len(msg.get("body") or b"")
                row["last_seen"] = time.time()
            await send(msg)

        return await self.app(scope, _recv, _send)


def _client_ip(req) -> str:
    """Best-effort client IP for slot/queue display (honors a proxy's X-Forwarded-For).
    #connections: the SAME _IP_RE validation as the accounting middleware — INFLIGHT rec.ip,
    load requested_by and the token bumps all JOIN against the middleware-keyed CLIENTS row,
    so both derivations must agree (a malformed XFF would otherwise split one client's stats
    across a real row and a ghost row keyed by raw header text, which also renders in HTML)."""
    xff = req.headers.get("x-forwarded-for")
    if xff:
        cand = xff.split(",")[0].strip()
        if _IP_RE.match(cand):
            return cand
    return req.client.host if req.client else "?"


def _not_found_json(model: str, mode: str) -> JSONResponse:
    """HTTP 404 for a truly-UNKNOWN model name (resolve_model_name ValueError). OpenAI callers
    (mode openai|openai_text) get the OpenAI error envelope w/ code model_not_found; Ollama callers
    get Ollama's {"error":"model '<name>' not found"} shape. Only the unknown case routes here —
    a present-but-not-loadable model keeps its distinct error elsewhere."""
    if mode in ("openai", "openai_text"):
        return JSONResponse({"error": {"message": f"The model '{model}' does not exist",
                             "type": "invalid_request_error", "code": "model_not_found"}},
                            status_code=404)
    return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)


def load_engine_config() -> None:
    try:
        with open(ENGINE_CONFIG_PATH, encoding="utf-8") as fh:
            ENGINE_CONFIG.update(json.load(fh))
    except Exception:
        pass


def save_engine_config() -> None:
    try:
        with open(ENGINE_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(ENGINE_CONFIG, fh, indent=2)
    except Exception as exc:
        print(f"[cfg] could not save engine config: {exc!r}")


# Controller activity log: a most-recent-first ring buffer of what the controller is doing
# (planning, handing out shards, serving weight chunks, unloading, downloading), surfaced in
# /status for the dashboard's activity panel and echoed to the console.
ACTIVITY: deque = deque(maxlen=80)
UNLOADS: deque = deque(maxlen=12)   # recent "why a model left" events (dashboard panel)
# #error-log: HTTP error responses (4xx/5xx) returned to any client/node — surfaced in the Logs UI
# so a 404/500/502 a caller saw is visible server-side without tailing the console.
ERRORS: deque = deque(maxlen=120)


def log_activity(msg: str) -> None:
    """Record a one-line 'what the controller is doing' event (newest first) + echo it."""
    ACTIVITY.appendleft({"t": round(time.time(), 1), "msg": msg})
    print(f"[activity] {msg}")


def log_error(method: str, path: str, status: int, ip: str = "?", detail: str = "") -> None:
    """Record an HTTP error response (status >= 400) for the Logs UI's Errors panel (newest first)."""
    ERRORS.appendleft({"t": round(time.time(), 1), "method": str(method or "?"),
                       "path": str(path or "?"), "status": int(status),
                       "ip": str(ip or "?"), "detail": (str(detail or ""))[:300]})


def _classify_unload(reason: str) -> str:
    """Bucket an unload reason so the dashboard can color it. The reason strings are
    all controller-internal (see the record_unload call sites), so keyword-matching is
    reliable enough and keeps each call site from having to name its own kind."""
    r = reason.lower()
    if "reap" in r or "left" in r or "disconnect" in r or "invalidat" in r:
        return "node-loss"        # non-graceful: a node died/dropped while holding a shard
    if "evict" in r or "room" in r or "free workers" in r or "cap" in r:
        return "evict"            # made room for another load (LRU / cap / tp prep)
    if "reload" in r:
        return "reload"           # same model being reloaded (new ctx/quant)
    return "manual"               # operator-requested unload


def record_unload(model: str, reason: str, hosts: Optional[list] = None,
                  kind: Optional[str] = None) -> None:
    """Record WHY a model left the fleet so the dashboard can SHOW it in a dedicated panel
    (distinct from the scrolling activity firehose, where the reason quickly scrolls away).
    Every path that removes a model from engine.models routes through here — graceful
    (manual/reload/evict) AND non-graceful (a node reaped/disconnected mid-serve)."""
    UNLOADS.appendleft({"t": round(time.time(), 1), "model": model, "reason": reason,
                        "kind": kind or _classify_unload(reason), "hosts": hosts or []})


def _release_ram() -> None:
    """Promptly free dereferenced objects and hand large freed blocks back to the OS. The
    controller holds each loaded model's speculative-decode DRAFT in its own RAM; torch models
    have reference cycles, so without an explicit gc the draft lingers (and the controller RSS
    stays high) until some later gc cycle. Call after any unload/evict/invalidate."""
    import gc
    gc.collect()
    with contextlib.suppress(Exception):
        import ctypes
        import sys
        if sys.platform.startswith("linux"):       # return freed arena to the OS (Windows
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # frees large tensor blocks on its own)


def _free_mtp_cuda() -> None:
    """#91: a controller-resident MTP head loaded on cuda:0 keeps its ~few-GB VRAM in the torch
    caching allocator after the Python object is dropped — only empty_cache() hands it back. Without
    this the freed head fouls the controller-box GPU (qwen3:4b couldn't load, qwen2.5:14b spilled to
    CPU). Best-effort; torch may be absent on a pure controller."""
    import gc
    gc.collect()
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Model specs + the layer-placement planner (incl. the #76 pre-load guardrail) live in placement.py
# now — split out of this file (#38) to shrink it. placement.py is controller-only and listed in
# EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync. CONVERGENCE BRIDGE: the (old)
# server.py that swapped IN this version may not have fetched placement.py yet (it wasn't in the old
# EXTRA_UPDATE_FILES), so if the import fails, pull the file to disk once and import — this avoids an
# import-time crash loop on the single self-update cycle before placement.py propagates everywhere.
try:
    import placement as _placement   # noqa: F401
except Exception:
    _pl_src = _fetch_repo_file("placement.py")
    if _pl_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement.py"), "wb") as _pl_f:
            _pl_f.write(_pl_src)
from placement import (ModelSpec, MODEL_SPECS, NodeMem, StageAssign, PlanResult, _mem_pref,
                       _node_layer_capacity, _plan_vram_first, _describe_plan, _round_ctx,
                       _assess_placement, plan_pipeline, _node_tp_bw)   # noqa: E402,F401

_CFG_SPEC_CACHE: dict = {}   # hf_id -> ModelSpec built from its downloaded config.json

# Name ALIASES: extra client-facing names that map to an existing registry key (canonical
# dash-form). Unlike a separate MODELS entry, an alias resolves to the SAME key — so the model
# loads/serves under ONE key no matter which name is requested: either name accesses the same
# resident copy, the load is idempotent (never loaded twice as both names), and a loaded-check for
# either name reports loaded. Keys/values are canonical dash forms (post _normalize_model_request).
MODEL_ALIASES: dict[str, str] = {
    "qwen2.5-14b": "qwen2.5-14b-instruct",   # 'qwen2.5:14b' <-> 'qwen2.5:14b-instruct'
}


def _aliases_for(friendly: str) -> list:
    """Reverse of MODEL_ALIASES: the display-form alias(es) that resolve to this registry key, so the
    dashboard can show them under the model's primary name (e.g. 'qwen2.5:14b-instruct' lists alias
    'qwen2.5:14b'). Rendered in the Ollama 'family:size' form via _ollama_name (resolved at call time)."""
    return [_ollama_name(a) for a, canon in MODEL_ALIASES.items()
            if canon == friendly and a != friendly]


def resolve_spec(model: str) -> Optional[ModelSpec]:
    model = MODEL_ALIASES.get(model, model)   # alias -> canonical registry key
    if model in MODELS:
        model = MODELS[model][0]
    spec = MODEL_SPECS.get(model)
    if spec is not None:
        return spec
    # Arbitrary (user-added) model with no hard-coded spec: build one from its
    # downloaded config.json so the planner can size + place it. Cached.
    if model in _CFG_SPEC_CACHE:
        return _CFG_SPEC_CACHE[model]
    d = _local_model_dir(model)
    if d:
        s = _spec_from_config(d, _friendly_from_hf(model))
        if s is not None:
            _CFG_SPEC_CACHE[model] = s
        return s
    return None


def resolve_model_name(name: str) -> str:
    """Normalize an API/dashboard model name to a key that resolve_spec + the internal
    dicts (engine.models, DOWNLOADING, ...) understand. Accepts ALL equivalent forms
    interchangeably: 'qwen3:4b', 'qwen3-4b', 'qwen3-4b:latest', 'qwen3:4b:latest' ->
    'qwen3-4b' (the registered dash-form key), plus arbitrary HF ids (org/name). Always
    returns an EXISTING registry key when one matches (backward compat: models already
    keyed 'qwen3-4b' in custom_models.json keep working, and 'qwen3:4b' now resolves to
    them too). Raises ValueError only for an unknown bare name (no '/')."""
    norm = _normalize_model_request(name)          # canonical dash form (or raw HF id)
    norm = MODEL_ALIASES.get(norm, norm)           # map an alias to its canonical registry key
    raw = (name or "").strip().lower()
    base = raw.split(":")[0] if "/" not in raw else raw   # legacy: bare name before ':'
    # Match registered keys first, trying the canonical dash form, the colon-display form
    # (in case a model was ever keyed that way), then the legacy/literal inputs.
    for cand in (norm, _ollama_name(norm), raw, base, name):
        if cand and (cand in MODELS or cand in MODEL_SPECS):
            return cand
    if "/" in norm:          # an arbitrary HF id — accept (download/spec resolve it)
        # #cache-case: _normalize_model_request LOWERCASES HF ids, but a model's dir + its
        # _shards/<quant> cache live under the ORIGINAL-CASE registered target (e.g.
        # 'mistralai/Devstral-Small-2-24B-Instruct-2512'). On a case-SENSITIVE filesystem (Linux/
        # om3nbox) the lowercased id resolves to a DIFFERENT, cache-less dir, so /weights can't see
        # the int4 cache and the load silently streams + serves bf16 (4x the memory; harmless on
        # case-insensitive Windows, which is why it only bit the Linux box). Map the (mis-cased) HF
        # id BACK to its registered target so serve + compile share ONE dir; else preserve the
        # caller's ORIGINAL case (never the lowercased norm) so a fresh download lands in the right dir.
        for _tgt, _draft in MODELS.values():
            if _tgt.lower() == norm.lower():
                return _tgt
        return name.strip() if (name and "/" in name) else norm
    raise ValueError(f"unknown model '{name}'; known: {', '.join(MODELS)}")


def run_self_test_plan(os_reserve_gb: float, gb_list: list[float], ctxs: list[int]) -> None:
    nodes = [NodeMem(f"n{i+1}", f"box{i+1}", int(max(0.0, gb - os_reserve_gb) * GB))
             for i, gb in enumerate(gb_list)]
    pool = sum(n.usable_bytes for n in nodes) / GB
    print("\nPartition planner self-test")
    print(f"  fleet (raw GB):    {gb_list}")
    print(f"  usable after {os_reserve_gb:g} GB OS reserve: "
          f"{[round(n.usable_bytes/GB, 2) for n in nodes]}  (pool {pool:.1f} GB)\n")
    for friendly, (target, _draft) in MODELS.items():
        spec = MODEL_SPECS.get(target)
        if not spec:
            continue
        for ctx in ctxs:
            r = plan_pipeline(spec, nodes, ctx_len=ctx)
            tag = "FITS " if r.ok else "NO   "
            print(f"[{tag}] {friendly:<20} ctx {ctx:>6}  "
                  f"need {r.required_gb:5.1f} / pool {r.pool_usable_gb:4.1f} GB")
            if r.ok:
                for s in r.stages:
                    flags = ("E" if s.has_embed else "-") + ("H" if s.has_head else "-")
                    print(f"          {s.hostname:<6} L{s.layer_start:>2}-{s.layer_end:<2} "
                          f"[{flags}] {s.num_layers:>2} layers  {s.est_gb:5.2f}/{s.usable_gb:4.1f} GB")
            else:
                print(f"          {r.error}")
        print()


# ---------------------------------------------------------------------------
# Data-plane framing (length-prefixed binary tensor frames; mirrors client.py)
# ---------------------------------------------------------------------------

# Tensor (un)packing lives in wire.py (shared with client.py); kept in sync on every node by
# the multi-file self-update (wire.py is in EXTRA_UPDATE_FILES) and present from a fresh
# git clone, so a plain import is safe.
from wire import (_pack_tensor, _unpack_tensor, _set_keepalive, _tp_hetsplit,   # noqa: F401
                  install_log_tee, tail_logs)


# code-split Inc 2: the CONTROL PLANE (frame IO, ControlLink, the resilient TCP listener,
# handle_control, reaper_loop, gen_stall_watchdog) lives in control_plane.py now -- a controller-only
# leaf module (bodies VERBATIM; globals injected by state.bind, see state.py). CONVERGENCE BRIDGE
# (same as formats.py/multimodal.py): an old checkout self-updating onto this server.py may not have
# fetched control_plane.py yet -- pull it once from the repo raw endpoint, then import.
try:
    import control_plane as _control_plane   # noqa: F401
except Exception:
    _cp_src = _fetch_repo_file("control_plane.py")
    if _cp_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_plane.py"), "wb") as _cp_f:
            _cp_f.write(_cp_src)
import control_plane
from control_plane import (_read_frame, _write_frame, _enc, ControlLink, _ResilientServer,
                           _resilient_serve, _resolve_pending, handle_control, reaper_loop,
                           gen_stall_watchdog)   # noqa: E402,F401


# ---------------------------------------------------------------------------
# M2d: controller-side model storage + chunk serving
# ---------------------------------------------------------------------------
# The controller is the single source of model weights: it downloads the full
# model once and serves each worker only its layer tensors over HTTP, which the
# worker loads straight into RAM. Workers keep NO model on disk, so the smallest
# disk no longer caps model size — only the controller's disk does.
#
# The model STORAGE + download/measure helpers (models/ dir resolution, the cache->models/
# migration, real-safetensors measurement, config->ModelSpec, Ollama-name normalization, and the
# HF-cache size/purge/gc/delete bookkeeping + ready/local-dir caches) live in model_store.py now —
# split out of this file (#38, step E) to shrink it. model_store.py is a controller-only leaf module
# (stdlib + huggingface_hub + placement.ModelSpec + three pure shards helpers, no server state, never
# imports server), listed in EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync. Its
# ONE controller dependency, the HF read token, is supplied by DEPENDENCY INJECTION
# (set_hf_token_provider below) — no back-import -> no import cycle.
#
# CONVERGENCE BRIDGE (same as placement.py/shards.py/formats.py/multimodal.py/graphs.py): the old
# server.py that swapped in this version may not have fetched model_store.py yet (it wasn't in the old
# EXTRA_UPDATE_FILES), so if the import fails, pull the file to disk once then import — no import-time
# crash loop on the single self-update cycle before model_store.py propagates everywhere.
#
# NOTE: the DOWNLOAD-STATE definitions (DOWNLOADING / DOWNLOAD_PROGRESS / DOWNLOAD_ERROR /
# DOWNLOAD_CONTROL / DOWNLOAD_STATE / DOWNLOAD_EPOCH / DOWNLOAD_STATE_PATH + load/save_download_state)
# STAY here (below): the self-updater's idle lambda reads DOWNLOADING as a LIVE server global, so
# moving the definitions would decouple the idle gate (the historical ENCODING hazard, state.py
# — ENCODING itself moved to media_encode.py WITH its mutators, Inc 11's other valid pattern). All writers
# mutate them IN PLACE (m4c155 + Inc 4: no rebinds anywhere), which is why the download ROUTES and
# _pull_repo_interruptible could move to downloads.py (code-split Inc 5) — a moved global rebind
# would decouple from server's name (the ENCODING hazard; resolved for ENCODING itself in Inc 11).
# resolve_spec / resolve_model_name / _ollama_name / _split_family_size also stay (they use the MODELS
# registry / MODEL_ALIASES); they call the moved helpers through this bridge import.
try:
    import model_store as _model_store   # noqa: F401
except Exception:
    _ms_src = _fetch_repo_file("model_store.py")
    if _ms_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_store.py"), "wb") as _ms_f:
            _ms_f.write(_ms_src)
import model_store
from model_store import (MODELS_DIR, _safe_name, _dir_has_model, _controller_model_dir,
                         _train_ctx_from_dir, measure_model_weights, _MEAS_CACHE,
                         spec_with_measurements, _local_model_dir, _LOCAL_DIR_CACHE,
                         _display_weight_bytes, _friendly_from_hf, _normalize_model_request,
                         _spec_from_config, _hf_total_bytes, _hf_cache_bytes, model_ready,
                         _READY_CACHE, _invalidate_ready_cache, _purge_hf_cache,
                         gc_redundant_cache, delete_model_cache,
                         convert_gguf_to_model_dir,
                         _is_diffusers_dir, _is_kokoro_dir, _tree_weight_bytes)   # noqa: E402,F401
model_store.set_hf_token_provider(lambda: HF_TOKEN)
# GGUF source lookup: a target (HF repo) in GGUF_FILES is normalized to safetensors at acquisition
# instead of pulled as safetensors. Set after GGUF_FILES is populated by load_custom_models() too,
# but the lambda reads the live dict so registering at import is fine.
model_store.set_gguf_provider(lambda repo: GGUF_FILES.get(repo))


# code-split Inc 5: _pull_repo_interruptible lives in downloads.py now (VERBATIM; the
# DOWNLOAD_* state it mutates stays defined below -- see the NOTE above load_download_state).


# A parameter-size token: 4b, 0.5b, 1.5b, 14b, 70b, 348m (dense) or 8x7b, 16x3b (MoE).
_SIZE_TOKEN_RE = re.compile(r"(?:\d+x\d+b|\d+(?:\.\d+)?[bm])$")
# A clean variant suffix that may trail the size in the tag: 'instruct', 'chat', 'it', 'base'…
# (purely alphabetic). This is what distinguishes a real Ollama tag ('14b-instruct') from a
# name that merely happens to contain a size-shaped segment mid-string ('348m-alpha-polish900').
_VARIANT_SEG_RE = re.compile(r"[a-z]+")


def _split_family_size(friendly: str) -> tuple[str, str]:
    """Split a dash-form friendly key into (family, tag) at the FIRST parameter-size
    segment, Ollama-style: 'qwen3-4b' -> ('qwen3', '4b'); 'qwen2.5-14b-instruct' ->
    ('qwen2.5', '14b-instruct'); 'mixtral-8x7b' -> ('mixtral', '8x7b'); 'olmoe-1b-7b' ->
    ('olmoe', '1b-7b'). The split only happens when EVERY segment from the size onward is
    a clean tag part — another size token (e.g. the '7b' in '1b-7b') or a purely-alphabetic
    variant word (e.g. 'instruct'). A name with a size-shaped segment buried mid-string and
    trailed by junk ('coneml-348m-alpha-polish900', trailed by 'polish900') is NOT split and
    is returned unchanged. Returns (friendly, '') when there is no clean trailing size."""
    parts = friendly.split("-")
    for i in range(1, len(parts)):                 # never treat the leading segment as the size
        if not _SIZE_TOKEN_RE.fullmatch(parts[i]):
            continue
        tail = parts[i:]                            # the size segment + everything after it
        if all(_SIZE_TOKEN_RE.fullmatch(s) or _VARIANT_SEG_RE.fullmatch(s) for s in tail):
            return "-".join(parts[:i]), "-".join(tail)
        return friendly, ""                         # size found but tail is unclean -> leave as-is
    return friendly, ""


def _ollama_name(friendly: str) -> str:
    """DISPLAY name in Ollama 'family:size' form: 'qwen3-4b' -> 'qwen3:4b';
    'qwen2.5-14b-instruct' -> 'qwen2.5:14b-instruct'. A name with no clean trailing size
    token (or a name that already contains a ':') is returned unchanged — so the size IS
    the tag and we never append ':latest' on top of it."""
    if ":" in friendly:
        return friendly
    family, tag = _split_family_size(friendly)
    return f"{family}:{tag}" if tag else friendly


# --- Model lifecycle on the controller -------------------------------------
# The controller is the single source of weights and NEVER auto-purges; models
# are kept until explicitly deleted. Only models fully present on disk are
# reported as available.
DOWNLOADING: set[str] = set()      # friendly names with an in-flight download
DOWNLOAD_PROGRESS: dict[str, dict] = {}   # friendly -> {"downloaded": bytes, "total": bytes}
DOWNLOAD_ERROR: dict[str, str] = {}       # friendly -> last download failure (shown until next try)
DOWNLOAD_CONTROL: dict[str, str] = {}     # friendly -> "pause" | "stop": live interrupt signal read
                                          # by the per-file pull loop between files (absent = run)
DOWNLOAD_STATE: dict[str, str] = {}       # friendly -> "paused" | "stopped": PERSISTED intent so a
                                          # halted download survives a controller restart (no auto-resume)
DOWNLOAD_EPOCH: dict[str, int] = {}       # friendly -> generation counter; bumped on every (re)start
                                          # AND on clear, so a superseded in-flight _dl can tell its
                                          # result is stale and must NOT re-write state (clear race)
DOWNLOAD_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_state.json")


def load_download_state() -> None:
    """Restore paused/stopped download intents so a restart doesn't lose them (the
    partial bytes live in the HF cache; the user resumes/clears when ready). Also seed
    each halted model's frozen progress from the on-disk cache so the dashboard shows
    where it stopped instead of a 0% bar (in-memory DOWNLOAD_PROGRESS is lost on restart)."""
    # m4c155: mutate IN-PLACE (clear()+update()), not rebind — preserves DOWNLOAD_STATE's object
    # identity so the state.publish() snapshot stays live for relocated readers (e.g. _model_entry
    # in status.py). This was the one DOWNLOAD_STATE rebind; all other writers already mutate in place.
    DOWNLOAD_STATE.clear()
    try:
        with open(DOWNLOAD_STATE_PATH, encoding="utf-8") as fh:
            DOWNLOAD_STATE.update({k: v for k, v in json.load(fh).items()
                                   if v in ("paused", "stopped")})
    except FileNotFoundError:
        pass
    except Exception as exc:        # present but unparseable -> don't silently lose; flag it
        print(f"[cfg] download_state.json unreadable ({exc!r}); starting with none")
    for friendly in DOWNLOAD_STATE:
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        with contextlib.suppress(Exception):
            DOWNLOAD_PROGRESS[friendly] = {"downloaded": _hf_cache_bytes(target), "total": 0}


def save_download_state() -> None:
    # Atomic write (temp + os.replace) so a crash/concurrent writer can't truncate the
    # one file whose corruption would defeat the whole persistence feature.
    try:
        tmp = DOWNLOAD_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(DOWNLOAD_STATE, fh, indent=2)
        os.replace(tmp, DOWNLOAD_STATE_PATH)
    except Exception as exc:
        print(f"[cfg] could not save download_state.json: {exc!r}")


# The controller's weight-SERVING helpers (build/stream per-stage safetensors blobs for /weights,
# /experts, /weights_tp) live in shards.py now — split out of this file (#38) to shrink it. It's a
# pure controller-only module (stdlib + safetensors/torch lazily + wire._tp_hetsplit), listed in
# EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync. CONVERGENCE BRIDGE (same as
# placement.py): if the import fails (the old server.py swapped in this version before shards.py
# propagated), fetch it to disk once then import — no import-time crash loop.
try:
    import shards as _shards   # noqa: F401
except Exception:
    _sh_src = _fetch_repo_file("shards.py")
    if _sh_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shards.py"), "wb") as _sh_f:
            _sh_f.write(_sh_src)
from shards import (_weight_map, _text_prefix, _head_key, _plan_weight_stream,
                    _plan_experts_chunk, _plan_experts_chunk_fused,
                    _build_weight_tp_blob, _fp8_dequant_part_bytes,
                    _nvfp4_dequant_part_bytes,
                    validate_arch_supported)   # noqa: E402,F401
# code-split Inc 9: the compile/pack family lives in shard_compile.py (SHARED module, both
# fleets' EXTRA_UPDATE_FILES; bind-free so the /compile_shards subprocess can import it bare).
try:
    import shard_compile as _shard_compile   # noqa: F401
except Exception:
    _sc_src = _fetch_repo_file("shard_compile.py")
    if _sc_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shard_compile.py"), "wb") as _sc_f:
            _sc_f.write(_sc_src)
from shard_compile import (compile_shards, verify_shard_cache, shard_cache_status,
                           cache_unit_path)   # noqa: E402,F401


# #shard-cache Inc 2 (serve-from-cache): a compiled int4 cache is served byte-for-byte as PRE-PACKED
# int4, so a corrupt cache = silent wrong logits. Full-sha verify before trusting it for a load, but
# memoize on the manifest's (mtime, size) so the tens-of-GB sha read runs once per (re)compile, not
# per load. {(model_dir, quant): ((mtime_ns, size), ok)}.
_CACHE_VERIFY_MEMO: dict = {}


def _shard_cache_ok(model_dir: str, quant: str) -> bool:
    """True iff a <quant> shard cache exists for this model AND passes a full sha256 integrity check.
    Memoized on the manifest's mtime+size (a recompile rewrites manifest.json -> new sig -> re-verify).
    Any miss/corruption -> False so the load transparently streams bf16 instead."""
    try:
        mf = os.path.join(model_dir, "_shards", quant, "manifest.json")
        stt = os.stat(mf)
    except OSError:
        return False
    key = (model_dir, quant)
    sig = (stt.st_mtime_ns, stt.st_size)
    hit = _CACHE_VERIFY_MEMO.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    ok, problems = verify_shard_cache(model_dir, quant)
    if not ok:
        log_activity(f"shard cache {quant} for {os.path.basename(model_dir)} FAILED verify "
                     f"({'; '.join(problems[:2])}) -> serving bf16")
    _CACHE_VERIFY_MEMO[key] = (sig, ok)
    return ok


# ---------------------------------------------------------------------------
# Engine: control links + load orchestration + networked generation
# ---------------------------------------------------------------------------

# code-split Inc 2: ControlLink lives in control_plane.py now (VERBATIM; back-imported).


@dataclass
class LoadedModel:
    friendly: str
    target_id: str
    spec: ModelSpec
    ctx: int
    plan: PlanResult
    stage_node_ids: list[str]
    tokenizer: object
    eos_ids: set
    loaded_at: float
    quant: str = "none"   # the quant this model was loaded with, so an auto-reload keeps it
    kv_quant: str = "none"  # #172 TurboQuant KV-cache preset (none|turbo2|turbo3|turbo4); shown on the card
    kv_offload: bool = False  # #kv-offload: KV cache in system RAM (OffloadedCache) instead of VRAM
    # #load-temp: per-model DEFAULT sampling temperature — applied when a request doesn't send one
    # (explicit request values always win). None = unset (requests keep the global 0.0 default).
    default_temperature: Optional[float] = None
    # #min-p: per-model DEFAULT min-p sampling floor (drop tokens with p < min_p x top-token p;
    # confidence-adaptive, pairs with high default temperature — useful band 0.05-0.1 at temp>=1).
    # Same precedence: only applied when the request sends no min_p. None = unset (0 = off).
    default_min_p: Optional[float] = None
    # #runtime-knobs: the REST of the runtime-mutable sampling defaults, one dict so signatures
    # stay stable as knobs accrue — top_p / top_k / repeat_penalty / repeat_last_n /
    # presence_penalty / frequency_penalty / seed / num_predict. Set via POST /model_config
    # (runtime, no reload; key absent = unset). Same precedence as temperature: the serving layer
    # reads a key only when the request itself doesn't carry that knob.
    sampling_defaults: dict = field(default_factory=dict)
    tp_size: int = 1      # tensor-parallel width (1 = pipeline/single-node); set by _load_tp_locked.
                          # Surfaced on the card + used by #88 /reconfigure (managed-reload to/from TP).
    stage0_writer: Optional[asyncio.StreamWriter] = None  # per-model pipeline conn (controller -> first stage)
    # #stage0-stale-reconnect: how to RE-dial stage 0 (host, port) — saved at load so the controller can
    # rebuild a stale/half-open stage0_writer WITHOUT consulting the (mutable) node registry. last_send_ts
    # = wall-clock of the last frame the controller pushed to stage 0. The controller's stage0_writer is
    # opened at LOAD then sits IDLE until the first generate; an idle socket can go SILENTLY half-open
    # (the write SUCCEEDS but bytes never arrive -> no logits -> ~600s GEN_TIMEOUT hang). This is the SAME
    # failure the workers already fixed for their next-hop by lazy fresh-connecting (client.py _send_next);
    # the controller's stage0 conn was the one socket still using the discredited pre-open-and-idle pattern.
    # Freshening it when stale (at generate start) is the cure (see _freshen_stage0).
    stage0_dial: tuple = ()
    last_send_ts: float = 0.0
    last_used: float = 0.0                                  # touched on each generate; LRU key (Inc 3)
    # Per-model generation lock (Inc 3b): different models run CONCURRENTLY (each holds its
    # own lock); same-model requests serialize (queue) on it. load/unload hold the engine lock.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Per-model speculative-decode draft (controller-local). Per-model so concurrent models
    # don't clobber each other's draft KV, and so each model uses its OWN draft vocab.
    draft_model: object = None
    draft_kv: object = None
    draft_id: Optional[str] = None
    # Live request counters (Inc 4 queue-depth): active = currently generating (0/1 — the
    # per-model lock serializes), queued = requests waiting on that lock.
    active: int = 0
    queued: int = 0
    # Live KV-cache depth: tokens in the current/last generation's context (prompt +
    # produced). Tracks the controller's cache_position; resets to the prompt length
    # at each prefill. Surfaced as "ctx used N / ctx" on the dashboard.
    kv_pos: int = 0
    # Live decode throughput (#46, observability only). last_tok_s = decode tok/s of the
    # most recent generation (produced tokens / wall-clock decode time); ema_tok_s = an
    # exponential moving average across generations so the dashboard read doesn't whip
    # around on a single short request. Both 0.0 until the model has decoded at least once.
    last_tok_s: float = 0.0
    ema_tok_s: float = 0.0
    # #gen-stall-watchdog: wall-clock of the last token this model emitted (any request). The
    # watchdog flags a model active>0 that hasn't produced a token for gen_stall_s as WEDGED (a
    # dead pipeline hop -> 0 tokens + idle data plane) and reclaims its leaked active slot. Seeded
    # at load so a fresh model isn't flagged before its first generate.
    last_token_ts: float = 0.0
    gen_started_ts: float = 0.0   # #active-decode-stall: gen-begin time; last_token_ts advances past it on token 1
    # #prefill-progress: controller-clock ts of the latest WORKER-reported per-layer forward
    # progress for this model (heartbeat "fwd_progress"). The watchdog's PREFILL branch counts it
    # as liveness so a slow-but-advancing prefill under GPU contention isn't reclaimed as wedged
    # (the endpoint-weather run-abort class); DECODE stays tokens-only so a dead hop reclaims fast.
    fwd_progress_ts: float = 0.0
    # Data-parallel replication (#39): `base` is the user-facing model name shared by all
    # copies; `friendly` is the unique registry key (base, then base#1, base#2 ...). Requests
    # for `base` are least-loaded / round-robin routed across its replicas. Each copy is a
    # full model on a DISJOINT node set (a worker keys shards by model_id, so two copies of
    # the same target can't share a node), adding one concurrent decode slot per replica.
    # base == "" means "not a replica" -> callers fall back to `friendly` via (base or friendly).
    base: str = ""
    replica_idx: int = 0
    # Human-readable placement basis (#65): the strategy + shape the planner used (auto
    # GPU-first / CPU-only / tensor-parallel; single node vs distributed). Set at load,
    # shown on the dashboard model card so a resident model explains where/how it landed.
    plan_basis: str = ""
    # #76 pre-load guardrail: human-readable warnings (KV spilling to RAM / weights on CPU) and
    # the raw assessment metrics, computed from the placement at load and surfaced on the card.
    load_warnings: list = field(default_factory=list)
    load_assess: dict = field(default_factory=dict)
    # Lifetime stats for the click-to-expand model-detail modal (#model-detail). load_seconds is
    # set once at load (wall-clock the distributed shard stream + placement took); the rest are
    # accumulated per served generation in the generate wrapper.
    load_seconds: float = 0.0     # how long the load itself took (shards streamed + placed)
    req_total: int = 0            # generations served over this model's lifetime (connections)
    tok_in_total: int = 0         # prompt tokens fed across all generations
    tok_out_total: int = 0        # tokens generated across all generations
    max_tok_s: float = 0.0        # peak decode tok/s ever observed for this model


# ---------------------------------------------------------------------------
# Multimodal (vision + audio + speech) encoder helpers
# ---------------------------------------------------------------------------
# The PURE multimodal helpers (#22 distributed-Omni controller side: image/audio decode + collect,
# the meta-load + tower-materialize encoders, the meta-tensor materializer, processor / feature-
# extractor caches, the audio-response encoder + speaker resolver) live in multimodal.py now — split
# out of this file (#38, step B) to shrink it. multimodal.py is a controller-only leaf module (no
# server state, never imports server), listed in EXTRA_UPDATE_FILES so the multi-file self-update
# keeps it in sync. Its ONE controller dependency, resolving a model's weights dir, is supplied by
# DEPENDENCY INJECTION (set_model_dir_resolver below) — no back-import -> no import cycle.
#
# CONVERGENCE BRIDGE (same as placement.py/shards.py/formats.py): the old server.py that swapped in
# this version may not have fetched multimodal.py yet (it wasn't in the old EXTRA_UPDATE_FILES), so if
# the import fails, pull the file to disk once then import — no import-time crash loop on the single
# self-update cycle before multimodal.py propagates everywhere.
#
# NOTE (Inc 11): the public encode entry points _encode_images / _encode_audio /
# _load_speech_components, the speech-out group, and Engine.generate_speech live in
# media_encode.py now — WITH the ENCODING idle-gate counter (canonical home there; the
# self-updater's idle lambda reads media_encode.ENCODING as a live module attribute).
# Their bodies resolve MODELS_DIR / _safe_name / HF_TOKEN / the helpers below via state.bind.
try:
    import multimodal as _multimodal   # noqa: F401
except Exception:
    _mm_src = _fetch_repo_file("multimodal.py")
    if _mm_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "multimodal.py"), "wb") as _mm_f:
            _mm_f.write(_mm_src)
import multimodal
from multimodal import (_get_tokenizer, _get_processor, _decode_image, _collect_images,
                        _audio_bytes_to_waveform, _decode_audio, _collect_audio,
                        _vlog, _materialize_meta_tensors, _pick_vision_device, _resolve_visual,
                        _vision_cfg_and_token, _load_vision_encoder, _get_image_processor,
                        _gemma4_preprocess,
                        _as_feature_tensor, _pick_merged_embeds, _pick_audio_device,
                        _load_audio_encoder, _get_audio_feature_extractor, _omni_audio_token_id,
                        _gemma4_audio_preprocess, _load_gemma4_audio_encoder, _gemma4_audio_token_ids,
                        _audio_out_lengths, _resolve_speaker, _encode_audio_response,
                        refresh_multimodal_backends,
                        _VISION_LOG,
                        _TOK_CACHE, _PROCESSOR_CACHE, _IMGPROC_CACHE, _AUDIOFE_CACHE,
                        _VISION_CACHE, _VISION_MAT, _AUDIO_CACHE, _AUDIO_MAT,
                        _OPENAI_VOICE_MAP)   # noqa: E402,F401
multimodal.set_model_dir_resolver(_controller_model_dir)
# Non-triggering local-dir lookup for the tokenizer loader: a model normalized from GGUF (and any
# downloaded model) has its tokenizer saved under models/<name>/, while the HF repo id may have NO
# usable tokenizer (a GGUF-only repo ships .gguf, not tokenizer.json). _local_model_dir returns the
# present dir WITHOUT downloading/converting (unlike _controller_model_dir), so a metadata-path
# tokenizer call can't trigger a heavy conversion.
multimodal.set_local_dir_resolver(_local_model_dir)
# Re-probe multimodal backends at startup and bust transformers' import-time PIL/torchvision/soundfile
# availability cache — so a fresh controller (or one restarted AFTER a `pip install pillow/soundfile`)
# reports the backends correctly. The per-request _ensure_backends() guard covers a dep installed while
# already running; this covers the just-restarted case without depending on the first request to warm it.
with contextlib.suppress(Exception):
    refresh_multimodal_backends()


class LoadInProgressError(RuntimeError):
    """Raised by engine.unload(None) when a blanket teardown is requested while a load is in flight —
    the /unload endpoint maps it to 409 (unload-all is refused mid-load; per-model unload still works)."""


# ---- m4c152 code-split: Engine relocated into mixin modules (see state.py) ----
# engine_load/gen/lifecycle hold Engine's methods VERBATIM; state.bind() (called in main)
# injects this module's namespace into them so the relocated bodies resolve their globals.
# Controller-only leaf modules (stdlib only, never import server), in EXTRA_UPDATE_FILES.
# CONVERGENCE BRIDGE (same as placement/shards/...): an old server.py that swaps in this
# version fetched server.py but not yet these files, so pull each from GitHub raw once if
# missing — no import-time crash loop on the single self-update cycle before they propagate.
for _csm in ("state", "engine_lifecycle", "engine_load", "engine_gen", "media_encode"):
    try:
        __import__(_csm)
    except Exception:
        _csrc = _fetch_repo_file(_csm + ".py")
        if _csrc:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), _csm + ".py"), "wb") as _cf:
                _cf.write(_csrc)
import state
import engine_load
import engine_gen
import engine_lifecycle
from engine_load import EngineLoadMixin
from engine_gen import EngineGenMixin
from engine_lifecycle import EngineLifecycleMixin
import media_encode   # code-split Inc 11: encode/speech family + canonical ENCODING home
from media_encode import (_encode_images, _encode_audio, _load_speech_components,
                          _SPEECH_MAT, EngineSpeechMixin)   # noqa: E402,F401
# Inc 11 NOTE: ENCODING is deliberately NOT back-imported — the four encode/speech mutators
# rebind it inside media_encode, and the idle lambda reads media_encode.ENCODING live. A
# from-import (or publishing it) would freeze a stale int copy and decouple the update gate.

# ---- m4c153 code-split: build_app routes relocated into register_*(app) modules (see state.py) ----
# routes_dashboard/lifecycle/api/diag hold ~57 route handlers VERBATIM; build_app() calls each
# module's register(app) to attach them. Their module globals (engine, registry, _serve, JSONResponse,
# Request …) are injected by state.bind() BEFORE build_app() runs (main publishes+binds right after
# parse_args; build_app is called later), so FastAPI can resolve the route annotations at register time.
# Controller-only leaf modules; in EXTRA_UPDATE_FILES; pull-once convergence bridge as elsewhere.
for _crm in ("routes_dashboard", "routes_lifecycle", "routes_api", "routes_diag",
             "routes_shards"):   # code-split Inc 6
    try:
        __import__(_crm)
    except Exception:
        _crsrc = _fetch_repo_file(_crm + ".py")
        if _crsrc:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), _crm + ".py"), "wb") as _crf:
                _crf.write(_crsrc)
import routes_dashboard
import routes_lifecycle
import routes_api
import routes_diag
import routes_shards
from routes_dashboard import register as routes_dashboard_register
from routes_lifecycle import register as routes_lifecycle_register
from routes_api import register as routes_api_register
from routes_diag import register as routes_diag_register
from routes_shards import register as routes_shards_register

# ---- code-split Inc 5: download/registry lifecycle relocated into downloads.py ----
# _pull_repo_interruptible + _start_download/_do_delete + the download/add_model/delete/forget +
# /api/pull + /api/delete routes, VERBATIM; attached via register(app) in build_app; bound by
# state.bind; bridged like the other route modules. The DOWNLOAD_* definitions stay in this file.
try:
    import downloads
except Exception:
    _dlsrc = _fetch_repo_file("downloads.py")
    if _dlsrc:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads.py"), "wb") as _dlf:
            _dlf.write(_dlsrc)
    import downloads
from downloads import register as downloads_register

# ---- m4c154 code-split: request-serving layer relocated into serving.py (see state.py) ----
# serving.py holds _serve VERBATIM (the Anthropic pair moved on below); back-imported here
# so the relocated routes_api resolves them via the published namespace, and bound by state.bind
# so its bodies resolve server globals. Controller-only leaf; in EXTRA_UPDATE_FILES; bridged.
try:
    import serving
except Exception:
    _svsrc = _fetch_repo_file("serving.py")
    if _svsrc:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving.py"), "wb") as _svf:
            _svf.write(_svsrc)
    import serving
from serving import _serve

# ---- code-split Inc 3: the Anthropic Messages engine relocated into serving_anthropic.py ----
# _serve_anthropic/_count_tokens_anthropic VERBATIM; back-imported here so the relocated
# routes_api resolves them via the published namespace; bound by state.bind; bridged like serving.
try:
    import serving_anthropic
except Exception:
    _sasrc = _fetch_repo_file("serving_anthropic.py")
    if _sasrc:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving_anthropic.py"), "wb") as _saf:
            _saf.write(_sasrc)
    import serving_anthropic
from serving_anthropic import _serve_anthropic, _count_tokens_anthropic

# ---- m4c155 code-split: status-building layer relocated into status.py (see state.py) ----
# build_status/_tag_entry are called by routes_dashboard/routes_api -> back-imported so they stay
# in the published namespace; bound by state.bind so their bodies resolve server globals.
try:
    import status
except Exception:
    _stsrc = _fetch_repo_file("status.py")
    if _stsrc:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.py"), "wb") as _stf:
            _stf.write(_stsrc)
    import status
from status import build_status, _tag_entry, _loading_view, _model_entry


class Engine(EngineLoadMixin, EngineGenMixin, EngineLifecycleMixin, EngineSpeechMixin):
    # Engine is composed from mixins (m4c152 code-split): EngineLoadMixin
    # (load/placement/TP/reconfigure), EngineGenMixin (prefill/decode/spec/MTP),
    # EngineLifecycleMixin (data-plane/recovery/replicas/unload), EngineSpeechMixin
    # (speech-out; media_encode.py — Inc 11, owns the ENCODING rebind). Only __init__
    # stays here. state.bind() injects the controller namespace into the mixin modules
    # at startup — see state.py.
    def __init__(self) -> None:
        self.links: dict[str, ControlLink] = {}
        # Model registry: friendly -> LoadedModel. Inc 1 holds <=1 (load still evicts
        # all before planning); Inc 3 lifts that to fit-as-many + LRU cap. req_ids are
        # globally unique, so self.pending routes logits frames regardless of model.
        self.models: dict[str, LoadedModel] = {}
        # Data-parallel dispatch state (#39): per-base round-robin cursor (tie-break among
        # equally-loaded replicas) and a "don't evict these" guard held during a multi-replica
        # load so loading copy N+1 can't evict copy N as "idle LRU".
        self._rr: dict[str, int] = {}
        self._no_evict_base: Optional[str] = None
        # #autoload-herd: friendly -> the in-flight auto-load task. Concurrent requests for the
        # same cold model (including Retry-After-honoring retries that land mid-load) AWAIT one
        # shared load instead of stacking duplicate load() calls behind the engine lock — each
        # later duplicate would acquire the lock AFTER the first load finished, see the model
        # resident, and RELOAD it (serial unload+reload churn that can kill the generation the
        # first request just started).
        self._autoload_tasks: dict[str, asyncio.Task] = {}
        self.pending: dict[int, asyncio.Future] = {}
        self.pending_model: dict[int, str] = {}   # req_id -> model target_id (so a dropped
        #   data connection fails ONLY its own models' requests, not every model's)
        # #5 replica-precise recovery: req_id -> the UNIQUE replica registry key (friendly) the
        # request was routed to. pending_model alone is target_id, which all replicas of a base
        # SHARE, so it can't tell a dead replica's in-flight request from a healthy sibling's. This
        # map lets invalidate_model + the gen-stall watchdog fail ONLY the dead replica's futures.
        self.pending_friendly: dict[int, str] = {}
        self.data_server: Optional[asyncio.AbstractServer] = None
        self.req_counter = 0
        # The engine lock guards ATOMIC mutation of engine state (self.models, self.loadings, node
        # assignments). Generation holds the PER-MODEL lock (LoadedModel.lock) instead, so different
        # models decode CONCURRENTLY while same-model requests queue. (Speculative draft is per-model.)
        # It is held only for SHORT critical sections (plan / reserve / dispatch / install) — a PIPELINE
        # load RELEASES it around the multi-minute weight-streaming gather and re-acquires after, so an
        # /unload of a DIFFERENT model AND a SECOND load can run meanwhile. Concurrent loads are kept
        # memory-safe by the reservation ledger (below); planning stays serialized by this lock, only
        # the streaming overlaps (#parallel-load, m4c48 design — m4c52 had serialized it to dodge the
        # single-progress-card clobber, now fixed by per-load cards in self.loadings).
        # CAVEAT: the TP and EMBEDDING load paths do NOT yet release the lock around their gather — they
        # hold it for their full duration, so a TP load (e.g. CPU-TP deepseek/minimax, the longest) still
        # blocks unload/2nd-load until it finishes. They stay out of the reservation ledger precisely
        # because they serialize. Parallelizing them is a follow-up (mirror the pipeline release pattern).
        self.lock = asyncio.Lock()
        # In-flight LOAD progress cards for the dashboard, keyed by reg_key (one per concurrent load) —
        # {"model","total","ready",...} | each enriched with a timer in _loading_view at /status.
        # Registered EARLY (first line of a load, under self.lock) so "is a load in progress?" is
        # answerable atomically under the lock for the whole load — NOT just after planning (the
        # unload-all TOCTOU fix: a blanket teardown checks self.loadings under self.lock).
        self.loadings: dict[str, dict] = {}
        # In-flight COMPILE (shard-cache) progress cards, keyed by "<friendly>::<quant>" — compiles run
        # CONCURRENTLY with loads and each other (own thread, bounded per-layer memory); same-target
        # dup compiles are deduped (409). Surfaced on /status alongside loadings.
        self.compiling: dict[str, dict] = {}
        # #88 in-flight reconfigure (managed reload to/from TP): {"model","from","to"} or None.
        # Surfaced on /status so the card shows "reconfiguring -> TP×N" instead of vanishing.
        self.reconfiguring: Optional[dict] = None
        # #juggler: hitless VRAM-promotion barrier. base -> asyncio.Event that is CLEAR while that
        # model is being re-placed (drained + reloaded VRAM-first) and absent otherwise. A new gen
        # request waits on a present-but-clear gate BEFORE it resolves a replica, so it transparently
        # rides the swap (the client's open connection just pauses — no reconnect) and lands on the
        # promoted copy. _juggle_lock serializes the promoter so at most one model juggles at a time.
        self._promote_gates: dict[str, asyncio.Event] = {}
        self._juggle_lock = asyncio.Lock()
        # #juggler anti-churn: base -> fleet free-VRAM (GB) measured at a promotion that landed the
        # model STILL hybrid (the fit-check was optimistic — the re-place couldn't reach full-GPU).
        # The sweep then skips re-promoting that model until the fleet has meaningfully MORE room than
        # it did then (else it would re-place it to the same spot every ~60s). Cleared on a fully-GPU
        # promotion or when room grows past the recorded level.
        self._juggle_stuck: dict[str, float] = {}
        # PARALLEL-LOAD reservation ledger: reg_key -> {node_id: {"ram": bytes, "vram": bytes}}.
        # A load reserves its planned per-node footprint under the engine lock, then RELEASES the
        # lock for the (slow) weight-streaming gather so a second load can plan + stream IN PARALLEL.
        # Every planner subtracts OTHER loads' reservations so two concurrent loads can't
        # over-provision a node — allocation is serialized ("as if one after the other") even though
        # the streaming overlaps. Cleared when the load finalizes into self.models (or fails).
        self._reservations: dict[str, dict] = {}
        # #distributed-packing: in-flight remote-pack requests. req_id -> Future (resolved by the
        # worker's POST /pack_result) and req_id -> {"bytes", "mtensors", ...} the received unit.
        self._pack_futures: dict[str, asyncio.Future] = {}
        self._pack_results: dict[str, dict] = {}
        # SAME-MODEL load dedup: reg_key -> Future resolved when an in-flight load of that key
        # finishes. A 2nd request for the SAME not-yet-resident model awaits this instead of starting
        # a duplicate load (it "queues" on the in-flight load, then serves the resident copy).
        self._loading_futures: dict[str, asyncio.Future] = {}
        # reg_key -> the asyncio.Task running the in-flight load (the OWNER's task). A force load
        # (#stuck-load-override) CANCELS this to evict a wedged load and restart fresh, instead of
        # racing a 2nd load onto the same nodes. Set by the owner in _load_impl, popped in load()'s
        # finally alongside the card/reservation/future.
        self._loading_tasks: dict[str, asyncio.Task] = {}
        # FORCED-UPDATE in progress: set by /update while it unloads + swaps code + restarts. Blocks
        # auto-load so a client request can't reload a model into the box we're tearing down (the
        # auto-load-during-update race). Cleared naturally by the restart (fresh process).
        self.updating: bool = False
        # Wall-clock of the last failed load. After a failure engine.models is empty so the
        # controller LOOKS idle and the self-updater could exit(42) mid-churn; a cool-down off this
        # keeps it out of the self-update path for a bit so a failed/retried load can't restart-loop.
        self._last_load_failure: float = 0.0


engine = Engine()
registry = Registry()
START_TIME = time.time()
ARGS: argparse.Namespace

# --- API throughput metrics (10 s rolling): tokens/s + HTTP bytes in/out ---
METRICS = {"tokens": 0, "api_in": 0, "api_out": 0}  # cumulative
_METRIC_HIST: deque = deque()  # (t, tokens, api_in, api_out)

# --- Server-measured per-node network (the CONTROLLER counts its own wire) ---
# The controller owns every socket a node talks to it on, so it meters the bytes
# itself rather than trusting a client's self-report. Perspective is the NODE's,
# to match the dashboard's ↓/↑ columns:
#   "in"  (↓) = bytes the node RECEIVED  = bytes the controller SENT to it
#   "out" (↑) = bytes the node SENT      = bytes the controller RECEIVED from it
# Topology caveat: the data path is a ring controller -> stage0 -> ... -> head ->
# controller, so the controller is physically on only the FIRST hop (to stage0)
# and the LAST (from the head). Middle hops are node-to-node and never cross the
# controller, so during decode a middle node shows only its (tiny) control-plane
# bytes here — that is the honest truth of what the controller's own wire sees.
# (At load time every node shows its full weight-serving download, controller->node.)
NODE_NET: dict[str, dict] = {}          # node_id -> {"in": cum_bytes, "out": cum_bytes}
_NODE_NET_HIST: dict[str, deque] = {}   # node_id -> deque[(t, in, out)] (10 s rolling, for rates)

# --- Persisted per-node traffic graph history (survives a controller restart) ---
# Sampled server-side at a fixed cadence and kept in a bounded ring per HOSTNAME
# (stable across reconnects). The dashboard pulls this instead of accumulating its
# own history in the browser, so a long-open tab can't pile up unbounded JS memory.
# Stored compactly as [t_ms:int, download_bps:int, upload_bps:int] (download =
# net_in, controller->node; upload = net_out, node->controller).
NET_HIST_SAMPLE_S = 2.0       # one graph point per node every 2 s
NET_HIST_MAX = 1800           # points kept per host (~1 h at 2 s)
NET_HIST_FLUSH_S = 30.0       # flush the whole history to disk this often
NET_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net_history.json")
NET_HISTORY: dict[str, deque] = {}                 # hostname -> deque[(t_ms, d_bps, u_bps)]
_NET_HIST_STATE = {"last_sample": 0.0, "last_flush": 0.0}


def load_net_history() -> None:
    """Restore the traffic graph history from disk at startup so a restart doesn't
    lose it. Timestamps are absolute ms, so the graphs simply continue."""
    try:
        with open(NET_HISTORY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[net-hist] could not load history: {exc!r}")
        return
    n = 0
    for host, pts in (data.get("hosts") or {}).items():
        dq: deque = deque(maxlen=NET_HIST_MAX)
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                dq.append((int(p[0]), int(p[1]), int(p[2])))
        if dq:
            NET_HISTORY[host] = dq
            n += len(dq)
    if n:
        print(f"[net-hist] restored {n} samples across {len(NET_HISTORY)} host(s)")


def save_net_history() -> None:
    """Atomically write the bounded history to disk (tmp + replace so a crash
    mid-write can't corrupt the file)."""
    try:
        data = {"sample_s": NET_HIST_SAMPLE_S, "cap": NET_HIST_MAX,
                "hosts": {h: list(dq) for h, dq in NET_HISTORY.items()}}
        tmp = NET_HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, NET_HISTORY_PATH)
    except Exception as exc:
        print(f"[net-hist] could not save history: {exc!r}")


# --- Persisted per-node RAM graph history (mirrors the traffic history above) ---
# Same bounded-ring / disk-flush design as NET_HISTORY, keyed by HOSTNAME. Sampled
# on the SAME cadence and flushed alongside net_history. Stored compactly as
# [t_ms:int, free_gb_tenths:int, total_gb_tenths:int] — GB scaled by 10 and kept as
# ints (one decimal place is plenty for a graph) so the JSON stays small. The ring
# cap, sample interval and flush interval are shared with the net history (NET_HIST_*).
RAM_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ram_history.json")
RAM_HISTORY: dict[str, deque] = {}                 # hostname -> deque[(t_ms, free_t, total_t)]


def load_ram_history() -> None:
    """Restore the RAM graph history from disk at startup (mirrors load_net_history)."""
    try:
        with open(RAM_HISTORY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[ram-hist] could not load history: {exc!r}")
        return
    n = 0
    for host, pts in (data.get("hosts") or {}).items():
        dq: deque = deque(maxlen=NET_HIST_MAX)
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                dq.append((int(p[0]), int(p[1]), int(p[2])))
        if dq:
            RAM_HISTORY[host] = dq
            n += len(dq)
    if n:
        print(f"[ram-hist] restored {n} samples across {len(RAM_HISTORY)} host(s)")


def save_ram_history() -> None:
    """Atomically write the bounded RAM history to disk (mirrors save_net_history)."""
    try:
        data = {"sample_s": NET_HIST_SAMPLE_S, "cap": NET_HIST_MAX,
                "hosts": {h: list(dq) for h, dq in RAM_HISTORY.items()}}
        tmp = RAM_HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, RAM_HISTORY_PATH)
    except Exception as exc:
        print(f"[ram-hist] could not save history: {exc!r}")


# --- Persisted per-node GPU VRAM graph history (mirrors the RAM history above) ---
# Same bounded-ring / disk-flush design as NET_HISTORY / RAM_HISTORY, keyed by
# HOSTNAME. Sampled on the SAME cadence and flushed alongside net/ram history, but
# ONLY for nodes that have a GPU (vram_total_gb > 0) so the dict stays small. Stored
# compactly as [t_ms:int, used_gb_tenths:int, total_gb_tenths:int] — GB scaled by 10
# and kept as ints (one decimal is plenty for a graph). The ring cap, sample interval
# and flush interval are shared with the net history (NET_HIST_*).
VRAM_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vram_history.json")
VRAM_HISTORY: dict[str, deque] = {}                 # hostname -> deque[(t_ms, used_t, total_t)]


def load_vram_history() -> None:
    """Restore the VRAM graph history from disk at startup (mirrors load_ram_history)."""
    try:
        with open(VRAM_HISTORY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[vram-hist] could not load history: {exc!r}")
        return
    n = 0
    for host, pts in (data.get("hosts") or {}).items():
        dq: deque = deque(maxlen=NET_HIST_MAX)
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                dq.append((int(p[0]), int(p[1]), int(p[2])))
        if dq:
            VRAM_HISTORY[host] = dq
            n += len(dq)
    if n:
        print(f"[vram-hist] restored {n} samples across {len(VRAM_HISTORY)} host(s)")


def save_vram_history() -> None:
    """Atomically write the bounded VRAM history to disk (mirrors save_ram_history)."""
    try:
        data = {"sample_s": NET_HIST_SAMPLE_S, "cap": NET_HIST_MAX,
                "hosts": {h: list(dq) for h, dq in VRAM_HISTORY.items()}}
        tmp = VRAM_HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, VRAM_HISTORY_PATH)
    except Exception as exc:
        print(f"[vram-hist] could not save history: {exc!r}")


# ---------------------------------------------------------------------------
# Server-rendered monitoring graphs (SVG strings built by hand) — moved to graphs.py
# ---------------------------------------------------------------------------
# The hand-rolled SVG sparkline / detail-graph renderers (_svg_esc, _fmt_hms, _fmt_bps,
# _downsample, _spark_svg, _detail_svg) + their color constants live in graphs.py now —
# split out of this file (#38, step C) to shrink it. graphs.py is a controller-only leaf
# module (stdlib only, no server state, never imports server), listed in EXTRA_UPDATE_FILES
# so the multi-file self-update keeps it in sync.
#
# The graph DATA — the bounded per-node history rings NET_HISTORY / RAM_HISTORY / VRAM_HISTORY
# — STAYS here (defined above): they're appended to by the metrics sampler and owned by the
# persistence section (load/save_*_history). They're supplied to graphs.py by DEPENDENCY
# INJECTION: set_history_sources(...) below passes the SAME dict objects the sampler mutates,
# so live data flows through the renderers without a back-import of server (no import cycle).
#
# CONVERGENCE BRIDGE (same as placement.py/shards.py/formats.py/multimodal.py): the old
# server.py that swapped in this version may not have fetched graphs.py yet (it wasn't in the
# old EXTRA_UPDATE_FILES), so if the import fails, pull the file to disk once then import — no
# import-time crash loop on the single self-update cycle before graphs.py propagates everywhere.
try:
    import graphs as _graphs   # noqa: F401
except Exception:
    _gr_src = _fetch_repo_file("graphs.py")
    if _gr_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs.py"), "wb") as _gr_f:
            _gr_f.write(_gr_src)
import graphs
from graphs import (_svg_esc, _fmt_hms, _fmt_bps, _downsample,
                    _spark_svg, _detail_svg)   # noqa: E402,F401
# Inject the history rings (defined above: NET_HISTORY/RAM_HISTORY/VRAM_HISTORY) — the SAME
# dict objects the metrics sampler appends to, so the renderers in graphs.py read live data.
graphs.set_history_sources(net=NET_HISTORY, ram=RAM_HISTORY, vram=VRAM_HISTORY)


def net_account(node_id: Optional[str], *, to_node: int = 0, from_node: int = 0) -> None:
    """Record bytes (and frames="packets") the controller SENT to (to_node) / RECEIVED
    from (from_node) a node. One call == one data-plane frame in the given direction."""
    if not node_id or (not to_node and not from_node):
        return
    c = NODE_NET.get(node_id)
    if c is None:
        c = NODE_NET[node_id] = {"in": 0, "out": 0, "in_pkts": 0, "out_pkts": 0}
    if to_node:
        c["in"] += to_node
        c["in_pkts"] = c.get("in_pkts", 0) + 1
    if from_node:
        c["out"] += from_node
        c["out_pkts"] = c.get("out_pkts", 0) + 1


def metric_rates() -> dict:
    now = time.time()
    if len(_METRIC_HIST) >= 2:
        t0, tok0, in0, out0 = _METRIC_HIST[0]
        span = now - t0
        if span > 0:
            return {"tokens_per_s": round((METRICS["tokens"] - tok0) / span, 2),
                    "api_in_bps": round((METRICS["api_in"] - in0) / span),
                    "api_out_bps": round((METRICS["api_out"] - out0) / span)}
    return {"tokens_per_s": 0.0, "api_in_bps": 0, "api_out_bps": 0}


async def metrics_sampler() -> None:
    while True:
        await asyncio.sleep(1.0)
        now = time.time()
        _METRIC_HIST.append((now, METRICS["tokens"], METRICS["api_in"], METRICS["api_out"]))
        while len(_METRIC_HIST) > 1 and now - _METRIC_HIST[0][0] > 10:
            _METRIC_HIST.popleft()
        # Server-measured per-node rates (10 s rolling): the CONTROLLER derives
        # each node's net_in/out from the bytes it counted on its own sockets.
        live = registry._nodes
        for nid, node in list(live.items()):
            c = NODE_NET.get(nid, {"in": 0, "out": 0})
            hist = _NODE_NET_HIST.setdefault(nid, deque())
            hist.append((now, c["in"], c["out"]))
            while len(hist) > 1 and now - hist[0][0] > 10:
                hist.popleft()
            span = now - hist[0][0]
            if span > 0 and len(hist) >= 2:
                node.net_in_bps = (c["in"] - hist[0][1]) / span
                node.net_out_bps = (c["out"] - hist[0][2]) / span
            else:
                node.net_in_bps = node.net_out_bps = 0.0
        for nid in list(_NODE_NET_HIST):       # forget nodes that have left
            if nid not in live:
                _NODE_NET_HIST.pop(nid, None)
                NODE_NET.pop(nid, None)
        # Persisted graph history: one point per live node every NET_HIST_SAMPLE_S,
        # keyed by hostname; flushed to disk every NET_HIST_FLUSH_S. Bounded ring.
        st = _NET_HIST_STATE
        if now - st["last_sample"] >= NET_HIST_SAMPLE_S:
            st["last_sample"] = now
            t_ms = int(now * 1000)
            for node in live.values():
                dq = NET_HISTORY.get(node.hostname)
                if dq is None:
                    dq = NET_HISTORY[node.hostname] = deque(maxlen=NET_HIST_MAX)
                dq.append((t_ms, int(node.net_in_bps), int(node.net_out_bps)))
                # RAM graph: free/total GB scaled by 10 (one decimal) and kept as ints.
                rq = RAM_HISTORY.get(node.hostname)
                if rq is None:
                    rq = RAM_HISTORY[node.hostname] = deque(maxlen=NET_HIST_MAX)
                rq.append((t_ms, int(round(node.free_mem_gb * 10)),
                           int(round(node.total_mem_gb * 10))))
                # VRAM graph: GPU nodes only (vram_total_gb > 0) so the dict stays
                # small; used/total GB scaled by 10 and kept as ints.
                if node.vram_total_gb > 0:
                    vq = VRAM_HISTORY.get(node.hostname)
                    if vq is None:
                        vq = VRAM_HISTORY[node.hostname] = deque(maxlen=NET_HIST_MAX)
                    vq.append((t_ms, int(round(node.vram_used_gb * 10)),
                               int(round(node.vram_total_gb * 10))))
            if now - st["last_flush"] >= NET_HIST_FLUSH_S:
                st["last_flush"] = now
                await asyncio.to_thread(save_net_history)
                await asyncio.to_thread(save_ram_history)
                await asyncio.to_thread(save_vram_history)


# code-split Inc 2: the resilient TCP listener (_ResilientServer/_resilient_serve) and the
# control plane (_resolve_pending / handle_control / reaper_loop / gen_stall_watchdog) live
# in control_plane.py now (bodies VERBATIM; back-imported via the bridge near _read_frame's
# old home; bound by state.bind).


# ---------------------------------------------------------------------------
# Ollama-compatible helpers
# ---------------------------------------------------------------------------
# The PURE format/helper functions (Ollama tag/model-info formatting, detokenization safety, and the
# Anthropic Messages API / tool-calling / mRoPE / token-estimation helpers) live in formats.py now —
# split out of this file (#38, step A) to shrink it. formats.py is a controller-only leaf module
# (stdlib + placement.ModelSpec, no server state, never imports server), listed in EXTRA_UPDATE_FILES
# so the multi-file self-update keeps it in sync. CONVERGENCE BRIDGE (same as placement.py/shards.py):
# the old server.py that swapped in this version may not have fetched formats.py yet (it wasn't in the
# old EXTRA_UPDATE_FILES), so if the import fails, pull the file to disk once then import — no
# import-time crash loop on the single self-update cycle before formats.py propagates everywhere.
try:
    import formats as _formats   # noqa: F401
except Exception:
    _fmt_src = _fetch_repo_file("formats.py")
    if _fmt_src:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "formats.py"), "wb") as _fmt_f:
            _fmt_f.write(_fmt_src)
from formats import (_iso, _digest, _human_params, _details, _model_info,
                     _to_id_list, _safe_decode, _decode_visible,
                     _parse_params, _parse_tool_calls, _strip_reasoning, _tool_instruction,
                     _anth_id, _anth_flatten, _anthropic_messages_to_chat,
                     _expand_image_placeholders, _mrope_position_ids, _audio_position_ids,
                     _anthropic_tools_to_hf, _tool_to_block, _extract_tools,
                     _partial_suffix_len, _segment_tools, _estimate_tokens)   # noqa: E402,F401


def build_app() -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Resilient accept loop (NOT asyncio.start_server): its internal Proactor
        # accept loop silently dies on a transient WinError 64 during a worker
        # reconnect storm, leaving serve_forever() awaiting forever with 0 nodes
        # accepted. _resilient_serve owns a per-accept try/except so the listener
        # survives. ctrl exposes .close()/.wait_closed() and ctrl.task is the
        # accept loop task, so the shutdown block below is unchanged.
        ctrl = await _resilient_serve(ARGS.host, ARGS.control_port, handle_control, "control")
        await engine.ensure_data_listener()
        serve = ctrl.task
        reaper = asyncio.create_task(reaper_loop())
        sampler = asyncio.create_task(metrics_sampler())
        stall_wd = asyncio.create_task(gen_stall_watchdog())   # #gen-stall-watchdog: reclaim wedged-gen slots
        async def _idle_unload_loop():
            # #idle-unload: unload any model with NO requests for > idle_unload_m minutes
            # (ENGINE_CONFIG knob, dashboard "Idle unload"; 0 = the default = loaded forever).
            # Read fresh each cycle so /config applies without a restart. Safety gates:
            #   - active/queued request -> never touched (last_used is stamped at request START,
            #     so a generation LONGER than a small threshold would otherwise look idle);
            #   - idleness counts from the freshest signal (max of last_used / last_token_ts);
            #   - 📌 pinned (persist_models) models are exempt — pinning means "keep resident".
            while True:
                await asyncio.sleep(60)
                try:
                    _im = float(ENGINE_CONFIG.get("idle_unload_m", 0) or 0)
                except (TypeError, ValueError):
                    _im = 0.0
                if _im <= 0:
                    continue
                now = time.time()
                pinned = set(ENGINE_CONFIG.get("persist_models") or {})
                pinned |= set(ENGINE_CONFIG.get("no_unload_models") or {})   # #no-unload: absolute veto
                # GROUP-wise judgment (review #idle-unload): engine.unload(base) CASCADES to every
                # data-parallel replica (base + base#N), and the serving path stamps last_used on
                # the BASE at arrival while the routed REPLICA carries active/last_token_ts — so
                # judging any single registry key can call a group "idle" while a sibling is
                # mid-decode. A group is stale only when EVERY member is quiet. lock.locked()
                # covers requests that don't bump active/queued (engine.embed holds model.lock
                # for its whole forward). Residual: a speech request's talker/vocoder TAIL longer
                # than the whole window (thinker steps stamp last_token_ts; the tail restamps once).
                groups: dict = {}
                for fr, m in list(engine.models.items()):
                    groups.setdefault(getattr(m, "base", fr) or fr, []).append(m)
                stale = []
                for base, ms in groups.items():
                    if base in pinned:
                        continue
                    if any((m.active or 0) > 0 or (m.queued or 0) > 0
                           or (getattr(m, "lock", None) is not None and m.lock.locked())
                           for m in ms):
                        continue
                    last = max(max(m.last_used or 0.0, getattr(m, "last_token_ts", 0.0) or 0.0)
                               for m in ms)
                    if now - last > _im * 60.0:
                        stale.append(base)
                for fr in stale:
                    log_activity(f"idle-unload {fr}: no requests for > {_im:g} min (idle_unload_m)")
                    with contextlib.suppress(Exception):
                        await engine.unload(fr)
                # #juggler: an idle-unload just freed VRAM — try promoting the hottest hybrid model
                # to VRAM-only (background: it drains + re-places, must not block the reaper loop).
                if stale and ENGINE_CONFIG.get("juggler", False):
                    asyncio.create_task(engine._maybe_juggle("idle-unload"))
        idle_unloader = asyncio.create_task(_idle_unload_loop())
        async def _juggler_sweep_loop():
            # #juggler periodic sweep: independent of idle-unload, check every ~60s whether a hybrid
            # (GPU+RAM) model can now be promoted to VRAM-only — so a promotion fires when VRAM frees
            # for ANY reason (a manual unload, a shrinking KV, an earlier promotion completing), not
            # only right after an idle-unload. No-op when juggler is off; _maybe_juggle self-guards,
            # serializes on _juggle_lock (so it can't collide with an idle-unload-triggered run), and
            # stays quiet when nothing is promotable. juggle_sweep_s tunes the cadence (default 60s).
            while True:
                await asyncio.sleep(max(10.0, float(ENGINE_CONFIG.get("juggle_sweep_s", 60.0) or 60.0)))
                if ENGINE_CONFIG.get("juggler", False):
                    with contextlib.suppress(Exception):
                        await engine._maybe_juggle("periodic sweep")
        juggler_sweeper = asyncio.create_task(_juggler_sweep_loop())
        async def _persist_reload():
            # #77: re-load the PERSISTED models on startup so a resident model survives a controller
            # restart/crash/deploy without a manual reload (workers drop their shards on link loss, so
            # recovery = re-stream). Wait for capable workers + a short settle so GPU models land on
            # the GPU, not CPU (the restart-timing trap). Idempotent / dedup-safe vs auto-load traffic.
            persist = dict(ENGINE_CONFIG.get("persist_models") or {})
            if not persist:
                return
            _t0 = time.time()                         # #autostart-delay: client-connect grace clock
            for _ in range(40):                       # up to ~60s for the fleet to come back up
                if any(n.can_infer for n in registry.alive_sorted()):
                    break
                await asyncio.sleep(1.5)
            # Wait until the GPU pool is actually REPORTED and has stopped growing before we
            # place. A fixed timer fired too early (GPU workers reconnect slower than CPU nodes
            # after a restart), so auto-placement saw little/no VRAM and spilled GPU models to
            # CPU — the restart-timing trap that left qwen3 on CPU + timing out after a /restart.
            # Settle when the GPU VRAM pool is stable for ~2 polls; if NO GPU ever shows up
            # (CPU-only fleet) bail after ~9s; hard cap ~75s either way.
            def _gpu_pool_gb() -> float:
                return sum(n.eff_vram_gb for n in registry.alive_sorted()
                           if n.can_infer and n.eff_vram_gb > 0)
            prev, stable = -1.0, 0
            for _ in range(50):                       # cap ~75s
                gpu = _gpu_pool_gb()
                stable = stable + 1 if abs(gpu - prev) < 0.5 else 0
                prev = gpu
                if gpu > 0 and stable >= 2:           # GPU reported & steady -> place now
                    break
                if gpu <= 0 and stable >= 6:          # ~9s with no GPU at all -> CPU-only fleet
                    break
                await asyncio.sleep(1.5)
            log_activity(f"persist: fleet settled (gpu_pool={prev:.1f} GB) — reloading "
                         f"{len(persist)} persisted model(s)")
            # #autostart-delay: give API clients time to (re)connect before we get busy streaming
            # weights — hold until at least autostart_delay_s of wall-clock has elapsed since startup
            # (ON TOP of the fleet-settle wait above). 0 disables the extra wait.
            _delay = float(ENGINE_CONFIG.get("autostart_delay_s", 60.0) or 0.0)
            _remain = _delay - (time.time() - _t0)
            if _remain > 0:
                log_activity(f"persist: autostart-delay — waiting {_remain:.0f}s more for "
                             f"clients to connect before reloading")
                await asyncio.sleep(_remain)
            for name, p in persist.items():
                if name in engine.models:
                    continue
                try:
                    log_activity(f"persist: auto-reloading {name} on startup "
                                 f"(ctx={p.get('ctx', 0)}, quant={p.get('quant', 'none')})")
                    await engine.load(name, int(p.get("ctx", 0) or 0),
                                      quant=(p.get("quant") or "none"))
                except Exception as exc:
                    log_activity(f"persist: auto-reload {name} FAILED ({exc!r})")
        persist_reloader = asyncio.create_task(_persist_reload())
        # "Ready to update" = no load/download/encode IN PROGRESS — a RESIDENT model no longer blocks
        # (user policy: don't defer if something is loaded; download, apply, restart NOW, dropping
        # in-flight gens which the controller re-streams). The engine.lock check stays essential: a
        # load that is mid-flight (copying weights, planning, awaiting worker 'ready') hasn't populated
        # engine.models yet, so without it a self-update would fire mid-load and reset it — exactly the
        # restart that killed a 426 GB MiniMax load. load/unload both hold engine.lock, so locked()==busy.
        updater = asyncio.create_task(
            _self_update_loop("server.py",
                              lambda: not DOWNLOADING and not media_encode.ENCODING
                              and not engine.lock.locked()
                              # a PARALLEL load releases engine.lock during its streaming gather, so
                              # lock+models can momentarily look idle mid-load — these in-flight
                              # ledgers keep the self-updater from exit(42)-ing during a load.
                              and not engine._reservations and not engine._loading_futures
                              # in-flight loads / shard-compiles defer the self-update — a compile now
                              # runs in a SUBPROCESS (m4c85) so it holds NO lock/ledger; without this
                              # an idle self-update (fires ~120s after any git push) restarts straight
                              # through it and kills the compile (the curl drops with exit 56).
                              and not engine.loadings and not engine.compiling
                              and (time.time() - engine._last_load_failure > 120)))
        print(f"[*] control plane on {ARGS.host}:{ARGS.control_port}")
        print(f"[*] dashboard + Ollama API: http://{_display_host()}:{ARGS.http_port}/")
        try:
            yield
        finally:
            # Graceful, NON-blocking shutdown. wait_closed() would otherwise hang
            # forever on Windows/py3.12+ waiting for the live worker connections
            # (each parked in readline()), so we drop those connections first and
            # time-box the wait — this is what makes Ctrl-C actually quit.
            print("\n[*] shutting down controller…")
            for t in (serve, reaper, sampler, updater, idle_unloader, juggler_sweeper, persist_reloader):
                t.cancel()
            ctrl.close()
            with contextlib.suppress(Exception):
                if engine.data_server is not None:
                    engine.data_server.close()
            for link in list(engine.links.values()):   # free parked readline()s
                with contextlib.suppress(Exception):
                    link.writer.close()
            for m in engine.models.values():       # close every loaded model's pipeline conn
                if m.stage0_writer is not None:
                    with contextlib.suppress(Exception):
                        m.stage0_writer.close()
            for t in (serve, reaper, sampler, updater):
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await t
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ctrl.wait_closed(), timeout=2.0)
            with contextlib.suppress(Exception):
                save_net_history()   # persist the latest traffic graph on the way out
            with contextlib.suppress(Exception):
                save_ram_history()   # ...and the RAM graph alongside it
            with contextlib.suppress(Exception):
                save_vram_history()  # ...and the GPU VRAM graph alongside it
            print("[*] controller stopped")

    app = FastAPI(title="InfiniteModel Controller", version=VERSION, lifespan=lifespan)
    app.add_middleware(_ClientAccounting)   # #connections: per-client byte/activity accounting

    @app.middleware("http")            # #error-log: capture every 4xx/5xx response for the Logs UI
    async def _capture_errors(request, call_next):
        resp = await call_next(request)
        try:
            if int(getattr(resp, "status_code", 200)) >= 400:
                detail = ""
                body = getattr(resp, "body", None)   # JSONResponse/Response have .body; streams don't
                if body:
                    try:
                        d = json.loads(bytes(body).decode("utf-8", "ignore"))
                        detail = d.get("error") or d.get("detail") or ""
                        if not isinstance(detail, str):
                            detail = json.dumps(detail)
                    except Exception:
                        detail = bytes(body).decode("utf-8", "ignore")[:200]
                ip = request.client.host if request.client else "?"
                log_error(request.method, request.url.path, resp.status_code, ip, detail)
        except Exception:
            pass
        return resp

    # m4c153 code-split: attach relocated routes (see routes_*.py / state.py)
    routes_dashboard_register(app)
    routes_lifecycle_register(app)
    routes_shards_register(app)   # code-split Inc 6: shard/pack/weights routes
    routes_api_register(app)
    routes_diag_register(app)
    downloads_register(app)

    # code-split Inc 1: _serve_embed + /api/embed + /api/embeddings + /v1/embeddings
    # relocated VERBATIM to routes_api.py (attached via routes_api_register above).

    # code-split Inc 5: _start_download/_do_delete + the download/add_model/delete/forget +
    # /api/pull + /api/delete routes live in downloads.py (attached via downloads_register
    # above); /nodeconfig + /nodeconfig_all live in routes_api.py.
    return app


def _display_host() -> str:
    return platform.node() if ARGS.host in ("0.0.0.0", "") else ARGS.host


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

# Dashboard + bandwidth HTML live in dashboard_html.py (server-only); kept in sync by the
# multi-file self-update (in EXTRA_UPDATE_FILES) + present from a fresh git clone -> plain import.
from dashboard_html import DASHBOARD_HTML, BANDWIDTH_HTML, CONFIG_HTML, LOGS_HTML, CHAT_HTML   # noqa: F401


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _cfg = wire.load_config()   # central config.json (ports/hosts) — single source of truth
    p = argparse.ArgumentParser(description="InfiniteModel controller (server + dashboard).")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    p.add_argument("--http-port", type=int, default=_cfg["http_port"],
                   help="HTTP/Ollama API + dashboard port (default from config.json)")
    p.add_argument("--control-port", type=int, default=_cfg["control_port"],
                   help="worker control-plane TCP port (default from config.json)")
    p.add_argument("--data-port", type=int, default=_cfg["data_port"],
                   help="controller data-plane port for returning logits (default from config.json)")
    p.add_argument("--os-reserve-gb", type=float, default=2.0,
                   help="memory each node leaves for its OS (default 2.0)")
    p.add_argument("--self-test-plan", action="store_true",
                   help="run the partition planner against a synthetic fleet and exit")
    p.add_argument("--fleet", default="16,8,16,32",
                   help="comma-separated raw GB per box for --self-test-plan")
    return p.parse_args()


def main() -> None:
    global ARGS
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    install_log_tee()   # #logs: mirror stdout/stderr into a ring buffer served by GET /logs
    ARGS = parse_args()
    # m4c152 code-split: register server's namespace and inject it into the relocated Engine
    # mixin modules so their (verbatim) bodies resolve their former globals. Must run before
    # any engine method is exercised (self-test or serving). See state.py.
    state.publish(globals())
    state.bind(engine_load, engine_gen, engine_lifecycle, control_plane, media_encode,
               routes_dashboard, routes_lifecycle, routes_api, routes_diag, serving,
               serving_anthropic, status, downloads, routes_shards)
    print(f"InfiniteModel controller {VERSION}")
    if HF_TOKEN:
        print(f"[hf] auth token loaded (...{HF_TOKEN[-4:]}) — model pulls authenticated")
    else:
        print("[hf] NO token — anonymous pulls (rate-limited, no gated repos); "
              "set HF_TOKEN env or create hf_token.txt beside server.py")
    load_node_config()
    if NODE_CONFIG:
        print(f"[cfg] loaded per-node tier config for {len(NODE_CONFIG)} node(s)")
    load_engine_config()
    print(f"[cfg] max_loaded={ENGINE_CONFIG['max_loaded']} auto_unload={ENGINE_CONFIG['auto_unload']}")
    load_custom_models()
    if CUSTOM_MODELS:
        print(f"[cfg] loaded {len(CUSTOM_MODELS)} user-added model(s): {', '.join(CUSTOM_MODELS)}")
    load_deleted_models()    # hide built-ins the user deleted (filter AFTER MODELS is fully seeded)
    if DELETED_MODELS:
        print(f"[cfg] {len(DELETED_MODELS)} deleted model(s) hidden from the list: "
              f"{', '.join(sorted(DELETED_MODELS))}")
    load_download_state()    # restore paused/stopped intents (no auto-resume — user-driven)
    if DOWNLOAD_STATE:
        print("[cfg] halted downloads (resume from cache when ready): "
              + ", ".join(f"{k}={v}" for k, v in DOWNLOAD_STATE.items()))
    load_net_history()
    load_ram_history()
    load_vram_history()
    if ARGS.self_test_plan:
        gb_list = [float(x) for x in ARGS.fleet.split(",") if x.strip()]
        run_self_test_plan(ARGS.os_reserve_gb, gb_list, ctxs=[8192, 32768])
        return
    # ── wrong-working-directory guard (m4c17) ──────────────────────────────────
    # The controller derives MODELS_DIR / HF_HOME from its own file location, so
    # launching it from the wrong folder yields an EMPTY model list that silently
    # looks healthy (recurring foot-gun). If NONE of the registered models are
    # present on disk here, that's almost certainly the wrong directory — refuse to
    # start rather than serve a misleading, empty view. Built-ins are always
    # registered, so an empty on-disk set means "wrong dir", not "fresh install"
    # in practice; IM_ALLOW_NO_MODELS=1 overrides for a genuinely models-less box.
    on_disk = [n for n, (tgt, _d) in MODELS.items() if model_ready(tgt)]
    if MODELS and not on_disk and os.environ.get("IM_ALLOW_NO_MODELS") != "1":
        print(f"[FATAL] MODELS_DIR = {MODELS_DIR}")
        print(f"[FATAL] {len(MODELS)} model(s) registered but NONE found on disk here "
              "— this is almost certainly the WRONG working directory.")
        print("[FATAL] launch the controller from the folder that holds your models/ "
              "(e.g. D:\\infinitemodel), or set IM_ALLOW_NO_MODELS=1 to start anyway.")
        sys.stdout.flush()
        sys.exit(1)   # exit!=42 -> supervisor does NOT relaunch (server stays stopped)
    print(f"[cfg] {len(on_disk)}/{len(MODELS)} registered model(s) present on disk")
    app = build_app()
    # uvicorn handles SIGINT (sets should_exit; its 0.1s tick then runs the
    # lifespan shutdown above). The try/except is a clean-exit safety net so a
    # Ctrl-C never dumps a traceback. Press Ctrl-C twice to force-quit immediately.
    try:
        uvicorn.run(app, host=ARGS.host, port=ARGS.http_port, log_level="warning")
    except KeyboardInterrupt:
        pass   # clean Ctrl-C -> stop (supervisor sees exit 0 and does NOT relaunch)
    except Exception as exc:
        # ANY other failure forces a SUPERVISED RELAUNCH instead of a dead "press any key".
        # Notably the Windows ProactorEventLoop throws asyncio InvalidStateError from its IOCP
        # _poll during shutdown (Python 3.14) — that propagated out of uvicorn.run, crashed the
        # process with a non-42 exit, and server.bat then PAUSED (controller stayed down). exit 42
        # makes the supervisor relaunch on the current code. os._exit bypasses the (already-broken)
        # asyncio/atexit teardown so the crash can't re-trigger mid-exit.
        import traceback as _tb
        print(f"[FATAL] controller crashed: {exc!r} — relaunching via supervisor (exit 42)\n"
              f"{_tb.format_exc()}", flush=True)
        sys.stdout.flush()
        os._exit(42)


if __name__ == "__main__":
    main()
