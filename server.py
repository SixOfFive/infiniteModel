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

VERSION = "0.2-m4c126"  # version tag only; full changelog -> CHANGELOG.md
OLLAMA_API_VERSION = "0.5.4"   # version string reported on /api/version for tool compat
GB = 1024 ** 3


# Every console line is date/time-stamped so an unexpected event in the log can be
# correlated after the fact. Shadows the builtin print for THIS module only (uvicorn
# uses logging; workers have their own). log_activity()'s console echo and all the
# [load]/[+]/[!]/[load] FAILED lines pick this up automatically.
import builtins as _builtins
def print(*args, **kwargs):  # noqa: A001 — intentional builtin shadow for timestamping
    _builtins.print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)


# --- Self-update: poll GitLab for a newer server.py; when idle (no model loaded),
# swap it in and exit(42) so the supervisor (server.bat loop on Windows / systemd)
# relaunches the new code. Workers reconnect automatically. ---
import wire   # shared: cluster config (load_config) + self-update source URL. wire.py is a core file
             # present in every checkout and kept in sync via EXTRA_UPDATE_FILES.
SELF_UPDATE_POLL_S = 120   # poll the repo every 2 minutes (fast deploys; idle-gated)


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
EXTRA_UPDATE_FILES: list[str] = ["wire.py", "dashboard_html.py", "placement.py", "shards.py",
                                 "formats.py", "multimodal.py", "graphs.py", "model_store.py",
                                 "mtp_core.py",   # #91 MTP head forward (controller-only import)
                                 "config.json"]   # central cluster config — synced like a module


def _self_update_check(fname: str, is_idle, force: bool = False) -> None:
    """Multi-file self-update: fetch the primary file + EXTRA_UPDATE_FILES, and if ANY changed
    (and we're idle, OR force=True) swap ALL changed files together, then restart. Abort the whole
    cycle if ANY file fails to fetch, so the on-disk module set never goes half-updated/inconsistent.
    force=True is the dashboard/API 'Update' button: swap NOW without waiting for idle (the caller
    has already unloaded models + told workers to free RAM)."""
    here = os.path.dirname(os.path.abspath(__file__))
    files = [fname] + [f for f in EXTRA_UPDATE_FILES if f != fname]
    fetched: dict = {}
    for fn in files:
        remote = _fetch_repo_file(fn)
        if remote is None or len(remote) < 5:    # fetch failed / empty -> abort (stay consistent)
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
        print(f"[update] {changed} newer on GitLab - deferring (load/download/encode in progress)")
        return
    print(f"[update] {changed} newer on GitLab - swapping in + restarting")
    for fn in changed:                           # write all .new first, then atomic-replace each
        path = os.path.join(here, fn)
        tmp = path + ".new"
        with open(tmp, "wb") as fh:
            fh.write(fetched[fn])
        os.replace(tmp, path)
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
REQUEUE_GRACE_S = 12.0  # #stage0-stale-reconnect: when a head's return data-conn closes, wait this
                        # long before dooming the requests it was serving — a head that just FRESHENED
                        # its return socket (drop+reconnect at prefill) re-delivers their logits on the
                        # new conn within its stage compute; only a genuinely dead head stays pending.
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
    global NODE_CONFIG
    try:
        with open(NODE_CONFIG_PATH, encoding="utf-8") as fh:
            NODE_CONFIG = json.load(fh)
    except Exception:
        NODE_CONFIG = {}


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


def load_custom_models() -> None:
    global CUSTOM_MODELS
    try:
        with open(CUSTOM_MODELS_PATH, encoding="utf-8") as fh:
            CUSTOM_MODELS = json.load(fh)
    except Exception:
        CUSTOM_MODELS = {}
    for friendly, hf in CUSTOM_MODELS.items():
        MODELS.setdefault(friendly, (hf, hf))   # draft = target (no speculative)


def save_custom_models() -> None:
    try:
        with open(CUSTOM_MODELS_PATH, "w", encoding="utf-8") as fh:
            json.dump(CUSTOM_MODELS, fh, indent=2)
    except Exception as exc:
        print(f"[cfg] could not save custom models: {exc!r}")


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
                        free_disk_gb: float = 0.0) -> None:
        async with self._lock:
            n = self._nodes.get(node_id)
            if n:
                n.last_heartbeat = time.time()
                n.free_mem_gb = free_mem_gb
                n.cpu_percent = cpu_percent
                if free_disk_gb:
                    n.free_disk_gb = free_disk_gb

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
    "distribute": (False, False),  # spread across the WHOLE fleet (CPUs + GPUs)
    "spread":     (False, False),  # like distribute, but FORCE a stage on every capable node
    "proportional": (False, False),  # #78: layers across EVERY capable node PROPORTIONAL to its
    #                                  capacity (Hamilton apportionment) — for a big int4 MoE
    #                                  (MiniMax-M2) too big for the GPU-first subset.
}

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
DEFAULT_QUEUE_DEPTH = 2  # waiters allowed per model beyond the one in the slot (Ollama's
#                          OLLAMA_MAX_QUEUE defaults to 512 — we keep it shallow on purpose)

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
                       # #77 persistence: models to AUTO-RELOAD on controller startup (survives a
                       # restart/crash/deploy). reg_key -> {"ctx", "quant"}. Workers drop their shards
                       # when the controller link drops, so recovery = re-stream on startup (after the
                       # fleet settles, so GPU models land on the GPU, not CPU). Opt-in (default empty);
                       # set via /config?persist=<model> / the dashboard 📌 toggle.
                       "persist_models": {}}
# A loaded model stays resident FOREVER by default (auto_unload off): a request never unloads it,
# and a new load that doesn't fit simply fails (unload one first). The ONLY automatic unload is
# when auto_unload is on — then a model idle (no requests) for IDLE_UNLOAD_S is unloaded, and an
# idle model may be evicted to make room for a new load. The countdown is shown by the checkbox.
IDLE_UNLOAD_S = 3600.0   # 60 min idle -> auto-unload (only when auto_unload is enabled)

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


def _client_ip(req) -> str:
    """Best-effort client IP for slot/queue display (honors a proxy's X-Forwarded-For)."""
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "?"


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


def log_activity(msg: str) -> None:
    """Record a one-line 'what the controller is doing' event (newest first) + echo it."""
    ACTIVITY.appendleft({"t": round(time.time(), 1), "msg": msg})
    print(f"[activity] {msg}")


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
        return norm
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


async def _read_frame(reader: asyncio.StreamReader) -> tuple[dict, bytes, int]:
    """Return (header, payload, wire_bytes). wire_bytes is the exact number of
    bytes pulled off the socket, so the controller can meter its own data-in."""
    hdr_len = int.from_bytes(await reader.readexactly(4), "big")
    hdr = json.loads((await reader.readexactly(hdr_len)).decode())
    raw = await reader.readexactly(hdr["nbytes"]) if hdr["nbytes"] else b""
    return hdr, raw, 4 + hdr_len + len(raw)


async def _write_frame(writer: asyncio.StreamWriter, hdr: dict, raw: bytes) -> int:
    """Write a frame and return the exact wire byte count (for controller metering)."""
    hdr = {**hdr, "nbytes": len(raw)}
    hb = json.dumps(hdr).encode()
    buf = len(hb).to_bytes(4, "big") + hb + raw
    writer.write(buf)
    await writer.drain()
    return len(buf)


def _enc(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


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
# NOTE: the whole DOWNLOAD-STATE group (DOWNLOADING / DOWNLOAD_PROGRESS / DOWNLOAD_ERROR /
# DOWNLOAD_CONTROL / DOWNLOAD_STATE / DOWNLOAD_EPOCH / DOWNLOAD_STATE_PATH + load/save_download_state +
# _pull_repo_interruptible) STAYS here (below): those globals are mutated AND rebound (global
# DOWNLOAD_STATE) by the FastAPI download routes and read by the self-updater's idle lambda
# (not DOWNLOADING) — a moved global rebind would decouple from server's name (the ENCODING hazard).
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
                         gc_redundant_cache, delete_model_cache)   # noqa: E402,F401
model_store.set_hf_token_provider(lambda: HF_TOKEN)


def _pull_repo_interruptible(friendly: str, repo_id: str):
    """Download a repo's *.safetensors/*.json into the HF cache ONE FILE AT A TIME so a
    pause/stop (DOWNLOAD_CONTROL[friendly]) can interrupt it between files. Returns
    'done' (all files present), 'paused', or 'stopped'.

    Why per-file and not snapshot_download: the heavy pull runs in a thread, and a
    Python thread can't be force-killed — so the only clean interrupt point is between
    files. The control flag is checked BEFORE each file and the current file is allowed
    to finish, so resume granularity is WHOLE-FILE: every completed shard stays in the
    cache and is skipped on resume, and nothing in flight is abandoned (we never kill a
    file mid-write). The cost is that pause/stop take effect only after the current shard
    finishes — up to a couple minutes for a big one. (A hard crash mid-shard is a
    different matter: huggingface_hub 1.x discards that one partial shard's bytes and
    re-pulls it, but the already-completed shards are kept.) hf_transfer (env) still
    accelerates each individual file. If the repo listing keeps failing, fall back to a
    single (non-interruptible) snapshot_download so a download is never blocked by a flaky
    list call — pause/stop won't bite until it finishes in that rare mode."""
    from huggingface_hub import HfApi, hf_hub_download
    tok = HF_TOKEN or None
    files = None
    for attempt in range(3):                          # transient list failures are common
        try:
            files = HfApi().list_repo_files(repo_id, token=tok)
            break
        except Exception:
            if attempt == 2:
                from huggingface_hub import snapshot_download
                print(f"[model] {friendly}: repo listing failed -> non-interruptible "
                      f"snapshot_download fallback (pause/stop won't apply this run)")
                snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "*.py"], token=tok)
                return "done"
    # include *.py: trust_remote_code models (auto_map) ship their modeling/configuration code as
    # .py — without them a worker builds the native class for the model_type (wrong arch -> meta
    # tensors, e.g. MiniMax-M2 'minimax' -> lightning Text-01). #78.
    wanted = [f for f in files if f.endswith((".safetensors", ".json", ".py"))]
    for f in wanted:
        ctrl = DOWNLOAD_CONTROL.get(friendly)        # checked BETWEEN files (cheap dict read)
        if ctrl in ("pause", "stop"):
            return "paused" if ctrl == "pause" else "stopped"
        # Windows WinError 32: an AV / a cache scanner can momentarily lock the freshly-pulled
        # blob right as huggingface_hub renames its `.incomplete` -> final, killing the download
        # at ~finalize with "used by another process". The bytes ARE on disk (hf_hub keeps the
        # `.incomplete`), so the finalize just needs to be retried once the handle releases —
        # retry with backoff instead of failing the whole pull. Non-WinError-32 errors re-raise.
        for _att in range(6):
            try:
                hf_hub_download(repo_id, f, token=tok)   # instant if the shard is already cached
                break
            except OSError as exc:                   # PermissionError (WinError 32) is an OSError
                if getattr(exc, "winerror", None) != 32 or _att == 5:
                    raise
                print(f"[model] {friendly}: {f} finalize locked (WinError 32) — "
                      f"retry {_att + 1}/5 after {2 * (_att + 1)}s", flush=True)
                time.sleep(2 * (_att + 1))
    return "done"


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
ENCODING: int = 0                  # >0 while a vision/audio encode is in flight (idle-gate guard)
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
    global DOWNLOAD_STATE
    try:
        with open(DOWNLOAD_STATE_PATH, encoding="utf-8") as fh:
            DOWNLOAD_STATE = {k: v for k, v in json.load(fh).items()
                              if v in ("paused", "stopped")}
    except FileNotFoundError:
        DOWNLOAD_STATE = {}
    except Exception as exc:        # present but unparseable -> don't silently lose; flag it
        print(f"[cfg] download_state.json unreadable ({exc!r}); starting with none")
        DOWNLOAD_STATE = {}
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
                    compile_shards, verify_shard_cache, shard_cache_status,
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

@dataclass
class ControlLink:
    node_id: str
    writer: asyncio.StreamWriter
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_load: Optional[asyncio.Future] = None
    pending_unload: Optional[asyncio.Future] = None

    async def send(self, obj: dict) -> None:
        async with self.lock:
            payload = _enc(obj)
            self.writer.write(payload)
            await self.writer.drain()
            net_account(self.node_id, to_node=len(payload))  # controller -> node


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
# NOTE: the public encode entry points _encode_images / _encode_audio / _load_speech_components and
# the speech-out group (_SPEECH_CACHE / _SPEECH_MAT / _ensure_spk_dict / _materialize_from_prefix)
# STAY here (below): they mutate the ENCODING idle-gate counter the self-updater reads, and/or use
# server-only globals (MODELS_DIR / _safe_name / HF_TOKEN / shutil). They call the moved helpers,
# which resolve through this bridge import.
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
                        _as_feature_tensor, _pick_merged_embeds, _pick_audio_device,
                        _load_audio_encoder, _get_audio_feature_extractor, _omni_audio_token_id,
                        _audio_out_lengths, _resolve_speaker, _encode_audio_response,
                        _VISION_LOG,
                        _TOK_CACHE, _PROCESSOR_CACHE, _IMGPROC_CACHE, _AUDIOFE_CACHE,
                        _VISION_CACHE, _VISION_MAT, _AUDIO_CACHE, _AUDIO_MAT,
                        _OPENAI_VOICE_MAP)   # noqa: E402,F401
multimodal.set_model_dir_resolver(_controller_model_dir)


def _encode_images(target_id: str, images: list) -> dict:
    """Run the image processor + vision tower. Returns {image_embeds [N,hidden], grid_thw,
    info}. image_embeds are the per-image-token features to splice into stage-0's
    embed_tokens output at the image-placeholder positions (increment 3)."""
    import torch
    global ENCODING
    ENCODING += 1   # guard: keep the self-update idle gate closed while we encode
    try:
        t0 = time.time()
        ip = _get_image_processor(target_id)
        model, dev = _load_vision_encoder(target_id)
        t_load = time.time()
        inputs = ip(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(dev)
        grid = inputs.get("image_grid_thw")
        grid_dev = grid.to(dev) if grid is not None else None
        info: dict = {"device": dev, "pixel_values_shape": list(pv.shape),
                      "load_s": round(t_load - t0, 1)}
        visual, _prefix = _resolve_visual(model)
        t_fwd = time.time()
        with torch.inference_mode():
            # The visual tower's OWN forward runs patch_embed -> blocks -> merger and returns
            # the LM-READY MERGED tokens [prod(grid)/merge^2, out_hidden(==text_hidden)].
            # get_image_features returns only the PRE-merge backbone [patches, vision_hidden],
            # which is the wrong dim/count to splice — so call visual() directly.
            try:
                feats = visual(pv, grid_dev)
                info["path"] = "visual(pixel_values, grid_thw)"
            except Exception as exc:
                info["visual_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                feats = model.get_image_features(pixel_values=pv, image_grid_thw=grid_dev)
                info["path"] = "get_image_features(fallback)"
        info["forward_s"] = round(time.time() - t_fwd, 1)
        info["raw_return_type"] = type(feats).__name__
        # Splice the MERGED tokens (pooler_output, dim == text hidden), not the pre-merge
        # backbone (last_hidden_state). Pick by hidden dim == out_hidden. (Omni-aware: vcfg +
        # image_token_id are nested under thinker_config for Qwen2.5-Omni.)
        vcfg, image_token_id = _vision_cfg_and_token(model)
        tcfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") \
            else getattr(model.config, "text_config", model.config)
        out_hidden = (getattr(vcfg, "out_hidden_size", None) if vcfg is not None else None) \
            or getattr(tcfg, "hidden_size", None)
        emb = _pick_merged_embeds(feats, out_hidden)
        info["embeds_shape"] = list(emb.shape)
        # Per-image merged-token COUNT = prod(t,h,w) / merge^2 — used to expand each single
        # <|image_pad|> placeholder into the right run and align positions with `emb` rows.
        merge = int(getattr(vcfg, "spatial_merge_size", 1) or 1) if vcfg is not None else 1
        counts = []
        if grid is not None:
            for row in grid.tolist():
                prod = 1
                for d in row:
                    prod *= int(d)
                counts.append(prod // (merge * merge))
        print(f"[vision] encoded {len(images)} image(s) on {dev}: load={info['load_s']}s "
              f"forward={info['forward_s']}s -> {list(emb.shape)} counts={counts}")
        return {"image_embeds": emb, "grid_thw": grid, "info": info, "counts": counts,
                "image_token_id": image_token_id, "out_hidden": out_hidden, "merge": merge,
                "grid_list": (grid.tolist() if grid is not None else [])}
    finally:
        ENCODING -= 1


def _encode_audio(target_id: str, audios: list, sampling_rate: int = 16000) -> dict:
    """Run the feature extractor + Omni audio tower. `audios` is a list of 1-D float32
    waveforms at `sampling_rate` Hz. Returns {audio_embeds [total_tokens, hidden], counts
    (per-audio token counts), audio_token_id, out_hidden, info}. audio_embeds are the
    per-audio-token features to splice at <|AUDIO|> positions (increment 5c)."""
    import torch
    global ENCODING
    ENCODING += 1   # keep the self-update idle gate closed while we encode
    try:
        t0 = time.time()
        fe = _get_audio_feature_extractor(target_id)
        model, dev = _load_audio_encoder(target_id)
        t_load = time.time()
        feats_in = fe(audios, sampling_rate=sampling_rate, return_tensors="pt",
                      return_attention_mask=True)
        input_features = feats_in["input_features"].to(dev)
        # The feature extractor returns 'attention_mask'; Omni's get_audio_features expects
        # 'feature_attention_mask'. Same tensor, renamed.
        fam = feats_in.get("feature_attention_mask", feats_in.get("attention_mask"))
        fam = fam.to(dev) if fam is not None else None
        info: dict = {"device": dev, "input_features_shape": list(input_features.shape),
                      "load_s": round(t_load - t0, 1),
                      "feature_attention_mask_shape": (list(fam.shape) if fam is not None else None)}
        thinker = getattr(model, "thinker", model)
        t_fwd = time.time()
        with torch.inference_mode():
            feats = thinker.get_audio_features(input_features, feature_attention_mask=fam)
        info["forward_s"] = round(time.time() - t_fwd, 1)
        info["raw_return_type"] = type(feats).__name__
        # text hidden lives under thinker_config.text_config (NULL on thinker_config itself);
        # get_text_config() resolves it canonically.
        cfg = model.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else \
            getattr(getattr(cfg, "thinker_config", cfg), "text_config",
                    getattr(cfg, "thinker_config", cfg))
        out_hidden = getattr(text_cfg, "hidden_size", None)
        # get_audio_features returns the tower last_hidden_state (already projected to text
        # hidden) — pick the tensor whose width == text hidden.
        emb = _pick_merged_embeds(feats, out_hidden)
        # Defensive: get_audio_features mask-selects to a FLAT [total_tokens, hidden] (clips
        # concatenated in batch order), which is what the splice expects. If a transformers
        # version instead returns a batched [n_clips, seq, hidden], flatten it so shape[0] is
        # the token count (else the sum(counts)==shape[0] check below would compare against
        # n_clips and wrongly reject). Harmless when already 2D.
        if emb.dim() == 3:
            emb = emb.reshape(-1, emb.shape[-1])
            info["flattened_from_3d"] = True
        info["embeds_shape"] = list(emb.shape)
        info["embeds_dtype"] = str(emb.dtype)
        # Per-audio token counts (to expand each <|AUDIO|> placeholder in 5c) — processor
        # formula, validated to SUM to the actual embed count.
        feature_lens = fam.sum(-1) if fam is not None else None
        counts, how = (None, "unavailable")
        if feature_lens is not None:
            counts = _audio_out_lengths(feature_lens)
            how = "processor_formula"
            if sum(counts) != int(emb.shape[0]):
                print(f"[audio] WARN counts sum {sum(counts)} != embeds {emb.shape[0]} "
                      f"(feature_lens={feature_lens.tolist()})")
                how += f"(MISMATCH:{sum(counts)}vs{int(emb.shape[0])})"
        if counts is None:                       # last-ditch: single audio => all tokens are it
            counts = [int(emb.shape[0])] if len(audios) == 1 else None
            how = "embeds_total(single-audio)" if counts else how
        info["counts_how"] = how
        audio_token_id = _omni_audio_token_id(model)
        print(f"[audio] encoded {len(audios)} clip(s) on {dev}: load={info['load_s']}s "
              f"forward={info['forward_s']}s -> {list(emb.shape)} counts={counts} "
              f"audio_token_id={audio_token_id}")
        return {"audio_embeds": emb, "counts": counts, "audio_token_id": audio_token_id,
                "out_hidden": out_hidden, "info": info}
    finally:
        ENCODING -= 1


# ===================== #P6 speech-out Phase 2: Talker + token2wav loader =====================
# The Thinker runs DISTRIBUTED (its hidden states come back via the Phase-1 capture transport).
# The Talker (a codec LM) + token2wav (DiT + BigVGAN vocoder) are the ~4B "speech head"; we run
# them on the CONTROLLER (CPU by default — float32 vocoder, shared beast GPU is crash-prone).
# We meta-build the full Omni (zero memory: thinker/talker/token2wav all on meta), then
# materialize ONLY talker + token2wav + the thinker embed matrix (needed for the talker-input
# assembly: bos/eos/pad embeds + thinker_token_embeds), and load spk_dict.pt (speaker conds).
_SPEECH_CACHE: dict = {}    # target_id -> dict(model, talker, token2wav, embed, speaker_map, dev)
_SPEECH_MAT: dict = {}      # target_id -> {component: [(name, shape, how)]}


def _ensure_spk_dict(target_id: str) -> str:
    """spk_dict.pt (speaker conditioning) is NOT a *.safetensors/*.json, so _controller_model_dir
    never fetched it. Make sure it's in the model dir; download the single file if missing."""
    local = os.path.join(MODELS_DIR, _safe_name(target_id), "spk_dict.pt")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    src = None
    with contextlib.suppress(Exception):
        src = hf_hub_download(target_id, "spk_dict.pt", local_files_only=True)
    if src is None:
        src = hf_hub_download(target_id, "spk_dict.pt", token=HF_TOKEN)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    shutil.copy2(os.path.realpath(src), local)
    return local


def _materialize_from_prefix(model, module, prefix: str, files: list, dev: str, target_id: str,
                             tag: str):
    """Load `module`'s weights from the safetensors keys under `prefix` (stripped), give any
    leftover computed meta buffers real storage, and move it to `dev`. Returns the count."""
    from safetensors import safe_open
    sd = {}
    for fn in files:
        with safe_open(fn, framework="pt") as fh:
            for k in fh.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = fh.get_tensor(k)
    if not sd:
        raise RuntimeError(f"no '{prefix}*' weights found for {tag}")
    module.load_state_dict(sd, strict=False, assign=True)
    mat = _materialize_meta_tensors(module, dev)
    _SPEECH_MAT.setdefault(target_id, {})[tag] = mat
    module.to(dev)
    return len(sd), mat


SPEECH_DEVICE = os.environ.get("INFINITEMODEL_SPEECH_DEVICE", "cpu").strip().lower()


def _load_speech_components(target_id: str) -> dict:
    """Build the talker + token2wav + thinker embed matrix needed to turn captured thinker
    hidden states into a waveform. Cached.

    IMPORTANT: build talker + token2wav on a REAL device (NOT meta). Their __init__ computes
    non-persistent buffers that are NOT in the safetensors — the DiT rotary inv_freq AND the
    BigVGAN kaiser-sinc resample FILTERS. A meta-build + generic _materialize_meta_tensors
    ZERO-FILLED those filter buffers -> the vocoder's resampling convolutions output zero ->
    SILENT audio. Building real (then load_state_dict the persistent params) keeps the filters
    correct. The talker + vocoder are small (~0.5B + DiT); only the 7B thinker needs to stay
    distributed (we only pull its embed matrix for the assembly)."""
    cached = _SPEECH_CACHE.get(target_id)
    if cached is not None:
        return cached
    import torch, glob
    import torch.nn as nn
    global ENCODING
    ENCODING += 1   # heavy one-time build; hold the self-update idle gate
    try:
        from transformers import AutoConfig
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniToken2WavModel)
        from safetensors import safe_open
        t0 = time.time()
        _vlog(f"[speech] load START {target_id}")
        cfg = AutoConfig.from_pretrained(target_id)
        dev = "cpu" if SPEECH_DEVICE in ("cpu", "") else SPEECH_DEVICE
        # build REAL (buffers computed correctly), default fp32 on CPU
        talker = Qwen2_5OmniTalkerForConditionalGeneration(cfg.talker_config).eval()
        token2wav = Qwen2_5OmniToken2WavModel(cfg.token2wav_config).eval()
        _vlog(f"[speech] built talker+token2wav (real) in {time.time()-t0:.1f}s")
        model_dir = _controller_model_dir(target_id)
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

        def _load_into(module, prefix, tag):
            sd = {}
            for fn in files:
                with safe_open(fn, framework="pt") as fh:
                    for k in fh.keys():
                        if k.startswith(prefix):
                            sd[k[len(prefix):]] = fh.get_tensor(k)
            if not sd:
                raise RuntimeError(f"no '{prefix}*' weights found for {tag}")
            r = module.load_state_dict(sd, strict=False)   # copy into real params; buffers kept
            _SPEECH_MAT.setdefault(target_id, {})[tag] = {
                "loaded": len(sd), "n_missing": len(r.missing_keys),
                "missing": list(r.missing_keys)[:20],
                "n_unexpected": len(r.unexpected_keys),
                "unexpected": list(r.unexpected_keys)[:20]}
            if r.missing_keys:
                print(f"[speech] WARN {tag}: {len(r.missing_keys)} missing persistent keys "
                      f"(e.g. {list(r.missing_keys)[:3]})")
            return len(sd)
        nt = _load_into(talker, "talker.", "talker")
        nw = _load_into(token2wav, "token2wav.", "token2wav")
        talker = talker.to(dev).float()        # fp32 on CPU (bf16 CPU ops unreliable)
        token2wav = token2wav.to(dev).float()  # token2wav MUST be fp32
        # thinker embed matrix (standalone nn.Embedding) for the talker-input assembly
        ew = None
        for fn in files:
            with safe_open(fn, framework="pt") as fh:
                if "thinker.model.embed_tokens.weight" in fh.keys():
                    ew = fh.get_tensor("thinker.model.embed_tokens.weight")
                    break
        if ew is None:
            raise RuntimeError("thinker.model.embed_tokens.weight not found")
        embed = nn.Embedding(ew.shape[0], ew.shape[1])
        with torch.no_grad():
            embed.weight.copy_(ew.float())
        embed = embed.to(dev).eval()
        spk_path = _ensure_spk_dict(target_id)
        speaker_map = torch.load(spk_path, weights_only=True)
        res = {"talker": talker, "token2wav": token2wav, "embed": embed,
               "speaker_map": speaker_map, "dev": dev,
               "n_talker": nt, "n_token2wav": nw, "n_embed": 1}
        _SPEECH_CACHE[target_id] = res
        _vlog(f"[speech] READY {target_id}: talker={nt} token2wav={nw} embed={list(ew.shape)} on "
              f"{dev}; speakers={list(speaker_map.keys())}; total {time.time()-t0:.1f}s")
        return res
    finally:
        ENCODING -= 1


class LoadInProgressError(RuntimeError):
    """Raised by engine.unload(None) when a blanket teardown is requested while a load is in flight —
    the /unload endpoint maps it to 409 (unload-all is refused mid-load; per-model unload still works)."""


class Engine:
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
        self.pending: dict[int, asyncio.Future] = {}
        self.pending_model: dict[int, str] = {}   # req_id -> model target_id (so a dropped
        #   data connection fails ONLY its own models' requests, not every model's)
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

    # -- data listener: receives logits frames from the last stage --
    async def ensure_data_listener(self) -> None:
        if self.data_server is None:
            # Resilient accept loop: a transient per-accept OSError (WinError 64
            # during a reconnect storm) is logged and skipped, never killing the
            # data-plane listener. _on_data's (reader, writer) signature unchanged.
            self.data_server = await _resilient_serve(
                ARGS.host, ARGS.data_port, self._on_data, "data")
            print(f"[*] data plane listening on {ARGS.host}:{ARGS.data_port}")

    async def _on_data(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        served: set = set()   # model_ids whose logits came back on THIS connection
        _set_keepalive(writer.get_extra_info("socket"))   # survive the load->generate idle gap
        try:
            while True:
                try:
                    hdr, raw, wire = await _read_frame(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break  # connection dropped; pending futures failed in finally
                # Frames carry model_id (= HF target_id); attribute net to THAT model's head
                # node (correct with several models streaming back concurrently).
                mid = hdr.get("model_id")
                if mid is not None:
                    served.add(mid)
                hm = next((mm for mm in self.models.values()
                           if mm.target_id == mid), None)
                net_account(self._head_id(hm), from_node=wire)  # head -> controller
                rid = hdr.get("req_id")
                fut = self.pending.pop(rid, None) if rid is not None else None
                if rid is not None:
                    self.pending_model.pop(rid, None)
                if hdr.get("kind") == "error":
                    if fut and not fut.done():
                        fut.set_exception(RuntimeError(hdr.get("error", "stage error")))
                    continue
                if fut and not fut.done():
                    try:
                        if hdr.get("hid_meta") is not None:
                            # #P6 speech: two-tensor result frame = logits ++ post-norm hidden.
                            ln = int(hdr["logits_nbytes"])
                            logits = _unpack_tensor(hdr, raw[:ln])
                            hidden = _unpack_tensor(hdr["hid_meta"], raw[ln:])
                            fut.set_result((logits, hidden))
                        else:
                            fut.set_result(_unpack_tensor(hdr, raw))
                    except Exception as exc:  # malformed/short frame
                        fut.set_exception(exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover
            print(f"[data] listener error: {exc!r}")
        finally:
            # A dead data connection dooms only the requests for the model(s) THIS connection was
            # serving — fail just those (not every model's), so an unrelated model's concurrent
            # generations survive one head's hiccup.
            # #stage0-stale-reconnect: a head node now FRESHENS its return conn at each prefill (it
            # drops the possibly-idle-half-open socket + reconnects), so a return-conn close is
            # usually a RECONNECT, not a death — and the request's logits are about to arrive on the
            # NEW conn. Dooming synchronously here killed exactly that request ("data connection
            # closed" right after an idle gap). So GRACE it: snapshot the in-flight reqs this conn
            # served, then doom only those STILL pending after REQUEUE_GRACE_S. A reconnect delivers
            # within the head stage's compute (well under the grace) -> future resolves -> skipped;
            # a genuinely dead head still fails fast (grace << gen-stall watchdog / GEN_TIMEOUT).
            at_close = [rid for rid, f in self.pending.items()
                        if not f.done() and self.pending_model.get(rid) in served]
            with contextlib.suppress(Exception):
                writer.close()
            if at_close:
                async def _grace_doom(rids):
                    await asyncio.sleep(REQUEUE_GRACE_S)
                    for rid in rids:
                        fut = self.pending.get(rid)
                        if fut is not None and not fut.done():
                            self.pending.pop(rid, None)
                            self.pending_model.pop(rid, None)
                            fut.set_exception(ConnectionError(
                                "data connection closed (head did not re-deliver within grace)"))
                with contextlib.suppress(Exception):
                    asyncio.create_task(_grace_doom(at_close))

    async def _connect_retry(self, host: str, port: int, timeout: float = 30) -> asyncio.StreamWriter:
        deadline = time.time() + timeout
        while True:
            try:
                _r, w = await asyncio.open_connection(host, port)
                _set_keepalive(w.get_extra_info("socket"))   # stage0 conn idles load->generate; keep alive
                return w
            except OSError:
                if time.time() > deadline:
                    raise
                await asyncio.sleep(0.4)

    def next_req(self) -> int:
        self.req_counter += 1
        return self.req_counter

    def _stage0_id(self, model: Optional[LoadedModel]) -> Optional[str]:
        """Node the controller pushes frames TO (first stage of `model`); None if absent."""
        return model.stage_node_ids[0] if model and model.stage_node_ids else None

    def _head_id(self, model: Optional[LoadedModel]) -> Optional[str]:
        """Node the controller receives logits FROM (last stage of `model`); None if absent."""
        return model.stage_node_ids[-1] if model and model.stage_node_ids else None

    def invalidate(self, reason: str) -> None:
        """A node in the active pipeline dropped — tear loaded model(s) down. (Inc 1: all;
        Inc 3 will scope this to only the models that used the dropped node.)"""
        if self.models:
            print(f"[!] pipeline invalidated: {reason} (reload required)")
        for m in self.models.values():
            if m.stage0_writer is not None:
                with contextlib.suppress(Exception):
                    m.stage0_writer.close()
            self._unload_draft(m)   # free this model's controller-local draft
        if getattr(self, "_mtp_heads", None):
            self._mtp_heads.clear()   # #91 free all controller-resident MTP heads
        self.models.clear()
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("pipeline invalidated"))
        self.pending.clear()
        self.pending_model.clear()
        _release_ram()              # actually hand the freed draft RAM back (gc cycles + OS)

    def invalidate_model(self, friendly: str, reason: str) -> None:
        """Tear down ONE resident model (a node it used dropped). Other models keep running.
        With disjoint placement (Inc 3a) a dropped node serves exactly one model; that model's
        in-flight generation (if any) fails via the broken data connection, so this just drops
        the model's resident state and frees its controller-local draft."""
        m = self.models.pop(friendly, None)
        if m is None:
            return
        print(f"[!] {friendly} invalidated: {reason} (reload required)")
        hosts = [s.hostname for s in m.plan.stages]
        # Non-graceful eviction (a node holding a shard died) — surface WHY on the dashboard.
        # This path used to be console-only, so a model that vanished on a node OOM/drop just
        # disappeared from the UI with no explanation.
        log_activity(f"DROPPED {friendly}: {reason} — lost shard host(s) " + ", ".join(hosts))
        record_unload(friendly, reason, hosts=hosts, kind="node-loss")
        if m.stage0_writer is not None:
            with contextlib.suppress(Exception):
                m.stage0_writer.close()
        for nid in m.stage_node_ids:
            # Tell the SURVIVING workers to drop this model's shard so a reaped-then-reconnected
            # node (or the other stages) don't leak the old shard into the next reload — which then
            # mis-plans against stale free memory. The dropped node has no link; the others free it. (#47)
            link = self.links.get(nid)
            if link is not None:
                with contextlib.suppress(Exception):
                    asyncio.create_task(link.send({"type": "unload", "model_id": m.target_id}))
            nd = registry._nodes.get(nid)
            if nd:
                nd.clear_assignment()
        self._unload_draft(m)
        _release_ram()

    async def recover_node_restart(self, stale: "Node", new_node_id: str) -> int:
        """A worker RE-REGISTERED (it restarted): `stale` is its OLD node entry, still listed
        with whatever model stage(s) it held, while `new_node_id` is its fresh registration. The
        restart wiped that worker's shard + KV and dropped its data connection, but the old link
        may not have errored yet (idle Windows sockets go half-open silently), so the controller
        still thinks the model is loaded — the next generate would route to the dead stage and hang
        out the full GEN_TIMEOUT (~600s) instead of failing.

        Recovery (deliberately the CONSERVATIVE minimum, not an auto-reload): mark every resident
        model that had a stage on the stale node BROKEN and drop it, so a generate fails FAST with
        a clear "worker restarted — reload" message. invalidate_model already (a) closes the
        pipeline conn, (b) fails this model's in-flight requests, (c) tells the SURVIVING stages to
        unload their slice (so they don't leak the old shard into a reload), (d) logs a DROPPED line
        + records the unload for the dashboard, and (e) clears node assignments. We do NOT auto-
        re-stream the shards here: a flapping worker would thrash the whole pipeline, and a wrong
        recovery that tears down a healthy model is worse than the status quo. Returns the count of
        models recovered."""
        affected = [fr for fr, m in self.models.items()
                    if stale.node_id in m.stage_node_ids]
        for fr in affected:
            self.invalidate_model(
                fr, f"worker {stale.hostname} restarted (re-registered as {new_node_id}) — reload")
        # Drop the now-superseded control link to the stale node id so a later unload/heartbeat
        # can't fire on a dead writer (the fresh registration installed its own link under the new
        # id). The stale Node entry itself is removed by the caller.
        old_link = self.links.pop(stale.node_id, None)
        if old_link is not None:
            with contextlib.suppress(Exception):
                old_link.writer.close()
        return len(affected)

    def _lru_friendly(self) -> Optional[str]:
        """Least-recently-used resident model (by last_used), or None if none loaded."""
        if not self.models:
            return None
        return min(self.models.items(), key=lambda kv: kv[1].last_used)[0]

    def _lru_evictable(self) -> Optional[str]:
        """LRU resident model with NO active/queued requests (safe to auto-unload), or None
        if every resident model is busy serving. Never picks an actively-serving model."""
        idle = [(fr, m) for fr, m in self.models.items()
                if m.active == 0 and m.queued == 0
                and (m.base or m.friendly) != self._no_evict_base]
        if not idle:
            return None
        return min(idle, key=lambda kv: kv[1].last_used)[0]

    # ---- data-parallel replication (#39) -------------------------------------------------
    def replicas_of(self, base: str) -> list["LoadedModel"]:
        """All resident replicas of a user-facing model `base`, ordered by replica_idx.
        A non-replicated model (base == friendly) returns its single LoadedModel."""
        rs = [m for m in self.models.values() if (m.base or m.friendly) == base]
        rs.sort(key=lambda m: m.replica_idx)
        return rs

    def replica_count(self, base: str) -> int:
        """Number of concurrent decode slots for `base` (one per resident replica; >=1)."""
        return max(1, len(self.replicas_of(base)))

    def _pick_replica(self, base: str) -> Optional["LoadedModel"]:
        """Route a request for `base` to the least-loaded resident replica (active+queued),
        round-robin on ties — the data-parallel dispatch. Skips replicas whose pipeline
        connection is down; falls back to a direct key hit for non-replicated models."""
        rs = [r for r in self.replicas_of(base) if r.stage0_writer is not None]
        if not rs:
            return self.models.get(base)
        best = min(r.active + r.queued for r in rs)
        cands = [r for r in rs if r.active + r.queued == best]
        if len(cands) == 1:
            return cands[0]
        i = self._rr.get(base, 0) % len(cands)
        self._rr[base] = i + 1
        return cands[i]

    def _fit_ctx(self, spec, mems, ctx: int, consolidate: bool, prefer_vram: bool,
                 floor: int, spread: bool = False, proportional: bool = False):
        """Binary-search the largest context in [floor, ctx] whose pipeline plan fits `mems`.
        Used when a load won't fit at the requested ctx and there's nothing to evict — trading
        context for fit instead of OOMing. Returns (ctx_used, plan); plan.ok is False only when
        not even `floor` tokens fit (weights alone exceed the pool)."""
        floor = min(floor, ctx)
        pf = plan_pipeline(spec, mems, floor, consolidate=consolidate, prefer_vram=prefer_vram,
                           spread=spread, proportional=proportional)
        if not pf.ok:
            return floor, pf                      # even the floor doesn't fit -> caller raises
        best_ctx, best_plan = floor, pf
        lo, hi = floor + 1, ctx
        while lo <= hi:
            mid = (lo + hi) // 2
            p = plan_pipeline(spec, mems, mid, consolidate=consolidate, prefer_vram=prefer_vram,
                              spread=spread, proportional=proportional)
            if p.ok:
                best_ctx, best_plan = mid, p
                lo = mid + 1
            else:
                hi = mid - 1
        aligned = max(floor, (best_ctx // 512) * 512)   # tidy number; a smaller ctx still fits
        if aligned != best_ctx:
            pa = plan_pipeline(spec, mems, aligned, consolidate=consolidate, prefer_vram=prefer_vram,
                               spread=spread, proportional=proportional)
            if pa.ok:
                return aligned, pa
        return best_ctx, best_plan

    async def ensure_loaded(self, friendly: str, ctx: int,
                            cpu_only: bool = False, auto_load: bool = False) -> LoadedModel:
        # If the model is resident we serve it as-is (ctx/cpu_only ignored for a live model — we
        # never reload a serving model). If it is NOT resident: an inference request now AUTO-LOADS
        # it with the default AUTO placement (GPU-VRAM-first, fewest nodes; ctx<=0 -> the model's
        # native training context), instead of failing — opt-in per call site (`auto_load=True` on
        # the serving paths only, NOT metadata like /api/show) and gated by ENGINE_CONFIG `auto_load`
        # (default on; set false via /config to restore the old explicit-load-only policy). A bad
        # model name is already rejected by resolve_model_name upstream, so this only loads a KNOWN
        # registered model. A load failure (capacity/etc.) propagates to the caller as the error.
        m = self.models.get(friendly)
        if m is not None:
            m.last_used = time.time()
            return m
        if self.updating:   # forced update in progress -> don't reload into a box being torn down
            raise ValueError(f"model '{friendly}' is not loaded — controller is updating, retry shortly")
        if auto_load and ENGINE_CONFIG.get("auto_load", True):
            # #autoload-smallest: an auto-loaded (requested-but-not-resident) model defaults to the
            # SMALLEST quant — int4 — so a request never streams the full bf16 just to serve it (int4
            # is ~1/4 the memory, fits more nodes, and serves PRE-PACKED when a shard cache exists).
            # Tunable via ENGINE_CONFIG `autoload_quant` (int4|int8|none). If int4/int8 fails for a
            # model the quantizer can't handle, fall back ONCE to bf16 so the request still succeeds
            # rather than erroring out — "int4 in almost all cases", bf16 for the rest. (CancelledError
            # is a BaseException, not Exception, so a client disconnect still aborts — never retried.)
            # #auto-defaults: an auto-load uses the SAME configured defaults as the dashboard's per-model
            # Load button — quant (int4), context (8k), and placement mode — so request-triggered loads
            # and click-loads behave identically. The request's own ctx (>0) still overrides the default.
            aq = str(ENGINE_CONFIG.get("autoload_quant", "int4") or "none")
            a_ctx = int(ENGINE_CONFIG.get("autoload_ctx", DEFAULT_CTX) or 0)
            use_ctx = ctx if (ctx and ctx > 0) else a_ctx
            a_mode = str(ENGINE_CONFIG.get("autoload_mode", "auto") or "auto")
            _cons, _pv = LOAD_MODES.get(a_mode, LOAD_MODES["auto"])
            _spread, _prop = (a_mode == "spread"), (a_mode == "proportional")
            log_activity(f"{friendly}: auto-load on request (not resident) -> mode={a_mode}, "
                         f"quant={aq}, ctx={use_ctx or 'train'}" + (" (CPU-only)" if cpu_only else ""))
            try:
                return await self.load(friendly, use_ctx, consolidate=_cons, prefer_vram=_pv,
                                       quant=aq, cpu_only=cpu_only, spread=_spread, proportional=_prop)
            except Exception as e:
                if aq != "none":
                    log_activity(f"{friendly}: auto-load at {aq} failed ({e!r}) -> retry at bf16")
                    return await self.load(friendly, use_ctx, consolidate=_cons, prefer_vram=_pv,
                                           quant="none", cpu_only=cpu_only, spread=_spread,
                                           proportional=_prop)
                raise
        raise ValueError(f"model '{friendly}' is not loaded — load it first")

    def _reserved_bytes(self, exclude_key: Optional[str] = None) -> tuple[dict, dict]:
        """Sum the in-flight load reservations (RAM, VRAM) per node, EXCLUDING `exclude_key` (a
        load never reserves against itself). Returns (ram_by_node, vram_by_node) in bytes. Used by
        every planner so a load that's already reserved + streaming is subtracted from a concurrent
        load's budget -> no over-provision."""
        ram: dict[str, int] = {}
        vram: dict[str, int] = {}
        for k, res in self._reservations.items():
            if k == exclude_key:
                continue
            for nid, b in res.items():
                ram[nid] = ram.get(nid, 0) + int(b.get("ram", 0))
                vram[nid] = vram.get(nid, 0) + int(b.get("vram", 0))
        return ram, vram

    async def load(self, friendly: str, ctx: int, consolidate: bool = True,
                   prefer_vram: bool = True, quant: str = "none", tp: int = 1,
                   cpu_only: bool = False, reg_key: Optional[str] = None,
                   exclude_nodes: Optional[set] = None, replica_idx: int = 0,
                   spread: bool = False, proportional: bool = False,
                   force: bool = False, moe_offload: bool = False) -> LoadedModel:
        # Thin wrapper over _load_impl that owns the cleanup of this load's reservation + progress card
        # + in-flight future. CRITICAL (review #parallel-load): only the call that actually CLAIMED the
        # load slot for reg_key cleans up — `_own["v"]` is set True by _load_impl iff THIS call
        # registered the card (became the owner). A same-key dedup-WAITER (which just awaits the owner's
        # future and returns the resident copy) — or one that gets CANCELLED while waiting — must NOT
        # pop the owner's live reservation/card/future (doing so dropped the reservation -> over-provision,
        # cleared the card -> unload-all could nuke the loading model, and resolved the owner's future
        # early -> duplicate load). pop is synchronous (no await between _load_impl returning and the
        # pops) so no concurrent op observes both the reservation AND the now-resident model.
        rk = reg_key or friendly
        _own = {"v": False}
        try:
            return await self._load_impl(friendly, ctx, consolidate=consolidate,
                                         prefer_vram=prefer_vram, quant=quant, tp=tp,
                                         cpu_only=cpu_only, reg_key=reg_key,
                                         exclude_nodes=exclude_nodes, replica_idx=replica_idx,
                                         spread=spread, proportional=proportional, force=force,
                                         moe_offload=moe_offload, _own=_own)
        finally:
            if _own["v"]:
                self._reservations.pop(rk, None)
                self.loadings.pop(rk, None)
                self._loading_tasks.pop(rk, None)   # owner's task done -> drop the cancel handle
                _f = self._loading_futures.pop(rk, None)   # wake any same-model requests queued on us
                if _f is not None and not _f.done():
                    _f.set_result(self.models.get(rk))

    async def _load_impl(self, friendly: str, ctx: int, consolidate: bool = True,
                   prefer_vram: bool = True, quant: str = "none", tp: int = 1,
                   cpu_only: bool = False, reg_key: Optional[str] = None,
                   exclude_nodes: Optional[set] = None, replica_idx: int = 0,
                   spread: bool = False, proportional: bool = False,
                   force: bool = False, moe_offload: bool = False,
                   _own: Optional[dict] = None) -> LoadedModel:
        # self.lock guards atomic engine-state mutation; it is DROPPED around the streaming gather so a
        # 2nd load AND an unload can run meanwhile. Concurrent loads stay memory-safe via the
        # reservation ledger; planning is serialized by this lock, only the streaming overlaps.
        # Manual acquire + `_held` (not `async with`) so a CancelledError delivered at a re-acquire
        # can't make the block release a lock this task doesn't hold — with _load_lock gone a contender
        # can hold self.lock during our gather, and `async with` __aexit__ would then release THEIR lock
        # (asyncio.Lock has no owner) -> desync. _held gates every release to "only if we hold it".
        await self.lock.acquire()
        _held = True
        try:
            # `friendly` stays the user-facing/base name (spec, target, tokenizer, draft all
            # resolve from it); `reg_key` is the registry key actually stored. For a single
            # model they're equal; for a replica (#39) reg_key is "base#i" and exclude_nodes
            # holds the nodes its siblings already occupy (disjoint placement).
            reg_key = reg_key or friendly
            # FORCE OVERRIDE (#stuck-load-override): a force load while ANOTHER load of this key is in
            # flight means "that one is wedged — kill it and restart". CANCEL the in-flight owner's task
            # and AWAIT its unwind (the cancelled owner's finally frees its partial shards + reservation
            # + card + future), so we then proceed as a clean fresh load (becoming the new owner below).
            # Without this, force just raced a 2nd load onto the same nodes. We drop the lock while the
            # cancelled load tears down (it needs the lock to free shards), then re-acquire — same
            # pattern as the same-model dedup wait. force=False never does this (it queues instead).
            if force:
                _old = self._loading_tasks.get(reg_key)
                if _old is not None and _old is not asyncio.current_task() and not _old.done():
                    log_activity(f"{friendly}: force override — cancelling the wedged in-flight load "
                                 f"and restarting")
                    _old.cancel()
                    self.lock.release()
                    _held = False
                    try:
                        with contextlib.suppress(BaseException):
                            await _old
                    finally:
                        await self.lock.acquire()
                        _held = True
            # Register the progress card IMMEDIATELY — before any interleavable await — so "is a load
            # in progress?" is answerable under self.lock for the WHOLE load (the unload-all teardown
            # checks self.loadings to refuse mid-load: the TOCTOU fix). Enriched with real shard/stage
            # counts at dispatch; cleared by load()'s finally. ONLY register if no card exists for this
            # key — a same-key dedup-waiter must NOT clobber the in-flight owner's rich card (which holds
            # node_ids for reaper grace, real progress, started). The call that registers becomes the
            # OWNER (_own["v"]=True) and is the sole one that cleans up reservation/card/future.
            if reg_key not in self.loadings:
                self.loadings[reg_key] = {
                    "model": friendly, "display_model": _ollama_name(friendly),
                    "target": MODELS[friendly][0] if friendly in MODELS else friendly,
                    "ready": 0, "total": 0, "stages_total": 0, "stages_ready": 0,
                    "basis": "planning…", "warnings": [], "started": time.time()}
                if _own is not None:
                    _own["v"] = True
                    self._loading_tasks[reg_key] = asyncio.current_task()   # cancel handle for force override
            from transformers import AutoTokenizer
            spec = resolve_spec(friendly)
            if spec is None:
                raise ValueError(f"unknown model '{friendly}'")
            target_id = MODELS[friendly][0] if friendly in MODELS else friendly
            # ENCODER / sentence-embedding model: a whole-model single-node load (no pipeline/TP/KV
            # planning, no lm_head). Branch BEFORE plan_pipeline; the slim loader keys self.models
            # the same way this path does (reg_key).
            if getattr(spec, "is_embedding", False):
                return await self._load_embedding_locked(friendly, target_id, spec, reg_key,
                                                         replica_idx=replica_idx)
            # (previous pipeline connections are torn down by _unload_locked below,
            # which closes every loaded model's stage0_writer before we re-plan.)
            # controller is the model source: download the full model once so the
            # /weights endpoint can serve each worker only its slice.
            model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
            # ctx<=0 => use the model's native training context (config.json).
            ctx_was_auto = ctx <= 0          # #76: only auto-cap an AUTO ctx, never an explicit one
            if ctx <= 0:
                ctx = _train_ctx_from_dir(model_dir, spec)
                print(f"[load] ctx=auto -> training context {ctx}")
            # Idempotent re-load: a duplicate /load for an ALREADY-resident model at the SAME
            # ctx+quant — e.g. an accidental dashboard double-click — is a NO-OP that returns the
            # live copy, instead of evicting + re-streaming it (which showed up as a spurious
            # "reload"). A DIFFERENT ctx or quant still reloads (that's how you change them).
            _resident = self.models.get(reg_key)
            if (not force and _resident is not None and _resident.ctx == ctx
                    and (_resident.quant or "none") == (quant or "none")):
                log_activity(f"load {friendly}: already resident @ ctx={ctx}"
                             + (f" {quant}" if quant and quant != "none" else "")
                             + " — duplicate load ignored (no-op)")
                _resident.last_used = time.time()
                return _resident
            # SAME-MODEL concurrent load (parallel-load): a load for this key is already in flight
            # (planned + streaming with the lock released). Don't double-load — QUEUE on it: wait for
            # it to finish, then serve the now-resident copy. force= skips this (a reconfigure intends
            # to reload). If the in-flight load FAILED, fall through and attempt the load ourselves.
            if not force:
                _inflight = self._loading_futures.get(reg_key)
                if _inflight is not None:
                    log_activity(f"{friendly}: already loading — queueing this request on the "
                                 f"in-flight load")
                    self.lock.release()
                    _held = False
                    try:
                        with contextlib.suppress(Exception):
                            await _inflight
                    finally:
                        await self.lock.acquire()   # if cancelled here, _held stays False -> outer
                        _held = True                # finally won't release a lock we don't hold
                    _m = self.models.get(reg_key)
                    if _m is not None:
                        _m.last_used = time.time()
                        return _m
            # Claim this key so concurrent same-model requests queue on us (resolved by load()).
            self._loading_futures[reg_key] = asyncio.get_event_loop().create_future()
            # Plan with REAL per-layer weight sizes from the safetensors headers, so
            # MoE (and any non-dense arch) is sized correctly rather than via the
            # dense formula. No-op (returns spec) if the files can't be measured.
            spec = await asyncio.to_thread(spec_with_measurements, spec, model_dir)
            # Build-transient reserve (m4am): the worker streams each layer in as bf16 and
            # quantizes it in RAM, so a node needs ~2x ONE layer's bf16 FREE (fetch blob +
            # deserialize) on top of its resident shard, or it OOMs mid-build (a single MiniMax
            # MoE layer is ~7 GB bf16 -> ~14 GB transient, more than a tiny node's whole RAM).
            # Computed from the BF16 spec (before for_quant); reserved per node below so nodes
            # too small for the transient get 0 layers. Scales with the model: tiny for small ones.
            bf16_layer_gb = ((spec.total_weight_bytes / max(1, spec.num_layers)) / GB) if spec else 0.0
            stream_load = (tp <= 1)   # streaming build path -> incurs the per-layer bf16 transient
            # Size the plan for the quantized footprint so the planner packs int8 layers
            # into VRAM (weights ~halve; KV is unchanged). Workers get quant in the load
            # message and quantize their slice after the bf16 mmap-load.
            total_bf16_bytes = spec.total_weight_bytes if spec else 0   # bf16 READ volume (PRE-quant) for the load timeout (#100)
            spec = spec.for_quant(quant)
            # quant=none loads the model's NATIVE precision: an fp32 checkpoint loads as fp32 instead
            # of being silently downcast to bf16 (the planner already sized the measured fp32 bytes,
            # so the reservation matches). bf16/fp16 sources still load bf16; int8/int4 load bf16 then
            # quantize. (Multi-stage fp32 carries fp32 activations over the same dtype-agnostic transport.)
            load_dtype = ("float32" if (quant == "none" and (spec.src_dtype or "") == "F32")
                          else "bfloat16")
            await self.ensure_data_listener()
            log_activity(f"load {friendly}: planning (ctx={ctx}, quant={quant}"
                         + (f", tp={tp}" if tp > 1 else "")
                         + (", CPU-ONLY (RAM, no VRAM)" if cpu_only else "") + ")")

            # AUTO-TRIGGER TP (#87 D): a model whose weights SUBSTANTIALLY exceed the GPU VRAM pool
            # would run mostly on CPU as a pipeline (sequential, slow). CPU tensor-parallelism splits
            # every layer across stable Linux nodes and aggregates their RAM bandwidth -> far faster
            # per token. Auto-pick a TP width and route there — no manual tp= switch. Conservative:
            # only the DEFAULT auto mode (consolidate+prefer_vram, no explicit tp/cpu_only/spread/
            # proportional), only when weights > auto_tp_ratio x GPU pool (else pipeline keeps the GPU),
            # gated by ENGINE_CONFIG auto_tp; any failure falls through to the pipeline path below.
            auto_tp_on = bool(ENGINE_CONFIG.get("auto_tp", True))
            if (tp == 1 and not cpu_only and not spread and not proportional
                    and consolidate and prefer_vram and auto_tp_on
                    and not getattr(spec, "is_embedding", False)):
                _flaky = {"steamdeck", "tablet", "mobile", "phone"}
                stable = [n for n in registry.alive_sorted()
                          if n.can_infer and n.eff_ram_gb > 0
                          and "windows" not in (n.os or "").lower()
                          and (n.hostname or "").lower() not in _flaky]
                gpu_pool = sum(n.eff_vram_gb for n in registry.alive_sorted() if n.can_infer)
                # Largest single PIPELINE-capable node (incl. a Windows host like beast — a pipeline,
                # unlike the TP all-reduce mesh, runs fine on Windows). If the model fits ONE such node's
                # RAM, a single-node CPU pipeline (NO per-layer all-reduce) is both faster and more
                # reliable than a TP mesh whose blocking-TCP all-reduce between separate boxes is
                # latency-bound — so DON'T auto-TP it. #40 crossover bench (m4c44, this fleet): 7B
                # pipeline-on-beast 1.95 tok/s vs tp2 1.76 (0.90x); 14B 1.09 vs 0.99 (0.91x); plus TP
                # loads 1.5-3x slower and the mesh intermittently stalls. CPU-TP only pays off when no
                # single node can hold the model (forcing a slow multi-node pipeline either way).
                biggest_node_gb = max((n.eff_ram_gb for n in registry.alive_sorted() if n.can_infer),
                                      default=0.0)
                model_gb = spec.total_weight_bytes / GB
                ratio = float(ENGINE_CONFIG.get("auto_tp_ratio", 1.5))
                if model_gb > ratio * gpu_pool and model_gb > biggest_node_gb and len(stable) >= 2:
                    nh, nkv = spec.num_heads, spec.num_kv_heads
                    ng = max(1, spec.intermediate_size // 128)
                    auto_t = 1
                    for _t in range(2, min(len(stable), 8) + 1):   # largest valid width (more nodes = more bw)
                        if nh % _t == 0 and (_t <= nkv or _t % nkv == 0) and _t <= ng:
                            auto_t = _t
                    if auto_t >= 2:
                        log_activity(f"{friendly}: AUTO-TP — ~{model_gb:.0f} GB weights >> {gpu_pool:.0f} GB "
                                     f"GPU pool AND > biggest single node ({biggest_node_gb:.0f} GB, so no "
                                     f"one-node pipeline) -> CPU tensor-parallel tp={auto_t}")
                        try:
                            if reg_key in self.models:
                                await self._unload_model_locked(reg_key, "reload (auto-tp)")
                            await self._await_free_refresh()
                            return await self._load_tp_locked(friendly, target_id, spec, ctx,
                                                              auto_t, quant, cpu_only=True)
                        except Exception as exc:
                            log_activity(f"{friendly}: auto-TP failed ({exc!r}) -> pipeline fallback")

            # TENSOR-PARALLEL now COEXISTS with other resident models (#87): NO fleet-wide unload.
            # _load_tp_locked selects its own tp_nodes and the workers hold its shards alongside the
            # others (only the chosen tp_nodes' assignment is (re)set). Reloading the SAME model at a
            # new config still evicts its old copy first so a re-tp doesn't double-load it.
            if tp > 1:
                if reg_key in self.models:
                    await self._unload_model_locked(reg_key, "reload (tp)")
                await self._await_free_refresh()
                return await self._load_tp_locked(friendly, target_id, spec, ctx, tp, quant,
                                                  cpu_only=cpu_only)

            # FIT-AS-MANY + NODE-SHARING (Inc 3a/3b): keep other resident models and place this
            # one wherever there's room — INCLUDING nodes already serving a model. Each node is
            # budgeted by what's actually left: live free RAM (free_mem_gb has already dropped for
            # resident models) + VRAM minus the bytes resident shards placed on its GPU. Reloading
            # the same model evicts its old copy first; the safety cap evicts LRU models.
            max_loaded = int(ENGINE_CONFIG.get("max_loaded", MAX_LOADED_MODELS))
            auto_unload = bool(ENGINE_CONFIG.get("auto_unload", True))
            if reg_key in self.models:
                await self._unload_model_locked(reg_key, "reload")
                await self._await_free_refresh()
            # Enforce the resident-model cap by evicting IDLE models (never a busy one).
            while len(self.models) >= max_loaded:
                victim = self._lru_evictable() if auto_unload else None
                if victim is None:
                    raise RuntimeError(
                        f"at the max of {max_loaded} resident model(s) and " +
                        ("all are busy serving requests" if auto_unload else "auto-unload is off") +
                        f" — unload one before loading '{friendly}'")
                await self._unload_model_locked(victim, f"evict idle LRU (cap {max_loaded})")
                await self._await_free_refresh()
            if self.models:
                await self._await_free_refresh()   # current free RAM before budgeting vs residents

            # Plan over CAPABLE nodes, each sized by memory LEFT after resident models (so a 2nd
            # model can share a node's spare RAM/VRAM). If a node fails the load (missing deps)
            # mark it incapable and replan; if it won't fit even sharing, evict LRU and retry.
            node_by_id: dict[str, Node] = {}
            stages: list[StageAssign] = []
            n_stages = 0
            oom_skip: set[str] = set()   # nodes that failed the KV-reserve probe this load (replan w/o them)
            drop_skip: set[str] = set()  # #99: nodes that dropped their link mid-load this load (replan w/o them)
            futs: dict = {}              # last attempt's dispatch futures (pre-init so the for/else free can't NameError)
            for attempt in range(8):
                committed: dict[str, int] = {}   # node_id -> VRAM bytes held by resident shards
                for rm in self.models.values():
                    for st in rm.plan.stages:
                        # reserve BOTH a resident shard's GPU weights AND the full-ctx KV it will grow
                        # into (#vram-coexist): else a 2nd model's weights eat the 1st's KV space and
                        # OOM its decode (the qwen3-on-beast breakage). gpu_kv_bytes is worker-reported.
                        committed[st.node_id] = (committed.get(st.node_id, 0)
                                                 + st.gpu_bytes + getattr(st, "gpu_kv_bytes", 0))
                # PARALLEL LOADS: also subtract OTHER in-flight loads' reserved footprint (a load
                # that's already planned + is streaming) so this plan can't claim the same bytes ->
                # no over-provision even though the streaming overlaps (#parallel-load).
                _res_ram, _res_vram = self._reserved_bytes(exclude_key=reg_key)
                for _nid, _vb in _res_vram.items():
                    committed[_nid] = committed.get(_nid, 0) + _vb
                _vram_weights_first = bool(ENGINE_CONFIG.get("vram_weights_first", True))
                node_by_id = {}
                mems = []
                for n in registry.alive_sorted():
                    if not n.can_infer or n.node_id in oom_skip or n.node_id in drop_skip:
                        continue
                    if exclude_nodes and n.node_id in exclude_nodes:
                        continue   # a sibling replica already owns this node (disjoint placement)
                    # cpu_only: plan against RAM ONLY (VRAM=0) so the model never lands in
                    # any GPU's VRAM — the worker is also told device='cpu' below.
                    # GPU budget = the MORE CONSERVATIVE of two views, so the planner never assigns
                    # more GPU layers than the WORKER can actually place (else the worker spills the
                    # overflow to CPU -> the "free VRAM but CPU-bound / 600s-timeout" bug):
                    #   (a) tracked: usable_vram - committed (resident weights + their reserved
                    #       full-ctx KV + other in-flight loads) — protects a co-resident model's
                    #       not-yet-faulted KV (#95 coexistence).
                    #   (b) live: vram_total - vram_used (heartbeat, ALL users incl. a desktop's
                    #       browser/Discord/etc. on a shared GPU) - other in-flight reservations not
                    #       yet faulted into vram_used. This is what the worker's mem_get_info sees.
                    # usable_vram (≈ total - reserve) ignores non-fleet GPU usage, so on a desktop-
                    # shared card (beast) it over-budgets; capping by live-free spreads layers to a
                    # genuinely-free GPU node (a headless worker) instead of overloading it -> CPU.
                    live_free = max(0.0, n.vram_total_gb - n.vram_used_gb
                                    - _res_vram.get(n.node_id, 0) / GB)
                    # #vram-weights-first: budget weights against PHYSICALLY-free VRAM (live_free already
                    # excludes resident weights + actually-faulted KV + other in-flight loads), so a new
                    # model uses resident models' reserved-but-unused KV headroom instead of spilling its
                    # weights to CPU. Off -> the conservative #95 view (also subtract reserved full-ctx KV).
                    if cpu_only:
                        free_vram = 0.0
                    elif _vram_weights_first:
                        free_vram = live_free
                    else:
                        free_vram = min(
                            n.free_vram_after_resident_gb(committed.get(n.node_id, 0)), live_free)
                    # Reserve a runtime VRAM floor so a thin-headroom GPU node isn't filled to the
                    # brink (decode activations + allocator fragmentation OOM it otherwise, dropping
                    # the stage mid-generation). RAM already keeps RAM_SAFETY_GB; VRAM had none.
                    free_vram = max(0.0, free_vram - PLAN_VRAM_FLOOR_GB)
                    # Reserve the build transient from RAM (the worker streams+builds each layer in
                    # CPU RAM before quant/placement, even for GPU-bound layers). Linux uses a tmpfs
                    # mmap (~1.3x one layer); Windows/no-shm loads in-RAM (~2.3x). A node that can't
                    # fit its OS-specific transient free is excluded — it would OOM mid-build.
                    node_reserve_gb = (bf16_layer_gb * LOAD_TRANSIENT_RAM) if stream_load else 0.0
                    if quant == "int4":   # #62: per-expert streaming caps the transient (chunk + small
                        node_reserve_gb = min(node_reserve_gb, STREAM_EXPERT_RESERVE_GB)   # layer blob)
                    # #78: the controller's CO-LOCATED worker (same box) must leave RAM for the controller
                    # to read+serve the full bf16 stream (OS cache + serving buffers) WHILE this worker
                    # builds its shard — else the box over-commits and the worker OOM-drops mid-load (the
                    # beast minimax crash). data_host in _LOCAL_IPS == same machine as the controller.
                    if n.data_host in _LOCAL_IPS:
                        node_reserve_gb += CONTROLLER_RAM_RESERVE_GB
                    ram_for_resident = (n.eff_ram_gb - node_reserve_gb
                                        - _res_ram.get(n.node_id, 0) / GB)   # #parallel-load reserve
                    if ram_for_resident <= 0:
                        continue   # too small for even one layer's build transient -> skip
                    usable = ram_for_resident + free_vram   # resident RAM budget (+ VRAM after residents/floor)
                    if usable <= 0:
                        continue
                    node_by_id[n.node_id] = n
                    mems.append(NodeMem(n.node_id, n.hostname, int(usable * GB),
                                        int(free_vram * GB), pref=_mem_pref(n)))
                if not mems:
                    victim = self._lru_evictable() if auto_unload else None
                    if victim is not None:
                        await self._unload_model_locked(victim, "evict idle LRU: no room for new model")
                        await self._await_free_refresh()
                        continue
                    if self.models:
                        raise RuntimeError("no room for the new model and resident model(s) are "
                                           + ("busy serving" if auto_unload else "kept (auto-unload off)"))
                    raise RuntimeError("no capable worker nodes connected "
                                       "(all missing inference deps, or both tiers disabled)")
                pv_eff = prefer_vram and not cpu_only
                plan = plan_pipeline(spec, mems, ctx, consolidate=consolidate, prefer_vram=pv_eff,
                                     spread=spread, proportional=proportional)
                if not plan.ok:
                    victim = self._lru_evictable() if auto_unload else None
                    if victim is not None:
                        await self._unload_model_locked(victim, "evict idle LRU: new model needs room")
                        await self._await_free_refresh()
                        continue
                    # Nothing to evict: auto-fit the context DOWN to what the pool can hold
                    # alongside the weights, instead of over-committing into an OOM (user policy).
                    fit_ctx, fplan = self._fit_ctx(spec, mems, ctx, consolidate, pv_eff,
                                                   CTX_AUTOFIT_FLOOR, spread=spread,
                                                   proportional=proportional)
                    if fplan.ok and fit_ctx < ctx:
                        log_activity(f"{friendly}: ctx {ctx} won't fit the pool alongside the "
                                     f"weights — auto-fitting ctx -> {fit_ctx} to avoid OOM")
                        print(f"[load] {friendly}: auto-fit ctx {ctx} -> {fit_ctx} (pool can't hold "
                              f"full-ctx KV + weights)")
                        ctx, plan = fit_ctx, fplan
                    else:
                        raise RuntimeError((plan.error or "planning failed")
                                           + f" — even ctx {CTX_AUTOFIT_FLOOR} won't fit; the model's "
                                             "weights exceed the usable pool (free memory or use a "
                                             "smaller quant)" + (
                            "; resident model(s) busy serving" if (self.models and auto_unload)
                            else "" if self.models else ""))
                stages = plan.stages
                # #76 guardrail: estimate the VRAM/RAM split for weights + full-ctx KV on THIS
                # placement (the plan only proved it fits each node's TOTAL RAM+VRAM, not that KV
                # lands in VRAM). For an AUTO ctx, cap it so the KV stays on the GPU (the deepseek
                # 128K first-token hang); for an EXPLICIT ctx, honor it but warn. The weight-spill
                # speed warning (model bigger than fleet VRAM) is informational either way.
                assess = _assess_placement(spec, ctx, mems, stages, cpu_only=cpu_only)
                cap = capreason = None
                if ctx_was_auto:
                    if (not assess["weight_bound"] and assess["suggested_ctx"]
                            and assess["suggested_ctx"] < ctx):
                        cap = max(CTX_AUTOFIT_FLOOR, assess["suggested_ctx"])   # keep KV in VRAM
                        capreason = (f"would put ~{assess['kv_ram_gb']:.1f} GB of KV in RAM "
                                     f"(GPU VRAM can't hold it) — keeping KV on the GPU")
                    elif assess["weight_bound"] and ctx > AUTO_CTX_SLOW_CAP:
                        cap = AUTO_CTX_SLOW_CAP   # model already CPU-spilled; avoid a huge RAM KV too
                        capreason = ("exceeds the fleet's VRAM (weights spill to CPU) — avoiding a "
                                     "large full-ctx KV buffer in RAM")
                if cap and cap < ctx:
                    rplan = plan_pipeline(spec, mems, cap, consolidate=consolidate,
                                          prefer_vram=pv_eff, spread=spread,
                                          proportional=proportional)
                    if rplan.ok:
                        log_activity(f"{friendly}: ctx {ctx} {capreason} — auto-capping ctx -> {cap} "
                                     f"(pass an explicit ctx to override)")
                        print(f"[load] {friendly}: ctx-guardrail {ctx} -> {cap}")
                        ctx, plan, stages = cap, rplan, rplan.stages
                        assess = _assess_placement(spec, ctx, mems, stages, cpu_only=cpu_only)
                    else:   # capping DOWN only frees memory, so this is unreachable in practice; if
                        # it ever happens, keep the already-valid original plan + its warnings rather
                        # than aborting a loadable model — never silently proceed unlogged.
                        log_activity(f"{friendly}: wanted to cap ctx {ctx} -> {cap} but that replan "
                                     f"failed ({rplan.error}); keeping ctx {ctx} (warnings stand)")
                load_warnings = assess["warnings"]
                for _w in load_warnings:
                    log_activity(f"{friendly}: ⚠ {_w}")
                n_stages = len(stages)
                for st in stages:                  # reset only the nodes THIS model will use
                    nd = node_by_id.get(st.node_id)
                    if nd:
                        nd.clear_assignment()
                # PARALLEL LOAD: record THIS load's planned per-node footprint BEFORE the lock is
                # released for the streaming gather, so a concurrent load subtracts it. Conservative:
                # RAM build-transient for every stage (every layer builds in RAM, ~est_bytes) + VRAM
                # for a GPU stage's resident weights -> never over-provisions during the overlap.
                _resv = {}
                for st in stages:
                    _nd = node_by_id.get(st.node_id)
                    _is_gpu = bool(_nd) and (not cpu_only) and _nd.eff_vram_gb > 0
                    _resv[st.node_id] = {"ram": int(st.est_bytes),
                                         "vram": int(st.est_bytes) if _is_gpu else 0}
                self._reservations[reg_key] = _resv
                # A "shard" is one Lxx layer-slice the controller streams to a worker (plus the
                # embed/head slices) — there are MANY per load (≈ model layer count), NOT one per
                # node. The dashboard progress must count these real shards; the node ("stage")
                # count is tracked separately so "X/Y shards · A/B nodes" reads unambiguously.
                total_shards = (sum(max(0, s.layer_end - s.layer_start) for s in stages)
                                + (1 if any(s.has_embed for s in stages) else 0)
                                + (1 if any(s.has_head for s in stages) else 0))
                basis = _describe_plan(stages, node_by_id, cpu_only, pv_eff, quant)
                log_activity(f"{friendly}: plan basis → {basis}")
                log_activity(f"{friendly}: handing out {total_shards} shard(s) across "
                             f"{n_stages} node(s) -> " + ", ".join(
                    f"{s.hostname}(L{s.layer_start}-{s.layer_end})" for s in stages))
                _started0 = (self.loadings.get(reg_key) or {}).get("started")  # keep the load-start time
                self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                                "target": target_id, "total": total_shards,
                                "ready": 0, "stages_total": n_stages, "stages_ready": 0,
                                "basis": basis, "warnings": load_warnings,
                                "node_ids": [s.node_id for s in stages],   # reaper grace for builders
                                "started": _started0 or time.time()}   # #76 guardrail + load-card timer
                # #shard-cache Inc 2 (serve-from-cache): if a VERIFIED int4 cache exists, tell every
                # worker to fetch PRE-PACKED int4 layers (cache=int4) instead of streaming the full
                # bf16 + re-quantizing — the big win for MoE/large loads (e.g. ~18 GB cache vs ~70 GB
                # bf16 stream). int4 + pipeline only: TP slices weights non-contiguously (its own
                # dispatch path, never reaches here) so it can't use the whole-layer cache. The
                # controller falls back to bf16 PER UNIT if any cache file is missing, and an old
                # worker that ignores the `cache` key just streams bf16 — both safe.
                _cache_quant = ""
                if quant == "int4":
                    try:
                        _cdir = await asyncio.to_thread(_controller_model_dir, target_id)
                        if _cdir and await asyncio.to_thread(_shard_cache_ok, _cdir, "int4"):
                            _cache_quant = "int4"
                            log_activity(f"{friendly}: serving from int4 shard cache "
                                         f"(skip bf16 stream + per-layer re-quant)")
                    except Exception as _ce:
                        log_activity(f"{friendly}: shard-cache check failed ({_ce!r}) -> bf16 stream")
                        _cache_quant = ""
                futs: dict[str, asyncio.Future] = {}
                loop = asyncio.get_event_loop()
                for i, st in enumerate(stages):
                    nd = node_by_id[st.node_id]
                    if i < n_stages - 1:
                        nxt = node_by_id[stages[i + 1].node_id]
                        next_host, next_port = nxt.data_host, nxt.data_port
                    else:
                        next_host, next_port = None, ARGS.data_port  # -> controller
                    link = self.links.get(st.node_id)
                    if link is None:
                        raise RuntimeError(f"no control link to {st.node_id}")
                    nd.stage = i
                    nd.layer_start, nd.layer_end = st.layer_start, st.layer_end
                    nd.load_state = "loading"     # red on the dashboard until this shard reports ready
                    # This stage's GPU budget — MUST mirror the planner's per-node free_vram (lines above)
                    # or the worker re-clamps the weights the planner placed on GPU back to CPU (the spill
                    # bug). LIVE free VRAM = vram_total - vram_used (all users incl. a desktop's apps on a
                    # shared GPU) - other in-flight loads, matching the worker's mem_get_info. 0 -> CPU.
                    # #vram-weights-first: budget against live-free (use resident models' reserved-but-
                    # unfaulted KV headroom); off -> the conservative #95 min() (also subtract reserved KV).
                    _live_free_gb = max(0.0, nd.vram_total_gb - nd.vram_used_gb
                                        - _res_vram.get(st.node_id, 0) / GB)
                    if cpu_only:
                        _gpu_budget_gb = 0.0
                    elif _vram_weights_first:
                        _gpu_budget_gb = max(0.0, _live_free_gb - PLAN_VRAM_FLOOR_GB)
                    else:
                        _gpu_budget_gb = max(0.0, min(
                            nd.free_vram_after_resident_gb(committed.get(st.node_id, 0)),
                            _live_free_gb) - PLAN_VRAM_FLOOR_GB)
                    fut = loop.create_future()
                    link.pending_load = fut
                    futs[st.node_id] = fut
                    await link.send({
                        "type": "load", "model_id": target_id,
                        "layer_start": st.layer_start, "layer_end": st.layer_end,
                        "has_embed": st.has_embed, "has_head": st.has_head,
                        "stage": i, "num_stages": n_stages,
                        "next_host": next_host, "next_port": next_port, "dtype": load_dtype,
                        "controller_http_port": ARGS.http_port,
                        # cpu_only forces RAM placement on every shard regardless of node tier;
                        # otherwise the node's tier config decides (auto/gpu/cpu).
                        "device": "cpu" if cpu_only else nd.load_device(),
                        "gpu_budget_gb": round(_gpu_budget_gb, 3),   # #95: committed-aware GPU cap for this stage
                        "moe_offload": moe_offload,  # #moe-offload: split MoE layers (attn->GPU, experts->CPU RAM)
                        "cache": _cache_quant,       # #shard-cache Inc 2: '' | 'int4' -> fetch pre-packed cache
                        "quant": quant,              # 'none' | 'int8' (load-time choice)
                        "ctx": ctx,                  # full ctx -> worker pre-reserves KV (fail-fast)
                        # #63: this stage's planned resident bytes (quantized). The worker reserves
                        # this much RAM up front (a balloon) and consumes it shard-by-shard as layers
                        # install — fail-fast if the node can't hold its share, peak ~ the plan. It is
                        # the full stage est even for a GPU stage: every layer is built in RAM and only
                        # moved to VRAM in _place_modules at the END, so build-phase RAM == est_bytes.
                        "plan_ram_bytes": int(st.est_bytes),
                    })

                log_activity(f"{friendly}: awaiting {n_stages} shard(s) — workers fetch weights, "
                             f"then mmap-load + fuse + place: {', '.join(s.hostname for s in stages)}")

                def _ready_cb(nd):                   # flip this node green + log AS it finishes (live progress)
                    host = nd.hostname if nd else "?"
                    def cb(fut):
                        try:
                            if fut.cancelled() or fut.exception() is not None:
                                return
                            r = fut.result()
                        except Exception:
                            return
                        if nd is not None:
                            nd.load_state = "ready"   # green on the dashboard
                        _card = self.loadings.get(reg_key)   # THIS load's own card (parallel-safe)
                        if _card is not None:
                            # a NODE finished its WHOLE range -> count nodes here; the per-Lxx
                            # shard count is advanced in the /weights serve path as each slice ships.
                            _card["stages_ready"] = _card.get("stages_ready", 0) + 1
                        gpb = r.get("gpu_bytes", 0) if isinstance(r, dict) else 0
                        tot = r.get("loaded_bytes", 0) if isinstance(r, dict) else 0
                        ram = max(0, tot - gpb)
                        if gpb and ram:
                            where = f"{gpb / GB:.1f} GB GPU + {ram / GB:.1f} GB RAM"
                        elif gpb:
                            where = f"{gpb / GB:.1f} GB on GPU"
                        else:
                            where = f"{ram / GB:.1f} GB in RAM"   # CPU node — not "0 GB on GPU"
                        _plc = r.get("placement") if isinstance(r, dict) else None
                        _moe = r.get("moe") if isinstance(r, dict) else None
                        log_activity(f"  {host}: shard loaded ({where})"
                                     + (f" | {_plc}" if _plc else "")
                                     + (f" | moe={_moe}" if _moe else ""))
                    return cb
                for _nid, _fut in futs.items():
                    _fut.add_done_callback(_ready_cb(node_by_id.get(_nid)))

                # Load-ack timeout scales with the bf16 READ volume, not a flat wall: every shard
                # streams its slice of the FULL bf16 from the controller's weights drive (often a
                # slow USB drive ~150 MB/s), and that drive serves all shards, so total read time
                # ~ total_bf16 / drive_MBps regardless of node count. CRITICAL: budget the PRE-quant
                # bf16 bytes — an int4 load still STREAMS the full bf16 (the worker quantizes after),
                # so sizing on the shrunken int4 footprint timed out the 426 GB minimax int4 build
                # (#100). 35 MB/s floor + 5 min + a per-GB quantize allowance; clamp [15 min, 4 h].
                read_bytes = total_bf16_bytes or getattr(spec, "total_weight_bytes", 0) or 0
                _quant_secs = (read_bytes / GB) * (4.0 if quant in ("int4", "int8") else 0.0)
                load_timeout = int(read_bytes / (35 * 1024 * 1024)) + 300 + int(_quant_secs)
                load_timeout = max(900, min(load_timeout, 4 * 3600))
                # DROP self.lock around the (multi-minute) streaming gather and re-acquire after, so a
                # 2nd load AND an /unload of a different model can run meanwhile (#parallel-load). Loads
                # are NOT serialized — they overlap; memory-safety comes from the reservation ledger
                # (committed + _reserved_bytes already ran under the lock, dispatch is done, the
                # reservation persists so a concurrent plan subtracts it). All weights stream from the
                # controller's single drive so overlap doesn't speed an individual load, but a small/fast
                # load no longer waits behind a huge one. Safe: the gather only awaits worker futures (no
                # engine-state mutation), and an unload freeing a co-resident model only RELAXES this
                # load's budget. _held tracks lock ownership so a CancelledError
                # delivered at the re-acquire can't make the method's finally release a lock we don't hold.
                self.lock.release()
                _held = False
                try:
                    results = await asyncio.gather(
                        *[asyncio.wait_for(f, timeout=load_timeout) for f in futs.values()],
                        return_exceptions=True)
                except asyncio.CancelledError:
                    # #stuck-load-override: a force load (or shutdown) cancelled us mid-stream. Re-acquire
                    # the lock and free any shards that DID build on workers, so the cancelled load leaves
                    # nothing resident on the fleet, then re-raise so cleanup (card/reservation) proceeds.
                    await self.lock.acquire()
                    _held = True
                    with contextlib.suppress(Exception):
                        await self._free_partial_stages(target_id, futs.keys(), node_by_id)
                    raise
                finally:
                    if not _held:
                        await self.lock.acquire()
                        _held = True
                incapable: list[str] = []
                oomed: list[str] = []
                dropped: list[str] = []   # #99: nodes whose link dropped this attempt (replan on survivors)
                hard_error: Optional[str] = None
                for nid, r in zip(futs.keys(), results):
                    err = (repr(r) if isinstance(r, Exception)
                           else str(r.get("error")) if isinstance(r, dict) and r.get("type") == "error"
                           else None)
                    if err is None:
                        continue
                    if any(k in err for k in ("No module named", "ModuleNotFoundError", "ImportError")):
                        nd = node_by_id.get(nid)
                        if nd:
                            nd.can_infer = False
                            nd.incapable_reason = "missing inference deps (e.g. torch)"
                            incapable.append(nd.hostname)
                    elif any(k in err for k in ("KV_RESERVE_OOM", "out of memory",
                                                "OutOfMemoryError", "CUDA error: out of memory")):
                        # node couldn't reserve its KV — skip it for THIS load and replan (not
                        # permanently incapable; it's fine for a smaller shard/ctx later).
                        oom_skip.add(nid)
                        nd = node_by_id.get(nid)
                        oomed.append(nd.hostname if nd else nid)
                    elif any(k in err for k in ("disconnected mid-operation", "ConnectionError",
                                                "ConnectionResetError", "ConnectionAbortedError",
                                                "BrokenPipeError", "IncompleteReadError")):
                        # a worker DROPPED its link mid-load (#99) — often a silent OOM-kill (PVE
                        # OOMScoreAdjust 800 SIGKILLs before a clean KV_RESERVE_OOM). Skip it for THIS
                        # load and replan on the survivors; if it ALREADY dropped this load it's
                        # flapping -> treat as a real failure (stop churning, don't burn all 8 attempts).
                        if nid in drop_skip:
                            hard_error = f"node {nid} dropped twice during load: {err}"
                        else:
                            drop_skip.add(nid)
                            nd = node_by_id.get(nid)
                            dropped.append(nd.hostname if nd else nid)
                    else:
                        hard_error = f"node {nid} load error: {err}"
                if hard_error:
                    await self._free_partial_stages(target_id, futs.keys(), node_by_id)   # #98: free shards that DID build
                    raise RuntimeError(hard_error)
                if incapable or oomed or dropped:
                    if incapable:
                        print(f"[load] excluding incapable node(s) {incapable} (no torch); replanning")
                    if oomed:
                        print(f"[load] {oomed} failed KV-reserve (can't hold ctx={ctx}); "
                              f"replanning without them")
                    if dropped:
                        print(f"[load] {dropped} dropped their link mid-load (#99); replanning on survivors")
                    await self._free_partial_stages(target_id, futs.keys(), node_by_id)   # #98: free shards that DID build
                    continue  # retry without the incapable / OOM'd / dropped nodes
                # success: record each stage's worker-reported on-GPU bytes (size_vram), on the
                # node AND the stage (the stage copy survives node-sharing, where a 2nd model
                # would overwrite the single Node.shard_gpu_bytes).
                stage_by_id = {s.node_id: s for s in stages}
                for nid, r in zip(futs.keys(), results):
                    gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
                    gpu_kv = int(r.get("gpu_kv_bytes", 0)) if isinstance(r, dict) else 0
                    nd = node_by_id.get(nid)
                    if nd:
                        nd.shard_gpu_bytes = gpu_b
                        # Re-assert this load's per-Node assignment: a co-resident model's unload during
                        # the released-lock gather can clear it (node-sharing -> last-writer-wins), which
                        # would drop the now-resident shard to the short reaper grace. load_state ready +
                        # a non-None stage restore SERVING_GRACE_S. (plan.stages stays the real source of
                        # truth for accounting; this only fixes the mutable scalars the reaper reads.)
                        nd.load_state = "ready"
                        if nd.stage is None and nid in stage_by_id:
                            _st = stage_by_id[nid]
                            nd.stage, nd.layer_start, nd.layer_end = 0, _st.layer_start, _st.layer_end
                    if nid in stage_by_id:
                        stage_by_id[nid].gpu_bytes = gpu_b
                        stage_by_id[nid].gpu_kv_bytes = gpu_kv   # reserve this model's KV vs coexisting loads
                break  # all stages loaded
            else:
                await self._free_partial_stages(target_id, futs.keys(), node_by_id)   # #98: free any built shards
                raise RuntimeError("load failed: no capable nodes left after exclusions")

            s0 = node_by_id[stages[0].node_id]
            _s0_dial = (_dial_host(s0.data_host), s0.data_port)
            stage0_writer = await self._connect_retry(*_s0_dial)
            tok = await asyncio.to_thread(_get_tokenizer, target_id)
            eos = self._eos_ids(tok)
            now = time.time()
            lm = LoadedModel(
                reg_key, target_id, spec, ctx, plan,
                [s.node_id for s in stages], tok, eos, now,
                quant=quant, stage0_writer=stage0_writer, last_used=now,
                stage0_dial=_s0_dial, last_send_ts=now)   # #stage0-stale-reconnect: how to re-dial + freshness clock
            lm.base, lm.replica_idx = friendly, replica_idx   # data-parallel grouping (#39)
            lm.plan_basis = basis                             # placement basis (#65)
            lm.load_warnings, lm.load_assess = load_warnings, assess   # pre-load guardrail (#76)
            # #cpu-bound-visibility: the #76 assess warns from ESTIMATES; here we know the ACTUAL
            # GPU/CPU split (worker-reported gpu_bytes). If the model actually landed heavily on CPU
            # (the GPU pool was full by load time — the real cause of later multi-model loads crawling
            # and looking "busy but network-idle"), append a LOUD persistent warning + log it, so a
            # ~0.1 tok/s CPU-bound model is never mistaken for a hang.
            _vram_b = sum(s.gpu_bytes for s in stages)
            _cpu_frac = ((spec.total_weight_bytes - _vram_b) / spec.total_weight_bytes
                         if spec.total_weight_bytes else 0.0)
            if _cpu_frac > 0.30:
                _sev = "SEVERE " if _cpu_frac > 0.6 else ""
                _wmsg = (f"{_cpu_frac*100:.0f}% of weights on CPU ({_sev}— GPU pool full) -> CPU-bound, "
                         f"slow generation. Unload a model or use a smaller quant for GPU speed.")
                if _wmsg not in lm.load_warnings:
                    lm.load_warnings = list(lm.load_warnings) + [_wmsg]
                log_activity(f"{_ollama_name(friendly)}: loaded {_cpu_frac*100:.0f}% on CPU — CPU-bound "
                             f"(slow); GPU pool full. Unload a model or lower quant for full speed.")
            _st0 = (self.loadings.get(reg_key) or {}).get("started")   # #model-detail: load wall-clock
            lm.load_seconds = max(0.0, now - _st0) if _st0 else 0.0
            # speculative decoding: load THIS model's small draft locally on the controller
            draft_id = MODELS.get(friendly, (target_id, target_id))[1]
            if draft_id and draft_id != target_id:
                try:
                    await asyncio.to_thread(self._load_draft, lm, draft_id)
                    print(f"[load] draft {draft_id} on controller -> speculative decode K={SPEC_K}")
                except Exception as exc:
                    print(f"[load] draft load failed ({exc!r}); plain KV-cache decode")
                    self._unload_draft(lm)
            self.models[reg_key] = lm
            self.loadings.pop(reg_key, None)   # card off -> dashboard flips to resident (finally also pops)
            registry.dirty = False
            print(f"[load] {reg_key} across {n_stages} stages: "
                  f"{[(s.hostname, s.num_layers) for s in stages]}")
            log_activity(f"{reg_key} READY across {n_stages} stage(s) "
                         f"[{len(self.models)} model(s) resident]")
            return lm
        finally:
            # Release self.lock iff THIS task currently holds it. Replaces the `async with self.lock`
            # exit: during the gather we drop the lock, and if cancelled at the re-acquire _held is
            # False — so we must NOT release (it would free a contender's lock; asyncio.Lock has no owner).
            if _held:
                self.lock.release()

    async def _load_embedding_locked(self, friendly: str, target_id: str, spec: ModelSpec,
                                     reg_key: Optional[str] = None,
                                     replica_idx: int = 0) -> LoadedModel:
        """Slim sibling of load(): an ENCODER (sentence-embedding) model loads WHOLE onto ONE
        capable node (no pipeline/TP/KV planning, no lm_head). Mirrors load()'s control-send +
        pending_load future + stage0_writer mechanism, and stores a minimal single-stage
        LoadedModel so the dashboard / /api/ps / model card render without special-casing.
        MUST be called with self.lock held (it's reached only from load(), which holds it)."""
        reg_key = reg_key or friendly
        model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
        spec = await asyncio.to_thread(spec_with_measurements, spec, model_dir)
        tok = await asyncio.to_thread(_get_tokenizer, target_id)
        await self.ensure_data_listener()
        # Reload of the same key -> drop the old copy first.
        if reg_key in self.models:
            await self._unload_model_locked(reg_key, "reload (embedding)")
            await self._await_free_refresh()
        # Pick ONE capable node: prefer a GPU+can_infer node, else any can_infer node.
        alive = [n for n in registry.alive_sorted() if n.can_infer]
        if not alive:
            raise RuntimeError("no capable worker nodes connected for the embedding model")
        node = next((n for n in alive if n.eff_vram_gb > 0), None) or alive[0]
        log_activity(f"load {friendly}: embedding (single node {node.hostname})")
        self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id, "total": 1, "ready": 0,
                        "stages_total": 1, "stages_ready": 0,
                        "basis": f"embedding: single-node ({node.hostname})", "warnings": [],
                        "node_ids": [node.node_id],
                        "started": (self.loadings.get(reg_key) or {}).get("started") or time.time()}
        link = self.links.get(node.node_id)
        if link is None:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"no control link to {node.node_id}")
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        link.pending_load = fut
        await link.send({
            "type": "load", "kind": "embedding", "model_id": target_id,
            "controller_http_port": ARGS.http_port,
            # No next hop: the worker replies straight to the controller's data port.
            "next_host": None, "next_port": ARGS.data_port,
            "device": "cpu" if node.eff_vram_gb <= 0 else node.load_device(),
            "dtype": "float32",
        })
        try:
            r = await asyncio.wait_for(fut, timeout=GEN_TIMEOUT_S)
        except Exception as exc:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"embedding load on {node.hostname} failed: {exc!r}")
        if isinstance(r, dict) and r.get("type") == "error":
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"embedding load on {node.hostname} failed: {r.get('error')}")
        gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
        # Minimal single-stage plan so build_status / /api/ps / the card render unchanged.
        stage = StageAssign(node_id=node.node_id, hostname=node.hostname,
                            layer_start=0, layer_end=spec.num_layers,
                            has_embed=True, has_head=True,
                            est_bytes=spec.total_weight_bytes,
                            usable_bytes=int(node.usable_total_gb * GB), gpu_bytes=gpu_b)
        plan = PlanResult(ok=True, model=spec.name, ctx_len=spec.max_ctx,
                          num_layers=spec.num_layers,
                          pool_usable_gb=node.usable_total_gb,
                          required_gb=spec.total_weight_bytes / GB, stages=[stage])
        node.shard_gpu_bytes = gpu_b
        _emb_dial = (_dial_host(node.data_host), node.data_port)
        stage0_writer = await self._connect_retry(*_emb_dial)
        now = time.time()
        lm = LoadedModel(
            reg_key, target_id, spec, spec.max_ctx, plan,
            [node.node_id], tok, set(), now,
            quant="none", stage0_writer=stage0_writer, last_used=now,
            stage0_dial=_emb_dial, last_send_ts=now)
        lm.base, lm.replica_idx = friendly, replica_idx
        lm.plan_basis = "embedding: single-node"
        self.models[reg_key] = lm
        self.loadings.pop(reg_key, None)
        registry.dirty = False
        print(f"[load] {reg_key} embedding on {node.hostname} "
              f"({spec.total_weight_bytes / GB:.2f} GB)")
        log_activity(f"{reg_key} READY (embedding, {node.hostname}) "
                     f"[{len(self.models)} model(s) resident]")
        return lm

    async def embed(self, friendly: str, input_ids, attention_mask) -> list:
        """Run one encoder forward on `friendly`'s single node and return the pooled, L2-normed
        sentence vectors as a list of float lists. Mirrors _send: pack ids+mask into ONE
        two-tensor frame, await the worker's single-tensor 'embedding' reply via self.pending."""
        model = self.models[friendly]
        target = model.target_id
        async with model.lock:
            if model.stage0_writer is None:
                raise RuntimeError("embedding model not connected")
            loop = asyncio.get_event_loop()
            rid = self.next_req()
            ids_meta, ids_raw = _pack_tensor(input_ids)
            mask_meta, mask_raw = _pack_tensor(attention_mask)
            fut = loop.create_future()
            self.pending[rid] = fut
            self.pending_model[rid] = target
            try:
                hdr = {"req_id": rid, "model_id": target, "kind": "embed",
                       **ids_meta, "ids_nbytes": len(ids_raw), "mask_meta": mask_meta}
                nbytes = await _write_frame(model.stage0_writer, hdr, ids_raw + mask_raw)
                net_account(self._stage0_id(model), to_node=nbytes)   # controller -> node
                vecs = await asyncio.wait_for(fut, timeout=GEN_TIMEOUT_S)
                return vecs.tolist()
            finally:
                self.pending.pop(rid, None)
                self.pending_model.pop(rid, None)

    async def replicate(self, friendly: str, ctx: int, count: int,
                        consolidate: bool = True, prefer_vram: bool = True,
                        quant: str = "none") -> list["LoadedModel"]:
        """Load `count` full copies of `friendly` on DISJOINT node sets — the small-model
        throughput lever (#39). Replica 0 is keyed `friendly`; replica i (i>=1) is keyed
        `friendly#i`. Requests for `friendly` are then least-loaded / round-robin routed
        across all copies, so each copy adds a concurrent decode slot. Each load excludes the
        nodes its siblings already use (a worker keys shards by model_id, so two copies of one
        target cannot share a node). Stops early (keeping the copies that loaded) if the fleet
        runs out of disjoint room; raises only if even the first copy fails."""
        count = max(1, int(count))
        await self.unload(friendly)               # clean slate -> end with EXACTLY `count` copies
        out: list[LoadedModel] = []
        used: set[str] = set()
        self._no_evict_base = friendly            # protect siblings from idle-LRU eviction
        try:
            for i in range(count):
                key = friendly if i == 0 else f"{friendly}#{i}"
                try:
                    lm = await self.load(friendly, ctx, consolidate=consolidate,
                                         prefer_vram=prefer_vram, quant=quant,
                                         reg_key=key, exclude_nodes=set(used), replica_idx=i)
                except Exception as exc:
                    if i == 0:
                        raise
                    log_activity(f"replicate {friendly}: stopped at {len(out)} replica(s) "
                                 f"(no disjoint room for #{i}): {exc}")
                    break
                out.append(lm)
                used.update(lm.stage_node_ids)
                log_activity(f"replicate {friendly}: copy {i+1}/{count} on "
                             f"{[s.hostname for s in lm.plan.stages]}")
        finally:
            self._no_evict_base = None
        return out

    async def _free_partial_stages(self, target_id, node_ids, node_by_id) -> None:
        """Free shards that DID build for a distributed load that then FAILED (or got replanned without
        them): tell every node that received a stage to unload target_id + reset its dashboard state, so
        a partial load doesn't LEAK weights on the nodes that succeeded (beast leaked ~84 GB after a
        failed minimax load, #98). Idempotent with the worker's own self-clean on a failed build and with
        handle_load's top-of-load unload — a redundant {"type":"unload"} on a missing model_id pops
        safely. No LoadedModel is inserted at these failure sites, so resident-model state is untouched.
        TOTAL (never raises out): every send is suppressed; clear_assignment/load_state can't throw."""
        for nid in list(node_ids or []):
            ln = self.links.get(nid)
            if ln is not None:
                with contextlib.suppress(Exception):
                    await ln.send({"type": "unload", "model_id": target_id})
            nd = node_by_id.get(nid) if node_by_id else None
            if nd is not None:
                with contextlib.suppress(Exception):
                    nd.clear_assignment()
                    nd.load_state = "idle"

    async def _load_tp_locked(self, friendly: str, target_id: str, spec: ModelSpec,
                              ctx: int, tp: int, quant: str, cpu_only: bool = False) -> LoadedModel:
        """M4 tensor-parallel load. Every node in the group holds 1/tp of each layer
        (full embed/head/norm); rank 0 is the SINGLE pipeline stage the controller talks to
        and drives the peers over the all-reduce mesh. TP-v2 (per-rank streaming): each rank
        fetches ONLY its 1/tp tensor slice from /weights_tp and builds reduced-dim modules
        directly (Shard.from_stream's TP path), so a node needs ~full/tp RAM, not the full model.
        cpu_only=True runs TP across CPU nodes (RAM-bandwidth aggregation) instead of GPUs —
        the all-reduce and weight-sharding are device-agnostic, so the only difference is node
        selection (by free RAM) and forcing device='cpu' on every rank."""
        from transformers import AutoTokenizer
        L = spec.num_layers
        nkv = spec.num_kv_heads
        nh_heads = spec.num_heads
        if tp <= nkv:
            if nkv % tp != 0:
                raise RuntimeError(f"tp={tp} must divide num_key_value_heads={nkv} "
                                   f"(try tp in {[d for d in (2, 4, 8) if nkv % d == 0]})")
        else:
            # KV-HEAD REPLICATION (#87): tp > num_kv_heads -> replicate each KV head across tp//nkv
            # ranks so a model with few KV heads still spreads across MANY ranks (wide CPU TP). Needs
            # tp % nkv == 0 (even replication) and tp | num_attention_heads (even Q split). Forces an
            # EVEN split below (het + replication not supported yet).
            if tp % nkv != 0 or nh_heads % tp != 0:
                raise RuntimeError(
                    f"tp={tp} > num_kv_heads={nkv} (KV-head replication) needs tp % {nkv} == 0 AND "
                    f"num_attention_heads={nh_heads} % tp == 0 — try a tp that is a multiple of {nkv} "
                    f"and divides {nh_heads}")
        # FFN-group guard: the per-rank idim split needs at least `tp` int4 groups (128 cols) to hand
        # out, else the last rank's idim goes negative. Unreachable for real dense models (tp<=nh <<
        # intermediate/128) but cheap insurance for very wide TP / tiny-FFN archs.
        _ffn_groups = spec.intermediate_size // 128
        if _ffn_groups and tp > _ffn_groups:
            raise RuntimeError(f"tp={tp} too wide: exceeds FFN group count "
                               f"(intermediate_size {spec.intermediate_size} // 128 = {_ffn_groups})")
        # MIXED CPU+GPU mesh, sized by THROUGHPUT (#87). Unified candidate pool: a TP group can mix
        # GPU and CPU ranks, each classified per-NODE (GPU rank -> slice in VRAM; CPU rank -> in RAM).
        # Ranks are sized by BANDWIDTH (bw), NOT capacity (#68's VRAM/RAM GB): the lockstep mesh runs
        # at its SLOWEST rank, so a capacity split hands the slow CPU the biggest slice (straggler).
        # cpu_only forces every rank to CPU (the dashboard "run on CPU" path, unchanged).
        def is_gpu_rank(n) -> bool:
            return (not cpu_only) and n.vram_enabled and n.eff_vram_gb > 0
        def bw(n) -> float:
            return _node_tp_bw(n, is_gpu_rank(n))
        # PARALLEL LOAD: subtract OTHER in-flight loads' reservations so a TP load can't pick a node
        # a concurrent pipeline load (mid-stream with the lock released) has already claimed.
        _res_ram_tp, _res_vram_tp = self._reserved_bytes(exclude_key=friendly)
        def cap(n) -> float:          # has this node ANY room to be a rank? (VRAM for GPU, RAM for CPU)
            if is_gpu_rank(n):
                return max(0.0, n.eff_vram_gb - _res_vram_tp.get(n.node_id, 0) / GB)
            return max(0.0, n.eff_ram_gb - _res_ram_tp.get(n.node_id, 0) / GB)
        def avail(n) -> float:        # fit budget for the per-rank holds-its-share check
            if is_gpu_rank(n):
                return max(0.0, n.usable_total_gb - _res_vram_tp.get(n.node_id, 0) / GB)
            return max(0.0, n.eff_ram_gb - _res_ram_tp.get(n.node_id, 0) / GB)
        # The lockstep blocking-TCP all-reduce is only as reliable as its WEAKEST rank: a Windows
        # worker stalls it, and a battery HANDHELD (steamdeck/tablet/phone) that suspends/sleeps
        # drops out mid-forward -> "peer rank stalled" timeout. Verified LIVE: an all-nuc tp=4 mesh
        # generates cleanly AND coexists with qwen3 ("Paris..."), but a mesh that includes steamdeck
        # times out. Prefer STABLE server-class Linux nodes for the mesh; fall back to the full pool
        # only if there aren't enough (so TP still forms a group on a tiny/odd fleet).
        _TP_FLAKY = {"steamdeck", "tablet", "mobile", "phone"}
        _allc = [n for n in registry.alive_sorted() if n.can_infer and cap(n) > 0]
        _stable = [n for n in _allc if "windows" not in (n.os or "").lower()
                   and (n.hostname or "").lower() not in _TP_FLAKY]
        _pool = _stable if len(_stable) >= tp else _allc
        # ALL-CPU-FIRST selection (user policy): build the mesh from the fastest CPU nodes; admit a
        # GPU only when it clearly DOMINATES a CPU rank (>= 3x its bandwidth -> it'd carry a big
        # share) — a fast GPU otherwise sits idle at the all-reduce barrier waiting on slow CPU ranks.
        # No GPU / cpu_only -> all-CPU mesh; too few CPUs -> GPUs fill the remaining slots.
        # CAPACITY-AWARE rank pick (anti-oversubscription — the RAM analogue of the VRAM live-free
        # cap): a rank must actually hold its ~1/tp share in LIVE-free memory. avail() is already
        # live (eff_ram_gb derives from the latest free_mem_gb), but picking the top-tp purely by
        # BANDWIDTH hands a starved high-bandwidth node (e.g. an LPDDR5 NUC co-hosting Proxmox VMs,
        # ~2 GB free) a slice it can't fit -> the worker rejects and the WHOLE load fails instead
        # of using a roomier node. So keep only candidates that can hold the per-rank share, then
        # take the fastest tp of those (GPU ranks must clear the floor too).
        _share_floor = (spec.total_weight_bytes / GB / tp) * 1.15
        _fits = lambda n: avail(n) >= _share_floor
        _cpu = sorted((n for n in _pool if not is_gpu_rank(n)), key=lambda n: (bw(n), _mem_pref(n)), reverse=True)
        _gpu = sorted((n for n in _pool if is_gpu_rank(n)), key=lambda n: bw(n), reverse=True)
        chosen = [n for n in _cpu if _fits(n)][:tp]
        for g in _gpu:
            if not _fits(g):
                continue
            if len(chosen) < tp:
                chosen.append(g)
            else:
                weakest = min(chosen, key=bw)
                if bw(g) >= 3.0 * bw(weakest):
                    chosen[chosen.index(weakest)] = g
        if len(chosen) < tp:
            _top = sorted(_pool, key=avail, reverse=True)[:tp + 2]
            raise RuntimeError(
                f"tp={tp}{' cpu' if cpu_only else ''}: need {tp} nodes each with >= {_share_floor:.1f} GB "
                f"free for ~{spec.total_weight_bytes / GB:.1f} GB (1/{tp} + headroom); only {len(chosen)} "
                f"qualify. Most free now: " + ", ".join(f"{n.hostname} {avail(n):.1f}GB" for n in _top))
        # cand = chosen first (rank0 = fastest -> drives the mesh + holds embed/head), then the rest of
        # the pool as fallback so the live-link / distinct-host filter below can still reach tp nodes.
        cand = sorted(chosen, key=lambda n: (bw(n), _mem_pref(n)), reverse=True)
        cand += [n for n in sorted(_pool, key=lambda n: (bw(n), _mem_pref(n)), reverse=True)
                 if n not in chosen]
        kind = "mixed"   # refined to GPU / CPU / "N GPU + M CPU" once tp_nodes is chosen (below)
        # Pre-flight (anti-churn / finding 1): a reconnecting worker can leave a STALE node_id whose
        # half-dead control link still .send()s without raising, and a host can briefly appear under
        # two node_ids. Either lets a TP rank be assigned to a node that never actually receives the
        # load -> rank 0 then waits the WHOLE gather timeout (the observed
        # "tp load failed on <host>: TimeoutError()", with both ranks' weights served to one host).
        # Require a LIVE control link AND one rank per distinct HOST (freshest id wins, cand is
        # ranked first so the first seen per host is the chosen resource).
        live, seen_hosts = [], set()
        for n in cand:
            if self.links.get(n.node_id) is None or n.hostname in seen_hosts:
                continue
            seen_hosts.add(n.hostname); live.append(n)
        if len(live) < tp:
            raise RuntimeError(f"tp={tp} needs {tp} distinct live {kind} nodes; only {len(live)} "
                               f"have a live control link {[n.hostname for n in live]}")
        tp_nodes = live[:tp]
        # The all-reduce ROOT (rank0) binds the mesh + does the most blocking-socket work each layer;
        # prefer a NON-Windows node as root (the Windows controller host as mesh-root stalls the
        # per-layer all-reduce -> generate-time timeout). Keep the fastest non-Windows rank as rank0
        # when the mesh is mixed-OS; an all-Windows mesh is left as-is.
        if "windows" in (tp_nodes[0].os or "").lower():
            _nonwin = next((n for n in tp_nodes if "windows" not in (n.os or "").lower()), None)
            if _nonwin is not None:
                tp_nodes = [_nonwin] + [n for n in tp_nodes if n is not _nonwin]
        # per-rank tier of the CHOSEN mesh (#87): each rank is GPU or CPU on its own; drives the
        # per-rank `device` sent below and a human-readable "kind" label for the basis/log.
        rank_is_gpu = [is_gpu_rank(n) for n in tp_nodes]
        n_gpu = sum(rank_is_gpu); n_cpu = tp - n_gpu
        kind = ("GPU" if n_cpu == 0 else "CPU" if n_gpu == 0 else f"{n_gpu} GPU + {n_cpu} CPU")
        # Instrumentation: log the ACTUAL rank->node assignment to the activity feed (readable via
        # /status) so a TP load's placement is verifiable — the "-> host" in the /weights_tp serving
        # log is unreliable (both ranks share layer range [0,L], so its _owns match always resolves
        # to the first node, mislabeling every slice to one host).
        log_activity("TP rank assignment: " + ", ".join(
            f"rank{r}={n.hostname}({n.node_id})" for r, n in enumerate(tp_nodes)))
        tp_basis = (f"tensor-parallel tp={tp} ({kind}) -> "
                    + ", ".join(n.hostname for n in tp_nodes)
                    + f" -- each rank holds 1/{tp} of every layer"
                    + ("" if quant in (None, "none", "") else f", {quant}"))
        log_activity(f"{friendly}: plan basis → {tp_basis}")
        full_gb = spec.total_weight_bytes / GB
        # HETEROGENEOUS TP (#68): when the ranks' capacities really differ, split each layer
        # PROPORTIONAL to capacity (usable VRAM for GPU TP, RAM for CPU TP) so a bigger GPU holds a
        # bigger slice — the smallest node no longer has to hold an equal 1/tp share it can't fit.
        # `tp_weights` is sent to every rank, which builds its reduced-dim structure from the SAME
        # wire._tp_hetsplit the server slices with. Near-equal capacities -> tp_weights=None -> the
        # uniform 1/tp split (also keeps a rolling-update OLD worker, which ignores tp_weights, in
        # sync). Per-rank fit: a rank holds ~ its capacity-share of the model (+15% transient).
        if tp > nkv:
            # KV-head replication uses an EVEN split + the replication geometry (het + replication is
            # not supported yet) — uniform caps so the fit-check, rank_bytes, serve slice and worker
            # structure all route through wire._tp_hetsplit's replication branch with matching shapes.
            caps = [1.0] * tp
            het = True
        else:
            caps = [max(0.1, bw(n)) for n in tp_nodes]   # THROUGHPUT weights -> faster rank, bigger slice
            het = (max(caps) / min(caps)) > 1.15
        wtot = sum(caps) or 1.0
        # If the bandwidth-proportional (het) split would overflow a chosen rank's LIVE-free memory
        # but a plain uniform 1/tp split fits every rank, prefer uniform (anti-oversubscription): a
        # slightly slower but PLACEABLE mesh beats a failed load. The tp>nkv replication path uses
        # uniform caps already and is left untouched (het stays True there for its slice geometry).
        if het and tp <= nkv:
            _ovf = [n for n, c in zip(tp_nodes, caps) if avail(n) < full_gb * (c / wtot) * 1.15]
            if _ovf and all(avail(n) >= (full_gb / tp) * 1.15 for n in tp_nodes):
                log_activity(f"{friendly}: TP het split overflows {[n.hostname for n in _ovf]} "
                             f"-> uniform 1/{tp} split (fits live RAM)")
                caps, het, wtot = [1.0] * tp, False, float(tp)
        tp_weights = [round(c, 3) for c in caps] if het else None
        if het:
            for n, c in zip(tp_nodes, caps):
                share = full_gb * (c / wtot) * 1.15
                if avail(n) < share:
                    raise RuntimeError(
                        f"tp het: {n.hostname} can't hold its ~{share:.1f} GB capacity-share of "
                        f"~{full_gb:.1f} GB ({kind} avail {avail(n):.1f} GB).")
            pct = ", ".join(f"{n.hostname} {100 * c / wtot:.0f}%" for n, c in zip(tp_nodes, caps))
            tp_basis = (f"heterogeneous tensor-parallel tp={tp} ({kind}) -> {pct}"
                        + ("" if quant in (None, "none", "") else f", {quant}"))
            log_activity(f"{friendly}: plan basis → {tp_basis}")
        else:
            per_rank_gb = (full_gb / tp) * 1.15
            for n in tp_nodes:
                if avail(n) < per_rank_gb:
                    raise RuntimeError(
                        f"tp v2 (per-rank streaming) needs ~{per_rank_gb:.1f} GB/rank "
                        f"(1/{tp} of ~{full_gb:.1f} GB + headroom); {n.hostname} has "
                        f"{avail(n):.1f} GB {kind} free.")
        root = tp_nodes[0]
        tp_port = root.data_port + 1
        # COEXISTENCE (#87): do NOT clear EVERY node — that wipes the display/accounting for OTHER
        # resident models (whose shards stay loaded on the workers). Only (re)assign the chosen
        # tp_nodes; nodes holding other models keep their assignment, so this TP load coexists.
        for n in tp_nodes:
            n.clear_assignment()
        # In-flight marker so the dashboard shows "loading <model> X%" during a TP load instead of
        # "none": each rank streams its full [0,L) range + embed + head from /weights_tp, whose
        # serving path bumps loading["ready"] per slice. total = tp*(L+2) is chunking-invariant —
        # the per-rank (end-start) sum is always L, +1 embed +1 head, however the layers are chunked.
        self.loadings[friendly] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id,
                        "total": tp * (L + 2), "ready": 0,
                        "stages_total": tp, "stages_ready": 0, "basis": tp_basis,
                        "node_ids": [n.node_id for n in tp_nodes],
                        "started": (self.loadings.get(friendly) or {}).get("started") or time.time()}
        loop = asyncio.get_event_loop()
        futs: dict[str, asyncio.Future] = {}
        for rank, n in enumerate(tp_nodes):
            link = self.links.get(n.node_id)
            if link is None:
                raise RuntimeError(f"no control link to {n.node_id}")
            n.stage, n.tp_rank, n.tp_size = 0, rank, tp
            n.layer_start, n.layer_end = 0, L
            n.load_state = "loading"
            # per-rank planned bytes: heterogeneous -> this rank's capacity share; else even 1/tp.
            rank_bytes = (int(spec.total_weight_bytes * caps[rank] / wtot) if het
                          else int(spec.total_weight_bytes / tp))
            msg = {"type": "load", "model_id": target_id,
                   "layer_start": 0, "layer_end": L, "has_embed": True, "has_head": True,
                   "stage": 0, "num_stages": 1, "dtype": "bfloat16",
                   "controller_http_port": ARGS.http_port,
                   # per-rank device from THIS node's tier (#87): a GPU rank gets its load_device()
                   # ("" -> worker cpu+gpu default, or "gpu"); a CPU rank gets explicit 'cpu'.
                   "device": n.load_device() if rank_is_gpu[rank] else "cpu",
                   "quant": quant, "tp_rank": rank, "tp_size": tp,
                   "tp_root_host": root.data_host, "tp_root_port": tp_port,
                   # #68: per-rank capacity weights (None when ~equal) -> heterogeneous split; every
                   # rank gets the SAME list so wire._tp_hetsplit is identical on all ranks + the serve.
                   "tp_weights": tp_weights,
                   # #63: this rank's planned resident bytes (its capacity share) -> RAM balloon.
                   "plan_ram_bytes": rank_bytes}
            if rank == 0:
                msg["next_host"], msg["next_port"] = None, ARGS.data_port  # -> controller
            fut = loop.create_future()
            link.pending_load = fut
            futs[n.node_id] = fut
            await link.send(msg)
        async def _abort_cleanup():
            # A failed TP load must NOT leave dirty per-rank state (bound mesh port, partial shard,
            # load_state='loading') — that arms the next load's churn (the observed cascade). Tell
            # EVERY rank to unload + reset, clear the in-flight marker, and arm a self-update
            # cool-down so the now-"idle"-looking controller doesn't immediately exit(42).
            for _n in tp_nodes:
                _ln = self.links.get(_n.node_id)
                if _ln is not None:
                    with contextlib.suppress(Exception):
                        await _ln.send({"type": "unload", "model_id": target_id})
                _n.clear_assignment(); _n.load_state = "idle"
            self.loadings.pop(friendly, None)
            self._last_load_failure = time.time()
        # Scale the TP load timeout by the bf16 READ volume like the pipeline path (#100): each rank
        # still streams its full bf16 slice and quantizes after, so a flat 900 s timed out big loads.
        # spec is already for_quant'd here -> recover ~bf16 (int4 ~/0.3, int8 x2). Clamp [15 min, 4 h].
        _tpb = getattr(spec, "total_weight_bytes", 0) or 0
        _tpb = int(_tpb / 0.3) if quant == "int4" else (_tpb * 2 if quant == "int8" else _tpb)
        tp_load_timeout = max(900, min(int(_tpb / (35 * 1024 * 1024)) + 300, 4 * 3600))
        results = await asyncio.gather(
            *[asyncio.wait_for(f, timeout=tp_load_timeout) for f in futs.values()], return_exceptions=True)
        tp_gpu_bytes = 0   # #69: total on-GPU bytes ACROSS all ranks (each reports its own)
        tp_gpu_kv_bytes = 0   # total full-ctx KV reserved on GPU across ranks (coexistence reserve)
        for nid, r in zip(futs.keys(), results):
            err = (repr(r) if isinstance(r, Exception)
                   else r.get("error") if isinstance(r, dict) and r.get("type") == "error" else None)
            if err is not None:
                hn = registry._nodes[nid].hostname if nid in registry._nodes else nid
                await _abort_cleanup()
                raise RuntimeError(f"tp load failed on {hn}: {err}")
            nd = registry._nodes.get(nid)
            if nd:
                nd.load_state = "ready"
                if isinstance(r, dict):
                    nd.shard_gpu_bytes = int(r.get("gpu_bytes", 0))
                    tp_gpu_bytes += int(r.get("gpu_bytes", 0))
                    tp_gpu_kv_bytes += int(r.get("gpu_kv_bytes", 0))
                _tcard = self.loadings.get(friendly)   # bump the node counter ("A/B nodes loaded")
                if _tcard is not None:
                    _tcard["stages_ready"] = _tcard.get("stages_ready", 0) + 1
        # the pipeline is just rank 0; the controller talks only to it
        _tp_dial = (_dial_host(root.data_host), root.data_port)
        stage0_writer = await self._connect_retry(*_tp_dial)
        tok = await asyncio.to_thread(_get_tokenizer, target_id)
        eos = self._eos_ids(tok)
        # (TP models carry no speculative draft — big-model decode is bandwidth-bound.)
        # #69: carry the TP group's TOTAL on-GPU bytes on the (single) TP stage so the dashboard's
        # vram_used = sum(stage.gpu_bytes) reflects reality (beast+theocomp VRAM), not 0. The TP
        # plan has one stage representing the whole group, so the aggregate belongs here.
        stage = StageAssign(root.node_id, root.hostname, 0, L, True, True,
                            int(spec.total_weight_bytes / tp), int(root.usable_total_gb * GB),
                            gpu_bytes=int(tp_gpu_bytes), gpu_kv_bytes=int(tp_gpu_kv_bytes))
        plan = PlanResult(ok=True, model=spec.name, ctx_len=ctx, num_layers=L,
                          pool_usable_gb=round(sum(n.usable_total_gb for n in tp_nodes), 2),
                          required_gb=round(full_gb, 2), stages=[stage])
        now = time.time()
        lm = LoadedModel(friendly, target_id, spec, ctx, plan,
                         [n.node_id for n in tp_nodes], tok, eos, now,
                         quant=quant, stage0_writer=stage0_writer, last_used=now,
                         stage0_dial=_tp_dial, last_send_ts=now)
        lm.plan_basis = tp_basis                          # placement basis (#65)
        lm.tp_size = tp                                    # #88: record TP width for the card + /reconfigure
        self.models[friendly] = lm
        self.loadings.pop(friendly, None)   # card off -> dashboard flips to resident (finally also pops)
        registry.dirty = False
        print(f"[load] TP tp={tp} {friendly}: rank0={root.hostname} "
              f"peers={[n.hostname for n in tp_nodes[1:]]} (1/{tp} of each layer per rank)")
        log_activity(f"{friendly} READY (tp={tp}): rank0={root.hostname}, "
                     f"peers={[n.hostname for n in tp_nodes[1:]]}")
        return lm

    @staticmethod
    def _eos_ids(tok) -> set:
        ids = set()
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        for t in ("<|im_end|>", "<|endoftext|>"):
            with contextlib.suppress(Exception):
                tid = tok.convert_tokens_to_ids(t)
                if tid is not None and tid >= 0:
                    ids.add(int(tid))
        return ids

    def _sample(self, row, temperature: float, top_p: float) -> int:
        import torch
        row = row.float()
        if not temperature or temperature <= 0:
            return int(row.argmax())
        probs = torch.softmax(row / temperature, dim=-1)
        if top_p and 0 < top_p < 1:
            sp, idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sp, 0)
            keep = cdf - sp <= top_p
            sp = sp * keep
            sp = sp / sp.sum()
            return int(idx[int(torch.multinomial(sp, 1))])
        return int(torch.multinomial(probs, 1))

    async def _freshen_stage0(self, model: LoadedModel, force: bool = False) -> None:
        """#stage0-stale-reconnect: rebuild model.stage0_writer FRESH if it may be stale. The
        controller's stage0 conn is opened at LOAD then sits IDLE between requests; an idle socket
        can go SILENTLY half-open (the write SUCCEEDS but the bytes vanish -> no logits -> ~600s
        GEN_TIMEOUT hang — the 'loaded but never replies' bug). Reconnecting a fresh socket at
        generate START (when idle past STAGE0_STALE_S) gives every request a hot, proven path —
        the SAME lazy-fresh-connect the workers already use for their next hop (client.py
        _send_next). force=True rebuilds unconditionally (used by _send after a write FAILED).
        Cheap: a TCP connect is ~ms vs a multi-token generation."""
        if not model.stage0_dial:
            return   # no saved dial target (shouldn't happen post-load) -> leave as-is
        now = time.time()
        if (not force and model.stage0_writer is not None
                and (now - model.last_send_ts) <= STAGE0_STALE_S):
            return   # used recently -> the connection is hot, reuse it (no churn on busy models)
        old = model.stage0_writer
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        model.stage0_writer = await self._connect_retry(*model.stage0_dial)
        model.last_send_ts = now
        with contextlib.suppress(Exception):
            print(f"[data] freshened stage0 conn for {model.friendly} -> "
                  f"{model.stage0_dial[0]}:{model.stage0_dial[1]} "
                  f"({'write failed' if force else 'idle'})", flush=True)

    async def _send(self, model: LoadedModel, x, cache_position: int, reset: bool,
                    all_logits: bool = False, mm=None, position_ids=None,
                    capture_hidden: bool = False, capture_pre_norm: bool = False):
        """Push one frame (token ids) through `model`'s pipeline and return last-stage
        logits — last position only, or every position when all_logits=True (verify).
        mm = (positions, embeds_tensor) (#22 inc 3): on a prefill (reset), a companion
        'mm' frame is sent FIRST with the same req_id so stage 0 splices those embeds into
        its embed output at `positions` before running the layers."""
        if model.stage0_writer is None:
            await self._freshen_stage0(model, force=True)   # rebuild from saved dial if dropped
        if model.stage0_writer is None:
            raise RuntimeError("pipeline not connected")
        loop = asyncio.get_event_loop()
        rid = self.next_req()
        meta, raw = _pack_tensor(x)
        fut = loop.create_future()
        self.pending[rid] = fut
        self.pending_model[rid] = model.target_id   # so a head drop fails only this model

        async def _flush(w) -> None:
            if mm is not None and reset:
                positions, embeds = mm
                emeta, eraw = _pack_tensor(embeds)
                nb = await _write_frame(w, {
                    "req_id": rid, "model_id": model.target_id, "kind": "mm",
                    "positions": list(positions), **emeta}, eraw)
                net_account(self._stage0_id(model), to_node=nb)  # controller -> stage0
            hdr = {"req_id": rid, "model_id": model.target_id, "kind": "ids",
                   "cache_position": cache_position,
                   "reset": reset, "all_logits": all_logits, **meta}
            if position_ids is not None:   # #22 inc 4: 3D mRoPE positions [3][q] (small JSON list)
                hdr["position_ids"] = position_ids
            if capture_hidden:   # #P6 speech: ask the head stage for post-norm hidden too
                hdr["capture_hidden"] = True
            if capture_pre_norm:   # #91 MTP: ask the head stage for the PRE-final-norm trunk hidden
                hdr["capture_pre_norm"] = True
            nb = await _write_frame(w, hdr, raw)
            net_account(self._stage0_id(model), to_node=nb)  # controller -> stage0

        try:
            try:
                await _flush(model.stage0_writer)
            except (ConnectionError, OSError, asyncio.IncompleteReadError):
                # stage0 conn died at/mid send -> rebuild FRESH + resend ONCE. The worker keys frames
                # by model_id and hasn't processed anything on the new socket, so resending the same
                # req_id is clean (mirrors the worker's reconnect-once-on-failure in _send_next).
                await self._freshen_stage0(model, force=True)
                await _flush(model.stage0_writer)
            model.last_send_ts = time.time()
            return await asyncio.wait_for(fut, timeout=GEN_TIMEOUT_S)
        finally:
            self.pending.pop(rid, None)  # never leak the future
            self.pending_model.pop(rid, None)

    async def _crop(self, model: LoadedModel, length: int) -> None:
        """Tell every stage of `model` to truncate its KV cache to `length` (spec rollback).
        Fire-and-forget: in-order delivery on each stage's connection guarantees the
        crop is applied before the next frame the controller sends afterwards."""
        if model.stage0_writer is not None:
            nbytes = await _write_frame(model.stage0_writer,
                                        {"model_id": model.target_id, "kind": "crop",
                                         "cache_position": length}, b"")
            net_account(self._stage0_id(model), to_node=nbytes)  # controller -> stage0

    # -- draft model (runs entirely on the controller; one per LoadedModel) --
    def _load_draft(self, model: LoadedModel, draft_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM
        _controller_model_dir(draft_id)
        model.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_id, dtype=torch.bfloat16, attn_implementation="eager").eval()
        model.draft_id = draft_id
        model.draft_kv = None

    def _unload_draft(self, model: LoadedModel) -> None:
        model.draft_model = None
        model.draft_kv = None
        model.draft_id = None

    def _draft_prefill(self, model: LoadedModel, prompt_ids):
        import torch
        from transformers import DynamicCache
        model.draft_kv = DynamicCache()
        with torch.inference_mode():
            out = model.draft_model(input_ids=torch.tensor([prompt_ids]),
                                    past_key_values=model.draft_kv, use_cache=True)
        return out.logits[0, -1]

    def _draft_step(self, model: LoadedModel, token: int, position: int):
        import torch
        with torch.inference_mode():
            out = model.draft_model(input_ids=torch.tensor([[token]]),
                                    past_key_values=model.draft_kv, use_cache=True,
                                    cache_position=torch.tensor([position]))
        return out.logits[0, -1]

    def _draft_crop(self, model: LoadedModel, length: int) -> None:
        if model.draft_kv is not None:
            with contextlib.suppress(Exception):
                model.draft_kv.crop(length)

    async def generate(self, friendly: str, prompt_ids: list[int], max_new: int,
                       temperature: float, top_p: float, speculative: bool = False,
                       rec=None, mm=None, mrope=None, spec_k: int = 0):
        """Dispatch generation for model `friendly`: speculative-greedy decode only when
        explicitly requested AND a draft is loaded AND decoding is greedy; otherwise plain
        KV-cache decode (M2e). Speculative is opt-in because it only wins when the target's
        per-traversal cost dwarfs the local draft cost (big model / many nodes) — on small
        targets it measures SLOWER, so it must never silently replace the fast default."""
        model = self._pick_replica(friendly)   # data-parallel: least-loaded replica (#39)
        if model is None or model.stage0_writer is None:
            raise RuntimeError("no model loaded")
        # PER-REPLICA lock: different models AND different replicas of one model decode
        # concurrently; requests routed to the SAME replica queue on its lock.
        # Track queue depth for /status (queued = waiting on this model's lock; active = generating).
        model.queued += 1
        acquired = False
        try:
            async with model.lock:
                acquired = True
                model.queued -= 1
                model.active += 1
                model.last_token_ts = time.time()   # #gen-stall-watchdog: start the no-progress timer at gen begin
                # #stage0-stale-reconnect: rebuild a stale (idle-since-last-request) stage0 conn BEFORE
                # the prefill so this request rides a fresh, proven socket instead of a possibly
                # half-open one (the 'loaded but never replies' / ~600s hang). No-op when hot (busy
                # model) or recently sent; under the lock so no concurrent decode is using the writer.
                with contextlib.suppress(Exception):
                    await self._freshen_stage0(model)
                _inflight_start(rec)   # slot acquired: queued -> running (dashboard)
                try:
                    model.last_used = time.time()
                    greedy = not temperature or temperature <= 0
                    # #46 throughput: count emitted tokens over wall-clock and store a
                    # smoothed decode tok/s on the model (observability only — no effect on
                    # generation). t0 starts after the per-replica lock is held so it times
                    # this request's decode, not its queue wait. Tokens = real tokens yielded
                    # (item[0] is not None); the trailing stop/length marker is skipped.
                    _t0 = time.monotonic()
                    _ntoks = 0
                    # Multimodal (mm) forces PLAIN decode: the controller-side draft model has
                    # no image embeds, so speculative would diverge — only the full pipeline
                    # gets the spliced vision tokens at prefill.
                    # #91 MTP: when speculative+greedy is requested but there's no separate draft
                    # model, fall through to the checkpoint's own MTP (nextn) self-draft if it has one.
                    mtp_head = None
                    if (speculative and greedy and mm is None and model.draft_model is None):
                        with contextlib.suppress(Exception):
                            mtp_head = await self._ensure_mtp_head(model)
                    if speculative and model.draft_model is not None and greedy and mm is None:
                        async for item in self._decode_spec(model, prompt_ids, max_new, spec_k):
                            if item[0] is not None:
                                _ntoks += 1
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
                                _dt = time.monotonic() - _t0
                                if _dt > 1e-6:           # LIVE decode rate -> card updates mid-gen (#46)
                                    model.last_tok_s = _ntoks / _dt
                            yield item
                    elif mtp_head is not None:
                        async for item in self._decode_spec_mtp(model, prompt_ids, max_new, mtp_head):
                            if item[0] is not None:
                                _ntoks += 1
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
                                _dt = time.monotonic() - _t0
                                if _dt > 1e-6:           # LIVE decode rate -> card updates mid-gen (#46)
                                    model.last_tok_s = _ntoks / _dt
                            yield item
                    else:
                        async for item in self._decode_plain(model, prompt_ids, max_new,
                                                             temperature, top_p, mm=mm, mrope=mrope):
                            if item[0] is not None:
                                _ntoks += 1
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
                                _dt = time.monotonic() - _t0
                                if _dt > 1e-6:           # LIVE decode rate -> card updates mid-gen (#46)
                                    model.last_tok_s = _ntoks / _dt
                            yield item
                finally:
                    model.active -= 1
                    # #model-detail lifetime counters (this is the main text-generation path; TP /
                    # speech paths don't update these). Count every served request + its tokens.
                    model.req_total += 1
                    model.tok_in_total += len(prompt_ids)
                    model.tok_out_total += _ntoks
                    # Record decode throughput once the generation finishes (or is cut
                    # short). Guard on a sane sample (>=1 token, measurable time) so a
                    # zero-token or instant request doesn't poison the read.
                    _dt = time.monotonic() - _t0
                    if _ntoks >= 1 and _dt > 1e-6:
                        ts = _ntoks / _dt
                        model.last_tok_s = ts
                        if ts > model.max_tok_s:        # peak decode tok/s (#model-detail)
                            model.max_tok_s = ts
                        # EMA (alpha=0.3): seed on the first sample, then blend.
                        model.ema_tok_s = ts if model.ema_tok_s <= 0.0 else \
                            0.3 * ts + 0.7 * model.ema_tok_s
        finally:
            if not acquired:               # cancelled while still waiting in the queue
                model.queued -= 1

    async def _decode_plain(self, model, prompt_ids, max_new, temperature, top_p, mm=None,
                            mrope=None):
        """Prefill-once + one-token-at-a-time KV-cache decode (M2e). mm=(positions, embeds)
        (#22 inc 3) splices multimodal embeds into the PREFILL only; decode steps are plain.
        mrope=(prefill_position_ids [3][q], base) (#22 inc 4) carries 3D image positions:
        the prefill uses the full layout; each decode token uses [base+step] on all 3 dims."""
        import torch
        # Empty prompt (a keep-warm/health probe whose text tokenizes to []) has nothing to
        # prefill: torch.tensor([[]]) is shape [1,0] and an empty forward crashes the worker's
        # tensor unpack. Short-circuit with zero generated tokens BEFORE any wire send.
        if not prompt_ids:
            yield None, "stop"
            return
        prefill_pos = mrope[0] if mrope else None
        base = mrope[1] if mrope else None
        logits = await self._send(model, torch.tensor([prompt_ids], dtype=torch.long), 0, True,
                                  mm=mm, position_ids=prefill_pos)
        cur = len(prompt_ids)
        model.kv_pos = cur          # KV depth so far (prompt); climbs per decode token
        produced = 0
        # #21: this model's lm_head can be WIDER than its text tokenizer (a multimodal
        # head carries vision/audio placeholder ids the text tokenizer can't decode).
        # Selecting one of those ids crashed detokenization ("list index out of
        # range") and showed up as empty/failed generation. Mask logits beyond the
        # tokenizer's decodable range so we only ever emit a real text token.
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            row = logits[0, -1]
            if ntok and ntok < int(row.shape[-1]):
                row = row.clone()
                row[ntok:] = float("-inf")
            tok_id = self._sample(row, temperature, top_p)
            if produced == 0:
                with contextlib.suppress(Exception):
                    print(f"[gen] {model.friendly}: first token id={tok_id} "
                          f"head_vocab={int(logits.shape[-1])} len(tok)={ntok} "
                          f"eos={tok_id in model.eos_ids}")
            produced += 1
            if tok_id in model.eos_ids:
                yield None, "stop"
                return
            yield tok_id, None
            if produced >= max_new:
                break
            # mRoPE decode position = base + step (same on t/h/w); else 1D (worker uses arange).
            dpos = [[base + produced - 1]] * 3 if base is not None else None
            logits = await self._send(model, torch.tensor([[tok_id]], dtype=torch.long), cur,
                                      False, position_ids=dpos)
            cur += 1
            model.kv_pos = cur
        yield None, "length"

    async def capture_thinker(self, friendly, prompt_ids, max_new, temperature=0.0,
                              top_p=1.0, mm=None, mrope=None):
        """#P6 speech: run the distributed Thinker like _decode_plain BUT with
        capture_hidden=True so the head stage returns the post-norm hidden per step. Collects
        the prefill hidden (all prompt positions) + each fed token's hidden, exactly the
        thinker_hidden_states the Talker consumes. Returns
        (gen_ids, prefill_hidden [1,P,H], step_hiddens [list of [1,1,H]], stop_reason).
        thinker_token_embeds are computed separately on the controller from the embed matrix."""
        import torch
        model = self.models[friendly]
        prefill_pos = mrope[0] if mrope else None
        base = mrope[1] if mrope else None
        logits, prefill_hidden = await self._send(
            model, torch.tensor([prompt_ids], dtype=torch.long), 0, True,
            mm=mm, position_ids=prefill_pos, capture_hidden=True)
        cur = len(prompt_ids)
        model.kv_pos = cur
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0
        gen_ids: list[int] = []
        step_hiddens: list = []
        produced = 0
        stop = "length"
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            row = logits[0, -1]
            if ntok and ntok < int(row.shape[-1]):
                row = row.clone()
                row[ntok:] = float("-inf")
            tok_id = self._sample(row, temperature, top_p)
            produced += 1
            gen_ids.append(tok_id)
            if tok_id in model.eos_ids:
                stop = "stop"
                break
            if produced >= max_new:
                break
            dpos = [[base + produced - 1]] * 3 if base is not None else None
            logits, hid = await self._send(
                model, torch.tensor([[tok_id]], dtype=torch.long), cur, False,
                position_ids=dpos, capture_hidden=True)
            step_hiddens.append(hid)   # hidden of the token we just fed (tok_id)
            cur += 1
            model.kv_pos = cur
        return gen_ids, prefill_hidden, step_hiddens, stop

    async def generate_speech(self, friendly, prompt_ids, max_new=256, speaker="Chelsie",
                              talker_max_new=2048):
        """#P6 speech-out Phase 3: distributed Thinker (captured hidden states) -> Talker (codec
        tokens) -> token2wav (waveform). Faithful to Qwen2.5-Omni's generate() talker assembly
        (modeling_qwen2_5_omni.py): builds thinker_reply_part = last-layer-hidden + token-embed
        for the generated tokens, talker_inputs_embeds for the prompt, prepends the speaker text
        bos + the first reply hidden, appends eos/pad embeds, then drives the REAL HF talker +
        token2wav. Returns (gen_ids, text_stop, waveform [N] float32 @24kHz, info)."""
        import torch
        model = self.models[friendly]
        target = model.target_id
        sc = await asyncio.to_thread(_load_speech_components, target)
        talker, token2wav, embed = sc["talker"], sc["token2wav"], sc["embed"]
        dev, speaker_map = sc["dev"], sc["speaker_map"]
        if speaker not in speaker_map:
            raise RuntimeError(f"speaker '{speaker}' not in {list(speaker_map.keys())}")
        spk = speaker_map[speaker]
        # 1) distributed thinker with hidden-state capture
        gen_ids, prefill_hidden, step_hiddens, stop = await self.capture_thinker(
            friendly, prompt_ids, max_new)
        info = {"prompt_len": len(prompt_ids), "gen_tokens": len(gen_ids),
                "captured_steps": len(step_hiddens), "text_stop": stop}

        def _assemble_and_run():
            f32 = torch.float32
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

            def emb(ids_t):
                return embed(ids_t.to(dev)).to(f32)

            # thinker_hidden_states (last-layer): [prefill_all, *per-fed-token]
            hs = [prefill_hidden.to(device=dev, dtype=f32)]
            hs += [h.to(device=dev, dtype=f32) for h in step_hiddens]
            # thinker_token_embeds (layer-0 = input embeddings); text-only -> no mm zeroing.
            # The fed tokens are exactly the ones we captured a hidden for (gen_ids[:len(step)]).
            fed = gen_ids[:len(step_hiddens)]
            te = [emb(input_ids)]
            for t in fed:
                te.append(emb(torch.tensor([[t]], dtype=torch.long)))
            bos_id = int(spk["bos_token"])
            gen_t = torch.tensor([gen_ids], dtype=torch.long, device=dev)
            talker_input_text_ids = torch.cat([
                input_ids,
                torch.tensor([[bos_id]], dtype=torch.long, device=dev),
                gen_t[:, :1]], dim=-1)
            talker_input_ids = torch.cat([
                torch.full_like(input_ids, fill_value=talker.codec_mask_token),
                torch.tensor([[talker.codec_pad_token]], dtype=torch.long, device=dev),
                torch.tensor([[talker.codec_bos_token]], dtype=torch.long, device=dev)], dim=1)
            reply = torch.cat(hs[1:], dim=1) + torch.cat(te[1:], dim=1)   # [1,F,H] generated
            talker_inputs_embeds = hs[0] + te[0]                          # [1,P,H] prompt
            bos_embed = emb(torch.tensor([[bos_id]], dtype=torch.long))
            talker_inputs_embeds = torch.cat(
                [talker_inputs_embeds, bos_embed, reply[:, :1, :]], dim=1)
            eos_embed = emb(torch.tensor([[talker.text_eos_token]], dtype=torch.long))
            pad_embed = emb(torch.tensor([[talker.text_pad_token]], dtype=torch.long))
            reply = torch.cat([reply[:, 1:, :], eos_embed, pad_embed], dim=1)
            with torch.inference_mode():
                # Talker generation params = Qwen2.5-Omni's generate() defaults (the talker is
                # TUNED for sampling; greedy degenerates -> noise + never stops). eos is BOTH
                # codec_pad (8292) and codec_eos (8294).
                talker_result = talker.generate(
                    input_ids=talker_input_ids,
                    input_text_ids=talker_input_text_ids,
                    thinker_reply_part=reply,
                    inputs_embeds=talker_inputs_embeds,
                    suppress_tokens=[talker.codec_bos_token],
                    do_sample=True, top_k=40, top_p=0.8, temperature=0.9,
                    repetition_penalty=1.05,
                    eos_token_id=[talker.codec_pad_token, talker.codec_eos_token],
                    max_new_tokens=int(talker_max_new))
                codes = talker_result[:, talker_input_ids.shape[1]:-1]
                info["codec_tokens"] = int(codes.shape[-1])
                info["talker_hit_cap"] = bool(talker_result.shape[1] - talker_input_ids.shape[1]
                                              >= int(talker_max_new))
                with contextlib.suppress(Exception):
                    info["codes_min"] = int(codes.min()); info["codes_max"] = int(codes.max())
                wav = token2wav(codes.to(dev),
                                conditioning=spk["cond"].to(dev).float(),
                                reference_mel=spk["ref_mel"].to(dev).float())
            return wav.float().reshape(-1)

        global ENCODING
        ENCODING += 1   # hold the self-update idle gate during the talker/vocoder run
        try:
            wav = await asyncio.to_thread(_assemble_and_run)
        finally:
            ENCODING -= 1
        info["wav_samples"] = int(wav.shape[0])
        info["wav_seconds"] = round(int(wav.shape[0]) / 24000.0, 2)
        return gen_ids, stop, wav, info

    async def _decode_spec(self, model, prompt_ids, max_new, k: int = 0):
        """Speculative greedy decode (M3): the local draft proposes K tokens, the
        pipeline verifies all K in one traversal, we accept the matched prefix + 1
        correction (bit-exact vs plain greedy), then roll the KV cache back.
        Falls back implicitly to M2e behaviour at K=0 acceptance (1 token/round).
        k>0 overrides SPEC_K (per-request, for tuning — a slower/more-distributed target
        favours a LARGER K so one verify pass amortizes more of the pipeline traversal)."""
        import torch
        # Empty prompt: nothing to prefill — short-circuit before the prefill _send (same guard
        # as _decode_plain; keeps the empty-ids probe off the wire).
        if not prompt_ids:
            yield None, "stop"
            return
        eos = model.eos_ids
        K = k if (k and k > 0) else SPEC_K
        a0 = (await self._send(model, torch.tensor([prompt_ids], dtype=torch.long), 0, True))[0, -1]
        cur = len(prompt_ids)
        d_logits = await asyncio.to_thread(self._draft_prefill, model, prompt_ids)
        produced = 0
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            # 1. draft K tokens greedily on the controller
            drafts = []
            dl = d_logits
            for k in range(K):
                dt = int(dl.argmax())
                drafts.append(dt)
                dl = await asyncio.to_thread(self._draft_step, model, dt, cur + k)
            # 2. verify all K on the target in ONE pipeline traversal
            V = await self._send(model, torch.tensor([drafts], dtype=torch.long), cur, False,
                                 all_logits=True)
            # 3. target's greedy tokens for positions cur..cur+K
            tg = [int(a0.argmax())] + [int(V[0, k].argmax()) for k in range(K)]
            # 4. accept the matched prefix, then one target token (correction/bonus)
            m = 0
            while m < K and tg[m] == drafts[m]:
                m += 1
            accepted = tg[:m + 1]
            # 5. roll target KV back to drop rejected draft positions
            await self._crop(model, cur + m)
            # 6. emit
            for t in accepted:
                produced += 1
                if t in eos:
                    yield None, "stop"
                    return
                yield t, None
                if produced >= max_new:
                    return
            # 7. re-establish a0 (+ draft) by feeding the last accepted token
            last = accepted[-1]
            a0 = (await self._send(model, torch.tensor([[last]], dtype=torch.long), cur + m, False))[0, -1]
            await asyncio.to_thread(self._draft_crop, model, cur + m)
            d_logits = await asyncio.to_thread(self._draft_step, model, last, cur + m)
            cur += m + 1
        yield None, "length"

    # -- MTP (nextn) self-speculation (#91) — the checkpoint's own draft head -------------------
    async def _ensure_mtp_head(self, model: LoadedModel):
        """Lazily build + cache the controller-resident MTP head for a model whose checkpoint ships
        one (mtp_num_hidden_layers>0). Returns the head or None (no MTP / load failed). The head is
        SMALL (embed + 1 layer + lm_head, a few GB) — NEVER the full model (see
        never-full-load-on-controller-box). First speculative request pays the one-time build."""
        if not hasattr(self, "_mtp_heads"):
            self._mtp_heads = {}
        if model.friendly in self._mtp_heads:
            return self._mtp_heads[model.friendly]
        d = await asyncio.to_thread(_controller_model_dir, model.target_id)

        def _has_mtp() -> bool:
            try:
                with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
                    cfg = json.load(fh)
                tc = cfg.get("text_config", cfg)
                return int(tc.get("mtp_num_hidden_layers", 0) or 0) > 0
            except Exception:
                return False

        if not await asyncio.to_thread(_has_mtp):
            self._mtp_heads[model.friendly] = None    # negative-cache: don't re-check every request
            return None
        try:
            import mtp_core
            head = await asyncio.to_thread(mtp_core.load_mtp_head, d, "cpu")
        except Exception as exc:
            log_activity(f"{model.friendly}: MTP head load failed ({exc!r}) — plain decode")
            self._mtp_heads[model.friendly] = None
            return None
        self._mtp_heads[model.friendly] = head
        log_activity(f"{model.friendly}: MTP self-speculation head ready (K=1)")
        return head

    async def _decode_spec_mtp(self, model, prompt_ids, max_new, head):
        """#91 MTP self-speculative greedy decode. Each round: the main model's next token t comes
        from the verified context; the MTP head drafts ONE more token d (the next-next) from the
        trunk hidden + t; we verify [t, d] in ONE pipeline traversal (all_logits + capture_pre_norm)
        and accept d iff it equals the target's greedy — so every emitted token is identical to
        plain greedy (bit-exact). On accept the next state comes free from the verify pass (2 tokens
        / 1 traversal); on reject we emit the target's correct token and re-feed it (2 tokens / 2
        traversals). At ~84% accept that's ~1.7x fewer traversals == ~1.7x decode speedup."""
        import mtp_core
        import torch
        if not prompt_ids:
            yield None, "stop"
            return
        eos = model.eos_ids
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0

        def _mask(row):
            if ntok and ntok < int(row.shape[-1]):
                row = row.clone()
                row[ntok:] = float("-inf")
            return row

        from transformers import DynamicCache
        # Prefill the MAIN model: per-position logits + PRE-norm hidden for the whole prompt.
        ml, h_pre = await self._send(model, torch.tensor([prompt_ids], dtype=torch.long), 0, True,
                                     all_logits=True, capture_pre_norm=True)
        P = len(prompt_ids)
        a0 = ml[0, P - 1]                     # logits predicting the token at position P
        h_prev = h_pre[:, P - 1:P, :]         # trunk hidden at position P-1
        # Prefill the MTP layer's OWN KV over the prompt so decode drafts attend the right context.
        mtp_kv = DynamicCache()
        if P >= 2:
            await asyncio.to_thread(mtp_core.mtp_prefill, head, h_pre[:, 0:P - 1, :],
                                    torch.tensor([prompt_ids[1:P]], dtype=torch.long), mtp_kv)
        mtp_len = P - 1                        # MTP-seq positions consumed so far (invariant: == cur-1)
        cur = P
        model.kv_pos = cur
        produced = 0
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            t = int(_mask(a0).argmax())
            produced += 1
            if t in eos:
                yield None, "stop"
                return
            yield t, None
            if produced >= max_new:
                return
            # Draft t_{cur+1}: consume t into the MTP cache (attends the prefilled + prior context).
            draft_row = await asyncio.to_thread(mtp_core.mtp_step, head, mtp_kv, h_prev, t, mtp_len)
            mtp_len += 1
            d = int(_mask(draft_row).argmax())
            # verify [t, d] in one traversal; capture per-position logits + pre-norm hidden.
            V, H = await self._send(model, torch.tensor([[t, d]], dtype=torch.long), cur, False,
                                    all_logits=True, capture_pre_norm=True)
            tgt1 = int(_mask(V[0, 0]).argmax())          # target greedy for position cur+1
            if d == tgt1:                                # accept: next state is free from the verify
                second = d
                next_a0, next_h, refeed = V[0, 1], H[:, 1:2, :], False
            else:                                        # reject: emit target's token, drop wrong d
                second = tgt1
                await self._crop(model, cur + 1)
                next_a0 = next_h = None
                refeed = True
            produced += 1
            if second in eos:
                yield None, "stop"
                return
            yield second, None
            if produced >= max_new:
                return
            # Commit `second` to the MTP cache (h_cur=H[0,0]) so subsequent drafts see it.
            await asyncio.to_thread(mtp_core.mtp_step, head, mtp_kv, H[:, 0:1, :], second, mtp_len)
            mtp_len += 1
            if refeed:                                   # re-establish a0/h by feeding the real token
                a0t, h_t = await self._send(model, torch.tensor([[second]], dtype=torch.long),
                                            cur + 1, False, capture_pre_norm=True)
                a0, h_prev = a0t[0, -1], h_t[:, -1:, :]
            else:
                a0, h_prev = next_a0, next_h
            cur += 2
            model.kv_pos = cur
        yield None, "length"

    async def reconfigure(self, friendly: str, tp: int, ctx: int, quant: str,
                          consolidate: bool, prefer_vram: bool, cpu_only: bool) -> LoadedModel:
        """#88 managed reload: switch a RESIDENT model to/from tensor-parallel (or change its TP
        width / ctx / quant) as ONE operation, rolling back to a WORKING pipeline copy if the new
        layout fails — so the model is NEVER left evicted-with-nothing-loaded. Reuses engine.load
        (force=True) for all the eviction/placement/guardrail logic (the worker wire has no in-place
        resharding, so a layout switch is inherently a re-stream); the only new behavior is the
        snapshot + rollback + the in-flight 'reconfiguring' marker for the dashboard. Does NOT hold
        self.lock itself — engine.load acquires it (asyncio.Lock is not reentrant)."""
        prev = self.models.get(friendly)
        if prev is None:
            raise ValueError(f"'{friendly}' is not resident — load it before reconfiguring")
        prev_tp, prev_ctx, prev_quant = getattr(prev, "tp_size", 1), prev.ctx, (prev.quant or "none")
        from_label = f"tp{prev_tp}" if prev_tp > 1 else "pipeline"
        to_label = ((f"tp{tp}" + ("-cpu" if cpu_only else "")) if tp > 1 else "pipeline")
        self.reconfiguring = {"model": _ollama_name(friendly), "from": from_label, "to": to_label,
                              "from_tp": prev_tp, "to_tp": tp}
        log_activity(f"{friendly}: RECONFIGURE {from_label} -> {to_label} "
                     f"(ctx {prev_ctx}->{ctx}, quant {prev_quant}->{quant})")
        try:
            lm = await self.load(friendly, ctx, consolidate=consolidate, prefer_vram=prefer_vram,
                                 quant=quant, tp=tp, cpu_only=cpu_only, force=True)
            log_activity(f"{friendly}: reconfigured -> {lm.plan_basis}")
            return lm
        except Exception as exc:
            # New layout failed; engine.load already evicted the old copy. Restore a WORKING copy: a
            # plain pipeline-auto load at the PREVIOUS ctx/quant (GPU-first, spills to CPU — the robust
            # path that always places). Better than leaving the model gone.
            log_activity(f"{friendly}: reconfigure to {to_label} FAILED ({exc!r}) -> rolling back to "
                         f"pipeline @ ctx={prev_ctx} {prev_quant}")
            try:
                await self.load(friendly, prev_ctx, quant=prev_quant, force=True)
                log_activity(f"{friendly}: rolled back to a pipeline copy (serving restored)")
            except Exception as exc2:
                log_activity(f"{friendly}: ROLLBACK ALSO FAILED ({exc2!r}) — model is NOT resident")
                raise RuntimeError(f"reconfigure failed AND rollback failed: {exc} || {exc2}")
            raise RuntimeError(f"reconfigure to {to_label} failed: {exc} (rolled back to pipeline)")
        finally:
            self.reconfiguring = None

    async def unload(self, friendly: Optional[str] = None, force: bool = False) -> None:
        async with self.lock:
            if friendly is None:
                # Blanket teardown sends EVERY worker a model-less unload (drops ALL shards) — including
                # any in-flight load's half-built ones. The decision is made HERE, under self.lock, so
                # it's atomic with a load registering its card at its first line: if a load is in flight
                # we refuse rather than nuke it mid-stream (the unload-all TOCTOU — a check at the HTTP
                # layer races the load's released-lock gather window). force= overrides (shutdown).
                if self.loadings and not force:
                    raise LoadInProgressError(sorted(self.loadings))
                await self._unload_locked("manual unload (all)")
            else:
                # Drop ALL replicas of this user-facing model (friendly, friendly#1, ...). Per-model
                # unload is always allowed — even during another model's load (it only frees that model;
                # it never touches the loading model's shards), which is the whole point of #parallel-load.
                keys = [m.friendly for m in self.replicas_of(friendly)] or [friendly]
                for k in keys:
                    await self._unload_model_locked(k, "manual unload")

    async def _unload_model_locked(self, friendly: str, reason: str) -> None:
        """Evict ONE resident model: tell the nodes holding its stages to drop just that
        shard (worker handles {type:unload,model_id}), clear their assignment, and remove it
        from the registry. Other resident models keep running. Caller holds self.lock."""
        m = self.models.get(friendly)
        if m is None:
            return
        log_activity(f"unload {friendly} ({reason}) — reclaiming "
                     + ", ".join(f"{s.hostname} ~{s.est_gb:.1f}GB" for s in m.plan.stages))
        record_unload(friendly, reason, hosts=[s.hostname for s in m.plan.stages])
        loop = asyncio.get_event_loop()
        for nid in m.stage_node_ids:
            link = self.links.get(nid)
            if link:
                fut = loop.create_future()
                link.pending_unload = fut
                with contextlib.suppress(Exception):
                    await link.send({"type": "unload", "model_id": m.target_id})
                    await asyncio.wait_for(fut, timeout=10)  # confirm teardown
            nd = registry._nodes.get(nid)
            if nd:
                nd.clear_assignment()
        if m.stage0_writer is not None:
            with contextlib.suppress(Exception):
                m.stage0_writer.close()
        self._unload_draft(m)        # free this model's controller-local draft
        if getattr(self, "_mtp_heads", None):
            self._mtp_heads.pop(friendly, None)   # #91 free the controller-resident MTP head
        self.models.pop(friendly, None)
        _release_ram()
        print(f"[unload] evicted {friendly} ({reason})")

    async def _unload_locked(self, reason: str) -> None:
        """Tear down the current model on every worker and confirm teardown. Caller holds
        self.lock. Unloads ALL connected workers (not just the loaded stages) so any
        orphaned/stale shard is dropped too."""
        # Capture residents first (invalidate clears them) so we can report the RAM reclaimed.
        freed = [(m.friendly, [(s.hostname, s.est_gb) for s in m.plan.stages])
                 for m in self.models.values()]
        loop = asyncio.get_event_loop()
        for nid in list(self.links):
            link = self.links.get(nid)
            if link:
                fut = loop.create_future()
                link.pending_unload = fut
                with contextlib.suppress(Exception):
                    await link.send({"type": "unload"})
                    await asyncio.wait_for(fut, timeout=10)  # confirm teardown
        for n in registry._nodes.values():
            n.clear_assignment()
        self.invalidate(reason)
        # Workers have acked (shard dropped + gc'd + mmap released) -> report what was reclaimed.
        for friendly, hosts in freed:
            log_activity(f"unloaded {friendly} ({reason}) — reclaimed "
                         + ", ".join(f"{h} ~{gb:.1f}GB" for h, gb in hosts))
            record_unload(friendly, reason, hosts=[h for h, _gb in hosts])
        if not freed:
            log_activity(f"unload-all ({reason}) — no resident models; cleared any orphan shards")

    async def _await_free_refresh(self, timeout: float = 12.0) -> None:
        """After unloading, wait for workers to gc and report FRESH free RAM (via the next
        heartbeat) so the planner budgets against true free memory, not RAM the old model
        still held. Waits until every alive, capable node has heartbeated since this call
        began (so free_mem_gb is post-unload), capped at `timeout`."""
        since = time.time()
        await asyncio.sleep(1.0)   # give workers a moment to gc / release mmaps
        while time.time() - since < timeout:
            nodes = [n for n in registry.alive_sorted() if n.can_infer]
            if nodes and all(n.last_heartbeat >= since for n in nodes):
                return
            await asyncio.sleep(0.5)


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


# ---------------------------------------------------------------------------
# Resilient TCP listener — manual accept loop that survives per-accept OSError
# ---------------------------------------------------------------------------
#
# asyncio.start_server() runs an internal accept loop inside the event loop's
# transport layer. On Windows (Proactor/IOCP) a single failed accept — e.g.
# WinError 64 "The specified network name is no longer available" during a
# reconnect storm — raises out of the IocpProactor.accept() task and SILENTLY
# kills that accept loop. serve_forever() keeps awaiting but no further
# connections are ever accepted (observed: 0 nodes for 7+ minutes after a
# self-update restart, only fixed by a manual controller restart).
#
# Fix: drive accept() ourselves with loop.sock_accept() so we own a per-accept
# try/except. A transient OSError on one accept is logged and skipped; the loop
# keeps running. Accepted raw sockets are bridged to StreamReader/StreamWriter
# exactly the way start_server would, so the existing (reader, writer) handlers
# (handle_control / EngineState._on_data) are unchanged.

class _ResilientServer:
    """Drop-in-ish replacement for asyncio.start_server() whose accept loop does
    NOT die on a transient per-accept OSError. Exposes .close() / .wait_closed()
    so existing shutdown code (ctrl.close(); await ctrl.wait_closed()) keeps
    working, plus .task for cancellation alongside the other lifespan tasks."""

    def __init__(self, sock: socket.socket, name: str) -> None:
        self._sock = sock
        self._name = name
        self.task: Optional[asyncio.Task] = None   # the accept loop task

    def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
        with contextlib.suppress(Exception):
            self._sock.close()

    async def wait_closed(self) -> None:
        if self.task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self.task


async def _resilient_serve(host: Optional[str], port: int, handler, name: str) -> _ResilientServer:
    """Bind a listening socket on (host, port) and start a fault-tolerant accept
    loop that dispatches each connection to `handler(reader, writer)`. Returns a
    _ResilientServer once the socket is listening (the accept loop runs as a
    background task). A per-accept OSError is logged and the loop continues — a
    single bad/aborted connection can never take the listener down."""
    loop = asyncio.get_running_loop()
    # Strong refs to in-flight per-connection bridge tasks. asyncio keeps only a WEAK ref to a bare
    # create_task(), so an un-referenced _bridge task can be garbage-collected mid-await -> the
    # "Task was destroyed but it is pending!" warning (and a prematurely dropped connection). Hold a
    # ref until each finishes (discard on done) so the GC can't reap a live connection's handler.
    _bridge_tasks: set = set()

    # Resolve the bind address the same way start_server does (None -> all
    # interfaces). Use getaddrinfo so an explicit host (e.g. "0.0.0.0" or a
    # hostname) and IPv4/IPv6 both work; bind the first usable result.
    bind_host = host if host else None
    infos = socket.getaddrinfo(bind_host, port, type=socket.SOCK_STREAM,
                               flags=socket.AI_PASSIVE)
    lsock: Optional[socket.socket] = None
    last_err: Optional[Exception] = None
    for af, socktype, proto, _canon, sockaddr in infos:
        try:
            s = socket.socket(af, socktype, proto)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
            s.listen(128)
            s.setblocking(False)
            lsock = s
            break
        except OSError as e:
            last_err = e
            with contextlib.suppress(Exception):
                s.close()
            continue
    if lsock is None:
        raise OSError(f"_resilient_serve({name}): could not bind {bind_host}:{port}: {last_err!r}")

    async def _bridge(conn: socket.socket) -> None:
        # Wrap a raw accepted socket in the canonical StreamReader/StreamWriter
        # pair, exactly as asyncio.start_server does internally, then hand off to
        # the existing handler. Guarded so one bad socket can't kill the loop.
        try:
            conn.setblocking(False)
            reader = asyncio.StreamReader()
            proto = asyncio.StreamReaderProtocol(reader)
            transport, _ = await loop.connect_accepted_socket(lambda: proto, conn)
            writer = asyncio.StreamWriter(transport, proto, reader, loop)
            await handler(reader, writer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[{name}] connection bridge error: {exc!r}")
            with contextlib.suppress(Exception):
                conn.close()

    async def _accept_loop() -> None:
        print(f"[*] {name} resilient accept loop on {host}:{port}")
        while True:
            try:
                conn, _addr = await loop.sock_accept(lsock)
            except asyncio.CancelledError:
                raise
            except OSError as e:
                # THE core fix: a transient accept failure (WinError 64, ECONNRESET,
                # EMFILE, etc.) must NOT kill the listener. Log it and keep going.
                print(f"[{name}] accept failed (transient, listener survives): {e!r}")
                await asyncio.sleep(0.05)
                continue
            except Exception as e:  # pragma: no cover — be paranoid for infra
                print(f"[{name}] accept unexpected error (listener survives): {e!r}")
                await asyncio.sleep(0.05)
                continue
            _bt = asyncio.create_task(_bridge(conn))   # keep a strong ref until done (no GC mid-flight)
            _bridge_tasks.add(_bt)
            _bt.add_done_callback(_bridge_tasks.discard)

    server = _ResilientServer(lsock, name)
    server.task = asyncio.create_task(_accept_loop())
    return server


# ---------------------------------------------------------------------------
# Control plane (bidirectional: heartbeats/responses in, commands out)
# ---------------------------------------------------------------------------

async def handle_control(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    peer_host = peer[0] if peer else "?"
    node: Optional[Node] = None
    try:
        line = await reader.readline()
        if not line:
            return
        msg = json.loads(line.decode())
        if msg.get("type") != "register":
            writer.write(_enc({"type": "error", "error": "expected register"}))
            await writer.drain()
            return
        node = await registry.add(msg, peer_host=peer_host)
        net_account(node.node_id, from_node=len(line))  # register msg, node -> controller
        engine.links[node.node_id] = ControlLink(node.node_id, writer)
        reg_reply = _enc({"type": "registered", "node_id": node.node_id,
                          "os_reserve_gb": ARGS.os_reserve_gb, "server_version": VERSION})
        writer.write(reg_reply)
        await writer.drain()
        net_account(node.node_id, to_node=len(reg_reply))  # controller -> node
        print(f"[+] joined  {node.node_id}  {node.hostname}  {node.device}  "
              f"{node.usable_mem_gb:.1f} GB usable  ({peer_host})")

        # Worker-restart auto-recovery (#77): if this same physical worker (same data endpoint +
        # hostname) was ALREADY registered under an older node id, it RESTARTED and re-registered
        # before its old link was noticed dropped. That restart wiped the worker's shard + KV, so
        # any model the old entry held is dead — recover it now (fail fast) instead of letting the
        # next generate hang out GEN_TIMEOUT on the stale stage, then drop the stale entry. The
        # fork's lazy-connect fix (client m4c10) heals the worker's OWN sockets on first send; this
        # is the controller side that frees the registry/model state the restart invalidated.
        for stale in await registry.find_stale_dupes(node):
            n_rec = await engine.recover_node_restart(stale, node.node_id)
            await registry.remove(stale.node_id)
            print(f"[+] {node.hostname} re-registered as {node.node_id} (was {stale.node_id}); "
                  f"recovered {n_rec} model(s) the restart invalidated")

        link = engine.links[node.node_id]
        while True:
            line = await reader.readline()
            if not line:
                break
            net_account(node.node_id, from_node=len(line))  # node -> controller
            msg = json.loads(line.decode())
            mtype = msg.get("type")
            if mtype == "heartbeat":
                # Net rates are measured by the controller itself (NODE_NET +
                # metrics_sampler); we deliberately ignore any client-reported
                # net_in_bps/net_out_bps here — the server watches its own wire.
                await registry.heartbeat(node.node_id, float(msg.get("free_mem_gb", 0.0)),
                                         float(msg.get("cpu_percent", 0.0)),
                                         float(msg.get("free_disk_gb", 0.0)))
                if "proc_rss_gb" in msg:    # worker python RSS (engine-memory split)
                    node.proc_rss_gb = float(msg.get("proc_rss_gb", 0.0))
                if "vram_used_gb" in msg:   # worker-reported GPU memory
                    node.vram_used_gb = float(msg.get("vram_used_gb", 0.0))
                    node.vram_total_gb = float(msg.get("vram_total_gb", node.vram_total_gb))
                if "gpu_util" in msg:       # worker-reported GPU compute utilization %
                    node.gpu_util = float(msg.get("gpu_util", 0.0))
                if "net_peers" in msg:      # worker per-peer data-plane bytes (bandwidth page)
                    node.peer_bytes = msg.get("net_peers") or {}
                if msg.get("logs"):         # #logs: worker relayed its new stdout/stderr lines
                    _lb = NODE_LOGS.setdefault(node.node_id, [])
                    _lb.extend(str(x) for x in msg["logs"])
                    if len(_lb) > NODE_LOGS_MAX:
                        del _lb[:len(_lb) - NODE_LOGS_MAX]
            elif mtype == "error" and msg.get("req_id") in engine._pack_futures:
                # #distributed-packing: a worker PACK failed. Resolve its pack future with the error
                # NOW so the caller (pack_probe / compile_dist _dispatch_layer) fails fast and falls
                # back to a local pack — instead of blocking on it for the full per-unit timeout.
                f = engine._pack_futures.get(msg.get("req_id"))
                if f is not None and not f.done():
                    f.set_exception(RuntimeError(f"{node.hostname} pack failed: {msg.get('error')}"))
            elif mtype == "packed":
                # success ack; the packed unit itself arrives via POST /pack_result (which resolves
                # the future). Nothing to do here — kept so it isn't mistaken for a load reply.
                pass
            elif mtype in ("ready", "error"):
                if link.pending_load and not link.pending_load.done():
                    link.pending_load.set_result(msg)
            elif mtype == "unloaded":
                if link.pending_unload and not link.pending_unload.done():
                    link.pending_unload.set_result(msg)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    except json.JSONDecodeError as exc:
        print(f"[!] bad control message from {peer_host}: {exc}")
    except Exception as exc:  # pragma: no cover
        print(f"[!] control error from {peer_host}: {exc!r}")
    finally:
        if node is not None:
            link = engine.links.pop(node.node_id, None)
            # If this node dropped while a load/unload was awaiting its reply, fail
            # that future now — otherwise the orchestrator blocks on it for the full
            # 900s timeout (a worker dying mid-load would hang the whole load).
            if link is not None:
                for fut in (link.pending_load, link.pending_unload):
                    if fut is not None and not fut.done():
                        fut.set_exception(ConnectionError(
                            f"{node.hostname} disconnected mid-operation"))
            for fr in [fr for fr, m in engine.models.items()
                       if node.node_id in m.stage_node_ids]:
                engine.invalidate_model(fr, f"node {node.node_id} ({node.hostname}) left")
            await registry.remove(node.node_id)
            print(f"[-] left    {node.node_id}  {node.hostname}")
        with contextlib.suppress(Exception):
            writer.close()


async def reaper_loop() -> None:
    while True:
        await asyncio.sleep(REAPER_INTERVAL_S)
        for n in await registry.reap_dead():
            print(f"[-] reaped {n.node_id} ({n.hostname}, heartbeat timeout)")
            for fr in [fr for fr, m in engine.models.items() if n.node_id in m.stage_node_ids]:
                engine.invalidate_model(fr, f"node {n.hostname} reaped (heartbeat timeout)")


async def gen_stall_watchdog() -> None:
    """#gen-stall-watchdog: reclaim a model WEDGED on a dead pipeline hop. When an inter-worker hop of
    a distributed generation dies (idle-socket death / a flaky node), the decode produces 0 tokens with
    an idle data plane, yet model.active stays >0 — and a disconnecting client can leak it entirely — so
    the model shows BUSY forever and new requests queue behind a ghost. If a loaded model has active>0
    but has emitted NO token for GEN_STALL_S, cancel its in-flight request(s), reset its slot/queue
    counters, and swap in a FRESH per-model lock (an orphaned wedged gen may still 'hold' the old one,
    which would block every queued/new request). The next request re-flows the pipeline and the worker
    reconnects the dead hop (#distributed-gen-idle-socket-death). Threshold > worst-case legit
    first-token wait so a slow big-model prefill is never false-killed."""
    while True:
        await asyncio.sleep(20.0)
        try:
            stall_s = float(ENGINE_CONFIG.get("gen_stall_s", GEN_STALL_S))
        except (TypeError, ValueError):
            stall_s = GEN_STALL_S
        if stall_s <= 0:
            continue   # watchdog disabled
        now = time.time()
        for key, m in list(engine.models.items()):
            if getattr(m, "active", 0) <= 0:
                continue
            last = getattr(m, "last_token_ts", 0.0) or 0.0
            if last <= 0 or (now - last) <= stall_s:
                continue
            idle = int(now - last)
            cancelled = 0
            for r in list(INFLIGHT.values()):
                try:
                    rf = resolve_model_name(r.get("model", ""))
                except Exception:
                    rf = r.get("model")
                if rf in (key, getattr(m, "base", "") or key, getattr(m, "friendly", key)):
                    r["cancel"] = True
                    t = r.get("task")
                    if t is not None and not t.done():
                        with contextlib.suppress(Exception):
                            t.cancel()
                    _inflight_release(r)
                    cancelled += 1
            old_active = m.active
            m.active = 0
            m.queued = 0
            m.last_tok_s = 0.0
            m.last_token_ts = now
            with contextlib.suppress(Exception):
                m.lock = asyncio.Lock()   # drop a lock an orphaned wedged gen may still hold -> unblock the queue
            engine._last_load_failure = time.time()   # arm the self-update cool-down (anti-churn after a fault)
            log_activity(f"gen-stall watchdog: {_ollama_name(key)} wedged — no token for {idle}s "
                         f"(active {old_active}, cancelled {cancelled} req) — reclaimed slot + reset pipeline lock")


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
                     _to_id_list, _safe_decode,
                     _parse_params, _parse_tool_calls, _strip_reasoning, _tool_instruction,
                     _anth_id, _anth_flatten, _anthropic_messages_to_chat,
                     _expand_image_placeholders, _mrope_position_ids, _audio_position_ids,
                     _anthropic_tools_to_hf, _tool_to_block, _extract_tools,
                     _partial_suffix_len, _segment_tools, _estimate_tokens)   # noqa: E402,F401


def _tag_entry(friendly: str) -> dict:
    target, _draft = MODELS[friendly]
    spec = resolve_spec(target)   # built-in or config-derived (custom models)
    size = _display_weight_bytes(target, spec) if spec else 0
    details = _details(spec) if spec else {
        "parent_model": "", "format": "safetensors", "family": "unknown",
        "families": ["unknown"], "parameter_size": "", "quantization_level": "BF16"}
    disp = _ollama_name(friendly)   # 'family:size' (the size IS the tag — no ':latest' on top)
    return {
        "name": disp, "model": disp,
        "modified_at": _iso(START_TIME), "size": size,
        "digest": _digest(target), "details": details,
        "infinitemodel": {"target": target, "draft": _draft, "distributed": True},
    }


# ---------------------------------------------------------------------------
# Status + HTTP API + dashboard
# ---------------------------------------------------------------------------

def _loading_view(ld: Optional[dict]) -> Optional[dict]:
    """Enrich a live load/compile card (from engine.loadings / engine.compiling) with a timer
    (elapsed + ETA) without mutating the original. ETA = elapsed * (1-frac)/frac from progress (ready/total);
    only shown once a few percent in (early fractions give wild estimates). Returns a shallow
    copy so the live object the load loop mutates stays clean. None -> None (no load running)."""
    if not ld:
        return ld
    started = ld.get("started")
    out = dict(ld)
    if started:
        elapsed = max(0.0, time.time() - started)
        out["elapsed_s"] = round(elapsed, 1)
        total = ld.get("total") or 0
        ready = ld.get("ready") or 0
        frac = (ready / total) if total > 0 else 0.0
        # need a stable-ish fraction before an ETA is meaningful (>=3% in); cap at 4h display
        out["eta_s"] = round(min(4 * 3600, elapsed * (1 - frac) / frac)) if frac >= 0.03 else None
    return out


def build_status() -> dict:
    nodes = registry.alive_sorted()
    # Pool aggregate respects the per-node CPU/GPU tier toggles: deselecting a node's CPU
    # drops its RAM from total/used/free; deselecting its GPU drops its VRAM. Per-node rows
    # still show the machine's real memory (heartbeats keep polling it regardless of tier).
    pool_total = sum((n.total_mem_gb if n.ram_enabled else 0.0)
                     + (n.vram_total_gb if n.vram_enabled else 0.0) for n in nodes)
    pool_ram = sum(n.eff_ram_gb for n in nodes)    # enabled tiers only (planner budget)
    pool_vram = sum(n.eff_vram_gb for n in nodes)  # disabled VRAM leaves the pool
    pool_usable = pool_ram + pool_vram            # usable pool now includes VRAM
    # LIVE physical used/free against the STABLE total — so the dashboard shows usage CLIMBING
    # as models load, not the total shrinking. RAM used = total-free (heartbeat); VRAM used = heartbeat.
    pool_used = sum(((n.total_mem_gb - n.free_mem_gb) if n.ram_enabled else 0.0)
                    + (n.vram_used_gb if n.vram_enabled else 0.0) for n in nodes)
    pool_used = max(0.0, min(pool_total, pool_used))
    pool_free = pool_total - pool_used
    # Split the LIVE physical free into RAM vs VRAM (what's available in what FORM), tier-aware.
    pool_ram_free = max(0.0, sum((n.free_mem_gb if n.ram_enabled else 0.0) for n in nodes))
    pool_vram_free = max(0.0, sum(((n.vram_total_gb - n.vram_used_gb) if n.vram_enabled else 0.0)
                                  for n in nodes))
    # Split "used" into ENGINE (our python + loaded-model footprint) vs OS/other. mmap subtlety:
    # safetensors weights are memory-mapped, so until faulted in they sit in the OS PAGE CACHE —
    # reported as reclaimable/'available' (so live free_mem HIDES them) and absent from proc_rss.
    # So the live used/engine UNDERCOUNT a loaded model's RAM weights (they leak into 'free', and
    # 'engine' showed smaller than the model card). Fix: size the models from their SPEC (the same
    # bytes the LOADED MODEL card reports), add the cache-hidden RAM back into 'used', and count
    # each model's RAM ONCE — faulted into RSS OR sitting in cache — via max(). Reconciles the bar.
    try:
        import psutil as _ps
        ctrl_rss_gb = _ps.Process().memory_info().rss / GB
    except Exception:
        ctrl_rss_gb = 0.0
    worker_rss_gb = sum(n.proc_rss_gb for n in nodes if n.ram_enabled)
    model_weight_gb = sum(lm.spec.total_weight_bytes for lm in engine.models.values()) / GB
    model_kv_gb = sum(lm.spec.kv_bytes_per_layer(lm.kv_pos) * lm.spec.num_layers
                      for lm in engine.models.values()) / GB
    model_vram_gb = sum(s.gpu_bytes for lm in engine.models.values()
                        for s in lm.plan.stages) / GB
    model_ram_gb = max(0.0, model_weight_gb - model_vram_gb) + model_kv_gb   # model's RAM commitment
    mmap_hidden = max(0.0, model_ram_gb - worker_rss_gb)   # model RAM in page-cache (hidden in 'free')
    pool_used = max(0.0, min(pool_total, pool_used + mmap_hidden))
    pool_free = pool_total - pool_used
    pool_ram_free = max(0.0, pool_ram_free - mmap_hidden)   # the hidden weights are RAM, not free
    pool_engine = max(0.0, min(pool_used,
                               ctrl_rss_gb + max(worker_rss_gb, model_ram_gb) + model_vram_gb))
    pool_os = max(0.0, pool_used - pool_engine)
    resident = list(engine.models.values())
    primary = max(resident, key=lambda m: m.last_used) if resident else None

    def _loaded_dict(lm: LoadedModel) -> dict:
        # Report WEIGHTS and KV separately so the card reconciles with the pool's engine
        # bar (measured), instead of the old est_bytes which folded in the FULL ctx KV
        # reservation (mostly empty at low ctx -> looked far bigger than measured usage).
        #   weights: VRAM = worker-reported on-GPU bytes (measured); RAM = the rest of the
        #            model's weights (mmap-backed, in the OS page cache until faulted in).
        #   KV: reserved for the full ctx vs actually used so far (from kv_pos). KV is
        #       allocated lazily during generation, so at ctx 0 it's ~empty.
        vram_used = sum(s.gpu_bytes for s in lm.plan.stages)
        ram_weights = max(0, lm.spec.total_weight_bytes - vram_used)
        kv_reserved = lm.spec.kv_bytes_per_layer(lm.ctx) * lm.spec.num_layers
        kv_used = lm.spec.kv_bytes_per_layer(lm.kv_pos) * lm.spec.num_layers
        _arch = (getattr(lm.spec, "arch", "") or "").lower()
        # best-effort MoE flag for the detail modal: fused/per-expert MoE arches all contain one of
        # these tokens ('moe' covers olmoe/qwen3*_moe; mixtral/minimax/deepseek_v2,v3 named directly).
        _is_moe = any(k in _arch for k in ("moe", "mixtral", "minimax", "deepseek_v"))
        return {
            "friendly": lm.friendly, "display_name": _ollama_name(lm.friendly),  # 'qwen3:4b'
            "target": lm.target_id, "ctx": lm.ctx,
            "base": lm.base or lm.friendly, "replica_idx": lm.replica_idx,  # data-parallel (#39)
            "active": lm.active, "queued": lm.queued,   # per-replica live load (#39 routing)
            "kv_pos": lm.kv_pos,   # tokens in the current/last generation's KV context
            # LIVE decode tok/s: the most-recent-gen rate WHILE generating, else 0 when idle —
            # last_tok_s lingers at its last value forever otherwise (card looked "busy" when idle).
            # The historical rate stays visible as ema_tok_s ("avg"). active==0 => idle => 0.
            "tok_s": round(lm.last_tok_s if lm.active > 0 else 0.0, 2),   # decode tok/s, live (#46)
            "ema_tok_s": round(lm.ema_tok_s, 2),     # smoothed decode tok/s across gens, historical (#46)
            "quant": lm.quant,     # the quant this model was loaded with (none/int8)
            "tp_size": getattr(lm, "tp_size", 1),            # #88: TP width (1 = pipeline)
            "is_tp": getattr(lm, "tp_size", 1) > 1,          # #88: card shows TP vs pipeline + reconfigure
            "num_layers": lm.spec.num_layers, "params": _human_params(lm.spec),
            "size_gb": round(lm.spec.total_weight_bytes / GB, 2),
            "vram_used_gb": round(vram_used / GB, 2),
            "ram_used_gb": round(ram_weights / GB, 2),       # weights resident in RAM (not KV)
            # #cpu-bound-visibility: ACTUAL fraction of WEIGHTS on CPU (from worker-reported gpu_bytes,
            # not the pre-load estimate). A high value = the model is CPU-bound and will decode SLOWLY
            # (CPU layers are ~50-100x a GPU layer) — the dashboard badges it so "slow" isn't read as
            # "hung/wedged". This is the real cause of a multi-model fleet's later loads crawling.
            "cpu_frac": round(ram_weights / lm.spec.total_weight_bytes, 3) if lm.spec.total_weight_bytes else 0.0,
            "kv_reserved_gb": round(kv_reserved / GB, 2),    # KV space reserved for the full ctx
            "kv_used_gb": round(kv_used / GB, 2),            # KV actually used so far (kv_pos)
            "loaded_at": _iso(lm.loaded_at),
            "loaded_at_ts": lm.loaded_at,                    # epoch s -> live uptime in the modal
            "last_used_ts": lm.last_used,                    # epoch s -> "idle for ..." in the modal
            "plan_basis": getattr(lm, "plan_basis", ""),   # placement basis (#65)
            "warnings": getattr(lm, "load_warnings", []),  # pre-load guardrail (#76)
            "speed_tier": (getattr(lm, "load_assess", {}) or {}).get("speed_tier", ""),
            # --- #model-detail (click-to-expand modal): arch/tags + lifetime stats ---
            "arch": getattr(lm.spec, "arch", ""),
            "is_moe": _is_moe,
            "is_embedding": bool(getattr(lm.spec, "is_embedding", False)),
            "load_seconds": round(getattr(lm, "load_seconds", 0.0), 1),
            "req_total": getattr(lm, "req_total", 0),
            "tok_in_total": getattr(lm, "tok_in_total", 0),
            "tok_out_total": getattr(lm, "tok_out_total", 0),
            "max_tok_s": round(getattr(lm, "max_tok_s", 0.0), 2),
            "stages": [s.to_dict() for s in lm.plan.stages],
        }
    loaded = _loaded_dict(primary) if primary else None         # active model (dashboard panel)
    loaded_models = [_loaded_dict(m) for m in resident]         # ALL resident (multi-model)
    # --- Compute load (#82): how busy the fleet's processors are vs capacity ("out of what
    # is possible"). CPU load is capacity-weighted by logical cores so a busy 32-core box
    # counts more than a busy 4-core box; GPU load averages each enabled GPU's utilization.
    # The combined headline treats every CPU core and every GPU as one comparable compute unit.
    cpu_nodes = [n for n in nodes if n.ram_enabled]
    gpu_nodes = [n for n in nodes if n.vram_total_gb > 0 and n.vram_enabled]
    cpu_cores = sum(max(1, n.cores) for n in cpu_nodes)
    cpu_busy  = sum((n.cpu_percent / 100.0) * max(1, n.cores) for n in cpu_nodes)  # busy-core-equiv
    cpu_load_pct = (100.0 * cpu_busy / cpu_cores) if cpu_cores else 0.0
    gpu_load_pct = (sum(n.gpu_util for n in gpu_nodes) / len(gpu_nodes)) if gpu_nodes else 0.0
    gpu_busy  = sum(n.gpu_util / 100.0 for n in gpu_nodes)                          # busy-GPU-equiv
    units_total = cpu_cores + len(gpu_nodes)
    units_busy  = cpu_busy + gpu_busy
    overall_pct = (100.0 * units_busy / units_total) if units_total else 0.0
    # Disk picture. With chunk serving, workers hold weights in RAM (no model on
    # worker disk), so the model-size ceiling is the CONTROLLER's free disk (to
    # hold the full model) AND the RAM pool (to run it) — not the smallest worker.
    # Measure the drive that actually HOLDS the weights (models/ under the program dir),
    # NOT the OS/home drive — they're often different disks (e.g. weights on a big USB/data
    # drive while the OS is on C:). Using ~ reported the wrong drive's free space, so the
    # disk ceiling + fits_disk were computed against a disk the model never touches.
    try:
        _disk_path = MODELS_DIR if os.path.isdir(MODELS_DIR) else _PROJECT_DIR
        ctrl_free_gb = shutil.disk_usage(_disk_path).free / GB
    except Exception:
        ctrl_free_gb = 0.0
    # STABLE capacity (total usable RAM/VRAM, tier-aware) — NOT live free RAM — so the
    # "Max model" card shows the fleet's ceiling and doesn't flicker as free RAM jitters
    # across nodes (esp. shared boxes like BEAST). Actual loads still plan on live free.
    mems = [NodeMem(n.node_id, n.hostname,
                    int(((n.usable_mem_gb if n.ram_enabled else 0.0) + n.eff_vram_gb) * GB),
                    int(n.eff_vram_gb * GB)) for n in nodes]
    servable = []
    for name, (tgt, _d) in MODELS.items():
        spec = resolve_spec(tgt)   # built-in or config-derived (custom, once downloaded)
        if not spec:
            continue               # custom model not yet downloaded -> no estimate yet
        # Downloaded models: size + fit use the REAL measured weights (MoE-correct), and
        # they already occupy disk so they always "fit disk". Undownloaded: formula + the
        # free-disk check (can we pull it).
        dl = model_ready(tgt)
        d = _local_model_dir(tgt) if dl else None
        plan_spec = spec_with_measurements(spec, d) if d else spec
        size_gb = plan_spec.total_weight_bytes / GB
        # Per-quant WEIGHT footprint estimate (#49) so the UI can show what each load option
        # costs and which fit. Cheap (for_quant just rescales weight bytes); the fit hint is a
        # weight-only check vs the planner budget (KV is reserved on top, shown separately).
        # 'none' loads the NATIVE dtype, so its size IS the measured on-disk size (fp32 stays fp32).
        # for_quant's int8/int4 scaling assumes a 2-byte (bf16) base, so for an fp32/fp8 checkpoint
        # normalize the measured bytes to a bf16-equivalent first, else int8/int4 are off by the dtype
        # factor (an fp32 model would show int8/int4 ~2x too big).
        import dataclasses as _dc
        _srcb = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M3": 1}.get(plan_spec.src_dtype, 2)
        _f = 2.0 / _srcb
        _sc = lambda v: (int(v * _f) if v is not None else None)
        qspec = (_dc.replace(plan_spec, meas_layer_w=_sc(plan_spec.meas_layer_w),
                             meas_embed=_sc(plan_spec.meas_embed), meas_head=_sc(plan_spec.meas_head),
                             meas_norm=_sc(plan_spec.meas_norm))
                 if (_f != 1.0 and plan_spec.meas_layer_w is not None) else plan_spec)
        quant_gb = {"none": round(plan_spec.total_weight_bytes / GB, 2),          # native dtype size
                    "int8": round(qspec.for_quant("int8").total_weight_bytes / GB, 2),
                    "int4": round(qspec.for_quant("int4").total_weight_bytes / GB, 2)}
        quant_fits = {q: (g <= round(pool_usable, 2)) for q, g in quant_gb.items()}
        fits_ram = bool(nodes) and plan_pipeline(plan_spec, mems, DEFAULT_CTX).ok
        fits_disk = dl or ctrl_free_gb >= size_gb
        servable.append({"name": _ollama_name(name), "internal_name": name,
                         "size_gb": round(size_gb, 2),
                         "default_ctx": plan_spec.max_ctx,   # native/training context (ctx=0 loads this)
                         "src_dtype": plan_spec.src_dtype,    # on-disk weight dtype (F32/BF16/...) for the UI
                         "quant_gb": quant_gb, "quant_fits": quant_fits,
                         "fits_ram": fits_ram, "fits_disk": fits_disk,
                         "runnable": fits_ram and fits_disk})
    runnable = [m for m in servable if m["runnable"]]
    largest = max(runnable, key=lambda m: m["size_gb"], default=None)
    min_worker_disk = min((n.free_disk_gb for n in nodes), default=0.0)
    # Controller's own wire (server-measured): out = bytes it pushed to nodes
    # (= sum of node ↓), in = bytes it pulled from nodes (= sum of node ↑).
    metrics = metric_rates()
    metrics["ctrl_out_bps"] = round(sum(n.net_in_bps for n in nodes))
    metrics["ctrl_in_bps"] = round(sum(n.net_out_bps for n in nodes))
    # Slots (1 running per model) + queue (waiters), with client IP + elapsed time.
    now = time.time()
    _inflight = sorted(INFLIGHT.values(), key=lambda r: r["enqueued"])
    slots = [{"id": r["id"], "ip": r["ip"], "model": r["model"],
              "running_s": round(now - r["started"], 1) if r["started"] else 0.0}
             for r in _inflight if r["state"] == "running"]
    queue = [{"id": r["id"], "ip": r["ip"], "model": r["model"],
              "waiting_s": round(now - r["enqueued"], 1)}
             for r in _inflight if r["state"] == "queued"]
    return {
        "controller": {
            "hostname": platform.node(), "os": f"{platform.system()} {platform.release()}",
            "version": VERSION, "uptime_s": round(time.time() - START_TIME, 1),
            "wire": ("wire" in sys.modules),   # True once wire.py is imported (not the fallback)
            "dash": ("dashboard_html" in sys.modules),   # True once dashboard_html.py is imported
            "http_port": ARGS.http_port, "control_port": ARGS.control_port,
            "data_port": ARGS.data_port, "os_reserve_gb": ARGS.os_reserve_gb,
            "free_disk_gb": round(ctrl_free_gb, 2),
            "hf_auth": (f"...{HF_TOKEN[-4:]}" if HF_TOKEN else False),
            "max_loaded": ENGINE_CONFIG.get("max_loaded", MAX_LOADED_MODELS),
            "auto_unload": ENGINE_CONFIG.get("auto_unload", True),
            "auto_load": ENGINE_CONFIG.get("auto_load", True),
            "autoload_quant": ENGINE_CONFIG.get("autoload_quant", "int4"),
            "autoload_ctx": ENGINE_CONFIG.get("autoload_ctx", DEFAULT_CTX),
            "autoload_mode": ENGINE_CONFIG.get("autoload_mode", "auto"),
            "vram_weights_first": ENGINE_CONFIG.get("vram_weights_first", True),
            "gen_stall_s": ENGINE_CONFIG.get("gen_stall_s", GEN_STALL_S),
            "queue_depth": ENGINE_CONFIG.get("queue_depth", DEFAULT_QUEUE_DEPTH),
        },
        "pool": {"nodes": len(nodes), "total_gb": round(pool_total, 2),
                 "used_gb": round(pool_used, 2), "free_gb": round(pool_free, 2),  # LIVE physical
                 "engine_gb": round(pool_engine, 2),   # our pythons + shards (RED on the bar)
                 "os_gb": round(pool_os, 2),           # OS/other (BLUE on the bar)
                 "ctrlr_gb": round(ctrl_rss_gb, 2),    # controller process RSS alone
                 "usable_gb": round(pool_usable, 2),   # planner budget (live free, for fit checks)
                 "ram_gb": round(pool_ram, 2), "vram_gb": round(pool_vram, 2),
                 # LIVE physical free split by form (what's available as RAM vs VRAM)
                 "ram_free_gb": round(pool_ram_free, 2), "vram_free_gb": round(pool_vram_free, 2)},
        "compute": {"overall_pct": round(overall_pct, 1),
                    "cpu_pct": round(cpu_load_pct, 1),
                    "cpu_busy_cores": round(cpu_busy, 1), "cpu_cores": int(cpu_cores),
                    "cpu_nodes": len(cpu_nodes),
                    "gpu_pct": round(gpu_load_pct, 1), "gpu_busy": round(gpu_busy, 2),
                    "gpus": len(gpu_nodes),
                    "units_busy": round(units_busy, 1), "units_total": int(units_total)},
        "metrics": metrics,
        "disk": {"controller_free_gb": round(ctrl_free_gb, 2),
                 "min_worker_free_gb": round(min_worker_disk, 2),
                 "largest_model": largest, "models": servable,
                 "note": "chunk serving: workers hold weights in RAM, so model size "
                         "is bounded by controller disk + RAM pool, not worker disks"},
        "cluster": {"state": "loaded" if resident else ("dirty" if registry.dirty else "idle"),
                    "model": primary.friendly if primary else None,
                    "display_model": _ollama_name(primary.friendly) if primary else None,
                    "loaded": loaded,
                    "loaded_models": loaded_models,
                    # parallel loads/compiles -> LISTS of cards (was a single 'loading' dict). 'loading'
                    # stays as the first in-flight load card for any old consumer; 'loadings' (all loads)
                    # + 'compiling' (all shard-compiles) are the full lists the dashboard renders.
                    "loading": next((_loading_view(c) for c in engine.loadings.values()), None),
                    "loadings": [_loading_view(c) for c in engine.loadings.values()],
                    "compiling": [_loading_view(c) for c in engine.compiling.values()],
                    "reconfiguring": getattr(engine, "reconfiguring", None),   # #88 managed reload
                    "slots": slots, "queue": queue,
                    "queue_depth": ENGINE_CONFIG.get("queue_depth", DEFAULT_QUEUE_DEPTH)},
        "models": [_model_entry(name, tgt, draft)
                   for name, (tgt, draft) in MODELS.items()],
        "nodes": [n.to_dict() for n in nodes],
        "activity": list(ACTIVITY),   # newest-first controller activity (dashboard panel)
        "unloads": list(UNLOADS),     # newest-first "why a model left" events (dashboard panel)
    }


def _model_entry(name: str, tgt: str, draft: str) -> dict:
    """Per-model status for the dashboard: ready (weights on controller),
    downloading (pull in flight), or absent. size from the spec when known."""
    spec = resolve_spec(tgt)   # built-in or config-derived (custom models)
    ready = model_ready(tgt)
    if ready:
        status = "ready"
    elif name in DOWNLOADING:                      # live pull; reflect a pending pause/stop
        ctl = DOWNLOAD_CONTROL.get(name)
        status = "pausing" if ctl == "pause" else "stopping" if ctl == "stop" else "downloading"
    elif name in DOWNLOAD_STATE:                    # halted by the user (paused/stopped), cache kept
        status = DOWNLOAD_STATE[name]
    else:
        status = "absent"
    loaded = name in engine.models
    # Size: spec-aware display bytes when we can model the arch; otherwise (e.g. the Omni
    # Thinker-Talker, which resolve_spec can't represent) fall back to the RAW on-disk
    # safetensors total so a downloaded-but-unrunnable model still shows a real size.
    size_bytes = None
    if spec:
        size_bytes = _display_weight_bytes(tgt, spec)
    else:
        d = _local_model_dir(tgt)
        if d:
            m = measure_model_weights(d)
            if m and m.get("total"):
                size_bytes = int(m["total"])
    # Display the Ollama 'family:size' name ('qwen3:4b'); the dashboard sends it back as the
    # op key and resolve_model_name() maps it to this dash-form key. internal_name is the raw
    # registry key, so existing tooling that keys off the dash form keeps working.
    entry = {"name": _ollama_name(name), "internal_name": name,
             "target": tgt, "draft": draft, "ready": ready,
             "status": status, "loaded": loaded,
             "size_gb": round(size_bytes / GB, 2) if size_bytes else None}
    if loaded:                                   # live request queue depth (Inc 4)
        lm = engine.models[name]
        entry["active"] = lm.active              # currently generating (0/1)
        entry["queued"] = lm.queued             # requests waiting on this model's lock
    if status in ("downloading", "pausing", "stopping", "paused", "stopped"):
        pr = DOWNLOAD_PROGRESS.get(name) or {}     # frozen at the halt point for paused/stopped
        dl, tot = pr.get("downloaded", 0), pr.get("total", 0)
        entry["dl_done_gb"] = round(dl / GB, 2)
        entry["dl_total_gb"] = round(tot / GB, 2) if tot else None
        entry["dl_pct"] = round(100 * dl / tot, 1) if tot else None
    elif status == "absent" and name in DOWNLOAD_ERROR:
        entry["dl_error"] = DOWNLOAD_ERROR[name]   # last pull failure (e.g. gated repo)
    if ready:                                       # #shard-cache: which quants are pre-compiled
        d3 = _local_model_dir(tgt)
        cs = shard_cache_status(d3) if d3 else {}
        if cs:
            entry["cached"] = cs   # {quant: {ok, size_gb, files, ...}}
    return entry


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
            # When auto_unload is on, unload any model idle (no requests) for > IDLE_UNLOAD_S. A
            # model mid-generation refreshes last_used every token (engine.generate), so it never
            # looks idle and is never yanked mid-request. Off (default) -> models stay forever.
            while True:
                await asyncio.sleep(60)
                if not bool(ENGINE_CONFIG.get("auto_unload", False)):
                    continue
                now = time.time()
                stale = [fr for fr, m in list(engine.models.items())
                         if now - m.last_used > IDLE_UNLOAD_S]
                for fr in stale:
                    log_activity(f"auto-unload {fr}: idle > {int(IDLE_UNLOAD_S // 60)} min")
                    with contextlib.suppress(Exception):
                        await engine.unload(fr)
        idle_unloader = asyncio.create_task(_idle_unload_loop())
        async def _persist_reload():
            # #77: re-load the PERSISTED models on startup so a resident model survives a controller
            # restart/crash/deploy without a manual reload (workers drop their shards on link loss, so
            # recovery = re-stream). Wait for capable workers + a short settle so GPU models land on
            # the GPU, not CPU (the restart-timing trap). Idempotent / dedup-safe vs auto-load traffic.
            persist = dict(ENGINE_CONFIG.get("persist_models") or {})
            if not persist:
                return
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
                              lambda: not DOWNLOADING and not ENCODING
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
            for t in (serve, reaper, sampler, updater, idle_unloader, persist_reloader):
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

    # ---- dashboard + introspection ----
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @app.get("/bandwidth", response_class=HTMLResponse)
    async def bandwidth_page() -> str:
        return BANDWIDTH_HTML

    @app.get("/bandwidthdata")       # full traffic picture: controller<->node + node<->node
    async def bandwidthdata() -> JSONResponse:
        """Combine the controller's own socket metering (authoritative for controller<->node)
        with each worker's per-peer counters (the ONLY source for node<->node hidden-state
        traffic the controller can't see). Cumulative bytes; the page derives rates. No
        double-counting: controller<->node comes from NODE_NET; node<->node from the sender's
        'out' counter (each directed hop reported once, by its sender)."""
        nodes = registry.alive_sorted()
        ip2host = {n.data_host: n.hostname for n in nodes if getattr(n, "data_host", None)}
        node_rows = []
        for n in nodes:
            c = NODE_NET.get(n.node_id, {})
            pb = (n.peer_bytes or {})
            # node<->node totals from this node's own counters (peer != controller)
            nn_in = sum(int(v.get("in", 0)) for p, v in pb.items() if p != "controller")
            nn_out = sum(int(v.get("out", 0)) for p, v in pb.items() if p != "controller")
            nn_in_p = sum(int(v.get("in_pkts", 0)) for p, v in pb.items() if p != "controller")
            nn_out_p = sum(int(v.get("out_pkts", 0)) for p, v in pb.items() if p != "controller")
            node_rows.append({
                "node_id": n.node_id, "hostname": n.hostname, "alive": n.alive,
                "ctrl_to_node": int(c.get("in", 0)), "node_to_ctrl": int(c.get("out", 0)),  # controller-measured
                "nn_in": nn_in, "nn_out": nn_out,                              # node<->node (worker)
                # "packets" = data-plane frames (each tensor send/recv is one frame)
                "ctrl_to_node_pkts": int(c.get("in_pkts", 0)), "node_to_ctrl_pkts": int(c.get("out_pkts", 0)),
                "nn_in_pkts": nn_in_p, "nn_out_pkts": nn_out_p,
                "net_in_bps": round(n.net_in_bps), "net_out_bps": round(n.net_out_bps)})
        edges = []   # directed node->node hops (sender's out), controller hops excluded
        for n in nodes:
            for peer, v in (n.peer_bytes or {}).items():
                if peer == "controller":
                    continue
                out = int(v.get("out", 0))
                if out > 0:
                    edges.append({"src": n.hostname, "dst": ip2host.get(peer, peer),
                                  "bytes": out, "pkts": int(v.get("out_pkts", 0))})
        return JSONResponse({"controller": _display_host(), "nodes": node_rows, "edges": edges})

    @app.get("/status")
    async def status(graphs: int = 0) -> JSONResponse:
        # graphs=1 attaches a server-rendered SVG sparkline per node (bandwidth + RAM)
        # so the dashboard can drop them straight into the DOM. Off by default to keep
        # /status lean for non-dashboard consumers (the Ollama-compat clients, scripts).
        st = build_status()
        if graphs:
            for nd in st.get("nodes", []):
                h = nd.get("hostname", "?")
                nd["spark_bw"] = _spark_svg(h, "bw")
                nd["spark_ram"] = _spark_svg(h, "ram")
                # GPU VRAM sparkline only for nodes that have a GPU
                if nd.get("vram_total_gb", 0) > 0:
                    nd["spark_vram"] = _spark_svg(h, "vram")
        return JSONResponse(st)

    @app.get("/graph/{kind}/{host}")
    async def graph(kind: str, host: str) -> Response:
        # Larger detail graph for a node, server-rendered (the mini sparkline's
        # click-target). kind in {bw, ram, vram}; anything else is a 404.
        if kind not in ("bw", "ram", "vram"):
            return Response(content="unknown graph kind", status_code=404,
                            media_type="text/plain")
        return Response(content=_detail_svg(host, kind), media_type="image/svg+xml")

    @app.get("/nethistory")
    async def nethistory(since: float = 0.0) -> JSONResponse:
        # Server-stored, disk-persisted per-node traffic graph. since>0 returns only
        # points newer than that ms timestamp (incremental, tiny payloads); since=0
        # returns the full bounded window (initial load / fresh tab).
        since_ms = int(since)
        hosts: dict[str, list] = {}
        for host, dq in NET_HISTORY.items():
            pts = list(dq) if since_ms <= 0 else [p for p in dq if p[0] > since_ms]
            if pts:
                hosts[host] = pts
        return JSONResponse({"sample_s": NET_HIST_SAMPLE_S, "cap": NET_HIST_MAX,
                             "now": int(time.time() * 1000), "hosts": hosts})

    @app.get("/plan")
    async def plan(model: str, ctx: int = 0, quant: str = "none", mode: str = "auto") -> JSONResponse:
        # #60 Preview: same inputs as /load (model, ctx, quant, mode) -> the placement + #76
        # assessment WITHOUT loading. tp modes are pipeline-planned here (TP frees the fleet and
        # plans differently at load); the dashboard tells the user TP preview is approximate.
        # Resolve the name first (like /load) so the Ollama 'family:size' form ('qwen3:4b') the
        # dashboard sends maps to the registry key/target before resolve_spec runs.
        try:
            friendly = resolve_model_name(model)
        except ValueError:
            return JSONResponse({"ok": False, "error": f"unknown model '{model}'"},
                                status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        spec = resolve_spec(target)
        if spec is None:
            return JSONResponse({"ok": False, "error": f"unknown model '{model}'"},
                                status_code=404)
        # Measure REAL safetensors bytes so MoE / any non-dense arch sizes correctly in the
        # PREVIEW too. The dense formula under-counts N experts (~4 GB est for the ~115 GB int4
        # MiniMax-M2, ~3.5 GB for the 66 GB Qwen3.6-35B-A3B), which made Preview claim a huge MoE
        # "fits on 3 GPUs" — wildly diverging from the live load (which DOES measure, line ~2856).
        # No-op if the model isn't downloaded yet. Cached by dir (_MEAS_CACHE).
        _pd = await asyncio.to_thread(_local_model_dir, target)
        if _pd:
            spec = await asyncio.to_thread(spec_with_measurements, spec, _pd)
        if ctx <= 0:   # default to the model's native training context (from spec)
            ctx = spec.max_ctx or DEFAULT_CTX
        spec = spec.for_quant(quant) if quant in ("int8", "int4") else spec   # size the real footprint
        cons, pv = LOAD_MODES.get(mode, LOAD_MODES["auto"])   # mirror /load's mode -> placement flags
        mems = []
        node_by_id = {}
        for n in registry.alive_sorted():
            fv = max(0.0, n.eff_vram_gb - PLAN_VRAM_FLOOR_GB)   # same VRAM floor as live load
            # #78: mirror the live load's controller-box RAM reserve so Preview's pool matches reality
            _ram = n.eff_ram_gb - (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
            mems.append(NodeMem(n.node_id, n.hostname,
                                int((max(0.0, _ram) + fv) * GB), int(fv * GB)))
            node_by_id[n.node_id] = n
        p = plan_pipeline(spec, mems, ctx_len=ctx, consolidate=cons, prefer_vram=pv,
                          spread=(mode == "spread"),
                          proportional=(mode == "proportional"))
        d = p.to_dict()
        if p.ok:   # #60/#76: surface the basis + pre-load assessment so a Preview matches the load
            d["basis"] = _describe_plan(p.stages, node_by_id, False, pv, quant)
            d["assess"] = _assess_placement(spec, ctx, mems, p.stages)
            # #78 guardrail: a CONSOLIDATING mode (auto/single) can pile a heavy shard onto the
            # controller's co-located worker, which must ALSO serve the whole stream -> it OOM-drops
            # mid-load (the beast minimax crash). Flag it so the dashboard offers 'proportional'
            # (spreads across the fleet) in a confirm() BEFORE the load commits. Fires only when the
            # co-located stage's RAM leaves < 2x the controller reserve free on that box.
            if cons and mode != "proportional":
                for s in p.stages:
                    nd = node_by_id.get(s.node_id)
                    if nd is not None and nd.data_host in _LOCAL_IPS:
                        if s.est_gb > (nd.eff_ram_gb - 2 * CONTROLLER_RAM_RESERVE_GB):
                            d["overload"] = {"node": nd.hostname, "mode": mode,
                                             "suggest": "proportional",
                                             "stage_gb": round(s.est_gb, 1),
                                             "node_ram_gb": round(nd.eff_ram_gb, 1)}
                        break
        return JSONResponse(d)

    @app.get("/shard_status")           # #shard-cache: which quants are pre-compiled per model
    async def shard_status_ep(model: Optional[str] = None) -> JSONResponse:
        def _status_for(friendly: str) -> dict:
            tgt = MODELS[friendly][0] if friendly in MODELS else friendly
            d = _local_model_dir(tgt)
            return shard_cache_status(d) if d else {}
        if model:
            try:
                friendly = resolve_model_name(model)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            return JSONResponse({"model": _ollama_name(friendly), "cache": _status_for(friendly)})
        # all registered models that are downloaded (cheap — just reads manifests)
        out = {}
        for friendly in MODELS:
            st = await asyncio.to_thread(_status_for, friendly)
            if st:
                out[_ollama_name(friendly)] = st
        return JSONResponse({"caches": out})

    @app.post("/verify_shards")         # #shard-cache: full sha256 integrity check (for the popup)
    async def verify_shards_ep(model: str, quant: str = "int4") -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_local_model_dir, tgt)
        if not d:
            return JSONResponse({"error": "model not downloaded"}, status_code=404)
        ok, problems = await asyncio.to_thread(verify_shard_cache, d, quant)
        return JSONResponse({"ok": ok, "problems": problems, "quant": quant})

    @app.post("/pack_result")   # #distributed-packing: a worker returns a packed shard-cache unit
    async def pack_result(req: Request, req_id: str = "", unit: str = "",
                          model_id: str = "", quant: str = "int4") -> JSONResponse:
        body = await req.body()
        mt = {}
        h = req.headers.get("x-manifest")
        if h:
            with contextlib.suppress(Exception):
                import base64
                mt = json.loads(base64.b64decode(h).decode())
        engine._pack_results[req_id] = {"unit": unit, "model_id": model_id, "quant": quant,
                                        "bytes": body, "mtensors": mt}
        f = engine._pack_futures.get(req_id)
        if f is not None and not f.done():
            f.set_result(req_id)
        return JSONResponse({"ok": True, "req_id": req_id, "bytes": len(body)})

    @app.post("/pack_probe")    # #distributed-packing Inc 1b: dispatch ONE unit to a worker, byte-check vs local
    async def pack_probe(model: str, node: str = "", layer: int = 0, quant: str = "int4") -> JSONResponse:
        """Offload-pack ONE decoder-layer unit on a worker and prove the result is BIT-IDENTICAL to a
        local compile (the gate before fanning the whole compile out across the fleet). Dense int4/int8."""
        import shards as _sh
        import urllib.parse as _up
        import urllib.request as _ur
        from safetensors.torch import load as _stload, save as _stsave
        if quant not in ("int4", "int8"):
            return JSONResponse({"ok": False, "error": "int4|int8 only"}, status_code=400)
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        mdir = await asyncio.to_thread(_controller_model_dir, tgt)
        if not mdir:
            return JSONResponse({"error": "model not available on the controller"}, status_code=404)
        cand = [n for n in registry.alive_sorted() if n.can_infer and (not node or n.hostname == node)]
        if not cand:
            return JSONResponse({"ok": False, "error": f"no alive worker matching node='{node}'"}, status_code=404)
        nd = cand[0]
        link = engine.links.get(nd.node_id)
        if link is None:
            return JSONResponse({"ok": False, "error": f"no control link to {nd.hostname}"}, status_code=503)
        scope = await asyncio.to_thread(_sh._quant_scope, mdir)   # exact scope (== local compile)
        lin2d = sorted(scope[0]) if scope else None
        exp3d = sorted(scope[1]) if scope else None
        wm = await asyncio.to_thread(_sh._weight_map, mdir)        # per-expert MoE -> worker must fuse (Inc 3b)
        _is_moe = bool(await asyncio.to_thread(_sh._has_moe_experts, wm))
        _moe_fused = any(s.endswith(".gate_up_proj") or s.endswith(".down_proj") for s in wm)
        _need_skel = _is_moe and not _moe_fused
        _skel = scope[2] if (scope and _need_skel) else None
        req_id = f"pk-{int(time.time()*1000)}-{layer}"
        unit = f"L{int(layer):04d}.safetensors"
        fut = asyncio.get_event_loop().create_future()
        engine._pack_futures[req_id] = fut
        frame = {"type": "pack", "req_id": req_id, "model_id": tgt, "quant": quant,
                 "group_size": _sh.INT4_GROUP, "unit": unit, "start": int(layer), "end": int(layer) + 1,
                 "embed": 0, "head": 0, "lin2d": lin2d, "exp3d": exp3d, "fuse": _need_skel,
                 "controller_http_port": ARGS.http_port}
        t0 = time.monotonic()
        try:
            await link.send(frame)
            await asyncio.wait_for(fut, timeout=600)
        except Exception as exc:
            engine._pack_futures.pop(req_id, None)
            return JSONResponse({"ok": False, "error": f"remote pack failed: {exc!r}"}, status_code=504)
        finally:
            engine._pack_futures.pop(req_id, None)
        res = engine._pack_results.pop(req_id, None)
        if not res:
            return JSONResponse({"ok": False, "error": "no pack result received"}, status_code=504)
        remote_ms = round((time.monotonic() - t0) * 1000)
        worker_blob = res["bytes"]

        def _local():   # reference pack of the SAME unit (our own /weights -> identical bytes -> identical pack)
            url = (f"http://127.0.0.1:{ARGS.http_port}/weights?model={_up.quote(tgt)}"
                   f"&start={int(layer)}&end={int(layer)+1}&embed=0&head=0&skip_experts=0")
            with _ur.urlopen(url, timeout=600) as r:
                raw = _stload(r.read())
            out_sd, _mt = _sh.pack_unit_tensors(
                raw, (set(lin2d) if lin2d is not None else None),
                (set(exp3d) if exp3d is not None else None), _skel, quant, _sh.INT4_GROUP)
            return _stsave(out_sd)
        local_blob = await asyncio.to_thread(_local)
        identical = (worker_blob == local_blob)
        tcmp = identical
        if not identical:           # robust fallback: metadata order can differ, compare tensors
            import torch as _t
            wsd, lsd = _stload(worker_blob), _stload(local_blob)
            tcmp = (set(wsd) == set(lsd)) and all(_t.equal(wsd[k], lsd[k]) for k in wsd)
        log_activity(f"pack_probe {_ollama_name(friendly)} {unit} on {nd.hostname}: "
                     f"byte_identical={identical} tensor_identical={tcmp} ({remote_ms} ms)")
        return JSONResponse({"ok": True, "node": nd.hostname, "unit": unit, "remote_ms": remote_ms,
                             "worker_bytes": len(worker_blob), "local_bytes": len(local_blob),
                             "byte_identical": identical, "tensor_identical": tcmp,
                             "tensors": len(res.get("mtensors") or {})})

    @app.post("/compile_dist")   # #distributed-packing Inc 2: compile a shard cache by fanning unit-packs across workers
    async def compile_dist(model: str, quant: str = "int4") -> JSONResponse:
        """Compile a model's pre-quantized shard cache by DISTRIBUTING the per-layer pack across the
        fleet (exo-inspired): each worker fetches a layer's bf16 from /weights, packs it with the
        SHARED shards.pack_unit_tensors (bit-identical to a local compile, proven by /pack_probe), and
        POSTs it back; the controller assembles the cache + manifest. embed/head are packed locally
        (few units, tied-embedding edge cases); any worker failure falls back to a LOCAL pack of that
        layer. Runs in the MAIN process (it owns the control links) — safe because the heavy packing is
        now ON THE WORKERS, not the controller. Dense int4/int8 only (MoE needs the worker skeleton =
        Inc 3); the proven local /compile_shards stays the path for MoE / single-box."""
        import hashlib
        import shards as _sh
        import urllib.parse as _up
        import urllib.request as _ur
        from safetensors.torch import load as _stload, save as _stsave
        if quant not in ("int4", "int8"):
            return JSONResponse({"ok": False, "error": "int4|int8 only"}, status_code=400)
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        mdir = await asyncio.to_thread(_controller_model_dir, tgt)
        if not mdir:
            return JSONResponse({"error": "model not available on the controller"}, status_code=404)
        wm = await asyncio.to_thread(_sh._weight_map, mdir)
        # #distributed-packing Inc 3a/3b: DENSE, FUSED-MoE and PER-EXPERT MoE (Mixtral/OLMoE) are all
        # supported. Fused-MoE needs no fusion (skel=None). Per-expert MoE (checkpoint has experts.N.*,
        # but transformers 5.x builds the model FUSED-3D) needs the worker to fuse per-expert->3D via
        # `_fuse_moe_experts` against a meta skeleton (built from /modelmeta) — we flag `fuse` in the
        # pack frame so the worker builds it, and pass the local skeleton to the local-fallback pack.
        # int8 MoE still has no 3D-expert quantizer -> reject (matches /compile_shards).
        _is_moe = bool(await asyncio.to_thread(_sh._has_moe_experts, wm))
        _moe_fused = any(s.endswith(".gate_up_proj") or s.endswith(".down_proj") for s in wm)
        _need_skel = _is_moe and not _moe_fused          # per-expert checkpoint -> fuse at pack time
        if _is_moe and quant != "int4":
            return JSONResponse({"ok": False, "error": "MoE distributed compile supports int4 only "
                                 "(no int8 3D-expert quantizer) — use int4"}, status_code=400)
        ckey = f"{friendly}::{quant}"
        if ckey in engine.compiling:
            return JSONResponse({"ok": False, "error": f"{_ollama_name(friendly)} {quant} already compiling"},
                                status_code=409)
        n_layers = await asyncio.to_thread(_sh._model_num_layers, mdir)
        out_dir = os.path.join(_sh._shard_cache_root(mdir), quant)
        await asyncio.to_thread(lambda: os.makedirs(out_dir, exist_ok=True))
        caps = [n for n in registry.alive_sorted() if n.can_infer and engine.links.get(n.node_id)]
        engine.compiling[ckey] = {"model": friendly, "display_model": _ollama_name(friendly), "target": tgt,
                                  "ready": 0, "total": n_layers + 2, "stages_total": max(1, len(caps)),
                                  "stages_ready": 0, "basis": f"distributed {quant} compile "
                                  f"({len(caps)} worker(s))", "warnings": [], "started": time.time()}
        log_activity(f"distributed {quant} compile for {_ollama_name(friendly)} -> "
                     f"{n_layers} layers across {len(caps)} worker(s)…")
        scope = await asyncio.to_thread(_sh._quant_scope, mdir)
        lin2d = sorted(scope[0]) if scope else None
        exp3d = sorted(scope[1]) if scope else None
        _lset = set(lin2d) if lin2d is not None else None
        _eset = set(exp3d) if exp3d is not None else None
        # Per-expert MoE (Inc 3b): the local-fallback pack must FUSE per-expert->3D too. The skeleton
        # is scope[2] (the same meta model the worker rebuilds). For dense / already-fused checkpoints
        # _fuse_moe_experts is a no-op, so passing it unconditionally when per-expert is safe.
        _skel = scope[2] if (scope and _need_skel) else None
        with open(os.path.join(mdir, "config.json"), encoding="utf-8") as fh:
            tied = bool(json.load(fh).get("tie_word_embeddings", False))
        base_local = f"http://127.0.0.1:{ARGS.http_port}"

        def _pack_local(start: int, end: int, embed: int, head: int):
            url = (f"{base_local}/weights?model={_up.quote(tgt)}&start={start}&end={end}"
                   f"&embed={int(embed)}&head={int(head)}&skip_experts=0")
            with _ur.urlopen(url, timeout=1800) as r:
                raw = _stload(r.read())
            out_sd, mt = _sh.pack_unit_tensors(raw, _lset, _eset, _skel, quant, _sh.INT4_GROUP)
            return _stsave(out_sd), mt

        _ptag = getattr(_sh, "_packer_tag", None)   # tolerate a lagged shards.py on the controller
        manifest = {"format": 1, "quant": quant, "group_size": _sh.INT4_GROUP, "num_layers": n_layers,
                    "tied": tied, "files": {}, "tensors": {},
                    "packer_hash": (_ptag(quant, _sh.INT4_GROUP) if _ptag else None),  # Inc 4 drift guard
                    "expert_layout": ("fused3d" if _is_moe else None)}   # Inc 3a: fused-MoE serve-from-cache
        _done = {"n": 0}

        def _write(unit: str, blob: bytes, mt: dict) -> None:   # inline (no thread) -> manifest dict race-free
            with open(os.path.join(out_dir, unit), "wb") as f:
                f.write(blob)
            manifest["files"][unit] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
            for name, meta in mt.items():
                manifest["tensors"][name] = {"file": unit, **meta}
            _done["n"] += 1
            c = engine.compiling.get(ckey)
            if c:
                c["ready"] = _done["n"]

        async def _dispatch_layer(node, i: int):
            link = engine.links.get(node.node_id)
            if link is None:
                raise RuntimeError(f"no link to {node.hostname}")
            req_id = f"cd-{int(time.time()*1000)}-{i}-{node.node_id}"
            fut = asyncio.get_event_loop().create_future()
            engine._pack_futures[req_id] = fut
            frame = {"type": "pack", "req_id": req_id, "model_id": tgt, "quant": quant,
                     "group_size": _sh.INT4_GROUP, "unit": f"L{i:04d}.safetensors",
                     "start": i, "end": i + 1, "embed": 0, "head": 0,
                     "lin2d": lin2d, "exp3d": exp3d, "fuse": _need_skel,   # Inc 3b: worker fuses per-expert->3D
                     "controller_http_port": ARGS.http_port}
            try:
                await link.send(frame)
                await asyncio.wait_for(fut, timeout=1800)
            finally:
                engine._pack_futures.pop(req_id, None)
            res = engine._pack_results.pop(req_id, None)
            if not res:
                raise RuntimeError("no pack result")
            return res["bytes"], res["mtensors"]

        _part: set = set()   # node_ids that have packed >=1 unit via the worker path (live node count)

        async def _run():
            try:
                eb, emt = await asyncio.to_thread(_pack_local, 0, 0, 1, 0)
                _write("embed.safetensors", eb, emt)
                q: asyncio.Queue = asyncio.Queue()
                for i in range(n_layers):
                    q.put_nowait(i)

                async def _node_loop(node):
                    while True:
                        try:
                            i = q.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        unit = f"L{i:04d}.safetensors"
                        try:
                            blob, mt = await _dispatch_layer(node, i)
                            if node.node_id not in _part:   # first unit from this worker -> live "N nodes" + log
                                _part.add(node.node_id)
                                c = engine.compiling.get(ckey)
                                if c:
                                    c["stages_ready"] = len(_part)
                                log_activity(f"compile_dist: {node.hostname} packing "
                                             f"{_ollama_name(friendly)} {quant} layers")
                        except Exception as exc:   # worker died / no shards.py / timeout -> local fallback
                            log_activity(f"compile_dist {unit} on {node.hostname} failed ({exc!r}) -> local pack")
                            blob, mt = await asyncio.to_thread(_pack_local, i, i + 1, 0, 0)
                        _write(unit, blob, mt)

                if caps:
                    await asyncio.gather(*[_node_loop(n) for n in caps])
                else:                              # no workers -> compile fully locally
                    for i in range(n_layers):
                        blob, mt = await asyncio.to_thread(_pack_local, i, i + 1, 0, 0)
                        _write(f"L{i:04d}.safetensors", blob, mt)
                hb, hmt = await asyncio.to_thread(_pack_local, 0, 0, 0, 1)
                _write("head.safetensors", hb, hmt)
                with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                _CACHE_VERIFY_MEMO.pop((mdir, quant), None)   # force a fresh verify on next load
                log_activity(f"distributed {quant} compile DONE for {_ollama_name(friendly)} "
                             f"({n_layers} layers, {len(caps)} worker(s))")
            except Exception as exc:
                log_activity(f"distributed compile FAILED for {_ollama_name(friendly)}: {exc!r}")
            finally:
                engine.compiling.pop(ckey, None)

        asyncio.create_task(_run())
        return JSONResponse({"ok": True, "model": _ollama_name(friendly), "quant": quant,
                             "distributed": True, "workers": len(caps), "layers": n_layers})

    @app.post("/compile_shards")        # #shard-cache: compile a model's pre-quantized cache on beast
    async def compile_shards_ep(model: str, quant: str = "int4") -> JSONResponse:
        if quant not in ("int4", "int8"):
            return JSONResponse({"ok": False, "error": f"shard cache supports int4|int8 (got '{quant}')"},
                                status_code=400)
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        mdir = await asyncio.to_thread(_controller_model_dir, tgt)
        if not mdir:
            return JSONResponse({"error": "model not available on the controller"}, status_code=404)
        # Compiles run CONCURRENTLY — with loads and with each other. CRITICAL: each runs in a
        # SUBPROCESS (its own GIL), NOT an in-process thread. The quantize work is Python-heavy (the
        # per-tensor loop + sha256), and asyncio.to_thread keeps it in the controller process where it
        # holds the GIL and starves the single event-loop thread — even on a 124-core box — stalling the
        # data plane enough to drop live generations' logits connection ("data connection closed" bursts).
        # The subprocess can't touch the controller's GIL/event loop; we read its progress over a pipe.
        # Only refuse an EXACT duplicate (same model+quant) since two writers to _shards/<quant>/ corrupt
        # it. Each compile gets its own card in engine.compiling (keyed model::quant) for the dashboard.
        ckey = f"{friendly}::{quant}"
        if ckey in engine.compiling:
            return JSONResponse({"ok": False, "error": f"{_ollama_name(friendly)} {quant} is already "
                                 "compiling"}, status_code=409)
        engine.compiling[ckey] = {"model": friendly, "display_model": _ollama_name(friendly),
                                  "target": tgt, "ready": 0, "total": 1, "stages_total": 1,
                                  "stages_ready": 0, "basis": f"compiling {quant} shard cache (subprocess)",
                                  "warnings": [], "started": time.time()}
        log_activity(f"compiling {quant} shard cache for {_ollama_name(friendly)} (subprocess)…")
        srv_dir = os.path.dirname(os.path.abspath(__file__))
        _script = (
            "import sys, json\n"
            "sys.path.insert(0, sys.argv[3])\n"
            "import shards\n"
            "def p(d, t):\n"
            "    sys.stdout.write('P %d %d\\n' % (d, t)); sys.stdout.flush()\n"
            "m = shards.compile_shards(sys.argv[1], sys.argv[2], progress=p)\n"
            "files = m.get('files', {})\n"
            "sys.stdout.write('DONE ' + json.dumps({'files': len(files),\n"
            "    'bytes': sum(int(v.get('bytes', 0)) for v in files.values()),\n"
            "    'num_layers': m.get('num_layers')}) + '\\n'); sys.stdout.flush()\n")
        # below-normal priority on Windows (beast) so serving is scheduled first; best-effort elsewhere.
        _kw: dict = {}
        if sys.platform == "win32" and hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
            _kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        elif sys.platform != "win32":
            _kw["preexec_fn"] = lambda: os.nice(10)   # Unix: deprioritize
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _script, mdir, quant, srv_dir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **_kw)
        except NotImplementedError:
            proc = None   # event loop (Selector on some Windows setups) can't spawn -> in-process fallback
        except Exception as exc:
            log_activity(f"compile subprocess spawn failed ({exc}); falling back to in-process")
            proc = None
        result: Optional[dict] = None
        err_msg: Optional[str] = None
        try:
            if proc is not None:
                async for raw in proc.stdout:                   # read progress WITHOUT blocking the loop
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("P "):
                        try:
                            _, d_, t_ = line.split()
                            card = engine.compiling.get(ckey)
                            if card is not None:
                                card["ready"], card["total"] = int(d_), int(t_)
                        except ValueError:
                            pass
                    elif line.startswith("DONE "):
                        with contextlib.suppress(Exception):
                            result = json.loads(line[5:])
                await proc.wait()
                if proc.returncode != 0 or result is None:
                    tail = (await proc.stderr.read()).decode("utf-8", "replace").strip().splitlines()
                    err_msg = tail[-1] if tail else f"compile subprocess exit {proc.returncode}"
            else:
                # FALLBACK (no subprocess support): in-process compile. May briefly affect serving on a
                # busy box (the GIL issue this whole change avoids) — logged so the cause is visible.
                log_activity("compile running IN-PROCESS (no subprocess support) — may affect serving")
                def _prog(done: int, total: int) -> None:
                    card = engine.compiling.get(ckey)
                    if card is not None:
                        card["ready"], card["total"] = done, total
                try:
                    man = await asyncio.to_thread(lambda: compile_shards(mdir, quant, progress=_prog))
                    files = man.get("files", {})
                    result = {"files": len(files),
                              "bytes": sum(int(v.get("bytes", 0)) for v in files.values()),
                              "num_layers": man.get("num_layers")}
                except Exception as exc:
                    err_msg = str(exc)
        finally:
            engine.compiling.pop(ckey, None)   # single-owner cleanup of the compile card
        if err_msg is not None or result is None:
            msg = err_msg or "compile failed"
            log_activity(f"shard compile FAILED for {_ollama_name(friendly)}: {msg}")
            return JSONResponse({"ok": False, "error": msg}, status_code=400)
        total_gb = int(result.get("bytes", 0)) / GB
        log_activity(f"shard cache compiled for {_ollama_name(friendly)} "
                     f"({quant}, {result.get('files')} files, {total_gb:.1f} GB)")
        return JSONResponse({"ok": True, "quant": quant, "files": result.get("files"),
                             "size_gb": round(total_gb, 2),
                             "num_layers": result.get("num_layers")})

    @app.post("/load")
    async def load(model: str, ctx: int = 0, mode: str = "auto",
                   consolidate: bool = True, quant: str = "none", tp: int = 1,
                   replicas: int = 1, cpu_only: bool = False,
                   moe_offload: bool = False, force: bool = False) -> JSONResponse:
        # force=1 (#stuck-load-override): if a load of this model is already IN FLIGHT, CANCEL it and
        # restart fresh (the manual escape hatch for a wedged 0%-forever load) instead of queueing on
        # it. Also reloads an already-resident copy (skips the idempotent no-op). Without force, a
        # concurrent same-model request still queues on the in-flight load as before.
        # ctx=0 (default) => the model's native training context (config.json).
        # `mode` chooses HOW the model is placed (maps to consolidate, prefer_vram):
        #   auto       (T, T) GPU-VRAM-first, fewest nodes — best decode latency [default]
        #   single     (T, F) fewest nodes by total RAM+VRAM — collapses to one box if it fits
        #   gpu-spread (F, T) fill every GPU's VRAM, spill across nodes
        #   distribute (F, F) spread across the WHOLE fleet (CPUs + GPUs)
        #   spread     (F, F) FORCE a stage on every capable node (incl. tiny ones)
        #   proportional (F, F) layers across EVERY capable node PROPORTIONAL to its capacity
        #                 (#78: big int4 MoE — MiniMax-M2 — too big for the GPU-first subset)
        # `quant`: 'none' (bf16), 'int8' (~1/2), or 'int4' (group-wise ~4.25-bit, ~1/4 — for
        # 200B+ MoEs that won't fit at int8). `tp` (M4): tensor-parallel group
        # size — split every layer across `tp` GPU nodes (rank 0 drives the group over the
        # all-reduce mesh). tp>1 overrides mode. tp must divide num_key_value_heads.
        # Legacy: if mode is omitted but consolidate=false is passed, honor it.
        if quant not in ("none", "int8", "int4"):
            return JSONResponse({"ok": False, "error": f"bad quant '{quant}' (none|int8|int4)"},
                                status_code=400)
        cons, pv = LOAD_MODES.get(mode, LOAD_MODES["auto"])
        if mode == "auto" and not consolidate:   # back-compat with the old checkbox
            cons, pv = False, True
        try:
            friendly = resolve_model_name(model)
            # replicas>1 (#39): load N full copies on disjoint nodes for data-parallel
            # throughput. Mutually exclusive with tp (tp splits one copy; replicas duplicate it).
            if tp <= 1 and replicas > 1:
                lms = await engine.replicate(friendly, ctx, replicas,
                                             consolidate=cons, prefer_vram=pv, quant=quant)
                return JSONResponse({"ok": True, "model": friendly, "ctx": lms[0].ctx,
                                     "mode": mode, "quant": quant, "replicas": len(lms),
                                     "placements": [{"key": m.friendly,
                                                     "hosts": [s.hostname for s in m.plan.stages]}
                                                    for m in lms]})
            lm = await engine.load(friendly, ctx, consolidate=cons, prefer_vram=pv,
                                   quant=quant, tp=tp, cpu_only=cpu_only,
                                   spread=(mode == "spread"),
                                   proportional=(mode == "proportional"),
                                   moe_offload=moe_offload, force=force)
            return JSONResponse({"ok": True, "model": lm.friendly, "ctx": lm.ctx,
                                 "mode": (("tp%d-cpu" % tp) if cpu_only else ("tp%d" % tp))
                                         if tp > 1 else mode, "quant": quant,
                                 "warnings": getattr(lm, "load_warnings", []),   # #76 guardrail
                                 "stages": [s.to_dict() for s in lm.plan.stages]})
        except Exception as exc:
            # (engine.load()'s finally already popped this load's progress card; nothing to clear here.)
            engine._last_load_failure = time.time()   # arm the self-update cool-down (anti-churn)
            # A failed load leaves no resident model; surface WHY on the dashboard so the
            # operator isn't left wondering why the model never appeared (or why an in-flight
            # big-MoE load died — e.g. a node OOM mid-load).
            log_activity(f"load {model}: FAILED — {exc}")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.post("/cancel")           # dashboard: disconnect/kill one in-flight request (#48)
    async def cancel(id: int) -> JSONResponse:
        rec = INFLIGHT.get(id)
        if rec is None:
            return JSONResponse({"ok": False, "error": f"no in-flight request id={id}"},
                                status_code=404)
        rec["cancel"] = True
        t = rec.get("task")
        if t is not None and not t.done():
            with contextlib.suppress(Exception):
                t.cancel()        # aborts _prepare/load/generate for this request (frees a wedge)
        _inflight_release(rec)
        log_activity(f"cancelled request id={id} ({rec.get('model')}, {rec.get('ip')})")
        return JSONResponse({"ok": True, "cancelled": id,
                             "model": rec.get("model"), "ip": rec.get("ip")})

    @app.post("/cancel_load")      # #stuck-load-override: kill a wedged in-flight MODEL LOAD (0%-forever)
    async def cancel_load(model: str = "") -> JSONResponse:
        """Cancel an in-flight (possibly wedged) model LOAD — the manual escape hatch for a load stuck
        at 0%. model='' cancels EVERY in-flight load. Cancelling the load task frees any partial shards
        it already built (the load's CancelledError cleanup), emptying it out so a fresh load can run."""
        friendly = ""
        if model:
            try:
                friendly = resolve_model_name(model)
            except ValueError:
                friendly = model
        cancelled = []
        for rk, t in list(engine._loading_tasks.items()):
            base = rk.split("#", 1)[0]
            if friendly and rk != friendly and base != friendly:
                continue
            if t is not None and not t.done():
                with contextlib.suppress(Exception):
                    t.cancel()
                cancelled.append(rk)
        if not cancelled:
            return JSONResponse({"ok": False, "error": "no in-flight load"
                                 + (f" for '{model}'" if model else "")}, status_code=404)
        log_activity(f"cancelled in-flight load(s): {', '.join(cancelled)}")
        return JSONResponse({"ok": True, "cancelled": cancelled})

    @app.post("/unload")
    async def unload(model: str = "") -> JSONResponse:
        # No model -> unload everything; model=X -> evict just that one (keep the rest).
        if model:
            try:
                friendly = resolve_model_name(model)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            await engine.unload(friendly)   # per-model: allowed any time, even during another's load
            return JSONResponse({"ok": True, "unloaded": [friendly]})
        # Blanket "unload everything" drops ALL shards on every worker — incl. any in-flight load's
        # half-built ones. engine.unload(None) decides UNDER self.lock (atomic with a load's card
        # registration) and raises LoadInProgressError if a load is in flight — no HTTP-layer TOCTOU.
        names = list(engine.models.keys())   # snapshot before the full teardown
        try:
            await engine.unload()
        except LoadInProgressError as exc:
            return JSONResponse({"ok": False, "error": "a load is in progress — wait for it, or unload a "
                                 "specific model (model=NAME); unload-all is blocked mid-load",
                                 "loading": list(exc.args[0]) if exc.args else []}, status_code=409)
        return JSONResponse({"ok": True, "unloaded": names})

    @app.post("/reconfigure")
    async def reconfigure(model: str, tp: int = 1, ctx: int = 0, quant: str = "keep",
                          mode: str = "auto", cpu_only: bool = False) -> JSONResponse:
        # #88 managed reload: switch a RESIDENT model to/from tensor-parallel (or change TP width /
        # ctx / quant) in ONE call, rolling back to a working pipeline copy on failure. ctx=0 or
        # quant='keep' INHERIT the resident copy's values (a pure layout switch keeps them). tp>=2 ->
        # tensor-parallel (cpu_only routes the mesh to RAM); tp<=1 -> pipeline (mode picks the strategy).
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        lm = engine.models.get(friendly)
        if lm is None:
            return JSONResponse({"ok": False, "error": f"'{friendly}' is not resident — load it first"},
                                status_code=404)
        if getattr(lm, "active", 0) > 0:   # never tear a model down mid-generate
            return JSONResponse({"ok": False, "error": f"'{friendly}' is busy ({lm.active} active "
                                 f"request(s)) — retry when idle"}, status_code=409)
        if quant not in ("keep", "none", "int8", "int4"):
            return JSONResponse({"ok": False, "error": f"bad quant '{quant}' (keep|none|int8|int4)"},
                                status_code=400)
        new_ctx = ctx if (ctx and ctx > 0) else lm.ctx
        new_quant = (lm.quant or "none") if quant == "keep" else quant
        from_tp = getattr(lm, "tp_size", 1)
        from_ctx, from_quant = lm.ctx, lm.quant
        if tp == from_tp and new_ctx == lm.ctx and new_quant == (lm.quant or "none"):
            return JSONResponse({"ok": True, "model": friendly, "noop": True,
                                 "from": {"tp": from_tp, "ctx": from_ctx, "quant": from_quant},
                                 "to": {"tp": from_tp, "ctx": new_ctx, "quant": new_quant},
                                 "basis": getattr(lm, "plan_basis", ""),
                                 "stages": [s.hostname for s in lm.plan.stages]})
        # Pre-validate the TP width (same guards as _load_tp_locked) BEFORE evicting, so an obviously
        # invalid width fails clean (400) instead of an evict-then-rollback churn.
        if tp > 1:
            spec = resolve_spec(friendly)
            nh, nkv = spec.num_heads, spec.num_kv_heads
            ng = max(1, spec.intermediate_size // 128)
            ok_geom = (nh % tp == 0) and ((tp <= nkv and nkv % tp == 0) or (tp > nkv and tp % nkv == 0)) and tp <= ng
            if not ok_geom:
                return JSONResponse({"ok": False, "error":
                    f"tp={tp} invalid for {friendly}: needs num_heads({nh})%tp==0, "
                    f"(nkv({nkv})%tp==0 if tp<=nkv else tp%nkv==0), and tp<=FFN_groups({ng})"},
                    status_code=400)
        cons, pv = LOAD_MODES.get(mode, LOAD_MODES["auto"])
        try:
            new = await engine.reconfigure(friendly, tp=tp, ctx=new_ctx, quant=new_quant,
                                           consolidate=cons, prefer_vram=pv, cpu_only=cpu_only)
            return JSONResponse({"ok": True, "model": new.friendly,
                                 "from": {"tp": from_tp, "ctx": from_ctx, "quant": from_quant},
                                 "to": {"tp": getattr(new, "tp_size", 1), "ctx": new.ctx,
                                        "quant": new.quant,
                                        "mode": (("tp%d-cpu" % tp) if cpu_only else ("tp%d" % tp))
                                                if tp > 1 else mode},
                                 "basis": getattr(new, "plan_basis", ""),
                                 "stages": [s.hostname for s in new.plan.stages]})
        except Exception as exc:
            # (the internal engine.load()'s finally already cleared any progress card.)
            log_activity(f"reconfigure {model}: FAILED — {exc}")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.post("/restart")
    async def restart(request: Request, workers: int = 1, force: bool = False) -> JSONResponse:
        # FULL-FLEET RESTART: signal every connected worker to restart, then restart the controller.
        # UNLIKE the idle-gated self-update, this is an EXPLICIT command and is NOT idle-gated.
        # Supervisors relaunch on exit 42 (server.bat / client.bat / systemd Restart=always).
        # workers=0 -> restart the controller only. GENTLER (#100): refuse while a load is IN PROGRESS
        # — restarting mid-build drops a node from the load (and can leave it not cleanly rejoining);
        # pass force=1 to abort a wedged/doomed load anyway (the original escape-hatch behavior).
        # Blocks on in-flight LOADS or COMPILES (both lose work on restart); force=1 overrides.
        if (engine.loadings or engine.compiling) and not force:
            return JSONResponse({"ok": False, "status": "load_in_progress",
                                 "reason": "a model load/compile is in progress; pass force=1 to restart "
                                           "anyway (aborts it)",
                                 "loading": [c.get("model") for c in engine.loadings.values()],
                                 "compiling": [c.get("model") for c in engine.compiling.values()]},
                                status_code=409)
        signaled = []
        if workers:
            for nid, link in list(engine.links.items()):
                with contextlib.suppress(Exception):
                    await link.send({"type": "restart"})
                    signaled.append(nid)
        who = _client_ip(request)   # who triggered it (dashboard browser / curl host) for the log
        msg = (f"FLEET RESTART requested by {who} -> {len(signaled)} worker(s) + controller "
               f"(exit 42){' [controller only]' if not workers else ''}")
        log_activity(msg)
        print(f"[restart] {msg}; controller exiting(42) in 2s")
        async def _bye():
            await asyncio.sleep(2.0)   # let worker frames flush + this HTTP response return
            os._exit(42)               # server.bat supervisor relaunches on the current code
        asyncio.create_task(_bye())
        return JSONResponse({"ok": True, "restarting_controller": True, "requested_by": who,
                             "workers_signaled": signaled, "worker_count": len(signaled)})

    @app.post("/update")
    async def update_endpoint(request: Request, workers: int = 0) -> JSONResponse:
        # FORCED UPDATE (dashboard 'Update' button / deploy API): pull the latest code from GitLab
        # and restart NOW — do NOT wait for idle. Mitigates the auto-load race: set engine.updating
        # so no request reloads a model mid-swap, UNLOAD all models, tell every worker to FREE its
        # RAM (and restart too if workers=1), then swap changed files + exit(42) -> supervisor
        # relaunches on the new code. (Plain /restart relaunches the CURRENT code; this updates first.)
        who = _client_ip(request)
        engine.updating = True               # block auto-load immediately (anti-reload-race)
        names = list(engine.models.keys())
        with contextlib.suppress(Exception):     # best-effort graceful unload (don't block on a
            # force=True: this is a deploy/restart — tear down even if a load is in flight (the process
            # is about to exit anyway), so the blanket teardown isn't refused by the in-load guard.
            await asyncio.wait_for(engine.unload(force=True), timeout=10)   # wedged in-flight load — exit anyway)
        freed = []
        for nid, link in list(engine.links.items()):
            with contextlib.suppress(Exception):
                await link.send({"type": "free_memory"})      # drop shards + gc + drop OS caches
                if workers:
                    await link.send({"type": "restart"})      # full worker relaunch too
                freed.append(nid)
        msg = (f"FORCED UPDATE by {who}: unloaded {names or 'none'}, freed RAM on {len(freed)} "
               f"worker(s){' + worker restart' if workers else ''} -> swap code + restart")
        log_activity(msg); print(f"[update] {msg}")
        async def _go():
            await asyncio.sleep(1.5)   # let unload/free acks + this HTTP response flush
            with contextlib.suppress(Exception):   # force-swap; _self_update_check exits if changed
                await asyncio.to_thread(_self_update_check, "server.py", (lambda: True), True)
            os._exit(42)               # nothing to swap (or already swapped) -> plain relaunch
        asyncio.create_task(_go())
        return JSONResponse({"ok": True, "updating": True, "unloaded": names,
                             "workers_freed": len(freed), "worker_restart": bool(workers),
                             "requested_by": who})

    # ---- chunk serving (workers fetch only their slice; nothing on worker disk) ----
    @app.get("/modelmeta")
    async def modelmeta(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
            return JSONResponse(json.load(fh))

    @app.get("/mtp_probe")
    async def mtp_probe(model: str = "qwen3.6-35b-a3b", mode: str = "dump",
                        prompt: str = "", fresh: int = 0) -> JSONResponse:
        # #91 Increment 1a (discovery): the checkpoint ships an MTP (nextn) head but the installed
        # transformers DROPS it (_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]) — no class to build
        # or run it. To reimplement the MTP forward for self-speculative decoding we first need the
        # EXACT module structure: which mtp.* tensors exist, their shapes/dtypes, and the embed /
        # lm_head / final-norm key names the MTP head shares. Reads the safetensors index only (no
        # model load). Returns a top-level prefix histogram + every mtp.* tensor + the shared-head
        # tensors so the hand-built module matches the checkpoint exactly.
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)

        def _dump() -> dict:
            from safetensors import safe_open
            idx = os.path.join(d, "model.safetensors.index.json")
            if os.path.exists(idx):
                with open(idx, encoding="utf-8") as fh:
                    wm = json.load(fh)["weight_map"]      # tensor_name -> shard filename
            else:                                          # single-file checkpoint
                wm = {}
                single = os.path.join(d, "model.safetensors")
                if os.path.exists(single):
                    with safe_open(single, framework="pt") as sf:
                        wm = {k: "model.safetensors" for k in sf.keys()}
            keys = sorted(wm)
            # prefix histogram (first two dotted segments) so the nesting is visible at a glance
            hist: dict = {}
            for k in keys:
                parts = k.split(".")
                pref = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
                hist[pref] = hist.get(pref, 0) + 1
            # resolve shape/dtype for a set of keys, opening each shard once
            def _meta(want: list) -> dict:
                want = [k for k in want if k in wm]
                by_file: dict = {}
                for k in want:
                    by_file.setdefault(wm[k], []).append(k)
                out: dict = {}
                for fn, ks in by_file.items():
                    with safe_open(os.path.join(d, fn), framework="pt") as sf:
                        for k in ks:
                            sl = sf.get_slice(k)
                            try:
                                dt = sl.get_dtype()
                            except Exception:
                                dt = "?"
                            out[k] = {"shape": list(sl.get_shape()), "dtype": str(dt), "file": fn}
                return out
            mtp_keys = [k for k in keys if k == "mtp" or k.startswith("mtp.")]
            # the shared head + embedding the MTP module reuses (names vary by multimodal nesting)
            shared = [k for k in keys if any(s in k for s in (
                "embed_tokens", "lm_head", "language_model.norm", ".model.norm.", )) or k.endswith("model.norm.weight")]
            return {
                "model": friendly, "target": target, "n_tensors": len(keys),
                "prefix_histogram": dict(sorted(hist.items())),
                "mtp": _meta(mtp_keys),
                "shared_head_candidates": _meta(shared),
            }

        async def _dprobe() -> dict:
            # #91 Increment 2 (distributed-hidden probe): with the model loaded DISTRIBUTED, run a
            # prefill that returns the pre-final-norm trunk hidden (capture_pre_norm), then run the
            # small controller-resident MTP head over the sequence and measure how often its drafted
            # token matches the pipeline's own greedy continuation. NEVER loads the full model here
            # (see never-full-load-on-controller-box) — only the ~few-GB MTP head.
            import importlib
            here = os.path.dirname(os.path.abspath(__file__))
            with contextlib.suppress(Exception):    # iterate the MTP forward w/o a controller restart
                remote = _fetch_repo_file("mtp_core.py")
                if remote and len(remote) > 80:
                    with open(os.path.join(here, "mtp_core.py"), "wb") as fh:
                        fh.write(remote)
            import mtp_core as _mc
            importlib.reload(_mc)
            m = engine.models.get(friendly) or engine._pick_replica(friendly)
            if m is None or getattr(m, "stage0_writer", None) is None:
                return {"error": f"{friendly} is not loaded distributed — load it first"}
            if not hasattr(engine, "_mtp_heads"):
                engine._mtp_heads = {}
            head = engine._mtp_heads.get(friendly)
            if head is None or fresh:
                head = await asyncio.to_thread(_mc.load_mtp_head, d, "cpu")
                engine._mtp_heads[friendly] = head
            import torch
            p = prompt or ("The capital of France is Paris. The capital of Japan is Tokyo. "
                           "The capital of Italy is Rome. The capital of Canada is Ottawa. "
                           "The capital of Germany is")
            ids = m.tokenizer(p, return_tensors="pt").input_ids
            S = int(ids.shape[1])
            if S < 4:
                return {"error": "prompt too short"}
            # prefill on the distributed pipeline; capture per-position logits + pre-norm hidden.
            async with m.lock:
                logits, h_pre = await engine._send(m, ids, 0, True, all_logits=True,
                                                   capture_pre_norm=True)
                await engine._crop(m, 0)   # reset the probe's KV so it can't pollute a later gen

            def _compute() -> dict:
                th = h_pre[:, 0:S - 1]
                nxt = ids[:, 1:S]
                main_greedy = logits[0].float().argmax(-1)   # main_greedy[j] predicts token j+1
                actual = ids[0]
                out = {}
                for off in (0, 1):
                    ml = _mc.mtp_forward_seq(head, th, nxt, position_offset=off)
                    mtp_pred = ml[0].float().argmax(-1)       # mtp_pred[i] predicts token i+2
                    n = S - 2
                    ag = aa = 0
                    ex = []
                    for i in range(n):
                        mp = int(mtp_pred[i]); tg = int(main_greedy[i + 1]); ta = int(actual[i + 2])
                        ag += (mp == tg); aa += (mp == ta)
                        if len(ex) < 8:
                            ex.append({"i": i, "mtp": mp, "greedy": tg, "actual": ta,
                                       "mtp_tok": m.tokenizer.decode([mp]),
                                       "greedy_tok": m.tokenizer.decode([tg])})
                    out[f"off{off}"] = {"acc_vs_greedy": round(ag / max(1, n), 3),
                                        "acc_vs_actual": round(aa / max(1, n), 3), "n": n,
                                        "examples": ex}
                # DIAGNOSTIC: incremental (decode-path) drafts vs the proven parallel forward_seq.
                # If these disagree, the KV/attention path mtp_step uses at decode time is broken.
                inc = _mc.mtp_incremental_drafts(head, th, nxt)         # [S-1] argmax tokens
                par = _mc.mtp_forward_seq(head, th, nxt, position_offset=0)[0].float().argmax(-1)
                n = S - 2
                same_par = sum(1 for i in range(S - 1) if inc[i] == int(par[i]))
                inc_vs_greedy = sum(1 for i in range(n) if inc[i] == int(main_greedy[i + 1]))
                out["incremental"] = {
                    "matches_parallel": round(same_par / max(1, S - 1), 3),
                    "acc_vs_greedy": round(inc_vs_greedy / max(1, n), 3), "n": n}
                return out

            out = await asyncio.to_thread(_compute)
            best = max((k for k in out if k.startswith("off")),
                       key=lambda k: out[k]["acc_vs_greedy"])
            return {"ok": True, "model": friendly, "S": S, "best": best,
                    "summary": {k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                for k, v in out.items()},
                    "examples": out[best]["examples"],
                    "load_missing": head.load_missing[:10],
                    "load_unexpected": head.load_unexpected[:10]}

        try:
            if mode == "run":
                # DISABLED (m4c122): the original run-mode loaded the FULL model on this co-hosted
                # controller box and OOM-crashed the controller. Use mode=dprobe (distributed hidden +
                # small MTP head) instead. See never-full-load-on-controller-box.
                return JSONResponse({"error": "mode=run disabled (crashed the co-hosted controller). "
                                     "Use mode=dprobe — distributed hidden + small MTP head."},
                                    status_code=400)
            if mode == "dprobe":
                return JSONResponse(await _dprobe())
            return JSONResponse(await asyncio.to_thread(_dump))
        except Exception as exc:
            import traceback
            return JSONResponse({"error": repr(exc), "tb": traceback.format_exc()[-1500:]},
                                status_code=500)

    @app.get("/modelcode")
    async def modelcode(model: str) -> JSONResponse:
        # Serve the model's trust_remote_code python files (the auto_map modeling/configuration
        # *.py) so a WORKER — which builds the skeleton from config alone — can construct the
        # CORRECT architecture instead of falling back to transformers' native class. Without this,
        # a remote-code model (e.g. MiniMax-M2, whose checkpoint is full-attention but whose
        # model_type 'minimax' maps natively to the OLDER lightning Text-01 arch) builds the wrong
        # modules and every mismatched tensor stays on 'meta'. Returns {filename: source}; empty
        # {} for a model with no auto_map (the worker then keeps the native-class path).
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        def _collect() -> dict:
            try:
                with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
                    if not (json.load(fh) or {}).get("auto_map"):
                        return {}                      # not a remote-code model -> nothing to ship
            except Exception:
                return {}
            def _read_py() -> dict:
                o = {}
                for fn in os.listdir(d):               # all .py in the snapshot (modeling + its imports)
                    if fn.endswith(".py"):
                        with contextlib.suppress(Exception):
                            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                                o[fn] = fh.read()
                return o
            out = _read_py()
            if not out:
                # auto_map is set but the dir has NO .py — a model pulled before *.py was added to the
                # download patterns (e.g. MiniMax-M2). Fetch the repo's .py from HF hub into the dir
                # ON-DEMAND (small), then re-read — fixes already-downloaded models without re-pulling
                # the weights. Best-effort: any failure leaves out={} (worker keeps the native path). #78
                with contextlib.suppress(Exception):
                    from huggingface_hub import HfApi, hf_hub_download
                    tok = HF_TOKEN or None
                    for f in HfApi().list_repo_files(target, token=tok):
                        if f.endswith(".py"):
                            with contextlib.suppress(Exception):
                                hf_hub_download(target, f, token=tok, local_dir=d)
                    out = _read_py()
                    if out:
                        print(f"[modelcode] fetched {len(out)} trust_remote_code .py for {target} "
                              f"(dir was missing them)", flush=True)
            return out
        return JSONResponse(await asyncio.to_thread(_collect))

    @app.get("/weights")
    async def weights(model: str, start: int, end: int, embed: int = 0, head: int = 0,
                      skip_experts: int = 0, cache: str = ""):
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        # #shard-cache Inc 2 (serve-from-cache): when the controller flagged this load cache=int4, the
        # worker requests pre-packed units. Each unit (embed / one layer / head) is its own cache file,
        # already EXACTLY this stage's int4-packed tensors (+ bf16 norms/biases) — stream it whole, no
        # plan/dequant/quant. The controller decides verify+enablement; here we just serve the file if
        # present. Missing file -> fall through to the bf16 stream (per-unit safe fallback).
        if cache and d:
            cunit = await asyncio.to_thread(
                cache_unit_path, d, cache, start, end, bool(embed), bool(head))
            if cunit and (bool(embed) or bool(head) or end - start == 1):
                ctotal = os.path.getsize(cunit)

                def _owns_c(n) -> bool:
                    ls, le = n.layer_start, n.layer_end
                    if ls is None or le is None:
                        return False
                    if end > start:
                        return ls <= start and end <= le
                    return ls == start
                cnid = next((n.node_id for n in registry._nodes.values() if _owns_c(n)), None)
                chost = registry._nodes[cnid].hostname if cnid in registry._nodes else "?"
                log_activity(f"serving {friendly} CACHED {cache} L{start}-{end} -> {chost} "
                             f"({ctotal / GB:.2f} GB)")

                def _cgen():
                    with open(cunit, "rb") as f:
                        while True:
                            chunk = f.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            net_account(cnid, to_node=len(chunk))
                            yield chunk
                    ld = next((c for c in engine.loadings.values()
                               if c.get("target") == target), None)
                    if ld is not None:
                        ld["ready"] = ld.get("ready", 0) + max(1, (end - start)
                                                               + (1 if embed else 0)
                                                               + (1 if head else 0))

                return StreamingResponse(
                    _cgen(), media_type="application/octet-stream",
                    headers={"Content-Length": str(ctotal),
                             "Content-Disposition":
                                 f'attachment; filename="{friendly}-cache-{start}_{end}.safetensors"'})
        # Stream the stage's tensors straight from the source files (raw bytes, 8 MB chunks):
        # bounded memory, no temp blob, no lock -> every worker pulls its full slice
        # concurrently in one smooth pass. skip_experts (#62) omits the fused 3D MoE experts so
        # the worker can stream them per-expert via /experts (no ~7 GB layer blob in RAM).
        header_bytes, parts, total = await asyncio.to_thread(
            _plan_weight_stream, d, start, end, bool(embed), bool(head), bool(skip_experts))
        # meter against the node whose layer range this serves (controller -> node), per
        # chunk so the dashboard rate tracks the real transfer instead of one upfront spike
        # Attribute this slice to its node. Per-layer streaming (m4ak) requests single layers
        # (start=i,end=i+1) and embed/head as start==end slices, so match by CONTAINING range
        # rather than exact endpoints (the old exact match missed every streamed fetch -> nid
        # None -> traffic unmetered + host '?'). A full-range fetch (TP/from_file) still matches.
        def _owns(n) -> bool:
            ls, le = n.layer_start, n.layer_end
            if ls is None or le is None:
                return False
            if end > start:                       # a layer slice -> node whose range contains it
                return ls <= start and end <= le
            return ls == start                    # embed/head slice (start==end) -> node starting here
        nid = next((n.node_id for n in registry._nodes.values() if _owns(n)), None)
        _host = registry._nodes[nid].hostname if nid in registry._nodes else "?"
        log_activity(f"serving {friendly} weights L{start}-{end} -> {_host} ({total / GB:.2f} GB)")

        def _gen():
            net_account(nid, to_node=len(header_bytes))
            yield header_bytes
            for p in parts:
                if p.get("kind") in ("fp8", "nvfp4"):   # quantized checkpoint: dequant -> bf16, stream bf16
                    deq = (_nvfp4_dequant_part_bytes(p) if p["kind"] == "nvfp4"
                           else _fp8_dequant_part_bytes(p))
                    for i in range(0, len(deq), 8 * 1024 * 1024):
                        chunk = deq[i:i + 8 * 1024 * 1024]
                        net_account(nid, to_node=len(chunk))
                        yield chunk
                    continue
                with open(p["fn"], "rb") as f:
                    f.seek(p["off"])
                    left = p["nbytes"]
                    while left > 0:
                        chunk = f.read(min(8 * 1024 * 1024, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                        net_account(nid, to_node=len(chunk))
                        yield chunk
            # whole slice streamed -> the worker now mmap-loads + fuses + places it
            log_activity(f"  {_host}: received L{start}-{end} ({total / GB:.2f} GB), building shard")
            # live per-shard progress: each Lxx layer-slice (+ the embed/head slices) is ONE shard
            # the dashboard counts. Workers pull their layers sequentially, so these completions
            # pace real load progress far better than the one-tick-per-node stage count.
            # match on the HF target id (the /weights `model` param is the target, e.g.
            # 'ModelCloud/MiniMax-M2-BF16', NOT the friendly 'minimax-m2') so the counter advances.
            # With parallel loads, find THIS model's card among the in-flight cards by target.
            ld = next((c for c in engine.loadings.values() if c.get("target") == target), None)
            if ld is not None:
                ld["ready"] = ld.get("ready", 0) + max(1, (end - start)
                                                       + (1 if embed else 0)
                                                       + (1 if head else 0))

        return StreamingResponse(
            _gen(), media_type="application/octet-stream",
            headers={"Content-Length": str(total),
                     "Content-Disposition":
                         f'attachment; filename="{friendly}-{start}_{end}.safetensors"'})

    @app.get("/weights_tp")
    async def weights_tp(model: str, start: int, end: int, tp_rank: int, tp_size: int,
                         embed: int = 0, head: int = 0, weights: str = ""):
        # TP-v2 per-rank serve (#62 follow-on): return this stage's tensors ALREADY SLICED for
        # (tp_rank, tp_size) — column-parallel q/k/v/gate/up on dim 0, row-parallel o/down on dim 1
        # (bias dropped), embed/norm/head/layernorm/rotary whole. The row slice is non-contiguous so
        # we read+materialize (NOT byte-range) and serve a small built safetensors blob. Lets a TP
        # rank hold only ~1/tp of each layer instead of the v1 load-full-then-shard footprint.
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        if tp_size < 1 or tp_rank < 0 or tp_rank >= tp_size:
            return JSONResponse({"error": f"bad tp (rank={tp_rank}, size={tp_size})"},
                                status_code=400)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        # heterogeneous TP: the rank passes the group's per-rank capacity weights (comma list) so the
        # serve slices match the rank's reduced-dim structure; empty -> uniform 1/tp (backward compat).
        try:
            wlist = [float(x) for x in weights.split(",") if x.strip()] if weights else None
        except ValueError:
            wlist = None
        try:
            blob = await asyncio.to_thread(
                _build_weight_tp_blob, d, start, end, bool(embed), bool(head), tp_rank, tp_size, wlist)
        except Exception as exc:
            return JSONResponse({"error": f"tp-slice build failed: {exc!r}"}, status_code=500)
        total = len(blob)
        # meter to the node whose layer range contains this slice (controller -> node), same _owns
        # logic as /weights; a TP rank requests its FULL [0,L) range so the containing match holds.
        def _owns(n) -> bool:
            ls, le = n.layer_start, n.layer_end
            if ls is None or le is None:
                return False
            if end > start:
                return ls <= start and end <= le
            return ls == start
        # Match by tp_rank too: both ranks share layer range [0,L], so _owns alone matches BOTH and
        # next() would mislabel every slice to the first node. Prefer the node assigned THIS tp_rank;
        # fall back to the old range-only match if none (keeps metering working pre-assignment).
        nid = next((n.node_id for n in registry._nodes.values()
                    if _owns(n) and getattr(n, "tp_rank", None) == tp_rank), None) \
            or next((n.node_id for n in registry._nodes.values() if _owns(n)), None)
        _host = registry._nodes[nid].hostname if nid in registry._nodes else "?"
        log_activity(f"serving {friendly} TP weights L{start}-{end} rank {tp_rank}/{tp_size} "
                     f"-> {_host} ({total / GB:.2f} GB)")

        def _gen():
            for i in range(0, total, 8 * 1024 * 1024):
                chunk = blob[i:i + 8 * 1024 * 1024]
                net_account(nid, to_node=len(chunk))
                yield chunk
            ld = next((c for c in engine.loadings.values() if c.get("target") == target), None)
            if ld is not None:
                ld["ready"] = ld.get("ready", 0) + max(1, (end - start)
                                                       + (1 if embed else 0)
                                                       + (1 if head else 0))

        return StreamingResponse(
            _gen(), media_type="application/octet-stream",
            headers={"Content-Length": str(total),
                     "Content-Disposition":
                         f'attachment; filename="{friendly}-{start}_{end}-tp{tp_rank}of{tp_size}.safetensors"'})

    @app.get("/experts")
    async def experts(model: str, layer: int, e0: int, k: int):
        # Serve experts [e0:e0+k] of one MoE layer as a safetensors blob, raw byte-range from the
        # source files, so a worker fetches + int4-packs one chunk of experts at a time and a big
        # MoE layer never lands whole in RAM (#62). TWO checkpoint layouts, ONE round-trip each:
        #  - NON-FUSED (e.g. MiniMax-M2: *.experts.{e}.{proj}.weight) -> keys '{local_e}.{proj}'
        #    (w1/w2/w3 or gate_proj/up_proj/down_proj), 2D per (expert, projection); the worker
        #    fuses gate+up then packs.
        #  - FUSED (e.g. qwen3.6-35b-a3b: 3D experts.gate_up_proj/down_proj) -> keys 'gate_up_proj'
        #    and 'down_proj', each a 3D [k, out, in] slice; the worker packs each slice directly.
        # The worker auto-detects which layout it got from the returned keys (#75). No per-chunk
        # activity log (thousands of lines).
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        if k <= 0 or e0 < 0 or layer < 0:    # guard: negative k -> negative nbytes -> invalid blob
            return JSONResponse({"error": f"bad range (layer={layer}, e0={e0}, k={k})"},
                                status_code=400)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        header_bytes, parts, total = await asyncio.to_thread(
            _plan_experts_chunk, d, layer, e0, k)
        if header_bytes is None:                 # FUSED checkpoint -> serve the 3D fused slices (#75)
            header_bytes, parts, total = await asyncio.to_thread(
                _plan_experts_chunk_fused, d, layer, e0, k)
        if header_bytes is None:
            return JSONResponse({"error": f"no expert tensors (layer {layer})"},
                                status_code=404)
        # meter to the node whose layer range contains this layer (controller -> node)
        def _owns(n) -> bool:
            ls, le = n.layer_start, n.layer_end
            return ls is not None and le is not None and ls <= layer < le
        nid = next((n.node_id for n in registry._nodes.values() if _owns(n)), None)

        def _gen():
            net_account(nid, to_node=len(header_bytes))
            yield header_bytes
            for fn, foff, nbytes in parts:
                with open(fn, "rb") as f:
                    f.seek(foff)
                    left = nbytes
                    while left > 0:
                        chunk = f.read(min(8 * 1024 * 1024, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                        net_account(nid, to_node=len(chunk))
                        yield chunk

        return StreamingResponse(_gen(), media_type="application/octet-stream",
                                 headers={"Content-Length": str(total)})

    # ---- Ollama API ----
    @app.get("/api/version")
    async def api_version() -> dict:
        return {"version": OLLAMA_API_VERSION}

    @app.get("/api/tags")
    async def api_tags() -> dict:
        # Only advertise models whose weights are actually present here — a model
        # that isn't downloaded yet can't be distributed, so it isn't "available".
        out = [_tag_entry(name) for name in MODELS if model_ready(MODELS[name][0])]
        # advertise ALIASES too (e.g. 'qwen2.5:14b' -> 'qwen2.5:14b-instruct') so a client can
        # discover + use the alias name; same target/size, just the alias display name.
        for alias, canon in MODEL_ALIASES.items():
            if canon in MODELS and model_ready(MODELS[canon][0]):
                e = _tag_entry(canon)
                e["name"] = e["model"] = _ollama_name(alias)
                out.append(e)
        return {"models": out}

    @app.get("/v1/models")
    async def v1_models() -> dict:
        names = [name for name in MODELS if model_ready(MODELS[name][0])]
        names += [a for a, c in MODEL_ALIASES.items() if c in MODELS and model_ready(MODELS[c][0])]
        return {"object": "list", "data": [
            {"id": _ollama_name(name), "object": "model",
             "created": int(START_TIME), "owned_by": "infinitemodel"} for name in names]}

    @app.post("/api/show")
    async def api_show(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        spec = resolve_spec(friendly)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if not model_ready(target):
            return JSONResponse(
                {"error": f"model '{friendly}' is not downloaded on the controller"},
                status_code=404)
        d = _local_model_dir(target)             # real measured params/size (MoE-correct)
        if d and spec:
            spec = spec_with_measurements(spec, d)
        caps = (["embedding"] if (spec and getattr(spec, "is_embedding", False))
                else ["completion", "chat"])
        return JSONResponse({
            "license": "see model card", "modelfile": f"# InfiniteModel distributed\nFROM {target}",
            "parameters": "", "template": "{{ .Prompt }}",
            "details": _details(spec), "model_info": _model_info(spec),
            "capabilities": caps,
            "infinitemodel": {"target": target, "draft": MODELS[friendly][1],
                              "distributed": True, "engine": VERSION},
        })

    @app.get("/api/ps")
    async def api_ps() -> dict:
        nodes = registry.alive_sorted()
        gpus = [n for n in nodes if n.vram_total_gb > 0]
        vram_total = int(sum(n.vram_total_gb for n in gpus) * GB)

        def _model_vram(lm: LoadedModel) -> int:
            # per-STAGE gpu_bytes (survives node-sharing; the node's single shard_gpu_bytes
            # would be overwritten when a 2nd model lands on it)
            return sum(s.gpu_bytes for s in lm.plan.stages)
        total_vram_used = sum(_model_vram(lm) for lm in engine.models.values())
        # Fleet GPU/VRAM + RAM summary so Ollama dashboards can show TOTAL GPU VRAM
        # (capacity across all GPU nodes). All byte counts, matching Ollama's convention.
        pool = {
            "vram_total": vram_total,
            "vram_used": int(total_vram_used),
            "vram_free": max(0, vram_total - int(total_vram_used)),
            "ram_total": int(sum(n.total_mem_gb for n in nodes) * GB),
            "usable_total": int(sum(n.usable_total_gb for n in nodes) * GB),
            "gpus": [{"name": n.hostname, "vram_total": int(n.vram_total_gb * GB),
                      "vram_used": int(n.vram_used_gb * GB)} for n in gpus],  # live (all users)
        }
        _alias_by_canon: dict[str, list] = {}     # canonical -> [alias keys] for the echo below
        for _a, _c in MODEL_ALIASES.items():
            _alias_by_canon.setdefault(_c, []).append(_a)
        models = []
        for lm in engine.models.values():
            entry = {
                "name": _ollama_name(lm.friendly), "model": _ollama_name(lm.friendly),
                "size": lm.spec.total_weight_bytes, "size_vram": _model_vram(lm),
                "digest": _digest(lm.target_id), "details": _details(lm.spec),
                "expires_at": _iso(time.time() + 365 * 86400),
                "context_length": lm.ctx,   # Ollama-standard field (loaded context window)
                "infinitemodel": {
                    "ctx": lm.ctx, "pool_usable_gb": round(lm.plan.pool_usable_gb, 2),
                    "stages": [{"host": s.hostname, "layers": [s.layer_start, s.layer_end],
                                "embed": s.has_embed, "head": s.has_head,
                                "est_gb": round(s.est_gb, 2)} for s in lm.plan.stages]},
            }
            models.append(entry)
            # ALIAS echo: also list the loaded model under any alias name (e.g. 'qwen2.5:14b' when
            # 'qwen2.5-14b-instruct' is loaded) so a client configured with the alias sees it running.
            for _a in _alias_by_canon.get(lm.friendly, []):
                ae = dict(entry); ae["name"] = ae["model"] = _ollama_name(_a)
                models.append(ae)
        return {"models": models, "pool": pool}

    async def _serve_embed(model: str, inputs, mode: str, ip: str = "?") -> JSONResponse:
        """Shared embedding serve for /api/embed, /api/embeddings (legacy) and /v1/embeddings.
        No auto-load (match the generate policy: 404 if the model isn't resident). Tokenizes on
        the controller (NO chat template, NO task-prefix), runs one encoder forward on the node,
        and shapes the response per `mode` ('ollama' | 'legacy' | 'openai')."""
        try:
            friendly = resolve_model_name(model)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            lm = await engine.ensure_loaded(friendly, 0)
        except ValueError as exc:   # not loaded -> 404 (no auto-load)
            return JSONResponse({"error": str(exc), "model": model}, status_code=404)
        if not getattr(lm.spec, "is_embedding", False):
            return JSONResponse(
                {"error": f"model '{friendly}' is not an embedding model; use /api/chat"},
                status_code=400)
        # Normalize inputs to list[str] (accept a string or a list of strings).
        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = [str(t) for t in inputs]
        else:
            return JSONResponse({"error": "input must be a string or a list of strings"},
                                status_code=400)
        if not texts:
            return JSONResponse({"error": "no input text provided"}, status_code=400)
        rec = _inflight_admit(ip, friendly, 1)
        if rec is None:
            return JSONResponse(
                {"error": f"queue full for '{friendly}' — retry shortly"}, status_code=503)
        try:
            _inflight_start(rec)
            tok = lm.tokenizer
            max_len = min(8192, int(getattr(lm.spec, "max_ctx", DEFAULT_CTX) or DEFAULT_CTX))
            enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            attn = enc["attention_mask"]
            vecs = await engine.embed(friendly, enc["input_ids"], attn)
        except Exception as exc:
            log_activity(f"embed {model}: FAILED — {exc!r}")
            print(f"[embed] {model} FAILED: {exc!r}", flush=True)
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "model": model},
                                status_code=500)
        finally:
            _inflight_release(rec)
        display = _ollama_name(friendly)
        n_tok = int(attn.sum())
        if mode == "openai":
            return JSONResponse({
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": v}
                         for i, v in enumerate(vecs)],
                "model": display,
                "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok}})
        if mode == "legacy":   # /api/embeddings -> single vector
            return JSONResponse({"embedding": vecs[0] if vecs else []})
        # /api/embed (Ollama)
        return JSONResponse({"model": display, "embeddings": vecs, "prompt_eval_count": n_tok})

    @app.post("/api/embed")
    async def api_embed(req: Request) -> JSONResponse:
        body = await req.json()
        inputs = body.get("input", body.get("prompt", body.get("text", "")))
        return await _serve_embed(body.get("model", ""), inputs, mode="ollama",
                                  ip=_client_ip(req))

    @app.post("/api/embeddings")   # legacy Ollama single-embedding endpoint
    async def api_embeddings(req: Request) -> JSONResponse:
        body = await req.json()
        inputs = body.get("prompt", body.get("input", ""))
        return await _serve_embed(body.get("model", ""), inputs, mode="legacy",
                                  ip=_client_ip(req))

    @app.post("/v1/embeddings")    # OpenAI-compatible
    async def v1_embeddings(req: Request) -> JSONResponse:
        body = await req.json()
        inputs = body.get("input", "")
        return await _serve_embed(body.get("model", ""), inputs, mode="openai",
                                  ip=_client_ip(req))

    @app.post("/api/push")
    @app.post("/api/create")
    @app.post("/api/copy")
    async def api_manage() -> JSONResponse:
        return JSONResponse({"status": "not supported by InfiniteModel "
                             "(models are configured server-side)"}, status_code=501)

    async def _start_download(friendly: str) -> dict:
        """Kick off a background download of a configured model to the controller
        cache (fire-and-forget). Idempotent: no-op if ready. If a pull is already in
        flight, a pending pause/stop is CANCELLED (so Resume-during-pausing un-pauses
        the live thread instead of being a silent no-op)."""
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if model_ready(target):
            return {"ok": True, "status": "ready"}
        if friendly in DOWNLOADING:
            # Already pulling — if a pause/stop was pending, drop it so the running thread
            # keeps going (the between-files check now sees no signal). No new _dl needed.
            if DOWNLOAD_CONTROL.pop(friendly, None) is not None:
                log_activity(f"download {friendly}: resume (cancelled pending halt)")
                return {"ok": True, "status": "resuming"}
            return {"ok": True, "status": "downloading"}
        DOWNLOADING.add(friendly)
        DOWNLOAD_ERROR.pop(friendly, None)   # clear any prior failure on a fresh attempt
        DOWNLOAD_CONTROL.pop(friendly, None)  # drop any stale pause/stop signal from a prior run
        if DOWNLOAD_STATE.pop(friendly, None) is not None:  # starting/resuming -> no longer halted
            save_download_state()
        epoch = DOWNLOAD_EPOCH[friendly] = DOWNLOAD_EPOCH.get(friendly, 0) + 1
        log_activity(f"download {friendly}: starting")

        async def _dl():
            total = await asyncio.to_thread(_hf_total_bytes, target)
            DOWNLOAD_PROGRESS[friendly] = {
                "downloaded": await asyncio.to_thread(_hf_cache_bytes, target),
                "total": total}

            async def _poll():   # update bytes-so-far while the download runs
                try:
                    while friendly in DOWNLOADING and DOWNLOAD_EPOCH.get(friendly) == epoch:
                        db = await asyncio.to_thread(_hf_cache_bytes, target)
                        pr = DOWNLOAD_PROGRESS.get(friendly)
                        if pr is not None:
                            pr["downloaded"] = db
                        await asyncio.sleep(2)
                except asyncio.CancelledError:
                    pass

            poller = asyncio.create_task(_poll())
            halted = None
            try:
                # Interruptible per-file pull -> 'done' | 'paused' | 'stopped'.
                result = await asyncio.to_thread(_pull_repo_interruptible, friendly, target)
                # If a clear (or a fresh start) bumped the epoch while we were pulling, THIS
                # run is stale — that handler already cleaned up; don't write/resurrect state.
                if DOWNLOAD_EPOCH.get(friendly) != epoch:
                    return
                if result in ("paused", "stopped"):
                    halted = result
                    DOWNLOAD_STATE[friendly] = result    # persist so a restart keeps it halted
                    await asyncio.to_thread(save_download_state)
                    done = (DOWNLOAD_PROGRESS.get(friendly) or {}).get("downloaded", 0)
                    print(f"[model] download {friendly} {result} at {done / GB:.1f} GB")
                    log_activity(f"download {friendly}: {result}")
                else:
                    # every file now in the HF cache -> migrate to models/ + purge the dup
                    await asyncio.to_thread(_controller_model_dir, target)
                    _invalidate_ready_cache(target)
                    print(f"[model] downloaded {friendly} ({target})")
                    log_activity(f"download {friendly}: complete")
            except Exception as exc:
                if DOWNLOAD_EPOCH.get(friendly) != epoch:
                    return                               # superseded -> swallow
                msg = f"{type(exc).__name__}: {exc}"
                low = msg.lower()
                if any(k in low for k in ("gated", "403", "401", "awaiting", "access to model",
                                          "restricted", "you must")):
                    msg += "  (gated repo — accept the license for this model on huggingface.co "
                    msg += "with the account whose token the controller uses)"
                DOWNLOAD_ERROR[friendly] = msg[:400]
                print(f"[model] download failed for {friendly}: {exc!r}")
                log_activity(f"download {friendly}: FAILED ({type(exc).__name__})")
            finally:
                poller.cancel()
                if DOWNLOAD_EPOCH.get(friendly) == epoch:   # only OUR run owns this state
                    DOWNLOADING.discard(friendly)
                    DOWNLOAD_CONTROL.pop(friendly, None)
                    # done/error -> drop the progress bar; pause/stop -> KEEP it frozen so the
                    # dashboard shows where it halted (and offers Resume from there).
                    if halted is None:
                        DOWNLOAD_PROGRESS.pop(friendly, None)

        asyncio.create_task(_dl())
        return {"ok": True, "status": "downloading"}

    async def _do_delete(friendly: str) -> dict:
        """Delete a model's weights from the controller cache. Refuses if the
        model is currently loaded (unload first)."""
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if friendly in engine.models:
            return {"ok": False, "error": "model is currently loaded — unload it first"}
        if friendly in DOWNLOADING:
            return {"ok": False, "error": "model is downloading — wait for it to finish"}
        deleted = await asyncio.to_thread(delete_model_cache, target)
        if deleted:
            print(f"[model] deleted {friendly} ({target}) from controller cache")
        return {"ok": deleted, "error": None if deleted else "model not present in cache"}

    @app.post("/download")           # dashboard: pull a configured model to controller
    async def download(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse(await _start_download(friendly))

    def _resolve_or_404(model: str):
        try:
            return resolve_model_name(model), None
        except ValueError as exc:
            return None, JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    @app.post("/download/pause")     # dashboard: pause an in-flight download (cache kept, resumable)
    async def download_pause(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly not in DOWNLOADING:
            return JSONResponse({"ok": False, "error": "not currently downloading"}, status_code=409)
        DOWNLOAD_CONTROL[friendly] = "pause"   # the per-file pull stops after the current shard
        log_activity(f"download {friendly}: pause requested")
        return JSONResponse({"ok": True, "status": "pausing"})

    @app.post("/download/stop")      # dashboard: stop an in-flight download (cache kept, resumable)
    async def download_stop(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly not in DOWNLOADING:
            return JSONResponse({"ok": False, "error": "not currently downloading"}, status_code=409)
        DOWNLOAD_CONTROL[friendly] = "stop"
        log_activity(f"download {friendly}: stop requested")
        return JSONResponse({"ok": True, "status": "stopping"})

    @app.post("/download/resume")    # dashboard: resume a paused/stopped download from the cache
    async def download_resume(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        # _start_download clears the persisted halt + stale control signal, then re-runs the
        # per-file pull — cached files are skipped instantly and the partial file resumes.
        return JSONResponse(await _start_download(friendly))

    @app.post("/download/clear")     # dashboard: wipe a model's cached + partial files ("reset")
    async def download_clear(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly in engine.models:
            return JSONResponse({"ok": False, "error": "model is loaded — unload it first"},
                                status_code=409)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if friendly in DOWNLOADING:
            DOWNLOAD_CONTROL[friendly] = "stop"   # ask the in-flight pull to stop between files
        else:
            DOWNLOAD_CONTROL.pop(friendly, None)  # nothing running -> don't leave a stale signal
        # Bump the epoch so the (possibly still-running) _dl for this model sees its result is
        # stale and won't re-persist DOWNLOAD_STATE after we clear it (the clear-resurrect race).
        DOWNLOAD_EPOCH[friendly] = DOWNLOAD_EPOCH.get(friendly, 0) + 1
        DOWNLOADING.discard(friendly)
        DOWNLOAD_PROGRESS.pop(friendly, None)
        DOWNLOAD_ERROR.pop(friendly, None)
        if DOWNLOAD_STATE.pop(friendly, None) is not None:
            save_download_state()
        # Measure both locations first (cache copy + any models/<name>), then delete both.
        # rmtree uses ignore_errors, so a file still locked by an in-flight pull thread is
        # skipped — a second Clear (after the thread exits between files) mops up any residue.
        def _measure() -> int:
            n = _hf_cache_bytes(target)
            mdir = os.path.join(MODELS_DIR, _safe_name(target))
            if os.path.isdir(mdir):
                for root, _dirs, files in os.walk(mdir):
                    for f in files:
                        with contextlib.suppress(OSError):
                            n += os.path.getsize(os.path.join(root, f))
            return n
        freed = await asyncio.to_thread(_measure)
        removed = await asyncio.to_thread(delete_model_cache, target)
        log_activity(f"download {friendly}: cache cleared (~{freed / GB:.1f} GB)")
        print(f"[model] cleared cache for {friendly} ({target})")
        return JSONResponse({"ok": True, "removed": removed, "freed_gb": round(freed / GB, 2)})

    @app.post("/add_model")          # dashboard: register + download ANY Hugging Face id
    async def add_model(model: str, name: str = "") -> JSONResponse:
        # `name` (optional): override the client-facing model name instead of deriving it from
        # the HF id. Lets a precision-suffixed repo (e.g. ModelCloud/MiniMax-M2-BF16) be served
        # under a clean, quant-agnostic name (e.g. minimax-m2) — quant is a load-time choice, so
        # it shouldn't live in the name. Re-registering an already-cached HF id under a new name
        # is instant (no re-download — the cache is keyed by HF id, not the friendly name).
        hf = (model or "").strip()
        if "/" not in hf or " " in hf or hf.count("/") > 1:
            return JSONResponse({"ok": False,
                                 "error": "enter a Hugging Face id like org/name"},
                                status_code=400)
        # A user-supplied override may be typed in the Ollama 'family:size' form ('qwen3:4b');
        # collapse it to the canonical colon-free dash key ('qwen3-4b') so the registry key,
        # the URL query param, and the on-disk filename stay simple — the colon display is
        # rendered on demand by _ollama_name(). Validate AFTER normalizing (so ':' is allowed
        # as input but never stored as a key).
        friendly = _normalize_model_request(name) if (name or "").strip() else _friendly_from_hf(hf)
        if not re.fullmatch(r"[a-z0-9._-]+", friendly):
            return JSONResponse({"ok": False,
                                 "error": "name must be lowercase [a-z0-9._-] (':' allowed as the size separator)"},
                                status_code=400)
        if friendly not in MODELS:
            MODELS[friendly] = (hf, hf)          # draft = target (no speculative)
            CUSTOM_MODELS[friendly] = hf
            save_custom_models()
            log_activity(f"added model {friendly} ({hf})")
        r = await _start_download(friendly)
        return JSONResponse({"ok": True, "friendly": friendly, "target": hf,
                             "status": r.get("status")})

    @app.post("/delete")             # dashboard: delete a model from controller
    async def delete_model(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        r = await _do_delete(friendly)
        return JSONResponse(r, status_code=200 if r["ok"] else 409)

    @app.post("/forget")             # remove a custom-model REGISTRY entry but KEEP its weight files
    async def forget_model(model: str) -> JSONResponse:
        """Unregister a custom (added) model: drop its friendly->HF mapping from CUSTOM_MODELS +
        MODELS + custom_models.json. UNLIKE /delete, this does NOT delete the cached weight files
        — the model stays on disk, just no longer registered. Refuses if currently loaded."""
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        if friendly in engine.models:
            return JSONResponse({"ok": False, "error": "model is loaded — unload it first"},
                                status_code=409)
        if friendly not in CUSTOM_MODELS:
            return JSONResponse({"ok": False, "error": f"'{friendly}' is not a registered custom "
                                "model (built-ins can't be forgotten)"}, status_code=400)
        hf = CUSTOM_MODELS.pop(friendly, None)
        MODELS.pop(friendly, None)
        save_custom_models()
        print(f"[model] forgot registry entry {friendly} ({hf}) — weight files KEPT", flush=True)
        return JSONResponse({"ok": True, "forgot": friendly, "hf": hf, "files_kept": True})

    @app.post("/nodeconfig")         # dashboard: enable/disable a node's CPU/RAM or GPU/VRAM
    async def nodeconfig(host: str, ram: Optional[bool] = None,
                         vram: Optional[bool] = None) -> JSONResponse:
        # Keyed by hostname so the choice sticks across reconnects; persisted to
        # node_config.json so it survives a controller restart. A tier change re-plans
        # ONLY the resident models that actually use a node on this host (surgical —
        # other models keep running; they'll pick up freed/added capacity on next load).
        cfg = NODE_CONFIG.setdefault(host, {"ram": True, "vram": True})
        if ram is not None:
            cfg["ram"] = ram
        if vram is not None:
            cfg["vram"] = vram
        save_node_config()
        host_nids = {nid for nid, n in registry._nodes.items() if n.hostname == host}
        for fr in [fr for fr, m in engine.models.items()
                   if any(nid in m.stage_node_ids for nid in host_nids)]:
            engine.invalidate_model(fr, f"tier change on {host}")
        return JSONResponse({"ok": True, "host": host, "config": cfg})

    @app.post("/nodeconfig_all")     # dashboard: enable/disable a tier on EVERY node at once
    async def nodeconfig_all(tier: str, enabled: bool) -> JSONResponse:
        """Bulk version of /nodeconfig: set one tier (ram|vram) for every known host, persist,
        and re-plan each resident model ONCE (fleet-wide capacity changed). Drives the
        'all CPU' / 'all GPU' master checkboxes."""
        if tier not in ("ram", "vram"):
            return JSONResponse({"ok": False, "error": "tier must be 'ram' or 'vram'"},
                                status_code=400)
        hosts = {n.hostname for n in registry.alive_sorted()} | set(NODE_CONFIG.keys())
        for h in hosts:
            NODE_CONFIG.setdefault(h, {"ram": True, "vram": True})[tier] = enabled
        save_node_config()
        for fr in list(engine.models.keys()):   # capacity changed everywhere -> re-plan all
            engine.invalidate_model(fr, f"bulk tier change ({tier}={enabled})")
        return JSONResponse({"ok": True, "tier": tier, "enabled": enabled, "hosts": len(hosts)})

    @app.get("/logs")                # #logs: curl-able log — controller's own, or a worker's (relayed)
    async def get_logs(tail: int = 200, node: str = "") -> Response:
        """GET /logs[?tail=N][&node=<host|node_id>]. No node -> the CONTROLLER's stdout/stderr ring.
        node given -> that worker's log lines relayed on its heartbeats (so a worker box with no
        console/journal access is still debuggable). Plain text, newest last."""
        tail = max(1, min(int(tail or 200), NODE_LOGS_MAX))
        if node:
            nid = node if node in NODE_LOGS else next(
                (i for i, n in registry._nodes.items() if n.hostname == node), node)
            buf = NODE_LOGS.get(nid)
            if not buf:
                return Response(content=f"(no logs buffered for node {node!r} — workers relay logs "
                                "on heartbeat once they're on m4c31+)\n", media_type="text/plain")
            return Response(content="\n".join(buf[-tail:]) + "\n", media_type="text/plain")
        return Response(content="\n".join(tail_logs(tail)) + "\n", media_type="text/plain")

    @app.post("/config")             # dashboard: runtime engine config (persisted)
    async def set_config(max_loaded: Optional[int] = None,
                         auto_unload: Optional[bool] = None,
                         queue_depth: Optional[int] = None,
                         auto_tp: Optional[bool] = None,
                         auto_tp_ratio: Optional[float] = None,
                         auto_load: Optional[bool] = None,
                         autoload_quant: Optional[str] = None,
                         autoload_ctx: Optional[int] = None,
                         autoload_mode: Optional[str] = None,
                         vram_weights_first: Optional[bool] = None,
                         gen_stall_s: Optional[float] = None,
                         persist: Optional[str] = None,
                         unpersist: Optional[str] = None) -> JSONResponse:
        if persist is not None:                          # #77: keep this model across restarts
            with contextlib.suppress(ValueError):
                fr = resolve_model_name(persist)
                _lm = engine.models.get(fr)
                _pm = dict(ENGINE_CONFIG.get("persist_models") or {})
                _pm[fr] = {"ctx": (_lm.ctx if _lm else 0), "quant": (_lm.quant if _lm else "none")}
                ENGINE_CONFIG["persist_models"] = _pm
                log_activity(f"persist: {fr} will auto-reload on startup "
                             f"(ctx={_pm[fr]['ctx']}, quant={_pm[fr]['quant']})")
        if unpersist is not None:
            with contextlib.suppress(ValueError):
                fr = resolve_model_name(unpersist)
                _pm = dict(ENGINE_CONFIG.get("persist_models") or {})
                if _pm.pop(fr, None) is not None:
                    ENGINE_CONFIG["persist_models"] = _pm
                    log_activity(f"persist: {fr} removed (no longer auto-reloaded on startup)")
        if max_loaded is not None:
            ENGINE_CONFIG["max_loaded"] = max(1, int(max_loaded))
        if auto_unload is not None:
            ENGINE_CONFIG["auto_unload"] = bool(auto_unload)
        if queue_depth is not None:
            ENGINE_CONFIG["queue_depth"] = max(0, int(queue_depth))
        if auto_tp is not None:                          # #87 D: auto-route cpu-bound models to CPU TP
            ENGINE_CONFIG["auto_tp"] = bool(auto_tp)
        if auto_tp_ratio is not None:                    # trigger when weights > ratio x GPU pool
            ENGINE_CONFIG["auto_tp_ratio"] = max(0.0, float(auto_tp_ratio))
        if auto_load is not None:                        # auto-load a requested model that isn't resident
            ENGINE_CONFIG["auto_load"] = bool(auto_load)
        if autoload_quant is not None:                   # #autoload-smallest: quant for auto-loads
            _aq = str(autoload_quant).lower()
            if _aq in ("int4", "int8", "none"):
                ENGINE_CONFIG["autoload_quant"] = _aq
        if autoload_ctx is not None:                      # #auto-defaults: default ctx for auto/click loads
            ENGINE_CONFIG["autoload_ctx"] = max(0, int(autoload_ctx))
        if autoload_mode is not None:                     # #auto-defaults: default placement mode
            _am = str(autoload_mode).lower()
            if _am in LOAD_MODES:
                ENGINE_CONFIG["autoload_mode"] = _am
        if vram_weights_first is not None:               # #vram-weights-first: pack weights into free VRAM
            ENGINE_CONFIG["vram_weights_first"] = bool(vram_weights_first)
        if gen_stall_s is not None:                       # #gen-stall-watchdog: wedged-gen reclaim threshold (0=off)
            ENGINE_CONFIG["gen_stall_s"] = max(0.0, float(gen_stall_s))
        save_engine_config()
        log_activity(f"config: max_loaded={ENGINE_CONFIG['max_loaded']} "
                     f"auto_unload={ENGINE_CONFIG['auto_unload']} "
                     f"queue_depth={ENGINE_CONFIG.get('queue_depth', DEFAULT_QUEUE_DEPTH)}")
        return JSONResponse({"ok": True, "config": ENGINE_CONFIG})

    @app.get("/inspect_audio")       # #22 inc 5: introspect an Omni model's AUDIO + Thinker interface
    async def inspect_audio(model: str = "qwen2.5-omni-7b") -> JSONResponse:
        """Meta-load (zero weights) a Qwen2.5-Omni-style checkpoint and report what's needed to
        (a) LOAD its Thinker text model on the pipeline and (b) run audio input: get_text_config
        shape + whether AutoModelForCausalLM can build it with sliceable .model.layers; the audio
        tower class + get_audio_features signature; audio_token_id; the feature extractor."""
        def _run():
            out: dict = {"model": model}
            try:
                import torch, inspect as _inspect
                from transformers import AutoConfig
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                cfg = AutoConfig.from_pretrained(_local_model_dir(target) or target)
                out["config_class"] = type(cfg).__name__
                def g(o, *names):
                    for n in names:
                        v = getattr(o, n, None)
                        if v is not None:
                            return v
                    return None
                out["audio_token_id"] = g(cfg, "audio_token_id", "audio_token_index")
                out["audio_start_token_id"] = g(cfg, "audio_start_token_id")
                out["image_token_id"] = g(cfg, "image_token_id", "image_token_index")
                out["top_config_keys"] = [k for k in vars(cfg).keys() if not k.startswith("_")][:40]
                tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", None)
                out["has_get_text_config"] = hasattr(cfg, "get_text_config")
                if tcfg is not None:
                    out["text_config_class"] = type(tcfg).__name__
                    out["text_num_layers"] = g(tcfg, "num_hidden_layers")
                    out["text_hidden_size"] = g(tcfg, "hidden_size")
                    out["text_rope_scaling"] = getattr(tcfg, "rope_scaling", None)
                # can the worker build the Thinker text model from text_config?
                try:
                    from transformers import AutoModelForCausalLM
                    with torch.device("meta"):
                        tm = AutoModelForCausalLM.from_config(tcfg)
                    out["text_model_class"] = type(tm).__name__
                    inner = getattr(tm, "model", tm)
                    out["text_inner_children"] = [n for n, _ in inner.named_children()][:20]
                    layers = getattr(inner, "layers", None)
                    out["text_layers_count"] = (len(layers) if layers is not None else None)
                    out["text_buildable"] = layers is not None
                except Exception as exc:
                    out["text_build_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                # audio tower: build the full Omni/thinker on meta and inspect
                for auto_name in ("AutoModelForTextToWaveform", "AutoModel"):
                    try:
                        import transformers as _tf
                        AutoCls = getattr(_tf, auto_name, None)
                        if AutoCls is None:
                            continue
                        with torch.device("meta"):
                            m = AutoCls.from_config(cfg)
                        out["full_model_class"] = type(m).__name__
                        out["full_auto_used"] = auto_name
                        thinker = getattr(m, "thinker", m)
                        out["thinker_children"] = [n for n, _ in thinker.named_children()][:20]
                        at = getattr(thinker, "audio_tower", None)
                        if at is not None:
                            out["audio_tower_class"] = type(at).__name__
                            with contextlib.suppress(Exception):
                                out["audio_tower_forward_sig"] = str(_inspect.signature(at.forward))
                        out["has_get_audio_features"] = hasattr(thinker, "get_audio_features")
                        with contextlib.suppress(Exception):
                            out["get_audio_features_sig"] = str(_inspect.signature(thinker.get_audio_features))
                        break
                    except Exception as exc:
                        out[f"{auto_name}_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                # feature extractor / processor for audio
                try:
                    from transformers import AutoProcessor
                    proc = AutoProcessor.from_pretrained(_local_model_dir(target) or target)
                    out["processor_class"] = type(proc).__name__
                    fe = getattr(proc, "feature_extractor", None)
                    out["feature_extractor_class"] = type(fe).__name__ if fe is not None else None
                    out["proc_audio_token"] = getattr(proc, "audio_token", None)
                except Exception as exc:
                    out["processor_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/audio_test")          # #22 inc 5b: run the AUDIO encoder end-to-end on a test tone
    async def audio_test(model: str = "qwen2.5-omni-7b", secs: float = 2.0,
                         freq: float = 440.0, clips: int = 1) -> JSONResponse:
        """Synthesize sine tone(s) @16 kHz, run the feature extractor + Omni audio tower, and
        report shapes — verifies increment 5b (the encoder) against the real model with NO
        text-model load. clips>1 synthesizes that many DISTINCT-duration tones to exercise the
        MULTI-CLIP encode path (per-clip counts must sum to the flat embed-row count)."""
        def _run():
            out: dict = {"model": model}
            try:
                import math
                friendly = resolve_model_name(model) if model else (
                    next(iter(engine.models)) if engine.models else "qwen2.5-omni-7b")
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                sr = 16000
                import numpy as _np
                nclips = max(1, int(clips))
                tones = []
                for ci in range(nclips):
                    # distinct duration + pitch per clip so per-clip counts differ (a real
                    # multi-clip alignment test, not N identical clips).
                    dur = float(secs) * (1.0 + 0.5 * ci)
                    f = float(freq) * (1.0 + 0.25 * ci)
                    n = max(1, int(dur * sr))
                    t = _np.arange(n, dtype=_np.float32) / sr
                    tones.append((0.2 * _np.sin(2.0 * math.pi * f * t)).astype(_np.float32))
                out["clips"] = nclips
                out["clip_durations_s"] = [round(len(x) / sr, 2) for x in tones]
                r = _encode_audio(target, tones, sampling_rate=sr)
                emb = r["audio_embeds"]
                cts = r.get("counts")
                out["counts_sum_matches_embeds"] = bool(
                    cts is not None and sum(cts) == int(emb.shape[0]))
                out["audio_embeds_shape"] = list(emb.shape)
                out["audio_embeds_dtype"] = str(emb.dtype)
                out["audio_embeds_device"] = str(emb.device)
                out["counts"] = r.get("counts")
                out["audio_token_id"] = r.get("audio_token_id")
                out["out_hidden"] = r.get("out_hidden")
                out["encode_info"] = r.get("info")
                mat = _AUDIO_MAT.get(target, [])
                out["materialized_meta_count"] = len(mat)
                out["materialized_meta"] = [{"name": nm, "shape": s, "how": h}
                                            for nm, s, h in mat][:30]
                out["missing_weights"] = [nm for nm, s, h in mat if "MISSING_WEIGHT" in h]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1800:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/inspect_mm")          # #22: introspect a multimodal model's vision interface
    async def inspect_mm(model: str = "") -> JSONResponse:
        """Report the EXACT vision interface of a multimodal checkpoint (attribute path,
        module class, forward signature, processor + image token) so the distributed-Omni
        encoder path can be written against the real structure. Meta-load only (no weights,
        no inference) — safe + cheap. Used to build #22 increment 2 (the vision encoder)."""
        def _run():
            out: dict = {"model": model}
            try:
                friendly = resolve_model_name(model) if model else (
                    next(iter(engine.models)) if engine.models else "")
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                import inspect as _inspect
                from transformers import AutoConfig, AutoProcessor
                cfg = AutoConfig.from_pretrained(target)
                out["model_type"] = getattr(cfg, "model_type", None)
                out["architectures"] = getattr(cfg, "architectures", None)
                vc = getattr(cfg, "vision_config", None)
                out["vision_config_keys"] = sorted(vars(vc).keys()) if vc is not None else None
                try:
                    proc = AutoProcessor.from_pretrained(target)
                    out["processor_class"] = type(proc).__name__
                    for attr in ("image_token", "image_token_id", "image_processor",
                                 "video_token", "audio_token"):
                        v = getattr(proc, attr, None)
                        out[f"proc_{attr}"] = type(v).__name__ if attr == "image_processor" else v
                except Exception as exc:
                    out["processor_error"] = f"{type(exc).__name__}: {exc}"
                # meta-load the full multimodal model (zero memory) to inspect structure
                try:
                    import torch
                    from transformers import AutoModelForImageTextToText
                    with torch.device("meta"):
                        m = AutoModelForImageTextToText.from_config(cfg)
                    out["model_class"] = type(m).__name__
                    out["top_children"] = [n for n, _ in m.named_children()]
                    inner = getattr(m, "model", m)
                    out["inner_children"] = [n for n, _ in inner.named_children()]
                    vis = None
                    for path in ("visual", "vision_tower", "vision_model"):
                        vis = getattr(inner, path, None) or getattr(m, path, None)
                        if vis is not None:
                            out["vision_attr"] = path
                            break
                    if vis is not None:
                        out["vision_class"] = type(vis).__name__
                        with contextlib.suppress(Exception):
                            out["vision_forward_sig"] = str(_inspect.signature(vis.forward))
                        with contextlib.suppress(Exception):
                            out["vision_children"] = [n for n, _ in vis.named_children()]
                    # how does the model splice image features? find the method names
                    out["mm_methods"] = [n for n in dir(m)
                                         if any(k in n.lower() for k in
                                                ("image", "visual", "merge", "multimodal", "rope", "position"))][:40]
                except Exception as exc:
                    out["model_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/vision_log")          # crash-surviving phase log of the last vision encode
    async def vision_log(lines: int = 60) -> JSONResponse:
        """Return the tail of vision_diag.log. Because the encode has hard-crash-restarted
        the controller, this file (flushed+fsync'd per phase) is the only way to see which
        step ran last BEFORE a fatal native fault — read it AFTER the relaunch."""
        def _run():
            try:
                with open(_VISION_LOG, encoding="utf-8") as fh:
                    tail = fh.read().splitlines()[-max(1, min(lines, 500)):]
                return {"log": tail, "path": _VISION_LOG}
            except FileNotFoundError:
                return {"log": [], "note": "no vision_diag.log yet"}
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/vision_test")         # #22 inc 2: run the vision encoder end-to-end on a test image
    async def vision_test(model: str = "") -> JSONResponse:
        """Generate a small test image, run the processor + vision tower, and report shapes —
        so increment 2 (the encoder) is verified against the real model before wiring it into
        the pipeline. No text-model load, no inference on the LM."""
        def _run():
            out: dict = {"model": model}
            try:
                from PIL import Image
                friendly = resolve_model_name(model) if model else (
                    next(iter(engine.models)) if engine.models else "qwen3.6-35b-a3b")
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                img = Image.new("RGB", (336, 336), (90, 140, 200))
                r = _encode_images(target, [img])
                emb = r["image_embeds"]
                out["image_embeds_shape"] = list(emb.shape)
                out["image_embeds_dtype"] = str(emb.dtype)
                out["image_embeds_device"] = str(emb.device)
                out["grid_thw"] = (r["grid_thw"].tolist() if r["grid_thw"] is not None else None)
                out["encode_info"] = r.get("info")
                mat = _VISION_MAT.get(target, [])
                out["materialized_meta_count"] = len(mat)
                out["materialized_meta"] = [{"name": n, "shape": s, "how": h} for n, s, h in mat][:30]
                out["missing_weights"] = [n for n, s, h in mat if "MISSING_WEIGHT" in h]
                # --- merger diagnostics (STRUCTURE ONLY, no extra forward) ---
                # get_image_features gave the pre-merge ViT backbone [patches, vision_hidden];
                # the LM consumes MERGED tokens [prod(grid)/merge^2, text_hidden]. Report the
                # config + submodule tree (cheap, cached meta-load) to design the merge step.
                try:
                    # NOTE: must NOT bind the name `model` here — `model` is the endpoint
                    # parameter referenced at the top of _run ({"model": model}); assigning it
                    # anywhere in _run makes it function-local and UnboundLocalErrors line 1.
                    vmodel, dev = _load_vision_encoder(target)
                    vcfg = getattr(vmodel.config, "vision_config", None)
                    tcfg = getattr(vmodel.config, "text_config", vmodel.config)
                    def _g(o, *names):
                        for n in names:
                            v = getattr(o, n, None)
                            if v is not None:
                                return v
                        return None
                    sm = (_g(vcfg, "spatial_merge_size") or 1) if vcfg is not None else 1
                    out["vision_cfg"] = None if vcfg is None else {
                        "hidden_size": _g(vcfg, "hidden_size"),
                        "out_hidden_size": _g(vcfg, "out_hidden_size", "output_hidden_size"),
                        "spatial_merge_size": sm,
                    }
                    out["text_hidden_size"] = _g(tcfg, "hidden_size")
                    out["image_token_id"] = _g(vmodel.config, "image_token_id", "image_token_index")
                    g = r["grid_thw"]
                    if g is not None:
                        out["expected_merged_tokens"] = int(g.prod().item()) // (sm * sm)
                    vis = vmodel.model.visual
                    out["visual_children"] = [n for n, _ in vis.named_children()]
                    merger = getattr(vis, "merger", None)
                    if merger is not None:
                        out["merger_class"] = type(merger).__name__
                        out["merger_children"] = [n for n, _ in merger.named_children()]
                        fc1 = getattr(merger, "linear_fc1", None)
                        fc2 = getattr(merger, "linear_fc2", None)
                        if fc1 is not None:
                            out["merger_fc1"] = [getattr(fc1, "in_features", None),
                                                 getattr(fc1, "out_features", None)]
                        if fc2 is not None:
                            out["merger_fc2"] = [getattr(fc2, "in_features", None),
                                                 getattr(fc2, "out_features", None)]
                    # Re-run visual() and dump EVERY tensor field of the return + try the
                    # merger explicitly — to find where the merged [100,2048] actually is.
                    import torch
                    ipd = _get_image_processor(target)
                    inpd = ipd(images=[img], return_tensors="pt")
                    pvd = inpd["pixel_values"].to(dev)
                    gdd = inpd.get("image_grid_thw")
                    gdd = gdd.to(dev) if gdd is not None else None
                    with torch.inference_mode():
                        raw = vis(pvd, gdd)
                    fields = {}
                    if isinstance(raw, torch.Tensor):
                        fields["<tensor>"] = list(raw.shape)
                    elif hasattr(raw, "keys"):
                        for k in raw.keys():
                            v = raw[k]
                            if hasattr(v, "shape"):
                                fields[k] = list(v.shape)
                    out["visual_return_fields"] = fields
                    if merger is not None:
                        lhs = _as_feature_tensor(raw)
                        try:
                            with torch.inference_mode():
                                merged = merger(lhs)
                            out["merger_direct_shape"] = list(_as_feature_tensor(merged).shape)
                        except Exception as me:
                            out["merger_direct_error"] = f"{type(me).__name__}: {str(me)[:200]}"
                except Exception as exc:
                    import traceback
                    out["merger_diag_error"] = f"{type(exc).__name__}: {exc}"
                    out["merger_diag_trace"] = traceback.format_exc()[-600:]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-2000:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/mm_inject_test")      # #22 inc 3: prove embed-injection changes the computation
    async def mm_inject_test(model: str = "qwen2.5-0.5b", positions: str = "1,2") -> JSONResponse:
        """Load a (small) model, run a baseline prefill and an identical prefill with RANDOM
        embeds spliced in at `positions`, and report whether the next-token logits differ.
        Same token ids both times -> any change PROVES the mm frame reached stage 0, was
        stashed, and the splice replaced those positions' embeddings before the layers ran.
        Pure mechanism check (random embeds, not a real image) on the live pipeline."""
        out: dict = {"model": model}
        try:
            import torch
            friendly = resolve_model_name(model)
            out["friendly"] = friendly
            lm = await engine.ensure_loaded(friendly, 0)
            ids = _to_id_list(lm.tokenizer("The capital of France is"))
            pos = [int(p) for p in positions.split(",") if p.strip() != ""]
            pos = [p for p in pos if 0 <= p < len(ids)]
            out["prompt_len"] = len(ids)
            out["inject_positions"] = pos
            # hidden size from the model config (text hidden)
            from transformers import AutoConfig
            target = MODELS[friendly][0] if friendly in MODELS else friendly
            mcfg = await asyncio.to_thread(AutoConfig.from_pretrained,
                                           _local_model_dir(target) or target)
            tcfg = getattr(mcfg, "text_config", mcfg)
            hid = int(getattr(tcfg, "hidden_size", 0) or getattr(mcfg, "hidden_size"))
            out["hidden_size"] = hid
            xt = torch.tensor([ids])
            async with lm.lock:
                base = await engine._send(lm, xt, 0, True, False)
                emb = torch.randn(len(pos), hid, dtype=torch.bfloat16)
                inj = await engine._send(lm, xt, 0, True, False, mm=(pos, emb))
            ba = torch.as_tensor(base).float()
            ia = torch.as_tensor(inj).float()
            out["baseline_argmax"] = int(ba.argmax())
            out["injected_argmax"] = int(ia.argmax())
            out["logits_changed"] = bool(not torch.allclose(ba, ia))
            out["max_abs_delta"] = float((ba - ia).abs().max())
            out["ok"] = True
        except Exception as exc:
            import traceback
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["trace"] = traceback.format_exc()[-1500:]
        return JSONResponse(out)

    @app.get("/vision_prompt_test")  # #22 inc 3b: verify prompt-build (no heavy text-model load)
    async def vision_prompt_test(model: str = "qwen3.6-35b-a3b") -> JSONResponse:
        """End-to-end check of the IMAGE PROMPT construction without loading the text LM:
        build a test-image Anthropic message -> keep_images chat -> render (one <|image_pad|>)
        -> encode the image -> expand placeholders -> confirm positions align with embeds.
        Uses only the tokenizer + cached vision encoder (cheap)."""
        def _run():
            out: dict = {"model": model}
            try:
                import io, base64
                from PIL import Image
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                buf = io.BytesIO()
                Image.new("RGB", (336, 336), (200, 60, 60)).save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                messages = [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": "What color is this image?"}]}]
                images = _collect_images(messages)
                out["num_images"] = len(images)
                chat = _anthropic_messages_to_chat(None, messages, keep_images=True)
                enc = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
                ids = _to_id_list(enc)
                enc_res = _encode_images(target, images)
                embeds = enc_res.get("image_embeds")
                counts = enc_res.get("counts") or []
                itid = enc_res.get("image_token_id")
                out["image_token_id"] = itid
                out["counts"] = counts
                out["embeds_shape"] = list(embeds.shape) if embeds is not None else None
                out["raw_image_pad_in_ids"] = sum(1 for t in ids if t == itid)
                new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts)
                out["placeholders_found"] = found
                out["num_positions"] = len(positions)
                out["prompt_len_before"] = len(ids)
                out["prompt_len_after"] = len(new_ids)
                out["positions_head"] = positions[:5]
                out["aligned"] = bool(found == len(counts)
                                      and len(positions) == (embeds.shape[0] if embeds is not None else -1))
                # #22 inc 4: also compute the 3D mRoPE positions and report base + samples
                merge = int(enc_res.get("merge") or 1)
                grid_list = enc_res.get("grid_list") or []
                pos3d, base = _mrope_position_ids(new_ids, grid_list, int(itid), merge)
                ip0 = positions[0] if positions else 0
                out["mrope_base"] = base
                out["mrope_seq_len"] = len(pos3d[0])
                out["mrope_delta"] = base - len(new_ids)
                out["mrope_text_head_thw"] = [row[:3] for row in pos3d]
                out["mrope_image_thw"] = [row[ip0:ip0 + 3] for row in pos3d]
                out["mrope_tail_thw"] = [row[-2:] for row in pos3d]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/speech_capture_test")  # #P6: verify the hidden-states transport (Phase 1)
    async def speech_capture_test(model: str = "qwen2.5-omni-7b", max_new: int = 8) -> JSONResponse:
        """Phase-1 check of the distributed Thinker HIDDEN-STATE capture for speech-out: run a
        short text generation with capture_hidden=True and report the prefill hidden shape +
        each step's hidden shape + the decoded text. Requires the model loaded; no talker yet."""
        out: dict = {"model": model}
        try:
            friendly = resolve_model_name(model)
            if friendly not in engine.models:
                return JSONResponse({"ok": False, "error": f"{friendly} not loaded"},
                                    status_code=409)
            lm = engine.models[friendly]
            tok = lm.tokenizer
            sys_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                          "Group, capable of perceiving auditory and visual inputs, as well as "
                          "generating text and speech.")
            chat = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Say hello in one short sentence."}]
            ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                      tokenize=True))
            out["prompt_len"] = len(ids)
            gen_ids, prefill_hidden, step_hiddens, stop = await engine.capture_thinker(
                friendly, ids, int(max_new))
            out["prefill_hidden_shape"] = list(prefill_hidden.shape)
            out["prefill_hidden_dtype"] = str(prefill_hidden.dtype)
            out["num_step_hiddens"] = len(step_hiddens)
            out["step_hidden_shape"] = (list(step_hiddens[0].shape) if step_hiddens else None)
            out["num_gen_ids"] = len(gen_ids)
            out["stop"] = stop
            with contextlib.suppress(Exception):
                out["text"] = _safe_decode(tok, gen_ids)
            # the prefill hidden must cover every prompt token; step hiddens are 1 token each
            out["prefill_covers_prompt"] = bool(prefill_hidden.shape[1] == len(ids))
            out["ok"] = True
        except Exception as exc:
            import traceback
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["trace"] = traceback.format_exc()[-1500:]
        return JSONResponse(out)

    @app.get("/speech_components_test")  # #P6 Phase 2: load + report the talker + token2wav
    async def speech_components_test(model: str = "qwen2.5-omni-7b") -> JSONResponse:
        """Phase-2 check: meta-build the full Omni and materialize the talker + token2wav + the
        thinker embed matrix + spk_dict, then report dims / codec tokens / speakers / missing
        weights. No thinker load (it's distributed); no generation yet."""
        def _run():
            out: dict = {"model": model}
            try:
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                sc = _load_speech_components(target)
                talker, token2wav = sc["talker"], sc["token2wav"]
                tcfg = talker.config
                out["dev"] = sc["dev"]
                out["speakers"] = list(sc["speaker_map"].keys())
                out["n_talker_tensors"] = sc["n_talker"]
                out["n_token2wav_tensors"] = sc["n_token2wav"]
                out["n_embed_tensors"] = sc["n_embed"]
                out["talker_cfg"] = {
                    "num_hidden_layers": getattr(tcfg, "num_hidden_layers", None),
                    "hidden_size": getattr(tcfg, "hidden_size", None),
                    "embedding_size": getattr(tcfg, "embedding_size", None),
                    "vocab_size": getattr(tcfg, "vocab_size", None),
                }
                out["codec_tokens"] = {
                    "bos": talker.codec_bos_token, "eos": talker.codec_eos_token,
                    "pad": talker.codec_pad_token, "mask": talker.codec_mask_token,
                    "text_bos": talker.text_bos_token, "text_eos": talker.text_eos_token,
                    "text_pad": talker.text_pad_token,
                }
                out["token2wav_dtype"] = str(next(token2wav.parameters()).dtype)
                # a speaker entry's keys (cond / ref_mel / bos_token) — what token2wav needs
                spk0 = sc["speaker_map"].get(out["speakers"][0]) if out["speakers"] else None
                if isinstance(spk0, dict):
                    out["speaker_keys"] = list(spk0.keys())
                out["load_report"] = _SPEECH_MAT.get(target, {})
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1800:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/speech_test")         # #P6 Phase 3/4: end-to-end text -> speech (returns WAV b64)
    async def speech_test(model: str = "qwen2.5-omni-7b", speaker: str = "Chelsie",
                          max_new: int = 64, talker_max_new: int = 2048,
                          text: str = "Say, word for word: the quick brown fox jumps over the lazy dog.") -> JSONResponse:
        """Phase 3/4 end-to-end: distributed Thinker (hidden-state capture) -> Talker -> token2wav
        -> 24kHz waveform, returned as base64 PCM16 WAV. Requires the Omni model loaded."""
        out: dict = {"model": model, "speaker": speaker}
        try:
            friendly = resolve_model_name(model)
            if friendly not in engine.models:
                return JSONResponse({"ok": False, "error": f"{friendly} not loaded"},
                                    status_code=409)
            tok = engine.models[friendly].tokenizer
            sys_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                          "Group, capable of perceiving auditory and visual inputs, as well as "
                          "generating text and speech.")
            chat = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}]
            ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                      tokenize=True))
            gen_ids, stop, wav, info = await engine.generate_speech(
                friendly, ids, max_new=int(max_new), speaker=speaker,
                talker_max_new=int(talker_max_new))
            with contextlib.suppress(Exception):
                out["text"] = _safe_decode(tok, gen_ids)
            out["info"] = info
            # encode waveform -> PCM16 WAV -> base64
            def _wav_b64():
                import io, wave, base64
                import numpy as _np
                a = wav.detach().cpu().numpy()
                a = _np.clip(a, -1.0, 1.0)
                pcm = (a * 32767.0).astype(_np.int16)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                    w.writeframes(pcm.tobytes())
                return base64.b64encode(buf.getvalue()).decode()
            out["wav_b64"] = await asyncio.to_thread(_wav_b64)
            out["ok"] = True
        except Exception as exc:
            import traceback
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["trace"] = traceback.format_exc()[-2000:]
        return JSONResponse(out)

    @app.get("/audio_prompt_test")   # #22 inc 5c: verify the AUDIO prompt build (no text-model load)
    async def audio_prompt_test(model: str = "qwen2.5-omni-7b", secs: float = 2.0) -> JSONResponse:
        """End-to-end check of the AUDIO PROMPT construction without loading the text LM:
        synth a tone -> WAV -> input_audio message -> keep_audio chat -> render (one <|AUDIO|>)
        -> encode the audio -> expand the placeholder to its token count -> confirm positions
        align with embeds + the sequential TMRoPE covers the expanded prompt. Tokenizer +
        cached audio encoder only (cheap)."""
        def _run():
            out: dict = {"model": model}
            try:
                import io, base64, math, wave
                import numpy as _np
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                # synth a 16 kHz mono PCM16 WAV tone
                sr = 16000
                n = max(1, int(secs * sr))
                t = _np.arange(n, dtype=_np.float32) / sr
                pcm = (0.2 * _np.sin(2 * math.pi * 440.0 * t) * 32767).astype(_np.int16)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes(pcm.tobytes())
                b64 = base64.b64encode(buf.getvalue()).decode()
                messages = [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": "What do you hear?"}]}]
                auds = _collect_audio(messages)
                out["num_audio"] = len(auds)
                out["waveform_len"] = (int(len(auds[0])) if auds else 0)
                chat = _anthropic_messages_to_chat(None, messages, keep_audio=True)
                out["chat"] = chat
                try:
                    ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                              tokenize=True))
                    out["template"] = "ok"
                except Exception as exc:
                    out["template_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    # fall back: insert the audio markers as raw text so the rest still verifies
                    raw = "<|audio_bos|><|AUDIO|><|audio_eos|>What do you hear?"
                    ids = _to_id_list(tok(raw)["input_ids"] if hasattr(tok(raw), "get")
                                      else tok(raw))
                    out["template"] = "fallback_raw_text"
                enc_res = _encode_audio(target, auds)
                embeds = enc_res.get("audio_embeds")
                counts = enc_res.get("counts") or []
                atid = enc_res.get("audio_token_id")
                out["audio_token_id"] = atid
                out["counts"] = counts
                out["embeds_shape"] = list(embeds.shape) if embeds is not None else None
                out["raw_audio_tok_in_ids"] = (sum(1 for x in ids if x == atid) if atid is not None else None)
                new_ids, positions, found = _expand_image_placeholders(ids, int(atid), counts)
                out["placeholders_found"] = found
                out["num_positions"] = len(positions)
                out["prompt_len_before"] = len(ids)
                out["prompt_len_after"] = len(new_ids)
                out["positions_head"] = positions[:5]
                out["aligned"] = bool(found == len(counts) and embeds is not None
                                      and len(positions) == embeds.shape[0])
                pos3d, base = _audio_position_ids(len(new_ids))
                out["tmrope_base"] = base
                out["tmrope_seq_len"] = len(pos3d[0])
                out["tmrope_head"] = [row[:3] for row in pos3d]
                out["tmrope_tail"] = [row[-3:] for row in pos3d]
                # decode a window around the first audio position to confirm bos/eos framing
                if positions:
                    p0 = positions[0]
                    with contextlib.suppress(Exception):
                        out["decoded_around_audio"] = tok.decode(new_ids[max(0, p0 - 2):p0 + 2]
                                                                 + new_ids[p0 + len(positions):p0 + len(positions) + 2])
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/rope_probe")          # #22 inc 4: can we compute 3D mRoPE positions on the controller?
    async def rope_probe(model: str = "qwen3.6-35b-a3b") -> JSONResponse:
        """Confirm the mRoPE plan: (1) the model exposes get_rope_index so we can compute the
        correct 3D (t/h/w) position ids for an image prompt on the controller (index math, no
        weights), and (2) the TEXT config carries an mrope rope_scaling section so the worker's
        rotary expects 3D positions. Builds a test-image prompt and runs get_rope_index."""
        def _run():
            out: dict = {"model": model}
            try:
                import io, base64, torch
                from PIL import Image
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                buf = io.BytesIO(); Image.new("RGB", (336, 336), (60, 160, 90)).save(buf, "PNG")
                msgs = [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": "describe"}]}]
                images = _collect_images(msgs)
                chat = _anthropic_messages_to_chat(None, msgs, keep_images=True)
                ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
                enc_res = _encode_images(target, images)
                counts = enc_res.get("counts") or []
                itid = int(enc_res.get("image_token_id"))
                new_ids, positions, _ = _expand_image_placeholders(ids, itid, counts)
                grid = enc_res.get("grid_thw")
                model_obj, _dev = _load_vision_encoder(target)
                cfg = model_obj.config
                tcfg = getattr(cfg, "text_config", cfg)
                out["has_get_rope_index"] = hasattr(model_obj, "get_rope_index")
                out["text_rope_scaling"] = getattr(tcfg, "rope_scaling", None)
                out["rope_scaling_top"] = getattr(cfg, "rope_scaling", None)
                out["seq_len"] = len(new_ids)
                if out["has_get_rope_index"]:
                    with torch.inference_mode():
                        res = model_obj.get_rope_index(
                            torch.tensor([new_ids]), image_grid_thw=grid)
                    pos = res[0] if isinstance(res, (tuple, list)) else res
                    out["rope_index_return_type"] = type(res).__name__
                    out["position_ids_shape"] = list(pos.shape)
                    # show the 3 dims around the image region (positions[0..]) to eyeball mRoPE
                    p = pos[:, 0, :].tolist() if pos.dim() == 3 else pos.tolist()
                    img0 = positions[0] if positions else 0
                    out["pos_sample_text_head"] = [row[:4] for row in p]
                    out["pos_sample_image"] = [row[img0:img0 + 4] for row in p]
                    out["pos_sample_tail"] = [row[-3:] for row in p]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/tok_probe")           # #22 inc 3b: how does THIS tokenizer represent image tokens?
    async def tok_probe(model: str = "qwen3.6-35b-a3b") -> JSONResponse:
        """Cheap tokenizer-only probe (no model load) to decide how to build the prompt ids
        with image placeholders: len(tokenizer) vs image_token_id, whether the image_pad
        STRING round-trips through the text tokenizer, the vision_start/end ids, and whether
        the chat template renders an image content block."""
        def _run():
            out: dict = {"model": model}
            try:
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                out["len_tokenizer"] = len(tok)
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(_local_model_dir(target) or target)
                def g(o, *names):
                    for k in names:
                        v = getattr(o, k, None)
                        if v is not None:
                            return v
                    return None
                out["image_token_id"] = g(cfg, "image_token_id", "image_token_index")
                out["vision_start_token_id"] = g(cfg, "vision_start_token_id")
                out["vision_end_token_id"] = g(cfg, "vision_end_token_id")
                for s in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
                    try:
                        out[f"encode {s}"] = tok.encode(s, add_special_tokens=False)
                    except Exception as e:
                        out[f"encode {s} ERR"] = str(e)[:100]
                itid = out.get("image_token_id")
                if itid is not None:
                    with contextlib.suppress(Exception):
                        out["convert_ids_to_tokens(image_token_id)"] = tok.convert_ids_to_tokens(int(itid))
                try:
                    msgs = [{"role": "user", "content": [{"type": "image"},
                                                         {"type": "text", "text": "hi"}]}]
                    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                       tokenize=False)
                    out["template_has_image_pad"] = "<|image_pad|>" in rendered
                    out["chat_template_tail"] = rendered[-260:]
                except Exception as e:
                    out["chat_template_err"] = f"{type(e).__name__}: {str(e)[:150]}"
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-800:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/gpudiag")             # per-process GPU usage on the CONTROLLER host (this box)
    async def gpudiag() -> JSONResponse:
        """Run nvidia-smi locally and return which PROCESSES hold this GPU. The controller
        runs on a GPU node (beast), so this distinguishes InfiniteModel's own worker python
        from other tenants (Ollama, other inference) when the GPU looks unexpectedly full."""
        def _run():
            import subprocess
            out = {"host": platform.node()}
            try:
                g = subprocess.run(["nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                out["gpu"] = g.stdout.strip()
                p = subprocess.run(["nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                procs = []
                for line in p.stdout.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 3:
                        procs.append({"pid": parts[0], "name": parts[1],
                                      "used_mib": parts[2]})   # may be "[N/A]" on Windows/WDDM
                def _mib(x):
                    try:
                        return int(x["used_mib"])
                    except (ValueError, TypeError):
                        return -1                              # non-numeric ([N/A]) sorts last
                out["processes"] = sorted(procs, key=lambda x: -_mib(x))
                if procs and all(_mib(x) < 0 for x in procs):
                    out["note"] = ("per-process VRAM is [N/A] on Windows/WDDM — "
                                   "PIDs/names listed, but only the GPU total is exact")
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.post("/gc_cache")           # dashboard: reclaim disk from redundant HF-cache copies
    async def gc_cache() -> JSONResponse:
        r = await asyncio.to_thread(gc_redundant_cache)
        if r.get("removed"):
            log_activity(f"cache GC: freed {r['freed_gb']} GB "
                         f"({len(r['removed'])} redundant copies removed)")
        return JSONResponse(r)

    @app.post("/api/pull")           # Ollama-compat pull -> background download
    async def api_pull(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        r = await _start_download(friendly)
        return JSONResponse({"status": "success" if r.get("status") == "ready"
                             else "pulling manifest"})

    @app.delete("/api/delete")       # Ollama-compat delete
    async def api_delete(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        r = await _do_delete(friendly)
        return (JSONResponse({"status": "success"}) if r["ok"]
                else JSONResponse({"error": r["error"]}, status_code=409))

    @app.post("/api/generate")
    async def api_generate(req: Request):
        body = await req.json()
        return await _serve(body.get("model", ""), body.get("prompt", ""), None,
                            body, mode="generate", ip=_client_ip(req))

    @app.post("/api/chat")
    async def api_chat(req: Request):
        body = await req.json()
        return await _serve(body.get("model", ""), None, body.get("messages", []),
                            body, mode="chat", ip=_client_ip(req))

    @app.post("/v1/chat/completions")
    async def v1_chat(req: Request):
        body = await req.json()
        return await _serve(body.get("model", ""), None, body.get("messages", []),
                            body, mode="openai", ip=_client_ip(req))

    # ---- Anthropic Messages API (Claude Code backend) ----
    @app.post("/v1/messages")
    async def v1_messages(req: Request):
        body = await req.json()
        return await _serve_anthropic(body, ip=_client_ip(req))

    @app.post("/v1/messages/count_tokens")
    async def v1_count_tokens(req: Request):
        body = await req.json()
        return await _count_tokens_anthropic(body)

    # ---- OpenAI-compatible Text-To-Speech (distributed Qwen2.5-Omni speech-out) ----
    @app.post("/v1/audio/speech")
    async def v1_audio_speech(req: Request):
        """OpenAI /v1/audio/speech: {model, input, voice, response_format}. Speaks `input`
        through the distributed Omni speech pipeline and returns the raw audio bytes
        (wav | pcm). `voice` maps OpenAI names -> our speakers (Chelsie/Ethan)."""
        body = await req.json()
        ip = _client_ip(req)
        text = (body.get("input") or "").strip()
        if not text:
            return JSONResponse({"error": {"message": "'input' is required"}}, status_code=400)
        try:
            friendly = resolve_model_name(body.get("model", "") or "qwen2.5-omni-7b")
        except Exception as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
        fmt = (body.get("response_format") or "wav").lower()
        voice = body.get("voice") or "Chelsie"
        # admit (1 slot + queue) so concurrent TTS doesn't pile onto the CPU vocoder
        rec = _inflight_admit(ip, friendly)
        if rec is None:
            return JSONResponse({"error": {"message": "server busy (speech queue full)"}},
                                status_code=503)
        try:
            try:
                resident = engine.models.get(friendly)
                ctx = resident.ctx if resident else 0
                lm = await engine.ensure_loaded(friendly, ctx, auto_load=True)
                tok = lm.tokenizer
            except Exception as exc:
                return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
            # resolve voice -> our speaker (load speech components to know available speakers)
            try:
                sc = await asyncio.to_thread(_load_speech_components, lm.target_id)
                speaker = _resolve_speaker(voice, sc["speaker_map"])
            except Exception as exc:
                return JSONResponse({"error": {"message": f"speech components: {exc}"}},
                                    status_code=500)
            # prompt the Omni to SPEAK the input verbatim (pure-TTS use of a chat speech model)
            sys_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                          "Group, capable of perceiving auditory and visual inputs, as well as "
                          "generating text and speech.")
            chat = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Read the following text aloud exactly as "
                     f"written, and say nothing else:\n\n{text}"}]
            ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                      tokenize=True))
            # scale the text budget to the input length (verbatim ~ input length + margin)
            max_new = max(64, min(1024, int(len(_to_id_list(tok(text))) * 1.5) + 32))
            gen_ids, stop, wav, info = await engine.generate_speech(
                friendly, ids, max_new=max_new, speaker=speaker)
            print(f"[v1/audio/speech] '{text[:40]}...' voice={speaker} -> "
                  f"{info.get('wav_seconds')}s ({info.get('codec_tokens')} codes)")
            audio_bytes, media = await asyncio.to_thread(_encode_audio_response, wav, fmt)
            return Response(content=audio_bytes, media_type=media)
        except Exception as exc:
            import traceback
            print(f"[v1/audio/speech] error: {exc!r}\n{traceback.format_exc()[-1200:]}")
            return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}"}},
                                status_code=500)
        finally:
            _inflight_release(rec)

    return app


async def _prepare(model: str, prompt: Optional[str], messages, body: dict):
    """Resolve+load model, build prompt token ids, and pull sampling options."""
    friendly = resolve_model_name(model)
    # Respect the ctx the model was loaded with (e.g. via the dashboard) instead
    # of forcing DEFAULT_CTX — otherwise the first generate after a smaller-ctx
    # load silently triggers a slow full reload. Only fall back to DEFAULT_CTX
    # when this model isn't already loaded.
    resident = engine.models.get(friendly)
    ctx = resident.ctx if resident else 0   # 0 => auto-load at the model's native training context
    # CPU-only request (Ollama convention): options.num_gpu == 0 means "offload 0 layers to
    # GPU" => load to RAM only, never VRAM. Also accept an explicit options.cpu_only bool.
    # If the model is ALREADY loaded, ensure_loaded ignores this and serves the live copy.
    _o = body.get("options") or {}
    _ng = _o.get("num_gpu")
    cpu_only = bool(_o.get("cpu_only", False))
    with contextlib.suppress(TypeError, ValueError):
        cpu_only = cpu_only or (_ng is not None and int(_ng) == 0)
    try:
        lm = await engine.ensure_loaded(friendly, ctx, cpu_only=cpu_only, auto_load=True)
    except ValueError as exc:   # unknown model (auto-load only loads KNOWN registered models)
        return JSONResponse({"error": str(exc), "model": model}, status_code=404)
    # An ENCODER can't decode tokens — reject it here so it never hits the generate path.
    # _serve wraps this ValueError into a clear 400 (the tuple-unpack-and-400 path).
    if getattr(lm.spec, "is_embedding", False):
        raise ValueError(f"model '{friendly}' is an embedding model; use /api/embed")
    tok = lm.tokenizer
    if messages is not None:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    else:
        enc = tok(prompt or "")
    ids = _to_id_list(enc)
    opts = body.get("options") or {}
    temperature = float(opts.get("temperature", body.get("temperature", 0.0)))
    top_p = float(opts.get("top_p", body.get("top_p", 1.0)))
    max_new = int(opts.get("num_predict", body.get("max_tokens", 256)))
    stream = bool(body.get("stream", True))
    speculative = bool(opts.get("speculative", body.get("speculative", False)))
    spec_k = int(opts.get("spec_k", body.get("spec_k", 0)) or 0)   # per-request SPEC_K override (0=default)
    return friendly, tok, ids, temperature, top_p, max_new, stream, speculative, spec_k


def _ka_is_unload(v) -> bool:
    """Ollama keep_alive: 0 -> unload now; negative -> keep forever; positive -> keep N seconds.
    Returns True ONLY for a zero keep_alive (the client's 'unload' signal). Accepts int/float and
    duration strings ('0', '0s', '0m'). bool is rejected (a stray True/False isn't a duration)."""
    if v is None or isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v == 0
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(v))
    return bool(m) and float(m.group(1)) == 0.0


async def _serve(model: str, prompt: Optional[str], messages, body: dict, mode: str,
                 ip: str = "?"):
    METRICS["api_in"] += len(json.dumps(body))
    # CLIENT UNLOAD (Ollama keep_alive: 0): a CLIENT asking to expire/unload a model is IGNORED — we
    # keep models resident (ONLY the backend interface/dashboard /unload evicts). Reply with the
    # Ollama 'unload' ack so the client is satisfied, but DON'T touch the resident model. Only a PURE
    # unload (keep_alive 0 + no prompt/messages) short-circuits; a real generate with keep_alive:0
    # still generates (we simply never auto-unload after).
    if _ka_is_unload(body.get("keep_alive")) and not (prompt or "").strip() and not messages:
        # silently ignore client keep_alive:0 unloads — no activity-log line (just keep the model)
        if mode == "openai":
            return JSONResponse({"id": "chatcmpl-noop", "object": "chat.completion",
                                 "created": int(time.time()), "model": model,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": ""}}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
        out = {"model": model, "created_at": _iso(), "done": True, "done_reason": "unload",
               "total_duration": 0, "load_duration": 0, "prompt_eval_count": 0,
               "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0}
        out["message" if mode == "chat" else "response"] = "" if mode != "chat" else {"role": "assistant", "content": ""}
        return JSONResponse(out)
    # Resolve + admit BEFORE loading so a request waiting on a model (even while it
    # loads) shows in that model's queue. 1 slot + queue_depth waiters; else 503.
    try:
        friendly = resolve_model_name(model)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    rec = _inflight_admit(ip, friendly, engine.replica_count(friendly))  # K slots for K replicas
    if rec is None:
        return JSONResponse(
            {"error": f"queue full for '{friendly}': 1 slot + "
                      f"{ENGINE_CONFIG.get('queue_depth', DEFAULT_QUEUE_DEPTH)} queued — "
                      f"retry shortly"}, status_code=503)
    created = int(time.time())
    cmpl_id = "chatcmpl-" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:24]
    state = {"tokens": 0}  # token count (run() updates it; stream funcs read it)
    stream = bool(body.get("stream", True))   # decided from the body alone — no model load needed
    P: dict = {}             # prepared values, filled by _prepare (resident: instant; else load-on-request)
    _KEEPALIVE_LOAD_S = 8.0  # while a requested model auto-loads, emit a keepalive this often (Ollama-style)

    async def run():
        """Yield (text_piece, done_reason_or_None). Incremental detokenization:
        decode the cumulative id list and emit only the newly-completed suffix,
        holding back trailing bytes that don't yet form a whole character (so
        multi-byte UTF-8 — emoji/CJK — isn't corrupted by per-token decoding).
        Entered ONLY after the model is loaded + tokenized (values live in P), and
        it OWNS the inflight slot from here — its finally releases rec exactly once."""
        friendly = P["friendly"]; ids = P["ids"]; tok = P["tok"]
        max_new = P["max_new"]; temperature = P["temperature"]; top_p = P["top_p"]
        speculative = P["speculative"]; spec_k = P["spec_k"]
        produced: list[int] = []
        prev = ""
        try:
            async for tid, reason in engine.generate(friendly, ids, max_new, temperature,
                                                     top_p, speculative, rec=rec, spec_k=spec_k):
                if tid is not None:
                    produced.append(tid)
                    state["tokens"] = len(produced)
                    METRICS["tokens"] += 1
                    text = _safe_decode(tok, produced)
                    if text.endswith("�"):   # incomplete multi-byte char; wait
                        continue
                    piece, prev = text[len(prev):], text
                    if piece:
                        yield piece, None
                if reason:
                    text = _safe_decode(tok, produced)
                    yield text[len(prev):], reason   # flush remainder + signal done
        finally:
            _inflight_release(rec)   # free the slot/queue entry when generation ends

    def _map_finish(reason: str) -> str:
        return "length" if reason == "length" else "stop"

    async def _prep_unpack():
        """Auto-load (Ollama-style load-on-request) + tokenize, then fill P. Returns None on success,
        or an error message string on failure (unknown model / auto-load off / embedding model).
        Raises only on unexpected errors (caught by the caller)."""
        res = await _prepare(model, prompt, messages, body)
        if isinstance(res, JSONResponse):   # _prepare's error path (unknown model / auto-load off)
            msg = "model unavailable"
            with contextlib.suppress(Exception):
                msg = json.loads(bytes(res.body).decode()).get("error", msg)
            return msg
        (P["friendly"], P["tok"], P["ids"], P["temperature"], P["top_p"],
         P["max_new"], _st, P["speculative"], P["spec_k"]) = res
        return None

    # ---------- streaming ----------
    if stream:
        async def ollama_stream():
            t0 = time.perf_counter_ns()
            done_reason = "stop"
            err = None
            entered = False   # True once run() owns rec (its finally releases it then)
            body_key = "message" if mode == "chat" else "response"
            empty_val = {"role": "assistant", "content": ""} if mode == "chat" else ""
            try:
                # AUTO-LOAD WITH KEEPALIVE (Ollama-style load-on-request): a non-resident model loads
                # now; emit empty done:false chunks every _KEEPALIVE_LOAD_S so headers go out at once
                # and the client/proxy doesn't time out waiting on a slow load. Resident -> one poll.
                prep_task = asyncio.ensure_future(_prep_unpack())
                emsg = None
                while True:
                    try:
                        emsg = await asyncio.wait_for(asyncio.shield(prep_task), _KEEPALIVE_LOAD_S)
                        break
                    except asyncio.TimeoutError:
                        s = json.dumps({"model": model, "created_at": _iso(),
                                        body_key: empty_val, "done": False}) + "\n"
                        METRICS["api_out"] += len(s)
                        yield s
                if emsg is not None:   # load/prepare error
                    final = {"model": model, "created_at": _iso(), "done": True,
                             "done_reason": "error", "error": emsg}
                    final[body_key] = empty_val
                    s = json.dumps(final) + "\n"; METRICS["api_out"] += len(s); yield s
                    return
                entered = True
                async for piece, reason in run():   # run() owns + releases rec from here
                    if piece:
                        val = {"role": "assistant", "content": piece} if mode == "chat" else piece
                        s = json.dumps({"model": model, "created_at": _iso(),
                                        body_key: val, "done": False}) + "\n"
                        METRICS["api_out"] += len(s)
                        yield s
                    if reason:
                        done_reason = reason
            except Exception as exc:  # load or generation failed
                err, done_reason = str(exc), "error"
            finally:
                if not entered:
                    _inflight_release(rec)   # run() never took ownership -> release here
            dur = time.perf_counter_ns() - t0
            final = {"model": model, "created_at": _iso(), "done": True,
                     "done_reason": done_reason, "total_duration": dur, "load_duration": 0,
                     "prompt_eval_count": len(P.get("ids", [])), "prompt_eval_duration": 0,
                     "eval_count": state["tokens"], "eval_duration": dur}
            final[body_key] = empty_val
            if err:
                final["error"] = err
            s = json.dumps(final) + "\n"
            METRICS["api_out"] += len(s)
            yield s

        async def openai_stream():
            finish = "stop"
            entered = False
            try:
                prep_task = asyncio.ensure_future(_prep_unpack())
                emsg = None
                while True:
                    try:
                        emsg = await asyncio.wait_for(asyncio.shield(prep_task), _KEEPALIVE_LOAD_S)
                        break
                    except asyncio.TimeoutError:
                        yield ": loading\n\n"   # SSE comment keepalive (ignored by clients)
                if emsg is not None:
                    s = ("data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                         "created": created, "model": model, "error": {"message": emsg},
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                         + "\n\ndata: [DONE]\n\n")
                    METRICS["api_out"] += len(s); yield s
                    return
                entered = True
                async for piece, reason in run():
                    if piece:
                        s = "data: " + json.dumps({
                            "id": cmpl_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": piece},
                                         "finish_reason": None}]}) + "\n\n"
                        METRICS["api_out"] += len(s)
                        yield s
                    if reason:
                        finish = _map_finish(reason)
            except Exception:
                finish = "stop"
            finally:
                if not entered:
                    _inflight_release(rec)
            s = "data: " + json.dumps({
                "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
            s += "data: [DONE]\n\n"
            METRICS["api_out"] += len(s)
            yield s

        if mode == "openai":
            return StreamingResponse(openai_stream(), media_type="text/event-stream")
        return StreamingResponse(ollama_stream(), media_type="application/x-ndjson")

    # ---------- non-streaming ----------
    # Non-stream inherently blocks until done (no keepalive possible) — load on request, then collect.
    try:
        emsg = await _prep_unpack()
    except Exception as exc:
        _inflight_release(rec)
        return JSONResponse({"error": str(exc), "model": model}, status_code=400)
    if emsg is not None:   # unknown model / auto-load off / embedding model
        _inflight_release(rec)
        return JSONResponse({"error": emsg, "model": model}, status_code=404)
    t0 = time.perf_counter_ns()
    text = ""
    done_reason = "stop"
    try:
        async for piece, reason in run():
            text += piece
            if reason:
                done_reason = reason
    except Exception as exc:
        import traceback as _tb
        # Surface the cause: a TP forward error (broken all-reduce mesh, shape/quant bug) often has
        # an EMPTY str(exc) (e.g. a dropped peer socket), so {"error": str(exc)} returned "" with no
        # hint. Log repr + traceback to the activity feed (and console) and return the type name.
        log_activity(f"generate {model}: FAILED — {exc!r}")
        print(f"[generate] {model} FAILED: {exc!r}\n{_tb.format_exc()}", flush=True)
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "model": model},
                            status_code=500)
    dur = time.perf_counter_ns() - t0
    n = state["tokens"]

    if mode == "openai":
        payload = {
            "id": cmpl_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": _map_finish(done_reason)}],
            "usage": {"prompt_tokens": len(P["ids"]), "completion_tokens": n,
                      "total_tokens": len(P["ids"]) + n}}
        METRICS["api_out"] += len(json.dumps(payload))
        return JSONResponse(payload)
    out = {"model": model, "created_at": _iso(), "done": True, "done_reason": done_reason,
           "total_duration": dur, "load_duration": 0, "prompt_eval_count": len(P["ids"]),
           "prompt_eval_duration": 0, "eval_count": n, "eval_duration": dur}
    if mode == "chat":
        out["message"] = {"role": "assistant", "content": text}
    else:
        out["response"] = text
    METRICS["api_out"] += len(json.dumps(out))
    return JSONResponse(out)


async def _serve_anthropic(body: dict, ip: str = "?"):
    """POST /v1/messages — the Anthropic Messages API, so Claude Code (and any
    Anthropic SDK client) can drive the distributed fleet. Translates the Anthropic
    request into the model's chat template (tools included), runs the same decode
    path, and renders either a single JSON message or the Anthropic SSE event
    stream. Qwen <tool_call>{...}</tool_call> output is mapped to tool_use blocks."""
    METRICS["api_in"] += len(json.dumps(body))
    model = body.get("model", "")
    try:
        friendly = resolve_model_name(model)
    except Exception as exc:
        return JSONResponse({"type": "error",
                             "error": {"type": "not_found_error", "message": str(exc)}},
                            status_code=404)
    rec = _inflight_admit(ip, friendly, engine.replica_count(friendly))  # K slots + queue; else 503
    if rec is None:
        return JSONResponse({"type": "error", "error": {"type": "overloaded_error",
            "message": f"queue full for '{friendly}': 1 slot + "
                       f"{ENGINE_CONFIG.get('queue_depth', DEFAULT_QUEUE_DEPTH)} queued"}},
            status_code=503)
    try:
        resident = engine.models.get(friendly)
        ctx = resident.ctx if resident else 0
        lm = await engine.ensure_loaded(friendly, ctx, auto_load=True)
        if getattr(lm.spec, "is_embedding", False):   # encoder can't decode -> reject
            raise ValueError(f"model '{friendly}' is an embedding model; use /api/embed")
        tok = lm.tokenizer
    except Exception as exc:
        _inflight_release(rec)
        # Claude Code reads this on model-selection: surface a clean Anthropic error.
        return JSONResponse({"type": "error",
                             "error": {"type": "not_found_error", "message": str(exc)}},
                            status_code=404)

    # #22 inc 3b/5c: pull any images + audio out so the chat template renders the per-item
    # placeholder (keep_images/keep_audio), then expand + splice their embeds below. Decode/
    # fetch runs OFF the event loop — _decode_image/_decode_audio may blocking-urlopen.
    images = await asyncio.to_thread(_collect_images, body.get("messages"))
    audios = await asyncio.to_thread(_collect_audio, body.get("messages"))
    target_id = MODELS[friendly][0] if friendly in MODELS else friendly
    mm = None
    mrope = None   # #22 inc 4/5c: (3D position_ids [3][q], base) when media embeds are spliced
    hf_tools = _anthropic_tools_to_hf(body.get("tools"))

    def _render_ids(chat):
        """Tokenize a chat with the tools-aware fallback (template throws on tools= for many
        multimodal-remapped checkpoints -> re-render without native tools + a text tool
        instruction; last-ditch: flatten). Shared by the vision and text-only renders."""
        try:
            if hf_tools:
                return _to_id_list(tok.apply_chat_template(chat, tools=hf_tools,
                                   add_generation_prompt=True, tokenize=True))
            return _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                               tokenize=True))
        except Exception as exc:
            print(f"[v1/messages] chat-template failed ({type(exc).__name__}: {exc}); "
                  f"re-rendering without native tools" + (" + tool instruction" if hf_tools else ""))
            chat2 = chat
            if hf_tools:
                instr = _tool_instruction(hf_tools)
                if chat and chat[0].get("role") == "system":
                    chat2 = [{"role": "system", "content": chat[0].get("content", "") + "\n\n" + instr}] + chat[1:]
                else:
                    chat2 = [{"role": "system", "content": instr}] + chat
            try:
                return _to_id_list(tok.apply_chat_template(chat2, add_generation_prompt=True,
                                   tokenize=True))
            except Exception:
                flat = "\n\n".join(f"{m['role']}: {_anth_flatten(m.get('content', ''))}"
                                   for m in chat2)
                return _to_id_list(tok(flat + "\n\nassistant:"))

    # Modality priority: audio-only OR vision-only (Omni's supported single-modality cases).
    # If both are present, prefer AUDIO and drop images (mixed audio+vision in one prompt =
    # a future increment; the single mm pair carries one embed set).
    do_audio = bool(audios)
    do_vision = bool(images) and not do_audio
    if images and do_audio:
        print(f"[v1/messages] both audio + image present -> audio path; {len(images)} "
              f"image(s) dropped (mixed AV not yet supported)")
    ids = _render_ids(_anthropic_messages_to_chat(body.get("system"), body.get("messages"),
                                                  keep_images=do_vision, keep_audio=do_audio))
    # #22 inc 5c: AUDIO path (Qwen2.5-Omni). Encode clip(s), expand each single <|AUDIO|> into
    # its token-count run, splice the embeds, and use sequential TMRoPE positions.
    if do_audio:
        try:
            enc_res = await asyncio.to_thread(_encode_audio, target_id, audios)
            embeds = enc_res.get("audio_embeds")
            counts = enc_res.get("counts") or []
            atid = enc_res.get("audio_token_id")
            n_emb = int(embeds.shape[0]) if embeds is not None else 0
            if atid is not None and n_emb and sum(counts) == n_emb:
                new_ids, positions, found = _expand_image_placeholders(ids, int(atid), counts)
                if found == len(counts) and len(positions) == n_emb:
                    ids, mm = new_ids, (positions, embeds)
                    # audio-only TMRoPE = sequential 0..seq-1 on all 3 dims (see
                    # _audio_position_ids); positions grow normally, unlike images.
                    mrope = _audio_position_ids(len(ids))
                    print(f"[v1/messages] audio: {len(audios)} clip(s) -> {len(positions)} "
                          f"audio tokens spliced (counts={counts}); TMRoPE base={mrope[1]}")
                else:
                    print(f"[v1/messages] audio MISMATCH: found {found} placeholder(s) "
                          f"(expected {len(counts)}), {len(positions)} positions vs {n_emb} "
                          f"embeds — text-only")
            else:
                print(f"[v1/messages] audio skip: audio_token_id={atid}, counts_sum="
                      f"{sum(counts)}, embeds={n_emb} — text-only")
        except Exception as exc:
            print(f"[v1/messages] audio encode failed ({type(exc).__name__}: {exc}); text-only")
        if mm is None:
            ids = _render_ids(_anthropic_messages_to_chat(body.get("system"),
                              body.get("messages"), keep_images=False, keep_audio=False))
            print("[v1/messages] audio unavailable -> rebuilt text-only prompt "
                  f"({len(ids)} tokens)")
    # #22 inc 3b: VISION path (only when no audio splice happened). Encode the image(s),
    # expand each single <|image_pad|> into its grid-derived run, stage embeds for splicing.
    if do_vision and mm is None:
        try:
            enc_res = await asyncio.to_thread(_encode_images, target_id, images)
            embeds = enc_res.get("image_embeds")
            counts = enc_res.get("counts") or []
            itid = enc_res.get("image_token_id")
            n_emb = int(embeds.shape[0]) if embeds is not None else 0
            if itid is not None and n_emb and sum(counts) == n_emb:
                new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts)
                if found == len(counts) and len(positions) == n_emb:
                    ids, mm = new_ids, (positions, embeds)
                    # #22 inc 4: 3D mRoPE positions for the expanded prompt (image tokens get
                    # t/h/w grid positions; the counter advances slowly past each image).
                    merge = int(enc_res.get("merge") or 1)
                    grid_list = enc_res.get("grid_list") or []
                    mrope = _mrope_position_ids(ids, grid_list, int(itid), merge)
                    print(f"[v1/messages] vision: {len(images)} image(s) -> {len(positions)} "
                          f"image tokens spliced (counts={counts}); mRoPE base={mrope[1]}")
                else:
                    print(f"[v1/messages] vision MISMATCH: found {found} placeholder(s) "
                          f"(expected {len(counts)}), {len(positions)} positions vs {n_emb} "
                          f"embeds — text-only")
            else:
                print(f"[v1/messages] vision skip: image_token_id={itid}, counts_sum="
                      f"{sum(counts)}, embeds={n_emb} — text-only")
        except Exception as exc:
            print(f"[v1/messages] vision encode failed ({type(exc).__name__}: {exc}); text-only")
        # On ANY vision failure/mismatch, REBUILD a genuinely text-only prompt so no raw
        # <|image_pad|> placeholders leak into the prefill (they'd embed as bare placeholder
        # tokens and degrade output). Only the success branch above keeps the expanded ids.
        if mm is None:
            ids = _render_ids(_anthropic_messages_to_chat(body.get("system"),
                              body.get("messages"), keep_images=False))
            print("[v1/messages] vision unavailable -> rebuilt text-only prompt "
                  f"({len(ids)} tokens)")
    # Reasoning models (Qwen3) whose template OPENS <think> in the prompt make the model
    # begin generation already mid-thought (output starts with reasoning, then </think>).
    # Detect it from the prompt tail so streaming can hold that reasoning back.
    starts_in_think = False
    with contextlib.suppress(Exception):
        tail = tok.decode(ids[-24:])
        starts_in_think = "<think>" in tail and "</think>" not in tail.split("<think>")[-1]

    max_new = int(body.get("max_tokens", 512) or 512)
    temperature = float(body.get("temperature", 0.0) or 0.0)
    top_p = float(body.get("top_p", 1.0) or 1.0)
    stream = bool(body.get("stream", False))
    state = {"tokens": 0}
    msg_id = _anth_id("msg")

    async def gen_raw():
        """Yield (text_piece, done_reason_or_None) — incremental, multibyte-safe,
        WITH the literal <tool_call>/<think> markup preserved for downstream parsing."""
        produced: list[int] = []
        prev = ""
        try:
          async for tid, reason in engine.generate(friendly, ids, max_new,
                                                   temperature, top_p, False, rec=rec, mm=mm,
                                                   mrope=mrope):
            if tid is not None:
                produced.append(tid)
                state["tokens"] = len(produced)
                METRICS["tokens"] += 1
                text = _safe_decode(tok, produced)
                if text.endswith("�"):
                    continue
                piece, prev = text[len(prev):], text
                if piece:
                    yield piece, None
            if reason:
                text = _safe_decode(tok, produced)
                yield text[len(prev):], reason
        finally:
            _inflight_release(rec)   # free the slot/queue entry when generation ends

    # ---------- streaming (Anthropic SSE) ----------
    if stream:
        async def sse():
            def ev(name: str, payload: dict) -> str:
                return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

            s = ev("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": len(ids), "output_tokens": 0}}})
            METRICS["api_out"] += len(s)
            yield s
            yield ev("ping", {"type": "ping"})

            raw = ""
            emitted_plain = emitted_tools = next_index = text_index = 0
            text_open = False
            finish = "stop"
            try:
                async for piece, reason in gen_raw():
                    if piece:
                        raw += piece
                        plain, tools = _segment_tools(raw, starts_in_think)
                        if len(plain) <= emitted_plain and len(tools) <= emitted_tools:
                            # tokens flowing but all held back (inside <think> or a
                            # partial tool call) — ping so the client doesn't time out.
                            yield ev("ping", {"type": "ping"})
                        if len(plain) > emitted_plain:
                            if not text_open:
                                text_index = next_index
                                next_index += 1
                                yield ev("content_block_start", {
                                    "type": "content_block_start", "index": text_index,
                                    "content_block": {"type": "text", "text": ""}})
                                text_open = True
                            delta = plain[emitted_plain:]
                            emitted_plain = len(plain)
                            yield ev("content_block_delta", {
                                "type": "content_block_delta", "index": text_index,
                                "delta": {"type": "text_delta", "text": delta}})
                        while emitted_tools < len(tools):
                            if text_open:
                                yield ev("content_block_stop", {
                                    "type": "content_block_stop", "index": text_index})
                                text_open = False
                            blk = _tool_to_block(tools[emitted_tools])
                            emitted_tools += 1
                            idx = next_index
                            next_index += 1
                            yield ev("content_block_start", {
                                "type": "content_block_start", "index": idx,
                                "content_block": {"type": "tool_use", "id": blk["id"],
                                                  "name": blk["name"], "input": {}}})
                            yield ev("content_block_delta", {
                                "type": "content_block_delta", "index": idx,
                                "delta": {"type": "input_json_delta",
                                          "partial_json": json.dumps(blk["input"])}})
                            yield ev("content_block_stop", {
                                "type": "content_block_stop", "index": idx})
                    if reason:
                        finish = reason
            except Exception as exc:
                # mid-stream failure: close any open block, then signal end
                if text_open:
                    yield ev("content_block_stop", {"type": "content_block_stop",
                                                    "index": text_index})
                    text_open = False
                yield ev("error", {"type": "error",
                                   "error": {"type": "api_error", "message": str(exc)}})
                return
            if text_open:
                yield ev("content_block_stop", {"type": "content_block_stop",
                                                "index": text_index})
            stop_reason = ("tool_use" if emitted_tools else
                           ("max_tokens" if finish == "length" else "end_turn"))
            yield ev("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": state["tokens"]}})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ---------- non-streaming ----------
    full = ""
    finish = "stop"
    try:
        async for piece, reason in gen_raw():
            full += piece
            if reason:
                finish = reason
    except Exception as exc:
        return JSONResponse({"type": "error",
                             "error": {"type": "api_error", "message": str(exc)}},
                            status_code=500)
    clean, raw_tools = _extract_tools(full)
    content = []
    if clean:
        content.append({"type": "text", "text": clean})
    for tb in raw_tools:
        content.append(_tool_to_block(tb))
    if not content:
        content.append({"type": "text", "text": ""})
    stop_reason = ("tool_use" if raw_tools else
                   ("max_tokens" if finish == "length" else "end_turn"))
    payload = {"id": msg_id, "type": "message", "role": "assistant", "model": model,
               "content": content, "stop_reason": stop_reason, "stop_sequence": None,
               "usage": {"input_tokens": len(ids), "output_tokens": state["tokens"]}}
    METRICS["api_out"] += len(json.dumps(payload))
    return JSONResponse(payload)


async def _count_tokens_anthropic(body: dict):
    """POST /v1/messages/count_tokens — Claude Code uses this for context budgeting.
    Exact via the resident tokenizer; a char/4 estimate if the model isn't loaded
    (so we never trigger a slow distributed load just to count)."""
    model = body.get("model", "")
    try:
        friendly = resolve_model_name(model)
    except ValueError as exc:
        return JSONResponse({"type": "error",
                             "error": {"type": "not_found_error", "message": str(exc)}},
                            status_code=404)
    chat = _anthropic_messages_to_chat(body.get("system"), body.get("messages"))
    hf_tools = _anthropic_tools_to_hf(body.get("tools"))
    n = None
    resident = engine.models.get(friendly)
    if resident is not None:
        tok = resident.tokenizer
        with contextlib.suppress(Exception):
            if hf_tools:
                enc = tok.apply_chat_template(chat, tools=hf_tools,
                                              add_generation_prompt=True, tokenize=True)
            else:
                enc = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
            n = len(_to_id_list(enc))
    if n is None:
        n = _estimate_tokens(chat)
    return JSONResponse({"input_tokens": n})


def _display_host() -> str:
    return platform.node() if ARGS.host in ("0.0.0.0", "") else ARGS.host


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

# Dashboard + bandwidth HTML live in dashboard_html.py (server-only); kept in sync by the
# multi-file self-update (in EXTRA_UPDATE_FILES) + present from a fresh git clone -> plain import.
from dashboard_html import DASHBOARD_HTML, BANDWIDTH_HTML   # noqa: F401


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
