#!/usr/bin/env python3
"""
InfiniteModel — worker client.

Connects to the controller, reports device/memory/disk, sends heartbeats, and (on
command) loads a slice of a model and serves it as one stage of a distributed
inference pipeline.

  M1  — capability probe + registry + heartbeat
  M2b — partial model load (owns only its layers)
  M2c — networked stage execution (prefill; KV-cache decode is later)
  M2d — CHUNK SERVING: the controller streams each worker only its layer tensors
        over HTTP; the worker loads them straight into RAM. Workers keep NO model
        on disk, so the smallest disk no longer caps model size. On startup the
        worker also purges stale model/chunk caches to free space.

torch/transformers/safetensors are imported lazily, so an M1-only worker needs
only psutil.

Run:
    python client.py --controller <controller-host>   # default: config.json controller_host
    python client.py --self-test-load --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import random
import shutil
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import deque

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dep. Install with:\n    pip install psutil\n"
        f"(import error: {exc})"
    )

VERSION = "0.2-m4c197"  # version tag only; full changelog -> CHANGELOG.md
# #stage0-stale-reconnect: if this worker hasn't forwarded a frame to a model's NEXT hop for this
# long, the (idle) next-hop socket may have gone silently half-open -> drop it at the next PREFILL
# (reset=True) so _send_next lazy-reconnects FRESH. Only checked at prefill, never per decode token,
# so a slow model's multi-second inter-token gaps never trip it. Mirrors the controller's STAGE0_STALE_S.
STAGE_STALE_S = 5.0
# #tp-mesh-keepalive: rank 0 pings the TP peers if no forward has crossed the mesh for this long, so
# the lockstep all-reduce sockets never sit idle long enough to go silently half-open (the failure
# that surfaced as "tp all-reduce failed — peer rank stalled" after an idle gap between requests).
# Well under the observed idle-death window (a fresh mesh died within ~24s idle); only fires when the
# model is idle (a busy model keeps its own mesh warm via real forwards).
TP_KEEPALIVE_S = 6.0
_STREAM_PREFETCH_MAX = 6  # max concurrent per-layer weight fetches during a streaming load
                          # (actual depth K is clamped to free RAM per node; see Shard.from_stream)
GB = 1024 ** 3
HOME = os.path.expanduser("~")
CHUNK_DIR = os.path.join(HOME, "infinitemodel", "chunks")

# cumulative network bytes this worker has moved (data-plane frames + weight
# downloads). The heartbeat turns these into a 10 s rolling in/out rate so the
# dashboard can show per-client traffic. Inter-stage chain traffic is invisible
# to the controller, so the worker must measure it itself.
NET = {"in": 0, "out": 0}


# Every worker console line is date/time-stamped (matches the controller) so events
# can be correlated after the fact — on Windows consoles and in journald alike.
# Shadows the builtin print for THIS module only; all [+]/[!]/[load]/[data] lines pick
# it up automatically.
import builtins as _builtins
def print(*args, **kwargs):  # noqa: A001 — intentional builtin shadow for timestamping
    _builtins.print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)


# code-split Inc 7: memory/GC helpers (_release_vram/_release_ram/_flush_os_cache/
# mem_maintenance_loop) + capability probes (detect_device/_gpu_mem_gb/_rocm_gpu_util/
# _using_gpu/free_disk_gb) live in worker_hw.py now (VERBATIM; back-imported below).

# ---------------------------------------------------------------------------
# Network route selection — ride the fastest LAN NIC (USB 2.5GbE > 1GbE > Wi-Fi)
#
# The worker sends NO explicit IP in its registration: the controller records the
# *source IP of this worker's control connection* as the worker's address, and
# every heavy data-plane frame (activations between pipeline stages, weight loads)
# is sent to <that IP>:data_port. So whichever NIC carries the control connection
# becomes the heavy-traffic path. We therefore pick the best LAN interface and bind
# our outbound sockets' source to it — preferring wired over Wi-Fi and the fastest
# wired link (a USB 2.5GbE dongle before the built-in 1GbE). This is why plugging in
# a faster NIC needs the worker cold-restarted (a live TCP conn is pinned to its
# original source IP); a self-update relaunch re-runs this selection automatically.
# ---------------------------------------------------------------------------

_ROUTE_SRC = ""   # chosen local source IP for all LAN traffic ("" = let the OS pick)


def _local_addr():
    """local_addr tuple for asyncio.open_connection (None = OS-default route)."""
    return (_ROUTE_SRC, 0) if _ROUTE_SRC else None


# code-split Inc 7: the READ-ONLY route detectors (_os_default_src/_iface_kind/select_route/
# _controller_is_local/_fmt_route) + RAM detection live in worker_hw.py now (VERBATIM;
# back-imported). _ROUTE_SRC + _local_addr above STAY here -- live rebind pair.

# code-split Inc 7: build_registration lives in worker_hw.py now (VERBATIM; back-imported).

def _enc(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


# code-split Inc 7: _dir_size + cleanup_storage live in worker_hw.py now (VERBATIM).

# ---------------------------------------------------------------------------
# Tensor selection (shared shape of what a stage owns) + framing
# ---------------------------------------------------------------------------

def _selected_names(all_names, start: int, end: int, has_embed: bool,
                    has_head: bool, tied: bool) -> list[str]:
    want: list[str] = []
    if has_embed:
        want.append("model.embed_tokens.weight")
    for i in range(start, end):
        want += [n for n in all_names if n.startswith(f"model.layers.{i}.")]
    if has_head:
        want.append("model.norm.weight")
        want.append("model.embed_tokens.weight" if tied else "lm_head.weight")
    return list(dict.fromkeys(want))


# Tensor (un)packing lives in wire.py (shared with server.py); kept in sync on every node by
# the multi-file self-update (wire.py is in EXTRA_UPDATE_FILES) and present from a fresh git
# clone, so a plain import is safe.
from wire import (_pack_tensor, _unpack_tensor, _set_keepalive, _tp_hetsplit,   # noqa: F401
                  install_log_tee, drain_new_logs, _fuse_moe_experts,
                  load_config, repo_raw_url, discover_controller)

# #wire-caps + #ntensor-manifest: manifest-frame helpers + capability constants (canonical home:
# wire.py, shared with the controller). GUARDED so a per-file self-update convergence window
# (this client.py newer than wire.py) still BOOTS the worker: with a stale wire.py the worker
# advertises no caps (build_registration reads wire.WIRE_CAPS off the module -> absent) and
# _pack_ntensor stays None, so worker_net's manifest branch is dead and every return frame keeps
# the legacy byte-identical format until the next self-update completes wire.py too.
try:
    from wire import (WIRE_CAPS, NT_LOGITS, NT_HIDDEN, NT_TOKEN_IDS,   # noqa: F401
                      NT_TOPK_VALS, NT_TOPK_IDX, _pack_ntensor, _unpack_ntensor)
except ImportError:   # stale wire.py — run capability-less until it converges
    WIRE_CAPS, _pack_ntensor, _unpack_ntensor = (), None, None
    NT_LOGITS, NT_HIDDEN, NT_TOKEN_IDS, NT_TOPK_VALS, NT_TOPK_IDX = 0, 1, 2, 3, 4


async def _read_frame(reader: asyncio.StreamReader) -> tuple[dict, bytes, int]:
    hdr_len = int.from_bytes(await reader.readexactly(4), "big")
    hb = await reader.readexactly(hdr_len)
    hdr = json.loads(hb.decode())
    raw = await reader.readexactly(hdr["nbytes"]) if hdr["nbytes"] else b""
    nb = 4 + hdr_len + len(raw)
    NET["in"] += nb
    return hdr, raw, nb


async def _write_frame(writer: asyncio.StreamWriter, hdr: dict, raw: bytes) -> int:
    hdr = {**hdr, "nbytes": len(raw)}
    hb = json.dumps(hdr).encode()
    writer.write(len(hb).to_bytes(4, "big") + hb + raw)
    await writer.drain()
    nb = 4 + len(hb) + len(raw)
    NET["out"] += nb
    return nb


# Per-PEER data-plane byte counters (cumulative): peer label -> {"in","out"}. Peer is
# "controller" or another node's IP. The controller meters only its OWN sockets (1st/last
# hop), so node<->node hidden-state traffic is invisible to it during decode — these
# worker-side counters fill that gap for the bandwidth page (heartbeat-reported).
NET_PEERS: dict = {}


def _net_peer(peer: str, *, rx: int = 0, tx: int = 0) -> None:
    # One call == one data-plane frame ("packet") in/out, so bump the frame counter
    # alongside the byte counter — the bandwidth page derives packets/s from these.
    c = NET_PEERS.get(peer)
    if c is None:
        c = NET_PEERS[peer] = {"in": 0, "out": 0, "in_pkts": 0, "out_pkts": 0}
    if rx:
        c["in"] += rx
        c["in_pkts"] = c.get("in_pkts", 0) + 1
    if tx:
        c["out"] += tx
        c["out_pkts"] = c.get("out_pkts", 0) + 1


def _http_get(url: str, timeout: float = 7200) -> bytes:   # #100: a huge shard slice on a slow drive
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    NET["in"] += len(data)
    return data


def _http_post(url: str, data: bytes, headers: dict | None = None, timeout: float = 7200) -> bytes:
    """POST a binary body (e.g. a remotely-packed shard-cache unit -> controller /pack_result,
    #distributed-packing). Counts the upload as outbound traffic."""
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = r.read()
    NET["out"] += len(data)
    return out


def _http_get_to_file(url: str, path: str, timeout: float = 7200) -> int:   # #100: match _http_get ceiling
    """Stream a response to disk in chunks (never holds the whole body in RAM).
    Used for weight slices so a big shard doesn't spike RAM to ~2x at load time."""
    total = 0
    with urllib.request.urlopen(url, timeout=timeout) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    NET["in"] += total
    return total


# ---------------------------------------------------------------------------
# Shard — owns layers [start, end); built from a controller-served weight blob
# (M2d) or directly from HF (self-test). Runs forward passes over its layers.
# ---------------------------------------------------------------------------

def _tp_shard_model_(model, rank: int, tp_size: int, cfg) -> None:
    """Tensor-parallel (M4): shard EVERY decoder layer IN PLACE for `rank` of `tp_size` —
    column-parallel q/k/v/gate/up (keep this rank's slice of the OUTPUT features), row-parallel
    o/down (keep this rank's slice of the INPUT features), attention head counts scaled to the
    rank. Embeddings, final norm and lm_head stay replicated (full) on every rank. The
    row-parallel o_proj/down_proj produce PARTIAL outputs the caller must all-reduce (sum) across
    ranks. Validated (theocomp, transformers 5.x): rank partials recombine to the exact full-layer
    output, GQA included. Requires tp_size | num_key_value_heads."""
    import copy
    import torch
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // nh)
    inter = cfg.intermediate_size
    qd, kvd, idim = nh * hd // tp_size, nkv * hd // tp_size, inter // tp_size

    def col(lin, per):   # column-parallel: this rank's slice of the OUTPUT rows
        w = lin.weight.data[rank * per:(rank + 1) * per].clone()
        nl = torch.nn.Linear(lin.in_features, per, bias=lin.bias is not None,
                             dtype=w.dtype, device=w.device)
        nl.weight = torch.nn.Parameter(w, requires_grad=False)
        if lin.bias is not None:
            nl.bias = torch.nn.Parameter(
                lin.bias.data[rank * per:(rank + 1) * per].clone(), requires_grad=False)
        return nl

    def row(lin, per):   # row-parallel: this rank's slice of the INPUT cols (no bias; add once)
        w = lin.weight.data[:, rank * per:(rank + 1) * per].clone()
        nl = torch.nn.Linear(per, lin.out_features, bias=False, dtype=w.dtype, device=w.device)
        nl.weight = torch.nn.Parameter(w, requires_grad=False)
        return nl

    for L in model.model.layers:
        a, mlp = L.self_attn, L.mlp
        a.q_proj, a.k_proj, a.v_proj = col(a.q_proj, qd), col(a.k_proj, kvd), col(a.v_proj, kvd)
        a.o_proj = row(a.o_proj, qd)
        mlp.gate_proj, mlp.up_proj = col(mlp.gate_proj, idim), col(mlp.up_proj, idim)
        mlp.down_proj = row(mlp.down_proj, idim)
        _tp_set_head_counts_(a, nh, nkv, tp_size)


def _tp_set_head_counts_(a, nh: int, nkv: int, tp_size: int, q_heads=None, kv_heads=None) -> None:
    """Set ONE attention module's head counts (+ its .config copy) to THIS rank's slice. Uniform
    1/tp by default (q_heads/kv_heads=None); heterogeneous when the explicit per-rank counts are
    given (a bigger node gets more heads). Factored out so the v2 make-structure path applies the
    EXACT same counts the served slice has. nh/nkv are the FULL (pre-shard) head counts."""
    import copy
    qh = q_heads if q_heads is not None else nh // tp_size
    kvh = kv_heads if kv_heads is not None else nkv // tp_size
    a.num_heads = qh
    a.num_key_value_heads = kvh
    a.num_key_value_groups = qh // kvh
    if hasattr(a, "config"):
        a.config = copy.copy(a.config)
        a.config.num_attention_heads = qh
        a.config.num_key_value_heads = kvh


def _tp_make_structure_(model, rank: int, tp_size: int, cfg, weights=None) -> None:
    """TP-v2 (per-rank streaming): build the REDUCED-DIM module STRUCTURE for `rank` of `tp_size`
    WITHOUT slicing any weights — every replaced linear is created on the META device (zero memory),
    so the per-rank SLICED weights can be streamed straight into it via load_state_dict(assign=True).
    Column-parallel q/k/v/gate/up keep this rank's OUTPUT rows (out_features), row-parallel o/down
    keep this rank's INPUT cols (in_features, bias=False), head counts scaled per rank. `weights`
    (per-rank capacity, len==tp_size) -> HETEROGENEOUS sizes via the SAME wire._tp_hetsplit the
    server slices with (so shapes match); None/mismatched -> the uniform 1/tp split. The server's
    /weights_tp serves tensors of exactly these shapes, so a plain assign-load fills them."""
    import torch
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // nh)
    inter = cfg.intermediate_size
    if weights and len(weights) == tp_size:
        geo = _tp_hetsplit(nh, nkv, hd, inter, 128, list(weights))[rank]   # 128 = int4 group align
        qd, kvd, idim = geo["q_len"], geo["kv_len"], geo["idim_len"]
        qh, kvh = geo["q_heads"], geo["kv_heads"]
    else:
        qd, kvd, idim = nh * hd // tp_size, nkv * hd // tp_size, inter // tp_size
        qh = kvh = None

    def col_struct(lin, per):   # column-parallel: out_features -> per (bias kept, sliced server-side)
        nl = torch.nn.Linear(lin.in_features, per, bias=lin.bias is not None,
                             dtype=lin.weight.dtype, device="meta")
        return nl

    def row_struct(lin, per):   # row-parallel: in_features -> per, bias dropped (added once post-AR)
        nl = torch.nn.Linear(per, lin.out_features, bias=False,
                             dtype=lin.weight.dtype, device="meta")
        return nl

    for L in model.model.layers:
        a, mlp = L.self_attn, L.mlp
        a.q_proj, a.k_proj, a.v_proj = (col_struct(a.q_proj, qd), col_struct(a.k_proj, kvd),
                                        col_struct(a.v_proj, kvd))
        a.o_proj = row_struct(a.o_proj, qd)
        mlp.gate_proj, mlp.up_proj = col_struct(mlp.gate_proj, idim), col_struct(mlp.up_proj, idim)
        mlp.down_proj = row_struct(mlp.down_proj, idim)
        _tp_set_head_counts_(a, nh, nkv, tp_size, qh, kvh)


# #tp-mesh-keepalive: sentinel broadcast payload that means "this is a liveness ping, not a forward
# input". A real forward input is a pickled tuple (starts with the pickle opcode b'\x80'), so this
# fixed marker can never collide with one. rank 0 sends it during IDLE gaps + reads a 1-byte ack from
# each peer; that round-trip keeps BOTH directions of every mesh socket warm so an idle connection
# can't go silently half-open (the bytes-vanish failure that breaks the lockstep mesh until reload).
_TP_PING = b"\x00__tp_ping__"


class _TPAllReduce:
    """Root-based sum-all-reduce over blocking TCP among a TP group's `tp_size` ranks. Rank 0
    binds and accepts the peers; ranks 1..N-1 connect to it. Called INSIDE the (sync) forward via
    forward-hooks on o_proj/down_proj. Payloads are tiny decode hidden-vectors -> latency-bound,
    fine on the 2.5/10GbE fabric. Validated token-identical to the unsharded model."""
    def __init__(self, rank: int, tp_size: int, root_host: str, root_port: int,
                 timeout: float = 600.0, op_timeout: float = 120.0) -> None:
        # timeout: how long rank 0 waits for ALL peers to connect (rendezvous). Generous (600s) so a
        # peer that is slow/recovering still joins instead of tripping a bare TimeoutError. op_timeout:
        # per-op deadline on rank 0's PEER sockets so a forward all-reduce against a stalled/dead peer
        # FAILS FAST with a clear error instead of blocking recv() forever (the empty-error 500 hang).
        import struct
        self.rank, self.N, self._struct = rank, tp_size, struct
        self.peers: list = []
        self.root = None
        if tp_size <= 1:
            return
        if rank == 0:
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", root_port))
            srv.listen(tp_size)
            srv.settimeout(timeout)
            try:
                for _ in range(tp_size - 1):
                    c, _addr = srv.accept()
                    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    _set_keepalive(c)          # mesh idles through the whole weight stream -> keep it alive
                    c.settimeout(op_timeout)   # active-op deadline; rank 0 only touches peers mid-forward
                    self.peers.append(c)
            except (socket.timeout, TimeoutError):
                raise RuntimeError(
                    f"tp rendezvous: rank 0 timed out after {timeout:.0f}s on :{root_port} "
                    f"({len(self.peers)}/{tp_size - 1} peers connected)") from None
            finally:
                srv.close()
        else:
            deadline = time.time() + timeout
            while True:
                try:
                    s = socket.create_connection((root_host, root_port), timeout=15,
                                                 source_address=_local_addr())
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    _set_keepalive(s)          # mesh idles through the whole weight stream -> keep it alive
                    self.root = s
                    break
                except OSError:
                    if time.time() > deadline:
                        raise RuntimeError(
                            f"tp rendezvous: rank {rank} could not reach root "
                            f"{root_host}:{root_port} within {timeout:.0f}s") from None
                    time.sleep(0.2)

    def _recvn(self, s, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            c = s.recv(n - len(buf))
            if not c:
                raise IOError("tp peer closed")
            buf += c
        return buf

    def _send(self, s, t) -> None:
        import pickle
        # Pickle the torch tensor directly — numpy() can't represent bfloat16, the fleet's
        # weight/activation dtype. torch tensors pickle fine for any dtype.
        b = pickle.dumps(t.detach().to("cpu").contiguous())
        s.sendall(self._struct.pack("!I", len(b)) + b)

    def _recv(self, s, like):
        import pickle
        n = self._struct.unpack("!I", self._recvn(s, 4))[0]
        t = pickle.loads(self._recvn(s, n))
        return t.to(like.device, like.dtype)

    def allreduce(self, t):
        if self.N <= 1:
            return t
        # Accumulate in float32 (transport stays the tensor's dtype) so summing the row-parallel
        # partials matches the single-device result without bf16 add-rounding drift. A socket
        # timeout/close here = a peer fell out of lockstep (reloaded/dead); surface it as a clear
        # error so the forward fails fast (reload required) instead of hanging into an empty 500.
        try:
            if self.rank == 0:
                acc = t.detach().float()
                for c in self.peers:
                    acc = acc + self._recv(c, t).float()
                out = acc.to(t.dtype)
                for c in self.peers:
                    self._send(c, out)
                return out
            self._send(self.root, t)
            return self._recv(self.root, t)
        except (socket.timeout, TimeoutError, OSError) as e:
            raise RuntimeError(f"tp all-reduce failed — peer rank stalled or closed ({e!r}); "
                               f"the TP mesh is broken, reload the model") from None

    def broadcast(self, payload: bytes) -> None:
        """rank 0 -> every peer: the per-forward input (raw bytes). Followers block in
        recv_broadcast() until this arrives, then run their sharded forward in lockstep."""
        hdr = self._struct.pack("!I", len(payload))
        for c in self.peers:
            c.sendall(hdr + payload)

    def recv_broadcast(self) -> bytes:
        """peer: block for the next forward's input from rank 0. b'' means 'stop'."""
        n = self._struct.unpack("!I", self._recvn(self.root, 4))[0]
        return self._recvn(self.root, n) if n else b""

    def keepalive(self) -> bool:
        """#tp-mesh-keepalive (rank 0): send a ping to every peer + read its 1-byte ack. The
        round-trip keeps BOTH directions of each idle mesh socket warm (prevents the silent
        half-open that breaks the lockstep) AND detects a dead/gone peer early. Returns False on
        any peer timeout/close so the caller can flag the mesh broken before a real forward does.
        MUST be called holding the same lock that guards forward all-reduces (never interleave)."""
        if self.N <= 1 or self.rank != 0:
            return True
        try:
            hdr = self._struct.pack("!I", len(_TP_PING))
            for c in self.peers:
                c.sendall(hdr + _TP_PING)
            for c in self.peers:
                if self._recvn(c, 1) != b"\x01":
                    return False
            return True
        except (socket.timeout, TimeoutError, OSError):
            return False

    def ack_ping(self) -> None:
        """#tp-mesh-keepalive (peer): reply to rank 0's liveness ping (1 byte)."""
        if self.root is not None:
            self.root.sendall(b"\x01")

    def close(self) -> None:
        for s in list(self.peers) + ([self.root] if self.root else []):
            with contextlib.suppress(Exception):
                s.close()
        self.peers, self.root = [], None


# code-split Inc 8: _missing_pkgs_from_err/_build_with_autodeps/EmbeddingModel live in
# worker_load.py now (VERBATIM, beside their only call site; back-imported below).


# ---- m4c153 code-split: Shard/Worker relocated into mixin modules (see state.py) ----
# Worker-side leaf modules holding Shard/Worker methods VERBATIM; state.bind() (at module end)
# injects this module's namespace so the relocated bodies resolve their globals. In client.py's
# EXTRA_UPDATE_FILES. CONVERGENCE BRIDGE: an old worker swapping in this client.py fetched it but
# not yet these files, so pull each from GitHub raw once if missing (repo_raw_url imported above).
import urllib.request as _wsreq
for _wsm in ("state", "shard_build", "shard_forward", "worker_load", "worker_net",
             "worker_hw", "worker_update", "shard_compile",
             "worker_quant"):   # code-split Inc 7 + 8 + 9 + 10
    try:
        __import__(_wsm)
    except Exception:
        # E2: bounded retry (raw-CDN propagation lag on a freshly-added module), then exit 42.
        for _wsa in range(3):
            try:
                with _wsreq.urlopen(repo_raw_url().format(f=_wsm + ".py"), timeout=30) as _wsr:
                    _wsb = _wsr.read()
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), _wsm + ".py"), "wb") as _wsf:
                    _wsf.write(_wsb)
                __import__(_wsm)
                break
            except Exception:
                time.sleep(5 * (_wsa + 1))
        else:
            # Still missing -> exit 42 so the supervisor RELAUNCHES (client.bat loops ONLY on 42;
            # any other exit drops to `pause` and the Windows worker dies permanently — systemd
            # workers restart regardless). Each relaunch retries the fetch until the CDN edge
            # catches up, so a bridge 404 is a bounded crash-loop, not a dead worker.
            print(f"[update] required module {_wsm}.py unavailable (import+fetch failed) - "
                  f"exit 42 for supervisor relaunch", flush=True)
            os._exit(42)
import state
import shard_build, shard_forward, worker_load, worker_net
import worker_hw
import worker_update
import worker_quant   # code-split Inc 10: quant/kernel family (canonical FLAG home — never from-import the flags)
from worker_update import (_fetch_repo_file, _self_update_check, _self_update_loop,
                           _fwd_watchdog_loop, _console_panel_loop)   # noqa: E402,F401
from worker_load import (_missing_pkgs_from_err, _build_with_autodeps, EmbeddingModel,
                         _weight_map, _load_tensors, _assemble_sd)   # noqa: E402,F401
from worker_hw import (_release_vram, _release_ram, _flush_os_cache, mem_maintenance_loop,
                       detect_device, _gpu_mem_gb, _rocm_gpu_util, _using_gpu, free_disk_gb,
                       _os_default_src, _iface_kind, select_route, _controller_is_local,
                       _fmt_route, _fmt_ram_mods, _detect_ram_windows, detect_ram,
                       build_registration, _dir_size, cleanup_storage)   # noqa: E402,F401
from worker_quant import (_quantize_int4_, _quantize_int8_, _quantize_experts4_,
                          _quantize_linear, _quantize_linear4, _assign_meta_from_sd,
                          _accelerate_cpu_linears, tune_cpu_threads, _find_moe_block,
                          _moe_bridge_cls, _model_has_fused_experts,
                          _model_has_nonfused_experts, _layer_has_meta_experts,
                          _layer_has_meta_experts_nonfused, _quantize_experts4_streamed,
                          _quantize_experts4_streamed_nonfused, _quant4_linear_cls,
                          _packed4_3d_cls, _INT4_GROUP, _install_fused_moe_forward,
                          _w4a16_triton_op, _w4a16_moe_op, _pack4_expert,
                          _pack4_3d, _quantize_int2_, _quantize_linear2,
                          _quant2_linear_cls, _INT2_GROUP)   # noqa: E402,F401
# Inc 10 NOTE: the quant FLAGS (_CPU_FP32_GEMM, _CPU_FP32_MIN_ROWS, _CPU_BF16_GEMM_OK,
# _FUSED_INT4) are deliberately NOT back-imported — they are rebound at runtime inside
# worker_quant (tune_cpu_threads / --no-cpu-fp32) and a from-import would freeze a stale
# copy in this namespace. Read/write them as live attrs: worker_quant._CPU_FP32_GEMM etc.
from shard_build import ShardBuildMixin
from shard_forward import ShardForwardMixin
from worker_load import WorkerLoadMixin
from worker_net import WorkerNetMixin


class Shard(ShardBuildMixin, ShardForwardMixin):
    # m4c153 code-split: Shard composed from ShardBuildMixin (placement/stream-load/from_*)
    # + ShardForwardMixin (forward/_forward_impl). __init__ and _finalize_placement stay here.
    # Inc 10: the quant flags moved to worker_quant.py and are REBOUND there at runtime, so
    # _finalize_placement reads them as LIVE module attrs (worker_quant._CPU_FP32_GEMM /
    # ._FUSED_INT4) — never through a from-import copy. state.bind injects the client
    # namespace into the mixin modules at startup — see state.py.
    def __init__(self, cfg, sd: dict, layer_start: int, layer_end: int,
                 has_embed: bool, has_head: bool, dtype,
                 device: str = "cpu", gpu_mem_gb: float = 0.0,
                 attn: str = "eager", quant: str = "none",
                 tp_rank: int = 0, tp_size: int = 1, tp_allreduce=None,
                 kv_quant: str = "none", kv_offload: bool = False,
                 kv_slots: int = 1) -> None:
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.kv_offload = bool(kv_offload)   # #kv-offload: KV in system RAM (OffloadedCache)
        # #kv-slots: per-request KV slot count C (dict slot->cache; shard_forward routes each
        # forward/crop to its request's slot). Set before placement so the KV budget scales xC.
        self.kv_slots = max(1, int(kv_slots or 1))
        # Qwen2.5-Omni: AutoModelForCausalLM can't build Qwen2_5OmniTextConfig. Build ONLY the
        # Thinker TEXT decoder (Qwen2_5OmniThinkerTextModel) + a fresh lm_head, wrapped so
        # .model.layers / .lm_head match the served 'model.*'/'lm_head' weights (controller
        # strips the 'thinker.' prefix). We DON'T build the audio_tower/visual towers on workers
        # — they only hold text layers, and constructing those heavy towers destabilized nodes.
        # self.cfg = the Thinker's text config.
        omni_thinker = getattr(cfg, "thinker_config", None)
        # #cudagraph: is this a multimodal-capable checkpoint (its prefill can carry image/audio
        # `inject` frames)? Captured here from the ORIGINAL top-level config, BEFORE the text-config
        # extraction below drops the vision/audio/thinker markers. Used only to gate the opt-in
        # CUDA-graph decode path OFF for multimodal models. Additive — no effect on any other path.
        self._mm_capable = bool(omni_thinker is not None
                                or getattr(cfg, "vision_config", None) is not None
                                or getattr(cfg, "audio_config", None) is not None)
        if omni_thinker is not None:
            self.cfg = cfg.get_text_config()
        else:
            # Other composite/multimodal checkpoints (e.g. Qwen3.6-35B-A3B) nest the text model
            # under .text_config; the top-level lacks num_hidden_layers etc. Build from the text
            # sub-config (weights already remapped to model.*). No-op for plain text models.
            if getattr(cfg, "text_config", None) is not None:
                cfg = cfg.get_text_config()
            self.cfg = cfg
        # Hybrid linear-attention arch (e.g. Qwen3.5-MoE: per-layer Gated-DeltaNet vs full
        # attention via cfg.layer_types). False (bit-identical to the old path) for every
        # dense/standard model (no layer_types) — incl. the Omni Thinker.
        _lt = getattr(self.cfg, "layer_types", None)
        self._hybrid = bool(_lt) and any(t != "full_attention" for t in _lt)
        # Qwen2.5-Omni uses CLASSIC multimodal RoPE (apply_multimodal_rotary_pos_emb does
        # cos[i % 3]), so cos/sin MUST be [3, bs, seq, dim] — the worker has to feed the rotary
        # 3D positions [3, bs, seq] even for plain TEXT (all three sections = the same sequential
        # positions). (The 35B's INTERLEAVED mRoPE tolerated 1D; Omni does not.)
        self._omni = omni_thinker is not None
        # #vl-vision: Qwen2.5-VL rotary needs 3D position_ids [3,bs,seq] even for text (like Omni).
        self._mrope3d = self._omni or str(getattr(self.cfg, "model_type", "")).lower() in (
            "qwen2_5_vl_text", "qwen2_5_vl")
        # Attention kernel: 'eager' (additive-mask matmul, bit-exact) or 'sdpa' (fused).
        self.cfg._attn_implementation = attn
        self.dtype = dtype
        self.layer_start, self.layer_end = layer_start, layer_end
        self.has_embed, self.has_head = has_embed, has_head

        with torch.device("meta"):
            if omni_thinker is not None:
                from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                    Qwen2_5OmniThinkerTextModel)
                class _OmniTextCausalLM(torch.nn.Module):   # minimal CausalLM shape: .model + .lm_head
                    def __init__(self, m, h):
                        super().__init__()
                        self.model = m
                        self.lm_head = h
                text_model = Qwen2_5OmniThinkerTextModel(self.cfg)
                lm_head = torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
                model = _OmniTextCausalLM(text_model, lm_head)
            elif str(getattr(self.cfg, "model_type", "")).lower() in ("qwen2_5_vl_text", "qwen2_5_vl"):
                # Qwen2.5-VL text decoder only (controller runs the vision tower). AutoModelForCausalLM
                # has no Qwen2_5_VLTextConfig mapping. (#vl-vision)
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
                class _VLTextCausalLM(torch.nn.Module):
                    def __init__(self, m, h):
                        super().__init__(); self.model = m; self.lm_head = h
                model = _VLTextCausalLM(
                    Qwen2_5_VLTextModel(self.cfg),
                    torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False))
            else:
                model = AutoModelForCausalLM.from_config(self.cfg)
        model = model.to(dtype)
        # MoE: fuse a legacy per-expert checkpoint into the stacked tensors this transformers
        # version expects (else the fused params stay on meta and .to(device) fails). No-op
        # for dense models / already-fused checkpoints.
        sd = _fuse_moe_experts(sd, model)
        # assign=True installs our real tensors; unlisted params stay on 'meta'
        # (zero memory) — that is what splits RAM across nodes.
        try:
            model.load_state_dict(sd, strict=False, assign=True)
        except TypeError:   # trust_remote_code model with a 4.x load_state_dict (no `assign`) — use base
            torch.nn.Module.load_state_dict(model, sd, strict=False, assign=True)
        _assign_meta_from_sd(model, sd)   # materialize buffers load_state_dict skipped (e.g. MiniMax e_score_correction_bias)
        rot = model.model.rotary_emb
        model.model.rotary_emb = type(rot)(self.cfg)  # text cfg (Omni: cfg is the full config); inv_freq computed
        model.eval()

        self.model = model
        self.owned_layers = [model.model.layers[i] for i in range(layer_start, layer_end)]
        self.embed = model.model.embed_tokens if has_embed else None
        self.norm = model.model.norm if has_head else None
        self.head = model.lm_head if has_head else None
        seen: set[int] = set()
        self.loaded_params = 0
        self.loaded_bytes = 0
        for t in sd.values():
            if t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr())
            self.loaded_params += t.numel()
            self.loaded_bytes += t.numel() * t.element_size()
        # Tensor parallelism (M4): shard every layer to this rank and all-reduce the
        # row-parallel projections (o_proj/down_proj) across the TP group. Done BEFORE quant
        # and placement so each rank quantizes/places only its 1/N slice. tp_size==1 -> no-op.
        self.tp_rank, self.tp_size, self.tp_allreduce = tp_rank, tp_size, tp_allreduce
        if tp_size > 1:
            _tp_shard_model_(model, tp_rank, tp_size, cfg)
            ar = tp_allreduce
            for lyr in self.owned_layers:
                lyr.self_attn.o_proj.register_forward_hook(lambda m, i, o: ar.allreduce(o))
                lyr.mlp.down_proj.register_forward_hook(lambda m, i, o: ar.allreduce(o))
            # recompute footprint from the sharded modules (1/N of the layer linears)
            mods = ([self.embed] if self.has_embed else []) + list(self.owned_layers)
            if self.has_head:
                mods += [self.norm, self.head]
            self.loaded_bytes = sum(self._mod_bytes(m) for m in mods if m is not None)
        # weight-only quant (before device placement, so we move quantized weights not bf16).
        # int8: ~1/2 footprint, head quantized too. int4: ~1/4 footprint, head LEFT bf16
        # (logits are quant-sensitive; the head is one matrix so the memory cost is small).
        # int2 (#int2): ~1/6 footprint (2-bit, group 64), head LEFT bf16 like int4; DENSE only
        # (the /load route downgrades int2-on-MoE to int4 — no 2-bit expert packer).
        self.quant = quant
        self.kv_quant = kv_quant   # #172 TurboQuant KV preset (none|turbo2|turbo3|turbo4); read in shard_forward
        if quant in ("int8", "int4", "int2"):
            qlayer = (_quantize_int4_ if quant == "int4"
                      else _quantize_int2_ if quant == "int2" else _quantize_int8_)
            for lyr in self.owned_layers:
                qlayer(lyr)
                if quant == "int4":
                    _quantize_experts4_(lyr)   # fused 3D MoE experts (the bulk of a big MoE)
            if quant == "int8" and self.has_head and self.head is not None:
                model.lm_head = _quantize_linear(model.lm_head)
                self.head = model.lm_head
            # recompute footprint to reflect the quantized weights
            mods = ([self.embed] if self.has_embed else []) + list(self.owned_layers)
            if self.has_head:
                mods += [self.norm, self.head]
            self.loaded_bytes = sum(self._mod_bytes(m) for m in mods if m is not None)
        self.kv = None  # per-generation KV cache (DynamicCache), reset on reset=True
        # #kv-slots: authoritative slot->cache store; self.kv is the RUNNING slot's binding
        # (shard_forward binds/writes-back under _fwd_lock). Empty + slot 0 == legacy behavior.
        self._kv_by_slot = {}
        # #fwd-serialize: forwards on ONE shard mutate the single shared self.kv. Normally a model's
        # forwards are strictly sequential (the controller's per-model lock + awaiting each result),
        # so this is uncontended. The exception is an ORPHANED forward: when the controller reclaims a
        # wedged gen (gen-stall watchdog / client disconnect), the worker's forward keeps running in an
        # uncancellable thread and a fresh request can spawn a SECOND forward on the same shard — both
        # mutate self.kv concurrently, desyncing the KV length from the causal mask -> the SDPA
        # "expanded size N must match M" crash. A non-blocking acquire makes the new forward fail FAST
        # (controller re-prefills) instead of racing the orphan, and never blocks a thread-pool slot.
        self._fwd_lock = threading.Lock()
        # rotary_emb stays on CPU; cos/sin are computed there and moved per-device.
        self.cpu = torch.device("cpu")
        self._place_modules(device, gpu_mem_gb)

    def _finalize_placement(self) -> None:
        """Detect the single-device case (every owned module on ONE device — the
        common consolidated-on-GPU shard) and pin rotary_emb there. The decode
        fast path in forward() then computes cos/sin + mask on that device and
        skips the per-token CPU compute and host->device copies. For the hybrid
        (cpu+gpu spill) or multi-device case, uniform_device stays None and the
        general path keeps moving positional aux per layer."""
        used = list(self.layer_devices)
        if self.has_embed:
            used.append(self.embed_device)
        if self.has_head:
            used += [self.norm_device, self.head_device]
        uniq = {(d.type, d.index) for d in used}
        self.uniform_device = used[0] if (used and len(uniq) == 1) else None
        # rotary lives on the uniform device (GPU when consolidated), else CPU for
        # the general path that moves cos/sin per layer.
        self.rotary_device = self.uniform_device if self.uniform_device is not None else self.cpu
        with contextlib.suppress(Exception):
            self.model.model.rotary_emb.to(self.rotary_device)
        # CPU matmul acceleration (runs for EVERY placement mode — cpu, hybrid cpu+gpu, and
        # gpu, since CPU spill can occur in any of them). Wrap each CPU-resident NATIVE
        # nn.Linear so its bf16 matmul runs as a transient fp32 GEMM (1-2 orders of magnitude
        # faster on CPU; see the CPU-matmul module header). Done here — the single chokepoint
        # both placement return-paths hit — AFTER every weight has its FINAL device, so it
        # sees only what actually landed on CPU. GPU-resident Linears are skipped (already
        # fast tensor-core kernels — untouched), and it composes with hybrid spill and TP-v2
        # reduced-dim modules (wraps whatever is on CPU). QuantLinear/QuantLinear4 handle fp32
        # in their own dequant, so they're skipped here. Per-call upcast only -> resident RAM
        # unchanged. Idempotent (safe if placement is ever re-run).
        if worker_quant._CPU_FP32_GEMM:   # Inc 10: live attr (rebound by --no-cpu-fp32)
            for _m in (([self.embed] if self.has_embed else [])
                       + list(self.owned_layers)
                       + ([self.norm, self.head] if self.has_head else [])):
                _accelerate_cpu_linears(_m)
        # Fused int4 (#71): every weight now has its FINAL device, so build the tinygemm fused-int4
        # kernel per QuantLinear4 (2D linears: attn, router, shared experts, dense). ~3.6x faster
        # int4 decode (no per-token re-dequant). Self-checked + naive fallback inside prepare_fused.
        # Packed4Tensor3D's prepare_fused is picked up by this same sweep: it doesn't build a fused
        # op (the MoE kernel binds at forward time) but re-allocates expert rows on an odd 64B
        # multiple (#dram-dealias, ROCm) — must also run AFTER final device placement. Idempotent.
        if worker_quant._FUSED_INT4:   # Inc 10: live attr
            _nf = 0
            for _m in (([self.embed] if self.has_embed else [])
                       + list(self.owned_layers)
                       + ([self.norm, self.head] if self.has_head else [])):
                for _sub in _m.modules():
                    if hasattr(_sub, "prepare_fused"):
                        _sub.prepare_fused()
                        if getattr(_sub, "_fused", None) is not None:
                            _nf += 1
            if _nf:
                print(f"[int4] fused tinygemm kernel active on {_nf} linear(s) "
                      f"({self.placement})", flush=True)
        # #int4-vram: the per-layer prepare_fused self-checks dequant each weight to bf16 transiently;
        # the caching allocator (esp. ROCm/gfx1151) holds those freed blocks, inflating resident VRAM.
        # Release them, then LOG THE TRUTH: real in-use (memory_allocated) vs the allocator pool
        # (memory_reserved) + an int4 census (packed qweight bytes vs bf16 param bytes) so a genuine
        # bf16-resident footprint is told apart from a reclaimable pool / an accounting over-count.
        try:
            import torch as _t
            _mods = (([self.embed] if self.has_embed else []) + list(self.owned_layers)
                     + ([self.norm, self.head] if (self.has_head and self.norm is not None) else []))
            _cuda = any(p.device.type == "cuda" for m in _mods for p in m.parameters()) \
                or any(b.device.type == "cuda" for m in _mods for b in m.buffers())
            if _cuda and _t.cuda.is_available():
                _t.cuda.empty_cache()
                _GB = 1024 ** 3
                _al = _t.cuda.memory_allocated() / _GB
                _rv = _t.cuda.memory_reserved() / _GB
                _nq = _qb = _bf = 0
                for _m in _mods:
                    for _s in _m.modules():
                        _qw = getattr(_s, "qweight", None)
                        if _qw is not None and getattr(_qw, "device", None) is not None and _qw.device.type == "cuda":
                            _nq += 1
                            _qb += _qw.numel() * _qw.element_size()
                    for _p in _m.parameters():
                        if _p.device.type == "cuda" and _p.dtype == _t.bfloat16:
                            _bf += _p.numel() * _p.element_size()
                print(f"[int4-vram] {self.placement} | in-use={_al:.2f}GB reserved={_rv:.2f}GB | "
                      f"QuantLinear4={_nq} qweight={_qb/_GB:.2f}GB bf16params={_bf/_GB:.2f}GB", flush=True)
        except Exception as _e:
            print(f"[int4-vram] probe failed: {_e!r}", flush=True)


# code-split Inc 8: _weight_map/_load_tensors/_assemble_sd live in worker_load.py now
# (VERBATIM; back-imported so shard_build resolves them via the published namespace).

# code-split Inc 8: the self-update machinery (SELF_UPDATE_* knobs, _extract_version,
# _ver_ordinal, _fetch_repo_file) lives in worker_update.py now (VERBATIM; back-imported).
# module (wire.py) is listed in BOTH client.py + server.py.
EXTRA_UPDATE_FILES: list[str] = ["wire.py", "config.json", "shards.py",
                                 "shard_compile.py",   # code-split Inc 9: SHARED compile/pack family
                                 "worker_hw.py",   # code-split Inc 7: hw/host helpers
                                 "worker_update.py",   # code-split Inc 8: self-update + watchdogs
                                 # m4c153 code-split: shared-state registry + Shard/Worker mixins
                                 "state.py", "shard_build.py", "shard_forward.py",
                                 "worker_load.py", "worker_net.py",   # config + shared packer
                                 "kv_quant.py",   # TurboQuant KV-cache quantizer (#172)
                                 "worker_quant.py",   # code-split Inc 10: quant/kernel family
                                 "worker_t2i.py",   # #t2i-serve: diffusion image engine (lazy import)
                                 "worker_tts.py",   # #tts-serve: Kokoro speech engine (lazy import)
                                 "worker_t2a.py"]   # #t2a-serve: ACE-Step music engine (lazy import)
# (#distributed-packing) synced like a module — shards.pack_unit_tensors is the shared packer the
# remote-pack handler calls, so a worker-packed cache unit is bit-identical to a controller-compiled one.


# code-split Inc 8: _self_update_check + _self_update_loop live in worker_update.py now
# (VERBATIM; they read EXTRA_UPDATE_FILES above + VERSION through the bound namespace).

class Worker(WorkerLoadMixin, WorkerNetMixin):
    # m4c153 code-split: Worker composed from WorkerLoadMixin (build/load/pack/unload/TP)
    # + WorkerNetMixin (next-hop connect/send + data-plane). Only __init__ stays here.
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = getattr(args, "device", "cpu")
        self.gpu_mem_gb = getattr(args, "gpu_mem_gb", 0.0)
        self.attn = getattr(args, "attn", "eager")
        self.quant = getattr(args, "quant", "none")
        # Multi-model: this node may hold one stage of several models at once. State is
        # keyed by model_id (the HF target id == load msg["model_id"] == frame model_id),
        # because the single inbound data port is shared by every model's pipeline.
        self.shards: dict[str, Shard] = {}                       # model_id -> this node's shard
        # Count of builds in flight. A shard isn't in self.shards until its (multi-minute, per-layer
        # streaming) build completes, so mid-build the process holds the partial shard's RAM while
        # looking "idle". Both the reclaim-restart (#51) and the self-update gate must treat a build
        # in progress as BUSY, else they restart the worker mid-load and wedge the whole TP group.
        self._building: int = 0
        # Fresh-process RSS baseline. After going idle (no shards) we compare against this; if the
        # OS didn't reclaim a dropped shard (Windows keeps committed bytes; glibc can retain arenas),
        # the only reliable fix is a restart -> see _maybe_self_restart_if_stuck.
        try:
            import psutil
            self._rss_baseline_gb = psutil.Process().memory_info().rss / GB
        except Exception:
            self._rss_baseline_gb = 0.0
        self.next_writers: dict[str, asyncio.StreamWriter] = {}  # model_id -> conn to its next stage
        self.next_peer: dict[str, str] = {}                      # model_id -> next-hop label (bandwidth)
        self._next_last_send: dict[str, float] = {}              # #stage0-stale-reconnect: model_id -> last forward ts
        self.assignments: dict[str, dict] = {}                   # model_id -> load msg (debug/reload)
        # #multi-controller: model_id -> the controller ("host:port") that loaded it. Lets
        # inventory() report only the asking controller's shards and lets a full-teardown unload be
        # scoped to one tenant — without this a second controller adopts, or after the 90s adopt
        # grace FREES, the first controller's live models. None/absent = unowned = anyone may claim
        # (back-compat: pre-existing and adopted shards).
        self.shard_owner: dict[str, str] = {}
        # #handoff: (host, port) of a controller this node is being transferred to. Set by
        # handle_handoff; run()'s reconnect loop re-targets to it and clears it. Shards are KEPT
        # (ownership already reassigned) so the new controller adopts them without a re-stream.
        self._handoff = None
        self._weight_tmps: dict[str, str] = {}                   # model_id -> temp file backing its mmap
        self.data_server: asyncio.AbstractServer | None = None   # shared data port; bound on first load
        self._tp = None              # _TPAllReduce when a load is tensor-parallel (single TP model)
        self._tp_thread = None       # follower loop thread (peer ranks, tp_rank>0)
        self._tp_stop = False
        self._tp_model_id = None     # model_id currently in TP mode, if any
        # #tp-mesh-keepalive: serialize ALL rank-0 mesh socket use (the broadcast+all-reduce of a
        # forward, and the idle keepalive ping) so they never interleave + corrupt the byte stream.
        self._tp_lock = threading.Lock()
        self._tp_last_fwd = 0.0      # monotonic-ish wall ts of rank 0's last mesh forward (warmth clock)
        self._tp_ka_thread = None    # rank-0 keepalive thread handle
        self._tp_broken = False      # set when a keepalive ping finds the mesh dead -> surfaced/served
        # #22 inc 3: multimodal embeds staged by a 'mm' frame, consumed by the next prefill
        # for the same (model_id, req_id). Only stage 0 (has_embed) ever populates this.
        self.pending_mm: dict[tuple, tuple] = {}
        # #distributed-packing Inc 3b: meta skeletons (per model_id) for per-expert MoE fuse-at-pack.
        # Cached so all N layers of one compile reuse a single from_config build (meta-only, cheap).
        self._pack_skel: dict = {}
        # #hop-recovery: the live control-link send helper (session()'s `reply`, wlock+_enc framed)
        # so the data plane can push an unsolicited hop_error to the controller when a next-hop send
        # dies mid-generation. Bound after register, cleared on disconnect. Same event loop as
        # _data_inbound/_send_next (the forward compute is the only thing offloaded to a thread), so a
        # plain await is safe — no cross-thread queue needed.
        self._ctrl_send = None       # Optional[Callable[[dict], Awaitable[None]]]
        self._node_id: str = ""      # our registered node id (stamped onto the hop_error frame)
        # #adopt: True once a register-ack advertised adoption — this controller re-adopts kept
        # shards, so the session teardown KEEPS loaded models across a controller-only restart.
        self._ctrl_adopts: bool = False


# ---------------------------------------------------------------------------
# Control session (bidirectional)
# ---------------------------------------------------------------------------

async def _heartbeat_loop(writer: asyncio.StreamWriter, lock: asyncio.Lock,
                          node_id: str, interval: float, gpu: bool = False,
                          worker=None) -> None:
    psutil.cpu_percent(interval=None)
    proc = psutil.Process()   # this worker's own process, for RSS reporting
    hist: deque = deque()  # (t, net_in, net_out) over the last 10 s
    _log_cursor = 0        # #logs: relay cursor — send only NEW stdout/stderr lines each beat
    while True:
        now = time.time()
        hist.append((now, NET["in"], NET["out"]))
        while len(hist) > 1 and now - hist[0][0] > 10:
            hist.popleft()
        span = now - hist[0][0]
        in_bps = (NET["in"] - hist[0][1]) / span if span > 0 else 0.0
        out_bps = (NET["out"] - hist[0][2]) / span if span > 0 else 0.0
        vm = psutil.virtual_memory()
        try:
            rss_gb = round(proc.memory_info().rss / GB, 2)   # RAM this worker process holds
        except Exception:
            rss_gb = 0.0
        hb = {"type": "heartbeat", "node_id": node_id,
              "free_mem_gb": round(vm.available / GB, 2),
              "free_disk_gb": round(free_disk_gb(), 2),
              "cpu_percent": psutil.cpu_percent(interval=None),
              "proc_rss_gb": rss_gb,
              "net_in_bps": round(in_bps), "net_out_bps": round(out_bps),
              # cumulative per-peer data-plane bytes (bandwidth page): node<->node hops the
              # controller can't see, plus this node's <->controller bytes (deduped server-side).
              "net_peers": {p: dict(c) for p, c in NET_PEERS.items()}}
        # #prefill-progress: report per-model forward liveness — shards whose forward is RUNNING
        # (_fwd_lock held) with a recent per-layer progress stamp (same signal the local
        # fwd-watchdog uses). The controller's gen-stall watchdog reads this to tell a
        # slow-but-advancing prefill (GPU contention) from a true wedge, so healthy long prefills
        # stop being reclaimed at the threshold while real wedges still are.
        fp = {}
        if worker is not None:
            for _mid, _sh in list(getattr(worker, "shards", {}).items()):
                try:
                    _lk = getattr(_sh, "_fwd_lock", None)
                    _ts = getattr(_sh, "_fwd_progress_ts", 0.0) or 0.0
                    if _lk is not None and _lk.locked() and (now - _ts) < 120.0:
                        # [rid, ts]: rid = the request this forward belongs to, so the controller
                        # credits progress only to a LIVE (still-pending) request — an orphaned
                        # forward's stamps can't keep a newer wedged gen alive.
                        fp[_mid] = [getattr(_sh, "_fwd_cur_rid", None) or "", round(_ts, 2)]
                except Exception:
                    pass
        if fp:
            hb["fwd_progress"] = fp
        _log_cursor, _new_logs = drain_new_logs(_log_cursor)
        if _new_logs:
            hb["logs"] = _new_logs[-300:]   # #logs: relay new log lines (capped per beat)
        if gpu:
            used, total = _gpu_mem_gb()
            hb["vram_used_gb"] = round(used, 2)
            hb["vram_total_gb"] = round(total, 2)
            # #vram-reusable: this process's VACANT torch allocator pool (reserved - allocated).
            # Device counters report it as USED, but any new torch allocation in THIS worker
            # reuses it first — the planner credits it back as free (it only returns to the OS
            # on a worker restart; on ROCm it can reach many GB after model churn).
            with contextlib.suppress(Exception):
                import torch
                hb["vram_reusable_gb"] = round(sum(
                    max(0, torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))
                    for i in range(torch.cuda.device_count())) / GB, 2)
            with contextlib.suppress(Exception):   # GPU compute utilization % (#46; needs pynvml)
                import torch
                hb["gpu_util"] = float(torch.cuda.utilization(0))
            if "gpu_util" not in hb:                # ROCm: torch.cuda.utilization needs the amdsmi
                _u = _rocm_gpu_util()               # python binding (absent in TheRock wheels) -> use
                if _u is not None:                  # the bundled rocm-smi CLI instead (AMD nodes 1:1)
                    hb["gpu_util"] = _u
        async with lock:
            writer.write(_enc(hb))
            await writer.drain()
        await asyncio.sleep(interval)


def _enable_keepalive(writer: asyncio.StreamWriter) -> None:
    """Turn on TCP keepalive so a vanished controller (half-open, no RST) is
    detected in ~1 min instead of the multi-minute TCP retransmit default."""
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    with contextlib.suppress(Exception):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)


async def session(args: argparse.Namespace, reg: dict, worker: Worker,
                  on_connected) -> None:
    """One connect → register → {heartbeat, command-reader} lifecycle. Returns on
    clean disconnect, raises on error — and ends the instant EITHER the heartbeat
    sender OR the command reader stops, so a dead/vanished controller is caught via
    heartbeat-send failure (or keepalive), not left blocking forever on readline()."""
    local_addr = _local_addr()
    try:
        reader, writer = await asyncio.open_connection(
            args.controller, args.control_port, local_addr=local_addr)
    except OSError:
        if local_addr is None:
            raise
        # The bound source may be invalid (NIC unplugged since startup, stale route).
        # Fall back to the OS-default route so a misselection never strands the worker;
        # clear the global so data-plane dials stop trying the bad source too. If THIS
        # also fails it's a real outage and propagates to the reconnect backoff.
        global _ROUTE_SRC
        reader, writer = await asyncio.open_connection(args.controller, args.control_port)
        print(f"[net] source-bind to {_ROUTE_SRC} failed; using OS-default route for all traffic")
        _ROUTE_SRC = ""
    _enable_keepalive(writer)
    wlock = asyncio.Lock()
    try:
        # #adopt: refresh the dynamic loaded-model inventory on EVERY (re)connect — a freshly
        # restarted controller uses it to re-ADOPT the models this worker kept across the
        # restart (old controllers just ignore the extra field).
        # #multi-controller: this session's tenant id. inventory() is filtered by it so we NEVER
        # hand a controller another controller's shards (which it would adopt, or free after the
        # 90s adopt grace). One controller => every shard is ours => byte-identical behaviour.
        tenant = f"{args.controller}:{args.control_port}"
        with contextlib.suppress(Exception):
            reg["loaded"] = worker.inventory(tenant)
        async with wlock:
            writer.write(_enc(reg))
            await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=15)
        if not line:
            raise ConnectionError("controller closed before registering")
        ack = json.loads(line.decode())
        if ack.get("type") != "registered":
            raise ConnectionError(f"unexpected ack: {ack}")
        node_id = ack["node_id"]
        # #adopt: does THIS controller re-adopt kept shards? Governs the session-teardown
        # choice below (keep vs drop). If we are holding shards kept for adoption but the
        # controller that came back CAN'T adopt (older code / rollback), drop them now —
        # it doesn't know about them, and they'd pin RAM/VRAM invisibly forever.
        worker._ctrl_adopts = bool(ack.get("adopt"))
        if not worker._ctrl_adopts and worker.shards:
            print(f"[adopt] controller lacks adoption — dropping {len(worker.shards)} kept model(s)")
            await worker.handle_unload(None, tenant)
        on_connected()  # registration succeeded -> reset reconnect backoff
        print(f"[+] registered as {node_id} on {args.controller}:{args.control_port} "
              f"(server {ack.get('server_version', '?')})")
        print(f"    {reg['hostname']}  {reg['device']}  {reg['usable_mem_gb']:.1f} GB "
              f"usable - {reg['free_disk_gb']:.1f} GB free disk")

        async def reply(obj: dict) -> None:
            async with wlock:
                writer.write(_enc(obj))
                await writer.drain()

        # #hop-recovery: expose the (wlock+_enc framed) control sender to the data plane so a dead
        # next-hop forward can push an unsolicited hop_error to the controller. Cleared in finally so a
        # stale closure from a dropped session is never used after reconnect.
        worker._ctrl_send = reply
        worker._node_id = node_id

        async def command_loop() -> None:
            while True:
                line2 = await reader.readline()
                if not line2:
                    return  # controller closed the connection cleanly
                msg = json.loads(line2.decode())
                mtype = msg.get("type")
                if mtype == "load":
                    try:
                        info = await worker.handle_load(msg, tenant)
                        # #1: echo model_id so the controller resolves THIS model's load future
                        # (a single shared future cross-resolved on a co-loaded node).
                        await reply({"type": "ready", "node_id": node_id,
                                     "model_id": msg.get("model_id"), **info})
                        if msg.get("kind") == "embedding":   # no layer range — whole encoder on one node
                            print(f"[load] embedding {msg.get('model_id')} "
                                  f"({info['loaded_bytes']/GB:.2f} GB)")
                        elif msg.get("kind") == "t2i":       # no layer range — whole pipeline on this node
                            print(f"[load] t2i {msg.get('model_id')} "
                                  f"({info['loaded_bytes']/GB:.2f} GB)")
                        else:
                            print(f"[load] stage {msg.get('stage')} "
                                  f"layers {msg['layer_start']}-{msg['layer_end']} "
                                  f"({info['loaded_bytes']/GB:.2f} GB, RAM-only)")
                    except Exception as exc:
                        await reply({"type": "error", "node_id": node_id,
                                     "model_id": msg.get("model_id"), "error": repr(exc)})  # #1: echo model_id
                        print(f"[load] FAILED: {exc!r}")
                elif mtype == "pack":      # #distributed-packing: pack a shard-cache unit for the controller
                    try:
                        info = await worker.handle_pack(msg)
                        await reply({"type": "packed", "node_id": node_id, **info})
                        print(f"[pack] {msg.get('unit')} -> {info['bytes']/1e6:.1f} MB "
                              f"({info['tensors']} tensors)")
                    except Exception as exc:
                        await reply({"type": "error", "node_id": node_id,
                                     "req_id": msg.get("req_id"), "error": repr(exc)})
                        print(f"[pack] FAILED: {exc!r}")
                elif mtype == "t2i_gen":
                    # #t2i-serve: renders take minutes — dispatch as a task so this loop keeps
                    # serving unload/ping; the handler replies keyed by req_id when done.
                    asyncio.create_task(worker.handle_t2i_gen(msg, reply))
                elif mtype == "tts_gen":
                    # #tts-serve: speech synthesis (esp. on CPU) takes many seconds — dispatch
                    # as a task so this loop keeps serving; handler replies keyed by req_id.
                    asyncio.create_task(worker.handle_tts_gen(msg, reply))
                elif mtype == "t2a_gen":
                    # #t2a-serve: a music render takes many seconds — dispatch as a task so this
                    # loop keeps serving unload/ping; handler replies keyed by req_id.
                    asyncio.create_task(worker.handle_t2a_gen(msg, reply))
                elif mtype == "unload":
                    await worker.handle_unload(msg.get("model_id"), tenant)
                elif mtype == "handoff":        # #handoff: move this node to another controller
                    res = await worker.handle_handoff(msg, tenant)
                    await reply({"type": "handoff_result", "node_id": node_id, **res})
                    if res.get("ok"):
                        return          # end the session; run() reconnects to the new controller
                    await reply({"type": "unloaded", "node_id": node_id,
                                 "model_id": msg.get("model_id")})
                    print(f"[unload] stage torn down (model_id={msg.get('model_id')})")
                elif mtype == "ping":
                    await reply({"type": "pong", "node_id": node_id})
                elif mtype == "free_memory":
                    # Controller asked us to EMPTY RAM (e.g. during a forced update/restart): drop
                    # any resident shards, then return freed heap + reclaimable OS cache to the OS so
                    # the box is left clean. Safe here — a forced update unloads all models first, so
                    # nothing is mmap-resident to evict. (#forced-update)
                    with contextlib.suppress(Exception):
                        await worker.handle_unload(None, tenant)   # drop OUR shards if any remain
                    freed = await asyncio.to_thread(_flush_os_cache)
                    with contextlib.suppress(Exception):
                        await reply({"type": "freed", "node_id": node_id, "free_gb": round(freed, 2)})
                    print(f"[free] controller requested RAM release -> {freed:.1f} GB free")
                elif mtype == "self_update":
                    # #fleet-update: forced fleet-wide deploy — run the self-update check NOW
                    # (apply changed files; restart only on a VERSION bump, the same rule as
                    # the idle poll). The controller sends this right after unloading models +
                    # free_memory, so forcing the idle gate true is safe here.
                    print("[update] controller requested an immediate self-update check")
                    asyncio.create_task(asyncio.to_thread(
                        _self_update_check, "client.py", (lambda: True)))
                elif mtype == "restart":
                    # Controller commanded a restart (fleet restart / forced deploy). Ack, then
                    # exit(42) so the supervisor (client.bat / systemd Restart=always) relaunches
                    # — dropping any resident shard cleanly. (#fleet-restart) With update:true
                    # (#fleet-update), stage the newest files FIRST so the relaunch comes back on
                    # fresh code immediately instead of waiting on the 15-min poll.
                    print("[restart] controller requested restart - exiting(42) for supervisor relaunch")
                    with contextlib.suppress(Exception):
                        await reply({"type": "restarting", "node_id": node_id})
                    if msg.get("update"):
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(_self_update_check, "client.py", (lambda: True))
                    os._exit(42)

        hb = asyncio.create_task(
            _heartbeat_loop(writer, wlock, node_id, args.heartbeat_interval, _using_gpu(args),
                            worker=worker))
        cmd = asyncio.create_task(command_loop())
        done, pending = await asyncio.wait({hb, cmd}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
            with contextlib.suppress(BaseException):
                await t
        for t in done:
            exc = t.exception()
            if exc:
                raise exc  # surfaces heartbeat-send failure -> reconnect
    finally:
        worker._ctrl_send = None   # #hop-recovery: drop the dead session's control sender
        # #adopt: if the controller advertises adoption, KEEP the loaded shards across the
        # link loss — a controller-only restart re-adopts them on reconnect (no re-stream).
        # Per-request transients still flush: staged multimodal embeds must not leak into a
        # fresh controller epoch (req_id restarts from 0). Every other teardown reason —
        # old controller, worker-side exit — keeps the full unload exactly as before.
        if getattr(worker, "_handoff", None) and worker.shards:
            # #handoff: never unload here — the shards are the whole point of the transfer.
            worker.pending_mm.clear()
            print(f"[handoff] keeping {len(worker.shards)} loaded model(s) for the new controller")
        elif getattr(worker, "_ctrl_adopts", False) and worker.shards:
            worker.pending_mm.clear()
            print(f"[adopt] control link lost — keeping {len(worker.shards)} loaded model(s) "
                  "for controller re-adoption")
        else:
            await worker.handle_unload(None, tenant)
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


def _hr(bps: float) -> str:
    """Human bytes/sec, fixed width-ish for the console panel."""
    for unit in ("B", "K", "M", "G"):
        if bps < 1024 or unit == "G":
            return f"{bps:4.0f}{unit}/s" if unit == "B" else f"{bps:4.1f}{unit}/s"
        bps /= 1024
    return f"{bps:.1f}G/s"


# code-split Inc 8: _fwd_watchdog_loop + _console_panel_loop live in worker_update.py now (VERBATIM).

def _controller_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    """Can we open the control-plane TCP port right now? Used ONLY to decide whether to fall back
    to #discovery at FIRST start — never in the reconnect loop, so a controller restart can't make
    a worker wander off to a different fleet mid-life."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


DISCOVERY_RETRY_S = 30.0    # #discovery: wait between broadcast attempts when nobody has answered


async def _resolve_controller(args: argparse.Namespace) -> None:
    """#discovery: settle args.controller/control_port BEFORE the first connect.

      controller_host "auto"/"discover"/""  -> broadcast-discover, RETRYING FOREVER
      a static host that is UNREACHABLE     -> discover ONCE as a rescue, then use what answers

    Discovery never gives up and never exits: a worker powered on before its controller (or during
    a controller restart) must simply keep asking and join the moment someone answers — that is what
    makes "clone, install deps, run" true even when the boxes boot in an arbitrary order. Every
    failed round prints an explicit NOT-CONNECTED line naming the reason, so a silent worker is
    never mistaken for a connected one.

    Any worker that can hear the broadcast is allowed to join — discovery is deliberately permissive
    (cluster_id defaults to unset = join whoever answers). Explicit --controller still wins whenever
    it is reachable, and static addressing remains the path for anything broadcast can't cross
    (subnets, VLANs, VPN)."""
    host = str(getattr(args, "controller", "") or "").strip()
    want = str(getattr(args, "cluster_id", "") or "")
    auto = host.lower() in ("auto", "discover", "")
    timeout = float(getattr(args, "discovery_timeout", 2.5))
    if not auto and _controller_reachable(host, args.control_port):
        return
    if not auto:
        # Static host configured but not answering: try discovery ONCE as a rescue, then fall
        # through to the normal reconnect loop (which already retries the static address).
        print(f"[discovery] controller {host}:{args.control_port} unreachable — "
              f"falling back to broadcast discovery", flush=True)
        got = await asyncio.to_thread(discover_controller, want, 0, timeout, True)
        if got:
            args.controller, args.control_port = got["host"], int(got["control_port"])
            return
        print(f"[discovery] nothing answered either; continuing with {host}:{args.control_port} "
              f"(the reconnect loop will keep retrying)", flush=True)
        return
    attempt = 0
    while True:
        attempt += 1
        got = await asyncio.to_thread(discover_controller, want, 0, timeout, True)
        if got:
            args.controller, args.control_port = got["host"], int(got["control_port"])
            print(f"[discovery] connected: controller {args.controller}:{args.control_port}"
                  + (f" ({got['name']})" if got.get("name") else ""), flush=True)
            return
        print(f"[discovery] NOT CONNECTED — no controller has replied to our broadcast yet "
              f"(attempt {attempt}); retrying in {DISCOVERY_RETRY_S:.0f}s. "
              f"Is the controller running on this LAN? Broadcast does not cross subnets/VLANs/VPN "
              f"— for those use: --controller <ip>", flush=True)
        await asyncio.sleep(DISCOVERY_RETRY_S)


async def run(args: argparse.Namespace) -> None:
    await _resolve_controller(args)   # #discovery: may rewrite args.controller/control_port
    reg = build_registration(args)
    # Pick + announce the LAN route BEFORE connecting: the chosen IP becomes our
    # control-connection source, hence the address the controller records and sends
    # all heavy data-plane traffic to. Prefer wired > Wi-Fi, USB 2.5GbE > built-in 1GbE.
    global _ROUTE_SRC
    route = select_route(args.controller, args.control_port)
    _ROUTE_SRC = route["ip"]
    # Co-located controller -> talk over loopback. Dialing our OWN external IP throws
    # WinError 64/1225 on Windows on any NIC/controller blip (beast's reconnect storms + the
    # gen ConnectionReset all traced to this). Register the real LAN IP as data_host so REMOTE
    # nodes (TP peers / pipeline hops) still reach us; the controller swaps in loopback for its
    # own connections back to us.
    if _controller_is_local(args.controller):
        if route["ip"]:
            reg["data_host"] = route["ip"]
        print(f"[net] controller {args.controller} is co-located -> loopback (127.0.0.1) for "
              f"controller traffic; remote nodes reach us at "
              f"{reg.get('data_host') or route['os_default'] or '(peer ip)'}")
        args.controller = "127.0.0.1"
        _ROUTE_SRC = ""
    worker = Worker(args)
    import threading as _wt   # #fwd-watchdog: backstop daemon for a forward stuck mid-op (see _fwd_watchdog_loop)
    _wt.Thread(target=_fwd_watchdog_loop, args=(worker,), daemon=True).start()
    # #74: live console status panel (opt-in IM_CONSOLE_PANEL=1, interactive TTY only — services /
    # redirected stdout keep plain line logging). Daemon thread, fully isolated from inference.
    if os.environ.get("IM_CONSOLE_PANEL") == "1" and sys.stdout.isatty():
        import threading
        threading.Thread(target=_console_panel_loop, daemon=True,
                         args=(worker, reg.get("hostname") or args.name or "worker")).start()
        print("[panel] live console panel ON (IM_CONSOLE_PANEL=1)")
    # Apply updates IMMEDIATELY even when shards are resident/serving (user policy: don't defer if
    # something is loaded — download, apply, restart). The ONLY guard is a build IN PROGRESS: restarting
    # mid-stream wedges the load (and a build isn't "loaded" yet). A resident/serving shard no longer
    # blocks the swap — the restart drops in-flight gens (recoverable; the controller re-streams).
    # #t2i: a live image render ALSO blocks the swap — unlike a text gen (recoverable: the
    # controller re-streams), a restart throws away a multi-minute render outright.
    asyncio.create_task(_self_update_loop(
        "client.py", lambda: not worker._building and not any(
            getattr(s, "kind", "") == "t2i" and getattr(s, "_gen_lock", None) is not None
            and s._gen_lock.locked() for s in list(worker.shards.values()))))
    asyncio.create_task(mem_maintenance_loop(worker, reg.get("hostname") or args.name or "worker"))
    print(f"InfiniteModel worker {VERSION} - {reg['hostname']} "
          f"({reg['device']}) device-mode={args.device}")
    if route["ip"]:
        tag = {"usb": "USB wired", "wired": "built-in wired",
               "wireless": "Wi-Fi"}.get(route["kind"], route["kind"])
        sp = f"{route['speed_mb']}Mb/s" if route["speed_mb"] > 0 else "speed?"
        print(f"[net] route to controller {args.controller}: {route['iface']} "
              f"{route['ip']} ({tag}, {sp}) — heavy data-plane traffic rides this NIC")
        if len(route["candidates"]) > 1:
            print("[net] interfaces (* = chosen):")
            for ln in _fmt_route(route):
                print(ln)
        if route["os_default"] and route["os_default"] != route["ip"]:
            print(f"[net] (OS default route would have used {route['os_default']}; "
                  f"binding to faster {route['ip']} instead)")
    else:
        print(f"[net] no LAN interface matched the controller subnet — letting the OS "
              f"pick the route (default source {route['os_default'] or '?'})")
    state = {"backoff": 1.0}

    def on_connected() -> None:
        state["backoff"] = 1.0  # a good connection resets the backoff

    while True:
        # #handoff: a controller moved this node to a peer — dial the new one from now on.
        _ho = getattr(worker, "_handoff", None)
        if _ho:
            worker._handoff = None
            args.controller, args.control_port = _ho[0], int(_ho[1])
            reg["loaded"] = []          # refreshed from the tenant-scoped inventory on register
            print(f"[handoff] re-targeting to controller {args.controller}:{args.control_port}")
            state["backoff"] = 1.0
        try:
            await session(args, reg, worker, on_connected)
        except (ConnectionRefusedError, ConnectionError, OSError, asyncio.TimeoutError) as exc:
            print(f"[!] controller {args.controller}:{args.control_port} unreachable ({exc})")
        except Exception as exc:  # pragma: no cover
            print(f"[!] connection lost ({exc!r})")
        else:
            print("[!] disconnected by controller")
        # exponential backoff (cap 30s) + jitter so all nodes don't stampede the
        # controller at once when it comes back up
        delay = state["backoff"] + random.uniform(0, min(state["backoff"], 3.0))
        print(f"[*] reconnecting in {delay:.1f}s")
        await asyncio.sleep(delay)
        state["backoff"] = min(state["backoff"] * 2, 30.0)


# ---------------------------------------------------------------------------
# Self-test (single-process load + correctness vs reference)
# ---------------------------------------------------------------------------

def run_self_test_load(model_id: str, attn: str = "eager", quant: str = "none",
                       device: str = "cpu") -> None:
    tune_cpu_threads()   # self-test runs CPU shards too; benefit from tuned threads + fp32 GEMM
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = cfg.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok("The capital of France is", return_tensors="pt").input_ids

    # Honor --device so the self-test exercises the SAME GPU path the worker uses (the test
    # used to hardcode CPU -> the int4/CUDA path went unverified on GPU boxes). _place_modules
    # maps cpu|gpu|cuda|auto|cpu+gpu|hybrid; a GPU mode with no CUDA falls back to CPU + a note.
    want_gpu = (device or "cpu").lower() in ("gpu", "cuda", "auto", "cpu+gpu", "hybrid")
    if want_gpu and not torch.cuda.is_available():
        print(f"  [notice] --device {device} requested a GPU but torch reports no CUDA "
              "device — running the self-test on CPU.")
    print(f"\nShard load self-test: {model_id}  ({n_layers} layers, attn={attn}, "
          f"quant={quant}, device={device})")
    full = Shard.from_hf(model_id, 0, n_layers, has_embed=True, has_head=True,
                         attn=attn, quant=quant, device=device)
    print(f"  placement: {getattr(full, 'placement', '?')}  |  footprint: {full.loaded_bytes/GB:.2f} GB")
    our_next = int(full.forward(ids)[0, -1].float().argmax())
    ref = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").eval()
    with torch.inference_mode():
        ref_next = int(ref(ids).logits[0, -1].float().argmax())
    print(f"  full shard:  {full.loaded_params/1e6:.1f}M params, {full.loaded_bytes/GB:.2f} GB")
    print(f"  our  next token: {our_next:>6}  {tok.decode([our_next])!r}")
    print(f"  ref  next token: {ref_next:>6}  {tok.decode([ref_next])!r}")
    print(f"  => {'MATCH' if our_next == ref_next else 'MISMATCH'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InfiniteModel worker client.")
    p.add_argument("--controller", default=load_config()["controller_host"],
                   help="controller host/IP, or 'auto' to find it by UDP broadcast "
                        "(default from config.json; an unreachable static host auto-discovers)")
    p.add_argument("--control-port", type=int, default=load_config()["control_port"])
    # #discovery: pin this worker to ONE fleet when several controllers share a LAN. Empty =
    # join whoever answers (the zero-config path).
    p.add_argument("--cluster-id", default=load_config().get("cluster_id", ""),
                   help="only join a controller advertising this cluster_id (default: any)")
    p.add_argument("--discovery-timeout", type=float, default=2.5,
                   help="seconds to wait for a controller to answer a discovery broadcast")
    p.add_argument("--data-port", type=int, default=50200,
                   help="local data-plane port for inter-stage tensors (default 50200)")
    p.add_argument("--os-reserve-gb", type=float, default=2.0,
                   help="memory to leave for this machine's OS (default 2.0)")
    p.add_argument("--heartbeat-interval", type=float, default=5.0)
    p.add_argument("--name", default=None, help="override reported hostname")
    p.add_argument("--ram", default="",
                   help="override RAM summary (e.g. '4x LPDDR5-5500') when "
                        "dmidecode needs a password / isn't available")
    p.add_argument("--device", default="cpu+gpu",
                   choices=["cpu", "gpu", "cuda", "auto", "cpu+gpu", "hybrid"],
                   help="where this worker runs its stage. Default 'cpu+gpu': fill the "
                        "VRAM budget on the GPU and spill any overflow to CPU RAM; falls "
                        "back to CPU if there's no CUDA device. 'auto': whole-GPU if it "
                        "fits else hybrid. 'cpu' forces CPU; 'gpu'/'cuda' whole stage on "
                        "GPU. The controller overrides this per-load ONLY when a dashboard "
                        "tier is toggled off (GPU-only or CPU-only); with both tiers on, "
                        "this default applies.")
    p.add_argument("--gpu-mem-gb", type=float, default=0.0,
                   help="VRAM budget for hybrid placement (0 = auto: ~85%% of free VRAM)")
    p.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"],
                   help="attention kernel. Default 'sdpa': torch scaled_dot_product_"
                        "attention, which itself auto-selects the fastest backend per "
                        "call (flash-attn / mem-efficient / math) for the device, dtype "
                        "and shapes — so it adapts automatically. 'eager': plain additive-"
                        "mask matmul, bit-exact and reproducible (for correctness checks) "
                        "but slower; pick it only when you need deterministic logits.")
    p.add_argument("--quant", default="none", choices=["none", "int8", "int4", "int2"],
                   help="weight quantization: 'none' (bf16, default), 'int8' "
                        "(per-channel weight-only — halves the footprint), 'int4' "
                        "(group-wise ~4.25-bit weight-only — ~1/4 footprint, for 200B+ "
                        "MoEs that won't fit at int8; small decode-speed cost), or 'int2' "
                        "(group-wise ~2.5-bit weight-only — ~1/6 footprint, dense models "
                        "only; a CAPACITY tier with visible quality loss). The "
                        "per-model quant in the controller's load message overrides this.")
    p.add_argument("--clean", action="store_true",
                   help="OPT-IN: purge cached models/chunks on startup. OFF by "
                        "default so a worker never wipes a model cache (esp. on the "
                        "controller box). Only pass this when you deliberately want "
                        "to reclaim disk.")
    p.add_argument("--no-clean", action="store_true",
                   help="deprecated no-op (cleanup is already off by default); "
                        "accepted for backward compatibility")
    p.add_argument("--no-cpu-fp32", action="store_true",
                   help="disable the CPU fp32-GEMM acceleration (transient fp32 upcast "
                        "for CPU-resident matmuls). ON by default — it makes CPU/hybrid "
                        "inference 3-5x faster on prefill without raising resident RAM. "
                        "Pass this only to A/B against the old bf16 CPU path. No effect "
                        "on GPU-resident modules either way.")
    p.add_argument("--self-test-load", action="store_true",
                   help="load a model as one shard, verify vs reference, and exit")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="model id for --self-test-load")
    return p.parse_args()


def _check_deps(args: argparse.Namespace) -> None:
    """Startup dependency check. Prints NOTICES only (never fatal): many fleet boxes
    are CPU-only or intentionally minimal, so a missing GPU/dep is fine — the worker
    still registers and contributes what it can. Just tells the user what to install
    if they DID want GPU/inference here."""
    import importlib.util
    have = lambda mod: importlib.util.find_spec(mod) is not None
    notes = []
    torch_ok = have("torch")
    if not torch_ok:
        notes.append("torch not installed - this worker can register but can't run "
                     "inference stages. For CPU: pip install torch ; for GPU: the CUDA build.")
    if not have("transformers"):
        notes.append("transformers not installed - needed to build model shards "
                     "(pip install transformers).")
    if not have("safetensors"):
        notes.append("safetensors not installed - needed to load served weights "
                     "(pip install safetensors).")
    wants_gpu = (args.device or "").lower() in ("gpu", "cuda", "auto", "cpu+gpu", "hybrid")
    if torch_ok and wants_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                notes.append(f"--device {args.device} asked for a GPU but torch reports no "
                             "CUDA device - this worker will run on CPU. (For GPU: a CUDA "
                             "torch build + NVIDIA driver. CPU-only is fine if that's intended.)")
        except Exception as exc:
            notes.append(f"couldn't query CUDA ({exc!r}); proceeding on CPU.")
    for n in notes:
        print(f"[notice] {n}")
    if not notes:
        print("[deps] torch + transformers + safetensors present"
              + (" ; CUDA ready" if (torch_ok and wants_gpu) else " ; CPU mode"))


def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    install_log_tee()   # #logs: mirror stdout/stderr into a ring; relayed to the controller on heartbeat
    args = parse_args()
    _check_deps(args)
    # Pin CPU thread count once at process start (physical cores) and report the CPU
    # fp32-GEMM policy. Harmless on GPU workers (torch threads just go unused there).
    if getattr(args, "no_cpu_fp32", False):
        worker_quant._CPU_FP32_GEMM = False   # Inc 10: canonical flag lives in worker_quant
    tune_cpu_threads()
    if args.self_test_load:
        run_self_test_load(args.model, args.attn, args.quant, args.device)
        return
    if not args.controller:
        raise SystemExit("--controller is required (or use --self-test-load)")
    if args.clean:   # opt-in only — never wipe a cache unless explicitly asked
        freed = cleanup_storage()
        if freed > 0.01:
            print(f"[clean] reclaimed {freed:.2f} GB of stale model/chunk cache")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[*] worker stopped")


# m4c153 code-split: register client namespace + inject it into the relocated Shard/Worker
# mixin modules so their (verbatim) bodies resolve their former globals. Module-level so every
# entry path (main / self-test) is covered before any Shard/Worker method runs. See state.py.
state.publish(globals())
state.bind(shard_build, shard_forward, worker_load, worker_net, worker_hw, worker_update)


if __name__ == "__main__":
    main()
