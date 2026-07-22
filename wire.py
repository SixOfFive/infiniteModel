"""Shared wire-protocol helpers for InfiniteModel's data plane.

Imported by BOTH server.py (controller) and client.py (worker) so the tensor (un)packing
stays in ONE place — these two functions were byte-identical in both files and had to be
hand-synced on every transport change (e.g. the #P6 two-tensor frame), which is exactly the
drift this module removes. Kept dependency-light (only stdlib at import time; torch/numpy are
imported lazily inside the functions) so importing it never pulls heavy deps at module load.

NOTE: _read_frame/_write_frame are intentionally NOT here — the worker's versions also mutate
its global NET byte-accounting dict, so they differ between controller and worker.

This file is kept in sync on every node by the multi-file self-update (it is listed in each
side's EXTRA_UPDATE_FILES). server.py/client.py import it defensively (inline fallback) so a
node that has not yet fetched this file still runs during the brief self-update convergence.
"""


def _proportional_ints(total: int, weights, min_each: int = 0) -> list:
    """Split integer `total` across len(weights) buckets PROPORTIONAL to `weights`, each >=
    min_each, summing EXACTLY to total (largest-remainder method). Deterministic. Equal weights
    with total % n == 0 -> an even split (so the heterogeneous TP path reproduces the old 1/tp
    split exactly when capacities are equal)."""
    n = len(weights)
    tot = float(sum(weights)) or 1.0
    raw = [total * (w / tot) for w in weights]
    base = [max(min_each, int(r)) for r in raw]
    diff = total - sum(base)
    # rank by how much MORE each bucket deserves (raw-base): a bucket bumped UP to min_each has
    # raw-base < 0 and must NOT also grab an extra unit (that's the bug where the smallest node
    # out-allocated a bigger one). Recompute the order each step so it stays correct as base grows.
    k = 0
    while diff > 0:
        j = max(range(n), key=lambda i: raw[i] - base[i])
        base[j] += 1
        diff -= 1
        k += 1
    while diff < 0:                                  # over-allocated -> trim the biggest (keep >= min_each)
        for j in sorted(range(n), key=lambda i: base[i], reverse=True):
            if base[j] > min_each:
                base[j] -= 1
                diff += 1
                break
        else:
            break
    return base


def _tp_hetsplit(nh: int, nkv: int, hd: int, inter: int, group_size: int, weights: list) -> list:
    """HETEROGENEOUS tensor-parallel shard geometry: split a layer across len(weights) ranks
    PROPORTIONAL to each rank's capacity `weights` (e.g. usable VRAM GB), so a bigger GPU holds a
    bigger slice. Splits the KV heads (and GQA-proportional Q heads) and the FFN intermediate; the
    o_proj/down_proj row-parallel input dims follow q/idim. Returns a per-rank list of dicts:
    {q_off,q_len, kv_off,kv_len, idim_off,idim_len, q_heads, kv_heads} in FEATURE units (idim in
    raw features). Identical on server (slice) and client (build) since it's a pure function of the
    same inputs. Equal weights -> the exact even 1/tp split (backward compatible). idim aligned to
    group_size so each rank's int4 groups stay whole on the row-parallel down_proj input.

    KV-HEAD REPLICATION (#87): when n (=tp) > nkv, KV heads can't be split (you can't give each of
    >nkv ranks its own KV head), so each KV head is REPLICATED across (tp//nkv) ranks: every rank
    holds nh/tp Q heads (contiguous, even) + the 1 KV head its Q heads belong to (kv_off OVERLAPS
    across the ranks sharing it). grp = nh/nkv = (nh/tp)*(tp/nkv) guarantees every rank's Q heads
    fall in exactly one KV group, so GQA stays correct; the o_proj/down_proj all-reduce still sums
    to the full output (Q heads + FFN cols are disjoint across ranks). Requires tp % nkv == 0 and
    tp | nh (the controller enforces this and forces an EVEN split — het + replication not yet)."""
    grp = nh // nkv                                  # GQA: query heads per KV head
    n = len(weights)
    iu = _proportional_ints(max(1, inter // group_size), weights, min_each=1)  # FFN groups per rank
    out = []
    ioff = 0
    if n > nkv:                                       # KV-head replication path (wide TP)
        rpk = n // nkv                               # ranks sharing each replicated KV head
        qh = nh // n                                 # Q heads per rank (even)
        for r in range(n):
            kidx = r // rpk                          # this rank's (replicated) KV head index
            idim = (inter - ioff) if r == n - 1 else iu[r] * group_size
            out.append({"q_off": r * qh * hd, "q_len": qh * hd,
                        "kv_off": kidx * hd, "kv_len": hd,   # one KV head, replicated across rpk ranks
                        "idim_off": ioff, "idim_len": idim, "q_heads": qh, "kv_heads": 1})
            ioff += idim
        return out
    kv_heads = _proportional_ints(nkv, weights, min_each=1)          # >=1 KV head per rank, sum=nkv
    qoff = kvoff = 0
    for r in range(n):
        qh = kv_heads[r] * grp
        kvh = kv_heads[r]
        # idim from aligned groups; the LAST rank absorbs any inter % group_size remainder so the
        # per-rank idim always sums EXACTLY to inter (no dropped FFN columns on odd intermediates).
        idim = (inter - ioff) if r == n - 1 else iu[r] * group_size
        out.append({"q_off": qoff, "q_len": qh * hd, "kv_off": kvoff, "kv_len": kvh * hd,
                    "idim_off": ioff, "idim_len": idim, "q_heads": qh, "kv_heads": kvh})
        qoff += qh * hd
        kvoff += kvh * hd
        ioff += idim
    return out


def _set_keepalive(sock, idle_s: int = 15, interval_s: int = 3, count: int = 5) -> None:
    """Enable aggressive TCP keepalive on a long-lived data/mesh socket so an IDLE connection
    isn't aborted during the gap between LOAD and the first generate. On Windows an idle socket
    (and the TP all-reduce mesh, which sits idle through the whole multi-minute weight stream) is
    killed by the OS / a stateful firewall — surfacing as ConnectionReset / WinError 10053 / 64
    on the next write, or the TP mesh timing out ('mesh is broken'). Keepalive probes keep the
    connection 'active' so it survives the idle gap. Cross-platform + fully best-effort (never
    raises): Windows SIO_KEEPALIVE_VALS (on, idle_ms, interval_ms); Linux TCP_KEEPIDLE/INTVL/CNT.
    Accepts a raw socket OR an asyncio transport's socket (caller unwraps get_extra_info)."""
    import socket as _s
    if sock is None:
        return
    try:
        sock.setsockopt(_s.SOL_SOCKET, _s.SO_KEEPALIVE, 1)
    except Exception:
        pass
    if hasattr(sock, "ioctl") and hasattr(_s, "SIO_KEEPALIVE_VALS"):   # Windows
        try:
            sock.ioctl(_s.SIO_KEEPALIVE_VALS, (1, idle_s * 1000, interval_s * 1000))
        except Exception:
            pass
    else:                                                              # Linux / other POSIX
        for _opt, _val in (("TCP_KEEPIDLE", idle_s), ("TCP_KEEPINTVL", interval_s),
                           ("TCP_KEEPCNT", count)):
            _o = getattr(_s, _opt, None)
            if _o is not None:
                try:
                    sock.setsockopt(_s.IPPROTO_TCP, _o, _val)
                except Exception:
                    pass


def _pack_tensor(t):
    """Tensor -> ({dtype, shape}, raw bytes). Reinterprets the tensor's storage as uint8 so any
    dtype (bf16/fp32/int64/...) round-trips losslessly over the socket."""
    import torch  # noqa: F401
    t = t.detach().contiguous().cpu()
    meta = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}
    # A zero-element tensor (e.g. an empty-prompt prefill) has no storage to reinterpret —
    # emit empty bytes directly so we never .view() a 0-length buffer to a wider dtype.
    if t.numel() == 0:
        return meta, b""
    raw = t.view(torch.uint8).reshape(-1).numpy().tobytes()
    return meta, raw


def _unpack_tensor(meta, raw):
    """({dtype, shape}, raw bytes) -> tensor. Inverse of _pack_tensor."""
    import torch
    import numpy as np
    shape = meta["shape"]
    dt = getattr(torch, meta["dtype"])
    # A zero-element tensor (e.g. an empty-prompt prefill, shape [1,0]) packs to empty bytes,
    # and a 0-length uint8 buffer cannot be .view()'d to a wider dtype (stride(-1) must be 1).
    # Build the empty tensor directly so it round-trips instead of raising.
    if not raw or any(int(d) == 0 for d in shape):
        return torch.empty(shape, dtype=dt)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(dt).reshape(shape)


# --- #wire-caps + #ntensor-manifest (wire capability negotiation + N-tensor return frame) --------
# WIRE_CAPS: the wire-protocol capabilities THIS build's wire code provides. The worker advertises
# them in its register hello ('caps': [...]; worker_hw.build_registration reads this via getattr,
# so a node still carrying an OLD wire.py during per-file self-update convergence advertises
# nothing). The controller records them on the Node (registry.add) and exposes them via
# registry.node_caps(node_id) -> frozenset; a gated wire feature turns on for a model only when
# EVERY node in that model's chain advertises it. Mixed-version safety, by construction:
#   - old controller + new worker: registry.add whitelist-parses the register dict (reg.get per
#     known key), so the extra 'caps' key is simply ignored;
#   - new controller + old worker: no 'caps' field -> empty set -> every gated feature stays off.
# 'ntdiet' (#logits-diet): this build's worker_net/shard_forward understand the nt_mode/nt_clip/
# nt_k request-header directive and can reply with reduced head frames (NT_TOKEN_IDS or
# NT_TOPK_VALS+NT_TOPK_IDX) instead of the full-vocab logits row. Distinct from 'ntensor' because
# a 14ae638-era node advertises 'ntensor' yet ignores nt_mode (it would reply full NT_LOGITS) and
# an intermediate of that vintage rebuilds the next-hop header WITHOUT the nt_* keys — the
# controller only requests the diet when EVERY chain node advertises 'ntdiet', and still accepts
# a full-logits reply as the downgrade path (per-file self-update convergence can split wire.py
# from worker_net.py/shard_forward.py on one box).
# 'pipefill' (#pipefill): this build's worker loop (worker_net._data_inbound) + shard
# (shard_forward) are VALIDATED for chunked pipelined prefill — the controller streaming one
# prompt as SEVERAL back-to-back ids frames (chunk 0 reset=True at cache_position 0, chunks
# 1..C-1 reset=False at increasing cache_position, each with its OWN req_id) so downstream
# stages compute chunk i while upstream stages compute chunk i+1. No NEW header keys ride the
# frames (reset/cache_position are day-one fields, and the multi-token reset=False forward is
# the production spec-verify path), so the cap is a VINTAGE marker, not a format switch: the
# controller only streams a chunk burst when EVERY chain node is of a build whose inbound loop
# / KV-reset / mask semantics were audited for multi-frame prefill bursts (#wire-caps
# all-or-nothing, same doctrine as 'ntdiet'). A chain with any older node keeps the one-shot
# single-frame prefill, byte-identical to before.
# 'kvslots' (#kv-slots): this build's worker_net/shard_forward understand the per-request
# 'slot' header field — the shard keeps a DICT of KV caches keyed by slot id (one independent
# stream per slot) and routes every forward/crop to THE REQUEST'S slot, so several
# generations can interleave through one pipeline without sharing a cache. An old worker
# would write EVERY slot's tokens into its single self.kv (silent cross-request KV
# corruption), so the controller uses slots>1 for a model ONLY when every chain node
# advertises this cap — the gate is LOAD-time and all-or-nothing (a /load?kv_slots>1
# REFUSES a chain with any older node; engine_load._kvslots_cap_check). 'slot' absent ==
# slot 0 == the legacy single-cache behavior, byte-identical frames end to end.
WIRE_CAPS = ("ntensor", "ntdiet", "pipefill", "kvslots")

# #ntensor-manifest tensor kinds (u8 on the wire). 0/1 = full tensors; 2-4 carry the #logits-diet
# reduced head replies: 2 = greedy argmax token ids (int64, [q]); 3/4 = top-K candidate values
# (model dtype) + their token ids (int64), both [K], for controller-side sampling.
NT_LOGITS = 0
NT_HIDDEN = 1
NT_TOKEN_IDS = 2
NT_TOPK_VALS = 3
NT_TOPK_IDX = 4


def _pack_ntensor(parts):
    """#ntensor-manifest: [(kind:int, tensor), ...] -> (tmeta list, raw bytes). The raw payload
    is a compact binary manifest followed by the tensor payloads, back to back:

        u8 count | count x (u8 kind, u32 nbytes big-endian) | count payloads (concatenated)

    dtype/shape can't ride the fixed-width binary manifest, so they ride the frame's JSON header
    instead (hdr['tensors'] = the returned tmeta, POSITIONALLY aligned with the manifest
    entries) — the same self-describing JSON meta every legacy frame already uses. Tensor bytes
    are _pack_tensor's storage reinterpret, so any dtype round-trips losslessly. The frame is
    sent ONLY when the controller asked for it (request-header 'ntensor' flag) AND the worker
    advertised the 'ntensor' cap — the legacy one/two-tensor formats remain the default and are
    byte-identical to before."""
    import struct
    metas, blobs = [], []
    head = struct.pack(">B", len(parts))
    for kind, t in parts:
        meta, raw = _pack_tensor(t)
        metas.append(meta)
        blobs.append(raw)
        head += struct.pack(">BI", int(kind), len(raw))
    return metas, head + b"".join(blobs)


def _unpack_ntensor(tmeta, raw):
    """#ntensor-manifest: inverse of _pack_ntensor -> [(kind:int, tensor), ...] in manifest
    order. `tmeta` is the frame header's 'tensors' list (dtype/shape per entry, positional).
    Validates count and byte-span consistency so a malformed/short frame RAISES — the
    controller's _on_data turns that into the request future's exception, exactly like a
    legacy short frame."""
    import struct
    if not raw:
        raise ValueError("ntensor frame: empty payload")
    count = raw[0]
    if count != len(tmeta or []):
        raise ValueError(f"ntensor frame: manifest count {count} != header metas "
                         f"{len(tmeta or [])}")
    off = 1 + 5 * count                      # payloads start right after the manifest entries
    if len(raw) < off:
        raise ValueError("ntensor frame: truncated manifest")
    out, pos = [], 1
    for i in range(count):
        kind, nb = struct.unpack_from(">BI", raw, pos)
        pos += 5
        if off + nb > len(raw):
            raise ValueError(f"ntensor frame: tensor {i} spans past payload end")
        out.append((int(kind), _unpack_tensor(tmeta[i], raw[off:off + nb])))
        off += nb
    if off != len(raw):
        raise ValueError(f"ntensor frame: {len(raw) - off} trailing bytes after last tensor")
    return out


# --- Process log ring (#logs API) ---------------------------------------------------------------
# A bounded in-memory ring of recent stdout/stderr lines so a process's live log is exposable over
# HTTP: the controller serves its own via GET /logs, and each worker relays new lines on its
# heartbeat so the controller can serve them via GET /logs?node=<host> — no console/journal access
# to the worker box needed (the gap that made the beast-only #moe-offload bug hard to debug).
# install_log_tee() wraps stdout/stderr to ALSO append here; drain_new_logs(cursor) returns only
# lines added since the caller's cursor (the heartbeat relay); tail_logs(n) returns the last n.
import collections as _collections
import threading as _threading

_LOG_RING = _collections.deque(maxlen=4000)
_LOG_LOCK = _threading.Lock()
_LOG_TOTAL = 0   # monotonic count of lines EVER appended (relay cursor; survives ring eviction)


def _log_append(line: str) -> None:
    global _LOG_TOTAL
    with _LOG_LOCK:
        _LOG_RING.append(line)
        _LOG_TOTAL += 1


class _LogTee:
    """Wrap a text stream so every write ALSO lands (line-split) in _LOG_RING, then falls through
    all other attributes to the original stream (encoding, fileno, isatty, reconfigure, ...)."""

    def __init__(self, orig):
        self.__dict__["_orig"] = orig
        self.__dict__["_buf"] = ""

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            buf = self._buf + s
            if "\n" in buf:
                parts = buf.split("\n")
                self.__dict__["_buf"] = parts[-1]
                for ln in parts[:-1]:
                    _log_append(ln)
            else:
                self.__dict__["_buf"] = buf
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.__dict__["_orig"], name)


def install_log_tee() -> None:
    """Tee sys.stdout + sys.stderr into the log ring (idempotent)."""
    import sys
    if not isinstance(sys.stdout, _LogTee):
        sys.stdout = _LogTee(sys.stdout)
    if not isinstance(sys.stderr, _LogTee):
        sys.stderr = _LogTee(sys.stderr)


def drain_new_logs(last_total: int):
    """(new_total, lines) appended since last_total — only the NEW lines (capped to the ring).
    For the worker heartbeat relay so each beat ships just what's new."""
    with _LOG_LOCK:
        total = _LOG_TOTAL
        n = total - int(last_total or 0)
        if n <= 0:
            return total, []
        n = min(n, len(_LOG_RING))
        lines = list(_LOG_RING)[len(_LOG_RING) - n:]
    return total, lines


def tail_logs(n: int = 200):
    """The last n lines of THIS process's log ring (for the controller's own GET /logs)."""
    with _LOG_LOCK:
        return list(_LOG_RING)[-int(n):]


def _fuse_moe_experts(sd: dict, model) -> dict:
    """Convert a LEGACY per-expert MoE checkpoint into the FUSED expert tensors that
    transformers 5.x builds from_config, so load_state_dict fills them (else they stay on
    meta and the later .to(device) raises 'Cannot copy out of meta tensor').

    Target NAMES come from the MODEL (from_config), matched to the checkpoint's per-expert
    groups BY LAYER INDEX — so it works even when transformers renamed the MoE submodule
    (Mixtral `block_sparse_moe.*` checkpoint -> `mlp.*` in 5.x), including moving the router
    (`...gate.weight`). Per-expert names handled: OLMoE gate_proj/up_proj/down_proj and
    Mixtral w1(gate)/w3(up)/w2(down). Orientation is auto-detected against the expected
    shape; gate goes before up. No-op for dense / already-fused checkpoints.

    SHARED (wire.py) so the worker load path (client._install) AND the controller shard-cache
    compile (shards.compile_shards) fuse a per-expert MoE IDENTICALLY — the cache is then
    bit-identical to a cold load by construction (#moe-cache non-fused). Pure: re + torch (lazy)."""
    import re
    import torch
    # Expected fused targets in the model, keyed by layer index: {L: {prefix, gate_up_proj, down_proj}}
    exp: dict = {}
    expected = {n: p for n, p in model.named_parameters()}
    for n in expected:
        m = re.search(r"(.*\.layers\.(\d+)\..*\.experts)\.(gate_up_proj|down_proj)$", n)
        if m:
            exp.setdefault(int(m.group(2)), {})["prefix"] = m.group(1)
            exp[int(m.group(2))][m.group(3)] = n
    if not exp:
        return sd                       # model doesn't use fused experts — nothing to convert

    # Checkpoint per-expert weights, grouped by layer index.
    pat = re.compile(r"(.*\.layers\.(\d+)\..*\.experts)\.(\d+)\.(\w+)\.weight$")
    by_layer: dict = {}
    for k, t in list(sd.items()):
        m = pat.search(k)
        if not m:
            continue
        L = int(m.group(2))
        g = by_layer.setdefault(L, {"prefix": m.group(1), "projs": {}})
        g["projs"].setdefault(m.group(4), {})[int(m.group(3))] = t
    if not by_layer:
        return sd                       # checkpoint already fused

    def pick(projs, *names):
        for nm in names:
            if nm in projs:
                return projs[nm]
        return None

    def stack_match(tensors, want):
        fused = torch.stack(tensors, dim=0)
        if tuple(fused.shape) != tuple(want.shape):     # auto-detect orientation
            fused = torch.stack([t.t().contiguous() for t in tensors], dim=0)
        return fused.to(want.dtype)

    for L, g in by_layer.items():
        if L not in exp:
            continue
        projs = g["projs"]
        gate = pick(projs, "gate_proj", "w1")
        up = pick(projs, "up_proj", "w3")
        down = pick(projs, "down_proj", "w2")
        if not (gate and up and down):
            continue
        E = len(down)
        if "gate_up_proj" in exp[L]:
            tgt = exp[L]["gate_up_proj"]
            sd[tgt] = stack_match([torch.cat([gate[e], up[e]], dim=0) for e in range(E)],
                                  expected[tgt])           # gate THEN up
        if "down_proj" in exp[L]:
            tgt = exp[L]["down_proj"]
            sd[tgt] = stack_match([down[e] for e in range(E)], expected[tgt])
        # Router rename if the MoE submodule moved (Mixtral block_sparse_moe.gate -> mlp.gate).
        ckpt_parent = g["prefix"].rsplit(".experts", 1)[0]
        exp_parent = exp[L]["prefix"].rsplit(".experts", 1)[0]
        if ckpt_parent != exp_parent:
            rk, ek = f"{ckpt_parent}.gate.weight", f"{exp_parent}.gate.weight"
            if rk in sd and ek not in sd:
                sd[ek] = sd.pop(rk)
        for proj, experts in projs.items():        # drop the now-fused per-expert keys
            for e in experts:
                sd.pop(f"{g['prefix']}.{e}.{proj}.weight", None)
    return sd


# --- Central cluster configuration (#public-release) ----------------------------------------------
# Single source of truth for the cluster's hosts/ports + self-update source, so they are not baked
# into each module/script. The committed root `config.json` overrides these; the built-in defaults are
# the SAFE FALLBACK so a node that has not yet fetched config.json still runs unchanged. Self-update
# pulls from the PUBLIC GitHub repo's raw endpoint — NO auth/token needed — so there are no secrets
# here or in config.json, and the source is safe to publish.
_CONFIG_DEFAULTS = {
    # "auto" = find the controller by UDP broadcast (#discovery). This is the DEFAULT so a fresh
    # clone works with no config edit; put an IP here/in config.json for subnet/VLAN/VPN spans.
    "controller_host": "auto",
    "http_port": 21434,
    "control_port": 50100,
    "data_port": 50101,
    "worker_data_port": 50200,
    "update_repo": "SixOfFive/infiniteModel",   # GitHub owner/name for self-update (public, no token)
    "update_branch": "main",
    # #discovery: zero-config controller discovery (UDP broadcast query -> unicast reply).
    # controller_host "auto" discovers ALWAYS; a static host that is unreachable at startup
    # falls back to discovery ONCE (so a fresh clone whose config.json points at someone
    # else's LAN still just runs). cluster_id "" = unset = join whichever controller answers;
    # set it on BOTH controller and workers to pin a worker to one fleet on a shared LAN.
    "discovery_port": 50099,
    "cluster_id": "",
}
_CONFIG_CACHE = None


def load_config() -> dict:
    """Cluster config = built-in defaults overlaid with the root config.json next to this module.
    The ONE place hosts/ports are defined; everything else reads from here. Internal LAN addresses
    only — never secrets. Missing / malformed file -> defaults (the node still runs). Cached."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    import os
    import json
    cfg = dict(_CONFIG_DEFAULTS)
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "config.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            cfg.update({k: v for k, v in data.items()
                        if v is not None and not str(k).startswith("_")})
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[config] config.json unreadable ({exc!r}); using built-in defaults", flush=True)
    _CONFIG_CACHE = cfg
    return cfg


# --- #discovery: zero-config controller discovery -------------------------------------------------
# Shape B1 (client broadcast QUERY -> controller unicast REPLY), chosen over mDNS deliberately:
# dependency-free (no zeroconf), works on the Windows controller, and — because the controller
# answers from the interface that received the query — the reply's source IP is BY CONSTRUCTION the
# right same-subnet address for that worker. That sidesteps the Tailscale wrong-IP trap that made
# mDNS resolve a node to its 100.x CGNAT address instead of its LAN IP.
#
# Broadcast does NOT cross subnets/VLANs/VPN. That is why static config stays the fallback and
# remains fully supported: discovery is an ergonomic win for the single-LAN case, never a
# replacement for explicit addressing.
DISCOVERY_MAGIC = "infinitemodel-discovery-v1"


def local_ip_for(peer: str) -> str:
    """The local IPv4 the OS would use as SOURCE to reach `peer`. UDP connect() only sets the
    socket's default destination — no packet is sent — so this is a free, portable routing-table
    lookup. This is what makes the discovery reply carry the correct per-subnet controller IP on a
    multi-homed / Tailscale'd box instead of a hostname lookup's first (often wrong) answer."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer, 9))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def _local_ipv4s() -> set:
    """Every non-loopback IPv4 this host owns (best effort, stdlib only)."""
    import socket
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    d = local_ip_for("8.8.8.8")          # the default-route NIC, even if the hostname doesn't resolve
    if d:
        ips.add(d)
    return {ip for ip in ips if not ip.startswith("127.")}


def _broadcast_targets() -> list:
    """255.255.255.255 plus a per-NIC directed /24 broadcast. The limited broadcast only leaves the
    default-route interface, so a dual-homed box (e.g. MOBILE, LAN + Tailscale) would otherwise miss
    a controller on its second NIC. The /24 assumption is a heuristic; it costs one stray datagram
    when wrong and the limited broadcast still covers the normal case."""
    tgts = ["255.255.255.255"]
    for ip in sorted(_local_ipv4s()):
        b = ip.rsplit(".", 1)[0] + ".255"
        if b not in tgts:
            tgts.append(b)
    return tgts


def discover_controller(cluster_id: str = "", port: int = 0, timeout: float = 2.5,
                        verbose: bool = True) -> dict:
    """Broadcast a discovery query and return the chosen controller as
    {host, control_port, http_port, cluster_id, name, version} — or {} if nothing answered.

    Re-sends every ~0.5s for `timeout` seconds (one dropped datagram must not fail onboarding).
    Ambiguity policy: a worker silently joining the WRONG fleet is far worse than a slow start, so
    when several distinct controllers answer we prefer (1) an exact cluster_id match, then (2) a
    controller on our own /24, and we always LOG every responder so a mis-join is visible."""
    import json
    import socket
    import time
    cfg = load_config()
    port = int(port or cfg.get("discovery_port") or 50099)
    want = str(cluster_id or cfg.get("cluster_id") or "")
    query = json.dumps({"im": DISCOVERY_MAGIC, "q": 1, "cluster_id": want}).encode()
    targets = _broadcast_targets()
    found: dict = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.25)
        s.bind(("", 0))
        if verbose:
            print(f"[discovery] searching for a controller on udp/{port} "
                  f"(cluster_id={want or '<any>'}) …", flush=True)
        deadline, next_send = time.time() + max(0.5, timeout), 0.0
        while time.time() < deadline:
            if time.time() >= next_send:
                next_send = time.time() + 0.5
                for t in targets:
                    try:
                        s.sendto(query, (t, port))
                    except OSError:
                        pass       # an interface may refuse broadcast; the others still go
            try:
                raw, addr = s.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict) or msg.get("im") != DISCOVERY_MAGIC or not msg.get("r"):
                continue
            cid = str(msg.get("cluster_id") or "")
            if want and cid != want:
                continue           # a different fleet answered — ignore it
            host = str(msg.get("host") or addr[0])
            try:
                cport = int(msg.get("control_port") or cfg["control_port"])
            except (TypeError, ValueError):
                continue
            found.setdefault((host, cport), {
                "host": host, "control_port": cport,
                "http_port": int(msg.get("http_port") or cfg["http_port"]),
                "cluster_id": cid, "name": str(msg.get("name") or ""),
                "version": str(msg.get("version") or ""),
            })
    finally:
        s.close()
    if not found:
        if verbose:
            print("[discovery] no controller answered", flush=True)
        return {}
    cands = list(found.values())
    if len(cands) > 1 and verbose:
        print(f"[discovery] ⚠ {len(cands)} controllers answered — set `cluster_id` on this worker "
              f"to pin it to one fleet:", flush=True)
        for c in cands:
            print(f"[discovery]     {c['host']}:{c['control_port']} "
                  f"({c['name'] or '?'}, cluster_id={c['cluster_id'] or '<unset>'})", flush=True)
    pick = None
    if want:
        pick = next((c for c in cands if c["cluster_id"] == want), None)
    if pick is None:                                   # prefer a controller on our own /24
        mine = {ip.rsplit(".", 1)[0] for ip in _local_ipv4s()}
        pick = next((c for c in cands if c["host"].rsplit(".", 1)[0] in mine), None)
    if pick is None:
        pick = cands[0]
    if verbose:
        print(f"[discovery] using controller {pick['host']}:{pick['control_port']}"
              + (f" ({pick['name']})" if pick["name"] else ""), flush=True)
    return pick


def repo_raw_url() -> str:
    """URL TEMPLATE (with a literal `{f}`) for fetching one repo file's raw bytes during self-update.
    Points at the PUBLIC GitHub repo's raw endpoint — no auth/token needed. Owner/name + branch come
    from config.json (`update_repo`, `update_branch`); e.g.
    https://raw.githubusercontent.com/SixOfFive/infiniteModel/main/{f}"""
    cfg = load_config()
    return (f"https://raw.githubusercontent.com/{cfg['update_repo']}/"
            f"{cfg['update_branch']}/{{f}}")
