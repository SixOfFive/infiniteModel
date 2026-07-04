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
    "controller_host": "192.168.15.38",
    "http_port": 21434,
    "control_port": 50100,
    "data_port": 50101,
    "worker_data_port": 50200,
    "update_repo": "SixOfFive/infiniteModel",   # GitHub owner/name for self-update (public, no token)
    "update_branch": "main",
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


def repo_raw_url() -> str:
    """URL TEMPLATE (with a literal `{f}`) for fetching one repo file's raw bytes during self-update.
    Points at the PUBLIC GitHub repo's raw endpoint — no auth/token needed. Owner/name + branch come
    from config.json (`update_repo`, `update_branch`); e.g.
    https://raw.githubusercontent.com/SixOfFive/infiniteModel/main/{f}"""
    cfg = load_config()
    return (f"https://raw.githubusercontent.com/{cfg['update_repo']}/"
            f"{cfg['update_branch']}/{{f}}")
