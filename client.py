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

VERSION = "0.2-m4c164"  # version tag only; full changelog -> CHANGELOG.md
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
# Fused-dequant int4 GEMM (torch tinygemm _weight_int4pack_mm): ~3.6x faster int4 decode by
# dequantizing INSIDE the matmul instead of re-expanding the whole weight every token. Built per
# QuantLinear4 at placement, self-checked vs the naive dequant, naive fallback on any mismatch /
# unsupported device. Off-switch: IM_FUSED_INT4=0.
_FUSED_INT4 = (os.environ.get("IM_FUSED_INT4", "1") != "0")
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


def _release_vram() -> None:
    """Return freed GPU memory to the pool after a shard is dropped. torch's caching
    allocator holds freed VRAM reserved inside the process, so without this the next
    model load sees ~0 free VRAM and spills its shard to RAM. Safe with other shards
    resident — empty_cache only frees UNUSED cached blocks, never in-use tensors. No-op
    on CPU-only workers / when torch isn't importable."""
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _release_ram(trim_working_set: bool = False) -> None:
    """Return freed CPU RAM to the OS after dropping shard(s). Python/torch keep freed
    heap inside the process: Linux glibc holds it until malloc_trim, Windows until the
    process working set is trimmed. Without this a node that held a big RAM-resident
    shard keeps REPORTING that RAM as used after unload, so the planner under-budgets it
    on the next load (the exact symptom on beast, which spilled ~46 GB of the 35B to RAM).
    The Windows trim evicts pages (they fault back on next access), so only do it when the
    worker is FULLY idle (trim_working_set=True). Best-effort; no-op on failure."""
    import gc
    gc.collect()
    with contextlib.suppress(Exception):
        import ctypes
        if sys.platform == "win32":
            if trim_working_set:                       # analog of malloc_trim on Windows
                k = ctypes.windll.kernel32
                k.SetProcessWorkingSetSize(k.GetCurrentProcess(),
                                           ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        else:
            ctypes.CDLL("libc.so.6").malloc_trim(0)    # return freed heap to the OS (glibc)


def _flush_os_cache() -> float:
    """Drop reclaimable OS cache + return freed heap so the next free-memory reading reflects
    TRUE free RAM for placement (#51). Linux: sync + drop_caches (passwordless sudo on the fleet
    workers). Windows: trim this process's working set. Best-effort (no-op without sudo/admin).
    MUST only be called when the worker is IDLE — dropping caches while a shard is mmap-resident
    would evict its pages and make the next inference fault them back from disk. Returns free GB."""
    import psutil
    _release_ram(trim_working_set=True)
    if sys.platform != "win32":
        with contextlib.suppress(Exception):
            import subprocess
            subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                           timeout=20, capture_output=True)
    return psutil.virtual_memory().available / GB


async def mem_maintenance_loop(worker, name: str, interval: float = 900.0) -> None:
    """Every `interval` (default 15 min), if the worker is IDLE (no resident shard), flush the OS
    cache and log the freed result, so the controller's next heartbeat sees accurate free RAM and
    the planner picks the best-fit hosts (#51). Skipped while serving (don't evict model pages)."""
    while True:
        await asyncio.sleep(interval)
        if getattr(worker, "shards", None):
            continue                               # serving — leave its mmap pages cached
        with contextlib.suppress(Exception):
            free = await asyncio.to_thread(_flush_os_cache)
            print(f"[mem] {name}: flushed OS cache (idle) — {free:.1f} GB free")


# ---------------------------------------------------------------------------
# Capability probe + registration
# ---------------------------------------------------------------------------

def detect_device() -> tuple[str, str, float]:
    try:
        import torch  # optional
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            _free, total = torch.cuda.mem_get_info(i)
            return f"cuda:{i}", name, total / GB
    except Exception:
        pass
    cpu = platform.processor() or platform.machine() or "cpu"
    return "cpu", cpu, psutil.virtual_memory().total / GB


def _gpu_mem_gb() -> tuple[float, float]:
    """(used_gb, total_gb) of GPU 0, or (0,0) if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return (total - free) / GB, total / GB
    except Exception:
        pass
    return 0.0, 0.0


def _using_gpu(args: argparse.Namespace) -> bool:
    """True when this worker is configured for, and has, a usable GPU."""
    if getattr(args, "device", "cpu") not in ("gpu", "cuda", "auto", "cpu+gpu", "hybrid"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def free_disk_gb() -> float:
    try:
        return shutil.disk_usage(HOME).free / GB
    except Exception:
        return 0.0


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


def _os_default_src(host: str, port: int) -> str:
    """Source IP the OS routing table would use to reach (host, port), discovered
    without sending a packet. '' if it can't be determined."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((host, port or 9))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


def _iface_kind(name: str) -> str:
    """Classify an interface by name: 'wireless' | 'usb' (USB-attached wired) |
    'wired' | 'virtual'. Spans Linux (enp/eno/enx/wlp), Windows ('Ethernet'/'Wi-Fi')
    and SteamOS naming."""
    n = name.lower()
    if any(k in n for k in ("loopback", "docker", "veth", "br-", "bridge",
                            "tailscale", "tun", "tap", "wg", "zt", "vethernet",
                            "virtual", "hyper-v", "vmware", "vbox", "bluetooth")):
        return "virtual"
    if n.startswith("wl") or "wi-fi" in n or "wifi" in n or "wireless" in n:
        return "wireless"
    if n.startswith("enx") or "usb" in n:        # Linux 'enx<MAC>' = USB Ethernet
        return "usb"
    if n.startswith(("eth", "en", "em")) or "ethernet" in n:
        return "wired"
    return "virtual"


def select_route(controller: str, control_port: int) -> dict:
    """Pick the local interface/IP this worker should use to reach the controller,
    preferring wired over wireless and the fastest wired link (USB 2.5GbE before
    built-in 1GbE). Returns {ip, iface, kind, speed_mb, candidates, os_default};
    ip='' means 'no LAN candidate found — let the OS choose'."""
    try:
        cip = socket.gethostbyname(controller)
    except Exception:
        cip = controller
    os_default = _os_default_src(cip, control_port)
    net24 = (".".join(cip.split(".")[:3]) + ".") if cip.count(".") == 3 else None

    cands = []
    try:
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except Exception:
        addrs, stats = {}, {}
    for name, alist in addrs.items():
        st = stats.get(name)
        if st is not None and not st.isup:
            continue
        ip4 = ""
        for a in alist:
            if a.family == socket.AF_INET and a.address and not a.address.startswith("127."):
                if net24 and a.address.startswith(net24):
                    ip4 = a.address          # prefer an address on the controller's /24
                    break
                if not ip4:
                    ip4 = a.address
        if not ip4:
            continue
        kind = _iface_kind(name)
        if kind == "virtual":
            continue
        speed = int(getattr(st, "speed", 0) or 0)
        eff = speed if speed > 0 else {"usb": 2500, "wired": 1000, "wireless": 300}.get(kind, 500)
        cands.append({"iface": name, "ip": ip4, "kind": kind, "speed_mb": speed, "eff": eff,
                      "on_subnet": bool(net24) and ip4.startswith(net24),
                      "is_default": ip4 == os_default})

    # Rank best-first: on the controller's subnet, then wired over wireless, then
    # fastest link, then a USB nudge, then the route the OS already prefers.
    cands.sort(key=lambda c: (c["on_subnet"], c["kind"] != "wireless", c["eff"],
                              c["kind"] == "usb", c["is_default"]), reverse=True)
    chosen = cands[0] if cands else None
    return {"ip": chosen["ip"] if chosen else "",
            "iface": chosen["iface"] if chosen else "",
            "kind": chosen["kind"] if chosen else "",
            "speed_mb": chosen["speed_mb"] if chosen else 0,
            "candidates": cands, "os_default": os_default}


def _controller_is_local(host: str) -> bool:
    """True if the controller runs on THIS machine (its IP is one of our local IPv4s, or
    loopback). When co-located we talk to it over 127.0.0.1 — robust + fastest — instead of
    our OWN external IP, which throws WinError 64/1225 on Windows during any NIC/controller
    blip (the cause of the beast worker's reconnect storms + the gen ConnectionReset)."""
    try:
        cip = socket.gethostbyname(host)
    except Exception:
        cip = host
    if cip == "localhost" or cip.startswith("127."):
        return True
    try:
        for _name, alist in psutil.net_if_addrs().items():
            for a in alist:
                if a.family == socket.AF_INET and a.address == cip:
                    return True
    except Exception:
        pass
    return False


def _fmt_route(r: dict) -> list:
    """Pretty per-interface lines for startup logging (chosen marked with '*')."""
    tags = {"usb": "wired-usb", "wired": "wired", "wireless": "wireless"}
    out = []
    for c in r["candidates"]:
        sp = f"{c['speed_mb']}Mb/s" if c["speed_mb"] > 0 else f"~{c['eff']}Mb/s?"
        star = "*" if (r["ip"] and c["ip"] == r["ip"]) else " "
        net = "" if c["on_subnet"] else " (off-subnet)"
        out.append(f"      {star} {c['iface']:<18} {c['ip']:<15} {sp:<9} "
                   f"{tags.get(c['kind'], c['kind'])}{net}")
    return out


def _fmt_ram_mods(mods: list) -> str:
    """Collapse (type, speed) module tuples into 'Nx TYPE-SPEED' groups joined by
    commas (e.g. '2x DDR5-5600', or '2x DDR5-5600, 1x DDR5-4800' when mixed).
    Shared by the Linux (dmidecode) and Windows (WMI) detect_ram paths."""
    if not mods:
        return ""
    from collections import Counter
    parts = []
    for (t, sp), n in Counter(mods).items():
        parts.append(f"{n}x {t}-{sp}" if sp else f"{n}x {t}")
    return ", ".join(parts)


# SMBIOS Memory Device "Type" codes (SMBIOS 3.6 spec, Table 78) -> friendly name.
# Win32_PhysicalMemory.SMBIOSMemoryType returns this raw byte verbatim; the older
# MemoryType field caps out at DDR4 (26) and reads 0/Unknown for DDR5+, so we prefer
# SMBIOSMemoryType. Codes verified against the SMBIOS spec + MS Learn (2026-06).
_SMBIOS_RAM = {20: "DDR", 21: "DDR2", 22: "DDR2", 24: "DDR3", 26: "DDR4",
               27: "LPDDR", 28: "LPDDR2", 29: "LPDDR3", 30: "LPDDR4",
               32: "HBM", 33: "HBM2", 34: "DDR5", 35: "LPDDR5", 36: "HBM3"}


def _detect_ram_windows() -> str:
    """RAM summary on Windows via WMI Win32_PhysicalMemory (PowerShell, no admin
    needed — unlike dmidecode). Emits one 'capacity|typecode|speed' line per module;
    ConfiguredClockSpeed is the actual negotiated MT/s (Speed is the rated max) so we
    prefer it, mirroring the Linux 'Configured Memory Speed' preference."""
    import subprocess
    ps = (
        'Get-CimInstance Win32_PhysicalMemory | ForEach-Object { '
        '$t = if ($_.SMBIOSMemoryType) { $_.SMBIOSMemoryType } else { $_.MemoryType }; '
        '$s = if ($_.ConfiguredClockSpeed) { $_.ConfiguredClockSpeed } else { $_.Speed }; '
        '"$($_.Capacity)|$t|$s" }'
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, timeout=12)
        out = r.stdout or ""
    except Exception:
        return ""
    mods: list[tuple[str, str]] = []
    for line in out.splitlines():
        cells = line.strip().split("|")
        if len(cells) != 3:
            continue
        cap, tcode, speed = (c.strip() for c in cells)
        if not cap or cap == "0":            # empty / unpopulated DIMM slot
            continue
        try:
            typ = _SMBIOS_RAM.get(int(tcode), "?")
        except ValueError:
            typ = "?"
        sp = "" if speed in ("", "0") else speed
        mods.append((typ, sp))
    return _fmt_ram_mods(mods)


def detect_ram(override: str = "") -> str:
    """Summarize RAM as e.g. '2x DDR4-2666' / '4x LPDDR5-5500'. Linux uses
    `sudo -n dmidecode` (passwordless sudo); Windows uses WMI Win32_PhysicalMemory
    via PowerShell. Returns '' if it can't (then the caller may pass --ram).
    override wins if given."""
    if override:
        return override
    if platform.system() == "Windows":
        return _detect_ram_windows()
    import subprocess
    try:
        r = subprocess.run(["sudo", "-n", "dmidecode", "-t", "memory"],
                           capture_output=True, text=True, timeout=8)
        text = r.stdout or ""
    except Exception:
        return ""
    mods: list[tuple[str, str]] = []
    cur: dict = {}

    def flush() -> None:
        size = cur.get("size", "")
        if size and "No Module" not in size:
            sp = (cur.get("cfg") or cur.get("speed") or "")
            sp = sp.replace("MT/s", "").replace("MHz", "").strip()
            if sp.lower() in ("unknown", ""):
                sp = ""
            mods.append((cur.get("type", "?"), sp))

    for raw in text.splitlines():
        s = raw.strip()
        if s == "Memory Device":
            flush(); cur = {}
        elif s.startswith("Size:"):
            cur["size"] = s.split(":", 1)[1].strip()
        elif s.startswith("Type:"):
            cur["type"] = s.split(":", 1)[1].strip()
        elif s.startswith("Configured Memory Speed:"):
            cur["cfg"] = s.split(":", 1)[1].strip()
        elif s.startswith("Speed:"):
            cur["speed"] = s.split(":", 1)[1].strip()
    flush()
    return _fmt_ram_mods(mods)


def build_registration(args: argparse.Namespace) -> dict:
    device, device_name, _dev_total = detect_device()
    # Report what this worker will actually use: 'cpu' forces CPU even if a GPU
    # exists; GPU modes report the CUDA device when present, else fall back to CPU.
    if getattr(args, "device", "cpu") == "cpu":
        device, device_name = "cpu", (platform.processor() or platform.machine() or "cpu")
    elif device == "cpu":  # a GPU mode was asked for but no CUDA available
        device_name += " (no CUDA → CPU)"
    total_gb = psutil.virtual_memory().total / GB
    usable_gb = max(0.0, total_gb - args.os_reserve_gb)
    reg = {
        "type": "register",
        "hostname": args.name or socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "device": device,
        "device_name": device_name,
        "total_mem_gb": round(total_gb, 2),
        "usable_mem_gb": round(usable_gb, 2),
        "free_disk_gb": round(free_disk_gb(), 2),
        "ram": detect_ram(args.ram),
        "cores": (os.cpu_count() or 1),
        "data_port": args.data_port,
        "client_version": VERSION,
        "wire": ("wire" in sys.modules),   # True once wire.py is imported (not the inline fallback)
    }
    if _using_gpu(args):
        reg["vram_total_gb"] = round(_gpu_mem_gb()[1], 2)
    return reg


def _enc(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


# ---------------------------------------------------------------------------
# Startup cleanup — reclaim space from stale models/chunks
# ---------------------------------------------------------------------------

def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, f))
    return total


def cleanup_storage() -> float:
    """Remove model artifacts a worker no longer needs (chunk-serving keeps weights
    in RAM, so the HF hub cache and any chunk dir are dead weight). Returns GB freed.
    OPT-IN only (runs via --clean) — off by default so a worker never wipes a model
    cache it might be sharing (a client on the controller box would otherwise delete
    the controller's full-model cache)."""
    freed = 0
    for target in (CHUNK_DIR,
                   os.path.join(HOME, ".cache", "huggingface", "hub")):
        if os.path.isdir(target):
            sz = _dir_size(target)
            shutil.rmtree(target, ignore_errors=True)
            freed += sz
    return freed / GB


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
                  load_config, repo_raw_url)


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
# CPU matmul acceleration (fp32 GEMM + thread tuning).
#
# THE PROBLEM. PyTorch CPU has NO fast bf16 GEMM — a bf16 matmul on CPU runs
# near-scalar (no MKL/OpenBLAS bf16 kernel), 1-2 orders of magnitude slower than
# the same matmul in fp32 (which hits a vectorized MKL/OpenBLAS GEMM). Our
# activations flow at self.dtype (bf16), so every CPU Linear — both the model's
# native nn.Linear and our QuantLinear (which dequants int8/int4 -> bf16) — pays
# the slow bf16 path. On a CPU-only or hybrid-spilled shard this makes a 70B int8
# produce ~0 tok/min. GPU is unaffected (real tensor-core bf16/int kernels).
#
# THE FIX. For a CPU-resident matmul, do a TRANSIENT fp32 upcast IN the forward:
# cast the activation (and the dequantized/transiently-upcast weight) to fp32,
# run F.linear in fp32 (fast GEMM), cast the result back to the working dtype.
# The resident weight STAYS bf16/int (the fp32 copy is per-call and freed after),
# so resident RAM is unchanged — the whole point of the fleet (fit big models).
#
# WHY SHAPE-ADAPTIVE (the threshold). Measured on this fleet's CPUs: the fp32
# win is huge for compute-bound matmuls (prefill, M=batch*seq large: 3-5x) but
# REVERSES for tiny-M single-token decode (M=1): a GEMV is memory-bandwidth
# bound, so reading the 2x-larger fp32 weight (incl. the one-time upcast/dequant
# of the whole weight) costs more than the slow-but-small bf16 GEMV saves. The
# crossover is ~12-16 rows. So we only upcast when the activation has at least
# _CPU_FP32_MIN_ROWS rows; below that the native bf16 path is kept (it's faster).
# This keeps decode fast AND makes prefill/large-batch fast — best of both.
# ---------------------------------------------------------------------------

# Min activation rows (flattened batch*seq) before a CPU matmul is worth the fp32
# upcast. Below this, bf16 GEMV is faster (memory-bound); above, fp32 GEMM wins.
# Conservative (16) so we never regress decode; prefill is always far above it.
_CPU_FP32_MIN_ROWS = 16
# Master switch (set False via --no-cpu-fp32 to A/B the old bf16 path).
_CPU_FP32_GEMM = True

# Some CPUs cannot run a NATIVE bf16 GEMM at all. On aarch64 without the ARM BF16 ISA
# extension (FEAT_BF16 — absent on e.g. Cortex-A55/A75, the Unisoc T310 tablet) PyTorch's
# oneDNN bf16 matmul does NOT fall back to fp32 the way x86 does; it raises
#   "mkldnn_matmul bf16 path needs a cpu with bf16 support" (ATen/native/mkldnn/Matmul.cpp).
# There the row-gated "keep bf16 for tiny-M decode" path above is not just slow — it CRASHES
# every decode step (rows=1), corrupting the shard's activations into garbage tokens. We probe
# this once at startup and, if unsupported, force the fp32 upcast for ALL CPU matmuls
# (_CPU_FP32_MIN_ROWS -> 1). No-op on x86, where the bf16 GEMM succeeds (just slower).
_CPU_BF16_GEMM_OK = None


def _cpu_bf16_gemm_ok() -> bool:
    """True iff this CPU's bf16 Linear is trustworthy for inference (probed once, cached).
    A CPU is rejected if a decode-shaped biased bf16 linear (addmm path) (a) RAISES — the
    aarch64-w/o-FEAT_BF16 hard 'bf16 path needs a cpu with bf16 support' check; (b) emits the
    oneDNN 'mkldnn_matmul failed, switching to ...' fallback WARNING — the tablet hits this and
    its fallback, while not crashing, is numerically degraded enough to derail a model; or
    (c) returns a result that diverges from the fp32 reference. Prints the probe result so the
    decision is visible at startup."""
    global _CPU_BF16_GEMM_OK
    if _CPU_BF16_GEMM_OK is None:
        _CPU_BF16_GEMM_OK = True
        try:
            import torch
            import warnings as _w
            g = torch.Generator().manual_seed(0)
            x = torch.randn(1, 512, generator=g)          # decode shape: rows=1, 2-D
            w = torch.randn(512, 512, generator=g)
            b = torch.randn(512, generator=g)
            ref = torch.nn.functional.linear(x, w, b)      # fp32 addmm (reference)
            with _w.catch_warnings(record=True) as wl:
                _w.simplefilter("always")
                got = torch.nn.functional.linear(          # bf16 addmm (the path under test)
                    x.bfloat16(), w.bfloat16(), b.bfloat16()).float()
            warned = any(("mkldnn" in str(m.message).lower()
                          or "bf16" in str(m.message).lower()) for m in wl)
            finite = bool(torch.isfinite(got).all())
            rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
            if warned or not finite or rel > 0.05:
                _CPU_BF16_GEMM_OK = False
            with contextlib.suppress(Exception):
                print(f"[cpu] bf16 GEMM probe: ok={_CPU_BF16_GEMM_OK} "
                      f"warned={warned} finite={finite} rel_err={rel:.4f}")
        except Exception as e:                             # raised -> definitely unusable
            _CPU_BF16_GEMM_OK = False
            with contextlib.suppress(Exception):
                print(f"[cpu] bf16 GEMM probe RAISED ({type(e).__name__}) -> forcing fp32")
    return _CPU_BF16_GEMM_OK


def _rows(x) -> int:
    """Flattened row count (batch*seq) of an activation [*, in_features]."""
    n = 1
    for d in x.shape[:-1]:
        n *= int(d)
    return n


def _cpu_fp32_worth(x) -> bool:
    """True when x is on CPU and big enough that an fp32 GEMM beats the bf16 path."""
    return (_CPU_FP32_GEMM and x.device.type == "cpu"
            and x.dtype != _torch_float32() and _rows(x) >= _CPU_FP32_MIN_ROWS)


_TORCH_F32 = None


def _torch_float32():
    global _TORCH_F32
    if _TORCH_F32 is None:
        import torch
        _TORCH_F32 = torch.float32
    return _TORCH_F32


def tune_cpu_threads() -> None:
    """Once, at process start: pin PyTorch CPU intra-op threads to the PHYSICAL core
    count (hyperthreads add overhead, not throughput, for GEMM-bound matmul). No-op
    if torch is absent. Honors a pre-set OMP_NUM_THREADS/MKL_NUM_THREADS (user/operator
    override) by not lowering an explicit env choice. Clamped to a sane range."""
    global _CPU_FP32_MIN_ROWS
    try:
        import torch
    except Exception:
        return
    # Respect an explicit operator override via env (don't stomp a deliberate choice).
    env_override = None
    for ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
        v = os.environ.get(ev)
        if v and v.strip().isdigit():
            env_override = int(v.strip())
            break
    if env_override:
        n = env_override
    else:
        try:
            n = psutil.cpu_count(logical=False) or 0
        except Exception:
            n = 0
        if not n:
            n = os.cpu_count() or 1
    n = max(1, min(int(n), 128))   # clamp: never 0, never absurd
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    # interop (inter-op task parallelism) gains nothing for a single sequential
    # forward and oversubscribes cores; keep it small. Must be set before any
    # parallel work starts, so wrap in suppress (raises if the pool is already up).
    with contextlib.suppress(Exception):
        torch.set_num_interop_threads(min(2, n))
    # A CPU that can't run a native bf16 GEMM (e.g. aarch64 w/o FEAT_BF16) would CRASH on the
    # tiny-M decode path rather than run it slow; force the fp32 upcast for EVERY CPU matmul
    # there (rows >= 1) so decode stays correct. No-op on x86. See _cpu_bf16_gemm_ok.
    if _CPU_FP32_GEMM and not _cpu_bf16_gemm_ok():
        _CPU_FP32_MIN_ROWS = 1
        with contextlib.suppress(Exception):
            print("[cpu] native bf16 GEMM unsupported here (aarch64 w/o bf16 ISA?) - "
                  "forcing fp32 upcast for ALL CPU matmuls")
    with contextlib.suppress(Exception):
        print(f"[cpu] torch intra-op threads set to {torch.get_num_threads()} "
              f"(physical cores{'/env' if env_override else ''}); fp32 CPU GEMM "
              f"{'on' if _CPU_FP32_GEMM else 'off'} (min {_CPU_FP32_MIN_ROWS} rows)")


def _wrap_cpu_linear_fp32(lin) -> None:
    """Wrap a CPU-resident native nn.Linear's forward so its matmul runs in fp32.
    The bf16 weight stays resident (source of truth); per call we transiently upcast
    activation+weight to fp32, F.linear (fast MKL/OpenBLAS GEMM), cast back. Only
    fires for compute-bound shapes (>= _CPU_FP32_MIN_ROWS rows); tiny-M decode falls
    through to the original bf16 forward (faster — see module header). Idempotent."""
    import torch
    import torch.nn.functional as F
    if getattr(lin, "_im_fp32_wrapped", False):
        return
    orig_forward = lin.forward

    def fp32_forward(x):
        w = lin.weight
        if (_CPU_FP32_GEMM and w.device.type == "cpu" and x.device.type == "cpu"
                and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
            out_dtype = x.dtype
            b = lin.bias
            y = F.linear(x.to(torch.float32), w.to(torch.float32),
                         None if b is None else b.to(torch.float32))
            return y.to(out_dtype)
        return orig_forward(x)

    lin.forward = fp32_forward
    lin._im_fp32_wrapped = True


def _accelerate_cpu_linears(module) -> None:
    """Wrap every CPU-resident native nn.Linear under `module` for the fp32 GEMM path.
    Skips GPU-resident Linears (already fast tensor-core kernels — must stay untouched)
    and our QuantLinear/QuantLinear4/Packed4Tensor3D (they handle fp32 in their own
    dequant). Called per-module from _place_modules AFTER placement, so it sees the
    final device of each weight — composing automatically with hybrid cpu+gpu spill
    and TP-v2 reduced-dim modules (it wraps whatever Linears ended up on CPU)."""
    if module is None:
        return
    from torch import nn
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            try:
                on_cpu = sub.weight.device.type == "cpu"
            except Exception:
                on_cpu = False
            if on_cpu:
                _wrap_cpu_linear_fp32(sub)


# ---------------------------------------------------------------------------
# int8 weight-only quantization (opt-in, --quant int8). Halves the weight
# footprint (RAM + VRAM) so bigger models fit. Per-output-channel symmetric
# int8; the original dtype weight is reconstructed on the fly in forward
# (qweight.to(dtype) * scale), so it's a memory win, not a speed win — for a
# model that already fits, prefer bf16. Self-contained: no external quant libs.
# ---------------------------------------------------------------------------

_QUANT_LINEAR = None


def _quant_linear_cls():
    global _QUANT_LINEAR
    if _QUANT_LINEAR is None:
        import torch
        from torch import nn
        import torch.nn.functional as F

        class QuantLinear(nn.Module):
            def __init__(self, qweight, scale, bias):
                super().__init__()
                self.register_buffer("qweight", qweight)   # int8 [out, in]
                self.register_buffer("scale", scale)        # dtype [out, 1]
                self.bias = bias                             # Parameter or None

            def forward(self, x):
                # CPU compute-bound path: dequant DIRECTLY to fp32 and run a fast fp32
                # GEMM (the dequant is paid either way, so targeting fp32 is free here and
                # the GEMM is the win). Tiny-M decode (or any GPU tensor) keeps the original
                # dequant-to-x.dtype path — bf16 GEMV is faster there (memory-bound). See the
                # CPU-matmul module header for the threshold rationale.
                if (_CPU_FP32_GEMM and x.device.type == "cpu"
                        and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
                    w = self.qweight.to(torch.float32) * self.scale.to(torch.float32)
                    b = self.bias
                    y = F.linear(x.to(torch.float32), w,
                                 None if b is None else b.to(torch.float32))
                    return y.to(x.dtype)
                w = self.qweight.to(x.dtype) * self.scale    # dequant one weight matrix
                return F.linear(x, w, self.bias)

        _QUANT_LINEAR = QuantLinear
    return _QUANT_LINEAR


def _quantize_linear(lin):
    """nn.Linear -> int8 weight-only QuantLinear (per-output-channel scale)."""
    import torch
    QL = _quant_linear_cls()
    W = lin.weight.data
    scale = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    qW = (W / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return QL(qW, scale.to(W.dtype), lin.bias)


def _quantize_int8_(module) -> None:
    """Recursively replace every nn.Linear under `module` with a QuantLinear."""
    from torch import nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear(child))
        else:
            _quantize_int8_(child)


# ---------------------------------------------------------------------------
# int4 weight-only quantization (opt-in, --quant int4). ~4.25 bits/weight
# (2 nibbles/byte + group scale/zero), so a model takes ~1/4 the RAM/VRAM of
# bf16 — the route for 200B+ MoEs that won't fit even at int8. Group-wise
# ASYMMETRIC (per-output-channel, group_size=128): w ~= (q - zero) * scale.
# The whole matrix is dequantized to the activation dtype on the fly in
# forward (memory win, not speed). CPU+GPU, pure torch, no external quant libs.
# The lm_head is left bf16 (logit-sensitive); only decoder Linears are quantized.
# ---------------------------------------------------------------------------

_QUANT_LINEAR4 = None
_INT4_GROUP = 128


def _quant4_linear_cls():
    global _QUANT_LINEAR4
    if _QUANT_LINEAR4 is None:
        import torch
        from torch import nn
        import torch.nn.functional as F

        class QuantLinear4(nn.Module):
            def __init__(self, qweight, scale, zero, bias, in_features, group_size):
                super().__init__()
                self.register_buffer("qweight", qweight)  # uint8 [out, in_pad//2] (2 nibbles/byte)
                self.register_buffer("scale", scale)       # dtype [out, n_groups]
                self.register_buffer("zero", zero)         # dtype [out, n_groups]
                self.bias = bias                            # Parameter or None (bf16)
                self.in_features = in_features
                self.group_size = group_size

            def _dequant(self, dtype):
                qw = self.qweight
                out = qw.shape[0]
                lo = (qw & 0x0F).to(torch.int16)           # even input columns
                hi = (qw >> 4).to(torch.int16)             # odd input columns
                q = torch.stack((lo, hi), dim=2).reshape(out, -1)   # [out, in_pad]
                ng = self.scale.shape[1]
                G = self.group_size
                qf = q.reshape(out, ng, G).to(dtype)
                w = (qf - self.zero.to(dtype).unsqueeze(2)) * self.scale.to(dtype).unsqueeze(2)
                return w.reshape(out, ng * G)[:, :self.in_features].contiguous()

            def prepare_fused(self):
                # Build torch's tinygemm fused-int4 weight ONCE (now that this module is on its FINAL
                # device). Decode re-dequants the whole weight every token in the naive path — the 3.6x
                # int4 slowdown; the fused op dequants inside the GEMM. Device-gated (CUDA sm80+ / CPU
                # op present), converted from OUR group-wise asymmetric format with NO re-quant
                # (w=(q-zero)*S == kernel's (q-8)*S + (8-zero)*S), then SELF-CHECKED vs the naive
                # dequant on a random input. Any mismatch / unsupported device / error -> keep naive.
                # Frees qweight on success (the packed mat2 replaces it -> int4 memory stays flat).
                if not _FUSED_INT4 or getattr(self, "_fused", None) is not None \
                        or getattr(self, "_fused_tried", False) or self.qweight is None:
                    return
                self._fused_tried = True
                dev = self.qweight.device
                aten = torch.ops.aten
                if dev.type == "cuda":
                    try:
                        ok = (torch.cuda.get_device_capability(dev) >= (8, 0)
                              and hasattr(aten, "_weight_int4pack_mm")
                              and hasattr(aten, "_convert_weight_to_int4pack"))
                    except Exception:
                        ok = False
                elif dev.type == "cpu":
                    ok = (hasattr(aten, "_weight_int4pack_mm_for_cpu")
                          and hasattr(aten, "_convert_weight_to_int4pack_for_cpu"))
                else:
                    ok = False
                if not ok:
                    return
                try:
                    qw = self.qweight
                    out = qw.shape[0]
                    G = self.group_size
                    ng = self.scale.shape[1]
                    in_pad = ng * G
                    q = torch.stack(((qw & 0x0F), (qw >> 4)), dim=2).reshape(out, in_pad)  # 0..15
                    S = self.scale.float()                                  # [out, ng]
                    Z = (8.0 - self.zero.float()) * S                       # int-zero -> float midpoint
                    sz = torch.cat([S.reshape(out, ng, 1), Z.reshape(out, ng, 1)], dim=2)
                    sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)  # [ng, out, 2]
                    if dev.type == "cuda":
                        # CUDA pack wants HIGH-nibble = even col (ours is LOW-even) -> re-nibble
                        packed = ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8).contiguous()
                        mat2 = aten._convert_weight_to_int4pack(packed, 8)   # innerKTiles=8
                        op = aten._weight_int4pack_mm
                        sz = sz.to(dev)
                    else:
                        mat2 = aten._convert_weight_to_int4pack_for_cpu(
                            q.to(torch.int32).contiguous(), 8)
                        op = aten._weight_int4pack_mm_for_cpu
                    # self-check: fused vs naive on a random input (catches zero-point/nibble/pack bugs
                    # that would silently corrupt logits — they are NOT exceptions).
                    xt = torch.randn(4, self.in_features, device=dev, dtype=torch.bfloat16)
                    xk = xt if in_pad == self.in_features else F.pad(xt, (0, in_pad - self.in_features))
                    yf = op(xk, mat2, G, sz).float()
                    yn = F.linear(xt, self._dequant(torch.bfloat16)).float()
                    rel = ((yf - yn).abs().mean() / (yn.abs().mean() + 1e-6)).item()
                    if rel < 0.05:
                        self._fused = (mat2, sz, op, in_pad)
                        self.qweight = None        # packed mat2 is now authoritative; free the source
                    else:
                        print(f"[int4] fused self-check rel={rel:.3f} on {dev} -> naive path",
                              flush=True)
                except Exception as exc:
                    print(f"[int4] fused prepare failed on {dev} ({exc!r}) -> naive path", flush=True)

            def forward(self, x):
                fz = getattr(self, "_fused", None)
                if fz is not None:
                    # fused-dequant int4 GEMM (2D only): flatten, bf16 activations, restore shape.
                    mat2, sz, op, in_pad = fz
                    xq = x.reshape(-1, self.in_features)
                    if xq.dtype != torch.bfloat16:
                        xq = xq.to(torch.bfloat16)
                    if in_pad != self.in_features:
                        xq = F.pad(xq, (0, in_pad - self.in_features))
                    y = op(xq.contiguous(), mat2, self.group_size, sz).reshape(*x.shape[:-1], -1)
                    y = y.to(x.dtype)
                    return y if self.bias is None else y + self.bias.to(y.dtype)
                # naive fallback: CPU compute-bound path dequants to fp32 + fp32 GEMM (see QuantLinear
                # / CPU-matmul header); int4 unpack+dequant is paid regardless, so the fp32 weight is
                # free and the fp32 GEMM is the win. Decode/GPU keep x.dtype.
                if (_CPU_FP32_GEMM and x.device.type == "cpu"
                        and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
                    b = self.bias
                    y = F.linear(x.to(torch.float32), self._dequant(torch.float32),
                                 None if b is None else b.to(torch.float32))
                    return y.to(x.dtype)
                return F.linear(x, self._dequant(x.dtype), self.bias)

        _QUANT_LINEAR4 = QuantLinear4
    return _QUANT_LINEAR4


def _quantize_linear4(lin, group_size: int = _INT4_GROUP):
    """nn.Linear -> group-wise asymmetric int4 QuantLinear4 (2 nibbles/byte)."""
    import torch
    import torch.nn.functional as F
    QL = _quant4_linear_cls()
    W = lin.weight.data
    out, in_f = W.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    Wp = F.pad(W, (0, in_pad - in_f)) if in_pad != in_f else W
    Wg = Wp.reshape(out, ng, G).float()
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)             # [out, ng]
    zero = torch.round(-wmin / scale).clamp(0, 15)            # [out, ng]
    q = torch.round(Wg / scale.unsqueeze(2) + zero.unsqueeze(2)).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out, in_pad)                                 # [out, in_pad]
    qpacked = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()    # [out, in_pad//2] uint8
    dt = W.dtype
    return QL(qpacked, scale.to(dt), zero.to(dt), lin.bias, in_f, G)


def _quantize_int4_(module) -> None:
    """Recursively replace every nn.Linear under `module` with a QuantLinear4."""
    from torch import nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear4(child))
        else:
            _quantize_int4_(child)


# --- int4 for FUSED MoE experts (3D gate_up_proj/down_proj nn.Parameters) -------------------
# Modern transformers MoE blocks (MiniMaxM2Experts, Glm4MoeNaiveMoe, Qwen3-MoE, ...) store all
# experts as ONE 3D Parameter [E, out, in] and index it per ROUTED expert in forward
# (self.gate_up_proj[expert_idx]). These are NOT nn.Linear, so _quantize_int4_ misses them —
# yet they are ~90% of a big MoE's params. We replace each 3D Parameter with a Packed4Tensor3D
# that dequantizes ONLY the indexed expert on the fly, so the routed forward keeps its 1/4
# footprint without ever materializing all experts.

_PACKED4_3D = None


def _packed4_3d_cls():
    global _PACKED4_3D
    if _PACKED4_3D is None:
        import torch
        from torch import nn

        class Packed4Tensor3D(nn.Module):
            def __init__(self, qweight, scale, zero, in_features, group_size):
                super().__init__()
                self.register_buffer("qweight", qweight)   # uint8 [E, out, in_pad//2]
                self.register_buffer("scale", scale)        # dtype [E, out, ng]
                self.register_buffer("zero", zero)          # dtype [E, out, ng]
                self.in_features = in_features
                self.group_size = group_size

            def __getitem__(self, e):
                # Returns the dequantized bf16 weight for one expert; the EAGER MoE host
                # forward then does F.linear(bf16_activation, this_weight). We deliberately
                # keep this bf16 (not fp32): the host contract requires the returned weight
                # match the bf16 activation dtype (an fp32 return would raise in its F.linear),
                # and intercepting the host matmul is too fragile. The fp32 CPU win still
                # applies to the layer's attention / router / shared-expert Linears (nn.Linear
                # / QuantLinear4), just not the per-routed-expert fused GEMM — which is already
                # the "correctness/memory > speed" eager path (see _quantize_experts4_).
                e = int(e)
                qw = self.qweight[e]                         # [out, in_pad//2]
                out = qw.shape[0]
                lo = (qw & 0x0F).to(torch.int16)
                hi = (qw >> 4).to(torch.int16)
                q = torch.stack((lo, hi), dim=2).reshape(out, -1)   # [out, in_pad]
                ng = self.scale.shape[2]
                G = self.group_size
                dt = self.scale.dtype
                qf = q.reshape(out, ng, G).to(dt)
                w = (qf - self.zero[e].to(dt).unsqueeze(2)) * self.scale[e].to(dt).unsqueeze(2)
                return w.reshape(out, ng * G)[:, :self.in_features].contiguous()

        _PACKED4_3D = Packed4Tensor3D
    return _PACKED4_3D


# --- MoE intra-layer offload (#moe-offload): attention on GPU, routed experts in CPU RAM ----------
# The llama.cpp --override-tensor "experts=CPU" strategy, intra-layer: a MoE layer's routed-expert
# FFN is ~90% of its bytes but each token activates only k of E experts, while the token-mixer
# (attention) + norms are small, used EVERY token, and latency-critical. Today placement is
# whole-layer (a spilled MoE layer drags its hot attention to CPU with the experts). With this on,
# a split layer keeps attention+norms on GPU and leaves the MoE block (router+experts+shared) on
# CPU. Gated to int4 (Packed4Tensor3D) experts only — those are always heap (no mmap), so unload
# reclaim is unaffected; bf16 experts (mmap Parameters) are left to the whole-layer path.
_MOE_BRIDGE = None


def _moe_bridge_cls():
    global _MOE_BRIDGE
    if _MOE_BRIDGE is None:
        import torch
        from torch import nn

        class _MoEDeviceBridge(nn.Module):
            """Wraps a layer's MoE block so it executes on CPU (where its router+experts+shared live)
            while the rest of the layer runs on GPU. forward: move the incoming hidden GPU->CPU, run
            the wrapped block on CPU, move the output(s) back to the input device. The big routed-
            expert GEMM stays in CPU RAM; attention stays on GPU. Hidden at decode is a few KB, so the
            per-layer round-trip is negligible (validated by the measure-first gate)."""

            def __init__(self, block, cpu_dev):
                super().__init__()
                self.block = block          # registered child; its params/buffers stay on CPU
                self._cpu = cpu_dev

            def forward(self, hidden_states, *args, **kwargs):
                dev = hidden_states.device
                h = hidden_states.to(self._cpu)
                a2 = tuple(x.to(self._cpu) if torch.is_tensor(x) else x for x in args)
                k2 = {kk: (vv.to(self._cpu) if torch.is_tensor(vv) else vv)
                      for kk, vv in kwargs.items()}
                out = self.block(h, *a2, **k2)
                if torch.is_tensor(out):
                    return out.to(dev)
                if isinstance(out, tuple):
                    return tuple(o.to(dev) if torch.is_tensor(o) else o for o in out)
                return out

        _MOE_BRIDGE = _MoEDeviceBridge
    return _MOE_BRIDGE


def _find_moe_block(layer):
    """Locate the routed-expert MoE block within one decoder layer: the DIRECT child of `layer` that
    holds the experts. Detected by an `experts` attribute (the routed-expert container — present on
    per-expert arches like Mixtral `block_sparse_moe.experts` / OLMoE AND fused arches like
    Qwen3-MoE/MiniMax) or, as a fallback, a fused 3D `gate_up_proj`/`down_proj` (Packed4Tensor3D or a
    raw 3D Parameter) anywhere under the child. Returns (attr_name, block_module), or (None, None) for
    a DENSE layer (its MLP has no experts) so a dense layer is never wrapped. The block is the whole
    sparse-MoE module (router gate + routed experts + any shared expert); wrapping it sends all of
    those to CPU and keeps only the token-mixer + norms on GPU. Splittability by quant (experts must
    be heap, not mmap) is gated by the caller (int4/int8 only)."""
    PT = _packed4_3d_cls()
    for name, child in layer.named_children():
        if getattr(child, "experts", None) is not None:
            return name, child
        for sub in child.modules():
            for attr in ("gate_up_proj", "down_proj"):
                a = getattr(sub, attr, None)
                if isinstance(a, PT) or (a is not None and hasattr(a, "dim") and a.dim() == 3):
                    return name, child
    return None, None


def _pack4_expert(We, ng: int, G: int, in_pad: int, in_f: int, dt):
    """Group-wise int4-pack ONE expert's 2D weight [out, in] -> (qpacked [out, in_pad//2] uint8,
    scale [out, ng], zero [out, ng]). The SINGLE source of the per-expert quant math, shared by
    _pack4_3d (in-RAM source, #61) and _stream_pack4_experts (streamed source, #62), so both stay
    bit-identical to the original whole-tensor path (group quant is independent across experts)."""
    import torch
    import torch.nn.functional as F
    if in_pad != in_f:
        We = F.pad(We, (0, in_pad - in_f))
    out = We.shape[0]
    Wg = We.reshape(out, ng, G).float()              # only ONE expert in float (~tens of MB)
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    sc = ((wmax - wmin) / 15.0).clamp(min=1e-8)      # [out, ng]
    ze = torch.round(-wmin / sc).clamp(0, 15)
    q = torch.round(Wg / sc.unsqueeze(2) + ze.unsqueeze(2)).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out, in_pad)
    qp = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()   # [out, in_pad//2]
    return qp, sc.to(dt), ze.to(dt)


def _pack4_3d(W3, group_size: int = _INT4_GROUP):
    """Quantize a fused expert tensor [E, out, in] to a Packed4Tensor3D (group-wise int4), ONE
    EXPERT AT A TIME (#61) — the old path `.float()`'d the WHOLE [E, out, in] at once (~14 GB heap
    spike for a MiniMax layer's 256 experts) on top of the bf16 source, OOM-killing memory-tight
    nodes during "building shard". Per-expert == whole-tensor bit-for-bit (verified, CHANGELOG
    m4at). Used when the bf16 source is already resident (mmap on Linux); see _stream_pack4_experts
    (#62) for the variant that streams the source so the layer's experts never land whole in RAM."""
    import torch
    PT = _packed4_3d_cls()
    E, out, in_f = W3.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    dt = W3.dtype
    qpacked = torch.empty((E, out, in_pad // 2), dtype=torch.uint8)
    scale = torch.empty((E, out, ng), dtype=dt)
    zero = torch.empty((E, out, ng), dtype=dt)
    for e in range(E):
        qp, sc, ze = _pack4_expert(W3[e], ng, G, in_pad, in_f, dt)   # one expert (mmap slice on Linux)
        qpacked[e] = qp; scale[e] = sc; zero[e] = ze
        del qp, sc, ze
    return PT(qpacked, scale, zero, in_f, G)


def _quantize_experts4_(module) -> None:
    """Replace fused MoE expert tensors (3D gate_up_proj/down_proj nn.Parameters) with int4
    Packed4Tensor3D. nn.Linear (attention, router gate, shared experts) is handled separately
    by _quantize_int4_; this catches ONLY the raw 3D expert Parameters those miss."""
    from torch import nn
    targets = []
    for sub in module.modules():
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                targets.append((sub, attr))
    for sub, attr in targets:
        p = getattr(sub, attr)
        delattr(sub, attr)                       # drop the bf16 Parameter
        setattr(sub, attr, _pack4_3d(p.data))    # install the int4 holder (submodule)
    # Force the EAGER experts forward on the modules we quantized. transformers 5.x dispatches
    # experts via config._experts_implementation; only "eager" indexes self.gate_up_proj[idx]
    # per routed expert (which our per-expert int4 holder supports). The grouped_mm/batched_mm/
    # deepgemm kernels take the WHOLE 3D weight tensor and would break on the holder. Eager loops
    # over hit experts — slower, but correctness/memory win > speed for a model that only fits at int4.
    seen = set()
    for sub, _attr in targets:
        cfg = getattr(sub, "config", None)
        if cfg is not None and hasattr(cfg, "_experts_implementation") and id(cfg) not in seen:
            cfg._experts_implementation = "eager"
            seen.add(id(cfg))


def _quantize_experts4_streamed(module, layer_idx: int, fetch_experts, dt) -> None:
    """Per-expert STREAMING build of a layer's MoE experts (#62). The model's fused gate_up_proj/
    down_proj are still META (skip_experts dropped the experts from the layer blob). Read the fused
    target shapes [E, out, in] from the meta Parameters, fetch the experts in chunks via
    fetch_experts(layer, e0, k), and int4-pack via _pack4_expert into the holder's [e] slot. Handles
    BOTH controller checkpoint layouts, auto-detected from the returned blob keys (#75):
      - NON-FUSED source -> {'{local_e}.{proj}': bf16 2D}: fuse each expert exactly as
        _fuse_moe_experts does (gate_up = cat([gate, up]); down = w2; orientation auto-detected).
      - FUSED source -> {'gate_up_proj': [kk,out,in], 'down_proj': [kk,out,in]}: pack each 3D expert
        slice straight in (already fused, no gate/up cat).
    BOTH holders built in one pass per chunk — so the layer's ~7 GB of experts never lands in RAM on
    a memory-tight node."""
    import torch
    from torch import nn
    PT = _packed4_3d_cls()
    G = _INT4_GROUP
    tgt: dict = {}      # attr -> {sub, E, out, in_f, ng, in_pad, qpacked, scale, zero}
    for sub in module.modules():
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                E, out, in_f = int(p.shape[0]), int(p.shape[1]), int(p.shape[2])
                ng = (in_f + G - 1) // G
                tgt[attr] = {"sub": sub, "E": E, "out": out, "in_f": in_f, "ng": ng, "in_pad": ng * G,
                             "qpacked": torch.empty((E, out, ng * G // 2), dtype=torch.uint8),
                             "scale": torch.empty((E, out, ng), dtype=dt),
                             "zero": torch.empty((E, out, ng), dtype=dt)}
    if not tgt:
        return
    E = next(iter(tgt.values()))["E"]
    gu = tgt.get("gate_up_proj")
    per = max(1, (gu["out"] * gu["in_f"] * 2) if gu else 1)        # one expert's gate_up bf16 bytes
    chunk = max(1, min(64, (256 * 1024 * 1024) // per))            # experts per fetch (>=1)

    def _pick(projs, *names):
        for nm in names:
            if nm in projs:
                return projs[nm]
        return None

    def _pack_into(attr, e, src2d):                                # src2d already [out, in_f]
        b = tgt[attr]
        qp, sc, ze = _pack4_expert(src2d.to(dt), b["ng"], G, b["in_pad"], b["in_f"], dt)
        b["qpacked"][e] = qp; b["scale"][e] = sc; b["zero"][e] = ze

    e = 0
    while e < E:
        k = min(chunk, E - e)
        blob = fetch_experts(layer_idx, e, k)                     # {'{le}.{proj}': bf16} OR fused
        # FUSED checkpoint (#75): the controller couldn't serve per-expert tensors so it sent the 3D
        # fused slices directly, keyed by projection ('gate_up_proj'/'down_proj', each [kk, out, in]).
        # Pack each expert slice straight in — no gate/up fusion (the tensor is already gate_up).
        if "gate_up_proj" in blob and "down_proj" in blob:   # complete fused blob (#75): need BOTH
            for attr in ("gate_up_proj", "down_proj"):
                if attr not in tgt or attr not in blob:
                    continue
                w3 = blob[attr]                                   # [kk, out, in]
                b = tgt[attr]
                for le in range(w3.shape[0]):
                    src2d = w3[le]
                    if tuple(src2d.shape) != (b["out"], b["in_f"]):
                        src2d = src2d.t()                         # orientation auto-detect
                    _pack_into(attr, e + le, src2d.contiguous())
            del blob
            e += k
            continue
        grouped: dict = {}
        for key, t in blob.items():
            if "." not in key:   # not a per-expert '{le}.{proj}' key (e.g. a partial fused blob) —
                raise RuntimeError(  # fail clearly instead of a cryptic unpack error
                    f"L{layer_idx}: unexpected expert blob key {key!r} — controller/worker expert "
                    "layout mismatch (deploy controller + workers on the same version)")
            le_s, proj = key.split(".", 1)
            grouped.setdefault(int(le_s), {})[proj] = t
        for le in range(k):
            projs = grouped.get(le, {})
            if "gate_up_proj" in tgt:
                gate, up = _pick(projs, "w1", "gate_proj"), _pick(projs, "w3", "up_proj")
                if gate is None or up is None:
                    raise RuntimeError(f"L{layer_idx} expert {e+le}: missing gate/up {list(projs)}")
                b = tgt["gate_up_proj"]
                guc = torch.cat([gate, up], dim=0)                # gate THEN up (matches _fuse_moe_experts)
                if tuple(guc.shape) != (b["out"], b["in_f"]):
                    guc = torch.cat([gate.t(), up.t()], dim=0)    # orientation auto-detect
                _pack_into("gate_up_proj", e + le, guc.contiguous())
            if "down_proj" in tgt:
                down = _pick(projs, "w2", "down_proj")
                if down is None:
                    raise RuntimeError(f"L{layer_idx} expert {e+le}: missing down {list(projs)}")
                b = tgt["down_proj"]
                if tuple(down.shape) != (b["out"], b["in_f"]):
                    down = down.t()
                _pack_into("down_proj", e + le, down.contiguous())
        del blob, grouped
        e += k
    seen = set()
    for attr, b in tgt.items():
        delattr(b["sub"], attr)                                   # drop the meta Parameter
        setattr(b["sub"], attr, PT(b["qpacked"], b["scale"], b["zero"], b["in_f"], G))
        cfg = getattr(b["sub"], "config", None)                   # force eager experts forward
        if cfg is not None and hasattr(cfg, "_experts_implementation") and id(cfg) not in seen:
            cfg._experts_implementation = "eager"; seen.add(id(cfg))
    print(f"[load] L{layer_idx}: streamed {E} experts (per-expert int4, chunk {chunk})")


def _quantize_experts4_streamed_nonfused(module, layer_idx: int, fetch_experts, dt) -> None:
    """Per-expert STREAMING int4 build for a NON-fused MoE layer (#78). The model stores experts as
    an nn.ModuleList of {w1,w2,w3}-style nn.Linears (e.g. MiniMax-M2 block_sparse_moe.experts.N.*,
    or Mixtral) — all META here because skip_experts dropped them from the layer blob. For each
    experts ModuleList: read E + the per-expert Linear proj names + shapes from the meta Linears,
    fetch experts in chunks via fetch_experts(layer, e0, k) (server returns {'{le}.{proj}': bf16
    [out,in]}, the same non-fused wire format _quantize_experts4_streamed already consumes), int4-pack
    each proj with _pack4_expert and replace the meta nn.Linear with a QuantLinear4 IN PLACE — exactly
    the holder the full-blob path's _quantize_int4_ would produce, so output is bit-identical. The
    transient is bounded by the chunk (~256 MiB), NOT the layer's full ~7 GB of experts: that is what
    lets a 129 GB non-fused MoE SPREAD across the fleet instead of cramming onto one big node (the
    full-blob path's failure). The expert Linears must be quantized BEFORE _quantize_int4_ walks the
    layer (caller orders it so) — once replaced they are no longer nn.Linear, so that walk skips them."""
    from torch import nn
    QL = _quant4_linear_cls()
    G = _INT4_GROUP
    # Locate the non-fused expert container(s) in this layer (an `experts` ModuleList of Linear-bearing
    # modules). Usually exactly one (block_sparse_moe.experts); handle >1 defensively.
    blocks = []
    for sub in module.modules():
        experts = getattr(sub, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0 \
                and any(isinstance(m, nn.Linear) for m in experts[0].modules()):
            blocks.append(experts)
    for experts in blocks:
        E = len(experts)
        # Per-expert Linear projections, discovered from expert 0 (attr name == the checkpoint proj
        # token the server keys by, e.g. w1/w3/w2). DIRECT children only — matches the server's flat
        # `experts.{N}.{proj}.weight` regex (nested experts wouldn't be served per-expert anyway).
        proj_names = [nm for nm, ch in experts[0].named_children() if isinstance(ch, nn.Linear)]
        if not proj_names:
            continue
        per = 0
        for nm in proj_names:                                   # per-expert bf16 footprint -> chunk size
            w = getattr(experts[0], nm).weight
            per += int(w.shape[0]) * int(w.shape[1]) * 2
        per = max(1, per)
        chunk = max(1, min(64, (256 * 1024 * 1024) // per))     # cap in-flight bf16 transient at ~256 MiB
        e = 0
        while e < E:
            k = min(chunk, E - e)
            blob = fetch_experts(layer_idx, e, k)               # {'{le}.{proj}': bf16 [out,in]}
            grouped: dict = {}
            for key, t in blob.items():
                if "." not in key:
                    raise RuntimeError(f"non-fused expert blob key {key!r} has no '.' (layer {layer_idx})")
                le_s, proj = key.split(".", 1)
                grouped.setdefault(int(le_s), {})[proj] = t
            for le, projs in grouped.items():
                gi = e + le                                     # server returns chunk-local indices [0..k)
                if gi >= E:
                    continue
                expert_mod = experts[gi]
                for proj, t in projs.items():
                    lin = getattr(expert_mod, proj, None)
                    if not isinstance(lin, nn.Linear):
                        continue                                # unexpected proj name -> skip (meta guard catches leftovers)
                    exp_out, exp_in = int(lin.weight.shape[0]), int(lin.weight.shape[1])
                    w2d = t
                    out, in_f = int(t.shape[0]), int(t.shape[1])
                    if (out, in_f) != (exp_out, exp_in) and (in_f, out) == (exp_out, exp_in):
                        w2d = t.t().contiguous(); out, in_f = exp_out, exp_in   # orientation auto-detect
                    ng = (in_f + G - 1) // G
                    in_pad = ng * G
                    qp, sc, ze = _pack4_expert(w2d.to(dt), ng, G, in_pad, in_f, dt)
                    setattr(expert_mod, proj, QL(qp, sc, ze, lin.bias, in_f, G))
            del blob, grouped
            e += k
        print(f"[load] L{layer_idx}: streamed {E} non-fused experts "
              f"({'/'.join(proj_names)}, per-expert int4, chunk {chunk})")


def _layer_has_meta_experts(lyr) -> bool:
    """True if the layer's fused 3D MoE experts are still META — i.e. skip_experts dropped the
    layer's expert tensors from the blob, so they must be STREAMED per-expert (#62). With the
    skip_experts filter matching BOTH fused (...gate_up_proj/down_proj) and per-expert
    (...experts.{N}.{proj}.weight) names, a big-MoE int4 layer ends up here (meta -> streamed),
    while a dense layer (no 3D expert params) or a load where skip_experts was off (experts resident,
    not meta) returns False and uses the in-place _quantize_experts4_. Auto-selects per layer."""
    from torch import nn
    for sub in lyr.modules():
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3 and p.is_meta:
                return True
    return False


def _layer_has_meta_experts_nonfused(lyr) -> bool:
    """True if the layer has a NON-fused `experts` ModuleList whose Linear weights are still META —
    i.e. skip_experts dropped them from the layer blob, so they must be STREAMED per-expert (#78,
    the non-fused analogue of _layer_has_meta_experts). False when experts are resident (skip_experts
    off -> full-blob path) or the layer has no such ModuleList."""
    from torch import nn
    for sub in lyr.modules():
        experts = getattr(sub, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            for m in experts[0].modules():
                if isinstance(m, nn.Linear) and getattr(m.weight, "is_meta", False):
                    return True
    return False


def _model_has_fused_experts(model) -> bool:
    """True if the model uses FUSED 3D MoE experts (gate_up_proj/down_proj nn.Parameters) — the
    layout the per-expert STREAMING packer (#62/#75) targets. A non-fused MoE (e.g. MiniMax-M2,
    whose experts are an nn.ModuleList of {w1,w2,w3} Linears) returns False, so its int4 load uses
    the full-blob path (_quantize_int4_ quantizes the expert Linears in place) instead of streaming
    into fused tensors that don't exist. Dense models also return False (no MoE)."""
    from torch import nn
    for sub in model.modules():
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                return True
    return False


def _model_has_nonfused_experts(model) -> bool:
    """True if the model uses NON-fused MoE experts: an nn.ModuleList named `experts` whose elements
    contain nn.Linear leaves (e.g. MiniMax-M2 / Mixtral experts.N.{w1,w3,w2}). The complement of
    _model_has_fused_experts (3D Parameters) and of dense models (no `experts` ModuleList). Gates the
    per-expert non-fused STREAMING packer (#78) so a big non-fused MoE never lands its whole layer of
    experts in RAM at once (the full-blob path that pins the model to one node)."""
    from torch import nn
    for sub in model.modules():
        experts = getattr(sub, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            if any(isinstance(m, nn.Linear) for m in experts[0].modules()):
                return True
    return False


def _assign_meta_from_sd(model, sd) -> int:
    """Materialize meta tensors that load_state_dict(assign=True) SKIPPED.

    load_state_dict only installs keys present in the model's own state_dict. A NON-PERSISTENT
    buffer is excluded from state_dict, so even though the served blob carries it, it is never
    assigned and stays on the 'meta' device — and the later module.to(device) placement then raises
    'Cannot copy out of meta tensor; no data!'. The canonical case is MiniMax-M2's sigmoid-router
    `block_sparse_moe.e_score_correction_bias` (DeepSeek-style routing); softmax-routed MoEs
    (Mixtral/OLMoE) have no such buffer, which is why only MiniMax tripped this.

    For every served tensor whose target attribute is STILL meta, assign the real data directly.
    Tensors NOT in sd (the int4 experts dropped by skip_experts) are untouched — they stay meta for
    the per-expert streaming build. Returns the number of tensors materialized."""
    import torch
    fixed = 0
    for key, t in sd.items():
        parent = model
        parts = key.split(".")
        try:
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            attr = parts[-1]
            cur = getattr(parent, attr, None)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if cur is None or not getattr(cur, "is_meta", False):
            continue
        val = (t if t.dtype == cur.dtype else t.to(cur.dtype)).detach().clone()
        if isinstance(cur, torch.nn.Parameter):
            setattr(parent, attr, torch.nn.Parameter(val, requires_grad=False))
        else:
            parent.register_buffer(attr, val, persistent=False)
        fixed += 1
    return fixed


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


def _missing_pkgs_from_err(exc: Exception) -> list[str]:
    """Best-effort extract pip package name(s) from an ImportError raised while BUILDING a model
    (esp. trust_remote_code modeling code, e.g. nomic-embed-text needing einops). Handles
    transformers' "Run `pip install X Y`" and "...not found in your environment: A, B" forms, plus
    plain "No module named 'X'". Returns ONLY safe package tokens so we never feed pip junk."""
    import re
    msg = str(exc)
    pkgs: list[str] = []
    m = re.search(r"pip install ([^\n`'\"]+)", msg)
    if m:
        pkgs = m.group(1).split()
    if not pkgs:
        m = re.search(r"not found in your environment:\s*([^\n.]+)", msg)
        if m:
            pkgs = [p.strip() for p in m.group(1).split(",")]
    if not pkgs:
        m = re.search(r"No module named ['\"]([A-Za-z0-9_][A-Za-z0-9_.\-]*)", msg)
        if m:
            pkgs = [m.group(1).split(".")[0]]
    return [p for p in pkgs if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", p or "")]


def _build_with_autodeps(build_fn, label: str = ""):
    """Run build_fn(); if it raises ImportError naming missing pip package(s), install them into
    THIS worker's env (sys.executable -m pip) and RETRY — so a model whose trust_remote_code needs
    a package the worker lacks (e.g. einops) self-heals ON LOAD instead of failing the whole load
    (#84). Bounded: each package is tried once and it gives up after a few rounds, so a genuinely
    broken import can't loop forever; a pip failure surfaces as a clear ImportError."""
    import subprocess
    tried: set = set()
    while True:
        try:
            return build_fn()
        except ImportError as exc:
            pkgs = [p for p in _missing_pkgs_from_err(exc) if p not in tried]
            if not pkgs or len(tried) >= 8:
                raise
            tried.update(pkgs)
            print(f"[deps] {label}: missing {pkgs} — pip-installing into worker env", flush=True)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
                import importlib
                importlib.invalidate_caches()   # make the freshly-installed package importable now
            except Exception as pe:
                raise ImportError(f"auto-install of {pkgs} failed ({pe}); install it on this "
                                  f"worker manually") from exc


class EmbeddingModel:
    """Single-node sentence encoder (BERT-family). One forward -> masked mean-pool -> L2 norm.
    No KV cache, no pipeline, no lm_head. Stored in Worker.shards like a Shard but only needs
    loaded_params/loaded_bytes + unloadability — it never enters the decoder data path (it only
    receives kind:"embed" frames), so it can omit the Shard-only attrs (next_writers/layer
    ranges/has_head). torch is imported lazily (module-scope torch isn't guaranteed), so encode
    uses `with torch.inference_mode()` rather than the decorator form."""
    def __init__(self, model_dir, device, dtype):
        from transformers import AutoModel
        # nomic's custom config may reject _attn_implementation="eager"; retry without it.
        try:
            self.model = AutoModel.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=dtype,
                _attn_implementation="eager").eval()
        except ImportError:
            raise   # a MISSING dep (e.g. einops) -> let _build_with_autodeps install + retry the
            #         whole build (don't waste a 2nd no-eager attempt that hits the same ImportError)
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=dtype).eval()
        try:
            self.model.to(device)
        except Exception:
            device = "cpu"
            self.model.to("cpu")
        self.device = device
        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.gpu_bytes = self.loaded_bytes if "cuda" in str(device) else 0

    def encode(self, input_ids, attention_mask):
        import torch
        with torch.inference_mode():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state                                   # [B,T,H]
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)          # masked mean
            # L2-normalize, return float32 on CPU -> [B,H]
            return torch.nn.functional.normalize(pooled, p=2, dim=1).to(torch.float32).cpu()


# ---- m4c153 code-split: Shard/Worker relocated into mixin modules (see state.py) ----
# Worker-side leaf modules holding Shard/Worker methods VERBATIM; state.bind() (at module end)
# injects this module's namespace so the relocated bodies resolve their globals. In client.py's
# EXTRA_UPDATE_FILES. CONVERGENCE BRIDGE: an old worker swapping in this client.py fetched it but
# not yet these files, so pull each from GitHub raw once if missing (repo_raw_url imported above).
import urllib.request as _wsreq
for _wsm in ("state", "shard_build", "shard_forward", "worker_load", "worker_net"):
    try:
        __import__(_wsm)
    except Exception:
        try:
            with _wsreq.urlopen(repo_raw_url().format(f=_wsm + ".py"), timeout=30) as _wsr:
                _wsb = _wsr.read()
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), _wsm + ".py"), "wb") as _wsf:
                _wsf.write(_wsb)
            __import__(_wsm)
        except Exception:
            pass
import state
import shard_build, shard_forward, worker_load, worker_net
from shard_build import ShardBuildMixin
from shard_forward import ShardForwardMixin
from worker_load import WorkerLoadMixin
from worker_net import WorkerNetMixin


class Shard(ShardBuildMixin, ShardForwardMixin):
    # m4c153 code-split: Shard composed from ShardBuildMixin (placement/stream-load/from_*)
    # + ShardForwardMixin (forward/_forward_impl). __init__ and _finalize_placement (reads the
    # rebound _CPU_FP32_GEMM global -> must read it live) stay here. state.bind injects the
    # client namespace into the mixin modules at startup — see state.py.
    def __init__(self, cfg, sd: dict, layer_start: int, layer_end: int,
                 has_embed: bool, has_head: bool, dtype,
                 device: str = "cpu", gpu_mem_gb: float = 0.0,
                 attn: str = "eager", quant: str = "none",
                 tp_rank: int = 0, tp_size: int = 1, tp_allreduce=None) -> None:
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        # Qwen2.5-Omni: AutoModelForCausalLM can't build Qwen2_5OmniTextConfig. Build ONLY the
        # Thinker TEXT decoder (Qwen2_5OmniThinkerTextModel) + a fresh lm_head, wrapped so
        # .model.layers / .lm_head match the served 'model.*'/'lm_head' weights (controller
        # strips the 'thinker.' prefix). We DON'T build the audio_tower/visual towers on workers
        # — they only hold text layers, and constructing those heavy towers destabilized nodes.
        # self.cfg = the Thinker's text config.
        omni_thinker = getattr(cfg, "thinker_config", None)
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
        self.quant = quant
        if quant in ("int8", "int4"):
            qlayer = _quantize_int4_ if quant == "int4" else _quantize_int8_
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
        if _CPU_FP32_GEMM:
            for _m in (([self.embed] if self.has_embed else [])
                       + list(self.owned_layers)
                       + ([self.norm, self.head] if self.has_head else [])):
                _accelerate_cpu_linears(_m)
        # Fused int4 (#71): every weight now has its FINAL device, so build the tinygemm fused-int4
        # kernel per QuantLinear4 (2D linears: attn, router, shared experts, dense). ~3.6x faster
        # int4 decode (no per-token re-dequant). Self-checked + naive fallback inside prepare_fused;
        # MoE 3D experts (Packed4Tensor3D) have no prepare_fused -> stay on the naive path. Idempotent.
        if _FUSED_INT4:
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


def _weight_map(model_dir: str) -> dict[str, str]:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        return {name: os.path.join(model_dir, fn) for name, fn in wm.items()}
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as fh:
            return {name: single for name in fh.keys()}
    raise FileNotFoundError(f"no safetensors found in {model_dir}")


def _load_tensors(names: list[str], weight_map: dict[str, str]) -> dict:
    from safetensors import safe_open
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    out = {}
    for fn, ns in by_file.items():
        with safe_open(fn, framework="pt") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n)
    return out


def _assemble_sd(tensors: dict, start: int, end: int, has_embed: bool,
                 has_head: bool, tied: bool) -> dict:
    """Map raw tensors to the state-dict keys load_state_dict expects, resolving
    the tied head to a (cloned) copy of the embedding matrix."""
    sd: dict = {}
    if has_embed:
        sd["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"]
    for i in range(start, end):
        for n in (x for x in tensors if x.startswith(f"model.layers.{i}.")):
            sd[n] = tensors[n]
    if has_head:
        sd["model.norm.weight"] = tensors["model.norm.weight"]
        if tied:
            sd["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        else:
            sd["lm_head.weight"] = tensors["lm_head.weight"]
    return sd


# ---------------------------------------------------------------------------
# Worker — owns the current stage's shard + data-plane wiring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Self-update: poll GitHub for a newer copy of THIS file; when idle, swap it in
# and exit(42) so the supervisor (systemd Restart=always on Linux / the .bat loop
# on Windows) relaunches the new code. Idle-gated so a running model is never cut.
# ---------------------------------------------------------------------------
SELF_UPDATE_POLL_S = 120   # poll GitHub every 2 minutes (fast deploys; idle-gated)
SELF_UPDATE_FETCH_TRIES = 4      # #3: bounded retry per file within a cycle (CDN propagation lag on a
SELF_UPDATE_FETCH_BACKOFF_S = 8  # freshly-added module 404s on raw.githubusercontent until it syncs)


def _extract_version(blob: bytes) -> str:
    # #4: regex the `VERSION = "0.2-m4cNNN"` constant out of a fetched client.py/server.py so the
    # self-updater restarts ONLY on a real VERSION bump, not on any byte diff (doc/comment commits).
    import re
    try:
        m = re.search(rb'^VERSION\s*=\s*["\']([^"\']+)["\']', blob, re.MULTILINE)
        return m.group(1).decode("utf-8", "replace") if m else ""
    except Exception:
        return ""


def _fetch_repo_file(fname: str):
    # Self-update fetches each file's latest bytes from the PUBLIC GitHub repo's raw endpoint
    # (repo_raw_url, owner/branch from config.json) — NO auth/token, so no secret is in the source
    # (#public-release). Any failure -> returns None (fail-closed; the worker keeps running).
    import urllib.request
    try:
        with urllib.request.urlopen(repo_raw_url().format(f=fname), timeout=30) as r:
            return r.read()
    except Exception:
        return None


# Extra repo files (besides client.py) to keep in sync on self-update. A client+server SHARED
# module (wire.py) is listed in BOTH client.py + server.py.
EXTRA_UPDATE_FILES: list[str] = ["wire.py", "config.json", "shards.py",
                                 # m4c153 code-split: shared-state registry + Shard/Worker mixins
                                 "state.py", "shard_build.py", "shard_forward.py",
                                 "worker_load.py", "worker_net.py"]   # config + shared packer
# (#distributed-packing) synced like a module — shards.pack_unit_tensors is the shared packer the
# remote-pack handler calls, so a worker-packed cache unit is bit-identical to a controller-compiled one.


def _self_update_check(fname: str, is_idle) -> None:
    """Multi-file self-update: fetch the primary file + EXTRA_UPDATE_FILES, and if ANY changed
    (and we're idle) stage ALL changed files together. RESTART only when the fetched primary-file
    VERSION differs from the running VERSION (#4: a same-VERSION doc/comment commit must NOT bounce
    the worker). Each fetch is bounded-retried with backoff so a CDN-propagation 404 on a freshly-
    added file (#3) gets time to sync; if a file STILL won't fetch, abort THIS cycle (never apply a
    half-updated set) and retry next poll."""
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
    if not is_idle():
        print(f"[update] {changed} newer on repo - deferring (build in progress)")
        return
    # #4: only RESTART on a VERSION bump in the primary file. Stage same-VERSION content changes to disk
    # (atomic, picked up on the next natural restart) but don't bounce the worker for a doc/comment commit.
    remote_ver = _extract_version(fetched.get(fname, b""))
    version_bumped = bool(remote_ver) and remote_ver != VERSION
    for fn in changed:                           # write all .new first, then atomic-replace each
        path = os.path.join(here, fn)
        tmp = path + ".new"
        with open(tmp, "wb") as fh:
            fh.write(fetched[fn])
        os.replace(tmp, path)
    if not version_bumped:
        print(f"[update] {changed} staged on disk (VERSION {VERSION} unchanged) - NOT restarting (#4)")
        return
    print(f"[update] {changed} newer on repo (VERSION {VERSION} -> {remote_ver or '?'}) - restarting")
    os._exit(42)                                 # supervisor relaunches on the new code


async def _self_update_loop(fname: str, is_idle) -> None:
    while True:
        await asyncio.sleep(SELF_UPDATE_POLL_S)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(_self_update_check, fname, is_idle)


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


# ---------------------------------------------------------------------------
# Control session (bidirectional)
# ---------------------------------------------------------------------------

async def _heartbeat_loop(writer: asyncio.StreamWriter, lock: asyncio.Lock,
                          node_id: str, interval: float, gpu: bool = False) -> None:
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
        _log_cursor, _new_logs = drain_new_logs(_log_cursor)
        if _new_logs:
            hb["logs"] = _new_logs[-300:]   # #logs: relay new log lines (capped per beat)
        if gpu:
            used, total = _gpu_mem_gb()
            hb["vram_used_gb"] = round(used, 2)
            hb["vram_total_gb"] = round(total, 2)
            with contextlib.suppress(Exception):   # GPU compute utilization % (#46; needs pynvml)
                import torch
                hb["gpu_util"] = float(torch.cuda.utilization(0))
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
                        info = await worker.handle_load(msg)
                        # #1: echo model_id so the controller resolves THIS model's load future
                        # (a single shared future cross-resolved on a co-loaded node).
                        await reply({"type": "ready", "node_id": node_id,
                                     "model_id": msg.get("model_id"), **info})
                        if msg.get("kind") == "embedding":   # no layer range — whole encoder on one node
                            print(f"[load] embedding {msg.get('model_id')} "
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
                elif mtype == "unload":
                    await worker.handle_unload(msg.get("model_id"))
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
                        await worker.handle_unload(None)   # drop ALL shards if any remain
                    freed = await asyncio.to_thread(_flush_os_cache)
                    with contextlib.suppress(Exception):
                        await reply({"type": "freed", "node_id": node_id, "free_gb": round(freed, 2)})
                    print(f"[free] controller requested RAM release -> {freed:.1f} GB free")
                elif mtype == "restart":
                    # Controller commanded a full-fleet restart (e.g. to abort a wedged load or
                    # force a fresh deploy). Ack, then exit(42) so the supervisor (client.bat /
                    # systemd Restart=always) relaunches on the current code — drops any resident
                    # shard cleanly on relaunch. (#fleet-restart)
                    print("[restart] controller requested restart - exiting(42) for supervisor relaunch")
                    with contextlib.suppress(Exception):
                        await reply({"type": "restarting", "node_id": node_id})
                    os._exit(42)

        hb = asyncio.create_task(
            _heartbeat_loop(writer, wlock, node_id, args.heartbeat_interval, _using_gpu(args)))
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
        await worker.handle_unload()
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


def _console_panel_loop(worker, node_name: str) -> None:
    """Live, NON-SCROLLING status panel pinned to the bottom of an interactive worker CONSOLE.
    OPT-IN (IM_CONSOLE_PANEL=1) + TTY only — never for a service / redirected stdout. Reserves the
    bottom PANEL_H lines via an ANSI scroll region so normal log prints scroll ABOVE while this
    redraws IN PLACE; every line is clipped to the terminal width so it can never wrap into a mess.
    Per-peer rows colour by live traffic: GREEN active, GREY idle, RED bandwidth-heavy, each labelled
    by the peer (other node's IP or 'controller'). Fully best-effort + exception-isolated — a render
    error never touches inference. Region is reset on exit. ASCII glyphs only (no console mojibake)."""
    import time as _t
    import shutil
    ESC = "\x1b"
    GREEN, GREY, RED, CYAN, YEL, RESET = ("\x1b[32m", "\x1b[90m", "\x1b[31m", "\x1b[36m",
                                          "\x1b[33m", "\x1b[0m")
    PANEL_H = 9
    HEAVY = 8 * 1024 * 1024            # bytes/s on a link -> RED
    ACTIVE = 1024                      # bytes/s -> GREEN (else GREY idle)
    if os.name == "nt":               # enable ANSI on the Windows console
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            k.GetConsoleMode(h, ctypes.byref(mode))
            k.SetConsoleMode(h, mode.value | 0x0004)   # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass

    def _clip(s: str, w: int) -> str:  # clip PLAIN text (no colour codes) to width
        return s if len(s) <= w else (s[:max(0, w - 2)] + "..")

    prev = {"t": _t.time(), "net": dict(NET), "peers": {}}
    last_size = (0, 0)
    try:
        while True:
            _t.sleep(1.2)
            try:
                cols, rows = shutil.get_terminal_size((100, 30))
                if rows < PANEL_H + 4 or cols < 30:
                    continue
                top = rows - PANEL_H
                now = _t.time()
                dt = max(0.5, now - prev["t"])
                din = (NET["in"] - prev["net"]["in"]) / dt
                dout = (NET["out"] - prev["net"]["out"]) / dt
                cpu = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                models = list(worker.shards.keys())
                building = getattr(worker, "_building", 0)
                act = din + dout
                doing = ("loading/building" if building else ("serving" if act > ACTIVE else "idle"))
                dcol = YEL if building else (GREEN if act > ACTIVE else GREY)
                # role in the current pipeline/TP (first shard's assignment)
                role = ""
                for mid, asn in list(worker.assignments.items()):
                    if int(asn.get("tp_size", 1)) > 1:
                        role = f" tp{asn.get('tp_rank')}/{asn.get('tp_size')}"
                    elif asn.get("num_stages", 1) and int(asn.get("num_stages", 1)) > 1:
                        role = f" stage{asn.get('stage')}"
                    break
                data = [(CYAN, f"# {node_name}{role}  [{doing}]  cpu {cpu:.0f}%  "
                               f"ram {vm.used / GB:.1f}/{vm.total / GB:.0f}G  "
                               f"net v{_hr(din)} ^{_hr(dout)}  models:{len(models)}")]
                if models:
                    data.append((GREY, "  resident: " + ", ".join(models)))
                for peer, c in sorted(NET_PEERS.items(),
                                      key=lambda kv: -(kv[1]["in"] + kv[1]["out"])):
                    p0 = prev["peers"].get(peer, {"in": 0, "out": 0})
                    rin = (c["in"] - p0["in"]) / dt
                    rout = (c["out"] - p0["out"]) / dt
                    rate = rin + rout
                    col = RED if rate > HEAVY else (GREEN if rate > ACTIVE else GREY)
                    data.append((col, f"  * {peer:<17} v{_hr(rin)} ^{_hr(rout)}"))
                    if len(data) >= PANEL_H:
                        break
                prev = {"t": now, "net": dict(NET),
                        "peers": {k: dict(v) for k, v in NET_PEERS.items()}}
                out = [f"{ESC}7"]                       # save cursor (DEC)
                if (cols, rows) != last_size:           # (re)set scroll region only on resize
                    out.append(f"{ESC}[1;{top}r")
                    last_size = (cols, rows)
                for i in range(PANEL_H):
                    if i < len(data):
                        color, text = data[i]
                        line = color + _clip(text, cols) + RESET
                    else:
                        line = ""
                    out.append(f"{ESC}[{top + 1 + i};1H{ESC}[2K{line}")
                out.append(f"{ESC}8")                   # restore cursor (back into the log region)
                sys.stdout.write("".join(out))
                sys.stdout.flush()
            except Exception:
                continue
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.write(f"{ESC}[r")                # reset scroll region on exit
            sys.stdout.flush()


async def run(args: argparse.Namespace) -> None:
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
    asyncio.create_task(_self_update_loop(
        "client.py", lambda: not worker._building))
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

def run_self_test_load(model_id: str, attn: str = "eager", quant: str = "none") -> None:
    tune_cpu_threads()   # self-test runs CPU shards too; benefit from tuned threads + fp32 GEMM
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = cfg.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok("The capital of France is", return_tensors="pt").input_ids

    print(f"\nShard load self-test: {model_id}  ({n_layers} layers, attn={attn}, quant={quant})")
    full = Shard.from_hf(model_id, 0, n_layers, has_embed=True, has_head=True, attn=attn, quant=quant)
    print(f"  footprint: {full.loaded_bytes/GB:.2f} GB")
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
                   help="controller host/IP (default from config.json)")
    p.add_argument("--control-port", type=int, default=load_config()["control_port"])
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
    p.add_argument("--quant", default="none", choices=["none", "int8", "int4"],
                   help="weight quantization: 'none' (bf16, default), 'int8' "
                        "(per-channel weight-only — halves the footprint), or 'int4' "
                        "(group-wise ~4.25-bit weight-only — ~1/4 footprint, for 200B+ "
                        "MoEs that won't fit at int8; small decode-speed cost). The "
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
        global _CPU_FP32_GEMM
        _CPU_FP32_GEMM = False
    tune_cpu_threads()
    if args.self_test_load:
        run_self_test_load(args.model, args.attn, args.quant)
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
state.bind(shard_build, shard_forward, worker_load, worker_net)


if __name__ == "__main__":
    main()
