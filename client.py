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

VERSION = "0.2-m4c37"  # version tag only; full changelog -> CHANGELOG.md
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


class Shard:
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
        # rotary_emb stays on CPU; cos/sin are computed there and moved per-device.
        self.cpu = torch.device("cpu")
        self._place_modules(device, gpu_mem_gb)

    @staticmethod
    def _mod_bytes(module) -> int:
        # params + buffers — QuantLinear stores its int8 weight + scale as buffers
        return sum(t.numel() * t.element_size()
                   for t in list(module.parameters()) + list(module.buffers()))

    @staticmethod
    def _mod_gpu_bytes(module) -> int:
        # Bytes of a module's tensors that ACTUALLY live on a CUDA device. Device-accurate (unlike
        # _mod_bytes) so a MoE-split layer — attention on GPU, experts on CPU inside the same module
        # tree — reports only its GPU-resident weight. Used for gpu_bytes/size_vram accounting.
        return sum(t.numel() * t.element_size()
                   for t in list(module.parameters()) + list(module.buffers())
                   if t.device.type == "cuda")

    def _kv_bytes_per_layer(self, ctx: int) -> int:
        """Full-ctx KV bytes ONE layer will grow into (k+v, bf16). Mirrors kv_reserve_probe so the
        GPU placement budget reserves the SAME KV the probe later allocates. 0 if ctx/dims unknown."""
        if not ctx or ctx <= 0:
            return 0
        cfg = self.cfg
        nh = int(getattr(cfg, "num_attention_heads", 0) or 0)
        nkv = int(getattr(cfg, "num_key_value_heads", nh) or nh or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        hd = int(getattr(cfg, "head_dim", 0) or (hidden // nh if nh else 0))
        if nkv <= 0 or hd <= 0:
            return 0
        return 2 * int(ctx) * nkv * hd * 2

    def _place_modules(self, device: str, gpu_mem_gb: float, ctx: int = 0,
                       gpu_budget_gb: float = -1.0) -> None:
        """Assign embed / each owned layer / norm+head to CPU or GPU and move them.
        Modes: cpu | gpu(cuda) whole-on-GPU | cpu+gpu(hybrid) offload-by-VRAM |
        auto whole-if-fits-else-hybrid. Always falls back to CPU without CUDA.
        gpu_budget_gb (#95): the controller's committed-aware GPU budget for THIS stage (free VRAM
        after co-resident models' weights + reserved KV, minus the plan floor). >=0 caps placement;
        <0 means the controller didn't send one (old controller) -> uncapped (legacy behavior)."""
        torch = self.torch
        cpu = self.cpu
        mode = (device or "cpu").lower()
        want_gpu = mode in ("gpu", "cuda", "auto", "cpu+gpu", "hybrid")
        cuda_ok = False
        if want_gpu:
            try:
                cuda_ok = torch.cuda.is_available()
            except Exception:
                cuda_ok = False
        if not (want_gpu and cuda_ok):
            self.embed_device = cpu
            self.layer_devices = [cpu] * len(self.owned_layers)
            self.layer_split = [False] * len(self.owned_layers)
            self.norm_device = self.head_device = cpu
            self.placement = "cpu" + ("" if not want_gpu else " (no CUDA → CPU)")
            self.gpu_bytes = 0
            self.gpu_kv_bytes = 0
            self._finalize_placement()
            return

        gpu = torch.device("cuda:0")
        free, _total = torch.cuda.mem_get_info(0)
        # NEVER oversubscribe the card: the controller may plan GPU placement optimistically (sizing
        # against an EMPTY GPU), but coexisting models already hold VRAM. Bound EVERY GPU budget by
        # the LIVE free VRAM (mem_get_info reflects what other resident models hold right now), and
        # reserve THIS shard's full-ctx KV (k+v grows on-GPU during decode) + a CUDA/activation
        # margin — so a 2nd model leaves room instead of filling VRAM to 100% and OOM-ing at
        # generation (which also kills coexisting models' decode). kv_per_layer=0 when ctx unknown.
        # #95 coexistence: clamp the VISIBLE free VRAM to the controller's committed-aware budget for
        # this stage (free VRAM after co-resident models' weights + reserved KV, minus the plan floor).
        # Done HERE so EVERY downstream GPU decision keys off it — the mode=auto whole-on-GPU check
        # (free*0.85), the hybrid default budget (free*0.85), live_free, and the placement string. A
        # co-resident model's card LOOKS free until it faults its full-ctx KV; without this clamp a 2nd
        # shard consolidates onto that VRAM and OOMs the resident model's decode (qwen3+14b). budget 0
        # (GPU fully committed) -> free 0 -> all layers spill to CPU, no GPU grab. <0 = no value sent
        # (old controller) -> uncapped, unchanged behavior.
        if gpu_budget_gb >= 0:
            free = min(int(free), int(gpu_budget_gb * GB))
        GPU_SAFETY = int(0.4 * GB)
        kv_per_layer = self._kv_bytes_per_layer(ctx)
        live_free = max(0, int(free) - GPU_SAFETY)
        nlyr = len(self.owned_layers)
        # #moe-offload: when enabled, a MoE layer that can't fit GPU whole is SPLIT — attention+norms
        # on GPU, the routed-expert block left on CPU (instead of dragging the whole layer to CPU).
        # Gated to int4/int8 — those quantize experts into HEAP buffers (Packed4Tensor3D fused, or
        # QuantLinear4 per-expert), so leaving them on CPU has no mmap-reclaim issue; bf16 experts
        # (possibly mmap) fall back to the whole-layer path.
        moe_off = (bool(getattr(self, "_moe_offload", False))
                   and self.quant in ("int4", "int8"))
        moe_blocks = ([_find_moe_block(l) for l in self.owned_layers]
                      if moe_off else [(None, None)] * nlyr)
        self.layer_split = [False] * nlyr
        whole_need = self.loaded_bytes + kv_per_layer * nlyr
        whole = ((mode in ("gpu", "cuda") and whole_need <= live_free)
                 or (mode == "auto" and whole_need < free * 0.85))
        if whole:
            self.embed_device = gpu if self.has_embed else cpu
            self.layer_devices = [gpu] * nlyr
            self.norm_device = self.head_device = gpu
            self.placement = f"cuda:all ({nlyr} layers)"
        else:  # hybrid: greedily fill a VRAM budget (capped by live-free), spill the rest to CPU
            budget = int(gpu_mem_gb * GB) if gpu_mem_gb > 0 else int(free * 0.85)
            budget = min(budget, live_free)   # live-free cap -> can't oversubscribe a shared card
            used = 0

            def fits(nbytes: int, kv: int = 0) -> bool:
                nonlocal used
                if used + nbytes + kv <= budget:
                    used += nbytes + kv
                    return True
                return False

            self.embed_device = gpu if (self.has_embed and fits(self._mod_bytes(self.embed))) else cpu
            # each GPU-resident layer must hold its weights AND the KV it will grow into at this ctx.
            # #moe-offload: a splittable MoE layer charges only its MIXER (attention+norms = whole
            # layer minus the MoE block) + KV to the GPU budget; the big expert block stays in RAM.
            self.layer_devices = []
            for i, l in enumerate(self.owned_layers):
                _blk = moe_blocks[i][1]
                if moe_off and _blk is not None:
                    mixer_b = self._mod_bytes(l) - self._mod_bytes(_blk)
                    if fits(mixer_b, kv_per_layer):
                        self.layer_devices.append(gpu)   # attention->GPU, experts stay CPU (split)
                        self.layer_split[i] = True
                    else:
                        self.layer_devices.append(cpu)   # mixer didn't fit -> whole layer to CPU
                elif fits(self._mod_bytes(l), kv_per_layer):
                    self.layer_devices.append(gpu)
                else:
                    self.layer_devices.append(cpu)
            if self.has_head:
                hb = self._mod_bytes(self.head) + self._mod_bytes(self.norm)
                self.norm_device = self.head_device = gpu if fits(hb) else cpu
            else:
                self.norm_device = self.head_device = cpu
            ng = sum(1 for d in self.layer_devices if d.type == "cuda")
            nsp = sum(1 for s in self.layer_split if s)
            self.placement = (f"cpu+gpu: {ng}/{nlyr} layers on GPU"
                              + (f" ({nsp} MoE-split: attn->GPU, experts->CPU)" if nsp else "")
                              + f" (budget {budget / GB:.1f} GB of {free / GB:.1f} free, "
                              f"+{kv_per_layer * ng / GB:.1f} GB KV)")

        if self.has_embed:
            self.embed.to(self.embed_device)
        for i, (lyr, d) in enumerate(zip(self.owned_layers, self.layer_devices)):
            if self.layer_split[i]:
                # #moe-offload split: move every child EXCEPT the MoE block (attention, norms) to GPU
                # and leave the block (router+experts+shared) on CPU. Moving only the non-MoE children
                # avoids a transient whole-layer GPU spike (the experts never touch the GPU). A bridge
                # wraps the block so the layer's forward bridges hidden GPU<->CPU around it.
                _attr, _blk = moe_blocks[i]
                for _nm, _child in lyr.named_children():
                    if _nm != _attr:
                        _child.to(gpu)
                for _pn, _p in list(lyr._parameters.items()):
                    if _p is not None:
                        _p.data = _p.data.to(gpu)
                for _bn, _b in list(lyr._buffers.items()):
                    if _b is not None:
                        lyr._buffers[_bn] = _b.to(gpu)
                setattr(lyr, _attr, _moe_bridge_cls()(_blk, self.cpu))
            else:
                lyr.to(d)
        if self.has_head:
            self.norm.to(self.norm_device)
            self.head.to(self.head_device)
        # CPU-resident layers: .to(cpu) is a NO-OP, so for a non-quantized (bf16) shard
        # they stay as mmap VIEWS into the weight temp file. On Windows a mapped file
        # can't be deleted and its pages can't be trimmed on unload -> the node retains
        # the whole spilled shard in RAM after unload (beast kept ~58 GB). Materialize
        # them to heap (clone) so the mmap drops once this build returns and the file is
        # deletable + the RAM reclaimable. int8 already copies to heap during quant, so
        # skip it. Guarded: only when the transient 2x (mmap + clones) fits free RAM,
        # else leave the mmap (correctness over a possible load-time OOM).
        if self.quant != "int8" and not getattr(self, "_streamed", False):
            self._materialize_cpu_layers()   # streamed shards are already heap (no mmap to drop)
        # #moe-offload post-split assertion: a split layer MUST keep its experts on CPU. If the
        # blanket move ever dragged them to GPU (the bug the critics flagged), fail the load LOUDLY
        # here rather than silently OOM the card / mis-report gpu_bytes.
        if any(self.layer_split):
            for i, lyr in enumerate(self.owned_layers):
                if self.layer_split[i]:
                    _br = getattr(lyr, moe_blocks[i][0])
                    if any(b.device.type != "cpu" for b in _br.buffers()):
                        raise RuntimeError(
                            f"moe_offload: layer {i} expert buffers not all on CPU after split")
        # bytes actually resident on the GPU (controller sums these for size_vram). DEVICE-ACCURATE
        # (_mod_gpu_bytes) so a split layer counts only its GPU mixer, not its CPU experts.
        gb = 0
        if self.has_embed:
            gb += self._mod_gpu_bytes(self.embed)
        for lyr in self.owned_layers:
            gb += self._mod_gpu_bytes(lyr)
        if self.has_head:
            gb += self._mod_gpu_bytes(self.norm) + self._mod_gpu_bytes(self.head)
        self.gpu_bytes = gb
        # full-ctx KV these GPU-resident layers will grow into — reported so the controller can
        # RESERVE it against coexisting loads (a 2nd model must not eat this model's KV space).
        self.gpu_kv_bytes = kv_per_layer * sum(1 for d in self.layer_devices if d.type == "cuda")
        # #moe-offload diagnostic: surface WHY the split did/didn't engage (on=flag+quant gate,
        # blocks=layers where a MoE block was detected, split=layers actually split).
        self._moe_dbg = {"on": bool(moe_off),
                         "blocks": sum(1 for b in moe_blocks if b[1] is not None),
                         "split": int(sum(self.layer_split)), "quant": self.quant}
        self._finalize_placement()

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

    def _materialize_cpu_layers(self) -> None:
        """Replace CPU-resident weight tensors (still mmap-backed file views) with heap
        clones so the weight temp file's mmap is dropped — letting unload delete the file
        and reclaim the RAM (the bf16-on-Windows non-release fix). Cloning all CPU params
        transiently needs ~2x the CPU portion (mmap + clones) before the mmap frees, so we
        only do it when free RAM comfortably covers that; otherwise we leave the mmap (the
        old behavior) rather than risk a load-time OOM."""
        torch = self.torch
        mods = []
        if self.has_embed and self.embed_device.type == "cpu":
            mods.append(self.embed)
        mods += [l for l, d in zip(self.owned_layers, self.layer_devices) if d.type == "cpu"]
        if self.has_head and self.head_device.type == "cpu":
            mods += [self.norm, self.head]
        cpu_bytes = sum(self._mod_bytes(m) for m in mods if m is not None)
        if not mods or cpu_bytes <= 0:
            self.cpu_materialized = True   # nothing on CPU -> nothing pinning the mmap
            return
        try:
            free_ram = psutil.virtual_memory().available
        except Exception:
            free_ram = 0
        if free_ram < int(cpu_bytes * 1.2):
            self.cpu_materialized = False   # not enough headroom — keep mmap (logged by caller)
            print(f"[load] CPU weights left mmap-backed (need ~{cpu_bytes*1.2/GB:.1f} GB free "
                  f"to materialize, have {free_ram/GB:.1f} GB) — RAM frees on next worker restart")
            return
        for m in mods:
            if m is None:
                continue
            for p in m.parameters(recurse=True):   # bf16 weights are Parameters; mmap-backed
                if p.device.type == "cpu":
                    p.data = p.data.clone()         # heap copy -> drops this param's mmap view
        self.cpu_materialized = True

    @classmethod
    def from_blob(cls, config_dict: dict, blob: bytes, layer_start: int, layer_end: int,
                  has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                  device: str = "cpu", gpu_mem_gb: float = 0.0,
                  attn: str = "eager", quant: str = "none") -> "Shard":
        """Build a shard from a controller-served safetensors blob (no HF download,
        no model on disk — the blob is loaded straight into RAM)."""
        import tempfile
        import torch
        from transformers import AutoConfig
        from safetensors.torch import load as st_load
        d = tempfile.mkdtemp(prefix="im_cfg_")
        try:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            cfg = AutoConfig.from_pretrained(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        dt = getattr(torch, dtype)
        sd = {k: v.to(dt) for k, v in st_load(blob).items()}
        return cls(cfg, sd, layer_start, layer_end, has_embed, has_head, dt,
                   device=device, gpu_mem_gb=gpu_mem_gb, attn=attn, quant=quant)

    @classmethod
    def from_stream(cls, config_dict: dict, fetch, layer_start: int, layer_end: int,
                    has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                    device: str = "cpu", gpu_mem_gb: float = 0.0,
                    attn: str = "eager", quant: str = "none", fetch_experts=None,
                    tp_rank: int = 0, tp_size: int = 1, tp_allreduce=None,
                    plan_ram_bytes: int = 0, tp_weights=None, ctx: int = 0,
                    gpu_budget_gb: float = -1.0, moe_offload: bool = False,
                    cache: str = "") -> "Shard":
        """Build a shard by STREAMING weights one layer at a time straight into RAM — no temp
        file, no disk. `fetch(start, end, embed, head) -> bytes` returns a safetensors blob for
        that slice. Each layer is fetched, loaded, quantized and FREED before the next, so peak
        RAM ~ the resident (int4) shard + one layer's bf16 — the full bf16 never lands on disk
        OR fully in RAM. Heap tensors (no mmap) -> unload reclaims RAM cleanly.

        TP-v2 (tp_size>1): the model is built on meta, _tp_make_structure_ replaces every linear
        with its REDUCED-DIM module (still meta), then each layer's PER-RANK SLICED weights (served
        by /weights_tp) are streamed straight in — so this rank ever holds only ~1/tp of each layer,
        NOT the v1 load-full-then-shard footprint. The caller must pass a `fetch` that hits
        /weights_tp and a connected tp_allreduce; the row-parallel o_proj/down_proj all-reduce hooks
        are wired here just as in __init__."""
        import gc
        import tempfile
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from safetensors.torch import load as st_load
        self = cls.__new__(cls)
        self.torch = torch
        d = tempfile.mkdtemp(prefix="im_cfg_")          # config (+ any remote modeling .py) dir
        _remote = config_dict.pop("__im_remote_code__", None) if isinstance(config_dict, dict) else None
        _trust = bool(_remote) and bool((config_dict or {}).get("auto_map"))
        try:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            if _remote:                                  # write the model's trust_remote_code .py so
                for _fn, _src in _remote.items():         # AutoConfig/from_config build the REAL arch —
                    with contextlib.suppress(Exception):  # else transformers' native class for this
                        with open(os.path.join(d, _fn), "w", encoding="utf-8") as _rf:   # model_type
                            _rf.write(_src)               # mismatches the checkpoint (all tensors meta)
            cfg = AutoConfig.from_pretrained(d, trust_remote_code=_trust)
            omni_thinker = getattr(cfg, "thinker_config", None)
            if omni_thinker is not None:
                self.cfg = cfg.get_text_config()
            else:
                if getattr(cfg, "text_config", None) is not None:
                    cfg = cfg.get_text_config()
                self.cfg = cfg
            _lt = getattr(self.cfg, "layer_types", None)
            self._hybrid = bool(_lt) and any(t != "full_attention" for t in _lt)
            self._omni = omni_thinker is not None
            self.cfg._attn_implementation = attn
            # transformers 5.x LlamaRotaryEmbedding reads cfg.rope_parameters["rope_type"] in __init__;
            # a 4.x-era custom config (e.g. MiniMax-M2) leaves rope_parameters=None -> 'NoneType' not
            # subscriptable at from_config. Synthesize it from the legacy rope_theta/rope_scaling so the
            # per-layer rotary builds. Only for remote-code (native configs populate it themselves). (#78)
            if _trust and getattr(self.cfg, "rope_parameters", None) is None:
                _rs = getattr(self.cfg, "rope_scaling", None)
                _rp = dict(_rs) if isinstance(_rs, dict) else {}
                _rp.setdefault("rope_type", _rp.get("type", "default"))
                _rp.setdefault("rope_theta", float(getattr(self.cfg, "rope_theta", 10000.0)))
                with contextlib.suppress(Exception):
                    self.cfg.rope_parameters = _rp
            dt = getattr(torch, dtype)
            self.dtype = dt
            self.layer_start, self.layer_end = layer_start, layer_end
            self.has_embed, self.has_head = has_embed, has_head
            self.tp_rank, self.tp_size, self.tp_allreduce = tp_rank, tp_size, tp_allreduce
            self.quant = quant
            # Build the meta skeleton WHILE the config dir (with the remote .py) is alive so
            # from_config can resolve a trust_remote_code class. For a remote-code model keep BUFFERS
            # REAL (accelerate include_buffers=False) — per-layer computed buffers (rotary inv_freq …)
            # then compute correctly instead of landing on 'meta' with no checkpoint to fill them; only
            # PARAMS go to meta (filled by the streamed weights, which carry their own dtype). Native
            # models are UNCHANGED (plain torch.device('meta') + model.to(dt)).
            if omni_thinker is not None:
                from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                    Qwen2_5OmniThinkerTextModel)
                class _OmniTextCausalLM(torch.nn.Module):
                    def __init__(self, m, h):
                        super().__init__(); self.model = m; self.lm_head = h
                with torch.device("meta"):
                    model = _OmniTextCausalLM(
                        Qwen2_5OmniThinkerTextModel(self.cfg),
                        torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False))
                model = model.to(dt)
            elif _trust:
                try:
                    from accelerate import init_empty_weights
                    _ctx = init_empty_weights(include_buffers=False)
                except Exception:
                    _ctx = torch.device("meta")
                # FORCE eager attention for remote-code archs (#78): MiniMax-M2 (and similar) declare
                # _supports_sdpa/_flash=False and implement their own eager attention; transformers
                # otherwise auto-selects sdpa and ABORTS ("does not support scaled_dot_product_attention").
                # The worker's default `attn` (set on cfg above) may be sdpa, so override here. Set the
                # config attr AND pass the kwarg (the kwarg is the path transformers actually honors).
                with contextlib.suppress(Exception):
                    self.cfg._attn_implementation = "eager"
                with _ctx:
                    try:
                        model = AutoModelForCausalLM.from_config(
                            self.cfg, trust_remote_code=True, attn_implementation="eager")
                    except TypeError:   # older transformers: not a from_config kwarg -> config attr set above
                        model = AutoModelForCausalLM.from_config(self.cfg, trust_remote_code=True)
                # do NOT model.to(dt): it would cast the real fp32 rotary inv_freq buffers to bf16.
            else:
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(self.cfg)
                model = model.to(dt)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        # Per-expert FETCH streaming for int4 MoE: drop the experts from the layer blob (skip_experts)
        # and stream+quantize them in bounded chunks, so the layer's full bf16 experts never land in
        # RAM at once. Two layouts: FUSED (3D gate_up_proj/down_proj, #62/#75) and NON-fused (an
        # `experts` ModuleList of {w1,w3,w2} Linears, e.g. MiniMax-M2 / Mixtral, #78). Exactly one
        # applies per model; non-fused only when not fused. bf16/int8 and dense models stream the full
        # blob (no expert skipping). For a big non-fused MoE the full-blob path's ~7 GB/layer transient
        # only fits one big node -> it couldn't spread; streaming bounds the transient to ~256 MiB/chunk.
        stream_experts = (quant == "int4" and fetch_experts is not None
                          and _model_has_fused_experts(model))
        stream_experts_nf = (quant == "int4" and fetch_experts is not None
                             and not stream_experts
                             and _model_has_nonfused_experts(model))
        # #shard-cache Inc 2 (serve-from-cache): each cached layer unit already carries its experts
        # PRE-PACKED (Packed4Tensor3D buffers), so there is no per-expert /experts streaming and no
        # fuse/quant — the cached install builds every holder directly. Force the streaming-expert
        # paths off so _quant_after never runs (the install dispatcher below skips it for cache).
        use_cache = (cache == "int4" and quant == "int4")
        if use_cache:
            stream_experts = stream_experts_nf = False
        # TP-v2: rebuild every layer's linears as REDUCED-DIM modules (still meta) BEFORE streaming
        # any weights, so the per-rank sliced tensors (served by /weights_tp, exact reduced shapes)
        # install via the same load_state_dict(assign=True) path. tp_size==1 -> no-op (full modules).
        if tp_size > 1:
            _tp_make_structure_(model, tp_rank, tp_size, self.cfg, tp_weights)
        self.model = model
        self.owned_layers = [model.model.layers[i] for i in range(layer_start, layer_end)]
        self.embed = model.model.embed_tokens if has_embed else None
        self.norm = model.model.norm if has_head else None
        self.head = model.lm_head if has_head else None

        self.loaded_params = 0
        _seen: set = set()

        from safetensors.torch import load_file as _load_file
        def _install(src) -> None:
            # src is BYTES (the fleet path since m4c25) -> in-RAM st_load builds heap tensors, freed
            # per layer. The str/PATH branch (mmap a file then unlink) is now DEAD for the fleet — kept
            # only for the legacy from_file/from_blob self-test callers; the worker never produces a
            # tmpfs path anymore (no /dev/shm temp files).
            if isinstance(src, str):
                try:
                    sd = {k: (v if v.dtype == dt else v.to(dt)) for k, v in _load_file(src).items()}
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(src)
            else:
                sd = {k: (v if v.dtype == dt else v.to(dt)) for k, v in st_load(src).items()}
            sd = _fuse_moe_experts(sd, model)
            for t in sd.values():            # logical bf16 param count, data_ptr-deduped (matches __init__)
                if t.data_ptr() not in _seen:
                    _seen.add(t.data_ptr()); self.loaded_params += t.numel()
            try:
                model.load_state_dict(sd, strict=False, assign=True)
            except TypeError:
                # a trust_remote_code model may OVERRIDE load_state_dict() with a 4.x signature lacking
                # `assign` (e.g. MiniMax-M2). Use the base nn.Module loader to keep the assign-install
                # (meta param -> our streamed tensor). Its override only did qkv-split + fp8-filter,
                # which M2 doesn't need (separate q/k/v, bf16). #78
                torch.nn.Module.load_state_dict(model, sd, strict=False, assign=True)
            _assign_meta_from_sd(model, sd)   # materialize buffers load_state_dict skipped (non-persistent, e.g. MiniMax e_score_correction_bias)
            del sd

        def _drop_slice_mmap(module) -> None:
            # NO-OP since m4c25. The fleet now streams each slice straight into RAM bytes and st_load
            # builds HEAP tensors (load_state_dict assign=True installs them directly), so there is no
            # tmpfs mmap to release. The old path mmap'd a /dev/shm slice and had to clone every CPU
            # float param to heap to drop the mapping; with pure-bytes that clone is a redundant
            # full-layer copy that DOUBLES per-layer transient for nothing. Kept as a no-op so the
            # call sites (and the legacy from_file/from_blob mmap self-test paths) stay intact.
            return

        def _quant_after(kind: str, li: int) -> None:
            if kind == "layer":
                lyr = model.model.layers[li]
                if quant == "int4":
                    if stream_experts_nf and _layer_has_meta_experts_nonfused(lyr):
                        # NON-fused streamed (#78): fill the layer's meta expert Linears with int4
                        # QuantLinear4 FIRST, so the _quantize_int4_ walk below skips them (no longer
                        # nn.Linear). attn/router/shared Linears are resident -> quantized by that walk.
                        _quantize_experts4_streamed_nonfused(lyr, li, fetch_experts, dt)
                    _quantize_int4_(lyr)                    # 2D linears (attn, router, shared experts)
                    if stream_experts and _layer_has_meta_experts(lyr):
                        _quantize_experts4_streamed(lyr, li, fetch_experts, dt)   # fused experts (#62)
                    elif not stream_experts_nf:
                        _quantize_experts4_(lyr)           # fused experts from the resident blob
                elif quant == "int8":
                    _quantize_int8_(lyr)
                _drop_slice_mmap(lyr)                       # release this layer's tmpfs slice mmap
            elif kind == "head":
                if quant == "int8" and self.head is not None:
                    model.lm_head = _quantize_linear(model.lm_head); self.head = model.lm_head
                if self.head is not None: _drop_slice_mmap(self.head)
                if self.norm is not None: _drop_slice_mmap(self.norm)
            elif kind == "embed" and self.embed is not None:
                _drop_slice_mmap(self.embed)               # embed kept bf16 -> clone to heap, free shm

        def _install_cached(src, kind, li) -> None:
            # SERVE-FROM-CACHE install (#shard-cache Inc 2). The controller streamed this unit's tensors
            # ALREADY int4-packed (bit-identical to load-time quant), so build the resident holders
            # DIRECTLY and skip the ~4x bf16 stream + the per-layer quant/fuse entirely:
            #   * '<lin>.weight.{qweight,scale,zero}' (2D)  -> QuantLinear4 in place of the meta nn.Linear
            #   * '<experts>.{gate_up,down}_proj.{qweight,scale,zero}' (3D) -> Packed4Tensor3D Parameter
            #   * everything else (norms / biases / embed / head) is bf16 passthrough -> load_state_dict.
            # in_features comes from the meta module we replace (the cache stores padded widths only), so
            # NO manifest is needed on the worker. NEVER call _fuse_moe_experts / _quantize_* here. The
            # post-loop meta-guard catches any tensor we failed to materialize.
            sd = (_load_file(src) if isinstance(src, str) else st_load(src))
            QL = _quant4_linear_cls()
            PT = _packed4_3d_cls()
            G = _INT4_GROUP
            packed: dict = {}      # base -> {'q':qweight, 's':scale, 'z':zero}
            plain: dict = {}       # bf16 passthrough keys
            for k, v in sd.items():
                if k.endswith(".qweight"):
                    packed.setdefault(k[:-8], {})["q"] = v
                elif k.endswith(".scale"):
                    packed.setdefault(k[:-6], {})["s"] = v
                elif k.endswith(".zero"):
                    packed.setdefault(k[:-5], {})["z"] = v
                else:
                    plain[k] = (v if v.dtype == dt else v.to(dt))

            def _nav(path):
                parent = model
                for p in path.split("."):
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                return parent

            eager_cfgs: set = set()
            for base, tr in packed.items():
                qw, sc, ze = tr.get("q"), tr.get("s"), tr.get("z")
                if qw is None or sc is None or ze is None:
                    raise RuntimeError(f"cache: incomplete packed tensor {base!r} (have {sorted(tr)})")
                if qw.dim() == 3:                      # fused 3D MoE experts -> Packed4Tensor3D
                    ppath, _, attr = base.rpartition(".")
                    parent = _nav(ppath)
                    metap = parent._parameters.get(attr)
                    if metap is None:
                        raise RuntimeError(f"cache: no meta param for 3D expert {base!r}")
                    in_f = int(metap.shape[2])
                    delattr(parent, attr)              # drop the meta Parameter, install the int4 holder
                    setattr(parent, attr, PT(qw, sc, ze, in_f, G))
                    self.loaded_params += int(qw.shape[0]) * int(qw.shape[1]) * in_f
                    cfg = getattr(parent, "config", None)     # eager experts forward (per-expert index)
                    if cfg is not None and hasattr(cfg, "_experts_implementation") \
                            and id(cfg) not in eager_cfgs:
                        cfg._experts_implementation = "eager"; eager_cfgs.add(id(cfg))
                else:                                  # dense 2D Linear -> QuantLinear4
                    mod_path = base[:-7] if base.endswith(".weight") else base   # strip '.weight'
                    ppath, _, attr = mod_path.rpartition(".")
                    parent = _nav(ppath)
                    metalin = getattr(parent, attr)
                    # The cache packs every 2D layer '.weight'; the worker's load-time quant only ever
                    # produced QuantLinear4 from nn.Linear. A 2D packed weight that maps to a NON-Linear
                    # here = the compile over-captured (cache would not match a cold load) -> refuse to
                    # guess in_features; fail loud (silent wrong logits is the one thing the cache forbids).
                    if not hasattr(metalin, "in_features"):
                        raise RuntimeError(
                            f"cache: {mod_path!r} is {type(metalin).__name__}, not nn.Linear — "
                            "cache/model layout mismatch; refusing to serve a possibly-divergent cache")
                    in_f = int(metalin.in_features)
                    bp = plain.pop(mod_path + ".bias", None)   # this Linear's bf16 bias, if any
                    bias = (torch.nn.Parameter(bp, requires_grad=False) if bp is not None else None)
                    setattr(parent, attr, QL(qw, sc, ze, bias, in_f, G))
                    self.loaded_params += int(qw.shape[0]) * in_f
            if plain:                                  # bf16 passthrough: norms / embed / head / leftover
                try:
                    model.load_state_dict(plain, strict=False, assign=True)
                except TypeError:
                    torch.nn.Module.load_state_dict(model, plain, strict=False, assign=True)
                _assign_meta_from_sd(model, plain)     # materialize non-persistent buffers it skipped
                for t in plain.values():
                    if t.data_ptr() not in _seen:
                        _seen.add(t.data_ptr()); self.loaded_params += t.numel()
            del sd

        def _do_install(src, kind, li) -> None:
            if use_cache:
                _install_cached(src, kind, li)         # holders built directly from the pre-packed cache
            else:
                _install(src); _quant_after(kind, li)  # bf16 stream -> install -> quant (default path)

        # Jobs in pipeline order. tuple = (kind, layer_idx, start, end, embed, head)
        jobs = []
        if has_embed:
            jobs.append(("embed", -1, layer_start, layer_start, 1, 0))
        for i in range(layer_start, layer_end):
            jobs.append(("layer", i, i, i + 1, 0, 0))
        if has_head:
            jobs.append(("head", -1, layer_start, layer_start, 0, 1))

        # MEMORY BALLOON (#63, request A): RESERVE this shard's PLANNED resident RAM up front, then
        # consume it one chunk per shard as each layer/embed/head installs. Two guarantees the bare
        # streaming path lacks: (1) FAIL-FAST — if the node can't actually commit its planned share
        # the load aborts NOW with a clear error, not at 60% after minutes of streaming; (2) the
        # resident lands INTO the reservation rather than ON TOP of it, so a concurrent allocation
        # can't steal the node's share mid-build and peak stays ~ the plan (never plan + shard).
        # Sized from the controller's plan (`plan_ram_bytes` = this stage's est resident, which is
        # what is RAM-resident during the build for EVERY placement mode — GPU layers only move to
        # VRAM in _place_modules at the very end). Pages are faulted so the reservation is REAL, not
        # just committed address space. Missing plan_ram_bytes (old controller / rolling self-update)
        # -> no balloon (unchanged behavior). Only a genuine MemoryError aborts; any other balloon
        # hiccup degrades silently to plain streaming.
        _balloon: list = []
        def _balloon_release_one() -> None:
            if _balloon:
                _balloon.pop()          # free one chunk -> room for the shard about to install
        if plan_ram_bytes and plan_ram_bytes > 0:
            try:
                import numpy as _np
                _PAGE = 4096
                n_chunks = max(1, len(jobs))
                chunk_bytes = max(1 << 20, int(plan_ram_bytes) // n_chunks)   # >= 1 MB/chunk
                _built = []
                for _ in range(n_chunks):
                    _b = bytearray(chunk_bytes)              # commit charge reserved here
                    try:                                    # fault every page -> physically held
                        _np.frombuffer(_b, dtype=_np.uint8)[::_PAGE] = 1
                    except Exception:
                        pass                                # best-effort fault; commit still holds
                    _built.append(_b)
                _balloon = _built
                print(f"[load] reserved {chunk_bytes * n_chunks / GB:.1f} GB RAM balloon "
                      f"({n_chunks} chunks) for this shard's planned footprint", flush=True)
            except MemoryError:
                _balloon = []
                gc.collect()
                raise RuntimeError(
                    f"cannot reserve this shard's planned {plan_ram_bytes / GB:.1f} GB of resident "
                    f"RAM — node is short on memory; load aborted before streaming (fail-fast)")
            except Exception as _be:
                _balloon = []
                print(f"[load] memory balloon skipped ({_be!r}); streaming without pre-reservation",
                      flush=True)

        # PARALLEL PREFETCH: the build (load_state_dict + quant) MUST stay serial — it mutates the
        # shared model and load_state_dict iterates it, so concurrent builds would corrupt it. But
        # the fetches are independent I/O, so we run up to K concurrently into a bounded window and
        # build them in order as they arrive — overlapping network with build and letting the
        # controller hand out K layers at once instead of one-by-one. K is clamped to free RAM so
        # the in-flight bf16 blobs never overcommit a memory-tight node.
        from concurrent.futures import ThreadPoolExecutor
        def _fetch_job(j):
            if (stream_experts or stream_experts_nf) and j[0] == "layer":   # int4 MoE: layer blob WITHOUT experts (#62/#78)
                return fetch(j[2], j[3], j[4], j[5], skip_experts=True)
            return fetch(j[2], j[3], j[4], j[5])
        ex = ThreadPoolExecutor(max_workers=_STREAM_PREFETCH_MAX)
        try:
            inflight = {0: ex.submit(_fetch_job, jobs[0])}
            src0 = inflight.pop(0).result()
            # slot = one slice's bytes. Each in-flight prefetched slice costs ~slot of RAM (the bytes
            # buffer), plus another ~slot transiently while st_load copies it into heap tensors.
            slot = max(1, len(src0))
            # Bound the prefetch window by FREE RAM only (m4c25: no more /dev/shm tmpfs spool, so no
            # separate, smaller fs cap to honor). Recomputed each layer since the resident shard grows
            # as the load proceeds, so a K sized when the node was empty would over-commit it later.
            def _bound_K() -> int:
                try: a = psutil.virtual_memory().available
                except Exception: a = slot * 2
                return max(1, min(_STREAM_PREFETCH_MAX, int(a * 0.35 / slot)))
            K = _bound_K()
            _balloon_release_one()   # free this shard's chunk before it installs (consume the reservation)
            _do_install(src0, jobs[0][0], jobs[0][1]); del src0; gc.collect()
            nxt = 1
            for _ in range(min(K, len(jobs) - nxt)):          # prime the window
                inflight[nxt] = ex.submit(_fetch_job, jobs[nxt]); nxt += 1
            for idx in range(1, len(jobs)):
                src = inflight.pop(idx).result()              # wait this slice's fetch (in order)
                _balloon_release_one()   # free this shard's chunk before it installs (consume the reservation)
                _do_install(src, jobs[idx][0], jobs[idx][1]); del src; gc.collect()
                # Re-clamp the prefetch window to CURRENT free RAM each layer (#61): the resident
                # int4 shard GROWS as the load proceeds, so a K sized when the node was empty can
                # over-commit it 50+ layers later (the planner reserves ~1 layer's transient, not K
                # in-flight blobs). Recompute K each layer — it shrinks toward 1 as free RAM falls
                # (and grows back if it frees). idx+1 is always already in flight (the while keeps
                # >=1 ahead since K>=1), so the in-order pop stays safe.
                K = _bound_K()   # re-clamp to CURRENT free RAM each layer
                while nxt < len(jobs) and nxt < idx + 1 + K:   # keep <=K fetches in flight ahead
                    inflight[nxt] = ex.submit(_fetch_job, jobs[nxt]); nxt += 1
        finally:
            ex.shutdown(wait=True)
            _balloon.clear()                # drop any unconsumed reservation (success: already empty)
            for _f in inflight.values():    # on error, delete any prefetched tmpfs slices not installed
                with contextlib.suppress(Exception):
                    _r = _f.result(timeout=0)
                    if isinstance(_r, str):
                        os.remove(_r)
        if not _trust:                       # native: rotary built on meta -> rebuild for real inv_freq.
            rot = model.model.rotary_emb
            model.model.rotary_emb = type(rot)(self.cfg)
        else:
            # REMOTE-CODE rotary rebuild (#78, e.g. MiniMax-M2). Two transformers-5.x problems: (a) the
            # per-layer self_attn.rotary_emb modules were built under torch.device('meta') (workers lack
            # accelerate -> the include_buffers=False path is skipped), so their inv_freq buffers are META
            # — they'd trip the final meta guard and be unusable; (b) 5.x compute_default_rope_parameters
            # takes the rotary dim from config.head_dim and IGNORES partial_rotary_factor, so a PARTIAL-
            # rotary model (rotary_dim < head_dim, M2: 64 < 128) gets the WRONG width. Fix BOTH: rebuild
            # every owned layer's rotary AND a model-level model.model.rotary_emb from a corrected rope
            # config (head_dim:=rotary_dim, rope_parameters synthesized) -> REAL, correct-width inv_freq.
            # The shared forward feeds those cos/sin via position_embeddings; the layer partial-slices to
            # rotary_dim (no-op since already that width). _finalize_placement pins the model-level one.
            import copy as _copy
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            _rcfg = _copy.deepcopy(self.cfg)
            _rdim = int(getattr(self.cfg, "rotary_dim", 0) or 0)
            _hd = int(getattr(self.cfg, "head_dim", 0) or 0)
            if _rdim and _hd and _rdim < _hd:
                _rcfg.head_dim = _rdim                      # 5.x rotary width comes from head_dim
            if getattr(_rcfg, "rope_parameters", None) is None:
                _rcfg.rope_parameters = {"rope_type": "default",
                                         "rope_theta": float(getattr(self.cfg, "rope_theta", 10000.0))}
            def _mk_rotary():
                return LlamaRotaryEmbedding(_rcfg)
            for _lyr in self.owned_layers:                 # materialize (real) + correct-width per-layer
                _sa = getattr(_lyr, "self_attn", None)
                if _sa is not None and getattr(_sa, "rotary_emb", None) is not None:
                    with contextlib.suppress(Exception):
                        _sa.rotary_emb = _mk_rotary()
            if getattr(model.model, "rotary_emb", None) is None:   # model-level for the shared forward
                with contextlib.suppress(Exception):
                    model.model.rotary_emb = _mk_rotary()
        model.eval()
        # TP-v2: the row-parallel o_proj/down_proj produce PARTIAL outputs — sum them across the TP
        # group via the same forward-hook + _TPAllReduce wiring as the v1 __init__ path. The reduced
        # modules are already filled with this rank's slice (streamed above), so we only add hooks.
        if tp_size > 1:
            ar = tp_allreduce
            for lyr in self.owned_layers:
                lyr.self_attn.o_proj.register_forward_hook(lambda m, i, o: ar.allreduce(o))
                lyr.mlp.down_proj.register_forward_hook(lambda m, i, o: ar.allreduce(o))
        mods = ([self.embed] if has_embed else []) + list(self.owned_layers)
        if has_head:
            mods += [self.norm, self.head]
        self.loaded_bytes = sum(self._mod_bytes(m) for m in mods if m is not None)
        self.kv = None
        self.cpu = torch.device("cpu")
        self._streamed = True             # heap tensors (no mmap) -> _materialize_cpu_layers is a no-op
        self.cpu_materialized = True       # already heap-resident; parity with the from_file path
        # Guard: any owned tensor still on 'meta' here would make _place_modules' .to(device) die with
        # the cryptic 'copy out of meta tensor'. Surface the exact names instead (the int4 experts are
        # already real Packed4Tensor3D by now, so anything meta is a genuine un-served/skipped tensor).
        _stuck = []
        _scan = ([("embed", self.embed)] if has_embed else []) \
            + [("L%d" % (layer_start + _i), _l) for _i, _l in enumerate(self.owned_layers)] \
            + ([("norm", self.norm), ("head", self.head)] if has_head else [])
        for _tag, _m in _scan:
            if _m is None:
                continue
            for _n, _p in list(_m.named_parameters()) + list(_m.named_buffers()):
                if getattr(_p, "is_meta", False):
                    _stuck.append("%s.%s" % (_tag, _n))
        if _stuck:
            raise RuntimeError("unmaterialized meta tensor(s) after streamed build (would crash "
                               ".to(device)): " + ", ".join(_stuck[:12])
                               + (" ...+%d more" % (len(_stuck) - 12) if len(_stuck) > 12 else ""))
        self._moe_offload = bool(moe_offload)   # #moe-offload: split MoE layers (attn->GPU, experts->CPU)
        self._place_modules(device, gpu_mem_gb, ctx, gpu_budget_gb)   # ctx -> reserve full-ctx KV; budget -> #95 coexistence cap
        return self

    @classmethod
    def from_file(cls, config_dict: dict, weights_path: str, layer_start: int, layer_end: int,
                  has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                  device: str = "cpu", gpu_mem_gb: float = 0.0,
                  attn: str = "eager", quant: str = "none",
                  tp_rank: int = 0, tp_size: int = 1, tp_allreduce=None) -> "Shard":
        """Build a shard from a safetensors file via MEMORY-MAP (the fleet path).
        Unlike from_blob (raw bytes + tensors both resident => ~2x RAM), load_file
        mmaps the file, so peak RAM ~ the shard's resident size. This is what lets a
        big model (32B/70B) load on memory-tight nodes without OOM at load time."""
        import tempfile
        import torch
        from transformers import AutoConfig
        from safetensors.torch import load_file
        d = tempfile.mkdtemp(prefix="im_cfg_")
        try:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            cfg = AutoConfig.from_pretrained(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        dt = getattr(torch, dtype)
        sd = load_file(weights_path)   # mmap-backed tensors (zero-copy)
        sd = {k: (v if v.dtype == dt else v.to(dt)) for k, v in sd.items()}
        return cls(cfg, sd, layer_start, layer_end, has_embed, has_head, dt,
                   device=device, gpu_mem_gb=gpu_mem_gb, attn=attn, quant=quant,
                   tp_rank=tp_rank, tp_size=tp_size, tp_allreduce=tp_allreduce)

    @classmethod
    def from_hf(cls, model_id: str, layer_start: int, layer_end: int,
                has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                device: str = "cpu", gpu_mem_gb: float = 0.0,
                attn: str = "eager", quant: str = "none") -> "Shard":
        """Build a shard by reading directly from the HF cache (used by the
        standalone self-test; the fleet path uses from_blob)."""
        import torch
        from transformers import AutoConfig
        from huggingface_hub import snapshot_download
        cfg = AutoConfig.from_pretrained(model_id)
        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        model_dir = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
        wm = _weight_map(model_dir)
        names = _selected_names(wm, layer_start, layer_end, has_embed, has_head, tied)
        tensors = _load_tensors(names, wm)
        sd = _assemble_sd(tensors, layer_start, layer_end, has_embed, has_head, tied)
        dt = getattr(torch, dtype)
        sd = {k: v.to(dt) for k, v in sd.items()}
        return cls(cfg, sd, layer_start, layer_end, has_embed, has_head, dt,
                   device=device, gpu_mem_gb=gpu_mem_gb, attn=attn, quant=quant)

    def crop(self, length: int) -> None:
        """Truncate the KV cache to `length` tokens (speculative-decode rollback)."""
        if self.kv is not None:
            with contextlib.suppress(Exception):
                self.kv.crop(length)

    def _splice_mm(self, h, inject):
        """#22 increment 3 (embed-injection): replace the token embeddings at multimodal
        placeholder positions with the controller's precomputed image/audio embeds. Only
        stage 0 (has_embed) ever does this. h is [1, q, hidden]; inject = (positions, embeds)
        with embeds [len(positions), hidden]. Positions outside this frame are skipped."""
        torch = self.torch
        positions, embeds = inject
        idx = torch.as_tensor(list(positions), dtype=torch.long, device=h.device)
        emb = embeds.to(device=h.device, dtype=h.dtype)
        # 1) reconcile counts FIRST, so idx and emb are equal-length before any boolean mask
        #    (a bool mask must match the indexed dim — masking emb with idx's mask before
        #    trimming would IndexError when the counts differ).
        if idx.numel() != emb.shape[0]:                    # count mismatch -> splice the overlap
            n = min(idx.numel(), int(emb.shape[0]))
            idx, emb = idx[:n], emb[:n]
        # 2) drop any positions outside this frame (mask now matches both tensors' length)
        if idx.numel() and int(idx.max()) >= h.shape[1]:
            keep = idx < h.shape[1]
            idx, emb = idx[keep], emb[keep]
        h = h.clone()                                      # embed output may be a view; clone first
        if idx.numel():
            h[0, idx] = emb
        return h

    def kv_reserve_probe(self, ctx: int) -> None:
        """#2 pre-alloc safety: actually allocate the full-ctx KV this shard will grow into (on
        each device its layers sit on), then free it. If it OOMs, raise KV_RESERVE_OOM so the
        LOAD fails fast and clean — instead of the stage dying mid-decode and dropping its data
        connection. Full-attn KV is sized for EVERY layer (an overestimate for the hybrid
        linear-attn / Gated-DeltaNet layers -> conservative). Skipped if dims are unknown."""
        torch = self.torch
        cfg = self.cfg
        n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        n_kv = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // n_heads if n_heads else 0))
        if ctx <= 0 or n_kv <= 0 or head_dim <= 0:
            return
        per_layer = 2 * int(ctx) * n_kv * head_dim * 2   # k+v, bf16 = 2 bytes/elem
        by_dev: dict = {}
        for d in self.layer_devices:
            by_dev[d] = by_dev.get(d, 0) + per_layer
        held = []
        try:
            for dev, nbytes in by_dev.items():
                if nbytes > 0:
                    held.append(torch.empty(int(nbytes), dtype=torch.uint8, device=dev))
        except Exception as exc:
            total = sum(by_dev.values()) / GB
            raise RuntimeError(
                f"KV_RESERVE_OOM: cannot reserve {total:.2f} GB KV for ctx={ctx} on "
                f"{[str(d) for d in by_dev]}: {exc}") from exc
        finally:
            held.clear()
            import gc
            gc.collect()
            with contextlib.suppress(Exception):
                if any(getattr(d, "type", "") == "cuda" for d in by_dev):
                    torch.cuda.empty_cache()
        print(f"[load] KV reserve probe OK: {sum(by_dev.values())/GB:.2f} GB for ctx={ctx} "
              f"across {len(by_dev)} device(s)")

    def forward(self, x, cache_start: int = 0, reset: bool = True,
                all_logits: bool = False, inject=None, position_ids=None,
                capture_hidden: bool = False):
        """Run this stage's layers with an incremental KV cache (M2e).
        x = token ids (first stage) or hidden states (mid stage), covering the
        `q` positions starting at absolute position `cache_start`. `reset` starts
        a fresh cache (prefill); otherwise the cached prior KV is reused (decode).
        inject = (positions, embeds) splices multimodal embeds into stage-0's embed
        output (#22 inc 3); None for the normal text path.
        Returns hidden states, or — on the last stage — logits for just the last
        position, or for ALL positions when all_logits=True (speculative verify)."""
        torch = self.torch
        if reset or self.kv is None:
            from transformers import DynamicCache
            # Hybrid arch: a config-typed cache pre-creates the right per-layer slot
            # (conv+recurrent for linear-attn layers, KV for full-attn) so the
            # Gated-DeltaNet layers can store/read state instead of IndexError-ing on
            # an empty generic cache. Reused across prefill + every decode step.
            self.kv = DynamicCache(config=self.cfg) if self._hybrid else DynamicCache()
        with torch.inference_mode():
            if self.uniform_device is not None:
                return self._forward_uniform(x, cache_start, all_logits, inject, position_ids,
                                             capture_hidden)
            h = self.embed(x.to(self.embed_device)) if self.has_embed else x
            if inject is not None and self.has_embed:
                h = self._splice_mm(h, inject)
            q = h.shape[1]
            total = cache_start + q
            # Positional aux is built on CPU (rotary_emb lives there) and moved to
            # each layer's device on demand — cos/sin depend only on positions, so
            # this is identical to the single-device path. aux is per-call.
            ref = torch.empty(1, dtype=self.dtype)
            pos_cpu = torch.arange(cache_start, cache_start + q).unsqueeze(0)   # [1,q] -> layers
            # #22 inc 4: feed 3D mRoPE positions [3,1,q] to the rotary (it interleaves the t/h/w
            # sections and returns standard [bs,q,dim] cos/sin); layers keep 1D position_ids
            # (unused for rotary since we pass position_embeddings). None -> plain 1D arange.
            rot_pos = pos_cpu
            if position_ids is not None:
                rot_pos = torch.as_tensor(position_ids, dtype=torch.long)
                if rot_pos.dim() == 2:
                    rot_pos = rot_pos.unsqueeze(1)
            elif self._omni:   # Omni classic mRoPE needs [3,bs,seq]; text = 3x the same positions
                rot_pos = pos_cpu.unsqueeze(0).expand(3, -1, -1).contiguous()
            cos_cpu, sin_cpu = self.model.model.rotary_emb(ref, rot_pos)
            cos_cpu, sin_cpu = cos_cpu.to(self.dtype), sin_cpu.to(self.dtype)
            # additive mask (1,1,q,total): new position i attends keys 0..cache_start+i
            mask_cpu = torch.zeros((q, total), dtype=self.dtype)
            if q > 1:  # causal among the new tokens; prior keys all visible
                mask_cpu[:, cache_start:] = torch.triu(
                    torch.full((q, q), float("-inf"), dtype=self.dtype), diagonal=1)
            mask_cpu = mask_cpu.view(1, 1, q, total)
            cpos_cpu = torch.arange(cache_start, cache_start + q)
            aux: dict = {}

            def aux_for(dev):
                a = aux.get(dev)
                if a is None:
                    a = (mask_cpu.to(dev), pos_cpu.to(dev),
                         (cos_cpu.to(dev), sin_cpu.to(dev)), cpos_cpu.to(dev))
                    aux[dev] = a
                return a

            for layer, dev in zip(self.owned_layers, self.layer_devices):
                if h.device != dev:
                    h = h.to(dev)
                mask, pos, pos_emb, cache_position = aux_for(dev)
                if self._hybrid:
                    # Per-layer-type mask: full-attn gets the causal mask; linear-attn
                    # (Gated-DeltaNet) gets None (text-only, no padding). The qwen layer
                    # has no cache_position param (it tracks position via the cache).
                    m = mask if getattr(layer, "layer_type", "full_attention") == "full_attention" else None
                    out = layer(h, attention_mask=m, position_ids=pos,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=pos_emb)
                else:
                    out = layer(h, attention_mask=mask, position_ids=pos,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=pos_emb, cache_position=cache_position)
                h = out[0] if isinstance(out, tuple) else out
            if self.has_head:
                if h.device != self.head_device:
                    h = h.to(self.head_device)
                # #P6 speech: when capturing thinker hidden states for the talker, compute the
                # post-norm hidden for ALL positions (talker needs every prompt token at prefill
                # + each decoded token); logits only for the sampled position(s).
                nh = self.norm(h)
                sel = nh if all_logits else nh[:, -1:, :]   # verify needs every position
                logits = self.head(sel).to(self.cpu)
                if capture_hidden:
                    return logits, nh.to(self.cpu)
                return logits
            return h.to(self.cpu)

    def _forward_uniform(self, x, cache_start: int, all_logits: bool, inject=None,
                         position_ids=None, capture_hidden: bool = False):
        """Single-device fast path (see _finalize_placement). Everything — embed,
        every layer, norm/head, rotary — lives on self.uniform_device, so cos/sin,
        positions, cache_position and (for prefill only) the causal mask are built
        directly there. No per-token CPU rotary compute, no host->device copies,
        and for single-token decode no mask at all (the lone query attends every
        cached key). Numerically identical to the general path on the same device;
        on CPU it stays bit-exact. Called inside torch.inference_mode()."""
        torch = self.torch
        dev = self.uniform_device
        h = self.embed(x.to(dev)) if self.has_embed else x.to(dev)
        if inject is not None and self.has_embed:
            h = self._splice_mm(h, inject)
        q = h.shape[1]
        total = cache_start + q
        pos = torch.arange(cache_start, cache_start + q, device=dev).unsqueeze(0)   # [1,q] -> layers
        rot_pos = pos   # #22 inc 4: 3D mRoPE positions feed the rotary; see general path
        if position_ids is not None:
            rot_pos = torch.as_tensor(position_ids, dtype=torch.long, device=dev)
            if rot_pos.dim() == 2:
                rot_pos = rot_pos.unsqueeze(1)
        elif self._omni:   # Omni classic mRoPE needs [3,bs,seq]
            rot_pos = pos.unsqueeze(0).expand(3, -1, -1).contiguous()
        ref = torch.empty(1, dtype=self.dtype, device=dev)
        cos, sin = self.model.model.rotary_emb(ref, rot_pos)
        pos_emb = (cos.to(self.dtype), sin.to(self.dtype))
        cache_position = torch.arange(cache_start, cache_start + q, device=dev)
        if q > 1:   # prefill: causal among the new tokens; all prior keys visible
            mask = torch.zeros((q, total), dtype=self.dtype, device=dev)
            mask[:, cache_start:] = torch.triu(
                torch.full((q, q), float("-inf"), dtype=self.dtype, device=dev), diagonal=1)
            mask = mask.view(1, 1, q, total)
        else:       # decode: one query sees every key -> no mask needed
            mask = None
        for layer in self.owned_layers:
            if self._hybrid:
                m = mask if getattr(layer, "layer_type", "full_attention") == "full_attention" else None
                out = layer(h, attention_mask=m, position_ids=pos,
                            past_key_values=self.kv, use_cache=True,
                            position_embeddings=pos_emb)
            else:
                out = layer(h, attention_mask=mask, position_ids=pos,
                            past_key_values=self.kv, use_cache=True,
                            position_embeddings=pos_emb, cache_position=cache_position)
            h = out[0] if isinstance(out, tuple) else out
        if self.has_head:
            nh = self.norm(h)
            sel = nh if all_logits else nh[:, -1:, :]
            logits = self.head(sel).to(self.cpu)
            if capture_hidden:
                return logits, nh.to(self.cpu)
            return logits
        return h.to(self.cpu)


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
# Self-update: poll GitLab for a newer copy of THIS file; when idle, swap it in
# and exit(42) so the supervisor (systemd Restart=always on Linux / the .bat loop
# on Windows) relaunches the new code. Idle-gated so a running model is never cut.
# ---------------------------------------------------------------------------
SELF_UPDATE_POLL_S = 120   # poll GitLab every 2 minutes (fast deploys; idle-gated)


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
EXTRA_UPDATE_FILES: list[str] = ["wire.py", "config.json", "shards.py"]   # config + shared packer
# (#distributed-packing) synced like a module — shards.pack_unit_tensors is the shared packer the
# remote-pack handler calls, so a worker-packed cache unit is bit-identical to a controller-compiled one.


def _self_update_check(fname: str, is_idle) -> None:
    """Multi-file self-update: fetch the primary file + EXTRA_UPDATE_FILES, and if ANY changed
    (and we're idle) swap ALL changed files together, then restart. Abort the whole cycle if
    ANY file fails to fetch, so the on-disk module set never goes half-updated/inconsistent."""
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
    if not is_idle():
        print(f"[update] {changed} newer on GitLab - deferring (build in progress)")
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


class Worker:
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
        self.assignments: dict[str, dict] = {}                   # model_id -> load msg (debug/reload)
        self._weight_tmps: dict[str, str] = {}                   # model_id -> temp file backing its mmap
        self.data_server: asyncio.AbstractServer | None = None   # shared data port; bound on first load
        self._tp = None              # _TPAllReduce when a load is tensor-parallel (single TP model)
        self._tp_thread = None       # follower loop thread (peer ranks, tp_rank>0)
        self._tp_stop = False
        self._tp_model_id = None     # model_id currently in TP mode, if any
        # #22 inc 3: multimodal embeds staged by a 'mm' frame, consumed by the next prefill
        # for the same (model_id, req_id). Only stage 0 (has_embed) ever populates this.
        self.pending_mm: dict[tuple, tuple] = {}

    def _build_shard(self, base: str, model_id: str, a: dict) -> Shard:
        import tempfile
        cfg = json.loads(_http_get(f"{base}/modelmeta?model={urllib.parse.quote(model_id)}"))
        if cfg.get("auto_map"):                  # trust_remote_code model: also fetch its modeling .py
            with contextlib.suppress(Exception): # so the shard builds the CORRECT architecture instead
                rc = json.loads(_http_get(      # of transformers' native fallback class (which can
                    f"{base}/modelcode?model={urllib.parse.quote(model_id)}"))   # mismatch the ckpt)
                if isinstance(rc, dict) and rc:
                    cfg["__im_remote_code__"] = rc
        tp_size = int(a.get("tp_size", 1))
        tp_rank = int(a.get("tp_rank", 0))
        device = a.get("device") or self.device   # controller's per-node tier choice wins
        quant = a.get("quant") or self.quant      # load-time quant (controller) wins over launch flag
        plan_ram_bytes = int(a.get("plan_ram_bytes", 0) or 0)   # #63: planned resident RAM to reserve
        # #95: controller's committed-aware GPU budget for this stage (free VRAM after co-resident
        # models, minus the plan floor). -1 when the controller didn't send one (old controller) ->
        # the worker placement stays uncapped (legacy behavior). >=0 caps GPU placement in _place_modules.
        gpu_budget_gb = float(a.get("gpu_budget_gb", -1.0))
        # #moe-offload: controller opt-in to keep a MoE layer's attention+norms on GPU and leave the
        # routed-expert block in CPU RAM (llama.cpp --override-tensor experts=CPU, intra-layer).
        # Pipeline-only (the TP path ignores it); the worker further gates to int4 experts.
        moe_offload = bool(a.get("moe_offload", False))
        # #shard-cache Inc 2 (serve-from-cache): controller flags '' | 'int4'. When 'int4', fetch
        # PRE-PACKED int4 layer units (cache=int4 on /weights) and install them directly — no bf16
        # stream, no per-layer re-quant. Pipeline + int4 only (the controller never sets it for TP).
        cache = (a.get("cache", "") or "") if quant == "int4" else ""
        if tp_size <= 1:
            # DEFAULT PATH: stream each slice ONE LAYER AT A TIME straight into RAM bytes, then
            # st_load -> HEAP tensors (m4c25). NO temp files anywhere. The old path staged each slice
            # in a /dev/shm tmpfs file and mmap-loaded it; but tmpfs IS RAM, so it double-charged
            # memory, capped a node at its (smaller) shm size INDEPENDENT of free RAM, and ENOSPC'd
            # mid-load (the /dev/shm FileNotFound failure). And for CPU-resident layers it cloned to
            # heap anyway (_drop_slice_mmap), so there was no 1x win to keep. Pure bytes uses the
            # node's FULL RAM, frees the buffer per layer (~2x ONE layer transient — same as before),
            # and deletes the temp-file / ENOSPC / cleanup-race failure mode entirely.
            def fetch(start: int, end: int, embed: int, head: int, skip_experts: bool = False):
                qd = {"model": model_id, "start": start, "end": end,
                      "embed": int(bool(embed)), "head": int(bool(head)),
                      "skip_experts": int(bool(skip_experts))}
                if cache:
                    qd["cache"] = cache   # #shard-cache Inc 2: fetch the pre-packed int4 unit
                q = urllib.parse.urlencode(qd)
                url = f"{base}/weights?{q}"
                last = None
                for attempt in range(3):       # L+2 small fetches -> bounded retry vs LAN hiccups
                    try:
                        return _http_get(url)               # straight into RAM bytes -> heap tensors
                    except Exception as exc:
                        last = exc
                        time.sleep(0.5 * (attempt + 1))
                raise last

            def fetch_experts(layer: int, e0: int, k: int):
                # Stream a CHUNK of per-expert source tensors [e0:e0+k] of one MoE layer (#62) ->
                # dict {'{local_e}.{proj}': bf16}. Small blob (~chunk experts); in-RAM bytes is fine.
                from safetensors.torch import load as st_load
                q = urllib.parse.urlencode({"model": model_id, "layer": layer, "e0": e0, "k": k})
                url = f"{base}/experts?{q}"
                last = None
                for attempt in range(3):
                    try:
                        return st_load(_http_get(url))
                    except Exception as exc:
                        last = exc
                        time.sleep(0.5 * (attempt + 1))
                raise last

            shard = Shard.from_stream(cfg, fetch, a["layer_start"], a["layer_end"],
                                      a["has_embed"], a["has_head"], a.get("dtype", "bfloat16"),
                                      device=device, gpu_mem_gb=self.gpu_mem_gb,
                                      attn=self.attn, quant=quant, fetch_experts=fetch_experts,
                                      plan_ram_bytes=plan_ram_bytes, ctx=int(a.get("ctx", 0) or 0),
                                      gpu_budget_gb=gpu_budget_gb,   # #95 coexistence cap
                                      moe_offload=moe_offload,       # #moe-offload (pipeline only)
                                      cache=cache)                   # #shard-cache Inc 2 serve-from-cache
        else:
            # TENSOR-PARALLEL PATH (tp>1) — TP-v2 PER-RANK STREAMING: this rank fetches ONLY its
            # 1/tp tensor slice from /weights_tp and builds reduced-dim modules directly, so a node
            # holds ~1/tp of each layer (NOT the v1 load-full-then-shard footprint). Stand up the
            # all-reduce mesh BEFORE fetching so ranks rendezvous early (rank 0 binds; peers connect).
            tp = _TPAllReduce(tp_rank, tp_size, a.get("tp_root_host"), int(a.get("tp_root_port", 0)))
            self._tp = tp
            self._tp_model_id = model_id
            tpw = a.get("tp_weights")   # #68: per-rank capacity weights -> heterogeneous split (else uniform)
            wstr = ",".join(str(x) for x in tpw) if tpw else ""
            def fetch(start: int, end: int, embed: int, head: int, skip_experts: bool = False):
                # PER-RANK slice serve: /weights_tp returns this stage's tensors already sliced for
                # (tp_rank, tp_size). One blob per layer-slice (+ embed/head); small -> retry-bounded.
                # Straight into RAM bytes -> heap tensors; no /dev/shm temp file (m4c25, see pipeline path).
                qd2 = {"model": model_id, "start": start, "end": end,
                       "embed": int(bool(embed)), "head": int(bool(head)),
                       "tp_rank": tp_rank, "tp_size": tp_size}
                if wstr:
                    qd2["weights"] = wstr   # must match what _tp_make_structure_ built from
                q = urllib.parse.urlencode(qd2)
                url = f"{base}/weights_tp?{q}"
                last = None
                for attempt in range(3):
                    try:
                        return _http_get(url)
                    except Exception as exc:
                        last = exc
                        time.sleep(0.5 * (attempt + 1))
                raise last

            shard = Shard.from_stream(cfg, fetch, a["layer_start"], a["layer_end"],
                                      a["has_embed"], a["has_head"], a.get("dtype", "bfloat16"),
                                      device=device, gpu_mem_gb=self.gpu_mem_gb,
                                      attn=self.attn, quant=quant,
                                      tp_rank=tp_rank, tp_size=tp_size, tp_allreduce=tp,
                                      plan_ram_bytes=plan_ram_bytes, tp_weights=tpw,
                                      ctx=int(a.get("ctx", 0) or 0),
                                      gpu_budget_gb=gpu_budget_gb)   # #95 coexistence cap
        print(f"[load] stage L{a['layer_start']}-{a['layer_end']} placement: {shard.placement}"
              f" device={device} attn={self.attn} quant={quant} ({shard.loaded_bytes/GB:.2f} GB)")
        # #2 pre-alloc: reserve the full-ctx KV now so a node that can't hold it fails the LOAD
        # (clean, replannable) instead of OOMing mid-generation. ctx comes from the load msg.
        ctx = int(a.get("ctx", 0) or 0)
        if ctx > 0:
            shard.kv_reserve_probe(ctx)
        return shard

    def _cleanup_weight_tmp(self, model_id: str) -> None:
        tmp = self._weight_tmps.pop(model_id, None)
        if tmp:
            with contextlib.suppress(Exception):
                os.remove(tmp)

    def _cleanup_all_weight_tmps(self) -> None:
        for mid in list(self._weight_tmps):
            self._cleanup_weight_tmp(mid)

    async def handle_load(self, msg: dict) -> dict:
        model_id = msg["model_id"]
        a = msg
        # Reload of the SAME model: drop just its old shard first; keep other models resident.
        # (The Inc 1/2 controller still unloads every node before a load, so usually nothing
        # else is resident yet — this matters once Inc 3 enables fit-as-many.)
        await self._unload_model(model_id)
        self.assignments[model_id] = msg
        # A tensor-parallel PEER (tp_rank>0) is NOT in the pipeline: it has no data port and
        # no 'next' — it's driven entirely by rank 0's broadcasts over the all-reduce mesh.
        is_peer = int(a.get("tp_size", 1)) > 1 and int(a.get("tp_rank", 0)) > 0
        try:
            if not is_peer and self.data_server is None:
                # Bind the shared data port ONCE; every model's pipeline reuses it (frames
                # carry model_id so _data_inbound routes each frame to the right shard).
                try:
                    self.data_server = await asyncio.start_server(
                        self._data_inbound, "0.0.0.0", self.args.data_port)
                except OSError as exc:
                    raise RuntimeError(
                        f"data port {self.args.data_port} unavailable on "
                        f"{socket.gethostname()} ({exc}); give each worker on a shared "
                        f"host a distinct --data-port") from exc
            base = f"http://{self.args.controller}:{msg['controller_http_port']}"
            # EMBEDDING load (encoder, BERT-family): build the whole model on THIS one node — no
            # pipeline, no Shard, no KV. Acquire the weights via snapshot_download (mirrors from_hf's
            # pattern) but ALSO pull *.py so nomic's custom modeling/tokenizer code comes down; the
            # repo is public, so no token (matches from_hf). The encoder holder lives in self.shards
            # like a Shard but only serves kind:"embed" frames.
            if a.get("kind") == "embedding":
                import torch
                from huggingface_hub import snapshot_download
                device = a.get("device") or self.device
                if device == "":
                    device = self.device
                dtype = torch.float32   # encoder runs fp32 (CPU default; tiny model)
                self._building += 1
                try:
                    def _build_embed():
                        local_dir = snapshot_download(
                            model_id,
                            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt",
                                            "*.model", "tokenizer*"])
                        # auto-install any pip dep the model's trust_remote_code needs (#84)
                        return _build_with_autodeps(
                            lambda: EmbeddingModel(local_dir, device, dtype), label=model_id)
                    em = await asyncio.to_thread(_build_embed)
                    self.shards[model_id] = em
                finally:
                    self._building -= 1
                # No next hop: the encoder replies straight to the controller over its inbound conn.
                self.next_writers.pop(model_id, None)
                self.next_peer[model_id] = "controller"
                print(f"[load] embedding {model_id} on {em.device} "
                      f"({em.loaded_bytes / GB:.2f} GB, {em.loaded_params/1e6:.0f}M params)",
                      flush=True)
                return {"loaded_params": em.loaded_params,
                        "loaded_bytes": em.loaded_bytes,
                        "gpu_bytes": getattr(em, "gpu_bytes", 0)}
            self._building += 1   # mark BUSY across the build so reclaim/self-update can't kill it
            try:
                shard = await asyncio.to_thread(self._build_shard, base, model_id, a)
                self.shards[model_id] = shard
            finally:
                self._building -= 1
            if is_peer:
                self._tp_stop = False
                self._tp_thread = threading.Thread(target=self._tp_follow, daemon=True)
                self._tp_thread.start()
            else:
                next_host = a.get("next_host") or self.args.controller
                # Do NOT pre-connect the next hop at LOAD. A connection opened here then left idle
                # until the first generate can silently go half-open — reliably so on the FIRST
                # generate after a CONTROLLER RESTART, where the worker's logits write SUCCEEDS (no
                # exception) but the bytes never reach the controller, so m4bz's reconnect-on-error
                # never fires and the controller just waits out GEN_TIMEOUT (~600s). Confirmed by
                # tracing both ends: the worker's stage ran and its logits write returned success
                # (bytes "sent"), yet the controller's _on_data never received a frame for that req.
                # Leaving next_writers UNSET makes _send_next lazy-connect it
                # FRESH on the first send (zero idle gap) — exactly what a manual unload+reload did to
                # self-heal. Decode-step sends reuse that hot connection (rapid, no idle gap).
                self.next_writers.pop(model_id, None)
                # label the next hop for the bandwidth page: "controller" (last stage) or the
                # next worker's IP. a.get("next_host") is None on the last stage -> controller.
                self.next_peer[model_id] = "controller" if not a.get("next_host") else str(next_host)
            return {"loaded_params": shard.loaded_params,
                    "loaded_bytes": shard.loaded_bytes,
                    "gpu_bytes": getattr(shard, "gpu_bytes", 0),
                    "gpu_kv_bytes": getattr(shard, "gpu_kv_bytes", 0),
                    "placement": getattr(shard, "placement", None),   # observability + #moe-offload diag
                    "moe": getattr(shard, "_moe_dbg", None)}
        except Exception as exc:
            import traceback
            print(f"[load] {model_id} build FAILED: {exc!r}\n{traceback.format_exc()}", flush=True)
            await self._unload_model(model_id)  # drop the half-built shard; stay connected + idle
            # Do NOT _maybe_self_restart_if_stuck() here: a FAILED build that exit(42)s turns one
            # recoverable load failure into a restart -> reconnect -> retry -> fail loop that
            # desyncs the TP mesh (the observed churn). Reclaim belongs to the explicit-unload path;
            # if a failed partial build leaked RAM, the periodic mem flush / a controller-sent
            # unload reclaims it without an uncontrolled process exit.
            raise

    def _tp_follow(self) -> None:
        """Peer-rank loop: block for rank 0's broadcast input, run the sharded forward
        (all-reducing with the group via the mesh), discard the result, repeat. Exits on a
        stop broadcast (b'') or when the mesh drops (unload)."""
        import pickle
        while not self._tp_stop:
            try:
                payload = self._tp.recv_broadcast()
            except Exception:
                break
            if not payload or self._tp_stop:
                break
            try:
                data = pickle.loads(payload)
                # tuple grew over versions: 4 (base) -> 5 (+inject) -> 6 (+position_ids);
                # tolerate all during a rolling self-update.
                inject = position_ids = None
                if len(data) >= 6:
                    xt, cache_start, reset, all_logits, inject, position_ids = data[:6]
                elif len(data) == 5:
                    xt, cache_start, reset, all_logits, inject = data
                else:
                    xt, cache_start, reset, all_logits = data
                self.shards[self._tp_model_id].forward(xt, cache_start, reset, all_logits,
                                                       inject, position_ids)
            except Exception as exc:
                print(f"[tp-follow] forward error: {exc!r}")
                break

    def _teardown_tp(self) -> None:
        """Tear down the all-reduce mesh (TP is a single-model mode). Caller joins the thread."""
        if self._tp is not None:
            self._tp_stop = True
            if getattr(self._tp, "rank", 1) == 0:
                with contextlib.suppress(Exception):
                    self._tp.broadcast(b"")   # tell peers to leave their follow loop
            with contextlib.suppress(Exception):
                self._tp.close()
        self._tp = None
        self._tp_model_id = None

    def _maybe_self_restart_if_stuck(self) -> None:
        """After going fully idle (no shards), if this process still holds far more RAM than its
        fresh baseline, the OS didn't reclaim the dropped shard — Windows keeps committed private
        bytes until the process exits; glibc can retain freed arenas. A restart is then the ONLY
        reliable reclaim, so exit(42) and let the supervisor relaunch clean. Guards: only when
        fully idle (no live shard is dropped) and only well above baseline. A fresh process sits
        near baseline, so it never restart-loops. Call ONLY from explicit-unload paths, never from
        the unload at the START of handle_load (that would kill an incoming load)."""
        if self.shards or self._building:   # a resident shard OR an in-flight build -> not idle
            return
        try:
            import psutil
            rss_gb = psutil.Process().memory_info().rss / GB
        except Exception:
            return
        if rss_gb > self._rss_baseline_gb + 8.0:
            print(f"[reclaim] idle but still holding {rss_gb:.1f} GB (fresh baseline "
                  f"{self._rss_baseline_gb:.1f} GB) — OS won't reclaim it; restarting to free it",
                  flush=True)
            os._exit(42)   # supervisor relaunches on the same code -> RAM returns to the OS

    async def _unload_model(self, model_id: str) -> None:
        """Drop ONE model's shard + next-hop conn + temp file, keeping other models resident.
        If it was the TP model, tear the mesh + follower thread down too. Closes the shared
        data server only once no shards remain."""
        if model_id == self._tp_model_id:
            self._teardown_tp()
            if self._tp_thread is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._tp_thread.join, 5)
                self._tp_thread = None
        w = self.next_writers.pop(model_id, None)
        if w is not None:
            with contextlib.suppress(Exception):
                w.close()
        self.shards.pop(model_id, None)
        self.assignments.pop(model_id, None)
        # Drop any staged multimodal embeds for THIS model so they can't be mis-consumed by a
        # later request after a controller restart (req_id resets to 0 -> key reuse).
        for k in [k for k in self.pending_mm if k[0] == model_id]:
            self.pending_mm.pop(k, None)
        import gc
        gc.collect()
        _release_vram()   # return this shard's VRAM to the pool so the next load can use the GPU
        self._cleanup_weight_tmp(model_id)   # release mmap (gc above) then delete the temp file
        _release_ram(trim_working_set=not self.shards)   # return freed CPU RAM (trim only if now idle)
        if not self.shards and self.data_server is not None:
            with contextlib.suppress(Exception):
                self.data_server.close()
            self.data_server = None

    async def handle_pack(self, msg: dict) -> dict:
        """#distributed-packing Inc 1b: pack ONE shard-cache unit FOR the controller (offloads the
        slow per-layer pack off the controller + uses the fleet's idle CPUs). Fetch the unit's bf16
        from /weights (the SAME stream a load uses -> renamed 'model.*' dict), pack via the SHARED
        shards.pack_unit_tensors (so the result is BIT-IDENTICAL to a controller-local compile by
        construction), serialize, and POST it back to /pack_result. Dense int4/int8 only for now:
        the controller sends the EXACT quant scope (lin2d/exp3d from _quant_scope); per-expert MoE
        fusion (which needs the meta skeleton on the worker) is a later increment, so skel=None."""
        import base64
        import shards
        from safetensors.torch import load as st_load, save as st_save
        base = f"http://{self.args.controller}:{msg['controller_http_port']}"
        quant = msg.get("quant", "int4")
        gs = int(msg.get("group_size", shards.INT4_GROUP))
        _l, _e = msg.get("lin2d"), msg.get("exp3d")
        lin2d = set(_l) if _l is not None else None    # None -> pack_unit_tensors name-heuristic
        exp3d = set(_e) if _e is not None else None
        qd = {"model": msg["model_id"], "start": int(msg.get("start", 0)),
              "end": int(msg.get("end", 0)), "embed": int(bool(msg.get("embed", 0))),
              "head": int(bool(msg.get("head", 0))), "skip_experts": 0}
        url = f"{base}/weights?{urllib.parse.urlencode(qd)}"

        def _work() -> tuple[bytes, dict]:
            raw = st_load(_http_get(url))                       # {model.* : bf16}, same as compile's raw
            out_sd, mtensors = shards.pack_unit_tensors(raw, lin2d, exp3d, None, quant, gs)
            return st_save(out_sd), mtensors

        blob, mtensors = await asyncio.to_thread(_work)
        hdr = base64.b64encode(json.dumps(mtensors).encode()).decode()
        purl = (f"{base}/pack_result?req_id={urllib.parse.quote(str(msg['req_id']))}"
                f"&unit={urllib.parse.quote(str(msg['unit']))}"
                f"&model_id={urllib.parse.quote(str(msg['model_id']))}&quant={quant}")
        await asyncio.to_thread(_http_post, purl, blob, {"X-Manifest": hdr})
        return {"req_id": msg.get("req_id"), "unit": msg.get("unit"),
                "bytes": len(blob), "tensors": len(mtensors)}

    async def handle_unload(self, model_id: str | None = None) -> None:
        """Per-model unload when model_id is given; otherwise a FULL teardown of every model
        (what the controller sends today, and what the session does on disconnect)."""
        if model_id is not None:
            await self._unload_model(model_id)
            self._maybe_self_restart_if_stuck()
            return
        self._teardown_tp()
        if self._tp_thread is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._tp_thread.join, 5)
            self._tp_thread = None
        for w in self.next_writers.values():
            with contextlib.suppress(Exception):
                w.close()
        self.next_writers.clear()
        if self.data_server is not None:
            with contextlib.suppress(Exception):
                self.data_server.close()
            self.data_server = None
        self.shards.clear()
        self.assignments.clear()
        # Full teardown (incl. on controller disconnect) -> flush ALL staged multimodal embeds
        # so a fresh controller epoch (req_id from 0) can't pop a stale entry into a new prefill.
        self.pending_mm.clear()
        import gc
        gc.collect()
        _release_vram()   # return all freed VRAM to the pool (see _unload_model)
        self._cleanup_all_weight_tmps()
        _release_ram(trim_working_set=True)   # full teardown -> idle: trim heap back to the OS
        self._maybe_self_restart_if_stuck()   # if the OS still won't reclaim, restart for a clean slate

    async def _connect_next(self, host: str, port: int) -> asyncio.StreamWriter:
        deadline = time.time() + 30
        while True:
            try:
                # Bind the chosen LAN source so outbound activations to the next
                # pipeline stage ride the fast NIC too (not just inbound traffic).
                _r, w = await asyncio.open_connection(host, port, local_addr=_local_addr())
                _set_keepalive(w.get_extra_info("socket"))   # survive the load->generate idle gap
                return w
            except OSError:
                if time.time() > deadline:
                    raise
                await asyncio.sleep(0.4)

    async def _reconnect_next(self, model_id: str) -> asyncio.StreamWriter:
        """Re-dial this model's NEXT-hop data connection from its stored load assignment.
        The pipeline next-hop conn is opened at LOAD and then sits IDLE until the first
        generate — often many minutes later. Windows aborts an idle socket (ConnectionReset
        / WinError 10053 'software caused connection abort'), so the first send after the gap
        finds it dead. Left unhandled the stage's write fails, the error is swallowed (it was
        being reported down the SAME dead hop), and the controller just waits out GEN_TIMEOUT
        — the observed 'loads then breaks' hang. Reconnecting on a write failure self-heals
        the pipeline. host/port come from the saved load message (assignments[model_id])."""
        a = self.assignments.get(model_id)
        if a is None:
            raise RuntimeError(f"no load assignment for model_id={model_id!r}; can't reconnect next hop")
        old = self.next_writers.pop(model_id, None)
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        next_host = a.get("next_host") or self.args.controller   # last stage -> controller
        next_port = a["next_port"]
        w = await self._connect_next(next_host, next_port)
        self.next_writers[model_id] = w
        self.next_peer[model_id] = "controller" if not a.get("next_host") else str(next_host)
        print(f"[data] reconnected next hop for {model_id} -> "
              f"{self.next_peer[model_id]} ({next_host}:{next_port})", flush=True)
        return w

    async def _send_next(self, model_id: str, hdr: dict, raw: bytes) -> int:
        """Send one frame to this model's next hop, RECONNECTING once if the (possibly
        idle-dead) connection fails. This is what makes a distributed generation survive the
        load->first-generate idle gap (and a transient next-hop blip). Returns bytes sent;
        raises only if the next hop is genuinely unreachable after a fresh reconnect."""
        nxt = self.next_writers.get(model_id)
        if nxt is not None:
            try:
                return await _write_frame(nxt, hdr, raw)
            except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
                print(f"[data] next-hop send for {model_id} failed ({exc!r}); "
                      f"reconnecting + retrying once", flush=True)
        nxt = await self._reconnect_next(model_id)   # no writer, or the send just died
        return await _write_frame(nxt, hdr, raw)

    def _run_stage(self, model_id, x, cache_start, reset, all_logits, inject=None,
                   position_ids=None, capture_hidden=False):
        # TP rank 0 drives the group: broadcast this forward's input to the peers (who run
        # their sharded forward in lockstep), then run ours, all-reducing via the mesh hooks.
        if self._tp is not None and self._tp.rank == 0 and model_id == self._tp_model_id:
            import pickle
            # include inject + position_ids so peers (replicated embeddings + rotary) match
            self._tp.broadcast(pickle.dumps(
                (x.detach().to("cpu"), int(cache_start), bool(reset), bool(all_logits),
                 inject, position_ids)))
        return self.shards[model_id].forward(x, cache_start, reset, all_logits, inject,
                                             position_ids, capture_hidden)

    async def _data_inbound(self, reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter) -> None:
        # Identify who's on the OTHER end of this inbound data conn for per-peer accounting:
        # the controller (-> stage 0) or the previous-stage worker's IP (-> mid/last stage).
        _pn = writer.get_extra_info("peername")
        peer_in = "controller" if (_pn and _pn[0] == self.args.controller) else (
            str(_pn[0]) if _pn else "?")
        _set_keepalive(writer.get_extra_info("socket"))   # survive the load->generate idle gap

        try:
            while True:
                hdr, raw, _nb = await _read_frame(reader)
                _net_peer(peer_in, rx=_nb)
                # The data port is shared across models; route every frame by model_id.
                model_id = hdr.get("model_id")
                shard = self.shards.get(model_id)
                nxt = self.next_writers.get(model_id)
                if hdr.get("kind") == "error":  # propagate upstream error downstream
                    with contextlib.suppress(Exception):
                        await self._send_next(model_id, hdr, b"")   # reconnect-safe (reach the controller)
                    continue
                if hdr.get("kind") == "crop":  # speculative-decode KV rollback
                    if shard is not None:
                        shard.crop(int(hdr.get("cache_position", 0)))
                    with contextlib.suppress(Exception):  # propagate down the chain
                        if nxt is not None:
                            await _write_frame(nxt, hdr, b"")
                    continue
                if hdr.get("kind") == "mm":   # #22 inc 3: stage multimodal embeds for the
                    # next prefill of this (model_id, req_id). Stage-0 only; NOT forwarded down.
                    if shard is not None:
                        emb = _unpack_tensor(hdr, raw)
                        self.pending_mm[(model_id, hdr.get("req_id"))] = (
                            hdr.get("positions") or [], emb)
                    continue
                if hdr.get("kind") == "embed":   # encoder: ONE two-tensor frame (ids ++ mask) ->
                    # masked mean-pool + L2-norm vecs, replied straight to the controller. Mirrors
                    # the server's two-tensor hid_meta packing (primary meta = ids, mask_meta = mask).
                    try:
                        em = shard
                        if em is None:
                            raise RuntimeError(f"no embedding model for model_id={model_id!r} on this node")
                        ids_nbytes = int(hdr["ids_nbytes"])
                        ids = _unpack_tensor(hdr, raw[:ids_nbytes])
                        mask = _unpack_tensor(hdr["mask_meta"], raw[ids_nbytes:])
                        vecs = await asyncio.to_thread(em.encode, ids, mask)
                        vmeta, vraw = _pack_tensor(vecs)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id,
                                "kind": "embedding", **vmeta}
                        _tx = await self._send_next(model_id, ohdr, vraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)
                    except Exception as exc:   # mirror the stage-error path: tell the controller
                        import traceback
                        tb = traceback.format_exc()
                        print(f"[data] embed error: {exc!r}\n{tb}")
                        frames = [ln.strip() for ln in tb.splitlines() if ln.strip().startswith("File ")]
                        with contextlib.suppress(Exception):
                            await self._send_next(model_id, {
                                "req_id": hdr.get("req_id"), "model_id": model_id, "kind": "error",
                                "error": f"{exc!r} | " + " <- ".join(frames[-3:])}, b"")
                    continue
                cache_start = int(hdr.get("cache_position", 0))
                reset = bool(hdr.get("reset", True))
                all_logits = bool(hdr.get("all_logits", False))
                try:
                    if shard is None:
                        raise RuntimeError(f"no shard for model_id={model_id!r} on this node")
                    x = _unpack_tensor(hdr, raw)
                    inject = self.pending_mm.pop((model_id, hdr.get("req_id")), None)
                    # #22 inc 4: 3D mRoPE positions ride the frame header (small list); every
                    # stage uses them for its rotary, so propagate them to the next stage too.
                    position_ids = hdr.get("position_ids")
                    # #P6 speech: capture thinker hidden states for the talker. The flag rides
                    # the header down the chain so the LAST stage (has_head) returns the
                    # post-norm hidden alongside the logits in a two-tensor result frame.
                    capture_hidden = bool(hdr.get("capture_hidden", False))
                    out = await asyncio.to_thread(self._run_stage, model_id, x, cache_start,
                                                  reset, all_logits, inject, position_ids,
                                                  capture_hidden)
                    kind = "logits" if shard.has_head else "hidden"
                    if capture_hidden and shard.has_head and isinstance(out, tuple):
                        logits_t, hidden_t = out
                        lmeta, lraw = _pack_tensor(logits_t)
                        hmeta, hraw = _pack_tensor(hidden_t)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id, "kind": kind,
                                "cache_position": cache_start, "reset": reset,
                                "all_logits": all_logits, **lmeta,
                                "logits_nbytes": len(lraw), "hid_meta": hmeta}
                        if position_ids is not None:
                            ohdr["position_ids"] = position_ids
                        _tx = await self._send_next(model_id, ohdr, lraw + hraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)
                    else:
                        meta, oraw = _pack_tensor(out)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id, "kind": kind,
                                "cache_position": cache_start, "reset": reset,
                                "all_logits": all_logits, **meta}
                        if position_ids is not None:
                            ohdr["position_ids"] = position_ids
                        # propagate the capture flag to the next stage so it reaches the head stage
                        if capture_hidden and not shard.has_head:
                            ohdr["capture_hidden"] = True
                        _tx = await self._send_next(model_id, ohdr, oraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)   # to next stage / controller
                except Exception as exc:  # stage failed -> tell the controller, fast
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[data] stage error: {exc!r}\n{tb}")
                    # surface the deepest FILE:line frames (skip caret-only lines) to the controller.
                    # Route the error through _send_next so it RECONNECTS the next hop first — else a
                    # downstream-conn failure would report the error over the SAME dead socket and be
                    # swallowed (the controller then only sees a GEN_TIMEOUT, never the real cause).
                    frames = [ln.strip() for ln in tb.splitlines() if ln.strip().startswith("File ")]
                    with contextlib.suppress(Exception):
                        await self._send_next(model_id, {
                            "req_id": hdr.get("req_id"), "model_id": model_id,
                            "kind": "error",
                            "error": f"{exc!r} | " + " <- ".join(frames[-3:])}, b"")
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:  # pragma: no cover
            print(f"[data] inbound error: {exc!r}")
        finally:
            with contextlib.suppress(Exception):
                writer.close()


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
                        await reply({"type": "ready", "node_id": node_id, **info})
                        if msg.get("kind") == "embedding":   # no layer range — whole encoder on one node
                            print(f"[load] embedding {msg.get('model_id')} "
                                  f"({info['loaded_bytes']/GB:.2f} GB)")
                        else:
                            print(f"[load] stage {msg.get('stage')} "
                                  f"layers {msg['layer_start']}-{msg['layer_end']} "
                                  f"({info['loaded_bytes']/GB:.2f} GB, RAM-only)")
                    except Exception as exc:
                        await reply({"type": "error", "node_id": node_id, "error": repr(exc)})
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


if __name__ == "__main__":
    main()
