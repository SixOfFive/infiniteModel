"""worker_hw: hardware/host-facing worker helpers, relocated VERBATIM from client.py
(code-split Inc 7): memory/GC (_release_vram/_release_ram/_flush_os_cache/mem_maintenance_loop),
the capability probes (detect_device/_gpu_mem_gb/_rocm_gpu_util/_using_gpu/free_disk_gb), the
READ-ONLY network-route detectors (_os_default_src/_iface_kind/select_route/_controller_is_local/
_fmt_route), RAM-module detection (_fmt_ram_mods/_detect_ram_windows/detect_ram),
build_registration, and startup cleanup (_dir_size/cleanup_storage).

HARD RULE (do not "tidy" this): _ROUTE_SRC and _local_addr STAY in client.py -- they are the
LIVE rebind pair (session()/run() rebind _ROUTE_SRC after route selection); relocating them
would freeze a stale copy and the data plane would dial a dead source IP.

Bodies BYTE-IDENTICAL; module globals (GB, HOME, CHUNK_DIR, VERSION, psutil, print(timestamped),
...) are injected at startup by state.bind() -- see state.py. Worker-side leaf; never imports
client; listed in client.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def _release_vram() -> None:
    """Return freed GPU memory to the pool after a shard is dropped. torch's caching
    allocator holds freed VRAM reserved inside the process, so without this the next
    model load sees ~0 free VRAM and spills its shard to RAM. Safe with other shards
    resident — empty_cache only frees UNUSED cached blocks, never in-use tensors. No-op
    on CPU-only workers / when torch isn't importable."""
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            # #vram-release-rocm: on ROCm/HIP (gfx1151 APU) a dropped shard's VRAM was seen NOT
            # returned by empty_cache alone — unload left the model resident (~13.5 GB int4 / ~44 GB
            # bf16) and only a process restart reclaimed it. Pending HIP frees can be DEFERRED until
            # the stream syncs, so sync FIRST so empty_cache sees the freed blocks. Then log
            # allocated-vs-reserved (ROCm only) so a lingering live REFERENCE (allocated stays high
            # after gc+empty_cache) is told apart from an allocator POOL the release SHOULD reclaim
            # (reserved high, allocated low). Harmless + silent on CUDA (BEAST behavior unchanged).
            is_rocm = bool(getattr(torch.version, "hip", None))
            if is_rocm:
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if is_rocm:
                with contextlib.suppress(Exception):
                    a = torch.cuda.memory_allocated() / (1024 ** 3)
                    r = torch.cuda.memory_reserved() / (1024 ** 3)
                    _builtins.print(f"[vram-release] ROCm after gc+empty_cache: "
                                    f"allocated={a:.2f}GB reserved={r:.2f}GB "
                                    f"({'LIVE-REF leak' if a > 1.0 else 'pool reclaimed' if r < 1.0 else 'POOL held'})",
                                    flush=True)
                    if a > 2.0:   # #39: name WHAT is still live so the leak stops being a mystery
                        _dump_live_cuda()


def _dump_live_cuda(top: int = 8) -> None:
    """#39 diagnostic: when an unload leaves GBs ALLOCATED (live refs, not pool), walk gc for
    live CUDA tensors, group by (shape, dtype), and print the top groups + the REFERRER type
    names of the biggest tensor — enough to name the holder (KV cache list? module? closure?).
    Best-effort and cheap relative to an unload; only called on the >2 GB residue path."""
    import gc as _gc
    with contextlib.suppress(Exception):
        import torch
        groups: dict = {}
        biggest = None
        for o in _gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    b = o.numel() * o.element_size()
                    k = (tuple(o.shape), str(o.dtype))
                    n, tot = groups.get(k, (0, 0))
                    groups[k] = (n + 1, tot + b)
                    if biggest is None or b > biggest[0]:
                        biggest = (b, o)
            except Exception:
                continue
        top_g = sorted(groups.items(), key=lambda kv: kv[1][1], reverse=True)[:top]
        for (shape, dt), (n, tot) in top_g:
            _builtins.print(f"[vram-live] {n}x {shape} {dt} = {tot / (1024 ** 3):.2f} GB",
                            flush=True)
        # name the OWNER of the (int, big-cuda-Tensor) tuples: dump the int values (layer index?
        # stream id?) and walk each tuple's referrers INCLUDING frames — a suspended coroutine /
        # generator frame prints its exact source location.
        allobjs = _gc.get_objects()
        ints: list = []
        shown = 0
        for o in allobjs:
            try:
                if not (isinstance(o, tuple) and len(o) == 2 and isinstance(o[0], int)
                        and torch.is_tensor(o[1]) and o[1].is_cuda
                        and o[1].numel() * o[1].element_size() > (1 << 27)):
                    continue
            except Exception:
                continue
            ints.append(o[0])
            if shown < 2:
                shown += 1
                _builtins.print(f"[vram-live] tuple (int {o[0]}, {tuple(o[1].shape)})", flush=True)
                for r in _gc.get_referrers(o)[:6]:
                    tn = type(r).__name__
                    if tn == "frame":
                        with contextlib.suppress(Exception):
                            _builtins.print(f"[vram-live]   FRAME {r.f_code.co_filename}:"
                                            f"{r.f_lineno} in {r.f_code.co_name}", flush=True)
                        continue
                    if isinstance(r, dict):
                        key = next((str(k)[:60] for k, v in r.items() if v is o), "?")
                        owner = next((type(x).__name__ for x in allobjs
                                      if getattr(x, "__dict__", None) is r), "plain-dict")
                        _builtins.print(f"[vram-live]   owner: {owner}[{key}]", flush=True)
                    elif tn not in ("list_iterator",):
                        _builtins.print(f"[vram-live]   ref: {tn}"
                                        + (f" len={len(r)}" if isinstance(r, (list, tuple)) else ""),
                                        flush=True)
        _builtins.print(f"[vram-live] {len(ints)} such tuples; ints={sorted(ints)[:12]}…", flush=True)


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


def _rocm_gpu_util():
    """GPU compute-busy % on ROCm/AMD via the bundled `rocm-smi` CLI. On ROCm,
    torch.cuda.utilization() needs the `amdsmi` python binding, which the TheRock
    gfx1151 wheels don't ship — so the CUDA telemetry path returns nothing on AMD
    nodes. This keeps the dashboard's GPU% 1:1 with CUDA. Guarded to ROCm (returns
    None on CUDA/CPU so those nodes keep using torch.cuda.utilization / pynvml).
    Best-effort; None on any failure. -> float | None."""
    try:
        import torch
        if not getattr(torch.version, "hip", None):
            return None
    except Exception:
        return None
    import os, sys, shutil, subprocess, re
    exe = shutil.which("rocm-smi") or os.path.join(sys.prefix, "bin", "rocm-smi")
    if not exe or not os.path.exists(exe):
        return None
    try:
        out = subprocess.run([exe, "--showuse"], capture_output=True,
                             text=True, timeout=4).stdout
    except Exception:
        return None
    m = re.search(r"GPU use \(%\):\s*([0-9]+)", out)
    return float(m.group(1)) if m else None


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
    # #worker-roles: a role-limited worker (e.g. --roles t2a for a dedicated ACE-Step worker on a
    # venv the LLM stack can't share) advertises exactly what it serves. The controller routes only
    # those roles to it and, unless "llm" is among them, keeps it out of LLM/t2i/tts/embed planning.
    _roles = [r.strip() for r in str(getattr(args, "roles", "") or "").split(",") if r.strip()]
    if _roles:
        reg["caps"] = _roles
    if _using_gpu(args):
        reg["vram_total_gb"] = round(_gpu_mem_gb()[1], 2)
    return reg


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
