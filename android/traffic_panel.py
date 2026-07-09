#!/usr/bin/env python3
"""
InfiniteModel tablet bandwidth panel  (standalone, READ-ONLY).

Polls the controller's /status and draws a static, non-scrolling table of every node's
traffic to the rest of the fleet (node <-> all) plus the controller and a fleet TOTAL.
Each node spans TWO rows: a stats line topped by a GREEN download sparkline, and a RED
upload sparkline beneath it.

  DOWN / UP  - current in / out rate (DOWN green, UP red)
  FROM→…→NEXT- the node's place in the ROUND TRIP, INFERRED from the loaded models' pipeline
               placement: where its data comes FROM and goes NEXT (from→node→next). The train
               leaves 'home' (the controller), runs out through the nodes, DROPS OFF at the head,
               and returns all the way back to 'home'. So a head reads '…→node→home', stage 0
               reads 'home→node→…', and the controller row pivots the loop ('head→home→stage0').
               'mesh' = a tensor-parallel stage (no single hop); blank = not in a loaded pipeline
  RESIDENT   - what's loaded ON this machine: <models>m <layers>L <GB> (summed across every
               model that has a stage here)
  MAX        - peak rate within the DISPLAYED sparkline window (the busier direction); both
               sparklines scale to it (full bar == this window's max)
  XFER       - bytes moved within the DISPLAYED window (the visible sparkline span) -- NOT
               lifetime; rows sort busiest-first by this

MAX and XFER cover ONLY the window currently shown (the last ~N seconds of bars), not traffic
before that. NEXT is an INFERENCE from placement, not a measured per-edge rate (the controller
doesn't expose per-peer bytes over /status).

Native Termux (no proot/torch); pure /status consumer -- touches nothing in the fleet client.
  python traffic_panel.py [controller_ip] [poll_seconds]
"""
import json
import os
import sys
import time
from collections import deque, defaultdict
from urllib.request import urlopen

CTRL_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.15.38"
POLL    = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
URL     = f"http://{CTRL_IP}:21434/status"

HIST       = 600                                   # in/out rate samples kept per row (>= sparkline)
LOG_PATH   = os.path.expanduser("~/.im/traffic.csv")
LOG_CAP    = 4 * 1024 * 1024
LOG_EVERY  = 10.0
SPARK      = " ▁▂▃▄▅▆▇█"
IDLE_FLOOR = 2048                                   # below this peak (B/s) a row's spark stays flat
CTRL       = "controller"                           # controller's row name + bookend of every chain

GRN, RED, DIM, MAG, BOLD, RST = (
    "\033[32m", "\033[31m", "\033[2m", "\033[35m", "\033[1m", "\033[0m")

hin  = defaultdict(lambda: deque(maxlen=HIST))     # download (in) rate samples per row
hout = defaultdict(lambda: deque(maxlen=HIST))     # upload  (out) rate samples per row
_last_log = [0.0]
_prev_cpu = [None]                                 # (total, idle) /proc/stat jiffies, for a delta CPU%
_cpu_hist = deque(maxlen=HIST)                      # this tablet's CPU% samples (for the cpu sparkline)


def human(bps):
    bps = float(bps or 0)
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if bps >= div:
            return f"{bps / div:5.1f}{unit}/s"
    return f"{bps:5.0f} b/s"


def human_bytes(b):
    b = float(b or 0)
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if b >= div:
            return f"{b / div:5.1f}{unit}B"
    return f"{b:5.0f} B"


def tablet_cpu():
    """(pct, label) for THIS tablet's CPU. `pct` is a float 0-100 (or None), `label` the display
    string. Primary: /proc/stat busy% (delta between polls). Android denies app-UID processes
    /proc/stat + /proc/loadavg, so fall back to a cpufreq clock proxy (mean cur/max freq, 'clk')."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)      # idle + iowait
        total = sum(v)
        prev = _prev_cpu[0]
        _prev_cpu[0] = (total, idle)
        if prev is not None and total > prev[0]:
            dt, di = total - prev[0], idle - prev[1]
            p = max(0.0, min(100.0, 100.0 * (dt - di) / dt))
            return p, f"{p:.0f}%"
        return None, "…"                               # /proc/stat readable, warming up
    except Exception:
        pass
    try:                                               # fallback: cpufreq clock-load proxy
        import glob
        rs = []
        for cf in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
            try:
                c = int(open(cf).read())
                mx = int(open(cf.replace("scaling_cur_freq", "cpuinfo_max_freq")).read())
                if mx:
                    rs.append(c / mx)
            except Exception:
                pass
        if rs:
            p = 100.0 * sum(rs) / len(rs)
            return p, f"{p:.0f}% clk"
    except Exception:
        pass
    return None, "n/a"


def pct_spark(values, width, ceiling=100.0):
    """Sparkline of the last `width` percent-values scaled 0..ceiling (0 -> blank, ceiling -> full).
    Unlike spark(), no byte noise-floor — meant for a 0-100 CPU graph."""
    s = list(values)[-width:]
    if not s:
        return " " * width
    mx = ceiling or (max(s) or 1)
    cells = "".join(SPARK[min(len(SPARK) - 1, int(min(max(v, 0.0), mx) / mx * (len(SPARK) - 1)))]
                    for v in s)
    return cells.rjust(width)


def tablet_mem():
    """THIS tablet's (used_kB, total_kB) from /proc/meminfo, or None."""
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if p:
                    mem[p[0].rstrip(":")] = int(p[1])
        total = mem.get("MemTotal", 0)
        avail = mem.get("MemAvailable", mem.get("MemFree", 0))
        return total - avail, total
    except Exception:
        return None


def tablet_load():
    """THIS tablet's 1-minute load average, or None."""
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


_cver_cache = [None]                               # cached VERSION read from the guest's client.py


def _local_cver():
    """VERSION of the worker's client.py deployed in the proot guest on THIS tablet (cached), or ''.
    A fallback for the title when the worker isn't registered (so /status has no version for us)."""
    if _cver_cache[0] is not None:
        return _cver_cache[0]
    import glob
    import re
    v = ""
    for p in glob.glob("/data/data/com.termux/files/usr/var/lib/proot-distro/"
                       "containers/*/rootfs/root/android/client.py"):
        try:
            with open(p) as f:
                m = re.search(r'^VERSION\s*=\s*"([^"]+)"', f.read(4000), re.M)
            if m:
                v = m.group(1)
                break
        except Exception:
            pass
    _cver_cache[0] = v
    return v


def spark(window, width, ceiling):
    """Sparkline of the last `width` samples, scaled to `ceiling` (the window's own max) so the
    tallest displayed bar is full. Idle rows (below the noise floor) render flat."""
    s = list(window)[-width:]
    if not s:
        return " " * width
    mx = ceiling if (ceiling and ceiling > IDLE_FLOOR) else (max(s) or 1)
    if mx < IDLE_FLOOR:
        return ("▁" * len(s)).rjust(width)
    cells = "".join(SPARK[min(len(SPARK) - 1, 1 + int(min(v, mx) / mx * (len(SPARK) - 2)))]
                    for v in s)
    return cells.rjust(width)


def track(name, i, o):
    hin[name].append(i)
    hout[name].append(o)


def wstats(name, sw):
    """Peak rate and total bytes over ONLY the displayed window (the last `sw` samples) -- so MAX
    and XFER reflect what's on screen, not lifetime. Returns (peak, win_bytes)."""
    dn = list(hin[name])[-sw:]
    up = list(hout[name])[-sw:]
    peak = max((max(dn) if dn else 0.0), (max(up) if up else 0.0))
    win_bytes = (sum(dn) + sum(up)) * POLL
    return peak, win_bytes


def build_edges(st):
    """Derive where each node sends activations next, from every loaded PIPELINE model's ordered
    stages (controller -> stage0 -> ... -> head -> controller). Returns (down, mesh):
      down  - hostname -> [downstream hostname, ...]  (insertion order, de-duped)
      mesh  - hostnames in a tensor-parallel stage (within-stage all-reduce; no single downstream)
    Hostnames are truncated like the table's row keys so they match. Pure /status reader."""
    down = defaultdict(list)                  # node -> downstream (where its data GOES next)
    up = defaultdict(list)                     # node -> upstream (where its data comes FROM)
    mesh = set()

    def add(a, b):
        if a and b:
            if b not in down[a]:
                down[a].append(b)
            if a not in up[b]:
                up[b].append(a)

    cl = (st or {}).get("cluster") or {}
    seen = set()
    for mdl in (cl.get("loaded_models") or []):
        stages = mdl.get("stages") or []
        if not stages:
            continue
        if mdl.get("is_tp") or (mdl.get("tp_size") or 1) > 1:   # TP stage = mesh, not a clean hop
            for s in stages:
                h = (s.get("hostname") or "")[:12]
                if h:
                    mesh.add(h)
            continue
        chain = [(s.get("hostname") or "")[:12]
                 for s in sorted(stages, key=lambda s: (s.get("layer_start") or 0))
                 if s.get("hostname")]
        if not chain:
            continue
        key = tuple(chain)
        if key in seen:                       # same placement reported twice (primary + resident)
            continue
        seen.add(key)
        prev = CTRL                           # controller feeds stage 0 ...
        for h in chain:
            add(prev, h)
            prev = h
        add(prev, CTRL)                       # ... and the head returns logits to the controller (home)
    return down, up, mesh


def route_label(host, up, down, mesh):
    """The node's place in the round trip: 'FROM→node→NEXT' — where its data comes from and where
    it goes. 'home' = the controller (the train's origin AND where the head drops off back to).
    A head stage reads '…→node→home'; stage 0 reads 'home→node→…'; the controller row pivots the
    loop ('head→home→stage0'). Blank when the node holds no loaded pipeline; 'mesh' for TP."""
    frm = (up.get(host) or [None])[0]
    nxt = (down.get(host) or [None])[0]
    if frm is None and nxt is None:
        return "mesh" if host in mesh else ""

    def nm(x):
        if x is None:
            return "·"
        return "home" if x == CTRL else x[:6]

    self_s = "home" if host == CTRL else host[:6]
    return (nm(frm) + "→" + self_s + "→" + nm(nxt))[:20]


def build_resident(st):
    """What's resident on each machine, from the loaded models' stages. Returns (per, n_models):
      per      - hostname -> (model_count, layer_count, gb) loaded on that node
      n_models - distinct models loaded across the fleet
    A node holding several models sums their layers + estimated weight GB."""
    agg = defaultdict(lambda: [set(), 0, 0.0])
    models = set()
    for mdl in ((st or {}).get("cluster", {}).get("loaded_models") or []):
        mid = (mdl.get("friendly") or mdl.get("display_name") or mdl.get("target")
               or str(mdl.get("loaded_at_ts")))
        models.add(mid)
        for s in (mdl.get("stages") or []):
            h = (s.get("hostname") or "")[:12]
            if not h:
                continue
            agg[h][0].add(mid)
            agg[h][1] += int(s.get("num_layers") or 0)
            agg[h][2] += float(s.get("est_gb") or 0.0)
    per = {h: (len(v[0]), v[1], v[2]) for h, v in agg.items()}
    return per, len(models)


def resid_label(v):
    if not v or not v[0]:
        return ""
    nm, nl, gb = v
    g = f"{gb:.1f}" if gb < 10 else f"{gb:.0f}"
    return f"{nm}m {nl}L {g}G"


# Columns: NODE DOWN UP ROUTE RESIDENT MAX XFER. DOWN is idx 1 (green), UP idx 2 (red).
COLW = ((10, "<"), (9, ">"), (9, ">"), (20, "<"), (12, "<"), (9, ">"), (9, ">"))
PREFIX_W = 1 + sum(w for w, _ in COLW) + (len(COLW) - 1)    # leading sp + cells + single-space gaps


def fmt_prefix(vals, base="", color=False):
    """The fixed-width column block (no sparkline). DOWN/UP get colored when `color`, each
    returning to `base` after so a base-colored row (controller/total) stays intact."""
    cells = [f"{v:{a}{w}}" for v, (w, a) in zip(vals, COLW)]
    if color:
        cells[1] = f"{GRN}{cells[1]}{RST}{base}"
        cells[2] = f"{RED}{cells[2]}{RST}{base}"
    return (base + " " + " ".join(cells)) if base else (" " + " ".join(cells))


def node_lines(name, i, o, nxt, resid, peak, winx, base, color, cols, sw):
    """Two stacked rows for one node: stats + green download spark, then the red upload spark
    aligned beneath. Colored rows fit by construction; plain rows are sliced to `cols`."""
    vals = (name, human(i), human(o), nxt, resid, human(peak), human_bytes(winx))
    if color:
        dn = f"{GRN}↓{spark(hin[name], sw, peak)}{RST}"
        up = f"{RED}↑{spark(hout[name], sw, peak)}{RST}"
        l1 = fmt_prefix(vals, base=base, color=True) + "  " + dn + RST
        l2 = base + " " * PREFIX_W + "  " + up + RST
        return [l1, l2]
    pre = fmt_prefix(vals, color=False)
    l1 = (pre + "  ↓" + spark(hin[name], sw, peak))[:cols]
    l2 = (" " * PREFIX_W + "  ↑" + spark(hout[name], sw, peak))[:cols]
    if base:
        l1, l2 = base + l1 + RST, base + l2 + RST
    return [l1, l2]


def log_csv(ts, samples):
    if ts - _last_log[0] < LOG_EVERY:
        return
    _last_log[0] = ts
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_CAP:
            with open(LOG_PATH) as f:
                all_lines = f.readlines()
            with open(LOG_PATH, "w") as f:
                f.writelines(all_lines[len(all_lines) // 2:])
        with open(LOG_PATH, "a") as f:
            for name, i, o in samples:
                f.write(f"{ts:.0f},{name},{i:.0f},{o:.0f}\n")
    except Exception:
        pass


def fetch():
    with urlopen(URL, timeout=4) as r:
        return json.load(r)


_last_size = [0, 0]


def render():
    try:                                              # query the tty directly each frame so resizes
        sz = os.get_terminal_size(sys.stdout.fileno())  # are tracked (shutil caches stale env)
        cols, lines = sz.columns, sz.lines
    except Exception:
        cols, lines = 110, 30
    cols = max(56, cols)
    lines = max(6, lines)
    if (cols, lines) != (_last_size[0], _last_size[1]):
        _last_size[0], _last_size[1] = cols, lines
        sys.stdout.write("\033[2J")                   # full clear on (re)size -> no stale artifacts
    sw = max(8, cols - (PREFIX_W + 4))                # bars width; +4 = "  " + arrow + 1 margin
    color = cols >= PREFIX_W + 12
    win_s = int(sw * POLL)
    out = []
    now = time.strftime("%H:%M:%S")
    try:
        st = fetch()
        err = None
    except Exception as e:
        st, err = None, str(e)

    cver = ""
    if st:                                             # the tablet's own client version, from /status
        for n in st.get("nodes", []):
            if "tablet" in (n.get("hostname") or "").lower():
                cver = n.get("client_version") or ""
                break
    cver = cver or _local_cver() or "?"                # fallback: read the guest client.py VERSION
    title = (f" FLEET BANDWIDTH  round trip: home → node → home   client {cver}"
             f"   (MAX·XFER=last {win_s}s)   poll {POLL:.0f}s   {now}")
    out.append(f"\033[1;36m{title[:cols].ljust(cols)}\033[0m")

    if err:
        out.append(f"\033[31m controller {CTRL_IP} unreachable: {err}\033[0m"[:cols + 9])
    else:
        down, up, mesh = build_edges(st)               # forward + reverse node edges (round trip)
        res, fleet_models = build_resident(st)
        cur = {}                                       # name -> (in, out) this poll
        ti = to = 0.0
        for n in st.get("nodes", []):
            name = (n.get("hostname") or "?")[:12]
            i = float(n.get("net_in_bps") or 0)
            o = float(n.get("net_out_bps") or 0)
            track(name, i, o)
            cur[name] = (i, o)
            ti += i
            to += o
        m = st.get("metrics", {})
        ci = float(m.get("ctrl_in_bps") or 0)
        co = float(m.get("ctrl_out_bps") or 0)
        track("controller", ci, co); cur["controller"] = (ci, co)
        track("__total__", ti, to);  cur["__total__"] = (ti, to)

        stat = {name: wstats(name, sw) for name in cur}   # (peak, win_bytes) over the shown window
        node_names = [nm for nm in cur if nm not in ("controller", "__total__")]
        node_names.sort(key=lambda nm: stat[nm][1], reverse=True)   # busiest-in-window first

        out.append(DIM + fmt_prefix(("NODE", "DOWN", "UP", "FROM→…→NEXT", "RESIDENT", "MAX", "XFER"))
                   + "  ↓ download / ↑ upload" + RST)
        budget = max(1, (lines - 9) // 2)              # 2 rows/node; reserve header/ctrl/total/legend/tablet/models
        shown = node_names[:budget]
        for name in shown:
            i, o = cur[name]
            peak, winx = stat[name]
            out.extend(node_lines(name, i, o, route_label(name, up, down, mesh),
                                  resid_label(res.get(name)), peak, winx, "", color, cols, sw))
        if len(node_names) > len(shown):
            out.append(f"{DIM}   …{len(node_names) - len(shown)} more node(s){RST}")

        cpeak, cwinx = stat["controller"]
        out.extend(node_lines("controller", ci, co, route_label("controller", up, down, mesh),
                              "", cpeak, cwinx, MAG, color, cols, sw))
        tpeak, twinx = stat["__total__"]
        fleet_gb = sum(v[2] for v in res.values())
        fleet_layers = sum(v[1] for v in res.values())
        tresid = resid_label((fleet_models, fleet_layers, fleet_gb))
        out.append(BOLD + fmt_prefix((f"TOTAL {len(node_names)}", human(ti), human(to), "",
                                      tresid, human(tpeak), human_bytes(twinx)),
                                     base=BOLD, color=color) + RST)
        out.append(f"  {GRN}██{RST} download  {RED}██{RST} upload   "
                   f"{DIM}↓green=download ↑red=upload (scaled to MAX) · "
                   f"FROM→node→NEXT: 'home'=controller (drop-off → home){RST}")
        # This tablet's own load (the panel process runs here, so /proc is the tablet's).
        cpu_pct, cpu_s = tablet_cpu()
        if cpu_pct is not None:
            _cpu_hist.append(cpu_pct)
        memv, ld = tablet_mem(), tablet_load()
        if memv and memv[1]:
            used, tot = memv
            mem_s = f"{used / 1048576:.1f}/{tot / 1048576:.1f}GB {100 * used / tot:.0f}%"
        else:
            mem_s = "—"
        ld_s = f"  load {ld:.2f}" if ld is not None else ""
        # cpu sparkline over the SAME window as the network graphs, right beside the current %
        # (green <60%, yellow <85%, red above). Width capped so the line never wraps.
        head, tail = f"  TABLET  cpu {cpu_s}  ", f"   mem {mem_s}{ld_s}"
        gw = min(sw, max(0, cols - len(head) - len(tail)))
        cgraph = pct_spark(_cpu_hist, gw, 100.0) if gw >= 6 else ""
        ccol = GRN if (cpu_pct or 0) < 60 else ("\033[33m" if (cpu_pct or 0) < 85 else RED)
        out.append(f"  {BOLD}TABLET{RST}  cpu {cpu_s}  {ccol}{cgraph}{RST}{tail}")
        # Loaded models on the fleet — names only, comma-separated (placement is in the chart).
        mnames = []
        for mdl in (st.get("cluster", {}).get("loaded_models") or []):
            nmn = (mdl.get("friendly") or mdl.get("display_name") or mdl.get("target") or "").strip()
            if nmn and nmn not in mnames:
                mnames.append(nmn)
        mtxt = ", ".join(mnames) if mnames else "(none loaded)"
        avail = max(12, cols - 10)
        if len(mtxt) > avail:
            mtxt = mtxt[:avail - 1] + "…"
        out.append(f"  {BOLD}MODELS{RST} {mtxt}")
        log_csv(time.time(), [(nm, *cur[nm]) for nm in node_names] + [("controller", ci, co)])

    sys.stdout.write("\033[H")
    sys.stdout.write("\033[K\n".join(out[:lines]))
    sys.stdout.write("\033[K\033[J")
    sys.stdout.flush()


def main():
    sys.stdout.write("\033[?25l")
    try:
        while True:
            render()
            time.sleep(POLL)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")


if __name__ == "__main__":
    main()
