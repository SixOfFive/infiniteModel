#!/usr/bin/env python3
"""
InfiniteModel tablet bandwidth panel  (standalone, READ-ONLY).

Polls the controller's /status and draws a static, non-scrolling table of every node's
traffic to the rest of the fleet (node <-> all) plus the controller and a fleet TOTAL.
A GRAPHED node spans TWO rows: a stats line topped by a GREEN download sparkline, and a RED
upload sparkline beneath it. A node is graphed only if it is worth the vertical space -- it is
MOVING data (>= IDLE_FLOOR), HOLDS a loaded model, or is OFF-BUILD (version skew, which must
stay visible). Every other node collapses into a single dim 'IDLE n: ...' list of names, and
the rows that buys are handed to the worker-log tail at the bottom. The controller is always
graphed.

  VER        - that node's client build (client_version). The controller row shows the SERVER
               build instead — a different version stream, so it is never flagged. A node whose
               build differs from the fleet majority is highlighted YELLOW (version skew is what
               stops a stale worker from registering)
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

Whatever vertical space is LEFT under the footers becomes a live WORKER LOG tail -- the recent
output of this tablet's own worker, read from its detached tmux pane ('wrk') via capture-pane.
That is a local read-only peek at a pane we already own; it sends nothing to the fleet. If the
node rows fill the screen there is no room left and the log is simply omitted.

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
YEL = "\033[33m"                                    # version skew (node build != fleet majority)

WRK_SESSION = "wrk"                                 # tmux session the worker runs in (see ~/.bashrc)
WLOG_TTL    = 3.0                                   # s between capture-pane calls (poll is 2s)
_wlog       = [0.0, []]                             # (last capture ts, cached lines)

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


def worker_log(maxlines):
    """Last `maxlines` non-blank lines of THIS tablet's worker output.

    The worker runs detached as `tmux new-session -d -s wrk 'proot-distro login debian -- ...
    start-client.sh'`, so its stdout/stderr live in that pane rather than a file — `capture-pane
    -p` prints it plain (escape sequences already stripped, since we don't pass -e). Read-only,
    and local: it peeks at a pane this user already owns and sends nothing to the fleet.
    Result is cached for WLOG_TTL so a 2s render loop doesn't fork tmux every single frame."""
    now = time.time()
    if _wlog[1] and now - _wlog[0] < WLOG_TTL:
        return _wlog[1][-maxlines:]
    try:
        import subprocess
        r = subprocess.run(["tmux", "capture-pane", "-p", "-t", WRK_SESSION, "-S", "-200"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        if r.returncode == 0:
            txt = r.stdout.decode("utf-8", "replace")
            out = []
            for ln in txt.splitlines():
                ln = "".join(c for c in ln if c >= " " or c == "\t").rstrip()
                if ln:
                    out.append(ln)
            lines = out or ["(worker started, no output yet)"]
        else:                                          # no such session => worker isn't running
            lines = [f"(no worker session '{WRK_SESSION}' — start it with: startworker)"]
    except FileNotFoundError:
        lines = ["(tmux not installed)"]
    except Exception as e:
        lines = [f"(worker log unavailable: {e})"]
    _wlog[0], _wlog[1] = now, lines
    return lines[-maxlines:]


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


# Columns: NODE VER DOWN UP ROUTE RESIDENT MAX XFER. Indices are named because inserting VER
# shifted DOWN/UP -- the sparkline width shrinks by VER's width, which is the intended trade.
COLW = ((10, "<"), (10, "<"), (9, ">"), (9, ">"), (20, "<"), (12, "<"), (9, ">"), (9, ">"))
COL_VER, COL_DOWN, COL_UP = 1, 2, 3
PREFIX_W = 1 + sum(w for w, _ in COLW) + (len(COLW) - 1)    # leading sp + cells + single-space gaps


def fmt_prefix(vals, base="", color=False, ver_warn=False):
    """The fixed-width column block (no sparkline). DOWN/UP get colored when `color`, each
    returning to `base` after so a base-colored row (controller/total) stays intact. VER renders
    dim, or YELLOW when this node's build is off the fleet majority (`ver_warn`)."""
    cells = [f"{v:{a}{w}}" for v, (w, a) in zip(vals, COLW)]
    if color:
        cells[COL_DOWN] = f"{GRN}{cells[COL_DOWN]}{RST}{base}"
        cells[COL_UP] = f"{RED}{cells[COL_UP]}{RST}{base}"
        vc = YEL if ver_warn else DIM
        cells[COL_VER] = f"{vc}{cells[COL_VER]}{RST}{base}"
    return (base + " " + " ".join(cells)) if base else (" " + " ".join(cells))


def node_lines(name, ver, i, o, nxt, resid, peak, winx, base, color, cols, sw, ver_warn=False):
    """Two stacked rows for one node: stats + green download spark, then the red upload spark
    aligned beneath. Colored rows fit by construction; plain rows are sliced to `cols`."""
    vals = (name, ver, human(i), human(o), nxt, resid, human(peak), human_bytes(winx))
    if color:
        dn = f"{GRN}↓{spark(hin[name], sw, peak)}{RST}"
        up = f"{RED}↑{spark(hout[name], sw, peak)}{RST}"
        l1 = fmt_prefix(vals, base=base, color=True, ver_warn=ver_warn) + "  " + dn + RST
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
        vers = {}                                      # name -> client build, for the VER column
        ti = to = 0.0
        for n in st.get("nodes", []):
            name = (n.get("hostname") or "?")[:12]
            i = float(n.get("net_in_bps") or 0)
            o = float(n.get("net_out_bps") or 0)
            track(name, i, o)
            cur[name] = (i, o)
            vers[name] = (n.get("client_version") or "?")[:10]
            ti += i
            to += o
        # Fleet-majority client build; anything else is skew and gets flagged yellow. Compared
        # among NODES only — the controller runs a separate server build (never flagged).
        tally = defaultdict(int)
        for v in vers.values():
            if v and v != "?":
                tally[v] += 1
        fleet_ver = max(tally, key=tally.get) if tally else ""
        srv_ver = ((st.get("controller") or {}).get("version") or "?")[:10]
        m = st.get("metrics", {})
        ci = float(m.get("ctrl_in_bps") or 0)
        co = float(m.get("ctrl_out_bps") or 0)
        track("controller", ci, co); cur["controller"] = (ci, co)
        track("__total__", ti, to);  cur["__total__"] = (ti, to)

        stat = {name: wstats(name, sw) for name in cur}   # (peak, win_bytes) over the shown window
        node_names = [nm for nm in cur if nm not in ("controller", "__total__")]
        node_names.sort(key=lambda nm: stat[nm][1], reverse=True)   # busiest-in-window first

        hdr = (fmt_prefix(("NODE", "VER", "DOWN", "UP", "FROM→…→NEXT", "RESIDENT", "MAX", "XFER"))
               + "  ↓ download / ↑ upload")          # plain (color=False) -> safe to slice
        out.append(DIM + hdr[:cols] + RST)

        # A quiet node with nothing loaded earns no graph -- it collapses into a one-line IDLE
        # list, and the two rows it would have cost go to the worker log instead. It still gets a
        # full graph if it is MOVING data (>= IDLE_FLOOR, the same threshold that already flattens
        # a sparkline), HOLDS a model, or is OFF-BUILD -- version skew must never hide in the list.
        def graphed(nm):
            return (stat[nm][0] >= IDLE_FLOOR
                    or bool(res.get(nm) and res[nm][0])
                    or (bool(fleet_ver) and vers.get(nm) != fleet_ver))

        graph_names = [nm for nm in node_names if graphed(nm)]     # keeps busiest-first order
        gset = set(graph_names)
        idle_names = [nm for nm in node_names if nm not in gset]

        idle_lines = []                                # built first: they cost the node budget rows
        if idle_names:
            lead = f"  IDLE {len(idle_names)}: "
            indent = " " * len(lead)
            line = lead
            for nm in idle_names:
                piece = nm + "  "
                if len(line) + len(piece) > cols and line.strip():
                    idle_lines.append(line.rstrip())
                    line = indent + piece
                else:
                    line += piece
            idle_lines.append(line.rstrip())

        budget = max(1, (lines - 9 - len(idle_lines)) // 2)   # 2 rows/node; reserve fixed footers
        shown = graph_names[:budget]
        for name in shown:
            i, o = cur[name]
            peak, winx = stat[name]
            out.extend(node_lines(name, vers.get(name, "?"), i, o,
                                  route_label(name, up, down, mesh),
                                  resid_label(res.get(name)), peak, winx, "", color, cols, sw,
                                  ver_warn=bool(fleet_ver) and vers.get(name) != fleet_ver))
        if len(graph_names) > len(shown):
            out.append(f"{DIM}   …{len(graph_names) - len(shown)} more active node(s){RST}")
        for ln in idle_lines:
            out.append(f"{DIM}{ln[:cols]}{RST}")

        cpeak, cwinx = stat["controller"]
        out.extend(node_lines("controller", srv_ver, ci, co,   # SERVER build, not a client build
                              route_label("controller", up, down, mesh),
                              "", cpeak, cwinx, MAG, color, cols, sw))
        tpeak, twinx = stat["__total__"]
        fleet_gb = sum(v[2] for v in res.values())
        fleet_layers = sum(v[1] for v in res.values())
        tresid = resid_label((fleet_models, fleet_layers, fleet_gb))
        out.append(BOLD + fmt_prefix((f"TOTAL {len(node_names)}", fleet_ver, human(ti), human(to),
                                      "", tresid, human(tpeak), human_bytes(twinx)),
                                     base=BOLD, color=color) + RST)
        # Legend: colored swatches (fixed plain width) + a tail truncated by PLAIN length, so the
        # line can never wrap and break the static layout no matter how narrow the terminal is.
        # Skew notice goes FIRST in the tail: it is the actionable bit, so it must survive the
        # width truncation rather than be the first thing clipped off the end.
        nskew = sum(1 for nm in node_names if fleet_ver and vers.get(nm) != fleet_ver)
        lead_plain = "  ██ download  ██ upload   "
        lead = f"  {GRN}██{RST} download  {RED}██{RST} upload   "
        skew_p = f"{nskew} node(s) OFF-BUILD · " if nskew else ""
        rest_p = (f"↓green ↑red (scaled to MAX) · VER=client build, ctrl row=server {srv_ver} · "
                  f"IDLE=quiet & unloaded & on-build · FROM→node→NEXT: 'home'=controller")
        room_l = max(0, cols - len(lead_plain))
        skew_p = skew_p[:room_l]
        rest_p = rest_p[:max(0, room_l - len(skew_p))]
        out.append(lead + (f"{YEL}{skew_p}{RST}" if skew_p else "") + DIM + rest_p + RST)
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

    # Whatever vertical room is LEFT becomes a live tail of this tablet's worker log. Sits outside
    # the else, so when /status is unreachable (short `out`, lots of room) the log still renders --
    # which is exactly when you want to see what the worker is saying.
    room = lines - len(out) - 1                        # -1 for the section header itself
    if room >= 2:
        out.append(f"  {BOLD}WORKER LOG{RST} {DIM}· this tablet · tmux '{WRK_SESSION}'{RST}")
        for ln in worker_log(room):
            out.append(f"  {DIM}{ln[:max(8, cols - 3)]}{RST}")

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
