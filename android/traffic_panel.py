#!/usr/bin/env python3
"""
InfiniteModel tablet bandwidth panel  (standalone, READ-ONLY).

Polls the controller's /status and draws a static, non-scrolling table of every node's
traffic to the rest of the fleet (node <-> all) plus the controller and a fleet TOTAL.
Each node spans TWO rows: a stats line topped by a GREEN download sparkline, and a RED
upload sparkline beneath it.

  DOWN / UP  - current in / out rate (DOWN green, UP red)
  NEXT       - where this node sends its activations, INFERRED from the loaded models'
               pipeline placement (controller -> stage0 -> ... -> head -> controller).
               'ctrl' = back to the controller (a head stage or a single-node model);
               'mesh' = a tensor-parallel stage (within-stage all-reduce, no single hop);
               a node serving >1 model shows the first destination + a count
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

CTRL_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.15.103"
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
    down = defaultdict(list)
    mesh = set()

    def add(a, b):
        if a and b and b not in down[a]:
            down[a].append(b)

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
        add(prev, CTRL)                       # ... and the head returns logits to the controller
    return down, mesh


def next_label(host, down, mesh):
    """Compact '->destination' tag for a node's UP column. 'ctrl' = back to the controller;
    'mesh' = a tensor-parallel stage; >1 destination shows the first + a count."""
    dl = down.get(host) or []
    if not dl:
        return "mesh" if host in mesh else ""
    names = ["ctrl" if d == CTRL else d for d in dl]
    if len(names) == 1:
        return "→" + names[0][:7]
    return "→" + names[0][:5] + "+%d" % (len(names) - 1)


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


# Columns: NODE DOWN UP NEXT RESIDENT MAX XFER. DOWN is idx 1 (green), UP idx 2 (red).
COLW = ((11, "<"), (9, ">"), (9, ">"), (8, "<"), (13, "<"), (9, ">"), (9, ">"))
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

    title = (f" FLEET BANDWIDTH  node → next hop   (MAX·XFER = last {win_s}s shown)"
             f"   poll {POLL:.0f}s   {now}")
    out.append(f"\033[1;36m{title[:cols].ljust(cols)}\033[0m")

    if err:
        out.append(f"\033[31m controller {CTRL_IP} unreachable: {err}\033[0m"[:cols + 9])
    else:
        down, mesh = build_edges(st)                   # inferred node -> downstream-node edges
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

        out.append(DIM + fmt_prefix(("NODE", "DOWN", "UP", "NEXT", "RESIDENT", "MAX", "XFER"))
                   + "  ↓ download / ↑ upload" + RST)
        budget = max(1, (lines - 7) // 2)              # 2 rows per node; reserve header/ctrl/total/legend
        shown = node_names[:budget]
        for name in shown:
            i, o = cur[name]
            peak, winx = stat[name]
            out.extend(node_lines(name, i, o, next_label(name, down, mesh),
                                  resid_label(res.get(name)), peak, winx, "", color, cols, sw))
        if len(node_names) > len(shown):
            out.append(f"{DIM}   …{len(node_names) - len(shown)} more node(s){RST}")

        cpeak, cwinx = stat["controller"]
        out.extend(node_lines("controller", ci, co, next_label("controller", down, mesh),
                              "", cpeak, cwinx, MAG, color, cols, sw))
        tpeak, twinx = stat["__total__"]
        fleet_gb = sum(v[2] for v in res.values())
        fleet_layers = sum(v[1] for v in res.values())
        tresid = resid_label((fleet_models, fleet_layers, fleet_gb))
        out.append(BOLD + fmt_prefix((f"TOTAL {len(node_names)}", human(ti), human(to), "",
                                      tresid, human(tpeak), human_bytes(twinx)),
                                     base=BOLD, color=color) + RST)
        out.append(f"  {GRN}██{RST} download    {RED}██{RST} upload    "
                   f"{DIM}stacked per node: ↓ green = download · ↑ red = upload (scaled to MAX){RST}")
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
