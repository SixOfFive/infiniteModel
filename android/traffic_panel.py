#!/usr/bin/env python3
"""
InfiniteModel tablet bandwidth panel  (standalone, READ-ONLY).

Polls the controller's /status and draws a static, non-scrolling table of every node's
traffic to the rest of the fleet (node <-> all) plus the controller and a fleet TOTAL.

  DOWN / UP  - current in / out rate.  DOWN is shown in GREEN, UP in RED
               (a legend states this below the table)
  NEXT       - where this node sends its activations, INFERRED from the loaded models'
               pipeline placement (controller -> stage0 -> ... -> head -> controller).
               'ctrl' = back to the controller (a head stage or a single-node model);
               'mesh' = a tensor-parallel stage (within-stage all-reduce, no single hop);
               a node serving >1 model shows the first destination + a count
  MAX        - peak combined (in+out) rate within the DISPLAYED sparkline window (NOT lifetime),
               and the sparkline is scaled to it (full bar == at this window's max)
  XFER       - bytes transferred since the panel started (resets on restart -- nothing persists);
               rows sort busiest-first by this

NEXT is an INFERENCE from placement, not a measured per-edge rate (the controller doesn't expose
per-peer byte counts over /status). The DOWN/UP numbers are each node's TOTAL across every
connection, not a per-edge measurement; the sparkline bars are the COMBINED in+out trend.

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

HIST       = 300                                   # combined-rate samples kept per row (>= sparkline)
LOG_PATH   = os.path.expanduser("~/.im/traffic.csv")
LOG_CAP    = 4 * 1024 * 1024
LOG_EVERY  = 10.0
SPARK      = " ▁▂▃▄▅▆▇█"
IDLE_FLOOR = 2048                                   # below this peak (B/s) a row's spark stays flat
CTRL       = "controller"                           # controller's row name + bookend of every chain

GRN, RED, DIM, MAG, BOLD, RST = (
    "\033[32m", "\033[31m", "\033[2m", "\033[35m", "\033[1m", "\033[0m")

hist  = defaultdict(lambda: deque(maxlen=HIST))    # combined (in+out) rate samples
xfer  = defaultdict(float)                          # cumulative bytes moved this session
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
    c = i + o
    hist[name].append(c)
    xfer[name] += c * POLL


def build_edges(st):
    """Derive where each node sends activations next, from every loaded PIPELINE model's ordered
    stages. The flow is controller -> stage0 -> ... -> head -> controller, so consecutive stages
    are the edges. Returns (down, mesh):
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
    'mesh' = a tensor-parallel stage; >1 destination (a node serving several models) shows the
    first + a count."""
    dl = down.get(host) or []
    if not dl:
        return "mesh" if host in mesh else ""
    names = ["ctrl" if d == CTRL else d for d in dl]
    if len(names) == 1:
        return "→" + names[0][:7]
    return "→" + names[0][:5] + "+%d" % (len(names) - 1)


# Column widths, in order: NODE DOWN UP NEXT MAX XFER. DOWN is index 1 (green), UP is index 2 (red).
COLW = ((11, "<"), (9, ">"), (9, ">"), (9, "<"), (9, ">"), (9, ">"))
FIXED = 1 + sum(w for w, _ in COLW) + (len(COLW) - 1) + 2   # leading sp + cells + gaps + "  " = 64


def fmt_row(vals, spk, cols, base="", color=False):
    """Assemble one fixed-width row. `vals` = the 6 column strings; `spk` = trailing sparkline.
    When `color`, DOWN (idx 1) is green and UP (idx 2) red, each returning to `base` after.
    Plain rows are sliced to `cols`; colored rows are sized to fit by construction (caller only
    enables color when the terminal is wide enough), so they're never sliced mid-escape."""
    cells = [f"{v:{a}{w}}" for v, (w, a) in zip(vals, COLW)]
    if color:
        cells[1] = f"{GRN}{cells[1]}{RST}{base}"
        cells[2] = f"{RED}{cells[2]}{RST}{base}"
        return base + " " + " ".join(cells) + "  " + spk + RST
    plain = " " + " ".join(cells) + "  " + spk
    plain = plain[:cols]
    return (base + plain + RST) if base else plain


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
    sw = max(6, cols - (FIXED + 1))
    color = cols >= FIXED + 8                          # only colorize when the row fits without wrap
    out = []
    now = time.strftime("%H:%M:%S")
    try:
        st = fetch()
        err = None
    except Exception as e:
        st, err = None, str(e)

    title = f" FLEET BANDWIDTH  node → next hop   (MAX = peak in window)   poll {POLL:.0f}s   {now}"
    out.append(f"\033[1;36m{title[:cols].ljust(cols)}\033[0m")

    if err:
        out.append(f"\033[31m controller {CTRL_IP} unreachable: {err}\033[0m"[:cols + 9])
    else:
        down, mesh = build_edges(st)                   # inferred node -> downstream-node edges
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

        wmax = {name: (max(list(hist[name])[-sw:]) if hist[name] else 0.0) for name in cur}

        node_names = [nm for nm in cur if nm not in ("controller", "__total__")]
        node_names.sort(key=lambda nm: xfer[nm], reverse=True)

        out.append(fmt_row(("NODE", "DOWN", "UP", "NEXT", "MAX", "XFER"),
                           "speed vs max"[:sw], cols, base=DIM))
        budget = max(1, lines - 6)                     # title+header+controller+total+legend(+more)
        shown = node_names[:budget]
        for name in shown:
            i, o = cur[name]
            out.append(fmt_row((name, human(i), human(o), next_label(name, down, mesh),
                                human(wmax[name]), human_bytes(xfer[name])),
                               spark(hist[name], sw, wmax[name]), cols, color=color))
        if len(node_names) > len(shown):
            out.append(f"{DIM}   …{len(node_names) - len(shown)} more node(s){RST}")

        out.append(fmt_row(("controller", human(ci), human(co), next_label("controller", down, mesh),
                            human(wmax["controller"]), human_bytes(xfer["controller"])),
                           spark(hist["controller"], sw, wmax["controller"]), cols,
                           base=MAG, color=color))
        out.append(fmt_row((f"TOTAL {len(node_names)}", human(ti), human(to), "",
                            human(wmax["__total__"]), human_bytes(xfer["__total__"])),
                           "", cols, base=BOLD, color=color))
        out.append(f"  {GRN}██{RST} download    {RED}██{RST} upload    "
                   f"{DIM}bars · combined in+out trend{RST}")
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
