#!/usr/bin/env python3
"""
InfiniteModel — server-rendered monitoring graphs (SVG strings, server-only leaf module).

Extracted from server.py (#38, step C) to shrink that file. These build the dashboard's
hand-rolled SVG sparklines (~110x30) and click-to-expand detail graphs (~720x260) for each
node's network traffic, free RAM, and GPU VRAM. The dashboard just drops the returned string
into the DOM — it never creates or stores the series itself. Tooltips are native SVG <title>
children (the browser shows them on hover, no JS needed), which is why the markup must be
inlined into the page (an <img> can't show them). Click-to-expand is a wrapping
<a href="/graph/{kind}/{host}"> to the server detail page. Colors are hardcoded hex that read
on both light and dark dashboards.

The graph DATA — the bounded per-node history rings NET_HISTORY / RAM_HISTORY / VRAM_HISTORY —
STAYS in server.py: they are appended to by the metrics sampler and owned by the persistence
section (load/save_*_history). They are supplied here by DEPENDENCY INJECTION: server.py calls
``set_history_sources(net=NET_HISTORY, ram=RAM_HISTORY, vram=VRAM_HISTORY)`` once at import time,
passing the SAME dict objects the sampler appends to, so live data flows through here without a
back-import of server (no import cycle). The renderers read ``_HIST['net']`` etc. — and because
those are the very dicts the sampler mutates, every new sample shows up here automatically.

This is a controller-only leaf module: it must NEVER ``import server``. It is listed in server.py's
EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync across the fleet, and server.py
imports its symbols back via a convergence-bridge import.
"""
from __future__ import annotations

import time


# ---------------------------------------------------------------------------
# Dependency injection: history rings (no back-import of server)
# ---------------------------------------------------------------------------
# server.py injects the SAME dict objects its metrics sampler appends to, so the
# renderers below read live data without importing server (no import cycle). Until
# the setter runs, an unset source reads as an empty dict -> a flat "collecting
# samples…" baseline (never a crash).
_HIST: dict = {}        # {'net': NET_HISTORY, 'ram': RAM_HISTORY, 'vram': VRAM_HISTORY}


def set_history_sources(**kw):
    """server.py injects its NET_HISTORY / RAM_HISTORY / VRAM_HISTORY rings here at import
    time, so the renderers can read live history WITHOUT importing server (no import cycle).
    Pass the SAME dict objects the sampler appends to (not copies)."""
    _HIST.update(kw)


# Graph colors — hardcoded hex that read on both light and dark dashboards. Used only by
# the renderers below, so they live here with them.
_GRAPH_DL = "#378add"     # download (controller -> node, = net_in)
_GRAPH_UL = "#e0833b"     # upload (node -> controller, = net_out)
_GRAPH_RAM = "#1d9e75"    # free RAM
_GRAPH_VRAM = "#7f77dd"   # GPU VRAM used (purple — distinct from RAM green & bw)
_GRAPH_TPS = "#17b0c4"    # per-model decode throughput tok/s (cyan — distinct from the above)
_GRAPH_GRID = "#888"      # axis / gridlines (low-contrast on either theme)


def _svg_esc(s) -> str:
    """Minimal XML escaping for text that goes inside SVG (<title>/labels)."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt_hms(t_ms: int) -> str:
    """Local HH:MM:SS for a ms timestamp (graph tooltips)."""
    try:
        return time.strftime("%H:%M:%S", time.localtime(t_ms / 1000.0))
    except Exception:
        return "?"


def _fmt_bps(bps) -> str:
    """Human bytes/s — matches the dashboard's KB/s, MB/s style."""
    b = float(bps or 0)
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if b < 1024 or unit == "GB/s":
            return (f"{b:.0f} {unit}" if (b >= 100 or unit == "B/s")
                    else f"{b:.1f} {unit}")
        b /= 1024.0
    return f"{b:.1f} GB/s"


def _downsample(seq, k: int):
    """Evenly sample `seq` down to <=k points, ALWAYS keeping the first and last (so the
    sparkline still ends at the current value). A 110px sparkline can't show more than ~110
    points, and emitting per-point markup for the full history ballooned /status?graphs=1 to
    ~7.6 MB across the fleet (#dash-bw) — downsampling + dropping per-point circles fixes it."""
    seq = list(seq)
    n = len(seq)
    if n <= k or k < 2:
        return seq
    step = (n - 1) / (k - 1)
    return [seq[min(n - 1, int(round(i * step)))] for i in range(k)]


def _spark_svg(host: str, kind: str) -> str:
    """Small inline sparkline SVG (~110x30) for one node, drawn from NET_HISTORY
    (kind 'bw': download + upload polylines), RAM_HISTORY (kind 'ram': free-GB line)
    or VRAM_HISTORY (kind 'vram': used-GB line). Native <title> tooltips per point
    ("HH:MM:SS · value") plus one summary <title> on the whole svg. Wrapped in an <a>
    to the detail page. Always returns valid standalone SVG — an empty/short history
    renders a flat baseline."""
    W, H, pad = (240, 44, 3) if kind == "tps" else (110, 30, 2)
    x0, x1, y0, y1 = pad, W - pad, pad, H - pad
    pts = (list(_HIST.get("ram", {}).get(host, ())) if kind == "ram"
           else list(_HIST.get("vram", {}).get(host, ())) if kind == "vram"
           else list(_HIST.get("tps", {}).get(host, ())) if kind == "tps"
           else list(_HIST.get("net", {}).get(host, ())))
    # tps: fewer, well-spaced points so per-point <title> hover targets are usable (loaded
    # models are few, so keeping per-point circles here is cheap — unlike the fleet-wide bw
    # sparklines that dropped them for payload size, #dash-bw). Others cap to the pixel width.
    pts = _downsample(pts, 90 if kind == "tps" else W)
    inner = []
    summary = host
    n = len(pts)

    def _xy(i, v, vmax):
        x = x0 if n <= 1 else x0 + (x1 - x0) * (i / (n - 1))
        y = y1 - (y1 - y0) * (v / vmax if vmax > 0 else 0)
        return x, y

    if n == 0:
        # no data yet — flat baseline so the cell never looks broken / never crashes
        inner.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
                     f'stroke="{_GRAPH_GRID}" stroke-width="1" opacity="0.4"/>')
        summary = f"{host} · collecting samples…"
    elif kind == "ram":
        free = [p[1] / 10.0 for p in pts]          # tenths -> GB
        total = pts[-1][2] / 10.0
        vmax = max(total, max(free), 0.1)
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                        (_xy(i, free[i], vmax) for i in range(n)))
        inner.append(f'<polyline points="{poly}" fill="none" '
                     f'stroke="{_GRAPH_RAM}" stroke-width="1.4"/>')
        summary = f"{host} · free {free[-1]:.1f} / {total:.1f} GB"
    elif kind == "vram":
        used = [p[1] / 10.0 for p in pts]          # tenths -> GB
        total = pts[-1][2] / 10.0
        vmax = max(total, max(used), 0.1)
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                        (_xy(i, used[i], vmax) for i in range(n)))
        inner.append(f'<polyline points="{poly}" fill="none" '
                     f'stroke="{_GRAPH_VRAM}" stroke-width="1.4"/>')
        summary = f"{host} · used {used[-1]:.1f} / {total:.1f} GB"
    elif kind == "tps":
        val = [p[1] / 10.0 for p in pts]          # tenths -> tok/s
        vmax = max(max(val), 1.0)
        xy = [_xy(i, val[i], vmax) for i in range(n)]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        inner.append(f'<polyline points="{poly}" fill="none" '
                     f'stroke="{_GRAPH_TPS}" stroke-width="1.5"/>')
        for i in range(n):                        # transparent per-point hover targets (<title>)
            tip = f"{_fmt_hms(pts[i][0])} · {val[i]:.1f} tok/s"
            inner.append(f'<circle cx="{xy[i][0]:.1f}" cy="{xy[i][1]:.1f}" r="3.5" '
                         f'fill="transparent"><title>{_svg_esc(tip)}</title></circle>')
        summary = f"{host} · {val[-1]:.1f} tok/s"
    else:  # bw
        dl = [p[1] for p in pts]
        ul = [p[2] for p in pts]
        vmax = max(max(dl), max(ul), 1)
        for series, color in ((dl, _GRAPH_DL), (ul, _GRAPH_UL)):
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                            (_xy(i, series[i], vmax) for i in range(n)))
            inner.append(f'<polyline points="{poly}" fill="none" '
                         f'stroke="{color}" stroke-width="1.2"/>')
        summary = f"{host} · ↓ {_fmt_bps(dl[-1])} · ↑ {_fmt_bps(ul[-1])}"

    body = "".join(inner)
    href = f"/graph/{kind}/{_svg_esc(host)}"
    return (f'<a xlink:href="{href}" href="{href}">'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'style="cursor:pointer;vertical-align:middle">'
            f'<title>{_svg_esc(summary)} (click to expand)</title>'
            f'{body}</svg></a>')


def _detail_svg(host: str, kind: str) -> str:
    """Larger detail graph (~720x260): time on x, value on y, gridlines, a few axis
    tick labels, the series polyline(s), and per-point <title> tooltips. Returned as
    a complete standalone SVG document (served at /graph/{kind}/{host})."""
    W, H = 720, 260
    ml, mr, mt, mb = 64, 16, 28, 30        # margins (room for axis labels + title)
    x0, x1, y0, y1 = ml, W - mr, mt, H - mb
    pts = (list(_HIST.get("ram", {}).get(host, ())) if kind == "ram"
           else list(_HIST.get("vram", {}).get(host, ())) if kind == "vram"
           else list(_HIST.get("tps", {}).get(host, ())) if kind == "tps"
           else list(_HIST.get("net", {}).get(host, ())))
    n = len(pts)
    title = (f"{_svg_esc(host)} · free RAM (GB)" if kind == "ram"
             else f"{_svg_esc(host)} · GPU VRAM used (GB)" if kind == "vram"
             else f"{_svg_esc(host)} · decode throughput (tok/s)" if kind == "tps"
             else f"{_svg_esc(host)} · traffic (↓ download / ↑ upload)")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="monospace" font-size="11">',
             f'<title>{title}</title>',
             f'<text x="{ml}" y="16" font-size="13" fill="{_GRAPH_GRID}">{title}</text>']

    if n == 0:
        parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{(y0 + y1) / 2:.0f}" '
                     f'text-anchor="middle" fill="{_GRAPH_GRID}">collecting samples…</text>')
        parts.append('</svg>')
        return "".join(parts)

    if kind == "ram":
        free = [p[1] / 10.0 for p in pts]
        total = pts[-1][2] / 10.0
        vmax = max(total, max(free), 0.1)
        unit_fmt = lambda v: f"{v:.0f}"           # GB labels
    elif kind == "vram":
        used = [p[1] / 10.0 for p in pts]
        total = pts[-1][2] / 10.0
        vmax = max(total, max(used), 0.1)
        unit_fmt = lambda v: f"{v:.0f}"           # GB labels
    elif kind == "tps":
        val = [p[1] / 10.0 for p in pts]          # tenths -> tok/s
        vmax = max(max(val), 1.0)
        unit_fmt = lambda v: f"{v:.0f}"           # tok/s labels
    else:
        dl = [p[1] for p in pts]
        ul = [p[2] for p in pts]
        vmax = max(max(dl), max(ul), 1)
        unit_fmt = _fmt_bps

    def _px(i):
        return x0 if n <= 1 else x0 + (x1 - x0) * (i / (n - 1))

    def _py(v):
        return y1 - (y1 - y0) * (v / vmax if vmax > 0 else 0)

    # value gridlines + y tick labels (5 rows)
    for g in range(6):
        v = vmax * g / 5.0
        yy = _py(v)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" '
                     f'stroke="{_GRAPH_GRID}" stroke-width="0.5" opacity="0.3"/>')
        parts.append(f'<text x="{x0 - 6}" y="{yy + 3:.1f}" text-anchor="end" '
                     f'fill="{_GRAPH_GRID}">{_svg_esc(unit_fmt(v))}</text>')
    # time tick labels along x (a few)
    for g in range(5):
        i = 0 if n <= 1 else round((n - 1) * g / 4)
        xx = _px(i)
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" '
                     f'stroke="{_GRAPH_GRID}" stroke-width="0.5" opacity="0.15"/>')
        anchor = "start" if g == 0 else ("end" if g == 4 else "middle")
        parts.append(f'<text x="{xx:.1f}" y="{y1 + 14:.0f}" text-anchor="{anchor}" '
                     f'fill="{_GRAPH_GRID}">{_fmt_hms(pts[i][0])}</text>')

    def _poly(series, color, width):
        pl = " ".join(f"{_px(i):.1f},{_py(series[i]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{pl}" fill="none" '
                     f'stroke="{color}" stroke-width="{width}"/>')

    if kind == "ram":
        _poly(free, _GRAPH_RAM, 1.8)
        for i in range(n):
            tip = f"{_fmt_hms(pts[i][0])} · free {free[i]:.1f} / {pts[i][2]/10.0:.1f} GB"
            parts.append(f'<circle cx="{_px(i):.1f}" cy="{_py(free[i]):.1f}" r="3" '
                         f'fill="transparent"><title>{_svg_esc(tip)}</title></circle>')
    elif kind == "vram":
        _poly(used, _GRAPH_VRAM, 1.8)
        for i in range(n):
            tip = f"{_fmt_hms(pts[i][0])} · used {used[i]:.1f} / {pts[i][2]/10.0:.1f} GB"
            parts.append(f'<circle cx="{_px(i):.1f}" cy="{_py(used[i]):.1f}" r="3" '
                         f'fill="transparent"><title>{_svg_esc(tip)}</title></circle>')
    elif kind == "tps":
        _poly(val, _GRAPH_TPS, 1.8)
        for i in range(n):
            tip = f"{_fmt_hms(pts[i][0])} · {val[i]:.1f} tok/s"
            parts.append(f'<circle cx="{_px(i):.1f}" cy="{_py(val[i]):.1f}" r="3" '
                         f'fill="transparent"><title>{_svg_esc(tip)}</title></circle>')
    else:
        _poly(dl, _GRAPH_DL, 1.6)
        _poly(ul, _GRAPH_UL, 1.6)
        for i in range(n):
            tip = f"{_fmt_hms(pts[i][0])} · ↓ {_fmt_bps(dl[i])} · ↑ {_fmt_bps(ul[i])}"
            parts.append(f'<circle cx="{_px(i):.1f}" cy="{_py(max(dl[i], ul[i])):.1f}" '
                         f'r="3" fill="transparent"><title>{_svg_esc(tip)}</title></circle>')
        # legend
        parts.append(f'<rect x="{x1 - 150}" y="{mt - 14}" width="10" height="10" fill="{_GRAPH_DL}"/>'
                     f'<text x="{x1 - 136}" y="{mt - 5}" fill="{_GRAPH_GRID}">download</text>'
                     f'<rect x="{x1 - 70}" y="{mt - 14}" width="10" height="10" fill="{_GRAPH_UL}"/>'
                     f'<text x="{x1 - 56}" y="{mt - 5}" fill="{_GRAPH_GRID}">upload</text>')

    parts.append('</svg>')
    return "".join(parts)
