"""worker_update: the worker's self-update machinery + background watchdogs, relocated
VERBATIM from client.py (code-split Inc 8): SELF_UPDATE_* knobs, _extract_version,
_ver_ordinal (the no-downgrade natural-sort), _fetch_repo_file, _self_update_check,
_self_update_loop, plus _fwd_watchdog_loop and _console_panel_loop.

EXTRA_UPDATE_FILES deliberately STAYS in client.py -- the primary file every worker is
guaranteed to refresh -- so every future module-registration edit lands there; the moved
_self_update_check reads it (and VERSION) through the bound namespace, both never rebound.
Bodies BYTE-IDENTICAL; module globals injected by state.bind() -- see state.py. Worker-side
leaf; never imports client; listed in client.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


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


def _ver_ordinal(v: str):
    """#no-downgrade: natural-sort key for VERSION tags ('0.2-m4c177') — digit runs compare
    NUMERICALLY, alpha runs lexicographically, so m4c9 < m4c176 (string compare gets this wrong)
    and m4bz < m4c0. Type-tagged tuples keep int/str comparisons well-defined at any divergence."""
    import re
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.findall(r"\d+|\D+", v or "")]


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
    # #no-downgrade: after a git push the raw CDN lags per-file by 1-5 min, so a worker running
    # code AHEAD of the CDN (pscp deploy, or restarting mid-propagation) used to see "content
    # differs" -> overwrite its files with the STALE repo copy and restart into a DOWNGRADE (live
    # hit: m4c177 -> m4c176 minutes after the m4c177 push, silently un-deploying the triton-race
    # fix). Refuse to apply a remote whose primary VERSION is strictly OLDER than the running one
    # — skipping the writes too, so stale extras never land on disk. (The worker loop is always
    # automatic; deliberate rollbacks go through the controller's forced /update.)
    if remote_ver and _ver_ordinal(remote_ver) < _ver_ordinal(VERSION):
        print(f"[update] repo VERSION {remote_ver} is OLDER than running {VERSION} - "
              f"ignoring (CDN lag / local ahead); re-check next poll")
        return
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


def _fwd_watchdog_loop(worker) -> None:
    """#fwd-watchdog: backstop for an orphaned forward stuck INSIDE one un-yieldable op, where the
    cooperative between-layer cancel in shard_forward can't bail it. Scans this node's shards; for one
    whose forward holds _fwd_lock but whose per-layer progress heartbeat (shard._fwd_progress_ts) has
    gone STALE, first trips _fwd_cancel (in case it's paused between ops), then — if still stalled —
    exits(42) so the supervisor relaunches and the controller auto-recovers the model(s) (#77). It
    measures NO-PROGRESS time, not total runtime, so a legitimately long forward that keeps advancing
    layers never trips it. Daemon; never mutates inference state beyond setting the cancel Event."""
    import time
    CANCEL_S = 120.0   # no layer progress this long -> ask the forward to yield (cooperative)
    EXIT_S = 300.0     # STILL no progress -> supervisor relaunch (genuinely wedged inside one op)
    while True:
        time.sleep(15)
        try:
            now = time.time()
            for mid, shard in list(getattr(worker, "shards", {}).items()):
                lock = getattr(shard, "_fwd_lock", None)
                if lock is None or not lock.locked():
                    continue                       # no forward running on this shard
                prog = getattr(shard, "_fwd_progress_ts", None)
                if prog is None:
                    continue                       # forward hasn't reached the layer loop yet
                stalled = now - prog
                if stalled > EXIT_S:
                    print(f"[fwd-watchdog] {mid}: forward stalled {stalled:.0f}s with no layer progress "
                          f"— cooperative cancel ineffective; exiting(42) for supervisor relaunch",
                          flush=True)
                    os._exit(42)
                if stalled > CANCEL_S:
                    cancel = getattr(shard, "_fwd_cancel", None)
                    if cancel is not None and not cancel.is_set():
                        print(f"[fwd-watchdog] {mid}: forward no progress for {stalled:.0f}s "
                              f"— signalling cancel", flush=True)
                        cancel.set()
        except Exception:
            pass


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
