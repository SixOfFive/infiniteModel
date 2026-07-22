"""peers.py — #federation Phase 1: controller <-> controller discovery + gossip.

WHY
---
The fleet runs more than one controller (e.g. a beast pool and an om3nbox pool). Until now each
controller was blind to the others: it could not see that a peer already had a model resident, could
not show it, and could not borrow it — so the same weights got loaded twice and idle capacity on one
side was invisible to the other.

This module makes controllers aware of each other WITHOUT touching the worker's single-tenant core.
It rides the #discovery UDP channel that already exists (udp/discovery_port, default 50099):

  * ANNOUNCE — we broadcast the ordinary discovery query with `"peer": 1` plus our own identity.
    Every controller that hears it records US as a peer (server.py's _DiscoveryResponder calls
    record_peer), and every controller that REPLIES is recorded by us. One packet, both directions.
  * GOSSIP  — we then poll each known peer's `GET /peer_info` over HTTP and cache the answer
    (its nodes, its resident models, its version). That cache is what the dashboard renders and
    what later phases (request federation, peer model pull, node lending) consult.

DESIGN NOTES
------------
* Read-only and side-effect free with respect to models: nothing here loads, unloads or places
  anything. A peer being visible must never change how this controller serves.
* Failure-tolerant by construction: a peer that stops answering is marked `stale` and then dropped
  after PEER_STALE_S. A gossip error is recorded on the peer, never raised into a caller.
* stdlib only (urllib, not httpx/requests) so this adds no dependency to a controller install.
* Globals come from `state` (the code-split shared-state registry) and are read LAZILY inside
  functions — this module is imported long before state.publish() runs in main().

Controller-only leaf module; listed in EXTRA_UPDATE_FILES so the multi-file self-update keeps it
in sync across the fleet.
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import state
import wire

PEER_POLL_S = 30.0          # how often we re-poll each peer's /peer_info
PEER_ANNOUNCE_S = 60.0      # how often we re-broadcast our presence
PEER_STALE_S = 180.0        # no successful contact for this long -> mark stale
PEER_DROP_S = 900.0         # ... and this long -> forget entirely (a decommissioned controller)
PEER_HTTP_TIMEOUT = 6.0

# key "host:http_port" -> peer record. Mutated IN PLACE only (state.publish shares references).
PEERS: dict = {}


# --- identity ------------------------------------------------------------------------------------

def _cfg() -> dict:
    return wire.load_config()


def _args():
    return getattr(state, "ARGS", None)


def my_http_port() -> int:
    a = _args()
    try:
        return int(getattr(a, "http_port", 0) or _cfg()["http_port"])
    except (TypeError, ValueError):
        return int(_cfg()["http_port"])


def my_identity() -> dict:
    """What we advertise about ourselves over UDP and at /peer_info."""
    return {
        "name": socket.gethostname(),
        "http_port": my_http_port(),
        "version": str(getattr(state, "VERSION", "") or ""),
        "cluster_id": str(_cfg().get("cluster_id") or ""),
    }


def _my_ips() -> set:
    """Every address that means 'this controller', so we never peer with ourselves."""
    ips = set(wire._local_ipv4s())
    ips.update({"127.0.0.1", "localhost", "::1"})
    a = _args()
    h = str(getattr(a, "host", "") or "")
    if h and h not in ("0.0.0.0", "::"):
        ips.add(h)
    return ips


def is_self(host: str, http_port: int) -> bool:
    return str(host) in _my_ips() and int(http_port or 0) == my_http_port()


def peer_key(host: str, http_port) -> str:
    return f"{host}:{int(http_port or 0)}"


# --- registry ------------------------------------------------------------------------------------

def record_peer(host: str, http_port, name: str = "", version: str = "",
                cluster_id: str = "", source: str = "udp") -> str:
    """Remember a controller we just heard from. Idempotent; returns its key ("" if ignored).

    Called from BOTH directions: the UDP responder (a peer announced to us) and our own announce
    (a peer replied to us). Never raises — discovery must not be able to break the control plane."""
    try:
        http_port = int(http_port or 0)
    except (TypeError, ValueError):
        return ""
    if not host or http_port <= 0 or is_self(host, http_port):
        return ""
    k = peer_key(host, http_port)
    now = time.time()
    p = PEERS.get(k)
    if p is None:
        p = PEERS[k] = {"host": host, "http_port": http_port, "first_seen": now,
                        "info": None, "last_ok": 0.0, "error": "", "source": source}
        print(f"[peers] discovered controller {k}" + (f" ({name})" if name else ""), flush=True)
    p["last_seen"] = now
    if name:
        p["name"] = name
    if version:
        p["version"] = version
    if cluster_id or "cluster_id" not in p:
        p["cluster_id"] = cluster_id
    return k


def forget_peer(host: str, http_port) -> bool:
    return PEERS.pop(peer_key(host, http_port), None) is not None


def peer_state(p: dict) -> str:
    age = time.time() - float(p.get("last_ok") or p.get("last_seen") or 0)
    if p.get("error") and not p.get("info"):
        return "error"
    return "ok" if age < PEER_STALE_S else "stale"


def peers_public() -> list:
    """Peer list for the API/dashboard — newest-contact first, with a derived health state."""
    out = []
    for k, p in sorted(PEERS.items()):
        info = p.get("info") or {}
        out.append({
            "key": k, "host": p["host"], "http_port": p["http_port"],
            "name": p.get("name") or info.get("name") or "",
            "version": p.get("version") or info.get("version") or "",
            "cluster_id": p.get("cluster_id", ""),
            "state": peer_state(p), "source": p.get("source", ""),
            "last_seen_s": round(time.time() - float(p.get("last_seen") or 0), 1),
            "last_ok_s": (round(time.time() - float(p["last_ok"]), 1) if p.get("last_ok") else None),
            "error": p.get("error", ""),
            "url": f"http://{p['host']}:{p['http_port']}/",
            "models": info.get("models") or [],
            "nodes": info.get("nodes") or [],
        })
    return out


# --- what we serve to peers ----------------------------------------------------------------------

def self_info() -> dict:
    """Payload for GET /peer_info — our identity, our nodes and our resident models.

    Deliberately derived from live engine/registry state at call time (never cached) and stripped to
    what a peer needs: enough to display us, decide whether to borrow a model, and later negotiate
    node ownership. No secrets, no request contents."""
    ident = my_identity()
    models, nodes = [], []
    engine = getattr(state, "engine", None)
    registry = getattr(state, "registry", None)
    try:
        for friendly, m in list(getattr(engine, "models", {}).items()):
            models.append({
                "friendly": friendly,
                "target": getattr(m, "target_id", "") or getattr(m, "target", ""),
                "quant": getattr(m, "quant", "") or "none",
                "ctx": int(getattr(m, "ctx", 0) or 0),
                "size_gb": round(float(getattr(m, "size_gb", 0) or 0), 2),
                "active": int(getattr(m, "active", 0) or 0),
                "stages": model_hosts(m) or None,
            })
    except Exception as exc:   # noqa: BLE001 — /peer_info must never 500 a peer's gossip round
        models = []
        print(f"[peers] self_info models unavailable ({exc!r})", flush=True)
    try:
        for n in (registry.alive_sorted() if registry else []):
            nodes.append({
                "hostname": getattr(n, "hostname", ""),
                "device": getattr(n, "device_name", "") or getattr(n, "device", ""),
                "vram_total_gb": round(float(getattr(n, "vram_total_gb", 0) or 0), 2),
                "vram_used_gb": round(float(getattr(n, "vram_used_gb", 0) or 0), 2),
                "free_mem_gb": round(float(getattr(n, "free_mem_gb", 0) or 0), 2),
                "ram_enabled": bool(getattr(n, "ram_enabled", True)),
                "vram_enabled": bool(getattr(n, "vram_enabled", True)),
            })
    except Exception as exc:   # noqa: BLE001
        nodes = []
        print(f"[peers] self_info nodes unavailable ({exc!r})", flush=True)
    return {"im": "infinitemodel-peer", "v": 1, **ident,
            "models": models, "nodes": nodes, "claims": my_claims(), "t": time.time()}


# --- gossip --------------------------------------------------------------------------------------

# --- #federation Phase 3: request federation -------------------------------------------------------

def federation_enabled() -> bool:
    """`federate` engine knob (default ON). Off = never route a request to a peer."""
    cfg = getattr(state, "ENGINE_CONFIG", None) or {}
    return bool(cfg.get("federate", True))


def _model_aliases(m: dict) -> set:
    """Every name a peer's model answers to, normalised for matching."""
    out = set()
    for v in (m.get("friendly"), m.get("target")):
        if v:
            out.add(str(v).strip().lower())
    for v in (m.get("aliases") or []):
        out.add(str(v).strip().lower())
    return out


def find_model_peer(name: str) -> tuple:
    """(peer, model) for a HEALTHY peer that already has `name` RESIDENT, else (None, None).

    Only "ok" peers are considered — a stale peer's inventory is a guess, and routing a live request
    at a controller we cannot currently reach would turn a servable request into a timeout. Among
    candidates we prefer the least busy (fewest active requests) so federation spreads rather than
    piles onto one box."""
    want = str(name or "").strip().lower()
    if not want:
        return (None, None)
    best = (None, None, 1 << 30)
    for p in PEERS.values():
        if peer_state(p) != "ok":
            continue
        for m in ((p.get("info") or {}).get("models") or []):
            if want in _model_aliases(m):
                act = int(m.get("active") or 0)
                if act < best[2]:
                    best = (p, m, act)
    return (best[0], best[1])


def peer_base(p: dict) -> str:
    return f"http://{p['host']}:{p['http_port']}"


def _http_get_json(url: str, timeout: float = PEER_HTTP_TIMEOUT) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "infinitemodel-peer/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


async def poll_peer(p: dict) -> bool:
    """Fetch one peer's /peer_info into its cache. Returns True on success; never raises."""
    url = f"http://{p['host']}:{p['http_port']}/peer_info"
    try:
        info = await asyncio.to_thread(_http_get_json, url)
        if not isinstance(info, dict) or info.get("im") != "infinitemodel-peer":
            raise ValueError("not an InfiniteModel controller")
        p["info"] = info
        p["last_ok"] = time.time()
        p["error"] = ""
        for f in ("name", "version", "cluster_id"):
            if info.get(f):
                p[f] = info[f]
        return True
    except Exception as exc:   # noqa: BLE001 — a dead peer must never break the loop
        p["error"] = f"{type(exc).__name__}: {exc}"
        return False


async def gossip_round() -> dict:
    """Poll every known peer once, then prune the long-dead. Returns a small summary."""
    targets = list(PEERS.values())
    results = await asyncio.gather(*(poll_peer(p) for p in targets), return_exceptions=True)
    ok = sum(1 for r in results if r is True)
    now = time.time()
    for k, p in list(PEERS.items()):
        last = float(p.get("last_ok") or p.get("last_seen") or 0)
        if now - last > PEER_DROP_S:
            PEERS.pop(k, None)
            print(f"[peers] dropped {k} (no contact for {PEER_DROP_S/60:.0f} min)", flush=True)
    return {"polled": len(targets), "ok": ok, "peers": len(PEERS)}


def _announce_once(timeout: float = 2.0) -> int:
    """Broadcast 'a controller lives here' and record everyone who answers. Returns #replies.

    Same datagram shape as a worker's discovery query so the existing responder handles it, plus
    peer:1 + our identity so the receiver can record US without a second round trip."""
    ident = my_identity()
    cfg = _cfg()
    try:
        port = int(cfg.get("discovery_port") or 50099)
    except (TypeError, ValueError):
        port = 50099
    q = json.dumps({"im": wire.DISCOVERY_MAGIC, "q": 1, "peer": 1,
                    "cluster_id": ident["cluster_id"], "http_port": ident["http_port"],
                    "name": ident["name"], "version": ident["version"]}).encode()
    seen = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.25)
        s.bind(("", 0))
        for t in wire._broadcast_targets():
            try:
                s.sendto(q, (t, port))
            except OSError:
                pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw, addr = s.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict) or msg.get("im") != wire.DISCOVERY_MAGIC or not msg.get("r"):
                continue
            if record_peer(str(msg.get("host") or addr[0]), msg.get("http_port"),
                           str(msg.get("name") or ""), str(msg.get("version") or ""),
                           str(msg.get("cluster_id") or ""), source="udp"):
                seen += 1
    finally:
        s.close()
    return seen


# --- #federation Phase 5: exclusive node ownership -------------------------------------------------
# Two controllers planning against ONE node's memory is the double-booking that OOMs it: each is
# blind to the other's reserved-but-unfaulted KV and to its in-flight reservations, so both read the
# same "free" bytes and both commit. Rather than teach two planners to share (which needs
# worker-owned admission control — Phase 6), ownership is made EXCLUSIVE: a node in use by one
# controller is simply invisible to the others. Lending is then just "stop using it here".


def respect_peer_claims() -> bool:
    cfg = getattr(state, "ENGINE_CONFIG", None) or {}
    return bool(cfg.get("respect_peer_claims", True))


def _node_hostnames() -> dict:
    """node_id -> hostname. LoadedModel records stage_node_ids (IDs, server.py:1662), NOT node
    objects, so every "which hosts is this model on" answer has to go through the registry."""
    registry = getattr(state, "registry", None)
    out = {}
    try:
        for nid, n in (getattr(registry, "_nodes", {}) or {}).items():
            h = getattr(n, "hostname", "")
            if h:
                out[nid] = h
    except Exception:   # noqa: BLE001
        pass
    return out


def model_hosts(m) -> list:
    """Hostnames a LoadedModel occupies, from EVERY source that can answer.

    Two sources because neither is complete on its own:
      * plan.stages carries hostname directly (what /status renders) — but an ADOPTED model
        (hitless controller restart, worker kept its shards) comes back with an empty plan.
      * stage_node_ids maps through the registry — but a worker that re-registered after a restart
        is minted a NEW node_id (server.py:659), so an adopted model's ids can be stale.
    Union both and dedupe. Empty is a legitimate answer (nothing resolvable) — callers treat "no
    claim" as "not claimed", which fails OPEN rather than stranding a node."""
    out = []
    try:
        for s in (getattr(getattr(m, "plan", None), "stages", None) or []):
            h = getattr(s, "hostname", "")
            if h and h not in out:
                out.append(h)
    except Exception:   # noqa: BLE001
        pass
    by_id = _node_hostnames()
    for nid in (getattr(m, "stage_node_ids", None) or []):
        h = by_id.get(nid)
        if h and h not in out:
            out.append(h)
    return out


def claims_diagnostic() -> dict:
    """Why a model does/doesn't advertise a claim — so an empty claim set is debuggable from the
    dashboard instead of being a silent hole in Phase 5's exclusivity."""
    engine = getattr(state, "engine", None)
    by_id = _node_hostnames()
    rows = []
    for fr, m in list(getattr(engine, "models", {}) or {}).items():
        plan_hosts = [getattr(s, "hostname", "") for s in
                      (getattr(getattr(m, "plan", None), "stages", None) or [])]
        ids = list(getattr(m, "stage_node_ids", None) or [])
        rows.append({"model": fr, "plan_stage_hosts": [h for h in plan_hosts if h],
                     "stage_node_ids": ids,
                     "ids_resolved": [by_id.get(i) for i in ids],
                     "hosts": model_hosts(m)})
    return {"registry_node_ids": sorted(by_id), "models": rows, "claims": my_claims()}


def my_claims() -> list:
    """Hostnames WE currently hold a shard on — advertised so peers keep off them."""
    engine = getattr(state, "engine", None)
    out = set()
    try:
        for m in list(getattr(engine, "models", {}).values()):
            out.update(model_hosts(m))
    except Exception:   # noqa: BLE001 — claims are advisory; never break a load computing them
        pass
    return sorted(out)


def peer_claimed_hosts() -> dict:
    """hostname -> peer name, for every node a HEALTHY peer says it is using.

    Only "ok" peers count: a stale peer's claim is a guess, and honouring a guess would strand
    capacity we could legitimately use."""
    out = {}
    if not respect_peer_claims():
        return out
    for p in PEERS.values():
        if peer_state(p) != "ok":
            continue
        for h in ((p.get("info") or {}).get("claims") or []):
            out.setdefault(str(h), p.get("name") or p.get("host") or "peer")
    return out


def is_peer_claimed(hostname: str) -> bool:
    """True if a healthy peer is using `hostname` AND we are not. Ours always wins: if we hold a
    shard there the peer's claim is stale, and stranding our own resident model would be worse."""
    if not hostname:
        return False
    if hostname in set(my_claims()):
        return False
    return hostname in peer_claimed_hosts()


# --- #federation Phase 4: pull a model's weights FROM a peer ---------------------------------------
# Progress ledger for in-flight peer pulls. target -> {...}. Mutated in place (state.publish shares
# references), read by GET /peer_pull_status and rendered on the Controllers page.
PULLS: dict = {}


def pull_state(target: str) -> dict:
    return PULLS.get(target) or {}


def pulls_public() -> list:
    out = []
    for t, p in sorted(PULLS.items()):
        tot, done = float(p.get("total_bytes") or 0), float(p.get("done_bytes") or 0)
        out.append({**p, "target": t,
                    "pct": (round(100.0 * done / tot, 1) if tot > 0 else None)})
    return out


def safe_rel(root: str, rel: str) -> str:
    """Resolve `rel` under `root`, refusing anything that escapes it.

    /peer_model_file takes a caller-supplied path, so this is the security boundary: absolute paths,
    drive letters, symlink games and ../ traversal all have to fail CLOSED, or a peer could read
    arbitrary files off this controller."""
    import os
    raw = str(rel or "")
    norm = raw.replace("\\", "/")
    if not norm or ".." in norm.split("/"):
        raise ValueError("invalid path")
    # REJECT absolute forms outright rather than coercing them to relative. Silently stripping a
    # leading "/" would turn /etc/passwd into <root>/etc/passwd — harmless but dishonest — and on
    # Windows os.path.join(root, "C:/Windows/x") returns the DRIVE-QUALIFIED path, a real escape.
    if norm.startswith("/") or os.path.isabs(raw) or (len(raw) > 1 and raw[1] == ":"):
        raise ValueError("absolute paths are not allowed")
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, norm))
    try:
        inside = os.path.commonpath([root_real, full]) == root_real
    except ValueError:                     # different drives on Windows -> definitively outside
        inside = False
    if not inside:
        raise ValueError("path escapes the model directory")
    return full


def dir_manifest(root: str) -> list:
    """Every real file under `root` as (rel, size), sorted — the peer-pull contract."""
    import os
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            fp = os.path.join(dirpath, fn)
            if not os.path.isfile(fp):
                continue
            rel = os.path.relpath(fp, root).replace("\\", "/")
            try:
                out.append({"path": rel, "size": os.path.getsize(fp)})
            except OSError:
                pass
    return sorted(out, key=lambda e: e["path"])


async def pull_from_peer(p: dict, model: str, target: str, dest: str) -> dict:
    """Download every file of `model` from peer `p` into `dest`. Updates PULLS[target] as it goes.

    Streams to `<file>.part` and renames on completion, so an interrupted pull can never leave a
    truncated weight file that looks whole to a later load. Skips files already present at the
    advertised size, so a re-run resumes rather than re-downloading."""
    import os
    st = PULLS[target] = {"state": "manifest", "peer": peer_base(p), "peer_name": p.get("name", ""),
                          "model": model, "started": time.time(), "done_bytes": 0,
                          "total_bytes": 0, "file": "", "files_done": 0, "files_total": 0,
                          "error": "", "finished": 0.0}
    try:
        man = await asyncio.to_thread(
            _http_get_json, f"{peer_base(p)}/peer_model_manifest?model={urllib.parse.quote(model)}", 30.0)
        if not man.get("ok"):
            raise RuntimeError(man.get("error") or "peer refused the manifest")
        files = man.get("files") or []
        if not files:
            raise RuntimeError("peer reports no files for that model")
        st["files_total"] = len(files)
        st["total_bytes"] = sum(int(f.get("size") or 0) for f in files)
        st["state"] = "downloading"
        os.makedirs(dest, exist_ok=True)
        for f in files:
            rel, size = f["path"], int(f.get("size") or 0)
            st["file"] = rel
            out = safe_rel(dest, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out) and os.path.getsize(out) == size and size > 0:
                st["done_bytes"] += size          # already have it (resume)
                st["files_done"] += 1
                continue
            url = (f"{peer_base(p)}/peer_model_file?model={urllib.parse.quote(model)}"
                   f"&path={urllib.parse.quote(rel)}")

            def _fetch(u=url, o=out):
                req = urllib.request.Request(u, headers={"User-Agent": "infinitemodel-peer/1"})
                got = 0
                with urllib.request.urlopen(req, timeout=120.0) as r, open(o + ".part", "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        st["done_bytes"] += len(chunk)
                os.replace(o + ".part", o)
                return got

            await asyncio.to_thread(_fetch)
            st["files_done"] += 1
        st["state"] = "done"
        st["finished"] = time.time()
        print(f"[peers] pulled {model} from {peer_base(p)} "
              f"({st['files_done']} files, {st['done_bytes']/1e9:.2f} GB)", flush=True)
        return st
    except Exception as exc:   # noqa: BLE001 — a failed pull must not take the controller down
        st["state"] = "error"
        st["error"] = f"{type(exc).__name__}: {exc}"
        st["finished"] = time.time()
        print(f"[peers] pull of {model} from {peer_base(p)} FAILED: {st['error']}", flush=True)
        return st


async def announce_loop() -> None:
    """Periodically announce ourselves so controllers that boot later still find us."""
    while True:
        try:
            await asyncio.to_thread(_announce_once)
        except Exception as exc:   # noqa: BLE001
            print(f"[peers] announce failed ({exc!r})", flush=True)
        await asyncio.sleep(PEER_ANNOUNCE_S)


async def gossip_loop() -> None:
    while True:
        await asyncio.sleep(PEER_POLL_S)
        try:
            await gossip_round()
        except Exception as exc:   # noqa: BLE001
            print(f"[peers] gossip round failed ({exc!r})", flush=True)
