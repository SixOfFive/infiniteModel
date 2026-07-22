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
                "stages": [getattr(n, "hostname", "") for n in getattr(m, "stage_nodes", [])] or None,
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
            "models": models, "nodes": nodes, "t": time.time()}


# --- gossip --------------------------------------------------------------------------------------

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
