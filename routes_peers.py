"""routes_peers: #federation cross-controller routes. Controller-only leaf; in EXTRA_UPDATE_FILES.

Module globals (JSONResponse …) are injected at startup by state.bind() — see state.py.
build_app() calls register(app) to attach them.

  GET  /peer_info               what WE advertise to other controllers (their gossip target)
  GET  /peers                   peers we know about + their cached inventory (dashboard source)
  POST /peer_refresh            force an announce + gossip round now
  POST /peer_add?host=&port=    add a peer by hand (for controllers broadcast can't reach)
  POST /peer_remove?host=&port= forget a peer
"""
from __future__ import annotations

import json as _json
import urllib.error
import urllib.request

import peers

# #federation Phase 3: POST endpoints whose body names a model. A request for a model WE do not have
# resident, that a healthy peer DOES have resident, is proxied to that peer instead of cold-loading
# a second copy of the same weights.
FEDERATED_PATHS = frozenset({
    "/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/messages",
})
FEDERATE_TIMEOUT_S = 900.0


def _local_has(name: str) -> bool:
    """Is `name` resident HERE? Resolved through the engine's own name resolution so an alias or a
    canonical target id is judged exactly the way the serving path would judge it."""
    try:
        fr = resolve_model_name(name)
    except Exception:      # noqa: BLE001 — unknown name: not local, let federation try
        return False
    return fr in getattr(engine, "models", {})


async def _proxy_to_peer(p: dict, path: str, body: bytes, ctype: str):
    """Forward one request to a peer controller and stream its answer straight back.

    Streams chunk-by-chunk so SSE / NDJSON token streams stay live rather than buffering to
    completion. urllib (stdlib) in a worker thread — the controller takes no new dependency."""
    url = peers.peer_base(p) + path

    def _open():
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": ctype or "application/json",
                     "Accept": "*/*", "X-InfiniteModel-Federated": "1"})
        return urllib.request.urlopen(req, timeout=FEDERATE_TIMEOUT_S)

    resp = await asyncio.to_thread(_open)

    async def _gen():
        try:
            while True:
                chunk = await asyncio.to_thread(resp.read, 8192)
                if not chunk:
                    break
                yield chunk
        finally:
            with contextlib.suppress(Exception):
                resp.close()

    return StreamingResponse(_gen(), status_code=getattr(resp, "status", 200),
                             media_type=resp.headers.get("Content-Type", "application/json"))


def register(app):

    @app.middleware("http")
    async def federate_middleware(request, call_next):
        """Route a request for a model only a PEER has resident to that peer.

        Guards, in order: federation enabled; a federated path; not already a federated hop (the
        X-InfiniteModel-Federated header breaks any A->B->A loop); a model name in the body; NOT
        resident locally; a healthy peer that has it. Anything unexpected falls through to normal
        local handling — federation must never be able to break serving."""
        if (request.method != "POST" or request.url.path not in FEDERATED_PATHS
                or request.headers.get("X-InfiniteModel-Federated")
                or not peers.federation_enabled() or not peers.PEERS):
            return await call_next(request)
        try:
            body = await request.body()          # cached on the Request, so downstream re-reads work
            name = (_json.loads(body or b"{}") or {}).get("model")
        except Exception:                        # noqa: BLE001 — unparseable body: not ours to judge
            return await call_next(request)
        if not name or _local_has(name):
            return await call_next(request)
        p, m = peers.find_model_peer(name)
        if not p:
            return await call_next(request)
        log_activity(f"{name}: not resident here — federating to {p.get('name') or p['host']} "
                     f"({peers.peer_base(p)}), which has it loaded")
        try:
            return await _proxy_to_peer(p, request.url.path, body,
                                        request.headers.get("Content-Type", "application/json"))
        except Exception as exc:                 # noqa: BLE001 — peer died mid-flight: serve locally
            p["error"] = f"federate: {type(exc).__name__}: {exc}"
            log_activity(f"{name}: federation to {p['host']} failed ({exc!r}) — serving locally")
            return await call_next(request)

    @app.get("/controllers", response_class=HTMLResponse)   # dashboard: Cross-controller page
    async def controllers_page() -> str:
        return CONTROLLERS_HTML

    @app.get("/peer_info")
    async def peer_info() -> JSONResponse:
        """Our identity + nodes + resident models, for a peer controller's gossip poll.

        Read-only and cheap: derived from live engine/registry state, no locks taken. This is the
        ONLY thing a peer reads from us, so it is deliberately the whole federation contract."""
        return JSONResponse(peers.self_info())

    @app.get("/peers")
    async def list_peers() -> JSONResponse:
        """Every controller we currently know about, with its last-gossiped inventory."""
        return JSONResponse({"ok": True, "self": peers.my_identity(),
                             "peers": peers.peers_public()})

    @app.post("/peer_refresh")
    async def peer_refresh() -> JSONResponse:
        """Announce ourselves and re-poll every peer immediately (the dashboard refresh button)."""
        found = await asyncio.to_thread(peers._announce_once)
        summary = await peers.gossip_round()
        return JSONResponse({"ok": True, "announced_replies": found, **summary,
                             "peers": peers.peers_public()})

    @app.post("/peer_add")
    async def peer_add(host: str, port: int = 0) -> JSONResponse:
        """Add a controller by address. For peers UDP broadcast cannot reach — a different subnet,
        a VLAN, or across a VPN — which is exactly where auto-discovery does not work."""
        p = int(port or peers._cfg()["http_port"])
        if peers.is_self(host, p):
            return JSONResponse({"ok": False, "error": "that is this controller"}, status_code=400)
        key = peers.record_peer(host, p, source="manual")
        if not key:
            return JSONResponse({"ok": False, "error": "invalid host/port"}, status_code=400)
        ok = await peers.poll_peer(peers.PEERS[key])
        return JSONResponse({"ok": True, "key": key, "reachable": ok,
                             "error": peers.PEERS[key].get("error", ""),
                             "peers": peers.peers_public()})

    @app.post("/peer_remove")
    async def peer_remove(host: str, port: int = 0) -> JSONResponse:
        p = int(port or peers._cfg()["http_port"])
        gone = peers.forget_peer(host, p)
        return JSONResponse({"ok": gone, "peers": peers.peers_public()})
