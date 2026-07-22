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

import peers


def register(app):

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
