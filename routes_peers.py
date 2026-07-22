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
import os
import urllib.parse
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

    # --- #federation Phase 4: serve our model files to a peer, and pull a peer's -----------------

    @app.get("/peer_model_manifest")
    async def peer_model_manifest(model: str) -> JSONResponse:
        """Every file of one of OUR models, so a peer can mirror it byte-for-byte.

        Resolves through _controller_model_dir, so a model still only in the HF cache is
        materialised into models/ first — the peer then sees exactly what a local load would."""
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        if not d or not os.path.isdir(d):
            return JSONResponse({"ok": False, "error": "model directory unavailable"},
                                status_code=404)
        files = await asyncio.to_thread(peers.dir_manifest, d)
        return JSONResponse({"ok": True, "model": friendly, "target": target,
                             "files": files,
                             "total_bytes": sum(f["size"] for f in files)})

    @app.get("/peer_model_file")
    async def peer_model_file(model: str, path: str):
        """Stream ONE file of one of our models. `path` is caller-supplied, so peers.safe_rel is
        the security boundary — traversal/absolute/symlink escapes fail closed."""
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        if not d or not os.path.isdir(d):
            return JSONResponse({"ok": False, "error": "model directory unavailable"},
                                status_code=404)
        try:
            full = peers.safe_rel(d, path)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        if not os.path.isfile(full):
            return JSONResponse({"ok": False, "error": "no such file"}, status_code=404)
        return FileResponse(full, media_type="application/octet-stream")

    @app.post("/peer_pull")
    async def peer_pull(host: str, model: str, port: int = 0) -> JSONResponse:
        """Copy a model's weights from a peer controller instead of re-downloading from HuggingFace.

        Runs in the background (a pull is GBs); poll /peer_pull_status. On success the model is
        registered locally exactly as /add_model would, so it is immediately loadable here."""
        p = int(port or peers._cfg()["http_port"])
        key = peers.peer_key(host, p)
        peer = peers.PEERS.get(key) or {"host": host, "http_port": p, "name": ""}
        target = model.strip()
        try:                                   # a peer may advertise the friendly name; keep both
            man = await asyncio.to_thread(
                peers._http_get_json,
                f"http://{host}:{p}/peer_model_manifest?model={urllib.parse.quote(target)}", 30.0)
            if man.get("ok") and man.get("target"):
                target = man["target"]
        except Exception as exc:               # noqa: BLE001
            return JSONResponse({"ok": False, "error": f"peer manifest failed: {exc}"},
                                status_code=502)
        cur = peers.pull_state(target)
        if cur.get("state") in ("manifest", "downloading"):
            return JSONResponse({"ok": False, "error": "a pull for that model is already running",
                                 "pull": cur}, status_code=409)
        dest = os.path.join(MODELS_DIR, _safe_name(target))

        async def _run():
            st = await peers.pull_from_peer(peer, model, target, dest)
            if st.get("state") != "done":
                return
            friendly = _friendly_from_hf(target)          # register exactly like /add_model
            if friendly not in MODELS:
                MODELS[friendly] = (target, target)
                CUSTOM_MODELS[friendly] = target
                save_custom_models()
                log_activity(f"added model {friendly} ({target}) — pulled from peer "
                             f"{peer.get('name') or host}")
            DELETED_MODELS.discard(friendly)
            st["friendly"] = friendly

        asyncio.create_task(_run())
        return JSONResponse({"ok": True, "started": True, "target": target,
                             "peer": f"{host}:{p}", "dest": dest})

    # --- #federation Phase 5: exclusive node ownership / lending ---------------------------------

    @app.get("/peer_nodes")
    async def peer_nodes() -> JSONResponse:
        """Who owns what, across every controller we know about.

        For each node WE can see: whether we are using it, whether a peer claims it, and whether it
        is currently enabled here. This is the view the Controllers page renders to make lending a
        one-click decision instead of guesswork."""
        claimed = peers.peer_claimed_hosts()
        mine = set(peers.my_claims())
        rows = []
        for n in registry.alive_sorted():
            h = n.hostname
            rows.append({
                "hostname": h,
                "device": getattr(n, "device_name", "") or getattr(n, "device", ""),
                "vram_total_gb": round(float(getattr(n, "vram_total_gb", 0) or 0), 2),
                "ram_enabled": bool(getattr(n, "ram_enabled", True)),
                "vram_enabled": bool(getattr(n, "vram_enabled", True)),
                "mine": h in mine,
                "peer": claimed.get(h, ""),
                "lent": (not getattr(n, "ram_enabled", True)
                         and not getattr(n, "vram_enabled", True)),
            })
        return JSONResponse({"ok": True, "respect_peer_claims": peers.respect_peer_claims(),
                             "my_claims": sorted(mine), "peer_claims": claimed, "nodes": rows})

    @app.get("/peer_claims_debug")
    async def peer_claims_debug() -> JSONResponse:
        """Why each resident model does or doesn't produce a claim (Phase 5 exclusivity depends on
        claims being non-empty, and an ADOPTED model can legitimately resolve to nothing)."""
        return JSONResponse({"ok": True, **peers.claims_diagnostic()})

    @app.post("/peer_lend")
    async def peer_lend(node: str) -> JSONResponse:
        """Stop using a node here so another controller can take it.

        Lending is deliberately just "disable both tiers locally" — it reuses the existing
        /nodeconfig machinery, needs no agreement protocol, and cannot double-book: a node disabled
        here is invisible to OUR planner, and the peer's planner only ever saw its own view anyway.
        Resident models on that node are re-planned by the same tier-change path /nodeconfig uses."""
        cfg = NODE_CONFIG.setdefault(node, {"ram": True, "vram": True})
        cfg["ram"] = False
        cfg["vram"] = False
        save_node_config()
        host_nids = {nid for nid, n in registry._nodes.items() if n.hostname == node}
        for fr in [fr for fr, m in engine.models.items()
                   if any(nid in m.stage_node_ids for nid in host_nids)]:
            engine.invalidate_model(fr, f"lent {node} to a peer controller")
        log_activity(f"federation: lent node {node} to a peer (both tiers disabled here)")
        return JSONResponse({"ok": True, "node": node, "config": cfg})

    @app.post("/peer_reclaim")
    async def peer_reclaim(node: str) -> JSONResponse:
        """Take a lent node back: re-enable both tiers here."""
        cfg = NODE_CONFIG.setdefault(node, {"ram": True, "vram": True})
        cfg["ram"] = True
        cfg["vram"] = True
        save_node_config()
        log_activity(f"federation: reclaimed node {node} (both tiers re-enabled here)")
        return JSONResponse({"ok": True, "node": node, "config": cfg})

    @app.get("/peer_pull_status")
    async def peer_pull_status() -> JSONResponse:
        return JSONResponse({"ok": True, "pulls": peers.pulls_public()})

    @app.post("/peer_remove")
    async def peer_remove(host: str, port: int = 0) -> JSONResponse:
        p = int(port or peers._cfg()["http_port"])
        gone = peers.forget_peer(host, p)
        return JSONResponse({"ok": gone, "peers": peers.peers_public()})
