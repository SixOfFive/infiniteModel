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

PEER_POLL_S = 10.0          # how often we re-poll each peer's /peer_info
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


def is_master() -> bool:
    """#master: this controller is the DESIGNATED primary owner of the fleet (a per-controller flag,
    engine_config.json — NOT synced). Drives the dashboard's one-click 'restore fleet to master'
    after a failover; it does not change any placement/serving behaviour on its own."""
    return bool((getattr(state, "ENGINE_CONFIG", None) or {}).get("master", False))


def my_identity() -> dict:
    """What we advertise about ourselves over UDP and at /peer_info."""
    return {
        "name": socket.gethostname(),
        "http_port": my_http_port(),
        "version": str(getattr(state, "VERSION", "") or ""),
        "cluster_id": str(_cfg().get("cluster_id") or ""),
        "master": is_master(),
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
            "master": bool(info.get("master")),   # #master: peer is the designated fleet owner
            "state": peer_state(p), "source": p.get("source", ""),
            "last_seen_s": round(time.time() - float(p.get("last_seen") or 0), 1),
            "last_ok_s": (round(time.time() - float(p["last_ok"]), 1) if p.get("last_ok") else None),
            "error": p.get("error", ""),
            "url": f"http://{p['host']}:{p['http_port']}/",
            "models": info.get("models") or [],
            "nodes": info.get("nodes") or [],
            "disk_models": info.get("disk_models") or [],   # on-disk = pullable (incl. not-loaded)
        })
    return out


# --- #unified-fleet: ONE fleet seen from either controller ------------------------------------------
# Phases 1-5 made a peer's inventory *reachable* (borrow a model, keep off its nodes). This makes it
# *visible in the ordinary UI*: every controller renders the whole fleet — its own nodes and models
# plus its peers' — and can drive a load/unload anywhere by proxying to the owner. The physical
# invariant is unchanged and non-negotiable: ONE controller drives any given shard (a shard has one
# KV cache and per-controller req_id/slot counters would interleave). So "shared" means one physical
# copy with two front doors, never two drivers.


def healthy_peers() -> list:
    return [p for p in PEERS.values() if peer_state(p) == "ok"]


def peer_label(p: dict) -> str:
    return str(p.get("name") or (p.get("info") or {}).get("name") or p.get("host") or "peer")


def _stamp(row: dict, p: dict) -> dict:
    """Mark a row as belonging to a peer, so the UI can never confuse it with something we drive."""
    return {**row, "federated": True, "owner": peer_label(p),
            "owner_key": peer_key(p["host"], p["http_port"]), "owner_url": peer_base(p)}


PEER_STATUS_FRESH_S = 45.0   # a cached /status older than this is not trusted for rendering


def _rich(p: dict) -> dict:
    """The peer's last /status snapshot, if recent enough to render. {} otherwise."""
    st = p.get("status")
    if not isinstance(st, dict):
        return {}
    return st if (time.time() - float(p.get("status_ts") or 0)) <= PEER_STATUS_FRESH_S else {}


def federated_nodes() -> list:
    """Every node our healthy peers own, stamped with the owner. Deduped against OUR OWN nodes:
    a node can only be driven by one controller, so if it is in our registry it is ours to show.

    Rows come from the peer's own /status when we have a fresh one (identical to what IT renders,
    sparklines included) and fall back to the /peer_info summary otherwise."""
    mine = set()
    registry = getattr(state, "registry", None)
    try:
        for n in (registry.alive_sorted() if registry else []):
            mine.add(getattr(n, "hostname", ""))
    except Exception:   # noqa: BLE001
        pass
    out, seen = [], set()
    for p in healthy_peers():
        rows = (_rich(p).get("nodes") or []) or ((p.get("info") or {}).get("nodes") or [])
        for n in rows:
            h = str(n.get("hostname") or "")
            if not h or h in mine or h in seen:
                continue
            seen.add(h)
            out.append(_stamp(n, p))
    return out


def _peer_model_card(m: dict, p: dict) -> dict:
    """One of the peer's OWN model cards, made honest about what is and isn't true HERE.

    Everything measured stays verbatim — that is the entire point. What changes is anything that
    would imply a local capability we do not have: the weights are not on OUR disk, so `ready` is
    false and the shard-cache block is dropped (otherwise the detail modal offers to compile a cache
    for files we do not hold)."""
    out = {**m, "ready": False, "status": "peer", "loaded": True}
    out.pop("cached", None)
    out.pop("upgrade", None)      # a placement upgrade is the OWNER's to apply, not ours
    return _stamp(out, p)


def federated_models() -> list:
    """Every model our healthy peers have RESIDENT, stamped with the owner.

    Excludes anything we hold ourselves (ours is the copy we can actually drive) so the caller can
    concatenate without producing two rows for one model."""
    engine = getattr(state, "engine", None)
    mine = set()
    try:
        for fr, m in list((getattr(engine, "models", None) or {}).items()):
            mine.add(str(fr).strip().lower())
            for a in _model_aliases({"friendly": fr, "target": getattr(m, "target_id", "")}):
                mine.add(a)
    except Exception:   # noqa: BLE001
        pass
    out, seen = [], set()
    for p in healthy_peers():
        # `not m["federated"]`: a peer's /status also carries ITS view of everyone else (including
        # us). Only rows the peer DRIVES are its own to advertise — re-broadcasting a second-hand
        # row would attribute a third controller's model to this one, and bounce our own models
        # back at us wearing the peer's name.
        rich = [m for m in (_rich(p).get("models") or [])
                if m.get("loaded") and not m.get("federated")]
        for m in (rich or ((p.get("info") or {}).get("models") or [])):
            # A /status card names itself with name/internal_name; a /peer_info summary with
            # friendly/target. Match on every one of them so dedupe works whichever we got.
            names = _model_aliases({"friendly": m.get("friendly") or m.get("internal_name"),
                                    "target": m.get("target"),
                                    "aliases": (m.get("aliases") or []) + [m.get("name")]})
            if not names or (names & mine) or (names & seen):
                continue
            seen |= names
            out.append(_peer_model_card(m, p) if rich else _stamp(m, p))
    return out


def federated_totals() -> dict:
    """The capacity + throughput our PEERS contribute, for the dashboard's fleet tiles.

    Capacity is recomputed from the DEDUPED peer node rows (federated_nodes) using the same
    arithmetic build_status applies to our own nodes — deliberately NOT summed from the peers'
    `pool` blocks, which would double-count the instant two controllers disagree about who owns a
    node. Throughput/busy DO come from each peer's own aggregate: a controller reports only traffic
    it served, so those cannot double-count.

    Everything is the peers' contribution ALONE; the caller adds ours. Keeping the two halves
    separate is what lets the UI say "of which N via peers" instead of one unattributable number."""
    nodes = federated_nodes()
    t = {"controllers": [], "nodes": len(nodes), "gpus": 0,
         "ram_total_gb": 0.0, "ram_free_gb": 0.0, "vram_total_gb": 0.0, "vram_free_gb": 0.0,
         "tokens_per_s": 0.0, "units_busy": 0.0, "units_total": 0}
    for n in nodes:
        try:
            if float(n.get("vram_total_gb") or 0) > 0:
                t["gpus"] += 1
            if n.get("ram_enabled", True):
                t["ram_total_gb"] += float(n.get("total_mem_gb") or 0)
                t["ram_free_gb"] += float(n.get("free_mem_gb") or 0)
            if n.get("vram_enabled", True):
                vt = float(n.get("vram_total_gb") or 0)
                t["vram_total_gb"] += vt
                t["vram_free_gb"] += max(0.0, vt - float(n.get("vram_used_gb") or 0))
        except (TypeError, ValueError):
            continue
    for p in healthy_peers():
        st = _rich(p)
        if not st:
            continue
        t["controllers"].append(peer_label(p))
        try:
            t["tokens_per_s"] += float((st.get("metrics") or {}).get("tokens_per_s") or 0)
            _c = st.get("compute") or {}
            t["units_busy"] += float(_c.get("units_busy") or 0)
            t["units_total"] += int(_c.get("units_total") or 0)
        except (TypeError, ValueError):
            pass
    for k in ("ram_total_gb", "ram_free_gb", "vram_total_gb", "vram_free_gb", "tokens_per_s",
              "units_busy"):
        t[k] = round(t[k], 2)
    return t


def find_peer(sel: str) -> dict | None:
    """Resolve a peer by "host:port", bare host, or advertised name. None if we don't know it."""
    s = str(sel or "").strip()
    if not s:
        return None
    if s in PEERS:
        return PEERS[s]
    for p in PEERS.values():
        if s == p.get("host") or s.lower() == peer_label(p).lower():
            return p
    return None


def peer_for_model(name: str) -> dict | None:
    """The healthy peer that has `name` resident (federated load/unload target)."""
    return find_model_peer(name)[0]


def http_get_text(url: str, timeout: float = 20.0) -> str:
    """GET a text/SVG body from a peer (the detail-graph proxy). Raises on failure — the caller
    decides whether to fall back."""
    req = urllib.request.Request(url, headers={"User-Agent": "infinitemodel-peer/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def http_post_json(url: str, timeout: float = 120.0) -> dict:
    """POST with no body — the shape every controller op route takes (params in the query string)."""
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"User-Agent": "infinitemodel-peer/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:          # a 4xx/5xx still carries the peer's JSON error
        try:
            return json.loads(e.read().decode("utf-8", "replace"))
        except Exception:                        # noqa: BLE001
            return {"ok": False, "error": f"peer HTTP {e.code}"}


async def kick(p: dict) -> None:
    """Re-poll one peer NOW. Called right after we change something on it, so our view reflects the
    change immediately instead of up to PEER_POLL_S later — that lag is what makes two controllers
    feel out of sync even when they agree."""
    try:
        await poll_peer(p)
    except Exception:   # noqa: BLE001
        pass


# --- what we serve to peers ----------------------------------------------------------------------

def _model_size_gb(m) -> float:
    """A resident model's weight size, from its SPEC (the LoadedModel carries no size_gb)."""
    try:
        b = float(getattr(getattr(m, "spec", None), "total_weight_bytes", 0) or 0)
        if b > 0:
            return round(b / 1e9, 2)
    except Exception:   # noqa: BLE001
        pass
    return round(float(getattr(m, "size_gb", 0) or 0), 2)


def disk_models() -> list:
    """Every model this controller has WEIGHTS ON DISK for — PULLABLE by a peer whether or not it is
    currently loaded. The resident list only shows what's running; this is what lets a peer copy a
    model we aren't serving right now (the "no list of models not loaded to transfer" gap). Cheap:
    model_ready is a TTL-cached filesystem check and specs for on-disk models resolve locally."""
    out = []
    engine = getattr(state, "engine", None)
    MODELS = getattr(state, "MODELS", None) or {}
    model_ready = getattr(state, "model_ready", None)
    resolve_spec = getattr(state, "resolve_spec", None)
    dwb = getattr(state, "_display_weight_bytes", None)
    ollama = getattr(state, "_ollama_name", None)
    if not (MODELS and callable(model_ready)):
        return out
    resident = set()
    try:
        for mm in (getattr(engine, "models", {}) or {}).values():
            t = getattr(mm, "target_id", "") or getattr(mm, "target", "")
            if t:
                resident.add(t)
    except Exception:   # noqa: BLE001
        pass
    for friendly, tv in list(MODELS.items()):
        try:
            tgt = tv[0] if isinstance(tv, (tuple, list)) else tv
            if not tgt or not model_ready(tgt):
                continue
            sz = 0.0
            try:
                spec = resolve_spec(tgt) if callable(resolve_spec) else None
                if spec and callable(dwb):
                    sz = float(dwb(tgt, spec) or 0) / 1e9
            except Exception:   # noqa: BLE001
                pass
            out.append({"friendly": friendly,
                        "display_name": ollama(friendly) if callable(ollama) else friendly,
                        "target": tgt, "size_gb": round(sz, 2),
                        "loaded": tgt in resident})
        except Exception:   # noqa: BLE001
            continue
    return sorted(out, key=lambda e: e["friendly"])


def _model_runtime(friendly: str, m) -> dict:
    """One resident model as a peer needs to SEE it — #unified-fleet.

    Phase 1-5 sent 7 fields (just enough to decide "can I borrow this?"). A peer that must RENDER
    the model in its own Models list needs what the local card carries: quant/ctx/footprint/speed/
    placement. Everything here is already public at /status; nothing new is exposed."""
    g = lambda a, d=None: getattr(m, a, d)
    out = {
        "friendly": friendly,
        "display_name": str(g("display_name", "") or friendly),
        "target": g("target_id", "") or g("target", ""),
        "aliases": list(g("aliases", None) or []),
        "quant": g("quant", "") or "none",
        "kv_quant": g("kv_quant", "") or "none",
        "ctx": int(g("ctx", 0) or 0),
        # size comes off the model's SPEC — LoadedModel has no `size_gb` attr, so the old
        # getattr(m,"size_gb") always read 0 (why every peer model showed "0 GB" in the UI).
        "size_gb": _model_size_gb(m),
        "active": int(g("active", 0) or 0),
        "queued": int(g("queued", 0) or 0),
        "stages": model_hosts(m) or None,
    }
    # Runtime numbers are best-effort: a model mid-load may not have them yet, and a missing
    # speed reading must never cost us the whole gossip payload.
    for src, dst, cast in (("num_layers", "num_layers", int), ("params", "params", str),
                           ("arch", "arch", str), ("is_moe", "is_moe", bool),
                           ("is_embedding", "is_embedding", bool),
                           ("is_tts", "is_tts", bool), ("is_t2a", "is_t2a", bool),
                           ("loaded_at_ts", "loaded_at_ts", float),
                           ("last_used", "last_used_ts", float)):
        try:
            v = g(src, None)
            if v is not None:
                out[dst] = cast(v)
        except (TypeError, ValueError):
            pass
    try:                       # measured footprint + decode speed, same basis as our own card
        out["vram_used_gb"] = round(sum(float(getattr(s, "gpu_bytes", 0) or 0)
                                        for s in (getattr(g("plan"), "stages", None) or [])) / 1e9, 2)
    except Exception:          # noqa: BLE001
        pass
    for a in ("tok_s", "ema_tok_s", "last_tok_s", "max_tok_s"):
        try:
            out[a] = round(float(g(a, 0) or 0), 2)
        except (TypeError, ValueError):
            pass
    return out


def self_info() -> dict:
    """Payload for GET /peer_info — our identity, our nodes and our resident models.

    Deliberately derived from live engine/registry state at call time (never cached) and stripped to
    what a peer needs: enough to display us, decide whether to borrow a model, and later negotiate
    node ownership. No secrets, no request contents.

    #unified-fleet: nodes are sent as the FULL Node.to_dict() — the same rows /status publishes —
    so a peer can render our fleet in its own Nodes grid instead of a 7-field stub. These are
    hardware/telemetry facts (memory, utilisation, versions) that /status already serves openly."""
    ident = my_identity()
    models, nodes = [], []
    engine = getattr(state, "engine", None)
    registry = getattr(state, "registry", None)
    try:
        for friendly, m in list(getattr(engine, "models", {}).items()):
            models.append(_model_runtime(friendly, m))
    except Exception as exc:   # noqa: BLE001 — /peer_info must never 500 a peer's gossip round
        models = []
        print(f"[peers] self_info models unavailable ({exc!r})", flush=True)
    try:
        for n in (registry.alive_sorted() if registry else []):
            try:
                nodes.append(n.to_dict())
            except Exception:   # noqa: BLE001 — one bad node must not blank the whole fleet
                nodes.append({"hostname": getattr(n, "hostname", "")})
    except Exception as exc:   # noqa: BLE001
        nodes = []
        print(f"[peers] self_info nodes unavailable ({exc!r})", flush=True)
    try:
        disk = disk_models()   # every model we have ON DISK, so a peer can pull one we aren't running
    except Exception as exc:   # noqa: BLE001
        disk = []
        print(f"[peers] self_info disk_models unavailable ({exc!r})", flush=True)
    return {"im": "infinitemodel-peer", "v": 1, **ident,
            "models": models, "nodes": nodes, "disk_models": disk,
            "claims": my_claims(), "t": time.time()}


# --- gossip --------------------------------------------------------------------------------------

# --- #federation Phase 3: request federation -------------------------------------------------------

def federation_enabled() -> bool:
    """`federate` engine knob (default ON). Off = never route a request to a peer."""
    cfg = getattr(state, "ENGINE_CONFIG", None) or {}
    return bool(cfg.get("federate", True))


def _norm_name(v) -> str:
    """Collapse a client-facing model name to its canonical dash form for MATCHING — mirrors the
    controller's own resolve_model_name normalisation, but dependency-free (peers.py is stdlib-only):
      lowercases; strips a trailing ':latest' (possibly stacked); turns the family:size ':' into '-'.
    So 'nomic-embed-text:latest', 'qwen3:4b', 'qwen3-4b:latest' all reduce the way the resolver does.
    Raw HF ids ('org/name') are left untouched. Federation matching skipped this, so a request that
    named a model in ANY colon form (the Ollama default) failed to match a peer's dash-form friendly
    name and never federated — it fell through to a local load that has no nodes to run it."""
    n = str(v or "").strip().lower()
    if not n or "/" in n:
        return n
    while n.endswith(":latest"):
        n = n[: -len(":latest")]
    if ":" not in n:
        return n
    head, _, tail = n.partition(":")
    return f"{head}-{tail}" if tail else head


def _model_aliases(m: dict) -> set:
    """Every name a peer's model answers to, normalised for matching. Includes BOTH the literal
    forms (friendly/target/aliases/name/internal_name — a /status card names itself differently
    than a /peer_info summary) AND their canonical dash forms, so matching is symmetric with the
    requester side (_name_candidates)."""
    out = set()
    for v in (m.get("friendly"), m.get("target"), m.get("name"), m.get("internal_name")):
        if v:
            out.add(str(v).strip().lower())
            out.add(_norm_name(v))
    for v in (m.get("aliases") or []):
        if v:                      # a None in the list would become the literal alias "none"
            out.add(str(v).strip().lower())
            out.add(_norm_name(v))
    out.discard("")
    return out


def _name_candidates(name: str) -> set:
    """The forms a requested name might match a peer model under: the raw lowercased name and its
    canonical dash form. Both, because a peer might advertise either."""
    raw = str(name or "").strip().lower()
    return {raw, _norm_name(raw)} - {""}


def find_model_peer(name: str) -> tuple:
    """(peer, model) for a HEALTHY peer that already has `name` RESIDENT, else (None, None).

    Only "ok" peers are considered — a stale peer's inventory is a guess, and routing a live request
    at a controller we cannot currently reach would turn a servable request into a timeout. Among
    candidates we prefer the least busy (fewest active requests) so federation spreads rather than
    piles onto one box."""
    want = _name_candidates(name)
    if not want:
        return (None, None)
    best = (None, None, 1 << 30)
    for p in PEERS.values():
        if peer_state(p) != "ok":
            continue
        for m in ((p.get("info") or {}).get("models") or []):
            if want & _model_aliases(m):
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
    except Exception as exc:   # noqa: BLE001 — a dead peer must never break the loop
        p["error"] = f"{type(exc).__name__}: {exc}"
        return False
    await _poll_peer_status(p)
    return True


async def _poll_peer_status(p: dict) -> None:
    """Also cache the peer's FULL dashboard payload — its own `/status?graphs=1`.

    /peer_info is a summary: enough to decide whether to borrow a model, not enough to RENDER one.
    A model's card (measured VRAM/RAM, KV reservation, CPU fraction, load time, lifetime totals) and
    its server-rendered tok/s sparkline are computed by the controller that drives it and exist
    nowhere else — reconstructing them from a summary is how the peer rows ended up hollow: no
    graphs, no memory utilisation. So we fetch the real thing on the gossip cadence and show the
    owner's own card verbatim.

    Failure is soft: the previous snapshot stays and the summary remains the fallback, because a
    peer whose /status hiccups should degrade to less detail, never to a blank fleet."""
    try:
        st = await asyncio.to_thread(
            _http_get_json, f"http://{p['host']}:{p['http_port']}/status?graphs=1", 20.0)
        if isinstance(st, dict) and st.get("nodes") is not None:
            p["status"] = st
            p["status_ts"] = time.time()
    except Exception as exc:   # noqa: BLE001 — richness is a bonus; never fail the gossip round
        p["status_error"] = f"{type(exc).__name__}: {exc}"


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
    for fr, m in list((getattr(engine, "models", None) or {}).items()):
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


def is_cache_file(rel: str) -> bool:
    """A pre-compiled shard-cache unit (_shards/<quant>/…) rather than a source weight file.

    These are regenerable — any controller can rebuild them from the weights — but they are often
    LARGER than the weights themselves (qwen3-4b: 7.4 GB of bf16 vs 15.2 GB with int4+int8 caches).
    On a small-disk peer that difference decides whether a model fits at all, so pulling them is
    opt-in."""
    return rel.replace("\\", "/").startswith("_shards/")


def split_manifest(files: list) -> tuple:
    """(weights, caches) — same split the manifest reports and the puller filters on."""
    w = [f for f in files if not is_cache_file(f.get("path", ""))]
    c = [f for f in files if is_cache_file(f.get("path", ""))]
    return w, c


async def pull_from_peer(p: dict, model: str, target: str, dest: str,
                         include_caches: bool = False) -> dict:
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
        if not include_caches:
            weights, caches = split_manifest(files)
            if caches:
                skipped = sum(int(f.get("size") or 0) for f in caches)
                st["skipped_cache_files"] = len(caches)
                st["skipped_cache_bytes"] = skipped
                print(f"[peers] {target}: skipping {len(caches)} shard-cache file(s) "
                      f"({skipped/1e9:.2f} GB) — pass caches=1 to include them", flush=True)
            files = weights
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
