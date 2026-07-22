"""routes_lifecycle: the model-lifecycle routes relocated from server.py build_app
(m4c153 code-split): /load, /model_config, /terminate, /cancel, /cancel_load, /unload,
/reconfigure, /restart, /update. Route bodies are BYTE-IDENTICAL to the originals; their
module globals (engine, registry, _serve, build_status, JSONResponse ...) are injected at
startup by state.bind() -- see state.py. build_app() calls register(app) to attach them.
The shard-cache / packing / weight-serving routes moved on to routes_shards.py (code-split
Inc 6). Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

# stdlib, imported HERE rather than relied on from the injected server namespace: server.py's
# `import urllib.request` sits inside a function, so `urllib` is not one of its module globals.
import urllib.parse   # noqa: E402


# --- #unified-fleet: drive a load/unload on a PEER controller ---------------------------------------
# A controller can only place work on nodes it owns (ownership is exclusive — see peers.py Phase 5),
# so on a controller with no nodes of its own the Load button used to be dead. Federating the
# OPERATION fixes that without breaking exclusivity: the owning controller performs the real load on
# its own nodes, and we then see the result through gossip and serve requests against it via Phase 3
# request federation. One physical copy, two front doors — never two drivers.

def _peers_mod():
    """peers.py, or None. Imported lazily and failing OPEN: a controller running without the
    federation leaf must behave exactly as it did before."""
    try:
        import peers
        return peers
    except Exception:   # noqa: BLE001
        return None


def _local_capacity() -> bool:
    """True if we own at least one alive node that could actually take a shard."""
    try:
        return any(getattr(n, "can_infer", True) for n in registry.alive_sorted())
    except Exception:   # noqa: BLE001
        return False


def _fed_target(peers, model: str, owner: str):
    """(peer, reason) for the controller that should perform this op — (None, why-not) otherwise."""
    if owner:
        p = peers.find_peer(owner)
        if p is None:
            return (None, f"unknown controller '{owner}'")
        if peers.peer_state(p) != "ok":
            return (None, f"controller '{owner}' is {peers.peer_state(p)}")
        return (p, "requested")
    if _local_capacity():
        return (None, "")          # we have nodes — this is our load to do
    live = peers.healthy_peers()
    if not live:
        return (None, "")          # nothing to hand to; let the local path produce its own error
    # Prefer a peer that already HAS the weights (no download), then the one with the most nodes.
    p, _m = peers.find_model_peer(model)
    if p is not None:
        return (p, "already resident there")
    best = max(live, key=lambda q: len(((q.get("info") or {}).get("nodes") or [])))
    return (best, "we own no nodes; it does")


def register(app):

    @app.post("/load")
    async def load(request: Request, model: str, ctx: int = 0, mode: str = "auto",
                   consolidate: bool = True, quant: str = "", tp: int = 1,
                   replicas: int = 1, cpu_only: bool = False,
                   moe_offload: bool = False, force: bool = False,
                   node: str = "", kv_quant: str = "",
                   kv_offload: bool = False, kv_slots: int = 1,
                   temperature: str = "",
                   min_p: str = "", precompile: bool = True,
                   draft_gpu: bool = False, draft_margin_gb: str = "",
                   t2i_offload: bool = False, owner: str = "") -> JSONResponse:
        _req_ip = _client_ip(request)   # #connections: attribute this load to its requester
        # #unified-fleet: hand the load to a peer controller when it — not us — owns the hardware.
        # `owner=<name|host>` forces a specific controller; with no owner this fires ONLY when we
        # own no usable nodes at all, so a controller with its own fleet is completely unaffected.
        _pm = _peers_mod()
        if _pm is not None and _pm.PEERS:
            _peer, _why = _fed_target(_pm, model, owner)
            if _peer is None and owner:
                return JSONResponse({"ok": False, "error": _why}, status_code=404)
            if _peer is not None:
                _q = "&".join(kv for kv in str(request.url.query or "").split("&")
                              if kv and not kv.startswith("owner="))
                # OUR defaults, not the peer's. The caller asked THIS controller to load something;
                # anything they left unspecified should follow the policy configured HERE. Without
                # this the request arrives bare and the peer fills the gaps from its own
                # autoload_quant/autoload_ctx, so a controller's own defaults quietly did nothing.
                _qkeys = {kv.split("=", 1)[0] for kv in _q.split("&") if kv}
                if "quant" not in _qkeys:
                    _q += f"&quant={ENGINE_CONFIG.get('autoload_quant') or 'int4'}"
                if "ctx" not in _qkeys and not ctx:
                    _actx = int(ENGINE_CONFIG.get("autoload_ctx") or 0)
                    if _actx > 0:
                        _q += f"&ctx={_actx}"
                _lbl = _pm.peer_label(_peer)
                log_activity(f"load {model}: federating to {_lbl} ({_why})")
                try:
                    _r = await asyncio.to_thread(
                        _pm.http_post_json, f"{_pm.peer_base(_peer)}/load?{_q}", 900.0)
                except Exception as _exc:      # noqa: BLE001 — peer unreachable: say so plainly
                    log_activity(f"load {model}: {_lbl} unreachable — {_exc}")
                    return JSONResponse({"ok": False, "federated_to": _lbl,
                                         "error": f"peer controller {_lbl} unreachable: {_exc}"},
                                        status_code=502)
                await _pm.kick(_peer)          # reflect the new model in our view immediately
                return JSONResponse({**_r, "federated_to": _lbl, "federated_reason": _why},
                                    status_code=(200 if _r.get("ok") else 400))
        # force=1 (#stuck-load-override): if a load of this model is already IN FLIGHT, CANCEL it and
        # restart fresh (the manual escape hatch for a wedged 0%-forever load) instead of queueing on
        # it. Also reloads an already-resident copy (skips the idempotent no-op). Without force, a
        # concurrent same-model request still queues on the in-flight load as before.
        # ctx=0 (default) => the model's native training context (config.json).
        # `mode` chooses HOW the model is placed (maps to consolidate, prefer_vram):
        #   auto       (T, T) GPU-VRAM-first, fewest nodes — best decode latency [default]
        #   single     (T, F) fewest nodes by total RAM+VRAM — collapses to one box if it fits
        #   gpu-spread (F, T) fill every GPU's VRAM, spill across nodes
        #   all-gpu    (F, F) a stage on EVERY GPU, NOTHING on CPU (proportional across the GPUs)
        #   distribute (F, F) spread across the WHOLE fleet (CPUs + GPUs)
        #   spread     (F, F) FORCE a stage on every capable node (incl. tiny ones)
        #   proportional (F, F) layers across EVERY capable node PROPORTIONAL to its capacity
        #                 (#78: big int4 MoE — MiniMax-M2 — too big for the GPU-first subset)
        # `quant`: 'none' (bf16), 'int8' (~1/2), 'int4' (group-wise ~4.25-bit, ~1/4 — for
        # 200B+ MoEs that won't fit at int8), or 'int2' (#int2: group-wise ~2.5-bit, ~1/6 —
        # a CAPACITY tier for dense models that won't fit at int4; visible quality loss; MoE
        # auto-downgrades to int4). `tp` (M4): tensor-parallel group
        # size — split every layer across `tp` GPU nodes (rank 0 drives the group over the
        # all-reduce mesh). tp>1 overrides mode. tp must divide num_key_value_heads.
        # Legacy: if mode is omitted but consolidate=false is passed, honor it.
        # #load-default-quant: an UNSPECIFIED quant inherits the fleet default (`autoload_quant`,
        # normally int4) — NOT bf16. The old hardcoded "none" default silently loaded a full-size
        # bf16 copy for any caller that omitted quant (a 30B MoE -> ~57 GB that spilled to CPU and
        # evicted its neighbours on a shared box), inconsistent with BOTH the dashboard load dialog
        # (defaults int4) and the auto-load path (autoload_quant). Explicit `quant=none` still loads
        # bf16 on purpose; the dashboard/T2I paths always send an explicit quant so are unaffected.
        if not quant:
            quant = str(ENGINE_CONFIG.get("autoload_quant") or "int4")
        if quant not in ("none", "int8", "int4", "int2"):
            return JSONResponse({"ok": False, "error": f"bad quant '{quant}' (none|int8|int4|int2)"},
                                status_code=400)
        # #172 TurboQuant KV preset. Empty -> _load_impl inherits the ENGINE_CONFIG knob (default 'none').
        if kv_quant and kv_quant not in ("none", "turbo2", "turbo3", "turbo4"):
            return JSONResponse({"ok": False,
                                 "error": f"bad kv_quant '{kv_quant}' (none|turbo2|turbo3|turbo4)"},
                                status_code=400)
        # #kv-offload: KV cache in system RAM instead of VRAM (frees VRAM for model layers at the
        # cost of decode speed — for pushing ctx past what the card holds). Mutually exclusive with
        # kv_quant: TurboQuantCache is a custom GPU-resident cache; combining them is untested.
        if kv_offload and kv_quant and kv_quant != "none":
            return JSONResponse({"ok": False, "error":
                                 "kv_offload and kv_quant are mutually exclusive (pick one)"},
                                status_code=400)
        # #kv-slots: per-replica concurrent decode slots C (opt-in; default 1 = today's exact
        # one-generation-per-replica behavior). Clamp band 1..8 (measured q-curves: CUDA forward
        # cost is flat to q=16, Strix steps at 16 — 8 is the validated opt-in ceiling). KV memory
        # scales xC: the planner reserves C x per-stream full-ctx KV and the load FAILS if that
        # doesn't fit. Hybrid/kv_quant/kv_offload/tp>1 are hard-gated to 1 inside engine.load;
        # a chain with any pre-'kvslots' worker REFUSES C>1 outright (wire-cap all-or-nothing).
        if not (1 <= int(kv_slots) <= 8):
            return JSONResponse({"ok": False,
                                 "error": f"bad kv_slots '{kv_slots}' (1-8)"}, status_code=400)
        # #load-temp: per-model DEFAULT temperature — used when a request doesn't send one
        # (explicit request values always win). Empty = unset (requests keep the global default).
        default_temp = None
        if str(temperature).strip():
            try:
                default_temp = float(temperature)
            except ValueError:
                return JSONResponse({"ok": False,
                                     "error": f"bad temperature '{temperature}' (float)"},
                                    status_code=400)
            if not (0.0 <= default_temp <= 2.0):
                return JSONResponse({"ok": False,
                                     "error": "temperature out of range (0.0 - 2.0)"},
                                    status_code=400)
        # #min-p: per-model DEFAULT min-p (confidence-adaptive sampling floor; pairs with a high
        # default temperature — 0.05-0.1 is the useful band at temperature >= 1.0). Same precedence
        # as temperature: applied only when a request sends no min_p of its own.
        default_min_p = None
        if str(min_p).strip():
            try:
                default_min_p = float(min_p)
            except ValueError:
                return JSONResponse({"ok": False,
                                     "error": f"bad min_p '{min_p}' (float)"}, status_code=400)
            if not (0.0 <= default_min_p <= 1.0):
                return JSONResponse({"ok": False,
                                     "error": "min_p out of range (0.0 - 1.0)"}, status_code=400)
        # #draft-gpu: opt-in — reserve the spec draft's bf16 size + margin on the CONTROLLER's
        # GPU at plan time (registry-pair models only), so the draft attaches on cuda instead of
        # CPU (a CPU draft step can cost more than the sweep it saves — see MODEL_TEST_STATUS
        # llama-3.3:70b). draft_margin_gb tunes _load_draft's VRAM cushion (default 4.0; a small
        # shared card may need 1.5-2.0). Single-pipeline loads only (replicas/tp share one draft).
        _draft_margin = 4.0
        if str(draft_margin_gb).strip():
            try:
                _draft_margin = float(draft_margin_gb)
            except ValueError:
                return JSONResponse({"ok": False,
                                     "error": f"bad draft_margin_gb '{draft_margin_gb}' (float)"},
                                    status_code=400)
            if not (0.0 <= _draft_margin <= 16.0):
                return JSONResponse({"ok": False,
                                     "error": "draft_margin_gb out of range (0.0 - 16.0)"},
                                    status_code=400)
        cons, pv = LOAD_MODES.get(mode, LOAD_MODES["auto"])
        if mode == "auto" and not consolidate:   # back-compat with the old checkbox
            cons, pv = False, True
        try:
            friendly = resolve_model_name(model)
            # #2: int8 on a MoE silently keeps the fused-3D routed experts in bf16 (the worker int8
            # path only quantizes 2D nn.Linears; there is no int8 3D-expert quantizer — same reason
            # /compile_shards + /compile_dist reject int8 MoE). That yields a ~bf16 footprint -> OOM /
            # CPU-spill. Auto-DOWNGRADE to int4 (which DOES pack experts) so the user gets a real memory
            # reduction. Reuse the controller's existing MoE detector (weight-map -> _has_moe_experts).
            # Best-effort: if we can't introspect the model, fall through and honor int8 as requested.
            if quant in ("int8", "int2"):
                # #int2: same shape of problem as int8-on-MoE — the int2 walker only quantizes 2D
                # nn.Linears (no 2-bit 3D-expert packer/kernel), so a MoE at int2 would keep its
                # routed experts (the bulk of the model) bf16. int4 DOES pack experts — downgrade.
                try:
                    import shards as _sh
                    _tgt = MODELS[friendly][0] if friendly in MODELS else friendly
                    _mdir = await asyncio.to_thread(_controller_model_dir, _tgt)
                    if _mdir:
                        _wm = await asyncio.to_thread(_sh._weight_map, _mdir)
                        if await asyncio.to_thread(_sh._has_moe_experts, _wm):
                            log_activity(f"{_ollama_name(friendly)}: {quant} on a MoE keeps experts bf16 "
                                         f"(no {quant} 3D-expert quantizer) — DOWNGRADING to int4 for a "
                                         "real memory reduction")
                            quant = "int4"
                except Exception as _moe_exc:
                    log_activity(f"{_ollama_name(friendly)}: MoE check for {quant} downgrade failed "
                                 f"({_moe_exc}) — honoring {quant} as requested")
            # #cache-on-first-load: for an int4 load with no shard cache yet, BUILD the cache first so
            # THIS load — and every future load — serves the small pre-packed int4 layers instead of
            # streaming full bf16 and re-quantizing on the fly. precompile=0 opts out. Shared with the
            # auto-load path (ensure_loaded) via engine._precompile_int4 so both compile identically;
            # the helper no-ops when a cache exists / quant!=int4 / tp>1 and is non-fatal on failure.
            if precompile:
                await engine._precompile_int4(friendly, quant, tp)
            # replicas>1 (#39): load N full copies on disjoint nodes for data-parallel
            # throughput. Mutually exclusive with tp (tp splits one copy; replicas duplicate it).
            if tp <= 1 and replicas > 1:
                lms = await engine.replicate(friendly, ctx, replicas,
                                             consolidate=cons, prefer_vram=pv, quant=quant,
                                             kv_quant=kv_quant, kv_offload=kv_offload,
                                             kv_slots=kv_slots,
                                             default_temp=default_temp,
                                             default_min_p=default_min_p)
                return JSONResponse({"ok": True, "model": friendly, "ctx": lms[0].ctx,
                                     "mode": mode, "quant": quant, "replicas": len(lms),
                                     "placements": [{"key": m.friendly,
                                                     "hosts": [s.hostname for s in m.plan.stages]}
                                                    for m in lms]})
            if node:               # #pin-device: pinning to one node is single-node pipeline (TP needs many)
                tp = 1
            lm = await engine.load(friendly, ctx, consolidate=cons, prefer_vram=pv,
                                   quant=quant, tp=tp, cpu_only=cpu_only,
                                   spread=(mode == "spread"),
                                   proportional=(mode == "proportional"),
                                   gpu_spread=(mode == "all-gpu"),
                                   moe_offload=moe_offload, force=force, pin_host=node,
                                   kv_quant=kv_quant, kv_offload=kv_offload,
                                   kv_slots=kv_slots,
                                   default_temp=default_temp, default_min_p=default_min_p,
                                   requested_by=_req_ip,
                                   draft_gpu=draft_gpu, draft_margin_gb=_draft_margin,
                                   t2i_offload=t2i_offload)
            _modelbl = ("pin:%s/%s" % (node, "cpu" if cpu_only else "gpu")) if node else \
                       (((("tp%d-cpu" % tp) if cpu_only else ("tp%d" % tp)) if tp > 1 else mode))
            return JSONResponse({"ok": True, "model": lm.friendly, "ctx": lm.ctx,
                                 "mode": _modelbl, "quant": quant,
                                 "warnings": getattr(lm, "load_warnings", []),   # #76 guardrail
                                 "stages": [s.to_dict() for s in lm.plan.stages]})
        except Exception as exc:
            # (engine.load()'s finally already popped this load's progress card; nothing to clear here.)
            engine._last_load_failure = time.time()   # arm the self-update cool-down (anti-churn)
            # A failed load leaves no resident model; surface WHY on the dashboard so the
            # operator isn't left wondering why the model never appeared (or why an in-flight
            # big-MoE load died — e.g. a node OOM mid-load).
            log_activity(f"load {model}: FAILED — {exc}")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.post("/model_config")
    async def model_config(model: str, temperature: Optional[str] = None,
                           min_p: Optional[str] = None, top_p: Optional[str] = None,
                           top_k: Optional[str] = None,
                           repeat_penalty: Optional[str] = None,
                           repeat_last_n: Optional[str] = None,
                           presence_penalty: Optional[str] = None,
                           frequency_penalty: Optional[str] = None,
                           seed: Optional[str] = None,
                           num_predict: Optional[str] = None) -> JSONResponse:
        """#runtime-config: change a LOADED model's runtime-adjustable sampling defaults IN PLACE —
        no reload; the very next request picks them up (the sampler reads them off the LoadedModel
        per request). temperature (0-2) + min_p (0-1) live as dedicated fields (they predate the
        dict); the #runtime-knobs family — top_p / top_k / repeat_penalty / repeat_last_n /
        presence_penalty / frequency_penalty / seed / num_predict — lives in lm.sampling_defaults.
        Param absent = leave unchanged; empty string = CLEAR back to unset. Applied to every
        replica of the base so data-parallel copies stay consistent. Load-time-only knobs
        (quant/ctx/kv_quant/kv_offload/placement) are rejected by omission — they need a
        reload/reconfigure."""
        try:
            friendly = resolve_model_name(model)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        lms = engine.replicas_of(friendly)
        if not lms:
            return JSONResponse({"ok": False, "error": f"'{friendly}' is not loaded "
                                 "(runtime settings live on the loaded model — load it first)"},
                                status_code=409)

        def _parse(val, lo, hi, label, as_int=False):
            if val is None:
                return False, None       # absent -> unchanged
            if not str(val).strip():
                return True, None        # "" -> clear to unset
            f = int(val) if as_int else float(val)   # ValueError -> caught below
            if not (lo <= f <= hi):
                raise ValueError(f"{label} out of range ({lo} - {hi})")
            return True, f
        try:
            set_t, new_t = _parse(temperature, 0.0, 2.0, "temperature")
            set_m, new_m = _parse(min_p, 0.0, 1.0, "min_p")
            # #runtime-knobs: the dict family. Ranges: top_p (0,1]; top_k 0-1000 (0=off);
            # repeat_penalty 0.5-2 (llama.cpp multiplicative; 1=off); repeat_last_n -1-32768
            # (-1=whole ctx, 0=off, default 64); presence/frequency -2-2 (OpenAI additive);
            # seed 0-2^53-1 (the stored default round-trips JSON -> JS float64 -> dashboard
            # Apply, so it must stay in float64-exact range or the panel re-sends a rounded
            # value; per-REQUEST seeds go up to int64 max); num_predict 1-131072 (fills a
            # request that sends none).
            knobs = {}
            for key, val, lo, hi, as_int in (
                    ("top_p", top_p, 0.01, 1.0, False),
                    ("top_k", top_k, 0, 1000, True),
                    ("repeat_penalty", repeat_penalty, 0.5, 2.0, False),
                    ("repeat_last_n", repeat_last_n, -1, 32768, True),
                    ("presence_penalty", presence_penalty, -2.0, 2.0, False),
                    ("frequency_penalty", frequency_penalty, -2.0, 2.0, False),
                    ("seed", seed, 0, 2**53 - 1, True),
                    ("num_predict", num_predict, 1, 131072, True)):
                was_set, new = _parse(val, lo, hi, key, as_int=as_int)
                if was_set:
                    knobs[key] = new     # None = clear this key
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        for lm in lms:
            if set_t:
                lm.default_temperature = new_t
            if set_m:
                lm.default_min_p = new_m
            if knobs:
                sd = getattr(lm, "sampling_defaults", None)
                if sd is None:
                    sd = lm.sampling_defaults = {}
                for key, new in knobs.items():
                    if new is None:
                        sd.pop(key, None)
                    else:
                        sd[key] = new
        _sd0 = getattr(lms[0], "sampling_defaults", None) or {}
        # one merged view of every set default (unset keys omitted) — what the dashboard shows
        defaults = {k: v for k, v in {"temperature": lms[0].default_temperature,
                                      "min_p": lms[0].default_min_p, **_sd0}.items()
                    if v is not None}
        log_activity(f"{_ollama_name(friendly)}: runtime settings -> "
                     + (" ".join(f"{k}={v}" for k, v in defaults.items()) or "all unset")
                     + (f" ({len(lms)} replicas)" if len(lms) > 1 else ""))
        return JSONResponse({"ok": True, "model": friendly,
                             "def_temperature": lms[0].default_temperature,
                             "def_min_p": lms[0].default_min_p,
                             "defaults": defaults,
                             "replicas": len(lms)})

    @app.post("/terminate")        # #connections: kill EVERY in-flight request from one client
    async def terminate(ip: str) -> JSONResponse:
        """Cancel all of a client's in-flight requests (the Connections panel's Terminate
        button). Reuses /cancel's mechanics per request: flag + task-cancel + slot release.
        HTTP keep-alive sockets close on their own once their request dies; the accounting
        row stays (history), it just goes idle."""
        hits = [r for r in list(INFLIGHT.values()) if r.get("ip") == ip]
        for rec in hits:
            rec["cancel"] = True
            t = rec.get("task")
            if t is not None and not t.done():
                with contextlib.suppress(Exception):
                    t.cancel()
            _inflight_release(rec)
        log_activity(f"terminated client {ip}: {len(hits)} in-flight request(s) cancelled")
        return JSONResponse({"ok": True, "ip": ip, "cancelled": len(hits)})

    @app.post("/cancel")           # dashboard: disconnect/kill one in-flight request (#48)
    async def cancel(id: int) -> JSONResponse:
        rec = INFLIGHT.get(id)
        if rec is None:
            return JSONResponse({"ok": False, "error": f"no in-flight request id={id}"},
                                status_code=404)
        rec["cancel"] = True
        t = rec.get("task")
        if t is not None and not t.done():
            with contextlib.suppress(Exception):
                t.cancel()        # aborts _prepare/load/generate for this request (frees a wedge)
        _inflight_release(rec)
        log_activity(f"cancelled request id={id} ({rec.get('model')}, {rec.get('ip')})")
        return JSONResponse({"ok": True, "cancelled": id,
                             "model": rec.get("model"), "ip": rec.get("ip")})

    @app.post("/cancel_load")      # #stuck-load-override: kill a wedged in-flight MODEL LOAD (0%-forever)
    async def cancel_load(model: str = "") -> JSONResponse:
        """Cancel an in-flight (possibly wedged) model LOAD — the manual escape hatch for a load stuck
        at 0%. model='' cancels EVERY in-flight load. Cancelling the load task frees any partial shards
        it already built (the load's CancelledError cleanup), emptying it out so a fresh load can run."""
        friendly = ""
        if model:
            try:
                friendly = resolve_model_name(model)
            except ValueError:
                friendly = model
        cancelled = []
        for rk, t in list(engine._loading_tasks.items()):
            base = rk.split("#", 1)[0]
            if friendly and rk != friendly and base != friendly:
                continue
            if t is not None and not t.done():
                with contextlib.suppress(Exception):
                    t.cancel()
                cancelled.append(rk)
        if not cancelled:
            return JSONResponse({"ok": False, "error": "no in-flight load"
                                 + (f" for '{model}'" if model else "")}, status_code=404)
        log_activity(f"cancelled in-flight load(s): {', '.join(cancelled)}")
        return JSONResponse({"ok": True, "cancelled": cancelled})

    @app.post("/unload")
    async def unload(model: str = "", owner: str = "") -> JSONResponse:
        # No model -> unload everything; model=X -> evict just that one (keep the rest).
        # #unified-fleet: a model WE don't hold but a peer does is unloaded ON that peer, so the
        # Unload button works on every row the dashboard shows. Deliberately per-model only —
        # blanket "unload everything" stays scoped to this controller's own models (reaching across
        # and tearing down a peer's whole fleet from our Unload-all is never what was meant).
        if model:
            _pm = _peers_mod()
            _resolves_here = False
            try:
                _resolves_here = resolve_model_name(model) in engine.models
            except Exception:   # noqa: BLE001 — unknown name here; a peer may still know it
                pass
            if _pm is not None and not _resolves_here:
                _peer = _pm.find_peer(owner) if owner else _pm.peer_for_model(model)
                if _peer is not None:
                    _lbl = _pm.peer_label(_peer)
                    log_activity(f"unload {model}: federating to {_lbl}")
                    try:
                        _r = await asyncio.to_thread(
                            _pm.http_post_json,
                            f"{_pm.peer_base(_peer)}/unload?model={urllib.parse.quote(model)}", 120.0)
                    except Exception as _exc:   # noqa: BLE001
                        return JSONResponse({"ok": False, "federated_to": _lbl,
                                             "error": f"peer controller {_lbl} unreachable: {_exc}"},
                                            status_code=502)
                    await _pm.kick(_peer)
                    return JSONResponse({**_r, "federated_to": _lbl},
                                        status_code=(200 if _r.get("ok") else 400))
            try:
                friendly = resolve_model_name(model)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            await engine.unload(friendly)   # per-model: allowed any time, even during another's load
            return JSONResponse({"ok": True, "unloaded": [friendly]})
        # Blanket "unload everything" drops ALL shards on every worker — incl. any in-flight load's
        # half-built ones. engine.unload(None) decides UNDER self.lock (atomic with a load's card
        # registration) and raises LoadInProgressError if a load is in flight — no HTTP-layer TOCTOU.
        names = list(engine.models.keys())   # snapshot before the full teardown
        try:
            await engine.unload()
        except LoadInProgressError as exc:
            return JSONResponse({"ok": False, "error": "a load is in progress — wait for it, or unload a "
                                 "specific model (model=NAME); unload-all is blocked mid-load",
                                 "loading": list(exc.args[0]) if exc.args else []}, status_code=409)
        return JSONResponse({"ok": True, "unloaded": names})

    @app.post("/load_faster")
    async def load_faster(model: str) -> JSONResponse:
        # #load-faster: one-click "upgrade this resident model to a faster placement" (dashboard badge —
        # only shown when engine._upgrade_for detected one). DRAINS the in-flight reply (up to ~2 min,
        # then forces) then HITLESSLY re-places VRAM-first / fewest-nodes, preserving full config and
        # rolling back to a working copy on failure (never left evicted). No confirm — the #juggler
        # barrier makes it hitless (parked clients ride the swap). 409 if nothing faster fits right now.
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        res = await engine.load_faster(friendly)
        return JSONResponse(res, status_code=(200 if res.get("ok") else 409))

    @app.post("/reconfigure")
    async def reconfigure(model: str, tp: int = 1, ctx: int = 0, quant: str = "keep",
                          mode: str = "auto", cpu_only: bool = False) -> JSONResponse:
        # #88 managed reload: switch a RESIDENT model to/from tensor-parallel (or change TP width /
        # ctx / quant) in ONE call, rolling back to a working pipeline copy on failure. ctx=0 or
        # quant='keep' INHERIT the resident copy's values (a pure layout switch keeps them). tp>=2 ->
        # tensor-parallel (cpu_only routes the mesh to RAM); tp<=1 -> pipeline (mode picks the strategy).
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        lm = engine.models.get(friendly)
        if lm is None:
            return JSONResponse({"ok": False, "error": f"'{friendly}' is not resident — load it first"},
                                status_code=404)
        if getattr(lm, "active", 0) > 0:   # never tear a model down mid-generate
            return JSONResponse({"ok": False, "error": f"'{friendly}' is busy ({lm.active} active "
                                 f"request(s)) — retry when idle"}, status_code=409)
        if quant not in ("keep", "none", "int8", "int4", "int2"):
            return JSONResponse({"ok": False, "error": f"bad quant '{quant}' (keep|none|int8|int4|int2)"},
                                status_code=400)
        new_ctx = ctx if (ctx and ctx > 0) else lm.ctx
        new_quant = (lm.quant or "none") if quant == "keep" else quant
        from_tp = getattr(lm, "tp_size", 1)
        from_ctx, from_quant = lm.ctx, lm.quant
        if tp == from_tp and new_ctx == lm.ctx and new_quant == (lm.quant or "none"):
            return JSONResponse({"ok": True, "model": friendly, "noop": True,
                                 "from": {"tp": from_tp, "ctx": from_ctx, "quant": from_quant},
                                 "to": {"tp": from_tp, "ctx": new_ctx, "quant": new_quant},
                                 "basis": getattr(lm, "plan_basis", ""),
                                 "stages": [s.hostname for s in lm.plan.stages]})
        # Pre-validate the TP width (same guards as _load_tp_locked) BEFORE evicting, so an obviously
        # invalid width fails clean (400) instead of an evict-then-rollback churn.
        if tp > 1:
            spec = resolve_spec(friendly)
            nh, nkv = spec.num_heads, spec.num_kv_heads
            ng = max(1, spec.intermediate_size // 128)
            ok_geom = (nh % tp == 0) and ((tp <= nkv and nkv % tp == 0) or (tp > nkv and tp % nkv == 0)) and tp <= ng
            if not ok_geom:
                return JSONResponse({"ok": False, "error":
                    f"tp={tp} invalid for {friendly}: needs num_heads({nh})%tp==0, "
                    f"(nkv({nkv})%tp==0 if tp<=nkv else tp%nkv==0), and tp<=FFN_groups({ng})"},
                    status_code=400)
        cons, pv = LOAD_MODES.get(mode, LOAD_MODES["auto"])
        try:
            new = await engine.reconfigure(friendly, tp=tp, ctx=new_ctx, quant=new_quant,
                                           consolidate=cons, prefer_vram=pv, cpu_only=cpu_only)
            return JSONResponse({"ok": True, "model": new.friendly,
                                 "from": {"tp": from_tp, "ctx": from_ctx, "quant": from_quant},
                                 "to": {"tp": getattr(new, "tp_size", 1), "ctx": new.ctx,
                                        "quant": new.quant,
                                        "mode": (("tp%d-cpu" % tp) if cpu_only else ("tp%d" % tp))
                                                if tp > 1 else mode},
                                 "basis": getattr(new, "plan_basis", ""),
                                 "stages": [s.hostname for s in new.plan.stages]})
        except Exception as exc:
            # (the internal engine.load()'s finally already cleared any progress card.)
            log_activity(f"reconfigure {model}: FAILED — {exc}")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.post("/restart")
    async def restart(request: Request, workers: int = 1, force: bool = False,
                      controller: int = 1) -> JSONResponse:
        # RESTART, three shapes (all EXPLICIT commands, never idle-gated; supervisors relaunch
        # on exit 42 — server.bat / client.bat / systemd Restart=always):
        #   workers=0            -> CONTROLLER ONLY. #adopt: workers on adopt-capable code KEEP
        #                           their loaded shards and re-register with an inventory; the
        #                           relaunched controller re-ADOPTS the resident models instead
        #                           of re-streaming them — a hitless controller bounce.
        #   workers=1&controller=0 -> WORKERS ONLY ("Restart fleet"): every worker process
        #                           relaunches; the controller stays up. Resident models drop
        #                           (a worker restart wipes its shards) and the existing
        #                           link-death invalidation cleans up their controller state;
        #                           they re-load on demand / by pins.
        #   workers=1 (default)  -> RESTART ALL: workers + controller — the full reset that
        #                           clears stale worker state, wedged loads and allocator VRAM.
        # GENTLER (#100): refuse while a load/compile/render is IN PROGRESS; force=1 overrides.
        _renders = getattr(engine, "_t2i_pending", None) or {}
        if (engine.loadings or engine.compiling or _renders) and not force:
            return JSONResponse({"ok": False, "status": "load_in_progress",
                                 "reason": "a model load/compile/render is in progress; pass force=1 "
                                           "to restart anyway (aborts it)",
                                 "loading": [c.get("model") for c in engine.loadings.values()],
                                 "compiling": [c.get("model") for c in engine.compiling.values()],
                                 "rendering": len(_renders)},
                                status_code=409)
        if not workers and not controller:
            return JSONResponse({"ok": False, "error": "nothing to restart "
                                 "(workers=0&controller=0)"}, status_code=400)
        signaled = []
        if workers:
            for nid, link in list(engine.links.items()):
                with contextlib.suppress(Exception):
                    await link.send({"type": "restart"})
                    signaled.append(nid)
        who = _client_ip(request)   # who triggered it (dashboard browser / curl host) for the log
        _shape = ("controller only" if not workers
                  else ("workers only" if not controller else "workers + controller"))
        msg = (f"RESTART ({_shape}) requested by {who} -> {len(signaled)} worker(s)"
               f"{' + controller (exit 42)' if controller else ''}")
        log_activity(msg)
        if controller:
            print(f"[restart] {msg}; controller exiting(42) in 2s")
            async def _bye():
                await asyncio.sleep(2.0)   # let worker frames flush + this HTTP response return
                os._exit(42)               # server.bat supervisor relaunches on the current code
            asyncio.create_task(_bye())
        else:
            print(f"[restart] {msg}; controller staying up")
        return JSONResponse({"ok": True, "restarting_controller": bool(controller),
                             "requested_by": who,
                             "workers_signaled": signaled, "worker_count": len(signaled)})

    @app.post("/restart_node")
    async def restart_node(request: Request, node: str = "") -> JSONResponse:
        # #node-restart: restart ONE worker process (exit 42 -> its supervisor relaunches it) —
        # the per-node "fresh start" that clears whatever VRAM/RAM the process is holding,
        # without touching the controller or the rest of the fleet. Models with a stage on the
        # node DROP (a worker restart wipes its shards); the route invalidates them
        # SYNCHRONOUSLY (#restart-stale) AND frees their surviving stages on OTHER nodes, so
        # an idle model costs nothing anywhere afterwards (it re-auto-loads on demand). A model
        # that is IN USE (serving/queued now, or used in the last 10 min) is RECOVERED: once
        # the invalidation lands, a background task re-loads it with its original ctx/quant/KV
        # knobs — the planner re-places it onto whatever capacity is up (other nodes' GPU/CPU;
        # the restarted node itself usually rejoins in seconds and is a candidate again).
        if not node:
            return JSONResponse({"ok": False, "error": "node is required (hostname or node id)"},
                                status_code=400)
        nd = registry._nodes.get(node) \
            or next((n for n in registry.alive_sorted()
                     if n.hostname.lower() == node.strip().lower()), None)
        if nd is None:
            return JSONResponse({"ok": False, "error": f"unknown node '{node}'"}, status_code=404)
        link = engine.links.get(nd.node_id)
        if link is None:
            return JSONResponse({"ok": False, "error": f"no control link to {nd.hostname} "
                                 "(already down? it will relaunch on its own supervisor)"},
                                status_code=409)
        # #restart-stale: match models to the target node by PHYSICAL identity — current
        # node_id OR stage hostname — never node_id alone. stage_node_ids is frozen at
        # load/adopt time while every re-registration mints a fresh id (registry.add), so a
        # worker that re-registered behind the controller's back (silent half-open old
        # socket, a data_host flip defeating find_stale_dupes, or a mid-adopt/mid-load
        # window) leaves resident models keyed under a node id that no longer exists
        # ANYWHERE — and the old id-only scan here returned models_affected=[] while the
        # model's only stage sat on the box (observed live 2026-07-21 om3nbox/qwen3-30b:
        # the restart wiped the shard, the stale row kept dialing the relaunched worker,
        # every request 500'd "no shard for model_id=..." until a manual /unload).
        _host_l = (nd.hostname or "").strip().lower()

        def _on_target(m) -> bool:
            if nd.node_id in m.stage_node_ids:
                return True
            _stgs = getattr(getattr(m, "plan", None), "stages", None) or []
            return any((getattr(s, "hostname", "") or "").strip().lower() == _host_l
                       for s in _stgs)

        affected = [fr for fr, m in engine.models.items() if _on_target(m)]
        # Snapshot the recovery set BEFORE the restart (invalidation wipes engine.models).
        recover: list[tuple] = []
        _now = time.time()
        for fr, m in engine.models.items():
            if not _on_target(m):
                continue
            in_use = (m.active > 0 or m.queued > 0
                      or (_now - (m.last_used or 0)) < 600)
            if not in_use:
                continue   # idle -> invalidation frees every stage; re-auto-loads on demand
            if getattr(m, "replica_idx", 0):
                log_activity(f"node-restart: replica {fr} not auto-recovered (re-add replicas "
                             "manually)")
                continue
            # Normalize the resident quant back into a /load-able tier (media models carry
            # display strings: 'int4-e2' -> int4, 'bf16-off' -> offload mode).
            q = m.quant or "none"
            kw = {"kv_quant": (m.kv_quant or ""), "kv_offload": bool(m.kv_offload)}
            if q == "bf16-off":
                q, kw["t2i_offload"] = "none", True
            elif q.startswith("int4"):
                q = "int4"
            recover.append((fr, m.ctx, q, kw))
        who = _client_ip(request)
        try:
            await link.send({"type": "restart"})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"restart send failed: {exc!r}"},
                                status_code=502)
        log_activity(f"NODE RESTART: {nd.hostname} ({nd.node_id}) requested by {who}"
                     + (f" — drops {', '.join(affected)}" if affected else ""))
        # #restart-stale: SYNCHRONOUSLY invalidate every affected model NOW instead of
        # trusting link-death detection to do it. The restart command is accepted (send
        # succeeded), so these shards are fate-sealed — but the link-death scan is keyed by
        # the DYING link's node id and misses a row whose stage ids are stale (above), and a
        # fast supervisor relaunch can re-register before the old socket even errors. Missing
        # the cleanup left the controller serving a wiped shard. invalidate_model is the
        # standard teardown (fails the model's in-flight requests, frees surviving stages on
        # OTHER nodes, records the drop for the dashboard); when link-death DOES fire later
        # it finds the row already gone — a no-op, not a double-drop. In-use models are then
        # re-placed by the recovery task below (the route's contract).
        for fr in affected:
            engine.invalidate_model(fr, f"node restart: {nd.hostname} ({nd.node_id}) — "
                                        "its shards are wiped by the restart")
        if recover:
            async def _recover():
                # #restart-stale: the route already invalidated the affected models
                # synchronously, so this loop normally exits on its first tick — it stays as
                # a belt (reloading while an old copy is somehow still registered would be
                # treated as a live-model reload).
                for _ in range(30):
                    await asyncio.sleep(1.0)
                    if all(fr not in engine.models for fr, _c, _q, _k in recover):
                        break
                await engine._await_free_refresh()   # plan against post-drop free numbers
                for fr, ctx, q, kw in recover:
                    if fr in engine.models:      # never dropped (or already re-loaded) — done
                        continue
                    try:
                        log_activity(f"node-restart recovery: {fr} was in use — re-placing it "
                                     f"onto the available fleet (was on {nd.hostname})")
                        await engine.load(fr, ctx, quant=q, **kw)
                    except Exception as exc:
                        log_activity(f"node-restart recovery: {fr} re-load FAILED ({exc!r}) — "
                                     "it will auto-load on the next request instead")
            asyncio.create_task(_recover())
        return JSONResponse({"ok": True, "node": nd.hostname, "node_id": nd.node_id,
                             "requested_by": who, "models_affected": affected,
                             "recovering": [fr for fr, _c, _q, _k in recover]})

    @app.post("/update")
    async def update_endpoint(request: Request, workers: int = 0, force: bool = False,
                              hitless: int = 0) -> JSONResponse:
        # FORCED UPDATE (dashboard 'Update' button / deploy API): pull the latest code from GitHub
        # and restart NOW — do NOT wait for idle. Mitigates the auto-load race: set engine.updating
        # so no request reloads a model mid-swap, UNLOAD all models, tell every worker to FREE its
        # RAM (and restart too if workers=1), then swap changed files + exit(42) -> supervisor
        # relaunches on the new code. (Plain /restart relaunches the CURRENT code; this updates first.)
        who = _client_ip(request)
        # #t2i: refuse while an image render is in flight — the render survives worker-side but its
        # result is ORPHANED (t2i_done hits the restarted controller's dead link; observed live: a
        # 12-min render finished into a broken pipe). Renders are minutes-bounded; wait or force=1.
        _renders = getattr(engine, "_t2i_pending", None) or {}
        if _renders and not force:
            _pg = getattr(engine, "_t2i_progress", None) or {}
            _steps = [f"{(_pg.get(r) or (0, 0))[0]}/{(_pg.get(r) or (0, 0))[1]}" for r in _renders]
            return JSONResponse({"ok": False, "status": "render_in_progress",
                                 "reason": "a text-to-image render is in flight (step "
                                           + ", ".join(_steps) + ") — updating now would orphan its "
                                           "result; retry after it completes or pass force=1",
                                 "rendering": len(_renders)}, status_code=409)
        # #hitless-update: pull the latest code and bounce the CONTROLLER ONLY, KEEPING every
        # resident model. Workers are left entirely alone, so they keep their loaded shards
        # (#adopt) and the relaunched controller re-ADOPTS them on the NEW code — no unload, no
        # re-stream, no re-quantize. This is the deploy path for CONTROLLER-side changes
        # (dashboard / routes / status / placement / serving / graphs / …). It is NOT for worker
        # code: a worker only runs new client.py/worker_*.py after a restart, which wipes its
        # shards — use a plain /update (or /restart?workers=1) for those. Same in-progress guard
        # as /restart: a controller bounce aborts an in-flight load/compile/render, so refuse
        # unless force=1. (Everything below this branch is the destructive full-unload path.)
        if hitless:
            if (engine.loadings or engine.compiling or _renders) and not force:
                return JSONResponse({"ok": False, "status": "load_in_progress",
                                     "reason": "a model load/compile/render is in progress; a "
                                               "hitless update bounces the controller and would "
                                               "abort it — pass force=1 to update anyway",
                                     "loading": [c.get("model") for c in engine.loadings.values()],
                                     "compiling": [c.get("model") for c in engine.compiling.values()],
                                     "rendering": len(_renders)}, status_code=409)
            engine.updating = True           # block auto-load during the brief swap (reset on relaunch)
            kept = list(engine.models.keys())
            msg = (f"HITLESS UPDATE by {who}: keeping {kept or 'no'} resident model(s), workers "
                   f"untouched -> swap controller code + controller-only restart (#adopt re-adopts)")
            log_activity(msg); print(f"[update] {msg}")
            async def _go_hitless():
                await asyncio.sleep(1.0)     # let this HTTP response flush
                with contextlib.suppress(Exception):   # applies changed files; exits(42) if any changed
                    await asyncio.to_thread(_self_update_check, "server.py", (lambda: True), True)
                os._exit(42)                 # nothing to swap (or already swapped) -> plain relaunch
            asyncio.create_task(_go_hitless())
            return JSONResponse({"ok": True, "updating": True, "hitless": True, "kept": kept,
                                 "requested_by": who})
        engine.updating = True               # block auto-load immediately (anti-reload-race)
        names = list(engine.models.keys())
        with contextlib.suppress(Exception):     # best-effort graceful unload (don't block on a
            # force=True: this is a deploy/restart — tear down even if a load is in flight (the process
            # is about to exit anyway), so the blanket teardown isn't refused by the in-load guard.
            await asyncio.wait_for(engine.unload(force=True), timeout=10)   # wedged in-flight load — exit anyway)
        freed = []
        for nid, link in list(engine.links.items()):
            with contextlib.suppress(Exception):
                await link.send({"type": "free_memory"})      # drop shards + gc + drop OS caches
                if workers:
                    # restart+update: the worker stages the newest files BEFORE its exit(42) so
                    # the supervisor relaunches it on the fresh code immediately (old workers
                    # ignore the extra key and restart on current code — their poll converges).
                    await link.send({"type": "restart", "update": True})
                else:
                    # #fleet-update: command an IMMEDIATE self-update check (apply files now;
                    # restart only on a VERSION bump — the same rule as the idle poll). This is
                    # what makes the "Update + Deploy" button fleet-wide immediate now that the
                    # automatic poll is 15 min. Old workers ignore the unknown message type.
                    await link.send({"type": "self_update"})
                freed.append(nid)
        msg = (f"FORCED UPDATE by {who}: unloaded {names or 'none'}, freed RAM + pushed update "
               f"to {len(freed)} worker(s){' + worker restart' if workers else ''} "
               f"-> swap code + restart")
        log_activity(msg); print(f"[update] {msg}")
        async def _go():
            await asyncio.sleep(1.5)   # let unload/free acks + this HTTP response flush
            with contextlib.suppress(Exception):   # force-swap; _self_update_check exits if changed
                await asyncio.to_thread(_self_update_check, "server.py", (lambda: True), True)
            os._exit(42)               # nothing to swap (or already swapped) -> plain relaunch
        asyncio.create_task(_go())
        return JSONResponse({"ok": True, "updating": True, "unloaded": names,
                             "workers_freed": len(freed), "worker_restart": bool(workers),
                             "requested_by": who})

    # code-split Inc 6: /shard_status /verify_shards /pack_result /pack_probe /compile_dist
    # /compile_shards + /weights /weights_tp /experts + the parked /mtp_probe /modelcode live
    # in routes_shards.py now (bodies VERBATIM).
