"""routes_lifecycle: the model-lifecycle routes relocated from server.py build_app
(m4c153 code-split): /load, /model_config, /terminate, /cancel, /cancel_load, /unload,
/reconfigure, /restart, /update. Route bodies are BYTE-IDENTICAL to the originals; their
module globals (engine, registry, _serve, build_status, JSONResponse ...) are injected at
startup by state.bind() -- see state.py. build_app() calls register(app) to attach them.
The shard-cache / packing / weight-serving routes moved on to routes_shards.py (code-split
Inc 6). Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def register(app):

    @app.post("/load")
    async def load(request: Request, model: str, ctx: int = 0, mode: str = "auto",
                   consolidate: bool = True, quant: str = "none", tp: int = 1,
                   replicas: int = 1, cpu_only: bool = False,
                   moe_offload: bool = False, force: bool = False,
                   node: str = "", kv_quant: str = "",
                   kv_offload: bool = False, temperature: str = "",
                   min_p: str = "", precompile: bool = True,
                   draft_gpu: bool = False, draft_margin_gb: str = "") -> JSONResponse:
        _req_ip = _client_ip(request)   # #connections: attribute this load to its requester
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
                                   default_temp=default_temp, default_min_p=default_min_p,
                                   requested_by=_req_ip,
                                   draft_gpu=draft_gpu, draft_margin_gb=_draft_margin)
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
    async def unload(model: str = "") -> JSONResponse:
        # No model -> unload everything; model=X -> evict just that one (keep the rest).
        if model:
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
    async def restart(request: Request, workers: int = 1, force: bool = False) -> JSONResponse:
        # FULL-FLEET RESTART: signal every connected worker to restart, then restart the controller.
        # UNLIKE the idle-gated self-update, this is an EXPLICIT command and is NOT idle-gated.
        # Supervisors relaunch on exit 42 (server.bat / client.bat / systemd Restart=always).
        # workers=0 -> restart the controller only. GENTLER (#100): refuse while a load is IN PROGRESS
        # — restarting mid-build drops a node from the load (and can leave it not cleanly rejoining);
        # pass force=1 to abort a wedged/doomed load anyway (the original escape-hatch behavior).
        # Blocks on in-flight LOADS or COMPILES (both lose work on restart); force=1 overrides.
        if (engine.loadings or engine.compiling) and not force:
            return JSONResponse({"ok": False, "status": "load_in_progress",
                                 "reason": "a model load/compile is in progress; pass force=1 to restart "
                                           "anyway (aborts it)",
                                 "loading": [c.get("model") for c in engine.loadings.values()],
                                 "compiling": [c.get("model") for c in engine.compiling.values()]},
                                status_code=409)
        signaled = []
        if workers:
            for nid, link in list(engine.links.items()):
                with contextlib.suppress(Exception):
                    await link.send({"type": "restart"})
                    signaled.append(nid)
        who = _client_ip(request)   # who triggered it (dashboard browser / curl host) for the log
        msg = (f"FLEET RESTART requested by {who} -> {len(signaled)} worker(s) + controller "
               f"(exit 42){' [controller only]' if not workers else ''}")
        log_activity(msg)
        print(f"[restart] {msg}; controller exiting(42) in 2s")
        async def _bye():
            await asyncio.sleep(2.0)   # let worker frames flush + this HTTP response return
            os._exit(42)               # server.bat supervisor relaunches on the current code
        asyncio.create_task(_bye())
        return JSONResponse({"ok": True, "restarting_controller": True, "requested_by": who,
                             "workers_signaled": signaled, "worker_count": len(signaled)})

    @app.post("/update")
    async def update_endpoint(request: Request, workers: int = 0) -> JSONResponse:
        # FORCED UPDATE (dashboard 'Update' button / deploy API): pull the latest code from GitHub
        # and restart NOW — do NOT wait for idle. Mitigates the auto-load race: set engine.updating
        # so no request reloads a model mid-swap, UNLOAD all models, tell every worker to FREE its
        # RAM (and restart too if workers=1), then swap changed files + exit(42) -> supervisor
        # relaunches on the new code. (Plain /restart relaunches the CURRENT code; this updates first.)
        who = _client_ip(request)
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
                    await link.send({"type": "restart"})      # full worker relaunch too
                freed.append(nid)
        msg = (f"FORCED UPDATE by {who}: unloaded {names or 'none'}, freed RAM on {len(freed)} "
               f"worker(s){' + worker restart' if workers else ''} -> swap code + restart")
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
