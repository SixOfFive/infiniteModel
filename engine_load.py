"""EngineLoadMixin: relocated Engine methods (m4c152 code-split). BODIES ARE BYTE-IDENTICAL
to the originals in server.py; their module globals (registry, log_activity, ModelSpec,
ENGINE_CONFIG …) are injected at startup by state.bind() — see state.py. Composed back
into the live class via ``class Engine(EngineLoadMixin, …)`` in server.py, so ``self.*`` resolves
across all mixins by MRO. Controller-only leaf module; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


class EngineLoadMixin:

    def _fit_ctx(self, spec, mems, ctx: int, consolidate: bool, prefer_vram: bool,
                 floor: int, spread: bool = False, proportional: bool = False,
                 gpu_spread: bool = False):
        """Binary-search the largest context in [floor, ctx] whose pipeline plan fits `mems`.
        Used when a load won't fit at the requested ctx and there's nothing to evict — trading
        context for fit instead of OOMing. Returns (ctx_used, plan); plan.ok is False only when
        not even `floor` tokens fit (weights alone exceed the pool)."""
        floor = min(floor, ctx)
        pf = plan_pipeline(spec, mems, floor, consolidate=consolidate, prefer_vram=prefer_vram,
                           spread=spread, proportional=proportional, gpu_spread=gpu_spread)
        if not pf.ok:
            return floor, pf                      # even the floor doesn't fit -> caller raises
        best_ctx, best_plan = floor, pf
        lo, hi = floor + 1, ctx
        while lo <= hi:
            mid = (lo + hi) // 2
            p = plan_pipeline(spec, mems, mid, consolidate=consolidate, prefer_vram=prefer_vram,
                              spread=spread, proportional=proportional, gpu_spread=gpu_spread)
            if p.ok:
                best_ctx, best_plan = mid, p
                lo = mid + 1
            else:
                hi = mid - 1
        aligned = max(floor, (best_ctx // 512) * 512)   # tidy number; a smaller ctx still fits
        if aligned != best_ctx:
            pa = plan_pipeline(spec, mems, aligned, consolidate=consolidate, prefer_vram=prefer_vram,
                               spread=spread, proportional=proportional, gpu_spread=gpu_spread)
            if pa.ok:
                return aligned, pa
        return best_ctx, best_plan

    async def ensure_loaded(self, friendly: str, ctx: int,
                            cpu_only: bool = False, auto_load: bool = False) -> LoadedModel:
        # If the model is resident we serve it as-is (ctx/cpu_only ignored for a live model — we
        # never reload a serving model). If it is NOT resident: an inference request now AUTO-LOADS
        # it with the default AUTO placement (GPU-VRAM-first, fewest nodes; ctx<=0 -> the model's
        # native training context), instead of failing — opt-in per call site (`auto_load=True` on
        # the serving paths only, NOT metadata like /api/show) and gated by ENGINE_CONFIG `auto_load`
        # (default on; set false via /config to restore the old explicit-load-only policy). A bad
        # model name is already rejected by resolve_model_name upstream, so this only loads a KNOWN
        # registered model. A load failure (capacity/etc.) propagates to the caller as the error.
        m = self.models.get(friendly)
        if m is not None:
            m.last_used = time.time()
            return m
        if self.updating:   # forced update in progress -> don't reload into a box being torn down
            raise ValueError(f"model '{friendly}' is not loaded — controller is updating, retry shortly")
        if auto_load and ENGINE_CONFIG.get("auto_load", True):
            # #autoload-smallest: an auto-loaded (requested-but-not-resident) model defaults to the
            # SMALLEST quant — int4 — so a request never streams the full bf16 just to serve it (int4
            # is ~1/4 the memory, fits more nodes, and serves PRE-PACKED when a shard cache exists).
            # Tunable via ENGINE_CONFIG `autoload_quant` (int4|int8|none). If int4/int8 fails for a
            # model the quantizer can't handle, fall back ONCE to bf16 so the request still succeeds
            # rather than erroring out — "int4 in almost all cases", bf16 for the rest. (CancelledError
            # is a BaseException, not Exception, so a client disconnect still aborts — never retried.)
            # #auto-defaults: an auto-load uses the SAME configured defaults as the dashboard's per-model
            # Load button — quant (int4), context (8k), and placement mode — so request-triggered loads
            # and click-loads behave identically. The request's own ctx (>0) still overrides the default.
            aq = str(ENGINE_CONFIG.get("autoload_quant", "int4") or "none")
            a_ctx = int(ENGINE_CONFIG.get("autoload_ctx", DEFAULT_CTX) or 0)
            use_ctx = ctx if (ctx and ctx > 0) else a_ctx
            a_mode = str(ENGINE_CONFIG.get("autoload_mode", "auto") or "auto")
            _cons, _pv = LOAD_MODES.get(a_mode, LOAD_MODES["auto"])
            _spread, _prop = (a_mode == "spread"), (a_mode == "proportional")
            _gpus = (a_mode == "all-gpu")
            log_activity(f"{friendly}: auto-load on request (not resident) -> mode={a_mode}, "
                         f"quant={aq}, ctx={use_ctx or 'train'}" + (" (CPU-only)" if cpu_only else ""))
            try:
                await self._precompile_int4(friendly, aq, 1)   # #cache-on-first-load: auto-load parity
                return await self.load(friendly, use_ctx, consolidate=_cons, prefer_vram=_pv,
                                       quant=aq, cpu_only=cpu_only, spread=_spread,
                                       proportional=_prop, gpu_spread=_gpus)
            except Exception as e:
                if aq != "none":
                    log_activity(f"{friendly}: auto-load at {aq} failed ({e!r}) -> retry at bf16")
                    return await self.load(friendly, use_ctx, consolidate=_cons, prefer_vram=_pv,
                                           quant="none", cpu_only=cpu_only, spread=_spread,
                                           proportional=_prop, gpu_spread=_gpus)
                raise
        raise ValueError(f"model '{friendly}' is not loaded — load it first")

    async def _precompile_int4(self, friendly: str, quant: str, tp: int) -> None:
        """#cache-on-first-load: for an int4 load with NO shard cache yet, BUILD it first (blocks
        until written) so THIS load — and every future load — serves the small pre-packed int4
        layers instead of streaming full bf16 and re-quantizing on the fly. No-op when an int4 cache
        already exists, when quant != int4, or for tp>1 (its dispatch path doesn't read the whole-
        layer cache). Reuses the /compile_shards SUBPROCESS (deprioritized, GIL-safe — an in-process
        compile would starve the event loop / drop live generations). Non-fatal: ANY failure falls
        through to the normal cold load. Shared by the /load route AND the auto-load path
        (ensure_loaded) so request-triggered loads compile-on-first-load identically to click-loads."""
        if not (quant == "int4" and tp <= 1):
            return
        try:
            import shard_compile as _sh   # code-split Inc 9: shard_cache_status moved
            import urllib.parse as _up
            _ctgt = MODELS[friendly][0] if friendly in MODELS else friendly
            _cdir = await asyncio.to_thread(_controller_model_dir, _ctgt)
            _cst = await asyncio.to_thread(_sh.shard_cache_status, _cdir) if _cdir else {}
            if _cdir and not (_cst.get("int4") or {}).get("ok"):
                log_activity(f"{_ollama_name(friendly)}: no int4 shard cache — building it now so this "
                             "and every future load serve pre-packed (first load is slower)…")
                _curl = (f"http://127.0.0.1:{ARGS.http_port}/compile_shards"
                         f"?model={_up.quote(friendly)}&quant=int4")

                def _build_cache():
                    import urllib.request as _u
                    with _u.urlopen(_u.Request(_curl, method="POST"), timeout=10800) as _r:
                        return _r.read()

                await asyncio.to_thread(_build_cache)
        except Exception as _ce:
            log_activity(f"{_ollama_name(friendly)}: pre-load cache build skipped ({_ce!r}) — cold load")

    def _reserved_bytes(self, exclude_key: Optional[str] = None) -> tuple[dict, dict]:
        """Sum the in-flight load reservations (RAM, VRAM) per node, EXCLUDING `exclude_key` (a
        load never reserves against itself). Returns (ram_by_node, vram_by_node) in bytes. Used by
        every planner so a load that's already reserved + streaming is subtracted from a concurrent
        load's budget -> no over-provision."""
        ram: dict[str, int] = {}
        vram: dict[str, int] = {}
        for k, res in self._reservations.items():
            if k == exclude_key:
                continue
            for nid, b in res.items():
                ram[nid] = ram.get(nid, 0) + int(b.get("ram", 0))
                vram[nid] = vram.get(nid, 0) + int(b.get("vram", 0))
        return ram, vram

    async def load(self, friendly: str, ctx: int, consolidate: bool = True,
                   prefer_vram: bool = True, quant: str = "none", tp: int = 1,
                   cpu_only: bool = False, reg_key: Optional[str] = None,
                   exclude_nodes: Optional[set] = None, replica_idx: int = 0,
                   spread: bool = False, proportional: bool = False,
                   force: bool = False, moe_offload: bool = False,
                   gpu_spread: bool = False, pin_host: str = "",
                   kv_quant: str = "", kv_offload: bool = False,
                   default_temp: Optional[float] = None,
                   default_min_p: Optional[float] = None,
                   requested_by: str = "",
                   draft_gpu: bool = False,
                   draft_margin_gb: float = 4.0) -> LoadedModel:
        # Thin wrapper over _load_impl that owns the cleanup of this load's reservation + progress card
        # + in-flight future. CRITICAL (review #parallel-load): only the call that actually CLAIMED the
        # load slot for reg_key cleans up — `_own["v"]` is set True by _load_impl iff THIS call
        # registered the card (became the owner). A same-key dedup-WAITER (which just awaits the owner's
        # future and returns the resident copy) — or one that gets CANCELLED while waiting — must NOT
        # pop the owner's live reservation/card/future (doing so dropped the reservation -> over-provision,
        # cleared the card -> unload-all could nuke the loading model, and resolved the owner's future
        # early -> duplicate load). pop is synchronous (no await between _load_impl returning and the
        # pops) so no concurrent op observes both the reservation AND the now-resident model.
        rk = reg_key or friendly
        _own = {"v": False}
        try:
            return await self._load_impl(friendly, ctx, consolidate=consolidate,
                                         prefer_vram=prefer_vram, quant=quant, tp=tp,
                                         cpu_only=cpu_only, reg_key=reg_key,
                                         exclude_nodes=exclude_nodes, replica_idx=replica_idx,
                                         spread=spread, proportional=proportional, force=force,
                                         moe_offload=moe_offload, gpu_spread=gpu_spread,
                                         pin_host=pin_host, kv_quant=kv_quant,
                                         kv_offload=kv_offload, default_temp=default_temp,
                                         default_min_p=default_min_p,
                                         requested_by=requested_by,
                                         draft_gpu=draft_gpu, draft_margin_gb=draft_margin_gb,
                                         _own=_own)
        finally:
            if _own["v"]:
                self._reservations.pop(rk, None)
                self.loadings.pop(rk, None)
                self._loading_tasks.pop(rk, None)   # owner's task done -> drop the cancel handle
                _f = self._loading_futures.pop(rk, None)   # wake any same-model requests queued on us
                if _f is not None and not _f.done():
                    _f.set_result(self.models.get(rk))

    async def _load_impl(self, friendly: str, ctx: int, consolidate: bool = True,
                   prefer_vram: bool = True, quant: str = "none", tp: int = 1,
                   cpu_only: bool = False, reg_key: Optional[str] = None,
                   exclude_nodes: Optional[set] = None, replica_idx: int = 0,
                   spread: bool = False, proportional: bool = False,
                   force: bool = False, moe_offload: bool = False,
                   gpu_spread: bool = False, pin_host: str = "",
                   kv_quant: str = "", kv_offload: bool = False,
                   default_temp: Optional[float] = None,
                   default_min_p: Optional[float] = None,
                   requested_by: str = "",
                   draft_gpu: bool = False,
                   draft_margin_gb: float = 4.0,
                   _own: Optional[dict] = None) -> LoadedModel:
        # self.lock guards atomic engine-state mutation; it is DROPPED around the streaming gather so a
        # 2nd load AND an unload can run meanwhile. Concurrent loads stay memory-safe via the
        # reservation ledger; planning is serialized by this lock, only the streaming overlaps.
        # Manual acquire + `_held` (not `async with`) so a CancelledError delivered at a re-acquire
        # can't make the block release a lock this task doesn't hold — with _load_lock gone a contender
        # can hold self.lock during our gather, and `async with` __aexit__ would then release THEIR lock
        # (asyncio.Lock has no owner) -> desync. _held gates every release to "only if we hold it".
        await self.lock.acquire()
        _held = True
        try:
            # `friendly` stays the user-facing/base name (spec, target, tokenizer, draft all
            # resolve from it); `reg_key` is the registry key actually stored. For a single
            # model they're equal; for a replica (#39) reg_key is "base#i" and exclude_nodes
            # holds the nodes its siblings already occupy (disjoint placement).
            reg_key = reg_key or friendly
            # FORCE OVERRIDE (#stuck-load-override): a force load while ANOTHER load of this key is in
            # flight means "that one is wedged — kill it and restart". CANCEL the in-flight owner's task
            # and AWAIT its unwind (the cancelled owner's finally frees its partial shards + reservation
            # + card + future), so we then proceed as a clean fresh load (becoming the new owner below).
            # Without this, force just raced a 2nd load onto the same nodes. We drop the lock while the
            # cancelled load tears down (it needs the lock to free shards), then re-acquire — same
            # pattern as the same-model dedup wait. force=False never does this (it queues instead).
            if force:
                _old = self._loading_tasks.get(reg_key)
                if _old is not None and _old is not asyncio.current_task() and not _old.done():
                    log_activity(f"{friendly}: force override — cancelling the wedged in-flight load "
                                 f"and restarting")
                    _old.cancel()
                    self.lock.release()
                    _held = False
                    try:
                        with contextlib.suppress(BaseException):
                            await _old
                    finally:
                        await self.lock.acquire()
                        _held = True
            # Register the progress card IMMEDIATELY — before any interleavable await — so "is a load
            # in progress?" is answerable under self.lock for the WHOLE load (the unload-all teardown
            # checks self.loadings to refuse mid-load: the TOCTOU fix). Enriched with real shard/stage
            # counts at dispatch; cleared by load()'s finally. ONLY register if no card exists for this
            # key — a same-key dedup-waiter must NOT clobber the in-flight owner's rich card (which holds
            # node_ids for reaper grace, real progress, started). The call that registers becomes the
            # OWNER (_own["v"]=True) and is the sole one that cleans up reservation/card/future.
            if reg_key not in self.loadings:
                self.loadings[reg_key] = {
                    "model": friendly, "display_model": _ollama_name(friendly),
                    "target": MODELS[friendly][0] if friendly in MODELS else friendly,
                    "ready": 0, "total": 0, "stages_total": 0, "stages_ready": 0,
                    "basis": "planning…", "warnings": [], "started": time.time(),
                    # #connections: which client asked for this load ("" = internal/auto)
                    "requested_by": requested_by or ""}
                if _own is not None:
                    _own["v"] = True
                    self._loading_tasks[reg_key] = asyncio.current_task()   # cancel handle for force override
            from transformers import AutoTokenizer
            spec = resolve_spec(friendly)
            if spec is None:
                raise ValueError(f"unknown model '{friendly}'")
            target_id = MODELS[friendly][0] if friendly in MODELS else friendly
            # ENCODER / sentence-embedding model: a whole-model single-node load (no pipeline/TP/KV
            # planning, no lm_head). Branch BEFORE plan_pipeline; the slim loader keys self.models
            # the same way this path does (reg_key).
            if getattr(spec, "is_embedding", False):
                return await self._load_embedding_locked(friendly, target_id, spec, reg_key,
                                                         replica_idx=replica_idx)
            # (previous pipeline connections are torn down by _unload_locked below,
            # which closes every loaded model's stage0_writer before we re-plan.)
            # controller is the model source: download the full model once so the
            # /weights endpoint can serve each worker only its slice.
            model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
            # EARLY ARCH GUARD (#6/#127): reject an exotic/unbuildable architecture HERE — before any
            # stage is dispatched — so it fails with a legible "unsupported architecture 'X'" instead
            # of a cryptic meta-tensor crash deep in the streamed worker build. Conservative: runs the
            # SAME meta skeleton the worker builds; trust_remote_code models pass through (the worker
            # fetches their .py via /modelcode). No-op if there's no readable config yet.
            await asyncio.to_thread(validate_arch_supported, model_dir)
            # ctx<=0 => use the model's native training context (config.json).
            ctx_was_auto = ctx <= 0          # #76: only auto-cap an AUTO ctx, never an explicit one
            if ctx <= 0:
                ctx = _train_ctx_from_dir(model_dir, spec)
                print(f"[load] ctx=auto -> training context {ctx}")
            # Idempotent re-load: a duplicate /load for an ALREADY-resident model at the SAME
            # ctx+quant — e.g. an accidental dashboard double-click — is a NO-OP that returns the
            # live copy, instead of evicting + re-streaming it (which showed up as a spurious
            # "reload"). A DIFFERENT ctx or quant still reloads (that's how you change them).
            _resident = self.models.get(reg_key)
            if (not force and _resident is not None and _resident.ctx == ctx
                    and (_resident.quant or "none") == (quant or "none")):
                log_activity(f"load {friendly}: already resident @ ctx={ctx}"
                             + (f" {quant}" if quant and quant != "none" else "")
                             + " — duplicate load ignored (no-op)")
                _resident.last_used = time.time()
                return _resident
            # SAME-MODEL concurrent load (parallel-load): a load for this key is already in flight
            # (planned + streaming with the lock released). Don't double-load — QUEUE on it: wait for
            # it to finish, then serve the now-resident copy. force= skips this (a reconfigure intends
            # to reload). If the in-flight load FAILED, fall through and attempt the load ourselves.
            if not force:
                _inflight = self._loading_futures.get(reg_key)
                if _inflight is not None:
                    log_activity(f"{friendly}: already loading — queueing this request on the "
                                 f"in-flight load")
                    self.lock.release()
                    _held = False
                    try:
                        with contextlib.suppress(Exception):
                            await _inflight
                    finally:
                        await self.lock.acquire()   # if cancelled here, _held stays False -> outer
                        _held = True                # finally won't release a lock we don't hold
                    _m = self.models.get(reg_key)
                    if _m is not None:
                        _m.last_used = time.time()
                        return _m
            # Claim this key so concurrent same-model requests queue on us (resolved by load()).
            self._loading_futures[reg_key] = asyncio.get_event_loop().create_future()
            # Plan with REAL per-layer weight sizes from the safetensors headers, so
            # MoE (and any non-dense arch) is sized correctly rather than via the
            # dense formula. No-op (returns spec) if the files can't be measured.
            spec = await asyncio.to_thread(spec_with_measurements, spec, model_dir)
            # Build-transient reserve (m4am): the worker streams each layer in as bf16 and
            # quantizes it in RAM, so a node needs ~2x ONE layer's bf16 FREE (fetch blob +
            # deserialize) on top of its resident shard, or it OOMs mid-build (a single MiniMax
            # MoE layer is ~7 GB bf16 -> ~14 GB transient, more than a tiny node's whole RAM).
            # Computed from the BF16 spec (before for_quant); reserved per node below so nodes
            # too small for the transient get 0 layers. Scales with the model: tiny for small ones.
            bf16_layer_gb = ((spec.total_weight_bytes / max(1, spec.num_layers)) / GB) if spec else 0.0
            stream_load = (tp <= 1)   # streaming build path -> incurs the per-layer bf16 transient
            # Size the plan for the quantized footprint so the planner packs int8 layers
            # into VRAM (weights ~halve; KV is unchanged). Workers get quant in the load
            # message and quantize their slice after the bf16 mmap-load.
            total_bf16_bytes = spec.total_weight_bytes if spec else 0   # bf16 READ volume (PRE-quant) for the load timeout (#100)
            spec = spec.for_quant(quant)
            # quant=none loads the model's NATIVE precision: an fp32 checkpoint loads as fp32 instead
            # of being silently downcast to bf16 (the planner already sized the measured fp32 bytes,
            # so the reservation matches). bf16/fp16 sources still load bf16; int8/int4 load bf16 then
            # quantize. (Multi-stage fp32 carries fp32 activations over the same dtype-agnostic transport.)
            load_dtype = ("float32" if (quant == "none" and (spec.src_dtype or "") == "F32")
                          else "bfloat16")
            # #172 TurboQuant KV: empty -> inherit the global ENGINE_CONFIG default; normalize+validate
            # here so EVERY caller (route, auto-load, replicate, reconfigure) gets the configured default.
            kv_quant = (kv_quant or str(ENGINE_CONFIG.get("kv_quant", "none") or "none")).strip().lower()
            if kv_quant not in ("none", "turbo2", "turbo3", "turbo4"):
                kv_quant = "none"
            # #kv-offload: inherit the global default when the caller didn't ask; exclusive with
            # kv_quant (route validates; auto-load paths land here too, so re-guard).
            kv_offload = bool(kv_offload or ENGINE_CONFIG.get("kv_offload", False))
            if kv_offload and kv_quant != "none":
                kv_offload = False
            await self.ensure_data_listener()
            log_activity(f"load {friendly}: planning (ctx={ctx}, quant={quant}"
                         + (f", kv_quant={kv_quant}" if kv_quant != "none" else "")
                         + (", KV-OFFLOAD (KV cache in system RAM)" if kv_offload else "")
                         + (f", default_temp={default_temp}" if default_temp is not None else "")
                         + (f", default_min_p={default_min_p}" if default_min_p is not None else "")
                         + (f", tp={tp}" if tp > 1 else "")
                         + (", CPU-ONLY (RAM, no VRAM)" if cpu_only else "") + ")")

            # AUTO-TRIGGER TP (#87 D): a model whose weights SUBSTANTIALLY exceed the GPU VRAM pool
            # would run mostly on CPU as a pipeline (sequential, slow). CPU tensor-parallelism splits
            # every layer across stable Linux nodes and aggregates their RAM bandwidth -> far faster
            # per token. Auto-pick a TP width and route there — no manual tp= switch. Conservative:
            # only the DEFAULT auto mode (consolidate+prefer_vram, no explicit tp/cpu_only/spread/
            # proportional), only when weights > auto_tp_ratio x GPU pool (else pipeline keeps the GPU),
            # gated by ENGINE_CONFIG auto_tp; any failure falls through to the pipeline path below.
            auto_tp_on = bool(ENGINE_CONFIG.get("auto_tp", True))
            if (tp == 1 and not cpu_only and not spread and not proportional and not pin_host
                    and consolidate and prefer_vram and auto_tp_on
                    and not getattr(spec, "is_embedding", False)):
                _flaky = {"steamdeck", "tablet", "mobile", "phone"}
                stable = [n for n in registry.alive_sorted()
                          if n.can_infer and n.eff_ram_gb > 0
                          and "windows" not in (n.os or "").lower()
                          and (n.hostname or "").lower() not in _flaky]
                gpu_pool = sum(n.eff_vram_gb for n in registry.alive_sorted() if n.can_infer)
                # Largest single PIPELINE-capable node (incl. a Windows host like beast — a pipeline,
                # unlike the TP all-reduce mesh, runs fine on Windows). If the model fits ONE such node's
                # RAM, a single-node CPU pipeline (NO per-layer all-reduce) is both faster and more
                # reliable than a TP mesh whose blocking-TCP all-reduce between separate boxes is
                # latency-bound — so DON'T auto-TP it. #40 crossover bench (m4c44, this fleet): 7B
                # pipeline-on-beast 1.95 tok/s vs tp2 1.76 (0.90x); 14B 1.09 vs 0.99 (0.91x); plus TP
                # loads 1.5-3x slower and the mesh intermittently stalls. CPU-TP only pays off when no
                # single node can hold the model (forcing a slow multi-node pipeline either way).
                biggest_node_gb = max((n.eff_ram_gb for n in registry.alive_sorted() if n.can_infer),
                                      default=0.0)
                model_gb = spec.total_weight_bytes / GB
                ratio = float(ENGINE_CONFIG.get("auto_tp_ratio", 1.5))
                if model_gb > ratio * gpu_pool and model_gb > biggest_node_gb and len(stable) >= 2:
                    nh, nkv = spec.num_heads, spec.num_kv_heads
                    ng = max(1, spec.intermediate_size // 128)
                    auto_t = 1
                    for _t in range(2, min(len(stable), 8) + 1):   # largest valid width (more nodes = more bw)
                        if nh % _t == 0 and (_t <= nkv or _t % nkv == 0) and _t <= ng:
                            auto_t = _t
                    if auto_t >= 2:
                        log_activity(f"{friendly}: AUTO-TP — ~{model_gb:.0f} GB weights >> {gpu_pool:.0f} GB "
                                     f"GPU pool AND > biggest single node ({biggest_node_gb:.0f} GB, so no "
                                     f"one-node pipeline) -> CPU tensor-parallel tp={auto_t}")
                        try:
                            if reg_key in self.models:
                                await self._unload_model_locked(reg_key, "reload (auto-tp)")
                            await self._await_free_refresh()
                            return await self._load_tp_locked(friendly, target_id, spec, ctx,
                                                              auto_t, quant, cpu_only=True,
                                                              kv_quant=kv_quant,
                                                              kv_offload=kv_offload,
                                                              default_temp=default_temp,
                                                              default_min_p=default_min_p)
                        except Exception as exc:
                            log_activity(f"{friendly}: auto-TP failed ({exc!r}) -> pipeline fallback")

            # TENSOR-PARALLEL now COEXISTS with other resident models (#87): NO fleet-wide unload.
            # _load_tp_locked selects its own tp_nodes and the workers hold its shards alongside the
            # others (only the chosen tp_nodes' assignment is (re)set). Reloading the SAME model at a
            # new config still evicts its old copy first so a re-tp doesn't double-load it.
            if tp > 1:
                if reg_key in self.models:
                    await self._unload_model_locked(reg_key, "reload (tp)")
                await self._await_free_refresh()
                return await self._load_tp_locked(friendly, target_id, spec, ctx, tp, quant,
                                                  cpu_only=cpu_only, kv_quant=kv_quant,
                                                  kv_offload=kv_offload,
                                                  default_temp=default_temp,
                                                  default_min_p=default_min_p)

            # FIT-AS-MANY + NODE-SHARING (Inc 3a/3b): keep other resident models and place this
            # one wherever there's room — INCLUDING nodes already serving a model. Each node is
            # budgeted by what's actually left: live free RAM (free_mem_gb has already dropped for
            # resident models) + VRAM minus the bytes resident shards placed on its GPU. Reloading
            # the same model evicts its old copy first; the safety cap evicts LRU models.
            max_loaded = int(ENGINE_CONFIG.get("max_loaded", MAX_LOADED_MODELS))
            auto_unload = bool(ENGINE_CONFIG.get("auto_unload", True))
            if reg_key in self.models:
                await self._unload_model_locked(reg_key, "reload")
                await self._await_free_refresh()
            # Enforce the resident-model cap by evicting IDLE models (never a busy one).
            while len(self.models) >= max_loaded:
                victim = self._lru_evictable() if auto_unload else None
                if victim is None:
                    raise RuntimeError(
                        f"at the max of {max_loaded} resident model(s) and " +
                        ("all are busy serving requests" if auto_unload else "auto-unload is off") +
                        f" — unload one before loading '{friendly}'")
                await self._unload_model_locked(victim, f"evict idle LRU (cap {max_loaded})")
                await self._await_free_refresh()
            if self.models:
                await self._await_free_refresh()   # current free RAM before budgeting vs residents

            # #draft-gpu: resolve the spec draft's REAL size EARLY (downloads it if missing — small)
            # so planning can reserve controller-GPU room for it. Opt-in per load (draft_gpu=1):
            # greedy GPU-first placement otherwise fills the controller's card with target layers
            # and the draft ALWAYS falls to CPU, where a draft step can cost more than the sweep it
            # saves (beast bench: plain 1.33 vs spec 0.99 tok/s — MODEL_TEST_STATUS llama-3.3:70b).
            # The reserve costs target-GPU layers, so default loads keep the old greedy behavior.
            draft_reserve_gb = 0.0
            if draft_gpu:
                _dg_id = MODELS.get(friendly, (target_id, target_id))[1]
                if _dg_id and _dg_id != target_id:
                    try:
                        _dg_dir = await asyncio.to_thread(_controller_model_dir, _dg_id)
                        _dg_b = sum(os.path.getsize(os.path.join(_dg_dir, f))
                                    for f in os.listdir(_dg_dir) if f.endswith(".safetensors"))
                        if _dg_b > 0:
                            draft_reserve_gb = _dg_b / GB + max(0.0, draft_margin_gb)
                            log_activity(f"{_ollama_name(friendly)}: reserving {draft_reserve_gb:.1f} GB "
                                         f"of controller GPU for the spec draft ({_dg_id}) — #draft-gpu")
                    except Exception as _dg_exc:
                        log_activity(f"{_ollama_name(friendly)}: draft-gpu reserve failed ({_dg_exc!r}) "
                                     f"— planning without it")
            # Plan over CAPABLE nodes, each sized by memory LEFT after resident models (so a 2nd
            # model can share a node's spare RAM/VRAM). If a node fails the load (missing deps)
            # mark it incapable and replan; if it won't fit even sharing, evict LRU and retry.
            node_by_id: dict[str, Node] = {}
            stages: list[StageAssign] = []
            n_stages = 0
            oom_skip: set[str] = set()   # nodes that failed the KV-reserve probe this load (replan w/o them)
            drop_skip: set[str] = set()  # #99: nodes that dropped their link mid-load this load (replan w/o them)
            futs: dict = {}              # last attempt's dispatch futures (pre-init so the for/else free can't NameError)
            for attempt in range(8):
                committed: dict[str, int] = {}   # node_id -> VRAM bytes held by resident shards
                for rm in self.models.values():
                    for st in rm.plan.stages:
                        # reserve BOTH a resident shard's GPU weights AND the full-ctx KV it will grow
                        # into (#vram-coexist): else a 2nd model's weights eat the 1st's KV space and
                        # OOM its decode (the qwen3-on-beast breakage). gpu_kv_bytes is worker-reported.
                        committed[st.node_id] = (committed.get(st.node_id, 0)
                                                 + st.gpu_bytes + getattr(st, "gpu_kv_bytes", 0))
                # PARALLEL LOADS: also subtract OTHER in-flight loads' reserved footprint (a load
                # that's already planned + is streaming) so this plan can't claim the same bytes ->
                # no over-provision even though the streaming overlaps (#parallel-load).
                _res_ram, _res_vram = self._reserved_bytes(exclude_key=reg_key)
                for _nid, _vb in _res_vram.items():
                    committed[_nid] = committed.get(_nid, 0) + _vb
                _vram_weights_first = bool(ENGINE_CONFIG.get("vram_weights_first", True))
                node_by_id = {}
                mems = []
                for n in registry.alive_sorted():
                    if not n.can_infer or n.node_id in oom_skip or n.node_id in drop_skip:
                        continue
                    if exclude_nodes and n.node_id in exclude_nodes:
                        continue   # a sibling replica already owns this node (disjoint placement)
                    if pin_host and n.hostname != pin_host:
                        continue   # #pin-device: user pinned this load to one node (dashboard placement)
                    # cpu_only: plan against RAM ONLY (VRAM=0) so the model never lands in
                    # any GPU's VRAM — the worker is also told device='cpu' below.
                    # GPU budget = the MORE CONSERVATIVE of two views, so the planner never assigns
                    # more GPU layers than the WORKER can actually place (else the worker spills the
                    # overflow to CPU -> the "free VRAM but CPU-bound / 600s-timeout" bug):
                    #   (a) tracked: usable_vram - committed (resident weights + their reserved
                    #       full-ctx KV + other in-flight loads) — protects a co-resident model's
                    #       not-yet-faulted KV (#95 coexistence).
                    #   (b) live: vram_total - vram_used (heartbeat, ALL users incl. a desktop's
                    #       browser/Discord/etc. on a shared GPU) - other in-flight reservations not
                    #       yet faulted into vram_used. This is what the worker's mem_get_info sees.
                    # usable_vram (≈ total - reserve) ignores non-fleet GPU usage, so on a desktop-
                    # shared card (beast) it over-budgets; capping by live-free spreads layers to a
                    # genuinely-free GPU node (a headless worker) instead of overloading it -> CPU.
                    live_free = max(0.0, n.vram_total_gb - n.vram_used_gb
                                    - _res_vram.get(n.node_id, 0) / GB)
                    # #vram-weights-first: budget weights against PHYSICALLY-free VRAM (live_free already
                    # excludes resident weights + actually-faulted KV + other in-flight loads), so a new
                    # model uses resident models' reserved-but-unused KV headroom instead of spilling its
                    # weights to CPU. Off -> the conservative #95 view (also subtract reserved full-ctx KV).
                    if cpu_only:
                        free_vram = 0.0
                    elif _vram_weights_first:
                        free_vram = live_free
                    else:
                        free_vram = min(
                            n.free_vram_after_resident_gb(committed.get(n.node_id, 0)), live_free)
                    # Reserve a runtime VRAM floor so a thin-headroom GPU node isn't filled to the
                    # brink (decode activations + allocator fragmentation OOM it otherwise, dropping
                    # the stage mid-generation). RAM already keeps RAM_SAFETY_GB; VRAM had none.
                    free_vram = max(0.0, free_vram - PLAN_VRAM_FLOOR_GB)
                    # #draft-gpu: keep the spec draft's slice of the CONTROLLER-co-located GPU out
                    # of this plan (the draft loads into the controller process on this same card
                    # AFTER placement; without the reserve, greedy GPU-first fills it first).
                    if draft_reserve_gb > 0.0 and n.data_host in _LOCAL_IPS:
                        free_vram = max(0.0, free_vram - draft_reserve_gb)
                    # Reserve the build transient from RAM (the worker streams+builds each layer in
                    # CPU RAM before quant/placement, even for GPU-bound layers). Linux uses a tmpfs
                    # mmap (~1.3x one layer); Windows/no-shm loads in-RAM (~2.3x). A node that can't
                    # fit its OS-specific transient free is excluded — it would OOM mid-build.
                    node_reserve_gb = (bf16_layer_gb * LOAD_TRANSIENT_RAM) if stream_load else 0.0
                    if quant == "int4":   # #62: per-expert streaming caps the transient (chunk + small
                        node_reserve_gb = min(node_reserve_gb, STREAM_EXPERT_RESERVE_GB)   # layer blob)
                    # #78: the controller's CO-LOCATED worker (same box) must leave RAM for the controller
                    # to read+serve the full bf16 stream (OS cache + serving buffers) WHILE this worker
                    # builds its shard — else the box over-commits and the worker OOM-drops mid-load (the
                    # beast minimax crash). data_host in _LOCAL_IPS == same machine as the controller.
                    if n.data_host in _LOCAL_IPS:
                        node_reserve_gb += CONTROLLER_RAM_RESERVE_GB
                    ram_for_resident = (n.eff_ram_gb - node_reserve_gb
                                        - _res_ram.get(n.node_id, 0) / GB)   # #parallel-load reserve
                    if ram_for_resident <= 0:
                        continue   # too small for even one layer's build transient -> skip
                    usable = ram_for_resident + free_vram   # resident RAM budget (+ VRAM after residents/floor)
                    if usable <= 0:
                        continue
                    node_by_id[n.node_id] = n
                    mems.append(NodeMem(n.node_id, n.hostname, int(usable * GB),
                                        int(free_vram * GB), pref=_mem_pref(n)))
                if not mems:
                    victim = self._lru_evictable() if auto_unload else None
                    if victim is not None:
                        await self._unload_model_locked(victim, "evict idle LRU: no room for new model")
                        await self._await_free_refresh()
                        continue
                    if self.models:
                        raise RuntimeError("no room for the new model and resident model(s) are "
                                           + ("busy serving" if auto_unload else "kept (auto-unload off)"))
                    raise RuntimeError("no capable worker nodes connected "
                                       "(all missing inference deps, or both tiers disabled)")
                pv_eff = prefer_vram and not cpu_only
                gpu_spread_eff = gpu_spread and not cpu_only   # all-GPU is meaningless under cpu_only
                plan = plan_pipeline(spec, mems, ctx, consolidate=consolidate, prefer_vram=pv_eff,
                                     spread=spread, proportional=proportional,
                                     gpu_spread=gpu_spread_eff)
                if not plan.ok:
                    victim = self._lru_evictable() if auto_unload else None
                    if victim is not None:
                        await self._unload_model_locked(victim, "evict idle LRU: new model needs room")
                        await self._await_free_refresh()
                        continue
                    # Nothing to evict: auto-fit the context DOWN to what the pool can hold
                    # alongside the weights, instead of over-committing into an OOM (user policy).
                    fit_ctx, fplan = self._fit_ctx(spec, mems, ctx, consolidate, pv_eff,
                                                   CTX_AUTOFIT_FLOOR, spread=spread,
                                                   proportional=proportional,
                                                   gpu_spread=gpu_spread_eff)
                    if fplan.ok and fit_ctx < ctx:
                        log_activity(f"{friendly}: ctx {ctx} won't fit the pool alongside the "
                                     f"weights — auto-fitting ctx -> {fit_ctx} to avoid OOM")
                        print(f"[load] {friendly}: auto-fit ctx {ctx} -> {fit_ctx} (pool can't hold "
                              f"full-ctx KV + weights)")
                        ctx, plan = fit_ctx, fplan
                    else:
                        raise RuntimeError((plan.error or "planning failed")
                                           + f" — even ctx {CTX_AUTOFIT_FLOOR} won't fit; the model's "
                                             "weights exceed the usable pool (free memory or use a "
                                             "smaller quant)" + (
                            "; resident model(s) busy serving" if (self.models and auto_unload)
                            else "" if self.models else ""))
                stages = plan.stages
                # #76 guardrail: estimate the VRAM/RAM split for weights + full-ctx KV on THIS
                # placement (the plan only proved it fits each node's TOTAL RAM+VRAM, not that KV
                # lands in VRAM). For an AUTO ctx, cap it so the KV stays on the GPU (the deepseek
                # 128K first-token hang); for an EXPLICIT ctx, honor it but warn. The weight-spill
                # speed warning (model bigger than fleet VRAM) is informational either way.
                assess = _assess_placement(spec, ctx, mems, stages, cpu_only=cpu_only)
                cap = capreason = None
                if ctx_was_auto:
                    if (not assess["weight_bound"] and assess["suggested_ctx"]
                            and assess["suggested_ctx"] < ctx):
                        cap = max(CTX_AUTOFIT_FLOOR, assess["suggested_ctx"])   # keep KV in VRAM
                        capreason = (f"would put ~{assess['kv_ram_gb']:.1f} GB of KV in RAM "
                                     f"(GPU VRAM can't hold it) — keeping KV on the GPU")
                    elif assess["weight_bound"] and ctx > AUTO_CTX_SLOW_CAP:
                        cap = AUTO_CTX_SLOW_CAP   # model already CPU-spilled; avoid a huge RAM KV too
                        capreason = ("exceeds the fleet's VRAM (weights spill to CPU) — avoiding a "
                                     "large full-ctx KV buffer in RAM")
                if cap and cap < ctx:
                    rplan = plan_pipeline(spec, mems, cap, consolidate=consolidate,
                                          prefer_vram=pv_eff, spread=spread,
                                          proportional=proportional, gpu_spread=gpu_spread_eff)
                    if rplan.ok:
                        log_activity(f"{friendly}: ctx {ctx} {capreason} — auto-capping ctx -> {cap} "
                                     f"(pass an explicit ctx to override)")
                        print(f"[load] {friendly}: ctx-guardrail {ctx} -> {cap}")
                        ctx, plan, stages = cap, rplan, rplan.stages
                        assess = _assess_placement(spec, ctx, mems, stages, cpu_only=cpu_only)
                    else:   # capping DOWN only frees memory, so this is unreachable in practice; if
                        # it ever happens, keep the already-valid original plan + its warnings rather
                        # than aborting a loadable model — never silently proceed unlogged.
                        log_activity(f"{friendly}: wanted to cap ctx {ctx} -> {cap} but that replan "
                                     f"failed ({rplan.error}); keeping ctx {ctx} (warnings stand)")
                load_warnings = assess["warnings"]
                for _w in load_warnings:
                    log_activity(f"{friendly}: ⚠ {_w}")
                n_stages = len(stages)
                for st in stages:                  # reset only the nodes THIS model will use
                    nd = node_by_id.get(st.node_id)
                    if nd:
                        nd.clear_assignment()
                # PARALLEL LOAD: record THIS load's planned per-node footprint BEFORE the lock is
                # released for the streaming gather, so a concurrent load subtracts it. Conservative:
                # RAM build-transient for every stage (every layer builds in RAM, ~est_bytes) + VRAM
                # for a GPU stage's resident weights -> never over-provisions during the overlap.
                _resv = {}
                for st in stages:
                    _nd = node_by_id.get(st.node_id)
                    _is_gpu = bool(_nd) and (not cpu_only) and _nd.eff_vram_gb > 0
                    _resv[st.node_id] = {"ram": int(st.est_bytes),
                                         "vram": int(st.est_bytes) if _is_gpu else 0}
                self._reservations[reg_key] = _resv
                # A "shard" is one Lxx layer-slice the controller streams to a worker (plus the
                # embed/head slices) — there are MANY per load (≈ model layer count), NOT one per
                # node. The dashboard progress must count these real shards; the node ("stage")
                # count is tracked separately so "X/Y shards · A/B nodes" reads unambiguously.
                total_shards = (sum(max(0, s.layer_end - s.layer_start) for s in stages)
                                + (1 if any(s.has_embed for s in stages) else 0)
                                + (1 if any(s.has_head for s in stages) else 0))
                basis = _describe_plan(stages, node_by_id, cpu_only, pv_eff, quant,
                                       gpu_spread=gpu_spread_eff)
                log_activity(f"{friendly}: plan basis → {basis}")
                log_activity(f"{friendly}: handing out {total_shards} shard(s) across "
                             f"{n_stages} node(s) -> " + ", ".join(
                    f"{s.hostname}(L{s.layer_start}-{s.layer_end})" for s in stages))
                _card0 = self.loadings.get(reg_key) or {}   # keep load-start time + requester
                self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                                "target": target_id, "total": total_shards,
                                "ready": 0, "stages_total": n_stages, "stages_ready": 0,
                                "basis": basis, "warnings": load_warnings,
                                "node_ids": [s.node_id for s in stages],   # reaper grace for builders
                                "started": _card0.get("started") or time.time(),   # #76 guardrail + load-card timer
                                # #connections: carry the requester across the dispatch rebuild —
                                # the streaming phase (the visible bulk of a load) is what the
                                # Connections panel attributes to the client
                                "requested_by": _card0.get("requested_by", "")}
                # #shard-cache Inc 2 (serve-from-cache): if a VERIFIED int4 cache exists, tell every
                # worker to fetch PRE-PACKED int4 layers (cache=int4) instead of streaming the full
                # bf16 + re-quantizing — the big win for MoE/large loads (e.g. ~18 GB cache vs ~70 GB
                # bf16 stream). int4 + pipeline only: TP slices weights non-contiguously (its own
                # dispatch path, never reaches here) so it can't use the whole-layer cache. The
                # controller falls back to bf16 PER UNIT if any cache file is missing, and an old
                # worker that ignores the `cache` key just streams bf16 — both safe.
                _cache_quant = ""
                if quant == "int4":
                    try:
                        _cdir = await asyncio.to_thread(_controller_model_dir, target_id)
                        if _cdir and await asyncio.to_thread(_shard_cache_ok, _cdir, "int4"):
                            _cache_quant = "int4"
                            log_activity(f"{friendly}: serving from int4 shard cache "
                                         f"(skip bf16 stream + per-layer re-quant)")
                    except Exception as _ce:
                        log_activity(f"{friendly}: shard-cache check failed ({_ce!r}) -> bf16 stream")
                        _cache_quant = ""
                futs: dict[str, asyncio.Future] = {}
                loop = asyncio.get_event_loop()
                for i, st in enumerate(stages):
                    nd = node_by_id[st.node_id]
                    if i < n_stages - 1:
                        nxt = node_by_id[stages[i + 1].node_id]
                        next_host, next_port = nxt.data_host, nxt.data_port
                    else:
                        next_host, next_port = None, ARGS.data_port  # -> controller
                    link = self.links.get(st.node_id)
                    if link is None:
                        raise RuntimeError(f"no control link to {st.node_id}")
                    nd.stage = i
                    nd.layer_start, nd.layer_end = st.layer_start, st.layer_end
                    nd.load_state = "loading"     # red on the dashboard until this shard reports ready
                    # This stage's GPU budget — MUST mirror the planner's per-node free_vram (lines above)
                    # or the worker re-clamps the weights the planner placed on GPU back to CPU (the spill
                    # bug). LIVE free VRAM = vram_total - vram_used (all users incl. a desktop's apps on a
                    # shared GPU) - other in-flight loads, matching the worker's mem_get_info. 0 -> CPU.
                    # #vram-weights-first: budget against live-free (use resident models' reserved-but-
                    # unfaulted KV headroom); off -> the conservative #95 min() (also subtract reserved KV).
                    _live_free_gb = max(0.0, nd.vram_total_gb - nd.vram_used_gb
                                        - _res_vram.get(st.node_id, 0) / GB)
                    if cpu_only:
                        _gpu_budget_gb = 0.0
                    elif _vram_weights_first:
                        _gpu_budget_gb = max(0.0, _live_free_gb - PLAN_VRAM_FLOOR_GB)
                    else:
                        _gpu_budget_gb = max(0.0, min(
                            nd.free_vram_after_resident_gb(committed.get(st.node_id, 0)),
                            _live_free_gb) - PLAN_VRAM_FLOOR_GB)
                    # #draft-gpu: mirror the planner's reserve on the co-located node or the worker
                    # re-expands into the VRAM the draft is about to claim.
                    if draft_reserve_gb > 0.0 and nd.data_host in _LOCAL_IPS:
                        _gpu_budget_gb = max(0.0, _gpu_budget_gb - draft_reserve_gb)
                    fut = loop.create_future()
                    link.pending_loads[target_id] = fut   # #1: key by model so a co-node load can't pop us
                    futs[st.node_id] = fut
                    await link.send({
                        "type": "load", "model_id": target_id,
                        "layer_start": st.layer_start, "layer_end": st.layer_end,
                        "has_embed": st.has_embed, "has_head": st.has_head,
                        "stage": i, "num_stages": n_stages,
                        "next_host": next_host, "next_port": next_port, "dtype": load_dtype,
                        "controller_http_port": ARGS.http_port,
                        # cpu_only forces RAM placement on every shard regardless of node tier;
                        # otherwise the node's tier config decides (auto/gpu/cpu).
                        "device": "cpu" if cpu_only else nd.load_device(),
                        "gpu_budget_gb": round(_gpu_budget_gb, 3),   # #95: committed-aware GPU cap for this stage
                        "moe_offload": moe_offload,  # #moe-offload: split MoE layers (attn->GPU, experts->CPU RAM)
                        "cache": _cache_quant,       # #shard-cache Inc 2: '' | 'int4' -> fetch pre-packed cache
                        "quant": quant,              # 'none' | 'int8' (load-time choice)
                        "kv_quant": kv_quant,        # #172 TurboQuant KV preset (none|turbo2|turbo3|turbo4)
                        "kv_offload": kv_offload,    # #kv-offload: KV cache in system RAM (OffloadedCache)
                        "ctx": ctx,                  # full ctx -> worker pre-reserves KV (fail-fast)
                        # #63: this stage's planned resident bytes (quantized). The worker reserves
                        # this much RAM up front (a balloon) and consumes it shard-by-shard as layers
                        # install — fail-fast if the node can't hold its share, peak ~ the plan. It is
                        # the full stage est even for a GPU stage: every layer is built in RAM and only
                        # moved to VRAM in _place_modules at the END, so build-phase RAM == est_bytes.
                        "plan_ram_bytes": int(st.est_bytes),
                    })

                log_activity(f"{friendly}: awaiting {n_stages} shard(s) — workers fetch weights, "
                             f"then mmap-load + fuse + place: {', '.join(s.hostname for s in stages)}")

                def _ready_cb(nd):                   # flip this node green + log AS it finishes (live progress)
                    host = nd.hostname if nd else "?"
                    def cb(fut):
                        try:
                            if fut.cancelled() or fut.exception() is not None:
                                return
                            r = fut.result()
                        except Exception:
                            return
                        if nd is not None:
                            nd.load_state = "ready"   # green on the dashboard
                        _card = self.loadings.get(reg_key)   # THIS load's own card (parallel-safe)
                        if _card is not None:
                            # a NODE finished its WHOLE range -> count nodes here; the per-Lxx
                            # shard count is advanced in the /weights serve path as each slice ships.
                            _card["stages_ready"] = _card.get("stages_ready", 0) + 1
                        gpb = r.get("gpu_bytes", 0) if isinstance(r, dict) else 0
                        tot = r.get("loaded_bytes", 0) if isinstance(r, dict) else 0
                        ram = max(0, tot - gpb)
                        if gpb and ram:
                            where = f"{gpb / GB:.1f} GB GPU + {ram / GB:.1f} GB RAM"
                        elif gpb:
                            where = f"{gpb / GB:.1f} GB on GPU"
                        else:
                            where = f"{ram / GB:.1f} GB in RAM"   # CPU node — not "0 GB on GPU"
                        _plc = r.get("placement") if isinstance(r, dict) else None
                        _moe = r.get("moe") if isinstance(r, dict) else None
                        log_activity(f"  {host}: shard loaded ({where})"
                                     + (f" | {_plc}" if _plc else "")
                                     + (f" | moe={_moe}" if _moe else ""))
                    return cb
                for _nid, _fut in futs.items():
                    _fut.add_done_callback(_ready_cb(node_by_id.get(_nid)))

                # Load-ack timeout scales with the bf16 READ volume, not a flat wall: every shard
                # streams its slice of the FULL bf16 from the controller's weights drive (often a
                # slow USB drive ~150 MB/s), and that drive serves all shards, so total read time
                # ~ total_bf16 / drive_MBps regardless of node count. CRITICAL: budget the PRE-quant
                # bf16 bytes — an int4 load still STREAMS the full bf16 (the worker quantizes after),
                # so sizing on the shrunken int4 footprint timed out the 426 GB minimax int4 build
                # (#100). 35 MB/s floor + 5 min + a per-GB quantize allowance; clamp [15 min, 4 h].
                read_bytes = total_bf16_bytes or getattr(spec, "total_weight_bytes", 0) or 0
                _quant_secs = (read_bytes / GB) * (4.0 if quant in ("int4", "int8") else 0.0)
                load_timeout = int(read_bytes / (35 * 1024 * 1024)) + 300 + int(_quant_secs)
                load_timeout = max(900, min(load_timeout, 4 * 3600))
                # DROP self.lock around the (multi-minute) streaming gather and re-acquire after, so a
                # 2nd load AND an /unload of a different model can run meanwhile (#parallel-load). Loads
                # are NOT serialized — they overlap; memory-safety comes from the reservation ledger
                # (committed + _reserved_bytes already ran under the lock, dispatch is done, the
                # reservation persists so a concurrent plan subtracts it). All weights stream from the
                # controller's single drive so overlap doesn't speed an individual load, but a small/fast
                # load no longer waits behind a huge one. Safe: the gather only awaits worker futures (no
                # engine-state mutation), and an unload freeing a co-resident model only RELAXES this
                # load's budget. _held tracks lock ownership so a CancelledError
                # delivered at the re-acquire can't make the method's finally release a lock we don't hold.
                self.lock.release()
                _held = False
                try:
                    results = await asyncio.gather(
                        *[asyncio.wait_for(f, timeout=load_timeout) for f in futs.values()],
                        return_exceptions=True)
                except asyncio.CancelledError:
                    # #stuck-load-override: a force load (or shutdown) cancelled us mid-stream. Re-acquire
                    # the lock and free any shards that DID build on workers, so the cancelled load leaves
                    # nothing resident on the fleet, then re-raise so cleanup (card/reservation) proceeds.
                    await self.lock.acquire()
                    _held = True
                    with contextlib.suppress(Exception):
                        await self._free_partial_stages(target_id, futs.keys(), node_by_id)
                    raise
                finally:
                    if not _held:
                        await self.lock.acquire()
                        _held = True
                incapable: list[str] = []
                oomed: list[str] = []
                dropped: list[str] = []   # #99: nodes whose link dropped this attempt (replan on survivors)
                hard_error: Optional[str] = None
                for nid, r in zip(futs.keys(), results):
                    err = (repr(r) if isinstance(r, Exception)
                           else str(r.get("error")) if isinstance(r, dict) and r.get("type") == "error"
                           else None)
                    if err is None:
                        continue
                    if any(k in err for k in ("No module named", "ModuleNotFoundError", "ImportError")):
                        nd = node_by_id.get(nid)
                        if nd:
                            nd.can_infer = False
                            nd.incapable_reason = "missing inference deps (e.g. torch)"
                            incapable.append(nd.hostname)
                    elif any(k in err for k in ("KV_RESERVE_OOM", "out of memory",
                                                "OutOfMemoryError", "CUDA error: out of memory")):
                        # node couldn't reserve its KV — skip it for THIS load and replan (not
                        # permanently incapable; it's fine for a smaller shard/ctx later).
                        oom_skip.add(nid)
                        nd = node_by_id.get(nid)
                        oomed.append(nd.hostname if nd else nid)
                    elif any(k in err for k in ("disconnected mid-operation", "ConnectionError",
                                                "ConnectionResetError", "ConnectionAbortedError",
                                                "BrokenPipeError", "IncompleteReadError")):
                        # a worker DROPPED its link mid-load (#99) — often a silent OOM-kill (PVE
                        # OOMScoreAdjust 800 SIGKILLs before a clean KV_RESERVE_OOM). Skip it for THIS
                        # load and replan on the survivors; if it ALREADY dropped this load it's
                        # flapping -> treat as a real failure (stop churning, don't burn all 8 attempts).
                        if nid in drop_skip:
                            hard_error = f"node {nid} dropped twice during load: {err}"
                        else:
                            drop_skip.add(nid)
                            nd = node_by_id.get(nid)
                            dropped.append(nd.hostname if nd else nid)
                    else:
                        hard_error = f"node {nid} load error: {err}"
                if hard_error:
                    await self._free_partial_stages(target_id, futs.keys(), node_by_id)   # #98: free shards that DID build
                    raise RuntimeError(hard_error)
                if incapable or oomed or dropped:
                    if incapable:
                        print(f"[load] excluding incapable node(s) {incapable} (no torch); replanning")
                    if oomed:
                        print(f"[load] {oomed} failed KV-reserve (can't hold ctx={ctx}); "
                              f"replanning without them")
                    if dropped:
                        print(f"[load] {dropped} dropped their link mid-load (#99); replanning on survivors")
                    await self._free_partial_stages(target_id, futs.keys(), node_by_id)   # #98: free shards that DID build
                    continue  # retry without the incapable / OOM'd / dropped nodes
                # success: record each stage's worker-reported on-GPU bytes (size_vram), on the
                # node AND the stage (the stage copy survives node-sharing, where a 2nd model
                # would overwrite the single Node.shard_gpu_bytes).
                stage_by_id = {s.node_id: s for s in stages}
                for nid, r in zip(futs.keys(), results):
                    gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
                    gpu_kv = int(r.get("gpu_kv_bytes", 0)) if isinstance(r, dict) else 0
                    nd = node_by_id.get(nid)
                    if nd:
                        nd.shard_gpu_bytes = gpu_b
                        # Re-assert this load's per-Node assignment: a co-resident model's unload during
                        # the released-lock gather can clear it (node-sharing -> last-writer-wins), which
                        # would drop the now-resident shard to the short reaper grace. load_state ready +
                        # a non-None stage restore SERVING_GRACE_S. (plan.stages stays the real source of
                        # truth for accounting; this only fixes the mutable scalars the reaper reads.)
                        nd.load_state = "ready"
                        if nd.stage is None and nid in stage_by_id:
                            _st = stage_by_id[nid]
                            nd.stage, nd.layer_start, nd.layer_end = 0, _st.layer_start, _st.layer_end
                    if nid in stage_by_id:
                        stage_by_id[nid].gpu_bytes = gpu_b
                        stage_by_id[nid].gpu_kv_bytes = gpu_kv   # reserve this model's KV vs coexisting loads
                        # #real-stats: the stage's MEASURED weight bytes (vs the spec estimate)
                        stage_by_id[nid].loaded_bytes = int(r.get("loaded_bytes", 0)) if isinstance(r, dict) else 0
                break  # all stages loaded
            else:
                await self._free_partial_stages(target_id, futs.keys(), node_by_id)   # #98: free any built shards
                raise RuntimeError("load failed: no capable nodes left after exclusions")

            s0 = node_by_id[stages[0].node_id]
            _s0_dial = (_dial_host(s0.data_host), s0.data_port)
            stage0_writer = await self._connect_retry(*_s0_dial)
            tok = await asyncio.to_thread(_get_tokenizer, target_id)
            eos = self._eos_ids(tok)
            now = time.time()
            lm = LoadedModel(
                reg_key, target_id, spec, ctx, plan,
                [s.node_id for s in stages], tok, eos, now,
                quant=quant, kv_quant=kv_quant, kv_offload=kv_offload,
                default_temperature=default_temp, default_min_p=default_min_p,
                stage0_writer=stage0_writer, last_used=now,
                stage0_dial=_s0_dial, last_send_ts=now)   # #stage0-stale-reconnect: how to re-dial + freshness clock
            lm.base, lm.replica_idx = friendly, replica_idx   # data-parallel grouping (#39)
            lm.plan_basis = basis                             # placement basis (#65)
            lm.load_warnings, lm.load_assess = load_warnings, assess   # pre-load guardrail (#76)
            # #cpu-bound-visibility: the #76 assess warns from ESTIMATES; here we know the ACTUAL
            # GPU/CPU split (worker-reported gpu_bytes). If the model actually landed heavily on CPU
            # (the GPU pool was full by load time — the real cause of later multi-model loads crawling
            # and looking "busy but network-idle"), append a LOUD persistent warning + log it, so a
            # ~0.1 tok/s CPU-bound model is never mistaken for a hang.
            _vram_b = sum(s.gpu_bytes for s in stages)
            # #real-stats: judge the split against the worker-MEASURED weight total, not the spec
            # estimate — the int4 estimate overshoots real packed MoE size ~10%, which fabricated a
            # phantom CPU fraction on fully-GPU-resident models. Spec fallback = old workers only.
            _wtotal_b = sum(getattr(s, "loaded_bytes", 0) or 0 for s in stages) or spec.total_weight_bytes
            _cpu_frac = ((_wtotal_b - _vram_b) / _wtotal_b if _wtotal_b else 0.0)
            if _cpu_frac > 0.30:
                _sev = "SEVERE " if _cpu_frac > 0.6 else ""
                _wmsg = (f"{_cpu_frac*100:.0f}% of weights on CPU ({_sev}— GPU pool full) -> CPU-bound, "
                         f"slow generation. Unload a model or use a smaller quant for GPU speed.")
                if _wmsg not in lm.load_warnings:
                    lm.load_warnings = list(lm.load_warnings) + [_wmsg]
                log_activity(f"{_ollama_name(friendly)}: loaded {_cpu_frac*100:.0f}% on CPU — CPU-bound "
                             f"(slow); GPU pool full. Unload a model or lower quant for full speed.")
            _st0 = (self.loadings.get(reg_key) or {}).get("started")   # #model-detail: load wall-clock
            lm.load_seconds = max(0.0, now - _st0) if _st0 else 0.0
            # speculative decoding: load THIS model's small draft locally on the controller
            draft_id = MODELS.get(friendly, (target_id, target_id))[1]
            if draft_id and draft_id != target_id:
                try:
                    lm.draft_margin_gb = draft_margin_gb   # #draft-gpu: _load_draft's VRAM cushion
                    await asyncio.to_thread(self._load_draft, lm, draft_id)
                    print(f"[load] draft {draft_id} on controller -> speculative decode K={SPEC_K}")
                except Exception as exc:
                    print(f"[load] draft load failed ({exc!r}); plain KV-cache decode")
                    self._unload_draft(lm)
            self.models[reg_key] = lm
            self.loadings.pop(reg_key, None)   # card off -> dashboard flips to resident (finally also pops)
            registry.dirty = False
            print(f"[load] {reg_key} across {n_stages} stages: "
                  f"{[(s.hostname, s.num_layers) for s in stages]}")
            log_activity(f"{reg_key} READY across {n_stages} stage(s) "
                         f"[{len(self.models)} model(s) resident]")
            return lm
        finally:
            # Release self.lock iff THIS task currently holds it. Replaces the `async with self.lock`
            # exit: during the gather we drop the lock, and if cancelled at the re-acquire _held is
            # False — so we must NOT release (it would free a contender's lock; asyncio.Lock has no owner).
            if _held:
                self.lock.release()

    async def _load_embedding_locked(self, friendly: str, target_id: str, spec: ModelSpec,
                                     reg_key: Optional[str] = None,
                                     replica_idx: int = 0) -> LoadedModel:
        """Slim sibling of load(): an ENCODER (sentence-embedding) model loads WHOLE onto ONE
        capable node (no pipeline/TP/KV planning, no lm_head). Mirrors load()'s control-send +
        pending_load future + stage0_writer mechanism, and stores a minimal single-stage
        LoadedModel so the dashboard / /api/ps / model card render without special-casing.
        MUST be called with self.lock held (it's reached only from load(), which holds it)."""
        reg_key = reg_key or friendly
        model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
        spec = await asyncio.to_thread(spec_with_measurements, spec, model_dir)
        tok = await asyncio.to_thread(_get_tokenizer, target_id)
        await self.ensure_data_listener()
        # Reload of the same key -> drop the old copy first.
        if reg_key in self.models:
            await self._unload_model_locked(reg_key, "reload (embedding)")
            await self._await_free_refresh()
        # Pick ONE capable node: prefer a GPU+can_infer node, else any can_infer node.
        alive = [n for n in registry.alive_sorted() if n.can_infer]
        if not alive:
            raise RuntimeError("no capable worker nodes connected for the embedding model")
        node = next((n for n in alive if n.eff_vram_gb > 0), None) or alive[0]
        log_activity(f"load {friendly}: embedding (single node {node.hostname})")
        self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id, "total": 1, "ready": 0,
                        "stages_total": 1, "stages_ready": 0,
                        "basis": f"embedding: single-node ({node.hostname})", "warnings": [],
                        "node_ids": [node.node_id],
                        "started": (self.loadings.get(reg_key) or {}).get("started") or time.time(),
                        # #connections: keep the requester across the rebuild (panel attribution)
                        "requested_by": (self.loadings.get(reg_key) or {}).get("requested_by", "")}
        link = self.links.get(node.node_id)
        if link is None:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"no control link to {node.node_id}")
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        link.pending_loads[target_id] = fut   # #1: key by model (co-node loads race)
        await link.send({
            "type": "load", "kind": "embedding", "model_id": target_id,
            "controller_http_port": ARGS.http_port,
            # No next hop: the worker replies straight to the controller's data port.
            "next_host": None, "next_port": ARGS.data_port,
            "device": "cpu" if node.eff_vram_gb <= 0 else node.load_device(),
            "dtype": "float32",
        })
        try:
            r = await asyncio.wait_for(fut, timeout=GEN_TIMEOUT_S)
        except Exception as exc:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"embedding load on {node.hostname} failed: {exc!r}")
        if isinstance(r, dict) and r.get("type") == "error":
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"embedding load on {node.hostname} failed: {r.get('error')}")
        gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
        tot_b = int(r.get("loaded_bytes", 0)) if isinstance(r, dict) else 0
        # Minimal single-stage plan so build_status / /api/ps / the card render unchanged.
        stage = StageAssign(node_id=node.node_id, hostname=node.hostname,
                            layer_start=0, layer_end=spec.num_layers,
                            has_embed=True, has_head=True,
                            est_bytes=spec.total_weight_bytes,
                            usable_bytes=int(node.usable_total_gb * GB), gpu_bytes=gpu_b,
                            loaded_bytes=tot_b)
        plan = PlanResult(ok=True, model=spec.name, ctx_len=spec.max_ctx,
                          num_layers=spec.num_layers,
                          pool_usable_gb=node.usable_total_gb,
                          required_gb=spec.total_weight_bytes / GB, stages=[stage])
        node.shard_gpu_bytes = gpu_b
        _emb_dial = (_dial_host(node.data_host), node.data_port)
        stage0_writer = await self._connect_retry(*_emb_dial)
        now = time.time()
        lm = LoadedModel(
            reg_key, target_id, spec, spec.max_ctx, plan,
            [node.node_id], tok, set(), now,
            quant="none", stage0_writer=stage0_writer, last_used=now,
            stage0_dial=_emb_dial, last_send_ts=now)
        lm.base, lm.replica_idx = friendly, replica_idx
        lm.plan_basis = "embedding: single-node"
        self.models[reg_key] = lm
        self.loadings.pop(reg_key, None)
        registry.dirty = False
        print(f"[load] {reg_key} embedding on {node.hostname} "
              f"({spec.total_weight_bytes / GB:.2f} GB)")
        log_activity(f"{reg_key} READY (embedding, {node.hostname}) "
                     f"[{len(self.models)} model(s) resident]")
        return lm

    async def replicate(self, friendly: str, ctx: int, count: int,
                        consolidate: bool = True, prefer_vram: bool = True,
                        quant: str = "none", kv_quant: str = "",
                        kv_offload: bool = False,
                        default_temp: Optional[float] = None,
                        default_min_p: Optional[float] = None) -> list["LoadedModel"]:
        """Load `count` full copies of `friendly` on DISJOINT node sets — the small-model
        throughput lever (#39). Replica 0 is keyed `friendly`; replica i (i>=1) is keyed
        `friendly#i`. Requests for `friendly` are then least-loaded / round-robin routed
        across all copies, so each copy adds a concurrent decode slot. Each load excludes the
        nodes its siblings already use (a worker keys shards by model_id, so two copies of one
        target cannot share a node). Stops early (keeping the copies that loaded) if the fleet
        runs out of disjoint room; raises only if even the first copy fails."""
        count = max(1, int(count))
        await self.unload(friendly)               # clean slate -> end with EXACTLY `count` copies
        out: list[LoadedModel] = []
        used: set[str] = set()
        self._no_evict_base = friendly            # protect siblings from idle-LRU eviction
        try:
            for i in range(count):
                key = friendly if i == 0 else f"{friendly}#{i}"
                try:
                    lm = await self.load(friendly, ctx, consolidate=consolidate,
                                         prefer_vram=prefer_vram, quant=quant, kv_quant=kv_quant,
                                         kv_offload=kv_offload, default_temp=default_temp,
                                         default_min_p=default_min_p,
                                         reg_key=key, exclude_nodes=set(used), replica_idx=i)
                except Exception as exc:
                    if i == 0:
                        raise
                    log_activity(f"replicate {friendly}: stopped at {len(out)} replica(s) "
                                 f"(no disjoint room for #{i}): {exc}")
                    break
                out.append(lm)
                used.update(lm.stage_node_ids)
                log_activity(f"replicate {friendly}: copy {i+1}/{count} on "
                             f"{[s.hostname for s in lm.plan.stages]}")
        finally:
            self._no_evict_base = None
        return out

    async def _free_partial_stages(self, target_id, node_ids, node_by_id) -> None:
        """Free shards that DID build for a distributed load that then FAILED (or got replanned without
        them): tell every node that received a stage to unload target_id + reset its dashboard state, so
        a partial load doesn't LEAK weights on the nodes that succeeded (beast leaked ~84 GB after a
        failed minimax load, #98). Idempotent with the worker's own self-clean on a failed build and with
        handle_load's top-of-load unload — a redundant {"type":"unload"} on a missing model_id pops
        safely. No LoadedModel is inserted at these failure sites, so resident-model state is untouched.
        TOTAL (never raises out): every send is suppressed; clear_assignment/load_state can't throw."""
        for nid in list(node_ids or []):
            ln = self.links.get(nid)
            if ln is not None:
                with contextlib.suppress(Exception):
                    await ln.send({"type": "unload", "model_id": target_id})
            nd = node_by_id.get(nid) if node_by_id else None
            if nd is not None:
                with contextlib.suppress(Exception):
                    nd.clear_assignment()
                    nd.load_state = "idle"

    async def _load_tp_locked(self, friendly: str, target_id: str, spec: ModelSpec,
                              ctx: int, tp: int, quant: str, cpu_only: bool = False,
                              kv_quant: str = "none", kv_offload: bool = False,
                              default_temp: Optional[float] = None,
                              default_min_p: Optional[float] = None) -> LoadedModel:
        """M4 tensor-parallel load. Every node in the group holds 1/tp of each layer
        (full embed/head/norm); rank 0 is the SINGLE pipeline stage the controller talks to
        and drives the peers over the all-reduce mesh. TP-v2 (per-rank streaming): each rank
        fetches ONLY its 1/tp tensor slice from /weights_tp and builds reduced-dim modules
        directly (Shard.from_stream's TP path), so a node needs ~full/tp RAM, not the full model.
        cpu_only=True runs TP across CPU nodes (RAM-bandwidth aggregation) instead of GPUs —
        the all-reduce and weight-sharding are device-agnostic, so the only difference is node
        selection (by free RAM) and forcing device='cpu' on every rank."""
        from transformers import AutoTokenizer
        L = spec.num_layers
        nkv = spec.num_kv_heads
        nh_heads = spec.num_heads
        if tp <= nkv:
            if nkv % tp != 0:
                raise RuntimeError(f"tp={tp} must divide num_key_value_heads={nkv} "
                                   f"(try tp in {[d for d in (2, 4, 8) if nkv % d == 0]})")
        else:
            # KV-HEAD REPLICATION (#87): tp > num_kv_heads -> replicate each KV head across tp//nkv
            # ranks so a model with few KV heads still spreads across MANY ranks (wide CPU TP). Needs
            # tp % nkv == 0 (even replication) and tp | num_attention_heads (even Q split). Forces an
            # EVEN split below (het + replication not supported yet).
            if tp % nkv != 0 or nh_heads % tp != 0:
                raise RuntimeError(
                    f"tp={tp} > num_kv_heads={nkv} (KV-head replication) needs tp % {nkv} == 0 AND "
                    f"num_attention_heads={nh_heads} % tp == 0 — try a tp that is a multiple of {nkv} "
                    f"and divides {nh_heads}")
        # FFN-group guard: the per-rank idim split needs at least `tp` int4 groups (128 cols) to hand
        # out, else the last rank's idim goes negative. Unreachable for real dense models (tp<=nh <<
        # intermediate/128) but cheap insurance for very wide TP / tiny-FFN archs.
        _ffn_groups = spec.intermediate_size // 128
        if _ffn_groups and tp > _ffn_groups:
            raise RuntimeError(f"tp={tp} too wide: exceeds FFN group count "
                               f"(intermediate_size {spec.intermediate_size} // 128 = {_ffn_groups})")
        # MIXED CPU+GPU mesh, sized by THROUGHPUT (#87). Unified candidate pool: a TP group can mix
        # GPU and CPU ranks, each classified per-NODE (GPU rank -> slice in VRAM; CPU rank -> in RAM).
        # Ranks are sized by BANDWIDTH (bw), NOT capacity (#68's VRAM/RAM GB): the lockstep mesh runs
        # at its SLOWEST rank, so a capacity split hands the slow CPU the biggest slice (straggler).
        # cpu_only forces every rank to CPU (the dashboard "run on CPU" path, unchanged).
        def is_gpu_rank(n) -> bool:
            return (not cpu_only) and n.vram_enabled and n.eff_vram_gb > 0
        def bw(n) -> float:
            return _node_tp_bw(n, is_gpu_rank(n))
        # PARALLEL LOAD: subtract OTHER in-flight loads' reservations so a TP load can't pick a node
        # a concurrent pipeline load (mid-stream with the lock released) has already claimed.
        _res_ram_tp, _res_vram_tp = self._reserved_bytes(exclude_key=friendly)
        def cap(n) -> float:          # has this node ANY room to be a rank? (VRAM for GPU, RAM for CPU)
            if is_gpu_rank(n):
                return max(0.0, n.eff_vram_gb - _res_vram_tp.get(n.node_id, 0) / GB)
            return max(0.0, n.eff_ram_gb - _res_ram_tp.get(n.node_id, 0) / GB)
        def avail(n) -> float:        # fit budget for the per-rank holds-its-share check
            if is_gpu_rank(n):
                return max(0.0, n.usable_total_gb - _res_vram_tp.get(n.node_id, 0) / GB)
            return max(0.0, n.eff_ram_gb - _res_ram_tp.get(n.node_id, 0) / GB)
        # The lockstep blocking-TCP all-reduce is only as reliable as its WEAKEST rank: a Windows
        # worker stalls it, and a battery HANDHELD (steamdeck/tablet/phone) that suspends/sleeps
        # drops out mid-forward -> "peer rank stalled" timeout. Verified LIVE: an all-nuc tp=4 mesh
        # generates cleanly AND coexists with qwen3 ("Paris..."), but a mesh that includes steamdeck
        # times out. Prefer STABLE server-class Linux nodes for the mesh; fall back to the full pool
        # only if there aren't enough (so TP still forms a group on a tiny/odd fleet).
        _TP_FLAKY = {"steamdeck", "tablet", "mobile", "phone"}
        _allc = [n for n in registry.alive_sorted() if n.can_infer and cap(n) > 0]
        _stable = [n for n in _allc if "windows" not in (n.os or "").lower()
                   and (n.hostname or "").lower() not in _TP_FLAKY]
        _pool = _stable if len(_stable) >= tp else _allc
        # ALL-CPU-FIRST selection (user policy): build the mesh from the fastest CPU nodes; admit a
        # GPU only when it clearly DOMINATES a CPU rank (>= 3x its bandwidth -> it'd carry a big
        # share) — a fast GPU otherwise sits idle at the all-reduce barrier waiting on slow CPU ranks.
        # No GPU / cpu_only -> all-CPU mesh; too few CPUs -> GPUs fill the remaining slots.
        # CAPACITY-AWARE rank pick (anti-oversubscription — the RAM analogue of the VRAM live-free
        # cap): a rank must actually hold its ~1/tp share in LIVE-free memory. avail() is already
        # live (eff_ram_gb derives from the latest free_mem_gb), but picking the top-tp purely by
        # BANDWIDTH hands a starved high-bandwidth node (e.g. an LPDDR5 NUC co-hosting Proxmox VMs,
        # ~2 GB free) a slice it can't fit -> the worker rejects and the WHOLE load fails instead
        # of using a roomier node. So keep only candidates that can hold the per-rank share, then
        # take the fastest tp of those (GPU ranks must clear the floor too).
        _share_floor = (spec.total_weight_bytes / GB / tp) * 1.15
        _fits = lambda n: avail(n) >= _share_floor
        _cpu = sorted((n for n in _pool if not is_gpu_rank(n)), key=lambda n: (bw(n), _mem_pref(n)), reverse=True)
        _gpu = sorted((n for n in _pool if is_gpu_rank(n)), key=lambda n: bw(n), reverse=True)
        chosen = [n for n in _cpu if _fits(n)][:tp]
        for g in _gpu:
            if not _fits(g):
                continue
            if len(chosen) < tp:
                chosen.append(g)
            else:
                weakest = min(chosen, key=bw)
                if bw(g) >= 3.0 * bw(weakest):
                    chosen[chosen.index(weakest)] = g
        if len(chosen) < tp:
            _top = sorted(_pool, key=avail, reverse=True)[:tp + 2]
            raise RuntimeError(
                f"tp={tp}{' cpu' if cpu_only else ''}: need {tp} nodes each with >= {_share_floor:.1f} GB "
                f"free for ~{spec.total_weight_bytes / GB:.1f} GB (1/{tp} + headroom); only {len(chosen)} "
                f"qualify. Most free now: " + ", ".join(f"{n.hostname} {avail(n):.1f}GB" for n in _top))
        # cand = chosen first (rank0 = fastest -> drives the mesh + holds embed/head), then the rest of
        # the pool as fallback so the live-link / distinct-host filter below can still reach tp nodes.
        cand = sorted(chosen, key=lambda n: (bw(n), _mem_pref(n)), reverse=True)
        cand += [n for n in sorted(_pool, key=lambda n: (bw(n), _mem_pref(n)), reverse=True)
                 if n not in chosen]
        kind = "mixed"   # refined to GPU / CPU / "N GPU + M CPU" once tp_nodes is chosen (below)
        # Pre-flight (anti-churn / finding 1): a reconnecting worker can leave a STALE node_id whose
        # half-dead control link still .send()s without raising, and a host can briefly appear under
        # two node_ids. Either lets a TP rank be assigned to a node that never actually receives the
        # load -> rank 0 then waits the WHOLE gather timeout (the observed
        # "tp load failed on <host>: TimeoutError()", with both ranks' weights served to one host).
        # Require a LIVE control link AND one rank per distinct HOST (freshest id wins, cand is
        # ranked first so the first seen per host is the chosen resource).
        live, seen_hosts = [], set()
        for n in cand:
            if self.links.get(n.node_id) is None or n.hostname in seen_hosts:
                continue
            seen_hosts.add(n.hostname); live.append(n)
        if len(live) < tp:
            raise RuntimeError(f"tp={tp} needs {tp} distinct live {kind} nodes; only {len(live)} "
                               f"have a live control link {[n.hostname for n in live]}")
        tp_nodes = live[:tp]
        # The all-reduce ROOT (rank0) binds the mesh + does the most blocking-socket work each layer;
        # prefer a NON-Windows node as root (the Windows controller host as mesh-root stalls the
        # per-layer all-reduce -> generate-time timeout). Keep the fastest non-Windows rank as rank0
        # when the mesh is mixed-OS; an all-Windows mesh is left as-is.
        if "windows" in (tp_nodes[0].os or "").lower():
            _nonwin = next((n for n in tp_nodes if "windows" not in (n.os or "").lower()), None)
            if _nonwin is not None:
                tp_nodes = [_nonwin] + [n for n in tp_nodes if n is not _nonwin]
        # per-rank tier of the CHOSEN mesh (#87): each rank is GPU or CPU on its own; drives the
        # per-rank `device` sent below and a human-readable "kind" label for the basis/log.
        rank_is_gpu = [is_gpu_rank(n) for n in tp_nodes]
        n_gpu = sum(rank_is_gpu); n_cpu = tp - n_gpu
        kind = ("GPU" if n_cpu == 0 else "CPU" if n_gpu == 0 else f"{n_gpu} GPU + {n_cpu} CPU")
        # Instrumentation: log the ACTUAL rank->node assignment to the activity feed (readable via
        # /status) so a TP load's placement is verifiable — the "-> host" in the /weights_tp serving
        # log is unreliable (both ranks share layer range [0,L], so its _owns match always resolves
        # to the first node, mislabeling every slice to one host).
        log_activity("TP rank assignment: " + ", ".join(
            f"rank{r}={n.hostname}({n.node_id})" for r, n in enumerate(tp_nodes)))
        tp_basis = (f"tensor-parallel tp={tp} ({kind}) -> "
                    + ", ".join(n.hostname for n in tp_nodes)
                    + f" -- each rank holds 1/{tp} of every layer"
                    + ("" if quant in (None, "none", "") else f", {quant}"))
        log_activity(f"{friendly}: plan basis → {tp_basis}")
        full_gb = spec.total_weight_bytes / GB
        # HETEROGENEOUS TP (#68): when the ranks' capacities really differ, split each layer
        # PROPORTIONAL to capacity (usable VRAM for GPU TP, RAM for CPU TP) so a bigger GPU holds a
        # bigger slice — the smallest node no longer has to hold an equal 1/tp share it can't fit.
        # `tp_weights` is sent to every rank, which builds its reduced-dim structure from the SAME
        # wire._tp_hetsplit the server slices with. Near-equal capacities -> tp_weights=None -> the
        # uniform 1/tp split (also keeps a rolling-update OLD worker, which ignores tp_weights, in
        # sync). Per-rank fit: a rank holds ~ its capacity-share of the model (+15% transient).
        if tp > nkv:
            # KV-head replication uses an EVEN split + the replication geometry (het + replication is
            # not supported yet) — uniform caps so the fit-check, rank_bytes, serve slice and worker
            # structure all route through wire._tp_hetsplit's replication branch with matching shapes.
            caps = [1.0] * tp
            het = True
        else:
            caps = [max(0.1, bw(n)) for n in tp_nodes]   # THROUGHPUT weights -> faster rank, bigger slice
            het = (max(caps) / min(caps)) > 1.15
        wtot = sum(caps) or 1.0
        # If the bandwidth-proportional (het) split would overflow a chosen rank's LIVE-free memory
        # but a plain uniform 1/tp split fits every rank, prefer uniform (anti-oversubscription): a
        # slightly slower but PLACEABLE mesh beats a failed load. The tp>nkv replication path uses
        # uniform caps already and is left untouched (het stays True there for its slice geometry).
        if het and tp <= nkv:
            _ovf = [n for n, c in zip(tp_nodes, caps) if avail(n) < full_gb * (c / wtot) * 1.15]
            if _ovf and all(avail(n) >= (full_gb / tp) * 1.15 for n in tp_nodes):
                log_activity(f"{friendly}: TP het split overflows {[n.hostname for n in _ovf]} "
                             f"-> uniform 1/{tp} split (fits live RAM)")
                caps, het, wtot = [1.0] * tp, False, float(tp)
        tp_weights = [round(c, 3) for c in caps] if het else None
        if het:
            for n, c in zip(tp_nodes, caps):
                share = full_gb * (c / wtot) * 1.15
                if avail(n) < share:
                    raise RuntimeError(
                        f"tp het: {n.hostname} can't hold its ~{share:.1f} GB capacity-share of "
                        f"~{full_gb:.1f} GB ({kind} avail {avail(n):.1f} GB).")
            pct = ", ".join(f"{n.hostname} {100 * c / wtot:.0f}%" for n, c in zip(tp_nodes, caps))
            tp_basis = (f"heterogeneous tensor-parallel tp={tp} ({kind}) -> {pct}"
                        + ("" if quant in (None, "none", "") else f", {quant}"))
            log_activity(f"{friendly}: plan basis → {tp_basis}")
        else:
            per_rank_gb = (full_gb / tp) * 1.15
            for n in tp_nodes:
                if avail(n) < per_rank_gb:
                    raise RuntimeError(
                        f"tp v2 (per-rank streaming) needs ~{per_rank_gb:.1f} GB/rank "
                        f"(1/{tp} of ~{full_gb:.1f} GB + headroom); {n.hostname} has "
                        f"{avail(n):.1f} GB {kind} free.")
        root = tp_nodes[0]
        tp_port = root.data_port + 1
        # COEXISTENCE (#87): do NOT clear EVERY node — that wipes the display/accounting for OTHER
        # resident models (whose shards stay loaded on the workers). Only (re)assign the chosen
        # tp_nodes; nodes holding other models keep their assignment, so this TP load coexists.
        for n in tp_nodes:
            n.clear_assignment()
        # In-flight marker so the dashboard shows "loading <model> X%" during a TP load instead of
        # "none": each rank streams its full [0,L) range + embed + head from /weights_tp, whose
        # serving path bumps loading["ready"] per slice. total = tp*(L+2) is chunking-invariant —
        # the per-rank (end-start) sum is always L, +1 embed +1 head, however the layers are chunked.
        self.loadings[friendly] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id,
                        "total": tp * (L + 2), "ready": 0,
                        "stages_total": tp, "stages_ready": 0, "basis": tp_basis,
                        "node_ids": [n.node_id for n in tp_nodes],
                        "started": (self.loadings.get(friendly) or {}).get("started") or time.time(),
                        # #connections: keep the requester across the rebuild (panel attribution)
                        "requested_by": (self.loadings.get(friendly) or {}).get("requested_by", "")}
        loop = asyncio.get_event_loop()
        futs: dict[str, asyncio.Future] = {}
        for rank, n in enumerate(tp_nodes):
            link = self.links.get(n.node_id)
            if link is None:
                raise RuntimeError(f"no control link to {n.node_id}")
            n.stage, n.tp_rank, n.tp_size = 0, rank, tp
            n.layer_start, n.layer_end = 0, L
            n.load_state = "loading"
            # per-rank planned bytes: heterogeneous -> this rank's capacity share; else even 1/tp.
            rank_bytes = (int(spec.total_weight_bytes * caps[rank] / wtot) if het
                          else int(spec.total_weight_bytes / tp))
            msg = {"type": "load", "model_id": target_id,
                   "layer_start": 0, "layer_end": L, "has_embed": True, "has_head": True,
                   "stage": 0, "num_stages": 1, "dtype": "bfloat16",
                   "kv_quant": kv_quant,   # #172 TurboQuant KV preset (per-rank shard honors it)
                   "kv_offload": kv_offload,   # #kv-offload: KV cache in system RAM
                   "controller_http_port": ARGS.http_port,
                   # per-rank device from THIS node's tier (#87): a GPU rank gets its load_device()
                   # ("" -> worker cpu+gpu default, or "gpu"); a CPU rank gets explicit 'cpu'.
                   "device": n.load_device() if rank_is_gpu[rank] else "cpu",
                   "quant": quant, "tp_rank": rank, "tp_size": tp,
                   "tp_root_host": root.data_host, "tp_root_port": tp_port,
                   # #68: per-rank capacity weights (None when ~equal) -> heterogeneous split; every
                   # rank gets the SAME list so wire._tp_hetsplit is identical on all ranks + the serve.
                   "tp_weights": tp_weights,
                   # #63: this rank's planned resident bytes (its capacity share) -> RAM balloon.
                   "plan_ram_bytes": rank_bytes}
            if rank == 0:
                msg["next_host"], msg["next_port"] = None, ARGS.data_port  # -> controller
            fut = loop.create_future()
            link.pending_loads[target_id] = fut   # #1: key by model (co-node loads race)
            futs[n.node_id] = fut
            await link.send(msg)
        async def _abort_cleanup():
            # A failed TP load must NOT leave dirty per-rank state (bound mesh port, partial shard,
            # load_state='loading') — that arms the next load's churn (the observed cascade). Tell
            # EVERY rank to unload + reset, clear the in-flight marker, and arm a self-update
            # cool-down so the now-"idle"-looking controller doesn't immediately exit(42).
            for _n in tp_nodes:
                _ln = self.links.get(_n.node_id)
                if _ln is not None:
                    with contextlib.suppress(Exception):
                        await _ln.send({"type": "unload", "model_id": target_id})
                _n.clear_assignment(); _n.load_state = "idle"
            self.loadings.pop(friendly, None)
            self._last_load_failure = time.time()
        # Scale the TP load timeout by the bf16 READ volume like the pipeline path (#100): each rank
        # still streams its full bf16 slice and quantizes after, so a flat 900 s timed out big loads.
        # spec is already for_quant'd here -> recover ~bf16 (int4 ~/0.3, int8 x2). Clamp [15 min, 4 h].
        _tpb = getattr(spec, "total_weight_bytes", 0) or 0
        _tpb = int(_tpb / 0.3) if quant == "int4" else (_tpb * 2 if quant == "int8" else _tpb)
        tp_load_timeout = max(900, min(int(_tpb / (35 * 1024 * 1024)) + 300, 4 * 3600))
        results = await asyncio.gather(
            *[asyncio.wait_for(f, timeout=tp_load_timeout) for f in futs.values()], return_exceptions=True)
        tp_gpu_bytes = 0   # #69: total on-GPU bytes ACROSS all ranks (each reports its own)
        tp_gpu_kv_bytes = 0   # total full-ctx KV reserved on GPU across ranks (coexistence reserve)
        tp_loaded_bytes = 0   # #real-stats: worker-MEASURED weight bytes across ranks (vs spec estimate)
        for nid, r in zip(futs.keys(), results):
            err = (repr(r) if isinstance(r, Exception)
                   else r.get("error") if isinstance(r, dict) and r.get("type") == "error" else None)
            if err is not None:
                hn = registry._nodes[nid].hostname if nid in registry._nodes else nid
                await _abort_cleanup()
                raise RuntimeError(f"tp load failed on {hn}: {err}")
            nd = registry._nodes.get(nid)
            if nd:
                nd.load_state = "ready"
                if isinstance(r, dict):
                    nd.shard_gpu_bytes = int(r.get("gpu_bytes", 0))
                    tp_gpu_bytes += int(r.get("gpu_bytes", 0))
                    tp_gpu_kv_bytes += int(r.get("gpu_kv_bytes", 0))
                    tp_loaded_bytes += int(r.get("loaded_bytes", 0))
                _tcard = self.loadings.get(friendly)   # bump the node counter ("A/B nodes loaded")
                if _tcard is not None:
                    _tcard["stages_ready"] = _tcard.get("stages_ready", 0) + 1
        # the pipeline is just rank 0; the controller talks only to it
        _tp_dial = (_dial_host(root.data_host), root.data_port)
        stage0_writer = await self._connect_retry(*_tp_dial)
        tok = await asyncio.to_thread(_get_tokenizer, target_id)
        eos = self._eos_ids(tok)
        # (TP models carry no speculative draft — big-model decode is bandwidth-bound.)
        # #69: carry the TP group's TOTAL on-GPU bytes on the (single) TP stage so the dashboard's
        # vram_used = sum(stage.gpu_bytes) reflects reality (beast+theocomp VRAM), not 0. The TP
        # plan has one stage representing the whole group, so the aggregate belongs here.
        stage = StageAssign(root.node_id, root.hostname, 0, L, True, True,
                            int(spec.total_weight_bytes / tp), int(root.usable_total_gb * GB),
                            gpu_bytes=int(tp_gpu_bytes), gpu_kv_bytes=int(tp_gpu_kv_bytes),
                            loaded_bytes=int(tp_loaded_bytes))
        plan = PlanResult(ok=True, model=spec.name, ctx_len=ctx, num_layers=L,
                          pool_usable_gb=round(sum(n.usable_total_gb for n in tp_nodes), 2),
                          required_gb=round(full_gb, 2), stages=[stage])
        now = time.time()
        lm = LoadedModel(friendly, target_id, spec, ctx, plan,
                         [n.node_id for n in tp_nodes], tok, eos, now,
                         quant=quant, kv_quant=kv_quant, kv_offload=kv_offload,
                         default_temperature=default_temp, default_min_p=default_min_p,
                         stage0_writer=stage0_writer, last_used=now,
                         stage0_dial=_tp_dial, last_send_ts=now)
        lm.plan_basis = tp_basis                          # placement basis (#65)
        lm.tp_size = tp                                    # #88: record TP width for the card + /reconfigure
        self.models[friendly] = lm
        self.loadings.pop(friendly, None)   # card off -> dashboard flips to resident (finally also pops)
        registry.dirty = False
        print(f"[load] TP tp={tp} {friendly}: rank0={root.hostname} "
              f"peers={[n.hostname for n in tp_nodes[1:]]} (1/{tp} of each layer per rank)")
        log_activity(f"{friendly} READY (tp={tp}): rank0={root.hostname}, "
                     f"peers={[n.hostname for n in tp_nodes[1:]]}")
        return lm

    async def reconfigure(self, friendly: str, tp: int, ctx: int, quant: str,
                          consolidate: bool, prefer_vram: bool, cpu_only: bool) -> LoadedModel:
        """#88 managed reload: switch a RESIDENT model to/from tensor-parallel (or change its TP
        width / ctx / quant) as ONE operation, rolling back to a WORKING pipeline copy if the new
        layout fails — so the model is NEVER left evicted-with-nothing-loaded. Reuses engine.load
        (force=True) for all the eviction/placement/guardrail logic (the worker wire has no in-place
        resharding, so a layout switch is inherently a re-stream); the only new behavior is the
        snapshot + rollback + the in-flight 'reconfiguring' marker for the dashboard. Does NOT hold
        self.lock itself — engine.load acquires it (asyncio.Lock is not reentrant)."""
        prev = self.models.get(friendly)
        if prev is None:
            raise ValueError(f"'{friendly}' is not resident — load it before reconfiguring")
        prev_tp, prev_ctx, prev_quant = getattr(prev, "tp_size", 1), prev.ctx, (prev.quant or "none")
        from_label = f"tp{prev_tp}" if prev_tp > 1 else "pipeline"
        to_label = ((f"tp{tp}" + ("-cpu" if cpu_only else "")) if tp > 1 else "pipeline")
        self.reconfiguring = {"model": _ollama_name(friendly), "from": from_label, "to": to_label,
                              "from_tp": prev_tp, "to_tp": tp}
        log_activity(f"{friendly}: RECONFIGURE {from_label} -> {to_label} "
                     f"(ctx {prev_ctx}->{ctx}, quant {prev_quant}->{quant})")
        try:
            lm = await self.load(friendly, ctx, consolidate=consolidate, prefer_vram=prefer_vram,
                                 quant=quant, tp=tp, cpu_only=cpu_only, force=True)
            log_activity(f"{friendly}: reconfigured -> {lm.plan_basis}")
            return lm
        except Exception as exc:
            # New layout failed; engine.load already evicted the old copy. Restore a WORKING copy: a
            # plain pipeline-auto load at the PREVIOUS ctx/quant (GPU-first, spills to CPU — the robust
            # path that always places). Better than leaving the model gone.
            log_activity(f"{friendly}: reconfigure to {to_label} FAILED ({exc!r}) -> rolling back to "
                         f"pipeline @ ctx={prev_ctx} {prev_quant}")
            try:
                await self.load(friendly, prev_ctx, quant=prev_quant, force=True)
                log_activity(f"{friendly}: rolled back to a pipeline copy (serving restored)")
            except Exception as exc2:
                log_activity(f"{friendly}: ROLLBACK ALSO FAILED ({exc2!r}) — model is NOT resident")
                raise RuntimeError(f"reconfigure failed AND rollback failed: {exc} || {exc2}")
            raise RuntimeError(f"reconfigure to {to_label} failed: {exc} (rolled back to pipeline)")
        finally:
            self.reconfiguring = None

    def _model_cpu_frac(self, m) -> float:
        """Fraction of a resident model's WEIGHTS currently on CPU (0.0 = fully on GPU), from the
        worker-reported per-stage gpu_bytes — mirrors status.py's cpu_frac so the juggler judges by
        the SAME 'is this hybrid?' signal the dashboard shows."""
        try:
            vram = sum(s.gpu_bytes for s in m.plan.stages)
            total = (sum(getattr(s, "loaded_bytes", 0) or 0 for s in m.plan.stages)
                     or m.spec.total_weight_bytes)
            return max(0.0, (total - vram) / total) if total else 0.0
        except Exception:
            return 0.0

    async def _juggle_would_fit_vram(self, m) -> bool:
        """Dry-run the VRAM-first planner for `m` against the LIVE free VRAM PLUS this model's own
        on-GPU bytes (which a reload reclaims), and return True only when every weight lands on GPU.
        Cheap go/no-go so the juggler never does a disruptive re-place that would stay hybrid."""
        try:
            spec = m.spec   # the model's OWN measured, quant-sized spec (what it was placed with)
            own = {}   # per-node VRAM this model holds now -> reclaimed on reload, so add it back
            for s in m.plan.stages:
                own[s.node_id] = own.get(s.node_id, 0) + (getattr(s, "gpu_bytes", 0) or 0)
            mems = []
            for n in registry.alive_sorted():
                fv = max(0.0, n.eff_vram_gb - PLAN_VRAM_FLOOR_GB) + own.get(n.node_id, 0) / GB
                ram = n.eff_ram_gb - (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
                mems.append(NodeMem(n.node_id, n.hostname,
                                    int((max(0.0, ram) + fv) * GB), int(fv * GB)))
            p = plan_pipeline(spec, mems, m.ctx, consolidate=True, prefer_vram=True)
            if not p.ok:
                return False
            assess = _assess_placement(spec, m.ctx, mems, p.stages, cpu_only=False)
            return (assess.get("cpu_weight_gb", 0.0) or 0.0) < 0.05
        except Exception as exc:
            log_activity(f"juggler: fit-check error ({exc!r})")
            return False

    async def _maybe_juggle(self, reason: str = "auto-unload") -> None:
        """#juggler: promote a resident model that is running split GPU+RAM (hybrid) to VRAM-only once
        the fleet has room. TWO triggers (both opt-in via ENGINE_CONFIG['juggler']): the idle-unload
        hook (right after a reclaim frees VRAM) AND a periodic ~60s sweep (_juggler_sweep_loop) — so a
        promotion also happens when VRAM frees for ANY reason (a manual unload, a shrinking KV, an
        earlier promotion), not just on idle-unload. Walks the promotable hybrids HOTTEST-first and
        promotes the hottest one whose weights would now fit ENTIRELY on GPU: a bigger, hotter hybrid
        that still won't fit must NOT block a smaller one that would, and a model that can never fit
        (too big for the fleet's GPUs) is simply skipped. Embeddings / encoder models are IGNORED —
        they run on CPU by design and aren't served through the generate() barrier, so they can't be
        promoted to GPU. HITLESS: engage a per-model barrier so new requests wait, DRAIN the in-flight
        generation, re-place VRAM-first via reconfigure (atomic + rollback), then release the barrier —
        the client's open connection just pauses across the swap, no reconnect. One promotion per call;
        serialized by _juggle_lock. #no-unload / persist PINNED models are eligible on purpose (a
        promotion is a reload-in-a-BETTER-way, not a removal — the finally restores a pinned target if a
        rare reconfigure+rollback double-failure ever evicts it, so "never auto-unloaded" still holds)."""
        if not ENGINE_CONFIG.get("juggler", False) or self._juggle_lock.locked():
            return
        async with self._juggle_lock:
            # 1) rank the PROMOTABLE hybrids by hotness (freshest use first, so a 'constantly in use'
            #    model wins). Skip embeddings/encoders (CPU-by-design, not served through the barrier)
            #    and anything already ~all-GPU (nothing to promote).
            cands = []
            for fr, m in list(self.models.items()):
                if getattr(m.spec, "is_embedding", False) or getattr(m, "is_embedding", False):
                    continue
                if self._model_cpu_frac(m) <= 0.02:
                    continue
                hot = max(m.last_used or 0.0, getattr(m, "last_token_ts", 0.0) or 0.0)
                cands.append((hot, fr, m))
            cands.sort(key=lambda t: t[0], reverse=True)    # hottest first
            # 2) promote the HOTTEST candidate that would now land FULLY on GPU. Skip (don't block on)
            #    any hotter one that still won't fit VRAM-only. Quiet no-op when nothing fits — this
            #    runs every ~60s, so it must NOT log on the common "nothing to do" case.
            fr = m = base = None
            for _hot, _fr, _m in cands:
                if await self._juggle_would_fit_vram(_m):
                    fr, m, base = _fr, _m, (_m.base or _fr)
                    break
            if fr is None or fr not in self.models:
                return
            # 3) HITLESS swap: barrier -> drain -> reconfigure VRAM-first -> release.
            gate = asyncio.Event()                          # CLEAR (present) = barrier engaged
            self._promote_gates[base] = gate
            try:
                drain_s = float(ENGINE_CONFIG.get("juggle_drain_s", 120.0))
                deadline = time.time() + drain_s
                while time.time() < deadline:               # let the current gen(s) finish
                    if not [r for r in self.replicas_of(base)
                            if (r.active or 0) > 0 or (r.queued or 0) > 0]:
                        break
                    await asyncio.sleep(0.25)
                else:
                    log_activity(f"juggler: {fr} still busy after {drain_s:g}s — deferring promotion")
                    return
                log_activity(f"juggler: promoting {fr} to VRAM-only "
                             f"({self._model_cpu_frac(m)*100:.0f}% was on CPU) [{reason}] — "
                             f"hitless re-place")
                await self.reconfigure(fr, tp=getattr(m, "tp_size", 1), ctx=m.ctx,
                                       quant=(m.quant or "none"), consolidate=True,
                                       prefer_vram=True, cpu_only=False)
                newm = self.models.get(fr)
                nf = self._model_cpu_frac(newm) if newm else 1.0
                log_activity(f"juggler: {fr} re-placed — now {nf*100:.0f}% on CPU "
                             + ("(fully on GPU)" if nf <= 0.02 else "(still hybrid — fleet shifted)"))
            except Exception as exc:
                log_activity(f"juggler: promote {fr} FAILED ({exc!r})")
            finally:
                # #no-unload safety: a promotion is a reload-in-a-BETTER-way, NEVER a removal — so a
                # pinned (do-not-auto-unload / persist) target must be resident again BEFORE the barrier
                # releases, even in the rare case reconfigure AND its rollback both failed. The gate
                # release is in an INNER finally so it ALWAYS runs — even if the restore await is
                # cancelled at shutdown (CancelledError is a BaseException that suppress(Exception)
                # would not catch), the barrier must never be left engaged with waiters parked on it.
                try:
                    _pinned = (base in (ENGINE_CONFIG.get("no_unload_models") or {})
                               or fr in (ENGINE_CONFIG.get("no_unload_models") or {})
                               or base in (ENGINE_CONFIG.get("persist_models") or {})
                               or fr in (ENGINE_CONFIG.get("persist_models") or {}))
                    if _pinned and fr not in self.models:
                        log_activity(f"juggler: {fr} is pinned but not resident after promotion — restoring")
                        with contextlib.suppress(Exception):
                            await self.load(fr, m.ctx, quant=(m.quant or "none"))
                finally:
                    gate.set()                              # wake every barrier waiter (ALWAYS)
                    self._promote_gates.pop(base, None)

    async def _await_free_refresh(self, timeout: float = 12.0) -> None:
        """After unloading, wait for workers to gc and report FRESH free RAM (via the next
        heartbeat) so the planner budgets against true free memory, not RAM the old model
        still held. Waits until every alive, capable node has heartbeated since this call
        began (so free_mem_gb is post-unload), capped at `timeout`."""
        since = time.time()
        await asyncio.sleep(1.0)   # give workers a moment to gc / release mmaps
        while time.time() - since < timeout:
            nodes = [n for n in registry.alive_sorted() if n.can_infer]
            if nodes and all(n.last_heartbeat >= since for n in nodes):
                return
            await asyncio.sleep(0.5)
