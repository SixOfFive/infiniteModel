"""EngineLoadMixin: relocated Engine methods (m4c152 code-split). BODIES ARE BYTE-IDENTICAL
to the originals in server.py; their module globals (registry, log_activity, ModelSpec,
ENGINE_CONFIG …) are injected at startup by state.bind() — see state.py. Composed back
into the live class via ``class Engine(EngineLoadMixin, …)`` in server.py, so ``self.*`` resolves
across all mixins by MRO. Controller-only leaf module; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

# #cache-reserve (audit #16): co-located controller RAM reserve for a CACHE-SERVED load.
# CONTROLLER_RAM_RESERVE_GB (server.py, default 20, #78) is sized for streaming the FULL bf16
# from the controller's disk (the 426 GB minimax serve) — but a load served from a VERIFIED
# pre-quantized _shards/ cache reads only the small pre-packed units (~0.27x bf16 for int4,
# e.g. ~18 GB vs ~70 GB for a 30B MoE), so charging the full reserve wastes 12-14 GB of
# plannable RAM on the controller boxes for the COMMON path (autoload_quant=int4 +
# #cache-on-first-load). Reserve a floor (controller process + serve buffers + OS-cache
# headroom) scaled by the ACTUAL cache read volume — big int4 caches can still be 50+ GB of
# reads. Defined here (not env) because these only tune the cache-served DISCOUNT; the bf16
# reserve stays env-tunable in server.py, and env=0 still disables both (min() below).
CACHE_CTRL_RESERVE_FLOOR_GB = 6.0    # never reserve less than this on a cache-served load
CACHE_CTRL_RESERVE_READ_FRAC = 0.10  # + scale: 10% of the cache read volume past the floor


def _peer_claimed_host(hostname: str) -> bool:
    """#federation Phase 5: is a PEER controller already using this node?

    Imported lazily and failure-tolerant on purpose: peers.py is a newer leaf, so a controller
    mid per-file self-update (this engine_load.py newer than peers.py) must still plan normally.
    ANY problem answers False = "not claimed" = behave exactly as before federation existed."""
    try:
        import peers
        return peers.is_peer_claimed(hostname)
    except Exception:      # noqa: BLE001 — never let federation break placement
        return False


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

    def _cap_terminal(self) -> bool:
        """#at-capacity: True when NOTHING could ever be auto-evicted to make room — auto-unload
        is off, or every resident is no_unload-pinned. Busy-ness is transient (a busy victim goes
        idle and a retry succeeds); the off switch and pins only clear when an operator acts, so
        a capacity failure in this state is PERMANENT and serving must NOT promise Retry-After.
        (No residents at all is also terminal: there is nothing to evict, the model simply does
        not fit the pool.)"""
        if not bool(ENGINE_CONFIG.get("auto_unload", True)):
            return True
        no_unload = set(ENGINE_CONFIG.get("no_unload_models") or {})
        return not any((m.base or m.friendly) not in no_unload for m in self.models.values())

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
            # #autoload-herd: ONE shared load task per cold model. Every concurrent request — and
            # every Retry-After-honoring retry that lands while the load is still running — awaits
            # the SAME task instead of calling load() again. A duplicate load() would queue on the
            # engine lock and, on finally acquiring it, find the model resident and treat it as an
            # explicit RELOAD (unload+reload churn, serially, once per stacked request — killing
            # the very generation the first request just started). shield() keeps the shared load
            # alive when the request that spawned it disconnects: the other waiters still need it,
            # and the finished load serves every future request (Ollama semantics).
            t = self._autoload_tasks.get(friendly)
            if t is None:
                t = asyncio.create_task(self._autoload_shared(friendly, ctx, cpu_only))
                self._autoload_tasks[friendly] = t
                t.add_done_callback(lambda _t, _f=friendly: self._autoload_tasks.pop(_f, None))
            return await asyncio.shield(t)
        raise ValueError(f"model '{friendly}' is not loaded — load it first")

    async def _autoload_shared(self, friendly: str, ctx: int, cpu_only: bool) -> LoadedModel:
        """Body of one request-triggered auto-load; ensure_loaded runs this as the per-model
        shared task (#autoload-herd) so N concurrent cold requests trigger exactly one load."""
        m = self.models.get(friendly)   # lost a check-then-spawn race -> already resident
        if m is not None:
            m.last_used = time.time()
            return m
        # #adopt: a just-restarted controller may be mid-ADOPTING this very model from the
        # workers that kept its shards — wait briefly for the adoption instead of racing a
        # duplicate placement into VRAM the kept shards still occupy. Bounded (~10s): if the
        # coverage never completes, fall through to a normal auto-load (the sweep frees the
        # kept shards after grace, so the reload plans against honest numbers).
        _tid = MODELS[friendly][0] if friendly in MODELS else friendly
        _pool = self.__dict__.get("_adopt_pool") or {}
        if _tid in _pool:
            for _ in range(40):
                await asyncio.sleep(0.25)
                m = self.models.get(friendly)
                if m is not None:
                    m.last_used = time.time()
                    print(f"[adopt] auto-load of {friendly} satisfied by adoption")
                    return m
                if _tid not in (self.__dict__.get("_adopt_pool") or {}):
                    break
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
                                   proportional=_prop, gpu_spread=_gpus, auto=True)
        except Exception as e:
            # #at-capacity: a CapacityError verdict only gets WORSE at a bigger size — bf16 is
            # strictly larger than int4/int8, and the resident-cap check doesn't depend on size
            # at all — so the bf16 fallback can only fail the same way while doubling the noise.
            if aq != "none" and not isinstance(e, CapacityError):
                log_activity(f"{friendly}: auto-load at {aq} failed ({e!r}) -> retry at bf16")
                return await self.load(friendly, use_ctx, consolidate=_cons, prefer_vram=_pv,
                                       quant="none", cpu_only=cpu_only, spread=_spread,
                                       proportional=_prop, gpu_spread=_gpus, auto=True)
            raise

    async def _precompile_int4(self, friendly: str, quant: str, tp: int) -> None:
        """#cache-on-first-load: for an int4 load with NO shard cache yet, BUILD it first
        (blocks until written) so THIS load — and every future load — serves the small pre-packed
        layers instead of streaming full bf16 and re-quantizing on the fly. No-op when that tier's
        cache already exists, when quant is anything but int4, or for tp>1 (its dispatch path
        doesn't read the whole-layer cache). int4 ONLY by design: an int2 cache is never auto-built
        on first load (RTN-int2 output is collapsed until the calibrated packer lands) — an
        operator builds one deliberately via the dashboard Precache button / POST /compile_shards,
        and an EXISTING int2 cache still serves (the serve gate is separate). Reuses the
        /compile_shards SUBPROCESS (deprioritized, GIL-safe — an in-process compile would starve
        the event loop / drop live generations). Non-fatal: ANY failure falls through to the
        normal cold load. Shared by the /load route AND the auto-load path (ensure_loaded) so
        request-triggered loads compile-on-first-load identically."""
        if not (quant == "int4" and tp <= 1):
            return
        try:
            import shard_compile as _sh   # code-split Inc 9: shard_cache_status moved
            import urllib.parse as _up
            _ctgt = MODELS[friendly][0] if friendly in MODELS else friendly
            _cdir = await asyncio.to_thread(_controller_model_dir, _ctgt)
            if _cdir and (_is_diffusers_dir(_cdir) or _is_kokoro_dir(_cdir)
                          or os.path.isdir(os.path.join(_cdir, "ace_step_transformer"))):
                return   # #t2i/#t2a/#tts: image/audio checkpoints have no LLM shard cache to compile
            # #embedding: an encoder / sentence-embedding checkpoint (nomic-embed, BERT, …) has no
            # 'model.embed_tokens.weight' — an int4 shard compile ALWAYS fails (KeyError) and, because
            # such models are typically persist/no_unload, RE-FIRES on every auto-load / re-adopt,
            # spamming /compile_shards 400s in the error log. Mirror model_store's is_embedding
            # classification on the on-disk config and skip (no LLM shard cache to build). Gotcha
            # documented long ago for the manual precompile sweep — this closes the auto-load path too.
            if _cdir:
                # GENERIC across all embedding models: is_embedding matches any encoder /
                # sentence-embedding arch (nomic_bert, bert, roberta, mpnet, bge/gte/e5, …), not just
                # nomic. Belt-and-suspenders name check covers an oddly-arch'd embedder is_embedding
                # might miss (skipping a precompile is harmless — the model still cold-loads).
                _esp = await asyncio.to_thread(_spec_from_config, _cdir, friendly)
                if ((_esp is not None and getattr(_esp, "is_embedding", False))
                        or "embed" in str(friendly).lower()):
                    return
            _cst = await asyncio.to_thread(_sh.shard_cache_status, _cdir) if _cdir else {}
            if _cdir and not (_cst.get(quant) or {}).get("ok"):
                log_activity(f"{_ollama_name(friendly)}: no {quant} shard cache — building it now so this "
                             "and every future load serve pre-packed (first load is slower)…")
                _curl = (f"http://127.0.0.1:{ARGS.http_port}/compile_shards"
                         f"?model={_up.quote(friendly)}&quant={quant}")

                def _build_cache():
                    import urllib.request as _u
                    with _u.urlopen(_u.Request(_curl, method="POST"), timeout=10800) as _r:
                        return _r.read()

                await asyncio.to_thread(_build_cache)
        except Exception as _ce:
            log_activity(f"{_ollama_name(friendly)}: pre-load cache build skipped ({_ce!r}) — cold load")

    def _lan_visible_host(self, host, receiver, link):
        """#loopback-nexthop: translate a LOOPBACK data-plane address to one the RECEIVING worker
        can actually reach. A worker co-located with the controller advertises 127.0.0.1 (fastest
        for the controller's OWN dials, see _dial_host) — but as a next-hop / TP-root handed to a
        REMOTE worker, a loopback dials that remote worker ITSELF (self-loop: stage outputs feed
        back into stage inputs — the 2026-07-09/10 wedge-storm engine). Returns `host` unchanged
        when it isn't loopback or when the receiver is ALSO controller-local (on-box loopback is
        correct + fastest); otherwise our address on the receiver's control link (its sockname),
        falling back to this box's first LAN IP."""
        _h = str(host or "")
        if not _h.startswith(("127.", "::1", "localhost")):
            return host
        _rh = str(getattr(receiver, "data_host", "") or "")
        if _rh in _LOCAL_IPS or _rh.startswith(("127.", "::1")):
            return host                       # receiver shares this box — loopback is right
        _lan = None
        with contextlib.suppress(Exception):
            _sn = link.writer.get_extra_info("sockname") if link is not None else None
            _lan = _sn[0] if _sn else None
        if not _lan or _lan.startswith(("127.", "::1")):
            _lan = next((ip for ip in sorted(_LOCAL_IPS) if not ip.startswith("127.")), None)
        return _lan or host

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

    def _kvslots_clamp(self, kv_slots: int, tp: int, kv_quant: str, kv_offload: bool,
                       model_dir: str) -> tuple:
        """#kv-slots load-time hard gates: returns (effective_kv_slots, reason). Only the
        plain-DynamicCache/#kv-prealloc shard branch supports slot-keyed caches, so anything
        else clamps to 1 (logged by the caller, never an error — the load still serves):
        - tp>1: the TP rank-0 broadcast carries no slot id (peers would desync);
        - kv_quant: the TurboQuant cache is a single-stream custom cache;
        - kv_offload: the OffloadedCache prefetch machinery is single-stream;
        - hybrid / per-type / omni / qwen2.5-VL architectures (layer_types, thinker_config,
          mrope3d model_type — the SAME conservative config.json sniff as _pipefill_arch_ok):
          linear-attention recurrent state and per-type KV are not slot-keyed;
        - an unreadable config.json clamps too (conservative — never guess).
        Also clamps to the 1..8 band (measured q-curves: CUDA flat to 16, Strix regime step at
        16; 8 is the validated opt-in ceiling)."""
        kv_slots = max(1, min(8, int(kv_slots or 1)))
        if kv_slots <= 1:
            return 1, ""
        if tp > 1:
            return 1, "tensor-parallel (the TP broadcast carries no slot id)"
        if (kv_quant or "none") != "none":
            return 1, f"kv_quant={kv_quant} (TurboQuant cache is single-stream)"
        if kv_offload:
            return 1, "kv_offload (OffloadedCache is single-stream)"
        try:
            with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
                _cfg = json.load(fh)
            _tc = _cfg.get("text_config", _cfg)
            _mt = str(_cfg.get("model_type") or _tc.get("model_type") or "").lower()
            if (_tc.get("layer_types") or []) or _cfg.get("thinker_config") is not None \
                    or _mt in ("qwen2_5_vl", "qwen2_5_vl_text"):
                return 1, ("hybrid/per-type/mrope architecture (recurrent or per-type state "
                           "is not slot-keyed)")
        except Exception:
            return 1, "architecture sniff failed (conservative)"
        return kv_slots, ""

    def _kvslots_cap_check(self, kv_slots: int, stages, node_by_id) -> None:
        """#kv-slots wire-cap gate (load-time, all-or-nothing — the pipefill doctrine): slots>1
        only when EVERY chain node advertises 'kvslots'. An old worker would write every slot
        into its single cache (silent cross-request KV corruption), so a chain with any older
        node REFUSES the load with a clear message instead of degrading."""
        if int(kv_slots or 1) <= 1:
            return
        _nocap = sorted({(node_by_id[s.node_id].hostname if s.node_id in node_by_id
                          else s.node_id)
                         for s in stages if "kvslots" not in registry.node_caps(s.node_id)})
        if _nocap:
            raise RuntimeError(
                f"kv_slots={kv_slots} needs the 'kvslots' wire cap on EVERY chain node — "
                f"missing on {', '.join(_nocap)} (stale worker code; self-update those nodes "
                f"or load with kv_slots=1)")

    async def load(self, friendly: str, ctx: int, consolidate: bool = True,
                   prefer_vram: bool = True, quant: str = "none", tp: int = 1,
                   cpu_only: bool = False, reg_key: Optional[str] = None,
                   exclude_nodes: Optional[set] = None, replica_idx: int = 0,
                   spread: bool = False, proportional: bool = False,
                   force: bool = False, moe_offload: bool = False,
                   gpu_spread: bool = False, pin_host: str = "",
                   kv_quant: str = "", kv_offload: bool = False,
                   kv_slots: int = 1,
                   default_temp: Optional[float] = None,
                   default_min_p: Optional[float] = None,
                   requested_by: str = "",
                   draft_gpu: bool = False,
                   draft_margin_gb: float = 4.0,
                   t2i_offload: bool = False,
                   auto: bool = False) -> LoadedModel:
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
                                         kv_offload=kv_offload, kv_slots=kv_slots,
                                         default_temp=default_temp,
                                         default_min_p=default_min_p,
                                         requested_by=requested_by,
                                         draft_gpu=draft_gpu, draft_margin_gb=draft_margin_gb,
                                         t2i_offload=t2i_offload,
                                         auto=auto,
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
                   kv_slots: int = 1,
                   default_temp: Optional[float] = None,
                   default_min_p: Optional[float] = None,
                   requested_by: str = "",
                   draft_gpu: bool = False,
                   draft_margin_gb: float = 4.0,
                   t2i_offload: bool = False,
                   auto: bool = False,
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
                # #t2i-serve: a registered DIFFUSERS checkpoint (image generation —
                # model_index.json, no top-level config.json) never builds an LLM spec;
                # it loads via the single-node image path instead (task #37).
                _tgt = MODELS[friendly][0] if friendly in MODELS else friendly
                _d = _local_model_dir(_tgt)
                # #t2a-serve: an ACE-Step music checkpoint (ace_step_transformer/ subfolder, no
                # model_index.json) isn't a diffusers-pipeline dir — route it to the audio path.
                if _d and os.path.isdir(os.path.join(_d, "ace_step_transformer")):
                    # #t2a-offload-default: ACE-Step defaults to OFFLOAD (weights RAM-resident, the
                    # DiT hops to the GPU per render) — it holds ZERO resident VRAM, can't OOM a card
                    # on load, and leaves the GPU free for LLMs (VRAM is precious). Renders still run
                    # on the GPU (~seconds), so speed is unaffected. Opt back into GPU-resident via
                    # /config?t2a_offload_default=0; cpu_only forces its own (offload-off) path.
                    return await self._load_t2a_locked(
                        friendly, _tgt, reg_key or friendly, quant, replica_idx=replica_idx,
                        offload=(t2i_offload or bool(ENGINE_CONFIG.get("t2a_offload_default", True))),
                        cpu_only=cpu_only)
                # #tts-serve: a Kokoro speech checkpoint (kokoro-v1_0.pth + voices/, no
                # safetensors, no model_index.json) loads via the single-node speech path.
                if _d and _is_kokoro_dir(_d):
                    return await self._load_tts_locked(friendly, _tgt, reg_key or friendly,
                                                       replica_idx=replica_idx)
                if _d and _is_diffusers_dir(_d):
                    return await self._load_t2i_locked(friendly, _tgt, reg_key or friendly,
                                                       quant, replica_idx=replica_idx,
                                                       offload=t2i_offload, force=force)
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
            _tctx = _train_ctx_from_dir(model_dir, spec)   # native training window (max_position_embeddings)
            if ctx <= 0:
                ctx = _tctx
                print(f"[load] ctx=auto -> training context {ctx}")
            elif _tctx > 0 and ctx > _tctx:
                # #ctx-ceiling: NEVER reserve more KV than the model was trained for — a 4k window on a
                # 2k-trained model (e.g. nomic) just wastes pool on context the model can't attend to.
                # Clamp DOWN to the training context; applies to an explicit ctx AND autoload_ctx alike.
                # One-directional: a SMALLER request is always honored unchanged.
                print(f"[load] ctx {ctx} > training context {_tctx} -> clamped to {_tctx}")
                ctx = _tctx
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
            # #kv-slots: hard-gate to C=1 for anything but the plain-DynamicCache branch
            # (tp / kv_quant / kv_offload / hybrid / per-type / omni / mrope3d), clamp to <=8,
            # then bake C into the SPEC's KV bytes so the whole plan (plan_pipeline, _fit_ctx,
            # the ctx guardrails, colo_need, /status kv_reserved) reserves C x per-stream KV —
            # a load whose C*KV doesn't fit FAILS here (CapacityError) or at the worker's own
            # xC kv_reserve_probe (KV_RESERVE_OOM -> replan), never OOMs mid-decode.
            _kvs_req = max(1, min(8, int(kv_slots or 1)))
            kv_slots, _kvs_why = self._kvslots_clamp(kv_slots, tp, kv_quant, kv_offload,
                                                     model_dir)
            if _kvs_req > 1 and kv_slots == 1 and _kvs_why:
                log_activity(f"{friendly}: kv_slots={_kvs_req} -> 1 — {_kvs_why} is hard-gated "
                             f"to a single KV stream")
            # (spec.for_kv_slots is applied AFTER the TP branches below — a TP/auto-TP load
            # never carries slots, so its plan must see the unscaled 1-stream KV figure.)
            await self.ensure_data_listener()
            log_activity(f"load {friendly}: planning (ctx={ctx}, quant={quant}"
                         + (f", kv_quant={kv_quant}" if kv_quant != "none" else "")
                         + (", KV-OFFLOAD (KV cache in system RAM)" if kv_offload else "")
                         + (f", kv_slots={kv_slots} (KV reserved x{kv_slots})" if kv_slots > 1 else "")
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

            # #kv-slots: bake C into the SPEC's KV bytes here — past the TP/auto-TP branches
            # (both are hard-gated to C=1 and must plan the unscaled 1-stream figure), before
            # the pipeline planner — so plan_pipeline / _fit_ctx / the ctx guardrails /
            # colo_need / the /status kv_reserved card all reserve C x per-stream full-ctx KV.
            spec = spec.for_kv_slots(kv_slots)

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
                    _term = self._cap_terminal()   # #at-capacity: retryable (busy) vs permanent
                    _why = ("auto-unload is off" if not auto_unload else
                            "the rest are pinned no_unload" if _term else
                            "all are busy serving requests")
                    raise CapacityError(
                        f"at the max of {max_loaded} resident model(s) and {_why}"
                        f" — unload one before loading '{friendly}'", terminal=_term)
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
            # #cache-reserve (audit #16) + #shard-cache Inc 2: decide ONCE, BEFORE placement,
            # whether this load SERVES FROM a complete pre-quantized shard cache. The decision
            # used to live after plan_pipeline (only dispatch needed it), so the planner could
            # not know and charged the co-located controller box the FULL bf16-stream reserve on
            # EVERY load. Hoisting is ~free (_shard_cache_ok memoizes on manifest mtime+size;
            # quant/model_dir are known here) and also spares the eviction-retry loop the
            # re-checks. Conservative default: any doubt (no cache, check error, quant is
            # none/int8) keeps the full bf16 reserve. Re-verified at dispatch (vanish guard).
            _cache_quant = ""
            _cache_read_gb = 0.0
            if quant in ("int4", "int2"):
                try:
                    if model_dir and await asyncio.to_thread(_shard_cache_ok, model_dir, quant):
                        _cache_quant = quant

                        def _cache_tree_bytes(_d=os.path.join(model_dir, "_shards", quant)):
                            _t = 0
                            for _r, _ds, _fs in os.walk(_d):
                                for _f in _fs:
                                    with contextlib.suppress(OSError):
                                        _t += os.path.getsize(os.path.join(_r, _f))
                            return _t
                        _cache_read_gb = (await asyncio.to_thread(_cache_tree_bytes)) / GB
                        log_activity(f"{friendly}: serving from {quant} shard cache "
                                     f"(~{_cache_read_gb:.1f} GB pre-packed; skip bf16 stream "
                                     f"+ per-layer re-quant)")
                except Exception as _ce:
                    log_activity(f"{friendly}: shard-cache check failed ({_ce!r}) -> bf16 stream")
                    _cache_quant, _cache_read_gb = "", 0.0
            # Cache-served: the controller reads/serves ~0.27x the bf16 volume, so a small
            # floor + a slice of the actual read volume protects its process/serve buffers.
            # min() = never MORE than the configured bf16 reserve (env 0 still disables both).
            _ctrl_reserve_gb = CONTROLLER_RAM_RESERVE_GB
            if _cache_quant:
                _ctrl_reserve_gb = min(CONTROLLER_RAM_RESERVE_GB,
                                       max(CACHE_CTRL_RESERVE_FLOOR_GB,
                                           _cache_read_gb * CACHE_CTRL_RESERVE_READ_FRAC))
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
                    # #federation Phase 5: a PEER controller is already using this node. Two
                    # controllers planning against the same node's memory is the double-booking that
                    # OOMs it — each is blind to the other's reserved-but-unfaulted KV and in-flight
                    # reservations. Ownership is therefore EXCLUSIVE: skip a node a healthy peer
                    # claims, unless we hold a shard on it ourselves (then it is ours and the peer's
                    # claim is stale). Off via /config?respect_peer_claims=0.
                    if _peer_claimed_host(n.hostname):
                        continue
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
                    # #vram-reusable: credit back the worker's VACANT allocator pool — device
                    # counters report it as used, but a new load in that worker allocates from it
                    # first (only a worker restart returns it to the OS; big on ROCm after churn:
                    # om3nbox measured ~13 GB "used" that was actually empty pool).
                    live_free = self._node_live_free_vram_gb(n, res_vram=_res_vram)
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
                    # to read+serve the bf16 stream (OS cache + serving buffers) WHILE this worker
                    # builds its shard — else the box over-commits and the worker OOM-drops mid-load (the
                    # beast minimax crash). data_host in _LOCAL_IPS == same machine as the controller.
                    # #cache-reserve (audit #16): _ctrl_reserve_gb (hoisted above the loop) is the FULL
                    # bf16 reserve only when this load actually streams bf16; a cache-served load reads
                    # ~0.27x, so it charges the small cache-scaled reserve instead — freeing 12-14 GB of
                    # plannable RAM on a DISCRETE co-located box (beast) for the common int4-cache path.
                    # #cache-reserve-unified: NOT on a unified-memory box. There the controller reserve
                    # is not spare RAM headroom — it is the SAME physical pool #17's clamp then hands to
                    # VRAM (it subtracts this very node_reserve_gb), so relaxing it for a cache-served
                    # load directly licenses ~14 GB MORE weight commitment. Observed live on om3nbox
                    # (InferenceEngine packed to 54/60 GB by stacked int4 auto-loads + evict/reload
                    # churn, its 5090 sibling being disabled so every load lands on the one box).
                    # Keep the full bf16 reserve on unified nodes — RAM spent there IS VRAM lost.
                    if n.data_host in _LOCAL_IPS:
                        node_reserve_gb += (CONTROLLER_RAM_RESERVE_GB if self._is_unified_node(n)
                                            else _ctrl_reserve_gb)
                    ram_for_resident = (n.eff_ram_gb - node_reserve_gb
                                        - _res_ram.get(n.node_id, 0) / GB)   # #parallel-load reserve
                    if ram_for_resident <= 0:
                        continue   # too small for even one layer's build transient -> skip
                    usable = ram_for_resident + free_vram   # resident RAM budget (+ VRAM after residents/floor)
                    # #unified-mem (audit #17): on an APU (om3nbox gfx1151, steamdeck) the line
                    # above double-counts — free GTT bytes ARE free-RAM bytes — so clamp to the
                    # LIVE physically-free pool minus the same reserves charged above (transient +
                    # controller reserve in node_reserve_gb, the VRAM floor, the co-located draft
                    # slice, both in-flight reservation ledgers — one pool, so the RAM and VRAM
                    # claims both come out of it). No-op on discrete-GPU/CPU nodes.
                    usable, free_vram = self._unified_mem_clamp(
                        n, usable, free_vram,
                        node_reserve_gb + PLAN_VRAM_FLOOR_GB
                        + (draft_reserve_gb if n.data_host in _LOCAL_IPS else 0.0)
                        + (_res_ram.get(n.node_id, 0) + _res_vram.get(n.node_id, 0)) / GB)
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
                        raise CapacityError("no room for the new model and resident model(s) are "
                                            + ("busy serving" if auto_unload else "kept (auto-unload off)"),
                                            terminal=self._cap_terminal())   # #at-capacity
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
                        raise CapacityError((plan.error or "planning failed")
                                            + (f" [kv_slots={kv_slots}: KV is reserved x{kv_slots} — "
                                               f"a smaller kv_slots or ctx may fit]" if kv_slots > 1
                                               else "")
                                            + f" — even ctx {CTX_AUTOFIT_FLOOR} won't fit; the model's "
                                              "weights exceed the usable pool (free memory or use a "
                                              "smaller quant)" + (
                            "; resident model(s) busy serving" if (self.models and auto_unload)
                            else "" if self.models else ""),
                            terminal=self._cap_terminal())   # #at-capacity
                stages = plan.stages
                # #kv-slots: all-or-nothing wire-cap gate — REFUSE (clear message) if any chosen
                # chain node lacks 'kvslots'; an old worker would fold every slot into its single
                # cache (silent cross-request KV corruption), so degrading is not an option.
                self._kvslots_cap_check(kv_slots, stages, node_by_id)
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
                # #render-oom-guard v2 (2026-07-18, user: "if it can't go to VRAM, send it to RAM —
                # the system should still load the requested model"): an AUTO-load whose weights
                # MOSTLY spill to CPU (cpu_weight_frac > 0.5) on the controller's OWN co-located box,
                # beside resident model(s) there, used to be REFUSED outright — the om3nbox
                # coder:32b-crashes-the-render incident (2026-07-17) showed a blind CPU spill
                # OOM-killing the co-located worker and dropping a live render. But refusing also
                # never loads the model. So DON'T blanket-refuse: LOAD IT INTO RAM when the
                # co-located box has the physically-free RAM to hold the model beside the residents,
                # and refuse ONLY when it genuinely wouldn't fit — a real OOM loads NOTHING and drops
                # the render too (strictly worse). free_mem_gb is live (already down for residents +
                # any active render); keep CONTROLLER_RAM_RESERVE_GB (the co-located controller needs
                # it to stream+serve while the worker builds) + RAM_SAFETY_GB. UNIFIED-MEMORY-SAFE:
                # on a Strix-Halo APU the "VRAM" slice is carved from the SAME pool as RAM, so the
                # need is sized by the model's FULL footprint (weights + full-ctx KV) on each
                # co-located node, not just the RAM-spilled part (discrete GPUs, RAM >> this, pass
                # trivially). Empty _colo_need (spill lands only on a REMOTE capacity node, e.g. the
                # dell CPU worker) falls through and loads as before; explicit /load (auto=False) and
                # idle boxes never reach here.
                if auto and self.models and assess.get("cpu_weight_frac", 0.0) > 0.5:
                    _ram_margin = CONTROLLER_RAM_RESERVE_GB + RAM_SAFETY_GB
                    _colo_need: dict = {}          # node_id -> full footprint bytes on a co-located node
                    for st in stages:
                        nd = node_by_id.get(st.node_id)
                        if nd is None or nd.data_host not in _LOCAL_IPS:
                            continue
                        b = st.num_layers * spec.per_layer_weight_bytes
                        if st.has_embed:
                            b += spec.embed_bytes
                        if st.has_head:
                            b += spec.head_bytes + spec.final_norm_bytes
                        b += st.num_layers * spec.kv_bytes_per_layer(ctx)
                        _colo_need[st.node_id] = _colo_need.get(st.node_id, 0) + b
                    _over = [(node_by_id[_nid].hostname, _b / GB,
                              max(0.0, node_by_id[_nid].free_mem_gb - _ram_margin))
                             for _nid, _b in _colo_need.items()
                             if _b / GB > node_by_id[_nid].free_mem_gb - _ram_margin]
                    if _over:
                        _h, _n, _a = _over[0]
                        raise CapacityError(
                            f"'{friendly}' needs ~{_n:.0f} GB (weights+KV) on {_h} but only ~{_a:.0f} "
                            f"GB is safely free there beside the {len(self.models)} resident model(s) "
                            f"— loading it would risk an out-of-memory that drops them mid-serve (a "
                            f"live render among them). Free the box, use a smaller quant, or load on "
                            f"a node with more room.", terminal=True)
                    if _colo_need:   # fits RAM beside the residents -> load it INTO RAM (slow, but loads)
                        _h0 = next(iter(_colo_need))
                        log_activity(
                            f"{friendly}: {assess['cpu_weight_frac']*100:.0f}% of weights won't fit "
                            f"VRAM here — loading ~{assess.get('cpu_weight_gb', 0):.0f} GB into RAM "
                            f"beside {len(self.models)} resident model(s) "
                            f"({node_by_id[_h0].free_mem_gb:.0f} GB free; slow CPU decode).")
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
                # #shard-cache Inc 2 (serve-from-cache): a VERIFIED cache makes every worker fetch
                # PRE-PACKED int4/int2 layers (cache=int4) instead of streaming the full bf16 +
                # re-quantizing — the big win for MoE/large loads (~18 GB cache vs ~70 GB bf16).
                # int4 + pipeline only: TP slices weights non-contiguously (its own dispatch path,
                # never reaches here). The controller falls back to bf16 PER UNIT if a cache file
                # is missing, and an old worker that ignores the `cache` key streams bf16 — safe.
                # #cache-reserve (audit #16): the decision itself (_cache_quant) is HOISTED above
                # the placement loop now — the co-located controller RAM reserve is sized by it.
                # RE-VERIFY here at dispatch (memoized on manifest mtime+size, ~free): a cache
                # that VANISHED since planning must NOT be streamed against the small cache-sized
                # reserve — the per-unit bf16 fallback would re-create the exact #78 over-commit
                # the reserve exists to prevent (narrow race, but the failure is an OOM-dropped
                # worker). Instead: restore the full bf16 reserve and REPLAN (the attempt loop
                # already re-enters cleanly from here). A cache that APPEARED since planning
                # (a concurrent /compile_shards finished) is adopted as-is — the plan reserved
                # MORE than needed, which is safe.
                if quant in ("int4", "int2"):
                    _now_cached = ""
                    try:
                        if model_dir and await asyncio.to_thread(_shard_cache_ok, model_dir, quant):
                            _now_cached = quant
                    except Exception as _ce:
                        log_activity(f"{friendly}: shard-cache re-check failed ({_ce!r}) "
                                     f"-> bf16 stream")
                    if _cache_quant and not _now_cached:
                        log_activity(f"{friendly}: ⚠ {quant} shard cache vanished after planning — "
                                     f"replanning with the full bf16 controller reserve "
                                     f"({CONTROLLER_RAM_RESERVE_GB:.0f} GB, was "
                                     f"{_ctrl_reserve_gb:.0f} GB cache-sized)")
                        _cache_quant, _cache_read_gb = "", 0.0
                        _ctrl_reserve_gb = CONTROLLER_RAM_RESERVE_GB
                        continue
                    if _now_cached and not _cache_quant:
                        log_activity(f"{friendly}: {quant} shard cache appeared after planning — "
                                     f"serving pre-packed (plan already reserved the larger "
                                     f"bf16 figure; safe)")
                    _cache_quant = _now_cached
                    # #38: int2 WITHOUT a valid calibrated cache must FAIL LOUD, never fall back.
                    # The bf16-stream fallback (correct for int4 — cold quant == cache by
                    # construction) would silently serve load-time RTN int2 = token salad. int2
                    # stays explicit-compile by policy (never auto-built on first load), so the
                    # operator gets the exact next step instead of a garbage model.
                    if quant == "int2" and not _cache_quant:
                        raise RuntimeError(
                            "int2 needs a valid CALIBRATED shard cache (packer v2 gptq) — this "
                            "model's int2 cache is missing, stale (v1 RTN), or corrupt. Build it "
                            "via the dashboard's 'Compile int2' or POST /compile_shards?model="
                            f"{_ollama_name(friendly)}&quant=int2, then load again")
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
                    # #loopback-nexthop: a worker CO-LOCATED with the controller advertises a
                    # LOOPBACK data_host (fastest for the controller's own stage0 dials) — but
                    # handed verbatim to a REMOTE stage as its next hop, "127.0.0.1:50200" dials
                    # the REMOTE WORKER ITSELF: every stage output loops straight back into its
                    # own input (stage0 then eats its own bf16 hidden as "ids"), and on the OLD
                    # code even the error frames cycled on the self-hop forever — the engine of
                    # the 2026-07-09/10 qwen2.5-vl wedge storm that fed beast's kernel panic
                    # (placements failing EXACTLY when a remote node's next stage sat on beast).
                    next_host = self._lan_visible_host(next_host, nd, link)
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
                                        + getattr(nd, "vram_reusable_gb", 0.0)   # #vram-reusable
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
                        "cache": _cache_quant,       # #shard-cache Inc 2: '' | 'int4' | 'int2' -> fetch pre-packed cache
                        "quant": quant,              # 'none' | 'int8' | 'int4' | 'int2' (load-time choice)
                        "kv_quant": kv_quant,        # #172 TurboQuant KV preset (none|turbo2|turbo3|turbo4)
                        "kv_offload": kv_offload,    # #kv-offload: KV cache in system RAM (OffloadedCache)
                        "kv_slots": kv_slots,        # #kv-slots: C per-request KV streams (worker reserves xC)
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
                # #cache-reserve (audit #16): a CACHE-SERVED load reads the pre-packed units, not
                # the full bf16 — #100's "budget PRE-quant bytes" rule is about a plain int4 load
                # (which really does stream bf16 then quantize); a VERIFIED cache does neither.
                # Size on the cache read volume x2 (headroom for scattered per-unit bf16
                # fallbacks if individual cache files go missing mid-load), never above the bf16
                # figure; the [15 min, 4 h] clamp below still applies. Tighter timeout = a wedged
                # cache-served load fails in minutes, not hours (no capacity effect either way).
                if _cache_quant and _cache_read_gb > 0:
                    read_bytes = min(read_bytes, int(_cache_read_gb * 2 * GB))
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
                kv_slots=kv_slots,
                default_temperature=default_temp, default_min_p=default_min_p,
                stage0_writer=stage0_writer, last_used=now,
                stage0_dial=_s0_dial, last_send_ts=now)   # #stage0-stale-reconnect: how to re-dial + freshness clock
            self._init_slot_pool(lm)   # #kv-slots: semaphore + slot pool (no-op at C=1)
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

    async def _load_t2i_locked(self, friendly: str, target_id: str, reg_key: str,
                               quant: str = "int4", replica_idx: int = 0,
                               offload: bool = False, force: bool = False) -> LoadedModel:
        """#t2i-serve (task #37): load a DIFFUSERS image-generation checkpoint (Qwen-Image
        class) WHOLE onto ONE controller-CO-LOCATED GPU worker — the diffusion sibling of
        _load_embedding_locked. v1 constraints, by design: the worker must share this box's
        filesystem (it reads the model dir directly and hands back PNG paths — the only GPUs
        that can host the 20B DiT in this fleet are on controller boxes anyway), and the DiT
        serves with the GATE-TESTED mixed-edge recipe: middle blocks RTN int4 g128, the first
        + last `edge` blocks bf16 (edge 2 ~= bf16 quality at ~13.5 GB; edge 1 is the tighter
        fallback a 16 GB card may need). Text encoder runs on the worker's CPU (encode-once).
        MUST be called with self.lock held (reached only from load())."""
        model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
        # int8/int2 have no meaning for the DiT path — coerce to the tested tier, loudly.
        if quant in ("int8", "int2"):
            log_activity(f"{_ollama_name(friendly)}: {quant} is not a t2i tier — using int4 (mixed-edge)")
            quant = "int4"
        # #t2i-offload (v2): the DiT rests bf16 in system RAM and accelerate streams each block
        # to the GPU just-in-time per forward — VRAM peak drops to transients (~blocks +
        # activations + VAE), so the card's residents stay loaded. bf16-only: the int4 fused
        # kernels are prepared per-device and don't survive block hopping; RAM is the budget
        # instead (~weights + slack). Slower per step (PCIe transfer each pass) — the
        # no-eviction trade for tight cards like beast's 16 GB.
        if offload:
            quant = "none"
        # Idempotent re-load (mirror the pipeline-path guard in _load_impl, which the t2i branch
        # RETURNS before ever reaching): a duplicate /load for an already-resident image model in
        # the SAME placement + precision is a NO-OP — return the live copy instead of evicting
        # ~tens of GB and re-quantizing/re-streaming the DiT (the `reg_key in self.models` block
        # below rebuilds UNCONDITIONALLY otherwise, so an accidental second Load click or a repeat
        # API load paid a full rebuild). The resident LoadedModel.quant encodes the mode:
        # 'bf16-off' (offload) | 'none' (GPU bf16) | 'int4-e{n}' (GPU int4). Edge count is NOT
        # compared — a redundant load shouldn't rebuild just to renegotiate edge. A DIFFERENT
        # precision/placement, or force=1 (the stuck-load override), still reloads.
        _resident = self.models.get(reg_key)
        if _resident is not None and not force and getattr(_resident, "is_t2i", False):
            _rq = _resident.quant or ""
            _have = ("bf16-off" if _rq == "bf16-off"
                     else "bf16-gpu" if _rq == "none"
                     else "int4-gpu" if _rq.startswith("int4") else _rq)
            _want = "bf16-off" if offload else ("bf16-gpu" if quant == "none" else "int4-gpu")
            if _have == _want:
                log_activity(f"load {_ollama_name(friendly)}: image model already resident "
                             f"({_want}) — duplicate load ignored (no-op)")
                _resident.last_used = time.time()
                return _resident
        # Real dims from the shipped transformer config (generic across Edit variants etc.).
        _tc = {}
        with contextlib.suppress(Exception):
            with open(os.path.join(model_dir, "transformer", "config.json"), encoding="utf-8") as _fh:
                _tc = json.load(_fh)
        _layers = int(_tc.get("num_layers") or 60)
        _heads = int(_tc.get("num_attention_heads") or 24)
        _hd = int(_tc.get("attention_head_dim") or 128)
        dit_b = await asyncio.to_thread(_tree_weight_bytes, os.path.join(model_dir, "transformer"))
        all_b = await asyncio.to_thread(_tree_weight_bytes, model_dir)
        spec = ModelSpec(name=friendly, hidden_size=_heads * _hd, num_layers=_layers,
                         num_heads=_heads, num_kv_heads=_heads, head_dim=_hd,
                         intermediate_size=int(_tc.get("joint_attention_dim") or 3584),
                         vocab_size=0, tie_embeddings=True, max_ctx=0, arch="t2i",
                         meas_layer_w=max(1, dit_b // max(1, _layers)),
                         meas_embed=0, meas_head=0, meas_norm=0,
                         meas_params=max(1, all_b // 2))

        def _est_gb(edge: int) -> float:
            # measured on the gate test: pure int4 DiT ~0.284x bf16; each protected edge
            # block adds back its bf16-minus-int4 delta; +0.5 GB VAE/embedders; the
            # activation/decode margin is added by the caller.
            if quant == "none":
                return dit_b / GB + 0.6
            blk = dit_b / max(1, _layers)
            return (0.284 * dit_b + 2 * edge * blk * 0.716) / GB + 0.5

        _MARGIN_GB = 2.2   # step activations (~1.8 GB @1024^2 with CFG) + allocator headroom
        _OFFLOAD_VRAM_GB = 4.0   # #t2i-offload: transient blocks + activations + VAE only
        want_edge = 2 if quant != "none" else 0
        await self.ensure_data_listener()
        if reg_key in self.models:
            await self._unload_model_locked(reg_key, "reload (t2i)")
            await self._await_free_refresh()
        node = edge = None
        _ctrl_host = socket.gethostname()
        _refreshed = False
        while True:
            # co-located = same box as the controller: hostname match (the robust signal —
            # a standalone worker may register its LAN IP, e.g. om3nbox's 192.168.x) or a
            # loopback/this-box data endpoint.
            # #media-node-optout (audit #28): skip a node whose NODE_CONFIG has BOTH tiers
            # disabled (fully opted out) — same rule as the t2a filter (see _load_t2a_locked
            # for the furnace incident that motivated it). t2i is co-located-only today, so
            # this is precautionary here; it becomes load-bearing the day t2i goes remote.
            cand = [n for n in registry.alive_sorted()
                    if n.can_infer and n.vram_total_gb > 0
                    and (n.ram_enabled or n.vram_enabled)
                    and (n.hostname == _ctrl_host
                         or str(n.data_host).startswith(("127.", "::1"))
                         or str(n.data_host) in _LOCAL_IPS)]
            # In-flight loads' reservations count as USED (they're streaming toward that size —
            # observed: a cache-served 14b auto-load planned seconds before this one filled the
            # card mid-t2i-build and OOM'd even the offload transients). Same discipline as the
            # LLM planner's _res_vram subtraction; recomputed each retry pass.
            _res_ram_b, _res_vram_b = self._reserved_bytes(exclude_key=reg_key)

            def _t2i_free(x):   # #vram-reusable: vacant pool counts as free (see the LLM planner)
                return (x.vram_total_gb - x.vram_used_gb + getattr(x, "vram_reusable_gb", 0.0)
                        - _res_vram_b.get(x.node_id, 0) / GB)
            for n in sorted(cand, key=_t2i_free, reverse=True):
                free = _t2i_free(n)
                if offload:
                    # #t2i-offload: only transient VRAM + enough free RAM for the bf16 weights.
                    # NEVER evicts — not disturbing residents is this mode's whole purpose.
                    if free >= _OFFLOAD_VRAM_GB and \
                            ((n.free_mem_gb or 0) - _res_ram_b.get(n.node_id, 0) / GB) \
                            >= all_b / GB + 6.0:
                        node, edge = n, 0
                        break
                    continue
                for e in ([want_edge, 1] if quant != "none" else [0]):
                    if free >= _est_gb(e) + _MARGIN_GB:
                        node, edge = n, e
                        break
                if node is not None:
                    break
            if node is not None:
                break
            victim = (self._lru_evictable()
                      if (not offload and bool(ENGINE_CONFIG.get("auto_unload", True))) else None)
            if victim is None:
                if offload:
                    if not _refreshed:
                        _refreshed = True
                        await self._await_free_refresh()
                        continue
                    raise RuntimeError(
                        f"t2i offload needs ~{_OFFLOAD_VRAM_GB:.0f} GB free VRAM (transients) and "
                        f"~{all_b / GB + 6.0:.0f} GB free RAM on the controller-co-located GPU node "
                        "— it never evicts residents (that is its point); free some VRAM/RAM or "
                        "use the normal GPU load")
                # right after a restart/unload the heartbeat can still show the OLD vram_used —
                # wait one stats refresh and re-check ONCE before declaring no room (bit us live:
                # a load fired seconds after /update saw pre-unload numbers + nothing evictable)
                if not _refreshed:
                    _refreshed = True
                    await self._await_free_refresh()
                    continue
                _need = _est_gb(1 if quant != "none" else 0) + _MARGIN_GB
                raise RuntimeError(
                    f"no controller-co-located GPU has ~{_need:.1f} GB free VRAM for the image "
                    f"model ({'no co-located GPU workers connected' if not cand else 'and nothing evictable'})"
                    " — v1 serves t2i only on a GPU sharing the controller's box")
            await self._unload_model_locked(victim, "evict idle LRU: image model needs VRAM")
            await self._await_free_refresh()
        if edge != want_edge and quant != "none":
            log_activity(f"{_ollama_name(friendly)}: tight VRAM on {node.hostname} — edge {edge} "
                         f"(first+last {edge} blocks bf16) instead of {want_edge}")
        self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id, "total": 1, "ready": 0,
                        "stages_total": 1, "stages_ready": 0,
                        "basis": f"t2i: single-node ({node.hostname}, "
                                 f"{'bf16 OFFLOAD (DiT in RAM)' if offload else (quant if quant == 'none' else f'int4 edge{edge}')})",
                        "warnings": [], "node_ids": [node.node_id],
                        "started": (self.loadings.get(reg_key) or {}).get("started") or time.time(),
                        "requested_by": (self.loadings.get(reg_key) or {}).get("requested_by", "")}
        # Register in the RESERVATION ledger so CONCURRENT loads budget around this one — without
        # it, resident auto-loads filled the card to ~0 free during the multi-minute t2i build and
        # even the offload mode's small transient allocations OOM'd (observed live on beast: the
        # post-update resident reload raced the offload load to 96 MB free). Offload reserves its
        # VRAM transient + the bf16 weights in RAM; the GPU path reserves its full estimate. The
        # load() wrapper's finally pops self._reservations[reg_key] on every exit path.
        self._reservations[reg_key] = {node.node_id: {
            "ram": int((all_b + 6 * GB) if offload else 2 * GB),
            "vram": int((_OFFLOAD_VRAM_GB if offload else (_est_gb(edge) + _MARGIN_GB)) * GB)}}
        link = self.links.get(node.node_id)
        if link is None:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"no control link to {node.node_id}")
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        link.pending_loads[target_id] = fut
        await link.send({
            "type": "load", "kind": "t2i", "model_id": target_id,
            "model_dir": model_dir, "quant": quant, "t2i_edge": edge,
            "t2i_offload": bool(offload),
            "controller_http_port": ARGS.http_port,
            "next_host": None, "next_port": ARGS.data_port,
            # a plain torch device, NOT load_device() — that returns the worker's tier mode
            # ('cpu+gpu'), which the Shard placement interprets but torch .to() rejects; the
            # t2i path only ever places on GPU nodes, so 'cuda' is always right here.
            "device": "cuda",
        })
        try:
            # 20 min: an OFFLOAD load reads the full ~58 GB bf16 pipeline from the weights disk
            # into RAM (~8 min on beast's spinning drive when quiet, worse under contention);
            # the GPU int4 path finishes far sooner and is unaffected by the roomier ceiling.
            r = await asyncio.wait_for(fut, timeout=max(GEN_TIMEOUT_S, 1200.0))
        except Exception as exc:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"t2i load on {node.hostname} failed: {exc!r}")
        if isinstance(r, dict) and r.get("type") == "error":
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"t2i load on {node.hostname} failed: {r.get('error')}")
        gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
        tot_b = int(r.get("loaded_bytes", 0)) if isinstance(r, dict) else 0
        stage = StageAssign(node_id=node.node_id, hostname=node.hostname,
                            layer_start=0, layer_end=spec.num_layers,
                            has_embed=True, has_head=True,
                            est_bytes=tot_b or spec.total_weight_bytes,
                            usable_bytes=int(node.usable_total_gb * GB), gpu_bytes=gpu_b,
                            loaded_bytes=tot_b)
        plan = PlanResult(ok=True, model=spec.name, ctx_len=0,
                          num_layers=spec.num_layers,
                          pool_usable_gb=node.usable_total_gb,
                          required_gb=(tot_b or spec.total_weight_bytes) / GB, stages=[stage])
        node.shard_gpu_bytes = gpu_b
        _dial = (_dial_host(node.data_host), node.data_port)
        stage0_writer = await self._connect_retry(*_dial)
        now = time.time()
        lm = LoadedModel(reg_key, target_id, spec, 0, plan,
                         [node.node_id], None, set(), now,
                         quant=("bf16-off" if offload else
                                ("none" if quant == "none" else f"int4-e{edge}")),
                         stage0_writer=stage0_writer, last_used=now,
                         stage0_dial=_dial, last_send_ts=now)
        lm.base, lm.replica_idx = friendly, replica_idx
        lm.plan_basis = "t2i: single-node"
        lm.is_t2i = True
        self.models[reg_key] = lm
        self.loadings.pop(reg_key, None)
        registry.dirty = False
        print(f"[load] {reg_key} t2i on {node.hostname} "
              f"({tot_b / GB:.1f} GB total, {gpu_b / GB:.1f} GB GPU)")
        log_activity(f"{reg_key} READY (t2i, {node.hostname}, "
                     f"{'bf16 OFFLOAD — DiT in RAM, blocks stream to GPU' if offload else ('bf16' if quant == 'none' else f'int4 edge{edge}')}) "
                     f"[{len(self.models)} model(s) resident]")
        return lm

    async def t2i_generate(self, friendly: str, prompt: str, negative_prompt: str = " ",
                           width: int = 1024, height: int = 1024, steps: int = 20,
                           cfg: float = 4.0, seed=None) -> tuple[bytes, dict]:
        """#t2i-serve: render one image on the model's worker. Serializes per model on
        LoadedModel.lock (same discipline as text gens); the request travels the control
        link, per-step progress arrives as t2i_step (see control_plane + /status), and the
        finished PNG comes back as a LOCAL path (co-located worker) read + deleted here."""
        lm = self.models.get(friendly)
        if lm is None:
            raise ValueError(f"model '{friendly}' is not loaded — load it first")
        if not getattr(lm, "is_t2i", False):
            raise ValueError(f"'{friendly}' is not an image-generation model")
        async with lm.lock:
            lm.last_used = time.time()
            link = self.links.get(lm.stage_node_ids[0])
            if link is None:
                raise RuntimeError("the image model's worker is disconnected — reload the model")
            rid = self._t2i_rid = getattr(self, "_t2i_rid", 0) + 1
            pend = getattr(self, "_t2i_pending", None)
            if pend is None:
                pend = self._t2i_pending = {}
            fut = asyncio.get_event_loop().create_future()
            pend[rid] = fut
            lm.active = 1
            lm.t2i_req = rid            # status: lets the card find this render's progress
            try:
                await link.send({"type": "t2i_gen", "model_id": lm.target_id, "req_id": rid,
                                 "prompt": str(prompt), "negative_prompt": str(negative_prompt or " "),
                                 "width": int(width), "height": int(height),
                                 "steps": int(steps), "cfg": float(cfg), "seed": seed})
                # slowest observed ~42 s/step (gfx1151 sharing the GPU) — generous flat margin
                r = await asyncio.wait_for(fut, timeout=300 + int(steps) * 120)
            finally:
                lm.active = 0
                lm.t2i_req = None
                pend.pop(rid, None)
                getattr(self, "_t2i_progress", {}).pop(rid, None)
            if not isinstance(r, dict) or r.get("type") == "t2i_err":
                raise RuntimeError(f"image generation failed: "
                                   f"{(r or {}).get('error', 'no result')}")
            path = r.get("path") or ""

            def _read() -> bytes:
                with open(path, "rb") as fh:
                    data = fh.read()
                with contextlib.suppress(OSError):
                    os.remove(path)
                return data

            data = await asyncio.to_thread(_read)
            lm.last_used = time.time()
            log_activity(f"{_ollama_name(friendly)}: image {width}x{height} steps={steps} "
                         f"in {r.get('seconds', '?')}s ({len(data) / 1e6:.1f} MB)")
            return data, {"seconds": r.get("seconds"), "steps": int(steps),
                          "width": int(width), "height": int(height), "seed": seed}

    async def _load_t2a_locked(self, friendly: str, target_id: str, reg_key: str,
                               quant: str = "none", replica_idx: int = 0,
                               offload: bool = False,
                               cpu_only: bool = False) -> LoadedModel:
        """#t2a-serve (M1): load an ACE-Step music checkpoint WHOLE onto ONE controller-
        CO-LOCATED GPU worker — the audio sibling of _load_t2i_locked. bf16-only for M1
        (edge-int4 is M2). offload=True keeps ACE-Step's components in RAM and hops the whole
        ~6.6 GB DiT to the GPU per render (low resident VRAM, ~8 GB transient, never evicts);
        offload=False keeps the ~8.3 GB pipeline GPU-resident (faster). #t2a-cpu: cpu_only=True
        runs the WHOLE pipeline on CPU (device='cpu', RAM-resident, no GPU) — EXPERIMENTAL and
        very slow (diffusion on CPU), opt-in via /load?cpu_only=1; it forces offload off (a
        GPU-hop concept) and never falls back to GPU. v1 co-location constraint for the
        co-located fast path (shared FS: model dir read + WAV path back). MUST hold self.lock."""
        model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
        if quant and quant not in ("none", ""):
            log_activity(f"{_ollama_name(friendly)}: {quant} is not a t2a tier (M1 bf16-only) — using bf16")
        quant = "none"
        if cpu_only:
            offload = False   # #t2a-cpu: CPU compute is RAM-resident; offload is a GPU-hop concept
            log_activity(f"{_ollama_name(friendly)}: CPU-only t2a requested — EXPERIMENTAL and very "
                         f"slow (diffusion on CPU), no GPU/offload fallback")
        dit_b = await asyncio.to_thread(_tree_weight_bytes,
                                        os.path.join(model_dir, "ace_step_transformer"))
        all_b = await asyncio.to_thread(_tree_weight_bytes, model_dir)
        spec = ModelSpec(name=friendly, hidden_size=2560, num_layers=24, num_heads=20,
                         num_kv_heads=20, head_dim=128, intermediate_size=6400,
                         vocab_size=0, tie_embeddings=True, max_ctx=0, arch="t2a",
                         meas_layer_w=max(1, dit_b // 24), meas_embed=0, meas_head=0,
                         meas_norm=0, meas_params=max(1, all_b // 2))
        _T2A_OFFLOAD_VRAM_GB = 8.0   # M0 whole-DiT-hop peak 6.67 GB + decode/activation margin
        # #t2a-render-peak: size a GPU-RESIDENT placement to the diffusion RENDER peak, not the
        # load footprint. ACE-Step's whole pipeline RESTS at ~8.3 GB, but a render climbs ~3+ GB
        # higher (denoising activations + audio latents). An 11.55 GB card (RTX 3060) passed the
        # old load-sized gate (all_b + 2.5 ≈ 10.8 GB), loaded fine, then CUDA-OOM'd MID-GENERATION
        # at ~11.4 GB used — 32 MiB short (beast pool, 2026-07-20). Require the render peak (a FLOOR
        # the 3060 can't clear, or all_b+margin for a larger t2a model) so a marginal card is
        # rejected at PLACEMENT — it then picks a bigger card, evicts an idle LRU model, or fails
        # clean — instead of dying partway through a render the user is waiting on. A too-small card
        # is still reachable via OFFLOAD mode (DiT-hop only, ~8 GB peak), which the loop falls back
        # to when no card clears the resident floor and nothing is evictable.
        _MARGIN_GB = 4.5             # ~8.3 GB resident + ~3 GB render activation peak + safety
        _T2A_RENDER_FLOOR_GB = 12.5  # min FREE VRAM for any ace-step GPU-resident render (observed
                                     # peak ~11.4 GB + safety) — excludes the 11.55 GB 3060 whatever
                                     # all_b measures
        def _need_gb() -> float:
            return _T2A_OFFLOAD_VRAM_GB if offload else max(all_b / GB + _MARGIN_GB,
                                                            _T2A_RENDER_FLOOR_GB)
        await self.ensure_data_listener()
        if reg_key in self.models:
            await self._unload_model_locked(reg_key, "reload (t2a)")
            await self._await_free_refresh()
        node = None
        _ctrl_host = socket.gethostname()
        _refreshed = False
        while True:
            def _is_colo(_n):
                return (_n.hostname == _ctrl_host
                        or str(_n.data_host).startswith(("127.", "::1"))
                        or str(_n.data_host) in _LOCAL_IPS)
            # #media-anywhere: serve on the co-located GPU OR any REMOTE GPU whose worker
            # advertised the acestep runtime (can_t2a) — the checkpoint streams to it via
            # snapshot_download and the WAV returns as base64 over the link, so no shared FS.
            # #media-node-optout (audit #28): honor the dashboard's per-node tier toggles like
            # the LLM planner does (its usable<=0 skip flows from eff_ram/eff_vram) — a node
            # with BOTH tiers disabled in NODE_CONFIG is fully opted out and must not receive
            # renders. This filter used to key off RAW vram_total_gb + can_t2a only, and the
            # can_t2a-first sort below made the om3nbox pool's ONLY acestep node ALWAYS win —
            # so music kept routing to furnace (RTX 5090, user-declared OFF-LIMITS, both tiers
            # off) and OOM'd it. A SINGLE disabled tier still admits the node (tier toggles are
            # placement knobs — RAM off = "GPU-only" still means the GPU is usable); only the
            # both-off state excludes. Default (no NODE_CONFIG entry) is both-on -> unchanged.
            cand = [n for n in registry.alive_sorted()
                    if n.can_infer and (cpu_only or n.vram_total_gb > 0)
                    and (n.ram_enabled or n.vram_enabled)
                    and (_is_colo(n) or getattr(n, "can_t2a", False))]
            # in-flight loads' reservations count as USED (same discipline as the t2i/LLM planners)
            _res_ram_b, _res_vram_b = self._reserved_bytes(exclude_key=reg_key)
            def _t2a_free(x):
                return (x.vram_total_gb - x.vram_used_gb + getattr(x, "vram_reusable_gb", 0.0)
                        - _res_vram_b.get(x.node_id, 0) / GB)
            # Prefer a worker that ADVERTISES the acestep runtime (can_t2a) — it's known-capable —
            # over a co-located worker that merely might have it; among equals, co-located first
            # (no network transfer), then most-free VRAM. So a co-located box WITHOUT acestep
            # (can_t2a False) is a last resort, never picked ahead of a capable remote GPU that
            # would otherwise fail the load with no failover.
            for n in sorted(cand, key=lambda _n: (bool(getattr(_n, "can_t2a", False)),
                                                   _is_colo(_n), _t2a_free(_n)), reverse=True):
                if cpu_only:
                    # #t2a-cpu: budget against RAM only (VRAM ignored) — whole pipeline + margin in RAM.
                    if ((n.free_mem_gb or 0) - _res_ram_b.get(n.node_id, 0) / GB) \
                            >= all_b / GB + _MARGIN_GB:
                        node = n
                        break
                    continue
                free = _t2a_free(n)
                if offload:
                    # never evicts (its point): needs only the transient VRAM + RAM for the weights
                    if free >= _T2A_OFFLOAD_VRAM_GB and \
                            ((n.free_mem_gb or 0) - _res_ram_b.get(n.node_id, 0) / GB) \
                            >= all_b / GB + 4.0:
                        node = n
                        break
                    continue
                if free >= _need_gb():
                    node = n
                    break
            if node is not None:
                break
            if cpu_only:
                # #t2a-cpu: no GPU/offload fallback — a worker either has the RAM + acestep or we fail.
                if not _refreshed:
                    _refreshed = True
                    await self._await_free_refresh()
                    continue
                raise RuntimeError(
                    f"no worker has ~{all_b / GB + _MARGIN_GB:.1f} GB free RAM plus the acestep "
                    f"runtime for CPU-only t2a "
                    f"({'no capable worker connected' if not cand else 'all workers too full'})"
                    " — CPU-only music is EXPERIMENTAL and extremely slow; the normal path is a GPU")
            victim = (self._lru_evictable()
                      if (not offload and bool(ENGINE_CONFIG.get("auto_unload", True))) else None)
            if victim is None:
                if not _refreshed:      # a just-freed heartbeat can lag one refresh — re-check once
                    _refreshed = True
                    await self._await_free_refresh()
                    continue
                if not offload:
                    # #t2a-offload-fallback: bf16 GPU-resident won't fit and nothing is evictable
                    # (pinned / busy residents hold the VRAM). Rather than fail the render, fall back
                    # to the RAM-offload recipe — components live in RAM and the DiT hops to the GPU
                    # per render (~8 GB transient VRAM + RAM for the weights, and it NEVER evicts). It
                    # is M1's proven serving mode, so it always beats a hard failure. Resident stays
                    # the fast default (we only reach here after trying it + evicting idle LRU); this
                    # triggers only when the card is genuinely full of un-evictable models.
                    log_activity(f"{_ollama_name(friendly)}: bf16 GPU-resident won't fit "
                                 f"(~{_need_gb():.1f} GB, nothing evictable) — falling back to RAM "
                                 f"offload (~{_T2A_OFFLOAD_VRAM_GB:.0f} GB transient + "
                                 f"~{all_b / GB + 4.0:.0f} GB RAM, never evicts)")
                    offload = True
                    _refreshed = False
                    continue
                raise RuntimeError(
                    f"no GPU has ~{_need_gb():.1f} GB free for the music model "
                    f"({'no capable GPU worker connected' if not cand else 'and nothing evictable'})"
                    " — t2a serves on the co-located GPU or any worker advertising the acestep "
                    "runtime (can_t2a)"
                    + (f"; or use offload (t2i_offload=1): ~{_T2A_OFFLOAD_VRAM_GB:.0f} GB transient "
                       f"+ ~{all_b / GB + 4.0:.0f} GB RAM, never evicts" if not offload else ""))
            await self._unload_model_locked(victim, "evict idle LRU: music model needs VRAM")
            await self._await_free_refresh()
        self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id, "total": 1, "ready": 0,
                        "stages_total": 1, "stages_ready": 0,
                        "basis": f"t2a: single-node ({node.hostname}, "
                                 + ("bf16 CPU (experimental, slow)" if cpu_only
                                    else "bf16 OFFLOAD (components in RAM)" if offload
                                    else "bf16 GPU-resident") + ")",
                        "warnings": [], "node_ids": [node.node_id],
                        "started": (self.loadings.get(reg_key) or {}).get("started") or time.time(),
                        "requested_by": (self.loadings.get(reg_key) or {}).get("requested_by", "")}
        self._reservations[reg_key] = {node.node_id: {
            "ram": int(all_b + _MARGIN_GB * GB) if cpu_only
                   else int((all_b + 4 * GB) if offload else 2 * GB),
            "vram": 0 if cpu_only
                    else int((_T2A_OFFLOAD_VRAM_GB if offload else _need_gb()) * GB)}}
        link = self.links.get(node.node_id)
        if link is None:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"no control link to {node.node_id}")
        fut = asyncio.get_event_loop().create_future()
        link.pending_loads[target_id] = fut
        await link.send({"type": "load", "kind": "t2a", "model_id": target_id,
                         "model_dir": model_dir, "quant": "none",
                         "t2a_offload": bool(offload),
                         "controller_http_port": ARGS.http_port,
                         "next_host": None, "next_port": ARGS.data_port,
                         "device": "cpu" if cpu_only else "cuda"})
        try:
            # generous: an offload load reads the ~8.3 GB pipeline into RAM then builds
            r = await asyncio.wait_for(fut, timeout=max(GEN_TIMEOUT_S, 1200.0))
        except Exception as exc:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"t2a load on {node.hostname} failed: {exc!r}")
        if isinstance(r, dict) and r.get("type") == "error":
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"t2a load on {node.hostname} failed: {r.get('error')}")
        gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
        tot_b = int(r.get("loaded_bytes", 0)) if isinstance(r, dict) else 0
        stage = StageAssign(node_id=node.node_id, hostname=node.hostname,
                            layer_start=0, layer_end=1, has_embed=True, has_head=True,
                            est_bytes=tot_b or 1, usable_bytes=int(node.usable_total_gb * GB),
                            gpu_bytes=gpu_b, loaded_bytes=tot_b)
        plan = PlanResult(ok=True, model=spec.name, ctx_len=0, num_layers=1,
                          pool_usable_gb=node.usable_total_gb,
                          required_gb=(tot_b or 1) / GB, stages=[stage])
        node.shard_gpu_bytes = gpu_b
        _dial = (_dial_host(node.data_host), node.data_port)
        stage0_writer = await self._connect_retry(*_dial)
        now = time.time()
        lm = LoadedModel(reg_key, target_id, spec, 0, plan, [node.node_id], None, set(), now,
                         quant="none", stage0_writer=stage0_writer, last_used=now,
                         stage0_dial=_dial, last_send_ts=now)
        lm.base, lm.replica_idx = friendly, replica_idx
        lm.plan_basis = "t2a: single-node" + (" (CPU)" if cpu_only else "")
        lm.is_t2a = True
        lm.t2a_offload = bool(offload)
        lm.t2a_cpu = bool(cpu_only)   # #t2a-cpu: render path gives a CPU pipeline a larger timeout
        lm.media = r.get("media") if isinstance(r, dict) else None   # #media-detail (may be None)
        self.models[reg_key] = lm
        self.loadings.pop(reg_key, None)
        registry.dirty = False
        print(f"[load] {reg_key} t2a (ace-step) on {node.hostname} "
              f"({tot_b / GB:.2f} GB, {gpu_b / GB:.2f} GB GPU, {'offload' if offload else 'resident'})")
        log_activity(f"{reg_key} READY (t2a/ace-step, {node.hostname}, "
                     f"{'offload' if offload else 'resident'}) [{len(self.models)} model(s) resident]")
        return lm

    async def _load_tts_locked(self, friendly: str, target_id: str, reg_key: str,
                               replica_idx: int = 0) -> LoadedModel:
        """#tts-serve: load a Kokoro-82M speech checkpoint WHOLE onto ONE controller-
        CO-LOCATED worker — the speech sibling of _load_t2i_locked, but tiny (~0.33 GB),
        so no VRAM edge/quant tiers and no eviction dance. Prefers a co-located GPU
        (beast: CUDA ~4x realtime); a CPU-only co-located node — or a GPU whose MIOpen
        JIT-fails (handled by the leaf's CPU fallback) — serves on CPU. MUST hold
        self.lock (reached only from load())."""
        model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
        await self.ensure_data_listener()
        if reg_key in self.models:
            await self._unload_model_locked(reg_key, "reload (tts)")
            await self._await_free_refresh()
        _ctrl_host = socket.gethostname()
        cand = [n for n in registry.alive_sorted()
                if n.can_infer and (n.hostname == _ctrl_host
                     or str(n.data_host).startswith(("127.", "::1"))
                     or str(n.data_host) in _LOCAL_IPS)]
        if not cand:
            raise RuntimeError("no controller-co-located worker for the tts (Kokoro) model "
                               "— v1 serves speech models only on a worker sharing the "
                               "controller's box")
        # prefer a GPU-capable co-located node (faster); fall back to a CPU node.
        node = next((n for n in cand if n.vram_total_gb > 0), cand[0])
        device = "cuda" if node.vram_total_gb > 0 else "cpu"
        # Kokoro is ~0.33 GB — a small reservation so a concurrent big load budgets around it.
        self._reservations[reg_key] = {node.node_id: {
            "ram": int(1.5 * GB), "vram": int(1.0 * GB) if device == "cuda" else 0}}
        self.loadings[reg_key] = {"model": friendly, "display_model": _ollama_name(friendly),
                        "target": target_id, "total": 1, "ready": 0,
                        "stages_total": 1, "stages_ready": 0,
                        "basis": f"tts: single-node ({node.hostname}, kokoro/{device})",
                        "warnings": [], "node_ids": [node.node_id],
                        "started": (self.loadings.get(reg_key) or {}).get("started") or time.time(),
                        "requested_by": (self.loadings.get(reg_key) or {}).get("requested_by", "")}
        link = self.links.get(node.node_id)
        if link is None:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"no control link to {node.node_id}")
        fut = asyncio.get_event_loop().create_future()
        link.pending_loads[target_id] = fut
        await link.send({"type": "load", "kind": "tts", "model_id": target_id,
                         "model_dir": model_dir, "quant": "none",
                         "controller_http_port": ARGS.http_port,
                         "next_host": None, "next_port": ARGS.data_port,
                         "device": device})
        try:
            r = await asyncio.wait_for(fut, timeout=max(GEN_TIMEOUT_S, 600.0))
        except Exception as exc:
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"tts load on {node.hostname} failed: {exc!r}")
        if isinstance(r, dict) and r.get("type") == "error":
            self.loadings.pop(reg_key, None)
            raise RuntimeError(f"tts load on {node.hostname} failed: {r.get('error')}")
        gpu_b = int(r.get("gpu_bytes", 0)) if isinstance(r, dict) else 0
        tot_b = int(r.get("loaded_bytes", 0)) if isinstance(r, dict) else 0
        spec = ModelSpec(name=friendly, hidden_size=1, num_layers=1, num_heads=1,
                         num_kv_heads=1, head_dim=1, intermediate_size=1, vocab_size=0,
                         tie_embeddings=True, max_ctx=0, arch="tts",
                         meas_layer_w=1, meas_embed=0, meas_head=0, meas_norm=0,
                         meas_params=max(1, tot_b))
        stage = StageAssign(node_id=node.node_id, hostname=node.hostname,
                            layer_start=0, layer_end=1, has_embed=True, has_head=True,
                            est_bytes=tot_b or 1, usable_bytes=int(node.usable_total_gb * GB),
                            gpu_bytes=gpu_b, loaded_bytes=tot_b)
        plan = PlanResult(ok=True, model=spec.name, ctx_len=0, num_layers=1,
                          pool_usable_gb=node.usable_total_gb,
                          required_gb=(tot_b or 1) / GB, stages=[stage])
        node.shard_gpu_bytes = gpu_b
        _dial = (_dial_host(node.data_host), node.data_port)
        stage0_writer = await self._connect_retry(*_dial)
        now = time.time()
        lm = LoadedModel(reg_key, target_id, spec, 0, plan, [node.node_id], None, set(), now,
                         quant="none", stage0_writer=stage0_writer, last_used=now,
                         stage0_dial=_dial, last_send_ts=now)
        lm.base, lm.replica_idx = friendly, replica_idx
        lm.plan_basis = "tts: single-node"
        lm.is_tts = True
        lm.is_kokoro = True
        lm.media = r.get("media") if isinstance(r, dict) else None   # voices/device/sr (#media-detail)
        self.models[reg_key] = lm
        self.loadings.pop(reg_key, None)
        registry.dirty = False
        print(f"[load] {reg_key} tts (kokoro) on {node.hostname} "
              f"({tot_b / GB:.2f} GB, {gpu_b / GB:.2f} GB GPU, {device})")
        log_activity(f"{reg_key} READY (tts/kokoro, {node.hostname}, {device}) "
                     f"[{len(self.models)} model(s) resident]")
        return lm

    async def tts_generate(self, friendly: str, text: str, voice: str = "",
                           speed: float = 1.0, fmt: str = "wav") -> tuple[bytes, dict]:
        """#tts-serve: synthesize speech on the model's Kokoro worker. Serializes per model
        on LoadedModel.lock (same discipline as text/image gens); the request travels the
        control link, per-chunk progress arrives as tts_step (see control_plane + /status),
        and the finished WAV comes back as a LOCAL path (co-located worker) read + deleted
        here. Returns (audio_bytes, meta)."""
        lm = self.models.get(friendly)
        if lm is None:
            raise ValueError(f"model '{friendly}' is not loaded — load it first")
        if not getattr(lm, "is_tts", False):
            raise ValueError(f"'{friendly}' is not a speech (tts) model")
        async with lm.lock:
            lm.last_used = time.time()
            link = self.links.get(lm.stage_node_ids[0])
            if link is None:
                raise RuntimeError("the tts model's worker is disconnected — reload the model")
            rid = self._tts_rid = getattr(self, "_tts_rid", 0) + 1
            pend = getattr(self, "_tts_pending", None)
            if pend is None:
                pend = self._tts_pending = {}
            fut = asyncio.get_event_loop().create_future()
            pend[rid] = fut
            lm.active = 1
            lm.tts_req = rid            # status: lets the card find this render's progress
            try:
                await link.send({"type": "tts_gen", "model_id": lm.target_id, "req_id": rid,
                                 "text": str(text), "voice": str(voice or ""),
                                 "speed": float(speed or 1.0), "fmt": str(fmt or "wav")})
                # CPU synthesis runs ~4x realtime; scale the ceiling to the text length.
                r = await asyncio.wait_for(
                    fut, timeout=max(GEN_TIMEOUT_S, 60.0 + len(str(text)) * 0.5))
            finally:
                lm.active = 0
                lm.tts_req = None
                pend.pop(rid, None)
                getattr(self, "_tts_progress", {}).pop(rid, None)
            if not isinstance(r, dict) or r.get("type") == "tts_err":
                raise RuntimeError(f"speech generation failed: "
                                   f"{(r or {}).get('error', 'no result')}")
            lm.last_render_s = r.get("seconds")      # wall time of this synth (#media-detail)
            lm.last_audio_s = r.get("audio_s")       # audio duration -> RTF in the modal
            path = r.get("path") or ""

            def _read() -> bytes:
                with open(path, "rb") as fh:
                    data = fh.read()
                with contextlib.suppress(OSError):
                    os.remove(path)
                return data

            data = await asyncio.to_thread(_read)
            lm.last_used = time.time()
            log_activity(f"{_ollama_name(friendly)}: speech {r.get('seconds', '?')}s "
                         f"({len(data) / 1e6:.2f} MB)")
            return data, {"seconds": r.get("seconds"), "voice": voice, "format": fmt}

    async def t2a_generate(self, friendly: str, prompt: str, lyrics: str = "",
                           duration: float = 30.0, steps: int = 60, guidance: float = 15.0,
                           seed=None, fmt: str = "wav") -> tuple[bytes, dict]:
        """#t2a-serve: render one music clip on the model's ACE-Step worker. Serializes per model
        on LoadedModel.lock (same discipline as text/image/speech gens); the request travels the
        control link, per-step progress arrives as t2a_step (see control_plane + /status), and the
        finished WAV comes back as a LOCAL path (co-located worker) read + deleted here. Returns
        (audio_bytes, meta)."""
        lm = self.models.get(friendly)
        if lm is None:
            raise ValueError(f"model '{friendly}' is not loaded — load it first")
        if not getattr(lm, "is_t2a", False):
            raise ValueError(f"'{friendly}' is not a music (t2a) model")
        async with lm.lock:
            lm.last_used = time.time()
            link = self.links.get(lm.stage_node_ids[0])
            if link is None:
                raise RuntimeError("the t2a model's worker is disconnected — reload the model")
            rid = self._t2a_rid = getattr(self, "_t2a_rid", 0) + 1
            pend = getattr(self, "_t2a_pending", None)
            if pend is None:
                pend = self._t2a_pending = {}
            fut = asyncio.get_event_loop().create_future()
            pend[rid] = fut
            lm.active = 1
            lm.t2a_req = rid            # status: lets the card find this render's progress
            _dur = max(3.0, min(240.0, float(duration)))
            try:
                await link.send({"type": "t2a_gen", "model_id": lm.target_id, "req_id": rid,
                                 "prompt": str(prompt or ""), "lyrics": str(lyrics or ""),
                                 "duration": _dur, "steps": int(steps),
                                 "guidance": float(guidance), "seed": seed})
                # GPU render is ~2x realtime (M0); offload slower; #t2a-cpu diffusion-on-CPU is
                # DRAMATICALLY slower — give a CPU-resident model a much larger upper bound so a
                # legitimately slow render isn't cut off (it's only a hang ceiling, costs nothing).
                _t2a_cpu = bool(getattr(lm, "t2a_cpu", False))
                r = await asyncio.wait_for(
                    fut, timeout=max(GEN_TIMEOUT_S,
                                     (600.0 + _dur * 120.0) if _t2a_cpu else (120.0 + _dur * 20.0)))
            finally:
                lm.active = 0
                lm.t2a_req = None
                pend.pop(rid, None)
                getattr(self, "_t2a_progress", {}).pop(rid, None)
            if not isinstance(r, dict) or r.get("type") == "t2a_err":
                raise RuntimeError(f"music generation failed: "
                                   f"{(r or {}).get('error', 'no result')}")
            lm.last_render_s = r.get("seconds")      # wall time of this render (#media-detail)
            lm.last_audio_s = r.get("audio_s")       # audio duration -> RTF in the modal
            _b64 = r.get("audio_b64")
            if _b64:
                # #media-anywhere: a REMOTE worker (no shared FS) returns the WAV as base64 over
                # the control link. Co-located workers may still send a local `path` (legacy).
                # Decode off the event loop — a max-length clip is tens of MB.
                import base64
                data = await asyncio.to_thread(base64.b64decode, _b64)
            else:
                path = r.get("path") or ""

                def _read() -> bytes:
                    with open(path, "rb") as fh:
                        data = fh.read()
                    with contextlib.suppress(OSError):
                        os.remove(path)
                    return data

                data = await asyncio.to_thread(_read)
            lm.last_used = time.time()
            log_activity(f"{_ollama_name(friendly)}: music {r.get('seconds', '?')}s "
                         f"({len(data) / 1e6:.2f} MB)")
            return data, {"seconds": r.get("seconds"), "audio_s": r.get("audio_s"),
                          "format": fmt}

    async def replicate(self, friendly: str, ctx: int, count: int,
                        consolidate: bool = True, prefer_vram: bool = True,
                        quant: str = "none", kv_quant: str = "",
                        kv_offload: bool = False, kv_slots: int = 1,
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
                                         kv_offload=kv_offload, kv_slots=kv_slots,
                                         default_temp=default_temp,
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
                   # #loopback-nexthop (same translation as the pipeline wiring): a controller-
                   # co-located rank0 advertises a loopback data_host; a REMOTE rank dialing that
                   # as the mesh root would connect to ITSELF. Swap in our address as this rank's
                   # control link reaches us; loopback stays for a local rank (fastest on-box).
                   "tp_root_host": self._lan_visible_host(root.data_host, n, link),
                   "tp_root_port": tp_port,
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
        _tpb = (int(_tpb / 0.3) if quant == "int4" else int(_tpb / 0.2) if quant == "int2"
                else (_tpb * 2 if quant == "int8" else _tpb))
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
                          consolidate: bool, prefer_vram: bool, cpu_only: bool,
                          kv_slots: Optional[int] = None) -> LoadedModel:
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
        # #kv-slots: None = PRESERVE the resident copy's slot count across the re-place (the
        # wedge-quarantine self-heal + juggler go through here — a reconfigure must not silently
        # drop a model back to C=1). An explicit value changes it (route callers). tp>1 clamps
        # to 1 downstream in load() regardless.
        prev_kvs = max(1, int(getattr(prev, "kv_slots", 1) or 1))
        kv_slots = prev_kvs if kv_slots is None else max(1, int(kv_slots or 1))
        from_label = f"tp{prev_tp}" if prev_tp > 1 else "pipeline"
        to_label = ((f"tp{tp}" + ("-cpu" if cpu_only else "")) if tp > 1 else "pipeline")
        self.reconfiguring = {"model": _ollama_name(friendly), "from": from_label, "to": to_label,
                              "from_tp": prev_tp, "to_tp": tp}
        log_activity(f"{friendly}: RECONFIGURE {from_label} -> {to_label} "
                     f"(ctx {prev_ctx}->{ctx}, quant {prev_quant}->{quant})")
        try:
            lm = await self.load(friendly, ctx, consolidate=consolidate, prefer_vram=prefer_vram,
                                 quant=quant, tp=tp, cpu_only=cpu_only, force=True,
                                 kv_slots=kv_slots)
            log_activity(f"{friendly}: reconfigured -> {lm.plan_basis}")
            return lm
        except Exception as exc:
            # New layout failed; engine.load already evicted the old copy. Restore a WORKING copy: a
            # plain pipeline-auto load at the PREVIOUS ctx/quant (GPU-first, spills to CPU — the robust
            # path that always places). Better than leaving the model gone.
            log_activity(f"{friendly}: reconfigure to {to_label} FAILED ({exc!r}) -> rolling back to "
                         f"pipeline @ ctx={prev_ctx} {prev_quant}")
            try:
                await self.load(friendly, prev_ctx, quant=prev_quant, force=True,
                                kv_slots=prev_kvs)
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

    def _node_live_free_vram_gb(self, n, *, own_bytes: int = 0,
                                res_vram: Optional[dict] = None) -> float:
        """The single source of truth for 'how much VRAM can a (re)placement actually use on
        node `n` right now'. = the heartbeat's LIVE `vram_total - vram_used`, PLUS the worker's
        vacant allocator pool (#vram-reusable — a new alloc reuses it even though device counters
        report it 'used'), PLUS `own_bytes` (VRAM the model being re-placed already holds here and
        reclaims on reload), MINUS other in-flight loads' reservations. Deliberately NOT
        `usable_vram_gb` (= vram_total - a static reserve): that ceiling ignores resident models
        and never moves when VRAM frees — using it silently pinned the juggler's fit-check and its
        anti-churn guard to a value that could never change, so a hybrid model never relocated even
        after a co-resident unloaded and freed the GPU. Both the load planner (below) and the
        juggler budget against THIS number so they can never disagree again."""
        rv = (res_vram.get(n.node_id, 0) if res_vram else 0)
        return max(0.0, n.vram_total_gb - n.vram_used_gb
                   + getattr(n, "vram_reusable_gb", 0.0)
                   + own_bytes / GB
                   - rv / GB)

    def _is_unified_node(self, n) -> bool:
        """#unified-mem: True when this node's "VRAM" is GTT carved from the SAME physical RAM as
        its heartbeat free_mem (om3nbox Strix Halo gfx1151; steamdeck Van Gogh) — so a RAM reserve
        and a VRAM budget spend ONE pool, and every GB not reserved is a GB the planner may hand to
        weights. Same guarded import + heuristic as _unified_mem_clamp (single source of truth): a
        stale placement.py (per-file CDN lag, #cdn-lag-deploy) reports False, which keeps the
        pre-existing behavior instead of crashing a load. Discrete-GPU/CPU nodes never match."""
        try:
            import placement as _pl
        except Exception:
            return False
        _ufn = getattr(_pl, "is_unified_mem_node", None)
        if _ufn is None:
            return False
        try:
            return bool(_ufn(n.device_name, n.vram_total_gb, n.total_mem_gb,
                             explicit=getattr(n, "unified_mem", None)))
        except Exception:
            return False

    def _unified_mem_clamp(self, n, usable_gb: float, free_vram_gb: float,
                           reserve_gb: float, own_gb: float = 0.0):
        """#unified-mem (audit #17): on an APU whose "VRAM" is GTT carved from the SAME physical
        RAM as the heartbeat's free_mem (om3nbox Strix Halo gfx1151; steamdeck Van Gogh), a
        budget built as ram_share + vram_share sums the one pool TWICE — an idle om3nbox reads
        ~165 GB "usable" on a 128 GB box, and a near-ceiling plan then over-commits into a
        cgroup OOM-kill mid-load (worker drop -> #99 replan churn, or a dropped live render —
        the very incident #render-oom-guard v2 patches for ONE path; this clamps the PLANNER,
        covering explicit /load, <=0.5-CPU hybrids and first-load-on-idle-box too). Clamp the
        summed budget to LIVE physically-free bytes: psutil free (GTT-pinned pages already
        depress it) + `own_gb` reclaimable bytes (a re-place frees its own shard) — the vacant
        allocator pool (#vram-reusable) is added inside the pure helper — minus `reserve_gb`
        (whatever the caller charged its RAM/VRAM budgets: controller/transient reserves,
        PLAN_VRAM_FLOOR, in-flight reservations) and the same adaptive RAM safety eff_ram_gb
        keeps. Returns (usable_gb, free_vram_gb); the GPU share is re-capped at the clamped
        total — one pool, so neither share may exceed it. Detection heuristic + its honest
        limits live in placement.is_unified_mem_node. Guarded imports: a stale placement.py
        (per-file CDN lag, see #cdn-lag-deploy) keeps the OLD unclamped behavior instead of
        crashing every load; discrete-GPU (CUDA) nodes never match, so this is inert for them."""
        if not self._is_unified_node(n):     # shared detection (see _is_unified_node)
            return usable_gb, free_vram_gb
        try:
            import placement as _pl
        except Exception:
            return usable_gb, free_vram_gb
        usable_gb = _pl.clamp_unified_usable_gb(
            usable_gb, n.free_mem_gb,
            getattr(n, "vram_reusable_gb", 0.0) + max(0.0, own_gb),
            max(0.0, reserve_gb) + min(RAM_SAFETY_GB, n.free_mem_gb * 0.4))
        return usable_gb, min(free_vram_gb, usable_gb)

    async def _juggle_would_fit_vram(self, m) -> bool:
        """Dry-run the VRAM-first planner for `m` against the LIVE free VRAM PLUS this model's own
        on-GPU bytes (which a reload reclaims), and return True only when every weight lands on GPU.
        Cheap go/no-go so the juggler never does a disruptive re-place that would stay hybrid."""
        try:
            spec = m.spec   # the model's OWN measured, quant-sized spec (what it was placed with)
            own = {}   # per-node VRAM this model holds now -> reclaimed on reload, so add it back
            for s in m.plan.stages:
                own[s.node_id] = own.get(s.node_id, 0) + (getattr(s, "gpu_bytes", 0) or 0)
            _, _res_vram = self._reserved_bytes()   # other in-flight loads' reservations
            mems = []
            for n in registry.alive_sorted():
                # LIVE-free VRAM (heartbeat vram_total-vram_used + reusable pool), NOT the static
                # usable_vram ceiling, + this model's own reclaimable bytes, then the runtime floor —
                # the SAME basis the load planner budgets weights against, so a 'would it fit
                # VRAM-only?' answer actually matches what a real re-place can achieve (a resident
                # co-tenant now correctly shrinks the target, and a freed GPU correctly opens up).
                fv = max(0.0, self._node_live_free_vram_gb(
                    n, own_bytes=own.get(n.node_id, 0), res_vram=_res_vram) - PLAN_VRAM_FLOOR_GB)
                ram = n.eff_ram_gb - (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
                # #unified-mem (audit #17): same shared-pool clamp as the load planner's mems
                # loop — a dry-run built on RAM+GTT double-counted bytes promises a promotion
                # the real re-place can then OOM on. own_gb: this model's reclaimable shard.
                tot, fv = self._unified_mem_clamp(
                    n, max(0.0, ram) + fv, fv,
                    (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
                    + PLAN_VRAM_FLOOR_GB + _res_vram.get(n.node_id, 0) / GB,
                    own_gb=own.get(n.node_id, 0) / GB)
                mems.append(NodeMem(n.node_id, n.hostname, int(tot * GB), int(fv * GB)))
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
        promoted to GPU. HITLESS: only a momentarily-IDLE model is promoted (a busy/backlogged one is
        skipped — engaging the barrier and draining it would stall its clients; a later sweep catches it
        at a gap) — engage a per-model barrier so new requests wait during the ~10-20s re-place,
        reconfigure VRAM-first (atomic + rollback), then release the barrier — the client's open
        connection just pauses across the swap, no reconnect. One promotion per call;
        serialized by _juggle_lock. #no-unload / persist PINNED models are eligible on purpose (a
        promotion is a reload-in-a-BETTER-way, not a removal — the finally restores a pinned target if a
        rare reconfigure+rollback double-failure ever evicts it, so "never auto-unloaded" still holds)."""
        if not ENGINE_CONFIG.get("juggler", False) or self._juggle_lock.locked():
            return
        async with self._juggle_lock:
            # 1) rank the PROMOTABLE hybrids by hotness (freshest use first, so a 'constantly in use'
            #    model wins). Skip embeddings/encoders (CPU-by-design, not served through the barrier)
            #    and anything already ~all-GPU (nothing to promote).
            # #anti-churn basis: LIVE-free fleet VRAM (so it actually rises when a co-resident
            # unloads), NOT the static usable_vram sum (which is invariant to load/unload and made
            # the "won't retry until the fleet frees more VRAM" guard latch forever).
            _fleet_free = sum(max(0.0, self._node_live_free_vram_gb(n) - PLAN_VRAM_FLOOR_GB)
                              for n in registry.alive_sorted()
                              if getattr(n, "can_infer", False) and n.vram_total_gb > 0)
            cands = []
            for fr, m in list(self.models.items()):
                if getattr(m.spec, "is_embedding", False) or getattr(m, "is_embedding", False):
                    continue
                # #t2i-serve: an image model's CPU share IS its text encoder — a DESIGNED
                # placement (encode-once per request; the TE never fits beside the DiT on a
                # 16 GB card), not a hybrid to promote. The juggler re-placing it unloads a
                # healthy pipeline (observed live: 'promoting qwen-image (56% was on CPU)'
                # 25s after its first load). Skip like embeddings.
                if getattr(m, "is_t2i", False) or getattr(m, "is_t2a", False) \
                        or getattr(m, "is_tts", False):
                    continue
                if self._model_cpu_frac(m) <= 0.02:
                    continue
                # #no-stall: only promote a model that is momentarily IDLE. Engaging the barrier on a
                # busy/backlogged model and waiting to drain it would STALL its new requests (a slow
                # CPU-bound model with a queue never drains) — far worse than leaving it hybrid. A
                # frequently-used model is caught at a gap by a later ~60s sweep instead.
                if (m.active or 0) > 0 or (m.queued or 0) > 0:
                    continue
                # #anti-churn: a prior promotion left this model STILL hybrid (fit-check was optimistic /
                # fleet couldn't hold it all on GPU) — skip re-promoting it until the fleet has
                # meaningfully MORE free VRAM than it did then, else we'd re-place it to the same spot
                # every sweep. Cleared on a fully-GPU promotion (below) or when room grows past it.
                _stuck = self._juggle_stuck.get(m.base or fr)
                if _stuck is not None and _fleet_free <= _stuck + 0.5:
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
            # 3) HITLESS swap: engage the barrier (holds NEW requests for the ~10-20s re-place ONLY,
            #    never a long drain) -> reconfigure VRAM-first -> release. The candidate was idle when
            #    selected; a request can only have slipped in during the fit-check awaits, so re-check
            #    under the now-engaged barrier and skip (don't stall it) if the model just went busy.
            gate = asyncio.Event()                          # CLEAR (present) = barrier engaged
            self._promote_gates[base] = gate
            try:
                if any((r.active or 0) > 0 or (r.queued or 0) > 0 for r in self.replicas_of(base)):
                    return                                   # became busy — try again next sweep at a gap
                log_activity(f"juggler: promoting {fr} to VRAM-only "
                             f"({self._model_cpu_frac(m)*100:.0f}% was on CPU) [{reason}] — "
                             f"hitless re-place")
                await self.reconfigure(fr, tp=getattr(m, "tp_size", 1), ctx=m.ctx,
                                       quant=(m.quant or "none"), consolidate=True,
                                       prefer_vram=True, cpu_only=False)
                newm = self.models.get(fr)
                nf = self._model_cpu_frac(newm) if newm else 1.0
                if nf <= 0.02:
                    self._juggle_stuck.pop(base, None)      # success — clear any anti-churn record
                    log_activity(f"juggler: {fr} re-placed — now fully on GPU")
                else:
                    # #anti-churn: the re-place couldn't reach full-GPU (the fit-check was optimistic /
                    # the fleet can't hold it all on VRAM right now). Record the room level so the sweep
                    # won't re-place it to the same spot every ~60s — it retries only once the fleet
                    # frees meaningfully MORE VRAM than this.
                    self._juggle_stuck[base] = _fleet_free
                    log_activity(f"juggler: {fr} re-placed to {(1.0 - nf) * 100:.0f}% on GPU "
                                 f"({nf * 100:.0f}% still on CPU) — best fit for now; won't retry "
                                 f"until the fleet frees more VRAM")
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

    # --- #load-faster: one-click "upgrade a resident model to a faster placement" (dashboard badge) ---
    # Detection reuses the juggler's live-free-VRAM planner; the APPLY reuses the #juggler barrier so a
    # click is HITLESS (parked clients ride the swap). Distinct from _maybe_juggle in TWO ways only:
    # (a) it also flags multi-node->fewer-node consolidation (not just CPU-spill->VRAM), and (b) on click
    # it WAITS for the in-flight reply to drain (then forces after a bound) instead of skipping busy
    # models. Kept SELF-CONTAINED (its own atomic reload + rollback) so a bug here can never destabilize
    # the auto juggler / wedge self-heal, which stay on the untouched reconfigure().
    LOAD_FASTER_DRAIN_S = 120.0     # wait this long for the current reply to finish, then FORCE the swap

    @staticmethod
    def _placement_label(cpu_frac: float, nstages: int) -> str:
        """Short human label for a placement, for the badge tooltip 'from X -> to Y'."""
        if nstages <= 1 and cpu_frac <= 0.02:
            return "single GPU (VRAM-resident)"
        parts = [f"{nstages} node" + ("s" if nstages != 1 else "")]
        if cpu_frac > 0.02:
            parts.append(f"{cpu_frac * 100:.0f}% on CPU")
        return ", ".join(parts)

    def _plan_vram_first(self, m):
        """Dry-run the fewest-nodes VRAM-first plan for resident model `m` against LIVE-free VRAM (+ this
        model's own on-GPU bytes, reclaimed on reload) — the SAME basis as _juggle_would_fit_vram, so the
        badge never promises a placement a real re-place can't reach. Returns (plan, cpu_weight_frac,
        n_stages), or (None, 1.0, 99) if nothing plans."""
        try:
            spec = m.spec
            own: dict = {}
            for s in m.plan.stages:
                own[s.node_id] = own.get(s.node_id, 0) + (getattr(s, "gpu_bytes", 0) or 0)
            _, _res_vram = self._reserved_bytes()
            mems = []
            for n in registry.alive_sorted():
                fv = max(0.0, self._node_live_free_vram_gb(
                    n, own_bytes=own.get(n.node_id, 0), res_vram=_res_vram) - PLAN_VRAM_FLOOR_GB)
                ram = n.eff_ram_gb - (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
                # #unified-mem (audit #17): shared-pool clamp — see _juggle_would_fit_vram; keeps
                # the upgrade badge from promising a placement built on double-counted bytes.
                tot, fv = self._unified_mem_clamp(
                    n, max(0.0, ram) + fv, fv,
                    (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
                    + PLAN_VRAM_FLOOR_GB + _res_vram.get(n.node_id, 0) / GB,
                    own_gb=own.get(n.node_id, 0) / GB)
                mems.append(NodeMem(n.node_id, n.hostname, int(tot * GB), int(fv * GB)))
            p = plan_pipeline(spec, mems, m.ctx, consolidate=True, prefer_vram=True)
            if not p.ok:
                return None, 1.0, 99
            assess = _assess_placement(spec, m.ctx, mems, p.stages, cpu_only=False)
            return p, (assess.get("cpu_weight_frac", 0.0) or 0.0), len(p.stages)
        except Exception:
            return None, 1.0, 99

    def _upgrade_for(self, m):
        """An {available, from, to, reason} upgrade suggestion for resident model `m`, or None. An
        upgrade exists when the fewest-nodes VRAM-first plan achievable with CURRENTLY-free fleet VRAM is
        strictly better than the current placement — LESS CPU spill or FEWER nodes. Text models only
        (media/embedding aren't served through the generate() barrier and never spill by mistake)."""
        if (getattr(m.spec, "is_embedding", False) or getattr(m, "is_embedding", False)
                or getattr(m, "is_t2i", False) or getattr(m, "is_t2a", False)
                or getattr(m, "is_tts", False)):
            return None
        cur_cpu = self._model_cpu_frac(m)
        cur_n = len(m.plan.stages)
        if cur_cpu <= 0.02 and cur_n <= 1:
            return None                                   # already single-GPU + VRAM-resident: optimal
        p, new_cpu, new_n = self._plan_vram_first(m)
        if p is None:
            return None
        less_cpu = new_cpu < cur_cpu - 0.02 and new_n <= cur_n
        fewer_nodes = new_n < cur_n and new_cpu <= cur_cpu + 0.02
        if not (less_cpu or fewer_nodes):
            return None
        why = []
        if less_cpu:
            why.append("more of it now fits in VRAM (less CPU spill)")
        if fewer_nodes:
            why.append(f"{cur_n}→{new_n} node{'s' if new_n != 1 else ''} (fewer network hops)")
        return {"available": True, "from": self._placement_label(cur_cpu, cur_n),
                "to": self._placement_label(new_cpu, new_n), "reason": "; ".join(why)}

    async def refresh_upgrades(self, min_interval: float = 30.0) -> None:
        """Throttled: recompute each resident text model's `.upgrade` at most every ~min_interval s
        (called from the /status route, polled ~2s). A cheap dry-run planner per loaded LLM. Never raises."""
        now_t = time.time()
        if now_t - getattr(self, "_upgrade_scan_ts", 0.0) < min_interval:
            return
        self._upgrade_scan_ts = now_t
        for m in list(self.models.values()):
            try:
                m.upgrade = self._upgrade_for(m)
            except Exception:
                m.upgrade = None

    async def load_faster(self, friendly: str) -> dict:
        """#load-faster APPLY: DRAIN the in-flight reply (wait up to LOAD_FASTER_DRAIN_S, then FORCE),
        then hitlessly re-place the model VRAM-first / fewest-nodes, PRESERVING its full config
        (ctx/quant/tp/kv_quant/kv_offload/sampling defaults), rolling back to a working copy if the faster
        layout fails. Reuses the #juggler barrier so parked clients ride the swap (connection pauses, no
        reconnect). Serialized under _juggle_lock. `friendly` must already be resolved. Text models only."""
        m = self.models.get(friendly)
        if m is None:
            return {"ok": False, "error": f"'{friendly}' is not resident"}
        if (getattr(m.spec, "is_embedding", False) or getattr(m, "is_embedding", False)
                or getattr(m, "is_t2i", False) or getattr(m, "is_t2a", False)
                or getattr(m, "is_tts", False)):
            return {"ok": False, "error": "load-faster applies to text models only"}
        base = m.base or friendly
        async with self._juggle_lock:
            m = self.models.get(friendly)
            if m is None:
                return {"ok": False, "error": "model is no longer resident"}
            up = self._upgrade_for(m)
            if up is None:
                return {"ok": False, "error": "no faster placement is available right now"}
            # snapshot FULL config so the swap is config-neutral (bare reconfigure would silently drop
            # kv_quant / kv_offload / sampling defaults); a kv_quant of 'none' maps to load()'s '' default.
            _kvq = getattr(m, "kv_quant", "none") or "none"
            snap = dict(ctx=m.ctx, quant=(m.quant or "none"), tp=getattr(m, "tp_size", 1),
                        kv_quant=(_kvq if _kvq != "none" else ""),
                        kv_offload=bool(getattr(m, "kv_offload", False)),
                        kv_slots=max(1, int(getattr(m, "kv_slots", 1) or 1)),   # #kv-slots kept
                        default_temp=getattr(m, "default_temperature", None),
                        default_min_p=getattr(m, "default_min_p", None),
                        moe_offload=bool(getattr(m, "moe_offload", False)))
            cpu0, n0 = self._model_cpu_frac(m), len(m.plan.stages)
            gate = asyncio.Event()                          # present+clear = barrier engaged (holds new reqs)
            self._promote_gates[base] = gate
            self.reconfiguring = {"model": _ollama_name(friendly), "from": up["from"], "to": up["to"]}
            try:
                deadline = time.time() + self.LOAD_FASTER_DRAIN_S     # DRAIN the in-flight reply, then FORCE
                forced = False
                while any((r.active or 0) > 0 or (r.queued or 0) > 0 for r in self.replicas_of(base)):
                    if time.time() >= deadline:
                        forced = True
                        break
                    await asyncio.sleep(0.5)
                log_activity(f"load-faster: re-placing {friendly} VRAM-first ({up['from']} -> {up['to']})"
                             + (" [forced: dropping an in-flight generation]" if forced else "") + " — hitless")
                try:
                    await self.load(friendly, snap["ctx"], consolidate=True, prefer_vram=True,
                                    quant=snap["quant"], tp=snap["tp"], cpu_only=False, force=True,
                                    kv_quant=snap["kv_quant"], kv_offload=snap["kv_offload"],
                                    kv_slots=snap["kv_slots"],
                                    default_temp=snap["default_temp"], default_min_p=snap["default_min_p"],
                                    moe_offload=snap["moe_offload"])
                except Exception as exc:
                    log_activity(f"load-faster: {friendly} faster placement FAILED ({exc!r}) — rolling back")
                    try:
                        await self.load(friendly, snap["ctx"], quant=snap["quant"], tp=snap["tp"],
                                        force=True, kv_quant=snap["kv_quant"], kv_offload=snap["kv_offload"],
                                        kv_slots=snap["kv_slots"],
                                        default_temp=snap["default_temp"],
                                        default_min_p=snap["default_min_p"], moe_offload=snap["moe_offload"])
                    except Exception as exc2:
                        log_activity(f"load-faster: {friendly} ROLLBACK ALSO FAILED ({exc2!r}) — NOT resident")
                        return {"ok": False, "error": f"upgrade failed AND rollback failed: {exc} || {exc2}"}
                    return {"ok": False, "error": f"upgrade failed, rolled back to a working copy: {exc}"}
                newm = self.models.get(friendly)
                if newm is not None:
                    newm.upgrade = self._upgrade_for(newm)      # refresh the badge state immediately
                cpu1, n1 = (self._model_cpu_frac(newm), len(newm.plan.stages)) if newm else (1.0, 0)
                log_activity(f"load-faster: {friendly} now {self._placement_label(cpu1, n1)}")
                return {"ok": True, "model": _ollama_name(friendly), "forced": forced,
                        "from": self._placement_label(cpu0, n0), "to": self._placement_label(cpu1, n1)}
            finally:
                gate.set()
                self._promote_gates.pop(base, None)
                self.reconfiguring = None

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

    # ---- #adopt: controller-restart shard adoption ------------------------------------------
    # When THIS controller restarts, workers (on adopt-capable code) KEEP their loaded shards
    # and report them in their re-register message (`loaded`: one entry per model, carrying the
    # ORIGINAL load-message `assign` this controller's predecessor sent, plus live byte counts).
    # Instead of re-streaming tens of GB, the controller REBUILDS each model's LoadedModel from
    # those recipes: spec/tokenizer/eos re-derive from the model dir on disk, StageAssign/
    # PlanResult from the assignments, and the stage0 data connection re-dials fresh. The
    # workers' inter-hop data plane self-heals lazily (#stage0-stale-reconnect freshens every
    # hop at prefill), so an adopted pipeline serves its first request without ceremony.
    # NOT adopted (reload required, shards freed by the grace sweep): tensor-parallel models
    # (mesh state doesn't survive the controller's tp_root bookkeeping) and anything whose
    # coverage never completes (a stage's worker stayed down). Spec-decode DRAFTS are
    # controller-local and are not re-attached — reload to restore speculative decode.
    _ADOPT_GRACE_S = 90.0     # unassembled reports older than this get their shards freed

    def _adopt_state(self) -> tuple[dict, bool]:
        """Lazy-init the adoption pool (engine __init__ lives in server.py; keeping state
        creation here makes adoption self-contained for mixed-version fleets)."""
        pool = self.__dict__.setdefault("_adopt_pool", {})
        have_sweeper = self.__dict__.get("_adopt_sweeper") is not None
        return pool, have_sweeper

    async def adopt_worker_models(self, node, inv: list) -> None:
        """Ingest one re-registering worker's kept-model inventory and adopt what completes.
        Called from control_plane's register path (fire-and-forget task)."""
        if getattr(self, "updating", False) or not inv:
            return
        pool, have_sweeper = self._adopt_state()
        touched: set = set()
        for e in inv:
            if not isinstance(e, dict):
                continue
            a = e.get("assign") or {}
            tid = e.get("model_id") or a.get("model_id")
            if not tid or not isinstance(a, dict):
                continue
            rep = {"node": node, "assign": a,
                   "gpu_bytes": int(e.get("gpu_bytes") or 0),
                   "gpu_kv_bytes": int(e.get("gpu_kv_bytes") or 0),
                   "loaded_bytes": int(e.get("loaded_bytes") or 0),
                   "media": e.get("media"), "ts": time.time()}
            slot = pool.setdefault(tid, {"reports": [], "first_ts": time.time()})
            # A node that re-reports (reconnect flap) replaces its old entry, not duplicates it.
            slot["reports"] = [r for r in slot["reports"]
                               if r["node"].hostname != node.hostname] + [rep]
            touched.add(tid)
            print(f"[adopt] {node.hostname} holds {tid} "
                  f"(stage {a.get('stage', 0)}, kind {a.get('kind') or 'llm'}, "
                  f"{rep['loaded_bytes'] / GB:.1f} GB)")
        if not have_sweeper:
            self._adopt_sweeper = asyncio.create_task(self._adopt_sweep())
        for tid in touched:
            try:
                await self._adopt_try_assemble(tid)
            except Exception as exc:
                # Assembly failure = fall back to a normal reload; the sweep frees the shards.
                print(f"[adopt] {tid}: assembly failed ({exc!r}) — will free and rely on reload")

    async def _adopt_try_assemble(self, target_id: str) -> None:
        """Adopt `target_id` if its reports now cover the whole model. Holds engine.lock for
        the state mutation (same discipline as load())."""
        pool, _ = self._adopt_state()
        slot = pool.get(target_id)
        if slot is None:
            return
        reports = slot["reports"]
        if not reports:
            return
        a0 = reports[0]["assign"]
        kind = a0.get("kind") or ""
        if int(a0.get("tp_size", 1) or 1) > 1:
            return   # TP is not adoptable — the sweep frees it after grace
        # ---- coverage check (outside the lock: read-only) ----
        if kind in ("embedding", "t2i", "t2a", "tts"):
            picked = [reports[0]]                       # single-node kinds: any one report serves
        else:
            n_stages = int(a0.get("num_stages", 0) or 0)
            if not n_stages:
                return   # not a pipeline recipe we understand (e.g. TP peer) — sweep frees it
            by_stage: dict = {}
            for r in reports:
                by_stage.setdefault(int(r["assign"].get("stage", -1)), r)
            if set(by_stage) != set(range(n_stages)):
                return   # a stage's worker hasn't re-registered (yet)
            picked = [by_stage[i] for i in range(n_stages)]
            for i in range(1, n_stages):                # contiguous layer coverage, in order
                if int(picked[i]["assign"].get("layer_start", -1)) != \
                        int(picked[i - 1]["assign"].get("layer_end", -2)):
                    return
            if not (picked[0]["assign"].get("has_embed") and picked[-1]["assign"].get("has_head")):
                return
        # #adopt-canonical: a target can be registered under BOTH an alias key and its
        # canonical key (om3nbox: 'qwen2.5-14b' aliases 'qwen2.5-14b-instruct', same HF
        # target). First-match adoption resurrected the model under the ALIAS key — which
        # resolve_model_name() maps AWAY from, so the adopted row was unaddressable (every
        # unload/load by name hit the canonical entry instead, and a later canonical load
        # built a doppelganger row). Prefer the key the resolver treats as canonical: one
        # MODEL_ALIASES doesn't map elsewhere. (MODEL_ALIASES arrives via state.bind, like
        # MODELS; plain dict constant — snapshot-safe.)
        _cands = [f for f, v in MODELS.items() if v and v[0] == target_id]
        _canon = [f for f in _cands if MODEL_ALIASES.get(f, f) == f]
        friendly = (_canon or _cands or [target_id])[0]
        reg_key = friendly
        async with self.lock:
            if pool.get(target_id) is not slot:          # raced with another assemble/sweep
                return
            if reg_key in self.models or reg_key in self.loadings or \
                    getattr(self, "updating", False):
                # Already resident (reloaded faster than we adopted) — these shards are
                # superseded; leave them for the sweep to free.
                return
            lm = await self._adopt_build_lm(friendly, reg_key, target_id, kind, picked)
            if lm is None:
                return
            self.models[reg_key] = lm
            # Consume the used reports; replica leftovers (same target on other nodes) stay
            # behind for the sweep to free — replica reg_keys aren't reconstructible.
            slot["reports"] = [r for r in reports if r not in picked]
            if not slot["reports"]:
                pool.pop(target_id, None)
        registry.dirty = False
        _hosts = [s.hostname for s in lm.plan.stages]
        _gpu = sum(s.gpu_bytes for s in lm.plan.stages) / GB
        print(f"[adopt] ADOPTED {reg_key} from {_hosts} ({_gpu:.1f} GB GPU) — no reload")
        log_activity(f"ADOPTED {_ollama_name(reg_key)} from the running fleet "
                     f"({len(lm.plan.stages)} stage(s) on {', '.join(_hosts)}, "
                     f"{_gpu:.1f} GB GPU) — controller restarted without reloading")

    async def _adopt_build_lm(self, friendly: str, reg_key: str, target_id: str,
                              kind: str, picked: list) -> Optional["LoadedModel"]:
        """Rebuild the LoadedModel for an adopted set — the mirror of each load path's tail,
        minus the actual weight streaming. Caller holds self.lock."""
        await self.ensure_data_listener()
        a0 = picked[0]["assign"]
        node0 = picked[0]["node"]
        now = time.time()
        tot0 = picked[0]["loaded_bytes"]

        def _stage(r, layer_start, layer_end, est):
            st = StageAssign(node_id=r["node"].node_id, hostname=r["node"].hostname,
                             layer_start=layer_start, layer_end=layer_end,
                             has_embed=bool(r["assign"].get("has_embed", layer_start == 0)),
                             has_head=bool(r["assign"].get("has_head", True)),
                             est_bytes=int(est), usable_bytes=int(r["node"].usable_total_gb * GB),
                             gpu_bytes=r["gpu_bytes"], loaded_bytes=r["loaded_bytes"])
            st.gpu_kv_bytes = r["gpu_kv_bytes"]          # coexistence VRAM reserve (#95)
            nd = r["node"]
            nd.shard_gpu_bytes = r["gpu_bytes"]          # mirror the load tail's node bookkeeping
            nd.load_state = "ready"
            if nd.stage is None:
                nd.stage, nd.layer_start, nd.layer_end = 0, layer_start, layer_end
            return st

        if kind == "embedding":
            spec = resolve_spec(friendly)
            model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
            spec = await asyncio.to_thread(spec_with_measurements, spec, model_dir)
            tok = await asyncio.to_thread(_get_tokenizer, target_id)
            st = _stage(picked[0], 0, spec.num_layers, spec.total_weight_bytes)
            plan = PlanResult(ok=True, model=spec.name, ctx_len=spec.max_ctx,
                              num_layers=spec.num_layers, pool_usable_gb=node0.usable_total_gb,
                              required_gb=spec.total_weight_bytes / GB, stages=[st])
            dial = (_dial_host(node0.data_host), node0.data_port)
            lm = LoadedModel(reg_key, target_id, spec, spec.max_ctx, plan,
                             [node0.node_id], tok, set(), now, quant="none",
                             stage0_writer=await self._connect_retry(*dial),
                             last_used=now, stage0_dial=dial, last_send_ts=now)
            lm.base, lm.replica_idx = friendly, 0
            lm.plan_basis = "embedding: single-node (adopted)"
            return lm

        if kind in ("t2i", "t2a", "tts"):
            # Media kinds: one co-located worker holds the whole pipeline. Their specs carry
            # accounting dims only (serving never consults them) — mirror each tail's shape.
            model_dir = a0.get("model_dir") or ""
            if kind == "t2i":
                _tc = {}
                with contextlib.suppress(Exception):
                    with open(os.path.join(model_dir, "transformer", "config.json"),
                              encoding="utf-8") as _fh:
                        _tc = json.load(_fh)
                _layers = int(_tc.get("num_layers") or 60)
                _heads = int(_tc.get("num_attention_heads") or 24)
                _hd = int(_tc.get("attention_head_dim") or 128)
                spec = ModelSpec(name=friendly, hidden_size=_heads * _hd, num_layers=_layers,
                                 num_heads=_heads, num_kv_heads=_heads, head_dim=_hd,
                                 intermediate_size=int(_tc.get("joint_attention_dim") or 3584),
                                 vocab_size=0, tie_embeddings=True, max_ctx=0, arch="t2i",
                                 meas_layer_w=max(1, tot0 // max(1, _layers)),
                                 meas_embed=0, meas_head=0, meas_norm=0,
                                 meas_params=max(1, tot0))
            elif kind == "t2a":
                spec = ModelSpec(name=friendly, hidden_size=2560, num_layers=24, num_heads=20,
                                 num_kv_heads=20, head_dim=128, intermediate_size=6400,
                                 vocab_size=0, tie_embeddings=True, max_ctx=0, arch="t2a",
                                 meas_layer_w=max(1, tot0 // 24), meas_embed=0, meas_head=0,
                                 meas_norm=0, meas_params=max(1, tot0))
            else:
                spec = ModelSpec(name=friendly, hidden_size=1, num_layers=1, num_heads=1,
                                 num_kv_heads=1, head_dim=1, intermediate_size=1, vocab_size=0,
                                 tie_embeddings=True, max_ctx=0, arch="tts",
                                 meas_layer_w=1, meas_embed=0, meas_head=0, meas_norm=0,
                                 meas_params=max(1, tot0))
            n_layers = spec.num_layers if kind == "t2i" else 1
            st = _stage(picked[0], 0, n_layers, tot0 or 1)
            plan = PlanResult(ok=True, model=spec.name, ctx_len=0, num_layers=n_layers,
                              pool_usable_gb=node0.usable_total_gb,
                              required_gb=(tot0 or 1) / GB, stages=[st])
            dial = (_dial_host(node0.data_host), node0.data_port)
            offload = bool(a0.get("t2i_offload") or a0.get("t2a_offload"))
            if kind == "t2i":
                q = ("bf16-off" if offload else
                     ("none" if (a0.get("quant") or "none") == "none"
                      else f"int4-e{int(a0.get('t2i_edge', 2) or 2)}"))
            else:
                q = "none"
            lm = LoadedModel(reg_key, target_id, spec, 0, plan, [node0.node_id],
                             None, set(), now, quant=q,
                             stage0_writer=await self._connect_retry(*dial),
                             last_used=now, stage0_dial=dial, last_send_ts=now)
            lm.base, lm.replica_idx = friendly, 0
            lm.plan_basis = f"{kind}: single-node (adopted)"
            if kind == "t2i":
                lm.is_t2i = True
            elif kind == "t2a":
                lm.is_t2a = True
                lm.t2a_offload = offload
            else:
                lm.is_tts = True
                lm.is_kokoro = True
            lm.media = picked[0].get("media")            # #media-detail (may be None)
            return lm

        # ---- pipeline LLM ----
        spec = resolve_spec(friendly)
        model_dir = await asyncio.to_thread(_controller_model_dir, target_id)
        spec = await asyncio.to_thread(spec_with_measurements, spec, model_dir)
        quant = a0.get("quant") or "none"
        spec = spec.for_quant(quant)
        last_end = int(picked[-1]["assign"].get("layer_end", 0))
        if last_end != spec.num_layers:
            print(f"[adopt] {reg_key}: reported layers 0-{last_end} != spec {spec.num_layers} "
                  f"— not adopting (model changed on disk?)")
            return None
        ctx = int(a0.get("ctx", 0) or 0)
        stages = [_stage(r, int(r["assign"]["layer_start"]), int(r["assign"]["layer_end"]),
                         r["loaded_bytes"] or max(1, spec.total_weight_bytes
                                                  * max(1, int(r["assign"]["layer_end"])
                                                        - int(r["assign"]["layer_start"]))
                                                  // max(1, spec.num_layers)))
                  for r in picked]
        plan = PlanResult(ok=True, model=spec.name, ctx_len=ctx, num_layers=spec.num_layers,
                          pool_usable_gb=sum(r["node"].usable_total_gb for r in picked),
                          required_gb=sum(s.est_bytes for s in stages) / GB, stages=stages)
        tok = await asyncio.to_thread(_get_tokenizer, target_id)
        eos = self._eos_ids(tok)
        dial = (_dial_host(node0.data_host), node0.data_port)
        # #kv-slots: the workers KEPT their slot-keyed shards (the load msg in `assign` carries
        # kv_slots) — re-adopt at the same C and rebuild the controller-side slot pool, so a
        # hitless controller restart doesn't silently serialize a slotted model back to C=1.
        # The adopted spec's KV accounting scales xC too (coexistence reserve honesty).
        _kvs = max(1, int(a0.get("kv_slots") or 1))
        spec = spec.for_kv_slots(_kvs)
        lm = LoadedModel(reg_key, target_id, spec, ctx, plan,
                         [s.node_id for s in stages], tok, eos, now,
                         quant=quant, kv_quant=(a0.get("kv_quant") or "none"),
                         kv_offload=bool(a0.get("kv_offload")),
                         kv_slots=_kvs,
                         stage0_writer=await self._connect_retry(*dial),
                         last_used=now, stage0_dial=dial, last_send_ts=now)
        self._init_slot_pool(lm)   # #kv-slots (no-op at C=1)
        lm.base, lm.replica_idx = friendly, 0
        lm.plan_basis = "adopted from running workers (controller restart)"
        return lm

    async def _adopt_sweep(self) -> None:
        """Free orphaned kept shards: reports that never assembled within the grace window
        (a stage's worker stayed down, TP recipes, replica leftovers, or shards superseded
        by a faster reload). Without this they'd silently pin worker RAM/VRAM forever."""
        pool, _ = self._adopt_state()
        try:
            while True:
                await asyncio.sleep(15.0)
                if not pool:
                    continue
                now = time.time()
                for tid in list(pool):
                    slot = pool.get(tid)
                    if not slot or now - slot["first_ts"] < self._ADOPT_GRACE_S:
                        continue
                    for r in slot.get("reports") or []:
                        nid = r["node"].node_id
                        link = self.links.get(nid)
                        if link is not None:
                            with contextlib.suppress(Exception):
                                await link.send({"type": "unload", "model_id": tid})
                        print(f"[adopt] freeing unadopted shard of {tid} on "
                              f"{r['node'].hostname} (grace expired)")
                    if slot.get("reports"):
                        log_activity(f"adoption incomplete for {_ollama_name(tid)} — freed "
                                     f"kept shard(s) on "
                                     + ", ".join(r["node"].hostname for r in slot["reports"])
                                     + " (reload to serve it)")
                    pool.pop(tid, None)
        finally:
            self.__dict__["_adopt_sweeper"] = None
