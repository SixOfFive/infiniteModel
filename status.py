"""status.py: the read-only status-building layer relocated from server.py (m4c155 code-split).
Holds build_status (the big /status + dashboard payload builder), _model_entry, _loading_view,
_tag_entry — all BYTE-IDENTICAL to the originals. Module globals (engine, registry, MODELS,
ENGINE_CONFIG, DOWNLOAD_STATE, metric_rates, shard_cache_status …) are injected at startup by
state.bind() — see state.py. server.py back-imports build_status/_tag_entry (called by
routes_dashboard/routes_api) so they stay in the published namespace. Controller-only leaf;
in EXTRA_UPDATE_FILES. (Reads DOWNLOAD_STATE, kept injection-safe by the m4c155 in-place fix in
server.load_download_state.)
"""
from __future__ import annotations




def _tag_entry(friendly: str) -> dict:
    target, _draft = MODELS[friendly]
    spec = resolve_spec(target)   # built-in or config-derived (custom models)
    size = _display_weight_bytes(target, spec) if spec else 0
    details = _details(spec) if spec else {
        "parent_model": "", "format": "safetensors", "family": "unknown",
        "families": ["unknown"], "parameter_size": "", "quantization_level": "BF16"}
    disp = _ollama_name(friendly)   # 'family:size' (the size IS the tag — no ':latest' on top)
    return {
        "name": disp, "model": disp,
        "modified_at": _iso(START_TIME), "size": size,
        "digest": _digest(target), "details": details,
        "infinitemodel": {"target": target, "draft": _draft, "distributed": True},
    }


# ---------------------------------------------------------------------------
# Status + HTTP API + dashboard
# ---------------------------------------------------------------------------

def _loading_view(ld: Optional[dict]) -> Optional[dict]:
    """Enrich a live load/compile card (from engine.loadings / engine.compiling) with a timer
    (elapsed + ETA) without mutating the original. ETA = elapsed * (1-frac)/frac from progress (ready/total);
    only shown once a few percent in (early fractions give wild estimates). Returns a shallow
    copy so the live object the load loop mutates stays clean. None -> None (no load running)."""
    if not ld:
        return ld
    started = ld.get("started")
    out = dict(ld)
    if started:
        elapsed = max(0.0, time.time() - started)
        out["elapsed_s"] = round(elapsed, 1)
        total = ld.get("total") or 0
        ready = ld.get("ready") or 0
        frac = (ready / total) if total > 0 else 0.0
        # need a stable-ish fraction before an ETA is meaningful (>=3% in); cap at 4h display
        out["eta_s"] = round(min(4 * 3600, elapsed * (1 - frac) / frac)) if frac >= 0.03 else None
    return out


def build_status() -> dict:
    nodes = registry.alive_sorted()
    # Pool aggregate respects the per-node CPU/GPU tier toggles: deselecting a node's CPU
    # drops its RAM from total/used/free; deselecting its GPU drops its VRAM. Per-node rows
    # still show the machine's real memory (heartbeats keep polling it regardless of tier).
    pool_total = sum((n.total_mem_gb if n.ram_enabled else 0.0)
                     + (n.vram_total_gb if n.vram_enabled else 0.0) for n in nodes)
    pool_ram = sum(n.eff_ram_gb for n in nodes)    # enabled tiers only (planner budget)
    pool_vram = sum(n.eff_vram_gb for n in nodes)  # disabled VRAM leaves the pool
    pool_usable = pool_ram + pool_vram            # usable pool now includes VRAM
    # Physical (pre-reserve) totals split by form. The dashboard's GPU/RAM pool bars render
    # LIVE used (total - free) against THESE so used and total share one base — the usable
    # splits above subtract OS/VRAM reserve and don't line up with the physical *_free below
    # (free is measured against the bigger physical total → used went negative; see #pool-base).
    pool_ram_total = sum(n.total_mem_gb for n in nodes if n.ram_enabled)
    pool_vram_total = sum(n.vram_total_gb for n in nodes if n.vram_enabled)
    # LIVE physical used/free against the STABLE total — so the dashboard shows usage CLIMBING
    # as models load, not the total shrinking. RAM used = total-free (heartbeat); VRAM used = heartbeat.
    pool_used = sum(((n.total_mem_gb - n.free_mem_gb) if n.ram_enabled else 0.0)
                    + (n.vram_used_gb if n.vram_enabled else 0.0) for n in nodes)
    pool_used = max(0.0, min(pool_total, pool_used))
    pool_free = pool_total - pool_used
    # Split the LIVE physical free into RAM vs VRAM (what's available in what FORM), tier-aware.
    pool_ram_free = max(0.0, sum((n.free_mem_gb if n.ram_enabled else 0.0) for n in nodes))
    pool_vram_free = max(0.0, sum(((n.vram_total_gb - n.vram_used_gb) if n.vram_enabled else 0.0)
                                  for n in nodes))
    # Split "used" into ENGINE (our python + loaded-model footprint) vs OS/other. mmap subtlety:
    # safetensors weights are memory-mapped, so until faulted in they sit in the OS PAGE CACHE —
    # reported as reclaimable/'available' (so live free_mem HIDES them) and absent from proc_rss.
    # So the live used/engine UNDERCOUNT a loaded model's RAM weights (they leak into 'free', and
    # 'engine' showed smaller than the model card). Fix: size the models from their SPEC (the same
    # bytes the LOADED MODEL card reports), add the cache-hidden RAM back into 'used', and count
    # each model's RAM ONCE — faulted into RSS OR sitting in cache — via max(). Reconciles the bar.
    try:
        import psutil as _ps
        ctrl_rss_gb = _ps.Process().memory_info().rss / GB
    except Exception:
        ctrl_rss_gb = 0.0
    worker_rss_gb = sum(n.proc_rss_gb for n in nodes if n.ram_enabled)
    model_weight_gb = sum(lm.spec.total_weight_bytes for lm in engine.models.values()) / GB
    model_kv_gb = sum(lm.spec.kv_bytes_per_layer(lm.kv_pos) * lm.spec.num_layers
                      for lm in engine.models.values()) / GB
    model_vram_gb = sum(s.gpu_bytes for lm in engine.models.values()
                        for s in lm.plan.stages) / GB
    model_ram_gb = max(0.0, model_weight_gb - model_vram_gb) + model_kv_gb   # model's RAM commitment
    mmap_hidden = max(0.0, model_ram_gb - worker_rss_gb)   # model RAM in page-cache (hidden in 'free')
    pool_used = max(0.0, min(pool_total, pool_used + mmap_hidden))
    pool_free = pool_total - pool_used
    pool_ram_free = max(0.0, pool_ram_free - mmap_hidden)   # the hidden weights are RAM, not free
    pool_engine = max(0.0, min(pool_used,
                               ctrl_rss_gb + max(worker_rss_gb, model_ram_gb) + model_vram_gb))
    pool_os = max(0.0, pool_used - pool_engine)
    resident = list(engine.models.values())
    primary = max(resident, key=lambda m: m.last_used) if resident else None

    def _loaded_dict(lm: LoadedModel) -> dict:
        # Report WEIGHTS and KV separately so the card reconciles with the pool's engine
        # bar (measured), instead of the old est_bytes which folded in the FULL ctx KV
        # reservation (mostly empty at low ctx -> looked far bigger than measured usage).
        #   weights: VRAM = worker-reported on-GPU bytes (measured); RAM = the rest of the
        #            model's weights (mmap-backed, in the OS page cache until faulted in).
        #   KV: reserved for the full ctx vs actually used so far (from kv_pos). KV is
        #       allocated lazily during generation, so at ctx 0 it's ~empty.
        vram_used = sum(s.gpu_bytes for s in lm.plan.stages)
        # #real-stats: RAM-resident weights = worker-MEASURED total minus measured on-GPU. The old
        # basis (spec.total_weight_bytes, a formulaic quant ESTIMATE) overshoots real packed MoE
        # int4 by ~10%, which fabricated a phantom "1.9 GB RAM / cpu_frac 0.106" on a fully-GPU
        # qwen3-30b-a3b. Spec fallback only when no stage reported loaded_bytes (old worker).
        weights_total = (sum(getattr(s, "loaded_bytes", 0) or 0 for s in lm.plan.stages)
                         or lm.spec.total_weight_bytes)
        ram_weights = max(0, weights_total - vram_used)
        kv_reserved = lm.spec.kv_bytes_per_layer(lm.ctx) * lm.spec.num_layers
        # #kv-slots: kv_reserved is honestly xC (spec.kv_slots is baked in — C streams ARE
        # reserved), but kv_pos tracks only the most recent generation's depth, so scale the
        # USED figure back to the single-stream estimate rather than claim C x one stream.
        kv_used = (lm.spec.kv_bytes_per_layer(lm.kv_pos) * lm.spec.num_layers
                   // max(1, int(getattr(lm.spec, "kv_slots", 1) or 1)))
        # #172: under TurboQuant KV the worker RESERVES the bit-packed footprint, not bf16 — so report
        # the honest reserved/used (packed all-layers + one bf16 dequant transient for reserved) instead
        # of the bf16 spec estimate, which would overstate the card ~4-5x. Falls back to the bf16
        # estimate on any error. kv_quant='none' leaves both untouched (bit-identical).
        _kvq = (getattr(lm, "kv_quant", "none") or "none")
        if _kvq != "none":
            try:
                import kv_quant
                _pt = kv_quant.kv_quant_bytes_per_token_per_layer(
                    _kvq, lm.spec.num_kv_heads, lm.spec.head_dim)
                if _pt > 0:
                    kv_reserved = _pt * lm.ctx * lm.spec.num_layers \
                        + lm.spec.kv_bytes_per_layer(lm.ctx)   # + one bf16 transient (mirrors worker)
                    kv_used = _pt * lm.kv_pos * lm.spec.num_layers
            except Exception:
                pass   # keep the bf16 spec estimate (conservative)
        _arch = (getattr(lm.spec, "arch", "") or "").lower()
        # best-effort MoE flag for the detail modal: fused/per-expert MoE arches all contain one of
        # these tokens ('moe' covers olmoe/qwen3*_moe; mixtral/minimax/deepseek_v2,v3 named directly).
        _is_moe = any(k in _arch for k in ("moe", "mixtral", "minimax", "deepseek_v"))
        # #media-detail: rich info block for a media (tts/t2i/t2a) model's detail modal — the
        # worker-reported metadata (voices/sample_rate/device for tts) plus derived device (from
        # whether any stage placed weights on GPU), weight size, and the last render's speed (RTF).
        _media = None
        if any(getattr(lm, f, False) for f in ("is_tts", "is_t2i", "is_t2a")):
            _media = dict(getattr(lm, "media", None) or {})
            _media["device"] = ("GPU" if any(getattr(s, "gpu_bytes", 0) > 0
                                             for s in lm.plan.stages) else "CPU")
            # measured weight size — prefer the worker's directly-reported loaded_bytes (rides in
            # `media`), else the stage bytes; the tts/t2i ModelSpec carries dummy dims so
            # spec.total_weight_bytes is ~0. (The CPU-load path can leave the stage bytes 0.)
            _lb = (_media.get("loaded_bytes")
                   or sum(getattr(s, "loaded_bytes", 0) or 0 for s in lm.plan.stages)
                   or lm.spec.total_weight_bytes)
            _media["size_gb"] = round(_lb / GB, 2)
            _media.pop("loaded_bytes", None)   # internal — size_gb is the public field
            if not _media.get("kind"):
                _media["kind"] = ("tts" if getattr(lm, "is_tts", False)
                                  else "t2i" if getattr(lm, "is_t2i", False) else "t2a")
            _lr, _la = getattr(lm, "last_render_s", None), getattr(lm, "last_audio_s", None)
            if _lr is not None:
                _media["last_render_s"] = round(float(_lr), 2)
            if _la is not None:
                _media["last_audio_s"] = round(float(_la), 2)
            if _lr and _la and float(_la) > 0:
                _media["last_rtf"] = round(float(_lr) / float(_la), 2)
        return {
            "friendly": lm.friendly, "display_name": _ollama_name(lm.friendly),  # 'qwen3:4b'
            "aliases": _aliases_for(lm.base or lm.friendly),  # display-form alias(es) -> shown under the name
            "target": lm.target_id, "ctx": lm.ctx,
            "base": lm.base or lm.friendly, "replica_idx": lm.replica_idx,  # data-parallel (#39)
            "active": lm.active, "queued": lm.queued,   # per-replica live load (#39 routing)
            "kv_pos": lm.kv_pos,   # tokens in the current/last generation's KV context
            # #prefill-progress: seconds since the last worker-reported per-layer forward progress
            # (heartbeat fwd_progress). Small + shrinking during a healthy prefill under load;
            # None = no report yet this gen. Observability for the endpoint-weather liveness path.
            "fwd_prog_age_s": (round(time.time() - lm.fwd_progress_ts, 1)
                               if getattr(lm, "fwd_progress_ts", 0.0) > 0 else None),
            # LIVE decode tok/s: the most-recent-gen rate WHILE generating, else 0 when idle —
            # last_tok_s lingers at its last value forever otherwise (card looked "busy" when idle).
            # The historical rate stays visible as ema_tok_s ("avg"). active==0 => idle => 0.
            "tok_s": round(lm.last_tok_s if lm.active > 0 else 0.0, 2),   # decode tok/s, live (#46)
            "ema_tok_s": round(lm.ema_tok_s, 2),     # smoothed decode tok/s across gens, historical (#46)
            # #detail: the RAW last-gen rate — NOT zeroed when idle, so the UI can freeze the last
            # HONEST measured tok/s between runs (tok_s above goes to 0 at idle by design). Never
            # recomputed while idle, so it can't drift into a dishonest number.
            "last_tok_s": round(getattr(lm, "last_tok_s", 0.0), 2),
            "quant": lm.quant,     # the quant this model was loaded with (none/int8)
            "kv_quant": getattr(lm, "kv_quant", "none"),     # #172 TurboQuant KV preset (none/turbo2/3/4)
            "kv_offload": bool(getattr(lm, "kv_offload", False)),  # #kv-offload: KV cache in system RAM
            # #kv-slots: per-replica decode slot count C + how many are currently leased.
            # C=1 (every legacy load): slots_active mirrors the classic active flag (0/1).
            "kv_slots": max(1, int(getattr(lm, "kv_slots", 1) or 1)),
            "slots_active": (int(getattr(lm, "slots_active", 0) or 0)
                             if int(getattr(lm, "kv_slots", 1) or 1) > 1
                             else min(1, int(lm.active or 0))),
            # #persist (autoload-on-restart) / #no-unload (absolute do-not-auto-unload veto): pin state
            # for this model, driving the detail-modal checkboxes. Keyed by friendly OR base (replicas).
            "persist": bool((getattr(lm, "friendly", None) in (ENGINE_CONFIG.get("persist_models") or {}))
                            or (getattr(lm, "base", None) in (ENGINE_CONFIG.get("persist_models") or {}))),
            "no_unload": bool((getattr(lm, "friendly", None) in (ENGINE_CONFIG.get("no_unload_models") or {}))
                              or (getattr(lm, "base", None) in (ENGINE_CONFIG.get("no_unload_models") or {}))),
            # #load-temp: per-model default temperature (None = unset -> requests default to 0.0)
            "def_temperature": getattr(lm, "default_temperature", None),
            # #min-p: per-model default min-p sampling floor (None = unset -> off)
            "def_min_p": getattr(lm, "default_min_p", None),
            # #runtime-knobs: the extended runtime-mutable sampling defaults (only SET keys:
            # top_p/top_k/repeat_penalty/repeat_last_n/presence_penalty/frequency_penalty/
            # seed/num_predict) — edited via POST /model_config, shown in the detail modal
            "sampling_defaults": dict(getattr(lm, "sampling_defaults", None) or {}),
            "tp_size": getattr(lm, "tp_size", 1),            # #88: TP width (1 = pipeline)
            "is_tp": getattr(lm, "tp_size", 1) > 1,          # #88: card shows TP vs pipeline + reconfigure
            "upgrade": getattr(lm, "upgrade", None) or None,  # #load-faster: {available,from,to,reason} or None
            "num_layers": lm.spec.num_layers, "params": _human_params(lm.spec),
            "size_gb": round(lm.spec.total_weight_bytes / GB, 2),
            "vram_used_gb": round(vram_used / GB, 2),
            "ram_used_gb": round(ram_weights / GB, 2),       # weights resident in RAM (not KV)
            # #cpu-bound-visibility: ACTUAL fraction of WEIGHTS on CPU (from worker-reported gpu_bytes,
            # not the pre-load estimate). A high value = the model is CPU-bound and will decode SLOWLY
            # (CPU layers are ~50-100x a GPU layer) — the dashboard badges it so "slow" isn't read as
            # "hung/wedged". This is the real cause of a multi-model fleet's later loads crawling.
            "cpu_frac": round(ram_weights / weights_total, 3) if weights_total else 0.0,
            "kv_reserved_gb": round(kv_reserved / GB, 2),    # KV space reserved for the full ctx
            "kv_used_gb": round(kv_used / GB, 2),            # KV actually used so far (kv_pos)
            "loaded_at": _iso(lm.loaded_at),
            "loaded_at_ts": lm.loaded_at,                    # epoch s -> live uptime in the modal
            "last_used_ts": lm.last_used,                    # epoch s -> "idle for ..." in the modal
            "plan_basis": getattr(lm, "plan_basis", ""),   # placement basis (#65)
            "warnings": getattr(lm, "load_warnings", []),  # pre-load guardrail (#76)
            "speed_tier": (getattr(lm, "load_assess", {}) or {}).get("speed_tier", ""),
            # --- #model-detail (click-to-expand modal): arch/tags + lifetime stats ---
            "arch": getattr(lm.spec, "arch", ""),
            "is_moe": _is_moe,
            "is_tts": bool(getattr(lm, "is_tts", False)),
            "is_t2a": bool(getattr(lm, "is_t2a", False)),
            "media": _media,   # #media-detail: None for LLMs; dict for tts/t2i/t2a
            "is_embedding": bool(getattr(lm.spec, "is_embedding", False)),
            "load_seconds": round(getattr(lm, "load_seconds", 0.0), 1),
            "req_total": getattr(lm, "req_total", 0),
            "tok_in_total": getattr(lm, "tok_in_total", 0),
            "tok_out_total": getattr(lm, "tok_out_total", 0),
            "max_tok_s": round(getattr(lm, "max_tok_s", 0.0), 2),
            "stages": [s.to_dict() for s in lm.plan.stages],
        }
    loaded = _loaded_dict(primary) if primary else None         # active model (dashboard panel)
    loaded_models = [_loaded_dict(m) for m in resident]         # ALL resident (multi-model)
    # --- Compute load (#82): how busy the fleet's processors are vs capacity ("out of what
    # is possible"). CPU load is capacity-weighted by logical cores so a busy 32-core box
    # counts more than a busy 4-core box; GPU load averages each enabled GPU's utilization.
    # The combined headline treats every CPU core and every GPU as one comparable compute unit.
    cpu_nodes = [n for n in nodes if n.ram_enabled]
    gpu_nodes = [n for n in nodes if n.vram_total_gb > 0 and n.vram_enabled]
    cpu_cores = sum(max(1, n.cores) for n in cpu_nodes)
    cpu_busy  = sum((n.cpu_percent / 100.0) * max(1, n.cores) for n in cpu_nodes)  # busy-core-equiv
    cpu_load_pct = (100.0 * cpu_busy / cpu_cores) if cpu_cores else 0.0
    gpu_load_pct = (sum(n.gpu_util for n in gpu_nodes) / len(gpu_nodes)) if gpu_nodes else 0.0
    gpu_busy  = sum(n.gpu_util / 100.0 for n in gpu_nodes)                          # busy-GPU-equiv
    units_total = cpu_cores + len(gpu_nodes)
    units_busy  = cpu_busy + gpu_busy
    overall_pct = (100.0 * units_busy / units_total) if units_total else 0.0
    # Disk picture. With chunk serving, workers hold weights in RAM (no model on
    # worker disk), so the model-size ceiling is the CONTROLLER's free disk (to
    # hold the full model) AND the RAM pool (to run it) — not the smallest worker.
    # Measure the drive that actually HOLDS the weights (models/ under the program dir),
    # NOT the OS/home drive — they're often different disks (e.g. weights on a big USB/data
    # drive while the OS is on C:). Using ~ reported the wrong drive's free space, so the
    # disk ceiling + fits_disk were computed against a disk the model never touches.
    try:
        _disk_path = MODELS_DIR if os.path.isdir(MODELS_DIR) else _PROJECT_DIR
        ctrl_free_gb = shutil.disk_usage(_disk_path).free / GB
    except Exception:
        ctrl_free_gb = 0.0
    # STABLE capacity (total usable RAM/VRAM, tier-aware) — NOT live free RAM — so the
    # "Max model" card shows the fleet's ceiling and doesn't flicker as free RAM jitters
    # across nodes (esp. shared boxes like BEAST). Actual loads still plan on live free.
    mems = [NodeMem(n.node_id, n.hostname,
                    int(((n.usable_mem_gb if n.ram_enabled else 0.0) + n.eff_vram_gb) * GB),
                    int(n.eff_vram_gb * GB)) for n in nodes]
    servable = []
    for name, (tgt, _d) in MODELS.items():
        spec = resolve_spec(tgt)   # built-in or config-derived (custom, once downloaded)
        if not spec:
            continue               # custom model not yet downloaded -> no estimate yet
        # Downloaded models: size + fit use the REAL measured weights (MoE-correct), and
        # they already occupy disk so they always "fit disk". Undownloaded: formula + the
        # free-disk check (can we pull it).
        dl = model_ready(tgt)
        d = _local_model_dir(tgt) if dl else None
        plan_spec = spec_with_measurements(spec, d) if d else spec
        size_gb = plan_spec.total_weight_bytes / GB
        # Per-quant WEIGHT footprint estimate (#49) so the UI can show what each load option
        # costs and which fit. Cheap (for_quant just rescales weight bytes); the fit hint is a
        # weight-only check vs the planner budget (KV is reserved on top, shown separately).
        # 'none' loads the NATIVE dtype, so its size IS the measured on-disk size (fp32 stays fp32).
        # for_quant's int8/int4 scaling assumes a 2-byte (bf16) base, so for an fp32/fp8 checkpoint
        # normalize the measured bytes to a bf16-equivalent first, else int8/int4 are off by the dtype
        # factor (an fp32 model would show int8/int4 ~2x too big).
        import dataclasses as _dc
        _srcb = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M3": 1}.get(plan_spec.src_dtype, 2)
        _f = 2.0 / _srcb
        _sc = lambda v: (int(v * _f) if v is not None else None)
        qspec = (_dc.replace(plan_spec, meas_layer_w=_sc(plan_spec.meas_layer_w),
                             meas_embed=_sc(plan_spec.meas_embed), meas_head=_sc(plan_spec.meas_head),
                             meas_norm=_sc(plan_spec.meas_norm))
                 if (_f != 1.0 and plan_spec.meas_layer_w is not None) else plan_spec)
        quant_gb = {"none": round(plan_spec.total_weight_bytes / GB, 2),          # native dtype size
                    "int8": round(qspec.for_quant("int8").total_weight_bytes / GB, 2),
                    "int4": round(qspec.for_quant("int4").total_weight_bytes / GB, 2),
                    "int2": round(qspec.for_quant("int2").total_weight_bytes / GB, 2)}   # #int2
        quant_fits = {q: (g <= round(pool_usable, 2)) for q, g in quant_gb.items()}
        fits_ram = bool(nodes) and plan_pipeline(plan_spec, mems, DEFAULT_CTX).ok
        fits_disk = dl or ctrl_free_gb >= size_gb
        servable.append({"name": _ollama_name(name), "internal_name": name,
                         "size_gb": round(size_gb, 2),
                         "default_ctx": plan_spec.max_ctx,   # native/training context (ctx=0 loads this)
                         "src_dtype": plan_spec.src_dtype,    # on-disk weight dtype (F32/BF16/...) for the UI
                         "quant_gb": quant_gb, "quant_fits": quant_fits,
                         "fits_ram": fits_ram, "fits_disk": fits_disk,
                         "runnable": fits_ram and fits_disk})
    runnable = [m for m in servable if m["runnable"]]
    largest = max(runnable, key=lambda m: m["size_gb"], default=None)
    min_worker_disk = min((n.free_disk_gb for n in nodes), default=0.0)
    # Controller's own wire (server-measured): out = bytes it pushed to nodes
    # (= sum of node ↓), in = bytes it pulled from nodes (= sum of node ↑).
    metrics = metric_rates()
    metrics["ctrl_out_bps"] = round(sum(n.net_in_bps for n in nodes))
    metrics["ctrl_in_bps"] = round(sum(n.net_out_bps for n in nodes))
    # Slots (1 running per model) + queue (waiters), with client IP + elapsed time.
    now = time.time()
    _inflight = sorted(INFLIGHT.values(), key=lambda r: r["enqueued"])
    slots = [{"id": r["id"], "ip": r["ip"], "model": r["model"],
              "running_s": round(now - r["started"], 1) if r["started"] else 0.0}
             for r in _inflight if r["state"] == "running"]
    queue = [{"id": r["id"], "ip": r["ip"], "model": r["model"],
              "waiting_s": round(now - r["enqueued"], 1)}
             for r in _inflight if r["state"] == "queued"]
    # #connections: per-client rows for the dashboard's Connections panel — live accounting
    # (bytes from the ASGI counter, tokens from the serving paths) JOINED with this client's
    # in-flight requests (which model it is using/loading right now) and any model load it
    # explicitly requested (loading-card requested_by).
    clients = []
    for r in sorted(CLIENTS.values(), key=lambda x: -x["last_seen"]):
        acts = [{"id": q["id"], "model": q["model"], "state": q["state"],
                 "s": round(now - (q["started"] or q["enqueued"]), 1)}
                for q in _inflight if q["ip"] == r["ip"]]
        lds = [c.get("display_model") or c.get("model") for c in engine.loadings.values()
               if c.get("requested_by") == r["ip"]]
        clients.append({"ip": r["ip"], "api": bool(r["api"]), "reqs": r["reqs"],
                        "bytes_in": r["bytes_in"], "bytes_out": r["bytes_out"],
                        "tok_in": r["tok_in"], "tok_out": r["tok_out"],
                        "last_model": r["last_model"],
                        "connected_s": round(now - r["first_seen"], 1),
                        "idle_s": round(now - r["last_seen"], 1),
                        "active": acts, "loading": lds})
    # #model-detail: build the per-model registry cards, then fold each RESIDENT model's RUNTIME
    # fields (ctx / quant / VRAM / RAM / KV / tok-s / placement stages / lifetime totals) into its
    # card. _model_entry alone knows only name/size/cached/loaded; the runtime lives in
    # loaded_models — without this merge the dashboard's loaded-model row + detail modal show
    # blank ctx/quant/VRAM/RAM and no placement (the reported bug).
    model_cards = [_model_entry(name, tgt, draft) for name, (tgt, draft) in MODELS.items()]
    _lm_by_key: dict = {}
    for _ld in loaded_models:
        for _k in (_ld.get("friendly"), _ld.get("display_name")):
            if _k:
                _lm_by_key[_k] = _ld
    _RUNTIME_KEYS = ("ctx", "quant", "kv_quant", "kv_offload", "kv_slots", "slots_active",
                     "def_temperature", "def_min_p",
                     "sampling_defaults", "vram_used_gb", "ram_used_gb", "cpu_frac",
                     "kv_reserved_gb", "kv_used_gb", "tok_s", "ema_tok_s", "max_tok_s",
                     "last_tok_s", "kv_pos", "fwd_prog_age_s", "active", "queued",
                     "is_embedding", "replica_idx",
                     "tp_size", "is_tp", "upgrade", "num_layers", "params", "stages", "plan_basis",
                     "speed_tier", "loaded_at_ts", "last_used_ts", "load_seconds",
                     "req_total", "tok_in_total", "tok_out_total", "arch", "is_moe",
                     "is_tts", "is_t2a", "media",   # #media-detail: media-model info block
                     "persist", "no_unload")
    for _e in model_cards:
        if _e.get("loaded"):
            _ld = _lm_by_key.get(_e.get("internal_name")) or _lm_by_key.get(_e.get("name"))
            if _ld:
                for _k in _RUNTIME_KEYS:
                    if _k in _ld and _e.get(_k) is None:
                        _e[_k] = _ld[_k]
    # #unified-fleet: fold in what our PEER controllers hold, so either controller renders the whole
    # fleet. Peer rows are stamped federated/owner and are strictly additive — a model or node we
    # drive ourselves is never replaced by a peer's view of it (peers.federated_* dedupe against us).
    # Both lists are also published raw so non-dashboard consumers can tell the two apart.
    _peer_nodes, _peer_models, _peer_tot = [], [], {}
    try:
        import peers as _peers
        _peer_nodes = _peers.federated_nodes()
        _peer_models = _peers.federated_models()
        _peer_tot = _peers.federated_totals()
    except Exception as _exc:   # noqa: BLE001 — federation is additive; /status must never 500 on it
        print(f"[status] peer view unavailable ({_exc!r})", flush=True)
    for _pm in _peer_models:
        # A peer's resident model presents as a LOADED card (it IS loaded — on the other
        # controller's nodes). When peers.py had a fresh /status from the owner, _pm IS that
        # controller's own card and is passed straight through — measured VRAM/RAM, KV reservation,
        # tok/s sparkline and all. Hand-rebuilding a card from the gossip summary was what left
        # these rows without graphs or memory figures.
        if _pm.get("name"):
            model_cards.append(_pm)
            continue
        # Fallback: only the /peer_info summary was available (peer's /status unreachable or
        # stale). Synthesise the thinner card so the model is at least listed and addressable.
        _nm = _pm.get("display_name") or _pm.get("friendly") or ""
        _card = {"name": _nm, "internal_name": _pm.get("friendly") or _nm,
                 "target": _pm.get("target") or "", "draft": "", "ready": False,
                 "status": "peer", "loaded": True, "federated": True,
                 "owner": _pm.get("owner"), "owner_url": _pm.get("owner_url"),
                 "aliases": _pm.get("aliases") or [], "capabilities": []}
        for _k in ("quant", "kv_quant", "ctx", "size_gb", "active", "queued", "num_layers",
                   "params", "arch", "is_moe", "is_embedding", "is_tts", "is_t2a",
                   "vram_used_gb", "tok_s", "ema_tok_s", "last_tok_s", "max_tok_s",
                   "loaded_at_ts", "last_used_ts"):
            if _pm.get(_k) is not None:
                _card[_k] = _pm[_k]
        if _pm.get("stages"):
            _card["stages"] = [{"hostname": h} for h in _pm["stages"]]
        model_cards.append(_card)
    return {
        "controller": {
            "hostname": platform.node(), "os": f"{platform.system()} {platform.release()}",
            "version": VERSION, "uptime_s": round(time.time() - START_TIME, 1),
            "code_date": CODE_DATE,   # newest self-update-set file mtime at process start
            "wire": ("wire" in sys.modules),   # True once wire.py is imported (not the fallback)
            "dash": ("dashboard_html" in sys.modules),   # True once dashboard_html.py is imported
            "http_port": ARGS.http_port, "control_port": ARGS.control_port,
            "data_port": ARGS.data_port, "os_reserve_gb": ARGS.os_reserve_gb,
            "free_disk_gb": round(ctrl_free_gb, 2),
            "hf_auth": (f"...{HF_TOKEN[-4:]}" if HF_TOKEN else False),
            "max_loaded": ENGINE_CONFIG.get("max_loaded", MAX_LOADED_MODELS),
            "auto_unload": ENGINE_CONFIG.get("auto_unload", True),
            "auto_load": ENGINE_CONFIG.get("auto_load", True),
            "autoload_quant": ENGINE_CONFIG.get("autoload_quant", "int4"),
            "autoload_ctx": ENGINE_CONFIG.get("autoload_ctx", DEFAULT_CTX),
            "autoload_mode": ENGINE_CONFIG.get("autoload_mode", "auto"),
            "vram_weights_first": ENGINE_CONFIG.get("vram_weights_first", True),
            "gen_stall_s": ENGINE_CONFIG.get("gen_stall_s", GEN_STALL_S),
            "gen_stall_decode_s": ENGINE_CONFIG.get("gen_stall_decode_s", GEN_STALL_DECODE_S),
            "queue_depth": ENGINE_CONFIG.get("queue_depth", DEFAULT_QUEUE_DEPTH),
            # #idle-unload: minutes with no requests before a model is unloaded (0 = keep forever)
            "idle_unload_m": ENGINE_CONFIG.get("idle_unload_m", 0.0),
            "juggler": ENGINE_CONFIG.get("juggler", False),                       # #juggler
            "master": ENGINE_CONFIG.get("master", False),                         # #master: designated fleet owner
            "autostart_delay_s": ENGINE_CONFIG.get("autostart_delay_s", 60.0),   # #autostart-delay
            "wedge_reload_n": ENGINE_CONFIG.get("wedge_reload_n", 3),             # #wedge-quarantine
            # #persist / #no-unload: friendly keys pinned for autoload-on-restart and never-auto-unload
            # (the detail modal reflects these for models that aren't currently loaded, too).
            "persist_models": sorted(ENGINE_CONFIG.get("persist_models") or {}),
            "no_unload_models": sorted(ENGINE_CONFIG.get("no_unload_models") or {}),
        },
        "pool": {"nodes": len(nodes), "total_gb": round(pool_total, 2),
                 "used_gb": round(pool_used, 2), "free_gb": round(pool_free, 2),  # LIVE physical
                 "engine_gb": round(pool_engine, 2),   # our pythons + shards (RED on the bar)
                 "os_gb": round(pool_os, 2),           # OS/other (BLUE on the bar)
                 "ctrlr_gb": round(ctrl_rss_gb, 2),    # controller process RSS alone
                 "usable_gb": round(pool_usable, 2),   # planner budget (live free, for fit checks)
                 "ram_gb": round(pool_ram, 2), "vram_gb": round(pool_vram, 2),
                 # PHYSICAL totals by form (pool-bar denominator; pairs with *_free below)
                 "ram_total_gb": round(pool_ram_total, 2), "vram_total_gb": round(pool_vram_total, 2),
                 # LIVE physical free split by form (what's available as RAM vs VRAM)
                 "ram_free_gb": round(pool_ram_free, 2), "vram_free_gb": round(pool_vram_free, 2)},
        "compute": {"overall_pct": round(overall_pct, 1),
                    "cpu_pct": round(cpu_load_pct, 1),
                    "cpu_busy_cores": round(cpu_busy, 1), "cpu_cores": int(cpu_cores),
                    "cpu_nodes": len(cpu_nodes),
                    "gpu_pct": round(gpu_load_pct, 1), "gpu_busy": round(gpu_busy, 2),
                    "gpus": len(gpu_nodes),
                    "units_busy": round(units_busy, 1), "units_total": int(units_total)},
        "metrics": metrics,
        "disk": {"controller_free_gb": round(ctrl_free_gb, 2),
                 "min_worker_free_gb": round(min_worker_disk, 2),
                 "largest_model": largest, "models": servable,
                 "note": "chunk serving: workers hold weights in RAM, so model size "
                         "is bounded by controller disk + RAM pool, not worker disks"},
        "cluster": {"state": "loaded" if resident else ("dirty" if registry.dirty else "idle"),
                    "model": primary.friendly if primary else None,
                    "display_model": _ollama_name(primary.friendly) if primary else None,
                    "loaded": loaded,
                    "loaded_models": loaded_models,
                    # parallel loads/compiles -> LISTS of cards (was a single 'loading' dict). 'loading'
                    # stays as the first in-flight load card for any old consumer; 'loadings' (all loads)
                    # + 'compiling' (all shard-compiles) are the full lists the dashboard renders.
                    "loading": next((_loading_view(c) for c in engine.loadings.values()), None),
                    "loadings": [_loading_view(c) for c in engine.loadings.values()],
                    "compiling": [_loading_view(c) for c in engine.compiling.values()],
                    "reconfiguring": getattr(engine, "reconfiguring", None),   # #88 managed reload
                    "slots": slots, "queue": queue,
                    "queue_depth": ENGINE_CONFIG.get("queue_depth", DEFAULT_QUEUE_DEPTH)},
        "clients": clients,      # #connections: per-client accounting + activity (dashboard panel)
        "models": model_cards,   # registry cards enriched with resident runtime (#model-detail)
        "nodes": [n.to_dict() for n in nodes],
        # #unified-fleet: our PEERS' nodes/models, stamped with their owner. Kept in separate keys
        # (not merged into "nodes") because everything downstream of "nodes" — the planner, the pool
        # arithmetic, per-node actions — means "nodes THIS controller drives", and quietly widening
        # that would let a peer's hardware into our capacity maths. The dashboard concatenates.
        "peer_nodes": _peer_nodes,
        "peer_models": _peer_models,
        # WHOLE-FLEET rollup for the dashboard's summary tiles: ours + every healthy peer's. On a
        # controller that owns no hardware, "pool" and "compute" are legitimately all zeros — they
        # describe what THIS controller drives — so the tiles read 0 nodes / 0 GB / 0 tok/s while
        # the page below them showed the whole fleet. This is the number a person means by "the
        # fleet"; `via_peers` keeps it attributable rather than one blended figure. Consumers that
        # want strictly-ours keep reading pool/compute/metrics, which are unchanged.
        "fleet": {
            "nodes": len(nodes) + int(_peer_tot.get("nodes") or 0),
            "gpus": len(gpu_nodes) + int(_peer_tot.get("gpus") or 0),
            "ram_total_gb": round(pool_ram_total + float(_peer_tot.get("ram_total_gb") or 0), 2),
            "ram_free_gb": round(pool_ram_free + float(_peer_tot.get("ram_free_gb") or 0), 2),
            "vram_total_gb": round(pool_vram_total + float(_peer_tot.get("vram_total_gb") or 0), 2),
            "vram_free_gb": round(pool_vram_free + float(_peer_tot.get("vram_free_gb") or 0), 2),
            "tokens_per_s": round(float(metrics.get("tokens_per_s") or 0)
                                  + float(_peer_tot.get("tokens_per_s") or 0), 2),
            "units_busy": round(units_busy + float(_peer_tot.get("units_busy") or 0), 2),
            "units_total": int(units_total) + int(_peer_tot.get("units_total") or 0),
            "via_peers": {"nodes": int(_peer_tot.get("nodes") or 0),
                          "gpus": int(_peer_tot.get("gpus") or 0),
                          "controllers": _peer_tot.get("controllers") or []},
        },
        "activity": list(ACTIVITY),   # newest-first controller activity (dashboard panel)
        "unloads": list(UNLOADS),     # newest-first "why a model left" events (dashboard panel)
        "errors": list(ERRORS),       # #error-log: newest-first HTTP 4xx/5xx responses (Logs UI)
    }


_CAPS_CACHE: dict = {}


def _dir_supports_tools(d) -> bool:
    """#tools: True when the model's chat template NATIVELY renders tool definitions — the honest
    per-model 'tools' capability signal (Qwen2.5/3, Llama-3.1+, Mistral, Hermes, gpt-oss, …). Reads
    the template from tokenizer_config.json['chat_template'] or a standalone chat_template.jinja
    (config-only; no model load). Models WITHOUT native support still work via the serve layer's text
    tool-instruction fallback, but the badge reflects genuine template support (what Ollama reports)."""
    import os
    import json as _json
    try:
        t = ""
        tc = os.path.join(d, "tokenizer_config.json")
        if os.path.exists(tc):
            with open(tc, encoding="utf-8") as fh:
                ct = _json.load(fh).get("chat_template")
            if isinstance(ct, str):
                t = ct
            elif isinstance(ct, list):   # some tokenizers ship a list of {name, template} entries
                t = " ".join(x.get("template", "") for x in ct if isinstance(x, dict))
        if not t:
            jj = os.path.join(d, "chat_template.jinja")   # Mistral3/Devstral ship it standalone
            if os.path.exists(jj):
                with open(jj, encoding="utf-8") as fh:
                    t = fh.read()
        if not t:
            cj = os.path.join(d, "chat_template.json")    # Qwen2.5-VL (processor-level template)
            if os.path.exists(cj):
                with open(cj, encoding="utf-8") as fh:
                    ct = _json.load(fh).get("chat_template")
                if isinstance(ct, str):
                    t = ct
        return bool(t) and (("tool_call" in t) or ("tools" in t))
    except Exception:
        return False


def _model_caps(tgt: str, spec=None) -> list:
    """Modality capabilities for the dashboard badge line, inferred from the model's local
    config.json (cached per target — configs don't change). Returns a subset of: embedding,
    image, video, stt (audio-in / speech understanding), tts (speech-out, Omni-style). A plain
    text LLM -> []. Reads the raw config DICT (no AutoConfig / trust_remote_code) so exotic
    arches don't break or prompt; absent (not-yet-downloaded) models -> [] until present."""
    if tgt in _CAPS_CACHE:
        return _CAPS_CACHE[tgt]
    caps: list = []
    d = None
    try:
        # Read MODALITIES from config first (so a multimodal model whose spec is mis-flagged
        # is_embedding — e.g. Qwen2.5-Omni — still shows image/stt/tts), then fall back to the
        # embedding badge only when the model has NO modalities (a pure encoder like nomic).
        import os
        import json as _json
        d = _local_model_dir(tgt)
        # #t2i: a diffusers image-generation checkpoint (model_index.json) is its own kind —
        # the dashboard badges it and hides the (unsupported) LLM Load action.
        if d and _is_diffusers_dir(d):
            _CAPS_CACHE[tgt] = ["t2i"]
            return ["t2i"]
        # #tts: a Kokoro speech checkpoint (kokoro-v1_0.pth + voices/) — badge it and, like
        # t2i, hide the (unsupported) LLM Load action in favor of the speech load/generate path.
        if d and _is_kokoro_dir(d):
            _CAPS_CACHE[tgt] = ["tts"]
            return ["tts"]
        # #t2a: an ACE-Step music checkpoint (ace_step_transformer/ component layout) — badge it
        # and offer the music load/generate path, like t2i/tts (no LLM Load action).
        if d and os.path.isdir(os.path.join(d, "ace_step_transformer")):
            _CAPS_CACHE[tgt] = ["t2a"]
            return ["t2a"]
        cfgd = None
        if d:
            p = os.path.join(d, "config.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    cfgd = _json.load(fh)
        if cfgd:
            base = cfgd.get("thinker_config") or cfgd   # Omni nests vision/audio under thinker
            _b = base if isinstance(base, dict) else {}

            def _has(*ks):
                return (any(cfgd.get(k) is not None for k in ks)
                        or any(_b.get(k) is not None for k in ks))
            if _b.get("vision_config") or cfgd.get("vision_config") \
                    or _has("image_token_id", "image_token_index"):
                caps.append("image")
            if _has("video_token_id", "video_token_index"):
                caps.append("video")
            if _b.get("audio_config") or cfgd.get("audio_config") \
                    or _has("audio_token_id", "audio_token_index"):
                caps.append("stt")
            mt = (cfgd.get("model_type") or "").lower()
            if "omni" in mt or cfgd.get("talker_config") or _b.get("talker_config"):
                caps.append("tts")
            # #ocr: OCR-SPECIALIST checkpoints (GOT-OCR2 model_type 'got_ocr2', DeepSeek-OCR,
            # olmOCR — a qwen2-vl arch whose repo name carries it) get their own badge; a generic
            # VL model can read text too but stays 'image' only — the badge marks models BUILT
            # for document/text extraction. Config-only signals: model_type, architectures, or
            # the model dir name.
            _arches = " ".join(cfgd.get("architectures") or []).lower()
            if "ocr" in mt or "ocr" in _arches or "ocr" in os.path.basename(d or "").lower():
                caps.append("ocr")
        if not caps and spec is not None and getattr(spec, "is_embedding", False):
            caps = ["embedding"]
        # #tools: native tool-calling badge (from the chat template). A causal chat LM whose template
        # renders tool defs gets 'tools'; embeddings never do. Serve-layer still supports tools on
        # non-native models via a text-instruction fallback — the badge just marks first-class support.
        if d and "embedding" not in caps and _dir_supports_tools(d):
            caps.append("tools")
    except Exception:
        pass
    if d:
        # Only cache once the model dir exists — caching [] for a not-yet-downloaded model
        # would freeze its badges empty until a controller restart (#t2i).
        _CAPS_CACHE[tgt] = caps
    return caps


def _model_entry(name: str, tgt: str, draft: str) -> dict:
    """Per-model status for the dashboard: ready (weights on controller),
    downloading (pull in flight), or absent. size from the spec when known."""
    spec = resolve_spec(tgt)   # built-in or config-derived (custom models)
    ready = model_ready(tgt)
    if ready:
        status = "ready"
    elif name in DOWNLOADING:                      # live pull; reflect a pending pause/stop
        ctl = DOWNLOAD_CONTROL.get(name)
        status = "pausing" if ctl == "pause" else "stopping" if ctl == "stop" else "downloading"
    elif name in DOWNLOAD_STATE:                    # halted by the user (paused/stopped), cache kept
        status = DOWNLOAD_STATE[name]
    else:
        status = "absent"
    loaded = name in engine.models
    # Size: spec-aware display bytes when we can model the arch; otherwise (e.g. the Omni
    # Thinker-Talker, which resolve_spec can't represent) fall back to the RAW on-disk
    # safetensors total so a downloaded-but-unrunnable model still shows a real size.
    size_bytes = None
    if spec:
        size_bytes = _display_weight_bytes(tgt, spec)
    else:
        d = _local_model_dir(tgt)
        if d:
            m = measure_model_weights(d)
            if m and m.get("total"):
                size_bytes = int(m["total"])
            if size_bytes is None:
                # diffusers layout (#t2i): weights live in component subfolders the flat
                # measurer can't see — fall back to the recursive on-disk safetensors sum
                t = _tree_weight_bytes(d)
                if t:
                    size_bytes = t
    # Display the Ollama 'family:size' name ('qwen3:4b'); the dashboard sends it back as the
    # op key and resolve_model_name() maps it to this dash-form key. internal_name is the raw
    # registry key, so existing tooling that keys off the dash form keeps working.
    entry = {"name": _ollama_name(name), "internal_name": name,
             "target": tgt, "draft": draft, "ready": ready,
             "status": status, "loaded": loaded,
             "size_gb": round(size_bytes / GB, 2) if size_bytes else None}
    _al = _aliases_for(name)
    if _al:                          # show alias(es) under the primary name in the dashboard
        entry["aliases"] = _al
    if loaded:                                   # live request queue depth (Inc 4)
        lm = engine.models[name]
        entry["active"] = lm.active              # currently generating (0/1; 0..C with #kv-slots)
        entry["queued"] = lm.queued             # requests waiting on this model's slot pool
        if getattr(lm, "is_t2i", False):         # #t2i-serve: image model — live render progress
            entry["t2i"] = True
            _pr = getattr(engine, "_t2i_progress", {}).get(getattr(lm, "t2i_req", None))
            if _pr:
                entry["t2i_step"], entry["t2i_total"] = _pr[0], _pr[1]
        if getattr(lm, "is_t2a", False):         # #t2a-serve: music model — live render progress
            entry["t2a"] = True
            _pr = getattr(engine, "_t2a_progress", {}).get(getattr(lm, "t2a_req", None))
            if _pr:
                entry["t2a_step"], entry["t2a_total"] = _pr[0], _pr[1]
    if status in ("downloading", "pausing", "stopping", "paused", "stopped"):
        pr = DOWNLOAD_PROGRESS.get(name) or {}     # frozen at the halt point for paused/stopped
        dl, tot = pr.get("downloaded", 0), pr.get("total", 0)
        entry["dl_done_gb"] = round(dl / GB, 2)
        entry["dl_total_gb"] = round(tot / GB, 2) if tot else None
        entry["dl_pct"] = round(min(100.0, 100 * dl / tot), 1) if tot else None  # never >100%
        # Speed + ETA — only while bytes are actually moving (the poller freezes pr on pause/stop,
        # so a stale rate would lie). rate is bytes/sec; expose MiB/s + seconds-remaining.
        if status in ("downloading", "pausing", "stopping"):
            rate = pr.get("rate") or 0
            if rate > 0:
                entry["dl_rate_mbps"] = round(rate / (1024 * 1024), 1)
                eta = pr.get("eta_s")
                if eta is not None and tot:
                    entry["dl_eta_s"] = int(eta)
    elif status == "absent" and name in DOWNLOAD_ERROR:
        entry["dl_error"] = DOWNLOAD_ERROR[name]   # last pull failure (e.g. gated repo)
    if ready:                                       # #shard-cache: which quants are pre-compiled
        d3 = _local_model_dir(tgt)
        cs = shard_cache_status(d3) if d3 else {}
        if cs:
            entry["cached"] = cs   # {quant: {ok, size_gb, files, ...}}
    caps = _model_caps(tgt, spec)
    if caps:
        entry["capabilities"] = caps   # modality badges under the name (image/video/stt/tts/embedding)
    return entry
