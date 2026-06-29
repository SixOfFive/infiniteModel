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
        ram_weights = max(0, lm.spec.total_weight_bytes - vram_used)
        kv_reserved = lm.spec.kv_bytes_per_layer(lm.ctx) * lm.spec.num_layers
        kv_used = lm.spec.kv_bytes_per_layer(lm.kv_pos) * lm.spec.num_layers
        _arch = (getattr(lm.spec, "arch", "") or "").lower()
        # best-effort MoE flag for the detail modal: fused/per-expert MoE arches all contain one of
        # these tokens ('moe' covers olmoe/qwen3*_moe; mixtral/minimax/deepseek_v2,v3 named directly).
        _is_moe = any(k in _arch for k in ("moe", "mixtral", "minimax", "deepseek_v"))
        return {
            "friendly": lm.friendly, "display_name": _ollama_name(lm.friendly),  # 'qwen3:4b'
            "aliases": _aliases_for(lm.base or lm.friendly),  # display-form alias(es) -> shown under the name
            "target": lm.target_id, "ctx": lm.ctx,
            "base": lm.base or lm.friendly, "replica_idx": lm.replica_idx,  # data-parallel (#39)
            "active": lm.active, "queued": lm.queued,   # per-replica live load (#39 routing)
            "kv_pos": lm.kv_pos,   # tokens in the current/last generation's KV context
            # LIVE decode tok/s: the most-recent-gen rate WHILE generating, else 0 when idle —
            # last_tok_s lingers at its last value forever otherwise (card looked "busy" when idle).
            # The historical rate stays visible as ema_tok_s ("avg"). active==0 => idle => 0.
            "tok_s": round(lm.last_tok_s if lm.active > 0 else 0.0, 2),   # decode tok/s, live (#46)
            "ema_tok_s": round(lm.ema_tok_s, 2),     # smoothed decode tok/s across gens, historical (#46)
            "quant": lm.quant,     # the quant this model was loaded with (none/int8)
            "tp_size": getattr(lm, "tp_size", 1),            # #88: TP width (1 = pipeline)
            "is_tp": getattr(lm, "tp_size", 1) > 1,          # #88: card shows TP vs pipeline + reconfigure
            "num_layers": lm.spec.num_layers, "params": _human_params(lm.spec),
            "size_gb": round(lm.spec.total_weight_bytes / GB, 2),
            "vram_used_gb": round(vram_used / GB, 2),
            "ram_used_gb": round(ram_weights / GB, 2),       # weights resident in RAM (not KV)
            # #cpu-bound-visibility: ACTUAL fraction of WEIGHTS on CPU (from worker-reported gpu_bytes,
            # not the pre-load estimate). A high value = the model is CPU-bound and will decode SLOWLY
            # (CPU layers are ~50-100x a GPU layer) — the dashboard badges it so "slow" isn't read as
            # "hung/wedged". This is the real cause of a multi-model fleet's later loads crawling.
            "cpu_frac": round(ram_weights / lm.spec.total_weight_bytes, 3) if lm.spec.total_weight_bytes else 0.0,
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
                    "int4": round(qspec.for_quant("int4").total_weight_bytes / GB, 2)}
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
    return {
        "controller": {
            "hostname": platform.node(), "os": f"{platform.system()} {platform.release()}",
            "version": VERSION, "uptime_s": round(time.time() - START_TIME, 1),
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
        },
        "pool": {"nodes": len(nodes), "total_gb": round(pool_total, 2),
                 "used_gb": round(pool_used, 2), "free_gb": round(pool_free, 2),  # LIVE physical
                 "engine_gb": round(pool_engine, 2),   # our pythons + shards (RED on the bar)
                 "os_gb": round(pool_os, 2),           # OS/other (BLUE on the bar)
                 "ctrlr_gb": round(ctrl_rss_gb, 2),    # controller process RSS alone
                 "usable_gb": round(pool_usable, 2),   # planner budget (live free, for fit checks)
                 "ram_gb": round(pool_ram, 2), "vram_gb": round(pool_vram, 2),
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
        "models": [_model_entry(name, tgt, draft)
                   for name, (tgt, draft) in MODELS.items()],
        "nodes": [n.to_dict() for n in nodes],
        "activity": list(ACTIVITY),   # newest-first controller activity (dashboard panel)
        "unloads": list(UNLOADS),     # newest-first "why a model left" events (dashboard panel)
    }


_CAPS_CACHE: dict = {}


def _model_caps(tgt: str, spec=None) -> list:
    """Modality capabilities for the dashboard badge line, inferred from the model's local
    config.json (cached per target — configs don't change). Returns a subset of: embedding,
    image, video, stt (audio-in / speech understanding), tts (speech-out, Omni-style). A plain
    text LLM -> []. Reads the raw config DICT (no AutoConfig / trust_remote_code) so exotic
    arches don't break or prompt; absent (not-yet-downloaded) models -> [] until present."""
    if tgt in _CAPS_CACHE:
        return _CAPS_CACHE[tgt]
    caps: list = []
    try:
        # Read MODALITIES from config first (so a multimodal model whose spec is mis-flagged
        # is_embedding — e.g. Qwen2.5-Omni — still shows image/stt/tts), then fall back to the
        # embedding badge only when the model has NO modalities (a pure encoder like nomic).
        import os
        import json as _json
        d = _local_model_dir(tgt)
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
        if not caps and spec is not None and getattr(spec, "is_embedding", False):
            caps = ["embedding"]
    except Exception:
        pass
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
        entry["active"] = lm.active              # currently generating (0/1)
        entry["queued"] = lm.queued             # requests waiting on this model's lock
    if status in ("downloading", "pausing", "stopping", "paused", "stopped"):
        pr = DOWNLOAD_PROGRESS.get(name) or {}     # frozen at the halt point for paused/stopped
        dl, tot = pr.get("downloaded", 0), pr.get("total", 0)
        entry["dl_done_gb"] = round(dl / GB, 2)
        entry["dl_total_gb"] = round(tot / GB, 2) if tot else None
        entry["dl_pct"] = round(min(100.0, 100 * dl / tot), 1) if tot else None  # never >100%
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
