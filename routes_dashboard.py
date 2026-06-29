"""routes_dashboard: routes relocated from server.py build_app (m4c153 code-split). Route bodies
are BYTE-IDENTICAL to the originals; their module globals (engine, registry, _serve,
build_status, JSONResponse …) are injected at startup by state.bind() — see state.py.
build_app() calls register(app) to attach them. Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def register(app):

    # ---- dashboard + introspection ----
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @app.get("/bandwidth", response_class=HTMLResponse)
    async def bandwidth_page() -> str:
        return BANDWIDTH_HTML

    @app.get("/bandwidthdata")       # full traffic picture: controller<->node + node<->node
    async def bandwidthdata() -> JSONResponse:
        """Combine the controller's own socket metering (authoritative for controller<->node)
        with each worker's per-peer counters (the ONLY source for node<->node hidden-state
        traffic the controller can't see). Cumulative bytes; the page derives rates. No
        double-counting: controller<->node comes from NODE_NET; node<->node from the sender's
        'out' counter (each directed hop reported once, by its sender)."""
        nodes = registry.alive_sorted()
        ip2host = {n.data_host: n.hostname for n in nodes if getattr(n, "data_host", None)}
        node_rows = []
        for n in nodes:
            c = NODE_NET.get(n.node_id, {})
            pb = (n.peer_bytes or {})
            # node<->node totals from this node's own counters (peer != controller)
            nn_in = sum(int(v.get("in", 0)) for p, v in pb.items() if p != "controller")
            nn_out = sum(int(v.get("out", 0)) for p, v in pb.items() if p != "controller")
            nn_in_p = sum(int(v.get("in_pkts", 0)) for p, v in pb.items() if p != "controller")
            nn_out_p = sum(int(v.get("out_pkts", 0)) for p, v in pb.items() if p != "controller")
            node_rows.append({
                "node_id": n.node_id, "hostname": n.hostname, "alive": n.alive,
                "ctrl_to_node": int(c.get("in", 0)), "node_to_ctrl": int(c.get("out", 0)),  # controller-measured
                "nn_in": nn_in, "nn_out": nn_out,                              # node<->node (worker)
                # "packets" = data-plane frames (each tensor send/recv is one frame)
                "ctrl_to_node_pkts": int(c.get("in_pkts", 0)), "node_to_ctrl_pkts": int(c.get("out_pkts", 0)),
                "nn_in_pkts": nn_in_p, "nn_out_pkts": nn_out_p,
                "net_in_bps": round(n.net_in_bps), "net_out_bps": round(n.net_out_bps)})
        edges = []   # directed node->node hops (sender's out), controller hops excluded
        for n in nodes:
            for peer, v in (n.peer_bytes or {}).items():
                if peer == "controller":
                    continue
                out = int(v.get("out", 0))
                if out > 0:
                    edges.append({"src": n.hostname, "dst": ip2host.get(peer, peer),
                                  "bytes": out, "pkts": int(v.get("out_pkts", 0))})
        return JSONResponse({"controller": _display_host(), "nodes": node_rows, "edges": edges})

    @app.get("/status")
    async def status(graphs: int = 0) -> JSONResponse:
        # graphs=1 attaches a server-rendered SVG sparkline per node (bandwidth + RAM)
        # so the dashboard can drop them straight into the DOM. Off by default to keep
        # /status lean for non-dashboard consumers (the Ollama-compat clients, scripts).
        st = build_status()
        if graphs:
            for nd in st.get("nodes", []):
                h = nd.get("hostname", "?")
                nd["spark_bw"] = _spark_svg(h, "bw")
                nd["spark_ram"] = _spark_svg(h, "ram")
                # GPU VRAM sparkline only for nodes that have a GPU
                if nd.get("vram_total_gb", 0) > 0:
                    nd["spark_vram"] = _spark_svg(h, "vram")
        return JSONResponse(st)

    @app.get("/graph/{kind}/{host}")
    async def graph(kind: str, host: str) -> Response:
        # Larger detail graph for a node, server-rendered (the mini sparkline's
        # click-target). kind in {bw, ram, vram}; anything else is a 404.
        if kind not in ("bw", "ram", "vram"):
            return Response(content="unknown graph kind", status_code=404,
                            media_type="text/plain")
        return Response(content=_detail_svg(host, kind), media_type="image/svg+xml")

    @app.get("/nethistory")
    async def nethistory(since: float = 0.0) -> JSONResponse:
        # Server-stored, disk-persisted per-node traffic graph. since>0 returns only
        # points newer than that ms timestamp (incremental, tiny payloads); since=0
        # returns the full bounded window (initial load / fresh tab).
        since_ms = int(since)
        hosts: dict[str, list] = {}
        for host, dq in NET_HISTORY.items():
            pts = list(dq) if since_ms <= 0 else [p for p in dq if p[0] > since_ms]
            if pts:
                hosts[host] = pts
        return JSONResponse({"sample_s": NET_HIST_SAMPLE_S, "cap": NET_HIST_MAX,
                             "now": int(time.time() * 1000), "hosts": hosts})

    @app.get("/plan")
    async def plan(model: str, ctx: int = 0, quant: str = "none", mode: str = "auto") -> JSONResponse:
        # #60 Preview: same inputs as /load (model, ctx, quant, mode) -> the placement + #76
        # assessment WITHOUT loading. tp modes are pipeline-planned here (TP frees the fleet and
        # plans differently at load); the dashboard tells the user TP preview is approximate.
        # Resolve the name first (like /load) so the Ollama 'family:size' form ('qwen3:4b') the
        # dashboard sends maps to the registry key/target before resolve_spec runs.
        try:
            friendly = resolve_model_name(model)
        except ValueError:
            return JSONResponse({"ok": False, "error": f"unknown model '{model}'"},
                                status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        spec = resolve_spec(target)
        if spec is None:
            return JSONResponse({"ok": False, "error": f"unknown model '{model}'"},
                                status_code=404)
        # Measure REAL safetensors bytes so MoE / any non-dense arch sizes correctly in the
        # PREVIEW too. The dense formula under-counts N experts (~4 GB est for the ~115 GB int4
        # MiniMax-M2, ~3.5 GB for the 66 GB Qwen3.6-35B-A3B), which made Preview claim a huge MoE
        # "fits on 3 GPUs" — wildly diverging from the live load (which DOES measure, line ~2856).
        # No-op if the model isn't downloaded yet. Cached by dir (_MEAS_CACHE).
        _pd = await asyncio.to_thread(_local_model_dir, target)
        if _pd:
            spec = await asyncio.to_thread(spec_with_measurements, spec, _pd)
        if ctx <= 0:   # default to the model's native training context (from spec)
            ctx = spec.max_ctx or DEFAULT_CTX
        spec = spec.for_quant(quant) if quant in ("int8", "int4") else spec   # size the real footprint
        cons, pv = LOAD_MODES.get(mode, LOAD_MODES["auto"])   # mirror /load's mode -> placement flags
        mems = []
        node_by_id = {}
        for n in registry.alive_sorted():
            fv = max(0.0, n.eff_vram_gb - PLAN_VRAM_FLOOR_GB)   # same VRAM floor as live load
            # #78: mirror the live load's controller-box RAM reserve so Preview's pool matches reality
            _ram = n.eff_ram_gb - (CONTROLLER_RAM_RESERVE_GB if n.data_host in _LOCAL_IPS else 0.0)
            mems.append(NodeMem(n.node_id, n.hostname,
                                int((max(0.0, _ram) + fv) * GB), int(fv * GB)))
            node_by_id[n.node_id] = n
        p = plan_pipeline(spec, mems, ctx_len=ctx, consolidate=cons, prefer_vram=pv,
                          spread=(mode == "spread"),
                          proportional=(mode == "proportional"),
                          gpu_spread=(mode == "all-gpu"))
        d = p.to_dict()
        if p.ok:   # #60/#76: surface the basis + pre-load assessment so a Preview matches the load
            d["basis"] = _describe_plan(p.stages, node_by_id, False, pv, quant,
                                        gpu_spread=(mode == "all-gpu"))
            d["assess"] = _assess_placement(spec, ctx, mems, p.stages)
            # #78 guardrail: a CONSOLIDATING mode (auto/single) can pile a heavy shard onto the
            # controller's co-located worker, which must ALSO serve the whole stream -> it OOM-drops
            # mid-load (the beast minimax crash). Flag it so the dashboard offers 'proportional'
            # (spreads across the fleet) in a confirm() BEFORE the load commits. Fires only when the
            # co-located stage's RAM leaves < 2x the controller reserve free on that box.
            if cons and mode != "proportional":
                # (1) controller-box RAM overload: a heavy shard on the co-located worker that ALSO
                # serves the whole stream -> OOM-drop risk (the beast minimax crash).
                for s in p.stages:
                    nd = node_by_id.get(s.node_id)
                    if nd is not None and nd.data_host in _LOCAL_IPS:
                        if s.est_gb > (nd.eff_ram_gb - 2 * CONTROLLER_RAM_RESERVE_GB):
                            d["overload"] = {"reason": "controller_ram", "node": nd.hostname,
                                             "mode": mode, "suggest": "proportional",
                                             "stage_gb": round(s.est_gb, 1),
                                             "node_ram_gb": round(nd.eff_ram_gb, 1)}
                        break
                # (2) #103: GPU oversubscribe -> CPU spill. auto/single consolidate onto the fewest
                # nodes, so a model bigger than that subset's free VRAM spills to CPU (slow) EVEN WHEN
                # other GPUs in the fleet sit idle. If proportional would put materially more weight on
                # GPU (fleet free-VRAM clearly exceeds the chosen subset's), suggest it BEFORE loading.
                if "overload" not in d and pv:
                    model_gb = spec.total_weight_bytes / GB
                    auto_gpu = sum(node_by_id[s.node_id].eff_vram_gb for s in p.stages
                                   if s.node_id in node_by_id and node_by_id[s.node_id].eff_vram_gb > 0)
                    fleet_gpu = sum(n.eff_vram_gb for n in registry.alive_sorted()
                                    if n.eff_vram_gb > 0)
                    on_cpu = model_gb - auto_gpu
                    if on_cpu > 2.0 and fleet_gpu > auto_gpu + 2.0:
                        d["overload"] = {"reason": "gpu_spill", "mode": mode, "suggest": "proportional",
                                         "model_gb": round(model_gb, 1),
                                         "auto_gpu_gb": round(auto_gpu, 1),
                                         "fleet_gpu_gb": round(fleet_gpu, 1),
                                         "on_cpu_gb": round(max(0.0, on_cpu), 1)}
        return JSONResponse(d)

    # ---- chunk serving (workers fetch only their slice; nothing on worker disk) ----
    @app.get("/modelmeta")
    async def modelmeta(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
            return JSONResponse(json.load(fh))

    @app.get("/history")   # #ctx-history: actual context IN/OUT for a loaded model (dashboard popup)
    async def history(model: str, dir: str = "both", n: int = 0) -> JSONResponse:
        # Returns the rolling per-request capture (newest first), decoding the stored token ids to
        # text LAZILY here (off the event loop) so the decode path never pays it. Empty when the model
        # isn't resident (history is cleared on unload). `dir`=in|out|both, `n`=limit to most-recent N.
        try:
            friendly = resolve_model_name(model)
        except Exception:
            friendly = model
        full = REQUEST_HISTORY.get(friendly) or []
        dq = list(full)[-n:] if (n and n > 0) else list(full)
        lm = engine.models.get(friendly)
        tok = getattr(lm, "tokenizer", None) if lm else None

        def _build():
            def _dec(ids):
                if not tok or not ids:
                    return ""
                try:
                    return tok.decode(ids, skip_special_tokens=False)
                except Exception:
                    return ""
            out = []
            for e in reversed(dq):   # newest first
                rec = {"ts": e.get("ts"), "tok_in": e.get("tok_in"), "tok_out": e.get("tok_out")}
                if dir in ("in", "both"):
                    rec["input"] = _dec(e.get("in_ids"))
                if dir in ("out", "both"):
                    rec["output"] = _dec(e.get("out_ids"))
                out.append(rec)
            return out

        entries = await asyncio.to_thread(_build)
        return JSONResponse({"model": friendly, "count": len(full), "entries": entries,
                             "tok_in_total": getattr(lm, "tok_in_total", 0) if lm else 0,
                             "tok_out_total": getattr(lm, "tok_out_total", 0) if lm else 0})

    @app.get("/logs")                # #logs: curl-able log — controller's own, or a worker's (relayed)
    async def get_logs(tail: int = 200, node: str = "") -> Response:
        """GET /logs[?tail=N][&node=<host|node_id>]. No node -> the CONTROLLER's stdout/stderr ring.
        node given -> that worker's log lines relayed on its heartbeats (so a worker box with no
        console/journal access is still debuggable). Plain text, newest last."""
        tail = max(1, min(int(tail or 200), NODE_LOGS_MAX))
        if node:
            nid = node if node in NODE_LOGS else next(
                (i for i, n in registry._nodes.items() if n.hostname == node), node)
            buf = NODE_LOGS.get(nid)
            if not buf:
                return Response(content=f"(no logs buffered for node {node!r} — workers relay logs "
                                "on heartbeat once they're on m4c31+)\n", media_type="text/plain")
            return Response(content="\n".join(buf[-tail:]) + "\n", media_type="text/plain")
        return Response(content="\n".join(tail_logs(tail)) + "\n", media_type="text/plain")

    @app.post("/config")             # dashboard: runtime engine config (persisted)
    async def set_config(max_loaded: Optional[int] = None,
                         auto_unload: Optional[bool] = None,
                         queue_depth: Optional[int] = None,
                         auto_tp: Optional[bool] = None,
                         auto_tp_ratio: Optional[float] = None,
                         auto_load: Optional[bool] = None,
                         autoload_quant: Optional[str] = None,
                         autoload_ctx: Optional[int] = None,
                         autoload_mode: Optional[str] = None,
                         vram_weights_first: Optional[bool] = None,
                         gen_stall_s: Optional[float] = None,
                         gen_stall_decode_s: Optional[float] = None,
                         persist: Optional[str] = None,
                         unpersist: Optional[str] = None) -> JSONResponse:
        if persist is not None:                          # #77: keep this model across restarts
            with contextlib.suppress(ValueError):
                fr = resolve_model_name(persist)
                _lm = engine.models.get(fr)
                _pm = dict(ENGINE_CONFIG.get("persist_models") or {})
                _pm[fr] = {"ctx": (_lm.ctx if _lm else 0), "quant": (_lm.quant if _lm else "none")}
                ENGINE_CONFIG["persist_models"] = _pm
                log_activity(f"persist: {fr} will auto-reload on startup "
                             f"(ctx={_pm[fr]['ctx']}, quant={_pm[fr]['quant']})")
        if unpersist is not None:
            with contextlib.suppress(ValueError):
                fr = resolve_model_name(unpersist)
                _pm = dict(ENGINE_CONFIG.get("persist_models") or {})
                if _pm.pop(fr, None) is not None:
                    ENGINE_CONFIG["persist_models"] = _pm
                    log_activity(f"persist: {fr} removed (no longer auto-reloaded on startup)")
        if max_loaded is not None:
            ENGINE_CONFIG["max_loaded"] = max(1, int(max_loaded))
        if auto_unload is not None:
            ENGINE_CONFIG["auto_unload"] = bool(auto_unload)
        if queue_depth is not None:
            ENGINE_CONFIG["queue_depth"] = max(0, int(queue_depth))
        if auto_tp is not None:                          # #87 D: auto-route cpu-bound models to CPU TP
            ENGINE_CONFIG["auto_tp"] = bool(auto_tp)
        if auto_tp_ratio is not None:                    # trigger when weights > ratio x GPU pool
            ENGINE_CONFIG["auto_tp_ratio"] = max(0.0, float(auto_tp_ratio))
        if auto_load is not None:                        # auto-load a requested model that isn't resident
            ENGINE_CONFIG["auto_load"] = bool(auto_load)
        if autoload_quant is not None:                   # #autoload-smallest: quant for auto-loads
            _aq = str(autoload_quant).lower()
            if _aq in ("int4", "int8", "none"):
                ENGINE_CONFIG["autoload_quant"] = _aq
        if autoload_ctx is not None:                      # #auto-defaults: default ctx for auto/click loads
            ENGINE_CONFIG["autoload_ctx"] = max(0, int(autoload_ctx))
        if autoload_mode is not None:                     # #auto-defaults: default placement mode
            _am = str(autoload_mode).lower()
            if _am in LOAD_MODES:
                ENGINE_CONFIG["autoload_mode"] = _am
        if vram_weights_first is not None:               # #vram-weights-first: pack weights into free VRAM
            ENGINE_CONFIG["vram_weights_first"] = bool(vram_weights_first)
        if gen_stall_s is not None:                       # #gen-stall-watchdog: wedged-gen reclaim threshold (0=off)
            ENGINE_CONFIG["gen_stall_s"] = max(0.0, float(gen_stall_s))
        if gen_stall_decode_s is not None:                # #active-decode-stall: tighter post-first-token stall (0=off)
            ENGINE_CONFIG["gen_stall_decode_s"] = max(0.0, float(gen_stall_decode_s))
        save_engine_config()
        log_activity(f"config: max_loaded={ENGINE_CONFIG['max_loaded']} "
                     f"auto_unload={ENGINE_CONFIG['auto_unload']} "
                     f"queue_depth={ENGINE_CONFIG.get('queue_depth', DEFAULT_QUEUE_DEPTH)}")
        return JSONResponse({"ok": True, "config": ENGINE_CONFIG})

    @app.get("/gpudiag")             # per-process GPU usage on the CONTROLLER host (this box)
    async def gpudiag() -> JSONResponse:
        """Run nvidia-smi locally and return which PROCESSES hold this GPU. The controller
        runs on a GPU node (beast), so this distinguishes InfiniteModel's own worker python
        from other tenants (Ollama, other inference) when the GPU looks unexpectedly full."""
        def _run():
            import subprocess
            out = {"host": platform.node()}
            try:
                g = subprocess.run(["nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                out["gpu"] = g.stdout.strip()
                p = subprocess.run(["nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                procs = []
                for line in p.stdout.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 3:
                        procs.append({"pid": parts[0], "name": parts[1],
                                      "used_mib": parts[2]})   # may be "[N/A]" on Windows/WDDM
                def _mib(x):
                    try:
                        return int(x["used_mib"])
                    except (ValueError, TypeError):
                        return -1                              # non-numeric ([N/A]) sorts last
                out["processes"] = sorted(procs, key=lambda x: -_mib(x))
                if procs and all(_mib(x) < 0 for x in procs):
                    out["note"] = ("per-process VRAM is [N/A] on Windows/WDDM — "
                                   "PIDs/names listed, but only the GPU total is exact")
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.post("/gc_cache")           # dashboard: reclaim disk from redundant HF-cache copies
    async def gc_cache() -> JSONResponse:
        r = await asyncio.to_thread(gc_redundant_cache)
        if r.get("removed"):
            log_activity(f"cache GC: freed {r['freed_gb']} GB "
                         f"({len(r['removed'])} redundant copies removed)")
        return JSONResponse(r)
