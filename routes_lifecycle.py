"""routes_lifecycle: routes relocated from server.py build_app (m4c153 code-split). Route bodies
are BYTE-IDENTICAL to the originals; their module globals (engine, registry, _serve,
build_status, JSONResponse …) are injected at startup by state.bind() — see state.py.
build_app() calls register(app) to attach them. Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def register(app):

    @app.get("/shard_status")           # #shard-cache: which quants are pre-compiled per model
    async def shard_status_ep(model: Optional[str] = None) -> JSONResponse:
        def _status_for(friendly: str) -> dict:
            tgt = MODELS[friendly][0] if friendly in MODELS else friendly
            d = _local_model_dir(tgt)
            return shard_cache_status(d) if d else {}
        if model:
            try:
                friendly = resolve_model_name(model)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            return JSONResponse({"model": _ollama_name(friendly), "cache": _status_for(friendly)})
        # all registered models that are downloaded (cheap — just reads manifests)
        out = {}
        for friendly in MODELS:
            st = await asyncio.to_thread(_status_for, friendly)
            if st:
                out[_ollama_name(friendly)] = st
        return JSONResponse({"caches": out})

    @app.post("/verify_shards")         # #shard-cache: full sha256 integrity check (for the popup)
    async def verify_shards_ep(model: str, quant: str = "int4") -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_local_model_dir, tgt)
        if not d:
            return JSONResponse({"error": "model not downloaded"}, status_code=404)
        ok, problems = await asyncio.to_thread(verify_shard_cache, d, quant)
        return JSONResponse({"ok": ok, "problems": problems, "quant": quant})

    @app.post("/pack_result")   # #distributed-packing: a worker returns a packed shard-cache unit
    async def pack_result(req: Request, req_id: str = "", unit: str = "",
                          model_id: str = "", quant: str = "int4") -> JSONResponse:
        body = await req.body()
        mt = {}
        h = req.headers.get("x-manifest")
        if h:
            with contextlib.suppress(Exception):
                import base64
                mt = json.loads(base64.b64decode(h).decode())
        engine._pack_results[req_id] = {"unit": unit, "model_id": model_id, "quant": quant,
                                        "bytes": body, "mtensors": mt}
        f = engine._pack_futures.get(req_id)
        if f is not None and not f.done():
            f.set_result(req_id)
        return JSONResponse({"ok": True, "req_id": req_id, "bytes": len(body)})

    @app.post("/pack_probe")    # #distributed-packing Inc 1b: dispatch ONE unit to a worker, byte-check vs local
    async def pack_probe(model: str, node: str = "", layer: int = 0, quant: str = "int4") -> JSONResponse:
        """Offload-pack ONE decoder-layer unit on a worker and prove the result is BIT-IDENTICAL to a
        local compile (the gate before fanning the whole compile out across the fleet). Dense int4/int8."""
        import shards as _sh
        import urllib.parse as _up
        import urllib.request as _ur
        from safetensors.torch import load as _stload, save as _stsave
        if quant not in ("int4", "int8"):
            return JSONResponse({"ok": False, "error": "int4|int8 only"}, status_code=400)
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        mdir = await asyncio.to_thread(_controller_model_dir, tgt)
        if not mdir:
            return JSONResponse({"error": "model not available on the controller"}, status_code=404)
        cand = [n for n in registry.alive_sorted() if n.can_infer and (not node or n.hostname == node)]
        if not cand:
            return JSONResponse({"ok": False, "error": f"no alive worker matching node='{node}'"}, status_code=404)
        nd = cand[0]
        link = engine.links.get(nd.node_id)
        if link is None:
            return JSONResponse({"ok": False, "error": f"no control link to {nd.hostname}"}, status_code=503)
        scope = await asyncio.to_thread(_sh._quant_scope, mdir)   # exact scope (== local compile)
        lin2d = sorted(scope[0]) if scope else None
        exp3d = sorted(scope[1]) if scope else None
        wm = await asyncio.to_thread(_sh._weight_map, mdir)        # per-expert MoE -> worker must fuse (Inc 3b)
        _is_moe = bool(await asyncio.to_thread(_sh._has_moe_experts, wm))
        _moe_fused = any(s.endswith(".gate_up_proj") or s.endswith(".down_proj") for s in wm)
        _need_skel = _is_moe and not _moe_fused
        _skel = scope[2] if (scope and _need_skel) else None
        req_id = f"pk-{int(time.time()*1000)}-{layer}"
        unit = f"L{int(layer):04d}.safetensors"
        fut = asyncio.get_event_loop().create_future()
        engine._pack_futures[req_id] = fut
        frame = {"type": "pack", "req_id": req_id, "model_id": tgt, "quant": quant,
                 "group_size": _sh.INT4_GROUP, "unit": unit, "start": int(layer), "end": int(layer) + 1,
                 "embed": 0, "head": 0, "lin2d": lin2d, "exp3d": exp3d, "fuse": _need_skel,
                 "controller_http_port": ARGS.http_port}
        t0 = time.monotonic()
        try:
            await link.send(frame)
            await asyncio.wait_for(fut, timeout=600)
        except Exception as exc:
            engine._pack_futures.pop(req_id, None)
            return JSONResponse({"ok": False, "error": f"remote pack failed: {exc!r}"}, status_code=504)
        finally:
            engine._pack_futures.pop(req_id, None)
        res = engine._pack_results.pop(req_id, None)
        if not res:
            return JSONResponse({"ok": False, "error": "no pack result received"}, status_code=504)
        remote_ms = round((time.monotonic() - t0) * 1000)
        worker_blob = res["bytes"]

        def _local():   # reference pack of the SAME unit (our own /weights -> identical bytes -> identical pack)
            url = (f"http://127.0.0.1:{ARGS.http_port}/weights?model={_up.quote(tgt)}"
                   f"&start={int(layer)}&end={int(layer)+1}&embed=0&head=0&skip_experts=0")
            with _ur.urlopen(url, timeout=600) as r:
                raw = _stload(r.read())
            out_sd, _mt = _sh.pack_unit_tensors(
                raw, (set(lin2d) if lin2d is not None else None),
                (set(exp3d) if exp3d is not None else None), _skel, quant, _sh.INT4_GROUP)
            return _stsave(out_sd)
        local_blob = await asyncio.to_thread(_local)
        identical = (worker_blob == local_blob)
        tcmp = identical
        if not identical:           # robust fallback: metadata order can differ, compare tensors
            import torch as _t
            wsd, lsd = _stload(worker_blob), _stload(local_blob)
            tcmp = (set(wsd) == set(lsd)) and all(_t.equal(wsd[k], lsd[k]) for k in wsd)
        log_activity(f"pack_probe {_ollama_name(friendly)} {unit} on {nd.hostname}: "
                     f"byte_identical={identical} tensor_identical={tcmp} ({remote_ms} ms)")
        return JSONResponse({"ok": True, "node": nd.hostname, "unit": unit, "remote_ms": remote_ms,
                             "worker_bytes": len(worker_blob), "local_bytes": len(local_blob),
                             "byte_identical": identical, "tensor_identical": tcmp,
                             "tensors": len(res.get("mtensors") or {})})

    @app.post("/compile_dist")   # #distributed-packing Inc 2: compile a shard cache by fanning unit-packs across workers
    async def compile_dist(model: str, quant: str = "int4") -> JSONResponse:
        """Compile a model's pre-quantized shard cache by DISTRIBUTING the per-layer pack across the
        fleet (exo-inspired): each worker fetches a layer's bf16 from /weights, packs it with the
        SHARED shards.pack_unit_tensors (bit-identical to a local compile, proven by /pack_probe), and
        POSTs it back; the controller assembles the cache + manifest. embed/head are packed locally
        (few units, tied-embedding edge cases); any worker failure falls back to a LOCAL pack of that
        layer. Runs in the MAIN process (it owns the control links) — safe because the heavy packing is
        now ON THE WORKERS, not the controller. Dense int4/int8 only (MoE needs the worker skeleton =
        Inc 3); the proven local /compile_shards stays the path for MoE / single-box."""
        import hashlib
        import shards as _sh
        import urllib.parse as _up
        import urllib.request as _ur
        from safetensors.torch import load as _stload, save as _stsave
        if quant not in ("int4", "int8"):
            return JSONResponse({"ok": False, "error": "int4|int8 only"}, status_code=400)
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        mdir = await asyncio.to_thread(_controller_model_dir, tgt)
        if not mdir:
            return JSONResponse({"error": "model not available on the controller"}, status_code=404)
        wm = await asyncio.to_thread(_sh._weight_map, mdir)
        # #distributed-packing Inc 3a/3b: DENSE, FUSED-MoE and PER-EXPERT MoE (Mixtral/OLMoE) are all
        # supported. Fused-MoE needs no fusion (skel=None). Per-expert MoE (checkpoint has experts.N.*,
        # but transformers 5.x builds the model FUSED-3D) needs the worker to fuse per-expert->3D via
        # `_fuse_moe_experts` against a meta skeleton (built from /modelmeta) — we flag `fuse` in the
        # pack frame so the worker builds it, and pass the local skeleton to the local-fallback pack.
        # int8 MoE still has no 3D-expert quantizer -> reject (matches /compile_shards).
        _is_moe = bool(await asyncio.to_thread(_sh._has_moe_experts, wm))
        _moe_fused = any(s.endswith(".gate_up_proj") or s.endswith(".down_proj") for s in wm)
        _need_skel = _is_moe and not _moe_fused          # per-expert checkpoint -> fuse at pack time
        if _is_moe and quant != "int4":
            return JSONResponse({"ok": False, "error": "MoE distributed compile supports int4 only "
                                 "(no int8 3D-expert quantizer) — use int4"}, status_code=400)
        ckey = f"{friendly}::{quant}"
        if ckey in engine.compiling:
            return JSONResponse({"ok": False, "error": f"{_ollama_name(friendly)} {quant} already compiling"},
                                status_code=409)
        n_layers = await asyncio.to_thread(_sh._model_num_layers, mdir)
        out_dir = os.path.join(_sh._shard_cache_root(mdir), quant)
        await asyncio.to_thread(lambda: os.makedirs(out_dir, exist_ok=True))
        caps = [n for n in registry.alive_sorted() if n.can_infer and engine.links.get(n.node_id)]
        engine.compiling[ckey] = {"model": friendly, "display_model": _ollama_name(friendly), "target": tgt,
                                  "ready": 0, "total": n_layers + 2, "stages_total": max(1, len(caps)),
                                  "stages_ready": 0, "basis": f"distributed {quant} compile "
                                  f"({len(caps)} worker(s))", "warnings": [], "started": time.time()}
        log_activity(f"distributed {quant} compile for {_ollama_name(friendly)} -> "
                     f"{n_layers} layers across {len(caps)} worker(s)…")
        scope = await asyncio.to_thread(_sh._quant_scope, mdir)
        lin2d = sorted(scope[0]) if scope else None
        exp3d = sorted(scope[1]) if scope else None
        _lset = set(lin2d) if lin2d is not None else None
        _eset = set(exp3d) if exp3d is not None else None
        # Per-expert MoE (Inc 3b): the local-fallback pack must FUSE per-expert->3D too. The skeleton
        # is scope[2] (the same meta model the worker rebuilds). For dense / already-fused checkpoints
        # _fuse_moe_experts is a no-op, so passing it unconditionally when per-expert is safe.
        _skel = scope[2] if (scope and _need_skel) else None
        with open(os.path.join(mdir, "config.json"), encoding="utf-8") as fh:
            tied = bool(json.load(fh).get("tie_word_embeddings", False))
        base_local = f"http://127.0.0.1:{ARGS.http_port}"

        def _pack_local(start: int, end: int, embed: int, head: int):
            url = (f"{base_local}/weights?model={_up.quote(tgt)}&start={start}&end={end}"
                   f"&embed={int(embed)}&head={int(head)}&skip_experts=0")
            with _ur.urlopen(url, timeout=1800) as r:
                raw = _stload(r.read())
            out_sd, mt = _sh.pack_unit_tensors(raw, _lset, _eset, _skel, quant, _sh.INT4_GROUP)
            return _stsave(out_sd), mt

        _ptag = getattr(_sh, "_packer_tag", None)   # tolerate a lagged shards.py on the controller
        manifest = {"format": 1, "quant": quant, "group_size": _sh.INT4_GROUP, "num_layers": n_layers,
                    "tied": tied, "files": {}, "tensors": {},
                    "packer_hash": (_ptag(quant, _sh.INT4_GROUP) if _ptag else None),  # Inc 4 drift guard
                    "expert_layout": ("fused3d" if _is_moe else None)}   # Inc 3a: fused-MoE serve-from-cache
        _done = {"n": 0}

        def _write(unit: str, blob: bytes, mt: dict) -> None:   # inline (no thread) -> manifest dict race-free
            with open(os.path.join(out_dir, unit), "wb") as f:
                f.write(blob)
            manifest["files"][unit] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
            for name, meta in mt.items():
                manifest["tensors"][name] = {"file": unit, **meta}
            _done["n"] += 1
            c = engine.compiling.get(ckey)
            if c:
                c["ready"] = _done["n"]

        async def _dispatch_layer(node, i: int):
            link = engine.links.get(node.node_id)
            if link is None:
                raise RuntimeError(f"no link to {node.hostname}")
            req_id = f"cd-{int(time.time()*1000)}-{i}-{node.node_id}"
            fut = asyncio.get_event_loop().create_future()
            engine._pack_futures[req_id] = fut
            frame = {"type": "pack", "req_id": req_id, "model_id": tgt, "quant": quant,
                     "group_size": _sh.INT4_GROUP, "unit": f"L{i:04d}.safetensors",
                     "start": i, "end": i + 1, "embed": 0, "head": 0,
                     "lin2d": lin2d, "exp3d": exp3d, "fuse": _need_skel,   # Inc 3b: worker fuses per-expert->3D
                     "controller_http_port": ARGS.http_port}
            try:
                await link.send(frame)
                await asyncio.wait_for(fut, timeout=1800)
            finally:
                engine._pack_futures.pop(req_id, None)
            res = engine._pack_results.pop(req_id, None)
            if not res:
                raise RuntimeError("no pack result")
            return res["bytes"], res["mtensors"]

        _part: set = set()   # node_ids that have packed >=1 unit via the worker path (live node count)

        async def _run():
            try:
                eb, emt = await asyncio.to_thread(_pack_local, 0, 0, 1, 0)
                _write("embed.safetensors", eb, emt)
                q: asyncio.Queue = asyncio.Queue()
                for i in range(n_layers):
                    q.put_nowait(i)

                async def _node_loop(node):
                    while True:
                        try:
                            i = q.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        unit = f"L{i:04d}.safetensors"
                        try:
                            blob, mt = await _dispatch_layer(node, i)
                            if node.node_id not in _part:   # first unit from this worker -> live "N nodes" + log
                                _part.add(node.node_id)
                                c = engine.compiling.get(ckey)
                                if c:
                                    c["stages_ready"] = len(_part)
                                log_activity(f"compile_dist: {node.hostname} packing "
                                             f"{_ollama_name(friendly)} {quant} layers")
                        except Exception as exc:   # worker died / no shards.py / timeout -> local fallback
                            log_activity(f"compile_dist {unit} on {node.hostname} failed ({exc!r}) -> local pack")
                            blob, mt = await asyncio.to_thread(_pack_local, i, i + 1, 0, 0)
                        _write(unit, blob, mt)

                if caps:
                    await asyncio.gather(*[_node_loop(n) for n in caps])
                else:                              # no workers -> compile fully locally
                    for i in range(n_layers):
                        blob, mt = await asyncio.to_thread(_pack_local, i, i + 1, 0, 0)
                        _write(f"L{i:04d}.safetensors", blob, mt)
                hb, hmt = await asyncio.to_thread(_pack_local, 0, 0, 0, 1)
                _write("head.safetensors", hb, hmt)
                with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                _CACHE_VERIFY_MEMO.pop((mdir, quant), None)   # force a fresh verify on next load
                log_activity(f"distributed {quant} compile DONE for {_ollama_name(friendly)} "
                             f"({n_layers} layers, {len(caps)} worker(s))")
            except Exception as exc:
                log_activity(f"distributed compile FAILED for {_ollama_name(friendly)}: {exc!r}")
            finally:
                engine.compiling.pop(ckey, None)

        asyncio.create_task(_run())
        return JSONResponse({"ok": True, "model": _ollama_name(friendly), "quant": quant,
                             "distributed": True, "workers": len(caps), "layers": n_layers})

    @app.post("/compile_shards")        # #shard-cache: compile a model's pre-quantized cache on beast
    async def compile_shards_ep(model: str, quant: str = "int4") -> JSONResponse:
        if quant not in ("int4", "int8"):
            return JSONResponse({"ok": False, "error": f"shard cache supports int4|int8 (got '{quant}')"},
                                status_code=400)
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tgt = MODELS[friendly][0] if friendly in MODELS else friendly
        mdir = await asyncio.to_thread(_controller_model_dir, tgt)
        if not mdir:
            return JSONResponse({"error": "model not available on the controller"}, status_code=404)
        # Compiles run CONCURRENTLY — with loads and with each other. CRITICAL: each runs in a
        # SUBPROCESS (its own GIL), NOT an in-process thread. The quantize work is Python-heavy (the
        # per-tensor loop + sha256), and asyncio.to_thread keeps it in the controller process where it
        # holds the GIL and starves the single event-loop thread — even on a 124-core box — stalling the
        # data plane enough to drop live generations' logits connection ("data connection closed" bursts).
        # The subprocess can't touch the controller's GIL/event loop; we read its progress over a pipe.
        # Only refuse an EXACT duplicate (same model+quant) since two writers to _shards/<quant>/ corrupt
        # it. Each compile gets its own card in engine.compiling (keyed model::quant) for the dashboard.
        ckey = f"{friendly}::{quant}"
        if ckey in engine.compiling:
            return JSONResponse({"ok": False, "error": f"{_ollama_name(friendly)} {quant} is already "
                                 "compiling"}, status_code=409)
        engine.compiling[ckey] = {"model": friendly, "display_model": _ollama_name(friendly),
                                  "target": tgt, "ready": 0, "total": 1, "stages_total": 1,
                                  "stages_ready": 0, "basis": f"compiling {quant} shard cache (subprocess)",
                                  "warnings": [], "started": time.time()}
        log_activity(f"compiling {quant} shard cache for {_ollama_name(friendly)} (subprocess)…")
        srv_dir = os.path.dirname(os.path.abspath(__file__))
        _script = (
            "import sys, json\n"
            "sys.path.insert(0, sys.argv[3])\n"
            "import shards\n"
            "def p(d, t):\n"
            "    sys.stdout.write('P %d %d\\n' % (d, t)); sys.stdout.flush()\n"
            "m = shards.compile_shards(sys.argv[1], sys.argv[2], progress=p)\n"
            "files = m.get('files', {})\n"
            "sys.stdout.write('DONE ' + json.dumps({'files': len(files),\n"
            "    'bytes': sum(int(v.get('bytes', 0)) for v in files.values()),\n"
            "    'num_layers': m.get('num_layers')}) + '\\n'); sys.stdout.flush()\n")
        # below-normal priority on Windows (beast) so serving is scheduled first; best-effort elsewhere.
        _kw: dict = {}
        if sys.platform == "win32" and hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
            _kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        elif sys.platform != "win32":
            _kw["preexec_fn"] = lambda: os.nice(10)   # Unix: deprioritize
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _script, mdir, quant, srv_dir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **_kw)
        except NotImplementedError:
            proc = None   # event loop (Selector on some Windows setups) can't spawn -> in-process fallback
        except Exception as exc:
            log_activity(f"compile subprocess spawn failed ({exc}); falling back to in-process")
            proc = None
        result: Optional[dict] = None
        err_msg: Optional[str] = None
        try:
            if proc is not None:
                async for raw in proc.stdout:                   # read progress WITHOUT blocking the loop
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("P "):
                        try:
                            _, d_, t_ = line.split()
                            card = engine.compiling.get(ckey)
                            if card is not None:
                                card["ready"], card["total"] = int(d_), int(t_)
                        except ValueError:
                            pass
                    elif line.startswith("DONE "):
                        with contextlib.suppress(Exception):
                            result = json.loads(line[5:])
                await proc.wait()
                if proc.returncode != 0 or result is None:
                    tail = (await proc.stderr.read()).decode("utf-8", "replace").strip().splitlines()
                    err_msg = tail[-1] if tail else f"compile subprocess exit {proc.returncode}"
            else:
                # FALLBACK (no subprocess support): in-process compile. May briefly affect serving on a
                # busy box (the GIL issue this whole change avoids) — logged so the cause is visible.
                log_activity("compile running IN-PROCESS (no subprocess support) — may affect serving")
                def _prog(done: int, total: int) -> None:
                    card = engine.compiling.get(ckey)
                    if card is not None:
                        card["ready"], card["total"] = done, total
                try:
                    man = await asyncio.to_thread(lambda: compile_shards(mdir, quant, progress=_prog))
                    files = man.get("files", {})
                    result = {"files": len(files),
                              "bytes": sum(int(v.get("bytes", 0)) for v in files.values()),
                              "num_layers": man.get("num_layers")}
                except Exception as exc:
                    err_msg = str(exc)
        finally:
            engine.compiling.pop(ckey, None)   # single-owner cleanup of the compile card
        if err_msg is not None or result is None:
            msg = err_msg or "compile failed"
            log_activity(f"shard compile FAILED for {_ollama_name(friendly)}: {msg}")
            return JSONResponse({"ok": False, "error": msg}, status_code=400)
        total_gb = int(result.get("bytes", 0)) / GB
        log_activity(f"shard cache compiled for {_ollama_name(friendly)} "
                     f"({quant}, {result.get('files')} files, {total_gb:.1f} GB)")
        return JSONResponse({"ok": True, "quant": quant, "files": result.get("files"),
                             "size_gb": round(total_gb, 2),
                             "num_layers": result.get("num_layers")})

    @app.post("/load")
    async def load(request: Request, model: str, ctx: int = 0, mode: str = "auto",
                   consolidate: bool = True, quant: str = "none", tp: int = 1,
                   replicas: int = 1, cpu_only: bool = False,
                   moe_offload: bool = False, force: bool = False,
                   node: str = "", kv_quant: str = "",
                   kv_offload: bool = False, temperature: str = "",
                   min_p: str = "", precompile: bool = True) -> JSONResponse:
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
        # `quant`: 'none' (bf16), 'int8' (~1/2), or 'int4' (group-wise ~4.25-bit, ~1/4 — for
        # 200B+ MoEs that won't fit at int8). `tp` (M4): tensor-parallel group
        # size — split every layer across `tp` GPU nodes (rank 0 drives the group over the
        # all-reduce mesh). tp>1 overrides mode. tp must divide num_key_value_heads.
        # Legacy: if mode is omitted but consolidate=false is passed, honor it.
        if quant not in ("none", "int8", "int4"):
            return JSONResponse({"ok": False, "error": f"bad quant '{quant}' (none|int8|int4)"},
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
            if quant == "int8":
                try:
                    import shards as _sh
                    _tgt = MODELS[friendly][0] if friendly in MODELS else friendly
                    _mdir = await asyncio.to_thread(_controller_model_dir, _tgt)
                    if _mdir:
                        _wm = await asyncio.to_thread(_sh._weight_map, _mdir)
                        if await asyncio.to_thread(_sh._has_moe_experts, _wm):
                            log_activity(f"{_ollama_name(friendly)}: int8 on a MoE keeps experts bf16 "
                                         "(no int8 3D-expert quantizer) — DOWNGRADING to int4 for a real "
                                         "memory reduction")
                            quant = "int4"
                except Exception as _moe_exc:
                    log_activity(f"{_ollama_name(friendly)}: MoE check for int8 downgrade failed "
                                 f"({_moe_exc}) — honoring int8 as requested")
            # #cache-on-first-load: for an int4 load with no shard cache yet, BUILD the cache first
            # (this blocks until it's written) so THIS load — and every future load — serves the small
            # pre-packed int4 layers instead of streaming full bf16 and re-quantizing on the fly. No-op
            # when an int4 cache already exists; precompile=0 opts out; skipped for tp (its dispatch path
            # doesn't read the whole-layer cache). Reuses the /compile_shards subprocess (deprioritized,
            # won't starve serving). Non-fatal: any failure falls through to the normal cold load.
            if quant == "int4" and precompile and tp <= 1:
                try:
                    import shards as _sh
                    import urllib.parse as _up
                    _ctgt = MODELS[friendly][0] if friendly in MODELS else friendly
                    _cdir = await asyncio.to_thread(_controller_model_dir, _ctgt)
                    _cst = await asyncio.to_thread(_sh.shard_cache_status, _cdir) if _cdir else {}
                    if _cdir and not (_cst.get("int4") or {}).get("ok"):
                        log_activity(f"{_ollama_name(friendly)}: no int4 shard cache — building it now so "
                                     "this and every future load serve pre-packed (first load is slower)…")
                        _curl = (f"http://127.0.0.1:{ARGS.http_port}/compile_shards"
                                 f"?model={_up.quote(friendly)}&quant=int4")

                        def _build_cache():
                            import urllib.request as _u
                            with _u.urlopen(_u.Request(_curl, method="POST"), timeout=10800) as _r:
                                return _r.read()

                        await asyncio.to_thread(_build_cache)
                except Exception as _ce:
                    log_activity(f"{_ollama_name(friendly)}: pre-load cache build skipped ({_ce!r}) — cold load")
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
                                   requested_by=_req_ip)
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
        if quant not in ("keep", "none", "int8", "int4"):
            return JSONResponse({"ok": False, "error": f"bad quant '{quant}' (keep|none|int8|int4)"},
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

    @app.get("/mtp_probe")
    async def mtp_probe(model: str = "qwen3.6-35b-a3b", mode: str = "dump",
                        prompt: str = "", fresh: int = 0) -> JSONResponse:
        # #91 Increment 1a (discovery): the checkpoint ships an MTP (nextn) head but the installed
        # transformers DROPS it (_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]) — no class to build
        # or run it. To reimplement the MTP forward for self-speculative decoding we first need the
        # EXACT module structure: which mtp.* tensors exist, their shapes/dtypes, and the embed /
        # lm_head / final-norm key names the MTP head shares. Reads the safetensors index only (no
        # model load). Returns a top-level prefix histogram + every mtp.* tensor + the shared-head
        # tensors so the hand-built module matches the checkpoint exactly.
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)

        def _dump() -> dict:
            from safetensors import safe_open
            idx = os.path.join(d, "model.safetensors.index.json")
            if os.path.exists(idx):
                with open(idx, encoding="utf-8") as fh:
                    wm = json.load(fh)["weight_map"]      # tensor_name -> shard filename
            else:                                          # single-file checkpoint
                wm = {}
                single = os.path.join(d, "model.safetensors")
                if os.path.exists(single):
                    with safe_open(single, framework="pt") as sf:
                        wm = {k: "model.safetensors" for k in sf.keys()}
            keys = sorted(wm)
            # prefix histogram (first two dotted segments) so the nesting is visible at a glance
            hist: dict = {}
            for k in keys:
                parts = k.split(".")
                pref = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
                hist[pref] = hist.get(pref, 0) + 1
            # resolve shape/dtype for a set of keys, opening each shard once
            def _meta(want: list) -> dict:
                want = [k for k in want if k in wm]
                by_file: dict = {}
                for k in want:
                    by_file.setdefault(wm[k], []).append(k)
                out: dict = {}
                for fn, ks in by_file.items():
                    with safe_open(os.path.join(d, fn), framework="pt") as sf:
                        for k in ks:
                            sl = sf.get_slice(k)
                            try:
                                dt = sl.get_dtype()
                            except Exception:
                                dt = "?"
                            out[k] = {"shape": list(sl.get_shape()), "dtype": str(dt), "file": fn}
                return out
            mtp_keys = [k for k in keys if k == "mtp" or k.startswith("mtp.")]
            # the shared head + embedding the MTP module reuses (names vary by multimodal nesting)
            shared = [k for k in keys if any(s in k for s in (
                "embed_tokens", "lm_head", "language_model.norm", ".model.norm.", )) or k.endswith("model.norm.weight")]
            return {
                "model": friendly, "target": target, "n_tensors": len(keys),
                "prefix_histogram": dict(sorted(hist.items())),
                "mtp": _meta(mtp_keys),
                "shared_head_candidates": _meta(shared),
            }

        async def _dprobe() -> dict:
            # #91 Increment 2 (distributed-hidden probe): with the model loaded DISTRIBUTED, run a
            # prefill that returns the pre-final-norm trunk hidden (capture_pre_norm), then run the
            # small controller-resident MTP head over the sequence and measure how often its drafted
            # token matches the pipeline's own greedy continuation. NEVER loads the full model here
            # (see never-full-load-on-controller-box) — only the ~few-GB MTP head.
            import importlib
            here = os.path.dirname(os.path.abspath(__file__))
            with contextlib.suppress(Exception):    # iterate the MTP forward w/o a controller restart
                remote = _fetch_repo_file("mtp_core.py")
                if remote and len(remote) > 80:
                    with open(os.path.join(here, "mtp_core.py"), "wb") as fh:
                        fh.write(remote)
            import mtp_core as _mc
            importlib.reload(_mc)
            m = engine.models.get(friendly) or engine._pick_replica(friendly)
            if m is None or getattr(m, "stage0_writer", None) is None:
                return {"error": f"{friendly} is not loaded distributed — load it first"}
            if fresh and getattr(engine, "_mtp_heads", None):
                engine._mtp_heads.pop(friendly, None)
            head = await engine._ensure_mtp_head(m)   # GPU-preferring, shared with decode
            if head is None:
                return {"error": f"no MTP head for {friendly} (mtp_num_hidden_layers==0 or load failed)"}
            import torch
            p = prompt or ("The capital of France is Paris. The capital of Japan is Tokyo. "
                           "The capital of Italy is Rome. The capital of Canada is Ottawa. "
                           "The capital of Germany is")
            ids = m.tokenizer(p, return_tensors="pt").input_ids
            S = int(ids.shape[1])
            if S < 4:
                return {"error": "prompt too short"}
            # prefill on the distributed pipeline; capture per-position logits + pre-norm hidden.
            async with m.lock:
                logits, h_pre = await engine._send(m, ids, 0, True, all_logits=True,
                                                   capture_pre_norm=True)
                await engine._crop(m, 0)   # reset the probe's KV so it can't pollute a later gen

            def _compute() -> dict:
                th = h_pre[:, 0:S - 1]
                nxt = ids[:, 1:S]
                main_greedy = logits[0].float().argmax(-1)   # main_greedy[j] predicts token j+1
                actual = ids[0]
                out = {}
                for off in (0, 1):
                    ml = _mc.mtp_forward_seq(head, th, nxt, position_offset=off)
                    mtp_pred = ml[0].float().argmax(-1)       # mtp_pred[i] predicts token i+2
                    n = S - 2
                    ag = aa = 0
                    ex = []
                    for i in range(n):
                        mp = int(mtp_pred[i]); tg = int(main_greedy[i + 1]); ta = int(actual[i + 2])
                        ag += (mp == tg); aa += (mp == ta)
                        if len(ex) < 8:
                            ex.append({"i": i, "mtp": mp, "greedy": tg, "actual": ta,
                                       "mtp_tok": m.tokenizer.decode([mp]),
                                       "greedy_tok": m.tokenizer.decode([tg])})
                    out[f"off{off}"] = {"acc_vs_greedy": round(ag / max(1, n), 3),
                                        "acc_vs_actual": round(aa / max(1, n), 3), "n": n,
                                        "examples": ex}
                # DIAGNOSTIC: incremental (decode-path) drafts vs the proven parallel forward_seq.
                # If these disagree, the KV/attention path mtp_step uses at decode time is broken.
                inc = _mc.mtp_incremental_drafts(head, th, nxt)         # [S-1] argmax tokens
                par = _mc.mtp_forward_seq(head, th, nxt, position_offset=0)[0].float().argmax(-1)
                n = S - 2
                same_par = sum(1 for i in range(S - 1) if inc[i] == int(par[i]))
                inc_vs_greedy = sum(1 for i in range(n) if inc[i] == int(main_greedy[i + 1]))
                out["incremental"] = {
                    "matches_parallel": round(same_par / max(1, S - 1), 3),
                    "acc_vs_greedy": round(inc_vs_greedy / max(1, n), 3), "n": n}
                return out

            out = await asyncio.to_thread(_compute)
            best = max((k for k in out if k.startswith("off")),
                       key=lambda k: out[k]["acc_vs_greedy"])
            return {"ok": True, "model": friendly, "S": S, "best": best,
                    "summary": {k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                for k, v in out.items()},
                    "examples": out[best]["examples"],
                    "load_missing": head.load_missing[:10],
                    "load_unexpected": head.load_unexpected[:10]}

        async def _qcheck() -> dict:
            # #91 (b) diagnostic: does a q=2 DECODE chunk (what the spec verify sends) produce the same
            # per-position logits as two sequential q=1 decodes? On qwen3.6's HYBRID Gated-DeltaNet this
            # is the suspected source of the verify divergence. Isolates WHERE: pos0 (first chunk token)
            # vs pos1 (chunked-continuation token). pos0 mismatch => prefill->decode handoff / first
            # chunk position bug (fixable); pos1-only => chunked-vs-recurrent linear-attn (fundamental).
            import torch
            m = engine.models.get(friendly) or engine._pick_replica(friendly)
            if m is None or getattr(m, "stage0_writer", None) is None:
                return {"error": f"{friendly} is not loaded distributed — load it first"}
            p = prompt or "The capital of France is Paris. The capital of Japan is"
            ids = m.tokenizer(p, return_tensors="pt").input_ids
            P = int(ids.shape[1])

            async with m.lock:
                prelog = await engine._send(m, ids, 0, True)            # prefill -> KV @ P
                a = int(prelog[0, -1].float().argmax())
                lb0, hb0 = await engine._send(m, torch.tensor([[a]], dtype=torch.long), P, False,
                                              all_logits=True, capture_pre_norm=True)
                b = int(lb0[0, 0].float().argmax())
                lb1, hb1 = await engine._send(m, torch.tensor([[b]], dtype=torch.long), P + 1, False,
                                              all_logits=True, capture_pre_norm=True)
                await engine._crop(m, P)
                la, ha = await engine._send(m, torch.tensor([[a, b]], dtype=torch.long), P, False,
                                            all_logits=True, capture_pre_norm=True)
                await engine._crop(m, 0)

            def _cmp(x, y) -> dict:
                xf, yf = x.float(), y.float()
                return {"max_abs": round(float((xf - yf).abs().max()), 4),
                        "argmax_eq": int(xf.argmax()) == int(yf.argmax()),
                        "argmax_chunk": int(xf.argmax()), "argmax_seq": int(yf.argmax())}

            return {"ok": True, "P": P, "a": a, "b": b,
                    "pos0_logits": _cmp(la[0, 0], lb0[0, 0]),
                    "pos1_logits": _cmp(la[0, 1], lb1[0, 0]),
                    "pos0_hidden_max_abs": round(float((ha[0, 0].float() - hb0[0, 0].float()).abs().max()), 4),
                    "pos1_hidden_max_abs": round(float((ha[0, 1].float() - hb1[0, 0].float()).abs().max()), 4)}

        try:
            if mode == "run":
                # DISABLED (m4c122): the original run-mode loaded the FULL model on this co-hosted
                # controller box and OOM-crashed the controller. Use mode=dprobe (distributed hidden +
                # small MTP head) instead. See never-full-load-on-controller-box.
                return JSONResponse({"error": "mode=run disabled (crashed the co-hosted controller). "
                                     "Use mode=dprobe — distributed hidden + small MTP head."},
                                    status_code=400)
            if mode == "dprobe":
                return JSONResponse(await _dprobe())
            if mode == "qcheck":
                return JSONResponse(await _qcheck())
            return JSONResponse(await asyncio.to_thread(_dump))
        except Exception as exc:
            import traceback
            return JSONResponse({"error": repr(exc), "tb": traceback.format_exc()[-1500:]},
                                status_code=500)

    @app.get("/modelcode")
    async def modelcode(model: str) -> JSONResponse:
        # Serve the model's trust_remote_code python files (the auto_map modeling/configuration
        # *.py) so a WORKER — which builds the skeleton from config alone — can construct the
        # CORRECT architecture instead of falling back to transformers' native class. Without this,
        # a remote-code model (e.g. MiniMax-M2, whose checkpoint is full-attention but whose
        # model_type 'minimax' maps natively to the OLDER lightning Text-01 arch) builds the wrong
        # modules and every mismatched tensor stays on 'meta'. Returns {filename: source}; empty
        # {} for a model with no auto_map (the worker then keeps the native-class path).
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        def _collect() -> dict:
            try:
                with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
                    if not (json.load(fh) or {}).get("auto_map"):
                        return {}                      # not a remote-code model -> nothing to ship
            except Exception:
                return {}
            def _read_py() -> dict:
                o = {}
                for fn in os.listdir(d):               # all .py in the snapshot (modeling + its imports)
                    if fn.endswith(".py"):
                        with contextlib.suppress(Exception):
                            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                                o[fn] = fh.read()
                return o
            out = _read_py()
            if not out:
                # auto_map is set but the dir has NO .py — a model pulled before *.py was added to the
                # download patterns (e.g. MiniMax-M2). Fetch the repo's .py from HF hub into the dir
                # ON-DEMAND (small), then re-read — fixes already-downloaded models without re-pulling
                # the weights. Best-effort: any failure leaves out={} (worker keeps the native path). #78
                with contextlib.suppress(Exception):
                    from huggingface_hub import HfApi, hf_hub_download
                    tok = HF_TOKEN or None
                    for f in HfApi().list_repo_files(target, token=tok):
                        if f.endswith(".py"):
                            with contextlib.suppress(Exception):
                                hf_hub_download(target, f, token=tok, local_dir=d)
                    out = _read_py()
                    if out:
                        print(f"[modelcode] fetched {len(out)} trust_remote_code .py for {target} "
                              f"(dir was missing them)", flush=True)
            return out
        return JSONResponse(await asyncio.to_thread(_collect))

    @app.get("/weights")
    async def weights(model: str, start: int, end: int, embed: int = 0, head: int = 0,
                      skip_experts: int = 0, cache: str = ""):
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        # #shard-cache Inc 2 (serve-from-cache): when the controller flagged this load cache=int4, the
        # worker requests pre-packed units. Each unit (embed / one layer / head) is its own cache file,
        # already EXACTLY this stage's int4-packed tensors (+ bf16 norms/biases) — stream it whole, no
        # plan/dequant/quant. The controller decides verify+enablement; here we just serve the file if
        # present. Missing file -> fall through to the bf16 stream (per-unit safe fallback).
        if cache and d:
            cunit = await asyncio.to_thread(
                cache_unit_path, d, cache, start, end, bool(embed), bool(head))
            if cunit and (bool(embed) or bool(head) or end - start == 1):
                ctotal = os.path.getsize(cunit)

                def _owns_c(n) -> bool:
                    ls, le = n.layer_start, n.layer_end
                    if ls is None or le is None:
                        return False
                    if end > start:
                        return ls <= start and end <= le
                    return ls == start
                cnid = next((n.node_id for n in registry._nodes.values() if _owns_c(n)), None)
                chost = registry._nodes[cnid].hostname if cnid in registry._nodes else "?"
                log_activity(f"serving {friendly} CACHED {cache} L{start}-{end} -> {chost} "
                             f"({ctotal / GB:.2f} GB)")

                def _cgen():
                    with open(cunit, "rb") as f:
                        while True:
                            chunk = f.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            net_account(cnid, to_node=len(chunk))
                            yield chunk
                    ld = next((c for c in engine.loadings.values()
                               if c.get("target") == target), None)
                    if ld is not None:
                        ld["ready"] = ld.get("ready", 0) + max(1, (end - start)
                                                               + (1 if embed else 0)
                                                               + (1 if head else 0))

                return StreamingResponse(
                    _cgen(), media_type="application/octet-stream",
                    headers={"Content-Length": str(ctotal),
                             "Content-Disposition":
                                 f'attachment; filename="{friendly}-cache-{start}_{end}.safetensors"'})
        # Stream the stage's tensors straight from the source files (raw bytes, 8 MB chunks):
        # bounded memory, no temp blob, no lock -> every worker pulls its full slice
        # concurrently in one smooth pass. skip_experts (#62) omits the fused 3D MoE experts so
        # the worker can stream them per-expert via /experts (no ~7 GB layer blob in RAM).
        header_bytes, parts, total = await asyncio.to_thread(
            _plan_weight_stream, d, start, end, bool(embed), bool(head), bool(skip_experts))
        # meter against the node whose layer range this serves (controller -> node), per
        # chunk so the dashboard rate tracks the real transfer instead of one upfront spike
        # Attribute this slice to its node. Per-layer streaming (m4ak) requests single layers
        # (start=i,end=i+1) and embed/head as start==end slices, so match by CONTAINING range
        # rather than exact endpoints (the old exact match missed every streamed fetch -> nid
        # None -> traffic unmetered + host '?'). A full-range fetch (TP/from_file) still matches.
        def _owns(n) -> bool:
            ls, le = n.layer_start, n.layer_end
            if ls is None or le is None:
                return False
            if end > start:                       # a layer slice -> node whose range contains it
                return ls <= start and end <= le
            return ls == start                    # embed/head slice (start==end) -> node starting here
        nid = next((n.node_id for n in registry._nodes.values() if _owns(n)), None)
        _host = registry._nodes[nid].hostname if nid in registry._nodes else "?"
        log_activity(f"serving {friendly} weights L{start}-{end} -> {_host} ({total / GB:.2f} GB)")

        def _gen():
            net_account(nid, to_node=len(header_bytes))
            yield header_bytes
            for p in parts:
                if p.get("kind") in ("fp8", "nvfp4"):   # quantized checkpoint: dequant -> bf16, stream bf16
                    deq = (_nvfp4_dequant_part_bytes(p) if p["kind"] == "nvfp4"
                           else _fp8_dequant_part_bytes(p))
                    for i in range(0, len(deq), 8 * 1024 * 1024):
                        chunk = deq[i:i + 8 * 1024 * 1024]
                        net_account(nid, to_node=len(chunk))
                        yield chunk
                    continue
                with open(p["fn"], "rb") as f:
                    f.seek(p["off"])
                    left = p["nbytes"]
                    while left > 0:
                        chunk = f.read(min(8 * 1024 * 1024, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                        net_account(nid, to_node=len(chunk))
                        yield chunk
            # whole slice streamed -> the worker now mmap-loads + fuses + places it
            log_activity(f"  {_host}: received L{start}-{end} ({total / GB:.2f} GB), building shard")
            # live per-shard progress: each Lxx layer-slice (+ the embed/head slices) is ONE shard
            # the dashboard counts. Workers pull their layers sequentially, so these completions
            # pace real load progress far better than the one-tick-per-node stage count.
            # match on the HF target id (the /weights `model` param is the target, e.g.
            # 'ModelCloud/MiniMax-M2-BF16', NOT the friendly 'minimax-m2') so the counter advances.
            # With parallel loads, find THIS model's card among the in-flight cards by target.
            ld = next((c for c in engine.loadings.values() if c.get("target") == target), None)
            if ld is not None:
                ld["ready"] = ld.get("ready", 0) + max(1, (end - start)
                                                       + (1 if embed else 0)
                                                       + (1 if head else 0))

        return StreamingResponse(
            _gen(), media_type="application/octet-stream",
            headers={"Content-Length": str(total),
                     "Content-Disposition":
                         f'attachment; filename="{friendly}-{start}_{end}.safetensors"'})

    @app.get("/weights_tp")
    async def weights_tp(model: str, start: int, end: int, tp_rank: int, tp_size: int,
                         embed: int = 0, head: int = 0, weights: str = ""):
        # TP-v2 per-rank serve (#62 follow-on): return this stage's tensors ALREADY SLICED for
        # (tp_rank, tp_size) — column-parallel q/k/v/gate/up on dim 0, row-parallel o/down on dim 1
        # (bias dropped), embed/norm/head/layernorm/rotary whole. The row slice is non-contiguous so
        # we read+materialize (NOT byte-range) and serve a small built safetensors blob. Lets a TP
        # rank hold only ~1/tp of each layer instead of the v1 load-full-then-shard footprint.
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        if tp_size < 1 or tp_rank < 0 or tp_rank >= tp_size:
            return JSONResponse({"error": f"bad tp (rank={tp_rank}, size={tp_size})"},
                                status_code=400)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        # heterogeneous TP: the rank passes the group's per-rank capacity weights (comma list) so the
        # serve slices match the rank's reduced-dim structure; empty -> uniform 1/tp (backward compat).
        try:
            wlist = [float(x) for x in weights.split(",") if x.strip()] if weights else None
        except ValueError:
            wlist = None
        try:
            blob = await asyncio.to_thread(
                _build_weight_tp_blob, d, start, end, bool(embed), bool(head), tp_rank, tp_size, wlist)
        except Exception as exc:
            return JSONResponse({"error": f"tp-slice build failed: {exc!r}"}, status_code=500)
        total = len(blob)
        # meter to the node whose layer range contains this slice (controller -> node), same _owns
        # logic as /weights; a TP rank requests its FULL [0,L) range so the containing match holds.
        def _owns(n) -> bool:
            ls, le = n.layer_start, n.layer_end
            if ls is None or le is None:
                return False
            if end > start:
                return ls <= start and end <= le
            return ls == start
        # Match by tp_rank too: both ranks share layer range [0,L], so _owns alone matches BOTH and
        # next() would mislabel every slice to the first node. Prefer the node assigned THIS tp_rank;
        # fall back to the old range-only match if none (keeps metering working pre-assignment).
        nid = next((n.node_id for n in registry._nodes.values()
                    if _owns(n) and getattr(n, "tp_rank", None) == tp_rank), None) \
            or next((n.node_id for n in registry._nodes.values() if _owns(n)), None)
        _host = registry._nodes[nid].hostname if nid in registry._nodes else "?"
        log_activity(f"serving {friendly} TP weights L{start}-{end} rank {tp_rank}/{tp_size} "
                     f"-> {_host} ({total / GB:.2f} GB)")

        def _gen():
            for i in range(0, total, 8 * 1024 * 1024):
                chunk = blob[i:i + 8 * 1024 * 1024]
                net_account(nid, to_node=len(chunk))
                yield chunk
            ld = next((c for c in engine.loadings.values() if c.get("target") == target), None)
            if ld is not None:
                ld["ready"] = ld.get("ready", 0) + max(1, (end - start)
                                                       + (1 if embed else 0)
                                                       + (1 if head else 0))

        return StreamingResponse(
            _gen(), media_type="application/octet-stream",
            headers={"Content-Length": str(total),
                     "Content-Disposition":
                         f'attachment; filename="{friendly}-{start}_{end}-tp{tp_rank}of{tp_size}.safetensors"'})

    @app.get("/experts")
    async def experts(model: str, layer: int, e0: int, k: int):
        # Serve experts [e0:e0+k] of one MoE layer as a safetensors blob, raw byte-range from the
        # source files, so a worker fetches + int4-packs one chunk of experts at a time and a big
        # MoE layer never lands whole in RAM (#62). TWO checkpoint layouts, ONE round-trip each:
        #  - NON-FUSED (e.g. MiniMax-M2: *.experts.{e}.{proj}.weight) -> keys '{local_e}.{proj}'
        #    (w1/w2/w3 or gate_proj/up_proj/down_proj), 2D per (expert, projection); the worker
        #    fuses gate+up then packs.
        #  - FUSED (e.g. qwen3.6-35b-a3b: 3D experts.gate_up_proj/down_proj) -> keys 'gate_up_proj'
        #    and 'down_proj', each a 3D [k, out, in] slice; the worker packs each slice directly.
        # The worker auto-detects which layout it got from the returned keys (#75). No per-chunk
        # activity log (thousands of lines).
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        if k <= 0 or e0 < 0 or layer < 0:    # guard: negative k -> negative nbytes -> invalid blob
            return JSONResponse({"error": f"bad range (layer={layer}, e0={e0}, k={k})"},
                                status_code=400)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        d = await asyncio.to_thread(_controller_model_dir, target)
        header_bytes, parts, total = await asyncio.to_thread(
            _plan_experts_chunk, d, layer, e0, k)
        if header_bytes is None:                 # FUSED checkpoint -> serve the 3D fused slices (#75)
            header_bytes, parts, total = await asyncio.to_thread(
                _plan_experts_chunk_fused, d, layer, e0, k)
        if header_bytes is None:
            return JSONResponse({"error": f"no expert tensors (layer {layer})"},
                                status_code=404)
        # meter to the node whose layer range contains this layer (controller -> node)
        def _owns(n) -> bool:
            ls, le = n.layer_start, n.layer_end
            return ls is not None and le is not None and ls <= layer < le
        nid = next((n.node_id for n in registry._nodes.values() if _owns(n)), None)

        def _gen():
            net_account(nid, to_node=len(header_bytes))
            yield header_bytes
            for fn, foff, nbytes in parts:
                with open(fn, "rb") as f:
                    f.seek(foff)
                    left = nbytes
                    while left > 0:
                        chunk = f.read(min(8 * 1024 * 1024, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                        net_account(nid, to_node=len(chunk))
                        yield chunk

        return StreamingResponse(_gen(), media_type="application/octet-stream",
                                 headers={"Content-Length": str(total)})
