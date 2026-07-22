"""routes_api: routes relocated from server.py build_app (m4c153 code-split). Route bodies
are BYTE-IDENTICAL to the originals; their module globals (engine, registry, _serve,
build_status, JSONResponse …) are injected at startup by state.bind() — see state.py.
build_app() calls register(app) to attach them. Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def register(app):

    # ---- Ollama API ----
    @app.get("/api/version")
    async def api_version() -> dict:
        return {"version": OLLAMA_API_VERSION}

    @app.get("/api/tags")
    async def api_tags() -> dict:
        # Only advertise models whose weights are actually present here — a model
        # that isn't downloaded yet can't be distributed, so it isn't "available".
        out = [_tag_entry(name) for name in MODELS if model_ready(MODELS[name][0])]
        seen = {e["name"] for e in out}   # #dedup (BUG-3): never advertise the same rendered name twice
        # advertise ALIASES too (e.g. 'qwen2.5:14b' -> 'qwen2.5:14b-instruct') so a client can
        # discover + use the alias name; same target/size, just the alias display name. Skip an alias
        # whose rendered Ollama name collides with a MODELS entry already listed (e.g. 'qwen2.5-14b'
        # is BOTH a MODELS key AND an alias -> two identical 'qwen2.5:14b' rows without this guard).
        for alias, canon in MODEL_ALIASES.items():
            if canon in MODELS and model_ready(MODELS[canon][0]):
                nm = _ollama_name(alias)
                if nm in seen:
                    continue
                e = _tag_entry(canon)
                e["name"] = e["model"] = nm
                seen.add(nm)
                out.append(e)
        # #federation: also advertise models a PEER controller has RESIDENT. We can genuinely serve
        # these — the Phase 3 middleware proxies the request to that peer — so hiding them would
        # make a usable model look unavailable. Tagged `federated`/`peer` so a client (and the
        # dashboard) can tell "served elsewhere" from "loaded here"; a name we already list always
        # wins, because local weights beat a remote hop.
        with contextlib.suppress(Exception):
            import peers
            if peers.federation_enabled():
                for p in peers.peers_public():
                    if p.get("state") != "ok":
                        continue
                    for m in (p.get("models") or []):
                        nm = _ollama_name(m.get("friendly") or "")
                        if not nm or nm in seen:
                            continue
                        seen.add(nm)
                        out.append({
                            "name": nm, "model": nm,
                            "modified_at": _iso(START_TIME),
                            "size": int(round(float(m.get("size_gb") or 0) * (1 << 30))),
                            "digest": "", "details": {"format": "safetensors", "families": []},
                            "infinitemodel": {"target": m.get("target", ""), "distributed": True,
                                              "federated": True,
                                              "peer": p.get("name") or p.get("host"),
                                              "peer_url": p.get("url", "")},
                        })
        return {"models": out}

    @app.get("/v1/models")
    async def v1_models() -> dict:
        names = [name for name in MODELS if model_ready(MODELS[name][0])]
        names += [a for a, c in MODEL_ALIASES.items() if c in MODELS and model_ready(MODELS[c][0])]
        data, seen = [], set()
        for name in names:                    # #dedup (BUG-3): no duplicate rendered ids
            nm = _ollama_name(name)
            if nm in seen:
                continue
            seen.add(nm)
            data.append({"id": nm, "object": "model",
                         "created": int(START_TIME), "owned_by": "infinitemodel"})
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_id:path}")   # OpenAI retrieve-model (LiteLLM + some clients validate here)
    async def v1_model_get(model_id: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model_id)
        except Exception:
            return _not_found_json(model_id, "openai")
        return JSONResponse({"id": _ollama_name(friendly), "object": "model",
                             "created": int(START_TIME), "owned_by": "infinitemodel"})

    def _media_load_error(exc: Exception, model: str, media: str) -> JSONResponse:
        """#at-capacity (b4a2db0, audit #32) for the MEDIA routes (t2i / tts / speech / t2a) —
        the SAME typed ladder _serve/_serve_embed use (serving.py:#cold-contract), in the OpenAI
        error envelope these routes already speak:
          ValueError (registered but not loaded; auto-load off/updating) -> 404 model_not_found
            (the old blanket except->404 told clients a CAPACITY-failed model doesn't exist —
            a terminal signal; the furnace acestep OOM routing surfaced exactly that);
          CapacityError.terminal (auto-unload off / all residents pinned) -> 503 code=at_capacity
            with NO Retry-After (a retry can never succeed until an operator frees something);
          anything else (capacity-busy / node / OOM load refusal) -> retryable 503 + Retry-After.
        Every refused load is log_activity'd — the #render-oom-guard 503s used to be INVISIBLE
        in telemetry (documented gap): the guard itself logs nothing on refuse, so this line is
        the only fleet-side record that a media auto-load was turned away."""
        if isinstance(exc, ValueError):
            return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error",
                                           "code": "model_not_found"}}, status_code=404)
        _term = isinstance(exc, CapacityError) and getattr(exc, "terminal", False)
        log_activity(f"{media} {model}: auto-load failed — {exc!r}"
                     + (" [at_capacity TERMINAL]" if _term else ""))
        _err = {"message": f"{media} model load failed: {exc}",
                "type": ("server_error" if _term else "model_loading")}
        if _term:
            _err["code"] = "at_capacity"
        return JSONResponse({"error": _err}, status_code=503,
                            headers=({} if _term else {"Retry-After": "3"}))

    @app.post("/v1/images/generations")   # OpenAI Images API (#t2i-serve, task #37)
    async def v1_images_generations(req: Request) -> JSONResponse:
        """Text-to-image via the OpenAI images shape: {model?, prompt, size?, n?,
        response_format?} + extensions {negative_prompt, steps|num_inference_steps,
        cfg|true_cfg_scale|guidance_scale, seed}. Returns b64_json data entries (v1: no
        URL hosting). `model` may be omitted when exactly one image model is loaded.
        Auto-loads a registered-but-cold image model like the chat endpoints do."""
        import base64
        try:
            body = await req.json()
        except Exception:
            return JSONResponse({"error": {"message": "invalid JSON body",
                                           "type": "invalid_request_error"}}, status_code=400)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse({"error": {"message": "'prompt' is required",
                                           "type": "invalid_request_error"}}, status_code=400)
        name = str(body.get("model") or "").strip()
        if not name:
            _t2i = [k for k, m in engine.models.items() if getattr(m, "is_t2i", False)]
            if len(_t2i) == 1:
                name = _t2i[0]
            else:
                return JSONResponse({"error": {"message":
                    "pass 'model' — " + ("no image model is loaded"
                                         if not _t2i else f"multiple loaded: {_t2i}"),
                    "type": "invalid_request_error"}}, status_code=400)
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}},
                                status_code=404)
        try:
            w, h = (int(x) for x in str(body.get("size") or "1024x1024").lower().split("x"))
        except Exception:
            return JSONResponse({"error": {"message": "size must look like '1024x1024'",
                                           "type": "invalid_request_error"}}, status_code=400)
        steps = max(1, min(100, int(body.get("num_inference_steps")
                                    or body.get("steps") or 20)))
        cfg = float(body.get("true_cfg_scale") or body.get("guidance_scale")
                    or body.get("cfg") or 4.0)
        neg = str(body.get("negative_prompt") or " ")
        seed = body.get("seed")
        n = max(1, min(4, int(body.get("n") or 1)))
        if friendly not in engine.models:
            try:
                # auto_load=True: make the docstring's "auto-loads like the chat endpoints" TRUE
                # — without it ensure_loaded (default False) could only ever raise the ValueError
                # this route then mislabeled as an untyped-503 "load failed" (audit #32 NUANCE:
                # the inversion opposite to the 404-on-capacity one).
                await engine.ensure_loaded(friendly, 0, auto_load=True)
            except Exception as exc:
                return _media_load_error(exc, friendly, "t2i")
        data, meta = [], {}
        try:
            for i in range(n):
                s_i = (int(seed) + i) if seed not in (None, "") else None
                png, meta = await engine.t2i_generate(
                    friendly, prompt, negative_prompt=neg, width=w, height=h,
                    steps=steps, cfg=cfg, seed=s_i)
                data.append({"b64_json": base64.b64encode(png).decode()})
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}},
                                status_code=400)
        except asyncio.TimeoutError:
            return JSONResponse({"error": {"message": "image generation timed out",
                                           "type": "server_error"}}, status_code=504)
        except Exception as exc:
            return JSONResponse({"error": {"message": f"image generation failed: {exc}",
                                           "type": "server_error"}}, status_code=500)
        return JSONResponse({"created": int(time.time()), "data": data,
                             "infinitemodel": {"model": friendly, **meta}})

    @app.post("/api/show")
    async def api_show(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError:
            return _not_found_json(name, "chat")   # canonical Ollama model-not-found shape
        spec = resolve_spec(friendly)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if not model_ready(target):
            return JSONResponse(
                {"error": f"model '{friendly}' is not downloaded on the controller"},
                status_code=404)
        d = _local_model_dir(target)             # real measured params/size (MoE-correct)
        # #t2i: a diffusers image checkpoint has NO top-level config.json, so resolve_spec is None
        # and the LLM fields below would 500 (spec.arch on None — the dashboard modal's "full model
        # info unavailable: HTTP 500"). Build the detail view from the diffusers layout instead:
        # model_index.json names the pipeline + its components, transformer/config.json carries the
        # DiT dims, and parameters come from the recursive weight sum (bf16 → ~2 bytes/param).
        if spec is None and d:
            import model_store as _ms
            if _ms._is_diffusers_dir(d):
                import asyncio as _aio
                import os as _os, json as _json

                def _jload(*parts):
                    try:
                        with open(_os.path.join(d, *parts), encoding="utf-8") as _f:
                            return _json.load(_f)
                    except Exception:
                        return {}
                _mi = _jload("model_index.json")
                _tcfg = _jload("transformer", "config.json")
                _comps = sorted(k for k, v in _mi.items()
                                if not k.startswith("_") and isinstance(v, (list, tuple)) and v[0])
                _pclass = str(_mi.get("_class_name") or "diffusers")
                try:
                    _wb = int(await _aio.to_thread(_ms._tree_weight_bytes, d))
                except Exception:
                    _wb = 0
                _pcount = (_wb // 2) or None            # bf16 checkpoint -> ~2 bytes/param
                _heads = _tcfg.get("num_attention_heads")
                _hdim = _tcfg.get("attention_head_dim")
                _minfo = {"general.architecture": _pclass,
                          "general.parameter_count": _pcount,
                          "diffusers.pipeline_class": _pclass,
                          "diffusers.components": ", ".join(_comps),
                          "diffusers.transformer.block_count": _tcfg.get("num_layers"),
                          "diffusers.transformer.hidden_size":
                              (_heads * _hdim) if (_heads and _hdim) else None,
                          "diffusers.transformer.attention.head_count": _heads,
                          "diffusers.transformer.attention.key_length": _hdim,
                          "diffusers.transformer.joint_attention_dim":
                              _tcfg.get("joint_attention_dim"),
                          "diffusers.transformer.in_channels": _tcfg.get("in_channels"),
                          "diffusers.transformer.patch_size": _tcfg.get("patch_size")}
                return JSONResponse({
                    "license": "see model card",
                    "modelfile": f"# InfiniteModel text-to-image (diffusers)\nFROM {target}",
                    "parameters": "", "template": "",
                    "details": {"parent_model": "", "format": "diffusers", "family": _pclass,
                                "families": [_pclass],
                                "parameter_size": (f"{_pcount / 1e9:.1f}B" if _pcount else ""),
                                "quantization_level": "BF16"},
                    "model_info": {k: v for k, v in _minfo.items() if v is not None},
                    "capabilities": ["t2i"],
                    "infinitemodel": {"target": target, "engine": VERSION, "t2i": True,
                                      "distributed": False,   # v1: one controller-co-located GPU node
                                      "num_params": _pcount,
                                      "num_layers": _tcfg.get("num_layers"),
                                      "components": _comps, "config": _tcfg,
                                      "text_encoder_config": _jload("text_encoder", "config.json"),
                                      "vae_config": _jload("vae", "config.json")},
                })
        if d and spec:
            spec = spec_with_measurements(spec, d)
        # Ollama-style capabilities list. Single-sourced from status._model_caps (config-only:
        # image/video/stt/tts/embedding/tools) so /api/show, /status badges, and the dashboard agree.
        import status as _status
        _mcaps = _status._model_caps(target, spec)
        if "t2i" in _mcaps:
            caps = ["t2i"]                   # diffusers image-gen checkpoint — not a chat model (#t2i)
        elif "embedding" in _mcaps:
            caps = ["embedding"]
        else:
            caps = ["completion", "chat"]
            if "tools" in _mcaps:            # #tools: native tool-calling (Ollama reports "tools")
                caps.append("tools")
            if "image" in _mcaps:            # Ollama names the vision capability "vision"
                caps.append("vision")
            for _extra in ("video", "stt", "tts", "ocr"):   # modality badges beyond Ollama's set
                if _extra in _mcaps:
                    caps.append(_extra)
        # #model-detail: surface the RAW on-disk config.json + generation_config.json so the dashboard
        # detail view can show EVERYTHING about a model (loaded or not) — rope theta, sliding window,
        # expert counts, sampling defaults, etc. that the curated model_info doesn't carry. Best-effort;
        # these are small files. Also echo spec-derived sizing so the UI needn't re-derive it.
        raw_cfg, gen_cfg = {}, {}
        if d:
            import os, json as _json
            for _fn, _is_cfg in (("config.json", True), ("generation_config.json", False)):
                _p = os.path.join(d, _fn)
                if os.path.exists(_p):
                    try:
                        with open(_p, "r", encoding="utf-8") as _f:
                            _obj = _json.load(_f)
                        if _is_cfg:
                            raw_cfg = _obj
                        else:
                            gen_cfg = _obj
                    except Exception:
                        pass
        _arch = (getattr(spec, "arch", "") or "").lower()
        im = {"target": target, "draft": MODELS[friendly][1], "distributed": True, "engine": VERSION,
              "default_ctx": getattr(spec, "max_ctx", None),
              "src_dtype": getattr(spec, "src_dtype", None),
              "num_params": getattr(spec, "param_count", None),
              "num_layers": getattr(spec, "num_layers", None),
              "hidden_size": getattr(spec, "hidden_size", None),
              "vocab_size": getattr(spec, "vocab_size", None),
              "num_heads": getattr(spec, "num_heads", None),
              "num_kv_heads": getattr(spec, "num_kv_heads", None),
              "is_moe": any(k in _arch for k in ("moe", "mixtral", "minimax", "deepseek_v")),
              "is_embedding": bool(getattr(spec, "is_embedding", False)),
              "config": raw_cfg, "generation_config": gen_cfg}
        return JSONResponse({
            "license": "see model card", "modelfile": f"# InfiniteModel distributed\nFROM {target}",
            "parameters": "", "template": "{{ .Prompt }}",
            # spec can be None for a registered dir the spec builder can't parse — degrade to
            # empty detail blocks instead of 500ing the whole endpoint (spec.arch on None)
            "details": _details(spec) if spec else {"parent_model": "", "format": "safetensors",
                                                    "family": "", "families": [],
                                                    "parameter_size": "",
                                                    "quantization_level": "BF16"},
            "model_info": _model_info(spec) if spec else {},
            "capabilities": caps,
            "infinitemodel": im,
        })

    @app.get("/api/ps")
    async def api_ps() -> dict:
        nodes = registry.alive_sorted()
        gpus = [n for n in nodes if n.vram_total_gb > 0]
        vram_total = int(sum(n.vram_total_gb for n in gpus) * GB)

        def _expires_at(lm: LoadedModel) -> float:
            # #idle-unload: when the knob is on, a model expires idle_unload_m minutes after its
            # last activity — unless pinned (persist_models), which idle-unload exempts.
            try:
                # min() also drops a hand-edited/legacy inf from engine_config.json — an infinite
                # window would overflow datetime.fromtimestamp below and 500 the whole /api/ps
                _im = min(float(ENGINE_CONFIG.get("idle_unload_m", 0) or 0), 527040.0)
            except (TypeError, ValueError):
                _im = 0.0
            pinned = set(ENGINE_CONFIG.get("persist_models") or {})
            if _im <= 0 or lm.friendly in pinned or getattr(lm, "base", lm.friendly) in pinned:
                return time.time() + 365 * 86400
            last = max(lm.last_used or 0.0, getattr(lm, "last_token_ts", 0.0) or 0.0)
            return last + _im * 60.0

        def _model_vram(lm: LoadedModel) -> int:
            # per-STAGE gpu_bytes (survives node-sharing; the node's single shard_gpu_bytes
            # would be overwritten when a 2nd model lands on it)
            return sum(s.gpu_bytes for s in lm.plan.stages)
        total_vram_used = sum(_model_vram(lm) for lm in engine.models.values())
        # Fleet GPU/VRAM + RAM summary so Ollama dashboards can show TOTAL GPU VRAM
        # (capacity across all GPU nodes). All byte counts, matching Ollama's convention.
        pool = {
            "vram_total": vram_total,
            "vram_used": int(total_vram_used),
            "vram_free": max(0, vram_total - int(total_vram_used)),
            "ram_total": int(sum(n.total_mem_gb for n in nodes) * GB),
            "usable_total": int(sum(n.usable_total_gb for n in nodes) * GB),
            "gpus": [{"name": n.hostname, "vram_total": int(n.vram_total_gb * GB),
                      "vram_used": int(n.vram_used_gb * GB)} for n in gpus],  # live (all users)
        }
        _alias_by_canon: dict[str, list] = {}     # canonical -> [alias keys] for the echo below
        for _a, _c in MODEL_ALIASES.items():
            _alias_by_canon.setdefault(_c, []).append(_a)
        models = []
        for lm in engine.models.values():
            entry = {
                "name": _ollama_name(lm.friendly), "model": _ollama_name(lm.friendly),
                "size": lm.spec.total_weight_bytes, "size_vram": _model_vram(lm),
                "digest": _digest(lm.target_id), "details": _details(lm.spec),
                # #idle-unload: honest Ollama expires_at — last activity + the idle window when
                # the knob is on (pinned models are exempt from idle-unload -> "forever");
                # idle_unload_m 0/off keeps the old effectively-never (+1y) value.
                "expires_at": _iso(_expires_at(lm)),
                "context_length": lm.ctx,   # Ollama-standard field (loaded context window)
                "infinitemodel": {
                    "ctx": lm.ctx, "pool_usable_gb": round(lm.plan.pool_usable_gb, 2),
                    "kv_quant": getattr(lm, "kv_quant", "none"),   # #172 TurboQuant KV preset
                    "stages": [{"host": s.hostname, "layers": [s.layer_start, s.layer_end],
                                "embed": s.has_embed, "head": s.has_head,
                                "est_gb": round(s.est_gb, 2)} for s in lm.plan.stages]},
            }
            models.append(entry)
            # ALIAS echo: also list the loaded model under any alias name (e.g. 'qwen2.5:14b' when
            # 'qwen2.5-14b-instruct' is loaded) so a client configured with the alias sees it running.
            # `alias_of` marks the echo rows as the SAME instance (not extra residents) — clients
            # counting real loaded instances filter on it; the admission cap never counted them.
            for _a in _alias_by_canon.get(lm.friendly, []):
                ae = dict(entry); ae["name"] = ae["model"] = _ollama_name(_a)
                ae["alias_of"] = _ollama_name(lm.friendly)
                models.append(ae)
        return {"models": models, "pool": pool}

    @app.post("/api/push")
    @app.post("/api/create")
    @app.post("/api/copy")
    async def api_manage() -> JSONResponse:
        return JSONResponse({"status": "not supported by InfiniteModel "
                             "(models are configured server-side)"}, status_code=501)

    @app.post("/api/generate")
    async def api_generate(req: Request):
        body = await req.json()
        return await _serve(body.get("model", ""), body.get("prompt", ""), None,
                            body, mode="generate", ip=_client_ip(req))

    @app.post("/api/chat")
    async def api_chat(req: Request):
        body = await req.json()
        return await _serve(body.get("model", ""), None, body.get("messages", []),
                            body, mode="chat", ip=_client_ip(req))

    @app.post("/v1/chat/completions")
    async def v1_chat(req: Request):
        body = await req.json()
        return await _serve(body.get("model", ""), None, body.get("messages", []),
                            body, mode="openai", ip=_client_ip(req))

    @app.post("/v1/completions")     # OpenAI legacy text-completion (prompt-based, like /api/generate)
    async def v1_completions(req: Request):
        body = await req.json()
        p = body.get("prompt", "")   # OpenAI allows a string OR an array of strings -> join
        if isinstance(p, list):
            p = "\n".join(x if isinstance(x, str) else str(x) for x in p)
        elif not isinstance(p, str):
            p = str(p)
        return await _serve(body.get("model", ""), p, None,
                            body, mode="openai_text", ip=_client_ip(req))

    # ---- Anthropic Messages API (Claude Code backend) ----
    @app.post("/v1/messages")
    async def v1_messages(req: Request):
        body = await req.json()
        return await _serve_anthropic(body, ip=_client_ip(req))

    @app.post("/v1/messages/count_tokens")
    async def v1_count_tokens(req: Request):
        body = await req.json()
        return await _count_tokens_anthropic(body)

    # ---- OpenAI-compatible Text-To-Speech (Kokoro TTS; Omni Talker fallback) ----
    @app.post("/v1/audio/speech")
    async def v1_audio_speech(req: Request):
        """OpenAI /v1/audio/speech: {model, input, voice, speed, response_format}. Speaks
        `input` and returns the raw audio bytes (wav | pcm). A Kokoro model routes to the
        dedicated single-node KokoroPipeline (#tts-serve); any other model — or none — falls
        through to the distributed Qwen2.5-Omni Talker path (choppy on that checkpoint; kept
        as a fallback). `voice` maps OpenAI names -> the target engine's speakers."""
        body = await req.json()
        ip = _client_ip(req)
        text = (body.get("input") or "").strip()
        if not text:
            return JSONResponse({"error": {"message": "'input' is required"}}, status_code=400)
        fmt = (body.get("response_format") or "wav").lower()
        # #tts-serve: if the requested model is a Kokoro speech checkpoint, route to the
        # single-node KokoroPipeline (engine.tts_generate). Anything else — or no model at
        # all — falls through to the distributed Qwen2.5-Omni speech path below.
        _req_model = (body.get("model") or "").strip()
        if _req_model:
            try:
                _kf = resolve_model_name(_req_model)
                _ktgt = MODELS[_kf][0] if _kf in MODELS else _kf
                _is_kok = bool(_is_kokoro_dir(_local_model_dir(_ktgt) or ""))
            except Exception:
                _kf, _is_kok = None, False
            if _is_kok:
                def _kvoice(v):
                    v = (v or "").strip()
                    if not v:
                        return "af_heart"
                    if "_" in v:                       # already a Kokoro voice id
                        return v
                    return {"alloy": "af_alloy", "echo": "am_echo", "fable": "bm_fable",
                            "onyx": "am_onyx", "nova": "af_nova", "shimmer": "af_bella",
                            "coral": "af_kore", "sage": "af_sarah", "ash": "am_adam",
                            "ballad": "bm_george", "verse": "am_michael"}.get(v.lower(), "af_heart")
                rec = _inflight_admit(ip, _kf)
                if rec is None:
                    return JSONResponse({"error": {"message": "server busy (speech queue full)"}},
                                        status_code=503)
                try:
                    try:
                        await engine.ensure_loaded(_kf, 0, auto_load=True)
                    except Exception as exc:
                        # #at-capacity (audit #32): was except->404 — a Kokoro auto-load refused
                        # for RAM/VRAM told the client the model doesn't exist. Typed ladder now.
                        return _media_load_error(exc, _kf, "tts")
                    try:
                        _speed = float(body.get("speed") or 1.0)
                    except Exception:
                        _speed = 1.0
                    _kv = _kvoice(body.get("voice"))
                    audio_bytes, _meta = await engine.tts_generate(
                        _kf, text, voice=_kv, speed=_speed, fmt=fmt)
                    _media = "audio/wav" if fmt in ("wav", "") else f"audio/{fmt}"
                    print(f"[v1/audio/speech] kokoro '{text[:40]}...' voice={_kv} "
                          f"-> {_meta.get('seconds')}s")
                    return Response(content=audio_bytes, media_type=_media)
                except Exception as exc:
                    import traceback
                    print(f"[v1/audio/speech kokoro] error: {exc!r}\n{traceback.format_exc()[-1000:]}")
                    return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}"}},
                                        status_code=500)
                finally:
                    _inflight_release(rec)
        try:
            friendly = resolve_model_name(body.get("model", "") or "qwen2.5-omni-7b")
        except Exception as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
        voice = body.get("voice") or "Chelsie"
        # admit (1 slot + queue) so concurrent TTS doesn't pile onto the CPU vocoder
        rec = _inflight_admit(ip, friendly)
        if rec is None:
            return JSONResponse({"error": {"message": "server busy (speech queue full)"}},
                                status_code=503)
        try:
            try:
                resident = engine.models.get(friendly)
                ctx = resident.ctx if resident else 0
                lm = await engine.ensure_loaded(friendly, ctx, auto_load=True)
                tok = lm.tokenizer
            except Exception as exc:
                # #at-capacity (audit #32, 4th site): the Omni speech auto-load had the same
                # blanket except->404 — capacity refusals masqueraded as model-not-found.
                return _media_load_error(exc, friendly, "speech")
            # resolve voice -> our speaker (load speech components to know available speakers)
            try:
                sc = await asyncio.to_thread(_load_speech_components, lm.target_id)
                speaker = _resolve_speaker(voice, sc["speaker_map"])
            except Exception as exc:
                return JSONResponse({"error": {"message": f"speech components: {exc}"}},
                                    status_code=500)
            # prompt the Omni to SPEAK the input verbatim (pure-TTS use of a chat speech model)
            sys_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                          "Group, capable of perceiving auditory and visual inputs, as well as "
                          "generating text and speech.")
            chat = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Read the following text aloud exactly as "
                     f"written, and say nothing else:\n\n{text}"}]
            ids = await asyncio.to_thread(   # #off-loop-tokenize: keep the event loop live
                lambda: _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                            tokenize=True)))
            # scale the text budget to the input length (verbatim ~ input length + margin)
            max_new = max(64, min(1024, int(len(_to_id_list(tok(text))) * 1.5) + 32))
            gen_ids, stop, wav, info = await engine.generate_speech(
                friendly, ids, max_new=max_new, speaker=speaker)
            print(f"[v1/audio/speech] '{text[:40]}...' voice={speaker} -> "
                  f"{info.get('wav_seconds')}s ({info.get('codec_tokens')} codes)")
            audio_bytes, media = await asyncio.to_thread(_encode_audio_response, wav, fmt)
            return Response(content=audio_bytes, media_type=media)
        except Exception as exc:
            import traceback
            print(f"[v1/audio/speech] error: {exc!r}\n{traceback.format_exc()[-1200:]}")
            return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}"}},
                                status_code=500)
        finally:
            _inflight_release(rec)

    @app.post("/v1/audio/music")
    async def v1_audio_music(req: Request):
        """#t2a-serve (M1): text-to-music. {model, prompt|input, lyrics, duration, steps,
        guidance, seed, response_format}. Renders `prompt` (genre/style tags; optional `lyrics`
        for vocals) through the single-node ACE-Step pipeline (engine.t2a_generate) and returns
        the WAV bytes. `model` must be a registered ACE-Step music checkpoint. If the model isn't
        resident it auto-loads GPU-resident (~11 GB); pre-load with t2i_offload=1 on tight cards."""
        body = await req.json()
        ip = _client_ip(req)
        prompt = (body.get("prompt") or body.get("input") or "").strip()
        if not prompt:
            return JSONResponse({"error": {"message": "'prompt' is required"}}, status_code=400)
        _req_model = (body.get("model") or "").strip()
        if not _req_model:
            return JSONResponse({"error": {"message": "'model' is required (an ACE-Step music model)"}},
                                status_code=400)
        try:
            friendly = resolve_model_name(_req_model)
        except Exception as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
        fmt = (body.get("response_format") or "wav").lower()

        def _f(key, default):
            try:
                return float(body.get(key, default))
            except Exception:
                return default

        def _i(key, default):
            try:
                return int(body.get(key, default))
            except Exception:
                return default

        rec = _inflight_admit(ip, friendly)
        if rec is None:
            return JSONResponse({"error": {"message": "server busy (music queue full)"}},
                                status_code=503)
        try:
            try:
                await engine.ensure_loaded(friendly, 0, auto_load=True)
            except Exception as exc:
                # #at-capacity (audit #32): THE observed failure — a t2a request routed to a
                # can_t2a node it couldn't fit (furnace acestep OOM) came back 404 "not found",
                # making retrying clients give up on a model that exists. Typed ladder now, and
                # the refusal finally lands in log_activity (render-oom-guard telemetry gap).
                return _media_load_error(exc, friendly, "t2a")
            audio_bytes, meta = await engine.t2a_generate(
                friendly, prompt, lyrics=(body.get("lyrics") or ""),
                duration=_f("duration", 30.0), steps=_i("steps", 60),
                guidance=_f("guidance", 15.0), seed=body.get("seed"), fmt=fmt)
            _media = "audio/wav" if fmt in ("wav", "") else f"audio/{fmt}"
            print(f"[v1/audio/music] '{prompt[:40]}...' {meta.get('audio_s')}s audio "
                  f"-> render {meta.get('seconds')}s")
            return Response(content=audio_bytes, media_type=_media)
        except Exception as exc:
            import traceback
            print(f"[v1/audio/music] error: {exc!r}\n{traceback.format_exc()[-1200:]}")
            return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}"}},
                                status_code=500)
        finally:
            _inflight_release(rec)

    # ---- Embeddings (code-split Inc 1): _serve_embed + the 3 embed routes ----
    # Bodies BYTE-IDENTICAL to their former server.py build_app originals; globals
    # (engine, resolve_model_name, _inflight_*, _client_tokens, ...) resolve via state.bind.
    async def _serve_embed(model: str, inputs, mode: str, ip: str = "?") -> JSONResponse:
        """Shared embedding serve for /api/embed, /api/embeddings (legacy) and /v1/embeddings.
        AUTO-LOADS a known-but-not-resident encoder (same policy as the generate paths, gated by
        the same ENGINE_CONFIG auto_load) — a cold embed request just works, and the #idle-unload
        knob reaps the encoder back off after the idle window like any other model. Tokenizes on
        the controller (NO chat template, NO task-prefix), runs one encoder forward on the node,
        and shapes the response per `mode` ('ollama' | 'legacy' | 'openai')."""
        try:
            friendly = resolve_model_name(model)
        except Exception:
            return _not_found_json(model, mode)   # unknown model -> 404 (OpenAI envelope|Ollama shape)
        try:
            lm = await engine.ensure_loaded(friendly, 0, auto_load=True)
        except ValueError as exc:   # not loaded AND auto-load off/updating -> 404
            return JSONResponse({"error": str(exc), "model": model}, status_code=404)
        except Exception as exc:    # auto-load FAILED (capacity/node) -> retryable 503, not a 500
            log_activity(f"embed {model}: auto-load failed — {exc!r}")
            # #at-capacity: terminal capacity (auto-unload off / all pinned) -> no Retry-After
            _term = isinstance(exc, CapacityError) and getattr(exc, "terminal", False)
            _out = {"error": f"embedding model load failed: {exc}", "model": model}
            if _term:
                _out["state"] = "at_capacity"
            return JSONResponse(_out, status_code=503,
                                headers=({} if _term else {"Retry-After": "3"}))
        if not getattr(lm.spec, "is_embedding", False):
            return JSONResponse(
                {"error": f"model '{friendly}' is not an embedding model; use /api/chat"},
                status_code=400)
        # Normalize inputs to list[str] (accept a string or a list of strings).
        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = [str(t) for t in inputs]
        else:
            return JSONResponse({"error": "input must be a string or a list of strings"},
                                status_code=400)
        if not texts:
            return JSONResponse({"error": "no input text provided"}, status_code=400)
        rec = _inflight_admit(ip, friendly, 1)
        if rec is None:
            return JSONResponse(
                {"error": f"queue full for '{friendly}' — retry shortly"}, status_code=503)
        try:
            _inflight_start(rec)
            tok = lm.tokenizer
            max_len = min(8192, int(getattr(lm.spec, "max_ctx", DEFAULT_CTX) or DEFAULT_CTX))
            enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            attn = enc["attention_mask"]
            vecs = await engine.embed(friendly, enc["input_ids"], attn)
        except Exception as exc:
            log_activity(f"embed {model}: FAILED — {exc!r}")
            print(f"[embed] {model} FAILED: {exc!r}", flush=True)
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "model": model},
                                status_code=500)
        finally:
            _inflight_release(rec)
        display = _ollama_name(friendly)
        n_tok = int(attn.sum())
        _client_tokens(ip, tok_in=n_tok, model=display)   # #connections: per-client token totals
        if mode == "openai":
            return JSONResponse({
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": v}
                         for i, v in enumerate(vecs)],
                "model": display,
                "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok}})
        if mode == "legacy":   # /api/embeddings -> single vector
            return JSONResponse({"embedding": vecs[0] if vecs else []})
        # /api/embed (Ollama)
        return JSONResponse({"model": display, "embeddings": vecs, "prompt_eval_count": n_tok})

    @app.post("/api/embed")
    async def api_embed(req: Request) -> JSONResponse:
        body = await req.json()
        inputs = body.get("input", body.get("prompt", body.get("text", "")))
        return await _serve_embed(body.get("model", ""), inputs, mode="ollama",
                                  ip=_client_ip(req))

    @app.post("/api/embeddings")   # legacy Ollama single-embedding endpoint
    async def api_embeddings(req: Request) -> JSONResponse:
        body = await req.json()
        inputs = body.get("prompt", body.get("input", ""))
        return await _serve_embed(body.get("model", ""), inputs, mode="legacy",
                                  ip=_client_ip(req))

    @app.post("/v1/embeddings")    # OpenAI-compatible
    async def v1_embeddings(req: Request) -> JSONResponse:
        body = await req.json()
        inputs = body.get("input", "")
        return await _serve_embed(body.get("model", ""), inputs, mode="openai",
                                  ip=_client_ip(req))

    # ---- Node tier config (code-split Inc 5): relocated from server.py build_app ----
    # Tier toggles are config, not downloads -- this existing zero-cost home keeps them
    # discoverable. Bodies BYTE-IDENTICAL; NODE_CONFIG loader is in-place as of Inc 4.
    @app.post("/nodeconfig")         # dashboard: enable/disable a node's CPU/RAM or GPU/VRAM
    async def nodeconfig(host: str, ram: Optional[bool] = None,
                         vram: Optional[bool] = None) -> JSONResponse:
        # Keyed by hostname so the choice sticks across reconnects; persisted to
        # node_config.json so it survives a controller restart. A tier change re-plans
        # ONLY the resident models that actually use a node on this host (surgical —
        # other models keep running; they'll pick up freed/added capacity on next load).
        cfg = NODE_CONFIG.setdefault(host, {"ram": True, "vram": True})
        if ram is not None:
            cfg["ram"] = ram
        if vram is not None:
            cfg["vram"] = vram
        save_node_config()
        host_nids = {nid for nid, n in registry._nodes.items() if n.hostname == host}
        for fr in [fr for fr, m in engine.models.items()
                   if any(nid in m.stage_node_ids for nid in host_nids)]:
            engine.invalidate_model(fr, f"tier change on {host}")
        return JSONResponse({"ok": True, "host": host, "config": cfg})

    @app.post("/nodeconfig_all")     # dashboard: enable/disable a tier on EVERY node at once
    async def nodeconfig_all(tier: str, enabled: bool) -> JSONResponse:
        """Bulk version of /nodeconfig: set one tier (ram|vram) for every known host, persist,
        and re-plan each resident model ONCE (fleet-wide capacity changed). Drives the
        'all CPU' / 'all GPU' master checkboxes."""
        if tier not in ("ram", "vram"):
            return JSONResponse({"ok": False, "error": "tier must be 'ram' or 'vram'"},
                                status_code=400)
        hosts = {n.hostname for n in registry.alive_sorted()} | set(NODE_CONFIG.keys())
        for h in hosts:
            NODE_CONFIG.setdefault(h, {"ram": True, "vram": True})[tier] = enabled
        save_node_config()
        for fr in list(engine.models.keys()):   # capacity changed everywhere -> re-plan all
            engine.invalidate_model(fr, f"bulk tier change ({tier}={enabled})")
        return JSONResponse({"ok": True, "tier": tier, "enabled": enabled, "hosts": len(hosts)})
