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
        if d and spec:
            spec = spec_with_measurements(spec, d)
        caps = (["embedding"] if (spec and getattr(spec, "is_embedding", False))
                else ["completion", "chat"])
        return JSONResponse({
            "license": "see model card", "modelfile": f"# InfiniteModel distributed\nFROM {target}",
            "parameters": "", "template": "{{ .Prompt }}",
            "details": _details(spec), "model_info": _model_info(spec),
            "capabilities": caps,
            "infinitemodel": {"target": target, "draft": MODELS[friendly][1],
                              "distributed": True, "engine": VERSION},
        })

    @app.get("/api/ps")
    async def api_ps() -> dict:
        nodes = registry.alive_sorted()
        gpus = [n for n in nodes if n.vram_total_gb > 0]
        vram_total = int(sum(n.vram_total_gb for n in gpus) * GB)

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
                "expires_at": _iso(time.time() + 365 * 86400),
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
            for _a in _alias_by_canon.get(lm.friendly, []):
                ae = dict(entry); ae["name"] = ae["model"] = _ollama_name(_a)
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

    # ---- OpenAI-compatible Text-To-Speech (distributed Qwen2.5-Omni speech-out) ----
    @app.post("/v1/audio/speech")
    async def v1_audio_speech(req: Request):
        """OpenAI /v1/audio/speech: {model, input, voice, response_format}. Speaks `input`
        through the distributed Omni speech pipeline and returns the raw audio bytes
        (wav | pcm). `voice` maps OpenAI names -> our speakers (Chelsie/Ethan)."""
        body = await req.json()
        ip = _client_ip(req)
        text = (body.get("input") or "").strip()
        if not text:
            return JSONResponse({"error": {"message": "'input' is required"}}, status_code=400)
        try:
            friendly = resolve_model_name(body.get("model", "") or "qwen2.5-omni-7b")
        except Exception as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
        fmt = (body.get("response_format") or "wav").lower()
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
                return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
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
            ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                      tokenize=True))
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
