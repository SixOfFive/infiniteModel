"""routes_diag: routes relocated from server.py build_app (m4c153 code-split). Route bodies
are BYTE-IDENTICAL to the originals; their module globals (engine, registry, _serve,
build_status, JSONResponse …) are injected at startup by state.bind() — see state.py.
build_app() calls register(app) to attach them. Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def register(app):

    @app.post("/refresh_backends")
    async def refresh_backends() -> JSONResponse:
        """Re-probe the multimodal backend packages (Pillow / soundfile / torchvision / …) and bust
        transformers' import-time availability cache IN PROCESS. transformers latches is_vision_available
        / is_soundfile_available at import, so a `pip install pillow` done AFTER this controller started
        stays invisible (vision keeps ImportError-ing) until a full restart. Hit this route after
        installing a backend to make it live WITHOUT restarting im-controller. Returns {name: available}."""
        avail = await asyncio.to_thread(multimodal.refresh_multimodal_backends, True)
        return JSONResponse({"ok": True, "backends": avail})

    @app.get("/code_manifest")
    async def code_manifest(grep: str = "") -> JSONResponse:
        """Ground-truth of the self-update file set AS IT SITS ON THIS BOX'S DISK — sha1(12)/size/mtime
        per file, plus the running VERSION/CODE_DATE. The raw.githubusercontent CDN edge lags a push
        per-controller (a forced /update can pull a STALE file even when the CDN looks fresh from
        elsewhere), so a deploy must verify the bytes actually landed here rather than trusting the
        /update 200. This makes that check a single HTTP call instead of SSH+grep. Pass ?grep=<marker>
        to also report, per file, whether that substring is present on disk (the automated 'grep the
        marker on the box' step)."""
        def _run() -> dict:
            import re as _re
            here = os.path.dirname(os.path.abspath(__file__))
            files = ["server.py"] + list(EXTRA_UPDATE_FILES) + ["client.py"]
            # WORKER-side files too (E1): client.py keeps its own EXTRA_UPDATE_FILES, which this
            # controller-side route can't see via the module global above — and importing client.py
            # to read it would execute the worker's module-level hardware/triton probes. Regex the
            # list literal out of its source instead. Controller+worker share a checkout on the
            # fleet boxes, so the on-disk truth for worker files is exactly what a worker deploy
            # needs verified (worker increments bump no controller VERSION — this is their only
            # HTTP-visible ground truth).
            with contextlib.suppress(Exception):
                with open(os.path.join(here, "client.py"), "r", encoding="utf-8", errors="replace") as fh:
                    _src = fh.read()
                _m = _re.search(r"^EXTRA_UPDATE_FILES\s*:\s*list\[str\]\s*=\s*\[(.*?)\]",
                                _src, _re.DOTALL | _re.MULTILINE)
                if _m:
                    for _f in _re.findall(r'"([^"]+)"', _m.group(1)):
                        if _f not in files:
                            files.append(_f)
            out: dict = {}
            for fn in files:
                path = os.path.join(here, fn)
                try:
                    with open(path, "rb") as fh:
                        blob = fh.read()
                except Exception as exc:
                    out[fn] = {"present": False, "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
                    continue
                norm = blob.replace(b"\r\n", b"\n")   # match the self-updater's CRLF-normalized compare
                rec = {"present": True, "size": len(blob),
                       "sha1_12": hashlib.sha1(norm).hexdigest()[:12],
                       "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))}
                if fn in ("server.py", "client.py"):
                    rec["version"] = _extract_version(blob)
                if grep:
                    rec["grep_hit"] = grep.encode("utf-8", "replace") in norm
                out[fn] = rec
            return {"ok": True, "running_version": VERSION, "code_date": CODE_DATE,
                    "grep": grep or None, "files": out}
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/inspect_audio")       # #22 inc 5: introspect an Omni model's AUDIO + Thinker interface
    async def inspect_audio(model: str = "qwen2.5-omni-7b") -> JSONResponse:
        """Meta-load (zero weights) a Qwen2.5-Omni-style checkpoint and report what's needed to
        (a) LOAD its Thinker text model on the pipeline and (b) run audio input: get_text_config
        shape + whether AutoModelForCausalLM can build it with sliceable .model.layers; the audio
        tower class + get_audio_features signature; audio_token_id; the feature extractor."""
        def _run():
            out: dict = {"model": model}
            try:
                import torch, inspect as _inspect
                from transformers import AutoConfig
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                cfg = AutoConfig.from_pretrained(_local_model_dir(target) or target)
                out["config_class"] = type(cfg).__name__
                def g(o, *names):
                    for n in names:
                        v = getattr(o, n, None)
                        if v is not None:
                            return v
                    return None
                out["audio_token_id"] = g(cfg, "audio_token_id", "audio_token_index")
                out["audio_start_token_id"] = g(cfg, "audio_start_token_id")
                out["image_token_id"] = g(cfg, "image_token_id", "image_token_index")
                out["top_config_keys"] = [k for k in vars(cfg).keys() if not k.startswith("_")][:40]
                tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", None)
                out["has_get_text_config"] = hasattr(cfg, "get_text_config")
                if tcfg is not None:
                    out["text_config_class"] = type(tcfg).__name__
                    out["text_num_layers"] = g(tcfg, "num_hidden_layers")
                    out["text_hidden_size"] = g(tcfg, "hidden_size")
                    out["text_rope_scaling"] = getattr(tcfg, "rope_scaling", None)
                # can the worker build the Thinker text model from text_config?
                try:
                    from transformers import AutoModelForCausalLM
                    with torch.device("meta"):
                        tm = AutoModelForCausalLM.from_config(tcfg)
                    out["text_model_class"] = type(tm).__name__
                    inner = getattr(tm, "model", tm)
                    out["text_inner_children"] = [n for n, _ in inner.named_children()][:20]
                    layers = getattr(inner, "layers", None)
                    out["text_layers_count"] = (len(layers) if layers is not None else None)
                    out["text_buildable"] = layers is not None
                except Exception as exc:
                    out["text_build_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                # audio tower: build the full Omni/thinker on meta and inspect
                for auto_name in ("AutoModelForTextToWaveform", "AutoModel"):
                    try:
                        import transformers as _tf
                        AutoCls = getattr(_tf, auto_name, None)
                        if AutoCls is None:
                            continue
                        with torch.device("meta"):
                            m = AutoCls.from_config(cfg)
                        out["full_model_class"] = type(m).__name__
                        out["full_auto_used"] = auto_name
                        thinker = getattr(m, "thinker", m)
                        out["thinker_children"] = [n for n, _ in thinker.named_children()][:20]
                        at = getattr(thinker, "audio_tower", None)
                        if at is not None:
                            out["audio_tower_class"] = type(at).__name__
                            with contextlib.suppress(Exception):
                                out["audio_tower_forward_sig"] = str(_inspect.signature(at.forward))
                        out["has_get_audio_features"] = hasattr(thinker, "get_audio_features")
                        with contextlib.suppress(Exception):
                            out["get_audio_features_sig"] = str(_inspect.signature(thinker.get_audio_features))
                        break
                    except Exception as exc:
                        out[f"{auto_name}_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                # feature extractor / processor for audio
                try:
                    from transformers import AutoProcessor
                    proc = AutoProcessor.from_pretrained(_local_model_dir(target) or target)
                    out["processor_class"] = type(proc).__name__
                    fe = getattr(proc, "feature_extractor", None)
                    out["feature_extractor_class"] = type(fe).__name__ if fe is not None else None
                    out["proc_audio_token"] = getattr(proc, "audio_token", None)
                except Exception as exc:
                    out["processor_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/audio_test")          # #22 inc 5b: run the AUDIO encoder end-to-end on a test tone
    async def audio_test(model: str = "qwen2.5-omni-7b", secs: float = 2.0,
                         freq: float = 440.0, clips: int = 1) -> JSONResponse:
        """Synthesize sine tone(s) @16 kHz, run the feature extractor + Omni audio tower, and
        report shapes — verifies increment 5b (the encoder) against the real model with NO
        text-model load. clips>1 synthesizes that many DISTINCT-duration tones to exercise the
        MULTI-CLIP encode path (per-clip counts must sum to the flat embed-row count)."""
        def _run():
            out: dict = {"model": model}
            try:
                import math
                friendly = resolve_model_name(model) if model else (
                    next(iter(engine.models)) if engine.models else "qwen2.5-omni-7b")
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                sr = 16000
                import numpy as _np
                nclips = max(1, int(clips))
                tones = []
                for ci in range(nclips):
                    # distinct duration + pitch per clip so per-clip counts differ (a real
                    # multi-clip alignment test, not N identical clips).
                    dur = float(secs) * (1.0 + 0.5 * ci)
                    f = float(freq) * (1.0 + 0.25 * ci)
                    n = max(1, int(dur * sr))
                    t = _np.arange(n, dtype=_np.float32) / sr
                    tones.append((0.2 * _np.sin(2.0 * math.pi * f * t)).astype(_np.float32))
                out["clips"] = nclips
                out["clip_durations_s"] = [round(len(x) / sr, 2) for x in tones]
                r = _encode_audio(target, tones, sampling_rate=sr)
                emb = r["audio_embeds"]
                cts = r.get("counts")
                out["counts_sum_matches_embeds"] = bool(
                    cts is not None and sum(cts) == int(emb.shape[0]))
                out["audio_embeds_shape"] = list(emb.shape)
                out["audio_embeds_dtype"] = str(emb.dtype)
                out["audio_embeds_device"] = str(emb.device)
                out["counts"] = r.get("counts")
                out["audio_token_id"] = r.get("audio_token_id")
                out["out_hidden"] = r.get("out_hidden")
                out["encode_info"] = r.get("info")
                mat = _AUDIO_MAT.get(target, [])
                out["materialized_meta_count"] = len(mat)
                out["materialized_meta"] = [{"name": nm, "shape": s, "how": h}
                                            for nm, s, h in mat][:30]
                out["missing_weights"] = [nm for nm, s, h in mat if "MISSING_WEIGHT" in h]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1800:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/inspect_mm")          # #22: introspect a multimodal model's vision interface
    async def inspect_mm(model: str = "") -> JSONResponse:
        """Report the EXACT vision interface of a multimodal checkpoint (attribute path,
        module class, forward signature, processor + image token) so the distributed-Omni
        encoder path can be written against the real structure. Meta-load only (no weights,
        no inference) — safe + cheap. Used to build #22 increment 2 (the vision encoder)."""
        def _run():
            out: dict = {"model": model}
            try:
                friendly = resolve_model_name(model) if model else (
                    next(iter(engine.models)) if engine.models else "")
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                import inspect as _inspect
                from transformers import AutoConfig, AutoProcessor
                cfg = AutoConfig.from_pretrained(target)
                out["model_type"] = getattr(cfg, "model_type", None)
                out["architectures"] = getattr(cfg, "architectures", None)
                vc = getattr(cfg, "vision_config", None)
                out["vision_config_keys"] = sorted(vars(vc).keys()) if vc is not None else None
                try:
                    proc = AutoProcessor.from_pretrained(target)
                    out["processor_class"] = type(proc).__name__
                    for attr in ("image_token", "image_token_id", "image_processor",
                                 "video_token", "audio_token"):
                        v = getattr(proc, attr, None)
                        out[f"proc_{attr}"] = type(v).__name__ if attr == "image_processor" else v
                except Exception as exc:
                    out["processor_error"] = f"{type(exc).__name__}: {exc}"
                # meta-load the full multimodal model (zero memory) to inspect structure
                try:
                    import torch
                    from transformers import AutoModelForImageTextToText
                    with torch.device("meta"):
                        m = AutoModelForImageTextToText.from_config(cfg)
                    out["model_class"] = type(m).__name__
                    out["top_children"] = [n for n, _ in m.named_children()]
                    inner = getattr(m, "model", m)
                    out["inner_children"] = [n for n, _ in inner.named_children()]
                    vis = None
                    for path in ("visual", "vision_tower", "vision_model"):
                        vis = getattr(inner, path, None) or getattr(m, path, None)
                        if vis is not None:
                            out["vision_attr"] = path
                            break
                    if vis is not None:
                        out["vision_class"] = type(vis).__name__
                        with contextlib.suppress(Exception):
                            out["vision_forward_sig"] = str(_inspect.signature(vis.forward))
                        with contextlib.suppress(Exception):
                            out["vision_children"] = [n for n, _ in vis.named_children()]
                    # how does the model splice image features? find the method names
                    out["mm_methods"] = [n for n in dir(m)
                                         if any(k in n.lower() for k in
                                                ("image", "visual", "merge", "multimodal", "rope", "position"))][:40]
                except Exception as exc:
                    out["model_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/vision_log")          # crash-surviving phase log of the last vision encode
    async def vision_log(lines: int = 60) -> JSONResponse:
        """Return the tail of vision_diag.log. Because the encode has hard-crash-restarted
        the controller, this file (flushed+fsync'd per phase) is the only way to see which
        step ran last BEFORE a fatal native fault — read it AFTER the relaunch."""
        def _run():
            try:
                with open(_VISION_LOG, encoding="utf-8") as fh:
                    tail = fh.read().splitlines()[-max(1, min(lines, 500)):]
                return {"log": tail, "path": _VISION_LOG}
            except FileNotFoundError:
                return {"log": [], "note": "no vision_diag.log yet"}
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/vision_test")         # #22 inc 2: run the vision encoder end-to-end on a test image
    async def vision_test(model: str = "") -> JSONResponse:
        """Generate a small test image, run the processor + vision tower, and report shapes —
        so increment 2 (the encoder) is verified against the real model before wiring it into
        the pipeline. No text-model load, no inference on the LM."""
        def _run():
            out: dict = {"model": model}
            try:
                from PIL import Image
                friendly = resolve_model_name(model) if model else (
                    next(iter(engine.models)) if engine.models else "qwen3.6-35b-a3b")
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                img = Image.new("RGB", (336, 336), (90, 140, 200))
                r = _encode_images(target, [img])
                emb = r["image_embeds"]
                out["image_embeds_shape"] = list(emb.shape)
                out["image_embeds_dtype"] = str(emb.dtype)
                out["image_embeds_device"] = str(emb.device)
                out["grid_thw"] = (r["grid_thw"].tolist() if r["grid_thw"] is not None else None)
                out["encode_info"] = r.get("info")
                mat = _VISION_MAT.get(target, [])
                out["materialized_meta_count"] = len(mat)
                out["materialized_meta"] = [{"name": n, "shape": s, "how": h} for n, s, h in mat][:30]
                out["missing_weights"] = [n for n, s, h in mat if "MISSING_WEIGHT" in h]
                # --- merger diagnostics (STRUCTURE ONLY, no extra forward) ---
                # get_image_features gave the pre-merge ViT backbone [patches, vision_hidden];
                # the LM consumes MERGED tokens [prod(grid)/merge^2, text_hidden]. Report the
                # config + submodule tree (cheap, cached meta-load) to design the merge step.
                try:
                    # NOTE: must NOT bind the name `model` here — `model` is the endpoint
                    # parameter referenced at the top of _run ({"model": model}); assigning it
                    # anywhere in _run makes it function-local and UnboundLocalErrors line 1.
                    vmodel, dev = _load_vision_encoder(target)
                    vcfg = getattr(vmodel.config, "vision_config", None)
                    tcfg = getattr(vmodel.config, "text_config", vmodel.config)
                    def _g(o, *names):
                        for n in names:
                            v = getattr(o, n, None)
                            if v is not None:
                                return v
                        return None
                    sm = (_g(vcfg, "spatial_merge_size") or 1) if vcfg is not None else 1
                    out["vision_cfg"] = None if vcfg is None else {
                        "hidden_size": _g(vcfg, "hidden_size"),
                        "out_hidden_size": _g(vcfg, "out_hidden_size", "output_hidden_size"),
                        "spatial_merge_size": sm,
                    }
                    out["text_hidden_size"] = _g(tcfg, "hidden_size")
                    out["image_token_id"] = _g(vmodel.config, "image_token_id", "image_token_index")
                    g = r["grid_thw"]
                    if g is not None:
                        out["expected_merged_tokens"] = int(g.prod().item()) // (sm * sm)
                    vis = vmodel.model.visual
                    out["visual_children"] = [n for n, _ in vis.named_children()]
                    merger = getattr(vis, "merger", None)
                    if merger is not None:
                        out["merger_class"] = type(merger).__name__
                        out["merger_children"] = [n for n, _ in merger.named_children()]
                        fc1 = getattr(merger, "linear_fc1", None)
                        fc2 = getattr(merger, "linear_fc2", None)
                        if fc1 is not None:
                            out["merger_fc1"] = [getattr(fc1, "in_features", None),
                                                 getattr(fc1, "out_features", None)]
                        if fc2 is not None:
                            out["merger_fc2"] = [getattr(fc2, "in_features", None),
                                                 getattr(fc2, "out_features", None)]
                    # Re-run visual() and dump EVERY tensor field of the return + try the
                    # merger explicitly — to find where the merged [100,2048] actually is.
                    import torch
                    ipd = _get_image_processor(target)
                    inpd = ipd(images=[img], return_tensors="pt")
                    pvd = inpd["pixel_values"].to(dev)
                    gdd = inpd.get("image_grid_thw")
                    gdd = gdd.to(dev) if gdd is not None else None
                    with torch.inference_mode():
                        raw = vis(pvd, gdd)
                    fields = {}
                    if isinstance(raw, torch.Tensor):
                        fields["<tensor>"] = list(raw.shape)
                    elif hasattr(raw, "keys"):
                        for k in raw.keys():
                            v = raw[k]
                            if hasattr(v, "shape"):
                                fields[k] = list(v.shape)
                    out["visual_return_fields"] = fields
                    if merger is not None:
                        lhs = _as_feature_tensor(raw)
                        try:
                            with torch.inference_mode():
                                merged = merger(lhs)
                            out["merger_direct_shape"] = list(_as_feature_tensor(merged).shape)
                        except Exception as me:
                            out["merger_direct_error"] = f"{type(me).__name__}: {str(me)[:200]}"
                except Exception as exc:
                    import traceback
                    out["merger_diag_error"] = f"{type(exc).__name__}: {exc}"
                    out["merger_diag_trace"] = traceback.format_exc()[-600:]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-2000:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/mm_inject_test")      # #22 inc 3: prove embed-injection changes the computation
    async def mm_inject_test(model: str = "qwen2.5-0.5b", positions: str = "1,2") -> JSONResponse:
        """Load a (small) model, run a baseline prefill and an identical prefill with RANDOM
        embeds spliced in at `positions`, and report whether the next-token logits differ.
        Same token ids both times -> any change PROVES the mm frame reached stage 0, was
        stashed, and the splice replaced those positions' embeddings before the layers ran.
        Pure mechanism check (random embeds, not a real image) on the live pipeline."""
        out: dict = {"model": model}
        try:
            import torch
            friendly = resolve_model_name(model)
            out["friendly"] = friendly
            lm = await engine.ensure_loaded(friendly, 0)
            ids = _to_id_list(lm.tokenizer("The capital of France is"))
            pos = [int(p) for p in positions.split(",") if p.strip() != ""]
            pos = [p for p in pos if 0 <= p < len(ids)]
            out["prompt_len"] = len(ids)
            out["inject_positions"] = pos
            # hidden size from the model config (text hidden)
            from transformers import AutoConfig
            target = MODELS[friendly][0] if friendly in MODELS else friendly
            mcfg = await asyncio.to_thread(AutoConfig.from_pretrained,
                                           _local_model_dir(target) or target)
            tcfg = getattr(mcfg, "text_config", mcfg)
            hid = int(getattr(tcfg, "hidden_size", 0) or getattr(mcfg, "hidden_size"))
            out["hidden_size"] = hid
            xt = torch.tensor([ids])
            # #kv-slots: lease a slot like a real generation — on a kv_slots>1 replica live
            # gens hold slot leases, not lm.lock, so locking lm.lock would exclude nothing and
            # these reset prefills on slot 0 would wipe a running gen's KV mid-decode.
            # _SlotLease degenerates to exactly ``async with lm.lock`` on every C=1 model.
            from engine_gen import _SlotLease
            async with _SlotLease(lm) as _slot:
                base = await engine._send(lm, xt, 0, True, False, slot=_slot)
                emb = torch.randn(len(pos), hid, dtype=torch.bfloat16)
                inj = await engine._send(lm, xt, 0, True, False, mm=(pos, emb), slot=_slot)
            ba = torch.as_tensor(base).float()
            ia = torch.as_tensor(inj).float()
            out["baseline_argmax"] = int(ba.argmax())
            out["injected_argmax"] = int(ia.argmax())
            out["logits_changed"] = bool(not torch.allclose(ba, ia))
            out["max_abs_delta"] = float((ba - ia).abs().max())
            out["ok"] = True
        except Exception as exc:
            import traceback
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["trace"] = traceback.format_exc()[-1500:]
        return JSONResponse(out)

    @app.get("/vision_prompt_test")  # #22 inc 3b: verify prompt-build (no heavy text-model load)
    async def vision_prompt_test(model: str = "qwen3.6-35b-a3b") -> JSONResponse:
        """End-to-end check of the IMAGE PROMPT construction without loading the text LM:
        build a test-image Anthropic message -> keep_images chat -> render (one <|image_pad|>)
        -> encode the image -> expand placeholders -> confirm positions align with embeds.
        Uses only the tokenizer + cached vision encoder (cheap)."""
        def _run():
            out: dict = {"model": model}
            try:
                import io, base64
                from PIL import Image
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                buf = io.BytesIO()
                Image.new("RGB", (336, 336), (200, 60, 60)).save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                messages = [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": "What color is this image?"}]}]
                images = _collect_images(messages)
                out["num_images"] = len(images)
                chat = _anthropic_messages_to_chat(None, messages, keep_images=True)
                enc = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
                ids = _to_id_list(enc)
                enc_res = _encode_images(target, images)
                embeds = enc_res.get("image_embeds")
                counts = enc_res.get("counts") or []
                itid = enc_res.get("image_token_id")
                out["image_token_id"] = itid
                out["counts"] = counts
                out["embeds_shape"] = list(embeds.shape) if embeds is not None else None
                out["raw_image_pad_in_ids"] = sum(1 for t in ids if t == itid)
                new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts)
                out["placeholders_found"] = found
                out["num_positions"] = len(positions)
                out["prompt_len_before"] = len(ids)
                out["prompt_len_after"] = len(new_ids)
                out["positions_head"] = positions[:5]
                out["aligned"] = bool(found == len(counts)
                                      and len(positions) == (embeds.shape[0] if embeds is not None else -1))
                # #22 inc 4: also compute the 3D mRoPE positions and report base + samples
                merge = int(enc_res.get("merge") or 1)
                grid_list = enc_res.get("grid_list") or []
                pos3d, base = _mrope_position_ids(new_ids, grid_list, int(itid), merge)
                ip0 = positions[0] if positions else 0
                out["mrope_base"] = base
                out["mrope_seq_len"] = len(pos3d[0])
                out["mrope_delta"] = base - len(new_ids)
                out["mrope_text_head_thw"] = [row[:3] for row in pos3d]
                out["mrope_image_thw"] = [row[ip0:ip0 + 3] for row in pos3d]
                out["mrope_tail_thw"] = [row[-2:] for row in pos3d]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/speech_capture_test")  # #P6: verify the hidden-states transport (Phase 1)
    async def speech_capture_test(model: str = "qwen2.5-omni-7b", max_new: int = 8) -> JSONResponse:
        """Phase-1 check of the distributed Thinker HIDDEN-STATE capture for speech-out: run a
        short text generation with capture_hidden=True and report the prefill hidden shape +
        each step's hidden shape + the decoded text. Requires the model loaded; no talker yet."""
        out: dict = {"model": model}
        try:
            friendly = resolve_model_name(model)
            if friendly not in engine.models:
                return JSONResponse({"ok": False, "error": f"{friendly} not loaded"},
                                    status_code=409)
            lm = engine.models[friendly]
            # #kv-slots: speech is a C=1 (Omni-only) doctrine — capture_thinker runs LOCK-FREE on
            # slot 0 with NO _SlotLease and threads NO slot id (see engine_gen.capture_thinker,
            # #idle-unload). The production speech path only ever loads Omni C=1, but this
            # diagnostic accepts an ARBITRARY loaded model; on a kv_slots>1 replica its reset
            # prefills on slot 0 would stomp a live slot-0 generation's KV (length/mask crash or
            # corrupted output) — the same slot-isolation class the probe fixes closed. Refuse a
            # multi-slot model here rather than lease inside the lock-free production path.
            _C = int(getattr(lm, "kv_slots", 1) or 1)
            if _C > 1:
                return JSONResponse(
                    {"ok": False, "error": f"{friendly} has kv_slots={_C}; speech capture is "
                     f"C=1 only (capture_thinker runs lock-free on slot 0 and would stomp a "
                     f"live slot's KV). Load an Omni model or a C=1 replica."},
                    status_code=409)
            tok = lm.tokenizer
            sys_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                          "Group, capable of perceiving auditory and visual inputs, as well as "
                          "generating text and speech.")
            chat = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Say hello in one short sentence."}]
            ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                      tokenize=True))
            out["prompt_len"] = len(ids)
            gen_ids, prefill_hidden, step_hiddens, stop = await engine.capture_thinker(
                friendly, ids, int(max_new))
            out["prefill_hidden_shape"] = list(prefill_hidden.shape)
            out["prefill_hidden_dtype"] = str(prefill_hidden.dtype)
            out["num_step_hiddens"] = len(step_hiddens)
            out["step_hidden_shape"] = (list(step_hiddens[0].shape) if step_hiddens else None)
            out["num_gen_ids"] = len(gen_ids)
            out["stop"] = stop
            with contextlib.suppress(Exception):
                out["text"] = _safe_decode(tok, gen_ids)
            # the prefill hidden must cover every prompt token; step hiddens are 1 token each
            out["prefill_covers_prompt"] = bool(prefill_hidden.shape[1] == len(ids))
            out["ok"] = True
        except Exception as exc:
            import traceback
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["trace"] = traceback.format_exc()[-1500:]
        return JSONResponse(out)

    @app.get("/speech_components_test")  # #P6 Phase 2: load + report the talker + token2wav
    async def speech_components_test(model: str = "qwen2.5-omni-7b") -> JSONResponse:
        """Phase-2 check: meta-build the full Omni and materialize the talker + token2wav + the
        thinker embed matrix + spk_dict, then report dims / codec tokens / speakers / missing
        weights. No thinker load (it's distributed); no generation yet."""
        def _run():
            out: dict = {"model": model}
            try:
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                sc = _load_speech_components(target)
                talker, token2wav = sc["talker"], sc["token2wav"]
                tcfg = talker.config
                out["dev"] = sc["dev"]
                out["speakers"] = list(sc["speaker_map"].keys())
                out["n_talker_tensors"] = sc["n_talker"]
                out["n_token2wav_tensors"] = sc["n_token2wav"]
                out["n_embed_tensors"] = sc["n_embed"]
                out["talker_cfg"] = {
                    "num_hidden_layers": getattr(tcfg, "num_hidden_layers", None),
                    "hidden_size": getattr(tcfg, "hidden_size", None),
                    "embedding_size": getattr(tcfg, "embedding_size", None),
                    "vocab_size": getattr(tcfg, "vocab_size", None),
                }
                out["codec_tokens"] = {
                    "bos": talker.codec_bos_token, "eos": talker.codec_eos_token,
                    "pad": talker.codec_pad_token, "mask": talker.codec_mask_token,
                    "text_bos": talker.text_bos_token, "text_eos": talker.text_eos_token,
                    "text_pad": talker.text_pad_token,
                }
                out["token2wav_dtype"] = str(next(token2wav.parameters()).dtype)
                # a speaker entry's keys (cond / ref_mel / bos_token) — what token2wav needs
                spk0 = sc["speaker_map"].get(out["speakers"][0]) if out["speakers"] else None
                if isinstance(spk0, dict):
                    out["speaker_keys"] = list(spk0.keys())
                out["load_report"] = _SPEECH_MAT.get(target, {})
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1800:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/speech_test")         # #P6 Phase 3/4: end-to-end text -> speech (returns WAV b64)
    async def speech_test(model: str = "qwen2.5-omni-7b", speaker: str = "Chelsie",
                          max_new: int = 64, talker_max_new: int = 2048,
                          text: str = "Say, word for word: the quick brown fox jumps over the lazy dog.") -> JSONResponse:
        """Phase 3/4 end-to-end: distributed Thinker (hidden-state capture) -> Talker -> token2wav
        -> 24kHz waveform, returned as base64 PCM16 WAV. Requires the Omni model loaded."""
        out: dict = {"model": model, "speaker": speaker}
        try:
            friendly = resolve_model_name(model)
            if friendly not in engine.models:
                return JSONResponse({"ok": False, "error": f"{friendly} not loaded"},
                                    status_code=409)
            tok = engine.models[friendly].tokenizer
            sys_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                          "Group, capable of perceiving auditory and visual inputs, as well as "
                          "generating text and speech.")
            chat = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}]
            ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                      tokenize=True))
            gen_ids, stop, wav, info = await engine.generate_speech(
                friendly, ids, max_new=int(max_new), speaker=speaker,
                talker_max_new=int(talker_max_new))
            with contextlib.suppress(Exception):
                out["text"] = _safe_decode(tok, gen_ids)
            out["info"] = info
            # encode waveform -> PCM16 WAV -> base64
            def _wav_b64():
                import io, wave, base64
                import numpy as _np
                a = wav.detach().cpu().numpy()
                a = _np.clip(a, -1.0, 1.0)
                pcm = (a * 32767.0).astype(_np.int16)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                    w.writeframes(pcm.tobytes())
                return base64.b64encode(buf.getvalue()).decode()
            out["wav_b64"] = await asyncio.to_thread(_wav_b64)
            out["ok"] = True
        except Exception as exc:
            import traceback
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["trace"] = traceback.format_exc()[-2000:]
        return JSONResponse(out)

    @app.get("/audio_prompt_test")   # #22 inc 5c: verify the AUDIO prompt build (no text-model load)
    async def audio_prompt_test(model: str = "qwen2.5-omni-7b", secs: float = 2.0) -> JSONResponse:
        """End-to-end check of the AUDIO PROMPT construction without loading the text LM:
        synth a tone -> WAV -> input_audio message -> keep_audio chat -> render (one <|AUDIO|>)
        -> encode the audio -> expand the placeholder to its token count -> confirm positions
        align with embeds + the sequential TMRoPE covers the expanded prompt. Tokenizer +
        cached audio encoder only (cheap)."""
        def _run():
            out: dict = {"model": model}
            try:
                import io, base64, math, wave
                import numpy as _np
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                # synth a 16 kHz mono PCM16 WAV tone
                sr = 16000
                n = max(1, int(secs * sr))
                t = _np.arange(n, dtype=_np.float32) / sr
                pcm = (0.2 * _np.sin(2 * math.pi * 440.0 * t) * 32767).astype(_np.int16)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes(pcm.tobytes())
                b64 = base64.b64encode(buf.getvalue()).decode()
                messages = [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": "What do you hear?"}]}]
                auds = _collect_audio(messages)
                out["num_audio"] = len(auds)
                out["waveform_len"] = (int(len(auds[0])) if auds else 0)
                chat = _anthropic_messages_to_chat(None, messages, keep_audio=True)
                out["chat"] = chat
                try:
                    ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                                                              tokenize=True))
                    out["template"] = "ok"
                except Exception as exc:
                    out["template_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    # fall back: insert the audio markers as raw text so the rest still verifies
                    raw = "<|audio_bos|><|AUDIO|><|audio_eos|>What do you hear?"
                    ids = _to_id_list(tok(raw)["input_ids"] if hasattr(tok(raw), "get")
                                      else tok(raw))
                    out["template"] = "fallback_raw_text"
                enc_res = _encode_audio(target, auds)
                embeds = enc_res.get("audio_embeds")
                counts = enc_res.get("counts") or []
                atid = enc_res.get("audio_token_id")
                out["audio_token_id"] = atid
                out["counts"] = counts
                out["embeds_shape"] = list(embeds.shape) if embeds is not None else None
                out["raw_audio_tok_in_ids"] = (sum(1 for x in ids if x == atid) if atid is not None else None)
                new_ids, positions, found = _expand_image_placeholders(ids, int(atid), counts)
                out["placeholders_found"] = found
                out["num_positions"] = len(positions)
                out["prompt_len_before"] = len(ids)
                out["prompt_len_after"] = len(new_ids)
                out["positions_head"] = positions[:5]
                out["aligned"] = bool(found == len(counts) and embeds is not None
                                      and len(positions) == embeds.shape[0])
                pos3d, base = _audio_position_ids(len(new_ids))
                out["tmrope_base"] = base
                out["tmrope_seq_len"] = len(pos3d[0])
                out["tmrope_head"] = [row[:3] for row in pos3d]
                out["tmrope_tail"] = [row[-3:] for row in pos3d]
                # decode a window around the first audio position to confirm bos/eos framing
                if positions:
                    p0 = positions[0]
                    with contextlib.suppress(Exception):
                        out["decoded_around_audio"] = tok.decode(new_ids[max(0, p0 - 2):p0 + 2]
                                                                 + new_ids[p0 + len(positions):p0 + len(positions) + 2])
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/rope_probe")          # #22 inc 4: can we compute 3D mRoPE positions on the controller?
    async def rope_probe(model: str = "qwen3.6-35b-a3b") -> JSONResponse:
        """Confirm the mRoPE plan: (1) the model exposes get_rope_index so we can compute the
        correct 3D (t/h/w) position ids for an image prompt on the controller (index math, no
        weights), and (2) the TEXT config carries an mrope rope_scaling section so the worker's
        rotary expects 3D positions. Builds a test-image prompt and runs get_rope_index."""
        def _run():
            out: dict = {"model": model}
            try:
                import io, base64, torch
                from PIL import Image
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                buf = io.BytesIO(); Image.new("RGB", (336, 336), (60, 160, 90)).save(buf, "PNG")
                msgs = [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": "describe"}]}]
                images = _collect_images(msgs)
                chat = _anthropic_messages_to_chat(None, msgs, keep_images=True)
                ids = _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
                enc_res = _encode_images(target, images)
                counts = enc_res.get("counts") or []
                itid = int(enc_res.get("image_token_id"))
                new_ids, positions, _ = _expand_image_placeholders(ids, itid, counts)
                grid = enc_res.get("grid_thw")
                model_obj, _dev = _load_vision_encoder(target)
                cfg = model_obj.config
                tcfg = getattr(cfg, "text_config", cfg)
                out["has_get_rope_index"] = hasattr(model_obj, "get_rope_index")
                out["text_rope_scaling"] = getattr(tcfg, "rope_scaling", None)
                out["rope_scaling_top"] = getattr(cfg, "rope_scaling", None)
                out["seq_len"] = len(new_ids)
                if out["has_get_rope_index"]:
                    with torch.inference_mode():
                        res = model_obj.get_rope_index(
                            torch.tensor([new_ids]), image_grid_thw=grid)
                    pos = res[0] if isinstance(res, (tuple, list)) else res
                    out["rope_index_return_type"] = type(res).__name__
                    out["position_ids_shape"] = list(pos.shape)
                    # show the 3 dims around the image region (positions[0..]) to eyeball mRoPE
                    p = pos[:, 0, :].tolist() if pos.dim() == 3 else pos.tolist()
                    img0 = positions[0] if positions else 0
                    out["pos_sample_text_head"] = [row[:4] for row in p]
                    out["pos_sample_image"] = [row[img0:img0 + 4] for row in p]
                    out["pos_sample_tail"] = [row[-3:] for row in p]
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-1500:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))

    @app.get("/tok_probe")           # #22 inc 3b: how does THIS tokenizer represent image tokens?
    async def tok_probe(model: str = "qwen3.6-35b-a3b") -> JSONResponse:
        """Cheap tokenizer-only probe (no model load) to decide how to build the prompt ids
        with image placeholders: len(tokenizer) vs image_token_id, whether the image_pad
        STRING round-trips through the text tokenizer, the vision_start/end ids, and whether
        the chat template renders an image content block."""
        def _run():
            out: dict = {"model": model}
            try:
                friendly = resolve_model_name(model)
                target = MODELS[friendly][0] if friendly in MODELS else friendly
                out["target"] = target
                tok = _get_tokenizer(target)
                out["len_tokenizer"] = len(tok)
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(_local_model_dir(target) or target)
                def g(o, *names):
                    for k in names:
                        v = getattr(o, k, None)
                        if v is not None:
                            return v
                    return None
                out["image_token_id"] = g(cfg, "image_token_id", "image_token_index")
                out["vision_start_token_id"] = g(cfg, "vision_start_token_id")
                out["vision_end_token_id"] = g(cfg, "vision_end_token_id")
                for s in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
                    try:
                        out[f"encode {s}"] = tok.encode(s, add_special_tokens=False)
                    except Exception as e:
                        out[f"encode {s} ERR"] = str(e)[:100]
                itid = out.get("image_token_id")
                if itid is not None:
                    with contextlib.suppress(Exception):
                        out["convert_ids_to_tokens(image_token_id)"] = tok.convert_ids_to_tokens(int(itid))
                try:
                    msgs = [{"role": "user", "content": [{"type": "image"},
                                                         {"type": "text", "text": "hi"}]}]
                    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                       tokenize=False)
                    out["template_has_image_pad"] = "<|image_pad|>" in rendered
                    out["chat_template_tail"] = rendered[-260:]
                except Exception as e:
                    out["chat_template_err"] = f"{type(e).__name__}: {str(e)[:150]}"
                out["ok"] = True
            except Exception as exc:
                import traceback
                out["error"] = f"{type(exc).__name__}: {exc}"
                out["trace"] = traceback.format_exc()[-800:]
            return out
        return JSONResponse(await asyncio.to_thread(_run))
