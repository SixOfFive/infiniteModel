"""serving_anthropic: the Anthropic Messages API engine (/v1/messages, the Claude Code
backend), relocated VERBATIM from serving.py (code-split Inc 3): _serve_anthropic +
_count_tokens_anthropic. This is where the vision/audio serve-path edit traffic lands
(qwen2.5-VL, gemma-4 vision/audio, pixtral IMG_BREAK are all /v1/messages-only). Bodies
BYTE-IDENTICAL to the originals. Module globals (engine, registry, resolve_model_name,
METRICS, formats helpers, _encode_images, ...) are injected at startup by state.bind() --
see state.py. The serve-layer helpers shared with the Ollama/OpenAI engine are imported
leaf-to-leaf from serving below (a leaf never imports server; serving is a sibling leaf,
synced in the same EXTRA_UPDATE_FILES set). Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

# leaf->leaf: shared serve-layer helpers that STAY in serving.py (both protocol engines use them)
from serving import (_coerce_knobs, _pixtral_break_end, _stream_fail, _transient_gen_exc, _wrap_image_runs)   # noqa: F401


async def _with_keepalive(agen, idle_s: float = 10.0):
    """Wrap a (piece, reason) token generator so a silent stretch yields a (None, None)
    keepalive tick instead of dead air. #prefill-keepalive: a Claude-Code-sized system
    prompt (10-20k tokens) prefills for 30-120s on an APU-class box, during which the
    SSE stream otherwise emits NOTHING after message_start — harness inactivity timeouts
    abort the healthy stream and re-send, re-prefilling forever ("no output" + a retry
    storm). The pending __anext__ future survives across ticks (asyncio.wait, no
    wait_for-style cancel), so generation is never disturbed."""
    it = agen.__aiter__()
    nt = None
    try:
        while True:
            if nt is None:
                nt = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({nt}, timeout=idle_s)
            if not done:
                yield None, None
                continue
            try:
                item = nt.result()
            except StopAsyncIteration:
                return
            finally:
                nt = None
            yield item
    finally:
        if nt is not None:
            nt.cancel()


async def _serve_anthropic(body: dict, ip: str = "?"):
    """POST /v1/messages — the Anthropic Messages API, so Claude Code (and any
    Anthropic SDK client) can drive the distributed fleet. Translates the Anthropic
    request into the model's chat template (tools included), runs the same decode
    path, and renders either a single JSON message or the Anthropic SSE event
    stream. Qwen <tool_call>{...}</tool_call> output is mapped to tool_use blocks."""
    METRICS["api_in"] += len(json.dumps(body))
    model = body.get("model", "")
    try:
        friendly = resolve_model_name(model)
    except Exception as exc:
        return JSONResponse({"type": "error",
                             "error": {"type": "not_found_error", "message": str(exc)}},
                            status_code=404)
    rec = _inflight_admit(ip, friendly, engine.replica_count(friendly))  # K slots + queue; else 429
    if rec is None:
        # #queue-depth: retryable overflow -> 429+Retry-After (Anthropic overloaded_error envelope).
        return JSONResponse({"type": "error", "error": {"type": "overloaded_error",
            "message": f"queue full for '{friendly}': 1 slot + "
                       f"{ENGINE_CONFIG.get('queue_depth', DEFAULT_QUEUE_DEPTH)} queued"}},
            status_code=429, headers={"Retry-After": "1"})
    # #cold-contract: known-but-cold model + auto-load OFF -> retryable typed signal (not a terminal
    # not_found_error). Unknown already 404'd above. With auto-load ON this never fires (loads below).
    if friendly not in engine.models and not ENGINE_CONFIG.get("auto_load", True):
        _inflight_release(rec)
        return JSONResponse({"type": "error", "error": {"type": "overloaded_error",
            "message": f"model '{friendly}' is not loaded and auto-load is disabled — "
                       f"POST /load it (or enable auto_load), then retry"}},
            status_code=503, headers={"Retry-After": "3"})
    try:
        resident = engine.models.get(friendly)
        ctx = resident.ctx if resident else 0
        lm = await engine.ensure_loaded(friendly, ctx, auto_load=True)
        if getattr(lm.spec, "is_embedding", False):   # encoder can't decode -> reject
            raise ValueError(f"model '{friendly}' is an embedding model; use /api/embed")
        tok = lm.tokenizer
    except ValueError as exc:
        # Misuse of a KNOWN model (embedding model on a chat endpoint) -> 400; transient
        # controller state ("updating", or an auto_load-flipped-off race reaching the "not
        # loaded" refusal) -> retryable 503. Never not_found — the model exists.
        _inflight_release(rec)
        _retry = ("updating" in str(exc)) or ("not loaded" in str(exc))
        return JSONResponse({"type": "error", "error": {
            "type": "overloaded_error" if _retry else "invalid_request_error",
            "message": str(exc)}}, status_code=503 if _retry else 400,
            headers=({"Retry-After": "3"} if _retry else {}))
    except Exception as exc:
        _inflight_release(rec)
        # #cold-contract + #at-capacity: a LOAD failure is NOT "model not found" — the model is
        # cataloged (unknown names already 404'd at resolve above). Previously this returned 404
        # not_found_error, so a capacity problem looked like a nonexistent model to Claude Code.
        # Retryable (busy/transient) -> 503 overloaded_error + Retry-After; a TERMINAL capacity
        # failure (auto-unload off / all pinned) -> 503 api_error "at_capacity", NO Retry-After.
        _term = isinstance(exc, CapacityError) and getattr(exc, "terminal", False)
        return JSONResponse({"type": "error", "error": {
            "type": "api_error" if _term else "overloaded_error",
            "message": f"{type(exc).__name__}: {exc}" + (
                " (at_capacity — a retry cannot succeed until a model is unloaded)" if _term else "")}},
            status_code=503, headers=({} if _term else {"Retry-After": "3"}))

    # #22 inc 3b/5c: pull any images + audio out so the chat template renders the per-item
    # placeholder (keep_images/keep_audio), then expand + splice their embeds below. Decode/
    # fetch runs OFF the event loop — _decode_image/_decode_audio may blocking-urlopen.
    images = await asyncio.to_thread(_collect_images, body.get("messages"))
    audios = await asyncio.to_thread(_collect_audio, body.get("messages"))
    target_id = MODELS[friendly][0] if friendly in MODELS else friendly
    mm = None
    mrope = None   # #22 inc 4/5c: (3D position_ids [3][q], base) when media embeds are spliced
    hf_tools = _anthropic_tools_to_hf(body.get("tools"))
    # #qwen3-thinking: map the Anthropic `thinking` request param onto the chat template's
    # enable_thinking switch (Qwen3-family). Absent (Claude Code's default) -> False -> the
    # template PRE-CLOSES the <think> block and the model answers DIRECTLY. Without this a
    # reasoning model burns its whole token budget on reasoning this path holds back/strips
    # -> the "loads fine but never produces output" qwen3.6 symptom (hundreds of hidden
    # reasoning tokens at APU decode speed look like a dead stream). Templates without the
    # variable simply ignore the extra kwarg, so it is passed unconditionally.
    _think_on = ((body.get("thinking") or {}).get("type") == "enabled")

    def _render_ids(chat):
        """Tokenize a chat with the tools-aware fallback (template throws on tools= for many
        multimodal-remapped checkpoints -> re-render without native tools + a text tool
        instruction; last-ditch: flatten). Shared by the vision and text-only renders."""
        try:
            if hf_tools:
                return _to_id_list(tok.apply_chat_template(chat, tools=hf_tools,
                                   add_generation_prompt=True, tokenize=True,
                                   enable_thinking=_think_on))
            return _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                               tokenize=True, enable_thinking=_think_on))
        except Exception as exc:
            print(f"[v1/messages] chat-template failed ({type(exc).__name__}: {exc}); "
                  f"re-rendering without native tools" + (" + tool instruction" if hf_tools else ""))
            # render-debug: enough shape info to reproduce a strict-template failure offline
            with contextlib.suppress(Exception):
                print(f"[v1/messages] render-debug: roles={[m.get('role') for m in chat]} "
                      f"tools={len(hf_tools or [])} think={_think_on}")
            chat2 = chat
            if hf_tools:
                instr = _tool_instruction(hf_tools)
                if chat and chat[0].get("role") == "system":
                    chat2 = [{"role": "system", "content": chat[0].get("content", "") + "\n\n" + instr}] + chat[1:]
                else:
                    chat2 = [{"role": "system", "content": instr}] + chat
            try:
                return _to_id_list(tok.apply_chat_template(chat2, add_generation_prompt=True,
                                   tokenize=True, enable_thinking=_think_on))
            except Exception:
                flat = "\n\n".join(f"{m['role']}: {_anth_flatten(m.get('content', ''))}"
                                   for m in chat2)
                return _to_id_list(tok(flat + "\n\nassistant:"))

    # Modality priority: audio-only OR vision-only (Omni's supported single-modality cases).
    # If both are present, prefer AUDIO and drop images (mixed audio+vision in one prompt =
    # a future increment; the single mm pair carries one embed set).
    do_audio = bool(audios)
    do_vision = bool(images) and not do_audio
    if images and do_audio:
        print(f"[v1/messages] both audio + image present -> audio path; {len(images)} "
              f"image(s) dropped (mixed AV not yet supported)")
    ids = _render_ids(_anthropic_messages_to_chat(body.get("system"), body.get("messages"),
                                                  keep_images=do_vision, keep_audio=do_audio))
    # #22 inc 5c: AUDIO path (Qwen2.5-Omni). Encode clip(s), expand each single <|AUDIO|> into
    # its token-count run, splice the embeds, and use sequential TMRoPE positions.
    if do_audio:
        try:
            enc_res = await asyncio.to_thread(_encode_audio, target_id, audios)
            embeds = enc_res.get("audio_embeds")
            counts = enc_res.get("counts") or []
            atid = enc_res.get("audio_token_id")
            n_emb = int(embeds.shape[0]) if embeds is not None else 0
            if atid is not None and n_emb and sum(counts) == n_emb:
                # #144 gemma4 audio: the chat template may render no audio placeholder (like the
                # Mistral3 vision case) — inject one audio token per clip after any leading BOS so
                # the run can be expanded + spliced. Archs whose template already emits it are as-is.
                if ids.count(int(atid)) != len(counts):
                    _bos = getattr(tok, "bos_token_id", None)
                    _p = 1 if (ids and _bos is not None and ids[0] == _bos) else 0
                    ids = list(ids[:_p]) + [int(atid)] * len(counts) + list(ids[_p:])
                    print(f"[v1/messages] injected {len(counts)} audio placeholder(s) (id {int(atid)})")
                ids = _wrap_image_runs(ids, int(atid), enc_res.get("wrap"))   # #144 gemma4 boa/eoa
                new_ids, positions, found = _expand_image_placeholders(ids, int(atid), counts)
                if found == len(counts) and len(positions) == n_emb:
                    ids, mm = new_ids, (positions, embeds)
                    if enc_res.get("pos_scheme") == "1d":
                        mrope = None   # #144 gemma4 audio: plain 1D positions (no TMRoPE)
                        print(f"[v1/messages] audio: {len(audios)} clip(s) -> {len(positions)} "
                              f"audio tokens spliced (counts={counts}, pos=1d)")
                    else:
                        # audio-only TMRoPE = sequential 0..seq-1 on all 3 dims (see
                        # _audio_position_ids); positions grow normally, unlike images.
                        mrope = _audio_position_ids(len(ids))
                        print(f"[v1/messages] audio: {len(audios)} clip(s) -> {len(positions)} "
                              f"audio tokens spliced (counts={counts}); TMRoPE base={mrope[1]}")
                else:
                    print(f"[v1/messages] audio MISMATCH: found {found} placeholder(s) "
                          f"(expected {len(counts)}), {len(positions)} positions vs {n_emb} "
                          f"embeds — text-only")
            else:
                print(f"[v1/messages] audio skip: audio_token_id={atid}, counts_sum="
                      f"{sum(counts)}, embeds={n_emb} — text-only")
        except Exception as exc:
            print(f"[v1/messages] audio encode failed ({type(exc).__name__}: {exc}); text-only")
        if mm is None:
            ids = _render_ids(_anthropic_messages_to_chat(body.get("system"),
                              body.get("messages"), keep_images=False, keep_audio=False))
            print("[v1/messages] audio unavailable -> rebuilt text-only prompt "
                  f"({len(ids)} tokens)")
    # #22 inc 3b: VISION path (only when no audio splice happened). Encode the image(s),
    # expand each single <|image_pad|> into its grid-derived run, stage embeds for splicing.
    if do_vision and mm is None:
        try:
            enc_res = await asyncio.to_thread(_encode_images, target_id, images)
            embeds = enc_res.get("image_embeds")
            counts = enc_res.get("counts") or []
            itid = enc_res.get("image_token_id")
            n_emb = int(embeds.shape[0]) if embeds is not None else 0
            if itid is not None and n_emb and sum(counts) == n_emb:
                # Some tokenizers render WITHOUT the image placeholder — e.g. Mistral3/Devstral have
                # no HF chat_template, so _render_ids falls back to a flat text join that drops image
                # parts, leaving nothing to splice at (found=0 -> text-only, image ignored). Inject
                # one image_token_id per image (just after any leading BOS) so the run can be
                # expanded + spliced. Archs whose template already emits the token are left as-is.
                if ids.count(int(itid)) != len(counts):
                    _bos = getattr(tok, "bos_token_id", None)
                    _p = 1 if (ids and _bos is not None and ids[0] == _bos) else 0
                    ids = list(ids[:_p]) + [int(itid)] * len(counts) + list(ids[_p:])
                    print(f"[v1/messages] injected {len(counts)} image placeholder(s) (id "
                          f"{int(itid)}) — tokenizer rendered none (no chat template)")
                ids = _wrap_image_runs(ids, int(itid), enc_res.get("wrap"))   # gemma4 boi/eoi
                _grc, _bk, _ei = _pixtral_break_end(tok, enc_res.get("grid_rc"))   # #150 Pixtral rows
                new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts,
                                                                       grid_rc=_grc, break_id=_bk, end_id=_ei)
                if found == len(counts) and len(positions) == n_emb:
                    ids, mm = new_ids, (positions, embeds)
                    if enc_res.get("pos_scheme") == "1d":
                        # Pixtral / Mistral3: standard 1D RoPE — spliced image tokens take normal
                        # sequential positions, no grid mRoPE. Leave mrope=None (worker default).
                        mrope = None
                        print(f"[v1/messages] vision: {len(images)} image(s) -> {len(positions)} "
                              f"image tokens spliced (counts={counts}); plain 1D positions")
                    else:
                        # #22 inc 4: 3D mRoPE positions for the expanded prompt (image tokens get
                        # t/h/w grid positions; the counter advances slowly past each image).
                        merge = int(enc_res.get("merge") or 1)
                        grid_list = enc_res.get("grid_list") or []
                        mrope = _mrope_position_ids(ids, grid_list, int(itid), merge)
                        print(f"[v1/messages] vision: {len(images)} image(s) -> {len(positions)} "
                              f"image tokens spliced (counts={counts}); mRoPE base={mrope[1]}")
                else:
                    print(f"[v1/messages] vision MISMATCH: found {found} placeholder(s) "
                          f"(expected {len(counts)}), {len(positions)} positions vs {n_emb} "
                          f"embeds — text-only")
            else:
                print(f"[v1/messages] vision skip: image_token_id={itid}, counts_sum="
                      f"{sum(counts)}, embeds={n_emb} — text-only")
        except Exception as exc:
            print(f"[v1/messages] vision encode failed ({type(exc).__name__}: {exc}); text-only")
        # On ANY vision failure/mismatch, REBUILD a genuinely text-only prompt so no raw
        # <|image_pad|> placeholders leak into the prefill (they'd embed as bare placeholder
        # tokens and degrade output). Only the success branch above keeps the expanded ids.
        if mm is None:
            ids = _render_ids(_anthropic_messages_to_chat(body.get("system"),
                              body.get("messages"), keep_images=False))
            print("[v1/messages] vision unavailable -> rebuilt text-only prompt "
                  f"({len(ids)} tokens)")
    # Reasoning models (Qwen3) whose template OPENS <think> in the prompt make the model
    # begin generation already mid-thought (output starts with reasoning, then </think>).
    # Detect it from the prompt tail so streaming can hold that reasoning back.
    starts_in_think = False
    with contextlib.suppress(Exception):
        tail = tok.decode(ids[-24:])
        starts_in_think = "<think>" in tail and "</think>" not in tail.split("<think>")[-1]

    # #runtime-knobs: same fallback chain as the Ollama/OpenAI path — request value, else the
    # model's runtime default (POST /model_config), else off. top_k is NATIVE Anthropic API;
    # the penalties/seed/min_p are cross-endpoint extensions accepted for parity.
    _sdef = getattr(lm, "sampling_defaults", None) or {}
    max_new = int(body.get("max_tokens") or _sdef.get("num_predict") or 512)
    # #load-temp: fall back to the model's per-load default temperature when the request sends none.
    _bt = body.get("temperature", None)
    _dt = getattr(lm, "default_temperature", None)
    temperature = float(_bt) if _bt is not None else float(_dt if _dt is not None else 0.0)
    # #min-p (extension: not in the Anthropic API, accepted for parity with the other endpoints)
    _bmp = body.get("min_p", None)
    _dmp = getattr(lm, "default_min_p", None)
    min_p = float(_bmp) if _bmp is not None else float(_dmp if _dmp is not None else 0.0)
    _btp = body.get("top_p", None)
    top_p = float(_btp) if _btp is not None else float(_sdef.get("top_p") or 1.0)

    def _bknob(k, alt=None):
        # step with `is None` so an explicit JSON null on the primary spelling can't shadow the alias
        for _key in (k, alt):
            if _key is not None:
                v = body.get(_key)
                if v is not None:
                    return v
        return _sdef.get(k)
    sampling = _coerce_knobs(_bknob)   # parse-time coercion -> malformed values fail pre-stream
    stream = bool(body.get("stream", False))
    state = {"tokens": 0}
    msg_id = _anth_id("msg")

    async def gen_raw():
        """Yield (text_piece, done_reason_or_None) — incremental, multibyte-safe,
        WITH the literal <tool_call>/<think> markup preserved for downstream parsing."""
        # #recovery: re-capture the real body-pump task (streaming returns the route task early) so the
        # gen-stall watchdog / /cancel can actually abort a wedged Claude-Code generation. See run().
        if rec is not None:
            with contextlib.suppress(Exception):
                rec["task"] = asyncio.current_task()
        det = IncrementalDetok(tok)   # #inc-detok: O(tail)/token, byte-identical to full re-decode
        prev = ""
        try:
          async for tid, reason in engine.generate(friendly, ids, max_new,
                                                   temperature, top_p, False, rec=rec, mm=mm,
                                                   mrope=mrope, min_p=min_p, sampling=sampling):
            if tid is not None:
                text = det.push(tid)
                state["tokens"] = det.n
                METRICS["tokens"] += 1
                if text.endswith("�"):
                    continue
                piece, prev = text[len(prev):], text
                if piece:
                    yield piece, None
            if reason:
                text = det.current()
                if text.endswith("�"):               # incomplete multi-byte at gen end -> drop partial
                    text = text.rstrip("�")           # (#detok-tail)
                yield text[len(prev):], reason
        finally:
            _inflight_release(rec)   # free the slot/queue entry when generation ends

    # ---------- streaming (Anthropic SSE) ----------
    if stream:
        async def sse():
            def ev(name: str, payload: dict) -> str:
                return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

            s = ev("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": len(ids), "output_tokens": 0}}})
            METRICS["api_out"] += len(s)
            yield s
            yield ev("ping", {"type": "ping"})

            raw = ""
            emitted_plain = emitted_tools = next_index = text_index = 0
            text_open = False
            think_open = False
            think_index = emitted_think = 0
            finish = "stop"
            try:
                async for piece, reason in _with_keepalive(gen_raw()):
                    if piece is None and reason is None:
                        # #prefill-keepalive tick: >10s with no token (prefill / slow decode)
                        yield ev("ping", {"type": "ping"})
                        continue
                    if piece:
                        raw += piece
                        # #qwen3-thinking: surface reasoning as a first-class Anthropic
                        # `thinking` content block streamed via thinking_delta — NOT silent
                        # hold-back pings (a long think phase at APU decode speed otherwise
                        # looks like a dead stream and agent harnesses give up). The think
                        # region: everything before </think> when the template opened the
                        # block in the prompt (starts_in_think) or the model self-opened one.
                        _ts = (raw[len("<think>"):] if raw.startswith("<think>")
                               else (raw if starts_in_think else None))
                        _th = _ts.split("</think>", 1)[0] if _ts is not None else ""
                        if len(_th) > emitted_think and not text_open:
                            if not think_open:
                                think_index = next_index
                                next_index += 1
                                yield ev("content_block_start", {
                                    "type": "content_block_start", "index": think_index,
                                    "content_block": {"type": "thinking", "thinking": ""}})
                                think_open = True
                            yield ev("content_block_delta", {
                                "type": "content_block_delta", "index": think_index,
                                "delta": {"type": "thinking_delta",
                                          "thinking": _th[emitted_think:]}})
                            emitted_think = len(_th)
                        plain, tools = _segment_tools(raw, starts_in_think)
                        if (len(plain) <= emitted_plain and len(tools) <= emitted_tools
                                and not think_open):
                            # tokens flowing but all held back (a partial tool call) —
                            # ping so the client doesn't time out.
                            yield ev("ping", {"type": "ping"})
                        if (len(plain) > emitted_plain or len(tools) > emitted_tools) and think_open:
                            # reasoning finished — close the thinking block (spec-shaped
                            # empty signature first) before text/tool blocks open.
                            yield ev("content_block_delta", {
                                "type": "content_block_delta", "index": think_index,
                                "delta": {"type": "signature_delta", "signature": ""}})
                            yield ev("content_block_stop", {
                                "type": "content_block_stop", "index": think_index})
                            think_open = False
                        if len(plain) > emitted_plain:
                            if not text_open:
                                text_index = next_index
                                next_index += 1
                                yield ev("content_block_start", {
                                    "type": "content_block_start", "index": text_index,
                                    "content_block": {"type": "text", "text": ""}})
                                text_open = True
                            delta = plain[emitted_plain:]
                            emitted_plain = len(plain)
                            yield ev("content_block_delta", {
                                "type": "content_block_delta", "index": text_index,
                                "delta": {"type": "text_delta", "text": delta}})
                        while emitted_tools < len(tools):
                            if text_open:
                                yield ev("content_block_stop", {
                                    "type": "content_block_stop", "index": text_index})
                                text_open = False
                            blk = _tool_to_block(tools[emitted_tools])
                            emitted_tools += 1
                            idx = next_index
                            next_index += 1
                            yield ev("content_block_start", {
                                "type": "content_block_start", "index": idx,
                                "content_block": {"type": "tool_use", "id": blk["id"],
                                                  "name": blk["name"], "input": {}}})
                            yield ev("content_block_delta", {
                                "type": "content_block_delta", "index": idx,
                                "delta": {"type": "input_json_delta",
                                          "partial_json": json.dumps(blk["input"])}})
                            yield ev("content_block_stop", {
                                "type": "content_block_stop", "index": idx})
                    if reason:
                        finish = reason
            except asyncio.CancelledError:
                # #endpoint-weather: watchdog reclaim mid-stream -> Anthropic `overloaded_error`
                # (what Claude Code / Anthropic SDKs back off on). uncancel() so the event flushes;
                # a genuine client disconnect / user cancel re-raises (no retry invite).
                if not (rec is not None and rec.get("reclaimed")):
                    raise
                with contextlib.suppress(Exception):
                    asyncio.current_task().uncancel()
                if think_open:
                    yield ev("content_block_stop", {"type": "content_block_stop",
                                                    "index": think_index})
                    think_open = False
                if text_open:
                    yield ev("content_block_stop", {"type": "content_block_stop",
                                                    "index": text_index})
                    text_open = False
                yield ev("error", {"type": "error", "error": {"type": "overloaded_error",
                    "message": "generation reclaimed under backend contention — retry with backoff"}})
                return
            except Exception as exc:
                # mid-stream failure: close any open block, then signal end. #endpoint-weather:
                # contention-class -> overloaded_error (retryable) so Claude Code backs off instead
                # of surfacing a hard api_error on a transient squeeze.
                if think_open:
                    yield ev("content_block_stop", {"type": "content_block_stop",
                                                    "index": think_index})
                    think_open = False
                if text_open:
                    yield ev("content_block_stop", {"type": "content_block_stop",
                                                    "index": text_index})
                    text_open = False
                _retry, _msg = _stream_fail(exc, rec)
                yield ev("error", {"type": "error", "error": {
                    "type": "overloaded_error" if _retry else "api_error", "message": _msg}})
                return
            if think_open:
                # generation ended still inside the reasoning block (ran out of budget
                # mid-thought) — close it cleanly so the client renders what it got.
                yield ev("content_block_delta", {
                    "type": "content_block_delta", "index": think_index,
                    "delta": {"type": "signature_delta", "signature": ""}})
                yield ev("content_block_stop", {"type": "content_block_stop",
                                                "index": think_index})
            if text_open:
                yield ev("content_block_stop", {"type": "content_block_stop",
                                                "index": text_index})
            stop_reason = ("tool_use" if emitted_tools else
                           ("max_tokens" if finish == "length" else "end_turn"))
            yield ev("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": state["tokens"]}})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ---------- non-streaming ----------
    full = ""
    finish = "stop"
    try:
        async for piece, reason in gen_raw():
            full += piece
            if reason:
                finish = reason
    except asyncio.CancelledError:
        # #endpoint-weather: watchdog reclaim (rec["reclaimed"]) -> Anthropic-typed retryable
        # overload (529 + overloaded_error is what Anthropic SDKs/Claude Code back off on).
        # User /cancel-/terminate, client disconnects and shutdown re-raise — no retry invite.
        if rec is not None and rec.get("reclaimed"):
            return JSONResponse({"type": "error", "error": {"type": "overloaded_error",
                                 "message": "generation reclaimed under backend contention — "
                                            "retry with backoff"}},
                                status_code=529, headers={"Retry-After": "15"})
        raise
    except Exception as exc:
        if _transient_gen_exc(exc):
            # #endpoint-weather: contention-class failure -> retryable overloaded_error, not a 500.
            return JSONResponse({"type": "error", "error": {"type": "overloaded_error",
                                 "message": f"transient backend contention "
                                            f"({type(exc).__name__}: {exc}) — retry with backoff"}},
                                status_code=529, headers={"Retry-After": "15"})
        return JSONResponse({"type": "error",
                             "error": {"type": "api_error", "message": str(exc)}},
                            status_code=500)
    clean, raw_tools = _extract_tools(full)
    # #qwen3-thinking: surface reasoning as a first-class `thinking` content block. Think
    # region = everything before </think> when the template opened the block in the prompt
    # (starts_in_think) or the model self-opened one. If the model ran out of budget while
    # STILL thinking (no </think>), _strip_reasoning can't strip the dangling reasoning —
    # without this guard it leaks verbatim into the text block (the raw "Thinking Process:"
    # text seen on qwen3.6) — so classify it all as thinking and empty the text.
    _ts = (full[len("<think>"):] if full.startswith("<think>")
           else (full if starts_in_think else None))
    think_txt = ""
    if _ts is not None:
        think_txt = _ts.split("</think>", 1)[0].strip()
        if "</think>" not in _ts:
            clean = ""
    content = []
    if think_txt:
        content.append({"type": "thinking", "thinking": think_txt, "signature": ""})
    if clean:
        content.append({"type": "text", "text": clean})
    for tb in raw_tools:
        content.append(_tool_to_block(tb))
    if not content:
        content.append({"type": "text", "text": ""})
    stop_reason = ("tool_use" if raw_tools else
                   ("max_tokens" if finish == "length" else "end_turn"))
    payload = {"id": msg_id, "type": "message", "role": "assistant", "model": model,
               "content": content, "stop_reason": stop_reason, "stop_sequence": None,
               "usage": {"input_tokens": len(ids), "output_tokens": state["tokens"]}}
    METRICS["api_out"] += len(json.dumps(payload))
    return JSONResponse(payload)


async def _count_tokens_anthropic(body: dict):
    """POST /v1/messages/count_tokens — Claude Code uses this for context budgeting.
    Exact via the resident tokenizer; a char/4 estimate if the model isn't loaded
    (so we never trigger a slow distributed load just to count)."""
    model = body.get("model", "")
    try:
        friendly = resolve_model_name(model)
    except ValueError as exc:
        return JSONResponse({"type": "error",
                             "error": {"type": "not_found_error", "message": str(exc)}},
                            status_code=404)
    chat = _anthropic_messages_to_chat(body.get("system"), body.get("messages"))
    hf_tools = _anthropic_tools_to_hf(body.get("tools"))
    n = None
    resident = engine.models.get(friendly)
    if resident is not None:
        tok = resident.tokenizer
        with contextlib.suppress(Exception):
            if hf_tools:
                enc = tok.apply_chat_template(chat, tools=hf_tools,
                                              add_generation_prompt=True, tokenize=True)
            else:
                enc = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
            n = len(_to_id_list(enc))
    if n is None:
        n = _estimate_tokens(chat)
    return JSONResponse({"input_tokens": n})
