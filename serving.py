"""serving.py: the request-serving layer relocated from server.py (m4c154 code-split).
Holds _serve (Ollama/OpenAI generate+chat), _serve_anthropic (Claude Code backend),
_count_tokens_anthropic, and _serve's private helpers _prepare/_ka_is_unload — all
BYTE-IDENTICAL to the originals. Module globals (engine, registry, resolve_model_name,
_not_found_json, METRICS, formats helpers …) are injected at startup by state.bind() — see
state.py. server.py back-imports _serve/_serve_anthropic/_count_tokens_anthropic so the
relocated routes_api resolves them through the published namespace. Controller-only leaf;
in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations




def _normalize_ollama_images(messages):
    """#vl-vision: Ollama's NATIVE chat API attaches images as a per-message `images: [<b64>|<url>]`
    array next to a plain-string `content` — a shape BOTH the HF chat template and `_collect_images`
    (which read only OpenAI/Anthropic content blocks) ignore, so those images were silently dropped
    ("model skips images"). Rewrite every such message so each entry becomes an OpenAI `image_url`
    content block (a data: URL for raw base64) appended to the content — byte-equivalent to the proven
    /v1/chat/completions path, so the template renders the vision placeholder and `_collect_images`
    finds the image. Returns a NEW list only when something changed; messages without `images` pass
    through untouched."""
    if not messages:
        return messages
    out, changed = [], False
    for m in messages:
        imgs = m.get("images") if isinstance(m, dict) else None
        if not imgs or not isinstance(imgs, (list, tuple)):
            out.append(m)
            continue
        changed = True
        c = m.get("content")
        blocks = list(c) if isinstance(c, list) else ([{"type": "text", "text": str(c)}] if c else [])
        for it in imgs:
            if isinstance(it, dict) and it.get("type") in ("image", "image_url"):
                blocks.append(it)                       # already a content block
                continue
            if isinstance(it, dict):
                s = it.get("url") or it.get("data")
            else:
                s = it
            if not isinstance(s, str) or not s.strip():
                continue
            s = s.strip()
            if s.startswith(("data:", "http://", "https://")):
                url = s
            else:                                        # raw base64 (the Ollama convention)
                url = "data:image/png;base64," + s       # mime ignored on decode (PIL sniffs the bytes)
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        nm = {k: v for k, v in m.items() if k != "images"}
        nm["content"] = blocks
        out.append(nm)
    return out if changed else messages


def _tool_args(call: dict) -> dict:
    """Normalize a parsed tool call's arguments to a dict — the parser may key them 'arguments'
    or 'parameters', or leave a JSON string (some models emit the args as a quoted string)."""
    a = call.get("arguments")
    if isinstance(a, dict):
        return a
    if isinstance(call.get("parameters"), dict):
        return call["parameters"]
    if isinstance(a, str):
        with contextlib.suppress(Exception):
            v = json.loads(a)
            if isinstance(v, dict):
                return v
    return {}


def _openai_tool_calls(calls: list, with_index: bool = False) -> list:
    """[{name,arguments}] -> OpenAI tool_calls [{id,type,function:{name,arguments:<JSON str>}}].
    OpenAI carries `arguments` as a JSON STRING (unlike Ollama, which uses an object)."""
    out = []
    for i, c in enumerate(calls):
        tc = {"id": _anth_id("call"), "type": "function",
              "function": {"name": c.get("name"), "arguments": json.dumps(_tool_args(c))}}
        if with_index:
            tc["index"] = i
        out.append(tc)
    return out


def _ollama_tool_calls(calls: list) -> list:
    """[{name,arguments}] -> Ollama tool_calls [{function:{name,arguments:<object>}}]."""
    return [{"function": {"name": c.get("name"), "arguments": _tool_args(c)}} for c in calls]


def _render_chat_ids(tok, chat, hf_tools):
    """#tools: tokenize a chat, injecting tool definitions when present. `hf_tools` is the
    OpenAI/HF shape ([{type:function, function:{name,description,parameters}}]) that /api/chat and
    /v1/chat/completions already send, passed straight to apply_chat_template(tools=). With NO tools
    this is byte-identical to the old plain render. 3-tier fallback mirrors _serve_anthropic._render_ids:
    native tools -> a text tool instruction in the system prompt (templates that reject tools=) ->
    a flat text join (templates that reject the chat entirely). So even a model whose HF template has
    no tool support still SEES the tools and can emit <tool_call> markup we parse back out."""
    if not hf_tools:
        return _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
    # A template WITHOUT tool handling doesn't throw on tools= — it silently renders without them
    # (e.g. Qwen2.5-VL's vision-only template), so the model never sees the defs. Detect that up
    # front and go straight to the text tool instruction.
    _tmpl = getattr(tok, "chat_template", None) or ""
    _native = isinstance(_tmpl, str) and ("tools" in _tmpl or "tool_call" in _tmpl)
    try:
        if not _native:
            raise ValueError("chat template has no tool support")
        return _to_id_list(tok.apply_chat_template(chat, tools=hf_tools,
                           add_generation_prompt=True, tokenize=True))
    except Exception as exc:
        print(f"[serve] no native template tools ({type(exc).__name__}: {exc}); "
              f"injecting a text tool instruction instead")
        instr = _tool_instruction(hf_tools)
        if chat and chat[0].get("role") == "system":
            chat2 = [{"role": "system",
                      "content": str(chat[0].get("content", "")) + "\n\n" + instr}] + list(chat[1:])
        else:
            chat2 = [{"role": "system", "content": instr}] + list(chat)
        try:
            return _to_id_list(tok.apply_chat_template(chat2, add_generation_prompt=True, tokenize=True))
        except Exception:
            flat = "\n\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in chat2)
            return _to_id_list(tok(flat + "\n\nassistant:"))


async def _prepare_vision(target_id: str, tok, ids: list, images: list):
    """#vl-vision: shared image splice for the OpenAI (/v1/chat/completions) + Ollama (/api/chat)
    serve path — the same machinery /v1/messages uses. Encodes the images, expands the image
    placeholder run in `ids` (injecting the placeholder after BOS if the chat template rendered
    none), and computes mRoPE positions. Returns (ids, mm, mrope); on no images or ANY failure
    returns (ids, None, None) so the caller degrades to a clean text-only prompt."""
    if not images:
        return ids, None, None
    try:
        enc_res = await asyncio.to_thread(_encode_images, target_id, images)
        embeds = enc_res.get("image_embeds")
        counts = enc_res.get("counts") or []
        itid = enc_res.get("image_token_id")
        n_emb = int(embeds.shape[0]) if embeds is not None else 0
        if not (itid is not None and n_emb and sum(counts) == n_emb):
            return ids, None, None
        if ids.count(int(itid)) != len(counts):   # template rendered no placeholder -> inject after BOS
            _bos = getattr(tok, "bos_token_id", None)
            _p = 1 if (ids and _bos is not None and ids[0] == _bos) else 0
            ids = list(ids[:_p]) + [int(itid)] * len(counts) + list(ids[_p:])
        new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts)
        if not (found == len(counts) and len(positions) == n_emb):
            return ids, None, None
        mm = (positions, embeds)
        if enc_res.get("pos_scheme") == "1d":
            mrope = None    # Pixtral/Mistral3/Gemma: plain 1D positions
        else:
            merge = int(enc_res.get("merge") or 1)
            grid_list = enc_res.get("grid_list") or []
            mrope = _mrope_position_ids(new_ids, grid_list, int(itid), merge)
        print(f"[serve] vision: {len(images)} image(s) -> {len(positions)} tokens spliced "
              f"(counts={counts}, pos={'1d' if mrope is None else 'mrope'})")
        return new_ids, mm, mrope
    except Exception as exc:
        print(f"[serve] vision encode failed ({type(exc).__name__}: {exc}); text-only")
        return ids, None, None


async def _prepare(model: str, prompt: Optional[str], messages, body: dict):
    """Resolve+load model, build prompt token ids, and pull sampling options."""
    friendly = resolve_model_name(model)
    # Respect the ctx the model was loaded with (e.g. via the dashboard) instead
    # of forcing DEFAULT_CTX — otherwise the first generate after a smaller-ctx
    # load silently triggers a slow full reload. Only fall back to DEFAULT_CTX
    # when this model isn't already loaded.
    resident = engine.models.get(friendly)
    ctx = resident.ctx if resident else 0   # 0 => auto-load at the model's native training context
    # CPU-only request (Ollama convention): options.num_gpu == 0 means "offload 0 layers to
    # GPU" => load to RAM only, never VRAM. Also accept an explicit options.cpu_only bool.
    # If the model is ALREADY loaded, ensure_loaded ignores this and serves the live copy.
    _o = body.get("options") or {}
    _ng = _o.get("num_gpu")
    cpu_only = bool(_o.get("cpu_only", False))
    with contextlib.suppress(TypeError, ValueError):
        cpu_only = cpu_only or (_ng is not None and int(_ng) == 0)
    try:
        lm = await engine.ensure_loaded(friendly, ctx, cpu_only=cpu_only, auto_load=True)
    except ValueError as exc:   # unknown model (auto-load only loads KNOWN registered models)
        return JSONResponse({"error": str(exc), "model": model}, status_code=404)
    # An ENCODER can't decode tokens — reject it here so it never hits the generate path.
    # _serve wraps this ValueError into a clear 400 (the tuple-unpack-and-400 path).
    if getattr(lm.spec, "is_embedding", False):
        raise ValueError(f"model '{friendly}' is an embedding model; use /api/embed")
    tok = lm.tokenizer
    if messages is not None:
        messages = _normalize_ollama_images(messages)   # #vl-vision: Ollama-native images[] -> blocks
        _tc = body.get("tools") if body.get("tool_choice") != "none" else None
        ids = _render_chat_ids(tok, messages, _tc)       # #tools: inject tool defs when the request sends them
    else:
        ids = _to_id_list(tok(prompt or ""))
    # #vl-vision: OpenAI/Ollama image support — extract images from the messages and splice their
    # embeds (was previously only wired into /v1/messages). No-op for text requests / generate mode.
    mm = mrope = None
    if messages is not None:
        _imgs = _collect_images(messages)
        if _imgs:
            ids, mm, mrope = await _prepare_vision(lm.target_id, tok, ids, _imgs)
    opts = body.get("options") or {}
    temperature = float(opts.get("temperature", body.get("temperature", 0.0)))
    top_p = float(opts.get("top_p", body.get("top_p", 1.0)))
    max_new = int(opts.get("num_predict", body.get("max_tokens", 256)))
    stream = bool(body.get("stream", True))
    speculative = bool(opts.get("speculative", body.get("speculative", False)))
    spec_k = int(opts.get("spec_k", body.get("spec_k", 0)) or 0)   # per-request SPEC_K override (0=default)
    # #ctx-guard: reject a prompt that exceeds the loaded context window BEFORE dispatch — an over-ctx
    # prefill overflows the worker's fixed KV cache and crashes the shard. (engine.generate also backstops
    # this universally; rejecting here surfaces a clean error to the Ollama/OpenAI paths.)
    _lc = int(getattr(lm, "ctx", 0) or 0)
    if _lc and len(ids) >= _lc:
        raise ValueError(f"prompt is {len(ids)} tokens but model '{friendly}' is loaded with a "
                         f"{_lc}-token context window — shorten the prompt or reload it at a larger ctx")
    return friendly, tok, ids, temperature, top_p, max_new, stream, speculative, spec_k, mm, mrope


def _ka_is_unload(v) -> bool:
    """Ollama keep_alive: 0 -> unload now; negative -> keep forever; positive -> keep N seconds.
    Returns True ONLY for a zero keep_alive (the client's 'unload' signal). Accepts int/float and
    duration strings ('0', '0s', '0m'). bool is rejected (a stray True/False isn't a duration)."""
    if v is None or isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v == 0
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(v))
    return bool(m) and float(m.group(1)) == 0.0


async def _serve(model: str, prompt: Optional[str], messages, body: dict, mode: str,
                 ip: str = "?"):
    METRICS["api_in"] += len(json.dumps(body))
    # CLIENT UNLOAD (Ollama keep_alive: 0): a CLIENT asking to expire/unload a model is IGNORED — we
    # keep models resident (ONLY the backend interface/dashboard /unload evicts). Reply with the
    # Ollama 'unload' ack so the client is satisfied, but DON'T touch the resident model. Only a PURE
    # unload (keep_alive 0 + no prompt/messages) short-circuits; a real generate with keep_alive:0
    # still generates (we simply never auto-unload after).
    if _ka_is_unload(body.get("keep_alive")) and not (prompt or "").strip() and not messages:
        # silently ignore client keep_alive:0 unloads — no activity-log line (just keep the model)
        if mode in ("openai", "openai_text"):
            return JSONResponse({"id": "chatcmpl-noop", "object": "chat.completion",
                                 "created": int(time.time()), "model": model,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": ""}}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
        out = {"model": model, "created_at": _iso(), "done": True, "done_reason": "unload",
               "total_duration": 0, "load_duration": 0, "prompt_eval_count": 0,
               "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0}
        out["message" if mode == "chat" else "response"] = "" if mode != "chat" else {"role": "assistant", "content": ""}
        return JSONResponse(out)
    # Resolve + admit BEFORE loading so a request waiting on a model (even while it
    # loads) shows in that model's queue. 1 slot + queue_depth waiters; else 503.
    try:
        friendly = resolve_model_name(model)
    except Exception:
        return _not_found_json(model, mode)   # unknown model -> 404 (OpenAI envelope|Ollama shape)
    rec = _inflight_admit(ip, friendly, engine.replica_count(friendly))  # K slots for K replicas
    if rec is None:
        # #queue-depth: overflow beyond (1 slot + queue_depth) is RETRYABLE — return 429+Retry-After
        # (a fan-out client should back off and retry, not treat it as a hard 503 outage).
        return JSONResponse(
            {"error": f"queue full for '{friendly}': 1 slot + "
                      f"{ENGINE_CONFIG.get('queue_depth', DEFAULT_QUEUE_DEPTH)} queued — "
                      f"retry shortly"}, status_code=429, headers={"Retry-After": "1"})
    # #cold-contract (BUG-1/2): a KNOWN-but-not-resident model with auto-load OFF must return a
    # RETRYABLE typed signal HERE — BEFORE any 200/SSE is opened — never the streaming empty-200
    # error frame or a bare non-stream 404 (a fan-out/quorum client misreads an empty-200 as a
    # finished-but-empty vote, and a bare 404 as an unknown model). Unknown models already 404'd
    # above with code=model_not_found (terminal); this is the distinct cold-but-cataloged case.
    # With auto-load ON this never fires — the keepalive path below loads the model and serves it.
    if friendly not in engine.models and not ENGINE_CONFIG.get("auto_load", True):
        _inflight_release(rec)
        _msg = (f"model '{friendly}' is not loaded and auto-load is disabled — "
                f"POST /load it (or enable auto_load), then retry")
        if mode in ("openai", "openai_text"):
            return JSONResponse({"error": {"message": _msg, "type": "model_loading",
                                 "code": "not_loaded"}}, status_code=503,
                                headers={"Retry-After": "3"})
        return JSONResponse({"model": friendly, "error": _msg, "state": "not_loaded"},
                            status_code=503, headers={"Retry-After": "3"})
    created = int(time.time())
    _idp = "cmpl-" if mode == "openai_text" else "chatcmpl-"   # OpenAI: text vs chat id prefix
    cmpl_id = _idp + hashlib.sha256(str(time.time()).encode()).hexdigest()[:24]
    state = {"tokens": 0}  # token count (run() updates it; stream funcs read it)
    # stream default: OpenAI endpoints default FALSE (single JSON when omitted); Ollama default TRUE.
    _stream_default = mode not in ("openai", "openai_text")
    stream = bool(body.get("stream", _stream_default))   # decided from the body alone — no model load needed
    P: dict = {}             # prepared values, filled by _prepare (resident: instant; else load-on-request)
    _KEEPALIVE_LOAD_S = 8.0  # while a requested model auto-loads, emit a keepalive this often (Ollama-style)

    async def run():
        """Yield (text_piece, done_reason_or_None). Incremental detokenization:
        decode the cumulative id list and emit only the newly-completed suffix,
        holding back trailing bytes that don't yet form a whole character (so
        multi-byte UTF-8 — emoji/CJK — isn't corrupted by per-token decoding).
        Entered ONLY after the model is loaded + tokenized (values live in P), and
        it OWNS the inflight slot from here — its finally releases rec exactly once."""
        friendly = P["friendly"]; ids = P["ids"]; tok = P["tok"]
        max_new = P["max_new"]; temperature = P["temperature"]; top_p = P["top_p"]
        speculative = P["speculative"]; spec_k = P["spec_k"]
        # #recovery: re-capture THIS body-pump task as the cancel handle. _inflight_admit captured the
        # route-handler task, but a streaming response RETURNS that task immediately and the real
        # generation runs here in the StreamingResponse body task — so the gen-stall watchdog's (and
        # /cancel's) rec["task"].cancel() would otherwise be a no-op against an already-finished task,
        # leaving the wedged generate() holding the lock + a leaked pending future until GEN_TIMEOUT.
        if rec is not None:
            with contextlib.suppress(Exception):
                rec["task"] = asyncio.current_task()
        produced: list[int] = []
        prev = ""
        try:
            async for tid, reason in engine.generate(friendly, ids, max_new, temperature,
                                                     top_p, speculative, rec=rec, spec_k=spec_k,
                                                     mm=P.get("mm"), mrope=P.get("mrope")):
                if tid is not None:
                    produced.append(tid)
                    state["tokens"] = len(produced)
                    METRICS["tokens"] += 1
                    text = _decode_visible(tok, produced)
                    if text.endswith("�"):   # incomplete multi-byte char; wait
                        continue
                    piece, prev = text[len(prev):], text
                    if piece:
                        yield piece, None
                if reason:
                    text = _decode_visible(tok, produced)
                    if text.endswith("�"):           # gen ended mid multi-byte char -> drop the
                        text = text.rstrip("�")       # partial (it can never complete now) (#detok-tail)
                    yield text[len(prev):], reason   # flush remainder + signal done
        finally:
            _inflight_release(rec)   # free the slot/queue entry when generation ends

    def _map_finish(reason: str) -> str:
        return "length" if reason == "length" else "stop"

    async def _prep_unpack():
        """Auto-load (Ollama-style load-on-request) + tokenize, then fill P. Returns None on success,
        or an error message string on failure (unknown model / auto-load off / embedding model).
        Raises only on unexpected errors (caught by the caller)."""
        res = await _prepare(model, prompt, messages, body)
        if isinstance(res, JSONResponse):   # _prepare's error path (unknown model / auto-load off)
            msg = "model unavailable"
            with contextlib.suppress(Exception):
                msg = json.loads(bytes(res.body).decode()).get("error", msg)
            return msg
        (P["friendly"], P["tok"], P["ids"], P["temperature"], P["top_p"],
         P["max_new"], _st, P["speculative"], P["spec_k"], P["mm"], P["mrope"]) = res
        return None

    # #cold-contract (BUG-1/2/6/8): LOAD (or fail) BEFORE opening ANY response — for streaming too.
    # Previously the streaming path opened HTTP 200 + SSE immediately, then ran the load inside the
    # stream; a load FAILURE (auto-load off, capacity/VRAM, unknown) could then only be emitted as a
    # terminal {done_reason:"error", content:""} frame — an empty-200 a fan-out/quorum client misreads
    # as a finished/empty vote. Now BOTH stream and non-stream block here until the model is ready, so
    # a failure surfaces as a real typed HTTP error (503+Retry-After retryable, or 4xx) with NO 200
    # ever opened. Success -> P is filled and the model is resident; the decode loop below never runs
    # against a mid-load/unstable pipeline (also closes BUG-8's gate).
    try:
        emsg = await _prep_unpack()
    except Exception as exc:
        _inflight_release(rec)
        log_activity(f"generate {model}: prepare FAILED — {exc!r}")
        # Resident-but-failed (e.g. ctx-too-long) = a genuine bad request -> 400 (not retryable).
        # Not-resident = the LOAD itself failed (capacity/placement) -> retryable 503 + Retry-After.
        _bad_req = friendly in engine.models
        _code = 400 if _bad_req else 503
        _hdr = {} if _bad_req else {"Retry-After": "3"}
        _emsg = f"{type(exc).__name__}: {exc}"
        if mode in ("openai", "openai_text"):
            return JSONResponse({"error": {"message": _emsg,
                                 "type": "invalid_request_error" if _bad_req else "model_loading"}},
                                status_code=_code, headers=_hdr)
        return JSONResponse({"error": _emsg, "model": friendly}, status_code=_code, headers=_hdr)
    if emsg is not None:   # couldn't load (auto-load off / capacity / embedding) -> RETRYABLE typed 503
        _inflight_release(rec)
        if mode in ("openai", "openai_text"):
            return JSONResponse({"error": {"message": emsg, "type": "model_loading", "code": "loading"}},
                                status_code=503, headers={"Retry-After": "3"})
        return JSONResponse({"model": friendly, "error": emsg, "state": "loading"},
                            status_code=503, headers={"Retry-After": "3"})

    # #tools: native tool-calling for /api/chat + /v1/chat/completions. Opt-in — only when the request
    # carries `tools` (and tool_choice != "none") — so the text/vision fast paths stay byte-unchanged.
    # The tool defs were injected into the prompt by _prepare/_render_chat_ids; here we parse the model's
    # emitted <tool_call>/<invoke>/<function> markup back into structured tool_calls (the same
    # format-agnostic parser the Anthropic path uses). openai_text (legacy completions) has no tools.
    tools_req = bool(body.get("tools")) and body.get("tool_choice") != "none" and mode in ("chat", "openai")
    starts_in_think = False   # reasoning template opened <think> in the prompt -> hold reasoning back
    if tools_req:
        with contextlib.suppress(Exception):
            _tail = P["tok"].decode(P["ids"][-24:])
            starts_in_think = "<think>" in _tail and "</think>" not in _tail.split("<think>")[-1]

    # ---------- streaming ----------
    if stream:
        # #tools: tool-aware streamers (used only when the request sent tools) — emit visible text as
        # normal content deltas and each COMPLETE tool call as a structured delta, mirroring the proven
        # Anthropic SSE segmentation (_segment_tools holds back partial markup + reasoning).
        async def openai_tool_stream():
            yield "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
            raw = ""; emitted_plain = 0; emitted_tools = 0; finish = "stop"
            try:
                async for piece, reason in run():
                    if piece:
                        raw += piece
                        plain, tools = _segment_tools(raw, starts_in_think)
                        if len(plain) > emitted_plain:
                            delta = plain[emitted_plain:]; emitted_plain = len(plain)
                            s = "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                                "created": created, "model": model, "choices": [{"index": 0,
                                "delta": {"content": delta}, "finish_reason": None}]}) + "\n\n"
                            METRICS["api_out"] += len(s); yield s
                        while emitted_tools < len(tools):
                            tc = _openai_tool_calls([tools[emitted_tools]], with_index=True)
                            tc[0]["index"] = emitted_tools; emitted_tools += 1
                            s = "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                                "created": created, "model": model, "choices": [{"index": 0,
                                "delta": {"tool_calls": tc}, "finish_reason": None}]}) + "\n\n"
                            METRICS["api_out"] += len(s); yield s
                    if reason:
                        finish = reason
            except Exception:
                pass
            fr = "tool_calls" if emitted_tools else _map_finish(finish)
            s = "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]}) + "\n\n"
            s += "data: [DONE]\n\n"
            METRICS["api_out"] += len(s); yield s

        async def ollama_tool_stream():
            t0 = time.perf_counter_ns(); done_reason = "stop"; err = None
            raw = ""; emitted_plain = 0; emitted_tools = 0
            try:
                async for piece, reason in run():
                    if piece:
                        raw += piece
                        plain, tools = _segment_tools(raw, starts_in_think)
                        if len(plain) > emitted_plain:
                            delta = plain[emitted_plain:]; emitted_plain = len(plain)
                            s = json.dumps({"model": model, "created_at": _iso(),
                                "message": {"role": "assistant", "content": delta}, "done": False}) + "\n"
                            METRICS["api_out"] += len(s); yield s
                        while emitted_tools < len(tools):
                            tc = _ollama_tool_calls([tools[emitted_tools]]); emitted_tools += 1
                            s = json.dumps({"model": model, "created_at": _iso(),
                                "message": {"role": "assistant", "content": "", "tool_calls": tc},
                                "done": False}) + "\n"
                            METRICS["api_out"] += len(s); yield s
                    if reason:
                        done_reason = reason
            except Exception as exc:
                err, done_reason = str(exc), "error"
            dur = time.perf_counter_ns() - t0
            final = {"model": model, "created_at": _iso(), "done": True, "done_reason": done_reason,
                     "total_duration": dur, "load_duration": 0,
                     "prompt_eval_count": len(P.get("ids", [])), "prompt_eval_duration": 0,
                     "eval_count": state["tokens"], "eval_duration": dur,
                     "message": {"role": "assistant", "content": ""}}
            if err:
                final["error"] = err
            s = json.dumps(final) + "\n"
            METRICS["api_out"] += len(s); yield s

        if tools_req:
            if mode == "openai":
                return StreamingResponse(openai_tool_stream(), media_type="text/event-stream")
            return StreamingResponse(ollama_tool_stream(), media_type="application/x-ndjson")

        async def ollama_stream():
            t0 = time.perf_counter_ns()
            done_reason = "stop"
            err = None
            body_key = "message" if mode == "chat" else "response"
            empty_val = {"role": "assistant", "content": ""} if mode == "chat" else ""
            # Model already loaded above (#cold-contract): stream the decode directly — no in-stream
            # load, no keepalive-empty-chunk that could become an empty-200 on a load failure. run()
            # owns + releases the inflight slot.
            try:
                async for piece, reason in run():
                    if piece:
                        val = {"role": "assistant", "content": piece} if mode == "chat" else piece
                        s = json.dumps({"model": model, "created_at": _iso(),
                                        body_key: val, "done": False}) + "\n"
                        METRICS["api_out"] += len(s)
                        yield s
                    if reason:
                        done_reason = reason
            except Exception as exc:  # generation failed mid-stream (model WAS ready); run() frees rec
                err, done_reason = str(exc), "error"
            dur = time.perf_counter_ns() - t0
            final = {"model": model, "created_at": _iso(), "done": True,
                     "done_reason": done_reason, "total_duration": dur, "load_duration": 0,
                     "prompt_eval_count": len(P.get("ids", [])), "prompt_eval_duration": 0,
                     "eval_count": state["tokens"], "eval_duration": dur}
            final[body_key] = empty_val
            if err:
                final["error"] = err
            s = json.dumps(final) + "\n"
            METRICS["api_out"] += len(s)
            yield s

        async def openai_stream():
            # One SSE streamer for /v1/chat/completions AND /v1/completions: chat emits
            # chat.completion.chunk w/ choices[].delta.content; text emits text_completion w/ choices[].text.
            _is_text = (mode == "openai_text")
            _obj = "text_completion" if _is_text else "chat.completion.chunk"
            def _chunk(piece, finish):
                ch = {"index": 0, "finish_reason": finish}
                if _is_text:
                    ch["text"] = piece or ""
                    ch["logprobs"] = None
                else:
                    ch["delta"] = ({"content": piece} if piece else {})
                return {"id": cmpl_id, "object": _obj, "created": created,
                        "model": model, "choices": [ch]}
            finish = "stop"
            # Model already loaded above (#cold-contract): stream directly; run() owns+releases rec.
            try:
                async for piece, reason in run():
                    if piece:
                        s = "data: " + json.dumps(_chunk(piece, None)) + "\n\n"
                        METRICS["api_out"] += len(s)
                        yield s
                    if reason:
                        finish = _map_finish(reason)
            except Exception:
                finish = "stop"
            s = "data: " + json.dumps(_chunk("", finish)) + "\n\n"
            s += "data: [DONE]\n\n"
            METRICS["api_out"] += len(s)
            yield s

        if mode in ("openai", "openai_text"):
            return StreamingResponse(openai_stream(), media_type="text/event-stream")
        return StreamingResponse(ollama_stream(), media_type="application/x-ndjson")

    # ---------- non-streaming ----------
    # Model already loaded above (#cold-contract — shared load gate); collect the full decode.
    t0 = time.perf_counter_ns()
    text = ""
    done_reason = "stop"
    try:
        async for piece, reason in run():
            text += piece
            if reason:
                done_reason = reason
    except Exception as exc:
        import traceback as _tb
        # Surface the cause: a TP forward error (broken all-reduce mesh, shape/quant bug) often has
        # an EMPTY str(exc) (e.g. a dropped peer socket), so {"error": str(exc)} returned "" with no
        # hint. Log repr + traceback to the activity feed (and console) and return the type name.
        log_activity(f"generate {model}: FAILED — {exc!r}")
        print(f"[generate] {model} FAILED: {exc!r}\n{_tb.format_exc()}", flush=True)
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "model": model},
                            status_code=500)
    dur = time.perf_counter_ns() - t0
    n = state["tokens"]
    # #tools: split the raw generation into visible text + structured calls (only when tools were sent)
    tcalls = []
    if tools_req:
        text, tcalls = _extract_tools(text)

    if mode in ("openai", "openai_text"):
        usage = {"prompt_tokens": len(P["ids"]), "completion_tokens": n,
                 "total_tokens": len(P["ids"]) + n}
        if mode == "openai_text":   # OpenAI legacy text completion (no tools)
            payload = {
                "id": cmpl_id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"text": text, "index": 0, "logprobs": None,
                             "finish_reason": _map_finish(done_reason)}],
                "usage": usage}
        else:
            msg = {"role": "assistant", "content": (text or None) if tcalls else text}
            if tcalls:
                msg["tool_calls"] = _openai_tool_calls(tcalls)
            payload = {
                "id": cmpl_id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if tcalls else _map_finish(done_reason)}],
                "usage": usage}
        METRICS["api_out"] += len(json.dumps(payload))
        return JSONResponse(payload)
    out = {"model": model, "created_at": _iso(), "done": True, "done_reason": done_reason,
           "total_duration": dur, "load_duration": 0, "prompt_eval_count": len(P["ids"]),
           "prompt_eval_duration": 0, "eval_count": n, "eval_duration": dur}
    if mode == "chat":
        msg = {"role": "assistant", "content": text}
        if tcalls:
            msg["tool_calls"] = _ollama_tool_calls(tcalls)   # Ollama: arguments as an object
        out["message"] = msg
    else:
        out["response"] = text
    METRICS["api_out"] += len(json.dumps(out))
    return JSONResponse(out)


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
    except Exception as exc:
        _inflight_release(rec)
        # Claude Code reads this on model-selection: surface a clean Anthropic error.
        return JSONResponse({"type": "error",
                             "error": {"type": "not_found_error", "message": str(exc)}},
                            status_code=404)

    # #22 inc 3b/5c: pull any images + audio out so the chat template renders the per-item
    # placeholder (keep_images/keep_audio), then expand + splice their embeds below. Decode/
    # fetch runs OFF the event loop — _decode_image/_decode_audio may blocking-urlopen.
    images = await asyncio.to_thread(_collect_images, body.get("messages"))
    audios = await asyncio.to_thread(_collect_audio, body.get("messages"))
    target_id = MODELS[friendly][0] if friendly in MODELS else friendly
    mm = None
    mrope = None   # #22 inc 4/5c: (3D position_ids [3][q], base) when media embeds are spliced
    hf_tools = _anthropic_tools_to_hf(body.get("tools"))

    def _render_ids(chat):
        """Tokenize a chat with the tools-aware fallback (template throws on tools= for many
        multimodal-remapped checkpoints -> re-render without native tools + a text tool
        instruction; last-ditch: flatten). Shared by the vision and text-only renders."""
        try:
            if hf_tools:
                return _to_id_list(tok.apply_chat_template(chat, tools=hf_tools,
                                   add_generation_prompt=True, tokenize=True))
            return _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True,
                               tokenize=True))
        except Exception as exc:
            print(f"[v1/messages] chat-template failed ({type(exc).__name__}: {exc}); "
                  f"re-rendering without native tools" + (" + tool instruction" if hf_tools else ""))
            chat2 = chat
            if hf_tools:
                instr = _tool_instruction(hf_tools)
                if chat and chat[0].get("role") == "system":
                    chat2 = [{"role": "system", "content": chat[0].get("content", "") + "\n\n" + instr}] + chat[1:]
                else:
                    chat2 = [{"role": "system", "content": instr}] + chat
            try:
                return _to_id_list(tok.apply_chat_template(chat2, add_generation_prompt=True,
                                   tokenize=True))
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
                new_ids, positions, found = _expand_image_placeholders(ids, int(atid), counts)
                if found == len(counts) and len(positions) == n_emb:
                    ids, mm = new_ids, (positions, embeds)
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
                new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts)
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

    max_new = int(body.get("max_tokens", 512) or 512)
    temperature = float(body.get("temperature", 0.0) or 0.0)
    top_p = float(body.get("top_p", 1.0) or 1.0)
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
        produced: list[int] = []
        prev = ""
        try:
          async for tid, reason in engine.generate(friendly, ids, max_new,
                                                   temperature, top_p, False, rec=rec, mm=mm,
                                                   mrope=mrope):
            if tid is not None:
                produced.append(tid)
                state["tokens"] = len(produced)
                METRICS["tokens"] += 1
                text = _decode_visible(tok, produced)
                if text.endswith("�"):
                    continue
                piece, prev = text[len(prev):], text
                if piece:
                    yield piece, None
            if reason:
                text = _decode_visible(tok, produced)
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
            finish = "stop"
            try:
                async for piece, reason in gen_raw():
                    if piece:
                        raw += piece
                        plain, tools = _segment_tools(raw, starts_in_think)
                        if len(plain) <= emitted_plain and len(tools) <= emitted_tools:
                            # tokens flowing but all held back (inside <think> or a
                            # partial tool call) — ping so the client doesn't time out.
                            yield ev("ping", {"type": "ping"})
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
            except Exception as exc:
                # mid-stream failure: close any open block, then signal end
                if text_open:
                    yield ev("content_block_stop", {"type": "content_block_stop",
                                                    "index": text_index})
                    text_open = False
                yield ev("error", {"type": "error",
                                   "error": {"type": "api_error", "message": str(exc)}})
                return
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
    except Exception as exc:
        return JSONResponse({"type": "error",
                             "error": {"type": "api_error", "message": str(exc)}},
                            status_code=500)
    clean, raw_tools = _extract_tools(full)
    content = []
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
