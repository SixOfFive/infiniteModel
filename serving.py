"""serving.py: the request-serving layer relocated from server.py (m4c154 code-split).
Holds _serve (Ollama/OpenAI generate+chat), its private helpers _prepare/_ka_is_unload, and
the shared serve-layer helpers (vision prep, tool-call shaping, knob coercion) BOTH protocol
engines consume. Bodies BYTE-IDENTICAL to the originals. Module globals (engine, registry,
resolve_model_name, _not_found_json, METRICS, formats helpers ...) are injected at startup by
state.bind() -- see state.py. server.py back-imports _serve so the relocated routes_api
resolves it through the published namespace. The Anthropic Messages engine (_serve_anthropic /
_count_tokens_anthropic) lives in serving_anthropic.py now (code-split Inc 3), which imports
this module's helpers leaf-to-leaf. Controller-only leaf; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations




def _transient_gen_exc(exc: BaseException) -> bool:
    """#endpoint-weather: True when a generation failure is CONTENTION-class — the box is busy or
    a data-plane hop flaked — so the client should RETRY with backoff (503/529 + Retry-After),
    never receive a bare 500. Watchdog reclaim surfaces as ConnectionError('gen-stall watchdog
    reclaim'); half-open/dropped data-plane sockets as the OSError family; hop deaths and send
    timeouts as TimeoutError; a shard still held by an orphaned forward as RuntimeError('shard
    busy … re-prefill required'). Anything else (shape/dtype/template bugs) stays a real 500 so
    genuine defects aren't hidden behind retries."""
    # Deterministic filesystem OSError subclasses (a missing/again-missing tokenizer file, a
    # permission error) are NOT contention — a retry can't fix them, so they stay a real 500/error
    # rather than inviting a retry storm. Socket/pipe OSErrors (dropped data plane) fall through to
    # the retryable branch below (ConnectionError/BrokenPipeError are ConnectionError subclasses).
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    s = str(exc).lower()
    return any(t in s for t in ("reclaim", "shard busy", "re-prefill", "connection closed",
                                "connection reset", "timed out", "no control link",
                                # node-drop/recovery races: the model is being replanned and a
                                # retry lands on the recovered pipeline
                                "no model loaded", "not connected", "went down"))


def _stream_fail(exc, rec):
    """#endpoint-weather (streaming): classify a MID-STREAM generation failure for the terminal
    error frame. Returns (retryable, message). A 200 + SSE is already open, so the HTTP status can
    no longer change — the only honest signal left is a typed TERMINAL frame. A watchdog reclaim
    (CancelledError once rec['reclaimed'] is set) and any _transient_gen_exc are RETRYABLE overload;
    everything else is a hard error the client must NOT retry. Each streamer formats this into its
    protocol's idiom (Ollama done_reason=error + retryable; OpenAI error object w/ overloaded_error;
    Anthropic `error` event w/ overloaded_error) — never a clean finish that hides the truncation."""
    if isinstance(exc, asyncio.CancelledError):
        return True, "generation reclaimed under backend contention — retry with backoff"
    if _transient_gen_exc(exc):
        return True, f"transient backend contention ({type(exc).__name__}: {exc}) — retry with backoff"
    return False, f"{type(exc).__name__}: {exc}"


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


def _openai_tool_call_delta_chunks(call: dict, index: int, chunk_size: int = 48) -> list:
    """OpenAI STREAMING shape for ONE tool call, split across SSE chunks the way the real API +
    SDK do: the FIRST delta.tool_calls entry carries id + type + function.name with EMPTY
    arguments, then each subsequent entry streams only a fragment of the arguments JSON string
    (same index, no re-sent id/name). Returns a list of tool_calls-delta payloads (one per chunk).
    Our parser only yields COMPLETE tools, so the arguments are chunked here for wire-fidelity
    rather than truly token-incremental; a single-chunk emit was valid but tripped strict clients
    that expect id/name established before arguments arrive."""
    args = json.dumps(_tool_args(call))
    chunks = [[{"index": index, "id": _anth_id("call"), "type": "function",
                "function": {"name": call.get("name"), "arguments": ""}}]]
    for i in range(0, max(1, len(args)), chunk_size):   # always >=1 arguments frag (even for "{}")
        chunks.append([{"index": index, "function": {"arguments": args[i:i + chunk_size]}}])
    return chunks


def _ollama_tool_calls(calls: list) -> list:
    """[{name,arguments}] -> Ollama tool_calls [{function:{name,arguments:<object>}}]."""
    return [{"function": {"name": c.get("name"), "arguments": _tool_args(c)}} for c in calls]


def _merge_system(chat, text):
    """Prepend `text` into the chat's system prompt (merging with an existing leading system
    message, else inserting one). Returns a NEW list; never mutates the caller's messages."""
    if not text:
        return chat
    if chat and isinstance(chat[0], dict) and chat[0].get("role") == "system":
        return ([{"role": "system", "content": str(chat[0].get("content", "")) + "\n\n" + text}]
                + list(chat[1:]))
    return [{"role": "system", "content": text}] + list(chat)


def _normalize_tool_messages(messages):
    """#tools: normalize the tool-calling REPLY turns Ollama/OpenAI clients send back so the HF
    chat template renders them correctly (the 2nd half of the agent loop):
    - assistant `tool_calls`: OpenAI carries function.arguments as a JSON STRING; HF templates
      expect a dict (Qwen does `| tojson` — a string would double-encode). Parse it, and force the
      {"type":"function","function":{name,arguments}} wrapper shape both conventions map onto.
    - `content: null` (OpenAI's tool-call turns) -> "" — templates do string ops on content.
    - all-TEXT list content (OpenAI content parts) -> joined string — plain-text templates choke on
      lists; lists holding image/audio blocks pass through untouched (the vision path needs them).
    - role "tool" results: coerce non-string content (clients send dict/list JSON) to a JSON string.
    Copy-on-write: returns a new list only when something changed."""
    if not messages:
        return messages
    out, changed = [], False
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        nm = m

        def _mut():
            nonlocal nm, changed
            if nm is m:
                nm = dict(m)
                changed = True
            return nm
        c = m.get("content")
        if c is None:
            _mut()["content"] = ""
        elif isinstance(c, list) and c and all(isinstance(b, dict) and b.get("type") == "text"
                                               for b in c):
            _mut()["content"] = "\n".join(str(b.get("text", "")) for b in c)
        elif m.get("role") == "tool" and not isinstance(c, (str, list)) and c is not None:
            with contextlib.suppress(Exception):
                _mut()["content"] = json.dumps(c)
        tcs = m.get("tool_calls")
        if tcs and isinstance(tcs, list):
            ntc = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
                e = {"type": "function", "function": {"name": fn.get("name"), "arguments": args}}
                if tc.get("id"):
                    e["id"] = tc["id"]
                ntc.append(e)
            if ntc != tcs:
                _mut()["tool_calls"] = ntc
        out.append(nm)
    return out if changed else messages


def _flatten_tool_roles(chat):
    """#tools: rewrite tool-loop turns into plain user/assistant text for templates that can't
    render them natively: assistant tool_calls -> literal <tool_call>{json}</tool_call> text
    (matching the instruction format the model was shown), tool results -> a user turn wrapped in
    <tool_response> tags (Qwen's own convention for tool results). Lossless for the model."""
    out = []
    for m in chat:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            txt = str(m.get("content") or "")
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                txt += (("\n" if txt else "")
                        + "<tool_call>" + json.dumps({"name": fn.get("name"),
                                                      "arguments": fn.get("arguments") or {}})
                        + "</tool_call>")
            out.append({"role": "assistant", "content": txt})
        elif m.get("role") == "tool":
            body = m.get("content")
            if not isinstance(body, str):
                body = json.dumps(body)
            out.append({"role": "user",
                        "content": "<tool_response>\n" + body + "\n</tool_response>"})
        else:
            out.append(m)
    return out


def _thinking_pref(body):
    """OpenAI/Ollama 'should the model think?' control -> enable_thinking bool, or None
    (= leave the chat template's own default). The Anthropic /v1/messages path has its own
    mapping (serving_anthropic.py #qwen3-thinking); this covers the OpenAI + Ollama endpoints,
    where the switch was previously UNREACHABLE (thinking:false / chat_template_kwargs were
    silently ignored). Honors, in order: vLLM/OpenAI-compat chat_template_kwargs.enable_thinking,
    Ollama `think`, then an explicit top-level `enable_thinking`."""
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and isinstance(ctk.get("enable_thinking"), bool):
        return ctk["enable_thinking"]
    for k in ("think", "enable_thinking"):
        if isinstance(body.get(k), bool):
            return body[k]
    return None


def _render_chat_ids(tok, chat, hf_tools, enable_thinking=None):
    """#tools: tokenize a chat, injecting tool definitions when present. `hf_tools` is the
    OpenAI/HF shape ([{type:function, function:{name,description,parameters}}]) that /api/chat and
    /v1/chat/completions already send, passed straight to apply_chat_template(tools=). With NO tools
    (and no tool-loop turns) this is byte-identical to the old plain render. Tiers:
    native tools= -> text tool instruction + flattened tool turns (templates without tool support
    SILENTLY ignore tools= — probe tok.chat_template up front, don't wait for an exception) ->
    flatten-and-retry on any template exception (e.g. a template that rejects the 'tool' role) ->
    a flat text join. So every model SEES the tools + results, whatever its template supports.
    #qwen3-thinking: `enable_thinking` (from _thinking_pref) is threaded into apply_chat_template
    ONLY when the template actually supports the switch, so non-Qwen3 templates are untouched;
    absent (None) keeps the template's own default (byte-identical to the old render)."""
    _tmpl = getattr(tok, "chat_template", None) or ""
    _native = isinstance(_tmpl, str) and ("tools" in _tmpl or "tool_call" in _tmpl)
    _ctk = {}
    if enable_thinking is not None and isinstance(_tmpl, str) and "enable_thinking" in _tmpl:
        _ctk["enable_thinking"] = bool(enable_thinking)
    if hf_tools and not _native:
        # template would silently drop tools= AND likely rejects tool turns: flatten + instruct
        chat = _merge_system(_flatten_tool_roles(chat), _tool_instruction(hf_tools))
        hf_tools = None
        print("[serve] no native template tools; injected a text tool instruction")
    try:
        if hf_tools:
            return _to_id_list(tok.apply_chat_template(chat, tools=hf_tools,
                               add_generation_prompt=True, tokenize=True, **_ctk))
        return _to_id_list(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, **_ctk))
    except Exception as exc:
        print(f"[serve] chat-template failed ({type(exc).__name__}: {exc}); "
              f"flattening tool turns" + (" + tool instruction" if hf_tools else ""))
        chat2 = _flatten_tool_roles(chat)
        if hf_tools:
            chat2 = _merge_system(chat2, _tool_instruction(hf_tools))
        try:
            return _to_id_list(tok.apply_chat_template(chat2, add_generation_prompt=True, tokenize=True, **_ctk))
        except Exception:
            flat = "\n\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in chat2)
            return _to_id_list(tok(flat + "\n\nassistant:"))


def _json_mode_instruction(body):
    """#json-mode: Ollama `format:"json"` / `format:{<JSON Schema>}` and OpenAI
    `response_format:{type:"json_object"|"json_schema"}` -> a best-effort system instruction
    (no constrained decoding yet). Returns None when the request doesn't ask for JSON."""
    want, schema = False, None
    fmt = body.get("format")
    if fmt == "json":
        want = True
    elif isinstance(fmt, dict) and fmt:
        want, schema = True, fmt
    rf = body.get("response_format")
    if isinstance(rf, dict):
        t = rf.get("type")
        if t == "json_object":
            want = True
        elif t == "json_schema":
            want = True
            js = rf.get("json_schema") or {}
            schema = js.get("schema") or js or None
    if not want:
        return None
    s = "Respond with VALID JSON only — no prose, no markdown code fences."
    if schema:
        with contextlib.suppress(Exception):
            s += " The JSON MUST conform to this JSON Schema:\n" + json.dumps(schema)
    return s


def _strip_json_fences(text: str) -> str:
    """#json-mode: models often wrap JSON in ```json fences despite instructions; strip a single
    leading/trailing fence pair so clients that json.loads() the content directly succeed."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _wrap_image_runs(ids: list, itid: int, wrap) -> list:
    """#143 gemma4: reproduce the HF processor's replace_image_token — each bare image
    placeholder becomes boi + placeholder + eoi BEFORE expansion (the chat template renders
    only the bare <|image|>; the model was trained with the bracketed run). No-op when the
    encoder doesn't request wrapping (`wrap` falsy — every other arch) or a placeholder is
    already preceded by boi (template-evolution guard against double-wrapping)."""
    if not wrap:
        return ids
    boi, eoi = int(wrap[0]), int(wrap[1])
    out: list = []
    for tid in ids:
        if tid == itid and (not out or out[-1] != boi):
            out += [boi, tid, eoi]
        else:
            out.append(tid)
    return out


def _pixtral_break_end(tok, grid_rc):
    """#150: resolve the [IMG_BREAK]/[IMG_END] ids from the tokenizer for a Pixtral/Mistral3 image
    run. Returns (grid_rc, break_id, end_id) to pass to _expand_image_placeholders so it inserts
    the trained row structure — or (None, None, None) when there is no grid (every other arch) or
    the tokens don't exist in this vocab, so the expander emits the flat run (unchanged)."""
    if not grid_rc:
        return None, None, None
    try:
        bk = tok.convert_tokens_to_ids("[IMG_BREAK]")
        ei = tok.convert_tokens_to_ids("[IMG_END]")
        # Round-trip: the ids MUST map back to the exact control tokens. A vocab without these
        # tokens resolves them to unk / None / an unrelated id — inserting THAT between rows would
        # corrupt the run — so verify rather than trust the forward lookup (covers unk_token_id
        # being None or 0, which a bare `== unk` guard would miss).
        if not (isinstance(bk, int) and isinstance(ei, int) and bk >= 0 and ei >= 0
                and tok.convert_ids_to_tokens(bk) == "[IMG_BREAK]"
                and tok.convert_ids_to_tokens(ei) == "[IMG_END]"):
            return None, None, None
    except Exception:
        return None, None, None
    return grid_rc, int(bk), int(ei)


async def _prepare_vision(target_id: str, tok, ids: list, images: list):
    """#vl-vision: shared image splice for the OpenAI (/v1/chat/completions) + Ollama (/api/chat)
    serve path — the same machinery /v1/messages uses. Encodes the images, expands the image
    placeholder run in `ids` (injecting the placeholder after BOS if the chat template rendered
    none), and computes mRoPE positions. Returns (ids, mm, mrope); on no images or ANY failure
    returns (ids, None, None) so the caller degrades to a clean text-only prompt."""
    if not images:
        return ids, None, None
    _orig = list(ids)   # pre-mutation snapshot: every failure path degrades from THIS
    try:
        enc_res = await asyncio.to_thread(_encode_images, target_id, images)
        embeds = enc_res.get("image_embeds")
        counts = enc_res.get("counts") or []
        itid = enc_res.get("image_token_id")
        n_emb = int(embeds.shape[0]) if embeds is not None else 0
        if not (itid is not None and n_emb and sum(counts) == n_emb):
            return _orig, None, None
        if ids.count(int(itid)) != len(counts):   # template rendered no placeholder -> inject after BOS
            _bos = getattr(tok, "bos_token_id", None)
            _p = 1 if (ids and _bos is not None and ids[0] == _bos) else 0
            ids = list(ids[:_p]) + [int(itid)] * len(counts) + list(ids[_p:])
        ids = _wrap_image_runs(ids, int(itid), enc_res.get("wrap"))   # gemma4 boi/eoi bracket
        _grc, _bk, _ei = _pixtral_break_end(tok, enc_res.get("grid_rc"))   # #150 Pixtral rows
        new_ids, positions, found = _expand_image_placeholders(ids, int(itid), counts,
                                                               grid_rc=_grc, break_id=_bk, end_id=_ei)
        if not (found == len(counts) and len(positions) == n_emb):
            # Degrade from the SNAPSHOT, stripped of any template-rendered bare placeholders —
            # returning the mutated ids here would serve injected placeholders + boi/eoi wrap
            # tokens with no embeds behind them (review-caught; mirrors /v1/messages'
            # keep_images=False rebuild).
            return [t for t in _orig if t != int(itid)], None, None
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
        return _orig, None, None


def _coerce_knobs(knob) -> dict:
    """#runtime-knobs: build the extended-sampling dict from a lookup fn, coercing every value
    AT PARSE TIME (exactly like temperature/top_p above/below) so a malformed value raises HERE
    — pre-stream, where the #cold-contract handler turns it into a clean 400 — and never inside
    the decode loop after the SSE 200 is already open (a post-stream crash surfaces as the
    silent empty-200 that handler exists to prevent). Also normalizes the seed: llama.cpp /
    SillyTavern / Ollama's own client default send seed=-1 meaning "random", so ANY negative
    seed = unset; values past int64 max are rejected (torch manual_seed would overflow)."""
    sampling = {}
    for k, cast, v in (("top_k", int, knob("top_k")),
                       ("repeat_penalty", float, knob("repeat_penalty", "repetition_penalty")),
                       ("repeat_last_n", int, knob("repeat_last_n")),
                       ("presence_penalty", float, knob("presence_penalty")),
                       ("frequency_penalty", float, knob("frequency_penalty")),
                       ("seed", int, knob("seed"))):
        if v is not None:
            sampling[k] = cast(v)   # ValueError/TypeError -> the caller's clean 400 path
    if "seed" in sampling and sampling["seed"] < 0:
        del sampling["seed"]        # negative = the "random" sentinel -> unset
    if sampling.get("seed", 0) > 2**63 - 1:
        raise ValueError("seed out of range (0 - 2^63-1)")
    return sampling


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
    # #vl-vision: Ollama /api/generate vision — TOP-LEVEL `images:[b64]` next to `prompt`. Convert
    # to a single-user chat (+ optional Ollama `system`) so the template renders the image
    # placeholder and the vision path fires, matching Ollama semantics (generate applies the model
    # template). Text-only generate is untouched (stays the raw-prompt path).
    if messages is None and body.get("images"):
        messages = (([{"role": "system", "content": body["system"]}] if body.get("system") else [])
                    + [{"role": "user", "content": prompt or "", "images": body.get("images")}])
    if messages is not None:
        messages = _normalize_ollama_images(messages)   # #vl-vision: Ollama-native images[] -> blocks
        messages = _normalize_tool_messages(messages)   # #tools: normalize reply-loop turns for the template
        _jm = _json_mode_instruction(body)              # #json-mode: format / response_format best-effort
        if _jm:
            messages = _merge_system(messages, _jm)
        _tch = body.get("tool_choice")
        _tc = body.get("tools") if _tch != "none" else None
        if _tc and (_tch == "required" or isinstance(_tch, dict)):   # OpenAI forced tool choice
            _fname = (_tch.get("function") or {}).get("name") if isinstance(_tch, dict) else None
            messages = _merge_system(messages,
                f"TOOL POLICY: your ENTIRE next reply must be exactly one call to the "
                f"{_fname or 'most appropriate'} tool — no plain text, no explanation, even if a "
                f"tool call seems unnecessary for this message. Use sensible defaults for any "
                f"missing arguments.")
        # #off-loop-tokenize: template render (Jinja) + encode of a large chat is 100-500ms of
        # CPU that previously STALLED the whole event loop (heartbeats, other streams, dashboard).
        # Fast-tokenizer encode is Rust &self (concurrent-safe); the win is loop liveness.
        ids = await asyncio.to_thread(_render_chat_ids, tok, messages, _tc,   # #tools
                                      enable_thinking=_thinking_pref(body))   # #qwen3-thinking
    else:
        _jm = _json_mode_instruction(body)
        ids = await asyncio.to_thread(
            lambda: _to_id_list(tok((prompt or "") + (("\n\n" + _jm) if _jm else ""))))
    # #vl-vision: OpenAI/Ollama image support — extract images from the messages and splice their
    # embeds (was previously only wired into /v1/messages). No-op for text requests / generate mode.
    mm = mrope = None
    if messages is not None:
        _imgs = _collect_images(messages)
        if _imgs:
            ids, mm, mrope = await _prepare_vision(lm.target_id, tok, ids, _imgs)
    opts = body.get("options") or {}
    # #load-temp: the model's per-load DEFAULT temperature applies only when the request sends
    # none (explicit request values — including an explicit 0 — always win).
    _dt = getattr(lm, "default_temperature", None)
    _t = opts.get("temperature", body.get("temperature", None))
    temperature = float(_t) if _t is not None else float(_dt if _dt is not None else 0.0)
    # #min-p: Ollama options.min_p / OpenAI-compat top-level min_p (llama.cpp/vLLM convention);
    # falls back to the model's per-load default. Same precedence rule as temperature.
    _dmp = getattr(lm, "default_min_p", None)
    _mp = opts.get("min_p", body.get("min_p", None))
    min_p = float(_mp) if _mp is not None else float(_dmp if _dmp is not None else 0.0)
    # #runtime-knobs: the extended sampling family — per-request (Ollama options.* / OpenAI
    # top-level; `repetition_penalty` accepted as the vLLM/HF spelling of repeat_penalty, in
    # EITHER location), else the model's runtime default (POST /model_config, stored in
    # lm.sampling_defaults), else off. Same precedence rule as temperature: an explicit request
    # value always wins.
    _sdef = getattr(lm, "sampling_defaults", None) or {}

    def _knob(k, alt=None):
        # layered lookup stepping each level with `is None` — dict.get(k, default) would let an
        # explicit JSON null on the primary spelling shadow the alias next to it
        for _src, _key in ((opts, k), (opts, alt), (body, k), (body, alt)):
            if _key is not None:
                v = _src.get(_key)
                if v is not None:
                    return v
        return _sdef.get(k)
    _tp = opts.get("top_p", body.get("top_p", None))
    top_p = float(_tp) if _tp is not None else float(_sdef.get("top_p") or 1.0)
    _mx = opts.get("num_predict", body.get("max_tokens", None))
    max_new = int(_mx) if _mx is not None else int(_sdef.get("num_predict") or 256)
    sampling = _coerce_knobs(_knob)
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
    return (friendly, tok, ids, temperature, top_p, max_new, stream, speculative, spec_k,
            mm, mrope, min_p, sampling)


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
    # #kv-slots: admission counts TOTAL decode slots — replicas x their per-replica kv_slots
    # (slot_count == replica_count on a fleet with no kv_slots loads, so C=1 is unchanged).
    _slots = engine.slot_count(friendly)
    rec = _inflight_admit(ip, friendly, _slots)
    if rec is None:
        # #queue-depth: overflow beyond (slots + queue_depth) is RETRYABLE — return 429+Retry-After
        # (a fan-out client should back off and retry, not treat it as a hard 503 outage).
        return JSONResponse(
            {"error": f"queue full for '{friendly}': {_slots} slot(s) + "
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
        det = IncrementalDetok(tok)   # #inc-detok: O(tail)/token, byte-identical to full re-decode
        prev = ""
        try:
            async for tid, reason in engine.generate(friendly, ids, max_new, temperature,
                                                     top_p, speculative, rec=rec, spec_k=spec_k,
                                                     mm=P.get("mm"), mrope=P.get("mrope"),
                                                     min_p=P.get("min_p", 0.0),
                                                     sampling=P.get("sampling")):
                if tid is not None:
                    text = det.push(tid)
                    state["tokens"] = det.n
                    METRICS["tokens"] += 1
                    if text.endswith("�"):   # incomplete multi-byte char; wait
                        continue
                    piece, prev = text[len(prev):], text
                    if piece:
                        yield piece, None
                if reason:
                    text = det.current()
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
         P["max_new"], _st, P["speculative"], P["spec_k"], P["mm"], P["mrope"],
         P["min_p"], P["sampling"]) = res
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
        # #at-capacity: EXCEPT a TERMINAL CapacityError (auto-unload off / every resident pinned):
        # retrying can never succeed until an operator unloads something, so the 503 carries code
        # "at_capacity" and NO Retry-After — Retry-After here made an honest retrying client loop
        # forever (the om3nbox 25×503-in-90s transcript).
        _term = isinstance(exc, CapacityError) and getattr(exc, "terminal", False)
        _bad_req = friendly in engine.models
        _code = 400 if _bad_req else 503
        _hdr = {} if (_bad_req or _term) else {"Retry-After": "3"}
        _emsg = f"{type(exc).__name__}: {exc}"
        if mode in ("openai", "openai_text"):
            _err = {"message": _emsg,
                    "type": ("invalid_request_error" if _bad_req else
                             "server_error" if _term else "model_loading")}
            if _term:
                _err["code"] = "at_capacity"
            return JSONResponse({"error": _err}, status_code=_code, headers=_hdr)
        _out = {"error": _emsg, "model": friendly}
        if _term:
            _out["state"] = "at_capacity"
        return JSONResponse(_out, status_code=_code, headers=_hdr)
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
                            # proper incremental deltas: id/name header, then arguments fragments
                            for tc in _openai_tool_call_delta_chunks(tools[emitted_tools], emitted_tools):
                                s = "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model, "choices": [{"index": 0,
                                    "delta": {"tool_calls": tc}, "finish_reason": None}]}) + "\n\n"
                                METRICS["api_out"] += len(s); yield s
                            emitted_tools += 1
                    if reason:
                        finish = reason
            except asyncio.CancelledError:
                # #endpoint-weather (streaming): watchdog reclaim mid-stream. Emit a RETRYABLE
                # error object (not a silent "stop") so a fan-out client backs off; uncancel() so
                # the frame flushes on 3.11+. A genuine client disconnect / user cancel re-raises.
                if not (rec is not None and rec.get("reclaimed")):
                    raise
                with contextlib.suppress(Exception):
                    asyncio.current_task().uncancel()
                s = "data: " + json.dumps({"error": {"message": "generation reclaimed under "
                    "backend contention — retry with backoff", "type": "overloaded_error",
                    "code": "overloaded"}}) + "\n\ndata: [DONE]\n\n"
                METRICS["api_out"] += len(s); yield s; return
            except Exception as exc:
                # #endpoint-weather: a mid-stream failure used to be swallowed and reported as a
                # clean finish — a fan-out/quorum client then counts a TRUNCATED answer as a valid
                # vote. Emit an error object instead (overloaded_error = retryable).
                _retry, _msg = _stream_fail(exc, rec)
                s = "data: " + json.dumps({"error": {"message": _msg,
                    "type": "overloaded_error" if _retry else "api_error"}}) + "\n\ndata: [DONE]\n\n"
                METRICS["api_out"] += len(s); yield s; return
            fr = "tool_calls" if emitted_tools else _map_finish(finish)
            s = "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]}) + "\n\n"
            s += "data: [DONE]\n\n"
            METRICS["api_out"] += len(s); yield s

        async def ollama_tool_stream():
            t0 = time.perf_counter_ns(); done_reason = "stop"; err = None; retryable = False
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
            except asyncio.CancelledError:   # #endpoint-weather: watchdog reclaim mid-stream
                if not (rec is not None and rec.get("reclaimed")):
                    raise
                with contextlib.suppress(Exception):
                    asyncio.current_task().uncancel()
                err, done_reason, retryable = ("generation reclaimed under backend contention — "
                                               "retry with backoff"), "error", True
            except Exception as exc:
                retryable, err = _stream_fail(exc, rec); done_reason = "error"
            dur = time.perf_counter_ns() - t0
            final = {"model": model, "created_at": _iso(), "done": True, "done_reason": done_reason,
                     "total_duration": dur, "load_duration": 0,
                     "prompt_eval_count": len(P.get("ids", [])), "prompt_eval_duration": 0,
                     "eval_count": state["tokens"], "eval_duration": dur,
                     "message": {"role": "assistant", "content": ""}}
            if err:
                final["error"] = err
            if retryable:                    # #endpoint-weather: contention -> client should retry
                final["retryable"] = True
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
            retryable = False
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
            except asyncio.CancelledError:   # #endpoint-weather: watchdog reclaim mid-stream
                if not (rec is not None and rec.get("reclaimed")):
                    raise
                with contextlib.suppress(Exception):
                    asyncio.current_task().uncancel()
                err, done_reason, retryable = ("generation reclaimed under backend contention — "
                                               "retry with backoff"), "error", True
            except Exception as exc:  # generation failed mid-stream (model WAS ready); run() frees rec
                retryable, err = _stream_fail(exc, rec); done_reason = "error"
            dur = time.perf_counter_ns() - t0
            final = {"model": model, "created_at": _iso(), "done": True,
                     "done_reason": done_reason, "total_duration": dur, "load_duration": 0,
                     "prompt_eval_count": len(P.get("ids", [])), "prompt_eval_duration": 0,
                     "eval_count": state["tokens"], "eval_duration": dur}
            final[body_key] = empty_val
            if err:
                final["error"] = err
            if retryable:                    # #endpoint-weather: contention -> client should retry
                final["retryable"] = True
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
            except asyncio.CancelledError:
                # #endpoint-weather: watchdog reclaim mid-stream — emit a RETRYABLE error object,
                # NOT a clean finish_reason:"stop" (which hid the truncation as a valid answer).
                if not (rec is not None and rec.get("reclaimed")):
                    raise
                with contextlib.suppress(Exception):
                    asyncio.current_task().uncancel()
                s = "data: " + json.dumps({"error": {"message": "generation reclaimed under "
                    "backend contention — retry with backoff", "type": "overloaded_error",
                    "code": "overloaded"}}) + "\n\ndata: [DONE]\n\n"
                METRICS["api_out"] += len(s); yield s; return
            except Exception as exc:
                # #endpoint-weather: was `finish="stop"` — a mid-stream failure presented as a
                # clean stop, so a fan-out/quorum client scored a truncated answer as complete.
                # Emit an error object (overloaded_error = retryable) before [DONE] instead.
                _retry, _msg = _stream_fail(exc, rec)
                s = "data: " + json.dumps({"error": {"message": _msg,
                    "type": "overloaded_error" if _retry else "api_error"}}) + "\n\ndata: [DONE]\n\n"
                METRICS["api_out"] += len(s); yield s; return
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
    except asyncio.CancelledError:
        # #endpoint-weather: the gen-stall watchdog reclaimed this request (rec["reclaimed"] set
        # before task.cancel()) — hand the client a clean RETRYABLE 503 instead of an aborted
        # socket. A user /cancel-/terminate (sets only "cancel"), a client disconnect, and server
        # shutdown all re-raise: those must NOT invite a retry.
        if rec is not None and rec.get("reclaimed"):
            log_activity(f"generate {model}: reclaimed under contention — returned retryable 503")
            return JSONResponse({"error": "generation reclaimed under backend contention — "
                                 "endpoint under load, retry with backoff",
                                 "model": model, "retryable": True},
                                status_code=503, headers={"Retry-After": "15"})
        raise
    except Exception as exc:
        import traceback as _tb
        # Surface the cause: a TP forward error (broken all-reduce mesh, shape/quant bug) often has
        # an EMPTY str(exc) (e.g. a dropped peer socket), so {"error": str(exc)} returned "" with no
        # hint. Log repr + traceback to the activity feed (and console) and return the type name.
        log_activity(f"generate {model}: FAILED — {exc!r}")
        print(f"[generate] {model} FAILED: {exc!r}\n{_tb.format_exc()}", flush=True)
        if _transient_gen_exc(exc):
            # #endpoint-weather: contention-class failure (watchdog reclaim, dropped data-plane
            # socket, hop timeout, orphaned-forward shard) -> RETRYABLE 503 + Retry-After, never a
            # bare 500 — a fan-out/agent client must know to back off and ride it out.
            return JSONResponse({"error": f"transient backend contention "
                                 f"({type(exc).__name__}: {exc}) — retry with backoff",
                                 "model": model, "retryable": True},
                                status_code=503, headers={"Retry-After": "15"})
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "model": model},
                            status_code=500)
    dur = time.perf_counter_ns() - t0
    n = state["tokens"]
    # #tools: split the raw generation into visible text + structured calls (only when tools were sent)
    tcalls = []
    if tools_req:
        text, tcalls = _extract_tools(text)
    if _json_mode_instruction(body) is not None:   # #json-mode: strip md fences so json.loads() works
        text = _strip_json_fences(text)

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


# code-split Inc 3: _serve_anthropic + _count_tokens_anthropic live in serving_anthropic.py
# now (bodies VERBATIM; it imports the helpers above leaf-to-leaf).
