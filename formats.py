#!/usr/bin/env python3
"""
InfiniteModel — pure format / helper functions for the controller (server-only leaf module).

Extracted from server.py (#38, step A) to shrink that file. These are SELF-CONTAINED helpers:
Ollama API formatting (tag/model-info), detokenization safety, and the Anthropic Messages API /
tool-calling / mRoPE / token-estimation helpers. None of them touch controller state (engine,
registry, MODELS, METRICS, app routes, …) — they take everything they need as arguments, use only
stdlib + ModelSpec.

This is a controller-only leaf module: it must NEVER ``import server`` (no back-import -> no import
cycle). It is listed in server.py's EXTRA_UPDATE_FILES so the multi-file self-update keeps it in
sync across the fleet, and server.py imports its symbols back via a convergence-bridge import.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Optional

from placement import ModelSpec


# ---------------------------------------------------------------------------
# Ollama-compatible helpers
# ---------------------------------------------------------------------------

def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts if ts else time.time(), timezone.utc).isoformat()


def _digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def _human_params(spec: ModelSpec) -> str:
    p = spec.param_count
    return f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"


def _details(spec: ModelSpec) -> dict:
    return {"parent_model": "", "format": "safetensors", "family": spec.arch,
            "families": [spec.arch], "parameter_size": _human_params(spec),
            "quantization_level": "BF16"}


def _model_info(spec: ModelSpec) -> dict:
    a = spec.arch
    return {
        "general.architecture": a,
        "general.parameter_count": spec.param_count,
        "general.file_type": 32,  # bf16-ish marker
        f"{a}.context_length": spec.max_ctx,
        f"{a}.block_count": spec.num_layers,
        f"{a}.embedding_length": spec.hidden_size,
        f"{a}.feed_forward_length": spec.intermediate_size,
        f"{a}.attention.head_count": spec.num_heads,
        f"{a}.attention.head_count_kv": spec.num_kv_heads,
        f"{a}.attention.key_length": spec.head_dim,
        f"{a}.attention.value_length": spec.head_dim,
        f"{a}.vocab_size": spec.vocab_size,
        "tokenizer.ggml.model": "gpt2",
    }


def _to_id_list(enc) -> list[int]:
    """Coerce a tokenizer result (list, BatchEncoding/dict, tensor, or batched
    nested list) into a flat list[int]."""
    import torch
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    if isinstance(enc, torch.Tensor):
        enc = enc.tolist()
    if enc and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(x) for x in enc]


# ---------------------------------------------------------------------------
# Detokenization safety (#21)
# ---------------------------------------------------------------------------
_DECODE_WARNED = False


def _safe_decode(tok, ids) -> str:
    """Decode token ids to text, surviving ids the tokenizer can't map.

    This model's lm_head vocab can be WIDER than the text tokenizer (a multimodal
    head carries vision/audio placeholder ids), so a stray out-of-range id makes a
    plain ``tok.decode`` raise "list index out of range". We mask those ids at the
    sampler now, but keep this as belt-and-suspenders: on failure, log once (with the
    offending ids) and decode id-by-id, skipping anything out of range/undecodable."""
    global _DECODE_WARNED
    try:
        return tok.decode(ids, skip_special_tokens=True)
    except Exception as exc:
        try:
            ntok = len(tok)
        except Exception:
            ntok = int(getattr(tok, "vocab_size", 0) or 0)
        if not _DECODE_WARNED:
            _DECODE_WARNED = True
            bad = [i for i in ids if ntok and i >= ntok]
            print(f"[decode] {type(exc).__name__}: {exc} — recovering id-by-id; "
                  f"len(tok)={ntok} vocab_size={getattr(tok, 'vocab_size', '?')} "
                  f"out_of_range={bad[:8]} ids_tail={ids[-8:]}")
        out = []
        for i in ids:
            if ntok and i >= ntok:
                continue
            with contextlib.suppress(Exception):
                out.append(tok.decode([i], skip_special_tokens=True))
        return "".join(out)


# ---------------------------------------------------------------------------
# Anthropic Messages API helpers (so Claude Code can use the fleet as a backend)
# ---------------------------------------------------------------------------
_ID_CTR = 0
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
# A model not given native tool framing improvises the format, and qwen3.6-35b-a3b is
# wildly inconsistent — across runs it has emitted ALL of these for the same call:
#   Hermes JSON:   <tool_call>{"name": "f", "arguments": {...}}</tool_call>
#   Claude XML:    <invoke name="f"><parameter name="k">v</parameter></invoke>
#   hybrid XML:    <tool_call><function=f><parameter=k>v</parameter></function></tool_call>
# So parse them ALL liberally rather than passing a clear tool-call intent through as text.
_INVOKE_RE = re.compile(
    r"<invoke\b[^>]*?\bname\s*=\s*[\"']?([^\"'>\s]+)[\"']?[^>]*>(.*?)</invoke>", re.DOTALL)
# <function=name>..</function> or <function name="name">..</function> (the inner block of
# the hybrid form, and a bare form some runs emit without the <tool_call> wrapper).
_FUNC_RE = re.compile(
    r"<function\b(?:\s*=\s*|[^>]*?\bname\s*=\s*)[\"']?([^\"'>\s]+)[\"']?[^>]*>(.*?)</function>",
    re.DOTALL)
_PARAM_RE = re.compile(
    r"<parameter(?:\s+name\s*=\s*[\"']?([^\"'>\s]+)[\"']?|\s*=\s*([^>\s]+))\s*>(.*?)</parameter>",
    re.DOTALL)
_FUNCCALLS_RE = re.compile(r"</?function_calls>", re.IGNORECASE)
# Earliest of any of these in a stream means "a tool call is starting" — stop emitting
# plain text and buffer from here so the markup never leaks to the client as text.
_TOOL_OPENERS = ("<tool_call>", "<invoke", "<function_calls>", "<function")


def _parse_params(body: str) -> dict:
    """Pull <parameter name="k">v</parameter> / <parameter=k>v</parameter> pairs from an
    XML tool-call body. Values are JSON-typed when possible, else kept as trimmed strings."""
    args = {}
    for pm in _PARAM_RE.finditer(body):
        key = pm.group(1) or pm.group(2)
        if not key:
            continue
        raw = pm.group(3).strip()
        try:
            args[key] = json.loads(raw)
        except Exception:
            args[key] = raw
    return args


def _parse_tool_calls(text: str) -> list:
    """Find every tool call in `text`, in any of the formats above. Returns a list of
    {"name","arguments"} dicts (the shape _tool_to_block expects)."""
    calls: list = []
    for m in _TOOLCALL_RE.finditer(text):       # Hermes JSON
        with contextlib.suppress(Exception):
            calls.append(json.loads(m.group(1)))
    for m in _INVOKE_RE.finditer(text):         # Claude <invoke>
        calls.append({"name": m.group(1), "arguments": _parse_params(m.group(2))})
    for m in _FUNC_RE.finditer(text):           # <function=..> (incl. inside <tool_call>)
        calls.append({"name": m.group(1), "arguments": _parse_params(m.group(2))})
    return calls


def _strip_reasoning(text: str) -> str:
    """Remove <think>…</think> reasoning. Also handles reasoning models (Qwen3) whose
    template OPENS <think> in the prompt, so the generation begins mid-thought and only
    emits a dangling </think>: everything up to that first close is reasoning."""
    text = _THINK_RE.sub("", text)
    if "<think>" not in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def _tool_instruction(hf_tools) -> str:
    """A text block that lists the tools and the exact <tool_call> output format — injected
    into the prompt when the model's chat template can't render tools natively (e.g. a
    multimodal-remapped tokenizer whose template throws on `tools=`), so tools aren't lost."""
    lines = ["You can call tools. To call one, output EXACTLY (and nothing else for that call):",
             '<tool_call>{"name": "<tool_name>", "arguments": {<json arguments>}}</tool_call>',
             "Available tools:"]
    for t in (hf_tools or []):
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')} "
                     f"parameters={json.dumps(fn.get('parameters', {}))}")
    return "\n".join(lines)


def _anth_id(prefix: str) -> str:
    global _ID_CTR
    _ID_CTR += 1
    h = hashlib.sha256(f"{prefix}{time.time()}{_ID_CTR}".encode()).hexdigest()[:24]
    return f"{prefix}_{h}"


def _anth_flatten(content) -> str:
    """Flatten an Anthropic content value (str | list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for blk in content:
        if not isinstance(blk, dict):
            parts.append(str(blk))
            continue
        t = blk.get("type")
        if t == "text":
            parts.append(blk.get("text", ""))
        elif t == "tool_result":
            parts.append(_anth_flatten(blk.get("content")))
    return "".join(parts)


def _anthropic_messages_to_chat(system, messages, keep_images: bool = False,
                                keep_audio: bool = False) -> list:
    """Convert Anthropic system+messages into an HF chat-template message list.
    tool_use blocks -> assistant.tool_calls; tool_result blocks -> tool-role msgs.
    keep_images=True: a user message's images become {"type":"image"} content entries (in
    order) so the vision chat template emits one <|image_pad|> placeholder per image (#22
    inc 3b — the controller then expands each to its grid-derived count + splices embeds).
    keep_audio=True: likewise, audio clips become {"type":"audio"} entries so the Omni
    template emits <|audio_bos|><|AUDIO|><|audio_eos|> per clip (#22 inc 5c — expanded to
    its token count + spliced). keep_*=False (default): that modality flattens to a text
    marker (text-only behavior)."""
    chat: list = []
    sys_text = _anth_flatten(system).strip()
    if sys_text:
        chat.append({"role": "system", "content": sys_text})
    for m in (messages or []):
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            chat.append({"role": role, "content": content})
            continue
        if content is None:
            chat.append({"role": role, "content": ""})
            continue
        text_parts, tool_calls, tool_results, n_images, n_audio = [], [], [], 0, 0
        for blk in content:
            if not isinstance(blk, dict):
                text_parts.append(str(blk))
                continue
            t = blk.get("type")
            if t == "text":
                text_parts.append(blk.get("text", ""))
            elif t == "tool_use":
                tool_calls.append({"type": "function", "id": blk.get("id"),
                                   "function": {"name": blk.get("name"),
                                                "arguments": json.dumps(blk.get("input") or {})}})
            elif t == "tool_result":
                tool_results.append(_anth_flatten(blk.get("content")))
            elif t in ("image", "image_url"):
                if keep_images:
                    n_images += 1
                else:
                    text_parts.append("[image omitted: text-only model]")
            elif t in ("audio", "audio_url", "input_audio"):
                if keep_audio:
                    n_audio += 1
                else:
                    text_parts.append("[audio omitted: text-only model]")
        if role == "assistant":
            msg = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            chat.append(msg)
        else:
            for rtext in tool_results:
                chat.append({"role": "tool", "content": rtext})
            txt = "".join(text_parts)
            if (keep_images and n_images) or (keep_audio and n_audio):
                # media entries FIRST (audio then image, in order), then the text -> the
                # template renders the per-clip/per-image placeholder markers, then text.
                # (the "audio" value is a placeholder; the actual waveform is processed
                # separately and its embeds are spliced at the <|AUDIO|> positions.)
                parts = [{"type": "audio", "audio": ""} for _ in range(n_audio)]
                parts += [{"type": "image"} for _ in range(n_images)]
                if txt:
                    parts.append({"type": "text", "text": txt})
                chat.append({"role": "user", "content": parts})
            elif txt.strip() or not tool_results:
                chat.append({"role": "user", "content": txt})
    return chat


def _expand_image_placeholders(ids, image_token_id, counts):
    """The vision chat template emits ONE image_token (image_token_id) per image; the LM
    needs `counts[i]` of them for image i (= its merged-token count). Replace each single
    placeholder with a run of that many and record the absolute positions (which align, in
    order, with the rows of the encoder's image_embeds). Returns (new_ids, positions, found)."""
    out: list[int] = []
    positions: list[int] = []
    ci = 0
    for tid in ids:
        if tid == image_token_id:
            c = counts[ci] if ci < len(counts) else 1
            ci += 1
            start = len(out)
            out.extend([image_token_id] * c)
            positions.extend(range(start, start + c))
        else:
            out.append(tid)
    return out, positions, ci


def _mrope_position_ids(ids, grid_list, image_token_id, merge):
    """#22 inc 4: compute Qwen3-VL 3D (t/h/w) mRoPE position ids for an EXPANDED prompt (one
    run of image_token_id per image). Faithful to transformers get_rope_index/get_vision_
    position_ids (validated against the reference): text tokens advance all 3 dims by 1; an
    image's tokens get t=start, h=start+row, w=start+col over its merged grid (h,w // merge,
    t // 1), and AFTER the image the counter advances by only max(h,w)//merge (positions
    'grow slowly'). The interleaving across freq bands is done by the worker's rotary.
    Returns (position_ids [3][seq] lists, base) where base = max position + 1 (decode start)."""
    t_row: list[int] = []
    h_row: list[int] = []
    w_row: list[int] = []
    cur = 0
    gi = 0
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == image_token_id:
            j = i
            while j < n and ids[j] == image_token_id:
                j += 1
            t, h, w = grid_list[gi] if gi < len(grid_list) else (1, merge, merge)
            gi += 1
            lt, lh, lw = int(t) // 1, int(h) // merge, int(w) // merge
            for ti in range(lt):
                for hi in range(lh):
                    for wi in range(lw):
                        t_row.append(cur + ti)   # time_interval=1
                        h_row.append(cur + hi)
                        w_row.append(cur + wi)
            cur += max(int(h), int(w)) // merge
            i = j
        else:
            j = i
            while j < n and ids[j] != image_token_id:
                j += 1
            for k in range(j - i):
                t_row.append(cur + k)
                h_row.append(cur + k)
                w_row.append(cur + k)
            cur += (j - i)
            i = j
    pos = [t_row, h_row, w_row]
    base = (max(t_row + h_row + w_row) + 1) if t_row else 0
    return pos, base


def _audio_position_ids(seq_len: int):
    """#22 inc 5c: 3D (t/h/w) TMRoPE position ids for an AUDIO-ONLY-plus-text prompt.
    Per Qwen2.5-Omni get_rope_index, the audio branch assigns each audio token
    `arange(audio_len) + st_idx` IDENTICALLY across t/h/w (no spatial split), with
    st_idx = prev_max + 1, while text/bos/eos advance +1 — so the WHOLE sequence is just
    sequential 0..seq-1 broadcast to all 3 dims (unlike images, audio positions do NOT grow
    slowly). Returns (position_ids [3][seq], base) where base = seq_len (decode start)."""
    row = list(range(seq_len))
    return [row, list(row), list(row)], seq_len


def _anthropic_tools_to_hf(tools):
    """Anthropic tool defs -> OpenAI/HF function-tool defs for apply_chat_template."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name"),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    return out or None


def _tool_to_block(tb: dict) -> dict:
    """A parsed <tool_call> JSON object -> an Anthropic tool_use content block."""
    args = tb.get("arguments")
    if not isinstance(args, dict):
        args = tb.get("parameters") if isinstance(tb.get("parameters"), dict) else {}
    return {"type": "tool_use", "id": _anth_id("toolu"),
            "name": tb.get("name"), "input": args}


def _extract_tools(text: str):
    """Split a full generation into (clean_text, [tool-call dicts]). Strips reasoning
    FIRST (so tool markup inside <think> isn't taken as a real call), then pulls out
    every tool call in any supported format."""
    no_think = _strip_reasoning(text)
    tools = _parse_tool_calls(no_think)
    clean = _TOOLCALL_RE.sub("", no_think)
    clean = _INVOKE_RE.sub("", clean)
    clean = _FUNC_RE.sub("", clean)
    clean = _FUNCCALLS_RE.sub("", clean)
    clean = clean.replace(_TOOL_OPEN, "").replace(_TOOL_CLOSE, "")   # leftover wrapper tags
    return clean.strip(), tools


def _partial_suffix_len(s: str, tag: str) -> int:
    """Longest suffix of s that is a proper prefix of tag — a possibly-incomplete
    opening tag we hold back rather than stream as plain text."""
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if s[-k:] == tag[:k]:
            return k
    return 0


def _segment_tools(raw: str, starts_in_think: bool = False):
    """Prefix-stable split of streamed raw text into (visible_plain, completed_tools).
    Reasoning is stripped and tool markup held back until complete, so neither leaks to
    the client as text. `starts_in_think` (the template opened <think> in the prompt, so
    the model begins mid-thought) holds EVERYTHING back until the closing </think>.
    Visible plain only ever grows; tools are every COMPLETE call after the first opener."""
    s = raw
    if starts_in_think:                     # began inside reasoning -> hold until it closes
        c = s.find("</think>")
        if c == -1:
            return "", []
        s = s[c + len("</think>"):]
    s = _THINK_RE.sub("", s)                # drop any finished <think>…</think> pairs
    ti = s.rfind("<think>")                 # unclosed reasoning -> hold back from it on
    if ti != -1 and "</think>" not in s[ti:]:
        s = s[:ti]
    hits = [s.find(o) for o in _TOOL_OPENERS]
    hits = [i for i in hits if i != -1]
    if not hits:                            # no opener yet — stream plain, hold any partial
        hold = max((_partial_suffix_len(s, o) for o in _TOOL_OPENERS + ("<think>", "</think>")),
                   default=0)
        return s[:len(s) - hold], []
    cut = min(hits)                         # stable: an opener's position doesn't move
    return s[:cut], _parse_tool_calls(s[cut:])


def _estimate_tokens(chat: list) -> int:
    chars = sum(len(m.get("content", "") or "") for m in chat)
    return max(1, chars // 4)
