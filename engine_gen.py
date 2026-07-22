"""EngineGenMixin: relocated Engine methods (m4c152 code-split). BODIES ARE BYTE-IDENTICAL
to the originals in server.py; their module globals (registry, log_activity, ModelSpec,
ENGINE_CONFIG …) are injected at startup by state.bind() — see state.py. Composed back
into the live class via ``class Engine(EngineGenMixin, …)`` in server.py, so ``self.*`` resolves
across all mixins by MRO. Controller-only leaf module; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def _bidir_spans_from_positions(positions):
    """#gemma4-bidir: group the image-embed slot positions (row-major, ascending) into contiguous
    half-open [start,end) runs — one per image block. Mirrors the reference's contiguous vision
    runs (get_block_sequence_ids_for_mask); the shard OR's a bidirectional overlay within each run.
    Returns a list of (start,end) tuples, or [] when there are no positions."""
    spans: list = []
    for p in positions or []:
        p = int(p)
        if spans and p == spans[-1][1]:
            spans[-1][1] = p + 1
        else:
            spans.append([p, p + 1])
    return [(s, e) for s, e in spans]


def _wire_topk_k() -> int:
    """#logits-diet: K for the top-K sampled-decode wire mode — the head ships only the top-K
    (values, ids) candidates instead of the full-vocab row. INFINITEMODEL_TOPK_WIRE (controller
    env), default 4096; 0 = full row = diet OFF for sampled requests (greedy/spec argmax mode is
    unaffected — it is bit-exact and has no truncation to opt out of). Top-4096 covers >>0.9999
    of the softmax mass on realistic peaked LM rows (spot-checked in scratch_logits_diet_test.py;
    a pathologically FLAT distribution is the case 0 exists for)."""
    import os
    try:
        return max(0, int(os.environ.get("INFINITEMODEL_TOPK_WIRE", "4096")))
    except Exception:
        return 4096


def _pipefill_conf() -> int:
    """#pipefill: chunk length (tokens) for the cross-stage pipelined prefill, or 0 = disabled.
    INFINITEMODEL_PIPEFILL (controller env, read per request like INFINITEMODEL_TOPK_WIRE):
    '0'/'off'/'false' -> opt-out (classic one-shot prefill everywhere); a positive int -> the
    chunk length; unset/anything else -> 2048. The default deliberately MATCHES
    INFINITEMODEL_PREFILL_CHUNK's default so each pipefill frame runs exactly one already-
    validated intra-stage chunk pass (shard_forward #prefill-chunk builds the identical
    [cl, cache_start+cl] mask for both) — the per-chunk compute and numerics are the SAME ops
    the intra-stage loop was validated with, just driven from the controller. Clamped to >=256
    so a typo can never flood the data plane with hundreds of tiny frames."""
    import os
    v = os.environ.get("INFINITEMODEL_PIPEFILL", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return 0
    try:
        n = int(v)
    except ValueError:
        return 2048
    return max(256, n) if n > 0 else 0


def _prefix_kv_min() -> int:
    """#prefix-kv: minimum shared-prefix length (tokens) worth a crop+resume, or 0 = feature
    disabled. Two controller envs, read per request like the other knobs here:
    INFINITEMODEL_PREFIX_KV — '0'/'off'/'false'/'no' -> hard opt-out (default ON; the
    _prefix_kv_ok gates already require the modern 'pipefill' worker caps, so a mixed/stale
    fleet self-gates to the classic full prefill);
    INFINITEMODEL_PREFIX_MIN — the reuse threshold, default 1024 tokens; clamped >=16 so a
    typo can never make every tiny retry crop a live cache for a negligible win."""
    import os
    v = os.environ.get("INFINITEMODEL_PREFIX_KV", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return 0
    try:
        n = int(os.environ.get("INFINITEMODEL_PREFIX_MIN", "1024"))
    except ValueError:
        return 1024
    return max(16, n)


def _lcp_len(a, b) -> int:
    """#kv-slots: longest-common-prefix length of two id lists (C-speed fast path for the
    every-turn agent shape — pure extension/retry — mirroring _prefill_reuse's own LCP)."""
    if not a or not b:
        return 0
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class _SlotLease:
    """#kv-slots: one generation's decode-slot lease on a replica — the async CM that replaced
    the whole-generation ``async with model.lock``.

    C == 1 (kv_slots absent/1 — every legacy load): acquires model.lock EXACTLY like the old
    ``async with`` (the lock OBJECT is captured at enter, so the gen-stall watchdog's
    lock-swap reclaim keeps today's semantics: the orphan releases the OLD object). slot 0,
    zero new state — byte-identical serialization.

    C > 1: acquires one permit of model.slot_sem (asyncio.Semaphore(C), FIFO like Lock) and
    picks a slot id from model.slot_free — preferring the free slot whose per-slot #prefix-kv
    record (model.kv_ids_slots) has the LONGEST common prefix with THIS request's prompt, so
    a multi-turn client tends to land back on the slot that already holds its context (the
    'longest LCP wins, its slot is reused' rule). The lease owns the slot for the generation's
    whole lifetime (prefill + decode + crops all carry its id). Ownership is tokenized
    (model.slot_owner[slot] is THIS lease's token): the per-slot gen-stall watchdog reclaim
    frees a wedged slot by swapping the token and releasing the semaphore ITSELF — the
    orphaned gen's __aexit__ then sees the mismatch and releases NOTHING (no double-release,
    no double-free of the slot id). Slot-pool exhaustion (all C busy -> this request queues)
    logs one activity line per model per minute, max."""

    def __init__(self, model, prompt_ids=None):
        self.model = model
        self.prompt_ids = prompt_ids or []
        self.slot = 0
        self.token = None          # C>1 ownership token (None on the C=1 lock path)
        self._lock = None          # C=1: the captured lock object
        self._sem = None           # C>1: the captured semaphore

    async def __aenter__(self):
        m = self.model
        C = int(getattr(m, "kv_slots", 1) or 1)
        if C <= 1 or getattr(m, "slot_sem", None) is None:
            self._lock = m.lock
            await self._lock.acquire()
            return 0
        sem = m.slot_sem
        if sem.locked():
            # all C slots busy -> queueing. Observability (once per minute per model, max):
            # without this line slot-pool exhaustion is invisible in telemetry.
            now = time.time()
            if now - (getattr(m, "_slot_full_log_ts", 0.0) or 0.0) > 60.0:
                m._slot_full_log_ts = now
                with contextlib.suppress(Exception):
                    log_activity(f"{m.friendly}: all {C} kv-slots busy — request queued "
                                 f"(slot pool exhausted)")
        await sem.acquire()
        self._sem = sem
        free = m.slot_free
        # longest-LCP slot choice over the FREE slots' per-slot #prefix-kv records
        recs = getattr(m, "kv_ids_slots", None) or {}
        best, best_l = free[0], -1
        for s in free:
            l = _lcp_len(recs.get(s), self.prompt_ids)
            if l > best_l:
                best, best_l = s, l
        free.remove(best)
        self.slot = best
        self.token = object()
        m.slot_owner[best] = self.token
        m.slots_active = int(getattr(m, "slots_active", 0) or 0) + 1
        return best

    async def __aexit__(self, *exc):
        m = self.model
        if self._sem is None:
            if self._lock is not None:
                with contextlib.suppress(Exception):
                    self._lock.release()
            return False
        # return the slot ONLY if the watchdog hasn't already reclaimed it (token match)
        if m.slot_owner.get(self.slot) is self.token:
            m.slot_owner.pop(self.slot, None)
            m.slot_state.pop(self.slot, None)
            m.slot_free.append(self.slot)
            m.slots_active = max(0, int(getattr(m, "slots_active", 0) or 0) - 1)
            self._sem.release()
        return False


class EngineGenMixin:

    # ---- #kv-slots: per-slot #prefix-kv record access -------------------------------------
    # C == 1 keeps the classic single model.kv_ids attribute (every existing writer/reader —
    # spec decode, capture paths, the C=1 watchdog reclaim — stays byte-identical); C > 1
    # stores one record per slot in model.kv_ids_slots so streams never cross-contaminate.

    def _kv_rec_get(self, model, slot: int = 0):
        if int(getattr(model, "kv_slots", 1) or 1) <= 1:
            return getattr(model, "kv_ids", None)
        return (getattr(model, "kv_ids_slots", None) or {}).get(int(slot))

    def _kv_rec_set(self, model, slot: int, ids) -> None:
        if int(getattr(model, "kv_slots", 1) or 1) <= 1:
            model.kv_ids = ids
            return
        d = getattr(model, "kv_ids_slots", None)
        if d is None:
            d = model.kv_ids_slots = {}
        d[int(slot)] = ids

    def _kv_rec_null(self, model, slot: int = 0) -> None:
        self._kv_rec_set(model, slot, None)

    def _pending_slot_map(self) -> dict:
        """rid -> slot map (parallel to pending_friendly) so the per-slot gen-stall watchdog can
        fail ONLY the wedged slot's in-flight futures. Lazy so mixin test harnesses need no init."""
        d = getattr(self, "pending_slot", None)
        if d is None:
            d = self.pending_slot = {}
        return d

    async def embed(self, friendly: str, input_ids, attention_mask) -> list:
        """Run one encoder forward on `friendly`'s single node and return the pooled, L2-normed
        sentence vectors as a list of float lists. Mirrors _send: pack ids+mask into ONE
        two-tensor frame, await the worker's single-tensor 'embedding' reply via self.pending."""
        model = self.models[friendly]
        target = model.target_id
        async with model.lock:
            if model.stage0_writer is None:
                raise RuntimeError("embedding model not connected")
            # usage bookkeeping: embeddings have no decode rate, so only these counters mark them as
            # used — they feed the request-activity graph (#models-usage-graph), the LRU last_used,
            # and the dashboard "requests served" count (all previously stuck at 0 for embedders,
            # since only the text-generation path bumped active/req_total/last_used).
            model.last_used = time.time()
            model.active += 1
            loop = asyncio.get_event_loop()
            rid = self.next_req()
            ids_meta, ids_raw = _pack_tensor(input_ids)
            mask_meta, mask_raw = _pack_tensor(attention_mask)
            fut = loop.create_future()
            self.pending[rid] = fut
            self.pending_model[rid] = target
            try:
                hdr = {"req_id": rid, "model_id": target, "kind": "embed",
                       **ids_meta, "ids_nbytes": len(ids_raw), "mask_meta": mask_meta}
                nbytes = await _write_frame(model.stage0_writer, hdr, ids_raw + mask_raw)
                net_account(self._stage0_id(model), to_node=nbytes)   # controller -> node
                vecs = await asyncio.wait_for(fut, timeout=GEN_TIMEOUT_S)
                model.req_total += 1
                return vecs.tolist()
            finally:
                model.active = max(0, model.active - 1)
                self.pending.pop(rid, None)
                self.pending_model.pop(rid, None)

    @staticmethod
    def _eos_ids(tok, model_dir: str = "") -> set:
        """Every token id that must STOP generation. Beyond the tokenizer's own eos_token_id we
        MUST include (a) the checkpoint's declared eos_token_id — often a LIST (Gemma, Llama-3) —
        and (b) the family end-of-turn markers, because many chat templates end the assistant turn
        with a token that is NOT the tokenizer eos: Gemma uses <end_of_turn> (106), Llama-3 uses
        <|eot_id|>, ChatML uses <|im_end|>. If those aren't registered, the model emits one, we
        don't recognize it as a stop, generation runs to max_new, the marker leaks into the text
        ("thought"/"off"/"<end_of_turn>") and in streaming the whole answer repeats. (#stop-eos) """
        import os, json as _json
        ids = set()
        if getattr(tok, "eos_token_id", None) is not None:
            ids.add(int(tok.eos_token_id))
        # Authoritative gen-time eos from the checkpoint (int OR list). generation_config wins.
        md = model_dir or getattr(tok, "name_or_path", "") or ""
        for fn in ("generation_config.json", "config.json"):
            if not md:
                break
            with contextlib.suppress(Exception):
                with open(os.path.join(md, fn), encoding="utf-8") as fh:
                    ev = _json.load(fh).get("eos_token_id")
                if isinstance(ev, int):
                    ids.add(ev)
                elif isinstance(ev, (list, tuple)):
                    ids.update(int(x) for x in ev if isinstance(x, int))
        # Family end-of-turn markers, resolved through THIS tokenizer's vocab. Guard against the
        # unk id (an absent token resolves to <unk> on many tokenizers — adding it would stop on
        # every unknown token), and against accidentally adding a normal-word id.
        unk = getattr(tok, "unk_token_id", None)
        for t in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>",
                  "<|endoftext|>", "<|end_of_text|>", "</s>"):
            with contextlib.suppress(Exception):
                tid = tok.convert_tokens_to_ids(t)
                if tid is not None and tid >= 0 and tid != unk:
                    ids.add(int(tid))
        # OpenAI-harmony (gpt-oss): <|end|> ends a CHANNEL (the analysis CoT), NOT the assistant
        # turn — the turn ends at <|return|> (or <|call|> for tools), already supplied by
        # generation_config.eos_token_id above. If <|end|> stays a stop, gen halts after the analysis
        # channel and never emits the final answer. So on a harmony tokenizer, DROP <|end|> from the
        # stops (and ensure <|return|> is in). Detected by the harmony marker tokens. (#harmony)
        with contextlib.suppress(Exception):
            ch = tok.convert_tokens_to_ids("<|channel|>")
            ret = tok.convert_tokens_to_ids("<|return|>")
            if ch not in (None, unk) and ch >= 0 and ret not in (None, unk) and ret >= 0:
                ids.discard(int(tok.convert_tokens_to_ids("<|end|>")))
                ids.add(int(ret))
        return {i for i in ids if isinstance(i, int) and i >= 0}

    def _sample(self, row, temperature: float, top_p: float, min_p: float = 0.0,
                top_k: int = 0, gen=None) -> int:
        import torch
        row = row.float()
        if not temperature or temperature <= 0:
            return int(row.argmax())
        probs = torch.softmax(row / temperature, dim=-1)
        # #min-p: confidence-ADAPTIVE floor — drop any token whose prob is below min_p x the top
        # token's prob (llama.cpp/vLLM convention: applied on the temperature-scaled probs, before
        # top_p). When the model is confident the filter is strict (few survivors); when it's
        # uncertain (flat distribution) it loosens up — which is why it pairs with HIGH temperature:
        # temp flattens the distribution and lets junk tokens in, min_p cuts them first. Useful
        # range ~0.05-0.1 at temperature >= 1.0.
        if min_p and 0 < min_p <= 1:
            keep = probs >= (min_p * probs.max())
            probs = probs * keep
            probs = probs / probs.sum()
        # #runtime-knobs top_k: keep only the k most-probable tokens (0 = off). After min-p (both
        # are absolute filters), before top-p — the usual llama.cpp chain. top_k=1 forces the top
        # token regardless of temperature (the determinism probe used by the validation battery).
        if top_k and 0 < top_k < probs.shape[-1]:
            kv, ki = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, ki, kv)
            probs = probs / probs.sum()
        if top_p and 0 < top_p < 1:
            sp, idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sp, 0)
            keep = cdf - sp <= top_p
            sp = sp * keep
            sp = sp / sp.sum()
            return int(idx[int(torch.multinomial(sp, 1, generator=gen))])
        return int(torch.multinomial(probs, 1, generator=gen))

    def _penalized(self, row, prompt_ids, out_ids, sp):
        """#runtime-knobs: apply repetition penalties to a logits row BEFORE sampling — pre-argmax,
        so they steer greedy decode too. Conventions match vLLM/llama.cpp: `repeat_penalty`
        (multiplicative; positive logits divided, negative multiplied) looks at the last
        `repeat_last_n` tokens of PROMPT+OUTPUT (default 64; -1 = everything; 0 = off);
        `presence_penalty` (flat, once per distinct token) and `frequency_penalty` (x occurrence
        count) are OpenAI-style additive penalties over the OUTPUT so far only. Returns the row
        untouched when every knob is off; clones before mutating (the caller may hand a view)."""
        import torch
        rp = float(sp.get("repeat_penalty") or 1.0)
        pp = float(sp.get("presence_penalty") or 0.0)
        fp = float(sp.get("frequency_penalty") or 0.0)
        if (rp <= 0 or rp == 1.0) and not pp and not fp:
            return row
        row = row.float().clone()
        if rp > 0 and rp != 1.0:
            n = sp.get("repeat_last_n")
            n = 64 if n is None else int(n)
            hist = prompt_ids + out_ids
            window = hist if n < 0 else (hist[-n:] if n else [])
            uniq = list({int(i) for i in window})
            if uniq:
                t = torch.tensor(uniq, dtype=torch.long)
                vals = row[t]
                row[t] = torch.where(vals > 0, vals / rp, vals * rp)
        if (pp or fp) and out_ids:
            from collections import Counter
            cnt = Counter(int(i) for i in out_ids)
            t = torch.tensor(list(cnt.keys()), dtype=torch.long)
            c = torch.tensor([float(v) for v in cnt.values()], dtype=row.dtype)
            row[t] = row[t] - pp - fp * c
        return row

    async def _freshen_stage0(self, model: LoadedModel, force: bool = False) -> None:
        """#stage0-stale-reconnect: rebuild model.stage0_writer FRESH if it may be stale. The
        controller's stage0 conn is opened at LOAD then sits IDLE between requests; an idle socket
        can go SILENTLY half-open (the write SUCCEEDS but the bytes vanish -> no logits -> ~600s
        GEN_TIMEOUT hang — the 'loaded but never replies' bug). Reconnecting a fresh socket at
        generate START (when idle past STAGE0_STALE_S) gives every request a hot, proven path —
        the SAME lazy-fresh-connect the workers already use for their next hop (client.py
        _send_next). force=True rebuilds unconditionally (used by _send after a write FAILED).
        Cheap: a TCP connect is ~ms vs a multi-token generation."""
        if not model.stage0_dial:
            return   # no saved dial target (shouldn't happen post-load) -> leave as-is
        now = time.time()
        if (not force and model.stage0_writer is not None
                and (now - model.last_send_ts) <= STAGE0_STALE_S):
            return   # used recently -> the connection is hot, reuse it (no churn on busy models)
        old = model.stage0_writer
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        model.stage0_writer = await self._connect_retry(*model.stage0_dial)
        model.last_send_ts = now
        with contextlib.suppress(Exception):
            print(f"[data] freshened stage0 conn for {model.friendly} -> "
                  f"{model.stage0_dial[0]}:{model.stage0_dial[1]} "
                  f"({'write failed' if force else 'idle'})", flush=True)

    def _wire_ntensor_ok(self, model) -> bool:
        """#wire-caps: may the controller request the #ntensor-manifest return frame from
        `model`'s chain? True only when EVERY stage node advertised the 'ntensor' cap at
        registration (registry.node_caps) — an old INTERMEDIATE worker rebuilds the next-hop
        header and would silently DROP the flag, and an old HEAD ignores it, so all-or-nothing
        is the only correct gate — AND this controller's own wire.py provides the unpacker
        (_unpack_ntensor is None during a per-file self-update convergence window; the
        controller counts as a wire peer too). Re-evaluated per _send, so a replan/adoption
        that swaps nodes is picked up automatically."""
        if globals().get("_unpack_ntensor") is None:
            return False
        ids = getattr(model, "stage_node_ids", None) or []
        return bool(ids) and all("ntensor" in registry.node_caps(nid) for nid in ids)

    def _wire_diet_ok(self, model) -> bool:
        """#logits-diet: may _send ask `model`'s head for a REDUCED reply (argmax token ids /
        top-K candidates) instead of the full-vocab logits row? Requires the full
        #ntensor-manifest gate PLUS:
        - every chain node advertising 'ntdiet' (#wire-caps): a 14ae638-era node advertises
          'ntensor' but its worker_net ignores nt_mode (harmless — it replies full logits — but
          pointless), and an intermediate of that vintage DROPS the nt_* keys when it rebuilds
          the next-hop header, so all-or-nothing on the newer cap is the honest gate;
        - tp_size == 1: the TP positional broadcast does not carry the directive to peer ranks.
          Rank 0's post-all-reduce head reduction is a purely local op and LIKELY safe, but
          that's unvalidated on a live mesh — stage 1 EXCLUDES TP models rather than assume.
        Even when this returns True the reply may still be a full row (a node whose wire.py
        outran its worker_net/shard_forward during per-file self-update convergence advertises
        the cap but can't compute the reduction) — every caller detects the reply type and
        falls back to the legacy row path, so the diet can only ever degrade to status quo."""
        if not self._wire_ntensor_ok(model):
            return False
        if int(getattr(model, "tp_size", 1) or 1) != 1:
            return False
        ids = getattr(model, "stage_node_ids", None) or []
        return bool(ids) and all("ntdiet" in registry.node_caps(nid) for nid in ids)

    def _pipefill_arch_ok(self, model) -> bool:
        """#pipefill: is this model's ARCHITECTURE safe for cross-stage chunked prefill? Must
        mirror shard_forward's own intra-stage do_chunk exclusion list (the audit-binding set):
        _per_type (Gemma4 sliding/full split), _hybrid (Gated-DeltaNet linear attention — a
        chunked pass diverges from one-shot, the same reason _has_mtp gates hybrid off), _omni
        and _mrope3d (Omni/Qwen2.5-VL: the shard feeds the rotary 3D positions even for plain
        text; the intra-stage loop excludes them, so the cross-stage split does too);
        bidir_spans is request-scoped and covered by the mm-is-None gate in _pipefill_chunk.
        Detected from the checkpoint's LOCAL config.json (a loaded model's weights streamed
        from THIS controller's models/ dir, so the read is zero-network by construction — the
        path is computed directly rather than via _controller_model_dir, whose populate path
        could touch the network). CONSERVATIVE + honest: any sniff failure, and ANY
        layer_types presence (that field is how both hybrid AND Gemma4's per-type rotary
        announce themselves; a hypothetical all-full_attention layer_types model is gated off
        too rather than guessing at its rotary), means False -> one-shot prefill. Cached on
        the LoadedModel (immutable per load)."""
        ok = getattr(model, "_pipefill_arch", None)
        if ok is not None:
            return ok
        ok = False
        try:
            with open(os.path.join(MODELS_DIR, _safe_name(model.target_id), "config.json"),
                      encoding="utf-8") as fh:
                cfg = json.load(fh)
            tc = cfg.get("text_config", cfg)
            _mt = str(cfg.get("model_type") or tc.get("model_type") or "").lower()
            ok = (not (tc.get("layer_types") or [])                    # hybrid / per-type
                  and cfg.get("thinker_config") is None                # _omni (multimodal.py:is_omni)
                  and _mt not in ("qwen2_5_vl", "qwen2_5_vl_text"))    # _mrope3d (client.py twin)
        except Exception:
            ok = False
        model._pipefill_arch = ok
        return ok

    def _pipefill_chunk(self, model, q: int, mm, position_ids) -> int:
        """#pipefill master gate: chunk length (tokens) for a chunked PIPELINED prefill of a
        q-token prompt, or 0 = keep the classic one-shot single-frame prefill. All gates are
        conservative — any doubt keeps the pre-pipefill behavior byte-identical:
        - INFINITEMODEL_PIPEFILL opt-out / chunk length (_pipefill_conf; 0 = off);
        - q must EXCEED one chunk (a short prompt's single frame is already optimal);
        - text-only this stage: mm embeds (splice + #gemma4-bidir spans key off ONE prefill
          frame's req_id/positions) and explicit mrope position_ids are EXCLUDED — the mm
          companion pairing and absolute-position layout were not built for a split prompt;
        - pipeline only (tp_size == 1): the TP rank-0 broadcast path is per-forward and
          unvalidated for chunk bursts;
        - MULTI-stage only (>=2 stages): on a single stage there is nothing to overlap — the
          intra-stage #prefill-chunk loop already covers mask memory, and chunking the WIRE
          would only add per-frame overhead (single-stage placements keep the untouched path);
        - every chain node advertises the 'pipefill' wire cap (#wire-caps all-or-nothing);
        - the architecture is chunk-safe (_pipefill_arch_ok, mirrors shard do_chunk)."""
        cstep = _pipefill_conf()
        if not cstep or q <= cstep:
            return 0
        if mm is not None or position_ids is not None:
            return 0
        if int(getattr(model, "tp_size", 1) or 1) != 1:
            return 0
        ids = getattr(model, "stage_node_ids", None) or []
        if len(ids) < 2:
            return 0
        if not all("pipefill" in registry.node_caps(nid) for nid in ids):
            return 0
        if not self._pipefill_arch_ok(model):
            return 0
        return cstep

    def _prefix_kv_ok(self, model) -> bool:
        """#prefix-kv master gate: may a new request RESUME this replica's resident KV (crop to
        the shared prefix + suffix-only prefill) instead of re-prefilling from position 0? All
        binding audit caveats, all conservative — any doubt keeps the classic full prefill:
        - tp_size == 1: the TP positional broadcast path was never validated for crop+resume
          (and TP peers don't see crop frames at all);
        - kv_offload off: the worker's OffloadedCache fallback can silently drop/rebuild
          self.kv mid-request, so a controller-side record of its contents can't be trusted;
        - kv_quant == none (#172): TurboQuantCache.crop exists, but a cross-request resume over
          the quantized-prefix + bf16-residual-window split is unvalidated — excluded;
        - every chain node advertises 'pipefill' (#wire-caps all-or-nothing): those workers run
          the reviewed sequential-inbound contract the suffix chunk burst leans on (reuses that
          machinery — no new cap);
        - _pipefill_arch_ok: excludes hybrid Gated-DeltaNet (a crop cannot roll back linear-
          attention recurrent state — documented not bit-exact), Gemma4 per-type sliding KV (a
          sliding layer has already DISCARDED early tokens, so cropping below the window edge is
          unrecoverable), omni and qwen2_5_vl mrope (absolute 3D-position layouts were never
          built for a split/resumed prompt)."""
        if int(getattr(model, "tp_size", 1) or 1) != 1:
            return False
        if getattr(model, "kv_offload", False):
            return False
        if (getattr(model, "kv_quant", "none") or "none") != "none":
            return False
        ids = getattr(model, "stage_node_ids", None) or []
        if not ids or not all("pipefill" in registry.node_caps(nid) for nid in ids):
            return False
        return self._pipefill_arch_ok(model)

    async def _prefill_reuse(self, model: LoadedModel, prompt_ids: list, mm=None,
                             position_ids=None, nt_mode=None, nt_clip: int = 0, nt_k: int = 0,
                             slot: int = 0):
        """#prefix-kv: prefill dispatcher with CROSS-REQUEST prefix reuse (audit finding 29).

        THE DEFECT this fixes: every request re-prefilled its ENTIRE prompt from position 0,
        yet multi-turn agent workloads (Claude Code via /v1/messages, agent loops, quorum
        re-asks) send prompt_N = prompt_{N-1} + answer_{N-1} + new turn — the shared tens-of-
        thousands-of-token prefix was recomputed every turn (tens of seconds to minutes of
        TTFT, worst on the bandwidth-bound Strix box) while the SAME tokens' KV sat resident
        on every stage (workers keep self.kv between generations; nothing frees it at gen end).

        MECHANISM (single-cache resume — THE one live cache per replica; no KV slots yet): the
        decode paths publish model.kv_ids = exactly the ids whose KV is in the shards (prompt +
        SENT decode tokens; spec-decode crop-aware). On the next text-only request for this
        replica (serialized by model.lock, so the read is race-free) compute the longest common
        prefix LCP(new, cached); when it clears INFINITEMODEL_PREFIX_MIN and every _prefix_kv_ok
        gate: crop every stage to L (the production spec-rollback frame — in-order on every hop,
        so it lands before the suffix) and prefill ONLY new_ids[L:] as a reset=False frame /
        chunk burst at cache_position=L via _send_prefill(base=L) — the proven spec-verify shape
        (#pipefill chunks it cross-stage when long; the shard's intra-stage #prefill-chunk loop
        caps mask memory either way, since it triggers on q>1 regardless of reset). POSITION
        INTEGRITY: RoPE positions for the suffix continue at L because every shard forward path
        computes them from the cache offset — arange(cache_start, cache_start+q) in both
        _forward_impl and _forward_uniform_eager — never from 0. EDGE: new prompt fully inside
        the cache (an identical retry) -> crop to len(new)-1 and prefill the last token, so
        there is always >=1 suffix token to forward (the head must produce next-token logits).
        Any gate/miss -> the classic full prefill, byte-identical to before.

        REPLY CONTRACT: identical to _send_prefill's — the head's logits (or #logits-diet
        reduced form) for the FINAL prompt position, all either call site consumes.
        RECORD CONTRACT: model.kv_ids is nulled (here via _crop/_send_prefill) BEFORE any KV
        mutation and re-published only by the CALLER after this returns — a failure anywhere
        leaves it None, so the next request safely full-prefills.
        #kv-slots: the record + crop + suffix frames are all SLOT-scoped — this generation
        owns `slot` (chosen at admission by longest-LCP over the free slots' records, so the
        best-matching resident stream is the one resumed); C=1 keeps the classic single
        model.kv_ids record via _kv_rec_get."""
        import torch
        cached = self._kv_rec_get(model, slot)
        L = 0
        _min = _prefix_kv_min()
        if (cached and _min and mm is None and position_ids is None
                and model.stage0_writer is not None      # a silent no-op _crop would desync KV
                and self._prefix_kv_ok(model)):
            q = len(prompt_ids)
            n = min(len(cached), q)
            if n >= _min:
                if cached[:n] == prompt_ids[:n]:   # C-speed fast path: pure extension / retry —
                    lcp = n                        # the every-turn agent shape (no Python loop)
                else:
                    lcp = 0
                    while lcp < n and cached[lcp] == prompt_ids[lcp]:
                        lcp += 1
                if lcp >= _min:
                    L = min(lcp, q - 1)   # EDGE: leave >=1 suffix token on an identical retry
        if L <= 0:
            return await self._send_prefill(model, torch.tensor([prompt_ids], dtype=torch.long),
                                            mm=mm, position_ids=position_ids,
                                            nt_mode=nt_mode, nt_clip=nt_clip, nt_k=nt_k,
                                            slot=slot)
        # -- resume: crop every stage to the shared prefix, prefill only the suffix -------------
        await self._crop(model, L, slot=slot)   # nulls the slot's record; in-order ahead of the suffix
        res = await self._send_prefill(model,
                                       torch.tensor([prompt_ids[L:]], dtype=torch.long),
                                       base=L, nt_mode=nt_mode, nt_clip=nt_clip, nt_k=nt_k,
                                       slot=slot)
        # #prefix-kv observability: without this line a reuse is invisible in telemetry (the
        # render-oom-guard lesson) — one activity line per HIT, with the tokens saved.
        with contextlib.suppress(Exception):
            log_activity(f"{model.friendly}: #prefix-kv HIT — reused {L} cached prefix tokens, "
                         f"prefilled {len(prompt_ids) - L} of {len(prompt_ids)} "
                         f"({100 * L // max(1, len(prompt_ids))}% of the prompt skipped)")
        return res

    async def _send(self, model: LoadedModel, x, cache_position: int, reset: bool,
                    all_logits: bool = False, mm=None, position_ids=None,
                    capture_hidden: bool = False, capture_pre_norm: bool = False,
                    ntensor: bool = False, nt_mode=None, nt_clip: int = 0, nt_k: int = 0,
                    prefill_wait: bool = False, slot: int = 0):
        """Push one frame (token ids) through `model`'s pipeline and return last-stage
        logits — last position only, or every position when all_logits=True (verify).
        mm = (positions, embeds_tensor) (#22 inc 3): on a prefill (reset), a companion
        'mm' frame is sent FIRST with the same req_id so stage 0 splices those embeds into
        its embed output at `positions` before running the layers.
        ntensor=True (#ntensor-manifest) asks the head stage for the N-tensor manifest
        return frame instead of the legacy one/two-tensor format — downgraded here to the
        legacy format unless the whole chain advertised the cap (#wire-caps), so callers can
        pass it unconditionally. The reply is dispatched by ITS OWN header in _on_data, so
        even a stale gate (old worker ignoring the flag) degrades cleanly.
        nt_mode (#logits-diet) asks the head for a REDUCED reply riding that manifest:
        'argmax' -> greedy token id(s) only (NT_TOKEN_IDS, int64 — bit-exact vs the legacy
        mask+argmax; with all_logits, one id per position, so a spec-verify round returns
        (K+1) ints instead of (K+1) full rows); 'topk' -> the top-nt_k candidate (values,
        ids) pair for controller-side sampling. nt_clip = tokenizer length (candidates from
        row[:clip]; 0 = raw-row semantics, the _decode_spec convention). Downgraded here
        unless the whole chain advertised 'ntdiet' AND tp_size==1 (_wire_diet_ok) and never
        combined with a capture rider, so callers pass it unconditionally and detect the
        reply TYPE: a [(kind, tensor), ...] list = diet reply; a plain tensor = full row.
        prefill_wait=True (#prefix-kv) gives a reset=False frame the reset branch's progress-
        aware patience — a long SUFFIX prefill at a cache offset is prefill-shaped work, not a
        decode step, so it must not die at the classic single GEN_TIMEOUT budget under GPU
        contention (the #endpoint-weather class). Default False keeps every existing
        decode/verify call site byte-identical."""
        # #logits-diet: downgrade the reduced-reply request unless the chain fully supports it.
        # Callers always handle a full-row reply, so a downgrade is a silent no-op, not an error.
        if nt_mode is not None and (capture_hidden or capture_pre_norm
                                    or not self._wire_diet_ok(model)):
            nt_mode = None
        if nt_mode == "topk" and int(nt_k or 0) <= 0:
            nt_mode = None   # INFINITEMODEL_TOPK_WIRE=0 -> full row for sampled decode
        if nt_mode is not None:
            ntensor = True   # the reduced reply rides the #ntensor-manifest frame
        # #wire-caps: never request the manifest from a chain that hasn't advertised it.
        if ntensor and not self._wire_ntensor_ok(model):
            ntensor = False
            nt_mode = None
        # #prefix-kv: a reset frame rebuilds every stage's KV from position 0 — whatever ids the
        # cross-request record claimed are gone the moment this dispatches. Null FIRST (before
        # any await can fail) so EVERY reset path (decode prefills, capture_thinker, MTP,
        # routes_diag/routes_shards qcheck probes) invalidates centrally; the decode paths that
        # KNOW the post-prefill contents re-publish after their prefill returns. (#kv-slots:
        # slot-scoped — a reset on slot s wipes only slot s's stream on every stage.)
        if reset:
            self._kv_rec_null(model, slot)
        if model.stage0_writer is None:
            await self._freshen_stage0(model, force=True)   # rebuild from saved dial if dropped
        if model.stage0_writer is None:
            raise RuntimeError("pipeline not connected")
        loop = asyncio.get_event_loop()
        rid = self.next_req()
        meta, raw = _pack_tensor(x)
        fut = loop.create_future()
        self.pending[rid] = fut
        self.pending_model[rid] = model.target_id   # so a head drop fails only this model
        self.pending_friendly[rid] = model.friendly   # #5 routed REPLICA key -> replica-precise recovery
        if slot:   # #kv-slots: rid -> slot so the per-slot watchdog fails ONLY this slot's futures
            self._pending_slot_map()[rid] = slot

        async def _flush(w) -> None:
            _bspans = None
            if mm is not None and reset:
                positions, embeds = mm
                emeta, eraw = _pack_tensor(embeds)
                nb = await _write_frame(w, {
                    "req_id": rid, "model_id": model.target_id, "kind": "mm",
                    "positions": list(positions), **emeta}, eraw)
                net_account(self._stage0_id(model), to_node=nb)  # controller -> stage0
                # #gemma4-bidir: image-token runs for block-bidirectional vision attention. Sent on
                # EVERY vision prefill; the shard gates on use_bidirectional_attention='vision', so a
                # causal-only vision model (Pixtral/Qwen-VL) simply ignores them. Prefill-only.
                _bspans = _bidir_spans_from_positions(positions)
            hdr = {"req_id": rid, "model_id": model.target_id, "kind": "ids",
                   "cache_position": cache_position,
                   "reset": reset, "all_logits": all_logits, **meta}
            if slot:   # #kv-slots: route every stage to THIS slot's KV stream (absent == 0 ==
                hdr["slot"] = slot   # legacy byte-identical frames on every C=1 model)
            # #mm-pairing: DECLARE the companion mm frame on the ids header. Stage 0 claims the
            # staged embeds only when declared (and fails LOUD if declared-but-missing, instead of
            # silently running the vision prefill unspliced); undeclared frames never claim, so a
            # leaked companion from a reclaimed gen + a controller-restart req_id collision can't
            # splice stale image embeds into an unrelated prompt.
            if mm is not None and reset:
                hdr["mm"] = True
            if position_ids is not None:   # #22 inc 4: 3D mRoPE positions [3][q] (small JSON list)
                hdr["position_ids"] = position_ids
            if _bspans:   # #gemma4-bidir: small JSON list of [start,end] image runs
                hdr["bidir_spans"] = _bspans
            if capture_hidden:   # #P6 speech: ask the head stage for post-norm hidden too
                hdr["capture_hidden"] = True
            if capture_pre_norm:   # #91 MTP: ask the head stage for the PRE-final-norm trunk hidden
                hdr["capture_pre_norm"] = True
            if ntensor:   # #ntensor-manifest: ask the head for the manifest return frame (gated
                # on #wire-caps above; old workers whitelist-read the header via hdr.get and
                # ignore this key harmlessly — their legacy reply is always still accepted)
                hdr["ntensor"] = True
                if nt_mode is not None:   # #logits-diet: the reduction directive (gated above;
                    # intermediate stages re-propagate these keys hop-by-hop like capture_hidden)
                    hdr["nt_mode"] = nt_mode
                    hdr["nt_clip"] = int(nt_clip or 0)
                    if nt_mode == "topk":
                        hdr["nt_k"] = int(nt_k)
            nb = await _write_frame(w, hdr, raw)
            net_account(self._stage0_id(model), to_node=nb)  # controller -> stage0

        try:
            try:
                await _flush(model.stage0_writer)
            except (ConnectionError, OSError, asyncio.IncompleteReadError):
                # stage0 conn died at/mid send -> rebuild FRESH + resend ONCE. The worker keys frames
                # by model_id and hasn't processed anything on the new socket, so resending the same
                # req_id is clean (mirrors the worker's reconnect-once-on-failure in _send_next).
                await self._freshen_stage0(model, force=True)
                await _flush(model.stage0_writer)
            model.last_send_ts = time.time()
            if not reset and not prefill_wait:   # decode/verify step: classic single-budget wait
                return await asyncio.wait_for(fut, timeout=GEN_TIMEOUT_S)
            # #endpoint-weather: a PREFILL is ONE frame/future for the whole prompt, so under GPU
            # contention a healthy prefill can legitimately outlast GEN_TIMEOUT_S — and dying here
            # re-enters the same slow prefill on every client retry (the 21% run-abort class).
            # Wait in slices and KEEP WAITING while workers report ADVANCING per-layer progress
            # (model.fwd_progress_ts, heartbeat-fed); bail only when the classic budget is spent
            # AND progress has gone quiet. True wedges die much earlier via the gen-stall
            # watchdog (which cancels this task); the hard ceiling below is the backstop for a
            # disabled watchdog so this can never hang forever.
            _PROG_QUIET_S = 120.0
            _PREFILL_HARD_S = max(3600.0, GEN_TIMEOUT_S)
            _t0 = time.time()
            while True:
                _left = _PREFILL_HARD_S - (time.time() - _t0)
                if _left <= 0:
                    raise TimeoutError(f"prefill exceeded the {int(_PREFILL_HARD_S)}s hard ceiling")
                try:
                    return await asyncio.wait_for(asyncio.shield(fut), timeout=min(30.0, _left))
                except asyncio.TimeoutError:
                    if fut.done():        # raced the timeout: result/exception landed — take it
                        return fut.result()
                    _fp = max(getattr(model, "fwd_progress_ts", 0.0) or 0.0, model.last_send_ts)
                    if (time.time() - _t0) >= GEN_TIMEOUT_S and (time.time() - _fp) > _PROG_QUIET_S:
                        raise
        finally:
            self.pending.pop(rid, None)  # never leak the future
            self.pending_model.pop(rid, None)
            self.pending_friendly.pop(rid, None)   # #5 keep replica map in lockstep
            getattr(self, "pending_slot", {}).pop(rid, None)   # #kv-slots: keep slot map in lockstep

    async def _send_prefill(self, model: LoadedModel, x, mm=None, position_ids=None,
                            nt_mode=None, nt_clip: int = 0, nt_k: int = 0, base: int = 0,
                            slot: int = 0):
        """#pipefill: prefill dispatcher. Eligible prompts (see _pipefill_chunk) are streamed
        through the pipeline as a BURST of chunk frames so the stages overlap; everything else
        falls through to the classic one-shot _send prefill, byte-identical to before.

        THE DEFECT this fixes: a one-shot prefill runs the S stages strictly SEQUENTIALLY —
        stage 0 computes the WHOLE prompt (internally chunked by INFINITEMODEL_PREFILL_CHUNK
        for mask memory, but shipped as ONE hidden block), only then does stage 1 start, etc.
        TTFT = sum of per-stage prefill times + full per-hop transfer times (the #endpoint-
        weather comment in _send documents healthy prefills outlasting GEN_TIMEOUT under
        contention for exactly this reason).

        MECHANISM — controller-orchestrated chunk streaming, chosen over a worker-forwarded
        redesign because it reuses the MOST production-proven machinery and needs ZERO new
        wire format: the prompt is split into ~2048-token chunks and written back-to-back to
        stage 0 as ordinary ids frames — chunk 0 reset=True at cache_position 0 (rebuilds
        every stage's KV via the existing #kv-reset-on-seqstart contract), chunks 1..C-1
        reset=False at their absolute cache_position, each with its OWN req_id/future. The
        multi-token reset=False forward at a cache offset is EXACTLY the spec-decode verify
        path (_decode_spec sends [pending]+drafts at cur, reset=False) which is exercised in
        production, so no worker forward change is required; per-stage KV chunks append in
        order because TCP is in-order and the worker inbound loop (worker_net._data_inbound)
        is strictly sequential per connection. The pipelining falls out of the EXISTING
        worker behavior: stage k's loop finishes chunk i, _send_next ships its hidden block,
        and reads chunk i+1 — while stage k+1 computes chunk i. Cap-gated ('pipefill',
        #wire-caps all-or-nothing) so a chain with any pre-audit node keeps one-shot.

        REPLY CONTRACT: identical to `await _send(model, x, 0, True, ...)` — the returned
        value is the LAST chunk's head reply (logits [1,1,V] for the final prompt position,
        or its #logits-diet reduced form), which is what both prefill call sites consume.
        Intermediate chunks' head replies are pipeline exhaust: with all_logits=False each is
        one logits row, requested in #logits-diet argmax form (8 bytes) when the chain
        supports it, awaited only for ERROR propagation and then discarded.

        FAILURE CONTRACT (audit caveats 3+4): ALL chunk futures are awaited via a fail-fast
        gather — hop_error/stage_error/watchdog fail ONLY their own rid, so awaiting just the
        last future would hang ~GEN_TIMEOUT when an intermediate chunk dies. The whole burst
        (including the FINAL chunk, which plain _send would give only the classic reset=False
        single budget) waits under _send's #endpoint-weather progress-aware patience: keep
        waiting while per-layer forward progress (model.fwd_progress_ts, heartbeat-fed and
        credited only to still-pending rids — every chunk rid stays pending until its reply)
        is advancing, bail once the classic budget is spent AND progress has gone quiet, with
        the same hard ceiling backstop. Cancellation (client drop / gen-stall reclaim /
        _ForwardSuperseded upstream) unwinds through the finally: every chunk rid is scrubbed
        from pending, so orphaned worker forwards report progress nobody credits (exactly the
        one-shot orphan contract) and their late replies/errors are ignored by _on_data.

        base>0 (#prefix-kv): `x` is a SUFFIX starting at absolute cache_position `base` —
        every frame goes out reset=False at base+off (the production spec-verify shape; the
        caller has already cropped every stage's KV to exactly `base`, in-order on the same
        sockets). base==0 keeps every path above byte-identical."""
        q = int(x.shape[1])
        # #prefix-kv: ANY prefill (full at 0 or suffix at base) mutates the shard KV — the
        # cross-request record is stale from here until the owning decode path re-publishes it.
        # (#kv-slots: only THIS slot's record/stream — sibling slots are untouched.)
        self._kv_rec_null(model, slot)
        cstep = self._pipefill_chunk(model, q, mm, position_ids)
        if not cstep:
            return await self._send(model, x, base, base == 0, mm=mm, position_ids=position_ids,
                                    nt_mode=nt_mode, nt_clip=nt_clip, nt_k=nt_k,
                                    prefill_wait=base > 0, slot=slot)
        # -- chunked pipelined path ------------------------------------------------------------
        # #logits-diet gating for the LAST chunk mirrors _send's own downgrade ladder, so the
        # final reply keeps the exact same wire mode the one-shot prefill would have used.
        if nt_mode is not None and not self._wire_diet_ok(model):
            nt_mode = None
        if nt_mode == "topk" and int(nt_k or 0) <= 0:
            nt_mode = None
        _diet_mid = self._wire_diet_ok(model)   # shrink DISCARDED intermediate replies to ~8 B
        if model.stage0_writer is None:
            await self._freshen_stage0(model, force=True)
        if model.stage0_writer is None:
            raise RuntimeError("pipeline not connected")
        loop = asyncio.get_event_loop()
        rids: list = []
        futs: list = []

        async def _write_chunk(rid: int, off: int, cl: int, last: bool) -> None:
            meta, raw = _pack_tensor(x[:, off:off + cl])
            # #prefix-kv: base offsets every chunk; reset only at ABSOLUTE position 0 (a suffix
            # burst must append to the cropped resident KV, never rebuild it).
            hdr = {"req_id": rid, "model_id": model.target_id, "kind": "ids",
                   "cache_position": base + off, "reset": base + off == 0,
                   "all_logits": False, **meta}
            if slot:   # #kv-slots: every chunk carries the gen's slot id (absent == 0 == legacy)
                hdr["slot"] = slot
            if last:
                if nt_mode is not None:   # the caller's reply directive rides the FINAL chunk
                    hdr["ntensor"] = True
                    hdr["nt_mode"] = nt_mode
                    hdr["nt_clip"] = int(nt_clip or 0)
                    if nt_mode == "topk":
                        hdr["nt_k"] = int(nt_k)
            elif _diet_mid:
                # Intermediate replies are discarded — ask for the argmax id (8 bytes) instead
                # of a ~300 KB full-vocab row purely to keep exhaust off the last hop's wire.
                hdr["ntensor"] = True
                hdr["nt_mode"] = "argmax"
                hdr["nt_clip"] = 0
            nb = await _write_frame(model.stage0_writer, hdr, raw)
            net_account(self._stage0_id(model), to_node=nb)   # controller -> stage0

        allf = None
        try:
            off = 0
            first = True
            while off < q:
                cl = min(cstep, q - off)
                rid = self.next_req()
                fut = loop.create_future()
                self.pending[rid] = fut
                self.pending_model[rid] = model.target_id
                self.pending_friendly[rid] = model.friendly   # #5 replica-precise recovery
                if slot:   # #kv-slots: rid -> slot for the per-slot watchdog fail
                    self._pending_slot_map()[rid] = slot
                rids.append(rid)
                futs.append(fut)
                try:
                    await _write_chunk(rid, off, cl, off + cl >= q)
                except (ConnectionError, OSError, asyncio.IncompleteReadError):
                    if not first:
                        # Mid-burst write failure: earlier chunks may already sit in the OLD
                        # socket's stream, and a resend on a FRESH socket would let TWO worker
                        # inbound loops interleave this prefill's frames (the one-frame
                        # in-flight window that makes _send's resend-once safe does not hold
                        # for a burst) -> KV append order would be undefined. Freshen for the
                        # NEXT request and fail THIS one; the retry's chunk 0 (reset=True,
                        # position 0) rebuilds every stage's cache, so nothing stale survives.
                        # (#prefix-kv suffix bursts too: this failure nulled model.kv_ids via
                        # the dispatcher, so the retry is a FULL reset prefill, never a resume.)
                        with contextlib.suppress(Exception):
                            await self._freshen_stage0(model, force=True)
                        raise
                    # First frame failed: nothing of this prefill reached the old stream ->
                    # mirror _send's reconnect-once-and-resend (worker keys frames by model_id
                    # and has processed nothing on the new socket, so the same rid is clean).
                    await self._freshen_stage0(model, force=True)
                    await _write_chunk(rid, off, cl, off + cl >= q)
                first = False
                off += cl
            model.last_send_ts = time.time()
            # Fail-fast gather over ALL chunk futures + the #endpoint-weather progress-aware
            # patience from _send's reset branch (see FAILURE CONTRACT above).
            allf = asyncio.gather(*futs)
            _PROG_QUIET_S = 120.0
            _PREFILL_HARD_S = max(3600.0, GEN_TIMEOUT_S)
            _t0 = time.time()
            while True:
                _left = _PREFILL_HARD_S - (time.time() - _t0)
                if _left <= 0:
                    raise TimeoutError(
                        f"pipefill prefill exceeded the {int(_PREFILL_HARD_S)}s hard ceiling")
                try:
                    res = await asyncio.wait_for(asyncio.shield(allf), timeout=min(30.0, _left))
                    return res[-1]
                except asyncio.TimeoutError:
                    if allf.done():       # raced the timeout: result/exception landed — take it
                        return allf.result()[-1]
                    _fp = max(getattr(model, "fwd_progress_ts", 0.0) or 0.0, model.last_send_ts)
                    if ((time.time() - _t0) >= GEN_TIMEOUT_S
                            and (time.time() - _fp) > _PROG_QUIET_S):
                        raise
        finally:
            # Do NOT cancel a still-pending gather here: cancelling a future that is (or was
            # just) wrapped in asyncio.shield makes the loop log "CancelledError exception in
            # shielded future" on every abort path (watchdog reclaim / client drop). Scrubbing
            # the rids from pending below already guarantees no chunk future can EVER resolve
            # after this point (_on_data / hop_error / the watchdog all key off pending), so
            # the un-awaited gather + its pending children just get GC'd — a pending future
            # never warns; only an unretrieved EXCEPTION does, and those are retrieved here.
            if allf is not None and allf.done() and not allf.cancelled():
                with contextlib.suppress(Exception):
                    allf.exception()
            for _rid in rids:   # never leak a chunk future (same contract as _send's finally)
                self.pending.pop(_rid, None)
                self.pending_model.pop(_rid, None)
                self.pending_friendly.pop(_rid, None)
                getattr(self, "pending_slot", {}).pop(_rid, None)   # #kv-slots lockstep
            for _f in futs:
                # Retrieve any SECOND chunk failure so a near-simultaneous double fault (gather
                # surfaced the first) can't log "exception was never retrieved" at GC.
                if _f.done() and not _f.cancelled():
                    with contextlib.suppress(Exception):
                        _f.exception()

    async def _crop(self, model: LoadedModel, length: int, slot: int = 0) -> None:
        """Tell every stage of `model` to truncate its KV cache to `length` (spec rollback,
        #prefix-kv resume). Fire-and-forget: in-order delivery on each stage's connection
        guarantees the crop is applied before the next frame the controller sends afterwards.
        #kv-slots: the crop acts on THE REQUEST'S slot only (the header key is added when >0;
        absent == slot 0 == legacy byte-identical frame) — a crop on slot 1 must never shorten
        slots 0/2's independent streams."""
        # #prefix-kv: KV contents change under this frame — invalidate the cross-request record;
        # the callers that know the exact post-crop ids (spec decode's round loop, the prefix
        # resume's owner) re-publish afterwards, everyone else safely full-prefills next.
        self._kv_rec_null(model, slot)
        if model.stage0_writer is not None:
            _chdr = {"model_id": model.target_id, "kind": "crop", "cache_position": length}
            if slot:
                _chdr["slot"] = slot
            nbytes = await _write_frame(model.stage0_writer, _chdr, b"")
            net_account(self._stage0_id(model), to_node=nbytes)  # controller -> stage0

    # -- draft model (runs entirely on the controller; one per LoadedModel) --
    def _load_draft(self, model: LoadedModel, draft_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM
        _controller_model_dir(draft_id)
        _dm = AutoModelForCausalLM.from_pretrained(
            draft_id, dtype=torch.bfloat16, attn_implementation="eager").eval()
        # #spec-draft-gpu: a CPU draft step reads the draft's full weights from DDR — on an
        # APU-class box that costs 150-350 ms/step, MORE than the target's ~285 ms verify
        # sweep, so CPU drafting loses outright (measured: llama-70b 3.50 plain -> 2.49
        # spec tok/s on om3nbox). On the GPU the same step is ~15-30 ms and spec pays off.
        # Use the GPU only when its REAL free VRAM fits the draft plus a margin (default 4 GB)
        # — the controller's allocation is invisible to the worker-side placement budget, so a
        # thin margin could squeeze a resident model's KV growth. CPU remains the fallback.
        # #draft-gpu: the margin is tunable per load (/load?draft_margin_gb=) because on a small
        # card (16 GB 4070TiS) the fixed 4 GB made a GPU draft unreachable even with the plan-time
        # reserve; an explicit draft_gpu load accepts a thinner cushion knowingly.
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                _free_b, _tot_b = torch.cuda.mem_get_info()
                _need = sum(p.numel() * p.element_size() for p in _dm.parameters())
                _margin_b = int(max(0.0, float(getattr(model, "draft_margin_gb", 4.0) or 4.0))
                                * (1024 ** 3))
                if _free_b > _need + _margin_b:
                    _dm = _dm.to("cuda")
                    print(f"[load] spec draft on GPU ({_need / 1024**3:.1f} GB model, "
                          f"{_free_b / 1024**3:.1f} GB free before)")
        model.draft_model = _dm
        model.draft_id = draft_id
        model.draft_kv = None

    def _unload_draft(self, model: LoadedModel) -> None:
        # #draft-gpu: a draft placed on cuda (draft_gpu load) keeps its ~GB of weights in the torch
        # caching allocator after the Python object is dropped — only empty_cache() hands it back
        # (exactly the MTP-head leak _free_mtp_cuda documents: a leaked controller-GPU tensor fouls
        # the box's GPU so the NEXT model spills to CPU — observed as qwen2.5:14b landing 75% on CPU
        # after a 70B GPU-draft unload). A CPU draft frees via _release_ram (callers already do that),
        # so only pay the gc+empty_cache when the draft was actually GPU-resident.
        _on_cuda = False
        with contextlib.suppress(Exception):
            _dm = model.draft_model
            _on_cuda = (_dm is not None
                        and next(_dm.parameters()).device.type == "cuda")
        model.draft_model = None
        model.draft_kv = None
        model.draft_id = None
        if _on_cuda:
            _free_mtp_cuda()   # gc.collect() + torch.cuda.empty_cache() (server.py; generic reclaim)

    def _draft_prefill(self, model: LoadedModel, prompt_ids):
        import torch
        from transformers import DynamicCache
        _dev = model.draft_model.device   # cpu or cuda (#spec-draft-gpu)
        model.draft_kv = DynamicCache()
        with torch.inference_mode():
            out = model.draft_model(input_ids=torch.tensor([prompt_ids], device=_dev),
                                    past_key_values=model.draft_kv, use_cache=True)
        return out.logits[0, -1]

    def _draft_step(self, model: LoadedModel, token: int, position: int):
        import torch
        _dev = model.draft_model.device
        with torch.inference_mode():
            out = model.draft_model(input_ids=torch.tensor([[token]], device=_dev),
                                    past_key_values=model.draft_kv, use_cache=True,
                                    cache_position=torch.tensor([position], device=_dev))
        return out.logits[0, -1]

    def _draft_crop(self, model: LoadedModel, length: int) -> None:
        if model.draft_kv is not None:
            with contextlib.suppress(Exception):
                model.draft_kv.crop(length)

    async def _await_promote_gate(self, base: str) -> None:
        """#juggler barrier: block until any in-progress VRAM-promotion of `base` has finished, so a
        request transparently rides the re-place and then resolves the FRESH replica. No-op (one dict
        lookup) when nothing is being juggled — the common case. The loop re-checks because a promotion
        can (rarely) start again while an earlier waiter is being woken."""
        for _ in range(10000):
            ev = self._promote_gates.get(base)
            if ev is None:
                return
            await ev.wait()

    async def generate(self, friendly: str, prompt_ids: list[int], max_new: int,
                       temperature: float, top_p: float, speculative: bool = False,
                       rec=None, mm=None, mrope=None, spec_k: int = 0,
                       min_p: float = 0.0, sampling=None):
        """Dispatch generation for model `friendly`: speculative-greedy decode only when
        explicitly requested AND a draft is loaded AND decoding is greedy; otherwise plain
        KV-cache decode (M2e). Speculative is opt-in because it only wins when the target's
        per-traversal cost dwarfs the local draft cost (big model / many nodes) — on small
        targets it measures SLOWER, so it must never silently replace the fast default.
        `sampling` (#runtime-knobs) bundles the extended knob family (top_k / repeat_penalty /
        repeat_last_n / presence_penalty / frequency_penalty / seed) for the PLAIN decode path;
        the speculative paths are greedy-only by construction and ignore it (penalties would
        break draft/target logit agreement)."""
        # #t2i-serve: an image model has no token path — refuse text generation with a
        # pointer to the right endpoint instead of a cryptic downstream crash.
        _lm0 = self.models.get(friendly)
        if _lm0 is not None and getattr(_lm0, "is_t2i", False):
            raise ValueError(f"'{friendly}' is an image-generation model — "
                             "use POST /v1/images/generations")
        # #juggler barrier: if this model is being promoted to VRAM (re-placed), hold the request
        # HERE — before it resolves a replica or takes a queue slot — until the swap finishes, then
        # fall through and pick the fresh copy. The client's connection just pauses (no reconnect).
        # No await between the gate returning and `model.queued += 1` below, so this is race-tight with
        # the promoter's idle re-check: a request that clears the gate is counted (active/queued)
        # before the reload starts, so the promoter sees it and SKIPS rather than reloading under it.
        await self._await_promote_gate(friendly)
        model = self._pick_replica(friendly)   # data-parallel: least-loaded replica (#39)
        if model is None or model.stage0_writer is None:
            raise RuntimeError("no model loaded")
        # #ctx-guard: REJECT a prompt that doesn't fit the loaded context window instead of dispatching
        # an over-ctx prefill — that overflows the worker's fixed (ctx-sized) KV cache and HARD-CRASHES
        # the shard (drops the node; observed: a 42k-token prompt into a 32768-ctx model). Also cap
        # max_new so prompt+generated can't overflow the KV during decode. The serving layer rejects
        # pre-stream too (clean 400); this is the universal backstop for EVERY entrypoint (Ollama /
        # OpenAI / Anthropic / future).
        _ctx = int(getattr(model, "ctx", 0) or 0)
        if _ctx:
            if len(prompt_ids) >= _ctx:
                raise ValueError(f"prompt is {len(prompt_ids)} tokens but the model is loaded with a "
                                 f"{_ctx}-token context window — shorten the prompt or reload it at a "
                                 f"larger ctx")
            if max_new > _ctx - len(prompt_ids):
                max_new = _ctx - len(prompt_ids)
        # PER-REPLICA admission: different models AND different replicas of one model decode
        # concurrently; requests routed to the SAME replica queue on its slot pool. #kv-slots:
        # kv_slots==1 (default) -> _SlotLease degenerates to EXACTLY the old whole-generation
        # ``async with model.lock`` (same object, same FIFO); kv_slots==C>1 -> a Semaphore(C)
        # permit + one slot id owned for this generation's lifetime (prefill+decode+crop all
        # carry it), so up to C generations interleave their per-token hops through the SAME
        # pipeline — the stages naturally overlap different slots' tokens with NO new scheduler.
        # Track queue depth for /status (queued = waiting on a slot; active = generating).
        model.queued += 1
        acquired = False
        _lease = _SlotLease(model, prompt_ids)
        try:
            async with _lease:
                _slot = _lease.slot
                acquired = True
                model.queued -= 1
                model.active += 1   # #kv-slots: 0..C concurrent gens — 'active' = sum over slots
                model.last_token_ts = time.time()   # #gen-stall-watchdog: start the no-progress timer at gen begin
                model.gen_started_ts = model.last_token_ts   # #active-decode-stall: prefill marker (token 1 advances last_token_ts past this)
                # #kv-slots (C>1): per-(model,slot) activity record — the gen-stall watchdog
                # reclaims per SLOT (one wedged slot must not reclaim its siblings), so it needs
                # per-slot token stamps + this request's INFLIGHT rec + the lease token (to free
                # the slot safely). None at C=1 — the model-level watchdog path is untouched.
                _sst = None
                if _lease.token is not None:
                    _sst = model.slot_state[_slot] = {
                        "last_token_ts": model.last_token_ts,
                        "gen_started_ts": model.gen_started_ts,
                        "rec": rec, "lease": _lease.token}
                # #stage0-stale-reconnect: rebuild a stale (idle-since-last-request) stage0 conn BEFORE
                # the prefill so this request rides a fresh, proven socket instead of a possibly
                # half-open one (the 'loaded but never replies' / ~600s hang). No-op when hot (busy
                # model) or recently sent; under the lock so no concurrent decode is using the writer.
                with contextlib.suppress(Exception):
                    await self._freshen_stage0(model)
                _inflight_start(rec)   # slot acquired: queued -> running (dashboard)
                try:
                    model.last_used = time.time()
                    greedy = not temperature or temperature <= 0
                    # #46 throughput: count emitted tokens over wall-clock and store a
                    # smoothed decode tok/s on the model (observability only — no effect on
                    # generation). t0 starts after the per-replica lock is held so it times
                    # this request's decode, not its queue wait. Tokens = real tokens yielded
                    # (item[0] is not None); the trailing stop/length marker is skipped.
                    _t0 = time.monotonic()
                    _ntoks = 0
                    _out_ids: list = []   # #ctx-history: accumulate generated token ids (decoded lazily)
                    # Multimodal (mm) forces PLAIN decode: the controller-side draft model has
                    # no image embeds, so speculative would diverge — only the full pipeline
                    # gets the spliced vision tokens at prefill.
                    # #91 MTP: when speculative+greedy is requested but there's no separate draft
                    # model, fall through to the checkpoint's own MTP (nextn) self-draft if it has one.
                    # #kv-slots: spec + MTP are gated to C=1 (_lease.token is None) — both drive a
                    # SINGLE controller-resident draft state (model.draft_kv / the MTP head's own
                    # cache), which two concurrent slots would corrupt. Plain decode serves C>1.
                    mtp_head = None
                    if (speculative and greedy and mm is None and model.draft_model is None
                            and _lease.token is None):
                        with contextlib.suppress(Exception):
                            mtp_head = await self._ensure_mtp_head(model)
                    if (speculative and model.draft_model is not None and greedy and mm is None
                            and _lease.token is None):
                        async for item in self._decode_spec(model, prompt_ids, max_new, spec_k):
                            if item[0] is not None:
                                _ntoks += 1
                                _out_ids.append(item[0])   # #ctx-history
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
                                _dt = time.monotonic() - _t0
                                if _dt > 1e-6:           # LIVE decode rate -> card updates mid-gen (#46)
                                    model.last_tok_s = _ntoks / _dt
                            yield item
                    elif mtp_head is not None:
                        async for item in self._decode_spec_mtp(model, prompt_ids, max_new, mtp_head):
                            if item[0] is not None:
                                _ntoks += 1
                                _out_ids.append(item[0])   # #ctx-history
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
                                _dt = time.monotonic() - _t0
                                if _dt > 1e-6:           # LIVE decode rate -> card updates mid-gen (#46)
                                    model.last_tok_s = _ntoks / _dt
                            yield item
                    else:
                        async for item in self._decode_plain(model, prompt_ids, max_new,
                                                             temperature, top_p, mm=mm, mrope=mrope,
                                                             min_p=min_p, sampling=sampling,
                                                             slot=_slot):
                            if item[0] is not None:
                                _ntoks += 1
                                _out_ids.append(item[0])   # #ctx-history
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
                                if _sst is not None:   # #kv-slots: per-slot progress (slot watchdog)
                                    _sst["last_token_ts"] = model.last_token_ts
                                _dt = time.monotonic() - _t0
                                if _dt > 1e-6:           # LIVE decode rate -> card updates mid-gen (#46)
                                    model.last_tok_s = _ntoks / _dt
                            yield item
                finally:
                    # max(0, ...): the gen-stall watchdog may have already zeroed model.active when it
                    # reclaimed THIS (now-unblocked) wedged gen — without the floor the double-decrement
                    # drives active negative, skewing _pick_replica routing + the dashboard counts.
                    model.active = max(0, model.active - 1)
                    # #model-detail lifetime counters (this is the main text-generation path; TP /
                    # speech paths don't update these). Count every served request + its tokens.
                    model.req_total += 1
                    model.tok_in_total += len(prompt_ids)
                    model.tok_out_total += _ntoks
                    # #connections: attribute this request's tokens to its client (rec carries the
                    # ip) — ONE spot covers every generate entry point (Ollama/OpenAI/Anthropic +
                    # the tools reply loop). rec is None for internal callers (warmup probes).
                    if rec is not None:
                        with contextlib.suppress(Exception):
                            _client_tokens(rec.get("ip"), len(prompt_ids), _ntoks, model.friendly)
                    with contextlib.suppress(Exception):   # #ctx-history: capture this request's in/out
                        _record_ctx_history(model.friendly, prompt_ids, _out_ids,
                                            len(prompt_ids), _ntoks)
                    # Record decode throughput once the generation finishes (or is cut
                    # short). Guard on a sane sample (>=1 token, measurable time) so a
                    # zero-token or instant request doesn't poison the read.
                    _dt = time.monotonic() - _t0
                    if _ntoks >= 1 and _dt > 1e-6:
                        ts = _ntoks / _dt
                        model.last_tok_s = ts
                        if ts > model.max_tok_s:        # peak decode tok/s (#model-detail)
                            model.max_tok_s = ts
                        # EMA (alpha=0.3): seed on the first sample, then blend.
                        model.ema_tok_s = ts if model.ema_tok_s <= 0.0 else \
                            0.3 * ts + 0.7 * model.ema_tok_s
        finally:
            if not acquired:               # cancelled while still waiting in the queue
                model.queued -= 1

    async def _decode_plain(self, model, prompt_ids, max_new, temperature, top_p, mm=None,
                            mrope=None, min_p: float = 0.0, sampling=None, slot: int = 0):
        """Prefill-once + one-token-at-a-time KV-cache decode (M2e). mm=(positions, embeds)
        (#22 inc 3) splices multimodal embeds into the PREFILL only; decode steps are plain.
        mrope=(prefill_position_ids [3][q], base) (#22 inc 4) carries 3D image positions:
        the prefill uses the full layout; each decode token uses [base+step] on all 3 dims.
        slot (#kv-slots): the decode slot this generation owns — every frame (prefill chunks,
        decode hops, crops) carries it so all stages route to THIS slot's KV stream. 0 on
        every C=1 model — byte-identical legacy frames (the header key is only added when >0)."""
        import torch
        # #runtime-knobs: unpack the extended sampling family once, outside the token loop.
        _sp = sampling or {}
        _top_k = int(_sp.get("top_k") or 0)
        _gen = None
        if _sp.get("seed") is not None:
            # FRESH per-request generator: same prompt + same seed + same knobs => same output,
            # independent of concurrent generations (never touches the global torch RNG).
            _gen = torch.Generator().manual_seed(int(_sp["seed"]))
        _pen = (float(_sp.get("repeat_penalty") or 1.0) not in (0.0, 1.0)
                or bool(_sp.get("presence_penalty")) or bool(_sp.get("frequency_penalty")))
        _hist: list[int] = []   # tokens emitted so far (the penalties' output window)
        # Empty prompt (a keep-warm/health probe whose text tokenizes to []) has nothing to
        # prefill: torch.tensor([[]]) is shape [1,0] and an empty forward crashes the worker's
        # tensor unpack. Short-circuit with zero generated tokens BEFORE any wire send.
        if not prompt_ids:
            yield None, "stop"
            return
        prefill_pos = mrope[0] if mrope else None
        base = mrope[1] if mrope else None
        # #21: this model's lm_head can be WIDER than its text tokenizer (a multimodal
        # head carries vision/audio placeholder ids the text tokenizer can't decode).
        # Selecting one of those ids crashed detokenization ("list index out of
        # range") and showed up as empty/failed generation. Mask logits beyond the
        # tokenizer's decodable range so we only ever emit a real text token.
        # (Computed BEFORE the prefill so the #logits-diet directive can carry it as nt_clip —
        # the head applies row[:ntok] FIRST, the exact worker-side twin of this mask.)
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0
        # #logits-diet: pick the per-request reduced-reply mode ONCE. Penalties need
        # arbitrary-id access to the full row (_penalized indexes any history id), so they keep
        # the legacy full-row wire. Greedy is exactly argmax(row[:ntok]) -> ship ids only
        # (bit-exact). Sampled -> top-K candidates (K=INFINITEMODEL_TOPK_WIRE, 0=off); the
        # controller then runs the SAME _sample over the K candidates and maps the draw through
        # the candidate ids (min_p keeps identical survivor sets — probability RATIOS are
        # renormalization-invariant; top_p/multinomial renormalize over K, dropping only the
        # beyond-K tail mass — the documented, negligible-at-4096 truncation). _send downgrades
        # to the legacy row unless the whole chain advertises 'ntdiet' (and on TP), so every
        # consumer below also handles a plain-tensor reply.
        _nt_mode, _nt_k = None, 0
        if not _pen:
            if not temperature or temperature <= 0:
                _nt_mode = "argmax"
            else:
                _nt_k = _wire_topk_k()
                if _nt_k > 0:
                    _nt_mode = "topk"
        # #pipefill: long multi-stage text prefills stream as a chunk burst (stages overlap).
        # #prefix-kv: when the LAST generation's KV (still resident on every stage) shares a
        # long prefix with this prompt, _prefill_reuse crops to the shared prefix and prefills
        # ONLY the suffix; everything else takes the classic full prefill — same reply shape.
        # #kv-slots (C>1): at most ONE in-PREFILL generation per replica at a time — a prefill
        # (esp. a #pipefill chunk burst) occupies stages for seconds, and the worker inbound
        # loop is strictly sequential per connection, so a second concurrent prefill would
        # head-of-line-block the sibling slots' single-token decode frames behind it. The
        # prefill semaphore (inside the slot semaphore — slot held, prefill serialized) keeps
        # decode slots flowing between bursts; #pipefill chunks yield between chunks, which is
        # exactly the interleave point. C=1: no lock exists — path byte-identical.
        _pfl = getattr(model, "prefill_lock", None) if int(
            getattr(model, "kv_slots", 1) or 1) > 1 else None
        if _pfl is not None:
            async with _pfl:
                res = await self._prefill_reuse(model, prompt_ids, mm=mm,
                                                position_ids=prefill_pos, nt_mode=_nt_mode,
                                                nt_clip=ntok, nt_k=_nt_k, slot=slot)
        else:
            res = await self._prefill_reuse(model, prompt_ids, mm=mm, position_ids=prefill_pos,
                                            nt_mode=_nt_mode, nt_clip=ntok, nt_k=_nt_k, slot=slot)
        cur = len(prompt_ids)
        model.kv_pos = cur          # KV depth so far (prompt); climbs per decode token
        # #prefix-kv: publish the cross-request record — exactly the ids now in every stage's
        # KV (the prefill round-tripped: the head's reply proves every stage appended; on a
        # resume, prefix[:L] + suffix == prompt_ids again). Appended IN PLACE after each
        # successful decode send below, so the record tracks SENT tokens only — the final
        # emitted token (length stop) and the eos are sampled but never sent (the audit's
        # off-by-one caveat). Any send failure nulls it; the next request then full-prefills.
        # mm (vision/audio) prompts publish NOTHING: the shard KV rows at spliced positions
        # were computed from IMAGE/AUDIO embeds, not from the placeholder token ids, so a
        # prompt_ids record would violate the contract ('exactly the ids whose KV is in the
        # shards') — a later TEXT-ONLY prompt whose id-LCP crossed a splice start would
        # silently reuse image-embed KV as text KV. The read gate requires mm is None anyway,
        # so a vision follow-up turn could never resume regardless; full-id publishing on mm
        # requests buys nothing. (_kv still appends in-loop below — harmless, unpublished.)
        # #kv-slots: the record is PER-SLOT (_kv_rec_*) — C=1 keeps the classic model.kv_ids.
        _kv = list(prompt_ids)
        self._kv_rec_set(model, slot, _kv if mm is None else None)
        produced = 0
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            tok_id = None
            if isinstance(res, list):
                # #logits-diet reduced reply: [(kind, tensor), ...] straight off the manifest
                # (engine_lifecycle passes non-legacy kind sets through verbatim).
                _by = {int(_k): _t for _k, _t in res}
                if NT_TOKEN_IDS in _by:            # argmax mode: the head already picked it
                    tok_id = int(_by[NT_TOKEN_IDS].reshape(-1)[-1])
                elif NT_TOPK_VALS in _by and NT_TOPK_IDX in _by:
                    # topk mode: run the REAL sampler over the K candidate logits (fp32-converted
                    # inside _sample exactly like the full row was), then map the drawn position
                    # through the candidate ids. The head clipped to ntok first, so every
                    # candidate is a decodable text token (#21) — no re-mask needed.
                    _cv = _by[NT_TOPK_VALS].reshape(-1)
                    _cpos = self._sample(_cv, temperature, top_p, min_p, top_k=_top_k, gen=_gen)
                    tok_id = int(_by[NT_TOPK_IDX].reshape(-1)[_cpos])
                else:   # unusable kind set — can't happen from our own workers; fail loud
                    raise RuntimeError(f"#logits-diet reply with unusable kinds {sorted(_by)}")
            if tok_id is None:      # legacy full-row reply (diet off / downgraded chain)
                row = res[0, -1]
                if ntok and ntok < int(row.shape[-1]):
                    row = row.clone()
                    row[ntok:] = float("-inf")
                if _pen:   # #runtime-knobs: repetition penalties reshape the logits pre-sampling
                    row = self._penalized(row, prompt_ids, _hist, _sp)
                tok_id = self._sample(row, temperature, top_p, min_p, top_k=_top_k, gen=_gen)
            _hist.append(tok_id)
            if produced == 0:
                with contextlib.suppress(Exception):
                    _hv = int(res.shape[-1]) if hasattr(res, "shape") else f"diet:{_nt_mode}"
                    print(f"[gen] {model.friendly}: first token id={tok_id} "
                          f"head_vocab={_hv} len(tok)={ntok} "
                          f"eos={tok_id in model.eos_ids}")
            produced += 1
            if tok_id in model.eos_ids:
                yield None, "stop"
                return
            yield tok_id, None
            if produced >= max_new:
                break
            # mRoPE decode position = base + step (same on t/h/w); else 1D (worker uses arange).
            dpos = [[base + produced - 1]] * 3 if base is not None else None
            try:
                res = await self._send(model, torch.tensor([[tok_id]], dtype=torch.long), cur,
                                       False, position_ids=dpos,
                                       nt_mode=_nt_mode, nt_clip=ntok, nt_k=_nt_k, slot=slot)
            except BaseException:
                # #prefix-kv: append state unknown mid-send (incl. CancelledError from a client
                # drop / gen-stall reclaim / _ForwardSuperseded landing in this await) -> the
                # record is invalid; null it so the next request full-prefills. (#kv-slots:
                # only THIS slot's record — sibling slots' records stay valid.)
                self._kv_rec_null(model, slot)
                raise
            cur += 1
            model.kv_pos = cur
            _kv.append(tok_id)   # #prefix-kv: tok_id's KV landed on every stage (reply received)
        yield None, "length"

    async def capture_thinker(self, friendly, prompt_ids, max_new, temperature=0.0,
                              top_p=1.0, mm=None, mrope=None):
        """#P6 speech: run the distributed Thinker like _decode_plain BUT with
        capture_hidden=True so the head stage returns the post-norm hidden per step. Collects
        the prefill hidden (all prompt positions) + each fed token's hidden, exactly the
        thinker_hidden_states the Talker consumes. Returns
        (gen_ids, prefill_hidden [1,P,H], step_hiddens [list of [1,1,H]], stop_reason).
        thinker_token_embeds are computed separately on the controller from the embed matrix."""
        import torch
        model = self.models[friendly]
        prefill_pos = mrope[0] if mrope else None
        base = mrope[1] if mrope else None
        # #pipefill: deliberately NOT chunked — capture_hidden needs the post-norm hidden for
        # EVERY prompt position in one reply; a chunk burst would return only the last chunk's.
        logits, prefill_hidden = await self._send(
            model, torch.tensor([prompt_ids], dtype=torch.long), 0, True,
            mm=mm, position_ids=prefill_pos, capture_hidden=True)
        cur = len(prompt_ids)
        model.kv_pos = cur
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0
        gen_ids: list[int] = []
        step_hiddens: list = []
        produced = 0
        stop = "length"
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            # #idle-unload: this path doesn't hold model.lock or bump active (unlike generate),
            # so stamp per-step progress — the idle-unload reaper reads last_token_ts and must
            # never call a mid-thinker speech/diag request "idle".
            model.last_token_ts = time.time()
            row = logits[0, -1]
            if ntok and ntok < int(row.shape[-1]):
                row = row.clone()
                row[ntok:] = float("-inf")
            tok_id = self._sample(row, temperature, top_p)
            produced += 1
            gen_ids.append(tok_id)
            if tok_id in model.eos_ids:
                stop = "stop"
                break
            if produced >= max_new:
                break
            dpos = [[base + produced - 1]] * 3 if base is not None else None
            logits, hid = await self._send(
                model, torch.tensor([[tok_id]], dtype=torch.long), cur, False,
                position_ids=dpos, capture_hidden=True)
            step_hiddens.append(hid)   # hidden of the token we just fed (tok_id)
            cur += 1
            model.kv_pos = cur
        return gen_ids, prefill_hidden, step_hiddens, stop

    async def _decode_spec(self, model, prompt_ids, max_new, k: int = 0):
        """Speculative greedy decode (M3): the local draft proposes K tokens, the
        pipeline verifies all K in one traversal, we accept the matched prefix + 1
        correction (bit-exact vs plain greedy), then roll the KV cache back.
        Falls back implicitly to M2e behaviour at K=0 acceptance (1 token/round).
        k>0 overrides SPEC_K (per-request, for tuning — a slower/more-distributed target
        favours a LARGER K so one verify pass amortizes more of the pipeline traversal)."""
        import torch
        # Empty prompt: nothing to prefill — short-circuit before the prefill _send (same guard
        # as _decode_plain; keeps the empty-ids probe off the wire).
        if not prompt_ids:
            yield None, "stop"
            return
        eos = model.eos_ids
        K = k if (k and k > 0) else SPEC_K
        # #spec-fold: ONE target sweep per round. The old shape paid TWO full weight sweeps
        # per round (the K-token verify PLUS a single-token _send to re-establish the target
        # logits after the bonus/correction token) — on a bandwidth-bound 70B that is ~570 ms
        # of sweeps per round, and spec measured SLOWER than plain greedy (3.63 vs 2.57 tok/s,
        # om3nbox). Folded: the bonus token is carried as `pending` (emitted but not yet in the
        # target KV) and rides at the FRONT of the next round's verify sequence, so its logits
        # come from the same traversal that verifies the drafts. Bit-exact vs plain greedy:
        # every emitted token is still the target's argmax over the full prefix.
        # #logits-diet: spec is greedy-BY-CONSTRUCTION and argmaxes the RAW row (no ntok mask —
        # the pre-existing plain-vs-spec inconsistency, deliberately PRESERVED: nt_clip=0 keeps
        # the head's argmax bit-exact with the legacy int(row.argmax()) below). A downgraded
        # chain replies the full row instead; both shapes are handled at every consumer.
        # #pipefill: spec's prefill is a plain text prefill (verify frames are untouched), so it
        # rides the same chunk-burst dispatcher; the last chunk's reply keeps nt argmax/nt_clip=0.
        # #prefix-kv: and the same cross-request prefix resume (spec is text-only by construction).
        _r0 = await self._prefill_reuse(model, prompt_ids, nt_mode="argmax", nt_clip=0)
        a0 = None if isinstance(_r0, list) else _r0[0, -1]
        cur = len(prompt_ids)              # target KV holds exactly tokens[0:cur]
        # #prefix-kv: publish the record (== tokens[0:cur], the invariant above). Each round
        # extends + re-publishes it after the crop (which nulls it); a verify/crop failure
        # leaves it None so the next request full-prefills.
        _kv = list(prompt_ids)
        model.kv_ids = _kv
        d_logits = await asyncio.to_thread(self._draft_prefill, model, prompt_ids)
        produced = 0
        rounds = drafted = matched = 0

        def _stats() -> None:
            if rounds:
                print(f"[spec] {model.friendly}: {produced} tok in {rounds} rounds "
                      f"(K={K}, accept {matched}/{drafted} = {matched / max(1, drafted):.0%}, "
                      f"{produced / rounds:.2f} tok/round)")

        # the prefill's own greedy token comes free — emit it and carry it as pending
        pending = (int(dict(_r0)[NT_TOKEN_IDS].reshape(-1)[-1]) if a0 is None   # #logits-diet
                   else int(a0.argmax()))
        produced += 1
        if pending in eos:
            yield None, "stop"
            return
        yield pending, None
        if produced >= max_new:
            yield None, "length"
            return
        d_logits = await asyncio.to_thread(self._draft_step, model, pending, cur)
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            # 1. draft K tokens greedily on the controller, chained after `pending`
            drafts = []
            dl = d_logits
            for i in range(K):
                dt = int(dl.argmax())
                drafts.append(dt)
                dl = await asyncio.to_thread(self._draft_step, model, dt, cur + 1 + i)
            # 2. ONE pipeline traversal verifies pending + all K drafts
            # #logits-diet: argmax mode + all_logits -> the head returns (K+1) int64 ids instead
            # of (K+1) full-vocab rows (~2.7 MB/round at K=8 on Qwen -> ~72 bytes). nt_clip=0
            # preserves this path's RAW-row argmax semantics; a downgraded chain still returns
            # the full [1,K+1,V] tensor and takes the legacy branch below — bit-identical tg
            # either way, so acceptance (and thus emitted tokens) never depends on the wire mode.
            try:
                V = await self._send(model, torch.tensor([[pending] + drafts], dtype=torch.long),
                                     cur, False, all_logits=True, nt_mode="argmax", nt_clip=0)
            except BaseException:
                # #prefix-kv: the K+1 appends may have landed on SOME stages only — record
                # invalid; null it so the next request full-prefills (binding audit caveat).
                model.kv_ids = None
                raise
            # 3. target's greedy tokens for positions cur+1 .. cur+K+1
            if isinstance(V, list):   # #logits-diet reduced reply
                tg = [int(t) for t in dict(V)[NT_TOKEN_IDS].reshape(-1).tolist()]
            else:
                tg = [int(V[0, i].argmax()) for i in range(K + 1)]
            # 4. accept the matched draft prefix + one target token (correction/bonus)
            m = 0
            while m < K and tg[m] == drafts[m]:
                m += 1
            accepted = tg[:m + 1]
            rounds += 1
            drafted += K
            matched += m
            # 5. roll target KV back: keep pending + the m accepted drafts, drop the rest
            await self._crop(model, cur + 1 + m)   # (nulls the #prefix-kv record)
            # #prefix-kv: the shard KV now holds _kv + [pending] + drafts[:m] (the matched
            # prefix == tg[:m]) — extend + re-publish so the record mirrors the cache exactly.
            # The bonus/correction token tg[m] is emitted but NOT in the KV (it rides the next
            # round's verify as `pending`), so it is deliberately not recorded.
            _kv.extend([pending] + drafts[:m])
            model.kv_ids = _kv
            # 6. emit
            for t in accepted:
                produced += 1
                if t in eos:
                    _stats()
                    yield None, "stop"
                    return
                yield t, None
                if produced >= max_new:
                    _stats()
                    yield None, "length"
                    return
            # 7. the bonus token becomes the next round's pending (target logits for it come
            # from the NEXT verify — no extra sweep); advance the draft to chain after it
            pending = accepted[-1]
            cur += 1 + m
            await asyncio.to_thread(self._draft_crop, model, cur)
            d_logits = await asyncio.to_thread(self._draft_step, model, pending, cur)
        _stats()
        yield None, "length"

    # -- MTP (nextn) self-speculation (#91) — the checkpoint's own draft head -------------------
    async def _ensure_mtp_head(self, model: LoadedModel):
        """Lazily build + cache the controller-resident MTP head for a model whose checkpoint ships
        one (mtp_num_hidden_layers>0). Returns the head or None (no MTP / load failed). The head is
        SMALL (embed + 1 layer + lm_head, a few GB) — NEVER the full model (see
        never-full-load-on-controller-box). First speculative request pays the one-time build."""
        if not hasattr(self, "_mtp_heads"):
            self._mtp_heads = {}
        if model.friendly in self._mtp_heads:
            return self._mtp_heads[model.friendly]
        d = await asyncio.to_thread(_controller_model_dir, model.target_id)

        def _has_mtp() -> bool:
            try:
                with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
                    cfg = json.load(fh)
                tc = cfg.get("text_config", cfg)
                if int(tc.get("mtp_num_hidden_layers", 0) or 0) <= 0:
                    return False
                # #91 (a): the 2-token spec VERIFY is NOT bit-exact on HYBRID linear-attention
                # (Gated-DeltaNet) layers — a q>1 decode chunk diverges from sequential q=1 steps
                # (chunked vs recurrent kernels), so accepted drafts follow a slightly-different
                # trajectory than plain greedy. Gate MTP off for hybrid checkpoints unless explicitly
                # allowed (mtp_allow_hybrid) — qwen3.6 is hybrid, so MTP self-spec is OFF by default.
                lt = tc.get("layer_types") or []
                if (any("linear" in str(x) for x in lt)
                        and not ENGINE_CONFIG.get("mtp_allow_hybrid", False)):
                    with contextlib.suppress(Exception):
                        log_activity(f"{model.friendly}: MTP self-spec OFF (hybrid linear-attn; q>1 "
                                     f"verify not bit-exact). Set config mtp_allow_hybrid=1 to override.")
                    return False
                return True
            except Exception:
                return False

        if not await asyncio.to_thread(_has_mtp):
            self._mtp_heads[model.friendly] = None    # negative-cache: don't re-check every request
            return None
        import mtp_core
        # Prefer the GPU: the MTP layer is a 256-expert MoE whose per-step forward must be << one
        # pipeline traversal or the draft overhead eats the speculation win. Best-effort with a CPU
        # fallback (the GPU may be full of the model's own shard). torch may be absent on a pure
        # controller, so import lazily.
        head = None
        try:
            import torch as _t
            devs = (["cuda:0"] if _t.cuda.is_available() else []) + ["cpu"]
        except Exception:
            devs = ["cpu"]
        for dev in devs:
            try:
                head = await asyncio.to_thread(mtp_core.load_mtp_head, d, dev)
                log_activity(f"{model.friendly}: MTP self-speculation head ready on {dev} (K=1)")
                break
            except Exception as exc:
                log_activity(f"{model.friendly}: MTP head load on {dev} failed ({exc!r})")
                with contextlib.suppress(Exception):
                    import torch as _t2
                    _t2.cuda.empty_cache()
        self._mtp_heads[model.friendly] = head    # None => negative-cache (plain decode)
        return head

    async def _decode_spec_mtp(self, model, prompt_ids, max_new, head):
        """#91 MTP self-speculative greedy decode. Each round: the main model's next token t comes
        from the verified context; the MTP head drafts ONE more token d (the next-next) from the
        trunk hidden + t; we verify [t, d] in ONE pipeline traversal (all_logits + capture_pre_norm)
        and accept d iff it equals the target's greedy — so every emitted token is identical to
        plain greedy (bit-exact). On accept the next state comes free from the verify pass (2 tokens
        / 1 traversal); on reject we emit the target's correct token and re-feed it (2 tokens / 2
        traversals).

        #91 CLOSED — NOT VIABLE on this fleet (kept gated off; see _has_mtp). The MTP forward is
        VALIDATED (~84-88% draft accept, perfect state-dict match), but two findings kill the win:
          (1) NO SPEEDUP on the compute-bound 2-stage GPU pipeline — a q=2 verify chunk costs ~2x a
              single token (measured x0.96), so fewer traversals != faster wall-clock here.
          (2) NOT BIT-EXACT on the HYBRID Gated-DeltaNet trunk — _crop (KV truncate) cannot roll back
              the linear-attention recurrent state on reject, and a q>1 chunk diverges from sequential
              q=1 steps (chunked vs recurrent kernels). qcheck saw pos0 logits diverge max_abs 9.125.
        Reviving it needs conv/recurrent state snapshot+restore around the verify AND a pipeline where
        a 2-token chunk is sub-linear. Left intact (validated, reusable) rather than deleted."""
        import mtp_core
        import torch
        if not prompt_ids:
            yield None, "stop"
            return
        eos = model.eos_ids
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0

        def _mask(row):
            if ntok and ntok < int(row.shape[-1]):
                row = row.clone()
                row[ntok:] = float("-inf")
            return row

        from transformers import DynamicCache
        # Prefill the MAIN model: per-position logits + PRE-norm hidden for the whole prompt.
        # #pipefill: deliberately NOT chunked — all_logits + capture_pre_norm need EVERY prompt
        # position in one reply; a chunk burst returns only the final chunk's. (#91 is gated
        # off anyway; this keeps the validated path intact for any future revival.)
        ml, h_pre = await self._send(model, torch.tensor([prompt_ids], dtype=torch.long), 0, True,
                                     all_logits=True, capture_pre_norm=True)
        P = len(prompt_ids)
        a0 = ml[0, P - 1]                     # logits predicting the token at position P
        h_prev = h_pre[:, P - 1:P, :]         # trunk hidden at position P-1
        # Prefill the MTP layer's OWN KV over the prompt so decode drafts attend the right context.
        mtp_kv = DynamicCache()
        if P >= 2:
            await asyncio.to_thread(mtp_core.mtp_prefill, head, h_pre[:, 0:P - 1, :],
                                    torch.tensor([prompt_ids[1:P]], dtype=torch.long), mtp_kv)
        mtp_len = P - 1                        # MTP-seq positions consumed so far (invariant: == cur-1)
        cur = P
        model.kv_pos = cur
        produced = 0
        accepts = rejects = 0
        _dbg = []                              # first rounds: (cur, t, d, tgt1, accepted) for tracing
        try:
            while produced < max_new:
                if model.friendly not in self.models or model.stage0_writer is None:
                    raise RuntimeError("pipeline went down mid-generation")
                t = int(_mask(a0).argmax())
                produced += 1
                if t in eos:
                    yield None, "stop"
                    return
                yield t, None
                if produced >= max_new:
                    return
                # Draft t_{cur+1}: consume t into the MTP cache (attends prefilled + prior context).
                draft_row = await asyncio.to_thread(mtp_core.mtp_step, head, mtp_kv, h_prev, t, mtp_len)
                mtp_len += 1
                d = int(_mask(draft_row).argmax())
                # verify [t, d] in one traversal; capture per-position logits + pre-norm hidden.
                V, H = await self._send(model, torch.tensor([[t, d]], dtype=torch.long), cur, False,
                                        all_logits=True, capture_pre_norm=True)
                tgt1 = int(_mask(V[0, 0]).argmax())      # target greedy for position cur+1
                if d == tgt1:                            # accept: next state is free from the verify
                    accepts += 1
                    second = d
                    next_a0, next_h, refeed = V[0, 1], H[:, 1:2, :], False
                else:                                    # reject: emit target token, drop wrong d
                    rejects += 1
                    second = tgt1
                    await self._crop(model, cur + 1)
                    next_a0 = next_h = None
                    refeed = True
                if len(_dbg) < 8:
                    _dbg.append((cur, t, d, tgt1, d == tgt1))
                produced += 1
                if second in eos:
                    yield None, "stop"
                    return
                yield second, None
                if produced >= max_new:
                    return
                # Commit `second` to the MTP cache (h_cur=H[0,0]) so subsequent drafts see it.
                await asyncio.to_thread(mtp_core.mtp_step, head, mtp_kv, H[:, 0:1, :], second, mtp_len)
                mtp_len += 1
                if refeed:                               # re-establish a0/h by feeding the real token
                    a0t, h_t = await self._send(model, torch.tensor([[second]], dtype=torch.long),
                                                cur + 1, False, capture_pre_norm=True)
                    a0, h_prev = a0t[0, -1], h_t[:, -1:, :]
                else:
                    a0, h_prev = next_a0, next_h
                cur += 2
                model.kv_pos = cur
            yield None, "length"
        finally:
            with contextlib.suppress(Exception):
                tot = accepts + rejects
                log_activity(f"[mtp] {model.friendly}: {accepts}/{tot} drafts accepted"
                             + (f" ({round(100 * accepts / tot)}%)" if tot else "")
                             + f"; first rounds (cur,t,d,tgt1,ok)={_dbg}")
