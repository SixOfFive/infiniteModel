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


class EngineGenMixin:

    async def embed(self, friendly: str, input_ids, attention_mask) -> list:
        """Run one encoder forward on `friendly`'s single node and return the pooled, L2-normed
        sentence vectors as a list of float lists. Mirrors _send: pack ids+mask into ONE
        two-tensor frame, await the worker's single-tensor 'embedding' reply via self.pending."""
        model = self.models[friendly]
        target = model.target_id
        async with model.lock:
            if model.stage0_writer is None:
                raise RuntimeError("embedding model not connected")
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
                return vecs.tolist()
            finally:
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

    async def _send(self, model: LoadedModel, x, cache_position: int, reset: bool,
                    all_logits: bool = False, mm=None, position_ids=None,
                    capture_hidden: bool = False, capture_pre_norm: bool = False):
        """Push one frame (token ids) through `model`'s pipeline and return last-stage
        logits — last position only, or every position when all_logits=True (verify).
        mm = (positions, embeds_tensor) (#22 inc 3): on a prefill (reset), a companion
        'mm' frame is sent FIRST with the same req_id so stage 0 splices those embeds into
        its embed output at `positions` before running the layers."""
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
            if not reset:   # decode/verify step: classic single-budget wait
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

    async def _crop(self, model: LoadedModel, length: int) -> None:
        """Tell every stage of `model` to truncate its KV cache to `length` (spec rollback).
        Fire-and-forget: in-order delivery on each stage's connection guarantees the
        crop is applied before the next frame the controller sends afterwards."""
        if model.stage0_writer is not None:
            nbytes = await _write_frame(model.stage0_writer,
                                        {"model_id": model.target_id, "kind": "crop",
                                         "cache_position": length}, b"")
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
        # PER-REPLICA lock: different models AND different replicas of one model decode
        # concurrently; requests routed to the SAME replica queue on its lock.
        # Track queue depth for /status (queued = waiting on this model's lock; active = generating).
        model.queued += 1
        acquired = False
        try:
            async with model.lock:
                acquired = True
                model.queued -= 1
                model.active += 1
                model.last_token_ts = time.time()   # #gen-stall-watchdog: start the no-progress timer at gen begin
                model.gen_started_ts = model.last_token_ts   # #active-decode-stall: prefill marker (token 1 advances last_token_ts past this)
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
                    mtp_head = None
                    if (speculative and greedy and mm is None and model.draft_model is None):
                        with contextlib.suppress(Exception):
                            mtp_head = await self._ensure_mtp_head(model)
                    if speculative and model.draft_model is not None and greedy and mm is None:
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
                                                             min_p=min_p, sampling=sampling):
                            if item[0] is not None:
                                _ntoks += 1
                                _out_ids.append(item[0])   # #ctx-history
                                model.last_token_ts = time.time()   # #gen-stall-watchdog progress marker
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
                            mrope=None, min_p: float = 0.0, sampling=None):
        """Prefill-once + one-token-at-a-time KV-cache decode (M2e). mm=(positions, embeds)
        (#22 inc 3) splices multimodal embeds into the PREFILL only; decode steps are plain.
        mrope=(prefill_position_ids [3][q], base) (#22 inc 4) carries 3D image positions:
        the prefill uses the full layout; each decode token uses [base+step] on all 3 dims."""
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
        logits = await self._send(model, torch.tensor([prompt_ids], dtype=torch.long), 0, True,
                                  mm=mm, position_ids=prefill_pos)
        cur = len(prompt_ids)
        model.kv_pos = cur          # KV depth so far (prompt); climbs per decode token
        produced = 0
        # #21: this model's lm_head can be WIDER than its text tokenizer (a multimodal
        # head carries vision/audio placeholder ids the text tokenizer can't decode).
        # Selecting one of those ids crashed detokenization ("list index out of
        # range") and showed up as empty/failed generation. Mask logits beyond the
        # tokenizer's decodable range so we only ever emit a real text token.
        try:
            ntok = len(model.tokenizer)
        except Exception:
            ntok = 0
        while produced < max_new:
            if model.friendly not in self.models or model.stage0_writer is None:
                raise RuntimeError("pipeline went down mid-generation")
            row = logits[0, -1]
            if ntok and ntok < int(row.shape[-1]):
                row = row.clone()
                row[ntok:] = float("-inf")
            if _pen:   # #runtime-knobs: repetition penalties reshape the logits pre-sampling
                row = self._penalized(row, prompt_ids, _hist, _sp)
            tok_id = self._sample(row, temperature, top_p, min_p, top_k=_top_k, gen=_gen)
            _hist.append(tok_id)
            if produced == 0:
                with contextlib.suppress(Exception):
                    print(f"[gen] {model.friendly}: first token id={tok_id} "
                          f"head_vocab={int(logits.shape[-1])} len(tok)={ntok} "
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
            logits = await self._send(model, torch.tensor([[tok_id]], dtype=torch.long), cur,
                                      False, position_ids=dpos)
            cur += 1
            model.kv_pos = cur
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
        a0 = (await self._send(model, torch.tensor([prompt_ids], dtype=torch.long), 0, True))[0, -1]
        cur = len(prompt_ids)              # target KV holds exactly tokens[0:cur]
        d_logits = await asyncio.to_thread(self._draft_prefill, model, prompt_ids)
        produced = 0
        rounds = drafted = matched = 0

        def _stats() -> None:
            if rounds:
                print(f"[spec] {model.friendly}: {produced} tok in {rounds} rounds "
                      f"(K={K}, accept {matched}/{drafted} = {matched / max(1, drafted):.0%}, "
                      f"{produced / rounds:.2f} tok/round)")

        # the prefill's own greedy token comes free — emit it and carry it as pending
        pending = int(a0.argmax())
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
            V = await self._send(model, torch.tensor([[pending] + drafts], dtype=torch.long),
                                 cur, False, all_logits=True)
            # 3. target's greedy tokens for positions cur+1 .. cur+K+1
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
            await self._crop(model, cur + 1 + m)
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
