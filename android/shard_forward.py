"""ShardForwardMixin: relocated Shard methods (m4c153 code-split). BODIES BYTE-IDENTICAL to the
originals in client.py; module globals injected at startup by state.bind() — see state.py.
Composed via ``class Shard(ShardForwardMixin, …)`` so self.* resolves across mixins by MRO. Worker-side
leaf module; in client.py EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import time

_ORPHAN_CANCEL_GRACE_S = 10.0   # #fwd-cancel: seconds a fresh forward waits for an orphan to yield


class _ForwardSuperseded(RuntimeError):
    """A running (orphaned) forward was asked to yield by a newer forward — raised at a layer
    boundary so the stale thread releases _fwd_lock promptly instead of running to completion.
    Propagates like the old 'shard busy' error: the abandoned req_id has no pending future, so the
    controller ignores it. #fwd-cancel."""


class ShardForwardMixin:

    def forward(self, x, cache_start: int = 0, reset: bool = True,
                all_logits: bool = False, inject=None, position_ids=None,
                capture_hidden: bool = False, capture_pre_norm: bool = False,
                bidir_spans=None):
        # #fwd-serialize: serialize forwards on this shard so a still-running ORPHANED forward (from a
        # reclaimed/disconnected gen — the worker thread can't be cancelled) can't concurrently mutate
        # the shared self.kv underneath a fresh forward, which desyncs the KV length from the causal
        # mask -> the SDPA "expanded size N must match M" crash. Non-blocking: a racing new forward
        # fails FAST (the controller re-prefills) rather than blocking a thread-pool slot for the
        # orphan's full (possibly minutes-long CPU prefill) runtime. Uncontended on the normal path
        # (a model's forwards are sequential via the controller's per-model lock).
        # Lazily ensure the lock exists — a Shard built via a path that doesn't run the full __init__
        # (cached/skeleton install) would otherwise AttributeError here and break ALL generation.
        lock = getattr(self, "_fwd_lock", None)
        if lock is None:
            lock = self._fwd_lock = threading.Lock()
        cancel = getattr(self, "_fwd_cancel", None)
        if cancel is None:
            cancel = self._fwd_cancel = threading.Event()
        if not lock.acquire(blocking=False):
            # #fwd-cancel: an ORPHANED forward holds the lock — the controller reclaimed its request
            # (hop timeout / disconnect / gen-stall watchdog) but a worker thread can't be cancelled,
            # so it keeps grinding and wedges the shard ("shard busy ... re-prefill required" on every
            # later request until it finishes — minutes when CPU-spilled). Signal it to bail at its
            # NEXT layer boundary (cooperative check in _forward_impl/_forward_uniform), then wait a
            # short grace for it to release. If it yields we proceed (the controller re-prefills into a
            # fresh cache); if it's stuck inside one un-yieldable op past the grace, fail fast as
            # before (worker-restart is the backstop). Self-healing: no controller protocol change.
            cancel.set()
            if not lock.acquire(timeout=_ORPHAN_CANCEL_GRACE_S):
                raise RuntimeError("shard busy with a prior (orphaned) forward — re-prefill required")
        cancel.clear()   # WE own the lock now — clear so a stale signal can't abort our own forward
        # #fwd-watchdog: stamp start + a per-layer progress heartbeat (updated in the layer loops).
        # The worker watchdog escalates a forward whose progress ts goes STALE (stuck inside one
        # un-yieldable op, where cooperative cancel can't help) to a supervisor relaunch.
        # #prefill-progress: adopt the req_id staged by worker_net AFTER winning the lock, so the
        # heartbeat's progress report names the forward that is ACTUALLY running (an orphaned
        # forward keeps its original rid — the controller ignores rids no longer pending).
        self._fwd_cur_rid = getattr(self, "_fwd_next_rid", None)
        self._fwd_started_ts = self._fwd_progress_ts = time.time()
        try:
            return self._forward_impl(x, cache_start, reset, all_logits, inject,
                                      position_ids, capture_hidden, capture_pre_norm,
                                      bidir_spans)
        finally:
            lock.release()

    def _make_kv_quant_cache(self, name):
        """#172 TurboQuant KV cache (kv_quant != 'none'): a DynamicCache whose layers store K/V
        QUANTIZED (rotated Lloyd-Max; un-rotated on read, so the model's attention runs UNCHANGED).
        Built against the INSTALLED transformers' DynamicCache/DynamicLayer (validated on 5.12.1).
        Correctness-first: ANY failure (module missing, unknown preset, transformers API drift)
        falls back to a plain bf16 DynamicCache so generation never breaks. Gated to non-hybrid
        models by the caller (linear-attn layers hold conv/recurrent state, not standard KV)."""
        from transformers import DynamicCache
        try:
            from transformers.cache_utils import DynamicLayer
            import kv_quant as _kq
            pb = _kq.preset_bits(name)
            if not pb:
                return DynamicCache()
            kb, vb = pb
            torch = self.torch
            def _qf(head_dim, device, dtype):
                return _kq.TurboQuantizer(torch, head_dim, key_bits=kb, value_bits=vb,
                                          device=device, dtype=dtype)
            # #172 small-model quality: keep the most-recent W tokens in full bf16 (KIVI-style
            # residual window) so turbo3/turbo4 stay coherent below ~14B. 0 (default) = the
            # deployed whole-cache-quant behaviour, byte-identical; recommend ~64-128 for small models.
            try:
                _rw = max(0, int(os.environ.get("INFINITEMODEL_KV_QUANT_RESIDUAL", "0")))
            except Exception:
                _rw = 0
            return _kq.make_turboquant_cache(DynamicCache, DynamicLayer, _qf, residual_window=_rw)
        except Exception as exc:
            print(f"[kv_quant] '{name}' unavailable ({exc!r}) -> plain bf16 KV", flush=True)
            return DynamicCache()

    def _forward_impl(self, x, cache_start: int = 0, reset: bool = True,
                      all_logits: bool = False, inject=None, position_ids=None,
                      capture_hidden: bool = False, capture_pre_norm: bool = False,
                      bidir_spans=None):
        """Run this stage's layers with an incremental KV cache (M2e). Always called holding
        self._fwd_lock (see forward()).
        x = token ids (first stage) or hidden states (mid stage), covering the
        `q` positions starting at absolute position `cache_start`. `reset` starts
        a fresh cache (prefill); otherwise the cached prior KV is reused (decode).
        inject = (positions, embeds) splices multimodal embeds into stage-0's embed
        output (#22 inc 3); None for the normal text path.
        Returns hidden states, or — on the last stage — logits for just the last
        position, or for ALL positions when all_logits=True (speculative verify)."""
        torch = self.torch
        # #kv-reset-on-seqstart: ALWAYS rebuild the cache when this frame starts a new sequence
        # (cache_start == 0), not only when the controller flags reset. A cancelled / gen-stall-
        # watchdog-reclaimed generation can leave a STALE KV cache on this stage: the controller
        # dropped the request, but our forward runs in a thread that can't be cancelled, so it
        # still finishes and populates self.kv. The next request's prefill arrives at position 0;
        # if `reset` were ever out of sync with cache_start we'd append to that stale KV and the
        # attention mask (q vs cache_start+q) would mismatch ("expanded size N must match M").
        # Treating position 0 as an unconditional fresh start makes stale KV impossible to reuse —
        # cache_start==0 is ONLY ever a sequence start (decode/verify always send cache_start>0).
        if reset or self.kv is None or cache_start == 0:
            from transformers import DynamicCache
            # Hybrid arch: a config-typed cache pre-creates the right per-layer slot
            # (conv+recurrent for linear-attn layers, KV for full-attn) so the
            # Gated-DeltaNet layers can store/read state instead of IndexError-ing on
            # an empty generic cache. Reused across prefill + every decode step.
            _kvq = getattr(self, "kv_quant", "none")
            if self._hybrid:
                self.kv = DynamicCache(config=self.cfg)
            elif _kvq and _kvq != "none":
                # #172 TurboQuant: quantized resting KV (un-rotated on read -> attention unchanged).
                self.kv = self._make_kv_quant_cache(_kvq)
            elif getattr(self, "kv_offload", False):
                # #kv-offload: KV lives in system RAM; transformers 5.x folded cache offloading
                # into DynamicCache(offloading=True) — each layer's K/V rests on CPU and is
                # prefetched to the compute device on a side stream during forward, so attention
                # still runs on-device. Only meaningful when layers sit on GPU (a CPU shard's KV
                # is already in RAM); ANY failure (no CUDA, transformers API drift) falls back to
                # a plain DynamicCache so generation never breaks.
                # ROCm/HIP: offloading is DISABLED — live-validated GARBLED on gfx1151 (TheRock
                # torch 2.12a): the side-stream H2D prefetch races the compute stream, so decode
                # was corrupted AND nondeterministic at temperature 0 (plain load: bit-identical).
                # An APU's "VRAM" is unified system RAM anyway, so offload buys nothing there.
                self.kv = None
                try:
                    if getattr(self.torch.version, "hip", None):
                        print("[kv_offload] ROCm/HIP: offloaded-KV prefetch garbles decode "
                              "(stream race, validated live) -> plain on-device KV", flush=True)
                    elif any(getattr(d, "type", "") == "cuda"
                             for d in (getattr(self, "layer_devices", None) or [])):
                        self.kv = DynamicCache(offloading=True)
                except Exception as exc:
                    print(f"[kv_offload] unavailable ({exc!r}) -> plain bf16 KV on device", flush=True)
                if self.kv is None:
                    self.kv = DynamicCache()
            else:
                self.kv = DynamicCache()
            # #cudagraph: a new sequence invalidates the graph's StaticCache mirror (it must be
            # re-synced from this fresh DynamicCache at the next decode). Cheap, idempotent.
            self._gkv_pos = 0
        # NOTE: a defensive "reconcile self.kv length to cache_start" was tried here but MISFIRES on
        # multi-stage shards — DynamicCache.get_seq_length() inspects layer 0, which a mid/tail stage
        # (e.g. layers 24-48) doesn't own, so it reports 0 and a length check falsely trips on every
        # decode. The per-shard forward lock (see forward()) is the actual fix for the concurrent
        # orphaned-forward KV/mask desync crash; the cache_start==0 rebuild above covers sequence
        # starts. Stale-KV from a lost spec-decode crop frame is a separate, speculative-only edge.
        with torch.inference_mode():
            if self.uniform_device is not None:
                return self._forward_uniform(x, cache_start, all_logits, inject, position_ids,
                                             capture_hidden, capture_pre_norm, bidir_spans)
            # #gemma4-bidir: OR image-span bidirectional attention into the PREFILL masks when the
            # text config asks for it (use_bidirectional_attention='vision') and the controller
            # sent the image runs. Prefill-only: a decoded token is text (block id -1) -> no effect.
            _bidir_sp = (bidir_spans if (bidir_spans
                         and getattr(self.cfg, "use_bidirectional_attention", None) == "vision")
                         else None)
            h = self.embed(x.to(self.embed_device)) if self.has_embed else x
            if inject is not None and self.has_embed:
                h = self._splice_mm(h, inject)
            q = h.shape[1]
            total = cache_start + q
            # Positional aux is built on CPU (rotary_emb lives there) and moved to
            # each layer's device on demand — cos/sin depend only on positions, so
            # this is identical to the single-device path. aux is per-call.
            ref = torch.empty(1, dtype=self.dtype)
            pos_cpu = torch.arange(cache_start, cache_start + q).unsqueeze(0)   # [1,q] -> layers
            # #22 inc 4: feed 3D mRoPE positions [3,1,q] to the rotary (it interleaves the t/h/w
            # sections and returns standard [bs,q,dim] cos/sin); layers keep 1D position_ids
            # (unused for rotary since we pass position_embeddings). None -> plain 1D arange.
            rot_pos = pos_cpu
            if position_ids is not None:
                rot_pos = torch.as_tensor(position_ids, dtype=torch.long)
                if rot_pos.dim() == 2:
                    rot_pos = rot_pos.unsqueeze(1)
            elif self._omni or getattr(self, "_mrope3d", False):   # Omni/Qwen2.5-VL mRoPE needs [3,bs,seq]; text = 3x the same positions
                rot_pos = pos_cpu.unsqueeze(0).expand(3, -1, -1).contiguous()
            _rotary = self.model.model.rotary_emb
            _lts = getattr(self.cfg, "layer_types", None)
            # Gemma 4: PER-attention-type rotary (sliding vs full) — the rotary exposes {type}_inv_freq
            # buffers and forward(x, pos, layer_type). Build cos/sin for each unique type; each layer
            # picks its own by global index and gets shared_kv_states={} (last-of-type layers WRITE it;
            # nothing READS it at num_kv_shared_layers=0). Other archs keep the single shared rotary.
            _per_type = bool(_lts) and hasattr(_rotary, "%s_inv_freq" % _lts[0])
            _ls = int(getattr(self, "layer_start", 0) or 0)
            cos_cpu = sin_cpu = None
            _cos_t, _sin_t, _skv, _pe = {}, {}, {}, {}
            if _per_type:
                for _t in dict.fromkeys(_lts):
                    _c, _s = _rotary(ref, rot_pos, _t)
                    _cos_t[_t], _sin_t[_t] = _c.to(self.dtype), _s.to(self.dtype)
            else:
                cos_cpu, sin_cpu = _rotary(ref, rot_pos)
                cos_cpu, sin_cpu = cos_cpu.to(self.dtype), sin_cpu.to(self.dtype)
            cpos_cpu = torch.arange(cache_start, cache_start + q)

            def _posemb_for(dev, lt):   # per-(dev,type) cos/sin for the Gemma4 per-type path
                k = (dev, lt); v = _pe.get(k)
                if v is None:
                    v = (_cos_t[lt].to(dev), _sin_t[lt].to(dev)); _pe[k] = v
                return v

            # #prefill-chunk: process a long PREFILL in query-chunks so the explicit additive mask never
            # forces SDPA to materialize the full [1,H,q,total] score tensor (the math-backend fallback on
            # a CPU shard or a no-flash GPU -> OOM on long prompts). _run_layers runs every owned layer
            # (across their devices, KV accumulating) over one (sub)sequence with a freshly-built, per-dev
            # additive mask (1,1,cl,cache_start+off+cl); math-identical to the single pass (validated).
            # Standard-attention only; per-type (Gemma4) / hybrid (linear-attn state) / mRoPE-omni keep the
            # original single full pass (do_chunk False below -> byte-identical pre-chunk behavior).
            def _run_layers(h_, off, cl):
                cs_off = cache_start + off                  # absolute start of this chunk's queries
                tot_off = cs_off + cl                       # cache length once this chunk's keys land
                if _bidir_sp and cl > 1:   # #gemma4-bidir: full causal mask + image-span overlay
                    m_cpu = self._causal_addmask(cs_off, cl, tot_off, "cpu", self.dtype, None, _bidir_sp)
                else:
                    m_cpu = torch.zeros((cl, tot_off), dtype=self.dtype)
                    if cl > 1:   # causal among the chunk's tokens; all prior keys visible
                        m_cpu[:, cs_off:] = torch.triu(
                            torch.full((cl, cl), float("-inf"), dtype=self.dtype), diagonal=1)
                    m_cpu = m_cpu.view(1, 1, cl, tot_off)
                # #gemma4-sliding: windowed causal mask for sliding_attention layers (per-type path).
                # #gemma4-bidir: same image-span overlay OR'd onto the sliding mask (HF OR's both).
                sm_cpu = None
                if _per_type:
                    _sw = getattr(self.cfg, "sliding_window", None)
                    if _sw:
                        sm_cpu = self._causal_addmask(cs_off, cl, tot_off, "cpu", self.dtype, _sw,
                                                      _bidir_sp if cl > 1 else None)
                p_cpu = pos_cpu[:, off:off + cl]
                cp_cpu = cpos_cpu[off:off + cl]
                cc_cpu = None if _per_type else cos_cpu[:, off:off + cl, :]
                sc_cpu = None if _per_type else sin_cpu[:, off:off + cl, :]
                _a: dict = {}

                def _aux(dev):
                    a = _a.get(dev)
                    if a is None:
                        pe = None if _per_type else (cc_cpu.to(dev), sc_cpu.to(dev))
                        sm = sm_cpu.to(dev) if sm_cpu is not None else None
                        a = (m_cpu.to(dev), p_cpu.to(dev), pe, cp_cpu.to(dev), sm)
                        _a[dev] = a
                    return a

                for _li, (layer, dev) in enumerate(zip(self.owned_layers, self.layer_devices)):
                    if self._fwd_cancel.is_set():   # #fwd-cancel: a newer forward asked this orphan to yield
                        raise _ForwardSuperseded("forward superseded by a newer request")
                    self._fwd_progress_ts = time.time()   # #fwd-watchdog: per-layer liveness heartbeat
                    if h_.device != dev:
                        h_ = h_.to(dev)
                    mask, pos, pos_emb, cache_position, smask = _aux(dev)
                    if _per_type:
                        # #gemma4-sliding: windowed mask for sliding layers, plain causal for full.
                        _m = smask if (smask is not None
                                       and getattr(getattr(layer, "self_attn", None),
                                                   "sliding_window", None)) else mask
                        out = layer(h_, attention_mask=_m, position_ids=pos,
                                    past_key_values=self.kv, use_cache=True,
                                    position_embeddings=_posemb_for(dev, _lts[_ls + _li]),
                                    shared_kv_states=_skv, cache_position=cache_position)
                    elif self._hybrid:
                        # Per-layer-type mask: full-attn gets the causal mask; linear-attn
                        # (Gated-DeltaNet) gets None (text-only, no padding). The qwen layer
                        # has no cache_position param (it tracks position via the cache).
                        m = mask if getattr(layer, "layer_type", "full_attention") == "full_attention" else None
                        out = layer(h_, attention_mask=m, position_ids=pos,
                                    past_key_values=self.kv, use_cache=True,
                                    position_embeddings=pos_emb)
                    else:
                        out = layer(h_, attention_mask=mask, position_ids=pos,
                                    past_key_values=self.kv, use_cache=True,
                                    position_embeddings=pos_emb, cache_position=cache_position)
                    h_ = out[0] if isinstance(out, tuple) else out
                return h_

            cstep = self._prefill_chunk_len(q)
            do_chunk = (q > 1 and cstep < q and not _per_type and not self._hybrid
                        and not self._omni and not getattr(self, "_mrope3d", False)
                        and position_ids is None and not _bidir_sp)   # #gemma4-bidir: no cross-chunk image split
            if not do_chunk:
                h = _run_layers(h, 0, q)
            else:
                outs = []
                off = 0
                while off < q:
                    cl = min(cstep, q - off)
                    outs.append(_run_layers(h[:, off:off + cl, :], off, cl))
                    off += cl
                h = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
            if self.has_head:
                if h.device != self.head_device:
                    h = h.to(self.head_device)
                # #P6 speech: when capturing thinker hidden states for the talker, compute the
                # post-norm hidden for ALL positions (talker needs every prompt token at prefill
                # + each decoded token); logits only for the sampled position(s).
                nh = self.norm(h)
                sel = nh if all_logits else nh[:, -1:, :]   # verify needs every position
                logits = self._softcap_logits(self.head(sel)).to(self.cpu)   # #gemma4 final logit softcap
                if capture_pre_norm:
                    # #91 MTP: return the PRE-final-norm trunk hidden (what the checkpoint's MTP
                    # head consumes via pre_fc_norm_hidden) — distinct from capture_hidden's
                    # POST-norm hidden (speech). Same position slice as the logits.
                    hsel = h if all_logits else h[:, -1:, :]
                    return logits, hsel.to(self.cpu)
                if capture_hidden:
                    return logits, nh.to(self.cpu)
                return logits
            return h.to(self.cpu)

    def _forward_uniform(self, x, cache_start: int, all_logits: bool, inject=None,
                         position_ids=None, capture_hidden: bool = False,
                         capture_pre_norm: bool = False, bidir_spans=None):
        """Single-device fast-path router. #cudagraph (opt-in, default OFF): for a graph-eligible
        single-node standard-attention model, a plain single-token decode is served by a captured
        CUDA graph via the COPY-HANDOFF path — prefill (and everything else) stays on the eager
        DynamicCache path below, untouched. _maybe_graph_decode returns None to defer to eager."""
        if self._graph_enabled() and inject is None and position_ids is None and bidir_spans is None:
            r = self._maybe_graph_decode(x, cache_start, all_logits, capture_hidden, capture_pre_norm)
            if r is not None:
                return r
        return self._forward_uniform_eager(x, cache_start, all_logits, inject, position_ids,
                                           capture_hidden, capture_pre_norm, bidir_spans)

    def _forward_uniform_eager(self, x, cache_start: int, all_logits: bool, inject=None,
                               position_ids=None, capture_hidden: bool = False,
                               capture_pre_norm: bool = False, bidir_spans=None):
        """The eager single-device path (DynamicCache). Bit-identical to the pre-cudagraph behavior —
        everything (embed, layers, norm/head, rotary) lives on self.uniform_device; single-token
        decode uses no mask (the lone query attends every cached key). Also serves as the cuda-graph
        self-check reference + permanent fallback. Called inside torch.inference_mode()."""
        torch = self.torch
        dev = self.uniform_device
        h = self.embed(x.to(dev)) if self.has_embed else x.to(dev)
        if inject is not None and self.has_embed:
            h = self._splice_mm(h, inject)
        q = h.shape[1]
        total = cache_start + q
        pos = torch.arange(cache_start, cache_start + q, device=dev).unsqueeze(0)   # [1,q] -> layers
        rot_pos = pos   # #22 inc 4: 3D mRoPE positions feed the rotary; see general path
        if position_ids is not None:
            rot_pos = torch.as_tensor(position_ids, dtype=torch.long, device=dev)
            if rot_pos.dim() == 2:
                rot_pos = rot_pos.unsqueeze(1)
        elif self._omni or getattr(self, "_mrope3d", False):   # Omni/Qwen2.5-VL mRoPE needs [3,bs,seq]
            rot_pos = pos.unsqueeze(0).expand(3, -1, -1).contiguous()
        ref = torch.empty(1, dtype=self.dtype, device=dev)
        _rotary = self.model.model.rotary_emb
        _lts = getattr(self.cfg, "layer_types", None)
        # Gemma 4: per-attention-type rotary (see _forward). Build cos/sin per type on `dev`; each
        # layer picks by global index + gets shared_kv_states={}. Other archs: one shared rotary.
        _per_type = bool(_lts) and hasattr(_rotary, "%s_inv_freq" % _lts[0])
        _ls = int(getattr(self, "layer_start", 0) or 0)
        # #gemma4-bidir: image runs attend bidirectionally in the PREFILL mask (prefill-only: q>1).
        _bidir_sp = (bidir_spans if (bidir_spans and q > 1
                     and getattr(self.cfg, "use_bidirectional_attention", None) == "vision")
                     else None)
        _pe_t, _skv = {}, {}
        pos_emb = None
        if _per_type:
            for _t in dict.fromkeys(_lts):
                _c, _s = _rotary(ref, rot_pos, _t)
                _pe_t[_t] = (_c.to(self.dtype), _s.to(self.dtype))
        else:
            cos, sin = _rotary(ref, rot_pos)
            pos_emb = (cos.to(self.dtype), sin.to(self.dtype))
        cache_position = torch.arange(cache_start, cache_start + q, device=dev)

        def _run(h_, mask_, pos_, pe_, cpos_, smask_=None):   # run this stage's layers over one (sub)sequence
            for _li, layer in enumerate(self.owned_layers):
                if self._fwd_cancel.is_set():   # #fwd-cancel: yield to a newer forward (orphan cleanup)
                    raise _ForwardSuperseded("forward superseded by a newer request")
                self._fwd_progress_ts = time.time()   # #fwd-watchdog: per-layer liveness heartbeat
                if _per_type:
                    # #gemma4-sliding: a sliding_attention layer must see a WINDOWED causal mask
                    # (only the last sliding_window keys), not the plain causal mask the full_attention
                    # layers get — otherwise its attention leaks past the window and generation diverges
                    # once the context exceeds sliding_window (validated bit-exact vs the HF reference).
                    _m = smask_ if (smask_ is not None
                                    and getattr(getattr(layer, "self_attn", None),
                                                "sliding_window", None)) else mask_
                    out = layer(h_, attention_mask=_m, position_ids=pos_,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=_pe_t[_lts[_ls + _li]],
                                shared_kv_states=_skv, cache_position=cpos_)
                elif self._hybrid:
                    m = mask_ if getattr(layer, "layer_type", "full_attention") == "full_attention" else None
                    out = layer(h_, attention_mask=m, position_ids=pos_,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=pe_)
                else:
                    out = layer(h_, attention_mask=mask_, position_ids=pos_,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=pe_, cache_position=cpos_)
                h_ = out[0] if isinstance(out, tuple) else out
            return h_

        # #prefill-chunk: long PREFILL is processed in query-chunks so the explicit additive mask never
        # forces SDPA to materialize the full [1,H,q,total] score tensor — the math-backend fallback on
        # devices without an efficient/flash attention kernel (notably ROCm gfx1151) that OOMs on long
        # prompts (the 43 GiB single-alloc seen on om3nbox). Each chunk runs every layer with its own
        # [1,1,cl,cache_start+off+cl] mask while the KV cache accumulates; math-identical to the single
        # pass (validated: max|dlogits|~1e-6, argmax exact at every position). Only the standard-attention
        # path chunks; per-type (Gemma4), hybrid (linear-attn recurrent state) and mRoPE/omni keep the
        # original single full pass (do_chunk False -> byte-identical to the pre-chunk behavior).
        cstep = self._prefill_chunk_len(q)
        do_chunk = (q > 1 and cstep < q and not _per_type and not self._hybrid
                    and not self._omni and not getattr(self, "_mrope3d", False)
                    and position_ids is None and not _bidir_sp)   # #gemma4-bidir: no cross-chunk image split
        if not do_chunk:
            if _per_type:
                # #gemma4-sliding: build BOTH masks; _run hands each layer the one for its type. The
                # full-causal mask (window=None) reproduces the old prefill-causal / decode-see-all
                # behavior for full_attention layers; the windowed mask restricts sliding layers.
                # #gemma4-bidir: _bidir_sp OR's the image-span overlay onto both (HF OR's both).
                fmask = self._causal_addmask(cache_start, q, total, dev, self.dtype, None, _bidir_sp)
                _sw = getattr(self.cfg, "sliding_window", None)
                smask = (self._causal_addmask(cache_start, q, total, dev, self.dtype, _sw, _bidir_sp)
                         if _sw else None)
                h = _run(h, fmask, pos, pos_emb, cache_position, smask)
            elif q > 1:   # prefill: causal among the new tokens; all prior keys visible
                if _bidir_sp:   # #gemma4-bidir: causal + image-span overlay (non-per-type bidir model)
                    mask = self._causal_addmask(cache_start, q, total, dev, self.dtype, None, _bidir_sp)
                else:
                    mask = torch.zeros((q, total), dtype=self.dtype, device=dev)
                    mask[:, cache_start:] = torch.triu(
                        torch.full((q, q), float("-inf"), dtype=self.dtype, device=dev), diagonal=1)
                    mask = mask.view(1, 1, q, total)
                h = _run(h, mask, pos, pos_emb, cache_position)
            else:       # decode: one query sees every key -> no mask needed
                h = _run(h, None, pos, pos_emb, cache_position)
        else:
            outs = []
            off = 0
            while off < q:
                cl = min(cstep, q - off)
                cs_off = cache_start + off       # absolute start of this chunk's queries
                tot_off = cs_off + cl            # cache length once this chunk's keys are appended
                mask = torch.zeros((cl, tot_off), dtype=self.dtype, device=dev)
                mask[:, cs_off:] = torch.triu(
                    torch.full((cl, cl), float("-inf"), dtype=self.dtype, device=dev), diagonal=1)
                mask = mask.view(1, 1, cl, tot_off)
                pe_c = (pos_emb[0][:, off:off + cl, :], pos_emb[1][:, off:off + cl, :])
                outs.append(_run(h[:, off:off + cl, :], mask, pos[:, off:off + cl],
                                 pe_c, cache_position[off:off + cl]))
                off += cl
            h = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        if self.has_head:
            nh = self.norm(h)
            sel = nh if all_logits else nh[:, -1:, :]
            logits = self._softcap_logits(self.head(sel)).to(self.cpu)   # #gemma4 final logit softcap
            if capture_pre_norm:   # #91 MTP: PRE-final-norm trunk hidden (see general path)
                hsel = h if all_logits else h[:, -1:, :]
                return logits, hsel.to(self.cpu)
            if capture_hidden:
                return logits, nh.to(self.cpu)
            return logits
        return h.to(self.cpu)

    def _causal_addmask(self, cache_start: int, q: int, total: int, dev, dtype, window=None,
                        spans=None):
        """#gemma4-sliding: additive attention mask [1,1,q,total]. A key at absolute position kpos is
        visible to a query at absolute position qpos iff 0 <= (qpos - kpos) (< window when given).
        window=None -> plain causal (full_attention); window=int -> sliding-window causal. This mirrors
        the HF reference's create_causal_mask / create_sliding_window_causal_mask exactly (validated
        bit-exact), so the per-type path can hand sliding_attention layers a windowed mask while
        full_attention layers keep the plain causal one. Cheap: two aranges + a compare.
        #gemma4-bidir: `spans` = list of (start,end) absolute half-open image-token runs. When given,
        OR-in a blockwise overlay so any two positions in the SAME run attend bidirectionally
        (use_bidirectional_attention='vision') — exactly HF's or_masks(base, blockwise_overlay(
        block_sequence_ids)). None -> pure causal/sliding (byte-identical to the pre-bidir mask)."""
        torch = self.torch
        qpos = torch.arange(cache_start, cache_start + q, device=dev).view(q, 1)
        kpos = torch.arange(0, total, device=dev).view(1, total)
        dist = qpos - kpos
        allowed = dist >= 0                       # causal (a query never sees a future key)
        if window:
            allowed = allowed & (dist < int(window))   # ...and only the last `window` keys
        if spans:
            # blockwise overlay: unmask iff q and k share the same image run (group >= 0)
            qsid = self._span_ids(cache_start, q, spans, dev).view(q, 1)
            ksid = self._span_ids(0, total, spans, dev).view(1, total)
            allowed = allowed | ((qsid == ksid) & (qsid >= 0))
        m = torch.zeros((q, total), dtype=dtype, device=dev)
        m = m.masked_fill(~allowed, float("-inf"))
        return m.view(1, 1, q, total)

    def _span_ids(self, start: int, n: int, spans, dev):
        """#gemma4-bidir: per-position block id [n] — the index of the image span containing each
        absolute position in [start, start+n), or -1 for text. Mirrors the reference's
        get_block_sequence_ids_for_mask (contiguous vision runs -> 0,1,2,...; text -> -1)."""
        torch = self.torch
        pos = torch.arange(start, start + n, device=dev)
        sid = torch.full((n,), -1, dtype=torch.long, device=dev)
        for i, (s, e) in enumerate(spans):
            sid = torch.where((pos >= int(s)) & (pos < int(e)),
                              torch.full_like(sid, i), sid)
        return sid

    def _softcap_logits(self, logits):
        """#gemma4: Gemma-4 (like Gemma-2) caps its final logits at ±final_logit_softcapping via
        logits = cap * tanh(logits / cap) — see Gemma4ForCausalLM.forward. Monotonic, so greedy
        argmax is unchanged, but it bounds the distribution temperature/top-p sampling sees (without
        it the head's raw logits sample differently from the reference). No-op for any model whose
        config lacks the field (every non-Gemma model here)."""
        cap = getattr(self.cfg, "final_logit_softcapping", None)
        if cap:
            logits = self.torch.tanh(logits / cap) * cap
        return logits

    def _prefill_chunk_len(self, q: int) -> int:
        """#prefill-chunk: query-chunk length for a long PREFILL pass. The shard hands HF layers an
        explicit additive float mask, which disables SDPA's flash backend; on a device without the
        mem-efficient backend (notably ROCm gfx1151, and the CPU math path) SDPA then materializes the
        full [1, H, q, total] score tensor -> O(H*q^2) memory -> OOM on long prompts (the 43 GiB single
        alloc seen on om3nbox). Chunking the query dim caps peak score memory to [1, H, C, total],
        cutting it by ~q/C; math-identical (validated). Returns the chunk length, or q (a single full
        pass = byte-identical to the pre-chunk path) for short prompts / decode / when disabled.
        Tunable: INFINITEMODEL_PREFILL_CHUNK=<tokens> (default 2048; 0 disables chunking entirely)."""
        import os
        try:
            c = int(os.environ.get("INFINITEMODEL_PREFILL_CHUNK", "2048"))
        except ValueError:
            c = 2048
        return q if (c <= 0 or q <= c) else c

    # ---- #cudagraph: opt-in single-node CUDA-graph decode (default OFF) -----------------------
    # A batch-1 decode step is ~80% per-op launch/dispatch overhead (measured ~5.6x on a 4070). With
    # INFINITEMODEL_CUDA_GRAPH set, a graph-eligible single-node standard-attention model serves plain
    # single-token decode by replaying a captured model.forward over a fixed-size StaticCache MIRROR.
    # COPY-HANDOFF: prefill (and verify/capture) stay on the proven eager DynamicCache path, untouched;
    # the mirror is synced from it at the first decode. The graph is trusted only after a TRUE
    # self-check — a replay at a position != the capture position, compared against the eager
    # DynamicCache decode (so a position-dependent capture bug is caught) — else it DISABLES permanently
    # and falls back to eager. Inert unless the env flag is set; the default path is byte-identical.
    # See docs/ACCELERATION.md.

    def _is_static_cache(self, c) -> bool:
        try:
            from transformers import StaticCache
            return isinstance(c, StaticCache)
        except Exception:
            return False

    def _graph_maxlen(self) -> int:
        """Fixed KV size for the StaticCache = the value of INFINITEMODEL_CUDA_GRAPH if it's an int
        (the operator sets it to the serving ctx), else a safe default."""
        import os
        try:
            n = int(os.environ.get("INFINITEMODEL_CUDA_GRAPH", ""))
            if n > 1:
                return n
        except Exception:
            pass
        return 8192

    def _graph_enabled(self) -> bool:
        """Model-level gate (cached): env flag on + single-GPU CUDA + owns embed&head + standard
        attention (no hybrid/omni/Gemma-per-type) + not multimodal. The per-model latch
        self._graph_ok goes False permanently on a failed build/self-check."""
        en = getattr(self, "_graph_en", None)
        if en is None:
            import os
            ok = False
            try:
                ud = self.uniform_device
                if (os.environ.get("INFINITEMODEL_CUDA_GRAPH")
                        and ud is not None and getattr(ud, "type", None) == "cuda"
                        and self.has_embed and self.has_head
                        and not self._hybrid and not self._omni and not getattr(self, "_mrope3d", False)
                        and getattr(self, "kv_quant", "none") == "none"   # #172: graph mirrors a
                        # StaticCache that can't re-quantize TurboQuant KV -> stay eager when active
                        and not getattr(self, "kv_offload", False)   # #kv-offload: graph can't mirror
                        # a CPU-resident OffloadedCache -> stay eager when active
                        and not getattr(self, "_mm_capable", False)):
                    _lts = getattr(self.cfg, "layer_types", None)
                    _rot = self.model.model.rotary_emb
                    per_type = bool(_lts) and hasattr(_rot, "%s_inv_freq" % _lts[0])
                    ok = not per_type
            except Exception:
                ok = False
            self._graph_en = en = ok
            if not hasattr(self, "_graph_ok"):
                self._graph_ok = None
        return bool(en) and (getattr(self, "_graph_ok", None) is not False)

    def _maybe_graph_decode(self, x, cache_start, all_logits, capture_hidden, capture_pre_norm):
        """Route a plain single-token decode to the captured graph; return None to defer to the eager
        DynamicCache path (prefill / verify / capture / overflow / disabled). On a non-decode frame
        while the graph mirror has advanced past the frozen DynamicCache, reconcile self.kv from the
        mirror first (so the eager path is correct) and stop graphing this model."""
        if getattr(self, "_graph_ok", None) is False:
            return None
        maxlen = self._graph_maxlen()
        plain = (x.shape[1] == 1 and not all_logits and not capture_hidden
                 and not capture_pre_norm and cache_start > 0)
        if not plain or cache_start + 1 > maxlen:
            if (getattr(self, "_gcap", None) is not None
                    and getattr(self, "_gkv_pos", 0) >= cache_start and cache_start > 0):
                try:
                    self._sync_static_to_dynamic(cache_start)   # rebuild self.kv from the mirror
                except Exception as exc:
                    print(f"[cudagraph] reconcile failed ({exc!r})", flush=True)
                self._graph_ok = False
            return None
        try:
            return self._graph_decode(x, cache_start, maxlen)
        except Exception as exc:
            self._graph_ok = False
            self._gcap = None
            try:
                if getattr(self, "_gkv_pos", 0) >= cache_start and cache_start > 0:
                    self._sync_static_to_dynamic(cache_start)
            except Exception:
                pass
            print(f"[cudagraph] decode error ({exc!r}) -> eager fallback", flush=True)
            return None

    def _ensure_gkv(self, maxlen):
        """Lazily create the graph's StaticCache mirror + force per-layer buffer allocation (a dummy
        decode forward — StaticCache layers init lazily; we overwrite the dummy via copy after)."""
        torch = self.torch
        if getattr(self, "_gkv", None) is not None and self._is_static_cache(self._gkv):
            return
        from transformers import StaticCache
        dev = self.uniform_device
        self._gkv = StaticCache(config=self.cfg, max_cache_len=maxlen)
        dummy = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.model(input_ids=dummy, past_key_values=self._gkv, use_cache=True,
                   cache_position=torch.zeros(1, dtype=torch.long, device=dev))

    def _copy_dynamic_to_static(self, n):
        """Copy KV positions 0..n-1 from the proven DynamicCache (self.kv) into the StaticCache mirror."""
        sl = self._gkv.layers
        dl = self.kv.layers
        for i in range(len(sl)):
            sl[i].keys[:, :, :n, :].copy_(dl[i].keys[:, :, :n, :])
            sl[i].values[:, :, :n, :].copy_(dl[i].values[:, :, :n, :])

    def _sync_static_to_dynamic(self, n):
        """Rebuild self.kv as a DynamicCache holding mirror positions 0..n-1 (used when a non-decode
        frame must run on the eager path but the graph mirror is ahead of the frozen DynamicCache)."""
        from transformers import DynamicCache
        dyn = DynamicCache(config=self.cfg) if self._hybrid else DynamicCache()
        for i, sl in enumerate(self._gkv.layers):
            dyn.update(sl.keys[:, :, :n, :].clone(), sl.values[:, :, :n, :].clone(), i)
        self.kv = dyn

    def _decode_compute(self):
        """The captured per-token step: full model.forward over the StaticCache MIRROR reading the
        static input/position buffers — HF handles rotary/mask/cache correctly. Returns logits [1,1,V]
        on device; graph-safe (the position-driven mask is built from the static _g_pos tensor)."""
        return self.model(input_ids=self._g_input, past_key_values=self._gkv,
                          use_cache=True, cache_position=self._g_pos).logits

    def _graph_decode(self, x, cache_start, maxlen):
        """Copy-handoff graph decode with a TRUE DynamicCache self-check.
        - contiguity: the mirror must hold 0..cache_start-1; if not, (re)sync it from the proven
          DynamicCache (first decode of a sequence, or after an intervening eager frame).
        - phase 0 (capture): capture model.forward at cache_start; ship the EAGER result (graph not yet
          trusted); self.kv advances via the eager decode.
        - phase 1 (validate): REPLAY at the next position (!= capture pos, so a baked/position-dependent
          capture bug is caught) and compare vs the eager DynamicCache decode; activate or DISABLE.
        - phase 2 (active): replay only.
        Eager decodes keep self.kv current through the 2-step check; after activation self.kv is frozen
        and the mirror leads (reconciled on demand by _maybe_graph_decode)."""
        torch = self.torch
        dev = self.uniform_device
        if getattr(self, "_gkv_pos", 0) != cache_start:
            # mirror not contiguous for this position -> rebuild from the proven DynamicCache
            self._ensure_gkv(maxlen)
            self._copy_dynamic_to_static(cache_start)
            self._gkv_pos = cache_start
            self._gcap = None
            self._g_phase = 0
        if getattr(self, "_gcap", None) is None:           # phase 0: capture + ship eager
            self._g_input = x.to(dev).reshape(1, 1).long().clone()
            self._g_pos = torch.tensor([cache_start], device=dev, dtype=torch.long)
            self._fwd_progress_ts = time.time()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    _ = self._decode_compute()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._g_logits = self._decode_compute()
            self._gcap = g
            self._gkv_pos = cache_start + 1
            self._g_phase = 1
            self._fwd_progress_ts = time.time()
            return self._forward_uniform_eager(x, cache_start, False, None, None, False, False)
        if getattr(self, "_g_phase", 2) == 1:              # phase 1: replay-at-new-pos self-check
            self._g_input.copy_(x.to(dev).reshape(1, 1).long())
            self._g_pos.fill_(cache_start)
            self._fwd_progress_ts = time.time()
            self._gcap.replay()
            self._gkv_pos = cache_start + 1
            gl = self._g_logits[:, -1:, :].to(self.cpu)
            ref = self._forward_uniform_eager(x, cache_start, False, None, None, False, False)
            rel = ((gl.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6)).item()
            self._fwd_progress_ts = time.time()
            if rel > 0.05:
                self._graph_ok = False
                self._gcap = None
                print(f"[cudagraph] replay self-check vs DynamicCache rel={rel:.3f} -> DISABLED (eager)",
                      flush=True)
                return ref
            self._g_phase = 2
            print(f"[cudagraph] decode ACTIVE (replay self-check vs DynamicCache rel={rel:.4f}, "
                  f"maxlen={maxlen})", flush=True)
            return gl
        self._g_input.copy_(x.to(dev).reshape(1, 1).long())   # phase 2: replay only
        self._g_pos.fill_(cache_start)
        self._fwd_progress_ts = time.time()
        self._gcap.replay()
        self._gkv_pos = cache_start + 1
        return self._g_logits[:, -1:, :].to(self.cpu)
