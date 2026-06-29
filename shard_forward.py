"""ShardForwardMixin: relocated Shard methods (m4c153 code-split). BODIES BYTE-IDENTICAL to the
originals in client.py; module globals injected at startup by state.bind() — see state.py.
Composed via ``class Shard(ShardForwardMixin, …)`` so self.* resolves across mixins by MRO. Worker-side
leaf module; in client.py EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


class ShardForwardMixin:

    def forward(self, x, cache_start: int = 0, reset: bool = True,
                all_logits: bool = False, inject=None, position_ids=None,
                capture_hidden: bool = False, capture_pre_norm: bool = False):
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
        if not lock.acquire(blocking=False):
            raise RuntimeError("shard busy with a prior (orphaned) forward — re-prefill required")
        try:
            return self._forward_impl(x, cache_start, reset, all_logits, inject,
                                      position_ids, capture_hidden, capture_pre_norm)
        finally:
            lock.release()

    def _forward_impl(self, x, cache_start: int = 0, reset: bool = True,
                      all_logits: bool = False, inject=None, position_ids=None,
                      capture_hidden: bool = False, capture_pre_norm: bool = False):
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
            self.kv = DynamicCache(config=self.cfg) if self._hybrid else DynamicCache()
        # NOTE: a defensive "reconcile self.kv length to cache_start" was tried here but MISFIRES on
        # multi-stage shards — DynamicCache.get_seq_length() inspects layer 0, which a mid/tail stage
        # (e.g. layers 24-48) doesn't own, so it reports 0 and a length check falsely trips on every
        # decode. The per-shard forward lock (see forward()) is the actual fix for the concurrent
        # orphaned-forward KV/mask desync crash; the cache_start==0 rebuild above covers sequence
        # starts. Stale-KV from a lost spec-decode crop frame is a separate, speculative-only edge.
        with torch.inference_mode():
            if self.uniform_device is not None:
                return self._forward_uniform(x, cache_start, all_logits, inject, position_ids,
                                             capture_hidden, capture_pre_norm)
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
            elif self._omni:   # Omni classic mRoPE needs [3,bs,seq]; text = 3x the same positions
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
            # additive mask (1,1,q,total): new position i attends keys 0..cache_start+i
            mask_cpu = torch.zeros((q, total), dtype=self.dtype)
            if q > 1:  # causal among the new tokens; prior keys all visible
                mask_cpu[:, cache_start:] = torch.triu(
                    torch.full((q, q), float("-inf"), dtype=self.dtype), diagonal=1)
            mask_cpu = mask_cpu.view(1, 1, q, total)
            cpos_cpu = torch.arange(cache_start, cache_start + q)
            aux: dict = {}

            def aux_for(dev):
                a = aux.get(dev)
                if a is None:
                    pe = None if _per_type else (cos_cpu.to(dev), sin_cpu.to(dev))
                    a = (mask_cpu.to(dev), pos_cpu.to(dev), pe, cpos_cpu.to(dev))
                    aux[dev] = a
                return a

            def _posemb_for(dev, lt):   # per-(dev,type) cos/sin for the Gemma4 per-type path
                k = (dev, lt); v = _pe.get(k)
                if v is None:
                    v = (_cos_t[lt].to(dev), _sin_t[lt].to(dev)); _pe[k] = v
                return v

            for _li, (layer, dev) in enumerate(zip(self.owned_layers, self.layer_devices)):
                if h.device != dev:
                    h = h.to(dev)
                mask, pos, pos_emb, cache_position = aux_for(dev)
                if _per_type:
                    out = layer(h, attention_mask=mask, position_ids=pos,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=_posemb_for(dev, _lts[_ls + _li]),
                                shared_kv_states=_skv, cache_position=cache_position)
                elif self._hybrid:
                    # Per-layer-type mask: full-attn gets the causal mask; linear-attn
                    # (Gated-DeltaNet) gets None (text-only, no padding). The qwen layer
                    # has no cache_position param (it tracks position via the cache).
                    m = mask if getattr(layer, "layer_type", "full_attention") == "full_attention" else None
                    out = layer(h, attention_mask=m, position_ids=pos,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=pos_emb)
                else:
                    out = layer(h, attention_mask=mask, position_ids=pos,
                                past_key_values=self.kv, use_cache=True,
                                position_embeddings=pos_emb, cache_position=cache_position)
                h = out[0] if isinstance(out, tuple) else out
            if self.has_head:
                if h.device != self.head_device:
                    h = h.to(self.head_device)
                # #P6 speech: when capturing thinker hidden states for the talker, compute the
                # post-norm hidden for ALL positions (talker needs every prompt token at prefill
                # + each decoded token); logits only for the sampled position(s).
                nh = self.norm(h)
                sel = nh if all_logits else nh[:, -1:, :]   # verify needs every position
                logits = self.head(sel).to(self.cpu)
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
                         capture_pre_norm: bool = False):
        """Single-device fast path (see _finalize_placement). Everything — embed,
        every layer, norm/head, rotary — lives on self.uniform_device, so cos/sin,
        positions, cache_position and (for prefill only) the causal mask are built
        directly there. No per-token CPU rotary compute, no host->device copies,
        and for single-token decode no mask at all (the lone query attends every
        cached key). Numerically identical to the general path on the same device;
        on CPU it stays bit-exact. Called inside torch.inference_mode()."""
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
        elif self._omni:   # Omni classic mRoPE needs [3,bs,seq]
            rot_pos = pos.unsqueeze(0).expand(3, -1, -1).contiguous()
        ref = torch.empty(1, dtype=self.dtype, device=dev)
        _rotary = self.model.model.rotary_emb
        _lts = getattr(self.cfg, "layer_types", None)
        # Gemma 4: per-attention-type rotary (see _forward). Build cos/sin per type on `dev`; each
        # layer picks by global index + gets shared_kv_states={}. Other archs: one shared rotary.
        _per_type = bool(_lts) and hasattr(_rotary, "%s_inv_freq" % _lts[0])
        _ls = int(getattr(self, "layer_start", 0) or 0)
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
        if q > 1:   # prefill: causal among the new tokens; all prior keys visible
            mask = torch.zeros((q, total), dtype=self.dtype, device=dev)
            mask[:, cache_start:] = torch.triu(
                torch.full((q, q), float("-inf"), dtype=self.dtype, device=dev), diagonal=1)
            mask = mask.view(1, 1, q, total)
        else:       # decode: one query sees every key -> no mask needed
            mask = None
        for _li, layer in enumerate(self.owned_layers):
            if _per_type:
                out = layer(h, attention_mask=mask, position_ids=pos,
                            past_key_values=self.kv, use_cache=True,
                            position_embeddings=_pe_t[_lts[_ls + _li]],
                            shared_kv_states=_skv, cache_position=cache_position)
            elif self._hybrid:
                m = mask if getattr(layer, "layer_type", "full_attention") == "full_attention" else None
                out = layer(h, attention_mask=m, position_ids=pos,
                            past_key_values=self.kv, use_cache=True,
                            position_embeddings=pos_emb)
            else:
                out = layer(h, attention_mask=mask, position_ids=pos,
                            past_key_values=self.kv, use_cache=True,
                            position_embeddings=pos_emb, cache_position=cache_position)
            h = out[0] if isinstance(out, tuple) else out
        if self.has_head:
            nh = self.norm(h)
            sel = nh if all_logits else nh[:, -1:, :]
            logits = self.head(sel).to(self.cpu)
            if capture_pre_norm:   # #91 MTP: PRE-final-norm trunk hidden (see general path)
                hsel = h if all_logits else h[:, -1:, :]
                return logits, hsel.to(self.cpu)
            if capture_hidden:
                return logits, nh.to(self.cpu)
            return logits
        return h.to(self.cpu)
