"""ShardBuildMixin: relocated Shard methods (m4c153 code-split). BODIES BYTE-IDENTICAL to the
originals in client.py; module globals injected at startup by state.bind() — see state.py.
Composed via ``class Shard(ShardBuildMixin, …)`` so self.* resolves across mixins by MRO. Worker-side
leaf module; in client.py EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


class ShardBuildMixin:

    @staticmethod
    def _mod_bytes(module) -> int:
        # params + buffers — QuantLinear stores its int8 weight + scale as buffers
        return sum(t.numel() * t.element_size()
                   for t in list(module.parameters()) + list(module.buffers()))

    @staticmethod
    def _mod_gpu_bytes(module) -> int:
        # Bytes of a module's tensors that ACTUALLY live on a CUDA device. Device-accurate (unlike
        # _mod_bytes) so a MoE-split layer — attention on GPU, experts on CPU inside the same module
        # tree — reports only its GPU-resident weight. Used for gpu_bytes/size_vram accounting.
        return sum(t.numel() * t.element_size()
                   for t in list(module.parameters()) + list(module.buffers())
                   if t.device.type == "cuda")

    def _kv_dims(self, ctx: int):
        """(num_kv_heads, head_dim) from cfg for full-ctx KV sizing, or (0, 0) if ctx<=0 / unknown."""
        if not ctx or ctx <= 0:
            return 0, 0
        cfg = self.cfg
        nh = int(getattr(cfg, "num_attention_heads", 0) or 0)
        nkv = int(getattr(cfg, "num_key_value_heads", nh) or nh or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        hd = int(getattr(cfg, "head_dim", 0) or (hidden // nh if nh else 0))
        if nkv <= 0 or hd <= 0:
            return 0, 0
        return nkv, hd

    def _kv_bf16_per_layer(self, ctx: int) -> int:
        """Full-ctx bf16 KV bytes ONE layer grows into (k+v). Also the per-layer TRANSIENT peak under
        kv_quant — dequant rebuilds one layer's full bf16 K/V at a time. 0 if ctx/dims unknown."""
        nkv, hd = self._kv_dims(ctx)
        return 2 * int(ctx) * nkv * hd * 2 if nkv else 0

    def _kv_bytes_per_layer(self, ctx: int) -> int:
        """Full-ctx KV bytes ONE layer RESTS at: bf16 normally, or the smaller BIT-PACKED TurboQuant
        footprint when kv_quant is active (self.kv_quant != 'none'). The placement budget reserves this
        per layer; kv_reserve_probe adds one bf16 transient/device for the sequential dequant peak — the
        two stay mirrored. kv_quant='none' -> bf16, bit-identical to the pre-#172 reservation. Any error
        resolving the quant size falls back to bf16 (conservative — never under-reserve -> decode OOM)."""
        name = (getattr(self, "kv_quant", "none") or "none")
        if name != "none":
            try:
                nkv, hd = self._kv_dims(ctx)
                if nkv:
                    import kv_quant
                    per_tok = kv_quant.kv_quant_bytes_per_token_per_layer(name, nkv, hd)
                    if per_tok > 0:
                        return int(ctx) * per_tok
            except Exception:
                pass   # bf16 fallback below (conservative)
        return self._kv_bf16_per_layer(ctx)

    def _kv_layer_mask(self) -> list:
        """#7: per-OWNED-layer bool — True = a layer that holds a growing full-ctx KV cache.
        For a hybrid linear-attention arch (cfg.layer_types: interleaved Gated-DeltaNet vs
        full-attention, e.g. qwen3-next / qwen3.6) ONLY the 'full_attention' layers grow a KV;
        the linear-attn layers keep a small fixed recurrent state (treated as ~0 here). For
        every dense/standard model (no layer_types) ALL True -> bit-identical to the old
        uniform reservation. CONSERVATIVE on any uncertainty (missing/short layer_types,
        out-of-range index, parse error): default a layer to True so we never under-reserve
        and risk decode OOM. owned layer i = global layer self.layer_start + i."""
        n = len(self.owned_layers)
        lt = getattr(self.cfg, "layer_types", None)
        if not getattr(self, "_hybrid", False) or not lt:
            return [True] * n
        base = int(getattr(self, "layer_start", 0) or 0)
        mask = []
        for i in range(n):
            gi = base + i
            try:
                mask.append(lt[gi] == "full_attention")
            except Exception:
                mask.append(True)   # unknown -> reserve full KV (never under-reserve)
        return mask

    def _place_modules(self, device: str, gpu_mem_gb: float, ctx: int = 0,
                       gpu_budget_gb: float = -1.0) -> None:
        """Assign embed / each owned layer / norm+head to CPU or GPU and move them.
        Modes: cpu | gpu(cuda) whole-on-GPU | cpu+gpu(hybrid) offload-by-VRAM |
        auto whole-if-fits-else-hybrid. Always falls back to CPU without CUDA.
        gpu_budget_gb (#95): the controller's committed-aware GPU budget for THIS stage (free VRAM
        after co-resident models' weights + reserved KV, minus the plan floor). >=0 caps placement;
        <0 means the controller didn't send one (old controller) -> uncapped (legacy behavior)."""
        torch = self.torch
        cpu = self.cpu
        mode = (device or "cpu").lower()
        want_gpu = mode in ("gpu", "cuda", "auto", "cpu+gpu", "hybrid")
        cuda_ok = False
        if want_gpu:
            try:
                cuda_ok = torch.cuda.is_available()
            except Exception:
                cuda_ok = False
        if not (want_gpu and cuda_ok):
            self.embed_device = cpu
            self.layer_devices = [cpu] * len(self.owned_layers)
            self.layer_split = [False] * len(self.owned_layers)
            self.norm_device = self.head_device = cpu
            self.placement = "cpu" + ("" if not want_gpu else " (no CUDA → CPU)")
            self.gpu_bytes = 0
            self.gpu_kv_bytes = 0
            self._finalize_placement()
            return

        gpu = torch.device("cuda:0")
        free, _total = torch.cuda.mem_get_info(0)
        # NEVER oversubscribe the card: the controller may plan GPU placement optimistically (sizing
        # against an EMPTY GPU), but coexisting models already hold VRAM. Bound EVERY GPU budget by
        # the LIVE free VRAM (mem_get_info reflects what other resident models hold right now), and
        # reserve THIS shard's full-ctx KV (k+v grows on-GPU during decode) + a CUDA/activation
        # margin — so a 2nd model leaves room instead of filling VRAM to 100% and OOM-ing at
        # generation (which also kills coexisting models' decode). kv_per_layer=0 when ctx unknown.
        # #95 coexistence: clamp the VISIBLE free VRAM to the controller's committed-aware budget for
        # this stage (free VRAM after co-resident models' weights + reserved KV, minus the plan floor).
        # Done HERE so EVERY downstream GPU decision keys off it — the mode=auto whole-on-GPU check
        # (free*0.85), the hybrid default budget (free*0.85), live_free, and the placement string. A
        # co-resident model's card LOOKS free until it faults its full-ctx KV; without this clamp a 2nd
        # shard consolidates onto that VRAM and OOMs the resident model's decode (qwen3+14b). budget 0
        # (GPU fully committed) -> free 0 -> all layers spill to CPU, no GPU grab. <0 = no value sent
        # (old controller) -> uncapped, unchanged behavior.
        if gpu_budget_gb >= 0:
            free = min(int(free), int(gpu_budget_gb * GB))
        GPU_SAFETY = int(0.4 * GB)
        kv_per_layer = self._kv_bytes_per_layer(ctx)
        # #172: under kv_quant, kv_per_layer above is the PACKED resting footprint; the dequant still
        # rebuilds ONE layer's full bf16 K/V at a time (pipeline runs layers sequentially), so reserve
        # one bf16 layer of transient headroom on top of the per-layer resting sum. 0 when
        # kv_quant='none' (kv_per_layer is then already bf16 -> the whole path is bit-identical).
        kv_transient = (self._kv_bf16_per_layer(ctx)
                        if (getattr(self, "kv_quant", "none") or "none") != "none" else 0)
        # #kv-offload: the KV cache lives in system RAM (OffloadedCache), so NO per-layer KV — nor the
        # dequant transient — is reserved against VRAM; that headroom goes to model layers instead.
        if getattr(self, "kv_offload", False):
            kv_per_layer = 0
            kv_transient = 0
        # #7: only full-attention layers grow a full-ctx KV; hybrid linear-attn layers don't.
        # kv_lyr[i] = the KV bytes owned layer i actually reserves (kv_per_layer or 0). For a
        # dense model this is kv_per_layer for every layer (unchanged). Mirrors kv_reserve_probe.
        kv_lyr = [kv_per_layer if h else 0 for h in self._kv_layer_mask()]
        live_free = max(0, int(free) - GPU_SAFETY)
        nlyr = len(self.owned_layers)
        # #moe-offload: when enabled, a MoE layer that can't fit GPU whole is SPLIT — attention+norms
        # on GPU, the routed-expert block left on CPU (instead of dragging the whole layer to CPU).
        # Gated to int4/int8 — those quantize experts into HEAP buffers (Packed4Tensor3D fused, or
        # QuantLinear4 per-expert), so leaving them on CPU has no mmap-reclaim issue; bf16 experts
        # (possibly mmap) fall back to the whole-layer path.
        moe_off = (bool(getattr(self, "_moe_offload", False))
                   and self.quant in ("int4", "int8"))
        moe_blocks = ([_find_moe_block(l) for l in self.owned_layers]
                      if moe_off else [(None, None)] * nlyr)
        self.layer_split = [False] * nlyr
        whole_need = self.loaded_bytes + sum(kv_lyr) + (kv_transient if any(kv_lyr) else 0)
        whole = ((mode in ("gpu", "cuda") and whole_need <= live_free)
                 or (mode == "auto" and whole_need < free * 0.85))
        if whole:
            self.embed_device = gpu if self.has_embed else cpu
            self.layer_devices = [gpu] * nlyr
            self.norm_device = self.head_device = gpu
            self.placement = f"cuda:all ({nlyr} layers)"
        else:  # hybrid: greedily fill a VRAM budget (capped by live-free), spill the rest to CPU
            budget = int(gpu_mem_gb * GB) if gpu_mem_gb > 0 else int(free * 0.85)
            budget = min(budget, live_free)   # live-free cap -> can't oversubscribe a shared card
            # #172: hold back one bf16 layer for the kv_quant dequant transient before placing layers,
            # so the greedy fill can't consume the headroom the per-step peak needs (0 when not kv_quant).
            budget = max(0, budget - kv_transient)
            used = 0

            def fits(nbytes: int, kv: int = 0) -> bool:
                nonlocal used
                if used + nbytes + kv <= budget:
                    used += nbytes + kv
                    return True
                return False

            self.embed_device = gpu if (self.has_embed and fits(self._mod_bytes(self.embed))) else cpu
            # each GPU-resident layer must hold its weights AND the KV it will grow into at this ctx.
            # #moe-offload: a splittable MoE layer charges only its MIXER (attention+norms = whole
            # layer minus the MoE block) + KV to the GPU budget; the big expert block stays in RAM.
            self.layer_devices = []
            for i, l in enumerate(self.owned_layers):
                _blk = moe_blocks[i][1]
                if moe_off and _blk is not None:
                    mixer_b = self._mod_bytes(l) - self._mod_bytes(_blk)
                    if fits(mixer_b, kv_lyr[i]):
                        self.layer_devices.append(gpu)   # attention->GPU, experts stay CPU (split)
                        self.layer_split[i] = True
                    else:
                        self.layer_devices.append(cpu)   # mixer didn't fit -> whole layer to CPU
                elif fits(self._mod_bytes(l), kv_lyr[i]):
                    self.layer_devices.append(gpu)
                else:
                    self.layer_devices.append(cpu)
            if self.has_head:
                hb = self._mod_bytes(self.head) + self._mod_bytes(self.norm)
                self.norm_device = self.head_device = gpu if fits(hb) else cpu
            else:
                self.norm_device = self.head_device = cpu
            ng = sum(1 for d in self.layer_devices if d.type == "cuda")
            nsp = sum(1 for s in self.layer_split if s)
            self.placement = (f"cpu+gpu: {ng}/{nlyr} layers on GPU"
                              + (f" ({nsp} MoE-split: attn->GPU, experts->CPU)" if nsp else "")
                              + f" (budget {budget / GB:.1f} GB of {free / GB:.1f} free, "
                              f"+{kv_per_layer * ng / GB:.1f} GB KV)")

        if self.has_embed:
            self.embed.to(self.embed_device)
        for i, (lyr, d) in enumerate(zip(self.owned_layers, self.layer_devices)):
            if self.layer_split[i]:
                # #moe-offload split: move every child EXCEPT the MoE block (attention, norms) to GPU
                # and leave the block (router+experts+shared) on CPU. Moving only the non-MoE children
                # avoids a transient whole-layer GPU spike (the experts never touch the GPU). A bridge
                # wraps the block so the layer's forward bridges hidden GPU<->CPU around it.
                _attr, _blk = moe_blocks[i]
                for _nm, _child in lyr.named_children():
                    if _nm != _attr:
                        _child.to(gpu)
                for _pn, _p in list(lyr._parameters.items()):
                    if _p is not None:
                        _p.data = _p.data.to(gpu)
                for _bn, _b in list(lyr._buffers.items()):
                    if _b is not None:
                        lyr._buffers[_bn] = _b.to(gpu)
                setattr(lyr, _attr, _moe_bridge_cls()(_blk, self.cpu))
            else:
                lyr.to(d)
        if self.has_head:
            self.norm.to(self.norm_device)
            self.head.to(self.head_device)
        # CPU-resident layers: .to(cpu) is a NO-OP, so for a non-quantized (bf16) shard
        # they stay as mmap VIEWS into the weight temp file. On Windows a mapped file
        # can't be deleted and its pages can't be trimmed on unload -> the node retains
        # the whole spilled shard in RAM after unload (beast kept ~58 GB). Materialize
        # them to heap (clone) so the mmap drops once this build returns and the file is
        # deletable + the RAM reclaimable. int8 already copies to heap during quant, so
        # skip it. Guarded: only when the transient 2x (mmap + clones) fits free RAM,
        # else leave the mmap (correctness over a possible load-time OOM).
        if self.quant != "int8" and not getattr(self, "_streamed", False):
            self._materialize_cpu_layers()   # streamed shards are already heap (no mmap to drop)
        # #moe-offload post-split assertion: a split layer MUST keep its experts on CPU. If the
        # blanket move ever dragged them to GPU (the bug the critics flagged), fail the load LOUDLY
        # here rather than silently OOM the card / mis-report gpu_bytes.
        if any(self.layer_split):
            for i, lyr in enumerate(self.owned_layers):
                if self.layer_split[i]:
                    _br = getattr(lyr, moe_blocks[i][0])
                    if any(b.device.type != "cpu" for b in _br.buffers()):
                        raise RuntimeError(
                            f"moe_offload: layer {i} expert buffers not all on CPU after split")
        # bytes actually resident on the GPU (controller sums these for size_vram). DEVICE-ACCURATE
        # (_mod_gpu_bytes) so a split layer counts only its GPU mixer, not its CPU experts.
        gb = 0
        if self.has_embed:
            gb += self._mod_gpu_bytes(self.embed)
        for lyr in self.owned_layers:
            gb += self._mod_gpu_bytes(lyr)
        if self.has_head:
            gb += self._mod_gpu_bytes(self.norm) + self._mod_gpu_bytes(self.head)
        self.gpu_bytes = gb
        # #int4-vram-probe: int4 weights have been seen resident ~bf16-sized on gfx1151/ROCm (devstral
        # 13.5 GB int4 cache -> 43.9 GB VRAM). Free any transient build/dequant buffers the caching
        # allocator still holds, then log the TRUTH — in-use tensor bytes vs the allocator's reserved
        # pool vs our per-module sum — so a real bf16-resident footprint (int4 not applied) is told
        # apart from a reclaimable pool / an accounting over-count. Safe: empty_cache never frees
        # in-use tensors.
        if any(d.type == "cuda" for d in self.layer_devices):
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
                _al = torch.cuda.memory_allocated() / GB
                _rv = torch.cuda.memory_reserved() / GB
                print(f"[int4-vram] L{self.layer_start}-{self.layer_end} quant={self.quant} "
                      f"sum={gb/GB:.2f}GB in-use={_al:.2f}GB reserved={_rv:.2f}GB", flush=True)
        # full-ctx KV these GPU-resident layers will grow into — reported so the controller can
        # RESERVE it against coexisting loads (a 2nd model must not eat this model's KV space).
        # #172: per-layer resting (packed under kv_quant) + one bf16 dequant transient when any KV
        # layer is GPU-resident (0 for kv_quant='none' -> bit-identical to the pre-#172 report).
        _ng_cuda = sum(1 for d in self.layer_devices if d.type == "cuda")
        self.gpu_kv_bytes = kv_per_layer * _ng_cuda + (kv_transient if _ng_cuda else 0)
        # #moe-offload diagnostic: surface WHY the split did/didn't engage (on=flag+quant gate,
        # blocks=layers where a MoE block was detected, split=layers actually split).
        self._moe_dbg = {"on": bool(moe_off),
                         "blocks": sum(1 for b in moe_blocks if b[1] is not None),
                         "split": int(sum(self.layer_split)), "quant": self.quant}
        self._finalize_placement()

    def _materialize_cpu_layers(self) -> None:
        """Replace CPU-resident weight tensors (still mmap-backed file views) with heap
        clones so the weight temp file's mmap is dropped — letting unload delete the file
        and reclaim the RAM (the bf16-on-Windows non-release fix). Cloning all CPU params
        transiently needs ~2x the CPU portion (mmap + clones) before the mmap frees, so we
        only do it when free RAM comfortably covers that; otherwise we leave the mmap (the
        old behavior) rather than risk a load-time OOM."""
        torch = self.torch
        mods = []
        if self.has_embed and self.embed_device.type == "cpu":
            mods.append(self.embed)
        mods += [l for l, d in zip(self.owned_layers, self.layer_devices) if d.type == "cpu"]
        if self.has_head and self.head_device.type == "cpu":
            mods += [self.norm, self.head]
        cpu_bytes = sum(self._mod_bytes(m) for m in mods if m is not None)
        if not mods or cpu_bytes <= 0:
            self.cpu_materialized = True   # nothing on CPU -> nothing pinning the mmap
            return
        try:
            free_ram = psutil.virtual_memory().available
        except Exception:
            free_ram = 0
        if free_ram < int(cpu_bytes * 1.2):
            self.cpu_materialized = False   # not enough headroom — keep mmap (logged by caller)
            print(f"[load] CPU weights left mmap-backed (need ~{cpu_bytes*1.2/GB:.1f} GB free "
                  f"to materialize, have {free_ram/GB:.1f} GB) — RAM frees on next worker restart")
            return
        for m in mods:
            if m is None:
                continue
            for p in m.parameters(recurse=True):   # bf16 weights are Parameters; mmap-backed
                if p.device.type == "cpu":
                    p.data = p.data.clone()         # heap copy -> drops this param's mmap view
        self.cpu_materialized = True

    @classmethod
    def from_blob(cls, config_dict: dict, blob: bytes, layer_start: int, layer_end: int,
                  has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                  device: str = "cpu", gpu_mem_gb: float = 0.0,
                  attn: str = "eager", quant: str = "none") -> "Shard":
        """Build a shard from a controller-served safetensors blob (no HF download,
        no model on disk — the blob is loaded straight into RAM)."""
        import tempfile
        import torch
        from transformers import AutoConfig
        from safetensors.torch import load as st_load
        d = tempfile.mkdtemp(prefix="im_cfg_")
        try:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            cfg = AutoConfig.from_pretrained(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        dt = getattr(torch, dtype)
        sd = {k: v.to(dt) for k, v in st_load(blob).items()}
        return cls(cfg, sd, layer_start, layer_end, has_embed, has_head, dt,
                   device=device, gpu_mem_gb=gpu_mem_gb, attn=attn, quant=quant)

    @classmethod
    def from_stream(cls, config_dict: dict, fetch, layer_start: int, layer_end: int,
                    has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                    device: str = "cpu", gpu_mem_gb: float = 0.0,
                    attn: str = "eager", quant: str = "none", fetch_experts=None,
                    tp_rank: int = 0, tp_size: int = 1, tp_allreduce=None,
                    plan_ram_bytes: int = 0, tp_weights=None, ctx: int = 0,
                    gpu_budget_gb: float = -1.0, moe_offload: bool = False,
                    cache: str = "", kv_quant: str = "none",
                    kv_offload: bool = False) -> "Shard":
        """Build a shard by STREAMING weights one layer at a time straight into RAM — no temp
        file, no disk. `fetch(start, end, embed, head) -> bytes` returns a safetensors blob for
        that slice. Each layer is fetched, loaded, quantized and FREED before the next, so peak
        RAM ~ the resident (int4) shard + one layer's bf16 — the full bf16 never lands on disk
        OR fully in RAM. Heap tensors (no mmap) -> unload reclaims RAM cleanly.

        TP-v2 (tp_size>1): the model is built on meta, _tp_make_structure_ replaces every linear
        with its REDUCED-DIM module (still meta), then each layer's PER-RANK SLICED weights (served
        by /weights_tp) are streamed straight in — so this rank ever holds only ~1/tp of each layer,
        NOT the v1 load-full-then-shard footprint. The caller must pass a `fetch` that hits
        /weights_tp and a connected tp_allreduce; the row-parallel o_proj/down_proj all-reduce hooks
        are wired here just as in __init__."""
        import gc
        import tempfile
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from safetensors.torch import load as st_load
        self = cls.__new__(cls)
        self.torch = torch
        d = tempfile.mkdtemp(prefix="im_cfg_")          # config (+ any remote modeling .py) dir
        _remote = config_dict.pop("__im_remote_code__", None) if isinstance(config_dict, dict) else None
        _trust = bool(_remote) and bool((config_dict or {}).get("auto_map"))
        try:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            if _remote:                                  # write the model's trust_remote_code .py so
                for _fn, _src in _remote.items():         # AutoConfig/from_config build the REAL arch —
                    with contextlib.suppress(Exception):  # else transformers' native class for this
                        with open(os.path.join(d, _fn), "w", encoding="utf-8") as _rf:   # model_type
                            _rf.write(_src)               # mismatches the checkpoint (all tensors meta)
            cfg = AutoConfig.from_pretrained(d, trust_remote_code=_trust)
            omni_thinker = getattr(cfg, "thinker_config", None)
            if omni_thinker is not None:
                self.cfg = cfg.get_text_config()
            else:
                if getattr(cfg, "text_config", None) is not None:
                    cfg = cfg.get_text_config()
                self.cfg = cfg
            _lt = getattr(self.cfg, "layer_types", None)
            self._hybrid = bool(_lt) and any(t != "full_attention" for t in _lt)
            self._omni = omni_thinker is not None
            # #vl-vision: Qwen2.5-VL's rotary (like Omni's) unconditionally indexes 3D position_ids
            # [3,bs,seq] — even a text-only forward crashes if fed 2D. Flag it so shard_forward builds
            # 3D rot positions when the controller passes no mRoPE (plain text turns).
            self._mrope3d = self._omni or str(getattr(self.cfg, "model_type", "")).lower() in (
                "qwen2_5_vl_text", "qwen2_5_vl")
            self.cfg._attn_implementation = attn
            # gpt-oss attention SINKS (a per-head learned logit concatenated into the softmax, then
            # dropped) are applied ONLY by transformers' eager_attention_forward (s_aux=self.sinks);
            # plain SDPA silently ignores s_aux -> the sink mass is lost and attention is subtly wrong.
            # Force eager for gpt_oss regardless of the requested `attn`. (gpt-oss's sliding-window
            # softmax layers still get the causal mask in shard_forward — they have no `.layer_type`
            # attr, so the hybrid mask-skip doesn't strip it; windowing exactness is a follow-up.)
            if str(getattr(self.cfg, "model_type", "")).lower() == "gpt_oss":
                self.cfg._attn_implementation = attn = "eager"
            # transformers 5.x LlamaRotaryEmbedding reads cfg.rope_parameters["rope_type"] in __init__;
            # a 4.x-era custom config (e.g. MiniMax-M2) leaves rope_parameters=None -> 'NoneType' not
            # subscriptable at from_config. Synthesize it from the legacy rope_theta/rope_scaling so the
            # per-layer rotary builds. Only for remote-code (native configs populate it themselves). (#78)
            if _trust and getattr(self.cfg, "rope_parameters", None) is None:
                _rs = getattr(self.cfg, "rope_scaling", None)
                _rp = dict(_rs) if isinstance(_rs, dict) else {}
                _rp.setdefault("rope_type", _rp.get("type", "default"))
                _rp.setdefault("rope_theta", float(getattr(self.cfg, "rope_theta", 10000.0)))
                with contextlib.suppress(Exception):
                    self.cfg.rope_parameters = _rp
            dt = getattr(torch, dtype)
            self.dtype = dt
            self.layer_start, self.layer_end = layer_start, layer_end
            self.has_embed, self.has_head = has_embed, has_head
            self.tp_rank, self.tp_size, self.tp_allreduce = tp_rank, tp_size, tp_allreduce
            self.quant = quant
            self.kv_quant = kv_quant   # #172 TurboQuant KV preset (none|turbo2|turbo3|turbo4); read in shard_forward
            # #kv-offload: KV cache lives in system RAM (transformers OffloadedCache, per-layer
            # prefetch) instead of VRAM — frees the GPU KV reserve for actual model layers. Read in
            # shard_forward (cache build) + _place_modules/kv_reserve_probe (no GPU KV reserve).
            self.kv_offload = bool(kv_offload)
            # Build the meta skeleton WHILE the config dir (with the remote .py) is alive so
            # from_config can resolve a trust_remote_code class. For a remote-code model keep BUFFERS
            # REAL (accelerate include_buffers=False) — per-layer computed buffers (rotary inv_freq …)
            # then compute correctly instead of landing on 'meta' with no checkpoint to fill them; only
            # PARAMS go to meta (filled by the streamed weights, which carry their own dtype). Native
            # models are UNCHANGED (plain torch.device('meta') + model.to(dt)).
            if omni_thinker is not None:
                from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                    Qwen2_5OmniThinkerTextModel)
                class _OmniTextCausalLM(torch.nn.Module):
                    def __init__(self, m, h):
                        super().__init__(); self.model = m; self.lm_head = h
                with torch.device("meta"):
                    model = _OmniTextCausalLM(
                        Qwen2_5OmniThinkerTextModel(self.cfg),
                        torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False))
                model = model.to(dt)
            elif str(getattr(self.cfg, "model_type", "")).lower() in ("qwen2_5_vl_text", "qwen2_5_vl"):
                # Qwen2.5-VL: AutoModelForCausalLM has no mapping for Qwen2_5_VLTextConfig, so the
                # generic build below raises. Build ONLY the text decoder (the worker never holds the
                # vision tower — the controller runs it and splices image embeds at stage 0), wrapped as
                # .model + .lm_head to match the served 'model.*'/'lm_head' weights. Mirrors the Omni
                # special-case above. (#vl-vision)
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
                class _VLTextCausalLM(torch.nn.Module):
                    def __init__(self, m, h):
                        super().__init__(); self.model = m; self.lm_head = h
                with torch.device("meta"):
                    model = _VLTextCausalLM(
                        Qwen2_5_VLTextModel(self.cfg),
                        torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False))
                model = model.to(dt)
            elif _trust:
                try:
                    from accelerate import init_empty_weights
                    _ctx = init_empty_weights(include_buffers=False)
                except Exception:
                    _ctx = torch.device("meta")
                # FORCE eager attention for remote-code archs (#78): MiniMax-M2 (and similar) declare
                # _supports_sdpa/_flash=False and implement their own eager attention; transformers
                # otherwise auto-selects sdpa and ABORTS ("does not support scaled_dot_product_attention").
                # The worker's default `attn` (set on cfg above) may be sdpa, so override here. Set the
                # config attr AND pass the kwarg (the kwarg is the path transformers actually honors).
                with contextlib.suppress(Exception):
                    self.cfg._attn_implementation = "eager"
                with _ctx:
                    try:
                        model = AutoModelForCausalLM.from_config(
                            self.cfg, trust_remote_code=True, attn_implementation="eager")
                    except TypeError:   # older transformers: not a from_config kwarg -> config attr set above
                        model = AutoModelForCausalLM.from_config(self.cfg, trust_remote_code=True)
                # do NOT model.to(dt): it would cast the real fp32 rotary inv_freq buffers to bf16.
            else:
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(self.cfg)
                model = model.to(dt)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        # Per-expert FETCH streaming for int4 MoE: drop the experts from the layer blob (skip_experts)
        # and stream+quantize them in bounded chunks, so the layer's full bf16 experts never land in
        # RAM at once. Two layouts: FUSED (3D gate_up_proj/down_proj, #62/#75) and NON-fused (an
        # `experts` ModuleList of {w1,w3,w2} Linears, e.g. MiniMax-M2 / Mixtral, #78). Exactly one
        # applies per model; non-fused only when not fused. bf16/int8 and dense models stream the full
        # blob (no expert skipping). For a big non-fused MoE the full-blob path's ~7 GB/layer transient
        # only fits one big node -> it couldn't spread; streaming bounds the transient to ~256 MiB/chunk.
        stream_experts = (quant == "int4" and fetch_experts is not None
                          and _model_has_fused_experts(model))
        stream_experts_nf = (quant == "int4" and fetch_experts is not None
                             and not stream_experts
                             and _model_has_nonfused_experts(model))
        # #shard-cache Inc 2 (serve-from-cache): each cached layer unit already carries its experts
        # PRE-PACKED (Packed4Tensor3D buffers), so there is no per-expert /experts streaming and no
        # fuse/quant — the cached install builds every holder directly. Force the streaming-expert
        # paths off so _quant_after never runs (the install dispatcher below skips it for cache).
        use_cache = (cache == quant and quant in ("int4", "int2"))
        if use_cache:
            stream_experts = stream_experts_nf = False
        # TP-v2: rebuild every layer's linears as REDUCED-DIM modules (still meta) BEFORE streaming
        # any weights, so the per-rank sliced tensors (served by /weights_tp, exact reduced shapes)
        # install via the same load_state_dict(assign=True) path. tp_size==1 -> no-op (full modules).
        if tp_size > 1:
            _tp_make_structure_(model, tp_rank, tp_size, self.cfg, tp_weights)
        self.model = model
        self.owned_layers = [model.model.layers[i] for i in range(layer_start, layer_end)]
        self.embed = model.model.embed_tokens if has_embed else None
        self.norm = model.model.norm if has_head else None
        self.head = model.lm_head if has_head else None

        self.loaded_params = 0
        _seen: set = set()

        from safetensors.torch import load_file as _load_file
        def _install(src) -> None:
            # src is BYTES (the fleet path since m4c25) -> in-RAM st_load builds heap tensors, freed
            # per layer. The str/PATH branch (mmap a file then unlink) is now DEAD for the fleet — kept
            # only for the legacy from_file/from_blob self-test callers; the worker never produces a
            # tmpfs path anymore (no /dev/shm temp files).
            if isinstance(src, str):
                try:
                    sd = {k: (v if v.dtype == dt else v.to(dt)) for k, v in _load_file(src).items()}
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(src)
            else:
                sd = {k: (v if v.dtype == dt else v.to(dt)) for k, v in st_load(src).items()}
            sd = _fuse_moe_experts(sd, model)
            for t in sd.values():            # logical bf16 param count, data_ptr-deduped (matches __init__)
                if t.data_ptr() not in _seen:
                    _seen.add(t.data_ptr()); self.loaded_params += t.numel()
            try:
                model.load_state_dict(sd, strict=False, assign=True)
            except TypeError:
                # a trust_remote_code model may OVERRIDE load_state_dict() with a 4.x signature lacking
                # `assign` (e.g. MiniMax-M2). Use the base nn.Module loader to keep the assign-install
                # (meta param -> our streamed tensor). Its override only did qkv-split + fp8-filter,
                # which M2 doesn't need (separate q/k/v, bf16). #78
                torch.nn.Module.load_state_dict(model, sd, strict=False, assign=True)
            _assign_meta_from_sd(model, sd)   # materialize buffers load_state_dict skipped (non-persistent, e.g. MiniMax e_score_correction_bias)
            del sd

        def _drop_slice_mmap(module) -> None:
            # NO-OP since m4c25. The fleet now streams each slice straight into RAM bytes and st_load
            # builds HEAP tensors (load_state_dict assign=True installs them directly), so there is no
            # tmpfs mmap to release. The old path mmap'd a /dev/shm slice and had to clone every CPU
            # float param to heap to drop the mapping; with pure-bytes that clone is a redundant
            # full-layer copy that DOUBLES per-layer transient for nothing. Kept as a no-op so the
            # call sites (and the legacy from_file/from_blob mmap self-test paths) stay intact.
            return

        def _quant_after(kind: str, li: int) -> None:
            if kind == "layer":
                lyr = model.model.layers[li]
                if quant == "int4":
                    if stream_experts_nf and _layer_has_meta_experts_nonfused(lyr):
                        # NON-fused streamed (#78): fill the layer's meta expert Linears with int4
                        # QuantLinear4 FIRST, so the _quantize_int4_ walk below skips them (no longer
                        # nn.Linear). attn/router/shared Linears are resident -> quantized by that walk.
                        _quantize_experts4_streamed_nonfused(lyr, li, fetch_experts, dt)
                    _quantize_int4_(lyr)                    # 2D linears (attn, router, shared experts)
                    if stream_experts and _layer_has_meta_experts(lyr):
                        _quantize_experts4_streamed(lyr, li, fetch_experts, dt)   # fused experts (#62)
                    elif not stream_experts_nf:
                        _quantize_experts4_(lyr)           # fused experts from the resident blob
                elif quant == "int2":
                    _quantize_int2_(lyr)                    # #int2: dense 2D linears (no expert tier)
                elif quant == "int8":
                    _quantize_int8_(lyr)
                _drop_slice_mmap(lyr)                       # release this layer's tmpfs slice mmap
            elif kind == "head":
                if quant == "int8" and self.head is not None:
                    model.lm_head = _quantize_linear(model.lm_head); self.head = model.lm_head
                if self.head is not None: _drop_slice_mmap(self.head)
                if self.norm is not None: _drop_slice_mmap(self.norm)
            elif kind == "embed" and self.embed is not None:
                _drop_slice_mmap(self.embed)               # embed kept bf16 -> clone to heap, free shm

        def _install_cached(src, kind, li) -> None:
            # SERVE-FROM-CACHE install (#shard-cache Inc 2). The controller streamed this unit's tensors
            # ALREADY int4-packed (bit-identical to load-time quant), so build the resident holders
            # DIRECTLY and skip the ~4x bf16 stream + the per-layer quant/fuse entirely:
            #   * '<lin>.weight.{qweight,scale,zero}' (2D)  -> QuantLinear4 in place of the meta nn.Linear
            #   * '<experts>.{gate_up,down}_proj.{qweight,scale,zero}' (3D) -> Packed4Tensor3D Parameter
            #   * everything else (norms / biases / embed / head) is bf16 passthrough -> load_state_dict.
            # in_features comes from the meta module we replace (the cache stores padded widths only), so
            # NO manifest is needed on the worker. NEVER call _fuse_moe_experts / _quantize_* here. The
            # post-loop meta-guard catches any tensor we failed to materialize.
            sd = (_load_file(src) if isinstance(src, str) else st_load(src))
            # #int2: the packed-tensor SHAPE is quant-ambiguous (both tiers ship qweight/scale/zero),
            # so the holder class + group come from the LOAD's quant — use_cache already guarantees
            # cache dir == quant, and both packers are bit-identical to their load-time twins.
            QL = _quant2_linear_cls() if quant == "int2" else _quant4_linear_cls()
            PT = _packed4_3d_cls()
            G = _INT2_GROUP if quant == "int2" else _INT4_GROUP
            packed: dict = {}      # base -> {'q':qweight, 's':scale, 'z':zero}
            plain: dict = {}       # bf16 passthrough keys
            for k, v in sd.items():
                if k.endswith(".qweight"):
                    packed.setdefault(k[:-8], {})["q"] = v
                elif k.endswith(".scale"):
                    packed.setdefault(k[:-6], {})["s"] = v
                elif k.endswith(".zero"):
                    packed.setdefault(k[:-5], {})["z"] = v
                else:
                    plain[k] = (v if v.dtype == dt else v.to(dt))

            def _nav(path):
                parent = model
                for p in path.split("."):
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                return parent

            eager_cfgs: set = set()
            for base, tr in packed.items():
                qw, sc, ze = tr.get("q"), tr.get("s"), tr.get("z")
                if qw is None or sc is None or ze is None:
                    raise RuntimeError(f"cache: incomplete packed tensor {base!r} (have {sorted(tr)})")
                if qw.dim() == 3:                      # fused 3D MoE experts -> Packed4Tensor3D
                    if quant != "int4":                # int2 cache is dense-only; never guess a holder
                        raise RuntimeError(f"cache: 3D expert tensor {base!r} in a {quant} cache — "
                                           "only int4 has a 3D-expert format; refusing")
                    ppath, _, attr = base.rpartition(".")
                    parent = _nav(ppath)
                    metap = parent._parameters.get(attr)
                    if metap is None:
                        raise RuntimeError(f"cache: no meta param for 3D expert {base!r}")
                    in_f = int(metap.shape[2])
                    delattr(parent, attr)              # drop the meta Parameter, install the int4 holder
                    setattr(parent, attr, PT(qw, sc, ze, in_f, G))
                    self.loaded_params += int(qw.shape[0]) * int(qw.shape[1]) * in_f
                    cfg = getattr(parent, "config", None)     # eager experts forward (per-expert index)
                    if cfg is not None and hasattr(cfg, "_experts_implementation") \
                            and id(cfg) not in eager_cfgs:
                        cfg._experts_implementation = "eager"; eager_cfgs.add(id(cfg))
                else:                                  # dense 2D Linear -> QuantLinear4
                    mod_path = base[:-7] if base.endswith(".weight") else base   # strip '.weight'
                    ppath, _, attr = mod_path.rpartition(".")
                    parent = _nav(ppath)
                    metalin = getattr(parent, attr)
                    # The cache packs every 2D layer '.weight'; the worker's load-time quant only ever
                    # produced QuantLinear4 from nn.Linear. A 2D packed weight that maps to a NON-Linear
                    # here = the compile over-captured (cache would not match a cold load) -> refuse to
                    # guess in_features; fail loud (silent wrong logits is the one thing the cache forbids).
                    if not hasattr(metalin, "in_features"):
                        raise RuntimeError(
                            f"cache: {mod_path!r} is {type(metalin).__name__}, not nn.Linear — "
                            "cache/model layout mismatch; refusing to serve a possibly-divergent cache")
                    in_f = int(metalin.in_features)
                    bp = plain.pop(mod_path + ".bias", None)   # this Linear's bf16 bias, if any
                    bias = (torch.nn.Parameter(bp, requires_grad=False) if bp is not None else None)
                    setattr(parent, attr, QL(qw, sc, ze, bias, in_f, G))
                    self.loaded_params += int(qw.shape[0]) * in_f
            if plain:                                  # bf16 passthrough: norms / embed / head / leftover
                try:
                    model.load_state_dict(plain, strict=False, assign=True)
                except TypeError:
                    torch.nn.Module.load_state_dict(model, plain, strict=False, assign=True)
                _assign_meta_from_sd(model, plain)     # materialize non-persistent buffers it skipped
                for t in plain.values():
                    if t.data_ptr() not in _seen:
                        _seen.add(t.data_ptr()); self.loaded_params += t.numel()
            del sd

        def _do_install(src, kind, li) -> None:
            if use_cache:
                _install_cached(src, kind, li)         # holders built directly from the pre-packed cache
            else:
                _install(src); _quant_after(kind, li)  # bf16 stream -> install -> quant (default path)

        # Jobs in pipeline order. tuple = (kind, layer_idx, start, end, embed, head)
        jobs = []
        if has_embed:
            jobs.append(("embed", -1, layer_start, layer_start, 1, 0))
        for i in range(layer_start, layer_end):
            jobs.append(("layer", i, i, i + 1, 0, 0))
        if has_head:
            jobs.append(("head", -1, layer_start, layer_start, 0, 1))

        # MEMORY BALLOON (#63, request A): RESERVE this shard's PLANNED resident RAM up front, then
        # consume it one chunk per shard as each layer/embed/head installs. Two guarantees the bare
        # streaming path lacks: (1) FAIL-FAST — if the node can't actually commit its planned share
        # the load aborts NOW with a clear error, not at 60% after minutes of streaming; (2) the
        # resident lands INTO the reservation rather than ON TOP of it, so a concurrent allocation
        # can't steal the node's share mid-build and peak stays ~ the plan (never plan + shard).
        # Sized from the controller's plan (`plan_ram_bytes` = this stage's est resident, which is
        # what is RAM-resident during the build for EVERY placement mode — GPU layers only move to
        # VRAM in _place_modules at the very end). Pages are faulted so the reservation is REAL, not
        # just committed address space. Missing plan_ram_bytes (old controller / rolling self-update)
        # -> no balloon (unchanged behavior). Only a genuine MemoryError aborts; any other balloon
        # hiccup degrades silently to plain streaming.
        _balloon: list = []
        def _balloon_release_one() -> None:
            if _balloon:
                _balloon.pop()          # free one chunk -> room for the shard about to install
        if plan_ram_bytes and plan_ram_bytes > 0:
            try:
                import numpy as _np
                _PAGE = 4096
                n_chunks = max(1, len(jobs))
                chunk_bytes = max(1 << 20, int(plan_ram_bytes) // n_chunks)   # >= 1 MB/chunk
                _built = []
                for _ in range(n_chunks):
                    _b = bytearray(chunk_bytes)              # commit charge reserved here
                    try:                                    # fault every page -> physically held
                        _np.frombuffer(_b, dtype=_np.uint8)[::_PAGE] = 1
                    except Exception:
                        pass                                # best-effort fault; commit still holds
                    _built.append(_b)
                _balloon = _built
                print(f"[load] reserved {chunk_bytes * n_chunks / GB:.1f} GB RAM balloon "
                      f"({n_chunks} chunks) for this shard's planned footprint", flush=True)
            except MemoryError:
                _balloon = []
                gc.collect()
                raise RuntimeError(
                    f"cannot reserve this shard's planned {plan_ram_bytes / GB:.1f} GB of resident "
                    f"RAM — node is short on memory; load aborted before streaming (fail-fast)")
            except Exception as _be:
                _balloon = []
                print(f"[load] memory balloon skipped ({_be!r}); streaming without pre-reservation",
                      flush=True)

        # PARALLEL PREFETCH: the build (load_state_dict + quant) MUST stay serial — it mutates the
        # shared model and load_state_dict iterates it, so concurrent builds would corrupt it. But
        # the fetches are independent I/O, so we run up to K concurrently into a bounded window and
        # build them in order as they arrive — overlapping network with build and letting the
        # controller hand out K layers at once instead of one-by-one. K is clamped to free RAM so
        # the in-flight bf16 blobs never overcommit a memory-tight node.
        from concurrent.futures import ThreadPoolExecutor
        def _fetch_job(j):
            if (stream_experts or stream_experts_nf) and j[0] == "layer":   # int4 MoE: layer blob WITHOUT experts (#62/#78)
                return fetch(j[2], j[3], j[4], j[5], skip_experts=True)
            return fetch(j[2], j[3], j[4], j[5])
        ex = ThreadPoolExecutor(max_workers=_STREAM_PREFETCH_MAX)
        try:
            inflight = {0: ex.submit(_fetch_job, jobs[0])}
            src0 = inflight.pop(0).result()
            # slot = one slice's bytes. Each in-flight prefetched slice costs ~slot of RAM (the bytes
            # buffer), plus another ~slot transiently while st_load copies it into heap tensors.
            slot = max(1, len(src0))
            # Bound the prefetch window by FREE RAM only (m4c25: no more /dev/shm tmpfs spool, so no
            # separate, smaller fs cap to honor). Recomputed each layer since the resident shard grows
            # as the load proceeds, so a K sized when the node was empty would over-commit it later.
            def _bound_K() -> int:
                try: a = psutil.virtual_memory().available
                except Exception: a = slot * 2
                return max(1, min(_STREAM_PREFETCH_MAX, int(a * 0.35 / slot)))
            K = _bound_K()
            _balloon_release_one()   # free this shard's chunk before it installs (consume the reservation)
            _do_install(src0, jobs[0][0], jobs[0][1]); del src0; gc.collect()
            nxt = 1
            for _ in range(min(K, len(jobs) - nxt)):          # prime the window
                inflight[nxt] = ex.submit(_fetch_job, jobs[nxt]); nxt += 1
            for idx in range(1, len(jobs)):
                src = inflight.pop(idx).result()              # wait this slice's fetch (in order)
                _balloon_release_one()   # free this shard's chunk before it installs (consume the reservation)
                _do_install(src, jobs[idx][0], jobs[idx][1]); del src; gc.collect()
                # Re-clamp the prefetch window to CURRENT free RAM each layer (#61): the resident
                # int4 shard GROWS as the load proceeds, so a K sized when the node was empty can
                # over-commit it 50+ layers later (the planner reserves ~1 layer's transient, not K
                # in-flight blobs). Recompute K each layer — it shrinks toward 1 as free RAM falls
                # (and grows back if it frees). idx+1 is always already in flight (the while keeps
                # >=1 ahead since K>=1), so the in-order pop stays safe.
                K = _bound_K()   # re-clamp to CURRENT free RAM each layer
                while nxt < len(jobs) and nxt < idx + 1 + K:   # keep <=K fetches in flight ahead
                    inflight[nxt] = ex.submit(_fetch_job, jobs[nxt]); nxt += 1
        finally:
            ex.shutdown(wait=True)
            _balloon.clear()                # drop any unconsumed reservation (success: already empty)
            for _f in inflight.values():    # on error, delete any prefetched tmpfs slices not installed
                with contextlib.suppress(Exception):
                    _r = _f.result(timeout=0)
                    if isinstance(_r, str):
                        os.remove(_r)
        if not _trust:                       # native: rotary built on meta -> rebuild for real inv_freq.
            rot = model.model.rotary_emb
            model.model.rotary_emb = type(rot)(self.cfg)
        else:
            # REMOTE-CODE rotary rebuild (#78, e.g. MiniMax-M2). Two transformers-5.x problems: (a) the
            # per-layer self_attn.rotary_emb modules were built under torch.device('meta') (workers lack
            # accelerate -> the include_buffers=False path is skipped), so their inv_freq buffers are META
            # — they'd trip the final meta guard and be unusable; (b) 5.x compute_default_rope_parameters
            # takes the rotary dim from config.head_dim and IGNORES partial_rotary_factor, so a PARTIAL-
            # rotary model (rotary_dim < head_dim, M2: 64 < 128) gets the WRONG width. Fix BOTH: rebuild
            # every owned layer's rotary AND a model-level model.model.rotary_emb from a corrected rope
            # config (head_dim:=rotary_dim, rope_parameters synthesized) -> REAL, correct-width inv_freq.
            # The shared forward feeds those cos/sin via position_embeddings; the layer partial-slices to
            # rotary_dim (no-op since already that width). _finalize_placement pins the model-level one.
            import copy as _copy
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            _rcfg = _copy.deepcopy(self.cfg)
            _rdim = int(getattr(self.cfg, "rotary_dim", 0) or 0)
            _hd = int(getattr(self.cfg, "head_dim", 0) or 0)
            if _rdim and _hd and _rdim < _hd:
                _rcfg.head_dim = _rdim                      # 5.x rotary width comes from head_dim
            if getattr(_rcfg, "rope_parameters", None) is None:
                _rcfg.rope_parameters = {"rope_type": "default",
                                         "rope_theta": float(getattr(self.cfg, "rope_theta", 10000.0))}
            def _mk_rotary():
                return LlamaRotaryEmbedding(_rcfg)
            for _lyr in self.owned_layers:                 # materialize (real) + correct-width per-layer
                _sa = getattr(_lyr, "self_attn", None)
                if _sa is not None and getattr(_sa, "rotary_emb", None) is not None:
                    with contextlib.suppress(Exception):
                        _sa.rotary_emb = _mk_rotary()
            if getattr(model.model, "rotary_emb", None) is None:   # model-level for the shared forward
                with contextlib.suppress(Exception):
                    model.model.rotary_emb = _mk_rotary()
        # Gemma-family scaled embeddings register a NON-persistent `embed_scale` (= sqrt(hidden_size))
        # buffer computed in __init__; the streamed build loads checkpoint weights but never fills it,
        # so it stays on META and trips the meta-guard below (would crash .to(device)). Recompute any
        # meta 'embed_scale' buffer under the embed module as sqrt(embedding_dim) in the embed dtype
        # (the Gemma normalizer) — mirrors the inv_freq rebuild above.
        if has_embed and self.embed is not None:
            for _em in self.embed.modules():
                _bs = _em._buffers.get("embed_scale", None)
                if _bs is not None and getattr(_bs, "is_meta", False):
                    _w = getattr(_em, "weight", None)
                    _dim = int(_w.shape[-1]) if (_w is not None and _w.dim() >= 1) \
                        else int(getattr(self.cfg, "hidden_size", 0) or 0)
                    _dt = _w.dtype if _w is not None else torch.float32
                    _em._buffers["embed_scale"] = torch.tensor(_dim ** 0.5, dtype=_dt)
        # Gemma-4 MoE router (Gemma4TextRouter) defines `scale` (hidden_size) and `per_expert_scale`
        # (num_experts) as nn.Parameters initialized to ONES — but the checkpoint OMITS them (only
        # router.proj.weight is stored). A streamed/cache build never fills them, so they stay on META
        # and trip the meta-guard below. Materialize any meta router scale param to ones — exactly what
        # HF from_pretrained does for an absent param (its _init_weights uses init.ones_). Gated on the
        # parent module name ending in "Router" so it can't clobber an unrelated `scale` elsewhere.
        for _lyr in self.owned_layers:
            for _mod in _lyr.modules():
                if not type(_mod).__name__.endswith("Router"):
                    continue
                for _pn in ("scale", "per_expert_scale"):
                    _pp = _mod._parameters.get(_pn, None)
                    if _pp is not None and getattr(_pp, "is_meta", False):
                        _dt = next((p.dtype for p in _lyr.parameters()
                                    if not getattr(p, "is_meta", False) and p.is_floating_point()),
                                   torch.bfloat16)
                        _mod._parameters[_pn] = torch.nn.Parameter(
                            torch.ones(tuple(_pp.shape), dtype=_dt), requires_grad=False)
        # ROCm fused-MoE decode fast path: install the fused grouped-expert forward on every fused-3D
        # experts module, regardless of how it was built (cold-resident / streamed / serve-from-cache).
        # Single catch-all so no load path is missed; idempotent + self-gating (no-op on CUDA, non-fused,
        # or CPU-offloaded experts). Device is decided at the first-decode self-check (placed by then).
        try:
            for _l in self.owned_layers:
                for _m in _l.modules():
                    _install_fused_moe_forward(_m)
        except Exception as _e:
            print(f"[int4] fused-MoE install sweep skipped ({_e!r})")
        model.eval()
        # TP-v2: the row-parallel o_proj/down_proj produce PARTIAL outputs — sum them across the TP
        # group via the same forward-hook + _TPAllReduce wiring as the v1 __init__ path. The reduced
        # modules are already filled with this rank's slice (streamed above), so we only add hooks.
        if tp_size > 1:
            ar = tp_allreduce
            for lyr in self.owned_layers:
                lyr.self_attn.o_proj.register_forward_hook(lambda m, i, o: ar.allreduce(o))
                lyr.mlp.down_proj.register_forward_hook(lambda m, i, o: ar.allreduce(o))
        mods = ([self.embed] if has_embed else []) + list(self.owned_layers)
        if has_head:
            mods += [self.norm, self.head]
        self.loaded_bytes = sum(self._mod_bytes(m) for m in mods if m is not None)
        self.kv = None
        self.cpu = torch.device("cpu")
        self._streamed = True             # heap tensors (no mmap) -> _materialize_cpu_layers is a no-op
        self.cpu_materialized = True       # already heap-resident; parity with the from_file path
        # Guard: any owned tensor still on 'meta' here would make _place_modules' .to(device) die with
        # the cryptic 'copy out of meta tensor'. Surface the exact names instead (the int4 experts are
        # already real Packed4Tensor3D by now, so anything meta is a genuine un-served/skipped tensor).
        _stuck = []
        _scan = ([("embed", self.embed)] if has_embed else []) \
            + [("L%d" % (layer_start + _i), _l) for _i, _l in enumerate(self.owned_layers)] \
            + ([("norm", self.norm), ("head", self.head)] if has_head else [])
        for _tag, _m in _scan:
            if _m is None:
                continue
            for _n, _p in list(_m.named_parameters()) + list(_m.named_buffers()):
                if getattr(_p, "is_meta", False):
                    _stuck.append("%s.%s" % (_tag, _n))
        if _stuck:
            raise RuntimeError("unmaterialized meta tensor(s) after streamed build (would crash "
                               ".to(device)): " + ", ".join(_stuck[:12])
                               + (" ...+%d more" % (len(_stuck) - 12) if len(_stuck) > 12 else ""))
        self._moe_offload = bool(moe_offload)   # #moe-offload: split MoE layers (attn->GPU, experts->CPU)
        self._place_modules(device, gpu_mem_gb, ctx, gpu_budget_gb)   # ctx -> reserve full-ctx KV; budget -> #95 coexistence cap
        return self

    @classmethod
    def from_file(cls, config_dict: dict, weights_path: str, layer_start: int, layer_end: int,
                  has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                  device: str = "cpu", gpu_mem_gb: float = 0.0,
                  attn: str = "eager", quant: str = "none",
                  tp_rank: int = 0, tp_size: int = 1, tp_allreduce=None) -> "Shard":
        """Build a shard from a safetensors file via MEMORY-MAP (the fleet path).
        Unlike from_blob (raw bytes + tensors both resident => ~2x RAM), load_file
        mmaps the file, so peak RAM ~ the shard's resident size. This is what lets a
        big model (32B/70B) load on memory-tight nodes without OOM at load time."""
        import tempfile
        import torch
        from transformers import AutoConfig
        from safetensors.torch import load_file
        d = tempfile.mkdtemp(prefix="im_cfg_")
        try:
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            cfg = AutoConfig.from_pretrained(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        dt = getattr(torch, dtype)
        sd = load_file(weights_path)   # mmap-backed tensors (zero-copy)
        sd = {k: (v if v.dtype == dt else v.to(dt)) for k, v in sd.items()}
        return cls(cfg, sd, layer_start, layer_end, has_embed, has_head, dt,
                   device=device, gpu_mem_gb=gpu_mem_gb, attn=attn, quant=quant,
                   tp_rank=tp_rank, tp_size=tp_size, tp_allreduce=tp_allreduce)

    @classmethod
    def from_hf(cls, model_id: str, layer_start: int, layer_end: int,
                has_embed: bool, has_head: bool, dtype: str = "bfloat16",
                device: str = "cpu", gpu_mem_gb: float = 0.0,
                attn: str = "eager", quant: str = "none", kv_quant: str = "none") -> "Shard":
        """Build a shard by reading directly from the HF cache (used by the
        standalone self-test; the fleet path uses from_blob)."""
        import torch
        from transformers import AutoConfig
        from huggingface_hub import snapshot_download
        cfg = AutoConfig.from_pretrained(model_id)
        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        model_dir = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
        wm = _weight_map(model_dir)
        names = _selected_names(wm, layer_start, layer_end, has_embed, has_head, tied)
        tensors = _load_tensors(names, wm)
        sd = _assemble_sd(tensors, layer_start, layer_end, has_embed, has_head, tied)
        dt = getattr(torch, dtype)
        sd = {k: v.to(dt) for k, v in sd.items()}
        return cls(cfg, sd, layer_start, layer_end, has_embed, has_head, dt,
                   device=device, gpu_mem_gb=gpu_mem_gb, attn=attn, quant=quant,
                   kv_quant=kv_quant)

    def crop(self, length: int) -> None:
        """Truncate the KV cache to `length` tokens (speculative-decode rollback)."""
        if self.kv is not None:
            with contextlib.suppress(Exception):
                self.kv.crop(length)

    def _splice_mm(self, h, inject):
        """#22 increment 3 (embed-injection): replace the token embeddings at multimodal
        placeholder positions with the controller's precomputed image/audio embeds. Only
        stage 0 (has_embed) ever does this. h is [1, q, hidden]; inject = (positions, embeds)
        with embeds [len(positions), hidden]. Positions outside this frame are skipped."""
        torch = self.torch
        positions, embeds = inject
        idx = torch.as_tensor(list(positions), dtype=torch.long, device=h.device)
        emb = embeds.to(device=h.device, dtype=h.dtype)
        # 1) reconcile counts FIRST, so idx and emb are equal-length before any boolean mask
        #    (a bool mask must match the indexed dim — masking emb with idx's mask before
        #    trimming would IndexError when the counts differ).
        if idx.numel() != emb.shape[0]:                    # count mismatch -> splice the overlap
            n = min(idx.numel(), int(emb.shape[0]))
            idx, emb = idx[:n], emb[:n]
        # 2) drop any positions outside this frame (mask now matches both tensors' length)
        if idx.numel() and int(idx.max()) >= h.shape[1]:
            keep = idx < h.shape[1]
            idx, emb = idx[keep], emb[keep]
        h = h.clone()                                      # embed output may be a view; clone first
        if idx.numel():
            h[0, idx] = emb
        return h

    def kv_reserve_probe(self, ctx: int) -> None:
        """#2 pre-alloc safety: actually allocate the full-ctx KV this shard will grow into (on
        each device its layers sit on), then free it. If it OOMs, raise KV_RESERVE_OOM so the
        LOAD fails fast and clean — instead of the stage dying mid-decode and dropping its data
        connection. Full-attn KV is sized for EVERY layer (an overestimate for the hybrid
        linear-attn / Gated-DeltaNet layers -> conservative). Under kv_quant (#172) it reserves the
        bit-packed resting footprint + one bf16 dequant transient/device. Skipped if dims unknown."""
        torch = self.torch
        cfg = self.cfg
        n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        n_kv = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads or 0)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // n_heads if n_heads else 0))
        if ctx <= 0 or n_kv <= 0 or head_dim <= 0:
            return
        per_layer = 2 * int(ctx) * n_kv * head_dim * 2   # k+v, bf16 = 2 bytes/elem (uniform fallback)
        # #172: under kv_quant each layer RESTS at the bit-packed footprint; the dequant transiently
        # rebuilds ONE layer's full bf16 K/V at a time (sequential pipeline), so the true peak per
        # device is (sum of packed resting) + one bf16 layer. Track the max bf16 layer per device and
        # add it once below. kv_quant='none' -> resting == pl (bf16), transient 0 -> bit-identical.
        _kvq = (getattr(self, "kv_quant", "none") or "none")
        _kvq_on = _kvq != "none"
        # #7: a hybrid arch's linear-attn (Gated-DeltaNet) layers grow no full-ctx KV — only its
        # full-attention layers do. Reserve per_layer on the KV-holding layers only (all of them
        # for a dense model). _kv_layer_mask is conservative (unknown -> True) so we never
        # under-reserve and risk decode OOM.
        kv_mask = self._kv_layer_mask()
        by_dev: dict = {}
        max_bf16: dict = {}
        for layer, d, holds_kv in zip(self.owned_layers, self.layer_devices, kv_mask):
            if not holds_kv:
                continue
            # #gemma4-kv: size each layer's KV from ITS OWN attention geometry. head_dim and
            # num_key_value_groups are set at module construction and are quant-invariant, so this
            # reads the real per-type dims: Gemma-4 full_attention layers use global_head_dim(512)
            # with num_global_key_value_heads(1) while sliding layers use head_dim(256)/8 — the
            # uniform `per_layer` above OVER-reserves the full-attn layers ~4x, which can false-OOM a
            # tight stage and trigger a needless replan (the distributed-load churn). Falls back to
            # `per_layer` for any module missing the dims -> bit-identical for uniform-geometry models.
            pl = per_layer
            nkv_l, hd_use = n_kv, head_dim
            sa = getattr(layer, "self_attn", None)
            hd_l = int(getattr(sa, "head_dim", 0) or 0) if sa is not None else 0
            grp_l = int(getattr(sa, "num_key_value_groups", 0) or 0) if sa is not None else 0
            if hd_l > 0 and grp_l > 0 and n_heads > 0:
                nkv_l, hd_use = max(1, n_heads // grp_l), hd_l
                pl = 2 * int(ctx) * nkv_l * hd_use * 2
            resting = pl
            if _kvq_on:   # packed resting for this layer's geometry; any failure -> bf16 pl (conservative)
                try:
                    import kv_quant
                    _pt = kv_quant.kv_quant_bytes_per_token_per_layer(_kvq, nkv_l, hd_use)
                    if _pt > 0:
                        resting = int(ctx) * _pt
                except Exception:
                    resting = pl
            by_dev[d] = by_dev.get(d, 0) + resting
            max_bf16[d] = max(max_bf16.get(d, 0), pl)
        if _kvq_on:   # + one bf16 dequant transient per device that holds any KV layer
            for d in list(by_dev):
                by_dev[d] += max_bf16.get(d, 0)
        # #kv-offload: the KV lives in system RAM regardless of where the layers sit, so probe the
        # WHOLE reservation against CPU RAM (allocate+free there) instead of the layer devices —
        # a GPU that can't hold the KV is exactly the case this mode exists for.
        if getattr(self, "kv_offload", False) and by_dev:
            by_dev = {self.cpu: sum(by_dev.values())}
        held = []
        try:
            for dev, nbytes in by_dev.items():
                if nbytes > 0:
                    held.append(torch.empty(int(nbytes), dtype=torch.uint8, device=dev))
        except Exception as exc:
            total = sum(by_dev.values()) / GB
            raise RuntimeError(
                f"KV_RESERVE_OOM: cannot reserve {total:.2f} GB KV for ctx={ctx} on "
                f"{[str(d) for d in by_dev]}: {exc}") from exc
        finally:
            held.clear()
            import gc
            gc.collect()
            with contextlib.suppress(Exception):
                if any(getattr(d, "type", "") == "cuda" for d in by_dev):
                    torch.cuda.empty_cache()
        print(f"[load] KV reserve probe OK: {sum(by_dev.values())/GB:.2f} GB for ctx={ctx} "
              f"across {len(by_dev)} device(s)")
