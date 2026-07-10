"""shard_compile: the pre-quantized shard-cache COMPILE/PACK family, relocated VERBATIM
from shards.py (code-split Inc 9): PACKER_VERSION/_packer_tag, pack_linear_int4/_3d,
pack_linear_int8, pack_unit_tensors (the SINGLE shared packer used by the controller's local
compile AND every worker's distributed pack -- bit-identical caches by construction),
_shard_cache_root, _quant_scope, _sha256_file, compile_shards, verify_shard_cache,
shard_cache_status, cache_unit_path.

Bind-free BY REQUIREMENT: the /compile_shards route runs `python -c "import shard_compile;
shard_compile.compile_shards(...)"` in a fresh subprocess (cache-on-first-load), so everything
here resolves through real imports -- shared read/dequant/skeleton helpers come from shards
(leaf-to-leaf; shards.py must NEVER import shard_compile back), the MoE fuse from wire.
INT4_GROUP stays in shards.py (both fleets' consumers read shards.INT4_GROUP). Shared module:
listed in BOTH server.py's and client.py's EXTRA_UPDATE_FILES (like wire.py/state.py).
"""

from typing import Optional
import json
import os

from shards import (INT4_GROUP, INT2_GROUP, _dequant_fp8_to_bf16, _dequant_nvfp4_to_bf16, _fp8_block_size, _fp8_scale_name, _has_moe_experts, _head_key, _is_fp8_meta_name, _model_num_layers, _nvfp4_global_scale_name, _nvfp4_group_size, _nvfp4_scale_name, _skeleton_from_cfg, _text_prefix, _weight_map, build_skeleton_from_config)   # noqa: F401  (shared helpers STAY in shards)
from wire import (_fuse_moe_experts)   # noqa: F401

# #distributed-packing Inc 4: a packer-identity tag stamped into the manifest so a load REJECTS a
# cache packed by a different packer version (fail-loud on packer/scope drift instead of silently
# serving divergent weights — also de-risks any future GPU-pack path). Bump PACKER_VERSION whenever
# the pack MATH changes (not for comments/refactors). A manifest WITHOUT packer_hash is legacy and is
# grandfathered (accepted) so existing caches keep working.
PACKER_VERSION = 1


def _packer_tag(quant: str, group_size: int) -> str:
    return f"v{PACKER_VERSION}-g{group_size}-{quant}"


def _default_group(quant: str, group_size: int) -> int:
    """Per-tier default group size: callers that pass the int4 default (or nothing) get the right
    group for the tier — int2 packs at INT2_GROUP (64), everything else keeps what was asked."""
    if quant == "int2" and group_size == INT4_GROUP:
        return INT2_GROUP
    return group_size


def pack_linear_int2(W, group_size: int = INT2_GROUP):
    """Group-wise asymmetric int2 pack of one Linear weight — the SHARED packer for the shard cache.
    Returns (qweight uint8 [out, in_pad//4], scale, zero, in_features). MUST stay BIT-IDENTICAL to
    worker_quant.py `_quantize_linear2` (same min/max/scale/zero/round/clamp/crumb-order: LOWEST
    2 bits = lowest input column) so a cached shard loads exactly as a freshly-quantized one."""
    import torch
    import torch.nn.functional as F
    out, in_f = W.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    Wp = F.pad(W, (0, in_pad - in_f)) if in_pad != in_f else W
    Wg = Wp.reshape(out, ng, G).float()
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    scale = ((wmax - wmin) / 3.0).clamp(min=1e-8)                  # [out, ng]
    zero = torch.round(-wmin / scale).clamp(0, 3)                  # [out, ng]
    q = torch.round(Wg / scale.unsqueeze(2) + zero.unsqueeze(2)).clamp(0, 3).to(torch.uint8)
    q = q.reshape(out, in_pad)
    qpacked = (q[:, 0::4] | (q[:, 1::4] << 2)
               | (q[:, 2::4] << 4) | (q[:, 3::4] << 6)).contiguous()   # [out, in_pad//4]
    return qpacked, scale.to(W.dtype), zero.to(W.dtype), in_f


def pack_linear_int4(W, group_size: int = INT4_GROUP):
    """Group-wise asymmetric int4 pack of one Linear weight — the SHARED packer for the shard cache.
    Returns (qweight uint8 [out, in_pad//2], scale, zero, in_features). MUST stay BIT-IDENTICAL to
    worker_quant.py `_quantize_linear4` (same min/max/scale/zero/round/clamp/nibble-order) so a cached
    shard loads exactly as a freshly-quantized one. W is a 2D bf16/fp16 tensor [out, in_features]."""
    import torch
    import torch.nn.functional as F
    out, in_f = W.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    Wp = F.pad(W, (0, in_pad - in_f)) if in_pad != in_f else W
    Wg = Wp.reshape(out, ng, G).float()
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)                 # [out, ng]
    zero = torch.round(-wmin / scale).clamp(0, 15)                 # [out, ng]
    q = torch.round(Wg / scale.unsqueeze(2) + zero.unsqueeze(2)).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out, in_pad)
    qpacked = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()        # [out, in_pad//2] uint8
    return qpacked, scale.to(W.dtype), zero.to(W.dtype), in_f


def pack_linear_int4_3d(W3, group_size: int = INT4_GROUP):
    """Group-wise int4 pack of a FUSED 3D MoE expert tensor [E, out, in] -> per-expert STACKED
    (qweight uint8 [E, out, in_pad//2], scale [E, out, ng], zero [E, out, ng], in_features, ng).
    Loops the SAME pack_linear_int4 per expert (group quant is independent across experts), so the
    result is BIT-IDENTICAL to the worker's _pack4_3d / _pack4_expert (which use the same per-expert
    math as _quantize_linear4 — already proven == pack_linear_int4 in m4c81). W3 is bf16/fp16."""
    import torch
    E = int(W3.shape[0])
    q0, s0, z0, in_f = pack_linear_int4(W3[0], group_size)
    qpacked = torch.empty((E,) + tuple(q0.shape), dtype=q0.dtype)
    scale = torch.empty((E,) + tuple(s0.shape), dtype=s0.dtype)
    zero = torch.empty((E,) + tuple(z0.shape), dtype=z0.dtype)
    qpacked[0], scale[0], zero[0] = q0, s0, z0
    for e in range(1, E):
        qe, se, ze, _ = pack_linear_int4(W3[e], group_size)
        qpacked[e], scale[e], zero[e] = qe, se, ze
    return qpacked, scale, zero, in_f, int(scale.shape[2])


def pack_linear_int8(W):
    """Per-output-channel symmetric int8 pack of one Linear weight — the SHARED int8 packer for the
    shard cache. Returns (qweight int8 [out, in], scale [out, 1]). MUST stay BIT-IDENTICAL to
    worker_quant.py `_quantize_linear` (same |W|.amax/127 scale, round, clamp(-127,127)) so a cached int8
    shard loads exactly as a freshly-quantized one. W is a 2D bf16/fp16 tensor [out, in]."""
    import torch
    scale = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    qW = (W / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return qW, scale.to(W.dtype)


def pack_unit_tensors(raw: dict, lin2d, exp3d, skel, quant: str = "int4",
                      group_size: int = INT4_GROUP):
    """Pack ONE cache unit's raw bf16 tensors (keyed by logical 'model.*' name) into the cache's
    packed safetensors dict + per-tensor manifest fragments. Returns (out_sd, manifest_tensors)
    where each manifest_tensors[name] has NO 'file' key (the caller stamps the unit filename).

    The SINGLE shared packer used by BOTH the controller's local compile (compile_shards._write_unit)
    AND the worker's remote-pack handler (#distributed-packing) — so a remotely-packed unit is
    BIT-IDENTICAL to a locally-packed one BY CONSTRUCTION (same fuse + same pack), exactly like
    `_fuse_moe_experts` is shared. Pure: no I/O, no globals. `skel` (the meta model) drives per-expert
    MoE fusion (None -> no fuse); `lin2d`/`exp3d` are the exact quant scope from `_quant_scope` (None
    -> name-heuristic fallback, fine for dense arches). int4 packs layer Linears + 3D experts; int2
    packs layer Linears only (dense tier — no 2-bit expert packer); int8 packs layer Linears +
    lm_head; everything else (norms/embed/biases/router) passes through bf16."""
    group_size = _default_group(quant, group_size)   # int2 packs at INT2_GROUP unless overridden
    if skel is not None:
        raw = _fuse_moe_experts(raw, skel)
    out_sd: dict = {}
    mtensors: dict = {}
    for out_name, W in raw.items():
        if lin2d is not None:
            # skeleton-exact scope: pack ONLY what the worker quantizes (matches cold load).
            is_expert3d = (quant == "int4" and W.dim() == 3 and out_name in exp3d)
            is_layer_lin = (W.dim() == 2 and out_name in lin2d)
        else:
            # name-heuristic fallback (skeleton build failed): dense arches have no non-Linear 2D
            # layer weights, so this still matches; a MoE router gate would over-capture (the worker
            # m4c32 cache install then fails loud rather than serving a divergent cache).
            is_expert3d = (quant == "int4" and W.dim() == 3 and ".experts." in out_name
                           and (out_name.endswith(".gate_up_proj")
                                or out_name.endswith(".down_proj")))
            is_layer_lin = W.dim() == 2 and out_name.endswith(".weight") and ".layers." in out_name
        is_int8_head = quant == "int8" and out_name == "lm_head.weight" and W.dim() == 2
        if is_expert3d:
            qw, sc, ze, in_f, ng = pack_linear_int4_3d(W, group_size)
            out_sd[out_name + ".qweight"] = qw
            out_sd[out_name + ".scale"] = sc
            out_sd[out_name + ".zero"] = ze
            mtensors[out_name] = {"q": True, "is_3d": True,
                                  "num_experts": int(W.shape[0]), "out": int(W.shape[1]),
                                  "in_features": in_f, "ng": ng, "group_size": group_size,
                                  "shape": [int(x) for x in W.shape]}
        elif is_layer_lin or is_int8_head:
            if quant in ("int4", "int2"):
                _pk = pack_linear_int4 if quant == "int4" else pack_linear_int2
                qw, sc, ze, in_f = _pk(W, group_size)
                out_sd[out_name + ".qweight"] = qw
                out_sd[out_name + ".scale"] = sc
                out_sd[out_name + ".zero"] = ze
                mtensors[out_name] = {"q": True, "in_features": in_f,
                                      "shape": [int(x) for x in W.shape]}
            else:   # int8 — per-channel symmetric, no zero point / group
                qw, sc = pack_linear_int8(W)
                out_sd[out_name + ".qweight"] = qw
                out_sd[out_name + ".scale"] = sc
                mtensors[out_name] = {"q": True, "shape": [int(x) for x in W.shape]}
        else:
            out_sd[out_name] = W.contiguous()
            mtensors[out_name] = {"q": False, "shape": [int(x) for x in W.shape]}
    return out_sd, mtensors


# --- Pre-compiled shard cache (#shard-cache): the controller (beast — fastest CPU/GPU + holds the
# weights) quantizes a model ONCE to _shards/<quant>/ so loads serve the small pre-quantized tensors
# instead of streaming the full bf16 + re-quantizing on every worker. The cache is machine-independent
# (the int4 pack is deterministic). Increment 1 = compile + verify (no load-path change). -----------
def _shard_cache_root(model_dir: str) -> str:
    return os.path.join(model_dir, "_shards")



def _quant_scope(model_dir: str):
    """(linear2d_names, expert3d_names, meta_model): the EXACT weights the WORKER int4/int8-quantizes —
    nn.Linear weights INSIDE decoder layers + fused 3D expert params (gate_up_proj/down_proj) — plus
    the meta skeleton itself, discovered by building the SAME model the worker builds (text sub-config,
    trust_remote_code). Lets compile pack PRECISELY what a cold load quantizes, so the cache is
    bit-identical even when a 2D layer '.weight' belongs to a NON-Linear module — e.g. a custom MoE
    router gate (Qwen3.6 `Qwen3_5MoeTopKRouter`, Mixtral `MixtralTopKRouter`) that a name heuristic
    ('2D .weight in .layers.') would WRONGLY pack while a cold load leaves it bf16. The meta_model is
    ALSO what `_fuse_moe_experts` needs to fuse a per-expert checkpoint (Mixtral/OLMoE) into the
    model's expected FUSED 3D layout at compile — identically to the worker's cold load (and to the
    worker's distributed pack, which builds the same skeleton via `build_skeleton_from_config`). Names
    are in the worker's 'model.*' text namespace (== compile `out_name`). Returns None on any build
    failure -> caller falls back to the name heuristic (dense arches are unaffected; per-expert MoE
    rejects). Runs in the compile subprocess on beast (has transformers/accelerate + the weights)."""
    try:
        from torch import nn
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = _skeleton_from_cfg(cfg)
        lin2d, exp3d = set(), set()
        _by_name = dict(model.named_modules())

        def _under_router(nm: str) -> bool:
            # True if any ANCESTOR module of `nm` is a router/gate. NEVER quantize a router/gate
            # projection: int4 on the gate weights corrupts the top-k expert selection -> garbage.
            # gemma4's Gemma4TextRouter exposes `proj` as a plain nn.Linear (so the isinstance walk
            # would catch it); custom routers (Mixtral/Qwen3.6) hold a raw weight Parameter with no
            # inner Linear, so they were already skipped. Matches the worker's _quantize_int4_ which
            # skips recursing into *Router/*Gate modules (keeps the cache == cold-load by construction).
            parts = nm.split(".")
            for _i in range(1, len(parts)):
                anc = _by_name.get(".".join(parts[:_i]))
                if anc is not None and type(anc).__name__.endswith(("Router", "Gate")):
                    return True
            return False
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and ".layers." in name and not _under_router(name):
                lin2d.add(name + ".weight")
        for name, p in model.named_parameters():
            if (p.dim() == 3 and ".experts." in name
                    and (name.endswith(".gate_up_proj") or name.endswith(".down_proj"))):
                exp3d.add(name)
        return (lin2d, exp3d, model) if (lin2d or exp3d) else None
    except Exception:
        return None



def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_shards(model_dir: str, quant: str = "int4", group_size: int = INT4_GROUP,
                   progress=None) -> dict:
    """Quantize a model to a pre-compiled shard cache at _shards/<quant>/ (per-layer safetensors of
    the quantized Linears + bf16 passthrough for embed/norms, plus manifest.json with a sha256 per
    file). Bit-identical to the worker's load-time quant (int4: pack_linear_int4 — group-wise
    asymmetric, head left bf16; int2: pack_linear_int2 — same shape at 2 bits / group 64, head left
    bf16, dense models only; int8: pack_linear_int8 — per-channel symmetric, head quantized too,
    matching the worker). Streams one layer at a time (bounded memory). Dense bf16/fp16, fp8 AND
    nvfp4 sources are supported — a quantized checkpoint (fp8 F8_E4M3 + weight_scale_inv, e.g.
    Mistral3 Devstral/Ministral; or compressed-tensors nvfp4 weight_packed + weight_scale +
    weight_global_scale, e.g. Qwen3.6-27B-NVFP4) is DEQUANTIZED to bf16 first with the SAME helper
    the /weights serve path uses (`_dequant_fp8_to_bf16` / `_dequant_nvfp4_to_bf16`) so the cache
    equals what the worker quantizes from the served bf16. MoE source still raises a clear error
    (per-expert serve-dequant is a follow-up). progress(done, total) is called per unit."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if quant not in ("int4", "int8", "int2"):
        raise ValueError(f"shard cache supports quant int4|int8|int2 (got {quant!r})")
    group_size = _default_group(quant, group_size)   # int2 caches pack at INT2_GROUP (64)
    wm = _weight_map(model_dir)
    fp8_block = _fp8_block_size(model_dir)   # None unless this is an fp8 checkpoint (then dequant->bf16)
    nvfp4_group = _nvfp4_group_size(model_dir)   # None unless compressed-tensors nvfp4 (then dequant->bf16)
    # EXACT quant scope + meta skeleton from the worker's own model build (so the cache packs precisely
    # the nn.Linear set + fused-3D experts a cold load quantizes — not a name heuristic that over-captures
    # a non-Linear MoE router gate; the skeleton also drives the per-expert->3D fusion below). None -> the
    # build failed; dense arches fall back to the name heuristic, per-expert MoE rejects (can't fuse).
    _scope = _quant_scope(model_dir)
    _lin2d, _exp3d, _skel = _scope if _scope else (None, None, None)
    # MoE compile (shard-cache Inc 2/#119): int4 MoE is supported in THREE layouts, all bit-identical
    # to the cold load by construction —
    #   * FUSED checkpoint (Qwen3.6): 3D gate_up_proj/down_proj already in the weights.
    #   * PER-EXPERT checkpoint the BUILD FUSES to 3D (Mixtral/OLMoE): `_fuse_moe_experts` stacks the
    #     experts into the skeleton's 3D params (== the worker cold load), so `_exp3d` is non-empty.
    #   * PER-EXPERT checkpoint the build KEEPS per-expert (MiniMax-M2): the model holds an `experts`
    #     ModuleList of 2D Linears, so the skeleton-exact `_lin2d` already lists each expert Linear and
    #     `pack_unit_tensors` int4-packs them as 2D — exactly what _quantize_experts4_streamed_nonfused
    #     produces at load. `_fuse_moe_experts` is a no-op here (no 3D targets in the skeleton).
    # Still rejected with a CLEAR message (never a silent wrong cache): int8 MoE (no worker int8
    # 3D-expert quant), fp8/nvfp4-source MoE (no 3D serve-dequant), and ANY per-expert MoE whose
    # skeleton failed to build (then we have neither a fused-3D layout to fuse into nor a per-expert
    # Linear scope to pack — can't guarantee the cache matches a cold load).
    _moe = _has_moe_experts(wm)
    _moe_fused = any(s.endswith(".gate_up_proj") or s.endswith(".down_proj") for s in wm)
    _skel_expert_lins = (_lin2d is not None and any(".experts." in n for n in _lin2d))
    _moe_per_expert = _moe and not _moe_fused and not _exp3d and _skel_expert_lins
    if _moe:
        if quant != "int4":
            raise ValueError("MoE shard-cache compile supports int4 only "
                             "(no worker int8/int2 3D-expert quantizer)")
        # fp8/nvfp4-source MoE (#nvfp4-moe): compressed-tensors quantizes nn.Linear modules, so a
        # quantized MoE stores experts PER-EXPERT (`...experts.<N>.<proj>.weight_packed`/`.weight`),
        # each a 2D tensor `_get_bf16` already dequantizes (the SAME path dense nvfp4/fp8 uses); the
        # fuse-to-3D or per-expert pack then runs on bf16 as usual. So per-expert quantized MoE IS
        # supported. Only a FUSED-3D quantized expert tensor (one 3D weight_packed per layer) would
        # need a 3D serve-dequant we don't have — reject just that (compressed-tensors doesn't emit it).
        if fp8_block is not None or nvfp4_group is not None:
            _q_perexpert = any(".experts." in s and s.split(".experts.", 1)[1][:1].isdigit()
                               for s in wm)
            if not _q_perexpert:
                raise ValueError("fused-3D fp8/nvfp4-source MoE shard compile not supported "
                                 "(no 3D serve-dequant path); per-expert quantized MoE is supported")
        if not _moe_fused and not _exp3d and not _moe_per_expert:
            raise ValueError("per-expert MoE shard compile needs the model skeleton (it failed to "
                             "build — no fused-3D layout to fuse into nor per-expert scope to pack); "
                             "cannot produce a correct cache")
    n_layers = _model_num_layers(model_dir)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        tied = bool(json.load(fh).get("tie_word_embeddings", False))
    prefix = _text_prefix(wm)
    out_dir = os.path.join(_shard_cache_root(model_dir), quant)
    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"format": 1, "quant": quant, "group_size": group_size,
                      "num_layers": n_layers, "tied": tied, "files": {}, "tensors": {},
                      "packer_hash": _packer_tag(quant, group_size),   # Inc 4: fail-loud on packer drift
                      # MoE serve-from-cache: fused-3D experts vs per-expert 2D Linears (M2). Informational
                      # (the worker install reads tensor shapes, not this), but kept accurate. (#119)
                      "expert_layout": (("per_expert" if _moe_per_expert else "fused3d") if _moe else None)}
    _open: dict = {}
    _hdr: dict = {}   # fn -> {tensor_name: data_offset_start} (safetensors header), cached per shard

    def _get(src_name: str):
        fn = wm[src_name]
        if fn not in _open:
            _open[fn] = safe_open(fn, framework="pt")
        return _open[fn].get_tensor(src_name)

    def _disk_key(src_name: str):
        # (#119) (shard_file, byte-offset) so a unit's reads sort into ASCENDING on-disk order. The OS
        # then reads each shard SEQUENTIALLY (readahead kicks in) instead of seeking once per tensor —
        # on a spinning weights drive that turns ~49 MB/s random reads into ~150+ MB/s sequential. The
        # win is large for many-tiny-tensor MoE layers (e.g. MiniMax-M2's 768 per-expert Linears, where
        # read dominates: ~150 s/layer read vs ~7 s pack). Pure read-order change -> output bit-identical.
        fn = wm[src_name]
        h = _hdr.get(fn)
        if h is None:
            h = {}
            try:
                with open(fn, "rb") as f:
                    hlen = int.from_bytes(f.read(8), "little")
                    meta = json.loads(f.read(hlen))
                for k, v in meta.items():
                    if k != "__metadata__" and isinstance(v, dict) and v.get("data_offsets"):
                        h[k] = int(v["data_offsets"][0])
            except Exception:
                h = {}
            _hdr[fn] = h
        return (fn, h.get(src_name, 0))

    def _get_bf16(src_name: str):
        """The source weight as bf16 — DEQUANTIZED for a quantized checkpoint exactly like the
        /weights serve path, so the cache equals what the worker quantizes from the served bf16
        (a raw `.to(bfloat16)` on fp8/packed bytes would be garbage — the #89 served-raw bug):
          - fp8 (F8_E4M3 '...weight' + '...weight_scale_inv') -> `_dequant_fp8_to_bf16`;
          - nvfp4 (U8 '...weight_packed' + fp8 '...weight_scale' + f32 '...weight_global_scale')
            -> `_dequant_nvfp4_to_bf16` (the source name here is the '...weight_packed' tensor).
        Plain bf16/fp16 tensors just cast to bf16."""
        t = _get(src_name)
        if fp8_block is not None and t.dtype == torch.float8_e4m3fn and src_name.endswith(".weight"):
            sname = _fp8_scale_name(src_name)
            if sname in wm:
                return _dequant_fp8_to_bf16(t, _get(sname), fp8_block)
        if (nvfp4_group is not None and t.dtype == torch.uint8
                and src_name.endswith(".weight_packed")):
            sname = _nvfp4_scale_name(src_name)
            gname = _nvfp4_global_scale_name(src_name)
            if sname in wm and gname in wm:
                logical = [int(t.shape[0]), int(t.shape[1]) * 2]   # [out, in_half*2] (mirrors serve)
                return _dequant_nvfp4_to_bf16(t, _get(sname), _get(gname), nvfp4_group, logical)
        return t.to(torch.bfloat16)

    def _out_name(src_name: str) -> str:
        """Output (logical) tensor name in plain 'model.*' space. For nvfp4 the checkpoint stores a
        Linear as '...weight_packed'; the worker's bf16 slot is '...weight', so strip the suffix
        (mirrors the serve path's '...weight_packed' -> '...weight' rename) — this is also what makes
        the renamed name match `is_layer_lin` so it gets quantized like every other layer Linear."""
        o = src_name.replace(prefix, "model.", 1)
        if nvfp4_group is not None and o.endswith(".weight_packed"):
            o = o[: -len(".weight_packed")] + ".weight"
        return o

    def _write_unit(unit: str, pairs: list[tuple[str, str]]) -> None:
        # Read this unit's source tensors to bf16 (keyed by logical 'model.*' name), then pack via the
        # SHARED `pack_unit_tensors` (fuse per-expert MoE + int4/int8 the layer Linears + 3D experts).
        # Same function the worker remote-pack handler calls -> a remote unit is bit-identical to this.
        raw: dict = {}
        for out_name, src_name in sorted(pairs, key=lambda p: _disk_key(p[1])):   # (#119) sequential reads
            raw[out_name] = _get_bf16(src_name)
        out_sd, mtensors = pack_unit_tensors(raw, _lin2d, _exp3d, _skel, quant, group_size)
        for out_name, meta in mtensors.items():
            manifest["tensors"][out_name] = {"file": unit, **meta}
        path = os.path.join(out_dir, unit)
        save_file(out_sd, path)
        manifest["files"][unit] = {"sha256": _sha256_file(path), "bytes": os.path.getsize(path)}

    done = 0
    total = n_layers + 2   # embed + layers + head
    _write_unit("embed.safetensors",
                [("model.embed_tokens.weight", f"{prefix}embed_tokens.weight")])
    done += 1
    if progress:
        progress(done, total)
    for i in range(n_layers):
        pairs = [(_out_name(n), n) for n in wm
                 if n.startswith(f"{prefix}layers.{i}.") and not _is_fp8_meta_name(n)]
        _write_unit(f"L{i:04d}.safetensors", pairs)
        done += 1
        if progress:
            progress(done, total)
    head_src = f"{prefix}embed_tokens.weight" if tied else _head_key(wm, prefix)
    _write_unit("head.safetensors", [("model.norm.weight", f"{prefix}norm.weight"),
                                     ("lm_head.weight", head_src)])
    done += 1
    if progress:
        progress(done, total)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


def verify_shard_cache(model_dir: str, quant: str) -> tuple[bool, list[str]]:
    """Integrity-check a compiled cache: manifest present + parseable, every file present with the
    recorded size AND sha256, and the expected file count (embed + N layers + head). Returns
    (ok, problems[]). Cheap size check first; sha256 only if sizes match."""
    d = os.path.join(_shard_cache_root(model_dir), quant)
    mf = os.path.join(d, "manifest.json")
    if not os.path.isfile(mf):
        return False, ["manifest.json missing"]
    try:
        with open(mf, encoding="utf-8") as fh:
            man = json.load(fh)
    except Exception as exc:
        return False, [f"manifest.json unreadable: {exc}"]
    problems: list[str] = []
    files = man.get("files", {})
    for fname, meta in files.items():
        p = os.path.join(d, fname)
        if not os.path.isfile(p):
            problems.append(f"missing file {fname}")
            continue
        if os.path.getsize(p) != meta.get("bytes"):
            problems.append(f"size mismatch {fname}")
            continue
        if _sha256_file(p) != meta.get("sha256"):
            problems.append(f"sha256 mismatch {fname} (corrupt)")
    exp = int(man.get("num_layers", 0)) + 2
    if len(files) != exp:
        problems.append(f"incomplete: {len(files)} files, expected {exp}")
    # Inc 4: reject a cache whose packer_hash is PRESENT but doesn't match this packer version (drift).
    # A MISSING packer_hash is legacy -> grandfathered (accepted) so pre-Inc-4 caches keep working.
    ph = man.get("packer_hash")
    if ph is not None:
        want = _packer_tag(man.get("quant", quant), int(man.get("group_size", INT4_GROUP)))
        if ph != want:
            problems.append(f"packer_hash mismatch ({ph} != {want}) — recompile")
    return (len(problems) == 0), problems


def shard_cache_status(model_dir: str) -> dict:
    """Per-quant summary of a model's compiled shard cache (for the dashboard): which quants are
    compiled, their size + file count, and a lightweight ok flag (manifest present + file count;
    full sha verify is verify_shard_cache, run before a load)."""
    root = _shard_cache_root(model_dir)
    out: dict = {}
    if not os.path.isdir(root):
        return out
    for quant in sorted(os.listdir(root)):
        d = os.path.join(root, quant)
        mf = os.path.join(d, "manifest.json")
        if not os.path.isfile(mf):
            out[quant] = {"ok": False, "reason": "no manifest"}
            continue
        try:
            with open(mf, encoding="utf-8") as fh:
                man = json.load(fh)
        except Exception:
            out[quant] = {"ok": False, "reason": "bad manifest"}
            continue
        files = man.get("files", {})
        total = sum(int(v.get("bytes", 0)) for v in files.values())
        exp = int(man.get("num_layers", 0)) + 2
        out[quant] = {"ok": len(files) == exp, "size_gb": round(total / (1024 ** 3), 2),
                      "files": len(files), "expected_files": exp,
                      "group_size": man.get("group_size"), "num_layers": man.get("num_layers")}
    return out


def cache_unit_path(model_dir: str, quant: str, start: int, end: int,
                    embed: bool, head: bool) -> Optional[str]:
    """Path to the pre-compiled shard-cache safetensors unit for ONE fetched stage slice
    (#shard-cache Inc 2 serve-from-cache), or None if absent. The worker's per-layer streaming path
    fetches exactly one unit per /weights call: the embed unit, the head unit, or a SINGLE decoder
    layer (start=i, end=i+1). The cache stores one file per unit (embed.safetensors / L{i}.safetensors
    / head.safetensors), each already EXACTLY that stage's tensors (dense Linears int4-packed, fused
    3D experts int4-packed, norms/biases bf16) — so a hit is served whole, byte-for-byte, no rebuild.
    A multi-layer range (end-start>1) has no single cache file -> None -> caller falls back to bf16."""
    d = os.path.join(_shard_cache_root(model_dir), quant)
    if embed:
        name = "embed.safetensors"
    elif head:
        name = "head.safetensors"
    elif end - start == 1:
        name = f"L{start:04d}.safetensors"
    else:
        return None
    p = os.path.join(d, name)
    return p if os.path.isfile(p) else None
