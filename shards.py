"""Controller-side weight SERVING for InfiniteModel (split out of server.py, #38).

Builds/streams the per-stage safetensors blobs the workers fetch over /weights, /experts, and
/weights_tp: select a stage's tensors from the checkpoint, rename the (possibly multimodal-nested)
keys to the plain `model.*` a text CausalLM expects, and either materialize a blob or — the bounded-
memory path — plan a list of raw byte-ranges to copy straight from the source files (no build-into-
RAM, no temp file). Also the MoE per-/fused-expert streaming planners and the tensor-parallel
per-rank slicer (column/row split, heterogeneous via wire._tp_hetsplit).

Pure + controller-only: only stdlib at module load, with safetensors/torch imported lazily inside
the functions (so importing this never pulls heavy deps), and wire._tp_hetsplit for the het-TP geo.
No controller globals (registry/engine/sockets) — every function is a function of (model_dir, …).
Listed in server.py's EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync; server.py
imports it with a fetch-to-disk convergence bridge (see the import site there).
"""

import json
import os
import tempfile
from typing import Optional

from wire import _tp_hetsplit, _fuse_moe_experts   # noqa: F401  (het-TP geometry + shared MoE fuse:
# _fuse_moe_experts is the SAME function the worker cold load uses, so a per-expert MoE cache is
# fused into the model's 3D layout IDENTICALLY -> bit-identical to a cold load by construction.)

# Shared int4 group size — MUST match client.py _INT4_GROUP so a controller-compiled shard cache is
# byte-identical to what the worker quantizes at load time (#shard-cache).
INT4_GROUP = 128

# #distributed-packing Inc 4: a packer-identity tag stamped into the manifest so a load REJECTS a
# cache packed by a different packer version (fail-loud on packer/scope drift instead of silently
# serving divergent weights — also de-risks any future GPU-pack path). Bump PACKER_VERSION whenever
# the pack MATH changes (not for comments/refactors). A manifest WITHOUT packer_hash is legacy and is
# grandfathered (accepted) so existing caches keep working.
PACKER_VERSION = 1


def _packer_tag(quant: str, group_size: int) -> str:
    return f"v{PACKER_VERSION}-g{group_size}-{quant}"


def pack_linear_int4(W, group_size: int = INT4_GROUP):
    """Group-wise asymmetric int4 pack of one Linear weight — the SHARED packer for the shard cache.
    Returns (qweight uint8 [out, in_pad//2], scale, zero, in_features). MUST stay BIT-IDENTICAL to
    client.py `_quantize_linear4` (same min/max/scale/zero/round/clamp/nibble-order) so a cached
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
    client.py `_quantize_linear` (same |W|.amax/127 scale, round, clamp(-127,127)) so a cached int8
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
    -> name-heuristic fallback, fine for dense arches). int4 packs layer Linears + 3D experts; int8
    packs layer Linears + lm_head; everything else (norms/embed/biases/router) passes through bf16."""
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
            if quant == "int4":
                qw, sc, ze, in_f = pack_linear_int4(W, group_size)
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


def _model_num_layers(model_dir: str) -> int:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    th = cfg.get("thinker_config") or {}
    for sub in (th.get("text_config"), th, cfg.get("text_config"), cfg):
        if isinstance(sub, dict) and sub.get("num_hidden_layers"):
            return int(sub["num_hidden_layers"])
    raise ValueError("num_hidden_layers not found in config.json")


def _has_moe_experts(wm: dict) -> bool:
    for s in wm:
        if ".experts." in s and (s.endswith(".gate_up_proj") or s.endswith(".down_proj")
                                 or s.split(".experts.", 1)[1][:1].isdigit()):
            return True
    return False


def _skeleton_from_cfg(cfg):
    """Build the META model skeleton from an AutoConfig — the SHARED build used by both `_quant_scope`
    (controller, config from the model dir) and `build_skeleton_from_config` (worker, config from
    /modelmeta), so both fuse a per-expert MoE into the IDENTICAL fused-3D layout (gate_up_proj /
    down_proj names + shapes). Meta only (no real weights): from_config under accelerate's
    init_empty_weights (or torch.device('meta')). Forces eager attention (some remote-code archs abort
    on sdpa at from_config). The param DTYPE is irrelevant to bit-identity — pack_linear_int4 widens
    its bf16 input to fp32 before quantizing, and bf16->fp32 is exact regardless of the skeleton dtype."""
    import torch
    from transformers import AutoModelForCausalLM
    try:
        from accelerate import init_empty_weights
    except Exception:
        init_empty_weights = None
    if (getattr(cfg, "thinker_config", None) is not None
            or getattr(cfg, "text_config", None) is not None):
        cfg = cfg.get_text_config()
    try:
        cfg._attn_implementation = "eager"   # some remote-code archs abort on sdpa at from_config
    except Exception:
        pass
    ctx = init_empty_weights() if init_empty_weights is not None else torch.device("meta")
    with ctx:
        try:
            return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                    attn_implementation="eager")
        except TypeError:
            return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)


def build_skeleton_from_config(config_dict: dict):
    """Worker-side META skeleton for distributed per-expert MoE packing (#distributed-packing Inc 3b):
    build the SAME meta model `_quant_scope` builds on the controller, but from a config DICT (the
    worker has no model dir — it fetches the config via /modelmeta, with any trust_remote_code .py in
    `__im_remote_code__`). Writes config (+ remote .py) to a temp dir, AutoConfig, then the shared
    `_skeleton_from_cfg`. Returns the meta model whose named_parameters drive `_fuse_moe_experts` ->
    the fused 3D layout the controller compile uses, so a remotely fused+packed unit is bit-identical
    to a local compile by construction. Pure-ish: temp dir, always cleaned."""
    import tempfile
    import contextlib
    import shutil
    from transformers import AutoConfig
    cd = dict(config_dict) if isinstance(config_dict, dict) else config_dict
    _remote = cd.pop("__im_remote_code__", None) if isinstance(cd, dict) else None
    _trust = bool(_remote) and bool((cd or {}).get("auto_map"))
    d = tempfile.mkdtemp(prefix="im_pack_cfg_")
    try:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cd, f)
        if _remote:                                   # ship the model's modeling/configuration .py so
            for _fn, _src in _remote.items():          # AutoConfig builds the REAL arch (else native
                with contextlib.suppress(Exception):   # fallback class can mismatch the checkpoint)
                    with open(os.path.join(d, _fn), "w", encoding="utf-8") as rf:
                        rf.write(_src)
        cfg = AutoConfig.from_pretrained(d, trust_remote_code=_trust)
        return _skeleton_from_cfg(cfg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and ".layers." in name:
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
    asymmetric, head left bf16; int8: pack_linear_int8 — per-channel symmetric, head quantized too,
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
    if quant not in ("int4", "int8"):
        raise ValueError(f"shard cache supports quant int4|int8 (got {quant!r})")
    wm = _weight_map(model_dir)
    fp8_block = _fp8_block_size(model_dir)   # None unless this is an fp8 checkpoint (then dequant->bf16)
    nvfp4_group = _nvfp4_group_size(model_dir)   # None unless compressed-tensors nvfp4 (then dequant->bf16)
    # EXACT quant scope + meta skeleton from the worker's own model build (so the cache packs precisely
    # the nn.Linear set + fused-3D experts a cold load quantizes — not a name heuristic that over-captures
    # a non-Linear MoE router gate; the skeleton also drives the per-expert->3D fusion below). None -> the
    # build failed; dense arches fall back to the name heuristic, per-expert MoE rejects (can't fuse).
    _scope = _quant_scope(model_dir)
    _lin2d, _exp3d, _skel = _scope if _scope else (None, None, None)
    # MoE compile (shard-cache Inc 2): FUSED 3D experts at int4 ARE supported — both already-fused
    # checkpoints (Qwen3.6) and per-expert checkpoints (Mixtral/OLMoE), the latter fused on the fly via
    # the SHARED `_fuse_moe_experts` (== the worker cold load) so the cache is bit-identical by
    # construction. Still rejected with a CLEAR message (never a silent wrong cache): int8 MoE (no worker
    # int8 3D-expert quant), fp8/nvfp4-source MoE (no 3D serve-dequant), and per-expert MoE when the
    # fused-3D skeleton couldn't be built (then there is no expected layout to fuse into).
    _moe = _has_moe_experts(wm)
    _moe_fused = any(s.endswith(".gate_up_proj") or s.endswith(".down_proj") for s in wm)
    if _moe:
        if quant != "int4":
            raise ValueError("MoE shard-cache compile supports int4 only "
                             "(no worker int8 3D-expert quantizer)")
        if fp8_block is not None or nvfp4_group is not None:
            raise ValueError("fp8/nvfp4-source MoE shard compile not yet supported "
                             "(no 3D serve-dequant path)")
        if not _moe_fused and (_skel is None or not _exp3d):
            raise ValueError("per-expert MoE shard compile needs a fused-3D model skeleton to fuse "
                             "into, which failed to build — cannot produce a correct cache")
    n_layers = _model_num_layers(model_dir)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        tied = bool(json.load(fh).get("tie_word_embeddings", False))
    prefix = _text_prefix(wm)
    out_dir = os.path.join(_shard_cache_root(model_dir), quant)
    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"format": 1, "quant": quant, "group_size": group_size,
                      "num_layers": n_layers, "tied": tied, "files": {}, "tensors": {},
                      "packer_hash": _packer_tag(quant, group_size),   # Inc 4: fail-loud on packer drift
                      "expert_layout": ("fused3d" if _moe else None)}   # MoE serve-from-cache (Inc 2)
    _open: dict = {}

    def _get(src_name: str):
        fn = wm[src_name]
        if fn not in _open:
            _open[fn] = safe_open(fn, framework="pt")
        return _open[fn].get_tensor(src_name)

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
        for out_name, src_name in pairs:
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


def _weight_map(model_dir: str) -> dict[str, str]:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        return {name: os.path.join(model_dir, fn) for name, fn in wm.items()}
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as fh:
            return {name: single for name in fh.keys()}
    raise FileNotFoundError(f"no safetensors found in {model_dir}")


def _load_tensors(names: list[str], weight_map: dict[str, str]) -> dict:
    from safetensors import safe_open
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    out = {}
    for fn, ns in by_file.items():
        with safe_open(fn, framework="pt") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n)
    return out


def _text_prefix(weight_map) -> str:
    """Submodule prefix the decoder layers live under. Multimodal checkpoints nest the text
    LM: Qwen3.6-35B-A3B at 'model.language_model.*'; Qwen2.5-Omni's Thinker at 'thinker.model.*'.
    Plain LMs use 'model.*'. We serve every stage RENAMED to 'model.*' so a worker loads the
    text CausalLM directly (the worker builds the matching text model from the text sub-config)."""
    for k in weight_map:
        if k.startswith("model.language_model.layers."):
            return "model.language_model."
        if k.startswith("language_model.model.layers."):  # Mistral3 (Devstral/Pixtral): nests the
            return "language_model.model."                 # other way — language_model.model.*
        if k.startswith("thinker.model.layers."):       # Qwen2.5-Omni Thinker
            return "thinker.model."
    return "model."


def _head_key(weight_map, prefix: str) -> str:
    """Where the lm_head weight lives in THIS checkpoint. Omni nests it at 'thinker.lm_head.weight';
    others keep a top-level 'lm_head.weight'. (Tied-embeddings models have no separate head.)"""
    if prefix == "thinker.model." and "thinker.lm_head.weight" in weight_map:
        return "thinker.lm_head.weight"
    if prefix == "language_model.model." and "language_model.lm_head.weight" in weight_map:
        return "language_model.lm_head.weight"   # Mistral3 (Devstral/Pixtral): head under language_model.*
    return "lm_head.weight"


def _selected_names(all_names, start: int, end: int, has_embed: bool,
                    has_head: bool, tied: bool, prefix: str = "model.") -> list[str]:
    want: list[str] = []
    if has_embed:
        want.append(f"{prefix}embed_tokens.weight")
    for i in range(start, end):
        want += [n for n in all_names if n.startswith(f"{prefix}layers.{i}.")]
    if has_head:
        want.append(f"{prefix}norm.weight")
        want.append(f"{prefix}embed_tokens.weight" if tied else "lm_head.weight")
    return list(dict.fromkeys(want))


def _assemble_sd(tensors: dict, start: int, end: int, has_embed: bool,
                 has_head: bool, tied: bool) -> dict:
    sd: dict = {}
    if has_embed:
        sd["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"]
    for i in range(start, end):
        for n in (x for x in tensors if x.startswith(f"model.layers.{i}.")):
            sd[n] = tensors[n]
    if has_head:
        sd["model.norm.weight"] = tensors["model.norm.weight"]
        if tied:
            sd["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        else:
            sd["lm_head.weight"] = tensors["lm_head.weight"]
    return sd


def _build_weight_blob(model_dir: str, start: int, end: int,
                       has_embed: bool, has_head: bool) -> str:
    """Write a safetensors file with exactly one stage's tensors; return its path."""
    from safetensors.torch import save_file
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        tied = bool(json.load(fh).get("tie_word_embeddings", False))
    wm = _weight_map(model_dir)
    names = _selected_names(wm, start, end, has_embed, has_head, tied)
    sd = _assemble_sd(_load_tensors(names, wm), start, end, has_embed, has_head, tied)
    fd, path = tempfile.mkstemp(suffix=".safetensors", prefix="im_blob_")
    os.close(fd)
    save_file(sd, path)
    return path


def _st_header(path: str) -> tuple[dict, int]:
    """Parse a safetensors file's header -> (tensor_info, data_section_offset). Reads only
    the small header, not tensor data."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


# ---- FP8 checkpoints (FineGrained / DeepSeek-style fp8) ------------------------------------
# Some checkpoints (e.g. Mistral3 Devstral/Ministral-3 "2512") ship their linear weights as
# F8_E4M3 with a companion '<name>.weight_scale_inv' (bf16) and an unused '<name>.activation_scale'.
# The quantization_config lives at the TOP-LEVEL config (not text_config), so the worker — which
# builds a plain bf16 text model from text_config — has no fp8 slots. We therefore DEQUANTIZE
# fp8 -> bf16 at SERVE time and never ship the *_scale sidecars; the worker loads plain bf16 and
# its own int8/int4 quant (if requested) still applies post-load. dequant = float(w) * scale_inv
# (DeepSeek/HF convention: weight_scale_inv is the multiply factor). lm_head/embed/norms stay bf16.
def _fp8_block_size(model_dir: str):
    """If this checkpoint is fp8-quantized, return its weight_block_size ([bh,bw] block-wise, or
    [] for per-tensor/static). None if not fp8."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            qc = (json.load(fh) or {}).get("quantization_config") or {}
    except Exception:
        return None
    if str(qc.get("quant_method", "")).lower() != "fp8":
        return None
    bs = qc.get("weight_block_size")
    return list(bs) if bs else []


# ---- NVFP4 checkpoints (compressed-tensors "nvfp4-pack-quantized") ------------------------------
# Unsloth/NVIDIA NVFP4 checkpoints (e.g. Qwen3.6-27B-NVFP4) store each quantized Linear as THREE
# tensors: '<name>.weight_packed' (U8 [out, in//2], two FP4 E2M1 codes/byte), '<name>.weight_scale'
# (F8_E4M3 [out, in//group] per-16-block scale), and '<name>.weight_global_scale' (F32 per-tensor).
# The fleet has NO native FP4 compute (max arch Ada sm_89), so — exactly like the fp8 path — we
# DEQUANTIZE nvfp4 -> bf16 at SERVE time and never ship the *_scale sidecars; the worker loads plain
# bf16 (and its own int4/int8 quant still applies post-load). The packed weight is renamed
# '...weight_packed' -> '...weight' so the worker's plain-bf16 Linear accepts it.
# Dequant (verified from the quant math): global_scale = FP4_MAX*FP8_MAX/amax (the LARGE per-tensor
# factor) and weight_scale_fp8 = block_scale*global_scale, so block_scale = weight_scale_fp8 /
# global_scale, and bf16 = E2M1[code] * block_scale (block scale broadcast over the group of 16).
def _nvfp4_group_size(model_dir: str):
    """Group size (16) if this checkpoint is compressed-tensors NVFP4 pack-quantized, else None."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            qc = (json.load(fh) or {}).get("quantization_config") or {}
    except Exception:
        return None
    fmt = str(qc.get("format", "")).lower()
    qm = str(qc.get("quant_method", "")).lower()
    if "nvfp4" not in fmt and "nvfp4" not in qm:
        return None
    for grp in (qc.get("config_groups") or {}).values():
        w = (grp or {}).get("weights") or {}
        if int(w.get("num_bits", 0)) == 4 and str(w.get("type", "")).lower() == "float":
            return int(w.get("group_size", 16))
    return int(qc.get("group_size", 16))


def _is_fp8_meta_name(name: str) -> bool:
    """fp8 per-weight sidecar tensors (scales) that have no home in a plain bf16 model — they are
    folded into the weight at serve time, so we never serve them. Matches *.weight_scale_inv,
    *.weight_scale, *.activation_scale, *.input_scale, ... (last name component contains 'scale')."""
    return "scale" in name.rsplit(".", 1)[-1]


def _fp8_scale_name(weight_name: str) -> str:
    """Companion scale tensor for an fp8 weight: '...weight' -> '...weight_scale_inv'."""
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".weight_scale_inv"
    return weight_name + ".weight_scale_inv"


_TORCH_ST_DTYPE = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}


def _dequant_fp8_to_bf16(w, scale, block):
    """Dequantize an fp8 weight tensor -> bf16: bf16 = float(w) * scale. scale is a scalar
    (per-tensor/static) or a 2D [ceil(out/bh), ceil(in/bw)] grid (block-wise, expanded to w's shape)."""
    import torch
    wf = w.to(torch.float32)
    s = scale.to(torch.float32)
    if s.numel() == 1:
        deq = wf * s.reshape(())
    else:
        bh, bw = (block or [1, 1])[:2]
        s = s.repeat_interleave(int(bh), dim=0).repeat_interleave(int(bw), dim=1)
        deq = wf * s[: wf.shape[0], : wf.shape[1]]
    return deq.to(torch.bfloat16)


def _fp8_dequant_part_bytes(part: dict) -> bytes:
    """Read a planned fp8 weight + its scale (byte-ranges from _plan_weight_stream) and return the
    dequantized bf16 bytes (little-endian, contiguous) ready to stream on the /weights serve path."""
    import torch
    with open(part["fn"], "rb") as f:
        f.seek(part["w_off"]); w_raw = f.read(part["w_nbytes"])
    with open(part["scale_fn"], "rb") as f:
        f.seek(part["scale_off"]); s_raw = f.read(part["scale_nbytes"])
    w = (torch.frombuffer(bytearray(w_raw), dtype=torch.uint8)
         .view(torch.float8_e4m3fn).reshape(part["shape"]))
    sdt = getattr(torch, _TORCH_ST_DTYPE.get(part.get("scale_dtype", "BF16"), "bfloat16"))
    s = torch.frombuffer(bytearray(s_raw), dtype=sdt)
    if part.get("scale_shape"):
        s = s.reshape(part["scale_shape"])
    deq = _dequant_fp8_to_bf16(w, s, part.get("block"))
    return deq.contiguous().flatten().view(torch.uint8).numpy().tobytes()


# FP4 E2M1 code -> value lookup (sign,2-bit exp,1-bit mantissa incl. subnormals); codes 0..15.
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _nvfp4_scale_name(packed_name: str) -> str:
    return packed_name.replace(".weight_packed", ".weight_scale")


def _nvfp4_global_scale_name(packed_name: str) -> str:
    return packed_name.replace(".weight_packed", ".weight_global_scale")


def _dequant_nvfp4_to_bf16(packed_u8, block_scale_f8, global_scale, group: int, logical_shape):
    """packed_u8: uint8 [out, in//2] (2 FP4 codes/byte, element 2k=LOW nibble, 2k+1=HIGH);
    block_scale_f8: fp8-e4m3 [out, in//group]; global_scale: scalar (f32). Returns bf16 [out, in]."""
    import torch
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
    p_out, p_in_half = packed_u8.shape
    in_full = p_in_half * 2
    b = packed_u8.to(torch.int64)
    lo = lut[b & 0x0F]               # element 2k   -> low nibble
    hi = lut[(b >> 4) & 0x0F]        # element 2k+1 -> high nibble
    codes = torch.stack((lo, hi), dim=2).reshape(p_out, in_full)   # interleave, low first
    bs = block_scale_f8.to(torch.float32) / global_scale.to(torch.float32).reshape(())  # real block scale
    bs = bs.repeat_interleave(int(group), dim=1)                   # [out, in_full] per-element scale
    deq = codes * bs[:, :in_full]
    out, in_ = logical_shape
    return deq[:out, :in_].to(torch.bfloat16)


def _nvfp4_dequant_part_bytes(part: dict) -> bytes:
    """Read a planned nvfp4 weight + its block scale + per-tensor global scale (byte-ranges from
    _plan_weight_stream) and return dequantized bf16 bytes for the /weights serve path."""
    import torch

    def _rd(fn, off, n):
        with open(fn, "rb") as f:
            f.seek(off)
            return f.read(n)

    w = (torch.frombuffer(bytearray(_rd(part["w_fn"], part["w_off"], part["w_nbytes"])),
                          dtype=torch.uint8).reshape(part["packed_shape"]))
    s = (torch.frombuffer(bytearray(_rd(part["s_fn"], part["s_off"], part["s_nbytes"])),
                          dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(part["s_shape"]))
    gdt = getattr(torch, _TORCH_ST_DTYPE.get(part.get("g_dtype", "F32"), "float32"))
    g = torch.frombuffer(bytearray(_rd(part["g_fn"], part["g_off"], part["g_nbytes"])), dtype=gdt)
    deq = _dequant_nvfp4_to_bf16(w, s, g, part["group"], part["shape"])
    return deq.contiguous().flatten().view(torch.uint8).numpy().tobytes()


def _plan_weight_stream(model_dir: str, start: int, end: int,
                        has_embed: bool, has_head: bool,
                        skip_experts: bool = False) -> tuple[bytes, list, int]:
    """Plan a streamed safetensors blob for one stage WITHOUT reading tensor data: build the
    output header and a list of raw byte-ranges to copy from the source files, in output
    order. Mirrors _assemble_sd's tensor selection/naming (incl. the tied lm_head alias).
    Returns (header_bytes, parts, total_size); parts = [(src_path, file_offset, nbytes), ...].
    Bounded memory at serve time -> no build-into-RAM, no temp file, no serialization.
    skip_experts (#62): OMIT the fused 3D MoE expert tensors (*.experts.gate_up_proj/down_proj)
    — those are ~90% of a big MoE's bytes and the worker fetches them per-expert via /experts to
    avoid landing a whole ~7 GB layer blob in RAM. The small tensors (attention, router, norms,
    shared experts) still come through this blob."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        _cfg = json.load(fh)
    tied = bool(_cfg.get("tie_word_embeddings", False))
    fp8_block = _fp8_block_size(model_dir)   # None unless this is an fp8 checkpoint
    nvfp4_group = _nvfp4_group_size(model_dir)   # None unless compressed-tensors nvfp4 (group 16)
    wm = _weight_map(model_dir)
    prefix = _text_prefix(wm)            # 'model.language_model.' for multimodal, else 'model.'
    names = _selected_names(wm, start, end, has_embed, has_head, tied, prefix)
    # (output_name -> source_name): SOURCE uses the checkpoint's prefix; OUTPUT is always the
    # plain 'model.*' the text CausalLM expects, so multimodal nesting is stripped at serve time.
    out_pairs: list[tuple[str, str]] = []
    if has_embed:
        out_pairs.append(("model.embed_tokens.weight", f"{prefix}embed_tokens.weight"))
    for i in range(start, end):
        out_pairs += [(n.replace(prefix, "model.", 1), n)
                      for n in names
                      if n.startswith(f"{prefix}layers.{i}.") and not _is_fp8_meta_name(n)]
    if has_head:
        out_pairs.append(("model.norm.weight", f"{prefix}norm.weight"))
        out_pairs.append(("lm_head.weight",
                          f"{prefix}embed_tokens.weight" if tied else _head_key(wm, prefix)))
    if skip_experts:    # drop MoE experts (fused 3D AND per-expert); worker streams them via /experts (#62)
        def _is_expert_w(s: str) -> bool:
            if ".experts." not in s:                     # leaves shared_experts / router / attn alone
                return False
            if s.endswith(".gate_up_proj") or s.endswith(".down_proj"):
                return True                               # fused 3D expert tensor
            return s.split(".experts.", 1)[1][:1].isdigit()   # per-expert: ...experts.{N}.{proj}.weight
        out_pairs = [(o, s) for (o, s) in out_pairs if not _is_expert_w(s)]
    hdr_cache: dict[str, tuple[dict, int]] = {}
    def _src(name: str):
        fn = wm[name]
        if fn not in hdr_cache:
            hdr_cache[fn] = _st_header(fn)
        return fn, hdr_cache[fn]
    out_hdr: dict = {}
    parts: list[dict] = []   # {"kind":"raw"|"fp8", ...}: raw=byte-range copy, fp8=dequant->bf16
    off = 0
    for out_name, src_name in out_pairs:
        fn, (shdr, dstart) = _src(src_name)
        info = shdr[src_name]
        b, e = info["data_offsets"]
        # fp8 checkpoint: an F8_E4M3 weight is served DEQUANTIZED to bf16 (fold in weight_scale_inv);
        # the worker builds a plain bf16 model and never sees fp8. Output header says BF16.
        if fp8_block is not None and info["dtype"] == "F8_E4M3" and src_name.endswith(".weight"):
            scale_name = _fp8_scale_name(src_name)
            if scale_name in wm:
                sfn, (sshdr, sdstart) = _src(scale_name)
                sinfo = sshdr[scale_name]
                sb, se = sinfo["data_offsets"]
                out_nbytes = 2          # bf16 = 2 bytes/elem
                for dim in info["shape"]:
                    out_nbytes *= int(dim)
                out_hdr[out_name] = {"dtype": "BF16", "shape": info["shape"],
                                     "data_offsets": [off, off + out_nbytes]}
                parts.append({"kind": "fp8", "fn": fn, "w_off": dstart + b, "w_nbytes": e - b,
                              "scale_fn": sfn, "scale_off": sdstart + sb, "scale_nbytes": se - sb,
                              "scale_dtype": sinfo["dtype"], "scale_shape": sinfo["shape"],
                              "shape": info["shape"], "block": fp8_block})
                off += out_nbytes
                continue
        # nvfp4 checkpoint: U8 '*.weight_packed' (+ fp8 block scale + f32 global scale) is served
        # DEQUANTIZED to bf16, renamed '*.weight_packed' -> '*.weight' for the worker's bf16 slot.
        if (nvfp4_group is not None and info["dtype"] == "U8"
                and src_name.endswith(".weight_packed")):
            sname = _nvfp4_scale_name(src_name)
            gname = _nvfp4_global_scale_name(src_name)
            if sname in wm and gname in wm:
                sfn, (sshdr, sds) = _src(sname); sinfo = sshdr[sname]; sb, se = sinfo["data_offsets"]
                gfn, (gshdr, gds) = _src(gname); ginfo = gshdr[gname]; gb2, ge2 = ginfo["data_offsets"]
                out_n2 = out_name.replace(".weight_packed", ".weight")
                p_out, p_in_half = int(info["shape"][0]), int(info["shape"][1])
                logical = [p_out, p_in_half * 2]
                out_nbytes = 2 * logical[0] * logical[1]   # bf16
                out_hdr[out_n2] = {"dtype": "BF16", "shape": logical,
                                   "data_offsets": [off, off + out_nbytes]}
                parts.append({"kind": "nvfp4", "group": int(nvfp4_group), "shape": logical,
                              "packed_shape": [p_out, p_in_half],
                              "w_fn": fn, "w_off": dstart + b, "w_nbytes": e - b,
                              "s_fn": sfn, "s_off": sds + sb, "s_nbytes": se - sb,
                              "s_shape": sinfo["shape"],
                              "g_fn": gfn, "g_off": gds + gb2, "g_nbytes": ge2 - gb2,
                              "g_dtype": ginfo["dtype"]})
                off += out_nbytes
                continue
        nbytes = e - b
        out_hdr[out_name] = {"dtype": info["dtype"], "shape": info["shape"],
                             "data_offsets": [off, off + nbytes]}
        parts.append({"kind": "raw", "fn": fn, "off": dstart + b, "nbytes": nbytes})
        off += nbytes
    hj = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    header_bytes = len(hj).to_bytes(8, "little") + hj
    return header_bytes, parts, len(header_bytes) + off


_ST_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                   "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M3": 1}


def _find_expert_tensor(wm: dict, layer: int, proj: str) -> Optional[str]:
    """Locate a layer's FUSED MoE expert tensor (3D [E, out, in]) in the weight map. proj is
    'gate_up_proj' or 'down_proj'. Handles the common naming (mlp.experts.<proj>) + variants."""
    needle = f"layers.{layer}."
    for n in wm:
        if needle in n and ".experts." in n and n.endswith("." + proj):
            return n
    return None


def _plan_expert_stream(model_dir: str, layer: int, proj: str, e0: int, k: int):
    """Plan a streamed safetensors blob of experts [e0:e0+k] of one MoE layer's FUSED tensor
    (experts.<proj>, shape [E, out, in]). Experts are the outer dim, so the slice is a CONTIGUOUS
    byte range inside the tensor's data — no copy, no whole-tensor read. Returns
    (header_bytes, parts, total) like _plan_weight_stream; the worker loads it as one tensor 'w'
    of shape [k, out, in]. Lets a worker fetch + quantize one expert (chunk) at a time so a big
    MoE layer never lands whole in RAM. Returns (None, None, 0) if the layer has no fused experts."""
    wm = _weight_map(model_dir)
    name = _find_expert_tensor(wm, layer, proj)
    if name is None:
        return None, None, 0
    fn = wm[name]
    shdr, dstart = _st_header(fn)
    info = shdr[name]
    shape = info["shape"]
    if len(shape) != 3:
        return None, None, 0
    E, out, in_f = shape
    dtype = info["dtype"]
    dsz = _ST_DTYPE_BYTES.get(dtype, 2)
    per = out * in_f * dsz                         # bytes for ONE expert
    b, _e = info["data_offsets"]
    e0 = max(0, min(e0, E)); e1 = min(e0 + k, E)
    kk = e1 - e0
    nbytes = kk * per
    file_off = dstart + b + e0 * per               # contiguous: expert e0 starts here
    # Valid single-tensor safetensors: 'w' = experts [e0:e0+k] as [kk, out, in]. The worker
    # already knows the total expert count E from the meta model's gate_up_proj.shape[0].
    out_hdr = {"w": {"dtype": dtype, "shape": [kk, out, in_f], "data_offsets": [0, nbytes]}}
    hj = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    header_bytes = len(hj).to_bytes(8, "little") + hj
    return header_bytes, [(fn, file_off, nbytes)], len(header_bytes) + nbytes


def _plan_experts_chunk_fused(model_dir: str, layer: int, e0: int, k: int):
    """Plan a streamed safetensors blob of experts [e0:e0+k] of one MoE layer's FUSED tensors,
    serving BOTH projections in ONE blob keyed by projection name: 'gate_up_proj' -> [kk, out, in]
    and 'down_proj' -> [kk, out, in] (the worker packs each 3D slice straight into its int4 holder —
    no per-expert gate/up fusion, the checkpoint is already fused). One round-trip per chunk, same
    as the non-fused _plan_experts_chunk path. Each projection's experts are the outer dim so the
    slice is a CONTIGUOUS byte range (no copy, no whole-tensor read). Returns (header_bytes, parts,
    total) like _plan_weight_stream; (None, None, 0) if the layer has no fused experts."""
    out_hdr: dict = {}
    parts: list[tuple[str, int, int]] = []
    off = 0
    wm = _weight_map(model_dir)
    for proj in ("gate_up_proj", "down_proj"):
        name = _find_expert_tensor(wm, layer, proj)
        if name is None:
            continue
        fn = wm[name]
        shdr, dstart = _st_header(fn)
        info = shdr[name]
        shape = info["shape"]
        if len(shape) != 3:
            continue
        E, out, in_f = shape
        dtype = info["dtype"]
        dsz = _ST_DTYPE_BYTES.get(dtype, 2)
        per = out * in_f * dsz                          # bytes for ONE expert of this projection
        b, _e = info["data_offsets"]
        s0 = max(0, min(e0, E)); s1 = min(e0 + k, E)
        kk = s1 - s0
        nbytes = kk * per
        file_off = dstart + b + s0 * per                # contiguous: expert s0 starts here
        out_hdr[proj] = {"dtype": dtype, "shape": [kk, out, in_f],
                         "data_offsets": [off, off + nbytes]}
        parts.append((fn, file_off, nbytes))
        off += nbytes
    # A fused MoE layer has BOTH projections. Serving only one would leave the worker's OTHER int4
    # holder uninitialized (silent corruption), so require both — otherwise report "not fused" so
    # /experts falls back to the per-expert planner (and 404s cleanly if neither layout exists).
    if "gate_up_proj" not in out_hdr or "down_proj" not in out_hdr:
        return None, None, 0
    hj = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    header_bytes = len(hj).to_bytes(8, "little") + hj
    return header_bytes, parts, len(header_bytes) + off


def _plan_experts_chunk(model_dir: str, layer: int, e0: int, k: int):
    """Plan a streamed safetensors blob of the PER-EXPERT source tensors for experts [e0:e0+k] of
    one MoE layer (NON-fused checkpoint, e.g. MiniMax-M2: *.experts.{e}.{proj}.weight, proj in
    w1/w2/w3 or gate_proj/up_proj/down_proj). Output keys are '{e-e0}.{proj}' so the worker groups
    by local index and fuses (#62). Byte-range copy from source — no tensor read, no whole-layer
    blob. Returns (header_bytes, parts, total) like _plan_weight_stream; (None, None, 0) if the
    layer has no per-expert tensors (a fused checkpoint -> caller falls back to _plan_expert_stream)."""
    import re
    wm = _weight_map(model_dir)
    pat = re.compile(r"\.layers\.%d\..*\.experts\.(\d+)\.(\w+)\.weight$" % int(layer))
    found = []
    for n in wm:
        m = pat.search(n)
        if m:
            e = int(m.group(1))
            if e0 <= e < e0 + k:
                found.append((e - e0, m.group(2), n))   # (local_idx, proj, src_name)
    if not found:
        return None, None, 0
    found.sort()
    hdr_cache: dict[str, tuple[dict, int]] = {}
    def _src(name: str):
        fn = wm[name]
        if fn not in hdr_cache:
            hdr_cache[fn] = _st_header(fn)
        return fn, hdr_cache[fn]
    out_hdr: dict = {}
    parts: list[tuple[str, int, int]] = []
    off = 0
    for le, proj, src_name in found:
        fn, (shdr, dstart) = _src(src_name)
        info = shdr[src_name]
        b, e = info["data_offsets"]
        nbytes = e - b
        out_hdr[f"{le}.{proj}"] = {"dtype": info["dtype"], "shape": info["shape"],
                                   "data_offsets": [off, off + nbytes]}
        parts.append((fn, dstart + b, nbytes))
        off += nbytes
    hj = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    header_bytes = len(hj).to_bytes(8, "little") + hj
    return header_bytes, parts, len(header_bytes) + off


def _tp_raw_dims(model_dir: str):
    """Raw (un-divided) decoder dims from config.json, reading the TEXT sub-config for multimodal
    checkpoints (text_config / thinker_config). Returns (nh, nkv, hd, inter)."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    sub = cfg.get("text_config") or cfg
    tc = cfg.get("thinker_config")
    if isinstance(tc, dict):
        sub = tc.get("text_config") or tc
    nh = sub["num_attention_heads"]
    nkv = sub.get("num_key_value_heads", nh)
    hd = sub.get("head_dim") or (sub["hidden_size"] // nh)
    inter = sub["intermediate_size"]
    return nh, nkv, hd, inter


def _tp_dims(model_dir: str, tp_size: int):
    """Uniform per-rank slice widths (qd, kvd, idim) = totals // tp_size — the EVEN TP split."""
    nh, nkv, hd, inter = _tp_raw_dims(model_dir)
    return nh * hd // tp_size, nkv * hd // tp_size, inter // tp_size


def _tp_geo_for_rank(model_dir: str, tp_rank: int, tp_size: int, weights) -> dict:
    """This rank's TP slice geometry {q_off,q_len,kv_off,kv_len,idim_off,idim_len,q_heads,kv_heads}.
    `weights` (per-rank capacity, len==tp_size) -> HETEROGENEOUS split via wire._tp_hetsplit so a
    bigger node holds a bigger slice; empty/mismatched -> the uniform 1/tp split (backward compat).
    The SAME pure function the worker calls to build its reduced-dim structure, so the server's
    slice and the worker's module shape match by construction (128 = int4 group alignment)."""
    nh, nkv, hd, inter = _tp_raw_dims(model_dir)
    if weights and len(weights) == tp_size:
        return _tp_hetsplit(nh, nkv, hd, inter, 128, list(weights))[tp_rank]
    qd, kvd, idim = nh * hd // tp_size, nkv * hd // tp_size, inter // tp_size
    return {"q_off": tp_rank * qd, "q_len": qd, "kv_off": tp_rank * kvd, "kv_len": kvd,
            "idim_off": tp_rank * idim, "idim_len": idim,
            "q_heads": nh // tp_size, "kv_heads": nkv // tp_size}


# col/row classification by output tensor-name suffix (output names are always 'model.*').
# Column-parallel keep this rank's OUTPUT rows (dim 0, bias sliced too); row-parallel keep this
# rank's INPUT cols (dim 1, bias DROPPED). Else (embed/norm/lm_head/layernorm/rotary) REPLICATED.
_TP_COL = (".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
           ".mlp.gate_proj.", ".mlp.up_proj.")
_TP_ROW = (".self_attn.o_proj.", ".mlp.down_proj.")


def _tp_kind_and_slice(out_name: str, geo: dict):
    """(kind, off, length) for a served tensor's parallel-dim slice, from THIS rank's geo. kind in
    {'col','row',None}; None -> replicated (serve whole). q/o -> q slice, k/v -> kv slice,
    gate/up/down -> idim slice — mirrors _tp_shard_model_'s per-projection dim choice, but with
    per-rank offsets (heterogeneous) instead of rank*per (uniform)."""
    if ".self_attn.q_proj." in out_name:
        return "col", geo["q_off"], geo["q_len"]
    if ".self_attn.k_proj." in out_name or ".self_attn.v_proj." in out_name:
        return "col", geo["kv_off"], geo["kv_len"]
    if ".mlp.gate_proj." in out_name or ".mlp.up_proj." in out_name:
        return "col", geo["idim_off"], geo["idim_len"]
    if ".self_attn.o_proj." in out_name:
        return "row", geo["q_off"], geo["q_len"]
    if ".mlp.down_proj." in out_name:
        return "row", geo["idim_off"], geo["idim_len"]
    return None, 0, 0


def _tp_slice_tensor(t, kind: str, off: int, length: int):
    """Slice ONE tensor for a TP rank. col: weight W[off:off+length, :] (+ bias by the caller);
    row: weight W[:, off:off+length] (bias dropped). Materialized (.contiguous()) — the row slice
    is non-contiguous in row-major, so no byte-range copy is possible."""
    if kind == "col":
        return t[off:off + length].contiguous()
    return t[:, off:off + length].contiguous()


def _build_weight_tp_blob(model_dir: str, start: int, end: int, has_embed: bool,
                          has_head: bool, tp_rank: int, tp_size: int, weights=None) -> bytes:
    """TP-v2 per-rank serve: build a safetensors blob of one stage's tensors ALREADY SLICED for
    (tp_rank, tp_size). Column-parallel (q/k/v/gate/up) sliced on dim 0, row-parallel (o/down) on
    dim 1 with bias dropped, everything else (embed/norm/head/layernorm/rotary) served whole. Mirrors
    _plan_weight_stream's tensor selection + output naming (plain 'model.*', tied lm_head alias), but
    reads+materializes (the row slice is non-contiguous so byte-range copy is impossible). Output keys
    and per-tensor SHAPES match exactly what the worker's reduced-dim TP structure expects, so a plain
    load_state_dict(assign=True) installs them. Returns the serialized safetensors bytes."""
    import torch                                  # noqa: F401 (used via save below)
    from safetensors.torch import save as st_save
    geo = _tp_geo_for_rank(model_dir, tp_rank, tp_size, weights)   # per-rank slice (het or uniform)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        tied = bool(json.load(fh).get("tie_word_embeddings", False))
    wm = _weight_map(model_dir)
    prefix = _text_prefix(wm)
    fp8_block = _fp8_block_size(model_dir)   # fold fp8 weight_scale_inv -> bf16 before TP slicing
    if _nvfp4_group_size(model_dir) is not None:
        raise NotImplementedError(
            "nvfp4 checkpoints are not supported in tensor-parallel mode yet — load nvfp4 in "
            "pipeline/proportional mode (the dequant-then-TP-slice path is a follow-up)")
    names = _selected_names(wm, start, end, has_embed, has_head, tied, prefix)
    out_pairs: list[tuple[str, str]] = []         # (output_name 'model.*', source_name)
    if has_embed:
        out_pairs.append(("model.embed_tokens.weight", f"{prefix}embed_tokens.weight"))
    for i in range(start, end):
        out_pairs += [(n.replace(prefix, "model.", 1), n)
                      for n in names
                      if n.startswith(f"{prefix}layers.{i}.") and not _is_fp8_meta_name(n)]
    if has_head:
        out_pairs.append(("model.norm.weight", f"{prefix}norm.weight"))
        out_pairs.append(("lm_head.weight",
                          f"{prefix}embed_tokens.weight" if tied else _head_key(wm, prefix)))
    # group source reads by file (one safe_open per file) to avoid reopening per tensor
    from safetensors import safe_open
    by_file: dict[str, list[tuple[str, str]]] = {}
    for out_name, src_name in out_pairs:
        by_file.setdefault(wm[src_name], []).append((out_name, src_name))
    sd: dict = {}
    for fn, pairs in by_file.items():
        with safe_open(fn, framework="pt") as fh:
            for out_name, src_name in pairs:
                kind, off, length = _tp_kind_and_slice(out_name, geo)
                t = fh.get_tensor(src_name)
                if fp8_block is not None and t.dtype == torch.float8_e4m3fn:   # fp8 -> bf16, then slice
                    sname = _fp8_scale_name(src_name)
                    sc = fh.get_tensor(sname) if sname in fh.keys() else None
                    if sc is None and sname in wm:
                        with safe_open(wm[sname], framework="pt") as sfh:
                            sc = sfh.get_tensor(sname)
                    if sc is not None:
                        t = _dequant_fp8_to_bf16(t, sc, fp8_block)
                if kind is None:                  # replicated: embed/norm/head/layernorm/rotary
                    sd[out_name] = t.clone() if out_name == "lm_head.weight" and tied else t
                    continue
                if out_name.endswith(".bias"):
                    if kind == "col":             # col bias sliced; row bias dropped (added once)
                        sd[out_name] = t[off:off + length].contiguous()
                    continue
                sd[out_name] = _tp_slice_tensor(t, kind, off, length)
    return st_save(sd)
