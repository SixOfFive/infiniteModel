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

# code-split Inc 9: the pack family (PACKER_VERSION/_packer_tag/pack_linear_*/pack_unit_tensors)
# + _shard_cache_root live in shard_compile.py now (VERBATIM). INT4_GROUP above STAYS here --
# consumers on both fleets read shards.INT4_GROUP and shard_compile imports it from here.
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
    if str(getattr(cfg, "model_type", "")).lower() in ("qwen2_5_vl_text", "qwen2_5_vl"):
        # #vl-vision: AutoModelForCausalLM can't build Qwen2_5_VLTextConfig; build the text-decoder
        # skeleton directly so the compile/pack scope matches the worker's cold build exactly.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
        class _VLTextCausalLM(torch.nn.Module):
            def __init__(self, m, h):
                super().__init__(); self.model = m; self.lm_head = h
        _ctx0 = init_empty_weights() if init_empty_weights is not None else torch.device("meta")
        with _ctx0:
            return _VLTextCausalLM(Qwen2_5_VLTextModel(cfg),
                                   torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False))
    try:
        cfg._attn_implementation = "eager"   # some remote-code archs abort on sdpa at from_config
    except Exception:
        pass
    # transformers 5.x LlamaRotaryEmbedding reads cfg.rope_parameters["rope_type"] in __init__; a
    # 4.x-era remote-code config (e.g. MiniMax-M2) leaves it None -> from_config raises 'NoneType' is
    # not subscriptable and the skeleton FAILS to build, which made per-expert MoE compile wrongly
    # report "no fused-3D skeleton". Synthesize it from the legacy rope_theta/rope_scaling — the SAME
    # fix the worker load applies (client.py _Shard build) so the meta skeleton builds identically.
    # Structure-only (the cache packs Linear weights, not rotary buffers) so any valid rope value is
    # fine; gated on `is None` so native configs (which populate it) are untouched. (#119)
    if getattr(cfg, "rope_parameters", None) is None:
        _rs = getattr(cfg, "rope_scaling", None)
        _rp = dict(_rs) if isinstance(_rs, dict) else {}
        _rp.setdefault("rope_type", _rp.get("type", "default"))
        _rp.setdefault("rope_theta", float(getattr(cfg, "rope_theta", 10000.0)))
        try:
            cfg.rope_parameters = _rp
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


# code-split Inc 9: _quant_scope lives in shard_compile.py now (VERBATIM).
def validate_arch_supported(model_dir: str) -> None:
    """Raise a clean ValueError if transformers cannot build this model's architecture (#6/#127):
    an unknown model_type / unresolvable `architectures` with NO trust_remote_code. Run EARLY (at
    load-plan time, before any stage is dispatched) so an exotic/unsupported arch fails LEGIBLY
    instead of as a cryptic meta-tensor crash deep in the streamed worker build. Conservative — it
    mirrors the worker's ACTUAL build path, so anything that passes here loads fine:
      * a trust_remote_code model (config has `auto_map`, e.g. MiniMax-M2) builds its REAL arch from
        the .py the worker fetches via /modelcode -> PASS THROUGH (the .py may not be local yet, so a
        build attempt could false-reject).
      * a native model (no auto_map) must RESOLVE to a transformers-registered config — we check that
        AutoConfig.from_pretrained succeeds and reject ONLY when it genuinely can't resolve the arch.
        We deliberately do NOT attempt a full model BUILD here: some natively-registered archs (e.g.
        Qwen2.5-Omni) are hand-built by the worker via a special path and a generic AutoModel build
        would FALSE-fail them — config resolution is the correct, conservative discriminator for "is
        this an architecture transformers even knows about".
    Reads config.json only (no weights) — controller-safe."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            cfg_d = json.load(fh) or {}
    except Exception:
        return   # no readable config yet (e.g. GGUF mid-normalize) -> let the existing path handle it
    if cfg_d.get("auto_map"):
        return   # trust_remote_code: the worker fetches the .py + builds the real arch (don't reject)
    mt = cfg_d.get("model_type") or "?"
    archs = cfg_d.get("architectures") or []
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_dir, trust_remote_code=True)   # resolves IFF the model_type is
    except Exception as exc:                                            # registered (or remote-coded)
        raise ValueError(
            f"unsupported architecture '{archs[0] if archs else mt}' (model_type={mt!r}) — "
            "transformers cannot resolve it (no registered model_type / no trust_remote_code). "
            f"[{type(exc).__name__}: {exc}]") from exc


# code-split Inc 9: _sha256_file + compile_shards + verify_shard_cache + shard_cache_status +
# cache_unit_path live in shard_compile.py now (VERBATIM; it imports this module's helpers).
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


# ---- MXFP4 checkpoints (gpt-oss: MoE experts in OCP microscaling FP4) ---------------------------
# gpt-oss quantizes ONLY its fused 3D MoE expert weights (mlp.experts.gate_up_proj / down_proj) to
# MXFP4; attention / router / embeddings stay bf16. Each quantized expert weight is TWO tensors:
#   '<name>_blocks': uint8 [E, out, G, 16] — 2 FP4 E2M1 codes/byte (low nibble = EVEN output elem),
#                    16 bytes = 32 codes = ONE microscaling block along the in-dim (so in = G*32).
#   '<name>_scales': uint8 [E, out, G]     — ONE E8M0 scale per 32-code block (biased: real exp =
#                    value-127; the block multiplier is a pure power of two, NOT a float like nvfp4).
# Dequant MIRRORS transformers.integrations.mxfp4._convert_moe_packed_tensors EXACTLY (tf 5.12.1):
#   value = E2M1[code] * 2**(scale_u8 - 127)  via ldexp, reshape -> [E, out, in], then TRANSPOSE(1,2)
#   -> [E, in, out] (gpt-oss applies experts in_features-major: y = x @ W). E2M1 LUT == _E2M1_VALUES.
# Unlike fp8/nvfp4 this is a NATIVELY 3D-FUSED MoE source (the exact case _assemble_sd's serve path
# rejects for fp8/nvfp4) — so wiring it into the int4 cache compile + gpt-oss arch support (attention
# sinks, clamped SwiGLU) is the remaining work. This primitive is the validated foundation
# (unit-tested bit-exact vs transformers; see #161).
_MXFP4_BLOCK = 32   # microscaling block size: FP4 codes per E8M0 scale


def _mxfp4_quantized(model_dir: str) -> bool:
    """True if this checkpoint is MXFP4-quantized (gpt-oss), read from quantization_config."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            qc = (json.load(fh) or {}).get("quantization_config") or {}
    except Exception:
        return False
    return "mxfp4" in str(qc.get("quant_method", "")).lower()


def _mxfp4_blocks_name(weight_name: str) -> str:
    """'<name>'/'<name>.weight' -> '<name>_blocks' (the packed FP4 tensor)."""
    base = weight_name[: -len(".weight")] if weight_name.endswith(".weight") else weight_name
    return base + "_blocks"


def _mxfp4_scales_name(weight_name: str) -> str:
    """'<name>'/'<name>.weight' -> '<name>_scales' (the E8M0 per-block scale tensor)."""
    base = weight_name[: -len(".weight")] if weight_name.endswith(".weight") else weight_name
    return base + "_scales"


def _dequant_mxfp4_to_bf16(blocks_u8, scales_u8):
    """Faithful mirror of transformers _convert_moe_packed_tensors (tf 5.12.1).
    blocks_u8: uint8 [*prefix, G, 16] (2 E2M1 codes/byte, LOW nibble = even output element).
    scales_u8: uint8 [*prefix, G]     (E8M0 biased exponent; real exp = value-127).
    For a gpt-oss 3D expert ([E, out, G, 16]) returns bf16 [E, in, out] (dequant then transpose(1,2));
    in = G*32. value = E2M1[code] * 2**(scale-127)."""
    import torch, math
    blocks = blocks_u8.to(torch.uint8)
    scales = scales_u8.to(torch.int32) - 127                 # E8M0 unbias (128 = 2**7 bias-of-127)
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]} != {scales.shape}"
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.bfloat16, device=blocks.device)
    *prefix, G, B = blocks.shape
    rows = math.prod(prefix) * G
    blk = blocks.reshape(rows, B)
    exp = scales.reshape(rows, 1)
    out = torch.empty(rows, B * 2, dtype=torch.bfloat16, device=blocks.device)
    out[:, 0::2] = lut[(blk & 0x0F).to(torch.int)]           # low nibble  -> even element
    out[:, 1::2] = lut[(blk >> 4).to(torch.int)]             # high nibble -> odd element
    torch.ldexp(out, exp, out=out)                           # * 2**(scale-127), per-block broadcast
    out = out.reshape(*prefix, G, B * 2).view(*prefix, G * B * 2)
    return out.transpose(1, 2).contiguous()                  # [E, out, in] -> [E, in, out]


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
