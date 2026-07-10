"""gptq_pack: GPTQ-class CALIBRATED int2 shard-cache compile (#38 — the int2 usability fix).

Plain round-to-nearest at 2 bits produced token salad on every model tried (7B and 0.5B, any
group size, MSE-clip, mixed-tier salvage — see MODEL_TEST_STATUS). The capacity tier is only
usable with a second-order calibrated packer: this module implements GPTQ (Frantar et al.) —
per-layer Hessians H = E[x xᵀ] estimated from real forward activations on a bundled calibration
corpus, then column-by-column quantization with Cholesky-based error compensation into the
remaining columns, group scale/zero chosen by an MSE shrink search instead of raw min/max.

OUTPUT FORMAT IS UNCHANGED: the same qweight/scale/zero (uint8 crumbs, LOWEST 2 bits = lowest
input column, group 64, asymmetric 0..3 grid) that pack_linear_int2 emits — so the worker's
QuantLinear2, the Triton w2a16 kernels, the cache install path and the size math all serve a
GPTQ cache with ZERO changes. What changes is only WHICH codes/scales land in the file.

Consequences vs the RTN tiers:
  * int2 caches are CACHE-ONLY quality: a cold (no-cache) int2 load still uses the worker's RTN
    _quantize_linear2 and stays token salad — but int2 is compile-first by policy anyway
    (cache-on-first-load builds this cache, then serves it).
  * NOT distributable: layer L's Hessians need layer L-1's QUANTIZED outputs (sequential error
    propagation), so /compile_dist and /pack_probe reject int2 (this compile is local).
  * packer_hash: int2 stamps v2 ("-gptq") via shard_compile._packer_tag — v1 RTN caches fail
    verify with "packer_hash mismatch — recompile", exactly the auto-invalidation wanted.
  * Byte-identity across boxes is NOT guaranteed (GPU vs CPU numerics in the solver); a cache is
    self-consistent via its own manifest sha256s, which is all serve-from-cache checks.

Scope: DENSE decoder-only models of the llama/qwen/mistral class (uniform `model.layers[i]`,
`model.embed_tokens`, model-level rotary). MoE was already rejected for int2 upstream. Anything
this can't drive fails LOUD (never a silent RTN fallback — that cache would be useless).

Calibration: `calib_corpus.txt` bundled in the repo (public-domain novel prose + RFC 9110
technical prose + this repo's own Python) — deterministic, offline, no datasets dependency.
Defaults 32 samples x 512 tokens (env INFINITEMODEL_GPTQ_SAMPLES / _SEQLEN / _PERCDAMP / _GRID).

Controller-only leaf (in server.py EXTRA_UPDATE_FILES together with calib_corpus.txt); imported
lazily by shard_compile.compile_shards' int2 branch, runs in the /compile_shards subprocess.
Uses cuda when available (Hessian + solver + layer forwards), else CPU fp32.
"""
from __future__ import annotations

import json
import os

from shards import (INT2_GROUP, _dequant_fp8_to_bf16, _dequant_nvfp4_to_bf16, _fp8_block_size,
                    _fp8_scale_name, _has_moe_experts, _is_fp8_meta_name, _model_num_layers,
                    _nvfp4_global_scale_name, _nvfp4_group_size, _nvfp4_scale_name,
                    _skeleton_from_cfg, _text_prefix, _weight_map)

GB = 1024 ** 3

# Sub-layer stages quantized SEQUENTIALLY inside one decoder layer (AutoGPTQ's
# inside_layer_modules): each stage's Hessian is collected with all EARLIER stages already
# quantized, so intra-layer error propagation is honored. Matched by Linear-name suffix; any
# layer Linear not matching falls into one trailing catch-all stage.
_STAGES = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",))


def _calib_tokens(model_dir: str, n_samples: int, seqlen: int):
    """Tokenize the bundled corpus with the MODEL'S tokenizer and cut n deterministic,
    evenly-strided windows of seqlen tokens. Returns LongTensor [n, seqlen] (CPU)."""
    import torch
    from transformers import AutoTokenizer
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_corpus.txt")
    if not os.path.isfile(path):
        raise RuntimeError("calib_corpus.txt missing next to gptq_pack.py — run /update "
                           "(it ships with the controller files)")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < seqlen + 1:
        raise RuntimeError(f"calibration corpus too short after tokenization ({ids.numel()} tokens)")
    n_fit = max(1, min(n_samples, ids.numel() // seqlen))
    span = ids.numel() - seqlen
    starts = [int(round(k * span / max(1, n_fit - 1))) for k in range(n_fit)] if n_fit > 1 else [0]
    return torch.stack([ids[s:s + seqlen] for s in starts])


def _find_params(Wg, grid: int, maxshrink: float):
    """Group scale/zero for one group's CURRENT weight block Wg [out, g] on the asymmetric 0..3
    grid — GPTQ's MSE shrink search (norm 2.4) over range-shrink candidates instead of raw
    min/max (at 2 bits the outlier-stretched RTN range wastes half the grid)."""
    import torch
    xmin = Wg.amin(dim=1).clamp(max=0.0)
    xmax = Wg.amax(dim=1).clamp(min=0.0)
    best_err = torch.full_like(xmin, float("inf"))
    best_s = torch.ones_like(xmin)
    best_z = torch.zeros_like(xmin)
    for k in range(int(maxshrink * grid)):
        p = 1.0 - k / grid
        s1 = ((xmax - xmin) * (p / 3.0)).clamp(min=1e-8)
        z1 = torch.round(-xmin * p / s1).clamp(0, 3)
        q = torch.clamp(torch.round(Wg / s1.unsqueeze(1) + z1.unsqueeze(1)), 0, 3)
        dq = (q - z1.unsqueeze(1)) * s1.unsqueeze(1)
        err = (dq - Wg).abs().pow(2.4).sum(dim=1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s1, best_s)
        best_z = torch.where(better, z1, best_z)
    return best_s, best_z


class _GPTQ:
    """One Linear's Hessian accumulator + solver. add_batch() from a forward-pre hook; solve()
    returns (qweight-crumbs, scale, zero) in pack_linear_int2's exact format."""

    def __init__(self, linear, device):
        import torch
        self.lin = linear
        self.dev = device
        self.in_f = linear.weight.shape[1]
        self.H = torch.zeros((self.in_f, self.in_f), dtype=torch.float32, device=device)
        self.n = 0

    def add_batch(self, x) -> None:
        import torch
        x = x.reshape(-1, self.in_f).to(self.dev, torch.float32)
        t = x.shape[0]
        self.H *= self.n / (self.n + t)
        self.n += t
        x = x * (2.0 / self.n) ** 0.5
        self.H += x.t() @ x

    def solve(self, group_size: int, grid: int, maxshrink: float, percdamp: float,
              blocksize: int = 128):
        import torch
        G = group_size
        W = self.lin.weight.data.to(self.dev, torch.float32).clone()
        out, in_f = W.shape
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        damp = percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(in_f, device=self.dev)
        H[idx, idx] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        ng = (in_f + G - 1) // G
        in_pad = ng * G
        qall = torch.zeros((out, in_pad), dtype=torch.uint8, device=self.dev)
        scales = torch.zeros((out, ng), dtype=torch.float32, device=self.dev)
        zeros = torch.zeros((out, ng), dtype=torch.float32, device=self.dev)
        DQ = torch.zeros_like(W)   # dequantized weight (written back for propagation)

        for i1 in range(0, in_f, blocksize):
            i2 = min(i1 + blocksize, in_f)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                col = i1 + i
                if col % G == 0:
                    g = col // G
                    sc, zp = _find_params(W[:, col:min(col + G, in_f)], grid, maxshrink)
                    scales[:, g] = sc
                    zeros[:, g] = zp
                g = col // G
                w = W1[:, i]
                d = Hinv1[i, i]
                s = scales[:, g]
                z = zeros[:, g]
                q = torch.clamp(torch.round(w / s + z), 0, 3)
                dq = (q - z) * s
                qall[:, col] = q.to(torch.uint8)
                DQ[:, col] = dq
                err = (w - dq) / d
                if i + 1 < count:
                    W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)
                Err1[:, i] = err
            if i2 < in_f:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        # padded tail columns (in_f % G != 0): q = the group's zero point -> dequant 0 (and the
        # kernel zero-pads activations there anyway, so the value is inert either way).
        if in_pad != in_f:
            qall[:, in_f:] = zeros[:, -1].round().clamp(0, 3).to(torch.uint8).unsqueeze(1)
        qpacked = (qall[:, 0::4] | (qall[:, 1::4] << 2)
                   | (qall[:, 2::4] << 4) | (qall[:, 3::4] << 6)).contiguous()

        # write the dequantized weight back into the live module: later stages' Hessians and the
        # layer's propagation forward then see EXACTLY what the served cache will compute.
        wdt = self.lin.weight.dtype
        self.lin.weight.data.copy_(DQ.to(wdt).to(self.lin.weight.device))
        del W, DQ, H, Hinv
        return qpacked.cpu(), scales.to(wdt).cpu(), zeros.to(wdt).cpu()


def compile_int2_gptq(model_dir: str, group_size: int = INT2_GROUP, progress=None) -> dict:
    """GPTQ-calibrated int2 shard-cache compile — the int2 body of shard_compile.compile_shards.
    Streams one decoder layer at a time: materialize the layer bf16 from disk, collect per-stage
    Hessians by running the calibration batch through it (earlier stages already quantized),
    solve every Linear, write the unit, run one final forward to hand the QUANTIZED layer's
    outputs to the next layer. Unit files/manifest match the RTN compile's layout exactly."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig

    n_samples = int(os.environ.get("INFINITEMODEL_GPTQ_SAMPLES", "32"))
    seqlen = int(os.environ.get("INFINITEMODEL_GPTQ_SEQLEN", "512"))
    percdamp = float(os.environ.get("INFINITEMODEL_GPTQ_PERCDAMP", "0.01"))
    grid = int(os.environ.get("INFINITEMODEL_GPTQ_GRID", "48"))
    maxshrink = 0.75

    wm = _weight_map(model_dir)
    if _has_moe_experts(wm):
        raise ValueError("int2 GPTQ compile supports DENSE models only (MoE has no 2-bit "
                         "expert path — use int4)")
    fp8_block = _fp8_block_size(model_dir)
    nvfp4_group = _nvfp4_group_size(model_dir)
    n_layers = _model_num_layers(model_dir)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        tied = bool(json.load(fh).get("tie_word_embeddings", False))
    prefix = _text_prefix(wm)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fdt = torch.bfloat16 if dev.type == "cuda" else torch.float32   # forward dtype

    # --- source reads (mirrors compile_shards' _get_bf16: fp8/nvfp4 sources dequant first) ----
    _open: dict = {}

    def _get(src: str):
        fn = wm[src]
        if fn not in _open:
            _open[fn] = safe_open(fn, framework="pt")
        return _open[fn].get_tensor(src)

    def _get_bf16(src: str):
        t = _get(src)
        if fp8_block is not None and t.dtype == torch.float8_e4m3fn and src.endswith(".weight"):
            sname = _fp8_scale_name(src)
            if sname in wm:
                return _dequant_fp8_to_bf16(t, _get(sname), fp8_block)
        if (nvfp4_group is not None and t.dtype == torch.uint8
                and src.endswith(".weight_packed")):
            sname, gname = _nvfp4_scale_name(src), _nvfp4_global_scale_name(src)
            if sname in wm and gname in wm:
                logical = [int(t.shape[0]), int(t.shape[1]) * 2]
                return _dequant_nvfp4_to_bf16(t, _get(sname), _get(gname), nvfp4_group, logical)
        return t.to(torch.bfloat16)

    def _out_name(src: str) -> str:
        o = src.replace(prefix, "model.", 1)
        if nvfp4_group is not None and o.endswith(".weight_packed"):
            o = o[: -len(".weight_packed")] + ".weight"
        return o

    # --- drivable skeleton: fresh REAL layer instances from the model's own config -------------
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    skel = _skeleton_from_cfg(cfg)     # meta — used for class discovery only
    inner = getattr(skel, "model", None)
    if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
        raise RuntimeError("int2 GPTQ compile can't drive this architecture (no model.layers/"
                           "embed_tokens) — dense llama/qwen/mistral class only")
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    layer_cls = type(inner.layers[0])
    rot_mod = getattr(inner, "rotary_emb", None)
    if rot_mod is None:
        raise RuntimeError("int2 GPTQ compile needs a model-level rotary_emb "
                           "(llama/qwen/mistral class) — not found on this architecture")
    rotary = type(rot_mod)(config=text_cfg).to(dev)   # fresh, real inv_freq

    # --- calibration inputs --------------------------------------------------------------------
    ids = _calib_tokens(model_dir, n_samples, seqlen)
    n, L = ids.shape
    embed_w = _get_bf16(f"{prefix}embed_tokens.weight")
    hs = torch.nn.functional.embedding(ids, embed_w).to(fdt)          # [n, L, hidden] (CPU pool)
    if str(getattr(text_cfg, "model_type", "")).startswith("gemma"):  # gemma scales embeddings
        hs = hs * (float(text_cfg.hidden_size) ** 0.5)
    pids = torch.arange(L, device=dev).unsqueeze(0)
    pos_emb = rotary(hs[:1].to(dev), pids)
    causal = torch.full((L, L), torch.finfo(fdt).min, dtype=fdt, device=dev).triu(1)[None, None]
    print(f"[gptq] int2 calibrated compile: {n}x{L} tokens, g{group_size}, grid {grid}, "
          f"damp {percdamp}, device {dev}", flush=True)

    def _call_layer(layer, h):
        out = layer(h, attention_mask=causal, position_ids=pids,
                    position_embeddings=pos_emb)
        return out[0] if isinstance(out, tuple) else out

    # --- manifest + unit writer (same layout/fields as the RTN compile) ------------------------
    from shard_compile import _packer_tag, _sha256_file, _shard_cache_root
    out_dir = os.path.join(_shard_cache_root(model_dir), "int2")
    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"format": 1, "quant": "int2", "group_size": group_size,
                      "num_layers": n_layers, "tied": tied, "files": {}, "tensors": {},
                      "packer_hash": _packer_tag("int2", group_size), "expert_layout": None,
                      "calib": {"samples": n, "seqlen": L, "grid": grid, "percdamp": percdamp}}

    def _save_unit(unit: str, out_sd: dict, mtensors: dict) -> None:
        for name, meta in mtensors.items():
            manifest["tensors"][name] = {"file": unit, **meta}
        path = os.path.join(out_dir, unit)
        save_file(out_sd, path)
        manifest["files"][unit] = {"sha256": _sha256_file(path), "bytes": os.path.getsize(path)}

    done, total = 0, n_layers + 2
    _save_unit("embed.safetensors", {"model.embed_tokens.weight": embed_w.contiguous()},
               {"model.embed_tokens.weight": {"q": False, "shape": [int(x) for x in embed_w.shape]}})
    done += 1
    if progress:
        progress(done, total)

    for li in range(n_layers):
        raw = {_out_name(s): _get_bf16(s) for s in wm
               if s.startswith(f"{prefix}layers.{li}.") and not _is_fp8_meta_name(s)}
        lp = f"model.layers.{li}."
        layer = layer_cls(text_cfg, layer_idx=li).to(fdt)
        missing, unexpected = layer.load_state_dict(
            {k[len(lp):]: v.to(fdt) for k, v in raw.items()}, strict=False)
        if missing:
            raise RuntimeError(f"layer {li} build incomplete — missing weights {sorted(missing)[:4]}")
        layer = layer.to(dev).eval()

        lins = {nm: m for nm, m in layer.named_modules()
                if isinstance(m, torch.nn.Linear)}
        staged: list[list[str]] = []
        seen: set = set()
        for stage in _STAGES:
            grp = [nm for nm in lins if nm.rsplit(".", 1)[-1] in stage]
            if grp:
                staged.append(sorted(grp))
                seen.update(grp)
        rest = sorted(nm for nm in lins if nm not in seen)
        if rest:
            staged.append(rest)

        packed: dict = {}
        with torch.no_grad():
            for grp in staged:
                solvers = {nm: _GPTQ(lins[nm], dev) for nm in grp}
                hooks = [lins[nm].register_forward_pre_hook(
                    (lambda s: (lambda _m, inp: s.add_batch(inp[0])))(sv))
                    for nm, sv in solvers.items()]
                for j in range(n):
                    _call_layer(layer, hs[j:j + 1].to(dev))
                for h in hooks:
                    h.remove()
                for nm in grp:
                    q, sc, zp = solvers[nm].solve(group_size, grid, maxshrink, percdamp)
                    packed[lp + nm + ".weight"] = (q, sc, zp)
                    del solvers[nm]
            for j in range(n):   # propagate the QUANTIZED layer's outputs to the next layer
                hs[j] = _call_layer(layer, hs[j:j + 1].to(dev))[0].to(hs.dtype).cpu()

        out_sd, mt = {}, {}
        for name, W in raw.items():
            pk = packed.get(name)
            if pk is not None:
                q, sc, zp = pk
                out_sd[name + ".qweight"], out_sd[name + ".scale"], out_sd[name + ".zero"] = q, sc, zp
                mt[name] = {"q": True, "in_features": int(W.shape[1]),
                            "shape": [int(x) for x in W.shape]}
            else:
                out_sd[name] = W.contiguous()
                mt[name] = {"q": False, "shape": [int(x) for x in W.shape]}
        _save_unit(f"L{li:04d}.safetensors", out_sd, mt)
        del layer, raw, packed, out_sd
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        done += 1
        print(f"[gptq] L{li}: {sum(len(g) for g in staged)} linears solved", flush=True)
        if progress:
            progress(done, total)

    from shards import _head_key
    head_src = f"{prefix}embed_tokens.weight" if tied else _head_key(wm, prefix)
    normw = _get_bf16(f"{prefix}norm.weight")
    headw = _get_bf16(head_src)
    _save_unit("head.safetensors",
               {"model.norm.weight": normw.contiguous(), "lm_head.weight": headw.contiguous()},
               {"model.norm.weight": {"q": False, "shape": [int(x) for x in normw.shape]},
                "lm_head.weight": {"q": False, "shape": [int(x) for x in headw.shape]}})
    done += 1
    if progress:
        progress(done, total)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest
