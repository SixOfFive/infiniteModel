"""#91 MTP (nextn) self-speculative decoding — the reusable MTP-head forward.

The Qwen3.5/3.6 checkpoint ships a 1-layer MTP head that the installed transformers DISCARDS
(_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]). This module hand-builds that head from the raw
checkpoint tensors so the controller can use it as a SELF-draft: given the main model's pre-final-norm
trunk hidden h_i and the embedding of the next token t_{i+1}, it predicts t_{i+2}.

Module (discovered from the checkpoint):
    mtp.pre_fc_norm_embedding   RMSNorm on embed(next_token)            (enorm)
    mtp.pre_fc_norm_hidden      RMSNorm on the trunk hidden             (hnorm)
    mtp.fc                      Linear 2H->H over cat([enorm, hnorm])   (eh_proj)
    mtp.layers.0                one full Qwen3_5MoeDecoderLayer (full-attn + 256-expert MoE)
    mtp.norm                    the MTP head's OWN final RMSNorm
    (shares model.language_model.embed_tokens + top-level lm_head)

Forward (Qwen3-Next / DeepSeek-MTP):
    x_i     = fc( cat[ enorm(embed(t_{i+1})), hnorm(h_i) ] )
    o       = decoder_layer(x)              # causal over the projected sequence
    logit_i = lm_head( mtp.norm(o_i) )      # predicts t_{i+2}

Kept SMALL + controller-resident on purpose: embed + 1 layer + head ≈ a few GB. NEVER load the full
model on the controller box (it co-hosts the controller) — see the never-full-load-on-controller-box
memory; trunk hidden comes from the DISTRIBUTED pipeline (capture_pre_norm).
"""
import json
import os


def load_mtp_head(model_dir: str, device: str = "cpu", dtype=None):
    """Build the MTP head from the checkpoint's mtp.* + shared embed/lm_head. Returns an MTPHead."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeTextRotaryEmbedding,
    )
    if dtype is None:
        dtype = torch.bfloat16

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    tcfg = cfg.get_text_config()
    H = tcfg.hidden_size

    # tensor_name -> shard file
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
    else:
        wm = {}
        single = os.path.join(model_dir, "model.safetensors")
        with safe_open(single, framework="pt") as sf:
            wm = {k: "model.safetensors" for k in sf.keys()}

    def _names():
        want = [k for k in wm if k == "mtp" or k.startswith("mtp.")]
        # shared embed + lm_head (names vary by multimodal nesting)
        embed = next((k for k in wm if k.endswith("embed_tokens.weight")
                      and "language_model" in k), None) \
            or next((k for k in wm if k.endswith("embed_tokens.weight")), None)
        lmhead = "lm_head.weight" if "lm_head.weight" in wm else \
            next((k for k in wm if k.endswith("lm_head.weight")), None)
        return want, embed, lmhead

    mtp_keys, embed_key, lmhead_key = _names()
    fetch = list(mtp_keys) + [embed_key, lmhead_key]
    by_file = {}
    for k in fetch:
        by_file.setdefault(wm[k], []).append(k)
    td = {}
    for f, ks in by_file.items():
        with safe_open(os.path.join(model_dir, f), framework="pt") as sf:
            for k in ks:
                td[k] = sf.get_tensor(k).to(dtype)

    def _rms(weight_key):
        n = Qwen3_5MoeRMSNorm(H, eps=tcfg.rms_norm_eps)
        with torch.no_grad():
            n.weight.copy_(td[weight_key])
        return n.to(device=device, dtype=dtype).eval()

    enorm = _rms("mtp.pre_fc_norm_embedding.weight")
    hnorm = _rms("mtp.pre_fc_norm_hidden.weight")
    mnorm = _rms("mtp.norm.weight")
    fc = torch.nn.Linear(2 * H, H, bias=False).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        fc.weight.copy_(td["mtp.fc.weight"])

    # The MTP decoder layer is full_attention (it has self_attn.*); force that and build idx 0.
    import copy as _copy
    lcfg = _copy.deepcopy(tcfg)
    lcfg.layer_types = ["full_attention"]
    layer = Qwen3_5MoeDecoderLayer(lcfg, 0).to(device=device, dtype=dtype).eval()
    lsd = {k[len("mtp.layers.0."):]: v for k, v in td.items() if k.startswith("mtp.layers.0.")}
    miss, unexp = layer.load_state_dict(lsd, strict=False)
    miss = [m for m in miss if "rotary" not in m and "inv_freq" not in m]

    rotary = Qwen3_5MoeTextRotaryEmbedding(tcfg).to(device=device).eval()
    embed_w = td[embed_key].to(device=device)
    lmhead_w = td[lmhead_key].to(device=device)
    return MTPHead(torch, tcfg, device, dtype, enorm, hnorm, mnorm, fc, layer, rotary,
                   embed_w, lmhead_w, miss, list(unexp))


class MTPHead:
    """Weight container only — the forward lives in the module-level functions below so it can be
    iterated (re-fetch + reload mtp_core) while REUSING an already-built, cached head's weights."""

    def __init__(self, torch, tcfg, device, dtype, enorm, hnorm, mnorm, fc, layer, rotary,
                 embed_w, lmhead_w, load_missing, load_unexpected):
        self.torch = torch
        self.tcfg = tcfg
        self.device = device
        self.dtype = dtype
        self.enorm, self.hnorm, self.mnorm = enorm, hnorm, mnorm
        self.fc, self.layer, self.rotary = fc, layer, rotary
        self.embed_w, self.lmhead_w = embed_w, lmhead_w
        self.load_missing = load_missing
        self.load_unexpected = load_unexpected


def _project(head, trunk_hidden, next_token_ids):
    """x_i = fc(cat[enorm(embed(t_{i+1})), hnorm(h_i)]) — the MTP input sequence."""
    torch = head.torch
    import torch.nn.functional as F
    th = trunk_hidden.to(device=head.device, dtype=head.dtype)
    tk = next_token_ids.to(device=head.device)
    e = F.embedding(tk, head.embed_w)
    return head.fc(torch.cat([head.enorm(e), head.hnorm(th)], dim=-1))


def mtp_prefill(head, trunk_hidden, next_token_ids, kv):
    """Fill the MTP layer's OWN KV cache over the prompt (#91 _decode_spec_mtp). The MTP layer is a
    real transformer layer whose self-attention needs prior context — drafting with an empty cache
    collapses acceptance (the contextless draft is wrong). Consumes MTP-seq positions 0..S-1
    (token t_{j+1} with hidden h_j), leaving `kv` ready for decode-time mtp_step at position S.
    trunk_hidden [1,S,H] = h_0..h_{S-1}; next_token_ids [1,S] = t_1..t_S; rotary/cache pos = index."""
    torch = head.torch
    with torch.inference_mode():
        x = _project(head, trunk_hidden, next_token_ids)
        S = x.shape[1]
        pos = torch.arange(0, S, device=head.device).unsqueeze(0)
        cos, sin = head.rotary(x, pos)
        pe = (cos.to(head.dtype), sin.to(head.dtype))
        causal = torch.triu(torch.full((S, S), float("-inf"), dtype=head.dtype,
                                       device=head.device), diagonal=1).view(1, 1, S, S)
        cache_position = torch.arange(0, S, device=head.device)
        head.layer(x, position_embeddings=pe, attention_mask=causal, position_ids=pos,
                   past_key_values=kv, cache_position=cache_position)


def mtp_step(head, kv, trunk_hidden, token_id: int, position: int):
    """One incremental MTP draft into the persistent cache `kv` (#91): consume token t at MTP-seq
    `position` with the trunk hidden h_prev, attending to all cached prior positions; return the
    logit row [V] predicting the next-next token. `position` == current cache length (the new slot);
    a lone decode query attends every cached key, so attention_mask=None is correct."""
    torch = head.torch
    import torch.nn.functional as F
    with torch.inference_mode():
        th = trunk_hidden.to(device=head.device, dtype=head.dtype)
        tk = torch.tensor([[int(token_id)]], device=head.device)
        e = F.embedding(tk, head.embed_w)
        x = head.fc(torch.cat([head.enorm(e), head.hnorm(th)], dim=-1))
        pos = torch.tensor([[int(position)]], device=head.device)
        cos, sin = head.rotary(x, pos)
        pe = (cos.to(head.dtype), sin.to(head.dtype))
        cache_position = torch.tensor([int(position)], device=head.device)
        o = head.layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
                       past_key_values=kv, cache_position=cache_position)
        if isinstance(o, tuple):
            o = o[0]
        return F.linear(head.mnorm(o), head.lmhead_w)[0, -1].float().cpu()


def mtp_forward_seq(head, trunk_hidden, next_token_ids, position_offset: int = 1):
    """Parallel teacher-forced forward over a sequence (acceptance probe).
    trunk_hidden [1,S,H] (pre-final-norm hidden at positions 0..S-1),
    next_token_ids [1,S] (token at position i+1, i.e. ids shifted left by one),
    -> logits [1,S,V] where logits[:,i] predicts the token at position i+2.
    position_offset chooses the rotary convention (1 = position of the consumed next-token;
    0 = the hidden's own position) — the probe tries both and keeps the better."""
    torch = head.torch
    import torch.nn.functional as F
    from transformers import DynamicCache
    with torch.inference_mode():
        x = _project(head, trunk_hidden, next_token_ids)
        S = x.shape[1]
        pos = torch.arange(position_offset, position_offset + S, device=head.device).unsqueeze(0)
        cos, sin = head.rotary(x, pos)
        pe = (cos.to(head.dtype), sin.to(head.dtype))
        causal = torch.triu(torch.full((S, S), float("-inf"), dtype=head.dtype,
                                       device=head.device), diagonal=1).view(1, 1, S, S)
        o = head.layer(x, position_embeddings=pe, attention_mask=causal,
                       position_ids=pos, past_key_values=DynamicCache())
        if isinstance(o, tuple):
            o = o[0]
        return F.linear(head.mnorm(o), head.lmhead_w)
