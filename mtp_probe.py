#!/usr/bin/env python3
"""#91 Increment 1 (probe): is the Qwen3.5/3.6 checkpoint's MTP (nextn) head correct + worth it
for self-speculative decoding?

The installed transformers DROPS the MTP head on load (_keys_to_ignore_on_load_unexpected =
[r"^mtp.*"]) — there is no class to build or run it. Before wiring MTP self-speculation into the
distributed pipeline (a real protocol change), this script reimplements the MTP forward from the raw
checkpoint tensors and measures, OFFLINE on one box, how often its drafted token matches the model's
own greedy continuation. That acceptance rate is the go/no-go signal: high (>~0.6) ⇒ the forward is
right AND speculation will pay off; low ⇒ either the forward is wrong or the head is weak.

MTP module (discovered via /mtp_probe on the checkpoint):
    mtp.pre_fc_norm_embedding   RMSNorm on embed(next_token)              (enorm)
    mtp.pre_fc_norm_hidden      RMSNorm on the trunk hidden               (hnorm)
    mtp.fc                      Linear 2H->H over cat([enorm, hnorm])     (eh_proj)
    mtp.layers.0                one full Qwen3_5MoeDecoderLayer (full-attn + 256-expert MoE)
    mtp.norm                    MTP's OWN final RMSNorm before the head
    (shares model.language_model.embed_tokens + top-level lm_head)

Canonical forward (Qwen3-Next / DeepSeek-MTP):
    x_i = fc( cat[ enorm(embed(t_{i+1})), hnorm(h_i) ] )   # h_i = main trunk hidden at position i
    o   = decoder_layer(x)                                 # causal over the projected sequence
    logit_i = lm_head( mtp.norm(o_i) )                     # predicts token t_{i+2}

Usage:  python mtp_probe.py <model_dir> [prompt]
Prints diagnostics + a final `RESULT {json}` line the controller endpoint parses.
"""
import json
import os
import sys
import time


def _log(*a):
    print(*a, flush=True)


def main() -> dict:
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    model_dir = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else (
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "The capital of Italy is Rome. The capital of Canada is Ottawa. "
        "The capital of Germany is")
    torch.manual_seed(0)
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 8)))
    except Exception:
        pass

    t0 = time.time()
    _log(f"[probe] loading config from {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    tcfg = cfg.get_text_config()
    H = tcfg.hidden_size
    _log(f"[probe] hidden={H} layers={tcfg.num_hidden_layers} vocab={tcfg.vocab_size} "
         f"heads={tcfg.num_attention_heads} kv={tcfg.num_key_value_heads} head_dim={tcfg.head_dim} "
         f"partial_rotary={getattr(tcfg,'partial_rotary_factor',None)} "
         f"attn_output_gate={getattr(tcfg,'attn_output_gate',None)} "
         f"mtp_layers={getattr(tcfg,'mtp_num_hidden_layers',None)}")

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    ids = tok(prompt, return_tensors="pt").input_ids
    L = ids.shape[1]
    _log(f"[probe] prompt tokens L={L}: {ids[0].tolist()}")

    _log("[probe] loading full model (cpu, bf16, eager) — this is the slow part…")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager", trust_remote_code=False).eval()
    _log(f"[probe] model loaded in {time.time()-t0:.1f}s")

    # Locate the text trunk's final norm + embed + lm_head regardless of multimodal nesting.
    lm = model
    base = getattr(model, "model", model)
    langm = getattr(base, "language_model", base)
    final_norm = langm.norm                       # model.language_model.norm
    embed = langm.embed_tokens                     # shared with MTP (mtp_use_dedicated_embeddings=False)
    lm_head = model.get_output_embeddings()        # top-level lm_head
    _log(f"[probe] embed={tuple(embed.weight.shape)} lm_head={tuple(lm_head.weight.shape)}")

    # Capture the trunk hidden = the INPUT to the final norm (== what mtp.pre_fc_norm_hidden consumes).
    captured = {}

    def _pre_hook(mod, args):
        captured["h"] = args[0].detach()
        return None
    hk = final_norm.register_forward_pre_hook(_pre_hook)

    with torch.inference_mode():
        out = lm(input_ids=ids, use_cache=False)
    hk.remove()
    logits = out.logits                            # [1, L, V] main-model logits
    trunk_h = captured["h"]                         # [1, L, H] pre-final-norm hidden
    _log(f"[probe] main forward done; trunk_h={tuple(trunk_h.shape)} logits={tuple(logits.shape)}")

    # --- hand-build the MTP module from mtp.* tensors -----------------------------------------
    def _load_mtp_tensors() -> dict:
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        want = {k: f for k, f in wm.items() if k == "mtp" or k.startswith("mtp.")}
        by_file = {}
        for k, f in want.items():
            by_file.setdefault(f, []).append(k)
        td = {}
        for f, ks in by_file.items():
            with safe_open(os.path.join(model_dir, f), framework="pt") as sf:
                for k in ks:
                    td[k] = sf.get_tensor(k)
        return td

    mt = _load_mtp_tensors()
    _log(f"[probe] loaded {len(mt)} mtp tensors")

    def _rms(weight_key) -> "torch.nn.Module":
        n = Qwen3_5MoeRMSNorm(H, eps=tcfg.rms_norm_eps)
        with torch.no_grad():
            n.weight.copy_(mt[weight_key].to(torch.bfloat16))
        return n.to(torch.bfloat16).eval()

    enorm = _rms("mtp.pre_fc_norm_embedding.weight")
    hnorm = _rms("mtp.pre_fc_norm_hidden.weight")
    mnorm = _rms("mtp.norm.weight")
    fc = torch.nn.Linear(2 * H, H, bias=False).to(torch.bfloat16).eval()
    with torch.no_grad():
        fc.weight.copy_(mt["mtp.fc.weight"].to(torch.bfloat16))

    # The single MTP decoder layer is full_attention (it has self_attn.*). Build it as such.
    import copy as _copy
    lcfg = _copy.deepcopy(tcfg)
    lcfg.layer_types = ["full_attention"]
    layer = Qwen3_5MoeDecoderLayer(lcfg, 0).to(torch.bfloat16).eval()
    lsd = {k[len("mtp.layers.0."):]: v.to(torch.bfloat16)
           for k, v in mt.items() if k.startswith("mtp.layers.0.")}
    miss, unexp = layer.load_state_dict(lsd, strict=False)
    miss = [m for m in miss if "rotary" not in m and "inv_freq" not in m]
    _log(f"[probe] decoder layer load: missing={miss[:8]}{'...' if len(miss)>8 else ''} "
         f"unexpected={list(unexp)[:8]}{'...' if len(unexp)>8 else ''}")

    rot = Qwen3_5MoeTextRotaryEmbedding(tcfg).eval()

    # --- run the MTP head over the (shifted) sequence -----------------------------------------
    # input at index i: embed(t_{i+1}) + trunk_h[i] -> predicts t_{i+2}, for i in 0..L-2
    Lm = L - 1
    with torch.inference_mode():
        e = embed(ids[:, 1:L]).to(torch.bfloat16)         # [1, Lm, H]
        hh = trunk_h[:, 0:Lm].to(torch.bfloat16)          # [1, Lm, H]
        x = fc(torch.cat([enorm(e), hnorm(hh)], dim=-1))  # [1, Lm, H]

        causal = torch.triu(torch.full((Lm, Lm), float("-inf"), dtype=torch.bfloat16), diagonal=1)
        causal = causal.view(1, 1, Lm, Lm)

        results_by_pos = {}
        for tag, pos in (("pos0", torch.arange(0, Lm)), ("pos1", torch.arange(1, Lm + 1))):
            from transformers import DynamicCache
            pe = rot(x, pos.unsqueeze(0))                  # (cos, sin)
            o = layer(x, position_embeddings=pe, attention_mask=causal,
                      position_ids=pos.unsqueeze(0), past_key_values=DynamicCache())
            if isinstance(o, tuple):
                o = o[0]
            mlog = lm_head(mnorm(o))                        # [1, Lm, V]
            results_by_pos[tag] = mlog.float()

    # compare for i in 0..L-3 (need t_{i+2} to exist)
    n = L - 2
    main_greedy = logits[0].float().argmax(-1)            # main_greedy[i] predicts t_{i+1}
    actual = ids[0]
    summary = {}
    examples = {}
    for tag, mlog in results_by_pos.items():
        mtp_pred = mlog[0].argmax(-1)                      # mtp_pred[i] predicts t_{i+2}
        acc_actual = acc_greedy = 0
        ex = []
        for i in range(0, n - 1):                          # i+2 <= L-1
            mp = int(mtp_pred[i])
            ta = int(actual[i + 2])
            tg = int(main_greedy[i + 1])                   # main's greedy t_{i+2}
            acc_actual += (mp == ta)
            acc_greedy += (mp == tg)
            if len(ex) < 8:
                ex.append({"i": i, "mtp": mp, "actual": ta, "main_greedy": tg,
                           "mtp_tok": tok.decode([mp]), "greedy_tok": tok.decode([tg])})
        denom = max(1, n - 1)
        summary[tag] = {"acc_vs_actual": round(acc_actual / denom, 3),
                        "acc_vs_greedy": round(acc_greedy / denom, 3), "n": denom}
        examples[tag] = ex

    best = max(summary, key=lambda t: summary[t]["acc_vs_greedy"])
    res = {"ok": True, "L": L, "compared": n - 1, "best_pos": best,
           "summary": summary, "examples": examples[best],
           "load_missing": miss[:12], "load_unexpected": list(unexp)[:12],
           "elapsed_s": round(time.time() - t0, 1)}
    return res


if __name__ == "__main__":
    try:
        r = main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        r = {"ok": False, "error": repr(exc)}
    print("RESULT " + json.dumps(r), flush=True)
