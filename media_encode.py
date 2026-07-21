"""media_encode.py: the controller's media/speech encode family (code-split Inc 11).

RELOCATED VERBATIM from server.py — originally every function/method body was BYTE-IDENTICAL
to its server.py original; only this header (and the EngineSpeechMixin class wrapper) was new.
(Since diverged: the #speech-idle-evict group below — audit #27 — touched _load_speech_components
and split generate_speech into a pin wrapper + _generate_speech_inner.) Members:
_encode_images, _encode_audio_gemma4, _encode_audio, the #P6 speech-out group (_SPEECH_CACHE /
_SPEECH_MAT / _ensure_spk_dict / _materialize_from_prefix / SPEECH_DEVICE /
_load_speech_components + the #speech-idle-evict reaper), and Engine.generate_speech as
EngineSpeechMixin (composed into ``class Engine(..., EngineSpeechMixin)`` in server.py;
self.capture_thinker resolves via MRO from EngineGenMixin).

THIS MODULE IS THE CANONICAL HOME OF ``ENCODING`` — the >0-while-encoding idle-gate counter the
self-updater reads. All FOUR mutators (`global ENCODING` in _encode_images / _encode_audio /
_load_speech_components / generate_speech) live here, so their rebinds land in THIS module's
namespace, and server.py's self-update idle lambda reads ``media_encode.ENCODING`` as a LIVE
module attribute. ENCODING must NEVER be back-imported into server.py or published via state
(an int from-import/snapshot freezes at its current value and silently decouples the gate —
the original "ENCODING hazard" of state.py's SAFETY NOTE, now resolved by this move).

Bound leaf (m4c152 convention): server-side globals the bodies reference (MODELS_DIR,
_safe_name, HF_TOKEN, _controller_model_dir, the multimodal.* helpers, the timestamping print
shadow) are injected at startup by state.bind() — see state.py. Module-level imports below are
only what executes at import time (SPEECH_DEVICE's env read) plus the leaf stdlib convention.
In server.py's EXTRA_UPDATE_FILES + its convergence bridge.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import shutil
import threading
import time

# Canonical idle-gate (relocated from server.py; see the module docstring).
ENCODING: int = 0                  # >0 while a vision/audio encode is in flight (idle-gate guard)


def _encode_images(target_id: str, images: list) -> dict:
    """Run the image processor + vision tower. Returns {image_embeds [N,hidden], grid_thw,
    info}. image_embeds are the per-image-token features to splice into stage-0's
    embed_tokens output at the image-placeholder positions (increment 3)."""
    import torch
    global ENCODING
    ENCODING += 1   # guard: keep the self-update idle gate closed while we encode
    try:
        t0 = time.time()
        model, dev = _load_vision_encoder(target_id)
        mtype = getattr(getattr(model, "config", None), "model_type", "") or ""
        if mtype == "gemma4_unified":
            # #143 Gemma 4 unified: encoder-free vision — the embedder projects raw merged
            # pixel patches straight into LM space. The HF image processor hard-requires
            # torchvision, so preprocessing is the pure-torch reimplementation in
            # multimodal._gemma4_preprocess. get_image_features(pixel_values,
            # image_position_ids) returns pooler_output ALREADY padding-stripped
            # [total_valid_patches, text_hidden] — LM-ready, splice as-is. counts are the
            # REAL per-image soft-token counts (the reference processor expands each
            # <|image|> to exactly that many). Plain 1D positions; 'wrap' tells the serve
            # path to bracket each expanded run in boi/eoi (replace_image_token parity).
            # NOTE: the reference runs BIDIRECTIONAL attention across each image block
            # (use_bidirectional_attention='vision'); the pipeline now HONORS this (039fb24) —
            # the image-span positions ride the frame header as bidir_spans and every stage's
            # _causal_addmask ORs a blockwise overlay onto the mask, gated on the text cfg flag
            # (byte-identical for non-bidir models). Live-validated e2e (26b int4, om3nbox).
            t_load = time.time()
            pre = _gemma4_preprocess(images, target_id)
            pv = pre["pixel_values"].to(dev)
            ipos = pre["image_position_ids"].to(dev)
            counts = [int(c) for c in pre["num_soft_tokens_per_image"]]
            info = {"device": dev, "pixel_values_shape": list(pv.shape),
                    "load_s": round(t_load - t0, 1)}
            t_fwd = time.time()
            with torch.inference_mode():
                feats = model.get_image_features(pixel_values=pv, image_position_ids=ipos)
            emb = getattr(feats, "pooler_output", None)
            if emb is None:
                emb = _as_feature_tensor(feats)
            emb = emb.reshape(-1, emb.shape[-1])
            cfg = model.config
            itid = getattr(cfg, "image_token_id", None)
            tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") \
                else getattr(cfg, "text_config", cfg)
            out_hidden = getattr(tcfg, "hidden_size", None) or int(emb.shape[-1])
            boi = getattr(cfg, "boi_token_id", None)
            eoi = getattr(cfg, "eoi_token_id", None)
            info.update({"arch": "gemma4_unified", "forward_s": round(time.time() - t_fwd, 1),
                         "raw_return_type": type(feats).__name__,
                         "embeds_shape": list(emb.shape),
                         "path": "get_image_features(pixel_values, image_position_ids)"})
            print(f"[vision] encoded {len(images)} image(s) on {dev} [gemma4_unified]: "
                  f"load={info['load_s']}s forward={info['forward_s']}s -> {list(emb.shape)} "
                  f"counts={counts} itid={itid}")
            return {"image_embeds": emb, "grid_thw": None, "info": info, "counts": counts,
                    "image_token_id": itid, "out_hidden": out_hidden, "merge": 1,
                    "grid_list": [], "pos_scheme": "1d",
                    "wrap": ((int(boi), int(eoi))
                             if (boi is not None and eoi is not None) else None)}
        if mtype == "gemma4":
            # Gemma 4 TOWER variant (31b-it, 26b-a4b-it): a REAL Gemma4VisionModel ViT
            # (patch_embedder -> 27-layer encoder -> 3x3 pooler) + the embed_vision
            # projector. Unlike the 12b 'gemma4_unified' (encoder-free; its processor
            # PRE-merges the 3x3 pool into 6912-dim patches), the tower does the pooling
            # INSIDE itself, so it consumes UNMERGED 768-dim (16x16x3) teacher patches +
            # image_position_ids [B, max_patches, 2] (the tower's learned 2D pos-emb + 2D
            # RoPE, theta=100). The real Gemma4ImageProcessor emits exactly pixel_values /
            # image_position_ids / num_soft_tokens_per_image (a torchvision-free PIL variant
            # exists), so — unlike the unified path — no pure-torch reimplementation is
            # needed. get_image_features(pixel_values, image_position_ids) runs tower +
            # projector and returns pooler_output ALREADY padding-stripped
            # [total_soft_tokens, text_hidden] — LM-ready, splice as-is. Plain 1D LM
            # positions; 'wrap' brackets each expanded run in boi/eoi.
            # NOTE: the reference runs LM-side BLOCK-BIDIRECTIONAL attention across each
            # image span (use_bidirectional_attention='vision'); the pipeline now HONORS this
            # (039fb24) via bidir_spans threaded to every stage's mask (same path as the 12b
            # unified), gated on the text cfg flag — byte-identical for non-bidir models.
            ip = _get_image_processor(target_id)
            t_load = time.time()
            inputs = ip(images=images, return_tensors="pt")
            pv = inputs["pixel_values"].to(dev)
            ipos = inputs["image_position_ids"].to(dev)
            nsoft = inputs.get("num_soft_tokens_per_image")
            counts = None
            if nsoft is not None:
                counts = [int(c) for c in (nsoft.tolist() if hasattr(nsoft, "tolist") else nsoft)]
            info = {"device": dev, "pixel_values_shape": list(pv.shape),
                    "load_s": round(t_load - t0, 1)}
            t_fwd = time.time()
            with torch.inference_mode():
                feats = model.get_image_features(pixel_values=pv, image_position_ids=ipos)
            emb = getattr(feats, "pooler_output", None)
            if emb is None:
                emb = _pick_merged_embeds(feats, None)
            emb = emb.reshape(-1, emb.shape[-1])
            cfg = model.config
            itid = getattr(cfg, "image_token_id", None)
            tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") \
                else getattr(cfg, "text_config", cfg)
            out_hidden = getattr(tcfg, "hidden_size", None) or int(emb.shape[-1])
            boi = getattr(cfg, "boi_token_id", None)
            eoi = getattr(cfg, "eoi_token_id", None)
            if counts is None and len(images) == 1:   # single image -> all rows are its soft tokens
                counts = [int(emb.shape[0])]
            if counts is not None and sum(counts) != emb.shape[0]:
                print(f"[vision] WARN gemma4-tower counts sum {sum(counts)} != embeds "
                      f"{emb.shape[0]}")
            info.update({"arch": "gemma4", "forward_s": round(time.time() - t_fwd, 1),
                         "raw_return_type": type(feats).__name__,
                         "embeds_shape": list(emb.shape),
                         "path": "get_image_features(pixel_values, image_position_ids) [tower]"})
            print(f"[vision] encoded {len(images)} image(s) on {dev} [gemma4-tower]: "
                  f"load={info['load_s']}s forward={info['forward_s']}s -> {list(emb.shape)} "
                  f"counts={counts} itid={itid}")
            return {"image_embeds": emb, "grid_thw": None, "info": info, "counts": counts,
                    "image_token_id": itid, "out_hidden": out_hidden, "merge": 1,
                    "grid_list": [], "pos_scheme": "1d",
                    "wrap": ((int(boi), int(eoi))
                             if (boi is not None and eoi is not None) else None)}
        ip = _get_image_processor(target_id)
        t_load = time.time()
        if mtype == "mistral3":
            # Pixtral resizes/pads to the MERGED patch grid (vision patch_size * spatial_merge_size
            # = 32); the bare PixtralImageProcessor defaults to 16, so pass the merged size
            # explicitly — matching the canonical PixtralProcessor. Without this, image_sizes align
            # to 16 and the tower sees a ~4x, off-distribution tiling (degraded understanding).
            _vc = getattr(model.config, "vision_config", None)
            _ps = int(getattr(_vc, "patch_size", 16) or 16) if _vc is not None else 16
            _sm = int(getattr(model.config, "spatial_merge_size", 2) or 2)
            inputs = ip(images=images, patch_size=_ps * _sm, return_tensors="pt")
        else:
            inputs = ip(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(dev)
        grid = inputs.get("image_grid_thw")
        grid_dev = grid.to(dev) if grid is not None else None
        info: dict = {"device": dev, "pixel_values_shape": list(pv.shape),
                      "load_s": round(t_load - t0, 1)}
        if mtype == "mistral3":
            # Pixtral / Mistral3: a SEPARATE vision_tower + multi_modal_projector (both
            # materialized by _load_vision_encoder). get_image_features(pixel_values, image_sizes)
            # returns pooler_output as a TUPLE of per-image [tokens_i, text_hidden] (already
            # LM-ready / projected). No image_grid_thw, no spatial-merge math here, image_token_id
            # from config (10), and PLAIN 1D positions (pos_scheme='1d' -> serving skips mRoPE).
            sizes = inputs.get("image_sizes")
            sizes_dev = sizes.to(dev) if sizes is not None else None
            t_fwd = time.time()
            with torch.inference_mode():
                # Pass vision_feature_layer explicitly (config default -1, an int) so correctness
                # doesn't hinge on the @merge_with_config_defaults decorator injecting it.
                feats = model.get_image_features(
                    pixel_values=pv, image_sizes=sizes_dev,
                    vision_feature_layer=getattr(model.config, "vision_feature_layer", -1))
            pooler = getattr(feats, "pooler_output", None)
            if pooler is None and isinstance(feats, (tuple, list)):   # @can_return_tuple path
                for x in feats:
                    if isinstance(x, (tuple, list)) and x and all(
                            isinstance(t, torch.Tensor) for t in x):
                        pooler = x
                        break
            parts = [pooler] if isinstance(pooler, torch.Tensor) \
                else [t for t in (pooler or []) if isinstance(t, torch.Tensor)]
            if not parts:
                raise RuntimeError("mistral3 get_image_features returned no image embeds "
                                   f"(type={type(feats).__name__})")
            parts = [t.reshape(-1, t.shape[-1]) for t in parts]   # each -> [tokens_i, hidden]
            emb = torch.cat(parts, dim=0)
            counts = [int(t.shape[0]) for t in parts]
            cfg = model.config
            itid = getattr(cfg, "image_token_id", None)
            if itid is None:
                itid = getattr(cfg, "image_token_index", None)
            tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") \
                else getattr(cfg, "text_config", cfg)
            out_hidden = getattr(tcfg, "hidden_size", None) or int(emb.shape[-1])
            info.update({"arch": "mistral3", "forward_s": round(time.time() - t_fwd, 1),
                         "raw_return_type": type(feats).__name__, "embeds_shape": list(emb.shape),
                         "path": "get_image_features(pixel_values, image_sizes)"})
            # #150 Pixtral row structure: per-image merged (H_tok, W_tok) grid so the serve path
            # can insert [IMG_BREAK] between rows + [IMG_END] at the end (the layout Pixtral/
            # Mistral3 was trained with). Reuse the EXACT merged cell the image processor was
            # called with (_ps * _sm, above) so image_sizes — which are multiples of that cell —
            # divide cleanly; a divergent fallback would silently disable the feature. Emit a grid
            # ONLY when H·W == this image's count; else [] there (expander falls back to the flat
            # run for that image — no behavior change).
            _cell = int(_ps) * int(_sm)
            grid_rc: list = []
            try:
                _sl = sizes.tolist() if hasattr(sizes, "tolist") else (list(sizes) if sizes is not None else [])
                for i, hw in enumerate(_sl):
                    h, w = int(hw[0]), int(hw[1])
                    r, cc = (h // _cell, w // _cell) if _cell > 0 else (0, 0)
                    grid_rc.append([r, cc] if (i < len(counts) and r * cc == counts[i]) else None)
                if all(g is None for g in grid_rc):
                    grid_rc = []
            except Exception:
                grid_rc = []
            print(f"[vision] encoded {len(images)} image(s) on {dev} [mistral3]: "
                  f"load={info['load_s']}s forward={info['forward_s']}s -> {list(emb.shape)} "
                  f"counts={counts} itid={itid} grid_rc={grid_rc}")
            return {"image_embeds": emb, "grid_thw": None, "info": info, "counts": counts,
                    "image_token_id": itid, "out_hidden": out_hidden, "merge": 1,
                    "grid_list": [], "pos_scheme": "1d", "grid_rc": grid_rc}
        visual, _prefix = _resolve_visual(model)
        t_fwd = time.time()
        with torch.inference_mode():
            # The visual tower's OWN forward runs patch_embed -> blocks -> merger and returns
            # the LM-READY MERGED tokens [prod(grid)/merge^2, out_hidden(==text_hidden)].
            # get_image_features returns only the PRE-merge backbone [patches, vision_hidden],
            # which is the wrong dim/count to splice — so call visual() directly.
            try:
                feats = visual(pv, grid_dev)
                info["path"] = "visual(pixel_values, grid_thw)"
            except Exception as exc:
                info["visual_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                feats = model.get_image_features(pixel_values=pv, image_grid_thw=grid_dev)
                info["path"] = "get_image_features(fallback)"
        info["forward_s"] = round(time.time() - t_fwd, 1)
        info["raw_return_type"] = type(feats).__name__
        # Splice the MERGED tokens (pooler_output, dim == text hidden), not the pre-merge
        # backbone (last_hidden_state). Pick by hidden dim == out_hidden. (Omni-aware: vcfg +
        # image_token_id are nested under thinker_config for Qwen2.5-Omni.)
        vcfg, image_token_id = _vision_cfg_and_token(model)
        tcfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") \
            else getattr(model.config, "text_config", model.config)
        out_hidden = (getattr(vcfg, "out_hidden_size", None) if vcfg is not None else None) \
            or getattr(tcfg, "hidden_size", None)
        emb = _pick_merged_embeds(feats, out_hidden)
        info["embeds_shape"] = list(emb.shape)
        # Per-image merged-token COUNT = prod(t,h,w) / merge^2 — used to expand each single
        # <|image_pad|> placeholder into the right run and align positions with `emb` rows.
        merge = int(getattr(vcfg, "spatial_merge_size", 1) or 1) if vcfg is not None else 1
        counts = []
        if grid is not None:
            for row in grid.tolist():
                prod = 1
                for d in row:
                    prod *= int(d)
                counts.append(prod // (merge * merge))
        print(f"[vision] encoded {len(images)} image(s) on {dev}: load={info['load_s']}s "
              f"forward={info['forward_s']}s -> {list(emb.shape)} counts={counts}")
        return {"image_embeds": emb, "grid_thw": grid, "info": info, "counts": counts,
                "image_token_id": image_token_id, "out_hidden": out_hidden, "merge": merge,
                "grid_list": (grid.tolist() if grid is not None else []), "pos_scheme": "mrope"}
    finally:
        ENCODING -= 1


def _encode_audio_gemma4(target_id: str, audios: list, sampling_rate: int, t0: float) -> dict:
    """#144 Gemma-4 unified audio (mirror of the #143 encoder-free vision path): frame the waveform
    into 640-sample soft tokens and project them through model.embed_audio via get_audio_features.
    Returns per-audio-token embeds to splice at the audio_token (258881) positions, bracketed
    boa/eoa, with plain 1D positions (pos_scheme '1d' — the serve path then skips TMRoPE)."""
    import torch
    model, dev = _load_gemma4_audio_encoder(target_id)
    inner = getattr(model, "model", model)
    pre = _gemma4_audio_preprocess(audios, target_id)
    if pre.get("dropped_frames"):   # #144: cap at audio_seq_length is a safety bound, never silent
        print(f"[audio] WARN gemma4 audio TRUNCATED: dropped {pre['dropped_frames']} frame(s) "
              f"(~{pre['dropped_seconds']}s) past the {pre['max_tokens']}-token cap")
    feats = pre["input_features"].to(dev)
    mask = pre["input_features_mask"].to(dev)
    t_load = time.time()
    cfg = model.config
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
    out_hidden = getattr(tcfg, "hidden_size", None)
    gaf = getattr(model, "get_audio_features", None) or getattr(inner, "get_audio_features")
    t_fwd = time.time()
    with torch.inference_mode():
        out = gaf(feats, input_features_mask=mask)
    ft = None
    for _a in ("audio_features", "audio_embeds", "last_hidden_state", "pooler_output"):
        _v = getattr(out, _a, None)
        if isinstance(_v, torch.Tensor):
            ft = _v
            break
    if ft is None:
        ft = _pick_merged_embeds(out, out_hidden)
    # get_audio_features has NO downsampling (unified model), so [B,T,H] aligns with the [B,T] mask —
    # mask-select the valid frames into a flat [sum(valid), H] in row-major (audio, token) order.
    if ft.dim() == 3:
        emb = ft[mask]
    else:
        emb = ft.reshape(-1, ft.shape[-1])
    counts = [int(x) for x in mask.sum(-1).tolist()]
    if out_hidden is None:
        out_hidden = int(emb.shape[-1])
    atid, boa, eoa = _gemma4_audio_token_ids(model)
    info = {"device": str(dev), "arch": "gemma4_unified", "load_s": round(t_load - t0, 1),
            "forward_s": round(time.time() - t_fwd, 1), "embeds_shape": list(emb.shape)}
    if sum(counts) != int(emb.shape[0]):
        print(f"[audio] WARN gemma4 counts sum {sum(counts)} != embeds {emb.shape[0]}")
    print(f"[audio] encoded {len(audios)} clip(s) on {dev} [gemma4_unified]: "
          f"load={info['load_s']}s forward={info['forward_s']}s -> {list(emb.shape)} "
          f"counts={counts} atid={atid} boa/eoa={boa}/{eoa}")
    return {"audio_embeds": emb, "counts": counts, "audio_token_id": atid,
            "out_hidden": out_hidden, "info": info,
            "wrap": ((boa, eoa) if (boa is not None and eoa is not None) else None),
            "pos_scheme": "1d"}


def _encode_audio(target_id: str, audios: list, sampling_rate: int = 16000) -> dict:
    """Run the feature extractor + audio tower. `audios` is a list of 1-D float32 waveforms at
    `sampling_rate` Hz. Returns {audio_embeds [total_tokens, hidden], counts (per-audio token
    counts), audio_token_id, out_hidden, info}. audio_embeds are the per-audio-token features to
    splice at the audio-placeholder positions. Routes Gemma-4 unified audio (#144) to its
    encoder-free path; every other audio model (Qwen2.5-Omni) uses the feature-extractor + tower."""
    import torch
    global ENCODING
    ENCODING += 1   # keep the self-update idle gate closed while we encode
    try:
        t0 = time.time()
        from transformers import AutoConfig
        _mtype = ""
        with contextlib.suppress(Exception):
            _mtype = getattr(AutoConfig.from_pretrained(target_id), "model_type", "") or ""
        if _mtype == "gemma4_unified":
            return _encode_audio_gemma4(target_id, audios, sampling_rate, t0)
        fe = _get_audio_feature_extractor(target_id)
        model, dev = _load_audio_encoder(target_id)
        t_load = time.time()
        feats_in = fe(audios, sampling_rate=sampling_rate, return_tensors="pt",
                      return_attention_mask=True)
        input_features = feats_in["input_features"].to(dev)
        # The feature extractor returns 'attention_mask'; Omni's get_audio_features expects
        # 'feature_attention_mask'. Same tensor, renamed.
        fam = feats_in.get("feature_attention_mask", feats_in.get("attention_mask"))
        fam = fam.to(dev) if fam is not None else None
        info: dict = {"device": dev, "input_features_shape": list(input_features.shape),
                      "load_s": round(t_load - t0, 1),
                      "feature_attention_mask_shape": (list(fam.shape) if fam is not None else None)}
        thinker = getattr(model, "thinker", model)
        t_fwd = time.time()
        with torch.inference_mode():
            feats = thinker.get_audio_features(input_features, feature_attention_mask=fam)
        info["forward_s"] = round(time.time() - t_fwd, 1)
        info["raw_return_type"] = type(feats).__name__
        # text hidden lives under thinker_config.text_config (NULL on thinker_config itself);
        # get_text_config() resolves it canonically.
        cfg = model.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else \
            getattr(getattr(cfg, "thinker_config", cfg), "text_config",
                    getattr(cfg, "thinker_config", cfg))
        out_hidden = getattr(text_cfg, "hidden_size", None)
        # get_audio_features returns the tower last_hidden_state (already projected to text
        # hidden) — pick the tensor whose width == text hidden.
        emb = _pick_merged_embeds(feats, out_hidden)
        # Defensive: get_audio_features mask-selects to a FLAT [total_tokens, hidden] (clips
        # concatenated in batch order), which is what the splice expects. If a transformers
        # version instead returns a batched [n_clips, seq, hidden], flatten it so shape[0] is
        # the token count (else the sum(counts)==shape[0] check below would compare against
        # n_clips and wrongly reject). Harmless when already 2D.
        if emb.dim() == 3:
            emb = emb.reshape(-1, emb.shape[-1])
            info["flattened_from_3d"] = True
        info["embeds_shape"] = list(emb.shape)
        info["embeds_dtype"] = str(emb.dtype)
        # Per-audio token counts (to expand each <|AUDIO|> placeholder in 5c) — processor
        # formula, validated to SUM to the actual embed count.
        feature_lens = fam.sum(-1) if fam is not None else None
        counts, how = (None, "unavailable")
        if feature_lens is not None:
            counts = _audio_out_lengths(feature_lens)
            how = "processor_formula"
            if sum(counts) != int(emb.shape[0]):
                print(f"[audio] WARN counts sum {sum(counts)} != embeds {emb.shape[0]} "
                      f"(feature_lens={feature_lens.tolist()})")
                how += f"(MISMATCH:{sum(counts)}vs{int(emb.shape[0])})"
        if counts is None:                       # last-ditch: single audio => all tokens are it
            counts = [int(emb.shape[0])] if len(audios) == 1 else None
            how = "embeds_total(single-audio)" if counts else how
        info["counts_how"] = how
        audio_token_id = _omni_audio_token_id(model)
        print(f"[audio] encoded {len(audios)} clip(s) on {dev}: load={info['load_s']}s "
              f"forward={info['forward_s']}s -> {list(emb.shape)} counts={counts} "
              f"audio_token_id={audio_token_id}")
        return {"audio_embeds": emb, "counts": counts, "audio_token_id": audio_token_id,
                "out_hidden": out_hidden, "info": info}
    finally:
        ENCODING -= 1


# ===================== #P6 speech-out Phase 2: Talker + token2wav loader =====================
# The Thinker runs DISTRIBUTED (its hidden states come back via the Phase-1 capture transport).
# The Talker (a codec LM) + token2wav (DiT + BigVGAN vocoder) are the ~4B "speech head"; we run
# them on the CONTROLLER (CPU by default — float32 vocoder, shared beast GPU is crash-prone).
# We meta-build the full Omni (zero memory: thinker/talker/token2wav all on meta), then
# materialize ONLY talker + token2wav + the thinker embed matrix (needed for the talker-input
# assembly: bos/eos/pad embeds + thinker_token_embeds), and load spk_dict.pt (speaker conds).
_SPEECH_CACHE: dict = {}    # target_id -> dict(model, talker, token2wav, embed, speaker_map, dev)
_SPEECH_MAT: dict = {}      # target_id -> {component: [(name, shape, how)]}

# #speech-idle-evict (audit #27): _SPEECH_CACHE pinned the ~14-18 GB fp32 talker+token2wav+embed
# head in controller RAM FOREVER — one speech request permanently cost beast the RAM the
# render-oom-guard / t2a-offload paths arbitrate over. The trio below + the lazy daemon thread
# give it the same idle discipline models get: no speech use for > the window -> drop the head
# (it lazy-rebuilds on the next request; heavy, ~minutes, an accepted trade for the RAM).
_SPEECH_LAST_USE: dict = {}     # target_id -> time.monotonic() of last cache touch (reaper input)
_SPEECH_INFLIGHT: dict = {}     # target_id -> live generate_speech count (reaper skip; covers the
#                                 thinker-capture phase where ENCODING is 0 but the head is in use)
_SPEECH_LOCK = threading.Lock()        # guards the two dicts above (reaper thread vs. servers)
_SPEECH_BUILD_LOCK = threading.Lock()  # serializes the heavy build (evict -> 2 cold requests race)
_SPEECH_REAPER_ON = False              # lazy one-shot; rebound only here (no outside reader)


def _speech_idle_window_s() -> float:
    """#speech-idle-evict window (seconds). Knob: ENGINE_CONFIG['speech_idle_unload_m'] —
    minutes of NO speech use before the cached head is dropped (engine_config.json; hand-set —
    /config has no dedicated param yet). Absent -> follow the model sweep's idle_unload_m, so
    turning idle-unload on reaps this cache too with the SAME window; <= 0 (the -1 'forever'
    sentinel included, same as every idle_unload_m consumer) -> keep forever (old behavior,
    and still the default default since idle_unload_m defaults to 0)."""
    try:
        cfg = ENGINE_CONFIG            # injected by state.bind(); NameError pre-bind/standalone
    except NameError:
        return 0.0
    v = cfg.get("speech_idle_unload_m")
    if v in (None, ""):
        v = cfg.get("idle_unload_m", 0)
    try:
        m = float(v or 0)
    except (TypeError, ValueError):
        m = 0.0
    return m * 60.0 if m > 0 else 0.0


def _speech_touch(target_id: str) -> None:
    """Restart a cached head's idle clock (cache hit / fresh build / request end)."""
    with _SPEECH_LOCK:
        _SPEECH_LAST_USE[target_id] = time.monotonic()


def _speech_pin(target_id: str) -> None:
    """Pin a head for one in-flight speech request (generate_speech wrapper)."""
    with _SPEECH_LOCK:
        _SPEECH_INFLIGHT[target_id] = _SPEECH_INFLIGHT.get(target_id, 0) + 1


def _speech_unpin(target_id: str) -> None:
    with _SPEECH_LOCK:
        n = _SPEECH_INFLIGHT.get(target_id, 0) - 1
        if n > 0:
            _SPEECH_INFLIGHT[target_id] = n
        else:
            _SPEECH_INFLIGHT.pop(target_id, None)
        # idle counts from request END — a minutes-long talker/vocoder tail must not look idle
        _SPEECH_LAST_USE[target_id] = time.monotonic()


def _speech_evict_idle() -> list:
    """One reaper pass: drop every cached speech head idle past the window; returns the evicted
    target_ids. Skips: window off, ANY encode in flight (ENCODING > 0 covers the build and the
    talker/vocoder phases — coarse but safe), and per-target pinned requests. Eviction is
    .pop() IN PLACE: _SPEECH_MAT is from-imported by server.py/routes_diag, so a rebind would
    silently decouple those readers (state.py SAFETY NOTE)."""
    window = _speech_idle_window_s()
    if window <= 0 or ENCODING > 0 or not _SPEECH_CACHE:
        return []
    now = time.monotonic()
    evicted = []
    with _SPEECH_LOCK:
        for tid in list(_SPEECH_CACHE.keys()):
            if _SPEECH_INFLIGHT.get(tid, 0) > 0:
                continue
            if now - _SPEECH_LAST_USE.get(tid, now) <= window:
                continue
            _SPEECH_CACHE.pop(tid, None)
            _SPEECH_MAT.pop(tid, None)
            _SPEECH_LAST_USE.pop(tid, None)
            evicted.append(tid)
    if evicted:
        gc.collect()                   # the multi-GB fp32 head: hand the pages back NOW
        if SPEECH_DEVICE not in ("cpu", ""):
            with contextlib.suppress(Exception):
                import torch
                torch.cuda.empty_cache()
        for tid in evicted:
            msg = (f"speech idle-evict {tid}: no speech use for > {window / 60.0:g} min — "
                   f"dropped the cached talker+token2wav+embed head from controller RAM "
                   f"(lazy-rebuilds on the next speech request)")
            try:
                log_activity(msg)      # injected by state.bind(); print-fallback standalone
            except NameError:
                print(f"[speech] {msg}")
    return evicted


def _ensure_speech_reaper() -> None:
    """Start the single daemon evict thread lazily on first cache insert. A thread, not a hook
    into server.py's _idle_unload_loop: that sweep walks engine.models (worker shards) and lives
    in a file this leaf must not reach back into; a private thread keeps the whole feature in
    ENCODING's canonical home and needs no event loop. Called under _SPEECH_BUILD_LOCK, so the
    one-shot flag needs no extra guard."""
    global _SPEECH_REAPER_ON
    if _SPEECH_REAPER_ON:
        return
    _SPEECH_REAPER_ON = True

    def _loop():
        while True:
            time.sleep(60.0)
            with contextlib.suppress(Exception):   # the reaper must never die
                _speech_evict_idle()
    threading.Thread(target=_loop, name="speech-idle-evict", daemon=True).start()


def _ensure_spk_dict(target_id: str) -> str:
    """spk_dict.pt (speaker conditioning) is NOT a *.safetensors/*.json, so _controller_model_dir
    never fetched it. Make sure it's in the model dir; download the single file if missing."""
    local = os.path.join(MODELS_DIR, _safe_name(target_id), "spk_dict.pt")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    src = None
    with contextlib.suppress(Exception):
        src = hf_hub_download(target_id, "spk_dict.pt", local_files_only=True)
    if src is None:
        src = hf_hub_download(target_id, "spk_dict.pt", token=HF_TOKEN)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    shutil.copy2(os.path.realpath(src), local)
    return local


def _materialize_from_prefix(model, module, prefix: str, files: list, dev: str, target_id: str,
                             tag: str):
    """Load `module`'s weights from the safetensors keys under `prefix` (stripped), give any
    leftover computed meta buffers real storage, and move it to `dev`. Returns the count."""
    from safetensors import safe_open
    sd = {}
    for fn in files:
        with safe_open(fn, framework="pt") as fh:
            for k in fh.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = fh.get_tensor(k)
    if not sd:
        raise RuntimeError(f"no '{prefix}*' weights found for {tag}")
    module.load_state_dict(sd, strict=False, assign=True)
    mat = _materialize_meta_tensors(module, dev)
    _SPEECH_MAT.setdefault(target_id, {})[tag] = mat
    module.to(dev)
    return len(sd), mat


SPEECH_DEVICE = os.environ.get("INFINITEMODEL_SPEECH_DEVICE", "cpu").strip().lower()


def _load_speech_components(target_id: str) -> dict:
    """Build the talker + token2wav + thinker embed matrix needed to turn captured thinker
    hidden states into a waveform. Cached.

    IMPORTANT: build talker + token2wav on a REAL device (NOT meta). Their __init__ computes
    non-persistent buffers that are NOT in the safetensors — the DiT rotary inv_freq AND the
    BigVGAN kaiser-sinc resample FILTERS. A meta-build + generic _materialize_meta_tensors
    ZERO-FILLED those filter buffers -> the vocoder's resampling convolutions output zero ->
    SILENT audio. Building real (then load_state_dict the persistent params) keeps the filters
    correct. The talker + vocoder are small (~0.5B + DiT); only the 7B thinker needs to stay
    distributed (we only pull its embed matrix for the assembly)."""
    cached = _SPEECH_CACHE.get(target_id)
    if cached is not None:
        _speech_touch(target_id)   # #speech-idle-evict: every hit restarts the idle clock
        return cached
    import torch, glob
    import torch.nn as nn
    global ENCODING
    # #speech-idle-evict: serialize the minutes-long multi-GB build. After an eviction, two
    # concurrent cold speech requests would otherwise BOTH build (a 2x controller-RAM spike);
    # the lock loser re-checks the cache and returns the winner's head. Blocking is safe —
    # every caller reaches here via asyncio.to_thread, never on the event loop.
    _SPEECH_BUILD_LOCK.acquire()
    cached = _SPEECH_CACHE.get(target_id)
    if cached is not None:
        _SPEECH_BUILD_LOCK.release()
        _speech_touch(target_id)
        return cached
    ENCODING += 1   # heavy one-time build; hold the self-update idle gate
    try:
        from transformers import AutoConfig
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniToken2WavModel)
        from safetensors import safe_open
        t0 = time.time()
        _vlog(f"[speech] load START {target_id}")
        cfg = AutoConfig.from_pretrained(target_id)
        dev = "cpu" if SPEECH_DEVICE in ("cpu", "") else SPEECH_DEVICE
        # build REAL (buffers computed correctly), default fp32 on CPU
        talker = Qwen2_5OmniTalkerForConditionalGeneration(cfg.talker_config).eval()
        token2wav = Qwen2_5OmniToken2WavModel(cfg.token2wav_config).eval()
        _vlog(f"[speech] built talker+token2wav (real) in {time.time()-t0:.1f}s")
        model_dir = _controller_model_dir(target_id)
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

        def _load_into(module, prefix, tag):
            sd = {}
            for fn in files:
                with safe_open(fn, framework="pt") as fh:
                    for k in fh.keys():
                        if k.startswith(prefix):
                            sd[k[len(prefix):]] = fh.get_tensor(k)
            if not sd:
                raise RuntimeError(f"no '{prefix}*' weights found for {tag}")
            r = module.load_state_dict(sd, strict=False)   # copy into real params; buffers kept
            _SPEECH_MAT.setdefault(target_id, {})[tag] = {
                "loaded": len(sd), "n_missing": len(r.missing_keys),
                "missing": list(r.missing_keys)[:20],
                "n_unexpected": len(r.unexpected_keys),
                "unexpected": list(r.unexpected_keys)[:20]}
            if r.missing_keys:
                print(f"[speech] WARN {tag}: {len(r.missing_keys)} missing persistent keys "
                      f"(e.g. {list(r.missing_keys)[:3]})")
            return len(sd)
        nt = _load_into(talker, "talker.", "talker")
        nw = _load_into(token2wav, "token2wav.", "token2wav")
        talker = talker.to(dev).float()        # fp32 on CPU (bf16 CPU ops unreliable)
        token2wav = token2wav.to(dev).float()  # token2wav MUST be fp32
        # thinker embed matrix (standalone nn.Embedding) for the talker-input assembly
        ew = None
        for fn in files:
            with safe_open(fn, framework="pt") as fh:
                if "thinker.model.embed_tokens.weight" in fh.keys():
                    ew = fh.get_tensor("thinker.model.embed_tokens.weight")
                    break
        if ew is None:
            raise RuntimeError("thinker.model.embed_tokens.weight not found")
        # #speech-idle-evict (audit #27b): store the embed matrix bf16, NOT fp32 — the checkpoint
        # is bf16 on disk and the SOLE consumer (generate_speech's emb()) casts every lookup to
        # f32 anyway, so this halves the ~2.2 GB copy losslessly. A lookup is a dtype-safe gather
        # even on CPU; do NOT extend bf16 to talker/token2wav — those COMPUTE on CPU and must
        # stay fp32 (unreliable bf16 CPU ops; the zero-filled-filter/silent-audio history above).
        embed = nn.Embedding(ew.shape[0], ew.shape[1], dtype=torch.bfloat16)
        with torch.no_grad():
            embed.weight.copy_(ew.to(torch.bfloat16))
        embed = embed.to(dev).eval()
        spk_path = _ensure_spk_dict(target_id)
        speaker_map = torch.load(spk_path, weights_only=True)
        res = {"talker": talker, "token2wav": token2wav, "embed": embed,
               "speaker_map": speaker_map, "dev": dev,
               "n_talker": nt, "n_token2wav": nw, "n_embed": 1}
        _SPEECH_CACHE[target_id] = res
        _speech_touch(target_id)       # #speech-idle-evict: fresh build = fresh idle clock
        _ensure_speech_reaper()        # lazy: the evict thread exists only once a head is cached
        _vlog(f"[speech] READY {target_id}: talker={nt} token2wav={nw} embed={list(ew.shape)} on "
              f"{dev}; speakers={list(speaker_map.keys())}; total {time.time()-t0:.1f}s")
        return res
    finally:
        ENCODING -= 1
        _SPEECH_BUILD_LOCK.release()

class EngineSpeechMixin:
    # Relocated Engine method (code-split Inc 11): BODY BYTE-IDENTICAL to the server.py
    # original. Composed into ``class Engine(...)`` in server.py; ``self.*`` resolves across
    # mixins by MRO (capture_thinker lives in EngineGenMixin). Its ``global ENCODING`` rebinds
    # THIS module's canonical idle-gate counter (the whole point of the move).
    async def generate_speech(self, friendly, prompt_ids, max_new=256, speaker="Chelsie",
                              talker_max_new=2048):
        """#speech-idle-evict pin wrapper around the (unchanged) Phase-3 body below: hold the
        per-target in-flight count for the WHOLE request. ENCODING covers the component build
        and the talker/vocoder tail, but sits at 0 during the distributed thinker capture —
        without the pin the reaper could evict the head mid-request there (memory-safe, the
        local refs keep the tensors alive, but the next request would pay a pointless
        minutes-long rebuild). Unpin also restamps last-use, so idle counts from request END."""
        target = self.models[friendly].target_id
        _speech_pin(target)
        try:
            return await self._generate_speech_inner(
                friendly, prompt_ids, max_new=max_new, speaker=speaker,
                talker_max_new=talker_max_new)
        finally:
            _speech_unpin(target)

    async def _generate_speech_inner(self, friendly, prompt_ids, max_new=256, speaker="Chelsie",
                                     talker_max_new=2048):
        """#P6 speech-out Phase 3: distributed Thinker (captured hidden states) -> Talker (codec
        tokens) -> token2wav (waveform). Faithful to Qwen2.5-Omni's generate() talker assembly
        (modeling_qwen2_5_omni.py): builds thinker_reply_part = last-layer-hidden + token-embed
        for the generated tokens, talker_inputs_embeds for the prompt, prepends the speaker text
        bos + the first reply hidden, appends eos/pad embeds, then drives the REAL HF talker +
        token2wav. Returns (gen_ids, text_stop, waveform [N] float32 @24kHz, info)."""
        import torch
        model = self.models[friendly]
        target = model.target_id
        sc = await asyncio.to_thread(_load_speech_components, target)
        talker, token2wav, embed = sc["talker"], sc["token2wav"], sc["embed"]
        dev, speaker_map = sc["dev"], sc["speaker_map"]
        if speaker not in speaker_map:
            raise RuntimeError(f"speaker '{speaker}' not in {list(speaker_map.keys())}")
        spk = speaker_map[speaker]
        # 1) distributed thinker with hidden-state capture
        gen_ids, prefill_hidden, step_hiddens, stop = await self.capture_thinker(
            friendly, prompt_ids, max_new)
        # #idle-unload: restart the idle clock before the (potentially minutes-long, CPU-bound)
        # talker+vocoder tail — capture_thinker stamped per step, this covers the assembly phase.
        model.last_token_ts = time.time()
        info = {"prompt_len": len(prompt_ids), "gen_tokens": len(gen_ids),
                "captured_steps": len(step_hiddens), "text_stop": stop}

        def _assemble_and_run():
            f32 = torch.float32
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

            def emb(ids_t):
                return embed(ids_t.to(dev)).to(f32)

            # thinker_hidden_states (last-layer): [prefill_all, *per-fed-token]
            hs = [prefill_hidden.to(device=dev, dtype=f32)]
            hs += [h.to(device=dev, dtype=f32) for h in step_hiddens]
            # thinker_token_embeds (layer-0 = input embeddings); text-only -> no mm zeroing.
            # The fed tokens are exactly the ones we captured a hidden for (gen_ids[:len(step)]).
            fed = gen_ids[:len(step_hiddens)]
            te = [emb(input_ids)]
            for t in fed:
                te.append(emb(torch.tensor([[t]], dtype=torch.long)))
            bos_id = int(spk["bos_token"])
            gen_t = torch.tensor([gen_ids], dtype=torch.long, device=dev)
            talker_input_text_ids = torch.cat([
                input_ids,
                torch.tensor([[bos_id]], dtype=torch.long, device=dev),
                gen_t[:, :1]], dim=-1)
            talker_input_ids = torch.cat([
                torch.full_like(input_ids, fill_value=talker.codec_mask_token),
                torch.tensor([[talker.codec_pad_token]], dtype=torch.long, device=dev),
                torch.tensor([[talker.codec_bos_token]], dtype=torch.long, device=dev)], dim=1)
            reply = torch.cat(hs[1:], dim=1) + torch.cat(te[1:], dim=1)   # [1,F,H] generated
            talker_inputs_embeds = hs[0] + te[0]                          # [1,P,H] prompt
            bos_embed = emb(torch.tensor([[bos_id]], dtype=torch.long))
            talker_inputs_embeds = torch.cat(
                [talker_inputs_embeds, bos_embed, reply[:, :1, :]], dim=1)
            eos_embed = emb(torch.tensor([[talker.text_eos_token]], dtype=torch.long))
            pad_embed = emb(torch.tensor([[talker.text_pad_token]], dtype=torch.long))
            reply = torch.cat([reply[:, 1:, :], eos_embed, pad_embed], dim=1)
            with torch.inference_mode():
                # Talker generation params = Qwen2.5-Omni's generate() defaults (the talker is
                # TUNED for sampling; greedy degenerates -> noise + never stops). eos is BOTH
                # codec_pad (8292) and codec_eos (8294).
                talker_result = talker.generate(
                    input_ids=talker_input_ids,
                    input_text_ids=talker_input_text_ids,
                    thinker_reply_part=reply,
                    inputs_embeds=talker_inputs_embeds,
                    suppress_tokens=[talker.codec_bos_token],
                    do_sample=True, top_k=40, top_p=0.8, temperature=0.9,
                    repetition_penalty=1.05,
                    eos_token_id=[talker.codec_pad_token, talker.codec_eos_token],
                    max_new_tokens=int(talker_max_new))
                codes = talker_result[:, talker_input_ids.shape[1]:-1]
                info["codec_tokens"] = int(codes.shape[-1])
                info["talker_hit_cap"] = bool(talker_result.shape[1] - talker_input_ids.shape[1]
                                              >= int(talker_max_new))
                with contextlib.suppress(Exception):
                    info["codes_min"] = int(codes.min()); info["codes_max"] = int(codes.max())
                wav = token2wav(codes.to(dev),
                                conditioning=spk["cond"].to(dev).float(),
                                reference_mel=spk["ref_mel"].to(dev).float())
            return wav.float().reshape(-1)

        global ENCODING
        ENCODING += 1   # hold the self-update idle gate during the talker/vocoder run
        try:
            wav = await asyncio.to_thread(_assemble_and_run)
        finally:
            ENCODING -= 1
        info["wav_samples"] = int(wav.shape[0])
        info["wav_seconds"] = round(int(wav.shape[0]) / 24000.0, 2)
        return gen_ids, stop, wav, info
