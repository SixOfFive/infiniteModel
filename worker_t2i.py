"""worker_t2i: the worker-side text-to-image engine (#t2i-serve, task #37).

Serves a DIFFUSERS-layout checkpoint (Qwen-Image class: model_index.json +
transformer//text_encoder//vae//tokenizer/ component subfolders) as a single-node
image generator, the diffusion sibling of worker_load's `kind:"embedding"` path:
the WHOLE pipeline lives on ONE worker, requests arrive over the control link
(`t2i_gen`), per-step progress mirrors back (`t2i_step`), and the finished PNG is
written to LOCAL disk with its path returned (`t2i_done`) — v1 serves t2i only on
a worker CO-LOCATED with the controller (same box, shared filesystem), so both the
model dir (read) and the result file (write) need no transfer at all.

Quant recipe (gate-tested 2026-07-10, om3nbox gfx1151, A/B vs bf16 at fixed seeds):
the DiT's middle blocks are quantized with the fleet's OWN RTN int4 g128 packer
(worker_quant._quantize_int4_, prepare_fused -> tinygemm on NVIDIA / Triton w4a16
on ROCm) while the FIRST `edge` and LAST `edge` transformer blocks stay bf16 —
pure int4 renders coherent images but drifts text glyphs ("$4.50"->"$6.50",
"WiFi"->"WiPi") and adds grain; protecting 2+2 edge blocks (~+2 GB) restored
exact text and the bf16 color grade. The text encoder (Qwen2.5-VL-7B) runs on CPU
in bf16 — it encodes ONCE per request (a ≤1024-token prefill, seconds) and would
not fit beside the DiT on a 16 GB card. VAE decodes on the GPU with tiling when
available; an OOM falls back to a CPU decode of the same latents (exact, slower).

DEPENDS ON `diffusers` (pip) — the one external lib this path allows: the DiT
forward (dual-stream MMDiT, MSRoPE, AdaLN) + FlowMatchEuler scheduler are proven
here rather than hand-rolled; a native forward can replace it later exactly like
every other subsystem. A missing diffusers import fails the LOAD with a pip hint.

Worker-side leaf: imported lazily by worker_load's t2i branch (fetch-if-missing
via worker_update._fetch_repo_file); heavy imports live inside methods so merely
importing this module costs nothing. In client.py's worker update file list +
server.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

GB = 1024 ** 3


class T2IPipeline:
    """One resident text-to-image model on this worker. Stored in worker.shards[model_id]
    like a Shard / EmbeddingModel (generic unload/teardown paths just drop the reference);
    `kind` lets dispatchers tell it apart. One generate at a time per model (_gen_lock) —
    the controller serializes on LoadedModel.lock too, this is the worker-side belt."""

    kind = "t2i"

    def __init__(self, model_dir: str, device: str, quant: str = "int4", edge: int = 2):
        try:
            from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                                   QwenImagePipeline, QwenImageTransformer2DModel)
        except Exception as exc:
            raise RuntimeError(
                "t2i serving needs the `diffusers` package on this worker "
                f"(pip install diffusers) — import failed: {exc!r}") from exc
        import torch
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        self.model_dir = model_dir
        # normalize fleet tier strings ('gpu', 'cpu+gpu') to a real torch device — torch.to()
        # rejects them; anything GPU-flavored means "use the GPU here"
        _dv = str(device or "")
        if not _dv or "gpu" in _dv:
            _dv = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = _dv
        self.quant = quant
        self.edge = max(0, int(edge))
        self._gen_lock = threading.Lock()
        self._doomed = False   # #t2i-vram-release: unload arrived mid-render -> free after it ends
        t0 = time.time()

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(model_dir, "scheduler"))
        transformer = QwenImageTransformer2DModel.from_pretrained(
            os.path.join(model_dir, "transformer"), torch_dtype=torch.bfloat16)
        vae = AutoencoderKLQwenImage.from_pretrained(
            os.path.join(model_dir, "vae"), torch_dtype=torch.bfloat16)
        # Text encoder stays on CPU bf16 (encode-once per request; won't fit beside the DiT
        # on a 16 GB card). Tokenizer from the repo's tokenizer/ subfolder.
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(model_dir, "text_encoder"), torch_dtype=torch.bfloat16)
        text_encoder.eval()
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

        # Mixed-edge int4 (the gate-test recipe): quantize the middle blocks in place with the
        # fleet packer, keep `edge` blocks at each end bf16. quant="none" skips entirely (bf16
        # serve — only sensible on big-unified-memory boxes).
        n_quant = 0
        if self.quant.startswith("int4"):
            import worker_quant
            blocks = list(transformer.transformer_blocks)
            lo, hi = self.edge, len(blocks) - self.edge
            for b in blocks[lo:hi]:
                worker_quant._quantize_int4_(b)
                n_quant += 1
        transformer.eval()

        transformer.to(self.device)
        vae.to(self.device)
        if self.quant.startswith("int4"):
            import worker_quant
            QL = worker_quant._quant4_linear_cls()
            fused = total = 0
            for m in transformer.modules():
                if isinstance(m, QL):
                    m.prepare_fused()
                    total += 1
                    fused += getattr(m, "_fused", None) is not None
            print(f"[t2i] {os.path.basename(model_dir)}: {n_quant}/{len(transformer.transformer_blocks)} "
                  f"blocks int4 (edge {self.edge} bf16), fused {fused}/{total} linears", flush=True)
        # Tiled VAE decode caps the decode's transient VRAM spike (~8 GB at 1328^2 untiled) —
        # essential on 16 GB cards sharing the device with ~13 GB of DiT weights. Best-effort.
        try:
            vae.enable_tiling()
        except Exception:
            pass

        # TWO pipeline views over shared components (the gate-harness topology, proven):
        # a single pipeline holding the CPU text encoder makes diffusers' _execution_device
        # resolve to CPU — latents/timesteps then prepare on CPU and the cuda DiT crashes with
        # 'mat1 is on cpu'. The ENCODER view (TE+tokenizer, no transformer) encodes on CPU;
        # the RENDER view (transformer+VAE, no TE) resolves cuda and runs the denoise loop.
        self.enc = QwenImagePipeline(scheduler=scheduler, vae=None, text_encoder=text_encoder,
                                     tokenizer=tokenizer, transformer=None)
        self.pipe = QwenImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None,
                                      tokenizer=None, transformer=transformer)

        def _module_bytes(mod) -> int:
            return sum(p.numel() * p.element_size() for p in mod.parameters()) + \
                sum(b.numel() * b.element_size() for b in mod.buffers())

        dit_b = _module_bytes(transformer)
        self.gpu_bytes = (dit_b + _module_bytes(vae)) if str(self.device).startswith("cuda") else 0
        self.loaded_bytes = dit_b + _module_bytes(vae) + _module_bytes(text_encoder)
        self.loaded_params = sum(p.numel() for p in transformer.parameters()) \
            + sum(p.numel() for p in text_encoder.parameters())
        self.last_gen_s = 0.0
        print(f"[t2i] ready on {self.device} in {time.time() - t0:.0f}s "
              f"(GPU {self.gpu_bytes / GB:.1f} GB, total {self.loaded_bytes / GB:.1f} GB)", flush=True)

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """#t2i-vram-release: free this pipeline's GPU tensor STORAGES IN PLACE on unload. The
        generic _release_shard_vram walk frees nothing here (a T2IPipeline has no model/embed/
        head attrs), and on ROCm a dropped ref + empty_cache does NOT return the DiT's ~12 GB
        (same lingering-ref behavior the shard path already works around) — observed pinned live.
        RENDER-SAFE: a live generate() still computes on these tensors (an unload-all can arrive
        mid-render — observed at step 5/20), so under a held _gen_lock we only mark _doomed and
        the render's own finally frees the moment it completes."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[t2i] unload during a live render — VRAM release deferred to render end",
                  flush=True)
            return
        self._free_now()

    def _free_now(self) -> None:
        import contextlib as _cl
        import torch
        seen: set = set()
        for mod in (getattr(self.pipe, "transformer", None), getattr(self.pipe, "vae", None)):
            if mod is None or not hasattr(mod, "parameters"):
                continue
            with _cl.suppress(Exception):
                for sub in mod.modules():   # drop fused tuples first (they alias qweight)
                    if getattr(sub, "_fused", None) is not None:
                        sub._fused = None
                    for _b in ("qweight", "scale", "zero"):
                        t = getattr(sub, _b, None)
                        if t is not None and getattr(t, "device", None) is not None \
                                and t.device.type == "cuda" and id(t) not in seen:
                            seen.add(id(t))
                            with _cl.suppress(Exception):
                                setattr(sub, _b, torch.empty(0, dtype=t.dtype, device=t.device))
                for t in list(mod.parameters(recurse=True)) + list(mod.buffers(recurse=True)):
                    if t is None or getattr(t, "device", None) is None or t.device.type != "cuda":
                        continue
                    if id(t) in seen:
                        continue
                    seen.add(id(t))
                    with _cl.suppress(Exception):   # #39: empty a padded view's BASE too
                        b = getattr(t, "_base", None)
                        if b is not None and getattr(b, "device", None) is not None \
                                and b.device.type == "cuda" and id(b) not in seen:
                            seen.add(id(b))
                            b.data = torch.empty(0, dtype=b.dtype, device=b.device)
                    with _cl.suppress(Exception):
                        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        with _cl.suppress(Exception):
            if str(self.device).startswith("cuda"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        print(f"[t2i] {os.path.basename(self.model_dir)}: GPU storages released "
              f"({len(seen)} tensors emptied)", flush=True)

    # -- generation ---------------------------------------------------------------------

    def generate(self, prompt: str, negative_prompt: str, width: int, height: int,
                 steps: int, cfg: float, seed, on_step=None) -> tuple[str, float]:
        """Render one image; returns (png_path, seconds). Runs in a worker thread
        (asyncio.to_thread) — one at a time per model via _gen_lock."""
        try:
            return self._generate(prompt, negative_prompt, width, height, steps, cfg, seed,
                                  on_step)
        finally:
            if self._doomed:   # unloaded mid-render (#t2i-vram-release) -> free now that it's over
                print("[t2i] deferred VRAM release: freeing the unloaded pipeline post-render",
                      flush=True)
                self._free_now()

    def _generate(self, prompt: str, negative_prompt: str, width: int, height: int,
                  steps: int, cfg: float, seed, on_step=None) -> tuple[str, float]:
        import torch
        with self._gen_lock:
            t0 = time.time()
            pipe = self.pipe
            dev = self.device
            # Geometry must be divisible by vae_scale_factor*2 (=16); snap silently.
            width = max(256, (int(width) // 16) * 16)
            height = max(256, (int(height) // 16) * 16)
            steps = max(1, min(100, int(steps)))
            cfg = float(cfg)
            if seed in (None, ""):
                seed = int.from_bytes(os.urandom(4), "big")
            g = torch.Generator("cpu").manual_seed(int(seed))

            with torch.no_grad():
                # Encode on CPU via the ENCODER view (the TE lives there); masks may
                # legitimately come back None (this diffusers build drops an all-ones mask).
                pe, pm = self.enc.encode_prompt(prompt=prompt, device="cpu")[:2]
                ne = nm = None
                if cfg > 1.0:
                    ne, nm = self.enc.encode_prompt(prompt=(negative_prompt or " "),
                                                    device="cpu")[:2]
                _m = lambda x: x.to(dev) if x is not None else None

                def _cb(_p, i, _t, kw):
                    if on_step is not None:
                        try:
                            on_step(i + 1, steps)
                        except Exception:
                            pass
                    return {}

                out = pipe(prompt_embeds=pe.to(dev), prompt_embeds_mask=_m(pm),
                           negative_prompt_embeds=_m(ne), negative_prompt_embeds_mask=_m(nm),
                           true_cfg_scale=cfg, num_inference_steps=steps,
                           width=width, height=height, generator=g,
                           output_type="latent", callback_on_step_end=_cb)
                latents = out.images
                img = self._decode(latents, height, width)

            path = os.path.join(tempfile.gettempdir(),
                                f"im_t2i_{os.getpid()}_{int(time.time() * 1000)}.png")
            img.save(path)
            self.last_gen_s = time.time() - t0
            print(f"[t2i] {width}x{height} steps={steps} cfg={cfg} seed={seed} "
                  f"-> {path} ({self.last_gen_s:.0f}s)", flush=True)
            return path, self.last_gen_s

    def _decode(self, latents, height: int, width: int):
        """Unpack + denormalize + VAE-decode the packed latents (the tail of the diffusers
        pipeline, inlined so a GPU OOM can fall back to an exact CPU decode of the SAME
        latents instead of failing a multi-minute render at its last step."""
        import torch
        pipe = self.pipe
        vae = pipe.vae
        lat = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        lat = lat.to(vae.dtype)
        lmean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        lstd = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        try:
            l_ = lat / lstd.to(lat.device, lat.dtype) + lmean.to(lat.device, lat.dtype)
            image = vae.decode(l_, return_dict=False)[0][:, :, 0]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print("[t2i] VAE decode OOM on GPU -> CPU decode fallback", flush=True)
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            vae.to("cpu")
            try:
                latc = lat.to("cpu")
                l_ = latc / lstd.to(latc.dtype) + lmean.to(latc.dtype)
                image = vae.decode(l_, return_dict=False)[0][:, :, 0]
            finally:
                vae.to(self.device)
        return pipe.image_processor.postprocess(image.float().cpu(), output_type="pil")[0]
