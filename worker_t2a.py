"""worker_t2a: the worker-side text-to-AUDIO (music) engine (#t2a-serve, M1).

The audio-generation sibling of worker_t2i: serves an ACE-Step v1 3.5B checkpoint
(diffusers-style component layout: ace_step_transformer// music_dcae_f8c8//
music_vocoder// umt5-base/ subfolders) as a single-node music generator. The WHOLE
pipeline lives on ONE controller-CO-LOCATED worker; requests arrive over the control
link (`t2a_gen`), per-step progress mirrors back (`t2a_step`), and the finished WAV
is written to LOCAL disk with its path returned (`t2a_done`) — v1, like t2i, serves
only on a worker sharing the controller's filesystem (model dir read + result write
need no transfer).

Unlike worker_t2i (which hand-drives a diffusers denoise loop over two pipeline
views), this WRAPS ACE-Step's own `ACEStepPipeline`: its diffusion loop carries
lyric conditioning, APG guidance, and the DCAE->vocoder two-stage decode that would
be error-prone to re-implement. iM layers on top: a soundfile save (torchaudio 2.11
routes .save through torchcodec, absent here), an optional forward-hook for per-step
progress, and the same GPU-VRAM release discipline as T2IPipeline.

DEPENDS ON the `acestep` package (pip) + `soundfile` — heavy imports live inside
methods so importing this module costs nothing.

Worker-side leaf: imported lazily by worker_load's t2a branch (fetch-if-missing via
worker_update._fetch_repo_file); in client.py's worker update file list +
server.py's EXTRA_UPDATE_FILES.

M0 findings baked in (see memory acestep-t2music-plan): ACE-Step's cpu_offload moves
the WHOLE ~6.6 GB DiT to the GPU for the diffusion call (not per-block streaming), so
offload here means "components rest on CPU, whole-DiT hops to GPU per generate" — it
needs ~7 GB free VRAM transiently but leaves nothing resident. quant is bf16-only for
M1 (edge-int4 is M2).
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

GB = 1024 ** 3


def _install_soundfile_save() -> None:
    """torchaudio 2.11 delegates .save to torchcodec (not installed); ACE-Step's
    save_wav_file calls torchaudio.save(..., backend="soundfile"). Route save through
    soundfile directly. Idempotent."""
    import torchaudio
    if getattr(torchaudio.save, "_im_soundfile", False):
        return
    import soundfile as sf

    def _save(path, tensor, sample_rate, **_kw):
        arr = tensor.detach().cpu().float().numpy()
        if arr.ndim == 2:          # torchaudio (channels, samples) -> soundfile (samples, channels)
            arr = arr.T
        sf.write(path, arr, int(sample_rate))
    _save._im_soundfile = True
    torchaudio.save = _save


def _quantize_dit_int4(dit) -> None:
    """#t2a-int4 (M2): in-place group-wise int4 of the ACE-Step DiT's big weight layers, reusing
    worker_quant's QuantLinear4. Covers BOTH the attention/embedder nn.Linear AND the 1x1 nn.Conv1d
    that GLUMBConv's FF is built from — a 1x1 conv is a pointwise Linear over channels, and the FF
    is the MAJORITY of each block (mlp_ratio 4), so quantizing only nn.Linear would leave most of
    the DiT bf16 and miss the fit. Measured ~3.75x on the FF / ~3.7x on Linear -> DiT ~6.6 GB down
    to ~2 GB, which fits a bf16-capable 6 GB card (RTX 3060) with headroom. Left bf16 (tiny or not
    a pointwise matmul): the depthwise conv (k=3, groups=C), the Conv2d patch-embed, RMSNorm,
    embeddings/rotary. prepare_fused is deliberately NOT called: ACE-Step's cpu_offload hops
    modules CPU<->GPU per render, and QuantLinear4's naive dequant path (forward with _fused unset)
    is device-agnostic; the int4-packed weights still cut resident + per-render transfer VRAM ~4x.
    (int4 needs a bf16-capable card anyway — the caller/placement gates t2a on compute cap >= (8,0),
    see #t2a-bf16-gate — so this tier never reaches a Pascal card.)"""
    import types
    from torch import nn
    import worker_quant as wq

    class _Conv1x1Int4(nn.Module):
        """A 1x1 Conv1d (pointwise, groups=1) served as int4 over channels: [B,C,L] -> [B,L,C] ->
        QuantLinear4 -> [B,L,C'] -> [B,C',L]. Identical I/O contract to the conv, so ConvLayer's
        norm/act and GLUMBConv's downstream gating are untouched."""
        def __init__(self, q):
            super().__init__()
            self.q = q

        def forward(self, x):
            return self.q(x.transpose(1, 2)).transpose(1, 2)

    def _conv_to_int4(conv):
        W = conv.weight.data.squeeze(-1).contiguous()          # [out, in, 1] -> [out, in]
        lin = types.SimpleNamespace(weight=types.SimpleNamespace(data=W), bias=conv.bias)
        return _Conv1x1Int4(wq._quantize_linear4(lin))

    def _walk_convs(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv1d) and tuple(child.kernel_size) == (1,) \
                    and child.groups == 1:
                setattr(module, name, _conv_to_int4(child))
            else:
                _walk_convs(child)

    wq._quantize_int4_(dit)   # every nn.Linear -> QuantLinear4 (DiT has no MoE router to skip)
    _walk_convs(dit)          # every 1x1 Conv1d -> int4 pointwise


class T2APipeline:
    """One resident text-to-audio (ACE-Step) model on this worker. Stored in
    worker.shards[model_id] like a T2IPipeline / EmbeddingModel; `kind` lets dispatchers
    tell it apart. One generate at a time per model (_gen_lock) — the controller also
    serializes on LoadedModel.lock, this is the worker-side belt."""

    kind = "t2a"

    def __init__(self, model_dir: str, device: str, quant: str = "none",
                 offload: bool = False):
        try:
            from acestep.pipeline_ace_step import ACEStepPipeline
        except Exception as exc:
            raise RuntimeError(
                "t2a serving needs the `acestep` package on this worker "
                f"(pip install acestep) — import failed: {exc!r}") from exc
        import torch

        self.model_dir = model_dir
        _dv = str(device or "")
        if not _dv or "gpu" in _dv:
            _dv = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = _dv
        self.quant = "int4" if str(quant).lower() == "int4" else "none"   # M2: edge-int4 DiT; else bf16
        self.offload = bool(offload)
        self._gen_lock = threading.Lock()
        self._doomed = False
        t0 = time.time()

        _install_soundfile_save()

        # ACE-Step drives its own device placement: cpu_offload=True keeps components on
        # CPU and hops the whole DiT to the GPU per generate() (low resident VRAM, ~7 GB
        # transient); cpu_offload=False keeps the whole pipeline GPU-resident (faster, no
        # per-call move). torch_compile off (parity + no first-run compile stall).
        self.pipe = ACEStepPipeline(
            checkpoint_dir=model_dir,
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=self.offload,
            overlapped_decode=False,
        )
        # #t2a-cpu: ACE-Step has NO cpu-force flag — __init__ sets self.device = cuda:0
        # whenever torch.cuda.is_available(), and load_checkpoint() moves EVERY component
        # (ace_step_transformer / music_dcae / text_encoder_model) with .to(self.device).
        # So on a GPU box it ignores our device='cpu' and loads onto the GPU (OOMing if the
        # card is full). Override the pipeline's device to CPU BEFORE load_checkpoint so the
        # whole model lands in RAM. cpu_offload is already False for a cpu_only load (the
        # controller never asks for offload+cpu), so there is no GPU hop at render time.
        if str(self.device).startswith("cpu"):
            import torch as _torch
            self.pipe.device = _torch.device("cpu")
        # Eager-load so `loaded` reflects reality and the first request isn't a cold load;
        # get_checkpoint_path uses model_dir as-is (has the 4 component subfolders).
        # #t2a-int4 (M2): quantize the DiT DURING load. ACE-Step's load_checkpoint loads the DiT
        # FIRST (ACEStepTransformer2DModel.from_pretrained) and the small DCAE/UMT5 after, so the
        # bf16 DiT (~6.6 GB) is the RAM peak of the WHOLE load. Patching from_pretrained to int4 the
        # DiT before it returns caps the peak at the DiT's own bf16 size instead of the whole bf16
        # pipeline (~7.7 GB) — that ~1 GB is what lets the int4 load fit a ~7 GB-free box (a 3060
        # laptop also running a desktop). The `.to(self.dtype)` load_checkpoint runs next is safe:
        # nn.Module.to(dtype) casts only floating tensors, leaving QuantLinear4's uint8 qweight
        # buffers untouched. `loaded_params` is captured pre-quant here (weights become buffers).
        self.loaded_params = 0
        _int4_done = {"v": False}
        if self.quant == "int4":
            from acestep.models.ace_step_transformer import ACEStepTransformer2DModel as _DiTCls
            _orig_fp = _DiTCls.from_pretrained.__func__

            def _fp_int4(cls, *a, **k):
                _m = _orig_fp(cls, *a, **k)
                self.loaded_params = sum(p.numel() for p in _m.parameters())
                _tq = time.time()
                _quantize_dit_int4(_m)
                _int4_done["v"] = True
                print(f"[t2a] int4 DiT quantized during load in {time.time() - _tq:.1f}s", flush=True)
                return _m
            _DiTCls.from_pretrained = classmethod(_fp_int4)
            try:
                self.pipe.load_checkpoint(model_dir)
            finally:
                _DiTCls.from_pretrained = classmethod(_orig_fp)
        else:
            self.pipe.load_checkpoint(model_dir)

        def _module_bytes(mod) -> int:
            if mod is None:
                return 0
            return sum(p.numel() * p.element_size() for p in mod.parameters()) + \
                sum(b.numel() * b.element_size() for b in mod.buffers())

        dit = getattr(self.pipe, "ace_step_transformer", None)
        dcae = getattr(self.pipe, "music_dcae", None)
        te = getattr(self.pipe, "text_encoder_model", None)
        # Safety net: if the load-time patch never fired (e.g. from_pretrained signature changed),
        # int4 the DiT post-hoc so quant is still correct — it just pays the higher bf16 RAM peak.
        if self.quant == "int4" and dit is not None and not _int4_done["v"]:
            self.loaded_params = sum(p.numel() for p in dit.parameters())
            _quantize_dit_int4(dit)
            print("[t2a] int4 DiT quantized post-load (from_pretrained patch did not fire)", flush=True)
        elif self.quant != "int4":
            self.loaded_params = sum(p.numel() for p in dit.parameters()) if dit is not None else 0
        dit_b = _module_bytes(dit)
        self.gpu_bytes = 0 if self.offload else (
            (dit_b + _module_bytes(dcae)) if str(self.device).startswith("cuda") else 0)
        self.loaded_bytes = dit_b + _module_bytes(dcae) + _module_bytes(te)
        self.last_gen_s = 0.0
        print(f"[t2a] ready on {self.device}"
              f"{' OFFLOAD (components in RAM, whole DiT hops per gen)' if self.offload else ''} "
              f"in {time.time() - t0:.0f}s "
              f"(GPU {self.gpu_bytes / GB:.1f} GB, total {self.loaded_bytes / GB:.1f} GB)", flush=True)

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """Free this pipeline's GPU tensor STORAGES in place on unload. RENDER-SAFE: a live
        generate() still computes on these tensors, so under a held _gen_lock we only mark
        _doomed and the render's own finally frees when it completes (mirrors T2IPipeline)."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[t2a] unload during a live render — VRAM release deferred to render end",
                  flush=True)
            return
        self._free_now()

    def _free_now(self) -> None:
        # #t2a-rss-leak: OFFLOAD keeps the DiT/DCAE/text-encoder in CPU RAM (~2.7 GB int4) and hops
        # only the DiT to the GPU per render, so at unload the BULK of this pipeline's bytes are
        # CPU-resident, not CUDA. The old pass emptied ONLY cuda tensors, so _unload_model's
        # malloc_trim (which DOES run on Linux) found nothing free to hand back — the CPU storages
        # were still alive, kept by any lingering ref until the pipeline object itself was GC'd. And
        # the DEFERRED post-render free (generate()'s finally) never re-enters _unload_model, so it
        # got no trim at all. Net effect: the worker's RSS climbed ~2-3 GB per load/unload cycle
        # (measured to 4.9 GB) and starved the box until a restart. Fix: empty BOTH cuda and cpu
        # storages IN PLACE (releases the bytes regardless of who still holds the module — the same
        # lesson as _release_shard_vram), then trim the glibc arena HERE so the RAM returns to the OS
        # on this path too (covering the deferred free). Render-safe: release_vram() defers to render
        # end while a gen holds _gen_lock, so by the time we run the model is idle.
        import contextlib as _cl
        import torch
        seen: set = set()
        n_cuda = n_cpu = 0
        for mod in (getattr(self.pipe, "ace_step_transformer", None),
                    getattr(self.pipe, "music_dcae", None),
                    getattr(self.pipe, "text_encoder_model", None)):
            if mod is None or not hasattr(mod, "parameters"):
                continue
            with _cl.suppress(Exception):
                for t in list(mod.parameters(recurse=True)) + list(mod.buffers(recurse=True)):
                    if t is None:
                        continue
                    dev = getattr(t, "device", None)
                    if dev is None or dev.type not in ("cuda", "cpu"):
                        continue   # skip meta/other; only real storages have bytes to release
                    if id(t) in seen:
                        continue
                    seen.add(id(t))
                    if dev.type == "cuda":
                        n_cuda += 1
                    else:
                        n_cpu += 1
                    with _cl.suppress(Exception):
                        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        with _cl.suppress(Exception):
            if str(self.device).startswith("cuda"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        # Return the freed CPU heap to the OS (glibc holds freed arenas until malloc_trim). Self-
        # contained so the DEFERRED post-render free trims too — that path never reaches
        # _unload_model's _release_ram. malloc_trim(0) only reclaims FREE arena, never in-use
        # allocations, so it can't disturb another model still resident in this worker.
        with _cl.suppress(Exception):
            import gc as _gc
            _gc.collect()
            import ctypes
            import sys as _sys
            if _sys.platform != "win32":
                ctypes.CDLL("libc.so.6").malloc_trim(0)
        print(f"[t2a] {os.path.basename(self.model_dir)}: storages released "
              f"({n_cuda} cuda + {n_cpu} cpu tensors emptied, heap trimmed)", flush=True)

    # -- generation ---------------------------------------------------------------------

    def generate(self, prompt: str, lyrics: str, duration: float, steps: int,
                 guidance: float, seed, on_step=None) -> tuple[str, float]:
        """Render one music clip; returns (wav_path, seconds). Runs in a worker thread
        (asyncio.to_thread) — one at a time per model via _gen_lock."""
        try:
            return self._generate(prompt, lyrics, duration, steps, guidance, seed, on_step)
        finally:
            if self._doomed:
                print("[t2a] deferred VRAM release: freeing the unloaded pipeline post-render",
                      flush=True)
                self._free_now()

    def _generate(self, prompt: str, lyrics: str, duration: float, steps: int,
                  guidance: float, seed, on_step=None) -> tuple[str, float]:
        import contextlib
        with self._gen_lock:
            t0 = time.time()
            duration = max(3.0, min(240.0, float(duration)))
            steps = max(1, min(200, int(steps)))
            guidance = float(guidance)
            if seed in (None, ""):
                seed = int.from_bytes(os.urandom(4), "big")
            path = os.path.join(tempfile.gettempdir(),
                                f"im_t2a_{os.getpid()}_{int(time.time() * 1000)}.wav")

            # Per-step progress: ACE-Step's diffusion loop exposes no callback, so hook the
            # DiT forward. With CFG (guidance>1) it runs ~2 forwards/step; scale the total so
            # progress tracks ~monotonically. Best-effort; detached in finally.
            hook = None
            dit = getattr(self.pipe, "ace_step_transformer", None)
            if on_step is not None and dit is not None:
                per_step = 2 if guidance > 1.0 else 1
                total_fwd = steps * per_step
                state = {"n": 0}

                def _pre_hook(_m, _inp):
                    state["n"] += 1
                    with contextlib.suppress(Exception):
                        on_step(min(steps, (state["n"] + per_step - 1) // per_step), steps)
                with contextlib.suppress(Exception):
                    hook = dit.register_forward_pre_hook(_pre_hook)

            # #gpu-share: render on a SEPARATE side stream (same fix as worker_t2i): on the
            # shared default stream a music render's minutes-deep kernel backlog starves a
            # co-resident LLM decode to ~0 tok/s; on a side stream the hardware interleaves
            # both queues at kernel boundaries. CPU device / failure -> old inline behavior.
            _rs = None
            with contextlib.suppress(Exception):
                if str(self.device).startswith("cuda"):
                    import torch
                    _rs = torch.cuda.Stream(device=self.device)
            if _rs is not None:
                import torch
                _sctx = torch.cuda.stream(_rs)
            else:
                _sctx = contextlib.nullcontext()
            try:
                with _sctx:
                    self.pipe(
                        prompt=prompt or "",
                        lyrics=lyrics or "",
                        audio_duration=duration,
                        infer_step=steps,
                        guidance_scale=guidance,
                        manual_seeds=[int(seed)],
                        save_path=path,
                        format="wav",
                        batch_size=1,
                    )
                if _rs is not None:   # drain before the caller reads the WAV off disk
                    with contextlib.suppress(Exception):
                        _rs.synchronize()
            finally:
                if hook is not None:
                    with contextlib.suppress(Exception):
                        hook.remove()

            self.last_gen_s = time.time() - t0
            print(f"[t2a] {duration:.0f}s audio steps={steps} guidance={guidance} seed={seed} "
                  f"-> {path} ({self.last_gen_s:.0f}s)", flush=True)
            return path, self.last_gen_s
