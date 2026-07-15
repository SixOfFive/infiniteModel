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
        self.quant = "none"          # M1: bf16 only (edge-int4 is M2)
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
        # Eager-load so `loaded` reflects reality and the first request isn't a cold load;
        # get_checkpoint_path uses model_dir as-is (has the 4 component subfolders).
        self.pipe.load_checkpoint(model_dir)

        def _module_bytes(mod) -> int:
            if mod is None:
                return 0
            return sum(p.numel() * p.element_size() for p in mod.parameters()) + \
                sum(b.numel() * b.element_size() for b in mod.buffers())

        dit = getattr(self.pipe, "ace_step_transformer", None)
        dcae = getattr(self.pipe, "music_dcae", None)
        te = getattr(self.pipe, "text_encoder_model", None)
        dit_b = _module_bytes(dit)
        self.gpu_bytes = 0 if self.offload else (
            (dit_b + _module_bytes(dcae)) if str(self.device).startswith("cuda") else 0)
        self.loaded_bytes = dit_b + _module_bytes(dcae) + _module_bytes(te)
        self.loaded_params = sum(p.numel() for p in dit.parameters()) if dit is not None else 0
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
        import contextlib as _cl
        import torch
        seen: set = set()
        for mod in (getattr(self.pipe, "ace_step_transformer", None),
                    getattr(self.pipe, "music_dcae", None),
                    getattr(self.pipe, "text_encoder_model", None)):
            if mod is None or not hasattr(mod, "parameters"):
                continue
            with _cl.suppress(Exception):
                for t in list(mod.parameters(recurse=True)) + list(mod.buffers(recurse=True)):
                    if t is None or getattr(t, "device", None) is None or t.device.type != "cuda":
                        continue
                    if id(t) in seen:
                        continue
                    seen.add(id(t))
                    with _cl.suppress(Exception):
                        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        with _cl.suppress(Exception):
            if str(self.device).startswith("cuda"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        print(f"[t2a] {os.path.basename(self.model_dir)}: GPU storages released "
              f"({len(seen)} tensors emptied)", flush=True)

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

            try:
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
            finally:
                if hook is not None:
                    with contextlib.suppress(Exception):
                        hook.remove()

            self.last_gen_s = time.time() - t0
            print(f"[t2a] {duration:.0f}s audio steps={steps} guidance={guidance} seed={seed} "
                  f"-> {path} ({self.last_gen_s:.0f}s)", flush=True)
            return path, self.last_gen_s
