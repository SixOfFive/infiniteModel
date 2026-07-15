"""worker_tts: the worker-side text-to-SPEECH engine (#tts-serve, Kokoro).

The speech sibling of worker_t2i / worker_t2a: serves a Kokoro-82M StyleTTS2
checkpoint (a single ``kokoro-v1_0.pth`` + a ``voices/`` pack dir + ``config.json``
— NO safetensors, NO diffusers ``model_index.json``) as a single-node speech
generator. The WHOLE model lives on ONE controller-CO-LOCATED worker; requests
arrive over the control link (``tts_gen``), per-chunk progress mirrors back
(``tts_step``), and the finished WAV is written to LOCAL disk with its path
returned (``tts_done``) — v1, like t2i/t2a, serves only on a worker sharing the
controller's filesystem (result write needs no transfer).

Why we drive Kokoro's ``KModel`` directly instead of its ``KPipeline``:
``kokoro``'s pipeline imports ``misaki.en`` -> ``spacy`` -> ``thinc`` -> ``blis``,
and blis has no wheel for py3.13/3.14 and fails to Cython-compile from source on
the fleet. So we install ``kokoro``/``misaki`` with ``--no-deps`` and phonemize
via ``misaki.espeak.EspeakFallback`` (spacy-free, uses the pip-bundled
``espeakng-loader`` binary — no system espeak-ng). ``KModel`` is language-blind:
it maps a phoneme STRING -> input_ids via its own vocab and takes a per-voice
256-d style vector, so it needs only torch + transformers + scipy + numpy.

Two runtime notes baked in from the M0 bring-up:
  * ``import kokoro`` still pulls ``KPipeline`` -> ``misaki.en`` -> ``spacy`` at
    module load, so we inject a harmless ``spacy`` stub into ``sys.modules`` first
    (we never touch ``misaki.en`` — English goes through EspeakFallback).
  * On gfx1151 (Strix Halo, om3nbox) MIOpen JIT-fails to compile the LSTM dropout
    kernel (``MIOpenDropoutHIP.cpp: '<utility>' file not found`` — a TheRock
    ROCm-7.13 bug), so a GPU warmup that raises a HIP/MIOpen compile error
    transparently RE-BUILDS the model on CPU. Kokoro is 82M params, so CPU
    synthesis (~4x realtime) is a fine fallback; beast's CUDA path is ~4x FASTER
    than realtime.

DEPENDS ON (pip, --no-deps for the first two): kokoro, misaki, plus loguru,
espeakng-loader, phonemizer-fork, num2words, regex, scipy, soundfile. Heavy
imports live inside methods so importing this module costs nothing.

Worker-side leaf: imported lazily by worker_load's tts branch (fetch-if-missing
via worker_update._fetch_repo_file); listed in client.py's worker update file
list + server.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time

GB = 1024 ** 3
SR = 24000                      # Kokoro output sample rate (Hz)
_MAX_PHONEMES = 508             # KModel context_length is ~512 incl. 2 bos/eos frames


def _wire_espeak() -> None:
    """Point phonemizer at the pip-bundled espeak-ng (these boxes have NO system
    espeak-ng). Idempotent — sets the library + data path once."""
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except Exception:
        return
    if getattr(EspeakWrapper, "_ESPEAK_LIBRARY", None):
        return
    import espeakng_loader
    with_lib = espeakng_loader.get_library_path()
    EspeakWrapper.set_library(with_lib)
    try:
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:
        pass


def _stub_spacy() -> None:
    """`import kokoro` transitively imports KPipeline -> misaki.en -> spacy, which
    is uninstallable here (blis won't build). We never use misaki.en (English via
    EspeakFallback), so a stub module lets the import chain complete inertly."""
    if "spacy" in sys.modules:
        return
    try:
        import spacy  # noqa: F401
    except Exception:
        import types
        sys.modules["spacy"] = types.ModuleType("spacy")


def _split_sentences(text: str) -> list:
    """Chunk text on sentence boundaries so each chunk phonemizes under KModel's
    context window. Keeps terminal punctuation; falls back to whitespace splitting
    for a run with no sentence punctuation."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


class KokoroPipeline:
    """One resident Kokoro TTS model on this worker. Stored in worker.shards[model_id]
    like a T2IPipeline / T2APipeline; `kind` lets dispatchers tell it apart. One
    generate at a time per model (_gen_lock) — the controller also serializes on
    LoadedModel.lock, this is the worker-side belt."""

    kind = "tts"
    DEFAULT_VOICE = "af_heart"

    def __init__(self, model_dir: str, device: str = "", quant: str = "none",
                 offload: bool = False):
        import torch
        self.model_dir = model_dir
        self.quant = "none"          # Kokoro is a small bf16/fp32 net; no quant tiers
        self.offload = False
        self._gen_lock = threading.Lock()
        self._doomed = False
        self._voice_cache: dict = {}
        self._g2p_cache: dict = {}
        t0 = time.time()

        cfg = os.path.join(model_dir, "config.json")
        pth = os.path.join(model_dir, "kokoro-v1_0.pth")
        if not (os.path.exists(cfg) and os.path.exists(pth)):
            raise RuntimeError(
                f"kokoro checkpoint incomplete in {model_dir!r} "
                f"(need config.json + kokoro-v1_0.pth + voices/)")
        self._voices_dir = os.path.join(model_dir, "voices")

        _wire_espeak()
        _stub_spacy()
        try:
            from kokoro import KModel
        except Exception as exc:
            raise RuntimeError(
                "tts serving needs the `kokoro` package on this worker "
                "(pip install --no-deps kokoro misaki ; pip install loguru "
                "espeakng-loader phonemizer-fork num2words regex scipy soundfile) "
                f"— import failed: {exc!r}") from exc
        self._KModel = KModel

        want = str(device or "")
        if not want or "gpu" in want:
            want = "cuda" if torch.cuda.is_available() else "cpu"
        # Build on the requested device, then a tiny warmup synth. If the GPU path
        # raises a HIP/MIOpen kernel-COMPILE error (gfx1151 LSTM-dropout bug), fall
        # back to CPU transparently — the model is tiny, correctness > speed.
        self.device = self._build_and_warm(want)

        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size()
                                for p in self.model.parameters()) + \
            sum(b.numel() * b.element_size() for b in self.model.buffers())
        self.gpu_bytes = self.loaded_bytes if str(self.device).startswith("cuda") else 0
        self.last_gen_s = 0.0
        n_voices = len([f for f in os.listdir(self._voices_dir)
                        if f.endswith(".pt")]) if os.path.isdir(self._voices_dir) else 0
        print(f"[tts] kokoro ready on {self.device} in {time.time() - t0:.1f}s "
              f"({self.loaded_params / 1e6:.0f}M params, {n_voices} voices, "
              f"{self.loaded_bytes / GB:.2f} GB)", flush=True)

    def _build_and_warm(self, device: str) -> str:
        import torch
        cfg = os.path.join(self.model_dir, "config.json")
        pth = os.path.join(self.model_dir, "kokoro-v1_0.pth")

        def _build(dev: str):
            m = self._KModel(config=cfg, model=pth).to(dev).eval()
            return m

        try:
            self.model = _build(device)
            if str(device).startswith("cuda"):
                # Warm the LSTM/dropout + attention kernels; this is where gfx1151
                # MIOpen JIT-fails if it's going to.
                self._synth_chunk("Ready.", self.DEFAULT_VOICE, 1.0)
            return device
        except Exception as exc:
            msg = repr(exc)
            hip = any(s in msg for s in ("MIOpen", "HIPRTC", "hiprtc", "HIP error",
                                         "hipErrorNoBinaryForGpu", "miopen",
                                         "Code object build failed"))
            if str(device).startswith("cuda") and hip:
                print(f"[tts] GPU kernel-compile failed on {device} ({exc!r}) — "
                      "falling back to CPU (Kokoro is 82M; CPU is fine)", flush=True)
                with _suppress():
                    del self.model
                    torch.cuda.empty_cache()
                self.model = _build("cpu")
                return "cpu"
            raise

    # -- G2P + voices -------------------------------------------------------------------

    def _g2p_for_voice(self, voice: str):
        """Pick a spacy-free G2P by the voice's language prefix (Kokoro voice ids are
        ``<lang><gender>_name``: a/b = American/British English -> EspeakFallback;
        others -> EspeakG2P for that espeak language). Cached per key."""
        pfx = (voice or "a")[0].lower()
        if pfx in self._g2p_cache:
            return self._g2p_cache[pfx]
        from misaki.espeak import EspeakFallback, EspeakG2P
        _LANG = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it",
                 "p": "pt-br", "j": "ja", "z": "cmn"}
        if pfx == "b":
            g = EspeakFallback(british=True)
        elif pfx == "a":
            g = EspeakFallback(british=False)
        else:
            g = EspeakG2P(language=_LANG.get(pfx, "en-us"))
        self._g2p_cache[pfx] = g
        return g

    def _phonemize(self, text: str, voice: str) -> str:
        g2p = self._g2p_for_voice(voice)
        # EspeakFallback(token) expects an object with .text and returns (ps, rating);
        # EspeakG2P(text) returns a plain string.
        from misaki.espeak import EspeakFallback
        if isinstance(g2p, EspeakFallback):
            out = g2p(type("T", (), {"text": text})())
            ps = out[0] if isinstance(out, tuple) else out
        else:
            ps = g2p(text)
        return (ps or "").strip()

    def _load_voice(self, voice: str):
        import torch
        if voice in self._voice_cache:
            return self._voice_cache[voice]
        p = os.path.join(self._voices_dir, f"{voice}.pt")
        if not os.path.exists(p):
            raise RuntimeError(f"unknown voice '{voice}' (no {voice}.pt in voices/)")
        pack = torch.load(p, weights_only=True, map_location="cpu")
        self._voice_cache[voice] = pack
        return pack

    def list_voices(self) -> list:
        if not os.path.isdir(self._voices_dir):
            return []
        return sorted(f[:-3] for f in os.listdir(self._voices_dir) if f.endswith(".pt"))

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """Free GPU tensor storages on unload. RENDER-SAFE: under a held _gen_lock we
        only mark _doomed and the render's own finally frees when it completes (mirrors
        T2IPipeline / T2APipeline)."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[tts] unload during a live render — VRAM release deferred to render end",
                  flush=True)
            return
        self._free_now()

    def _free_now(self) -> None:
        import torch
        with _suppress():
            for t in list(self.model.parameters(recurse=True)) + \
                    list(self.model.buffers(recurse=True)):
                if t is not None and getattr(t, "device", None) is not None \
                        and t.device.type == "cuda":
                    t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        with _suppress():
            if str(self.device).startswith("cuda"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        print(f"[tts] {os.path.basename(self.model_dir)}: GPU storages released", flush=True)

    # -- generation ---------------------------------------------------------------------

    def _synth_chunk(self, text: str, voice: str, speed: float):
        """One phoneme chunk -> float32 audio tensor (CPU). Raises on empty phonemes."""
        import torch
        ps = self._phonemize(text, voice)
        if not ps:
            return None
        if len(ps) > _MAX_PHONEMES:
            ps = ps[:_MAX_PHONEMES]
        pack = self._load_voice(voice)
        ref_s = pack[len(ps) - 1]
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)
        ref_s = ref_s.to(self.model.device)
        with torch.no_grad():
            audio = self.model(ps, ref_s, speed=float(speed))
        return audio.detach().cpu().float()

    def generate(self, text: str, voice: str = "", speed: float = 1.0,
                 fmt: str = "wav", on_step=None) -> tuple:
        """Synthesize `text` in `voice`; returns (audio_path, seconds). Runs in a
        worker thread (asyncio.to_thread) — one at a time per model via _gen_lock."""
        try:
            return self._generate(text, voice, speed, fmt, on_step)
        finally:
            if self._doomed:
                print("[tts] deferred VRAM release: freeing the unloaded model post-render",
                      flush=True)
                self._free_now()

    def _generate(self, text: str, voice: str, speed: float, fmt: str, on_step) -> tuple:
        import numpy as np
        import soundfile as sf
        with self._gen_lock:
            t0 = time.time()
            voice = (voice or self.DEFAULT_VOICE).strip()
            speed = max(0.5, min(2.0, float(speed or 1.0)))
            chunks = _split_sentences(text)
            if not chunks:
                raise RuntimeError("empty input text")
            total = len(chunks)
            pieces = []
            gap = np.zeros(int(SR * 0.10), dtype=np.float32)   # 100ms between sentences
            for i, ch in enumerate(chunks):
                audio = self._synth_chunk(ch, voice, speed)
                if audio is not None and audio.numel():
                    pieces.append(audio.numpy())
                    pieces.append(gap)
                if on_step is not None:
                    with _suppress():
                        on_step(i + 1, total)
            if not pieces:
                raise RuntimeError("synthesis produced no audio")
            wav = np.concatenate(pieces).astype(np.float32)
            ext = "wav" if str(fmt).lower() not in ("flac", "ogg", "wav") else str(fmt).lower()
            path = os.path.join(tempfile.gettempdir(),
                                f"im_tts_{os.getpid()}_{int(time.time() * 1000)}.{ext}")
            sf.write(path, wav, SR)
            self.last_gen_s = time.time() - t0
            secs = len(wav) / float(SR)
            print(f"[tts] voice={voice} {total} chunk(s) -> {secs:.1f}s audio "
                  f"in {self.last_gen_s:.1f}s (RTF {self.last_gen_s / max(secs, 1e-6):.2f}) "
                  f"-> {path}", flush=True)
            return path, self.last_gen_s


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib at module top."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return True
