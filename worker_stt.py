"""worker_stt: the worker-side speech-to-TEXT engine (#stt-serve, Whisper).

The transcription sibling of worker_tts / worker_t2a: serves an OpenAI Whisper
checkpoint (``WhisperForConditionalGeneration`` + ``WhisperProcessor``) as a
single-node speech-recognition leaf. Whisper is a small (~0.8-1.5B) ENCODER-
DECODER model, so the WHOLE model runs on ONE worker — it never touches the
pipeline layer-split machinery the decoder-only LLMs use.

Data flow (mirror of tts/t2a, but the audio travels IN and the result is TEXT):
the controller ships the raw uploaded audio file as base64 over the control link
(``stt_transcribe``); this worker DECODES it (soundfile -> 16 kHz mono float32,
scipy/​numpy resample), runs Whisper, and returns the transcript in ``stt_done``.
Because the deliverable is text on the control reply — not a WAV on a shared
filesystem — an STT model works #media-anywhere: any capable GPU/CPU worker, not
only one co-located with the controller (a remote worker ``snapshot_download``s
the checkpoint itself, exactly like the t2a path).

DEPENDS ON (pip): transformers (Whisper is built in), soundfile (decode the input
audio; wav/flac/ogg), and — only when the input isn't already 16 kHz — scipy for
a high-quality resample (a numpy linear-interp fallback covers a scipy-less box).
Heavy imports live inside methods so importing this module costs nothing.

Worker-side leaf: imported lazily by worker_load's stt branch (fetch-if-missing
via worker_update._fetch_repo_file); listed in client.py's worker update file
list + server.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import io
import os
import threading
import time

GB = 1024 ** 3
SR = 16000                 # Whisper's fixed input sample rate (Hz)
_CHUNK_S = 30              # Whisper's native receptive window (30 s of 16 kHz audio)

# #bazarr-asr: code -> English name, used ONLY when transformers' own LANGUAGES map cannot be
# imported. Covers the languages a media library realistically contains; anything else falls back
# to echoing the code, which still satisfies the contract (language_code is the field that matters).
_LANG_FALLBACK = {
    "en": "english", "es": "spanish", "fr": "french", "de": "german", "it": "italian",
    "pt": "portuguese", "nl": "dutch", "sv": "swedish", "no": "norwegian", "da": "danish",
    "fi": "finnish", "pl": "polish", "ru": "russian", "uk": "ukrainian", "cs": "czech",
    "tr": "turkish", "el": "greek", "he": "hebrew", "ar": "arabic", "hi": "hindi",
    "ja": "japanese", "ko": "korean", "zh": "chinese", "th": "thai", "vi": "vietnamese",
    "id": "indonesian", "ro": "romanian", "hu": "hungarian", "bg": "bulgarian", "hr": "croatian",
    "sr": "serbian", "sk": "slovak", "sl": "slovenian", "ca": "catalan", "fa": "persian",
}


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib at module top."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return True


def _decode_audio(audio_bytes: bytes) -> "object":
    """Raw audio file bytes -> 16 kHz mono float32 numpy array. soundfile decodes
    wav/flac/ogg; a multi-channel signal is averaged to mono; a non-16 kHz signal is
    resampled (scipy.resample_poly, or a numpy linear-interp fallback)."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:                 # stereo / multi-channel -> mono
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if sr != SR and y.size:
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), SR) or 1
            y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
        except Exception:
            # scipy-less fallback: linear interpolation (adequate for ASR features).
            n = int(round(len(y) * SR / float(sr)))
            if n > 0:
                xp = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
                xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
                y = np.interp(xq, xp, y).astype(np.float32)
    return y


class WhisperPipeline:
    """One resident Whisper ASR model on this worker. Stored in worker.shards[model_id]
    like a KokoroPipeline / T2APipeline; ``kind`` lets dispatchers tell it apart. One
    transcribe at a time per model (_gen_lock) — the controller also serializes on
    LoadedModel.lock, this is the worker-side belt."""

    kind = "stt"

    def __init__(self, model_dir: str, device: str = "", quant: str = "none",
                 offload: bool = False):
        import torch
        self.model_dir = model_dir
        self.quant = "none"          # Whisper is a small bf16/fp16 net; no quant tiers in v1
        self.offload = False
        self._gen_lock = threading.Lock()
        self._doomed = False
        t0 = time.time()
        try:
            from transformers import (WhisperForConditionalGeneration,  # noqa: F401
                                      WhisperProcessor)
        except Exception as exc:
            raise RuntimeError(
                "stt serving needs a transformers with Whisper support on this worker "
                f"— import failed: {exc!r}") from exc
        self._Cls = WhisperForConditionalGeneration
        self._processor = WhisperProcessor.from_pretrained(model_dir)

        want = str(device or "")
        if not want or "gpu" in want:
            want = "cuda" if torch.cuda.is_available() else "cpu"
        # gfx1151 / ROCm: Whisper stays on the GPU (decode is fast once the kernels are warm), but
        # the FIRST GPU inference JIT-compiles conv/attention kernels — ~8 min on this chip, and (on
        # TheRock ROCm 7.13) the result is NOT cached to disk, so a warmup on the load would pay it
        # every time AND race the controller restart. So on ROCm we DEFER the warmup: the load
        # returns instantly, the first transcribe pays the JIT once, and the compiled kernels then
        # stay hot in-process for the worker's lifetime (a restart re-pays it — rare). NVIDIA/CUDA
        # warms on load (fast there, no MIOpen). Set INFINITEMODEL_STT_CPU=1 to force CPU instead
        # (instant load, but ~30x-realtime transcribes on gfx1151).
        _rocm = bool(want.startswith("cuda") and getattr(torch.version, "hip", None))
        if want.startswith("cuda") and os.environ.get("INFINITEMODEL_STT_CPU") == "1":
            print("[stt] INFINITEMODEL_STT_CPU=1 — forcing Whisper onto CPU", flush=True)
            want, _rocm = "cpu", False
        elif _rocm:
            print("[stt] ROCm/gfx1151 — Whisper on GPU with warmup DEFERRED to the first transcribe "
                  "(the one-time ~8-min MIOpen JIT won't block the load)", flush=True)
        # Build on the requested device (+ warmup unless deferred on ROCm). A GPU path that raises a
        # HIP/MIOpen kernel-COMPILE error still falls back to CPU transparently.
        self.device = self._build_and_warm(want, warm=not _rocm)

        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size()
                                for p in self.model.parameters()) + \
            sum(b.numel() * b.element_size() for b in self.model.buffers())
        self.gpu_bytes = self.loaded_bytes if str(self.device).startswith("cuda") else 0
        self.last_gen_s = 0.0
        self.last_audio_s = 0.0
        print(f"[stt] whisper ready on {self.device} in {time.time() - t0:.1f}s "
              f"({self.loaded_params / 1e6:.0f}M params, "
              f"{self.loaded_bytes / GB:.2f} GB)", flush=True)

    def _build_and_warm(self, device: str, warm: bool = True) -> str:
        import torch
        import numpy as np

        def _build(dev: str):
            dt = torch.float16 if str(dev).startswith("cuda") else torch.float32
            m = self._Cls.from_pretrained(self.model_dir, torch_dtype=dt)
            return m.to(dev).eval()

        try:
            self.model = _build(device)
            if warm and str(device).startswith("cuda"):
                # Warm the encoder/decoder kernels on 0.2 s of silence — this is where
                # gfx1151 MIOpen JIT-fails if it's going to. Skipped when the caller defers the
                # warmup (ROCm: the ~8-min JIT rides the first real transcribe instead of the load).
                self._run(np.zeros(SR // 5, dtype=np.float32), "", "transcribe")
            return device
        except Exception as exc:
            import worker_hw
            if str(device).startswith("cuda") and worker_hw.gpu_exec_unsupported(exc):
                print(f"[stt] GPU cannot execute this torch build on {device} ({exc!r}) — "
                      "falling back to CPU (Whisper is small; CPU is fine)", flush=True)
                with _suppress():
                    del self.model
                    torch.cuda.empty_cache()
                self.model = _build("cpu")
                return "cpu"
            raise

    def media_info(self) -> dict:
        """Static metadata for the controller's /status + detail modal (reported in the load
        reply). Device is the ACTUAL device after any GPU->CPU fallback."""
        return {"kind": "stt", "engine": "whisper", "device": str(self.device),
                "sample_rate": SR, "params": int(self.loaded_params),
                "loaded_bytes": int(self.loaded_bytes)}

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """Free GPU tensor storages on unload. RENDER-SAFE: under a held _gen_lock we only
        mark _doomed and the transcribe's own finally frees when it completes (mirrors
        KokoroPipeline / T2APipeline)."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[stt] unload during a live transcription — VRAM release deferred to end",
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
        print(f"[stt] {os.path.basename(self.model_dir)}: GPU storages released", flush=True)

    # -- transcription ------------------------------------------------------------------

    def _run(self, samples, language: str, task: str) -> str:
        """One <=30 s window of 16 kHz mono float32 -> text. The processor pads/truncates
        to Whisper's 30 s mel window on its own."""
        import torch
        feats = self._processor(samples, sampling_rate=SR,
                                return_tensors="pt").input_features
        dtype = next(self.model.parameters()).dtype
        feats = feats.to(self.model.device, dtype=dtype)
        kw: dict = {}
        if language:
            kw["language"] = language
        if task:
            kw["task"] = task
        with torch.no_grad():
            ids = self.model.generate(feats, **kw)
        out = self._processor.batch_decode(ids, skip_special_tokens=True)
        return (out[0] if out else "").strip()

    def transcribe(self, audio_bytes: bytes, language: str = "",
                   task: str = "transcribe") -> tuple:
        """Decode + transcribe raw audio file bytes; returns (text, wall_seconds, audio_seconds).
        Runs in a worker thread (asyncio.to_thread) — one at a time per model via _gen_lock."""
        try:
            return self._transcribe(audio_bytes, language, task)
        finally:
            if self._doomed:
                print("[stt] deferred VRAM release: freeing the unloaded model post-transcription",
                      flush=True)
                self._free_now()

    def _transcribe(self, audio_bytes: bytes, language: str, task: str) -> tuple:
        with self._gen_lock:
            t0 = time.time()
            y = _decode_audio(audio_bytes)
            audio_s = len(y) / float(SR) if len(y) else 0.0
            if not len(y):
                raise RuntimeError("empty / undecodable audio")
            task = (task or "transcribe").strip().lower()
            if task not in ("transcribe", "translate"):
                task = "transcribe"
            win = SR * _CHUNK_S
            parts = []
            if len(y) <= win:
                parts.append(self._run(y, language, task))
            else:
                # Long-form: split into <=30 s windows and concatenate (Whisper's native
                # window is 30 s; a segment shorter than 0.2 s is silence tail, skip it).
                for i in range(0, len(y), win):
                    seg = y[i:i + win]
                    if len(seg) < SR // 5:
                        continue
                    parts.append(self._run(seg, language, task))
            text = " ".join(p for p in parts if p).strip()
            self.last_gen_s = time.time() - t0
            self.last_audio_s = audio_s
            print(f"[stt] {audio_s:.1f}s audio -> {len(text)} chars in {self.last_gen_s:.1f}s "
                  f"(RTF {self.last_gen_s / max(audio_s, 1e-6):.2f})", flush=True)
            return text, self.last_gen_s, audio_s

    # -- timestamps + language detection (#bazarr-asr) ------------------------------------

    def _window_segments(self, ids, offset_s: float, win_end_s: float) -> tuple:
        """One window's generated ids -> ([{start,end,text}], how) in ABSOLUTE seconds.

        `how` names the path that produced them, and is logged: a silent fall back to
        whole-window timestamps still yields valid SRT, so without saying which path ran there is
        no way to tell real per-utterance timings from 30 s blocks.

        Two extraction paths because the tokenizer API differs across transformers releases, and
        this worker must not break on the version the fleet happens to have:
          1. `decode(..., output_offsets=True)` — the supported API, gives (start, end) per segment.
          2. Parsing the literal `<|0.00|>` timestamp tokens out of a timestamped decode. Ugly, but
             those tokens are part of Whisper's vocabulary rather than an API, so it survives
             version drift.
        Neither working returns [] and lets the caller fall back to the window bounds."""
        import re as _re
        tok = getattr(self._processor, "tokenizer", None) or self._processor
        seq = ids[0] if hasattr(ids, "__getitem__") and not isinstance(ids, (list, tuple)) else ids
        try:
            out = tok.decode(seq, skip_special_tokens=True, output_offsets=True)
            segs = []
            for o in ((out or {}).get("offsets") or []):
                ts = o.get("timestamp") or (None, None)
                if ts[0] is None:
                    continue
                a = offset_s + float(ts[0])
                b = offset_s + float(ts[1]) if ts[1] is not None else a
                txt = (o.get("text") or "").strip()
                if txt:
                    segs.append({"start": a, "end": max(b, a), "text": txt})
            if segs:
                return segs, "offsets"
        except Exception:
            pass
        try:
            raw = tok.decode(seq, decode_with_timestamps=True, skip_special_tokens=False)
            segs = []
            for a, txt, b in _re.findall(r"<\|(\d+\.\d+)\|>([^<]*)<\|(\d+\.\d+)\|>", raw or ""):
                t = txt.strip()
                if t:
                    segs.append({"start": offset_s + float(a),
                                 "end": offset_s + float(b), "text": t})
            if segs:
                return segs, "markers"
        except Exception:
            pass
        return [], "none"

    def transcribe_segments(self, audio_bytes: bytes, language: str = "",
                            task: str = "transcribe") -> tuple:
        """#bazarr-asr: like transcribe(), but returns TIMESTAMPED segments for SRT output.

        Returns ([{start,end,text}], full_text, wall_s, audio_s). Same 30 s windowing as
        _transcribe — deliberately, so the two share the long-form behaviour that is already
        proven here — with each window's timestamps shifted by that window's absolute offset.
        A window whose timestamps cannot be recovered contributes ONE segment spanning the window,
        so the result is always renderable as SRT even on a transformers build whose tokenizer
        API has moved."""
        import torch
        with self._gen_lock:
            t0 = time.time()
            y = _decode_audio(audio_bytes)
            audio_s = len(y) / float(SR) if len(y) else 0.0
            if not len(y):
                raise RuntimeError("empty / undecodable audio")
            task = (task or "transcribe").strip().lower()
            if task not in ("transcribe", "translate"):
                task = "transcribe"
            win = SR * _CHUNK_S
            segments: list = []
            hows: dict = {}
            for i in range(0, max(len(y), 1), win):
                seg = y[i:i + win]
                if len(seg) < SR // 5:      # <0.2 s tail is silence
                    continue
                off = i / float(SR)
                end = off + len(seg) / float(SR)
                feats = self._processor(seg, sampling_rate=SR,
                                        return_tensors="pt").input_features
                dtype = next(self.model.parameters()).dtype
                feats = feats.to(self.model.device, dtype=dtype)
                kw: dict = {"return_timestamps": True}
                if language:
                    kw["language"] = language
                if task:
                    kw["task"] = task
                with torch.no_grad():
                    ids = self.model.generate(feats, **kw)
                got, how = self._window_segments(ids, off, end)
                hows[how] = hows.get(how, 0) + 1
                if not got:
                    txt = (self._processor.batch_decode(ids, skip_special_tokens=True) or [""])[0]
                    txt = txt.strip()
                    if txt:
                        got = [{"start": off, "end": end, "text": txt}]
                segments.extend(got)
            # Clamp to the real audio length and drop empties: Whisper happily emits a timestamp
            # past the end of a padded final window, which renders as an SRT cue running beyond
            # the video.
            out = []
            for s in segments:
                a = max(0.0, min(float(s["start"]), audio_s))
                b = max(a, min(float(s["end"]), audio_s))
                if s.get("text"):
                    out.append({"start": a, "end": b, "text": s["text"]})
            text = " ".join(s["text"] for s in out).strip()
            self.last_gen_s = time.time() - t0
            self.last_audio_s = audio_s
            print(f"[stt] {audio_s:.1f}s audio -> {len(out)} segment(s), {len(text)} chars in "
                  f"{self.last_gen_s:.1f}s (timestamps: "
                  f"{', '.join(f'{k}x{v}' for k, v in hows.items()) or 'none'})", flush=True)
            return out, text, self.last_gen_s, audio_s

    def detect_language(self, audio_bytes: bytes) -> tuple:
        """#bazarr-asr: detect the spoken language from the FIRST 30 s. -> (code, name).

        Whisper predicts the language as a single `<|xx|>` token before any text, so this needs
        one window and one token, not a transcription — which is why Bazarr's /detect-language is
        near-instant next to /asr."""
        import torch
        with self._gen_lock:
            y = _decode_audio(audio_bytes)
            if not len(y):
                raise RuntimeError("empty / undecodable audio")
            y = y[:SR * _CHUNK_S]
            feats = self._processor(y, sampling_rate=SR, return_tensors="pt").input_features
            dtype = next(self.model.parameters()).dtype
            feats = feats.to(self.model.device, dtype=dtype)
            tok = getattr(self._processor, "tokenizer", None) or self._processor
            code = ""
            # Preferred: the model's own detector (transformers >= 4.39).
            try:
                with torch.no_grad():
                    lid = self.model.detect_language(input_features=feats)
                t = tok.convert_ids_to_tokens(int(lid[0][0] if hasattr(lid[0], "__len__") else lid[0]))
                code = str(t).strip("<|>")
            except Exception:
                # Fallback: generate a couple of tokens with NO forced language and read the
                # language token out of the prefix.
                try:
                    import re as _re
                    with torch.no_grad():
                        ids = self.model.generate(feats, max_new_tokens=2, return_timestamps=False)
                    raw = tok.decode(ids[0], skip_special_tokens=False)
                    # The prefix looks like <|startoftranscript|><|en|><|transcribe|>...
                    # so the FIRST 2-3 letter tag is the language.
                    m = _re.search(r"<\|([a-z]{2,3})\|>", raw or "")
                    if m:
                        code = m.group(1)
                except Exception:
                    code = ""
            code = (code or "en").lower()
            name = ""
            try:    # transformers ships the canonical code -> English-name map
                from transformers.models.whisper.tokenization_whisper import LANGUAGES
                name = str(LANGUAGES.get(code) or "")
            except Exception:
                name = ""
            if not name:
                name = _LANG_FALLBACK.get(code, code)
            print(f"[stt] detected language: {code} ({name})", flush=True)
            return code, name.lower()
