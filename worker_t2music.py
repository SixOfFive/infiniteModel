"""worker_t2music: the worker-side text-to-MUSIC engine (#t2music-serve, MusicGen).

The music-generation sibling of worker_t2a (ACE-Step), but a DIFFERENT architecture
on purpose: MusicGen is an AUTOREGRESSIVE transformer that decodes discrete EnCodec
audio tokens step by step (vs ACE-Step's continuous latent DIFFUSION). That matters
for portability:

  * It ships INSIDE HuggingFace `transformers` (`MusicgenForConditionalGeneration` +
    `AutoProcessor`) — no separate `acestep` package, no GitHub source install.
  * Text->music needs NO `torchaudio`: generate() returns an audio tensor and we write
    the WAV with `soundfile`. (ACE-Step hard-depends on torchaudio, which has no ROCm
    build — the wall that makes ACE-Step un-runnable on gfx1151.)
  * The heavy compute is the transformer decode (pure attention/matmul — fast on every
    backend incl. ROCm/gfx1151, no MIOpen). The ONLY conv is EnCodec's one-shot decode
    at the end, a bounded MIOpen JIT-once-then-fast tax exactly like Whisper's encoder.
  * There IS a CPU fallback (slow, but it runs) — ACE-Step has none.

So this leaf runs on AMD GPU (ROCm), NVIDIA GPU (CUDA) and CPU. Whole model on ONE
worker (like tts/stt/t2a) — it never touches the pipeline layer-split machinery.

Data flow (mirror of t2a): the controller ships a text prompt over the control link
(`t2music_gen`); this worker runs MusicGen, encodes the waveform to a WAV in memory
(soundfile), and returns it as base64 in `t2music_done` (#media-anywhere — no shared
FS needed, so any capable GPU/CPU worker can serve it, not only a co-located one).

MELODY (#t2music-melody): a `*-melody` checkpoint is NOT a bigger MusicGen, it is a
DIFFERENT transformers architecture (`MusicgenMelody*`) whose conditioning is a chroma
PREFIX computed from a reference waveform instead of cross-attention over text. This leaf
picks the model class off the checkpoint's own config, and `generate(..., melody_audio=...)`
takes the reference clip — file bytes, or base64/data-URI text, because the control link is
JSON. A NON-melody checkpoint REJECTS melody_audio rather than quietly rendering a text-only
clip the caller would read as a failed melody. The controller side of that parameter (an HTTP
field -> `t2music_gen` -> worker_load.handle_t2music_gen -> this kwarg) is NOT wired yet; those
files are outside this leaf.

DEPENDS ON (pip): transformers (MusicGen is built in) + soundfile (WAV encode / melody decode)
— both already required by the stt leaf, so a `can_stt` worker is also `can_t2music`. Melody
resampling prefers scipy (as in worker_stt) but degrades to a numpy interp. Heavy imports live
inside methods so importing this module costs nothing.

Worker-side leaf: imported lazily by worker_load's t2music branch (fetch-if-missing via
worker_update._fetch_repo_file); listed in client.py's worker update file list +
server.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import io
import os
import threading
import time

GB = 1024 ** 3


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib at module top."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return True


def _is_melody_dir(model_dir: str) -> bool:
    """True when this checkpoint is a MusicGen-MELODY variant (chroma/audio conditioning).
    Decided by the checkpoint's OWN config — parameter count cannot tell musicgen-melody
    (1.5B) from musicgen-medium (1.5B), and guessing wrong picks the wrong model class."""
    import json
    try:
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return False                      # unreadable config -> treat as plain MusicGen
    if "melody" in str(cfg.get("model_type") or "").lower():
        return True
    return any("melody" in str(a).lower() for a in (cfg.get("architectures") or []))


def _as_audio_bytes(blob) -> bytes:
    """Melody conditioning crosses the control link, which is JSON — so accept either raw file
    bytes (a co-located caller) or base64 / a data: URI (the wire form, the same shape the stt
    leaf's input audio uses)."""
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return bytes(blob)
    if isinstance(blob, str):
        import base64
        s = blob.strip()
        if s.startswith("data:"):                 # data:audio/wav;base64,AAAA...
            s = s.split(",", 1)[-1]
        try:
            return base64.b64decode(s)
        except Exception as exc:
            raise RuntimeError(
                f"melody audio is text but not decodable base64: {exc!r}") from exc
    raise RuntimeError("melody audio must be audio-file bytes or base64 text, got "
                       f"{type(blob).__name__}")


def _decode_audio(audio_bytes: bytes, target_sr: int):
    """Audio file bytes -> mono float32 numpy at `target_sr`. Mirrors worker_stt._decode_audio:
    soundfile decodes wav/flac/ogg, multi-channel averages to mono, and the rate is converted
    with scipy.resample_poly (numpy linear-interp fallback on a scipy-less box). The conversion
    is NOT optional — the caller's clip is whatever they had, and handing the chroma extractor a
    waveform at the wrong rate transposes the melody instead of failing."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:                 # stereo / multi-channel -> mono
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if int(sr) != int(target_sr) and y.size:
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), int(target_sr)) or 1
            y = resample_poly(y, int(target_sr) // g, int(sr) // g).astype(np.float32)
        except Exception:
            n = int(round(len(y) * float(target_sr) / float(sr)))
            if n > 0:
                xp = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
                xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
                y = np.interp(xq, xp, y).astype(np.float32)
    return y


class MusicGenPipeline:
    """One resident MusicGen model on this worker. Stored in worker.shards[model_id] like a
    T2APipeline / WhisperPipeline; ``kind`` lets dispatchers tell it apart. One render at a
    time per model (_gen_lock) — the controller also serializes on LoadedModel.lock, this is
    the worker-side belt."""

    kind = "t2music"

    def __init__(self, model_dir: str, device: str = "", quant: str = "none",
                 offload: bool = False):
        import torch
        self.model_dir = model_dir
        self.quant = "none"          # MusicGen is a small bf16/fp16 net; no quant tiers in v1
        self.offload = False
        self._gen_lock = threading.Lock()
        self._doomed = False
        t0 = time.time()
        try:
            from transformers import (AutoProcessor,  # noqa: F401
                                      MusicgenForConditionalGeneration)
        except Exception as exc:
            raise RuntimeError(
                "t2music serving needs a transformers with MusicGen support on this worker "
                f"— import failed: {exc!r}") from exc
        # #t2music-melody: a melody checkpoint is a SEPARATE architecture in transformers
        # (MusicgenMelody*, conditioning as a chroma prefix rather than cross-attention), with
        # its own config class and processor. Loading one through MusicgenForConditionalGeneration
        # mismatches the decoder weights, so choose the class from the checkpoint's config.
        # model_store._is_musicgen_dir already admits both 'musicgen' and 'musicgen_melody', so
        # both kinds reach this leaf.
        self.is_melody = _is_melody_dir(model_dir)
        self._Cls = MusicgenForConditionalGeneration
        if self.is_melody:
            try:
                from transformers import MusicgenMelodyForConditionalGeneration
            except Exception as exc:
                raise RuntimeError(
                    "this checkpoint is a MusicGen-MELODY variant, but this worker's transformers "
                    "has no MusicgenMelody support (added in 4.40) — import failed: "
                    f"{exc!r}") from exc
            self._Cls = MusicgenMelodyForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_dir)

        want = str(device or "")
        if not want or "gpu" in want:
            want = "cuda" if torch.cuda.is_available() else "cpu"
        # gfx1151 / ROCm: KEEP MusicGen on the GPU — the bottleneck is the autoregressive
        # transformer (pure attention/matmul, no MIOpen, fast on ROCm). The ONLY conv is
        # EnCodec's decode, which JIT-compiles through MIOpen the FIRST time it runs; that
        # cost rides the first render (not the load), then stays hot for the worker's life.
        # We therefore do NOT warm on load (a warmup render is expensive anyway). CUDA warms
        # naturally on first render (no MIOpen). Set INFINITEMODEL_T2MUSIC_CPU=1 to force CPU.
        self._rocm = bool(want.startswith("cuda") and getattr(torch.version, "hip", None))
        if want.startswith("cuda") and os.environ.get("INFINITEMODEL_T2MUSIC_CPU") == "1":
            print("[t2music] INFINITEMODEL_T2MUSIC_CPU=1 — forcing MusicGen onto CPU", flush=True)
            want, self._rocm = "cpu", False
        elif self._rocm:
            print("[t2music] ROCm/gfx1151 — MusicGen on GPU (transformer is MIOpen-free; the "
                  "one-time EnCodec-decode MIOpen JIT rides the FIRST render)", flush=True)
        self.device = self._build(want)

        ae = getattr(self.model.config, "audio_encoder", None)
        self.sample_rate = int(getattr(ae, "sampling_rate", 32000) or 32000)
        # tokens/sec of generated audio — duration -> max_new_tokens. EnCodec@32k = 50 Hz.
        self.frame_rate = float(getattr(ae, "frame_rate", 50.0) or 50.0)
        # #t2music-cap: the audio-token decoder has a FIXED position table
        # (decoder.max_position_embeddings, 2048 for MusicGen). Its delay pattern adds
        # num_codebooks-1 positions + a BOS, so generating past ~that many tokens indexes OUT of the
        # table -> a CUDA device-side assert (which then WEDGES the worker's CUDA context) or a CPU
        # crash. Clamp EVERY render to this hard ceiling. ~2036 tokens ≈ 40 s at 50 Hz.
        _dec = getattr(self.model.config, "decoder", None)
        _pos = int(getattr(_dec, "max_position_embeddings", 2048) or 2048)
        self.num_codebooks = int(getattr(_dec, "num_codebooks", 4) or 4)
        self.max_tokens = max(16, _pos - self.num_codebooks - 8)
        self.max_duration_s = round(self.max_tokens / self.frame_rate, 1)
        # Rate the melody feature extractor computes chroma at (32 kHz for musicgen-melody). Read
        # it off the processor rather than assuming it equals the OUTPUT rate — a conditioning clip
        # gets resampled to THIS, and the two are only incidentally the same number today.
        _fe = getattr(self._processor, "feature_extractor", None)
        self.melody_sr = int(getattr(_fe, "sampling_rate", 0) or self.sample_rate) \
            if self.is_melody else 0
        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size()
                                for p in self.model.parameters()) + \
            sum(b.numel() * b.element_size() for b in self.model.buffers())
        self.gpu_bytes = self.loaded_bytes if str(self.device).startswith("cuda") else 0
        self.last_gen_s = 0.0
        self.last_audio_s = 0.0
        print(f"[t2music] {'musicgen-melody' if self.is_melody else 'musicgen'} ready on "
              f"{self.device} in {time.time() - t0:.1f}s "
              f"({self.loaded_params / 1e6:.0f}M params, {self.loaded_bytes / GB:.2f} GB, "
              f"{self.sample_rate} Hz"
              f"{f', melody conditioning @ {self.melody_sr} Hz' if self.is_melody else ''})",
              flush=True)

    def _build(self, device: str) -> str:
        import torch

        def _b(dev: str):
            dt = torch.float16 if str(dev).startswith("cuda") else torch.float32
            m = self._Cls.from_pretrained(self.model_dir, torch_dtype=dt)
            return m.to(dev).eval()

        try:
            self.model = _b(device)
            return device
        except Exception as exc:
            import worker_hw
            # include_oom: for MusicGen a GPU that can't FIT the model is also a CPU case
            # ("out of memory" also substring-matches "CUDA out of memory").
            if str(device).startswith("cuda") and \
                    worker_hw.gpu_exec_unsupported(exc, include_oom=True):
                print(f"[t2music] GPU build failed on {device} ({exc!r}) — falling back to CPU "
                      "(MusicGen runs on CPU; slower)", flush=True)
                with _suppress():
                    del self.model
                    torch.cuda.empty_cache()
                self.model = _b("cpu")
                return "cpu"
            raise

    def media_info(self) -> dict:
        """Static metadata for the controller's /status + detail modal (reported in the load
        reply). Device is the ACTUAL device after any GPU->CPU fallback. Rich on purpose —
        the model-detail page surfaces all of it."""
        _sz = self.loaded_params
        # Melody is a CONFIG fact, not a size bucket: the old chain guessed "melody" from a
        # parameter count, which labelled musicgen-SMALL (~0.4B once its T5 encoder and EnCodec
        # are counted) "melody" and the real 1.5B musicgen-melody "medium" — exactly backwards.
        _label = ("melody" if self.is_melody else "large" if _sz > 2.5e9
                  else "medium" if _sz > 1.0e9 else "small")
        return {"kind": "t2music", "engine": "musicgen", "device": str(self.device),
                "sample_rate": self.sample_rate, "frame_rate": self.frame_rate,
                "params": int(self.loaded_params), "loaded_bytes": int(self.loaded_bytes),
                "variant": _label,
                # #t2music-melody: tells the controller/UI whether a melody clip is even offerable
                # — and at what rate the worker will resample it to before extracting chroma.
                "melody": bool(self.is_melody), "melody_sample_rate": int(self.melody_sr),
                "max_duration_s": int(self.max_duration_s), "default_guidance": 3.0,
                "backend": ("ROCm" if self._rocm else
                            "CUDA" if str(self.device).startswith("cuda") else "CPU"),
                "channels": 1, "codec": "EnCodec", "arch": "autoregressive-transformer"}

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """Free GPU tensor storages on unload. RENDER-SAFE: under a held _gen_lock we only mark
        _doomed and the render's own finally frees when it completes (mirrors T2APipeline)."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[t2music] unload during a live render — VRAM release deferred to end",
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
        print(f"[t2music] {os.path.basename(self.model_dir)}: GPU storages released", flush=True)

    # -- generation ---------------------------------------------------------------------

    def _wav_bytes(self, samples, sr: int) -> bytes:
        """float32 mono [-1,1] -> 16-bit PCM WAV bytes via soundfile (NO torchaudio)."""
        import numpy as np
        import soundfile as sf
        y = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:                       # guard against clipping if the decoder overshoots
            y = y / peak
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def generate(self, prompt: str, duration: float = 15.0, guidance: float = 3.0,
                 temperature: float = 1.0, top_k: int = 250, top_p: float = 0.0,
                 seed=None, on_step=None, melody_audio=None) -> tuple:
        """Text prompt (+ optional melody conditioning) -> (wav_bytes, wall_seconds,
        audio_seconds). Runs in a worker thread (asyncio.to_thread) — one at a time per model via
        _gen_lock. `melody_audio` is an audio FILE (bytes, or base64/data-URI text off the control
        link); it is keyword-safe LAST so the existing positional call in
        worker_load.handle_t2music_gen keeps working unchanged."""
        try:
            return self._generate(prompt, duration, guidance, temperature, top_k, top_p, seed,
                                  on_step, melody_audio)
        finally:
            if self._doomed:
                print("[t2music] deferred VRAM release: freeing the unloaded model post-render",
                      flush=True)
                self._free_now()

    def _generate(self, prompt, duration, guidance, temperature, top_k, top_p, seed,
                  on_step, melody_audio=None) -> tuple:
        import torch
        with self._gen_lock:
            t0 = time.time()
            prompt = (prompt or "").strip()
            if not prompt:
                raise RuntimeError("empty prompt")
            dur = max(1.0, min(float(self.max_duration_s), float(duration)))
            # NEVER exceed the decoder's position table — overrunning it is the CUDA device-side
            # assert (which wedges the worker) / CPU crash. self.max_tokens is the hard ceiling.
            max_new = min(max(16, int(round(dur * self.frame_rate))), self.max_tokens)
            if seed is not None:
                with _suppress():
                    torch.manual_seed(int(seed))
                    if str(self.device).startswith("cuda"):
                        torch.cuda.manual_seed_all(int(seed))
            mel = None
            if melody_audio:                     # None / b"" / "" all mean "no conditioning"
                if not self.is_melody:
                    raise RuntimeError(
                        f"{os.path.basename(self.model_dir)} is a plain MusicGen checkpoint: it "
                        "has no audio-conditioning path, so a melody clip would be SILENTLY "
                        "dropped and the render would come back as an ordinary text-only clip. "
                        "Load a melody variant (e.g. facebook/musicgen-melody), or drop the "
                        "melody input.")
                mel = _decode_audio(_as_audio_bytes(melody_audio), self.melody_sr)
                if not mel.size:
                    raise RuntimeError(
                        "melody conditioning decoded to 0 samples — the clip is empty, or "
                        "soundfile could not read that container")
            if mel is not None:
                # The processor OWNS chroma extraction; we only hand it a mono float32 waveform at
                # the rate it declared. Text goes in as well: the melody drives the tune, the
                # prompt still drives instrumentation and style.
                inputs = self._processor(audio=mel, sampling_rate=self.melody_sr, text=[prompt],
                                         padding=True, return_tensors="pt")
            else:
                inputs = self._processor(text=[prompt], padding=True, return_tensors="pt")
            _dt = next(self.model.parameters()).dtype

            def _to_dev(v):
                # Chroma comes out of the feature extractor as float32 and feeds a projection that
                # is fp16 on a GPU build — cast float tensors or the render dies on a dtype
                # mismatch. Integer tensors (input_ids / attention_mask) must NOT be cast.
                if getattr(v, "is_floating_point", None) and v.is_floating_point():
                    return v.to(self.model.device, dtype=_dt)
                return v.to(self.model.device)
            inputs = {k: _to_dev(v) for k, v in inputs.items() if hasattr(v, "to")}
            if self.is_melody:
                # #t2music-melody: MusicgenMelody is a PREFIX model — the projected text hidden
                # states AND the chroma frames are CONCATENATED in front of the audio tokens in the
                # decoder's own sequence, so conditioning spends positions out of the SAME table the
                # max_tokens clamp above exists to protect. Budget it out: otherwise a long melody
                # walks the decode off the end of the table, which is the CUDA device-side assert
                # that wedges this worker's context. Plain MusicGen is exempt — its text goes in by
                # CROSS-attention and costs the decoder no positions at all. Note this applies to a
                # melody checkpoint even with NO clip: the text prefix alone still costs positions.
                pre = sum(int(t.shape[1]) for t in (inputs.get("input_ids"),
                                                    inputs.get("input_features"))
                          if t is not None and getattr(t, "ndim", 0) >= 2)
                room = self.max_tokens - pre
                if room < 16:
                    raise RuntimeError(
                        f"conditioning claims {pre} of the decoder's {self.max_tokens} usable "
                        "positions — nothing left to generate; use a shorter melody clip")
                if max_new > room:
                    print(f"[t2music] melody prefix costs {pre} positions (text"
                          f"{' + chroma' if mel is not None else ''}) — capping this render at "
                          f"{room} tokens ({room / self.frame_rate:.1f}s; asked {max_new})",
                          flush=True)
                    max_new = room
            gen_kw = dict(max_new_tokens=max_new, do_sample=True,
                          guidance_scale=float(guidance),
                          temperature=max(1e-3, float(temperature)),
                          top_k=int(top_k))
            if top_p and float(top_p) > 0:
                gen_kw["top_p"] = float(top_p)
            # optional coarse token-progress (best-effort; never breaks generation)
            crit = _mk_progress(on_step, max_new) if on_step else None
            if crit is not None:
                try:
                    from transformers import StoppingCriteriaList
                    gen_kw["stopping_criteria"] = StoppingCriteriaList([crit])
                except Exception:
                    pass
            with torch.no_grad():
                audio = self.model.generate(**inputs, **gen_kw)
            arr = audio[0].detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
            audio_s = len(arr) / float(self.sample_rate) if len(arr) else 0.0
            wav = self._wav_bytes(arr, self.sample_rate)
            self.last_gen_s = time.time() - t0
            self.last_audio_s = audio_s
            print(f"[t2music] '{prompt[:48]}'{' +melody' if mel is not None else ''} -> "
                  f"{audio_s:.1f}s audio in {self.last_gen_s:.1f}s "
                  f"(RTF {self.last_gen_s / max(audio_s, 1e-6):.2f}, {len(wav) / 1e6:.2f} MB)",
                  flush=True)
            return wav, self.last_gen_s, audio_s


def _mk_progress(on_step, total: int):
    """A StoppingCriteria that never stops — it just reports decode progress. Guarded so any
    transformers-version API drift degrades to 'no progress' rather than a failed render."""
    try:
        from transformers import StoppingCriteria
    except Exception:
        return None

    _every = max(1, total // 40)          # ~40 progress messages max — don't flood the control link

    class _Prog(StoppingCriteria):
        def __init__(self):
            self._n0 = None
            self._last = -10 ** 9

        def __call__(self, input_ids, scores, **kw):
            try:
                n = int(input_ids.shape[-1])
                if self._n0 is None:
                    self._n0 = n
                cur = max(0, n - self._n0)
                if cur - self._last >= _every or cur >= total:
                    self._last = cur
                    on_step(min(cur, total), total)
            except Exception:
                pass
            return False       # never stop — this criteria only reports progress

    with _suppress():
        return _Prog()
    return None
