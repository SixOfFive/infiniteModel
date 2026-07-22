"""worker_quant.py: the worker's quantization/kernel family (m4c189 code-split, Inc 10).

RELOCATED VERBATIM from client.py — every def/class body below this header is BYTE-IDENTICAL
to its client.py original; only this header is new. The span: the guarded module-level
triton/tl import + the #triton-race Autotuner serialization patch + _FUSED_INT4, the CPU
fp32-GEMM family (flags + tune_cpu_threads + _accelerate_cpu_linears), the int8/int4 quant
cores (QuantLinear/QuantLinear4), the w4a16 triton kernels, the int4 packers (Packed4Tensor3D,
_pack4_expert/_pack4_3d), fused-MoE + gpt-oss installs, the MoE offload bridge, the
per-expert/streamed builds, the meta-expert detectors, and _assign_meta_from_sd.

SELF-CONTAINED LEAF (shard_compile.py precedent): deliberately NOT in client.py's state.bind()
list — this module is the CANONICAL HOME of the quant flag family (_CPU_FP32_GEMM,
_CPU_FP32_MIN_ROWS, _CPU_BF16_GEMM_OK, _FUSED_INT4) and of the lazy kernel singletons. Those
globals are REBOUND at runtime (tune_cpu_threads, --no-cpu-fp32, the lazy builders), so nothing
may hold a from-import copy: client.py back-imports only FUNCTIONS/classes (never the flags)
and reads/writes flags as LIVE module attributes (worker_quant._CPU_FP32_GEMM = False in
main(); reads in Shard._finalize_placement). A state.bind() injection would stomp these live
values — do not add this module to bind(), and never leave a duplicate flag def in client.py.

triton/tl MUST be imported at THIS module's top level: the @triton.jit kernels resolve them
from the defining module's __globals__ (triton 3.2 does not capture closure freevars — see the
relocated comment below). torch stays function-local-lazy (a dep-minimal worker must be able to
import this module with only psutil installed). The print shadow below mirrors client.py's
timestamping shadow so the relocated bodies keep identical log formatting.

Deploy: listed in client.py's EXTRA_UPDATE_FILES + its convergence-bridge tuple (fetch-once
from GitHub raw on old checkouts, exit 42 on failure); /code_manifest picks it up via the
EXTRA_UPDATE_FILES regex (E1).
"""
from __future__ import annotations

import builtins as _builtins
import contextlib
import os
import threading
import time

import psutil


def print(*args, **kwargs):  # noqa: A001 — intentional builtin shadow for timestamping
    _builtins.print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)


# Triton imported at MODULE level (guarded) so the @triton.jit kernels below resolve `triton`/`tl`
# from module globals. They must NOT be imported as locals inside the kernel-builder functions:
# that makes `tl` a closure freevar, which triton 3.7 captures but triton 3.2 does NOT (it only
# reads __globals__) -> "NameError: tl is not defined" at compile on older triton (seen on a CUDA
# node, triton 3.2). None on CPU-only workers (no triton); the builders are GPU-only and guarded.
try:
    import triton            # noqa: F401
    import triton.language as tl
except Exception:
    triton = None
    tl = None
# #triton-race: triton's Autotuner is NOT thread-safe — run() stashes the call's args in
# `self.nargs` (read by _bench/prune during autotuning) and sets it back to None on exit, and
# mutates self.cache/best_config/configs_timings with no lock. Our @triton.autotune'd kernels
# (`_ksk` dense decode GEMV, `_mk` fused MoE) are process-wide singletons shared by EVERY shard
# on this worker, so two models decoding concurrently race: model A's run() nulls `nargs` while
# model B is mid-benchmark for a new (N,K)/(B,N,K) key -> TypeError("'NoneType' object is not a
# mapping") at autotuner._bench — a deterministic crash whenever a new shape key benches while
# ANY other int4 model decodes (hit live on om3nbox with qwen3-30b + qwen2.5-vl:3b resident).
# Fix: serialize Autotuner.run with ONE process-wide lock. Steady-state cost is a lock acquire
# per int4 GEMM launch (~100ns vs ms-scale decode — the GPU executes async either way); during a
# bench window other int4 launches briefly WAIT instead of crashing. RLock in case a pre/post
# hook re-enters an autotuned kernel. Idempotent (guarded by _im_serialized) so a re-imported
# module never double-wraps.
if triton is not None:
    try:
        from triton.runtime.autotuner import Autotuner as _TritonAutotuner
        if not getattr(_TritonAutotuner, "_im_serialized", False):
            _TRITON_RUN_LOCK = threading.RLock()
            _tt_orig_run = _TritonAutotuner.run

            def _tt_locked_run(self, *a, **k):
                with _TRITON_RUN_LOCK:
                    return _tt_orig_run(self, *a, **k)
            _TritonAutotuner.run = _tt_locked_run
            _TritonAutotuner._im_serialized = True
    except Exception:
        pass                 # unexpected triton layout: keep the unpatched behavior
# Fused-dequant int4 GEMM (torch tinygemm _weight_int4pack_mm): ~3.6x faster int4 decode by
# dequantizing INSIDE the matmul instead of re-expanding the whole weight every token. Built per
# QuantLinear4 at placement, self-checked vs the naive dequant, naive fallback on any mismatch /
# unsupported device. Off-switch: IM_FUSED_INT4=0.
_FUSED_INT4 = (os.environ.get("IM_FUSED_INT4", "1") != "0")
# Fused-dequant int2 GEMM (Triton w2a16 — there is NO torch tinygemm for 2-bit): same contract
# as _FUSED_INT4 but for QuantLinear2; the one Triton kernel serves BOTH CUDA and ROCm (no-triton
# workers self-gate to the naive dequant path). Off-switch: IM_FUSED_INT2=0.
_FUSED_INT2 = (os.environ.get("IM_FUSED_INT2", "1") != "0")
# #large-m-naive: the fused Triton quant linears' M>1 kernels are DECODE-tuned (fixed BM=16
# tl.dot tiles, no autotune) — at prefill row counts they can lose to "dequant the weight ONCE
# + one BLAS GEMM" (the naive path, remat amortized over the whole chunk). Which side wins is
# shape/arch-dependent, so it is MEASURED per shape at prepare time (_bench_large_m_naive) and
# forward falls through above the benched threshold. Off-switch: IM_LARGE_M_NAIVE=0.
_LARGE_M_NAIVE = (os.environ.get("IM_LARGE_M_NAIVE", "1") != "0")


# ---------------------------------------------------------------------------
# CPU matmul acceleration (fp32 GEMM + thread tuning).
#
# THE PROBLEM. PyTorch CPU has NO fast bf16 GEMM — a bf16 matmul on CPU runs
# near-scalar (no MKL/OpenBLAS bf16 kernel), 1-2 orders of magnitude slower than
# the same matmul in fp32 (which hits a vectorized MKL/OpenBLAS GEMM). Our
# activations flow at self.dtype (bf16), so every CPU Linear — both the model's
# native nn.Linear and our QuantLinear (which dequants int8/int4 -> bf16) — pays
# the slow bf16 path. On a CPU-only or hybrid-spilled shard this makes a 70B int8
# produce ~0 tok/min. GPU is unaffected (real tensor-core bf16/int kernels).
#
# THE FIX. For a CPU-resident matmul, do a TRANSIENT fp32 upcast IN the forward:
# cast the activation (and the dequantized/transiently-upcast weight) to fp32,
# run F.linear in fp32 (fast GEMM), cast the result back to the working dtype.
# The resident weight STAYS bf16/int (the fp32 copy is per-call and freed after),
# so resident RAM is unchanged — the whole point of the fleet (fit big models).
#
# WHY SHAPE-ADAPTIVE (the threshold). Measured on this fleet's CPUs: the fp32
# win is huge for compute-bound matmuls (prefill, M=batch*seq large: 3-5x) but
# REVERSES for tiny-M single-token decode (M=1): a GEMV is memory-bandwidth
# bound, so reading the 2x-larger fp32 weight (incl. the one-time upcast/dequant
# of the whole weight) costs more than the slow-but-small bf16 GEMV saves. The
# crossover is ~12-16 rows. So we only upcast when the activation has at least
# _CPU_FP32_MIN_ROWS rows; below that the native bf16 path is kept (it's faster).
# This keeps decode fast AND makes prefill/large-batch fast — best of both.
# ---------------------------------------------------------------------------

# Min activation rows (flattened batch*seq) before a CPU matmul is worth the fp32
# upcast. Below this, bf16 GEMV is faster (memory-bound); above, fp32 GEMM wins.
# Conservative (16) so we never regress decode; prefill is always far above it.
_CPU_FP32_MIN_ROWS = 16
# Master switch (set False via --no-cpu-fp32 to A/B the old bf16 path).
_CPU_FP32_GEMM = True

# Some CPUs cannot run a NATIVE bf16 GEMM at all. On aarch64 without the ARM BF16 ISA
# extension (FEAT_BF16 — absent on e.g. Cortex-A55/A75, the Unisoc T310 tablet) PyTorch's
# oneDNN bf16 matmul does NOT fall back to fp32 the way x86 does; it raises
#   "mkldnn_matmul bf16 path needs a cpu with bf16 support" (ATen/native/mkldnn/Matmul.cpp).
# There the row-gated "keep bf16 for tiny-M decode" path above is not just slow — it CRASHES
# every decode step (rows=1), corrupting the shard's activations into garbage tokens. We probe
# this once at startup and, if unsupported, force the fp32 upcast for ALL CPU matmuls
# (_CPU_FP32_MIN_ROWS -> 1). No-op on x86, where the bf16 GEMM succeeds (just slower).
_CPU_BF16_GEMM_OK = None


def _cpu_bf16_gemm_ok() -> bool:
    """True iff this CPU's bf16 Linear is trustworthy for inference (probed once, cached).
    A CPU is rejected if a decode-shaped biased bf16 linear (addmm path) (a) RAISES — the
    aarch64-w/o-FEAT_BF16 hard 'bf16 path needs a cpu with bf16 support' check; (b) emits the
    oneDNN 'mkldnn_matmul failed, switching to ...' fallback WARNING — the tablet hits this and
    its fallback, while not crashing, is numerically degraded enough to derail a model; or
    (c) returns a result that diverges from the fp32 reference. Prints the probe result so the
    decision is visible at startup."""
    global _CPU_BF16_GEMM_OK
    if _CPU_BF16_GEMM_OK is None:
        _CPU_BF16_GEMM_OK = True
        try:
            import torch
            import warnings as _w
            g = torch.Generator().manual_seed(0)
            x = torch.randn(1, 512, generator=g)          # decode shape: rows=1, 2-D
            w = torch.randn(512, 512, generator=g)
            b = torch.randn(512, generator=g)
            ref = torch.nn.functional.linear(x, w, b)      # fp32 addmm (reference)
            with _w.catch_warnings(record=True) as wl:
                _w.simplefilter("always")
                got = torch.nn.functional.linear(          # bf16 addmm (the path under test)
                    x.bfloat16(), w.bfloat16(), b.bfloat16()).float()
            warned = any(("mkldnn" in str(m.message).lower()
                          or "bf16" in str(m.message).lower()) for m in wl)
            finite = bool(torch.isfinite(got).all())
            rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
            if warned or not finite or rel > 0.05:
                _CPU_BF16_GEMM_OK = False
            with contextlib.suppress(Exception):
                print(f"[cpu] bf16 GEMM probe: ok={_CPU_BF16_GEMM_OK} "
                      f"warned={warned} finite={finite} rel_err={rel:.4f}")
        except Exception as e:                             # raised -> definitely unusable
            _CPU_BF16_GEMM_OK = False
            with contextlib.suppress(Exception):
                print(f"[cpu] bf16 GEMM probe RAISED ({type(e).__name__}) -> forcing fp32")
    return _CPU_BF16_GEMM_OK


def _rows(x) -> int:
    """Flattened row count (batch*seq) of an activation [*, in_features]."""
    n = 1
    for d in x.shape[:-1]:
        n *= int(d)
    return n


def _m_bucket(m: int) -> int:
    """#m-bucket: round a Triton-launch row count UP to its compile bucket — the next power of
    two (0/1 pass through; decode M=1 has its own GEMV kernel). Every distinct M presented to a
    @triton.jit kernel is fresh JIT-specialization/autotune surface, and on a slow-compile box
    (gfx1151 APU) a request stream of arbitrary prompt lengths turned that into 20-103s stalls
    per NOVEL length (#large-m-naive live regression, 2026-07-21). Padding rows up to the bucket
    and slicing them back off caps the space at ~log2(chunk) shapes TOTAL (M=1..64 ->
    {2,4,8,16,32,64}: six), costs <2x rows worst-case on the padded call, and is numerically
    exact — each output row of these GEMM/GEMV kernels depends only on its own input row, so
    zero pad rows never touch real rows (same argument as the existing zero K-padding)."""
    if m <= 1:
        return m
    b = 2
    while b < m:
        b <<= 1
    return b


def _cpu_fp32_worth(x) -> bool:
    """True when x is on CPU and big enough that an fp32 GEMM beats the bf16 path."""
    return (_CPU_FP32_GEMM and x.device.type == "cpu"
            and x.dtype != _torch_float32() and _rows(x) >= _CPU_FP32_MIN_ROWS)


_TORCH_F32 = None


def _torch_float32():
    global _TORCH_F32
    if _TORCH_F32 is None:
        import torch
        _TORCH_F32 = torch.float32
    return _TORCH_F32


def tune_cpu_threads() -> None:
    """Once, at process start: pin PyTorch CPU intra-op threads to the PHYSICAL core
    count (hyperthreads add overhead, not throughput, for GEMM-bound matmul). No-op
    if torch is absent. Honors a pre-set OMP_NUM_THREADS/MKL_NUM_THREADS (user/operator
    override) by not lowering an explicit env choice. Clamped to a sane range."""
    global _CPU_FP32_MIN_ROWS
    try:
        import torch
    except Exception:
        return
    # Respect an explicit operator override via env (don't stomp a deliberate choice).
    env_override = None
    for ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
        v = os.environ.get(ev)
        if v and v.strip().isdigit():
            env_override = int(v.strip())
            break
    if env_override:
        n = env_override
    else:
        try:
            n = psutil.cpu_count(logical=False) or 0
        except Exception:
            n = 0
        if not n:
            n = os.cpu_count() or 1
    n = max(1, min(int(n), 128))   # clamp: never 0, never absurd
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    # interop (inter-op task parallelism) gains nothing for a single sequential
    # forward and oversubscribes cores; keep it small. Must be set before any
    # parallel work starts, so wrap in suppress (raises if the pool is already up).
    with contextlib.suppress(Exception):
        torch.set_num_interop_threads(min(2, n))
    # A CPU that can't run a native bf16 GEMM (e.g. aarch64 w/o FEAT_BF16) would CRASH on the
    # tiny-M decode path rather than run it slow; force the fp32 upcast for EVERY CPU matmul
    # there (rows >= 1) so decode stays correct. No-op on x86. See _cpu_bf16_gemm_ok.
    if _CPU_FP32_GEMM and not _cpu_bf16_gemm_ok():
        _CPU_FP32_MIN_ROWS = 1
        with contextlib.suppress(Exception):
            print("[cpu] native bf16 GEMM unsupported here (aarch64 w/o bf16 ISA?) - "
                  "forcing fp32 upcast for ALL CPU matmuls")
    with contextlib.suppress(Exception):
        print(f"[cpu] torch intra-op threads set to {torch.get_num_threads()} "
              f"(physical cores{'/env' if env_override else ''}); fp32 CPU GEMM "
              f"{'on' if _CPU_FP32_GEMM else 'off'} (min {_CPU_FP32_MIN_ROWS} rows)")


def _wrap_cpu_linear_fp32(lin) -> None:
    """Wrap a CPU-resident native nn.Linear's forward so its matmul runs in fp32.
    The bf16 weight stays resident (source of truth); per call we transiently upcast
    activation+weight to fp32, F.linear (fast MKL/OpenBLAS GEMM), cast back. Only
    fires for compute-bound shapes (>= _CPU_FP32_MIN_ROWS rows); tiny-M decode falls
    through to the original bf16 forward (faster — see module header). Idempotent."""
    import torch
    import torch.nn.functional as F
    if getattr(lin, "_im_fp32_wrapped", False):
        return
    orig_forward = lin.forward

    def fp32_forward(x):
        w = lin.weight
        if (_CPU_FP32_GEMM and w.device.type == "cpu" and x.device.type == "cpu"
                and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
            out_dtype = x.dtype
            b = lin.bias
            y = F.linear(x.to(torch.float32), w.to(torch.float32),
                         None if b is None else b.to(torch.float32))
            return y.to(out_dtype)
        return orig_forward(x)

    lin.forward = fp32_forward
    lin._im_fp32_wrapped = True


def _accelerate_cpu_linears(module) -> None:
    """Wrap every CPU-resident native nn.Linear under `module` for the fp32 GEMM path.
    Skips GPU-resident Linears (already fast tensor-core kernels — must stay untouched)
    and our QuantLinear/QuantLinear4/Packed4Tensor3D (they handle fp32 in their own
    dequant). Called per-module from _place_modules AFTER placement, so it sees the
    final device of each weight — composing automatically with hybrid cpu+gpu spill
    and TP-v2 reduced-dim modules (it wraps whatever Linears ended up on CPU)."""
    if module is None:
        return
    from torch import nn
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            try:
                on_cpu = sub.weight.device.type == "cpu"
            except Exception:
                on_cpu = False
            if on_cpu:
                _wrap_cpu_linear_fp32(sub)


# ---------------------------------------------------------------------------
# int8 weight-only quantization (opt-in, --quant int8). Halves the weight
# footprint (RAM + VRAM) so bigger models fit. Per-output-channel symmetric
# int8; the original dtype weight is reconstructed on the fly in forward
# (qweight.to(dtype) * scale), so it's a memory win, not a speed win — for a
# model that already fits, prefer bf16. Self-contained: no external quant libs.
# ---------------------------------------------------------------------------

_QUANT_LINEAR = None


def _quant_linear_cls():
    global _QUANT_LINEAR
    if _QUANT_LINEAR is None:
        import torch
        from torch import nn
        import torch.nn.functional as F

        class QuantLinear(nn.Module):
            def __init__(self, qweight, scale, bias):
                super().__init__()
                self.register_buffer("qweight", qweight)   # int8 [out, in]
                self.register_buffer("scale", scale)        # dtype [out, 1]
                self.bias = bias                             # Parameter or None

            def forward(self, x):
                # CPU compute-bound path: dequant DIRECTLY to fp32 and run a fast fp32
                # GEMM (the dequant is paid either way, so targeting fp32 is free here and
                # the GEMM is the win). Tiny-M decode (or any GPU tensor) keeps the original
                # dequant-to-x.dtype path — bf16 GEMV is faster there (memory-bound). See the
                # CPU-matmul module header for the threshold rationale.
                if (_CPU_FP32_GEMM and x.device.type == "cpu"
                        and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
                    w = self.qweight.to(torch.float32) * self.scale.to(torch.float32)
                    b = self.bias
                    y = F.linear(x.to(torch.float32), w,
                                 None if b is None else b.to(torch.float32))
                    return y.to(x.dtype)
                w = self.qweight.to(x.dtype) * self.scale    # dequant one weight matrix
                return F.linear(x, w, self.bias)

        _QUANT_LINEAR = QuantLinear
    return _QUANT_LINEAR


def _quantize_linear(lin):
    """nn.Linear -> int8 weight-only QuantLinear (per-output-channel scale)."""
    import torch
    QL = _quant_linear_cls()
    W = lin.weight.data
    scale = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    qW = (W / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return QL(qW, scale.to(W.dtype), lin.bias)


def _quantize_int8_(module) -> None:
    """Recursively replace every nn.Linear under `module` with a QuantLinear."""
    from torch import nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear(child))
        else:
            _quantize_int8_(child)


# ---------------------------------------------------------------------------
# int4 weight-only quantization (opt-in, --quant int4). ~4.25 bits/weight
# (2 nibbles/byte + group scale/zero), so a model takes ~1/4 the RAM/VRAM of
# bf16 — the route for 200B+ MoEs that won't fit even at int8. Group-wise
# ASYMMETRIC (per-output-channel, group_size=128): w ~= (q - zero) * scale.
# The whole matrix is dequantized to the activation dtype on the fly in
# forward (memory win, not speed). CPU+GPU, pure torch, no external quant libs.
# The lm_head is left bf16 (logit-sensitive); only decoder Linears are quantized.
# ---------------------------------------------------------------------------

_QUANT_LINEAR4 = None
_INT4_GROUP = 128


def _quant4_linear_cls():
    global _QUANT_LINEAR4
    if _QUANT_LINEAR4 is None:
        import torch
        from torch import nn
        import torch.nn.functional as F

        class QuantLinear4(nn.Module):
            def __init__(self, qweight, scale, zero, bias, in_features, group_size):
                super().__init__()
                self.register_buffer("qweight", qweight)  # uint8 [out, in_pad//2] (2 nibbles/byte)
                self.register_buffer("scale", scale)       # dtype [out, n_groups]
                self.register_buffer("zero", zero)         # dtype [out, n_groups]
                self.bias = bias                            # Parameter or None (bf16)
                self.in_features = in_features
                self.group_size = group_size

            def _dequant(self, dtype):
                qw = self.qweight
                out = qw.shape[0]
                lo = (qw & 0x0F).to(torch.int16)           # even input columns
                hi = (qw >> 4).to(torch.int16)             # odd input columns
                q = torch.stack((lo, hi), dim=2).reshape(out, -1)   # [out, in_pad]
                ng = self.scale.shape[1]
                G = self.group_size
                qf = q.reshape(out, ng, G).to(dtype)
                w = (qf - self.zero.to(dtype).unsqueeze(2)) * self.scale.to(dtype).unsqueeze(2)
                return w.reshape(out, ng * G)[:, :self.in_features].contiguous()

            def prepare_fused(self):
                # Build torch's tinygemm fused-int4 weight ONCE (now that this module is on its FINAL
                # device). Decode re-dequants the whole weight every token in the naive path — the 3.6x
                # int4 slowdown; the fused op dequants inside the GEMM. Device-gated (CUDA sm80+ / CPU
                # op present), converted from OUR group-wise asymmetric format with NO re-quant
                # (w=(q-zero)*S == kernel's (q-8)*S + (8-zero)*S), then SELF-CHECKED vs the naive
                # dequant on a random input. Any mismatch / unsupported device / error -> keep naive.
                # Frees qweight on success (the packed mat2 replaces it -> int4 memory stays flat).
                if not _FUSED_INT4 or getattr(self, "_fused", None) is not None \
                        or getattr(self, "_fused_tried", False) or self.qweight is None:
                    return
                self._fused_tried = True
                dev = self.qweight.device
                aten = torch.ops.aten
                # ROCm (AMD): torch's _weight_int4pack_mm is CDNA2+-only, so on RDNA (gfx1151) the
                # naive path rematerializes the whole bf16 weight per token (~5-20x slower). Use the
                # Triton w4a16 GEMM that reads int4 directly. Self-checked vs naive; falls back on
                # mismatch/unavailable. NVIDIA + CPU keep the torch tinygemm path below, untouched.
                if dev.type == "cuda" and getattr(torch.version, "hip", None):
                    op = _w4a16_triton_op()
                    if op is None:
                        return
                    try:
                        G = self.group_size
                        ng = self.scale.shape[1]
                        in_pad = ng * G
                        # #dram-dealias: the GEMV walks qweight along N with a row stride of K/2
                        # bytes. When that stride is an EVEN multiple of 64B (worst case a power of
                        # two — llama-70b's K=8192 -> 4096B) every row lands on the same DRAM
                        # channels/banks, and any weight too big for the 32MB MALL collapses to
                        # ~17-67 GB/s (llama-3.3-70b decoded 0.61 tok/s). Re-allocating rows on an
                        # ODD multiple of 64B restores 130-210 GB/s (bench_w4a16 matrix, gfx1151).
                        # The kernels read via qweight.stride(0), so a padded VIEW needs no other
                        # change; cost is 64B/row (~1% for the mats this matters to) and never hurts
                        # the aligned case (32b shapes got faster too).
                        qw = self.qweight
                        if qw.shape[1] % 128 == 0:
                            buf = torch.zeros((qw.shape[0], qw.shape[1] + 64),
                                              dtype=torch.uint8, device=dev)
                            buf[:, :qw.shape[1]].copy_(qw)
                            self.qweight = buf[:, :qw.shape[1]]
                        sz = (self.scale, self.zero)
                        xt = torch.randn(8, self.in_features, device=dev, dtype=torch.bfloat16)
                        xk = xt if in_pad == self.in_features else F.pad(xt, (0, in_pad - self.in_features))
                        yf = op(xk.contiguous(), self.qweight, G, sz).float()
                        yn = F.linear(xt, self._dequant(torch.bfloat16)).float()
                        rel = ((yf - yn).abs().mean() / (yn.abs().mean() + 1e-6)).item()
                        if rel < 0.05:
                            self._fused = (self.qweight, sz, op, in_pad)   # kernel reads qweight; keep it
                            print(f"[int4] triton w4a16 kernel active on {dev}", flush=True)
                            # #large-m-naive: fused is now proven for correctness; bench whether
                            # prefill-shaped M should fall through to dequant-once+BLAS instead
                            # of the decode-tuned _k (threshold 0 = never; cached per shape).
                            self._naive_m_min = _bench_large_m_naive(self, "int4")
                        else:
                            print(f"[int4] triton w4a16 self-check rel={rel:.3f} on {dev} -> naive", flush=True)
                    except Exception as exc:
                        print(f"[int4] triton w4a16 prepare failed on {dev} ({exc!r}) -> naive", flush=True)
                    return
                if dev.type == "cuda":
                    try:
                        ok = (torch.cuda.get_device_capability(dev) >= (8, 0)
                              and hasattr(aten, "_weight_int4pack_mm")
                              and hasattr(aten, "_convert_weight_to_int4pack"))
                    except Exception:
                        ok = False
                elif dev.type == "cpu":
                    ok = (hasattr(aten, "_weight_int4pack_mm_for_cpu")
                          and hasattr(aten, "_convert_weight_to_int4pack_for_cpu"))
                else:
                    ok = False
                if not ok:
                    return
                try:
                    qw = self.qweight
                    out = qw.shape[0]
                    G = self.group_size
                    ng = self.scale.shape[1]
                    in_pad = ng * G
                    q = torch.stack(((qw & 0x0F), (qw >> 4)), dim=2).reshape(out, in_pad)  # 0..15
                    S = self.scale.float()                                  # [out, ng]
                    Z = (8.0 - self.zero.float()) * S                       # int-zero -> float midpoint
                    sz = torch.cat([S.reshape(out, ng, 1), Z.reshape(out, ng, 1)], dim=2)
                    sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)  # [ng, out, 2]
                    if dev.type == "cuda":
                        # CUDA pack wants HIGH-nibble = even col (ours is LOW-even) -> re-nibble
                        packed = ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8).contiguous()
                        mat2 = aten._convert_weight_to_int4pack(packed, 8)   # innerKTiles=8
                        op = aten._weight_int4pack_mm
                        sz = sz.to(dev)
                    else:
                        mat2 = aten._convert_weight_to_int4pack_for_cpu(
                            q.to(torch.int32).contiguous(), 8)
                        op = aten._weight_int4pack_mm_for_cpu
                    # self-check: fused vs naive on a random input (catches zero-point/nibble/pack bugs
                    # that would silently corrupt logits — they are NOT exceptions).
                    xt = torch.randn(4, self.in_features, device=dev, dtype=torch.bfloat16)
                    xk = xt if in_pad == self.in_features else F.pad(xt, (0, in_pad - self.in_features))
                    yf = op(xk, mat2, G, sz).float()
                    yn = F.linear(xt, self._dequant(torch.bfloat16)).float()
                    rel = ((yf - yn).abs().mean() / (yn.abs().mean() + 1e-6)).item()
                    if rel < 0.05:
                        self._fused = (mat2, sz, op, in_pad)
                        self.qweight = None        # packed mat2 is now authoritative; free the source
                        # #sz-free: scale/zero live on ONLY inside the fused sz tensor now (the
                        # [ng,out,2] bf16 repack above) — the originals are dead weight, ~6% of
                        # qweight bytes held TWICE (~0.4-0.5 GB on a 14 GB dense-int4 shard set).
                        # Safe to drop: with qweight None both _dequant callers are unreachable
                        # (forward's naive tail is gated behind the fused branch + the
                        # #large-m-naive gate requires qweight; prepare_fused re-entry is
                        # _fused_tried/qweight-None guarded), and a None buffer is already the
                        # shipped qweight=None pattern — nn.Module buffers()/state_dict()/_apply
                        # skip None entries and the teardown sweeps None-guard (worker_load
                        # _release_shard_vram, worker_t2i offload, client [int4-vram] census).
                        # ROCm never gets here (early return above) — its sz TUPLE aliases these
                        # live buffers and the Triton kernel reads them every call: keep BOTH.
                        self.scale = None
                        self.zero = None
                    else:
                        print(f"[int4] fused self-check rel={rel:.3f} on {dev} -> naive path",
                              flush=True)
                except Exception as exc:
                    print(f"[int4] fused prepare failed on {dev} ({exc!r}) -> naive path", flush=True)

            def forward(self, x):
                fz = getattr(self, "_fused", None)
                # #large-m-naive + #m-bucket: calls whose ROW BUCKET (the shape the fused Triton
                # kernel would actually execute after _m_bucket padding) reaches the prepare-
                # time-benched threshold skip the decode-tuned kernel and take the naive tail
                # below — dequant ONCE + one BLAS GEMM, i.e. the self-check's own reference
                # numerics. Bucket-compare, not raw rows: a 1500-row chunk tail pads up to the
                # same 2048 bucket the bench measured, so it must fall through with it (naive
                # cost is monotonic in M, so naive winning at the bucket covers every M the
                # bucket serves). Triton-only by construction: needs the retained qweight
                # (tinygemm frees it and never sets a threshold), and decode M=1 can never
                # reach the threshold.
                _nm = getattr(self, "_naive_m_min", 0)
                if fz is not None and _nm and self.qweight is not None \
                        and _m_bucket(_rows(x)) >= _nm:
                    fz = None
                if fz is not None:
                    # fused-dequant int4 GEMM (2D only): flatten, bf16 activations, restore shape.
                    mat2, sz, op, in_pad = fz
                    xq = x.reshape(-1, self.in_features)
                    if xq.dtype != torch.bfloat16:
                        xq = xq.to(torch.bfloat16)
                    if in_pad != self.in_features:
                        xq = F.pad(xq, (0, in_pad - self.in_features))
                    y = op(xq.contiguous(), mat2, self.group_size, sz).reshape(*x.shape[:-1], -1)
                    y = y.to(x.dtype)
                    return y if self.bias is None else y + self.bias.to(y.dtype)
                # naive fallback: CPU compute-bound path dequants to fp32 + fp32 GEMM (see QuantLinear
                # / CPU-matmul header); int4 unpack+dequant is paid regardless, so the fp32 weight is
                # free and the fp32 GEMM is the win. Decode/GPU keep x.dtype.
                if (_CPU_FP32_GEMM and x.device.type == "cpu"
                        and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
                    b = self.bias
                    y = F.linear(x.to(torch.float32), self._dequant(torch.float32),
                                 None if b is None else b.to(torch.float32))
                    return y.to(x.dtype)
                return F.linear(x, self._dequant(x.dtype), self.bias)

        _QUANT_LINEAR4 = QuantLinear4
    return _QUANT_LINEAR4


# --- Triton w4a16 int4 GEMM (ROCm fast int4 decode) ------------------------------------------
# torch's fused int4 (_weight_int4pack_mm) is CDNA2+-only on ROCm, so on RDNA (e.g. AMD Strix
# Halo gfx1151) int4 decode falls back to the naive path that rematerializes the whole bf16
# weight every token (GPU-bound, ~5-20x slower). This Triton kernel reads the packed int4
# weight and dequantizes INSIDE the GEMM, in the worker's exact group-wise asymmetric format:
# qweight uint8 [N, K//2] (byte j -> col 2j low nibble / 2j+1 high nibble), scale/zero bf16
# [N, K//group], w=(q-zero)*scale per group. Bit-identical to the naive path (self-checked in
# prepare_fused). Lazily built on first use; ROCm-only — never touches the NVIDIA/CPU paths.
_W4A16_OP = None
_W4A16_TRIED = False
# #triton-race: one lock for all three lazy kernel/class builders below. The old pattern set
# _TRIED=True BEFORE building — a second shard-install thread arriving mid-build saw
# (_TRIED=True, _OP=None) and captured the naive 5-20x-slower path PERMANENTLY (ops are bound at
# prepare time). Under the lock, _TRIED flips only after _OP is final, and a waiting racer
# re-checks and returns the finished op.
_W4A16_BUILD_LOCK = threading.RLock()   # RLock: _w4a16_expert_cls builds INSIDE it and calls
                                        # _w4a16_triton_op, which takes the same lock (re-entry)


def _w4a16_triton_op():
    """Callable op(x[M,Kpad] bf16, qweight uint8[N,Kpad//2], group, (scale,zero) bf16[N,ng]) ->
    y[M,N] bf16, or None if triton is unavailable / fails to build. Thread-safe lazy build."""
    global _W4A16_OP, _W4A16_TRIED
    if _W4A16_TRIED:
        return _W4A16_OP
    with _W4A16_BUILD_LOCK:
        return _w4a16_triton_op_locked()


def _w4a16_triton_op_locked():
    global _W4A16_OP, _W4A16_TRIED
    if _W4A16_TRIED:         # a racer built it while we waited on the lock
        return _W4A16_OP
    try:
        import torch
        if triton is None:                # module-level import (see top); None on no-triton workers
            raise ImportError("triton unavailable")

        @triton.jit
        def _k(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, M, N, K,
               sxm, sxk, sqk, sqn, ssn, ssg, szn, szg, sym, syn,
               GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BM + tl.arange(0, BM)
            offs_n = pid_n * BN + tl.arange(0, BN)
            offs_h = tl.arange(0, GROUP // 2)            # byte index within a K-group
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for kb in range(0, K // GROUP):
                k0 = kb * GROUP
                mm = offs_m[:, None] < M
                xe = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 2 * offs_h)[None, :] * sxk,
                             mask=mm, other=0.0)
                xo = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 2 * offs_h + 1)[None, :] * sxk,
                             mask=mm, other=0.0)
                qp = q_ptr + (k0 // 2 + offs_h)[:, None] * sqk + offs_n[None, :] * sqn
                b = tl.load(qp, mask=offs_n[None, :] < N, other=0).to(tl.int32)
                lo = (b & 0xF).to(tl.float32)
                hi = ((b >> 4) & 0xF).to(tl.float32)
                s = tl.load(s_ptr + offs_n * ssn + kb * ssg, mask=offs_n < N, other=0.0).to(tl.float32)
                z = tl.load(z_ptr + offs_n * szn + kb * szg, mask=offs_n < N, other=0.0).to(tl.float32)
                wlo = ((lo - z[None, :]) * s[None, :]).to(tl.bfloat16)
                whi = ((hi - z[None, :]) * s[None, :]).to(tl.bfloat16)
                acc += tl.dot(xe.to(tl.bfloat16), wlo)
                acc += tl.dot(xo.to(tl.bfloat16), whi)
            yp = y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn
            tl.store(yp, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

        # DECODE (M=1) GEMV — split over K for occupancy. The tl.dot kernel above launches only
        # ~cdiv(N,BN) programs at M=1, far too few to hide memory latency on the iGPU (~13-50% of
        # peak BW measured). This splits K across SPLITK programs per N-block (grid grows ~SPLITKx)
        # and atomic-adds partials into an fp32 accumulator -> 3.5-3.9x on the dense GEMV (bench).
        # num_warps is the dominant knob for a BW-bound GEMV (more warps = more in-flight loads to
        # hide memory latency); RDNA/iGPU often wants more than sm_89, so sweep {4,8} x SPLITK and
        # let autotune pick per (N,K)/arch. SPLITK=4/w4 == the prior default, so never worse.
        # #dram-dealias: the BN=64 / num_warps=16 configs are what the 70B dims (N 8192-28672,
        # K 8192/28672) want once the row stride is de-aliased (see prepare_fused) — measured
        # 0.67ms vs 1.94ms on the 28672x8192 gate/up with the old space (gfx1151 matrix bench).
        # reset_to_zero: this kernel atomic-adds into y_ptr, so autotune's timing reruns would
        # accumulate into the same buffer and corrupt the first call for each (N,K) — zero it per
        # launch. (Was missing before; first-token-per-shape corruption was masked by the load-time
        # self-check absorbing it.)
        @triton.autotune(
            configs=[triton.Config({"BN": 128, "SPLITK": s}, num_warps=w)
                     for s in (4, 8, 16) for w in (4, 8)]
                    + [triton.Config({"BN": bn, "SPLITK": s}, num_warps=w)
                       for (bn, s, w) in ((64, 4, 8), (64, 4, 16), (64, 8, 16), (64, 16, 16),
                                          (64, 32, 16), (128, 4, 16), (128, 8, 16))],
            key=["N", "K"],
            reset_to_zero=["y_ptr"],
        )
        @triton.jit
        def _ksk(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, N, K,
                 sxk, sqk, sqn, ssn, ssg, szn, szg, syn,
                 GROUP: tl.constexpr, BN: tl.constexpr, SPLITK: tl.constexpr):
            pid_n = tl.program_id(0)
            pid_k = tl.program_id(1)
            offs_n = pid_n * BN + tl.arange(0, BN)
            nmask = offs_n < N
            offs_h = tl.arange(0, GROUP // 2)
            ngroups = K // GROUP
            gps = (ngroups + SPLITK - 1) // SPLITK       # K-groups this split reduces
            g0 = pid_k * gps
            acc = tl.zeros((BN,), dtype=tl.float32)
            for gi in range(0, gps):
                kb = g0 + gi
                if kb < ngroups:
                    k0 = kb * GROUP
                    xe = tl.load(x_ptr + (k0 + 2 * offs_h) * sxk)
                    xo = tl.load(x_ptr + (k0 + 2 * offs_h + 1) * sxk)
                    qp = q_ptr + (k0 // 2 + offs_h)[:, None] * sqk + offs_n[None, :] * sqn
                    b = tl.load(qp, mask=nmask[None, :], other=0).to(tl.int32)
                    lo = (b & 0xF).to(tl.float32)
                    hi = ((b >> 4) & 0xF).to(tl.float32)
                    s = tl.load(s_ptr + offs_n * ssn + kb * ssg, mask=nmask, other=0.0).to(tl.float32)
                    z = tl.load(z_ptr + offs_n * szn + kb * szg, mask=nmask, other=0.0).to(tl.float32)
                    acc += tl.sum(xe[:, None] * ((lo - z[None, :]) * s[None, :]), axis=0)
                    acc += tl.sum(xo[:, None] * ((hi - z[None, :]) * s[None, :]), axis=0)
            tl.atomic_add(y_ptr + offs_n * syn, acc, mask=nmask)

        def _op(x, qweight, group_size, sz):
            scale, zero = sz
            if x.dim() != 2:
                x = x.reshape(-1, x.shape[-1])
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            x = x.contiguous()
            Kpad = qweight.shape[1] * 2                  # pad activations to the packed width
            if x.shape[1] != Kpad:                       # (no-op for QuantLinear4, which pre-pads;
                import torch.nn.functional as _F          #  used by the MoE expert path)
                x = _F.pad(x, (0, Kpad - x.shape[1]))
            M, K = x.shape
            N = qweight.shape[0]
            if M == 1:                                   # decode: split-K GEMV (occupancy) -> fp32 acc
                yf = torch.zeros((N,), device=x.device, dtype=torch.float32)
                grid = lambda meta: (triton.cdiv(N, meta["BN"]), meta["SPLITK"])  # noqa: E731
                _ksk[grid](x.view(-1), qweight, scale, zero, yf, N, K,
                           x.stride(1), qweight.stride(1), qweight.stride(0),
                           scale.stride(0), scale.stride(1), zero.stride(0), zero.stride(1),
                           yf.stride(0), GROUP=group_size)
                return yf.to(torch.bfloat16).view(1, N)
            # #m-bucket: pad rows up to the power-of-two bucket so this decode-tuned tl.dot
            # kernel is only ever launched (and JIT-specialized) at ~log2(chunk) distinct M
            # shapes — every novel prompt length otherwise presents a fresh M to Triton (seen
            # live on gfx1151 as tens of seconds per NOVEL length). Zero pad rows are exact
            # (row-independent dot products) and sliced back off below.
            Mp = _m_bucket(M)
            if Mp != M:
                import torch.nn.functional as _F
                x = _F.pad(x, (0, 0, 0, Mp - M))
            y = torch.empty((Mp, N), device=x.device, dtype=torch.bfloat16)
            BM, BN = 16, 128
            grid = (triton.cdiv(Mp, BM), triton.cdiv(N, BN))
            _k[grid](x, qweight, scale, zero, y, Mp, N, K,
                     x.stride(0), x.stride(1), qweight.stride(1), qweight.stride(0),
                     scale.stride(0), scale.stride(1), zero.stride(0), zero.stride(1),
                     y.stride(0), y.stride(1), GROUP=group_size, BM=BM, BN=BN)
            return y if Mp == M else y[:M]

        _W4A16_OP = _op
        _builtins.print("[int4] triton w4a16 kernel built (ROCm fast int4)", flush=True)
    except Exception as exc:
        _builtins.print(f"[int4] triton w4a16 unavailable ({exc!r}) -> naive int4", flush=True)
        _W4A16_OP = None
    _W4A16_TRIED = True      # #triton-race: only AFTER _W4A16_OP is final (see _W4A16_BUILD_LOCK)
    return _W4A16_OP


# #large-m-naive: measured large-M dispatch decisions, keyed (tag, N, in_pad, group) so 40
# layers of the same shape pay ONE bench (the _pad_choice precedent). Value = the benched
# production-chunk bucket when the naive path wins there (forward falls through when a call's
# ROW BUCKET — _m_bucket(_rows(x)) — reaches it); 0 = keep fused everywhere.
_LARGE_M_CHOICE: dict = {}


def _prefill_bench_m() -> int:
    """#large-m-naive + #m-bucket: the ONE row count the fused side is benched at — the
    _m_bucket of the production prefill chunk (INFINITEMODEL_PREFILL_CHUNK, default 2048;
    0 = chunking disabled -> keep 2048 as the whole-prompt proxy). Production compiles this
    exact bucket anyway on its first full chunk, so benching it triggers NO bench-only Triton
    shapes — unlike the old (64, 256, 2048) sweep, whose M=64/256 probes JIT-specialized
    shapes at EVERY load that the request stream might never present (+66s/load observed on
    gfx1151, 2026-07-21)."""
    try:
        c = int(os.environ.get("INFINITEMODEL_PREFILL_CHUNK", "2048") or "2048")
    except (TypeError, ValueError):
        c = 2048
    if c <= 1:
        c = 2048
    return _m_bucket(c)


def _bench_large_m_naive(mod, tag: str) -> int:
    """#large-m-naive: pick the fused-vs-naive dispatch threshold for PREFILL-shaped calls on a
    fused TRITON quant linear (the paths that RETAIN qweight, so the naive fall-through costs no
    extra residency — never the tinygemm path, which frees qweight/scale/zero).

    Why: the M>1 tl.dot kernels are decode-tuned (fixed BM=16/BN=128 tiles, no autotune) — a
    2048-row prefill chunk revisits and RE-DEQUANTS every weight tile cdiv(M,16) times, while the
    naive path dequants the weight ONCE per call and hands one bf16 GEMM to hipBLAS/cuBLAS,
    amortizing the remat over the whole chunk. Which side wins is shape/arch-dependent (the
    APU's BLAS may itself be below peak), so — like the MoE pad-vs-unpadded #dram-dealias bench —
    the choice is MEASURED on the production tensors/op at prepare time, never guessed.

    Bench discipline (#m-bucket): the fused side is timed at EXACTLY ONE row count — the
    production prefill-chunk bucket (_prefill_bench_m) — because every fused probe is a Triton
    launch whose shape gets JIT-specialized, and probing shapes the request stream may never
    present recompiles them at every load. The naive side (plain BLAS, no JIT) is timed at the
    same M for the comparison. M==1 is deliberately NOT probed: decode has its own autotuned
    GEMV and never consults this threshold. Returns the chunk bucket when dequant-once+GEMM
    beats the fused kernel there by >=13%, else 0 (keep fused). forward compares each call's
    ROW BUCKET against the threshold — sub-chunk tails that would PAD UP to the losing bucket
    fall through with it (naive cost is monotonic in M, so naive winning at the bucket covers
    every M the bucket serves); smaller buckets keep the fused kernel (the pre-#large-m-naive
    status quo, now compile-bounded by _m_bucket). Numerics: the fall-through IS the
    self-check's reference path (_dequant + F.linear), so no new self-check is needed. Any
    bench failure -> 0 (keep fused, uncached so a transient hiccup doesn't pin the shape).
    Off-switch: IM_LARGE_M_NAIVE=0."""
    if not _LARGE_M_NAIVE:
        return 0
    try:
        import torch
        import torch.nn.functional as F
        qw, szt, op, in_pad = mod._fused
        key = (tag, qw.shape[0], in_pad, mod.group_size)
        hit = _LARGE_M_CHOICE.get(key)
        if hit is not None:
            return hit
        dev = qw.device

        def _ms(fn, iters=2):
            fn()                                  # warmup (Triton JIT / allocator steady-state)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

        M = _prefill_bench_m()
        xt = torch.randn(M, mod.in_features, device=dev, dtype=torch.bfloat16)
        xk = xt if in_pad == mod.in_features else F.pad(xt, (0, in_pad - mod.in_features))
        xk = xk.contiguous()
        t_f = _ms(lambda: op(xk, qw, mod.group_size, szt))
        t_n = _ms(lambda: F.linear(xt, mod._dequant(torch.bfloat16)))
        thr = M if t_n < t_f * 0.87 else 0        # same >=13%-win margin as the de-alias bench
        _LARGE_M_CHOICE[key] = thr
        _builtins.print(f"[{tag}] large-M dispatch [N={qw.shape[0]},Kpad={in_pad}]: "
                        f"M={M} fused={t_f:.2f}ms naive={t_n:.2f}ms -> "
                        f"{f'naive at row-bucket>={thr}' if thr else 'fused for all M'}",
                        flush=True)
        return thr
    except Exception as exc:                      # never let a bench hiccup break placement
        _builtins.print(f"[{tag}] large-M bench skipped ({exc!r})", flush=True)
        return 0


_W4A16_MOE_OP = None
_W4A16_MOE_TRIED = False


def _w4a16_moe_op():
    """Callable for a FUSED grouped int4 MoE-expert GEMM (ROCm decode fast path).

    Replaces the eager per-expert Python loop (`for e in hit: F.linear(x, gate_up[e])`) with ONE
    Triton launch over all B = tokens*top_k expert applications: program (b, n_block) gathers expert
    eid[b]'s packed int4 weight tile and computes x[b] @ W_e^T. This kills the per-expert kernel
    launches + tensor-subclass dispatch that dominate batch-1 decode (the GEMV itself is bandwidth-
    bound either way; the win is collapsing ~8 launches + 8 Python F.linear/subclass allocs per MoE
    layer into one). Same group-wise asymmetric dequant as `_w4a16_triton_op`, but accumulates the
    GEMV in fp32 (no bf16 round of the weight operand) — numerically equivalent within decode
    tolerance, slightly MORE accurate than the 2D path, not bit-identical to it. Rows carry
    different experts so we can't share a `tl.dot` weight tile across them — each program does a
    per-row GEMV reduction. Worthwhile only for small token counts (decode); the caller keeps the
    eager batched-GEMM loop for prefill. None if triton is unavailable.

    op(x[B,Kin] bf16, eid[B] int, q uint8[E,N,Kpad//2], scale/zero[E,N,ng] bf16, group, in_features)
      -> y[B,N] bf16
    """
    global _W4A16_MOE_OP, _W4A16_MOE_TRIED
    if _W4A16_MOE_TRIED:
        return _W4A16_MOE_OP
    with _W4A16_BUILD_LOCK:      # #triton-race: thread-safe lazy build (see _w4a16_triton_op)
        return _w4a16_moe_op_locked()


def _w4a16_moe_op_locked():
    global _W4A16_MOE_OP, _W4A16_MOE_TRIED
    if _W4A16_MOE_TRIED:         # a racer built it while we waited on the lock
        return _W4A16_MOE_OP
    try:
        import torch
        if triton is None:                # module-level import (see top); None on no-triton workers
            raise ImportError("triton unavailable")
        import torch.nn.functional as _F

        # Autotuned per (B,N,K). This GEMV is bandwidth-bound, so block-width / warps / pipeline
        # depth / K-PARALLELISM are the knobs that move it (the bytes read are fixed). The original
        # serial-K kernel topped out ~35-48% of peak BW on the iGPU: at decode B=tokens*top_k is tiny
        # (~8) so the grid (B, cdiv(N,BN)) launches far too few programs to saturate memory. SPLIT-K
        # fixes that — each program reduces a K-SLICE and atomic-adds its fp32 partial, growing the
        # grid ~SPLITKx (same trick the dense decode GEMV `_ksk` uses for its 3.5-3.9x). SPLITK=1 ==
        # the prior serial-K kernel (so autotune is never worse); >1 trades a little atomic traffic
        # for occupancy. fp32 atomic accumulation -> within decode tolerance (not bit-identical across
        # SPLITK, like all atomic reductions). Lean config set bounds first-decode JIT cost (esp.
        # Windows/ROCm). Re-tuned per (B,N,K) on each arch (sm_89, gfx1151, ...).
        # reset_to_zero: the kernel ATOMIC-ADDS into y_ptr (split-K), so autotune's per-config timing
        # reruns would otherwise pile their partials into the SAME buffer -> the first call for each
        # (B,N,K) returns garbage (a multiple of the result). reset_to_zero zeros y before every
        # trial AND before the real launch, so every launch is clean. (Mandatory for any atomic-acc
        # kernel under autotune; the dense _ksk needs it for the same reason.)
        @triton.autotune(
            configs=[
                triton.Config({"BN": 128, "SPLITK": 1}, num_warps=4, num_stages=2),   # == prior default
                triton.Config({"BN": 128, "SPLITK": 2}, num_warps=4, num_stages=2),
                triton.Config({"BN": 128, "SPLITK": 4}, num_warps=4, num_stages=3),
                triton.Config({"BN": 128, "SPLITK": 8}, num_warps=4, num_stages=3),
                triton.Config({"BN": 64, "SPLITK": 4}, num_warps=2, num_stages=3),
                triton.Config({"BN": 64, "SPLITK": 8}, num_warps=4, num_stages=2),
                # #dram-dealias: what the de-aliased (row-padded) gemma-26b gate_up wants on
                # gfx1151 (+8% over BN=128 there; never picked where it loses)
                triton.Config({"BN": 256, "SPLITK": 4}, num_warps=8, num_stages=2),
            ],
            # sqn (within-expert row stride) is in the key so the load-time pad-vs-unpadded
            # bench (Packed4Tensor3D.prepare_fused) tunes each variant separately instead of
            # inheriting whichever ran first — and the winning tensor keeps its own best config.
            key=["B", "N", "K", "sqn"],
            reset_to_zero=["y_ptr"],
        )
        @triton.jit
        def _mk(x_ptr, e_ptr, q_ptr, s_ptr, z_ptr, y_ptr, B, N, K,
                sxb, sxk, sqe, sqk, sqn, sse, ssn, ssg, sze, szn, szg, syb, syn,
                GROUP: tl.constexpr, BN: tl.constexpr, SPLITK: tl.constexpr):
            pid_b = tl.program_id(0)                       # one (token, expert-slot) application
            pid_n = tl.program_id(1)                       # a BN-wide block of output channels
            pid_k = tl.program_id(2)                       # which K-slice this program reduces (split-K)
            e = tl.load(e_ptr + pid_b).to(tl.int64)        # this row's expert id (64-bit weight base)
            offs_n = pid_n * BN + tl.arange(0, BN)
            offs_h = tl.arange(0, GROUP // 2)              # byte index within a K-group
            nmask = offs_n < N
            ngroups = K // GROUP
            gps = (ngroups + SPLITK - 1) // SPLITK         # K-groups this split reduces
            g0 = pid_k * gps
            acc = tl.zeros((BN,), dtype=tl.float32)
            for gi in range(0, gps):
                kb = g0 + gi
                if kb < ngroups:                           # tail split may reduce fewer groups
                    k0 = kb * GROUP
                    xe = tl.load(x_ptr + pid_b * sxb + (k0 + 2 * offs_h) * sxk)        # [G/2] even cols
                    xo = tl.load(x_ptr + pid_b * sxb + (k0 + 2 * offs_h + 1) * sxk)    # [G/2] odd cols
                    qp = q_ptr + e * sqe + (k0 // 2 + offs_h)[:, None] * sqk + offs_n[None, :] * sqn
                    bb = tl.load(qp, mask=nmask[None, :], other=0).to(tl.int32)        # [G/2, BN] packed
                    lo = (bb & 0xF).to(tl.float32)
                    hi = ((bb >> 4) & 0xF).to(tl.float32)
                    s = tl.load(s_ptr + e * sse + offs_n * ssn + kb * ssg, mask=nmask, other=0.0).to(tl.float32)
                    z = tl.load(z_ptr + e * sze + offs_n * szn + kb * szg, mask=nmask, other=0.0).to(tl.float32)
                    wlo = (lo - z[None, :]) * s[None, :]    # [G/2, BN] dequant low nibble
                    whi = (hi - z[None, :]) * s[None, :]    # [G/2, BN] dequant high nibble
                    acc += tl.sum(xe[:, None] * wlo, axis=0)  # GEMV reduce over the K-group -> [BN]
                    acc += tl.sum(xo[:, None] * whi, axis=0)
            tl.atomic_add(y_ptr + pid_b * syb + offs_n * syn, acc, mask=nmask)   # fp32 partial -> y

        def _op(x, eid, q, scale, zero, group_size, in_features):
            if x.dim() != 2:
                x = x.reshape(-1, x.shape[-1])
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            x = x.contiguous()
            Kpad = q.shape[2] * 2                          # pad activations to the packed width
            if x.shape[1] != Kpad:
                x = _F.pad(x, (0, Kpad - x.shape[1]))
            eid = eid.to(torch.int32).contiguous()
            B = x.shape[0]
            N = q.shape[1]
            # #m-bucket: the autotuner keys this kernel on EXACT B (["B","N","K","sqn"]), so
            # every novel tiny-prompt B (tokens*top_k; the standard-MoE caller gates tokens <=
            # _FUSED_MOE_T_MAX) re-benched the whole config set — novel-prompt-length stalls on
            # slow-compile boxes. Bucket the decode/tiny-prompt regime to powers of two (pad
            # rows with expert-0 ids + zero activations -> zero partials, sliced back off) so
            # at most log2(128) B keys ever exist there. B > 128 (gpt-oss prefill, which has
            # no eager fallback) stays EXACT: pow2-padding whole prefill chunks would tax
            # every call up to 2x on this per-row GEMV, and chunk-sized B repeats anyway.
            Bp = _m_bucket(B) if B <= 128 else B
            if Bp != B:
                x = _F.pad(x, (0, 0, 0, Bp - B))
                eid = _F.pad(eid, (0, Bp - B))
            # split-K atomic-adds fp32 partials -> y must be fp32 + zero-initialized, then cast to bf16
            y = torch.zeros((Bp, N), device=x.device, dtype=torch.float32)
            grid = lambda meta: (Bp, triton.cdiv(N, meta["BN"]), meta["SPLITK"])  # noqa: E731
            _mk[grid](x, eid, q, scale, zero, y, Bp, N, Kpad,
                      x.stride(0), x.stride(1),
                      q.stride(0), q.stride(2), q.stride(1),
                      scale.stride(0), scale.stride(1), scale.stride(2),
                      zero.stride(0), zero.stride(1), zero.stride(2),
                      y.stride(0), y.stride(1),
                      GROUP=group_size)
            yb = y.to(torch.bfloat16)
            return yb if Bp == B else yb[:B]

        _W4A16_MOE_OP = _op
        _builtins.print("[int4] triton fused-MoE w4a16 kernel built (ROCm decode fast path)", flush=True)
    except Exception as exc:
        _builtins.print(f"[int4] fused-MoE w4a16 unavailable ({exc!r}) -> per-expert path", flush=True)
        _W4A16_MOE_OP = None
    _W4A16_MOE_TRIED = True      # #triton-race: only AFTER _W4A16_MOE_OP is final
    return _W4A16_MOE_OP


_W4A16_EXPERT = None


def _w4a16_expert_cls():
    """torch.Tensor subclass for ONE MoE expert's int4 weight (ROCm). Packed4Tensor3D.__getitem__
    returns this instead of a dequantized bf16 weight: it intercepts F.linear(activation, this)
    via __torch_function__ and routes to the Triton w4a16 kernel (reads int4 directly — no
    per-expert full-weight bf16 rematerialization), and materializes to bf16 for ANY other op so
    nothing breaks. The MoE host calls F.linear(state, gate_up_proj[e]) / (state, down_proj[e]),
    so this fuses the routed-expert GEMMs. None if triton is unavailable.
    Thread-safe lazy build (#triton-race): two shard-install threads racing here previously built
    two distinct subclass types (benign per-instance, but confusing) — now one wins under the lock."""
    global _W4A16_EXPERT
    if _W4A16_EXPERT is not None:
        return _W4A16_EXPERT
    with _W4A16_BUILD_LOCK:
        if _W4A16_EXPERT is not None:   # a racer built it while we waited
            return _W4A16_EXPERT
        return _w4a16_expert_cls_locked()


def _w4a16_expert_cls_locked():
    global _W4A16_EXPERT
    op = _w4a16_triton_op()
    if op is None:
        return None
    try:
        import torch
        import torch.nn.functional as F

        class _W4A16Weight(torch.Tensor):
            @staticmethod
            def __new__(cls, packed, scale, zero, group, in_features):
                out = packed.shape[0]
                t = torch.Tensor._make_wrapper_subclass(
                    cls, (out, in_features), dtype=scale.dtype,
                    device=packed.device, requires_grad=False)
                t._packed = packed
                t._scale = scale
                t._zero = zero
                t._group = group
                t._infeat = in_features
                return t

            def _materialize(self):
                qw = self._packed
                out = qw.shape[0]
                lo = (qw & 0x0F).to(torch.int16)
                hi = (qw >> 4).to(torch.int16)
                q = torch.stack((lo, hi), dim=2).reshape(out, -1)
                ng = self._scale.shape[1]
                G = self._group
                dt = self._scale.dtype
                qf = q.reshape(out, ng, G).to(dt)
                w = (qf - self._zero.to(dt).unsqueeze(2)) * self._scale.to(dt).unsqueeze(2)
                return w.reshape(out, ng * G)[:, :self._infeat].contiguous()

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is F.linear or getattr(func, "__name__", "") == "linear":
                    inp = args[0]
                    w = args[1] if len(args) > 1 else kwargs.get("weight")
                    bias = args[2] if len(args) > 2 else kwargs.get("bias")
                    if isinstance(w, cls):
                        y = op(inp, w._packed, w._group, (w._scale, w._zero))
                        if inp.dim() > 2:
                            y = y.reshape(*inp.shape[:-1], y.shape[-1])
                        return y if bias is None else y + bias.to(y.dtype)
                mat = [a._materialize() if isinstance(a, cls) else a for a in args]
                mkw = {k: (v._materialize() if isinstance(v, cls) else v) for k, v in kwargs.items()}
                return func(*mat, **mkw)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # Required by _make_wrapper_subclass. The fast path is handled in
                # __torch_function__ (F.linear); anything that reaches the aten dispatcher with a
                # wrapper operand just materializes to bf16 and re-runs — correctness over speed.
                kwargs = kwargs or {}
                mat = [a._materialize() if isinstance(a, cls) else a for a in args]
                mkw = {k: (v._materialize() if isinstance(v, cls) else v) for k, v in kwargs.items()}
                return func(*mat, **mkw)

        _W4A16_EXPERT = _W4A16Weight
    except Exception as exc:
        _builtins.print(f"[int4] triton w4a16 expert subclass unavailable ({exc!r})", flush=True)
        _W4A16_EXPERT = None
    return _W4A16_EXPERT


def _quantize_linear4(lin, group_size: int = _INT4_GROUP):
    """nn.Linear -> group-wise asymmetric int4 QuantLinear4 (2 nibbles/byte)."""
    import torch
    import torch.nn.functional as F
    QL = _quant4_linear_cls()
    W = lin.weight.data
    out, in_f = W.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    Wp = F.pad(W, (0, in_pad - in_f)) if in_pad != in_f else W
    Wg = Wp.reshape(out, ng, G).float()
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)             # [out, ng]
    zero = torch.round(-wmin / scale).clamp(0, 15)            # [out, ng]
    q = torch.round(Wg / scale.unsqueeze(2) + zero.unsqueeze(2)).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out, in_pad)                                 # [out, in_pad]
    qpacked = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()    # [out, in_pad//2] uint8
    dt = W.dtype
    return QL(qpacked, scale.to(dt), zero.to(dt), lin.bias, in_f, G)


def _quantize_int4_(module) -> None:
    """Recursively replace every nn.Linear under `module` with a QuantLinear4 — EXCEPT inside a
    router/gate module. int4 on a router gate corrupts the top-k expert selection -> garbage
    (gemma4's Gemma4TextRouter exposes `proj` as a plain nn.Linear; custom routers hold a raw weight
    Parameter so they had no inner Linear to skip). Mirrors the cache packer's `_quant_scope`
    exclusion so a cold load stays bit-identical to the serve-from-cache install."""
    from torch import nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear4(child))
        elif type(child).__name__.endswith(("Router", "Gate")):
            continue   # leave router/gate projections bf16 (precision-sensitive routing)
        else:
            _quantize_int4_(child)


# --- int2: 2-bit weight-only quant (#int2) ---------------------------------------------------
# The int4 family cloned down to 2 bits: group-wise ASYMMETRIC per-output-channel, 4 values per
# byte (LOWEST 2 bits = lowest input column), w = (q - zero) * scale, q in 0..3. Group 64 (vs
# int4's 128) because 2-bit round-to-nearest needs finer groups to stay usable — effective
# ~2.5 bits/weight (2 + bf16 scale/zero per 64). Head/embed/norms/router stay bf16 exactly like
# int4. DENSE Linears only in this increment: fused 3D MoE experts have no 2-bit packer/kernel
# yet (the /load route downgrades int2-on-MoE to int4, mirroring the int8-on-MoE downgrade).
# Quality note: RTN 2-bit is a capacity tier, not a fidelity tier — expect visible degradation
# on small models; its use case is fitting a model that would otherwise not fit at all.
_QUANT_LINEAR2 = None
_INT2_GROUP = 64


def _quant2_linear_cls():
    global _QUANT_LINEAR2
    if _QUANT_LINEAR2 is None:
        import torch
        from torch import nn
        import torch.nn.functional as F

        class QuantLinear2(nn.Module):
            def __init__(self, qweight, scale, zero, bias, in_features, group_size):
                super().__init__()
                self.register_buffer("qweight", qweight)  # uint8 [out, in_pad//4] (4 cols/byte)
                self.register_buffer("scale", scale)       # dtype [out, n_groups]
                self.register_buffer("zero", zero)         # dtype [out, n_groups]
                self.bias = bias                            # Parameter or None (bf16)
                self.in_features = in_features
                self.group_size = group_size

            def _dequant(self, dtype):
                qw = self.qweight
                out = qw.shape[0]
                q0 = (qw & 0x3).to(torch.int16)            # input col 4j
                q1 = ((qw >> 2) & 0x3).to(torch.int16)     # col 4j+1
                q2 = ((qw >> 4) & 0x3).to(torch.int16)     # col 4j+2
                q3 = ((qw >> 6) & 0x3).to(torch.int16)     # col 4j+3
                q = torch.stack((q0, q1, q2, q3), dim=2).reshape(out, -1)   # [out, in_pad]
                ng = self.scale.shape[1]
                G = self.group_size
                qf = q.reshape(out, ng, G).to(dtype)
                w = (qf - self.zero.to(dtype).unsqueeze(2)) * self.scale.to(dtype).unsqueeze(2)
                return w.reshape(out, ng * G)[:, :self.in_features].contiguous()

            def prepare_fused(self):
                # int2 has NO torch tinygemm — the Triton w2a16 GEMM is the only fused path, and
                # (unlike int4, where NVIDIA gets tinygemm) it serves BOTH CUDA and ROCm. Same
                # contract as QuantLinear4.prepare_fused: built once on the FINAL device,
                # self-checked vs the naive dequant, ANY mismatch/build failure -> keep naive
                # (never wrong, just slower). CPU: no fused op — the naive path's fp32-GEMM
                # prefill acceleration still applies.
                if not _FUSED_INT2 or getattr(self, "_fused", None) is not None \
                        or getattr(self, "_fused_tried", False) or self.qweight is None:
                    return
                self._fused_tried = True
                dev = self.qweight.device
                if dev.type != "cuda":
                    return
                op = _w2a16_triton_op()
                if op is None:
                    return
                try:
                    G = self.group_size
                    ng = self.scale.shape[1]
                    in_pad = ng * G
                    # #dram-dealias (see QuantLinear4.prepare_fused): the GEMV walks qweight along
                    # N with a row stride of K/4 bytes; an even multiple of 64B collapses DRAM
                    # channel parallelism on big weights. Re-allocate rows on an ODD multiple of
                    # 64B (64B/row cost, never hurts the aligned case).
                    qw = self.qweight
                    if qw.shape[1] % 128 == 0:
                        buf = torch.zeros((qw.shape[0], qw.shape[1] + 64),
                                          dtype=torch.uint8, device=dev)
                        buf[:, :qw.shape[1]].copy_(qw)
                        self.qweight = buf[:, :qw.shape[1]]
                    sz = (self.scale, self.zero)
                    xt = torch.randn(8, self.in_features, device=dev, dtype=torch.bfloat16)
                    xk = xt if in_pad == self.in_features else F.pad(xt, (0, in_pad - self.in_features))
                    yf = op(xk.contiguous(), self.qweight, G, sz).float()
                    yn = F.linear(xt, self._dequant(torch.bfloat16)).float()
                    rel = ((yf - yn).abs().mean() / (yn.abs().mean() + 1e-6)).item()
                    if rel < 0.05:
                        self._fused = (self.qweight, sz, op, in_pad)   # kernel reads qweight; keep it
                        print(f"[int2] triton w2a16 kernel active on {dev}", flush=True)
                        # #large-m-naive: same measured prefill dispatch as QuantLinear4 — the
                        # M>1 _k2 kernel shares _k's fixed decode-tuned tiling, and qweight is
                        # retained here on BOTH vendors, so the fall-through is free everywhere
                        # this fused path exists.
                        self._naive_m_min = _bench_large_m_naive(self, "int2")
                    else:
                        print(f"[int2] triton w2a16 self-check rel={rel:.3f} on {dev} -> naive", flush=True)
                except Exception as exc:
                    print(f"[int2] triton w2a16 prepare failed on {dev} ({exc!r}) -> naive", flush=True)

            def forward(self, x):
                fz = getattr(self, "_fused", None)
                # #large-m-naive + #m-bucket: mirrors QuantLinear4 — when the ROW BUCKET (the
                # shape _k2 would execute after _m_bucket padding) reaches the prepare-time-
                # benched threshold, the decode-tuned _k2 loses to dequant-once+BLAS; fall
                # through to the naive tail (the self-check's reference numerics). 0/absent =
                # always fused.
                _nm = getattr(self, "_naive_m_min", 0)
                if fz is not None and _nm and self.qweight is not None \
                        and _m_bucket(_rows(x)) >= _nm:
                    fz = None
                if fz is not None:
                    # fused-dequant int2 GEMM: flatten, bf16 activations, restore shape.
                    qw, sz, op, in_pad = fz
                    xq = x.reshape(-1, self.in_features)
                    if xq.dtype != torch.bfloat16:
                        xq = xq.to(torch.bfloat16)
                    if in_pad != self.in_features:
                        xq = F.pad(xq, (0, in_pad - self.in_features))
                    y = op(xq.contiguous(), qw, self.group_size, sz).reshape(*x.shape[:-1], -1)
                    y = y.to(x.dtype)
                    return y if self.bias is None else y + self.bias.to(y.dtype)
                # naive fallback: mirrors QuantLinear4 — CPU big-M dequants to fp32 + fp32 GEMM
                # (the dequant is paid regardless, so the fp32 weight is free); else x.dtype.
                if (_CPU_FP32_GEMM and x.device.type == "cpu"
                        and x.dtype != torch.float32 and _rows(x) >= _CPU_FP32_MIN_ROWS):
                    b = self.bias
                    y = F.linear(x.to(torch.float32), self._dequant(torch.float32),
                                 None if b is None else b.to(torch.float32))
                    return y.to(x.dtype)
                return F.linear(x, self._dequant(x.dtype), self.bias)

        _QUANT_LINEAR2 = QuantLinear2
    return _QUANT_LINEAR2


# --- Triton w2a16 int2 GEMM (fused 2-bit decode; CUDA + ROCm) ---------------------------------
# The w4a16 kernel family adapted to 4 values/byte, in the worker's exact int2 group-wise
# asymmetric format: qweight uint8 [N, K//4] (byte j -> input cols 4j..4j+3, LOWEST 2 bits =
# col 4j), scale/zero [N, K//group], w=(q-zero)*scale. Bit-identical to the naive path
# (self-checked in prepare_fused). Lazily built on first use, shared build lock with the w4a16
# family. Unlike int4 (tinygemm on NVIDIA/CPU), this single kernel is the fused path EVERYWHERE
# a GPU + triton exist; no-triton workers (e.g. Windows) fall back to naive automatically.
_W2A16_OP = None
_W2A16_TRIED = False


def _w2a16_triton_op():
    """Callable op(x[M,Kpad] bf16, qweight uint8[N,Kpad//4], group, (scale,zero) [N,ng]) ->
    y[M,N] bf16, or None if triton is unavailable / fails to build. Thread-safe lazy build."""
    global _W2A16_OP, _W2A16_TRIED
    if _W2A16_TRIED:
        return _W2A16_OP
    with _W4A16_BUILD_LOCK:     # #triton-race: same lock discipline as the w4a16 builders
        return _w2a16_triton_op_locked()


def _w2a16_triton_op_locked():
    global _W2A16_OP, _W2A16_TRIED
    if _W2A16_TRIED:             # a racer built it while we waited on the lock
        return _W2A16_OP
    try:
        import torch
        if triton is None:       # module-level import (see top); None on no-triton workers
            raise ImportError("triton unavailable")

        @triton.jit
        def _k2(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, M, N, K,
                sxm, sxk, sqk, sqn, ssn, ssg, szn, szg, sym, syn,
                GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BM + tl.arange(0, BM)
            offs_n = pid_n * BN + tl.arange(0, BN)
            offs_b = tl.arange(0, GROUP // 4)            # byte index within a K-group
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for kb in range(0, K // GROUP):
                k0 = kb * GROUP
                mm = offs_m[:, None] < M
                x0 = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 4 * offs_b + 0)[None, :] * sxk,
                             mask=mm, other=0.0)
                x1 = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 4 * offs_b + 1)[None, :] * sxk,
                             mask=mm, other=0.0)
                x2 = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 4 * offs_b + 2)[None, :] * sxk,
                             mask=mm, other=0.0)
                x3 = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 4 * offs_b + 3)[None, :] * sxk,
                             mask=mm, other=0.0)
                qp = q_ptr + (k0 // 4 + offs_b)[:, None] * sqk + offs_n[None, :] * sqn
                b = tl.load(qp, mask=offs_n[None, :] < N, other=0).to(tl.int32)
                s = tl.load(s_ptr + offs_n * ssn + kb * ssg, mask=offs_n < N, other=0.0).to(tl.float32)
                z = tl.load(z_ptr + offs_n * szn + kb * szg, mask=offs_n < N, other=0.0).to(tl.float32)
                q0 = (b & 0x3).to(tl.float32)
                q1 = ((b >> 2) & 0x3).to(tl.float32)
                q2 = ((b >> 4) & 0x3).to(tl.float32)
                q3 = ((b >> 6) & 0x3).to(tl.float32)
                acc += tl.dot(x0.to(tl.bfloat16), ((q0 - z[None, :]) * s[None, :]).to(tl.bfloat16))
                acc += tl.dot(x1.to(tl.bfloat16), ((q1 - z[None, :]) * s[None, :]).to(tl.bfloat16))
                acc += tl.dot(x2.to(tl.bfloat16), ((q2 - z[None, :]) * s[None, :]).to(tl.bfloat16))
                acc += tl.dot(x3.to(tl.bfloat16), ((q3 - z[None, :]) * s[None, :]).to(tl.bfloat16))
            yp = y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn
            tl.store(yp, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

        # DECODE (M=1) split-K GEMV — same occupancy rationale + autotune space as the w4a16
        # _ksk (see that kernel's header); reset_to_zero because it atomic-adds into y_ptr.
        @triton.autotune(
            configs=[triton.Config({"BN": 128, "SPLITK": s}, num_warps=w)
                     for s in (4, 8, 16) for w in (4, 8)]
                    + [triton.Config({"BN": bn, "SPLITK": s}, num_warps=w)
                       for (bn, s, w) in ((64, 4, 8), (64, 4, 16), (64, 8, 16), (64, 16, 16),
                                          (64, 32, 16), (128, 4, 16), (128, 8, 16))],
            key=["N", "K"],
            reset_to_zero=["y_ptr"],
        )
        @triton.jit
        def _ksk2(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, N, K,
                  sxk, sqk, sqn, ssn, ssg, szn, szg, syn,
                  GROUP: tl.constexpr, BN: tl.constexpr, SPLITK: tl.constexpr):
            pid_n = tl.program_id(0)
            pid_k = tl.program_id(1)
            offs_n = pid_n * BN + tl.arange(0, BN)
            nmask = offs_n < N
            offs_b = tl.arange(0, GROUP // 4)
            ngroups = K // GROUP
            gps = (ngroups + SPLITK - 1) // SPLITK       # K-groups this split reduces
            g0 = pid_k * gps
            acc = tl.zeros((BN,), dtype=tl.float32)
            for gi in range(0, gps):
                kb = g0 + gi
                if kb < ngroups:
                    k0 = kb * GROUP
                    x0 = tl.load(x_ptr + (k0 + 4 * offs_b + 0) * sxk)
                    x1 = tl.load(x_ptr + (k0 + 4 * offs_b + 1) * sxk)
                    x2 = tl.load(x_ptr + (k0 + 4 * offs_b + 2) * sxk)
                    x3 = tl.load(x_ptr + (k0 + 4 * offs_b + 3) * sxk)
                    qp = q_ptr + (k0 // 4 + offs_b)[:, None] * sqk + offs_n[None, :] * sqn
                    b = tl.load(qp, mask=nmask[None, :], other=0).to(tl.int32)
                    s = tl.load(s_ptr + offs_n * ssn + kb * ssg, mask=nmask, other=0.0).to(tl.float32)
                    z = tl.load(z_ptr + offs_n * szn + kb * szg, mask=nmask, other=0.0).to(tl.float32)
                    q0 = (b & 0x3).to(tl.float32)
                    q1 = ((b >> 2) & 0x3).to(tl.float32)
                    q2 = ((b >> 4) & 0x3).to(tl.float32)
                    q3 = ((b >> 6) & 0x3).to(tl.float32)
                    acc += tl.sum(x0[:, None] * ((q0 - z[None, :]) * s[None, :]), axis=0)
                    acc += tl.sum(x1[:, None] * ((q1 - z[None, :]) * s[None, :]), axis=0)
                    acc += tl.sum(x2[:, None] * ((q2 - z[None, :]) * s[None, :]), axis=0)
                    acc += tl.sum(x3[:, None] * ((q3 - z[None, :]) * s[None, :]), axis=0)
            tl.atomic_add(y_ptr + offs_n * syn, acc, mask=nmask)

        def _op2(x, qweight, group_size, sz):
            scale, zero = sz
            if x.dim() != 2:
                x = x.reshape(-1, x.shape[-1])
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            x = x.contiguous()
            Kpad = qweight.shape[1] * 4                  # pad activations to the packed width
            if x.shape[1] != Kpad:
                import torch.nn.functional as _F
                x = _F.pad(x, (0, Kpad - x.shape[1]))
            M, K = x.shape
            N = qweight.shape[0]
            if M == 1:                                   # decode: split-K GEMV -> fp32 acc
                yf = torch.zeros((N,), device=x.device, dtype=torch.float32)
                grid = lambda meta: (triton.cdiv(N, meta["BN"]), meta["SPLITK"])  # noqa: E731
                _ksk2[grid](x.view(-1), qweight, scale, zero, yf, N, K,
                            x.stride(1), qweight.stride(1), qweight.stride(0),
                            scale.stride(0), scale.stride(1), zero.stride(0), zero.stride(1),
                            yf.stride(0), GROUP=group_size)
                return yf.to(torch.bfloat16).view(1, N)
            # #m-bucket: same bounded compile space as the w4a16 _op — pad rows to the pow2
            # bucket (exact: row-independent dot products), slice the pad back off.
            Mp = _m_bucket(M)
            if Mp != M:
                import torch.nn.functional as _F
                x = _F.pad(x, (0, 0, 0, Mp - M))
            y = torch.empty((Mp, N), device=x.device, dtype=torch.bfloat16)
            BM, BN = 16, 128
            grid = (triton.cdiv(Mp, BM), triton.cdiv(N, BN))
            _k2[grid](x, qweight, scale, zero, y, Mp, N, K,
                      x.stride(0), x.stride(1), qweight.stride(1), qweight.stride(0),
                      scale.stride(0), scale.stride(1), zero.stride(0), zero.stride(1),
                      y.stride(0), y.stride(1), GROUP=group_size, BM=BM, BN=BN)
            return y if Mp == M else y[:M]

        _W2A16_OP = _op2
        _builtins.print("[int2] triton w2a16 kernel built (fused 2-bit GEMM)", flush=True)
    except Exception as exc:
        _builtins.print(f"[int2] triton w2a16 unavailable ({exc!r}) -> naive int2", flush=True)
        _W2A16_OP = None
    _W2A16_TRIED = True          # #triton-race: only AFTER _W2A16_OP is final
    return _W2A16_OP


def _quantize_linear2(lin, group_size: int = _INT2_GROUP):
    """nn.Linear -> group-wise asymmetric int2 QuantLinear2 (4 values/byte). Same math shape as
    _quantize_linear4 with a 0..3 grid and group 64 — MUST stay bit-identical to the shard
    cache's pack_linear_int2 (shard_compile.py)."""
    import torch
    import torch.nn.functional as F
    QL = _quant2_linear_cls()
    W = lin.weight.data
    out, in_f = W.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    Wp = F.pad(W, (0, in_pad - in_f)) if in_pad != in_f else W
    Wg = Wp.reshape(out, ng, G).float()
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    scale = ((wmax - wmin) / 3.0).clamp(min=1e-8)              # [out, ng]
    zero = torch.round(-wmin / scale).clamp(0, 3)              # [out, ng]
    q = torch.round(Wg / scale.unsqueeze(2) + zero.unsqueeze(2)).clamp(0, 3).to(torch.uint8)
    q = q.reshape(out, in_pad)                                 # [out, in_pad]
    qpacked = (q[:, 0::4] | (q[:, 1::4] << 2)
               | (q[:, 2::4] << 4) | (q[:, 3::4] << 6)).contiguous()   # [out, in_pad//4] uint8
    dt = W.dtype
    return QL(qpacked, scale.to(dt), zero.to(dt), lin.bias, in_f, G)


def _quantize_int2_(module) -> None:
    """Recursively replace every nn.Linear under `module` with a QuantLinear2 — EXCEPT inside a
    router/gate module (same exclusion as _quantize_int4_: 2-bit on a router gate corrupts the
    top-k expert selection even worse than 4-bit). Mirrors the cache packer's `_quant_scope`
    exclusion so a cold load stays bit-identical to the serve-from-cache install."""
    from torch import nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear2(child))
        elif type(child).__name__.endswith(("Router", "Gate")):
            continue   # leave router/gate projections bf16 (precision-sensitive routing)
        else:
            _quantize_int2_(child)


# --- int4 for FUSED MoE experts (3D gate_up_proj/down_proj nn.Parameters) -------------------
# Modern transformers MoE blocks (MiniMaxM2Experts, Glm4MoeNaiveMoe, Qwen3-MoE, ...) store all
# experts as ONE 3D Parameter [E, out, in] and index it per ROUTED expert in forward
# (self.gate_up_proj[expert_idx]). These are NOT nn.Linear, so _quantize_int4_ misses them —
# yet they are ~90% of a big MoE's params. We replace each 3D Parameter with a Packed4Tensor3D
# that dequantizes ONLY the indexed expert on the fly, so the routed forward keeps its 1/4
# footprint without ever materializing all experts.

_PACKED4_3D = None


def _packed4_3d_cls():
    global _PACKED4_3D
    if _PACKED4_3D is None:
        import torch
        from torch import nn

        class Packed4Tensor3D(nn.Module):
            def __init__(self, qweight, scale, zero, in_features, group_size):
                super().__init__()
                self.register_buffer("qweight", qweight)   # uint8 [E, out, in_pad//2]
                self.register_buffer("scale", scale)        # dtype [E, out, ng]
                self.register_buffer("zero", zero)          # dtype [E, out, ng]
                self.in_features = in_features
                self.group_size = group_size
                self._expert_triton = None                  # ROCm: lazily enabled in __getitem__

            def __getitem__(self, e):
                # Returns the dequantized bf16 weight for one expert; the EAGER MoE host
                # forward then does F.linear(bf16_activation, this_weight). We deliberately
                # keep this bf16 (not fp32): the host contract requires the returned weight
                # match the bf16 activation dtype (an fp32 return would raise in its F.linear),
                # and intercepting the host matmul is too fragile. The fp32 CPU win still
                # applies to the layer's attention / router / shared-expert Linears (nn.Linear
                # / QuantLinear4), just not the per-routed-expert fused GEMM — which is already
                # the "correctness/memory > speed" eager path (see _quantize_experts4_).
                e = int(e)
                # ROCm fast path: hand the MoE host a tensor subclass whose F.linear routes into
                # the Triton w4a16 kernel (reads int4 directly) instead of rematerializing this
                # expert's full bf16 weight every call. One-time self-check vs the bf16 dequant;
                # disabled on mismatch / non-ROCm / no triton.
                if self._expert_triton is None:
                    self._expert_triton = False
                    with contextlib.suppress(Exception):
                        if getattr(torch.version, "hip", None) and self.qweight.device.type == "cuda":
                            wc = _w4a16_expert_cls()
                            if wc is not None:
                                import torch.nn.functional as _F
                                w0 = wc(self.qweight[0], self.scale[0], self.zero[0],
                                        self.group_size, self.in_features)
                                xt = torch.randn(8, self.in_features, device=self.qweight.device,
                                                 dtype=torch.bfloat16)
                                yn = _F.linear(xt, w0._materialize()).float()
                                yf = _F.linear(xt, w0).float()
                                rel = ((yf - yn).abs().mean() / (yn.abs().mean() + 1e-6)).item()
                                if rel < 0.05:
                                    self._expert_triton = True
                                    _builtins.print(f"[int4] triton w4a16 experts active (rel={rel:.4f})", flush=True)
                                else:
                                    _builtins.print(f"[int4] triton w4a16 experts self-check rel={rel:.3f} -> bf16 dequant", flush=True)
                if self._expert_triton:
                    return _w4a16_expert_cls()(self.qweight[e], self.scale[e], self.zero[e],
                                               self.group_size, self.in_features)
                qw = self.qweight[e]                         # [out, in_pad//2] (row-strided ok)
                out = qw.shape[0]
                lo = (qw & 0x0F).to(torch.int16)
                hi = (qw >> 4).to(torch.int16)
                q = torch.stack((lo, hi), dim=2).reshape(out, -1)   # [out, in_pad]
                ng = self.scale.shape[2]
                G = self.group_size
                dt = self.scale.dtype
                qf = q.reshape(out, ng, G).to(dt)
                w = (qf - self.zero[e].to(dt).unsqueeze(2)) * self.scale[e].to(dt).unsqueeze(2)
                return w.reshape(out, ng * G)[:, :self.in_features].contiguous()

            _pad_choice = {}   # (E, N, rs) -> bool, shared across layers (one bench per shape)

            def _dealias_ms(self, op, q, iters=25):
                # time the FUSED expert GEMV on `q` at decode shape (B=top_k-ish), a FRESH random
                # expert subset per call so consecutive calls miss the 32MB MALL the way real
                # decode does (every layer's experts are cold every token; a fixed subset would
                # measure cache, not DRAM).
                import time as _time
                E = q.shape[0]
                B = min(8, E)
                x = torch.randn(B, self.in_features, device=q.device, dtype=torch.bfloat16) * 0.1
                eids = [torch.randperm(E, device=q.device)[:B].to(torch.int32)
                        for _ in range(iters + 3)]
                for i in range(3):                       # warmup + autotune (keyed on sqn too)
                    op(x, eids[i], q, self.scale, self.zero, self.group_size, self.in_features)
                torch.cuda.synchronize()
                t0 = _time.perf_counter()
                for i in range(iters):
                    op(x, eids[3 + i], q, self.scale, self.zero, self.group_size, self.in_features)
                torch.cuda.synchronize()
                return (_time.perf_counter() - t0) / iters * 1e3

            def prepare_fused(self):
                # #dram-dealias (MoE): the fused expert GEMV walks rows along N with a
                # WITHIN-EXPERT row stride of rs = K_pad/2 bytes, and a layer's expert stack
                # (134-280 MB) is far past the 32MB MALL, so decode reads are DRAM-cold —
                # the same regime where the DENSE GEMV collapsed on even-64B row strides (see
                # QuantLinear4.prepare_fused). But unlike dense, the MoE collapse is
                # ALLOCATION-dependent, not a stride rule: the same (E,N,rs) can collapse in
                # one allocation and run full-speed in another (gfx1151 2026-07-07 — synthetic
                # bench_moe_dealias: gemma-26b gate_up 64->188 GB/s from padding while qwen's
                # pow-2 shapes got HALVED by it; live cache-installed tensors: the OPPOSITE —
                # qwen gate_up 0.088->0.052ms from padding, gemma fine unpadded). Presumably
                # the DRAM channel hash includes physical-page bits, so no static rule works.
                # Hence the choice is MEASURED on the actual tensor at load: time the
                # production op unpadded vs padded ([E, N, rs+64] buffer kept as the
                # [:, :, :rs] view) on DRAM-cold random expert subsets and keep the winner
                # (padded only on a >=15% win; decision cached per (E,N,rs) so 40 layers pay 1
                # bench). All consumers (fused _mk kernel, per-expert 2D op, eager dequant)
                # read via .stride()/indexing — the strided view needs no kernel change; the
                # fused kernel's expert base is e*stride(0), which the view carries. Runtime-
                # only: _shards/ caches stay bit-identical (pad at load, never at pack time).
                # Called by the _finalize_placement sweep AFTER final device placement,
                # ROCm-only like the dense pad. Idempotent.
                if getattr(self, "_dealiased", False):
                    return
                self._dealiased = True
                qw = self.qweight
                if qw is None or qw.dim() != 3 or not (qw.device.type == "cuda"
                                                       and getattr(torch.version, "hip", None)):
                    return
                E, N, rs = qw.shape
                if rs % 128 != 0 or qw.stride(2) != 1 or qw.stride(1) != rs:
                    return               # rows already an odd multiple of 64B (or already padded)
                op = _w4a16_moe_op()
                if op is None:
                    return
                choice = Packed4Tensor3D._pad_choice.get((E, N, rs))
                try:
                    if choice is None:
                        buf = torch.zeros((E, N, rs + 64), dtype=torch.uint8, device=qw.device)
                        buf[:, :, :rs].copy_(qw)
                        t_un = self._dealias_ms(op, qw)
                        t_pad = self._dealias_ms(op, buf[:, :, :rs])
                        choice = t_pad < t_un * 0.87
                        Packed4Tensor3D._pad_choice[(E, N, rs)] = choice
                        _builtins.print(f"[int4] moe row de-alias [E={E},N={N},rs={rs}]: "
                                        f"unpadded={t_un:.3f}ms padded={t_pad:.3f}ms -> "
                                        f"{'PAD' if choice else 'keep unpadded'}", flush=True)
                        if choice:
                            self.qweight = buf[:, :, :rs]
                        return
                    if choice:
                        buf = torch.zeros((E, N, rs + 64), dtype=torch.uint8, device=qw.device)
                        buf[:, :, :rs].copy_(qw)
                        self.qweight = buf[:, :, :rs]
                except Exception as exc:     # never let a bench hiccup break placement
                    _builtins.print(f"[int4] moe row de-alias skipped ({exc!r})", flush=True)

        _PACKED4_3D = Packed4Tensor3D
    return _PACKED4_3D


_FUSED_MOE_T_MAX = 8   # only fuse small token counts (decode); prefill keeps the eager batched loop


def _install_fused_moe_forward(experts_mod) -> None:
    """ROCm decode fast path: patch a fused-3D-expert module's forward to run ALL routed experts
    through one Triton launch (`_w4a16_moe_op`) instead of the eager per-expert Python loop. Gated to
    ROCm + int4 Packed4Tensor3D experts + triton. A one-time self-check vs the ORIGINAL forward (on a
    synthetic input) confirms bit-equivalence before the fused path is trusted, and every call falls
    back to the original on a non-decode token count or any exception. No-op on CUDA (the fleet) and
    on non-fused experts — keeps it inert everywhere it isn't proven. This is a pure decode-latency
    optimization: it removes ~top_k Python F.linear/subclass dispatches + kernel launches per MoE
    layer per token (the dominant batch-1 overhead), not a numerics change."""
    import torch, os
    if os.environ.get("INFINITEMODEL_NO_FUSED_MOE"):
        return                                            # A/B kill-switch (measure fused on vs off)
    if not getattr(torch.version, "hip", None):
        # CUDA / CPU: the DEFAULT int4 path is portable (tinygemm `_weight_int4pack_mm` dense +
        # bf16-rematerialize routed experts) and runs everywhere incl. Windows. The fused Triton
        # expert kernel is an OPT-IN UPGRADE for Linux+NVIDIA only (Triton is unreliable on Windows) —
        # enable with INFINITEMODEL_CUDA_FUSED_MOE=1. Self-checked + auto-fallback, so opt-in is safe.
        # See docs/ACCELERATION.md. (ROCm/RDNA always uses it — it's the only fast int4 path there.)
        if not os.environ.get("INFINITEMODEL_CUDA_FUSED_MOE"):
            return
    PT = _packed4_3d_cls()
    gu = getattr(experts_mod, "gate_up_proj", None)
    dn = getattr(experts_mod, "down_proj", None)
    if not (isinstance(gu, PT) and isinstance(dn, PT) and hasattr(experts_mod, "act_fn")):
        return                                            # not a fused-3D int4 experts module
    if getattr(experts_mod, "_fused_moe_installed", False):
        return
    # NOTE: do NOT gate on device here — install runs pre-placement (experts still on CPU heap), so a
    # cuda check would skip every path. The device decision is deferred to the first-decode self-check.
    op = _w4a16_moe_op()
    if op is None:
        return
    import types

    orig_forward = experts_mod.forward                    # bound method (the eager loop)

    def _compute(self, hidden_states, top_k_index, top_k_weights):
        T = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        eid = top_k_index.reshape(-1)                     # [B] expert id per (token, slot)
        w = top_k_weights.reshape(-1).to(hidden_states.dtype)     # [B] gate weight
        xb = hidden_states.repeat_interleave(top_k, dim=0)        # [B, H] token per application
        gu_h, dn_h = self.gate_up_proj, self.down_proj
        yb = op(xb, eid, gu_h.qweight, gu_h.scale, gu_h.zero, gu_h.group_size, gu_h.in_features)
        gate, up = yb.chunk(2, dim=-1)                    # gate_up_proj output is [gate(I) | up(I)]
        h = self.act_fn(gate) * up
        zb = op(h, eid, dn_h.qweight, dn_h.scale, dn_h.zero, dn_h.group_size, dn_h.in_features)
        zb = zb * w[:, None]
        final = torch.zeros_like(hidden_states)
        tok = torch.arange(T, device=hidden_states.device).repeat_interleave(top_k)
        final.index_add_(0, tok, zb.to(final.dtype))     # sum the top_k contributions per token
        return final

    def _selfcheck(self):
        try:
            gu_h, dn_h = self.gate_up_proj, self.down_proj
            if gu_h.qweight.device.type != "cuda":
                return False                              # experts on CPU (offload) -> no fused kernel
            E = int(gu_h.qweight.shape[0])
            H = int(gu_h.in_features)
            dev = gu_h.qweight.device
            k = min(8, E)
            x = torch.randn(2, H, device=dev, dtype=torch.bfloat16) * 0.1
            idx = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(2)])
            wts = (torch.rand(2, k, device=dev, dtype=torch.bfloat16) + 0.1)
            # The reference must be INDEPENDENT of any Triton path, else orig_forward routes the SAME
            # w4a16 kernel (Packed4Tensor3D.__getitem__'s _expert_triton subclass) and a shared bug
            # would pass. Force both holders to the bf16 dequant for the reference, then restore lazy
            # state so the eager fallback path keeps its own fast per-expert kernel.
            sv = (gu_h._expert_triton, dn_h._expert_triton)
            gu_h._expert_triton = dn_h._expert_triton = False
            try:
                ref = orig_forward(x, idx, wts).float()
            finally:
                gu_h._expert_triton, dn_h._expert_triton = sv
            out = _compute(self, x, idx, wts).float()
            scale = ref.abs().mean() + 1e-6
            rel = ((out - ref).abs().mean() / scale).item()
            relmax = ((out - ref).abs().max() / (ref.abs().max() + 1e-6)).item()   # worst element vs signal
            ok = rel < 0.03 and relmax < 0.1
            _builtins.print(f"[int4] fused-MoE self-check rel={rel:.4f} max={relmax:.4f} -> "
                            f"{'ACTIVE' if ok else 'fallback (per-expert)'}", flush=True)
            return ok
        except Exception as exc:
            _builtins.print(f"[int4] fused-MoE self-check failed ({exc!r}) -> per-expert", flush=True)
            return False

    def _fused_forward(self, hidden_states, top_k_index, top_k_weights):
        # Only the decode path (small T) uses the fused kernel; defer the one-time self-check until a
        # decode-eligible call so a leading prefill (T>max) doesn't pay for a check it won't use.
        if hidden_states.shape[0] <= _FUSED_MOE_T_MAX:
            if self._fused_moe_ok is None:
                self._fused_moe_ok = _selfcheck(self)
            if self._fused_moe_ok:
                try:
                    return _compute(self, hidden_states, top_k_index, top_k_weights)
                except Exception:
                    pass                                  # any runtime hiccup -> trusted eager path
        return orig_forward(hidden_states, top_k_index, top_k_weights)

    experts_mod._fused_moe_ok = None
    experts_mod._fused_moe_installed = True
    experts_mod.forward = types.MethodType(_fused_forward, experts_mod)


# --- gpt-oss int4 (#166): transpose-packed fused experts + interleaved clamped SwiGLU + biases -----
# gpt-oss experts differ from the generic fused-MoE case: gate_up_proj is [E, hidden, 2*inter]
# (IN-major, applied as `x @ W`, NOT F.linear's [out,in]), the SwiGLU is INTERLEAVED + clamped with
# per-expert biases, alpha=1.702, limit=7.0. To reuse the w4a16 kernel (F.linear semantics) each
# expert weight is TRANSPOSE-packed to [E, out, in]; the eager bf16-rematerialize host path (in-major)
# then can't consume it, so gpt-oss int4 REQUIRES the fused kernel and has NO eager fallback.

def _is_gptoss_experts(module) -> bool:
    """True if `module` is a gpt-oss fused-expert block (interleaved clamped SwiGLU + expert biases)."""
    if type(module).__name__ == "GptOssExperts":
        return True
    return (hasattr(module, "gate_up_proj_bias") and hasattr(module, "down_proj_bias")
            and hasattr(module, "alpha") and hasattr(module, "limit"))


def _gptoss_fused_ok() -> bool:
    """gpt-oss int4 needs the fused w4a16 MoE kernel (its experts are transpose-packed out-major, so
    the eager in-major `x @ W` host path can't consume them). Available on ROCm always, on CUDA with
    INFINITEMODEL_CUDA_FUSED_MOE=1; never CPU-only (no triton)."""
    import os, torch
    if os.environ.get("INFINITEMODEL_NO_FUSED_MOE"):
        return False
    if _w4a16_moe_op() is None:
        return False
    if getattr(torch.version, "hip", None):
        return True
    return bool(os.environ.get("INFINITEMODEL_CUDA_FUSED_MOE"))


def _install_gptoss_fused_forward(experts_mod) -> None:
    """Install gpt-oss's int4 fused-expert forward: transpose-packed gate_up/down (Packed4Tensor3D)
    through `_w4a16_moe_op`, with the INTERLEAVED clamped SwiGLU (gate=y[...,::2], up=y[...,1::2],
    clamp +/-limit, h=(up+1)*gate*sigmoid(alpha*gate)) and per-expert biases. Runs for ALL token
    counts — the eager host path is invalid once weights are transpose-packed, so there is no
    fallback (the recipe is validated standalone, rel ~ int4 noise). Caller MUST have transpose-packed
    the experts + confirmed _gptoss_fused_ok()."""
    import torch, types
    op = _w4a16_moe_op()
    alpha = float(getattr(experts_mod, "alpha", 1.702))
    limit = float(getattr(experts_mod, "limit", 7.0))

    def _forward(self, hidden_states, router_indices=None, routing_weights=None):
        T = hidden_states.shape[0]
        top_k = router_indices.shape[1]
        eid = router_indices.reshape(-1)                          # [B] expert id per (token, slot)
        w = routing_weights.reshape(-1).to(hidden_states.dtype)   # [B] gate weight
        xb = hidden_states.repeat_interleave(top_k, dim=0)        # [B, H]
        guh, dnh = self.gate_up_proj, self.down_proj              # transpose-packed Packed4Tensor3D
        yb = op(xb, eid, guh.qweight, guh.scale, guh.zero, guh.group_size, guh.in_features)  # [B, 2I]
        yb = yb + self.gate_up_proj_bias[eid]
        gate = yb[..., ::2].clamp(max=limit)                      # gpt-oss INTERLEAVED gate/up
        up = yb[..., 1::2].clamp(min=-limit, max=limit)
        h = (up + 1) * (gate * torch.sigmoid(gate * alpha))       # [B, I]
        zb = op(h, eid, dnh.qweight, dnh.scale, dnh.zero, dnh.group_size, dnh.in_features)   # [B, H]
        zb = (zb + self.down_proj_bias[eid]) * w[:, None]
        final = torch.zeros_like(hidden_states)
        tok = torch.arange(T, device=hidden_states.device).repeat_interleave(top_k)
        final.index_add_(0, tok, zb.to(final.dtype))              # sum the top_k contributions
        return final

    experts_mod._gptoss_fused_installed = True
    experts_mod.forward = types.MethodType(_forward, experts_mod)
    _builtins.print("[int4] gpt-oss fused experts active (transpose-packed w4a16 + interleaved SwiGLU)",
                    flush=True)


# --- MoE intra-layer offload (#moe-offload): attention on GPU, routed experts in CPU RAM ----------
# The llama.cpp --override-tensor "experts=CPU" strategy, intra-layer: a MoE layer's routed-expert
# FFN is ~90% of its bytes but each token activates only k of E experts, while the token-mixer
# (attention) + norms are small, used EVERY token, and latency-critical. Today placement is
# whole-layer (a spilled MoE layer drags its hot attention to CPU with the experts). With this on,
# a split layer keeps attention+norms on GPU and leaves the MoE block (router+experts+shared) on
# CPU. Gated to int4 (Packed4Tensor3D) experts only — those are always heap (no mmap), so unload
# reclaim is unaffected; bf16 experts (mmap Parameters) are left to the whole-layer path.
_MOE_BRIDGE = None


def _moe_bridge_cls():
    global _MOE_BRIDGE
    if _MOE_BRIDGE is None:
        import torch
        from torch import nn

        class _MoEDeviceBridge(nn.Module):
            """Wraps a layer's MoE block so it executes on CPU (where its router+experts+shared live)
            while the rest of the layer runs on GPU. forward: move the incoming hidden GPU->CPU, run
            the wrapped block on CPU, move the output(s) back to the input device. The big routed-
            expert GEMM stays in CPU RAM; attention stays on GPU. Hidden at decode is a few KB, so the
            per-layer round-trip is negligible (validated by the measure-first gate)."""

            def __init__(self, block, cpu_dev):
                super().__init__()
                self.block = block          # registered child; its params/buffers stay on CPU
                self._cpu = cpu_dev

            def forward(self, hidden_states, *args, **kwargs):
                dev = hidden_states.device
                h = hidden_states.to(self._cpu)
                a2 = tuple(x.to(self._cpu) if torch.is_tensor(x) else x for x in args)
                k2 = {kk: (vv.to(self._cpu) if torch.is_tensor(vv) else vv)
                      for kk, vv in kwargs.items()}
                out = self.block(h, *a2, **k2)
                if torch.is_tensor(out):
                    return out.to(dev)
                if isinstance(out, tuple):
                    return tuple(o.to(dev) if torch.is_tensor(o) else o for o in out)
                return out

        _MOE_BRIDGE = _MoEDeviceBridge
    return _MOE_BRIDGE


def _find_moe_block(layer):
    """Locate the routed-expert MoE block within one decoder layer: the DIRECT child of `layer` that
    holds the experts. Detected by an `experts` attribute (the routed-expert container — present on
    per-expert arches like Mixtral `block_sparse_moe.experts` / OLMoE AND fused arches like
    Qwen3-MoE/MiniMax) or, as a fallback, a fused 3D `gate_up_proj`/`down_proj` (Packed4Tensor3D or a
    raw 3D Parameter) anywhere under the child. Returns (attr_name, block_module), or (None, None) for
    a DENSE layer (its MLP has no experts) so a dense layer is never wrapped. The block is the whole
    sparse-MoE module (router gate + routed experts + any shared expert); wrapping it sends all of
    those to CPU and keeps only the token-mixer + norms on GPU. Splittability by quant (experts must
    be heap, not mmap) is gated by the caller (int4/int8 only)."""
    PT = _packed4_3d_cls()
    for name, child in layer.named_children():
        if getattr(child, "experts", None) is not None:
            return name, child
        for sub in child.modules():
            for attr in ("gate_up_proj", "down_proj"):
                a = getattr(sub, attr, None)
                if isinstance(a, PT) or (a is not None and hasattr(a, "dim") and a.dim() == 3):
                    return name, child
    return None, None


def _pack4_expert(We, ng: int, G: int, in_pad: int, in_f: int, dt):
    """Group-wise int4-pack ONE expert's 2D weight [out, in] -> (qpacked [out, in_pad//2] uint8,
    scale [out, ng], zero [out, ng]). The SINGLE source of the per-expert quant math, shared by
    _pack4_3d (in-RAM source, #61) and _stream_pack4_experts (streamed source, #62), so both stay
    bit-identical to the original whole-tensor path (group quant is independent across experts)."""
    import torch
    import torch.nn.functional as F
    if in_pad != in_f:
        We = F.pad(We, (0, in_pad - in_f))
    out = We.shape[0]
    Wg = We.reshape(out, ng, G).float()              # only ONE expert in float (~tens of MB)
    wmin = Wg.amin(dim=2)
    wmax = Wg.amax(dim=2)
    sc = ((wmax - wmin) / 15.0).clamp(min=1e-8)      # [out, ng]
    ze = torch.round(-wmin / sc).clamp(0, 15)
    q = torch.round(Wg / sc.unsqueeze(2) + ze.unsqueeze(2)).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out, in_pad)
    qp = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()   # [out, in_pad//2]
    return qp, sc.to(dt), ze.to(dt)


def _pack4_3d(W3, group_size: int = _INT4_GROUP):
    """Quantize a fused expert tensor [E, out, in] to a Packed4Tensor3D (group-wise int4), ONE
    EXPERT AT A TIME (#61) — the old path `.float()`'d the WHOLE [E, out, in] at once (~14 GB heap
    spike for a MiniMax layer's 256 experts) on top of the bf16 source, OOM-killing memory-tight
    nodes during "building shard". Per-expert == whole-tensor bit-for-bit (verified, CHANGELOG
    m4at). Used when the bf16 source is already resident (mmap on Linux); see _stream_pack4_experts
    (#62) for the variant that streams the source so the layer's experts never land whole in RAM."""
    import torch
    PT = _packed4_3d_cls()
    E, out, in_f = W3.shape
    G = group_size
    ng = (in_f + G - 1) // G
    in_pad = ng * G
    dt = W3.dtype
    qpacked = torch.empty((E, out, in_pad // 2), dtype=torch.uint8)
    scale = torch.empty((E, out, ng), dtype=dt)
    zero = torch.empty((E, out, ng), dtype=dt)
    for e in range(E):
        qp, sc, ze = _pack4_expert(W3[e], ng, G, in_pad, in_f, dt)   # one expert (mmap slice on Linux)
        qpacked[e] = qp; scale[e] = sc; zero[e] = ze
        del qp, sc, ze
    return PT(qpacked, scale, zero, in_f, G)


def _quantize_experts4_(module) -> None:
    """Replace fused MoE expert tensors (3D gate_up_proj/down_proj nn.Parameters) with int4
    Packed4Tensor3D. nn.Linear (attention, router gate, shared experts) is handled separately
    by _quantize_int4_; this catches ONLY the raw 3D expert Parameters those miss."""
    from torch import nn
    targets = []
    for sub in module.modules():
        is_go = _is_gptoss_experts(sub)
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                targets.append((sub, attr, is_go))
    if any(is_go for _s, _a, is_go in targets) and not _gptoss_fused_ok():
        raise RuntimeError("gpt-oss int4 requires the fused w4a16 MoE kernel (ROCm, or CUDA with "
                           "INFINITEMODEL_CUDA_FUSED_MOE=1) — load gpt-oss at quant=none (bf16) here")
    for sub, attr, is_go in targets:
        p = getattr(sub, attr)
        delattr(sub, attr)                       # drop the bf16 Parameter
        # gpt-oss experts are IN-major [E,in,out] (applied `x @ W`) -> transpose to [E,out,in] so the
        # w4a16 kernel's F.linear semantics reproduce it; other MoEs are already [E,out,in].
        w = p.data.transpose(1, 2).contiguous() if is_go else p.data
        setattr(sub, attr, _pack4_3d(w))         # install the int4 holder (submodule)
    # Force the EAGER experts forward on the modules we quantized. transformers 5.x dispatches
    # experts via config._experts_implementation; only "eager" indexes self.gate_up_proj[idx]
    # per routed expert (which our per-expert int4 holder supports). The grouped_mm/batched_mm/
    # deepgemm kernels take the WHOLE 3D weight tensor and would break on the holder. Eager loops
    # over hit experts — slower, but correctness/memory win > speed for a model that only fits at int4.
    # (gpt-oss skips this: its transpose-packed experts have NO valid eager host path — see below.)
    seen = set()
    for sub, _attr, is_go in targets:
        if is_go:
            continue
        cfg = getattr(sub, "config", None)
        if cfg is not None and hasattr(cfg, "_experts_implementation") and id(cfg) not in seen:
            cfg._experts_implementation = "eager"
            seen.add(id(cfg))
    done = {}
    for sub, _attr, is_go in targets:
        if id(sub) in done:
            continue
        done[id(sub)] = True
        if is_go:                                # gpt-oss: dedicated fused forward (all T, no fallback)
            _install_gptoss_fused_forward(sub)
        else:                                    # ROCm fused-MoE decode fast path (no-op on CUDA)
            _install_fused_moe_forward(sub)


def _quantize_experts4_streamed(module, layer_idx: int, fetch_experts, dt) -> None:
    """Per-expert STREAMING build of a layer's MoE experts (#62). The model's fused gate_up_proj/
    down_proj are still META (skip_experts dropped the experts from the layer blob). Read the fused
    target shapes [E, out, in] from the meta Parameters, fetch the experts in chunks via
    fetch_experts(layer, e0, k), and int4-pack via _pack4_expert into the holder's [e] slot. Handles
    BOTH controller checkpoint layouts, auto-detected from the returned blob keys (#75):
      - NON-FUSED source -> {'{local_e}.{proj}': bf16 2D}: fuse each expert exactly as
        _fuse_moe_experts does (gate_up = cat([gate, up]); down = w2; orientation auto-detected).
      - FUSED source -> {'gate_up_proj': [kk,out,in], 'down_proj': [kk,out,in]}: pack each 3D expert
        slice straight in (already fused, no gate/up cat).
    BOTH holders built in one pass per chunk — so the layer's ~7 GB of experts never lands in RAM on
    a memory-tight node."""
    import torch
    from torch import nn
    PT = _packed4_3d_cls()
    G = _INT4_GROUP
    tgt: dict = {}      # attr -> {sub, E, out, in_f, ng, in_pad, go, qpacked, scale, zero}
    gptoss_any = False
    for sub in module.modules():
        is_go = _is_gptoss_experts(sub)
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                E, d1, d2 = int(p.shape[0]), int(p.shape[1]), int(p.shape[2])
                # gpt-oss experts are IN-major [E,in,out]; pack the TRANSPOSE [E,out,in] for the w4a16
                # kernel (each fetched slice is .t()'d below). Other MoEs are already [E,out,in].
                out, in_f = (d2, d1) if is_go else (d1, d2)
                gptoss_any = gptoss_any or is_go
                ng = (in_f + G - 1) // G
                tgt[attr] = {"sub": sub, "E": E, "out": out, "in_f": in_f, "ng": ng, "in_pad": ng * G,
                             "go": is_go,
                             "qpacked": torch.empty((E, out, ng * G // 2), dtype=torch.uint8),
                             "scale": torch.empty((E, out, ng), dtype=dt),
                             "zero": torch.empty((E, out, ng), dtype=dt)}
    if not tgt:
        return
    if gptoss_any and not _gptoss_fused_ok():
        raise RuntimeError("gpt-oss int4 requires the fused w4a16 MoE kernel (ROCm, or CUDA with "
                           "INFINITEMODEL_CUDA_FUSED_MOE=1) — load gpt-oss at quant=none (bf16) here")
    E = next(iter(tgt.values()))["E"]
    gu = tgt.get("gate_up_proj")
    per = max(1, (gu["out"] * gu["in_f"] * 2) if gu else 1)        # one expert's gate_up bf16 bytes
    chunk = max(1, min(64, (256 * 1024 * 1024) // per))            # experts per fetch (>=1)

    def _pick(projs, *names):
        for nm in names:
            if nm in projs:
                return projs[nm]
        return None

    def _pack_into(attr, e, src2d):                                # src2d already [out, in_f]
        b = tgt[attr]
        qp, sc, ze = _pack4_expert(src2d.to(dt), b["ng"], G, b["in_pad"], b["in_f"], dt)
        b["qpacked"][e] = qp; b["scale"][e] = sc; b["zero"][e] = ze

    e = 0
    while e < E:
        k = min(chunk, E - e)
        blob = fetch_experts(layer_idx, e, k)                     # {'{le}.{proj}': bf16} OR fused
        # FUSED checkpoint (#75): the controller couldn't serve per-expert tensors so it sent the 3D
        # fused slices directly, keyed by projection ('gate_up_proj'/'down_proj', each [kk, out, in]).
        # Pack each expert slice straight in — no gate/up fusion (the tensor is already gate_up).
        if "gate_up_proj" in blob and "down_proj" in blob:   # complete fused blob (#75): need BOTH
            for attr in ("gate_up_proj", "down_proj"):
                if attr not in tgt or attr not in blob:
                    continue
                w3 = blob[attr]                                   # [kk, out, in]
                b = tgt[attr]
                for le in range(w3.shape[0]):
                    src2d = w3[le]
                    if b.get("go"):
                        src2d = src2d.t()                         # gpt-oss: [in,out]->[out,in] (always, incl. square)
                    elif tuple(src2d.shape) != (b["out"], b["in_f"]):
                        src2d = src2d.t()                         # orientation auto-detect
                    _pack_into(attr, e + le, src2d.contiguous())
            del blob
            e += k
            continue
        grouped: dict = {}
        for key, t in blob.items():
            if "." not in key:   # not a per-expert '{le}.{proj}' key (e.g. a partial fused blob) —
                raise RuntimeError(  # fail clearly instead of a cryptic unpack error
                    f"L{layer_idx}: unexpected expert blob key {key!r} — controller/worker expert "
                    "layout mismatch (deploy controller + workers on the same version)")
            le_s, proj = key.split(".", 1)
            grouped.setdefault(int(le_s), {})[proj] = t
        for le in range(k):
            projs = grouped.get(le, {})
            if "gate_up_proj" in tgt:
                gate, up = _pick(projs, "w1", "gate_proj"), _pick(projs, "w3", "up_proj")
                if gate is None or up is None:
                    raise RuntimeError(f"L{layer_idx} expert {e+le}: missing gate/up {list(projs)}")
                b = tgt["gate_up_proj"]
                guc = torch.cat([gate, up], dim=0)                # gate THEN up (matches _fuse_moe_experts)
                if tuple(guc.shape) != (b["out"], b["in_f"]):
                    guc = torch.cat([gate.t(), up.t()], dim=0)    # orientation auto-detect
                _pack_into("gate_up_proj", e + le, guc.contiguous())
            if "down_proj" in tgt:
                down = _pick(projs, "w2", "down_proj")
                if down is None:
                    raise RuntimeError(f"L{layer_idx} expert {e+le}: missing down {list(projs)}")
                b = tgt["down_proj"]
                if tuple(down.shape) != (b["out"], b["in_f"]):
                    down = down.t()
                _pack_into("down_proj", e + le, down.contiguous())
        del blob, grouped
        e += k
    seen = set()
    subs = {}
    for attr, b in tgt.items():
        delattr(b["sub"], attr)                                   # drop the meta Parameter
        setattr(b["sub"], attr, PT(b["qpacked"], b["scale"], b["zero"], b["in_f"], G))
        subs[id(b["sub"])] = b["sub"]
        if _is_gptoss_experts(b["sub"]):                          # gpt-oss uses its own fused forward
            continue
        cfg = getattr(b["sub"], "config", None)                   # force eager experts forward
        if cfg is not None and hasattr(cfg, "_experts_implementation") and id(cfg) not in seen:
            cfg._experts_implementation = "eager"; seen.add(id(cfg))
    for sub in subs.values():
        if _is_gptoss_experts(sub):                              # gpt-oss: dedicated fused forward (all T)
            _install_gptoss_fused_forward(sub)
        else:                                                    # ROCm fused-MoE decode fast path (no-op on CUDA)
            _install_fused_moe_forward(sub)
    print(f"[load] L{layer_idx}: streamed {E} experts (per-expert int4, chunk {chunk})")


def _quantize_experts4_streamed_nonfused(module, layer_idx: int, fetch_experts, dt) -> None:
    """Per-expert STREAMING int4 build for a NON-fused MoE layer (#78). The model stores experts as
    an nn.ModuleList of {w1,w2,w3}-style nn.Linears (e.g. MiniMax-M2 block_sparse_moe.experts.N.*,
    or Mixtral) — all META here because skip_experts dropped them from the layer blob. For each
    experts ModuleList: read E + the per-expert Linear proj names + shapes from the meta Linears,
    fetch experts in chunks via fetch_experts(layer, e0, k) (server returns {'{le}.{proj}': bf16
    [out,in]}, the same non-fused wire format _quantize_experts4_streamed already consumes), int4-pack
    each proj with _pack4_expert and replace the meta nn.Linear with a QuantLinear4 IN PLACE — exactly
    the holder the full-blob path's _quantize_int4_ would produce, so output is bit-identical. The
    transient is bounded by the chunk (~256 MiB), NOT the layer's full ~7 GB of experts: that is what
    lets a 129 GB non-fused MoE SPREAD across the fleet instead of cramming onto one big node (the
    full-blob path's failure). The expert Linears must be quantized BEFORE _quantize_int4_ walks the
    layer (caller orders it so) — once replaced they are no longer nn.Linear, so that walk skips them."""
    from torch import nn
    QL = _quant4_linear_cls()
    G = _INT4_GROUP
    # Locate the non-fused expert container(s) in this layer (an `experts` ModuleList of Linear-bearing
    # modules). Usually exactly one (block_sparse_moe.experts); handle >1 defensively.
    blocks = []
    for sub in module.modules():
        experts = getattr(sub, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0 \
                and any(isinstance(m, nn.Linear) for m in experts[0].modules()):
            blocks.append(experts)
    for experts in blocks:
        E = len(experts)
        # Per-expert Linear projections, discovered from expert 0 (attr name == the checkpoint proj
        # token the server keys by, e.g. w1/w3/w2). DIRECT children only — matches the server's flat
        # `experts.{N}.{proj}.weight` regex (nested experts wouldn't be served per-expert anyway).
        proj_names = [nm for nm, ch in experts[0].named_children() if isinstance(ch, nn.Linear)]
        if not proj_names:
            continue
        per = 0
        for nm in proj_names:                                   # per-expert bf16 footprint -> chunk size
            w = getattr(experts[0], nm).weight
            per += int(w.shape[0]) * int(w.shape[1]) * 2
        per = max(1, per)
        chunk = max(1, min(64, (256 * 1024 * 1024) // per))     # cap in-flight bf16 transient at ~256 MiB
        e = 0
        while e < E:
            k = min(chunk, E - e)
            blob = fetch_experts(layer_idx, e, k)               # {'{le}.{proj}': bf16 [out,in]}
            grouped: dict = {}
            for key, t in blob.items():
                if "." not in key:
                    raise RuntimeError(f"non-fused expert blob key {key!r} has no '.' (layer {layer_idx})")
                le_s, proj = key.split(".", 1)
                grouped.setdefault(int(le_s), {})[proj] = t
            for le, projs in grouped.items():
                gi = e + le                                     # server returns chunk-local indices [0..k)
                if gi >= E:
                    continue
                expert_mod = experts[gi]
                for proj, t in projs.items():
                    lin = getattr(expert_mod, proj, None)
                    if not isinstance(lin, nn.Linear):
                        continue                                # unexpected proj name -> skip (meta guard catches leftovers)
                    exp_out, exp_in = int(lin.weight.shape[0]), int(lin.weight.shape[1])
                    w2d = t
                    out, in_f = int(t.shape[0]), int(t.shape[1])
                    if (out, in_f) != (exp_out, exp_in) and (in_f, out) == (exp_out, exp_in):
                        w2d = t.t().contiguous(); out, in_f = exp_out, exp_in   # orientation auto-detect
                    ng = (in_f + G - 1) // G
                    in_pad = ng * G
                    qp, sc, ze = _pack4_expert(w2d.to(dt), ng, G, in_pad, in_f, dt)
                    setattr(expert_mod, proj, QL(qp, sc, ze, lin.bias, in_f, G))
            del blob, grouped
            e += k
        print(f"[load] L{layer_idx}: streamed {E} non-fused experts "
              f"({'/'.join(proj_names)}, per-expert int4, chunk {chunk})")


def _layer_has_meta_experts(lyr) -> bool:
    """True if the layer's fused 3D MoE experts are still META — i.e. skip_experts dropped the
    layer's expert tensors from the blob, so they must be STREAMED per-expert (#62). With the
    skip_experts filter matching BOTH fused (...gate_up_proj/down_proj) and per-expert
    (...experts.{N}.{proj}.weight) names, a big-MoE int4 layer ends up here (meta -> streamed),
    while a dense layer (no 3D expert params) or a load where skip_experts was off (experts resident,
    not meta) returns False and uses the in-place _quantize_experts4_. Auto-selects per layer."""
    from torch import nn
    for sub in lyr.modules():
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3 and p.is_meta:
                return True
    return False


def _layer_has_meta_experts_nonfused(lyr) -> bool:
    """True if the layer has a NON-fused `experts` ModuleList whose Linear weights are still META —
    i.e. skip_experts dropped them from the layer blob, so they must be STREAMED per-expert (#78,
    the non-fused analogue of _layer_has_meta_experts). False when experts are resident (skip_experts
    off -> full-blob path) or the layer has no such ModuleList."""
    from torch import nn
    for sub in lyr.modules():
        experts = getattr(sub, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            for m in experts[0].modules():
                if isinstance(m, nn.Linear) and getattr(m.weight, "is_meta", False):
                    return True
    return False


def _model_has_fused_experts(model) -> bool:
    """True if the model uses FUSED 3D MoE experts (gate_up_proj/down_proj nn.Parameters) — the
    layout the per-expert STREAMING packer (#62/#75) targets. A non-fused MoE (e.g. MiniMax-M2,
    whose experts are an nn.ModuleList of {w1,w2,w3} Linears) returns False, so its int4 load uses
    the full-blob path (_quantize_int4_ quantizes the expert Linears in place) instead of streaming
    into fused tensors that don't exist. Dense models also return False (no MoE)."""
    from torch import nn
    for sub in model.modules():
        for attr in ("gate_up_proj", "down_proj"):
            p = sub._parameters.get(attr)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                return True
    return False


def _model_has_nonfused_experts(model) -> bool:
    """True if the model uses NON-fused MoE experts: an nn.ModuleList named `experts` whose elements
    contain nn.Linear leaves (e.g. MiniMax-M2 / Mixtral experts.N.{w1,w3,w2}). The complement of
    _model_has_fused_experts (3D Parameters) and of dense models (no `experts` ModuleList). Gates the
    per-expert non-fused STREAMING packer (#78) so a big non-fused MoE never lands its whole layer of
    experts in RAM at once (the full-blob path that pins the model to one node)."""
    from torch import nn
    for sub in model.modules():
        experts = getattr(sub, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            if any(isinstance(m, nn.Linear) for m in experts[0].modules()):
                return True
    return False


def _assign_meta_from_sd(model, sd) -> int:
    """Materialize meta tensors that load_state_dict(assign=True) SKIPPED.

    load_state_dict only installs keys present in the model's own state_dict. A NON-PERSISTENT
    buffer is excluded from state_dict, so even though the served blob carries it, it is never
    assigned and stays on the 'meta' device — and the later module.to(device) placement then raises
    'Cannot copy out of meta tensor; no data!'. The canonical case is MiniMax-M2's sigmoid-router
    `block_sparse_moe.e_score_correction_bias` (DeepSeek-style routing); softmax-routed MoEs
    (Mixtral/OLMoE) have no such buffer, which is why only MiniMax tripped this.

    For every served tensor whose target attribute is STILL meta, assign the real data directly.
    Tensors NOT in sd (the int4 experts dropped by skip_experts) are untouched — they stay meta for
    the per-expert streaming build. Returns the number of tensors materialized."""
    import torch
    fixed = 0
    for key, t in sd.items():
        parent = model
        parts = key.split(".")
        try:
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            attr = parts[-1]
            cur = getattr(parent, attr, None)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if cur is None or not getattr(cur, "is_meta", False):
            continue
        val = (t if t.dtype == cur.dtype else t.to(cur.dtype)).detach().clone()
        if isinstance(cur, torch.nn.Parameter):
            setattr(parent, attr, torch.nn.Parameter(val, requires_grad=False))
        else:
            parent.register_buffer(attr, val, persistent=False)
        fixed += 1
    return fixed
