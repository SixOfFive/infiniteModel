"""TurboQuant KV-cache quantization for InfiniteModel.

TurboQuant (Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026,
arXiv:2504.19874) is a DATA-FREE near-optimal vector quantizer. The idea: a random
orthogonal rotation makes a vector's coordinates near-i.i.d. (Beta on the sphere ->
N(0, 1/d) in high d), so each coordinate can be quantized with the SAME precomputed
optimal (Lloyd-Max) scalar codebook. Two quantizers:

  * KEYS  -> TurboQuant_mse: rotate the unit direction, quantize each coord to its
    nearest centroid (b bits), store the per-vector norm. Minimizes reconstruction MSE.
  * VALUES -> TurboQuant_prod (two-stage, unbiased INNER PRODUCT): MSE at (b-1) bits,
    then a 1-bit QJL (sign(S r)) on the residual + the residual norm. E[<y,x~>] = <y,x>.

Crucially the rotation is UN-DONE on dequant (x~ = norm * Pi^T y~), so the reconstructed
K/V come back in the ORIGINAL basis — the model's attention is used UNCHANGED (no Q
rotation, no patched attention). The rotation only shapes the quantization noise.

WHY for InfiniteModel: KV is kept in bf16 today and is the binding VRAM constraint on
context length + model coexistence (the #76 ctx guardrail caps ctx precisely because
bf16 KV won't fit VRAM). TurboQuant stores KV at ~3 bits/coord at near-FP quality.

This module provides the math (TurboQuantizer) + a transformers Cache subclass
(TurboQuantCache) that quantize-stores K/V and dequant-returns them to the unpatched
attention. The CORRECTNESS path (dequant-on-read) lands first; the PEAK-VRAM win needs a
chunked/fused quantized-attention that never materializes the full bf16 KV — a documented
follow-up (see TurboQuantCache.NOTE). Controller-only-importable + worker-importable
(stdlib + torch only); listed in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import math

# Lloyd-Max optimal scalar-quantizer centroids for the UNIT normal N(0,1), b = bits.
# Coordinates of a rotated unit vector are ~N(0, 1/d), so the codebook used is these
# values scaled by 1/sqrt(d) (applied at runtime). b=1/2 match the paper's quoted
# centroids (±sqrt(2/pi), ±0.4528/±1.5104); b=3/4 are the standard Max (1960) tables.
_LLOYD_MAX = {
    1: [-0.79788456, 0.79788456],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    4: [-2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326],
}

# Named presets -> (key_bits, value_bits). value_bits is the TOTAL (MSE b-1 + 1 QJL).
_PRESETS = {
    "turbo3": (3, 3),
    "turbo4": (4, 4),
    "turbo2": (2, 2),
}


def preset_bits(name: str):
    """('turbo3'|'turbo4'|'turbo2') -> (key_bits, value_bits), or None if not a TurboQuant preset."""
    return _PRESETS.get((name or "").strip().lower())


def kv_quant_bytes_per_token_per_layer(name: str, n_kv_heads: int, head_dim: int) -> int:
    """Stored bytes for ONE token's K+V on ONE layer under preset `name` (for KV-reservation
    accounting). int8 index storage in the foundation (1 byte/coord) + per-vector fp16 norms;
    bit-packing to b bits is a follow-up. bf16 baseline is 2*2*nkv*hd. Returns 0 if not TurboQuant."""
    pb = preset_bits(name)
    if not pb:
        return 0
    d = n_kv_heads * head_dim
    # K: 1 idx byte/coord + 1 fp16 norm/head ; V: 1 idx byte/coord + 1 QJL byte/coord + fp16 norms/head.
    # (Foundation storage; a bit-packed variant reaches ~b/8 bytes/coord.)
    k = d * 1 + n_kv_heads * 2
    v = d * 1 + d * 1 + n_kv_heads * 2 * 2
    return k + v


class TurboQuantizer:
    """Per-(layer,head) TurboQuant quantizer over the last dim = head_dim. Rotation Pi + QJL
    matrix S are generated ONCE per head_dim (fixed seed) and shared by quant + dequant."""

    def __init__(self, torch, head_dim: int, key_bits: int = 3, value_bits: int = 3,
                 device="cpu", dtype=None, seed: int = 0, value_qjl: bool = False):
        self.torch = torch
        self.d = int(head_dim)
        self.kb = int(key_bits)
        # values: by default plain MSE at value_bits (same family as keys — best per-vector
        # RECONSTRUCTION, which is what the attention weighted-sum o=attn@V needs). The TurboQuant
        # two-stage (MSE b-1 + 1-bit QJL residual) gives an UNBIASED inner product but, at 1 bit/dim,
        # higher per-vector variance (validated: worse reconstruction on random V) — its benefit is
        # for structured real-model V, so it's OPT-IN (value_qjl=True) pending real-model tuning.
        self.value_qjl = bool(value_qjl)
        self.vb_mse = (max(1, int(value_bits) - 1) if self.value_qjl else int(value_bits))
        self.dtype = dtype or torch.bfloat16
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        # random orthogonal Pi via QR of a Gaussian (column-orthonormal), and the QJL Gaussian S
        a = torch.randn(self.d, self.d, generator=g, dtype=torch.float32)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)        # fix QR sign ambiguity -> deterministic
        self.Pi = q.to(device=device, dtype=torch.float32)        # [d,d] orthogonal
        self.S = torch.randn(self.d, self.d, generator=g, dtype=torch.float32).to(device)  # QJL
        inv = 1.0 / math.sqrt(self.d)
        self._ck = torch.tensor(_LLOYD_MAX[self.kb], dtype=torch.float32, device=device) * inv
        self._cv = torch.tensor(_LLOYD_MAX[self.vb_mse], dtype=torch.float32, device=device) * inv

    # ---- shared scalar quant of a (rotated) unit direction ----
    def _q_mse(self, x, centroids):
        """x [...,d] -> (idx uint8 [...,d], norm fp [...,1]). Rotate the unit direction, nearest centroid."""
        t = self.torch
        xf = x.to(t.float32)
        norm = xf.norm(dim=-1, keepdim=True)
        u = xf / (norm + 1e-12)
        y = u @ self.Pi.T                                          # rotate [...,d]
        d2 = (y.unsqueeze(-1) - centroids).abs()                   # [...,d,2^b]
        idx = d2.argmin(dim=-1).to(t.uint8)
        return idx, norm

    def _dq_mse(self, idx, norm, centroids):
        """(idx, norm) -> x~ [...,d] in the ORIGINAL basis (un-rotated)."""
        y = centroids[idx.long()]                                 # [...,d]
        u = y @ self.Pi                                            # un-rotate (Pi^T y for row vecs)
        return (u * norm).to(self.dtype)

    # ---- keys: MSE quantizer ----
    def quant_keys(self, k):
        return self._q_mse(k, self._ck)                           # (idx, norm)

    def dequant_keys(self, idx, norm):
        return self._dq_mse(idx, norm, self._ck)

    # ---- values: MSE (default) or two-stage MSE + 1-bit QJL residual (opt-in, unbiased inner prod) ----
    def quant_values(self, v):
        t = self.torch
        idx, norm = self._q_mse(v, self._cv)
        if not self.value_qjl:
            return idx, norm, None, None                         # plain MSE (default)
        x_mse = self._dq_mse(idx, norm, self._cv).to(t.float32)
        r = v.to(t.float32) - x_mse                              # residual
        qjl = (r @ self.S.T >= 0)                                # bool [...,d] (sign of S r)
        gamma = r.norm(dim=-1, keepdim=True)                     # [...,1]
        return idx, norm, qjl, gamma

    def dequant_values(self, idx, norm, qjl=None, gamma=None):
        t = self.torch
        x_mse = self._dq_mse(idx, norm, self._cv)
        if qjl is None:
            return x_mse                                         # plain MSE
        x = x_mse.to(t.float32)
        sgn = qjl.to(t.float32) * 2.0 - 1.0                      # {0,1}->{-1,+1}
        x_qjl = (math.sqrt(math.pi / 2.0) / self.d) * gamma * (sgn @ self.S)
        return (x + x_qjl).to(self.dtype)


# transformers Cache integration — built lazily so this module imports with torch only.
def make_turboquant_cache(dynamic_cache_cls, dynamic_layer_cls, quantizer_factory):
    """Return a TurboQuantCache INSTANCE that keeps each layer's K/V QUANTIZED and returns the
    DEQUANTIZED (un-rotated, original-basis) full K/V from update() — so the model's attention runs
    UNCHANGED. Written against the transformers 5.x Cache API (validated on 5.12.1): a DynamicCache
    holds one DynamicLayer per layer in ``self.layers``; the model's attention consumes the TUPLE
    update() RETURNS (it does NOT re-read the cache mid-forward), and the causal mask is sized from
    ``layer.get_seq_length()``. So we subclass DynamicLayer, store only the quantized reps, and
    reconstruct the full bf16 K/V as update()'s return value.

    quantizer_factory(head_dim, device, dtype) -> TurboQuantizer (preset bits bound by the caller).

    MEMORY: each layer persistently holds only the quantized reps (uint8 index/coord + a few fp32
    norms/head ~= 1 byte/coord vs bf16's 2) — a ~2x RESTING KV reduction at the foundation; bit-
    packing the indices to b bits reaches ~b/8. update() transiently rebuilds ONE layer's full bf16
    K/V for the attention (freed when the forward returns); because the pipeline runs layers
    sequentially, only one layer's bf16 lives at a time, so even the per-step PEAK drops
    (all-layers-quantized + one-layer-bf16 << all-layers-bf16). The KV RESERVATION still budgets
    bf16 (conservative) until that peak headroom is measured and tightened — see kv_reserve_probe."""

    class _TurboQuantLayer(dynamic_layer_cls):
        """A DynamicLayer that stores K/V QUANTIZED (TurboQuant) and reconstructs full bf16 on
        update(). Every method the model/cache uses for length + masking (get_seq_length, and via it
        get_mask_sizes) is overridden or routes through our token counter, so the causal mask stays
        correct; get_max_cache_shape (==-1, unbounded) + reset (no-op while uninitialized) are safe
        as inherited because we never lazy-initialize the bf16 self.keys/self.values buffers."""

        def __init__(self, config=None):
            super().__init__()
            self._tq = None                              # this layer's TurboQuantizer (built on 1st update)
            self._ki = self._kn = None                   # key   indices [b,h,S,d] uint8, norms [b,h,S,1]
            self._vi = self._vn = None                   # value indices/norms
            self._vq = self._vg = None                   # value QJL sign-bits/gammas (None unless value_qjl)
            self._len = 0                                # cached token count (drives get_seq_length + mask)

        def _q(self, k):
            if self._tq is None:
                self._tq = quantizer_factory(k.shape[-1], k.device, k.dtype)
            return self._tq

        def update(self, key_states, value_states, *args, **kwargs):
            import torch
            q = self._q(key_states)
            ki, kn = q.quant_keys(key_states)
            vi, vn, vq, vg = q.quant_values(value_states)
            if self._ki is None:
                self._ki, self._kn = ki, kn
                self._vi, self._vn, self._vq, self._vg = vi, vn, vq, vg
            else:                                        # append the new step's reps along the seq dim
                self._ki = torch.cat([self._ki, ki], dim=-2)
                self._kn = torch.cat([self._kn, kn], dim=-2)
                self._vi = torch.cat([self._vi, vi], dim=-2)
                self._vn = torch.cat([self._vn, vn], dim=-2)
                if vq is not None:
                    self._vq = torch.cat([self._vq, vq], dim=-2)
                    self._vg = torch.cat([self._vg, vg], dim=-2)
            self._len += int(key_states.shape[-2])
            keys = q.dequant_keys(self._ki, self._kn)            # full bf16, transient (returned only)
            vals = q.dequant_values(self._vi, self._vn, self._vq, self._vg)
            return keys, vals

        def get_seq_length(self) -> int:
            return int(self._len)

        def crop(self, max_length: int) -> None:
            # spec-decode rollback: truncate the quantized reps to `max_length` tokens (seq dim = -2).
            if max_length < 0:
                max_length = self._len - abs(max_length)
            if self._ki is None or self._len <= max_length:
                return
            def _sl(t):
                return t[..., :max_length, :] if t is not None else None
            self._ki, self._kn = _sl(self._ki), _sl(self._kn)
            self._vi, self._vn = _sl(self._vi), _sl(self._vn)
            self._vq, self._vg = _sl(self._vq), _sl(self._vg)
            self._len = int(max_length)

        def reorder_cache(self, beam_idx) -> None:
            # beam search isn't used by this engine; reorder the quantized reps if ever invoked.
            if self._ki is None:
                return
            bi = beam_idx.to(self._ki.device)
            for nm in ("_ki", "_kn", "_vi", "_vn", "_vq", "_vg"):
                t = getattr(self, nm)
                if t is not None:
                    setattr(self, nm, t.index_select(0, bi))

    class _TurboQuantCache(dynamic_cache_cls):
        """A DynamicCache whose lazily-appended layers are _TurboQuantLayer (quantized KV). Built with
        NO config so DynamicCache uses its lazy layer_class_to_replicate path — kv_quant is gated to
        non-hybrid (plain full-attention) models, exactly the case that takes that path."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.layer_class_to_replicate = _TurboQuantLayer

    return _TurboQuantCache()
