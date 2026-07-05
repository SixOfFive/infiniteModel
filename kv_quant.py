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
    accounting). BIT-PACKED indices: b bits/coord -> ceil(d*b/8) bytes, plus a per-head fp16 norm.
    Reflects the DEFAULT (MSE) value path — value_qjl adds a 1-bit residual sign + gamma but is
    opt-in and untuned, so it isn't budgeted here. bf16 baseline is 2*2*nkv*hd. 0 if not TurboQuant."""
    pb = preset_bits(name)
    if not pb:
        return 0
    kb, vb = pb
    d = n_kv_heads * head_dim
    k = math.ceil(d * kb / 8) + n_kv_heads * 2       # packed K idx (kb bits/coord) + fp16 norm/head
    v = math.ceil(d * vb / 8) + n_kv_heads * 2       # packed V idx (vb bits/coord) + fp16 norm/head
    return k + v


def _pack_bits(torch, idx, bits: int):
    """Pack a uint8 tensor `idx` [...,d] whose values are in [0, 2**bits) into bytes
    [...,ceil(d*bits/8)] along the last dim: each value emitted MSB-first, the bitstream
    concatenated and zero-padded up to a byte boundary, then regrouped 8 bits/byte. Exact and
    vectorized for any bits in 1..8 (b=3 packs across byte boundaries, unlike a nibble hack)."""
    d = int(idx.shape[-1])
    lead = tuple(idx.shape[:-1])
    sh = torch.arange(bits - 1, -1, -1, device=idx.device, dtype=torch.uint8)
    b = (idx.unsqueeze(-1) >> sh) & 1                          # [...,d,bits] {0,1}
    b = b.reshape(*lead, d * bits)
    pad = (-(d * bits)) % 8
    if pad:
        b = torch.nn.functional.pad(b, (0, pad))
    b = b.reshape(*lead, -1, 8).to(torch.int32)               # [...,nbytes,8]
    w = (1 << torch.arange(7, -1, -1, device=idx.device, dtype=torch.int32))
    return (b * w).sum(-1).to(torch.uint8)                    # [...,nbytes]


def _unpack_bits(torch, packed, bits: int, d: int):
    """Inverse of _pack_bits: bytes [...,nbytes] -> uint8 idx [...,d] (values in [0, 2**bits))."""
    lead = tuple(packed.shape[:-1])
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    b = (packed.unsqueeze(-1) >> sh) & 1                       # [...,nbytes,8]
    b = b.reshape(*lead, -1)[..., :d * bits].reshape(*lead, d, bits).to(torch.int32)
    w = (1 << torch.arange(bits - 1, -1, -1, device=packed.device, dtype=torch.int32))
    return (b * w).sum(-1).to(torch.uint8)                    # [...,d]


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
def make_turboquant_cache(dynamic_cache_cls, dynamic_layer_cls, quantizer_factory,
                          residual_window: int = 0):
    """Return a TurboQuantCache INSTANCE that keeps each layer's K/V QUANTIZED and returns the
    DEQUANTIZED (un-rotated, original-basis) full K/V from update() — so the model's attention runs
    UNCHANGED. Written against the transformers 5.x Cache API (validated on 5.12.1): a DynamicCache
    holds one DynamicLayer per layer in ``self.layers``; the model's attention consumes the TUPLE
    update() RETURNS (it does NOT re-read the cache mid-forward), and the causal mask is sized from
    ``layer.get_seq_length()``. So we subclass DynamicLayer, store only the quantized reps, and
    reconstruct the full bf16 K/V as update()'s return value.

    quantizer_factory(head_dim, device, dtype) -> TurboQuantizer (preset bits bound by the caller).

    RESIDUAL WINDOW (small-model quality, KIVI/KVQuant-style): if ``residual_window`` W > 0, the
    most-recent W tokens' K/V are kept in FULL bf16 (never quantized) and only tokens that age out
    beyond the window are TurboQuant-quantized. Recent tokens dominate attention and are the most
    sensitive to quantization noise, so a small full-precision tail restores coherence on models
    that otherwise collapse under whole-cache quant (turbo3/turbo4 below ~14B). W == 0 (default) is
    the original whole-cache path, BYTE-IDENTICAL to the deployed #172 behaviour. Cost of W > 0 is a
    fixed W-token bf16 buffer per layer (negligible: e.g. W=128 on a 14B layer ~0.25 MB), so the KV
    reservation is only marginally under-counted — safe for any modest W.

    MEMORY: each layer persistently holds only the quantized reps — the indices BIT-PACKED to b
    bits/coord (ceil(d*b/8) bytes, via _pack_bits) + a few fp16 norms/head, so ~b/8 the bytes of
    bf16's 2 (turbo3 ~3/16 -> ~4-5x resting reduction incl. norms). update() unpacks the full seq
    and transiently rebuilds ONE layer's full bf16 K/V for the attention (freed when the forward
    returns); because the pipeline runs layers sequentially only one layer's bf16 lives at a time,
    so the per-step PEAK is (all-layers-packed + one-layer-bf16) << all-layers-bf16. The KV
    reservation is tightened to exactly that peak — see _kv_bytes_per_layer / kv_reserve_probe."""

    class _TurboQuantLayer(dynamic_layer_cls):
        """A DynamicLayer that stores K/V QUANTIZED (TurboQuant) and reconstructs full bf16 on
        update(). Every method the model/cache uses for length + masking (get_seq_length, and via it
        get_mask_sizes) is overridden or routes through our token counter, so the causal mask stays
        correct; get_max_cache_shape (==-1, unbounded) + reset (no-op while uninitialized) are safe
        as inherited because we never lazy-initialize the bf16 self.keys/self.values buffers."""

        def __init__(self, config=None):
            super().__init__()
            self._tq = None                              # this layer's TurboQuantizer (built on 1st update)
            self._d = None                               # head_dim (needed to unpack the bit-packed idx)
            self._ki = self._kn = None                   # key   idx PACKED [b,h,S,ceil(d*kb/8)] uint8, norms [b,h,S,1]
            self._vi = self._vn = None                   # value idx PACKED / norms
            self._vq = self._vg = None                   # value QJL sign-bits (PACKED 1-bit) / gammas (None unless value_qjl)
            self._W = int(residual_window or 0)          # full-bf16 recent-token window (0 = whole-cache quant)
            self._rk = self._rv = None                   # bf16 residual buffers: the most-recent <=W tokens, UN-quantized
            self._len = 0                                # cached token count (drives get_seq_length + mask)

        def _q(self, k):
            if self._tq is None:
                self._tq = quantizer_factory(k.shape[-1], k.device, k.dtype)
            return self._tq

        def _append_quant(self, q, ek, ev):
            """Quantize + bit-pack a chunk of K/V (seq dim -2) and append it to the packed store."""
            import torch
            ki, kn = q.quant_keys(ek)                            # ki [b,h,q,d] uint8, values in [0,2^kb)
            vi, vn, vq, vg = q.quant_values(ev)
            ki = _pack_bits(torch, ki, q.kb)                     # -> [b,h,q,ceil(d*kb/8)]
            vi = _pack_bits(torch, vi, q.vb_mse)
            vq = _pack_bits(torch, vq.to(torch.uint8), 1) if vq is not None else None
            if self._ki is None:
                self._ki, self._kn = ki, kn
                self._vi, self._vn, self._vq, self._vg = vi, vn, vq, vg
            else:                                        # append the PACKED reps along the seq dim
                self._ki = torch.cat([self._ki, ki], dim=-2)
                self._kn = torch.cat([self._kn, kn], dim=-2)
                self._vi = torch.cat([self._vi, vi], dim=-2)
                self._vn = torch.cat([self._vn, vn], dim=-2)
                if vq is not None:
                    self._vq = torch.cat([self._vq, vq], dim=-2)
                    self._vg = torch.cat([self._vg, vg], dim=-2)

        def _deq_prefix(self, q, d):
            """Un-pack + dequant the WHOLE quantized store -> (keys, vals) bf16, or (None, None) if empty."""
            import torch
            if self._ki is None:
                return None, None
            ki_u = _unpack_bits(torch, self._ki, q.kb, d)        # unpack full seq -> uint8 idx (transient)
            vi_u = _unpack_bits(torch, self._vi, q.vb_mse, d)
            vq_u = _unpack_bits(torch, self._vq, 1, d).bool() if self._vq is not None else None
            keys = q.dequant_keys(ki_u, self._kn)                # full bf16, transient (returned only)
            vals = q.dequant_values(vi_u, self._vn, vq_u, self._vg)
            return keys, vals

        def update(self, key_states, value_states, *args, **kwargs):
            import torch
            q = self._q(key_states)
            d = int(key_states.shape[-1])
            self._d = d
            if self._W <= 0:
                # ---- whole-cache quant (original #172 path; byte-identical results) ----
                self._append_quant(q, key_states, value_states)
                self._len += int(key_states.shape[-2])
                return self._deq_prefix(q, d)
            # ---- residual-window path: hold the most-recent W tokens in FULL bf16 ----
            if self._rk is None:
                self._rk, self._rv = key_states, value_states
            else:                                        # append the new step to the bf16 residual buffer
                self._rk = torch.cat([self._rk, key_states], dim=-2)
                self._rv = torch.cat([self._rv, value_states], dim=-2)
            self._len += int(key_states.shape[-2])
            r = int(self._rk.shape[-2])
            if r > self._W:                              # age out the oldest (r-W) tokens into the quantized store
                e = r - self._W
                self._append_quant(q, self._rk[..., :e, :], self._rv[..., :e, :])
                self._rk = self._rk[..., e:, :]
                self._rv = self._rv[..., e:, :]
            qk, qv = self._deq_prefix(q, d)              # dequant the aged-out prefix (None until first eviction)
            if qk is None:
                return self._rk, self._rv
            keys = torch.cat([qk, self._rk], dim=-2)     # oldest (quantized) first, recent (bf16) last
            vals = torch.cat([qv, self._rv], dim=-2)
            return keys, vals

        def get_seq_length(self) -> int:
            return int(self._len)

        def crop(self, max_length: int) -> None:
            # spec-decode rollback: truncate to `max_length` tokens (seq dim = -2), across BOTH the
            # quantized prefix and (when W>0) the bf16 residual tail.
            if max_length < 0:
                max_length = self._len - abs(max_length)
            if self._len <= max_length:
                return
            def _sl(t, n):
                return t[..., :n, :] if t is not None else None
            if self._W <= 0:
                self._ki, self._kn = _sl(self._ki, max_length), _sl(self._kn, max_length)
                self._vi, self._vn = _sl(self._vi, max_length), _sl(self._vn, max_length)
                self._vq, self._vg = _sl(self._vq, max_length), _sl(self._vg, max_length)
                self._len = int(max_length)
                return
            rlen = int(self._rk.shape[-2]) if self._rk is not None else 0
            qlen = self._len - rlen                      # quantized prefix occupies [0, qlen)
            if max_length >= qlen:                       # keep all quantized, crop the residual tail
                keep = max_length - qlen
                self._rk, self._rv = _sl(self._rk, keep), _sl(self._rv, keep)
            else:                                        # drop the residual, crop into the quantized prefix
                self._rk = self._rv = None
                self._ki, self._kn = _sl(self._ki, max_length), _sl(self._kn, max_length)
                self._vi, self._vn = _sl(self._vi, max_length), _sl(self._vn, max_length)
                self._vq, self._vg = _sl(self._vq, max_length), _sl(self._vg, max_length)
            self._len = int(max_length)

        def reorder_cache(self, beam_idx) -> None:
            # beam search isn't used by this engine; reorder both stores if ever invoked.
            ref = self._ki if self._ki is not None else self._rk
            if ref is None:
                return
            bi = beam_idx.to(ref.device)
            for nm in ("_ki", "_kn", "_vi", "_vn", "_vq", "_vg", "_rk", "_rv"):
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
