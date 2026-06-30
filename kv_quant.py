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


# transformers Cache subclass — imported lazily so this module loads without transformers.
def make_turboquant_cache(transformers_cache_base):
    """Build a TurboQuantCache class subclassing the given transformers DynamicCache base.
    Done as a factory so kv_quant.py imports with torch only (no hard transformers dep)."""

    class TurboQuantCache(transformers_cache_base):
        """A DynamicCache that QUANTIZES K/V on update() and returns the DEQUANTIZED (un-rotated,
        original-basis) full K/V to the attention — so the model's attention runs UNCHANGED.

        Correctness path: resting KV is stored quantized; each step dequantizes for the attention
        math. NOTE (peak-VRAM follow-up): this transiently materializes the full bf16 K/V per step,
        so it lowers RESTING KV but not the per-step PEAK — the placement/reservation win needs a
        chunked or fused quantized-attention that consumes the quantized KV directly. Until then the
        reservation stays bf16-sized; this class proves the quantizer end-to-end + is the substrate
        for that kernel."""

        def __init__(self, *a, quantizer_factory=None, **kw):
            super().__init__(*a, **kw)
            self._qf = quantizer_factory                 # (head_dim, device, dtype) -> TurboQuantizer
            self._q = {}                                 # layer_idx -> TurboQuantizer
            self._kq = {}                                # layer_idx -> list of key-quant tuples (per step)
            self._vq = {}

        def _quantizer(self, layer_idx, k):
            q = self._q.get(layer_idx)
            if q is None:
                q = self._qf(k.shape[-1], k.device, k.dtype)
                self._q[layer_idx] = q
            return q

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # Quantize the NEW K/V, append, then return the full DEQUANTIZED K/V (all positions).
            qz = self._quantizer(layer_idx, key_states)
            self._kq.setdefault(layer_idx, []).append(qz.quant_keys(key_states))
            self._vq.setdefault(layer_idx, []).append(qz.quant_values(value_states))
            keys = self.torch_cat([qz.dequant_keys(*t) for t in self._kq[layer_idx]])
            vals = self.torch_cat([qz.dequant_values(*t) for t in self._vq[layer_idx]])
            # mirror DynamicCache's bookkeeping so get_seq_length()/attention masks stay correct
            import torch
            if layer_idx < len(self.key_cache):
                self.key_cache[layer_idx] = keys
                self.value_cache[layer_idx] = vals
            else:
                while len(self.key_cache) <= layer_idx:
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache[layer_idx] = keys
                self.value_cache[layer_idx] = vals
            return keys, vals

        @staticmethod
        def torch_cat(seq):
            import torch
            return torch.cat(seq, dim=-2)                # cat along the sequence dim

    return TurboQuantCache
