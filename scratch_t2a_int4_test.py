"""#t2a-int4 test — worker_t2a._quantize_dit_int4 quantizes the RIGHT layers of an ACE-Step-style
DiT block and preserves the forward within int4 error.

WHY THIS SHAPE. The int4 win depends on quantizing ACE-Step's FF, whose big layers are 1x1
nn.Conv1d (GLUMBConv), NOT nn.Linear — quantizing only the linears would leave the majority of the
DiT bf16 and miss the fit. So this asserts the shipped quantizer's contract on a synthetic block
that mirrors ACE-Step's shapes:
  * a 1x1 nn.Conv1d (the FF's pointwise convs)      -> replaced by an int4 QuantLinear4 wrapper;
  * an nn.Linear (attention/embedder projections)    -> QuantLinear4;
  * a DEPTHWISE nn.Conv1d (k=3, groups=C)            -> LEFT bf16 (not a pointwise matmul);
  * forward output within int4 error; weight bytes shrink.
Runs the REAL worker_t2a._quantize_dit_int4 (not a transcription). Skips cleanly where torch /
worker_quant are unavailable (e.g. the controller box has no torch)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch
    from torch import nn
    import worker_quant
    import worker_t2a
except Exception as exc:  # controller has no torch — this test is a worker-box test
    print(f"SKIP — torch/worker_quant/worker_t2a unavailable: {exc!r}")
    sys.exit(0)

torch.manual_seed(0)
DT = torch.bfloat16
fails: list[str] = []


def _mbytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) + \
        sum(b.numel() * b.element_size() for b in m.buffers())


def _rel(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-8)


class Block(nn.Module):
    """Mirrors an ACE-Step block's quantizable shapes (dims multiples of the int4 group size)."""
    def __init__(self, dim=256, hidden=1024):
        super().__init__()
        self.pw_in = nn.Conv1d(dim, hidden, 1)                               # 1x1 pointwise -> int4
        self.dw = nn.Conv1d(hidden, hidden, 3, groups=hidden, padding=1)     # depthwise    -> bf16
        self.pw_out = nn.Conv1d(hidden, dim, 1)                              # 1x1 pointwise -> int4
        self.attn_q = nn.Linear(dim, dim)                                    # attention proj -> int4
        self.norm = nn.LayerNorm(dim)                                        # -> bf16


blk = Block().to(DT).eval()
QL4 = worker_quant._quant4_linear_cls()

x_seq = torch.randn(2, 16, 256, dtype=DT)     # [B, L, C] for the Linear
x_conv = torch.randn(2, 256, 16, dtype=DT)    # [B, C, L] for the 1x1 conv
with torch.no_grad():
    y_lin0 = blk.attn_q(x_seq)
    y_pw0 = blk.pw_in(x_conv)
b0 = _mbytes(blk)

worker_t2a._quantize_dit_int4(blk)            # <-- the shipped quantizer under test
b1 = _mbytes(blk)

# 1) the RIGHT layers were converted / left alone
if isinstance(blk.pw_in, nn.Conv1d):
    fails.append("pw_in 1x1 conv NOT quantized (still nn.Conv1d)")
if isinstance(blk.pw_out, nn.Conv1d):
    fails.append("pw_out 1x1 conv NOT quantized (still nn.Conv1d)")
if not isinstance(blk.dw, nn.Conv1d):
    fails.append("depthwise conv WAS quantized (must stay bf16 — it is not a pointwise matmul)")
if not isinstance(blk.attn_q, QL4):
    fails.append(f"attn_q Linear -> {type(blk.attn_q).__name__}, want QuantLinear4")
if not (hasattr(blk.pw_in, "q") and isinstance(blk.pw_in.q, QL4)):
    fails.append("pw_in wrapper does not hold a QuantLinear4")

# 2) forward within int4 error (drop-in equivalence)
with torch.no_grad():
    r_lin = _rel(blk.attn_q(x_seq), y_lin0)
    r_pw = _rel(blk.pw_in(x_conv), y_pw0)
if r_lin > 0.15:
    fails.append(f"Linear int4 rel_err {r_lin:.3f} > 0.15")
if r_pw > 0.15:
    fails.append(f"1x1-conv int4 rel_err {r_pw:.3f} > 0.15")

# 3) it actually shrank the weights
shrink = b0 / max(b1, 1)
if shrink < 2.0:
    fails.append(f"block shrank only {shrink:.2f}x (want >= 2x)")

if fails:
    print("FAIL — #t2a-int4:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"PASS — 1x1-conv FF + Linear int4'd, depthwise conv left bf16; "
      f"rel_err lin={r_lin:.3f} pw={r_pw:.3f}; weights shrank {shrink:.2f}x")
