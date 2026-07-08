"""Fused grouped int4 MoE-expert GEMM for InfiniteModel — correctness + benchmark harness.

WHY: at batch-1 decode the eager MoE forward loops the top_k routed experts in PYTHON, doing a
separate `F.linear(x, gate_up_proj[e])` (+ down) per expert — each a tiny kernel launch plus
tensor-subclass dispatch. On the bandwidth-limited Strix Halo iGPU that per-expert launch/Python
overhead, not the GEMV math, dominates the MoE layer's decode cost. This harness validates and
benchmarks the FUSED path that client.py installs on a Qwen3-MoE-style experts module: ONE Triton
launch over all B = tokens*top_k applications (program (b, n) gathers expert eid[b]'s int4 tile),
replacing the Python loop.

It mirrors the worker's exact group-wise asymmetric int4 format and the fused kernel byte-for-byte,
and the per-expert reference mirrors transformers' Qwen3_5MoeExperts.forward, so PASS here means the
in-engine self-check will pass too.

Run on the AMD box:  ~/imenv/bin/python bench_moe_w4a16.py
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception as _e:                     # pragma: no cover
    _HAVE_TRITON = False
    print("triton import failed:", repr(_e))

G = 128


# --- packing (matches client.py _quantize_linear4 / _pack4_expert) ---------------------------
def pack_expert(W, group=G):
    """W [out, in] bf16 -> (qweight uint8 [out, in_pad//2], scale/zero bf16 [out, ng]), in_features."""
    out, in_f = W.shape
    ng = (in_f + group - 1) // group
    in_pad = ng * group
    Wp = F.pad(W, (0, in_pad - in_f)) if in_pad != in_f else W
    Wg = Wp.reshape(out, ng, group).float()
    wmin, wmax = Wg.amin(2), Wg.amax(2)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    zero = torch.round(-wmin / scale).clamp(0, 15)
    q = torch.round(Wg / scale.unsqueeze(2) + zero.unsqueeze(2)).clamp(0, 15).to(torch.uint8).reshape(out, in_pad)
    qpacked = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return qpacked, scale.to(torch.bfloat16), zero.to(torch.bfloat16), in_f


def dequant_expert(qw, scale, zero, in_f, group=G):
    """One expert int4 -> bf16 [out, in_f] (ground-truth reference dequant)."""
    out = qw.shape[0]
    lo = (qw & 0x0F).to(torch.int16)
    hi = (qw >> 4).to(torch.int16)
    q = torch.stack((lo, hi), dim=2).reshape(out, -1)
    ng = scale.shape[1]
    qf = q.reshape(out, ng, group).to(torch.bfloat16)
    w = (qf - zero.to(torch.bfloat16).unsqueeze(2)) * scale.to(torch.bfloat16).unsqueeze(2)
    return w.reshape(out, ng * group)[:, :in_f].contiguous()


# --- fused kernel (copy of client.py _w4a16_moe_op _mk, incl. its autotune set) -------------
if _HAVE_TRITON:
    # split-K MoE GEMV — byte-for-byte mirror of client.py _w4a16_moe_op _mk (incl. its autotune set)
    @triton.autotune(
        configs=[
            triton.Config({"BN": 128, "SPLITK": 1}, num_warps=4, num_stages=2),   # == prior default
            triton.Config({"BN": 128, "SPLITK": 2}, num_warps=4, num_stages=2),
            triton.Config({"BN": 128, "SPLITK": 4}, num_warps=4, num_stages=3),
            triton.Config({"BN": 128, "SPLITK": 8}, num_warps=4, num_stages=3),
            triton.Config({"BN": 64, "SPLITK": 4}, num_warps=2, num_stages=3),
            triton.Config({"BN": 64, "SPLITK": 8}, num_warps=4, num_stages=2),
            triton.Config({"BN": 256, "SPLITK": 4}, num_warps=8, num_stages=2),   # #dram-dealias
        ],
        key=["B", "N", "K", "sqn"],   # sqn: see client.py — pad-vs-unpadded variants tune apart
        reset_to_zero=["y_ptr"],   # atomic-acc kernel: zero y before each autotune trial + launch
    )
    @triton.jit
    def _mk(x_ptr, e_ptr, q_ptr, s_ptr, z_ptr, y_ptr, B, N, K,
            sxb, sxk, sqe, sqk, sqn, sse, ssn, ssg, sze, szn, szg, syb, syn,
            GROUP: tl.constexpr, BN: tl.constexpr, SPLITK: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)                            # which K-slice this program reduces
        e = tl.load(e_ptr + pid_b).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_h = tl.arange(0, GROUP // 2)
        nmask = offs_n < N
        ngroups = K // GROUP
        gps = (ngroups + SPLITK - 1) // SPLITK
        g0 = pid_k * gps
        acc = tl.zeros((BN,), dtype=tl.float32)
        for gi in range(0, gps):
            kb = g0 + gi
            if kb < ngroups:
                k0 = kb * GROUP
                xe = tl.load(x_ptr + pid_b * sxb + (k0 + 2 * offs_h) * sxk)
                xo = tl.load(x_ptr + pid_b * sxb + (k0 + 2 * offs_h + 1) * sxk)
                qp = q_ptr + e * sqe + (k0 // 2 + offs_h)[:, None] * sqk + offs_n[None, :] * sqn
                bb = tl.load(qp, mask=nmask[None, :], other=0).to(tl.int32)
                lo = (bb & 0xF).to(tl.float32)
                hi = ((bb >> 4) & 0xF).to(tl.float32)
                s = tl.load(s_ptr + e * sse + offs_n * ssn + kb * ssg, mask=nmask, other=0.0).to(tl.float32)
                z = tl.load(z_ptr + e * sze + offs_n * szn + kb * szg, mask=nmask, other=0.0).to(tl.float32)
                wlo = (lo - z[None, :]) * s[None, :]
                whi = (hi - z[None, :]) * s[None, :]
                acc += tl.sum(xe[:, None] * wlo, axis=0)
                acc += tl.sum(xo[:, None] * whi, axis=0)
        tl.atomic_add(y_ptr + pid_b * syb + offs_n * syn, acc, mask=nmask)

    def moe_op(x, eid, q, scale, zero, in_features, group=G):
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()
        Kpad = q.shape[2] * 2
        if x.shape[1] != Kpad:
            x = F.pad(x, (0, Kpad - x.shape[1]))
        eid = eid.to(torch.int32).contiguous()
        B, N = x.shape[0], q.shape[1]
        y = torch.zeros((B, N), device=x.device, dtype=torch.float32)   # split-K atomic-add accumulator
        grid = lambda meta: (B, triton.cdiv(N, meta["BN"]), meta["SPLITK"])  # noqa: E731
        _mk[grid](x, eid, q, scale, zero, y, B, N, Kpad,
                  x.stride(0), x.stride(1),
                  q.stride(0), q.stride(2), q.stride(1),
                  scale.stride(0), scale.stride(1), scale.stride(2),
                  zero.stride(0), zero.stride(1), zero.stride(2),
                  y.stride(0), y.stride(1), GROUP=group)
        return y.to(torch.bfloat16)


def act(x):
    return F.silu(x)


def ref_forward(x, idx, wts, GUq, GUs, GUz, GU_in, DNq, DNs, DNz, DN_in):
    """Per-expert ground truth — mirrors transformers Qwen3_5MoeExperts.forward exactly."""
    T = x.shape[0]
    final = torch.zeros_like(x)
    E = GUq.shape[0]
    expert_mask = F.one_hot(idx, num_classes=E).permute(2, 1, 0)
    hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for ei in hit:
        ei = ei[0]
        pos, tok = torch.where(expert_mask[ei])
        cur = x[tok]
        gu = F.linear(cur, dequant_expert(GUq[ei], GUs[ei], GUz[ei], GU_in))
        gate, up = gu.chunk(2, dim=-1)
        h = act(gate) * up
        dn = F.linear(h, dequant_expert(DNq[ei], DNs[ei], DNz[ei], DN_in))
        dn = dn * wts[tok, pos, None]
        final.index_add_(0, tok, dn.to(final.dtype))
    return final


def fused_forward(x, idx, wts, GUq, GUs, GUz, GU_in, DNq, DNs, DNz, DN_in):
    T, top_k = x.shape[0], idx.shape[1]
    eid = idx.reshape(-1)
    w = wts.reshape(-1).to(x.dtype)
    xb = x.repeat_interleave(top_k, dim=0)
    yb = moe_op(xb, eid, GUq, GUs, GUz, GU_in)
    gate, up = yb.chunk(2, dim=-1)
    h = act(gate) * up
    zb = moe_op(h, eid, DNq, DNs, DNz, DN_in) * w[:, None]
    final = torch.zeros_like(x)
    tok = torch.arange(T, device=x.device).repeat_interleave(top_k)
    final.index_add_(0, tok, zb.to(final.dtype))
    return final


def _bench(fn, iters=50):
    import time
    torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3


def main():
    assert torch.cuda.is_available(), "no GPU"
    dev = "cuda"
    print("device:", torch.cuda.get_device_name(0), "| hip:", torch.version.hip,
          "| triton:", (triton.__version__ if _HAVE_TRITON else "MISSING"))
    if not _HAVE_TRITON:
        return
    # (hidden H, moe-intermediate I, experts E, top_k) — qwen3-a3b-ish shapes
    configs = [(2048, 768, 128, 8), (4096, 1536, 128, 8)]
    print("\n== correctness (fused vs per-expert reference; in-engine threshold 0.03) ==")
    ok_all = True
    for (H, I, E, k) in configs:
        GU = (torch.randn(E, 2 * I, H, device=dev) * 0.02)
        DN = (torch.randn(E, H, I, device=dev) * 0.02)
        GUq = torch.empty(E, 2 * I, (H + G - 1) // G * G // 2, dtype=torch.uint8, device=dev)
        GUs = torch.empty(E, 2 * I, (H + G - 1) // G, dtype=torch.bfloat16, device=dev)
        GUz = torch.empty_like(GUs)
        DNq = torch.empty(E, H, (I + G - 1) // G * G // 2, dtype=torch.uint8, device=dev)
        DNs = torch.empty(E, H, (I + G - 1) // G, dtype=torch.bfloat16, device=dev)
        DNz = torch.empty_like(DNs)
        for e in range(E):
            GUq[e], GUs[e], GUz[e], GU_in = pack_expert(GU[e].bfloat16())
            DNq[e], DNs[e], DNz[e], DN_in = pack_expert(DN[e].bfloat16())
        for T in (1, 4):
            x = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1
            idx = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)])
            wts = torch.rand(T, k, device=dev, dtype=torch.bfloat16) + 0.1
            yr = ref_forward(x, idx, wts, GUq, GUs, GUz, GU_in, DNq, DNs, DNz, DN_in).float()
            yf = fused_forward(x, idx, wts, GUq, GUs, GUz, GU_in, DNq, DNs, DNz, DN_in).float()
            rel = ((yf - yr).abs().mean() / (yr.abs().mean() + 1e-6)).item()
            relmax = ((yf - yr).abs().max() / (yr.abs().max() + 1e-6)).item()   # in-engine gate: max<0.1
            ok = rel < 0.03 and relmax < 0.1
            flag = "ok" if ok else "FAIL"
            ok_all &= ok
            print(f"  H={H:5d} I={I:4d} E={E:3d} k={k} T={T}  rel={rel:.4f}  max={relmax:.4f}  {flag}")

        # decode (T=1) timing: per-expert reference vs fused (one MoE layer's experts)
        x = torch.randn(1, H, device=dev, dtype=torch.bfloat16) * 0.1
        idx = torch.randperm(E, device=dev)[:k].unsqueeze(0)
        wts = torch.rand(1, k, device=dev, dtype=torch.bfloat16) + 0.1
        t_ref = _bench(lambda: ref_forward(x, idx, wts, GUq, GUs, GUz, GU_in, DNq, DNs, DNz, DN_in))
        t_fus = _bench(lambda: fused_forward(x, idx, wts, GUq, GUs, GUz, GU_in, DNq, DNs, DNz, DN_in))
        print(f"  -> decode T=1 experts: ref={t_ref:.3f}ms  fused={t_fus:.3f}ms  speedup={t_ref / t_fus:.2f}x")
    print("\nCORRECTNESS:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
