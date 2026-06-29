"""Triton w4a16 GEMM for InfiniteModel int4 — correctness + benchmark harness.

WHY: on AMD RDNA (e.g. Strix Halo gfx1151) torch's fused int4 kernel
(_weight_int4pack_mm) is CDNA2+-only, so int4 decode falls back to the naive path
that rematerializes the whole bf16 weight every token (~3.6x slower, GPU-bound). This
kernel reads the packed int4 weights and dequantizes inside the GEMM, matching the
worker's exact group-wise asymmetric format:

    qweight : uint8  [N, K//2]   byte j -> col 2j = low nibble, col 2j+1 = high nibble
    scale   : bf16   [N, ng]     ng = K // group
    zero    : bf16   [N, ng]
    w[n,i]  = (nibble(n,i) - zero[n, i//G]) * scale[n, i//G]
    y       = x @ w.T            (F.linear semantics)

Run on an AMD box:  ~/imenv/bin/python bench_w4a16.py
Validates rel-error vs the naive dequant (must beat the 0.05 self-check threshold the
worker uses) and benchmarks decode (M=1) naive-vs-triton.
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


if _HAVE_TRITON:
    @triton.jit
    def _w4a16_kernel(
        x_ptr, q_ptr, s_ptr, z_ptr, y_ptr,
        M, N, K,
        sxm, sxk, sqk, sqn, ssn, ssg, szn, szg, sym, syn,
        GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_h = tl.arange(0, GROUP // 2)            # byte index within a K-group
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        nblocks = K // GROUP
        for kb in range(0, nblocks):
            k0 = kb * GROUP
            m_mask = offs_m[:, None] < M
            xe = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 2 * offs_h)[None, :] * sxk,
                         mask=m_mask, other=0.0)                       # [BM, GROUP/2]
            xo = tl.load(x_ptr + offs_m[:, None] * sxm + (k0 + 2 * offs_h + 1)[None, :] * sxk,
                         mask=m_mask, other=0.0)
            # packed weight tile [GROUP/2, BN] (byte index outer, output channel inner)
            qp = q_ptr + (k0 // 2 + offs_h)[:, None] * sqk + offs_n[None, :] * sqn
            b = tl.load(qp, mask=offs_n[None, :] < N, other=0).to(tl.int32)
            lo = (b & 0xF).to(tl.float32)
            hi = ((b >> 4) & 0xF).to(tl.float32)
            s = tl.load(s_ptr + offs_n * ssn + kb * ssg, mask=offs_n < N, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + offs_n * szn + kb * szg, mask=offs_n < N, other=0.0).to(tl.float32)
            wlo = ((lo - z[None, :]) * s[None, :]).to(tl.bfloat16)     # [GROUP/2, BN]
            whi = ((hi - z[None, :]) * s[None, :]).to(tl.bfloat16)
            acc += tl.dot(xe.to(tl.bfloat16), wlo)
            acc += tl.dot(xo.to(tl.bfloat16), whi)
        yp = y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn
        tl.store(yp, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def w4a16(x, qweight, scale, zero, group=128, BM=16, BN=128):
        """x [M,K] bf16, qweight uint8 [N,K//2], scale/zero bf16 [N,K//group] -> y [M,N] bf16."""
        x = x.contiguous()
        M, K = x.shape
        N = qweight.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _w4a16_kernel[grid](
            x, qweight, scale, zero, y, M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(1), qweight.stride(0),
            scale.stride(0), scale.stride(1),
            zero.stride(0), zero.stride(1),
            y.stride(0), y.stride(1),
            GROUP=group, BM=BM, BN=BN,
        )
        return y


def naive_dequant(qweight, scale, zero, G):
    out = qweight.shape[0]
    lo = (qweight & 0x0F).to(torch.int16)
    hi = (qweight >> 4).to(torch.int16)
    q = torch.stack((lo, hi), dim=2).reshape(out, -1)
    ng = scale.shape[1]
    qf = q.reshape(out, ng, G).to(torch.bfloat16)
    w = (qf - zero.to(torch.bfloat16).unsqueeze(2)) * scale.to(torch.bfloat16).unsqueeze(2)
    return w.reshape(out, ng * G).contiguous()


def _rand_int4(N, K, G, dev):
    ng = K // G
    qweight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(N, ng, device=dev) * 0.02 + 0.001).to(torch.bfloat16)
    zero = torch.randint(0, 16, (N, ng), device=dev).to(torch.bfloat16)
    return qweight, scale, zero


def _bench(fn, iters=50):
    torch.cuda.synchronize()
    import time
    fn(); torch.cuda.synchronize()                 # warmup/compile
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3          # ms/call


def main():
    assert torch.cuda.is_available(), "no GPU"
    dev = "cuda"
    print("device:", torch.cuda.get_device_name(0), "| hip:", torch.version.hip,
          "| triton:", (triton.__version__ if _HAVE_TRITON else "MISSING"))
    if not _HAVE_TRITON:
        return
    G = 128
    shapes = [(2048, 2048), (4096, 4096), (4096, 14336), (5120, 5120)]
    print("\n== correctness (rel err vs naive bf16 dequant; worker threshold 0.05) ==")
    ok_all = True
    for (K, N) in shapes:
        for M in (1, 8, 64):
            qw, sc, ze = _rand_int4(N, K, G, dev)
            x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1)
            yt = w4a16(x, qw, sc, ze, G).float()
            yn = F.linear(x, naive_dequant(qw, sc, ze, G)).float()
            rel = ((yt - yn).abs().mean() / (yn.abs().mean() + 1e-6)).item()
            flag = "ok" if rel < 0.05 else "FAIL"
            ok_all &= rel < 0.05
            print(f"  K={K:5d} N={N:5d} M={M:3d}  rel={rel:.4f}  {flag}")
    print("CORRECTNESS:", "PASS" if ok_all else "FAIL")

    print("\n== decode (M=1) naive vs triton ==")
    for (K, N) in shapes:
        qw, sc, ze = _rand_int4(N, K, G, dev)
        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16) * 0.1
        t_naive = _bench(lambda: F.linear(x, naive_dequant(qw, sc, ze, G)))
        t_tri = _bench(lambda: w4a16(x, qw, sc, ze, G))
        print(f"  K={K:5d} N={N:5d}  naive={t_naive:.3f}ms  triton={t_tri:.3f}ms  "
              f"speedup={t_naive / t_tri:.2f}x")


if __name__ == "__main__":
    main()
