"""Standalone MXFP4 -> HF bf16 safetensors converter (gpt-oss).

gpt-oss ships its MoE expert weights in OCP microscaling FP4 (MXFP4): each fused 3D expert
weight is two tensors, `<name>_blocks` (uint8, 2 E2M1 codes/byte, 16 bytes = one 32-code block)
and `<name>_scales` (uint8 E8M0 per-block power-of-two). Native MXFP4 compute needs Hopper+;
the fleet (Ada / Ampere / RDNA3.5) has none. Rather than serve-time dequant of a NATIVELY-3D-fused
quantized MoE (a layout the fp8/nvfp4 serve path doesn't handle), InfiniteModel NORMALIZES gpt-oss
to a plain bf16 HF checkpoint ONCE at add/convert time — exactly the gguf_convert.py pattern. After
that gpt-oss is an ordinary bf16 model downstream: chunk-streamed, run on the distributed pipeline
(transformers' real GptOss modules handle attention sinks + clamped SwiGLU + router natively), and
int4 shard-cacheable once the [in,out] expert layout is taught to the packer (a follow-up).

MEMORY-BOUNDED BY DESIGN: a full from_pretrained(dequantize) would materialize ~42 GB (21B params
in bf16) and OOM a co-hosted box that is also serving. This converter instead streams ONE source
file at a time, dequantizes each expert tensor on its own, and writes per-shard safetensors — peak
RAM is a few GB (the largest single layer), so it is safe to run while the fleet serves.

Run as a SUBPROCESS (model_store), like gguf_convert.py:

    python mxfp4_convert.py <src_dir> <dst_dir>

<src_dir> is the HF snapshot dir holding config.json + *.safetensors (+ the safetensors index).
Prints a final ``MXFP4_CONVERT_OK <dst_dir>`` line on success; exits non-zero on failure.
"""
import json
import math
import os
import shutil
import sys


# E2M1 code -> value (mirrors shards._E2M1_VALUES / transformers FP4_VALUES). Codes 0..15.
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _dequant_mxfp4_to_bf16(blocks_u8, scales_u8):
    """Faithful mirror of transformers _convert_moe_packed_tensors (+ shards._dequant_mxfp4_to_bf16):
    blocks_u8 uint8 [E, out, G, 16] (2 E2M1 codes/byte, LOW nibble = even element),
    scales_u8 uint8 [E, out, G] (E8M0 biased exp; real exp = value-127)
    -> bf16 [E, in, out] (dequant then transpose(1,2)); in = G*32."""
    import torch
    blocks = blocks_u8.to(torch.uint8)
    scales = scales_u8.to(torch.int32) - 127
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]} != {scales.shape}"
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.bfloat16)
    *prefix, G, B = blocks.shape
    rows = math.prod(prefix) * G
    blk = blocks.reshape(rows, B)
    exp = scales.reshape(rows, 1)
    out = torch.empty(rows, B * 2, dtype=torch.bfloat16)
    out[:, 0::2] = lut[(blk & 0x0F).to(torch.int)]
    out[:, 1::2] = lut[(blk >> 4).to(torch.int)]
    torch.ldexp(out, exp, out=out)
    out = out.reshape(*prefix, G, B * 2).view(*prefix, G * B * 2)
    return out.transpose(1, 2).contiguous()


def _weight_map(src_dir: str) -> dict:
    """name -> source safetensors filename. Reads the index if present, else the single file."""
    idx = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as fh:
            return json.load(fh)["weight_map"]
    # single-file checkpoint
    one = "model.safetensors"
    if not os.path.exists(os.path.join(src_dir, one)):
        st = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
        if len(st) != 1:
            raise SystemExit(f"no index and {len(st)} safetensors in {src_dir}")
        one = st[0]
    from safetensors import safe_open
    with safe_open(os.path.join(src_dir, one), framework="pt") as f:
        return {k: one for k in f.keys()}


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: mxfp4_convert.py <src_dir> <dst_dir>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(os.path.join(src, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config") or {}
    if "mxfp4" not in str(qc.get("quant_method", "")).lower():
        print(f"[mxfp4-convert] {src} is not MXFP4-quantized (quant_method={qc.get('quant_method')!r})",
              file=sys.stderr)
        return 3

    wm = _weight_map(src)
    # Group output tensors by their SOURCE file so we open each shard once and stream in order.
    # An expert weight is emitted from its '<base>_blocks' source (consumes '<base>_scales'); the
    # '_scales' entries themselves are dropped. Everything else passes through unchanged.
    by_file: dict = {}
    for name, fn in wm.items():
        if name.endswith("_scales"):
            continue                                   # consumed alongside its _blocks
        by_file.setdefault(fn, []).append(name)

    os.makedirs(dst, exist_ok=True)
    files = sorted(by_file)
    n_shards = len(files)
    new_map: dict = {}
    total_bytes = 0
    opened: dict = {}

    def _src(fn: str):
        if fn not in opened:
            opened[fn] = safe_open(os.path.join(src, fn), framework="pt")
        return opened[fn]

    for si, fn in enumerate(files):
        shard_name = f"model-{si + 1:05d}-of-{n_shards:05d}.safetensors"
        tensors: dict = {}
        for name in by_file[fn]:
            if name.endswith("_blocks"):
                base = name[: -len("_blocks")]          # '...experts.gate_up_proj_blocks' -> '...gate_up_proj'
                sname = base + "_scales"
                if sname not in wm:
                    raise SystemExit(f"{name} has no matching {sname}")
                blocks = _src(fn).get_tensor(name)
                scales = _src(wm[sname]).get_tensor(sname)
                t = _dequant_mxfp4_to_bf16(blocks, scales)   # bf16 [E, in, out]
                out_name = base
            else:
                t = _src(fn).get_tensor(name)
                if t.dtype not in (torch.bfloat16, torch.float16):
                    t = t.to(torch.bfloat16)            # normalize f32 sidecars; keep bf16/f16 as-is
                t = t.contiguous()
                out_name = name
            tensors[out_name] = t
            new_map[out_name] = shard_name
            total_bytes += t.numel() * t.element_size()
        save_file(tensors, os.path.join(dst, shard_name),
                  metadata={"format": "pt"})
        del tensors
        print(f"[mxfp4-convert] wrote {shard_name} ({si + 1}/{n_shards})", flush=True)

    # safetensors weight index
    with open(os.path.join(dst, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": new_map}, fh)

    # config.json WITHOUT quantization_config -> downstream builds a plain bf16 GptOss
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    # copy tokenizer + aux files (everything that isn't weights / the old index / config)
    for f in os.listdir(src):
        if (f.endswith(".safetensors") or f == "model.safetensors.index.json"
                or f == "config.json" or f.startswith(".")):
            continue
        sp = os.path.join(src, f)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(dst, f))

    print(f"MXFP4_CONVERT_OK {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
