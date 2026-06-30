# CPU / RAM worker node

Pure-CPU workers in the InfiniteModel fleet. These nodes carry no GPU; they exist to add
**capacity** — fitting models too big for the GPU pool by holding pipeline stages in system
RAM — **not** single-stream speed. CPU decode stays slow (see [Optimal settings](#4-optimal-settings)).

> Cross-links: int4-on-CPU decode details and the CPU fp32-GEMM notes live in
> [../ACCELERATION.md](../ACCELERATION.md). The GPU kernel recipes are in
> [../ROCM.md](../ROCM.md) (ROCm) and [../ACCELERATION.md](../ACCELERATION.md) (NVIDIA) — neither
> applies here; the Triton kernels are GPU-only.

---

## 1. Overview

| | |
|---|---|
| **Hosts** | PVE `nuc01`–`nuc04` + `mini05` (Proxmox VMs/CTs), `prodesk`, `zippy`, `tablet` |
| **GPU / arch** | none — pure CPU |
| **VRAM** | n/a; capacity is bounded by the box's system RAM (and its `MemoryMax` cap) |
| **OS** | Linux (Proxmox guests) for the PVE/mini nodes; the others as configured (verify) |
| **Role in fleet** | Capacity tier — hold pipeline stages in RAM so the controller can place a model that won't fit the GPU pool. Distributed, the fleet's CPU/RAM pool fits much larger models than any single GPU. |

CPU workers are placed **faster-RAM-first**: the controller prefers DDR5 over DDR4 over DDR3,
since CPU decode is DDR-bandwidth-bound (~50–90 GB/s, vs 150–670 GB/s on a GPU — see
[../ACCELERATION.md](../ACCELERATION.md)).

---

## 2. Install

A Python venv + the **CPU** torch build + the worker deps. The torch CPU wheel index is the
key difference from a GPU node — do **not** install a CUDA/ROCm wheel here.

```bash
# 1) venv (use the box's python3)
python3 -m venv ~/imenv

# 2) PyTorch — CPU build (no CUDA/ROCm runtime pulled in)
~/imenv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) App deps (same pinned versions as the rest of the fleet — from ../ROCM.md)
~/imenv/bin/python -m pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

> The dep set and pins are quoted from [../ROCM.md](../ROCM.md) (the CUDA fleet uses the same
> versions). No build toolchain (`gcc` / `python3-dev`) is needed on a CPU worker — that
> requirement is for the Triton JIT, which is GPU-only and never taken on CPU.

---

## 3. Run the worker

Launch `client.py` pointed at the controller, with `--device cpu`:

```bash
~/imenv/bin/python client.py --device cpu --controller <beast-ip>
```

`--controller` overrides `config.json`'s `controller_host`; prefer the flag over editing
`config.json` (the controller is BEAST at its LAN IP, control port 50100 by default —
quote the IP your fleet uses).

### Persistence — systemd unit with a memory cap

CPU workers run under a **systemd unit** (system unit, or `--user`). The cap is the important
part: set `MemoryMax=<cap>` and `MemorySwapMax=0` so that an over-budget load is a **clean
OOM-kill** (the controller detects the drop and replans) instead of the host freezing on swap
thrash. Size `<cap>` to leave headroom for the OS.

```ini
# /etc/systemd/system/im-worker.service   (system unit; for --user, drop User= and
# install under ~/.config/systemd/user/, then `systemctl --user enable --now im-worker`)
[Unit]
Description=InfiniteModel CPU worker
After=network-online.target
Wants=network-online.target

[Service]
User=<user>
WorkingDirectory=%h/infinitemodel
ExecStart=%h/imenv/bin/python client.py --device cpu --controller <beast-ip>
Restart=always
RestartSec=5
# Memory safety — clean OOM-kill -> controller replan, never a host freeze:
MemoryMax=<cap>
MemorySwapMax=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now im-worker
```

For a `--user` unit, run `loginctl enable-linger "$USER"` once so the service starts at boot
without an active login (per [../ROCM.md](../ROCM.md)).

---

## 4. Optimal settings

- **Quant: int4.** Smallest footprint → most capacity per box, and at batch-1 the int4 unpack
  is paid regardless, so the fp32 weight is effectively free. Dense int4 on CPU uses torch's
  CPU tinygemm (`_weight_int4pack_mm_for_cpu`) when present, otherwise a dequant→fp32 GEMM
  (the faster CPU path at batch-1). MoE experts always bf16-rematerialize per expert. See
  [../ACCELERATION.md](../ACCELERATION.md).
- **Context:** keep modest — CPU RAM holds both weights and the KV cache, and longer context
  multiplies KV. There is no per-tier "best" ctx documented here; size it to the box's RAM and
  the model (verify against your `MemoryMax`).
- **Placement mode:** CPU workers participate as **pipeline stages** in a distributed model;
  they are the spill/capacity tier behind the GPU nodes. CPU tensor-parallel is **not** worth
  it here — per [../ACCELERATION.md](../ACCELERATION.md), CPU-TP never beats pipelining onto a
  GPU on this fleet.
- **Models that fit/run well:** the value case is a **big model that won't fit the GPU pool**,
  placed across CPU RAM. For tolerable tok/s prefer a **sparse MoE with a low active-param
  ratio** (≈3B active) over a big dense model of similar quality — decode reads only the active
  bytes per token (see "Choosing a fast model" in [../ACCELERATION.md](../ACCELERATION.md)).
- **What to avoid:**
  - Don't expect speed — a dense model decodes ~all params/token on slow DDR; a CPU worker
    adds capacity, not single-stream throughput (~5–10× slower than a GPU regardless of kernel).
  - The GPU kernel work (fused MoE, split-K dense, autotune) does **nothing** on CPU — none of
    those Triton kernels run off-`cuda` (the self-check returns False). Don't set
    `INFINITEMODEL_CUDA_FUSED_MOE` on a CPU box; it has no effect.

---

## 5. Verify

1. **torch is the CPU build, no GPU expected:**
   ```bash
   ~/imenv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())"
   ```
   `cuda avail` should be `False` on a pure-CPU node, and a matmul must run:
   ```bash
   ~/imenv/bin/python -c "import torch; a=torch.randn(2048,2048); (a@a).sum().item(); print('cpu matmul OK')"
   ```
   (Adapted from the ROCm sanity check in [../ROCM.md](../ROCM.md), with the device left on CPU.)
2. **Worker registered with the controller:** check the controller dashboard / `GET /status`
   on `<beast-ip>:21434` and confirm the node's hostname appears with its CPU/RAM.
3. **Logs:** `GET /logs?node=<host>` on the controller streams this worker's log on its
   heartbeat (per [../ACCELERATION.md](../ACCELERATION.md)'s log reference).
4. **Gen test:** place/load a model that spans this node and run one short generation through
   the controller API; confirm coherent output (it will be slow — that's expected for CPU).

---

## 6. Gotchas

- **Set `MemoryMax` + `MemorySwapMax=0`.** Without the cap (or with swap enabled) an
  over-budget load thrashes swap and **freezes the host**; with the cap the kernel OOM-kills
  the worker cleanly and the controller **replans**. This is the whole reason CPU workers run
  under systemd here.
- **One worker per box.** Two workers sharing a hostname fight over the controller
  registration (noted for GPU boxes in [../ACCELERATION.md](../ACCELERATION.md); same applies).
- **CPU is capacity, not speed.** Don't troubleshoot "slow decode" on a CPU node — it's
  bandwidth-bound by design. If you need speed, place the model on a GPU.
- **The GPU acceleration env switches are inert here.** `INFINITEMODEL_CUDA_FUSED_MOE`,
  split-K, CUDA-graph, etc. are GPU-only ([../ACCELERATION.md](../ACCELERATION.md)). The one
  switch that *is* CPU-relevant is `INFINITEMODEL_PREFILL_CHUNK` (default 2048): it chunks long
  prefill so the SDPA **math** backend — which the CPU path uses — doesn't materialize the full
  `[1, H, q, total]` score tensor and OOM on long prompts. Standard-attention models only;
  lower it (512–1024) on a memory-tight box, `0` to disable.
- **Faster RAM first.** Placement prefers DDR5 > DDR4 > DDR3; a DDR3 box will be the slowest
  link in a pipeline that spans it.
