"""Offline validation of the VRAM-aware planner (no cluster).
Checks that prefer_vram keeps layers on GPU first and only spills to CPU when needed."""
import server
from server import plan_pipeline, NodeMem, MODEL_SPECS, GB

spec = MODEL_SPECS["Qwen/Qwen2.5-7B-Instruct"]   # 28 layers, ~14 GB
GBi = int(GB)


def show(label, nodes):
    r = plan_pipeline(spec, nodes, ctx_len=2048, prefer_vram=True)
    print(f"\n== {label} ==  ok={r.ok}")
    if not r.ok:
        print("   ", r.error)
        return r
    for s in r.stages:
        print(f"    {s.hostname:10s} L{s.layer_start}-{s.layer_end} ({s.num_layers} layers) {s.est_gb:.1f} GB")
    return r


# A: 7B fits one GPU (beast VRAM 15) -> expect beast alone
A = show("A: beast 15GB VRAM + theocomp 10.6GB VRAM (7B fits one card)", [
    NodeMem("n1", "beast",    int(140.8*GB), int(15.0*GB)),
    NodeMem("n2", "theocomp", int(39.9*GB),  int(10.6*GB)),
    NodeMem("n3", "work",     int(13.2*GB),  0),
])

# B: shrink each GPU to 8GB so 7B (14GB) needs BOTH GPUs -> expect beast+theocomp, all on GPU
B = show("B: two 8GB GPUs (7B spans both, all on GPU)", [
    NodeMem("n1", "beast",    int(120*GB), int(8.0*GB)),
    NodeMem("n2", "theocomp", int(25*GB),  int(8.0*GB)),
    NodeMem("n3", "work",     int(13.2*GB), 0),
])

# C: one 8GB GPU + CPU node -> GPU fills, overflow spills to CPU
C = show("C: one 8GB GPU + a CPU node (overflow spills to CPU)", [
    NodeMem("n1", "beast", int(120*GB), int(8.0*GB)),
    NodeMem("n2", "work",  int(60*GB),  0),
])

# D: beast has a GPU but its VRAM tier is DISABLED (eff_vram -> vram_bytes 0), so
# the planner must treat beast as CPU-only and place GPU layers on theocomp.
D = show("D: beast VRAM tier disabled (vram_bytes=0) -> theocomp is the only GPU", [
    NodeMem("n1", "beast",    int(140.8*GB), 0),            # VRAM toggled off
    NodeMem("n2", "theocomp", int(39.9*GB),  int(10.6*GB)),
    NodeMem("n3", "work",     int(13.2*GB),  0),
])

print("\n== verdict (planner) ==")
# A: 7B fits beast's 15GB VRAM -> beast alone (fewest stages, all GPU)
print("A picks only beast        :", A.ok and {s.hostname for s in A.stages} == {"beast"})
# B: 7B needs >8GB -> spans both 8GB GPUs, all on GPU
print("B uses both GPUs          :", B.ok and {s.hostname for s in B.stages} == {"beast", "theocomp"})
# C: only one GPU; 7B fits beast (8GB GPU + own RAM spill) -> beast alone is right
print("C beast alone (GPU+RAM)   :", C.ok and {s.hostname for s in C.stages} == {"beast"})
# D: beast's GPU left the fast tier -> theocomp must carry GPU placement (beast may
# still appear as a CPU spill stage, but theocomp must be present).
print("D GPU work on theocomp    :", D.ok and "theocomp" in {s.hostname for s in D.stages})


# ---- Node-level tier-config logic (eff_*/usable_total_gb/load_device) ----
from server import Node, NODE_CONFIG

def mknode(host, ram_gb, vram_gb):
    return Node(node_id="n1", hostname=host, os="Linux", device="cuda:0",
                device_name="test", total_mem_gb=ram_gb + 2, usable_mem_gb=ram_gb,
                data_host="127.0.0.1", data_port=1, connected_at=0.0,
                last_heartbeat=0.0, vram_total_gb=vram_gb)

print("\n== verdict (Node tiers) ==")
# Default (no config) -> both tiers on, device auto, full usable.
NODE_CONFIG.clear()
n = mknode("box", 100.0, 12.0)   # usable_vram = 12 - VRAM_RESERVE(1) = 11
# m3v: both tiers on -> "" (no override; worker uses its own --device default)
both = (n.ram_enabled and n.vram_enabled and n.load_device() == ""
        and abs(n.usable_total_gb - (100.0 + 11.0)) < 1e-6)
print("default: both on, no-override:", both)

# VRAM disabled -> eff_vram 0, usable_total = RAM only, device cpu.
NODE_CONFIG["box"] = {"ram": True, "vram": False}
vram_off = (n.eff_vram_gb == 0.0 and abs(n.usable_total_gb - 100.0) < 1e-6
            and n.load_device() == "cpu")
print("vram off: cpu, ram-only    :", vram_off)

# RAM disabled (GPU-only) -> eff_ram 0, device gpu.
NODE_CONFIG["box"] = {"ram": False, "vram": True}
ram_off = (n.eff_ram_gb == 0.0 and abs(n.usable_total_gb - 11.0) < 1e-6
           and n.load_device() == "gpu")
print("ram off: gpu, vram-only    :", ram_off)

# Both off -> usable_total 0 (engine.load filters these out entirely).
NODE_CONFIG["box"] = {"ram": False, "vram": False}
both_off = n.usable_total_gb == 0.0
print("both off: usable_total 0    :", both_off)
NODE_CONFIG.clear()
