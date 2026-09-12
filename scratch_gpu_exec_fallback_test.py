"""#gpu-exec-fallback test — a GPU that cannot execute this torch build must fall back to CPU,
on the CUDA side of the fleet as well as the ROCm side, and the rule must have ONE definition.

WHY THIS SHAPE. The bug this locks down was not "the fallback is missing" — the fallback was
there, in all three media leaves, and it had been exercised and verified on gfx1151. It was
keyed on a substring list containing ONLY ROCm markers (MIOpen / HIPRTC / hipErrorNoBinaryForGpu
/ "Code object build failed"), because ROCm was the only place it had ever been needed. On
amdcomp (GTX 1070, sm_61) the installed torch 2.11+cu128 ships no sm_61 cubins, so the warmup
raises `CUDA error: no kernel image is available for execution on the device` — which matches
none of those markers, so the guard was INERT there and the load hard-failed instead of falling
back. Three copies, all wrong the same way, all passing their own per-leaf tests.

So this asserts two things a per-leaf behavioural test cannot see:
  1. BEHAVIOUR — the real CUDA strings observed on amdcomp are matched, the ROCm ones still are,
     and unrelated exceptions are NOT (the negative control: a predicate that returns True for
     everything would "fix" the bug and silently swallow every real load error).
  2. PARITY — the substring list is spelled exactly once, and no leaf re-spells its own copy.

Source-level for (2) on purpose: it must run on the controller box, which has no torch.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from worker_hw import gpu_exec_unsupported  # noqa: E402

failures: list[str] = []


def check(name: str, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


# ---------------------------------------------------------------- 1. behaviour
# Verbatim from amdcomp (GTX 1070, sm_61, torch 2.11.0+cu128) — the strings that used to escape.
CUDA_REAL = [
    RuntimeError("CUDA error: no kernel image is available for execution on the device"),
    RuntimeError("cuDNN version 91900 is not compatible with devices with SM < 7.5. "
                 "Please install a version of PyTorch with a compatible cuDNN version."),
    RuntimeError("NVIDIA GeForce GTX 1070 with CUDA capability sm_61 is not compatible with "
                 "the current PyTorch installation."),
]
for e in CUDA_REAL:
    check(f"CUDA {type(e).__name__}: {str(e)[:48]}", gpu_exec_unsupported(e), True)

# ROCm/gfx1151 — the original reason the fallback exists. Must not regress.
ROCM_REAL = [
    RuntimeError("MIOpenDropoutHIP.cpp: '<utility>' file not found"),
    RuntimeError("HIP error: hipErrorNoBinaryForGpu"),
    RuntimeError("Code object build failed for gfx1151"),
]
for e in ROCM_REAL:
    check(f"ROCm {str(e)[:48]}", gpu_exec_unsupported(e), True)

# NEGATIVE CONTROL — a predicate that just returned True would pass everything above while
# silently converting every genuine load failure into a mystery CPU fallback.
UNRELATED = [
    FileNotFoundError("kokoro-v1_0.pth"),
    RuntimeError("unknown voice 'af_nope' (no af_nope.pt in voices/)"),
    ValueError("Trying to set a tensor of shape torch.Size([1]) as a parameter"),
    KeyError("model.encoder.layers.0.weight"),
    ConnectionResetError("peer closed the control link"),
]
for e in UNRELATED:
    check(f"unrelated {type(e).__name__}", gpu_exec_unsupported(e), False)

# OOM is opt-in: t2music treats "can't fit" as a CPU case, tts/stt must let the controller see it
# as the placement failure it is.
oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
check("OOM default (tts/stt)", gpu_exec_unsupported(oom), False)
check("OOM include_oom (t2music)", gpu_exec_unsupported(oom, include_oom=True), True)


# ---------------------------------------------------------------- 2. parity
# No leaf may carry its own copy of the marker list. One definition, in worker_hw.
LEAVES = ["worker_tts.py", "worker_stt.py", "worker_t2music.py", "worker_t2i.py", "worker_t2a.py"]
RESPELL = re.compile(r"""hip\s*=\s*any\(|["']hipErrorNoBinaryForGpu["']|["']Code object build failed["']""")

for leaf in LEAVES:
    p = os.path.join(ROOT, leaf)
    if not os.path.exists(p):
        continue
    src = open(p, encoding="utf-8").read()
    # Strip comments/docstring prose so a mention in a COMMENT can't make this pass or fail
    # for the wrong reason (a guard that "matched a comment" is how one shipped inert before).
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    if RESPELL.search(code):
        failures.append(f"{leaf}: re-spells the GPU-unsupported marker list inline — "
                        f"it must call worker_hw.gpu_exec_unsupported()")
    # and each leaf that HAS a cuda->cpu fallback must actually route through the helper
    if "_build(\"cpu\")" in code or "_b(\"cpu\")" in code:
        if "gpu_exec_unsupported" not in code:
            failures.append(f"{leaf}: has a CPU-fallback branch but never calls "
                            f"worker_hw.gpu_exec_unsupported()")

# The definition itself must exist exactly once across the tree.
defs = [f for f in os.listdir(ROOT)
        if f.endswith(".py") and not f.startswith("scratch_")
        and re.search(r"^def gpu_exec_unsupported", open(os.path.join(ROOT, f), encoding="utf-8",
                                                         errors="ignore").read(), re.M)]
check("single definition site", sorted(defs), ["worker_hw.py"])


# ---------------------------------------------------------------- report
if failures:
    print(f"FAIL ({len(failures)})")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PASS — CUDA + ROCm both fall back, unrelated errors still raise, "
      "OOM opt-in, one definition, no leaf re-spells it")
