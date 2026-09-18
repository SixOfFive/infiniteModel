"""#t2a-bf16-gate test — a Pascal (sm<80) card must NOT be a valid ACE-Step t2a target; an
Ampere+ (sm>=80) card must be.

WHY (found 2026-09-18): `can_t2a` was advertised purely from `find_spec("acestep")` succeeding.
amdcomp's GTX 1070 (Pascal sm_61) imports acestep fine and reported can_t2a=True — so the controller
placed a music load onto it, and it CRASHED MID-RENDER with the cuDNN error
`RuntimeError('GET was unable to find an engine to execute this computation')`, because ACE-Step's
M1 pipeline is bf16-only and Pascal has no bf16 engine. The fix requires real bf16 hardware (CUDA
compute capability >= (8, 0), the same threshold worker_quant's tinygemm gate and
perf_profile.classify_device already use) so those cards report can_t2a=False and a request fails
CLEANLY at placement instead of partway through a render.

WHAT THIS SHAPE BUYS. Two independent predicates enforce the one rule — worker_hw._t2a_bf16_capable
(the worker withholds can_t2a on a pre-Ampere card) and engine_load._t2a_capable (a controller
backstop that rejects a can_t2a node whose reported compute_cap is a known pre-Ampere one, covering
OLD workers in the self-update convergence window). Both must agree on the (8, 0) threshold, and the
placement filters must actually route through the controller predicate rather than reading the raw
`can_t2a` flag. This tests the REAL shipped code of both — the worker helper by import, the
controller predicate by AST-extracting and executing the actual function (engine_load can't be
imported: the controller box has no torch) — so it can't pass while the shipped rule disagrees.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures: list[str] = []


def _src(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def _extract_func(src: str, name: str):
    """Compile+exec JUST the named top-level-or-nested FunctionDef, so we test the shipped code
    without importing the whole module (engine_load pulls torch, absent on the controller)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            ns: dict = {}
            exec(compile(ast.parse(seg), f"<{name}>", "exec"), ns)
            return ns[name]
    return None


class _FakeNode:
    """Minimal stand-in for a registry Node — only the attrs _t2a_capable reads."""
    def __init__(self, can_t2a, compute_cap):
        self.can_t2a = can_t2a
        self.compute_cap = compute_cap


# ---- capability cases: (compute_cap, bf16-capable?, name) ---------------------------------------
PASCAL = (6, 1)      # GTX 1070, Quadro P620 — the cards that triggered this
TURING = (7, 5)      # sm_75, still no native bf16
AMPERE = (8, 6)      # RTX 3060 — first tier with bf16
CASES = [(PASCAL, False, "GTX1070/P620 Pascal"), (TURING, False, "Turing sm75"),
         ((8, 0), True, "A100 sm80"), (AMPERE, True, "RTX3060 sm86"),
         ((8, 9), True, "4070TiS sm89"), ((12, 0), True, "RTX5090")]

# ---------------------------------------------------------------- 1. worker helper (by import)
try:
    import worker_hw
    wcap = worker_hw._t2a_bf16_capable
    for cap, want, nm in CASES:
        if wcap(cap) is not want:
            failures.append(f"worker_hw._t2a_bf16_capable({cap}) = {wcap(cap)}, want {want} ({nm})")
    # None / no-GPU / ROCm / empty -> the worker knows its own hw has no bf16 GPU -> False
    for bad in (None, (), (7,)):
        if wcap(bad) is not False:
            failures.append(f"worker_hw._t2a_bf16_capable({bad!r}) must be False (no bf16 GPU)")
except Exception as exc:
    failures.append(f"could not import/exercise worker_hw._t2a_bf16_capable: {exc!r}")

# ---------------------------------------------------------------- 2. controller predicate (real fn)
ctrl = _extract_func(_src("engine_load.py"), "_t2a_capable")
if ctrl is None:
    failures.append("engine_load.py: _t2a_capable is GONE — the controller t2a bf16 backstop")
else:
    # can_t2a=True + a KNOWN pre-Ampere cap -> rejected (the whole point)
    for cap, ok, nm in CASES:
        got = ctrl(_FakeNode(True, list(cap)))
        if bool(got) is not ok:
            failures.append(f"engine_load._t2a_capable(can_t2a=True, cap={cap}) = {got}, "
                            f"want {ok} ({nm})")
    # unknown cap (old worker predating #sm-probe) -> TRUST can_t2a (never slander an absent field)
    if ctrl(_FakeNode(True, None)) is not True:
        failures.append("engine_load._t2a_capable: unknown compute_cap must TRUST can_t2a "
                        "(a pre-#sm-probe worker legitimately advertised it)")
    # no runtime advertised -> not a target regardless of a modern card
    if ctrl(_FakeNode(False, list(AMPERE))) is not False:
        failures.append("engine_load._t2a_capable: can_t2a=False must never be a t2a target")

# ---------------------------------------------------------------- 3. worker GATES can_t2a on it
whw = _src("worker_hw.py")
m = re.search(r'reg\["can_t2a"\]\s*=\s*(.+)', whw)
if not m:
    failures.append('worker_hw.py: reg["can_t2a"] assignment not found')
elif "_t2a_bf16_capable" not in m.group(1):
    failures.append('worker_hw.py: reg["can_t2a"] no longer gated on _t2a_bf16_capable — a Pascal '
                    'card with the acestep package would advertise can_t2a=True again')

# ---------------------------------------------------------------- 4. placement filters route through it
# The t2a candidate filters must consult _t2a_capable, NOT the raw can_t2a flag — otherwise a stale
# worker's can_t2a=True on Pascal reaches placement and crashes mid-render. The ONLY bare
# getattr(..,"can_t2a"..) allowed is the one INSIDE _t2a_capable's own body.
el = _src("engine_load.py")
el_tree = ast.parse(el)
_fn = next((n for n in ast.walk(el_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_t2a_capable"), None)
_fn_span = (_fn.lineno, _fn.end_lineno) if _fn else (0, 0)
for m in re.finditer(r'getattr\([^,]+,\s*"can_t2a"', el):
    ln = el[:m.start()].count("\n") + 1
    if not (_fn_span[0] <= ln <= _fn_span[1]):
        failures.append(f"engine_load.py:{ln}: reads raw can_t2a via getattr OUTSIDE _t2a_capable — "
                        f"route t2a placement through _t2a_capable so the bf16 gate applies")

# ---------------------------------------------------------------- 5. both sides share the (8,0) threshold
for fn, needle, why in (
        ("worker_hw.py", "(8, 0)", "worker_hw._t2a_bf16_capable lost the (8, 0) bf16 threshold"),
        ("engine_load.py", "(8, 0)", "engine_load._t2a_capable lost the (8, 0) bf16 threshold")):
    fn_src = _extract_func(_src(fn),
                           "_t2a_bf16_capable" if fn == "worker_hw.py" else "_t2a_capable")
    # re-read the function's own source text for the literal threshold
    tree = ast.parse(_src(fn))
    seg = next((ast.get_source_segment(_src(fn), n) for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and n.name in ("_t2a_bf16_capable", "_t2a_capable")), "")
    if needle not in seg:
        failures.append(f"{fn}: {why} (expected literal {needle} in the predicate)")

if failures:
    print("FAIL — #t2a-bf16-gate:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS — Pascal (sm_61) rejected + Ampere accepted on BOTH sides; "
      f"{len(CASES)} caps checked; placement filters route through _t2a_capable; "
      f"unknown-cap trusts can_t2a; (8, 0) threshold shared.")
