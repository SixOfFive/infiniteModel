#!/usr/bin/env python3
"""Recompute the runtime module closure for the package allowlist.

Static AST walk of every top-level `.py` in the repo root, following only
imports of the repo's OWN modules, seeded from server.py (controller) and
client.py (worker). Prints the union closure and the excluded (dev/bench/test)
modules. Add the two subprocess-invoked converters (gguf_convert, mxfp4_convert)
by hand — nothing *imports* them, so a closure cannot see them.

Usage:  python3 packaging/tools/closure.py [REPO_ROOT]
"""
import ast
import os
import sys

root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
local = {f[:-3] for f in os.listdir(root) if f.endswith(".py")}


def imports_of(path: str) -> set[str]:
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read(), path)
    except Exception:
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out & local


def closure(entries: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(entries)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        p = os.path.join(root, m + ".py")
        if os.path.exists(p):
            stack.extend(d for d in imports_of(p) if d not in seen)
    return seen


ctrl = closure(["server"])
work = closure(["client"])
union = sorted(ctrl | work)
print(f"controller closure ({len(ctrl)}):\n  " + " ".join(sorted(ctrl)))
print(f"worker closure ({len(work)}):\n  " + " ".join(sorted(work)))
print(f"UNION ({len(union)}):\n  " + " ".join(union))
print("  + subprocess tools (add by hand): gguf_convert mxfp4_convert")
print(f"EXCLUDED ({len(local) - len(union)}) — dev/bench/test, NOT shipped:\n  "
      + " ".join(sorted(local - set(union))))
