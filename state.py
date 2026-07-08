"""Shared controller-state registry for the split-out controller modules (m4c152 code-split).

WHY THIS EXISTS
---------------
server.py runs as ``__main__`` and owns the controller singletons (``engine``,
``registry``, ``ENGINE_CONFIG`` …) plus dozens of free helper functions. Code that
depends on those globals therefore cannot simply ``from server import engine`` — that
would re-execute server as a second module (double init / import cycle). To keep
server.py small *without* a package restructure, the big classes/route-blocks are
relocated VERBATIM into sibling flat modules (engine_load.py, engine_gen.py,
engine_lifecycle.py, routes_*.py …). Their method/function bodies are byte-identical
to the originals — the only thing that changes is *where* their module globals come
from: this registry injects server.py's namespace into them at startup.

HOW IT WORKS
------------
1. server.py imports the split modules (so ``class Engine(EngineLoadMixin, …)`` can be
   composed) — at import time the relocated bodies are merely *defined*, never run, so
   their bare global references (``registry``, ``log_activity``, ``ModelSpec`` …) do
   not need to resolve yet.
2. Early in ``main()`` (after ``ARGS`` is set, before serving) server.py calls
   ``state.publish(globals())`` then ``state.bind(engine_load, engine_gen, …)``.
   ``publish`` snapshots server's full module namespace; ``bind`` copies it into each
   split module's ``__dict__`` so the relocated bodies resolve their former globals.
3. ``publish`` also mirrors every name onto THIS module, so route modules that prefer
   the explicit form can read ``state.engine`` / ``state.registry`` / ``state.cfg``.

SAFETY NOTE (the "ENCODING hazard")
-----------------------------------
A snapshot shares object *references*, so in-place mutation (``ENGINE_CONFIG[...] = x``,
``engine.models[...] = m``) is visible everywhere. What it does NOT track is a *rebind*
(``global X; X = new_obj``). Two valid patterns keep rebound globals coherent:
(1) STAY-BEHIND: the DOWNLOAD_STATE group's rebinder + reader both stay in server.py.
(2) CANONICAL-HOME-MOVES (Inc 11): ``ENCODING`` moved to media_encode.py TOGETHER with all
four of its ``global ENCODING`` mutators; the one outside reader (server.py's self-update
idle lambda) reads ``media_encode.ENCODING`` as a live module attribute, and the name is
never back-imported or published (an int snapshot would freeze and decouple the gate).
If you later relocate code that rebinds a shared global, use pattern (2), make the global
canonical HERE (``state.X``), or keep rebinder+reader together — never split them.

DEPLOY
------
This is a controller-only leaf module (stdlib only, never imports server). It is listed
in server.py's EXTRA_UPDATE_FILES so the multi-file self-update keeps it in sync across
the fleet, and server.py imports it through the standard CONVERGENCE BRIDGE (pull-once
from GitHub raw if an old checkout doesn't have it yet).
"""
from __future__ import annotations

import sys

_NS: dict = {}
_mod = sys.modules[__name__]


def publish(server_globals: dict) -> None:
    """Register server.py's module namespace. Call ONCE in main() after ARGS is set.

    Snapshots every non-dunder global so the split modules can resolve the controller's
    functions/classes/singletons/config by name, and mirrors them onto this module so
    ``state.<name>`` also works (the explicit form used by the route modules).
    """
    _NS.clear()
    for k, v in server_globals.items():
        if k.startswith("__"):
            continue
        _NS[k] = v
        setattr(_mod, k, v)


def bind(*modules) -> None:
    """Inject the published namespace into each module's globals.

    Relocated method/function bodies live in these modules but reference server.py's
    former module globals by bare name; this makes those names resolve. Harmless to
    over-inject (unused names are ignored), so the FULL namespace is delivered — that
    guarantees no reference is ever missed.
    """
    if not _NS:
        raise RuntimeError("state.bind() called before state.publish()")
    for m in modules:
        m.__dict__.update(_NS)
