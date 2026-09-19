"""InfiniteModel — distributed LLM/multimodal inference across mixed nodes.

This is the thin *launcher* package. It carries no inference logic of its own;
the runtime source tree (``server.py``, ``client.py`` and ~45 sibling modules)
ships as bundled payload under ``infinitemodel/_payload/`` and is materialised
into a writable app directory at first run — see ``_launch.py`` for why.

``__version__`` is stamped from the runtime's ``server.py`` ``VERSION`` constant
by the packaging build (packaging/build.sh); the ``0.0.0`` fallback here only
appears if someone imports the un-built source tree directly.
"""

__version__ = "0.0.0"  # replaced at build time with server.py's VERSION
