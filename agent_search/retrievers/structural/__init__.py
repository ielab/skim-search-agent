"""Compatibility package: `agent_search.retrievers.structural.*` moved one level up.

`bql/` is `agent_search.retrievers.bql`, `indri/` is
`agent_search.retrievers.indri`, `lucene/` is `agent_search.retrievers.lucene` and
`structural/backend` is `agent_search.retrievers.backend`. Importing any old path returns the
same module object as the new one. Remove after one release.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys

_OLD = __name__
_NEW = "agent_search.retrievers"
_MOVED = ("bql", "indri", "lucene", "backend")


def _alias(name: str) -> None:
    mod = importlib.import_module(f"{_NEW}.{name}")
    sys.modules[f"{_OLD}.{name}"] = mod
    globals()[name] = mod
    path = getattr(mod, "__path__", None)
    if path:
        for info in pkgutil.iter_modules(path):
            sub = importlib.import_module(f"{_NEW}.{name}.{info.name}")
            sys.modules[f"{_OLD}.{name}.{info.name}"] = sub


for _n in _MOVED:
    _alias(_n)
