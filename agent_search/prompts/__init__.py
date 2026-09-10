"""Compatibility package: `agent_search.prompts` moved to `agent_search.legacy.prompts` (the
YAML prompt registry). Tasks are `agent_search.tasks`; manuals live with their tools."""
from __future__ import annotations

import importlib
import pkgutil
import sys

_NEW = "agent_search.legacy.prompts"
_pkg = importlib.import_module(_NEW)
sys.modules[__name__] = _pkg
for _info in pkgutil.iter_modules(_pkg.__path__):
    sys.modules[f"{__name__}.{_info.name}"] = importlib.import_module(f"{_NEW}.{_info.name}")
