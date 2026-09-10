"""Compatibility package: `agent_search.agent.tools` moved to `agent_search.legacy.workspaces`
(the pre-0.3 workspaces). The atomic tools are `agent_search.tools`."""
from __future__ import annotations

import importlib
import pkgutil
import sys

_NEW = "agent_search.legacy.workspaces"
_pkg = importlib.import_module(_NEW)
sys.modules[__name__] = _pkg
for _info in pkgutil.iter_modules(_pkg.__path__):
    sys.modules[f"{__name__}.{_info.name}"] = importlib.import_module(f"{_NEW}.{_info.name}")
