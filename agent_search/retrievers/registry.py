"""Retriever registry: the single extension point for retrieval conditions.

Adding a retrieval method or condition never touches the eval harness. Next to
your retriever, register a builder::

    from agent_search.retrievers.registry import register, RetrieverConfig

    @register("my_method")
    def _build(cfg: RetrieverConfig, name: str):
        return lambda: MyRetriever(...)        # a zero-arg factory -> Retriever

…then it is selectable as ``--retriever my_method`` and appears in ``available()``.
The eval driver only ever calls ``build_factory()`` / ``available()``; it has no
per-method knowledge. Heavy deps (torch, pyserini, vLLM) must be imported *inside*
the builder so registration stays cheap.

This registers a whole retriever (a standalone eval condition), not an engine: a tool asks
for the engine kinds it needs (its ``engines`` tuple, see ``tools/base.py``) from the
per-corpus registry in ``engines.py``. Every condition declared in ``agent_search.strategies``
registers itself here as ``agent_<name>`` through ``register_conditions`` (called below, after
the built-ins load).

## Plugins (out-of-tree retrievers)

Besides the built-in auto-discovery walk (below), two additional mechanisms let code
outside this package register retrievers without editing it:

  - env ``SKIMSEARCHAGENT_PLUGINS``: a comma-separated list of dotted module names,
    each imported once. A plugin module calls ``register(...)``/``register_dataset(...)``
    at its own import time, exactly like a built-in retriever module does.
  - entry points in group ``skimsearchagent.plugins``: any installed package can
    declare one (e.g. in ``pyproject.toml``: ``[project.entry-points."skimsearchagent.
    plugins"]`` / ``my_plugin = "my_pkg.my_module"``); each entry point is loaded and
    imported the same way.

Both are best-effort: a plugin that fails to import logs a warning (stderr) and is
skipped, since one broken plugin must never prevent every other retriever (built-in or
plugin) from registering.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

from agent_search.core.interfaces import Retriever

RetrieverFactory = Callable[[], Retriever]


@dataclass
class RetrieverConfig:
    """Everything a builder might need, filled from the eval CLI. Builders read
    only what they use; unused fields are harmless."""
    model: str | None = None
    dense_model: str | None = None
    index_root: str = "indexes"
    rebuild: bool = False
    policy: str = "stub"            # "stub" | "llm"
    backend: str = "vllm"          # "vllm" | "api"
    tp: int = 1
    api_base: str = "http://localhost:8000/v1"
    domain: str = "code"
    field_profile: str | None = None   # per-dataset BQL manual variant; None -> domain
    max_steps: int = 50
    prompt_override: str | None = None
    temperature: float = 0.6        # LLM sampling temperature (agent policies)
    seed: Optional[int] = 42         # sampling seed; fixed 42 for reproducibility (None disables)


# builder(cfg, name) -> zero-arg retriever factory. `name` lets one builder serve several
# condition names: every condition's `agent_<name>` retriever registers a builder that
# delegates to `build_condition_agent`, varying only the condition name it passes through.
Builder = Callable[["RetrieverConfig", str], RetrieverFactory]
_REGISTRY: dict[str, Builder] = {}


def register(*names: str) -> Callable[[Builder], Builder]:
    """Decorator: bind one builder to one or more condition names."""
    def deco(fn: Builder) -> Builder:
        for n in names:
            if n in _REGISTRY and _REGISTRY[n] is not fn:
                raise ValueError(f"retriever {n!r} is already registered")
            _REGISTRY[n] = fn
        return fn
    return deco


_loaded = False


def _ensure_loaded() -> None:
    """Import every retriever module so its @register runs. Discovery walks the
    `agent_search.retrievers` package, so a new method file self-registers with no edit
    here. Module top levels must stay light (heavy deps load lazily inside the builders),
    so this is cheap and import-safe even without torch/pyserini.

    `_loaded` is set only after the whole walk (built-ins + plugins) completes, not at
    the top of this function, so a module that raises partway through the walk leaves
    `_loaded` False and the next call retries the whole discovery instead of silently
    freezing the registry in a half-populated state for the rest of the process."""
    global _loaded
    if _loaded:
        return
    import importlib
    import pkgutil

    import agent_search.retrievers as _pkg

    def _onerror(module_name: str) -> None:
        # One broken module must never abort discovery of every other retriever:
        # log and keep walking (pkgutil calls this, then continues the walk itself).
        print(f"  [registry] WARNING: failed to import retriever module {module_name!r}: "
              f"{sys.exc_info()[1]!r} — skipping it", file=sys.stderr, flush=True)

    for _m in pkgutil.walk_packages(_pkg.__path__, _pkg.__name__ + ".", onerror=_onerror):
        try:
            importlib.import_module(_m.name)
        except Exception as e:                     # noqa: BLE001, same skip-and-continue policy
            print(f"  [registry] WARNING: failed to import retriever module {_m.name!r}: "
                  f"{e!r} — skipping it", file=sys.stderr, flush=True)
    # conditions declared in agent_search.strategies (tasks x strategies) register as agent_<name>
    try:
        from agent_search.evaluation.agent_runner import register_conditions
        register_conditions()
    except Exception as e:                         # noqa: BLE001
        print(f"  [registry] WARNING: failed to register conditions: {e!r} — skipping them",
              file=sys.stderr, flush=True)
    _load_plugins()
    _loaded = True


def _load_plugins() -> None:
    """Import plugin modules named in env `SKIMSEARCHAGENT_PLUGINS` (comma-separated
    dotted module names) and every entry point in group `skimsearchagent.plugins`. A
    plugin module registers things by calling `register(...)`/`register_dataset(...)`
    at its own import time; see this module's docstring's "Plugins" section. Missing/
    failed loads are tolerated (warning to stderr), never fatal to the rest of
    discovery."""
    import importlib

    for name in os.environ.get("SKIMSEARCHAGENT_PLUGINS", "").split(","):
        name = name.strip()
        if not name:
            continue
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"  [registry] WARNING: failed to import plugin module {name!r}: "
                  f"{e!r} — skipping it", file=sys.stderr, flush=True)

    try:
        from importlib.metadata import entry_points
    except Exception:
        return                                      # no importlib.metadata -> no entry-point plugins
    try:
        eps = entry_points(group="skimsearchagent.plugins")
    except Exception as e:
        print(f"  [registry] WARNING: failed to enumerate 'skimsearchagent.plugins' "
              f"entry points: {e!r} — skipping", file=sys.stderr, flush=True)
        return
    for ep in eps:
        try:
            ep.load()
        except Exception as e:
            print(f"  [registry] WARNING: failed to load plugin entry point {ep.name!r} "
                  f"({ep.value!r}): {e!r} — skipping it", file=sys.stderr, flush=True)


def build_factory(name: str, cfg: RetrieverConfig | None = None) -> RetrieverFactory:
    """Resolve a condition name to a zero-arg retriever factory."""
    _ensure_loaded()
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown retriever {name!r}; choose from {sorted(_REGISTRY)}") from None
    return builder(cfg or RetrieverConfig(), name)


def available() -> set[str]:
    """All registered condition names (built-ins + anything imported)."""
    _ensure_loaded()
    return set(_REGISTRY)
