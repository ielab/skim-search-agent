"""Retriever registry — the single extension point for retrieval conditions.

Adding a retrieval method or condition does NOT touch the eval harness. Next to
your retriever, register a builder::

    from agent_search.retrievers.registry import register, RetrieverConfig

    @register("my_method")
    def _build(cfg: RetrieverConfig, name: str):
        return lambda: MyRetriever(...)        # a zero-arg factory -> Retriever

…then it is selectable as ``--retriever my_method`` and appears in ``available()``.
The eval driver only ever calls ``build_factory()`` / ``available()`` — it has no
per-method knowledge. Heavy deps (torch, pyserini, vLLM) must be imported *inside*
the builder so registration stays cheap.

NOT the same as ``agent/retriever.py``'s ``register_tool_engine`` — that registers an
engine the AGENT can call as a tool mid-episode; this registers a whole RETRIEVER
(a standalone eval condition).
"""
from __future__ import annotations

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


# builder(cfg, name) -> zero-arg retriever factory. `name` lets one builder serve
# several conditions (e.g. agent_bql/agent_grep share one builder, vary the tool).
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
    """Import every retriever module so its @register runs — AUTO-DISCOVERED by
    walking the `agent_search.retrievers` package, so a NEW method file self-registers
    with no edit here. Module top levels must stay light (heavy deps load lazily inside
    the builders), so this is cheap and import-safe even without torch/pyserini."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import importlib
    import pkgutil

    import agent_search.retrievers as _pkg
    for _m in pkgutil.walk_packages(_pkg.__path__, _pkg.__name__ + "."):
        importlib.import_module(_m.name)
    # the agent-as-retriever lives outside retrievers/ (it composes tools), import it too
    from agent_search.agent import retriever as _agent_ret                     # noqa: F401


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
