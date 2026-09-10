"""SkimSearchAgent — a research framework for deep-search agents over your own corpus.

Quick programmatic use::

    from agent_search import research
    result = research("Which treaty ended the Mexican-American War?", docs, model="gpt-4o-mini")

Layers (each a directory, each replaceable through a documented contract — see
``agent_search.core.interfaces`` and docs/EXTENDING.md):

* ``corpus``      — documents -> retrievable ``Unit``s; dataset loaders live in
                    ``evaluation.datasets``.
* ``retrievers``  — BM25 (local / Lucene), dense, hybrid fusion, BQL fielded retrieval,
                    Indri-style structured retrieval; ``retrievers.registry`` is the plugin point.
* ``tools``       — one folder per atomic tool: its declaration, its code, its manual.
* ``tasks``       — one folder per task: the prompt template and the answer protocol.
* ``strategies``  — combinations of tools with options; a condition is a task with a strategy.
* ``agent``       — the reason-act-observe loop, policies, the Agents-SDK driver.
* ``models``      — model providers behind one ``messages -> text`` callable.
* ``evaluation``  — the experiment runner, metrics, judge, and the run record.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("skimsearchagent")
except PackageNotFoundError:  # running from a checkout without an install
    __version__ = "0.0.0+local"

__all__ = ["__version__", "research", "build_agent", "ResearchResult", "STRATEGIES"]


def __getattr__(name: str):
    # Lazy: keep `import agent_search` free of prompt/YAML and evaluation imports.
    if name in ("research", "build_agent", "ResearchResult", "as_units"):
        from agent_search import api
        return getattr(api, name)
    if name == "STRATEGIES":
        from agent_search.strategies.names import STRATEGIES
        return STRATEGIES
    raise AttributeError(name)
