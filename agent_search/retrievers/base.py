"""The retriever contract: index a corpus, rank it for a query.

Every built-in retriever subclasses `Retriever` and registers a builder in
`agent_search.retrievers.registry`; a user-provided one does the same. `Hit` and
`Observation` are the structured result the BQL and Indri executors return.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agent_search.corpus.units import CodeUnit as Unit


class Retriever(ABC):
    """Index a corpus of units, then rank them for a query.

    Built once per corpus identity (``key``) and reused across queries. Optional
    capabilities the harness reads with ``getattr``:

    * ``returns_full_set = True``: ``search`` returns the retriever's own complete ranking
      (an agent's surfaced set), so the harness must not pad it to the set-metric pool.
    * ``is_cached(key) -> bool``: a persistent index for ``key`` already exists on disk
      (lets ``build_indexes`` skip parsing the corpus).
    * ``search_with_scores(query, k) -> list[tuple[str, float]]``: scored ranking, when
      the engine has scores to show.
    """

    name: str = "retriever"

    @abstractmethod
    def index(self, units: Sequence[Unit], key: Optional[str] = None) -> "Retriever":
        """Build or load the index for a corpus. ``key`` is the corpus identity persistent
        backends cache under (e.g. the dataset name); ``None`` means in-memory only."""
        ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[str]:
        """Return up to ``k`` unit ``doc_id``s, best first."""
        ...


@dataclass
class Hit:
    """One ranked result with optional provenance (used by the structured executors)."""
    doc_id: str
    score: float
    path: str | None = None
    line: int | None = None
    region: str | None = None
    snippet: str | None = None


@dataclass
class Observation:
    """A structured retrieval result: what a search tool renders for the agent."""
    n_hits: int
    hits: Sequence[Hit]
    clause_df: dict[str, int] = field(default_factory=dict)  # per-clause document frequency
    typecheck_ok: bool = True
    error: str | None = None  # parse/type error -> the agent reacts instead of crashing


__all__ = ["Retriever", "Hit", "Observation", "Unit"]
