"""BM25 baseline using the pure-Python engine (no Java/deps). The lexical floor."""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.ranking import BM25
from agent_search.core.interfaces import Retriever


class BM25Local(Retriever):
    name = "bm25_local"

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self._bm = BM25(k1=k1, b=b)

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "BM25Local":
        self._bm.index({u.doc_id: f"{u.qualname} {u.code}" for u in units})
        return self

    def search(self, query: str, k: int) -> list[str]:
        return [doc_id for doc_id, _ in self._bm.search(query, k=k)]


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("bm25_local")
def _build_bm25_local(cfg: RetrieverConfig, name: str):
    return lambda: BM25Local()
