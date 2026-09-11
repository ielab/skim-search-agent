"""Direct BQL retriever: the `bql` floor.

The loop-free retriever for already-formulated BQL strings (the `bql` entry in
`agent_search/strategies/retrieval_only.py`). It uses the same parser, type checker and
engine the `search_bql` tool lowers to; the only difference is that no model loop turns a
natural-language task into the field-tagged surface. The engine follows the corpus kind
(`agent_search/retrievers/backend.py`): the Lucene structured index for documents, the
in-memory Boolean executor for a code repository.
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit as Unit
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.types import check


class BQLRetriever(Retriever):
    name = "bql"

    def __init__(self, index_root: str = "indexes", domain: str = "general"):
        self.index_root = index_root
        self.domain = domain
        self._executor = None
        self.last_error: str | None = None

    def index(self, units: Sequence[Unit], key: Optional[str] = None) -> "BQLRetriever":
        from agent_search.retrievers.backend import build_bql_engine
        self._executor = build_bql_engine(units, self.index_root, key, domain=self.domain)
        self.last_error = None
        return self

    def search(self, query: str, k: int) -> list[str]:
        return [doc_id for doc_id, _ in self.search_with_scores(query, k=k)]

    def search_with_scores(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return self.search_with_count(query, k=k)[0]

    def search_with_count(self, query: str, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """Top-k BQL hits plus the untruncated Boolean match count. A parse or type error is
        an empty result and is kept in `last_error`."""
        if self._executor is None:
            raise RuntimeError("BQLRetriever.index() must be called before search()")
        self.last_error = None
        parsed = parse(query)
        if not parsed.ok:
            self.last_error = f"parse error: {parsed.error}"
            return [], 0
        typed = check(parsed.expr)
        if not typed.ok:
            self.last_error = f"type error: {typed.error}"
            return [], 0
        return self._executor.run_with_count(parsed.expr, k=k)


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("bql")
def _build_bql(cfg: RetrieverConfig, name: str):
    return lambda: BQLRetriever(index_root=cfg.index_root, domain=cfg.domain)
