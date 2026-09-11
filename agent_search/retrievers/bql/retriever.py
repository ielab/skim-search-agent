"""Direct BQL retriever.

This is the loop-free retriever for already-formulated BQL strings (the `bql` entry in
`agent_search/strategies/retrieval_only.py`). It uses the same parser, type checker, and
structural executor the `search_bql` tool (`agent_search/tools/search_bql/`) lowers to;
the only difference is that there is no model loop translating a natural-language task
into the field-tagged surface. For that, run a condition whose strategy carries
`search_bql`: `agent_codefix` for code, `agent_research_snip` (the `sieve_bm25`
strategy) for documents.
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.core.units import Unit
from agent_search.core.interfaces import Retriever
from agent_search.retrievers.bql.executor import StructuralExecutor
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.types import check


class BQLRetriever(Retriever):
    name = "bql"

    def __init__(self):
        self._executor: StructuralExecutor | None = None
        self.last_error: str | None = None

    def index(self, units: Sequence[Unit], key: Optional[str] = None) -> "BQLRetriever":
        self._executor = StructuralExecutor(units)
        self.last_error = None
        return self

    def search(self, query: str, k: int) -> list[str]:
        return [doc_id for doc_id, _ in self.search_with_scores(query, k=k)]

    def search_with_scores(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return self.search_with_count(query, k=k)[0]

    def search_with_count(self, query: str, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """Return top-k BQL hits plus the untruncated Boolean match count.

        Parse/type errors are treated as empty retrieval results. That keeps the
        eval harness robust while making this condition usable only when the
        caller actually supplies BQL.
        """
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
    return lambda: BQLRetriever()
