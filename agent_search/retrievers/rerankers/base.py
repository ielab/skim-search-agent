"""The reranker contract: reorder a candidate list for a query by reading the documents.

A retriever ranks ids from an index; a reranker scores (query, document text) pairs and
reorders a pool of candidates the retriever found. Rerankers live one file each under this
package, and `register_reranker` makes one selectable by name (`retrieval.rerank_method`).
The composition that turns "a retriever plus a reranker" into an engine is
`agent_search/retrievers/reranked.py`.

A method not imported by the package is found by name: `build_reranker("x")` imports
`agent_search.retrievers.rerankers.x` and uses what it registers. A reranker file dropped into
this folder (a checkpoint that is not public yet, say) is selectable without a line changed
anywhere else.

A reranker scores online: the pairs are read by the model during the run. That is the point
of reranking and is unlike document embeddings, which are always built ahead of a run.
"""
from __future__ import annotations

from typing import Optional, Sequence

RERANKERS: dict[str, type] = {}

Candidate = tuple[str, str]        # (doc_id, text)


class Reranker:
    name: str = ""
    default_model: Optional[str] = None     # the checkpoint `retrieval.rerank_model: null` loads
    default_max_length: int = 512           # tokens per pair when `retrieval.rerank_max_length` is null

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: Optional[int] = None) -> list[tuple[str, float]]:
        """`[(doc_id, score)]` best first, at most `k` of them (all when `k` is None)."""
        raise NotImplementedError

    def describe(self) -> dict:
        """The method and its parameters, for the run record."""
        return {"reranker": self.name}


def register_reranker(cls: type) -> type:
    RERANKERS[cls.name] = cls
    return cls


def reranker_class(name: str) -> type:
    """The registered class for `name`, importing `agent_search.retrievers.rerankers.<name>`
    first when nothing registered it yet. Unknown names are errors, so a typo in an experiment
    file cannot silently fall back to a default."""
    if name not in RERANKERS:
        import importlib
        try:
            importlib.import_module(f"agent_search.retrievers.rerankers.{name}")
        except ModuleNotFoundError:
            pass
    try:
        return RERANKERS[name]
    except KeyError:
        raise ValueError(f"unknown reranker {name!r}; choose from {sorted(RERANKERS)}") from None


def build_reranker(name: str, **params) -> Reranker:
    """A reranker by name (`reranker_class`). Unknown parameters are errors."""
    return reranker_class(name)(**params)


__all__ = ["Reranker", "RERANKERS", "register_reranker", "reranker_class", "build_reranker", "Candidate"]
