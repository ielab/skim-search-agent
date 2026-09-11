"""The reranker contract: reorder a candidate list for a query by reading the documents.

A retriever ranks ids from an index; a reranker scores (query, document text) pairs and
reorders a pool of candidates the retriever found. Rerankers live one file each under this
package, and `register_reranker` makes one selectable by name (`retrieval.rerank_method`).
The composition that turns "a retriever plus a reranker" into an engine is
`agent_search/retrievers/reranked.py`.

A reranker scores online: the pairs are read by the model during the run. That is the point
of reranking and is unlike document embeddings, which are always built ahead of a run.
"""
from __future__ import annotations

from typing import Optional, Sequence

RERANKERS: dict[str, type] = {}

Candidate = tuple[str, str]        # (doc_id, text)


class Reranker:
    name: str = ""

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


def build_reranker(name: str, **params) -> Reranker:
    """A reranker by name. Unknown names and parameters are errors, so a typo in an experiment
    file cannot silently fall back to a default."""
    try:
        cls = RERANKERS[name]
    except KeyError:
        raise ValueError(f"unknown reranker {name!r}; choose from {sorted(RERANKERS)}") from None
    return cls(**params)


__all__ = ["Reranker", "RERANKERS", "register_reranker", "build_reranker", "Candidate"]
