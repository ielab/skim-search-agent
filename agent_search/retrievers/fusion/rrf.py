"""Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, 2009).

    score(d) = sum over the rankings that contain d of 1 / (k + rank(d))

with 1-based ranks. A document absent from a ranking contributes nothing for it, never a
penalty. Ties are broken by doc id ascending, so two documents on the same fused score come out
in the same order on every run. Only ranks are read; scores are ignored.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.retrievers.fusion.base import Fusion, Ranking, register_fusion

DEFAULT_K = 60


def rrf_k_from_env() -> int:
    """`RRF_K` (default 60): the constant the hybrid strategies use."""
    return int(os.environ.get("RRF_K", str(DEFAULT_K)))


@register_fusion
class RRF(Fusion):
    name = "rrf"
    needs_scores = False

    def __init__(self, k: int = DEFAULT_K):
        self.k = int(k)

    def fuse(self, rankings: Sequence[Ranking], k: Optional[int] = None) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _score) in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        ranked = sorted(scores, key=lambda d: (-scores[d], d))
        return ranked[:k] if k is not None else ranked

    def describe(self) -> dict:
        return {"fusion": self.name, "rrf_k": self.k}


__all__ = ["RRF", "DEFAULT_K", "rrf_k_from_env"]
