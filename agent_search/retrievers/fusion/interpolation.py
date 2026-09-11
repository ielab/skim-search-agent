"""Linear score interpolation.

Each ranking's scores are normalised to [0, 1] (min-max within that ranking, a constant list
maps to 1), then combined as `sum_i w_i * norm_i(d)`; a document absent from a ranking scores
0 there. Weights default to equal. Ties are broken by doc id ascending. Every component must
report scores, so a rank-only engine cannot be interpolated (the hybrid engine says so).
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.retrievers.fusion.base import Fusion, Ranking, register_fusion


def _minmax(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    values = [s for _, s in ranking]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {d: 1.0 for d, _ in ranking}
    return {d: (s - lo) / (hi - lo) for d, s in ranking}


@register_fusion
class Interpolation(Fusion):
    name = "interpolation"
    needs_scores = True

    def __init__(self, weights: Optional[Sequence[float]] = None):
        self.weights = [float(w) for w in weights] if weights else None

    def fuse(self, rankings: Sequence[Ranking], k: Optional[int] = None) -> list[str]:
        n = len(rankings)
        weights = self.weights or [1.0 / n] * n if n else []
        if len(weights) != n:
            raise ValueError(f"interpolation has {len(weights)} weights for {n} rankings")
        scores: dict[str, float] = {}
        for w, ranking in zip(weights, rankings):
            for doc_id, s in _minmax(ranking).items():
                scores[doc_id] = scores.get(doc_id, 0.0) + w * s
        ranked = sorted(scores, key=lambda d: (-scores[d], d))
        return ranked[:k] if k is not None else ranked

    def describe(self) -> dict:
        return {"fusion": self.name, "weights": self.weights}


__all__ = ["Interpolation"]
