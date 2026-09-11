"""The fusion contract: combine the rankings of several retrievers into one ranking.

A fusion method takes one ranked list per retriever, each entry a `(doc_id, score)` pair with
the best document first, and returns doc ids best first. `needs_scores` says whether the
method reads the scores (interpolation) or only the ranks (RRF). One file per method under this
package; `register_fusion` makes a method selectable by name (`retrieval.hybrid_fusion`).
"""
from __future__ import annotations

from typing import Optional, Sequence

FUSIONS: dict[str, type] = {}

Ranking = Sequence[tuple[str, float]]


class Fusion:
    name: str = ""
    needs_scores: bool = False

    def fuse(self, rankings: Sequence[Ranking], k: Optional[int] = None) -> list[str]:
        """Doc ids best first, at most `k` of them (all when `k` is None)."""
        raise NotImplementedError

    def describe(self) -> dict:
        """The method and its parameters, for the run record."""
        return {"fusion": self.name}


def register_fusion(cls: type) -> type:
    FUSIONS[cls.name] = cls
    return cls


def build_fusion(name: str, **params) -> Fusion:
    """A fusion method by name. Unknown parameters are an error, so a typo in an experiment
    file cannot silently fall back to a default."""
    try:
        cls = FUSIONS[name]
    except KeyError:
        raise ValueError(f"unknown fusion method {name!r}; choose from {sorted(FUSIONS)}") from None
    return cls(**params)


__all__ = ["Fusion", "FUSIONS", "Ranking", "register_fusion", "build_fusion"]
