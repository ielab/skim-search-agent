"""Hybrid retrieval: several standalone retrievers, one fusion method.

`HybridEngine` holds the component engines by name (any kind the engine registry builds:
`bm25`, `dense`, `bql`, `indri`, ...), queries each to a pool of `pool` candidates, and fuses
the rankings with a `Fusion` (`agent_search.retrievers.fusion`). The components and the method
come from the run: `retrieval.hybrid_retrievers` (default `bm25,dense`),
`retrieval.hybrid_fusion` (default `rrf`, or `interpolation`), `retrieval.hybrid_weights`
(interpolation only), `retrieval.rrf_k` and `listing.hybrid_pool`. The paper's hybrid arms are
the defaults: BM25 and the dense model, RRF with k = 60, 100 candidates per side.

`HybridRetriever` is the same thing as a standalone retriever (`strategy=hybrid`), for
retrieval-only evaluation.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.errors import SetupError
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.fusion import Fusion, build_fusion, rrf_k_from_env

DEFAULT_COMPONENTS = ("bm25", "dense")
DEFAULT_FUSION = "rrf"
DEFAULT_POOL = 100


def components_from_env() -> list[str]:
    """`HYBRID_RETRIEVERS`: comma-separated engine kinds (default `bm25,dense`)."""
    raw = os.environ.get("HYBRID_RETRIEVERS", ",".join(DEFAULT_COMPONENTS))
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if len(names) < 2:
        raise SetupError(f"HYBRID_RETRIEVERS names {names!r}; a hybrid needs at least two retrievers")
    return names


def fusion_from_env() -> Fusion:
    """`HYBRID_FUSION` (default `rrf`) with its parameters: `RRF_K`, or `HYBRID_WEIGHTS`
    (comma-separated floats, one per retriever) for interpolation."""
    name = (os.environ.get("HYBRID_FUSION") or DEFAULT_FUSION).strip().lower()
    if name == "rrf":
        return build_fusion("rrf", k=rrf_k_from_env())
    if name == "interpolation":
        raw = (os.environ.get("HYBRID_WEIGHTS") or "").strip()
        weights = [float(w) for w in raw.split(",") if w.strip()] or None
        return build_fusion("interpolation", weights=weights)
    return build_fusion(name)


def pool_from_env() -> int:
    return int(os.environ.get("HYBRID_POOL", str(DEFAULT_POOL)))


def scored_search(name: str, engine, query: str, k: int, needs_scores: bool) -> list:
    """`[(doc_id, score)]` best first from one component. An engine that reports ranks only
    serves RRF; interpolation needs scores and says so instead of guessing."""
    if hasattr(engine, "search_scored"):
        return list(engine.search_scored(query, k) or [])
    if needs_scores:
        raise SetupError(f"the {name!r} engine reports ranks only; interpolation needs scores")
    if hasattr(engine, "top_k_doc_ids"):
        ids = engine.top_k_doc_ids(query, k=k) or []
    else:
        ids = engine.search(query, k=k) or []
    return [(d, 0.0) for d in ids]


class HybridEngine:
    """The fused engine the tools use: `search(query, k)` -> doc ids best first."""

    def __init__(self, components: dict, fusion: Fusion, pool: int = DEFAULT_POOL):
        if len(components) < 2:
            raise SetupError("a hybrid needs at least two retrievers")
        self.components = dict(components)
        self.fusion = fusion
        self.pool = int(pool)

    def search(self, query: str, k: Optional[int] = None) -> list:
        rankings = [scored_search(name, eng, query, self.pool, self.fusion.needs_scores)
                    for name, eng in self.components.items()]
        return self.fusion.fuse(rankings, k)

    def top_k_doc_ids(self, query: str, k: Optional[int] = None) -> list:
        return self.search(query, k)

    def describe(self) -> dict:
        return {"retrievers": list(self.components), "pool": self.pool, **self.fusion.describe()}


class HybridRetriever(Retriever):
    """`strategy=hybrid`: the fused engine as a standalone retriever."""
    name = "hybrid"

    def __init__(self, components: Optional[Sequence[str]] = None, fusion: Optional[Fusion] = None,
                 pool: Optional[int] = None, index_root: str = "indexes", rebuild: bool = False,
                 dense_model: Optional[str] = None, engines=None):
        self.component_names = list(components) if components else components_from_env()
        self.fusion = fusion or fusion_from_env()
        self.pool = pool or pool_from_env()
        self.index_root, self.rebuild, self.dense_model = index_root, rebuild, dense_model
        self._engines = engines                  # a prebuilt registry (tests); else built in index()
        self._engine: Optional[HybridEngine] = None

    def index(self, units, key: Optional[str] = None) -> "HybridRetriever":
        if self._engines is None:
            from agent_search.retrievers.engines import Engines
            self._engines = Engines(units, key, index_root=self.index_root, rebuild=self.rebuild,
                                    dense_model=self.dense_model)
        components = {n: self._engines.get(n) for n in self.component_names}
        self._engine = HybridEngine(components, self.fusion, self.pool)
        return self

    def search(self, query: str, k: int) -> list:
        if self._engine is None:
            raise RuntimeError("call index() first")
        return self._engine.search(query, k)


# --- registry -------------------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("hybrid")
def _build_hybrid(cfg: RetrieverConfig, name: str):
    return lambda: HybridRetriever(index_root=cfg.index_root, rebuild=cfg.rebuild, dense_model=cfg.dense_model)


__all__ = ["HybridEngine", "HybridRetriever", "scored_search", "components_from_env", "fusion_from_env", "pool_from_env"]
