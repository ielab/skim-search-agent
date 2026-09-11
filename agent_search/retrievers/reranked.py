"""Reranked retrieval: one standalone retriever, one reranker.

`RerankedEngine` asks the base engine (any kind the engine registry builds: `bm25`, `dense`,
`hybrid`, `bql`, ...) for a pool of `pool` candidates, hands their texts to a `Reranker`
(`agent_search.retrievers.rerankers`), and returns the reranked top `k`. The base, the
reranker and the pool come from the run: `retrieval.rerank_base` (default `bm25`),
`retrieval.rerank_method` (default `cross_encoder`), `retrieval.rerank_model` (the
reranker's model id), `retrieval.rerank_pool` (default 100).

`RerankedRetriever` is the same thing as a standalone retriever (`strategy=reranked`), for
retrieval-only evaluation. The `search_reranked` tool uses the engine inside an agent.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from agent_search.errors import SetupError
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.rerankers import Reranker, build_reranker
from agent_search.retrievers.rerankers.cross_encoder import DEFAULT_MODEL

DEFAULT_BASE = "bm25"
DEFAULT_METHOD = "cross_encoder"
DEFAULT_POOL = 100


def base_from_env() -> str:
    """`RERANK_BASE`: the engine kind that supplies the candidate pool (default `bm25`)."""
    name = (os.environ.get("RERANK_BASE") or DEFAULT_BASE).strip()
    if name == "reranked":
        raise SetupError("a reranked engine cannot rerank itself")
    return name


def reranker_from_env() -> Reranker:
    """`RERANK_METHOD` (default `cross_encoder`) with its model (`RERANK_MODEL`), batch size
    (`RERANK_BATCH_SIZE`) and pair length (`RERANK_MAX_LENGTH`)."""
    method = (os.environ.get("RERANK_METHOD") or DEFAULT_METHOD).strip().lower()
    params = {}
    if method == "cross_encoder":
        params = {"model": os.environ.get("RERANK_MODEL") or DEFAULT_MODEL,
                  "batch_size": int(os.environ.get("RERANK_BATCH_SIZE", "32")),
                  "max_length": int(os.environ.get("RERANK_MAX_LENGTH", "512"))}
    return build_reranker(method, **params)


def pool_from_env() -> int:
    return int(os.environ.get("RERANK_POOL", str(DEFAULT_POOL)))


def unit_text(u) -> str:
    """The text a reranker reads for a unit: title and body (or code) joined."""
    if u is None:
        return ""
    return "\n".join(p for p in (u.title, u.body if u.body is not None else u.code) if p)


class RerankedEngine:
    """The engine the tools use: `search(query, k)` -> doc ids best first, `search_scored` for
    the scores."""

    def __init__(self, base, reranker: Reranker, text_of: Callable[[str], str], pool: int = DEFAULT_POOL,
                 base_name: str = DEFAULT_BASE):
        self.base, self.reranker, self.text_of = base, reranker, text_of
        self.pool = int(pool)
        self.base_name = base_name

    def _pool_ids(self, query: str) -> list:
        if hasattr(self.base, "top_k_doc_ids"):
            return list(self.base.top_k_doc_ids(query, k=self.pool) or [])
        return list(self.base.search(query, k=self.pool) or [])

    def search_scored(self, query: str, k: Optional[int] = None) -> list:
        ids = self._pool_ids(query)
        if not ids:
            return []
        return self.reranker.rerank(query, [(d, self.text_of(d)) for d in ids], k)

    def search(self, query: str, k: Optional[int] = None) -> list:
        return [d for d, _ in self.search_scored(query, k)]

    def top_k_doc_ids(self, query: str, k: Optional[int] = None) -> list:
        return self.search(query, k)

    def describe(self) -> dict:
        return {"base": self.base_name, "pool": self.pool, **self.reranker.describe()}


class RerankedRetriever(Retriever):
    """`strategy=reranked`: the reranked engine as a standalone retriever."""
    name = "reranked"

    def __init__(self, base: Optional[str] = None, reranker: Optional[Reranker] = None,
                 pool: Optional[int] = None, index_root: str = "indexes", rebuild: bool = False,
                 dense_model: Optional[str] = None, domain: str = "general", engines=None):
        self.base_name = base or base_from_env()
        self.reranker = reranker or reranker_from_env()
        self.pool = pool or pool_from_env()
        self.index_root, self.rebuild, self.dense_model, self.domain = index_root, rebuild, dense_model, domain
        self._engines = engines                  # a prebuilt registry (tests); else built in index()
        self._engine: Optional[RerankedEngine] = None

    def index(self, units, key: Optional[str] = None) -> "RerankedRetriever":
        if self._engines is None:
            from agent_search.retrievers.engines import Engines
            self._engines = Engines(units, key, index_root=self.index_root, rebuild=self.rebuild,
                                    dense_model=self.dense_model, domain=self.domain)
        self._engine = RerankedEngine(self._engines.get(self.base_name), self.reranker,
                                      self._engines.text_of, self.pool, base_name=self.base_name)
        return self

    def search(self, query: str, k: int) -> list:
        if self._engine is None:
            raise RuntimeError("call index() first")
        return self._engine.search(query, k)

    def search_with_scores(self, query: str, k: int) -> list:
        if self._engine is None:
            raise RuntimeError("call index() first")
        return self._engine.search_scored(query, k)


# --- registry -------------------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("reranked")
def _build_reranked(cfg: RetrieverConfig, name: str):
    return lambda: RerankedRetriever(index_root=cfg.index_root, rebuild=cfg.rebuild,
                                     dense_model=cfg.dense_model, domain=cfg.domain)


__all__ = ["RerankedEngine", "RerankedRetriever", "unit_text", "base_from_env", "reranker_from_env", "pool_from_env"]
