"""The engines a run builds once per corpus and every tool shares.

`Engines(units, key, ...)` hands out the BM25 engine, the dense engine (`DenseBelief`), the BQL
executor in its three rankings (plain BM25, fused with the dense model, dense only), the Indri
executor, the hybrid engine (several of the others fused, `agent_search.retrievers.hybrid`) and
the reranked engine (one of the others, reordered by a reranker, `agent_search.retrievers.reranked`).
Each is built on first request under a lock, so concurrent episodes never load the same index
twice, and persisted under `index_root` keyed by the corpus. Document corpora rank on Lucene
only; `domain="code"` selects the in-memory Boolean executor for a repository
(`agent_search.retrievers.backend`). A missing dense cache or Lucene index is a `SetupError`:
nothing is encoded or indexed during a run.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional

from agent_search.errors import SetupError


class Engines:
    def __init__(self, units, key: Optional[str], *, index_root: str = "indexes", rebuild: bool = False,
                 dense_model: Optional[str] = None, domain: str = "general"):
        self.units, self.key = units, key
        self.index_root, self.rebuild = index_root, rebuild
        self.domain = domain
        self.dense_model = dense_model
        self._lock = threading.Lock()
        self._built: dict[str, Any] = {}

    # --- what a tool asks for, by kind -----------------------------------------------------
    def get(self, kind: str):
        builders = {"bm25": self.bm25, "dense": self.dense, "bql": self.bql, "bql_fused": self.bql_fused,
                    "bql_dense": self.bql_dense_only, "bql_plain": self.bql_plain, "indri": self.indri,
                    "hybrid": self.hybrid, "reranked": self.reranked}
        if kind not in builders:
            raise ValueError(f"unknown engine kind {kind!r}; choose from {sorted(builders)}")
        return builders[kind]()

    def build(self, kinds) -> dict:
        """Build (or load) every kind in `kinds` now; the harness calls this before the first
        episode so a missing artifact stops the run there."""
        return {k: self.get(k) for k in kinds}

    # --- the engines ---------------------------------------------------------------------
    def bm25(self):
        with self._lock:
            if "bm25" not in self._built:
                from agent_search.retrievers.lexical import build_bm25_engine
                self._built["bm25"] = build_bm25_engine(self.units, self.index_root, self.rebuild, self.key)
        return self._built["bm25"]

    def dense(self):
        with self._lock:
            return self._dense_locked()

    def _dense_locked(self):
        if "dense" not in self._built:
            from agent_search.retrievers.dense import DenseBelief, DenseRetriever
            model = self.dense_model or self._default_dense_model()
            probe = DenseRetriever(model=model, index_root=self.index_root, encoder=object())
            if not self.rebuild and not probe.is_cached(self.key):
                raise SetupError(
                    f"no persisted dense embedding cache for corpus key {self.key!r} and model {model!r} at "
                    f"{probe._cache_dir(self.key)!r}; build it with: skimsearchagent-build-indexes --retriever dense "
                    f"--model {model} ... (documents are never encoded online during a run)")
            self._built["dense"] = DenseBelief(model=model, index_root=self.index_root).build_or_load(self.units, key=self.key)
        return self._built["dense"]

    def _default_dense_model(self) -> str:
        from agent_search.evaluation.datasets import default_dense_model
        return default_dense_model(self.domain)

    def bql(self):
        """The BQL executor as the paper's BM25-ranked Sieve uses it; `BQL_DENSE=1` fuses the
        dense model in when its cache exists (degrading with a warning otherwise)."""
        with self._lock:
            if "bql" not in self._built:
                from agent_search.retrievers.backend import build_bql_engine
                from agent_search.retrievers.bql.dense_fuse import bql_dense_enabled
                dense = None
                if bql_dense_enabled():
                    try:
                        dense = self._dense_locked()
                    except Exception as e:  # noqa: BLE001
                        print(f"WARNING: BQL_DENSE=1 but the dense engine could not be built ({e!r}); "
                              f"degrading to plain bm25 BQL ranking.", file=sys.stderr)
                self._built["bql"] = build_bql_engine(self.units, self.index_root, self.key, dense=dense, domain=self.domain)
        return self._built["bql"]

    def bql_plain(self):
        """The BQL executor with no dense model (the code task)."""
        with self._lock:
            if "bql" not in self._built:
                from agent_search.retrievers.backend import build_bql_engine
                self._built["bql"] = build_bql_engine(self.units, self.index_root, self.key, dense=None, domain=self.domain)
        return self._built["bql"]

    def bql_fused(self):
        """The BQL executor ranking with BM25 and the dense model fused (Sieve)."""
        with self._lock:
            if "bql" not in self._built:
                from agent_search.retrievers.backend import build_bql_engine
                self._built["bql"] = build_bql_engine(self.units, self.index_root, self.key,
                                                      dense=self._dense_locked(), domain=self.domain)
        return self._built["bql"]

    def bql_dense_only(self):
        """The BQL executor ranking with the dense model alone (`sieve_dense`)."""
        with self._lock:
            if "bql" not in self._built:
                from agent_search.retrievers.backend import build_bql_engine_dense_only
                self._built["bql"] = build_bql_engine_dense_only(self.units, self.index_root, self.key,
                                                                 dense=self._dense_locked(), domain=self.domain)
        return self._built["bql"]

    def indri(self):
        """The Indri executor; `INDRI_DENSE=1` attaches the dense model, degrading with a warning."""
        with self._lock:
            if "indri" not in self._built:
                from agent_search.retrievers.backend import build_indri_engine
                dense = None
                if os.environ.get("INDRI_DENSE") in ("1", "true", "yes"):
                    try:
                        dense = self._dense_locked()
                    except Exception as e:  # noqa: BLE001
                        print(f"WARNING: INDRI_DENSE=1 but the dense engine could not be built ({e!r}); "
                              f"degrading to lexical-only Indri retrieval.", file=sys.stderr)
                self._built["indri"] = build_indri_engine(self.units, self.index_root, self.key, dense=dense, domain=self.domain)
        return self._built["indri"]

    def hybrid(self):
        """The fused engine over the retrievers the run names (`HYBRID_RETRIEVERS`, default
        `bm25,dense`) with the run's fusion method (`HYBRID_FUSION`, default `rrf`)."""
        with self._lock:
            if "hybrid" not in self._built:
                from agent_search.retrievers.hybrid import HybridEngine, components_from_env, fusion_from_env, pool_from_env
                names = components_from_env()
                if "hybrid" in names:
                    raise SetupError("a hybrid cannot contain itself")
                components = {}
                for n in names:
                    if n not in self._built:
                        self._built[n] = self._build_unlocked(n)
                    components[n] = self._built[n]
                self._built["hybrid"] = HybridEngine(components, fusion_from_env(), pool_from_env())
        return self._built["hybrid"]

    def reranked(self):
        """The reranked engine: the base retriever the run names (`RERANK_BASE`, default `bm25`)
        supplies a pool, the reranker (`RERANK_METHOD`, `RERANK_MODEL`) reorders it
        (`agent_search.retrievers.reranked`)."""
        with self._lock:
            if "reranked" not in self._built:
                from agent_search.retrievers.reranked import RerankedEngine, base_from_env, pool_from_env, reranker_from_env
                base = base_from_env()
                if base not in self._built:
                    self._built[base] = self._build_unlocked(base)
                self._built["reranked"] = RerankedEngine(self._built[base], reranker_from_env(), self.text_of,
                                                         pool_from_env(), base_name=base)
        return self._built["reranked"]

    def text_of(self, doc_id: str) -> str:
        """The text of one unit (title and body, or code), what a reranker reads."""
        from agent_search.retrievers.reranked import unit_text
        by_id = self.units.by_id if getattr(self.units, "lazy", False) else self._by_id()
        return unit_text(by_id.get(doc_id))

    def _by_id(self) -> dict:
        if not hasattr(self, "_ubyid"):
            self._ubyid = {u.doc_id: u for u in self.units}
        return self._ubyid

    def _build_unlocked(self, kind: str):
        """Build one kind while the lock is already held (the lock is not re-entrant)."""
        self._lock.release()
        try:
            return self.get(kind)
        finally:
            self._lock.acquire()

    # the harness records which engines were built and reads them for tests
    @property
    def built(self) -> dict:
        return dict(self._built)


__all__ = ["Engines"]
