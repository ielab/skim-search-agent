"""Structural retrieval backends: `structural/indri` (Python Dirichlet-smoothed belief
scorer) and `structural/bql` (Python index-free Boolean executor) are the REFERENCE
engines; `structural/lucene` (`LuceneStructuredEngine`) is a real-Lucene alternative that
compiles the SAME two query languages (indri-QL, BQL) to Lucene queries and scores them
with `LMDirichletSimilarity`/`BM25Similarity`.

`structured_backend()` / `build_indri_engine` / `build_bql_engine` are the ONE place that
resolves env `STRUCTURED_BACKEND` (python|lucene, default python) to a built engine object
-- mirrors `agent_search.retrievers.lexical.build_bm25_engine`'s pattern exactly (same env-
knob shape, same "unknown value raises loud" contract, same "production callers always pass
a real index_root/key; a `None` key is the ad-hoc/test in-memory fallback" convention).
Shared by:

  - `agent_search.agent.retriever.AgentRetriever.index()` (the real per-episode construction
    site for the indri/indrivisit/indrisnip arms and the BQL-family arms: doc/docv2/docsnip/
    bqlvisit -- supplies a real `index_root`/`key`/`rebuild` so a `lucene` backend opens the
    SAME on-disk index `evaluation/build_indexes.py --retriever search_lucene`
    pre-builds under `indexes/lucene_structured/<key>/`).
  - Every doc-arm workspace's `executor=None` fallback default (`agent_search.agent.tools.
    doc_indri.IndriFetchWorkspace`, `agent_search.agent.tools.doc_research.DocSearchFetch`)
    -- so a workspace built WITHOUT an explicit `executor` (tests, ad-hoc scripts) still
    respects the knob instead of silently hardcoding the Python engine.

Both python-backed and lucene-backed engines expose the IDENTICAL duck-typed surface each
workspace actually calls (see `agent_search.retrievers.structural.lucene.adapters`'s module
docstring for the exact methods and why they're shaped this way), so swapping the backend
never touches a caller's listing/best_line rendering -- only which engine answers a query.

`STRUCTURED_BACKEND=lucene` REQUIRES a real `key` (the corpus/dataset name, e.g.
`hotpotqa_structured` -- matches `evaluation.datasets`' `corpus_id`, which IS the dataset
name for every shared-document dataset): `LuceneStructuredEngine` opens a NAMED prebuilt
index directory, it does not build one in memory the way the python engines do when given
no `index_root`/`key`. A missing index / missing key raises loud (never silently degrades to
a different backend -- a typo'd env value, or `lucene` requested without a prebuilt index,
must fail the run, not quietly score differently than intended).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit


def structured_backend() -> str:
    """Env `STRUCTURED_BACKEND` (default `python`, case-insensitive). Any other value
    raises (a typo must fail loud, not silently fall back) -- same contract as
    `build_bm25_engine`'s `BM25_BACKEND` resolution."""
    backend = (os.environ.get("STRUCTURED_BACKEND") or "python").strip().lower()
    if backend not in ("python", "lucene"):
        raise ValueError(
            f"unknown STRUCTURED_BACKEND={backend!r} — choose 'python' (default) or 'lucene'.")
    return backend


def _lucene_engine(index_root: str, key: Optional[str]):
    from agent_search.retrievers.structural.lucene.engine import get_engine
    if not key:
        raise ValueError(
            "STRUCTURED_BACKEND=lucene needs a real corpus `key` (the dataset name, e.g. "
            "'hotpotqa_structured') to open a prebuilt indexes/lucene_structured/<key>/ "
            "index -- got key=None. This is the ad-hoc/test in-memory-fallback code path; "
            "production callers (agent_search.agent.retriever.AgentRetriever.index()) always "
            "pass the real corpus key.")
    return get_engine(index_root=index_root, dataset=key)


def build_indri_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                       key: Optional[str] = None, rebuild: bool = False, dense=None):
    """Return the Indri-surface engine `IndriFetchWorkspace`/`IndriVisitWorkspace` (doc_indri.py)
    consume, per `STRUCTURED_BACKEND`:

      python  (default) — `structural.indri.model.load_or_build`: prewarmed-if-cached else an
              in-memory build (unchanged pre-existing behavior).
      lucene  — `LuceneIndriAdapter` wrapping the prebuilt `indexes/lucene_structured/<key>/`
              Lucene index (see `lucene.adapters`'s module docstring for the exact interface
              + the dense-fusion score-normalization design).

    `dense` (a `structural.indri.dense_belief.DenseBelief`, or None) is forwarded either way:
    the python path attaches it via `load_or_build`'s existing `dense=` kwarg (unchanged); the
    lucene path wires it into `LuceneIndriAdapter`'s own belief-combination pass (see there)."""
    backend = structured_backend()
    if backend == "lucene":
        from agent_search.retrievers.structural.lucene.adapters import LuceneIndriAdapter
        return LuceneIndriAdapter(_lucene_engine(index_root, key), dense=dense)
    from agent_search.retrievers.structural.indri.model import load_or_build
    return load_or_build(units, index_root, key, rebuild, dense=dense)


def build_bql_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                     key: Optional[str] = None, rebuild: bool = False, dense=None):
    """Return the BQL-surface executor `DocSearchFetch`/`BqlVisitWorkspace` (doc_research.py)
    consume (via `execute_bql`, bql/executor.py), per `STRUCTURED_BACKEND`:

      python  (default) — `structural.bql.executor.load_or_build`: prewarmed-if-cached else an
              in-memory build (unchanged pre-existing behavior).
      lucene  — `LuceneBqlAdapter` wrapping the prebuilt `indexes/lucene_structured/<key>/`
              Lucene index for the exact-hit path; see `lucene.adapters`'s module docstring
              for the 0-hit soft-fallback/coverage-ranking scope decision.

    `dense` (a `structural.indri.dense_belief.DenseBelief`, or None — BQL_DENSE dense-fused
    ranking, DEFAULT OFF, see `bql/dense_fuse.py`) is forwarded either way: the python path
    attaches it via `load_or_build`'s `dense=` kwarg; the lucene path forwards it to
    `LuceneBqlAdapter`, which attaches it to its lazily-built python fallback executor (the
    exact same one `soft_topk`/`coverage_topk` already delegate to under that backend) so
    dense fusion works identically regardless of `STRUCTURED_BACKEND`."""
    backend = structured_backend()
    if backend == "lucene":
        from agent_search.retrievers.structural.lucene.adapters import LuceneBqlAdapter
        return LuceneBqlAdapter(_lucene_engine(index_root, key), units, dense=dense)
    from agent_search.retrievers.structural.bql.executor import load_or_build
    return load_or_build(units, index_root, key, rebuild, dense=dense)


def build_bql_engine_dense_only(units: Sequence[CodeUnit], index_root: str = "indexes",
                                key: Optional[str] = None, rebuild: bool = False, dense=None):
    """NEW, additive-only DENSE-ONLY sibling of `build_bql_engine` (research_bql_donly_visit /
    research_bql_donly_snip — see `bql/dense_fuse.py`'s "dense-ONLY ordering" section): returns
    the SAME BQL-surface executor interface, per the SAME `STRUCTURED_BACKEND` resolution, except
    the ranking of filter-passing candidates is pure dense rank instead of RRF(bm25, dense):

      python  — `structural.bql.executor.load_or_build_dense_only`: SAME on-disk/in-memory
              load-or-build as `load_or_build`, then a `DenseOnlyStructuralExecutor` class swap.
      lucene  — `LuceneBqlDonlyAdapter` (the dense-only sibling of `LuceneBqlAdapter`) wrapping
              the SAME prebuilt `indexes/lucene_structured/<key>/` Lucene index.

    `build_bql_engine` itself is UNCHANGED (still returns the RRF-fused engine every existing
    BQL_DENSE condition uses) — this is a parallel function, not a modified one, so a caller
    that wants dense-only ranking asks for it explicitly instead of build_bql_engine growing a
    new ranking-mode parameter that could accidentally change an existing condition's default."""
    backend = structured_backend()
    if backend == "lucene":
        from agent_search.retrievers.structural.lucene.adapters import LuceneBqlDonlyAdapter
        return LuceneBqlDonlyAdapter(_lucene_engine(index_root, key), units, dense=dense)
    from agent_search.retrievers.structural.bql.executor import load_or_build_dense_only
    return load_or_build_dense_only(units, index_root, key, rebuild, dense=dense)
