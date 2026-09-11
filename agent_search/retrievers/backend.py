"""Structural retrieval backends: `indri/` (the Python Dirichlet-smoothed belief scorer) and
`bql/` (the Python index-free Boolean executor) are the reference engines; `lucene/`
(`LuceneStructuredEngine`) is a real-Lucene alternative that compiles both query languages
(indri-QL, BQL) to Lucene queries and scores them with `LMDirichletSimilarity`/`BM25Similarity`.

`structured_backend()`, `build_indri_engine`, `build_bql_engine` and `build_bql_engine_dense_only`
resolve env `STRUCTURED_BACKEND` (python|lucene, default python) to a built engine object, with
the same env-knob shape and "unknown value raises loud" contract as
`agent_search.retrievers.lexical.build_bm25_engine`: production callers always pass a real
`index_root`/`key`; a `None` key is the ad-hoc/test in-memory fallback.

`agent_search.retrievers.engines.Engines` is the per-corpus registry every tool shares; its
`bql`, `bql_fused`, `bql_dense_only`, `bql_plain` and `indri` methods call into this module once
per corpus, under a lock, and cache the result.

Both python-backed and lucene-backed engines expose the identical duck-typed surface their
callers use (see `agent_search.retrievers.lucene.adapters`'s module docstring for the exact
methods and why they're shaped this way), so swapping the backend never touches a caller's
listing/best_line rendering, only which engine answers a query.

`STRUCTURED_BACKEND=lucene` requires a real `key` (the corpus/dataset name, e.g.
`hotpotqa_structured`, matching `agent_search.evaluation.datasets`' `corpus_id`, the dataset
name for every shared-document dataset): `LuceneStructuredEngine` opens a named prebuilt
index directory, it does not build one in memory the way the python engines do when given
no `index_root`/`key`. A missing index or missing key raises immediately: a typo'd env value,
or `lucene` requested without a prebuilt index, must fail the run rather than quietly score
with a different backend.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit


def structured_backend() -> str:
    """Env `STRUCTURED_BACKEND` (default `python`, case-insensitive). Any other value
    raises: a typo must fail loud, not silently fall back. Same contract as
    `build_bm25_engine`'s `BM25_BACKEND` resolution."""
    backend = (os.environ.get("STRUCTURED_BACKEND") or "python").strip().lower()
    if backend not in ("python", "lucene"):
        raise ValueError(
            f"unknown STRUCTURED_BACKEND={backend!r} — choose 'python' (default) or 'lucene'.")
    return backend


def _lucene_engine(index_root: str, key: Optional[str]):
    from agent_search.retrievers.lucene.engine import get_engine
    if not key:
        raise ValueError(
            "STRUCTURED_BACKEND=lucene needs a real corpus `key` (the dataset name, e.g. "
            "'hotpotqa_structured') to open a prebuilt indexes/lucene_structured/<key>/ "
            "index -- got key=None. This is the ad-hoc/test in-memory-fallback code path; "
            "production callers (agent_search.evaluation.agent_runner.ConditionAgent.index()) "
            "always pass the real corpus key.")
    eng = get_engine(index_root=index_root, dataset=key)
    # Open (and validate) the index now, at construction time, instead of lazily on the
    # first search: otherwise a missing prebuilt index turns every search call into an
    # "ERROR: ..." tool observation partway through an episode instead of stopping the run
    # before it starts. `_ensure_open` is idempotent and cheap on a second call (it returns
    # immediately once open), so calling it here costs nothing extra for `build_indri_engine`,
    # `build_bql_engine` and `build_bql_engine_dense_only`, which all funnel through this
    # function.
    eng._ensure_open()
    return eng


def build_indri_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                       key: Optional[str] = None, rebuild: bool = False, dense=None):
    """Build the Indri executor `agent_search.retrievers.engines.Engines.indri` hands to the
    `search_indri` tool, per `STRUCTURED_BACKEND`:

      python  (default): `agent_search.retrievers.indri.model.load_or_build`, which loads a
              cached index if one exists, otherwise builds one in memory.
      lucene: `LuceneIndriAdapter` wraps the prebuilt `indexes/lucene_structured/<key>/`
              Lucene index (see `lucene.adapters`'s module docstring for the exact interface
              and the dense-fusion score-normalization design).

    `dense` (an `agent_search.retrievers.dense.belief.DenseBelief`, or None) is forwarded
    either way: the python path attaches it via `load_or_build`'s `dense=` kwarg; the lucene
    path wires it into `LuceneIndriAdapter`'s own belief-combination pass (see there)."""
    backend = structured_backend()
    if backend == "lucene":
        from agent_search.retrievers.lucene.adapters import LuceneIndriAdapter
        return LuceneIndriAdapter(_lucene_engine(index_root, key), dense=dense)
    from agent_search.retrievers.indri.model import load_or_build
    return load_or_build(units, index_root, key, rebuild, dense=dense)


def build_bql_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                     key: Optional[str] = None, rebuild: bool = False, dense=None):
    """Build the BQL executor `agent_search.retrievers.engines.Engines` hands to the
    `search_bql` tool (via `execute_bql`, `bql/executor.py`), per `STRUCTURED_BACKEND`:

      python  (default): `agent_search.retrievers.bql.executor.load_or_build`, which loads a
              cached index if one exists, otherwise builds one in memory.
      lucene: `LuceneBqlAdapter` wraps the prebuilt `indexes/lucene_structured/<key>/`
              Lucene index for the exact-hit path; see `lucene.adapters`'s module docstring
              for the 0-hit soft-fallback/coverage-ranking scope decision.

    `dense` (an `agent_search.retrievers.dense.belief.DenseBelief`, or None; dense-fused ranking
    is off unless `BQL_DENSE=1`, see `bql/dense_fuse.py`) is forwarded either way: the python
    path attaches it via `load_or_build`'s `dense=` kwarg; the lucene path forwards it to
    `LuceneBqlAdapter`, which attaches it to its lazily-built python fallback executor, the
    same one `soft_topk`/`coverage_topk` delegate to under that backend, so dense fusion works
    the same regardless of `STRUCTURED_BACKEND`."""
    backend = structured_backend()
    if backend == "lucene":
        from agent_search.retrievers.lucene.adapters import LuceneBqlAdapter
        return LuceneBqlAdapter(_lucene_engine(index_root, key), units, dense=dense)
    from agent_search.retrievers.bql.executor import load_or_build
    return load_or_build(units, index_root, key, rebuild, dense=dense)


def build_bql_engine_dense_only(units: Sequence[CodeUnit], index_root: str = "indexes",
                                key: Optional[str] = None, rebuild: bool = False, dense=None):
    """The dense-only sibling of `build_bql_engine` (the strategies `sieve_dense` and
    `sieve_visit_dense`, aliased as `research_bql_donly_snip`/`research_bql_donly_visit`; see
    `bql/dense_fuse.py`'s "dense-only ordering" section). Same BQL-surface executor interface
    and the same `STRUCTURED_BACKEND` resolution as `build_bql_engine`, except the ranking of
    filter-passing candidates is pure dense rank instead of RRF(bm25, dense):

      python: `agent_search.retrievers.bql.executor.load_or_build_dense_only`, the same
              on-disk/in-memory load-or-build as `load_or_build`, with a
              `DenseOnlyStructuralExecutor` class swap.
      lucene: `LuceneBqlDonlyAdapter` (the dense-only sibling of `LuceneBqlAdapter`)
              wrapping the same prebuilt `indexes/lucene_structured/<key>/` Lucene index.

    `build_bql_engine` still returns the RRF-fused engine; this is a separate function so a
    caller that wants dense-only ranking asks for it explicitly, instead of `build_bql_engine`
    taking a ranking-mode parameter that could change an existing condition's default."""
    backend = structured_backend()
    if backend == "lucene":
        from agent_search.retrievers.lucene.adapters import LuceneBqlDonlyAdapter
        return LuceneBqlDonlyAdapter(_lucene_engine(index_root, key), units, dense=dense)
    from agent_search.retrievers.bql.executor import load_or_build_dense_only
    return load_or_build_dense_only(units, index_root, key, rebuild, dense=dense)
