"""Thin adapters presenting the same duck-typed interface the pure-Python
`IndriExecutor`/`StructuralExecutor` classes expose, backed by `LuceneStructuredEngine`
instead. Callers never see the difference: `agent_search/tools/search_indri/tool.py`
calls `self.iex.search(query, k)` directly, and `agent_search/tools/search_bql/tool.py`
calls `agent_search.retrievers.bql.executor.execute_bql`, which calls
`executor.run_with_count(expr, k)` on whichever executor it was given.
Constructed only via `agent_search.retrievers.backend`'s `STRUCTURED_BACKEND`
resolver, through `agent_search.retrievers.engines.Engines` (the per-corpus registry
every tool shares); a tool never imports this module directly, the same way it never
imports `BM25Pyserini` directly (see `lexical/__init__.py`'s module docstring for the
parallel).

## Indri surface: `LuceneIndriAdapter`

`agent_search/tools/search_indri/tool.py`'s `_search_impl` calls `self.iex.search(query, k)`
and reads `.error`, `.hits` (a `list[(doc_id, score)]`, unpacked `for rank, (doc_id, _score) in
enumerate(res.hits, ...)`), and `.diagnostics` (`list[(str, float)]`, read only via
`min(res.diagnostics, key=...)` and only when non-empty). `LuceneIndriAdapter.search` returns
the same `indri.model.IndriResult` dataclass the Python engine returns, populated from
`LuceneStructuredEngine.search_indri`.

**Deviation 1 (diagnostics):** the Python engine's `#dense` diagnostics entry is
reproduced (see Deviation 2 below); the per-`#combine`-child weakest-constraint
breakdown (`IndriExecutor._search_expr`'s `for child in self._root_children(root):
diagnostics.append((render(child), self._belief(child, ...)))`) is not: that walk
re-scores each child's own log-belief for the top hit, a Python-engine-internal
computation (`_belief`) with no single-Lucene-query equivalent short of one
sub-search per `#combine` child per top hit. `search_indri/tool.py`'s `_search_impl`
already guards `if res.diagnostics:` before rendering the "weakest constraint" line,
so an empty list here is a graceful degradation (no diagnostics line under this
backend), not an error.

**Deviation 2 (dense-belief fusion, `INDRI_DENSE=1`): the score-normalization decision.**
The Python engine's `_combine_dense` (`indri/model.py`) combines `(1-w)*lex_log_belief +
w*log(dense_norm)`, where `lex_log_belief` is used raw (unnormalized) because
`IndriExecutor`'s Dirichlet `_belief` is a genuine, calibrated log-probability: only the
dense cosine side needs pool-relative min-max normalization to make it commensurable.
Lucene's `LMDirichletSimilarity` score is not that same calibrated log-probability:
Lucene's internal LM-Dirichlet formula folds in its own normalization and boost terms
and is only guaranteed monotonic in relevance, not on the same additive log-probability
scale as the Python reference's `_dirichlet()` (see `indri_compiler.py`'s own module
docstring on this backend being a graded-ranking equivalent, not a byte-identical one).
Treating a raw Lucene score as directly additive with `w*log(dense_norm)` would
therefore combine two incommensurable scales. So `LuceneIndriAdapter._combine_dense`
normalizes both sides symmetrically: pool-relative min-max to `[DENSE_NORM_EPS, 1.0]`,
then `(1-w)*log(lex_norm) + w*log(dense_norm)`, the same additive-in-log-space
combination shape `_combine_dense` uses, applied evenly instead of asymmetrically. At
`w<=0` the original Lucene ranking and scores pass through unchanged (matching
`_combine_dense`'s own `if w <= 0: return scored` short-circuit), and a `#dense`
diagnostics entry is still computed and attached to the top hit either way (contribution
visible without affecting ranking), matching the Python engine's own behavior.

**Deviation 3 (dense pool expansion, recall half): narrower than the Python engine.**
The Python engine's dense side has two independent effects (see `indri/model.py`'s
Deviations): pool expansion (a paraphrased document with zero lexical term overlap
still enters scoring, because Dirichlet smoothing gives every candidate some finite
belief) and belief combination (reranking). Lucene's `BooleanQuery(SHOULD)`
structurally cannot score a document with zero term overlap with any query clause:
such a document is not a "hit" at all, and there is no cheap way to force one
without reimplementing Dirichlet smoothing as a Lucene scorer. So this adapter
reproduces belief combination (rerank Lucene's own returned pool by blending in
dense similarity) but not pool expansion: `search(query, k)` asks Lucene for
`max(k, INDRI_DENSE_EXPAND_K)` hits (so the combine step has a same-sized pool to
rerank within, mirroring the Python engine's `_dense_expand_k` sizing) and then
reranks exactly that set; it never surfaces a document Lucene's own query would not
have returned. This is a real, bounded capability gap of the `lucene` backend, not a
bug.

## BQL surface: `LuceneBqlAdapter`

`execute_bql` (bql/executor.py, called from `agent_search/tools/search_bql/tool.py`)
parses and typechecks a BQL string itself, then calls
`executor.run_with_count(expr, k) -> (ranked, n_hits)` since this executor has that
method. `LuceneBqlAdapter.run_with_count` compiles the already-parsed `Expr` straight
to a Lucene `BooleanQuery` (`LuceneStructuredEngine.search_bql_expr`: a boolean filter
clause for the exact match set plus a BM25-scored should clause for ranking, matching
`bm25_pyserini`'s k1=0.9/b=0.4) and reads the exact total match count off
`LuceneStructuredEngine.count_bql_expr` (`IndexSearcher.count`, not a truncated
`len(hits)`), so the search listing's "(N matches, top K)" header stays accurate under
this backend.

**Deviation (0-hit fallback and coverage ranking): documented, bounded scope.**
`soft_topk` (the 0-exact-hit graceful-degradation BM25-over-terms fallback that
`agent_search/tools/search_bql/tool.py` calls) and `coverage_topk` (BQL's
constraint-coverage ranking, rendered by that same tool's `_coverage_render`, forced
on by the `sieve_v2` strategy's `coverage=True` option) are not reimplemented
against Lucene here: `coverage_topk` needs a per-(document, AND-child) boolean match
mask (`StructuralExecutor._eval` walking each child against a candidate), and there
is no single Lucene query that returns "which of these N clauses did this document
satisfy" per document. It would need one boolean sub-query per child per candidate,
defeating the point of using Lucene at all for what is already the rare, degenerate
code path (a 0-exact-hit AND query). Instead, `LuceneBqlAdapter` lazily builds (on the
first call to either method, not at construction) an in-memory `StructuralExecutor`
over the same units and delegates to it, so the common, non-degenerate case (an exact
or near-exact query that returns hits) runs genuinely on Lucene, and only the rare
0-hit fallback pays the Python engine's build cost, amortized across the whole run:
the adapter, like the engine it wraps, is built once per corpus and reused for every
query in every episode sharing that corpus.
"""
from __future__ import annotations

import math
import threading
from typing import Optional, Sequence

from agent_search.retrievers.indri.model import IndriResult
from agent_search.retrievers.lucene.engine import LuceneStructuredEngine


class LuceneIndriAdapter:
    """`IndriExecutor`-shaped: `.search(query, k) -> IndriResult`. See module docstring."""

    def __init__(self, engine: LuceneStructuredEngine, dense=None):
        self._engine = engine
        self.dense = dense                        # an agent_search.retrievers.dense.belief.DenseBelief, or None

    def search(self, query: str, k: int = 5) -> IndriResult:
        query = query or ""
        from agent_search.retrievers.indri.model import _dense_expand_k
        pool_k = max(k, _dense_expand_k()) if self.dense is not None else k
        r = self._engine.search_indri(query, k=pool_k)
        if r.error:
            return IndriResult(hits=[], error=r.error, diagnostics=[], warning=r.warning)
        hits = [(h.doc_id, float(h.score)) for h in r.hits]
        diagnostics: list = []
        if self.dense is not None and query.strip() and hits:
            hits, dense_log_by_id = self._combine_dense(hits, query)
            top_id = hits[0][0]
            dl = dense_log_by_id.get(top_id)
            if dl is not None:
                diagnostics.append(("#dense", dl))
        return IndriResult(hits=hits[:k], error=None, diagnostics=diagnostics, warning=r.warning)

    def _combine_dense(self, hits: list, query: str) -> tuple:
        """See module docstring's "Deviation 2" for the normalization rationale: both the
        Lucene score and the dense cosine similarity are pool-relative min-max normalized to
        `[DENSE_NORM_EPS, 1.0]` before an additive-in-log-space combine (unlike the Python
        engine, which keeps its own calibrated log-belief raw, since Lucene's LMD score is not
        on that same calibrated scale). Returns `(new_hits, dense_log_by_doc_id)`; `new_hits`
        is unchanged (original Lucene order and scores) at `INDRI_DENSE_W<=0`, matching
        `indri.model.IndriExecutor._combine_dense`'s own short-circuit."""
        from agent_search.retrievers.indri.model import DENSE_NORM_EPS, _dense_weight
        doc_ids = [d for d, _ in hits]
        try:
            sims = self.dense.score(query, doc_ids)
        except Exception:                          # never break the agent loop, see indri/model.py
            return hits, {}
        if not sims:
            return hits, {}
        dvals = list(sims.values())
        lo_d, hi_d = min(dvals), max(dvals)
        rng_d = hi_d - lo_d
        dense_log_by_id: dict = {}
        for doc_id, _lex in hits:
            sim = sims.get(doc_id)
            if sim is None:                        # not in the dense cache -> pool minimum, never fabricated
                dense_norm = DENSE_NORM_EPS
            else:
                dense_norm = (1.0 if rng_d <= 0 else
                             DENSE_NORM_EPS + (1.0 - DENSE_NORM_EPS) * (sim - lo_d) / rng_d)
            dense_log_by_id[doc_id] = math.log(dense_norm)
        w = _dense_weight()
        if w <= 0:
            return hits, dense_log_by_id
        lex_vals = [s for _, s in hits]
        lo_l, hi_l = min(lex_vals), max(lex_vals)
        rng_l = hi_l - lo_l
        scored = []
        for doc_id, lex in hits:
            lex_norm = (1.0 if rng_l <= 0 else
                       DENSE_NORM_EPS + (1.0 - DENSE_NORM_EPS) * (lex - lo_l) / rng_l)
            combined = (1.0 - w) * math.log(lex_norm) + w * dense_log_by_id[doc_id]
            scored.append((doc_id, combined))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored, dense_log_by_id


class LuceneBqlAdapter:
    """`StructuralExecutor`-shaped for `execute_bql`: `.run_with_count(expr, k) ->
    (ranked, n_hits)`, plus `.soft_topk(terms, k)` / `.coverage_topk(expr, k)` for
    `search_bql`'s 0-exact-hit fallback paths (see `agent_search/tools/search_bql/tool.py`).
    See module docstring.

    `dense` (BQL_DENSE dense-fused ranking, default off, see `bql/dense_fuse.py`) is
    forwarded to the lazily built Python `_fallback()` executor, so `soft_topk`'s and
    `coverage_topk`'s dense fusion work identically to the Python backend, since they
    already delegate there (see module docstring's BQL-surface Deviation). It is also
    applied directly to `run_with_count`'s own Lucene-native top-k hits, rerunning the
    same single-tier RRF fuse the Python `StructuralExecutor.run_with_count` uses,
    restricted to the doc_ids Lucene's own boolean filter clause already returned:
    filter semantics are therefore preserved identically to the Python path (a
    non-matching document can never enter `r.hits` in the first place, dense fusion
    or not)."""

    def __init__(self, engine: LuceneStructuredEngine, units: Sequence, dense=None):
        self._engine = engine
        self._units = units if getattr(units, "lazy", False) else list(units)
        self._fallback_ex = None
        self._fallback_lock = threading.Lock()
        self.dense = dense


    def _units_for_fallback(self):
        """The in-memory executor behind the 0-hit fallback needs every unit; refuse an on-disk
        corpus instead of loading it."""
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(self._units, "the BQL fallback ranker", "a prebuilt fallback pool (not available for on-disk corpora yet)")
        return self._units

    def _fallback(self):
        """Lazily build (once, thread-safe) an in-memory `StructuralExecutor` over the same
        units, for the 0-hit soft-fallback and coverage-ranking paths only (see module
        docstring's BQL-surface Deviation). Never built if a corpus's queries always hit
        something under this backend."""
        if self._fallback_ex is None:
            with self._fallback_lock:
                if self._fallback_ex is None:
                    from agent_search.retrievers.bql.executor import StructuralExecutor
                    self._fallback_ex = StructuralExecutor(self._units_for_fallback()).prewarm().attach_dense(self.dense)
        return self._fallback_ex

    def run_with_count(self, expr, k: int = 100) -> tuple:
        r = self._engine.search_bql_expr(expr, k)
        if r.error:
            # execute_bql already parsed and typechecked `expr` with the same Python bql
            # parser and type-checker before calling us, so the Lucene compiler rejecting it
            # here would mean the two compilers disagree on what's a valid AST. Fail loud
            # rather than silently returning an empty ranking.
            raise RuntimeError(f"lucene bql compile/execution error: {r.error}")
        ranked = [(h.doc_id, float(h.score)) for h in r.hits]
        if self.dense is not None and ranked:
            from agent_search.retrievers.bql.executor import _rank_leaves
            from agent_search.retrievers.bql.dense_fuse import fuse_ranked
            leaves = _rank_leaves(expr)
            ranked = fuse_ranked(self.dense, " ".join(leaves), ranked)
        n_hits = self._engine.count_bql_expr(expr)
        return ranked, n_hits

    def soft_topk(self, terms, k: int = 5) -> list:
        return self._fallback().soft_topk(terms, k=k)

    def coverage_topk(self, expr, k: int = 5) -> list:
        return self._fallback().coverage_topk(expr, k=k)


class LuceneBqlDonlyAdapter(LuceneBqlAdapter):
    """The dense-only ranking sibling of `LuceneBqlAdapter`, used for the dense-only sieve
    strategies (`sieve_dense`, `sieve_visit_dense` in `agent_search/strategies/sieve.py`;
    `SearchBql(ranking="dense")`) under `STRUCTURED_BACKEND=lucene`. Identical to
    `LuceneBqlAdapter`: same Lucene boolean filter clause, same exact-match count, same
    0-hit fallback structure. The only difference is that ranking of filter-passing
    candidates uses `fuse_ranked_dense_only`/`DenseOnlyStructuralExecutor` (pure dense
    rank) instead of `fuse_ranked`/`StructuralExecutor` (RRF). This is a separate
    subclass because `LuceneBqlAdapter.run_with_count`/`_fallback` hardcode the RRF
    fuse functions and class, which the RRF-ranked BQL conditions still need. See
    `agent_search/retrievers/bql/dense_fuse.py`'s "dense-only ordering" section and
    `bql/executor.py`'s `DenseOnlyStructuralExecutor` (the Python-backend twin of this
    class)."""

    def run_with_count(self, expr, k: int = 100) -> tuple:
        r = self._engine.search_bql_expr(expr, k)
        if r.error:
            raise RuntimeError(f"lucene bql compile/execution error: {r.error}")
        ranked = [(h.doc_id, float(h.score)) for h in r.hits]
        if self.dense is not None and ranked:
            from agent_search.retrievers.bql.executor import _rank_leaves
            from agent_search.retrievers.bql.dense_fuse import fuse_ranked_dense_only
            leaves = _rank_leaves(expr)
            # Dense-only ranking (see class docstring): the candidate set `ranked` already
            # holds is the same Lucene-filtered set; only the order changes (pure dense
            # rank instead of RRF).
            ranked = fuse_ranked_dense_only(self.dense, " ".join(leaves), ranked)
        n_hits = self._engine.count_bql_expr(expr)
        return ranked, n_hits

    def _fallback(self):
        """Dense-only sibling of `LuceneBqlAdapter._fallback`: lazily builds a
        `DenseOnlyStructuralExecutor` (not the plain `StructuralExecutor`) over the same units
        for the 0-hit soft-fallback and coverage-ranking paths, so `coverage_topk`'s within-tier
        ordering is dense-only here too (`fuse_coverage_tiers_dense_only`), matching
        `run_with_count`'s ranking axis. `StructuralExecutor.prewarm().attach_dense(...)` is
        reused verbatim (a pure `__class__` swap on the constructed instance), the same
        approach `bql/executor.py`'s `load_or_build_dense_only` uses."""
        if self._fallback_ex is None:
            with self._fallback_lock:
                if self._fallback_ex is None:
                    from agent_search.retrievers.bql.executor import (
                        DenseOnlyStructuralExecutor, StructuralExecutor)
                    ex = StructuralExecutor(self._units_for_fallback()).prewarm().attach_dense(self.dense)
                    ex.__class__ = DenseOnlyStructuralExecutor
                    self._fallback_ex = ex
        return self._fallback_ex
