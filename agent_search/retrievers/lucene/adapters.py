"""The document engines behind the Indri and BQL tools, on `LuceneStructuredEngine`.

`LuceneIndriAdapter.search(query, k)` is what `agent_search/tools/search_indri/tool.py` calls;
`LuceneBqlAdapter.run_with_count(expr, k)`, `soft_topk` and `coverage_topk` are what
`agent_search/tools/search_bql/tool.py` calls (through `bql.executor.execute_bql`). The code
repository executor (`bql/executor.py`, `StructuralExecutor`) exposes the same three BQL
methods, so a tool never knows which engine answered. Adapters are constructed by
`agent_search.retrievers.backend` through `agent_search.retrievers.engines.Engines`; a tool
never imports this module.

The scoring notes below compare this backend with the paper's Python reference engines, which
were the original implementation and are no longer in the library. They record the decisions
that make Lucene a graded-ranking equivalent rather than a byte-identical one.

## Indri surface: `LuceneIndriAdapter`

`agent_search/tools/search_indri/tool.py`'s `_search_impl` calls `self.iex.search(query, k)`
and reads `.error`, `.hits` (a `list[(doc_id, score)]`, unpacked `for rank, (doc_id, _score) in
enumerate(res.hits, ...)`), and `.diagnostics` (`list[(str, float)]`, read only via
`min(res.diagnostics, key=...)` and only when non-empty). `LuceneIndriAdapter.search` returns
the `indri.result.IndriResult` dataclass, populated from
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

**Zero-hit fallback and coverage ranking run on Lucene too.** `soft_topk` (the fallback
`agent_search/tools/search_bql/tool.py` calls when an exact query matches nothing) ranks the
index by the query's positive terms alone (`LuceneStructuredEngine.rank_by_terms`, the same
BM25 scoring clause the exact path orders its hits with). `coverage_topk` (constraint-coverage
ranking for a zero-hit AND, on under the `sieve_v2` strategy) takes the top
`_COVERAGE_POOL_CAP` documents matching at least one positive child, then asks Lucene once per
AND child which of those documents satisfy it (`LuceneStructuredEngine.match_ids`), and ranks
by (children matched, BM25 score, id), the same order as the code executor's own
`coverage_topk`. No Python scorer is built over a document corpus anywhere on this path.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence

from agent_search.retrievers.indri.result import IndriResult
from agent_search.retrievers.lucene.engine import LuceneStructuredEngine

# The dense-belief blend for Indri (`INDRI_DENSE=1`), read live so a test can override the
# environment without a reload. `DENSE_NORM_EPS` is the floor of the min-max normalisation:
# the worst document in a pool keeps a finite log-belief.
DENSE_NORM_EPS = 1e-6


def indri_dense_weight() -> float:
    """`INDRI_DENSE_W`: the dense belief's weight in the combined score."""
    return float(os.environ.get("INDRI_DENSE_W", "0.35"))


def indri_dense_expand_k() -> int:
    """`INDRI_DENSE_EXPAND_K`: how many hits Lucene returns for the dense rerank pool."""
    return int(os.environ.get("INDRI_DENSE_EXPAND_K", "50"))


class LuceneIndriAdapter:
    """`IndriExecutor`-shaped: `.search(query, k) -> IndriResult`. See module docstring."""

    def __init__(self, engine: LuceneStructuredEngine, dense=None):
        self._engine = engine
        self.dense = dense                        # an agent_search.retrievers.dense.belief.DenseBelief, or None

    def search(self, query: str, k: int = 5) -> IndriResult:
        query = query or ""
        pool_k = max(k, indri_dense_expand_k()) if self.dense is not None else k
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
        w = indri_dense_weight()
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
    `search_bql`'s zero-hit fallback paths (see `agent_search/tools/search_bql/tool.py`).
    See module docstring.

    `dense` (BQL_DENSE dense-fused ranking, default off, see `bql/dense_fuse.py`) reranks
    the exact hits, the fallback pool and each coverage tier with the same RRF fuse the code
    executor uses, restricted to the ids Lucene already returned: filter semantics are
    preserved (a non-matching document can never enter a ranking, dense fusion or not)."""

    def __init__(self, engine: LuceneStructuredEngine, dense=None):
        self._engine = engine
        self.dense = dense

    # --- the fusion hooks the dense-only sibling overrides ---------------------------
    def _fuse_ranked(self, terms, ranked: list) -> list:
        from agent_search.retrievers.bql.dense_fuse import fuse_ranked
        return fuse_ranked(self.dense, " ".join(terms), ranked)

    def _fuse_tiers(self, terms, rows: list) -> list:
        from agent_search.retrievers.bql.dense_fuse import fuse_coverage_tiers
        return fuse_coverage_tiers(self.dense, " ".join(terms), rows)

    # --- the exact path -----------------------------------------------------------------
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
            ranked = self._fuse_ranked(_rank_leaves(expr), ranked)
        n_hits = self._engine.count_bql_expr(expr)
        return ranked, n_hits

    # --- the zero-hit fallbacks ---------------------------------------------------------
    def soft_topk(self, terms, k: int = 5) -> list:
        """Rank by BM25 over `terms` alone: `[(doc_id, score)]` best first, at most `k`. With
        a dense model attached the pool is the union of the lexically closest `BQL_SOFT_POOL`
        documents and the dense side's nearest neighbours (read from the persisted cache,
        never encoded online), ordered by the fusion rule."""
        from agent_search.retrievers.bql.ast import Or, Term
        from agent_search.retrievers.bql.executor import _SOFT_POOL
        terms = [t for t in terms if t]
        if not terms:
            return []
        pool_n = max(k, _SOFT_POOL)
        expr = Or(tuple(Term(t) for t in terms)) if len(terms) > 1 else Term(terms[0])
        r = self._engine.rank_by_terms(expr, pool_n)
        if r.error:
            raise RuntimeError(f"lucene bql fallback error: {r.error}")
        pool = [(h.doc_id, float(h.score)) for h in r.hits if h.score > 0]
        if self.dense is not None:
            try:
                dense_top = list(self.dense.top_k_doc_ids(" ".join(terms), k=pool_n) or [])
            except Exception:  # noqa: BLE001, a dense-side failure degrades to the BM25 pool
                dense_top = []
            have = {d for d, _ in pool}
            pool += [(d, 0.0) for d in dense_top if d not in have]
            if pool:
                pool = self._fuse_ranked(terms, pool)
        return pool[:k]

    def coverage_topk(self, expr, k: int = 5) -> list:
        """Constraint-coverage ranking for a zero-hit AND: `[(doc_id, matched_mask, n_matched,
        score)]`, mask aligned with the AND's children, ordered by children matched, then BM25
        over the query's positive terms, then id. A non-AND query degrades to `soft_topk`
        with a one-element mask, the same contract as the code executor."""
        from agent_search.retrievers.bql.ast import And, Not, Or
        from agent_search.retrievers.bql.executor import _COVERAGE_POOL_CAP, _rank_leaves
        terms = _rank_leaves(expr)
        if not isinstance(expr, And):
            return [(d, (True,), 1, s) for d, s in self.soft_topk(terms, k=k)]
        children = list(expr.children)
        positive = [c for c in children if not isinstance(c, Not)]
        # The pool: documents matching at least one positive child, best BM25 first. A NOT
        # child can be satisfied by any document, so with one present the pool is every
        # document carrying at least one query term.
        if positive and len(positive) == len(children):
            r = self._engine.search_bql_expr(Or(tuple(positive)) if len(positive) > 1 else positive[0],
                                             _COVERAGE_POOL_CAP)
        else:
            r = self._engine.rank_by_terms(expr, _COVERAGE_POOL_CAP)
        if r.error:
            raise RuntimeError(f"lucene bql coverage error: {r.error}")
        pool = [(h.doc_id, float(h.score)) for h in r.hits]
        if not pool:
            return []
        ids = [d for d, _ in pool]
        matched = [self._engine.match_ids(c, ids) for c in children]
        rows = []
        for doc_id, score in pool:
            mask = tuple(doc_id in m for m in matched)
            n = sum(mask)
            if n:
                rows.append((doc_id, mask, n, score))
        rows.sort(key=lambda r: (-r[2], -r[3], r[0]))
        if self.dense is not None and rows:
            rows = self._fuse_tiers(terms, rows)
        return rows[:k]


class LuceneBqlDonlyAdapter(LuceneBqlAdapter):
    """The dense-only ranking sibling of `LuceneBqlAdapter`, used by the dense-only sieve
    strategies (`sieve_dense`, `sieve_visit_dense` in `agent_search/strategies/sieve.py`;
    `SearchBql(ranking="dense")`). Same Lucene boolean filter, same exact-match count, same
    fallback structure; only the order of filter-passing candidates changes, from
    RRF(bm25, dense) to the dense rank alone (`bql/dense_fuse.py`, "dense-only ordering")."""

    def _fuse_ranked(self, terms, ranked: list) -> list:
        from agent_search.retrievers.bql.dense_fuse import fuse_ranked_dense_only
        return fuse_ranked_dense_only(self.dense, " ".join(terms), ranked)

    def _fuse_tiers(self, terms, rows: list) -> list:
        from agent_search.retrievers.bql.dense_fuse import fuse_coverage_tiers_dense_only
        return fuse_coverage_tiers_dense_only(self.dense, " ".join(terms), rows)
