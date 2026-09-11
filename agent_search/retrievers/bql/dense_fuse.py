"""Dense-fused BQL ranking (`BQL_DENSE`, default off): the answer to "why not BQL with
dense instead of Indri with dense?" BQL-as-filter over a lexical ranker already beats the
SERP baseline on the strength of its QL filter alone; this module fuses dense evidence into
the ranking of whatever the filter already selected, so the filter's precision and the
dense ranker's recall and paraphrase-robustness combine instead of trading off.

## Mechanism

`StructuralExecutor` (bql/executor.py) executes the boolean/field/date filter exactly as
without dense fusion; nothing here changes candidate selection. Only the order of
filter-passing candidates changes, and only when a `DenseBelief` has been attached
(`attach_dense`, the same pattern as `indri/model.py`'s `IndriExecutor.dense`) and the query
text is non-empty:

  fused_rank = RRF_k60( bm25_rank(filter_passing_ids), dense_rank(filter_passing_ids) )

Two rankings, same candidate set, Reciprocal Rank Fusion (`RRF_K`, default 60, the
Cormack/Clarke/Buettcher 2009 constant every other RRF user in this codebase also pins;
see `agent_search.tools.budgets.RRF_K`). `rrf_fuse` here is a separate, local
implementation, not an import of `agent_search.tools.common.rrf_fuse`: this module lives in
the BQL executor's own dependency layer (bql/ -> dense_fuse.py), which must not reach up into
agent_search/tools/; duplicating ~10 lines of arithmetic is cheaper than an inverted import.

## Why dense scoring is restricted to the filter-passing candidate ids

`dense_rank_for_candidates` calls `DenseBelief.score(query_text, doc_ids=candidate_ids)`,
never `DenseBelief.top_k_doc_ids` (global nearest-neighbour search). This is the one
correctness-critical property of the whole mechanism: BQL's boolean/field/date filter is the
retrieval contract this surface exists to guarantee (a `date[1980..1989]` clause, a `NOT`
exclusion, a field scope), so a document that fails that filter must never appear in the
result, no matter how high its embedding similarity. `DenseBelief.score` batches one query
encode against exactly the candidate rows (a single (m, d) matmul; see dense/belief.py's
`_similarities`), so there is no global dense ranking in play that a bug could accidentally
leak a non-candidate back in from, and no separate "intersect with the filter" step whose
removal could silently reintroduce global dense results. Restricting the input makes the
invariant structural instead of a runtime check.

## Coverage tiers

`coverage_topk`'s 0-exact-hit constraint-coverage fallback (BQL v2 Feature 2) ranks docs by
`(matched AND-children DESC, bm25 DESC)`: a doc matching 5/6 constraints always outranks one
matching 2/6, regardless of lexical score. Fusing dense in naively (one global RRF over the
whole coverage_topk output) would let a high dense similarity move a 2/6-coverage doc above a
5/6-coverage one, silently discarding the coverage signal the whole feature exists to
provide. `fuse_coverage_tiers` instead groups the (already `-n_matched`-sorted) rows into
tiers by `n_matched`, and RRF-fuses bm25 order against dense order only within each tier, so
a doc can never be promoted across a coverage boundary by dense similarity; fusion only
resolves ties and reorders within "these docs satisfied the same number of constraints".

## Dense-only ordering (`sieve_dense` / `sieve_visit_dense`)

`fuse_ranked_dense_only`/`fuse_coverage_tiers_dense_only` (bottom of this module) are siblings of
`fuse_ranked`/`fuse_coverage_tiers` above that order filter-passing candidates purely by
`dense_rank_for_candidates`: no RRF, no bm25 signal in the ordering at all (bm25 order is used
only as the graceful-degradation fallback when dense scoring is empty or unavailable). Everything
else, the boolean/field/date filter, the coverage-tier structure, the "restricted to
filter-passing candidates only" invariant, is identical to the RRF path. See
`agent_search.retrievers.bql.executor.DenseOnlyStructuralExecutor`, which calls these
instead of `fuse_ranked`/`fuse_coverage_tiers`."""
from __future__ import annotations

import os
from itertools import groupby
from typing import Optional, Sequence

RRF_K = int(os.environ.get("BQL_DENSE_RRF_K", "60"))


def bql_dense_enabled() -> bool:
    """`BQL_DENSE` (env, default off): whether `Engines.bql()` (`agent_search/retrievers/
    engines.py`) attaches a `DenseBelief` to the shared BQL executor for the general
    `sieve`/`sieve_visit` family of strategies, the same env-knob pattern `INDRI_DENSE` uses.
    The dedicated conditions `sieve_visit_fused` and `sieve` (aliased as
    `research_bql_dense_visit`/`research_bql_dense_snip`) attach a `DenseBelief`
    unconditionally through `Engines.bql_fused()` instead, so this function is not consulted
    for them at all. Recorded in `env_knobs` by `agent_search/evaluation/run_eval.py`,
    alongside `STRUCTURED_BACKEND`/`BM25_BACKEND`/`INDRI_DENSE`."""
    return os.environ.get("BQL_DENSE", "0").strip().lower() in ("1", "true", "yes")


def _rank_map(ordered_ids: Sequence[str]) -> dict:
    """doc_id -> 1-based rank in `ordered_ids` (first = rank 1)."""
    return {d: i + 1 for i, d in enumerate(ordered_ids)}


def rrf_fuse(bm25_ranked_ids: Sequence[str], dense_ranked_ids: Sequence[str],
             k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion of the two rankings of the filter-passing candidates, through
    `agent_search.retrievers.fusion.RRF` (the same method the hybrid engine uses). A doc id
    present in only one input contributes nothing from the other side; ties break by doc id."""
    from agent_search.retrievers.fusion import RRF
    return RRF(k=k).fuse([[(d, 0.0) for d in bm25_ranked_ids], [(d, 0.0) for d in dense_ranked_ids]])


def dense_rank_for_candidates(dense_belief, query_text: str,
                              candidate_ids: Sequence[str]) -> list:
    """Dense-similarity ranking of `candidate_ids` only; see the module docstring's "Why dense
    scoring is restricted..." section. Returns `[]` (never raises) if `dense_belief` is
    None or not ready, `candidate_ids`/`query_text` is empty, or the dense side fails for any
    reason (missing cache, offline, corrupt index). Callers must treat `[]` as "skip fusion,
    keep the plain bm25 order," the same graceful-degradation contract `indri/model.py`'s own
    `_combine_dense`/`_dense_expand` use for a dense-side failure."""
    if dense_belief is None or not candidate_ids or not (query_text or "").strip():
        return []
    try:
        if hasattr(dense_belief, "is_ready") and not dense_belief.is_ready():
            return []
        sims = dense_belief.score(query_text, doc_ids=list(candidate_ids))
    except Exception:      # noqa: BLE001, dense fusion is an ablation-knob extra; never
        return []          # break a search call if the dense side misbehaves
    if not sims:
        return []
    return sorted(sims, key=lambda d: (-sims[d], d))


def fuse_ranked(dense_belief, query_text: str,
                bm25_ranked: Sequence[tuple]) -> list:
    """Fuse a single-tier bm25-ranked `[(doc_id, score), ...]` list (the common exact-match
    path, `run_with_count`: all of `bm25_ranked` passed the filter, so it is one coverage
    tier by construction) with dense similarity restricted to those same doc_ids. Returns a
    re-ordered `[(doc_id, score), ...]` list (original bm25 scores preserved for display;
    only the order changes, matching `LuceneIndriAdapter._combine_dense`'s "scores are
    informational, order is what fusion changes" convention). `[]`/unfusable input, or a
    dense-side miss, returns `bm25_ranked` unchanged: never worse than the pre-fusion
    ranking, never raises."""
    if not bm25_ranked:
        return list(bm25_ranked)
    bm_ids = [d for d, _ in bm25_ranked]
    dense_ids = dense_rank_for_candidates(dense_belief, query_text, bm_ids)
    if not dense_ids:
        return list(bm25_ranked)
    fused_ids = rrf_fuse(bm_ids, dense_ids)
    score_by = dict(bm25_ranked)
    return [(d, score_by.get(d, 0.0)) for d in fused_ids]


def fuse_coverage_tiers(dense_belief, query_text: str, rows: Sequence[tuple]) -> list:
    """Fuse `coverage_topk`'s rows (`[(doc_id, mask, n_matched, score), ...]`, already sorted
    `(-n_matched, -score, doc_id)` by the caller) within each `n_matched` tier only (see the
    module docstring's "Coverage tiers" section): a doc can be reordered against others that
    satisfied the same number of AND-children, never promoted past a doc that satisfied more.
    Preserves tier order (descending `n_matched`) and each row's own `(mask, n_matched, score)`
    payload untouched; only the doc_id order within a tier changes. `[]`/no dense attached
    returns `rows` unchanged."""
    rows = list(rows)
    if not rows or dense_belief is None:
        return rows
    out: list = []
    for _n, group_iter in groupby(rows, key=lambda r: r[2]):
        group = list(group_iter)
        if len(group) < 2:                 # nothing to reorder within a singleton tier
            out.extend(group)
            continue
        bm_ids = [r[0] for r in group]
        dense_ids = dense_rank_for_candidates(dense_belief, query_text, bm_ids)
        if not dense_ids:
            out.extend(group)
            continue
        fused_ids = rrf_fuse(bm_ids, dense_ids)
        by_id = {r[0]: r for r in group}
        out.extend(by_id[d] for d in fused_ids if d in by_id)
    return out


# --- Dense-only ordering (sieve_dense / sieve_visit_dense, `research_bql_donly_snip` /
# --- `research_bql_donly_visit`)
#
# The answer to "what does BQL_DENSE's RRF fusion itself contribute, vs pure dense re-ranking of
# the same filter-passing candidates?" `fuse_ranked`/`fuse_coverage_tiers` above (still consumed
# by sieve_visit_fused/sieve exactly as before) RRF-combine the BM25 rerank with dense
# similarity; `fuse_ranked_dense_only`/`fuse_coverage_tiers_dense_only` instead order
# filter-passing candidates purely by `dense_rank_for_candidates`. The bm25 order is consulted
# only as the graceful-degradation fallback when the dense side is empty or unready, the same
# "never worse than pre-fusion, never raise" contract `fuse_ranked`/`fuse_coverage_tiers` use.
# Candidate selection (the boolean/field/date filter) and, for the coverage path, the tier
# structure (a doc can never cross a coverage-tier boundary) are completely untouched; only
# the order within what the filter already selected changes. See
# `agent_search.retrievers.bql.executor.DenseOnlyStructuralExecutor`, the sibling
# executor that calls these instead of `fuse_ranked`/`fuse_coverage_tiers`.

def fuse_ranked_dense_only(dense_belief, query_text: str,
                           bm25_ranked: Sequence[tuple]) -> list:
    """Dense-only sibling of `fuse_ranked`: reorders a single-tier bm25-ranked
    `[(doc_id, score), ...]` list purely by `dense_rank_for_candidates` restricted to those same
    doc_ids (never a global dense search, the same restriction `dense_rank_for_candidates` itself
    enforces). Original bm25 scores are preserved for display; only the order changes, matching
    `fuse_ranked`'s own convention. `[]`/unfusable input, or a dense-side miss, returns
    `bm25_ranked` unchanged: never worse, never raises.

    `dense_rank_for_candidates` may return fewer ids than `bm_ids` (a candidate absent from
    the dense index/corpus, e.g. an id the dense cache was never built for); those missing
    candidates are appended after the dense-ranked ones, in their incoming bm25 order, so a
    filter-passing candidate is never silently dropped from the result just because the dense
    side has no opinion on it."""
    if not bm25_ranked:
        return list(bm25_ranked)
    bm_ids = [d for d, _ in bm25_ranked]
    dense_ids = dense_rank_for_candidates(dense_belief, query_text, bm_ids)
    if not dense_ids:
        return list(bm25_ranked)
    dense_set = set(dense_ids)
    missing = [d for d in bm_ids if d not in dense_set]   # never dropped -> appended, bm25 order
    score_by = dict(bm25_ranked)
    return [(d, score_by.get(d, 0.0)) for d in (*dense_ids, *missing)]


def fuse_coverage_tiers_dense_only(dense_belief, query_text: str,
                                   rows: Sequence[tuple]) -> list:
    """Dense-only sibling of `fuse_coverage_tiers`: reorders `coverage_topk`'s rows within each
    `n_matched` tier purely by `dense_rank_for_candidates`. The coverage-tier structure
    is preserved exactly as `fuse_coverage_tiers` preserves it (a doc can never cross a tier
    boundary; this isolates the fusion signal, not the tiering). `[]`/no dense attached returns
    `rows` unchanged; a tier whose dense scoring fails or returns nothing keeps its incoming
    (bm25/coverage) order for that tier only, the same graceful-degradation contract as
    `fuse_coverage_tiers`. As in `fuse_ranked_dense_only`, a tier candidate absent from the
    dense side (`dense_rank_for_candidates` returns fewer ids than the tier has) is appended
    after the dense-ranked ones, in its incoming (coverage/bm25) order within that tier,
    never dropped."""
    rows = list(rows)
    if not rows or dense_belief is None:
        return rows
    out: list = []
    for _n, group_iter in groupby(rows, key=lambda r: r[2]):
        group = list(group_iter)
        if len(group) < 2:                 # nothing to reorder within a singleton tier
            out.extend(group)
            continue
        bm_ids = [r[0] for r in group]
        dense_ids = dense_rank_for_candidates(dense_belief, query_text, bm_ids)
        if not dense_ids:
            out.extend(group)
            continue
        by_id = {r[0]: r for r in group}
        dense_set = set(dense_ids)
        missing = [d for d in bm_ids if d not in dense_set]    # never dropped -> appended
        out.extend(by_id[d] for d in (*dense_ids, *missing) if d in by_id)
    return out
