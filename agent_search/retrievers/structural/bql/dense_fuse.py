"""Dense-fused BQL ranking (`BQL_DENSE`, DEFAULT OFF) — the answer to "why not BQL with
dense instead of Indri with dense?": BQL-as-filter over a lexical ranker already beats the
SERP baseline on the strength of its QL filter alone; this module fuses dense evidence into
the RANKING of whatever the filter already selected, so the filter's precision and the
dense ranker's recall/paraphrase-robustness can combine instead of trading off.

## Mechanism

`StructuralExecutor` (bql/executor.py) executes the boolean/field/date FILTER exactly as
before — nothing here changes candidate SELECTION. Only the ORDER of filter-passing
candidates changes, and only when a `DenseBelief` has been attached (`attach_dense`, mirrors
`indri/model.py`'s `IndriExecutor.dense` pattern byte-for-byte) AND the query text is
non-empty:

  fused_rank = RRF_k60( bm25_rank(filter_passing_ids), dense_rank(filter_passing_ids) )

Two rankings, same candidate set, Reciprocal Rank Fusion (`RRF_K`, default 60 — the
Cormack/Clarke/Buettcher 2009 constant every other RRF user in this codebase also pins,
see `agent_search.agent.tools.doc_research.RRF_K`). `rrf_fuse` here is a SEPARATE, local
implementation (not an import of doc_research.rrf_fuse) — this module lives in the BQL
executor's own dependency layer (bql/ -> ranking.py), which must not reach up into
agent/tools/; duplicating ~10 lines of arithmetic is cheaper than an inverted import.

## Why dense scoring is restricted to the filter-passing candidate ids

`dense_rank_for_candidates` calls `DenseBelief.score(query_text, doc_ids=candidate_ids)` —
NEVER `DenseBelief.top_k_doc_ids` (global nearest-neighbour search). This is the one
correctness-critical property of the whole mechanism: BQL's boolean/field/date filter is the
retrieval CONTRACT this surface exists to guarantee (a `date[1980..1989]` clause, a `NOT`
exclusion, a field scope) — a document that fails that filter must NEVER appear in the
result, no matter how high its embedding similarity. `DenseBelief.score` batches ONE query
encode against exactly the candidate rows (a single (m, d) matmul — see dense_belief.py's
`_similarities`), so there is no global dense ranking in play that a bug could accidentally
leak a non-candidate back in from, and no separate "intersect with the filter" step whose
removal (a future refactor) could silently reintroduce global dense results. Restricting the
INPUT makes the invariant structural instead of a runtime check.

## Coverage tiers

`coverage_topk`'s 0-exact-hit constraint-COVERAGE fallback (BQL v2 Feature 2) ranks docs by
`(matched AND-children DESC, bm25 DESC)` — a doc matching 5/6 constraints always outranks one
matching 2/6, regardless of lexical score. Fusing dense in NAIVELY (one global RRF over the
whole coverage_topk output) would let a high dense similarity move a 2/6-coverage doc above a
5/6-coverage one — silently discarding the coverage signal the whole feature exists to
provide. `fuse_coverage_tiers` instead groups the (already `-n_matched`-sorted) rows into
tiers by `n_matched`, and RRF-fuses bm25 order against dense order ONLY WITHIN each tier —
so a doc can never be promoted across a coverage boundary by dense similarity; fusion only
resolves ties/reorders WITHIN "these docs satisfied the same number of constraints".

## NEW, additive-only: dense-ONLY ordering (research_bql_donly_visit / research_bql_donly_snip)

`fuse_ranked_dense_only`/`fuse_coverage_tiers_dense_only` (bottom of this module) are siblings of
`fuse_ranked`/`fuse_coverage_tiers` above that order filter-passing candidates PURELY by
`dense_rank_for_candidates` — no RRF, no bm25 signal in the ordering at all (bm25 order is used
only as the graceful-degradation fallback when dense scoring is empty/unavailable). Everything
else — the boolean/field/date FILTER, the coverage-tier PRIMARY structure, the "restricted to
filter-passing candidates only" invariant — is identical to the RRF path. See
`agent_search.retrievers.structural.bql.executor.DenseOnlyStructuralExecutor`, which calls these
instead of `fuse_ranked`/`fuse_coverage_tiers`."""
from __future__ import annotations

import os
from itertools import groupby
from typing import Optional, Sequence

RRF_K = int(os.environ.get("BQL_DENSE_RRF_K", "60"))


def bql_dense_enabled() -> bool:
    """`BQL_DENSE` (env, default OFF) — whether `AgentRetriever.index()` attaches a
    `DenseBelief` to the shared BQL/doc-vN executor for the general search->fetch arms
    (doc/docv2/docsnip/bqlvisit), mirroring `INDRI_DENSE`'s retrofit-via-env-knob pattern
    exactly. The two DEDICATED conditions (`research_bql_dense_visit`,
    `research_bql_dense_snip`) attach a `DenseBelief` unconditionally in their own arm branch
    (like `research_dense`/`research_dense_fetch` never need an env toggle to use dense
    retrieval) — for them this function is not consulted at all. Recorded in `env_knobs` by
    `evaluation/run_eval.py`, alongside `STRUCTURED_BACKEND`/`BM25_BACKEND`/`INDRI_DENSE`."""
    return os.environ.get("BQL_DENSE", "0").strip().lower() in ("1", "true", "yes")


def _rank_map(ordered_ids: Sequence[str]) -> dict:
    """doc_id -> 1-based rank in `ordered_ids` (first = rank 1)."""
    return {d: i + 1 for i, d in enumerate(ordered_ids)}


def rrf_fuse(bm25_ranked_ids: Sequence[str], dense_ranked_ids: Sequence[str],
             k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion of two doc_id rankings over (nominally) the SAME candidate set.

    `fused_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))` (1-based ranks); a doc_id
    present in only one input contributes 0 from the other side rather than being dropped —
    defensive (the two inputs are always built from the same candidate set here, but a
    caller that ever passes mismatched pools degrades gracefully instead of losing docs).
    Deterministic tie-break: doc_id ascending — matches every other ranker in this codebase
    (`FlatIndex.search`, `BM25.score_subset`)."""
    bm_rank = _rank_map(bm25_ranked_ids)
    dn_rank = _rank_map(dense_ranked_ids)
    ids = set(bm_rank) | set(dn_rank)
    scored = []
    for d in ids:
        s = 0.0
        if d in bm_rank:
            s += 1.0 / (k + bm_rank[d])
        if d in dn_rank:
            s += 1.0 / (k + dn_rank[d])
        scored.append((d, s))
    scored.sort(key=lambda p: (-p[1], p[0]))
    return [d for d, _ in scored]


def dense_rank_for_candidates(dense_belief, query_text: str,
                              candidate_ids: Sequence[str]) -> list:
    """Dense-similarity ranking of `candidate_ids` ONLY — see module docstring's "Why dense
    scoring is restricted..." section. Returns `[]` (never raises) if `dense_belief` is
    None/not ready, `candidate_ids`/`query_text` is empty, or the dense side fails for any
    reason (missing cache, offline, corrupt index) — callers must treat `[]` as "skip fusion,
    keep the plain bm25 order," the SAME graceful-degradation contract `indri/model.py`'s own
    `_combine_dense`/`_dense_expand` use for a dense-side failure."""
    if dense_belief is None or not candidate_ids or not (query_text or "").strip():
        return []
    try:
        if hasattr(dense_belief, "is_ready") and not dense_belief.is_ready():
            return []
        sims = dense_belief.score(query_text, doc_ids=list(candidate_ids))
    except Exception:      # noqa: BLE001 — dense fusion is an ablation-knob extra; never
        return []          # break a search call if the dense side misbehaves
    if not sims:
        return []
    return sorted(sims, key=lambda d: (-sims[d], d))


def fuse_ranked(dense_belief, query_text: str,
                bm25_ranked: Sequence[tuple]) -> list:
    """Fuse a SINGLE-tier bm25-ranked `[(doc_id, score), ...]` list (the common exact-match
    path, `run_with_count` — ALL of `bm25_ranked` passed the filter, so it is one coverage
    tier by construction) with dense similarity RESTRICTED to those SAME doc_ids. Returns a
    re-ordered `[(doc_id, score), ...]` list (original bm25 scores preserved for display —
    only the ORDER changes, matching `LuceneIndriAdapter._combine_dense`'s "scores are
    informational, order is what fusion changes" convention). `[]`/unfusable input, or a
    dense-side miss, returns `bm25_ranked` UNCHANGED (byte-identical fallback — never worse
    than the pre-fusion ranking, never raises)."""
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
    """Fuse `coverage_topk`'s rows — `[(doc_id, mask, n_matched, score), ...]`, ALREADY sorted
    `(-n_matched, -score, doc_id)` by the caller — WITHIN each `n_matched` TIER only (see
    module docstring's "Coverage tiers" section): a doc can be reordered against others that
    satisfied the SAME number of AND-children, never promoted past a doc that satisfied MORE.
    Preserves tier order (descending `n_matched`) and each row's own `(mask, n_matched, score)`
    payload untouched — only the doc_id ORDER within a tier changes. `[]`/no dense attached
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


# --- NEW, additive-only DENSE-ONLY ordering (research_bql_donly_visit / research_bql_donly_snip)
#
# The answer to "what does BQL_DENSE's RRF fusion itself contribute, vs pure dense re-ranking of
# the SAME filter-passing candidates?" `fuse_ranked`/`fuse_coverage_tiers` above (UNCHANGED, still
# consumed by research_bql_dense_visit/research_bql_dense_snip exactly as before) RRF-combine the
# BM25 rerank with dense similarity; `fuse_ranked_dense_only`/`fuse_coverage_tiers_dense_only`
# instead order filter-passing candidates PURELY by `dense_rank_for_candidates` — the bm25 order
# is consulted only as the graceful-degradation fallback when the dense side is empty/unready
# (mirrors `fuse_ranked`/`fuse_coverage_tiers`'s own "never worse than pre-fusion, never raise"
# contract). Candidate SELECTION (the boolean/field/date filter) and, for the coverage path, the
# TIER STRUCTURE (a doc can never cross a coverage-tier boundary) are completely untouched — only
# the ORDER within what the filter already selected changes. See
# `agent_search.retrievers.structural.bql.executor.DenseOnlyStructuralExecutor`, the sibling
# executor that calls these instead of `fuse_ranked`/`fuse_coverage_tiers`.

def fuse_ranked_dense_only(dense_belief, query_text: str,
                           bm25_ranked: Sequence[tuple]) -> list:
    """DENSE-ONLY sibling of `fuse_ranked`: reorders a SINGLE-tier bm25-ranked
    `[(doc_id, score), ...]` list purely by `dense_rank_for_candidates` restricted to those SAME
    doc_ids (never a global dense search — same restriction `dense_rank_for_candidates` itself
    enforces). Original bm25 scores are preserved for display (only the ORDER changes, matching
    `fuse_ranked`'s own convention). `[]`/unfusable input, or a dense-side miss, returns
    `bm25_ranked` UNCHANGED (byte-identical fallback — never worse, never raises)."""
    if not bm25_ranked:
        return list(bm25_ranked)
    bm_ids = [d for d, _ in bm25_ranked]
    dense_ids = dense_rank_for_candidates(dense_belief, query_text, bm_ids)
    if not dense_ids:
        return list(bm25_ranked)
    score_by = dict(bm25_ranked)
    return [(d, score_by.get(d, 0.0)) for d in dense_ids]


def fuse_coverage_tiers_dense_only(dense_belief, query_text: str,
                                   rows: Sequence[tuple]) -> list:
    """DENSE-ONLY sibling of `fuse_coverage_tiers`: reorders `coverage_topk`'s rows WITHIN each
    `n_matched` TIER purely by `dense_rank_for_candidates` — the coverage-tier PRIMARY structure
    is preserved exactly as `fuse_coverage_tiers` preserves it (a doc can never cross a tier
    boundary; this isolates the fusion SIGNAL, not the tiering). `[]`/no dense attached returns
    `rows` unchanged; a tier whose dense scoring fails/returns nothing keeps its incoming
    (bm25/coverage) order for that tier only — same graceful-degradation contract as
    `fuse_coverage_tiers`."""
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
        out.extend(by_id[d] for d in dense_ids if d in by_id)
    return out
