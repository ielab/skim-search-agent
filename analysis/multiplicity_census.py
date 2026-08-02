#!/usr/bin/env python
"""Multiplicity census + dense-fusion support audit.

Answers, with numbers rather than assertion, an adversarial reviewer claim:

  "Dense fusion is one of Sieve's three named components ... its ONLY significant support
   anywhere in the paper is the BrowseComp-Plus no-dense-evidence rung (p=0.0104 EM /
   0.0145 judge); both Wikipedia versions are null. Under any correction wider than the
   paper's own hand-drawn m=4 -- including a single-dataset m=33 -- dense fusion has ZERO
   significant support in the entire paper."  (also: ~138 tests reported, ~88 in no family)

Three jobs:

T1  CENSUS.  Every hypothesis test the paper reports (latex/main.tex, latex/sections/*.tex,
    latex/tables/*.tex, latex/figures/*.tex), each with its p-value, its declared correction
    family (if any) and that family's m.  p-values come from the paper text where the paper
    prints one; where the paper reports only "significant"/"not significant" (table asterisks,
    Finding 1 judge rows, cross-backbone rows) the p is taken from the repo artifact that
    produced the table -- comparison_result.md (scripts/compare_cells.py output),
    analysis/xbackbone_*_data.json, analysis/*_data.json -- and the source is recorded per row.

T2  DENSE-FUSION SUPPORT.  Every contrast in the data whose ONLY difference is dense fusion
    (RRF fusion of a query surface with the dense evidence node), including contrasts the paper
    never tested.  These are computed live here with the project's own paired machinery
    (scripts.compare_cells.{metrics,mcnemar_p} + the recovery overlay), never re-derived.

T3  ROBUSTNESS LADDER.  Holm-Bonferroni at progressively wider pools, reported for all three
    named components (query surface's dense fusion, snippet listings, section-level fetch) so
    the comparison is even-handed.

No file under latex/ or runs/ is written or modified; this script is read-only over both.

Run: PYTHONPATH=. envs/bin/python analysis/multiplicity_census.py
Writes: analysis/multiplicity_census_data.json, analysis/multiplicity_census.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_qrels, load_judge_cache, metrics, mcnemar_p, pct,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "analysis" / "multiplicity_census_data.json"
OUT_MD = ROOT / "analysis" / "multiplicity_census.md"
JUDGE_COVERAGE_MIN = 0.90

# --------------------------------------------------------------------------------------------
# Cell registry for the contrasts computed here.  (subdir, condition) pairs are copied verbatim
# from scripts/compare_cells.py's REGISTRY so cell identity is structural, not retyped guesswork.
# --------------------------------------------------------------------------------------------
CELL = {
    # query surface (bql) family, BrowseComp-Plus-Structured
    "bql_visit":        ("_fullvisit", "agent_research_bql_visit"),
    "bql_dense_visit":  ("_fullvisit", "agent_research_bql_dense_visit"),
    "bql_fetch":        ("_headline_validation", "agent_research"),
    "bql_dense_fetch":  ("_headline_validation", "agent_research_bql_dense_fetch"),
    "bql_snip":         ("_headline_validation", "agent_research_snip"),
    "bql_dense_snip":   ("_headline_validation", "agent_research_bql_dense_snip"),   # SIEVE
    # plain-BM25 family
    "bm25_visit":       ("_visit_uncapped", "agent_research_bm25"),                   # BASELINE
    "hybrid_visit":     ("_fullvisit", "agent_research_hybrid"),
    "bm25_snip":        ("_headline_validation", "agent_research_bm25_fetch_snip"),
    "hybrid_snip":      ("_headline_validation", "agent_research_hybrid_fetch_snip"),
    # Indri family (BrowseComp-Plus-Structured only, both read levels)
    "indri_visit":      ("_fullvisit", "agent_research_indri_visit"),
    "indri_dense_visit": ("_fullvisit_dense", "agent_research_indri_visit"),
    "indri_snip":       ("_headline_validation", "agent_research_indri_snip"),
    "indri_dense_snip": ("_dense_validation", "agent_research_indri_snip"),
    # dense-only family (for the section-fetch axis)
    "dense_snip":       ("_headline_validation", "agent_research_dense_fetch"),
    "dense_fetch_plain": ("_headline_validation", "agent_research_dense_fetch_plain"),
}

_CACHE: dict = {}


def cell(dataset: str, key: str, qrels):
    dataset = DSKEY.get(dataset, dataset)
    """(metrics_dict, judge_coverage) for a named cell, memoized."""
    ck = (dataset, key)
    if ck in _CACHE:
        return _CACHE[ck]
    subdir, cond = CELL[key]
    rows = cell_rows(subdir, dataset, cond)
    if rows is None:
        _CACHE[ck] = (None, None)
        return _CACHE[ck]
    m = metrics(rows, qrels, dataset, load_judge_cache(cell_dir(subdir, dataset, cond)))
    njudged = sum(1 for v in m.values() if v["judge"] is not None)
    cov = njudged / len(m) if m else 0.0
    _CACHE[ck] = (m, cov)
    return _CACHE[ck]


def paired_em(a: dict, b: dict):
    """A - B on the shared instance set.  Exact McNemar on discordant pairs."""
    mut = set(a) & set(b)
    if not mut:
        return None
    nb = sum(1 for i in mut if b[i]["em"] and not a[i]["em"])
    nc = sum(1 for i in mut if a[i]["em"] and not b[i]["em"])
    ea, eb = pct([a[i]["em"] for i in mut]), pct([b[i]["em"] for i in mut])
    return dict(n=len(mut), em_a=ea, em_b=eb, delta=ea - eb, b=nb, c=nc, p=mcnemar_p(nb, nc))


def paired_judge(a, b, cov_a, cov_b):
    """Judge-metric version, gated at the project's own >=90% coverage rule on BOTH sides."""
    if cov_a is None or cov_b is None or cov_a < JUDGE_COVERAGE_MIN or cov_b < JUDGE_COVERAGE_MIN:
        return None
    mut = [i for i in (set(a) & set(b))
           if a[i]["judge"] is not None and b[i]["judge"] is not None]
    if not mut:
        return None
    nb = sum(1 for i in mut if b[i]["judge"] and not a[i]["judge"])
    nc = sum(1 for i in mut if a[i]["judge"] and not b[i]["judge"])
    ja, jb = pct([a[i]["judge"] for i in mut]), pct([b[i]["judge"] for i in mut])
    return dict(n=len(mut), j_a=ja, j_b=jb, delta=ja - jb, b=nb, c=nc, p=mcnemar_p(nb, nc))


# --------------------------------------------------------------------------------------------
# Holm-Bonferroni
# --------------------------------------------------------------------------------------------
def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.
    Sorting ascending p_(1)<=...<=p_(m), adjusted at rank i is
    max_{j<=i} min(1, (m+1-j) * p_(j))  -- the same formula latex/tables/ablation_family.tex
    states in its caption."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):           # rank is 0-based; j = rank+1
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def holm_survives(p: float, pool: list[float]) -> dict:
    """Holm-adjusted value of p when tested inside `pool` (p must be a member of pool)."""
    adj = holm(pool)
    i = pool.index(p)
    return dict(m=len(pool), raw=p, adjusted=adj[i], significant=adj[i] < 0.05)


def k_max(p: float, alpha: float = 0.05) -> int:
    """Largest k such that k*p <= alpha: the maximum number of tests with p-value >= p that a
    pool may contain before Holm kills this test (its Holm multiplier is exactly that count)."""
    import math
    return int(math.floor(alpha / p))


# --------------------------------------------------------------------------------------------
# T1: THE CENSUS.  One dict per hypothesis test the paper reports.
#   family: name of a correction family the PAPER declares and applies, else None
#   family_status: "declared" | "stated-but-disclaimed" | "none"
# --------------------------------------------------------------------------------------------
BCP = "BrowseComp-Plus-Structured"
HQA = "HotpotQA"
MSQ = "MuSiQue"
# display name -> on-disk dataset key used by runs/ and scripts.compare_cells
DSKEY = {BCP: "browsecomp_plus_structured", HQA: "hotpotqa_structured",
         MSQ: "musique_structured"}

# fmt: off
CENSUS = [
    # ---- Finding 1 baseline family (declared, m=6) -----------------------------------------
    dict(id="B1", grp="baseline", loc="results.tex Finding 1 / noise_floor.tex", ds=BCP, cmp="Sieve vs BM25 search+visit", metric="EM", n=830, p=0.217, family="baseline-6", m=6, status="declared", src="paper"),
    dict(id="B2", grp="baseline", loc="results.tex Finding 1", ds=BCP, cmp="Sieve vs BM25 search+visit", metric="judge", n=830, p=0.327, family="baseline-6", m=6, status="declared", src="comparison_result.md"),
    dict(id="B3", grp="baseline", loc="noise_floor.tex", ds=HQA, cmp="Sieve vs BM25 search+visit", metric="EM", n=7343, p=0.00186, family="baseline-6", m=6, status="declared", src="paper/comparison_result.md"),
    dict(id="B4", grp="baseline", loc="results.tex Finding 1", ds=HQA, cmp="Sieve vs BM25 search+visit", metric="judge", n=7343, p=0.79, family="baseline-6", m=6, status="declared", src="comparison_result.md"),
    dict(id="B5", grp="baseline", loc="noise_floor.tex", ds=MSQ, cmp="Sieve vs BM25 search+visit", metric="EM", n=2409, p=0.00073, family="baseline-6", m=6, status="declared", src="paper"),
    dict(id="B6", grp="baseline", loc="results.tex Finding 1", ds=MSQ, cmp="Sieve vs BM25 search+visit", metric="judge", n=2409, p=0.0123, family="baseline-6", m=6, status="declared", src="comparison_result.md"),

    # ---- Finding 2 single-axis ablation subfamily (declared, m=4) ---------------------------
    dict(id="A1", grp="ablation-single", loc="ablation_family.tex", ds=BCP, cmp="Sieve vs No snippets [drops (3)]", metric="EM", n=830, p=4.31e-10, family="ablation-single-4", m=4, status="declared", src="paper"),
    dict(id="A2", grp="ablation-single", loc="ablation_family.tex", ds=BCP, cmp="Sieve vs No dense evidence [drops (2)]", metric="EM", n=830, p=0.0104, family="ablation-single-4", m=4, status="declared", src="paper"),
    dict(id="A3", grp="ablation-single", loc="ablation_family.tex", ds=HQA, cmp="Sieve vs No snippets [drops (3)]", metric="EM", n=7343, p=1.39e-8, family="ablation-single-4", m=4, status="declared", src="paper"),
    dict(id="A4", grp="ablation-single", loc="ablation_family.tex", ds=MSQ, cmp="Sieve vs No snippets [drops (3)]", metric="EM", n=2409, p=7.73e-8, family="ablation-single-4", m=4, status="declared", src="paper"),

    # ---- Finding 2 compound ablation subfamily (declared, m=8) ------------------------------
    dict(id="A5", grp="ablation-compound", loc="ablation_family.tex", ds=BCP, cmp="Sieve vs Query surface + visit [drops (2)-(4)]", metric="EM", n=830, p=5.34e-4, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A6", grp="ablation-compound", loc="ablation_family.tex", ds=BCP, cmp="Sieve vs Sparse only, same interface [drops (1)-(2)]", metric="EM", n=830, p=0.0025, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A7", grp="ablation-compound", loc="ablation_family.tex", ds=BCP, cmp="Sieve vs Dense only, same interface [drops (1)-(2)]", metric="EM", n=830, p=6.67e-8, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A8", grp="ablation-compound", loc="ablation_family.tex", ds=BCP, cmp="Sieve vs Dense only, no snippets [drops (1)-(3)]", metric="EM", n=830, p=1.38e-19, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A9", grp="ablation-compound", loc="ablation_family.tex", ds=HQA, cmp="Sieve vs Sparse only, same interface [drops (1)-(2)]", metric="EM", n=7343, p=1.19e-14, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A10", grp="ablation-compound", loc="ablation_family.tex", ds=HQA, cmp="Sieve vs Dense only, no snippets [drops (1)-(3)]", metric="EM", n=7343, p=4.96e-35, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A11", grp="ablation-compound", loc="ablation_family.tex", ds=MSQ, cmp="Sieve vs Sparse only, same interface [drops (1)-(2)]", metric="EM", n=2409, p=1.37e-12, family="ablation-compound-8", m=8, status="declared", src="paper"),
    dict(id="A12", grp="ablation-compound", loc="ablation_family.tex", ds=MSQ, cmp="Sieve vs Dense only, no snippets [drops (1)-(3)]", metric="EM", n=2409, p=7.65e-13, family="ablation-compound-8", m=8, status="declared", src="paper"),

    # ---- the two late no-dense rungs: explicitly OUTSIDE the family -------------------------
    dict(id="A13", grp="ablation-single-late", loc="ablation_family.tex last block", ds=HQA, cmp="Sieve vs No dense evidence [drops (2)]", metric="EM", n=7343, p=0.103, family=None, m=None, status="none", src="paper"),
    dict(id="A14", grp="ablation-single-late", loc="ablation_family.tex last block", ds=MSQ, cmp="Sieve vs No dense evidence [drops (2)]", metric="EM", n=2409, p=0.0949, family=None, m=None, status="none", src="paper"),

    # ---- the six BCP-S Sieve-relative JUDGE versions (prose; family is EM-based) ------------
    dict(id="J1", grp="ablation-judge", loc="results.tex Finding 2 prose", ds=BCP, cmp="Sieve vs No dense evidence", metric="judge", n=830, p=0.0145, family=None, m=None, status="none", src="paper"),
    dict(id="J2", grp="ablation-judge", loc="results.tex Finding 2 prose", ds=BCP, cmp="Sieve vs Sparse only, same interface", metric="judge", n=830, p=0.0032, family=None, m=None, status="none", src="paper"),
    dict(id="J3", grp="ablation-judge", loc="results.tex Finding 2 prose", ds=BCP, cmp="Sieve vs Query surface + visit", metric="judge", n=830, p=5.8e-5, family=None, m=None, status="none", src="paper"),
    dict(id="J4", grp="ablation-judge", loc="results.tex Finding 2 prose", ds=BCP, cmp="Sieve vs Dense only, same interface", metric="judge", n=830, p=4.0e-6, family=None, m=None, status="none", src="paper"),
    dict(id="J5", grp="ablation-judge", loc="results.tex Finding 2 prose", ds=BCP, cmp="Sieve vs No snippets", metric="judge", n=830, p=4.6e-11, family=None, m=None, status="none", src="paper"),
    dict(id="J6", grp="ablation-judge", loc="results.tex Finding 2 prose", ds=BCP, cmp="Sieve vs Dense only, no snippets", metric="judge", n=830, p=2.9e-18, family=None, m=None, status="none", src="paper"),

    # ---- DCI family (declared, m=3 primary; m=5 variant folds in two judge tests) -----------
    dict(id="D1", grp="dci", loc="results.tex 'Comparison with DCI'", ds=BCP, cmp="Sieve vs DCI", metric="EM", n=830, p=1.99e-15, family="dci-3", m=3, status="declared", src="computed here"),
    dict(id="D2", grp="dci", loc="results.tex 'Comparison with DCI'", ds=HQA, cmp="Sieve vs DCI", metric="EM", n=7343, p=1.06e-40, family="dci-3", m=3, status="declared", src="computed here"),
    dict(id="D3", grp="dci", loc="results.tex 'Comparison with DCI'", ds=MSQ, cmp="Sieve vs DCI", metric="EM", n=2409, p=0.0121, family="dci-3", m=3, status="declared", src="computed here"),
    dict(id="D4", grp="dci", loc="results.tex 'Comparison with DCI'", ds=BCP, cmp="Sieve vs DCI", metric="judge", n=830, p=9.5e-21, family="dci-5(variant)", m=5, status="declared", src="paper"),
    dict(id="D5", grp="dci", loc="results.tex 'Comparison with DCI'", ds=BCP, cmp="Sieve vs BM25-filtered DCI", metric="judge", n=830, p=0.0068, family="dci-5(variant)", m=5, status="declared", src="paper"),
    dict(id="D6", grp="dci", loc="results.tex 'Comparison with DCI'", ds=BCP, cmp="Sieve vs BM25-filtered DCI", metric="EM", n=830, p=1.24e-4, family=None, m=None, status="none", src="computed here"),

    # ---- isolation block: 12 tests, paper reports them uncorrected (Holm-12 stated, disclaimed)
    dict(id="I1", grp="isolation", loc="results.tex 'Single-component isolation'", ds=BCP, cmp="Query surface alone vs Sparse only (same interface)", metric="EM", n=830, p=0.531, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I2", grp="isolation", loc="results.tex", ds=HQA, cmp="Query surface alone vs Sparse only (same interface)", metric="EM", n=7343, p=3.6e-10, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I3", grp="isolation", loc="results.tex", ds=MSQ, cmp="Query surface alone vs Sparse only (same interface)", metric="EM", n=2409, p=3.02e-8, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I4", grp="isolation", loc="results.tex", ds=BCP, cmp="Dense only vs Sparse only (same interface)", metric="EM", n=830, p=0.00893, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I5", grp="isolation", loc="results.tex", ds=HQA, cmp="Dense only vs Sparse only (same interface)", metric="EM", n=7343, p=0.00112, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I6", grp="isolation", loc="results.tex", ds=MSQ, cmp="Dense only vs Sparse only (same interface)", metric="EM", n=2409, p=0.236, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I7", grp="isolation", loc="results.tex", ds=BCP, cmp="Sieve vs Hybrid RRF + snippets + section fetch", metric="EM", n=830, p=0.102, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I8", grp="isolation", loc="results.tex", ds=HQA, cmp="Sieve vs Hybrid RRF + snippets + section fetch", metric="EM", n=7343, p=1.6e-27, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I9", grp="isolation", loc="results.tex", ds=MSQ, cmp="Sieve vs Hybrid RRF + snippets + section fetch", metric="EM", n=2409, p=1.6e-11, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I10", grp="isolation", loc="results.tex", ds=BCP, cmp="Query surface alone vs Sparse only (same interface)", metric="judge", n=830, p=0.506, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I11", grp="isolation", loc="results.tex", ds=BCP, cmp="Dense only vs Sparse only (same interface)", metric="judge", n=830, p=0.0628, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),
    dict(id="I12", grp="isolation", loc="results.tex", ds=BCP, cmp="Sieve vs Hybrid RRF + snippets + section fetch", metric="judge", n=830, p=0.248, family="isolation-12", m=12, status="stated-but-disclaimed", src="paper"),

    # ---- recall-inversion trio (uncorrected; paper notes an m=3 Holm would keep all three) ---
    dict(id="R1", grp="recall", loc="results.tex recall paragraph", ds=BCP, cmp="Sieve vs Hybrid control, gold-doc recall", metric="recall", n=830, p=8.3e-7, family="recall-3", m=3, status="stated-but-disclaimed", src="paper"),
    dict(id="R2", grp="recall", loc="results.tex recall paragraph", ds=HQA, cmp="Sieve vs Hybrid control, gold-doc recall", metric="recall", n=7343, p=3.5e-8, family="recall-3", m=3, status="stated-but-disclaimed", src="paper"),
    dict(id="R3", grp="recall", loc="results.tex recall paragraph", ds=MSQ, cmp="Sieve vs Hybrid control, gold-doc recall", metric="recall", n=2409, p=0.0115, family="recall-3", m=3, status="stated-but-disclaimed", src="paper"),

    # ---- Finding 1b tokens/calls (appendix stats-full), uncorrected, no family --------------
    dict(id="T1", grp="cost", loc="appendix.tex tab:stats-full", ds=BCP, cmp="Sieve vs baseline, tokens/episode", metric="tokens", n=830, p=1.13e-17, family=None, m=None, status="none", src="paper"),
    dict(id="T2", grp="cost", loc="appendix.tex tab:stats-full", ds=HQA, cmp="Sieve vs baseline, tokens/episode", metric="tokens", n=7343, p=3.13e-146, family=None, m=None, status="none", src="paper"),
    dict(id="T3", grp="cost", loc="appendix.tex tab:stats-full", ds=MSQ, cmp="Sieve vs baseline, tokens/episode", metric="tokens", n=2409, p=3.27e-228, family=None, m=None, status="none", src="paper"),
    dict(id="T4", grp="cost", loc="appendix.tex tab:stats-full", ds=BCP, cmp="Sieve vs baseline, LLM calls/episode", metric="calls", n=830, p=0.273, family=None, m=None, status="none", src="paper"),
    dict(id="T5", grp="cost", loc="appendix.tex tab:stats-full", ds=HQA, cmp="Sieve vs baseline, LLM calls/episode", metric="calls", n=7343, p=7.72e-150, family=None, m=None, status="none", src="paper"),
    dict(id="T6", grp="cost", loc="appendix.tex tab:stats-full", ds=MSQ, cmp="Sieve vs baseline, LLM calls/episode", metric="calls", n=2409, p=3.62e-14, family=None, m=None, status="none", src="paper"),

    # ---- Finding 5 second backbone, uncorrected, no family ----------------------------------
    dict(id="X1", grp="xbackbone", loc="results.tex Finding 5", ds=BCP, cmp="Qwen: Sieve vs baseline", metric="EM", n=830, p=1.2e-22, family=None, m=None, status="none", src="xbackbone_full_data.json"),
    dict(id="X2", grp="xbackbone", loc="results.tex Finding 5", ds=BCP, cmp="Qwen: Sieve vs baseline", metric="judge", n=830, p=2.6e-23, family=None, m=None, status="none", src="paper"),
    dict(id="X3", grp="xbackbone", loc="results.tex Finding 5", ds=BCP, cmp="Qwen: Sieve vs baseline, tokens", metric="tokens", n=830, p=2.35e-9, family=None, m=None, status="none", src="paper"),
    dict(id="X4", grp="xbackbone", loc="results.tex Finding 5", ds=BCP, cmp="Qwen: Sieve vs baseline, LLM calls", metric="calls", n=830, p=6.16e-16, family=None, m=None, status="none", src="xbackbone_full_data.json"),
    dict(id="X5", grp="xbackbone", loc="results.tex Finding 5", ds=MSQ, cmp="Qwen: Sieve vs baseline", metric="EM", n=2409, p=9.17e-8, family=None, m=None, status="none", src="paper"),
    dict(id="X6", grp="xbackbone", loc="results.tex Finding 5", ds=MSQ, cmp="Qwen: Sieve vs baseline, tokens", metric="tokens", n=2409, p=0.0085, family=None, m=None, status="none", src="paper"),
    dict(id="X7", grp="xbackbone", loc="results.tex Finding 5", ds=MSQ, cmp="Qwen: Sieve vs baseline, LLM calls", metric="calls", n=2409, p=2.33e-44, family=None, m=None, status="none", src="xbackbone_musique_data.json"),
    dict(id="X8", grp="xbackbone", loc="results.tex Finding 5", ds=MSQ, cmp="Qwen: Sieve vs baseline, gold-doc recall", metric="recall", n=2409, p=0.655, family=None, m=None, status="none", src="xbackbone_musique_data.json"),

    # ---- Appendix factorial grid: 7 visit->fetch+snippets contrasts (declared Holm, m=7) -----
    dict(id="F1", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="BM25: visit -> fetch+snippets", metric="EM", n=830, p=0.066, family="factorial-7", m=7, status="declared", src="paper"),
    dict(id="F2", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="Dense: visit -> fetch+snippets", metric="EM", n=830, p=0.0014, family="factorial-7", m=7, status="declared", src="paper"),
    dict(id="F3", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="Hybrid RRF: visit -> fetch+snippets", metric="EM", n=830, p=0.368, family="factorial-7", m=7, status="declared", src="paper"),
    dict(id="F4", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="Query surface: visit -> fetch+snippets", metric="EM", n=830, p=0.307, family="factorial-7", m=7, status="declared", src="paper"),
    dict(id="F5", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="Query surface+dense (Sieve): visit -> fetch+snippets", metric="EM", n=830, p=2.8e-6, family="factorial-7", m=7, status="declared", src="paper"),
    dict(id="F6", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="Indri: visit -> fetch+snippets", metric="EM", n=830, p=0.0098, family="factorial-7", m=7, status="declared", src="paper"),
    dict(id="F7", grp="factorial", loc="appendix.tex tab:factorial-grid", ds=BCP, cmp="Indri+dense: visit -> fetch+snippets", metric="EM", n=830, p=0.0055, family="factorial-7", m=7, status="declared", src="paper"),

    # ---- noise-floor replicate ---------------------------------------------------------------
    dict(id="N1", grp="noise-floor", loc="results.tex / noise_floor.tex / benchmark.tex", ds=BCP, cmp="Structure-blind baseline: flat vs structured corpus (null replicate)", metric="EM", n=830, p=0.372, family=None, m=None, status="none", src="paper"),

    # ---- full-text-dump judge deltas vs baseline (prose) --------------------------------------
    dict(id="P1", grp="table-delta", loc="results.tex 'Why a partially judged cell is withheld'", ds=BCP, cmp="Dense full-text dump vs baseline", metric="judge", n=830, p=4.5e-33, family=None, m=None, status="none", src="paper"),
    dict(id="P2", grp="table-delta", loc="results.tex 'Why a partially judged cell is withheld'", ds=BCP, cmp="BM25 full-text dump vs baseline", metric="judge", n=830, p=9.0e-36, family=None, m=None, status="none", src="paper"),

    # ---- TOST equivalence tests (4 margins) ---------------------------------------------------
    dict(id="E1", grp="tost", loc="results.tex Finding 1b", ds=BCP, cmp="TOST equivalence, margin +/-1pp", metric="EM", n=830, p=None, family=None, m=None, status="none", src="equivalence_and_latency.md", note="fails to certify"),
    dict(id="E2", grp="tost", loc="results.tex Finding 1b", ds=BCP, cmp="TOST equivalence, margin +/-2pp", metric="EM", n=830, p=None, family=None, m=None, status="none", src="equivalence_and_latency.md", note="fails to certify"),
    dict(id="E3", grp="tost", loc="results.tex Finding 1b", ds=BCP, cmp="TOST equivalence, margin +/-3pp", metric="EM", n=830, p=None, family=None, m=None, status="none", src="equivalence_and_latency.md", note="fails to certify"),
    dict(id="E4", grp="tost", loc="results.tex Finding 1b", ds=BCP, cmp="TOST equivalence, margin +/-5pp", metric="EM", n=830, p=None, family=None, m=None, status="none", src="equivalence_and_latency.md", note="fails to certify"),

    # ---- Shapiro-Wilk normality diagnostics on the six paired-difference distributions --------
    dict(id="S1", grp="diagnostic", loc="appendix.tex app:stats-full", ds="all", cmp="Shapiro-Wilk on paired token differences (3 datasets)", metric="normality", n=None, p=2.7e-29, family=None, m=None, status="none", src="paper", note="reported as '<=2.7e-29 throughout'; 3 tests"),
    dict(id="S2", grp="diagnostic", loc="appendix.tex app:stats-full", ds="all", cmp="Shapiro-Wilk on paired call-count differences (3 datasets)", metric="normality", n=None, p=2.7e-29, family=None, m=None, status="none", src="paper", note="3 tests"),
]
# fmt: on

# The 54 baseline-relative Delta-EM tests carried by the three results tables' "*" convention.
# p-values from comparison_result.md (scripts/compare_cells.py, the generator of those columns).
# fmt: off
TABLE_DELTAS = {
    BCP: [
        ("Retrieve-then-read (BM25)", 1.49e-79), ("Retrieve-then-read (dense)", 6.4e-55),
        ("BM25 full-text dump", 3.26e-32), ("Dense full-text dump", 1.17e-30),
        ("DCI", 1.61e-13), ("BM25-filtered DCI", 0.0126),
        ("BM25 search + visit (k=10)", 0.678), ("Dense search + visit", 0.201),
        ("Hybrid sparse-dense (RRF) + visit", 0.173), ("Query surface + visit", 0.0337),
        ("Query surface + dense evidence + visit", 3.86e-4), ("Indri + visit", 0.00521),
        ("Indri + dense evidence + visit", 0.0842), ("Sparse only, same interface", 0.0664),
        ("Hybrid RRF + snippets + section fetch", 0.763), ("Indri + section fetch", 2.25e-12),
        ("Indri + snippets + section fetch", 1.88e-7),
        ("Indri + dense + snippets + section fetch", 8.65e-6),
        ("Dense only, same interface", 1.11e-5), ("Dense only, no snippets", 6.5e-16),
        ("No snippets", 1.91e-6), ("No dense evidence", 0.264),
    ],
    HQA: [
        ("Retrieve-then-read (BM25)", 3.47e-119), ("Retrieve-then-read (dense)", 3.46e-113),
        ("BM25 full-text dump", 0.0155), ("Dense full-text dump", 2.23e-13),
        ("BM25 search + visit (k=10)", 0.0153), ("Dense search + visit", 0.158),
        ("Hybrid sparse-dense (RRF) + visit", 1.12e-11),
        ("Indri + dense evidence + visit", 0.0022), ("Sparse only, same interface", 2.32e-6),
        ("Hybrid RRF + snippets + section fetch", 2.47e-15),
        ("Indri + dense + snippets + section fetch", 7.38e-5),
        ("Dense only, same interface", 7.71e-16), ("Dense only, no snippets", 7.55e-21),
        ("DCI", 1.56e-28), ("No snippets", 0.0131), ("No dense evidence", 0.126),
    ],
    MSQ: [
        ("Retrieve-then-read (BM25)", 1.29e-96), ("Retrieve-then-read (dense)", 4.68e-67),
        ("BM25 full-text dump", 1.58e-11), ("Dense full-text dump", 2.11e-13),
        ("BM25 search + visit (k=10)", 0.0417), ("Dense search + visit", 0.237),
        ("Hybrid sparse-dense (RRF) + visit", 3.12e-9),
        ("Indri + dense evidence + visit", 0.536), ("Sparse only, same interface", 2.91e-4),
        ("Hybrid RRF + snippets + section fetch", 0.00117),
        ("Indri + dense + snippets + section fetch", 0.958),
        ("Dense only, same interface", 1.07e-6), ("Dense only, no snippets", 8.24e-5),
        ("DCI", 0.608), ("No snippets", 0.0615), ("No dense evidence", 0.0782),
    ],
}
# fmt: on

for _ds, _rows in TABLE_DELTAS.items():
    for _i, (_label, _p) in enumerate(_rows):
        CENSUS.append(dict(
            id=f"TD-{_ds[:3]}-{_i+1}", grp="table-delta",
            loc="main_results.tex / wiki_results.tex ('*' = p<0.05, paired exact McNemar)",
            ds=_ds, cmp=f"{_label} vs BM25 search + visit baseline", metric="EM", n=None,
            p=_p, family=None, m=None, status="none", src="comparison_result.md"))


# --------------------------------------------------------------------------------------------
# T2: every contrast in the data whose ONLY difference is dense fusion, plus the two other
# components' single-axis contrasts, computed live.
# --------------------------------------------------------------------------------------------
DENSE_CONTRASTS = [
    # (key, dataset, cell_with_fusion, cell_without, read-level, surface, in_paper?)
    ("bql@fetch+snip (Sieve vs No dense evidence)", BCP, "bql_dense_snip", "bql_snip",
     "section fetch + snippets", "structured query surface", "YES - the rung in question"),
    ("bql@fetch+snip", HQA, "bql_dense_snip", "bql_snip",
     "section fetch + snippets", "structured query surface", "YES - the rung in question"),
    ("bql@fetch+snip", MSQ, "bql_dense_snip", "bql_snip",
     "section fetch + snippets", "structured query surface", "YES - the rung in question"),
    ("bql@fetch, plain listing", BCP, "bql_dense_fetch", "bql_fetch",
     "section fetch, plain listing", "structured query surface", "NO - cell excluded from paper"),
    ("bql@visit", BCP, "bql_dense_visit", "bql_visit",
     "whole-document visit", "structured query surface",
     "descriptive only (29.6 vs 32.3 quoted, no p)"),
    ("bm25@fetch+snip", BCP, "hybrid_snip", "bm25_snip",
     "section fetch + snippets", "plain BM25 (unstructured)", "NO"),
    ("bm25@fetch+snip", HQA, "hybrid_snip", "bm25_snip",
     "section fetch + snippets", "plain BM25 (unstructured)", "NO"),
    ("bm25@fetch+snip", MSQ, "hybrid_snip", "bm25_snip",
     "section fetch + snippets", "plain BM25 (unstructured)", "NO"),
    ("bm25@visit", BCP, "hybrid_visit", "bm25_visit",
     "whole-document visit", "plain BM25 (unstructured)", "YES - table Delta-EM row"),
    ("bm25@visit", HQA, "hybrid_visit", "bm25_visit",
     "whole-document visit", "plain BM25 (unstructured)", "YES - table Delta-EM row"),
    ("bm25@visit", MSQ, "hybrid_visit", "bm25_visit",
     "whole-document visit", "plain BM25 (unstructured)", "YES - table Delta-EM row"),
    ("indri@fetch+snip", BCP, "indri_dense_snip", "indri_snip",
     "section fetch + snippets", "Indri graded surface", "NO"),
    ("indri@visit", BCP, "indri_dense_visit", "indri_visit",
     "whole-document visit", "Indri graded surface", "NO"),
]

# single-axis contrasts for the OTHER two components, for an even-handed ladder
OTHER_CONTRASTS = [
    # snippets (3): fetch-level, dense-fused structured surface, snippets on vs off  == paper's rung
    ("snippets (3) @ Sieve", BCP, "bql_dense_snip", "bql_dense_fetch", "component (3) snippets"),
    ("snippets (3) @ Sieve", HQA, "bql_dense_snip", "bql_dense_fetch", "component (3) snippets"),
    ("snippets (3) @ Sieve", MSQ, "bql_dense_snip", "bql_dense_fetch", "component (3) snippets"),
    # snippets on other surfaces (not in the paper's family)
    ("snippets (3) @ dense surface", BCP, "dense_snip", "dense_fetch_plain", "component (3) snippets"),
    ("listing+read change @ Indri surface", BCP, "indri_snip", "indri_visit",
     "NOT single-axis: changes both read mode and listing"),
    # section fetch (4): the ONLY single-axis read-axis contrast that exists, plain listings both sides
    ("section fetch (4) @ query surface+dense", BCP, "bql_dense_fetch", "bql_dense_visit",
     "component (4) section fetch"),
    ("section fetch (4) @ query surface, no dense", BCP, "bql_fetch", "bql_visit",
     "component (4) section fetch"),
]


def compute_contrasts():
    out = {"dense": [], "other": []}
    for label, ds, ka, kb, read, surface, in_paper in DENSE_CONTRASTS:
        qrels = load_qrels(DSKEY[ds])
        a, cov_a = cell(ds, ka, qrels)
        b, cov_b = cell(ds, kb, qrels)
        if a is None or b is None:
            out["dense"].append(dict(label=label, dataset=ds, status="cell missing",
                                     cell_a=ka, cell_b=kb))
            continue
        em = paired_em(a, b)
        ju = paired_judge(a, b, cov_a, cov_b)
        out["dense"].append(dict(label=label, dataset=ds, read_level=read, surface=surface,
                                 in_paper=in_paper, cell_a=ka, cell_b=kb, em=em, judge=ju,
                                 judge_cov_a=cov_a, judge_cov_b=cov_b))
    for label, ds, ka, kb, kind in OTHER_CONTRASTS:
        qrels = load_qrels(DSKEY[ds])
        a, cov_a = cell(ds, ka, qrels)
        b, cov_b = cell(ds, kb, qrels)
        if a is None or b is None:
            out["other"].append(dict(label=label, dataset=ds, status="cell missing"))
            continue
        out["other"].append(dict(label=label, dataset=ds, kind=kind, cell_a=ka, cell_b=kb,
                                 em=paired_em(a, b),
                                 judge=paired_judge(a, b, cov_a, cov_b)))
    return out


def compute_dci():
    """Sieve vs DCI / BM25-filtered DCI on EM -- the paper's m=3 family, whose p-values the paper
    reports only as 'significant by a wide margin'."""
    res = {}
    pairs = [(BCP, "agent", "agent_research_dci", "DCI"),
             (HQA, "agent", "agent_research_dci", "DCI"),
             (MSQ, "agent", "agent_research_dci", "DCI"),
             (BCP, "agent", "agent_research_bm25_dci", "BM25-filtered DCI")]
    for ds, subdir, cond, label in pairs:
        dk = DSKEY[ds]
        qrels = load_qrels(dk)
        sieve, _ = cell(ds, "bql_dense_snip", qrels)
        rows = cell_rows(subdir, dk, cond)
        if sieve is None or rows is None:
            res[f"{ds}|{label}"] = "missing"
            continue
        other = metrics(rows, qrels, dk, load_judge_cache(cell_dir(subdir, dk, cond)))
        res[f"{ds}|{label}"] = paired_em(sieve, other)
    return res


# --------------------------------------------------------------------------------------------
# Sanity gate: recompute two published numbers before trusting anything else this script says.
# --------------------------------------------------------------------------------------------
def sanity_gate():
    qrels = load_qrels(DSKEY[BCP])
    sieve, cov_s = cell(BCP, "bql_dense_snip", qrels)
    nodense, cov_n = cell(BCP, "bql_snip", qrels)
    nosnip, _ = cell(BCP, "bql_dense_fetch", qrels)
    g1 = paired_em(sieve, nodense)
    g1j = paired_judge(sieve, nodense, cov_s, cov_n)
    g2 = paired_em(sieve, nosnip)
    checks = [
        ("no-dense EM delta", round(g1["delta"], 1), 4.7),
        ("no-dense EM p", float(f"{g1['p']:.4f}"), 0.0104),
        ("no-dense judge p", float(f"{g1j['p']:.4f}"), 0.0145),
        ("no-snippets EM delta", round(g2["delta"], 1), 11.6),
    ]
    ok = all(abs(c - e) < 1e-6 for _, c, e in checks)
    return dict(checks=[dict(label=l, computed=c, expected=e, ok=abs(c - e) < 1e-6)
                        for l, c, e in checks], passed=ok)


# --------------------------------------------------------------------------------------------
# T3: the robustness ladder
# --------------------------------------------------------------------------------------------
def build_ladder(census):
    """For each named component's supporting test(s), Holm-adjust inside progressively wider
    pools, and report survival."""
    pv = {t["id"]: t["p"] for t in census if t["p"] is not None}

    def pool(ids):
        return [pv[i] for i in ids]

    def entry(name, tid, ids, note=""):
        ps = pool(ids)
        r = holm_survives(pv[tid], ps)
        return dict(level=name, test=tid, m=r["m"], raw=r["raw"], adjusted=r["adjusted"],
                    significant=r["significant"], note=note)

    all_ids = [t["id"] for t in census if t["p"] is not None]
    bcp_all = [t["id"] for t in census if t["p"] is not None and t["ds"] == BCP]
    bcp_acc = [t["id"] for t in census if t["p"] is not None and t["ds"] == BCP
               and t["metric"] in ("EM", "judge")]
    bcp_em = [t["id"] for t in census if t["p"] is not None and t["ds"] == BCP
              and t["metric"] == "EM"]
    single_axis = ["A1", "A2", "A3", "A4", "A13", "A14"]          # the 6 that exist on EM
    single_axis4 = ["A1", "A2", "A3", "A4"]
    ablation12 = [f"A{i}" for i in range(1, 13)]
    ablation14 = ablation12 + ["A13", "A14"]

    ladders = {}
    # --- dense fusion, EM (A2) and judge (J1) ---
    ladders["dense_fusion"] = [
        entry("(a) paper's declared single-axis subfamily", "A2", single_axis4,
              "as published: adj p = 0.0104, tested last, multiplier 1"),
        entry("(a') paper's own m=6 variant (family + 2 late wiki rungs)", "A2", single_axis,
              "paper states this in the ablation-table caption"),
        entry("(a'') whole 12-comparison ablation family pooled", "A2", ablation12, ""),
        entry("(a''') 14 = 12-family + 2 late wiki rungs", "A2", ablation14, ""),
        entry("(b) all BrowseComp-Plus-Structured EM tests pooled", "A2", bcp_em, ""),
        entry("(b') all BCP-S accuracy tests (EM+judge) pooled", "A2", bcp_acc, ""),
        entry("(b'') all BCP-S tests of any kind pooled", "A2", bcp_all, ""),
        entry("(c) all single-axis ablation comparisons, all datasets", "A2", single_axis, ""),
        entry("(d) every test in the paper pooled (global)", "A2", all_ids, ""),
    ]
    ladders["dense_fusion_judge"] = [
        entry("(b) all BCP-S EM tests + this judge test", "J1", bcp_em + ["J1"], ""),
        entry("(b') all BCP-S accuracy tests pooled", "J1", bcp_acc, ""),
        entry("(b'') all BCP-S tests pooled", "J1", bcp_all, ""),
        entry("(d) global", "J1", all_ids, ""),
    ]
    # --- snippets: the three single-axis rungs ---
    for tid, ds in (("A1", BCP), ("A3", HQA), ("A4", MSQ)):
        ladders[f"snippets_{ds}"] = [
            entry("(a) paper's declared single-axis subfamily", tid, single_axis4, ""),
            entry("(a''') 14-comparison ablation pool", tid, ablation14, ""),
            entry("(b) all BCP-S EM tests pooled" if tid == "A1"
                  else "(b) a same-size (43-test) dataset pool, this test injected",
                  tid, bcp_em + ([] if tid == "A1" else [tid]), ""),
            entry("(c) all single-axis ablation comparisons, all datasets", tid, single_axis, ""),
            entry("(d) every test in the paper pooled (global)", tid, all_ids, ""),
        ]
    return ladders, dict(all=len(all_ids), bcp_all=len(bcp_all), bcp_acc=len(bcp_acc),
                         bcp_em=len(bcp_em))


def global_correction(census):
    """Holm over every p-valued test in the paper: who lives, who dies.  This is a deliberately
    over-conservative standard (no published paper applies one correction across its whole
    results section); it is reported to show the shape of the ladder's far end, not as the
    'right' analysis."""
    tests = [t for t in census if t["p"] is not None]
    adj = holm([t["p"] for t in tests])
    rows = [dict(id=t["id"], ds=t["ds"], cmp=t["cmp"], metric=t["metric"], raw=t["p"],
                 adjusted=a, survives=a < 0.05, grp=t["grp"])
            for t, a in zip(tests, adj)]
    rows.sort(key=lambda r: r["raw"])
    return dict(m=len(tests), n_survive=sum(1 for r in rows if r["survives"]),
                n_die=sum(1 for r in rows if not r["survives"]), rows=rows)


def blockers(census, p_ref: float, pool_ids: list[str]):
    """How many tests in a pool have p >= p_ref -- exactly the Holm multiplier that kills a test
    at p_ref, so this is the mechanical reason a test dies in a wider pool."""
    pv = {t["id"]: t["p"] for t in census if t["p"] is not None}
    return sum(1 for i in pool_ids if pv[i] >= p_ref)


def fmt_p(p):
    if p is None:
        return "--"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}"


def write_md(data):
    C = data["census"]
    acc = data["accounting"]
    L = data["ladders"]
    g = data["global"]
    pool_line = acc["pool_sizes"]
    o = []
    A = o.append
    A("# Multiplicity census and the dense-fusion support audit\n")
    A("Generated by `analysis/multiplicity_census.py` (read-only over `latex/` and `runs/`). "
      "Answers an adversarial reviewer's claim that dense fusion -- one of \\textsc{Sieve}'s three "
      "named components -- has significant support at exactly one place in the paper, and none at "
      "all under any wider multiplicity correction.\n")
    A("**Sanity gate.** Before anything else this script recomputes two published numbers from "
      "`runs/` with the project's own paired machinery "
      "(`scripts.compare_cells.metrics`/`mcnemar_p` + the forced-answer recovery overlay): the "
      "no-dense-evidence rung on \\textsc{BrowseComp-Plus-Structured} (+4.7 EM, p=0.0104; judge "
      "p=0.0145) and the no-snippets rung (+11.6 EM). Both reproduce exactly. Every number below "
      "is computed on that same path.\n")

    # ---------------- T1 ----------------
    A("\n## T1. The census: every hypothesis test the paper reports\n")
    A(f"**{acc['total_tests']} hypothesis tests.** "
      f"{acc['in_declared_family']} sit inside a correction family the paper declares *and* "
      f"applies; {acc['in_stated_but_disclaimed_family']} sit inside a family the paper computes "
      "but explicitly disclaims (\"we report them uncorrected ... we state what a correction would "
      f"do anyway\"); **{acc['outside_any_family']} sit in no family at all** "
      f"({acc['outside_any_family_strict']} if the disclaimed families are counted as uncorrected, "
      "which is how the paper itself describes them).\n")
    A("| declared family | m | tests | status |")
    A("|---|---:|---:|---|")
    for k, v in acc["families"].items():
        A(f"| {k} | {v['m']} | {v['n']} | {v['status']} |")
    A("")
    A("**Against the reviewer's count.** The reviewer says ~138 tests, ~88 in no declared family. "
      f"My count is **{acc['total_tests']} / {acc['outside_any_family']}**. The reviewer's numbers "
      "are accurate to within counting convention; the 7-test gap is entirely about whether one "
      "counts the six Shapiro--Wilk normality diagnostics, the six Wilcoxon cross-checks the "
      "appendix asserts without printing ('no conclusion changes relative to a nonparametric "
      "alternative'), and the four TOST margins as separate tests. Nothing in the reviewer's "
      "arithmetic is wrong, and the substantive point it carries -- that roughly two thirds of the "
      "paper's tests are reported with no multiplicity control -- is correct.\n")
    A("The largest single block outside any family is the 54 baseline-relative $\\Delta$EM tests "
      "carried by the `*` convention in Tables 2--4 and 10--11 (22 on \\textsc{BCP-S}, 16 on each "
      "Wikipedia dataset). These are real paired McNemar tests -- the asterisk *is* a significance "
      "claim -- and no family covers them. The next largest are the 12-test isolation block "
      "(corrected only hypothetically), the 8 cross-backbone tests, the 6 \\textsc{BCP-S} "
      "Sieve-relative judge versions (the declared ablation family is EM-only), and the 6 "
      "token/call comparisons.\n")
    A("\n### Full census\n")
    A("| id | dataset | comparison | metric | n | raw p | family (m) | status |")
    A("|---|---|---|---|---:|---:|---|---|")
    for t in C:
        fam = f"{t['family']} ({t['m']})" if t["family"] else "--"
        A(f"| {t['id']} | {t['ds']} | {t['cmp']} | {t['metric']} | "
          f"{t['n'] if t['n'] else '--'} | {fmt_p(t['p'])} | {fam} | {t['status']} |")
    A("")

    # ---------------- T2 ----------------
    A("\n## T2. Every contrast in the data that isolates dense fusion\n")
    A("Dense fusion = component (2), the RRF fusion of a query surface's BM25 ranking with the "
      "dense evidence node. A contrast isolates it only if the two arms are identical in query "
      "surface, listing style and read mode, and differ solely in whether dense evidence is fused "
      "in. Thirteen such contrasts exist in `runs/`. The paper reports a test for six of them -- "
      "the three no-dense-evidence rungs (plus the BCP-S judge version) and, through its tables' "
      "asterisk convention, the three hybrid-vs-plain-BM25 visit rows -- and the other seven exist "
      "only in the run data. A fourteenth does not exist: the second backbone ran only the "
      "baseline and \\textsc{Sieve} (`runs/_xbackbone` holds `agent_research_bm25` and "
      "`agent_research_bql_dense_snip` and nothing else), so dense fusion is untested there.\n")
    A("| dataset | query surface | read level | $\\Delta$EM | p (EM) | $\\Delta$judge | p (judge) | in the paper? |")
    A("|---|---|---|---:|---:|---:|---:|---|")
    for r in data["dense_fusion_contrasts"]:
        if not r.get("em"):
            continue
        j = r.get("judge")
        jd = f"{j['delta']:+.1f}" if j else "--"
        jp = fmt_p(j["p"]) if j else "--"
        A(f"| {r['dataset']} | {r['surface']} | {r['read_level']} | "
          f"{r['em']['delta']:+.1f} | {fmt_p(r['em']['p'])} | {jd} | {jp} | {r['in_paper']} |")
    A("")
    A("**What this changes and what it does not.** The reviewer's inventory of *significant* "
      "support is right: the \\textsc{BrowseComp-Plus-Structured} no-dense-evidence rung "
      "(+4.7 EM, p=0.0104; +4.6 judge, p=0.0145) is the only dense-fusion-isolating contrast "
      "anywhere in this project that reaches p<0.05 in dense fusion's favour, on either metric, "
      "on any dataset, at any read level, on any query surface.\n")
    A("But the reviewer's inventory of *evidence* is incomplete in a way that cuts both ways:\n")
    A("- **For dense fusion.** At \\textsc{Sieve}'s own read level (section fetch + snippet "
      "listings) on \\textsc{BCP-S}, fusion is directionally positive on all three query surfaces "
      "it can be tested on -- the structured surface (+4.7 EM / +4.6 judge), plain BM25 (+2.8 / "
      "+3.6) and the Indri surface (+1.7 / +2.9) -- six point estimates, all positive, on two "
      "metrics. That is a coherent pattern rather than one lucky cell, and the reviewer's summary "
      "does not mention it. None of the five non-\\textsc{Sieve} estimates is individually "
      "significant (p=0.087--0.365), so it is corroboration, not additional significant support.\n")
    A("- **Against dense fusion.** The same operation is significantly *harmful* three times: on "
      "plain BM25 under whole-document visit on HotpotQA (-3.5, p=1.1e-11) and MuSiQue (-4.7, "
      "p=3.1e-9), and on plain BM25 at \\textsc{Sieve}'s own read level on HotpotQA (-1.7, "
      "p=0.0015). On the structured surface under visit it is -2.7 (p=0.145), and at the "
      "plain-listing fetch level it is -0.1 (p=1.0) -- the cell the paper prints nowhere. Counting "
      "significant results alone, dense fusion is 1 for and 3 against.\n")
    A("- The three read levels on \\textsc{BCP-S}, on the structured surface, are -2.7 (visit, "
      "plain listing), -0.1 (fetch, plain listing), +4.7 (fetch + snippets). That is the paper's "
      "own interaction claim, and it is the strongest honest reading of the component: dense "
      "fusion is worth something only in the presence of snippet listings, and only on the long "
      "web corpus.\n")

    A("\n### The other two components, on the same basis\n")
    A("| contrast | dataset | $\\Delta$EM | p | what it isolates |")
    A("|---|---|---:|---:|---|")
    for r in data["other_component_contrasts"]:
        if not r.get("em"):
            continue
        A(f"| {r['label']} | {r['dataset']} | {r['em']['delta']:+.1f} | "
          f"{fmt_p(r['em']['p'])} | {r['kind']} |")
    A("")
    A("Snippets (3) are supported by three significant single-axis removals, one per dataset, plus "
      "a fourth significant removal on the dense-only surface (+6.7, p=7.7e-05) that the paper "
      "does not present as such. (The Indri row in the table above is listed for completeness and "
      "is NOT single-axis -- it changes read mode and listing together.) Section-level fetch (4) is the component with the weakest "
      "evidence in the paper, not dense fusion: it has **no** single-axis rung the paper tests, "
      "and the two single-axis read-axis contrasts that exist in the data both run *against* it "
      "(-2.3, p=0.244 with dense fusion present; -4.8, p=0.014 without it, i.e. significantly "
      "worse). The paper's own contribution bullet already concedes the first of these "
      "(\"cuts distinct tokens per episode by 26\\% and costs 2.3 exact-match points\"), which "
      "makes fetch a token-efficiency component by the paper's own account.\n")

    # ---------------- T3 ----------------
    A("\n## T3. The robustness ladder\n")
    A("Holm--Bonferroni step-down, $\\alpha=0.05$, using the same formula the ablation table's "
      "caption states. A test at raw p survives a pool iff (number of tests in that pool with "
      "p >= its own) x p <= 0.05. For dense fusion's EM rung (p=0.0104) that "
      f"budget is **{data['holm_survival_thresholds']['dense_em_kmax']} tests**; for its judge "
      f"version (p=0.0145) it is **{data['holm_survival_thresholds']['dense_judge_kmax']}**. That "
      "single number decides the whole ladder.\n")
    A(f"**On the reviewer's m=33.** No pool of BrowseComp-Plus-Structured tests I can construct "
      f"from the paper has m=33: it is {pool_line['bcp_em']} counting EM tests only, "
      f"{pool_line['bcp_acc']} counting EM and judge, {pool_line['bcp_all']} counting everything "
      "on that dataset. The verdict is insensitive to which of these the reviewer meant, and to "
      "the 10-test gap, because what kills the rung is not m but the number of that dataset's own "
      f"tests that are *less* significant than it: {data['blockers']['bcp_em_ge_dense']} of the "
      f"{pool_line['bcp_em']} BCP-S EM tests have p >= 0.0104, and only 4 are affordable. To "
      "rescue the rung at the dataset level one would have to delete 13 of those 17 nulls from "
      "the pool.\n")
    A("### Dense fusion (component 2)\n")
    A("| level | pool | m | Holm-adjusted p | verdict |")
    A("|---|---|---:|---:|---|")
    for r in L["dense_fusion"]:
        A(f"| EM | {r['level']} | {r['m']} | {r['adjusted']:.4g} | "
          f"{'**survives**' if r['significant'] else 'dies'} |")
    for r in L["dense_fusion_judge"]:
        A(f"| judge | {r['level']} | {r['m']} | {r['adjusted']:.4g} | "
          f"{'**survives**' if r['significant'] else 'dies'} |")
    A("")
    A("### Snippet listings (component 3)\n")
    A("| dataset | level | m | Holm-adjusted p | verdict |")
    A("|---|---|---:|---:|---|")
    for ds in (BCP, HQA, MSQ):
        for r in L[f"snippets_{ds}"]:
            A(f"| {ds} | {r['level']} | {r['m']} | {r['adjusted']:.3g} | "
              f"{'**survives**' if r['significant'] else 'dies'} |")
    A("")
    A("### Section-level fetch (component 4)\n")
    A("No rung to correct. The component has no significant supporting test at any correction "
      "level including none at all, because it has no supporting test: its only two single-axis "
      "contrasts are negative, one of them significantly so uncorrected.\n")
    A(f"### The far end: one Holm correction over all {g['m']} tests in the paper\n")
    A(f"{g['n_survive']} of {g['m']} tests survive; {g['n_die']} do not. This is a deliberately "
      "over-conservative standard -- essentially no published paper corrects across its entire "
      "results section, and doing so treats a token-count test and an accuracy test as members of "
      "one hypothesis family, which they are not. It is reported here because it is the ceiling of "
      "the reviewer's argument, and because what survives it is informative:\n")
    A("| id | dataset | comparison | metric | raw p | global adj p |")
    A("|---|---|---|---|---:|---:|")
    for r in g["rows"]:
        if r["survives"]:
            A(f"| {r['id']} | {r['ds']} | {r['cmp']} | {r['metric']} | {fmt_p(r['raw'])} | "
              f"{fmt_p(r['adjusted'])} |")
    A("")
    A("The honest asymmetry, stated plainly: **all three of the snippet-removal rungs survive a "
      "global correction over every test in the paper. Dense fusion's single supporting rung does "
      "not survive a correction over its own dataset.**\n")
    A("And the symmetric caution, because the global level is indiscriminate: it also kills two of "
      "the paper's three headline accuracy results (HotpotQA EM, raw p=0.00186, adj. 0.099; "
      "MuSiQue judge, raw 0.0123, adj. 0.49), the query-surface-vs-plain-BM25 contrast on "
      "\\textsc{BCP-S} (A6, raw 0.0025, adj. 0.13) and the paper's own MuSiQue recall test (raw "
      "0.0115, adj. 0.47). Only MuSiQue EM survives from Finding 1. A reviewer who applies this "
      "level to dense fusion alone is applying it selectively.\n")

    # ---------------- T4 ----------------
    A("\n## T4. The budget confound under the two Wikipedia nulls\n")
    A("Confirmed from `analysis/step_budget_audit.md` sections 1 and 2, whose provenance comes "
      "from configuration only (per-row `max_steps` stamped by `evaluation/run_eval.py`, shard "
      "`config.json`, cell `config.json`) and never from observed step counts:\n")
    A("| dataset | cell | condition | configured step cap |")
    A("|---|---|---|---|")
    A("| HotpotQA | \\textsc{Sieve} | `agent_research_bql_dense_snip` | **50** (7343/7343 rows) |")
    A("| HotpotQA | No dense evidence | `agent_research_snip` | **100** (7343/7343 rows) |")
    A("| MuSiQue | \\textsc{Sieve} | `agent_research_bql_dense_snip` | **50** (2409/2409 rows) |")
    A("| MuSiQue | No dense evidence | `agent_research_snip` | **100** (2409/2409 rows) |")
    A("| \\textsc{BCP-S} | both arms | -- | 100 / 100 (matched) |")
    A("")
    A("The audit's repair table records the consequence: for both Wikipedia no-dense comparisons "
      "the cap cross-tabulation is `50|100` on every one of the shared instances, so the "
      "budget-matched subset is **empty** (n matched = 0) and the audit's own verdict is "
      "**VERDICT UNTESTABLE**, not 'verdict holds'. The \\textsc{BCP-S} rung is `100|100` on all "
      "830 and its verdict does hold.\n")
    A("**What this does to the interpretation.** The arm *without* dense fusion had twice the "
      "step budget of the arm with it. A null measured that way is not evidence that dense fusion "
      "fails to help on Wikipedia; it is an uninterpretable comparison, and it is uninterpretable "
      "in the direction that flatters the ablation. Note also that \\textsc{Sieve} still scored "
      "*higher* on both datasets (+0.8, +1.3) on half the budget, so the point estimates are "
      "handicapped in dense fusion's disfavour and still positive. Every sentence in the paper "
      "that reads these two cells as a substantive dataset contrast -- 'the one rung that does "
      "*not* replicate', 'on either short-article Wikipedia corpus we cannot detect that it "
      "carries anything', 'Dense fusion is therefore the one component whose marginal necessity "
      "inside \\textsc{Sieve} is dataset-dependent, a verdict resting on all three datasets' -- "
      "is currently resting on a comparison the project's own audit calls untestable. The paper "
      "does disclose the caps (\\S\\ref{sec:setup-fairness}, Appendix A) and does say those four "
      "contrasts are not budget-matched; what it does not do is carry that qualification into the "
      "sentences that draw the dataset-dependence conclusion.\n")
    A("**The repair.** A re-run of \\textsc{Sieve} at cap 100 on both Wikipedia datasets is in "
      "flight into `runs/_budget100` (shard lists present, no `rows.jsonl` yet at the time this "
      "was written; partial data deliberately not used here). It will settle whether the two "
      "Wikipedia no-dense comparisons are nulls at a matched budget -- which is the only version "
      "of them that can be read at all. It will **not** give dense fusion new significant support "
      "even in the best case: a matched-budget Wikipedia rung that came out positive would still "
      "have to clear the same multiplicity ladder above, and a second and third significant rung "
      "at p around 0.01 would still die in any dataset-wide pool. What the repair can deliver is "
      "an interpretable statement of dataset-dependence, which the paper currently asserts on "
      "uninterpretable evidence.\n")

    # ---------------- verdict ----------------
    A("\n## Verdict\n")
    A("**PARTLY TRUE, and true in its most important part.**\n")
    A("True: dense fusion is named in the abstract, in a contribution bullet and in the method's "
      "three-component decomposition, and exactly one comparison in the entire paper supports it "
      "significantly -- the \\textsc{BrowseComp-Plus-Structured} no-dense-evidence rung, on two "
      "metrics of the same 830 instances, which is one cell and not two independent results. Both "
      "Wikipedia versions are null. Pooled with the rest of that dataset's own tests (m=43 on EM "
      "alone) it goes to adjusted p=0.18, and globally (m=127) to 0.44. Snippets survive both. "
      "The claim's central asymmetry is real and the paper should state it.\n")
    A("Overstated in three specifics. (i) 'Any correction wider than m=4' is wrong: the rung "
      "survives the paper's own stated m=6 variant (adj. 0.031), a pooled m=12 correction over the "
      "whole ablation family (adj. 0.0104, it is the family maximum) and m=14 (adj. 0.031). It "
      "dies when pooled with the paper's *null* results, not merely when the family grows -- the "
      "mechanism is that only four tests with p above 0.0104 may share its pool. (ii) 'ZERO "
      "significant support in the entire paper' is right about significance but skips the "
      "corroborating pattern: at \\textsc{Sieve}'s read level on \\textsc{BCP-S}, fusion is "
      "positive on all three query surfaces on both metrics. (iii) The two Wikipedia nulls the "
      "claim leans on are budget-confounded in the ablation's favour and are not usable evidence "
      "in either direction.\n")
    A("**What the paper should say about dense fusion.** Keep the component, change the claim. "
      "The supportable statement is: *dense fusion's contribution is established on one dataset, "
      "at one read interface, by one paired comparison (EM p=0.0104, judge p=0.0145 on the same "
      "830 instances), which survives correction within the ablation family it was declared in but "
      "not a correction pooled across that dataset's other tests; the two Wikipedia measurements "
      "of the same rung ran with the ablated arm at twice the step budget and are not "
      "interpretable; and the same fusion operation significantly hurts a plain-BM25 surface on "
      "both Wikipedia datasets under whole-document visit.* Concretely: (1) drop 'a verdict "
      "resting on all three datasets' from the Conclusion -- it rests on one; (2) attach the "
      "step-cap asymmetry to the two Wikipedia no-dense sentences wherever they are read as "
      "evidence, not only to Appendix A; (3) say in Finding 2 what the multiplicity budget "
      "actually is, i.e. that this rung's significance does not survive pooling with the rest of "
      "its own dataset, as the paper already does for the isolation block's dense-alone contrast; "
      "and (4) since the paper already says no component is independently justified and that the "
      "claim it defends is the full combination, the fix is a scope sentence, not a retraction.\n")
    A("\n**Even-handed footnote on standards.** A single Holm correction over all 127 tests is a "
      "standard almost no published paper meets, and applying it selectively to the one component "
      "a reviewer dislikes would be its own error. Reported symmetrically: at the global level "
      f"{g['n_survive']} of {g['m']} tests survive, including every snippet rung, the token "
      "results, the DCI family, the two Wikipedia hybrid contrasts and the cross-backbone results; "
      "the whole Finding 1 baseline family except MuSiQue EM does not, and neither does the "
      "paper's flagship \\textsc{BCP-S} accuracy comparison, which was already a declared null.\n")
    import re
    md = "\n".join(o) + "\n"
    # this is a markdown artifact, not a LaTeX include: strip the LaTeX-isms
    md = re.sub(r"\\textsc\{([^}]*)\}", r"\1", md)
    md = md.replace("$\\Delta$", "delta ").replace("$\\alpha=0.05$", "alpha=0.05")
    md = md.replace("$\\ge$", ">=").replace("$\\times$", "x")
    md = re.sub(r"\\S\\ref\{([^}]*)\}", r"(\1)", md)
    md = md.replace("\\%", "%").replace("\\textbf", "")
    OUT_MD.write_text(md)
    print(f"wrote {OUT_MD}")


def main():
    gate = sanity_gate()
    if not gate["passed"]:
        print("SANITY GATE FAILED", json.dumps(gate, indent=2))
        sys.exit(1)

    dci = compute_dci()
    # patch the census's DCI p-values with the live computation
    live = {"D1": dci.get(f"{BCP}|DCI"), "D2": dci.get(f"{HQA}|DCI"),
            "D3": dci.get(f"{MSQ}|DCI"), "D6": dci.get(f"{BCP}|BM25-filtered DCI")}
    for t in CENSUS:
        if t["id"] in live and isinstance(live[t["id"]], dict):
            t["p"] = live[t["id"]]["p"]
            t["n"] = live[t["id"]]["n"]

    contrasts = compute_contrasts()
    ladders, pool_sizes = build_ladder(CENSUS)

    # census accounting
    n_total = len(CENSUS)
    n_declared = sum(1 for t in CENSUS if t["status"] == "declared")
    n_stated = sum(1 for t in CENSUS if t["status"] == "stated-but-disclaimed")
    n_none = sum(1 for t in CENSUS if t["status"] == "none")
    fams = {}
    for t in CENSUS:
        if t["family"]:
            fams.setdefault(t["family"], dict(m=t["m"], status=t["status"], n=0))["n"] += 1

    data = dict(
        sanity_gate=gate,
        census=CENSUS,
        accounting=dict(
            total_tests=n_total,
            in_declared_family=n_declared,
            in_stated_but_disclaimed_family=n_stated,
            outside_any_family=n_none,
            outside_any_family_strict=n_none + n_stated,
            families=fams,
            pool_sizes=pool_sizes,
        ),
        dense_fusion_contrasts=contrasts["dense"],
        other_component_contrasts=contrasts["other"],
        dci_live=dci,
        ladders=ladders,
        **{"global": global_correction(CENSUS)},
        blockers=dict(
            bcp_em_ge_dense=blockers(CENSUS, 0.0104,
                                    [t["id"] for t in CENSUS if t["p"] is not None
                                     and t["ds"] == BCP and t["metric"] == "EM"]),
            all_ge_dense=blockers(CENSUS, 0.0104,
                                  [t["id"] for t in CENSUS if t["p"] is not None]),
        ),
        holm_survival_thresholds=dict(
            dense_em_p=0.0104, dense_em_kmax=k_max(0.0104),
            dense_judge_p=0.0145, dense_judge_kmax=k_max(0.0145),
            snippets_bcp_p=4.31e-10, snippets_bcp_kmax=k_max(4.31e-10),
            snippets_hqa_p=1.39e-8, snippets_hqa_kmax=k_max(1.39e-8),
            snippets_msq_p=7.73e-8, snippets_msq_kmax=k_max(7.73e-8),
        ),
    )
    OUT_JSON.write_text(json.dumps(data, indent=1, default=str))
    print(f"wrote {OUT_JSON}")
    write_md(data)
    print(json.dumps(data["accounting"], indent=1))
    print("\nDENSE-FUSION CONTRASTS")
    for r in contrasts["dense"]:
        if r.get("em"):
            j = f", judge d={r['judge']['delta']:+.1f} p={r['judge']['p']:.4g}" if r.get("judge") else ""
            print(f"  {r['dataset'][:12]:12s} {r['label'][:34]:34s} n={r['em']['n']:5d} "
                  f"d={r['em']['delta']:+5.1f} p={r['em']['p']:.4g}{j}   [{r['in_paper']}]")
        else:
            print(f"  {r['dataset']} {r['label']}: {r.get('status')}")
    print("\nOTHER COMPONENTS")
    for r in contrasts["other"]:
        if r.get("em"):
            print(f"  {r['dataset'][:12]:12s} {r['label'][:38]:38s} n={r['em']['n']:5d} "
                  f"d={r['em']['delta']:+5.1f} p={r['em']['p']:.4g}  ({r['kind']})")
    print("\nLADDERS")
    for k, rows in ladders.items():
        print(f" {k}")
        for r in rows:
            print(f"   {r['level'][:56]:56s} m={r['m']:4d} adj={r['adjusted']:.4g} "
                  f"{'SURVIVES' if r['significant'] else 'DIES'}")
    return data


if __name__ == "__main__":
    main()
