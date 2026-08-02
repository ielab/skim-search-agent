#!/usr/bin/env python
"""ARR review response: the six quantities three reviewers asked for and the paper does not compute.

Read-only over `runs/`, `latex/`, `latex_acl8/`. Writes ONLY
`analysis/arr_review_response{.md,_data.json}` (plus the shared, disposable
`analysis/.compare_cache/` metric cache that `scripts/compare_cells.py` owns).

METHOD PARITY IS STRUCTURAL, NOT RETYPED. Nothing statistical is reimplemented here:
  scripts.compare_cells      : cell_dir, cell_rows, metrics, _compute_cell (the cached,
                               unit-test-proven equivalent of `metrics(cell_rows(...))` --
                               tests/test_compare_cells_incremental.py asserts the two agree;
                               PART 0 below re-proves it at runtime on a real cell before any
                               number is trusted), mcnemar_p (exact two-sided McNemar via
                               scipy.stats.binomtest on the discordant pairs), pct, load_qrels,
                               load_judge_cache, gold_doc_recall (via metrics/_row_intrinsic)
  evaluation.metrics         : answer_em (transitively, through metrics/_row_intrinsic)
  scripts.force_answer_backfill : load_rows_with_recovery / needs_recovery (the recovery overlay
                               and the `empty` predicate), transitively
  analysis.ablation_deltas   : JUDGE_COVERAGE_MIN (the project's >=90% both-sides judge gate)
  analysis.recall_inversion_test : paired_recall, _is_binary_indicator (exact McNemar on the
                               per-instance BINARY gold-doc-recall indicator)
  analysis.step_budget_audit : per_row_caps (the CONFIGURED `max_steps` stamped per row -- never
                               an observed step count, which is a post-treatment outcome)
  analysis.multiplicity_census : CENSUS (the 131-test census), holm (Holm-Bonferroni step-down)

MANDATORY SANITY GATE (PART 1, runs FIRST, one target per dataset family; a failure aborts with
a nonzero exit and NO new numbers):
  (i)  Sieve vs "No dense evidence", browsecomp_plus_structured : +4.70 EM, p=0.0104
  (ii) Sieve vs hybrid control, hotpotqa_structured at matched cap 100 : +5.84 EM, p=2.84e-29

Run: PYTHONPATH=. envs/bin/python analysis/arr_review_response.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    _compute_cell, cell_dir, cell_rows, load_judge_cache, load_qrels, mcnemar_p, metrics, pct,
)
from analysis.ablation_deltas import JUDGE_COVERAGE_MIN  # noqa: E402
from analysis.recall_inversion_test import _is_binary_indicator, paired_recall  # noqa: E402
from analysis.step_budget_audit import per_row_caps  # noqa: E402
from analysis.multiplicity_census import CENSUS, holm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "analysis" / "arr_review_response_data.json"
MD_PATH = ROOT / "analysis" / "arr_review_response.md"

BCP = "browsecomp_plus_structured"
HQA = "hotpotqa_structured"
MSQ = "musique_structured"
DATASETS = [BCP, HQA, MSQ]
PRETTY = {BCP: "BrowseComp-Plus-Structured", HQA: "HotpotQA", MSQ: "MuSiQue"}

# (tier, condition) -- copied verbatim from scripts/compare_cells.py REGISTRY / the analysis
# scripts that already use them, so cell identity is structural rather than retyped guesswork.
SIEVE50 = ("_headline_validation", "agent_research_bql_dense_snip")   # the tabulated arm
SIEVE100 = ("_budget100", "agent_research_bql_dense_snip")            # wiki repair arm, cap 100
BASE_K5 = ("_visit_uncapped", "agent_research_bm25")                  # SERP bm25 [BASELINE], k=5
BASE_K10 = ("_visit_uncapped_k10", "agent_research_bm25")             # SERP bm25 k=10
HYBRID = ("_headline_validation", "agent_research_hybrid_fetch_snip")  # k=10
SPARSE = ("_headline_validation", "agent_research_bm25_fetch_snip")   # k=10
NODENSE = ("_headline_validation", "agent_research_snip")             # k=5
DENSESNIP = ("_headline_validation", "agent_research_dense_fetch")    # k=10
NOSNIP = ("_headline_validation", "agent_research_bql_dense_fetch")   # k=5

LABEL = {
    SIEVE50: "Sieve @cap50 (bql+dense+snip, k=5)",
    SIEVE100: "Sieve @cap100 (bql+dense+snip, k=5)",
    BASE_K5: "SERP bm25 [BASELINE] (k=5)",
    BASE_K10: "SERP bm25 k=10",
    HYBRID: "hybrid control (RRF + snip + fetch, k=10)",
    SPARSE: "sparse only, same interface (bm25+snip fetch, k=10)",
    NODENSE: "no dense evidence (bql+snip fetch, k=5)",
    DENSESNIP: "dense only, same interface (dense+snip fetch, k=10)",
    NOSNIP: "no snippets (bql+dense fetch, k=5)",
}

_CACHE: dict = {}
_CAPCACHE: dict = {}


def cell(dataset: str, key, qrels) -> dict | None:
    """Per-instance metrics dict for one cell, or None if rows.jsonl is absent. Uses
    `scripts.compare_cells._compute_cell`, the cached primitive `compare_cells.main()` itself
    dispatches to, proven equal to `metrics(cell_rows(...))` both by unit test and, at runtime,
    by PART 0 below."""
    ck = (dataset, key)
    if ck in _CACHE:
        return _CACHE[ck]
    tier, cond = key
    exists, m = _compute_cell(str(cell_dir(tier, dataset, cond)), dataset, qrels, True)
    _CACHE[ck] = m if exists and m else None
    return _CACHE[ck]


def caps(dataset: str, key) -> dict:
    """{configured max_steps -> count} for a cell, from the per-row `max_steps` stamp."""
    tier, cond = key
    d = per_row_caps(dataset, tier, cond, _CAPCACHE)
    out: dict = {}
    for v in d.values():
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def cap_ids(dataset: str, key, cap: int) -> set:
    """The instance ids in a cell whose CONFIGURED cap equals `cap` -- used to carve the
    budget-matched subset out of MuSiQue's internally-mixed baseline cell (2209 episodes at 50,
    200 at 100)."""
    tier, cond = key
    return {i for i, v in per_row_caps(dataset, tier, cond, _CAPCACHE).items() if v == cap}


def restrict(m: dict, ids: set) -> dict:
    return {i: v for i, v in m.items() if i in ids}


def coverage(m: dict) -> float:
    return (sum(1 for v in m.values() if v["judge"] is not None) / len(m)) if m else 0.0


def paired_em(a: dict, b: dict) -> dict:
    """A - B on the shared instance set; exact McNemar on the discordant pairs.
    b = B-correct/A-wrong, c = A-correct/B-wrong (mcnemar_p is symmetric in the two)."""
    mut = set(a) & set(b)
    nb = sum(1 for i in mut if b[i]["em"] and not a[i]["em"])
    nc = sum(1 for i in mut if a[i]["em"] and not b[i]["em"])
    ea, eb = pct([a[i]["em"] for i in mut]), pct([b[i]["em"] for i in mut])
    return dict(n=len(mut), em_a=ea, em_b=eb, delta=ea - eb, b=nb, c=nc, p=mcnemar_p(nb, nc))


def paired_judge(a: dict, b: dict) -> dict | None:
    """Judge version, gated at the project's >=90% coverage rule on BOTH sides."""
    ca, cb = coverage(a), coverage(b)
    if ca < JUDGE_COVERAGE_MIN or cb < JUDGE_COVERAGE_MIN:
        return dict(status="insufficient judge coverage", cov_a=ca, cov_b=cb)
    mut = [i for i in (set(a) & set(b))
           if a[i]["judge"] is not None and b[i]["judge"] is not None]
    if not mut:
        return dict(status="no judged overlap", cov_a=ca, cov_b=cb)
    nb = sum(1 for i in mut if b[i]["judge"] and not a[i]["judge"])
    nc = sum(1 for i in mut if a[i]["judge"] and not b[i]["judge"])
    ja, jb = pct([a[i]["judge"] for i in mut]), pct([b[i]["judge"] for i in mut])
    return dict(status="computed", n=len(mut), judge_a=ja, judge_b=jb, delta=ja - jb,
                b=nb, c=nc, p=mcnemar_p(nb, nc), cov_a=ca, cov_b=cb)


def recall_pair(a: dict, b: dict) -> dict:
    n, ra, rb, delta, nb, nc, p = paired_recall(a, b)
    return dict(n=n, recall_a=ra, recall_b=rb, delta=delta, b=nb, c=nc, p=p,
                binary=bool(_is_binary_indicator(a) and _is_binary_indicator(b)))


# ==============================================================================================
# PART 0 -- runtime proof that the cached loader equals the canonical metrics() path
# ==============================================================================================

def part0_parity(qrels_by_ds: dict) -> dict:
    """`_compute_cell` is documented and unit-tested as equal to `metrics(cell_rows(...))`.
    Re-prove it here on a REAL cell (BrowseComp-Plus-Structured's baseline, 830 rows) so the
    speed-ups below rest on a measurement, not on a docstring."""
    ds, key = BCP, BASE_K5
    tier, cond = key
    rows = cell_rows(tier, ds, cond)
    canonical = metrics(rows, qrels_by_ds[ds], ds, load_judge_cache(cell_dir(tier, ds, cond)))
    cached = cell(ds, key, qrels_by_ds[ds])
    fields = ("em", "judge", "recall", "empty", "lenient", "surfaced")
    same_ids = set(canonical) == set(cached)
    mismatches = [i for i in canonical
                  if any(canonical[i][f] != cached[i][f] for f in fields)]
    return dict(cell=f"{ds} / {tier} / {cond}", n=len(canonical), same_id_set=same_ids,
                n_field_mismatches=len(mismatches),
                passed=bool(same_ids and not mismatches),
                fields_compared=list(fields))


# ==============================================================================================
# PART 1 -- MANDATORY SANITY GATE (one target per dataset family)
# ==============================================================================================

GATES = [
    dict(key="i", label='Sieve vs "No dense evidence", browsecomp_plus_structured (both cap 100)',
         dataset=BCP, a=SIEVE50, b=NODENSE, exp_delta=4.70, exp_p=0.0104, p_tol_rel=0.02),
    dict(key="ii", label="Sieve vs hybrid control, hotpotqa_structured at matched cap 100",
         dataset=HQA, a=SIEVE100, b=HYBRID, exp_delta=5.84, exp_p=2.8e-29, p_tol_rel=0.05),
]


def part1_gate(qrels_by_ds: dict) -> dict:
    out = {"gates": [], "passed": True}
    for g in GATES:
        ds = g["dataset"]
        a = cell(ds, g["a"], qrels_by_ds[ds])
        b = cell(ds, g["b"], qrels_by_ds[ds])
        if a is None or b is None:
            out["gates"].append(dict(g, error="cell missing", passed=False))
            out["passed"] = False
            continue
        r = paired_em(a, b)
        ok_d = abs(r["delta"] - g["exp_delta"]) <= 0.01
        ok_p = abs(r["p"] - g["exp_p"]) <= g["p_tol_rel"] * g["exp_p"]
        rec = dict(key=g["key"], label=g["label"], dataset=ds,
                   arm_a=LABEL[g["a"]], arm_b=LABEL[g["b"]],
                   caps_a=caps(ds, g["a"]), caps_b=caps(ds, g["b"]),
                   n=r["n"], em_a=r["em_a"], em_b=r["em_b"], delta=r["delta"],
                   b=r["b"], c=r["c"], p=r["p"],
                   expected_delta=g["exp_delta"], expected_p=g["exp_p"],
                   delta_ok=ok_d, p_ok=ok_p, passed=bool(ok_d and ok_p))
        out["gates"].append(rec)
        out["passed"] = out["passed"] and rec["passed"]
    return out


# ==============================================================================================
# TASK 1 -- the coaching confound's EM-vs-judge signature (Reviewer 3)
# ==============================================================================================

# The Sieve-vs-baseline contrast, per dataset, with the arm choice pinned to budget parity:
#   BCP-S  : Sieve is only run at cap 100 there; the baseline is cap 100 too.
#   Wiki   : the paper's Finding 1 pairs the tabulated cap-50 Sieve arm against the cap-50
#            baseline. The cap-100 repair arm (`_budget100`) carries NO judge cache at all, so
#            a judge cross-tab is only computable on the cap-50 arm; caps are reported per pair.
T1_PAIRS = {BCP: (SIEVE50, BASE_K5), HQA: (SIEVE50, BASE_K5), MSQ: (SIEVE50, BASE_K5)}


def shortcircuit_audit(cells_by_name: dict) -> dict:
    """(a) The judge short-circuit, verified from the DATA as well as from the code.

    `scripts/judge_cells.py::classify_row` records `judge_correct=True` with
    `method="em_shortcircuit"` for every row where `evaluation.metrics.answer_em` is 1, with no
    LLM call; `evaluation/llm_judge.py::judge_answer_detail` additionally short-circuits its own
    normalized-exact matches. Both imply EM-correct => judge-correct. The falsifiable
    consequence is that NO judged instance anywhere can be (em=True, judge=False); that is what
    this counts, over every judged cell used in this report."""
    viol = {}
    method_counts = {}
    for name, m in cells_by_name.items():
        judged = [v for v in m.values() if v["judge"] is not None]
        if not judged:
            continue
        n_viol = sum(1 for v in m.values()
                     if v["judge"] is not None and v["em"] and not v["judge"])
        n_em_judged = sum(1 for v in m.values() if v["judge"] is not None and v["em"])
        viol[name] = dict(n_judged=len(judged), n_em_and_judged=n_em_judged,
                          n_em_true_judge_false=n_viol)
    return dict(per_cell=viol,
                total_violations=sum(v["n_em_true_judge_false"] for v in viol.values()),
                method_counts=method_counts)


def judge_crosstab(a: dict, b: dict) -> dict:
    """(b)-(d). Full 2x2x2 cross-tabulation of {EM, judge} x {baseline(B), Sieve(A)} on the
    instances where BOTH arms carry a judge verdict, plus the decomposition of the paired EM
    delta by the pair's joint judge status.

    Because EM-correct implies judge-correct (verified above), only three of the four judge
    partitions can carry EM movement:
      TT (judge-correct in both arms) : both directions possible -- R3's "already correct" set
      FT (judge 0 -> 1, a content gain) : only EM gains possible (baseline judge False => baseline
                                          EM False)
      TF (judge 1 -> 0, a content loss) : only EM losses possible
      FF (judge-wrong in both)          : no EM movement possible at all
    and    delta_EM = [ (b_TT - c_TT) + b_FT - c_TF ] / n."""
    mut = sorted(i for i in (set(a) & set(b))
                 if a[i]["judge"] is not None and b[i]["judge"] is not None)
    n = len(mut)
    tab: dict = {}
    for i in mut:
        k = (bool(b[i]["em"]), bool(a[i]["em"]), bool(b[i]["judge"]), bool(a[i]["judge"]))
        tab[k] = tab.get(k, 0) + 1

    def count(pred):
        return sum(v for k, v in tab.items() if pred(*k))

    # EM movement
    b_em = count(lambda eb, ea, jb, ja: ea and not eb)     # EM gained by Sieve
    c_em = count(lambda eb, ea, jb, ja: eb and not ea)     # EM lost by Sieve
    # judge movement
    b_j = count(lambda eb, ea, jb, ja: ja and not jb)
    c_j = count(lambda eb, ea, jb, ja: jb and not ja)

    # R3's "already correct" set and its complement, within the EM-gain set
    gain_already = count(lambda eb, ea, jb, ja: ea and not eb and jb and ja)
    gain_content = count(lambda eb, ea, jb, ja: ea and not eb and not jb)      # => ja True
    loss_already = count(lambda eb, ea, jb, ja: eb and not ea and jb and ja)
    loss_content = count(lambda eb, ea, jb, ja: eb and not ea and not ja)      # => jb True

    net_em = b_em - c_em
    net_tt = gain_already - loss_already
    net_ft = gain_content
    net_tf = -loss_content

    def share(x):
        return (100.0 * x / net_em) if net_em else float("nan")

    # Is each half of the decomposition itself distinguishable from zero? Same exact McNemar,
    # applied to the discordant pairs of that half only (the halves partition the discordant
    # pairs, so the two tests are on disjoint pair sets).
    p_tt = mcnemar_p(loss_already, gain_already)
    p_flip = mcnemar_p(loss_content, gain_content)

    return dict(
        n=n,
        em_a=pct([a[i]["em"] for i in mut]), em_b=pct([b[i]["em"] for i in mut]),
        em_delta=pct([a[i]["em"] for i in mut]) - pct([b[i]["em"] for i in mut]),
        em_b_disc=b_em, em_c_disc=c_em, em_p=mcnemar_p(c_em, b_em),
        judge_a=pct([a[i]["judge"] for i in mut]), judge_b=pct([b[i]["judge"] for i in mut]),
        judge_delta=pct([a[i]["judge"] for i in mut]) - pct([b[i]["judge"] for i in mut]),
        judge_b_disc=b_j, judge_c_disc=c_j, judge_p=mcnemar_p(c_j, b_j),
        gain_already_judge_correct=gain_already,
        gain_judge_flip=gain_content,
        loss_already_judge_correct=loss_already,
        loss_judge_flip=loss_content,
        net_em_pairs=net_em,
        net_TT_pairs=net_tt, net_FT_pairs=net_ft, net_TF_pairs=net_tf,
        contrib_TT_pts=100.0 * net_tt / n, contrib_FT_pts=100.0 * net_ft / n,
        contrib_TF_pts=100.0 * net_tf / n,
        contrib_flip_pts=100.0 * (net_ft + net_tf) / n,
        p_TT_component=p_tt, p_flip_component=p_flip,
        share_of_em_gain_already_judge_correct_gross=(
            100.0 * gain_already / b_em if b_em else float("nan")),
        share_of_net_em_gain_already_judge_correct=share(net_tt),
        share_of_net_em_gain_from_judge_flips=share(net_ft + net_tf),
        crosstab={f"em_base={k[0]},em_sieve={k[1]},judge_base={k[2]},judge_sieve={k[3]}": v
                  for k, v in sorted(tab.items())},
    )


def task1(qrels_by_ds: dict) -> dict:
    out = {"pairs": {}}
    judged_cells = {}
    for ds in DATASETS:
        ka, kb = T1_PAIRS[ds]
        a, b = cell(ds, ka, qrels_by_ds[ds]), cell(ds, kb, qrels_by_ds[ds])
        judged_cells[f"{ds} / {LABEL[ka]}"] = a
        judged_cells[f"{ds} / {LABEL[kb]}"] = b
        cov_a, cov_b = coverage(a), coverage(b)
        rec = dict(dataset=ds, arm_sieve=LABEL[ka], arm_baseline=LABEL[kb],
                   caps_sieve=caps(ds, ka), caps_baseline=caps(ds, kb),
                   judge_coverage_sieve=cov_a, judge_coverage_baseline=cov_b,
                   judge_coverage_min=JUDGE_COVERAGE_MIN,
                   adequate_coverage=bool(cov_a >= JUDGE_COVERAGE_MIN
                                          and cov_b >= JUDGE_COVERAGE_MIN))
        rec["em_all_instances"] = paired_em(a, b)
        if rec["adequate_coverage"]:
            rec["crosstab"] = judge_crosstab(a, b)
        # MuSiQue's baseline cell is internally mixed (2209 episodes at cap 50, 200 at cap 100)
        # while the Sieve arm is uniformly 50: repeat on the cap-50 subset, where the pair is
        # exactly budget-matched.
        if len(rec["caps_baseline"]) > 1:
            ids = cap_ids(ds, kb, 50) & set(a) & set(b)
            ra, rb = restrict(a, ids), restrict(b, ids)
            rec["cap50_matched_subset"] = dict(
                n=len(ids), em=paired_em(ra, rb),
                crosstab=judge_crosstab(ra, rb) if rec["adequate_coverage"] else None)
        out["pairs"][ds] = rec
    # also cross-tab the OTHER fully-judged BCP-S headline contrasts, so the signature is not
    # read off a single pair on the one dataset where every cell is judged
    extra = {}
    for name, (ka, kb) in {"Sieve vs sparse only, same interface": (SIEVE50, SPARSE),
                           "Sieve vs no dense evidence": (SIEVE50, NODENSE),
                           "Sieve vs hybrid control": (SIEVE50, HYBRID)}.items():
        a, b = cell(BCP, ka, qrels_by_ds[BCP]), cell(BCP, kb, qrels_by_ds[BCP])
        judged_cells[f"{BCP} / {LABEL[kb]}"] = b
        if min(coverage(a), coverage(b)) >= JUDGE_COVERAGE_MIN:
            extra[name] = judge_crosstab(a, b)
    out["bcp_extra_contrasts"] = extra
    out["shortcircuit_audit"] = shortcircuit_audit(judged_cells)
    return out


# ==============================================================================================
# TASK 2 -- signing the pool-size bound (Reviewers 1 and 2)
# ==============================================================================================

# The paper's headline "+6.0/+4.1/+4.8 over plain BM25 at a matched read interface": Sieve (k=5)
# against `Sparse only, same interface` (k=10). On the Wikipedia datasets the control runs at cap
# 100, so the paper reads it against the cap-100 Sieve arm; on BCP-S both arms are cap 100.
T2_HEADLINE = {BCP: (SIEVE50, SPARSE), HQA: (SIEVE100, SPARSE), MSQ: (SIEVE100, SPARSE)}


def task2(qrels_by_ds: dict) -> dict:
    out = {"pool_effect": {}, "headline": {}, "bound": {}}
    for ds in DATASETS:
        qr = qrels_by_ds[ds]
        k10, k5 = cell(ds, BASE_K10, qr), cell(ds, BASE_K5, qr)
        em = paired_em(k10, k5)                       # k=10 minus k=5
        rec_ = recall_pair(k10, k5)
        # empty-answer / recovery parity between the two pool arms (a live confound: the two
        # arms did not both receive a forced-answer backfill pass)
        mut = sorted(set(k10) & set(k5))
        empty10 = pct([k10[i]["empty"] for i in mut])
        empty5 = pct([k5[i]["empty"] for i in mut])
        rec10 = sum(k10[i]["recovered"] for i in mut)
        rec5 = sum(k5[i]["recovered"] for i in mut)
        both_ans = [i for i in mut if not k10[i]["empty"] and not k5[i]["empty"]]
        nb = sum(1 for i in both_ans if k5[i]["em"] and not k10[i]["em"])
        nc = sum(1 for i in both_ans if k10[i]["em"] and not k5[i]["em"])
        out["pool_effect"][ds] = dict(
            arm_k10=LABEL[BASE_K10], arm_k5=LABEL[BASE_K5],
            caps_k10=caps(ds, BASE_K10), caps_k5=caps(ds, BASE_K5),
            em=em, recall=rec_,
            empty_pct_k10=empty10, empty_pct_k5=empty5,
            n_recovered_k10=rec10, n_recovered_k5=rec5,
            diagnostic_both_answered=dict(
                n=len(both_ans),
                em_k10=pct([k10[i]["em"] for i in both_ans]),
                em_k5=pct([k5[i]["em"] for i in both_ans]),
                delta=pct([k10[i]["em"] for i in both_ans]) - pct([k5[i]["em"] for i in both_ans]),
                b=nb, c=nc, p=mcnemar_p(nb, nc)),
        )
        if len(caps(ds, BASE_K5)) > 1:      # MuSiQue's mixed baseline: budget-matched subset
            ids = cap_ids(ds, BASE_K5, 50) & set(k10) & set(k5)
            out["pool_effect"][ds]["cap50_matched_subset"] = dict(
                n=len(ids), em=paired_em(restrict(k10, ids), restrict(k5, ids)),
                recall=recall_pair(restrict(k10, ids), restrict(k5, ids)))
        ka, kb = T2_HEADLINE[ds]
        a, b = cell(ds, ka, qr), cell(ds, kb, qr)
        h_em = paired_em(a, b)
        h_rec = recall_pair(a, b)
        out["headline"][ds] = dict(arm_sieve=LABEL[ka], arm_control=LABEL[kb],
                                   caps_sieve=caps(ds, ka), caps_control=caps(ds, kb),
                                   em=h_em, recall=h_rec)
        p_em = em["delta"]          # what k=10 alone is worth on EM (signed)
        p_rc = rec_["delta"]        # ... and on gold recall
        out["bound"][ds] = dict(
            headline_em_delta=h_em["delta"], pool_em_effect=p_em,
            residual_em=h_em["delta"] + p_em,
            pool_em_significant=bool(em["p"] < 0.05),
            headline_recall_delta=h_rec["delta"], pool_recall_effect=p_rc,
            residual_recall=h_rec["delta"] + p_rc,
            pool_recall_significant=bool(rec_["p"] < 0.05),
        )
    # the recall inversion: Sieve vs the hybrid control (also k=10)
    inv = {}
    for ds in DATASETS:
        qr = qrels_by_ds[ds]
        ka = SIEVE50 if ds == BCP else SIEVE100
        a, b = cell(ds, ka, qr), cell(ds, HYBRID, qr)
        r = recall_pair(a, b)
        pool_rc = out["pool_effect"][ds]["recall"]["delta"]
        inv[ds] = dict(arm_sieve=LABEL[ka], arm_control=LABEL[HYBRID],
                       caps_sieve=caps(ds, ka), caps_control=caps(ds, HYBRID),
                       recall=r, pool_recall_effect=pool_rc,
                       residual_recall=r["delta"] + pool_rc)
    out["recall_inversion_bound"] = inv
    return out


# ==============================================================================================
# TASK 3 -- the recall inversion against a pool-MATCHED comparator (Reviewer 1)
# ==============================================================================================

def task3(qrels_by_ds: dict) -> dict:
    out = {}
    for ds in DATASETS:
        qr = qrels_by_ds[ds]
        rows = []
        # vs the k=5 (pool-MATCHED) baseline and the k=10 baseline: both baselines run at the
        # cap the tabulated Sieve arm runs at on that dataset, so use SIEVE50 there; the hybrid
        # control runs at cap 100 on the wiki datasets, so that pair reads the cap-100 arm.
        for comparator, key, sieve_key in (
                ("SERP bm25 [BASELINE] k=5 (POOL-MATCHED)", BASE_K5, SIEVE50),
                ("SERP bm25 k=10", BASE_K10, SIEVE50),
                ("hybrid control (k=10)", HYBRID, SIEVE50 if ds == BCP else SIEVE100),
                ("sparse only, same interface (k=10)", SPARSE, SIEVE50 if ds == BCP else SIEVE100),
                ("dense only, same interface (k=10)", DENSESNIP,
                 SIEVE50 if ds == BCP else SIEVE100),
                ("no dense evidence (k=5, POOL-MATCHED)", NODENSE,
                 SIEVE50 if ds == BCP else SIEVE100),
        ):
            a, b = cell(ds, sieve_key, qr), cell(ds, key, qr)
            if a is None or b is None:
                continue
            r = recall_pair(a, b)
            e = paired_em(a, b)
            rows.append(dict(comparator=comparator, sieve_arm=LABEL[sieve_key],
                             caps_sieve=caps(ds, sieve_key), caps_comparator=caps(ds, key),
                             pool_matched=key in (BASE_K5, NODENSE),
                             recall=r, em=e))
        out[ds] = rows
    return out


# ==============================================================================================
# TASK 4 -- no-answer rates (Reviewer 2)
# ==============================================================================================

T4_CELLS = [BASE_K5, BASE_K10, SIEVE50, SIEVE100, SPARSE, HYBRID, DENSESNIP, NODENSE, NOSNIP]


def task4(qrels_by_ds: dict) -> dict:
    out = {"rates": {}, "conditional_em": {}}
    for ds in DATASETS:
        qr = qrels_by_ds[ds]
        rows = []
        for key in T4_CELLS:
            m = cell(ds, key, qr)
            if m is None:
                continue
            vals = list(m.values())
            rows.append(dict(cell=LABEL[key], tier=key[0], cond=key[1], n=len(vals),
                             caps=caps(ds, key),
                             empty_pct=pct([v["empty"] for v in vals]),
                             n_recovered=sum(v["recovered"] for v in vals),
                             em_pct=pct([v["em"] for v in vals])))
        out["rates"][ds] = rows

        # DIAGNOSTIC ONLY (post-treatment conditioning): EM among instances that produced an
        # answer. Reported two ways: per-arm on that arm's own answered subset (unpaired), and
        # paired on the instances BOTH arms answered.
        ka, kb = T1_PAIRS[ds]
        a, b = cell(ds, ka, qr), cell(ds, kb, qr)
        mut = sorted(set(a) & set(b))
        a_ans = [i for i in mut if not a[i]["empty"]]
        b_ans = [i for i in mut if not b[i]["empty"]]
        both = [i for i in mut if not a[i]["empty"] and not b[i]["empty"]]
        nb = sum(1 for i in both if b[i]["em"] and not a[i]["em"])
        nc = sum(1 for i in both if a[i]["em"] and not b[i]["em"])
        out["conditional_em"][ds] = dict(
            arm_sieve=LABEL[ka], arm_baseline=LABEL[kb], n_shared=len(mut),
            unconditional=paired_em(a, b),
            n_answered_sieve=len(a_ans), n_answered_baseline=len(b_ans),
            em_given_answer_sieve=pct([a[i]["em"] for i in a_ans]),
            em_given_answer_baseline=pct([b[i]["em"] for i in b_ans]),
            paired_both_answered=dict(
                n=len(both), em_sieve=pct([a[i]["em"] for i in both]),
                em_baseline=pct([b[i]["em"] for i in both]),
                delta=pct([a[i]["em"] for i in both]) - pct([b[i]["em"] for i in both]),
                b=nb, c=nc, p=mcnemar_p(nb, nc)),
        )
    return out


# ==============================================================================================
# TASK 5 -- apply the wide correction SYMMETRICALLY (Reviewer 3)
# ==============================================================================================

BCP_DISPLAY = "BrowseComp-Plus-Structured"


# The paper's published pooled correction (p~=0.18 for the no-dense rung) was produced by
# `analysis/multiplicity_census.py`, whose main() patches four DCI rows with a live recomputation
# before correcting. The persisted artifact therefore IS the pool the paper used; the module-level
# CENSUS literal is the same list with the pre-patch DCI p-values. Use the artifact, and record
# the difference rather than silently choosing.
CENSUS_JSON = ROOT / "analysis" / "multiplicity_census_data.json"

# The claims the paper makes on BrowseComp-Plus-Structured that a reader would call headline,
# id -> where the paper states it.
BCP_HEADLINES = {
    "B1": "Finding 1 flagship: Sieve vs BM25 search+visit (EM) -- already reported as a null",
    "B2": "Finding 1 flagship: Sieve vs BM25 search+visit (judge) -- already reported as a null",
    "A6": "abstract-leading '+6.0 EM over plain BM25 at a matched read interface'",
    "J2": "'+6.0 judge points over plain BM25 at a matched read interface'",
    "A1": "Finding 2 single-axis: snippets are necessary (EM)",
    "J5": "Finding 2 single-axis: snippets are necessary (judge)",
    "A2": "Finding 2 single-axis: dense fusion is necessary (EM) -- the one the paper DOES correct",
    "J1": "Finding 2 single-axis: dense fusion is necessary (judge)",
    "D1": "Sieve vs DCI (EM)",
    "D4": "Sieve vs DCI (judge)",
    "I7": "Sieve vs the matched hybrid control (EM) -- reported as directional",
    "I4": "dense fusion alone is worse than plain BM25 (EM)",
    "X1": "cross-backbone replication (Qwen): Sieve vs baseline (EM)",
}


def load_census() -> tuple:
    """(census, provenance) -- the persisted census artifact if present, else the literal."""
    if CENSUS_JSON.exists():
        c = json.loads(CENSUS_JSON.read_text())["census"]
        lit = {t["id"]: t["p"] for t in CENSUS}
        diffs = [dict(id=t["id"], artifact_p=t["p"], literal_p=lit.get(t["id"]))
                 for t in c if t["p"] is not None and lit.get(t["id"]) is not None
                 and abs(t["p"] - lit[t["id"]]) > 1e-9 * max(abs(t["p"]), abs(lit[t["id"]]))]
        return c, dict(source=str(CENSUS_JSON.relative_to(ROOT)),
                       note="module-level CENSUS literal differs on these ids (DCI rows are "
                            "live-patched by multiplicity_census.main() before correcting)",
                       differences=diffs)
    return CENSUS, dict(source="analysis.multiplicity_census.CENSUS (literal)", differences=[])


def task5() -> dict:
    """Holm-Bonferroni inside the SAME pooled 43-test BrowseComp-Plus-Structured exact-match
    family the paper uses to retire the no-dense-evidence rung, applied to every test in it --
    including every headline claim the paper makes on that dataset."""
    census, prov = load_census()
    tests = [t for t in census if t["p"] is not None and t["ds"] == BCP_DISPLAY]
    em = [t for t in tests if t["metric"] == "EM"]
    acc = [t for t in tests if t["metric"] in ("EM", "judge")]
    all_bcp = tests

    def adjust(pool):
        adj = holm([t["p"] for t in pool])
        return [dict(id=t["id"], grp=t["grp"], cmp=t["cmp"], metric=t["metric"], n=t["n"],
                     raw=t["p"], adjusted=a, sig_raw=t["p"] < 0.05, sig_adj=a < 0.05,
                     family=t.get("family"), status=t.get("status"))
                for t, a in zip(pool, adj)]

    em_rows = sorted(adjust(em), key=lambda r: r["raw"])
    acc_rows = sorted(adjust(acc), key=lambda r: r["raw"])
    all_rows = sorted(adjust(all_bcp), key=lambda r: r["raw"])

    # the specific verification R3 asks for
    def find(rows, tid):
        return next((r for r in rows if r["id"] == tid), None)

    verify = dict(
        headline_plain_bm25_A6=find(em_rows, "A6"),
        no_dense_rung_A2=find(em_rows, "A2"),
        flagship_baseline_B1=find(em_rows, "B1"),
        m_em=len(em_rows), m_acc=len(acc_rows), m_all=len(all_rows),
    )
    # per-headline status, EM claims corrected in the m=43 EM pool and judge claims in the
    # m=58 accuracy pool (the two pools the paper itself names)
    headlines = []
    for tid, what in BCP_HEADLINES.items():
        r_em, r_acc = find(em_rows, tid), find(acc_rows, tid)
        r = r_em or r_acc
        if r is None:
            continue
        headlines.append(dict(id=tid, claim=what, metric=r["metric"], raw=r["raw"],
                              adj_em43=r_em["adjusted"] if r_em else None,
                              adj_acc58=r_acc["adjusted"] if r_acc else None,
                              sig_raw=r["sig_raw"],
                              sig_pooled=bool((r_em or r_acc)["sig_adj"])))
    headlines.sort(key=lambda r: r["raw"])
    return dict(census_provenance=prov, em_pool=em_rows, accuracy_pool=acc_rows,
                all_bcp_pool=all_rows, verify=verify, headlines=headlines,
                n_survivors_em=sum(1 for r in em_rows if r["sig_adj"]),
                n_sig_raw_em=sum(1 for r in em_rows if r["sig_raw"]))


# ==============================================================================================
# TASK 6 -- two typos (Reviewer 3)
# ==============================================================================================

def task6() -> dict:
    tt = json.loads((ROOT / "analysis" / "paired_ttests_data.json").read_text())
    sw = []
    for e in tt:
        for metric in ("tok_once", "llm_calls"):
            s = e[metric]
            sw.append(dict(dataset=e["dataset"], metric=metric, shapiro_p=s["shapiro_p"],
                           n=s["shapiro_n"], subsampled=s["shapiro_subsampled"],
                           skew=s.get("skew"), excess_kurtosis=s.get("excess_kurtosis")))
    worst = max(sw, key=lambda r: r["shapiro_p"])
    claim_re = re.compile(r"Shapiro--Wilk \$p\\le([0-9.]+)\\times10\^\{(-?[0-9]+)\}\$")
    claims = []
    for p in (ROOT / "latex/sections/appendix.tex", ROOT / "latex_acl8/sections/appendix.tex"):
        for i, ln in enumerate(p.read_text().splitlines(), 1):
            m = claim_re.search(ln)
            if m:
                claims.append(dict(file=str(p.relative_to(ROOT)), line=i,
                                   claimed=f"{m.group(1)}e{m.group(2)}",
                                   claimed_value=float(m.group(1)) * 10 ** int(m.group(2))))
    shapiro = dict(per_cell=sw, max_shapiro_p=worst,
                   claims=claims,
                   claim_holds=all(worst["shapiro_p"] <= c["claimed_value"] for c in claims),
                   correct_bound=worst["shapiro_p"])

    # cross-reference
    xref = []
    pat = re.compile(r"pre-declared family\s*\n?\s*\(Appendix~\\ref\{([^}]+)\}\)")
    for p in (ROOT / "latex/sections/experimental_setup.tex",
              ROOT / "latex_acl8/sections/experimental_setup.tex"):
        text = p.read_text()
        for m in re.finditer(r"pre-declared family[^.]*?\(Appendix~\\ref\{([^}]+)\}\)",
                             text, re.S):
            line = text[:m.start()].count("\n") + 1
            xref.append(dict(file=str(p.relative_to(ROOT)), line=line, target=m.group(1),
                             snippet=" ".join(m.group(0).split())))
    # what each appendix label actually is, and where families ARE defined
    app_titles = {}
    for p in (ROOT / "latex/sections/appendix.tex", ROOT / "latex_acl8/sections/appendix.tex"):
        cur = None
        order = []
        for ln in p.read_text().splitlines():
            ms = re.match(r"\\section\{(.*)\}", ln)
            if ms:
                cur = ms.group(1)
            ml = re.match(r"\\label\{(app:[^}]+)\}", ln)
            if ml and cur:
                order.append((ml.group(1), cur))
        app_titles[str(p.relative_to(ROOT))] = [
            dict(letter=chr(ord("A") + i), label=lab, title=t) for i, (lab, t) in enumerate(order)]
    fam_def = []
    for p in sorted((ROOT / "latex_acl8/sections").glob("*.tex")) + \
             sorted((ROOT / "latex/sections").glob("*.tex")):
        for i, ln in enumerate(p.read_text().splitlines(), 1):
            if "sec:setup-stats" in ln and "label" in ln:
                fam_def.append(dict(file=str(p.relative_to(ROOT)), line=i, label="sec:setup-stats"))
    return dict(shapiro=shapiro, crossref=dict(occurrences=xref, appendix_sections=app_titles,
                                               family_definition_labels=fam_def))


# ==============================================================================================
# rendering
# ==============================================================================================

def fp(p) -> str:
    if p is None:
        return "--"
    return f"{p:.3g}"


def render(d: dict) -> str:
    L = []
    A = L.append
    A("# ARR review response: the six quantities the reviewers asked for\n")
    A("Generated by `analysis/arr_review_response.py` (read-only over `runs/`, `latex/`, "
      "`latex_acl8/`). Every EM/judge number is `evaluation.metrics.answer_em` + the "
      "forced-answer recovery overlay via `scripts.compare_cells`; every paired test is the "
      "exact two-sided McNemar (`scripts.compare_cells.mcnemar_p`) on the discordant pairs; "
      "recall is `scripts.compare_cells.gold_doc_recall`, the per-instance BINARY "
      "\"at least one gold document surfaced\" indicator, tested with the same exact McNemar "
      "(`analysis.recall_inversion_test.paired_recall`); every step cap is read from the "
      "CONFIGURED `max_steps` stamped on each row (`analysis.step_budget_audit.per_row_caps`), "
      "never from an observed step count. Judge deltas obey the project's "
      f"$\\ge${JUDGE_COVERAGE_MIN*100:.0f}% both-sides coverage rule.\n")

    # PART 0
    p0 = d["part0_parity"]
    A("## Part 0. Loader parity (proof, not assumption)\n")
    A(f"`_compute_cell` vs the canonical `metrics(cell_rows(...))` path on `{p0['cell']}` "
      f"(n={p0['n']}): identical id set = {p0['same_id_set']}, field mismatches across "
      f"{p0['fields_compared']} = {p0['n_field_mismatches']}. "
      f"**{'PASS' if p0['passed'] else 'FAIL'}**\n")

    # PART 1
    g = d["gate"]
    A("## Part 1. Mandatory sanity gate (one target per dataset family)\n")
    A("| gate | arms | caps | n | EM(A) | EM(B) | Δ | expected Δ | p | expected p | result |")
    A("|---|---|---|---:|---:|---:|---:|---:|---|---|---|")
    for r in g["gates"]:
        if "error" in r:
            A(f"| ({r['key']}) | {r.get('label')} | | | | | | | | | **FAIL ({r['error']})** |")
            continue
        A(f"| ({r['key']}) {r['label']} | {r['arm_a']} vs {r['arm_b']} | "
          f"{r['caps_a']} / {r['caps_b']} | {r['n']} | {r['em_a']:.2f} | {r['em_b']:.2f} | "
          f"{r['delta']:+.2f} | {r['expected_delta']:+.2f} | {fp(r['p'])} | "
          f"{r['expected_p']:.3g} | **{'PASS' if r['passed'] else 'FAIL'}** |")
    A(f"\n**Gate overall: {'PASS' if g['passed'] else 'FAIL'}**\n")

    # TASK 1
    t1 = d["task1"]
    A("## Task 1. The coaching confound's EM-vs-judge signature (Reviewer 3)\n")
    A("### (a) Does the judge really short-circuit on exact match?\n")
    A("Yes, twice over, and the code is unambiguous:\n")
    A("- `scripts/judge_cells.py::classify_row` -- if `answer_em(ans, gold)` is 1 the row is "
      "recorded as `judge_correct=True`, `method=\"em_shortcircuit\"`, **with no LLM call**; "
      "that record is what `judge_cache.jsonl` stores and what "
      "`scripts.compare_cells.load_judge_cache` overlays at scoring time.\n"
      "- `evaluation/llm_judge.py::judge_answer_detail` short-circuits a *second* time on its own "
      "`_norm_exact` normalization (strip markdown/trailing punctuation, casefold), returning "
      "`judge_correct=True` with no call.\n"
      "- So EM-correct $\\Rightarrow$ judge-correct is an identity of the scoring pipeline, not a "
      "property of the judge model. (An empty answer is judged `False` with no call.)\n")
    sc = t1["shortcircuit_audit"]
    A(f"Falsifiable consequence, checked over every judged cell used in this report: no instance "
      f"anywhere can be (EM-correct, judge-incorrect). Observed violations: "
      f"**{sc['total_violations']}** out of "
      f"{sum(v['n_em_and_judged'] for v in sc['per_cell'].values())} judged EM-correct instances "
      f"across {len(sc['per_cell'])} cells. The short-circuit works exactly as R3 describes.\n")
    A("### (b)-(d) Per-instance cross-tabulation, Sieve vs the BM25 search+visit baseline\n")
    A("Judge coverage rule: BrowseComp-Plus-Structured is fully judged; on the Wikipedia datasets "
      "only a subset of cells was judged, and a judge delta is computed only where BOTH arms "
      f"clear {JUDGE_COVERAGE_MIN*100:.0f}% coverage. The cap-100 Sieve repair arm "
      "(`runs/_budget100`) carries **no judge cache at all**, so every judge cross-tab below "
      "necessarily reads the tabulated cap-50 Sieve arm on the Wikipedia datasets -- the same arm "
      "the paper's own Finding 1 reads, and budget-matched to the cap-50 baseline (caps shown).\n")
    A("Because EM-correct implies judge-correct, only three joint-judge partitions can carry any "
      "EM movement, and the paired EM delta decomposes exactly:\n")
    A("$$\\Delta\\mathrm{EM} = \\underbrace{(b_{TT}-c_{TT})/n}_{\\text{judge-correct in BOTH "
      "arms: pure answer-form}} + \\underbrace{b_{FT}/n}_{\\text{judge } 0\\to1} - "
      "\\underbrace{c_{TF}/n}_{\\text{judge } 1\\to0}$$\n")
    A("| dataset | caps (Sieve / base) | judge cov. | n | ΔEM | p(EM) | Δjudge | p(judge) | "
      "EM gains | of which already judge-correct | EM gains that flip judge | EM losses | "
      "already judge-correct | flip judge |")
    A("|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        r = t1["pairs"][ds]
        if "crosstab" not in r:
            A(f"| {PRETTY[ds]} | {r['caps_sieve']} / {r['caps_baseline']} | "
              f"{r['judge_coverage_sieve']*100:.0f}% / {r['judge_coverage_baseline']*100:.0f}% | "
              "| | | | | (below coverage gate) | | | | |")
            continue
        x = r["crosstab"]
        A(f"| {PRETTY[ds]} | {r['caps_sieve']} / {r['caps_baseline']} | "
          f"{r['judge_coverage_sieve']*100:.0f}% / {r['judge_coverage_baseline']*100:.0f}% | "
          f"{x['n']} | {x['em_delta']:+.2f} | {fp(x['em_p'])} | {x['judge_delta']:+.2f} | "
          f"{fp(x['judge_p'])} | {x['em_b_disc']} | {x['gain_already_judge_correct']} | "
          f"{x['gain_judge_flip']} | {x['em_c_disc']} | {x['loss_already_judge_correct']} | "
          f"{x['loss_judge_flip']} |")
    A("")
    A("Decomposition of the paired EM delta by joint judge status (points of EM). The two halves "
      "partition the discordant pairs, so each carries its own exact McNemar:\n")
    A("| dataset | ΔEM total | TT: judge-correct in BOTH arms (answer-form only) | p(TT half) | "
      "FT: judge 0→1 | TF: judge 1→0 | net judge-flip half | p(flip half) | share of net EM gain "
      "that is already-judge-correct | gross share of EM gains already judge-correct |")
    A("|---|---:|---:|---|---:|---:|---:|---|---:|---:|")
    for ds in DATASETS:
        r = t1["pairs"][ds]
        if "crosstab" not in r:
            continue
        x = r["crosstab"]
        A(f"| {PRETTY[ds]} | {x['em_delta']:+.2f} | {x['contrib_TT_pts']:+.2f} | "
          f"{fp(x['p_TT_component'])} | {x['contrib_FT_pts']:+.2f} | {x['contrib_TF_pts']:+.2f} | "
          f"{x['contrib_flip_pts']:+.2f} | {fp(x['p_flip_component'])} | "
          f"{x['share_of_net_em_gain_already_judge_correct']:.0f}% | "
          f"{x['share_of_em_gain_already_judge_correct_gross']:.0f}% |")
    A("")
    for ds in DATASETS:
        sub = t1["pairs"][ds].get("cap50_matched_subset")
        if sub and sub.get("crosstab"):
            x, e = sub["crosstab"], sub["em"]
            A(f"Budget-matched robustness ({PRETTY[ds]}'s baseline cell is internally mixed): on "
              f"the cap-50 subset only (n={sub['n']}), ΔEM {e['delta']:+.2f} (p={fp(e['p'])}), "
              f"Δjudge {x['judge_delta']:+.2f} (p={fp(x['judge_p'])}), already-judge-correct share "
              f"of the net EM gain {x['share_of_net_em_gain_already_judge_correct']:.0f}%.\n")
    A("**Verdict, per dataset (this is the number the paper needs).**\n")
    for ds in DATASETS:
        r = t1["pairs"][ds]
        if "crosstab" not in r:
            continue
        x = r["crosstab"]
        sig_tt = x["p_TT_component"] < 0.05
        sig_fl = x["p_flip_component"] < 0.05
        if sig_tt and not sig_fl:
            call = ("**R3 is right on this dataset.** The only half of the exact-match gain that "
                    "is distinguishable from zero is the half on answers the judge already "
                    "accepted in both arms")
        elif sig_fl and not sig_tt:
            call = ("**R3 is wrong on this dataset.** The gain is carried by answers that were "
                    "judged incorrect at baseline and correct under the method")
        elif sig_tt and sig_fl:
            call = ("**R3 is partly right on this dataset.** Both halves move significantly, so "
                    "the gain is not purely answer-form")
        else:
            call = ("**Neither half is distinguishable from zero on this dataset**, consistent "
                    "with the overall null")
        A(f"- **{PRETTY[ds]}**: of the {x['em_delta']:+.2f}-point exact-match gain, "
          f"{x['contrib_TT_pts']:+.2f} points ({x['share_of_net_em_gain_already_judge_correct']:.0f}% "
          f"of the net gain; p={fp(x['p_TT_component'])}) comes from instances the judge scored "
          f"CORRECT in BOTH arms -- answers that were already accepted and merely became "
          f"exact-match-formatted. The remaining {x['contrib_flip_pts']:+.2f} points "
          f"(p={fp(x['p_flip_component'])}) come from answers whose judge verdict actually "
          f"changed. In raw counts, {x['gain_already_judge_correct']} of the {x['em_b_disc']} "
          f"instances that flipped EM-incorrect -> EM-correct "
          f"({x['share_of_em_gain_already_judge_correct_gross']:.0f}%) were judge-correct in both "
          f"arms; {x['gain_judge_flip']} flipped judge-incorrect -> judge-correct. {call}.")
    A("")
    A("Two limits on how far this can be pushed. (1) An answer-form gain is *consistent with* "
      "the answer-type coaching R3 points at, but this decomposition does not identify the "
      "mechanism: an interface that surfaces a cleaner extractable span would leave the same "
      "signature. (2) The judge is itself an instrument with error; a judge-flip is evidence of "
      "content movement only to the extent the judge is right.\n")
    A("Same cross-tabulation for the other fully-judged BrowseComp-Plus-Structured contrasts "
      "(all cells judged on all 830):\n")
    A("| contrast | n | ΔEM | p | Δjudge | p | EM gains | already judge-correct | judge flips | "
      "share of net EM gain already judge-correct |")
    A("|---|---:|---:|---|---:|---|---:|---:|---:|---:|")
    for name, x in d["task1"]["bcp_extra_contrasts"].items():
        A(f"| {name} | {x['n']} | {x['em_delta']:+.2f} | {fp(x['em_p'])} | "
          f"{x['judge_delta']:+.2f} | {fp(x['judge_p'])} | {x['em_b_disc']} | "
          f"{x['gain_already_judge_correct']} | {x['gain_judge_flip']} | "
          f"{x['share_of_net_em_gain_already_judge_correct']:.0f}% |")
    A("")

    # TASK 2
    t2 = d["task2"]
    A("## Task 2. Signing the pool-size bound (Reviewers 1 and 2)\n")
    A("`SERP bm25 k=10` vs `SERP bm25 [BASELINE]` (k=5) is the paper's own measurement of the "
      "pool effect with nothing else changed. Caps are shown per arm; the recovery-overlay parity "
      "of the two arms is reported because it is a live confound in this particular pair.\n")
    A("| dataset | caps k=10 / k=5 | n | EM k=10 | EM k=5 | ΔEM (pool) | b/c | p | recall k=10 | "
      "recall k=5 | Δrecall | b/c | p | empty% k=10 / k=5 | recovered k=10 / k=5 |")
    A("|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|---|---|")
    for ds in DATASETS:
        r = t2["pool_effect"][ds]
        e, rc = r["em"], r["recall"]
        A(f"| {PRETTY[ds]} | {r['caps_k10']} / {r['caps_k5']} | {e['n']} | {e['em_a']:.2f} | "
          f"{e['em_b']:.2f} | {e['delta']:+.2f} | {e['b']}/{e['c']} | {fp(e['p'])} | "
          f"{rc['recall_a']:.2f} | {rc['recall_b']:.2f} | {rc['delta']:+.2f} | "
          f"{rc['b']}/{rc['c']} | {fp(rc['p'])} | {r['empty_pct_k10']:.1f} / "
          f"{r['empty_pct_k5']:.1f} | {r['n_recovered_k10']} / {r['n_recovered_k5']} |")
    A("")
    for ds in DATASETS:
        sub = t2["pool_effect"][ds].get("cap50_matched_subset")
        if sub:
            A(f"Budget-matched robustness ({PRETTY[ds]}'s k=5 baseline cell is internally mixed): "
              f"on the cap-50 subset only (n={sub['n']}), the pool effect is "
              f"ΔEM {sub['em']['delta']:+.2f} (p={fp(sub['em']['p'])}) and "
              f"Δrecall {sub['recall']['delta']:+.2f} (p={fp(sub['recall']['p'])}).\n")
    A("Diagnostic only (post-treatment conditioning; cannot support a causal claim): the same "
      "pool contrast restricted to instances BOTH arms answered.\n")
    A("| dataset | n both answered | EM k=10 | EM k=5 | Δ | b/c | p |")
    A("|---|---:|---:|---:|---:|---|---|")
    for ds in DATASETS:
        x = t2["pool_effect"][ds]["diagnostic_both_answered"]
        A(f"| {PRETTY[ds]} | {x['n']} | {x['em_k10']:.2f} | {x['em_k5']:.2f} | "
          f"{x['delta']:+.2f} | {x['b']}/{x['c']} | {fp(x['p'])} |")
    A("")
    A("### The approximate bound on the headline\n")
    A("The headline contrast is Sieve (k=5) against `Sparse only, same interface` (k=10). "
      "Matching the pool on either side shifts the contrast by the measured pool effect $P$ "
      "(residual $= \\Delta + P$). **This is an approximate bound, not a corrected estimate**: "
      "the pool effect is measured on a different condition (BM25 search+visit) at a different "
      "read interface, and interface and pool effects need not be additive.\n")
    A("| dataset | caps Sieve / control | headline ΔEM | p | pool effect P (EM) | P significant | "
      "residual ΔEM (approx. bound) | headline Δrecall | pool effect P (recall) | residual Δrecall |")
    A("|---|---|---:|---|---:|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        h = t2["headline"][ds]
        bd = t2["bound"][ds]
        A(f"| {PRETTY[ds]} | {h['caps_sieve']} / {h['caps_control']} | "
          f"{h['em']['delta']:+.2f} | {fp(h['em']['p'])} | {bd['pool_em_effect']:+.2f} | "
          f"{'yes' if bd['pool_em_significant'] else 'no'} | {bd['residual_em']:+.2f} | "
          f"{h['recall']['delta']:+.2f} | {bd['pool_recall_effect']:+.2f} | "
          f"{bd['residual_recall']:+.2f} |")
    A("")
    A("Same bound applied to the recall inversion against the hybrid control (also k=10):\n")
    A("| dataset | caps Sieve / hybrid | Δrecall (Sieve − hybrid) | b/c | p | pool effect P | "
      "residual Δrecall (approx. bound) |")
    A("|---|---|---:|---|---|---:|---:|")
    for ds in DATASETS:
        r = t2["recall_inversion_bound"][ds]
        A(f"| {PRETTY[ds]} | {r['caps_sieve']} / {r['caps_control']} | "
          f"{r['recall']['delta']:+.2f} | {r['recall']['b']}/{r['recall']['c']} | "
          f"{fp(r['recall']['p'])} | {r['pool_recall_effect']:+.2f} | "
          f"{r['residual_recall']:+.2f} |")
    A("")
    A("**Verdict.** Signed, the pool effect on exact match is **negative on all three datasets**: "
      "showing the agent ten candidates instead of five *lowers* EM, by "
      + ", ".join(f"{abs(t2['pool_effect'][ds]['em']['delta']):.2f} points on {PRETTY[ds]} "
                  f"(p={fp(t2['pool_effect'][ds]['em']['p'])})" for ds in DATASETS)
      + ". The paper currently states the k=5/k=10 asymmetry is \"conservative for our claims, "
      "since those controls see twice as many candidates per call\". On exact match that is the "
      "wrong sign: the k=10 controls are, if anything, handicapped by their larger pool, so part "
      "of the headline margin against them may be pool size rather than the manipulation of "
      "interest. Subtracting the measured pool effect leaves an approximate bound of "
      + ", ".join(f"{t2['bound'][ds]['residual_em']:+.2f} on {PRETTY[ds]}" for ds in DATASETS)
      + " (against headline "
      + "/".join(f"{t2['headline'][ds]['em']['delta']:+.2f}" for ds in DATASETS)
      + "). On gold-document recall the sign is the opposite and the paper's framing holds: "
      "k=10 *raises* recall, by "
      + ", ".join(f"{t2['pool_effect'][ds]['recall']['delta']:+.2f} on {PRETTY[ds]} "
                  f"(p={fp(t2['pool_effect'][ds]['recall']['p'])})" for ds in DATASETS)
      + ", so the k=10 controls' recall advantage over Sieve is partly bought by their pool. "
      "One caveat on the MuSiQue pool effect: its k=5 baseline cell is internally mixed, and on "
      "the budget-matched cap-50 subset the EM effect is -1.13 (p=0.136), so that dataset's "
      "significance rests on the 200 cap-100 episodes.\n")
    A("One confound inside this very estimate, stated rather than adjusted away: the two pool "
      "arms did not receive the same forced-answer backfill. The k=5 baseline carries a recovery "
      "sidecar on all three datasets (72/168/101 rows refilled) and the k=10 arm carries one only "
      "on BrowseComp-Plus-Structured (4 rows). On the two Wikipedia datasets the k=10 arm is "
      "therefore scored with an un-backfilled empty rate (8.9% vs 6.0%; 8.1% vs 6.0%), which "
      "pushes the pool effect negative for a reason that has nothing to do with pool size. The "
      "answered-only diagnostic above puts the pool effect within noise on all three datasets "
      "(-0.97, +0.24, -0.43; all p>0.6), which is what one expects if most of the raw negative "
      "pool effect is the missing backfill. Both readings are reported; neither is a correction.\n")

    # TASK 3
    A("## Task 3. The recall inversion against a pool-matched comparator (Reviewer 1)\n")
    A("Gold-document recall is the per-instance binary indicator; the test is the exact McNemar "
      "on it. `b` = comparator surfaced a gold document and Sieve did not; `c` = Sieve did and "
      "the comparator did not.\n")
    for ds in DATASETS:
        A(f"### {PRETTY[ds]}\n")
        A("| comparator | pool | caps Sieve / comp. | n | recall Sieve | recall comp. | Δ | b/c | "
          "p | EM Sieve | EM comp. | ΔEM | p(EM) |")
        A("|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|")
        for r in d["task3"][ds]:
            rc, e = r["recall"], r["em"]
            A(f"| {r['comparator']} | {'MATCHED (k=5)' if r['pool_matched'] else 'k=10'} | "
              f"{r['caps_sieve']} / {r['caps_comparator']} | {rc['n']} | {rc['recall_a']:.2f} | "
              f"{rc['recall_b']:.2f} | {rc['delta']:+.2f} | {rc['b']}/{rc['c']} | {fp(rc['p'])} | "
              f"{e['em_a']:.2f} | {e['em_b']:.2f} | {e['delta']:+.2f} | {fp(e['p'])} |")
        A("")
    A("**Verdict.** R1's factual claim is confirmed: on BrowseComp-Plus-Structured, Sieve's "
      "gold-document recall (61.2) is *higher* than the pool-matched k=5 baseline's (58.9), not "
      "lower -- directionally, at p=0.24, so the right statement is \"no significant "
      "difference, point estimate in Sieve's favour\", not \"Sieve recalls more\":\n")
    for ds in DATASETS:
        rows = {r["comparator"]: r for r in d["task3"][ds]}
        b5 = rows["SERP bm25 [BASELINE] k=5 (POOL-MATCHED)"]["recall"]
        b10 = rows["SERP bm25 k=10"]["recall"]
        hy = rows["hybrid control (k=10)"]["recall"]
        nd = rows["no dense evidence (k=5, POOL-MATCHED)"]["recall"]
        A(f"- **{PRETTY[ds]}**: Sieve {b5['recall_a']:.1f} vs the k=5 baseline "
          f"{b5['recall_b']:.1f} ({b5['delta']:+.2f}, p={fp(b5['p'])}); vs the k=10 baseline "
          f"{b10['recall_b']:.1f} ({b10['delta']:+.2f}, p={fp(b10['p'])}); vs the k=10 hybrid "
          f"control {hy['recall_b']:.1f} ({hy['delta']:+.2f}, p={fp(hy['p'])}); vs the "
          f"pool-matched no-dense ablation {nd['recall_b']:.1f} ({nd['delta']:+.2f}, "
          f"p={fp(nd['p'])}).")
    A("")
    sig_less_matched = [
        (ds, r) for ds in DATASETS for r in d["task3"][ds]
        if r["pool_matched"] and r["recall"]["delta"] < 0 and r["recall"]["p"] < 0.05]
    sig_more_matched = [
        (ds, r) for ds in DATASETS for r in d["task3"][ds]
        if r["pool_matched"] and r["recall"]["delta"] > 0 and r["recall"]["p"] < 0.05]
    A(f"Across all {sum(len(d['task3'][ds]) for ds in DATASETS)} comparisons above, the number of "
      f"POOL-MATCHED (k=5) comparators against which Sieve recalls significantly LESS is "
      f"**{len(sig_less_matched)}**; the number against which it recalls significantly MORE is "
      f"**{len(sig_more_matched)}** ("
      + "; ".join(f"{PRETTY[ds]} / {r['comparator']}: {r['recall']['delta']:+.2f}, "
                  f"p={fp(r['recall']['p'])}" for ds, r in sig_more_matched)
      + "). Every comparator against which Sieve recalls significantly less is a k=10 cell. "
      "**The inversion as the paper states it -- \"Sieve retrieves fewer gold documents\" -- does "
      "not survive a pool-matched comparison.** What does survive is the narrower statement that "
      "Sieve scores higher than every control that out-recalls it, and that those controls "
      "out-recall it at least partly because they are shown twice as many candidates. On the two "
      "Wikipedia datasets the deficit against the hybrid control is still negative after "
      "subtracting the pool effect ("
      + ", ".join(f"{t2['recall_inversion_bound'][ds]['residual_recall']:+.2f} on {PRETTY[ds]}"
                  for ds in (HQA, MSQ))
      + "), but small; on BrowseComp-Plus-Structured the pool effect accounts for "
      f"{100 * t2['pool_effect'][BCP]['recall']['delta'] / abs(t2['recall_inversion_bound'][BCP]['recall']['delta']):.0f}% "
      "of the 9.9-point deficit the paper prices as the cost of Sieve's filter.\n")

    # TASK 4
    t4 = d["task4"]
    A("## Task 4. No-answer rates (Reviewer 2)\n")
    A("`empty%` is the post-backfill no-answer rate: blank, whitespace-only, or a placeholder "
      "like \"...\", measured AFTER the forced-answer recovery overlay "
      "(`scripts.force_answer_backfill.needs_recovery`, the same predicate the backfill script "
      "selects rows on). `recov` counts rows whose blank answer that pass successfully refilled; "
      "a cell with `recov=0` never received a backfill pass.\n")
    for ds in DATASETS:
        A(f"### {PRETTY[ds]}\n")
        A("| condition | caps | n | empty% (post-backfill) | recovered | EM% |")
        A("|---|---|---:|---:|---:|---:|")
        for r in t4["rates"][ds]:
            A(f"| {r['cell']} | {r['caps']} | {r['n']} | {r['empty_pct']:.2f} | "
              f"{r['n_recovered']} | {r['em_pct']:.2f} |")
        A("")
    A("### DIAGNOSTIC ONLY: EM conditioned on producing an answer\n")
    A("**Post-treatment conditioning. Whether an episode produces an answer at all is an outcome "
      "of the condition, so conditioning on it breaks randomisation and these numbers cannot "
      "support a causal claim.** They are reported because a reviewer asked what the accuracy "
      "gap looks like with the no-answer channel removed.\n")
    A("| dataset | unconditional ΔEM | p | EM\\|answered Sieve | EM\\|answered baseline | "
      "n answered (Sieve/base) | paired on both-answered: Δ | b/c | p |")
    A("|---|---:|---|---:|---:|---|---:|---|---|")
    for ds in DATASETS:
        c = t4["conditional_em"][ds]
        u, pb = c["unconditional"], c["paired_both_answered"]
        A(f"| {PRETTY[ds]} | {u['delta']:+.2f} | {fp(u['p'])} | "
          f"{c['em_given_answer_sieve']:.2f} | {c['em_given_answer_baseline']:.2f} | "
          f"{c['n_answered_sieve']}/{c['n_answered_baseline']} | {pb['delta']:+.2f} | "
          f"{pb['b']}/{pb['c']} | {fp(pb['p'])} |")
    A("")
    A("**Verdict.** Post-backfill no-answer rates run "
      f"{min(r['empty_pct'] for ds in DATASETS for r in t4['rates'][ds]):.1f}%-"
      f"{max(r['empty_pct'] for ds in DATASETS for r in t4['rates'][ds]):.1f}% across the "
      "headline conditions and they are NOT uniform across arms, which the paper never reports. "
      "Two asymmetries matter, and they point in opposite directions:\n")
    def erate(ds, key):
        return next(r["empty_pct"] for r in t4["rates"][ds] if r["cell"] == LABEL[key])

    A("- On BrowseComp-Plus-Structured, Sieve has the **lowest** no-answer rate of any headline "
      f"condition ({erate(BCP, SIEVE50):.2f}% against the baseline's "
      f"{erate(BCP, BASE_K5):.2f}% and the dense control's {erate(BCP, DENSESNIP):.2f}%), so "
      "part of its EM lead there is that it answers at all more often. On the two Wikipedia "
      "datasets the ordering reverses: Sieve's rate is *higher* than the baseline's ("
      + ", ".join(f"{PRETTY[ds]} {erate(ds, SIEVE50):.2f}% vs {erate(ds, BASE_K5):.2f}%"
                  for ds in (HQA, MSQ))
      + "), so there the accuracy gain is achieved despite answering less often.\n")
    A("- The forced-answer backfill was **not applied uniformly**. Of the headline cells, only "
      "the BrowseComp-Plus-Structured cells and the two Wikipedia k=5 BM25-baseline cells carry a "
      "recovery sidecar at all; both Wikipedia Sieve arms, both Wikipedia k=10 cells and every "
      "Wikipedia fetch-family cell have `recov=0`, i.e. never received a pass. The project's own "
      "practice note says the "
      "backfill is rank-changing and must be run uniformly across conditions before judging; on "
      "the Wikipedia datasets it was not. This is a fairness gap in the *baseline's* favour "
      "(the baseline got the pass, Sieve did not), so it does not threaten the method's gain, "
      "but it must be disclosed.\n")
    A("The conditional-EM diagnostic is consistent with both readings: on BrowseComp-Plus-"
      "Structured the +2.53 unconditional gap collapses to -0.27 (p=0.95) once both arms are "
      "required to have answered, i.e. essentially all of that dataset's directional gain is the "
      "no-answer channel; on the two Wikipedia datasets the gap *grows* (+1.57 -> +2.55; "
      "+2.74 -> +3.59). **These are post-treatment conditionings and support no causal claim**; "
      "they are a diagnostic about where the movement sits, nothing more.\n")

    # TASK 5
    t5 = d["task5"]
    v = t5["verify"]
    A("## Task 5. The wide correction, applied symmetrically (Reviewer 3)\n")
    A(f"The paper computes a pooled {v['m_em']}-test BrowseComp-Plus-Structured exact-match "
      "Holm correction exactly once -- for the one claim it kills (the no-dense-evidence rung, "
      f"$\\tilde p={v['no_dense_rung_A2']['adjusted']:.3g}$). Here is that identical pool applied "
      "to **every** test in it. Pool membership and raw $p$-values are "
      f"`{t5['census_provenance']['source']}` unchanged; the correction is "
      "`analysis.multiplicity_census.holm` unchanged. (`--` in the n column means the census "
      "carries no per-row n for that table-delta entry; every BrowseComp-Plus-Structured cell is "
      "n=830.)\n")
    a6 = v["headline_plain_bm25_A6"]
    A(f"**R3's specific claim, verified**: the abstract-leading \"+6.0 EM over plain BM25 at a "
      f"matched read interface\" (`{a6['id']}`, raw $p={a6['raw']:.4g}$) becomes "
      f"$\\tilde p={a6['adjusted']:.3g}$ in the same {v['m_em']}-test pool -- "
      f"**{'NOT significant' if not a6['sig_adj'] else 'still significant'}** at $\\alpha=0.05$. "
      f"R3 computed $\\approx0.055$; the exact value is {a6['adjusted']:.4g}.\n")
    A(f"Of the {v['m_em']} BrowseComp-Plus-Structured exact-match tests, "
      f"{t5['n_sig_raw_em']} are significant uncorrected and **{t5['n_survivors_em']}** survive "
      "the pooled Holm correction.\n")
    A("### Pooled-correction status of every BrowseComp-Plus-Structured headline claim\n")
    A("EM claims are corrected in the paper's own m=%d exact-match pool; judge claims in the "
      "m=%d accuracy (EM+judge) pool the paper also names.\n" % (v["m_em"], v["m_acc"]))
    A("| id | claim | metric | raw p | Holm (m=%d, EM pool) | Holm (m=%d, accuracy pool) | "
      "survives pooled correction |" % (v["m_em"], v["m_acc"]))
    A("|---|---|---|---:|---:|---:|---|")
    for r in t5["headlines"]:
        e = f"{r['adj_em43']:.3g}" if r["adj_em43"] is not None else "--"
        a = f"{r['adj_acc58']:.3g}" if r["adj_acc58"] is not None else "--"
        A(f"| {r['id']} | {r['claim']} | {r['metric']} | {r['raw']:.3g} | {e} | {a} | "
          f"{'**yes**' if r['sig_pooled'] else 'NO'} |")
    A("")
    A("### The full pool, every test\n")
    A("| id | comparison | metric | n | raw p | Holm-adjusted (m=%d) | sig raw | sig adjusted |"
      % v["m_em"])
    A("|---|---|---|---:|---:|---:|---|---|")
    for r in t5["em_pool"]:
        nn = r["n"] if r["n"] else "--"
        A(f"| {r['id']} | {r['cmp']} | {r['metric']} | {nn} | {r['raw']:.3g} | "
          f"{r['adjusted']:.3g} | {'yes' if r['sig_raw'] else 'no'} | "
          f"{'**yes**' if r['sig_adj'] else 'no'} |")
    A("")
    A(f"The same pool widened to all {v['m_acc']} BrowseComp-Plus-Structured accuracy tests "
      "(EM + judge), for the judge-metric headlines:\n")
    A("| id | comparison | metric | raw p | Holm-adjusted (m=%d) | sig adjusted |" % v["m_acc"])
    A("|---|---|---|---:|---:|---|")
    for r in t5["accuracy_pool"]:
        A(f"| {r['id']} | {r['cmp']} | {r['metric']} | {r['raw']:.3g} | {r['adjusted']:.3g} | "
          f"{'**yes**' if r['sig_adj'] else 'no'} |")
    A("")

    n_head_dead = sum(1 for r in t5["headlines"] if r["sig_raw"] and not r["sig_pooled"])
    n_head_live = sum(1 for r in t5["headlines"] if r["sig_pooled"])
    A(f"**Verdict.** Applied evenly, the pooled correction the paper reserves for one claim kills "
      f"{n_head_dead} of its own BrowseComp-Plus-Structured headline claims that are significant "
      f"uncorrected, and leaves {n_head_live} standing. The asymmetry R3 identifies is real: the "
      "pooled 43-test correction currently appears in the paper exactly once, attached to the "
      "no-dense-evidence rung, whose $\\tilde p=0.18$ is used to concede that rung; the same pool "
      "applied to the abstract-leading plain-BM25 margin gives $\\tilde p=0.055$ and is never "
      "mentioned. Either report the pooled column for every claim on this dataset or for none of "
      "them.\n")

    # TASK 6
    t6 = d["task6"]
    sh = t6["shapiro"]
    A("## Task 6. Two typos (Reviewer 3)\n")
    A("### (1) Shapiro--Wilk bound in the token/model-call appendix\n")
    for c in sh["claims"]:
        A(f"- `{c['file']}` line {c['line']} claims **$p \\le "
          f"{c['claimed_value']:.1e}$ throughout** (typeset as "
          f"$2.7\\times10^{{-29}}$).")
    A("")
    A("The actual Shapiro--Wilk $p$-values behind that table "
      "(`analysis/paired_ttests_data.json`, produced by `analysis/paired_ttests.py`):\n")
    A("| dataset | quantity | Shapiro--Wilk p | n | subsampled | skew | excess kurtosis |")
    A("|---|---|---:|---:|---|---:|---:|")
    for r in sh["per_cell"]:
        A(f"| {r['dataset']} | {r['metric']} | {r['shapiro_p']:.4g} | {r['n']} | "
          f"{'yes' if r['subsampled'] else 'no'} | {r['skew']:.2f} | {r['excess_kurtosis']:.1f} |")
    A("")
    w = sh["max_shapiro_p"]
    A(f"**Maximum over all six cells: $p = {w['shapiro_p']:.3g}$** "
      f"({w['dataset']}, {w['metric']}, n={w['n']}). The claimed bound is therefore "
      f"**{'correct' if sh['claim_holds'] else 'WRONG'}**: "
      f"{w['shapiro_p']:.3g} $>$ 2.7e-29. R3's figure ($\\approx1.0\\times10^{{-18}}$) is right. "
      "The correct sentence is **Shapiro--Wilk $p \\le 1.1\\times10^{-18}$ throughout** "
      f"(exact max {w['shapiro_p']:.4g}). Nothing downstream changes: every value is still far "
      "below any conventional threshold, so the non-normality statement the sentence exists to "
      "make is unaffected -- only the numeral is wrong.\n")
    A("### (2) The \"pre-declared family\" cross-reference\n")
    xr = t6["crossref"]
    for o in xr["occurrences"]:
        A(f"- `{o['file']}` line {o['line']}: \"{o['snippet']}\" $\\rightarrow$ "
          f"`{o['target']}`")
    A("")
    for f, secs in xr["appendix_sections"].items():
        letters = ", ".join(f"{s['letter']}={s['title']}" for s in secs)
        A(f"- `{f}` appendix order: {letters}")
    A("")
    A("So the reference resolves to Appendix H, *Full Token and Model-Call Statistics*, which "
      "contains no family definition at all -- it is the paired $t$-test table for tokens and "
      "calls. The families are defined in the body: the baseline family ($m{=}6$) and the "
      "ablation family ($m{=}12$, split $4{+}8$) are declared in the Results section, and the "
      "correction protocol itself in the statistics paragraph of the experimental setup "
      "(`sec:setup-stats`). The correct target is `\\S\\ref{sec:results-main}` (where each "
      "family's size and membership is stated), optionally alongside "
      "`Table~\\ref{tab:ablation-family}`, which carries the Holm-adjusted values.\n")
    # -------- what this costs the paper --------
    hq = t1["pairs"][HQA]["crosstab"]
    ms = t1["pairs"][MSQ]["crosstab"]
    bcp_pool_share = (100 * t2["pool_effect"][BCP]["recall"]["delta"]
                      / abs(t2["recall_inversion_bound"][BCP]["recall"]["delta"]))
    A("## What the paper must change\n")
    A("### Claims that must WEAKEN\n")
    A(f"1. **HotpotQA's Finding-1 exact-match gain ({hq['em_delta']:+.2f}, p={fp(hq['em_p'])}) "
      f"cannot be presented as a retrieval gain.** {hq['contrib_TT_pts']:+.2f} of those "
      f"{hq['em_delta']:.2f} points ({hq['share_of_net_em_gain_already_judge_correct']:.0f}% of "
      f"the net gain) come from answers the judge already accepted in BOTH arms, and that half is "
      f"the only one significant (p={fp(hq['p_TT_component'])}); the judge-flip half is "
      f"{hq['contrib_flip_pts']:+.2f} points, p={fp(hq['p_flip_component'])}. With the flat judge "
      "accuracy the paper already reports (49.4 -> 49.5, p=0.79) and the acknowledged fact that "
      "the baseline renders no tool manual, the honest reading is that HotpotQA's EM movement is "
      "dominated by answer form. Report the decomposition and stop citing HotpotQA EM as one of "
      "the two datasets where the interface effect replicates.")
    A("2. **The k=5/k=10 disclosure must be signed, and its stated direction is wrong on exact "
      "match.** The paper says the asymmetry is \"conservative for our claims\"; measured on its "
      "own k=10 baseline row, the larger pool *lowers* EM ("
      + ", ".join(f"{t2['pool_effect'][ds]['em']['delta']:.2f} on {PRETTY[ds]}, "
                  f"p={fp(t2['pool_effect'][ds]['em']['p'])}" for ds in DATASETS)
      + "), significantly on both Wikipedia datasets -- though MuSiQue's significance rests on "
      "the 200 cap-100 episodes in its internally-mixed baseline cell (on the budget-matched "
      "cap-50 subset the effect is -1.13, p=0.136). Replace \"conservative\" with the measured "
      "pool effect and report the residual bound ("
      + "/".join(f"{t2['bound'][ds]['residual_em']:+.2f}" for ds in DATASETS)
      + " against headline "
      + "/".join(f"{t2['headline'][ds]['em']['delta']:+.2f}" for ds in DATASETS)
      + "), labelled as an approximate bound and not a corrected estimate.")
    A("3. **The recall-inversion framing must be pool-qualified.** \"Sieve retrieves fewer gold "
      "documents\" holds only against k=10 comparators. Against the pool-matched k=5 baseline "
      "Sieve recalls MORE on BrowseComp-Plus-Structured and is indistinguishable on both "
      "Wikipedia datasets; against the pool-matched no-dense ablation it recalls significantly "
      f"more. Pool size accounts for {bcp_pool_share:.0f}% of the 9.9-point deficit against the "
      "hybrid that the paper currently prices as the cost of Sieve's filter.")
    A(f"4. **The pooled correction must be applied symmetrically.** {n_head_dead} "
      "BrowseComp-Plus-Structured headline claims that are significant uncorrected do not "
      "survive the same 43-test pool the paper already applies to one of them -- including the "
      "abstract-leading \"+6.0 EM over plain BM25\" ($\\tilde p=0.055$) and its judge twin "
      "($\\tilde p=0.090$). Report the pooled column for all of them, or for none.")
    A("5. **Disclose the non-uniform forced-answer backfill and the per-condition no-answer "
      "rates.** The recovery pass ran on the BrowseComp-Plus-Structured cells and the two "
      "Wikipedia k=5 BM25 baselines but on no Wikipedia Sieve, k=10 or fetch-family cell; "
      "post-backfill no-answer "
      "rates run 2.3%-10.2% across headline conditions and appear in no table. (The gap favours "
      "the baseline, so it does not threaten the method's gain -- but it is a stated project "
      "practice that was not met.)")
    A("6. **Fix the two errors in Task 6**: the Shapiro--Wilk bound "
      f"($p\\le2.7\\times10^{{-29}}$ -> $p\\le1.1\\times10^{{-18}}$, exact max "
      f"{sh['correct_bound']:.3e}) and the "
      "\"pre-declared family\" cross-reference (currently Appendix H, the token appendix; should "
      "be the Results section where the families are declared).")
    A("")
    A("### Claims that can be STRENGTHENED\n")
    A("1. **MuSiQue's Finding-1 gain passes the same test that damages HotpotQA's.** Both halves "
      f"of its {ms['em_delta']:+.2f}-point EM gain move significantly (answer-form "
      f"{ms['contrib_TT_pts']:+.2f}, p={fp(ms['p_TT_component'])}; judge-flip "
      f"{ms['contrib_flip_pts']:+.2f}, p={fp(ms['p_flip_component'])}), and judge accuracy moves "
      f"with it ({ms['judge_delta']:+.2f}, p={fp(ms['judge_p'])}). MuSiQue, not HotpotQA, is the "
      "Wikipedia dataset where the interface effect is demonstrably a content gain; the paper "
      "currently treats the two as interchangeable.")
    A("2. **The BrowseComp-Plus-Structured ablation rungs are content gains, not answer-form "
      "gains.** For Sieve vs the sparse-only control and vs the no-dense ablation, "
      f"{d['task1']['bcp_extra_contrasts']['Sieve vs sparse only, same interface']['share_of_net_em_gain_already_judge_correct']:.0f}% "
      "and "
      f"{d['task1']['bcp_extra_contrasts']['Sieve vs no dense evidence']['share_of_net_em_gain_already_judge_correct']:.0f}% "
      "of the net EM gain sits on already-judge-correct answers -- essentially none -- and both "
      "rungs' judge deltas track their EM deltas almost exactly (+6.02/+6.02; +4.70/+4.58). The "
      "coaching-confound objection does not reach these contrasts and the paper can say so.")
    A("3. **The pool asymmetry really is conservative on the recall side**, now a measurement "
      f"rather than an assumption: k=10 buys {t2['pool_effect'][BCP]['recall']['delta']:+.2f} "
      f"recall points on BrowseComp-Plus-Structured (p={fp(t2['pool_effect'][BCP]['recall']['p'])}), "
      "so every k=10 control's recall advantage is partly purchased, and Sieve's accuracy lead "
      "over those controls is achieved from a candidate set handicapped by design.")
    _oth = [r["empty_pct"] for r in t4["rates"][BCP] if r["cell"] != LABEL[SIEVE50]]
    A("4. **Sieve answers more often than any other headline condition on "
      f"BrowseComp-Plus-Structured** ({erate(BCP, SIEVE50):.2f}% no-answer against "
      f"{min(_oth):.2f}%-{max(_oth):.2f}% for the rest) -- a robustness property of the "
      "interface the pipeline already measures and the paper never reports.")
    A("")
    return "\n".join(L) + "\n"


def main() -> int:
    qrels_by_ds = {ds: load_qrels(ds) for ds in DATASETS}
    out: dict = {}
    print("[part 0] loader parity ...", flush=True)
    out["part0_parity"] = part0_parity(qrels_by_ds)
    if not out["part0_parity"]["passed"]:
        print("LOADER PARITY FAILED -- aborting", file=sys.stderr)
        print(json.dumps(out["part0_parity"], indent=2), file=sys.stderr)
        return 2
    print("[part 1] sanity gate ...", flush=True)
    out["gate"] = part1_gate(qrels_by_ds)
    for r in out["gate"]["gates"]:
        print("   ", json.dumps({k: r[k] for k in ("key", "n", "delta", "p", "passed")
                                 if k in r}), flush=True)
    if not out["gate"]["passed"]:
        print("SANITY GATE FAILED -- no new numbers produced", file=sys.stderr)
        JSON_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
        return 1
    for name, fn in (("task1", task1), ("task2", task2), ("task3", task3), ("task4", task4)):
        print(f"[{name}] ...", flush=True)
        out[name] = fn(qrels_by_ds)
    print("[task5] ...", flush=True)
    out["task5"] = task5()
    print("[task6] ...", flush=True)
    out["task6"] = task6()

    JSON_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    MD_PATH.write_text(render(out))
    print(f"wrote: {JSON_PATH}")
    print(f"wrote: {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
