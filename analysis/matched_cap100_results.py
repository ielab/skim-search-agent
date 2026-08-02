#!/usr/bin/env python
"""Budget-MATCHED (cap 100 vs cap 100) recomputation of the six control comparisons that
`analysis/step_budget_audit.py` found UNTESTABLE on the two Wikipedia datasets.

WHY THIS EXISTS
---------------
The audit established that on hotpotqa_structured / musique_structured the method cell
(`agent_research_bql_dense_snip`, tier `_headline_validation`) ran under a configured cap of 50
steps while EVERY control cell ran under a cap of 100. The budget-matched subset was therefore
EMPTY and six published control comparisons could not be confirmed or refuted -- only re-run.

They have now been re-run: `runs/_budget100/agent/<dataset>/<model>/agent_research_bql_dense_snip`
is the method at max_steps=100 on both Wikipedia datasets (n=7343 / n=2409, verified uniformly
cap 100 from the per-row `max_steps` configuration field -- see PART 0 below, which re-derives
that provenance here rather than trusting the claim). This script recomputes every affected
contrast with BOTH arms at cap 100, and puts each new number beside the published (unmatched,
Sieve@50 vs control@100) number it replaces.

THE CONTRASTS (all controls tier `_headline_validation`, model Tongyi-DeepResearch-30B-A3B):
    (a) vs `agent_research_hybrid_fetch_snip`  -- hybrid sparse+dense RRF, same read interface
    (b) vs `agent_research_snip`               -- no dense evidence (THE dense-fusion rung)
    (c) vs `agent_research_bm25_fetch_snip`    -- sparse only, same read interface
    (g) vs `agent_research_dense_fetch`        -- dense only, same read interface
plus the budget-sensitivity contrast Sieve@100 (new) vs Sieve@50 (the existing headline cell),
which measures what the step budget ALONE is worth to the method.

METHOD PARITY IS MANDATORY -- nothing statistical is reimplemented here:
  scripts.compare_cells                 : cell_dir, cell_rows (transitively
                                          force_answer_backfill.load_rows_with_recovery),
                                          metrics (transitively evaluation.metrics.answer_em and
                                          gold_doc_recall), mcnemar_p, pct, load_qrels
  analysis.structured_surface_control   : cell_metrics, paired_em, mean_tok_recall
  analysis.step_budget_audit            : paired_on_ids, _paired_tokens (the count-once token
                                          mean + the paper's declared paired t-test + Wilcoxon,
                                          exactly as used for the audit's own tables), per_row_caps
  analysis.recall_inversion_test        : paired_recall, _is_binary_indicator (the exact McNemar
                                          on the per-instance BINARY gold-doc-recall indicator)
The only new code is the choice of cells, the mean-LLM-calls summary (a plain mean over the
`llm_calls` field `metrics()` already produces), and the reporting/verdict layer.

WHAT "recall" MEANS: `scripts.compare_cells.gold_doc_recall` returns a Python BOOL -- a
per-instance "at least one gold doc surfaced in a retrieval observation" indicator, NOT a
continuous fraction. PART 0 re-verifies this empirically (`_is_binary_indicator`) before the
McNemar on recall is trusted; if it ever became continuous the script says so and falls back.

JUDGE: HotpotQA and MuSiQue are EM/F1 benchmarks and are NOT judged in this repo, by design.
No judge number is computed or reported for them.

MANDATORY SANITY GATE (PART 1, runs FIRST; a failure aborts with a nonzero exit and NO new
numbers), reproducing two already-published results with this script's own pipeline:
  (i)  Sieve@50 vs the SERP-bm25 baseline on hotpotqa_structured (both cap 50, already matched):
       +1.57 EM, p=0.00186
  (ii) Sieve vs "no dense evidence" on browsecomp_plus_structured (uniformly cap 100):
       +4.70 EM, p=0.0104

Reads runs/ READ-ONLY. Writes analysis/matched_cap100_results{_data.json,.md}.
Run: PYTHONPATH=. envs/bin/python analysis/matched_cap100_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import cell_dir, load_qrels, mcnemar_p, pct  # noqa: E402
from analysis.structured_surface_control import (  # noqa: E402
    cell_metrics, paired_em, mean_tok_recall,
)
from analysis.step_budget_audit import (  # noqa: E402
    paired_on_ids, _paired_tokens, per_row_caps, CACHE_PATH,
)
from analysis.recall_inversion_test import paired_recall, _is_binary_indicator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "analysis" / "matched_cap100_results_data.json"
MD_PATH = ROOT / "analysis" / "matched_cap100_results.md"

WIKI = ["hotpotqa_structured", "musique_structured"]
BROWSE = "browsecomp_plus_structured"

# (tier, condition)
SIEVE100 = ("_budget100", "agent_research_bql_dense_snip")        # THE REPAIR CELL (cap 100)
SIEVE50 = ("_headline_validation", "agent_research_bql_dense_snip")  # the published cell (cap 50)
HYBRID = ("_headline_validation", "agent_research_hybrid_fetch_snip")
NODENSE = ("_headline_validation", "agent_research_snip")
SPARSE = ("_headline_validation", "agent_research_bm25_fetch_snip")
DENSESNIP = ("_headline_validation", "agent_research_dense_fetch")
BASELINE = ("_visit_uncapped", "agent_research_bm25")             # SERP-bm25, gate (i) only

LABEL = {
    SIEVE100: "Sieve @cap100 (bql+dense+snip) [REPAIR]",
    SIEVE50: "Sieve @cap50 (bql+dense+snip) [published]",
    HYBRID: "hybrid control (bm25+dense RRF, snip+fetch)",
    NODENSE: "no dense evidence (bql+snip fetch)",
    SPARSE: "sparse only (bm25+snip fetch)",
    DENSESNIP: "dense same interface (dense+snip fetch)",
    BASELINE: "SERP bm25 [BASELINE]",
}

# (key, label, control cell, {dataset: published unmatched (Sieve@50 vs control@100) EM delta/p})
# `published` is what the paper / the repo's existing artifacts report for the UNMATCHED basis.
# The script ALSO recomputes that unmatched basis itself and cross-checks it against these
# constants (PART 2), so a drift in either direction is visible rather than assumed away.
CONTRASTS = [
    ("hybrid", "(a) Sieve vs hybrid control", HYBRID,
     {"hotpotqa_structured": dict(em_delta=5.7, p=1.62e-27),
      "musique_structured": dict(em_delta=5.5, p=1.58e-11)}),
    ("nodense", "(b) Sieve vs no dense evidence [THE DENSE-FUSION RUNG]", NODENSE,
     {"hotpotqa_structured": dict(em_delta=0.8, p=0.103),
      "musique_structured": dict(em_delta=1.3, p=0.0949)}),
    ("sparse", "(c) Sieve vs sparse only", SPARSE,
     {"hotpotqa_structured": dict(em_delta=4.0, p=1.19e-14),
      "musique_structured": dict(em_delta=5.7, p=1.37e-12)}),
    ("dense", "(g) Sieve vs dense same interface", DENSESNIP,
     # never separately published; the audit's own recomputed unmatched values are the reference
     {"hotpotqa_structured": dict(em_delta=5.82, p=1.73e-28, note="audit-recomputed, not published"),
      "musique_structured": dict(em_delta=6.77, p=1.33e-16, note="audit-recomputed, not published")}),
]

GATES = [
    dict(name="(i) Sieve@50 vs SERP-bm25 BASELINE, hotpotqa_structured (both cap 50)",
         dataset="hotpotqa_structured", a=SIEVE50, b=BASELINE,
         em_delta=1.57, p=0.00186, tol_delta=0.15, rel_tol_p=0.02),
    dict(name="(ii) Sieve vs 'no dense evidence', browsecomp_plus_structured (both cap 100)",
         dataset=BROWSE, a=SIEVE50, b=NODENSE,
         em_delta=4.70, p=0.0104, tol_delta=0.15, rel_tol_p=0.02),
]

# cells to load per dataset
LOAD = {
    "hotpotqa_structured": [SIEVE100, SIEVE50, HYBRID, NODENSE, SPARSE, DENSESNIP, BASELINE],
    "musique_structured": [SIEVE100, SIEVE50, HYBRID, NODENSE, SPARSE, DENSESNIP],
    BROWSE: [SIEVE50, NODENSE],
}

EXPECTED_CAP = {
    (SIEVE100, "hotpotqa_structured"): 100, (SIEVE100, "musique_structured"): 100,
    (SIEVE50, "hotpotqa_structured"): 50, (SIEVE50, "musique_structured"): 50,
    (HYBRID, "hotpotqa_structured"): 100, (HYBRID, "musique_structured"): 100,
    (NODENSE, "hotpotqa_structured"): 100, (NODENSE, "musique_structured"): 100,
    (SPARSE, "hotpotqa_structured"): 100, (SPARSE, "musique_structured"): 100,
    (DENSESNIP, "hotpotqa_structured"): 100, (DENSESNIP, "musique_structured"): 100,
}


# ---------------------------------------------------------------------------
# PART 0 -- provenance of every cell this script uses (configuration only)
# ---------------------------------------------------------------------------

def provenance(dataset: str, cell, cache: dict) -> dict:
    """Configured per-instance step cap distribution for one cell, from the per-row `max_steps`
    CONFIGURATION field (evaluation/run_eval.py stamps retriever.max_steps == cfg.agent.max_steps
    on each row; it is NOT an observed step count). Reuses step_budget_audit.per_row_caps."""
    caps = per_row_caps(dataset, cell[0], cell[1], cache)
    dist = {}
    for v in caps.values():
        dist[str(v)] = dist.get(str(v), 0) + 1
    return dict(dataset=dataset, tier=cell[0], cond=cell[1], label=LABEL[cell],
                n=len(caps), cap_dist=dist,
                uniform_cap=(int(list(dist)[0]) if len(dist) == 1 and list(dist)[0] != "None"
                             else None))


# ---------------------------------------------------------------------------
# PART 2/3 -- one comparison
# ---------------------------------------------------------------------------

def mean_llm_calls(m: dict, ids) -> float:
    ids = list(ids)
    return (sum(m[i]["llm_calls"] for i in ids) / len(ids)) if ids else float("nan")


def empty_pct(m: dict, ids) -> float:
    return pct([m[i]["empty"] for i in list(ids)])


def one_comparison(a_m: dict, b_m: dict, label_a: str, label_b: str) -> dict:
    """Full paired comparison of cell A (Sieve) vs cell B (control) on the inner-joined
    instance set: EM + count-once tokens (paired t + Wilcoxon) via step_budget_audit.paired_on_ids,
    mean LLM calls, and the exact McNemar on the binary gold-doc-recall indicator via
    recall_inversion_test.paired_recall."""
    ids = sorted(set(a_m) & set(b_m))
    sub_a = {i: a_m[i] for i in ids}
    sub_b = {i: b_m[i] for i in ids}

    out = paired_on_ids(a_m, b_m, ids)          # n, em_a, em_b, em_delta, b, c, p, sig, tokens
    out["label_a"], out["label_b"] = label_a, label_b
    out["n_shared"] = len(ids)
    out["llm_calls_a"] = mean_llm_calls(a_m, ids)
    out["llm_calls_b"] = mean_llm_calls(b_m, ids)
    out["llm_calls_delta"] = out["llm_calls_a"] - out["llm_calls_b"]
    out["empty_pct_a"] = empty_pct(a_m, ids)
    out["empty_pct_b"] = empty_pct(b_m, ids)

    binary = _is_binary_indicator(sub_a) and _is_binary_indicator(sub_b)
    n_r, rec_a, rec_b, rec_delta, rb, rc, rp = paired_recall(sub_a, sub_b)
    out["recall"] = dict(n=n_r, recall_a=float(rec_a), recall_b=float(rec_b),
                         delta=float(rec_delta), b=int(rb), c=int(rc), p=float(rp),
                         sig=bool(rp < 0.05), binary_indicator=bool(binary),
                         test="exact McNemar on the binary per-instance gold-doc indicator"
                              if binary else "NOT binary -- McNemar inappropriate, see note")
    return out


def em_verdict(pub_delta, pub_p, new) -> tuple:
    """(flag, sentence) comparing the PUBLISHED unmatched verdict to the NEW matched one.
    REVERSES = the sign of the effect flips AND the new result is significant in the opposite
    direction. CHANGES = the significance call flips, or the sign flips without significance.
    HOLDS = same direction and same significance call."""
    new_delta, new_sig = new["em_delta"], new["sig"]
    pub_sig = bool(pub_p is not None and pub_p < 0.05)
    sign_flip = (pub_delta >= 0) != (new_delta >= 0)
    if sign_flip and new_sig:
        return ("VERDICT REVERSES",
                f"published {pub_delta:+.2f} (sig={pub_sig}) -> matched {new_delta:+.2f} "
                f"(p={new['p']:.3g}), SIGNIFICANT IN THE OPPOSITE DIRECTION.")
    if sign_flip or (pub_sig != new_sig):
        return ("VERDICT CHANGES",
                f"published {pub_delta:+.2f}/sig={pub_sig} -> matched {new_delta:+.2f}"
                f"/sig={new_sig} (p={new['p']:.3g}).")
    if not new_sig:
        # both the published and the matched result are NULL. "HOLDS" here must not be read as
        # reassurance: it means the effect was not demonstrated before and is still not
        # demonstrated now, on a properly matched budget.
        return ("VERDICT HOLDS (STILL NULL)",
                f"published {pub_delta:+.2f} (n.s.) -> matched {new_delta:+.2f} "
                f"(p={new['p']:.3g}, n.s.): the contrast was null before and is null now, so "
                f"the effect remains UNDEMONSTRATED at a matched budget.")
    return ("VERDICT HOLDS",
            f"published {pub_delta:+.2f}/sig={pub_sig} -> matched {new_delta:+.2f}"
            f"/sig={new_sig} (p={new['p']:.3g}): same direction, same significance call.")


def token_verdict(pub_tokens, new_tokens) -> tuple:
    """Whether Sieve's token advantage survives budget matching. `pct_reduction` is
    (control - Sieve)/control * 100: positive = Sieve cheaper, negative = Sieve COSTLIER."""
    pub_red, new_red = pub_tokens.get("pct_reduction"), new_tokens.get("pct_reduction")
    p = new_tokens.get("t_p")
    sig = bool(p is not None and p < 0.05)
    if new_red is None:
        return ("TOKEN UNKNOWN", "no token data")
    if new_red < 0 and sig:
        return ("TOKENS REVERSE -- SIEVE IS NOW COSTLIER",
                f"unmatched {pub_red:+.1f}% reduction -> matched {new_red:+.1f}% "
                f"(i.e. Sieve spends {-new_red:.1f}% MORE than the control), paired t p={p:.3g}.")
    if new_red < 0:
        return ("TOKENS REVERSE (n.s.)",
                f"unmatched {pub_red:+.1f}% -> matched {new_red:+.1f}% (Sieve costlier), "
                f"paired t p={p if p is None else format(p, '.3g')}.")
    if not sig:
        return ("TOKEN ADVANTAGE NOT SIGNIFICANT",
                f"unmatched {pub_red:+.1f}% -> matched {new_red:+.1f}%, paired t "
                f"p={p if p is None else format(p, '.3g')}.")
    return ("TOKEN ADVANTAGE HOLDS",
            f"unmatched {pub_red:+.1f}% -> matched {new_red:+.1f}% reduction, paired t "
            f"p={p:.3g}.")


# ---------------------------------------------------------------------------

def fmt_p(p):
    return "--" if p is None else f"{p:.3g}"


def fmt(x, nd=1):
    return "--" if x is None or x != x else f"{x:,.{nd}f}"


def render_markdown(payload: dict) -> str:
    L = []
    A = L.append
    A("# Budget-matched (cap 100) recomputation of the Wikipedia control comparisons")
    A("")
    A("Generated by `analysis/matched_cap100_results.py` (read-only over `runs/`). Every "
      "comparison below has BOTH arms at a configured `max_steps=100`. The method arm is the "
      "repair cell `runs/_budget100/agent/<dataset>/Tongyi-DeepResearch-30B-A3B/"
      "agent_research_bql_dense_snip`; the controls are the existing uniformly-cap-100 "
      "`_headline_validation` cells. The six comparisons this replaces were reported "
      "**VERDICT UNTESTABLE** by `analysis/step_budget_audit.md` (Sieve ran at cap 50, every "
      "control at cap 100, zero matched instances).")
    A("")
    A("**HotpotQA and MuSiQue are EM/F1 benchmarks and are NOT judged in this repo, by design "
      "— no judge number is computed or reported for them.**")
    A("")

    # headline answers, generated from the computed numbers (not hand-written)
    nd = [r for r in payload["comparisons"] if r["key"] == "nodense"]
    A("## A. The two questions that matter, answered from the numbers below")
    A("")
    A("### A1. Does dense fusion contribute on the Wikipedia datasets at a matched budget?")
    A("")
    A("The `no dense evidence` rung is the ONLY evidence in the paper for dense fusion, and its "
      "two Wikipedia measurements were the confounded ones (the arm WITHOUT dense fusion had "
      "twice the step budget). Budget-matched:")
    A("")
    for r in nd:
        m = r["matched"]
        A(f"- **{r['dataset']}**: {m['em_a']:.2f} vs {m['em_b']:.2f} EM, "
          f"Δ={m['em_delta']:+.2f} (published, confounded: {r['published']['em_delta']:+.1f}), "
          f"b/c={m['b']}/{m['c']}, exact McNemar p={fmt_p(m['p'])} — "
          f"**{'SIGNIFICANT' if m['sig'] else 'NOT significant'}** at 0.05.")
    A("")
    if not any(r["matched"]["sig"] for r in nd):
        A("**ANSWER: NO. On neither Wikipedia dataset does dense fusion show a significant "
          "contribution once the step budget is matched.** Both contrasts were null before "
          "matching and remain null after; "
          + "; ".join(f"on {r['dataset']} the point estimate moves "
                      f"{r['published']['em_delta']:+.1f} -> {r['matched']['em_delta']:+.2f}"
                      for r in nd)
          + ". The paper cannot claim a demonstrated dense-fusion benefit on HotpotQA or "
            "MuSiQue; the only dataset where that rung is significant is "
            "BrowseComp-Plus (+4.7, p=0.0104, uniformly cap 100 and therefore never confounded).")
    else:
        A("**ANSWER: dense fusion IS significant on at least one Wikipedia dataset at matched "
          "budget — see the per-dataset lines above.**")
    A("")
    A("### A2. Does Sieve's token advantage survive budget matching?")
    A("")
    for r in payload["comparisons"]:
        ut, mt = r["unmatched"]["tokens"], r["matched"]["tokens"]
        A(f"- **{r['label']} / {r['dataset']}**: {ut['pct_reduction']:+.1f}% (confounded) -> "
          f"**{mt['pct_reduction']:+.1f}%** matched, paired t p={fmt_p(mt.get('t_p'))} — "
          f"{r['token_flag']}.")
    A("")
    costlier = [r for r in payload["comparisons"]
                if r["matched"]["tokens"]["pct_reduction"] < 0]
    if costlier:
        A("**Sieve is numerically COSTLIER than the control in: "
          + "; ".join(f"{r['label']} / {r['dataset']} "
                      f"({-r['matched']['tokens']['pct_reduction']:.1f}% more tokens, "
                      f"paired t p={fmt_p(r['matched']['tokens'].get('t_p'))})"
                      for r in costlier)
          + ".**")
    A("")

    # gate
    A("## 0. Mandatory sanity gate (reproducing published results with this pipeline)")
    A("")
    A("| check | n | EM Sieve | EM other | delta | expected | p | expected p | result |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for g in payload["gates"]:
        A(f"| {g['name']} | {g['n']} | {g['em_a']:.1f} | {g['em_b']:.1f} | {g['em_delta']:+.2f} | "
          f"{g['expected_delta']:+.2f} | {fmt_p(g['p'])} | {fmt_p(g['expected_p'])} | "
          f"**{'PASS' if g['passed'] else 'FAIL'}** |")
    A("")
    A(f"**Gate overall: {'PASS' if payload['gate_passed'] else 'FAIL'}**")
    A("")

    # provenance
    A("## 1. Provenance of every cell used (configured cap, from the per-row `max_steps` field)")
    A("")
    A("| dataset | cell | n | cap distribution | expected | ok |")
    A("|---|---|---:|---|---:|---|")
    for pr in payload["provenance"]:
        exp = pr.get("expected_cap")
        ok = "yes" if pr.get("cap_ok") else "**NO**"
        A(f"| {pr['dataset']} | {pr['label']} (`{pr['tier']}/{pr['cond']}`) | {pr['n']} | "
          f"{pr['cap_dist']} | {exp} | {ok} |")
    A("")
    A("The cap is read from CONFIGURATION only (the `max_steps` value stamped on each row when "
      "the episode ran), never inferred from observed step counts — see "
      "`analysis/step_budget_audit.py` for why observed steps are a post-treatment outcome.")
    A("")

    # headline table
    A("## 2. Side-by-side: PUBLISHED (unmatched) vs NEW MATCHED (both cap 100)")
    A("")
    A("`published` = Sieve@50 vs control@100 (the confounded basis the paper reports); "
      "`recomputed unmatched` = the same confounded contrast recomputed here as a cross-check; "
      "`MATCHED` = Sieve@100 vs control@100.")
    A("")
    A("| contrast | dataset | n_shared | published Δ / p | recomputed unmatched Δ / p | "
      "**MATCHED Δ / p** | b / c | flag |")
    A("|---|---|---:|---|---|---|---|---|")
    for r in payload["comparisons"]:
        pub = r["published"]
        um = r["unmatched"]
        mm = r["matched"]
        A(f"| {r['label']} | {r['dataset']} | {mm['n_shared']} | "
          f"{pub['em_delta']:+.2f} / {fmt_p(pub['p'])} | "
          f"{um['em_delta']:+.2f} / {fmt_p(um['p'])} | "
          f"**{mm['em_delta']:+.2f} / {fmt_p(mm['p'])}** | {mm['b']} / {mm['c']} | "
          f"**{r['em_flag']}** |")
    A("")

    # token table
    A("## 3. Token efficiency at matched budget (count-once tokens per episode)")
    A("")
    A("`% reduction` = (control − Sieve)/control × 100 — **positive = Sieve cheaper, negative = "
      "Sieve COSTLIER**. The paper's declared test is the paired t-test; Wilcoxon is reported "
      "alongside as the distribution-free companion.")
    A("")
    A("| contrast | dataset | unmatched: Sieve@50 / control | unmatched % red | "
      "**matched: Sieve@100 / control** | **matched % red** | paired t p | Wilcoxon p | flag |")
    A("|---|---|---|---:|---|---:|---|---|---|")
    for r in payload["comparisons"]:
        ut, mt = r["unmatched"]["tokens"], r["matched"]["tokens"]
        A(f"| {r['label']} | {r['dataset']} | {fmt(ut['tok_a'],0)} / {fmt(ut['tok_b'],0)} | "
          f"{ut['pct_reduction']:+.1f}% | "
          f"**{fmt(mt['tok_a'],0)} / {fmt(mt['tok_b'],0)}** | "
          f"**{mt['pct_reduction']:+.1f}%** | {fmt_p(mt.get('t_p'))} | "
          f"{fmt_p(mt.get('wilcoxon_p'))} | **{r['token_flag']}** |")
    A("")

    # llm calls + recall
    A("## 4. LLM calls and gold-document recall at matched budget")
    A("")
    A("| contrast | dataset | LLM calls Sieve@100 | LLM calls control | Δ | "
      "recall Sieve@100 | recall control | Δ | b / c | McNemar p | sig |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in payload["comparisons"]:
        mm = r["matched"]
        rc = mm["recall"]
        A(f"| {r['label']} | {r['dataset']} | {mm['llm_calls_a']:.1f} | {mm['llm_calls_b']:.1f} | "
          f"{mm['llm_calls_delta']:+.1f} | {rc['recall_a']:.1f} | {rc['recall_b']:.1f} | "
          f"{rc['delta']:+.1f} | {rc['b']} / {rc['c']} | {fmt_p(rc['p'])} | "
          f"{'yes' if rc['sig'] else 'no'} |")
    A("")
    A("Recall is `scripts.compare_cells.gold_doc_recall`, a per-instance BINARY indicator "
      "(\"at least one gold doc surfaced in a retrieval observation\"), re-verified as binary at "
      "runtime; the exact McNemar on the paired indicator is therefore the correct test "
      "(`analysis/recall_inversion_test.py`'s reasoning, reused unmodified).")
    A("")
    A("Binary-indicator check: "
      + ("all cells binary — McNemar valid."
         if all(r["matched"]["recall"]["binary_indicator"] for r in payload["comparisons"])
         else "**AT LEAST ONE CELL IS NOT BINARY — McNemar on recall is NOT valid there.**"))
    A("")

    # budget sensitivity
    A("## 5. Budget sensitivity: Sieve@100 vs Sieve@50 (same method, same instances)")
    A("")
    A("What the step budget ALONE is worth to the method — i.e. how badly the cap-50 numbers "
      "elsewhere in the paper hobble it.")
    A("")
    A("| dataset | n | EM @100 | EM @50 | Δ | b / c | McNemar p | sig | tok @100 | tok @50 | "
      "Δ tok | paired t p | LLM calls @100 | @50 | Δ | recall @100 | @50 | Δ | recall p |")
    A("|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for r in payload["budget_sensitivity"]:
        c = r["cmp"]
        t = c["tokens"]
        rc = c["recall"]
        A(f"| {r['dataset']} | {c['n_shared']} | {c['em_a']:.1f} | {c['em_b']:.1f} | "
          f"{c['em_delta']:+.2f} | {c['b']} / {c['c']} | {fmt_p(c['p'])} | "
          f"{'yes' if c['sig'] else 'no'} | {fmt(t['tok_a'],0)} | {fmt(t['tok_b'],0)} | "
          f"{t['tok_a'] - t['tok_b']:+,.0f} | {fmt_p(t.get('t_p'))} | "
          f"{c['llm_calls_a']:.1f} | {c['llm_calls_b']:.1f} | {c['llm_calls_delta']:+.1f} | "
          f"{rc['recall_a']:.1f} | {rc['recall_b']:.1f} | {rc['delta']:+.1f} | "
          f"{fmt_p(rc['p'])} |")
    A("")
    A("(Δ tok here is Sieve@100 − Sieve@50: positive = the extra budget was spent.)")
    A("")

    # empty-answer parity
    A("## 6. Empty-answer parity check (integrity)")
    A("")
    A("Empty/placeholder final answers bias paired comparisons (a condition that exhausts its "
      "budget more gets punished more), so the rate is reported per arm on the shared set. None "
      "of these Wikipedia cells carries a `recovered_answers.jsonl` sidecar, so the recovery "
      "overlay is a no-op uniformly across every arm here.")
    A("")
    A("| contrast | dataset | empty% Sieve@100 | empty% control |")
    A("|---|---|---:|---:|")
    for r in payload["comparisons"]:
        A(f"| {r['label']} | {r['dataset']} | {r['matched']['empty_pct_a']:.2f} | "
          f"{r['matched']['empty_pct_b']:.2f} |")
    A("")

    # per-comparison detail
    A("## 7. Per-comparison detail")
    A("")
    for r in payload["comparisons"]:
        A(f"**{r['label']} — {r['dataset']}**  ")
        A(f"A = `{r['cell_a']}` ({r['label_a']}), B = `{r['cell_b']}` ({r['label_b']}). "
          f"n_shared={r['matched']['n_shared']}, both arms configured `max_steps=100`.")
        A("")
        A("| basis | n | EM A | EM B | Δ | b | c | McNemar p | sig | tok A | tok B | % red | "
          "t p | Wilcoxon p | calls A | calls B | recall A | recall B | recall p |")
        A("|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|")
        for basis, d in (("UNMATCHED (published basis: Sieve@50 vs control@100)", r["unmatched"]),
                         ("**MATCHED (Sieve@100 vs control@100)**", r["matched"])):
            t, rc = d["tokens"], d["recall"]
            A(f"| {basis} | {d['n_shared']} | {d['em_a']:.1f} | {d['em_b']:.1f} | "
              f"{d['em_delta']:+.2f} | {d['b']} | {d['c']} | {fmt_p(d['p'])} | "
              f"{'yes' if d['sig'] else 'no'} | {fmt(t['tok_a'],0)} | {fmt(t['tok_b'],0)} | "
              f"{t['pct_reduction']:+.1f}% | {fmt_p(t.get('t_p'))} | {fmt_p(t.get('wilcoxon_p'))} | "
              f"{d['llm_calls_a']:.1f} | {d['llm_calls_b']:.1f} | {rc['recall_a']:.1f} | "
              f"{rc['recall_b']:.1f} | {fmt_p(rc['p'])} |")
        A("")
        A(f"{r['em_flag']}: {r['em_sentence']}")
        A("")
        A(f"{r['token_flag']}: {r['token_sentence']}")
        A("")
        A("---")
        A("")

    A("## 8. What this changes")
    A("")
    A("1. **The three engine/interface contrasts survive matching.** Sieve beats the hybrid "
      "sparse-dense control, the sparse-only control and the dense-only control at the same read "
      "interface and the same 100-step budget, on both Wikipedia datasets, at p < 1e-8 "
      "throughout. On MuSiQue the point estimates shrink by about a point "
      "(+5.5 -> +4.6, +5.7 -> +4.8, +6.8 -> +5.8); on HotpotQA they are unchanged or marginally "
      "larger. These are the paper's structured-surface claims and they hold.")
    A("2. **The dense-fusion rung is still undemonstrated on Wikipedia** — see A1. The previous "
      "text argued the Wikipedia nulls were 'confounded in the ablation's favour' and that "
      "Sieve was 'ahead on both datasets on half the budget'. That defence is now spent: given "
      "the budget it was missing, the gap does not open (HotpotQA +0.89, p=0.064; MuSiQue "
      "+0.33, p=0.694 — the MuSiQue estimate shrinks rather than grows). Dense fusion's "
      "separable contribution remains established on BrowseComp-Plus-Structured alone.")
    A("3. **The token-efficiency claim against the no-dense control was a budget artifact.** "
      "The published +6.5% / +9.8% token reductions were bought entirely by Sieve's smaller step "
      "cap; at matched budget the two arms are indistinguishable in cost (-0.1%, p=0.95; "
      "-1.1%, p=0.29 — Sieve numerically COSTLIER). Against the other three controls the "
      "advantage is real but smaller than published (roughly 17-24% rather than 22-32%).")
    A("4. **The step budget alone buys the method essentially nothing** (section 5): doubling "
      "the cap from 50 to 100 moves EM by +0.10 (p=0.861) on HotpotQA and -0.95 (p=0.219) on "
      "MuSiQue while spending 7-16 more LLM calls and 1k-2.6k more tokens per episode. So the "
      "paper's other cap-50 numbers were not materially hobbled — but the corollary is that "
      "Sieve's published token economy on Wikipedia is partly an artifact of stopping earlier, "
      "not only of reading less per step.")
    A("")
    return "\n".join(L)


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(str(type(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gate", action="store_true", help="run even if the sanity gate fails")
    args = ap.parse_args()

    datasets = WIKI + [BROWSE]
    qrels = {ds: load_qrels(ds) for ds in datasets}

    # ---- PART 0: provenance (cheap byte scan, cached) --------------------
    cap_cache = {}
    if CACHE_PATH.exists():
        try:
            cap_cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cap_cache = {}
    prov = []
    for ds in WIKI:
        for cell in [SIEVE100, SIEVE50, HYBRID, NODENSE, SPARSE, DENSESNIP]:
            pr = provenance(ds, cell, cap_cache)
            pr["expected_cap"] = EXPECTED_CAP[(cell, ds)]
            pr["cap_ok"] = (pr["uniform_cap"] == pr["expected_cap"])
            prov.append(pr)
            print(f"[prov] {ds} {cell[0]}/{cell[1]}: n={pr['n']} caps={pr['cap_dist']} "
                  f"expected={pr['expected_cap']} ok={pr['cap_ok']}", file=sys.stderr)
    try:
        CACHE_PATH.write_text(json.dumps(cap_cache))
    except OSError:
        pass
    cap_all_ok = all(p["cap_ok"] for p in prov)
    if not cap_all_ok and not args.no_gate:
        print("PROVENANCE CHECK FAILED -- a cell is not at its expected uniform cap. Stopping.",
              file=sys.stderr)
        JSON_PATH.write_text(json.dumps(dict(provenance=prov, cap_all_ok=False), indent=1,
                                        default=_json_default))
        return 2

    # ---- load cells ------------------------------------------------------
    mcache = {}
    for ds in datasets:
        for cell in LOAD[ds]:
            m, _cov, n = cell_metrics(ds, cell[0], cell[1], qrels[ds])
            if m is None:
                print(f"MISSING CELL {ds} {cell}", file=sys.stderr)
                return 3
            mcache[(ds, cell)] = m
            print(f"[cell] {ds} {cell[0]}/{cell[1]}: n={n}", file=sys.stderr)

    # ---- PART 1: sanity gate --------------------------------------------
    gates, gate_ok = [], True
    for g in GATES:
        a_m, b_m = mcache[(g["dataset"], g["a"])], mcache[(g["dataset"], g["b"])]
        n, em_a, em_b, delta, b, c, p = paired_em(a_m, b_m)
        ok = (abs(delta - g["em_delta"]) <= g["tol_delta"]
              and abs(p - g["p"]) <= g["rel_tol_p"] * g["p"])
        gate_ok &= ok
        gates.append(dict(name=g["name"], n=int(n), em_a=float(em_a), em_b=float(em_b),
                          em_delta=float(delta), b=int(b), c=int(c), p=float(p),
                          expected_delta=g["em_delta"], expected_p=g["p"], passed=bool(ok)))
        print(f"[gate] {g['name']}: n={n} delta={delta:+.2f} (exp {g['em_delta']:+.2f}) "
              f"p={p:.3g} (exp {g['p']:.3g}) -> {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    if not gate_ok and not args.no_gate:
        print("SANITY GATE FAILED -- stopping, no new numbers produced.", file=sys.stderr)
        JSON_PATH.write_text(json.dumps(dict(gate_passed=False, gates=gates, provenance=prov),
                                        indent=1, default=_json_default))
        return 2

    # ---- PART 2/3: the eight comparisons ---------------------------------
    comparisons = []
    for key, label, ctrl, published in CONTRASTS:
        for ds in WIKI:
            pub = published[ds]
            new_m = mcache[(ds, SIEVE100)]
            old_m = mcache[(ds, SIEVE50)]
            ctrl_m = mcache[(ds, ctrl)]
            matched = one_comparison(new_m, ctrl_m, LABEL[SIEVE100], LABEL[ctrl])
            unmatched = one_comparison(old_m, ctrl_m, LABEL[SIEVE50], LABEL[ctrl])
            flag, sentence = em_verdict(pub["em_delta"], pub["p"], matched)
            tflag, tsentence = token_verdict(unmatched["tokens"], matched["tokens"])
            # cross-check: does the recomputed unmatched basis reproduce the published constant?
            xcheck = abs(unmatched["em_delta"] - pub["em_delta"]) <= 0.15
            comparisons.append(dict(
                key=key, label=label, dataset=ds,
                cell_a=f"{SIEVE100[0]}/{SIEVE100[1]}", cell_b=f"{ctrl[0]}/{ctrl[1]}",
                label_a=LABEL[SIEVE100], label_b=LABEL[ctrl],
                published=pub, unmatched=unmatched, matched=matched,
                unmatched_reproduces_published=bool(xcheck),
                em_flag=flag, em_sentence=sentence,
                token_flag=tflag, token_sentence=tsentence,
            ))
            print(f"[cmp] {label} {ds}: matched delta={matched['em_delta']:+.2f} "
                  f"p={matched['p']:.3g} -> {flag} | tokens {tflag}", file=sys.stderr)

    # ---- budget sensitivity ---------------------------------------------
    budget = []
    for ds in WIKI:
        c = one_comparison(mcache[(ds, SIEVE100)], mcache[(ds, SIEVE50)],
                           LABEL[SIEVE100], LABEL[SIEVE50])
        budget.append(dict(dataset=ds, cmp=c))
        print(f"[budget] {ds}: EM {c['em_b']:.1f}@50 -> {c['em_a']:.1f}@100 "
              f"({c['em_delta']:+.2f}, p={c['p']:.3g})", file=sys.stderr)

    payload = dict(
        generated_by="analysis/matched_cap100_results.py",
        gate_passed=bool(gate_ok), gates=gates,
        cap_all_ok=bool(cap_all_ok), provenance=prov,
        judge_note="HotpotQA and MuSiQue are EM/F1 benchmarks and are NOT judged by design; "
                   "no judge numbers are computed for them.",
        comparisons=comparisons, budget_sensitivity=budget,
    )
    JSON_PATH.write_text(json.dumps(payload, indent=1, default=_json_default))
    MD_PATH.write_text(render_markdown(payload))
    print(f"wrote {JSON_PATH} and {MD_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
