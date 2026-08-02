#!/usr/bin/env python
"""Round-5 reviewer fixes M5 and M3 (docs/reviews/round5_full.md).

M5 -- THE QUERY-SURFACE-ALONE ISOLATION. `comparison_result.md` never reports a direct paired
test between "bql+snip fetch" (structured query surface, no dense fusion) and "bm25+snip fetch"
(plain BM25, same snip+fetch interface) even though both cells are released. This script computes
that pair directly (cell A vs cell B, NOT either cell vs the dataset baseline -- that pairing is
what compare_cells.py / comparison_result.md already report and must not be conflated with this
one). It also computes the complementary isolation ("dense fusion alone": dense+snip fetch vs
bm25+snip fetch, same interface) where the cells exist.

M3 -- HOLM ON THE HEADLINE FAMILY. The paper Holm-corrects the 12-test ablation family but not the
"full method beats BM25-search-and-visit" family (Finding 1: 3 datasets x up to 2 metrics). This
script assembles that family from `comparison_result.md`'s winner row (bql+dense+snip fetch) on
each dataset, taking the p_em/p_judge columns as already computed by compare_cells.py, and applies
Holm-Bonferroni.

Reuses (does not reimplement):
  - evaluation.metrics.answer_em                          -- canonical EM, via compare_cells.metrics()
  - scripts.force_answer_backfill.load_rows_with_recovery  -- recovery overlay (sidecar-safe)
  - scripts.compare_cells.metrics()                        -- per-instance em/judge dict (uses
                                                                 gold_doc_recall internally too)
  - scripts.compare_cells.mcnemar_p()                      -- exact two-sided McNemar
  - scripts.compare_cells.load_qrels / load_judge_cache / cell_dir / cell_rows / pct

Streams via load_rows_with_recovery -> load_rows_tolerant (plain per-line json.loads over
rows.jsonl); the cells touched here are the browsecomp_plus_structured _headline_validation
snip-fetch family (~200-260MB each) -- NOT the 17-31GB auto-read cells, which this script never
opens.

Run: PYTHONPATH=. envs/bin/python analysis/round5_analyses.py
Writes analysis/round5_analyses.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_qrels, load_judge_cache, metrics, mcnemar_p, pct,
)

JUDGE_COVERAGE_MIN = 0.90
ALPHA = 0.05

# ================================================================================================
# M5: query-surface-alone / dense-fusion-alone isolation
# ================================================================================================

# Every cell below lives under runs/_headline_validation/agent/<dataset>/<MODEL>/<cond>, sharing
# the same snippet-bearing listing + section-fetch read interface; datasets differ only in whether
# the corresponding "agent_research_snip" (bql+snip fetch, no dense) / "agent_research_dense_fetch"
# (pure dense, no dense-fused-BQL) cells were actually run for that dataset -- checked at runtime,
# not assumed (both are ABSENT for hotpotqa_structured/musique_structured; grep of runs/ confirms
# only browsecomp_plus_structured has them, 2026-07-22).
QUERY_SURFACE_ISOLATIONS = [
    dict(
        key="query_surface",
        title="Query surface alone: structured (BQL) vs plain BM25, same snip+fetch interface, no dense fusion",
        isolates=(
            "Whether the field-tagged Boolean query SURFACE itself (vs plain keyword BM25) helps, "
            "holding the read interface (snippet-bearing listing + named-section fetch) and the "
            "absence of dense fusion fixed on BOTH sides. Both cells share: snippet display, "
            "section-fetch reading, and no dense evidence at all."
        ),
        does_not_isolate=(
            "Anything about dense fusion (neither side has it), the 'visit whole document' reading "
            "style (neither side uses it -- both fetch named sections), or listing size (both are "
            "the same shape). It also does not tell you whether BQL's Boolean filter vs BQL's "
            "field-tag vocabulary specifically drives any gap -- BQL bundles both."
        ),
        A=("bql+snip fetch", "agent_research_snip"),
        B=("bm25+snip fetch", "agent_research_bm25_fetch_snip"),
    ),
    dict(
        key="dense_fusion",
        title="Dense fusion alone: pure dense vs plain BM25, same snip+fetch interface, no structured query",
        isolates=(
            "Whether swapping the query engine from plain BM25 to pure dense (embedding) retrieval "
            "helps, holding the read interface (snippet listing + section-fetch) fixed and the BQL "
            "structured query surface absent from BOTH sides. This is the complementary isolation "
            "to the query-surface test above: together the two say what each single ingredient "
            "(structured query surface; dense retrieval) is worth ALONE at this interface, before "
            "either is fused with the other (as the winning bql+dense+snip fetch cell does)."
        ),
        does_not_isolate=(
            "Anything about the BQL structured query surface (neither side has it) or about dense "
            "fusion INSIDE a structured query (that is 'bql+dense+snip fetch' vs 'bql+snip fetch', "
            "already reported by analysis/ablation_deltas.py, not this pair)."
        ),
        A=("dense+snip fetch", "agent_research_dense_fetch"),
        B=("bm25+snip fetch", "agent_research_bm25_fetch_snip"),
    ),
]

DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]
SUBDIR = "_headline_validation"


def cell_metrics(dataset: str, cond: str, qrels: dict):
    """(metrics_dict, judge_coverage_frac) for one cell, or (None, None) if rows.jsonl absent."""
    rows = cell_rows(SUBDIR, dataset, cond)
    if rows is None:
        return None, None
    cdir = cell_dir(SUBDIR, dataset, cond)
    jc = load_judge_cache(cdir)
    m = metrics(rows, qrels, dataset, jc)
    n = len(m)
    njudged = sum(1 for v in m.values() if v["judge"] is not None)
    cov = (njudged / n) if n else 0.0
    return m, cov


def paired_em(m_a: dict, m_b: dict):
    """(n_shared, em_a_pct, em_b_pct, delta_a_minus_b, mcnemar_p) on the shared instance set."""
    mut = set(m_a) & set(m_b)
    if not mut:
        return None
    b = sum(1 for i in mut if m_a[i]["em"] and not m_b[i]["em"])
    c = sum(1 for i in mut if m_b[i]["em"] and not m_a[i]["em"])
    em_a = pct([m_a[i]["em"] for i in mut])
    em_b = pct([m_b[i]["em"] for i in mut])
    return len(mut), em_a, em_b, em_a - em_b, mcnemar_p(b, c)


def paired_judge(m_a: dict, m_b: dict, cov_a: float, cov_b: float):
    """Same shape as paired_em but for the judge verdict, gated on BOTH cells clearing
    JUDGE_COVERAGE_MIN (matching compare_cells' own >=90% gate for rendering a judge% at all).
    Returns None if either side's coverage is below the gate, or if no shared id has a judge
    verdict on both sides -- both cases render as 'pending' by the caller."""
    if cov_a is None or cov_b is None or cov_a < JUDGE_COVERAGE_MIN or cov_b < JUDGE_COVERAGE_MIN:
        return None
    mut = set(m_a) & set(m_b)
    jmut = [i for i in mut if m_a[i]["judge"] is not None and m_b[i]["judge"] is not None]
    if not jmut:
        return None
    jb = sum(1 for i in jmut if m_a[i]["judge"] and not m_b[i]["judge"])
    jc = sum(1 for i in jmut if m_b[i]["judge"] and not m_a[i]["judge"])
    j_a = pct([m_a[i]["judge"] for i in jmut])
    j_b = pct([m_b[i]["judge"] for i in jmut])
    return len(jmut), j_a, j_b, j_a - j_b, mcnemar_p(jb, jc)


def run_m5():
    out = ["# M5 -- the query-surface-alone isolation (paired, direct cell-vs-cell)\n"]
    out.append(
        "Every pair below is a DIRECT paired comparison between two non-baseline, non-full-method "
        "cells on the instance set they share -- neither side is the dataset's SERP-bm25 baseline "
        "(that pairing is what `comparison_result.md` already reports) nor the full method "
        "(that pairing is what `analysis/ablation_deltas.py` already reports). This is the specific "
        "pairing round5_full.md's M5 says is never reported despite both cells existing.\n"
    )
    sentences = []
    availability_notes = []

    for iso in QUERY_SURFACE_ISOLATIONS:
        out.append(f"\n## {iso['title']}\n")
        out.append(f"**Isolates:** {iso['isolates']}\n")
        out.append(f"**Does NOT isolate:** {iso['does_not_isolate']}\n")
        out.append("| dataset | n_shared | EM_A% | EM_B% | ΔEM (A-B) | McNemar p_em | judge_A% | judge_B% | Δjudge | McNemar p_judge |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        (label_a, cond_a), (label_b, cond_b) = iso["A"], iso["B"]
        for dataset in DATASETS:
            qrels = load_qrels(dataset)
            m_a, cov_a = cell_metrics(dataset, cond_a, qrels)
            m_b, cov_b = cell_metrics(dataset, cond_b, qrels)
            if m_a is None or m_b is None:
                missing = []
                if m_a is None:
                    missing.append(f"`{cond_a}` ({label_a})")
                if m_b is None:
                    missing.append(f"`{cond_b}` ({label_b})")
                out.append(f"| {dataset} | -- | | | | | | | | (cell not released: {', '.join(missing)}) |")
                availability_notes.append(
                    f"{iso['key']} on {dataset}: not computable -- {', '.join(missing)} cell(s) "
                    f"not present under runs/{SUBDIR}/agent/{dataset}/.../."
                )
                continue

            res = paired_em(m_a, m_b)
            if res is None:
                out.append(f"| {dataset} | 0 | | | | | | | | (no shared instances) |")
                continue
            n_shared, em_a, em_b, delta, p = res
            jres = paired_judge(m_a, m_b, cov_a, cov_b)
            if jres is not None:
                _, j_a, j_b, jdelta, jp = jres
                j_a_s, j_b_s, jd_s, jp_s = f"{j_a:.1f}", f"{j_b:.1f}", f"{jdelta:+.1f}", f"{jp:.3g}"
            else:
                cov_note = []
                if cov_a is not None and cov_a < JUDGE_COVERAGE_MIN:
                    cov_note.append(f"{label_a}={cov_a*100:.0f}%")
                if cov_b is not None and cov_b < JUDGE_COVERAGE_MIN:
                    cov_note.append(f"{label_b}={cov_b*100:.0f}%")
                note = f"pending (judge cov: {', '.join(cov_note)})" if cov_note else "pending (no overlap)"
                j_a_s = j_b_s = jd_s = jp_s = note

            out.append(
                f"| {dataset} | {n_shared} | {em_a:.1f} | {em_b:.1f} | {delta:+.1f} | {p:.3g} | "
                f"{j_a_s} | {j_b_s} | {jd_s} | {jp_s} |"
            )

            if dataset == "browsecomp_plus_structured":
                rel = (delta / em_b * 100.0) if em_b else float("nan")
                sig = "statistically significant" if p < ALPHA else "not statistically significant"
                sentence = (
                    f"{iso['title'].split(':')[0]}: on browsecomp_plus_structured, {label_a} scores "
                    f"{em_a:.1f}% EM vs {label_b}'s {em_b:.1f}% EM on the {n_shared} shared instances "
                    f"({delta:+.1f} pts, {rel:+.1f}% relative; exact McNemar p={p:.3g}, {sig})"
                )
                if jres is not None:
                    sentence += f"; judge accuracy moves {jdelta:+.1f} pts (p={jp:.3g})."
                else:
                    sentence += "; judge-accuracy comparison pending (coverage gate not cleared)."
                sentences.append(sentence)

    out.append("\n## Availability notes\n")
    if availability_notes:
        for n in availability_notes:
            out.append(f"- {n}")
    else:
        out.append("- All pairs computable for all three datasets.")
    out.append(
        "\nOnly `browsecomp_plus_structured` has both `agent_research_snip` (bql+snip fetch) and "
        "`agent_research_dense_fetch` (dense+snip fetch, non-plain) released under "
        "`runs/_headline_validation/`; `hotpotqa_structured` and `musique_structured` only have "
        "`agent_research_dense_fetch_plain` (no-snippet dense) at this interface, which is a "
        "DIFFERENT interface (no per-result excerpt) from `agent_research_bm25_fetch_snip` (has "
        "the excerpt) -- so a same-interface dense-vs-sparse pair does not exist for the wiki "
        "datasets in the released cells, and neither isolation is fabricated for them."
    )

    return "\n".join(out), sentences


# ================================================================================================
# M3: Holm-Bonferroni on the headline (Finding 1) family
# ================================================================================================

# (dataset, metric, delta, p) -- read verbatim from comparison_result.md's winner row
# ("bql+dense+snip fetch") on each dataset, vs that dataset's "SERP bm25 [BASELINE]" row (the
# BM25-search-and-visit baseline). Not recomputed here (comparison_result.md's own Δ/p columns
# ARE this exact paired McNemar computation, produced by compare_cells.py from the same
# primitives this file reuses for M5) -- copied so the correction step is transparent and citable
# against the specific markdown table cells the paper draws from.
HEADLINE_FAMILY_RAW = [
    ("browsecomp_plus_structured", "EM", 2.5, 0.217),
    ("browsecomp_plus_structured", "judge", 2.0, 0.327),
    ("hotpotqa_structured", "EM", 1.6, 0.00186),
    ("hotpotqa_structured", "judge", 0.1, 0.79),
    ("musique_structured", "EM", 2.7, 0.00073),
    ("musique_structured", "judge", 2.2, 0.0123),
]


def holm_bonferroni(pvals: list, alpha: float = ALPHA):
    """Standard Holm-Bonferroni step-down: sort ascending, reject p_(i) while
    p_(i) <= alpha/(m+1-i) (1-indexed) AND all smaller-p tests were also rejected (step-down --
    the first non-rejection stops the whole cascade). Returns a list of (orig_index, p, threshold,
    reject: bool) in SORTED order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = []
    still_going = True
    for rank, idx in enumerate(order, start=1):
        thresh = alpha / (m + 1 - rank)
        p = pvals[idx]
        reject = still_going and (p <= thresh)
        if not reject:
            still_going = False  # step-down: once one fails, all subsequent (larger p) fail too
        out.append((idx, p, thresh, reject))
    return out


def run_m3():
    out = ["# M3 -- Holm-Bonferroni on the headline (Finding 1) family\n"]
    m = len(HEADLINE_FAMILY_RAW)
    out.append(
        f"**Family (m={m}):** full method (bql+dense+snip fetch) vs the BM25-search-and-visit "
        "baseline (SERP bm25 [BASELINE]), on each of the 3 datasets this paper reports, for both "
        "EM and judge accuracy where a judge number exists in `comparison_result.md`. "
        f"m={m} (not 5, as the round5 review's prose count assumed): the round-5 draft's TEXT "
        "describes 'five reported p-values' because it treats HotpotQA judge accuracy as having "
        "\"no reported significance test,\" but `comparison_result.md` (compare_cells.py's own "
        "output, computed from the same paired-McNemar machinery as every other cell in that "
        "table) already carries a p_judge value for that row (p=0.79) -- the datum exists even "
        "though the paper's prose never quotes it. We include it: the fairest reading of "
        "\"the family carrying the central claim\" is every comparison compare_cells.py actually "
        "computed for Finding 1, not only the ones the current draft's text happens to narrate.\n"
    )

    pvals = [p for _, _, _, p in HEADLINE_FAMILY_RAW]
    holm = holm_bonferroni(pvals, ALPHA)

    out.append("## Family, raw p-values (as tabulated)\n")
    out.append("| dataset | metric | Δ (full - baseline) | raw p |")
    out.append("|---|---|---:|---:|")
    for ds, metric, delta, p in HEADLINE_FAMILY_RAW:
        out.append(f"| {ds} | {metric} | {delta:+.1f} | {p:.3g} |")

    out.append("\n## Holm-Bonferroni step-down (α=0.05, m=" + str(m) + ")\n")
    out.append("Sorted ascending; reject while p₍ᵢ₎ ≤ α/(m+1-i) AND every smaller-p test also rejected.\n")
    out.append("| rank i | dataset | metric | raw p | threshold α/(m+1-i) | survives Holm? |")
    out.append("|---:|---|---|---:|---:|---|")
    survivors, fallers = [], []
    for rank, (idx, p, thresh, reject) in enumerate(holm, start=1):
        ds, metric, delta, _ = HEADLINE_FAMILY_RAW[idx]
        verdict = "**survives**" if reject else "falls"
        out.append(f"| {rank} | {ds} | {metric} | {p:.3g} | {thresh:.5f} | {verdict} |")
        (survivors if reject else fallers).append((ds, metric, delta, p))

    # Cross-check against statsmodels if available.
    sm_line = ""
    try:
        from statsmodels.stats.multitest import multipletests
        sm_reject, sm_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
        agree = all(
            bool(sm_reject[idx]) == reject for idx, _, _, reject in holm
        )
        agree_note = ("Matches this script's hand-rolled Holm decisions exactly." if agree
                      else "DISAGREES with this script -- investigate before reporting.")
        sm_line = (
            f"\n**statsmodels cross-check:** `multipletests(pvals, alpha=0.05, method='holm')` "
            f"reject mask = {list(map(bool, sm_reject))} (dataset order: "
            f"{[f'{d}/{me}' for d, me, _, _ in HEADLINE_FAMILY_RAW]}); "
            f"adjusted p-values = {[f'{x:.4g}' for x in sm_adj]}. "
            f"{agree_note}\n"
        )
    except ImportError:
        sm_line = "\n**statsmodels cross-check:** statsmodels not importable in this env; hand-rolled Holm only.\n"
    out.append(sm_line)

    out.append("## Consequence for the paper's claims\n")
    n_uncorrected_sig = sum(1 for _, _, _, p in HEADLINE_FAMILY_RAW if p < ALPHA)
    out.append(
        f"- **{len(survivors)}/{m} comparisons survive Holm correction**: "
        + ", ".join(f"{ds} {metric}" for ds, metric, _, _ in survivors) + ".\n"
        f"- **{len(fallers)}/{m} fall**: " + ", ".join(f"{ds} {metric}" for ds, metric, _, _ in fallers) + ".\n"
        f"- Before any correction, {n_uncorrected_sig}/{m} comparisons already cleared uncorrected "
        f"α=0.05; Holm correction does not change WHICH comparisons clear threshold in this family "
        f"(same {len(survivors)} survive both corrected and uncorrected) -- consistent with the "
        "round-5 review's own quick-check comment that Holm 'does not change which comparisons "
        "clear threshold.'\n"
        "- The honest consequence is not that correction flips a result, but what it leaves "
        "unmasked: **both flagship BrowseComp-Plus comparisons (EM p=0.217, judge p=0.327) are "
        "non-significant even BEFORE any multiple-comparison correction is applied** -- Holm only "
        "confirms this, it does not cause it. The paper's only comparisons that clear a "
        "family-wise-corrected 0.05 threshold are on the two Wikipedia multi-hop datasets "
        "(HotpotQA EM, MuSiQue EM, MuSiQue judge), not on the paper's flagship benchmark. The "
        "musique judge comparison (raw p=0.0123) survives by a narrow margin (Holm threshold at "
        "that rank is 0.0125) -- a single additional non-significant test folded into this family, "
        "or a slightly different m, could flip it, so this survival should be reported as fragile, "
        "not as a robust anchor.\n"
    )

    sentence = (
        f"Applying Holm-Bonferroni (α=0.05) to the full m={m}-test headline family (3 datasets x "
        f"EM + judge accuracy) leaves {len(survivors)} comparisons significant -- "
        + ", ".join(f"{ds.replace('_structured','').replace('_plus','+')} {metric}" for ds, metric, _, _ in survivors)
        + f" -- and confirms (rather than newly reveals) that neither BrowseComp-Plus comparison "
        "(the paper's flagship benchmark) reaches significance at any corrected or uncorrected "
        f"threshold in this family (EM p=0.217, judge p=0.327)."
    )

    return "\n".join(out), [sentence]


def main():
    m5_md, m5_sentences = run_m5()
    m3_md, m3_sentences = run_m3()

    text = (
        "# Round-5 analyses: M5 (query-surface isolation) and M3 (Holm on headline family)\n\n"
        "Generated by `analysis/round5_analyses.py`. Addresses docs/reviews/round5_full.md items "
        "M5 and M3.\n\n---\n\n"
        + m5_md + "\n\n---\n\n" + m3_md + "\n\n---\n\n"
        "## Lift-able sentences\n\n### M5\n\n" + "\n\n".join(f"- {s}" for s in m5_sentences)
        + "\n\n### M3\n\n" + "\n\n".join(f"- {s}" for s in m3_sentences) + "\n"
    )

    out_path = Path(__file__).resolve().parent / "round5_analyses.md"
    out_path.write_text(text)
    print(text)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
