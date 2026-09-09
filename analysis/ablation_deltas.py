#!/usr/bin/env python
"""Paired instance-level ablation deltas: FULL METHOD cell vs each comparison cell, per dataset.

Fixes a prose bug in the ablation section: earlier drafts quoted each ablation row's delta vs
the dataset's SERP-bm25 BASELINE (as tabulated by scripts/compare_cells.py, which always pairs
against the registry baseline) and mislabeled it as "vs full method". This script instead pairs
every comparison cell DIRECTLY against the full-method cell, on the instance set the two cells
actually share, using the exact same EM/recovery-overlay/McNemar machinery as compare_cells.py.

Reuses (does not reimplement):
  - agent_search.evaluation.metrics.answer_em            -- canonical EM, via scripts.compare_cells.metrics()
  - scripts.force_answer_backfill.load_rows_with_recovery -- recovery overlay (sidecar-safe)
  - scripts.compare_cells.metrics()          -- per-instance em/judge dict from overlaid rows
  - scripts.compare_cells.mcnemar_p()        -- exact two-sided McNemar on discordant pairs
  - scripts.compare_cells.load_qrels/load_judge_cache/cell_dir/cell_rows

Run: PYTHONPATH=. python analysis/ablation_deltas.py
Writes analysis/ablation_deltas.md and prints the results-section sentences for Finding 2.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_qrels, load_judge_cache, metrics, mcnemar_p, pct,
)

FULL = ("_headline_validation", "agent_research_bql_dense_snip")  # bql+dense+snip fetch

# dataset -> [(label, subdir, cond), ...] comparison cells (FULL excluded; added separately)
COMPARISONS = {
    "browsecomp_plus_structured": [
        ("No snippets (bql+dense fetch)", "_headline_validation", "agent_research_bql_dense_fetch"),
        ("No dense evidence (bql+snip fetch)", "_headline_validation", "agent_research_snip"),
        ("Hard Boolean filter + visit (bql visit)", "_fullvisit", "agent_research_bql_visit"),
        ("Sparse same interface (bm25+snip fetch)", "_headline_validation", "agent_research_bm25_fetch_snip"),
        ("Dense same interface (dense+snip fetch)", "_headline_validation", "agent_research_dense_fetch"),
        ("Dense no snippets (plain dense fetch)", "_headline_validation", "agent_research_dense_fetch_plain"),
    ],
    "hotpotqa_structured": [
        ("No snippets (bql+dense fetch)", "_headline_validation", "agent_research_bql_dense_fetch"),
        ("Sparse same interface (bm25+snip fetch)", "_headline_validation", "agent_research_bm25_fetch_snip"),
        ("Dense no snippets (plain dense fetch)", "_headline_validation", "agent_research_dense_fetch_plain"),
    ],
    "musique_structured": [
        ("No snippets (bql+dense fetch)", "_headline_validation", "agent_research_bql_dense_fetch"),
        ("Sparse same interface (bm25+snip fetch)", "_headline_validation", "agent_research_bm25_fetch_snip"),
        ("Dense no snippets (plain dense fetch)", "_headline_validation", "agent_research_dense_fetch_plain"),
    ],
}

JUDGE_COVERAGE_MIN = 0.90


def cell_metrics(dataset: str, subdir: str, cond: str, qrels: dict):
    """(metrics_dict, judge_coverage_frac) for one cell, or (None, None) if rows.jsonl absent."""
    rows = cell_rows(subdir, dataset, cond)
    if rows is None:
        return None, None
    cdir = cell_dir(subdir, dataset, cond)
    jc = load_judge_cache(cdir)
    m = metrics(rows, qrels, dataset, jc)
    n = len(m)
    njudged = sum(1 for v in m.values() if v["judge"] is not None)
    cov = (njudged / n) if n else 0.0
    return m, cov


def paired(full_m: dict, other_m: dict):
    """Paired EM comparison on the shared instance set. Returns
    (n_shared, em_full_pct, em_other_pct, delta, p) or None if no shared instances."""
    mut = set(full_m) & set(other_m)
    if not mut:
        return None
    b = sum(1 for i in mut if other_m[i]["em"] and not full_m[i]["em"])
    c = sum(1 for i in mut if full_m[i]["em"] and not other_m[i]["em"])
    em_full = pct([full_m[i]["em"] for i in mut])
    em_other = pct([other_m[i]["em"] for i in mut])
    return len(mut), em_full, em_other, em_full - em_other, mcnemar_p(b, c)


def paired_judge(full_m: dict, other_m: dict, full_cov: float, other_cov: float):
    """Paired judge-verdict comparison, only if BOTH cells clear JUDGE_COVERAGE_MIN judge
    coverage (matching compare_cells' own >=90% gate). Returns (n_jshared, j_full, j_other,
    jdelta, jp) or None (either: coverage gate failed, or no ids where both have a judge
    verdict -- both render as "pending" by the caller)."""
    if full_cov is None or other_cov is None:
        return None
    if full_cov < JUDGE_COVERAGE_MIN or other_cov < JUDGE_COVERAGE_MIN:
        return None
    mut = set(full_m) & set(other_m)
    jmut = [i for i in mut if full_m[i]["judge"] is not None and other_m[i]["judge"] is not None]
    if not jmut:
        return None
    jb = sum(1 for i in jmut if other_m[i]["judge"] and not full_m[i]["judge"])
    jc = sum(1 for i in jmut if full_m[i]["judge"] and not other_m[i]["judge"])
    j_full = pct([full_m[i]["judge"] for i in jmut])
    j_other = pct([other_m[i]["judge"] for i in jmut])
    return len(jmut), j_full, j_other, j_full - j_other, mcnemar_p(jb, jc)


def main():
    out_lines = ["# Ablation deltas: FULL METHOD vs each comparison cell (paired, per dataset)\n",
                 "Full method cell: `_headline_validation/.../agent_research_bql_dense_snip` "
                 "(bql+dense+snip fetch). Every row below is a DIRECT paired comparison against "
                 "this cell on the instance set the two cells share -- NOT vs the SERP-bm25 "
                 "baseline (that pairing is what `comparison_result.md` / compare_cells.py "
                 "reports and must not be conflated with this one).\n"]
    sentences = []

    for dataset, comps in COMPARISONS.items():
        qrels = load_qrels(dataset)
        full_m, full_cov = cell_metrics(dataset, FULL[0], FULL[1], qrels)
        if full_m is None:
            out_lines.append(f"\n## {dataset}\n\n(FULL METHOD cell missing -- skipped)\n")
            continue
        em_full_all = pct([v["em"] for v in full_m.values()])

        out_lines.append(f"\n## {dataset}\n")
        out_lines.append(f"FULL METHOD: n={len(full_m)}, EM%={em_full_all:.1f}, "
                          f"judge_coverage={full_cov*100:.1f}%\n")
        out_lines.append("| comparison | n_shared | EM_full | EM_other | delta | McNemar_p | judge_delta |")
        out_lines.append("|---|---:|---:|---:|---:|---:|---:|")

        for label, subdir, cond in comps:
            other_m, other_cov = cell_metrics(dataset, subdir, cond, qrels)
            if other_m is None:
                out_lines.append(f"| {label} | -- | | | | | (rows.jsonl missing) |")
                continue
            res = paired(full_m, other_m)
            if res is None:
                out_lines.append(f"| {label} | 0 | | | | | (no shared instances) |")
                continue
            n_shared, em_full, em_other, delta, p = res
            jres = paired_judge(full_m, other_m, full_cov, other_cov)
            if jres is not None:
                _, j_full, j_other, jdelta, jp = jres
                jstr = f"{jdelta:+.1f} (p={jp:.3g})"
            else:
                cov_note = []
                if full_cov < JUDGE_COVERAGE_MIN:
                    cov_note.append(f"full={full_cov*100:.0f}%")
                if other_cov < JUDGE_COVERAGE_MIN:
                    cov_note.append(f"other={other_cov*100:.0f}%")
                jstr = "pending" + (f" ({', '.join(cov_note)} judge cov)" if cov_note else " (no overlap)")
            out_lines.append(
                f"| {label} | {n_shared} | {em_full:.1f} | {em_other:.1f} | {delta:+.1f} | "
                f"{p:.3g} | {jstr} |")

            rel = (delta / em_other * 100.0) if em_other > 0 else float("nan")
            sentences.append(
                f"On {dataset}, the full method outperforms the {label.lower()} ablation by "
                f"{delta:.1f} EM points ({em_full:.1f}% vs {em_other:.1f}%, a {rel:.1f}% relative "
                f"gain; exact McNemar p={p:.3g}, n={n_shared})."
            )

    text = "\n".join(out_lines) + "\n"
    out_path = Path(__file__).resolve().parent / "ablation_deltas.md"
    out_path.write_text(text)
    print(text)
    print("## Finding 2 sentences (ARISE-style: absolute + relative delta + p)\n")
    for s in sentences:
        print("- " + s)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
