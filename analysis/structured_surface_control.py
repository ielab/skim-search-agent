#!/usr/bin/env python
"""Paired comparison: Sieve (full method) vs the strongest non-structured control at the SAME
read interface -- hybrid sparse-dense RRF with snippets and section fetch.

This contrast isolates the structured QUERY SURFACE: Sieve's structured query language (bql)
plus dense retrieval is replaced by plain BM25+dense RRF (hybrid), while the READ interface
(snippet preview, section-level fetch) is held fixed identically across both cells. Both cells:

    dataset : browsecomp_plus_structured
    tier    : _headline_validation
    n       : 830
    FULL    (Sieve)   : agent_research_bql_dense_snip
    CONTROL (hybrid)  : agent_research_hybrid_fetch_snip

Requested ad hoc by a reviewer; this script makes it a reproducible artifact before the paper
cites it. Reuses -- does NOT reimplement -- the repo's own primitives, exactly as
analysis/ablation_deltas.py does:
    - scripts.compare_cells.metrics()          -- per-instance em/judge/tok/recall dict, built on
                                                    evaluation.metrics.answer_em and the recovery
                                                    overlay (scripts.force_answer_backfill.
                                                    load_rows_with_recovery, via cell_rows())
    - scripts.compare_cells.mcnemar_p()        -- exact two-sided McNemar on discordant pairs
    - scripts.compare_cells.cell_dir/cell_rows/load_judge_cache/load_qrels/pct

Before trusting the new number, this script first REPRODUCES an already-published contrast
(Sieve vs "No dense evidence", i.e. agent_research_snip, on the same dataset/tier) and checks it
against the published +4.7 EM / p=0.0104 (see analysis/ablation_deltas.md). If that check fails,
the script stops and reports the mismatch instead of printing the new comparison as if it were
trustworthy.

Run: PYTHONPATH=. envs/bin/python analysis/structured_surface_control.py
Writes: analysis/structured_surface_control_data.json, analysis/structured_surface_control.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_qrels, load_judge_cache, metrics, mcnemar_p, pct,
)

DATASET = "browsecomp_plus_structured"
TIER = "_headline_validation"
FULL_COND = "agent_research_bql_dense_snip"        # Sieve
CONTROL_COND = "agent_research_hybrid_fetch_snip"   # hybrid sparse-dense RRF, snip+fetch
SANITY_COND = "agent_research_snip"                 # "No dense evidence" -- published check

JUDGE_COVERAGE_MIN = 0.90

# Published reference (analysis/ablation_deltas.md, "No dense evidence (bql+snip fetch)" row):
# n=830, EM full=38.9, EM other=34.2, delta=+4.7, McNemar p=0.0104.
SANITY_EXPECTED = dict(n=830, em_full=38.9, em_other=34.2, delta=4.7, p=0.0104)
SANITY_TOL = dict(em=0.15, delta=0.15, p=0.002)  # rounding tolerance vs the published 1-dp/3-sig-fig text


def cell_metrics(dataset: str, subdir: str, cond: str, qrels: dict):
    """(metrics_dict, judge_coverage_frac, n_rows) for one cell, or (None, None, 0) if missing."""
    rows = cell_rows(subdir, dataset, cond)
    if rows is None:
        return None, None, 0
    cdir = cell_dir(subdir, dataset, cond)
    jc = load_judge_cache(cdir)
    m = metrics(rows, qrels, dataset, jc)
    n = len(m)
    njudged = sum(1 for v in m.values() if v["judge"] is not None)
    cov = (njudged / n) if n else 0.0
    return m, cov, n


def paired_em(a_m: dict, b_m: dict):
    """(n_shared, em_a, em_b, delta, b_count, c_count, p) on the shared instance set.
    b_count = instances where B is right and A is wrong; c_count = A right, B wrong
    (mirrors ablation_deltas.paired's b/c convention: b favors the 'other' cell)."""
    mut = sorted(set(a_m) & set(b_m))
    b = sum(1 for i in mut if b_m[i]["em"] and not a_m[i]["em"])
    c = sum(1 for i in mut if a_m[i]["em"] and not b_m[i]["em"])
    em_a = pct([a_m[i]["em"] for i in mut])
    em_b = pct([b_m[i]["em"] for i in mut])
    return len(mut), em_a, em_b, em_a - em_b, b, c, mcnemar_p(b, c)


def paired_judge(a_m: dict, b_m: dict, a_cov: float, b_cov: float):
    """Same shape as paired_em but restricted to ids where BOTH cells have a judge verdict, and
    only computed at all if both cells clear the >=90% judge-coverage gate used throughout this
    codebase. Returns None if the gate fails or there's no judged overlap."""
    if a_cov is None or b_cov is None or a_cov < JUDGE_COVERAGE_MIN or b_cov < JUDGE_COVERAGE_MIN:
        return None
    mut = sorted(set(a_m) & set(b_m))
    jmut = [i for i in mut if a_m[i]["judge"] is not None and b_m[i]["judge"] is not None]
    if not jmut:
        return None
    b = sum(1 for i in jmut if b_m[i]["judge"] and not a_m[i]["judge"])
    c = sum(1 for i in jmut if a_m[i]["judge"] and not b_m[i]["judge"])
    j_a = pct([a_m[i]["judge"] for i in jmut])
    j_b = pct([b_m[i]["judge"] for i in jmut])
    return len(jmut), j_a, j_b, j_a - j_b, b, c, mcnemar_p(b, c)


def mean_tok_recall(m: dict, ids):
    ids = list(ids)
    if not ids:
        return float("nan"), float("nan")
    tok = sum(m[i]["tok"] for i in ids) / len(ids)
    rec = 100.0 * sum(1 for i in ids if m[i]["recall"]) / len(ids)
    return tok, rec


def run_sanity_check(qrels: dict) -> dict:
    """Reproduce the published Sieve vs 'No dense evidence' contrast. Returns a result dict with
    a `passed` bool; does not raise -- caller decides whether to proceed."""
    full_m, full_cov, full_n = cell_metrics(DATASET, TIER, FULL_COND, qrels)
    sanity_m, sanity_cov, sanity_n = cell_metrics(DATASET, TIER, SANITY_COND, qrels)
    if full_m is None or sanity_m is None:
        return dict(passed=False, reason="rows.jsonl missing for FULL or sanity cell")
    n_shared, em_full, em_other, delta, b, c, p = paired_em(full_m, sanity_m)
    ok = (
        n_shared == SANITY_EXPECTED["n"]
        and abs(em_full - SANITY_EXPECTED["em_full"]) <= SANITY_TOL["em"]
        and abs(em_other - SANITY_EXPECTED["em_other"]) <= SANITY_TOL["em"]
        and abs(delta - SANITY_EXPECTED["delta"]) <= SANITY_TOL["delta"]
        and abs(p - SANITY_EXPECTED["p"]) <= SANITY_TOL["p"]
    )
    return dict(
        passed=bool(ok), n_shared=n_shared, em_full=em_full, em_other=em_other, delta=delta,
        b=b, c=c, p=p, expected=SANITY_EXPECTED,
    )


def main():
    qrels = load_qrels(DATASET)

    sanity = run_sanity_check(qrels)
    print("=== Sanity check: Sieve vs 'No dense evidence' (published: +4.7 EM, p=0.0104) ===")
    if sanity.get("passed"):
        print(f"PASS: n={sanity['n_shared']}, EM {sanity['em_full']:.1f} vs {sanity['em_other']:.1f}, "
              f"delta={sanity['delta']:+.1f}, b={sanity['b']}, c={sanity['c']}, p={sanity['p']:.3g}\n")
    else:
        print(f"FAIL / could not reproduce: {sanity}\n")
        print("Stopping: refusing to report the new comparison until the pipeline reproduces the "
              "known-published number. See output above for the discrepancy.")
        sys.exit(1)

    # --- the actual requested comparison -----------------------------------------------------
    full_m, full_cov, full_n = cell_metrics(DATASET, TIER, FULL_COND, qrels)
    ctrl_m, ctrl_cov, ctrl_n = cell_metrics(DATASET, TIER, CONTROL_COND, qrels)
    if full_m is None or ctrl_m is None:
        print("FULL or CONTROL cell rows.jsonl missing -- aborting.")
        sys.exit(1)

    mut = sorted(set(full_m) & set(ctrl_m))
    n_shared, em_full, em_ctrl, em_delta, em_b, em_c, em_p = paired_em(full_m, ctrl_m)

    jres = paired_judge(full_m, ctrl_m, full_cov, ctrl_cov)

    tok_full, rec_full = mean_tok_recall(full_m, mut)
    tok_ctrl, rec_ctrl = mean_tok_recall(ctrl_m, mut)

    result = dict(
        dataset=DATASET, tier=TIER,
        full_cond=FULL_COND, control_cond=CONTROL_COND,
        full_label="Sieve (agent_research_bql_dense_snip)",
        control_label="hybrid sparse-dense RRF + snip + fetch (agent_research_hybrid_fetch_snip)",
        n_full=full_n, n_control=ctrl_n, n_shared=n_shared,
        judge_coverage_full=full_cov, judge_coverage_control=ctrl_cov,
        judge_coverage_gate=JUDGE_COVERAGE_MIN,
        em=dict(full=em_full, control=em_ctrl, delta=em_delta, b=em_b, c=em_c, p=em_p,
                 significant_at_0_05=bool(em_p < 0.05)),
        tokens_once_mean=dict(full=tok_full, control=tok_ctrl),
        gold_doc_recall_pct=dict(full=rec_full, control=rec_ctrl),
        sanity_check=sanity,
        expected_ad_hoc=dict(
            em_full=38.9, em_control=35.7, em_delta=3.3, em_b=140, em_c=113, em_p=0.102,
            judge_full=41.3, judge_control=38.9, judge_delta=2.4, judge_b=145, judge_c=125,
            judge_p=0.248, tok_full=45000, tok_control=44000,
            recall_full=61.2, recall_control=71.1,
        ),
    )
    if jres is not None:
        j_n, j_full, j_ctrl, j_delta, j_b, j_c, j_p = jres
        result["judge"] = dict(n_shared=j_n, full=j_full, control=j_ctrl, delta=j_delta,
                                b=j_b, c=j_c, p=j_p, significant_at_0_05=bool(j_p < 0.05))
    else:
        result["judge"] = None
        result["judge_note"] = (
            f"judge coverage gate (>={JUDGE_COVERAGE_MIN*100:.0f}%) not met or no judged overlap: "
            f"full_cov={full_cov}, control_cov={ctrl_cov}"
        )

    def _json_default(o):
        # scipy's binomtest pvalue is numpy.float64 / numpy.bool_ -- not stdlib-JSON-serializable.
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    out_json = Path(__file__).resolve().parent / "structured_surface_control_data.json"
    out_json.write_text(json.dumps(result, indent=2, default=_json_default))

    # --- markdown report ------------------------------------------------------------------------
    lines = []
    lines.append("# Structured query surface control: Sieve vs hybrid sparse-dense RRF (same read interface)\n")
    lines.append(
        f"Isolates the structured query surface: **Sieve** (`{FULL_COND}`) vs the strongest "
        f"non-structured control at the *same* read interface -- plain BM25+dense RRF with "
        f"snippets and section fetch (`{CONTROL_COND}`). Both on `{DATASET}`, tier `{TIER}`, "
        f"n={full_n} / n={ctrl_n} (n_shared={n_shared} after inner join on instance_id).\n"
    )
    lines.append(
        f"**Sanity check** (must reproduce a published number before this comparison is "
        f"trusted): Sieve vs \"No dense evidence\" (`{SANITY_COND}`) reproduced at "
        f"n={sanity['n_shared']}, EM {sanity['em_full']:.1f} vs {sanity['em_other']:.1f}, "
        f"delta={sanity['delta']:+.1f}, p={sanity['p']:.3g} "
        f"(published: +4.7 EM, p=0.0104) -- **{'PASS' if sanity['passed'] else 'FAIL'}**.\n"
    )
    lines.append("## Results\n")
    lines.append("| metric | Sieve (full) | hybrid+snip+fetch (control) | delta (full-control) | b | c | McNemar p | sig @ 0.05 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| EM % | {em_full:.1f} | {em_ctrl:.1f} | {em_delta:+.1f} | {em_b} | {em_c} | "
        f"{em_p:.3g} | {'yes' if em_p < 0.05 else 'no'} |"
    )
    if jres is not None:
        j_n, j_full, j_ctrl, j_delta, j_b, j_c, j_p = jres
        lines.append(
            f"| Judge accuracy % | {j_full:.1f} | {j_ctrl:.1f} | {j_delta:+.1f} | {j_b} | {j_c} | "
            f"{j_p:.3g} | {'yes' if j_p < 0.05 else 'no'} |"
        )
    else:
        lines.append(f"| Judge accuracy % | -- | -- | -- | -- | -- | -- | judge gate/coverage failed |")
    lines.append(
        f"| Mean count-once tokens/episode | {tok_full:,.0f} | {tok_ctrl:,.0f} | "
        f"{tok_full - tok_ctrl:+,.0f} | | | | |"
    )
    lines.append(
        f"| Mean gold-doc recall % | {rec_full:.1f} | {rec_ctrl:.1f} | {rec_full - rec_ctrl:+.1f} | | | | |"
    )
    lines.append("")
    lines.append(f"Judge coverage: full={full_cov*100:.1f}%, control={ctrl_cov*100:.1f}% "
                  f"(gate: >= {JUDGE_COVERAGE_MIN*100:.0f}%).\n")
    lines.append(
        "Convention (matches `ablation_deltas.py`): `b` = instances the control cell got right "
        "and Sieve got wrong; `c` = instances Sieve got right and the control got wrong. McNemar's "
        "test is symmetric in {b, c} so `p` is unaffected by labeling, but note the reviewer's ad "
        "hoc b/c labels below are swapped relative to this convention (140/113 vs this script's "
        "113/140 for EM) -- the underlying discordant-pair counts {113, 140} match exactly; only "
        "which one is called b vs c differs.\n"
    )

    lines.append("## Comparison to the reviewer's ad hoc numbers\n")
    lines.append("| metric | ad hoc (reviewer) | this script |")
    lines.append("|---|---:|---:|")
    lines.append(f"| EM full | 38.9 | {em_full:.1f} |")
    lines.append(f"| EM control | 35.7 | {em_ctrl:.1f} |")
    lines.append(f"| EM delta | +3.3 | {em_delta:+.1f} |")
    lines.append(f"| EM b/c | 140/113 | {em_b}/{em_c} |")
    lines.append(f"| EM McNemar p | 0.102 | {em_p:.3g} |")
    if jres is not None:
        lines.append(f"| Judge full | 41.3 | {j_full:.1f} |")
        lines.append(f"| Judge control | 38.9 | {j_ctrl:.1f} |")
        lines.append(f"| Judge delta | +2.4 | {j_delta:+.1f} |")
        lines.append(f"| Judge b/c | 145/125 | {j_b}/{j_c} |")
        lines.append(f"| Judge McNemar p | 0.248 | {j_p:.3g} |")
    lines.append(f"| Mean tokens full | ~45k | {tok_full:,.0f} |")
    lines.append(f"| Mean tokens control | ~44k | {tok_ctrl:,.0f} |")
    lines.append(f"| Mean recall full | 61.2% | {rec_full:.1f}% |")
    lines.append(f"| Mean recall control | 71.1% | {rec_ctrl:.1f}% |")
    lines.append("")

    lines.append("## Significance statements\n")
    lines.append(
        f"- EM delta ({em_delta:+.1f} points, Sieve {em_full:.1f}% vs control {em_ctrl:.1f}%, "
        f"McNemar p={em_p:.3g}, n={n_shared}) is "
        f"{'STATISTICALLY SIGNIFICANT' if em_p < 0.05 else 'NOT statistically significant'} at alpha=0.05."
    )
    if jres is not None:
        lines.append(
            f"- Judge-accuracy delta ({j_delta:+.1f} points, Sieve {j_full:.1f}% vs control "
            f"{j_ctrl:.1f}%, McNemar p={j_p:.3g}, n={j_n}) is "
            f"{'STATISTICALLY SIGNIFICANT' if j_p < 0.05 else 'NOT statistically significant'} at alpha=0.05."
        )
    else:
        lines.append("- Judge-accuracy delta: could not be computed (coverage gate or overlap failed).")

    out_md = Path(__file__).resolve().parent / "structured_surface_control.md"
    out_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote: {out_json}")
    print(f"wrote: {out_md}")


if __name__ == "__main__":
    main()
