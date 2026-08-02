#!/usr/bin/env python
"""Structured query surface control (Sieve vs hybrid sparse-dense RRF, same read interface),
extended to all THREE datasets now that hotpotqa_structured (n=7343) and musique_structured
(n=2409) cells for `agent_research_hybrid_fetch_snip` have been merged -- previously only
browsecomp_plus_structured (n=830) had this control, so the paper reported it as a null
(non-significant) on a single dataset. This script recomputes the paired comparison uniformly
across all three datasets.

THE COMPARISON: the full method **Sieve** (`agent_research_bql_dense_snip`) vs the strongest
non-structured control at the SAME read interface -- plain BM25+dense RRF with snippets and
section-level fetch (`agent_research_hybrid_fetch_snip`). This isolates the structured query
surface: it swaps the structured surface for plain BM25+dense RRF while holding the read
interface (snippet preview, section fetch) fixed identically across both cells. All cells:
tier `_headline_validation`, model dir `Tongyi-DeepResearch-30B-A3B`.

METHOD PARITY IS MANDATORY. This script does NOT reimplement EM, the recovery overlay, or
McNemar's test -- it imports and reuses, unmodified:
  - scripts.compare_cells.{load_qrels, mcnemar_p, pct}   (transitively: evaluation.metrics.
    answer_em, scripts.force_answer_backfill.load_rows_with_recovery via cell_rows/metrics)
  - analysis.structured_surface_control.{cell_metrics, paired_em, paired_judge, mean_tok_recall,
    run_sanity_check, FULL_COND, CONTROL_COND, SANITY_COND, DATASET, TIER,
    JUDGE_COVERAGE_MIN, SANITY_EXPECTED}
    -- i.e. the EXACT same per-cell/per-pair helpers the single-dataset (browsecomp-only)
    version of this comparison already uses, so the browsecomp row computed here is provably
    the same computation, not a re-typed formula that could silently drift.

MANDATORY SANITY GATE (run first, before anything new is trusted):
  (a) reproduce the published Sieve vs "No dense evidence" (agent_research_snip) contrast on
      browsecomp_plus_structured: +4.7 EM, p=0.0104 (analysis/ablation_deltas.md). Reuses
      structured_surface_control.run_sanity_check() verbatim -- this IS the same check that
      script runs before trusting its own (single-dataset) hybrid-control number.
  (b) reproduce the EXISTING published browsecomp hybrid-control result itself (analysis/
      structured_surface_control.md / structured_surface_control_data.json): EM delta +3.3,
      p=0.102; judge delta +2.4, p=0.248.
If EITHER gate fails, the script stops (nonzero exit) and reports the mismatch instead of
printing the new hotpotqa/musique numbers as if they were trustworthy.

JUDGE COVERAGE GATE: BrowseComp-Plus is the LLM-judge benchmark and both its cells are fully
judged (>=90% coverage gate, JUDGE_COVERAGE_MIN). HotpotQA and MuSiQue are EM/F1 benchmarks and
are DELIBERATELY NOT judged in this repo -- for those two datasets this script reports
"not judged by design" rather than computing a judge delta or treating missing judge_cache.jsonl
as a coverage-gate failure.

Run: PYTHONPATH=. envs/bin/python analysis/structured_surface_control_3way.py
Writes (new files -- does NOT overwrite structured_surface_control.*):
  - analysis/structured_surface_control_3way_data.json
  - analysis/structured_surface_control_3way.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import load_qrels  # noqa: E402
from analysis.structured_surface_control import (  # noqa: E402
    cell_metrics, paired_em, paired_judge, mean_tok_recall, run_sanity_check,
    DATASET as SANITY_DATASET, TIER, FULL_COND, CONTROL_COND, SANITY_COND,
    JUDGE_COVERAGE_MIN,
)

DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]

# Only browsecomp_plus_structured is an LLM-judge benchmark with a populated judge_cache.jsonl.
# hotpotqa_structured / musique_structured are EM/F1 benchmarks -- NOT judged by design.
JUDGED_DATASETS = {"browsecomp_plus_structured"}

# Gate (b): the EXISTING published browsecomp hybrid-control result (analysis/
# structured_surface_control_data.json, computed by analysis/structured_surface_control.py) --
# must reproduce before the new (hotpotqa/musique) numbers are trusted.
GATE_B_EXPECTED = dict(em_delta=3.3, em_p=0.102, judge_delta=2.4, judge_p=0.248)
GATE_B_TOL = dict(delta=0.15, p=0.002)

JSON_PATH = Path(__file__).resolve().parent / "structured_surface_control_3way_data.json"
MD_PATH = Path(__file__).resolve().parent / "structured_surface_control_3way.md"


def run_gate_b(qrels: dict) -> dict:
    """Reproduce the EXISTING browsecomp hybrid-control result (same cells, same helpers as
    structured_surface_control.py's main()). Returns a dict with a `passed` bool."""
    full_m, full_cov, full_n = cell_metrics(SANITY_DATASET, TIER, FULL_COND, qrels)
    ctrl_m, ctrl_cov, ctrl_n = cell_metrics(SANITY_DATASET, TIER, CONTROL_COND, qrels)
    if full_m is None or ctrl_m is None:
        return dict(passed=False, reason="FULL or CONTROL cell rows.jsonl missing on browsecomp")

    n_shared, em_full, em_ctrl, em_delta, em_b, em_c, em_p = paired_em(full_m, ctrl_m)
    jres = paired_judge(full_m, ctrl_m, full_cov, ctrl_cov)
    if jres is None:
        return dict(passed=False, reason="judge comparison unavailable on browsecomp (coverage "
                                          "gate or no judged overlap) -- cannot check judge gate")
    j_n, j_full, j_ctrl, j_delta, j_b, j_c, j_p = jres

    ok = (
        abs(em_delta - GATE_B_EXPECTED["em_delta"]) <= GATE_B_TOL["delta"]
        and abs(em_p - GATE_B_EXPECTED["em_p"]) <= GATE_B_TOL["p"]
        and abs(j_delta - GATE_B_EXPECTED["judge_delta"]) <= GATE_B_TOL["delta"]
        and abs(j_p - GATE_B_EXPECTED["judge_p"]) <= GATE_B_TOL["p"]
    )
    return dict(
        passed=bool(ok), n_shared=n_shared, em_delta=em_delta, em_p=em_p,
        judge_delta=j_delta, judge_p=j_p, expected=GATE_B_EXPECTED,
    )


def compute_dataset(dataset: str, qrels: dict) -> dict:
    entry = {"dataset": dataset, "tier": TIER, "full_cond": FULL_COND, "control_cond": CONTROL_COND}
    full_m, full_cov, full_n = cell_metrics(dataset, TIER, FULL_COND, qrels)
    ctrl_m, ctrl_cov, ctrl_n = cell_metrics(dataset, TIER, CONTROL_COND, qrels)
    if full_m is None:
        entry["error"] = "FULL (Sieve) cell missing (rows.jsonl absent)"
        return entry
    if ctrl_m is None:
        entry["error"] = "CONTROL (hybrid) cell missing (rows.jsonl absent)"
        return entry

    n_shared, em_full, em_ctrl, em_delta, em_b, em_c, em_p = paired_em(full_m, ctrl_m)
    mut = sorted(set(full_m) & set(ctrl_m))
    tok_full, rec_full = mean_tok_recall(full_m, mut)
    tok_ctrl, rec_ctrl = mean_tok_recall(ctrl_m, mut)

    entry.update(
        n_full=full_n, n_control=ctrl_n, n_shared=n_shared,
        judge_coverage_full=full_cov, judge_coverage_control=ctrl_cov,
        judge_coverage_gate=JUDGE_COVERAGE_MIN,
        em=dict(full=em_full, control=em_ctrl, delta=em_delta, b=em_b, c=em_c, p=em_p,
                 significant_at_0_05=bool(em_p < 0.05)),
        tokens_once_mean=dict(full=tok_full, control=tok_ctrl),
        gold_doc_recall_pct=dict(full=rec_full, control=rec_ctrl),
    )

    if dataset in JUDGED_DATASETS:
        jres = paired_judge(full_m, ctrl_m, full_cov, ctrl_cov)
        if jres is not None:
            j_n, j_full, j_ctrl, j_delta, j_b, j_c, j_p = jres
            entry["judge"] = dict(
                n_shared=j_n, full=j_full, control=j_ctrl, delta=j_delta, b=j_b, c=j_c, p=j_p,
                significant_at_0_05=bool(j_p < 0.05),
            )
            entry["judge_status"] = "computed"
        else:
            entry["judge"] = None
            entry["judge_status"] = (
                f"judge coverage gate (>={JUDGE_COVERAGE_MIN*100:.0f}%) not met or no judged "
                f"overlap: full_cov={full_cov}, control_cov={ctrl_cov}"
            )
    else:
        entry["judge"] = None
        entry["judge_status"] = "not judged by design (EM/F1 benchmark, not the LLM-judge benchmark)"

    return entry


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def render_markdown(gate_a: dict, gate_b: dict, results: dict) -> str:
    lines = []
    lines.append("# Structured query surface control, three datasets: Sieve vs hybrid sparse-dense RRF (same read interface)\n")
    lines.append(
        f"Isolates the structured query surface: **Sieve** (`{FULL_COND}`) vs the strongest "
        f"non-structured control at the *same* read interface -- plain BM25+dense RRF with "
        f"snippets and section fetch (`{CONTROL_COND}`). Tier `{TIER}`, model dir "
        f"`Tongyi-DeepResearch-30B-A3B`, on all three datasets. Previously this contrast existed "
        f"only on browsecomp_plus_structured (n=830), reported as a null; hotpotqa_structured "
        f"(n=7343) and musique_structured (n=2409) cells have now been merged, completing it on "
        f"all three.\n"
    )

    lines.append("## Mandatory sanity gate (run before trusting anything new)\n")
    lines.append(
        f"**(a)** Sieve vs \"No dense evidence\" (`{SANITY_COND}`) on browsecomp_plus_structured "
        f"reproduces the published +4.7 EM / p=0.0104: "
        f"n={gate_a.get('n_shared')}, EM {gate_a.get('em_full', float('nan')):.1f} vs "
        f"{gate_a.get('em_other', float('nan')):.1f}, delta={gate_a.get('delta', float('nan')):+.1f}, "
        f"p={gate_a.get('p', float('nan')):.3g} -- **{'PASS' if gate_a.get('passed') else 'FAIL'}**.\n"
    )
    lines.append(
        f"**(b)** The existing published browsecomp hybrid-control result reproduces at +3.3 EM "
        f"(p=0.102) / +2.4 judge (p=0.248): "
        f"EM delta={gate_b.get('em_delta', float('nan')):+.1f} (p={gate_b.get('em_p', float('nan')):.3g}), "
        f"judge delta={gate_b.get('judge_delta', float('nan')):+.1f} "
        f"(p={gate_b.get('judge_p', float('nan')):.3g}) -- **{'PASS' if gate_b.get('passed') else 'FAIL'}**.\n"
    )
    both_passed = bool(gate_a.get("passed") and gate_b.get("passed"))
    lines.append(f"**Sanity gate overall: {'PASS' if both_passed else 'FAIL -- STOPPED, new numbers below are NOT trustworthy'}**\n")

    lines.append("## Results (all three datasets)\n")
    lines.append(
        "| dataset | n_shared | EM Sieve | EM control | EM delta | b | c | McNemar p | sig@.05 | "
        "judge Sieve | judge control | judge delta | judge p | judge sig@.05 | judge status | "
        "tok Sieve | tok control | recall Sieve % | recall control % |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for dataset in DATASETS:
        e = results[dataset]
        if "error" in e:
            lines.append(f"| {dataset} | -- ({e['error']}) | | | | | | | | | | | | | | | | | |")
            continue
        em = e["em"]
        tok = e["tokens_once_mean"]
        rec = e["gold_doc_recall_pct"]
        if e["judge"] is not None:
            j = e["judge"]
            j_full_s, j_ctrl_s, j_delta_s, j_p_s, j_sig_s = (
                f"{j['full']:.1f}", f"{j['control']:.1f}", f"{j['delta']:+.1f}", f"{j['p']:.3g}",
                "yes" if j["significant_at_0_05"] else "no",
            )
        else:
            j_full_s = j_ctrl_s = j_delta_s = j_p_s = "--"
            j_sig_s = "--"
        lines.append(
            f"| {dataset} | {e['n_shared']} | {em['full']:.1f} | {em['control']:.1f} | "
            f"{em['delta']:+.1f} | {em['b']} | {em['c']} | {em['p']:.3g} | "
            f"{'yes' if em['significant_at_0_05'] else 'no'} | "
            f"{j_full_s} | {j_ctrl_s} | {j_delta_s} | {j_p_s} | {j_sig_s} | {e['judge_status']} | "
            f"{tok['full']:,.0f} | {tok['control']:,.0f} | {rec['full']:.1f} | {rec['control']:.1f} |"
        )
    lines.append("")

    lines.append("## Significance statements (per dataset)\n")
    for dataset in DATASETS:
        e = results[dataset]
        if "error" in e:
            lines.append(f"- {dataset}: {e['error']}")
            continue
        em = e["em"]
        sig = "STATISTICALLY SIGNIFICANT" if em["significant_at_0_05"] else "NOT statistically significant"
        lines.append(
            f"- **{dataset}**: EM delta {em['delta']:+.1f} points (Sieve {em['full']:.1f}% vs "
            f"control {em['control']:.1f}%, McNemar p={em['p']:.3g}, n={e['n_shared']}) is "
            f"{sig} at alpha=0.05. Judge: {e['judge_status']}."
        )
    lines.append("")

    lines.append(
        "Convention (matches `structured_surface_control.py` / `ablation_deltas.py`): `b` = "
        "instances the control cell got right and Sieve got wrong; `c` = instances Sieve got "
        "right and the control got wrong. No multiple-comparison correction is applied here -- "
        "raw per-comparison exact McNemar p-values only.\n"
    )
    return "\n".join(lines) + "\n"


def main():
    qrels_by_dataset = {d: load_qrels(d) for d in DATASETS}

    print("=== Mandatory sanity gate ===")
    gate_a = run_sanity_check(qrels_by_dataset[SANITY_DATASET])
    if gate_a.get("passed"):
        print(f"(a) PASS: Sieve vs no-dense-evidence -- n={gate_a['n_shared']}, "
              f"EM {gate_a['em_full']:.1f} vs {gate_a['em_other']:.1f}, "
              f"delta={gate_a['delta']:+.1f}, p={gate_a['p']:.3g} (published: +4.7, p=0.0104)")
    else:
        print(f"(a) FAIL: {gate_a}")

    gate_b = run_gate_b(qrels_by_dataset[SANITY_DATASET])
    if gate_b.get("passed"):
        print(f"(b) PASS: existing hybrid-control result -- EM delta={gate_b['em_delta']:+.1f} "
              f"(p={gate_b['em_p']:.3g}), judge delta={gate_b['judge_delta']:+.1f} "
              f"(p={gate_b['judge_p']:.3g}) (published: +3.3/p=0.102, +2.4/p=0.248)")
    else:
        print(f"(b) FAIL: {gate_b}")

    if not (gate_a.get("passed") and gate_b.get("passed")):
        print("\nSTOPPING: sanity gate failed -- refusing to report the new (hotpotqa/musique) "
              "numbers until the pipeline reproduces both known-published results.")
        JSON_PATH.write_text(json.dumps(
            dict(gate_a=gate_a, gate_b=gate_b, results=None, stopped=True),
            indent=2, default=_json_default))
        sys.exit(1)

    print("\nBoth sanity checks PASSED. Proceeding to the three-dataset comparison.\n")

    results = {d: compute_dataset(d, qrels_by_dataset[d]) for d in DATASETS}

    out = dict(gate_a=gate_a, gate_b=gate_b, results=results, stopped=False)
    JSON_PATH.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"wrote: {JSON_PATH}")

    md = render_markdown(gate_a, gate_b, results)
    MD_PATH.write_text(md)
    print(f"wrote: {MD_PATH}")
    print()
    print(md)


if __name__ == "__main__":
    main()
