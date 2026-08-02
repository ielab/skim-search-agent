#!/usr/bin/env python
"""Paired significance test for the recall/accuracy inversion (reviewer m2, carried several
rounds): the paper currently states DESCRIPTIVELY, with no test, that the hybrid control
(`agent_research_hybrid_fetch_snip`) "retrieves more of the right documents" than the full
method Sieve (`agent_research_bql_dense_snip`) while scoring LOWER accuracy. This script tests
that recall difference itself -- on the shared instance set, per dataset -- so the sentence can
say "significantly more" where true and stop short where it is not.

THE COMPARISON (identical cells to analysis/structured_surface_control_3way.py):
    FULL    (Sieve)   : agent_research_bql_dense_snip
    CONTROL (hybrid)  : agent_research_hybrid_fetch_snip
    tier    : _headline_validation
    model   : Tongyi-DeepResearch-30B-A3B
    datasets: browsecomp_plus_structured (n=830), hotpotqa_structured (n=7343),
              musique_structured (n=2409)

METHOD PARITY IS MANDATORY. This script does NOT reimplement gold-doc recall, EM, the recovery
overlay, or McNemar's test -- it imports and reuses, unmodified:
  - scripts.compare_cells.{load_qrels, mcnemar_p, pct}   (transitively: evaluation.metrics.
    answer_em, scripts.force_answer_backfill.load_rows_with_recovery via cell_dir/cell_rows,
    and the gold_doc_recall regex via metrics()/_row_intrinsic())
  - analysis.structured_surface_control.{cell_metrics, run_sanity_check, FULL_COND, CONTROL_COND,
    SANITY_COND, DATASET (browsecomp), TIER}
    -- i.e. the EXACT same per-cell helper (`cell_metrics`, itself built on compare_cells'
    `cell_dir`/`cell_rows`/`metrics`) the existing EM/judge comparison scripts use, so the
    per-instance `recall` field consumed here is provably the SAME value already reported (and
    published) as `gold_doc_recall_pct` in structured_surface_control(_3way).py -- not a
    re-typed formula that could silently drift.

WHAT "recall" MEANS HERE (read `scripts.compare_cells.gold_doc_recall` before trusting this):
    `gold_doc_recall(row, gold_ids)` returns a Python bool: whether ANY gold corpus id for that
    instance's query was found (via a regex over the observation text) anywhere in the episode's
    retrieval-context observations. This is NOT a continuous fraction of gold docs recovered (it
    does not count how many of a possibly-multi-doc gold set were found) -- it is a per-instance
    BINARY indicator ("at least one gold doc surfaced"). `scripts.compare_cells.metrics()` /
    `analysis.structured_surface_control.cell_metrics()` expose this unchanged as `m[iid]["recall"]`
    (a bool), and `mean_tok_recall()`'s `recall%` column is just `100 * mean(bool)` over an id set
    -- i.e. exactly the proportion-of-True-values summary a binary indicator gets.
    Because the per-instance indicator is BINARY, not continuous, an exact **McNemar test** on the
    paired discordant counts (b = control-surfaced-but-Sieve-didn't, c = Sieve-surfaced-but-
    control-didn't) is the correct paired test -- the same test already used elsewhere in this
    repo for paired EM/judge comparisons (`scripts.compare_cells.mcnemar_p`). A **Wilcoxon
    signed-rank test is NOT used**: Wilcoxon is for paired continuous (or at least non-binary
    ordinal) differences, and on a binary indicator with any ties at 0 it degenerates and is
    documented as inappropriate practice; McNemar is the standard paired test for a 2x2 binary
    outcome. (This script still checks the binary assumption empirically at runtime --
    `_is_binary_indicator()` below -- and would fall back to Wilcoxon-on-paired-fractions with an
    explicit note if the assumption were ever violated by a future change to `gold_doc_recall`.)

MANDATORY SANITY GATE (run first, before anything is trusted): reproduce the three PUBLISHED
recall pairs (analysis/structured_surface_control(_3way).md /
analysis/structured_surface_control_3way_data.json `gold_doc_recall_pct`) from this script's own
shared-set computation, to within 0.1 point:
    browsecomp_plus_structured : 61.2 (Sieve) vs 71.1 (control)
    hotpotqa_structured        : 94.8 (Sieve) vs 96.4 (control)
    musique_structured         : 95.4 (Sieve) vs 96.4 (control)
If ANY of the three fails to reproduce, the script stops (nonzero exit) and reports the mismatch
instead of printing new significance numbers as if they were trustworthy.

Run: PYTHONPATH=. envs/bin/python analysis/recall_inversion_test.py
Writes (new files -- does not overwrite anything under analysis/, runs/, latex/, latex_acl8/):
  - analysis/recall_inversion_test_data.json
  - analysis/recall_inversion_test.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import load_qrels, mcnemar_p, pct  # noqa: E402
from analysis.structured_surface_control import (  # noqa: E402
    cell_metrics, run_sanity_check, FULL_COND, CONTROL_COND, SANITY_COND,
    DATASET as SANITY_DATASET, TIER,
)

DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]

# Published recall numbers this script must reproduce (sanity gate). Source: the task/reviewer
# statement, cross-checked against analysis/structured_surface_control_3way_data.json
# `gold_doc_recall_pct` (browsecomp there matches structured_surface_control.py's own
# `expected_ad_hoc.recall_full/recall_control` = 61.2 / 71.1 exactly).
PUBLISHED_RECALL = {
    "browsecomp_plus_structured": dict(full=61.2, control=71.1),
    "hotpotqa_structured": dict(full=94.8, control=96.4),
    "musique_structured": dict(full=95.4, control=96.4),
}
RECALL_TOL = 0.1

JSON_PATH = Path(__file__).resolve().parent / "recall_inversion_test_data.json"
MD_PATH = Path(__file__).resolve().parent / "recall_inversion_test.md"


def _is_binary_indicator(m: dict) -> bool:
    """True iff every m[iid]["recall"] is a bool / 0-1 value -- i.e. the per-instance recall
    field really is the binary "any gold doc surfaced" indicator gold_doc_recall() returns, not a
    continuous fraction. Determines McNemar (binary) vs Wilcoxon (continuous) below."""
    vals = {v["recall"] for v in m.values()}
    return vals <= {True, False, 0, 1}


def paired_recall(full_m: dict, ctrl_m: dict):
    """(n_shared, rec_full, rec_ctrl, delta, b, c, p) on the inner-joined instance_id set, where
    b = control surfaced a gold doc and Sieve did not, c = Sieve surfaced a gold doc and control
    did not (same b/c convention as `structured_surface_control.paired_em`: b favors the
    'other'/control cell). p is the exact two-sided McNemar p-value on {b, c}
    (`scripts.compare_cells.mcnemar_p`, unmodified)."""
    mut = sorted(set(full_m) & set(ctrl_m))
    full_vals = [bool(full_m[i]["recall"]) for i in mut]
    ctrl_vals = [bool(ctrl_m[i]["recall"]) for i in mut]
    b = sum(1 for i in mut if ctrl_m[i]["recall"] and not full_m[i]["recall"])
    c = sum(1 for i in mut if full_m[i]["recall"] and not ctrl_m[i]["recall"])
    rec_full = pct(full_vals)
    rec_ctrl = pct(ctrl_vals)
    p = mcnemar_p(b, c)
    return len(mut), rec_full, rec_ctrl, rec_full - rec_ctrl, b, c, p


def wilcoxon_fallback(full_m: dict, ctrl_m: dict, mut):
    """Only invoked if `_is_binary_indicator` ever returns False for a cell (not expected given
    the current `gold_doc_recall` implementation, kept here so the script degrades honestly
    rather than silently mis-applying McNemar to a continuous quantity)."""
    from scipy.stats import wilcoxon
    diffs = [float(ctrl_m[i]["recall"]) - float(full_m[i]["recall"]) for i in mut]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return dict(statistic=0.0, p=1.0, mean_diff=0.0, n_nonzero=0)
    stat, p = wilcoxon(nonzero)
    mean_diff = sum(diffs) / len(diffs)
    return dict(statistic=float(stat), p=float(p), mean_diff=mean_diff, n_nonzero=len(nonzero))


def run_sanity_gate(qrels_by_dataset: dict) -> dict:
    """Two parts:
    (a) reuse `structured_surface_control.run_sanity_check` verbatim -- the SAME published-number
        reproduction gate `structured_surface_control(_3way).py` require before trusting anything
        new from these cells (Sieve vs 'No dense evidence' EM contrast, browsecomp only).
    (b) THIS script's own mandatory gate: reproduce all three published recall pairs
        (PUBLISHED_RECALL) to within RECALL_TOL, computed from this script's own shared-set
        `paired_recall`.
    """
    gate_a = run_sanity_check(qrels_by_dataset[SANITY_DATASET])

    gate_b = {}
    gate_b_passed = True
    for dataset in DATASETS:
        full_m, full_cov, full_n = cell_metrics(dataset, TIER, FULL_COND, qrels_by_dataset[dataset])
        ctrl_m, ctrl_cov, ctrl_n = cell_metrics(dataset, TIER, CONTROL_COND, qrels_by_dataset[dataset])
        if full_m is None or ctrl_m is None:
            gate_b[dataset] = dict(passed=False, reason="FULL or CONTROL cell rows.jsonl missing")
            gate_b_passed = False
            continue
        n_shared, rec_full, rec_ctrl, delta, b, c, p = paired_recall(full_m, ctrl_m)
        expected = PUBLISHED_RECALL[dataset]
        ok = (
            abs(rec_full - expected["full"]) <= RECALL_TOL
            and abs(rec_ctrl - expected["control"]) <= RECALL_TOL
        )
        gate_b[dataset] = dict(
            passed=bool(ok), n_shared=n_shared, recall_full=rec_full, recall_control=rec_ctrl,
            expected=expected,
        )
        gate_b_passed = gate_b_passed and ok

    return dict(gate_a=gate_a, gate_b=gate_b, passed=bool(gate_a.get("passed") and gate_b_passed))


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

    mut = sorted(set(full_m) & set(ctrl_m))
    full_binary = _is_binary_indicator({i: full_m[i] for i in mut})
    ctrl_binary = _is_binary_indicator({i: ctrl_m[i] for i in mut})
    is_binary = full_binary and ctrl_binary

    n_shared, rec_full, rec_ctrl, delta, b, c, p = paired_recall(full_m, ctrl_m)

    entry.update(
        n_full=full_n, n_control=ctrl_n, n_shared=n_shared,
        recall=dict(
            full=rec_full, control=rec_ctrl, delta=delta, b=b, c=c,
        ),
        is_binary_indicator=bool(is_binary),
    )

    if is_binary:
        entry["test"] = "exact McNemar"
        entry["test_reason"] = (
            "scripts.compare_cells.gold_doc_recall() returns a per-instance bool (\"any gold "
            "doc surfaced\"), confirmed empirically over this dataset's shared instance set -- "
            "the paired outcome is binary, so exact McNemar on the discordant pair counts (b, c) "
            "is the correct paired test (same test already used for paired EM/judge comparisons "
            "elsewhere in this repo)."
        )
        entry["recall"]["statistic"] = dict(b=b, c=c)
        entry["recall"]["p"] = p
        entry["recall"]["significant_at_0_05"] = bool(p < 0.05)
    else:
        wr = wilcoxon_fallback(full_m, ctrl_m, mut)
        entry["test"] = "Wilcoxon signed-rank"
        entry["test_reason"] = (
            "gold_doc_recall() did NOT behave as a binary indicator on this dataset's shared set "
            "(non-{0,1} values observed) -- McNemar would be inappropriate for a continuous paired "
            "quantity, so a Wilcoxon signed-rank test on the paired per-instance recall values is "
            "used instead, with the mean paired difference reported alongside."
        )
        entry["recall"]["statistic"] = dict(wilcoxon_W=wr["statistic"], n_nonzero=wr["n_nonzero"])
        entry["recall"]["p"] = wr["p"]
        entry["recall"]["mean_diff"] = wr["mean_diff"]
        entry["recall"]["significant_at_0_05"] = bool(wr["p"] < 0.05)

    return entry


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def render_markdown(gate: dict, results: dict) -> str:
    lines = []
    lines.append("# Recall/accuracy inversion: paired significance test (Sieve vs hybrid sparse-dense RRF)\n")
    lines.append(
        "The paper currently states, DESCRIPTIVELY and with no test, that the hybrid control "
        f"(`{CONTROL_COND}`) \"retrieves more of the right documents\" than Sieve "
        f"(`{FULL_COND}`) while scoring lower accuracy -- the recall/accuracy inversion. "
        "Reviewer m2 (carried several rounds) asked for a paired test of that recall difference. "
        "This script computes it on the shared instance set (inner join on instance_id) for all "
        f"three datasets, tier `{TIER}`, model `Tongyi-DeepResearch-30B-A3B`. Zero new compute -- "
        "all data already exists; this only reuses `scripts.compare_cells`'s recall machinery.\n"
    )

    lines.append("## Mandatory sanity gate (run first)\n")
    ga = gate["gate_a"]
    lines.append(
        f"**(a)** Sieve vs \"No dense evidence\" (`{SANITY_COND}`) EM contrast on "
        f"browsecomp_plus_structured reproduces the published +4.7 EM / p=0.0104: "
        f"n={ga.get('n_shared')}, EM {ga.get('em_full', float('nan')):.1f} vs "
        f"{ga.get('em_other', float('nan')):.1f}, delta={ga.get('delta', float('nan')):+.1f}, "
        f"p={ga.get('p', float('nan')):.3g} -- **{'PASS' if ga.get('passed') else 'FAIL'}**.\n"
    )
    lines.append("**(b)** Published recall pairs reproduced (own shared-set computation, tol=0.1):\n")
    lines.append("| dataset | n_shared | recall Sieve (computed / published) | recall control (computed / published) | gate |")
    lines.append("|---|---:|---:|---:|---|")
    for dataset in DATASETS:
        gb = gate["gate_b"].get(dataset, {})
        if not gb.get("passed", False) and "recall_full" not in gb:
            lines.append(f"| {dataset} | -- | -- | -- | **FAIL ({gb.get('reason', 'unknown')})** |")
            continue
        exp = gb["expected"]
        lines.append(
            f"| {dataset} | {gb['n_shared']} | {gb['recall_full']:.1f} / {exp['full']:.1f} | "
            f"{gb['recall_control']:.1f} / {exp['control']:.1f} | "
            f"{'**PASS**' if gb['passed'] else '**FAIL**'} |"
        )
    lines.append("")
    lines.append(f"**Sanity gate overall: {'PASS' if gate['passed'] else 'FAIL -- STOPPED, results below (if any) are NOT trustworthy'}**\n")

    if not gate["passed"]:
        return "\n".join(lines) + "\n"

    lines.append("## Results: paired recall test, all three datasets\n")
    lines.append("| dataset | n_shared | recall Sieve | recall control | difference | test used | statistic | p | significant @ 0.05 |")
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---|")
    for dataset in DATASETS:
        e = results[dataset]
        if "error" in e:
            lines.append(f"| {dataset} | -- ({e['error']}) | | | | | | | |")
            continue
        r = e["recall"]
        if e["test"] == "exact McNemar":
            stat_s = f"b={r['statistic']['b']}, c={r['statistic']['c']}"
        else:
            stat_s = f"W={r['statistic']['wilcoxon_W']:.1f}, n_nz={r['statistic']['n_nonzero']}"
        lines.append(
            f"| {dataset} | {e['n_shared']} | {r['full']:.1f} | {r['control']:.1f} | "
            f"{r['delta']:+.1f} | {e['test']} | {stat_s} | {r['p']:.3g} | "
            f"{'yes' if r['significant_at_0_05'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Test choice, per dataset\n")
    for dataset in DATASETS:
        e = results[dataset]
        if "error" in e:
            continue
        lines.append(f"- **{dataset}**: {e['test_reason']}")
    lines.append("")

    lines.append("## Significance statements (per dataset, plain language)\n")
    for dataset in DATASETS:
        e = results[dataset]
        if "error" in e:
            lines.append(f"- {dataset}: {e['error']}")
            continue
        r = e["recall"]
        if r["significant_at_0_05"]:
            verdict = (
                f"the control's recall advantage ({r['delta']:+.1f} points, "
                f"{e['test']} p={r['p']:.3g}) IS statistically significant at alpha=0.05 -- the "
                f"sentence may say \"significantly more\" for this dataset."
            )
        else:
            verdict = (
                f"the control's recall advantage ({r['delta']:+.1f} points, "
                f"{e['test']} p={r['p']:.3g}) is NOT statistically significant at alpha=0.05 -- "
                f"the sentence must NOT say \"significantly more\" / \"more\" for this dataset; "
                f"it should say the difference was not statistically significant."
            )
        lines.append(
            f"- **{dataset}** (n_shared={e['n_shared']}): Sieve recall {r['full']:.1f}% vs "
            f"control recall {r['control']:.1f}% -- {verdict}"
        )
    lines.append("")

    lines.append(
        "Convention: `b` = instances the control surfaced a gold doc and Sieve did not; `c` = "
        "instances Sieve surfaced a gold doc and the control did not (matches the b/c convention "
        "used throughout this repo's paired EM/judge comparisons). No multiple-comparison "
        "correction is applied -- raw per-dataset exact McNemar p-values only.\n"
    )
    return "\n".join(lines) + "\n"


def main():
    qrels_by_dataset = {d: load_qrels(d) for d in DATASETS}

    print("=== Mandatory sanity gate ===")
    gate = run_sanity_gate(qrels_by_dataset)
    ga = gate["gate_a"]
    if ga.get("passed"):
        print(f"(a) PASS: Sieve vs no-dense-evidence EM contrast -- n={ga['n_shared']}, "
              f"EM {ga['em_full']:.1f} vs {ga['em_other']:.1f}, delta={ga['delta']:+.1f}, "
              f"p={ga['p']:.3g} (published: +4.7, p=0.0104)")
    else:
        print(f"(a) FAIL: {ga}")

    for dataset in DATASETS:
        gb = gate["gate_b"].get(dataset, {})
        if "recall_full" in gb:
            print(f"(b) {dataset}: recall Sieve {gb['recall_full']:.1f} (published "
                  f"{gb['expected']['full']:.1f}), control {gb['recall_control']:.1f} "
                  f"(published {gb['expected']['control']:.1f}) -- "
                  f"{'PASS' if gb['passed'] else 'FAIL'}")
        else:
            print(f"(b) {dataset}: FAIL -- {gb.get('reason')}")

    if not gate["passed"]:
        print("\nSTOPPING: sanity gate failed -- refusing to report the recall significance "
              "test until the pipeline reproduces the published recall numbers.")
        JSON_PATH.write_text(json.dumps(dict(gate=gate, results=None, stopped=True),
                                         indent=2, default=_json_default))
        MD_PATH.write_text(render_markdown(gate, {}))
        print(f"wrote: {JSON_PATH}")
        print(f"wrote: {MD_PATH}")
        sys.exit(1)

    print("\nSanity gate PASSED (both parts). Proceeding to the paired recall significance test.\n")

    results = {d: compute_dataset(d, qrels_by_dataset[d]) for d in DATASETS}

    out = dict(gate=gate, results=results, stopped=False)
    JSON_PATH.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"wrote: {JSON_PATH}")

    md = render_markdown(gate, results)
    MD_PATH.write_text(md)
    print(f"wrote: {MD_PATH}")
    print()
    print(md)


if __name__ == "__main__":
    main()
