#!/usr/bin/env python
"""Paired bootstrap 95% CIs for the Sieve vs hybrid-sparse-dense-RRF-fetch-snip control, on all
three datasets, requested by an ACL reviewer so the paper can distinguish "we lost on
BrowseComp-Plus" from "we are underpowered on BrowseComp-Plus".

THE CONTRAST (unchanged from analysis/structured_surface_control.py /
structured_surface_control_3way.py): Sieve (`agent_research_bql_dense_snip`) vs hybrid
sparse-dense RRF + snippets + section fetch (`agent_research_hybrid_fetch_snip`), tier
`_headline_validation`, model dir `Tongyi-DeepResearch-30B-A3B`. The paper currently reports this
as a null on browsecomp_plus_structured (EM delta +3.3, McNemar p=0.102) but a real, significant
effect on hotpotqa_structured (+5.7, p=1.62e-27) and musique_structured (+5.5, p=1.58e-11) -- same
direction, same sign, on 7343- and 2409-instance datasets. A paired bootstrap CI on the
browsecomp delta tells us whether that null is "no effect" or "not enough n to see the same-size
effect the other two datasets show".

METHOD PARITY IS MANDATORY -- this script does NOT reimplement EM, the recovery overlay, McNemar,
or a bootstrap procedure. It reuses, unmodified:
  - analysis.structured_surface_control.{cell_metrics, paired_em, paired_judge, DATASET, TIER,
    FULL_COND, CONTROL_COND, JUDGE_COVERAGE_MIN} -- the exact per-cell/per-pair helpers
    structured_surface_control.py and structured_surface_control_3way.py already use (themselves
    built on scripts.compare_cells.{cell_dir, cell_rows, load_judge_cache, load_qrels, metrics,
    mcnemar_p, pct}, which is transitively evaluation.metrics.answer_em and
    scripts.force_answer_backfill.load_rows_with_recovery via cell_rows()).
  - analysis.equivalence_and_latency.{bootstrap_diff_pp, bootstrap_ci, BOOT_SEED} -- the repo's
    OWN existing paired-bootstrap-over-instances helper, built for the M1 equivalence/CI work on
    this exact same method vs. the BM25 baseline. Reused verbatim here with n_boot=10000 (this
    task's requested resample count) instead of that script's N_BOOT=20000, and its own fixed seed
    (BOOT_SEED=20260723) is reused as-is and reported below -- not re-derived or hand-tuned.

MANDATORY SANITY GATE (run first, before any CI is trusted): reproduce the three published EM
point estimates (+3.3 / +5.7 / +5.5) from a fresh shared-set computation, via the same
cell_metrics()+paired_em() call every other script in this repo uses for this contrast. If any of
the three fails to reproduce within tolerance, this script stops (nonzero exit) and reports the
mismatch instead of printing untrustworthy CIs.

JUDGE: browsecomp_plus_structured is the only fully-judged cell pair (>=90% coverage gate on both
sides); a judge-delta paired bootstrap CI is computed for it only. hotpotqa_structured and
musique_structured are EM/F1 benchmarks, not judged by design -- no judge CI is computed for them.

Run: PYTHONPATH=. envs/bin/python analysis/hybrid_control_ci.py
Writes (new files -- does not overwrite any existing analysis output):
  - analysis/hybrid_control_ci_data.json
  - analysis/hybrid_control_ci.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import load_qrels  # noqa: E402
from analysis.structured_surface_control import (  # noqa: E402
    cell_metrics, paired_em, paired_judge,
    DATASET as BCP_DATASET, TIER, FULL_COND, CONTROL_COND, JUDGE_COVERAGE_MIN,
)
from analysis.equivalence_and_latency import (  # noqa: E402
    bootstrap_diff_pp, bootstrap_ci, BOOT_SEED,
)

DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]
JUDGED_DATASETS = {"browsecomp_plus_structured"}  # only fully-judged pair; see module docstring

N_BOOT = 10000  # this task's requested resample count (equivalence_and_latency.py used 20000)
CI_SEED = BOOT_SEED  # reused as-is from analysis/equivalence_and_latency.py, not re-derived

# Published point estimates this script must reproduce before any CI is trusted (analysis/
# structured_surface_control.md for browsecomp; analysis/structured_surface_control_3way.md for
# hotpotqa/musique).
PUBLISHED_EM_DELTA = {
    "browsecomp_plus_structured": 3.3,
    "hotpotqa_structured": 5.7,
    "musique_structured": 5.5,
}
PUBLISHED_EM_P = {
    "browsecomp_plus_structured": 0.102,
    "hotpotqa_structured": 1.62e-27,
    "musique_structured": 1.58e-11,
}
SANITY_TOL_DELTA = 0.15  # matches structured_surface_control.py's own SANITY_TOL['delta']

JSON_PATH = Path(__file__).resolve().parent / "hybrid_control_ci_data.json"
MD_PATH = Path(__file__).resolve().parent / "hybrid_control_ci.md"


def sanity_gate(qrels_by_dataset: dict) -> dict:
    """Reproduce the three published EM point estimates from a fresh shared-set computation,
    using the exact same cell_metrics()+paired_em() calls every other script in this repo uses
    for this contrast. Returns per-dataset detail plus an overall `passed` bool."""
    per_dataset = {}
    all_passed = True
    for ds in DATASETS:
        qrels = qrels_by_dataset[ds]
        full_m, full_cov, full_n = cell_metrics(ds, TIER, FULL_COND, qrels)
        ctrl_m, ctrl_cov, ctrl_n = cell_metrics(ds, TIER, CONTROL_COND, qrels)
        if full_m is None or ctrl_m is None:
            per_dataset[ds] = dict(passed=False, reason="FULL or CONTROL cell rows.jsonl missing")
            all_passed = False
            continue
        n_shared, em_full, em_ctrl, em_delta, b, c, p = paired_em(full_m, ctrl_m)
        expected = PUBLISHED_EM_DELTA[ds]
        ok = abs(em_delta - expected) <= SANITY_TOL_DELTA
        per_dataset[ds] = dict(
            passed=bool(ok), n_shared=n_shared, em_full=em_full, em_control=em_ctrl,
            em_delta=em_delta, mcnemar_p=p, published_delta=expected,
            published_p=PUBLISHED_EM_P[ds],
        )
        all_passed = all_passed and ok
    return dict(passed=bool(all_passed), per_dataset=per_dataset)


def compute_em_ci(dataset: str, qrels: dict) -> dict:
    full_m, full_cov, full_n = cell_metrics(dataset, TIER, FULL_COND, qrels)
    ctrl_m, ctrl_cov, ctrl_n = cell_metrics(dataset, TIER, CONTROL_COND, qrels)
    n_shared, em_full, em_ctrl, em_delta, b, c, p = paired_em(full_m, ctrl_m)

    mut = sorted(set(full_m) & set(ctrl_m))
    a_vals = [bool(full_m[i]["em"]) for i in mut]
    b_vals = [bool(ctrl_m[i]["em"]) for i in mut]
    assert len(a_vals) == n_shared == len(mut)

    boot = bootstrap_diff_pp(a_vals, b_vals, n_boot=N_BOOT)
    ci_lo, ci_hi = bootstrap_ci(boot, 0.95)

    return dict(
        dataset=dataset, metric="em", n_shared=n_shared,
        point_estimate_pp=em_delta, em_full=em_full, em_control=em_ctrl,
        mcnemar_p=p, b_disc=b, c_disc=c,
        n_boot=N_BOOT, seed=CI_SEED,
        boot_mean_pp=mean(boot),
        ci95_lo_pp=ci_lo, ci95_hi_pp=ci_hi,
    )


def compute_judge_ci(dataset: str, qrels: dict) -> dict | None:
    full_m, full_cov, full_n = cell_metrics(dataset, TIER, FULL_COND, qrels)
    ctrl_m, ctrl_cov, ctrl_n = cell_metrics(dataset, TIER, CONTROL_COND, qrels)
    jres = paired_judge(full_m, ctrl_m, full_cov, ctrl_cov)
    if jres is None:
        return None
    j_n, j_full, j_ctrl, j_delta, j_b, j_c, j_p = jres

    mut = sorted(set(full_m) & set(ctrl_m))
    jmut = [i for i in mut if full_m[i]["judge"] is not None and ctrl_m[i]["judge"] is not None]
    assert len(jmut) == j_n
    a_vals = [bool(full_m[i]["judge"]) for i in jmut]
    b_vals = [bool(ctrl_m[i]["judge"]) for i in jmut]

    boot = bootstrap_diff_pp(a_vals, b_vals, n_boot=N_BOOT)
    ci_lo, ci_hi = bootstrap_ci(boot, 0.95)

    return dict(
        dataset=dataset, metric="judge", n_shared=j_n,
        point_estimate_pp=j_delta, judge_full=j_full, judge_control=j_ctrl,
        mcnemar_p=j_p, b_disc=j_b, c_disc=j_c,
        judge_coverage_full=full_cov, judge_coverage_control=ctrl_cov,
        n_boot=N_BOOT, seed=CI_SEED,
        boot_mean_pp=mean(boot),
        ci95_lo_pp=ci_lo, ci95_hi_pp=ci_hi,
    )


def overlap_statement(bcp_em: dict, hp_em: dict, mu_em: dict) -> dict:
    lo, hi = bcp_em["ci95_lo_pp"], bcp_em["ci95_hi_pp"]
    hp_in = lo <= hp_em["point_estimate_pp"] <= hi
    mu_in = lo <= mu_em["point_estimate_pp"] <= hi
    return dict(
        bcp_ci95_pp=(lo, hi),
        hotpotqa_point_pp=hp_em["point_estimate_pp"], hotpotqa_in_bcp_ci=bool(hp_in),
        musique_point_pp=mu_em["point_estimate_pp"], musique_in_bcp_ci=bool(mu_in),
        both_in_bcp_ci=bool(hp_in and mu_in),
    )


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def render_markdown(gate: dict, em_results: dict, judge_result: dict, overlap: dict) -> str:
    lines = []
    lines.append("# Paired bootstrap 95% CIs: Sieve vs hybrid sparse-dense RRF + snip + fetch, all three datasets\n")
    lines.append(
        f"Reviewer-requested power check on the browsecomp_plus_structured null (EM delta +3.3, "
        f"McNemar p=0.102). Contrast: **Sieve** (`{FULL_COND}`) vs **hybrid sparse-dense RRF + "
        f"snip + fetch** (`{CONTROL_COND}`), tier `{TIER}`, model dir `Tongyi-DeepResearch-30B-A3B`. "
        f"Paired bootstrap over instances (resample the shared instance set with replacement, "
        f"recompute the paired EM/judge delta each draw), {N_BOOT} resamples, seed={CI_SEED} "
        f"(reused as-is from `analysis/equivalence_and_latency.py`'s `BOOT_SEED`), percentile "
        f"2.5/97.5 CI. Bootstrap machinery reused verbatim from `analysis.equivalence_and_latency."
        f"{{bootstrap_diff_pp, bootstrap_ci}}`; EM/judge/recovery machinery reused verbatim from "
        f"`analysis.structured_surface_control.{{cell_metrics, paired_em, paired_judge}}` "
        f"(transitively `scripts.compare_cells` and `evaluation.metrics.answer_em`).\n"
    )

    lines.append("## Mandatory sanity gate (run before trusting any CI)\n")
    lines.append(
        "Reproduce the three published EM point estimates from a fresh shared-set computation "
        f"(tolerance ±{SANITY_TOL_DELTA} pp):\n"
    )
    lines.append("| dataset | n_shared | reproduced delta | published delta | McNemar p (reproduced) | published p | gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for ds in DATASETS:
        g = gate["per_dataset"][ds]
        lines.append(
            f"| {ds} | {g['n_shared']} | {g['em_delta']:+.2f} | {g['published_delta']:+.1f} | "
            f"{g['mcnemar_p']:.3g} | {g['published_p']:.3g} | "
            f"{'PASS' if g['passed'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"**Sanity gate overall: {'PASS' if gate['passed'] else 'FAIL -- STOPPED, no CIs below are trustworthy'}**\n")

    lines.append("## Paired bootstrap 95% CIs on the EM delta (Sieve − control), all three datasets\n")
    lines.append("| dataset | n_shared | point estimate (pp) | bootstrap mean (pp) | 95% CI (pp) | McNemar p |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for ds in DATASETS:
        e = em_results[ds]
        lines.append(
            f"| {ds} | {e['n_shared']} | {e['point_estimate_pp']:+.2f} | {e['boot_mean_pp']:+.2f} | "
            f"[{e['ci95_lo_pp']:+.2f}, {e['ci95_hi_pp']:+.2f}] | {e['mcnemar_p']:.3g} |"
        )
    lines.append("")

    lines.append("## BrowseComp-Plus judge-delta paired bootstrap 95% CI (both cells fully judged)\n")
    if judge_result is not None:
        j = judge_result
        lines.append(
            f"n_shared={j['n_shared']} (judge coverage full={j['judge_coverage_full']:.1%}, "
            f"control={j['judge_coverage_control']:.1%}), point estimate={j['point_estimate_pp']:+.2f} pp, "
            f"bootstrap mean={j['boot_mean_pp']:+.2f} pp, **95% CI [{j['ci95_lo_pp']:+.2f}, "
            f"{j['ci95_hi_pp']:+.2f}] pp**, McNemar p={j['mcnemar_p']:.3g} (published: +2.4, p=0.248).\n"
        )
    else:
        lines.append("Judge CI unavailable (coverage gate or judged-overlap failed).\n")

    lines.append("## Does the BrowseComp-Plus CI overlap the HotpotQA / MuSiQue point estimates?\n")
    lo, hi = overlap["bcp_ci95_pp"]
    lines.append(
        f"BrowseComp-Plus EM-delta 95% CI: **[{lo:+.2f}, {hi:+.2f}] pp**. "
        f"HotpotQA point estimate: {overlap['hotpotqa_point_pp']:+.2f} pp -- "
        f"{'INSIDE' if overlap['hotpotqa_in_bcp_ci'] else 'outside'} the BrowseComp-Plus CI. "
        f"MuSiQue point estimate: {overlap['musique_point_pp']:+.2f} pp -- "
        f"{'INSIDE' if overlap['musique_in_bcp_ci'] else 'outside'} the BrowseComp-Plus CI.\n"
    )
    if overlap["both_in_bcp_ci"]:
        lines.append(
            "**Both the HotpotQA and MuSiQue point estimates fall inside the BrowseComp-Plus 95% "
            "CI.** This is consistent with BrowseComp-Plus being underpowered to detect an effect "
            "of the same size the method shows on the other two datasets (n=830 vs n=7343/2409), "
            "rather than evidence that the true effect on BrowseComp-Plus is absent or smaller. "
            "This is NOT proof the underlying effect is equal on BrowseComp-Plus -- overlap of a "
            "wide CI with another dataset's point estimate is consistent with, not evidence for, "
            "an equal true effect; the interval is also consistent with a genuinely smaller (or "
            "zero) BrowseComp-Plus effect. The honest statement is: the BrowseComp-Plus data "
            "cannot statistically distinguish 'the effect is the same size here but n is too small "
            "to detect it' from 'the effect really is smaller/absent on this harder benchmark'."
        )
    else:
        lines.append(
            "At least one of the HotpotQA/MuSiQue point estimates falls OUTSIDE the "
            "BrowseComp-Plus 95% CI, which would argue against the pure underpowering "
            "explanation for that comparison."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    qrels_by_dataset = {d: load_qrels(d) for d in DATASETS}

    print("=== Mandatory sanity gate: reproduce published EM deltas +3.3 / +5.7 / +5.5 ===")
    gate = sanity_gate(qrels_by_dataset)
    for ds in DATASETS:
        g = gate["per_dataset"][ds]
        status = "PASS" if g.get("passed") else "FAIL"
        print(f"  {ds}: n={g.get('n_shared')}, delta={g.get('em_delta', float('nan')):+.2f} "
              f"(published {g.get('published_delta'):+.1f}), p={g.get('mcnemar_p', float('nan')):.3g} "
              f"(published {g.get('published_p'):.3g}) -- {status}")

    if not gate["passed"]:
        print("\nSTOPPING: sanity gate failed -- refusing to compute or report bootstrap CIs "
              "until the published point estimates reproduce.")
        JSON_PATH.write_text(json.dumps(dict(gate=gate, stopped=True), indent=2, default=_json_default))
        sys.exit(1)

    print("\nSanity gate PASSED on all three datasets. Proceeding to paired bootstrap CIs "
          f"({N_BOOT} resamples, seed={CI_SEED}).\n")

    em_results = {ds: compute_em_ci(ds, qrels_by_dataset[ds]) for ds in DATASETS}
    for ds in DATASETS:
        e = em_results[ds]
        print(f"[EM] {ds}: n={e['n_shared']}, point={e['point_estimate_pp']:+.2f} pp, "
              f"95% CI=[{e['ci95_lo_pp']:+.2f}, {e['ci95_hi_pp']:+.2f}] pp")

    judge_result = compute_judge_ci(BCP_DATASET, qrels_by_dataset[BCP_DATASET])
    if judge_result is not None:
        j = judge_result
        print(f"[judge] {BCP_DATASET}: n={j['n_shared']}, point={j['point_estimate_pp']:+.2f} pp, "
              f"95% CI=[{j['ci95_lo_pp']:+.2f}, {j['ci95_hi_pp']:+.2f}] pp")
    else:
        print(f"[judge] {BCP_DATASET}: unavailable")

    overlap = overlap_statement(
        em_results["browsecomp_plus_structured"],
        em_results["hotpotqa_structured"],
        em_results["musique_structured"],
    )
    print(f"\nOverlap check: BCP CI=[{overlap['bcp_ci95_pp'][0]:+.2f}, {overlap['bcp_ci95_pp'][1]:+.2f}] pp, "
          f"hotpotqa point {overlap['hotpotqa_point_pp']:+.2f} pp "
          f"({'inside' if overlap['hotpotqa_in_bcp_ci'] else 'outside'}), "
          f"musique point {overlap['musique_point_pp']:+.2f} pp "
          f"({'inside' if overlap['musique_in_bcp_ci'] else 'outside'})")

    out = dict(
        contrast=dict(full_cond=FULL_COND, control_cond=CONTROL_COND, tier=TIER,
                      model_dir="Tongyi-DeepResearch-30B-A3B"),
        n_boot=N_BOOT, seed=CI_SEED,
        gate=gate, em=em_results, judge={BCP_DATASET: judge_result}, overlap=overlap,
        stopped=False,
    )
    JSON_PATH.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"\nwrote: {JSON_PATH}")

    md = render_markdown(gate, em_results, judge_result, overlap)
    MD_PATH.write_text(md)
    print(f"wrote: {MD_PATH}")
    print()
    print(md)


if __name__ == "__main__":
    main()
