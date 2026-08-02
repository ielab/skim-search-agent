#!/usr/bin/env python
"""Second backbone (Qwen-AgentWorld-35B-A3B) on MuSiQue: closes the confound where the second
backbone previously existed on browsecomp_plus_structured ONLY, which meant "backbone dependence"
and "dataset identity" could not be told apart. Two new cells just finished and merged
(runs/_xbackbone/agent/musique_structured/Qwen-AgentWorld-35B-A3B/{agent_research_bm25,
agent_research_bql_dense_snip}/rows.jsonl, both n=2409) so the SAME second backbone now has the
baseline and the full method (Sieve) on MuSiQue too.

This script does two things:
  1. A SANITY GATE (runs first): reproduces the already-published second-backbone EM delta on
     browsecomp_plus_structured (+18.3, from analysis/xbackbone_full.md) using the exact same
     reused primitives this script uses for the new MuSiQue numbers. If this does not reproduce,
     the MuSiQue numbers below are not to be trusted either — the gate result is printed and
     written to the JSON/markdown outputs either way.
  2. The NEW MuSiQue paired comparison (EM, count-once tokens, LLM calls, gold-document recall)
     for the second backbone, on the shared instance set (inner join on instance_id).
  3. The full 2-backbone x 2-dataset grid (EM delta, token reduction %), assembled from this
     script's own new numbers PLUS companion numbers already computed elsewhere in the repo and
     read (not recomputed) from their existing artifacts:
       - primary backbone (Tongyi-DeepResearch-30B-A3B) x browsecomp_plus_structured and x
         musique_structured: analysis/xbackbone_full.md (item 5) and
         analysis/token_efficiency_data.json (precise tok_once means) respectively.
       - second backbone (Qwen-AgentWorld-35B-A3B) x browsecomp_plus_structured:
         analysis/xbackbone_full.md / analysis/xbackbone_full_data.json.

REUSE, not reimplementation (per project directive):
  - `evaluation.metrics.answer_em`                    — EM scoring (inside `compare_cells.metrics`)
  - `scripts.compare_cells.metrics`                    — per-row em/recall/tok/tok_in/tok_out/llm_calls
  - `scripts.compare_cells.mcnemar_p`                  — exact two-sided McNemar
  - `scripts.compare_cells.load_qrels`/`gold_doc_recall` (used INSIDE `metrics()`) — gold-document
                                                          recall, same definition as comparison_result.md's
                                                          recall% column
  - `scripts.compare_cells.load_judge_cache`           — sibling judge_cache.jsonl reader (called for
                                                          parity even though MuSiQue has no judge_cache —
                                                          `metrics()` just gets an empty dict back)
  - `scripts.force_answer_backfill.load_rows_with_recovery` — recovery-overlaid row loader

MuSiQue is an EM/F1 benchmark and is NOT judged — no judge numbers are computed or reported here.

Usage:
    PYTHONPATH=. envs/bin/python analysis/xbackbone_musique.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats  # noqa: E402

from scripts.compare_cells import (  # noqa: E402
    MODEL_DIR as PRIMARY_MODEL_DIR,
    load_judge_cache,
    load_qrels,
    mcnemar_p,
    metrics,
)
from scripts.force_answer_backfill import load_rows_with_recovery  # noqa: E402

SECOND_MODEL_DIR = "Qwen-AgentWorld-35B-A3B"
XBB_ROOT = Path("runs/_xbackbone/agent")
BASELINE_COND = "agent_research_bm25"
METHOD_COND = "agent_research_bql_dense_snip"

# Sanity-gate target: the already-published second-backbone EM delta on browsecomp_plus_structured
# (analysis/xbackbone_full.md, item 1).
GATE_DATASET = "browsecomp_plus_structured"
GATE_EXPECTED_EM_DELTA = 18.3
GATE_EXPECTED_EM_P = 1.2e-22

MUSIQUE_DATASET = "musique_structured"


def pct(xs) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _cond_dir(model_dir: str, dataset: str, cond: str) -> Path:
    return XBB_ROOT / dataset / model_dir / cond


def _cell_metrics(model_dir: str, dataset: str, cond: str, qrels: dict | None) -> dict:
    """instance_id -> per-row metrics dict, via the SAME `compare_cells.metrics`/
    `load_rows_with_recovery` primitives used across the repo's analysis scripts."""
    cond_dir = _cond_dir(model_dir, dataset, cond)
    rows = load_rows_with_recovery(str(cond_dir))
    return metrics(rows, qrels=qrels, dataset=dataset, judge_cache=load_judge_cache(cond_dir))


def run_sanity_gate() -> dict:
    """Reproduce the already-published second-backbone browsecomp_plus_structured EM delta
    (+18.3, analysis/xbackbone_full.md) using this script's own primitives, BEFORE trusting the
    new MuSiQue numbers below."""
    print("=" * 78)
    print("SANITY GATE — reproducing published second-backbone EM delta on "
          f"{GATE_DATASET}")
    print("=" * 78)
    base_m = _cell_metrics(SECOND_MODEL_DIR, GATE_DATASET, BASELINE_COND, qrels=None)
    meth_m = _cell_metrics(SECOND_MODEL_DIR, GATE_DATASET, METHOD_COND, qrels=None)
    shared = sorted(set(base_m) & set(meth_m))
    em_base = pct([base_m[i]["em"] for i in shared])
    em_meth = pct([meth_m[i]["em"] for i in shared])
    b = sum(1 for i in shared if meth_m[i]["em"] and not base_m[i]["em"])
    c = sum(1 for i in shared if base_m[i]["em"] and not meth_m[i]["em"])
    p = float(mcnemar_p(b, c))
    delta = em_meth - em_base
    passed = abs(delta - GATE_EXPECTED_EM_DELTA) / GATE_EXPECTED_EM_DELTA <= 0.05 and \
        abs(math.log10(max(p, 1e-300)) - math.log10(GATE_EXPECTED_EM_P)) < 0.3
    print(f"  n_shared={len(shared)}  baseline={em_base:.2f}%  method={em_meth:.2f}%  "
          f"delta={delta:+.2f}  (expected +{GATE_EXPECTED_EM_DELTA})  p={p:.4g} "
          f"(expected ~{GATE_EXPECTED_EM_P:.2g})")
    print(f"  GATE {'PASSED' if passed else 'FAILED'}")
    return dict(dataset=GATE_DATASET, n_shared=len(shared), em_baseline=em_base, em_method=em_meth,
                em_delta=delta, em_b=b, em_c=c, em_p=p,
                expected_em_delta=GATE_EXPECTED_EM_DELTA, expected_em_p=GATE_EXPECTED_EM_P,
                passed=bool(passed))


def run_musique() -> dict:
    print("\n" + "=" * 78)
    print(f"SECOND BACKBONE ({SECOND_MODEL_DIR}) x {MUSIQUE_DATASET} — Sieve vs baseline")
    print("=" * 78)

    qrels = load_qrels(MUSIQUE_DATASET)
    base_m = _cell_metrics(SECOND_MODEL_DIR, MUSIQUE_DATASET, BASELINE_COND, qrels=qrels)
    meth_m = _cell_metrics(SECOND_MODEL_DIR, MUSIQUE_DATASET, METHOD_COND, qrels=qrels)
    n_baseline, n_method = len(base_m), len(meth_m)
    shared = sorted(set(base_m) & set(meth_m))
    n_shared = len(shared)
    print(f"\nn_baseline={n_baseline}  n_method={n_method}  n_shared={n_shared}")

    result: dict = dict(
        dataset=MUSIQUE_DATASET, second_backbone=SECOND_MODEL_DIR,
        primary_backbone=PRIMARY_MODEL_DIR,
        baseline_cond=BASELINE_COND, method_cond=METHOD_COND,
        n_baseline=n_baseline, n_method=n_method, n_shared=n_shared,
    )

    # --- 1. EM -------------------------------------------------------------------------------
    em_base = pct([base_m[i]["em"] for i in shared])
    em_meth = pct([meth_m[i]["em"] for i in shared])
    b_em = sum(1 for i in shared if meth_m[i]["em"] and not base_m[i]["em"])
    c_em = sum(1 for i in shared if base_m[i]["em"] and not meth_m[i]["em"])
    p_em = float(mcnemar_p(b_em, c_em))
    delta_em = em_meth - em_base
    sig_em = p_em < 0.05
    print("\n--- 1. EM (exact match) ---")
    print(f"  baseline={em_base:.2f}%  method={em_meth:.2f}%  delta={delta_em:+.2f}  "
          f"b(method_only)={b_em}  c(baseline_only)={c_em}  p={p_em:.4g}  "
          f"significant@0.05={sig_em}")
    result.update(em_baseline=em_base, em_method=em_meth, delta_em=delta_em,
                   em_b=b_em, em_c=c_em, em_p=p_em, em_significant_05=sig_em)

    # --- 2. Tokens per episode (count-once) ---------------------------------------------------
    print("\n--- 2. Tokens per episode (count-once: initial_prompt + context_once + output) ---")
    tok_base = [base_m[i]["tok"] for i in shared]
    tok_meth = [meth_m[i]["tok"] for i in shared]
    mean_tok_base, mean_tok_meth = mean(tok_base), mean(tok_meth)
    tok_reduction_pct = 100.0 * (mean_tok_base - mean_tok_meth) / mean_tok_base if mean_tok_base else float("nan")
    t_tok = stats.ttest_rel(tok_meth, tok_base)
    w_tok = stats.wilcoxon(tok_meth, tok_base)
    print(f"  mean tok baseline={mean_tok_base:,.0f}  mean tok method={mean_tok_meth:,.0f}  "
          f"reduction={tok_reduction_pct:+.2f}%")
    print(f"  PAIRED T-TEST (declared/reported test): t={t_tok.statistic:.3f}  p={t_tok.pvalue:.4g}")
    print(f"  Wilcoxon signed-rank (secondary, NOT the declared test): "
          f"stat={w_tok.statistic:.1f}  p={w_tok.pvalue:.4g}")
    result.update(tok_baseline_mean=mean_tok_base, tok_method_mean=mean_tok_meth,
                   tok_reduction_pct=tok_reduction_pct,
                   tok_ttest_stat=float(t_tok.statistic), tok_ttest_p=float(t_tok.pvalue),
                   tok_wilcoxon_stat=float(w_tok.statistic), tok_wilcoxon_p=float(w_tok.pvalue))

    # --- 3. LLM calls per episode --------------------------------------------------------------
    print("\n--- 3. LLM calls per episode ---")
    calls_base = [base_m[i]["llm_calls"] for i in shared]
    calls_meth = [meth_m[i]["llm_calls"] for i in shared]
    mean_calls_base, mean_calls_meth = mean(calls_base), mean(calls_meth)
    t_calls = stats.ttest_rel(calls_meth, calls_base)
    w_calls = stats.wilcoxon(calls_meth, calls_base)
    print(f"  mean calls baseline={mean_calls_base:.2f}  mean calls method={mean_calls_meth:.2f}  "
          f"delta={mean_calls_meth - mean_calls_base:+.2f}")
    print(f"  PAIRED T-TEST (declared/reported test): t={t_calls.statistic:.3f}  p={t_calls.pvalue:.4g}")
    print(f"  Wilcoxon signed-rank (secondary, NOT the declared test): "
          f"stat={w_calls.statistic:.1f}  p={w_calls.pvalue:.4g}")
    result.update(llm_calls_baseline_mean=mean_calls_base, llm_calls_method_mean=mean_calls_meth,
                   llm_calls_delta=mean_calls_meth - mean_calls_base,
                   llm_calls_ttest_stat=float(t_calls.statistic), llm_calls_ttest_p=float(t_calls.pvalue),
                   llm_calls_wilcoxon_stat=float(w_calls.statistic), llm_calls_wilcoxon_p=float(w_calls.pvalue))

    # --- 4. Gold-document recall ----------------------------------------------------------------
    print("\n--- 4. Gold-document recall ---")
    recall_base = pct([base_m[i]["recall"] for i in shared])
    recall_meth = pct([meth_m[i]["recall"] for i in shared])
    b_rec = sum(1 for i in shared if meth_m[i]["recall"] and not base_m[i]["recall"])
    c_rec = sum(1 for i in shared if base_m[i]["recall"] and not meth_m[i]["recall"])
    p_rec = float(mcnemar_p(b_rec, c_rec))
    print(f"  n_qrels_queries={len(qrels)}")
    print(f"  baseline={recall_base:.2f}%  method={recall_meth:.2f}%  delta={recall_meth - recall_base:+.2f}  "
          f"b(method_only)={b_rec}  c(baseline_only)={c_rec}  p={p_rec:.4g}")
    result.update(recall_baseline=recall_base, recall_method=recall_meth,
                   recall_delta=recall_meth - recall_base,
                   recall_b=b_rec, recall_c=c_rec, recall_p=p_rec, n_qrels_queries=len(qrels))

    return result


# --- 2x2 grid: read companion numbers from existing artifacts, never recomputed ---------------
def load_companion_numbers() -> dict:
    """Primary-backbone x {browsecomp,musique} and second-backbone x browsecomp numbers, read
    directly from the JSON artifacts other analysis scripts already wrote (never recomputed
    here)."""
    xbb_full = json.loads((Path(__file__).resolve().parent / "xbackbone_full_data.json").read_text())
    tok_eff = json.loads((Path(__file__).resolve().parent / "token_efficiency_data.json").read_text())
    tok_eff_by_ds = {item["dataset"]: item for item in tok_eff if isinstance(item, dict) and "dataset" in item}

    pb_bc = tok_eff_by_ds["browsecomp_plus_structured"]
    pb_mu = tok_eff_by_ds["musique_structured"]

    return dict(
        primary_backbone=xbb_full["primary_backbone"],
        second_backbone=xbb_full["second_backbone"],
        # primary backbone x browsecomp_plus_structured (source: token_efficiency_data.json +
        # xbackbone_full_data.json's own item-5 primary-backbone contrast, cross-checked equal)
        primary_browsecomp_em_delta=float(pb_bc["accuracy"]["d_em"]),
        primary_browsecomp_em_p=float(pb_bc["accuracy"]["p_em"]),
        primary_browsecomp_tok_reduction_pct=xbb_full["primary_backbone_tok_reduction_pct"],
        # primary backbone x musique_structured (source: token_efficiency_data.json)
        primary_musique_em_delta=float(pb_mu["accuracy"]["d_em"]),
        primary_musique_em_p=float(pb_mu["accuracy"]["p_em"]),
        primary_musique_tok_reduction_pct=pb_mu["tok_once"]["pct_reduction"],
        # second backbone x browsecomp_plus_structured (source: xbackbone_full_data.json)
        second_browsecomp_em_delta=xbb_full["delta_em"],
        second_browsecomp_em_p=xbb_full["em_p"],
        second_browsecomp_tok_reduction_pct=xbb_full["tok_reduction_pct"],
        second_browsecomp_judge_delta=xbb_full.get("delta_judge"),
    )


def main() -> int:
    gate = run_sanity_gate()
    if not gate["passed"]:
        print("\n" + "!" * 78)
        print("SANITY GATE FAILED — the reused primitives did not reproduce the published "
              "second-backbone browsecomp_plus_structured EM delta. STOPPING before computing "
              "or trusting the new MuSiQue numbers.")
        print("!" * 78)
        out_json = Path(__file__).resolve().parent / "xbackbone_musique_data.json"
        out_json.write_text(json.dumps(dict(sanity_gate=gate, aborted=True), indent=2))
        return 1

    musique = run_musique()
    companion = load_companion_numbers()

    grid = dict(
        primary_backbone=companion["primary_backbone"],
        second_backbone=companion["second_backbone"],
        rows=[
            dict(backbone=companion["primary_backbone"], dataset="browsecomp_plus_structured",
                 em_delta=companion["primary_browsecomp_em_delta"],
                 em_p=companion["primary_browsecomp_em_p"],
                 tok_reduction_pct=companion["primary_browsecomp_tok_reduction_pct"],
                 source="analysis/token_efficiency_data.json + analysis/xbackbone_full.md (item 5)"),
            dict(backbone=companion["primary_backbone"], dataset="musique_structured",
                 em_delta=companion["primary_musique_em_delta"],
                 em_p=companion["primary_musique_em_p"],
                 tok_reduction_pct=companion["primary_musique_tok_reduction_pct"],
                 source="analysis/token_efficiency_data.json"),
            dict(backbone=companion["second_backbone"], dataset="browsecomp_plus_structured",
                 em_delta=companion["second_browsecomp_em_delta"],
                 em_p=companion["second_browsecomp_em_p"],
                 tok_reduction_pct=companion["second_browsecomp_tok_reduction_pct"],
                 source="analysis/xbackbone_full.md / analysis/xbackbone_full_data.json"),
            dict(backbone=companion["second_backbone"], dataset="musique_structured",
                 em_delta=musique["delta_em"],
                 em_p=musique["em_p"],
                 tok_reduction_pct=musique["tok_reduction_pct"],
                 source="THIS SCRIPT (new)"),
        ],
    )

    print("\n" + "=" * 78)
    print("2 BACKBONE x 2 DATASET GRID (EM delta, token reduction %)")
    print("=" * 78)
    for row in grid["rows"]:
        print(f"  {row['backbone']:28s} x {row['dataset']:28s}  "
              f"EM delta={row['em_delta']:+.2f} (p={row['em_p']:.3g})  "
              f"token reduction={row['tok_reduction_pct']:+.2f}%   [{row['source']}]")

    result = dict(sanity_gate=gate, musique=musique, grid=grid, aborted=False)

    out_json = Path(__file__).resolve().parent / "xbackbone_musique_data.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")

    _write_markdown_report(result)
    return 0


def _write_markdown_report(result: dict) -> None:
    gate = result["sanity_gate"]
    m = result["musique"]
    grid = result["grid"]
    md_path = Path(__file__).resolve().parent / "xbackbone_musique.md"
    lines = []
    lines.append("# Cross-backbone x cross-dataset: second backbone on MuSiQue")
    lines.append("")
    lines.append("Generated by `analysis/xbackbone_musique.py`. Closes the confound where the "
                  "second backbone (Qwen-AgentWorld-35B-A3B) previously existed on "
                  "browsecomp_plus_structured only, which meant \"backbone dependence\" and "
                  "\"dataset identity\" could not be told apart.")
    lines.append("")

    lines.append("## Sanity gate")
    lines.append("")
    lines.append(f"Reproduces the already-published second-backbone EM delta on "
                  f"**{gate['dataset']}** (analysis/xbackbone_full.md), using this script's own "
                  f"primitives, before trusting the new MuSiQue numbers below.")
    lines.append("")
    lines.append(f"n_shared={gate['n_shared']}  baseline={gate['em_baseline']:.2f}%  "
                  f"method={gate['em_method']:.2f}%  delta={gate['em_delta']:+.2f} "
                  f"(expected +{gate['expected_em_delta']})  p={gate['em_p']:.4g} "
                  f"(expected ~{gate['expected_em_p']:.2g})")
    lines.append("")
    lines.append(f"**GATE {'PASSED' if gate['passed'] else 'FAILED'}**")
    lines.append("")

    lines.append("## Second backbone (Qwen-AgentWorld-35B-A3B) x MuSiQue — NEW paired result")
    lines.append("")
    lines.append(f"Baseline `{m['baseline_cond']}` vs full method (Sieve) `{m['method_cond']}`, "
                  f"dataset `{m['dataset']}`, on the SHARED instance set (inner join on "
                  f"instance_id). n_baseline={m['n_baseline']}  n_method={m['n_method']}  "
                  f"**n_shared={m['n_shared']}**.")
    lines.append("")
    lines.append("| # | quantity | baseline | method | delta | test | statistic | p-value | significant@0.05 |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|:---:|")
    lines.append(f"| 1 | EM% | {m['em_baseline']:.2f} | {m['em_method']:.2f} | {m['delta_em']:+.2f} "
                  f"| exact McNemar (b={m['em_b']}, c={m['em_c']}) | — | {m['em_p']:.4g} | "
                  f"{'yes' if m['em_significant_05'] else 'no'} |")
    lines.append(f"| 2 | tokens/episode (count-once) | {m['tok_baseline_mean']:,.0f} | "
                  f"{m['tok_method_mean']:,.0f} | {m['tok_reduction_pct']:+.2f}% (reduction) | "
                  f"**paired t-test (declared/reported)** | {m['tok_ttest_stat']:.3f} | "
                  f"{m['tok_ttest_p']:.4g} | {'yes' if m['tok_ttest_p'] < 0.05 else 'no'} |")
    lines.append(f"|   | tokens/episode (count-once) | {m['tok_baseline_mean']:,.0f} | "
                  f"{m['tok_method_mean']:,.0f} | {m['tok_reduction_pct']:+.2f}% (reduction) | "
                  f"Wilcoxon signed-rank (secondary) | {m['tok_wilcoxon_stat']:.1f} | "
                  f"{m['tok_wilcoxon_p']:.4g} | {'yes' if m['tok_wilcoxon_p'] < 0.05 else 'no'} |")
    lines.append(f"| 3 | LLM calls/episode | {m['llm_calls_baseline_mean']:.2f} | "
                  f"{m['llm_calls_method_mean']:.2f} | {m['llm_calls_delta']:+.2f} | "
                  f"**paired t-test (declared/reported)** | {m['llm_calls_ttest_stat']:.3f} | "
                  f"{m['llm_calls_ttest_p']:.4g} | {'yes' if m['llm_calls_ttest_p'] < 0.05 else 'no'} |")
    lines.append(f"|   | LLM calls/episode | {m['llm_calls_baseline_mean']:.2f} | "
                  f"{m['llm_calls_method_mean']:.2f} | {m['llm_calls_delta']:+.2f} | "
                  f"Wilcoxon signed-rank (secondary) | {m['llm_calls_wilcoxon_stat']:.1f} | "
                  f"{m['llm_calls_wilcoxon_p']:.4g} | "
                  f"{'yes' if m['llm_calls_wilcoxon_p'] < 0.05 else 'no'} |")
    lines.append(f"| 4 | gold-document recall% | {m['recall_baseline']:.2f} | "
                  f"{m['recall_method']:.2f} | {m['recall_delta']:+.2f} | exact McNemar "
                  f"(b={m['recall_b']}, c={m['recall_c']}) | — | {m['recall_p']:.4g} | "
                  f"{'yes' if m['recall_p'] < 0.05 else 'no'} |")
    lines.append("")
    lines.append(f"(n_qrels_queries={m['n_qrels_queries']} distinct queries with at least one "
                  f"gold corpus id in `data/musique_structured/qrels/`.)")
    lines.append("")

    lines.append("## 2 backbone x 2 dataset grid (EM delta, token reduction %)")
    lines.append("")
    lines.append("Rows 1-3 are companion numbers ALREADY COMPUTED elsewhere in the repo, read "
                  "(never recomputed) from their existing artifacts. Row 4 (second backbone x "
                  "MuSiQue) is the new result from this script.")
    lines.append("")
    lines.append("| backbone | dataset | EM delta | EM p | token reduction % | source |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in grid["rows"]:
        lines.append(f"| {row['backbone']} | {row['dataset']} | {row['em_delta']:+.2f} | "
                      f"{row['em_p']:.3g} | {row['tok_reduction_pct']:+.2f}% | {row['source']} |")
    lines.append("")

    pb = grid["rows"][0]["backbone"]
    sb = grid["rows"][2]["backbone"]
    pb_bc, pb_mu = grid["rows"][0], grid["rows"][1]
    sb_bc, sb_mu = grid["rows"][2], grid["rows"][3]
    lines.append("### Reading the grid")
    lines.append("")
    lines.append(f"- **{pb}** (primary backbone): EM delta {pb_bc['em_delta']:+.2f} on BrowseComp "
                  f"vs {pb_mu['em_delta']:+.2f} on MuSiQue — a small, dataset-stable positive "
                  f"margin.")
    lines.append(f"- **{sb}** (second backbone): EM delta {sb_bc['em_delta']:+.2f} on BrowseComp "
                  f"vs {sb_mu['em_delta']:+.2f} on MuSiQue.")
    if sb_mu["em_delta"] >= 0.5 * sb_bc["em_delta"] and sb_mu["em_delta"] > pb_mu["em_delta"]:
        verdict = ("The second backbone's large EM margin REPLICATES on MuSiQue (well above the "
                   "primary backbone's MuSiQue margin) — this looks like a genuine backbone "
                   "effect, not a BrowseComp artifact.")
    elif sb_mu["em_delta"] < 0.3 * sb_bc["em_delta"]:
        verdict = ("The second backbone's large EM margin does NOT replicate on MuSiQue (it "
                   "collapses toward the primary backbone's MuSiQue margin) — this looks "
                   "BrowseComp-specific, not a general backbone effect.")
    else:
        verdict = ("The second backbone's EM margin on MuSiQue is directionally positive but "
                   "smaller than on BrowseComp — a partial replication, not a clean backbone "
                   "effect nor a pure BrowseComp artifact.")
    lines.append(f"- **Verdict**: {verdict}")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    raise SystemExit(main())
