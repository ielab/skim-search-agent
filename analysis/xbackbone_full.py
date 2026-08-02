#!/usr/bin/env python
"""Full cross-backbone analysis artifact: every cross-backbone number that appears in the
paper, backed by a reproducible script.

`analysis/xbackbone_analysis.py` covers only the EM replication check (and is left untouched
— this is a NEW, separate script/artifact, per the task that created it). Several other
cross-backbone quantities (judge accuracy, token reduction, LLM-call reduction, and the
primary-vs-second-backbone token-reduction contrast) were computed ad hoc during a paper
revision and never got a script backing them. This script closes that gap.

Both cells (second backbone Qwen-AgentWorld-35B-A3B, browsecomp_plus_structured,
baseline=agent_research_bm25 vs full method=agent_research_bql_dense_snip) are COMPLETE at
n=830 in `runs/_xbackbone/agent/browsecomp_plus_structured/Qwen-AgentWorld-35B-A3B/<cond>/`
(confirmed against data/browsecomp_plus_structured/queries.jsonl, also n=830), so unlike
xbackbone_analysis.py this script does not need the partial-run/shard-union machinery — it
reads the canonical `rows.jsonl` + sibling `judge_cache.jsonl` directly.

REUSE, not reimplementation (per project directive — a subtly different definition would
silently contradict the rest of the paper):
  - `evaluation.metrics.answer_em`            — EM scoring (used INSIDE `compare_cells.metrics`)
  - `scripts.compare_cells.metrics`           — per-row em/judge/tok/tok_in/tok_out/llm_calls,
                                                 including the exact count-once token fallback
                                                 (`total_tokens_once` else
                                                 initial_prompt_tokens + context_once_tokens +
                                                 output_tokens) that backs comparison_result.md's
                                                 `avg_tok/inst` column.
  - `scripts.compare_cells.mcnemar_p`         — exact two-sided McNemar (binomial on discordant
                                                 pairs) for EM and judge accuracy.
  - `scripts.compare_cells.load_judge_cache`  — sibling judge_cache.jsonl reader.
  - `scripts.compare_cells.REGISTRY`/`cell_dir` — canonical dirs for the PRIMARY backbone's
                                                 matching baseline/method cells (item 5), so that
                                                 comparison is never hand-pathed.
  - `scripts.force_answer_backfill.load_rows_with_recovery` — recovery-overlaid row loader
                                                 (same primitive `compare_cells.cell_rows` uses).

Every quantity is computed on the SHARED instance set (inner join on instance_id) unless noted
otherwise, and n_shared is always reported. The paper declares the PAIRED T-TEST as its
reported significance test for tokens/llm_calls; the Wilcoxon signed-rank result is ALSO
computed and printed, but clearly labelled as secondary — a prior revision leaked a Wilcoxon
p-value where a t-test p-value was declared, and this script exists partly to prevent that
recurring.

Judge coverage gate: mirrors the >=90%-judged gate `scripts/compare_cells.py` itself applies
before rendering a judge% cell (`judged and len(judged) >= 0.9 * len(vals)`) — below that
threshold this script reports "insufficient coverage" instead of a number.

Usage:
    PYTHONPATH=. envs/bin/python analysis/xbackbone_full.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats  # noqa: E402

from scripts.compare_cells import (  # noqa: E402
    MODEL_DIR as PRIMARY_MODEL_DIR,
    REGISTRY,
    cell_dir,
    cell_rows,
    load_judge_cache,
    mcnemar_p,
    metrics,
)
from scripts.force_answer_backfill import load_rows_with_recovery  # noqa: E402

DATASET = "browsecomp_plus_structured"
SECOND_MODEL_DIR = "Qwen-AgentWorld-35B-A3B"
XBB_ROOT = Path("runs/_xbackbone/agent")
BASELINE_COND = "agent_research_bm25"
METHOD_COND = "agent_research_bql_dense_snip"
BASELINE_LABEL = "SERP bm25 [BASELINE]"   # exact REGISTRY label, disambiguates from
METHOD_LABEL = "bql+dense+snip fetch"     # "SERP bm25 k=10" / "bm25 auto-read" etc (same cond)

JUDGE_COVERAGE_THRESHOLD = 0.90  # same gate compare_cells.py applies before rendering judge%

# Values quoted in the task brief as independently-computed expectations. Compared against
# (never substituted for) this script's own numbers; any mismatch is flagged explicitly in the
# output, both numbers shown, rather than silently preferring one.
EXPECTED = dict(
    em_delta=18.3, em_p=1.2e-22,
    judge_baseline=25.06, judge_method=44.10, judge_delta=19.04, judge_p=2.644e-23,
    judge_b=212, judge_c=54,
    tok_reduction_pct=17.4,
    llm_calls_baseline=36.16, llm_calls_method=25.07,
    primary_backbone_tok_reduction_pct=29.6,
)


def _second_backbone_dir(cond: str) -> Path:
    return XBB_ROOT / DATASET / SECOND_MODEL_DIR / cond


def _cell_metrics_second(cond: str) -> dict:
    """instance_id -> per-row metrics dict (em/judge/tok/tok_in/tok_out/llm_calls) for the
    second backbone, via the SAME `compare_cells.metrics`/`load_rows_with_recovery` primitives
    the primary-backbone comparison table uses. qrels=None (recall/surfaced unused here, we only
    read em/judge/tok/llm_calls off the returned dict)."""
    cond_dir = _second_backbone_dir(cond)
    rows = load_rows_with_recovery(str(cond_dir))
    return metrics(rows, qrels=None, dataset=DATASET, judge_cache=load_judge_cache(cond_dir))


def _primary_backbone_cell(label: str, cond: str) -> tuple[dict, int]:
    """(metrics_dict, n) for the PRIMARY backbone's matching cell, resolved via
    REGISTRY (label+dataset+cond, exact match — several REGISTRY rows share the same `cond`
    string under different subdirs/labels, e.g. 'SERP bm25 [BASELINE]' vs 'SERP bm25 k=10' both
    use agent_research_bm25, so label disambiguates) + `cell_dir`/`cell_rows`, never a
    hardcoded path."""
    matches = [e for e in REGISTRY if e[2] == DATASET and e[3] == cond and e[0] == label]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one REGISTRY match for label={label!r} dataset={DATASET!r} "
            f"cond={cond!r}, got {len(matches)}"
        )
    _, subdir, _, _, _ = matches[0]
    cdir = cell_dir(subdir, DATASET, cond)
    rows = cell_rows(subdir, DATASET, cond)
    if rows is None:
        raise RuntimeError(f"no rows.jsonl at {cdir}")
    m = metrics(rows, qrels=None, dataset=DATASET, judge_cache=load_judge_cache(cdir))
    return m, len(rows)


def pct(xs) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _flag(label: str, computed: float, expected: float, *, rel_tol: float = 0.05,
          is_pvalue: bool = False) -> dict:
    """Compare `computed` to the task brief's `expected` value. For p-values, match by order of
    magnitude (log10 difference < 0.3, i.e. within ~2x) since exact p-values are sensitive to
    the last discordant pair; for everything else, relative tolerance `rel_tol` (default 5%)."""
    import math
    if is_pvalue:
        try:
            ok = abs(math.log10(max(computed, 1e-300)) - math.log10(max(expected, 1e-300))) < 0.3
        except ValueError:
            ok = False
    else:
        denom = abs(expected) if expected else 1.0
        ok = abs(computed - expected) / denom <= rel_tol
    return dict(label=label, computed=computed, expected=expected, matches=bool(ok))


def main() -> int:
    print("=" * 78)
    print(f"CROSS-BACKBONE FULL ANALYSIS — {SECOND_MODEL_DIR} / {DATASET}")
    print(f"baseline={BASELINE_COND}  method={METHOD_COND}")
    print("=" * 78)

    base_m = _cell_metrics_second(BASELINE_COND)
    meth_m = _cell_metrics_second(METHOD_COND)
    n_baseline, n_method = len(base_m), len(meth_m)
    shared = sorted(set(base_m) & set(meth_m))
    n_shared = len(shared)
    print(f"\nn_baseline={n_baseline}  n_method={n_method}  n_shared={n_shared}")

    checks: list[dict] = []
    result: dict = dict(
        dataset=DATASET, second_backbone=SECOND_MODEL_DIR,
        primary_backbone=PRIMARY_MODEL_DIR,
        baseline_cond=BASELINE_COND, method_cond=METHOD_COND,
        n_baseline=n_baseline, n_method=n_method, n_shared=n_shared,
    )

    # --- 1. EM -----------------------------------------------------------------------------
    em_base = pct([base_m[i]["em"] for i in shared])
    em_meth = pct([meth_m[i]["em"] for i in shared])
    b_em = sum(1 for i in shared if meth_m[i]["em"] and not base_m[i]["em"])
    c_em = sum(1 for i in shared if base_m[i]["em"] and not meth_m[i]["em"])
    p_em = float(mcnemar_p(b_em, c_em))
    delta_em = em_meth - em_base
    print("\n--- 1. EM (exact match) ---")
    print(f"  baseline={em_base:.2f}%  method={em_meth:.2f}%  delta={delta_em:+.2f}  "
          f"b(method_only)={b_em}  c(baseline_only)={c_em}  p={p_em:.4g}")
    checks.append(_flag("EM delta (pts)", delta_em, EXPECTED["em_delta"]))
    checks.append(_flag("EM McNemar p", p_em, EXPECTED["em_p"], is_pvalue=True))
    result.update(em_baseline=em_base, em_method=em_meth, delta_em=delta_em,
                   em_b=b_em, em_c=c_em, em_p=p_em)

    # --- 2. LLM-as-judge accuracy -----------------------------------------------------------
    print("\n--- 2. LLM-as-judge accuracy ---")
    cov_base = sum(1 for i in shared if base_m[i]["judge"] is not None) / n_shared if n_shared else 0.0
    cov_meth = sum(1 for i in shared if meth_m[i]["judge"] is not None) / n_shared if n_shared else 0.0
    print(f"  judge coverage on shared set: baseline={100*cov_base:.1f}%  method={100*cov_meth:.1f}%"
          f"  (gate: >= {100*JUDGE_COVERAGE_THRESHOLD:.0f}%)")
    if cov_base < JUDGE_COVERAGE_THRESHOLD or cov_meth < JUDGE_COVERAGE_THRESHOLD:
        print("  INSUFFICIENT COVERAGE — no judge number reported.")
        result.update(judge="insufficient coverage")
    else:
        jshared = [i for i in shared if base_m[i]["judge"] is not None and meth_m[i]["judge"] is not None]
        j_base = pct([base_m[i]["judge"] for i in jshared])
        j_meth = pct([meth_m[i]["judge"] for i in jshared])
        b_j = sum(1 for i in jshared if meth_m[i]["judge"] and not base_m[i]["judge"])
        c_j = sum(1 for i in jshared if base_m[i]["judge"] and not meth_m[i]["judge"])
        p_j = float(mcnemar_p(b_j, c_j))
        delta_j = j_meth - j_base
        print(f"  baseline={j_base:.2f}%  method={j_meth:.2f}%  delta={delta_j:+.2f}  "
              f"b(method_only)={b_j}  c(baseline_only)={c_j}  p={p_j:.4g}  (n_judged={len(jshared)})")
        checks.append(_flag("judge baseline%", j_base, EXPECTED["judge_baseline"]))
        checks.append(_flag("judge method%", j_meth, EXPECTED["judge_method"]))
        checks.append(_flag("judge delta (pts)", delta_j, EXPECTED["judge_delta"]))
        checks.append(_flag("judge McNemar p", p_j, EXPECTED["judge_p"], is_pvalue=True))
        checks.append(_flag("judge b (method_only)", b_j, EXPECTED["judge_b"], rel_tol=0.0))
        checks.append(_flag("judge c (baseline_only)", c_j, EXPECTED["judge_c"], rel_tol=0.0))
        result.update(judge_baseline=j_base, judge_method=j_meth, delta_judge=delta_j,
                       judge_b=b_j, judge_c=c_j, judge_p=p_j, n_judged_shared=len(jshared))

    # --- 3. Tokens per episode (count-once) -------------------------------------------------
    print("\n--- 3. Tokens per episode (count-once: initial_prompt + context_once + output) ---")
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
    checks.append(_flag("token reduction %", tok_reduction_pct, EXPECTED["tok_reduction_pct"]))
    result.update(tok_baseline_mean=mean_tok_base, tok_method_mean=mean_tok_meth,
                   tok_reduction_pct=tok_reduction_pct,
                   tok_ttest_stat=float(t_tok.statistic), tok_ttest_p=float(t_tok.pvalue),
                   tok_wilcoxon_stat=float(w_tok.statistic), tok_wilcoxon_p=float(w_tok.pvalue))

    # --- 4. LLM calls per episode ------------------------------------------------------------
    print("\n--- 4. LLM calls per episode ---")
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
    checks.append(_flag("llm_calls baseline mean", mean_calls_base, EXPECTED["llm_calls_baseline"]))
    checks.append(_flag("llm_calls method mean", mean_calls_meth, EXPECTED["llm_calls_method"]))
    result.update(llm_calls_baseline_mean=mean_calls_base, llm_calls_method_mean=mean_calls_meth,
                   llm_calls_delta=mean_calls_meth - mean_calls_base,
                   llm_calls_ttest_stat=float(t_calls.statistic), llm_calls_ttest_p=float(t_calls.pvalue),
                   llm_calls_wilcoxon_stat=float(w_calls.statistic), llm_calls_wilcoxon_p=float(w_calls.pvalue))

    # --- 5. Primary-backbone token reduction, same dataset/condition pair, for contrast -----
    print("\n--- 5. Primary backbone token reduction (for direct contrast) ---")
    pb_base_m, pb_n_base = _primary_backbone_cell(BASELINE_LABEL, BASELINE_COND)
    pb_meth_m, pb_n_meth = _primary_backbone_cell(METHOD_LABEL, METHOD_COND)
    pb_shared = sorted(set(pb_base_m) & set(pb_meth_m))
    pb_tok_base = [pb_base_m[i]["tok"] for i in pb_shared]
    pb_tok_meth = [pb_meth_m[i]["tok"] for i in pb_shared]
    pb_mean_base, pb_mean_meth = mean(pb_tok_base), mean(pb_tok_meth)
    pb_reduction_pct = 100.0 * (pb_mean_base - pb_mean_meth) / pb_mean_base if pb_mean_base else float("nan")
    print(f"  {PRIMARY_MODEL_DIR}: n_baseline={pb_n_base} n_method={pb_n_meth} n_shared={len(pb_shared)}")
    print(f"  mean tok baseline={pb_mean_base:,.0f}  mean tok method={pb_mean_meth:,.0f}  "
          f"reduction={pb_reduction_pct:+.2f}%")
    print(f"\n  CONTRAST: {SECOND_MODEL_DIR} token reduction = {tok_reduction_pct:+.2f}%   "
          f"vs   {PRIMARY_MODEL_DIR} token reduction = {pb_reduction_pct:+.2f}%")
    checks.append(_flag("primary-backbone token reduction %", pb_reduction_pct,
                         EXPECTED["primary_backbone_tok_reduction_pct"]))
    result.update(
        primary_backbone_n_baseline=pb_n_base, primary_backbone_n_method=pb_n_meth,
        primary_backbone_n_shared=len(pb_shared),
        primary_backbone_tok_baseline_mean=pb_mean_base,
        primary_backbone_tok_method_mean=pb_mean_meth,
        primary_backbone_tok_reduction_pct=pb_reduction_pct,
    )

    # --- expected-value reconciliation --------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPECTED-VALUE RECONCILIATION")
    print("=" * 78)
    any_mismatch = False
    for chk in checks:
        tag = "MATCH" if chk["matches"] else "MISMATCH"
        if not chk["matches"]:
            any_mismatch = True
        print(f"  [{tag:8s}] {chk['label']:32s} computed={chk['computed']!r:>18}  "
              f"expected={chk['expected']!r:>18}")
    if any_mismatch:
        print("\n  >>> AT LEAST ONE MISMATCH — see flags above; both numbers reported, "
              "computed value takes precedence over the quoted expectation. <<<")
    else:
        print("\n  All computed quantities match their quoted expected values.")
    result["checks"] = checks
    result["any_mismatch"] = any_mismatch

    out_json = Path(__file__).resolve().parent / "xbackbone_full_data.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")

    _write_markdown_report(result, checks)
    return 0


def _write_markdown_report(result: dict, checks: list[dict]) -> None:
    md_path = Path(__file__).resolve().parent / "xbackbone_full.md"
    lines = []
    lines.append("# Cross-backbone full analysis")
    lines.append("")
    lines.append(f"Generated by `analysis/xbackbone_full.py`. Second backbone "
                  f"**{result['second_backbone']}**, dataset **{result['dataset']}**, "
                  f"baseline `{result['baseline_cond']}` vs full method `{result['method_cond']}`, "
                  f"on the SHARED instance set (inner join on instance_id).")
    lines.append("")
    lines.append(f"n_baseline={result['n_baseline']}  n_method={result['n_method']}  "
                  f"**n_shared={result['n_shared']}**")
    lines.append("")
    lines.append("## Quantities")
    lines.append("")
    lines.append("| # | quantity | baseline | method | delta | test | statistic | p-value |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")
    lines.append(f"| 1 | EM% | {result['em_baseline']:.2f} | {result['em_method']:.2f} | "
                  f"{result['delta_em']:+.2f} | exact McNemar (b={result['em_b']}, c={result['em_c']}) "
                  f"| — | {result['em_p']:.4g} |")
    if result.get("judge") == "insufficient coverage":
        lines.append("| 2 | judge% | insufficient coverage | insufficient coverage | — | — | — | — |")
    else:
        lines.append(f"| 2 | judge% | {result['judge_baseline']:.2f} | {result['judge_method']:.2f} | "
                      f"{result['delta_judge']:+.2f} | exact McNemar (b={result['judge_b']}, "
                      f"c={result['judge_c']}) | — | {result['judge_p']:.4g} |")
    lines.append(f"| 3 | tokens/episode (count-once) | {result['tok_baseline_mean']:,.0f} | "
                 f"{result['tok_method_mean']:,.0f} | {result['tok_reduction_pct']:+.2f}% "
                 f"(reduction) | **paired t-test (declared/reported)** | "
                 f"{result['tok_ttest_stat']:.3f} | {result['tok_ttest_p']:.4g} |")
    lines.append(f"|   | tokens/episode (count-once) | {result['tok_baseline_mean']:,.0f} | "
                 f"{result['tok_method_mean']:,.0f} | {result['tok_reduction_pct']:+.2f}% "
                 f"(reduction) | Wilcoxon signed-rank (secondary) | "
                 f"{result['tok_wilcoxon_stat']:.1f} | {result['tok_wilcoxon_p']:.4g} |")
    lines.append(f"| 4 | LLM calls/episode | {result['llm_calls_baseline_mean']:.2f} | "
                 f"{result['llm_calls_method_mean']:.2f} | {result['llm_calls_delta']:+.2f} | "
                 f"**paired t-test (declared/reported)** | {result['llm_calls_ttest_stat']:.3f} | "
                 f"{result['llm_calls_ttest_p']:.4g} |")
    lines.append(f"|   | LLM calls/episode | {result['llm_calls_baseline_mean']:.2f} | "
                 f"{result['llm_calls_method_mean']:.2f} | {result['llm_calls_delta']:+.2f} | "
                 f"Wilcoxon signed-rank (secondary) | {result['llm_calls_wilcoxon_stat']:.1f} | "
                 f"{result['llm_calls_wilcoxon_p']:.4g} |")
    lines.append("")
    lines.append(f"## 5. Primary-backbone contrast (same dataset/condition pair)")
    lines.append("")
    lines.append(f"Primary backbone **{result['primary_backbone']}** (`{result['baseline_cond']}` vs "
                 f"`{result['method_cond']}`, {result['dataset']}, n_shared="
                 f"{result['primary_backbone_n_shared']}), resolved via `scripts/compare_cells.py`'s "
                 f"REGISTRY (never a hardcoded path):")
    lines.append("")
    lines.append(f"- {result['primary_backbone']}: mean tok baseline="
                 f"{result['primary_backbone_tok_baseline_mean']:,.0f}, mean tok method="
                 f"{result['primary_backbone_tok_method_mean']:,.0f}, reduction="
                 f"**{result['primary_backbone_tok_reduction_pct']:+.2f}%**")
    lines.append(f"- {result['second_backbone']}: mean tok baseline="
                 f"{result['tok_baseline_mean']:,.0f}, mean tok method="
                 f"{result['tok_method_mean']:,.0f}, reduction=**{result['tok_reduction_pct']:+.2f}%**")
    lines.append("")
    lines.append(f"-> The paper's **{result['tok_reduction_pct']:.1f}% vs "
                 f"{result['primary_backbone_tok_reduction_pct']:.1f}%** contrast.")
    lines.append("")
    lines.append("## Expected-value reconciliation")
    lines.append("")
    lines.append("Values quoted in the task brief as an independently-computed cross-check, "
                 "compared against (never substituted for) this script's own numbers.")
    lines.append("")
    lines.append("| quantity | computed | expected | match |")
    lines.append("|---|---:|---:|:---:|")
    for chk in checks:
        tag = "yes" if chk["matches"] else "**NO — see note above**"
        lines.append(f"| {chk['label']} | {chk['computed']!r} | {chk['expected']!r} | {tag} |")
    lines.append("")
    if result["any_mismatch"]:
        lines.append("**At least one quantity did not match its quoted expected value — see the "
                     "table above for both numbers.**")
    else:
        lines.append("All computed quantities matched their quoted expected values (see table above "
                     "for the exact computed figures).")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    raise SystemExit(main())
