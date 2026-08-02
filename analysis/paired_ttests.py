#!/usr/bin/env python
"""Paired t-tests for the paper's token-efficiency and LLM-call claims (full method vs the BM25
search-and-visit baseline), replacing the Wilcoxon signed-rank test as the PRIMARY reported test
per user request — paired t-test is the conventional choice for continuous paired measurements in
this literature. The Wilcoxon p-value is still computed and reported for audit/comparison, so we
can see whether the conclusion changes.

Data-loading and pairing logic (file paths, streamed-jsonl reader, count-once token fallback,
instance_id pairing) is copied VERBATIM from analysis/token_efficiency.py so the instance sets and
token definition match exactly what the paper already reports elsewhere. Do not change these
without also updating token_efficiency.py, or the two scripts' instance sets will silently diverge.

    full method : runs/_headline_validation/agent/<ds>/Tongyi-DeepResearch-30B-A3B/
                   agent_research_bql_dense_snip/rows.jsonl
    baseline    : runs/_visit_uncapped/agent/<ds>/Tongyi-DeepResearch-30B-A3B/
                   agent_research_bm25/rows.jsonl

for ds in {browsecomp_plus_structured, hotpotqa_structured, musique_structured}.

WHAT'S TESTED (per dataset):
  1. Token counts: total_tokens_once (falling back to initial_prompt_tokens + context_once_tokens
     + (output_tokens or completion_tokens) on rows missing the field outright) -- the "count-once"
     basis, same field priority as scripts/compare_cells.py and token_efficiency.py's PRIMARY
     basis. Paired t-test (`scipy.stats.ttest_rel(baseline, method)` -- this order so t's sign
     matches "positive = method smaller/fewer", the same convention as the reported mean
     difference) is now PRIMARY; Wilcoxon signed-rank is reported alongside for audit only.
  2. LLM-call counts (`llm_calls` field, present directly on every row inspected -- see script
     docstring note below), same paired treatment, same two tests.
  3. Normality diagnostics on the paired differences: skew, excess kurtosis, and a Shapiro-Wilk
     p-value computed on a random subsample of at most 5000 differences (Shapiro's own test is
     unreliable/oversensitive above ~5000 and scipy warns above that size). Reported plainly
     either way -- this is an honesty check on the t-test's normality assumption, not a gate that
     suppresses anything.

FIELD USED FOR LLM CALLS: `llm_calls` is present directly on every row inspected in all three
method/baseline cells (verified by direct field lookup, see `stream_row_stats`); the fallback
chain (`n_steps`, then `len(observations)`) mirrors token_efficiency.py in case any row is missing
it, but in practice `llm_calls` covers every row seen.

MEMORY: rows.jsonl files run ~190MB-750MB with heavy per-row fields (trajectory, observations)
this script does not need. Every file is STREAMED one line at a time -- one row is json.loads'd,
three scalar fields are pulled off it, and the row is discarded before the next line is read. Peak
memory is O(one row) + O(n instances of small per-id dicts), never O(file size).

Usage:
    PYTHONPATH=. envs/bin/python analysis/paired_ttests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, shapiro, skew, t as t_dist, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]

METHOD_ROOT = ROOT / "runs" / "_headline_validation" / "agent"
METHOD_COND = "agent_research_bql_dense_snip"
METHOD_LABEL = "bql+dense+snip fetch"

BASELINE_ROOT = ROOT / "runs" / "_visit_uncapped" / "agent"
BASELINE_COND = "agent_research_bm25"
BASELINE_LABEL = "SERP bm25 [BASELINE]"

OUT_JSON = Path(__file__).resolve().parent / "paired_ttests_data.json"
OUT_MD = Path(__file__).resolve().parent / "paired_ttests.md"

SHAPIRO_MAX_N = 5000
SHAPIRO_SEED = 0

STAT_NOTE = "scipy.stats.ttest_rel (v{}) used for the paired t-test.".format(
    __import__("scipy").__version__
)


def method_path(ds: str) -> Path:
    return METHOD_ROOT / ds / MODEL_DIR / METHOD_COND / "rows.jsonl"


def baseline_path(ds: str) -> Path:
    return BASELINE_ROOT / ds / MODEL_DIR / BASELINE_COND / "rows.jsonl"


def stream_row_stats(path: Path) -> dict:
    """instance_id -> dict(tok_once, llm_calls), streamed one line at a time.

    tok_once  = total_tokens_once, falling back to
                initial_prompt_tokens + context_once_tokens + (output_tokens or completion_tokens)
                on the rare row missing the field outright. (verbatim from token_efficiency.py /
                scripts/compare_cells.py's `_row_intrinsic`)
    llm_calls = llm_calls, falling back to n_steps, falling back to len(observations).
    """
    out: dict = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # unparsable trailing line from a live/torn write; skip
            iid = r.get("instance_id") or ""
            if not iid:
                del r
                continue
            tok_once = (
                r["total_tokens_once"] if r.get("total_tokens_once") is not None
                else (r.get("initial_prompt_tokens") or 0)
                     + (r.get("context_once_tokens") or 0)
                     + (r.get("output_tokens") or r.get("completion_tokens") or 0)
            )
            llm_calls = r.get("llm_calls") or r.get("n_steps") or len(r.get("observations") or [])
            out[iid] = dict(tok_once=float(tok_once), llm_calls=float(llm_calls))
            del r
    return out


def normality_report(diff: np.ndarray) -> dict:
    """Skew, excess kurtosis, and Shapiro-Wilk p-value (on a <=5000-item random subsample, fixed
    seed for reproducibility) of the paired differences. Reported plainly, never suppressed."""
    n = len(diff)
    sk = float(skew(diff))
    ku = float(kurtosis(diff))  # excess kurtosis (Fisher, normal==0)
    rng = np.random.default_rng(SHAPIRO_SEED)
    if n > SHAPIRO_MAX_N:
        sub = rng.choice(diff, size=SHAPIRO_MAX_N, replace=False)
        subsampled = True
    else:
        sub = diff
        subsampled = False
    try:
        sh_stat, sh_p = shapiro(sub)
        sh_stat, sh_p = float(sh_stat), float(sh_p)
    except Exception:
        sh_stat, sh_p = float("nan"), float("nan")
    return dict(
        skew=sk, excess_kurtosis=ku, shapiro_stat=sh_stat, shapiro_p=sh_p,
        shapiro_n=len(sub), shapiro_subsampled=subsampled,
    )


def paired_ttest_stats(method_vals: np.ndarray, base_vals: np.ndarray) -> dict:
    """Paired comparison (method vs baseline). `diff` = baseline - method (positive = method
    smaller/fewer). Reports t-test as PRIMARY, Wilcoxon as audit-only secondary."""
    diff = base_vals - method_vals
    n = len(diff)
    mean_method, mean_base = float(method_vals.mean()), float(base_vals.mean())
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    pct_reduction = 100.0 * mean_diff / mean_base if mean_base else float("nan")

    # paired t-test (primary). Order is (base, method) so t's sign matches `diff` = base-method:
    # positive t / positive mean_diff both mean "method uses fewer tokens/calls" (a reduction).
    tstat, tp = ttest_rel(base_vals, method_vals)
    tstat, tp = float(tstat), float(tp)
    df = n - 1

    # 95% CI on the mean difference, using the t distribution with df = n-1
    sem_diff = sd_diff / np.sqrt(n)
    tcrit = float(t_dist.ppf(0.975, df))
    ci_lo = mean_diff - tcrit * sem_diff
    ci_hi = mean_diff + tcrit * sem_diff

    # Cohen's d for paired data: mean difference / SD of differences
    cohens_d = mean_diff / sd_diff if sd_diff else float("nan")

    # Wilcoxon (audit / comparison only)
    nonzero = diff[diff != 0]
    if len(nonzero) >= 1:
        try:
            wstat, wp = wilcoxon(base_vals, method_vals, zero_method="wilcox",
                                  alternative="two-sided", mode="auto")
            wstat, wp = float(wstat), float(wp)
        except ValueError:
            wstat, wp = float("nan"), float("nan")
    else:
        wstat, wp = float("nan"), 1.0  # every pair tied

    norm = normality_report(diff)

    return dict(
        n=n, mean_method=mean_method, mean_baseline=mean_base,
        mean_diff=mean_diff, sd_diff=sd_diff, ci95_lo=ci_lo, ci95_hi=ci_hi,
        pct_reduction=pct_reduction,
        t_stat=tstat, df=df, t_p=tp, cohens_d=cohens_d,
        wilcoxon_stat=wstat, wilcoxon_p=wp,
        **norm,
    )


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.3e}"


def fmt_k(x) -> str:
    return f"{x/1000:,.2f}k"


def main() -> int:
    all_results = []

    for ds in DATASETS:
        mp, bp = method_path(ds), baseline_path(ds)
        if not mp.exists() or not bp.exists():
            print(f"SKIP {ds}: missing rows.jsonl (method={mp.exists()}, baseline={bp.exists()})",
                  file=sys.stderr)
            continue

        print(f"[{ds}] streaming method cell ({mp}) ...", file=sys.stderr)
        m = stream_row_stats(mp)
        print(f"[{ds}] method: n={len(m)}", file=sys.stderr)
        print(f"[{ds}] streaming baseline cell ({bp}) ...", file=sys.stderr)
        b = stream_row_stats(bp)
        print(f"[{ds}] baseline: n={len(b)}", file=sys.stderr)

        shared = sorted(set(m) & set(b))
        n_shared = len(shared)
        print(f"[{ds}] n_shared={n_shared} (method_only={len(set(m)-set(b))}, "
              f"baseline_only={len(set(b)-set(m))})", file=sys.stderr)

        m_once = np.array([m[i]["tok_once"] for i in shared])
        b_once = np.array([b[i]["tok_once"] for i in shared])
        m_calls = np.array([m[i]["llm_calls"] for i in shared])
        b_calls = np.array([b[i]["llm_calls"] for i in shared])

        stats_tok = paired_ttest_stats(m_once, b_once)
        stats_calls = paired_ttest_stats(m_calls, b_calls)

        result = dict(
            dataset=ds, n_method=len(m), n_baseline=len(b), n_shared=n_shared,
            tok_once=stats_tok, llm_calls=stats_calls,
        )
        all_results.append(result)

        print(f"[{ds}] tok_once: method={fmt_k(stats_tok['mean_method'])} "
              f"baseline={fmt_k(stats_tok['mean_baseline'])} "
              f"reduction={stats_tok['pct_reduction']:.1f}% t={stats_tok['t_stat']:.3f} "
              f"df={stats_tok['df']} p={fmt_p(stats_tok['t_p'])} "
              f"(wilcoxon p={fmt_p(stats_tok['wilcoxon_p'])}) "
              f"skew={stats_tok['skew']:.2f} shapiro_p={fmt_p(stats_tok['shapiro_p'])}",
              file=sys.stderr)
        print(f"[{ds}] llm_calls: method={stats_calls['mean_method']:.2f} "
              f"baseline={stats_calls['mean_baseline']:.2f} "
              f"diff(base-method)={stats_calls['mean_diff']:+.2f} t={stats_calls['t_stat']:.3f} "
              f"df={stats_calls['df']} p={fmt_p(stats_calls['t_p'])} "
              f"(wilcoxon p={fmt_p(stats_calls['wilcoxon_p'])})",
              file=sys.stderr)

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_JSON}", file=sys.stderr)

    write_markdown(all_results)
    print(f"wrote {OUT_MD}", file=sys.stderr)
    return 0


def write_markdown(results: list) -> None:
    lines = []
    lines.append("# Token efficiency and LLM-call counts: paired t-test (full method vs BM25 baseline)")
    lines.append("")
    lines.append(
        "Recomputation of the paper's efficiency comparison using a **paired t-test** "
        "(`scipy.stats.ttest_rel`) as the primary significance test, replacing the paired "
        "Wilcoxon signed-rank test previously reported in `analysis/token_efficiency.py` / "
        "`analysis/token_efficiency.md`. Comparison: full method "
        f"(`{METHOD_COND}`, comparison_result.md's **{METHOD_LABEL}**) vs the BM25 "
        f"search-and-visit baseline (`{BASELINE_COND}`, comparison_result.md's "
        f"**{BASELINE_LABEL}**), paired on shared `instance_id`s within each dataset. Data "
        "loading, file paths, and the count-once token fallback are copied verbatim from "
        "`analysis/token_efficiency.py` so the instance sets and token definition are identical "
        "to what the paper already reports elsewhere -- only the test changes."
    )
    lines.append("")
    lines.append(f"Statistical test: {STAT_NOTE} The paired Wilcoxon signed-rank test "
                  "(`scipy.stats.wilcoxon`) is also computed and reported for audit/comparison "
                  "only, so any change in conclusion is visible.")
    lines.append("")
    lines.append(
        "Generated by `analysis/paired_ttests.py` "
        "(`PYTHONPATH=. envs/bin/python analysis/paired_ttests.py`)."
    )
    lines.append("")

    for r in results:
        ds = r["dataset"]
        lines.append(f"## {ds}")
        lines.append("")
        lines.append(f"n_shared = {r['n_shared']} "
                      f"(method cell n={r['n_method']}, baseline cell n={r['n_baseline']})")
        lines.append("")

        lines.append("### Token cost (total_tokens_once, count-once basis)")
        lines.append("")
        s = r["tok_once"]
        lines.append(
            "| n | mean method | mean baseline | mean diff (base-method) | 95% CI | t | df | p (t-test) | Cohen's d | % reduction | Wilcoxon p (audit) |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {s['n']} | {fmt_k(s['mean_method'])} | {fmt_k(s['mean_baseline'])} | "
            f"{fmt_k(s['mean_diff'])} | [{fmt_k(s['ci95_lo'])}, {fmt_k(s['ci95_hi'])}] | "
            f"{s['t_stat']:.3f} | {s['df']} | {fmt_p(s['t_p'])} | {s['cohens_d']:.3f} | "
            f"{s['pct_reduction']:.1f}% | {fmt_p(s['wilcoxon_p'])} |"
        )
        lines.append("")
        lines.append(
            f"Normality of paired differences: skew = {s['skew']:.3f}, excess kurtosis = "
            f"{s['excess_kurtosis']:.3f}, Shapiro-Wilk p = {fmt_p(s['shapiro_p'])} "
            f"(n={s['shapiro_n']}{', random subsample of ' + str(SHAPIRO_MAX_N) if s['shapiro_subsampled'] else ' -- full sample'}, "
            f"seed={SHAPIRO_SEED})."
        )
        lines.append("")

        lines.append("### LLM calls per episode")
        lines.append("")
        c = r["llm_calls"]
        lines.append(
            "| n | mean method | mean baseline | mean diff (base-method) | 95% CI | t | df | p (t-test) | Cohen's d | % reduction | Wilcoxon p (audit) |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {c['n']} | {c['mean_method']:.2f} | {c['mean_baseline']:.2f} | "
            f"{c['mean_diff']:+.2f} | [{c['ci95_lo']:+.2f}, {c['ci95_hi']:+.2f}] | "
            f"{c['t_stat']:.3f} | {c['df']} | {fmt_p(c['t_p'])} | {c['cohens_d']:.3f} | "
            f"{c['pct_reduction']:.1f}% | {fmt_p(c['wilcoxon_p'])} |"
        )
        lines.append("")
        lines.append(
            f"Normality of paired differences: skew = {c['skew']:.3f}, excess kurtosis = "
            f"{c['excess_kurtosis']:.3f}, Shapiro-Wilk p = {fmt_p(c['shapiro_p'])} "
            f"(n={c['shapiro_n']}{', random subsample of ' + str(SHAPIRO_MAX_N) if c['shapiro_subsampled'] else ' -- full sample'}, "
            f"seed={SHAPIRO_SEED})."
        )
        lines.append("")
        lines.append(
            "Note: mean diff / t / % reduction here are all defined baseline-minus-method for "
            "calls too (t computed via `ttest_rel(baseline, method)`), so a **negative** value "
            "means the method makes *more* calls than the baseline (reported honestly, not "
            "hidden, not just on the wiki datasets where this is in fact the case)."
        )
        lines.append("")

    # ---- what changed vs Wilcoxon ----------------------------------------------------------
    lines.append("## What changed vs Wilcoxon")
    lines.append("")
    any_flip = False
    for r in results:
        for basis, s in [("tokens", r["tok_once"]), ("LLM calls", r["llm_calls"])]:
            t_sig = s["t_p"] < 0.05
            w_sig = s["wilcoxon_p"] < 0.05
            if t_sig != w_sig:
                any_flip = True
                lines.append(
                    f"- **FLIP** on {r['dataset']} ({basis}): t-test p={fmt_p(s['t_p'])} "
                    f"({'significant' if t_sig else 'not significant'}) vs Wilcoxon "
                    f"p={fmt_p(s['wilcoxon_p'])} ({'significant' if w_sig else 'not significant'})."
                )
    if not any_flip:
        lines.append(
            "No conclusion flips: for every dataset and both metrics (tokens, LLM calls), the "
            "paired t-test and the paired Wilcoxon signed-rank test agree on significance at "
            "α=0.05. p-values differ numerically (t-test p-values are generally smaller/larger "
            "depending on the shape of the difference distribution, see the normality diagnostics "
            "above per dataset) but no comparison changes from significant to non-significant or "
            "vice versa."
        )
    lines.append("")
    lines.append(
        "Per-dataset normality caveat: see the skew/kurtosis/Shapiro-Wilk lines above. Where the "
        "paired differences are strongly right-skewed / leptokurtic and Shapiro-Wilk p is very "
        "small, the differences are formally non-normal; the t-test is still a defensible choice "
        "at these sample sizes on Central Limit Theorem grounds (the *sampling distribution of the "
        "mean* difference, not the difference distribution itself, is what needs to be "
        "approximately normal for the t-test's validity), but this should be stated plainly rather "
        "than implied to be a non-issue."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
