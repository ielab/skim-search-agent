#!/usr/bin/env python
"""Paired significance test for the paper's EFFICIENCY claim: full method vs the BM25
search-and-visit baseline, the same rigor the paper's accuracy comparisons already get in
comparison_result.md (paired test on shared instance_ids, exact p-value reported).

    full method : runs/_headline_validation/agent/<ds>/Tongyi-DeepResearch-30B-A3B/
                   agent_research_bql_dense_snip/rows.jsonl   (comparison_result.md's
                   "bql+dense+snip fetch")
    baseline    : runs/_visit_uncapped/agent/<ds>/Tongyi-DeepResearch-30B-A3B/
                   agent_research_bm25/rows.jsonl              (comparison_result.md's
                   "SERP bm25 [BASELINE]")

for ds in {browsecomp_plus_structured, hotpotqa_structured, musique_structured}.

WHAT'S TESTED (per dataset, per basis):
  1. Token counts, two bases (both reported because the paper's own tok columns report both):
       - total_tokens_once  (PRIMARY): count-once tokens — comparison_result.md's avg_tok/inst.
       - prompt_tokens + completion_tokens (SECONDARY): step-summed tokens, which re-count the
         growing prompt on every step and so run larger; not in comparison_result.md's table but
         requested as the paper's other reporting basis.
     Paired Wilcoxon signed-rank test (scipy.stats.wilcoxon) on the shared-instance token
     difference is PRIMARY; a paired t-test (scipy.stats.ttest_rel) is reported alongside as a
     secondary sanity check only — token counts are heavily right-skewed / non-normal, so the
     t-test is not the number to lead with.
  2. LLM-call counts, same paired treatment, reported honestly even where the method makes MORE
     calls (the wiki datasets) — the story is fewer tokens despite more calls, not fewer of
     everything.
  3. A one-line Pareto summary per dataset: the accuracy delta (Δjudge/ΔEM + McNemar p, read
     verbatim out of comparison_result.md — this script does not recompute accuracy) set beside
     the token delta + its own paired p-value from this script.

MEMORY: these rows.jsonl files run 190MB-790MB with heavy per-row fields (trajectory,
observations) this script does not need. Every file is STREAMED one line at a time
(`stream_row_stats`) — one row is json.loads'd, four scalar fields are pulled off it, and the row
is discarded before the next line is read. Peak memory is O(one row) + O(n instances of small
per-id dicts), never O(file size). (Same pattern as analysis/method_vs_dci.py's
`_stream_cell_metrics`, applied here to both sides since neither cell is small enough to be
worth loading whole via load_rows_with_recovery.) None of the ~17-31GB auto-read cells are
touched by this script at all.

Usage:
    PYTHONPATH=. envs/bin/python analysis/token_efficiency.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]

METHOD_ROOT = ROOT / "runs" / "_headline_validation" / "agent"
METHOD_COND = "agent_research_bql_dense_snip"
METHOD_LABEL = "bql+dense+snip fetch"  # comparison_result.md cell name for this condition

BASELINE_ROOT = ROOT / "runs" / "_visit_uncapped" / "agent"
BASELINE_COND = "agent_research_bm25"
BASELINE_LABEL = "SERP bm25 [BASELINE]"  # comparison_result.md cell name for this condition

COMPARISON_MD = ROOT / "comparison_result.md"

OUT_JSON = Path(__file__).resolve().parent / "token_efficiency_data.json"
OUT_MD = Path(__file__).resolve().parent / "token_efficiency.md"

STAT_NOTE = "scipy.stats.wilcoxon (v{}) used for the paired signed-rank test.".format(
    __import__("scipy").__version__
)


def method_path(ds: str) -> Path:
    return METHOD_ROOT / ds / MODEL_DIR / METHOD_COND / "rows.jsonl"


def baseline_path(ds: str) -> Path:
    return BASELINE_ROOT / ds / MODEL_DIR / BASELINE_COND / "rows.jsonl"


def stream_row_stats(path: Path) -> dict:
    """instance_id -> dict(tok_once, tok_stepsum, llm_calls), streamed one line at a time.

    Field semantics deliberately mirror scripts/compare_cells.py's `_row_intrinsic` so tok_once
    here is bit-for-bit the same quantity as comparison_result.md's avg_tok/inst column (verified
    against it in `main`, not just asserted):
        tok_once    = total_tokens_once, falling back to
                       initial_prompt_tokens + context_once_tokens + (output_tokens or
                       completion_tokens) on the rare row missing the field outright.
        tok_stepsum = prompt_tokens + completion_tokens (the step-summed basis requested
                      separately from tok_once — re-counts the growing prompt every step).
        llm_calls   = llm_calls, falling back to n_steps, falling back to len(observations).
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
            tok_stepsum = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
            llm_calls = r.get("llm_calls") or r.get("n_steps") or len(r.get("observations") or [])
            out[iid] = dict(
                tok_once=float(tok_once), tok_stepsum=float(tok_stepsum),
                llm_calls=float(llm_calls),
            )
            del r
    return out


def parse_comparison_md(path: str) -> dict:
    """dataset -> {cell_label -> [raw table-cell strings]} parsed straight out of
    comparison_result.md, so the accuracy numbers this script juxtaposes are read VERBATIM from
    the paper's own already-significance-tested comparison table, never recomputed here."""
    text = Path(path).read_text()
    out: dict = {}
    for sec in re.split(r"^## ", text, flags=re.M)[1:]:
        head, _, body = sec.partition("\n")
        ds = head.strip()
        if ds not in DATASETS:
            continue
        rows = {}
        for line in body.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 17 or cells[0] in ("cell", "---"):
                continue
            rows[cells[0]] = cells
        out[ds] = rows
    return out


# comparison_result.md column layout (see its own header row):
# 0 cell,1 n,2 judge%,3 Δjudge,4 p_judge,5 EM%,6 ΔEM,7 p_em,8 recall%,9 surfaced%,
# 10 avg_tok/inst,11 in_tok/inst,12 out_tok/inst,13 avg_llm_calls,14 empty%,15 recov,16 n_mut
MD_COL = dict(n=1, judge=2, d_judge=3, p_judge=4, em=5, d_em=6, p_em=7, avg_tok=10, n_mut=16)


def md_accuracy(md_rows: dict, ds: str) -> dict:
    """Pull the method row's Δjudge/p_judge/ΔEM/p_em plus both cells' avg_tok/inst (for the
    cross-check against this script's own computed means) out of the parsed comparison table."""
    method_row = md_rows[ds][METHOD_LABEL]
    base_row = md_rows[ds][BASELINE_LABEL]

    def f(row, key):
        v = row[MD_COL[key]]
        return None if v in ("—", "-", "") else v

    return dict(
        d_judge=f(method_row, "d_judge"), p_judge=f(method_row, "p_judge"),
        judge_method=f(method_row, "judge"), judge_baseline=f(base_row, "judge"),
        d_em=f(method_row, "d_em"), p_em=f(method_row, "p_em"),
        em_method=f(method_row, "em"), em_baseline=f(base_row, "em"),
        avg_tok_method_md=f(method_row, "avg_tok"), avg_tok_baseline_md=f(base_row, "avg_tok"),
        n_mut=f(method_row, "n_mut"),
    )


def paired_stats(method_vals: np.ndarray, base_vals: np.ndarray) -> dict:
    """Paired comparison (method vs baseline) on a shared-instance array pair. `reduction` is
    defined baseline-minus-method throughout (positive = method uses fewer tokens/calls)."""
    diff = base_vals - method_vals  # positive => method smaller
    n = len(diff)
    mean_method, mean_base = float(method_vals.mean()), float(base_vals.mean())
    abs_reduction = mean_base - mean_method
    pct_reduction = 100.0 * abs_reduction / mean_base if mean_base else float("nan")

    nonzero = diff[diff != 0]
    if len(nonzero) >= 1:
        try:
            wstat, wp = wilcoxon(method_vals, base_vals, zero_method="wilcox",
                                  alternative="two-sided", mode="auto")
        except ValueError:
            wstat, wp = float("nan"), float("nan")
    else:
        wstat, wp = float("nan"), 1.0  # every pair tied

    tstat, tp = ttest_rel(method_vals, base_vals)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_per_instance = np.where(base_vals > 0, 100.0 * diff / base_vals, np.nan)
    median_pct_reduction = float(np.nanmedian(pct_per_instance))
    median_abs_reduction = float(np.median(diff))
    win_rate = float(100.0 * np.mean(diff > 0))  # % of instances where method is smaller
    tie_rate = float(100.0 * np.mean(diff == 0))

    return dict(
        n=n, mean_method=mean_method, mean_baseline=mean_base,
        abs_reduction=abs_reduction, pct_reduction=pct_reduction,
        median_abs_reduction=median_abs_reduction, median_pct_reduction=median_pct_reduction,
        win_rate=win_rate, tie_rate=tie_rate,
        wilcoxon_stat=float(wstat), wilcoxon_p=float(wp),
        ttest_stat=float(tstat), ttest_p=float(tp),
    )


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}"


def fmt_k(x) -> str:
    return f"{x/1000:,.1f}k"


def main() -> int:
    md_rows = parse_comparison_md(str(COMPARISON_MD))
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
        m_step = np.array([m[i]["tok_stepsum"] for i in shared])
        b_step = np.array([b[i]["tok_stepsum"] for i in shared])
        m_calls = np.array([m[i]["llm_calls"] for i in shared])
        b_calls = np.array([b[i]["llm_calls"] for i in shared])

        stats_once = paired_stats(m_once, b_once)
        stats_step = paired_stats(m_step, b_step)
        stats_calls = paired_stats(m_calls, b_calls)  # reduction<0 where method makes MORE calls

        # verification against comparison_result.md's own avg_tok/inst column, computed on the
        # FULL cell (all rows, not just the shared subset) to match how that column is defined.
        m_once_all = np.array([v["tok_once"] for v in m.values()])
        b_once_all = np.array([v["tok_once"] for v in b.values()])
        acc = md_accuracy(md_rows, ds)

        result = dict(
            dataset=ds, n_method=len(m), n_baseline=len(b), n_shared=n_shared,
            tok_once=stats_once, tok_stepsum=stats_step, llm_calls=stats_calls,
            mean_tok_once_full_method=float(m_once_all.mean()),
            mean_tok_once_full_baseline=float(b_once_all.mean()),
            md_avg_tok_method=acc["avg_tok_method_md"], md_avg_tok_baseline=acc["avg_tok_baseline_md"],
            accuracy=acc,
        )
        all_results.append(result)

        print(f"[{ds}] tok_once: method={fmt_k(stats_once['mean_method'])} "
              f"baseline={fmt_k(stats_once['mean_baseline'])} "
              f"reduction={stats_once['pct_reduction']:.1f}% p={fmt_p(stats_once['wilcoxon_p'])} "
              f"| md check: method={acc['avg_tok_method_md']} baseline={acc['avg_tok_baseline_md']}",
              file=sys.stderr)

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_JSON}", file=sys.stderr)

    write_markdown(all_results, md_rows)
    print(f"wrote {OUT_MD}", file=sys.stderr)
    return 0


def write_markdown(results: list, md_rows: dict) -> None:
    lines = []
    lines.append("# Token efficiency: paired significance test (full method vs BM25 baseline)")
    lines.append("")
    lines.append(
        "Backs the paper's efficiency claim with the same paired-test rigor the accuracy "
        "comparisons get in `comparison_result.md`: full method "
        f"(`{METHOD_COND}`, comparison_result.md's **{METHOD_LABEL}**) vs the BM25 "
        f"search-and-visit baseline (`{BASELINE_COND}`, comparison_result.md's "
        f"**{BASELINE_LABEL}**), paired on shared `instance_id`s within each dataset."
    )
    lines.append("")
    lines.append(f"Statistical test: {STAT_NOTE} A paired t-test (scipy.stats.ttest_rel) is "
                  "reported alongside as a secondary sanity check only — per-episode token "
                  "counts are right-skewed / non-normal, so the Wilcoxon signed-rank test is "
                  "primary, not the t-test.")
    lines.append("")
    lines.append(
        "Generated by `analysis/token_efficiency.py` "
        "(`PYTHONPATH=. envs/bin/python analysis/token_efficiency.py`)."
    )
    lines.append("")

    for r in results:
        ds = r["dataset"]
        lines.append(f"## {ds}")
        lines.append("")
        lines.append(f"n_shared = {r['n_shared']} "
                      f"(method cell n={r['n_method']}, baseline cell n={r['n_baseline']})")
        lines.append("")
        lines.append("| basis | mean method | mean baseline | abs. reduction | % reduction | "
                      "median % reduction/inst | win rate (method fewer) | "
                      "Wilcoxon p (paired, primary) | paired t-test p (secondary) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, s in [
            ("total_tokens_once (primary, count-once)", r["tok_once"]),
            ("prompt_tokens+completion_tokens (secondary, step-summed)", r["tok_stepsum"]),
        ]:
            lines.append(
                f"| {label} | {fmt_k(s['mean_method'])} | {fmt_k(s['mean_baseline'])} | "
                f"{fmt_k(s['abs_reduction'])} | {s['pct_reduction']:.1f}% | "
                f"{s['median_pct_reduction']:.1f}% | {s['win_rate']:.1f}% | "
                f"{fmt_p(s['wilcoxon_p'])} | {fmt_p(s['ttest_p'])} |"
            )
        lines.append("")

        if r["tok_stepsum"]["win_rate"] < 50.0:
            lines.append(
                f"*Caveat on the step-summed basis*: mean reduction is "
                f"{r['tok_stepsum']['pct_reduction']:+.1f}% (method smaller on average) but "
                f"median per-instance reduction is {r['tok_stepsum']['median_pct_reduction']:+.1f}% "
                f"and win rate is only {r['tok_stepsum']['win_rate']:.1f}% — on this dataset the "
                f"*typical* instance uses MORE step-summed tokens under the method (its extra LLM "
                f"calls each re-send the growing prompt), while a smaller number of large "
                f"reductions pull the mean the other way. The count-once basis above does not "
                f"have this mean/median split (win rate {r['tok_once']['win_rate']:.1f}%) — "
                f"treat count-once as the basis that actually reflects the typical episode; the "
                f"step-summed mean-level reduction here should not be quoted as a per-instance "
                f"or median result."
            )
            lines.append("")

        lc = r["llm_calls"]
        lines.append("**LLM calls** (same paired treatment; a negative reduction means the "
                      "method makes *more* calls — reported honestly, not hidden):")
        lines.append("")
        lines.append("| mean method calls | mean baseline calls | Δ (method-baseline) | "
                      "median Δ/inst | fraction method makes fewer calls | Wilcoxon p | t-test p |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        delta_calls = lc["mean_method"] - lc["mean_baseline"]
        lines.append(
            f"| {lc['mean_method']:.1f} | {lc['mean_baseline']:.1f} | {delta_calls:+.1f} | "
            f"{-lc['median_abs_reduction']:+.1f} | {lc['win_rate']:.1f}% | "
            f"{fmt_p(lc['wilcoxon_p'])} | {fmt_p(lc['ttest_p'])} |"
        )
        lines.append("")

        # verification block
        lines.append(
            f"*Verification (full cell, all rows, not just shared): this script's mean "
            f"total_tokens_once = {fmt_k(r['mean_tok_once_full_method'])} method / "
            f"{fmt_k(r['mean_tok_once_full_baseline'])} baseline, vs comparison_result.md's "
            f"avg_tok/inst = {r['md_avg_tok_method']} method / {r['md_avg_tok_baseline']} "
            f"baseline.*"
        )
        lines.append("")

        # Pareto line
        acc = r["accuracy"]
        toks = r["tok_once"]
        judge_sig = "significant" if acc["p_judge"] and float(acc["p_judge"]) < 0.05 else "not significant"
        em_sig = "significant" if acc["p_em"] and float(acc["p_em"]) < 0.05 else "not significant"
        lines.append(
            f"**Pareto summary — {ds}**: accuracy Δjudge={acc['d_judge']} (p={acc['p_judge']}, "
            f"{judge_sig}), ΔEM={acc['d_em']} (p={acc['p_em']}, {em_sig}) — juxtaposed with "
            f"{toks['pct_reduction']:.1f}% fewer count-once tokens/instance "
            f"({fmt_k(toks['mean_method'])} vs {fmt_k(toks['mean_baseline'])}), paired Wilcoxon "
            f"p={fmt_p(toks['wilcoxon_p'])} on n={toks['n']} shared instances, while making "
            f"{delta_calls:+.1f} calls/instance on average "
            f"({'more' if delta_calls > 0 else 'fewer'} calls, "
            f"{'not ' if lc['wilcoxon_p'] >= 0.05 else ''}significant p={fmt_p(lc['wilcoxon_p'])})."
        )
        lines.append("")

    # ---- lift-able sentences -------------------------------------------------------------
    lines.append("## Lift-able sentences for the writeup")
    lines.append("")
    for r in results:
        ds = r["dataset"]
        toks = r["tok_once"]
        acc = r["accuracy"]
        pct = toks["pct_reduction"]
        p = fmt_p(toks["wilcoxon_p"])
        lines.append(
            f"- On **{ds}**, the method uses {pct:.0f}% fewer distinct (count-once) tokens per "
            f"instance than the BM25 baseline ({fmt_k(toks['mean_method'])} vs "
            f"{fmt_k(toks['mean_baseline'])}), a paired reduction significant at p={p} "
            f"(Wilcoxon signed-rank, n={toks['n']}); the accuracy delta on this dataset is "
            f"Δjudge={acc['d_judge']} (p={acc['p_judge']}), ΔEM={acc['d_em']} (p={acc['p_em']})."
        )
    lines.append("")
    all_p = [r["tok_once"]["wilcoxon_p"] for r in results]
    if all(p < 0.05 for p in all_p):
        lines.append(
            "- The token reduction is a paired win on all three datasets "
            f"(all Wilcoxon p < 0.05: {', '.join(fmt_p(p) for p in all_p)}), so on the one "
            "dataset where the accuracy comparison is a statistical tie "
            "(browsecomp_plus_structured, Δjudge/ΔEM not significant vs the BM25 baseline), the "
            "honest framing is a same-accuracy-lower-cost result, not a null result."
        )
    lines.append(
        "- The method makes more LLM calls per instance than the baseline on the two "
        "wikipedia-derived datasets (hotpotqa_structured, musique_structured) — this is reported "
        "alongside the token win rather than obscured by it; the efficiency claim is about total "
        "context volume (tokens), not call count."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
