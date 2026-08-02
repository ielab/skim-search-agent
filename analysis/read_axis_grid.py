#!/usr/bin/env python
"""The full READ-axis contrast grid on `browsecomp_plus_structured`, n=830 per cell.

Context: a reviewer recomputed, ad hoc, the deltas behind
`latex_acl8/sections/appendix.tex`'s Table~\\ref{tab:factorial-grid} ("The crossed grid") and found
its caption overreaches -- it reports every visit-to-fetch+snippets delta as a flat "gains"/"costs"
without saying which are statistically significant, and it omits the Hybrid RRF engine's row
entirely. This script makes those numbers a reproducible artifact: for each SEARCH engine, it pairs
the whole-document VISIT read interface against the section-FETCH+snippets read interface, holding
the search engine fixed, computes EM delta + exact McNemar p + discordant b/c, judge-accuracy delta
(gated at >=90% coverage on BOTH cells), and mean count-once tokens/episode for both cells.

METHOD PARITY (mandatory, per task spec) -- every number below is computed via the SAME machinery
every other paired comparison in this repo uses, imported directly rather than reimplemented:
  - analysis.ablation_deltas.{cell_metrics, paired, paired_judge, JUDGE_COVERAGE_MIN}
    (cell_metrics -> scripts.compare_cells.{cell_dir, cell_rows, load_judge_cache, metrics, pct};
     metrics() -> evaluation.metrics.answer_em, scripts.force_answer_backfill.load_rows_with_recovery;
     paired()/paired_judge() -> scripts.compare_cells.mcnemar_p, exact two-sided McNemar via
     scipy.stats.binomtest on the discordant-pair counts)
  - scripts.compare_cells.{load_qrels, cell_dir, cell_rows, load_judge_cache}

MANDATORY SANITY GATE (run first, every invocation): reproduce the already-published Sieve
(`_headline_validation/agent_research_bql_dense_snip`) vs "No dense evidence"
(`_headline_validation/agent_research_snip`) comparison on browsecomp_plus_structured. Published:
n=830, EM 38.9 vs 34.2, delta=+4.7, McNemar p=0.0104 (analysis/ablation_deltas.md). If this does not
reproduce, the script STOPS (raises) before computing the grid -- a silent methodology drift must
never produce a new grid.

CELL RESOLUTION (all on browsecomp_plus_structured, n=830 each -- see module-level `PAIRS` for the
exact (subdir, condition) pair resolved for every cell, cross-checked against
`scripts.compare_cells.REGISTRY`'s labels so the mapping is auditable):
  BM25:                     visit _visit_uncapped/agent_research_bm25            fetch _headline_validation/agent_research_bm25_fetch_snip
  Dense:                    visit _fullvisit/agent_research_dense                fetch _headline_validation/agent_research_dense_fetch
  Hybrid sparse-dense RRF:  visit _fullvisit/agent_research_hybrid               fetch _headline_validation/agent_research_hybrid_fetch_snip
  Query surface (no dense): visit _fullvisit/agent_research_bql_visit            fetch _headline_validation/agent_research_snip
  Query surface + dense:    visit _fullvisit/agent_research_bql_dense_visit      fetch _headline_validation/agent_research_bql_dense_snip
  Indri:                    visit _fullvisit/agent_research_indri_visit         fetch _headline_validation/agent_research_indri_snip
  Indri + dense:            visit _fullvisit_dense/agent_research_indri_visit    fetch _dense_validation/agent_research_indri_snip

The last two rows disambiguate a REGISTRY condition-name collision: `agent_research_indri_snip`
appears at THREE different subdirs in REGISTRY (`_headline_validation` -> "indri+snip fetch", no
dense fusion; `_dense_validation` -> "indri+dense+snip fetch", WITH dense fusion via the
INDRI_DENSE env knob attached per-subdir; `_qwen_dense_validation` -> the qwen-embedder variant, out
of scope here). The Indri (no dense) row therefore resolves to `_headline_validation`; the Indri +
dense row resolves to `_dense_validation` (REGISTRY label "indri+dense+snip fetch") to match its
visit sibling's `_fullvisit_dense` (REGISTRY label "indri+dense visit"). Both are unambiguous once
paired against their own visit row's dense/no-dense status.

Run: PYTHONPATH=. envs/bin/python analysis/read_axis_grid.py
Writes:
  - analysis/read_axis_grid_data.json  (full machine-readable results incl. every run dir used)
  - analysis/read_axis_grid.md         (the grid + significance list)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import load_qrels, cell_dir  # noqa: E402
from analysis.ablation_deltas import (  # noqa: E402
    cell_metrics, paired, paired_judge, JUDGE_COVERAGE_MIN,
)

DATASET = "browsecomp_plus_structured"
ALPHA = 0.05

# --- mandatory sanity gate: reproduce a published number before trusting this module -----------
SANITY_FULL = ("_headline_validation", "agent_research_bql_dense_snip")   # Sieve
SANITY_OTHER = ("_headline_validation", "agent_research_snip")            # No dense evidence
SANITY_EXPECTED = dict(n=830, em_full=38.9, em_other=34.2, delta=4.7, p=0.0104)


def run_sanity_gate() -> dict:
    qrels = load_qrels(DATASET)
    full_m, _ = cell_metrics(DATASET, SANITY_FULL[0], SANITY_FULL[1], qrels)
    other_m, _ = cell_metrics(DATASET, SANITY_OTHER[0], SANITY_OTHER[1], qrels)
    if full_m is None or other_m is None:
        raise SystemExit("SANITY GATE FAILED: Sieve or 'No dense evidence' cell missing rows.jsonl "
                          "-- STOPPING before computing the grid (per task spec).")
    res = paired(full_m, other_m)
    if res is None:
        raise SystemExit("SANITY GATE FAILED: no shared instances between Sieve and 'No dense "
                          "evidence' -- STOPPING before computing the grid.")
    n, em_full, em_other, delta, p = res
    passed = (n == SANITY_EXPECTED["n"]
              and abs(em_full - SANITY_EXPECTED["em_full"]) < 0.05
              and abs(em_other - SANITY_EXPECTED["em_other"]) < 0.05
              and abs(delta - SANITY_EXPECTED["delta"]) < 0.05
              and abs(p - SANITY_EXPECTED["p"]) < 0.001)
    result = dict(n=n, em_full=em_full, em_other=em_other, delta=delta, p=p,
                  expected=SANITY_EXPECTED, passed=passed)
    if not passed:
        raise SystemExit(
            "SANITY GATE FAILED: reproduced "
            f"n={n}, em_full={em_full:.2f}, em_other={em_other:.2f}, delta={delta:+.2f}, p={p:.4g} "
            f"but expected {SANITY_EXPECTED} -- STOPPING before computing the grid, per task spec. "
            "Do not trust any grid numbers until this is understood."
        )
    return result


# --- the seven READ-axis pairs, each cell's REGISTRY label carried alongside for audit ----------
# (engine_label, visit_subdir, visit_cond, visit_registry_label, fetch_subdir, fetch_cond, fetch_registry_label)
PAIRS = [
    ("BM25",
     "_visit_uncapped", "agent_research_bm25", "SERP bm25 [BASELINE]",
     "_headline_validation", "agent_research_bm25_fetch_snip", "bm25+snip fetch"),
    ("Dense",
     "_fullvisit", "agent_research_dense", "dense visit",
     "_headline_validation", "agent_research_dense_fetch", "dense+snip fetch"),
    ("Hybrid sparse-dense RRF",
     "_fullvisit", "agent_research_hybrid", "hybrid rrf visit",
     "_headline_validation", "agent_research_hybrid_fetch_snip", "hybrid+snip fetch"),
    ("Query surface (no dense)",
     "_fullvisit", "agent_research_bql_visit", "bql visit",
     "_headline_validation", "agent_research_snip", "bql+snip fetch"),
    ("Query surface + dense",
     "_fullvisit", "agent_research_bql_dense_visit", "bql+dense visit",
     "_headline_validation", "agent_research_bql_dense_snip", "bql+dense+snip fetch"),
    ("Indri",
     "_fullvisit", "agent_research_indri_visit", "indri visit",
     "_headline_validation", "agent_research_indri_snip", "indri+snip fetch"),
    ("Indri + dense",
     "_fullvisit_dense", "agent_research_indri_visit", "indri+dense visit",
     "_dense_validation", "agent_research_indri_snip", "indri+dense+snip fetch"),
]


def mean_tok(m: dict) -> dict:
    """Mean count-once tokens/episode (total, input-half, output-half) over ALL rows in a cell --
    same convention as scripts.compare_cells.py's avg_tok/in_tok/out_tok columns (whole-cell means,
    not restricted to the paired/shared instance subset)."""
    vals = list(m.values())
    n = max(len(vals), 1)
    return dict(tok=sum(v["tok"] for v in vals) / n,
                tok_in=sum(v["tok_in"] for v in vals) / n,
                tok_out=sum(v["tok_out"] for v in vals) / n)


def compute_pair(engine, v_sub, v_cond, v_label, f_sub, f_cond, f_label, qrels: dict) -> dict:
    v_dir = str(cell_dir(v_sub, DATASET, v_cond))
    f_dir = str(cell_dir(f_sub, DATASET, f_cond))
    entry = dict(engine=engine,
                visit=dict(subdir=v_sub, condition=v_cond, registry_label=v_label, run_dir=v_dir),
                fetch=dict(subdir=f_sub, condition=f_cond, registry_label=f_label, run_dir=f_dir))

    visit_m, visit_cov = cell_metrics(DATASET, v_sub, v_cond, qrels)
    fetch_m, fetch_cov = cell_metrics(DATASET, f_sub, f_cond, qrels)
    if visit_m is None or fetch_m is None:
        entry["error"] = "one or both cells missing rows.jsonl"
        return entry

    entry["visit"]["n"] = len(visit_m)
    entry["fetch"]["n"] = len(fetch_m)
    entry["visit"].update(mean_tok(visit_m))
    entry["fetch"].update(mean_tok(fetch_m))
    entry["visit"]["em_all"] = sum(v["em"] for v in visit_m.values()) * 100.0 / max(len(visit_m), 1)
    entry["fetch"]["em_all"] = sum(v["em"] for v in fetch_m.values()) * 100.0 / max(len(fetch_m), 1)

    # delta = fetch minus visit: paired(fetch_m, visit_m) -> (n, em_fetch, em_visit, em_fetch-em_visit, p)
    res = paired(fetch_m, visit_m)
    if res is None:
        entry["error"] = "no shared instances between visit and fetch cells"
        return entry
    n_shared, em_fetch, em_visit, delta, p = res
    mut = set(fetch_m) & set(visit_m)
    b = sum(1 for i in mut if visit_m[i]["em"] and not fetch_m[i]["em"])  # visit-right/fetch-wrong
    c = sum(1 for i in mut if fetch_m[i]["em"] and not visit_m[i]["em"])  # fetch-right/visit-wrong
    entry.update(
        n_shared=n_shared,
        em_visit=em_visit,
        em_fetch=em_fetch,
        delta_em=delta,
        b=b,
        c=c,
        mcnemar_p=p,
        significant=bool(p < ALPHA),
        visit_judge_coverage=visit_cov,
        fetch_judge_coverage=fetch_cov,
        judge_coverage_min=JUDGE_COVERAGE_MIN,
    )

    jres = paired_judge(fetch_m, visit_m, fetch_cov, visit_cov)
    if jres is not None:
        n_j, j_fetch, j_visit, jdelta, jp = jres
        entry.update(judge_status="computed", judge_n_shared=n_j, judge_visit=j_visit,
                     judge_fetch=j_fetch, judge_delta=jdelta, judge_mcnemar_p=jp,
                     judge_significant=bool(jp < ALPHA))
    else:
        entry["judge_status"] = (
            "withheld: below >=90% judge coverage gate "
            f"(visit={ (visit_cov or 0)*100:.1f}%, fetch={(fetch_cov or 0)*100:.1f}%)"
        )

    return entry


def compute() -> dict:
    sanity = run_sanity_gate()
    qrels = load_qrels(DATASET)
    pairs = [compute_pair(*p, qrels) for p in PAIRS]
    return dict(dataset=DATASET, alpha=ALPHA, sanity_gate=sanity, pairs=pairs)


def render_markdown(results: dict) -> str:
    sg = results["sanity_gate"]
    lines = [
        "# READ-axis contrast grid: whole-document VISIT vs section-FETCH+snippets, per SEARCH engine",
        "",
        f"Dataset: `{results['dataset']}`, n=830 per cell. Generated by "
        "`analysis/read_axis_grid.py`; data in `analysis/read_axis_grid_data.json`.",
        "",
        "**Method parity**: every EM/delta/McNemar number below is computed via "
        "`analysis.ablation_deltas.{cell_metrics, paired, paired_judge}`, which are themselves "
        "thin wrappers over `scripts.compare_cells.{cell_dir, cell_rows, load_judge_cache, "
        "metrics, mcnemar_p}`, `evaluation.metrics.answer_em`, and "
        "`scripts.force_answer_backfill.load_rows_with_recovery` -- no EM/recovery/McNemar logic "
        "is reimplemented in this script.",
        "",
        f"**Mandatory sanity gate** (Sieve vs \"No dense evidence\", published +4.7 EM / p=0.0104): "
        f"reproduced n={sg['n']}, EM {sg['em_full']:.1f} vs {sg['em_other']:.1f}, "
        f"delta={sg['delta']:+.2f}, McNemar p={sg['p']:.4g} -- **{'PASS' if sg['passed'] else 'FAIL'}**.",
        "",
        "## The grid",
        "",
        "| SEARCH engine | visit run dir | fetch+snip run dir | n_shared | EM visit | EM fetch | "
        "delta (fetch-visit) | b | c | McNemar p | sig.\\ (α=0.05) | judge visit | judge fetch | "
        "judge delta | judge p | tok/inst visit | tok/inst fetch |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    sig_list, nonsig_list, withheld_judge = [], [], []
    for e in results["pairs"]:
        if "error" in e:
            lines.append(f"| {e['engine']} | {e['visit']['run_dir']} | {e['fetch']['run_dir']} | "
                          f"-- | | | | | | | | | | | | | ({e['error']}) |")
            continue
        sig_mark = "**YES**" if e["significant"] else "no"
        (sig_list if e["significant"] else nonsig_list).append(
            f"{e['engine']}: {e['delta_em']:+.2f} EM (p={e['mcnemar_p']:.3g})")
        if e["judge_status"] == "computed":
            jv, jf = f"{e['judge_visit']:.1f}", f"{e['judge_fetch']:.1f}"
            jd = f"{e['judge_delta']:+.1f}"
            jp = f"{e['judge_mcnemar_p']:.3g}" + (" *" if e["judge_significant"] else "")
        else:
            jv = jf = jd = jp = "withheld (<90% cov.)"
            withheld_judge.append(e["engine"])
        lines.append(
            f"| {e['engine']} | `{e['visit']['run_dir']}` | `{e['fetch']['run_dir']}` | "
            f"{e['n_shared']} | {e['em_visit']:.1f} | {e['em_fetch']:.1f} | {e['delta_em']:+.2f} | "
            f"{e['b']} | {e['c']} | {e['mcnemar_p']:.3g} | {sig_mark} | {jv} | {jf} | {jd} | {jp} | "
            f"{e['visit']['tok']:.0f} | {e['fetch']['tok']:.0f} |"
        )

    lines += [
        "",
        f"n_shared / em_visit / em_fetch above are all computed on the inner-joined "
        "(shared-instance-id) subset of the two cells -- reported as `n_shared`, never assumed "
        "equal to each cell's raw n (which is 830 for every cell here; the inner join is over "
        "instance_id, not just row count).",
        "",
        f"## Significant at alpha={results['alpha']} (McNemar p < {results['alpha']})",
        "",
    ]
    lines += [f"- {s}" for s in sig_list] if sig_list else ["- (none)"]
    lines += ["", f"## NOT significant at alpha={results['alpha']}", ""]
    lines += [f"- {s}" for s in nonsig_list] if nonsig_list else ["- (none)"]
    if withheld_judge:
        lines += ["", "## Judge accuracy withheld (below >=90% coverage gate)", ""]
        lines += [f"- {e}" for e in withheld_judge]
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    results = compute()

    json_path = Path(__file__).resolve().parent / "read_axis_grid_data.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote: {json_path}")

    md_path = Path(__file__).resolve().parent / "read_axis_grid.md"
    text = render_markdown(results)
    md_path.write_text(text)
    print(f"wrote: {md_path}")
    print()
    print(text)


if __name__ == "__main__":
    main()
