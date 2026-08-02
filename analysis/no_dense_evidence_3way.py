#!/usr/bin/env python
"""No-dense-evidence ablation (condition `agent_research_snip`) vs FULL METHOD / Sieve
(`agent_research_bql_dense_snip`), now on all THREE datasets.

Context: this single-axis rung ("drops dense fusion (2) alone", see
latex/sections/results.tex Finding 2 / latex/tables/ablation_family.tex) previously only had
`agent_research_snip` cells for browsecomp_plus_structured and hotpotqa_structured --
musique_structured's cell was missing. It has now been merged (n=2409 rows at
runs/_headline_validation/agent/musique_structured/Tongyi-DeepResearch-30B-A3B/
agent_research_snip/rows.jsonl), so this script recomputes the paired comparison on all three
datasets uniformly and adds the missing MuSiQue number.

Reuses (does not reimplement) the EXACT same machinery `analysis/ablation_deltas.py` uses for
every other row in this ablation family, imported directly from that module and from
scripts/compare_cells.py so method parity is structural, not just "the same formula typed
twice":
  - analysis.ablation_deltas.{FULL, cell_metrics, paired, paired_judge, JUDGE_COVERAGE_MIN}
  - scripts.compare_cells.{load_qrels, mcnemar_p}   (mcnemar_p: exact two-sided McNemar via
    scipy.stats.binomtest on the discordant-pair counts, min(b,c) convention)
  - transitively: evaluation.metrics.answer_em (canonical EM), scripts.compare_cells.metrics()
    (per-instance em/judge dict), scripts.force_answer_backfill.load_rows_with_recovery
    (sidecar-safe recovery overlay), scripts.compare_cells.{cell_dir, cell_rows,
    load_judge_cache, pct}.

Shared-set convention (matches ablation_deltas.py exactly): inner join (set intersection) of
instance_ids present in both cells' overlaid metrics dicts -- NOT an assumption that the two
id sets are identical. n_shared is always reported.

Judge gate (matches ablation_deltas.py / compare_cells.py exactly): a judge delta is computed
ONLY if BOTH cells clear >=90% judge coverage (JUDGE_COVERAGE_MIN); otherwise this dataset's
judge column reports "insufficient judge coverage" rather than a delta computed over a
partially-judged cell.

No multiple-comparison correction is applied here -- raw per-comparison exact McNemar p-values
only (Holm-Bonferroni-per-family is applied downstream, at paper-assembly time).

Run: PYTHONPATH=. envs/bin/python analysis/no_dense_evidence_3way.py
Writes:
  - analysis/no_dense_evidence_3way_data.json  (full machine-readable results)
  - appends a new, clearly-marked section to analysis/ablation_deltas.md (the existing content
    of that file is never edited or restated -- this only adds a section at the end).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import load_qrels  # noqa: E402
from analysis.ablation_deltas import (  # noqa: E402
    FULL, cell_metrics, paired, paired_judge, JUDGE_COVERAGE_MIN,
)

LABEL = "No dense evidence (bql+snip fetch)"
COND = "agent_research_snip"
SUBDIR = "_headline_validation"

DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]

MD_PATH = Path(__file__).resolve().parent / "ablation_deltas.md"
JSON_PATH = Path(__file__).resolve().parent / "no_dense_evidence_3way_data.json"

SECTION_MARK = "## NEW: no-dense-evidence ablation, three-dataset update (musique_structured merged)"


def compute() -> dict:
    results = {}
    for dataset in DATASETS:
        entry = {"dataset": dataset, "label": LABEL, "condition": COND, "subdir": SUBDIR}
        qrels = load_qrels(dataset)
        full_m, full_cov = cell_metrics(dataset, FULL[0], FULL[1], qrels)
        other_m, other_cov = cell_metrics(dataset, SUBDIR, COND, qrels)

        if full_m is None:
            entry["error"] = "FULL METHOD cell missing (rows.jsonl absent)"
            results[dataset] = entry
            continue
        if other_m is None:
            entry["error"] = "comparison cell missing (rows.jsonl absent)"
            results[dataset] = entry
            continue

        res = paired(full_m, other_m)
        if res is None:
            entry["error"] = "no shared instances between the two cells"
            results[dataset] = entry
            continue
        n_shared, em_full, em_other, delta, p = res

        # Discordant pair counts, same definition `paired()` uses internally to feed mcnemar_p
        # (b = other-correct/full-wrong, c = full-correct/other-wrong; mcnemar_p is symmetric in
        # b vs c since it always tests min(b, c) against n = b + c).
        mut = set(full_m) & set(other_m)
        b = sum(1 for i in mut if other_m[i]["em"] and not full_m[i]["em"])
        c = sum(1 for i in mut if full_m[i]["em"] and not other_m[i]["em"])

        entry.update(
            n_shared=n_shared,
            em_sieve=em_full,
            em_no_dense=em_other,
            delta=delta,
            b=b,
            c=c,
            mcnemar_p=p,
            full_judge_coverage=full_cov,
            other_judge_coverage=other_cov,
            judge_coverage_min=JUDGE_COVERAGE_MIN,
        )

        jres = paired_judge(full_m, other_m, full_cov, other_cov)
        if jres is not None:
            n_j, j_full, j_other, jdelta, jp = jres
            entry.update(
                judge_status="computed",
                judge_n_shared=n_j,
                judge_sieve=j_full,
                judge_no_dense=j_other,
                judge_delta=jdelta,
                judge_mcnemar_p=jp,
            )
        else:
            entry["judge_status"] = "insufficient judge coverage"

        results[dataset] = entry
    return results


def render_markdown_section(results: dict) -> str:
    lines = [
        "\n" + SECTION_MARK + "\n",
        "Generated by `analysis/no_dense_evidence_3way.py`. Same conventions as the rest of this "
        "file: paired against **FULL METHOD** (`_headline_validation/.../"
        "agent_research_bql_dense_snip`, Sieve) on the instance set the two cells share "
        "(inner join on instance_id, n_shared reported, no assumption the id sets are "
        "identical); exact two-sided McNemar via `scipy.stats.binomtest` on the discordant pair "
        "counts (b, c); judge delta computed only when BOTH cells clear "
        f"{JUDGE_COVERAGE_MIN*100:.0f}% judge coverage, else `insufficient judge coverage`; no "
        "multiple-comparison correction applied here (raw per-comparison p only). "
        "musique_structured's `agent_research_snip` cell was previously missing and has now "
        "been merged (n=2409), completing this rung on all three datasets.\n",
        "| dataset | n_shared | EM(Sieve) | EM(no-dense) | delta | b | c | McNemar p | judge delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        e = results[dataset]
        if "error" in e:
            lines.append(f"| {dataset} | -- | | | | | | | ({e['error']}) |")
            continue
        if e["judge_status"] == "computed":
            jstr = f"{e['judge_delta']:+.1f} (p={e['judge_mcnemar_p']:.3g}, n={e['judge_n_shared']})"
        else:
            jstr = "insufficient judge coverage"
        lines.append(
            f"| {dataset} | {e['n_shared']} | {e['em_sieve']:.1f} | {e['em_no_dense']:.1f} | "
            f"{e['delta']:+.1f} | {e['b']} | {e['c']} | {e['mcnemar_p']:.3g} | {jstr} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    results = compute()

    JSON_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote: {JSON_PATH}")

    section = render_markdown_section(results)
    existing = MD_PATH.read_text()
    if SECTION_MARK in existing:
        raise SystemExit(
            f"{MD_PATH} already contains a '{SECTION_MARK}' section -- refusing to append a "
            "duplicate. Edit/remove the stale section manually if this script is being re-run "
            "after a genuine data update."
        )
    with MD_PATH.open("a") as fh:
        fh.write(section)
    print(f"appended new section to: {MD_PATH}")

    print()
    print(section)


if __name__ == "__main__":
    main()
