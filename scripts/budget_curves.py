#!/usr/bin/env python3
"""Post-hoc "direction vs persistence" step-budget curves — NO new runs.

Every row in runs/agent/<dataset>/<model>/<condition>/rows.jsonl records the FULL
trajectory the episode actually ran, plus n_steps (the step it answered/stopped at).
An episode that answered at step s would have produced the exact same answer under
any larger budget b >= s (the trajectory up to s doesn't change). Under a smaller
budget b < s, the episode never got to answer, so it counts as wrong. That gives:

    EM@b = fraction of rows with (n_steps <= b) AND (recomputed answer is correct)

This is a purely retrospective slice of the SAME frozen rows -- no retrieval or
generation is re-run. CAVEAT (stated again in the generated doc): this slightly
UNDERSTATES small-b accuracy for episodes that hit the run's OWN step cap (e.g.
max_steps=100 for the main runs, max_steps=50 for the budget50-ablation runs) and
got force-answered at b_max with an empty/low-quality answer -- under a smaller
"what-if" budget b_max' > b_max the model might have kept going and eventually
answered correctly, but we have no trajectory steps past the cap to know that, so
those episodes are (correctly, per the definition above, but pessimistically vs.
a hypothetical larger cap) scored wrong at every b.

Answer extraction/scoring is the FIXED pipeline (extract_answer_span pulls the
LAST-opened <answer>...</answer> span, not the first, which avoids the ~37%
mis-extraction bug from a naive non-greedy findall): prediction is derived from the
last trajectory step's raw_output (falling back to the row's final_answer if the
trajectory is empty), then scored with answer_em against gold_answer.

Usage:
    envs/bin/python scripts/budget_curves.py

Reads only; writes docs/budget_curves.md. Streams rows.jsonl line-by-line (only
n_steps + the last trajectory step + gold_answer are kept per row) so memory stays
flat regardless of file size. CPU-only, deterministic, idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.doc_scoring import extract_answer_span  # noqa: E402
from evaluation.metrics import answer_em  # noqa: E402

BUDGETS = [10, 20, 30, 40, 50, 60, 80, 100]
MIN_ROWS = 100  # skip conditions with fewer rows than this (too noisy to plot)

MAIN_DATASET_DIRS = [
    "browsecomp_plus_structured",
    "browsecomp_plus_flat",
    "musique_structured",
    "musique_flat",
]

# runs_old/ablation_50steps_*/<ds>__<model>__<cond>/rows.jsonl -- only these
# dataset/condition combos are in scope (per the analysis spec).
ABLATION_DATASETS = [
    "browsecomp_plus_structured",
    "browsecomp_plus_flat",
    "musique_structured",
    "musique_flat",
]
ABLATION_CONDITIONS = ["agent_research", "agent_research_bm25"]


def iter_rows(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def score_row(row: dict) -> tuple[int | None, bool]:
    """Return (n_steps, correct) for one row, recomputed with the fixed extractor."""
    traj = row.get("trajectory") or []
    if traj:
        raw = traj[-1].get("raw_output", "") or ""
    else:
        raw = row.get("final_answer", "") or ""
    pred = extract_answer_span(raw)
    gold = row.get("gold_answer", "") or ""
    correct = answer_em(pred, gold) == 1
    n_steps = row.get("n_steps")
    return n_steps, correct


def analyze_condition(rows_path: Path) -> dict:
    """Stream rows_path once; return per-budget EM/answered stats + step buckets.

    Only n_steps, the last trajectory step, and gold_answer are extracted per row;
    the parsed row dict is dropped immediately after, so memory stays O(1) in the
    number of rows.
    """
    n_total = 0
    correct_total = 0
    em_num = {b: 0 for b in BUDGETS}
    answered_num = {b: 0 for b in BUDGETS}
    n_missing_steps = 0

    # Buckets: (0,10], (10,20], ..., (80,100], (100, inf) -- one more than BUDGETS,
    # the last one catching episodes that never answered within the largest budget.
    edges = [0] + BUDGETS
    bucket_total = [0] * (len(edges))  # index len(BUDGETS) is the ">100" tail
    bucket_correct = [0] * (len(edges))

    for row in iter_rows(rows_path):
        n_total += 1
        n_steps, correct = score_row(row)
        if correct:
            correct_total += 1
        if n_steps is None:
            n_missing_steps += 1
            continue
        for b in BUDGETS:
            if n_steps <= b:
                answered_num[b] += 1
                if correct:
                    em_num[b] += 1
        idx = len(edges) - 1  # default: tail bucket (n_steps > max(BUDGETS))
        for i in range(len(edges) - 1):
            if edges[i] < n_steps <= edges[i + 1]:
                idx = i
                break
        bucket_total[idx] += 1
        if correct:
            bucket_correct[idx] += 1

    return {
        "path": str(rows_path),
        "n_total": n_total,
        "n_missing_steps": n_missing_steps,
        "correct_total": correct_total,
        "final_em": (correct_total / n_total) if n_total else 0.0,
        "em_at_b": {b: (em_num[b] / n_total if n_total else 0.0) for b in BUDGETS},
        "answered_at_b": {b: (answered_num[b] / n_total if n_total else 0.0) for b in BUDGETS},
        "bucket_edges": edges,
        "bucket_total": bucket_total,
        "bucket_correct": bucket_correct,
    }


def discover_main_conditions() -> dict[str, dict[str, dict]]:
    """dataset_dir -> condition_label -> stats, for conditions with >= MIN_ROWS rows."""
    out: dict[str, dict[str, dict]] = {}
    runs_agent = REPO_ROOT / "runs" / "agent"
    for ds in MAIN_DATASET_DIRS:
        ds_dir = runs_agent / ds
        if not ds_dir.is_dir():
            continue
        conditions: dict[str, dict] = {}
        for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for cond_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                rows_path = cond_dir / "rows.jsonl"
                if not rows_path.is_file():
                    continue
                stats = analyze_condition(rows_path)
                if stats["n_total"] < MIN_ROWS:
                    continue
                label = cond_dir.name
                if len(list(ds_dir.iterdir())) > 1 or label in conditions:
                    label = f"{model_dir.name}/{cond_dir.name}"
                conditions[label] = stats
        if conditions:
            out[ds] = conditions
    return out


def discover_ablation_conditions() -> tuple[dict[str, dict[str, dict]], str]:
    """dataset -> condition -> stats, for the budget50-ablation runs_old runs.

    Returns (conditions, ablation_dir_name); ablation_dir_name is "" if no
    ablation_50steps_* directory was found under runs_old/.
    """
    out: dict[str, dict[str, dict]] = {}
    runs_old = REPO_ROOT / "runs_old"
    if not runs_old.is_dir():
        return out, ""
    ablation_dirs = sorted(runs_old.glob("ablation_50steps_*"))
    if not ablation_dirs:
        return out, ""
    ablation_dir = ablation_dirs[-1]  # most recent timestamp, deterministic
    for ds in ABLATION_DATASETS:
        conditions: dict[str, dict] = {}
        for cond in ABLATION_CONDITIONS:
            matches = sorted(ablation_dir.glob(f"{ds}__*__{cond}"))
            for run_dir in matches:
                rows_path = run_dir / "rows.jsonl"
                if not rows_path.is_file():
                    continue
                stats = analyze_condition(rows_path)
                conditions[cond] = stats
        if conditions:
            out[ds] = conditions
    return out, ablation_dir.name


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}"


def em_table(conditions: dict[str, dict]) -> list[str]:
    header = ["condition", "n"] + [f"EM@{b}" for b in BUDGETS] + ["final EM"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for label in sorted(conditions):
        s = conditions[label]
        row = [label, str(s["n_total"])]
        row += [fmt_pct(s["em_at_b"][b]) for b in BUDGETS]
        row += [fmt_pct(s["final_em"])]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def answered_table(conditions: dict[str, dict]) -> list[str]:
    header = ["condition", "n"] + [f"%answered@{b}" for b in BUDGETS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for label in sorted(conditions):
        s = conditions[label]
        row = [label, str(s["n_total"])]
        row += [fmt_pct(s["answered_at_b"][b]) for b in BUDGETS]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def bucket_label(edges: list[int], i: int) -> str:
    if i == len(edges) - 1:
        return f">{edges[-1]}"
    return f"({edges[i]},{edges[i + 1]}]"


def conditional_table(conditions: dict[str, dict]) -> list[str]:
    edges = next(iter(conditions.values()))["bucket_edges"]
    labels = [bucket_label(edges, i) for i in range(len(edges))]
    header = ["condition"] + labels
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for label in sorted(conditions):
        s = conditions[label]
        cells = []
        for i in range(len(s["bucket_edges"])):
            tot = s["bucket_total"][i]
            corr = s["bucket_correct"][i]
            if tot == 0:
                cells.append("--")
            else:
                cells.append(f"{fmt_pct(corr / tot)} (n={tot})")
        lines.append("| " + " | ".join([label] + cells) + " |")
    return lines


def step_to_95pct_final(stats: dict) -> str:
    final_em = stats["final_em"]
    if final_em <= 0:
        return "n/a (final EM = 0)"
    target = 0.95 * final_em
    for b in BUDGETS:
        if stats["em_at_b"][b] >= target:
            return str(b)
    return f">{BUDGETS[-1]}"


def post50_tail_accuracy(stats: dict) -> str:
    edges = stats["bucket_edges"]
    tot = corr = 0
    for i in range(len(edges) - 1):
        if edges[i] >= 50:
            tot += stats["bucket_total"][i]
            corr += stats["bucket_correct"][i]
    tot += stats["bucket_total"][-1]
    corr += stats["bucket_correct"][-1]
    if tot == 0:
        return "n/a (no episodes answered after step 50)"
    return f"{fmt_pct(corr / tot)}% (n={tot})"


def build_doc(main: dict[str, dict[str, dict]],
              ablation: dict[str, dict[str, dict]],
              ablation_name: str) -> str:
    lines = []
    lines.append("# Budget curves: accuracy as a function of step budget")
    lines.append("")
    lines.append(
        "Auto-generated by `scripts/budget_curves.py` from EXISTING `rows.jsonl` "
        "files -- no new agent runs. Every row records the full trajectory an "
        "episode actually ran plus `n_steps` (the step it answered/stopped at). An "
        "episode that answered at step `s` would have produced the same answer "
        "under any budget `b >= s` (nothing about the trajectory up to `s` changes "
        "with a larger cap); under `b < s` it never got to answer, so it counts as "
        "wrong at that budget:"
    )
    lines.append("")
    lines.append("    EM@b = fraction of rows with n_steps <= b AND correct answer")
    lines.append("")
    lines.append(
        "**Caveat.** This slightly UNDERSTATES small-`b` accuracy for episodes "
        "that hit the run's OWN step cap (`max_steps` in `config.json` -- 100 for "
        "the main runs, 50 for the `budget50-ablation` runs) and got "
        "force-answered at `b_max`: those rows have `n_steps` at (or just past) "
        "the cap with an empty/low-quality forced answer, so they score wrong at "
        "every budget we evaluate, including the largest one. A hypothetical run "
        "with a bigger cap might have let some of them keep working and eventually "
        "answer correctly -- we have no trajectory steps past the cap to know, so "
        "this analysis cannot recover that; it is a lower bound on accuracy under "
        "larger hypothetical budgets, not an upper bound."
    )
    lines.append("")
    lines.append(
        "Answer extraction uses the FIXED pipeline: `extract_answer_span` (last "
        "`<answer>...</answer>` span, not the first) applied to the last "
        "trajectory step's `raw_output` (or `final_answer` if the trajectory is "
        "empty), scored with `answer_em` against `gold_answer`. This is "
        "recomputed fresh from raw text, not read from the row's stored "
        "`answer_em` field."
    )
    lines.append("")
    lines.append(f"Conditions included: >= {MIN_ROWS} rows.")
    lines.append("")

    for ds in MAIN_DATASET_DIRS:
        if ds not in main:
            continue
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("### EM@b")
        lines.append("")
        lines += em_table(main[ds])
        lines.append("")
        lines.append("### % episodes answered by step b (n_steps <= b)")
        lines.append("")
        lines += answered_table(main[ds])
        lines.append("")
        lines.append(
            "### Marginal value of budget: P(correct | answered in step bucket)"
        )
        lines.append("")
        lines += conditional_table(main[ds])
        lines.append("")

    lines.append(f"## Budget-50 ablation runs (label: `budget50-ablation`, source: `runs_old/{ablation_name}`)")
    lines.append("")
    lines.append(
        "These runs were capped at `max_steps=50`, so `EM@60`/`EM@80`/`EM@100` "
        "are mechanically equal (or nearly equal) to `EM@50`/final EM -- there are "
        "essentially no rows with `n_steps` between 51 and 100 to add. They are "
        "included anyway for consistency with the main tables above."
    )
    lines.append("")
    for ds in ABLATION_DATASETS:
        if ds not in ablation:
            continue
        lines.append(f"### {ds} (budget50-ablation)")
        lines.append("")
        lines += em_table(ablation[ds])
        lines.append("")
        lines += answered_table(ablation[ds])
        lines.append("")
        lines += conditional_table(ablation[ds])
        lines.append("")

    # -- Reading section -----------------------------------------------------
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Step at which each condition reaches 95% of its own final EM (i.e. the "
        "smallest `b` in the sweep with `EM@b >= 0.95 * final_EM`), and the "
        "conditional accuracy of episodes that first answered after step 50 (the "
        "\"grind tail\" -- did persisting past step 50 pay off?):"
    )
    lines.append("")
    header = ["dataset", "condition", "n", "final EM", "step @ 95% of final EM", "post-50 tail accuracy"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for ds in MAIN_DATASET_DIRS:
        if ds not in main:
            continue
        for label in sorted(main[ds]):
            s = main[ds][label]
            lines.append("| " + " | ".join([
                ds, label, str(s["n_total"]), f"{fmt_pct(s['final_em'])}",
                step_to_95pct_final(s), post50_tail_accuracy(s),
            ]) + " |")
    for ds in ABLATION_DATASETS:
        if ds not in ablation:
            continue
        for label in sorted(ablation[ds]):
            s = ablation[ds][label]
            lines.append("| " + " | ".join([
                f"{ds} (budget50-ablation)", label, str(s["n_total"]),
                f"{fmt_pct(s['final_em'])}", step_to_95pct_final(s),
                post50_tail_accuracy(s),
            ]) + " |")
    lines.append("")
    lines.append(
        "Caveat (repeated from above): the post-50 tail accuracy and the "
        "small-`b` end of every EM@b curve are a lower bound -- rows that hit the "
        "run's own step cap and got force-answered contribute 0 at every budget, "
        "including the largest one evaluated here."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    main_conditions = discover_main_conditions()
    ablation_conditions, ablation_name = discover_ablation_conditions()

    doc = build_doc(main_conditions, ablation_conditions, ablation_name)
    out_path = REPO_ROOT / "docs" / "budget_curves.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    print(f">> wrote {out_path}")

    for ds, conds in main_conditions.items():
        for label, s in conds.items():
            print(f"{ds}/{label}: n={s['n_total']} final_EM={fmt_pct(s['final_em'])}%")
    for ds, conds in ablation_conditions.items():
        for label, s in conds.items():
            print(f"[ablation] {ds}/{label}: n={s['n_total']} final_EM={fmt_pct(s['final_em'])}%")


if __name__ == "__main__":
    main()
