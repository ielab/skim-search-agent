#!/usr/bin/env python3
"""Mine SUCCESSFUL agent trajectories out of completed runs into SFT-ready chat data.

A "run" lives at ``runs/agent/<dataset>/<model>/<condition_dir>/rows.jsonl`` (or under
``runs/_headline_validation/agent/...`` etc. — any root, same trailing
``<dataset>/<model>/<condition_dir>`` shape). Each row is one episode: a `trajectory` of
steps (`raw_output` = the assistant's full turn text, `observation` = the tool response),
plus `question` / `gold_answer` / `final_answer`.

A row QUALIFIES as a distillation example iff:
  - it has a non-empty `question` and every trajectory step has a non-empty `raw_output`
    (rows missing either are skipped — nothing to train on);
  - `answer_em(extract_answer_span(<last step's raw_output, else final_answer>), gold_answer)
    == 1.0` — the episode's declared answer, after stripping the ``<answer>`` tag the same
    way evaluation/doc_scoring.py does, exactly matches gold under the canonical QA EM;
  - its step count is within [--min-steps, --max-steps];
  - (only with --require-grounded) the extracted answer is a verbatim contiguous span in
    the episode's observations (evaluation.doc_scoring.answer_in_evidence) — i.e. the
    correct answer was actually read off a document, not recalled from parametric memory.

Each qualifying row is re-emitted as ONE chat-formatted JSON object:

    {"instance_id", "dataset", "condition", "n_steps",
     "messages": [system, user, assistant, user, assistant, ..., assistant]}

where `messages` replays the episode: the condition's composed system prompt
(agent_search.prompts.load_condition, with the ``{{step_budget}}`` placeholder — which
load_condition intentionally leaves unrendered — filled in as "100"), a user turn with the
question, then one assistant turn per trajectory step (`raw_output` verbatim) interleaved
with a user turn wrapping that step's `observation` in ``<tool_response>`` tags — except
after the LAST step, whose assistant turn (the one carrying ``<answer>...</answer>``) is
left as the final message (there is no tool response after the episode ends).

This is DATA PREP ONLY — it does not launch or configure any training.

Usage:
    envs/bin/python scripts/make_sft_data.py \\
        --conditions "runs/agent/hotpotqa_structured/*/agent_research" \\
        --out data/sft/hotpotqa_research.jsonl

    # multiple globs / explicit dirs in one pass, deduped by instance_id across all of them:
    envs/bin/python scripts/make_sft_data.py \\
        --conditions "runs/agent/hotpotqa_structured/*/agent_research" \\
                     "runs/agent/browsecomp_plus_structured/*/agent_research" \\
        --out data/sft/mixed_research.jsonl --require-grounded
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from agent_search.prompts import load_condition
from evaluation.doc_scoring import answer_in_evidence, extract_answer_span
from evaluation.metrics import answer_em

CURRENT_DATE_LINE = "Current date: 2026-07-09"
STEP_BUDGET = "100"


# --- run-dir discovery -------------------------------------------------------

def find_run_dirs(patterns: list[str]) -> list[str]:
    """Expand each glob pattern (or literal path) to run dirs that hold a rows.jsonl,
    deduped and sorted for a deterministic processing order."""
    matched: set[str] = set()
    for pat in patterns:
        hits = glob.glob(pat)
        if not hits and os.path.isdir(pat):
            hits = [pat]
        for h in hits:
            matched.add(os.path.normpath(h))
    return sorted(p for p in matched if os.path.isfile(os.path.join(p, "rows.jsonl")))


def run_dir_meta(run_dir: str) -> tuple[str, str, str, str]:
    """(dataset, model, condition_name, condition_dir) from .../<dataset>/<model>/<condition_dir>."""
    parts = Path(run_dir).parts
    condition_dir = parts[-1]
    model = parts[-2] if len(parts) >= 2 else ""
    dataset = parts[-3] if len(parts) >= 3 else ""
    condition_name = condition_dir[len("agent_"):] if condition_dir.startswith("agent_") else condition_dir
    return dataset, model, condition_name, condition_dir


# --- row -> qualification + chat messages ------------------------------------

def rows_of(run_dir: str):
    with open(os.path.join(run_dir, "rows.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def qualifies(row: dict[str, Any], *, min_steps: int, max_steps: int,
             require_grounded: bool) -> tuple[bool, str, int]:
    """Return (ok, extracted_answer_span, n_steps)."""
    trajectory = row.get("trajectory") or []
    question = row.get("question")
    if not question or not trajectory:
        return False, "", 0
    if any(not step.get("raw_output") for step in trajectory):
        return False, "", 0

    n_steps = len(trajectory)
    if n_steps < min_steps or n_steps > max_steps:
        return False, "", n_steps

    last_raw = trajectory[-1].get("raw_output") or row.get("final_answer", "")
    pred = extract_answer_span(last_raw)
    gold = row.get("gold_answer", "")
    if answer_em(pred, gold) != 1.0:
        return False, "", n_steps

    if require_grounded:
        observations = row.get("observations")
        if not observations:
            observations = [step.get("observation", "") for step in trajectory]
        if not answer_in_evidence(pred, observations):
            return False, "", n_steps

    return True, pred, n_steps


def build_messages(system_prompt: str, question: str, trajectory: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{CURRENT_DATE_LINE}\n\n{question}"},
    ]
    last_idx = len(trajectory) - 1
    for i, step in enumerate(trajectory):
        messages.append({"role": "assistant", "content": step.get("raw_output", "")})
        if i != last_idx:
            observation = step.get("observation", "")
            messages.append({"role": "user", "content": f"<tool_response>\n{observation}\n</tool_response>"})
    return messages


# --- per-condition processing --------------------------------------------------

def process_run_dir(run_dir: str, *, args: argparse.Namespace, seen_instance_ids: set[str],
                     system_prompt_cache: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset, model, condition_name, condition_dir = run_dir_meta(run_dir)

    system_prompt = system_prompt_cache.get(condition_name)
    if system_prompt is None:
        try:
            profile = load_condition(condition_name)
        except ValueError as exc:
            print(f"WARNING: skipping {run_dir}: cannot load condition {condition_name!r}: {exc}",
                  file=sys.stderr)
            return [], {"run_dir": run_dir, "dataset": dataset, "condition": condition_dir,
                        "scanned": 0, "qualified": 0, "written": 0, "dedup": 0, "mean_steps": 0.0}
        system_prompt = profile.system.replace("{{step_budget}}", STEP_BUDGET)
        system_prompt_cache[condition_name] = system_prompt

    scanned = qualified = written = dedup = 0
    written_step_counts: list[int] = []
    out_rows: list[dict[str, Any]] = []

    for row in rows_of(run_dir):
        scanned += 1
        ok, pred, n_steps = qualifies(row, min_steps=args.min_steps, max_steps=args.max_steps,
                                       require_grounded=args.require_grounded)
        if not ok:
            continue
        qualified += 1

        instance_id = row["instance_id"]
        if args.dedup_by == "instance":
            if instance_id in seen_instance_ids:
                dedup += 1
                continue
            seen_instance_ids.add(instance_id)

        messages = build_messages(system_prompt, row["question"], row["trajectory"])
        out_rows.append({
            "instance_id": instance_id,
            "dataset": dataset,
            "condition": condition_dir,
            "n_steps": n_steps,
            "messages": messages,
        })
        written += 1
        written_step_counts.append(n_steps)

    summary = {
        "run_dir": run_dir, "dataset": dataset, "condition": condition_dir,
        "scanned": scanned, "qualified": qualified, "written": written, "dedup": dedup,
        "mean_steps": statistics.mean(written_step_counts) if written_step_counts else 0.0,
    }
    return out_rows, summary


# --- CLI -----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="glob pattern(s) and/or explicit run-dir path(s), e.g. "
                         "'runs/agent/hotpotqa_structured/*/agent_research'")
    ap.add_argument("--out", required=True, help="output JSONL path, e.g. data/sft/hotpotqa_research.jsonl")
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--dedup-by", choices=["instance", "none"], default="instance",
                    help="'instance' keeps only the first qualifying episode per instance_id, "
                         "across ALL matched run dirs in processing order; 'none' disables dedup")
    ap.add_argument("--require-grounded", action="store_true",
                    help="also require the extracted answer to be a verbatim span in the episode's "
                         "tool observations (evaluation.doc_scoring.answer_in_evidence)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dirs = find_run_dirs(args.conditions)
    if not run_dirs:
        print(f"no run dirs with rows.jsonl matched {args.conditions!r}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_instance_ids: set[str] = set()
    system_prompt_cache: dict[str, str] = {}
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows, summary = process_run_dir(run_dir, args=args, seen_instance_ids=seen_instance_ids,
                                         system_prompt_cache=system_prompt_cache)
        all_rows.extend(rows)
        summaries.append(summary)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")

    header = f"{'dataset':<30}{'condition':<24}{'scanned':>9}{'qualified':>11}{'written':>9}{'mean_steps':>12}{'dedup':>7}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['dataset']:<30}{s['condition']:<24}{s['scanned']:>9}{s['qualified']:>11}"
              f"{s['written']:>9}{s['mean_steps']:>12.2f}{s['dedup']:>7}")
    total_scanned = sum(s["scanned"] for s in summaries)
    total_qualified = sum(s["qualified"] for s in summaries)
    total_written = sum(s["written"] for s in summaries)
    total_dedup = sum(s["dedup"] for s in summaries)
    print("-" * len(header))
    print(f"{'TOTAL':<30}{'':<24}{total_scanned:>9}{total_qualified:>11}{total_written:>9}{'':>12}{total_dedup:>7}")
    print(f"\nwrote {total_written} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
