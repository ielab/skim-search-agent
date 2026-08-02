#!/usr/bin/env python3
"""Extract things from a run's rows.jsonl without re-running anything.

  python scripts/extract_traces.py RUN_DIR traj astropy__astropy-12907   # pretty-print one episode
  python scripts/extract_traces.py RUN_DIR failed                        # ids + queries of recall@10=0 rows
  python scripts/extract_traces.py RUN_DIR errors                        # error-message frequency table
  python scripts/extract_traces.py RUN_DIR sft > wins.jsonl              # winning episodes (SFT mining)
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def rows_of(run_dir: str):
    with open(f"{run_dir}/rows.jsonl") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def cmd_traj(run_dir: str, instance_id: str) -> int:
    for r in rows_of(run_dir):
        if r["instance_id"] != instance_id:
            continue
        print(f"== {instance_id}  recall@10={r.get('recall@10')} acc@10={r.get('acc@10')}  "
              f"stopped={r.get('stopped')}  llm_calls={r.get('llm_calls')}")
        print(f"   gold: {r.get('gold_ids')}")
        if r.get("declared"):
            print(f"   declared: {r.get('declared')}")
        if r.get("final_answer"):
            print(f"   answer: {r.get('final_answer')!r}  (em={r.get('answer_em')} f1={r.get('answer_f1')})")
        for i, s in enumerate(r.get("trajectory", []), 1):
            print(f"\n-- step {i} [{s.get('action')}]  {s.get('t_llm_s', 0)}s gen / "
                  f"{s.get('t_tool_s', 0)}s tool  "
                  f"({s.get('prompt_tokens', 0)}p+{s.get('completion_tokens', 0)}c tok)")
            if s.get("query"):
                print(f"   call: {s.get('action')}({s.get('query')!r})")
            obs = s.get("observation", "")
            if obs:
                for ln in obs.splitlines()[:8]:
                    print(f"   obs| {ln}")
            raw = s.get("raw_output", "")
            if raw:
                print(f"   raw ({len(raw)} chars):")
                for ln in raw.splitlines()[:20]:
                    print(f"   | {ln}")
        return 0
    print(f"instance {instance_id!r} not found", file=sys.stderr)
    return 1


def cmd_failed(run_dir: str) -> int:
    for r in rows_of(run_dir):
        if r.get("skipped") or r.get("recall@10", 1) > 0:
            continue
        print(f"{r['instance_id']}  gold={r.get('gold_ids')}")
        for q in r.get("queries", []):
            print(f"    {q}")
    return 0


def _step_error(step: dict) -> str:
    """A step's error message, if any — failed tools surface as an `ERROR: ...`
    observation (the trajectory has no separate error field)."""
    obs = step.get("observation") or ""
    return obs if obs.startswith("ERROR") else ""


def cmd_errors(run_dir: str) -> int:
    counts: Counter = Counter()
    for r in rows_of(run_dir):
        for step in r.get("trajectory", []):
            e = _step_error(step)
            if e:
                counts[": ".join(e.split(":", 2)[:2])[:70]] += 1   # group by error prefix
    for msg, n in counts.most_common():
        print(f"{n:4d}  {msg}")
    if not counts:
        print("no errors recorded")
    return 0


def cmd_sft(run_dir: str) -> int:
    """Winning episodes (full gold found) as JSONL — the SFT mining source."""
    for r in rows_of(run_dir):
        if r.get("skipped") or r.get("recall@10", 0) < 1.0:
            continue
        json.dump({"instance_id": r["instance_id"],
                   "gold_ids": r.get("gold_ids"),
                   "stopped": r.get("stopped"),
                   "steps": [{"query": s.get("query", ""), "action": s.get("action"),
                              "raw_output": s.get("raw_output", ""),
                              "n_hits": s.get("n_hits", 0), "error": _step_error(s) or None}
                             for s in r.get("trajectory", [])]}, sys.stdout)
        sys.stdout.write("\n")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir, cmd = sys.argv[1], sys.argv[2]
    if cmd == "traj":
        return cmd_traj(run_dir, sys.argv[3])
    if cmd == "failed":
        return cmd_failed(run_dir)
    if cmd == "errors":
        return cmd_errors(run_dir)
    if cmd == "sft":
        return cmd_sft(run_dir)
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
