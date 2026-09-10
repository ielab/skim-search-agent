"""Replay recorded trajectories through the pre-0.3 workspace agent and the condition runner
and compare every observation: old code against new code on the same index (must be equal),
and both against what the run recorded (differs only where the index or its precision changed).

    python scripts/replay_check.py --rows runs/.../rows.jsonl --condition research_dedup_dense \\
        --dataset infoseek_eval_sample [--index-root indexes] [--limit N]

The run's recorded environment knobs (config.json next to rows.jsonl) are exported first so the
tools read the same budgets. The model's recorded generations drive both agents, so the tool
calls are the recorded ones; no model is needed. Kept for the release that still ships the old
code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading


def _export_recorded_knobs(rows_path: str) -> None:
    cfg_path = os.path.join(os.path.dirname(rows_path), "config.json")
    if not os.path.exists(cfg_path):
        return
    knobs = (json.load(open(cfg_path)) or {}).get("env_knobs") or {}
    for k, v in knobs.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)


class RawReplay:
    """A policy that returns the recorded generations in order."""

    def __init__(self, raws):
        self._raws = list(raws)
        self.last_raw = ""

    def propose(self, task, history) -> str:
        i = len(history)
        self.last_raw = self._raws[i] if i < len(self._raws) else "<answer></answer>"
        return self.last_raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index-root", default="indexes")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    _export_recorded_knobs(args.rows)

    from agent_search.evaluation.agent_runner import build_condition_agent
    from agent_search.evaluation.corpus_units import _corpus_key, _units_for_instance
    from agent_search.evaluation.datasets import load_dataset_by_name
    from agent_search.legacy.retriever import _build_legacy_agent
    from agent_search.retrievers.registry import RetrieverConfig

    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    rows = [r for r in rows if not r.get("error")][: args.limit]
    insts = {i.instance_id: i for i in load_dataset_by_name(args.dataset)}
    first = next(iter(insts.values()))
    units, _by_file, files = _units_for_instance(first, "data/repos", False, {}, threading.Lock())
    key = _corpus_key(first)
    cfg = RetrieverConfig(policy="stub", index_root=args.index_root,
                          dense_model=os.environ.get("DENSE_MODEL") or None)
    old = _build_legacy_agent(cfg, f"agent_{args.condition}")()
    new = build_condition_agent(cfg, args.condition)()
    for r in (old, new):
        if getattr(r, "needs_files", False) and files is not None:
            r.set_files(files)
        r.index(units, key=key)

    def _set_policy(r, factory):
        for a in ("_policy_factory", "policy_factory"):
            if hasattr(r, a):
                setattr(r, a, factory)

    n_steps = n_old_new = n_new_rec = 0
    for row in rows:
        inst = insts.get(row["instance_id"])
        if inst is None:
            continue
        rec = row["trajectory"] if isinstance(row["trajectory"], list) else row["trajectory"]["trajectory"]
        raws = [st.get("raw_output", "") for st in rec]
        outs = []
        for r in (old, new):
            _set_policy(r, lambda raws=raws: RawReplay(raws))
            r.search(inst.problem_statement, 10)
            outs.append([(s.name, s.observation) for s in r.last_trajectory.steps])
        o, n = outs
        for i in range(max(len(o), len(n), len(rec))):
            n_steps += 1
            so = o[i] if i < len(o) else None
            sn = n[i] if i < len(n) else None
            sr = (rec[i]["action"], rec[i]["observation"]) if i < len(rec) else None
            if so != sn:
                n_old_new += 1
                print(f"[old != new] {row['instance_id']} step {i}: old={str(so)[:160]!r} new={str(sn)[:160]!r}")
            if sn != sr:
                n_new_rec += 1
        print(f"{row['instance_id']}: {len(rec)} recorded steps; old==new for all: {o == n}; "
              f"new==recorded steps: {sum(1 for i in range(min(len(n), len(rec))) if n[i] == (rec[i]['action'], rec[i]['observation']))}/{len(rec)}")
    print(f"TOTAL steps {n_steps}; old!=new {n_old_new}; new!=recorded {n_new_rec}")
    return 1 if n_old_new else 0


if __name__ == "__main__":
    sys.exit(main())
