#!/usr/bin/env python
"""Re-score existing runs IN PLACE with the corrected metrics — NO regeneration, NO model calls.

For every rows.jsonl it recomputes the answer metrics from the STORED final_answer / gold_answer /
observations (via evaluation.doc_scoring.score_answer, which now emits `answer_cover_em` — the
non-strict EM that credits a gold answer contained in a free-form response, the correct metric for
a verbose agent answerer). It writes the rows back atomically and re-aggregates results.json with
the fixed aggregator (so answer_cover_em / timeout_rate / fix_file_ok all land in results.json).

  envs/bin/python scripts/patch_evaluation.py                     # all finished runs
  envs/bin/python scripts/patch_evaluation.py --dir runs/agent --workers 4
  envs/bin/python scripts/patch_evaluation.py --force             # include live-being-written files

SAFETY: a rows.jsonl written within --skip-recent seconds is assumed to have a LIVE appender (a
still-running job) and is SKIPPED — rewriting it would race the appender and lose rows. Re-run
after the sweep finishes to catch those. Code rows (no gold_answer) are left untouched.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.doc_scoring import score_answer
from evaluation.run_eval import _aggregate


def _rescore(r: dict) -> bool:
    """Recompute a doc-QA row's answer metrics in place. Returns True if anything changed.
    Non-doc rows (no gold_answer — e.g. code fix rows) are left exactly as-is."""
    gold = r.get("gold_answer")
    if not gold:
        return False
    try:
        new = score_answer(r.get("final_answer", "") or "", gold, r.get("observations") or [])
    except Exception:                       # a malformed row must not sink the file
        return False
    changed = any(r.get(k) != v for k, v in new.items())
    r.update(new)
    return changed


def patch_file(path: str) -> tuple[int, int]:
    rows, changed = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                    # partial trailing line from a killed run — drop it
            if not r.get("skipped") and _rescore(r):
                changed += 1
            rows.append(r)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    os.replace(tmp, path)                    # atomic swap on the same filesystem
    rj = os.path.join(os.path.dirname(path), "results.json")
    if os.path.exists(rj):
        res = json.load(open(rj))
        res["metrics"] = _aggregate([r for r in rows if not r.get("skipped")])
        json.dump(res, open(rj, "w"), indent=2)
    return len(rows), changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/agent")
    ap.add_argument("--skip-recent", type=int, default=300,
                    help="skip rows.jsonl modified within N seconds (a live appender)")
    ap.add_argument("--force", action="store_true", help="process even actively-written files")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    now = time.time()
    files = sorted(glob.glob(os.path.join(args.dir, "**", "rows.jsonl"), recursive=True))
    todo, skipped = [], []
    for f in files:
        if not args.force and (now - os.path.getmtime(f)) < args.skip_recent:
            skipped.append(f)
        else:
            todo.append(f)
    for f in skipped:
        print(f"  SKIP (live) {os.path.relpath(os.path.dirname(f), args.dir)}", flush=True)

    def work(f):
        n, ch = patch_file(f)
        return os.path.relpath(os.path.dirname(f), args.dir), n, ch

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, f) for f in todo]):
            d, n, ch = fut.result()
            print(f"  patched {d}: {n} rows, {ch} rescored", flush=True)
    print(f"DONE: {len(todo)} files patched, {len(skipped)} skipped (live). "
          f"Re-run after the sweep to patch the skipped ones.")


if __name__ == "__main__":
    main()
