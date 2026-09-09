"""Merge scripts/shard_cell.sh's per-shard rows.jsonl back into a cell's canonical
rows.jsonl.

Companion to scripts/shard_cell.sh (INSTANCE-SHARDING): each shard task ran
`run_eval --only-instances shard_<i>of<N>.txt --runs-dir <tier>/__shards/<CONDITION>/
shard_<i>of<N>`, a run-dir DISTINCT from the canonical cell so its appends never
collided with the canonical rows.jsonl or with other shards. This script folds those
shard rows back into the canonical dir agent_search/evaluation/config.py:results_dir_for computes
for the same (dataset, model, condition): <runs_dir>/agent/<dataset>/<model>/<condition>/.

Reuses agent_search.evaluation.run_eval._load_rows (the same parser run_eval itself uses for
resume) rather than re-parsing rows.jsonl by hand, so "done" here means exactly what
run_eval would treat as done on its next invocation.

Safety:
  - backs up the canonical rows.jsonl to rows.jsonl.premerge.bak before any write
    (skipped if there's nothing to add — see idempotency below)
  - dedups by instance_id: canonical rows win over shard rows, and a row already
    folded in by an earlier shard in this same pass wins over a later duplicate
    (shards SHOULD be disjoint by construction, but this is defensive, not assumed)
  - idempotent: re-running when every shard row is already in the canonical file
    adds nothing and does not touch the canonical file or its backup

Usage:
    python scripts/merge_shards.py \\
        --runs-dir runs/_visit_uncapped --dataset browsecomp_plus_structured \\
        --condition agent_research_bm25_autoread --num-shards 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_search.evaluation.run_eval import _load_rows  # noqa: E402  (see module docstring: reuse, don't reimplement)


def _resolve_model_tag(runs_dir: str, dataset: str, condition: str, model: str | None) -> str:
    """Same basename agent_search/evaluation/config.py:results_dir_for uses for the path segment.
    Auto-discovered by globbing the canonical layout when --model isn't given, so the
    caller doesn't have to know/repeat the exact --model string used to run the cell."""
    if model:
        return model.split("/")[-1]
    pattern = os.path.join(runs_dir, "agent", dataset, "*", condition, "rows.jsonl")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(
            f"could not auto-discover MODEL: no canonical rows.jsonl matches {pattern!r}. "
            "Pass --model explicitly.")
    tags = {os.path.basename(os.path.dirname(os.path.dirname(h))) for h in hits}
    if len(tags) > 1:
        raise SystemExit(
            f"ambiguous MODEL for {dataset}/{condition}: found {sorted(tags)} under {runs_dir}. "
            "Pass --model explicitly.")
    return tags.pop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True,
                    help="the TIER root the cell lives under (e.g. runs/_visit_uncapped) "
                         "— the SAME value passed as RUNS_DIR to shard_cell.sh")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", required=True, help="the retriever/condition name")
    ap.add_argument("--num-shards", type=int, required=True,
                    help="NUM_SHARDS the cell was split into (must match shard_cell.sh)")
    ap.add_argument("--model", default=None,
                    help="override the auto-discovered MODEL path segment")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing anything")
    args = ap.parse_args()

    model_tag = _resolve_model_tag(args.runs_dir, args.dataset, args.condition, args.model)
    canonical_dir = os.path.join(args.runs_dir, "agent", args.dataset, model_tag, args.condition)
    canonical_rows_path = os.path.join(canonical_dir, "rows.jsonl")
    shard_root = os.path.join(args.runs_dir, "__shards", args.condition)

    canonical_rows, done = _load_rows(canonical_rows_path)
    n_before = len(canonical_rows)
    print(f">> canonical: {canonical_rows_path}")
    print(f"   canonical_before = {n_before} rows")

    merged_new_rows: list[dict] = []
    per_shard_added: dict[int, int] = {}
    per_shard_dupe: dict[int, int] = {}
    n_missing_shards = 0

    for i in range(args.num_shards):
        shard_rows_path = os.path.join(
            shard_root, f"shard_{i}of{args.num_shards}",
            "agent", args.dataset, model_tag, args.condition, "rows.jsonl")
        if not os.path.exists(shard_rows_path):
            print(f"   shard {i}: NO rows.jsonl at {shard_rows_path} "
                  f"(job not finished / never started?) -- skipped")
            n_missing_shards += 1
            continue
        shard_rows, _ = _load_rows(shard_rows_path)
        added = dupe = 0
        for row in shard_rows:
            iid = row["instance_id"]
            if iid in done:                      # already canonical, or already folded
                dupe += 1                          # in from an earlier shard this pass
                continue
            done.add(iid)
            merged_new_rows.append(row)
            added += 1
        per_shard_added[i] = added
        per_shard_dupe[i] = dupe
        print(f"   shard {i}: {len(shard_rows)} rows -> +{added} new, {dupe} duplicate (skipped)")

    n_added = len(merged_new_rows)
    n_dupe_total = sum(per_shard_dupe.values())
    n_after = n_before + n_added

    print(f"   canonical_after  = {n_after} rows  (+{n_added} added, {n_dupe_total} duplicates skipped)")
    if n_missing_shards:
        print(f"   WARNING: {n_missing_shards}/{args.num_shards} shard(s) had no rows.jsonl yet.")

    if n_added == 0:
        print(">> nothing to merge (idempotent no-op) — canonical unchanged, no backup written.")
        return

    if args.dry_run:
        print(">> --dry-run: not writing anything.")
        return

    backup_path = canonical_rows_path + ".premerge.bak"
    os.makedirs(canonical_dir, exist_ok=True)
    if os.path.exists(canonical_rows_path):
        shutil.copy2(canonical_rows_path, backup_path)
        print(f">> backed up canonical -> {backup_path}")
    else:
        print(">> canonical rows.jsonl did not exist yet -- no backup needed.")

    tmp_path = f"{canonical_rows_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as fh:
        for row in canonical_rows:
            fh.write(json.dumps(row) + "\n")
        for row in merged_new_rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp_path, canonical_rows_path)     # atomic
    print(f">> wrote {canonical_rows_path} ({n_after} rows).")


if __name__ == "__main__":
    main()
