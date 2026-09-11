#!/usr/bin/env python3
"""Surgical row removal from a rows.jsonl, for per-instance resume.

Use case: a handful of rows in a LIVE rows.jsonl are known-bad (e.g. produced
during a degraded-retriever window, or a resume-induced duplicate) and need to
be deleted so the next resume re-runs exactly those instance_ids and nothing
else.

  python scripts/prune_rows.py runs/_dense_validation/.../rows.jsonl \
      browsecomp_plus_structured__14 browsecomp_plus_structured__88 --live-ok

  python scripts/prune_rows.py <rows.jsonl> --ids-file bad_ids.txt --live-ok

WHAT IT DOES
  (a) Backs up the CURRENT file to `<rows.jsonl>.pre_prune_<stamp>` in the
      same directory (stamp = UTC `YYYYMMDDTHHMMSSffffff`, collision-safe).
  (b) Rewrites rows.jsonl ATOMICALLY (temp file in the same directory,
      `os.replace()`), excluding rows whose `instance_id` is in the removal
      set.
  (c) Prints removed/kept counts (and a malformed-line count, see below).

Both the backup and the rewritten file are derived from a SINGLE read of
rows.jsonl taken at the start of the run, not backup-then-reread, so the
two are always mutually consistent with each other.

SAFETY, rows.jsonl may be LIVE (a running SLURM job's eval loop appends to
it after every completed instance). This script CANNOT reliably detect "is a
job currently writing this directory" from inside a plain script (no lock
file, no PID registry) -- so it does not try. Instead:

  * You must pass --live-ok. This is not a correctness check; it is a
    forcing function: the caller is asserting they have confirmed (e.g. via
    `squeue`/`sacct`, or by knowing the job finished/was cancelled) that no
    job is actively appending to this exact rows.jsonl right now.
  * Even with --live-ok, the read -> filter -> os.replace sequence is NOT
    transactional against a concurrent appender. Worst case: a writer
    appends a new row after this script's read but BEFORE its os.replace.
    That row is invisible to this run (it wasn't in the snapshot we read)
    and is silently overwritten/lost when os.replace() lands, because
    os.replace() swaps the whole file, not just the lines we changed. This
    is why --live-ok is required rather than assumed: appending is safe
    against a concurrent appender (worst case one write is delayed a few
    ms); pruning is NOT, because it replaces the whole file wholesale.
    If in doubt, confirm the job is not running before pruning.

MALFORMED TAIL LINE POLICY: a JSON-decode failure on the LAST non-empty line
of the file (only) is treated as a torn write -- a live writer's `f.write()`
followed by `f.flush()` is not atomic against a concurrent reader, so the
last line can legitimately be a partial JSON object mid-append. This script's
policy is PRESERVE: any line that fails to parse is written back to the
output verbatim, UNCHANGED, and counted separately as "malformed (kept)" --
never silently dropped. Rationale: we cannot determine a malformed line's
instance_id, so we cannot know whether it was meant to be pruned; dropping
it would be an unrecoverable, unlogged data loss, whereas keeping it is
always safe (worst case: the next append completes it, or a human inspects
it later). This mirrors the rest of this codebase's stance on rows.jsonl
(e.g. scripts/force_answer_backfill.py: "rows.jsonl is only ever read, never
written or mutated" by auxiliary tooling) -- pruning is the one sanctioned
exception, and even it defaults to not-losing-data over cleanliness.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile


def _read_ids_file(path: str) -> list[str]:
    ids = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def _backup_path(rows_path: str) -> str:
    d = os.path.dirname(rows_path) or "."
    base = os.path.basename(rows_path)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    candidate = os.path.join(d, f"{base}.pre_prune_{stamp}")
    # Collision-safe: astronomically unlikely at microsecond resolution, but
    # never silently clobber an existing backup.
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(d, f"{base}.pre_prune_{stamp}-{n}")
        n += 1
    return candidate


def prune_rows(rows_path: str, remove_ids: set[str], live_ok: bool) -> dict:
    if not live_ok:
        raise SystemExit(
            "refusing to run: --live-ok was not given.\n"
            "This script cannot detect whether a SLURM job is currently appending to\n"
            f"{rows_path!r}. Confirm (e.g. `squeue -u $USER`) that no job is actively\n"
            "writing this run's condition dir, then re-run with --live-ok. See this\n"
            "script's module docstring for exactly what --live-ok does and does not\n"
            "guarantee."
        )
    if not os.path.isfile(rows_path):
        raise SystemExit(f"no such file: {rows_path!r}")
    if not remove_ids:
        raise SystemExit("no instance_ids given (positional args and/or --ids-file)")

    # Single read: this snapshot is the source of truth for both the backup
    # and the rewrite, so the two are always consistent with each other.
    with open(rows_path, "r") as fh:
        raw_lines = fh.readlines()

    kept_lines = []
    removed = 0
    malformed_kept = 0
    matched_ids = set()
    n = len(raw_lines)
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue  # drop genuinely blank lines silently (not data)
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            is_last = i == n - 1 or all(not l.strip() for l in raw_lines[i + 1 :])
            if is_last:
                # Torn tail write during a live append: preserve verbatim, never drop.
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                malformed_kept += 1
                continue
            # A malformed line that is not the tail is unexpected corruption,
            # not a live-append artifact -- surface it loudly rather than
            # guessing, but still preserve it (same no-silent-data-loss policy).
            print(
                f"WARNING: {rows_path}:{i + 1} is malformed JSON and is NOT the "
                "last line (not explainable by a live tail-append); preserving "
                "verbatim.",
                file=sys.stderr,
            )
            kept_lines.append(line if line.endswith("\n") else line + "\n")
            malformed_kept += 1
            continue

        rid = row.get("instance_id")
        if rid in remove_ids:
            removed += 1
            matched_ids.add(rid)
            continue
        kept_lines.append(line if line.endswith("\n") else line + "\n")

    kept = len(kept_lines) - malformed_kept

    backup_path = _backup_path(rows_path)
    with open(backup_path, "w") as fh:
        fh.writelines(raw_lines)

    d = os.path.dirname(rows_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".prune_rows.", dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.writelines(kept_lines)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, rows_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return {
        "backup_path": backup_path,
        "removed": removed,
        "kept": kept,
        "malformed_kept": malformed_kept,
        "requested_ids": len(remove_ids),
        "unmatched_ids": sorted(remove_ids - matched_ids),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rows_path", help="path to the rows.jsonl to prune")
    ap.add_argument(
        "instance_ids", nargs="*", help="instance_id(s) to remove (in addition to --ids-file)"
    )
    ap.add_argument(
        "--ids-file",
        help="file with one instance_id per line (blank lines / '#' comments ignored)",
    )
    ap.add_argument(
        "--live-ok",
        action="store_true",
        help="required: asserts the caller has confirmed no job is actively writing this file",
    )
    args = ap.parse_args()

    ids = list(args.instance_ids)
    if args.ids_file:
        ids.extend(_read_ids_file(args.ids_file))
    remove_ids = set(ids)

    result = prune_rows(args.rows_path, remove_ids, args.live_ok)

    print(f"backup:         {result['backup_path']}")
    print(f"requested ids:  {result['requested_ids']}")
    print(f"removed:        {result['removed']}")
    print(f"kept:           {result['kept']}" + (
        f"  (+{result['malformed_kept']} malformed line(s) preserved verbatim)"
        if result["malformed_kept"] else ""
    ))
    if result["unmatched_ids"]:
        print(
            f"note: {len(result['unmatched_ids'])} requested id(s) were not found in "
            f"{args.rows_path} (no-op for those): {result['unmatched_ids']}"
        )


if __name__ == "__main__":
    main()
