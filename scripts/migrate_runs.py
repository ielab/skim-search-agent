#!/usr/bin/env python3
"""One-time: reorganize old flat runs/<name>/ into runs/{agent,retrieval_only}/
and strip the `__limit=N` segment from names (so debug-limit runs merge into the
full run). Merges rows.jsonl by instance_id on collision, then re-evals.

  python scripts/migrate_runs.py runs            # dry-run (prints the plan)
  python scripts/migrate_runs.py runs --apply
"""
import argparse, json, os, re, shutil, sys
sys.path.insert(0, os.getcwd())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", default="runs", nargs="?")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = a.runs_dir
    moves = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        # skip already-organized subfolders and non-run dirs
        if name in ("agent", "retrieval_only", "debug") or not os.path.isdir(d):
            continue
        if not os.path.isfile(os.path.join(d, "rows.jsonl")):
            continue
        retr = name.split("__", 2)[1] if "__" in name else ""
        kind = "agent" if retr.startswith("agent") else "retrieval_only"
        new_name = re.sub(r"__limit=\d+", "", name)
        dst = os.path.join(root, kind, new_name)
        moves.append((d, dst))

    for src, dst in moves:
        print(f"{src}  ->  {dst}")
        if not a.apply:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.move(src, dst)
        else:                                   # merge rows.jsonl, dedup by id
            seen = set()
            merged = []
            for f in (os.path.join(dst, "rows.jsonl"), os.path.join(src, "rows.jsonl")):
                for line in open(f):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["instance_id"] in seen:
                        continue
                    seen.add(r["instance_id"]); merged.append(line.rstrip("\n"))
            open(os.path.join(dst, "rows.jsonl"), "w").write("\n".join(merged) + "\n")
            shutil.rmtree(src)
    if not a.apply:
        print(f"\n(dry-run; {len(moves)} dirs. re-run with --apply, then:")
        print(" python scripts/recompute_metrics.py runs --all)")
    else:
        print(f"\nmoved {len(moves)} dirs. now: python scripts/recompute_metrics.py runs --all")

if __name__ == "__main__":
    main()
