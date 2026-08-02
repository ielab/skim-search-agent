#!/usr/bin/env python
"""Extract a SWE-bench `preds.jsonl` from a finished patch-mode run dir.

The patch-mode agent (agent_codefix_patch / agent_codefix_grep_patch) writes a compiled
`model_patch` into rows.jsonl. This turns a run dir into the exact JSONL the SWE-bench
harness / sb-cli grade:

    {"instance_id": ..., "model_name_or_path": ..., "model_patch": <unified diff>}

Runs OFFLINE (compute/login node) — generation and grading are decoupled: produce preds
here, then submit with sb-cli from a node with internet.

  python scripts/make_swebench_preds.py runs/agent/swebench_verified/<model>/agent_codefix_patch
  python scripts/make_swebench_preds.py <run_dir> --apply-check   # verify each git-applies

--apply-check extracts each instance's base_commit tree (git archive from the repo cache)
and runs `git apply --check`, reporting the REAL applyable rate — the ceiling on resolve rate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

# runnable as a file (`python scripts/make_swebench_preds.py`) without PYTHONPATH=.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_rows(run_dir: str) -> list[dict]:
    fp = os.path.join(run_dir, "rows.jsonl")
    if not os.path.isfile(fp):
        sys.exit(f"no rows.jsonl in {run_dir}")
    return [json.loads(l) for l in open(fp) if l.strip()]


def _model_name(run_dir: str, override: str | None) -> str:
    if override:
        return override
    cfg = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg):
        c = json.load(open(cfg))
        m = c.get("model") or c.get("args", {}).get("model")
        r = c.get("retriever") or c.get("args", {}).get("retriever") or "agent"
        if m:
            return f"{m.split('/')[-1]}__{r}"
    return os.path.basename(run_dir.rstrip("/")) or "agent"


def _apply_ok(inst_id: str, patch: str, cache_dir: str) -> bool | None:
    """True/False if the patch git-applies against base_commit; None if the instance's
    repo/commit isn't available locally (can't judge — not counted as a failure)."""
    from evaluation.datasets import load_dataset_by_name  # lazy: heavy import
    global _INSTS
    try:
        inst = _INSTS[inst_id]
    except Exception:
        return None
    clone = os.path.join(cache_dir, inst.repo.replace("/", "__"))
    if not os.path.isdir(clone):
        return None
    d = tempfile.mkdtemp()
    try:
        ex = subprocess.run(f"git -C {clone} archive {inst.base_commit} | tar -x -C {d}",
                            shell=True, capture_output=True)
        if ex.returncode != 0:
            return None
        subprocess.run(["git", "-C", d, "init", "-q"], check=True)
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        subprocess.run(["git", "-C", d, "-c", "user.email=a@b.c", "-c", "user.name=x",
                        "commit", "-qm", "base"], check=True)
        pf = os.path.join(d, "p.diff")
        open(pf, "w").write(patch)
        chk = subprocess.run(["git", "-C", d, "apply", "--check", "p.diff"],
                             capture_output=True, text=True)
        return chk.returncode == 0
    except Exception:
        return None
    finally:
        subprocess.run(["rm", "-rf", d])


_INSTS: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None, help="preds path (default: <run_dir>/preds.jsonl)")
    ap.add_argument("--model-name", default=None, help="model_name_or_path label (default: from config.json)")
    ap.add_argument("--apply-check", action="store_true", help="git apply --check each patch")
    ap.add_argument("--cache-dir", default="data/repos", help="repo cache for --apply-check")
    ap.add_argument("--dataset", default=None, help="dataset name for --apply-check (default: from config.json)")
    args = ap.parse_args()

    rows = _load_rows(args.run_dir)
    model = _model_name(args.run_dir, args.model_name)
    out = args.out or os.path.join(args.run_dir, "preds.jsonl")

    # dedupe by instance_id (a resumed run may retry) — keep the LAST patch seen.
    preds: dict[str, dict] = {}
    n_rows = len(rows)
    for r in rows:
        iid = r.get("instance_id")
        mp = r.get("model_patch")
        if iid and mp:
            preds[iid] = {"instance_id": iid, "model_name_or_path": model, "model_patch": mp}

    applied = None
    if args.apply_check:
        cfg = os.path.join(args.run_dir, "config.json")
        ds = args.dataset or (json.load(open(cfg)).get("dataset") if os.path.isfile(cfg) else None)
        if ds:
            from evaluation.datasets import load_dataset_by_name
            global _INSTS
            _INSTS = {i.instance_id: i for i in load_dataset_by_name(ds)}
        applied = 0
        checked = 0
        for p in preds.values():
            ok = _apply_ok(p["instance_id"], p["model_patch"], args.cache_dir)
            if ok is not None:
                checked += 1
                applied += int(ok)
        applied = (applied, checked)

    with open(out, "w") as f:
        for p in preds.values():
            f.write(json.dumps(p) + "\n")

    print(f"rows={n_rows}  instances_with_patch={len(preds)}  ->  {out}")
    print(f"model_name_or_path = {model}")
    if applied is not None:
        a, c = applied
        rate = f"{100*a/c:.1f}%" if c else "n/a"
        print(f"git apply --check: {a}/{c} applied cleanly ({rate})   [ceiling on resolve rate]")
    print("\nnext (on a node WITH internet):")
    print(f"  pip install sb-cli   # then set SWEBENCH_API_KEY")
    print(f"  sb-cli submit swe-bench_verified test --predictions_path {out} --run_id <id>")


if __name__ == "__main__":
    main()
