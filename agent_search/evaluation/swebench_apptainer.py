#!/usr/bin/env python
"""Self-hosted SWE-bench resolve-rate harness via apptainer (no Docker).

Grades a `preds.jsonl` ({instance_id, model_patch}) locally on the cluster, unlimited runs,
both lite and verified, by reusing SWE-bench's prebuilt per-instance images through apptainer
instead of Docker. Follows the official harness: in the instance container (repo @ base_commit,
conda `testbed` env), apply the model patch, run the test_spec eval script (resets test files,
applies the gold test_patch, runs FAIL_TO_PASS + PASS_TO_PASS), then grade the log with
`swebench.harness.grading` -> resolved iff all F2P pass and all P2P still pass.

Run with the isolated venv (has the `swebench` package):
  module load apptainer
  .venv_swebench/bin/python -m agent_search.evaluation.swebench_apptainer preds.jsonl --subset lite -o report.json
  # first run pulls each image to .cache/swebench_sif (cached); --limit N for a smoke.

The hosted images encode "__" as "_1776_": swebench/sweb.eval.x86_64.<id-with-_1776_>:latest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SUBSETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def image_url(instance_id: str, namespace: str = "swebench") -> str:
    # SWE-bench hosts prebuilt eval images; the tag encodes "__" as "_1776_".
    tag = instance_id.replace("__", "_1776_")
    return f"docker://{namespace}/sweb.eval.x86_64.{tag}:latest"


def ensure_sif(instance_id: str, sif_dir: str, cache_dir: str, pull: bool = True) -> str | None:
    """Return the cached .sif, pulling it if missing. `pull=False` (offline compute nodes)
    only checks for an already-warmed SIF and returns None if absent, never hits the network."""
    os.makedirs(sif_dir, exist_ok=True)
    sif = os.path.join(sif_dir, f"{instance_id}.sif")
    if os.path.exists(sif) and os.path.getsize(sif) > 0:
        return sif
    if not pull:
        return None
    # apptainer builds each SIF in a TMPDIR. Two traps: (1) the default is the harness's
    # session-namespaced /tmp/<user>.<session>, which is wiped when the terminal closes, so
    # every later pull fails ("failed to create build parent dir"); (2) Lustre scratch survives
    # but is slow for the many-small-file SIF build. Use a fixed-name dir on local /tmp instead
    # (fast NVMe, not session-scoped so it survives session close). Override via
    # SWEBENCH_APPT_TMPDIR.
    import getpass
    tmp = os.environ.get("SWEBENCH_APPT_TMPDIR") or f"/tmp/{getpass.getuser()}-apptainer-build"
    os.makedirs(tmp, exist_ok=True)
    env = {**os.environ, "APPTAINER_CACHEDIR": cache_dir, "SINGULARITY_CACHEDIR": cache_dir,
           "APPTAINER_TMPDIR": tmp, "SINGULARITY_TMPDIR": tmp, "TMPDIR": tmp}
    r = subprocess.run(["apptainer", "pull", "--force", sif, image_url(instance_id)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(sif):
        sys.stderr.write(f"[pull FAIL] {instance_id}: {r.stderr.strip()[-300:]}\n")
        return None
    return sif


def run_one(instance: dict, model_patch: str, sif_dir: str, cache_dir: str,
            work_root: str, timeout: int, no_pull: bool = False) -> dict:
    """Grade one instance. Returns {instance_id, status, resolved, applied}."""
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import get_eval_report
    from swebench.harness.constants import (KEY_INSTANCE_ID, KEY_PREDICTION, KEY_MODEL,
                                            APPLY_PATCH_FAIL, APPLY_PATCH_PASS)

    iid = instance["instance_id"]
    out = {"instance_id": iid, "resolved": False, "applied": False, "status": "error", "error": ""}
    if not model_patch or not model_patch.strip():
        out["status"] = "no_patch"
        return out

    ts = make_test_spec(instance)
    sif = ensure_sif(iid, sif_dir, cache_dir, pull=not no_pull)
    if sif is None:
        out["status"] = "image_missing"
        out["error"] = ("image not warmed (run warm_swebench_images or drop --no-pull)"
                        if no_pull else "image pull failed")
        return out

    work = tempfile.mkdtemp(dir=work_root)
    try:
        with open(os.path.join(work, "patch.diff"), "w") as f:
            f.write(model_patch if model_patch.endswith("\n") else model_patch + "\n")
        with open(os.path.join(work, "eval.sh"), "w") as f:
            f.write(ts.eval_script)
        log_path = os.path.join(work, "test_output.log")
        # apply the model patch (git apply, then a lenient `patch` fallback), then run the
        # test_spec eval script (it resets test files, applies the gold test_patch, runs tests).
        inner = (
            "cd /testbed && "
            "(git apply --verbose /work/patch.diff || "
            " patch --batch --fuzz=5 -p1 -i /work/patch.diff) && "
            f"echo '{APPLY_PATCH_PASS}' && "
            "/bin/bash /work/eval.sh || "
            f"echo '{APPLY_PATCH_FAIL}'"
        )
        env = {**os.environ, "APPTAINER_CACHEDIR": cache_dir, "SINGULARITY_CACHEDIR": cache_dir}
        with open(log_path, "w") as lf:
            try:
                subprocess.run(
                    ["apptainer", "exec", "--writable-tmpfs", "--bind", f"{work}:/work",
                     sif, "bash", "-c", inner],
                    env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                out["status"] = "timeout"
                return out
        log_txt = open(log_path, errors="ignore").read()
        out["applied"] = (APPLY_PATCH_FAIL not in log_txt) and (APPLY_PATCH_PASS in log_txt)
        pred = {KEY_INSTANCE_ID: iid, KEY_PREDICTION: model_patch, KEY_MODEL: "local"}
        report = get_eval_report(ts, pred, log_path, include_tests_status=True)
        rec = report.get(iid, {})
        out["resolved"] = bool(rec.get("resolved", False))
        out["status"] = "resolved" if out["resolved"] else ("applied_unresolved" if out["applied"] else "apply_failed")
        # keep the test tallies for the report
        ts_status = rec.get("tests_status", {})
        out["fail_to_pass"] = ts_status.get("FAIL_TO_PASS", {})
        out["pass_to_pass"] = ts_status.get("PASS_TO_PASS", {})
        return out
    except Exception as e:  # noqa: BLE001 - one instance's failure must not sink the batch
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    finally:
        subprocess.run(["rm", "-rf", work])


def load_preds(path: str) -> dict:
    """Accept either a preds.jsonl or a run dir. From a run dir, read the compiled
    `model_patch` straight out of rows.jsonl (preferring preds.jsonl if already extracted).
    This is what makes it a one-command pipeline: point it at a result set, get graded."""
    if os.path.isdir(path):
        pj = os.path.join(path, "preds.jsonl")
        if os.path.isfile(pj):
            path = pj
        else:
            rows = os.path.join(path, "rows.jsonl")
            if not os.path.isfile(rows):
                sys.exit(f"{path}: no preds.jsonl or rows.jsonl")
            path = rows
    preds = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("model_patch"):
                preds[r["instance_id"]] = r["model_patch"]   # last write wins (resumed runs)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", help="preds.jsonl OR a run dir (reads rows.jsonl/preds.jsonl)")
    ap.add_argument("--subset", choices=list(SUBSETS), default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("-o", "--out", default=None, help="report json (default: <preds dir>/apptainer_report.json)")
    ap.add_argument("--limit", type=int, default=None, help="grade only the first N submitted instances")
    ap.add_argument("--instances", default=None, help="comma-separated instance_ids to grade")
    ap.add_argument("--workers", type=int, default=4, help="concurrent containers (heavy: RAM+CPU per one)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-instance seconds")
    ap.add_argument("--no-pull", action="store_true",
                    help="offline (compute node): grade only instances whose SIF is already warmed")
    ap.add_argument("--sif-dir", default=os.path.join(REPO_ROOT, ".cache/swebench_sif"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".cache/apptainer"))
    ap.add_argument("--work-root", default=os.path.join(REPO_ROOT, ".cache/swebench_work"))
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(SUBSETS[args.subset], split=args.split)
    by_id = {r["instance_id"]: r for r in ds}
    total = len(by_id)

    preds = load_preds(args.preds)
    ids = [i for i in preds if i in by_id]
    if args.instances:
        want = set(args.instances.split(","))
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[:args.limit]
    os.makedirs(args.work_root, exist_ok=True)
    print(f"grading {len(ids)} submitted instances (subset={args.subset}, total={total}, "
          f"workers={args.workers})", flush=True)

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, by_id[i], preds[i], args.sif_dir, args.cache_dir,
                          args.work_root, args.timeout, args.no_pull): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            res = fut.result()
            results[i] = res
            print(f"  [{n}/{len(ids)}] {i:34s} {res['status']:18s} "
                  f"resolved={res['resolved']} applied={res['applied']}", flush=True)

    resolved = [i for i, r in results.items() if r["resolved"]]
    applied = [i for i, r in results.items() if r["applied"]]
    report = {
        "subset": args.subset, "split": args.split,
        "total_instances": total, "submitted": len(results),
        "resolved_count": len(resolved),
        "applied_count": len(applied),
        "resolved_rate_total": len(resolved) / total if total else 0.0,
        "resolved_rate_submitted": len(resolved) / len(results) if results else 0.0,
        "apply_rate_submitted": len(applied) / len(results) if results else 0.0,
        "resolved_ids": sorted(resolved),
        "per_instance": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    base = args.preds if os.path.isdir(args.preds) else os.path.dirname(os.path.abspath(args.preds))
    out = args.out or os.path.join(base, "apptainer_report.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n% Resolved (total {total}):     {report['resolved_rate_total']*100:.2f}%  "
          f"({len(resolved)}/{total})")
    print(f"% Resolved (of submitted):     {report['resolved_rate_submitted']*100:.2f}%  "
          f"({len(resolved)}/{len(results)})")
    print(f"apply rate (of submitted):     {report['apply_rate_submitted']*100:.2f}%  "
          f"({len(applied)}/{len(results)})")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
