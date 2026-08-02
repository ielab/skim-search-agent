#!/usr/bin/env python
"""Pre-pull & PERSIST all SWE-bench eval images (SIFs) for lite + verified, ONCE.

A resilient, self-waiting daemon: pulls until Docker Hub's per-window rate limit, then SLEEPS
and resumes automatically — no manual re-run needed. After this, grading is fully offline and
each image is reused read-only forever (apptainer runs with --writable-tmpfs; the SIF is never
mutated — no backup/restore needed).

  module load apptainer
  .venv_swebench/bin/python scripts/warm_swebench_images.py --subsets lite,verified

RESUMABLE: skips SIFs already present, so a kill + relaunch continues where it stopped.

RATE LIMIT: anonymous Docker Hub pulls are limited PER IP (~100/6h) and the login node's IP is
SHARED with other users. On a rate-limit hit the daemon waits `--wait` seconds (default 1h) and
retries the SAME image, up to `--max-waits` times (~covers the rolling 6h window), then skips it.
To raise the limit, authenticate (200/6h, dedicated to your account):
  apptainer registry login --username <dockerhub-user> docker://docker.io
  # or export APPTAINER_DOCKER_USERNAME / APPTAINER_DOCKER_PASSWORD before running.

Storage ~1GB/image; 707 unique across lite+verified (~707GB). The pull CACHE (blobs) is cleaned
periodically so only the flat SIFs persist.
"""
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.swebench_apptainer import image_url, SUBSETS, REPO_ROOT

_RATE_MARKERS = ("toomanyrequests", "rate limit", "429", "too many requests")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def pull_one(iid: str, sif_dir: str, cache_dir: str) -> str:
    """Return 'cached' | 'ok' | 'ratelimited' | 'failed:<reason>'."""
    sif = os.path.join(sif_dir, f"{iid}.sif")
    if os.path.exists(sif) and os.path.getsize(sif) > 0:
        return "cached"
    # build SIFs in FIXED-name LOCAL /tmp (fast NVMe, survives session close — a session-scoped
    # /tmp path is what broke the overnight run; Lustre survives but is slow to build on).
    tmp = os.environ.get("SWEBENCH_APPT_TMPDIR") or f"/tmp/{getpass.getuser()}-apptainer-build"
    os.makedirs(tmp, exist_ok=True)
    env = {**os.environ, "APPTAINER_CACHEDIR": cache_dir, "SINGULARITY_CACHEDIR": cache_dir,
           "APPTAINER_TMPDIR": tmp, "SINGULARITY_TMPDIR": tmp, "TMPDIR": tmp}
    r = subprocess.run(["apptainer", "pull", "--force", sif, image_url(iid)],
                       env=env, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(sif) and os.path.getsize(sif) > 0:
        return "ok"
    err = ((r.stderr or "") + (r.stdout or "")).lower()
    if os.path.exists(sif):
        os.remove(sif)                                   # drop partial/empty
    if any(m in err for m in _RATE_MARKERS):
        return "ratelimited"
    return "failed:" + (r.stderr or "").strip()[-160:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="lite,verified", help="comma list of: lite,verified,full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sif-dir", default=os.path.join(REPO_ROOT, ".cache/swebench_sif"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".cache/apptainer"))
    ap.add_argument("--wait", type=int, default=3600, help="seconds to sleep on a rate-limit hit")
    ap.add_argument("--max-waits", type=int, default=8, help="max consecutive waits per image before skipping")
    ap.add_argument("--workers", type=int, default=6, help="concurrent pulls (authenticated 200/hr allows it)")
    args = ap.parse_args()

    from datasets import load_dataset
    ids, seen = [], set()
    for sub in args.subsets.split(","):
        sub = sub.strip()
        if sub not in SUBSETS:
            sys.exit(f"unknown subset {sub}")
        for r in load_dataset(SUBSETS[sub], split=args.split):
            if r["instance_id"] not in seen:
                seen.add(r["instance_id"]); ids.append(r["instance_id"])

    os.makedirs(args.sif_dir, exist_ok=True)
    have = {f[:-4] for f in os.listdir(args.sif_dir) if f.endswith(".sif")}
    todo = [i for i in ids if i not in have]
    _log(f"{len(ids)} unique images; {len(have)} cached; {len(todo)} to pull "
         f"(workers={args.workers}, wait={args.wait}s, max_waits={args.max_waits})")

    def pull_with_retry(iid: str):
        """Pull one image, waiting+retrying on rate-limit. Returns (status, iid, detail)."""
        waits = 0
        while True:
            st = pull_one(iid, args.sif_dir, args.cache_dir)
            if st in ("ok", "cached"):
                return ("pulled", iid, st)
            if st == "ratelimited":
                waits += 1
                if waits > args.max_waits:
                    return ("skipped", iid, "rate-limited (gave up)")
                time.sleep(args.wait)
                continue
            return ("skipped", iid, st[7:] if st.startswith("failed:") else st)

    pulled = skipped = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(pull_with_retry, iid) for iid in todo]
        for fut in as_completed(futs):
            status, iid, detail = fut.result()
            done += 1
            if status == "pulled":
                pulled += 1
                _log(f"[{done}/{len(todo)}] pulled {iid}  (total {len(have)+pulled}/{len(ids)})")
            else:
                skipped += 1
                _log(f"[{done}/{len(todo)}] SKIP {iid} — {detail}")

    # clean the pull-cache ONCE at the end (mid-run cleaning races with concurrent pulls)
    subprocess.run(["apptainer", "cache", "clean", "-f"],
                   env={**os.environ, "APPTAINER_CACHEDIR": args.cache_dir}, capture_output=True)
    now = len([f for f in os.listdir(args.sif_dir) if f.endswith(".sif")])
    size = subprocess.run(["du", "-sh", args.sif_dir], capture_output=True, text=True).stdout.split()[0]
    _log(f"DONE this run: pulled={pulled} skipped={skipped} | SIFs={now}/{len(ids)} ({size}). "
         f"Relaunch to retry any skipped/missing.")


if __name__ == "__main__":
    main()
