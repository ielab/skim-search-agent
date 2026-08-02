"""Pre-stage SWE-bench repos for OFFLINE eval.

Run on a node WITH internet (e.g. the login node). Clones each unique repo into the
cache and verifies every base_commit is present (fetching it if a full clone somehow
lacks it). The GPU node then reads everything offline via `git archive`
(`agent_search.corpus.repo`) — no cloning at eval time.

    python scripts/prefetch_repos.py --dataset swebench_verified --repo-cache data/repos
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.datasets import load_dataset_by_name  # noqa: E402
from agent_search.corpus.code_repo import clone_dir, ensure_repo     # noqa: E402


def _has_commit(clone: str, commit: str) -> bool:
    return subprocess.run(["git", "-C", clone, "cat-file", "-e", f"{commit}^{{commit}}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swebench_verified",
                    choices=["swebench_verified", "swebench_lite", "loc_bench"])
    ap.add_argument("--repo-cache", default="data/repos")
    args = ap.parse_args()

    instances = load_dataset_by_name(args.dataset)
    repos: dict = {}
    for inst in instances:
        repos.setdefault(inst.repo, set()).add(inst.base_commit)

    missing = []
    for repo, commits in repos.items():
        print(f">> {repo}: {len(commits)} commits -> {clone_dir(repo, args.repo_cache)}")
        clone = ensure_repo(repo, args.repo_cache, allow_clone=True)
        for c in commits:
            if not _has_commit(clone, c):
                subprocess.run(["git", "-C", clone, "fetch", "--quiet", "origin", c],
                               stderr=subprocess.DEVNULL)
                if not _has_commit(clone, c):
                    missing.append((repo, c))

    print(f"\nDONE: {len(instances)} instances, {len(repos)} repos cached at "
          f"{args.repo_cache}. Missing commits: {len(missing)}")
    for repo, c in missing[:20]:
        print(f"  MISSING {repo} {c}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
