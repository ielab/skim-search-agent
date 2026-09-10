"""SWE-bench / Loc-Bench: the code-localization arm. An `Instance` here carries the issue
text and the gold patch; `files` is None (real repos, resolved by `repo.get_files` at the
base commit) except for the built-in fixture in `fixtures.py`.
"""
from __future__ import annotations

import os

from agent_search.evaluation.datasets.base import Instance, data_dir, register_dataset


def _local_dir(hf_name: str) -> str:
    """Where a pre-downloaded split lives, e.g. data/SWE-bench_Verified."""
    return os.path.join(data_dir(), hf_name.split("/")[-1])


def load_swebench(name: str = "princeton-nlp/SWE-bench_Verified",
                  split: str = "test") -> list[Instance]:
    """Load a SWE-bench split. Reads the pre-downloaded copy from `data/` if present
    (offline GPU node); otherwise downloads via HF (needs internet)."""
    local = _local_dir(name)
    if os.path.isdir(local) and os.listdir(local):
        from datasets import load_from_disk
        ds = load_from_disk(local)            # offline, from data/ — no network, no HF cache
    else:
        import sys
        print(f"[agent_search] WARNING: {local} is NOT staged — falling back to HuggingFace "
              f"({name}) via network / HF cache. This is the 'caching' you may see. "
              f"Run `bash scripts/download_data.sh` to stage it for offline use.",
              file=sys.stderr)
        from datasets import load_dataset  # lazy import: heavy dep
        ds = load_dataset(name, split=split)
    return [
        Instance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row["patch"],
            files=None,
        )
        for row in ds
    ]


def _swebench(hf_name: str):
    def load(limit=None, corpus_limit=None):
        rows = load_swebench(hf_name)
        return rows[:limit] if limit is not None else rows
    return load


register_dataset("swebench_verified")(_swebench("princeton-nlp/SWE-bench_Verified"))
register_dataset("swebench_lite")(_swebench("princeton-nlp/SWE-bench_Lite"))
register_dataset("loc_bench")(_swebench("czlll/Loc-Bench_V1"))  # LocAgent's own benchmark
