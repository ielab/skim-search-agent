"""Resolve an instance's source files at its base commit, offline-friendly.

Fixtures carry inline `files`. Real SWE-bench instances are read from a local clone
cache via `git archive`, read-only, so:
  * no `git checkout`, so no working-tree mutation: safe under concurrent --workers,
  * works fully offline (the GPU node usually has no internet).

Workflow on a cluster: pre-stage the repos once on a node with internet (a plain
`git clone` of each repository into `<shared dir>/<owner>__<repo>`; there is no automatic
prefetch step here, the codefix task that would need it is not part of the document
workflow), then run the eval with `--repo-cache <shared dir>` on the GPU node.
"""
from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from typing import Any


class RepoError(RuntimeError):
    pass


def clone_dir(repo: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, repo.replace("/", "__"))


def _is_git_repo(path: str) -> bool:
    return subprocess.run(["git", "-C", path, "rev-parse", "--git-dir"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ensure_repo(repo: str, cache_dir: str, allow_clone: bool = False) -> str:
    """Return a local clone path for `repo`. Clones only if `allow_clone` (needs
    internet); otherwise the clone must already exist in the cache."""
    dest = clone_dir(repo, cache_dir)
    if os.path.isdir(dest) and _is_git_repo(dest):
        return dest
    if not allow_clone:
        raise RepoError(
            f"repo {repo} not cached at {dest}. Pre-stage it on a node with internet "
            f"(git clone https://github.com/{repo} {dest}), or pass --allow-clone.")
    if os.path.isdir(dest):                       # exists but broken -> reclone
        subprocess.run(["rm", "-rf", dest], check=True)
    os.makedirs(cache_dir, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet",
                    f"https://github.com/{repo}.git", dest], check=True)
    return dest


def _archive_py_files(clone: str, commit: str) -> dict:
    """Read all .py files at `commit` via `git archive` (read-only, streamed)."""
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(
            ["git", "-C", clone, "archive", "--format=tar", commit],
            stdout=subprocess.PIPE, stderr=errf)
        files: dict = {}
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith(".py"):
                        f = tf.extractfile(m)
                        if f is not None:
                            files[m.name] = f.read().decode("utf-8", "ignore")
        except tarfile.TarError:
            files = {}                            # empty/failed stream -> handled below
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
        if proc.returncode != 0:
            errf.seek(0)
            err = errf.read().decode("utf-8", "ignore").strip()
            raise RepoError(
                f"git archive {commit[:12]} failed in {clone}: {err} "
                f"(commit missing from the cached clone? re-clone or `git fetch` it).")
    return files


def get_files(instance: Any, cache_dir: str = "data/repos",
              allow_clone: bool = False) -> dict:
    """Return {path: source} for all .py files at the instance's base commit
    (read-only, in-memory, for retrieval where nothing is edited)."""
    if instance.files is not None:
        return instance.files
    clone = ensure_repo(instance.repo, cache_dir, allow_clone=allow_clone)
    return _archive_py_files(clone, instance.base_commit)
