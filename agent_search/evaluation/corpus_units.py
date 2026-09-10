"""Corpus units for one instance: build them, or reuse a cached build.

Covers the code arm (parse a repo's files into `CodeUnit`s, with a disk
cache keyed by repo@commit) and the document arm (wrap a fixed corpus's
docs as units). `_units_for_instance` is the single entry point `scoring.py`
calls; it picks inline files, a shared docstore, a shared doc list, or the
disk cache, depending on what the instance carries.
"""
from __future__ import annotations

import os
import pickle
import re
import threading
from typing import Optional, Sequence

from agent_search.corpus.code_repo import get_files
from agent_search.corpus.units import (
    CodeUnit,
    is_test_path,
    units_from_documents,
    units_from_python_source,
)

from .datasets import Instance


def _build_corpus(files: dict) -> tuple[list[CodeUnit], dict]:
    units: list[CodeUnit] = []
    by_file: dict = {}
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        if is_test_path(path):       # SWE-bench/CoRNStack convention, all conditions
            continue
        file_units = units_from_python_source(path, source)
        units.extend(file_units)
        by_file[path] = file_units
    return units, by_file


def _build_document_corpus(docs: list[dict]) -> list[CodeUnit]:
    return units_from_documents(docs)


def _to_file_ranking(doc_ids: Sequence[str]) -> list[str]:
    """Collapse unit doc_ids ("path::qual") to a deduped file ranking."""
    seen, out = set(), []
    for d in doc_ids:
        path = d.split("::", 1)[0]
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _corpus_key(inst: Instance) -> str:
    """Stable identity of a corpus = repo @ base commit. Instances sharing this
    reuse the same persisted index."""
    if inst.corpus_id:
        return inst.corpus_id.replace("/", "__")
    return f"{inst.repo.replace('/', '__')}@{inst.base_commit[:12]}"


# Bump when chunking/filtering changes (is_test exclusion, decorator spans, ...)
# so stale disk caches are never silently reused.
_UNITS_CACHE_VERSION = "v2"


def _units_disk_path(inst: Instance, cache_dir: str) -> str:
    """Visible sibling of the repo cache (default: data/units_cache/), one pickle
    per repo@commit — the build-once parsed-corpus artifact, NOT a retrieval
    index (the method stays index-free; this caches AST chunking only)."""
    key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", f"{inst.repo}@{inst.base_commit}")
    parent = os.path.dirname(os.path.normpath(cache_dir)) or "."
    return os.path.join(parent, "units_cache", f"{key}-{_UNITS_CACHE_VERSION}.pkl")


def _by_file(units: Sequence[CodeUnit]) -> dict:
    by_file: dict = {}
    for u in units:
        by_file.setdefault(u.path, []).append(u)
    return by_file


def _units_for_instance(inst: Instance, cache_dir: str, allow_clone: bool,
                        units_cache: dict, units_lock: threading.Lock
                        ) -> tuple[list[CodeUnit], dict, Optional[dict]]:
    """Build corpus units, reusing shared fixed-corpus document chunks across queries.

    Returns (units, by_file, files) where `files` is the raw {path -> source} dict the
    multi-tool localization agent reads — or None when units came from the disk cache
    (which stores only the AST chunking, not the raw files)."""
    if inst.docstore is not None:
        # an on-disk corpus: one lazy view per corpus key, shared by every query
        from agent_search.corpus.docstore import LazyUnits
        key = _corpus_key(inst)
        with units_lock:
            cached = units_cache.get(key)
            if cached is None:
                cached = (LazyUnits(inst.docstore), {})
                units_cache[key] = cached
            return cached[0], cached[1], {}

    if inst.docs is not None:
        key = _corpus_key(inst)
        with units_lock:
            cached = units_cache.get(key)
            if cached is None:
                cached = (_build_document_corpus(inst.docs), {})
                units_cache[key] = cached
            return cached[0], cached[1], {}

    if inst.files is not None:                   # inline fixture: cheap, no disk cache
        files = get_files(inst, cache_dir, allow_clone=allow_clone)
        units, by_file = _build_corpus(files)
        return units, by_file, files

    # Disk cache per repo@commit: git-archive + AST-chunking a whole repo is the
    # per-instance bottleneck, repeated across the 4 agent conditions and every
    # resume. Best-effort: corrupt/missing cache just rebuilds.
    path = _units_disk_path(inst, cache_dir)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                units = pickle.load(fh)
            return units, _by_file(units), None   # raw files not kept in the units cache
        except Exception:
            pass
    files = get_files(inst, cache_dir, allow_clone=allow_clone)
    units, by_file = _build_corpus(files)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "wb") as fh:
            pickle.dump(units, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                    # atomic; concurrent writers race harmlessly
    except Exception:
        pass
    return units, by_file, files
