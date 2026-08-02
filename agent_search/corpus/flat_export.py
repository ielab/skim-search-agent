"""Materialize a unit collection to a FLAT FILE TREE — the DCI baseline's corpus shape.

DCI (Direct Corpus Interaction, `chen2026dci`; RISE's brute-force reference arm) gives the
agent `bash` + `read` over the raw corpus FILESYSTEM, not the in-memory unit list every other
condition searches. There is no such tree lying around (the framework is index-free — units
live in memory), so this module writes ONE ``<doc_id>.txt`` per unit under a directory, then
hands that directory to `agent_search.agent.tools.doc_dci.DciWorkspace`.

This is a plain filesystem write, not a new persisted index: the tree lives under a tempdir
(or `AGENT_SEARCH_DCI_CACHE` if set) and is cheap per doc (one small text write), same order of
work as materializing a BM25/dense index for the corpus — see the module docstring's cost note.
Keyed by `key` (the run's `corpus_id`, same identity BQL/dense index caching already uses) so a
shared document corpus is exported ONCE per process and every DCI episode over it reuses the
tree; a per-instance code corpus (small, ~tens of files) is cheap enough to export per episode
and is not cached across instances.

FLAG (heavy corpora): exporting the FULL browsecomp_plus (~100k docs) or a full multi-hop corpus
writes ~100k small files once per process — a few seconds to tens of seconds, comparable to a
BM25/dense prebuild, but on a shared/networked filesystem (Lustre/NFS) many small-file writes can
be slow. Prefer a LOCAL disk or tmpfs (``AGENT_SEARCH_DCI_CACHE=/local/scratch/dci``) for a big
corpus; the default is ``tempfile.gettempdir()`` (usually local, but confirm on your cluster).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit

# a doc_id may contain characters unsafe as a bare filename (path separators, etc.); replace
# them so every doc_id maps to exactly one file, reversibly enough for the common case (plain
# alnum/._- ids, which is what every shipped corpus uses).
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]")

_CACHE_ROOT_ENV = "AGENT_SEARCH_DCI_CACHE"
# process-lifetime cache: {key: export_dir}, so a shared corpus (key = corpus_id) is written
# once and every subsequent DCI episode over it (this process) reuses the same tree instead of
# re-exporting per query — mirrors AgentRetriever building its BQL/BM25 engine once in index().
_EXPORTED: dict[str, Path] = {}


def _safe_filename(doc_id: str) -> str:
    return _UNSAFE.sub("_", doc_id) + ".txt"


def _cache_root() -> Path:
    root = os.environ.get(_CACHE_ROOT_ENV)
    return Path(root) if root else Path(tempfile.gettempdir()) / "agent_search_dci"


def stage_units_into(export_dir: Path, units: Sequence[CodeUnit]) -> dict[str, str]:
    """Write one ``<safe_doc_id>.txt`` per unit into an ALREADY-EXISTING directory; return
    {doc_id: relpath}. This is the per-file write step factored OUT of `export_flat_corpus`'s
    loop so a caller that stages docs INCREMENTALLY (`Bm25DciWorkspace`'s live per-call
    retrieval — see its module docstring) reuses the exact same `title\\n\\n{body_or_code}`
    format/naming instead of duplicating it; `export_flat_corpus` itself now just calls this
    once for its whole unit list.
    """
    doc_to_rel: dict[str, str] = {}
    for u in units:
        rel = _safe_filename(u.doc_id)
        title = (u.title or "").strip()
        text = u.body if u.body is not None else u.code
        body = f"{title}\n\n{text}" if title else (text or "")
        (export_dir / rel).write_text(body, encoding="utf-8")
        doc_to_rel[u.doc_id] = rel
    return doc_to_rel


def export_flat_corpus(units: Sequence[CodeUnit], key: Optional[str] = None,
                       rebuild: bool = False) -> tuple[Path, dict[str, str]]:
    """Write one ``<safe_doc_id>.txt`` per unit under a directory; return
    (export_dir, {doc_id: relpath}).

    Each file's body is ``title\\n\\n{body_or_code}`` when the unit carries a title (docs),
    else just ``code`` (a code-corpus unit — DCI is only wired for the doc arm, but this stays
    generic). ``key`` scopes the export dir so a shared corpus is exported once per process and
    reused (see module docstring); a code corpus (per-instance, no `key`) gets a fresh temp dir
    every call — cheap (tens of small files) and never collides across concurrent instances.
    """
    if key and not rebuild and key in _EXPORTED:
        d = _EXPORTED[key]
        if d.is_dir():
            return d, _load_map(d)

    if key:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        export_dir = _cache_root() / digest
    else:
        export_dir = Path(tempfile.mkdtemp(prefix="agent_search_dci_"))
    export_dir.mkdir(parents=True, exist_ok=True)

    marker = export_dir / ".manifest_complete"
    if marker.exists() and not rebuild:
        doc_to_rel = _load_map(export_dir)
        if key:
            _EXPORTED[key] = export_dir
        return export_dir, doc_to_rel

    doc_to_rel = stage_units_into(export_dir, units)

    _write_map(export_dir, doc_to_rel)
    if key:
        _EXPORTED[key] = export_dir
    return export_dir, doc_to_rel


def _map_path(export_dir: Path) -> Path:
    return export_dir / ".docid_map.tsv"


def _write_map(export_dir: Path, doc_to_rel: dict[str, str]) -> None:
    lines = [f"{doc_id}\t{rel}" for doc_id, rel in doc_to_rel.items()]
    _map_path(export_dir).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (export_dir / ".manifest_complete").write_text(str(len(doc_to_rel)), encoding="utf-8")


def _load_map(export_dir: Path) -> dict[str, str]:
    p = _map_path(export_dir)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            doc_id, rel = line.split("\t", 1)
            out[doc_id] = rel
    return out
