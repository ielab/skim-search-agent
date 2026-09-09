"""Content fingerprint for every persistent, corpus-keyed cache (dense embeddings, the
BQL structural index, pyserini's Lucene BM25 index, the lucene_structured fielded index).

Every one of those caches already guards against a corpus-identity mismatch by comparing
doc_id lists/counts (see e.g. `retrievers/dense/dense.py`'s `DenseRetriever.index` /
`bql/executor.py`'s `attach_units` / `lexical/pyserini.py`'s `_is_built`) — but a doc_id
count/order match says nothing about whether the underlying TEXT changed: editing a
function's body, a document's title, or its section headings in place (same doc_id, same
count, same order) would silently keep serving a stale cache with WRONG content. This
module's `corpus_fingerprint` closes that gap: a single hex digest over exactly the fields
each cache actually indexes, so "the corpus's content changed" becomes a cheap, deterministic
check the same shape as the existing doc-id check, not a second O(N) re-derivation of what
each backend already computes for its own artifact.

Deterministic and O(total text): one SHA-1 hasher fed incrementally, in unit order, so the
whole corpus is never held as a single concatenated string in memory.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

# Field separator: a control character that can never appear in a unit's own text (source
# code/prose), so two fields can never be mistaken for a boundary shift (e.g. unit A's
# "code" ending in "x" + unit B's "qualname" starting with "y" must hash differently from
# unit A's "code" ending in "xy" + unit B's "qualname" starting with ""). Unit separator
# (\x1f) + field separator (\x1e) are the ASCII control codes designed for exactly this.
_FIELD_SEP = b"\x1e"
_UNIT_SEP = b"\x1f"


def corpus_fingerprint(units: Iterable) -> str:
    """Hex SHA-1 over `(doc_id, qualname, code, body or "", section or "")` for every unit,
    in the given order. Same corpus (same units, same order, same content) -> same
    fingerprint; anything else (a unit's text edited in place, units reordered, a unit
    added/removed/renamed) -> a different one. Callers compare this against a value
    persisted at build time — a mismatch means "stale cache," exactly like the existing
    doc-id/count checks each backend already performs."""
    h = hashlib.sha1()
    for u in units:
        h.update(_UNIT_SEP)
        h.update((u.doc_id or "").encode("utf-8", "surrogatepass"))
        h.update(_FIELD_SEP)
        h.update((u.qualname or "").encode("utf-8", "surrogatepass"))
        h.update(_FIELD_SEP)
        h.update((u.code or "").encode("utf-8", "surrogatepass"))
        h.update(_FIELD_SEP)
        h.update((u.body or "").encode("utf-8", "surrogatepass"))
        h.update(_FIELD_SEP)
        h.update((u.section or "").encode("utf-8", "surrogatepass"))
        meta = getattr(u, "metadata", None) or {}
        if meta:
            h.update(_FIELD_SEP)
            h.update(repr(sorted((str(k), str(v)) for k, v in meta.items())).encode("utf-8", "surrogatepass"))
    return h.hexdigest()


__all__ = ["corpus_fingerprint"]
