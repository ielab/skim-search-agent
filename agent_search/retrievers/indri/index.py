"""Indri backend index: postings / doc lengths / collection frequencies, per field.

Persisted structure only (no `CodeUnit` objects, no per-doc token positions — see
`model.py`'s module docstring for the persistence rationale, same as the BQL
executor's slim-pickle pattern). Positions used by proximity operators (`#odN`,
`#uwN`) are derived lazily, per candidate, from the attached live units at query
time and are never stored here.

Fields: "body" (the default/unscoped field: `u.body` or `u.code`), "title"
(`u.title`), "section" (`u.section`, or its headings joined if the unit ships
`sections` but no flat `section` string), "author" and "date" (`unit.metadata`).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize

FIELDS = ("body", "title", "section", "author", "date")


def field_text(u: CodeUnit, field: str) -> str:
    """The raw text of one document field, per the mapping in the module docstring."""
    if field == "body":
        t = u.body if u.body is not None else u.code
        return t or ""
    if field == "title":
        return u.title or ""
    if field == "section":
        if u.section is not None:
            return u.section
        if u.sections:
            return " ".join(h for h, _ in u.sections if h)
        return ""
    meta = u.metadata or {}
    if field == "author":
        return str(meta.get("author") or "")
    if field == "date":
        return str(meta.get("date") or "")
    return ""


class IndriIndex:
    """Slim, fully-picklable postings index. `doc_ids[i]` <-> unit index `i`
    (validated against the live corpus by `IndriExecutor.attach_units`, same
    "order must match" contract as the BQL executor's `_doc_id_order`)."""

    def __init__(self) -> None:
        self.doc_ids: list = []
        # field -> term -> {doc_idx: tf}
        self.postings: dict = {f: {} for f in FIELDS}
        # field -> [doc_idx] -> token length
        self.doc_len: dict = {f: [] for f in FIELDS}
        # field -> term -> collection frequency (sum of tf across all docs)
        self.cf: dict = {f: {} for f in FIELDS}
        # field -> total collection tokens
        self.total: dict = {f: 0 for f in FIELDS}

    # --- build ---------------------------------------------------------------

    @classmethod
    def build(cls, units: Sequence[CodeUnit]) -> "IndriIndex":
        idx = cls()
        idx.doc_ids = [u.doc_id for u in units]
        n = len(units)
        for f in FIELDS:
            idx.doc_len[f] = [0] * n
        for i, u in enumerate(units):
            for f in FIELDS:
                toks = code_tokenize(field_text(u, f))
                idx.doc_len[f][i] = len(toks)
                if not toks:
                    continue
                counts = Counter(toks)
                post = idx.postings[f]
                cf = idx.cf[f]
                for term, c in counts.items():
                    post.setdefault(term, {})[i] = c
                    cf[term] = cf.get(term, 0) + c
                idx.total[f] += len(toks)
        return idx

    # --- lookups ---------------------------------------------------------------

    def tf(self, field: str, term: str, doc_idx: int) -> int:
        return self.postings.get(field, {}).get(term, {}).get(doc_idx, 0)

    def doclen(self, field: str, doc_idx: int) -> int:
        lens = self.doc_len.get(field)
        if lens is None or doc_idx >= len(lens):
            return 0
        return lens[doc_idx]

    def cf_of(self, field: str, term: str) -> int:
        return self.cf.get(field, {}).get(term, 0)

    def total_of(self, field: str) -> int:
        return self.total.get(field, 0)

    def doc_ids_for_term(self, field: str, term: str) -> set:
        return set(self.postings.get(field, {}).get(term, {}).keys())

    def doc_ids_for_prefix(self, field: str, prefix: str) -> set:
        out: set = set()
        post = self.postings.get(field, {})
        for term, docs in post.items():
            if term.startswith(prefix):
                out.update(docs.keys())
        return out

    def terms_for_prefix(self, field: str, prefix: str) -> list:
        return [t for t in self.postings.get(field, {}) if t.startswith(prefix)]

    def vocab(self, field: str) -> Iterable:
        return self.postings.get(field, {}).keys()
