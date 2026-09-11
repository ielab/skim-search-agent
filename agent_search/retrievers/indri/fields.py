"""The document fields an Indri query can restrict to, and where each one's text comes from
on a `CodeUnit`. The Lucene index builder (`lucene/index_builder.py`) writes these fields
and the Indri compiler scores them.

Fields: "body" (the default field: `u.body` or `u.code`), "title" (`u.title`), "section"
(`u.section`, or its headings joined if the unit ships `sections` but no flat `section`
string), "author" and "date" (`unit.metadata`).
"""
from __future__ import annotations

from agent_search.corpus.units import CodeUnit

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


__all__ = ["FIELDS", "field_text"]
