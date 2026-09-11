"""What an Indri search returns, and the field-name check both the engine and the tool use.

The Indri query language is answered by the Lucene structured engine
(`agent_search/retrievers/lucene/engine.py`, `search_indri`) through `LuceneIndriAdapter`
(`lucene/adapters.py`). This module holds the result shape the `search_indri` tool reads and
the walk that spots a `.field` restriction naming a field the index does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

from agent_search.retrievers.indri.fields import FIELDS
from agent_search.retrievers.indri.parser import (
    Band, Combine, FieldExpr, FilReq, FilRej, Max, Not, Or, Syn, Weight, Window, WSyn,
)


@dataclass
class IndriResult:
    hits: list                                  # [(doc_id, score), ...] best-first
    error: Optional[str] = None
    diagnostics: list = dc_field(default_factory=list)   # [(child_repr, log_belief), ...]
    # An unrecognized `.field` name is not a parse or type error in Indri QL: it is a valid
    # restriction to a field with no postings, so the search runs and returns fewer or zero
    # hits. `warning` tells the agent that this happened (the tool appends it to the
    # rendered result) without changing the hits.
    warning: Optional[str] = None


_KNOWN_FIELDS = frozenset(f.lower() for f in FIELDS)


def unknown_query_fields(expr) -> list:
    """Every field name referenced by a `.field` / `.field.(context)` restriction anywhere in
    `expr` that is not one of `FIELDS` (body/title/section/author/date), in first-seen order.
    Diagnostic only: an unknown field name is a valid query, so this never raises."""
    seen: list = []

    def _note(name: Optional[str]) -> None:
        if name and name.lower() not in _KNOWN_FIELDS and name not in seen:
            seen.append(name)

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, FieldExpr):
            for f in (node.fields or ()):
                _note(f)
            _note(node.context)
            walk(node.child)
        elif isinstance(node, Not):
            walk(node.child)
        elif isinstance(node, (Window, Syn, Combine, Or, Max, Band)):
            for c in node.children:
                walk(c)
        elif isinstance(node, (Weight, WSyn)):
            for _, c in node.pairs:
                walk(c)
        elif isinstance(node, FilReq) or isinstance(node, FilRej):
            walk(node.a)
            walk(node.q)
        # Term/Wildcard/DateBefore/DateAfter/DateBetween: no children, no fields.

    walk(expr)
    return seen


def field_warning(unknown: list) -> Optional[str]:
    if not unknown:
        return None
    names = ", ".join(repr(f) for f in unknown)
    return (f"warning: unrecognized field {names} (known: "
            f"{', '.join(sorted(_KNOWN_FIELDS))}) -- this restricts to a field with "
            f"no postings, so it contributes nothing; check for a typo")


__all__ = ["IndriResult", "unknown_query_fields", "field_warning"]
