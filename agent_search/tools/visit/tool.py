"""`visit`: read the full text of one document, by its rank in the last listing or its id.

The read is capped at `MAX_VISIT_TOKENS` (tokens, never characters). A rank refers to the last
search's listing; an id may be a document id or an unambiguous title fragment. The document is
added to the surfaced set and to the episode's reads.
"""
from __future__ import annotations

from agent_search.tools.budgets import MAX_VISIT_TOKENS
from agent_search.tools.common import _cap_tokens
from agent_search.tools.base import Tool


def resolve_reference(state, ubyid, ref):
    """(doc_id, None) for a rank in the last listing, a document id, or a unique title fragment;
    (None, error text) otherwise."""
    last = state.last_hits
    if isinstance(ref, (int, float)) or (isinstance(ref, str) and ref.strip().isdigit()):
        rank = int(ref)
        if 1 <= rank <= len(last):
            return last[rank - 1], None
        if str(ref).strip() in ubyid:                 # a real doc_id that looks numeric
            return str(ref).strip(), None
        if not last:
            return None, f"ERROR: no prior search — rank {rank} has nothing to refer to."
        return None, f"ERROR: rank {rank} out of range (last search had {len(last)} results)."
    ref = (ref or "").strip()
    if ref in ubyid:
        return ref, None
    low = ref.lower()
    cands = [i for i, u in ubyid.items() if low in (u.title or u.qualname or "").lower()]
    if len(cands) == 1:
        return cands[0], None
    if cands:
        opts = ", ".join(f"{i} ({ubyid[i].title or ubyid[i].qualname})" for i in cands[:5])
        return None, f"ERROR: ambiguous doc {ref!r} — did you mean: {opts}"
    return None, f"ERROR: no such doc {ref!r} — use a rank from the last search or a doc_id."


class Visit(Tool):
    name = "visit"
    description = "Read a whole document that a previous search ranked (the retrieve-then-visit baseline). Returns the full doc text, capped."
    parameters = {"type": "object", "properties": {"rank": {"type": "integer", "description": "The 1-based rank from the last search, or a doc id."}}, "required": ["rank"]}

    aliases = ("visit", "visit_q", "visit_d", "visit_h", "visit_v", "visit_bv", "visit_bqld", "visit_bqldo")

    def run(self, args: dict) -> str:
        ref = args.get("rank") or args.get("id") or args.get("doc") or args.get("doc_id")
        doc_id, err = resolve_reference(self.state, self.ubyid, ref)
        if err:
            return err
        self.state.seen.add(doc_id)
        self.state.reads.append(doc_id)
        u = self.ubyid[doc_id]
        text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS, " …(truncated — this is the whole-doc cap)")
        return f"{doc_id}  {(u.title or u.qualname or '')!r}:\n{text}"


__all__ = ["Visit", "resolve_reference"]
