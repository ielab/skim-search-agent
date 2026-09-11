"""`get_document`: open one document by its DocID (the second tool in ITER's dedup strategies).

Accepts a bare doc_id, a ``DocID:<id>`` string (as rendered by `agent_search.tools.search_dedup`),
or, when the id is purely numeric and not itself a doc_id, a 1-based rank into the last
search's `state.last_hits`. The read is capped at `MAX_VISIT_TOKENS` (tokens, never
characters). The document is added to `state.seen` and `state.reads`.
"""
from __future__ import annotations

import re

from agent_search.tools.base import Tool
from agent_search.tools.budgets import MAX_VISIT_TOKENS
from agent_search.tools.common import _cap_tokens

_DOCID = re.compile(r"DocID:\s*(\S+)")


class GetDocument(Tool):
    name = "get_document"
    description = "Retrieve the full content of one document given its DocID from a search result."
    parameters = {"type": "object",
                  "properties": {"docid": {"type": "string",
                                           "description": "The exact DocID shown in a search result."}},
                  "required": ["docid"]}

    def run(self, args: dict) -> str:
        args = args or {}
        d = args.get("docid") or args.get("doc_id") or args.get("id") or args.get("doc") or args.get("rank")
        if isinstance(d, list):
            d = d[0] if d else ""
        return self.get_document(d)

    def get_document(self, docid) -> str:
        ref = str(docid or "").strip()
        m = _DOCID.match(ref)
        if m:
            ref = m.group(1)
        last_hits = self.state.last_hits
        if ref.isdigit() and ref not in self.ubyid and 1 <= int(ref) <= len(last_hits):
            ref = last_hits[int(ref) - 1]
        u = self.ubyid.get(ref)
        if u is None:
            return (f"ERROR: no document with id {ref!r}. Use the exact DocID from a search "
                    f"result.")
        self.state.seen.add(ref)
        self.state.reads.append(ref)
        text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                           " …(truncated — this is the whole-doc cap)")
        title = u.title or u.qualname or ""
        return f"DocID:{ref}\n[{title}]\n{text}"


__all__ = ["GetDocument"]
