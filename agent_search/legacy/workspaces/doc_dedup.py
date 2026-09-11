"""The pre-0.3 version of ITER's search strategy: `search` with de-duplication, `get_document` by id.

Kept so the parity tests can compare against it. The current equivalent is the `dedup_bm25`/
`dedup_dense` strategies in `agent_search/strategies/dedup.py`, using the `search_dedup` and
`get_document` tools.

The agent issues keyword searches over a chunked collection and opens documents by id. Each
search over-fetches a pool (`DEDUP_POOL_K`), drops every document already surfaced by an earlier
search in this episode, and shows the top `DEDUP_TOPK` of the rest. Documents that would have
ranked in the top-k but were surfaced before are listed under "Already-seen" so the agent can
reopen them with `get_document`. This is the tool behaviour ITER's trajectories were generated
with; the retriever behind `search` is either BM25 (`dedup_bm25`) or the run's dense model
(`dedup_dense`, which also applies the history-conditioned query style when one is set).

Result rendering follows ITER: ``DocID:<id>``, ``[<title>]``, then the opening `SNIPPET_TOKENS`
tokens of the document (ITER used 64).
"""
from __future__ import annotations

import os
import re
from typing import Callable, Optional, Sequence

from agent_search.legacy.retriever import WorkspaceContext, register_workspace
from agent_search.legacy.workspaces.budgets import MAX_VISIT_TOKENS, SNIPPET_TOKENS
from agent_search.legacy.workspaces.common import opening_line
from agent_search.core.seen import OrderedSeen
from agent_search.core.tokens import cap_tokens as _cap_tokens
from agent_search.corpus.units import CodeUnit

DEDUP_POOL_K = int(os.environ.get("DEDUP_POOL_K", "100"))
DEDUP_TOPK = int(os.environ.get("DEDUP_TOPK", "10"))

_DOCID = re.compile(r"DocID:\s*(\S+)")


class DedupSearchWorkspace:
    """`search(query)` with over-fetch and drop-seen; `get_document(docid)` returns the text."""

    tools = ("search", "get_document")

    def __init__(self, units: Sequence[CodeUnit], search_fn: Callable[[str, int], Sequence[str]],
                 ubyid: Optional[dict] = None, *, pool_k: int = DEDUP_POOL_K, top_k: int = DEDUP_TOPK,
                 tool_names: Optional[tuple] = None):
        self.units = units
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in units}
        self._search = search_fn
        self.pool_k, self.top_k = pool_k, top_k
        self.seen = OrderedSeen()
        self.searched: list[str] = []          # ids surfaced by earlier searches (dedup pool)
        self.last_hits: list[str] = []
        self.previous_queries: list[str] = []
        if tool_names:
            self.tools = tuple(tool_names)

    @property
    def surfaced(self):
        return list(self.seen)

    # --- tools ---------------------------------------------------------------------
    def search(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        pool = [str(d) for d in self._search(query, max(self.top_k, self.pool_k))]
        seen = set(self.searched)
        hidden = [d for d in pool[: self.top_k] if d in seen]
        results = [d for d in pool if d not in seen][: self.top_k]
        self.previous_queries.append(query)
        if not results:
            self.last_hits = []
            return f"No results found for '{query}'. Try with a more general query."
        for d in results:
            if d not in seen:
                self.searched.append(d)
        self.last_hits = list(results)
        self.seen.update(results)
        blocks = []
        for d in results:
            u = self.ubyid.get(d)
            if u is None:
                continue
            title = u.title or u.qualname or ""
            blocks.append(f"DocID:{d}\n[{title}]\n{opening_line(u, SNIPPET_TOKENS)}")
        out = (f"A search for '{query}' found {len(blocks)} results:\n\n## Web Results\n"
               + "\n\n".join(blocks))
        if hidden:
            names = " ; ".join(f"DocID:{d} [{(self.ubyid.get(d).title if self.ubyid.get(d) else '') or ''}]"
                               for d in hidden)
            out += ("\n\n## Already-seen (relevant docs hidden from this ranking because surfaced "
                    f"earlier; use get_document to revisit any)\n{names}")
        return out

    def get_document(self, docid) -> str:
        ref = str(docid or "").strip()
        m = _DOCID.match(ref)
        if m:
            ref = m.group(1)
        if ref.isdigit() and ref not in self.ubyid and 1 <= int(ref) <= len(self.last_hits):
            ref = self.last_hits[int(ref) - 1]
        u = self.ubyid.get(ref)
        if u is None:
            return (f"ERROR: no document with id {ref!r}. Use the exact DocID from a search "
                    f"result.")
        self.seen.add(ref)
        text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                           " …(truncated — this is the whole-doc cap)")
        title = u.title or u.qualname or ""
        return f"DocID:{ref}\n[{title}]\n{text}"

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("search", "bm25_search", "dense_search"):
                q = args.get("query") or args.get("q") or ""
                if isinstance(q, list):
                    q = q[0] if q else ""
                return self.search(str(q))
            if name in ("get_document", "visit"):
                d = args.get("docid") or args.get("doc_id") or args.get("id") or args.get("doc") or args.get("rank")
                if isinstance(d, list):
                    d = d[0] if d else ""
                return self.get_document(d)
        except Exception as e:  # noqa: BLE001 — a tool error is an observation, never a crash
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


def _bm25_builder(ctx: WorkspaceContext):
    bm = ctx.bm25()
    return DedupSearchWorkspace(ctx.units, lambda q, k: bm.search(q, k=k), ubyid=ctx.ubyid,
                                tool_names=("bm25_search", "get_document"))


def _dense_builder(ctx: WorkspaceContext):
    dense = ctx.dense()
    return DedupSearchWorkspace(ctx.units, lambda q, k: dense.top_k_doc_ids(q, k=k), ubyid=ctx.ubyid)


register_workspace("dedup_bm25", tools=("bm25_search", "get_document"), builder=_bm25_builder)
register_workspace("dedup_dense", tools=("search", "get_document"), builder=_dense_builder)

__all__ = ["DedupSearchWorkspace", "DEDUP_POOL_K", "DEDUP_TOPK"]
