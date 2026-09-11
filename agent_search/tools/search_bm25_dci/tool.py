"""`bm25_search`: the bounded bm25->DCI search-and-stage tool.

Retrieval runs on every call, over the same BM25 engine and query semantics as
`agent_search.tools.search_bm25.tool.SearchBm25`. Every hit that is not already staged is
written into the DCI staging directory (`agent_search.tools.bash.tool.get_dci_dir`,
`bounded=True`) immediately, via `agent_search.corpus.flat_export.stage_units_into`, so
`bash`/`read` (rooted at that same directory) can grep or read it. A doc that drops out of a
later ranking stays staged: once surfaced, it stays readable.

Its 0-match rendering differs from `SearchBm25`'s (no "previous results still available"
suffix), so it is its own tool rather than an option on `SearchBm25`.

`on_bind` seeds the staging directory with one live search on the episode's question, so the
first `bash`/`read` call never sees an empty corpus directory.
"""
from __future__ import annotations

import os
from typing import Optional

from agent_search.corpus.flat_export import stage_units_into
from agent_search.tools.bash.tool import get_dci_dir
from agent_search.tools.base import Tool
from agent_search.tools.common import opening_line

# The retrieval-stage cutoff: how many bm25 hits a `bm25_search` call surfaces and stages by
# default. This is a fixed constant, not a per-call k.
BM25_DCI_TOPK = int(os.environ.get("BM25_DCI_TOPK", "10"))


class SearchBm25Dci(Tool):
    name = "bm25_search"
    description = ("Keyword search over the document corpus (BM25); returns ranked documents "
                   "(with `fetch` available: each hit's title + section names + infobox keys; "
                   "otherwise title + a short opening snippet). Depending on the toolset: "
                   "`visit` a ranked doc for its full text, `fetch` a named section of a ranked "
                   "doc, or (bm25_dci) `bash`/`read` the ranked docs directly — no other "
                   "documents are reachable.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "string",
                                           "description": "A keyword query, for example: treaty that ended the Mexican-American War."}},
                  "required": ["query"]}
    engines = ("bm25",)

    topk: int = BM25_DCI_TOPK

    def on_bind(self) -> None:
        self._staged: set = set()
        self.query = ""
        get_dci_dir(self.state, self.units, self.corpus_key, bounded=True)
        self.search(self.state.question, self.topk)

    def run(self, args: dict) -> str:
        args = args or {}
        k = args.get("k")
        query = args.get("query") or args.get("q") or self.query
        return self.search(query, int(k) if k else None)

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        self.query = query
        state = self.state
        ids = list(self.engine["bm25"].search(query, k=k or self.topk))
        state.last_hits = ids
        if not ids:
            return f"search: {query}   (0 matches)"

        lines = [f"search: {query}   ({len(ids)} matches):"]
        new_units = []
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            state.seen.add(doc_id)
            if doc_id not in self._staged:
                new_units.append(u)
            snip = opening_line(u)
            lines.append(f"  {rank}  {doc_id}  {(u.title or u.qualname or '')!r}  {snip}…")

        # Only docs not already on disk get written, using the same writer `export_flat_corpus`
        # uses (`stage_units_into`). A doc surfaced by an earlier call stays staged (never
        # rewritten, never removed), so bash/read's view only ever grows.
        if new_units:
            corpus_dir = state.scratch["dci_dir"]
            rel_to_doc = state.scratch["dci_rel_to_doc"]
            new_map = stage_units_into(corpus_dir, new_units)
            rel_to_doc.update({rel: doc_id for doc_id, rel in new_map.items()})
            self._staged.update(new_map)

        return "\n".join(lines)


__all__ = ["SearchBm25Dci", "BM25_DCI_TOPK"]
