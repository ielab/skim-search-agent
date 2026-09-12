"""`search`: ITER's search strategy, keyword/dense search with de-duplication.

Each call over-fetches a pool (`pool_k`), drops every document already surfaced by an earlier
search this episode (kept in `state.scratch["dedup_searched"]`), and shows the top `top_k` of
the rest. Documents that would have ranked in the top-k but were surfaced before are listed
under "Already-seen" so the agent can reopen them with `get_document`
(`agent_search.tools.get_document`).

`ranking="dense"` (default) queries the run's dense model; `ranking="bm25"` queries BM25.
Result rendering follows ITER: ``DocID:<id>``, ``[<title>]``, then the passage cut to
`DEDUP_SNIPPET_TOKENS` model tokens (ITER's runs: 64, by the served model's tokenizer; here
the library's ruler). `snippet=` swaps the method.
"""
from __future__ import annotations

import os
from typing import Optional

from agent_search.tools.base import Tool, query_text
from agent_search.tools.budgets import DEDUP_SNIPPET_TOKENS
from agent_search.snippets import OpeningLine, Snippet

DEDUP_POOL_K = int(os.environ.get("DEDUP_POOL_K", "100"))
DEDUP_TOPK = int(os.environ.get("DEDUP_TOPK", "10"))


class SearchDedup(Tool):
    name = "search"
    description = ("Search the collection and return the top results with their DocID, title "
                   "and snippet. Documents: a keyword query. Code repositories: a Boolean "
                   "field-tagged query over code units (def, call, sig, comment, string, path, "
                   "body; AND/OR/NOT, phrases, wildcards), returning ranked files with the "
                   "matched function names.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "string",
                                           "description": "The search query (plain keywords, or a Boolean query for code)."},
                                 "k": {"type": "integer",
                                      "description": "Number of results to list (code search only; default 5)."}},
                  "required": ["query"]}

    ranking: str = "dense"        # "dense" -> engines ("dense",); "bm25" -> engines ("bm25",)
    pool_k: int = DEDUP_POOL_K
    top_k: int = DEDUP_TOPK

    # what the `bm25_search` name showed in the ITER-with-BM25 prompt (the shared BM25 text)
    BM25_DESCRIPTION = ("Keyword search over the document corpus (BM25); returns ranked documents (with "
                        "`fetch` available: each hit's title + section names + infobox keys; otherwise "
                        "title + a short opening snippet). Depending on the toolset: `visit` a ranked doc "
                        "for its full text, `fetch` a named section of a ranked doc, or (bm25_dci) "
                        "`bash`/`read` the ranked docs directly — no other documents are reachable.")
    BM25_PARAMETERS = {"type": "object",
                       "properties": {"query": {"type": "string",
                                                "description": "A keyword query, for example: treaty that ended the Mexican-American War."}},
                       "required": ["query"]}

    snippet: Snippet = OpeningLine()   # the text under each hit (agent_search.snippets); ITER's cut

    def __init__(self, name: Optional[str] = None, **options):
        super().__init__(name=name, **options)
        self.engines = ("bm25",) if self.ranking == "bm25" else ("dense",)
        if self.ranking == "bm25":
            self.description, self.parameters = self.BM25_DESCRIPTION, self.BM25_PARAMETERS

    def _retrieve(self, query: str, k: int):
        if self.ranking == "bm25":
            return self.engine["bm25"].search(query, k=k)
        return self.engine["dense"].top_k_doc_ids(query, k=k) or []

    def run(self, args: dict) -> str:
        args = args or {}
        query = query_text(args)
        if isinstance(query, list):
            query = query[0] if query else ""
        return self.search(str(query))

    def search(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        state = self.state
        pool = [str(d) for d in self._retrieve(query, max(self.top_k, self.pool_k))]
        searched = state.scratch.setdefault("dedup_searched", [])
        seen_pool = set(searched)
        hidden = [d for d in pool[: self.top_k] if d in seen_pool]
        results = [d for d in pool if d not in seen_pool][: self.top_k]
        state.previous_queries.append(query)
        if not results:
            state.last_hits = []
            return f"No results found for '{query}'. Try with a more general query."
        for d in results:
            if d not in seen_pool:
                searched.append(d)
        state.last_hits = list(results)
        state.seen.update(results)
        blocks = []
        for d in results:
            u = self.ubyid.get(d)
            if u is None:
                continue
            title = u.title or u.qualname or ""
            blocks.append(f"DocID:{d}\n[{title}]\n{self.snippet.render(u, width=DEDUP_SNIPPET_TOKENS)}")
        out = (f"A search for '{query}' found {len(blocks)} results:\n\n## Web Results\n"
               + "\n\n".join(blocks))
        if hidden:
            names = " ; ".join(f"DocID:{d} [{(self.ubyid.get(d).title if self.ubyid.get(d) else '') or ''}]"
                               for d in hidden)
            out += ("\n\n## Already-seen (relevant docs hidden from this ranking because surfaced "
                    f"earlier; use get_document to revisit any)\n{names}")
        return out


__all__ = ["SearchDedup", "DEDUP_POOL_K", "DEDUP_TOPK"]
