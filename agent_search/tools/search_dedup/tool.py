"""`search`: ITER's search strategy, keyword/dense search with de-duplication.

Each call over-fetches a pool (`pool_k`), drops every document already surfaced by an earlier
search this episode (kept in `state.scratch["dedup_searched"]`), and shows the top `top_k` of
the rest. Documents that would have ranked in the top-k but were surfaced before are listed
under "Already-seen" so the agent can reopen them with `get_document`
(`agent_search.tools.get_document`).

`ranking="dense"` (default) queries the run's dense model; `ranking="bm25"` queries BM25;
`ranking="splade"` and `ranking="colbert"` take the run's learned engines
(`agent_search.retrievers.learned`); `ranking="reranked"` takes the run's reranked engine (a pool from `RERANK_BASE`, reordered by
`RERANK_METHOD`), so a reranker sits between the first stage and the listing.
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

    ranking: str = "dense"        # the engine kind behind the tool: dense | bm25 | reranked | splade | colbert | diffretriever
    pool_k: int = DEDUP_POOL_K
    top_k: int = DEDUP_TOPK
    # dedup=False is the standard top-k listing in ITER's result format (DocID, title, 64-token
    # snippet), with no over-fetch and no "already-seen" section: the ITER paper evaluates every
    # retriever this way (Sec. 5.3, "unfiltered, non-de-duplicated top-k rankings"); the
    # de-duplicated setting is how its training trajectories were collected.
    dedup: bool = True

    snippet: Snippet = OpeningLine()   # the text under each hit (agent_search.snippets); ITER's cut

    def __init__(self, name: Optional[str] = None, **options):
        super().__init__(name=name, **options)
        self.engines = {"bm25": ("bm25",), "reranked": ("reranked",), "splade": ("splade",),
                        "colbert": ("colbert",), "diffretriever": ("diffretriever",)}.get(self.ranking, ("dense",))
        # the name, description and parameters do not depend on the ranking: the task prompts name
        # `search`, and a cell must differ from another only in what ranks behind it
        # The dedup notice is the tool's own manual, shown only when it de-duplicates; the task
        # prompts place it through {{tool_manuals}}, so no family needs a second prompt for it.
        self.manual = "dedup_notice.md" if self.dedup else None

    def _retrieve(self, query: str, k: int):
        if self.ranking == "bm25":
            return self.engine["bm25"].search(query, k=k)
        if self.ranking in ("reranked", "splade", "colbert", "diffretriever"):
            return self.engine[self.ranking].search(query, k=k) or []
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
        if not self.dedup:
            pool = [str(d) for d in self._retrieve(query, self.top_k)]
            searched, seen_pool, hidden, results = [], set(), [], pool[: self.top_k]
        else:
            pool = [str(d) for d in self._retrieve(query, max(self.top_k, self.pool_k))]
            searched = state.scratch.setdefault("dedup_searched", [])
            seen_pool = set(searched)
            hidden = [d for d in pool[: self.top_k] if d in seen_pool]
            results = [d for d in pool if d not in seen_pool][: self.top_k]
        state.previous_queries.append(query)
        if not results:
            state.last_hits = []
            return f"No results found for '{query}'. Try with a more general query."
        if self.dedup:
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
