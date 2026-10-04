"""`search`: OpenSeeker's batched web search over the run's retriever.

The call carries an array of queries. Each query is searched on its own and gets its own block,
in the layout of OpenSeeker's search tool (`src/tools/search.py`, the Serper listing):

    ### A Google search for '<query>' found 10 results:

    1. [Title](https://corpus/<doc id>)

    <snippet>

Blocks are joined by a line of three dashes. A corpus document has no web address, so its
link is `https://corpus/<doc id>`, which `visit` takes back.
"""
from __future__ import annotations

import os

from agent_search.corpus.units import code_tokenize
from agent_search.snippets.term_window import TermWindow
from agent_search.tools.base import Tool

RESULTS_PER_QUERY = int(os.environ.get("SEARCH_QUERIES_TOPK", "10"))            # Serper's default page
SNIPPET_TOKENS = int(os.environ.get("SEARCH_QUERIES_SNIPPET_TOKENS", "48"))     # about a Google snippet
CORPUS_URL = "https://corpus/"


class SearchQueries(Tool):
    name = "search"
    description = ("Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 "
                   "results for each query in one call.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "array", "items": {"type": "string"},
                                           "description": "Array of query strings. Include multiple "
                                                          "complementary search queries in a single call."}},
                  "required": ["query"]}

    def __init__(self, name=None, ranking: str = "dense", **options):
        super().__init__(name=name, **options)
        self.ranking = ranking          # the engine kind behind the tool: dense | bm25
        self.engines = (ranking,)
        self.snippet = TermWindow()

    def _retrieve(self, query: str, k: int) -> list:
        if self.ranking == "bm25":
            return self.engine["bm25"].search(query, k=k)
        return self.engine["dense"].top_k_doc_ids(query, k=k) or []

    def _block(self, query: str) -> tuple[str, list]:
        ids = [str(d) for d in self._retrieve(query, RESULTS_PER_QUERY)]
        rows, shown = [], []
        for d in ids:
            u = self.ubyid.get(d)
            if u is None:
                continue
            shown.append(d)
            title = (u.title or u.qualname or f"Doc {d}").replace("\n", " ")
            snippet = self.snippet.render(u, code_tokenize(query), SNIPPET_TOKENS).replace("\n", " ")
            rows.append(f"{len(shown)}. [{title}]({CORPUS_URL}{d})\n\n{snippet}")
        if not rows:
            return f"No results found for '{query}'. Try with a more general query.", []
        return f"### A Google search for '{query}' found {len(rows)} results:\n\n" + "\n\n".join(rows), shown

    def run(self, args: dict) -> str:
        query = (args or {}).get("query")
        if query in (None, "", []):
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        queries = [query] if isinstance(query, str) else [str(q) for q in query if str(q).strip()]
        state, blocks, hits = self.state, [], []
        for q in queries:
            state.previous_queries.append(q)
            text, shown = self._block(q.strip())
            blocks.append(text)
            hits.extend(d for d in shown if d not in hits)
        state.last_hits = hits
        state.seen.update(hits)
        return "\n---\n".join(blocks)


__all__ = ["SearchQueries", "RESULTS_PER_QUERY", "SNIPPET_TOKENS", "CORPUS_URL"]
