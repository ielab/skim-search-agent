"""`browser.search`: gpt-oss's browser search over the run's retriever.

The query's top results become one page of links, `【id†title†domain】 snippet`, shown through
the browser window (`agent_search/tools/browser_pages.py`). The model follows a link with
`browser.open`. As in gpt-oss's tool, the page holds `MAX_RESULTS` results whatever `topn` the
call names, and the window shows as many as fit; the rest is reached by moving the window.
"""
from __future__ import annotations

import os
import time

from agent_search.corpus.units import code_tokenize
from agent_search.snippets.term_window import TermWindow
from agent_search.tools.base import Tool, query_text
from agent_search.tools.browser_pages import BrowserError, Page, clean_text, doc_url, stack_of

MAX_RESULTS = int(os.environ.get("BROWSER_SEARCH_RESULTS", "20"))      # gpt-oss: max_search_results
SNIPPET_TOKENS = int(os.environ.get("BROWSER_SNIPPET_TOKENS", "48"))   # OpenResearcher: 180 characters


class BrowserSearch(Tool):
    name = "browser.search"
    aliases = ("search",)
    description = ("Searches for information related to a query and displays top N results. Returns a "
                   "list of search results with titles, URLs, and summaries.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "string", "description": "The search query string"},
                                 "topn": {"type": "integer", "description": "Number of results to display",
                                          "default": 10}},
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

    def run(self, args: dict) -> str:
        query = query_text(args or {})
        if isinstance(query, list):                  # the tool takes one query: the first runs
            query = query[0] if query else ""
        query = str(query or "").strip()
        if not query:
            return "Error during search for ``: empty query"
        state = self.state
        state.previous_queries.append(query)
        ids = [str(d) for d in self._retrieve(query, MAX_RESULTS)]
        rows, links, doc_ids = [], {}, {}
        for d in ids:
            u = self.ubyid.get(d)
            if u is None:
                continue
            link = str(len(links))
            links[link], doc_ids[link] = doc_url(d), d
            title = clean_text(u.title or u.qualname or f"Doc {d}").replace("†", "‡").replace("\n", " ")
            summary = clean_text(self.snippet.render(u, code_tokenize(query), SNIPPET_TOKENS)).replace("\n", " ")
            rows.append(f"  * 【{link}†{title}†corpus】 {summary}")
        if not rows:
            state.last_hits = []
            return f"Error during search for `{query}`: No results returned for any query: ['{query}']"
        url = f"web-search://ts={int(time.time())}"
        page = Page(url=url, title=query, text=f"\nURL: {url}\n# Search Results\n\n" + "\n".join(rows),
                    links=links, doc_ids=doc_ids, is_listing=True)
        try:
            text, in_view = stack_of(state).show(page, loc=0)
        except BrowserError as e:
            return str(e)
        shown = [doc_ids[i] for i in in_view if i in doc_ids]
        state.last_hits = shown
        state.seen.update(shown)
        return text


__all__ = ["BrowserSearch", "MAX_RESULTS", "SNIPPET_TOKENS"]
