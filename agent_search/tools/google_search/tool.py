"""`google_search`: MiroThinker's search tool over the run's retriever.

MiroFlow's search server (`search_and_scrape_webpage`, a Serper client) returns one JSON string
with the organic results and the search parameters. This tool returns the same shape for a
corpus: each result has `title`, `link`, `snippet` and `position`. A corpus document has no
web address, so its link is `https://corpus/<doc id>`, which `scrape_and_extract_info` takes
back. `num` and `page` work as in the web tool: `num` results per page (10 by default), `page`
counts from 1. The region, language and date arguments have no meaning over a corpus and are
ignored.
"""
from __future__ import annotations

import json
import os

from agent_search.corpus.units import code_tokenize
from agent_search.snippets.term_window import TermWindow
from agent_search.tools.base import Tool

SNIPPET_TOKENS = int(os.environ.get("GOOGLE_SEARCH_SNIPPET_TOKENS", "48"))      # about a Google snippet
CORPUS_URL = "https://corpus/"


class GoogleSearch(Tool):
    name = "google_search"
    aliases = ("search",)
    description = "Tool to perform web searches and retrieve the organic results."
    parameters = {"type": "object",
                  "properties": {"q": {"title": "Q", "type": "string"},
                                 "num": {"default": None, "title": "Num", "type": "integer"},
                                 "page": {"default": None, "title": "Page", "type": "integer"}},
                  "required": ["q"]}

    def __init__(self, name=None, ranking: str = "dense", **options):
        super().__init__(name=name, **options)
        self.ranking = ranking          # the engine kind behind the tool: dense | bm25
        self.engines = (ranking,)
        self.snippet = TermWindow()

    def _retrieve(self, query: str, k: int) -> list:
        if self.ranking == "bm25":
            return self.engine["bm25"].search(query, k=k)
        return self.engine["dense"].top_k_doc_ids(query, k=k) or []

    @staticmethod
    def _count(value, default: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    def run(self, args: dict) -> str:
        args = args or {}
        query = str(args.get("q") or args.get("query") or "").strip()
        if not query:
            return json.dumps({"success": False, "error": "Search query 'q' is required and cannot be empty",
                               "results": []}, ensure_ascii=False)
        num, page = self._count(args.get("num"), 10), self._count(args.get("page"), 1)
        state = self.state
        state.previous_queries.append(query)
        ids = [str(d) for d in self._retrieve(query, num * page)][num * (page - 1):]
        organic, shown = [], []
        for d in ids:
            u = self.ubyid.get(d)
            if u is None:
                continue
            shown.append(d)
            organic.append({"title": u.title or u.qualname or f"Doc {d}", "link": f"{CORPUS_URL}{d}",
                            "snippet": self.snippet.render(u, code_tokenize(query), SNIPPET_TOKENS).replace("\n", " "),
                            "position": num * (page - 1) + len(shown)})
        state.last_hits = shown
        state.seen.update(shown)
        return json.dumps({"organic": organic,
                           "searchParameters": {"q": query, "type": "search", "num": num, "page": page,
                                                "engine": "google"}}, ensure_ascii=False)


__all__ = ["GoogleSearch", "SNIPPET_TOKENS", "CORPUS_URL"]
