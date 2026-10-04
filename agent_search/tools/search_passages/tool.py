"""`search`: QUEST's corpus search, long passages and no read tool.

QUEST answers BrowseComp-Plus from search results alone (`inference/search_faiss_bridge.py` in
github.com/OSU-NLP-Group/QUEST). The call carries an array of queries; every query runs and
returns its top five documents, each with its score and the first 512 model tokens of its text:

    A search for '<query>' found 5 results:

    Title: Document <doc id>
    Link: bm25://<doc id>
    Score: 0.6123
    Snipptes: <the opening of the document>

The misspelt `Snipptes` is the official tool's own label and the model was trained on it. The
blocks of several queries are joined by a blank line.
"""
from __future__ import annotations

import os

from agent_search.snippets.opening import OpeningLine
from agent_search.tools.base import Tool

TOP_K = int(os.environ.get("SEARCH_PASSAGES_TOPK", "5"))                         # FAISS_TOP_K
PASSAGE_TOKENS = int(os.environ.get("SEARCH_PASSAGES_TOKENS", "512"))            # FAISS_SNIPPET_MAX_TOKENS


class SearchPassages(Tool):
    name = "search"
    description = ("Search a fixed document corpus and return the top results. Each result includes a "
                   "document ID (as a bm25://<docid> URL), a relevance score, and a short text snippet. "
                   "You may issue multiple queries in one call.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "array",
                                           "items": {"type": "string", "description": "The search query."},
                                           "minItems": 1, "description": "The list of search queries."}},
                  "required": ["query"]}

    def __init__(self, name=None, ranking: str = "dense", **options):
        super().__init__(name=name, **options)
        self.ranking = ranking          # the engine kind behind the tool: dense | bm25
        self.engines = (ranking,)
        self.snippet = OpeningLine()

    def _scored(self, query: str, k: int) -> list:
        """`[(doc id, score)]` best first."""
        if self.ranking == "bm25":
            return [(str(d), float(s)) for d, s in self.engine["bm25"].search_scored(query, k)]
        return [(str(d), float(s)) for d, s in self.engine["dense"].search_scored(query, k)]

    def _block(self, query: str) -> tuple[str, list]:
        rows, shown = [], []
        for doc_id, score in self._scored(query, TOP_K):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            shown.append(doc_id)
            rows.append(f"Title: Document {doc_id}\nLink: bm25://{doc_id}\nScore: {score:.4f}\n"
                        f"Snipptes: {self.snippet.render(u, None, PASSAGE_TOKENS)}")
        if not rows:
            return f"No results found for query: '{query}'. Use a less specific query.", []
        return f"A search for '{query}' found {len(rows)} results:\n\n" + "\n\n".join(rows), shown

    def run(self, args: dict) -> str:
        query = (args or {}).get("query")
        if not query:
            return "[Tool Error] Search query cannot be empty."
        queries = [query] if isinstance(query, str) else [str(q) for q in query if str(q).strip()]
        if not queries:
            return "[Tool Error] Search query cannot be empty."
        state, blocks, hits = self.state, [], []
        for q in queries:
            state.previous_queries.append(q)
            text, shown = self._block(q.strip())
            blocks.append(text)
            hits.extend(d for d in shown if d not in hits)
        state.last_hits = hits
        state.seen.update(hits)
        return "\n\n".join(blocks)


__all__ = ["SearchPassages", "TOP_K", "PASSAGE_TOKENS"]
