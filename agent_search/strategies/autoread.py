"""AutoRead: every search returns the full text of its top hits; there is no read tool.

The same three rankers as Search-Visit. The tool descriptions are the paper's
(`bm25_read_search`, `dense_read_search`, `hybrid_read_search`).
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_dense.tool import SearchDense
from agent_search.tools.search_hybrid.tool import SearchHybrid

_BM25_READ = ("Keyword search over the document corpus (BM25); returns the FULL TEXT of the top-ranked documents "
              "directly (not a listing to visit) — there is no separate visit or fetch tool in this condition, "
              "so everything you need from a hit is already in this response.")
_DENSE_READ = ("Semantic/dense retrieval search over the document corpus (dense embeddings, cosine similarity); "
               "returns the FULL TEXT of the top-ranked documents directly (not a listing to visit) — there is "
               "no separate visit or fetch tool in this condition, so everything you need from a hit is already "
               "in this response.")
_QUERY = {"type": "object", "properties": {"query": {"type": "string",
          "description": "A keyword query, for example: treaty that ended the Mexican-American War."}}, "required": ["query"]}
_NL_QUERY = {"type": "object", "properties": {"query": {"type": "string",
             "description": "A natural-language query describing what you are looking for, for example: treaty that ended the Mexican-American War."}},
             "required": ["query"]}

autoread = register_strategy(Strategy(
    name="autoread", toolset_name="bm25_autoread",
    description="every BM25 search returns the full text of its top hits",
    tools=(SearchBm25(name="bm25_read_search", full_text=True, description=_BM25_READ, parameters=_QUERY),)))

autoread_dense = register_strategy(Strategy(
    name="autoread_dense", toolset_name="dense_autoread",
    description="every dense search returns the full text of its top hits",
    tools=(SearchDense(name="dense_read_search", full_text=True, description=_DENSE_READ, parameters=_NL_QUERY),)))

autoread_hybrid = register_strategy(Strategy(
    name="autoread_hybrid", toolset_name="hybrid_autoread",
    description="every hybrid search returns the full text of its top hits",
    tools=(SearchHybrid(name="hybrid_read_search", full_text=True),)))
