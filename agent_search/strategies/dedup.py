"""ITER's strategy: a search that drops documents already shown, then a whole-document read.

`search_dedup` over-fetches a pool, removes the documents listed earlier in the episode and
shows the top of what is left; `get_document` returns one document by its DocID. Two
variants, one per ranker. Pair with the `research_dedup` task for ITER's prompt.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.get_document.tool import GetDocument
from agent_search.tools.search_dedup.tool import SearchDedup

dedup_dense = register_strategy(Strategy(
    name="dedup_dense", toolset_name="dedup_dense",
    description="ITER: dense search that drops already-listed documents, then get_document",
    tools=(SearchDedup(name="search", ranking="dense"), GetDocument(name="get_document"))))

dedup_bm25 = register_strategy(Strategy(
    name="dedup_bm25", toolset_name="dedup_bm25",
    description="ITER's loop with BM25: search that drops already-listed documents, then get_document",
    tools=(SearchDedup(name="bm25_search", ranking="bm25"), GetDocument(name="get_document"))))
