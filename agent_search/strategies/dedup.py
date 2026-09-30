"""ITER's strategy: a search that drops documents already shown, then a whole-document read.

`search_dedup` over-fetches a pool, removes the documents listed earlier in the episode and
shows the top of what is left; `get_document` returns one document by its DocID. Two
variants, one per ranker. The dedup notice is the search tool's own manual, shown whenever it de-duplicates, so any family task pairs with it.

`iter_dense` and `iter_bm25` are the same tools without the de-duplication: the standard top-10
listing in ITER's result format. That is how the ITER paper evaluates every retriever (Sec. 5.3);
the de-duplicated setting is how it collected training trajectories. Pair with `research_tongyi`.
`iter_reranked` is `iter_dense` with the listing reordered by the run's reranker
(`retrieval.rerank_method`, `rerank_model`, `rerank_pool`; `agent_search/retrievers/reranked.py`),
the pool coming from the run's dense model.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.get_document.tool import GetDocument
from agent_search.tools.search_dedup.tool import SearchDedup

dedup_dense = register_strategy(Strategy(
    name="dedup_dense", toolset_name="dedup_dense",
    description="ITER: dense search that drops already-listed documents, then get_document",
    tools=(SearchDedup(name="search", ranking="dense"), GetDocument(name="get_document", aliases=("visit",)))))

dedup_bm25 = register_strategy(Strategy(
    name="dedup_bm25", toolset_name="dedup_bm25",
    description="ITER's loop with BM25: search that drops already-listed documents, then get_document",
    tools=(SearchDedup(name="bm25_search", ranking="bm25"), GetDocument(name="get_document", aliases=("visit",)))))

iter_dense = register_strategy(Strategy(
    name="iter_dense", toolset_name="iter_dense",
    description="ITER's evaluation tools: standard top-10 dense search in ITER's format, then get_document",
    tools=(SearchDedup(name="search", ranking="dense", dedup=False), GetDocument(name="get_document", aliases=("visit",)))))

iter_bm25 = register_strategy(Strategy(
    name="iter_bm25", toolset_name="iter_bm25",
    description="ITER's evaluation tools with BM25: standard top-10 search in ITER's format, then get_document",
    tools=(SearchDedup(name="bm25_search", ranking="bm25", dedup=False), GetDocument(name="get_document", aliases=("visit",)))))

iter_reranked = register_strategy(Strategy(
    name="iter_reranked", toolset_name="iter_reranked",
    description="ITER's evaluation tools with a reranker: the dense pool reordered by the run's reranker, top-10 in ITER's format, then get_document",
    tools=(SearchDedup(name="search", ranking="reranked", dedup=False), GetDocument(name="get_document", aliases=("visit",)))))
