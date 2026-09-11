"""Retrieval-only floors: rank once with one engine, no agent, no model.

These are strategies with no loop; each names the registered retriever that does the ranking.
"""
from agent_search.strategies.base import Strategy, register_strategy

bm25 = register_strategy(Strategy(name="bm25", description="rank once with Lucene BM25 (Pyserini)", loop=False, retriever="bm25_pyserini"))
dense = register_strategy(Strategy(name="dense", description="rank once with the run's dense model", loop=False, retriever="dense"))
bql = register_strategy(Strategy(name="bql", description="rank once with a BQL query", loop=False, retriever="bql"))
grep = register_strategy(Strategy(name="grep", description="rank once with a regex over the units", loop=False, retriever="grep"))
reranked = register_strategy(Strategy(name="reranked", description="rank once with a retriever, then a reranker over its pool (RERANK_BASE, RERANK_METHOD, RERANK_MODEL, RERANK_POOL)", loop=False, retriever="reranked"))
hybrid = register_strategy(Strategy(name="hybrid", description="rank once with the fused hybrid engine (HYBRID_RETRIEVERS, HYBRID_FUSION)", loop=False, retriever="hybrid"))
