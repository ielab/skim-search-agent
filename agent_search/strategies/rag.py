"""One-shot RAG strategies: rank once, put the top documents in one prompt, one model call.

No loop and no tools; the harness is `agent_search.harness.rag.OneShotRag`. One strategy
per ranker: BM25, the dense model, the hybrid engine.
"""
from agent_search.harness.rag import OneShotRag
from agent_search.strategies.base import Strategy, register_strategy

rag = register_strategy(Strategy(
    name="rag", description="one-shot RAG: top-5 by BM25 in one prompt, one model call",
    harness=OneShotRag("bm25")))

rag_dense = register_strategy(Strategy(
    name="rag_dense", description="one-shot RAG: top-5 by the dense model in one prompt, one model call",
    harness=OneShotRag("dense")))

rag_hybrid = register_strategy(Strategy(
    name="rag_hybrid", description="one-shot RAG: top-5 by RRF of BM25 and dense in one prompt, one model call",
    harness=OneShotRag("hybrid")))


__all__ = ["rag", "rag_dense", "rag_hybrid"]
