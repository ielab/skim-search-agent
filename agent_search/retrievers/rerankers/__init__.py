"""Rerankers, one file each: `cross_encoder` (a sequence-classification model scoring
(query, document) pairs). `build_reranker("cross_encoder", model="BAAI/bge-reranker-v2-m3")`.
The composition with a retriever is `agent_search/retrievers/reranked.py`."""
from agent_search.retrievers.rerankers.base import RERANKERS, Candidate, Reranker, build_reranker, register_reranker
from agent_search.retrievers.rerankers.cross_encoder import CrossEncoderReranker

__all__ = ["Reranker", "RERANKERS", "Candidate", "build_reranker", "register_reranker", "CrossEncoderReranker"]
