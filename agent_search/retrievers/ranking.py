"""Compatibility module: the BM25 scorer lives in `agent_search.retrievers.lexical.scorer`."""
from agent_search.corpus.units import code_tokenize  # noqa: F401  (older callers imported it from here)
from agent_search.retrievers.lexical.scorer import BM25  # noqa: F401

__all__ = ["BM25", "code_tokenize"]
