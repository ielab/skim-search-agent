"""Compatibility module. The dense retrievers now live in one file per family:
`base.py` (DenseRetriever), `bge.py`, `coderank.py`, `qwen3_embedding.py`, `trained.py`, and
`belief.py` (DenseBelief). The names below keep older imports working; new code should import
from `agent_search.retrievers.dense`.
"""
from __future__ import annotations

from typing import Optional

from agent_search.retrievers.dense.base import _ENCODER_CACHE, _ENCODER_LOCK  # noqa: F401  (tests reset the cache)
from agent_search.retrievers.dense.base import (DenseRetriever, SERVING_NOTE, encode_query as _encode_query,
                                                 external_index_path, family_for, local_snapshot as _local_snapshot,
                                                 register_family, to_numpy as _to_numpy)
from agent_search.retrievers.dense.coderank import CodeRankEmbedRetriever
from agent_search.retrievers.dense.qwen3_embedding import QWEN3_EMBED_INSTRUCT as _QWEN3_EMBED_INSTRUCT
from agent_search.retrievers.dense.qwen3_embedding import Qwen3EmbeddingRetriever
from agent_search.retrievers.dense.trained import read_serving_note as serving_note

# the query prefixes of the hub families, by model id (kept for older callers; the families own them)
_QUERY_PREFIX = {
    "nomic-ai/CodeRankEmbed": CodeRankEmbedRetriever.query_prefix,
    "cornstack/CodeRankEmbed": CodeRankEmbedRetriever.query_prefix,
    "Qwen/Qwen3-Embedding-0.6B": Qwen3EmbeddingRetriever.query_prefix,
    "Qwen/Qwen3-Embedding-4B": Qwen3EmbeddingRetriever.query_prefix,
    "Qwen/Qwen3-Embedding-8B": Qwen3EmbeddingRetriever.query_prefix,
}


def _probe(model_id: str, **kw) -> DenseRetriever:
    """A retriever object for `model_id` without loading anything (the family resolves from the name
    and the files on disk)."""
    return DenseRetriever(model_id, encoder=object(), **kw)


def query_prefix_for(model_id: str) -> str:
    return _probe(model_id).query_prefix_for()


def resolve_pooling(model_id: str, note: Optional[dict] = None) -> Optional[str]:
    return _probe(model_id).resolve_pooling()


def resolve_dtype(model_id: str, note: Optional[dict] = None, device: Optional[str] = None) -> str:
    return _probe(model_id).resolve_dtype(device)


def query_seq_length(model_id: str, default: int) -> int:
    return _probe(model_id, max_seq_length=default).query_seq_length()


def _shared_encoder(model_id: str, device, max_seq_length: int, dtype: Optional[str] = None):
    return DenseRetriever(model_id, max_seq_length=max_seq_length, device=device, dtype=dtype)._shared_encoder(device)


__all__ = ["DenseRetriever", "SERVING_NOTE", "serving_note", "query_prefix_for", "resolve_pooling",
           "resolve_dtype", "query_seq_length", "external_index_path", "register_family", "family_for",
           "_QUERY_PREFIX", "_local_snapshot", "_shared_encoder", "_to_numpy", "_encode_query",
           "_ENCODER_CACHE", "_ENCODER_LOCK"]
