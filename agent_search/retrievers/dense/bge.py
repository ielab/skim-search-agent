"""BAAI bge encoders (`BAAI/bge-base-en-v1.5`, the paper's document embedder, and its siblings).

A bge model ships its own sentence-transformers configuration (CLS pooling, normalisation), takes
no query prefix, and is served in float32. Nothing else is needed; the class exists so the
family is explicit and a reader can see what the paper's runs used.
"""
from __future__ import annotations

from agent_search.retrievers.dense.base import DenseRetriever, register_family


@register_family
class BgeRetriever(DenseRetriever):
    query_prefix = ""
    pooling = None            # the checkpoint's own config: CLS pooling
    default_dtype = "float32"

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return model_id.startswith("BAAI/bge-")
