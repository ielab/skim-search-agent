"""Qwen3-Embedding (`Qwen/Qwen3-Embedding-0.6B`, `-4B`, `-8B`), the base of ITER's retrievers.

Queries carry the instruction from the model card (`Instruct: <task>\\nQuery:` with no space after
the colon, the model's own training format and its sentence-transformers `prompts.query`);
documents get none. The hub checkpoint ships a sentence-transformers configuration with
last-token pooling, so nothing is rebuilt. Served in float32 unless a run sets
`retrieval.dense_dtype` (ITER used bfloat16).
"""
from __future__ import annotations

from agent_search.retrievers.dense.base import DenseRetriever, register_family

QWEN3_EMBED_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


@register_family
class Qwen3EmbeddingRetriever(DenseRetriever):
    query_prefix = QWEN3_EMBED_INSTRUCT
    pooling = None            # the checkpoint's own config: last-token pooling
    default_dtype = "float32"

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "Qwen3-Embedding" in model_id and model_id.startswith("Qwen/")
