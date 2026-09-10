"""CodeRankEmbed (`nomic-ai/CodeRankEmbed`), the code-domain embedder.

Queries carry the prefix the model was trained with; documents get none. The model ships its
own sentence-transformers configuration and is served in float32. Documents are indexed as
`qualname\\ncode`, the same text every other condition sees, rather than bare code as in the
CoRNStack evaluation.
"""
from __future__ import annotations

from agent_search.retrievers.dense.base import DenseRetriever, register_family


@register_family
class CodeRankEmbedRetriever(DenseRetriever):
    query_prefix = "Represent this query for searching relevant code: "
    pooling = None
    default_dtype = "float32"

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return model_id.endswith("/CodeRankEmbed")
