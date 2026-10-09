"""E5 encoders fine-tuned for search agents.

`liuwenhan/Agentic-R_e5` (Agentic-R, arXiv 2601.11888; base `intfloat/e5-base-v2`): a query is
`query: <original question> [SEP] <agent query>` and a page is `passage: <text>` (its model
card). The run's query style `question_sep` writes the two query fields; this class adds the
prefixes.

`Yuqi-Zhou/LRAT-multilingual-e5-large` (LRAT, arXiv 2604.04949; base
`intfloat/multilingual-e5-large-instruct`): the base model's format, an instruction in front of
the query and nothing in front of a page. The instruction is the plain web-search one, the same
this library serves LRAT's Qwen3 checkpoint with.

Both checkpoints ship no sentence-transformers configuration, so sentence-transformers loads
them with mean pooling over the real tokens, which is E5's pooling. They read 512 tokens.
"""
from __future__ import annotations

from agent_search.retrievers.dense.base import DenseRetriever, register_family


@register_family
class AgenticRRetriever(DenseRetriever):
    query_prefix = "query: "
    document_template = "passage: {title}\n{body}"
    pooling = None            # sentence-transformers' default for a bare checkpoint: mean
    default_dtype = "float32"

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "agentic-r" in model_id.lower()


@register_family
class LratE5Retriever(DenseRetriever):
    query_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    pooling = None            # mean, as above
    default_dtype = "float32"

    @classmethod
    def matches(cls, model_id: str) -> bool:
        name = model_id.lower()
        return "lrat" in name and "e5" in name
