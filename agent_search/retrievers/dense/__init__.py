"""Dense retrievers: a base class and one file per encoder family.

    DenseRetriever("BAAI/bge-base-en-v1.5")            -> BgeRetriever
    DenseRetriever("nomic-ai/CodeRankEmbed")            -> CodeRankEmbedRetriever
    DenseRetriever("Qwen/Qwen3-Embedding-0.6B")         -> Qwen3EmbeddingRetriever
    DenseRetriever("models/my-checkpoint")              -> TrainedRetriever (serves the checkpoint as trained)

Each family file states its query prefix, pooling, precision and lengths. To add one, subclass
`DenseRetriever` in a new file, give it `matches(model_id)`, and decorate it with
`register_family`. `DenseBelief` wraps one retriever for the agent arms; `vector_index` holds
the persisted index backends.
"""
from agent_search.retrievers.dense.base import (DenseRetriever, FAMILIES, SERVING_NOTE, encode_query,
                                                 external_index_path, family_for, local_snapshot,
                                                 register_family, to_numpy)
from agent_search.retrievers.dense.bge import BgeRetriever
from agent_search.retrievers.dense.coderank import CodeRankEmbedRetriever
from agent_search.retrievers.dense.qwen3_embedding import QWEN3_EMBED_INSTRUCT, Qwen3EmbeddingRetriever
from agent_search.retrievers.dense.trained import TrainedRetriever, read_serving_note
from agent_search.retrievers.dense.belief import DEFAULT_TOP_K, DenseBelief

__all__ = ["DenseRetriever", "BgeRetriever", "CodeRankEmbedRetriever", "Qwen3EmbeddingRetriever",
           "TrainedRetriever", "DenseBelief", "DEFAULT_TOP_K", "FAMILIES", "SERVING_NOTE",
           "QWEN3_EMBED_INSTRUCT", "register_family", "family_for", "local_snapshot",
           "external_index_path", "encode_query", "to_numpy", "read_serving_note"]

# the `dense` retrieval-only condition: the run's dense model (`--dense-model`), no agent
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("dense")
def _build_dense(cfg: RetrieverConfig, name: str):
    return lambda: DenseRetriever(cfg.dense_model or cfg.model or "nomic-ai/CodeRankEmbed",
                                  index_root=cfg.index_root, rebuild=cfg.rebuild)
