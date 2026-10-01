"""RepLLaMA (Ma et al. 2023, "Fine-Tuning LLaMA for Multi-Stage Text Retrieval"): Llama-2-7B fine-tuned
with LoRA on MS MARCO passage as a single-vector dense retriever, `castorini/repllama-v1-7b-lora-passage`.

Served as its model card serves it: a query is `query: {query}</s>`, a page `passage: {title} {text}</s>`,
the embedding is the hidden state at that end token, L2-normalised, and the score is their dot
product. Llama-2's tokenizer starts a text with `<s>` but does not end it with `</s>`, and a cut
long page would lose a literal one, so the encoder truncates first and appends the end token
(`append_eos`). The checkpoint runs in fp16.

The released adapter needs `peft` to merge into the base model, which the library does not carry;
`scripts/merge_lora.py` merges it once into a plain checkpoint (run it in an environment with peft),
and a run points `retrieval.dense_model` at that directory. Any model id or path containing
`repllama` is served this way.
"""
from __future__ import annotations

from agent_search.retrievers.dense.base import DenseRetriever, register_family


@register_family
class RepLlamaRetriever(DenseRetriever):
    query_prefix = "query: "
    document_template = "passage: {title} {body}"
    pooling = "last_token"
    normalize = True
    default_dtype = "float16"
    append_eos = True

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "repllama" in model_id.lower()


__all__ = ["RepLlamaRetriever"]
