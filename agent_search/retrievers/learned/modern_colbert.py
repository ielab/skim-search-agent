"""A ColBERT checkpoint released in PyLate's format, served without PyLate: LightOn's
ModernBERT late-interaction models (`lightonai/Agent-ModernColBERT`,
`lightonai/Reason-ModernColBERT`, `lightonai/GTE-ModernColBERT-v1`).

The index, the exact MaxSim ranking and the knobs are `colbert.py`'s. What differs is read from
the checkpoint: the encoder is whatever `config.json` names (ModernBERT), the 768 -> 128 head is
the `1_Dense` module, the markers after `[CLS]` are the tokens `[Q] ` and `[D] `, the tokens
dropped from page vectors are `skiplist_words`, and a query is not padded with `[MASK]` when
`do_query_expansion` is false (`config_sentence_transformers.json`).

Agent-ModernColBERT was trained on AgentIR's queries (its model card): the instruction, then the
reasoning of the turn that issues the search, then the query. `query_text` writes that from the
episode's history. A checkpoint that is not an agent one gets the bare query.

`COLBERT_DOC_LENGTH` and `COLBERT_QUERY_LENGTH` set the lengths of a run; the checkpoint's own
(4096 and 8192 for Agent-ModernColBERT) are the upper limits.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from agent_search.retrievers.learned.colbert import ColbertRetriever

# The model card's prefix, byte for byte: it ends with a backslash and an `n`, not a newline.
AGENT_QUERY_PREFIX = ("Instruct: Given a user's reasoning followed by a web search query, retrieve relevant "
                      "passages that answer the query while incorporating the user's reasoning\\nQuery:")


def pylate_config(model_id: str) -> Optional[dict]:
    """The checkpoint's `config_sentence_transformers.json` when it is a PyLate ColBERT
    (its `modules.json` names `pylate.models.Dense`), else None."""
    from agent_search.retrievers.dense.base import local_snapshot
    d = local_snapshot(model_id)
    modules = os.path.join(d, "modules.json") if d else None
    if not modules or not os.path.exists(modules):
        return None
    with open(modules) as fh:
        if not any(str(m.get("type", "")).startswith("pylate.") for m in json.load(fh)):
            return None
    with open(os.path.join(d, "config_sentence_transformers.json")) as fh:
        return json.load(fh)


def agent_query(reasoning: str, query: str) -> str:
    """AgentIR's query as the model card writes it; `Empty` stands for a turn without reasoning."""
    return f"{AGENT_QUERY_PREFIX}Reasoning: {reasoning or 'Empty'}\n\nQuery: {query}"


class ModernColbertRetriever(ColbertRetriever):
    name = "colbert"

    def __init__(self, model: Optional[str] = None, **kw):
        super().__init__(model, **kw)
        self.config = pylate_config(self.model_id) or {}
        self.doc_length = min(self.doc_length, int(self.config.get("document_length") or self.doc_length))
        self.query_length = min(self.query_length, int(self.config.get("query_length") or self.query_length))
        self.agent_queries = "agent-moderncolbert" in self.model_id.lower()

    def _encoder(self):
        if self._model is None:
            import torch
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            from transformers import AutoModel, AutoTokenizer
            dev = self.torch_device()
            path = self.model_id if os.path.isdir(self.model_id) else snapshot_download(self.model_id)
            tok = AutoTokenizer.from_pretrained(path)
            body = AutoModel.from_pretrained(path).to(dev).eval()
            weight = load_file(os.path.join(path, "1_Dense", "model.safetensors"))["linear.weight"]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            linear.weight.data.copy_(weight)
            linear = linear.to(dev).eval()
            markers = tok.convert_tokens_to_ids([self.config.get("query_prefix") or "[Q] ",
                                                 self.config.get("document_prefix") or "[D] "])
            skip = set(tok.convert_tokens_to_ids(list(self.config.get("skiplist_words") or [])))
            self._model = (tok, body, linear, dev, markers[0], markers[1], skip)
        return self._model

    def query_text(self, query: str) -> str:
        """The text the query encoder reads for the agent's `query`."""
        if not self.agent_queries:
            return query
        from agent_search.training.history import CURRENT
        ctx = CURRENT.get()
        return agent_query(ctx.current_reasoning if ctx is not None else "", query)

    def query_vectors(self, query: str):
        """`[tokens, 128]`: one vector per query token, no padding."""
        import torch
        _, _, _, dev, q_marker, _, _ = self._encoder()
        ids = self._marked([self.query_text(query)], self.query_length, q_marker)[0]
        t = torch.tensor([ids], device=dev)
        return self._embed(t, torch.ones_like(t))[0]


__all__ = ["ModernColbertRetriever", "pylate_config", "agent_query", "AGENT_QUERY_PREFIX"]
