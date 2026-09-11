"""A checkpoint trained with this library, or a released decoder checkpoint without a
sentence-transformers configuration (ITER's `ielabgroup/ITER-Qwen3-Embedding-*`, LRAT).

The serving note `skimsearchagent_dense.json`, written by `skimsearchagent-train-retriever`,
says how the model was trained: the query instruction, the pooling, whether embeddings are
normalised, the document and query lengths, the precision. This class serves the checkpoint
exactly that way. A checkpoint without a note (a released one) is served with last-token
pooling, normalisation, and the plain Qwen3-Embedding instruction, which is what those models
were trained with; a run overrides any of it with the `DENSE_*` knobs.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from agent_search.retrievers.dense.base import SERVING_NOTE, DenseRetriever, local_snapshot, register_family

_DECODER_TYPES = ("qwen", "llama", "mistral", "gemma")
PLAIN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def read_serving_note(model_id: str) -> dict:
    """The note in the checkpoint's directory (or cached hub snapshot); {} without one."""
    try:
        d = local_snapshot(model_id)
        p = os.path.join(d, SERVING_NOTE) if d else None
        if p and os.path.exists(p):
            with open(p) as fh:
                return json.load(fh) or {}
    except Exception:  # noqa: BLE001: a bad note must not break loading
        pass
    return {}


def is_decoder_checkpoint(model_id: str) -> bool:
    """A local or cached directory whose config.json names a decoder model type and which has
    no sentence-transformers configuration."""
    d = local_snapshot(model_id)
    if not d or os.path.exists(os.path.join(d, "modules.json")):
        return False
    cfg = os.path.join(d, "config.json")
    if not os.path.exists(cfg):
        return False
    try:
        with open(cfg) as fh:
            mtype = str((json.load(fh) or {}).get("model_type", "")).lower()
    except Exception:  # noqa: BLE001
        return False
    return mtype.startswith(_DECODER_TYPES)


@register_family
class TrainedRetriever(DenseRetriever):
    """Served the way the checkpoint was trained (from its serving note)."""

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return bool(read_serving_note(model_id)) or is_decoder_checkpoint(model_id)

    def __init__(self, model: str, *args, **kwargs):
        note = read_serving_note(model)
        self.note = note
        instruction = note.get("query_instruction") or PLAIN_INSTRUCTION
        self.query_prefix = f"Instruct: {instruction}\nQuery: "
        self.pooling = str(note.get("pooling") or "last_token")
        self.normalize = bool(note.get("normalize", True))
        self.default_dtype = str(note.get("dtype") or "float32")
        self.query_max_len = int(note["query_max_len"]) if note.get("query_max_len") else None
        self.encoder_seq_length = int(note["max_seq_length"]) if note.get("max_seq_length") else None
        super().__init__(model, *args, **kwargs)

    def query_prefix_for(self) -> str:
        instr = os.environ.get("DENSE_QUERY_INSTRUCTION")
        if instr:
            return f"Instruct: {instr}\nQuery: "
        return self.query_prefix
