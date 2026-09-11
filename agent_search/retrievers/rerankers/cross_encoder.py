"""A cross-encoder reranker: one forward pass per (query, document) pair, the score is the
model's relevance logit. Served through sentence-transformers' `CrossEncoder`, so any
sequence-classification reranker on the Hub works (`BAAI/bge-reranker-v2-m3`,
`BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, a local directory).

The model loads once per process and is shared across worker threads. The device follows
`AGENT_SEARCH_DENSE_DEVICE`, the same knob the dense encoders read, so a run whose GPU is
owned by a vLLM server puts the reranker on the CPU with one setting.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Sequence

from agent_search.retrievers.rerankers.base import Candidate, Reranker, register_reranker

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

_MODELS: dict = {}
_MODELS_LOCK = threading.Lock()


def _shared_model(model_id: str, device: Optional[str], max_length: int):
    key = (model_id, device or "auto", max_length)
    with _MODELS_LOCK:
        m = _MODELS.get(key)
        if m is None:
            from sentence_transformers import CrossEncoder  # heavy import, on first use
            m = CrossEncoder(model_id, max_length=max_length, device=device, trust_remote_code=True)
            _MODELS[key] = m
    return m


@register_reranker
class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32, max_length: int = 512,
                 device: Optional[str] = None):
        self.model_id = model
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
        self._model = None

    def _encoder(self):
        if self._model is None:
            self._model = _shared_model(self.model_id, self.device, self.max_length)
        return self._model

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: Optional[int] = None) -> list[tuple[str, float]]:
        cands = [(d, t or "") for d, t in candidates]
        if not cands:
            return []
        pairs = [(query, text) for _, text in cands]
        scores = self._encoder().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip((d for d, _ in cands), (float(s) for s in scores)),
                        key=lambda x: (-x[1], x[0]))
        return ranked[:k] if k is not None else ranked

    def describe(self) -> dict:
        return {"reranker": self.name, "model": self.model_id, "max_length": self.max_length}


__all__ = ["CrossEncoderReranker", "DEFAULT_MODEL"]
