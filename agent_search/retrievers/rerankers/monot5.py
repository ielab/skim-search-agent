"""monoT5 (Nogueira et al. 2020): a T5 model fine-tuned on MS MARCO to answer `true` or `false` for
`Query: {query} Document: {document} Relevant:`. The score is the log-probability of `true` after a
softmax over the `true` and `false` logits at the first decoder step, as pygaggle scores it. The
default is `castorini/monot5-3b-msmarco-10k`; any monoT5 checkpoint loads here.

The pair is at most 512 tokens, the length monoT5 was trained at, built from token ids with the
document cut to what fits, so the query and the `Relevant:` cue are never truncated away. T5 runs in
bfloat16 on a GPU (fp16 overflows in T5's feed-forward layers). The model loads once per process
and is shared across worker threads behind one lock.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Sequence

from agent_search.retrievers.dense.base import encoder_prefix
from agent_search.retrievers.rerankers.base import Candidate, Reranker, register_reranker

DEFAULT_MODEL = "castorini/monot5-3b-msmarco-10k"

_MODELS: dict = {}
_MODELS_LOCK = threading.Lock()


def _shared_model(model_id: str, device: Optional[str]):
    key = (model_id, device or "auto")
    with _MODELS_LOCK:
        m = _MODELS.get(key)
        if m is None:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if dev != "cpu" else torch.float32
            tok = AutoTokenizer.from_pretrained(model_id)
            model = T5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(dev).eval()
            vocab = tok.get_vocab()
            m = (tok, model, vocab["▁false"], vocab["▁true"], dev, threading.Lock())
            _MODELS[key] = m
    return m


@register_reranker
class MonoT5Reranker(Reranker):
    name = "monot5"
    default_model = DEFAULT_MODEL
    default_max_length = 512

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 16, max_length: int = 512,
                 device: Optional[str] = None):
        self.model_id = model
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
        self._loaded = None

    def _model(self):
        if self._loaded is None:
            self._loaded = _shared_model(self.model_id, self.device)
        return self._loaded

    def prompt(self, query: str, document: str) -> str:
        return f"Query: {query} Document: {document} Relevant:"

    def _inputs(self, tok, query: str, texts: Sequence[str]) -> list[list[int]]:
        """Token ids per pair: `Query: {query} Document:`, the page cut to what is left of
        `max_length`, ` Relevant:`, the end token. Built from ids, so the cue the model answers is
        never truncated away (truncating the formatted string, as pygaggle does, drops it on any
        page longer than the room) and no page is decoded and re-tokenised."""
        head = tok(f"Query: {query} Document:", add_special_tokens=False)["input_ids"]
        tail = tok(" Relevant:", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        room = max(self.max_length - len(head) - len(tail), 16)
        out = []
        for t in texts:
            doc = tok(encoder_prefix(t or "", tok, room), add_special_tokens=False, truncation=True,
                      max_length=room)["input_ids"]
            out.append(head + doc + tail)
        return out

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        import torch
        tok, model, false_id, true_id, dev, lock = self._model()
        with lock:
            inputs = self._inputs(tok, query, texts)
            out: list[float] = []
            with torch.no_grad():
                for i in range(0, len(inputs), self.batch_size):
                    enc = tok.pad({"input_ids": inputs[i:i + self.batch_size]}, padding=True,
                                  return_tensors="pt").to(dev)
                    start = torch.full((enc["input_ids"].shape[0], 1), model.config.decoder_start_token_id,
                                       dtype=torch.long, device=dev)
                    logits = model(**enc, decoder_input_ids=start).logits[:, 0, [false_id, true_id]].float()
                    out.extend(torch.log_softmax(logits, dim=-1)[:, 1].tolist())
        return out

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: Optional[int] = None) -> list[tuple[str, float]]:
        cands = [(d, t or "") for d, t in candidates]
        if not cands:
            return []
        s = self.scores(query, [t for _, t in cands])
        ranked = sorted(zip((d for d, _ in cands), s), key=lambda x: (-x[1], x[0]))
        return ranked[:k] if k is not None else ranked

    def describe(self) -> dict:
        return {"reranker": self.name, "model": self.model_id, "max_length": self.max_length}


__all__ = ["MonoT5Reranker", "DEFAULT_MODEL"]
