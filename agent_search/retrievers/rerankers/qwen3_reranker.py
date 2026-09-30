"""The Qwen3-Reranker family: a causal language model asked a yes/no question about one
(query, document) pair. The score is `logit(yes) - logit(no)` at the last position, the way
the model card serves it. `Qwen/Qwen3-Reranker-0.6B`, `-4B`, `-8B` and any fine-tune of them
(a Hub id or a local directory) load here; a sequence-classification reranker such as
`BAAI/bge-reranker-v2-m3` belongs to `cross_encoder.py`.

The prompt is the model card's: a fixed system line, then `<Instruct>`, `<Query>` and
`<Document>` in one user turn, then an empty think block the model continues from. The
instruction is the model card's retrieval default; `instruction=` changes it. `query_block`
is the text placed after `<Query>:`; here that is the search query itself, and a subclass for
a fine-tune trained on a longer block (the main question, the documents read so far) overrides
it.

The model loads once per process and is shared across worker threads. The device follows
`AGENT_SEARCH_DENSE_DEVICE`, the same knob the dense encoders and the cross-encoder read.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Sequence

from agent_search.retrievers.dense.base import encoder_prefix
from agent_search.retrievers.rerankers.base import Candidate, Reranker, register_reranker

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_MODELS: dict = {}
_MODELS_LOCK = threading.Lock()


def _shared_model(model_id: str, device: Optional[str]):
    """One (tokenizer, model, yes_id, no_id, device, lock) per checkpoint and device, loaded on first use."""
    key = (model_id, device or "auto")
    with _MODELS_LOCK:
        m = _MODELS.get(key)
        if m is None:
            import torch                                                   # heavy import, on first use
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if dev != "cpu" else torch.float32
            tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(dev).eval()
            yes_id = tok.convert_tokens_to_ids("yes")
            no_id = tok.convert_tokens_to_ids("no")
            # one lock per shared model: the fast tokenizer is not thread-safe ("Already
            # borrowed" when two worker threads change its truncation at once), and one GPU
            # serves the pairs in order anyway
            m = (tok, model, yes_id, no_id, dev, threading.Lock())
            _MODELS[key] = m
    return m


@register_reranker
class Qwen3Reranker(Reranker):
    name = "qwen3_reranker"
    default_model = DEFAULT_MODEL
    default_max_length = 512

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32, max_length: int = 512,
                 device: Optional[str] = None, instruction: str = DEFAULT_INSTRUCTION):
        self.model_id = model
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)       # tokens per (query, document) pair, prefix and suffix included
        self.device = device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
        self.instruction = instruction
        self._loaded = None

    # -- the prompt --------------------------------------------------------------------------

    def query_block(self, query: str) -> str:
        """What follows `<Query>:`. The plain reranker judges the search query alone."""
        return query

    def prompt(self, query: str, document: str) -> str:
        """The full prompt as text (for reading and tests); scoring builds the same thing as ids."""
        return f"{PREFIX}{self._pair_text(query, document)}{SUFFIX}"

    # -- scoring -----------------------------------------------------------------------------

    def _model(self):
        if self._loaded is None:
            self._loaded = _shared_model(self.model_id, self.device)
        return self._loaded

    def _pair_text(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {self.query_block(query)}\n<Document>: {document}"

    def _inputs(self, tok, query: str, texts: Sequence[str]) -> list[list[int]]:
        """Token ids per candidate, built the way the model card's reference code builds them:
        the system prefix and the assistant suffix tokenized once, the pair text tokenized and
        truncated to what is left of `max_length`, the three joined as ids. Every input is at
        most `max_length` tokens by construction. (An earlier version cut the document, decoded
        it and re-tokenized the prompt; a page with stray bytes grew several times over in that
        round trip and one batch asked for 54 GB.) A candidate is a whole page, so it goes
        through `encoder_prefix` first and the tokenizer never reads the tail."""
        pre = tok(PREFIX, add_special_tokens=False)["input_ids"]
        suf = tok(SUFFIX, add_special_tokens=False)["input_ids"]
        room = max(self.max_length - len(pre) - len(suf), 16)
        out = []
        for t in texts:
            pair = self._pair_text(query, encoder_prefix(t or "", tok, room))
            ids = tok(pair, add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
            out.append(pre + ids + suf)
        return out

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        import torch
        tok, model, yes_id, no_id, dev, lock = self._model()
        with lock:
            inputs = self._inputs(tok, query, texts)
            out: list[float] = []
            with torch.no_grad():
                for i in range(0, len(inputs), self.batch_size):
                    batch = tok.pad({"input_ids": inputs[i:i + self.batch_size]}, padding=True,
                                    return_tensors="pt").to(dev)
                    # the yes/no logits at the last position only: the full logits tensor
                    # (batch x length x 150k vocabulary) is ~10 GB per batch of 32 x 1024
                    hidden = model.model(**batch).last_hidden_state[:, -1, :]
                    logits = model.lm_head(hidden)
                    out.extend((logits[:, yes_id] - logits[:, no_id]).float().tolist())
        return out

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: Optional[int] = None) -> list[tuple[str, float]]:
        cands = [(d, t or "") for d, t in candidates]
        if not cands:
            return []
        scores = self.scores(query, [t for _, t in cands])
        ranked = sorted(zip((d for d, _ in cands), scores), key=lambda x: (-x[1], x[0]))
        return ranked[:k] if k is not None else ranked

    def describe(self) -> dict:
        return {"reranker": self.name, "model": self.model_id, "max_length": self.max_length,
                "instruction": self.instruction}


__all__ = ["Qwen3Reranker", "DEFAULT_MODEL", "DEFAULT_INSTRUCTION", "PREFIX", "SUFFIX"]
