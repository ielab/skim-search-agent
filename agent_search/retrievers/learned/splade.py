"""SPLADE: learned sparse retrieval. A masked-language-model head scores every vocabulary term
for a text; the representation is `max over positions of log(1 + relu(logit))`, so a page
becomes a few hundred weighted terms. The score of a page is the dot product of its weights
with the query's, computed exactly over the whole corpus.

The default is SPLADE++ (`naver/splade-cocondenser-ensembledistil`), served as its card and
Pyserini serve it: pages cut to 512 tokens, queries to 256. The index is a terms x pages sparse
matrix (`scipy.sparse`, compressed by term), so a query reads only the rows of its own terms.

Knobs: `SPLADE_MODEL`, `SPLADE_DOC_LENGTH`, `SPLADE_QUERY_LENGTH` (`retrieval.splade_*`).
"""
from __future__ import annotations

import os
from typing import Optional

from agent_search.retrievers.learned.base import LearnedIndexRetriever

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
DEFAULT_DOC_LENGTH = 512
DEFAULT_QUERY_LENGTH = 256
# A SPLADE++ page keeps a few hundred terms. Far more means the masked-LM head is not the trained
# one (a checkpoint without it loads with a random head), so the build stops instead of filling memory.
MAX_MEAN_TERMS = 2000


class SpladeRetriever(LearnedIndexRetriever):
    name = "splade"

    def __init__(self, model: Optional[str] = None, doc_length: Optional[int] = None,
                 query_length: Optional[int] = None, **kw):
        super().__init__(model or os.environ.get("SPLADE_MODEL") or DEFAULT_MODEL,
                         int(doc_length or os.environ.get("SPLADE_DOC_LENGTH") or DEFAULT_DOC_LENGTH), **kw)
        self.query_length = int(query_length or os.environ.get("SPLADE_QUERY_LENGTH") or DEFAULT_QUERY_LENGTH)
        self._model = None
        self._matrix = None

    def length_tag(self) -> str:
        return f"dl{self.doc_length}"           # the query length changes no stored weight

    # --- the model ---------------------------------------------------------------------------
    def _encoder(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            dev = self.torch_device()
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForMaskedLM.from_pretrained(self.model_id).to(dev).eval()
            if dev != "cpu":
                model = model.to(torch.bfloat16)
            self._model = (tok, model, dev)
        return self._model

    def _weights(self, texts: list[str], max_length: int):
        """`[len(texts), vocab]` term weights, on the model's device, float32."""
        import torch
        from agent_search.retrievers.dense.base import encoder_prefix
        tok, model, dev = self._encoder()
        texts = [encoder_prefix(t or "", tok, max_length) for t in texts]
        batch = tok(texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**batch).logits.float()
        w = torch.log1p(torch.relu(logits)) * batch["attention_mask"].unsqueeze(-1)
        return w.max(dim=1).values

    # --- the index ---------------------------------------------------------------------------
    def encode_corpus(self, texts: list[str], out_dir: str) -> dict:
        import numpy as np
        import scipy.sparse as sp
        rows, cols, vals = [], [], []
        for start in range(0, len(texts), self.batch_size):
            w = self._weights(texts[start:start + self.batch_size], self.doc_length)
            nz = w.nonzero(as_tuple=False)
            rows.append((nz[:, 0] + start).cpu().numpy().astype(np.int32))
            cols.append(nz[:, 1].cpu().numpy().astype(np.int32))
            vals.append(w[nz[:, 0], nz[:, 1]].cpu().numpy().astype(np.float32))
            terms = sum(len(r) for r in rows) / (start + len(w))
            if terms > MAX_MEAN_TERMS:
                raise RuntimeError(f"SPLADE pages average {terms:.0f} non-zero terms (> {MAX_MEAN_TERMS}): "
                                   f"{self.model_id!r} is not loading a trained SPLADE head")
            if start // self.batch_size % 200 == 0:
                print(f"  [splade] {start + len(w)}/{len(texts)} pages, {terms:.0f} terms per page", flush=True)
        vocab = self._encoder()[1].config.vocab_size
        m = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                           (np.concatenate(cols), np.concatenate(rows))), shape=(vocab, len(texts)))
        sp.save_npz(os.path.join(out_dir, "weights.npz"), m)
        return {"query_length": self.query_length, "nnz": int(m.nnz), "vocab": vocab}

    def load_index(self, cache_dir: str, meta: dict) -> None:
        import scipy.sparse as sp
        self._matrix = sp.load_npz(os.path.join(cache_dir, "weights.npz")).tocsr()

    def scores(self, query: str):
        import numpy as np
        q = self._weights([query], self.query_length)[0]
        terms = q.nonzero(as_tuple=True)[0].cpu().numpy()
        if not len(terms):
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        w = q[terms].cpu().numpy().astype(np.float32)
        return np.asarray(self._matrix[terms].T @ w).ravel()

    def describe(self) -> dict:
        return {**super().describe(), "query_length": self.query_length}


__all__ = ["SpladeRetriever", "DEFAULT_MODEL"]
