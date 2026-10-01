"""DiffRetriever: a diffusion language model as a multi-vector retriever (ielabgroup/diffretriever-*,
arXiv:2605.07210). The model appends K masked positions to a retrieval prompt and reads their
hidden states in one bidirectional pass: 4 vectors per query and 16 per passage on the multi
checkpoint, 3584 dimensions each on Dream-7B.

The model pins transformers 4.54 and peft, so it runs in its own environment behind
`scripts/serve_diffretriever.py` and this class talks to it over HTTP (`DIFFRETRIEVER_URL`),
the way the backbone is reached through vLLM. The server tokenises and encodes exactly as the
model card does; the score is the card's: normalised vectors, for each query vector the best
passage vector, clamped at zero, summed over the query vectors. Every page is scored (no
candidate stage).

The model is hybrid, and `DIFFRETRIEVER_MODE` picks the score:

* `dense` (default): the card's multi-vector MaxSim above.
* `sparse`: the dot product of query and page term weights, built the way the authors' encoder
  builds them (content words only, top 128 per masked position, x100 rounded, max over positions;
  the server does it from the same forward pass).
* `hybrid`: the authors' fusion (PromptReps style): the top `FUSION_DEPTH` pages of each, each
  list min-max normalised, `FUSION_ALPHA` x dense + (1 - `FUSION_ALPHA`) x sparse, a page absent
  from one list counting 0 there.

The index stores the K_p passage vectors of every page at fp16, already normalised, and for the
sparse and hybrid modes a terms x pages sparse matrix next to them (`sparse.npz`), built in its
own pass when the dense index already exists. At search time the vectors sit on the visible GPU
with the most free memory when they fit there, else in CPU memory, scored in chunks
(`DIFFRETRIEVER_STORE_DEVICE` forces either).

Knobs: `DIFFRETRIEVER_URL`, `DIFFRETRIEVER_MODEL`, `DIFFRETRIEVER_MODE`, `DIFFRETRIEVER_DOC_LENGTH`
(text tokens of a page after the prompt; the card's total is 156, prompt included),
`DIFFRETRIEVER_STORE_DEVICE` (`retrieval.diffretriever_*`).
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Optional

from agent_search.errors import SetupError
from agent_search.retrievers.learned.base import LearnedIndexRetriever

DEFAULT_MODEL = "ielabgroup/diffretriever-dream-7b-multi-q4-p16"
DEFAULT_DOC_LENGTH = 101          # the card's 156 tokens minus the passage prompt (23 + 32)
CHUNK_PAGES = 4096                # pages scored per step
BATCH = 16                        # pages per request while building
MODES = ("dense", "sparse", "hybrid")
FUSION_DEPTH = 1000               # pages per list the hybrid fuses (the authors' --top_k)
FUSION_ALPHA = 0.5                # weight of the dense list (the authors' normalized_fusion)


class DiffRetrieverRetriever(LearnedIndexRetriever):
    name = "diffretriever"

    def __init__(self, model: Optional[str] = None, doc_length: Optional[int] = None,
                 url: Optional[str] = None, store_device: Optional[str] = None, mode: Optional[str] = None, **kw):
        kw.setdefault("batch_size", BATCH)
        super().__init__(model or os.environ.get("DIFFRETRIEVER_MODEL") or DEFAULT_MODEL,
                         int(doc_length or os.environ.get("DIFFRETRIEVER_DOC_LENGTH") or DEFAULT_DOC_LENGTH), **kw)
        self.url = (url or os.environ.get("DIFFRETRIEVER_URL") or "").rstrip("/")
        self.store_device = (store_device or os.environ.get("DIFFRETRIEVER_STORE_DEVICE") or "auto").lower()
        self.mode = (mode or os.environ.get("DIFFRETRIEVER_MODE") or "dense").lower()
        if self.mode not in MODES:
            raise ValueError(f"DIFFRETRIEVER_MODE={self.mode!r}; choose from {MODES}")
        self._store = None            # [pages, K_p, H] fp16, normalised
        self._sparse = None           # terms x pages scipy CSR, for the sparse and hybrid modes
        self._k = 0

    # --- the server ----------------------------------------------------------------------------
    def _post(self, texts: list[str], is_query: bool, sparse: bool = False):
        import numpy as np
        if not self.url:
            raise SetupError("DiffRetriever needs DIFFRETRIEVER_URL: start scripts/serve_diffretriever.py "
                             "in the model's own environment and point the run at it")
        req = {"texts": texts, "is_query": is_query, "max_text_tokens": None if is_query else self.doc_length}
        if sparse:
            req["sparse"] = True
        body = json.dumps(req).encode()
        req = urllib.request.Request(self.url + "/encode", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        if "error" in out:
            raise RuntimeError(f"diffretriever server: {out['error']}")
        dense = np.frombuffer(base64.b64decode(out["data"]), dtype=np.float16).reshape(out["shape"])
        return (dense, out.get("sparse")) if sparse else dense

    def vectors(self, texts: list[str], is_query: bool):
        """`[n, K, H]` float32 torch tensor, L2-normalised per vector."""
        import torch
        v = torch.from_numpy(self._post(texts, is_query).astype("float32"))
        return torch.nn.functional.normalize(v, p=2, dim=-1)

    def vectors_and_terms(self, texts: list[str], is_query: bool):
        """The vectors and each text's sparse terms (`{"ids", "vals"}`) from one forward pass."""
        import torch
        dense, terms = self._post(texts, is_query, sparse=True)
        v = torch.nn.functional.normalize(torch.from_numpy(dense.astype("float32")), p=2, dim=-1)
        return v, terms

    # --- the index ---------------------------------------------------------------------------
    def is_cached(self, key: Optional[str] = None) -> bool:
        if self.mode == "dense":
            return self.index_complete(key)
        return self.index_complete(key) and os.path.exists(os.path.join(self._cache_dir(key), "sparse.npz"))

    def index(self, units, key: Optional[str] = None) -> "DiffRetrieverRetriever":
        super().index(units, key)
        if self.mode != "dense" and self._sparse is None:
            # the dense index existed already: add the sparse part in its own pass
            from agent_search.retrievers.learned.base import page_texts
            self._write_sparse(self._cache_dir(key), page_texts(units))
            self._load_sparse(self._cache_dir(key))
        return self

    def _write_sparse(self, out_dir: str, texts: list[str], terms: Optional[list] = None) -> None:
        import numpy as np
        import scipy.sparse as sp
        if terms is None:
            terms = []
            for start in range(0, len(texts), self.batch_size):
                terms += self.vectors_and_terms(texts[start:start + self.batch_size], is_query=False)[1]
                if start // self.batch_size % 200 == 0:
                    print(f"  [diffretriever] sparse {start}/{len(texts)} pages", flush=True)
        rows = np.concatenate([np.asarray(t["ids"], dtype=np.int32) for t in terms]) if terms else np.zeros(0, np.int32)
        cols = np.concatenate([np.full(len(t["ids"]), j, dtype=np.int32) for j, t in enumerate(terms)]) if terms else np.zeros(0, np.int32)
        vals = np.concatenate([np.asarray(t["vals"], dtype=np.float32) for t in terms]) if terms else np.zeros(0, np.float32)
        vocab = int(rows.max()) + 1 if len(rows) else 1
        m = sp.csr_matrix((vals, (rows, cols)), shape=(vocab, len(terms)))
        sp.save_npz(os.path.join(out_dir, "sparse.npz"), m)

    def _load_sparse(self, cache_dir: str) -> None:
        import scipy.sparse as sp
        self._sparse = sp.load_npz(os.path.join(cache_dir, "sparse.npz")).tocsr()

    def length_tag(self) -> str:
        return f"dl{self.doc_length}"

    def encode_corpus(self, texts: list[str], out_dir: str) -> dict:
        k = hidden = 0
        want_sparse = self.mode != "dense"
        terms: list = []
        with open(os.path.join(out_dir, "vectors.f16"), "wb") as fh:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                if want_sparse:
                    v, t = self.vectors_and_terms(batch, is_query=False)
                    terms += t
                else:
                    v = self.vectors(batch, is_query=False)
                k, hidden = int(v.shape[1]), int(v.shape[2])
                fh.write(v.half().numpy().tobytes())
                if start // self.batch_size % 200 == 0:
                    print(f"  [diffretriever] {start + len(v)}/{len(texts)} pages", flush=True)
        if want_sparse:
            self._write_sparse(out_dir, texts, terms)
        return {"k_passage": k, "hidden": hidden}

    def _where(self, size: int) -> str:
        import torch
        if self.store_device != "auto":
            return self.store_device
        best, free_best = "cpu", 0
        for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            free, _ = torch.cuda.mem_get_info(i)
            if free > free_best:
                best, free_best = f"cuda:{i}", free
        return best if free_best > size + (3 << 30) else "cpu"

    def load_index(self, cache_dir: str, meta: dict) -> None:
        import numpy as np
        import torch
        n, self._k, hidden = int(meta["n"]), int(meta["k_passage"]), int(meta["hidden"])
        disk = np.memmap(os.path.join(cache_dir, "vectors.f16"), dtype=np.float16, mode="r",
                         shape=(n, self._k, hidden))
        where = self._where(n * self._k * hidden * 2)
        store = torch.empty((n, self._k, hidden), dtype=torch.float16, device=where)
        for a in range(0, n, 8192):
            store[a:a + 8192] = torch.from_numpy(np.array(disk[a:a + 8192])).to(where)
        self._store = store
        if os.path.exists(os.path.join(cache_dir, "sparse.npz")):
            self._load_sparse(cache_dir)
        print(f"  [diffretriever] {n} pages x {self._k} vectors x {hidden} ({n * self._k * hidden * 2 / 2**30:.1f} GB) "
              f"on {where}", flush=True)

    def scores(self, query: str):
        if self.mode == "dense":
            return self._dense_scores(self.vectors([query], is_query=True)[0])
        q, terms = self.vectors_and_terms([query], is_query=True)
        sparse = self._sparse_scores(terms[0])
        if self.mode == "sparse":
            return sparse
        return self._fused(self._dense_scores(q[0]).float().cpu().numpy(), sparse)

    def _sparse_scores(self, terms: dict):
        import numpy as np
        ids = np.asarray(terms["ids"], dtype=np.int64)
        vals = np.asarray(terms["vals"], dtype=np.float32)
        keep = ids < self._sparse.shape[0]                  # a query term no page has scores 0
        if not keep.any():
            return np.zeros(self._sparse.shape[1], dtype=np.float32)
        return np.asarray(self._sparse[ids[keep]].T @ vals[keep]).ravel()

    @staticmethod
    def _fused(dense, sparse):
        """The authors' fusion: each score list cut to its top FUSION_DEPTH, min-max normalised,
        FUSION_ALPHA x dense + (1 - FUSION_ALPHA) x sparse, 0 where a page is missing from a list;
        pages in neither list score -1."""
        import numpy as np

        def top_minmax(s):
            k = min(FUSION_DEPTH, len(s))
            top = np.argpartition(-s, k - 1)[:k]
            v = s[top]
            lo, hi = float(v.min()), float(v.max())
            return top, (v - lo) / max(hi - lo, 1e-9)

        out = np.full(len(dense), -1.0, dtype=np.float32)
        d_ids, d_norm = top_minmax(dense)
        s_ids, s_norm = top_minmax(sparse)
        fused = {}
        for i, v in zip(d_ids, d_norm):
            fused[int(i)] = FUSION_ALPHA * float(v)
        for i, v in zip(s_ids, s_norm):
            fused[int(i)] = fused.get(int(i), 0.0) + (1 - FUSION_ALPHA) * float(v)
        idx = np.fromiter(fused.keys(), dtype=np.int64)
        out[idx] = np.fromiter(fused.values(), dtype=np.float32)
        return out

    def _dense_scores(self, qv):
        import torch
        dev = self._store.device if self._store.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = qv.to(dev, torch.float16)                                            # [K_q, H]
        n = self._store.shape[0]
        out = torch.empty(n, dtype=torch.float32, device=dev)
        for a in range(0, n, CHUNK_PAGES):
            p = self._store[a:a + CHUNK_PAGES].to(dev, non_blocking=True)        # [c, K_p, H]
            sim = torch.einsum("kh,cdh->ckd", q, p).float()                      # [c, K_q, K_p]
            out[a:a + CHUNK_PAGES] = sim.max(dim=-1).values.clamp(min=0).sum(dim=-1)
        return out

    def describe(self) -> dict:
        return {**super().describe(), "url": self.url, "mode": self.mode}


__all__ = ["DiffRetrieverRetriever", "DEFAULT_MODEL"]
