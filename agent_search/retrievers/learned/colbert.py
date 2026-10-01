"""ColBERT: late-interaction retrieval. Every token of a page keeps its own 128-dimensional
vector; a query's score against a page is the sum, over the query's tokens, of the best cosine
match among the page's tokens (MaxSim). The ranking here is exact: every page is scored, with
no candidate stage in front.

Served the way ColBERTv2 (`colbert-ir/colbertv2.0`) was trained and its own code serves it: a
`[unused0]` marker after `[CLS]` on queries and `[unused1]` on pages, queries padded with
`[MASK]` to 32 tokens (the padding is not attended but is scored), punctuation dropped from
page vectors, every vector L2-normalised, a linear 768 -> 128 head over BERT. The checkpoint was
trained on 180-token pages; `COLBERT_DOC_LENGTH` sets the length a run indexes at.

The index is one fp16 file of all page-token vectors plus each page's offset into it. At
search time it sits on the GPU when it fits next to whatever else holds that GPU, and is
streamed from CPU memory in chunks when it does not (`COLBERT_STORE_DEVICE` forces either).

Knobs: `COLBERT_MODEL`, `COLBERT_DOC_LENGTH`, `COLBERT_QUERY_LENGTH`, `COLBERT_STORE_DEVICE`
(`retrieval.colbert_*`).
"""
from __future__ import annotations

import os
import string
from typing import Optional

from agent_search.retrievers.learned.base import LearnedIndexRetriever

DEFAULT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_DOC_LENGTH = 180          # ColBERTv2's training length
DEFAULT_QUERY_LENGTH = 32
DIM = 128
CHUNK_TOKENS = 4_000_000          # page tokens scored per step


class ColbertRetriever(LearnedIndexRetriever):
    name = "colbert"

    def __init__(self, model: Optional[str] = None, doc_length: Optional[int] = None,
                 query_length: Optional[int] = None, store_device: Optional[str] = None, **kw):
        super().__init__(model or os.environ.get("COLBERT_MODEL") or DEFAULT_MODEL,
                         int(doc_length or os.environ.get("COLBERT_DOC_LENGTH") or DEFAULT_DOC_LENGTH), **kw)
        self.query_length = int(query_length or os.environ.get("COLBERT_QUERY_LENGTH") or DEFAULT_QUERY_LENGTH)
        self.store_device = (store_device or os.environ.get("COLBERT_STORE_DEVICE") or "auto").lower()
        self._model = None
        self._store = None            # [tokens, DIM] fp16, on the GPU or in CPU memory
        self._offsets = None          # [pages + 1] int64, page i owns tokens offsets[i]:offsets[i+1]

    # --- the model ---------------------------------------------------------------------------
    def _encoder(self):
        if self._model is None:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer, BertModel
            dev = self.torch_device()
            path = self.model_id if os.path.isdir(self.model_id) else snapshot_download(self.model_id)
            tok = AutoTokenizer.from_pretrained(path)
            bert = BertModel.from_pretrained(path, add_pooling_layer=False).to(dev).eval()
            linear = torch.nn.Linear(bert.config.hidden_size, DIM, bias=False)
            linear.weight.data.copy_(_linear_weight(path))
            linear = linear.to(dev).eval()
            if dev != "cpu":
                bert, linear = bert.half(), linear.half()
            ids = tok.convert_tokens_to_ids(["[unused0]", "[unused1]"])
            skip = {tok.encode(c, add_special_tokens=False)[0] for c in string.punctuation}
            self._model = (tok, bert, linear, dev, ids[0], ids[1], skip)
        return self._model

    def _embed(self, input_ids, attention_mask):
        import torch
        _, bert, linear, *_ = self._encoder()
        with torch.no_grad():
            h = linear(bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        return torch.nn.functional.normalize(h.float(), p=2, dim=-1)

    def _marked(self, texts: list[str], max_length: int, marker: int) -> list[list[int]]:
        """Token ids with the marker after `[CLS]`, at most `max_length` long."""
        from agent_search.retrievers.dense.base import encoder_prefix
        tok = self._encoder()[0]
        texts = [encoder_prefix(t or "", tok, max_length) for t in texts]
        ids = tok(texts, max_length=max_length - 1, truncation=True, add_special_tokens=True)["input_ids"]
        return [[i[0], marker] + i[1:] for i in ids]

    def query_vectors(self, query: str):
        """`[query_length, DIM]`: the query padded with `[MASK]`, every position scored."""
        import torch
        tok, _, _, dev, q_marker, _, _ = self._encoder()
        ids = self._marked([query], self.query_length, q_marker)[0]
        pad = self.query_length - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [tok.mask_token_id] * pad
        t = torch.tensor([ids], device=dev)
        return self._embed(t, torch.tensor([mask], device=dev))[0]

    # --- the index ---------------------------------------------------------------------------
    def length_tag(self) -> str:
        return f"dl{self.doc_length}"

    def encode_corpus(self, texts: list[str], out_dir: str) -> dict:
        import numpy as np
        import torch
        tok, _, _, dev, _, d_marker, skip = self._encoder()
        lengths = []
        with open(os.path.join(out_dir, "tokens.f16"), "wb") as fh:
            for start in range(0, len(texts), self.batch_size):
                ids = self._marked(texts[start:start + self.batch_size], self.doc_length, d_marker)
                width = max(map(len, ids))
                mask = [[1] * len(i) + [0] * (width - len(i)) for i in ids]
                padded = [i + [tok.pad_token_id] * (width - len(i)) for i in ids]
                vec = self._embed(torch.tensor(padded, device=dev), torch.tensor(mask, device=dev))
                # a boolean mask (an integer tensor here would gather rows 0 and 1 instead of masking)
                keep = torch.tensor([[bool(m) and t not in skip for t, m in zip(i, mk)] for i, mk in zip(padded, mask)],
                                    dtype=torch.bool, device=dev)
                for v, k in zip(vec, keep):
                    kept = v[k].half().cpu().numpy()
                    fh.write(kept.tobytes())
                    lengths.append(len(kept))
                if start // self.batch_size % 200 == 0:
                    print(f"  [colbert] {start + len(ids)}/{len(texts)} pages", flush=True)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        return {"query_length": self.query_length, "tokens": int(offsets[-1]), "dim": DIM}

    def load_index(self, cache_dir: str, meta: dict) -> None:
        import numpy as np
        import torch
        self._offsets = np.load(os.path.join(cache_dir, "offsets.npy"))
        n = int(self._offsets[-1])
        disk = np.memmap(os.path.join(cache_dir, "tokens.f16"), dtype=np.float16, mode="r", shape=(n, DIM))
        size = n * DIM * 2
        where = self.store_device
        if where == "auto":
            where = "cpu"
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                where = "cuda" if free > size + (3 << 30) else "cpu"
        store = torch.empty((n, DIM), dtype=torch.float16, device=where)
        step = 8_000_000
        for a in range(0, n, step):
            store[a:a + step] = torch.from_numpy(np.array(disk[a:a + step])).to(where)
        self._store = store
        print(f"  [colbert] {len(self._offsets) - 1} pages, {n} token vectors ({size / 2**30:.1f} GB) on {where}",
              flush=True)

    def scores(self, query: str):
        import numpy as np
        import torch
        q = self.query_vectors(query).half()                         # [Lq, DIM] on the model's device
        off = self._offsets
        pages = len(off) - 1
        out = torch.empty(pages, dtype=torch.float32, device=q.device)
        page = 0
        while page < pages:
            # the pages whose tokens fit in one chunk, at least one page
            fit = int(np.searchsorted(off, off[page] + CHUNK_TOKENS, side="right")) - 1
            end = min(pages, max(page + 1, fit))
            a, b = int(off[page]), int(off[end])
            block = self._store[a:b].to(q.device, non_blocking=True)
            sim = (block @ q.T).float()                               # [tokens, Lq]
            lengths = torch.from_numpy(off[page + 1:end + 1] - off[page:end]).to(q.device)
            best = torch.segment_reduce(sim, "max", lengths=lengths, axis=0, unsafe=True)
            out[page:end] = torch.nan_to_num(best, neginf=0.0).sum(dim=1)
            page = end
        return out

    def describe(self) -> dict:
        return {**super().describe(), "query_length": self.query_length}


def _linear_weight(path: str):
    """The 768 -> 128 head stored next to the BERT weights (`linear.weight`)."""
    p = os.path.join(path, "model.safetensors")
    if os.path.exists(p):
        from safetensors.torch import load_file
        return load_file(p)["linear.weight"]
    import torch
    return torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")["linear.weight"]


__all__ = ["ColbertRetriever", "DEFAULT_MODEL"]
