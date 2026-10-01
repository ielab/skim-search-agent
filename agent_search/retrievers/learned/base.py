"""The frame shared by retrievers that read every page with a neural model ahead of a run and
search the result exactly: a learned sparse index (`splade.py`) and a late-interaction index
(`colbert.py`).

A subclass encodes pages and queries and scores; this class owns what is the same for both:
the text a page is indexed under (its title line and body, as the dense retrievers use), the
cut to the model's document length (`encoder_prefix`, so a tokenizer never reads a megabyte
tail), the cache folder under `index_root` keyed by model, length and corpus, the corpus
fingerprint that refuses a stale cache, the device, and one lock per process so concurrent
episodes share one model and one index.

Nothing is encoded during a run: `Engines` builds a learned engine only from a persisted index
(`skimsearchagent-build-indexes --retriever splade|colbert`), and a missing one is a SetupError.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Optional, Sequence

from agent_search.retrievers.base import Retriever


def page_texts(units: Sequence) -> list[str]:
    """The text each page is indexed under: its title line, then its body."""
    return [f"{u.qualname}\n{u.code}" for u in units]


class LearnedIndexRetriever(Retriever):
    """Build or load a persisted index of every page; `search(query, k)` ranks all of them."""

    name = "learned"
    returns_full_set = False

    def __init__(self, model: str, doc_length: int, index_root: str = "indexes", rebuild: bool = False,
                 device: Optional[str] = None, batch_size: int = 32):
        self.model_id = model
        self.doc_length = int(doc_length)
        self.index_root, self.rebuild = index_root, rebuild
        self.device = device or os.environ.get("AGENT_SEARCH_DENSE_DEVICE") or None
        self.batch_size = int(batch_size)
        self.doc_ids: list[str] = []
        self._lock = threading.Lock()
        self._loaded = False

    # --- what a subclass provides ------------------------------------------------------------
    def length_tag(self) -> str:
        """What in the folder name separates two indexes of one model (the lengths)."""
        return f"dl{self.doc_length}"

    def encode_corpus(self, texts: list[str], out_dir: str) -> dict:
        """Encode every page and write the index files into `out_dir`; return extra metadata."""
        raise NotImplementedError

    def load_index(self, cache_dir: str, meta: dict) -> None:
        """Load the index files written by `encode_corpus`."""
        raise NotImplementedError

    def scores(self, query: str):
        """A score for every page, in `self.doc_ids` order (a 1-D torch or numpy array)."""
        raise NotImplementedError

    # --- device ------------------------------------------------------------------------------
    def torch_device(self) -> str:
        import torch
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- the cache ---------------------------------------------------------------------------
    def _cache_dir(self, key: Optional[str]) -> str:
        model_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", self.model_id)
        corpus_key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", key or "default")
        return os.path.join(self.index_root, self.name, f"{model_key}-{self.length_tag()}", corpus_key)

    def index_complete(self, key: Optional[str] = None) -> bool:
        """The index `encode_corpus` writes is on disk (its meta.json is written last)."""
        return os.path.exists(os.path.join(self._cache_dir(key), "meta.json"))

    def is_cached(self, key: Optional[str] = None) -> bool:
        """Everything a run of this retriever reads is on disk. A subclass that adds a part
        (DiffRetriever's sparse index) extends this, never `index_complete`."""
        return self.index_complete(key)

    def index(self, units: Sequence, key: Optional[str] = None) -> "LearnedIndexRetriever":
        from agent_search.corpus.fingerprint import corpus_fingerprint
        cache_dir = self._cache_dir(key)
        fp = corpus_fingerprint(units)
        expected = [u.doc_id for u in units]
        with self._lock:
            if not self.rebuild and self.index_complete(key):
                meta = json.load(open(os.path.join(cache_dir, "meta.json")))
                ids = json.load(open(os.path.join(cache_dir, "doc_ids.json")))
                if ids == expected and meta.get("corpus_fingerprint") == fp:
                    self.doc_ids = ids
                    self.load_index(cache_dir, meta)
                    self._loaded = True
                    return self
                print(f"  [{self.name}] cached index at {cache_dir!r} does not match the corpus; rebuilding",
                      flush=True)
            os.makedirs(cache_dir, exist_ok=True)
            self.doc_ids = expected
            extra = self.encode_corpus(page_texts(units), cache_dir)
            json.dump(self.doc_ids, open(os.path.join(cache_dir, "doc_ids.json"), "w"))
            meta = {"model": self.model_id, "doc_length": self.doc_length, "n": len(self.doc_ids),
                    "corpus_fingerprint": fp, **extra}
            # meta.json last: its presence is what marks the index complete
            json.dump(meta, open(os.path.join(cache_dir, "meta.json"), "w"), indent=1)
            self.load_index(cache_dir, meta)
            self._loaded = True
        return self

    # --- ranking -----------------------------------------------------------------------------
    def search_scored(self, query: str, k: Optional[int] = None) -> list[tuple[str, float]]:
        query = " ".join((query or "").split())
        if not query or not self._loaded:
            return []
        import numpy as np
        with self._lock:
            s = self.scores(query)
        s = s.float().cpu().numpy() if hasattr(s, "cpu") else np.asarray(s, dtype=np.float32)
        k = min(int(k or 10), len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(self.doc_ids[i], float(s[i])) for i in top]

    def search(self, query: str, k: int) -> list[str]:
        return [d for d, _ in self.search_scored(query, k)]

    def top_k_doc_ids(self, query: str, k: Optional[int] = None) -> list[str]:
        return self.search(query, k or 10)

    def search_with_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_scored(query, k)

    def describe(self) -> dict:
        return {"retriever": self.name, "model": self.model_id, "doc_length": self.doc_length}


__all__ = ["LearnedIndexRetriever", "page_texts"]
