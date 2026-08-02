"""Pluggable vector index — the storage + nearest-neighbour layer under the dense
retriever, separated from the text encoder so it scales independently of the model.

Embeddings of a FIXED corpus are a one-time, immutable artifact: build once, persist,
reuse forever. This module owns that artifact's storage format and search. Three
backends, one interface, chosen by corpus size and whether FAISS is installed:

  flat   numpy, float16, brute-force cosine — DEFAULT, no dependency. Exact. At 100k
         docs this is ~0.15 GB and a few-ms matmul; nothing faster is needed.
  hnsw   FAISS IndexHNSWFlat — graph ANN, sub-millisecond search. Engages for large
         corpora (~1M+) when faiss is available.
  ivfpq  FAISS IndexIVFPQ — product-quantized, COMPRESSED storage for huge/memory-
         bound corpora (10M+): 30 GB of float32 → hundreds of MB.

Selection is automatic (`choose_backend`) but overridable with AGENT_SEARCH_ANN
(flat|hnsw|ivfpq|auto). The default thresholds keep small/medium corpora on `flat`
with no FAISS dependency, so adding a much larger corpus later is a config flip, not
a rewrite — and a missing/broken faiss degrades to `flat`, never a crash.

flat / faiss-flat: `flat`'s numpy `emb(float16) @ qv(float32)` has no BLAS kernel for
float16, so numpy falls back to a slow elementwise path (measured ~178ms/query at
67.7k docs). AGENT_SEARCH_FLAT_FAISS=1 (default OFF) opts a `FlatIndex` into an
in-memory `faiss.IndexFlatIP` built once (lazily, on first search, from the SAME
stored embeddings cast fp16->fp32) instead of the numpy matmul — still EXACT brute-
force search (not ANN), just SIMD+threaded (measured ~2-5ms/query on the same
corpus). This changes nothing about the on-disk cache: no new required artifact, no
identity/name change, and it is fully opt-in — with the env var unset, `FlatIndex`
behaves byte-identically to before this existed. Missing faiss with the flag set
degrades to the numpy path, never crashes.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional, Sequence


# --- backend selection ------------------------------------------------------

def _faiss():
    """Import faiss or return None (cluster-only optional dependency)."""
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _flat_faiss_enabled() -> bool:
    """AGENT_SEARCH_FLAT_FAISS opt-in (default OFF) for the exact faiss.IndexFlatIP
    fast path inside `FlatIndex`. Unset/empty/"0"/"false" -> disabled (current numpy
    behavior, untouched)."""
    return os.environ.get("AGENT_SEARCH_FLAT_FAISS", "").strip().lower() in (
        "1", "true", "yes", "on")


def choose_backend(n_docs: int) -> str:
    """flat for small/medium corpora (no faiss needed); FAISS ANN for large ones.

    Override with AGENT_SEARCH_ANN=flat|hnsw|ivfpq|auto. Thresholds:
    AGENT_SEARCH_ANN_MIN (hnsw, default 1,000,000), AGENT_SEARCH_ANN_PQ_MIN (ivfpq,
    default 8,000,000)."""
    env = os.environ.get("AGENT_SEARCH_ANN", "auto").lower()
    if env in ("flat", "hnsw", "ivfpq"):
        return env
    if _faiss() is None:
        return "flat"
    hnsw_min = int(os.environ.get("AGENT_SEARCH_ANN_MIN", "1000000"))
    pq_min = int(os.environ.get("AGENT_SEARCH_ANN_PQ_MIN", "8000000"))
    if n_docs >= pq_min:
        return "ivfpq"
    if n_docs >= hnsw_min:
        return "hnsw"
    return "flat"


# --- atomic write helpers (concurrent --workers must never read a half file) -

def _tmp(path: str) -> str:
    return f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"


def _atomic_json(path: str, obj) -> None:
    t = _tmp(path)
    with open(t, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(t, path)


def _atomic_npy(path: str, arr) -> None:
    import numpy as np
    t = _tmp(path)
    np.save(t + ".npy", arr)
    os.replace(t + ".npy", path)


def _atomic_faiss(path: str, index) -> None:
    faiss = _faiss()
    t = _tmp(path)
    faiss.write_index(index, t)
    os.replace(t, path)


# --- backends ---------------------------------------------------------------

class VectorIndex:
    """A built, searchable index over one corpus. Subclasses own their payload."""
    backend = "base"

    def __init__(self, doc_ids: Sequence[str]):
        self.doc_ids = list(doc_ids)

    def search(self, qv, k: int) -> list[str]:        # qv: 1-D normalized float vector
        raise NotImplementedError

    def _save_payload(self, cache_dir: str) -> None:
        raise NotImplementedError

    @classmethod
    def _load_payload(cls, cache_dir: str, doc_ids: list[str]) -> "VectorIndex":
        raise NotImplementedError


class FlatIndex(VectorIndex):
    """Exact brute-force cosine over a float16 matrix. No dependency beyond numpy."""
    backend = "flat"
    PAYLOAD = "embeddings.npy"

    def __init__(self, emb, doc_ids):
        super().__init__(doc_ids)
        self.emb = emb                                # (n, d), float16 (or legacy f32)
        self._flat_faiss = None                        # lazy exact IndexFlatIP (opt-in)
        self._flat_faiss_lock = threading.Lock()

    @classmethod
    def build(cls, emb, doc_ids):
        import numpy as np
        return cls(np.ascontiguousarray(emb, dtype=np.float16), doc_ids)

    def _faiss_flat_index(self):
        """Lazily build (once) and cache an exact faiss.IndexFlatIP over `self.emb`
        cast fp16->fp32 — the SAME stored embeddings, no new on-disk artifact. Built
        on first use (not eagerly at construction, since a FlatIndex built/saved by
        an offline indexing job may never be searched in-process) and cached for the
        object's lifetime, so the fp16->fp32 cast and index build cost is paid once,
        never per query. Returns None if faiss is unavailable (caller falls back to
        the numpy path)."""
        if self._flat_faiss is not None:
            return self._flat_faiss
        faiss = _faiss()
        if faiss is None:
            return None
        with self._flat_faiss_lock:
            if self._flat_faiss is None:               # re-check inside the lock
                import numpy as np
                emb32 = np.ascontiguousarray(self.emb, dtype=np.float32)
                index = faiss.IndexFlatIP(emb32.shape[1])
                index.add(emb32)
                self._flat_faiss = index
        return self._flat_faiss

    def search(self, qv, k):
        import numpy as np
        n = len(self.doc_ids)
        k = min(k, n)
        if k <= 0:
            return []
        if _flat_faiss_enabled():
            fidx = self._faiss_flat_index()
            if fidx is not None:
                q = np.ascontiguousarray(np.asarray(qv, dtype=np.float32)[None, :])
                _scores, ids = fidx.search(q, k)
                return [self.doc_ids[i] for i in ids[0] if i != -1]
            # faiss requested but unavailable -> fall through to the numpy path below
        qv = np.asarray(qv, dtype=np.float32)         # f16 @ f32 -> f32 (precise enough)
        scores = self.emb @ qv
        # Top-k with a FULLY DETERMINISTIC tie-break by doc_id (matches the BM25
        # ranker's (-score, doc_id) rule), so results are reproducible even when many
        # units share an identical cosine (common with duplicate/trivial embeddings).
        # argpartition gives a cheap O(n) cutoff; we then resolve the boundary exactly:
        #   - everything strictly above the cutoff is certainly in the top-k;
        #   - the remaining budget is filled from the score==cutoff tie group, chosen
        #     by smallest doc_id (deterministic, never argpartition's arbitrary order).
        part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
        cutoff = scores[part].min()
        above = np.flatnonzero(scores > cutoff).tolist()      # certainly in top-k
        if len(above) >= k:
            cand = above
        else:
            tied = sorted(np.flatnonzero(scores == cutoff).tolist(),
                          key=lambda i: self.doc_ids[i])       # ties: smallest doc_id
            cand = above + tied[: k - len(above)]
        order = sorted(cand, key=lambda i: (-float(scores[i]), self.doc_ids[i]))
        return [self.doc_ids[i] for i in order[:k]]

    def _save_payload(self, cache_dir):
        _atomic_npy(os.path.join(cache_dir, self.PAYLOAD), self.emb)

    @classmethod
    def _load_payload(cls, cache_dir, doc_ids):
        import numpy as np
        emb = np.load(os.path.join(cache_dir, cls.PAYLOAD), mmap_mode="r")
        return cls(emb, doc_ids)


class _FaissIndex(VectorIndex):
    """Shared save/load/search for FAISS-backed indexes (inner product = cosine on
    normalized vectors). Stores one `ann.faiss` payload."""
    backend = "faiss"
    PAYLOAD = "ann.faiss"

    def __init__(self, index, doc_ids):
        super().__init__(doc_ids)
        self.index = index

    def search(self, qv, k):
        import numpy as np
        k = min(k, len(self.doc_ids))
        if k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(qv, dtype=np.float32)[None, :])
        _scores, idx = self.index.search(q, k)
        return [self.doc_ids[i] for i in idx[0] if i != -1]

    def _save_payload(self, cache_dir):
        _atomic_faiss(os.path.join(cache_dir, self.PAYLOAD), self.index)

    @classmethod
    def _load_payload(cls, cache_dir, doc_ids):
        faiss = _faiss()
        index = faiss.read_index(os.path.join(cache_dir, cls.PAYLOAD))
        return cls(index, doc_ids)


class HnswIndex(_FaissIndex):
    """Graph ANN — sub-ms search at million-doc scale, exact float32 storage."""
    backend = "hnsw"

    @classmethod
    def build(cls, emb, doc_ids):
        import numpy as np
        faiss = _faiss()
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(emb)
        index.hnsw.efSearch = 128
        return cls(index, doc_ids)


def _pq_subquantizers(d: int) -> int:
    for m in (96, 64, 48, 32, 24, 16, 8, 4, 2, 1):
        if d % m == 0:
            return m
    return 1


class IvfpqIndex(_FaissIndex):
    """Product-quantized IVF — COMPRESSED storage for huge/memory-bound corpora."""
    backend = "ivfpq"

    @classmethod
    def build(cls, emb, doc_ids):
        import numpy as np
        faiss = _faiss()
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        n, d = emb.shape
        if n < 256:                       # PQ (nbits=8) trains 256 centroids/subquantizer
            raise ValueError(f"ivfpq needs >=256 vectors to train, got {n}")
        nlist = max(1, min(int(4 * (n ** 0.5)), max(1, n // 39)))  # train needs ~39/centroid
        m = _pq_subquantizers(d)
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quant, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.add(emb)
        index.nprobe = min(32, nlist)
        return cls(index, doc_ids)


_BACKENDS = {b.backend: b for b in (FlatIndex, HnswIndex, IvfpqIndex)}


# --- build / persist / load (meta.json is the last-written sentinel) ---------

def build_index(emb, doc_ids: Sequence[str], backend: Optional[str] = None) -> VectorIndex:
    backend = backend or choose_backend(len(doc_ids))
    if backend not in _BACKENDS:
        raise ValueError(f"unknown vector backend {backend!r}")
    if backend != "flat" and _faiss() is None:
        backend = "flat"                              # no faiss -> degrade, never crash
    try:
        return _BACKENDS[backend].build(emb, list(doc_ids))
    except Exception as e:
        if backend == "flat":
            raise
        import sys
        print(f"  [vector_index] {backend} build failed ({type(e).__name__}: {e}); "
              f"falling back to exact flat index", file=sys.stderr, flush=True)
        return FlatIndex.build(emb, list(doc_ids))    # any ANN failure -> exact flat


def save_index(index: VectorIndex, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    _atomic_json(os.path.join(cache_dir, "doc_ids.json"), index.doc_ids)
    index._save_payload(cache_dir)
    # meta LAST: a reader that sees meta.json is guaranteed the rest is complete.
    _atomic_json(os.path.join(cache_dir, "meta.json"),
                 {"backend": index.backend, "n": len(index.doc_ids), "format": 2})


def load_index(cache_dir: str) -> Optional[VectorIndex]:
    ids_path = os.path.join(cache_dir, "doc_ids.json")
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(ids_path):
        return None
    with open(ids_path) as fh:
        doc_ids = json.load(fh)
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            backend = json.load(fh).get("backend", "flat")
        cls = _BACKENDS.get(backend, FlatIndex)
        if cls is not FlatIndex and _faiss() is None:
            return None                               # ANN payload but no faiss -> rebuild flat
        return cls._load_payload(cache_dir, doc_ids)
    # legacy cache (pre-backends): a bare float32 embeddings.npy is a flat index.
    if os.path.exists(os.path.join(cache_dir, FlatIndex.PAYLOAD)):
        return FlatIndex._load_payload(cache_dir, doc_ids)
    return None


def index_exists(cache_dir: str) -> bool:
    """True iff a COMPLETE persisted index is present — without loading its payload.

    Mirrors ``load_index``'s success condition so a caller can SKIP rebuilding an
    already-indexed corpus (the unit parse + encode) cheaply: meta.json is written last
    (see save_index), so its presence guarantees doc_ids + payload are complete. Returns
    False for an ANN index when faiss is unavailable, exactly as load_index would (it
    rebuilds flat in that case)."""
    if not os.path.exists(os.path.join(cache_dir, "doc_ids.json")):
        return False
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as fh:
                backend = json.load(fh).get("backend", "flat")
        except Exception:
            return False
        cls = _BACKENDS.get(backend, FlatIndex)
        return cls is FlatIndex or _faiss() is not None
    return os.path.exists(os.path.join(cache_dir, FlatIndex.PAYLOAD))  # legacy flat cache
