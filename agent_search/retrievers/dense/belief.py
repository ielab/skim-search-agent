"""DenseBelief: the dense engine every agent arm shares.

One `DenseRetriever` (the run's `dense_model`, any family) wrapped for two uses: `top_k_doc_ids`
for nearest-neighbour candidates (the dense search arms, Sieve's dense fusion and its fallback,
the hybrid baselines) and `score` for a cosine similarity per document (Indri's dense belief).
The query goes through the episode's history context first (`agent_search.training.history.current_query_for`),
so a history-conditioned retriever sees the query it was trained on. The embeddings come from
the same persisted cache the `dense` retriever builds; nothing is encoded during a run.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.dense.base import DenseRetriever
from agent_search.training.history import current_query_for

DEFAULT_TOP_K = 50

_OP_TOKEN_RE = re.compile(r"#[\w:]+")
_FIELD_SUFFIX_RE = re.compile(r"\.[A-Za-z]+\b")
_PUNCT_RE = re.compile(r'[#().{}<>"]')


def default_model() -> str:
    """The run's dense model when none is given: the general-domain default (`DENSE_MODEL`)."""
    from agent_search.evaluation.datasets import default_dense_model
    return default_dense_model("general")


def _plain_terms(query: str) -> list:
    """Plain word tokens of an Indri query string, `#operator` names and `.field` suffixes
    stripped: `'#combine( #1(bank management) treaty.title )'` -> `['bank', 'management', 'treaty']`."""
    q = _OP_TOKEN_RE.sub(" ", query or "")
    q = _FIELD_SUFFIX_RE.sub("", q)
    q = _PUNCT_RE.sub(" ", q)
    return [t for t in q.split() if t]


def _plain_text(query: str) -> str:
    """The text handed to the encoder: a query-language string is not a sentence it was trained on."""
    return " ".join(_plain_terms(query))


class DenseBelief:
    """`top_k_doc_ids(query, k)` for candidates; `score(query, doc_ids)` for cosine similarities.
    Call `build_or_load(units, key)` once per corpus first."""

    def __init__(self, model: Optional[str] = None, index_root: str = "indexes",
                 encoder=None, device: Optional[str] = None,
                 max_seq_length: int = 1024, top_k: int = DEFAULT_TOP_K):
        self._retriever = DenseRetriever(
            model=model or default_model(), index_root=index_root, encoder=encoder, device=device,
            max_seq_length=max_seq_length)
        self.top_k = top_k
        self._doc_id_to_row: Optional[dict] = None
        self._q_cache: Optional[tuple] = None

    def build_or_load(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "DenseBelief":
        """Load the persisted per-corpus embeddings if present, else encode and persist (the
        same cache the `dense` retriever uses)."""
        self._retriever.index(units, key=key)
        ids = self._retriever._doc_ids
        # `score()` needs doc_id -> row; skipped at the millions scale, `_similarities` builds it on demand
        self._doc_id_to_row = {d: i for i, d in enumerate(ids)} if len(ids) < 2_000_000 else None
        return self

    def is_ready(self) -> bool:
        return self._retriever._index is not None

    def top_k_doc_ids(self, query_text: str, k: Optional[int] = None) -> list:
        """Nearest neighbours of the (operator-stripped) query, through the history context."""
        if self._retriever._index is None:
            return []
        qv = self._encode_query_vector(current_query_for(_plain_text(query_text)))
        return self._retriever._index.search(qv, k or self.top_k)

    def score(self, query_text: str, doc_ids: Optional[Sequence[str]] = None) -> dict:
        """`{doc_id: cosine similarity}` for `doc_ids` (every indexed document if None); raw
        values in [-1, 1], normalised across the pool by the caller."""
        if self._retriever._index is None:
            return {}
        qv = self._encode_query_vector(current_query_for(_plain_text(query_text)))
        return self._similarities(qv, doc_ids)

    def _encode_query_vector(self, plain_text: str):
        """The query vector through the retriever's own prefix, encoder, lock and query length;
        memoised on the last query so one search pass (expansion plus scoring) encodes once."""
        if self._q_cache is not None and self._q_cache[0] == plain_text:
            return self._q_cache[1]
        qv = self._retriever.encode_query(plain_text)
        self._q_cache = (plain_text, qv)
        return qv

    def _similarities(self, qv, doc_ids) -> dict:
        idx = self._retriever._index
        emb = getattr(idx, "emb", None)
        if emb is None:
            return self._similarities_via_search(qv, doc_ids)
        import numpy as np
        qv = np.asarray(qv, dtype=np.float32)
        row_of = self._doc_id_to_row or {d: i for i, d in enumerate(idx.doc_ids)}
        ids = list(doc_ids) if doc_ids is not None else idx.doc_ids
        found = [d for d in ids if d in row_of]
        if not found:
            return {}
        rows = [row_of[d] for d in found]
        sub = np.asarray(emb[rows], dtype=np.float32)          # one batched matmul
        scores = sub @ qv
        sims = dict(zip(found, (float(s) for s in scores)))
        missing = [d for d in ids if d not in row_of]
        if missing:
            floor = min(sims.values())
            for d in missing:
                sims[d] = floor                                 # never drop a candidate
        return sims

    def _similarities_via_search(self, qv, doc_ids) -> dict:
        """ANN backends keep no embedding matrix: a rank-based proxy in [0, 1] from the ranked
        order, so a large corpus degrades to something usable rather than raising."""
        idx = self._retriever._index
        n = len(idx.doc_ids)
        if n == 0:
            return {}
        ranked = idx.search(qv, n)
        want = set(doc_ids) if doc_ids is not None else None
        denom = max(n - 1, 1)
        out = {}
        for rank, d in enumerate(ranked):
            if want is not None and d not in want:
                continue
            out[d] = 1.0 - (rank / denom)
        return out


__all__ = ["DenseBelief", "DEFAULT_TOP_K", "default_model", "_plain_terms", "_plain_text"]
