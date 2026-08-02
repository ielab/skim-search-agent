"""DenseBelief: a dense-embedding belief source for the Indri graded retrieval engine
(`agent_search/retrievers/structural/indri/model.py`). OFF by default — `IndriExecutor`
only touches this module's API when a `DenseBelief` instance is explicitly attached
(`dense=...` / `attach_dense(...)`); see that module's Deviations section for the
belief-combination math and pool-expansion design this class feeds.

Motivation (measured): BrowseComp-style questions are paraphrased/obfuscated, so pure
lexical scoring (Indri QL, same as BM25 in this respect) misses documents whose wording
differs from the clue even though their MEANING matches. A dense encoder closes that gap
— but only if it's wired into the SAME candidate pool + ranking pass, not a separate
top-k merged post-hoc (which would need a second, uncalibrated fusion step outside the
belief-combination semantics `model.py` otherwise guarantees for every other operator).

Reuse, not reimplementation: this class is a THIN wrapper around
`agent_search.retrievers.dense.dense.DenseRetriever` — the SAME encoder, SAME per-model
query-prefix table, SAME `indexes/dense/<model>-sl<len>/<corpus_key>/` cache layout the
existing `dense` retriever condition already builds/persists (default doc model:
BAAI/bge-base-en-v1.5, matching `indexes/dense/BAAI__bge-base-en-v1.5-sl1024/`). All
encode+cache logic (`DenseRetriever.index`) and the shared-encoder/device-selection
machinery (`DenseRetriever._encoder`) are imported and called, never duplicated. The one
piece `DenseRetriever`'s public API does NOT expose — per-doc raw cosine similarity for
an arbitrary POOL of doc_ids (`.search()` only returns an ordered doc_id list, no
scores) — is wrapped minimally below (`_similarities`), by reading the already-built
`FlatIndex`'s `.emb` matrix directly rather than re-encoding or reimplementing anything
encoder-side.

Query text handling: `IndriExecutor` passes the RAW query surface text (e.g.
`#combine( #1(exact phrase) name.title #date:between(2002-01-01 2002-12-31) )`), not a
plain-English sentence — the dense encoder was never trained on Indri query-language
syntax. `_plain_terms`/`_plain_text` strip `#operator` tokens and `.field` suffixes
before encoding, mirroring `agent_search.agent.tools.doc_indri._indri_query_terms`
(kept as a small standalone copy here rather than a cross-package import, since the
`indri` package otherwise has no dependency on the `agent` package's tool-workspace
layer — see Deviations below).
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.dense.dense import DenseRetriever, _QUERY_PREFIX

# Env `DENSE_MODEL` overrides the default doc embedder for every DenseBelief-based arm
# (Indri's INDRI_DENSE belief, and the densevisit/densefetch baselines via
# agent_search.agent.retriever's DENSE_BASELINE_MODEL import) — same knob, same rationale,
# as evaluation.datasets.default_dense_model (this package is always general-domain, so no
# code-vs-general split is needed here). Unset -> unchanged default.
DEFAULT_MODEL = os.environ.get("DENSE_MODEL") or "BAAI/bge-base-en-v1.5"
DEFAULT_TOP_K = 50

# --- query text -> plain terms, stripping Indri operator syntax --------------
# Mirrors `agent_search.agent.tools.doc_indri._indri_query_terms` byte-for-byte in
# behavior (same three regexes, same order) — duplicated rather than imported (see
# module Deviations: this package doesn't otherwise depend on `agent_search.agent`).
_OP_TOKEN_RE = re.compile(r"#[\w:]+")
_FIELD_SUFFIX_RE = re.compile(r"\.[A-Za-z]+\b")
_PUNCT_RE = re.compile(r'[#().{}<>"]')


def _plain_terms(query: str) -> list:
    """Plain word tokens of an Indri query string, `#operator` names and `.field`
    suffixes stripped — e.g. `'#combine( #1(bank management) treaty.title )'` ->
    `['bank', 'management', 'treaty']`. Corpus-free (operates only on the query
    text itself)."""
    q = _OP_TOKEN_RE.sub(" ", query or "")
    q = _FIELD_SUFFIX_RE.sub("", q)
    q = _PUNCT_RE.sub(" ", q)
    return [t for t in q.split() if t]


def _plain_text(query: str) -> str:
    """Space-joined `_plain_terms(query)` — the RAW text handed to the dense
    encoder (an Indri query-language string is never a natural-language sentence
    the encoder was trained on)."""
    return " ".join(_plain_terms(query))


class DenseBelief:
    """Dense-embedding belief source for `IndriExecutor`. Two entry points:

    - `top_k_doc_ids(query_text, k)` — dense top-K doc_ids for POOL EXPANSION
      (recall: surfaces paraphrased docs with zero lexical term overlap).
    - `score(query_text, doc_ids)` — ONE batched query encode -> raw cosine
      similarity (`[-1, 1]`, embeddings are L2-normalized) for the given doc_ids
      (or every indexed doc if `doc_ids=None`), for belief COMBINATION.

    `build_or_load(units, key)` reuses `DenseRetriever.index` verbatim (same
    encode+cache path/format the `dense` retriever condition already uses) — call
    it once per corpus before `score`/`top_k_doc_ids`.
    """

    def __init__(self, model: str = DEFAULT_MODEL, index_root: str = "indexes",
                 encoder=None, device: Optional[str] = None,
                 max_seq_length: int = 1024, top_k: int = DEFAULT_TOP_K):
        self._retriever = DenseRetriever(
            model=model, index_root=index_root, encoder=encoder, device=device,
            max_seq_length=max_seq_length)
        self.top_k = top_k
        self._doc_id_to_row: Optional[dict] = None
        # memo of the LAST query's (plain_text, vector): `IndriExecutor` calls
        # `top_k_doc_ids` (pool expansion) then `score` (belief combination) with
        # the SAME query text within one search pass — the memo keeps that to ONE
        # encoder call per search (asserted in tests/test_indri_dense.py).
        self._q_cache: Optional[tuple] = None

    # --- build/load (delegates entirely to DenseRetriever.index) -----------------

    def build_or_load(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "DenseBelief":
        """Load the persisted per-corpus embeddings if present, else encode+persist
        (SAME cache dir the `dense` retriever condition uses:
        `indexes/dense/<model>-sl<len>/<key>/`) — delegates the actual encode+cache
        work to `DenseRetriever.index` untouched."""
        self._retriever.index(units, key=key)
        self._doc_id_to_row = {d: i for i, d in enumerate(self._retriever._doc_ids)}
        return self

    def is_ready(self) -> bool:
        return self._retriever._index is not None

    # --- pool expansion (recall) --------------------------------------------------

    def top_k_doc_ids(self, query_text: str, k: Optional[int] = None) -> list:
        """Dense top-K doc_ids for `query_text` (operator-stripped, see
        `_plain_text`) — nearest-neighbour search via the SAME `VectorIndex.search`
        `DenseRetriever.search` uses (exact for the default `flat` backend; ANN for
        `hnsw`/`ivfpq`), but through the memoized query vector so one
        `IndriExecutor` search pass (expansion + `score`) encodes the query ONCE."""
        if self._retriever._index is None:
            return []
        qv = self._encode_query_vector(_plain_text(query_text))
        return self._retriever._index.search(qv, k or self.top_k)

    # --- belief combination (precision) -------------------------------------------

    def score(self, query_text: str, doc_ids: Optional[Sequence[str]] = None) -> dict:
        """ONE batched query encode -> `{doc_id: cosine_similarity}` for `doc_ids`
        (or every indexed doc if `None`). `cosine_similarity` is raw (`[-1, 1]`,
        unnormalized across the pool — `IndriExecutor._combine_dense` does the
        pool-relative min-max normalization, per `model.py`'s Deviations)."""
        if self._retriever._index is None:
            return {}
        qv = self._encode_query_vector(_plain_text(query_text))
        return self._similarities(qv, doc_ids)

    # --- internals -----------------------------------------------------------------

    def _encode_query_vector(self, plain_text: str):
        """Mirrors `DenseRetriever.search`'s own encode block exactly (same query
        prefix table, same shared-encoder lock) — the ONE minimal duplication this
        wrapper needs, since `DenseRetriever.search` returns only ranked doc_ids,
        never the raw query vector `score()` requires for arbitrary-pool cosine
        similarity (see module docstring). Memoized on the last query text (see
        `_q_cache`) so expansion + scoring within one search pass encode once."""
        if self._q_cache is not None and self._q_cache[0] == plain_text:
            return self._q_cache[1]
        q = _QUERY_PREFIX.get(self._retriever.model_id, "") + plain_text
        model = self._retriever._encoder()
        lock = getattr(model, "_agent_search_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        finally:
            if lock is not None:
                lock.release()
        self._q_cache = (plain_text, qv)
        return qv

    def _similarities(self, qv, doc_ids) -> dict:
        idx = self._retriever._index
        emb = getattr(idx, "emb", None)
        if emb is None:
            # Non-flat (ANN/FAISS) backend: `VectorIndex`'s public API exposes no
            # per-doc raw-vector accessor (see agent_search/retrievers/dense/
            # vector_index.py), so an exact cosine isn't available without reaching
            # into FAISS internals. Documented, minimal fallback below — in
            # practice unreachable for this repo's current corpora
            # (browsecomp_plus_structured / hotpotqa_structured are both well
            # under `choose_backend`'s ~1M-doc ANN threshold, so they always use
            # `FlatIndex`, which has `.emb`).
            return self._similarities_via_search(qv, doc_ids)
        import numpy as np
        qv = np.asarray(qv, dtype=np.float32)
        row_of = self._doc_id_to_row or {d: i for i, d in enumerate(idx.doc_ids)}
        ids = list(doc_ids) if doc_ids is not None else idx.doc_ids
        found = [d for d in ids if d in row_of]
        if not found:
            return {}
        rows = [row_of[d] for d in found]
        sub = np.asarray(emb[rows], dtype=np.float32)          # (m, d), one batched matmul
        scores = sub @ qv
        return dict(zip(found, (float(s) for s in scores)))

    def _similarities_via_search(self, qv, doc_ids) -> dict:
        """ANN-backend fallback: a rank-based proxy score in `[0, 1]` (NOT a true
        cosine similarity) derived from `VectorIndex.search`'s ranked doc_id order.
        See `_similarities`'s docstring — unreachable for this repo's current
        corpora, kept only so a future large (ANN-backed) corpus degrades to
        *something* usable rather than raising."""
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
