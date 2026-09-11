"""The in-memory BM25 scorer for code repositories.

This is the one in-memory scorer in the library, and it exists for one corpus kind: a code
repository, which is small and can change while an agent works on it. The grep ranker
(`grep.py`) and the code Boolean executor (`bql/executor.py`) rank with it. Document corpora
never use it: their BM25 is Lucene (`pyserini.py`), and their structured queries run on the
Lucene structured index (`agent_search/retrievers/lucene/`).

The index is updated, not rebuilt: `add` and `remove` keep the document frequencies, lengths
and average length consistent, so a caller that notices a changed file replaces that file's
units and leaves the rest of the index alone. The tokenizer is
`agent_search.corpus.units.code_tokenize`.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

from agent_search.corpus.units import code_tokenize


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self._tfs: dict[str, Counter] = {}     # doc_id -> term frequencies
        self._lens: dict[str, int] = {}        # doc_id -> token count
        self._df: Counter = Counter()          # term -> number of docs holding it
        self._total_len: int = 0

    # --- building and updating ---------------------------------------------------------
    def index(self, docs: Mapping[str, str]) -> "BM25":
        """Replace the whole index with `docs` (doc_id -> text)."""
        return self.index_tokenized({d: code_tokenize(t) for d, t in docs.items()})

    def index_tokenized(self, docs: Mapping[str, Sequence[str]]) -> "BM25":
        """Replace the whole index with pre-tokenized docs (doc_id -> tokens)."""
        self._tfs, self._lens, self._df, self._total_len = {}, {}, Counter(), 0
        self.add_tokenized(docs)
        return self

    def add(self, docs: Mapping[str, str]) -> "BM25":
        return self.add_tokenized({d: code_tokenize(t) for d, t in docs.items()})

    def add_tokenized(self, docs: Mapping[str, Sequence[str]]) -> "BM25":
        """Add documents. A doc_id already in the index is replaced."""
        self.remove(d for d in docs if d in self._tfs)
        for doc_id, toks in docs.items():
            tf = Counter(toks)
            self._tfs[doc_id] = tf
            self._lens[doc_id] = len(toks)
            self._total_len += len(toks)
            self._df.update(tf.keys())
        return self

    def remove(self, doc_ids: Iterable[str]) -> "BM25":
        """Drop documents; unknown ids are ignored."""
        for doc_id in list(doc_ids):
            tf = self._tfs.pop(doc_id, None)
            if tf is None:
                continue
            self._total_len -= self._lens.pop(doc_id)
            for term in tf:
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]
        return self

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._tfs

    def __len__(self) -> int:
        return len(self._tfs)

    # --- scoring --------------------------------------------------------------------------
    @property
    def _avgdl(self) -> float:
        return (self._total_len / len(self._tfs)) if self._tfs else 0.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        n = len(self._tfs)
        # Lucene-style non-negative idf: log(1 + (N - df + 0.5)/(df + 0.5))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _score_doc(self, doc_id: str, terms: Sequence[str]) -> float:
        tf, dl = self._tfs[doc_id], self._lens[doc_id]
        avgdl = self._avgdl or 1.0
        score = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score += self._idf(term) * (f * (self.k1 + 1)) / denom
        return score

    def score_terms(self, terms: Sequence[str]) -> list[tuple[str, float]]:
        """Score every indexed doc against `terms`, keeping zero scores, sorted by
        (-score, doc_id). A Boolean candidate matched through file scope or its qualified
        name shares no term with the query and must still appear in the ranking."""
        scored = [(d, self._score_doc(d, terms)) for d in self._tfs]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def score_subset(self, terms: Sequence[str],
                     doc_ids: Sequence[str]) -> list[tuple[str, float]]:
        """Score only `doc_ids` against the whole index's statistics, sorted by
        (-score, doc_id). Ids not in the index keep a 0.0 score and stay ranked."""
        scored = [(d, self._score_doc(d, terms) if d in self._tfs else 0.0) for d in doc_ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def search_with_count(self, query: str, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """Top-k plus the untruncated number of docs with a positive score."""
        scored = [(d, s) for d, s in self.score_terms(code_tokenize(query)) if s > 0]
        return scored[:k], len(scored)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return self.search_with_count(query, k=k)[0]
