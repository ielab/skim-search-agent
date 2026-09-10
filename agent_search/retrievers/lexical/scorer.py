"""The pure-Python BM25 scorer (no Java, no dependencies).

Used by `BM25Local` (the in-memory BM25 retriever), by the grep baseline, and by the BQL
executor to order the candidate set a Boolean query selects, so every index-free condition ranks
with the same scorer and differs only in how it picks candidates. The tokenizer is
`agent_search.corpus.units.code_tokenize`. For Lucene's BM25 use `pyserini.py`.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

from agent_search.corpus.units import code_tokenize  # shared identifier-aware tokenizer


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0
        self._pos: dict[str, int] = {}

    def index(self, docs: Mapping[str, str]) -> "BM25":
        return self.index_tokenized({d: code_tokenize(t) for d, t in docs.items()})

    def index_tokenized(self, docs: Mapping[str, Sequence[str]]) -> "BM25":
        """Index pre-tokenized docs (callers that already hold token lists —
        e.g. the structural executor — skip re-tokenization)."""
        self._doc_ids, self._tfs, self._lens = [], [], []
        self._df = Counter()
        for doc_id, toks in docs.items():
            tf = Counter(toks)
            self._doc_ids.append(doc_id)
            self._tfs.append(tf)
            self._lens.append(len(toks))
            self._df.update(tf.keys())
        self._n = len(self._doc_ids)
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0
        self._pos = {d: i for i, d in enumerate(self._doc_ids)}  # doc_id -> row
        return self

    def _score_doc(self, i: int, terms: Sequence[str]) -> float:
        tf, dl = self._tfs[i], self._lens[i]
        score = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score += self._idf(term) * (f * (self.k1 + 1)) / denom
        return score

    def score_terms(self, terms: Sequence[str]) -> list[tuple[str, float]]:
        """Score EVERY indexed doc against `terms`, keeping zero scores.

        Set->ranking use: the docs are a Boolean candidate set, so membership is
        already decided — a candidate whose own text shares no term with the
        query (matched via file scope / qualname) must stay in the ranking, not
        vanish the way `search`'s score>0 filter would make it. Sorted by
        (-score, doc_id): deterministic."""
        scored = [(d, self._score_doc(i, terms)) for i, d in enumerate(self._doc_ids)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def score_subset(self, terms: Sequence[str],
                     doc_ids: Sequence[str]) -> list[tuple[str, float]]:
        """Score only `doc_ids` (a Boolean candidate set) against THIS index's
        corpus statistics — the scorer is built once over the whole corpus and a
        query just reads off the precomputed per-doc tf/len + corpus idf for its
        candidates, instead of rebuilding a fresh per-candidate-set index every
        call (which made a broad query over a large corpus re-tokenize hundreds of
        big docs each turn). Candidates absent from the corpus, or matched via
        file/qualname scope with no shared term, keep a 0.0 score and stay ranked.
        Sorted (-score, doc_id): deterministic."""
        scored = [(d, self._score_doc(self._pos[d], terms) if d in self._pos else 0.0)
                  for d in doc_ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # Lucene-style non-negative idf: log(1 + (N - df + 0.5)/(df + 0.5))
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search_with_count(self, query: str, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """Top-k plus the UNTRUNCATED number of matching (score>0) docs — so an
        agent observation can report the true match count, consistent with the
        grep and BQL tools, instead of a k-capped one."""
        q_terms = code_tokenize(query)
        scored = [(d, s) for d, s in self.score_terms(q_terms) if s > 0]
        return scored[:k], len(scored)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        q_terms = code_tokenize(query)
        scored: list[tuple[str, float]] = []
        for i, doc_id in enumerate(self._doc_ids):
            tf, dl = self._tfs[i], self._lens[i]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
