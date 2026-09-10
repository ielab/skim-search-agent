"""Index-free grep baseline (GrepRAG-style) — the fair comparison for SkimSearchAgent.

Both this and the structural SkimSearchAgent method are
**index-free**, so comparing them isolates the value of *structure* (Boolean + AST)
rather than "index vs no-index".

Two literature-verified grep semantics, selected by the query's surface form:
- UNQUOTED query → GrepRAG (Wang et al. 2026, arXiv 2601.23254 — code completion):
  emit keyword patterns → grep the live files (exact substring per keyword, ANY may
  match) → **BM25-rerank the matched candidates** ("identifier-weighted
  re-ranking"). Candidate SELECTION is index-free (the live substring scan); the
  rerank uses the SHARED corpus-wide BM25 scorer (built once, reused) — the same
  ranking layer the structural executor uses, so grep and BQL rank by identical
  corpus statistics and differ ONLY in selection (see ranking.py / score_subset).
- QUOTED query → SWE-agent's search_dir semantics (verified against the repo:
  tools/search/bin/search_dir): the whole term is ONE literal substring passed to
  grep verbatim. (SWE-agent renders `<file> (N matches)` capped at 100 files with a
  "narrow your search" nudge — our shared renderer mirrors the cap/nudge.)

This is the *flat* grep baseline — keyword matching, no Boolean logic, no AST scope
(which is what the method adds).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.lexical.scorer import BM25
from agent_search.core.interfaces import Retriever

# minimal stop set so keyword patterns are discriminative identifiers, not glue words
_STOP = {
    "the", "a", "an", "is", "are", "be", "to", "of", "and", "or", "in", "on", "for",
    "with", "should", "would", "that", "this", "it", "its", "so", "not", "no", "but",
    "if", "when", "then", "as", "at", "by", "from", "was", "were", "has", "have", "had",
    "def", "return", "self", "none", "true", "false", "class", "import", "you", "we",
}


class GrepBaseline(Retriever):
    name = "grep"

    def __init__(self, max_patterns: int = 10):
        self.max_patterns = max_patterns          # GrepRAG uses m=10 patterns
        self._units: list[CodeUnit] = []
        self._haystacks: list[str] = []           # per-unit "qualname code", lowercased ONCE
        self._bm: Optional[BM25] = None           # corpus-wide rerank scorer, built once

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "GrepBaseline":
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the grep baseline", "a retriever with prebuilt-index support")
        # NO retrieval index — but two things are STATIC across queries, so do them once:
        # (1) the per-unit searchable text, lowercased here instead of rebuilding
        #     `f"{qualname} {code}".lower()` for every unit on every query (the grep scan);
        # (2) the BM25 rerank scorer (corpus stats don't change per query) — built lazily
        #     on first query and reused, NOT rebuilt over each candidate set every call.
        # Rebuilding (2) per query re-tokenized thousands of big docs each turn, which on a
        # large shared document corpus (hotpotqa/2wiki/musique) was the minutes-per-query tail.
        self._units = list(units)
        self._haystacks = [f"{u.qualname} {u.code}".lower() for u in self._units]
        self._bm = None
        return self

    def _corpus_bm(self) -> BM25:
        """BM25 over the WHOLE corpus, built once and reused (mirrors the structural
        executor's _corpus_bm). Candidate SELECTION stays index-free (the live grep
        scan); only the shared SCORING layer is prebuilt, so grep and BQL rank by the
        same corpus idf and differ ONLY in selection."""
        if self._bm is None:
            self._bm = BM25().index(
                {u.doc_id: f"{u.qualname} {u.code}" for u in self._units})
        return self._bm

    def _keywords(self, query: str) -> list[str]:
        q = query.strip()
        # Quoted query -> ONE literal substring pattern (SWE-agent's search_dir
        # semantics: the term goes to grep verbatim). Unquoted -> keyword
        # patterns (GrepRAG semantics). Mirrors how people actually use grep.
        if len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'":
            literal = q[1:-1].strip().lower()
            return [literal] if literal else []
        toks = [t for t in code_tokenize(query) if t not in _STOP and len(t) > 2]
        return [w for w, _ in Counter(toks).most_common(self.max_patterns)]

    def search(self, query: str, k: int) -> list[str]:
        return [doc_id for doc_id, _ in self.search_with_scores(query, k=k)]

    def search_with_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_with_count(query, k=k)[0]

    def search_with_count(self, query: str, k: int) -> tuple[list[tuple[str, float]], int]:
        """Top-k reranked candidates plus the UNTRUNCATED candidate count (one
        scan — count and candidates from the same pass)."""
        keywords = self._keywords(query)
        if not keywords:
            return [], 0
        # ripgrep exact-substring emulation over the pre-lowercased haystacks (ANY keyword matches)
        candidates = [u.doc_id for u, hay in zip(self._units, self._haystacks)
                      if any(kw in hay for kw in keywords)]
        if not candidates:
            return [], 0
        # GrepRAG identifier-weighted re-ranking: score the candidate set with the
        # corpus-wide BM25 (built once), reading off precomputed per-doc stats + corpus
        # idf instead of rebuilding an index over the candidates every query.
        ranked = self._corpus_bm().score_subset(code_tokenize(query), candidates)
        return ranked[:k], len(candidates)


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("grep")
def _build_grep(cfg: RetrieverConfig, name: str):
    return lambda: GrepBaseline()
