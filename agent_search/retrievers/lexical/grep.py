"""Index-free grep baseline (GrepRAG-style), the fair comparison for SkimSearchAgent.

Both this and the BQL structural method are index-free, so comparing them isolates the
value of structure (Boolean + AST) rather than "index vs no-index".

Two literature-verified grep semantics, selected by the query's surface form:
- Unquoted query: GrepRAG (Wang et al. 2026, arXiv 2601.23254, code completion). Emit
  keyword patterns, grep the live files (exact substring per keyword, any pattern may
  match), then BM25-rerank the matched candidates ("identifier-weighted re-ranking").
  Candidate selection is index-free (the live substring scan); the rerank uses the
  shared corpus-wide BM25 scorer (built once, reused), the same ranking layer
  `StructuralExecutor._corpus_bm` uses, so grep and BQL rank by identical corpus
  statistics and differ only in selection (see `scorer.py`'s `score_subset`).
- Quoted query: SWE-agent's search_dir semantics (verified against the repo:
  tools/search/bin/search_dir). The whole term is one literal substring passed to grep
  verbatim. SWE-agent renders `<file> (N matches)` capped at 100 files with a "narrow
  your search" nudge; the shared renderer here mirrors that cap and nudge.

This is the flat grep baseline: keyword matching, no Boolean logic, no AST scope (which
is what the BQL method adds).
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
        self._haystacks: list[str] = []           # per-unit "qualname code", lowercased once
        self._bm: Optional[BM25] = None           # corpus-wide rerank scorer, built once

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "GrepBaseline":
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the grep baseline", "a retriever with prebuilt-index support")
        # There is no retrieval index, but two things stay static across queries, so build
        # them once here:
        # (1) the per-unit searchable text, lowercased here instead of rebuilding
        #     `f"{qualname} {code}".lower()` for every unit on every query (the grep scan);
        # (2) the BM25 rerank scorer (corpus stats don't change per query), built lazily
        #     on the first query and reused rather than rebuilt over each candidate set.
        # Rebuilding (2) per query re-tokenizes thousands of big docs each turn; on a large
        # shared document corpus (hotpotqa/2wiki/musique) that was the minutes-per-query tail.
        self._units = list(units)
        self._haystacks = [f"{u.qualname} {u.code}".lower() for u in self._units]
        self._bm = None
        return self

    def _corpus_bm(self) -> BM25:
        """BM25 over the whole corpus, built once and reused, the same role
        `StructuralExecutor._corpus_bm` plays for BQL. Candidate selection stays
        index-free (the live grep scan); only the shared scoring layer is prebuilt, so
        grep and BQL rank by the same corpus idf and differ only in selection."""
        if self._bm is None:
            self._bm = BM25().index(
                {u.doc_id: f"{u.qualname} {u.code}" for u in self._units})
        return self._bm

    def _keywords(self, query: str) -> list[str]:
        q = query.strip()
        # A quoted query becomes one literal substring pattern (SWE-agent's search_dir
        # semantics: the term goes to grep verbatim); an unquoted one becomes keyword
        # patterns (GrepRAG semantics). This matches how people actually use grep.
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
        """Top-k reranked candidates plus the untruncated candidate count: one scan
        produces both, so the count reflects the same candidates that were ranked."""
        keywords = self._keywords(query)
        if not keywords:
            return [], 0
        # ripgrep exact-substring emulation over the pre-lowercased haystacks (any keyword matches)
        candidates = [u.doc_id for u, hay in zip(self._units, self._haystacks)
                      if any(kw in hay for kw in keywords)]
        if not candidates:
            return [], 0
        # GrepRAG identifier-weighted re-ranking: score the candidate set with the
        # corpus-wide BM25 (built once), reading off precomputed per-doc stats and corpus
        # idf instead of rebuilding an index over the candidates every query.
        ranked = self._corpus_bm().score_subset(code_tokenize(query), candidates)
        return ranked[:k], len(candidates)


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("grep")
def _build_grep(cfg: RetrieverConfig, name: str):
    return lambda: GrepBaseline()
