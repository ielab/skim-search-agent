"""The grep ranker for code repositories (GrepRAG-style), the index-free floor the Boolean
code method is compared against.

Two grep semantics, chosen by the query's surface form:
- Unquoted query: GrepRAG (Wang et al. 2026, arXiv 2601.23254). Turn the query into keyword
  patterns, scan the live files for exact substrings (any pattern may match), then rerank
  the matched units with BM25 ("identifier-weighted re-ranking").
- Quoted query: SWE-agent's search_dir. The whole term is one literal substring.

Candidate selection is index-free (the substring scan). The rerank uses the in-memory
code scorer (`scorer.py`), the same one the code Boolean executor ranks with, so grep and
BQL rank by identical corpus statistics and differ only in selection.

The scorer is an index that follows the repository: `index(units)` builds it the first
time and, on every later call, re-indexes only the files whose units changed (a file's
fingerprint is the hash of its units' ids and code). A task that edits files calls
`index` again with the fresh units, and nothing else is rebuilt. This is the one place
the library keeps an in-memory scorer; document corpora rank on Lucene.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.lexical.scorer import BM25
from agent_search.retrievers.base import Retriever

# minimal stop set so keyword patterns are discriminative identifiers, not glue words
_STOP = {
    "the", "a", "an", "is", "are", "be", "to", "of", "and", "or", "in", "on", "for",
    "with", "should", "would", "that", "this", "it", "its", "so", "not", "no", "but",
    "if", "when", "then", "as", "at", "by", "from", "was", "were", "has", "have", "had",
    "def", "return", "self", "none", "true", "false", "class", "import", "you", "we",
}


def file_fingerprints(units: Sequence[CodeUnit]) -> dict[str, tuple[str, list[CodeUnit]]]:
    """path -> (fingerprint of that file's units, the units), in corpus order. Two files
    with the same units in the same order and the same code have the same fingerprint."""
    by_path: dict[str, list[CodeUnit]] = {}
    for u in units:
        by_path.setdefault(u.path, []).append(u)
    out = {}
    for path, us in by_path.items():
        h = hashlib.sha1()
        for u in us:
            h.update(u.doc_id.encode("utf-8", "surrogatepass"))
            h.update(b"\x1e")
            h.update((u.qualname or "").encode("utf-8", "surrogatepass"))
            h.update(b"\x1e")
            h.update((u.code or "").encode("utf-8", "surrogatepass"))
            h.update(b"\x1f")
        out[path] = (h.hexdigest(), us)
    return out


class GrepBaseline(Retriever):
    name = "grep"

    def __init__(self, max_patterns: int = 10):
        self.max_patterns = max_patterns          # GrepRAG uses m=10 patterns
        self._units: list[CodeUnit] = []
        self._haystacks: list[str] = []           # per-unit "qualname code", lowercased once
        self._bm = BM25()                         # corpus-wide rerank scorer, kept up to date
        self._files: dict[str, tuple[str, list[CodeUnit]]] = {}   # path -> (fingerprint, units)

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "GrepBaseline":
        """Build the index, or bring it up to date: only files whose units changed are
        re-indexed, files that disappeared are dropped."""
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the grep ranker", "a Lucene BM25 index (bm25_pyserini)")
        fresh = file_fingerprints(units)
        changed = [p for p, (fp, _) in fresh.items() if self._files.get(p, ("", None))[0] != fp]
        gone = [p for p in self._files if p not in fresh]
        for path in changed + gone:
            _, old = self._files.get(path, ("", []))
            self._bm.remove(u.doc_id for u in old)
        for path in changed:
            self._bm.add({u.doc_id: f"{u.qualname} {u.code}" for u in fresh[path][1]})
        self._files = fresh
        self._units = list(units)
        self._haystacks = [f"{u.qualname} {u.code}".lower() for u in self._units]
        return self

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
        # corpus-wide scorer, reading precomputed per-doc statistics and corpus idf.
        ranked = self._bm.score_subset(code_tokenize(query), candidates)
        return ranked[:k], len(candidates)


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("grep")
def _build_grep(cfg: RetrieverConfig, name: str):
    return lambda: GrepBaseline()
