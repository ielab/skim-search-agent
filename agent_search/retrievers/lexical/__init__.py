"""Lexical (BM25) retrievers: `bm25.BM25Local` (dependency-free approximation, code_tokenize
analyzer) and `pyserini.BM25Pyserini` (canonical Lucene BM25 — Porter stemming + stopwords,
k1=0.9/b=0.4, matching SWE-bench's own BM25 baseline).

`build_bm25_engine` is the ONE place that resolves env `BM25_BACKEND` (local|pyserini,
default local) to a built engine. Shared by:
  - `agent_search.agent.retriever.AgentRetriever.index()` (the real per-episode construction
    site for the bm25/bm25dci/bm25fetch/bm25q/bm25fetchsnip arms — supplies a real
    `index_root`/`rebuild`/corpus `key` so a `pyserini` backend loads/persists the SAME
    on-disk Lucene index `agent_search/evaluation/build_indexes.py --retriever bm25_pyserini` pre-builds).
  - Every doc-arm workspace's `engine=None` fallback default (agent_search.agent.tools.
    doc_research.Bm25Visit/Bm25FetchWorkspace/Bm25FetchSnipWorkspace, agent_search.agent.
    tools.doc_bm25_dci.Bm25DciWorkspace) — so a workspace built WITHOUT an explicit `engine`
    (tests, ad-hoc scripts) still respects the knob instead of silently hardcoding BM25Local.

Both engines expose the IDENTICAL `search(query, k) -> list[str]` interface
(agent_search.core.interfaces.Retriever), so swapping the backend never touches a caller's
listing/best_line rendering — only which engine answers a query.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit


def build_bm25_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                      rebuild: bool = False, key: Optional[str] = None):
    """Env `BM25_BACKEND` (default `local`, case-insensitive):

      local     (default, unset also means this) BM25Local — in-memory, built fresh from
                `units` every call; `key` is accepted but unused (matches BM25Local.index's
                own signature, kept for a uniform call shape with the pyserini branch).
      pyserini  BM25Pyserini — canonical Lucene BM25, PERSISTED under
                `index_root/bm25_pyserini/<key>/lucene/` (built once, mmap-loaded after).
                Motivated by an empirical ~0.546 top-5 Jaccard divergence between the two
                engines' rankings for the SAME query/corpus on browsecomp_plus (BM25Local's
                dependency-free tokenizer is not a faithful stand-in for canonical BM25 on
                prose corpora) — see pyserini.py's module docstring.

    Any other value raises (a typo in BM25_BACKEND must fail loud, not silently fall back)."""
    backend = (os.environ.get("BM25_BACKEND") or "local").strip().lower()
    if backend == "pyserini":
        from agent_search.retrievers.lexical.pyserini import BM25Pyserini
        return BM25Pyserini(index_root=index_root, rebuild=rebuild).index(units, key=key)
    if backend not in ("local", ""):
        raise ValueError(
            f"unknown BM25_BACKEND={backend!r} — choose 'local' (default) or 'pyserini'.")
    if getattr(units, "lazy", False):
        from agent_search.core.errors import SetupError
        raise SetupError(
            f"corpus key {key!r} is an on-disk document store; the in-memory BM25 cannot load it. "
            f"Use BM25_BACKEND=pyserini with BM25_INDEX_PATH (retrieval.bm25_backend / "
            f"retrieval.bm25_index) pointing at a prebuilt Lucene index.")
    from agent_search.retrievers.lexical.bm25 import BM25Local
    return BM25Local().index(units, key=key)
