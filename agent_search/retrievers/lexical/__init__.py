"""Lexical retrievers.

`pyserini.BM25Pyserini` is the BM25 engine: Lucene, persisted under
`<index_root>/bm25_pyserini/<key>/lucene/` and built once per corpus. Every BM25 tool and
the `bm25` floor use it, for documents and for code repositories alike. `grep.GrepBaseline`
is the index-free grep ranker for code repositories; it reranks with `scorer.BM25`, the one
in-memory scorer in the library, kept for code because a repository is small and can change
while the agent works on it. There is no in-memory BM25 for document corpora.

`build_bm25_engine` is what `agent_search.retrievers.engines.Engines` calls for the `bm25`
engine kind; tools never construct a retriever themselves.
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit


def build_bm25_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                      rebuild: bool = False, key: Optional[str] = None):
    """The BM25 engine for `units`: `BM25Pyserini`, built into or loaded from
    `index_root/bm25_pyserini/<key>/lucene/`. Set `BM25_INDEX_PATH` (`retrieval.bm25_index`)
    to open an existing Lucene index instead, which is also how an on-disk document store
    gets its BM25."""
    from agent_search.retrievers.lexical.pyserini import BM25Pyserini
    return BM25Pyserini(index_root=index_root, rebuild=rebuild).index(units, key=key)
