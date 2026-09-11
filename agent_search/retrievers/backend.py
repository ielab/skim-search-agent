"""Which engine answers a structured query (BQL, Indri) for a corpus. The corpus kind decides;
there is no environment switch.

Document corpora run on Lucene (`agent_search/retrievers/lucene/`): a prebuilt fielded index
under `indexes/lucene_structured/<key>/`, BM25 scoring for BQL and LMDirichlet scoring for
Indri, with the zero-hit fallback and the coverage ranking on the same index. The index is
built once per corpus: ahead of a run with `skimsearchagent-build-indexes --retriever
search_lucene --dataset <name>` (the way to do it for a large corpus, off the clock), or on
first use for an in-memory corpus, the same way the Lucene BM25 index is. An on-disk corpus
with no index stops the run before its first episode.

A code repository (`domain="code"`) runs on the in-memory Boolean executor
(`agent_search/retrievers/bql/executor.py`): it is the only engine that evaluates the code
AST regions (`IN(def, x)`, `IN(call, x)`, `IN(sig, x)`), and a repository is small enough to
hold in memory and refresh when a file changes. Indri has no code strategy, so it is Lucene
only. Dense fusion (`BQL_DENSE`) is a document feature and is refused for a repository.

`agent_search.retrievers.engines.Engines` is the per-corpus registry every tool shares; its
`bql`, `bql_fused`, `bql_dense_only`, `bql_plain` and `indri` methods call into this module
once per corpus, under a lock, and cache the result. Every engine exposes the surface its
tool calls (`run_with_count`, `soft_topk`, `coverage_topk`; `search`), so a tool never
knows which engine answered.
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.errors import SetupError


def is_code_corpus(domain: Optional[str]) -> bool:
    return (domain or "").strip().lower() == "code"


def _lucene_engine(units: Sequence[CodeUnit], index_root: str, key: Optional[str]):
    """Open the Lucene structured index for this corpus, building it first when it does not
    exist and the corpus is in memory. A corpus with no key gets one from its fingerprint.
    An on-disk corpus (a document store) must have been indexed ahead of the run
    (`skimsearchagent-build-indexes --retriever search_lucene`): building would load it whole."""
    from agent_search.retrievers.lucene import index_builder
    from agent_search.retrievers.lucene.engine import get_engine
    if not key:
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the Lucene structured index build", "a corpus key and a prebuilt index")
        from agent_search.corpus.fingerprint import corpus_fingerprint
        key = "adhoc_" + corpus_fingerprint(units)[:16]
    if not index_builder.is_built(index_root, key):
        if getattr(units, "lazy", False):
            raise SetupError(
                f"no Lucene structured index for corpus key {key!r} under {index_root!r}; build it "
                f"first with: skimsearchagent-build-indexes --dataset {key} --retriever search_lucene")
        index_builder.build(list(units), index_root, key, progress=False)
    eng = get_engine(index_root=index_root, dataset=key)
    # Open and validate the index now, not on the first search, so a broken index stops the
    # run before it starts instead of turning every search into an error observation.
    eng._ensure_open()
    return eng


def build_bql_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                     key: Optional[str] = None, dense=None, domain: str = "general"):
    """The BQL executor behind the `search_bql` and `fetch_code` tools and the `bql` floor.
    Documents: `LuceneBqlAdapter` over the prebuilt Lucene index. Code: the in-memory
    `StructuralExecutor` over the repository's units. `dense` (a `DenseBelief`, documents
    only) turns on the RRF fusion of BM25 and dense rank (`bql/dense_fuse.py`)."""
    if is_code_corpus(domain):
        if dense is not None:
            raise SetupError("dense-fused BQL ranking is a document feature; a code repository "
                             "ranks with the in-memory BM25 scorer only")
        from agent_search.retrievers.bql.executor import StructuralExecutor
        return StructuralExecutor(units)
    from agent_search.retrievers.lucene.adapters import LuceneBqlAdapter
    return LuceneBqlAdapter(_lucene_engine(units, index_root, key), dense=dense)


def build_bql_engine_dense_only(units: Sequence[CodeUnit], index_root: str = "indexes",
                                key: Optional[str] = None, dense=None, domain: str = "general"):
    """The dense-only sibling of `build_bql_engine` (strategies `sieve_dense` and
    `sieve_visit_dense`): the same Lucene filter, candidates ordered by dense rank alone."""
    if is_code_corpus(domain):
        raise SetupError("dense-only BQL ranking is a document feature; a code repository "
                         "ranks with the in-memory BM25 scorer only")
    from agent_search.retrievers.lucene.adapters import LuceneBqlDonlyAdapter
    return LuceneBqlDonlyAdapter(_lucene_engine(units, index_root, key), dense=dense)


def build_indri_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                       key: Optional[str] = None, dense=None, domain: str = "general"):
    """The Indri executor behind the `search_indri` tool: `LuceneIndriAdapter` over the
    prebuilt Lucene index. `dense` (a `DenseBelief`, `INDRI_DENSE=1`) blends the dense
    similarity into the belief (see `lucene/adapters.py`)."""
    if is_code_corpus(domain):
        raise SetupError("Indri retrieval runs on the Lucene structured index and has no code "
                         "repository engine")
    from agent_search.retrievers.lucene.adapters import LuceneIndriAdapter
    return LuceneIndriAdapter(_lucene_engine(units, index_root, key), dense=dense)


__all__ = ["build_bql_engine", "build_bql_engine_dense_only", "build_indri_engine", "is_code_corpus"]
