"""Lucene-backed structured retrieval -- the Indri-QL/BQL query languages compiled
to real Lucene `Query` objects and run over a persisted, fielded Lucene index, via
raw pyjnius JNI (NOT pyserini's high-level `LuceneSearcher`, which only supports
plain-text `{id, contents}` indexing and cannot express fielded documents, span
queries, or `LMDirichletSimilarity`).

This package is strictly ADDITIVE: a SIBLING of `structural/indri/` and
`structural/bql/` (the pure-Python reference engines, which remain the semantics
source of truth -- see `agent_search/prompts/skills/indri_doc.md` and `docs/bql_spec.md`), not a
replacement. It reuses both engines' existing PARSERS (`indri.parser.parse`,
`bql.parser.parse`) rather than re-parsing the query languages, and it never
imports from or modifies `agent_search/retrievers/lexical/pyserini.py`.

Modules:
    schema.py          - field-schema decision (one Lucene Document per CodeUnit;
                          stemmed/exact field pairs) -- see its module docstring.
    jni_utils.py        - shared pyjnius class handles + analyzer helpers.
    index_builder.py    - offline index build (`indexes/lucene_structured/<dataset>/`).
    indri_compiler.py   - Indri QL AST -> Lucene Query (LMDirichletSimilarity).
    bql_compiler.py     - BQL AST -> Lucene Query (BM25Similarity).
    engine.py           - `LuceneStructuredEngine`: the searcher class the tool
                          layer/tests talk to (`search_indri`, `search_bql`).

Public API:
    LuceneStructuredEngine(index_root=..., dataset=...) -- see engine.py
    get_engine(index_root, dataset) -> cached LuceneStructuredEngine
"""
from agent_search.retrievers.structural.lucene.engine import (
    LuceneHit,
    LuceneResult,
    LuceneStructuredEngine,
    get_engine,
)

__all__ = ["LuceneStructuredEngine", "LuceneHit", "LuceneResult", "get_engine"]
