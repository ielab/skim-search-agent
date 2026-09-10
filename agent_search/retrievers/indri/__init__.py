"""Indri retrieval-model backend — a SIBLING of the Boolean `bql/` engine.

Faithful (documented-deviations) reimplementation of Indri Query Language semantics:
parsing (`parser.py`), index building (`index.py`) and Dirichlet-smoothed scoring
(`model.py`). See `agent_search/tools/search_indri/indri_doc.md` for the source-of-truth operator
syntax and belief-combination math this package implements.

This package is strictly ADDITIVE: it imports from `agent_search.corpus.units`
(`CodeUnit`, `code_tokenize`) but does not modify anything under
`agent_search/retrievers/bql/` or any other existing module.

Public API (stable — the agent tool layer builds against this):
    IndriExecutor(units, mu=None)
    IndriExecutor.search(query, k=5) -> IndriResult(hits, error, diagnostics)
    IndriExecutor.save(path) / IndriExecutor.load(path) [staticmethod]
    IndriExecutor.attach_units(units)
    indri_index_path(index_root, key) -> str
    load_or_build(units, index_root=None, key=None) -> IndriExecutor
"""
from agent_search.retrievers.indri.model import (
    IndriExecutor,
    IndriResult,
    indri_index_path,
    load_or_build,
)
from agent_search.retrievers.indri.parser import (
    IndriParseError,
    ParseResult,
    parse,
)

__all__ = [
    "IndriExecutor",
    "IndriResult",
    "indri_index_path",
    "load_or_build",
    "IndriParseError",
    "ParseResult",
    "parse",
]
