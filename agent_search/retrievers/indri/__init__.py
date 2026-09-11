"""Indri retrieval-model backend: a sibling of the Boolean `bql/` engine.

Faithful (documented-deviations) reimplementation of Indri Query Language semantics:
parsing (`parser.py`), index building (`index.py`) and Dirichlet-smoothed scoring
(`model.py`). See `agent_search/tools/search_indri/indri_doc.md` for the source-of-truth operator
syntax and belief-combination math this package implements.

This package only imports from `agent_search.corpus.units` (`CodeUnit`, `code_tokenize`); it
does not depend on `agent_search/retrievers/bql/` or any other retriever family.

Public API the `search_indri` tool (`agent_search/tools/search_indri/tool.py`) builds against:
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
