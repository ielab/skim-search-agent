"""The retrievable unit: the atom a corpus is made of and a retriever ranks.

Canonically defined in ``agent_search.corpus.units`` (with the builders that chunk
code/documents into units); re-exported here so the core contracts live in one
place. ``Unit`` is the framework-neutral name; ``CodeUnit`` is the concrete type
(it carries optional document fields: title, section, body, so the same unit serves
both the code and the general-document corpus regimes).
"""
from __future__ import annotations

from ..corpus.units import CodeUnit, code_tokenize, units_from_documents, units_from_python_source

Unit = CodeUnit

__all__ = ["Unit", "CodeUnit", "code_tokenize",
           "units_from_documents", "units_from_python_source"]
