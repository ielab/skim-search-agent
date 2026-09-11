"""The Indri query language: `parser.py` (the AST and parser), `fields.py` (the fields a
query can restrict to), `result.py` (what a search returns). Scoring runs on the Lucene
structured index: `agent_search/retrievers/lucene/indri_compiler.py` compiles the AST to
Lucene queries scored with LMDirichlet, and `lucene/adapters.py`'s `LuceneIndriAdapter` is
the engine the `search_indri` tool calls.

See `agent_search/tools/search_indri/indri_doc.md` for the operator syntax.
"""
from agent_search.retrievers.indri.fields import FIELDS, field_text
from agent_search.retrievers.indri.parser import IndriParseError, ParseResult, parse
from agent_search.retrievers.indri.result import IndriResult, field_warning, unknown_query_fields

__all__ = [
    "FIELDS",
    "field_text",
    "IndriResult",
    "field_warning",
    "unknown_query_fields",
    "IndriParseError",
    "ParseResult",
    "parse",
]
