"""Compatibility shim for the doc-research tool split.

The workspace classes that used to live in this one file now live in, per family:

- `budgets.py` — the env-tunable size knobs, read once at import time.
- `common.py` — helpers shared by more than one family (section parsing, snippets, RRF).
- `search_visit.py` — the retrieve-then-visit and retrieve-and-read baselines (bm25, dense,
  hybrid).
- `search_fetch.py` — the retrieve-then-fetch baselines (bm25, dense, hybrid) paired with
  named-section fetch.
- `sieve.py` — the BQL field-tagged Boolean/date search (the paper's method) and its
  dense-only-ranked siblings.

Everything the old module exposed is re-exported here, so
`from agent_search.legacy.workspaces.doc_research import X` keeps working unchanged.
"""
from __future__ import annotations

import importlib
import os
import re
from typing import Optional, Sequence

from agent_search.core.seen import OrderedSeen
from agent_search.core.tokens import cap_tokens as _cap_tokens
from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.bql.ast import And, In, Not, Or, Phrase, Prefix, Term
from agent_search.retrievers.bql.executor import (
    StructuralExecutor, execute_bql, _rank_leaves)
from agent_search.retrievers.bql.parser import parse as bql_parse
from agent_search.retrievers.bql.surface import to_bql

# `budgets`/`common` cache their env-derived constants and defaults at import time; reloading
# this shim (tests/test_snippet_listing.py's test_env_override_is_picked_up_on_import) must
# re-read a changed env var, so reload them in dependency order first — a plain
# `from .budgets import X` would otherwise just re-fetch the stale cached module.
from . import budgets as _budgets
importlib.reload(_budgets)
from . import common as _common
importlib.reload(_common)

from .budgets import (
    AUTOREAD_TOPK, BM25_FETCH_TOPK, BM25_VISIT_TOPK, DENSE_FETCH_TOPK, DENSE_VISIT_TOPK,
    HYBRID_FETCH_TOPK, HYBRID_POOL, HYBRID_VISIT_TOPK, MAX_SECTION_TOKENS, MAX_VISIT_TOKENS,
    RRF_K, SNIPPET_TOKENS)
from .common import (
    _INTRO, _HEADING, _SeenMixin, _infobox, best_line, opening_line, rrf_fuse,
    sections_from_body)
from .sieve import (
    _DATE_NUDGE_CAP, _DATE_NUDGE_HINT, _DATE_RANGE_PREFIX, _DATE_SCOPE_RE, _DECADE_RE,
    _MONTH_YEAR_RE, _YEAR_RE, _child_repr, _has_bare_temporal_clue, BqlDonlyVisitWorkspace,
    BqlVisitWorkspace, DocSearchFetch, DocSearchFetchDonlySnip)
from .search_visit import (
    Bm25AutoRead, Bm25Visit, DenseAutoRead, DenseVisit, HybridAutoRead, HybridVisit)
from .search_fetch import (
    Bm25FetchSnipWorkspace, Bm25FetchWorkspace, DenseFetchPlainWorkspace, DenseFetchWorkspace,
    HybridFetchSnipWorkspace)

__all__ = [
    "AUTOREAD_TOPK", "And", "BM25_FETCH_TOPK", "BM25_VISIT_TOPK", "Bm25AutoRead",
    "Bm25FetchSnipWorkspace", "Bm25FetchWorkspace", "Bm25Visit", "BqlDonlyVisitWorkspace",
    "BqlVisitWorkspace", "CodeUnit", "DENSE_FETCH_TOPK", "DENSE_VISIT_TOPK", "DenseAutoRead",
    "DenseFetchPlainWorkspace", "DenseFetchWorkspace", "DenseVisit", "DocSearchFetch",
    "DocSearchFetchDonlySnip", "HYBRID_FETCH_TOPK", "HYBRID_POOL", "HYBRID_VISIT_TOPK",
    "HybridAutoRead", "HybridFetchSnipWorkspace", "HybridVisit", "In", "MAX_SECTION_TOKENS",
    "MAX_VISIT_TOKENS", "Not", "Optional", "Or", "OrderedSeen", "Phrase", "Prefix", "RRF_K",
    "SNIPPET_TOKENS", "Sequence", "StructuralExecutor", "Term", "_DATE_NUDGE_CAP",
    "_DATE_NUDGE_HINT", "_DATE_RANGE_PREFIX", "_DATE_SCOPE_RE", "_DECADE_RE", "_HEADING",
    "_INTRO", "_MONTH_YEAR_RE", "_SeenMixin", "_YEAR_RE", "_cap_tokens", "_child_repr",
    "_has_bare_temporal_clue", "_infobox", "_rank_leaves", "best_line", "bql_parse",
    "code_tokenize", "execute_bql", "opening_line", "os", "re", "rrf_fuse",
    "sections_from_body", "to_bql",
]
