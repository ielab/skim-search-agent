"""No excerpt: the listing shows structure only (a title, section names, infobox keys). The
plain Sieve arm (`search_bqldf`), the plain Indri arm and the plain search-fetch arms
(`bm25_search`, `dense_search_fp`) use it."""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.snippets.base import Snippet, register_snippet
from agent_search.tools.budgets import SNIPPET_TOKENS


@register_snippet
class NoSnippet(Snippet):
    name = "none"
    shows_excerpt = False

    def render(self, u: CodeUnit, terms: Optional[Sequence[str]] = None,
               width: int = SNIPPET_TOKENS) -> str:
        return ""


__all__ = ["NoSnippet"]
