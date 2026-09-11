"""The opening line: the document's first `width` tokens, whatever the query. The excerpt of
the search-visit listings (`bm25_search`, `dense_search`), of the DCI staging listing and of
ITER's `search`."""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.snippets.base import Snippet, register_snippet, unit_tokens
from agent_search.tools.budgets import SNIPPET_TOKENS


@register_snippet
class OpeningLine(Snippet):
    name = "opening"

    def render(self, u: CodeUnit, terms: Optional[Sequence[str]] = None,
               width: int = SNIPPET_TOKENS) -> str:
        return " ".join(unit_tokens(u)[:width])


__all__ = ["OpeningLine"]
