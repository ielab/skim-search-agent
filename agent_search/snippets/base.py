"""The snippet contract: the excerpt a search listing shows under one hit.

How a hit is presented is part of a strategy: the opening line, a window around the query
terms, or nothing at all changes what the model sees and what it does next. Snippet methods
live one file each under this package; a tool takes one as its `snippet=` option, so a
strategy swaps presentation without touching the tool. Widths are token counts
(`agent_search/tools/budgets.py`, `SNIPPET_TOKENS`), never characters.
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.tools.budgets import SNIPPET_TOKENS

SNIPPETS: dict[str, type] = {}


def unit_tokens(u: CodeUnit) -> list[str]:
    """The whitespace tokens of a unit's text (its body, or its code)."""
    return ((u.body if u.body is not None else u.code) or "").split()


class Snippet:
    name: str = ""
    shows_excerpt: bool = True      # False for the method that shows nothing

    def render(self, u: CodeUnit, terms: Optional[Sequence[str]] = None,
               width: int = SNIPPET_TOKENS) -> str:
        """The excerpt for `u`, at most `width` tokens. `terms` are the query's tokens; a
        method that ignores them still accepts them."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"snippet": self.name}


def register_snippet(cls: type) -> type:
    SNIPPETS[cls.name] = cls
    return cls


def build_snippet(name: str, **params) -> Snippet:
    """A snippet method by name; an unknown name is an error."""
    try:
        cls = SNIPPETS[name]
    except KeyError:
        raise ValueError(f"unknown snippet method {name!r}; choose from {sorted(SNIPPETS)}") from None
    return cls(**params)


__all__ = ["Snippet", "SNIPPETS", "register_snippet", "build_snippet", "unit_tokens"]
