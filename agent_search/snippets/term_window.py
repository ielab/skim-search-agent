"""The query-term window: the document's best-matching window for the query, at most `width`
model tokens.

One pass over the document's words, sliding a window of `width` words one word at a time and
counting how many distinct query terms (`code_tokenize`d, case-insensitive) it holds. The
highest count wins; ties go to the earliest window. With no terms (an unparseable query) the
window is the opening one. The winning window is then cut to `width` model tokens on the
library's ruler (a word is never shorter than one token, so the window is wide enough, and a
cut keeps its start, where the matching terms sit). This is the excerpt Sieve's result cards
show (`search_s`), and the fairness variant of the BM25 listing (`bm25q_search`).
"""
from __future__ import annotations

from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.snippets.base import Snippet, register_snippet, unit_tokens
from agent_search.tokens import truncate_tokens
from agent_search.tools.budgets import SNIPPET_TOKENS


def best_window(toks: Sequence[str], terms: Sequence[str], width: int) -> str:
    """The `width`-token window of `toks` holding the most distinct `terms`; the opening
    window when `terms` is empty."""
    if not toks:
        return ""
    term_set = {t.lower() for t in (terms or [])}
    if not term_set:
        return " ".join(toks[:width])
    tok_terms = [set(code_tokenize(t)) & term_set for t in toks]
    counts: dict = {}
    score = 0

    def _add(i: int) -> None:
        nonlocal score
        for t in tok_terms[i]:
            c = counts.get(t, 0)
            if c == 0:
                score += 1
            counts[t] = c + 1

    def _drop(i: int) -> None:
        nonlocal score
        for t in tok_terms[i]:
            c = counts[t] - 1
            counts[t] = c
            if c == 0:
                score -= 1

    n = len(toks)
    w = min(width, n)
    for i in range(w):
        _add(i)
    best_start, best_score = 0, score
    for start in range(1, n - w + 1):
        _drop(start - 1)
        _add(start + w - 1)
        if score > best_score:
            best_score, best_start = score, start
    return " ".join(toks[best_start:best_start + w])


@register_snippet
class TermWindow(Snippet):
    name = "term_window"

    def render(self, u: CodeUnit, terms: Optional[Sequence[str]] = None,
               width: int = SNIPPET_TOKENS) -> str:
        return truncate_tokens(best_window(unit_tokens(u), terms or [], width), width, "")


__all__ = ["TermWindow", "best_window"]
