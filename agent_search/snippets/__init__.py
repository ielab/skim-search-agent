"""Snippet methods, one file each: `opening` (the first tokens), `term_window` (the best
window for the query terms), `none` (structure only). A search tool takes one as `snippet=`."""
from agent_search.snippets.base import SNIPPETS, Snippet, build_snippet, register_snippet, unit_tokens
from agent_search.snippets.none import NoSnippet
from agent_search.snippets.opening import OpeningLine
from agent_search.snippets.term_window import TermWindow, best_window

__all__ = ["Snippet", "SNIPPETS", "build_snippet", "register_snippet", "unit_tokens",
           "NoSnippet", "OpeningLine", "TermWindow", "best_window"]
