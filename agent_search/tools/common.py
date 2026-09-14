"""Shared helpers used by more than one document tool.

`sections_from_body`/`_infobox` parse a document's `##` sections and infobox facts.
The listing snippet is a method from `agent_search/snippets/` (the query-term window and the opening
respectively) shared by `search_bm25`, `search_dense`, `search_hybrid`, `search_bql`, and
`search_indri`.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit


_INTRO = "(intro)"
# a markdown/wiki heading line: leading #'s (level) then the heading text. The structured
# doc corpora carry `## History` markers in the body; a flat doc has none -> one section.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def sections_from_body(body: str) -> "dict[str, str]":
    """Split a doc body into {heading: text}, split on `##` markers. Text before
    the first heading is the '(intro)'. A body with no markers is one '(intro)' section (a
    flat doc), so the same fetch contract works on flat and structured corpora alike.
    Duplicate headings are disambiguated ('History', 'History (2)')."""
    out: "dict[str, str]" = {}
    cur, buf = _INTRO, []

    def flush(name: str, lines: list) -> None:
        if not lines and name == _INTRO:
            return
        text = "\n".join(lines).strip()
        key, n = name, 2
        while key in out:                       # disambiguate a repeated heading
            key, n = f"{name} ({n})", n + 1
        out[key] = text

    for line in (body or "").splitlines():
        m = _HEADING.match(line)
        if m:
            flush(cur, buf)
            cur, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    flush(cur, buf)
    if not out:                                 # empty body -> a single empty intro
        out[_INTRO] = ""
    return out


def _infobox(u: CodeUnit) -> "dict[str, str]":
    """The doc's infobox facts, if the corpus carried any (structured wiki). Stored in
    metadata['infobox'] as 'k: v; k: v' by the corpus builder, else absent."""
    meta = u.metadata or {}
    raw = str(meta.get("infobox") or "")
    facts: "dict[str, str]" = {}
    for part in raw.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            if k.strip():
                facts[k.strip()] = v.strip()
    return facts


def structure_str(named: Sequence[str], keys: Sequence[str]) -> "tuple[str, str]":
    """The `§[...]` and `ib[...]` lists of one result card: the section names and infobox
    keys, cut at `LISTING_SECTIONS` / `LISTING_INFOBOX_KEYS` (0 = all) with `,…` marking
    a cut. Every search tool's card uses this, so a knob moves every strategy alike."""
    from agent_search.tools.budgets import LISTING_INFOBOX_KEYS, LISTING_SECTIONS
    named, keys = list(named), list(keys)
    ns = len(named) if LISTING_SECTIONS <= 0 else LISTING_SECTIONS
    nk = len(keys) if LISTING_INFOBOX_KEYS <= 0 else LISTING_INFOBOX_KEYS
    sec_str = "·".join(named[:ns]) + (",…" if len(named) > ns else "")
    ib_str = "·".join(keys[:nk]) + (",…" if len(keys) > nk else "")
    return sec_str, ib_str


class _SeenMixin:
    """Adds a `surfaced` property, the first-seen-order list from an `OrderedSeen` held in
    `self.seen` (`agent_search.tools.seen`). Current tools get `surfaced` from `EpisodeState`
    in `agent_search/tools/base.py` instead; this mixin is not used by any built-in tool."""

    @property
    def surfaced(self) -> list:
        return list(self.seen)



def _cap_tokens(text: str, max_tokens: int, marker: str = "") -> str:
    """Truncate `text` to `max_tokens` on the library's ruler, appending `marker` when cut."""
    from agent_search.tokens import cap_tokens
    return cap_tokens(text, max_tokens, marker)
