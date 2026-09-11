"""Shared helpers used by more than one document tool.

`sections_from_body`/`_infobox` parse a document's `##` sections and infobox facts.
`best_line`/`opening_line` compute the listing snippet (query-biased and opening-window
respectively) shared by `search_bm25`, `search_dense`, `search_hybrid`, `search_bql`, and
`search_indri`. `rrf_fuse` is the Reciprocal Rank Fusion used by `search_hybrid`.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize

from .budgets import RRF_K, SNIPPET_TOKENS

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


def best_line(u: CodeUnit, terms: "list[str]", width: int = SNIPPET_TOKENS) -> str:
    """The document's best-matching ~`width`-token window for `terms` (`width` defaults to
    `SNIPPET_TOKENS`, env-settable). A single pass over the whitespace-tokenized body,
    incrementally tracking how many distinct `terms` (`code_tokenize`d, case-insensitive) the
    current window contains as it slides one token at a time (add the entering token, drop the
    leaving one). The highest-scoring window wins; ties go to the earliest window (a strict `>`
    keeps the first max). Empty `terms` (an unparseable query) falls back to the document's
    opening `width` tokens.

    The window is measured in tokens only, with no character cap alongside it, so `width` is
    the whole story: a `SNIPPET_TOKENS` change always changes the snippet actually shown.

    A module-level function, not a method, so it is shared by every tool that needs a
    query-biased excerpt: `search_bm25`, `search_dense`, `search_hybrid`, `search_bql`, and
    `search_indri` each call it (directly, or through their own `_best_line` wrapper) with
    their own term source."""
    toks = ((u.body if u.body is not None else u.code) or "").split()
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


def opening_line(u: CodeUnit, width: int = SNIPPET_TOKENS) -> str:
    """The document's opening `width`-token window, the non-query-biased listing snippet used
    by `search_bm25`, `search_dense`, `search_hybrid`, `search_bm25_dci`, and `search_dedup`.

    Delegates to `best_line` with no terms, using its opening-window fallback, so the two
    snippet kinds share one window implementation and one knob (`SNIPPET_TOKENS`).
    """
    return best_line(u, [], width=width)


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


class _SeenMixin:
    """Adds a `surfaced` property, the first-seen-order list from an `OrderedSeen` held in
    `self.seen` (`agent_search.core.seen`). Current tools get `surfaced` from `EpisodeState`
    in `agent_search/tools/base.py` instead; this mixin is not used by any built-in tool."""

    @property
    def surfaced(self) -> list:
        return list(self.seen)


def rrf_fuse(bm25_ids: "Sequence[str]", dense_ids: "Sequence[str]", k: int = RRF_K,
            topk: Optional[int] = None) -> list:
    """Reciprocal Rank Fusion over two ranked doc_id lists (a BM25 pool, a dense pool).

        score(d) = sum_i  1 / (k + rank_i(d))     for each list i in which d appears
                                                    (rank_i is 1-based; a doc absent from a
                                                    list contributes nothing for it, never
                                                    an infinite or penalized rank)

    Docs are ranked by descending score; ties are broken by doc_id ascending. This matches
    `FlatIndex.search`'s own tie-break convention, so results stay reproducible even when two
    docs land on an identical fused score, for example both absent from one list and tied in
    the other. `topk` truncates the returned list; `None` returns every doc_id appearing in
    either list (the full union), still score-sorted.

    A pure function of the two id lists, with no corpus or engine access, so it is
    unit-testable against a hand-computed fixture with no retrieval stack at all (see
    tests/test_hybrid.py)."""
    scores: dict = {}
    for ids in (bm25_ids, dense_ids):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return ranked[:topk] if topk is not None else ranked


def _cap_tokens(text: str, max_tokens: int, marker: str = "") -> str:
    """Truncate `text` to `max_tokens` on the library's ruler, appending `marker` when cut."""
    from agent_search.core.tokens import cap_tokens
    return cap_tokens(text, max_tokens, marker)
