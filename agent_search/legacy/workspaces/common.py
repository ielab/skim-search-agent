"""Pre-0.3 shared helpers used by more than one doc-research tool family.

Kept so the parity tests can compare against it. `agent_search/tools/common.py` replaces it.

`_SeenMixin` gives every doc workspace its `surfaced` ranking view. `sections_from_body`/
`_infobox` parse a document's `##` sections and infobox facts. `best_line`/`opening_line`
compute the listing snippet (query-biased and opening-window respectively) shared by the
visit, fetch, and sieve families. `rrf_fuse` is the Reciprocal Rank Fusion used by the
hybrid visit and fetch workspaces.
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
    """Split a doc body into {heading: text} LIVE on `##` markers. Text before
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
    """The doc's best-matching ~`width`-token window for `terms` (`width` defaults to
    `SNIPPET_TOKENS`, env-settable) — a single pass over the
    whitespace-tokenized body, incrementally tracking how many DISTINCT `terms` (`code_tokenize`
    'd, case-insensitive) the current window contains as it slides one token at a time (add the
    entering token, drop the leaving one); the highest-scoring window wins, ties -> earliest (a
    strict `>` keeps the first max). Empty `terms` (unparseable query) falls back to the doc's
    opening `width` tokens.

    The window is defined by TOKENS ONLY. An earlier version also clipped the result to 160
    characters, which meant a token-count knob did not actually control the snippet: on
    ordinary prose (~6.4 chars/token) the cap bound at around 25 tokens, so any wider setting
    was silently truncated. Removed, so `width` is the whole story.

    MODULE-LEVEL (not a method) so it's shared verbatim by `DocSearchFetch._best_line`
    (research_snip/research_indri_snip's leaf-token-driven excerpt) and `Bm25Visit`'s
    query-biased excerpt mode — one best-matching-window implementation, two
    callers with different term sources."""
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
    """The doc's OPENING `width`-token window — the non-query-biased listing snippet the
    visit-family baselines show (Bm25Visit, DenseVisit, HybridVisit, Bm25DciWorkspace).

    Delegates to `best_line` with no terms (its documented opening fallback), so the two
    snippet kinds share one window implementation and one knob. This replaces a fixed
    120-CHARACTER slice: with the method arm measured in tokens and the baseline arm in
    characters the two listings were not commensurable, and a snippet-width sweep moved only
    one side of the comparison. Both are now governed by SNIPPET_TOKENS.
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
    """Shared by every doc workspace: `self.seen` is an `OrderedSeen` (agent_search.core.seen)
    tracking every doc_id surfaced, in first-seen order. `surfaced` is the read-only view
    `agent/loop.py::run_episode` reads (`getattr(workspace, "surfaced", [])`) as the episode's
    retrieval ranking for hit@k/recall@k/nDCG — without it a doc workspace only exposed `seen`
    (unordered), so every document-research run reported those rank-based metrics as 0 even when
    the gold document was read."""

    @property
    def surfaced(self) -> list:
        return list(self.seen)


def rrf_fuse(bm25_ids: "Sequence[str]", dense_ids: "Sequence[str]", k: int = RRF_K,
            topk: Optional[int] = None) -> list:
    """Reciprocal Rank Fusion over two ranked doc_id lists (a BM25 pool, a dense pool).

        score(d) = sum_i  1 / (k + rank_i(d))     for each list i in which d appears
                                                    (rank_i is 1-based; a doc absent from a
                                                    list contributes NOTHING for it — never
                                                    an infinite/penalized rank)

    Docs are ranked by DESCENDING score; ties are broken by doc_id ASCENDING (deterministic —
    matches FlatIndex.search's own tie-break convention, so results are reproducible even when
    two docs land on an identical fused score, e.g. both absent from one list and tied in the
    other). `topk` truncates the returned list (None = every doc_id appearing in EITHER list,
    i.e. the full union, still score-sorted).

    Pure function of the two id lists — no corpus/engine access — so it is unit-testable
    against a hand-computed fixture with no retrieval stack at all (see tests/test_hybrid.py)."""
    scores: dict = {}
    for ids in (bm25_ids, dense_ids):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return ranked[:topk] if topk is not None else ranked
