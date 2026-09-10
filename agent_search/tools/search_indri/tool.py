"""`isearch`: Indri graded query-language search over a structure table.

Ported from `agent_search.agent.tools.doc_indri.IndriFetchWorkspace`/`IndriVisitWorkspace`,
logic unchanged. isearch(query, k) scores documents by Dirichlet-smoothed belief (the manual,
`indri_doc.md`, has the combination math) -- a query never hard-zeros -- and renders the same
structure table `search_bql` renders (section names, infobox keys, no bodies), plus one line
naming the weakest constraint for the top hit. An operator nudge (a hint toward
`#combine`/`.field`/`#date:between` syntax) is appended, capped, whenever the raw query used
no operator at all; this arm has no earlier baseline whose behavior must stay put, so the
nudge is on by default. Pairs with `fetch` (agent_search.tools.fetch) or `visit`
(agent_search.tools.visit, alias `visit_v`) depending on the toolset a strategy assembles.

Options:
  snippets -- append a one-line best-matching excerpt per hit (`isearch_s`, `isearch_v`).
"""
from __future__ import annotations

import re

from agent_search.tools.base import Tool
from agent_search.tools.common import _INTRO, _infobox, best_line, sections_from_body

# a mechanical, CORPUS-FREE mid-episode nudge -- derived only from the agent's own raw query
# text -- toward the `#combine`/`.field`/`#date:between` operator surface the manual teaches,
# for a query that used none of it.
_HASH_OP_RE = re.compile(r"#")
_FIELD_SUFFIX_RE = re.compile(r"\w\.[A-Za-z]+")

_OP_NUDGE_HINT = (
    "hint: bare keywords work, but structure is sharper — e.g. #combine( #1(exact phrase) "
    "name.title #date:between(2002-01-01 2002-12-31) ); #filreq(term ...) to require a term.")
_OP_NUDGE_CAP = 3


def _is_bare_keyword_query(query: str) -> bool:
    """True if `query` (the raw surface text) uses no `#` operator and no `.field` suffix --
    a plain bag-of-keywords isearch call. Operates only on the agent's own query string, so
    the nudge can never leak corpus information."""
    if not query:
        return False
    return not _HASH_OP_RE.search(query) and not _FIELD_SUFFIX_RE.search(query)


# plain word terms from a raw Indri query (`snippets=True`): the query surface text stripped
# of `#operator` names and `.field` suffixes, leaving the plain content words `best_line`
# scores a doc's best-matching window against -- e.g. '#combine( #1(bank management)
# treaty.title )' -> ['bank', 'management', 'treaty'].
_INDRI_OP_TOKEN_RE = re.compile(r"#[\w:]+")
_INDRI_FIELD_SUFFIX_RE = re.compile(r"\.[A-Za-z]+\b")
_INDRI_PUNCT_RE = re.compile(r'[#().{}<>"]')


def _indri_query_terms(query: str) -> list:
    """Plain word tokens of an Indri query string, operator names and field suffixes stripped
    (see the module comment above). Corpus-free -- operates only on the agent's own query
    text."""
    q = _INDRI_OP_TOKEN_RE.sub(" ", query or "")
    q = _INDRI_FIELD_SUFFIX_RE.sub("", q)
    q = _INDRI_PUNCT_RE.sub(" ", q)
    return [t for t in q.split() if t]


class SearchIndri(Tool):
    name = "isearch"
    DESCRIPTION = ("Indri structured query over the corpus (#combine/#weight/#band, "
                  "#odN/#uwN windows, #syn, term.field, #filreq/#filrej, #date:between) — "
                  "returns ranked docs with per-constraint belief diagnostics; graded "
                  "matching: a query NEVER returns zero docs.")
    SNIPPET_SENTENCE = " Each hit includes a one-line best-matching excerpt."
    description = DESCRIPTION
    manual = "indri_doc.md"
    parameters = {"type": "object",
                  "properties": {
                      "query": {"type": "string",
                                "description": "An Indri query, e.g. #combine(dog train), "
                                               "#od1(white house), term.title, "
                                               "#date:between(2002-01-01 2002-12-31)."},
                      "k": {"type": "integer",
                           "description": "Max ranked candidates to return (default 5)."}},
                  "required": ["query"]}
    engines = ("indri",)

    aliases = ("isearch", "isearch_s", "isearch_v", "search")

    # options a strategy sets
    snippets: bool = False         # a one-line best-matching excerpt per hit
    op_nudge: bool = True          # the operator-syntax hint; no baseline needs it off

    def __init__(self, name=None, **options):
        super().__init__(name, **options)
        self.description = self.DESCRIPTION + (self.SNIPPET_SENTENCE if self.snippets else "")
        self._op_nudge_emitted = 0

    def on_bind(self) -> None:
        self._op_nudge_emitted = 0

    @property
    def iex(self):
        return self.engine["indri"]

    def _best_line(self, u, terms: list) -> str:
        return best_line(u, terms)

    # -- section/infobox cache, shared with the paired `fetch` tool via state.listing -----

    def _secs(self, doc_id: str) -> dict:
        entry = self.state.listing.get(doc_id)
        if entry is not None and "sections" in entry:
            return entry["sections"]
        u = self.ubyid.get(doc_id)
        explicit = getattr(u, "sections", None) if u is not None else None
        if explicit:
            out: dict = {}
            for h, t in explicit:
                name = h or _INTRO
                key, n = name, 2
                while key in out:
                    key, n = f"{name} ({n})", n + 1
                out[key] = t
            secs = out or {_INTRO: ""}
        else:
            secs = sections_from_body(u.body if (u and u.body is not None)
                                      else (u.code if u else ""))
        entry = self.state.listing.setdefault(doc_id, {})
        entry["sections"] = secs
        entry["infobox"] = _infobox(u) if u is not None else {}
        return secs

    # -- isearch: Indri belief ranking -> STRUCTURE table (no bodies) --------------------

    def _search_impl(self, query: str, k: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        res = self.iex.search(query, k=k)
        # an unrecognized `.field` name is a VALID Indri QL query that just restricts to a
        # field with zero postings -- quietly returning fewer/zero hits, indistinguishable
        # from a genuinely zero-hit query on a real field. `.warning` (populated identically
        # by every backend via `unknown_query_fields`) makes that visible here, on every
        # return path (error / 0-hit / hit-bearing) -- `getattr` guards a test double that
        # predates this field.
        warning = getattr(res, "warning", None)
        if res.error:
            # the engine's structured parse/execution error IS the syntax feedback (names the
            # unsupported/malformed op) -- surfaced verbatim, no re-wording.
            text = f"ERROR: {res.error}"
            return f"{text}\n{warning}" if warning else text
        state = self.state
        if not res.hits:
            prior = "  (previous results still available to fetch)" if state.last_hits else ""
            text = f"isearch: {query}   (0 hits){prior}"
            return f"{text}\n{warning}" if warning else text
        state.last_hits = [doc_id for doc_id, _ in res.hits]
        state.seen.update(state.last_hits)
        terms = _indri_query_terms(query) if self.snippets else []
        lines = [f"isearch: {query}   ({len(res.hits)} hits):"]
        for rank, (doc_id, _score) in enumerate(res.hits, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            if self.snippets:
                snip = self._best_line(u, terms)
                if snip:
                    line += f"  » {snip}"
            lines.append(line)
        if res.diagnostics:
            # the child contributing the LOWEST log-belief to the TOP hit -- the constraint
            # the top-ranked doc satisfies least well (the manual coaches reading this).
            weakest_repr, weakest_logb = min(res.diagnostics, key=lambda d: d[1])
            lines.append(f"weakest constraint for top hit: {weakest_repr} (logb={weakest_logb:.2f})")
        if warning:
            lines.append(warning)
        return "\n".join(lines)

    def run(self, args: dict) -> str:
        query = args.get("query") or args.get("q") or ""
        k = int(args.get("k", 5) or 5)
        result = self._search_impl(query, k)
        if (self.op_nudge and self._op_nudge_emitted < _OP_NUDGE_CAP
                and _is_bare_keyword_query(query)):
            self._op_nudge_emitted += 1
            result = f"{result}\n{_OP_NUDGE_HINT}"
        return result


__all__ = ["SearchIndri"]
