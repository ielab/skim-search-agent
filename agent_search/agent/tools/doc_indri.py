"""IndriFetchWorkspace: `isearch` (the Indri graded query-language backend) + `fetch` (the
SAME structured section-fetch DocSearchFetch/Bm25FetchWorkspace use).

The `indri` toolset (tools.yaml) drives this workspace. Like `Bm25FetchWorkspace` reuses
`DocSearchFetch`'s section cache + `fetch` verbatim (see doc_research.py's module docstring),
`IndriFetchWorkspace` subclasses `DocSearchFetch` and reuses everything EXCEPT retrieval:
`isearch` queries an `IndriExecutor` (Dirichlet-smoothed belief scoring — a query NEVER
hard-zeros; see skills/indri_doc.md) instead of the BQL executor. `self.ex` (the BQL executor
DocSearchFetch's inherited methods never touch — fetch only needs `self.ubyid`/`self._secs`)
is left None, matching Bm25FetchWorkspace's own pattern.

Two related workspaces cross graded Indri search with the OTHER arms' READ strategies
(disentangling "search interface" from "read granularity" so the two axes can vary
independently):

  IndriVisitWorkspace (`indri_visit` toolset) — isearch_v/visit_v: the SAME graded search as
    `isearch` (ranking, weakest-constraint diagnostic, op_nudge), rendered WITH a per-hit
    content snippet (`snippets=True` — fairness parity with the bm25 baseline's opening-snippet
    listing), + a whole-doc VISIT read (mirroring Bm25Visit.visit exactly). A strict single-axis
    swap vs research_bm25: same read, same content-bearing listing, only the search ENGINE
    differs (graded Indri vs BM25 keyword).

  IndriFetchWorkspace(snippets=True) (`research_indri_snip` x `indri_snip` toolset — the one of
    these three registered in conditions.yaml) — isearch_s/fetch: the SAME graded search +
    snippet listing, but a structured SECTION fetch read (the SAME `fetch` the plain `indri`
    arm uses) instead of a whole-doc visit.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from agent_search.agent.tools.doc_research import (
    DocSearchFetch, MAX_VISIT_TOKENS, _cap_tokens, _INTRO, _infobox)
from agent_search.core.seen import OrderedSeen
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.indri.model import IndriExecutor

# --- indri operator-nudge (on by default for this workspace, which has no prior baseline to
# preserve): a mechanical, CORPUS-FREE mid-episode nudge —
# derived only from the agent's own raw query text — toward the `#combine`/`.field`/
# `#date:between` operator surface the skill teaches, for a query that used none of it.
_HASH_OP_RE = re.compile(r"#")
_FIELD_SUFFIX_RE = re.compile(r"\w\.[A-Za-z]+")

_OP_NUDGE_HINT = (
    "hint: bare keywords work, but structure is sharper — e.g. #combine( #1(exact phrase) "
    "name.title #date:between(2002-01-01 2002-12-31) ); #filreq(term ...) to require a term.")
_OP_NUDGE_CAP = 3


def _is_bare_keyword_query(query: str) -> bool:
    """True if `query` (the raw surface text) uses NO `#` operator and NO `.field` suffix —
    i.e. it's a plain bag-of-keywords isearch call. Operates purely on the agent's own query
    string — no corpus lookups — so the nudge can never leak corpus information."""
    if not query:
        return False
    return not _HASH_OP_RE.search(query) and not _FIELD_SUFFIX_RE.search(query)


# --- IndriVisitWorkspace / research_indri_snip: plain word terms from a raw Indri query ----
# (`snippets=True`): the query surface text stripped of `#operator` names and `.field` suffixes,
# leaving the plain content words `DocSearchFetch._best_line` scores a doc's best-matching
# window against — e.g. '#combine( #1(bank management) treaty.title )' -> ['bank', 'management',
# 'treaty'] (no '#combine'/'#1' operator-name artifacts, no 'title' field-suffix artifact).
_INDRI_OP_TOKEN_RE = re.compile(r"#[\w:]+")
_INDRI_FIELD_SUFFIX_RE = re.compile(r"\.[A-Za-z]+\b")
_INDRI_PUNCT_RE = re.compile(r'[#().{}<>"]')


def _indri_query_terms(query: str) -> list:
    """Plain word tokens of an Indri query string, operator names and field suffixes stripped
    (see module comment above). Corpus-free — operates only on the agent's own query text."""
    q = _INDRI_OP_TOKEN_RE.sub(" ", query or "")
    q = _INDRI_FIELD_SUFFIX_RE.sub("", q)
    q = _INDRI_PUNCT_RE.sub(" ", q)
    return [t for t in q.split() if t]


class IndriFetchWorkspace(DocSearchFetch):
    """isearch(query, k) -> Indri graded-QL ranking (rendered as the SAME structure table
    DocSearchFetch.search renders, no bodies) + a belief-diagnostics line for the top hit.
    fetch(specs) is INHERITED from DocSearchFetch unchanged (the section cache + `_secs`/
    `_fetch_one`/`fetch` machinery needs only `self.ubyid`, not the BQL executor)."""

    tools = ("isearch", "fetch")

    def __init__(self, units: Sequence[CodeUnit],
                 executor: Optional[IndriExecutor] = None,
                 ubyid: Optional[dict] = None,
                 op_nudge: bool = True,
                 snippets: bool = False):
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if executor is not None:
            self.iex = executor
        else:
            # env STRUCTURED_BACKEND-selectable (default 'python', unchanged) — see
            # agent_search.retrievers.structural.backend.build_indri_engine. Production
            # callers (agent_search.agent.retriever) always pass `executor` explicitly (built
            # with a real index_root/key so a 'lucene' backend opens the prebuilt index); this
            # fallback is for direct/standalone construction (tests, ad-hoc scripts).
            from agent_search.retrievers.structural.backend import build_indri_engine
            self.iex = build_indri_engine(self.units)
        self.ex = None                                   # unused: retrieval is Indri, not BQL
        self._sections: dict[str, dict] = {}
        self.seen = OrderedSeen()
        self.last_hits: list[str] = []
        self.coverage = False                             # inherited attr; irrelevant here
        # DocSearchFetch's own date_nudge default (this workspace's own `search`/`_search_impl`
        # never reads it — isearch has its own op_nudge below — but the attribute must still
        # exist so an inherited method that DOES check it (`DocSearchFetch.search`) never raises
        # AttributeError if reached on this class or a subclass of it).
        self.date_nudge = False
        self._date_nudge_emitted = 0
        # op_nudge DEFAULT TRUE: DocSearchFetch's date_nudge defaults False because the
        # `research` condition's behavior depends on it staying off; IndriFetchWorkspace has no
        # such condition depending on op_nudge, so it can default on.
        self.op_nudge = op_nudge
        self._op_nudge_emitted = 0
        # `snippets` (DEFAULT OFF; `research_indri_snip` sets it True): `_search_impl` appends a
        # one-line best-matching excerpt (DocSearchFetch._best_line, reused verbatim) to every
        # isearch hit — see `_indri_query_terms` above. `IndriVisitWorkspace` forces this True
        # unconditionally (see below), since its listing is always content-bearing.
        self.snippets = snippets

    # -- isearch: Indri belief ranking -> STRUCTURE table (no bodies) --------------------

    def search(self, query: str, k: int = 5) -> str:
        """isearch(query) -> the structure TABLE (see `_search_impl`), plus ONE appended hint
        line (capped at `_OP_NUDGE_CAP` per instance, `op_nudge=True` by default) whenever the
        RAW query used no `#` operator and no `.field` suffix. Single choke point so every
        `_search_impl` return path (hits, 0-hit, error) gets the same treatment."""
        result = self._search_impl(query, k)
        if (self.op_nudge and self._op_nudge_emitted < _OP_NUDGE_CAP
                and _is_bare_keyword_query(query)):
            self._op_nudge_emitted += 1
            result = f"{result}\n{_OP_NUDGE_HINT}"
        return result

    def _search_impl(self, query: str, k: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        res = self.iex.search(query, k=k)
        # LOW finding (adversarial verification): an unrecognized `.field` name (e.g.
        # `.foo`) is a VALID Indri QL query that just restricts to a field with zero
        # postings -- both engines "search anyway" and quietly return fewer/zero hits,
        # indistinguishable from a genuinely zero-hit query on a real field. `.warning`
        # (indri.model.IndriResult / lucene.engine.LuceneResult, populated identically
        # by both backends via `unknown_query_fields`) makes that visible to the agent
        # here, on EVERY return path (error / 0-hit / hit-bearing) — `getattr` guards a
        # test double that predates this field.
        warning = getattr(res, "warning", None)
        if res.error:
            # the engine's structured parse/execution error IS the syntax feedback (names the
            # unsupported/malformed op) — surfaced verbatim, no re-wording.
            text = f"ERROR: {res.error}"
            return f"{text}\n{warning}" if warning else text
        if not res.hits:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            text = f"isearch: {query}   (0 hits){prior}"
            return f"{text}\n{warning}" if warning else text
        self.last_hits = [doc_id for doc_id, _ in res.hits]
        self.seen.update(self.last_hits)
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
            # the child contributing the LOWEST log-belief to the TOP hit — the constraint the
            # top-ranked doc satisfies least well (skills/indri_doc.md coaches reading this).
            weakest_repr, weakest_logb = min(res.diagnostics, key=lambda d: d[1])
            lines.append(f"weakest constraint for top hit: {weakest_repr} (logb={weakest_logb:.2f})")
        if warning:
            lines.append(warning)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            # "isearch_s" is research_indri_snip's tool NAME (an alias of isearch — see
            # tools.yaml's indri_snip toolset comment): a distinct tool name gets it its own
            # manual entry, but the workspace method (and snippet rendering, via self.snippets)
            # is the same.
            if name in ("isearch", "search", "isearch_s"):
                q = args.get("query") or args.get("q") or ""
                return self.search(q, int(args.get("k", 5) or 5))
            if name == "fetch":
                # DELEGATE the read verbatim to DocSearchFetch's fetch (arg-normalization +
                # section logic) — the whole point is that only retrieval differs from `research`.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class IndriVisitWorkspace(IndriFetchWorkspace):
    """isearch_v(query, k) -> the SAME graded Indri ranking `IndriFetchWorkspace.search` renders
    (structure table + weakest-constraint diagnostic + op_nudge), ALWAYS with a per-hit content
    snippet (`snippets=True`, forced — a content-blind listing would handicap this cell vs the
    bm25 baseline's opening-snippet listing on the search-axis comparison).
    visit_v(rank_or_id) -> the FULL text of a ranked document (capped), mirroring
    `Bm25Visit.visit`/`Bm25Visit._resolve` exactly (rank-or-doc_id resolution, numeric-doc_id
    handling, whole-doc token cap).

    So `IndriVisitWorkspace` (the `indri_visit` toolset) is a strict single-axis swap vs
    `research_bm25`: SAME read (whole-doc visit), SAME content-bearing
    listing shape, only the search ENGINE differs (graded Indri vs BM25 keyword) — disentangling
    "search interface" from "read granularity" (`research_bm25` and the plain `indri` arm
    conflate the two: bm25 = keyword search + whole-doc read; indri = graded search +
    section read)."""

    tools = ("isearch_v", "visit_v")

    def __init__(self, units: Sequence[CodeUnit],
                 executor: Optional[IndriExecutor] = None,
                 ubyid: Optional[dict] = None,
                 op_nudge: bool = True):
        super().__init__(units, executor=executor, ubyid=ubyid, op_nudge=op_nudge,
                         snippets=True)               # ALWAYS on for this arm — not a caller knob

    # -- visit_v: whole-doc read, mirroring Bm25Visit.visit/_resolve exactly ------------------

    def _resolve(self, ref):
        """Mirrors `Bm25Visit._resolve` exactly: a rank (int or numeric string) resolves
        against `self.last_hits`; a digit that is NOT a valid rank may itself be a real doc_id
        (numeric doc_ids do occur), so try that before erroring; otherwise fall back to a
        doc_id / title lookup."""
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
            rank = int(ref)
            if 1 <= rank <= len(self.last_hits):
                return self.last_hits[rank - 1], None
            if str(ref).strip() in self.ubyid:
                return str(ref).strip(), None
            if not self.last_hits:
                return None, f"ERROR: no prior search — rank {rank} has nothing to refer to."
            return None, (f"ERROR: rank {rank} out of range "
                          f"(last search had {len(self.last_hits)} results).")
        ref = (ref or "").strip()
        if ref in self.ubyid:
            return ref, None
        low = ref.lower()
        cands = [i for i, u in self.ubyid.items() if low in (u.title or u.qualname or "").lower()]
        if len(cands) == 1:
            return cands[0], None
        if cands:
            opts = ", ".join(f"{i} ({self.ubyid[i].title or self.ubyid[i].qualname})"
                             for i in cands[:5])
            return None, f"ERROR: ambiguous doc {ref!r} — did you mean: {opts}"
        return None, f"ERROR: no such doc {ref!r} — use a rank from the last search or a doc_id."

    def visit(self, rank_or_id) -> str:
        doc_id, err = self._resolve(rank_or_id)
        if err:
            return err
        self.seen.add(doc_id)
        u = self.ubyid[doc_id]
        text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                           " …(truncated — this is the whole-doc cap)")
        return f"{doc_id}  {(u.title or u.qualname or '')!r}:\n{text}"

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("isearch_v", "isearch", "search"):
                q = args.get("query") or args.get("q") or ""
                return self.search(q, int(args.get("k", 5) or 5))
            if name in ("visit_v", "visit"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."
