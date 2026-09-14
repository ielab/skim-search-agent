"""`search_reranked`: a query over the document corpus, a pool from one retriever, the pool
reordered by a reranker.

The engine is the run's reranked engine (`agent_search/retrievers/reranked.py`): the base
retriever (`RERANK_BASE`, default `bm25`) supplies `RERANK_POOL` candidates and the reranker
(`RERANK_METHOD`, `RERANK_MODEL`) reads each (query, document) pair. The listing is the same
as `search_bm25`'s: rank, id, title and a snippet per hit, the snippet a method from
`agent_search/snippets/` (`snippet=`, the opening line by default). With `full_text=True` every
hit is rendered in full, capped at `MAX_VISIT_TOKENS`. With `structure=True` the listing shows
each hit's section names and infobox keys for the paired `fetch` tool, `k` is readable per
call, and the pool defaults to `RERANK_FETCH_TOPK` instead of `RERANK_VISIT_TOPK`; there the
snippet defaults to none.
"""
from __future__ import annotations

from typing import Optional

from agent_search.tools.budgets import AUTOREAD_TOPK, MAX_VISIT_TOKENS, RERANK_FETCH_TOPK, RERANK_VISIT_TOPK
from agent_search.tools.common import _INTRO, _cap_tokens, _infobox, sections_from_body
from agent_search.corpus.units import code_tokenize
from agent_search.snippets import NoSnippet, OpeningLine, Snippet
from agent_search.tools.base import Tool, query_text


class SearchReranked(Tool):
    name = "reranked_search"
    description = "Search over the document corpus: a first-stage retriever finds candidates and a reranker model reads each one against your query and reorders them; returns ranked documents (title + a short opening snippet). `visit_r` a ranked doc for its full text — no other documents are reachable."
    parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "A keyword or natural-language query, for example: treaty that ended the Mexican-American War."}}, "required": ["query"]}
    engines = ("reranked",)

    STRUCTURE_DESCRIPTION = ("Search over the document corpus (a first-stage retriever, then a reranker "
                            "model reads each candidate against your query); returns ranked docs plus "
                            "their section names + infobox keys{snip} — NOT full text; fetch a named "
                            "section to read. `fetch` a named section (or the infobox) of a ranked doc — "
                            "no other documents are reachable.")
    STRUCTURE_SNIPPET_FRAGMENT = " + a best-matching excerpt for your query"
    STRUCTURE_PARAMETERS = {"type": "object",
                            "properties": {
                                "query": {"type": "string",
                                          "description": "A keyword or natural-language query, for example: "
                                                          "treaty that ended the Mexican-American War."},
                                "k": {"type": "integer",
                                     "description": "Max ranked candidates to return (default 5)."}},
                            "required": ["query"]}

    # options a strategy sets
    full_text: bool = False        # render every hit in full (AutoRead); no read tool needed
    k: int = 0                     # results per search; 0 = the listing knob for this mode
    structure: bool = False        # list structure (sections/infobox), paired with `fetch`
    # the excerpt under each hit (agent_search.snippets): the opening line for the plain listing,
    # nothing for the structure listing, unless the strategy says otherwise
    snippet: Optional[Snippet] = None

    def __init__(self, name=None, **options):
        super().__init__(name, **options)
        if self.snippet is None:
            self.snippet = NoSnippet() if self.structure else OpeningLine()
        if self.full_text:
            self.refusals = {n: 'ERROR: no visit tool in this condition — search already returns full documents.' for n in ('visit', 'visit_q', 'visit_d', 'visit_v', 'visit_h', 'visit_r', 'visit_bv', 'visit_bqld', 'fetch')}
        if self.structure:
            snip = self.STRUCTURE_SNIPPET_FRAGMENT if self.snippet.shows_excerpt else ""
            self.description = self.STRUCTURE_DESCRIPTION.format(snip=snip)
            self.parameters = self.STRUCTURE_PARAMETERS

    # -- section/infobox cache, shared with the paired `fetch` tool via state.listing -------

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

    def _run_structure(self, query: str, args: dict) -> str:
        state = self.state
        state.previous_queries.append(query)
        call_k = args.get("k")
        try:
            call_k = int(call_k) if call_k else None
        except (TypeError, ValueError):
            call_k = None
        k = call_k or self.k or RERANK_FETCH_TOPK
        ids = list(self.engine["reranked"].search(query, k=k))
        if not ids:
            prior = "  (previous results still available to fetch)" if state.last_hits else ""
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        state.last_hits = list(ids)
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            state.seen.add(i)
            named = [s for s in self._secs(i) if s != _INTRO] or list(self._secs(i))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or i
            line = f"  {rank}  {i}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self.snippet.render(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, args: dict) -> str:
        query = query_text(args)
        if isinstance(query, list):
            query = query[0] if query else ""
        query = str(query).strip()
        if not query:
            return "empty query"
        if self.structure:
            return self._run_structure(query, args)
        k = self.k or (AUTOREAD_TOPK if self.full_text else RERANK_VISIT_TOPK)
        ids = self.engine["reranked"].search(query, k=k)
        state = self.state
        state.previous_queries.append(query)
        if not ids:
            prior = ("  (previous results still available)" if self.full_text else
                     "  (previous results still available to fetch)") if state.last_hits else ""
            return f"search: {query}   (0 matches){prior}"
        state.last_hits = list(ids)
        if self.full_text:
            blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
            for rank, i in enumerate(ids, start=1):
                u = self.ubyid.get(i)
                if u is None:
                    continue
                state.seen.add(i)
                state.reads.append(i)
                text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS, " …(truncated — this is the whole-doc cap)")
                blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
            return "\n\n".join(blocks)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        terms = code_tokenize(query)
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            state.seen.add(i)
            snip = self.snippet.render(u, terms)
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)


__all__ = ["SearchReranked"]
