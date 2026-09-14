"""`search_hybrid`: BM25 and dense rankings fused by Reciprocal Rank Fusion.

The plain listing's declaration is deliberately short and takes only `query`. The earlier
declaration spelled out the fusion (pool sizes, RRF constant) and offered a `k` argument the
plain path never read; under that declaration Tongyi opened 20% of its turns with an empty
<tool_call></tool_call> (39% of episodes at step 0). The short declaration brings that to 0.1%,
the same as the BM25 and dense tools (A/B on 40 BrowseComp-Plus questions, 2026-09-13).

Each engine is queried to a pool of `HYBRID_POOL` documents; the two pools are fused with RRF
(constant `RRF_K`) and the top `k` are listed like `search_bm25`'s results. With
`full_text=True` (AutoRead) every hit is rendered in full, capped at `MAX_VISIT_TOKENS`.

With `structure=True` (the search-fetch family, `hybrid_search_snip`) the listing shows
structure instead of a body snippet: each hit's section names and infobox keys plus a
one-line best-matching excerpt, so the paired `fetch` tool (agent_search.tools.fetch) can
pull a named section. `k` is then also readable per call, and the post-fusion pool defaults
to `HYBRID_FETCH_TOPK` instead of `HYBRID_VISIT_TOPK`. The snippet is a method from
`agent_search/snippets/` (`snippet=`, the query-term window there by default; `NoSnippet()`
leaves the excerpt off the structure row).
"""
from __future__ import annotations

from typing import Optional

from agent_search.corpus.units import code_tokenize
from agent_search.snippets import NoSnippet, OpeningLine, Snippet, TermWindow
from agent_search.tools.base import Tool, query_text
from agent_search.tools.budgets import AUTOREAD_TOPK, HYBRID_FETCH_TOPK, HYBRID_POOL, HYBRID_VISIT_TOPK, MAX_VISIT_TOKENS
from agent_search.tools.common import _INTRO, _cap_tokens, _infobox, sections_from_body


class SearchHybrid(Tool):
    name = "hybrid_search"
    description = "Keyword and semantic search over the document corpus (BM25 and dense embeddings fused); returns ranked documents (title + a short opening snippet). `visit_h` a ranked doc for its full text — no other documents are reachable."
    parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "A keyword or natural-language query, for example: treaty that ended the Mexican-American War."}}, "required": ["query"]}
    engines = ("hybrid",)

    # the structure-listing text (`structure=True`).
    STRUCTURE_DESCRIPTION = ("Hybrid keyword+semantic search over the document corpus "
                            "(Reciprocal Rank Fusion, RRF k=60, over a top-100 canonical BM25 "
                            "pool and a top-100 dense-embedding pool). Returns ranked docs "
                            "plus their section names + infobox keys + a best-matching "
                            "excerpt for your query — NOT full text; fetch a named section to "
                            "read. `fetch` a named section (or the infobox) of a ranked doc — "
                            "no other documents are reachable.")
    STRUCTURE_PARAMETERS = {"type": "object",
                            "properties": {
                                "query": {"type": "string",
                                          "description": "A keyword or natural-language query, "
                                                          "for example: treaty that ended the "
                                                          "Mexican-American War."},
                                "k": {"type": "integer",
                                     "description": "Max fused candidates to return (default 5)."}},
                            "required": ["query"]}

    aliases = ("search", "hybrid_search", "hybrid_read_search")
    full_text: bool = False
    k: int = 0
    pool: int = HYBRID_POOL
    structure: bool = False        # list structure (sections/infobox), paired with `fetch`
    # the excerpt under each hit (agent_search.snippets): the opening line for the plain listing,
    # the query-term window for the structure listing, unless the strategy says otherwise
    snippet: Optional[Snippet] = None

    def __init__(self, name=None, **options):
        super().__init__(name, **options)
        if self.snippet is None:
            self.snippet = TermWindow() if self.structure else OpeningLine()
        if self.full_text:
            self.refusals = {n: 'ERROR: no visit tool in this condition — search already returns full documents.' for n in ('visit', 'visit_q', 'visit_d', 'visit_v', 'visit_h', 'visit_bv', 'visit_bqld', 'fetch')}
        if self.structure:
            self.description = self.STRUCTURE_DESCRIPTION
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
        k = call_k or self.k or HYBRID_FETCH_TOPK
        ids = list(self.engine["hybrid"].search(query, k))
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
        k = self.k or (AUTOREAD_TOPK if self.full_text else HYBRID_VISIT_TOPK)
        ids = list(self.engine["hybrid"].search(query, k))
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
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            state.seen.add(i)
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {self.snippet.render(u, code_tokenize(query))}…")
        return "\n".join(lines)


__all__ = ["SearchHybrid"]
