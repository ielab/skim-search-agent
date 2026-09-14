"""`search_bm25`: a keyword query over the flat document text, ranked by BM25.

The listing shows rank, id, title and a snippet per hit. The snippet is a method from
`agent_search/snippets/` (`snippet=`): the opening line by default, `TermWindow()` for the
best-matching window for the query (`bm25q_search`). With `full_text=True` (the AutoRead strategy) every hit is rendered in full,
capped at `MAX_VISIT_TOKENS`, so a search is also the read and no read tool is needed. The
engine is the run's Lucene BM25 (`agent_search/retrievers/lexical/pyserini.py`).

With `structure=True` (the search-fetch family, `bm25_search`/`bm25_search_snip`) the listing
shows structure instead of a body snippet: each hit's section names and infobox keys, no
text, so the paired `fetch` tool (agent_search.tools.fetch) can pull a named section. `k` is
then also readable per call, and the pool defaults to `BM25_FETCH_TOPK` instead of
`BM25_VISIT_TOPK`. There the snippet defaults to none; `snippet=TermWindow()` adds the
best-matching excerpt to the structure row (`bm25_search_snip`).
"""
from __future__ import annotations

from typing import Optional

from agent_search.tools.budgets import AUTOREAD_TOPK, BM25_FETCH_TOPK, BM25_VISIT_TOPK, MAX_VISIT_TOKENS
from agent_search.tools.common import _cap_tokens, _infobox, _INTRO, sections_from_body, structure_str
from agent_search.corpus.units import code_tokenize
from agent_search.snippets import NoSnippet, OpeningLine, Snippet
from agent_search.tools.base import Tool, query_text


class SearchBm25(Tool):
    name = "bm25_search"
    description = "Keyword search over the document corpus (BM25); returns ranked documents (with `fetch` available: each hit's title + section names + infobox keys; otherwise title + a short opening snippet). Depending on the toolset: `visit` a ranked doc for its full text, `fetch` a named section of a ranked doc, or (bm25_dci) `bash`/`read` the ranked docs directly — no other documents are reachable."
    parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "A keyword query, for example: treaty that ended the Mexican-American War."}}, "required": ["query"]}
    engines = ("bm25",)

    # the structure-listing text (`structure=True`), with and without the excerpt sentence
    # an excerpt-showing snippet adds. Set onto `self.description`/`self.parameters` in `__init__`; the
    # plain (structure=False) class attributes above stay untouched.
    STRUCTURE_DESCRIPTION = ("Keyword search over the document corpus (BM25); returns ranked "
                            "docs plus their section names + infobox keys{snip} — NOT full "
                            "text; fetch a named section to read. `fetch` a named section (or "
                            "the infobox) of a ranked doc — no other documents are reachable.")
    STRUCTURE_SNIPPET_FRAGMENT = " + a best-matching excerpt for your query"
    STRUCTURE_PARAMETERS = {"type": "object",
                            "properties": {
                                "query": {"type": "string",
                                          "description": "A keyword query, for example: treaty "
                                                          "that ended the Mexican-American War."},
                                "k": {"type": "integer",
                                     "description": "Max ranked candidates to return (default 5)."}},
                            "required": ["query"]}

    # options a strategy sets
    aliases = ("search", "bm25_search", "bm25q_search", "bm25_read_search")
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
            self.refusals = {n: 'ERROR: no visit tool in this condition — search already returns full documents.' for n in ('visit', 'visit_q', 'visit_d', 'visit_v', 'visit_h', 'visit_bv', 'visit_bqld', 'fetch')}
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
        k = call_k or self.k or BM25_FETCH_TOPK
        ids = list(self.engine["bm25"].search(query, k=k))
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
            sec_str, ib_str = structure_str(named, list(_infobox(u)))
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
        k = self.k or (AUTOREAD_TOPK if self.full_text else BM25_VISIT_TOPK)
        ids = self.engine["bm25"].search(query, k=k)
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


__all__ = ["SearchBm25"]
