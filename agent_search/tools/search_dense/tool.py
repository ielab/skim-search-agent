"""`search_dense`: a query embedded by the run's dense model, ranked by cosine similarity.

The listing is rendered exactly like `search_bm25`'s (rank, id, title, opening snippet); with
`full_text=True` (AutoRead) every hit is rendered in full, capped at `MAX_VISIT_TOKENS`. The
engine is the run's dense model over its persisted embedding cache; nothing is encoded during
a run.

With `structure=True` (the search-fetch family, `dense_search_f`/`dense_search_fp`) the
listing shows structure instead of a body snippet: each hit's section names and infobox keys,
so the paired `fetch` tool (agent_search.tools.fetch) can pull a named section. `k` is then
also readable per call, and the pool defaults to `DENSE_FETCH_TOPK` instead of
`DENSE_VISIT_TOPK`. `snippets=True` adds a one-line best-matching excerpt to that structure
row (`dense_search_f`); `dense_search_fp` (`snippets=False`) is the same listing with the
excerpt left off. `structure=False` ignores `snippets`.
"""
from __future__ import annotations

from agent_search.corpus.units import code_tokenize
from agent_search.tools.base import Tool
from agent_search.tools.budgets import AUTOREAD_TOPK, DENSE_FETCH_TOPK, DENSE_VISIT_TOPK, MAX_VISIT_TOKENS
from agent_search.tools.common import _INTRO, _cap_tokens, _infobox, best_line, opening_line, sections_from_body


class SearchDense(Tool):
    name = "dense_search"
    description = "Semantic similarity search over the document corpus (dense embeddings, cosine similarity); returns ranked documents (title + a short opening snippet). `visit` a ranked doc for its full text — no other documents are reachable."
    parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "A natural-language query describing what you are looking for, for example: treaty that ended the Mexican-American War."}}, "required": ["query"]}
    engines = ("dense",)

    # the tools.yaml text for the structure listing (`structure=True`) — dense_search_f's
    # text, used for BOTH dense_search_f and dense_search_fp (the latter is not in tools.yaml;
    # it gets dense_search_f's declaration, per that toolset's own comment).
    STRUCTURE_DESCRIPTION = ("Semantic similarity search over the document corpus (dense "
                            "embeddings, cosine similarity); returns ranked docs plus their "
                            "section names + infobox keys + a best-matching excerpt for your "
                            "query — NOT full text; fetch a named section to read. `fetch` a "
                            "named section (or the infobox) of a ranked doc — no other "
                            "documents are reachable.")
    STRUCTURE_PARAMETERS = {"type": "object",
                            "properties": {
                                "query": {"type": "string",
                                          "description": "A natural-language query describing "
                                                          "what you are looking for, for "
                                                          "example: treaty that ended the "
                                                          "Mexican-American War."},
                                "k": {"type": "integer",
                                     "description": "Max ranked candidates to return (default 10)."}},
                            "required": ["query"]}

    aliases = ("search", "dense_search", "dense_read_search")
    full_text: bool = False
    k: int = 0
    structure: bool = False        # list structure (sections/infobox), paired with `fetch`
    snippets: bool = False         # structure=True only: a best-matching excerpt per hit

    def __init__(self, name=None, **options):
        super().__init__(name, **options)
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
        k = call_k or self.k or DENSE_FETCH_TOPK
        ids = list(self.engine["dense"].top_k_doc_ids(query, k=k) or [])
        if not ids:
            prior = "  (previous results still available to fetch)" if state.last_hits else ""
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        state.last_hits = list(ids)
        terms = code_tokenize(query) if self.snippets else None
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
            if self.snippets:
                snip = best_line(u, terms)
                if snip:
                    line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, args: dict) -> str:
        query = (args.get("query") or args.get("q") or "")
        if isinstance(query, list):
            query = query[0] if query else ""
        query = str(query).strip()
        if not query:
            return "empty query"
        if self.structure:
            return self._run_structure(query, args)
        k = self.k or (AUTOREAD_TOPK if self.full_text else DENSE_VISIT_TOPK)
        ids = list(self.engine["dense"].top_k_doc_ids(query, k=k) or [])
        state = self.state
        state.previous_queries.append(query)
        if not ids:
            prior = ("  (previous results still available)" if self.full_text else
                     "  (previous results still available to fetch)") if state.last_hits else ""
            return f"search: {query}   (0 matches){prior}"
        state.last_hits = ids
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
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {opening_line(u)}…")
        return "\n".join(lines)


__all__ = ["SearchDense"]
