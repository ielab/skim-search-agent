"""Retrieve-then-fetch workspaces: a live search pool (bm25, dense, or hybrid) paired with
DocSearchFetch's named-section fetch.

`Bm25FetchWorkspace`/`DenseFetchWorkspace`/`HybridFetchSnipWorkspace` re-run their retrieval
live on every search call and render a structure table (section names + infobox keys, no
body); `fetch` delegates verbatim to `DocSearchFetch.fetch`. `Bm25FetchSnipWorkspace` adds a
best-matching excerpt to the bm25 listing; `DenseFetchPlainWorkspace` removes it from the
dense listing. All five subclass `DocSearchFetch` for the fetch machinery but replace its
BQL retrieval with their own engine.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.core.seen import OrderedSeen
from agent_search.corpus.units import CodeUnit, code_tokenize

from .budgets import BM25_FETCH_TOPK, DENSE_FETCH_TOPK, HYBRID_FETCH_TOPK, HYBRID_POOL
from .common import _INTRO, _infobox, rrf_fuse
from .sieve import DocSearchFetch

class Bm25FetchWorkspace(DocSearchFetch):
    """The RISE-style hybrid: bm25 LIVE RETRIEVAL + BQL structured SECTION-FETCH read.

    This is `DocSearchFetch`'s READ side (search lists STRUCTURE — section names + infobox
    keys, no bodies; fetch pulls one named section) placed on `Bm25Visit`'s RETRIEVAL (plain
    BM25 over flat doc text). So the ONLY difference from `research` (the method) is the
    retrieval surface — bm25 instead of the BQL executor — and the ONLY difference from
    `research_bm25` is the read (a named section instead of the whole doc). Completing the
    controlled set with `research_bm25_dci` (same bm25 retrieval, shell read):

        research           : BQL search  -> fetch a section   (the method)
        research_bm25      : bm25 search  -> visit whole doc   (retrieve-then-visit)
        research_bm25_dci  : bm25 search  -> bash/read shell   (retrieval-bounded brute-force)
        research_bm25_fetch: bm25 search  -> fetch a section   (this — retrieval-bounded structured read)

    Retrieval is LIVE, per `bm25_search` call — the SAME retrieval BEHAVIOR as
    research_bm25/Bm25Visit for the same query/engine/k (each call re-runs bm25 on the
    agent's own query string; nothing is fixed at construction). This makes the controlled
    comparison honest: the ONLY difference from `research_bm25` is the READ (named-section
    fetch vs whole-doc visit), and the ONLY difference from `research` is the RETRIEVAL
    surface (plain BM25 vs field-tagged Boolean) — not "one arm re-retrieves and the other
    doesn't". Uncoached (no BQL skill — retrieval is a plain search box)."""

    # tools.yaml's `bm25_fetch` toolset names the retrieval tool `bm25_search` (matching the
    # other bm25 arms, since retrieval is the same operation); `fetch` is DocSearchFetch's.
    tools = ("bm25_search", "fetch")

    def __init__(self, units: Sequence[CodeUnit], query: str, engine=None,
                 ubyid: Optional[dict] = None, topk: int = BM25_FETCH_TOPK):
        # DocSearchFetch.__init__ builds the section cache + resolves ubyid; we do NOT need its
        # BQL StructuralExecutor (retrieval is bm25 here), so pass a lightweight executor stand-in
        # is avoided — instead call super() and simply never use self.ex. But super() would build
        # a StructuralExecutor over all units (O(N) postings) we never query, so set it up by hand.
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.ex = None                                   # unused: retrieval is bm25, not BQL
        self._sections: dict[str, dict] = {}
        self.seen = OrderedSeen()
        # DocSearchFetch's own constructor defaults — set by hand here since this __init__
        # skips super().__init__ (see comment above). Without these, any inherited method that
        # reads them (`_render_hits`'s `self.snippets` check, `search`'s `self.date_nudge`/
        # `self._date_nudge_emitted` gate) would raise AttributeError if ever reached on this
        # class or a subclass of it.
        self.snippets = False
        self.date_nudge = False
        self._date_nudge_emitted = 0
        if engine is None:
            # SAME env BM25_BACKEND fallback as Bm25Visit above — see that constructor's comment.
            from agent_search.retrievers.lexical import build_bm25_engine
            engine = build_bm25_engine(self.units)
        self.bm = engine
        self.topk = topk
        # No retrieval at construction: `self.query` is kept only as the `bm25_search` fallback
        # when a tool call omits `query` (mirrors the episode's opening question); every actual
        # `bm25_search` call retrieves LIVE against the query it was given (see search() below).
        self.query = (query or "").strip()
        self.last_hits: list[str] = []

    # -- bm25_search: LIVE bm25 retrieval, rendered as a STRUCTURE table (no bodies) -------

    def search(self, query: str, k: Optional[int] = None) -> str:
        # LIVE per call — exactly like Bm25Visit.search (SAME engine/call shape, so retrieval
        # matches research_bm25 for the same query/k). Same STRUCTURE listing as
        # DocSearchFetch.search (title + §[section names] + ib[infobox keys], NO bodies), so the
        # agent knows what to fetch; just driven by bm25 hits, not BQL leaves (so no `matched:`
        # field — bm25 has no field-tagged leaves to attribute).
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.bm.search(query, k=k or self.topk))
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            lines.append(f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("bm25_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim to DocSearchFetch's fetch (arg-normalization +
                # section logic) — the whole point is that only retrieval differs from `research`.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class Bm25FetchSnipWorkspace(Bm25FetchWorkspace):
    """`Bm25FetchWorkspace` (research_bm25_fetch) with a CONTENT-BEARING search listing —
    the fair-listing sibling that completes the grid. Every other method fetch cell
    (research_snip, research_indri_snip, research_dense_fetch) shows each hit's best-matching
    excerpt (`best_line`) in its search listing; research_bm25_fetch's listing was
    CONTENT-BLIND (structure only — no excerpt), an inconsistency across the grid. This
    workspace fixes exactly that, ONE deliberate departure from `Bm25FetchWorkspace`'s bare
    structure table — the SAME amendment `DenseFetchWorkspace` makes over `Bm25FetchWorkspace`'s
    own bare table (see its docstring) and `IndriVisitWorkspace` makes over
    `IndriFetchWorkspace`'s default-off snippets: a content-blind listing handicaps a cell once
    ANY sibling cell shows content.

    Retrieval, fetch, and construction are ALL inherited from `Bm25FetchWorkspace` UNCHANGED
    (same live-per-call bm25 search, same `topk`/`engine`/`query` construction, same
    `fetch`-delegates-to-DocSearchFetch machinery) — `search` is the ONLY override, and it is
    `Bm25FetchWorkspace.search` plus ONE appended `» excerpt` line per hit
    (`_best_line`, the SAME module-level `best_line` window-scoring research_snip/
    research_indri_snip/research_dense_fetch already share). Excerpt terms come
    from the raw query text (`code_tokenize`) — bm25 has no BQL leaves to draw from, the SAME
    term source `Bm25Visit.query_biased`/`DenseFetchWorkspace` use. So the ranking is
    IDENTICAL to `Bm25FetchWorkspace` for the same query/engine/k (same retrieval call, same
    top-k doc ids) — the ONLY difference from `research_bm25_fetch` is the listing's content,
    matching this cell's factorial-grid position exactly: {bm25 search} x {structure->parts
    read}, now WITH the same fairness-parity excerpt every other fetch cell carries."""

    # tools.yaml's `bm25_fetch_snip` toolset names the retrieval tool `bm25_search_snip`
    # (distinguishing it from `bm25_search`, the content-blind sibling's tool, so each gets its
    # own manual/description entry — SAME naming move `dense_search_f` makes over `dense_search`);
    # `fetch` is DocSearchFetch's, reused unmodified.
    tools = ("bm25_search_snip", "fetch")

    # -- bm25_search_snip: LIVE bm25 retrieval, rendered as a STRUCTURE table + excerpt -------

    def search(self, query: str, k: Optional[int] = None) -> str:
        # IDENTICAL retrieval call to Bm25FetchWorkspace.search (same engine, same query, same
        # topk fallback) — ranking parity is a property of this line alone; only the rendering
        # below differs (an appended excerpt per hit).
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.bm.search(query, k=k or self.topk))
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self._best_line(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("bm25_search_snip", "bm25_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim, exactly like Bm25FetchWorkspace — the only
                # difference from `research_bm25_fetch` is the search listing's content.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class DenseFetchWorkspace(DocSearchFetch):
    """The DENSE-retrieval structured-read ACI: dense embedding LIVE RETRIEVAL + BQL structured
    SECTION-FETCH read. This fills the search x read factorial cell {dense search} x
    {structure->parts read} — Bm25FetchWorkspace already fills {bm25 search} x {structure->parts
    read} (research_bm25_fetch), DenseVisit fills {dense search} x {whole-doc visit}
    (research_dense). Wired EXACTLY like Bm25FetchWorkspace (see its docstring): retrieval is
    LIVE per `dense_search_f` call (`engine.top_k_doc_ids(query, k)` re-run on the agent's OWN
    query text every time — nothing fixed at construction, same live-retrieval fix
    Bm25FetchWorkspace already made over the old one-shot design); the section-fetch machinery
    (`fetch`) is INHERITED from `DocSearchFetch` unchanged, never duplicated.

    ONE deliberate departure from Bm25FetchWorkspace's bare structure table: research_snip
    established that a content-bearing listing (a one-line best-matching excerpt per hit,
    `best_line` — the SAME window-scoring research_snip/research_indri_snip use)
    is the fair default once ANY arm shows one, and `dense_search_f`'s tool description promises
    exactly that ("... plus a best-matching excerpt — NOT full text; fetch a named section to
    read"). So this workspace's listing is Bm25FetchWorkspace's STRUCTURE table (rank, doc_id,
    title, §[section names], ib[infobox keys]) with ONE appended `» excerpt` line per hit.
    Excerpt terms come from the raw query text (`code_tokenize`) — dense retrieval has no BQL
    leaves to draw from, the same term source Bm25Visit's `query_biased` excerpt mode uses.

    `engine` is a `DenseBelief` already `build_or_load`ed over this corpus (the SAME persisted
    `indexes/dense/<model>-sl<len>/<corpus_key>/` cache DenseVisit/the `dense` retriever
    condition already build) — retriever.py's 'densefetch' arm builds + validates that cache at
    index() time and raises the SAME clear RuntimeError as the 'densevisit' arm if it's missing
    (this baseline never live-encodes a whole corpus at eval time). Uncoached (no skill) — like
    Bm25FetchWorkspace, retrieval is a plain search box and section-fetch needs no BQL skill (its
    listing already names the sections)."""

    # tools.yaml's `dense_fetch` toolset names the retrieval tool `dense_search_f` (distinguishing
    # it from `dense_search`, DenseVisit's whole-doc-read tool, so each gets its own manual/entry
    # even though neither renders one — UNCOACHED); `fetch` is DocSearchFetch's.
    tools = ("dense_search_f", "fetch")

    def __init__(self, units: Sequence[CodeUnit], query: str, engine=None,
                 ubyid: Optional[dict] = None, topk: int = DENSE_FETCH_TOPK,
                 corpus_key: Optional[str] = None):
        # SAME construction shape as Bm25FetchWorkspace.__init__: DocSearchFetch.__init__ would
        # build a StructuralExecutor over all units (O(N) postings) we never query (retrieval is
        # dense, not BQL) — set the essentials up by hand instead of calling super().__init__.
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.ex = None                                   # unused: retrieval is dense, not BQL
        self._sections: dict[str, dict] = {}
        self.seen = OrderedSeen()
        # DocSearchFetch's own constructor defaults — see Bm25FetchWorkspace.__init__'s comment.
        self.snippets = False
        self.date_nudge = False
        self._date_nudge_emitted = 0
        if engine is None:
            from agent_search.retrievers.indri.dense_belief import DenseBelief
            engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.engine = engine
        self.topk = topk
        # No retrieval at construction: `self.query` is kept only as the `dense_search_f`
        # fallback when a tool call omits `query` (mirrors Bm25FetchWorkspace.query); every
        # actual call retrieves LIVE against the query it was given (see search() below).
        self.query = (query or "").strip()
        self.last_hits: list[str] = []

    # -- dense_search_f: LIVE dense retrieval, rendered as a STRUCTURE table + excerpt -------

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k or self.topk) or [])
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self._best_line(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_search_f", "dense_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim to DocSearchFetch's fetch, exactly like
                # Bm25FetchWorkspace — the only difference from `research` is retrieval, and the
                # only difference from `research_bm25_fetch` is the retrieval ENGINE.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class DenseFetchPlainWorkspace(DenseFetchWorkspace):
    """`DenseFetchWorkspace` (research_dense_fetch, "dense+snip fetch") with the excerpt REMOVED
    — the missing PLAIN (no-excerpt) dense fetch cell. Every other query engine's fetch family
    already has both a plain and a snip sibling: Bm25FetchWorkspace (research_bm25_fetch, plain)
    vs Bm25FetchSnipWorkspace (research_bm25_fetch_snip, excerpt); IndriFetchWorkspace's own
    `snippets=False` default (the plain `indri` arm) vs `snippets=True` (research_indri_snip); the BQL
    method's plain `search`/`fetch` (research) vs `DocSearchFetch(snippets=True)`
    (research_snip) — and the SAME split for the dense-fused BQL executor,
    `DocSearchFetch()`/'bqldensefetch' (research_bql_dense_fetch) vs
    `DocSearchFetch(snippets=True)`/'bqldensesnip' (research_bql_dense_snip). `DenseFetchWorkspace`
    alone never got the split — its `search()` has no `snippets` flag at all and appends the
    `» excerpt` line unconditionally — so dense was the one query engine (bm25/bql/indri/dense)
    lacking a plain fetch cell. This workspace supplies exactly that missing sibling, the same
    way `research_bql_dense_fetch` supplies `research_bql_dense_snip`'s missing plain sibling for
    the dense-fused BQL executor (see that condition's doc_research.py/conditions.yaml comments).

    Retrieval, construction, and the section-`fetch` read are ALL inherited from
    `DenseFetchWorkspace` UNCHANGED (same live-per-call `engine.top_k_doc_ids`, same
    `topk`/`engine`/`corpus_key` construction, same `fetch`-delegates-to-DocSearchFetch
    machinery) — `search` is the ONLY override, and it is
    `DenseFetchWorkspace.search` with the `code_tokenize`/`_best_line` excerpt computation and its
    appended `» excerpt` line removed; everything else (rank/doc_id/title/§sections/ib[infobox])
    renders identically. Ranking is therefore IDENTICAL to `research_dense_fetch` for the same
    query/engine/k — the ONLY difference from `research_dense_fetch` is the listing's content (no
    per-hit excerpt), isolating exactly what the snippet contributes on top of pure dense
    retrieval + structured section-fetch, the SAME isolation `research_bql_dense_fetch` performs
    for the dense-fused BQL executor. `DenseFetchWorkspace` itself is left completely untouched —
    this is a new sibling class, not a modification."""

    # tools.yaml's `dense_fetch_plain` toolset names the retrieval tool `dense_search_fp`
    # (distinguishing it from `dense_search_f`, the excerpt-bearing sibling's tool, so each gets
    # its own manual/description entry — SAME naming move `dense_search_f` makes over
    # `dense_search`); `fetch` is DocSearchFetch's, reused unmodified.
    tools = ("dense_search_fp", "fetch")

    # -- dense_search_fp: LIVE dense retrieval, rendered as a PLAIN STRUCTURE table (no excerpt) --

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k or self.topk) or [])
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            # NO `» excerpt` line — the ONLY rendering difference from DenseFetchWorkspace.search.
            lines.append(f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_search_fp", "dense_search_f", "dense_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim, exactly like DenseFetchWorkspace/Bm25FetchWorkspace
                # — the only difference from `research_dense_fetch` is the listing's content.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class HybridFetchSnipWorkspace(DocSearchFetch):
    """The BM25+DENSE HYBRID {search} x {structure->parts read} cell (research_hybrid_fetch_snip)
    — the fetch-mode twin of `HybridVisit`, mirroring `Bm25FetchSnipWorkspace`'s listing shape
    exactly: the SAME RRF-fused retrieval as `HybridVisit` (see its docstring / `rrf_fuse` for
    the formula and `HYBRID_POOL`), rendered as the bm25/dense fetch cells' STRUCTURE table
    (rank, doc_id, title, §[section names], ib[infobox keys]) plus a one-line query-biased
    best-matching excerpt per hit (`_best_line`, inherited from `DocSearchFetch`), paired with
    the SAME structured section-`fetch` read `research`/`research_bm25_fetch`/
    `research_dense_fetch` use (inherited from `DocSearchFetch`, unmodified — delegated
    verbatim in `run`, exactly like `Bm25FetchWorkspace`/`DenseFetchWorkspace`).

    Retrieval is LIVE per `hybrid_search_snip` call — BOTH pools are re-queried on the agent's
    own query text every call, nothing fixed at construction — mirroring `Bm25FetchWorkspace`/
    `DenseFetchWorkspace`'s live-per-call retrieval exactly. Uncoached (no skill) — like every
    other bm25/dense-family fetch arm; section-fetch needs no BQL skill (its listing already
    names the sections)."""

    tools = ("hybrid_search_snip", "fetch")

    def __init__(self, units: Sequence[CodeUnit], query: str, bm25_engine=None,
                 dense_engine=None, ubyid: Optional[dict] = None,
                 topk: int = HYBRID_FETCH_TOPK, pool: int = HYBRID_POOL,
                 corpus_key: Optional[str] = None):
        # SAME manual-construction shape as Bm25FetchWorkspace/DenseFetchWorkspace: skip
        # DocSearchFetch.__init__'s BQL StructuralExecutor build (retrieval here is bm25+dense
        # fusion, not BQL) and set the essentials up by hand.
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.ex = None                                   # unused: retrieval is bm25+dense, not BQL
        self._sections: dict[str, dict] = {}
        self.seen = OrderedSeen()
        # DocSearchFetch's own constructor defaults — see Bm25FetchWorkspace.__init__'s comment.
        self.snippets = False
        self.date_nudge = False
        self._date_nudge_emitted = 0
        if bm25_engine is None:
            from agent_search.retrievers.lexical import build_bm25_engine
            bm25_engine = build_bm25_engine(self.units)
        self.bm = bm25_engine
        if dense_engine is None:
            from agent_search.retrievers.indri.dense_belief import DenseBelief
            dense_engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.dense_engine = dense_engine
        self.topk = topk
        self.pool = pool
        # No retrieval at construction: `self.query` is kept only as the `hybrid_search_snip`
        # fallback when a tool call omits `query` (mirrors Bm25FetchWorkspace.query); every
        # actual call retrieves LIVE against the query it was given (see search() below).
        self.query = (query or "").strip()
        self.last_hits: list[str] = []

    # -- hybrid_search_snip: LIVE bm25+dense retrieval, RRF-fused, rendered as a STRUCTURE
    # table + excerpt (IDENTICAL rendering shape to Bm25FetchSnipWorkspace.search/
    # DenseFetchWorkspace.search — only the ranking differs) -------------------------------

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        bm25_ids = list(self.bm.search(query, k=self.pool))
        dense_ids = list(self.dense_engine.top_k_doc_ids(query, k=self.pool) or [])
        ids = rrf_fuse(bm25_ids, dense_ids, topk=k or self.topk)
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self._best_line(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("hybrid_search_snip", "hybrid_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim, exactly like Bm25FetchWorkspace/DenseFetchWorkspace
                # — the only difference from those cells is the retrieval RANKING, not the read.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# --- arms
from agent_search.legacy.retriever import register_workspace  # noqa: E402

register_workspace("bm25fetch", tools=("bm25_search", "fetch"), engines=("bm25",),
                   builder=lambda ctx: Bm25FetchWorkspace(ctx.units, ctx.query, engine=ctx.bm25(), ubyid=ctx.ubyid))
register_workspace("bm25fetchsnip", tools=("bm25_search_snip", "fetch"), engines=("bm25",),
                   builder=lambda ctx: Bm25FetchSnipWorkspace(ctx.units, ctx.query, engine=ctx.bm25(), ubyid=ctx.ubyid))
register_workspace("densefetch", tools=("dense_search_f", "fetch"), engines=("dense",),
                   builder=lambda ctx: DenseFetchWorkspace(ctx.units, ctx.query, engine=ctx.dense(), ubyid=ctx.ubyid,
                                                           corpus_key=ctx.corpus_key))
register_workspace("densefetchplain", tools=("dense_search_fp", "fetch"), engines=("dense",),
                   builder=lambda ctx: DenseFetchPlainWorkspace(ctx.units, ctx.query, engine=ctx.dense(), ubyid=ctx.ubyid,
                                                                corpus_key=ctx.corpus_key))
register_workspace("hybridfetchsnip", tools=("hybrid_search_snip", "fetch"), engines=("bm25", "dense"),
                   builder=lambda ctx: HybridFetchSnipWorkspace(ctx.units, ctx.query, bm25_engine=ctx.bm25(),
                                                                dense_engine=ctx.dense(), ubyid=ctx.ubyid,
                                                                corpus_key=ctx.corpus_key))
