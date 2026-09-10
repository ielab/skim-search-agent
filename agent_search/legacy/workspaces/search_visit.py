"""Retrieve-then-visit and retrieve-and-read workspaces: search a flat doc index, then read.

`Bm25Visit`/`DenseVisit`/`HybridVisit` show a title + opening-snippet listing (bm25, dense
cosine, and RRF-fused bm25+dense retrieval respectively) and read a whole document via
`visit`. `Bm25AutoRead`/`DenseAutoRead`/`HybridAutoRead` use the same three retrieval
rankings but render the top hits' full text directly from `search` — there is no separate
visit tool in those three. All six are uncoached baselines with no BQL structure.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from agent_search.core.seen import OrderedSeen
from agent_search.core.tokens import cap_tokens as _cap_tokens
from agent_search.corpus.units import CodeUnit, code_tokenize

from .budgets import (
    AUTOREAD_TOPK, BM25_VISIT_TOPK, DENSE_VISIT_TOPK, HYBRID_POOL, HYBRID_VISIT_TOPK,
    MAX_VISIT_TOKENS)
from .common import _SeenMixin, best_line, opening_line, rrf_fuse

class Bm25Visit(_SeenMixin):
    """The retrieve-then-visit baseline: search(query, k) is plain BM25 over the flat doc
    text (title + body); results show title + a short snippet, NO structure. visit(rank_or_id)
    returns the doc's FULL text (capped). Uncoached (no skill), like the code grep baseline.

    `query_biased` (DEFAULT OFF, set True by the bm25q arm): the search listing's per-hit
    snippet becomes a QUERY-BIASED best-matching excerpt (the SAME `best_line` window-scoring
    the method cells' snippets use — see doc_research.py's module-level `best_line`) instead of
    the doc's fixed OPENING slice. Motivated by fairness: real search engines show query-biased
    snippets in their result listing, and our method cells (research_snip/research_indri_snip)
    already do via `_best_line` — the bm25 baseline's opening-snippet listing would otherwise be
    a listing-axis advantage for OUR method that a hardened baseline must not concede. Callers
    other than the bm25q arm (agent/retriever.py) leave this off, keeping the plain
    opening-snippet listing."""

    # tools.yaml's toolset (and therefore the rendered <tools> block the agent sees) names
    # this tool `bm25_search` (distinguishing it from the method's `search`); `run()` accepts
    # both spellings so the workspace matches what a real episode's model actually calls.
    # the bm25q arm's toolset instead names it `bm25q_search`/`visit_q` — `__init__` swaps
    # `self.tools` to that pair when `query_biased=True` (an instance override, same pattern
    # DenseVisit uses at the class level for its own tool names).
    tools = ("bm25_search", "visit")

    def __init__(self, units: Sequence[CodeUnit], engine=None, ubyid: Optional[dict] = None,
                 query_biased: bool = False):
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if engine is None:
            # env BM25_BACKEND-selectable (default 'local', unchanged) — see
            # agent_search.retrievers.lexical.build_bm25_engine. Production callers
            # (agent_search.legacy.retriever) always pass `engine` explicitly (built with a
            # real index_root/key so a 'pyserini' backend persists); this fallback is for
            # direct/standalone construction (tests, ad-hoc scripts).
            from agent_search.retrievers.lexical import build_bm25_engine
            engine = build_bm25_engine(self.units)
        self.bm = engine
        self.seen = OrderedSeen()
        self.last_hits: list[str] = []
        self.query_biased = query_biased
        if query_biased:
            self.tools = ("bm25q_search", "visit_q")

    def search(self, query: str, k: int = BM25_VISIT_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = self.bm.search(query, k=k)
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        # query_biased=False (the default) shows the doc's OPENING window instead of a
        # query-biased one; both are SNIPPET_TOKENS wide (see `opening_line`).
        terms = code_tokenize(query) if self.query_biased else None
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            if self.query_biased:
                snip = best_line(u, terms)
            else:
                snip = opening_line(u)
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)

    def _resolve(self, ref):
        if isinstance(ref, (int, float)) or (isinstance(ref, str) and ref.strip().isdigit()):
            rank = int(ref)
            if 1 <= rank <= len(self.last_hits):
                return self.last_hits[rank - 1], None
            if str(ref).strip() in self.ubyid:            # a real doc_id that looks numeric
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
            # tools.yaml's toolset (and the rendered <tools> spec) names this `bm25_search` —
            # a real episode's model calls it by that name, so `run()` must accept it (the bare
            # `search` is also accepted for direct/standalone construction).
            # `bm25q_search`/`visit_q` are the bm25q arm's tool NAMES for this SAME class
            # (`query_biased=True` — see __init__/search above); accepted here too so a real
            # episode's model calling by that name works identically.
            if name in ("bm25_search", "bm25q_search", "search"):
                # listing depth is BM25_VISIT_TOPK (env knob) ONLY — the tool schema exposes
                # just `query`, so a hallucinated `k` arg must not resize the SERP listing.
                return self.search(args.get("query") or args.get("q") or "",
                                   BM25_VISIT_TOPK)
            if name in ("visit", "visit_q"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class Bm25AutoRead(Bm25Visit):
    """The "retrieve-and-read" baseline (research_bm25_autoread): SAME plain BM25 ranking as
    `Bm25Visit` (over the flat doc title+body — same engine, same env `BM25_BACKEND` fallback),
    but `search(query)` itself renders the FULL TEXT of every one of the top `AUTOREAD_TOPK`
    hits — there is NO visit/fetch tool at all in this condition; a search call IS the read.
    Each hit's text is capped at MAX_VISIT_TOKENS via the SAME `_cap_tokens` helper (same cap,
    same " …(truncated — this is the whole-doc cap)" marker) `Bm25Visit.visit` uses — so a
    single doc's rendered length is identical to what a `research_bm25` episode gets FROM
    visiting that same rank, only here every top-k hit is rendered, unconditionally, on every
    search call.

    Sits between the one-shot RAG baseline (scripts/oneshot_rag.py — a single stuffed prompt,
    no agent loop, no per-query retrieval choice at all) and the SERP retrieve-then-visit
    baseline (research_bm25 — a listing the agent must then CHOOSE to visit): isolates the value
    of the agent's choose-what-to-read step by removing it entirely, while keeping everything
    else (BM25 ranking, the agent loop, the <answer> contract) identical to research_bm25.
    Uncoached (no skill), like the bm25/dense baselines it's built from.

    tools.yaml's `bm25_autoread` toolset names this tool `bm25_read_search` (a distinct name
    from `bm25_search`, since its description says results already include full document text)
    — `run()` accepts both `bm25_read_search` and the bare `search`. There is deliberately NO
    `visit`/`visit_q`/`visit_d`/`fetch` dispatch: a real episode's model attempting one of those
    (a hallucinated call, or a habit carried over from a sibling condition's manual/memory) gets
    a clear ERROR string explaining why, not a silent "unknown tool" or a reused baseline read."""

    # ONE tool only — no read/visit tool exists in this condition (search already returns full
    # text). Overrides Bm25Visit's class-level `tools = ("bm25_search", "visit")`.
    tools = ("bm25_read_search",)

    def __init__(self, units: Sequence[CodeUnit], engine=None, ubyid: Optional[dict] = None):
        # SAME construction as Bm25Visit (query_biased is irrelevant here — there is no listing
        # snippet to bias, the search response IS the full text) — pass the default explicitly
        # for clarity, then restore this class's own `tools` (Bm25Visit.__init__ only swaps
        # `self.tools` when `query_biased=True`, so this is a no-op today, but explicit is safer
        # than relying on that not changing).
        super().__init__(units, engine=engine, ubyid=ubyid, query_biased=False)
        self.tools = ("bm25_read_search",)

    def search(self, query: str, k: int = AUTOREAD_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = self.bm.search(query, k=k)
        if not ids:
            prior = ("  (previous results still available)" if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            # SAME cap/marker as Bm25Visit.visit — a doc read via autoread's search is truncated
            # identically to the same doc read via research_bm25's visit.
            text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                               " …(truncated — this is the whole-doc cap)")
            blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
        return "\n\n".join(blocks)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("bm25_read_search", "search"):
                # listing depth is AUTOREAD_TOPK (env knob) ONLY — the tool schema exposes just
                # `query`, so a hallucinated `k` arg must not resize how many docs are read.
                return self.search(args.get("query") or args.get("q") or "", AUTOREAD_TOPK)
            if name in ("visit", "visit_q", "visit_d", "visit_v", "visit_h", "visit_bv",
                       "visit_bqld", "fetch"):
                return ("ERROR: no visit tool in this condition — search already returns full "
                        "documents.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class DenseVisit(Bm25Visit):
    """The DENSE retrieve-then-visit baseline (the modern RAG default): search(query, k) is
    dense embedding cosine similarity over the document corpus — otherwise the SAME as
    `Bm25Visit` (same rendering: rank, doc_id, title, opening snippet; same `visit`/`_resolve`,
    inherited unchanged). Uncoached (no skill), like the bm25 baseline.

    `engine` is a `DenseBelief` (agent_search.retrievers.indri.dense_belief) already
    `build_or_load`ed over this corpus — reused, not reimplemented: `top_k_doc_ids(query, k)`
    is DenseBelief's memoized query-encode + cosine-top-k over its cached embedding matrix (the
    SAME persisted `indexes/dense/<model>-sl<len>/<corpus_key>/` cache the `dense` retriever
    condition and Indri's dense belief already build/persist — see DEFAULT_MODEL there,
    BAAI/bge-base-en-v1.5). retriever.py's 'densevisit' arm builds + validates that cache at
    index() time and raises a CLEAR error if it's missing, rather than silently falling back to
    live-encoding the whole corpus (which this baseline does NOT support). Tests inject a stub
    `engine` exposing `top_k_doc_ids(query, k)` the same way test_doc_bm25_fetch_tools injects a
    stub BM25 engine for `Bm25Visit`."""

    # tools.yaml's toolset names this tool `dense_search` (distinguishing it from the bm25 arm's
    # `bm25_search`); `run()` accepts both `dense_search` and the bare `search`. `visit_d` is
    # tools.yaml's name for the (inherited, unmodified) whole-doc read; `run()` accepts `visit` too.
    tools = ("dense_search", "visit_d")

    def __init__(self, units: Sequence[CodeUnit], engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None):
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if engine is None:
            from agent_search.retrievers.indri.dense_belief import DenseBelief
            engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.engine = engine
        self.seen = OrderedSeen()
        self.last_hits: list[str] = []

    def search(self, query: str, k: int = DENSE_VISIT_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k) or [])
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = ids
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            snip = opening_line(u)
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_search", "search"):
                # listing depth is DENSE_VISIT_TOPK (env knob) ONLY — the tool schema exposes
                # just `query`, so a hallucinated `k` arg must not resize the SERP listing.
                return self.search(args.get("query") or args.get("q") or "",
                                   DENSE_VISIT_TOPK)
            if name in ("visit_d", "visit"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class DenseAutoRead(DenseVisit):
    """The DENSE "retrieve-and-read" baseline (research_dense_autoread): SAME dense-embedding
    ranking as `DenseVisit` (cosine similarity over the SAME persisted
    `indexes/dense/<model>-sl<len>/<corpus_key>/` doc-embedding cache, via
    `DenseBelief.top_k_doc_ids` — same engine, same fail-loud missing-cache contract), but
    `search(query)` itself renders the FULL TEXT of every one of the top `AUTOREAD_TOPK` hits —
    there is NO visit/fetch tool at all in this condition; a search call IS the read. Each hit's
    text is capped at MAX_VISIT_TOKENS via the SAME `_cap_tokens` helper (same cap, same
    " …(truncated — this is the whole-doc cap)" marker) `DenseVisit.visit`/`Bm25AutoRead.search`
    use — so a single doc's rendered length is identical to what a `research_dense` episode gets
    FROM visiting that same rank, only here every top-k hit is rendered, unconditionally, on
    every search call.

    The dense analog of `Bm25AutoRead` exactly as `DenseVisit` is the dense analog of
    `Bm25Visit`: reuses the SAME `AUTOREAD_TOPK` env knob (not a second one — the two autoread
    baselines show the same-sized pool) and the SAME MAX_VISIT_TOKENS cap. The ONLY difference
    from `research_bm25_autoread` is the retrieval engine (dense cosine similarity instead of
    BM25 keyword search) — auto-read behavior, top-k, the per-doc cap, and the <answer> contract
    are all identical. Uncoached (no skill), like the bm25/dense baselines it's built from.

    tools.yaml's `dense_autoread` toolset names this tool `dense_read_search` (a distinct name
    from `dense_search`, since its description says results already include full document text)
    — `run()` accepts both `dense_read_search` and the bare `search`. There is deliberately NO
    `visit`/`visit_d`/`visit_q`/`visit_v`/`fetch` dispatch: a real episode's model attempting one
    of those (a hallucinated call, or a habit carried over from a sibling condition's manual/
    memory) gets a clear ERROR string explaining why, not a silent "unknown tool" or a reused
    baseline read."""

    # ONE tool only — no read/visit tool exists in this condition (search already returns full
    # text). Overrides DenseVisit's class-level `tools = ("dense_search", "visit_d")`.
    tools = ("dense_read_search",)

    def __init__(self, units: Sequence[CodeUnit], engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None):
        # SAME construction as DenseVisit (engine/corpus_key resolve the SAME persisted
        # doc-embedding cache) — `tools` is already this class's own value via the class
        # attribute above; DenseVisit.__init__ never overrides `self.tools` (unlike Bm25Visit's
        # query_biased branch), so no restore-after-super() is needed here.
        super().__init__(units, engine=engine, ubyid=ubyid, corpus_key=corpus_key)

    def search(self, query: str, k: int = AUTOREAD_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k) or [])
        if not ids:
            prior = ("  (previous results still available)" if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = ids
        blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            # SAME cap/marker as DenseVisit.visit (inherited from Bm25Visit) — a doc read via
            # autoread's search is truncated identically to the same doc read via
            # research_dense's visit_d.
            text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                               " …(truncated — this is the whole-doc cap)")
            blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
        return "\n\n".join(blocks)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_read_search", "search"):
                # listing depth is AUTOREAD_TOPK (env knob) ONLY — the tool schema exposes just
                # `query`, so a hallucinated `k` arg must not resize how many docs are read.
                return self.search(args.get("query") or args.get("q") or "", AUTOREAD_TOPK)
            if name in ("visit", "visit_d", "visit_q", "visit_v", "visit_h", "visit_bv",
                       "visit_bqld", "fetch"):
                return ("ERROR: no visit tool in this condition — search already returns full "
                        "documents.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class HybridVisit(Bm25Visit):
    """The BM25+DENSE HYBRID retrieve-then-visit baseline (research_hybrid) — see the module
    docstring's HybridVisit/HybridFetchSnipWorkspace paragraph for the full RRF formula/pool
    sizes. `search(query, k)` queries BOTH rankers to `HYBRID_POOL` (100) depth, fuses with
    `rrf_fuse` (RRF_K=60), and renders the fused top-`k` EXACTLY like `Bm25Visit.search`
    (title + a short opening snippet, no query bias) — same listing format, only the ranking
    differs. `visit`/`_resolve` are INHERITED from `Bm25Visit`
    unchanged (whole-doc read, capped at MAX_VISIT_TOKENS). Uncoached (no skill), like the
    bm25/dense baselines it's built from.

    `bm25_engine` falls back to `build_bm25_engine` (env `BM25_BACKEND`-selectable, same
    fallback every bm25-consuming workspace uses — production callers, agent/retriever.py,
    always pass a real engine). `dense_engine` falls back to a fresh `DenseBelief.build_or_load`
    (same persisted `indexes/dense/<model>-sl<len>/<key>/` cache DenseVisit/DenseFetchWorkspace
    use; production callers pass the SAME prebuilt-and-validated belief those arms use)."""

    tools = ("hybrid_search", "visit_h")

    def __init__(self, units: Sequence[CodeUnit], bm25_engine=None, dense_engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None,
                 pool: int = HYBRID_POOL):
        # SAME manual-construction shape as DenseVisit.__init__ (not Bm25Visit.__init__'s
        # super() call — this workspace owns TWO engines, neither is "the" engine a bare
        # super().__init__ would wire to self.bm alone).
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if bm25_engine is None:
            from agent_search.retrievers.lexical import build_bm25_engine
            bm25_engine = build_bm25_engine(self.units)
        self.bm = bm25_engine
        if dense_engine is None:
            from agent_search.retrievers.indri.dense_belief import DenseBelief
            dense_engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.dense_engine = dense_engine
        self.pool = pool
        self.seen = OrderedSeen()
        self.last_hits: list[str] = []

    def search(self, query: str, k: int = HYBRID_VISIT_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        bm25_ids = list(self.bm.search(query, k=self.pool))
        dense_ids = list(self.dense_engine.top_k_doc_ids(query, k=self.pool) or [])
        ids = rrf_fuse(bm25_ids, dense_ids, topk=k)
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        # IDENTICAL rendering to Bm25Visit.search's query_biased=False branch (opening window,
        # not query-biased).
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            snip = opening_line(u)
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("hybrid_search", "search"):
                # listing depth is HYBRID_VISIT_TOPK (env knob) ONLY — the tool schema exposes
                # just `query`, so a hallucinated `k` arg must not resize the SERP listing.
                return self.search(args.get("query") or args.get("q") or "",
                                   HYBRID_VISIT_TOPK)
            if name in ("visit_h", "visit"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class HybridAutoRead(HybridVisit):
    """The AutoRead analog of research_hybrid: SAME RRF-fused bm25+dense retrieval as
    `HybridVisit` (SAME `HYBRID_POOL`/`RRF_K`/two engines — see `HybridVisit`/`rrf_fuse`), but
    `search(query)` itself renders the FULL TEXT of every one of the top `AUTOREAD_TOPK` hits —
    there is NO visit/fetch tool at all here; a search call IS the read. Each hit's text is
    capped at MAX_VISIT_TOKENS via the SAME `_cap_tokens` helper (same cap, same
    " …(truncated — this is the whole-doc cap)" marker) `Bm25AutoRead`/`DenseAutoRead`/
    `HybridVisit.visit` use — so a single doc's rendered length matches what a `research_hybrid`
    episode gets FROM visiting that same rank, only here every top-k hit is rendered,
    unconditionally, on every search call.

    Reuses the SAME `AUTOREAD_TOPK` env knob the other two autoread baselines use (not a third
    one — all three "retrieve-and-read" cells show the same-sized pool) and the SAME
    MAX_VISIT_TOKENS cap. The ONLY difference from `research_hybrid` is the render (full text vs
    a listing to visit) — exactly like `Bm25AutoRead`/`DenseAutoRead` over their own visit
    siblings. Uncoached (no skill), like `HybridVisit`."""

    # ONE tool only — no read/visit tool exists here (search already returns full text).
    # Overrides HybridVisit's class-level `tools = ("hybrid_search", "visit_h")`.
    tools = ("hybrid_read_search",)

    def __init__(self, units: Sequence[CodeUnit], bm25_engine=None, dense_engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None,
                 pool: int = HYBRID_POOL):
        # SAME construction as HybridVisit (two engines, SAME pool depth) — `tools` is already
        # this class's own value via the class attribute above; HybridVisit.__init__ never
        # overrides `self.tools`, so no restore-after-super() is needed (mirrors DenseAutoRead's
        # own comment on DenseVisit.__init__).
        super().__init__(units, bm25_engine=bm25_engine, dense_engine=dense_engine,
                         ubyid=ubyid, corpus_key=corpus_key, pool=pool)

    def search(self, query: str, k: int = AUTOREAD_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        bm25_ids = list(self.bm.search(query, k=self.pool))
        dense_ids = list(self.dense_engine.top_k_doc_ids(query, k=self.pool) or [])
        ids = rrf_fuse(bm25_ids, dense_ids, topk=k)
        if not ids:
            prior = ("  (previous results still available)" if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            # SAME cap/marker as HybridVisit.visit (inherited from Bm25Visit) — a doc read via
            # autoread's search is truncated identically to the same doc read via
            # research_hybrid's visit_h.
            text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                               " …(truncated — this is the whole-doc cap)")
            blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
        return "\n\n".join(blocks)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("hybrid_read_search", "search"):
                # listing depth is AUTOREAD_TOPK (env knob) ONLY — the tool schema exposes just
                # `query`, so a hallucinated `k` arg must not resize how many docs are read.
                return self.search(args.get("query") or args.get("q") or "", AUTOREAD_TOPK)
            if name in ("visit", "visit_q", "visit_d", "visit_v", "visit_h", "visit_bv",
                       "visit_bqld", "fetch"):
                return ("ERROR: no visit tool in this condition — search already returns full "
                        "documents.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# --- arms (the toolset of a condition selects the workspace; engines are built once per corpus)
from agent_search.legacy.retriever import register_workspace  # noqa: E402

register_workspace("bm25", tools=("bm25_search", "visit"), engines=("bm25",),
                   builder=lambda ctx: Bm25Visit(ctx.units, engine=ctx.bm25(), ubyid=ctx.ubyid))
register_workspace("bm25q", tools=("bm25q_search", "visit_q"), engines=("bm25",),
                   builder=lambda ctx: Bm25Visit(ctx.units, engine=ctx.bm25(), ubyid=ctx.ubyid, query_biased=True))
register_workspace("bm25autoread", tools=("bm25_read_search",), engines=("bm25",),
                   builder=lambda ctx: Bm25AutoRead(ctx.units, engine=ctx.bm25(), ubyid=ctx.ubyid))
register_workspace("densevisit", tools=("dense_search", "visit_d"), engines=("dense",),
                   builder=lambda ctx: DenseVisit(ctx.units, engine=ctx.dense(), ubyid=ctx.ubyid, corpus_key=ctx.corpus_key))
register_workspace("denseautoread", tools=("dense_read_search",), engines=("dense",),
                   builder=lambda ctx: DenseAutoRead(ctx.units, engine=ctx.dense(), ubyid=ctx.ubyid, corpus_key=ctx.corpus_key))
register_workspace("hybridvisit", tools=("hybrid_search", "visit_h"), engines=("bm25", "dense"),
                   builder=lambda ctx: HybridVisit(ctx.units, bm25_engine=ctx.bm25(), dense_engine=ctx.dense(),
                                                   ubyid=ctx.ubyid, corpus_key=ctx.corpus_key))
register_workspace("hybridautoread", tools=("hybrid_read_search",), engines=("bm25", "dense"),
                   builder=lambda ctx: HybridAutoRead(ctx.units, bm25_engine=ctx.bm25(), dense_engine=ctx.dense(),
                                                      ubyid=ctx.ubyid, corpus_key=ctx.corpus_key))
