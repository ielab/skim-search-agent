"""The pre-0.3 agent, exposed as a Retriever: one class for every agent condition.

Kept so the parity tests can compare against it. `agent_search.evaluation.agent_runner.
ConditionAgent` is the current runner; each arm below has a current strategy in
`agent_search/strategies/` (see `docs/ARCHITECTURE.md` for the current names).

`index(units)` then `search(issue)` drives one episode of the search-then-fetch loop and
returns the ranking (empty for the code-fix arm, which is scored on its <fix>; the doc arm
surfaces evidence). The condition is entirely the toolset plus domain (from the prompt
profile), which selects the arm's workspace:

  code (toolset search_fetch, domain code)     -> CodeFixWorkspace (files -> functions, <fix>)
  code (toolset grep_read,    domain code)     -> GrepReadWorkspace (regex lines -> read, <fix>)
  docs (toolset research,     domain general)  -> DocSearchFetch   (articles -> sections, <answer>)
  docs (toolset research_bm25, domain general) -> Bm25Visit        (retrieve-then-visit baseline)
  docs (toolset dense_visit,  domain general)  -> DenseVisit       (retrieve-then-visit, dense
                                                    embeddings instead of bm25)
  docs (toolset dci,          domain general)  -> DciWorkspace     (bash/read, whole corpus, <answer>)
  docs (toolset bm25_dci,     domain general)  -> Bm25DciWorkspace (bash/read bounded to bm25 top-k)
  docs (toolset bm25_fetch,   domain general)  -> Bm25FetchWorkspace (bm25 retrieval + section fetch)
  docs (toolset search_fetch_v2, domain general) -> DocSearchFetch(coverage=True) (BQL v2: typed
                                                    date[RANGE] ranges + constraint-coverage feedback)
  docs (toolset indri,        domain general)  -> IndriFetchWorkspace (Indri graded-QL backend)
  docs (toolset indri_visit,  domain general)  -> IndriVisitWorkspace (graded Indri search,
                                                    content-bearing listing, whole-doc visit read)
  docs (toolset indri_snip,   domain general)  -> IndriFetchWorkspace(snippets=True) (graded
                                                    Indri search + snippet listing, section fetch)
  docs (toolset bm25q_visit,  domain general)  -> Bm25Visit(query_biased=True) (a hardened
                                                    bm25 baseline: the same retrieve-then-visit
                                                    shape as research_bm25, but the listing's
                                                    per-hit snippet is query-biased, for fairness
                                                    parity with research_snip/research_indri_snip's
                                                    excerpt)
  docs (toolset bql_visit,   domain general)   -> BqlVisitWorkspace (the {BQL search} x
                                                    {whole-doc visit} factorial cell: the same
                                                    BQL v2 search as the search_fetch_v2 arm
                                                    above, content-bearing listing like the other
                                                    visit cells, whole-doc visit read)
  docs (toolset bm25_fetch_snip, domain general) -> Bm25FetchSnipWorkspace (research_bm25_fetch_snip:
                                                    the fair-listing sibling of the plain
                                                    bm25_fetch arm above, same bm25 retrieval and
                                                    section fetch, with a per-hit best-matching
                                                    excerpt in the listing)
  docs (toolset bql_dense_visit, domain general) -> BqlVisitWorkspace(tool_names=...) (the
                                                    bql_visit arm above with the shared BQL
                                                    executor's DenseBelief attached: BQL_DENSE
                                                    dense-fused ranking, RRF of bm25 and dense
                                                    similarity restricted to the same
                                                    filter-passing candidates; see
                                                    agent_search/retrievers/bql/dense_fuse.py)
  docs (toolset bql_dense_snip,  domain general) -> DocSearchFetch(snippets=True) (research_
                                                    bql_dense_snip: research_snip's search/fetch
                                                    shape with the same dense-attached BQL executor)
  docs (toolset bm25_autoread, domain general)   -> Bm25AutoRead (research_bm25_autoread: the
                                                    retrieve-and-read baseline, same bm25 ranking
                                                    as research_bm25, but search() itself returns
                                                    the full text of every top-k hit; no
                                                    visit/fetch tool exists in this condition)
  docs (toolset dense_autoread, domain general)  -> DenseAutoRead (research_dense_autoread: the
                                                    dense analog of research_bm25_autoread, same
                                                    retrieve-and-read shape, same AUTOREAD_TOPK
                                                    and MAX_VISIT_TOKENS, but dense embedding
                                                    cosine similarity instead of bm25 ranking; no
                                                    visit/fetch tool exists in this condition)
  docs (toolset hybrid_autoread, domain general) -> HybridAutoRead (the bm25+dense hybrid analog
                                                    of bm25_autoread/dense_autoread: same
                                                    RRF-fused retrieval as research_hybrid, but
                                                    search() itself returns the full text of
                                                    every top-k hit; no visit/fetch tool exists
                                                    in this condition)
  docs (toolset bql_donly_visit, domain general) -> BqlDonlyVisitWorkspace (the dense-only
                                                    sibling of the bql_dense_visit arm above:
                                                    same BQL filter/listing/whole-doc visit read,
                                                    but the shared BQL executor is a
                                                    DenseOnlyStructuralExecutor, candidates
                                                    ordered by pure dense rank, not RRF)
  docs (toolset bql_donly_snip,  domain general) -> DocSearchFetchDonlySnip (research_bql_donly_
                                                    snip: the dense-only sibling of the
                                                    bql_dense_snip arm above, same BQL
                                                    filter/snippet listing/section-fetch read,
                                                    same coverage-tier structure, but within-tier
                                                    ordering is pure dense rank, not RRF)

So which arm and which tools apply is configuration, not code. The shared BQL executor is
reused verbatim by both search-then-fetch arms; only the surface, tools and task differ.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from agent_search.agent.loop import Task, run_episode
from agent_search.core.interfaces import Retriever
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.lexical import build_bm25_engine as _build_bm25_engine


# TOOL-ENGINE registry — the mid-episode-tool extension point (kept for parity with the
# retriever registry). A NEW agent tool registers a builder `units -> engine` here and
# joins a toolset in tools.yaml + a condition in conditions.yaml, with no edit to this
# file. The built-in arms (search/fetch/visit) construct their own workspaces below.
TOOL_ENGINES: dict = {}


def register_tool_engine(name: str, builder=None):
    """Register (or decorate) a builder `units -> engine` for an agent tool `name`."""
    def deco(fn):
        TOOL_ENGINES[name] = fn
        return fn
    return deco(builder) if builder is not None else deco


# --- workspace registry: the extension point for a NEW tool surface --------------------
# A condition's toolset selects a *workspace* (the object whose `run(name, args)` answers the
# agent's tool calls). Built-in workspaces are matched by the marker tools in `_arm` below;
# a plugin registers its own with `register_workspace` and never touches this module:
#
#     from agent_search.legacy.retriever import register_workspace
#
#     @register_workspace("my_arm", tools=("my_search", "my_read"))
#     def build_my_workspace(ctx):           # ctx: WorkspaceContext
#         return MyWorkspace(ctx.units, engine=ctx.bm25(), ubyid=ctx.ubyid)
#
# A condition whose toolset contains ALL of `tools` then drives MyWorkspace. The most
# specific registration (most marker tools) wins when several match.

_ENGINE_LOCKS: dict = {}
_ENGINE_LOCKS_GUARD = threading.Lock()


def _engine_lock(retriever) -> threading.Lock:
    """One lock per retriever for the lazily built shared engines. Concurrent episodes
    (`--workers N`) must not each load the same index; the first builds, the rest wait."""
    lock = getattr(retriever, "_engine_build_lock", None)
    if lock is None:
        with _ENGINE_LOCKS_GUARD:
            lock = getattr(retriever, "_engine_build_lock", None)
            if lock is None:
                lock = threading.Lock()
                try:
                    retriever._engine_build_lock = lock
                except Exception:  # noqa: BLE001 — a retriever without settable attributes
                    lock = _ENGINE_LOCKS.setdefault(id(retriever), threading.Lock())
    return lock


@dataclass
class WorkspaceContext:
    """What a workspace builder gets: the indexed corpus plus lazily-built shared engines."""
    units: list
    ubyid: dict
    corpus_key: Optional[str]
    index_root: str
    rebuild: bool
    query: str
    domain: str
    retriever: "AgentRetriever"

    def bm25(self):
        """The local/Lucene BM25 engine over this corpus (built once, cached on the retriever)."""
        r = self.retriever
        with _engine_lock(r):
            if getattr(r, "_bm25", None) is None:
                r._bm25 = _build_bm25_engine(self.units, self.index_root, self.rebuild, self.corpus_key)
        return r._bm25

    def bql(self):
        """The BQL structured executor over this corpus (persisted under index_root when keyed)."""
        r = self.retriever
        with _engine_lock(r):
            if getattr(r, "_bql", None) is None:
                from agent_search.retrievers.backend import build_bql_engine
                r._bql = build_bql_engine(self.units, self.index_root, self.corpus_key, self.rebuild)
        return r._bql

    def dense(self):
        """The dense belief engine (requires a pre-built embedding cache for `corpus_key`)."""
        r = self.retriever
        with _engine_lock(r):
            return self._dense_locked(r)

    @property
    def files(self) -> dict:
        """The raw repository files (code arms), set by the harness through `set_files`."""
        return getattr(self.retriever, "_files", {}) or {}

    def bql_plain(self):
        """The BQL executor with no dense model attached (the code arm)."""
        r = self.retriever
        with _engine_lock(r):
            if getattr(r, "_bql", None) is None:
                from agent_search.retrievers.backend import build_bql_engine
                r._bql = build_bql_engine(self.units, self.index_root, self.corpus_key, self.rebuild, dense=None)
        return r._bql

    def bql_dense(self):
        """The BQL executor with the run's dense model attached for fused ranking (Sieve)."""
        r = self.retriever
        with _engine_lock(r):
            if getattr(r, "_bql", None) is None:
                from agent_search.retrievers.backend import build_bql_engine
                r._bql = build_bql_engine(self.units, self.index_root, self.corpus_key, self.rebuild,
                                          dense=self._dense_locked(r))
        return r._bql

    def bql_dense_only(self):
        """The BQL executor whose ranking is the dense model alone (`sieve_dense`)."""
        r = self.retriever
        with _engine_lock(r):
            if getattr(r, "_bql", None) is None:
                from agent_search.retrievers.backend import build_bql_engine_dense_only
                r._bql = build_bql_engine_dense_only(self.units, self.index_root, self.corpus_key, self.rebuild,
                                                     dense=self._dense_locked(r))
        return r._bql

    def indri(self):
        """The Indri executor; `INDRI_DENSE=1` attaches the dense model, degrading with a warning
        when its cache is missing."""
        r = self.retriever
        with _engine_lock(r):
            if getattr(r, "_indri", None) is None:
                import os
                import sys
                from agent_search.retrievers.backend import build_indri_engine
                dense = None
                if os.environ.get("INDRI_DENSE") in ("1", "true", "yes"):
                    try:
                        dense = self._dense_locked(r)
                    except Exception as e:  # noqa: BLE001
                        print(f"WARNING: INDRI_DENSE=1 but the dense engine could not be built ({e!r}); "
                              f"degrading to lexical-only Indri retrieval.", file=sys.stderr)
                r._indri = build_indri_engine(self.units, self.index_root, self.corpus_key, self.rebuild, dense=dense)
        return r._indri

    def _dense_locked(self, r):
        if getattr(r, "_dense_belief", None) is None:
            from agent_search.core.errors import SetupError
            from agent_search.retrievers.dense import DenseRetriever
            from agent_search.retrievers.dense import DenseBelief
            probe = DenseRetriever(model=r.dense_model, index_root=self.index_root)
            if not self.rebuild and not probe.is_cached(self.corpus_key):
                raise SetupError(
                    f"no persisted dense embedding cache for corpus key {self.corpus_key!r} and "
                    f"model {r.dense_model!r} at {probe._cache_dir(self.corpus_key)!r}; build it "
                    f"with: skimsearchagent-build-indexes --retriever dense --model {r.dense_model} "
                    f"... (documents are never encoded online during a run)")
            r._dense_belief = DenseBelief(model=r.dense_model, index_root=self.index_root
                                          ).build_or_load(self.units, key=self.corpus_key)
        return r._dense_belief


@dataclass
class WorkspaceSpec:
    """One registered arm: the marker tools that select it, the builder, the engines to prebuild
    once per corpus, and what the harness must know about it."""
    arm: str
    tools: frozenset
    builder: Callable[[WorkspaceContext], object]
    engines: tuple = ()               # names of WorkspaceContext methods to call at index() time
    domain: Optional[str] = None      # "code" | "general" | None (any)
    needs_files: bool = False         # the workspace reads the raw repository files
    sdk: bool = True                  # may run through the Agents-SDK driver


WORKSPACES: dict[str, WorkspaceSpec] = {}


def register_workspace(arm: str, *, tools: Sequence[str], builder=None, engines: Sequence[str] = (),
                       domain: Optional[str] = None, needs_files: bool = False, sdk: bool = True):
    """Register (or decorate) `builder(ctx: WorkspaceContext) -> workspace` for the arm selected
    by a toolset containing every name in `tools`. `engines` names the shared engines
    (`bm25`, `bql`, `bql_dense`, `bql_dense_only`, `bql_plain`, `dense`, `indri`) built once
    per corpus before the first episode; `domain` restricts the arm to code or general corpora."""
    marker = frozenset(tools)
    if not marker:
        raise ValueError("register_workspace needs at least one marker tool")

    def deco(fn):
        WORKSPACES[arm] = WorkspaceSpec(arm=arm, tools=marker, builder=fn, engines=tuple(engines),
                                        domain=domain, needs_files=needs_files, sdk=sdk)
        return fn
    return deco(builder) if builder is not None else deco


def _registered_arm(toolset: Sequence[str], domain: Optional[str] = None) -> Optional[str]:
    """The registered arm whose marker tools are all in `toolset` (and whose domain fits), the
    most specific registration (most markers) winning; None when no registration matches."""
    ts = set(toolset)
    best, best_n = None, 0
    for arm, spec in WORKSPACES.items():
        if spec.domain is not None and domain is not None and spec.domain != domain:
            continue
        if spec.tools <= ts and len(spec.tools) > best_n:
            best, best_n = arm, len(spec.tools)
    return best


class AgentRetriever(Retriever):
    name = "agent"
    returns_full_set = True       # capability flag: the agent emits its own ranking
                                  # (located/surfaced), so run_eval must not pad to SET_K

    def __init__(self, policy_factory: Callable[[], object], toolset: Sequence[str],
                 max_steps: int = 50, prompt_path: Optional[str] = None,
                 dense_model: str = "nomic-ai/CodeRankEmbed",
                 index_root: str = "indexes", rebuild: bool = False, domain: str = "code",
                 tool: Optional[str] = None, model: Optional[str] = None,
                 api_base: Optional[str] = None, field_profile: Optional[str] = None,
                 driver: str = "loop"):
        self._policy_factory = policy_factory
        self._model = model                       # used by the SDK driver
        self._api_base = api_base
        self._field_profile = field_profile       # selects the same skill manual the loop loads
        self._driver = driver                     # "sdk" (native tool-calling) or "loop"; env overrides
        self.toolset = tuple(toolset)
        self.max_steps = max_steps
        self.prompt_path = prompt_path
        self.domain = domain
        self.tool = tool                          # the condition label (for rows.jsonl)
        self.tool_description = "toolset: " + ", ".join(self.toolset)
        self.dense_model = dense_model
        self.index_root = index_root
        self.rebuild = rebuild
        self._units: list = []
        self._key: Optional[str] = None
        self._files: dict = {}
        self._bql = None                          # prewarmed StructuralExecutor (search arm)
        self._bm25 = None                         # BM25Local or BM25Pyserini (env BM25_BACKEND,
                                                  # see _build_bm25_engine) — the doc retrieve-
                                                  # then-visit baseline, and the bm25->dci arm's
                                                  # identical retrieval stage
        self._indri = None                        # prewarmed IndriExecutor (the Indri-family arms)
        self._dense_belief = None                 # prewarmed DenseBelief (research_dense arm)
        self._ubyid: dict = {}                    # doc_id -> unit, built ONCE per corpus (reused
                                                  # by every per-query workspace — see search())
        self._tl = threading.local()

    @property
    def _arm(self) -> str:
        """Which workspace this condition drives, chosen by TOOLSET (never hardcoded per
        condition name):
          'code'    : search->fetch, files->functions, <fix>              (the method)
          'grep'    : grep->read,   regex lines->file read, <fix>         (code baseline)
          'doc'     : search->fetch, articles->sections, <answer>          (the method)
          'bm25'    : bm25_search->visit, whole-doc, <answer>              (retrieve-then-visit)
          'dci'     : bash->read,   raw filesystem grep/read, <answer>     (brute-force baseline)
          'bm25dci' : bm25_search->bash->read, bounded to bm25 top-k, <answer>
                      (retrieval-bounded brute-force — SAME retrieval as 'bm25', SAME shell as
                      'dci'; checked BEFORE 'bm25'/'dci' since its toolset carries both markers)
          'bm25fetch': bm25_search->fetch a section, bounded to bm25 top-k, <answer>
                      (retrieval-bounded structured read — SAME retrieval as 'bm25', SAME
                      section-fetch as 'doc'; checked BEFORE 'bm25' since its toolset also
                      carries bm25_search — the `fetch` marker distinguishes it)
          'docv2'   : search_v2->fetch_v2, articles->sections, <answer>            (BQL v2 —
                      typed date[RANGE] ranges + constraint-coverage feedback)
          'indri'   : isearch->fetch,     articles->sections, <answer>            (the Indri
                      graded query-language backend)
          'docsnip' : search_s->fetch_s,  articles->sections, <answer>            (research_snip
                      — SAME BQL search/fetch as 'doc', but each search hit gets an appended
                      one-line best-matching excerpt)
          'indrivisit': isearch_v->visit_v, content-bearing listing->whole doc, <answer>
                      (graded Indri search LIKE 'indri', WITH a per-hit
                      snippet like 'bm25', but read is a whole-doc VISIT like 'bm25' instead of
                      a section fetch; checked BEFORE 'indri' since its toolset
                      also carries an isearch-family marker)
          'indrisnip' : isearch_s->fetch,  articles->sections, <answer>            (research_indri_snip
                      — graded Indri search LIKE 'indri' WITH a per-hit snippet like 'docsnip',
                      SAME section-fetch read as 'indri'; checked BEFORE 'indri')
          'densevisit': dense_search->visit_d, whole-doc, <answer>          (research_dense —
                      the modern RAG default: SAME retrieve-then-visit shape as 'bm25', dense
                      embedding cosine similarity instead of BM25)
          'densefetch': dense_search_f->fetch a section, <answer>          (research_dense_fetch
                      — the {dense search} x {structure->parts read} cell: SAME dense retrieval
                      engine as 'densevisit', SAME structured section-fetch read as 'bm25fetch'/
                      'doc'; checked BEFORE 'densevisit' since its toolset also
                      carries a dense_search-family marker)
          'densefetchplain': dense_search_fp->fetch a section, <answer>   (research_dense_fetch_
                      plain — the missing PLAIN (no-excerpt) dense fetch cell: SAME dense
                      retrieval engine + SAME structured section-fetch read as 'densefetch', minus
                      the per-hit excerpt in the listing — research_dense_fetch's SAME search
                      minus the excerpt, mirroring 'bqldensefetch' over 'bqldensesnip';
                      checked BEFORE 'densefetch' — its own unique toolset marker
                      (dense_search_fp) resolves first, no ordering conflict)
          'bm25q'   : bm25q_search->visit_q, whole-doc, <answer>            (the
                      HARDENED bm25 baseline: SAME bm25 retrieval + whole-doc visit read as
                      'bm25', but the listing's per-hit snippet is QUERY-BIASED (Bm25Visit
                      query_biased=True) instead of the doc opening — a fairness fix, not a new
                      read/retrieval strategy)
          'bqlvisit': search_bv->visit_bv, content-bearing listing->whole doc, <answer>
                      (the {BQL search} x {whole-doc visit} factorial cell:
                      SAME BQL v2 search as 'docv2' (coverage_topk + date nudge), WITH a per-hit
                      snippet like 'indrivisit'/'bm25', but no fetch/structure tools — visit is
                      the only read; its own unique toolset marker (search_bv),
                      no ordering conflict with any other arm)
          'bm25fetchsnip': bm25_search_snip->fetch a section, <answer>          (research_bm25_fetch_snip
                      — the FAIR-LISTING sibling of 'bm25fetch': SAME live bm25 retrieval + SAME
                      structured section-fetch read, but the listing carries a per-hit
                      best-matching excerpt (like 'densefetch' over 'bm25fetch');
                      checked BEFORE 'bm25fetch' since its toolset also carries a bm25_search-
                      family marker — its own unique marker (bm25_search_snip) resolves it first)
          'hybridvisit': hybrid_search->visit_h, whole-doc, <answer>          (research_hybrid —
                      the BM25+DENSE HYBRID baseline: Reciprocal Rank Fusion (RRF, k=60) over a
                      top-100 pyserini/Lucene BM25 pool and a top-100 FAISS/dense pool (SAME
                      embedder/cache 'densevisit' uses), rendered EXACTLY like 'bm25'/Bm25Visit's
                      retrieve-then-visit listing; its own unique toolset marker
                      (hybrid_search), no ordering conflict with any other arm)
          'hybridfetchsnip': hybrid_search_snip->fetch a section, <answer>     (research_hybrid_
                      fetch_snip — the fetch-mode twin of 'hybridvisit': SAME RRF-fused bm25+dense
                      retrieval, SAME structured section-fetch read as 'bm25fetch'/'densefetch',
                      WITH a per-hit best-matching excerpt in the listing; checked
                      BEFORE 'hybridvisit' since its toolset also carries a hybrid_search-family
                      marker — its own unique marker (hybrid_search_snip) resolves it first)
          'bqldensevisit': search_bqld->visit_bqld, content-bearing listing->whole doc, <answer>
                      (BQL_DENSE dense-fused ranking, see
                      agent_search/retrievers/bql/dense_fuse.py: BYTE-FOR-BYTE
                      'bqlvisit' EXCEPT the shared BQL executor has a DenseBelief attached, so
                      run_with_count/coverage_topk RRF-fuse bm25 with dense similarity
                      restricted to the SAME filter-passing candidates; its own
                      unique toolset marker (search_bqld), no ordering conflict with any other arm)
          'bqldensesnip': search_bqlds->fetch_bqlds, articles->sections, <answer>
                      (research_bql_dense_snip — BQL_DENSE dense-fused ranking: BYTE-FOR-BYTE
                      'docsnip' EXCEPT the shared BQL executor has a DenseBelief attached, SAME
                      mechanism as 'bqldensevisit'; checked BEFORE 'docsnip' since
                      it needs its own unique toolset marker, no ordering conflict)
          'bqldensefetch': search_bqldf->fetch_bqldf, articles->sections, <answer>
                      (research_bql_dense_fetch — BQL_DENSE dense-fused ranking: BYTE-FOR-BYTE
                      'doc' (the PLAIN search->fetch method, no per-hit excerpt) EXCEPT the
                      shared BQL executor has a DenseBelief attached, SAME mechanism as
                      'bqldensesnip'/'bqldensevisit' — this is research_bql_dense_snip's SAME
                      search minus the excerpt; its own unique toolset marker
                      (search_bqldf), no ordering conflict with any other arm)
          'bm25autoread': bm25_read_search (ONLY tool), <answer>                 (research_
                      bm25_autoread — the "retrieve-and-read" baseline: SAME bm25 ranking as
                      'bm25', but search() itself returns the FULL TEXT of every top-k hit; NO
                      visit/fetch tool exists in this toolset at all; its own
                      unique toolset marker (bm25_read_search), no ordering conflict with any
                      other arm — checked before 'bm25' purely for readability, not necessity)
          'denseautoread': dense_read_search (ONLY tool), <answer>               (research_
                      dense_autoread — the DENSE analog of 'bm25autoread': SAME dense-embedding
                      ranking as 'densevisit', but search() itself returns the FULL TEXT of
                      every top-k hit; NO visit/fetch tool exists in this toolset at all;
                      its own unique toolset marker (dense_read_search), no ordering
                      conflict with any other arm — checked before 'densevisit' purely for
                      readability, not necessity)
          'hybridautoread': hybrid_read_search (ONLY tool), <answer>            (research_
                      hybrid_autoread — the BM25+DENSE HYBRID analog of 'bm25autoread'/
                      'denseautoread': SAME RRF-fused retrieval as 'hybridvisit', but search()
                      itself returns the FULL TEXT of every top-k hit; NO visit/fetch tool
                      exists in this toolset at all; its own unique toolset
                      marker (hybrid_read_search), no ordering conflict with any other arm)
          'bqldonlyvisit': search_bqldo->visit_bqldo, content-bearing listing->whole doc,
                      <answer>          (the DENSE-ONLY sibling of
                      'bqldensevisit': BYTE-FOR-BYTE 'bqlvisit' EXCEPT the shared BQL executor is
                      a DenseOnlyStructuralExecutor (agent_search/retrievers/bql/
                      dense_fuse.py) — filter-passing candidates ordered by pure dense rank, not
                      RRF; its own unique toolset marker (search_bqldo), no
                      ordering conflict with any other arm)
          'bqldonlysnip': search_bqldos->fetch_bqldos, articles->sections, <answer>
                      (research_bql_donly_snip — the DENSE-ONLY sibling of 'bqldensesnip':
                      BYTE-FOR-BYTE 'docsnip' EXCEPT the shared BQL executor is a
                      DenseOnlyStructuralExecutor, SAME mechanism as 'bqldonlyvisit' — coverage-
                      tier structure unchanged, within-tier order is pure dense rank;
                      checked BEFORE 'docsnip' since it needs its own unique toolset
                      marker, no ordering conflict)
        """
        registered = _registered_arm(self.toolset, self.domain)   # the most specific registration wins
        if registered is not None:
            return registered
        ts = set(self.toolset)
        if "bm25_read_search" in ts:
            return "bm25autoread"
        if "dense_read_search" in ts:
            return "denseautoread"
        if "hybrid_read_search" in ts:
            return "hybridautoread"
        if "search_bqldo" in ts:
            return "bqldonlyvisit"
        if "search_bqldos" in ts and "fetch_bqldos" in ts:
            return "bqldonlysnip"
        if "hybrid_search_snip" in ts:
            return "hybridfetchsnip"
        if "hybrid_search" in ts:
            return "hybridvisit"
        if "search_bqld" in ts:
            return "bqldensevisit"
        if "search_bqlds" in ts and "fetch_bqlds" in ts:
            return "bqldensesnip"
        if "search_bqldf" in ts and "fetch_bqldf" in ts:
            return "bqldensefetch"
        if "isearch_v" in ts:
            return "indrivisit"
        if "isearch_s" in ts:
            return "indrisnip"
        if "isearch" in ts:
            return "indri"
        if "search_s" in ts and "fetch_s" in ts:
            return "docsnip"
        if "search_bv" in ts:
            return "bqlvisit"
        if "search_v2" in ts and "fetch_v2" in ts:
            return "docv2"
        if "dense_search_fp" in ts:
            return "densefetchplain"
        if "dense_search_f" in ts:
            return "densefetch"
        if "dense_search" in ts:
            return "densevisit"
        if "bm25q_search" in ts:
            return "bm25q"
        if "bm25_search_snip" in ts:
            return "bm25fetchsnip"
        if "bash" in ts and "bm25_search" in ts:
            return "bm25dci"
        if "bm25_search" in ts and "fetch" in ts:
            return "bm25fetch"
        if "visit" in ts or "bm25_search" in ts:
            return "bm25"
        if "bash" in ts:
            return "dci"
        if "grep" in ts and "read" in ts:
            return "grep"
        return "code" if self.domain == "code" else "doc"

    # the code arms (search->fetch AND grep->read) read raw file sources — the code grep
    # arm's `read` slices them directly, and the search->fetch arm's `fetch` needs them for an
    # L-range. The doc arms (doc/bm25/dci/bm25dci/bm25fetch) build their own corpus view, none.
    @property
    def needs_files(self) -> bool:
        spec = WORKSPACES.get(self._arm)
        return spec.needs_files if spec is not None else self._arm in ("code", "grep")

    @property
    def last_trajectory(self):
        return getattr(self._tl, "traj", None)

    @property
    def last_trajectory_meta(self):
        return getattr(self._tl, "meta", None)

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "AgentRetriever":
        if getattr(units, "lazy", False):
            # an on-disk document store (agent_search.corpus.docstore): units are looked up by
            # id on demand and retrieval goes through prebuilt indexes; never materialise it
            self._units = units
            self._ubyid = units.by_id
        else:
            self._units = list(units)
            self._ubyid = {u.doc_id: u for u in self._units}   # once per corpus (not per query)
        self._key = key
        arm = self._arm
        if arm in ("code", "doc", "docv2", "docsnip", "bqlvisit",
                  "bqldensevisit", "bqldensesnip", "bqldensefetch"):
            # The field-tagged `search`/`search_v2`/`search_s`/`search_bv`/`search_bqld`/
            # `search_bqlds` tool lowers to this executor. Load the prewarmed index built
            # offline by build_indexes.py if one exists for this corpus; else build in memory
            # (lazy) — keeps the O(N) postings+BM25 build OFF the agent's clock for big corpora.
            # 'docv2'/'docsnip'/'bqlvisit'/'bqldensevisit'/'bqldensesnip' reuse the SAME
            # executor/index artifact as 'doc' (BQL v2's date-range + coverage_topk, and
            # research_snip's excerpt scoring,
            # are executor-adjacent METHODS/workspace logic, not a separate index).
            # Env `STRUCTURED_BACKEND` (default 'python', unchanged) selects the engine —
            # see agent_search.retrievers.backend.build_bql_engine; 'lucene' opens
            # the prebuilt indexes/lucene_structured/<key>/ index instead of the .pkl.
            #
            # BQL_DENSE dense-fused ranking (agent_search/retrievers/bql/dense_fuse.py):
            # 'bqldensevisit'/'bqldensesnip' attach a DenseBelief to this SAME executor
            # UNCONDITIONALLY (these two dedicated arms don't need the env var at all, exactly
            # like 'densevisit'/'densefetch' never need a toggle to use dense retrieval) — SAME
            # fail-loud missing-cache contract as 'densevisit'/'densefetch'/'hybridvisit' below,
            # since this ranking knob never live-encodes a whole corpus at eval time. The other
            # arms sharing this SAME executor ('code'/'doc'/'docv2'/'docsnip'/'bqlvisit') instead
            # RETROFIT dense fusion via the ambient env var `BQL_DENSE=1` (mirrors `INDRI_DENSE`'s
            # retrofit-via-env pattern exactly, incl. graceful degradation on a missing cache — a
            # WARNING, not a raise, since flipping this env var must never turn a baseline arm's
            # run into a hard failure): `BQL_DENSE` unset/0 (the default) means `dense_belief`
            # stays None for every arm here except the two dedicated ones, so the plain BQL arms
            # (docv2, docsnip, bqlvisit) keep bm25-only ranking. Built + forwarded to
            # `build_bql_engine`'s `dense=` kwarg (not a post-hoc attach) so it also reaches the
            # STRUCTURED_BACKEND=lucene path's LuceneBqlAdapter uniformly.
            from agent_search.retrievers.dense import DenseRetriever
            from agent_search.retrievers.dense import DenseBelief
            dense_belief = None
            if arm in ("bqldensevisit", "bqldensesnip", "bqldensefetch"):
                probe = DenseRetriever(model=self.dense_model, index_root=self.index_root)
                cond_label = {"bqldensevisit": "research_bql_dense_visit",
                             "bqldensesnip": "research_bql_dense_snip",
                             "bqldensefetch": "research_bql_dense_fetch"}[arm]
                if not self.rebuild and not probe.is_cached(key):
                    raise RuntimeError(
                        f"{cond_label} ({arm} arm) needs a persisted dense doc-embedding cache "
                        f"for corpus key {key!r} at {probe._cache_dir(key)!r} — none found. "
                        f"Prebuild it with: python -m agent_search.evaluation.build_indexes --retriever dense "
                        f"--model {self.dense_model} ... (this ranking knob does not "
                        f"live-encode the corpus at eval time).")
                try:
                    dense_belief = DenseBelief(
                        model=self.dense_model, index_root=self.index_root).build_or_load(
                            self._units, key=key)
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(
                        f"{cond_label} ({arm} arm): failed to load the dense embedding cache or "
                        f"model for corpus key {key!r}: {type(e).__name__}: {e}") from e
            elif arm != "code":
                from agent_search.retrievers.bql.dense_fuse import bql_dense_enabled
                if bql_dense_enabled():
                    probe = DenseRetriever(model=self.dense_model, index_root=self.index_root)
                    if self.rebuild or probe.is_cached(key):
                        try:
                            dense_belief = DenseBelief(
                                model=self.dense_model,
                                index_root=self.index_root).build_or_load(self._units, key=key)
                        except Exception as e:
                            import sys
                            print(f"WARNING: BQL_DENSE=1 but DenseBelief build/load failed "
                                  f"({e!r}); degrading to plain bm25 BQL ranking.", file=sys.stderr)
                            dense_belief = None
                    else:
                        import sys
                        print(f"WARNING: BQL_DENSE=1 but no persisted dense doc-embedding cache "
                              f"for corpus key {key!r}; degrading to plain bm25 BQL ranking. "
                              f"Prebuild with: python -m agent_search.evaluation.build_indexes --retriever "
                              f"dense --model {self.dense_model} ...", file=sys.stderr)
            from agent_search.retrievers.backend import build_bql_engine
            self._bql = build_bql_engine(self._units, self.index_root, key, self.rebuild,
                                         dense=dense_belief)
        if arm in ("bqldonlyvisit", "bqldonlysnip"):
            # DENSE-ONLY BQL ranking (see agent_search/retrievers/bql/dense_fuse.py's
            # "dense-ONLY ordering" section and executor.py's DenseOnlyStructuralExecutor). Kept
            # as its OWN block (NOT folded into the 'code'/'doc'/.../'bqldensefetch' block above)
            # so the shared `build_bql_engine(...)` call at the end of that block — which builds
            # the RRF-fusing engine the BQL_DENSE arms use — is never reached for these two arms;
            # instead `build_bql_engine_dense_only` (backend.py, the dense-only sibling of
            # `build_bql_engine`) resolves the SAME `STRUCTURED_BACKEND` env (python ->
            # `load_or_build_dense_only`'s DenseOnlyStructuralExecutor class-swap; lucene ->
            # `LuceneBqlDonlyAdapter`, the dense-only sibling of `LuceneBqlAdapter` — see
            # docs/RUN_RECORD.md's env_knobs for how STRUCTURED_BACKEND/BQL_DENSE are recorded).
            # SAME persisted dense doc-embedding cache + SAME fail-loud missing-cache contract as
            # 'bqldensevisit'/'bqldensesnip' above (this ranking knob never live-encodes a whole
            # corpus at eval time either).
            from agent_search.retrievers.dense import DenseRetriever
            from agent_search.retrievers.dense import DenseBelief
            probe = DenseRetriever(model=self.dense_model, index_root=self.index_root)
            cond_label = {"bqldonlyvisit": "research_bql_donly_visit",
                         "bqldonlysnip": "research_bql_donly_snip"}[arm]
            if not self.rebuild and not probe.is_cached(key):
                raise RuntimeError(
                    f"{cond_label} ({arm} arm) needs a persisted dense doc-embedding cache "
                    f"for corpus key {key!r} at {probe._cache_dir(key)!r} — none found. "
                    f"Prebuild it with: python -m agent_search.evaluation.build_indexes --retriever dense "
                    f"--model {self.dense_model} ... (this ranking knob does not "
                    f"live-encode the corpus at eval time).")
            try:
                dense_belief = DenseBelief(
                    model=self.dense_model, index_root=self.index_root).build_or_load(
                        self._units, key=key)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{cond_label} ({arm} arm): failed to load the dense embedding cache or "
                    f"model for corpus key {key!r}: {type(e).__name__}: {e}") from e
            from agent_search.retrievers.backend import build_bql_engine_dense_only
            self._bql = build_bql_engine_dense_only(self._units, self.index_root, key,
                                                    self.rebuild, dense=dense_belief)
        if arm in ("bm25", "bm25dci", "bm25fetch", "bm25q", "bm25fetchsnip",
                   "hybridvisit", "hybridfetchsnip", "bm25autoread", "hybridautoread"):
            # 'bm25q' reuses the SAME bm25 retrieval as 'bm25' — the
            # hardened baseline's only difference is the listing's snippet render, not retrieval.
            # 'bm25fetchsnip' (research_bm25_fetch_snip) likewise reuses the SAME bm25
            # retrieval as 'bm25fetch' — the fair-listing sibling's only difference is the
            # listing's content, not retrieval. 'hybridvisit'/'hybridfetchsnip' (research_hybrid/
            # research_hybrid_fetch_snip) ALSO reuse this SAME bm25 engine — it's one of the two
            # rankers RRF fuses (see search_visit.py's HybridVisit / search_fetch.py's HybridFetchSnipWorkspace); the
            # dense half is built separately below, alongside densevisit/densefetch's.
            # 'bm25autoread' (research_bm25_autoread) ALSO reuses this SAME bm25 engine — its
            # ranking matches 'bm25'; only the RENDER (full text vs a listing to
            # visit) differs (see search_visit.py's Bm25AutoRead). The engine itself (BM25Local
            # vs canonical-Lucene BM25Pyserini) is env `BM25_BACKEND`-selectable — see
            # `_build_bm25_engine` above; default 'local' keeps every one of these arms on the
            # same in-process engine. 'hybridautoread'
            # ALSO reuses this SAME bm25 engine — it's one of the two rankers RRF fuses, exactly
            # like 'hybridvisit'/'hybridfetchsnip'.
            self._bm25 = _build_bm25_engine(self._units, self.index_root, self.rebuild, key)
        if arm in ("indri", "indrivisit", "indrisnip"):
            # SAME load_or_build pattern as the BQL executor above (prewarmed-if-cached, else
            # build in memory and persist) — see agent_search.retrievers.indri.model.
            # 'indrivisit'/'indrisnip' reuse the SAME executor/index artifact as 'indri' (the
            # snippet rendering and whole-doc-visit read are workspace-level, not a separate
            # index).
            # Env `STRUCTURED_BACKEND` (default 'python', unchanged) selects the engine —
            # see agent_search.retrievers.backend.build_indri_engine; 'lucene' opens
            # the prebuilt indexes/lucene_structured/<key>/ index instead of the .pkl.
            from agent_search.retrievers.backend import build_indri_engine
            # INDRI_DENSE=1 attaches the dense-embedding belief (needs the bge doc-embedding
            # cache — built once by a GPU job; missing cache degrades to lexical-only with a
            # warning). OFF by default (env unset), so Indri retrieval stays lexical-only.
            dense = None
            import os
            if os.environ.get("INDRI_DENSE") in ("1", "true", "yes"):
                try:
                    from agent_search.retrievers.dense import DenseBelief
                    # DEFAULT_MODEL (env `DENSE_MODEL`-overridable), passed explicitly — SAME
                    # doc embedder the densevisit/densefetch arms below use (DENSE_BASELINE_MODEL
                    # is this SAME import under a different local name).
                    dense = DenseBelief(model=self.dense_model).build_or_load(
                        self._units, key=key)
                except Exception as e:
                    import sys
                    print(f"WARNING: INDRI_DENSE=1 but DenseBelief build/load failed ({e!r}); "
                          f"degrading to lexical-only Indri retrieval.", file=sys.stderr)
                    dense = None
            self._indri = build_indri_engine(self._units, self.index_root, key, self.rebuild,
                                             dense=dense)
        if arm in ("densevisit", "densefetch", "densefetchplain", "hybridvisit",
                   "hybridfetchsnip", "denseautoread", "hybridautoread"):
            # These seven dense/hybrid arms (research_dense, research_dense_fetch, its
            # excerpt-free "plain" sibling, research_hybrid, research_hybrid_fetch_snip,
            # research_dense_autoread, and its hybrid analog) ALL need a persisted dense
            # doc-embedding cache (this baseline does not live-encode a whole corpus at eval time
            # — that would silently turn a "read strategy" ablation into an "also pay an
            # embed-the-corpus tax" run, and for a corpus the size of browsecomp_plus/hotpotqa a
            # cold encode is minutes-to-hours on the agent's clock). Prebuild it with
            # agent_search/evaluation/build_indexes.py --retriever dense (same cache Indri's DenseBelief / the
            # `dense` retriever condition already use). FAIL LOUD here — at index() time, before
            # any episode runs — rather than degrading, so a missing cache is never mistaken for
            # "the model just didn't find anything." SAME check/cache for all seven arms (only the
            # workspace built in _workspace() below differs; 'densefetchplain' reuses the SAME
            # self._dense_belief as 'densefetch', no separate cache/model;
            # 'hybridvisit'/'hybridfetchsnip'/'hybridautoread' also need self._bm25, built in the
            # block above — RRF fuses both; 'denseautoread' (research_dense_autoread) reuses the
            # SAME self._dense_belief as 'densevisit' — its ranking matches, only the
            # RENDER differs, exactly like 'bm25autoread' reuses self._bm25 above; 'hybridautoread'
            # reuses the SAME self._bm25/self._dense_belief
            # as 'hybridvisit' — same reasoning).
            from agent_search.retrievers.dense import DenseRetriever
            from agent_search.retrievers.dense import DenseBelief
            probe = DenseRetriever(model=self.dense_model, index_root=self.index_root)
            arm_label = arm    # "densevisit" | "densefetch" | "densefetchplain" | "hybridvisit" |
                                # "hybridfetchsnip" | "denseautoread" | "hybridautoread"
            cond_label = {"densevisit": "research_dense", "densefetch": "research_dense_fetch",
                         "densefetchplain": "research_dense_fetch_plain",
                         "hybridvisit": "research_hybrid",
                         "hybridfetchsnip": "research_hybrid_fetch_snip",
                         "denseautoread": "research_dense_autoread",
                         "hybridautoread": "research_hybrid_autoread"}[arm]
            if not self.rebuild and not probe.is_cached(key):
                raise RuntimeError(
                    f"{cond_label} ({arm_label} arm) needs a persisted dense doc-embedding "
                    f"cache for corpus key {key!r} at {probe._cache_dir(key)!r} — none found. "
                    f"Prebuild it with: python -m agent_search.evaluation.build_indexes --retriever dense "
                    f"--model {self.dense_model} ... (this baseline does not live-encode the "
                    f"corpus at eval time).")
            try:
                self._dense_belief = DenseBelief(
                    model=self.dense_model, index_root=self.index_root).build_or_load(
                        self._units, key=key)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{cond_label} ({arm_label} arm): failed to load the dense embedding cache "
                    f"or model for corpus key {key!r}: {type(e).__name__}: {e}") from e
        # 'grep' (live regex scan) and 'dci' (bash+read over a flat export) build NOTHING
        # persistent here — grep re-scans the in-memory units per call (like GrepBaseline);
        # dci's flat-file export is itself cached by corpus key inside DciWorkspace/
        # flat_export.export_flat_corpus, not a retriever-level index. 'bm25dci' and 'bm25fetch'
        # prewarm the SAME BM25Local as 'bm25' above (same retrieval); 'bm25dci' stages
        # a bounded flat export per query, and 'bm25fetch'/'densefetch' derive sections live (no
        # BQL executor) — 'densefetch' reuses the SAME self._dense_belief built above.
        # 'hybridvisit'/'hybridfetchsnip'/'hybridautoread' reuse BOTH self._bm25 (built above)
        # and self._dense_belief (built here) — nothing new to build, only fusion at query time.
        # 'bqldonlyvisit'/'bqldonlysnip' build their OWN dense-attached executor above (their own
        # block, not this one) — self._bql there is a DenseOnlyStructuralExecutor, not the plain
        # StructuralExecutor 'bqldensevisit'/'bqldensesnip' get.
        return self

    def _doc_text(self, doc_id: str):
        """'title\nbody' of a unit, for history-conditioned retrieval queries."""
        u = self._ubyid.get(doc_id)
        if u is None:
            return None
        return "\n".join(p for p in (u.title, u.body or u.code) if p)

    def set_files(self, files: dict) -> None:
        self._files = files or {}

    def _workspace(self, k: int, query: str = ""):
        """Build this condition's workspace (per episode). One doc_id map and one prewarmed
        executor/engine are reused across queries (built in index()).

        `query` is consumed by the 'bm25dci' and 'bm25fetch' arms: their bm25 retrieval (and, for
        bm25dci, the bounded flat-export staging) happens ONCE at construction, so they need the
        episode's query text before the agent's first tool call — every other arm's workspace is
        query-agnostic at construction (their `search`/`bm25_search` tool takes the query per CALL
        instead), so they simply ignore the parameter."""
        arm = self._arm
        if arm in WORKSPACES:
            builder = WORKSPACES[arm].builder
            return builder(WorkspaceContext(
                units=self._units, ubyid=self._ubyid, corpus_key=self._key,
                index_root=self.index_root, rebuild=self.rebuild, query=query,
                domain=self.domain, retriever=self))
        if arm == "code":
            from agent_search.legacy.workspaces.code_fix import CodeFixWorkspace
            return CodeFixWorkspace(self._units, self._files, executor=self._bql,
                                    ubyid=self._ubyid)
        if arm == "grep":
            from agent_search.legacy.workspaces.code_grep import GrepReadWorkspace
            return GrepReadWorkspace(self._units, self._files)
        if arm == "bm25":
            from agent_search.legacy.workspaces.search_visit import Bm25Visit
            return Bm25Visit(self._units, engine=self._bm25, ubyid=self._ubyid)
        if arm == "bm25q":
            from agent_search.legacy.workspaces.search_visit import Bm25Visit
            # query_biased=True ONLY here (the bm25q arm) — the SAME bm25
            # retrieval engine as 'bm25' above, just with the listing's snippet rendered
            # query-biased (see search_visit.py's Bm25Visit.search). The plain 'bm25' arm above
            # keeps the constructor default (False).
            return Bm25Visit(self._units, engine=self._bm25, ubyid=self._ubyid,
                             query_biased=True)
        if arm == "bm25autoread":
            from agent_search.legacy.workspaces.search_visit import Bm25AutoRead
            # SAME self._bm25 as 'bm25'/'bm25q' above (same ranking) — the
            # "retrieve-and-read" baseline's only difference is that `search` renders full text
            # instead of a listing to visit (see search_visit.py's Bm25AutoRead).
            return Bm25AutoRead(self._units, engine=self._bm25, ubyid=self._ubyid)
        if arm == "densevisit":
            from agent_search.legacy.workspaces.search_visit import DenseVisit
            # self._dense_belief was built + cache-validated in index() (raises there if the
            # persisted dense cache/model is unavailable — this baseline never live-encodes).
            return DenseVisit(self._units, engine=self._dense_belief, ubyid=self._ubyid,
                              corpus_key=self._key)
        if arm == "denseautoread":
            from agent_search.legacy.workspaces.search_visit import DenseAutoRead
            # SAME self._dense_belief as 'densevisit' above (same ranking) — the
            # "retrieve-and-read" baseline's only difference is that `search` renders full text
            # instead of a listing to visit (see search_visit.py's DenseAutoRead), exactly
            # mirroring how 'bm25autoread' reuses self._bm25 over 'bm25'.
            return DenseAutoRead(self._units, engine=self._dense_belief, ubyid=self._ubyid,
                                 corpus_key=self._key)
        if arm == "densefetch":
            from agent_search.legacy.workspaces.search_fetch import DenseFetchWorkspace
            # SAME self._dense_belief as 'densevisit' above (built + cache-validated in index());
            # `query` is the fallback the workspace uses only if a `dense_search_f` call omits it
            # (mirrors 'bm25fetch' passing `query` to Bm25FetchWorkspace) — retrieval is LIVE per
            # call either way.
            return DenseFetchWorkspace(self._units, query, engine=self._dense_belief,
                                       ubyid=self._ubyid, corpus_key=self._key)
        if arm == "densefetchplain":
            from agent_search.legacy.workspaces.search_fetch import DenseFetchPlainWorkspace
            # SAME self._dense_belief as 'densevisit'/'densefetch' above (built + cache-validated
            # in index()); `query` is the fallback the workspace uses only if a `dense_search_fp`
            # call omits it — retrieval is LIVE per call either way. Only the listing differs from
            # 'densefetch' (no per-hit excerpt); see search_fetch.py's `DenseFetchPlainWorkspace`.
            return DenseFetchPlainWorkspace(self._units, query, engine=self._dense_belief,
                                            ubyid=self._ubyid, corpus_key=self._key)
        if arm == "hybridvisit":
            from agent_search.legacy.workspaces.search_visit import HybridVisit
            # SAME self._bm25 (built alongside 'bm25'/'bm25fetch' etc.) and self._dense_belief
            # (built + cache-validated alongside 'densevisit'/'densefetch') — RRF fusion happens
            # LIVE per search() call, nothing new to build here.
            return HybridVisit(self._units, bm25_engine=self._bm25, dense_engine=self._dense_belief,
                               ubyid=self._ubyid, corpus_key=self._key)
        if arm == "hybridfetchsnip":
            from agent_search.legacy.workspaces.search_fetch import HybridFetchSnipWorkspace
            # SAME self._bm25/self._dense_belief as 'hybridvisit'; `query` is the fallback the
            # workspace uses only if a `hybrid_search_snip` call omits it (mirrors 'bm25fetch'/
            # 'densefetch' passing `query` to their own workspaces) — retrieval is LIVE per call.
            return HybridFetchSnipWorkspace(self._units, query, bm25_engine=self._bm25,
                                            dense_engine=self._dense_belief, ubyid=self._ubyid,
                                            corpus_key=self._key)
        if arm == "hybridautoread":
            from agent_search.legacy.workspaces.search_visit import HybridAutoRead
            # SAME self._bm25/self._dense_belief as 'hybridvisit'/'hybridfetchsnip' above — the
            # "retrieve-and-read" baseline's only difference is that `search` renders full text
            # instead of a listing to visit (see search_visit.py's HybridAutoRead), exactly
            # mirroring how 'bm25autoread'/'denseautoread' reuse their own single-engine siblings.
            return HybridAutoRead(self._units, bm25_engine=self._bm25,
                                  dense_engine=self._dense_belief, ubyid=self._ubyid,
                                  corpus_key=self._key)
        if arm == "dci":
            from agent_search.legacy.workspaces.doc_dci import DciWorkspace
            return DciWorkspace(self._units, corpus_key=self._key)
        if arm == "bm25dci":
            from agent_search.legacy.workspaces.doc_bm25_dci import Bm25DciWorkspace
            return Bm25DciWorkspace(self._units, query, engine=self._bm25, ubyid=self._ubyid)
        if arm == "bm25fetch":
            from agent_search.legacy.workspaces.search_fetch import Bm25FetchWorkspace
            return Bm25FetchWorkspace(self._units, query, engine=self._bm25, ubyid=self._ubyid)
        if arm == "bm25fetchsnip":
            from agent_search.legacy.workspaces.search_fetch import Bm25FetchSnipWorkspace
            # SAME construction shape as 'bm25fetch' (self._bm25 built in index() above) — the
            # fair-listing sibling's only difference is the search listing's rendering, not
            # retrieval or construction (see search_fetch.py's Bm25FetchSnipWorkspace).
            return Bm25FetchSnipWorkspace(self._units, query, engine=self._bm25, ubyid=self._ubyid)
        if arm == "docv2":
            from agent_search.legacy.workspaces.sieve import DocSearchFetch
            # date_nudge=True ONLY here (the docv2 arm) — a mechanical, corpus-free
            # mid-episode hint toward the typed date[RANGE] surface its skill teaches (see
            # sieve.py's `_has_bare_temporal_clue`). The plain 'doc' arm below keeps the
            # constructor default (False).
            return DocSearchFetch(self._units, executor=self._bql, ubyid=self._ubyid,
                                  coverage=True, date_nudge=True)
        if arm == "docsnip":
            from agent_search.legacy.workspaces.sieve import DocSearchFetch
            # snippets=True ONLY here (the docsnip/research_snip arm) — each search hit gets an
            # appended one-line best-matching excerpt (see sieve.py's `_best_line`). The
            # plain 'doc' arm above keeps the constructor default (False).
            return DocSearchFetch(self._units, executor=self._bql, ubyid=self._ubyid,
                                  snippets=True)
        if arm == "bqlvisit":
            from agent_search.legacy.workspaces.sieve import BqlVisitWorkspace
            # BqlVisitWorkspace forces coverage=True + date_nudge=True (the docv2 arm's SAME BQL v2
            # search) + snippets=True (fairness parity, forced inside the class itself — not a
            # caller knob, matching IndriVisitWorkspace's own pattern) — nothing extra to pass.
            return BqlVisitWorkspace(self._units, executor=self._bql, ubyid=self._ubyid)
        if arm == "bqldensevisit":
            from agent_search.legacy.workspaces.sieve import BqlVisitWorkspace
            # BYTE-FOR-BYTE the 'bqlvisit' construction above, EXCEPT: (1) `self._bql` here is
            # the DENSE-ATTACHED executor built in index() (BQL_DENSE — see
            # bql/dense_fuse.py), and (2) `tool_names` swaps the tool surface to
            # search_bqld/visit_bqld (this condition's own toolset marker — see tools.yaml's
            # bql_dense_visit). No other behavior differs.
            return BqlVisitWorkspace(self._units, executor=self._bql, ubyid=self._ubyid,
                                     tool_names=("search_bqld", "visit_bqld"))
        if arm == "bqldonlyvisit":
            from agent_search.legacy.workspaces.sieve import BqlDonlyVisitWorkspace
            # BYTE-FOR-BYTE the 'bqldensevisit' construction above, EXCEPT (1) `self._bql` here
            # is a DenseOnlyStructuralExecutor (built by 'bqldonlyvisit's own index() block, dense
            # rank ONLY — see bql/dense_fuse.py's "dense-ONLY ordering" section) and (2) the
            # workspace class is `BqlDonlyVisitWorkspace` (a thin dispatch subclass so this
            # condition's own tool names — search_bqldo/visit_bqldo — need no changes to
            # `BqlVisitWorkspace.run`). `tool_names` is unaffected — passed through unchanged.
            return BqlDonlyVisitWorkspace(self._units, executor=self._bql, ubyid=self._ubyid,
                                          tool_names=("search_bqldo", "visit_bqldo"))
        if arm == "bqldensesnip":
            from agent_search.legacy.workspaces.sieve import DocSearchFetch
            # BYTE-FOR-BYTE the 'docsnip' construction below, EXCEPT `self._bql` is the
            # DENSE-ATTACHED executor built in index() (BQL_DENSE — see bql/dense_fuse.py).
            # `run()` already accepts search_bqlds/fetch_bqlds as aliases (see sieve.py);
            # `self.tools` stays the class default here, matching 'docsnip's own
            # pattern of not overriding it for its search_s/fetch_s tool names either.
            return DocSearchFetch(self._units, executor=self._bql, ubyid=self._ubyid,
                                  snippets=True)
        if arm == "bqldonlysnip":
            from agent_search.legacy.workspaces.sieve import DocSearchFetchDonlySnip
            # BYTE-FOR-BYTE the 'bqldensesnip' construction above, EXCEPT (1) `self._bql` here is
            # a DenseOnlyStructuralExecutor (built by 'bqldonlysnip's own index() block — see
            # bql/dense_fuse.py's "dense-ONLY ordering" section) and (2) the workspace class is
            # `DocSearchFetchDonlySnip` (a thin dispatch subclass so this condition's own tool
            # names — search_bqldos/fetch_bqldos — need no changes to `DocSearchFetch.run`).
            return DocSearchFetchDonlySnip(self._units, executor=self._bql, ubyid=self._ubyid,
                                           snippets=True)
        if arm == "bqldensefetch":
            from agent_search.legacy.workspaces.sieve import DocSearchFetch
            # BYTE-FOR-BYTE the plain 'doc' construction (fallback at the bottom of this method)
            # EXCEPT `self._bql` is the DENSE-ATTACHED executor built in index() (BQL_DENSE — see
            # bql/dense_fuse.py). `snippets` stays the constructor default (False) — this is
            # research_bql_dense_snip's SAME search minus the excerpt. `run()` already accepts
            # search_bqldf/fetch_bqldf as aliases (see sieve.py); `self.tools` stays the
            # class default here, matching 'docsnip'/'bqldensesnip's own pattern.
            return DocSearchFetch(self._units, executor=self._bql, ubyid=self._ubyid)
        if arm == "indri":
            from agent_search.legacy.workspaces.doc_indri import IndriFetchWorkspace
            return IndriFetchWorkspace(self._units, executor=self._indri, ubyid=self._ubyid)
        if arm == "indrivisit":
            from agent_search.legacy.workspaces.doc_indri import IndriVisitWorkspace
            # snippets=True is FORCED inside IndriVisitWorkspace itself (not a caller knob) —
            # see doc_indri.py's IndriVisitWorkspace docstring (content-bearing
            # listing for fairness parity with the bm25 baseline).
            return IndriVisitWorkspace(self._units, executor=self._indri, ubyid=self._ubyid)
        if arm == "indrisnip":
            from agent_search.legacy.workspaces.doc_indri import IndriFetchWorkspace
            # snippets=True ONLY here (the indrisnip/research_indri_snip arm) — each isearch hit
            # gets an appended one-line best-matching excerpt (see doc_indri.py's
            # `_indri_query_terms`). The plain 'indri' arm above keeps the constructor default
            # (False).
            return IndriFetchWorkspace(self._units, executor=self._indri, ubyid=self._ubyid,
                                       snippets=True)
        from agent_search.legacy.workspaces.sieve import DocSearchFetch
        return DocSearchFetch(self._units, executor=self._bql, ubyid=self._ubyid)

    def search(self, query: str, k: int) -> list:
        ws = self._workspace(k, query)
        # DRIVER SWITCH: AGENT_DRIVER=sdk runs the doc arms through the OpenAI Agents SDK (native
        # tool-calling) instead of the text-parsed loop. Only the doc arms (plain <answer> terminal)
        # are wired; code arms' <fix>-guard terminal isn't ported. The SDK result is adapted into the
        # SAME Trajectory the harness scores (same token decomposition).
        import os
        # driver: env AGENT_DRIVER wins; else the per-run default (SDK for OpenAI/Gemini LLM runs,
        # loop for stub/tests + vLLM until Tongyi tool-calling is validated on served vLLM).
        driver = os.environ.get("AGENT_DRIVER") or self._driver
        _spec = WORKSPACES.get(self._arm)
        if driver == "sdk" and (_spec is not None and _spec.sdk) and self._arm in (
                "doc", "bm25", "dci", "bm25dci", "bm25fetch", "code", "grep", "docv2", "indri",
                "docsnip", "indrivisit", "indrisnip", "densevisit", "densefetch",
                "densefetchplain", "bm25q",
                "bqlvisit", "bm25fetchsnip", "hybridvisit", "hybridfetchsnip",
                "bqldensevisit", "bqldensesnip", "bqldensefetch", "bm25autoread",
                "denseautoread", "hybridautoread", "bqldonlyvisit", "bqldonlysnip",
                "dedup_bm25", "dedup_dense"):
            traj = self._run_sdk(ws, query)
            self._tl.traj = traj
            self._tl.meta = _trajectory_meta(traj, ws)
            return traj.located
        import agent_search.models as backends
        backends.reset_usage()
        policy = self._policy_factory()
        # a <fix>-terminal arm (code/grep) is gated by a grounding guard: the fix's file must
        # have been searched/grepped AND fetched/read this episode (never a blind guess). The
        # <answer>-terminal arms (doc/bm25/dci/bm25dci/bm25fetch) have no <fix>, no mid-episode gate —
        # their grounding is scored post-hoc (agent_search.evaluation.doc_scoring, against traj.observations).
        if self._arm == "code":
            guard = _code_fix_guard(ws)
        elif self._arm == "grep":
            guard = _code_grep_guard(ws)
        else:
            guard = None
        # The episode's search history: (a) every step gets `hit_ids`/`read_ids` for the run
        # record (training data needs to know what a search listed and what a read opened),
        # (b) a history-conditioned dense retriever (DENSE_QUERY_STYLE != plain) renders its
        # query from the same context, byte-identical to how training rendered it.
        from agent_search.training.history import CURRENT, QueryContext, dense_query_style
        from agent_search.training.triples import is_read_action, is_search_action, read_ids
        ctx = QueryContext(question=query, text_of=self._doc_text, style=dense_query_style())

        def _on_step(step) -> None:
            hits = list(getattr(ws, "last_hits", []) or [])
            step.hit_ids = hits if is_search_action(step.name) else []
            step.read_ids = read_ids({"args": step.args}, hits) if is_read_action(step.name) else []
            ctx.observe(step, hits)

        def _before_tool(name, args, raw) -> None:
            ctx.note(raw)

        token = CURRENT.set(ctx)
        try:
            traj = run_episode(policy, Task(task_id="q", query=query), ws,
                               self._units, max_steps=self.max_steps,
                               usage_fn=backends.usage_events, domain=self.domain,
                               before_tool=_before_tool,
                               fix_guard=guard, on_step=_on_step)
        finally:
            CURRENT.reset(token)
        self._tl.traj = traj
        self._tl.meta = _trajectory_meta(traj, ws)
        return traj.located

    def _run_sdk(self, ws, query: str):
        """Run one episode through the OpenAI Agents SDK (native tool-calling) and ADAPT the result
        into the harness `Trajectory`, so `_trajectory_meta` + the cache-aware token decomposition in
        run_eval consume it unchanged. Doc arms only. To feed that decomposition: step 0's
        `prompt_tokens` gets the FIRST model call's input (= `initial_prompt_tokens`, counted once);
        each Step's observation feeds `retrieved_doc_tokens` (tiktoken); `completion_tokens`/
        `reasoning_tokens` carry the generation. Model routes via `make_agent_model` in the driver."""
        import re as _re
        from datetime import date
        from agent_search.agent.sdk_driver import run_episode_sdk
        from agent_search.agent.loop import Trajectory, Step, resolve_locations
        from agent_search.models import DEFAULT_MODEL
        from agent_search.legacy.prompts import load_prompt_text
        # IDENTICAL prompt to the loop's AgentPolicy: same system (skill manual + task) and same first
        # user turn (`Current date: …\n\n{query}`). The ONLY difference between drivers is the
        # execution mechanism — native SDK tool-calling vs the text-parsed loop — NOT the prompt.
        system = _re.sub(r"\n{3,}", "\n\n", load_prompt_text(self.prompt_path, self._field_profile)).strip()
        system = system.replace("{{step_budget}}", str(self.max_steps))   # loop fills this in run_episode; SDK must too
        user_input = f"Current date: {date.today().isoformat()}\n\n{query}"
        sdk = run_episode_sdk(ws, user_input, model=self._model or DEFAULT_MODEL,
                              api_base=self._api_base, max_turns=self.max_steps,
                              instructions=system)
        steps = [Step(name=n, args=a, observation=o) for (n, a, o) in sdk.steps]
        if steps:                                   # step 0 carries the fixed initial-prompt input
            steps[0].prompt_tokens = sdk.first_input_tokens
        # located = surfaced ranking (last search's ordered hits, else any seen doc); gold-doc
        # coverage reads `ws.seen` directly in _trajectory_meta, so this only feeds ranking metrics.
        located = list(getattr(ws, "last_hits", []) or []) or list(getattr(ws, "seen", []) or [])
        # code arm: ranking is resolved from the <fix>'s declared file (like the loop), else surfaced.
        if self._arm in ("code", "grep") and sdk.fix_text:
            from agent_search.evaluation.fix_scoring import fix_file
            f = fix_file(sdk.fix_text)
            located = resolve_locations([f], self._units) if f else located
        return Trajectory(
            task_id="q", steps=steps, located=located, declared=[],
            llm_calls=sdk.requests, prompt_tokens=sdk.input_tokens,
            completion_tokens=sdk.output_tokens, cached_input_tokens=sdk.cached_input_tokens,
            reasoning_tokens=sdk.reasoning_tokens,
            stopped_reason=("fix" if sdk.fix_text else getattr(sdk, "stopped_reason", "answer")),
            final_answer=sdk.final_answer, fix_text=sdk.fix_text,
            elicitation=sdk.elicitation)


# --- code-fix grounding guard: a <fix> must be searched + fetched, never guessed --------

def _code_fix_guard(ws):
    """Build a fix_guard(fix_text, steps) -> (ok, why) closure over a CodeFixWorkspace.

    A <fix> is accepted only after the episode has (a) run a successful search and (b)
    fetched the exact file the fix names — otherwise it's a guess, bounced back with an
    actionable reason (same policy as the sandbox's probe_code grounding guard). Path
    matching is suffix-lenient, matching agent_search.evaluation.fix_scoring."""
    from agent_search.evaluation.fix_scoring import fix_file, is_grounded

    def guard(fix_text: str, steps) -> tuple:
        searched = any(
            s.name == "search" and not s.observation.startswith(("ERROR", "empty query"))
            and "(0 hits)" not in s.observation
            for s in steps)
        # a path is 'read' only if that fetch spec's own body was not an ERROR line.
        read_paths: set = set()
        import re as _re
        for s in steps:
            if s.name != "fetch":
                continue
            for block in _re.split(r"\n(?=\[\d+\] )", s.observation.strip()):
                fm = _re.match(r"\[\d+\]\s+(\S+)\s+::[^\n]*\n?(.*)", block, _re.DOTALL)
                if fm and not fm.group(2).lstrip().startswith("ERROR"):
                    read_paths.add(fm.group(1))
        file = fix_file(fix_text)
        if not searched:
            return False, ("you have not run a successful search yet. Do NOT guess the fix. "
                           "Your next message must be a single <tool_call> that searches "
                           "for a symbol from the issue.")
        if not file:
            return False, ("your <fix> block has no parseable 'file:' line. Re-emit it with a "
                           "line exactly like 'file: path/to/file.py' (a path you have "
                           "fetched), then 'function:' and 'change:'.")
        if not is_grounded(file, read_paths):
            return False, (f"'{file}' is not a file you have fetched. fetch the exact file you "
                           "intend to change (a path from a search hit), then re-emit <fix> "
                           "with that path.")
        return True, ""

    return guard


# --- code-GREP grounding guard: same policy, grep/read observation shapes ---------------

def _code_grep_guard(ws):
    """The grep baseline's fix_guard(fix_text, steps) -> (ok, why): a <fix> is accepted only
    after a successful `grep` (>=1 match) AND a `read` of the exact file named in `file:` —
    identical POLICY to `_code_fix_guard`, just reading `grep`/`read` observation shapes instead
    of `search`/`fetch` ones."""
    from agent_search.evaluation.fix_scoring import fix_file, is_grounded

    def guard(fix_text: str, steps) -> tuple:
        searched = any(
            s.name == "grep" and not s.observation.startswith("0 matches")
            for s in steps)
        # GrepReadWorkspace.read() echoes "{path} lines s-e of N:\n..." — the resolved path is
        # the observation's first token, UNLESS it errored (no such file / ambiguous name).
        read_paths: set = set()
        for s in steps:
            if s.name != "read" or s.observation.lstrip().startswith("ERROR"):
                continue
            first_line = s.observation.split("\n", 1)[0]
            path = first_line.split(" lines ", 1)[0].strip()
            if path:
                read_paths.add(path)
        file = fix_file(fix_text)
        if not searched:
            return False, ("you have not run a successful grep yet. Do NOT guess the fix. "
                           "Your next message must be a single <tool_call> that greps "
                           "for a symbol from the issue.")
        if not file:
            return False, ("your <fix> block has no parseable 'file:' line. Re-emit it with a "
                           "line exactly like 'file: path/to/file.py' (a path you have "
                           "read), then 'function:' and 'change:'.")
        if not is_grounded(file, read_paths):
            return False, (f"'{file}' is not a file you have read. read the exact file you "
                           "intend to change (a path from a grep hit), then re-emit <fix> "
                           "with that path.")
        return True, ""

    return guard


def _trajectory_meta(traj, ws=None) -> dict:
    """Serialize an episode into the rows.jsonl shape the analysis tools expect.

    `ws` (the episode's workspace) supplies the arm-specific surfaced-doc set for the
    doc arm's gold-doc coverage; the code arm carries its <fix> text for fix-scoring."""
    import re
    steps = []
    for s in traj.steps:
        # the count from a search-shaped observation header — "(N units in M files ...)"
        # (code search->fetch), "(N matches ...)" (doc), or "N units matched /pat/" (code
        # grep baseline); 0 otherwise (dci's bash/read have no such header). Advisory only.
        m = (re.search(r"\((\d+)\s+(?:units|matches)\b", s.observation)
             or re.match(r"(\d+)\s+units matched\b", s.observation))
        n_hits = int(m.group(1)) if m else 0
        steps.append({
            "action": s.name, "args": s.args,
            "query": (s.args.get("query") or s.args.get("q") or ""),
            "observation": s.observation, "raw_output": s.raw_output, "n_hits": n_hits,
            "hit_ids": list(getattr(s, "hit_ids", []) or []),
            "read_ids": list(getattr(s, "read_ids", []) or []),
            "t_llm_s": round(s.t_llm, 3), "t_tool_s": round(s.t_tool, 3),
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
        })
    meta = {
        "queries": [s["query"] for s in steps],
        "actions": [s["action"] for s in steps],
        "hits_per_step": [s["n_hits"] for s in steps],
        "stopped": traj.stopped_reason, "final_answer": traj.final_answer,
        # provenance of a non-empty final_answer on a force-answer-gated episode: None (the
        # episode never reached the reserved final turn), "nudge" (model complied with the
        # inline budget nudge directly), "prefill_inline" (the shared forced-answer-elicitation call
        # filled it in — agent_search/agent/forced_answer.py), "prefill_failed" (that call fired but
        # produced nothing usable). SDK-driven episodes carry the analogous "ask_retry_inline"/
        # "ask_retry_failed" (see agent_search/agent/sdk_driver.py). See loop.py::Trajectory.
        "elicitation": getattr(traj, "elicitation", None),
        "declared": traj.declared, "llm_calls": traj.llm_calls, "n_steps": len(traj.steps),
        "prompt_tokens": traj.prompt_tokens, "completion_tokens": traj.completion_tokens,
        "cached_input_tokens": traj.cached_input_tokens,
        "reasoning_tokens": traj.reasoning_tokens,
        "fix_text": traj.fix_text,
        # docs surfaced this episode (search hits + fetch/visit targets), for gold-doc coverage.
        "surfaced_docs": sorted(getattr(ws, "seen", set()) or set()),
        # every tool response the agent saw lives, in full and uncapped, on
        # `trajectory[i]["observation"]` (see agent_search.evaluation.rows.observations_of).
        "trajectory": steps,
    }
    return meta


# --- registry: ONE builder, ONE condition per conditions.yaml binding -----------
# A condition `agent_<name>` is auto-registered for EVERY binding in conditions.yaml
# (a binding = task template x toolset), so adding a condition is one line of YAML
# with zero code here. The bare `agent` alias maps to a DEFAULT condition, itself a
# parameter (AGENT_DEFAULT_CONDITION) — so no single condition is hardwired as "the
# agent"; change the default and the alias just points elsewhere (or drops out).
import os  # noqa: E402

from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402
from agent_search.legacy.prompts.registry import PROMPTS  # noqa: E402

AGENT_DEFAULT_CONDITION = os.environ.get("AGENT_DEFAULT_CONDITION", "research_snip")
# PROMPTS is {domain: {condition_name: spec}}; gather every condition name across domains.
_REGISTERED = {name for conds_in_domain in PROMPTS.values() for name in conds_in_domain}
_AGENT_CONDITIONS = (tuple(sorted(f"agent_{name}" for name in _REGISTERED))
                     + (("agent",) if AGENT_DEFAULT_CONDITION in _REGISTERED else ()))  # bare alias


def _profile_for(name: str) -> str:
    return AGENT_DEFAULT_CONDITION if name == "agent" else name[len("agent_"):]


def _toolset_for(prompt_path: str) -> tuple:
    from agent_search.legacy.prompts import load_prompt_profile
    return tuple(load_prompt_profile(prompt_path).tool_names)


# built-in workspaces registered through `register_workspace` live in their own modules
from agent_search.legacy.workspaces import doc_dedup as _doc_dedup  # noqa: E402,F401


def _REGISTRY_NAMES():
    from agent_search.retrievers.registry import _REGISTRY
    return _REGISTRY


def _build_agent(cfg: RetrieverConfig, name: str):
    # a condition declared in agent_search.strategies runs through the condition runner
    try:
        from agent_search.strategies.conditions import CONDITIONS
        cond_name = _profile_for(name)
        if cond_name in CONDITIONS:
            from agent_search.evaluation.agent_runner import build_condition_agent
            return build_condition_agent(cfg, cond_name)
    except ImportError:
        pass
    return _build_legacy_agent(cfg, name)


# names the condition registry (agent_search.strategies) already owns run through it; the
# rest keep the workspace path until it is retired
try:
    from agent_search.strategies.conditions import CONDITIONS as _NEW_CONDITIONS
except ImportError:                                            # pragma: no cover
    _NEW_CONDITIONS = {}
for _n in _AGENT_CONDITIONS:
    if _n not in _REGISTRY_NAMES() and _profile_for(_n) not in _NEW_CONDITIONS:
        register(_n)(_build_agent)


def _build_legacy_agent(cfg: RetrieverConfig, name: str):
    """The workspace-based agent (prompts YAML, arm switch). Kept for one release so an
    episode can be compared against the condition runner; retired with this module."""
    from agent_search.legacy.prompts import get_prompt_spec
    # A condition carries its OWN domain (code vs general, from the task front-matter). Use it
    # to pick the arm/workspace, so agent_research is a doc arm even when the CLI's default
    # domain is 'code' — the config's domain only overrides when the caller set it explicitly
    # (run_eval derives it from the dataset).
    spec = get_prompt_spec(_profile_for(name), cfg.domain)
    domain = spec.domain
    prompt_path = cfg.prompt_override or spec.path
    toolset = _toolset_for(prompt_path)
    # ONE dense model per run: the configured `--dense-model` / `DENSE_MODEL`, else the domain's
    # default (`BAAI/bge-base-en-v1.5` for documents, CodeRankEmbed for code). Every dense arm —
    # Sieve's ranker and its fallback, the dense/hybrid baselines, Indri's dense belief — reads
    # the persisted embedding cache built for exactly this model.
    from agent_search.evaluation.datasets import default_dense_model
    dense_model = cfg.dense_model or default_dense_model(domain)

    driver = "loop"                                  # stub/tests never touch the SDK
    if cfg.policy == "stub":
        from agent_search.agent.policies import KeywordPolicy
        policy_factory = lambda: KeywordPolicy(toolset)  # noqa: E731
    else:
        from agent_search.agent.policies import AgentPolicy
        from agent_search.models import (DEFAULT_MODEL, is_gemini_model,
                                                  is_openai_model, make_generate)
        mdl = cfg.model or DEFAULT_MODEL
        gen = make_generate(model=mdl, backend=cfg.backend, api_base=cfg.api_base,
                            tp=cfg.tp, temperature=cfg.temperature, seed=cfg.seed)
        profile = cfg.field_profile or domain       # selects the field-tagged manual variant
        policy_factory = lambda: AgentPolicy(  # noqa: E731
            generate=gen, prompt_path=prompt_path, field_profile=profile)
        # SDK is the DEFAULT. The text-parsed chat-template loop is the fallback ONLY when the SDK
        # cannot drive the model — i.e. IN-PROCESS vLLM (`backend=vllm`), which the SDK can't reach
        # (it only talks HTTP). OpenAI, Gemini, and SERVED vLLM (`backend=api`, OpenAI-compatible,
        # incl. served Tongyi) all default to the SDK. `AGENT_DRIVER=loop|sdk` overrides.
        sdk_reachable = is_openai_model(mdl) or is_gemini_model(mdl) or cfg.backend == "api"
        if sdk_reachable:
            try:
                import agents  # noqa: F401  (openai-agents, optional)
                driver = "sdk"
            except ImportError:
                print("[agent] optional OpenAI Agents SDK not installed "
                      "(`pip install openai-agents`) — falling back to the text-parsed "
                      "loop driver. Set AGENT_DRIVER=sdk after installing it to silence this.")

    return lambda: AgentRetriever(
        policy_factory=policy_factory, toolset=toolset, max_steps=cfg.max_steps,
        prompt_path=prompt_path, dense_model=dense_model, index_root=cfg.index_root,
        rebuild=cfg.rebuild, domain=domain, tool=name,
        # SDK path: same model + SAME skill prompt the loop's AgentPolicy loads; driver auto-selected.
        model=cfg.model, api_base=cfg.api_base, field_profile=cfg.field_profile or domain,
        driver=driver)
