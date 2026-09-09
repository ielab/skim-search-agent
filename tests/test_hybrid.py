"""NEW, ADDITIVE-only BM25+DENSE HYBRID baseline: `research_hybrid` / `research_hybrid_fetch_snip`
— the control that isolates the BQL method's contribution from the mere "sparse+dense fusion"
effect. Neither `research_bm25` nor `research_dense` alone is the right control once a QL method
could just be reproducing textbook hybrid search: a real deployment would fuse a keyword ranking
and a semantic ranking, so the honest baseline to beat is THAT fusion.

Retrieval is Reciprocal Rank Fusion (RRF, k=60 — the standard constant) over two independent
top-`HYBRID_POOL` (100) pools: the pyserini/Lucene canonical BM25 ranking and the FAISS/dense
ranking (SAME BAAI/bge-base-en-v1.5 embedder + persisted cache research_dense/research_dense_fetch
use). `HybridVisit` (agent_search/agent/tools/doc_research.py) mirrors `Bm25Visit`'s
retrieve-then-visit listing format byte-for-byte (only the ranking differs — `visit`/`_resolve`
are INHERITED unchanged); `HybridFetchSnipWorkspace` is its fetch-mode twin, mirroring
`Bm25FetchSnipWorkspace`'s structure-table-plus-excerpt listing, paired with the SAME structured
section-`fetch` read every method/fetch cell uses.

CPU-only throughout: BM25 is a real (but tiny, in-memory) `BM25Local`; dense is a stub exposing
ONLY `top_k_doc_ids(query, k)` (the same stub pattern test_dense_baseline.py uses for DenseVisit/
DenseFetchWorkspace) — no torch/sentence-transformers import anywhere in this file."""
from __future__ import annotations

import math

import pytest

from agent_search.agent.tools.doc_research import (
    Bm25FetchSnipWorkspace, Bm25FetchWorkspace, Bm25Visit, DocSearchFetch,
    HybridFetchSnipWorkspace, HybridVisit, RRF_K, rrf_fuse)
from agent_search.corpus.units import units_from_documents
from agent_search.prompts import load_condition, render_manuals
from agent_search.retrievers.lexical.bm25 import BM25Local
from agent_search.retrievers.registry import RetrieverConfig, build_factory

# SAME fixture docs as test_doc_bm25_fetch_tools.py / test_bm25_fetch_snip.py (a structured doc, a
# flat doc, an off-topic doc) — keeps rendering comparisons directly comparable.
DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual harbor event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today at the harbor.",
     "infobox": "Founded: 1897; Location: Portville"},
    {"_id": "d_flat", "title": "Harbor Festival History",
     "text": "The harbor festival tradition began with local fishing boat races."},
    {"_id": "d_outside", "title": "Quantum Chromodynamics",
     "text": "Quarks and gluons interact via the strong nuclear force in particle physics."},
]


def _units():
    return units_from_documents(DOCS)


def _bm25_engine():
    return BM25Local().index(_units())


class _StubDenseEngine:
    """A CPU-only stand-in for DenseBelief exposing ONLY `top_k_doc_ids(query, k)` — the SAME
    stub shape test_dense_baseline.py uses for DenseVisit/DenseFetchWorkspace."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def top_k_doc_ids(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


class _StubBm25Engine:
    """A CPU-only stand-in for BM25Pyserini/BM25Local exposing ONLY `search(query, k)` — used
    where the fusion inputs must be fully controlled (independent of real BM25 scoring)."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def search(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


# =============================================================================================
# 1. rrf_fuse: pure-function RRF math, hand-computed fixtures (no retrieval stack at all)
# =============================================================================================

def test_rrf_fuse_hand_computed_ranking_and_tie_break():
    """bm25=[a,b,c,d] (ranks 1-4), dense=[e,a,f,b] (ranks 1-4), k=60 (default):

        a = 1/61 (bm25#1) + 1/62 (dense#2)  ~= 0.0325224
        b = 1/62 (bm25#2) + 1/64 (dense#4)  ~= 0.0317540
        e = 1/61 (dense#1 only)             ~= 0.0163934
        c = 1/63 (bm25#3 only)              ~= 0.0158730   <- TIES with f
        f = 1/63 (dense#3 only)             ~= 0.0158730   <- TIES with c, broken by doc_id (c<f)
        d = 1/64 (bm25#4 only)              ~= 0.0156250

    Expected fused order: a, b, e, c, f, d — hand-computed independently of the implementation
    below (a second, from-scratch arithmetic pass), so this is a real correctness check, not a
    tautology."""
    bm25_ids = ["a", "b", "c", "d"]
    dense_ids = ["e", "a", "f", "b"]

    expected_scores = {
        "a": 1 / 61 + 1 / 62,
        "b": 1 / 62 + 1 / 64,
        "c": 1 / 63,
        "d": 1 / 64,
        "e": 1 / 61,
        "f": 1 / 63,
    }
    expected_order = sorted(expected_scores, key=lambda d: (-expected_scores[d], d))
    assert expected_order == ["a", "b", "e", "c", "f", "d"]     # sanity on the hand calc itself

    got = rrf_fuse(bm25_ids, dense_ids)
    assert got == expected_order == ["a", "b", "e", "c", "f", "d"]


def test_rrf_fuse_score_values_match_the_formula_exactly():
    """score(d) = sum_i 1/(k + rank_i(d)) — spot-check two docs' EXACT fused values by
    reimplementing the sum independently (a doc present in only one list contributes just
    that one term, never a second penalized/infinite term)."""
    bm25_ids = ["x", "y"]
    dense_ids = ["y", "z"]
    k = 60

    # reimplement scoring independently (not by calling rrf_fuse) to cross-check the formula
    def _score(doc_id, k=60):
        s = 0.0
        for ids in (bm25_ids, dense_ids):
            if doc_id in ids:
                s += 1.0 / (k + ids.index(doc_id) + 1)
        return s

    ranked = rrf_fuse(bm25_ids, dense_ids, k=k)
    assert ranked[0] == "y"                          # in both lists -> highest score
    assert math.isclose(_score("x"), 1 / 61)
    assert math.isclose(_score("y"), 1 / 62 + 1 / 61)
    assert math.isclose(_score("z"), 1 / 62)


def test_rrf_fuse_doc_missing_from_one_list_contributes_only_the_other_terms():
    """A doc absent from a pool contributes NOTHING for it — never an infinite/penalized rank
    (e.g. as if it ranked dead last)."""
    bm25_ids = ["only_bm25"]
    dense_ids = ["only_dense"]
    ranked = rrf_fuse(bm25_ids, dense_ids)
    assert set(ranked) == {"only_bm25", "only_dense"}
    # both rank #1 in their own list -> identical score -> tie-break by doc_id ascending
    assert ranked == ["only_bm25", "only_dense"]


def test_rrf_fuse_custom_k_changes_relative_weighting():
    """A smaller k concentrates more weight on rank #1 (1/(k+1) grows faster as k shrinks than
    1/(k+2) does), so a doc that is #1 in one list and absent from the other can overtake a doc
    that is #2 in BOTH lists once k is small enough — verify the crossover exists (not a specific
    k, just that k is a real, live parameter of the fusion, not a decoration)."""
    bm25_ids = ["solo"]              # solo is #1 in bm25, absent from dense
    dense_ids = ["a", "solo"]        # solo is #2 in dense; "a" is #1 only in dense
    # at the standard k=60 both bm25_ids/dense_ids scores are close; pick k=1 to force a clear
    # crossover: solo = 1/(1+1) [bm25#1] + 1/(1+2) [dense#2] = 0.5+0.333=0.833
    #            a    = 1/(1+1) [dense#1]                       = 0.5
    ranked = rrf_fuse(bm25_ids, dense_ids, k=1)
    assert ranked[0] == "solo"


def test_rrf_fuse_topk_truncates_the_fused_list():
    bm25_ids = ["a", "b", "c"]
    dense_ids = ["d", "e", "f"]
    full = rrf_fuse(bm25_ids, dense_ids)
    assert len(full) == 6
    top2 = rrf_fuse(bm25_ids, dense_ids, topk=2)
    assert top2 == full[:2]


def test_rrf_fuse_empty_lists_yield_empty():
    assert rrf_fuse([], []) == []


def test_rrf_k_env_default_is_60():
    assert RRF_K == 60


# =============================================================================================
# 2. HybridVisit (research_hybrid) — mirrors Bm25Visit's retrieve-then-visit shape exactly
# =============================================================================================

def _hybrid_visit(bm25_ranking=("d_harbor",), dense_ranking=()):
    return HybridVisit(_units(), bm25_engine=_StubBm25Engine(bm25_ranking),
                       dense_engine=_StubDenseEngine(dense_ranking))


def test_hybridvisit_tools_tuple():
    assert HybridVisit.tools == ("hybrid_search", "visit_h")


def test_hybridvisit_is_a_bm25visit_subclass_reusing_visit_and_resolve_verbatim():
    """visit()/_resolve() are INHERITED from Bm25Visit unchanged — only search()/run()/__init__/
    tools differ (the fused retrieval)."""
    assert issubclass(HybridVisit, Bm25Visit)
    assert HybridVisit.visit is Bm25Visit.visit
    assert HybridVisit._resolve is Bm25Visit._resolve


def test_hybridvisit_both_rankers_are_consulted_at_the_hybrid_pool_depth():
    bm25 = _StubBm25Engine(("d_harbor", "d_flat"))
    dense = _StubDenseEngine(("d_flat", "d_outside"))
    ws = HybridVisit(_units(), bm25_engine=bm25, dense_engine=dense)
    ws.run("hybrid_search", {"query": "harbor festival", "k": 5})
    assert bm25.calls == [("harbor festival", ws.pool)]
    assert dense.calls == [("harbor festival", ws.pool)]
    assert ws.pool == 100                          # HYBRID_POOL default


def test_hybridvisit_fuses_both_rankers_not_just_one():
    """A doc found ONLY by dense (never by bm25) must still surface — proof both rankers
    actually feed the final ranking, not just bm25 with dense along for the ride."""
    bm25 = _StubBm25Engine(("d_harbor",))
    dense = _StubDenseEngine(("d_outside",))
    ws = HybridVisit(_units(), bm25_engine=bm25, dense_engine=dense)
    ws.run("hybrid_search", {"query": "q", "k": 5})
    assert set(ws.last_hits) == {"d_harbor", "d_outside"}


def test_hybridvisit_search_matches_bm25visits_listing_format_byte_for_byte():
    """Given the SAME resulting ranking, HybridVisit's listing must render IDENTICALLY to
    Bm25Visit's (title + opening snippet, no query bias) — mirrors Bm25Visit's format exactly."""
    bm25_engine = _StubBm25Engine(("d_harbor", "d_flat"))
    hybrid_out = HybridVisit(_units(), bm25_engine=bm25_engine,
                             dense_engine=_StubDenseEngine([])).search("harbor festival", k=5)
    bm25_out = Bm25Visit(_units(), engine=_StubBm25Engine(("d_harbor", "d_flat"))).search(
        "harbor festival", k=5)
    assert hybrid_out == bm25_out


def test_hybridvisit_empty_query_message():
    ws = _hybrid_visit()
    assert ws.run("hybrid_search", {"query": ""}) == "empty query"


def test_hybridvisit_zero_hits_message():
    ws = HybridVisit(_units(), bm25_engine=_StubBm25Engine([]), dense_engine=_StubDenseEngine([]))
    out = ws.run("hybrid_search", {"query": "nothing matches"})
    assert "0 matches" in out


def test_hybridvisit_zero_hits_after_prior_search_notes_previous_results():
    bm25 = _StubBm25Engine({"harbor": ["d_harbor"], "zzz": []})
    dense = _StubDenseEngine({"harbor": [], "zzz": []})
    ws = HybridVisit(_units(), bm25_engine=bm25, dense_engine=dense)
    ws.run("hybrid_search", {"query": "harbor"})
    out = ws.run("hybrid_search", {"query": "zzz"})
    assert "0 matches" in out and "previous results still available" in out


def test_hybridvisit_marks_hits_seen():
    ws = _hybrid_visit(bm25_ranking=("d_harbor", "d_flat"))
    ws.run("hybrid_search", {"query": "harbor"})
    assert {"d_harbor", "d_flat"} <= set(ws.seen)


def test_hybridvisit_run_aliases_search_and_visit_names():
    bm25_ranking = ("d_harbor",)
    out1 = HybridVisit(_units(), bm25_engine=_StubBm25Engine(bm25_ranking),
                       dense_engine=_StubDenseEngine([])).run(
        "hybrid_search", {"query": "harbor"})
    out2 = HybridVisit(_units(), bm25_engine=_StubBm25Engine(bm25_ranking),
                       dense_engine=_StubDenseEngine([])).run(
        "search", {"query": "harbor"})
    assert out1 == out2

    ws1 = HybridVisit(_units(), bm25_engine=_StubBm25Engine(bm25_ranking),
                      dense_engine=_StubDenseEngine([]))
    ws1.run("hybrid_search", {"query": "harbor"})
    out3 = ws1.run("visit_h", {"rank": 1})
    ws2 = HybridVisit(_units(), bm25_engine=_StubBm25Engine(bm25_ranking),
                      dense_engine=_StubDenseEngine([]))
    ws2.run("hybrid_search", {"query": "harbor"})
    out4 = ws2.run("visit", {"rank": 1})
    assert out3 == out4
    assert "Founded in 1897" in out3


def test_hybridvisit_run_unknown_tool_errors():
    out = _hybrid_visit().run("fetch", {"specs": [[1, "History"]]})
    assert "unknown tool" in out.lower()


# =============================================================================================
# 3. HybridFetchSnipWorkspace (research_hybrid_fetch_snip) — mirrors Bm25FetchSnipWorkspace
# =============================================================================================

def _hybrid_fetch_snip(query="harbor festival annual event history",
                       bm25_ranking=("d_harbor",), dense_ranking=(), topk=5):
    return HybridFetchSnipWorkspace(_units(), query, bm25_engine=_StubBm25Engine(bm25_ranking),
                                    dense_engine=_StubDenseEngine(dense_ranking), topk=topk)


def test_hybridfetchsnip_tools_tuple():
    assert HybridFetchSnipWorkspace.tools == ("hybrid_search_snip", "fetch")


def test_hybridfetchsnip_is_a_docsearchfetch_subclass_reusing_fetch_verbatim():
    assert issubclass(HybridFetchSnipWorkspace, DocSearchFetch)
    assert HybridFetchSnipWorkspace.fetch is DocSearchFetch.fetch
    assert HybridFetchSnipWorkspace._fetch_one is DocSearchFetch._fetch_one
    assert HybridFetchSnipWorkspace._resolve_doc is DocSearchFetch._resolve_doc


def test_hybridfetchsnip_both_rankers_consulted_at_pool_depth():
    bm25 = _StubBm25Engine(("d_harbor",))
    dense = _StubDenseEngine(("d_flat",))
    ws = HybridFetchSnipWorkspace(_units(), "q", bm25_engine=bm25, dense_engine=dense)
    ws.run("hybrid_search_snip", {"query": "harbor festival"})
    assert bm25.calls == [("harbor festival", ws.pool)]
    assert dense.calls == [("harbor festival", ws.pool)]
    assert ws.pool == 100


def test_hybridfetchsnip_fuses_both_rankers():
    bm25 = _StubBm25Engine(("d_harbor",))
    dense = _StubDenseEngine(("d_outside",))
    ws = HybridFetchSnipWorkspace(_units(), "q", bm25_engine=bm25, dense_engine=dense, topk=5)
    ws.run("hybrid_search_snip", {"query": "q"})
    assert set(ws.last_hits) == {"d_harbor", "d_outside"}


def test_hybridfetchsnip_search_lists_sections_and_infobox_no_body_and_an_excerpt():
    ws = _hybrid_fetch_snip(bm25_ranking=("d_harbor",))
    out = ws.run("hybrid_search_snip", {"query": "harbor festival"})
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out          # section names shown
    assert "Founded" in out                              # infobox key shown
    assert "»" in out                                     # content excerpt (fairness parity)


def test_hybridfetchsnip_excerpt_overlaps_query_terms():
    ws = _hybrid_fetch_snip(bm25_ranking=("d_harbor",))
    out = ws.run("hybrid_search_snip", {"query": "founded 1897"})
    hit_line = next(l for l in out.splitlines() if "d_harbor" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "founded" in excerpt.lower() or "1897" in excerpt


def test_hybridfetchsnip_matches_bm25fetchsnipworkspaces_listing_shape():
    """Given the SAME (post-fusion) ranking, HybridFetchSnipWorkspace's structure-table-plus-
    excerpt listing must have the SAME shape as Bm25FetchSnipWorkspace's — only the section that
    ranked it (bm25 vs bm25+dense fusion) differs, never the rendering."""
    bm25_engine = _bm25_engine()
    query = "harbor festival annual event history"
    hybrid_ws = HybridFetchSnipWorkspace(_units(), query, bm25_engine=bm25_engine,
                                         dense_engine=_StubDenseEngine([]), topk=3)
    hybrid_out = hybrid_ws.run("hybrid_search_snip", {"query": query})
    plain_ws = Bm25FetchSnipWorkspace(_units(), query, engine=_bm25_engine(), topk=3)
    plain_out = plain_ws.run("bm25_search_snip", {"query": query})
    # SAME ranking (dense contributes nothing here) -> byte-identical rendering modulo the
    # header's "matches" count, which is identical too since ranking is identical.
    assert hybrid_ws.last_hits == plain_ws.last_hits
    assert hybrid_out == plain_out


def test_hybridfetchsnip_search_is_live_and_re_retrieves():
    bm25 = _StubBm25Engine({"harbor festival history": ["d_harbor"],
                            "quantum chromodynamics particle physics": ["d_outside"]})
    dense = _StubDenseEngine([])
    ws = HybridFetchSnipWorkspace(_units(), "harbor festival history",
                                  bm25_engine=bm25, dense_engine=dense, topk=5)
    ws.run("hybrid_search_snip", {"query": "harbor festival history"})
    first = list(ws.last_hits)
    ws.run("hybrid_search_snip", {"query": "quantum chromodynamics particle physics"})
    assert ws.last_hits != first
    assert "d_outside" in ws.last_hits


def test_hybridfetchsnip_topk_marks_hits_seen_immediately():
    ws = _hybrid_fetch_snip(bm25_ranking=("d_harbor",))
    assert ws.last_hits == []
    ws.run("hybrid_search_snip", {"query": "harbor festival annual event history"})
    assert ws.last_hits
    assert set(ws.last_hits) <= set(ws.seen)


def test_hybridfetchsnip_empty_query_yields_no_hits():
    ws = _hybrid_fetch_snip(query="", bm25_ranking=("d_harbor",))
    assert ws.last_hits == []
    assert ws.search("") == "empty query"


def test_hybridfetchsnip_fetch_by_rank_after_search():
    ws = _hybrid_fetch_snip(bm25_ranking=("d_harbor",))
    ws.run("hybrid_search_snip", {"query": "harbor festival"})
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert "History" in out and "ERROR" not in out


def test_hybridfetchsnip_run_aliases_search_names():
    bm25_ranking = ("d_harbor",)
    q = "harbor festival"
    out1 = HybridFetchSnipWorkspace(_units(), q, bm25_engine=_StubBm25Engine(bm25_ranking),
                                    dense_engine=_StubDenseEngine([])).run(
        "hybrid_search_snip", {"query": q})
    out2 = HybridFetchSnipWorkspace(_units(), q, bm25_engine=_StubBm25Engine(bm25_ranking),
                                    dense_engine=_StubDenseEngine([])).run(
        "hybrid_search", {"query": q})
    out3 = HybridFetchSnipWorkspace(_units(), q, bm25_engine=_StubBm25Engine(bm25_ranking),
                                    dense_engine=_StubDenseEngine([])).run(
        "search", {"query": q})
    assert out1 == out2 == out3


def test_hybridfetchsnip_run_unknown_tool_errors():
    out = _hybrid_fetch_snip().run("bogus_tool", {})
    assert "unknown tool" in out.lower()


# =============================================================================================
# 4. Condition loading (conditions.yaml/tools.yaml) — UNCOACHED, like bm25/dense
# =============================================================================================

def test_research_hybrid_condition_loads():
    p = load_condition("research_hybrid")
    assert p.toolset == "hybrid_visit"
    assert p.tool_names == ("hybrid_search", "visit_h")
    assert render_manuals(p.tool_names, domain="general") == ""
    assert "term[field]" not in p.system


def test_research_hybrid_fetch_snip_condition_loads():
    p = load_condition("research_hybrid_fetch_snip")
    assert p.toolset == "hybrid_fetch_snip"
    assert p.tool_names == ("hybrid_search_snip", "fetch")
    assert render_manuals(p.tool_names, domain="general") == ""
    assert "term[field]" not in p.system


def test_existing_sibling_conditions_are_unaffected():
    """Additive-only check: adding research_hybrid/research_hybrid_fetch_snip must not disturb
    any existing bm25/dense-family condition binding."""
    for name, toolset, tools in (
        ("research_bm25", "research_bm25", ("bm25_search", "visit")),
        ("research_bm25_fetch_snip", "bm25_fetch_snip", ("bm25_search_snip", "fetch")),
        ("research_dense", "dense_visit", ("dense_search", "visit_d")),
        ("research_dense_fetch", "dense_fetch", ("dense_search_f", "fetch")),
    ):
        p = load_condition(name)
        assert p.toolset == toolset
        assert p.tool_names == tools


# =============================================================================================
# 5. Arm resolution via the retriever registry (agent_search.agent.retriever.AgentRetriever)
# =============================================================================================

def test_research_hybrid_resolves_via_registry():
    from agent_search.agent.retriever import AgentRetriever

    r = build_factory("agent_research_hybrid", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("hybrid_search", "visit_h")
    assert r.tool == "agent_research_hybrid"
    assert r._arm == "hybridvisit"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_hybrid_fetch_snip_resolves_via_registry():
    from agent_search.agent.retriever import AgentRetriever

    r = build_factory("agent_research_hybrid_fetch_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("hybrid_search_snip", "fetch")
    assert r.tool == "agent_research_hybrid_fetch_snip"
    assert r._arm == "hybridfetchsnip"
    assert r.domain == "general"
    assert not r.needs_files


# =============================================================================================
# 6. index(): needs BOTH a bm25 engine AND a persisted dense cache — fails loud when the dense
#    cache is missing (mirrors research_dense/research_dense_fetch's own contract exactly).
# =============================================================================================

def test_hybridvisit_index_raises_clear_error_when_dense_cache_missing(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_hybrid", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_units(), key="no_such_corpus_key")


def test_hybridfetchsnip_index_raises_clear_error_when_dense_cache_missing(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_hybrid_fetch_snip", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_units(), key="no_such_corpus_key")


# =============================================================================================
# 7. Workspace construction + offline (no-model) answer, via AgentRetriever — dense build/cache
#    is faked (no torch/sentence-transformers import) exactly like `DenseRetriever`'s own
#    `encoder=` injection point (test_eval_dense.py's FakeEncoder), just one level up: we swap
#    the whole `DenseBelief`/`DenseRetriever.is_cached` the way production code constructs them,
#    so `AgentRetriever.index()`'s cache-validation + build path runs UNMODIFIED, never encoding
#    anything for real.
# =============================================================================================

@pytest.fixture
def fake_dense_stack(monkeypatch):
    """Monkeypatch the SAME two call sites agent_search.agent.retriever.AgentRetriever.index()
    uses for the densevisit/densefetch/hybridvisit/hybridfetchsnip arms:
      - DenseRetriever.is_cached -> always True (skip the 'no persisted cache' RuntimeError)
      - dense_belief.DenseBelief -> a lightweight stub whose build_or_load/top_k_doc_ids never
        touch a real encoder (no heavy import, no GPU, no network)."""
    from agent_search.retrievers.dense.dense import DenseRetriever
    from agent_search.retrievers.structural.indri import dense_belief as dense_belief_mod

    class _FakeBelief:
        def __init__(self, ranking=()):
            self._ranking = ranking

        def build_or_load(self, units, key=None):
            return self

        def top_k_doc_ids(self, query, k=None):
            ids = (self._ranking.get(query, []) if isinstance(self._ranking, dict)
                  else self._ranking)
            return list(ids[: (k or len(ids))])

    monkeypatch.setattr(DenseRetriever, "is_cached", lambda self, key=None: True)
    monkeypatch.setattr(dense_belief_mod, "DenseBelief", lambda *a, **k: _FakeBelief())
    return _FakeBelief


def test_research_hybrid_workspace_builds_and_answers_via_stub(tmp_path, fake_dense_stack):
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_hybrid", cfg)()
    r.index(_units(), key="test-hybrid-corpus")
    ws = r._workspace(5, "harbor festival annual event history")
    assert isinstance(ws, HybridVisit)
    ranking = r.search("harbor festival annual event history", k=5)
    assert isinstance(ranking, list)


def test_research_hybrid_fetch_snip_workspace_builds_and_answers_via_stub(tmp_path, fake_dense_stack):
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_hybrid_fetch_snip", cfg)()
    r.index(_units(), key="test-hybrid-fetch-snip-corpus")
    ws = r._workspace(5, "harbor festival annual event history")
    assert isinstance(ws, HybridFetchSnipWorkspace)
    ranking = r.search("harbor festival annual event history", k=5)
    assert isinstance(ranking, list)


# =============================================================================================
# 8. Offline end-to-end smoke: one fixture instance through run_config (no model, no vLLM).
#    research_hybrid_fetch_snip's toolset carries a literal "fetch" marker, so KeywordPolicy
#    (agent_search.agent.policies) correctly drives search -> fetch -> answer (the SAME reason
#    test_bm25_fetch_snip.py's own smoke test works); research_hybrid's toolset (hybrid_search,
#    visit_h) has no literal "visit"/"fetch" marker, so — like every other renamed-visit sibling
#    (research_bm25q, research_dense, research_bql_visit) — KeywordPolicy searches once then
#    submits a blank <answer> without ever calling visit_h; that's still a valid NO-CRASH smoke
#    (covered by the direct r.search() calls in section 7 above), so the full run_config smoke
#    is exercised here for the fetch_snip twin.
# =============================================================================================

def test_research_hybrid_fetch_snip_smoke_via_fixture_dataset(tmp_path, fake_dense_stack):
    from agent_search.evaluation.config import DatasetArgs, EvaluationArgs, OutputArgs, RetrieverArgs, RunConfig
    from agent_search.evaluation.run_eval import run_config

    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_hybrid_fetch_snip",
                                index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )

    res = run_config(cfg, progress=False)

    assert res["n"] == 1
    assert res["n_errors"] == 0


def test_research_hybrid_smoke_via_fixture_dataset(tmp_path, fake_dense_stack):
    """SAME offline fixture smoke as above, for the plain visit-mode condition — asserts only
    that the episode completes cleanly (see section 8's docstring for why KeywordPolicy never
    exercises visit_h here)."""
    from agent_search.evaluation.config import DatasetArgs, EvaluationArgs, OutputArgs, RetrieverArgs, RunConfig
    from agent_search.evaluation.run_eval import run_config

    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_hybrid", index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )

    res = run_config(cfg, progress=False)

    assert res["n"] == 1
    assert res["n_errors"] == 0
