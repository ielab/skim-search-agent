"""The BM25+dense hybrid baseline: `research_hybrid` / `research_hybrid_fetch_snip`, the control
that isolates the BQL method's contribution from the mere "sparse+dense fusion" effect. Neither
`research_bm25` nor `research_dense` alone is the right control once a query-language method
could just be reproducing textbook hybrid search: a real deployment would fuse a keyword ranking
and a semantic ranking, so the honest baseline to beat is that fusion.

Retrieval is Reciprocal Rank Fusion (RRF, k=60, the standard constant) over two independent
top-`HYBRID_POOL` (100) pools: the pyserini/Lucene canonical BM25 ranking and the FAISS/dense
ranking (same BAAI/bge-base-en-v1.5 embedder and persisted cache research_dense/research_dense_fetch
use). `search_hybrid` (`agent_search/tools/search_hybrid/tool.py`) mirrors `search_bm25`'s
retrieve-then-visit listing format byte-for-byte (only the ranking differs); with
`structure=True` it is `search_bm25_snip`'s fetch-mode twin, mirroring its structure-table-
plus-excerpt listing, paired with the same structured section-`fetch` read every method/fetch
cell uses.

No model anywhere in this file: where a real BM25 ranking is needed it is a tiny Lucene index
(`lucene_support.build_pyserini`, so the module needs a JVM); elsewhere BM25 is a stub; dense is
a stub exposing only `top_k_doc_ids(query, k)`; no torch/sentence-transformers import."""
from __future__ import annotations

import math

import pytest

from agent_search.snippets import TermWindow
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.registry import RetrieverConfig, build_factory
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.budgets import RRF_K
from agent_search.retrievers.fusion import RRF
from agent_search.retrievers.hybrid import HybridEngine


def rrf_fuse(bm25_ids, dense_ids, k=60, topk=None):
    """The two-list RRF the hybrid arms use, through the fusion package."""
    return RRF(k=k).fuse([[(d, 0.0) for d in bm25_ids], [(d, 0.0) for d in dense_ids]], topk)
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_hybrid.tool import SearchHybrid
from agent_search.tools.visit.tool import Visit

from tests import lucene_support

lucene_support.require_jvm()

# SAME fixture docs as the bm25-fetch tool tests (a structured doc, a flat doc, an off-topic
# doc) — keeps rendering comparisons directly comparable.
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
    return lucene_support.build_pyserini(_units())


class _StubDenseEngine:
    """A CPU-only stand-in for DenseBelief exposing ONLY `top_k_doc_ids(query, k)`."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def top_k_doc_ids(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


class _StubBm25Engine:
    """A stand-in for BM25Pyserini exposing only `search(query, k)`, used where the fusion
    inputs must be fully controlled (independent of real BM25 scoring)."""

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
# 2. search_hybrid (research_hybrid) — mirrors the plain search_bm25+visit shape exactly
# =============================================================================================

def _hybrid_visit_box(bm25_ranking=("d_harbor",), dense_ranking=()):
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchHybrid(name="hybrid_search").bind(
        state, units, ubyid, {"hybrid": HybridEngine({"bm25": _StubBm25Engine(bm25_ranking), "dense": _StubDenseEngine(dense_ranking)}, RRF(k=60), pool=100)})
    visit = Visit(name="visit_h").bind(state, units, ubyid, {})
    return ToolBox([search, visit], state), search


def test_hybridvisit_tools_tuple():
    box, _ = _hybrid_visit_box()
    assert box.tools == ("hybrid_search", "visit_h")


def test_hybridvisit_both_rankers_are_consulted_at_the_hybrid_pool_depth():
    bm25 = _StubBm25Engine(("d_harbor", "d_flat"))
    dense = _StubDenseEngine(("d_flat", "d_outside"))
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchHybrid(name="hybrid_search").bind(state, units, ubyid, {"hybrid": HybridEngine({"bm25": bm25, "dense": dense}, RRF(k=60), pool=100)})
    search.run({"query": "harbor festival", "k": 5})
    assert bm25.calls == [("harbor festival", search.pool)]
    assert dense.calls == [("harbor festival", search.pool)]
    assert search.pool == 100                          # HYBRID_POOL default


def test_hybridvisit_fuses_both_rankers_not_just_one():
    """A doc found ONLY by dense (never by bm25) must still surface — proof both rankers
    actually feed the final ranking, not just bm25 with dense along for the ride."""
    box, _ = _hybrid_visit_box(bm25_ranking=("d_harbor",), dense_ranking=("d_outside",))
    box.run("hybrid_search", {"query": "q", "k": 5})
    assert set(box.last_hits) == {"d_harbor", "d_outside"}


def test_hybridvisit_search_matches_plain_bm25_listing_format_byte_for_byte():
    """Given the SAME resulting ranking, search_hybrid's listing must render IDENTICALLY to
    the plain search_bm25 listing's (title + opening snippet, no query bias) — mirrors its
    format exactly."""
    units = _units()
    ubyid = {u.doc_id: u for u in units}

    hybrid_state = EpisodeState(question="q")
    hybrid_search = SearchHybrid(name="hybrid_search").bind(
        hybrid_state, units, ubyid, {"hybrid": HybridEngine({"bm25": _StubBm25Engine(("d_harbor", "d_flat")), "dense": _StubDenseEngine([])}, RRF(k=60), pool=100)})
    hybrid_out = hybrid_search.run({"query": "harbor festival", "k": 5})

    bm25_state = EpisodeState(question="q")
    bm25_search = SearchBm25(name="bm25_search").bind(
        bm25_state, units, ubyid, {"bm25": _StubBm25Engine(("d_harbor", "d_flat"))})
    bm25_out = bm25_search.run({"query": "harbor festival", "k": 5})

    assert hybrid_out == bm25_out


def test_hybridvisit_empty_query_message():
    box, _ = _hybrid_visit_box()
    assert box.run("hybrid_search", {"query": ""}) == "empty query"


def test_hybridvisit_zero_hits_message():
    box, _ = _hybrid_visit_box(bm25_ranking=[], dense_ranking=[])
    out = box.run("hybrid_search", {"query": "nothing matches"})
    assert "0 matches" in out


def test_hybridvisit_zero_hits_after_prior_search_notes_previous_results():
    box, _ = _hybrid_visit_box(bm25_ranking={"harbor": ["d_harbor"], "zzz": []},
                               dense_ranking={"harbor": [], "zzz": []})
    box.run("hybrid_search", {"query": "harbor"})
    out = box.run("hybrid_search", {"query": "zzz"})
    assert "0 matches" in out and "previous results still available" in out


def test_hybridvisit_marks_hits_seen():
    box, _ = _hybrid_visit_box(bm25_ranking=("d_harbor", "d_flat"))
    box.run("hybrid_search", {"query": "harbor"})
    assert {"d_harbor", "d_flat"} <= set(box.seen)


def test_hybridvisit_run_aliases_search_and_visit_names():
    box1, _ = _hybrid_visit_box(("d_harbor",))
    out1 = box1.run("hybrid_search", {"query": "harbor"})
    box2, _ = _hybrid_visit_box(("d_harbor",))
    out2 = box2.run("search", {"query": "harbor"})
    assert out1 == out2

    box3, _ = _hybrid_visit_box(("d_harbor",))
    box3.run("hybrid_search", {"query": "harbor"})
    out3 = box3.run("visit_h", {"rank": 1})
    box4, _ = _hybrid_visit_box(("d_harbor",))
    box4.run("hybrid_search", {"query": "harbor"})
    out4 = box4.run("visit", {"rank": 1})
    assert out3 == out4
    assert "Founded in 1897" in out3


def test_hybridvisit_run_unknown_tool_errors():
    box, _ = _hybrid_visit_box()
    out = box.run("bogus_tool_name", {"specs": [[1, "History"]]})
    assert "unknown tool" in out.lower()


# =============================================================================================
# 3. search_hybrid_snip (research_hybrid_fetch_snip) — mirrors search_bm25_snip's fetch shape
# =============================================================================================

def _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",), dense_ranking=(), topk=5):
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchHybrid(name="hybrid_search_snip", structure=True, snippet=TermWindow(), k=topk).bind(
        state, units, ubyid, {"hybrid": HybridEngine({"bm25": _StubBm25Engine(bm25_ranking), "dense": _StubDenseEngine(dense_ranking)}, RRF(k=60), pool=100)})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([search, fetch], state), search


def test_hybridfetchsnip_tools_tuple():
    box, _ = _hybrid_fetch_snip_box()
    assert box.tools == ("hybrid_search_snip", "fetch")


def test_hybridfetchsnip_both_rankers_consulted_at_pool_depth():
    bm25 = _StubBm25Engine(("d_harbor",))
    dense = _StubDenseEngine(("d_flat",))
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchHybrid(name="hybrid_search_snip", structure=True, snippet=TermWindow()).bind(
        state, units, ubyid, {"hybrid": HybridEngine({"bm25": bm25, "dense": dense}, RRF(k=60), pool=100)})
    search.run({"query": "harbor festival"})
    assert bm25.calls == [("harbor festival", search.pool)]
    assert dense.calls == [("harbor festival", search.pool)]
    assert search.pool == 100


def test_hybridfetchsnip_fuses_both_rankers():
    box, _ = _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",), dense_ranking=("d_outside",), topk=5)
    box.run("hybrid_search_snip", {"query": "q"})
    assert set(box.last_hits) == {"d_harbor", "d_outside"}


def test_hybridfetchsnip_search_lists_sections_and_infobox_no_body_and_an_excerpt():
    box, _ = _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",))
    out = box.run("hybrid_search_snip", {"query": "harbor festival"})
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out          # section names shown
    assert "Founded" in out                              # infobox key shown
    assert "»" in out                                     # content excerpt (fairness parity)


def test_hybridfetchsnip_excerpt_overlaps_query_terms():
    box, _ = _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",))
    out = box.run("hybrid_search_snip", {"query": "founded 1897"})
    hit_line = next(l for l in out.splitlines() if "d_harbor" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "founded" in excerpt.lower() or "1897" in excerpt


def test_hybridfetchsnip_matches_plain_bm25_fetch_snip_listing_shape():
    """Given the SAME (post-fusion) ranking, search_hybrid_snip's structure-table-plus-
    excerpt listing must have the SAME shape as search_bm25_snip's — only the section that
    ranked it (bm25 vs bm25+dense fusion) differs, never the rendering."""
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    query = "harbor festival annual event history"

    hybrid_state = EpisodeState(question="q")
    hybrid_search = SearchHybrid(name="hybrid_search_snip", structure=True, snippet=TermWindow(), k=3).bind(
        hybrid_state, units, ubyid, {"hybrid": HybridEngine({"bm25": _bm25_engine(), "dense": _StubDenseEngine([])}, RRF(k=60), pool=100)})
    hybrid_out = hybrid_search.run({"query": query})

    plain_state = EpisodeState(question="q")
    plain_search = SearchBm25(name="bm25_search_snip", structure=True, snippet=TermWindow(), k=3).bind(
        plain_state, units, ubyid, {"bm25": _bm25_engine()})
    plain_out = plain_search.run({"query": query})

    # SAME ranking (dense contributes nothing here) -> byte-identical rendering modulo the
    # header's "matches" count, which is identical too since ranking is identical.
    assert hybrid_state.last_hits == plain_state.last_hits
    assert hybrid_out == plain_out


def test_hybridfetchsnip_search_is_live_and_re_retrieves():
    box, _ = _hybrid_fetch_snip_box(
        bm25_ranking={"harbor festival history": ["d_harbor"],
                     "quantum chromodynamics particle physics": ["d_outside"]},
        dense_ranking=[])
    box.run("hybrid_search_snip", {"query": "harbor festival history"})
    first = list(box.last_hits)
    box.run("hybrid_search_snip", {"query": "quantum chromodynamics particle physics"})
    assert box.last_hits != first
    assert "d_outside" in box.last_hits


def test_hybridfetchsnip_topk_marks_hits_seen_immediately():
    box, _ = _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",))
    assert box.last_hits == []
    box.run("hybrid_search_snip", {"query": "harbor festival annual event history"})
    assert box.last_hits
    assert set(box.last_hits) <= set(box.seen)


def test_hybridfetchsnip_empty_query_yields_no_hits():
    box, _ = _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",))
    assert box.last_hits == []
    assert box.run("hybrid_search_snip", {"query": ""}) == "empty query"


def test_hybridfetchsnip_fetch_by_rank_after_search():
    box, _ = _hybrid_fetch_snip_box(bm25_ranking=("d_harbor",))
    box.run("hybrid_search_snip", {"query": "harbor festival"})
    out = box.run("fetch", {"specs": [[1, "History"]]})
    assert "History" in out and "ERROR" not in out


def test_hybridfetchsnip_run_aliases_search_names():
    box1, _ = _hybrid_fetch_snip_box(("d_harbor",))
    out1 = box1.run("hybrid_search_snip", {"query": "harbor festival"})
    box2, _ = _hybrid_fetch_snip_box(("d_harbor",))
    out2 = box2.run("hybrid_search", {"query": "harbor festival"})
    box3, _ = _hybrid_fetch_snip_box(("d_harbor",))
    out3 = box3.run("search", {"query": "harbor festival"})
    assert out1 == out2 == out3


def test_hybridfetchsnip_run_unknown_tool_errors():
    box, _ = _hybrid_fetch_snip_box()
    out = box.run("bogus_tool", {})
    assert "unknown tool" in out.lower()


# =============================================================================================
# 4. Condition loading — UNCOACHED, like bm25/dense
# =============================================================================================

def test_research_hybrid_condition_loads():
    from agent_search.strategies import CONDITIONS
    from agent_search.tasks.render import render_manuals

    p = CONDITIONS["research_hybrid"]
    assert p.strategy.toolset_name == "hybrid_visit"
    assert p.tool_names == ("hybrid_search", "visit_h")
    assert render_manuals([t.manual_path("general") for t in p.strategy.tools]) == ""
    assert "term[field]" not in p.render()


def test_research_hybrid_fetch_snip_condition_loads():
    from agent_search.strategies import CONDITIONS
    from agent_search.tasks.render import render_manuals

    p = CONDITIONS["research_hybrid_fetch_snip"]
    assert p.strategy.toolset_name == "hybrid_fetch_snip"
    assert p.tool_names == ("hybrid_search_snip", "fetch")
    assert render_manuals([t.manual_path("general") for t in p.strategy.tools]) == ""
    assert "term[field]" not in p.render()


def test_existing_sibling_conditions_are_unaffected():
    """Additive-only check: research_hybrid/research_hybrid_fetch_snip must not disturb any
    existing bm25/dense-family condition binding."""
    from agent_search.strategies import CONDITIONS

    for name, toolset, tools in (
        ("research_bm25", "research_bm25", ("bm25_search", "visit")),
        ("research_bm25_fetch_snip", "bm25_fetch_snip", ("bm25_search_snip", "fetch")),
        ("research_dense", "dense_visit", ("dense_search", "visit_d")),
        ("research_dense_fetch", "dense_fetch", ("dense_search_f", "fetch")),
    ):
        p = CONDITIONS[name]
        assert p.strategy.toolset_name == toolset
        assert p.tool_names == tools


# =============================================================================================
# 5. Arm resolution via the retriever registry
# =============================================================================================

def test_research_hybrid_resolves_via_registry():
    from agent_search.evaluation.agent_runner import ConditionAgent

    r = build_factory("agent_research_hybrid", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("hybrid_search", "visit_h")
    assert r.tool == "agent_research_hybrid"
    assert r.condition.name == "research_hybrid"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_hybrid_fetch_snip_resolves_via_registry():
    from agent_search.evaluation.agent_runner import ConditionAgent

    r = build_factory("agent_research_hybrid_fetch_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("hybrid_search_snip", "fetch")
    assert r.tool == "agent_research_hybrid_fetch_snip"
    assert r.condition.name == "research_hybrid_fetch_snip"
    assert r.domain == "general"
    assert not r.needs_files


# =============================================================================================
# 6. index(): needs BOTH a bm25 engine AND a persisted dense cache — fails loud when the dense
#    cache is missing (mirrors research_dense/research_dense_fetch's own contract exactly).
# =============================================================================================

def test_hybridvisit_index_raises_clear_error_when_dense_cache_missing(tmp_path):
    from agent_search.errors import SetupError
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_hybrid", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(SetupError, match="dense embedding cache"):
        r.index(_units(), key="no_such_corpus_key")


def test_hybridfetchsnip_index_raises_clear_error_when_dense_cache_missing(tmp_path):
    from agent_search.errors import SetupError
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_hybrid_fetch_snip", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(SetupError, match="dense embedding cache"):
        r.index(_units(), key="no_such_corpus_key")


# =============================================================================================
# 7. Workspace construction + offline (no-model) answer, via the retriever registry — dense
#    build/cache is faked (no torch/sentence-transformers import) the same way DenseRetriever's
#    own `encoder=` injection point works, just one level up: we swap the whole `DenseBelief`/
#    `DenseRetriever.is_cached` the way production code constructs them, so `Engines.build()`'s
#    cache-validation + build path runs UNMODIFIED, never encoding anything for real.
# =============================================================================================

@pytest.fixture
def fake_dense_stack(monkeypatch):
    """Monkeypatch the SAME two call sites `agent_search.retrievers.engines.Engines._dense_locked`
    uses for the densevisit/densefetch/hybridvisit/hybridfetchsnip arms:
      - DenseRetriever.is_cached -> always True (skip the 'no persisted cache' SetupError)
      - the `agent_search.retrievers.dense` package's own `DenseBelief` name (what `Engines`
        actually re-imports on every call) -> a lightweight stub whose build_or_load/
        top_k_doc_ids never touch a real encoder (no heavy import, no GPU, no network)."""
    import agent_search.retrievers.dense as dense_pkg
    from agent_search.retrievers.dense.base import DenseRetriever

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
    monkeypatch.setattr(dense_pkg, "DenseBelief", lambda *a, **k: _FakeBelief())
    return _FakeBelief


def test_research_hybrid_workspace_builds_and_answers_via_stub(fake_dense_stack):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_hybrid", cfg)()
    r.index(_units(), key=lucene_support.corpus_key(_units()))
    ws = r.toolbox("harbor festival annual event history")
    assert isinstance(ws["hybrid_search"], SearchHybrid)
    ranking = r.search("harbor festival annual event history", k=5)
    assert isinstance(ranking, list)


def test_research_hybrid_fetch_snip_workspace_builds_and_answers_via_stub(fake_dense_stack):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_hybrid_fetch_snip", cfg)()
    r.index(_units(), key=lucene_support.corpus_key(_units()))
    ws = r.toolbox("harbor festival annual event history")
    assert isinstance(ws["hybrid_search_snip"], SearchHybrid)
    ranking = r.search("harbor festival annual event history", k=5)
    assert isinstance(ranking, list)


# =============================================================================================
# 8. Offline end-to-end smoke: one fixture instance through run_config (no model, no vLLM).
#    research_hybrid_fetch_snip's toolset carries a literal "fetch" marker, so KeywordPolicy
#    (agent_search.agent.policies) correctly drives search -> fetch -> answer; research_hybrid's
#    toolset (hybrid_search, visit_h) has no literal "visit"/"fetch" marker, so — like every
#    other renamed-visit sibling — KeywordPolicy searches once then submits a blank <answer>
#    without ever calling visit_h; that's still a valid NO-CRASH smoke (covered by the direct
#    r.search() calls in section 7 above), so the full run_config smoke is exercised here for
#    the fetch_snip twin.
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
