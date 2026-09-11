"""`research_dense`: the dense retrieve-then-visit baseline (`search_dense` + `visit`, formerly
`DenseVisit`), a byte-identical clone of the plain `search_bm25` listing (search renders rank,
doc_id, title, snippet; visit returns the whole capped doc) with a dense embedding engine (a
DenseBelief) swapped in for BM25. CPU-only here via a stub engine; no torch/sentence-transformers
import.
"""
from __future__ import annotations

import pytest

from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_dense.tool import SearchDense
from agent_search.tools.visit.tool import Visit

# the SAME fixture doc set the sieve/bm25-fetch tool tests use, so search_dense's rendering can
# be compared line-for-line against search_bm25's for the same hits.
DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today.",
     "infobox": "Founded: 1897; Location: Portville"},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    {"_id": "65405", "title": "Integer Id Doc",
     "text": "A document whose id is an integer string."},
]


def _units():
    return units_from_documents(DOCS)


class _StubDenseEngine:
    """A CPU-only stand-in for DenseBelief exposing ONLY the API search_dense calls
    (`top_k_doc_ids(query, k)` -> ranked doc_ids) — no torch/sentence-transformers import, no
    GPU, no persisted cache. `ranking` is a fixed doc_id list (or a {query: [...]} map for
    query-dependent tests)."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def top_k_doc_ids(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


def _toolbox(ranking=("d_harbor", "d_flat")):
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchDense(name="dense_search").bind(state, units, ubyid, {"dense": _StubDenseEngine(ranking)})
    visit = Visit(name="visit_d").bind(state, units, ubyid, {})
    return ToolBox([search, visit], state)


# --- tools/dispatch -----------------------------------------------------------------------

def test_tools_tuple_is_dense_search_and_visit_d():
    assert _toolbox().tools == ("dense_search", "visit_d")


# --- search: ranked + opening snippet, SAME rendering as the plain bm25 listing -------------

def test_search_renders_rank_doc_id_title_and_snippet():
    box = _toolbox(("d_harbor",))
    out = box.run("dense_search", {"query": "harbor festival", "k": 5})
    assert "search: harbor festival   (1 matches):" in out
    assert "1  d_harbor  'Harbor Festival'" in out
    assert "Harbor Festival is an annual event" in out          # opening snippet, not a section


def test_search_output_matches_plain_bm25_rendering_for_the_same_hit_order():
    """search_dense must render BYTE-IDENTICALLY to the plain search_bm25 listing given the
    SAME ranked doc_ids — the only difference between the two arms is WHICH engine produced the
    ranking, never the observation shape."""
    units = _units()
    ubyid = {u.doc_id: u for u in units}

    dense_state = EpisodeState(question="q")
    dense_search = SearchDense(name="dense_search").bind(
        dense_state, units, ubyid, {"dense": _StubDenseEngine(("d_harbor", "d_flat"))})
    dense_out = dense_search.run({"query": "harbor festival", "k": 5})

    class _FixedBm25Engine:
        def search(self, query, k=5):
            return ["d_harbor", "d_flat"][:k]

    bm25_state = EpisodeState(question="q")
    bm25_search = SearchBm25(name="bm25_search").bind(
        bm25_state, units, ubyid, {"bm25": _FixedBm25Engine()})
    bm25_out = bm25_search.run({"query": "harbor festival", "k": 5})

    assert dense_out == bm25_out


def test_search_marks_hits_seen():
    box = _toolbox(("d_harbor", "d_flat"))
    box.run("dense_search", {"query": "harbor"})
    assert {"d_harbor", "d_flat"} <= set(box.seen)


def test_empty_query_message():
    box = _toolbox()
    assert box.run("dense_search", {"query": ""}) == "empty query"


def test_zero_hits_message():
    box = _toolbox([])
    out = box.run("dense_search", {"query": "nothing matches"})
    assert "0 matches" in out


def test_zero_hits_after_a_prior_search_notes_previous_results_available():
    box = _toolbox({"harbor": ["d_harbor"], "zzz": []})
    box.run("dense_search", {"query": "harbor"})            # establishes last_hits
    out = box.run("dense_search", {"query": "zzz"})
    assert "0 matches" in out and "previous results still available" in out


def test_engine_receives_the_raw_query_and_the_knob_depth():
    """run() passes the RAW query through, with DENSE_VISIT_TOPK as k — the dense_search schema
    exposes ONLY `query`, so a hallucinated `k` in the tool-call args is ignored; the env knob
    alone sets the SERP listing depth."""
    import agent_search.tools.budgets as m
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    engine = _StubDenseEngine(("d_harbor",))
    search = SearchDense(name="dense_search").bind(state, units, ubyid, {"dense": engine})
    search.run({"query": "harbor festival", "k": 3})
    assert engine.calls == [("harbor festival", m.DENSE_VISIT_TOPK)]


# --- run() dispatch: dense_search/search, visit_d/visit aliases ----------------------------

def test_run_aliases_search_name():
    box1 = _toolbox(("d_harbor",))
    out1 = box1.run("dense_search", {"query": "harbor"})
    box2 = _toolbox(("d_harbor",))
    out2 = box2.run("search", {"query": "harbor"})
    assert out1 == out2


def test_run_aliases_visit_name():
    box1 = _toolbox(("d_harbor",))
    box1.run("dense_search", {"query": "harbor"})
    out1 = box1.run("visit_d", {"rank": 1})
    assert "Founded in 1897" in out1
    box2 = _toolbox(("d_harbor",))
    box2.run("dense_search", {"query": "harbor"})
    out2 = box2.run("visit", {"rank": 1})
    assert out1 == out2


def test_visit_returns_whole_doc_not_a_section():
    box = _toolbox(("d_harbor",))
    box.run("dense_search", {"query": "harbor"})
    out = box.run("visit_d", {"rank": 1})
    assert "Founded in 1897" in out and "Still held today" in out


def test_integer_doc_id_not_mistaken_for_rank():
    box = _toolbox(("65405",))
    box.run("dense_search", {"query": "integer"})
    out = box.run("visit_d", {"rank": "65405"})
    assert "integer string" in out and "out of range" not in out


def test_run_unknown_tool_errors():
    box = _toolbox()
    out = box.run("bogus_tool_name", {"specs": [[1, "History"]]})
    assert "unknown tool" in out.lower()


# --- condition wiring: research_dense loads + resolves via the retriever registry ----------

def test_research_dense_condition_loads_uncoached():
    from agent_search.strategies import CONDITIONS
    from agent_search.tasks.render import render_manuals

    p = CONDITIONS["research_dense"]
    assert p.strategy.toolset_name == "dense_visit"
    assert set(p.tool_names) == {"dense_search", "visit_d"}
    # UNCOACHED like research_bm25: no manual renders for this toolset.
    assert render_manuals([t.manual_path("general") for t in p.strategy.tools]) == ""
    assert "term[field]" not in p.render()


def test_research_dense_resolves_via_registry_as_densevisit_arm():
    from agent_search.evaluation.agent_runner import ConditionAgent
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("dense_search", "visit_d")
    assert r.tool == "agent_research_dense"
    assert r.condition.name == "research_dense"
    assert r.domain == "general"
    assert not r.needs_files


def test_densevisit_index_raises_clear_error_when_cache_missing(tmp_path):
    """This baseline needs a persisted dense doc-embedding cache — a missing cache must raise
    a CLEAR error at index() time, never silently fall back to live-encoding the corpus."""
    from agent_search.errors import SetupError
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(SetupError, match="dense embedding cache"):
        r.index(_units(), key="no_such_corpus_key")
