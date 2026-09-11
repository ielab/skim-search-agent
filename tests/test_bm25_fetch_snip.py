"""`research_bm25_fetch_snip`: the content-bearing-listing sibling of `research_bm25_fetch`.
The factorial cell "bm25 search x structure->fetch" (research_bm25_fetch, `Bm25FetchWorkspace`)
has a content-blind search listing (structure only: title, section names, infobox keys, no
excerpt), while every other method fetch cell (research_snip, research_indri_snip,
research_dense_fetch) shows each hit's best-matching excerpt via the shared module-level
`best_line` (common.py). This condition removes that inconsistency across the search x read
factorial grid.

`Bm25FetchSnipWorkspace` (agent_search/legacy/workspaces/search_fetch.py) uses the same bm25
retrieval and the same structured section-fetch tools as `Bm25FetchWorkspace` (construction,
`fetch`, `topk`/`engine`/`query` are inherited unchanged), with only the search listing's
rendering changed to append a one-line query-biased best-matching excerpt per hit, the same
move `DenseFetchWorkspace` makes over `Bm25FetchWorkspace`'s own bare table.

CPU-only; reuses test_doc_bm25_fetch_tools.py's small synthetic corpus (a structured doc, a flat
doc, an off-topic doc) so the ranking-parity assertion is a direct, apples-to-apples comparison
against the existing Bm25FetchWorkspace tests."""
from __future__ import annotations

from agent_search.legacy.workspaces.search_fetch import Bm25FetchSnipWorkspace, Bm25FetchWorkspace
from agent_search.corpus.units import units_from_documents
from agent_search.legacy.prompts import load_condition
from agent_search.retrievers.lexical.bm25 import BM25Local
from agent_search.retrievers.registry import RetrieverConfig, build_factory

# SAME fixture docs as test_doc_bm25_fetch_tools.py (a structured doc, a flat doc, an off-topic
# doc bm25 never scores) — keeps the ranking-parity comparison directly comparable.
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


def _engine():
    return BM25Local().index(_units())


def _ws(query="harbor festival annual event history", topk=5, engine=None):
    return Bm25FetchSnipWorkspace(_units(), query, engine=engine or _engine(), topk=topk)


# --- 1. shape / inheritance --------------------------------------------------------------

def test_tools_tuple_is_bm25_search_snip_and_fetch():
    assert Bm25FetchSnipWorkspace.tools == ("bm25_search_snip", "fetch")


def test_is_a_bm25fetchworkspace_subclass():
    ws = _ws()
    assert isinstance(ws, Bm25FetchWorkspace)


# --- 2. ranking parity: IDENTICAL top-k doc ids to Bm25FetchWorkspace --------------------

def test_ranking_identical_to_bm25fetchworkspace_for_same_query_engine_k():
    """The whole point of the fair-listing sibling: retrieval must be byte-identical to
    research_bm25_fetch's for the same query/engine/k — ONLY the listing's content differs."""
    engine = _engine()
    query = "harbor festival annual event history"
    snip_ws = Bm25FetchSnipWorkspace(_units(), query, engine=engine, topk=3)
    snip_ws.run("bm25_search_snip", {"query": query, "k": 3})
    plain_ws = Bm25FetchWorkspace(_units(), query, engine=engine, topk=3)
    plain_ws.run("bm25_search", {"query": query, "k": 3})
    assert snip_ws.last_hits == plain_ws.last_hits


def test_ranking_identical_across_multiple_queries():
    engine = _engine()
    for query in ("harbor festival", "quantum chromodynamics particle physics", "history founded"):
        snip_ws = Bm25FetchSnipWorkspace(_units(), query, engine=engine, topk=5)
        snip_ws.run("bm25_search_snip", {"query": query})
        plain_ws = Bm25FetchWorkspace(_units(), query, engine=engine, topk=5)
        plain_ws.run("bm25_search", {"query": query})
        assert snip_ws.last_hits == plain_ws.last_hits


# --- 3. search: structure table (SAME as Bm25FetchWorkspace) + a per-hit excerpt ---------

def test_search_lists_sections_and_infobox_no_body_and_an_excerpt():
    ws = _ws(topk=5)
    out = ws.run("bm25_search_snip", {"query": "harbor festival"})
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out       # section names shown
    assert "Founded" in out                           # infobox key shown
    assert "»" in out                                 # NEW: content excerpt (fairness parity)


def test_excerpt_overlaps_query_terms():
    ws = _ws(topk=5)
    out = ws.run("bm25_search_snip", {"query": "founded 1897"})
    hit_line = next(l for l in out.splitlines() if "d_harbor" in l)
    assert "»" in hit_line
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "founded" in excerpt.lower() or "1897" in excerpt


def test_search_is_live_and_re_retrieves():
    ws = _ws(query="harbor festival history", topk=5)
    ws.run("bm25_search_snip", {"query": "harbor festival history"})
    first = list(ws.last_hits)
    ws.run("bm25_search_snip", {"query": "quantum chromodynamics particle physics"})
    assert ws.last_hits != first
    assert "d_outside" in ws.last_hits


def test_topk_marks_hits_as_seen_immediately():
    ws = _ws(topk=5)
    assert ws.last_hits == []
    ws.run("bm25_search_snip", {"query": "harbor festival annual event history"})
    assert ws.last_hits
    assert set(ws.last_hits) <= set(ws.seen)


def test_empty_query_yields_no_hits():
    ws = _ws(query="", topk=5)
    assert ws.last_hits == []
    assert ws.search("") == "empty query"


# --- 4. run() dispatch: aliases work -------------------------------------------------------

def test_run_aliases_search_names():
    engine = _engine()
    query = "harbor festival"
    out1 = Bm25FetchSnipWorkspace(_units(), query, engine=engine, topk=5).run(
        "bm25_search_snip", {"query": query})
    out2 = Bm25FetchSnipWorkspace(_units(), query, engine=engine, topk=5).run(
        "bm25_search", {"query": query})
    out3 = Bm25FetchSnipWorkspace(_units(), query, engine=engine, topk=5).run(
        "search", {"query": query})
    assert out1 == out2 == out3


def test_run_unknown_tool_errors():
    out = _ws(topk=5).run("bogus_tool", {})
    assert "unknown tool" in out.lower()


# --- 6. condition loading + retriever factory ---------------------------------------------

def test_research_bm25_fetch_snip_condition_loads():
    p = load_condition("research_bm25_fetch_snip")
    assert p.toolset == "bm25_fetch_snip"
    assert p.tool_names == ("bm25_search_snip", "fetch")


def test_existing_research_dense_fetch_condition_is_unaffected():
    p = load_condition("research_dense_fetch")
    assert p.toolset == "dense_fetch"
    assert p.tool_names == ("dense_search_f", "fetch")


def test_research_bm25_fetch_snip_resolves_via_registry():
    from agent_search.evaluation.agent_runner import ConditionAgent

    r = build_factory("agent_research_bm25_fetch_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("bm25_search_snip", "fetch")
    assert r.tool == "agent_research_bm25_fetch_snip"
    assert r.condition.name == "research_bm25_fetch_snip"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_bm25_fetch_snip_workspace_builds(tmp_path):
    from agent_search.tools.search_bm25.tool import SearchBm25

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_bm25_fetch_snip", cfg)()
    r.index(_units(), key="test-bm25-fetch-snip-corpus")
    ws = r.toolbox("harbor festival annual event history")
    assert isinstance(ws["bm25_search_snip"], SearchBm25)


def test_research_bm25_fetch_snip_workspace_answers_via_stub(tmp_path):
    from agent_search.legacy.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_bm25_fetch_snip", cfg)()
    r.index(_units(), key="test-bm25-fetch-snip-corpus")
    ranking = r.search("harbor festival annual event history", k=5)
    assert isinstance(ranking, list)


# --- 7. offline end-to-end smoke: one fixture instance through run_config (no model, no vLLM) --

def test_research_bm25_fetch_snip_smoke_via_fixture_dataset(tmp_path):
    """ONE instance end-to-end through the SAME offline fixture harness test_bql_visit.py's
    test_research_bql_visit_smoke_via_fixture_dataset uses — here for research_bm25_fetch_snip,
    via the `browsecomp_plus_fixture` 1-instance/3-doc dataset and the no-model `stub` policy
    (KeywordPolicy). No vLLM, no API."""
    from agent_search.evaluation.config import DatasetArgs, EvaluationArgs, OutputArgs, RetrieverArgs, RunConfig
    from agent_search.evaluation.run_eval import run_config

    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_bm25_fetch_snip", index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )

    res = run_config(cfg, progress=False)

    assert res["n"] == 1
    assert res["n_errors"] == 0
