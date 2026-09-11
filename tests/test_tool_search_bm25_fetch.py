"""The retrieval-bounded structured-read tool pair: `search_bm25` (`structure=True`) then
`fetch` (bm25 top-k retrieval, then fetch a named section — `SearchBm25`'s read side on
`Bm25Visit`'s retrieval, the current tool.py implementation of what used to be
`Bm25FetchWorkspace`).

It is the sieve's read (search lists structure, section names and infobox keys, no bodies;
fetch pulls one named section) on plain BM25 retrieval, so the only difference from
`research` is the retrieval stage, and the only difference from `research_bm25` is the read.
Retrieval is live per bm25_search call, with retrieval behavior identical to `search_bm25`
with `structure=False` (the search_visit family) for the same queries; the section-fetch
machinery is shared with the sieve's `fetch` tool, not duplicated. Completes the controlled
set with research_bm25 (whole-doc visit) and research_bm25_dci (bash/read shell) over
identical bm25 retrieval.
"""
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lexical.bm25 import BM25Local
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bm25.tool import SearchBm25

# a structured doc (## markers -> sections + infobox), a flat doc, and an off-topic doc that a
# harbor query never scores (bm25's score>0 filter) — the doc a fetch must not be able to reach.
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

# ~30 docs, each built around ONE distinctive topic word repeated for a clear BM25 signal, so a
# query on that word ranks its doc unambiguously first. d1/d2/d3 additionally carry `##`-style
# structured sections (History/Career) to exercise the structure-table rendering.
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
          "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
          "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu", "apple",
          "banana", "cherry", "dolphin"]


def _word_doc(i: int, word: str) -> dict:
    d = {"_id": f"d{i + 1}", "title": f"{word.capitalize()} Report",
         "text": f"{word} is the central subject of this report. {word} details: {word} "
                 f"background and {word} facts, repeated for emphasis: {word} {word}."}
    if i < 3:                                      # d1, d2, d3 get real sections
        d["sections"] = [
            {"heading": "History", "text": f"{word} has a long documented history."},
            {"heading": "Career", "text": f"{word}'s career spans several notable events."},
        ]
    return d


WORD_DOCS = [_word_doc(i, w) for i, w in enumerate(_WORDS)]


def _units():
    return units_from_documents(DOCS)


def _word_units():
    return units_from_documents(WORD_DOCS)


def _engine(units=None):
    return BM25Local().index(units if units is not None else _units())


def _toolbox(units, engine, topk=None, question="q"):
    state = EpisodeState(question=question)
    opts = {"structure": True}
    if topk is not None:
        opts["k"] = topk
    ubyid = {u.doc_id: u for u in units}
    search = SearchBm25(name="bm25_search", **opts).bind(state, units, ubyid, {"bm25": engine})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([search, fetch], state)


def _ws(query="harbor festival annual event history", topk=5, engine=None):
    box = _toolbox(_units(), engine or _engine(), topk=topk)
    if query:
        box.run("bm25_search", {"query": query, "k": topk})
    return box


# --- retrieval: bm25 top-k, byte-identical to the plain search_bm25 listing -----------------

def test_retrieval_matches_plain_search_bm25_for_same_engine_and_query():
    """Retrieval must be byte-identical to research_bm25's for the same query — the whole point
    is that ONLY the read strategy (section-fetch vs whole-doc visit) differs between the arms.
    Both arms retrieve LIVE per search call over the same engine."""
    from agent_search.tools.visit.tool import Visit

    units = _units()
    engine = _engine(units)
    query = "harbor festival annual event history"

    fetch_box = _toolbox(units, engine, topk=3)
    fetch_box.run("bm25_search", {"query": query, "k": 3})

    state = EpisodeState(question="q")
    ubyid = {u.doc_id: u for u in units}
    visit_search = SearchBm25(name="bm25_search").bind(state, units, ubyid, {"bm25": engine})
    Visit(name="visit").bind(state, units, ubyid, {})
    visit_search.run({"query": query, "k": 3})

    assert fetch_box.last_hits == state.last_hits


def test_topk_marks_hits_as_seen_immediately():
    # LIVE retrieval: nothing retrieved before the first call; a search's hits are seen the
    # moment they are ranked (same contract as the plain search_bm25 listing).
    box = _toolbox(_units(), _engine(), topk=5)
    assert box.last_hits == []
    box.run("bm25_search", {"query": "harbor festival annual event history"})
    assert box.last_hits                              # harbor query matches >=1 doc
    assert set(box.last_hits) <= set(box.seen)


def test_empty_query_yields_no_hits():
    box = _toolbox(_units(), _engine(), topk=5)
    assert box.last_hits == []
    # a blank query is rejected as "empty query", not rendered as a 0-hit ranking (there was
    # no retrieval to render).
    assert box.run("bm25_search", {"query": ""}) == "empty query"


# --- search: LISTS structure (sections + infobox keys), NO bodies ---------------------------

def test_search_lists_sections_and_infobox_no_body():
    box = _toolbox(_units(), _engine(), topk=5)
    out = box.run("bm25_search", {"query": "harbor festival"})
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out       # section names shown
    assert "Founded" in out                           # infobox key shown
    assert "A. Smith" not in out                      # NOT the body content
    assert "1897" not in out                          # infobox VALUE not shown (keys only)


def test_search_is_live_and_re_retrieves():
    """LIVE retrieval: a NEW query string in a later bm25_search call re-runs bm25 and CHANGES
    the ranking — search does not replay a ranking fixed at construction from the raw episode
    question while ignoring the agent's actual query."""
    box = _toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search", {"query": "harbor festival history"})
    first = list(box.last_hits)
    box.run("bm25_search", {"query": "quantum chromodynamics particle physics"})
    assert box.last_hits != first                      # a different query re-retrieves
    assert "d_outside" in box.last_hits                 # ...and can surface docs the first missed


# --- fetch: pull ONE named section, shared with the plain search listing --------------------

def test_fetch_named_section_by_rank():
    box = _toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search", {"query": "harbor festival"})   # establishes ranking (rank 1 = d_harbor)
    out = box.run("fetch", {"specs": [[1, "History"]]})
    assert "Founded in 1897" in out and "History" in out


def test_fetch_infobox():
    box = _toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search", {"query": "harbor festival"})
    out = box.run("fetch", {"specs": [[1, "infobox"]]})
    assert "Founded=1897" in out


def test_fetch_bad_section_lists_available():
    box = _toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search", {"query": "harbor festival"})
    out = box.run("fetch", {"specs": [[1, "Nonexistent"]]})
    assert "no section" in out and "History" in out       # a section NOT on the hit isn't fetchable


def test_fetch_marks_doc_seen():
    box = _toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search", {"query": "harbor festival"})
    box.run("fetch", {"specs": [[1, "History"]]})
    assert "d_harbor" in box.seen


def test_fetch_reads_from_the_latest_ranking():
    box = _toolbox(_word_units(), _engine(_word_units()), topk=5)
    box.run("bm25_search", {"query": "alpha"})
    box.run("bm25_search", {"query": "bravo"})          # now ranked on bravo
    out = box.run("fetch", {"specs": [[1, ""]]})
    assert "d2" in out
    assert "History" not in out or "documented history" in out  # d2 is flat: intro text only


def test_seen_accumulates_across_both_searches():
    box = _toolbox(_word_units(), _engine(_word_units()), topk=5)
    box.run("bm25_search", {"query": "alpha"})
    box.run("bm25_search", {"query": "bravo"})
    assert "d1" in box.seen and "d2" in box.seen


def test_off_topic_doc_is_not_in_the_ranking():
    """d_outside shares no vocabulary with the harbor query, so bm25 never ranks it (score>0
    filter) — it cannot be fetched by rank, and its content never appears."""
    box = _toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search", {"query": "harbor festival annual event"})
    assert "d_outside" not in box.last_hits
    listing = box.run("bm25_search", {"query": "harbor festival"})
    assert "Quantum Chromodynamics" not in listing


def test_zero_hit_bm25_search_keeps_generic_hint():
    box = _toolbox(_word_units(), _engine(_word_units()), topk=5)
    out = box.run("bm25_search", {"query": "zzzznonexistentterm"})
    assert "0 matches" in out
    assert "hint: loosen the query" in out


# --- run() dispatch ---------------------------------------------------------------------------

def test_run_unknown_tool_errors():
    box = _toolbox(_units(), _engine(), topk=5)
    out = box.run("visit", {"rank": 1})           # visit is not a tool of this arm
    assert "unknown tool" in out.lower()


def test_tools_tuple_matches_toolset():
    box = _toolbox(_units(), _engine(), topk=5)
    assert box.tools == ("bm25_search", "fetch")


# =============================================================================================
# research_bm25_fetch_snip: the content-bearing-listing sibling of the plain fetch above.
#
# The plain cell ("bm25 search x structure->fetch") has a content-blind search listing
# (structure only: title, section names, infobox keys, no excerpt), while every other method
# fetch cell (research_snip, research_indri_snip, research_dense_fetch) shows each hit's
# best-matching excerpt via the shared module-level `best_line` (agent_search/tools/common.py).
# This condition removes that inconsistency across the search x read factorial grid: SAME bm25
# retrieval and SAME structured section-fetch tool as the plain cell above (`snippets=True` is
# the only difference), the same move `SearchDense(structure=True, snippets=True)` makes over
# the plain `dense_search_fp` cell.
#
# Reuses this file's DOCS/units fixture (a structured doc, a flat doc, an off-topic doc) so the
# ranking-parity assertion below is a direct, apples-to-apples comparison against the plain cell.
# =============================================================================================

def _snip_toolbox(units, engine, topk=None):
    state = EpisodeState(question="q")
    opts = {"structure": True, "snippets": True}
    if topk is not None:
        opts["k"] = topk
    ubyid = {u.doc_id: u for u in units}
    search = SearchBm25(name="bm25_search_snip", **opts).bind(state, units, ubyid, {"bm25": engine})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([search, fetch], state)


def test_snip_tools_tuple_is_bm25_search_snip_and_fetch():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    assert box.tools == ("bm25_search_snip", "fetch")


def test_snip_ranking_identical_to_the_plain_cell_for_same_query_engine_k():
    """The whole point of the fair-listing sibling: retrieval must be byte-identical to the
    plain fetch cell's for the same query/engine/k — ONLY the listing's content differs."""
    engine = _engine()
    query = "harbor festival annual event history"
    snip_box = _snip_toolbox(_units(), engine, topk=3)
    snip_box.run("bm25_search_snip", {"query": query, "k": 3})
    plain_box = _toolbox(_units(), engine, topk=3)
    plain_box.run("bm25_search", {"query": query, "k": 3})
    assert snip_box.last_hits == plain_box.last_hits


def test_snip_ranking_identical_across_multiple_queries():
    engine = _engine()
    for query in ("harbor festival", "quantum chromodynamics particle physics", "history founded"):
        snip_box = _snip_toolbox(_units(), engine, topk=5)
        snip_box.run("bm25_search_snip", {"query": query})
        plain_box = _toolbox(_units(), engine, topk=5)
        plain_box.run("bm25_search", {"query": query})
        assert snip_box.last_hits == plain_box.last_hits


def test_snip_search_lists_sections_and_infobox_no_body_and_an_excerpt():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    out = box.run("bm25_search_snip", {"query": "harbor festival"})
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out       # section names shown
    assert "Founded" in out                           # infobox key shown
    assert "»" in out                                 # the content excerpt (fairness parity)


def test_snip_excerpt_overlaps_query_terms():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    out = box.run("bm25_search_snip", {"query": "founded 1897"})
    hit_line = next(l for l in out.splitlines() if "d_harbor" in l)
    assert "»" in hit_line
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "founded" in excerpt.lower() or "1897" in excerpt


def test_snip_search_is_live_and_re_retrieves():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    box.run("bm25_search_snip", {"query": "harbor festival history"})
    first = list(box.last_hits)
    box.run("bm25_search_snip", {"query": "quantum chromodynamics particle physics"})
    assert box.last_hits != first
    assert "d_outside" in box.last_hits


def test_snip_topk_marks_hits_as_seen_immediately():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    assert box.last_hits == []
    box.run("bm25_search_snip", {"query": "harbor festival annual event history"})
    assert box.last_hits
    assert set(box.last_hits) <= set(box.seen)


def test_snip_empty_query_yields_no_hits():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    assert box.last_hits == []
    assert box.run("bm25_search_snip", {"query": ""}) == "empty query"


def test_snip_run_aliases_search_names():
    engine = _engine()
    query = "harbor festival"
    out1 = _snip_toolbox(_units(), engine, topk=5).run("bm25_search_snip", {"query": query})
    out2 = _snip_toolbox(_units(), engine, topk=5).run("bm25_search", {"query": query})
    out3 = _snip_toolbox(_units(), engine, topk=5).run("search", {"query": query})
    assert out1 == out2 == out3


def test_snip_run_unknown_tool_errors():
    box = _snip_toolbox(_units(), _engine(), topk=5)
    out = box.run("bogus_tool", {})
    assert "unknown tool" in out.lower()


def test_research_bm25_fetch_snip_condition_loads():
    from agent_search.strategies import CONDITIONS

    p = CONDITIONS["research_bm25_fetch_snip"]
    assert p.strategy.toolset_name == "bm25_fetch_snip"
    assert p.tool_names == ("bm25_search_snip", "fetch")


def test_existing_research_dense_fetch_condition_is_unaffected():
    from agent_search.strategies import CONDITIONS

    p = CONDITIONS["research_dense_fetch"]
    assert p.strategy.toolset_name == "dense_fetch"
    assert p.tool_names == ("dense_search_f", "fetch")


def test_research_bm25_fetch_snip_resolves_via_registry():
    from agent_search.evaluation.agent_runner import ConditionAgent
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bm25_fetch_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("bm25_search_snip", "fetch")
    assert r.tool == "agent_research_bm25_fetch_snip"
    assert r.condition.name == "research_bm25_fetch_snip"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_bm25_fetch_snip_workspace_builds(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_bm25_fetch_snip", cfg)()
    r.index(_units(), key="test-bm25-fetch-snip-corpus")
    ws = r.toolbox("harbor festival annual event history")
    assert isinstance(ws["bm25_search_snip"], SearchBm25)


def test_research_bm25_fetch_snip_workspace_answers_via_stub(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_bm25_fetch_snip", cfg)()
    r.index(_units(), key="test-bm25-fetch-snip-corpus")
    ranking = r.search("harbor festival annual event history", k=5)
    assert isinstance(ranking, list)


def test_research_bm25_fetch_snip_smoke_via_fixture_dataset(tmp_path):
    """ONE instance end-to-end through run_config's offline fixture harness, via the
    `browsecomp_plus_fixture` 1-instance/3-doc dataset and the no-model `stub` policy
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
