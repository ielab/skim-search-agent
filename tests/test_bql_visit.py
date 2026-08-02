"""NEW, ADDITIVE-only condition: `research_bql_visit` — the last missing cell of the
SEARCH x READ factorial (SEARCH {bm25, dense, BQL, Indri, Indri+dense} x READ {visit, fetch}):
{BQL search} x {whole-doc visit}. `research`/`research_v2` already fill {BQL search} x
{structure->parts read}; `research_bm25`/`research_dense`/`research_indri_visit` fill
{bm25/dense/indri search} x {whole-doc visit} — this is the ONLY remaining cross.

`BqlVisitWorkspace` (agent_search/agent/tools/doc_research.py) is the SAME BQL v2 search
`research_v2` uses (`DocSearchFetch(coverage=True, date_nudge=True)`), forced with
`snippets=True` for fairness parity with the other visit cells' content-bearing listings
(the same amendment `IndriVisitWorkspace` makes over `IndriFetchWorkspace`), paired with a
whole-doc VISIT read copied verbatim from `Bm25Visit.visit`/`_resolve`. No fetch/structure
tools are exposed.

CPU-only; a small (~25-doc) synthetic corpus with dates/sections, mirroring
tests/test_new_conditions.py's fixture so the coverage/date-nudge assertions are directly
comparable to research_v2's own tests. The existing `research_v2`/`research_indri_visit`
conditions must be byte-identical after this change (no shared code was touched, only new
siblings added) — spot-checked below.
"""
from __future__ import annotations

import pytest

from agent_search.agent.tools import doc_research
from agent_search.agent.tools.doc_research import BqlVisitWorkspace, DocSearchFetch
from agent_search.corpus.units import CodeUnit
from agent_search.prompts import load_condition
from agent_search.retrievers.registry import RetrieverConfig, build_factory


def _mk(doc_id: str, body: str, title: str | None = None, date: str | None = None) -> CodeUnit:
    meta = {"date": date} if date is not None else {}
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=body, body=body, title=title, metadata=meta)


def _corpus() -> list[CodeUnit]:
    units = [
        # --- constraint-coverage fixture: AND(foo, bar, baz, qux) 0-hits exactly (SAME shape
        # as test_new_conditions.py's, so research_v2's own coverage semantics carry over) ---
        _mk("docA", "## History\nfoo bar baz text here", title="Doc A", date="2001-01-01"),
        _mk("docB", "## History\nfoo bar text", title="Doc B", date="2001-01-02"),
        _mk("docC", "## History\nnothing relevant here", title="Doc C", date="2001-01-03"),
        _mk("docD", "## History\nfoo only here", title="Doc D", date="2001-01-04"),
        # --- plain search + excerpt fixture ---
        _mk("d_harbor", "Harbor Festival is an annual event.\n\n## History\n"
                        "Founded in 1897 by A. Smith.\n\n## Legacy\nStill held today.",
            title="Harbor Festival"),
        _mk("d_flat", "The Adams-Onis Treaty of 1819 concerned Florida.",
            title="Adams-Onis Treaty"),
        _mk("65405", "A document whose id is an integer string.", title="Integer Id Doc"),
    ]
    filler_vocab = ["widget", "gadget", "sprocket", "lever", "cog", "gear", "pulley",
                    "spring", "bolt", "nut", "washer", "hinge", "clamp", "bracket", "rivet",
                    "screw", "plank", "beam", "girder"]
    for i, w in enumerate(filler_vocab):
        units.append(_mk(f"filler{i}", f"## History\n{w} placeholder text", title=f"Filler {i}",
                         date=f"200{i % 10}-01-01"))
    return units          # 7 + 19 = 26 docs, >= the ~25 asked for


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


def _ws(units, **kw) -> BqlVisitWorkspace:
    return BqlVisitWorkspace(units, **kw)


# --- 1. shape / inheritance --------------------------------------------------------------

def test_tools_tuple_is_search_bv_and_visit_bv():
    assert BqlVisitWorkspace.tools == ("search_bv", "visit_bv")


def test_is_a_docsearchfetch_subclass_forcing_coverage_datenudge_snippets(units):
    ws = _ws(units)
    assert isinstance(ws, DocSearchFetch)
    assert ws.coverage is True
    assert ws.date_nudge is True
    assert ws.snippets is True                    # forced — not a caller knob


def test_no_fetch_or_structure_tools_available(units):
    ws = _ws(units)
    ws.run("search_bv", {"query": "harbor[title]"})
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert out.startswith("ERROR: unknown tool")


# --- 2. search_bv: BQL-ranked hits, WITH a per-hit content excerpt -----------------------

def test_search_bv_returns_ranked_hits_with_structure_and_excerpt(units):
    ws = _ws(units)
    out = ws.run("search_bv", {"query": "harbor[title]"})
    assert "ERROR" not in out
    assert "d_harbor" in out
    assert "§[" in out and "ib[" in out            # SAME structure table as research_v2
    assert "matched:" in out
    assert "»" in out                              # content excerpt, forced (fairness parity)


def test_search_bv_excerpt_overlaps_query_terms(units):
    ws = _ws(units)
    out = ws.run("search_bv", {"query": "founded[body]"})
    hit_line = next(l for l in out.splitlines() if "d_harbor" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "founded" in excerpt.lower() or "1897" in excerpt


def test_search_bv_marks_hits_seen(units):
    ws = _ws(units)
    ws.run("search_bv", {"query": "harbor[title]"})
    assert "d_harbor" in ws.seen


# --- 3. constraint-coverage ranking: the SAME BQL v2 executor research_v2 uses -----------

def test_search_bv_coverage_topk_renders_cov_and_miss_on_0_hit_and(units):
    ws = _ws(units)
    out = ws.run("search_bv", {"query": "foo[body] AND bar[body] AND baz[body] AND qux[body]"})
    assert "CONSTRAINT COVERAGE" in out
    assert "miss=[" in out
    lines = out.split("\n")
    # docA matches foo/bar/baz (misses qux) -> the strongest coverage hit, ranked first
    assert lines[1].split()[1] == "docA"
    assert "cov=3/4" in lines[1]
    assert "qux" in lines[1]


# --- 4. date-nudge: the SAME mechanical, corpus-free hint research_v2 emits -------------

_DATE_HINT = "hint: temporal clues match documents by METADATA date"


def test_search_bv_date_nudge_hints_on_bare_temporal_clue_and_caps_at_3(units):
    ws = _ws(units)

    out1 = ws.run("search_bv", {"query": "founded 2018 ceremony"})     # bare year -> nudge #1
    assert _DATE_HINT in out1

    out2 = ws.run("search_bv", {"query": "date[2018] ceremony"})       # inside date[...] -> none
    assert _DATE_HINT not in out2

    out3 = ws.run("search_bv", {"query": "released 1985 archive"})     # bare year -> nudge #2
    assert _DATE_HINT in out3

    out4 = ws.run("search_bv", {"query": "since 1990 archive"})        # bare year -> nudge #3
    assert _DATE_HINT in out4

    out5 = ws.run("search_bv", {"query": "circa 1975 archive"})        # 4th eligible -> cap reached
    assert _DATE_HINT not in out5


# --- 5. visit_bv: whole-doc read, mirroring Bm25Visit.visit/_resolve exactly ------------

def test_visit_bv_by_rank_returns_full_body_and_marks_seen(units):
    ws = _ws(units)
    ws.run("search_bv", {"query": "harbor[title]"})
    out = ws.run("visit_bv", {"rank": 1})
    assert "ERROR" not in out
    assert "Founded in 1897" in out and "Still held today" in out     # WHOLE doc, not a section
    assert ws.last_hits[0] in ws.seen


def test_visit_bv_by_doc_id_works(units):
    ws = _ws(units)
    ws.run("search_bv", {"query": "harbor[title]"})
    out = ws.run("visit_bv", {"rank": "d_harbor"})
    assert "Founded in 1897" in out
    assert "d_harbor" in ws.seen


def test_visit_bv_bad_rank_errors(units):
    ws = _ws(units)
    ws.run("search_bv", {"query": "harbor[title]"})
    out = ws.run("visit_bv", {"rank": 999})
    assert out.startswith("ERROR:")
    assert "out of range" in out


def test_visit_bv_no_prior_search_errors(units):
    ws = _ws(units)
    out = ws.run("visit_bv", {"rank": 1})
    assert out.startswith("ERROR:")
    assert "no prior search" in out


def test_visit_bv_integer_doc_id_not_mistaken_for_rank(units):
    ws = _ws(units)
    ws.run("search_bv", {"query": "integer[body]"})
    out = ws.run("visit_bv", {"rank": "65405"})
    assert "integer string" in out and "out of range" not in out


def test_visit_bv_capped_at_max_visit_tokens(units, monkeypatch):
    """The whole-doc read is capped at MAX_VISIT_TOKENS, same knob/default as Bm25Visit.visit
    (see doc_research.py module docstring / test_configs.py's env_knobs default of 1200).
    Monkeypatch the module-level constant (read live at call time, same pattern
    test_configs.py documents) down to a small value to exercise the cap deterministically."""
    monkeypatch.setattr(doc_research, "MAX_VISIT_TOKENS", 3)
    long_body = " ".join(f"word{i}" for i in range(50))
    long_units = list(units) + [_mk("d_long", long_body, title="Long Doc")]
    ws = _ws(long_units)
    ws.run("search_bv", {"query": "word0[body]"})
    out = ws.run("visit_bv", {"rank": "d_long"})
    assert "…(truncated — this is the whole-doc cap)" in out
    assert "word49" not in out                     # truncated well before the doc's end


def test_visit_bv_under_cap_is_not_truncated(units, monkeypatch):
    monkeypatch.setattr(doc_research, "MAX_VISIT_TOKENS", 1200)
    ws = _ws(units)
    ws.run("search_bv", {"query": "harbor[title]"})
    out = ws.run("visit_bv", {"rank": 1})
    assert "truncated" not in out


# --- 6. run() dispatch: search/search_v2/search_bv, visit/visit_bv aliases --------------

def test_run_aliases_search_names(units):
    out1 = _ws(units).run("search_bv", {"query": "harbor[title]"})
    out2 = _ws(units).run("search", {"query": "harbor[title]"})
    out3 = _ws(units).run("search_v2", {"query": "harbor[title]"})
    assert out1 == out2 == out3


def test_run_aliases_visit_name(units):
    ws1 = _ws(units)
    ws1.run("search_bv", {"query": "harbor[title]"})
    out1 = ws1.run("visit_bv", {"rank": 1})
    ws2 = _ws(units)
    ws2.run("search_bv", {"query": "harbor[title]"})
    out2 = ws2.run("visit", {"rank": 1})
    assert out1 == out2


def test_run_unknown_tool_errors(units):
    out = _ws(units).run("bogus_tool", {})
    assert "unknown tool" in out.lower()


def test_empty_query_message(units):
    ws = _ws(units)
    assert ws.run("search_bv", {"query": ""}) == "empty query"


# --- 7. condition loading + retriever factory --------------------------------------------

def test_research_bql_visit_condition_loads():
    p = load_condition("research_bql_visit")
    assert p.toolset == "bql_visit"
    assert p.tool_names == ("search_bv", "visit_bv")
    # the SAME BQL v2 skill research_v2 teaches (no new manual forked)
    assert "date[1980..1989]" in p.system
    assert "CONSTRAINT COVERAGE" in p.system


def test_existing_research_v2_condition_is_unaffected():
    p = load_condition("research_v2")
    assert p.toolset == "search_fetch_v2"
    assert p.tool_names == ("search_v2", "fetch_v2")


def test_existing_research_indri_visit_condition_is_unaffected():
    p = load_condition("research_indri_visit")
    assert p.toolset == "indri_visit"
    assert p.tool_names == ("isearch_v", "visit_v")


def test_research_bql_visit_resolves_via_registry():
    from agent_search.agent.retriever import AgentRetriever

    r = build_factory("agent_research_bql_visit", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("search_bv", "visit_bv")
    assert r.tool == "agent_research_bql_visit"
    assert r._arm == "bqlvisit"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_bql_visit_workspace_builds_with_forced_flags(units, tmp_path):
    from agent_search.agent.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_bql_visit", cfg)()
    r.index(units, key="test-bql-visit-corpus")
    ws = r._workspace(5, "founded 2018 ceremony")
    assert isinstance(ws, BqlVisitWorkspace)
    assert ws.coverage is True
    assert ws.date_nudge is True
    assert ws.snippets is True


def test_research_bql_visit_workspace_builds_and_answers_via_stub(units, tmp_path):
    from agent_search.agent.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_bql_visit", cfg)()
    r.index(units, key="test-bql-visit-corpus")
    ranking = r.search("harbor festival history", k=5)
    assert isinstance(ranking, list)


# --- 8. offline end-to-end smoke: one fixture instance through run_config (no model, no vLLM) --

def test_research_bql_visit_smoke_via_fixture_dataset(tmp_path):
    """ONE instance end-to-end through the SAME offline fixture harness
    test_configs.py's test_run_config_fixture uses for agent_codefix — here for the doc arm,
    via the `browsecomp_plus_fixture` 1-instance/3-doc dataset (evaluation/datasets.py's
    `_doc_corpus_fixture`) and the no-model `stub` policy (KeywordPolicy). No vLLM, no API."""
    from evaluation.config import DatasetArgs, EvaluationArgs, OutputArgs, RetrieverArgs, RunConfig
    from evaluation.run_eval import run_config

    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_bql_visit", index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )

    res = run_config(cfg, progress=False)

    assert res["n"] == 1
    assert res["n_errors"] == 0
