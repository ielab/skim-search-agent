"""NEW, additive-only PLAIN dense fetch cell: `research_dense_fetch_plain`
(agent_search.agent.tools.doc_research.DenseFetchPlainWorkspace).

Context: dense was the ONLY query engine among bm25/bql/indri/dense lacking a plain (no-excerpt)
fetch cell — `DenseFetchWorkspace` (research_dense_fetch, "dense+snip fetch") has no `snippets`
flag at all; its `search()` appends a `» excerpt` line unconditionally. This condition supplies
the missing plain sibling, mirroring how `research_bql_dense_fetch` supplies
`research_bql_dense_snip`'s missing plain sibling for the dense-fused BQL executor (see
tests/test_bql_dense.py's section 10 for that pattern, mirrored here tool-for-tool).

  1. DenseFetchPlainWorkspace: tools/dispatch, plain listing (no `»`), fetch delegation, run()
     aliases, hallucinated-tool errors — CPU-only via a stub dense engine (no torch import),
     exactly like test_dense_baseline.py's DenseFetchWorkspace section.
  2. Side-by-side: SAME stub engine/query -> `research_dense_fetch_plain`'s listing and
     `research_dense_fetch`'s listing differ ONLY by the `»` excerpt line.
  3. Condition loads/resolves via the retriever registry as the 'densefetchplain' arm.
  4. index() raises a clear RuntimeError when the dense doc-embedding cache is missing (SAME
     fail-loud contract as 'densevisit'/'densefetch').
  5. Full AgentRetriever offline smoke (fake dense stack, no torch/network — mirrors
     tests/test_hybrid.py's `fake_dense_stack` fixture) confirming `_workspace()` wiring works.
  6. Parity spot-check: `research_dense_fetch`/`DenseFetchWorkspace` are completely untouched.
"""
from __future__ import annotations

import pytest

from agent_search.agent.tools.doc_research import (
    DenseFetchPlainWorkspace, DenseFetchWorkspace, DocSearchFetch)
from agent_search.corpus.units import units_from_documents

_PAD = "x"          # same padding trick as test_dense_baseline.py — isolates "did the MID-BODY
                    # window win" from "is the char/token cap doing something weird" (only
                    # relevant to the excerpt-bearing sibling here, kept for the side-by-side test).


def _padded_doc(doc_id, title, needle, pad_before=30, pad_after=30):
    before = " ".join([_PAD] * pad_before)
    after = " ".join([_PAD] * pad_after)
    return {"_id": doc_id, "title": title, "text": f"{before} {needle}\n\n## History\n{after}"}


DOCS = [
    _padded_doc("d_mid", "Mid-body Match", "zephyrquokka marker phrase right here"),
    {"_id": "d_plain2", "title": "Plain Doc Two", "text": "A short document about nothing special."},
]


def _units():
    return units_from_documents(DOCS)


class _StubDenseEngine:
    """SAME CPU-only stand-in as test_dense_baseline.py's own `_StubDenseEngine` (no torch
    import) — `top_k_doc_ids(query, k)` -> ranked doc_ids, `ranking` is a fixed list or a
    {query: [...]} map."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def top_k_doc_ids(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


def _ws(query="zephyrquokka", ranking=("d_mid",), topk=5, units=None):
    return DenseFetchPlainWorkspace(units or _units(), query,
                                    engine=_StubDenseEngine(ranking), topk=topk)


# =============================================================================================
# 1. tools/dispatch, plain listing, fetch delegation, hallucinated tool
# =============================================================================================

def test_tools_tuple_is_dense_search_fp_and_fetch():
    assert DenseFetchPlainWorkspace.tools == ("dense_search_fp", "fetch")


def test_is_a_densefetchworkspace_subclass_reusing_fetch_verbatim():
    """DenseFetchPlainWorkspace is DenseFetchWorkspace's plain twin (search->fetch, read side
    INHERITED, never reimplemented) — only `search`/`run`/`tools` differ (the excerpt)."""
    assert issubclass(DenseFetchPlainWorkspace, DenseFetchWorkspace)
    assert issubclass(DenseFetchPlainWorkspace, DocSearchFetch)
    assert DenseFetchPlainWorkspace.fetch is DocSearchFetch.fetch
    assert DenseFetchPlainWorkspace._fetch_one is DocSearchFetch._fetch_one
    assert DenseFetchPlainWorkspace._resolve_doc is DocSearchFetch._resolve_doc
    # __init__ is NOT overridden — same construction shape as DenseFetchWorkspace.
    assert DenseFetchPlainWorkspace.__init__ is DenseFetchWorkspace.__init__


def test_search_lists_structure_with_no_excerpt():
    ws = _ws()
    out = ws.run("dense_search_fp", {"query": "zephyrquokka"})
    assert "d_mid" in out and "'Mid-body Match'" in out
    assert "History" in out               # section name shown (structure)
    assert "»" not in out                 # NO excerpt marker — the whole point of this cell


def test_search_marks_hits_seen():
    ws = _ws(ranking=("d_mid", "d_plain2"))
    ws.run("dense_search_fp", {"query": "zephyrquokka"})
    assert {"d_mid", "d_plain2"} <= ws.seen


def test_empty_query_message():
    ws = DenseFetchPlainWorkspace(_units(), "", engine=_StubDenseEngine(("d_mid",)))
    assert ws.last_hits == []
    assert ws.search("") == "empty query"


def test_zero_hits_message():
    ws = DenseFetchPlainWorkspace(_units(), "q", engine=_StubDenseEngine([]))
    out = ws.run("dense_search_fp", {"query": "nothing matches"})
    assert "0 matches" in out


def test_engine_receives_raw_query_and_topk():
    engine = _StubDenseEngine(("d_mid",))
    ws = DenseFetchPlainWorkspace(_units(), "q", engine=engine, topk=7)
    ws.run("dense_search_fp", {"query": "zephyrquokka"})
    assert engine.calls == [("zephyrquokka", 7)]


def test_search_is_live_and_re_retrieves():
    engine = _StubDenseEngine({"zephyrquokka": ["d_mid"], "plain doc": ["d_plain2"]})
    ws = DenseFetchPlainWorkspace(_units(), "zephyrquokka", engine=engine)
    ws.run("dense_search_fp", {"query": "zephyrquokka"})
    first = list(ws.last_hits)
    ws.run("dense_search_fp", {"query": "plain doc"})
    assert ws.last_hits != first
    assert "d_plain2" in ws.last_hits


def test_fetch_by_rank_after_search():
    ws = _ws()
    ws.run("dense_search_fp", {"query": "zephyrquokka"})   # rank 1 = d_mid
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert "History" in out and "ERROR" not in out


def test_fetch_bad_section_lists_available():
    ws = _ws()
    ws.run("dense_search_fp", {"query": "zephyrquokka"})
    out = ws.run("fetch", {"specs": [[1, "Nonexistent"]]})
    assert "no section" in out and "History" in out


def test_fetch_marks_doc_seen():
    ws = _ws()
    ws.run("dense_search_fp", {"query": "zephyrquokka"})
    ws.run("fetch", {"specs": [[1, "History"]]})
    assert "d_mid" in ws.seen


def test_run_aliases_dense_search_fp_dense_search_f_dense_search_and_search_names():
    out1 = _ws().run("dense_search_fp", {"query": "zephyrquokka"})
    out2 = _ws().run("dense_search_f", {"query": "zephyrquokka"})
    out3 = _ws().run("dense_search", {"query": "zephyrquokka"})
    out4 = _ws().run("search", {"query": "zephyrquokka"})
    assert out1 == out2 == out3 == out4


def test_run_unknown_tool_errors():
    out = _ws().run("visit", {"rank": 1})            # visit is not a tool of this arm
    assert "unknown tool" in out.lower()


def test_run_hallucinated_tool_names_error_not_silent_success():
    ws = _ws()
    ws.run("dense_search_fp", {"query": "zephyrquokka"})
    for bad_tool in ("dense_search_snip", "visit_d", "fetch_bqlds", "not_a_real_tool"):
        out = ws.run(bad_tool, {"rank": 1})
        assert out.startswith("ERROR: unknown tool"), (bad_tool, out)


# =============================================================================================
# 2. Side-by-side: SAME engine/query -> plain vs snip listings differ ONLY by the `»` line
# =============================================================================================

def test_dense_search_fp_and_dense_search_f_differ_only_by_the_excerpt_line():
    units = _units()
    engine_plain = _StubDenseEngine(("d_mid", "d_plain2"))
    engine_snip = _StubDenseEngine(("d_mid", "d_plain2"))

    plain_out = DenseFetchPlainWorkspace(units, "zephyrquokka", engine=engine_plain).run(
        "dense_search_fp", {"query": "zephyrquokka"})
    snip_out = DenseFetchWorkspace(units, "zephyrquokka", engine=engine_snip).run(
        "dense_search_f", {"query": "zephyrquokka"})

    assert "»" not in plain_out
    assert "»" in snip_out
    strip_excerpt = lambda s: [ln.split("»", 1)[0].rstrip() for ln in s.splitlines()]
    assert strip_excerpt(plain_out) == strip_excerpt(snip_out)


# =============================================================================================
# 3. Condition loads + resolves via the retriever registry
# =============================================================================================

def test_research_dense_fetch_plain_condition_loads_uncoached():
    from agent_search.prompts import load_condition, render_manuals

    p = load_condition("research_dense_fetch_plain")
    assert p.toolset == "dense_fetch_plain"
    assert set(p.tool_names) == {"dense_search_fp", "fetch"}
    # UNCOACHED like research_dense_fetch/research_bm25_fetch: no manual renders for this toolset.
    assert render_manuals(p.tool_names, domain="general") == ""
    assert "term[field]" not in p.system


def test_research_dense_fetch_plain_resolves_via_registry_as_densefetchplain_arm():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense_fetch_plain", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("dense_search_fp", "fetch")
    assert r.tool == "agent_research_dense_fetch_plain"
    assert r._arm == "densefetchplain"
    assert r.domain == "general"
    assert not r.needs_files


def test_densefetchplain_index_raises_clear_error_when_cache_missing(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense_fetch_plain", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_units(), key="no_such_corpus_key")


# =============================================================================================
# 4. Full AgentRetriever offline smoke (fake dense stack — no torch/network), mirrors
#    tests/test_hybrid.py's `fake_dense_stack` fixture pattern.
# =============================================================================================

@pytest.fixture
def fake_dense_stack(monkeypatch):
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


def test_agentretriever_densefetchplain_workspace_builds_and_answers_via_stub(
        tmp_path, fake_dense_stack):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_dense_fetch_plain", cfg)()
    r.index(_units(), key="test-densefetchplain-corpus")
    ws = r._workspace(5, "zephyrquokka")
    assert isinstance(ws, DenseFetchPlainWorkspace)
    out = ws.run("dense_search_fp", {"query": "zephyrquokka"})
    assert "ERROR" not in out
    assert "»" not in out
    ranking = r.search("zephyrquokka", k=5)
    assert isinstance(ranking, list)


# =============================================================================================
# 5. Parity spot-check: research_dense_fetch / DenseFetchWorkspace are completely untouched.
# =============================================================================================

def test_research_dense_fetch_condition_unaffected_by_the_new_plain_cell():
    from agent_search.prompts import load_condition

    p = load_condition("research_dense_fetch")
    assert p.toolset == "dense_fetch"
    assert set(p.tool_names) == {"dense_search_f", "fetch"}


def test_densefetchworkspace_still_always_renders_the_excerpt():
    units = _units()
    ws = DenseFetchWorkspace(units, "zephyrquokka", engine=_StubDenseEngine(("d_mid",)))
    out = ws.run("dense_search_f", {"query": "zephyrquokka"})
    assert "»" in out


def test_research_dense_fetch_resolves_via_registry_as_densefetch_arm_unaffected():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense_fetch", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("dense_search_f", "fetch")
    assert r._arm == "densefetch"
    assert r.domain == "general"
