"""The `query_biased` fairness mechanism on `search_bm25` (formerly `Bm25Visit`'s
`query_biased` kwarg), once exposed as the hardened bm25 baseline `research_bm25q`, now pruned
from the paper's kept conditions. The plain `research_bm25` search listing shows each hit's
fixed doc opening snippet, while the method cells (research_snip, research_indri_snip) show a
query-biased best-matching excerpt per hit via `agent_search/tools/common.py`'s `best_line`.
Real search engines show query-biased snippets, so a fair bm25 baseline needs to as well;
that fairness fix is what `query_biased=True` provides, tested directly here since the
condition itself is gone.

`query_biased=True` keeps the plain listing's bm25 retrieval and whole-doc `visit` read
(`bm25q_search`/`visit_q`, aliasing the plain listing's `search`/`visit`), with only the
listing's snippet swapped to the module-level `best_line` window-scoring the method cells use.
`query_biased=False` (the default) reproduces the plain rendering byte-for-byte; the existing
`research_bm25` condition/toolset is unaffected.

CPU-only, no network: a stub BM25 engine.
"""
from __future__ import annotations

from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.budgets import SNIPPET_TOKENS
from agent_search.tools.common import best_line
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_bql.tool import SearchBql
from agent_search.tools.visit.tool import Visit

_PAD = "x"          # a 1-char filler token; pad_before is chosen large enough that the doc's
                    # fixed opening snippet never reaches the needle, isolating "opening slice"
                    # from "query-biased window".


def _padded_doc(doc_id: str, title: str, needle: str,
                pad_before: int = SNIPPET_TOKENS + 5,
                pad_after: int = SNIPPET_TOKENS + 5) -> dict:
    before = " ".join([_PAD] * pad_before)
    after = " ".join([_PAD] * pad_after)
    return {"_id": doc_id, "title": title, "text": f"{before} {needle} {after}"}


DOCS = [
    _padded_doc("d_mid", "Mid-body Match", "zephyrquokka marker phrase right here"),
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
]


def _units():
    return units_from_documents(DOCS)


class _StubBm25:
    """A CPU-only stand-in for BM25Local exposing ONLY `search(query, k)` -> ranked doc_ids."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


def _toolbox(ranking=("d_mid",), query_biased=False):
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    name_search, name_visit = ("bm25q_search", "visit_q") if query_biased else ("bm25_search", "visit")
    search = SearchBm25(name=name_search, query_biased=query_biased).bind(
        state, units, ubyid, {"bm25": _StubBm25(ranking)})
    visit = Visit(name=name_visit).bind(state, units, ubyid, {})
    return ToolBox([search, visit], state), search


# --- 1. `best_line` is module-level, and search_bql's own `_best_line` is a thin wrapper -------

def test_best_line_is_module_level_and_search_bql_wraps_it():
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchBql(name="search").bind(state, units, ubyid, {"bql": None})
    u = search.ubyid["d_mid"]
    assert search._best_line(u, ["zephyrquokka"]) == best_line(u, ["zephyrquokka"])
    # sanity: it actually finds the needle window, not the (all-filler) opening.
    assert "zephyrquokka" in search._best_line(u, ["zephyrquokka"])


# --- 2. tools tuple switches with query_biased (instance override) ---------------------------

def test_tools_tuple_default_is_bm25_search_and_visit():
    box, _ = _toolbox(())
    assert box.tools == ("bm25_search", "visit")


def test_tools_tuple_query_biased_is_bm25q_search_and_visit_q():
    box, _ = _toolbox((), query_biased=True)
    assert box.tools == ("bm25q_search", "visit_q")


# --- 3. query_biased=False: byte-parity with the plain default rendering ----------------------

def test_query_biased_false_is_byte_identical_to_the_default():
    ranking = ("d_mid", "d_flat")
    box_a, search_a = _toolbox(ranking)                     # default (no query_biased kwarg)
    box_b, search_b = _toolbox(ranking, query_biased=False)  # explicit False
    out_a = box_a.run("bm25_search", {"query": "zephyrquokka marker"})
    out_b = box_b.run("bm25_search", {"query": "zephyrquokka marker"})
    assert out_a == out_b


def test_query_biased_false_opening_snippet_misses_the_mid_body_needle():
    """The default rendering is the doc's OPENING window (`opening_line`, SNIPPET_TOKENS
    wide) — the fixture pads one full window of filler before the needle, so the opening
    snippet never reaches it at any configured width."""
    box, _ = _toolbox(("d_mid",), query_biased=False)
    out = box.run("bm25_search", {"query": "zephyrquokka marker"})
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    assert "zephyrquokka" not in hit_line


# --- 4. run() dispatch: bm25q_search/bm25_search/search + visit_q/visit aliases ---------------

def test_run_aliases_bm25q_search_and_bm25_search_and_search():
    box1, _ = _toolbox(("d_mid",), query_biased=True)
    out1 = box1.run("bm25q_search", {"query": "zephyrquokka"})
    box2, _ = _toolbox(("d_mid",), query_biased=True)
    out2 = box2.run("bm25_search", {"query": "zephyrquokka"})
    box3, _ = _toolbox(("d_mid",), query_biased=True)
    out3 = box3.run("search", {"query": "zephyrquokka"})
    assert out1 == out2 == out3


def test_run_aliases_visit_q_and_visit():
    box1, _ = _toolbox(("d_mid",), query_biased=True)
    box1.run("bm25q_search", {"query": "zephyrquokka"})
    out1 = box1.run("visit_q", {"rank": 1})
    box2, _ = _toolbox(("d_mid",), query_biased=True)
    box2.run("bm25q_search", {"query": "zephyrquokka"})
    out2 = box2.run("visit", {"rank": 1})
    assert out1 == out2
    assert "zephyrquokka" in out1


def test_run_unknown_tool_errors():
    box, _ = _toolbox(query_biased=True)
    out = box.run("fetch", {"specs": [[1, "History"]]})
    assert "unknown tool" in out.lower()


# --- 5. engine call shape (query passthrough + knob depth, unaffected by query_biased) --------

def test_engine_receives_the_raw_query_and_the_knob_depth():
    """run() passes the RAW query through, with BM25_VISIT_TOPK as k — the bm25q_search schema
    exposes ONLY `query`, so a hallucinated `k` in the tool-call args is ignored; the env knob
    alone sets the SERP listing depth."""
    import agent_search.tools.budgets as m
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    engine = _StubBm25(("d_mid",))
    search = SearchBm25(name="bm25q_search", query_biased=True).bind(state, units, ubyid, {"bm25": engine})
    box = ToolBox([search, Visit(name="visit_q").bind(state, units, ubyid, {})], state)
    box.run("bm25q_search", {"query": "zephyrquokka marker", "k": 3})
    assert engine.calls == [("zephyrquokka marker", m.BM25_VISIT_TOPK)]


# --- 6. condition wiring: research_bm25q was pruned from the paper's kept conditions — the
# query_biased machinery above is still exercised directly, but the condition/registry wiring
# tests for the removed `research_bm25q` arm are gone with it.


def test_existing_research_bm25_condition_is_unaffected():
    from agent_search.strategies import CONDITIONS

    p = CONDITIONS["research_bm25"]
    assert p.strategy.toolset_name == "research_bm25"
    assert p.tool_names == ("bm25_search", "visit")
