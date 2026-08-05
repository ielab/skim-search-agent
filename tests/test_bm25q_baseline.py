"""`research_bm25q`: the HARDENED bm25 baseline (Bm25Visit, `query_biased=True`) — a fairness
fix, not a new capability. `research_bm25`'s search listing shows each hit's fixed doc OPENING
snippet; our method cells (research_snip/research_indri_snip) show a QUERY-BIASED
best-matching excerpt per hit via `doc_research.py`'s `_best_line`. Real search engines show
query-biased snippets, so the hardened bm25 baseline must too, or the listing axis is unfair
in our favor.

`research_bm25q` is `research_bm25`'s SAME bm25 retrieval + whole-doc `visit` read
(`bm25q_search`/`visit_q`, aliasing `Bm25Visit`'s `search`/`visit`), with ONLY the listing's
snippet swapped to the module-level `best_line` window-scoring the method cells use.
`query_biased=False` (the default) must reproduce the OLD `Bm25Visit.search` rendering
byte-for-byte — the existing `research_bm25` condition/toolset is completely unaffected.

CPU-only, no network — a stub BM25 engine, the same pattern test_doc_research_tools.py /
test_dense_baseline.py use for their stub retrieval engines.
"""
from __future__ import annotations

from agent_search.agent.tools.doc_research import Bm25Visit, DocSearchFetch, best_line
from agent_search.corpus.units import units_from_documents

_PAD = "x"          # a 1-char filler token; pad_before is chosen large enough that the doc's
                    # fixed 120-CHARACTER opening slice (Bm25Visit's OLD snippet) never reaches
                    # the needle, isolating "opening slice" from "query-biased window".


def _padded_doc(doc_id: str, title: str, needle: str, pad_before: int = 80,
                pad_after: int = 30) -> dict:
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
    """A CPU-only stand-in for BM25Local exposing ONLY `search(query, k)` -> ranked doc_ids —
    same pattern test_doc_research_tools.py hands Bm25Visit a real-but-tiny BM25Local, and
    test_dense_baseline.py hands DenseVisit a stub engine."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


def _ws(ranking=("d_mid",), query_biased=False):
    return Bm25Visit(_units(), engine=_StubBm25(ranking), query_biased=query_biased)


# --- 1. module-level `best_line` refactor: DocSearchFetch._best_line is a thin wrapper --------

def test_best_line_is_module_level_and_docsearchfetch_wraps_it():
    ws = DocSearchFetch(_units())
    u = ws.ubyid["d_mid"]
    assert ws._best_line(u, ["zephyrquokka"]) == best_line(u, ["zephyrquokka"])
    # sanity: it actually finds the needle window, not the (all-filler) opening.
    assert "zephyrquokka" in ws._best_line(u, ["zephyrquokka"])


# --- 2. tools tuple switches with query_biased (instance override, DenseVisit-style) ----------

def test_tools_tuple_default_is_bm25_search_and_visit():
    assert Bm25Visit(_units(), engine=_StubBm25(())).tools == ("bm25_search", "visit")


def test_tools_tuple_query_biased_is_bm25q_search_and_visit_q():
    assert (Bm25Visit(_units(), engine=_StubBm25(()), query_biased=True).tools
            == ("bm25q_search", "visit_q"))


# --- 3. query_biased=False: byte-parity with the OLD Bm25Visit.search rendering ---------------

def test_query_biased_false_is_byte_identical_to_the_old_default():
    units = _units()
    a = Bm25Visit(units, engine=_StubBm25(("d_mid", "d_flat")))                  # old default
    b = Bm25Visit(units, engine=_StubBm25(("d_mid", "d_flat")), query_biased=False)  # explicit
    out_a = a.search("zephyrquokka marker")
    out_b = b.search("zephyrquokka marker")
    assert out_a == out_b


def test_query_biased_false_opening_snippet_misses_the_mid_body_needle():
    """The OLD/default rendering is a fixed 120-CHARACTER opening slice — with pad_before=80
    filler tokens, that slice never reaches the mid-body needle."""
    ws = _ws(("d_mid",), query_biased=False)
    out = ws.run("bm25_search", {"query": "zephyrquokka marker"})
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    assert "zephyrquokka" not in hit_line


# --- 4. run() dispatch: bm25q_search/bm25_search/search + visit_q/visit aliases ---------------

def test_run_aliases_bm25q_search_and_bm25_search_and_search():
    ws1 = _ws(("d_mid",), query_biased=True)
    out1 = ws1.run("bm25q_search", {"query": "zephyrquokka"})
    ws2 = _ws(("d_mid",), query_biased=True)
    out2 = ws2.run("bm25_search", {"query": "zephyrquokka"})
    ws3 = _ws(("d_mid",), query_biased=True)
    out3 = ws3.run("search", {"query": "zephyrquokka"})
    assert out1 == out2 == out3


def test_run_aliases_visit_q_and_visit():
    ws1 = _ws(("d_mid",), query_biased=True)
    ws1.run("bm25q_search", {"query": "zephyrquokka"})
    out1 = ws1.run("visit_q", {"rank": 1})
    ws2 = _ws(("d_mid",), query_biased=True)
    ws2.run("bm25q_search", {"query": "zephyrquokka"})
    out2 = ws2.run("visit", {"rank": 1})
    assert out1 == out2
    assert "zephyrquokka" in out1


def test_run_unknown_tool_errors():
    out = _ws(query_biased=True).run("fetch", {"specs": [[1, "History"]]})
    assert "unknown tool" in out.lower()


# --- 6. engine call shape (query passthrough + knob depth, unaffected by query_biased) --------

def test_engine_receives_the_raw_query_and_the_knob_depth():
    """run() passes the RAW query through, with BM25_VISIT_TOPK as k — tools.yaml's
    bm25q_search schema exposes ONLY `query`, so a hallucinated `k` in the tool-call args is
    ignored; the env knob alone sets the SERP listing depth (see doc_research.BM25_VISIT_TOPK
    and test_doc_research_tools.py's SERP-listing-depth section)."""
    import agent_search.agent.tools.doc_research as m
    engine = _StubBm25(("d_mid",))
    ws = Bm25Visit(_units(), engine=engine, query_biased=True)
    ws.run("bm25q_search", {"query": "zephyrquokka marker", "k": 3})
    assert engine.calls == [("zephyrquokka marker", m.BM25_VISIT_TOPK)]


# --- 7. condition wiring: research_bm25q was pruned from conditions.yaml (paper's 15 kept
# conditions) — the query_biased machinery above is still exercised directly, but the
# condition/registry wiring tests for the removed `research_bm25q` arm are gone with it.


def test_existing_research_bm25_condition_is_unaffected():
    from agent_search.prompts import load_condition

    p = load_condition("research_bm25")
    assert p.toolset == "research_bm25"
    assert p.tool_names == ("bm25_search", "visit")
