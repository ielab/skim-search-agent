"""The dense structured-fetch pair: `search_dense` (`structure=True`) then `fetch`.

Two arms share this shape: `dense_search_f` (`snippet=TermWindow()`, `research_dense_fetch`) always
renders a best-matching `»` excerpt per hit; `dense_search_fp` (`snippet=NoSnippet()`,
the missing plain sibling — dense was the only query engine among bm25/bql/indri/dense that
used to lack a plain, no-excerpt fetch cell) renders the identical listing minus that excerpt
line, the same way `research_bql_dense_fetch` supplies `research_bql_dense_snip`'s missing
plain sibling for the dense-fused BQL executor. Both share the SAME retrieval and the SAME
structured section-fetch tool (`agent_search.tools.fetch.Fetch`) — only the listing's excerpt
differs.

CPU-only via a stub dense engine (no torch import).
"""
from __future__ import annotations

from agent_search.snippets import NoSnippet, TermWindow
from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_dense.tool import SearchDense

_PAD = "x"          # padding trick to isolate "did the MID-BODY window win" from "is the
                    # excerpt cap doing something weird" (relevant to the snip variant).


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
    """CPU-only stand-in (no torch import): `top_k_doc_ids(query, k)` -> ranked doc_ids,
    `ranking` is a fixed list or a {query: [...]} map."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def top_k_doc_ids(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


def _toolbox(name, snippet, ranking=("d_mid",), topk=None, units=None):
    units = units if units is not None else _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    opts = {"structure": True, "snippet": snippet}
    if topk is not None:
        opts["k"] = topk
    search = SearchDense(name=name, **opts).bind(state, units, ubyid, {"dense": _StubDenseEngine(ranking)})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([search, fetch], state)


def _plain_box(ranking=("d_mid",), topk=None, units=None):
    return _toolbox("dense_search_fp", snippet=NoSnippet(), ranking=ranking, topk=topk, units=units)


def _snip_box(ranking=("d_mid",), topk=None, units=None):
    return _toolbox("dense_search_f", snippet=TermWindow(), ranking=ranking, topk=topk, units=units)


# =============================================================================================
# 1. dense_search_fp (plain, no excerpt): tools/dispatch, listing, fetch delegation
# =============================================================================================

def test_plain_tools_tuple_is_dense_search_fp_and_fetch():
    assert _plain_box().tools == ("dense_search_fp", "fetch")


def test_plain_search_lists_structure_with_no_excerpt():
    box = _plain_box()
    out = box.run("dense_search_fp", {"query": "zephyrquokka"})
    assert "d_mid" in out and "'Mid-body Match'" in out
    assert "History" in out               # section name shown (structure)
    assert "»" not in out                 # NO excerpt marker — the whole point of this cell


def test_plain_search_marks_hits_seen():
    box = _plain_box(ranking=("d_mid", "d_plain2"))
    box.run("dense_search_fp", {"query": "zephyrquokka"})
    assert {"d_mid", "d_plain2"} <= set(box.seen)


def test_plain_empty_query_message():
    box = _plain_box(ranking=("d_mid",))
    assert box.run("dense_search_fp", {"query": ""}) == "empty query"


def test_plain_zero_hits_message():
    box = _plain_box(ranking=[])
    out = box.run("dense_search_fp", {"query": "nothing matches"})
    assert "0 matches" in out


def test_plain_engine_receives_raw_query_and_topk():
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    engine = _StubDenseEngine(("d_mid",))
    search = SearchDense(name="dense_search_fp", structure=True, snippet=NoSnippet(), k=7).bind(
        state, units, ubyid, {"dense": engine})
    box = ToolBox([search, Fetch(name="fetch").bind(state, units, ubyid, {})], state)
    box.run("dense_search_fp", {"query": "zephyrquokka"})
    assert engine.calls == [("zephyrquokka", 7)]


def test_plain_search_is_live_and_re_retrieves():
    box = _toolbox("dense_search_fp", snippet=NoSnippet(),
                   ranking={"zephyrquokka": ["d_mid"], "plain doc": ["d_plain2"]})
    box.run("dense_search_fp", {"query": "zephyrquokka"})
    first = list(box.last_hits)
    box.run("dense_search_fp", {"query": "plain doc"})
    assert box.last_hits != first
    assert "d_plain2" in box.last_hits


def test_plain_fetch_by_rank_after_search():
    box = _plain_box()
    box.run("dense_search_fp", {"query": "zephyrquokka"})   # rank 1 = d_mid
    out = box.run("fetch", {"specs": [[1, "History"]]})
    assert "History" in out and "ERROR" not in out


def test_plain_fetch_bad_section_lists_available():
    box = _plain_box()
    box.run("dense_search_fp", {"query": "zephyrquokka"})
    out = box.run("fetch", {"specs": [[1, "Nonexistent"]]})
    assert "no section" in out and "History" in out


def test_plain_fetch_marks_doc_seen():
    box = _plain_box()
    box.run("dense_search_fp", {"query": "zephyrquokka"})
    box.run("fetch", {"specs": [[1, "History"]]})
    assert "d_mid" in box.seen


def test_plain_run_aliases_dense_search_and_search_names():
    """`dense_search_fp` is the tool's own registered name; `dense_search`/`search` are its
    declared aliases (agent_search.tools.search_dense.tool.SearchDense.aliases). There is no
    `dense_search_f` alias on THIS instance — that name belongs to the separate snippet=TermWindow()
    instance below (dropped from the old alias-fan-out test: the two are distinct tool
    instances with different `snippets` settings now, not interchangeable names of one
    generic workspace)."""
    out1 = _plain_box().run("dense_search_fp", {"query": "zephyrquokka"})
    out2 = _plain_box().run("dense_search", {"query": "zephyrquokka"})
    out3 = _plain_box().run("search", {"query": "zephyrquokka"})
    assert out1 == out2 == out3


def test_plain_run_unknown_tool_errors():
    box = _plain_box()
    out = box.run("visit", {"rank": 1})            # visit is not a tool of this arm
    assert "unknown tool" in out.lower()


def test_plain_run_hallucinated_tool_names_error_not_silent_success():
    """`fetch_bqlds` is dropped from the old hallucinated-name list: `Fetch` is now one shared
    tool serving every search-fetch pairing (agent_search.tools.fetch.Fetch.aliases includes
    `fetch_bqlds`), so it legitimately answers that name here too — the old per-workspace exact
    dispatch that rejected it is gone by design, not a regression."""
    box = _plain_box()
    box.run("dense_search_fp", {"query": "zephyrquokka"})
    for bad_tool in ("dense_search_snip", "visit_d", "not_a_real_tool"):
        out = box.run(bad_tool, {"rank": 1})
        assert out.startswith("ERROR: unknown tool"), (bad_tool, out)


# =============================================================================================
# 2. Side-by-side: SAME engine/query -> plain vs snip listings differ ONLY by the `»` line
# =============================================================================================

def test_dense_search_fp_and_dense_search_f_differ_only_by_the_excerpt_line():
    units = _units()
    plain_out = _plain_box(ranking=("d_mid", "d_plain2"), units=units).run(
        "dense_search_fp", {"query": "zephyrquokka"})
    snip_out = _snip_box(ranking=("d_mid", "d_plain2"), units=units).run(
        "dense_search_f", {"query": "zephyrquokka"})

    assert "»" not in plain_out
    assert "»" in snip_out
    strip_excerpt = lambda s: [ln.split("»", 1)[0].rstrip() for ln in s.splitlines()]
    assert strip_excerpt(plain_out) == strip_excerpt(snip_out)


# =============================================================================================
# 3. dense_search_f (research_dense_fetch) — the {dense search} x {structure->parts read}
# factorial cell: dense embedding LIVE RETRIEVAL (search_dense's engine/DenseBelief) + BQL
# structured SECTION-FETCH read (the plain fetch cell's read, shared via Fetch unchanged), PLUS
# a one-line best-matching excerpt per hit (research_snip's `best_line`) since research_snip
# established a content-bearing listing is the fair default once any arm shows one.
# =============================================================================================

def test_snip_tools_tuple_is_dense_search_f_and_fetch():
    assert _snip_box().tools == ("dense_search_f", "fetch")


def test_snip_search_lists_structure_plus_excerpt_not_full_text():
    box = _snip_box()
    out = box.run("dense_search_f", {"query": "zephyrquokka"})
    assert "d_mid" in out and "'Mid-body Match'" in out
    assert "History" in out                      # section name shown (structure)
    assert "»" in out                             # excerpt marker present (content-bearing)
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "zephyrquokka" in excerpt
    from agent_search.tools.budgets import SNIPPET_TOKENS
    assert len(excerpt.split()) <= SNIPPET_TOKENS  # a bounded window, not the whole padded body


def test_snip_excerpt_is_mid_body_not_the_doc_opening():
    box = _snip_box()
    out = box.run("dense_search_f", {"query": "zephyrquokka"})
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    opening = " ".join(["x"] * 30)
    assert excerpt != opening


def test_snip_marks_hits_seen():
    box = _snip_box(ranking=("d_mid", "d_plain2"))
    box.run("dense_search_f", {"query": "zephyrquokka"})
    assert {"d_mid", "d_plain2"} <= set(box.seen)


def test_snip_empty_query_message():
    box = _snip_box(ranking=("d_mid",))
    assert box.run("dense_search_f", {"query": ""}) == "empty query"


def test_snip_zero_hits_message():
    box = _snip_box(ranking=[])
    out = box.run("dense_search_f", {"query": "nothing matches"})
    assert "0 matches" in out


def test_snip_engine_receives_raw_query_and_topk():
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    engine = _StubDenseEngine(("d_mid",))
    search = SearchDense(name="dense_search_f", structure=True, snippet=TermWindow(), k=7).bind(
        state, units, ubyid, {"dense": engine})
    box = ToolBox([search, Fetch(name="fetch").bind(state, units, ubyid, {})], state)
    box.run("dense_search_f", {"query": "zephyrquokka"})
    assert engine.calls == [("zephyrquokka", 7)]


def test_snip_search_is_live_and_re_retrieves():
    """LIVE retrieval: a NEW query string in a later dense_search_f call re-runs the dense
    engine and CHANGES the ranking — same live-per-call contract as the bm25 cells."""
    box = _toolbox("dense_search_f", snippet=TermWindow(),
                   ranking={"zephyrquokka": ["d_mid"], "plain doc": ["d_plain2"]})
    box.run("dense_search_f", {"query": "zephyrquokka"})
    first = list(box.last_hits)
    box.run("dense_search_f", {"query": "plain doc"})
    assert box.last_hits != first
    assert "d_plain2" in box.last_hits


def test_snip_fetch_by_rank_after_search():
    box = _snip_box()
    box.run("dense_search_f", {"query": "zephyrquokka"})   # rank 1 = d_mid
    out = box.run("fetch", {"specs": [[1, "History"]]})
    assert "History" in out and "ERROR" not in out


def test_snip_fetch_bad_section_lists_available():
    box = _snip_box()
    box.run("dense_search_f", {"query": "zephyrquokka"})
    out = box.run("fetch", {"specs": [[1, "Nonexistent"]]})
    assert "no section" in out and "History" in out


def test_snip_fetch_marks_doc_seen():
    box = _snip_box()
    box.run("dense_search_f", {"query": "zephyrquokka"})
    box.run("fetch", {"specs": [[1, "History"]]})
    assert "d_mid" in box.seen


def test_snip_run_aliases_dense_search_and_search_names():
    out1 = _snip_box().run("dense_search_f", {"query": "zephyrquokka"})
    out2 = _snip_box().run("dense_search", {"query": "zephyrquokka"})
    out3 = _snip_box().run("search", {"query": "zephyrquokka"})
    assert out1 == out2 == out3


def test_snip_run_unknown_tool_errors():
    box = _snip_box()
    out = box.run("visit", {"rank": 1})            # visit is not a tool of this arm
    assert "unknown tool" in out.lower()


def test_research_dense_fetch_condition_loads_uncoached():
    from agent_search.strategies import CONDITIONS
    from agent_search.tasks.render import render_manuals

    p = CONDITIONS["research_dense_fetch"]
    assert p.strategy.toolset_name == "dense_fetch"
    assert set(p.tool_names) == {"dense_search_f", "fetch"}
    # UNCOACHED like research_bm25_fetch/research_dense: no manual renders for this toolset.
    assert render_manuals([t.manual_path("general") for t in p.strategy.tools]) == ""
    assert "term[field]" not in p.render()


def test_research_dense_fetch_resolves_via_registry_as_densefetch_arm():
    from agent_search.evaluation.agent_runner import ConditionAgent
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense_fetch", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("dense_search_f", "fetch")
    assert r.tool == "agent_research_dense_fetch"
    assert r.condition.name == "research_dense_fetch"
    assert r.domain == "general"
    assert not r.needs_files


# =============================================================================================
# 4. Condition wiring: `research_dense_fetch_plain` was pruned from the paper's kept
# conditions — the `dense_search_fp` cell it wired up is still exercised directly above.
# =============================================================================================
