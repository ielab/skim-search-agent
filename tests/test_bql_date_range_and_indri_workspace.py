"""Tests for the BQL v2 toolset (typed date[RANGE] ranges plus constraint-coverage feedback,
formerly the `research_v2` condition) and the Indri graded query-language tools (formerly
`research_indri`). Both conditions were later pruned from conditions.yaml; the underlying
tools (`SearchBql`/`SearchIndri` + `Fetch`) are still tested directly here, on the document
engines (`LuceneBqlAdapter`, `LuceneIndriAdapter` over a prebuilt Lucene index built by
`tests/lucene_support.py`), so the module skips without a JVM.

CPU-only; a small (~25-doc) synthetic corpus with dates/sections. The plain `research`
condition/toolset/`SearchBql` default behavior must match exactly regardless of these
additions; several assertions below pin that explicitly.
"""
from __future__ import annotations

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bql.tool import SearchBql
from agent_search.tools.search_indri.tool import SearchIndri
from tests.lucene_support import build_lucene_bql, build_lucene_indri, require_jvm

require_jvm()


# --- shared ~25-doc corpus: dates + sections ------------------------------------------

def _mk(doc_id: str, body: str, title: str | None = None, date: str | None = None) -> CodeUnit:
    meta = {"date": date} if date is not None else {}
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=body, body=body, title=title, metadata=meta)


def _corpus() -> list[CodeUnit]:
    units = [
        # --- constraint-coverage fixture: AND(foo, bar, baz, qux) 0-hits exactly ---
        _mk("docA", "## History\nfoo bar baz text here", title="Doc A", date="2001-01-01"),
        _mk("docB", "## History\nfoo bar text", title="Doc B", date="2001-01-02"),
        _mk("docC", "## History\nnothing relevant here", title="Doc C", date="2001-01-03"),
        _mk("docD", "## History\nfoo only here", title="Doc D", date="2001-01-04"),
        # --- indri #combine no-full-match fixture (5 rare terms, no doc has all 5) ---
        _mk("nz1", "## History\nalpha bravo filler filler filler", title="NZ1"),
        _mk("nz2", "## History\ncharlie delta filler filler filler", title="NZ2"),
        _mk("nz3", "## History\necho filler filler filler filler", title="NZ3"),
        _mk("nz4", "## History\nfiller filler filler filler filler", title="NZ4"),
        # --- date-range / #date:between fixture ---
        _mk("d1979", "## History\nalpha before", title="D1979", date="1979-12-31"),
        _mk("d1980", "## History\nalpha instart", title="D1980", date="1980-01-01"),
        _mk("d1985", "## History\nalpha midyear", title="D1985", date="1985-06-15"),
        _mk("d1989", "## History\nalpha inend", title="D1989", date="1989-12-31"),
        _mk("d1990", "## History\nalpha after", title="D1990", date="1990-01-01"),
        _mk("d2002", "## History\nbank management ceremony held", title="Bank 2002",
            date="2002-06-01"),
        _mk("d1999", "## History\nbank management ceremony held", title="Bank 1999",
            date="1999-06-01"),
    ]
    filler_vocab = ["widget", "gadget", "sprocket", "lever", "cog", "gear", "pulley",
                    "spring", "bolt", "nut", "washer", "hinge", "clamp", "bracket", "rivet"]
    for i, w in enumerate(filler_vocab):
        units.append(_mk(f"filler{i}", f"## History\n{w} placeholder text", title=f"Filler {i}",
                         date=f"200{i % 10}-01-01"))
    return units          # 14 + 15 = 29 docs


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


class _BqlToolbox:
    """Call-shape wrapper over a (search, fetch) `ToolBox`, like `DocSearchFetch`."""

    def __init__(self, units, **search_opts):
        units = list(units)
        ubyid = {u.doc_id: u for u in units}
        ex = build_lucene_bql(units)
        state = EpisodeState(question="q")
        self._search = SearchBql(name="search", **search_opts).bind(state, units, ubyid, {"bql": ex})
        self.date_nudge = self._search.date_nudge
        fe = Fetch(name="fetch").bind(state, units, ubyid, {})
        self.box = ToolBox([self._search, fe], state)

    def search(self, query, k=5):
        return self.box.run("search", {"query": query, "k": k})

    def fetch(self, specs):
        return self.box.run("fetch", {"specs": specs})


class _IndriToolbox:
    """Call-shape wrapper over an (isearch, fetch) `ToolBox`, like `IndriFetchWorkspace`."""

    def __init__(self, units, **search_opts):
        units = list(units)
        ubyid = {u.doc_id: u for u in units}
        ex = build_lucene_indri(units)
        state = EpisodeState(question="q")
        self._search = SearchIndri(name="isearch", **search_opts).bind(state, units, ubyid, {"indri": ex})
        self.op_nudge = self._search.op_nudge
        fe = Fetch(name="fetch").bind(state, units, ubyid, {})
        self.box = ToolBox([self._search, fe], state)

    def run(self, name, args):
        return self.box.run(name, args)


# --- 1. condition loading: `research_v2`/`research_indri` were pruned from conditions.yaml
# (paper's 15 kept conditions). The coverage/date-nudge/indri machinery they exercised is
# still tested directly on the underlying classes below (sections 2/3/5/6).


# --- 2. SearchBql(coverage=True): constraint-coverage rendering --------------------

def test_coverage_true_renders_cov_and_miss_on_0_hit_and(units):
    ws = _BqlToolbox(units, coverage=True)
    obs = ws.search("foo[body] AND bar[body] AND baz[body] AND qux[body]")
    assert "CONSTRAINT COVERAGE" in obs
    assert "miss=[" in obs
    # docA matches foo/bar/baz (misses qux): the strongest coverage hit, ranked first
    lines = obs.split("\n")
    assert lines[1].split()[1] == "docA"
    assert "cov=3/4" in lines[1]
    assert "qux" in lines[1]


def test_coverage_hits_are_fetchable_by_rank(units):
    ws = _BqlToolbox(units, coverage=True)
    ws.search("foo[body] AND bar[body] AND baz[body] AND qux[body]")
    out = ws.fetch([[1, "History"]])
    assert "ERROR" not in out
    assert "foo bar baz" in out


def test_coverage_false_reproduces_old_fallback_header_exactly(units):
    ws = _BqlToolbox(units, coverage=False)
    obs = ws.search("foo[body] AND bar[body] AND baz[body] AND qux[body]")
    assert "CONSTRAINT COVERAGE" not in obs
    assert "0 exact matches — showing top" in obs
    assert "CLOSEST docs by term relevance" in obs


# --- 3. SearchIndri: isearch (graded) + fetch ----------------------

def test_isearch_combine_returns_hits_when_no_doc_has_every_term(units):
    ws = _IndriToolbox(units)
    out = ws.run("isearch", {"query": "#combine(alpha bravo charlie delta echo)"})
    assert "ERROR" not in out
    assert "hits):" in out
    assert "(0 hits)" not in out


def test_isearch_unsupported_op_errors_naming_the_op(units):
    ws = _IndriToolbox(units)
    out = ws.run("isearch", {"query": "#prior(RECENT)"})
    assert out.startswith("ERROR:")
    assert "prior" in out.lower()


def test_isearch_date_between_filters(units):
    ws = _IndriToolbox(units)
    out = ws.run("isearch", {
        "query": "#combine(bank management ceremony #date:between(2002-01-01 2002-12-31))",
        "k": 10})
    assert "d2002" in out
    assert "d1999" not in out


def test_isearch_then_fetch_by_rank_works(units):
    ws = _IndriToolbox(units)
    ws.run("isearch", {"query": "#combine(alpha bravo)"})
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert "ERROR" not in out
    assert "§History" in out


def test_isearch_aliases_search_name_too(units):
    ws = _IndriToolbox(units)
    out = ws.run("search", {"query": "#combine(alpha bravo)"})
    assert "ERROR" not in out
    assert "hits):" in out


# --- 5. v2 date-nudge: mechanical, corpus-free mid-episode hint (SearchBql.date_nudge) --

_DATE_HINT = "hint: temporal clues match documents by METADATA date"


def test_date_nudge_true_hints_on_bare_temporal_clue_and_caps_at_3(units):
    ws = _BqlToolbox(units, date_nudge=True)

    out1 = ws.search("founded 2018 ceremony")           # bare year -> nudge #1
    assert _DATE_HINT in out1

    out2 = ws.search("date[2018] ceremony")              # year fully inside date[...] -> no hint
    assert _DATE_HINT not in out2

    out3 = ws.search("released 1985 archive")            # bare year -> nudge #2
    assert _DATE_HINT in out3

    out4 = ws.search("since 1990 archive")                # bare year -> nudge #3
    assert _DATE_HINT in out4

    out5 = ws.search("circa 1975 archive")                # 4th eligible call -> cap reached
    assert _DATE_HINT not in out5


def test_date_nudge_detects_decade_and_month_year_forms(units):
    ws = _BqlToolbox(units, date_nudge=True)
    assert _DATE_HINT in ws.search("history in the 1980s")
    ws2 = _BqlToolbox(units, date_nudge=True)
    assert _DATE_HINT in ws2.search("as of December 2023")


def test_date_nudge_default_false_never_hints(units):
    ws = _BqlToolbox(units)
    assert ws.date_nudge is False
    out = ws.search("founded 2018 ceremony")
    assert _DATE_HINT not in out


def test_date_nudge_false_explicit_never_hints_even_with_coverage(units):
    # coverage=True (the docv2 arm's other kwarg) must not implicitly turn the nudge on:
    # the two are independent knobs; only the docv2 strategy passes both True.
    ws = _BqlToolbox(units, coverage=True, date_nudge=False)
    out = ws.search("founded 2018 ceremony")
    assert _DATE_HINT not in out


# --- 6. indri operator-nudge: mechanical, corpus-free mid-episode hint (op_nudge) -------------

_OP_HINT = "hint: bare keywords work, but structure is sharper"


def test_indri_op_nudge_default_true_hints_on_bare_keywords_and_caps_at_3(units):
    ws = _IndriToolbox(units)
    assert ws.op_nudge is True

    out1 = ws.run("isearch", {"query": "alpha bravo"})           # bare -> nudge #1
    assert _OP_HINT in out1

    out2 = ws.run("isearch", {"query": "#combine(alpha bravo)"})  # has # -> no hint
    assert _OP_HINT not in out2

    out3 = ws.run("isearch", {"query": "charlie delta"})          # bare -> nudge #2
    assert _OP_HINT in out3

    out4 = ws.run("isearch", {"query": "echo filler"})            # bare -> nudge #3
    assert _OP_HINT in out4

    out5 = ws.run("isearch", {"query": "widget gadget"})          # 4th eligible -> cap reached
    assert _OP_HINT not in out5


def test_indri_op_nudge_field_suffix_also_suppresses_hint(units):
    ws = _IndriToolbox(units)
    out = ws.run("isearch", {"query": "alpha.title"})
    assert _OP_HINT not in out


def test_indri_op_nudge_false_never_hints(units):
    ws = _IndriToolbox(units, op_nudge=False)
    out = ws.run("isearch", {"query": "alpha bravo"})
    assert _OP_HINT not in out


# --- 7. hardened skills: research_v2/research_indri were the only conditions carrying these
# imperative RULE blocks, and both were pruned from conditions.yaml; nothing left to pin here.
