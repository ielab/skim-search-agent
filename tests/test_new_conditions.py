"""New, ADDITIVE-only agent conditions: `research_v2` (BQL v2 — typed date[RANGE] ranges +
constraint-coverage feedback) and `research_indri` (the Indri graded query-language backend).

CPU-only; a small (~25-doc) synthetic corpus with dates/sections. The existing `research`
condition/toolset/DocSearchFetch default behavior must be byte-identical after this change —
several assertions below pin that explicitly.
"""
from __future__ import annotations

import pytest

from agent_search.agent.tools.doc_indri import IndriFetchWorkspace
from agent_search.agent.tools.doc_research import DocSearchFetch
from agent_search.corpus.units import CodeUnit


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
    return units          # 14 + 15 = 29 docs, >= the ~25 asked for


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


# --- 1. condition loading: `research_v2`/`research_indri` were pruned from conditions.yaml
# (paper's 15 kept conditions) — the coverage/date-nudge/indri machinery they exercised is
# still tested directly on the underlying classes below (sections 2/3/5/6).


# --- 2. DocSearchFetch(coverage=True): constraint-coverage rendering --------------------

def test_coverage_true_renders_cov_and_miss_on_0_hit_and(units):
    ws = DocSearchFetch(units, coverage=True)
    obs = ws.search("foo[body] AND bar[body] AND baz[body] AND qux[body]")
    assert "CONSTRAINT COVERAGE" in obs
    assert "miss=[" in obs
    # docA matches foo/bar/baz (misses qux) -> the strongest coverage hit, ranked first
    lines = obs.split("\n")
    assert lines[1].split()[1] == "docA"
    assert "cov=3/4" in lines[1]
    assert "qux" in lines[1]


def test_coverage_hits_are_fetchable_by_rank(units):
    ws = DocSearchFetch(units, coverage=True)
    ws.search("foo[body] AND bar[body] AND baz[body] AND qux[body]")
    out = ws.fetch([[1, "History"]])
    assert "ERROR" not in out
    assert "foo bar baz" in out


def test_coverage_false_reproduces_old_fallback_header_exactly(units):
    ws = DocSearchFetch(units, coverage=False)
    obs = ws.search("foo[body] AND bar[body] AND baz[body] AND qux[body]")
    assert "CONSTRAINT COVERAGE" not in obs
    assert "0 exact matches — showing top" in obs
    assert "CLOSEST docs by term relevance" in obs


# --- 3. IndriFetchWorkspace: isearch (graded, never 0-hit) + fetch ----------------------

def test_isearch_combine_never_hard_zeros_on_no_full_match(units):
    ws = IndriFetchWorkspace(units)
    out = ws.run("isearch", {"query": "#combine(alpha bravo charlie delta echo)"})
    assert "ERROR" not in out
    assert "hits):" in out
    assert "weakest constraint for top hit:" in out


def test_isearch_unsupported_op_errors_naming_the_op(units):
    ws = IndriFetchWorkspace(units)
    out = ws.run("isearch", {"query": "#prior(RECENT)"})
    assert out.startswith("ERROR:")
    assert "prior" in out.lower()


def test_isearch_date_between_filters(units):
    ws = IndriFetchWorkspace(units)
    out = ws.run("isearch", {
        "query": "#combine(bank management ceremony #date:between(2002-01-01 2002-12-31))",
        "k": 10})
    assert "d2002" in out
    assert "d1999" not in out


def test_isearch_then_fetch_by_rank_works(units):
    ws = IndriFetchWorkspace(units)
    ws.run("isearch", {"query": "#combine(alpha bravo)"})
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert "ERROR" not in out
    assert "§History" in out


def test_isearch_aliases_search_name_too(units):
    ws = IndriFetchWorkspace(units)
    out = ws.run("search", {"query": "#combine(alpha bravo)"})
    assert "ERROR" not in out
    assert "hits):" in out


# --- 5. v2 date-nudge: mechanical, corpus-free mid-episode hint (DocSearchFetch.date_nudge) --

_DATE_HINT = "hint: temporal clues match documents by METADATA date"


def test_date_nudge_true_hints_on_bare_temporal_clue_and_caps_at_3(units):
    ws = DocSearchFetch(units, date_nudge=True)

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
    ws = DocSearchFetch(units, date_nudge=True)
    assert _DATE_HINT in ws.search("history in the 1980s")
    ws2 = DocSearchFetch(units, date_nudge=True)
    assert _DATE_HINT in ws2.search("as of December 2023")


def test_date_nudge_default_false_never_hints(units):
    ws = DocSearchFetch(units)
    assert ws.date_nudge is False
    out = ws.search("founded 2018 ceremony")
    assert _DATE_HINT not in out


def test_date_nudge_false_explicit_never_hints_even_with_coverage(units):
    # coverage=True (the docv2 arm's OTHER kwarg) must not implicitly turn the nudge on —
    # the two are independent knobs; only retriever.py's docv2 branch passes both True.
    ws = DocSearchFetch(units, coverage=True, date_nudge=False)
    out = ws.search("founded 2018 ceremony")
    assert _DATE_HINT not in out


# --- 6. indri operator-nudge: mechanical, corpus-free mid-episode hint (op_nudge) -------------

_OP_HINT = "hint: bare keywords work, but structure is sharper"


def test_indri_op_nudge_default_true_hints_on_bare_keywords_and_caps_at_3(units):
    ws = IndriFetchWorkspace(units)
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
    ws = IndriFetchWorkspace(units)
    out = ws.run("isearch", {"query": "alpha.title"})
    assert _OP_HINT not in out


def test_indri_op_nudge_false_never_hints(units):
    ws = IndriFetchWorkspace(units, op_nudge=False)
    out = ws.run("isearch", {"query": "alpha bravo"})
    assert _OP_HINT not in out


# --- 7. hardened skills: research_v2/research_indri were the only conditions carrying these
# imperative RULE blocks, and both were pruned from conditions.yaml — nothing left to pin here.
