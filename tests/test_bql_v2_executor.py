"""BQL v2 executor-layer features: typed date-range queries and constraint-coverage
ranking (see the surface.py / executor.py module docstrings for the design rationale).

Both features address measured BrowseComp-Plus deficits: (1) temporal clues ("in the
1980s") were inexpressible against ISO `date` metadata — only exact-token matching; (2) a
hard multi-constraint AND that matches nothing gave the agent no signal about WHICH
constraint failed, only `soft_topk`'s undifferentiated bag-of-terms fallback.

CPU-only; a small (~30-unit) synthetic corpus with `metadata={'date': ...}`.
"""
from __future__ import annotations

import os

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.bql import executor as execmod
from agent_search.retrievers.bql.executor import StructuralExecutor
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.surface import to_bql
from agent_search.retrievers.bql.types import check


@pytest.fixture(autouse=True)
def _restore_gate_and_threshold():
    """Isolate the env gate + the module-global prefilter threshold from other test
    files (same pattern as tests/test_bql_prefilter.py's `_restore_threshold`)."""
    gate = os.environ.pop("BQL_DATE_RANGE", None)
    threshold = execmod._PREFILTER_MIN_UNITS
    yield
    if gate is None:
        os.environ.pop("BQL_DATE_RANGE", None)
    else:
        os.environ["BQL_DATE_RANGE"] = gate
    execmod._PREFILTER_MIN_UNITS = threshold


def _mk(doc_id: str, code: str, date: str | None = None) -> CodeUnit:
    meta = {"date": date} if date is not None else {}
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.py", qualname=doc_id,
                    start_line=1, end_line=1, code=code, metadata=meta)


def _corpus() -> list[CodeUnit]:
    units = [
        # --- date-range boundary docs (also double as the "alpha" combined-AND fixture) ---
        _mk("d1979", "# alpha before", date="1979-12-31"),
        _mk("d1980", "# alpha instart", date="1980-01-01"),
        _mk("d1989", "# alpha inend", date="1989-12-31"),
        _mk("d1990", "# alpha after", date="1990-01-01"),
        _mk("d1985", "# gamma mid", date="1985-06-15"),
        _mk("dmissing", "# alpha nodatehere"),                       # no 'date' key at all
        _mk("dmalformed", "# alpha baddate", date="13/25/2002"),     # not ISO / not a real date
        # --- constraint-coverage fixture: AND(foo, bar, baz, qux) 0-hits ---
        _mk("docA", "# foo bar baz", date="2001-01-01"),             # matches foo,bar,baz -> 3/4
        _mk("docB", "# foo bar", date="2001-01-02"),                 # matches foo,bar -> 2/4
        _mk("docC", "# nothing relevant here", date="2001-01-03"),   # matches none -> 0/4 (absent)
        _mk("docD", "# foo only", date="2001-01-04"),                # matches foo -> 1/4
    ]
    # filler docs: pad toward ~30 units, distinct vocab so they never contaminate the
    # date/coverage assertions above.
    filler_vocab = ["widget", "gadget", "sprocket", "lever", "cog", "gear", "pulley",
                     "spring", "bolt", "nut", "washer", "hinge", "clamp", "bracket",
                     "rivet", "gasket", "valve", "piston", "rotor", "flange"]
    for i, w in enumerate(filler_vocab):
        units.append(_mk(f"filler{i}", f"# {w} placeholder text", date=f"200{i % 10}-01-01"))
    return units


def _run(ex: StructuralExecutor, bql: str, k: int = 50):
    r = parse(bql)
    assert r.ok, r.error
    t = check(r.expr)
    assert t.ok, t.error
    return ex.run_with_count(r.expr, k=k)


# --- 1. surface: date[RANGE] lowers to a parseable, round-tripping BQL string -----------

def test_surface_date_range_round_trips():
    bql = to_bql("treaty date[1980..1989]", domain="doc")
    r = parse(bql)
    assert r.ok, r.error
    t = check(r.expr)
    assert t.ok, t.error
    assert "__daterange__1980-01-01__1989-12-31" in bql


@pytest.mark.parametrize("spec,lo,hi", [
    ("1980..1989", "1980-01-01", "1989-12-31"),
    ("2002", "2002-01-01", "2002-12-31"),
    ("<2023-12", None, "2023-11-30"),
    (">=2019-06", "2019-06-01", None),
    ("2019-06..2021", "2019-06-01", "2021-12-31"),
])
def test_surface_date_range_encodings(spec, lo, hi):
    bql = to_bql(f"date[{spec}]", domain="doc")
    expect = f"__daterange__{lo or 'open'}__{hi or 'open'}"
    assert bql == f"IN(date, {expect})"
    r = parse(bql)
    assert r.ok and check(r.expr).ok


# --- 2. date semantics against unit.metadata['date'] ------------------------------------

@pytest.fixture(scope="module")
def units():
    return _corpus()


@pytest.fixture()
def ex(units):
    return StructuralExecutor(list(units))


def test_date_range_matches_exactly_the_bounded_docs(ex):
    bql = to_bql("date[1980..1989]", domain="doc")
    hits, n = _run(ex, bql)
    ids = {d for d, _ in hits}
    # d1985 (1985-06-15) also legitimately falls inside [1980-01-01, 1989-12-31].
    assert ids == {"d1980", "d1989", "d1985"}
    assert n == 3


def test_date_lt_matches_only_before(ex):
    bql = to_bql("date[<1980]", domain="doc")
    hits, n = _run(ex, bql)
    assert {d for d, _ in hits} == {"d1979"}
    assert n == 1


def test_bare_year_matches_that_years_doc(ex):
    bql = to_bql("date[1985]", domain="doc")
    hits, n = _run(ex, bql)
    assert {d for d, _ in hits} == {"d1985"}
    assert n == 1


def test_missing_and_malformed_dates_never_match(ex):
    # a range wide enough to span the whole corpus's valid dates
    bql = to_bql("date[1900..2100]", domain="doc")
    hits, n = _run(ex, bql)
    ids = {d for d, _ in hits}
    assert "dmissing" not in ids
    assert "dmalformed" not in ids
    # and every id that DID match really does have a parseable ISO date
    assert "d1980" in ids and "d1989" in ids


def test_combined_and_intersects_correctly(ex):
    bql = to_bql("alpha AND date[1980..1989]", domain="doc")
    hits, n = _run(ex, bql)
    ids = {d for d, _ in hits}
    assert ids == {"d1980", "d1989"}
    assert n == 2


def test_date_range_matches_under_the_prefilter_too(units):
    """Force the inverted-index prefilter on (tiny threshold) so the lazy sorted date
    index's binary-search candidates path (StructuralExecutor._date_range_indices) is
    exercised, not just the plain live scan — results must be identical either way."""
    execmod._PREFILTER_MIN_UNITS = 0
    e = StructuralExecutor(list(units))
    assert e._prefilter_on is True
    bql = to_bql("alpha AND date[1980..1989]", domain="doc")
    hits, n = _run(e, bql)
    assert {d for d, _ in hits} == {"d1980", "d1989"}
    assert n == 2


# --- 3. gate: BQL_DATE_RANGE=0 falls back to the OLD passthrough-field lowering ---------

def test_gate_disables_date_range_lowering(monkeypatch):
    monkeypatch.setenv("BQL_DATE_RANGE", "1")
    on = to_bql("date[1980..1989]", domain="doc")
    assert "__daterange__" in on

    monkeypatch.setenv("BQL_DATE_RANGE", "0")
    off = to_bql("date[1980..1989]", domain="doc")
    assert "__daterange__" not in off
    # OLD mechanism: "date" read as a bare term, "1980..1989" as an (unknown) passthrough
    # field -- exactly what to_bql produced for this input before this feature existed.
    assert off == "IN(1980..1989, date)"


def test_gate_off_leaves_old_field_tag_form_untouched(monkeypatch):
    monkeypatch.setenv("BQL_DATE_RANGE", "0")
    # the OLD, still-supported "value[date]" exact-token form is unaffected either way
    assert to_bql("1980s[date]", domain="doc") == "IN(date, 1980s)"


# --- 4. coverage_topk: constraint-coverage ranking for a 0-hit AND ----------------------

@pytest.fixture()
def coverage_expr():
    r = parse("AND(foo, bar, baz, qux)")
    assert r.ok, r.error
    assert check(r.expr).ok
    return r.expr


def test_coverage_topk_ranks_by_children_matched(ex, coverage_expr):
    # sanity: the exact AND really is a 0-hit query (the motivating scenario)
    hits, n = ex.run_with_count(coverage_expr, k=50)
    assert n == 0

    rows = ex.coverage_topk(coverage_expr, k=5)
    ids = [r[0] for r in rows]
    assert ids[0] == "docA"
    assert ids[1] == "docB"
    assert "docC" not in ids                 # 0-matched docs are dropped, not just ranked last

    by_id = {r[0]: r for r in rows}
    doc_id, mask, n_matched, score = by_id["docA"]
    assert n_matched == 3 == sum(mask)
    assert len(mask) == 4                    # aligned with the AND's 4 children
    doc_id, mask, n_matched, score = by_id["docB"]
    assert n_matched == 2 == sum(mask)
    # coverage-desc ordering: docA's count must exceed docB's
    assert by_id["docA"][2] > by_id["docB"][2]


def test_coverage_topk_mask_alignment(ex, coverage_expr):
    rows = ex.coverage_topk(coverage_expr, k=10)
    by_id = {r[0]: r for r in rows}
    # docA matches foo(0), bar(1), baz(2), not qux(3)
    assert by_id["docA"][1] == (True, True, True, False)
    # docD matches only foo(0)
    assert by_id["docD"][1] == (True, False, False, False)


def test_coverage_topk_single_term_degrades_to_soft_topk(ex):
    r = parse("foo")
    assert r.ok
    rows = ex.coverage_topk(r.expr, k=5)
    soft = ex.soft_topk(["foo"], k=5)
    assert [row[0] for row in rows] == [d for d, _ in soft]
    for doc_id, mask, n_matched, score in rows:
        assert mask == (True,)
        assert n_matched == 1


def test_coverage_topk_not_child_counts_as_matched_when_negation_holds():
    # docE has foo, bar but NOT baz -> AND(foo, bar, NOT(baz)) should be a 3/3 full match
    units = [
        _mk("docE", "# foo bar", date="2005-01-01"),
        _mk("docF", "# foo bar baz", date="2005-01-02"),   # NOT(baz) fails here -> 2/3
    ]
    e = StructuralExecutor(units)
    r = parse("AND(foo, bar, NOT(baz))")
    assert r.ok and check(r.expr).ok
    rows = e.coverage_topk(r.expr, k=5)
    by_id = {row[0]: row for row in rows}
    assert by_id["docE"][2] == 3 and all(by_id["docE"][1])
    assert by_id["docF"][2] == 2


# --- 5. slim-pickle path: save() then load()+attach_units() answers identically --------

def test_slim_pickle_date_range_matches_fresh(units, tmp_path):
    fresh = StructuralExecutor(list(units))
    bql = to_bql("alpha AND date[1980..1989]", domain="doc")
    fresh_hits, fresh_n = _run(fresh, bql)

    path = str(tmp_path / "bql_index.pkl")
    StructuralExecutor(list(units)).save(path)
    loaded = StructuralExecutor.load(path)
    loaded.attach_units(list(units))
    loaded_hits, loaded_n = _run(loaded, bql)

    assert loaded_hits == fresh_hits
    assert loaded_n == fresh_n == 2
