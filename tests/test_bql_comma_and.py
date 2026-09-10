"""Top-level comma list is implicit AND (surface sugar models reach for):
`A, B` parses as AND(A, B). Genuine trailing junk is still rejected."""
from agent_search.retrievers.bql.ast import And
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.types import check


def test_top_level_comma_is_implicit_and():
    r = parse("IN(def, OR(RST, write)), IN(string, header_rows)")
    assert r.ok and isinstance(r.expr, And) and len(r.expr.children) == 2
    assert check(r.expr).ok


def test_three_comma_clauses():
    r = parse("a, b, c")
    assert r.ok and isinstance(r.expr, And) and len(r.expr.children) == 3


def test_single_clause_is_not_wrapped():
    r = parse("IN(def, foo)")
    assert r.ok and not isinstance(r.expr, And)


def test_trailing_junk_still_errors():
    assert not parse("AND(a, b) xyz").ok


# --- infix AND / OR (canonical is prefix; this is surface tolerance) --------

def test_infix_and():
    r = parse("IN(def, TimeSeries) AND IN(call, remove_column)")
    assert r.ok and isinstance(r.expr, And) and len(r.expr.children) == 2
    assert check(r.expr).ok


def test_infix_and_with_not():
    r = parse("IN(call, identify_format) AND NOT(IN(file, PREFIX(test)))")
    assert r.ok and check(r.expr).ok          # NOT is a valid clause inside the AND


def test_infix_or():
    from agent_search.retrievers.bql.ast import Or
    r = parse("IN(def, write) OR IN(call, write)")
    assert r.ok and isinstance(r.expr, Or) and len(r.expr.children) == 2


def test_or_binds_looser_than_and():
    from agent_search.retrievers.bql.ast import Or
    r = parse("a AND b OR c")                  # (a AND b) OR c
    assert r.ok and isinstance(r.expr, Or)
    assert isinstance(r.expr.children[0], And)
