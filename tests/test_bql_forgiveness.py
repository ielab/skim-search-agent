"""Parser forgiveness + teaching errors + f-string-robust string region.

Regressions for the 2026-06-19 BQL engine audit (real agent-run failures):
  A. bare multi-word operands -> implicit PHRASE (the model's natural instinct);
     every previously-valid query still parses identically.
  B. genuine parse failures suggest the fix (quote / PHRASE).
  C. IN(string, ...) reliably matches f-string literals on any CPython >= 3.8.
"""
from agent_search.retrievers.bql.ast import (
    And, In, Near, Or, Phrase, Region, Term,
)
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.types import check


# --- A. parser forgiveness: bare multi-word -> implicit PHRASE ----------------

def test_bare_multiword_top_level_is_phrase():
    r = parse("auth token handler")
    assert r.ok
    assert r.expr == Phrase((Term("auth"), Term("token"), Term("handler")))


def test_bare_multiword_inside_in_region():
    r = parse("IN(string, TimeSeries object is invalid)")
    assert r.ok
    assert r.expr == In(
        Region.STRING,
        Phrase((Term("TimeSeries"), Term("object"), Term("is"), Term("invalid"))),
    )


def test_bare_multiword_inside_in_def():
    r = parse("IN(def, class TimeSeries)")
    assert r.ok and isinstance(r.expr, In)
    assert r.expr.child == Phrase((Term("class"), Term("TimeSeries")))


def test_bare_multiword_inside_and_arg():
    r = parse("AND(auth, foo bar)")
    assert r.ok and isinstance(r.expr, And)
    assert r.expr.children[0] == Term("auth")
    assert r.expr.children[1] == Phrase((Term("foo"), Term("bar")))


def test_bare_multiword_inside_near_operand():
    r = parse("NEAR/w5(foo bar, baz)")
    assert r.ok and isinstance(r.expr, Near)
    assert r.expr.left == Phrase((Term("foo"), Term("bar")))
    assert r.expr.right == Term("baz")


def test_implicit_phrase_type_checks_as_token_level():
    assert check(parse("IN(string, TimeSeries object is invalid)").expr).ok


def test_single_bare_word_is_unchanged_term():
    assert parse("auth").expr == Term("auth")


def test_quoted_multiword_stays_single_quoted_term():
    assert parse('"auth token"').expr == Term("auth token", quoted=True)


def test_infix_operators_still_bind_not_swallowed_into_phrase():
    assert parse("auth AND token").expr == And((Term("auth"), Term("token")))
    assert parse("a OR b").expr == Or((Term("a"), Term("b")))
    r = parse("foo bar AND baz")
    assert r.ok and isinstance(r.expr, And)
    assert r.expr.children[0] == Phrase((Term("foo"), Term("bar")))
    assert r.expr.children[1] == Term("baz")


def test_call_keyword_not_swallowed_into_phrase():
    r = parse("AND(foo bar, NOT(baz))")
    assert r.ok and isinstance(r.expr, And)
    assert r.expr.children[0] == Phrase((Term("foo"), Term("bar")))


def test_string_then_bare_word_forms_phrase():
    r = parse('"foo bar" baz')
    assert r.ok
    assert r.expr == Phrase((Term("foo bar", quoted=True), Term("baz")))


# --- B. teaching error messages ----------------------------------------------

def test_genuine_failure_message_suggests_quote_or_phrase():
    r = parse("EXPAND(auth, symbol extra)")
    assert not r.ok
    assert "PHRASE" in r.error and "quote" in r.error


def test_unbalanced_paren_still_errors():
    assert not parse("AND(auth, token").ok
    assert not parse("foo (bar").ok


# --- C. f-string-robust string region ----------------------------------------

def test_in_string_matches_fstring_literal():
    from agent_search.retrievers.bql.structure import region_token_bags
    from agent_search.retrievers.bql.executor import StructuralExecutor
    from agent_search.corpus.units import CodeUnit

    code = (
        "def _check_required_columns(self, col):\n"
        "    if col not in self.colnames:\n"
        "        raise ValueError(\n"
        "            f\"TimeSeries object is invalid - expected '{col}' "
        "as a column but found it missing\"\n"
        "        )\n"
    )
    bags = region_token_bags(code)
    # code_tokenize splits TimeSeries -> time, series (camelCase boundary)
    assert "time" in bags["string"] and "series" in bags["string"]
    assert "invalid" in bags["string"]
    assert bags["string"].count("invalid") == 1   # no double-count

    u = CodeUnit(
        doc_id="ts.py::TimeSeries._check_required_columns", path="ts.py",
        qualname="TimeSeries._check_required_columns", code=code,
        start_line=1, end_line=6,
    )
    ex = StructuralExecutor([u])
    assert len(ex.run(parse('IN(string, "TimeSeries object is invalid")').expr)) >= 1
    assert len(ex.run(parse("IN(string, TimeSeries object is invalid)").expr)) >= 1


def test_fstring_format_spec_literal_text_collected():
    from agent_search.retrievers.bql.structure import region_token_bags
    bags = region_token_bags('x = f"value {n:>{w}d} end"\n')
    assert "value" in bags["string"] and "end" in bags["string"]
    assert "d" in bags["string"]
    assert bags["string"].count("d") == 1          # nested format-spec not double-counted
