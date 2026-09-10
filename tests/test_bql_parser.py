"""Parser behavior: BQL prefix string -> typed AST (bql_spec.md §1).

Malformed input must return a structured ParseResult error, never raise.
"""
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.ast import (
    Term, And, Or, Not, Near, In, Expand, Phrase, Prefix, Region, Strategy,
)


def test_parses_bare_term():
    r = parse("auth")
    assert r.ok and r.expr == Term("auth")


def test_parses_quoted_term_preserving_spaces():
    r = parse('"auth token"')
    assert r.ok and r.expr == Term("auth token", quoted=True)


def test_parses_and_of_two_terms():
    r = parse("AND(auth, token)")
    assert r.ok and r.expr == And((Term("auth"), Term("token")))


def test_parses_or_with_three_children():
    r = parse("OR(a, b, c)")
    assert r.ok and r.expr == Or((Term("a"), Term("b"), Term("c")))


def test_expand_defaults_to_synonym_strategy():
    r = parse("EXPAND(auth)")
    assert r.ok and r.expr == Expand(Term("auth"), Strategy.SYNONYM)


def test_expand_with_symbol_strategy():
    r = parse("EXPAND(auth, symbol)")
    assert r.ok and r.expr == Expand(Term("auth"), Strategy.SYMBOL)


def test_parses_prefix():
    r = parse("PREFIX(auth)")
    assert r.ok and r.expr == Prefix("auth")


def test_parses_phrase():
    r = parse("PHRASE(open, file)")
    assert r.ok and r.expr == Phrase((Term("open"), Term("file")))


def test_parses_in_region():
    r = parse("IN(def, auth)")
    assert r.ok and r.expr == In(Region.DEF, Term("auth"))


def test_parses_near_named_granularity():
    r = parse("NEAR/func(auth, token)")
    assert r.ok
    n = r.expr
    assert isinstance(n, Near)
    assert n.spec == "func"
    assert n.left == Term("auth") and n.right == Term("token")


def test_parses_near_token_window():
    r = parse("NEAR/w5(auth, token)")
    assert r.ok and r.expr.spec == "w5"


def test_parses_not_inside_and():
    r = parse("AND(auth, NOT(token))")
    assert r.ok and r.expr == And((Term("auth"), Not(Term("token"))))


def test_parses_standalone_not_syntactically():
    # Parser accepts NOT anywhere; placement is a *type* error, not a parse error.
    r = parse("NOT(token)")
    assert r.ok and r.expr == Not(Term("token"))


def test_parses_worked_example_structure():
    q = ("AND(IN(def, NEAR/func(EXPAND(auth, symbol), EXPAND(token, symbol))), "
         "NOT(IN(file, PREFIX(test))))")
    r = parse(q)
    assert r.ok
    top = r.expr
    assert isinstance(top, And) and len(top.children) == 2
    assert isinstance(top.children[1], Not)


def test_unbalanced_parens_is_error_not_exception():
    r = parse("AND(auth, token")
    assert not r.ok and r.error


def test_unknown_operator_is_error():
    r = parse("XOR(a, b)")
    assert not r.ok and r.error


def test_invalid_region_is_error():
    r = parse("IN(nonsense, auth)")
    assert not r.ok and r.error


def test_empty_input_is_error():
    r = parse("   ")
    assert not r.ok and r.error


# --- Unicode + `+`/`#` identifiers (the ident class accepts \w, plus +/# after the first
# char) -- a non-ASCII term used to hard-error ("unexpected character") once the tail
# tokenizer ran out of ASCII characters to consume; `+`/`#`-bearing names (C++, C#) used to
# split into multiple stray tokens with no operator between them. ------------------------

def test_parses_unicode_term():
    r = parse("café")
    assert r.ok and r.expr == Term("café")


def test_parses_cplusplus_term():
    r = parse("C++")
    assert r.ok and r.expr == Term("C++")


def test_parses_csharp_term():
    r = parse("C#")
    assert r.ok and r.expr == Term("C#")


def test_parses_unicode_term_in_region():
    r = parse("IN(title, Zürich)")
    assert r.ok and r.expr == In(Region.TITLE, Term("Zürich"))
