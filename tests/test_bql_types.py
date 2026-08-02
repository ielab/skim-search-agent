"""Granularity type checking (bql_spec.md §3).

Term/Phrase/Prefix/Expand are TOKEN-level; NEAR/g and IN(region,.) lift; NOT is
valid only as a clause of AND (set-difference), never standalone / inside OR/NEAR;
an AND of only negations has no positive clause and is rejected.
"""
from agent_search.retrievers.structural.bql.parser import parse
from agent_search.retrievers.structural.bql.types import check
from agent_search.retrievers.structural.bql.ast import Granularity


def _ast(q):
    r = parse(q)
    assert r.ok, r.error
    return r.expr


def test_bare_term_is_token_level():
    res = check(_ast("auth"))
    assert res.ok and res.level == Granularity.TOKEN


def test_or_of_terms_is_token_level():
    res = check(_ast("OR(a, b)"))
    assert res.ok and res.level == Granularity.TOKEN


def test_in_def_lifts_to_func_level():
    res = check(_ast("IN(def, auth)"))
    assert res.ok and res.level == Granularity.FUNC


def test_in_comment_lifts_to_block_level():
    res = check(_ast("IN(comment, todo)"))
    assert res.ok and res.level == Granularity.BLOCK


def test_near_func_lifts_to_func_level():
    res = check(_ast("NEAR/func(auth, token)"))
    assert res.ok and res.level == Granularity.FUNC


def test_not_inside_and_is_ok():
    res = check(_ast("AND(auth, NOT(token))"))
    assert res.ok


def test_standalone_not_is_rejected():
    res = check(_ast("NOT(token)"))
    assert not res.ok and res.error


def test_not_inside_or_is_rejected():
    res = check(_ast("OR(auth, NOT(token))"))
    assert not res.ok and res.error


def test_and_of_only_negations_is_rejected():
    res = check(_ast("AND(NOT(a), NOT(b))"))
    assert not res.ok and res.error


def test_worked_example_typechecks_ok():
    q = ("AND(IN(def, NEAR/func(EXPAND(auth, symbol), EXPAND(token, symbol))), "
         "NOT(IN(file, PREFIX(test))))")
    res = check(_ast(q))
    assert res.ok and res.level == Granularity.FUNC
