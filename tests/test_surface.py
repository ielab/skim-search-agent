"""The BQL field-tagged Boolean surface (retrievers/bql/surface.to_bql).

The agent-facing surface for BOTH arms — `term[field]`, AND/OR/NOT, wildcard*, "phrase" —
lowers to a BQL string the EXISTING parser/typechecker/executor run unchanged. These tests
pin the lowering shape and that every lowered query parses + type-checks (so the surface can
never advertise a form the engine rejects)."""
import pytest

from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.surface import CODE_FIELDS, DOC_FIELDS, to_bql
from agent_search.retrievers.bql.types import check


def _valid(bql: str) -> bool:
    r = parse(bql)
    return r.ok and check(r.expr).ok


# --- lowering shape (the exact translation) ---------------------------------

@pytest.mark.parametrize("query,expected", [
    ("isnan[call]", "IN(call, isnan)"),
    ("sqlmigrate[def]", "IN(def, sqlmigrate)"),
    ('"cannot rollback DDL"[string]', 'IN(string, "cannot rollback DDL")'),
    ("save[def] NOT test[file]", "IN(def, save) AND NOT(IN(file, test))"),
    ("seri*[def]", "IN(def, PREFIX(seri))"),
    ("open[def] OR connect[def]", "IN(def, open) OR IN(def, connect)"),
    ("melanoma[title,body]", "OR(IN(title, melanoma), IN(body, melanoma))"),
])
def test_code_surface_lowers_to_expected_bql(query, expected):
    assert to_bql(query, domain="code") == expected


def test_binary_not_becomes_and_not():
    # Field-tagged binary NOT is set difference; BQL's NOT is unary-in-AND.
    assert to_bql("a[def] NOT b[file]", "code") == "IN(def, a) AND NOT(IN(file, b))"


# --- regressions: adversarial-verification MEDIUM finding -------------------
# `-term` silently dropped its negation (the leading `-` matched no tokenizer
# alternative and was skipped, leaving the bare positive term in the query --
# a semantic INVERSION, not just data loss), and `A AND NOT B` mistranslated to
# the malformed, unparseable `A AND AND NOT(B)` (an explicit AND already present,
# plus the unconditional "AND " prepended for the bare-NOT-keyword synthesis).

def test_dash_prefix_negates_a_single_term_live_repro():
    """Live repro: `alpha -gamma` must exclude `gamma`, not silently include it as a
    positive conjunct (the exact adversarial-verification finding)."""
    bql = to_bql("alpha -gamma", "code")
    assert bql == "AND(alpha, NOT(gamma))"
    r = parse(bql)
    assert r.ok and check(r.expr).ok


def test_dash_prefix_negation_with_explicit_and():
    assert to_bql("alpha AND -gamma", "code") == "alpha AND NOT(gamma)"


def test_dash_prefix_negates_quoted_phrase():
    bql = to_bql('alpha -"exact phrase"', "code")
    assert bql == 'AND(alpha, NOT("exact phrase"))'
    assert _valid(bql)


def test_dash_prefix_multiple_negations_in_one_run():
    assert to_bql("alpha -gamma -delta", "code") == "AND(alpha, NOT(gamma), NOT(delta))"


def test_dash_prefix_does_not_affect_internal_hyphens():
    # "well-known" is ONE word with an internal hyphen -- must not be misread as a
    # negation of "known" (the `word` pattern already allows internal `-`; only a
    # LEADING `-` on an atom triggers negation).
    assert to_bql("well-known", "code") == "well-known"


def test_explicit_and_not_does_not_double_up_live_repro():
    """Live repro: `A AND NOT B` must lower to `A AND NOT(B)`, not the malformed
    `A AND AND NOT(B)` (the exact adversarial-verification finding)."""
    bql = to_bql("save[def] AND NOT test[file]", "code")
    assert bql == "IN(def, save) AND NOT(IN(file, test))"
    assert "AND AND" not in bql
    r = parse(bql)
    assert r.ok and check(r.expr).ok


def test_explicit_and_not_matches_binary_not_shape():
    """`A AND NOT B` and the shorthand `A NOT B` must lower to the SAME BQL."""
    assert to_bql("save[def] AND NOT test[file]", "code") == \
        to_bql("save[def] NOT test[file]", "code")


def test_unquoted_multiword_before_field_is_anded_per_word():
    # a bare multi-word entity name typed before [field] -> AND of the field-scoped terms
    # (forgiving; a strict adjacent phrase 0-hits too often).
    assert to_bql("Command handle[def]", "code") == "AND(IN(def, Command), IN(def, handle))"


def test_quoted_span_before_field_stays_a_phrase():
    assert to_bql('"Command handle"[def]', "code") == 'IN(def, "Command handle")'


def test_grouping_and_infix_pass_through():
    assert to_bql("(open[def] OR connect[def]) AND socket[file]", "code") == \
        "(IN(def, open) OR IN(def, connect)) AND IN(file, socket)"


# --- every advertised field lowers to a valid, type-checked query -----------

def test_all_code_fields_lower_valid():
    for alias in CODE_FIELDS:
        assert _valid(to_bql(f"x[{alias}]", "code")), f"code field {alias!r} did not lower valid"


def test_all_doc_fields_lower_valid():
    for alias in DOC_FIELDS:
        assert _valid(to_bql(f"x[{alias}]", "doc")), f"doc field {alias!r} did not lower valid"


@pytest.mark.parametrize("query", [
    "isnan[call]", "save[def] NOT test[file]", '"could not convert"[string]',
    "seri*[def]", "handle[def] OR connect[def]", "Logger[call] NOT test[file]",
])
def test_code_examples_parse_and_typecheck(query):
    assert _valid(to_bql(query, "code")), f"{query!r} did not lower to valid BQL"


@pytest.mark.parametrize("query", [
    "treaty[title]", "founder[infobox]", "history[section]", "film[title,body]",
    "munoz[title] OR muñoz[title]", 'harbor[title] AND film[body]',
])
def test_doc_examples_parse_and_typecheck(query):
    assert _valid(to_bql(query, "doc")), f"{query!r} did not lower to valid BQL"


# --- unknown field: passed through so the type checker rejects it -----------

def test_unknown_field_is_passed_through_and_rejected():
    bql = to_bql("x[module]", "code")            # 'module' is not a real region/field
    assert "module" in bql                        # passthrough (not silently dropped)
    r = parse(bql)
    assert (not r.ok) or (not check(r.expr).ok)   # the executor rejects it -> agent feedback


def test_empty_query_lowers_to_empty_string():
    assert to_bql("", "code") == ""
    assert to_bql("   ", "doc") == ""


# --- Unicode words: the tokenizer must not silently DROP a non-ASCII letter mid-word -----
#
# `munoz[title] OR muñoz[title]` above only checks that the lowering still PARSES/TYPECHECKS
# -- it would pass even if `muñoz` silently split into two stray words ("mu"/"oz", the `ñ`
# dropped) since `AND(IN(title, mu), IN(title, oz))` is still valid BQL, just semantically
# WRONG. These pin the exact shape: one field-tagged term, not several.

def test_unicode_word_lowers_to_a_single_title_term():
    assert to_bql("Zürich[title]", domain="doc") == "IN(title, Zürich)"


def test_unicode_word_survives_a_multiword_run():
    assert to_bql("café culture[title]", domain="doc") == \
        "AND(IN(title, café), IN(title, culture))"
