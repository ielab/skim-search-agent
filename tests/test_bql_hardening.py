"""Regressions for the 2026-06-10 parser-hardening audit: the parser
must never raise on adversarial LLM output, the surface syntax must accept the
standard Boolean forms (parens, infix anywhere, case-insensitive operators,
Lucene-style NEAR/5), and the granularity lattice must reject coarser-than-binder
operands instead of silently mistyping them.
"""
import random
import string

from agent_search.retrievers.structural.bql.ast import And, In, Near, Or, Term
from agent_search.retrievers.structural.bql.parser import parse
from agent_search.retrievers.structural.bql.types import Granularity, check


# --- 1. RecursionError must not escape (critical) ----------------------------

def test_deep_nesting_returns_error_not_crash():
    r = parse("NOT(" * 2000 + "x" + ")" * 2000)
    assert not r.ok and "nested" in r.error


def test_deep_nesting_unclosed_also_safe():
    r = parse("AND(" * 5000 + "x")
    assert not r.ok  # either depth error or parse error — never an exception


def test_fuzz_never_raises():
    rng = random.Random(0)
    alphabet = list(string.ascii_letters[:8]) + ["(", ")", ",", '"', "'", " ",
                                                 "AND", "OR", "NOT", "NEAR/w5", "IN"]
    for _ in range(300):
        soup = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        res = parse(soup)            # must return a ParseResult, never raise
        if res.ok:
            check(res.expr)          # and the checker must not raise either


# --- 2. parenthesized grouping ------------------------------------------------

def test_paren_grouping():
    r = parse("(auth OR token) AND handler")
    assert r.ok and isinstance(r.expr, And)
    assert isinstance(r.expr.children[0], Or)


def test_paren_group_comma_is_and():
    r = parse("(a, b) OR c")
    assert r.ok and isinstance(r.expr, Or) and isinstance(r.expr.children[0], And)


# --- 3. infix inside argument lists -------------------------------------------

def test_infix_or_inside_and_args():
    r = parse("AND(foo, bar OR baz)")
    assert r.ok and isinstance(r.expr, And)
    assert isinstance(r.expr.children[1], Or)


def test_infix_or_inside_in_child():
    r = parse("IN(def, a OR b)")
    assert r.ok and isinstance(r.expr, In) and isinstance(r.expr.child, Or)


def test_comma_still_separates_args():
    r = parse("AND(a, b, c)")
    assert r.ok and len(r.expr.children) == 3


# --- 4. NEAR lattice rule -------------------------------------------------------

def test_near_rejects_coarser_operand():
    r = parse("NEAR/w5(NEAR/file(a, b), c)")
    assert r.ok
    t = check(r.expr)
    assert not t.ok and "NEAR/w5" in t.error


def test_in_rejects_coarser_pattern():
    r = parse("IN(comment, NEAR/file(a, b))")
    assert r.ok
    t = check(r.expr)
    assert not t.ok and t.error_node == "IN"


def test_near_same_level_ok():
    r = parse("IN(def, NEAR/func(a, b))")
    assert r.ok and check(r.expr).ok


# --- 5. case-insensitive operators ---------------------------------------------

def test_lowercase_prefix_ops():
    r = parse("and(a, or(b, c))")
    assert r.ok and isinstance(r.expr, And) and isinstance(r.expr.children[1], Or)


def test_lowercase_region_and_strategy():
    assert parse("IN(DEF, foo)").ok
    assert parse("EXPAND(auth, SYMBOL)").ok


def test_bare_keyword_is_a_term():
    r = parse("and")                 # no call form -> plain search term
    assert r.ok and isinstance(r.expr, Term)


# --- 6. NEAR spec forms ----------------------------------------------------------

def test_numeric_near_spec_is_token_window():
    r = parse("NEAR/5(auth, token)")
    assert r.ok and isinstance(r.expr, Near) and r.expr.spec == "w5"
    assert check(r.expr).ok


def test_case_insensitive_near():
    r = parse("near/FUNC(a, b)")
    assert r.ok and isinstance(r.expr, Near)
    assert check(r.expr).level == Granularity.FUNC


def test_single_operand_near_keeps_lift():
    r = parse("NEAR/func(auth)")
    assert r.ok and isinstance(r.expr, Near)       # not collapsed to Term
    assert check(r.expr).level == Granularity.FUNC


# --- 7. string terms ---------------------------------------------------------------

def test_escaped_quote_in_string():
    r = parse(r'"foo\"bar"')
    assert r.ok and isinstance(r.expr, Term) and r.expr.text == 'foo"bar'


def test_empty_quoted_term_rejected():
    assert not parse('""').ok
    assert not parse("AND(a, '')").ok


# --- 8. NEAR/file is file-scope co-occurrence (executor) ----------------------------

def test_in_def_matches_class_via_method_qualname():
    """Observed miss (astropy-14182): IN(def, RST) found nothing although
    rst.py defines `class RST` — methods are the units and their bodies never
    mention the class name. The unit's qualname is part of its definition."""
    from agent_search.retrievers.structural.bql.executor import StructuralExecutor
    from agent_search.corpus.units import CodeUnit
    from agent_search.retrievers.structural.bql.parser import parse

    units = [
        CodeUnit(doc_id="rst.py::RST.write", path="rst.py", qualname="RST.write",
                 code="def write(self, lines):\n    return lines\n",
                 start_line=10, end_line=11),
        CodeUnit(doc_id="other.py::f", path="other.py", qualname="f",
                 code="def f():\n    pass\n", start_line=1, end_line=2),
    ]
    ex = StructuralExecutor(units)
    hits = [d for d, _ in ex.run(parse("IN(def, RST)").expr)]
    assert hits == ["rst.py::RST.write"]


def test_near_file_cooccurs_across_units():
    from agent_search.retrievers.structural.bql.executor import StructuralExecutor
    from agent_search.corpus.units import CodeUnit

    units = [
        CodeUnit(doc_id="m.py::f", path="m.py", qualname="f",
                 code="def f():\n    alpha()\n", start_line=1, end_line=2),
        CodeUnit(doc_id="m.py::g", path="m.py", qualname="g",
                 code="def g():\n    beta()\n", start_line=3, end_line=4),
        CodeUnit(doc_id="o.py::h", path="o.py", qualname="h",
                 code="def h():\n    alpha()\n", start_line=1, end_line=2),
    ]
    ex = StructuralExecutor(units)
    # alpha and beta never share a unit, but they share file m.py
    hits = [d for d, _ in ex.run(parse("NEAR/file(alpha, beta)").expr)]
    assert "m.py::f" in hits and "m.py::g" in hits
    assert "o.py::h" not in hits
    # same-unit co-occurrence (func) still requires the same unit
    assert [d for d, _ in ex.run(parse("NEAR/func(alpha, beta)").expr)] == []


def test_region_bags_survive_col0_multiline_string():
    """A method whose body holds a multiline string with column-0 content defeats
    textwrap.dedent; the wrapper-class re-parse must keep the AST bags populated
    (previously ALL IN(...) regions silently went empty for such units)."""
    from agent_search.retrievers.structural.bql.structure import region_token_bags

    code = '    def write(self, lines):\n        t = """\ncol0 text\n"""\n        return self.render(t)\n'
    bags = region_token_bags(code)
    assert "write" in bags["def"]
    assert "render" in bags["call"]
    assert "wrap" not in bags["def"]          # synthetic wrapper never leaks


def test_attribute_assignment_in_def_bag():
    from agent_search.retrievers.structural.bql.structure import region_token_bags
    bags = region_token_bags("def f(self):\n    self.required_columns = []\n")
    assert "required" in bags["def"] and "columns" in bags["def"]


def _exec_for(src):
    from agent_search.retrievers.structural.bql.executor import StructuralExecutor
    from agent_search.corpus.units import units_from_python_source
    return StructuralExecutor(units_from_python_source("m.py", src))


def test_window_near_honors_or_operand():
    """NEAR/wN(OR(...), x) must enforce the WINDOW, not degrade to unit
    co-occurrence (observed: matched with every pair far apart)."""
    from agent_search.retrievers.structural.bql.parser import parse
    far = ("def f():\n    alpha = 1\n"
           "    a1 = a2 = a3 = a4 = a5 = a6 = a7 = 0\n"
           "    gamma = 2\n"
           "    b1 = b2 = b3 = b4 = b5 = b6 = b7 = 0\n"
           "    beta = 3\n")
    near = "def g():\n    alpha = 1\n    x = 0\n    gamma = beta\n"
    assert [d for d, _ in _exec_for(far).run(parse("NEAR/w2(OR(alpha, gamma), beta)").expr)] == []
    assert [d for d, _ in _exec_for(near).run(parse("NEAR/w2(OR(alpha, gamma), beta)").expr)] == ["m.py::g"]


def test_window_near_expand_is_prefix_consistent():
    from agent_search.retrievers.structural.bql.parser import parse
    src = "def f():\n    authenticate_user = 1\n    x = 0\n    token = 2\n"
    ex = _exec_for(src)
    # EXPAND(auth) matches 'authenticate' by prefix at token distance 5 from 'token'
    assert [d for d, _ in ex.run(parse("NEAR/w5(EXPAND(auth, symbol), token)").expr)] == ["m.py::f"]
    assert [d for d, _ in ex.run(parse("NEAR/w2(EXPAND(auth, symbol), token)").expr)] == []


def test_window_near_phrase_uses_span_coverage():
    from agent_search.retrievers.structural.bql.parser import parse
    src = "def f():\n    make_token = factory\n"
    # 'factory' is adjacent to the END of the make/token span
    assert [d for d, _ in _exec_for(src).run(
        parse("NEAR/w1(PHRASE(make, token), factory)").expr)] == ["m.py::f"]
