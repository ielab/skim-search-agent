"""The BQL inverted-index prefilter (filter-then-verify) for large corpora.

The prefilter narrows the live O(N) scan to a recall-safe candidate subset, then the
exact `_eval` verifies — so results must be BYTE-IDENTICAL to the plain live scan,
only faster. These tests pin that equivalence and that it actually narrows.
"""
import random

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql import executor as ex
from agent_search.retrievers.structural.bql.parser import parse
from agent_search.retrievers.structural.bql.types import check


def _corpus(n=400, seed=0):
    rng = random.Random(seed)
    vocab = ("warn session token expire create class TimeSeries error value raise "
             "foo bar baz alpha beta cookie").split()
    units = []
    for i in range(n):
        f = f"f{i % 9}.py"
        body = " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 14)))
        units.append(CodeUnit(f"{f}::u{i}", f, f"u{i}", 1, 3,
                              f"def u{i}():\n  # {body}\n  warn({rng.choice(vocab)})"))
    return units


# every operator shape — the prefilter must agree with the live scan on all of them
_QUERIES = [
    "warn", "TimeSeries", "session token", "AND(session, token)", "OR(warn, error)",
    "AND(warn, NOT(error))", "PHRASE(create, session)", "PREFIX(sess)",
    "EXPAND(token, symbol)", "NEAR/w3(session, token)", "NEAR/line2(warn, error)",
    "NEAR/func(session, token)", "NEAR/file(session, token)", "IN(string, session)",
    "IN(comment, warn)", "IN(call, warn)", "IN(def, foo)", "IN(file, session)",
    "IN(file, NEAR/func(session, token))", "AND(OR(session, cookie), warn)",
    "AND(IN(call, warn), NOT(IN(file, PREFIX(test))))",
    "IN(def, NEAR/func(session, token))",
]


@pytest.fixture(autouse=True)
def _restore_threshold():
    """These tests flip the module-global threshold; always restore it so the
    setting can't leak into other test files."""
    orig = ex._PREFILTER_MIN_UNITS
    yield
    ex._PREFILTER_MIN_UNITS = orig


@pytest.fixture(scope="module")
def units():
    return _corpus()


def _run(units, query, prefilter_min):
    ex._PREFILTER_MIN_UNITS = prefilter_min
    r = parse(query)
    assert r.ok and check(r.expr).ok, (query, r.error)
    return ex.StructuralExecutor(units).run(r.expr, k=1000)


@pytest.mark.parametrize("query", _QUERIES)
def test_prefilter_result_identical_to_live_scan(units, query):
    live = _run(units, query, 10 ** 9)     # prefilter OFF (scan all units)
    pref = _run(units, query, 0)           # prefilter ON  (threshold 0)
    assert pref == live                    # identical ids AND identical order/scores


def test_prefilter_actually_narrows_for_a_selective_query():
    """A distinctive single term should produce a candidate set far smaller than N
    (otherwise the prefilter is a no-op and there's no speedup)."""
    units = _corpus()
    ex._PREFILTER_MIN_UNITS = 0
    e = ex.StructuralExecutor(units)
    cand = e._candidate_units(parse("TimeSeries").expr)
    assert 0 < len(cand) < len(units)      # narrowed, not all-or-nothing


def test_prefilter_handles_sig_region_numeric_normalization():
    """`sig` region bags include ast.unparse'd annotations, which NORMALIZE numeric
    literals (0xFF -> "255"), so that token isn't in the source-token postings. The
    prefilter must NOT narrow code AST regions, or it would drop the match."""
    from agent_search.corpus.units import units_from_python_source
    u = units_from_python_source("x.py", "def f(a: 0xFF) -> 1_000:\n    return a")
    for q in ("IN(sig, 255)", "IN(sig, 1000)"):
        assert _run(u, q, 0) == _run(u, q, 10 ** 9)   # prefilter == live scan (1 hit)
        assert _run(u, q, 0)                          # and it's actually a hit


def test_prefilter_handles_divergent_document_fields():
    """A custom CodeUnit whose title/body/section carry a token NOT in code+path must
    still be found via IN(field, ...) under the prefilter (the field bags are indexed)."""
    u = [CodeUnit("p::f", "p", "f", 1, 1, "plaincode",
                  title="uniqtitle", body="uniqbody", section="uniqsection")]
    for q in ("IN(title, uniqtitle)", "IN(body, uniqbody)", "IN(section, uniqsection)"):
        assert _run(u, q, 0) == _run(u, q, 10 ** 9) == [("p::f", 0.0)], q


def test_small_corpus_keeps_the_plain_live_scan(units):
    """Below the threshold the prefilter is off — the index-free path is untouched."""
    ex._PREFILTER_MIN_UNITS = 5000
    e = ex.StructuralExecutor(units)       # 400 units < 5000
    assert e._prefilter_on is False
    assert e._candidate_units(parse("warn").expr) is e.units   # scans everything
