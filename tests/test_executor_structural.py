"""The index-free structural executor (the method): Boolean + AST scope over live
units, no index. These tests are the spec for matching semantics."""
import os

from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.executor import StructuralExecutor
from agent_search.corpus.units import CodeUnit, units_from_documents


def _units():
    return [
        CodeUnit("a.py::create_session_token", "a.py", "create_session_token", 4, 6,
                 "def create_session_token(user):\n"
                 "    token = make_token(user)\n"
                 "    return token"),
        CodeUnit("a.py::make_token", "a.py", "make_token", 9, 10,
                 "def make_token(user):\n    return str(user)"),
        CodeUnit("tests/test_a.py::test_token", "tests/test_a.py", "test_token", 1, 2,
                 "def test_token():\n    assert make_token(1)"),
    ]


def _run(q, units=None):
    ex = StructuralExecutor(units or _units())
    r = parse(q)
    assert r.ok, r.error
    return [d for d, _ in ex.run(r.expr)]


def test_term_matches_units_containing_token():
    r = _run("token")
    assert "a.py::create_session_token" in r
    assert "a.py::make_token" in r            # qualname make_token -> 'token'


def test_and_intersects():
    assert _run("AND(session, token)") == ["a.py::create_session_token"]


def test_or_unions():
    r = _run("OR(session, str)")
    assert "a.py::create_session_token" in r   # has 'session'
    assert "a.py::make_token" in r             # has 'str'


def test_not_excludes_within_and():
    r = _run("AND(token, NOT(session))")
    assert "a.py::make_token" in r
    assert "a.py::create_session_token" not in r


def test_in_def_near_func_co_occurrence():
    assert _run("IN(def, NEAR/func(session, token))") == ["a.py::create_session_token"]


def test_not_in_test_file_uses_path_scope():
    r = _run("AND(token, NOT(IN(file, PREFIX(test))))")
    assert "tests/test_a.py::test_token" not in r   # excluded by path
    assert "a.py::make_token" in r


def test_prefix_matches_identifier_variant():
    r = _run("PREFIX(sess)")
    assert "a.py::create_session_token" in r   # 'session' startswith 'sess'
    assert "a.py::make_token" not in r


def test_phrase_requires_adjacency_in_order():
    units = [
        CodeUnit("x.py::f", "x.py", "f", 1, 1, "open file here"),
        CodeUnit("x.py::g", "x.py", "g", 2, 2, "file open here"),
    ]
    r = _run("PHRASE(open, file)", units)
    assert r == ["x.py::f"]                     # 'open file' adjacent only in f


def test_near_window_distance():
    units = [
        CodeUnit("y.py::near", "y.py", "near", 1, 1, "alpha beta gamma delta"),
        CodeUnit("y.py::far", "y.py", "far", 2, 2, "alpha one two three four five delta"),
    ]
    # alpha within 2 tokens of delta only in 'near'... actually neither: tighten to w1
    r = _run("NEAR/w2(beta, gamma)", units)
    assert r == ["y.py::near"]


def test_ranking_prefers_more_matches():
    units = [
        CodeUnit("z.py::many", "z.py", "many", 1, 1, "token token token"),
        CodeUnit("z.py::few", "z.py", "few", 2, 2, "token other"),
    ]
    assert _run("token", units) == ["z.py::many", "z.py::few"]


def test_boolean_candidates_are_bm25_reranked_and_zero_scores_kept():
    units = [
        CodeUnit("pkg/session.py::a", "pkg/session.py", "a", 1, 1, "unrelated"),
        CodeUnit("pkg/session.py::b", "pkg/session.py", "b", 2, 2, "unrelated"),
    ]

    # Both units match only because the file path contains "session". The unit text
    # itself has no "session", so both BM25 rerank scores are zero but the full
    # Boolean candidate set must remain rankable for Recall@k / MRR.
    ex = StructuralExecutor(units)
    r = parse("IN(file, session)")
    assert r.ok, r.error
    ranked, count = ex.run_with_count(r.expr, k=10)

    assert count == 2
    assert [doc for doc, score in ranked] == ["pkg/session.py::a", "pkg/session.py::b"]
    assert [score for doc, score in ranked] == [0.0, 0.0]


def test_index_free_writes_nothing(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        before = set(os.listdir("."))
        _run("AND(session, token)")
        assert set(os.listdir(".")) == before   # no index/files created
    finally:
        os.chdir(cwd)


def test_document_field_scope_title_body_section():
    units = units_from_documents([
        {
            "doc_id": "d1",
            "title": "Treaty of Guadalupe Hidalgo",
            "section": "Mexican-American War",
            "text": "This treaty ended the war in 1848.",
        },
        {
            "doc_id": "d2",
            "title": "Treaty of Paris",
            "section": "Spanish-American War",
            "text": "The Treaty of Guadalupe Hidalgo is mentioned as context.",
        },
    ])

    assert _run("IN(title, hidalgo)", units) == ["d1"]
    assert _run("IN(body, hidalgo)", units) == ["d2"]
    assert _run("AND(IN(section, mexican), IN(body, 1848))", units) == ["d1"]


def test_rerank_scorer_built_once_and_reused():
    """The corpus-level BM25 reranker is built once and reused across queries —
    the fix for the per-query re-index blowup (a broad query over a large corpus
    used to re-tokenize the whole matched set every turn)."""
    ex = StructuralExecutor(_units())
    r1 = parse("make_token")
    assert r1.ok
    ex.run(r1.expr)
    bm_first = ex._bm
    assert bm_first is not None
    r2 = parse("token")
    assert r2.ok
    ex.run(r2.expr)
    assert ex._bm is bm_first   # same object: built once, not per query


def _shape_units():
    """A unit whose call site is `polygon(...)` while `polygons` (plural) lives only in
    the body as a variable — lets us tell region scopes apart in the suggest() tests."""
    return [
        CodeUnit("draw.py::draw_shapes", "draw.py", "draw_shapes", 1, 4,
                 "def draw_shapes():\n"
                 "    polygons = 3\n"
                 "    return polygon(polygons)"),
        CodeUnit("draw.py::helper", "draw.py", "helper", 6, 7,
                 "def helper():\n    return circle()"),
    ]


def test_suggest_misspelling_in_region_names_real_token():
    ex = StructuralExecutor(_shape_units())
    hint = ex.suggest("IN(call, polygns)")    # typo for the called `polygon`
    assert "polygon" in hint
    assert "call" in hint                     # region-qualified


def test_suggest_empty_when_token_exists():
    ex = StructuralExecutor(_shape_units())
    assert ex.suggest("IN(call, polygon)") == ""   # really called -> the 0 result is real
    assert ex.suggest("polygon") == ""             # bare term exists globally


def test_suggest_is_region_aware():
    # `polygons` EXISTS in the corpus (body/doc), so a bare query has nothing to suggest,
    # but it is NOT a call -> querying IN(call, polygons) still grounds to `polygon`.
    ex = StructuralExecutor(_shape_units())
    assert ex.suggest("polygons") == ""                 # present globally, no typo
    hint = ex.suggest("IN(call, polygons)")
    assert "polygon" in hint and "call" in hint


def test_suggest_bare_term_uses_global_scope():
    ex = StructuralExecutor(_shape_units())
    hint = ex.suggest("polygns")              # global near-miss, no region qualifier
    assert "polygon" in hint
    assert "region" not in hint               # bare leaf -> no 'in region `...`'


def test_suggest_unparseable_returns_empty():
    ex = StructuralExecutor(_shape_units())
    assert ex.suggest("AND(") == ""
    assert ex.suggest("") == ""


def test_suggest_skips_negated_terms():
    # an ABSENT excluded token is not a typo to ground; only the positive miss is.
    ex = StructuralExecutor(_shape_units())
    hint = ex.suggest("AND(IN(call, polygns), NOT(nonexistentxyz))")
    assert "polygon" in hint
    assert "nonexistentxyz" not in hint


def test_score_subset_matches_corpus_idf_ranking():
    """score_subset over the matched ids must equal scoring the whole corpus then
    filtering to those ids — same corpus statistics, just restricted."""
    from agent_search.retrievers.ranking import BM25
    docs = {f"d{i}": ["alpha", "beta"] if i % 2 else ["alpha", "gamma", "gamma"]
            for i in range(6)}
    bm = BM25().index_tokenized(docs)
    subset = ["d1", "d3", "d4"]
    want = [(d, s) for d, s in bm.score_terms(["gamma"]) if d in subset]
    want.sort(key=lambda x: (-x[1], x[0]))
    got = bm.score_subset(["gamma"], subset)
    assert [d for d, _ in got] == [d for d, _ in want]
    assert all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(got, want))
