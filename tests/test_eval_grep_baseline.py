"""The index-free grep baseline (GrepRAG-style): keyword grep -> BM25 rerank, no index."""
from agent_search.evaluation.datasets import fixture_instances
from agent_search.retrievers.lexical.grep import GrepBaseline
from agent_search.evaluation.run_eval import evaluate
from agent_search.corpus.units import CodeUnit


def _units():
    return [
        CodeUnit("a.py::create_session_token", "a.py", "create_session_token", 1, 3,
                 "def create_session_token(user):\n    token = make_token(user)\n    return token"),
        CodeUnit("a.py::render_page", "a.py", "render_page", 5, 6,
                 "def render_page(title, body):\n    return title + body"),
    ]


def test_grep_keyword_match_then_rerank():
    g = GrepBaseline().index(_units())
    r = g.search("session token expiry", k=10)
    assert r[0] == "a.py::create_session_token"   # only it matches session/token


def test_grep_returns_nothing_when_no_keyword_hits():
    g = GrepBaseline().index(_units())
    assert g.search("kubernetes deployment yaml", k=10) == []


def test_grep_is_index_free(tmp_path):
    import os
    os.chdir(tmp_path)
    before = set(os.listdir("."))
    GrepBaseline().index(_units()).search("token", k=5)
    assert set(os.listdir(".")) == before          # nothing written


def test_grep_plugs_into_eval_harness_on_fixture():
    res = evaluate(fixture_instances(), lambda: GrepBaseline(), ks=[1, 5, 10],
                   level="function")
    assert res["n"] == 1
    assert res["metrics"]["recall@10"] == 1.0       # finds the gold function
