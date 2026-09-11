"""Rerankers and the reranked engine: the composition of one retriever's pool with one reranker
(`agent_search/retrievers/reranked.py`), the reranker registry, and the environment knobs.
The cross-encoder itself needs a model and is exercised on the cluster, not here."""
from __future__ import annotations

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.errors import SetupError
from agent_search.retrievers.reranked import (RerankedEngine, RerankedRetriever, base_from_env, pool_from_env,
                                              reranker_from_env, unit_text)
from agent_search.retrievers.rerankers import RERANKERS, Reranker, build_reranker, register_reranker


class _Longest(Reranker):
    """Scores a candidate by the length of its text: a deterministic stand-in for a model."""
    name = "longest"

    def rerank(self, query, candidates, k=None):
        ranked = sorted(((d, float(len(t))) for d, t in candidates), key=lambda x: (-x[1], x[0]))
        return ranked[:k] if k is not None else ranked


class _Base:
    """A retriever whose ranking is fixed, so the test sees what the reranker changed."""

    def __init__(self, order):
        self.order = list(order)
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.order[:k]


def _unit(doc_id, text, title=None):
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id, start_line=1, end_line=1,
                    code=text, body=text, title=title)


UNITS = [_unit("a", "short"), _unit("b", "a much longer document text", title="B"), _unit("c", "medium text")]
TEXT = {u.doc_id: unit_text(u) for u in UNITS}


def test_reranked_engine_reorders_the_base_pool_and_reports_scores():
    base = _Base(["a", "b", "c"])
    eng = RerankedEngine(base, _Longest(), TEXT.get, pool=10, base_name="stub")
    assert eng.search("q", 2) == ["b", "c"]
    scored = eng.search_scored("q", 3)
    assert [d for d, _ in scored] == ["b", "c", "a"] and all(isinstance(s, float) for _, s in scored)
    assert base.calls == [("q", 10), ("q", 10)]          # the pool size, not k, goes to the base


def test_reranked_engine_empty_pool_is_empty():
    eng = RerankedEngine(_Base([]), _Longest(), TEXT.get, pool=5)
    assert eng.search("q", 3) == [] and eng.search_scored("q", 3) == []


def test_describe_names_base_pool_and_reranker():
    eng = RerankedEngine(_Base(["a"]), _Longest(), TEXT.get, pool=7, base_name="bm25")
    assert eng.describe() == {"base": "bm25", "pool": 7, "reranker": "longest"}


def test_unit_text_joins_title_and_body():
    assert unit_text(UNITS[1]) == "B\na much longer document text"
    assert unit_text(UNITS[0]) == "short"
    assert unit_text(None) == ""


def test_registry_builds_by_name_and_rejects_unknown():
    register_reranker(_Longest)
    assert isinstance(build_reranker("longest"), _Longest)
    assert "cross_encoder" in RERANKERS
    with pytest.raises(ValueError):
        build_reranker("no_such_reranker")


def test_env_knobs(monkeypatch):
    monkeypatch.delenv("RERANK_BASE", raising=False)
    monkeypatch.delenv("RERANK_POOL", raising=False)
    assert base_from_env() == "bm25" and pool_from_env() == 100
    monkeypatch.setenv("RERANK_BASE", "dense")
    monkeypatch.setenv("RERANK_POOL", "25")
    assert base_from_env() == "dense" and pool_from_env() == 25
    monkeypatch.setenv("RERANK_BASE", "reranked")
    with pytest.raises(SetupError):
        base_from_env()
    monkeypatch.setenv("RERANK_METHOD", "longest")
    register_reranker(_Longest)
    assert isinstance(reranker_from_env(), _Longest)


def test_retriever_uses_a_prebuilt_registry():
    class _Engines:
        def get(self, kind):
            assert kind == "stub"
            return _Base(["a", "b", "c"])

        def text_of(self, doc_id):
            return TEXT.get(doc_id, "")

    r = RerankedRetriever(base="stub", reranker=_Longest(), pool=10, engines=_Engines()).index(UNITS, key="k")
    assert r.search("q", 2) == ["b", "c"]
    assert [d for d, _ in r.search_with_scores("q", 1)] == ["b"]
