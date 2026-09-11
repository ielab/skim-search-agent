"""The hybrid engine: any retrievers the run names, one fusion method; the defaults reproduce
the paper's BM25 + dense RRF arms, and a rank-only engine cannot be interpolated."""
import pytest

from agent_search.errors import SetupError
from agent_search.retrievers.fusion import RRF, Interpolation
from agent_search.retrievers.hybrid import HybridEngine, HybridRetriever, components_from_env, fusion_from_env


class _Scored:
    def __init__(self, ranking): self.ranking = ranking
    def search_scored(self, query, k): return self.ranking[:k]
    def search(self, query, k): return [d for d, _ in self.ranking[:k]]


class _RanksOnly:
    def __init__(self, ids): self.ids = ids
    def top_k_doc_ids(self, query, k=None): return self.ids[:k]


def test_rrf_over_two_components_matches_the_old_hybrid_arm():
    bm25 = _RanksOnly(["a", "b", "c"]); dense = _RanksOnly(["b", "d"])
    eng = HybridEngine({"bm25": bm25, "dense": dense}, RRF(k=60), pool=100)
    assert eng.search("q", 3) == ["b", "a", "d"]
    assert eng.describe() == {"retrievers": ["bm25", "dense"], "pool": 100, "fusion": "rrf", "rrf_k": 60}


def test_interpolation_needs_scores():
    eng = HybridEngine({"bm25": _Scored([("a", 2.0), ("b", 1.0)]), "dense": _Scored([("b", 0.9), ("c", 0.1)])},
                       Interpolation(weights=[0.5, 0.5]), pool=10)
    assert eng.search("q", 2) == ["a", "b"] or eng.search("q", 2) == ["b", "a"]
    bad = HybridEngine({"bm25": _Scored([("a", 2.0)]), "dense": _RanksOnly(["a"])}, Interpolation(), pool=10)
    with pytest.raises(SetupError):
        bad.search("q", 1)


def test_three_components_and_the_env_knobs(monkeypatch):
    monkeypatch.setenv("HYBRID_RETRIEVERS", "bm25,dense,bql")
    monkeypatch.setenv("HYBRID_FUSION", "interpolation")
    monkeypatch.setenv("HYBRID_WEIGHTS", "0.2,0.3,0.5")
    assert components_from_env() == ["bm25", "dense", "bql"]
    f = fusion_from_env()
    assert isinstance(f, Interpolation) and f.weights == [0.2, 0.3, 0.5]
    monkeypatch.setenv("HYBRID_RETRIEVERS", "bm25")
    with pytest.raises(SetupError):
        components_from_env()


def test_hybrid_retriever_over_a_prebuilt_registry():
    class _Registry:
        def get(self, kind):
            return {"bm25": _RanksOnly(["a", "b"]), "dense": _RanksOnly(["b", "c"])}[kind]
    r = HybridRetriever(components=["bm25", "dense"], fusion=RRF(k=60), pool=10, engines=_Registry())
    r.index([], key="k")
    assert r.search("q", 2) == ["b", "a"]


def test_engines_build_the_hybrid_kind_from_the_run(monkeypatch):
    from agent_search.retrievers.engines import Engines
    e = Engines([], "k")
    monkeypatch.setattr(Engines, "bm25", lambda self: _RanksOnly(["x", "y"]))
    monkeypatch.setattr(Engines, "dense", lambda self: _RanksOnly(["y", "z"]))
    monkeypatch.setenv("HYBRID_RETRIEVERS", "bm25,dense"); monkeypatch.setenv("HYBRID_FUSION", "rrf")
    h = e.get("hybrid")
    assert h.search("q", 3) == ["y", "x", "z"]
    assert e.get("hybrid") is h
