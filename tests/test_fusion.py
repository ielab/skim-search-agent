"""Fusion methods over ranked lists: RRF reproduces the constant and tie-break the hybrid
strategies always used; interpolation combines normalised scores with weights."""
import pytest

from agent_search.retrievers.fusion import FUSIONS, Interpolation, RRF, build_fusion


def _ranked(ids, start=10.0):
    return [(d, start - i) for i, d in enumerate(ids)]


def test_rrf_scores_and_tie_break():
    f = RRF(k=60)
    out = f.fuse([_ranked(["a", "b", "c"]), _ranked(["b", "d"])])
    # b: 1/62 + 1/61 is the largest; a and d tie on 1/61 and sort by id; c last
    assert out == ["b", "a", "d", "c"]
    assert f.fuse([_ranked(["a", "b", "c"]), _ranked(["b", "d"])], k=2) == ["b", "a"]
    assert f.fuse([[], []]) == []


def test_rrf_matches_the_hand_formula():
    f = RRF(k=60)
    a = _ranked(["x", "y"]); b = _ranked(["y", "z"])
    out = f.fuse([a, b])
    score = {"x": 1 / 61, "y": 1 / 62 + 1 / 61, "z": 1 / 62}
    assert out == sorted(score, key=lambda d: (-score[d], d))


def test_interpolation_normalises_and_weights():
    f = Interpolation()
    out = f.fuse([[("a", 10.0), ("b", 0.0)], [("b", 1.0), ("c", 0.0)]])
    assert out == ["a", "b", "c"]                      # a: 0.5*1, b: 0.5*0 + 0.5*1 tie -> id order; c: 0
    heavy = Interpolation(weights=[0.1, 0.9])
    assert heavy.fuse([[("a", 10.0), ("b", 0.0)], [("b", 1.0), ("c", 0.0)]]) == ["b", "a", "c"]
    with pytest.raises(ValueError):
        Interpolation(weights=[1.0]).fuse([[("a", 1.0)], [("b", 1.0)]])


def test_registry_and_describe():
    assert set(FUSIONS) >= {"rrf", "interpolation"}
    assert build_fusion("rrf", k=10).describe() == {"fusion": "rrf", "rrf_k": 10}
    assert build_fusion("interpolation", weights=[0.3, 0.7]).describe()["weights"] == [0.3, 0.7]
    with pytest.raises(ValueError):
        build_fusion("nope")
