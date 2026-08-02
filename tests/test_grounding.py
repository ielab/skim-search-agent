"""The corpus-grounding helper that turns a silent empty search into a 'did you mean'."""
from agent_search.agent.tools.grounding import format_did_you_mean, nearest_tokens


def test_nearest_tokens_prefix_and_typo():
    vocab = ["polygons2mask", "polygon", "polygonize", "mask", "segment"]
    # a guessed plural -> the real prefix family ranks first
    near = nearest_tokens("polygons2masks", vocab, n=3)
    assert "polygons2mask" in near
    # a typo within edit distance
    assert "segment" in nearest_tokens("segmnet", vocab)
    # nothing close -> empty (a real miss, not a typo)
    assert nearest_tokens("zzzzqqq", vocab) == []
    # the term itself is never suggested back
    assert "mask" not in nearest_tokens("mask", vocab)


def test_format_did_you_mean():
    assert format_did_you_mean("foo", []) == ""           # nothing to say
    msg = format_did_you_mean("polygons2masks", ["polygons2mask"], region="call")
    assert "polygons2masks" in msg and "polygons2mask" in msg and "region `call`" in msg
    assert "region" not in format_did_you_mean("foo", ["bar"])   # global (no region)
