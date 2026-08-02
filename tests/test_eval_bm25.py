from agent_search.retrievers.ranking import BM25, code_tokenize


def test_code_tokenize_splits_camel_and_snake():
    toks = set(code_tokenize("getUserToken auth_token"))
    assert {"get", "user", "token", "auth"} <= toks


def test_bm25_ranks_matching_doc_first():
    bm = BM25().index({
        "d1": "authenticate user token",
        "d2": "render html page layout",
        "d3": "user login token session",
    })
    ids = [doc for doc, _ in bm.search("token", k=3)]
    assert ids[0] in {"d1", "d3"}
    assert "d2" not in ids[:1]


def test_bm25_respects_k():
    bm = BM25().index({"d1": "a b", "d2": "a c", "d3": "a d"})
    assert len(bm.search("a", k=2)) == 2


def test_score_terms_keeps_zero_score_candidates():
    bm = BM25().index({"d1": "alpha", "d2": "beta"})
    ranked = bm.score_terms(["alpha"])

    assert [doc for doc, _ in ranked] == ["d1", "d2"]
    assert ranked[1][1] == 0.0
