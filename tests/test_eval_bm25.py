"""The in-memory BM25 scorer (`agent_search/retrievers/lexical/scorer.py`) and the grep
ranker that keeps it up to date (`lexical/grep.py`). Both exist for code repositories only;
document corpora rank on Lucene and never touch this scorer.

Covered: the identifier tokenizer, ranking and top-k, zero-score candidates in `score_terms`,
the incremental `add`/`remove` (document frequencies, lengths and the average length stay
equal to a fresh index over the same docs), and `GrepBaseline.index(units)` re-indexing only
the files whose units changed.
"""
from collections import Counter

from agent_search.corpus.units import code_tokenize, units_from_python_source
from agent_search.retrievers.lexical.grep import GrepBaseline, file_fingerprints
from agent_search.retrievers.lexical.scorer import BM25


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


# --- incremental add / remove ------------------------------------------------------------

_DOCS = {
    "d1": "authenticate user token",
    "d2": "render html page layout",
    "d3": "user login token session",
    "d4": "token cache refresh token",
}


def _stats(bm: BM25) -> tuple[Counter, dict, int, float]:
    return Counter(bm._df), dict(bm._lens), bm._total_len, bm._avgdl


def test_add_then_remove_matches_a_fresh_index():
    fresh = BM25().index({k: v for k, v in _DOCS.items() if k != "d2"})

    bm = BM25().index({k: _DOCS[k] for k in ("d1", "d2", "d3")})
    bm.add({"d4": _DOCS["d4"]})
    bm.remove(["d2"])

    assert len(bm) == 3 and "d2" not in bm and "d4" in bm
    assert _stats(bm) == _stats(fresh)
    assert bm.search("token", k=10) == fresh.search("token", k=10)
    assert bm.score_terms(["render", "token"]) == fresh.score_terms(["render", "token"])


def test_add_replaces_an_existing_doc_and_keeps_df_consistent():
    bm = BM25().index(dict(_DOCS))
    bm.add({"d2": "token parser"})                 # same id, new text

    fresh = BM25().index({**_DOCS, "d2": "token parser"})
    assert _stats(bm) == _stats(fresh)
    assert bm._df["render"] == 0 or "render" not in bm._df
    assert bm._df["token"] == 4


def test_remove_unknown_id_is_ignored_and_empty_index_has_zero_avgdl():
    bm = BM25().index({"d1": "alpha beta"})
    bm.remove(["nope"])
    assert len(bm) == 1
    bm.remove(["d1"])
    assert len(bm) == 0 and bm._total_len == 0 and bm._avgdl == 0.0
    assert dict(bm._df) == {}
    assert bm.search("alpha", k=5) == []


# --- GrepBaseline.index re-indexes changed files only ---------------------------------------

_SRC_A = (
    "def frobnicate_widget(widget):\n"
    "    return widget.frob()\n"
    "\n"
    "def render_widget(widget):\n"
    "    return str(widget)\n"
)
_SRC_B_V1 = (
    "def parse_header(line):\n"
    "    return line.split(':')\n"
)
_SRC_B_V2 = (
    "def parse_header(line):\n"
    "    frobnicate_widget(line)\n"
    "    return line.split(':')\n"
)


def _repo(src_b: str):
    return units_from_python_source("a.py", _SRC_A) + units_from_python_source("b.py", src_b)


def test_grep_index_reindexes_only_the_changed_file():
    g = GrepBaseline()
    v1 = _repo(_SRC_B_V1)
    g.index(v1)
    assert g.search("frobnicate", k=10) == ["a.py::frobnicate_widget"]

    added: list[str] = []
    removed: list[str] = []
    real_add, real_remove = g._bm.add, g._bm.remove

    def spy_add(docs):
        added.extend(docs)
        return real_add(docs)

    def spy_remove(ids):
        ids = list(ids)
        removed.extend(ids)
        return real_remove(ids)

    g._bm.add, g._bm.remove = spy_add, spy_remove

    v2 = _repo(_SRC_B_V2)
    g.index(v2)

    # only b.py's units were dropped and re-added; a.py was left alone
    assert removed == ["b.py::parse_header"]
    assert added == ["b.py::parse_header"]
    fp_v1, fp_v2 = file_fingerprints(v1), file_fingerprints(v2)
    assert fp_v1["a.py"][0] == fp_v2["a.py"][0]
    assert fp_v1["b.py"][0] != fp_v2["b.py"][0]
    assert {p: fp for p, (fp, _) in g._files.items()} == {p: fp for p, (fp, _) in fp_v2.items()}

    # the search reflects the edit
    hits = g.search("frobnicate", k=10)
    assert set(hits) == {"a.py::frobnicate_widget", "b.py::parse_header"}
    # and the scorer's statistics equal a fresh index over the same units
    fresh = BM25().index({u.doc_id: f"{u.qualname} {u.code}" for u in v2})
    assert _stats(g._bm) == _stats(fresh)


def test_grep_index_drops_a_file_that_disappeared():
    g = GrepBaseline()
    g.index(_repo(_SRC_B_V2))
    assert "b.py::parse_header" in g._bm
    g.index(units_from_python_source("a.py", _SRC_A))
    assert "b.py::parse_header" not in g._bm
    assert set(g._files) == {"a.py"}
    assert g.search("frobnicate", k=10) == ["a.py::frobnicate_widget"]
