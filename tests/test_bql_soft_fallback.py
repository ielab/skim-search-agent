"""The Boolean sieve's soft-AND fallback: `LuceneBqlAdapter.soft_topk` + `SearchBql.run`.

A 0-exact-hit Boolean query degrades to a whole-corpus BM25 ranking over the query's own
positive leaf terms, instead of just handing back a generic "loosen the query" hint. Exact
AND is brittle under paraphrase/obfuscation, so the right doc is often lexically close but not
an exact conjunctive match. The fallback is corpus-fair (BM25 over the agent's own terms only,
no corpus-vocabulary peeking) and its hits are fetchable (last_hits/seen updated). Documents
rank on Lucene, so the engine is the Lucene BQL adapter from `tests/lucene_support.py`.
"""
from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bql.tool import SearchBql
from tests import lucene_support

lucene_support.require_jvm()

# ~30 docs, each built around ONE distinctive topic word repeated for a clear BM25 signal, so a
# query on that word ranks its doc unambiguously first. d1/d2/d3 additionally carry `##`-style
# structured sections (History/Career) to exercise the structure-table rendering.
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
          "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
          "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu", "apple",
          "banana", "cherry", "dolphin"]


def _doc(i: int, word: str) -> dict:
    d = {"_id": f"d{i + 1}", "title": f"{word.capitalize()} Report",
         "text": f"{word} is the central subject of this report. {word} details: {word} "
                 f"background and {word} facts, repeated for emphasis: {word} {word}."}
    if i < 3:                                      # d1, d2, d3 get real sections
        d["sections"] = [
            {"heading": "History", "text": f"{word} has a long documented history."},
            {"heading": "Career", "text": f"{word}'s career spans several notable events."},
        ]
    return d


DOCS = [_doc(i, w) for i, w in enumerate(_WORDS)]


def _units():
    return units_from_documents(DOCS)


def _ex():
    return lucene_support.build_lucene_bql(_units())


def _toolbox(units=None, executor=None):
    units = units if units is not None else _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    search = SearchBql(name="search").bind(state, units, ubyid, {"bql": executor or _ex()})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([search, fetch], state)


# --- LuceneBqlAdapter.soft_topk -------------------------------------------------------------

def test_soft_topk_ranks_the_shared_term_doc_first():
    ex = _ex()
    # "alpha" only appears (repeatedly) in d1; "zzzznonexistent" appears nowhere.
    hits = ex.soft_topk(["alpha", "zzzznonexistent"], k=5)
    assert hits, "soft_topk should find d1 via the shared 'alpha' term"
    assert hits[0][0] == "d1"
    assert hits[0][1] > 0


def test_soft_topk_excludes_zero_score_docs():
    ex = _ex()
    hits = ex.soft_topk(["alpha", "zzzznonexistent"], k=len(_units()))
    # every doc that shares NO term with the query must be excluded, not just truncated
    assert all(score > 0 for _, score in hits)
    assert "d2" not in [d for d, _ in hits]          # bravo doc shares nothing with "alpha"


def test_soft_topk_no_shared_terms_returns_empty():
    ex = _ex()
    assert ex.soft_topk(["zzzznonexistent", "yyyyabsent"], k=5) == []


# --- SearchBql.run: 0-hit fallback -------------------------------------------------------------

def test_zero_hit_fallback_shows_marker_and_is_fetchable():
    box = _toolbox()
    out = box.run("search", {"query": "alpha AND zzzznonexistentterm"})
    assert "0 exact matches" in out
    assert "d1" in out                                # the soft-ranked top hit is listed
    # the fallback ranking must be fetchable by rank, exactly like an exact-hit ranking
    fetched = box.run("fetch", {"specs": [[1, ""]]})
    assert "ERROR" not in fetched
    assert "d1" in fetched
    assert "d1" in box.seen


def test_zero_hit_fallback_no_shared_terms_falls_back_to_generic_hint():
    box = _toolbox()
    out = box.run("search", {"query": "zzzznonexistentterm AND yyyyabsentterm"})
    assert "0 matches" in out
    assert "0 exact matches" not in out               # no soft hits -> the OLD generic path
    assert "hint: loosen the query" in out


def test_exact_hit_path_unchanged_no_fallback_marker():
    box = _toolbox()
    out = box.run("search", {"query": "bravo[title]"})
    assert "0 exact matches" not in out
    assert "d2" in out and "'Bravo Report'" in out
    assert "matches, top" in out


def test_exact_hit_path_renders_sections_and_infobox_like_before():
    box = _toolbox()
    out = box.run("search", {"query": "alpha[title]"})
    assert "History" in out and "Career" in out
    assert "matched:" in out
