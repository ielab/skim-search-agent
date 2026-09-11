"""Two surgical fixes to the doc-research ACI:

1. soft-AND fallback (StructuralExecutor.soft_topk + DocSearchFetch.search): a 0-exact-hit
   Boolean query degrades to a whole-corpus BM25 ranking over the query's own positive leaf
   terms, instead of just handing back a generic "loosen the query" hint. Exact AND is brittle
   under paraphrase/obfuscation, so the right doc is often lexically close but not an exact
   conjunctive match. The fallback is corpus-fair (BM25 over the agent's own terms only, no
   corpus-vocabulary peeking) and its hits are fetchable (last_hits/seen updated).

2. live retrieval in Bm25FetchWorkspace: `bm25_search` retrieves live on every call, exactly
   like Bm25Visit.search, rather than replaying a ranking fixed at construction from the raw
   episode question while ignoring the agent's actual query. This keeps the arm's controlled
   comparison to research_bm25 (same retrieval, different read) honest.
"""
from agent_search.legacy.workspaces.search_fetch import Bm25FetchWorkspace
from agent_search.legacy.workspaces.sieve import DocSearchFetch
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lexical.bm25 import BM25Local
from agent_search.retrievers.bql.executor import StructuralExecutor

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


def _engine():
    return BM25Local().index(_units())


def _ex():
    return StructuralExecutor(_units())


def _ws():
    return DocSearchFetch(_units())


# --- 1a. StructuralExecutor.soft_topk ----------------------------------------

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


# --- 1b. DocSearchFetch.search: 0-hit fallback --------------------------------

def test_zero_hit_fallback_shows_marker_and_is_fetchable():
    ws = _ws()
    out = ws.search("alpha AND zzzznonexistentterm", k=5)
    assert "0 exact matches" in out
    assert "d1" in out                                # the soft-ranked top hit is listed
    # the fallback ranking must be fetchable by rank, exactly like an exact-hit ranking
    fetched = ws.fetch([[1, ""]])
    assert "ERROR" not in fetched
    assert "d1" in fetched
    assert "d1" in ws.seen


def test_zero_hit_fallback_no_shared_terms_falls_back_to_generic_hint():
    ws = _ws()
    out = ws.search("zzzznonexistentterm AND yyyyabsentterm", k=5)
    assert "0 matches" in out
    assert "0 exact matches" not in out               # no soft hits -> the OLD generic path
    assert "hint: loosen the query" in out


def test_exact_hit_path_unchanged_no_fallback_marker():
    ws = _ws()
    out = ws.search("bravo[title]", k=5)
    assert "0 exact matches" not in out
    assert "d2" in out and "'Bravo Report'" in out
    assert "matches, top" in out


def test_exact_hit_path_renders_sections_and_infobox_like_before():
    ws = _ws()
    out = ws.search("alpha[title]", k=5)
    assert "History" in out and "Career" in out
    assert "matched:" in out


# --- 2. Bm25FetchWorkspace: live per-call retrieval ---------------------------

def test_fetch_reads_from_the_latest_ranking():
    ws = Bm25FetchWorkspace(_units(), "something irrelevant", engine=_engine(), topk=5)
    ws.run("bm25_search", {"query": "alpha"})
    ws.run("bm25_search", {"query": "bravo"})          # now ranked on bravo
    out = ws.run("fetch", {"specs": [[1, ""]]})
    assert "d2" in out
    assert "History" not in out or "documented history" in out  # d2 is flat: intro text only


def test_seen_accumulates_across_both_searches():
    ws = Bm25FetchWorkspace(_units(), "something irrelevant", engine=_engine(), topk=5)
    ws.run("bm25_search", {"query": "alpha"})
    ws.run("bm25_search", {"query": "bravo"})
    assert "d1" in ws.seen and "d2" in ws.seen


def test_zero_hit_bm25_search_keeps_generic_hint():
    ws = Bm25FetchWorkspace(_units(), "something irrelevant", engine=_engine(), topk=5)
    out = ws.run("bm25_search", {"query": "zzzznonexistentterm"})
    assert "0 matches" in out
    assert "hint: loosen the query" in out
