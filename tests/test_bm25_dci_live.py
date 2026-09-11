"""Live-retrieval test for the bounded_dci strategy's `bm25_search` tool
(agent_search.tools.search_bm25_dci) paired with `bash`/`read` over its staging directory.

`bm25_search` retrieves live on every call rather than replaying a ranking fixed at
construction on the raw episode query: a tool call with a different query string must
re-rank, so a doc outside the construction-time top-k can still surface. This matches its
live-retrieval sibling `search_bm25` (agent_search.tools.search_bm25), which re-runs bm25
per call.

This file builds a small multi-topic corpus where the construction-time query (the episode
question, seeded by `on_bind`) and a later in-episode query are deliberately disjoint
(near-zero term overlap), so a doc that could only ever be found by the later query is a
direct probe of whether re-searching changes the ranking.
"""
from agent_search.corpus.units import units_from_documents
from agent_search.tools.bash.tool import Bash
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.read.tool import Read
from agent_search.tools.search_bm25_dci.tool import SearchBm25Dci

from tests import lucene_support

lucene_support.require_jvm()

# doc_a is strongly on-topic for the CONSTRUCTION query ("solarflare mission alpha"); doc_b
# shares NO vocabulary with it at all, so at construction time (topk small) doc_b is reliably
# pushed out of the initial ranking. doc_b IS the top hit for the LATER query ("gizmo quantum
# beta"), which shares no vocabulary with the construction query either. This is the pair the
# one-shot bug could never surface via a second search. The engine is Lucene BM25
# (`lucene_support.build_pyserini`), the only BM25 a document corpus has.
DOCS = [
    {"_id": "doc_a", "title": "Solar Mission Alpha",
     "text": "A solarflare mission alpha briefing describing the alpha launch sequence."},
    {"_id": "doc_b", "title": "Gizmo Quantum Beta",
     "text": "A gizmo quantum beta prototype uses beta-phase quantum gizmo circuitry."},
] + [
    # ~18 filler docs, each with its OWN distinct topic word, so the corpus is a realistic
    # multi-topic pool bm25 actually has to rank over (not just a 2-doc toy).
    {"_id": f"doc_filler{i}", "title": f"Filler Topic {i}",
     "text": f"filler{i} padding content about topic filler{i} and nothing else relevant."}
    for i in range(2, 20)
]

QUERY_INIT = "solarflare mission alpha"
QUERY_LATER = "gizmo quantum beta"


def _units():
    return units_from_documents(DOCS)


def _engine():
    return lucene_support.build_pyserini(_units())


def _ws(query=QUERY_INIT, topk=3, engine=None):
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question=query)
    search = SearchBm25Dci(topk=topk).bind(state, units, ubyid, {"bm25": engine or _engine()})
    bash = Bash(bounded=True).bind(state, units, ubyid, {})
    read = Read().bind(state, units, ubyid, {})
    return ToolBox([search, bash, read], state)


# --- 1. live retrieval: a later query genuinely re-retrieves ------------------

def test_construction_query_does_not_surface_doc_b():
    ws = _ws(topk=3)
    assert "doc_b" not in ws.last_hits


def test_later_search_surfaces_doc_b_the_one_shot_bug_could_not():
    ws = _ws(topk=3)
    assert "doc_b" not in ws.last_hits          # sanity: not present before the later search
    out = ws.run("bm25_search", {"query": QUERY_LATER})
    assert "doc_b" in out
    assert "doc_b" in ws.last_hits               # last_hits reflects the NEW ranking, live


def test_last_hits_reflects_new_ranking_not_the_old_one():
    ws = _ws(topk=3)
    initial_hits = list(ws.last_hits)
    ws.run("bm25_search", {"query": QUERY_LATER})
    assert ws.last_hits != initial_hits
    assert "doc_a" not in ws.last_hits or ws.last_hits != initial_hits


# --- 2. incremental staging: newly-surfaced docs land on disk immediately ----

def _rel_to_doc(ws):
    return ws.state.scratch["dci_rel_to_doc"]


def test_doc_b_file_exists_and_is_readable_after_later_search():
    ws = _ws(topk=3)
    ws.run("bm25_search", {"query": QUERY_LATER})
    rel_to_doc = _rel_to_doc(ws)
    assert "doc_b" in rel_to_doc.values()
    rel = [r for r, d in rel_to_doc.items() if d == "doc_b"][0]
    out = ws.run("read", {"path": rel})
    assert "gizmo" in out.lower()


def test_doc_b_is_greppable_via_bash_after_later_search():
    ws = _ws(topk=3)
    ws.run("bm25_search", {"query": QUERY_LATER})
    out = ws.run("bash", {"command": "grep -rl 'gizmo' ."})
    assert ".txt" in out


def test_run_dispatch_stages_incrementally_too():
    ws = _ws(topk=3)
    ws.run("bm25_search", {"query": QUERY_LATER})
    out = ws.run("bash", {"command": "grep -rl 'quantum' ."})
    assert ".txt" in out


# --- 3. the initial staged docs remain readable after later searches --------

def test_initial_docs_stay_staged_and_readable_after_later_search():
    ws = _ws(topk=3)
    initial_hits = list(ws.last_hits)
    assert initial_hits, "construction query should have staged at least one doc"
    ws.run("bm25_search", {"query": QUERY_LATER})    # a DIFFERENT search happens afterward
    rel_to_doc = _rel_to_doc(ws)
    for doc_id in initial_hits:
        rel = [r for r, d in rel_to_doc.items() if d == doc_id][0]
        out = ws.run("read", {"path": rel})
        assert not out.startswith("Error")


# --- 4. a zero-match query is reported, not an error -------------------------

def test_zero_match_query_is_reported_with_query_text():
    # a query built entirely from tokens absent from every doc's vocabulary matches nothing
    # in Lucene, so this must report "(0 matches)", not an error.
    ws = _ws(topk=3)
    out = ws.run("bm25_search", {"query": "zzz_nonexistent_token_qqq"})
    assert "0 matches" in out


# --- 5. seen accumulates hits from BOTH the init retrieval and later search --

def test_seen_accumulates_across_init_and_later_search():
    ws = _ws(topk=3)
    init_hits = set(ws.last_hits)
    assert init_hits <= set(ws.seen)
    ws.run("bm25_search", {"query": QUERY_LATER})
    assert "doc_b" in ws.seen
    # both the construction-time hits AND the later hit are present simultaneously
    assert init_hits <= set(ws.seen)
    assert "doc_b" in ws.seen and init_hits.issubset(ws.seen)


# --- boundedness still holds absent a later search that finds it ------------

def test_doc_never_searched_for_is_not_staged():
    ws = _ws(topk=3)
    ws.run("bm25_search", {"query": QUERY_LATER})
    # doc_filler19 shares vocabulary with neither query and was never in any top-k
    rel_to_doc = _rel_to_doc(ws)
    assert "doc_filler19" not in rel_to_doc.values()
    out = ws.run("bash", {"command": "grep -rl 'filler19' ."})
    assert "no matches found" in out or ".txt" not in out
