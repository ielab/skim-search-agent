"""LIVE-retrieval regression test for `Bm25DciWorkspace` (agent_search.agent.tools.doc_bm25_dci).

BUGFIX under test: `bm25_search` used to be a ONE-SHOT retrieval fixed at construction on the
raw episode query — a tool call with a DIFFERENT query string just replayed that same fixed
ranking, so a doc outside the construction-time top-k could never be surfaced no matter what
the agent searched for. This starved the arm relative to its live-retrieval sibling
(`Bm25Visit` in doc_research.py), which re-runs bm25 per call.

This file builds a small multi-topic corpus where the construction-time query and a LATER
in-episode query are deliberately disjoint (near-zero term overlap), so a doc that could only
ever be found by the later query is a direct probe of "does re-searching change the ranking."
"""
from agent_search.agent.tools.doc_bm25_dci import Bm25DciWorkspace
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lexical.bm25 import BM25Local

# doc_a is strongly on-topic for the CONSTRUCTION query ("solarflare mission alpha"); doc_b
# shares NO vocabulary with it at all, so at construction time (topk small) doc_b is reliably
# pushed out of the initial ranking. doc_b IS the top hit for the LATER query ("gizmo quantum
# beta"), which shares no vocabulary with the construction query either — this is the pair the
# one-shot bug could never surface via a second search.
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
    return BM25Local().index(_units())


def _ws(query=QUERY_INIT, topk=3, engine=None):
    units = _units()
    return Bm25DciWorkspace(units, query, engine=engine or _engine(),
                            ubyid={u.doc_id: u for u in units}, topk=topk)


# --- 1. live retrieval: a later query genuinely re-retrieves ------------------

def test_construction_query_does_not_surface_doc_b():
    ws = _ws(topk=3)
    assert "doc_b" not in ws.last_hits


def test_later_search_surfaces_doc_b_the_one_shot_bug_could_not():
    ws = _ws(topk=3)
    assert "doc_b" not in ws.last_hits          # sanity: not present before the later search
    out = ws.search(QUERY_LATER)
    assert "doc_b" in out
    assert "doc_b" in ws.last_hits               # last_hits reflects the NEW ranking, live


def test_last_hits_reflects_new_ranking_not_the_old_one():
    ws = _ws(topk=3)
    initial_hits = list(ws.last_hits)
    ws.search(QUERY_LATER)
    assert ws.last_hits != initial_hits
    assert "doc_a" not in ws.last_hits or ws.last_hits != initial_hits


# --- 2. incremental staging: newly-surfaced docs land on disk immediately ----

def test_doc_b_file_exists_and_is_readable_after_later_search():
    ws = _ws(topk=3)
    ws.search(QUERY_LATER)
    assert "doc_b" in ws._rel_to_doc.values()
    rel = [r for r, d in ws._rel_to_doc.items() if d == "doc_b"][0]
    out = ws.read(rel)
    assert "gizmo" in out.lower()


def test_doc_b_is_greppable_via_bash_after_later_search():
    ws = _ws(topk=3)
    ws.search(QUERY_LATER)
    out = ws.bash("grep -rl 'gizmo' .")
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
    ws.search(QUERY_LATER)                       # a DIFFERENT search happens afterward
    for doc_id in initial_hits:
        rel = [r for r, d in ws._rel_to_doc.items() if d == doc_id][0]
        out = ws.read(rel)
        assert not out.startswith("Error")


# --- 4. a zero-match query is reported, not an error -------------------------

def test_zero_match_query_is_reported_with_query_text():
    # a query built entirely from tokens absent from every doc's vocabulary scores 0 for
    # everything (bm25's `score > 0` filter), so this must report "(0 matches)", not an error.
    ws = _ws(topk=3)
    out = ws.search("zzz_nonexistent_token_qqq")
    assert "0 matches" in out


# --- 5. seen accumulates hits from BOTH the init retrieval and later search --

def test_seen_accumulates_across_init_and_later_search():
    ws = _ws(topk=3)
    init_hits = set(ws.last_hits)
    assert init_hits <= ws.seen
    ws.search(QUERY_LATER)
    assert "doc_b" in ws.seen
    # both the construction-time hits AND the later hit are present simultaneously
    assert init_hits <= ws.seen
    assert "doc_b" in ws.seen and init_hits.issubset(ws.seen)


# --- boundedness still holds absent a later search that finds it ------------

def test_doc_never_searched_for_is_not_staged():
    ws = _ws(topk=3)
    ws.search(QUERY_LATER)
    # doc_filler19 shares vocabulary with neither query and was never in any top-k
    assert "doc_filler19" not in ws._rel_to_doc.values()
    out = ws.bash("grep -rl 'filler19' .")
    assert "no matches found" in out or ".txt" not in out
