"""The RETRIEVAL-BOUNDED structured-read ACI: Bm25FetchWorkspace (bm25 top-k retrieval, then
fetch a named SECTION — DocSearchFetch's read side on Bm25Visit's retrieval).

It is DocSearchFetch's read (search LISTS structure — section names + infobox keys, NO bodies;
fetch pulls one named section) on plain BM25 retrieval, so the ONLY difference from `research`
is the retrieval stage, and the ONLY difference from `research_bm25` is the read. Retrieval is
LIVE per bm25_search call (identical retrieval BEHAVIOR to Bm25Visit for the same queries — the
one-shot-at-construction design was a bug that broke the controlled comparison); the section-fetch
machinery is INHERITED from DocSearchFetch (not duplicated). Completes the controlled set with
research_bm25 (whole-doc visit) and research_bm25_dci (bash/read shell) over identical bm25 retrieval."""
from agent_search.agent.tools.doc_research import Bm25FetchWorkspace, Bm25Visit
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lexical.bm25 import BM25Local

# a structured doc (## markers -> sections + infobox), a flat doc, and an off-topic doc that a
# harbor query never scores (bm25's score>0 filter) — the doc a fetch must not be able to reach.
DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual harbor event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today at the harbor.",
     "infobox": "Founded: 1897; Location: Portville"},
    {"_id": "d_flat", "title": "Harbor Festival History",
     "text": "The harbor festival tradition began with local fishing boat races."},
    {"_id": "d_outside", "title": "Quantum Chromodynamics",
     "text": "Quarks and gluons interact via the strong nuclear force in particle physics."},
]


def _units():
    return units_from_documents(DOCS)


def _engine():
    return BM25Local().index(_units())


def _ws(query="harbor festival annual event history", topk=5, engine=None):
    return Bm25FetchWorkspace(_units(), query, engine=engine or _engine(), topk=topk)


# --- retrieval: bm25 top-k, byte-identical to Bm25Visit's engine -------------

def test_retrieval_matches_bm25visit_for_same_engine_and_query():
    """Retrieval must be byte-identical to research_bm25's for the same query — the whole point is
    that ONLY the read strategy (section-fetch vs whole-doc visit) differs between the arms.
    Both arms retrieve LIVE per search call over the same engine."""
    engine = _engine()
    query = "harbor festival annual event history"
    fetch_ws = Bm25FetchWorkspace(_units(), query, engine=engine, topk=3)
    fetch_ws.run("bm25_search", {"query": query, "k": 3})
    visit_ws = Bm25Visit(_units(), engine=engine)
    visit_ws.search(query, k=3)
    assert fetch_ws.last_hits == visit_ws.last_hits


def test_topk_marks_hits_as_seen_immediately():
    # LIVE retrieval: construction does not retrieve; a search's hits are seen the moment
    # they are ranked (same contract as Bm25Visit.search).
    ws = _ws(topk=5)
    assert ws.last_hits == []                        # nothing retrieved at construction
    ws.run("bm25_search", {"query": "harbor festival annual event history"})
    assert ws.last_hits                              # harbor query matches >=1 doc
    assert set(ws.last_hits) <= set(ws.seen)


def test_empty_query_yields_no_hits():
    ws = _ws(query="", topk=5)
    assert ws.last_hits == []
    # live-retrieval convention (matches Bm25Visit.search): a blank query is rejected as
    # "empty query", not rendered as a 0-hit ranking (there was no retrieval to render).
    assert ws.search("") == "empty query"


# --- search: LISTS structure (sections + infobox keys), NO bodies ------------

def test_search_lists_sections_and_infobox_no_body():
    ws = _ws(topk=5)
    out = ws.run("bm25_search", {"query": "harbor festival"})
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out       # section names shown
    assert "Founded" in out                           # infobox key shown
    assert "A. Smith" not in out                      # NOT the body content
    assert "1897" not in out                          # infobox VALUE not shown (keys only)


def test_search_is_live_and_re_retrieves():
    """LIVE retrieval: a NEW query string in a later bm25_search call re-runs bm25 and CHANGES
    the ranking (same behavior as Bm25Visit.search) — the one-shot design this replaces ignored
    the agent's queries entirely (measured: gold doc surfaced 4% vs 59% on browsecomp)."""
    ws = _ws(query="harbor festival history", topk=5)
    ws.run("bm25_search", {"query": "harbor festival history"})
    first = list(ws.last_hits)
    ws.run("bm25_search", {"query": "quantum chromodynamics particle physics"})
    assert ws.last_hits != first                      # a different query re-retrieves
    assert "d_outside" in ws.last_hits                # ...and can surface docs the first missed


# --- fetch: pull ONE named section, inherited from DocSearchFetch ------------

def test_fetch_named_section_by_rank():
    ws = _ws(topk=5)
    ws.run("bm25_search", {"query": "harbor festival"})   # establishes ranking (rank 1 = d_harbor)
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert "Founded in 1897" in out and "History" in out


def test_fetch_infobox():
    ws = _ws(topk=5)
    ws.run("bm25_search", {"query": "harbor festival"})
    out = ws.run("fetch", {"specs": [[1, "infobox"]]})
    assert "Founded=1897" in out


def test_fetch_bad_section_lists_available():
    ws = _ws(topk=5)
    ws.run("bm25_search", {"query": "harbor festival"})
    out = ws.run("fetch", {"specs": [[1, "Nonexistent"]]})
    assert "no section" in out and "History" in out       # a section NOT on the hit isn't fetchable


def test_fetch_marks_doc_seen():
    ws = _ws(topk=5)
    ws.run("bm25_search", {"query": "harbor festival"})
    ws.run("fetch", {"specs": [[1, "History"]]})
    assert "d_harbor" in ws.seen


def test_off_topic_doc_is_not_in_the_ranking():
    """d_outside shares no vocabulary with the harbor query, so bm25 never ranks it (score>0
    filter) — it cannot be fetched by rank, and its content never appears."""
    ws = _ws(query="harbor festival annual event", topk=5)
    assert "d_outside" not in ws.last_hits
    listing = ws.run("bm25_search", {"query": "harbor festival"})
    assert "Quantum Chromodynamics" not in listing


# --- run() dispatch -----------------------------------------------------------

def test_run_unknown_tool_errors():
    out = _ws(topk=5).run("visit", {"rank": 1})           # visit is not a tool of this arm
    assert "unknown tool" in out.lower()


def test_tools_tuple_matches_toolset():
    assert Bm25FetchWorkspace.tools == ("bm25_search", "fetch")
