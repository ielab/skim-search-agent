"""The bounded bm25->DCI ACI: Bm25DciWorkspace (bm25 top-k search, then bash+read rooted at
a flat export of only those k docs).

`search`/`bm25_search` replays the top-k ranking computed once at construction (the same
`BM25Local` engine `Bm25Visit` uses, so retrieval is byte-identical to research_bm25 for the
same query/engine). `bash`/`read` are DciWorkspace's shell, but rooted at a staging dir that
holds only the retrieved docs; a doc outside the top-k is not on disk, so it cannot be
grepped or read at all. Bounded is a filesystem fact, not a runtime check."""
from agent_search.legacy.workspaces.doc_bm25_dci import Bm25DciWorkspace
from agent_search.legacy.workspaces.search_visit import Bm25Visit
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lexical.bm25 import BM25Local

# d_harbor and d_flat are strongly on-topic for "harbor festival"; d_outside shares NO
# vocabulary with the query at all, so with k=1 it is reliably pushed out of the bm25 top-k —
# the doc a bash/read call must NOT be able to see.
DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\nFounded in 1897 by A. Smith in Portville."},
    {"_id": "d_flat", "title": "Harbor Festival History",
     "text": "The harbor festival tradition began with local fishing boat races."},
    {"_id": "d_outside", "title": "Quantum Chromodynamics",
     "text": "Quarks and gluons interact via the strong nuclear force in particle physics."},
]


def _units():
    return units_from_documents(DOCS)


def _engine():
    return BM25Local().index(_units())


def _ws(query="harbor festival annual event", topk=1, engine=None):
    return Bm25DciWorkspace(_units(), query, engine=engine or _engine(), topk=topk)


# --- retrieval: bm25 top-k, byte-identical to Bm25Visit's engine -------------

def test_search_returns_topk_hits():
    ws = _ws(topk=2)
    assert len(ws.last_hits) == 2
    out = ws.search("harbor festival annual event")
    assert "d_harbor" in out or "d_flat" in out


def test_retrieval_matches_bm25visit_for_same_engine_and_query():
    """The FIRST stage must be byte-identical to research_bm25's retrieval: same engine,
    same query, same top-k doc_ids — only the READ strategy differs between the two arms."""
    engine = _engine()
    query = "harbor festival annual event"
    bm_ws = Bm25DciWorkspace(_units(), query, engine=engine, topk=2)
    visit_ws = Bm25Visit(_units(), engine=engine)
    visit_ws.search(query, k=2)
    assert bm_ws.last_hits == visit_ws.last_hits


def test_topk_marks_hits_as_seen_immediately():
    # the bm25 top-k are "surfaced" at construction, before any shell call — matching
    # Bm25Visit's search() which marks a hit seen the moment it is ranked.
    ws = _ws(topk=2)
    assert set(ws.last_hits) <= set(ws.seen)


def test_empty_query_yields_no_hits():
    ws = _ws(query="", topk=5)
    assert ws.last_hits == []
    # live-retrieval convention (matches Bm25Visit.search): a blank query is rejected as
    # "empty query", not rendered as a 0-hit ranking (there was no retrieval to render).
    assert ws.search("") == "empty query"


# --- boundedness: the shell can ONLY see the top-k staged docs ---------------

def test_bash_ls_lists_only_the_topk_files():
    ws = _ws(topk=1)
    assert len(ws.last_hits) == 1
    out = ws.bash("ls")
    n_txt = sum(1 for tok in out.split() if tok.endswith(".txt"))
    assert n_txt == 1


def test_outside_topk_doc_is_not_greppable():
    """The controlled property: a doc excluded from the bm25 top-k is not on disk in the
    staging dir at all, so grepping for its unique content finds nothing — this is what
    'bounded to the retrieval' means, verified as a filesystem fact."""
    ws = _ws(topk=1)                      # d_outside is NOT in the top-1
    assert "d_outside" not in ws.last_hits
    out = ws.bash("grep -rl 'Quantum Chromodynamics' .")
    assert "no matches found" in out or ".txt" not in out
    assert "d_outside" not in ws.seen


def test_in_topk_doc_is_greppable_and_readable():
    ws = _ws(topk=1)
    doc_id = ws.last_hits[0]
    listing = ws.bash("grep -rl 'Harbor Festival\\|harbor festival' .")
    assert ".txt" in listing
    fname = [tok for tok in listing.split() if tok.endswith(".txt")][0].lstrip("./")
    out = ws.read(fname)
    assert out.strip()
    assert doc_id in ws.seen


def test_wider_topk_admits_a_previously_excluded_doc():
    """A query that scores ALL three docs (shares a term with each): topk=1 stages only
    the top hit; topk=3 stages all three — proving the staged set tracks `topk`, not a
    hardcoded exclusion. (bm25 never pads with zero-overlap docs regardless of k — see
    ranking.BM25.search's `score > 0` filter — so the query must genuinely match all three,
    unlike the harbor-only query used elsewhere in this file.)"""
    query = "harbor festival particle physics event force"     # hits every doc a little
    narrow = _ws(query=query, topk=1)
    wide = _ws(query=query, topk=3)
    assert len(narrow.last_hits) == 1
    assert set(wide.last_hits) == {"d_harbor", "d_flat", "d_outside"}
    out = wide.bash("grep -rl 'Quantum Chromodynamics' .")
    assert ".txt" in out


# --- run() dispatch -----------------------------------------------------------

def test_run_dispatches_bm25_search():
    ws = _ws(topk=2)
    out = ws.run("bm25_search", {"query": "harbor festival annual event"})
    assert "d_harbor" in out or "d_flat" in out


def test_run_dispatches_bash_and_read():
    ws = _ws(topk=1)
    out = ws.run("bash", {"command": "ls"})
    assert ".txt" in out


def test_run_unknown_tool_errors():
    out = _ws(topk=1).run("fetch", {"specs": []})
    assert "unknown tool" in out.lower()


def test_tools_tuple_matches_toolset():
    assert Bm25DciWorkspace.tools == ("bm25_search", "bash", "read")
