"""The bounded bm25->DCI ACI: `bm25_search` (SearchBm25Dci, bm25 top-k search that stages hits
incrementally) paired with `bash`/`read` (Bash(bounded=True)/Read) rooted at a staging dir that
holds only the retrieved docs.

`bm25_search` ranks with the same Lucene BM25 engine (`BM25Pyserini`) `search_bm25` uses, so
retrieval is byte-identical to research_bm25 for the same query/engine. `bash`/`read` are the
DCI shell/read tools, but rooted at a staging dir that only ever grows with what has been
retrieved; a doc outside the top-k is not on disk, so it cannot be grepped or read at all.
Bounded is a filesystem fact, not a runtime check."""
from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.bash.tool import Bash
from agent_search.tools.read.tool import Read
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_bm25_dci.tool import SearchBm25Dci

from tests import lucene_support

lucene_support.require_jvm()

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
    return lucene_support.build_pyserini(_units())


def _toolbox(query="harbor festival annual event", topk=1, engine=None):
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question=query)
    engines = {"bm25": engine or _engine()}
    tools = [SearchBm25Dci(topk=topk), Bash(bounded=True), Read()]
    bound = [t.bind(state, units, ubyid, engines) for t in tools]
    return ToolBox(bound, state)


# --- retrieval: bm25 top-k, byte-identical to search_bm25's engine -----------

def test_search_returns_topk_hits():
    box = _toolbox(topk=2)
    assert len(box.last_hits) == 2
    out = box.run("bm25_search", {"query": "harbor festival annual event"})
    assert "d_harbor" in out or "d_flat" in out


def test_retrieval_matches_search_bm25_for_same_engine_and_query():
    """The FIRST stage must be byte-identical to research_bm25's retrieval: same engine,
    same query, same top-k doc_ids — only the READ strategy differs between the two arms."""
    engine = _engine()
    query = "harbor festival annual event"
    units = _units()
    ubyid = {u.doc_id: u for u in units}

    bm_box = _toolbox(query=query, topk=2, engine=engine)

    visit_state = EpisodeState(question=query)
    visit_search = SearchBm25(name="bm25_search").bind(visit_state, units, ubyid, {"bm25": engine})
    visit_search.run({"query": query, "k": 2})

    assert bm_box.last_hits == visit_state.last_hits


def test_topk_marks_hits_as_seen_immediately():
    # the bm25 top-k are "surfaced" at bind time (on_bind seeds one search on the episode's
    # question), before any explicit tool call — matching search_bm25's search() which marks a
    # hit seen the moment it is ranked.
    box = _toolbox(topk=2)
    assert set(box.last_hits) <= set(box.seen)


def test_empty_query_yields_no_hits():
    box = _toolbox(query="", topk=5)
    assert box.last_hits == []
    # live-retrieval convention (matches search_bm25): a blank query is rejected as
    # "empty query", not rendered as a 0-hit ranking (there was no retrieval to render).
    assert box.run("bm25_search", {"query": ""}) == "empty query"


# --- boundedness: the shell can ONLY see the top-k staged docs ---------------

def test_bash_ls_lists_only_the_topk_files():
    box = _toolbox(topk=1)
    assert len(box.last_hits) == 1
    out = box.run("bash", {"command": "ls"})
    n_txt = sum(1 for tok in out.split() if tok.endswith(".txt"))
    assert n_txt == 1


def test_outside_topk_doc_is_not_greppable():
    """The controlled property: a doc excluded from the bm25 top-k is not on disk in the
    staging dir at all, so grepping for its unique content finds nothing — this is what
    'bounded to the retrieval' means, verified as a filesystem fact."""
    box = _toolbox(topk=1)                      # d_outside is NOT in the top-1
    assert "d_outside" not in box.last_hits
    out = box.run("bash", {"command": "grep -rl 'Quantum Chromodynamics' ."})
    assert "no matches found" in out or ".txt" not in out
    assert "d_outside" not in box.seen


def test_in_topk_doc_is_greppable_and_readable():
    box = _toolbox(topk=1)
    doc_id = box.last_hits[0]
    listing = box.run("bash", {"command": "grep -rl 'Harbor Festival\\|harbor festival' ."})
    assert ".txt" in listing
    fname = [tok for tok in listing.split() if tok.endswith(".txt")][0].lstrip("./")
    out = box.run("read", {"path": fname})
    assert out.strip()
    assert doc_id in box.seen


def test_wider_topk_admits_a_previously_excluded_doc():
    """A query that matches ALL three docs (shares a term with each): topk=1 stages only
    the top hit; topk=3 stages all three, proving the staged set tracks `topk`, not a
    hardcoded exclusion. Lucene never pads with zero-overlap docs regardless of k, so the
    query must genuinely match all three, unlike the harbor-only query used elsewhere in
    this file."""
    query = "harbor festival particle physics event force"     # hits every doc a little
    narrow = _toolbox(query=query, topk=1)
    wide = _toolbox(query=query, topk=3)
    assert len(narrow.last_hits) == 1
    assert set(wide.last_hits) == {"d_harbor", "d_flat", "d_outside"}
    out = wide.run("bash", {"command": "grep -rl 'Quantum Chromodynamics' ."})
    assert ".txt" in out


# --- run() dispatch -----------------------------------------------------------

def test_run_dispatches_bm25_search():
    box = _toolbox(topk=2)
    out = box.run("bm25_search", {"query": "harbor festival annual event"})
    assert "d_harbor" in out or "d_flat" in out


def test_run_dispatches_bash_and_read():
    box = _toolbox(topk=1)
    out = box.run("bash", {"command": "ls"})
    assert ".txt" in out


def test_run_unknown_tool_errors():
    out = _toolbox(topk=1).run("fetch", {"specs": []})
    assert "unknown tool" in out.lower()


def test_tools_tuple_matches_toolset():
    assert _toolbox(topk=1).tools == ("bm25_search", "bash", "read")
