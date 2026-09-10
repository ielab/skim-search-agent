"""The atomic tools render exactly what the workspaces they replace rendered, call for call.
Kept until the old workspace modules are removed; then these become golden-file tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox

DOCS = [
    {"_id": "d1", "title": "Treaty of Guadalupe Hidalgo", "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848.\n\n## Terms\nMexico ceded land."},
    {"_id": "d2", "title": "Mexican-American War", "text": "The war was fought from 1846 to 1848.\n\n## Causes\nTexas annexation."},
    {"_id": "d3", "title": "Nicholas Trist", "text": "Nicholas Trist negotiated the treaty in 1848 at Guadalupe Hidalgo."},
    {"_id": "d4", "title": "Rio Grande", "text": "The Rio Grande became the border after the treaty."},
    {"_id": "d5", "title": "Gadsden Purchase", "text": "A later purchase of land from Mexico in 1853."},
]
UNITS = units_from_documents(DOCS)
UBYID = {u.doc_id: u for u in UNITS}


class StubBm25:
    def search(self, query, k=5):
        order = {"treaty": ["d1", "d3", "d4", "d2"], "war": ["d2", "d1", "d5"], "nothing": []}
        return order.get(query, ["d5", "d4"])[:k]


class StubDense:
    def top_k_doc_ids(self, query, k=5):
        order = {"treaty": ["d3", "d1", "d2"], "war": ["d2", "d5"], "nothing": []}
        return order.get(query, ["d4"])[:k]


CALLS = [("search", {"query": "treaty"}), ("visit", {"rank": 1}), ("visit", {"rank": "d3"}),
         ("search", {"query": "nothing"}), ("visit", {"rank": 9}), ("search", {"query": "war"}),
         ("visit", {"rank": "Rio"}), ("search", {"query": ""})]


def _drive(ws, search_name, visit_name):
    out = []
    for name, args in CALLS:
        real = search_name if name == "search" else visit_name
        out.append(ws.run(real, dict(args)))
    return out, list(ws.surfaced)


def _toolbox(tools, engines):
    state = EpisodeState(question="q")
    bound = [t.bind(state, UNITS, UBYID, engines) for t in tools]
    return ToolBox(bound, state)


@pytest.mark.parametrize("variant", ["bm25", "bm25q", "dense", "hybrid"])
def test_search_visit_family_matches_the_old_workspaces(variant):
    from agent_search.agent.tools.search_visit import Bm25Visit, DenseVisit, HybridVisit
    from agent_search.tools.search_bm25.tool import SearchBm25
    from agent_search.tools.search_dense.tool import SearchDense
    from agent_search.tools.search_hybrid.tool import SearchHybrid
    from agent_search.tools.visit.tool import Visit
    engines = {"bm25": StubBm25(), "dense": StubDense()}
    if variant == "bm25":
        old = Bm25Visit(UNITS, engine=engines["bm25"], ubyid=UBYID); names = ("bm25_search", "visit")
        new = _toolbox([SearchBm25(name="bm25_search"), Visit(name="visit")], engines)
    elif variant == "bm25q":
        old = Bm25Visit(UNITS, engine=engines["bm25"], ubyid=UBYID, query_biased=True); names = ("bm25q_search", "visit_q")
        new = _toolbox([SearchBm25(name="bm25q_search", query_biased=True), Visit(name="visit_q")], engines)
    elif variant == "dense":
        old = DenseVisit(UNITS, engine=engines["dense"], ubyid=UBYID); names = ("dense_search", "visit_d")
        new = _toolbox([SearchDense(name="dense_search"), Visit(name="visit_d")], engines)
    else:
        old = HybridVisit(UNITS, bm25_engine=engines["bm25"], dense_engine=engines["dense"], ubyid=UBYID); names = ("hybrid_search", "visit_h")
        new = _toolbox([SearchHybrid(name="hybrid_search"), Visit(name="visit_h")], engines)
    old_out, old_surf = _drive(old, *names)
    new_out, new_surf = _drive(new, *names)
    assert new_out == old_out
    assert new_surf == old_surf


@pytest.mark.parametrize("variant", ["bm25", "dense", "hybrid"])
def test_autoread_family_matches_the_old_workspaces(variant):
    from agent_search.agent.tools.search_visit import Bm25AutoRead, DenseAutoRead, HybridAutoRead
    from agent_search.tools.search_bm25.tool import SearchBm25
    from agent_search.tools.search_dense.tool import SearchDense
    from agent_search.tools.search_hybrid.tool import SearchHybrid
    engines = {"bm25": StubBm25(), "dense": StubDense()}
    calls = [("s", {"query": "treaty"}), ("s", {"query": "nothing"}), ("s", {"query": "war"})]
    if variant == "bm25":
        old = Bm25AutoRead(UNITS, engine=engines["bm25"], ubyid=UBYID); name = "bm25_read_search"
        new = _toolbox([SearchBm25(name=name, full_text=True)], engines)
    elif variant == "dense":
        old = DenseAutoRead(UNITS, engine=engines["dense"], ubyid=UBYID); name = "dense_read_search"
        new = _toolbox([SearchDense(name=name, full_text=True)], engines)
    else:
        old = HybridAutoRead(UNITS, bm25_engine=engines["bm25"], dense_engine=engines["dense"], ubyid=UBYID); name = "hybrid_read_search"
        new = _toolbox([SearchHybrid(name=name, full_text=True)], engines)
    assert [new.run(name, a) for _, a in calls] == [old.run(name, a) for _, a in calls]
    assert list(new.surfaced) == list(old.surfaced)


# =============================================================================================
# The BQL sieve (search_bql/fetch) and the code arm (fetch_code) — sieve.py/code_fix.py.
# =============================================================================================

from agent_search.retrievers.bql.executor import (
    DenseOnlyStructuralExecutor, StructuralExecutor)

BQL_DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today.",
     "infobox": "Founded: 1897; Location: Portville"},
    {"_id": "d_treaty", "title": "Treaty of Guadalupe Hidalgo",
     "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848.\n\n"
             "## Terms\nMexico ceded land."},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    {"_id": "65405", "title": "Integer Id Doc",
     "text": "A document whose id is an integer string."},
]
BQL_UNITS = units_from_documents(BQL_DOCS)
BQL_UBYID = {u.doc_id: u for u in BQL_UNITS}

BQL_CALLS = [
    ("search", {"query": "harbor[title]", "k": 5}),                    # fielded, exact hit
    ("fetch", {"specs": [[1, "History"]]}),                            # fetch by rank
    ("fetch", {"specs": [[1, "infobox"]]}),                            # infobox fetch
    ("fetch", {"specs": [[1, "Nonexistent"]]}),                        # bad section
    ("search", {"query": "harbor[title] AND spaceship[body]", "k": 5}),  # AND, 0 exact hits
    ("search", {"query": "zzznotarealword[title]", "k": 5}),           # 0-hit query
    ("search", {"query": ""}),                                        # empty query
    ("fetch", {"specs": [["d_flat", "(intro)"]]}),                     # fetch by doc_id
    ("fetch", {"specs": [[99, "(intro)"]]}),                           # bad rank
    ("fetch", {"specs": [["65405", "(intro)"]]}),                      # integer doc_id, not a rank
    ("search", {"query": "treaty[title]", "k": 5}),
]


def _bql_toolbox(search_tool, fetch_tool, engine_kind, executor):
    state = EpisodeState(question="q")
    sb = search_tool.bind(state, BQL_UNITS, BQL_UBYID, {engine_kind: executor})
    fe = fetch_tool.bind(state, BQL_UNITS, BQL_UBYID, {})
    return ToolBox([sb, fe], state)


@pytest.mark.parametrize("variant", ["plain", "snippets", "coverage_date_nudge"])
def test_bql_sieve_family_matches_the_old_workspace(variant):
    from agent_search.agent.tools.sieve import DocSearchFetch
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_bql.tool import SearchBql

    ex_old = StructuralExecutor(BQL_UNITS).prewarm()
    ex_new = StructuralExecutor(BQL_UNITS).prewarm()
    if variant == "plain":
        old = DocSearchFetch(BQL_UNITS, executor=ex_old, ubyid=BQL_UBYID)
        new_search = SearchBql(name="search")
    elif variant == "snippets":
        old = DocSearchFetch(BQL_UNITS, executor=ex_old, ubyid=BQL_UBYID, snippets=True)
        new_search = SearchBql(name="search", snippets=True)
    else:
        old = DocSearchFetch(BQL_UNITS, executor=ex_old, ubyid=BQL_UBYID,
                             coverage=True, date_nudge=True)
        new_search = SearchBql(name="search", coverage=True, date_nudge=True)
    new = _bql_toolbox(new_search, Fetch(name="fetch"), "bql", ex_new)

    old_out = [old.run(name, dict(args)) for name, args in BQL_CALLS]
    new_out = [new.run(name, dict(args)) for name, args in BQL_CALLS]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)


def test_bqlvisit_matches_the_old_workspace():
    """`BqlVisitWorkspace`: the same BQL search, forced coverage/date_nudge/snippets, paired
    with a whole-doc `visit` instead of `fetch`."""
    from agent_search.agent.tools.sieve import BqlVisitWorkspace
    from agent_search.tools.search_bql.tool import SearchBql
    from agent_search.tools.visit.tool import Visit

    ex_old = StructuralExecutor(BQL_UNITS).prewarm()
    ex_new = StructuralExecutor(BQL_UNITS).prewarm()
    old = BqlVisitWorkspace(BQL_UNITS, executor=ex_old, ubyid=BQL_UBYID)

    state = EpisodeState(question="q")
    sb = SearchBql(name="search", coverage=True, date_nudge=True, snippets=True).bind(
        state, BQL_UNITS, BQL_UBYID, {"bql": ex_new})
    vb = Visit(name="visit").bind(state, BQL_UNITS, BQL_UBYID, {})
    new = ToolBox([sb, vb], state)

    calls = [
        ("search", {"query": "harbor[title]", "k": 5}),
        ("visit", {"rank": 1}),
        ("search", {"query": "war fought in 1980s", "k": 5}),          # bare temporal clue
        ("visit", {"rank": "d_flat"}),
        ("search", {"query": "harbor[title] AND spaceship[body]", "k": 5}),
        ("search", {"query": "zzznotarealword[title]", "k": 5}),
        ("search", {"query": ""}),
        ("visit", {"rank": 99}),
        ("visit", {"rank": "65405"}),
        ("visit", {"rank": "Rio"}),
    ]
    old_out = [old.run(name, dict(args)) for name, args in calls]
    new_out = [new.run(name, dict(args)) for name, args in calls]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)


def test_bql_donly_snip_matches_the_old_workspace():
    """`DocSearchFetchDonlySnip`: `DocSearchFetch(snippets=True)` over a dense-only-ranked
    executor (dense unattached here, so ordering degrades to the plain filter/coverage
    structure — the dense-attached fusion math is covered by test_bql_dense.py)."""
    from agent_search.agent.tools.sieve import DocSearchFetchDonlySnip
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_bql.tool import SearchBql

    ex_old = DenseOnlyStructuralExecutor(BQL_UNITS).prewarm()
    ex_new = DenseOnlyStructuralExecutor(BQL_UNITS).prewarm()
    old = DocSearchFetchDonlySnip(BQL_UNITS, executor=ex_old, ubyid=BQL_UBYID, snippets=True)
    new = _bql_toolbox(SearchBql(name="search", snippets=True, ranking="dense"),
                       Fetch(name="fetch"), "bql_dense", ex_new)

    old_out = [old.run(name, dict(args)) for name, args in BQL_CALLS]
    new_out = [new.run(name, dict(args)) for name, args in BQL_CALLS]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)


CODE_SRC = (
    "class Command:\n"
    "    def handle(self, *args, **options):\n"
    "        return self.sqlmigrate(args)\n"
    "\n"
    "    def sqlmigrate(self, args):\n"
    "        # cannot rollback DDL on this backend\n"
    "        return atomic(args)\n"
    "\n"
    "def atomic(x):\n"
    "    return x\n"
)


def test_code_fix_matches_the_old_workspace():
    """`CodeFixWorkspace`: search groups hits by FILE (not by unit); fetch pulls a named
    function/method or an L-range. Neither tracks `seen`/`surfaced` in the old workspace, so
    only the rendered text is compared here."""
    from agent_search.agent.tools.code_fix import CodeFixWorkspace
    from agent_search.corpus.units import units_from_python_source
    from agent_search.tools.fetch_code.tool import FetchCode, SearchCode

    files = {"core/mgmt.py": CODE_SRC}
    units = units_from_python_source("core/mgmt.py", CODE_SRC)
    ubyid = {u.doc_id: u for u in units}
    ex_old = StructuralExecutor(units).prewarm()
    ex_new = StructuralExecutor(units).prewarm()
    old = CodeFixWorkspace(units, files, executor=ex_old, ubyid=ubyid)

    state = EpisodeState(question="q")
    sc = SearchCode(name="search").bind(state, units, ubyid, {"bql_plain": ex_new}, files=files)
    fc = FetchCode(name="fetch").bind(state, units, ubyid, {}, files=files)
    new = ToolBox([sc, fc], state)

    calls = [
        ("search", {"query": "sqlmigrate[def]", "k": 5}),               # fielded, exact hit
        ("fetch", {"specs": [[1, "Command.sqlmigrate"]]}),              # fetch by rank + qualname
        ("fetch", {"specs": [[1, "sqlmigrate"]]}),                      # fetch by bare name (suffix match)
        ("search", {"query": '"cannot rollback DDL"[comment]', "k": 5}),  # fielded, comment scope
        ("fetch", {"specs": [[1, "L1-3"]]}),                            # fetch by L-range
        ("fetch", {"specs": [[9, "x"]]}),                                # bad rank
        ("fetch", {"specs": [[1, "nonexistent_method"]]}),              # bad part
        ("search", {"query": "sqlmigrate atomic handle", "k": 5}),      # 0-hit phrase -> OR rerun
        ("search", {"query": "x[module]", "k": 5}),                     # unknown field -> error
        ("search", {"query": ""}),                                      # empty query
    ]
    old_out = [old.run(name, dict(args)) for name, args in calls]
    new_out = [new.run(name, dict(args)) for name, args in calls]
    assert new_out == old_out


# =============================================================================================
# DCI (bash/read over a flat corpus export), bounded DCI (bm25_search staging into the same
# shell), the code grep+read baseline, and ITER's dedup search+get_document.
# =============================================================================================

def _bind(tools, engines=None, corpus_key=None, units=UNITS, ubyid=UBYID, files=None, question="q"):
    state = EpisodeState(question=question)
    bound = [t.bind(state, units, ubyid, engines or {}, files=files, corpus_key=corpus_key) for t in tools]
    return ToolBox(bound, state)


def test_dci_bash_read_matches_the_old_workspace(tmp_path, monkeypatch):
    """`Bash`+`Read(source="export")` over the whole flat export match `DciWorkspace` call
    for call, including the shell-shaped bash errors and the corpus-escaping read guard."""
    monkeypatch.setenv("AGENT_SEARCH_DCI_CACHE", str(tmp_path))
    from agent_search.agent.tools.doc_dci import DciWorkspace
    from agent_search.tools.bash.tool import Bash
    from agent_search.tools.read.tool import Read

    old = DciWorkspace(UNITS, corpus_key="dci_parity_old")
    new = _bind([Bash(), Read()], corpus_key="dci_parity_new")

    calls = [
        ("bash", {"command": "ls"}),
        ("bash", {"command": "grep -rl 'Guadalupe Hidalgo' ."}),
        ("read", {"path": "d1.txt"}),
        ("read", {"path": "d1.txt", "offset": 2, "limit": 1}),
        ("read", {"path": "nonexistent_doc.txt"}),
        ("read", {"path": "../../etc/passwd"}),
        ("bash", {"command": ""}),
        ("bash", {"command": "grep -rl 'zz_definitely_absent_zz' ."}),
    ]
    old_out = [old.run(name, dict(args)) for name, args in calls]
    new_out = [new.run(name, dict(args)) for name, args in calls]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)


def test_bounded_dci_bm25_search_bash_read_matches_the_old_workspace():
    """`SearchBm25Dci`+`Bash(bounded=True)`+`Read` match `Bm25DciWorkspace`: retrieval is live
    on every call, hits are staged incrementally, and the empty-hit rendering has no "previous
    results" suffix (the reason this is its own tool, not a `SearchBm25` option)."""
    from agent_search.agent.tools.doc_bm25_dci import Bm25DciWorkspace
    from agent_search.tools.bash.tool import Bash
    from agent_search.tools.read.tool import Read
    from agent_search.tools.search_bm25_dci.tool import SearchBm25Dci

    engine = StubBm25()
    old = Bm25DciWorkspace(UNITS, "treaty", engine=engine, ubyid=UBYID, topk=2)
    new = _bind([SearchBm25Dci(topk=2), Bash(bounded=True), Read()],
               engines={"bm25": engine}, question="treaty")

    calls = [
        ("bm25_search", {"query": "treaty"}),
        ("bash", {"command": "ls"}),
        ("bm25_search", {"query": "war"}),
        ("bash", {"command": "ls"}),
        ("read", {"path": "d1.txt"}),
        ("bm25_search", {"query": "nothing"}),
        ("bm25_search", {"query": ""}),
    ]
    old_out = [old.run(name, dict(args)) for name, args in calls]
    new_out = [new.run(name, dict(args)) for name, args in calls]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)
    old.close()


def test_grep_read_matches_the_old_workspace():
    """`Grep`+`Read(source="repo")` over the code fixture's files match `GrepReadWorkspace`:
    real-regex line hits, the invalid-regex literal fallback, and the bare-basename read."""
    from agent_search.agent.tools.code_grep import GrepReadWorkspace
    from agent_search.corpus.units import units_from_python_source
    from agent_search.evaluation.datasets.fixtures import fixture_instances
    from agent_search.tools.grep.tool import Grep
    from agent_search.tools.read.tool import Read

    files = fixture_instances()[0].files
    units = []
    for path, src in files.items():
        units.extend(units_from_python_source(path, src))

    old = GrepReadWorkspace(units, files)
    new = _bind([Grep(), Read(source="repo")], units=units,
               ubyid={u.doc_id: u for u in units}, files=files)

    calls = [
        ("grep", {"pattern": "create_session_token"}),
        ("grep", {"pattern": "MAKE_TOKEN"}),
        ("grep", {"pattern": "make_token("}),                  # invalid regex -> literal fallback
        ("grep", {"pattern": "zz_not_a_real_token_zz"}),
        ("read", {"path": "auth/session.py", "start": 1, "end": 3}),
        ("read", {"path": "session.py", "start": 1, "end": 2}),   # bare basename suffix match
        ("read", {"path": "nope.py"}),
        ("grep", {"query": "render_page"}),                    # query alias
    ]
    old_out = [old.run(name, dict(args)) for name, args in calls]
    new_out = [new.run(name, dict(args)) for name, args in calls]
    assert new_out == old_out


def test_dedup_search_and_get_document_matches_the_old_workspace():
    """`SearchDedup(ranking="bm25")`+`GetDocument` match `DedupSearchWorkspace`: the over-fetch
    pool, the drop-seen dedup, the "Already-seen" listing on a repeated search, and
    `get_document` by `DocID:`-prefixed id, bare id, numeric rank fallback, and a bad id."""
    from agent_search.agent.tools.doc_dedup import DedupSearchWorkspace
    from agent_search.tools.get_document.tool import GetDocument
    from agent_search.tools.search_dedup.tool import SearchDedup

    class _StubBm25Pool:
        def __init__(self, order):
            self._order = order

        def search(self, query, k=5):
            return self._order[:k]

    order = ["d1", "d2", "d3", "d4", "d5"]
    old = DedupSearchWorkspace(UNITS, lambda q, k: order[:k], ubyid=UBYID, pool_k=4, top_k=2)
    new = _bind([SearchDedup(ranking="bm25", pool_k=4, top_k=2), GetDocument()],
               engines={"bm25": _StubBm25Pool(order)})

    calls = [
        ("search", {"query": "treaty"}),
        ("search", {"query": "treaty again"}),                 # repeated: triggers Already-seen
        ("get_document", {"docid": "DocID:d1"}),
        ("get_document", {"docid": "999"}),                    # bad id
        ("get_document", {"docid": "d2"}),
        ("search", {"query": ""}),
        ("get_document", {"docid": "1"}),                      # numeric rank fallback into last_hits
    ]
    old_out = [old.run(name, dict(args)) for name, args in calls]
    new_out = [new.run(name, dict(args)) for name, args in calls]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)


# =============================================================================================
# The search-fetch family (bm25/dense/hybrid live retrieval + structured section fetch) and
# the Indri family (graded belief search, paired with fetch or with a whole-doc visit).
# =============================================================================================

# a search, a fetch by rank, a fetch by doc_id + named section, a bad rank, a second search —
# reused for every search-fetch variant (same small fixture corpus as the search-visit family).
FETCH_CALLS = [
    ("search", {"query": "treaty"}),
    ("fetch", {"specs": [[1, ""]]}),
    ("fetch", {"specs": [["d1", "Terms"]]}),
    ("fetch", {"specs": [[99, ""]]}),
    ("search", {"query": "war"}),
]


def _drive_named(ws, calls, search_name):
    out = []
    for name, args in calls:
        real = search_name if name == "search" else name
        out.append(ws.run(real, dict(args)))
    return out, list(ws.surfaced)


def test_bm25fetch_matches_the_old_workspace():
    from agent_search.agent.tools.search_fetch import Bm25FetchWorkspace
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_bm25.tool import SearchBm25

    old = Bm25FetchWorkspace(UNITS, "treaty", engine=StubBm25(), ubyid=UBYID)
    new = _toolbox([SearchBm25(name="bm25_search", structure=True), Fetch(name="fetch")],
                   {"bm25": StubBm25()})

    old_out, old_surf = _drive_named(old, FETCH_CALLS, "bm25_search")
    new_out, new_surf = _drive_named(new, FETCH_CALLS, "bm25_search")
    assert new_out == old_out
    assert new_surf == old_surf


def test_bm25fetchsnip_matches_the_old_workspace():
    from agent_search.agent.tools.search_fetch import Bm25FetchSnipWorkspace
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_bm25.tool import SearchBm25

    old = Bm25FetchSnipWorkspace(UNITS, "treaty", engine=StubBm25(), ubyid=UBYID)
    new = _toolbox([SearchBm25(name="bm25_search_snip", structure=True, snippets=True), Fetch(name="fetch")],
                   {"bm25": StubBm25()})

    old_out, old_surf = _drive_named(old, FETCH_CALLS, "bm25_search_snip")
    new_out, new_surf = _drive_named(new, FETCH_CALLS, "bm25_search_snip")
    assert new_out == old_out
    assert new_surf == old_surf


def test_densefetch_matches_the_old_workspace():
    from agent_search.agent.tools.search_fetch import DenseFetchWorkspace
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_dense.tool import SearchDense

    old = DenseFetchWorkspace(UNITS, "treaty", engine=StubDense(), ubyid=UBYID)
    new = _toolbox([SearchDense(name="dense_search_f", structure=True, snippets=True), Fetch(name="fetch")],
                   {"dense": StubDense()})

    old_out, old_surf = _drive_named(old, FETCH_CALLS, "dense_search_f")
    new_out, new_surf = _drive_named(new, FETCH_CALLS, "dense_search_f")
    assert new_out == old_out
    assert new_surf == old_surf


def test_densefetchplain_matches_the_old_workspace():
    from agent_search.agent.tools.search_fetch import DenseFetchPlainWorkspace
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_dense.tool import SearchDense

    old = DenseFetchPlainWorkspace(UNITS, "treaty", engine=StubDense(), ubyid=UBYID)
    new = _toolbox([SearchDense(name="dense_search_fp", structure=True, snippets=False), Fetch(name="fetch")],
                   {"dense": StubDense()})

    old_out, old_surf = _drive_named(old, FETCH_CALLS, "dense_search_fp")
    new_out, new_surf = _drive_named(new, FETCH_CALLS, "dense_search_fp")
    assert new_out == old_out
    assert new_surf == old_surf


def test_hybridfetchsnip_matches_the_old_workspace():
    from agent_search.agent.tools.search_fetch import HybridFetchSnipWorkspace
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_hybrid.tool import SearchHybrid

    old = HybridFetchSnipWorkspace(UNITS, "treaty", bm25_engine=StubBm25(), dense_engine=StubDense(), ubyid=UBYID)
    new = _toolbox([SearchHybrid(name="hybrid_search_snip", structure=True, snippets=True), Fetch(name="fetch")],
                   {"bm25": StubBm25(), "dense": StubDense()})

    old_out, old_surf = _drive_named(old, FETCH_CALLS, "hybrid_search_snip")
    new_out, new_surf = _drive_named(new, FETCH_CALLS, "hybrid_search_snip")
    assert new_out == old_out
    assert new_surf == old_surf


# -- Indri: a graded belief search over the same small fixture corpus, paired with fetch or
# a whole-doc visit -----------------------------------------------------------------------

ISEARCH_FETCH_CALLS = [
    ("search", {"query": "treaty of hidalgo"}),
    ("fetch", {"specs": [[1, ""]]}),
    ("fetch", {"specs": [["d1", "Terms"]]}),
    ("fetch", {"specs": [[99, ""]]}),
    ("search", {"query": "war"}),
]

ISEARCH_VISIT_CALLS = [
    ("search", {"query": "treaty of hidalgo"}),
    ("visit", {"rank": 1}),
    ("visit", {"rank": "d1"}),
    ("visit", {"rank": 99}),
    ("search", {"query": "war"}),
]


def test_indri_matches_the_old_workspace():
    from agent_search.agent.tools.doc_indri import IndriFetchWorkspace
    from agent_search.retrievers.indri.model import IndriExecutor
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_indri.tool import SearchIndri

    old = IndriFetchWorkspace(UNITS, executor=IndriExecutor(UNITS), ubyid=UBYID)
    new = _toolbox([SearchIndri(name="isearch"), Fetch(name="fetch")], {"indri": IndriExecutor(UNITS)})

    old_out, old_surf = _drive_named(old, ISEARCH_FETCH_CALLS, "isearch")
    new_out, new_surf = _drive_named(new, ISEARCH_FETCH_CALLS, "isearch")
    assert new_out == old_out
    assert new_surf == old_surf


def test_indrisnip_matches_the_old_workspace():
    from agent_search.agent.tools.doc_indri import IndriFetchWorkspace
    from agent_search.retrievers.indri.model import IndriExecutor
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_indri.tool import SearchIndri

    old = IndriFetchWorkspace(UNITS, executor=IndriExecutor(UNITS), ubyid=UBYID, snippets=True)
    new = _toolbox([SearchIndri(name="isearch_s", snippets=True), Fetch(name="fetch")],
                   {"indri": IndriExecutor(UNITS)})

    old_out, old_surf = _drive_named(old, ISEARCH_FETCH_CALLS, "isearch_s")
    new_out, new_surf = _drive_named(new, ISEARCH_FETCH_CALLS, "isearch_s")
    assert new_out == old_out
    assert new_surf == old_surf


def test_indrivisit_matches_the_old_workspace():
    from agent_search.agent.tools.doc_indri import IndriVisitWorkspace
    from agent_search.retrievers.indri.model import IndriExecutor
    from agent_search.tools.search_indri.tool import SearchIndri
    from agent_search.tools.visit.tool import Visit

    old = IndriVisitWorkspace(UNITS, executor=IndriExecutor(UNITS), ubyid=UBYID)
    new = _toolbox([SearchIndri(name="isearch_v", snippets=True), Visit(name="visit_v")],
                   {"indri": IndriExecutor(UNITS)})

    old_out = [old.run(name if name != "visit" else "visit_v", dict(args)) for name, args in ISEARCH_VISIT_CALLS]
    new_out = [new.run("isearch_v" if name == "search" else "visit_v", dict(args)) for name, args in ISEARCH_VISIT_CALLS]
    assert new_out == old_out
    assert list(new.surfaced) == list(old.surfaced)


# -- declarations: what the model sees for the new tool names must be byte-identical to the
# corresponding tools.yaml entry (the same JSON the old YAML-backed loader rendered) --------

def test_search_fetch_and_indri_declarations_match_tools_yaml():
    import yaml
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_bm25.tool import SearchBm25
    from agent_search.tools.search_dense.tool import SearchDense
    from agent_search.tools.search_hybrid.tool import SearchHybrid
    from agent_search.tools.search_indri.tool import SearchIndri

    yaml_tools = yaml.safe_load(
        (Path(__file__).parent.parent / "agent_search" / "legacy" / "prompts" / "tools.yaml").read_text()
    )["tools"]

    def _decl(name, description, parameters):
        return {"name": name, "description": description, "parameters": parameters}

    cases = [
        SearchBm25(name="bm25_search_snip", structure=True, snippets=True),
        SearchDense(name="dense_search_f", structure=True, snippets=True),
        SearchHybrid(name="hybrid_search_snip", structure=True, snippets=True),
        Fetch(name="fetch"),
        SearchIndri(name="isearch_s", snippets=True),
    ]
    for tool in cases:
        expected = yaml_tools[tool.name]
        assert tool.declaration() == _decl(tool.name, expected["description"], expected["parameters"])

    # dense_search_fp is not in tools.yaml (an unused arm); it gets dense_search_f's declaration.
    plain = SearchDense(name="dense_search_fp", structure=True, snippets=False)
    expected = yaml_tools["dense_search_f"]
    assert plain.description == expected["description"]
    assert plain.parameters == expected["parameters"]
