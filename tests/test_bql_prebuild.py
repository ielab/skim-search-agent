"""The BQL structural index is pre-buildable offline (Phase 1 / STEP 0) and loaded
by the agent, so its O(N) postings+BM25 build is paid ONCE off the clock — never
inside the first search_bql tool call of an episode (the localize/deep-research tail).

Mirrors the dense/pyserini persistent-index contract: build_indexes.py writes it,
the agent's index() loads it, and a missing/corrupt artifact silently rebuilds so
correctness never depends on the cache.
"""
import tempfile

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql.executor import (
    BQLIndexBuilder, StructuralExecutor, bql_index_path, execute_bql, load_or_build,
)


def _units(n=12):
    return [CodeUnit(doc_id=f"d{i}", path=f"d{i}", qualname=f"Doc {i}", start_line=0,
                     end_line=0, code="", title=f"Title {i}", body="treaty election recount",
                     section="", metadata={"author": "x", "date": "1848"}) for i in range(n)]


def test_prebuildable_for_lists_search_bql_for_search_fetch_conditions():
    from evaluation.build_indexes import prebuildable_for
    # the search -> fetch arms (`search` tool) lower to the executor -> pre-build its index.
    assert prebuildable_for("agent_codefix") == ["search_bql"]
    assert prebuildable_for("agent_research") == ["search_bql"]
    assert prebuildable_for("bql") == ["search_bql"]      # the direct BQL floor loads the same artifact
    # the retrieve-then-visit baseline uses in-memory BM25Local — no step 0.
    assert prebuildable_for("agent_research_bm25") == []
    assert prebuildable_for("grep") == []
    # the two NEW baselines are also in-memory/index-free — no persistent step 0:
    # codefix_grep re-scans the units live (like GrepBaseline); research_dci's flat-file
    # export is its own per-corpus cache, not a retriever-level prebuild artifact.
    assert prebuildable_for("agent_codefix_grep") == []
    assert prebuildable_for("agent_research_dci") == []


def test_builder_writes_artifact_and_reports_cached():
    root = tempfile.mkdtemp()
    b = BQLIndexBuilder(index_root=root)
    key = "hotpotqa_flat@corpus"
    assert not b.is_cached(key)
    b.index(_units(), key=key)
    assert b.is_cached(key)


def test_load_or_build_loads_prebuilt_without_reconstructing():
    root = tempfile.mkdtemp()
    key = "corpus@v1"
    us = _units()
    BQLIndexBuilder(index_root=root).index(us, key=key)
    ex = load_or_build(us, index_root=root, key=key)      # must come from disk, not a fresh build
    r = execute_bql("IN(body, treaty)", ex, ex._ubyid, k=5)
    assert r.typecheck_ok and r.n_hits == len(us)         # every unit's body has "treaty"


def test_load_or_build_falls_back_when_no_artifact():
    # no index_root/key -> in-memory build (the unchanged path); still fully queryable.
    ex = load_or_build(_units(), index_root=None, key=None)
    assert isinstance(ex, StructuralExecutor)
    assert execute_bql("IN(body, election)", ex, ex._ubyid, k=5).n_hits == 12


def test_saved_index_round_trips_identically():
    us = _units()
    live = StructuralExecutor(us)
    path = bql_index_path(tempfile.mkdtemp(), "k")
    StructuralExecutor(us).save(path)
    loaded = StructuralExecutor.load(path)
    loaded.attach_units(us)          # slim pkl persists only the index; re-attach units (as load_or_build does)
    q = "OR(IN(body, treaty), IN(title, Title))"
    assert (execute_bql(q, loaded, loaded._ubyid, k=20).n_hits
            == execute_bql(q, live, live._ubyid, k=20).n_hits)
