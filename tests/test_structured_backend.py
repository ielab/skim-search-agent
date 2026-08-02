"""Tests for `STRUCTURED_BACKEND` (python|lucene) — the env knob that swaps the pure-Python
Indri/BQL reference engines for `LuceneStructuredEngine` (`agent_search/retrievers/structural/
lucene/`) underneath the SAME doc-arm workspaces (`IndriFetchWorkspace`/`IndriVisitWorkspace`,
`DocSearchFetch`/`BqlVisitWorkspace`), unchanged. See `structural/backend.py`'s module
docstring (the resolver) and `structural/lucene/adapters.py`'s module docstring (the two
adapters + documented deviations, incl. dense-belief score-normalization) for the design.

1. Backend-selection: `structured_backend()`'s env resolution, `build_indri_engine`/
   `build_bql_engine`'s dispatch (python -> the real Python executor class; lucene -> the
   adapter class, requires a real `key`), and the workspaces' own `executor=None` fallback
   respecting the knob (mirrors `build_bm25_engine`'s fallback pattern).
2. Interface parity: the SAME tool calls (`search_bv`/`visit_bv`, `isearch_v`/`visit_v`)
   against a python-backed vs lucene-backed workspace over the IDENTICAL fixture corpus —
   listings render, ranks resolve, the 0-hit coverage-fallback (delegated to a lazy in-memory
   `StructuralExecutor` under the lucene backend — see adapters.py) still works.
3. Dense fusion with the lucene backend (`LuceneIndriAdapter`'s score-normalization design):
   a stub dense source reranks Lucene's own returned pool at high `INDRI_DENSE_W`, is a no-op
   at `w=0` (original Lucene order/scores preserved), still emits a `#dense` diagnostics entry
   at `w=0` (visible without affecting ranking, mirroring `indri/model.py`), and a raising
   dense source degrades silently to lexical-only (never crashes the search call).

Requires a real JVM (pyserini/pyjnius) — same module-scoped skip-if-unavailable guard as
`tests/test_lucene_structured.py` (see that file's docstring for the JVM-boot landmine this
sidesteps: `_jni._boot()` directly, never a bare `pytest.importorskip("jnius", ...)`).
"""
from __future__ import annotations

import pytest

from agent_search.agent.tools.doc_indri import IndriFetchWorkspace, IndriVisitWorkspace
from agent_search.agent.tools.doc_research import BqlVisitWorkspace, DocSearchFetch
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql.executor import StructuralExecutor
from agent_search.retrievers.structural.indri.model import IndriExecutor
from agent_search.retrievers.structural.lucene import index_builder
from agent_search.retrievers.structural.lucene import jni_utils as _jni
from agent_search.retrievers.structural.lucene.engine import LuceneStructuredEngine

try:
    _jni._boot()
except Exception as e:                          # pragma: no cover - environment-dependent
    pytest.skip(f"lucene backend needs a working JVM (pyserini/pyjnius): {e}",
                allow_module_level=True)

from agent_search.retrievers.structural.backend import (       # noqa: E402
    build_bql_engine, build_indri_engine, structured_backend)
from agent_search.retrievers.structural.lucene.adapters import (  # noqa: E402
    LuceneBqlAdapter, LuceneIndriAdapter)

_DATASET_KEY = "fixture_backend_test"


def _mk(doc_id: str, body: str, title: str | None = None) -> CodeUnit:
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=body, body=body, title=title)


def _corpus() -> list[CodeUnit]:
    return [
        # --- constraint-coverage fixture (mirrors test_bql_visit.py's, same expected ranks) --
        _mk("docA", "## History\nfoo bar baz text here", title="Doc A"),
        _mk("docB", "## History\nfoo bar text", title="Doc B"),
        _mk("docC", "## History\nnothing relevant here", title="Doc C"),
        _mk("docD", "## History\nfoo only here", title="Doc D"),
        # --- field-tagged search + section/visit fixture ---
        _mk("d_harbor", "Harbor Festival is an annual event.\n\n## History\n"
                        "Founded in 1897 by A. Smith.\n\n## Legacy\nStill held today.",
            title="Harbor Festival"),
        _mk("d_flat", "The Adams-Onis Treaty of 1819 concerned Florida.",
            title="Adams-Onis Treaty"),
        # --- Indri graded-search fixture ---
        _mk("both", "dog train dog train station", title="Trains and Dogs"),
        _mk("onlydog", "dog dog dog dog park walk", title="Dog Park"),
        _mk("onlytrain", "train station schedule arrival", title="Train Schedule"),
    ]


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture(scope="module")
def index_root(tmp_path_factory, units) -> str:
    root = str(tmp_path_factory.mktemp("lucene_structured_backend"))
    index_builder.build(units, root, _DATASET_KEY, progress=False)
    return root


@pytest.fixture(scope="module")
def engine(index_root) -> LuceneStructuredEngine:
    return LuceneStructuredEngine(index_root=index_root, dataset=_DATASET_KEY)


# === 1. backend-selection ==========================================================


def test_default_backend_is_python(monkeypatch):
    monkeypatch.delenv("STRUCTURED_BACKEND", raising=False)
    assert structured_backend() == "python"


def test_env_selects_lucene(monkeypatch):
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    assert structured_backend() == "lucene"


def test_unknown_backend_value_raises_loud(monkeypatch):
    monkeypatch.setenv("STRUCTURED_BACKEND", "solr")
    with pytest.raises(ValueError, match="STRUCTURED_BACKEND"):
        structured_backend()


def test_build_indri_engine_default_returns_python_executor(units, monkeypatch):
    monkeypatch.delenv("STRUCTURED_BACKEND", raising=False)
    assert isinstance(build_indri_engine(units), IndriExecutor)


def test_build_bql_engine_default_returns_python_executor(units, monkeypatch):
    monkeypatch.delenv("STRUCTURED_BACKEND", raising=False)
    assert isinstance(build_bql_engine(units), StructuralExecutor)


def test_build_indri_engine_lucene_needs_a_real_key(units, monkeypatch):
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    with pytest.raises(ValueError, match="key"):
        build_indri_engine(units)


def test_build_bql_engine_lucene_needs_a_real_key(units, monkeypatch):
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    with pytest.raises(ValueError, match="key"):
        build_bql_engine(units)


def test_build_indri_engine_lucene_returns_adapter(units, monkeypatch, index_root):
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    eng = build_indri_engine(units, index_root=index_root, key=_DATASET_KEY)
    assert isinstance(eng, LuceneIndriAdapter)


def test_build_bql_engine_lucene_returns_adapter(units, monkeypatch, index_root):
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    eng = build_bql_engine(units, index_root=index_root, key=_DATASET_KEY)
    assert isinstance(eng, LuceneBqlAdapter)


class _StubDense:
    def __init__(self, sims: dict):
        self.sims = sims

    def score(self, query_text, doc_ids=None):
        ids = doc_ids if doc_ids is not None else list(self.sims)
        return {d: self.sims.get(d, 0.0) for d in ids}


def test_build_indri_engine_lucene_wires_dense_through(units, monkeypatch, index_root):
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    stub = _StubDense({})
    eng = build_indri_engine(units, index_root=index_root, key=_DATASET_KEY, dense=stub)
    assert isinstance(eng, LuceneIndriAdapter)
    assert eng.dense is stub


def test_workspace_default_executor_respects_backend(units, monkeypatch):
    monkeypatch.delenv("STRUCTURED_BACKEND", raising=False)
    assert isinstance(DocSearchFetch(units).ex, StructuralExecutor)
    assert isinstance(IndriFetchWorkspace(units).iex, IndriExecutor)


def test_workspace_default_executor_lucene_backend_raises_without_key(units, monkeypatch):
    # ad-hoc construction (no executor=) under STRUCTURED_BACKEND=lucene has no dataset key
    # to resolve a prebuilt index from — fails loud rather than silently using python.
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    with pytest.raises(ValueError, match="key"):
        DocSearchFetch(units)
    with pytest.raises(ValueError, match="key"):
        IndriFetchWorkspace(units)


# === 2. interface parity: same workspace calls, python vs lucene backend =====================


def test_bql_visit_workspace_parity_search_and_visit(units, engine):
    ws_py = BqlVisitWorkspace(units)
    ws_lu = BqlVisitWorkspace(units, executor=LuceneBqlAdapter(engine, units))
    for ws in (ws_py, ws_lu):
        out = ws.run("search_bv", {"query": "harbor[title]"})
        assert "ERROR" not in out
        assert "d_harbor" in out
        assert "§[" in out and "ib[" in out         # SAME structure table shape
        assert "matched:" in out                     # computed by the workspace, not the engine
        assert "»" in out                             # content excerpt (forced, fairness parity)
        visit_out = ws.run("visit_bv", {"rank": 1})
        assert "ERROR" not in visit_out
        assert "d_harbor" in visit_out
        assert "Harbor Festival" in visit_out


def test_bql_visit_workspace_parity_coverage_fallback(units, engine):
    """The 0-exact-hit constraint-COVERAGE path (BQL v2 Feature 2) — under the lucene
    backend this is served by `LuceneBqlAdapter`'s lazily-built in-memory fallback
    executor (see adapters.py's documented BQL-surface deviation), so the rendering must
    be IDENTICAL to the python backend's own native coverage_topk."""
    ws_py = BqlVisitWorkspace(units)
    ws_lu = BqlVisitWorkspace(units, executor=LuceneBqlAdapter(engine, units))
    query = {"query": "foo[body] AND bar[body] AND baz[body] AND qux[body]"}
    for ws in (ws_py, ws_lu):
        out = ws.run("search_bv", query)
        assert "CONSTRAINT COVERAGE" in out
        assert "miss=[" in out
        lines = out.split("\n")
        assert lines[1].split()[1] == "docA"          # matches foo/bar/baz, misses qux
        assert "cov=3/4" in lines[1]


def test_docsearchfetch_parity_n_hits_header_matches(units, engine):
    """`obs.n_hits` (the search listing's "(N matches, top K)" count) must be the EXACT
    total, not a truncated `len(hits)`, under lucene too (`LuceneBqlAdapter.run_with_count`
    uses `LuceneStructuredEngine.count_bql_expr`, not `len(top-k hits)`)."""
    ws_py = DocSearchFetch(units)
    ws_lu = DocSearchFetch(units, executor=LuceneBqlAdapter(engine, units))
    for ws in (ws_py, ws_lu):
        out = ws.run("search", {"query": "doc[title]", "k": 2})
        assert "ERROR" not in out
        # docA/docB/docC/docD all title-match "doc" exactly; k=2 truncates the listing.
        assert "4 matches, top 2" in out


def test_indri_visit_workspace_parity_search_and_visit(units, engine):
    ws_py = IndriVisitWorkspace(units)
    ws_lu = IndriVisitWorkspace(units, executor=LuceneIndriAdapter(engine))
    for ws in (ws_py, ws_lu):
        out = ws.run("isearch_v", {"query": "#combine(dog train)"})
        assert "ERROR" not in out
        assert "hits):" in out
        assert "»" in out
        visit_out = ws.run("visit_v", {"rank": 1})
        assert "ERROR" not in visit_out


def test_indri_fetch_workspace_parity_section_fetch(units, engine):
    ws_py = IndriFetchWorkspace(units)
    ws_lu = IndriFetchWorkspace(units, executor=LuceneIndriAdapter(engine))
    for ws in (ws_py, ws_lu):
        out = ws.run("isearch", {"query": "#combine(dog train)"})
        assert "ERROR" not in out
        fetch_out = ws.run("fetch", {"specs": [[1, ""]]})
        assert "ERROR" not in fetch_out


# === 3. dense-belief fusion with the lucene backend ===========================================


class _RaisingDense:
    def score(self, query_text, doc_ids=None):
        raise RuntimeError("boom")


def test_dense_none_is_a_pure_lucene_search(engine):
    r = LuceneIndriAdapter(engine).search("#combine(dog train)", k=5)
    assert r.error is None
    assert r.hits
    assert r.diagnostics == []


def test_dense_reranks_lucene_pool_at_high_weight(engine, monkeypatch):
    plain = LuceneIndriAdapter(engine)
    r0 = plain.search("#combine(dog train)", k=5)
    doc_ids = [d for d, _ in r0.hits]
    assert len(doc_ids) >= 2
    boosted = doc_ids[-1]                    # Lucene's WORST-ranked returned hit — boost it
    sims = {d: (10.0 if d == boosted else 0.0) for d in doc_ids}
    monkeypatch.setenv("INDRI_DENSE_W", "0.95")
    r1 = LuceneIndriAdapter(engine, dense=_StubDense(sims)).search("#combine(dog train)", k=5)
    assert [d for d, _ in r1.hits][0] == boosted
    assert any(name == "#dense" for name, _ in r1.diagnostics)


def test_dense_w_zero_preserves_lucene_order_and_scores(engine, monkeypatch):
    plain = LuceneIndriAdapter(engine)
    r0 = plain.search("#combine(dog train)", k=5)
    sims = {d: float(i) for i, (d, _) in enumerate(r0.hits)}    # arbitrary — irrelevant at w=0
    monkeypatch.setenv("INDRI_DENSE_W", "0")
    r1 = LuceneIndriAdapter(engine, dense=_StubDense(sims)).search("#combine(dog train)", k=5)
    assert r1.hits == r0.hits
    # still computed + attached (contribution visible without affecting ranking) — mirrors
    # indri/model.py's own `_combine_dense` w<=0 contract.
    assert any(name == "#dense" for name, _ in r1.diagnostics)


def test_dense_source_raising_degrades_to_lexical_only(engine, monkeypatch):
    plain = LuceneIndriAdapter(engine)
    r0 = plain.search("#combine(dog train)", k=5)
    monkeypatch.setenv("INDRI_DENSE_W", "0.5")
    r1 = LuceneIndriAdapter(engine, dense=_RaisingDense()).search("#combine(dog train)", k=5)
    assert r1.error is None
    assert r1.hits == r0.hits
    assert r1.diagnostics == []
