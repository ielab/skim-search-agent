"""Tests for `agent_search/retrievers/backend.py`: which engine answers a structured query
(BQL, Indri) for a corpus. The corpus kind decides; there is no environment switch.

1. Dispatch by domain: `domain="code"` -> the in-memory `StructuralExecutor` for BQL, and a
   `SetupError` for the document-only engines (dense fusion, dense-only ranking, Indri);
   `domain="general"` -> the Lucene adapters over `indexes/lucene_structured/<key>/`. An
   in-memory corpus gets that index built on first use (and an ad hoc key from its
   fingerprint when none is given); an on-disk corpus with no key or no prebuilt index is a
   `SetupError` before the first search.
2. The document tools on the Lucene engines (`search_bv`/`visit_bv`, `search`/`fetch`,
   `isearch_v`/`visit_v`, `isearch`/`fetch`): listings render, ranks resolve, the exact match
   count reaches the header, and the 0-hit coverage fallback runs on Lucene.
3. Dense fusion on `LuceneIndriAdapter` (score-normalization design, `lucene/adapters.py`
   Deviation 2): a stub dense source reranks Lucene's own returned pool at high
   `INDRI_DENSE_W`, is a no-op at `w=0` (original order and scores preserved) while still
   emitting a `#dense` diagnostics entry, and a raising dense source degrades silently to
   lexical-only.

Needs a JVM (pyserini/pyjnius); the module skips without one.
"""
from __future__ import annotations

import pytest

from agent_search.snippets import TermWindow
from agent_search.corpus.units import CodeUnit
from agent_search.errors import SetupError
from agent_search.retrievers.bql.executor import StructuralExecutor
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bql.tool import SearchBql
from agent_search.tools.search_indri.tool import SearchIndri
from agent_search.tools.visit.tool import Visit
from tests import lucene_support
from tests.lucene_support import build_lucene_bql, build_lucene_indri, require_jvm

require_jvm()

from agent_search.retrievers.backend import (       # noqa: E402
    build_bql_engine, build_bql_engine_dense_only, build_indri_engine, is_code_corpus)
from agent_search.retrievers.lucene.adapters import (  # noqa: E402
    LuceneBqlAdapter, LuceneBqlDonlyAdapter, LuceneIndriAdapter)


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
def index_root() -> str:
    return lucene_support.index_root()


@pytest.fixture(scope="module")
def key(units) -> str:
    """The prebuilt Lucene structured index for `units`."""
    return lucene_support.build_structured_index(units)


def _toolbox(units, tools, engines):
    state = EpisodeState(question="q")
    ubyid = {u.doc_id: u for u in units}
    bound = [t.bind(state, units, ubyid, engines) for t in tools]
    return ToolBox(bound, state)


def _bql_visit_toolbox(units, executor):
    """`search_bv`/`visit_bv`: the old BqlVisitWorkspace's forced options (coverage, date_nudge,
    a per-hit snippet) on top of a whole-doc visit read."""
    return _toolbox(units,
                    [SearchBql(name="search_bv", coverage=True, date_nudge=True, snippet=TermWindow()),
                     Visit(name="visit_bv")],
                    {"bql": executor})


def _docsearchfetch_toolbox(units, executor):
    """The plain search->fetch method, no per-hit excerpt."""
    return _toolbox(units, [SearchBql(name="search"), Fetch(name="fetch")], {"bql": executor})


def _indri_visit_toolbox(units, executor):
    return _toolbox(units, [SearchIndri(name="isearch_v", snippet=TermWindow()), Visit(name="visit_v")],
                    {"indri": executor})


def _indri_fetch_toolbox(units, executor):
    return _toolbox(units, [SearchIndri(name="isearch"), Fetch(name="fetch")], {"indri": executor})


class _StubDense:
    def __init__(self, sims: dict):
        self.sims = sims

    def score(self, query_text, doc_ids=None):
        ids = doc_ids if doc_ids is not None else list(self.sims)
        return {d: self.sims.get(d, 0.0) for d in ids}


# === 1. dispatch by domain ============================================================


@pytest.mark.parametrize("domain,expected", [
    ("code", True), ("CODE", True), (" code ", True),
    ("general", False), ("wiki", False), ("", False), (None, False),
])
def test_is_code_corpus(domain, expected):
    assert is_code_corpus(domain) is expected


def test_code_domain_bql_is_the_in_memory_executor(units):
    assert isinstance(build_bql_engine(units, domain="code"), StructuralExecutor)


def test_code_domain_bql_needs_no_key_or_index(units, tmp_path):
    eng = build_bql_engine(units, index_root=str(tmp_path / "nothing_here"), key=None, domain="code")
    assert isinstance(eng, StructuralExecutor)


def test_code_domain_refuses_dense_fusion(units):
    with pytest.raises(SetupError, match="document feature"):
        build_bql_engine(units, dense=_StubDense({}), domain="code")


def test_code_domain_refuses_dense_only_ranking(units):
    with pytest.raises(SetupError, match="document feature"):
        build_bql_engine_dense_only(units, dense=_StubDense({}), domain="code")


def test_code_domain_has_no_indri_engine(units):
    with pytest.raises(SetupError, match="Indri"):
        build_indri_engine(units, domain="code")


def test_general_domain_bql_is_the_lucene_adapter(units, index_root, key):
    eng = build_bql_engine(units, index_root=index_root, key=key, domain="general")
    assert isinstance(eng, LuceneBqlAdapter)
    assert not isinstance(eng, LuceneBqlDonlyAdapter)
    assert eng.dense is None


def test_general_domain_is_the_default(units, index_root, key):
    assert isinstance(build_bql_engine(units, index_root, key), LuceneBqlAdapter)
    assert isinstance(build_indri_engine(units, index_root, key), LuceneIndriAdapter)


def test_general_domain_dense_only_is_the_donly_adapter(units, index_root, key):
    stub = _StubDense({})
    eng = build_bql_engine_dense_only(units, index_root=index_root, key=key, dense=stub)
    assert isinstance(eng, LuceneBqlDonlyAdapter)
    assert eng.dense is stub


def test_general_domain_indri_is_the_lucene_adapter(units, index_root, key):
    eng = build_indri_engine(units, index_root=index_root, key=key)
    assert isinstance(eng, LuceneIndriAdapter)
    assert eng.dense is None


def test_build_bql_engine_wires_dense_through(units, index_root, key):
    stub = _StubDense({})
    eng = build_bql_engine(units, index_root=index_root, key=key, dense=stub)
    assert isinstance(eng, LuceneBqlAdapter)
    assert eng.dense is stub


def test_build_indri_engine_wires_dense_through(units, index_root, key):
    stub = _StubDense({})
    eng = build_indri_engine(units, index_root=index_root, key=key, dense=stub)
    assert isinstance(eng, LuceneIndriAdapter)
    assert eng.dense is stub


class _LazyUnits(list):
    """An on-disk corpus stand-in: `is_lazy` (agent_search/corpus/docstore.py) only reads the
    `lazy` attribute, so a list subclass with it set is enough to take that branch."""
    lazy = True


@pytest.mark.parametrize("build", [build_bql_engine, build_bql_engine_dense_only, build_indri_engine])
def test_in_memory_corpus_without_key_gets_an_ad_hoc_index(units, tmp_path, build):
    """No key: an in-memory corpus is indexed under a key derived from its fingerprint."""
    from agent_search.corpus.fingerprint import corpus_fingerprint
    from agent_search.retrievers.lucene import index_builder
    root = str(tmp_path / "adhoc_root")
    eng = build(units, index_root=root, key=None, dense=_StubDense({}) if build is build_bql_engine_dense_only else None)
    assert index_builder.is_built(root, "adhoc_" + corpus_fingerprint(units)[:16], expected_n_docs=len(units))
    assert eng is not None


def test_in_memory_corpus_missing_index_is_built_on_first_use(units, tmp_path):
    """An in-memory corpus with a key but no prebuilt index gets one built under `index_root`
    at engine construction, the same way the Lucene BM25 index is."""
    from agent_search.retrievers.lucene import index_builder
    root = str(tmp_path / "fresh_root")
    assert not index_builder.is_built(root, "fresh_key")
    eng = build_indri_engine(units, index_root=root, key="fresh_key")
    assert index_builder.is_built(root, "fresh_key", expected_n_docs=len(units))
    r = eng.search("#combine(dog train)", k=3)
    assert r.error is None and r.hits[0][0] == "both"
    bql = build_bql_engine(units, index_root=root, key="fresh_key")
    assert isinstance(bql, LuceneBqlAdapter)


@pytest.mark.parametrize("build", [build_bql_engine, build_bql_engine_dense_only, build_indri_engine])
def test_lazy_corpus_missing_index_is_a_setup_error(units, tmp_path, build):
    """An on-disk corpus is never loaded whole to build an index during a run: a missing
    prebuilt index stops the run at engine construction, before the first episode, instead
    of turning every search into an error observation."""
    lazy = _LazyUnits(units)
    with pytest.raises(SetupError, match="build-indexes") as ei:
        build(lazy, index_root=str(tmp_path / "no_such_index_root"), key="no_such_dataset_at_all",
              dense=_StubDense({}) if build is build_bql_engine_dense_only else None)
    assert "no_such_dataset_at_all" in str(ei.value)


def test_lazy_corpus_without_key_is_a_setup_error(units, tmp_path):
    lazy = _LazyUnits(units)
    with pytest.raises(SetupError, match="on-disk document store"):
        build_bql_engine(lazy, index_root=str(tmp_path / "no_such_index_root"), key=None)


# === 2. the document tools on the Lucene engines =============================================


def test_bql_visit_search_and_visit(units):
    tb = _bql_visit_toolbox(units, build_lucene_bql(units))
    out = tb.run("search_bv", {"query": "harbor[title]"})
    assert "ERROR" not in out
    assert "d_harbor" in out
    assert "§[" in out and "ib[" in out         # the structure table shape
    assert "matched:" in out                     # computed by the tool, not the engine
    assert "»" in out                             # content excerpt (forced, fairness parity)
    visit_out = tb.run("visit_bv", {"rank": 1})
    assert "ERROR" not in visit_out
    assert "d_harbor" in visit_out
    assert "Harbor Festival" in visit_out


def test_bql_visit_coverage_fallback_runs_on_lucene(units):
    """The 0-exact-hit constraint-coverage path (BQL v2 Feature 2) is served by
    `LuceneBqlAdapter.coverage_topk` (one `match_ids` call per AND child over the pool)."""
    tb = _bql_visit_toolbox(units, build_lucene_bql(units))
    out = tb.run("search_bv", {"query": "foo[body] AND bar[body] AND baz[body] AND qux[body]"})
    assert "CONSTRAINT COVERAGE" in out
    assert "miss=[" in out
    lines = out.split("\n")
    assert lines[1].split()[1] == "docA"          # matches foo/bar/baz, misses qux
    assert "cov=3/4" in lines[1]
    assert "qux" in lines[1]


def test_docsearchfetch_n_hits_header_is_the_exact_count(units):
    """`obs.n_hits` (the "(N matches, top K)" count) is the exact total from
    `LuceneStructuredEngine.count_bql_expr`, not a truncated `len(hits)`."""
    tb = _docsearchfetch_toolbox(units, build_lucene_bql(units))
    out = tb.run("search", {"query": "doc[title]", "k": 2})
    assert "ERROR" not in out
    # docA/docB/docC/docD all title-match "doc"; k=2 truncates the listing.
    assert "4 matches, top 2" in out


def test_indri_visit_search_and_visit(units):
    tb = _indri_visit_toolbox(units, build_lucene_indri(units))
    out = tb.run("isearch_v", {"query": "#combine(dog train)"})
    assert "ERROR" not in out
    assert "hits):" in out
    assert "»" in out
    assert out.splitlines()[1].split()[1] == "both"     # the doc with both terms ranks first
    visit_out = tb.run("visit_v", {"rank": 1})
    assert "ERROR" not in visit_out
    assert "dog train dog train station" in visit_out


def test_indri_fetch_section_fetch(units):
    tb = _indri_fetch_toolbox(units, build_lucene_indri(units))
    out = tb.run("isearch", {"query": "#combine(dog train)"})
    assert "ERROR" not in out
    fetch_out = tb.run("fetch", {"specs": [[1, ""]]})
    assert "ERROR" not in fetch_out


# === 3. dense-belief fusion on the Indri adapter ===========================================


class _RaisingDense:
    def score(self, query_text, doc_ids=None):
        raise RuntimeError("boom")


def test_dense_none_is_a_pure_lucene_search(units):
    r = build_lucene_indri(units).search("#combine(dog train)", k=5)
    assert r.error is None
    assert r.hits
    assert r.diagnostics == []


def test_dense_reranks_lucene_pool_at_high_weight(units, monkeypatch):
    r0 = build_lucene_indri(units).search("#combine(dog train)", k=5)
    doc_ids = [d for d, _ in r0.hits]
    assert len(doc_ids) >= 2
    boosted = doc_ids[-1]                    # Lucene's worst-ranked returned hit
    sims = {d: (10.0 if d == boosted else 0.0) for d in doc_ids}
    monkeypatch.setenv("INDRI_DENSE_W", "0.95")
    r1 = build_lucene_indri(units, dense=_StubDense(sims)).search("#combine(dog train)", k=5)
    assert [d for d, _ in r1.hits][0] == boosted
    assert any(name == "#dense" for name, _ in r1.diagnostics)


def test_dense_w_zero_preserves_lucene_order_and_scores(units, monkeypatch):
    r0 = build_lucene_indri(units).search("#combine(dog train)", k=5)
    sims = {d: float(i) for i, (d, _) in enumerate(r0.hits)}    # arbitrary: irrelevant at w=0
    monkeypatch.setenv("INDRI_DENSE_W", "0")
    r1 = build_lucene_indri(units, dense=_StubDense(sims)).search("#combine(dog train)", k=5)
    assert r1.hits == r0.hits
    # still computed and attached: the contribution is visible without affecting the ranking
    assert any(name == "#dense" for name, _ in r1.diagnostics)


def test_dense_source_raising_degrades_to_lexical_only(units, monkeypatch):
    r0 = build_lucene_indri(units).search("#combine(dog train)", k=5)
    monkeypatch.setenv("INDRI_DENSE_W", "0.5")
    r1 = build_lucene_indri(units, dense=_RaisingDense()).search("#combine(dog train)", k=5)
    assert r1.error is None
    assert r1.hits == r0.hits
    assert r1.diagnostics == []
