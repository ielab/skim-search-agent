"""`indri_visit` (isearch_v/visit_v: graded Indri search plus whole-doc visit read) and
`research_indri_snip` (isearch_s/fetch: graded Indri search with content snippets plus
structured section fetch). Both disentangle "search interface" from "read granularity", where
the plain cells conflate them (bm25 = keyword search + whole-doc read; indri = graded search +
section read).

CPU-only; the Indri engine is `LuceneIndriAdapter` over a prebuilt Lucene index
(`tests/lucene_support.py`), so the module skips without a JVM. The `research_indri_snip`
condition's `SearchIndri` default behavior (snippet=NoSnippet()) must be byte-identical to plain
search, asserted explicitly below.
"""
from __future__ import annotations

import pytest

from agent_search.snippets import NoSnippet, TermWindow
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.registry import RetrieverConfig, build_factory
from agent_search.strategies import CONDITIONS
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_indri.tool import SearchIndri, _indri_query_terms
from agent_search.tools.visit.tool import Visit
from tests import lucene_support
from tests.lucene_support import build_lucene_indri, require_jvm

require_jvm()


def _mk(doc_id: str, body: str, title: str | None = None) -> CodeUnit:
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=body, body=body, title=title)


def _corpus() -> list[CodeUnit]:
    units = [
        _mk("d2002", "## History\nbank management ceremony held in 2002", title="Bank 2002"),
        _mk("d1999", "## History\nbank management ceremony held in 1999", title="Bank 1999"),
        _mk("dogtrain", "## History\ndog train dog train station", title="Trains and Dogs"),
    ]
    filler_vocab = ["widget", "gadget", "sprocket", "lever", "cog", "gear", "pulley",
                    "spring", "bolt", "nut", "washer", "hinge", "clamp", "bracket", "rivet",
                    "screw", "plank", "beam", "girder", "rafter", "joist", "shingle",
                    "gutter", "flue", "chimney", "hearth"]
    for i, w in enumerate(filler_vocab):
        units.append(_mk(f"filler{i}", f"## History\n{w} placeholder text", title=f"Filler {i}"))
    return units          # 3 + 26 = 29 docs


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture(scope="module")
def corpus_key(units) -> str:
    """The prebuilt Lucene structured index every strategy-level test below opens."""
    return lucene_support.build_structured_index(units)


def _visit_toolbox(units):
    """isearch_v (snippets forced on) + visit_v: the `indri_visit` strategy's tools."""
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    engines = {"indri": build_lucene_indri(units)}
    sv = SearchIndri(name="isearch_v", snippet=TermWindow()).bind(state, units, ubyid, engines)
    vv = Visit(name="visit_v").bind(state, units, ubyid, {})
    return ToolBox([sv, vv], state), ubyid


def _fetch_toolbox(units, snippet=NoSnippet(), op_nudge: bool = True):
    """A search_indri + fetch pair, bound as isearch_s when snippets, else isearch (the
    `indri`/`indri_plain` strategies' tools). `SearchIndri.aliases` covers both names either
    way, so `.run("isearch", ...)` / `.run("isearch_s", ...)` both resolve regardless."""
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    engines = {"indri": build_lucene_indri(units)}
    name = "isearch_s" if snippet.shows_excerpt else "isearch"
    sc = SearchIndri(name=name, snippet=snippet, op_nudge=op_nudge).bind(state, units, ubyid, engines)
    fc = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([sc, fc], state)


# --- 1. indri_visit: isearch_v (ranked, content-bearing) + visit_v (whole-doc read) ---

def test_isearch_v_returns_ranked_hits_with_snippet(units):
    box, _ = _visit_toolbox(units)
    out = box.run("isearch_v", {"query": "#combine(bank management)"})
    assert "ERROR" not in out
    assert "hits):" in out
    # Content-bearing listing (fairness parity with the bm25 baseline).
    assert "»" in out


def test_visit_v_by_rank_returns_full_body_and_marks_seen(units):
    box, ubyid = _visit_toolbox(units)
    box.run("isearch_v", {"query": "#combine(bank management)"})
    out = box.run("visit_v", {"rank": 1})
    assert "ERROR" not in out
    u = ubyid[box.last_hits[0]]
    assert (u.body or "").strip() in out
    assert box.last_hits[0] in box.seen


def test_visit_v_by_doc_id_works(units):
    box, _ = _visit_toolbox(units)
    box.run("isearch_v", {"query": "#combine(bank management)"})
    out = box.run("visit_v", {"rank": "d2002"})
    assert "ERROR" not in out
    assert "bank management ceremony held in 2002" in out
    assert "d2002" in box.seen


def test_visit_v_bad_rank_errors_like_bm25visit(units):
    box, _ = _visit_toolbox(units)
    box.run("isearch_v", {"query": "#combine(bank management)"})
    out = box.run("visit_v", {"rank": 999})
    assert out.startswith("ERROR:")
    assert "out of range" in out


def test_visit_v_no_prior_search_errors(units):
    box, _ = _visit_toolbox(units)
    out = box.run("visit_v", {"rank": 1})
    assert out.startswith("ERROR:")
    assert "no prior search" in out


def test_isearch_v_aliases_search_and_isearch_names(units):
    box, _ = _visit_toolbox(units)
    out = box.run("search", {"query": "#combine(bank management)"})
    assert "ERROR" not in out
    out2 = box.run("isearch", {"query": "#combine(bank management)"})
    assert "ERROR" not in out2


def test_visit_v_aliases_visit_name(units):
    box, _ = _visit_toolbox(units)
    box.run("isearch_v", {"query": "#combine(bank management)"})
    out = box.run("visit", {"rank": 1})
    assert "ERROR" not in out


# --- 2. conditions load + retriever factory builds both new arms ------------------------

def test_research_indri_snip_condition_loads():
    cond = CONDITIONS["research_indri_snip"]
    assert cond.strategy.toolset_name == "indri_snip"
    assert cond.tool_names == ("isearch_s", "fetch")
    assert "#combine" in cond.render()


def test_research_indri_snip_resolves_via_registry():
    from agent_search.evaluation.agent_runner import ConditionAgent

    r = build_factory("agent_research_indri_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("isearch_s", "fetch")
    assert r.tool == "agent_research_indri_snip"
    assert r.condition.name == "research_indri_snip"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_indri_snip_workspace_builds_and_answers_via_stub(units, corpus_key):
    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key=corpus_key)
    ws = r.toolbox("bank management ceremony")
    assert isinstance(ws["isearch_s"], SearchIndri)
    assert ws["isearch_s"].snippet.shows_excerpt
    ranking = r.search("bank management ceremony", k=5)
    assert isinstance(ranking, list)


# --- 3. SearchIndri(snippet=TermWindow()): hits carry '»' + query-overlap; False unchanged ---

def test_snippets_true_shows_excerpt_overlapping_query_terms(units):
    box = _fetch_toolbox(units, snippet=TermWindow())
    out = box.run("isearch_s", {"query": "#combine(bank management)"})
    assert "»" in out
    hit_line = next(l for l in out.splitlines() if "d2002" in l or "d1999" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "bank" in excerpt or "management" in excerpt


def test_snippets_false_is_byte_identical_to_captured_render(units):
    # byte-parity guard: snippet=NoSnippet() (the plain research_indri default) must reproduce
    # the no-snippets rendering exactly: no '»' anywhere, same header/body.
    box_off = _fetch_toolbox(units, snippet=NoSnippet())
    box_default = _fetch_toolbox(units)          # snippet=NoSnippet() is SearchIndri's own default
    out_off = box_off.run("isearch", {"query": "#combine(bank management)"})
    out_default = box_default.run("isearch", {"query": "#combine(bank management)"})
    assert out_off == out_default
    assert "»" not in out_off


def test_isearch_s_alias_works_and_fetch_is_inherited(units):
    box = _fetch_toolbox(units, snippet=TermWindow())
    out = box.run("isearch_s", {"query": "#combine(bank management)"})
    assert "»" in out
    fetched = box.run("fetch", {"specs": [[1, "History"]]})
    assert "ERROR" not in fetched


# --- 4. _indri_query_terms: strips operator names + field suffixes ----------------------

def test_indri_query_terms_strips_operators_and_field_suffixes():
    terms = _indri_query_terms("#combine( #1(bank management) treaty.title )")
    assert "bank" in terms
    assert "management" in terms
    assert "treaty" in terms
    assert "#combine" not in terms
    assert not any("combine" in t for t in terms)
    assert "title" not in terms


def test_indri_query_terms_empty_query_returns_empty_list():
    assert _indri_query_terms("") == []
    assert _indri_query_terms(None) == []


def test_indri_query_terms_plain_keywords_pass_through():
    assert _indri_query_terms("alpha bravo") == ["alpha", "bravo"]


# --- 4b. unknown `.field` warning surfaces in the rendered isearch text -------------------
# An unrecognized field name (e.g. `.bogusfield`) is a valid Indri QL query: the engine does
# not error, it restricts to a field with no postings and returns zero hits. Nothing in the
# hit count alone distinguishes that from a deliberately zero-hit query, so
# `SearchIndri._search_impl` appends the engine's `.warning` on every return path.

def test_isearch_unknown_field_warning_appears_on_zero_hits(units):
    box = _fetch_toolbox(units, op_nudge=False)
    out = box.run("isearch", {"query": "bank.bogusfield"})
    assert "ERROR" not in out
    assert "(0 hits)" in out
    assert "warning" in out.lower() and "bogusfield" in out


def test_isearch_unknown_field_warning_survives_a_nonsense_query(units):
    box = _fetch_toolbox(units, op_nudge=False)
    out = box.run("isearch", {"query": "zzz_definitely_not_a_term_zzz.bogusfield"})
    assert "warning" in out.lower() and "bogusfield" in out


def test_isearch_known_field_has_no_warning(units):
    box = _fetch_toolbox(units, op_nudge=False)
    out = box.run("isearch", {"query": "bank.title"})
    assert "warning" not in out.lower()


# --- 5. INDRI_DENSE env-gated dense-belief attachment (agent_search/retrievers/engines.py) --
# The indri-family arms ('indri'/'indri_visit'/'indri_plain') optionally attach a
# `DenseBelief` to the shared Indri engine when `INDRI_DENSE` is truthy. Off by default. The
# tests below patch `agent_search.retrievers.dense.DenseBelief` (the class `Engines` imports
# locally at call time, so patching the module attribute is sufficient) rather than touching
# real GPU/model code.

class _RaisingDenseBelief:
    """Stand-in for DenseBelief that must never be constructed when INDRI_DENSE is off."""

    def __init__(self, *a, **kw):
        raise AssertionError("DenseBelief() must not be constructed when INDRI_DENSE is off")


class _StubDenseBelief:
    """Stand-in for DenseBelief recording construction/build calls, for the INDRI_DENSE=1
    happy path."""

    instances: list = []

    def __init__(self, model=None, index_root="indexes", **_kw):
        self.model = model
        self.index_root = index_root
        self.built_key = None
        _StubDenseBelief.instances.append(self)

    def build_or_load(self, units, key=None):
        self.built_units = list(units)
        self.built_key = key
        return self


class _RaisingBuildDenseBelief:
    """Stand-in whose `build_or_load` raises (e.g. missing GPU-built embedding cache):
    the run must degrade to lexical-only, never break."""

    def __init__(self, model=None, **_kw):
        self.model = model

    def build_or_load(self, units, key=None):
        raise RuntimeError("simulated missing dense doc-embedding cache")


def test_indri_dense_off_by_default_does_not_construct_dense_belief(units, corpus_key, monkeypatch):
    monkeypatch.delenv("INDRI_DENSE", raising=False)
    monkeypatch.setattr(
        "agent_search.retrievers.dense.DenseBelief",
        _RaisingDenseBelief)

    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key=corpus_key)     # would raise if DenseBelief() were called
    assert r.engines.get("indri").dense is None


def test_indri_dense_on_attaches_stub_dense_belief_to_engine(units, corpus_key, monkeypatch):
    """INDRI_DENSE=1 attaches the run's dense belief to the Indri engine when the persisted
    embedding cache exists. The engine registry probes `DenseRetriever.is_cached` first and
    degrades with a warning when it is missing, so the stub here also stands in for the cache
    check."""
    from agent_search.retrievers.dense import DenseRetriever
    monkeypatch.setattr(DenseRetriever, "is_cached", lambda self, key=None: True)
    monkeypatch.delenv("DENSE_MODEL", raising=False)
    _StubDenseBelief.instances = []
    monkeypatch.setenv("INDRI_DENSE", "1")
    monkeypatch.setattr(
        "agent_search.retrievers.dense.DenseBelief",
        _StubDenseBelief)

    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key=corpus_key)

    assert len(_StubDenseBelief.instances) == 1
    stub = _StubDenseBelief.instances[0]
    assert stub.model == "BAAI/bge-base-en-v1.5"
    assert stub.built_key == corpus_key
    # attached on the engine the toolbox receives (`LuceneIndriAdapter.dense`)
    assert r.engines.get("indri").dense is stub
    ws = r.toolbox("bank management ceremony")
    assert ws["isearch_s"].iex.dense is stub


def test_indri_dense_on_raising_dense_belief_degrades_with_warning(units, corpus_key, monkeypatch, capsys):
    from agent_search.retrievers.dense import DenseRetriever
    monkeypatch.setattr(DenseRetriever, "is_cached", lambda self, key=None: True)
    monkeypatch.setenv("INDRI_DENSE", "1")
    monkeypatch.setattr(
        "agent_search.retrievers.dense.DenseBelief",
        _RaisingBuildDenseBelief)

    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key=corpus_key)     # must not raise: degrades

    assert r.engines.get("indri").dense is None
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "INDRI_DENSE" in captured.err
