"""NEW, ADDITIVE-only hybrid agent conditions: `research_indri_visit` (graded Indri SEARCH +
whole-doc VISIT read) and `research_indri_snip` (graded Indri SEARCH with content snippets +
structured SECTION fetch). Both disentangle "search interface" from "read granularity" — the
existing cells conflate them (bm25 = keyword search + whole-doc read; indri = graded search +
section read).

CPU-only. The existing `research_indri` condition/toolset/IndriFetchWorkspace default behavior
(snippets=False) must be byte-identical after this change — asserted explicitly below.
"""
from __future__ import annotations

import pytest

from agent_search.agent.tools.doc_indri import (
    IndriFetchWorkspace, IndriVisitWorkspace, _indri_query_terms)
from agent_search.corpus.units import CodeUnit
from agent_search.prompts import load_condition
from agent_search.retrievers.registry import RetrieverConfig, build_factory


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
    return units          # 3 + 26 = 29 docs, >= the ~25 asked for


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


# --- 1. IndriVisitWorkspace: isearch_v (ranked, content-bearing) + visit_v (whole-doc read) ---

def test_isearch_v_returns_ranked_hits_with_snippet(units):
    ws = IndriVisitWorkspace(units)
    out = ws.run("isearch_v", {"query": "#combine(bank management)"})
    assert "ERROR" not in out
    assert "hits):" in out
    assert "weakest constraint for top hit:" in out
    # Content-bearing listing (fairness parity with the bm25 baseline).
    assert "»" in out


def test_visit_v_by_rank_returns_full_body_and_marks_seen(units):
    ws = IndriVisitWorkspace(units)
    ws.run("isearch_v", {"query": "#combine(bank management)"})
    out = ws.run("visit_v", {"rank": 1})
    assert "ERROR" not in out
    u = ws.ubyid[ws.last_hits[0]]
    assert (u.body or "").strip() in out
    assert ws.last_hits[0] in ws.seen


def test_visit_v_by_doc_id_works(units):
    ws = IndriVisitWorkspace(units)
    ws.run("isearch_v", {"query": "#combine(bank management)"})
    out = ws.run("visit_v", {"rank": "d2002"})
    assert "ERROR" not in out
    assert "bank management ceremony held in 2002" in out
    assert "d2002" in ws.seen


def test_visit_v_bad_rank_errors_like_bm25visit(units):
    ws = IndriVisitWorkspace(units)
    ws.run("isearch_v", {"query": "#combine(bank management)"})
    out = ws.run("visit_v", {"rank": 999})
    assert out.startswith("ERROR:")
    assert "out of range" in out


def test_visit_v_no_prior_search_errors(units):
    ws = IndriVisitWorkspace(units)
    out = ws.run("visit_v", {"rank": 1})
    assert out.startswith("ERROR:")
    assert "no prior search" in out


def test_isearch_v_aliases_search_and_isearch_names(units):
    ws = IndriVisitWorkspace(units)
    out = ws.run("search", {"query": "#combine(bank management)"})
    assert "ERROR" not in out
    out2 = ws.run("isearch", {"query": "#combine(bank management)"})
    assert "ERROR" not in out2


def test_visit_v_aliases_visit_name(units):
    ws = IndriVisitWorkspace(units)
    ws.run("isearch_v", {"query": "#combine(bank management)"})
    out = ws.run("visit", {"rank": 1})
    assert "ERROR" not in out


# --- 2. conditions load + retriever factory builds both new arms ------------------------

def test_research_indri_snip_condition_loads():
    p = load_condition("research_indri_snip")
    assert p.toolset == "indri_snip"
    assert p.tool_names == ("isearch_s", "fetch")
    assert "#combine" in p.system


def test_research_indri_snip_resolves_via_registry():
    from agent_search.agent.retriever import AgentRetriever

    r = build_factory("agent_research_indri_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("isearch_s", "fetch")
    assert r.tool == "agent_research_indri_snip"
    assert r._arm == "indrisnip"
    assert r.domain == "general"
    assert not r.needs_files


def test_research_indri_snip_workspace_builds_and_answers_via_stub(units, tmp_path):
    from agent_search.agent.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key="test-indri-hybrids-corpus")
    ws = r._workspace(5, "bank management ceremony")
    assert isinstance(ws, IndriFetchWorkspace)
    assert not isinstance(ws, IndriVisitWorkspace)
    assert ws.snippets is True
    ranking = r.search("bank management ceremony", k=5)
    assert isinstance(ranking, list)


# --- 3. IndriFetchWorkspace(snippets=True): hits carry '»' + query-overlap; False unchanged ---

def test_snippets_true_shows_excerpt_overlapping_query_terms(units):
    ws = IndriFetchWorkspace(units, snippets=True)
    out = ws.run("isearch_s", {"query": "#combine(bank management)"})
    assert "»" in out
    hit_line = next(l for l in out.splitlines() if "d2002" in l or "d1999" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "bank" in excerpt or "management" in excerpt


def test_snippets_false_is_byte_identical_to_captured_render(units):
    # byte-parity guard: snippets=False (the plain research_indri default) must reproduce
    # the pre-existing (no-snippets) rendering exactly — no '»' anywhere, same header/body.
    ws_off = IndriFetchWorkspace(units, snippets=False)
    ws_default = IndriFetchWorkspace(units)          # old default (no snippets kwarg at all)
    out_off = ws_off.run("isearch", {"query": "#combine(bank management)"})
    out_default = ws_default.run("isearch", {"query": "#combine(bank management)"})
    assert out_off == out_default
    assert "»" not in out_off


def test_isearch_s_alias_works_and_fetch_is_inherited(units):
    ws = IndriFetchWorkspace(units, snippets=True)
    out = ws.run("isearch_s", {"query": "#combine(bank management)"})
    assert "»" in out
    fetched = ws.run("fetch", {"specs": [[1, "History"]]})
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


# --- 4b. unknown `.field` warning surfaces in the rendered isearch text (LOW finding) ----
# Regression for the adversarial-verification finding: an unrecognized field name (e.g.
# `.bogusfield`) is a VALID Indri QL query -- neither engine errors -- but python "searches
# anyway" (smoothing never hard-zeros) while Lucene silently returns 0 hits, and nothing in
# the old rendered text distinguished either case from a normal, deliberately-zero-hit
# query. `IndriFetchWorkspace._search_impl` now appends the engine's `.warning` on every
# return path.

def test_isearch_unknown_field_warning_appears_with_hits(units):
    ws = IndriFetchWorkspace(units, op_nudge=False)
    out = ws.run("isearch", {"query": "bank.bogusfield"})
    assert "ERROR" not in out
    assert "hits" in out                 # smoothing still returns real hits (python engine)
    assert "warning" in out.lower() and "bogusfield" in out


def test_isearch_unknown_field_warning_survives_a_nonsense_query(units):
    # The python engine's Dirichlet smoothing never hard-zeros (see indri/model.py's
    # Deviations: "graceful ... never a hard zero"), so even a nonsense query term still
    # returns k hits from the whole pool -- the TRUE 0-hit case only manifests under the
    # `lucene` backend (see tests/test_lucene_structured.py's analogous test, which needs
    # a real JVM/index). Here we just confirm the warning survives regardless of hit count.
    ws = IndriFetchWorkspace(units, op_nudge=False)
    out = ws.run("isearch", {"query": "zzz_definitely_not_a_term_zzz.bogusfield"})
    assert "warning" in out.lower() and "bogusfield" in out


def test_isearch_known_field_has_no_warning(units):
    ws = IndriFetchWorkspace(units, op_nudge=False)
    out = ws.run("isearch", {"query": "bank.title"})
    assert "warning" not in out.lower()


# --- 5. INDRI_DENSE env-gated dense-belief attachment (agent_search/agent/retriever.py) --
# NEW, ADDITIVE-only: the indri-family arms ('indri'/'indrivisit'/'indrisnip') optionally
# attach a `DenseBelief` to the shared `IndriExecutor` when `INDRI_DENSE` is truthy. OFF by
# default — all three tests below patch
# `agent_search.retrievers.structural.indri.dense_belief.DenseBelief` (the class the
# `index()` branch imports LOCALLY at call time, so patching the module attribute is
# sufficient) rather than touching real GPU/model code.

class _RaisingDenseBelief:
    """Stand-in for DenseBelief that must NEVER be constructed when INDRI_DENSE is unset."""

    def __init__(self, *a, **kw):
        raise AssertionError("DenseBelief() must not be constructed when INDRI_DENSE is off")


class _StubDenseBelief:
    """Stand-in for DenseBelief recording construction/build calls, for the INDRI_DENSE=1
    happy path."""

    instances: list = []

    def __init__(self, model=None):
        self.model = model
        self.built_key = None
        _StubDenseBelief.instances.append(self)

    def build_or_load(self, units, key=None):
        self.built_units = list(units)
        self.built_key = key
        return self


class _RaisingBuildDenseBelief:
    """Stand-in whose `build_or_load` raises (e.g. missing GPU-built embedding cache) —
    must degrade the run to lexical-only, never break it."""

    def __init__(self, model=None):
        self.model = model

    def build_or_load(self, units, key=None):
        raise RuntimeError("simulated missing dense doc-embedding cache")


def test_indri_dense_off_by_default_does_not_construct_dense_belief(units, tmp_path, monkeypatch):
    monkeypatch.delenv("INDRI_DENSE", raising=False)
    monkeypatch.setattr(
        "agent_search.retrievers.structural.indri.dense_belief.DenseBelief",
        _RaisingDenseBelief)
    from agent_search.agent.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key="test-indri-hybrids-dense-off")     # would raise if DenseBelief() called
    assert r._indri.dense is None


def test_indri_dense_on_attaches_stub_dense_belief_to_executor(units, tmp_path, monkeypatch):
    _StubDenseBelief.instances = []
    monkeypatch.setenv("INDRI_DENSE", "1")
    monkeypatch.setattr(
        "agent_search.retrievers.structural.indri.dense_belief.DenseBelief",
        _StubDenseBelief)
    from agent_search.agent.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key="test-indri-hybrids-dense-on")

    assert len(_StubDenseBelief.instances) == 1
    stub = _StubDenseBelief.instances[0]
    assert stub.model == "BAAI/bge-base-en-v1.5"
    assert stub.built_key == "test-indri-hybrids-dense-on"
    # attached on the executor the workspace receives ('the attribute the engine stores it
    # on' is IndriExecutor.dense, per model.py's attach_dense/`dense=` constructor kwarg)
    assert r._indri.dense is stub
    ws = r._workspace(5, "bank management ceremony")
    assert ws.iex.dense is stub


def test_indri_dense_on_raising_dense_belief_degrades_with_warning(units, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INDRI_DENSE", "1")
    monkeypatch.setattr(
        "agent_search.retrievers.structural.indri.dense_belief.DenseBelief",
        _RaisingBuildDenseBelief)
    from agent_search.agent.retriever import AgentRetriever

    cfg = RetrieverConfig(policy="stub", index_root=str(tmp_path))
    r = build_factory("agent_research_indri_snip", cfg)()
    r.index(units, key="test-indri-hybrids-dense-raise")   # must NOT raise -> degrades

    assert r._indri.dense is None
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "INDRI_DENSE" in captured.err
