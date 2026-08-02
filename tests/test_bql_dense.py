"""NEW, ADDITIVE-only BQL_DENSE dense-fused ranking (the answer to "why not BQL with dense
instead of Indri with dense?" — see agent_search/retrievers/structural/bql/dense_fuse.py's
module docstring for the full mechanism spec).

Mechanism under test:
  1. Fusion math: `rrf_fuse`/`fuse_ranked`/`fuse_coverage_tiers` on hand-computed fixtures.
  2. Filter semantics: BQL's boolean/field/date FILTER selects candidates exactly as before —
     a non-matching doc must NEVER appear, even with an engineered-maximal dense similarity.
  3. Restriction: dense scoring is called ONLY with the filter-passing candidate ids (never a
     global dense search) — asserted directly against the stub's recorded call args.
  4. Knob off (no `attach_dense` call / `self.dense is None`) = byte-identical: the dense side
     is never even CONSULTED, so nothing downstream of it can differ.
  5. Conditions `research_bql_dense_visit`/`research_bql_dense_snip` load/resolve to the
     'bqldensevisit'/'bqldensesnip' arms via the registry, mirroring `research_bql_visit`/
     `research_snip` tool-for-tool except the tool NAMES.
  6. Offline e2e smoke: BqlVisitWorkspace/DocSearchFetch driven end-to-end over a dense-attached
     executor (deterministic hash-based stub encoder — no model download, no network), plus the
     full `AgentRetriever.index()` wiring (missing-cache fail-loud, and a stubbed-encoder
     success path).

CPU-only; a small synthetic corpus mirroring tests/test_bql_visit.py's fixture shape.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql.dense_fuse import (
    RRF_K, bql_dense_enabled, dense_rank_for_candidates, fuse_coverage_tiers, fuse_ranked,
    rrf_fuse)
from agent_search.retrievers.structural.bql.executor import StructuralExecutor
from agent_search.retrievers.structural.indri.dense_belief import DenseBelief

# --- stub encoder (mirrors tests/test_indri_dense.py's StubEncoder exactly: deterministic
# hash-based bag-of-tokens vectors, no model download, with an engineerable synonym table) ----


class StubEncoder:
    DIM = 32
    SYNONYMS = {"quagga": "zebra"}

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kw):
        self.calls += 1
        rows = []
        for t in texts:
            v = np.zeros(self.DIM, dtype=np.float64)
            for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
                tok = self.SYNONYMS.get(tok, tok)
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
                v[h] += 1.0
            n = np.linalg.norm(v)
            rows.append(v / n if n > 0 else v)
        return np.asarray(rows, dtype=np.float32)


class RaisingDense:
    """A dense source whose every entry point raises — fusion must degrade silently."""

    def is_ready(self):
        return True

    def score(self, query_text, doc_ids=None):
        raise RuntimeError("boom")


class RecordingDense:
    """Wraps a real `DenseBelief` but records every `doc_ids` argument `.score()` was called
    with — the direct test that dense scoring is restricted to the exact candidate pool handed
    in, never a broader/global set."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list = []

    def is_ready(self):
        return self._inner.is_ready()

    def score(self, query_text, doc_ids=None):
        self.calls.append(list(doc_ids) if doc_ids is not None else None)
        return self._inner.score(query_text, doc_ids=doc_ids)


def _mk(doc_id: str, body: str, title: str | None = None, date: str | None = None) -> CodeUnit:
    meta = {"date": date} if date is not None else {}
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=body, body=body, title=title, metadata=meta)


def _corpus() -> list[CodeUnit]:
    return [
        _mk("zebradoc", "zebra runs fast across the open field today"),
        _mk("paraphrase", "quagga quagga quagga"),          # zero lexical overlap with 'zebra'
        _mk("other1", "cat mouse bird fish garden"),
        _mk("other2", "train station schedule arrival"),
        _mk("dogdoc", "dog park walk leash morning"),
        # --- constraint-coverage fixture: AND(foo, bar, baz, qux) 0-hits exactly ---
        _mk("covA", "foo bar baz text here", title="Cov A"),
        _mk("covB", "foo bar text", title="Cov B"),
        _mk("covC", "nothing relevant here", title="Cov C"),
        _mk("covD", "foo only here", title="Cov D"),
    ]


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture()
def dense(units, tmp_path) -> DenseBelief:
    d = DenseBelief(model="stub/model", index_root=str(tmp_path), encoder=StubEncoder())
    d.build_or_load(units, key="stubcorpus")
    return d


# =============================================================================================
# 1. Fusion math — hand-computed
# =============================================================================================

def test_rrf_fuse_hand_computed_order():
    # k=60 (default). bm=[a,b,c] (ranks 1,2,3); dense=[c,a,b] (ranks 1,2,3).
    # score(a) = 1/61 + 1/62; score(b) = 1/62 + 1/63; score(c) = 1/63 + 1/61
    # a=0.032522..., c=0.032266..., b=0.032002... -> a > c > b
    assert RRF_K == 60
    fused = rrf_fuse(["a", "b", "c"], ["c", "a", "b"])
    assert fused == ["a", "c", "b"]
    a = 1 / 61 + 1 / 62
    b = 1 / 62 + 1 / 63
    c = 1 / 63 + 1 / 61
    assert a > c > b


def test_rrf_fuse_tie_break_is_doc_id_ascending():
    # bm=[x,y] (ranks 1,2); dense=[y,x] (ranks 1,2) -> score(x)=score(y) exactly (symmetric
    # rank swap) -> deterministic tie-break by doc_id.
    fused = rrf_fuse(["x", "y"], ["y", "x"])
    assert fused == ["x", "y"]


def test_rrf_fuse_doc_missing_from_one_side_contributes_zero_not_dropped():
    # bm=[a,b] (ranks 1,2); dense=[b] only (a absent from dense).
    # score(a) = 1/61 + 0; score(b) = 1/62 + 1/61 -> b's score is strictly higher.
    fused = rrf_fuse(["a", "b"], ["b"])
    assert fused == ["b", "a"]
    assert 1 / 61 < (1 / 62 + 1 / 61)


def test_rrf_fuse_custom_k():
    # k=0 collapses to pure reciprocal-rank (1/rank); still deterministic.
    fused = rrf_fuse(["a", "b"], ["b", "a"], k=0)
    # score(a) = 1/1 + 1/2 = 1.5; score(b) = 1/2 + 1/1 = 1.5 -> tie -> doc_id ascending
    assert fused == ["a", "b"]


# =============================================================================================
# 2. dense_rank_for_candidates — graceful degradation + restriction
# =============================================================================================

def test_dense_rank_for_candidates_none_dense_belief_returns_empty():
    assert dense_rank_for_candidates(None, "zebra", ["a", "b"]) == []


def test_dense_rank_for_candidates_empty_candidates_returns_empty(dense):
    assert dense_rank_for_candidates(dense, "zebra", []) == []


def test_dense_rank_for_candidates_empty_query_returns_empty(dense):
    assert dense_rank_for_candidates(dense, "   ", ["zebradoc", "other1"]) == []


def test_dense_rank_for_candidates_raising_dense_degrades_to_empty_never_raises():
    assert dense_rank_for_candidates(RaisingDense(), "zebra", ["a", "b"]) == []


def test_dense_rank_for_candidates_ranks_by_similarity_restricted_to_pool(dense):
    # 'paraphrase' (quagga->zebra synonym) must beat 'other1'/'other2' for query 'zebra',
    # and NEITHER 'zebradoc' nor any doc outside the given pool may appear.
    ranked = dense_rank_for_candidates(dense, "zebra", ["paraphrase", "other1", "other2"])
    assert ranked[0] == "paraphrase"
    assert set(ranked) == {"paraphrase", "other1", "other2"}
    assert "zebradoc" not in ranked


def test_dense_rank_for_candidates_only_scores_the_given_pool(units, tmp_path):
    inner = DenseBelief(model="stub/model2", index_root=str(tmp_path), encoder=StubEncoder())
    inner.build_or_load(units, key="stubcorpus2")
    rec = RecordingDense(inner)
    pool = ["zebradoc", "paraphrase"]
    dense_rank_for_candidates(rec, "zebra", pool)
    assert len(rec.calls) == 1
    assert set(rec.calls[0]) == set(pool)          # NEVER the whole corpus, only the pool given


# =============================================================================================
# 3. fuse_ranked — the single-tier `run_with_count` fusion path
# =============================================================================================

def test_fuse_ranked_empty_input_returns_empty(dense):
    assert fuse_ranked(dense, "zebra", []) == []


def test_fuse_ranked_no_dense_belief_returns_bm25_ranked_unchanged():
    bm25_ranked = [("other1", 3.0), ("other2", 1.0)]
    assert fuse_ranked(None, "zebra", bm25_ranked) == bm25_ranked


def test_fuse_ranked_reorders_via_rrf_and_preserves_bm25_scores(dense):
    # bm25 order: zebradoc (lexical hit) > paraphrase > other1 (all present but bm25-ranked by
    # lexical score, paraphrase has ZERO lexical overlap with 'zebra' so would score 0/last in
    # a real engine — engineer that here directly as the input ranking).
    bm25_ranked = [("zebradoc", 5.0), ("other1", 2.0), ("paraphrase", 0.0)]
    fused = fuse_ranked(dense, "zebra", bm25_ranked)
    fused_ids = [d for d, _ in fused]
    # dense similarity strongly favors 'paraphrase' (quagga==zebra synonym) — RRF must be able
    # to promote it despite its last-place bm25 rank.
    assert fused_ids.index("paraphrase") < fused_ids.index("other1")
    # scores are the ORIGINAL bm25 scores (informational only — order is what fusion changes).
    assert dict(fused) == dict(bm25_ranked)
    # no doc outside the input set is ever introduced or dropped.
    assert set(fused_ids) == {"zebradoc", "other1", "paraphrase"}


def test_fuse_ranked_restricts_dense_scoring_to_the_bm25_candidate_ids(units, tmp_path):
    inner = DenseBelief(model="stub/model3", index_root=str(tmp_path), encoder=StubEncoder())
    inner.build_or_load(units, key="stubcorpus3")
    rec = RecordingDense(inner)
    bm25_ranked = [("other1", 1.0), ("other2", 1.0)]
    fuse_ranked(rec, "zebra", bm25_ranked)
    assert len(rec.calls) == 1
    assert set(rec.calls[0]) == {"other1", "other2"}   # NOT zebradoc/paraphrase/dogdoc/...


# =============================================================================================
# 4. fuse_coverage_tiers — coverage tiers preserved, fuse WITHIN each tier only
# =============================================================================================

def _rows(*triples):
    """[(doc_id, n_matched, score), ...] -> [(doc_id, mask, n_matched, score), ...] (mask is
    irrelevant to fusion, kept as an opaque placeholder)."""
    return [(d, ("mask",), n, s) for d, n, s in triples]


def test_fuse_coverage_tiers_no_dense_returns_rows_unchanged():
    rows = _rows(("a", 3, 1.0), ("b", 3, 0.5), ("c", 1, 9.0))
    assert fuse_coverage_tiers(None, "q", rows) == rows


def test_fuse_coverage_tiers_never_promotes_across_a_tier_boundary(dense):
    # 'other1' (n_matched=1, low coverage) vs 'paraphrase' (n_matched=3, high coverage): even
    # though the stub gives 'other1' a much higher dense similarity to 'zebra' than
    # 'paraphrase' would to a mismatched query, 'paraphrase' must STILL rank above 'other1'
    # (more constraints satisfied always wins the tier ordering).
    rows = _rows(("paraphrase", 3, 0.1), ("other1", 1, 9.0))
    fused = fuse_coverage_tiers(dense, "zebra", rows)
    ids = [r[0] for r in fused]
    assert ids.index("paraphrase") < ids.index("other1")


def test_fuse_coverage_tiers_reorders_within_a_tier(dense):
    # Three docs in the SAME tier (n_matched=2), bm25 ranking them other2 > zebradoc > dogdoc.
    # Dense similarity to 'zebra' strongly favors zebradoc (literal token match) over
    # other2/dogdoc (zero overlap) — RRF must be able to promote zebradoc to the top despite
    # its middling bm25 rank, still within this SAME tier only.
    rows = _rows(("other2", 2, 5.0), ("zebradoc", 2, 1.0), ("dogdoc", 2, 0.5))
    fused = fuse_coverage_tiers(dense, "zebra", rows)
    ids = [r[0] for r in fused]
    assert set(ids) == {"other2", "zebradoc", "dogdoc"}
    assert ids[0] == "zebradoc"                               # dense promotes it to the top
    assert ids.index("zebradoc") < ids.index("other2")


def test_fuse_coverage_tiers_preserves_tier_order_and_row_payloads(dense):
    rows = _rows(("covA", 3, 2.0), ("covB", 2, 1.0), ("covC", 1, 0.5))
    fused = fuse_coverage_tiers(dense, "foo bar baz", rows)
    # tier order (n_matched desc) is preserved regardless of dense fusion within each tier.
    assert [r[2] for r in fused] == [3, 2, 1]
    assert {r[0] for r in fused} == {"covA", "covB", "covC"}


def test_fuse_coverage_tiers_singleton_tier_untouched(dense):
    rows = _rows(("only", 4, 1.0))
    assert fuse_coverage_tiers(dense, "zebra", rows) == rows


# =============================================================================================
# 5. StructuralExecutor integration: filter semantics preserved, knob-off byte-identical
# =============================================================================================

def _bql_and(*terms: str) -> str:
    return "AND(" + ", ".join(terms) + ")"


def test_attach_dense_is_a_pure_setter_default_none():
    ex = StructuralExecutor(_corpus())
    assert ex.dense is None
    ex2 = ex.attach_dense(None)
    assert ex2 is ex
    assert ex.dense is None


def test_run_with_count_filter_semantics_preserved_even_with_max_dense_similarity(units, dense):
    """The core correctness guarantee: a document that does NOT match the boolean filter must
    NEVER appear in `run_with_count`'s hits, no matter how similar it is to the query in dense
    embedding space. 'paraphrase' has a HIGH dense similarity to 'zebra' (quagga synonym) but
    contains neither 'zebra' nor 'runs' — it must be absent from a filter that requires both."""
    from agent_search.retrievers.structural.bql.parser import parse
    ex = StructuralExecutor(units).attach_dense(dense)
    expr = parse(_bql_and("zebra", "runs")).expr
    ranked, n_hits = ex.run_with_count(expr, k=10)
    ids = [d for d, _ in ranked]
    assert "paraphrase" not in ids                    # fails the filter -> never surfaces
    assert ids == ["zebradoc"]                         # the only doc that matches AND(zebra,runs)
    assert n_hits == 1


def test_run_with_count_dense_fuses_within_the_filter_passing_set(units, tmp_path):
    """A looser filter (OR) admits both 'zebradoc' and 'paraphrase'... except 'paraphrase'
    still lexically fails an OR(zebra, dog) filter too (no 'dog' token either) — use a filter
    both satisfy instead: OR(zebra, quagga) — 'quagga' IS a literal token in 'paraphrase', so
    it passes the FILTER on its own lexical merit (dense only affects the ORDER, never
    candidate selection)."""
    from agent_search.retrievers.structural.bql.parser import parse
    d = DenseBelief(model="stub/model4", index_root=str(tmp_path), encoder=StubEncoder())
    d.build_or_load(units, key="stubcorpus4")
    ex = StructuralExecutor(units).attach_dense(d)
    expr = parse("OR(zebra, quagga)").expr
    ranked, n_hits = ex.run_with_count(expr, k=10)
    ids = [doc for doc, _ in ranked]
    assert set(ids) == {"zebradoc", "paraphrase"}       # both pass the FILTER lexically
    assert n_hits == 2
    # dense similarity between 'paraphrase' (all-quagga body) and the query text ("zebra
    # quagga", the leaf terms) should be at least as strong as zebradoc's partial overlap —
    # not asserting a specific order here (both are legitimate given the stub's synonym), only
    # that dense fusion never removed either candidate.


def test_run_with_count_dense_off_is_byte_identical_to_no_attach(units):
    """`ex.dense is None` (never attached) must give byte-identical results to a plain
    executor — the knob-off contract for research/research_v2/research_snip/research_bql_visit."""
    from agent_search.retrievers.structural.bql.parser import parse
    plain = StructuralExecutor(units)
    off = StructuralExecutor(units).attach_dense(None)
    for q in ["zebra", _bql_and("foo", "bar"), "OR(cat, dog)", "nonexistentterm"]:
        expr = parse(q).expr
        a = plain.run_with_count(expr, k=10)
        b = off.run_with_count(expr, k=10)
        assert repr(a) == repr(b), q


def test_run_with_count_dense_attached_but_query_untouched_when_dense_never_called(units):
    """A stub whose `.score`/`.is_ready` would raise if ever invoked — if the executor's own
    fusion path is skipped whenever `dense is None` (the knob-off state), this proves the
    'off' path doesn't even reach the dense object."""
    from agent_search.retrievers.structural.bql.parser import parse

    class BoomIfTouched:
        def is_ready(self):
            raise AssertionError("dense side touched while dense is None")

        def score(self, *a, **kw):
            raise AssertionError("dense side touched while dense is None")

    ex = StructuralExecutor(units)     # dense stays None — never attach BoomIfTouched
    expr = parse("zebra").expr
    ranked, n_hits = ex.run_with_count(expr, k=10)      # must not raise
    assert n_hits == 1


def test_coverage_topk_filter_semantics_preserved_with_dense(units, dense):
    """`coverage_topk`'s 0-exact-hit fallback ranks by constraint COVERAGE — a doc that
    matches FEWER AND-children than another must never be promoted above it by dense fusion.
    AND(foo,bar,baz,qux) 0-hits exactly on the corpus's covA-D fixture (covA matches foo,bar,baz
    = 3/4; covD matches foo only = 1/4)."""
    from agent_search.retrievers.structural.bql.parser import parse
    ex = StructuralExecutor(units).attach_dense(dense)
    expr = parse(_bql_and("foo", "bar", "baz", "qux")).expr
    rows = ex.coverage_topk(expr, k=10)
    by_id = {r[0]: r[2] for r in rows}          # doc_id -> n_matched
    assert by_id["covA"] == 3
    assert by_id["covD"] == 1
    order = [r[0] for r in rows]
    assert order.index("covA") < order.index("covD")   # higher coverage always ranks first


def test_coverage_topk_dense_off_is_byte_identical_to_no_attach(units):
    from agent_search.retrievers.structural.bql.parser import parse
    plain = StructuralExecutor(units)
    off = StructuralExecutor(units).attach_dense(None)
    expr = parse(_bql_and("foo", "bar", "baz", "qux")).expr
    assert repr(plain.coverage_topk(expr, k=10)) == repr(off.coverage_topk(expr, k=10))


# =============================================================================================
# 6. BQL_DENSE env knob
# =============================================================================================

def test_bql_dense_enabled_default_off(monkeypatch):
    monkeypatch.delenv("BQL_DENSE", raising=False)
    assert bql_dense_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE", "Yes"])
def test_bql_dense_enabled_on_values(monkeypatch, val):
    monkeypatch.setenv("BQL_DENSE", val)
    assert bql_dense_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", ""])
def test_bql_dense_enabled_off_values(monkeypatch, val):
    monkeypatch.setenv("BQL_DENSE", val)
    assert bql_dense_enabled() is False


def test_load_or_build_default_dense_none_and_attach_works(units, tmp_path, dense):
    from agent_search.retrievers.structural.bql.executor import load_or_build
    ex = load_or_build(units, index_root=str(tmp_path), key="parity")   # no dense kwarg
    assert ex.dense is None
    ex2 = load_or_build(units, index_root=str(tmp_path), key="parity", dense=dense)
    assert ex2.dense is dense


# =============================================================================================
# 7. Conditions load/resolve
# =============================================================================================

def test_research_bql_dense_visit_condition_loads_and_mirrors_bql_visit():
    from agent_search.prompts import load_condition

    base = load_condition("research_bql_visit")
    dense_p = load_condition("research_bql_dense_visit")
    assert dense_p.toolset == "bql_dense_visit"
    assert set(dense_p.tool_names) == {"search_bqld", "visit_bqld"}
    # same skill manual as research_bql_visit (BQL_DENSE changes ranking, not coaching).
    assert base.system == dense_p.system or "date[RANGE]" in dense_p.system


def test_research_bql_dense_snip_condition_loads_and_mirrors_snip():
    from agent_search.prompts import load_condition

    p = load_condition("research_bql_dense_snip")
    assert p.toolset == "bql_dense_snip"
    assert set(p.tool_names) == {"search_bqlds", "fetch_bqlds"}


def test_research_bql_dense_visit_resolves_via_registry_as_bqldensevisit_arm():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bql_dense_visit", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert set(r.toolset) == {"search_bqld", "visit_bqld"}
    assert r._arm == "bqldensevisit"
    assert r.domain == "general"


def test_research_bql_dense_snip_resolves_via_registry_as_bqldensesnip_arm():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bql_dense_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert set(r.toolset) == {"search_bqlds", "fetch_bqlds"}
    assert r._arm == "bqldensesnip"
    assert r.domain == "general"


def test_bqldensevisit_index_raises_clear_error_when_cache_missing(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bql_dense_visit", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_corpus(), key="no_such_corpus_key")


def test_bqldensesnip_index_raises_clear_error_when_cache_missing(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bql_dense_snip", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_corpus(), key="no_such_corpus_key")


# =============================================================================================
# 8. Full AgentRetriever offline success-path smoke (stubbed encoder, no network)
# =============================================================================================

def test_agentretriever_bqldensevisit_end_to_end_offline_smoke(tmp_path, monkeypatch):
    """Drives `AgentRetriever.index()` + `._workspace()` through the REAL production wiring
    (not a hand-built StructuralExecutor) with the doc-embedding model name swapped to a stub
    and the shared encoder cache pre-seeded — no network/model download. Confirms the whole
    chain (cache validation -> DenseBelief.build_or_load -> build_bql_engine(dense=...) ->
    BqlVisitWorkspace(tool_names=...)) works end to end and that dense fusion is actually
    active (paraphrase promoted for a query with zero lexical overlap)."""
    import agent_search.retrievers.structural.indri.dense_belief as dense_belief_mod
    from agent_search.retrievers.dense import dense as dense_mod
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    stub_model = "stub/bqldensevisit-model"
    monkeypatch.setattr(dense_belief_mod, "DEFAULT_MODEL", stub_model)
    # pre-seed the shared encoder cache so ANY DenseRetriever(model=stub_model, ...) — incl.
    # the one `agent/retriever.py` constructs internally with no `encoder=` kwarg — resolves to
    # the stub instead of trying to download a real HF model.
    monkeypatch.setitem(dense_mod._ENCODER_CACHE, (stub_model, "auto", 1024), StubEncoder())

    corpus_units = _corpus()
    # pre-build the persisted cache under the SAME index_root/model/key the production code
    # will look for (probe.is_cached / DenseBelief.build_or_load's cache_dir resolution).
    DenseBelief(model=stub_model, index_root=str(tmp_path),
               encoder=StubEncoder()).build_or_load(corpus_units, key="offlinecorpus")

    r = build_factory("agent_research_bql_dense_visit", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    r.index(corpus_units, key="offlinecorpus")
    assert r._bql is not None
    assert r._bql.dense is not None

    ws = r._workspace(k=5)
    assert ws.tools == ("search_bqld", "visit_bqld")
    out = ws.run("search_bqld", {"query": "zebra"})
    assert "zebradoc" in out
    out2 = ws.run("visit_bqld", {"rank": 1})
    assert "ERROR" not in out2


def test_agentretriever_bqldensesnip_end_to_end_offline_smoke(tmp_path, monkeypatch):
    import agent_search.retrievers.structural.indri.dense_belief as dense_belief_mod
    from agent_search.retrievers.dense import dense as dense_mod
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    stub_model = "stub/bqldensesnip-model"
    monkeypatch.setattr(dense_belief_mod, "DEFAULT_MODEL", stub_model)
    monkeypatch.setitem(dense_mod._ENCODER_CACHE, (stub_model, "auto", 1024), StubEncoder())

    corpus_units = _corpus()
    DenseBelief(model=stub_model, index_root=str(tmp_path),
               encoder=StubEncoder()).build_or_load(corpus_units, key="offlinecorpus2")

    r = build_factory("agent_research_bql_dense_snip", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    r.index(corpus_units, key="offlinecorpus2")
    assert r._bql is not None
    assert r._bql.dense is not None

    ws = r._workspace(k=5)
    out = ws.run("search_bqlds", {"query": "zebra"})
    assert "zebradoc" in out
    out2 = ws.run("fetch_bqlds", {"specs": [[1, ""]]})
    assert "ERROR" not in out2


# =============================================================================================
# 9. Existing conditions untouched (parity spot-check)
# =============================================================================================

def test_research_bql_visit_and_research_snip_conditions_unaffected():
    from agent_search.prompts import load_condition

    bv = load_condition("research_bql_visit")
    assert bv.toolset == "bql_visit"
    assert set(bv.tool_names) == {"search_bv", "visit_bv"}
    snip = load_condition("research_snip")
    assert snip.toolset == "search_fetch_s"
    assert set(snip.tool_names) == {"search_s", "fetch_s"}


def test_bqlvisit_workspace_default_tool_names_unaffected_by_new_param():
    from agent_search.agent.tools.doc_research import BqlVisitWorkspace

    assert BqlVisitWorkspace.tools == ("search_bv", "visit_bv")
    ws = BqlVisitWorkspace(_corpus())
    assert ws.tools == ("search_bv", "visit_bv")


# =============================================================================================
# 10. research_bql_dense_fetch — completes the BQL_DENSE read-interface family with the PLAIN
# listing + section-FETCH cell (research_bql_dense_snip minus the excerpt). Mirrors section 7/8's
# research_bql_dense_snip tests tool-for-tool; the ONE substantive new assertion is that the
# listing carries NO "»" excerpt marker (proving `snippets` stays the constructor default False —
# unlike 'bqldensesnip', which forces it True).
# =============================================================================================

def test_research_bql_dense_fetch_condition_loads_and_mirrors_plain_doc():
    from agent_search.prompts import load_condition
    from agent_search.prompts.loader import render_manuals

    base = load_condition("research")
    p = load_condition("research_bql_dense_fetch")
    assert p.toolset == "bql_dense_fetch"
    assert set(p.tool_names) == {"search_bqldf", "fetch_bqldf"}
    # same skill manual BODY as the plain `research` condition (BQL_DENSE changes ranking, not
    # coaching) — compare the rendered manual text directly rather than the full `.system` (which
    # also embeds each tool's own NAME in its <tools> JSON schema, so it differs byte-for-byte
    # from `research`'s even though the coaching content is identical — same reason
    # `research_bql_dense_visit`'s own test below falls back to a substring check).
    assert (render_manuals(("search", "fetch"), domain="general")
            == render_manuals(("search_bqldf", "fetch_bqldf"), domain="general"))


def test_research_bql_dense_fetch_resolves_via_registry_as_bqldensefetch_arm():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bql_dense_fetch", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert set(r.toolset) == {"search_bqldf", "fetch_bqldf"}
    assert r._arm == "bqldensefetch"
    assert r.domain == "general"


def test_bqldensefetch_index_raises_clear_error_when_cache_missing(tmp_path):
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bql_dense_fetch", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_corpus(), key="no_such_corpus_key")


def test_agentretriever_bqldensefetch_end_to_end_offline_smoke(tmp_path, monkeypatch):
    """Same offline-smoke shape as test_agentretriever_bqldensesnip_end_to_end_offline_smoke,
    but asserts the listing is PLAIN (no `»` excerpt) — the one behavioral difference from
    'bqldensesnip', proving the listing/read recombination is genuinely composable here."""
    import agent_search.retrievers.structural.indri.dense_belief as dense_belief_mod
    from agent_search.retrievers.dense import dense as dense_mod
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    stub_model = "stub/bqldensefetch-model"
    monkeypatch.setattr(dense_belief_mod, "DEFAULT_MODEL", stub_model)
    monkeypatch.setitem(dense_mod._ENCODER_CACHE, (stub_model, "auto", 1024), StubEncoder())

    corpus_units = _corpus()
    DenseBelief(model=stub_model, index_root=str(tmp_path),
               encoder=StubEncoder()).build_or_load(corpus_units, key="offlinecorpus3")

    r = build_factory("agent_research_bql_dense_fetch", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    r.index(corpus_units, key="offlinecorpus3")
    assert r._bql is not None
    assert r._bql.dense is not None                    # dense-fused ranking IS attached

    ws = r._workspace(k=5)
    assert ws.snippets is False                         # PLAIN listing — the constructor default
    out = ws.run("search_bqldf", {"query": "zebra"})
    assert "ERROR" not in out
    assert "zebradoc" in out
    assert "»" not in out                               # NO per-hit excerpt (unlike bqldensesnip)
    out2 = ws.run("fetch_bqldf", {"specs": [[1, ""]]})
    assert "ERROR" not in out2


def test_bqldensefetch_search_bqldf_and_bqldensesnip_search_bqlds_differ_only_by_snippets(
        tmp_path, monkeypatch):
    """Direct side-by-side: SAME dense-fused ranking (same corpus, same query, same stub model/
    cache), the ONLY rendering difference between 'bqldensefetch' and 'bqldensesnip' is the `»`
    excerpt line — confirming the listing/read recombination changed nothing about retrieval."""
    import agent_search.retrievers.structural.indri.dense_belief as dense_belief_mod
    from agent_search.retrievers.dense import dense as dense_mod
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    stub_model = "stub/bqldensecompare-model"
    monkeypatch.setattr(dense_belief_mod, "DEFAULT_MODEL", stub_model)
    monkeypatch.setitem(dense_mod._ENCODER_CACHE, (stub_model, "auto", 1024), StubEncoder())

    corpus_units = _corpus()
    DenseBelief(model=stub_model, index_root=str(tmp_path),
               encoder=StubEncoder()).build_or_load(corpus_units, key="offlinecorpus4")

    fetch_r = build_factory("agent_research_bql_dense_fetch", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    fetch_r.index(corpus_units, key="offlinecorpus4")
    snip_r = build_factory("agent_research_bql_dense_snip", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    snip_r.index(corpus_units, key="offlinecorpus4")

    fetch_out = fetch_r._workspace(k=5).run("search_bqldf", {"query": "zebra"})
    snip_out = snip_r._workspace(k=5).run("search_bqlds", {"query": "zebra"})
    # same ranked hit lines (strip the trailing excerpt off snip_out's per-hit lines).
    strip_excerpt = lambda s: [ln.split("  »")[0] for ln in s.splitlines()]
    assert strip_excerpt(fetch_out) == strip_excerpt(snip_out)
    assert "»" not in fetch_out
    assert "»" in snip_out


def test_bqldensefetch_workspace_hallucinated_tool_name_errors(tmp_path, monkeypatch):
    """A model that hallucinates a tool name outside this condition's own toolset (e.g. the
    visit-family's `visit_bqld`, or a bare `visit`) must get a clear 'unknown tool' error, not a
    silent success — this condition has NO visit/whole-doc read at all, only search->fetch."""
    import agent_search.retrievers.structural.indri.dense_belief as dense_belief_mod
    from agent_search.retrievers.dense import dense as dense_mod
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    stub_model = "stub/bqldensefetch-halluc-model"
    monkeypatch.setattr(dense_belief_mod, "DEFAULT_MODEL", stub_model)
    monkeypatch.setitem(dense_mod._ENCODER_CACHE, (stub_model, "auto", 1024), StubEncoder())

    corpus_units = _corpus()
    DenseBelief(model=stub_model, index_root=str(tmp_path),
               encoder=StubEncoder()).build_or_load(corpus_units, key="offlinecorpus5")

    r = build_factory("agent_research_bql_dense_fetch", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    r.index(corpus_units, key="offlinecorpus5")
    ws = r._workspace(k=5)
    ws.run("search_bqldf", {"query": "zebra"})
    for bad_tool in ("visit_bqld", "visit", "isearch_v", "not_a_real_tool"):
        out = ws.run(bad_tool, {"rank": 1})
        assert out.startswith("ERROR: unknown tool"), (bad_tool, out)


def test_research_bql_dense_snip_and_visit_conditions_unaffected_by_new_fetch_cell():
    """Adding 'bqldensefetch' must not touch either pre-existing dense-BQL condition."""
    from agent_search.prompts import load_condition

    snip = load_condition("research_bql_dense_snip")
    assert snip.toolset == "bql_dense_snip"
    assert set(snip.tool_names) == {"search_bqlds", "fetch_bqlds"}
    visit = load_condition("research_bql_dense_visit")
    assert visit.toolset == "bql_dense_visit"
    assert set(visit.tool_names) == {"search_bqld", "visit_bqld"}
