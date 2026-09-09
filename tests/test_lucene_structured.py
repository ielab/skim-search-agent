"""Semantics validation for the `lucene` structured backend
(`agent_search/retrievers/structural/lucene/`) against BOTH pure-Python reference
engines (`indri`, `bql`) it compiles for -- see that package's `__init__.py`.

Per the task's validation contract (documented, not exact-ranking equality):
  1. Pure-boolean/filter queries (analyzer-light: `#band`, `#filreq`/`#filrej`,
     date ranges, `AND`/`OR`/`NOT`) must return the SAME MATCH SET as the Python
     reference.
  2. Graded queries (`#combine`/`#weight`/BM25 ranking) must be DIRECTIONALLY
     consistent -- a rank correlation threshold, not exact score/order equality
     (the Lucene compiler documents real approximations: SUM-of-scores vs
     weighted-MEAN-of-log-beliefs; see `indri_compiler.py`'s module docstring).
  3. Span/date/field ops are checked on hand-built docs with KNOWN answers.

Requires a real JVM (pyserini/pyjnius) -- module-scoped fixtures build one small
Lucene index and reuse it for every test in this file (JVM boot + a ~30-doc index
build is the expensive part, not the per-test queries).
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.structural.bql.executor import StructuralExecutor
from agent_search.retrievers.structural.bql.parser import parse as bql_parse
from agent_search.retrievers.structural.bql.types import check as bql_check
from agent_search.retrievers.structural.indri.model import IndriExecutor
from agent_search.retrievers.structural.lucene import index_builder
from agent_search.retrievers.structural.lucene import jni_utils as _jni
from agent_search.retrievers.structural.lucene.adapters import LuceneIndriAdapter
from agent_search.retrievers.structural.lucene.engine import LuceneStructuredEngine

# NOT `pytest.importorskip("jnius", ...)`: a bare `import jnius` starts the JVM with
# NO classpath (pyjnius auto-starts on module import in this environment) --
# `pyserini.pyclass`'s classpath configuration only takes effect if IT is the first
# thing to touch jnius. `_boot()` goes through that same pyserini-classpath-aware
# path, so it must run before anything else in the process bare-imports `jnius`
# (see `jni_utils.py`'s module docstring for this "one JVM per process" landmine).
try:
    _jni._boot()
except Exception as e:                          # pragma: no cover - environment-dependent
    pytest.skip(f"lucene backend needs a working JVM (pyserini/pyjnius): {e}",
                allow_module_level=True)


# --- shared fixture corpus: built via `units_from_documents`, the SAME path a real
# dataset takes (agent_search/evaluation/datasets.py), so `code` = title+body join exactly like
# production -- unlike a hand-rolled CodeUnit fixture, this makes the BQL default/
# DOC-scope comparison meaningful (see module docstring point 1). --------------

def _docs() -> list[dict]:
    return [
        # --- combine/weight ranking ---
        {"_id": "both", "title": "Trains and Dogs", "text": "dog train dog train station"},
        {"_id": "onlydog", "title": "Dog Park", "text": "dog dog dog dog park walk"},
        {"_id": "onlytrain", "title": "Train Schedule", "text": "train station schedule arrival"},
        {"_id": "neither", "title": "Animals", "text": "cat mouse bird fish garden"},
        # --- phrase / window ---
        {"_id": "phrase", "title": "Phrase doc", "text": "the white house announced today"},
        {"_id": "scrambled", "title": "Scrambled doc", "text": "house white today announced the"},
        {"_id": "gap1", "title": "Gap1 doc", "text": "white small house announced"},
        # --- syn ---
        {"_id": "hascar", "title": "Car doc", "text": "i bought a car yesterday"},
        {"_id": "hasauto", "title": "Auto doc", "text": "i bought an automobile yesterday"},
        {"_id": "hasneither", "title": "Bike doc", "text": "i bought a bicycle yesterday"},
        # --- field restriction ---
        {"_id": "titlehit", "title": "dog show winners", "text": "irrelevant filler body text here"},
        {"_id": "bodyhit", "title": "irrelevant filler", "text": "dog show winners announced today"},
        # --- section restriction ---
        {"_id": "sectionhit", "title": "no title hit", "text": "irrelevant body words only",
         "sections": [{"heading": "kangaroo overview", "text": "irrelevant body words only"}]},
        # --- filreq/filrej ---
        {"_id": "sheepdolly", "title": "Dolly", "text": "scientists cloned a sheep named dolly"},
        {"_id": "sheepwool", "title": "Wool", "text": "the sheep produced fine wool"},
        {"_id": "dollynosheep", "title": "Music", "text": "dolly parton released a new album"},
        # --- date ---
        {"_id": "d1979", "title": "d1979", "text": "alpha before", "date": "1979-12-31"},
        {"_id": "d1980", "title": "d1980", "text": "alpha instart", "date": "1980-01-01"},
        {"_id": "d1985", "title": "d1985", "text": "alpha midyear", "date": "1985-06-15"},
        {"_id": "d1989", "title": "d1989", "text": "alpha inend", "date": "1989-12-31"},
        {"_id": "d1990", "title": "d1990", "text": "alpha after", "date": "1990-01-01"},
        {"_id": "dmissing", "title": "dmissing", "text": "alpha nodate"},
        # --- band ---
        {"_id": "bandboth", "title": "RGB", "text": "red blue green"},
        {"_id": "bandonered", "title": "Red", "text": "red only here"},
        {"_id": "bandoneblue", "title": "Blue", "text": "blue only here"},
        # --- wildcard ---
        {"_id": "wild1", "title": "Wild1", "text": "wildlife wilderness wilting"},
        {"_id": "wild2", "title": "Wild2", "text": "gentle calm quiet"},
        # --- author ---
        {"_id": "authored", "title": "Authored doc", "text": "some unrelated content here",
         "author": "Jane Smith"},
        {"_id": "authored2", "title": "Authored doc two", "text": "other unrelated content",
         "author": "John Doe"},
        # --- no-zero-hit-pathology ---
        {"_id": "nz1", "title": "nz1", "text": "kappa lambda filler filler filler"},
        {"_id": "nz2", "title": "nz2", "text": "mu nu filler filler filler"},
        {"_id": "nz3", "title": "nz3", "text": "xi filler filler filler filler"},
    ]


@pytest.fixture(scope="module")
def units():
    return units_from_documents(_docs())


@pytest.fixture(scope="module")
def indri_ex(units) -> IndriExecutor:
    return IndriExecutor(units, mu=2500)


@pytest.fixture(scope="module")
def bql_ex(units) -> StructuralExecutor:
    return StructuralExecutor(units)


@pytest.fixture(scope="module")
def lucene_eng(tmp_path_factory, units) -> LuceneStructuredEngine:
    idx_root = str(tmp_path_factory.mktemp("lucene_structured_validation"))
    index_builder.build(units, idx_root, "fixture", rebuild=True, progress=False)
    eng = LuceneStructuredEngine(index_root=idx_root, dataset="fixture", mu=2500)
    yield eng
    eng.close()
    shutil.rmtree(idx_root, ignore_errors=True)


N = 40   # > corpus size, so search(k=N) always returns the FULL match set


def _indri_set(indri_ex, q: str) -> set:
    r = indri_ex.search(q, k=N)
    assert r.error is None, f"indri reference error for {q!r}: {r.error}"
    return {d for d, _ in r.hits}


def _lucene_indri_set(lucene_eng, q: str) -> set:
    r = lucene_eng.search_indri(q, k=N)
    assert r.error is None, f"lucene indri compile/exec error for {q!r}: {r.error}"
    return {h.doc_id for h in r.hits}


def _bql_set(bql_ex, q: str) -> set:
    r = bql_parse(q)
    assert r.ok, f"bql reference parse error for {q!r}: {r.error}"
    t = bql_check(r.expr)
    assert t.ok, f"bql reference type error for {q!r}: {t.error}"
    ranked, _ = bql_ex.run_with_count(r.expr, k=N)
    return {d for d, _ in ranked}


def _lucene_bql_set(lucene_eng, q: str) -> set:
    r = lucene_eng.search_bql(q, k=N)
    assert r.error is None, f"lucene bql compile/exec error for {q!r}: {r.error}"
    return {h.doc_id for h in r.hits}


def _spearman(a: list, b: list) -> float:
    """Spearman rank correlation over the doc_ids common to both lists `a`/`b`
    (best-first doc_id lists). Returns 1.0 if fewer than 2 docs overlap (nothing to
    disagree on)."""
    common = [d for d in a if d in set(b)]
    if len(common) < 2:
        return 1.0
    ra = {d: i for i, d in enumerate(a)}
    rb = {d: i for i, d in enumerate(b)}
    xs = [ra[d] for d in common]
    ys = [rb[d] for d in common]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 1.0
    return cov / (var_x * var_y) ** 0.5


# --- (1) match-set parity: pure-boolean / filter Indri queries ------------------

INDRI_BOOLEAN_QUERIES = [
    "#band(dog train)",
    "#band(red blue)",
    "#filreq(sheep dolly)",
    "#filrej(sheep dolly)",
    "#date:before(1985)",
    "#date:after(1985)",
    "#date:between(1980 1989)",
    # NOTE: bare #odN/#uwN/.field/#syn (NOT wrapped in #band/#filreq) are GRADED
    # queries in Indri, not boolean filters -- the Python reference's candidate
    # POOL includes every doc containing the window's/field's INDIVIDUAL terms
    # (even with zero actual ordered/positional match), because Dirichlet
    # smoothing gives every pooled doc a finite belief ("never a hard zero", see
    # indri/model.py's Deviations). Lucene's SpanNearQuery/field TermQuery, by
    # contrast, can ONLY match docs with an actual literal/positional occurrence
    # -- it has no mechanism to "smooth in" a doc that doesn't structurally
    # satisfy the query, which is standard (and desirable, at corpus scale) Lucene
    # behavior, not a bug. So bare window/field-restricted queries are tested for
    # their TRUE positional/field semantics directly (see the "known answer"
    # tests below), and for exact match-SET parity only when wrapped in #band
    # (whose "matches" test IS a hard boolean gate in BOTH engines):
    "#band(#od1(white house))",
    "#band(#1(white house))",
    "#band(#uw8(white house))",
    "#band(dog.title)",
]


@pytest.mark.parametrize("q", INDRI_BOOLEAN_QUERIES)
def test_indri_lucene_match_set_parity(indri_ex, lucene_eng, q):
    ref = _indri_set(indri_ex, q)
    got = _lucene_indri_set(lucene_eng, q)
    assert got == ref, f"{q!r}: lucene={sorted(got)} vs reference={sorted(ref)}"


def test_indri_lucene_field_restriction_exact_match_set(indri_ex, lucene_eng):
    """`#band(dog.title)` forces the EXACT (unstemmed) field path in both engines
    -- true match-set parity. The BARE (unwrapped) `dog.title` is a GRADED query
    (see INDRI_BOOLEAN_QUERIES's note) and additionally uses Lucene's STEMMED
    `title` field for scoring (documented in indri_compiler.py's compile table:
    "broader recall, standard IR practice"), so an English-stemmed plural like
    "Dogs" legitimately matches a `dog` query in Lucene but not in the
    never-stems Python reference -- a real, accepted, documented divergence
    checked here (not asserted equal, just bounded: the reference's exact hits
    must be a SUBSET of Lucene's broader stemmed hits)."""
    ref = _indri_set(indri_ex, "#band(dog.title)")
    got = _lucene_indri_set(lucene_eng, "#band(dog.title)")
    assert got == ref == {"onlydog", "titlehit"}

    ref_bare = _indri_set(indri_ex, "dog.title")
    got_bare = _lucene_indri_set(lucene_eng, "dog.title")
    assert "titlehit" in got_bare and "bodyhit" not in got_bare
    assert ref_bare <= got_bare, (
        "lucene's stemmed-field scoring should be a strict superset of the "
        f"reference's unstemmed matches -- ref={ref_bare} got={got_bare}")


def test_indri_lucene_section_restriction_match_set(indri_ex, lucene_eng):
    ref = _indri_set(indri_ex, "kangaroo.section")
    got = _lucene_indri_set(lucene_eng, "kangaroo.section")
    assert got == ref == {"sectionhit"}


def test_indri_lucene_wildcard_match_set(indri_ex, lucene_eng):
    ref = _indri_set(indri_ex, "wild*")
    got = _lucene_indri_set(lucene_eng, "wild*")
    assert got == ref == {"wild1"}


# --- (2) graded ranking: rank-correlation threshold (documented approximation) ---
# Threshold chosen empirically for this fixture + the SUM-vs-weighted-MEAN
# approximation documented in indri_compiler.py; a hard 1.0 would fail on the
# documented deviation alone, and 0.0 would let a broken compiler pass.
RANK_CORR_THRESHOLD = 0.6

INDRI_GRADED_QUERIES = [
    "#combine(dog train)",
    "#weight(1.0 dog 0.5 train)",
    "dog",
    "#max(dog train)",
]


@pytest.mark.parametrize("q", INDRI_GRADED_QUERIES)
def test_indri_lucene_rank_correlation(indri_ex, lucene_eng, q):
    ref = indri_ex.search(q, k=N)
    got = lucene_eng.search_indri(q, k=N)
    assert got.error is None
    ref_ids = [d for d, _ in ref.hits]
    got_ids = [h.doc_id for h in got.hits]
    assert set(got_ids) == set(ref_ids), (
        f"{q!r}: graded queries should still cover the SAME candidate set "
        f"(no zero-hit pathology) -- lucene={sorted(got_ids)} ref={sorted(ref_ids)}")
    rho = _spearman(ref_ids, got_ids)
    assert rho >= RANK_CORR_THRESHOLD, f"{q!r}: spearman={rho:.3f} ref={ref_ids} got={got_ids}"


def test_indri_lucene_or_candidate_set_parity(indri_ex, lucene_eng):
    """`#or` is excluded from the strict rank-correlation check above: the
    reference's `#or(a b)` belief (`log(1 - (1-p_a)(1-p_b))`) gives EVERY pooled
    doc a contribution from BOTH `a` and `b` (Dirichlet smoothing never zeroes
    out an absent term -- see indri/model.py's Deviations). Lucene's SHOULD-
    composed disjunction, by contrast, only scores the clauses a doc ACTUALLY
    matches -- a doc matching only `a` gets NO credit for `b` at all. On a small
    fixture (n=4 candidates here) that materially reorders results (confirmed:
    an ordinary rank-correlation run measured rho=-0.2 on this exact query,
    despite an otherwise-correct compiler -- see git history), so only candidate-
    SET parity (not ranking direction) is asserted for `#or`."""
    ref = _indri_set(indri_ex, "#or(dog cat)")
    got = _lucene_indri_set(lucene_eng, "#or(dog cat)")
    assert got == ref


# --- (3) BQL: match-set parity ----------------------------------------------------

BQL_QUERIES = [
    "dog",
    "AND(dog, train)",
    "AND(dog, NOT(cat))",
    "OR(dog, cat)",
    "IN(title, dog)",
    "IN(section, kangaroo)",
    "PREFIX(wild)",
    "NEAR/w5(dog, train)",
    'PHRASE(white, house)',
    "IN(author, jane)",
    "IN(author, smith)",
]


@pytest.mark.parametrize("q", BQL_QUERIES)
def test_bql_lucene_match_set_parity(bql_ex, lucene_eng, q):
    ref = _bql_set(bql_ex, q)
    got = _lucene_bql_set(lucene_eng, q)
    assert got == ref, f"{q!r}: lucene={sorted(got)} vs reference={sorted(ref)}"


def test_bql_lucene_date_range(bql_ex, lucene_eng):
    q = "IN(date, __daterange__1980-01-01__1989-12-31)"
    ref = _bql_set(bql_ex, q)
    got = _lucene_bql_set(lucene_eng, q)
    assert got == ref == {"d1980", "d1985", "d1989"}


def test_bql_lucene_unsupported_region_raises_documented_error(lucene_eng):
    r = lucene_eng.search_bql("IN(def, dog)", k=5)
    assert r.error is not None and "code-AST regions" in r.error


# --- (4) hand-built docs with known answers (spans / date / field ops) ----------

def test_span_od_vs_uw_known_answer(lucene_eng):
    """#od1 (exact phrase) must NOT match the scrambled/gapped docs; #uw8 must
    match all three (unordered-within-window)."""
    od = _lucene_indri_set(lucene_eng, "#od1(white house)")
    uw = _lucene_indri_set(lucene_eng, "#uw8(white house)")
    assert od == {"phrase"}
    assert {"phrase", "scrambled"} <= uw


def test_date_before_after_between_known_answer(lucene_eng):
    before = _lucene_indri_set(lucene_eng, "#date:before(1985)")
    after = _lucene_indri_set(lucene_eng, "#date:after(1985)")
    between = _lucene_indri_set(lucene_eng, "#date:between(1980 1989)")
    assert before == {"d1979", "d1980"}
    assert after == {"d1989", "d1990"}
    assert between == {"d1980", "d1985", "d1989"}


# --- date-inside-combine regression: a SEPARATE small corpus/index (not the
# shared `units`/`lucene_eng` fixtures) so its dates don't collide with the
# `before(1985)`/`after(1985)`/`between(1980,1989)` known-answer assertions
# above -- this is the exact adversarial-verification HIGH finding's repro
# shape: `#combine(#date:between(...) TERM)` must exclude out-of-range docs
# that match TERM, not leak them in via a scored SHOULD clause. -------------

def _date_combine_docs() -> list[dict]:
    return [
        {"_id": "in_range_plain", "title": "d1", "text": "alpha filler", "date": "2002-06-01"},
        {"_id": "in_range_univ", "title": "d2", "text": "alpha university", "date": "2002-09-01"},
        {"_id": "out_range_univ", "title": "d3", "text": "alpha university", "date": "1999-01-01"},
        {"_id": "out_range_plain", "title": "d4", "text": "alpha filler", "date": "1999-06-01"},
    ]


@pytest.fixture(scope="module")
def date_combine_units():
    return units_from_documents(_date_combine_docs())


@pytest.fixture(scope="module")
def date_combine_indri_ex(date_combine_units) -> IndriExecutor:
    return IndriExecutor(date_combine_units, mu=2500)


@pytest.fixture(scope="module")
def date_combine_lucene_eng(tmp_path_factory, date_combine_units) -> LuceneStructuredEngine:
    idx_root = str(tmp_path_factory.mktemp("lucene_date_combine_validation"))
    index_builder.build(date_combine_units, idx_root, "fixture2", rebuild=True, progress=False)
    eng = LuceneStructuredEngine(index_root=idx_root, dataset="fixture2", mu=2500)
    yield eng
    eng.close()
    shutil.rmtree(idx_root, ignore_errors=True)


def test_date_filter_inside_combine_is_hard_gate_not_should(
        date_combine_indri_ex, date_combine_lucene_eng):
    """Regression for the adversarial-verification HIGH finding: a `#date:...`
    operator embedded inside `#combine(...)` was compiling to a constant-score
    `Occur.SHOULD` clause -- an OPTIONAL scored clause, not a filter -- so any
    doc matching the OTHER #combine term (e.g. "university") leaked into the
    results regardless of date. Live repro that caught this:
    `#combine(#date:between(2002-01-01 2002-12-31) university)` returned
    61/80 out-of-range docs against a real corpus (python reference returns 0).
    Here: "out_range_univ" (dated 1999, outside [2002-01-01,2002-12-31])
    matches "university" and must NOT appear; "in_range_univ" (dated 2002)
    must appear; "in_range_plain" (in range, no "university") must still
    appear via the filter alone (date is a hard gate, "university" only ranks
    -- matching model.py's `Combine` belief: `mean()` collapses to `NEG_INF`,
    i.e. excluded, the instant the date child fails, regardless of the other
    child's score)."""
    q = "#combine(#date:between(2002-01-01 2002-12-31) university)"
    ref = _indri_set(date_combine_indri_ex, q)
    got = _lucene_indri_set(date_combine_lucene_eng, q)
    assert got == ref, f"lucene={sorted(got)} vs reference={sorted(ref)}"
    assert got == {"in_range_plain", "in_range_univ"}
    assert "out_range_univ" not in got, (
        "out-of-range doc leaked into #combine results via the date operator's "
        "SHOULD clause -- the date filter regression")


def test_date_filter_inside_weight_is_hard_gate(date_combine_indri_ex, date_combine_lucene_eng):
    """Same HIGH-finding regression as `#combine`, but for `#weight` (same
    weighted-MEAN belief math in model.py -- a nonzero-weight date child still
    hard-gates the result set)."""
    q = "#weight(1.0 #date:between(2002-01-01 2002-12-31) 2.0 university)"
    ref = _indri_set(date_combine_indri_ex, q)
    got = _lucene_indri_set(date_combine_lucene_eng, q)
    assert got == ref
    assert "out_range_univ" not in got


def test_filreq_date_operator_accepted_as_filter(date_combine_indri_ex, date_combine_lucene_eng):
    """Regression: the Python reference has no first-argument restriction that
    rejects a date operator as `#filreq`/`#filrej`'s filter argument -- `_matches`
    falls through to `belief != NEG_INF` for any node it doesn't special-case,
    and a date operator's belief IS already that hard 0.0-or-NEG_INF gate. The
    Lucene compiler previously raised `LuceneCompileError` here (`compile_exact`
    only accepted `_LEAFISH` nodes); it must now ACCEPT it and match the
    reference's match set exactly."""
    q = "#filreq(#date:between(2002-01-01 2002-12-31) university)"
    ref = _indri_set(date_combine_indri_ex, q)
    got = date_combine_lucene_eng.search_indri(q, k=N)
    assert got.error is None, f"lucene should accept #filreq(#date:...): {got.error}"
    got_set = {h.doc_id for h in got.hits}
    assert got_set == ref == {"in_range_plain", "in_range_univ"}


def test_filreq_filrej_known_answer(lucene_eng):
    """`A` (sheep) FILTERS the match set; `Q` (dolly) only RANKS it (Occur.SHOULD,
    not MUST -- see indri_compiler.py's Deviations) so an A-matching doc lacking
    Q's terms is still included, just ranked low. "sheepwool" contains "sheep"
    (matches A) but not "dolly" -- correctly present in filreq's match set."""
    filreq = _lucene_indri_set(lucene_eng, "#filreq(sheep dolly)")
    filrej = _lucene_indri_set(lucene_eng, "#filrej(sheep dolly)")
    assert filreq == {"sheepdolly", "sheepwool"}
    filreq_top = lucene_eng.search_indri("#filreq(sheep dolly)", k=1)
    assert filreq_top.hits[0].doc_id == "sheepdolly"     # Q still ranks "dolly" first
    assert "sheepdolly" not in filrej and "sheepwool" not in filrej   # A excludes both


def test_band_known_answer(lucene_eng):
    got = _lucene_indri_set(lucene_eng, "#band(red blue)")
    assert got == {"bandboth"}


# --- (5) latency smoke: the compiled query actually executes end-to-end ---------

def test_indri_search_returns_scored_hits_with_metadata(lucene_eng):
    r = lucene_eng.search_indri("#combine(dog train)", k=3)
    assert r.error is None
    assert r.hits
    assert all(h.score >= 0 or h.score <= 0 for h in r.hits)   # score is a real float
    assert r.hits[0].doc_id is not None
    # matched_fields is a best-effort explain()-derived hint, not guaranteed non-empty
    # for every query shape, but must never raise (see engine.py's _matched_fields).
    assert isinstance(r.hits[0].matched_fields, tuple)


# --- doc-count congruence guard (adversarial-verification HIGH-latent finding) --------------

def test_is_built_rejects_stale_doc_count_and_rebuilds(tmp_path):
    """Regression: `index_builder.is_built`/`has_segments` used to be a bare
    `segments_*`-file presence check, so an index left on disk under
    `<index_root>/lucene_structured/<dataset>/` whose corpus later changed (docs
    added/removed) was silently reused by `build()`'s skip-if-already-built path --
    serving hits for the WRONG document set with no error anywhere. `is_built` now
    also cross-checks the index's own `numDocs()` (a cheap `DirectoryReader` open,
    no postings I/O -- see `_lucene_doc_count`) against the CURRENT corpus size when
    `build()` calls it, mirroring the doc-count congruence checks added to
    `retrievers/dense/dense.py` and `retrievers/lexical/pyserini.py`."""
    idx_root = str(tmp_path)
    dataset = "congruence_test"

    docs_a = [{"_id": f"a{i}", "title": f"a{i}", "text": f"filler {i}"} for i in range(5)]
    units_a = units_from_documents(docs_a)
    stats_a = index_builder.build(units_a, idx_root, dataset, rebuild=True, progress=False)
    assert stats_a["n_docs"] == 5 and not stats_a["skipped"]
    assert index_builder.is_built(idx_root, dataset, expected_n_docs=5)
    assert not index_builder.is_built(idx_root, dataset, expected_n_docs=8)

    # A different-sized corpus reusing the SAME dataset key, rebuild=False (the
    # normal reuse path) -- build() must detect the incongruence and rebuild.
    docs_b = [{"_id": f"b{i}", "title": f"b{i}", "text": f"filler {i}"} for i in range(8)]
    units_b = units_from_documents(docs_b)
    stats_b = index_builder.build(units_b, idx_root, dataset, rebuild=False, progress=False)
    assert not stats_b["skipped"], "stale doc-count index was reused instead of rebuilt"
    assert index_builder.is_built(idx_root, dataset, expected_n_docs=8)

    eng = LuceneStructuredEngine(index_root=idx_root, dataset=dataset, mu=2500)
    try:
        r = eng.search_indri("filler", k=20)
        got = {h.doc_id for h in r.hits}
        assert got and got <= {u.doc_id for u in units_b}
        assert not (got & {u.doc_id for u in units_a}), (
            "stale corpus_a doc_ids leaked through -- the doc-count congruence "
            "check did not trigger a rebuild")
    finally:
        eng.close()


def test_is_built_rejects_stale_fingerprint_same_doc_count_and_rebuilds(tmp_path):
    """Same doc COUNT (and same doc_ids) but DIFFERENT content -- the doc-count check above
    can't see a unit edited in place. `build()` also writes a `corpus_fingerprint`
    (agent_search.corpus.fingerprint.corpus_fingerprint) into a `meta.json` sidecar and
    `is_built` cross-checks it; a same-count, different-CONTENT reuse of the same dataset
    key must still trigger a rebuild instead of silently serving stale postings."""
    idx_root = str(tmp_path)
    dataset = "fingerprint_congruence_test"

    docs_a = [{"_id": "d0", "title": "d0", "text": "alpha content zero"}]
    units_a = units_from_documents(docs_a)
    stats_a = index_builder.build(units_a, idx_root, dataset, rebuild=True, progress=False)
    assert stats_a["n_docs"] == 1 and not stats_a["skipped"]
    meta_path = os.path.join(index_builder.index_dir(idx_root, dataset), "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta.get("corpus_fingerprint")

    # SAME doc_id, SAME count, DIFFERENT text -- the doc-count check alone would trust this.
    docs_b = [{"_id": "d0", "title": "d0", "text": "totally different beta wording"}]
    units_b = units_from_documents(docs_b)
    assert not index_builder.is_built(
        idx_root, dataset, expected_n_docs=1,
        expected_fingerprint=index_builder.corpus_fingerprint(units_b))
    stats_b = index_builder.build(units_b, idx_root, dataset, rebuild=False, progress=False)
    assert not stats_b["skipped"], "stale-content index was reused instead of rebuilt"

    eng = LuceneStructuredEngine(index_root=idx_root, dataset=dataset, mu=2500)
    try:
        got_alpha = {h.doc_id for h in eng.search_indri("alpha", k=10).hits}
        got_beta = {h.doc_id for h in eng.search_indri("beta", k=10).hits}
        assert not got_alpha, "stale content served after an in-place edit"
        assert got_beta == {"d0"}
    finally:
        eng.close()


# --- unknown-field warning: LOW finding (adversarial verification) --------------------
# `#date:...` aside, an unrecognized `.field` name (e.g. `.bogusfield`) is a VALID Indri
# QL query in both engines' compilers -- neither raises -- but the two engines diverge
# SILENTLY on what it returns: python's Dirichlet smoothing never hard-zeros (still
# returns real hits), Lucene's `_resolve_fields` "unknown -> filter-only, no scoring
# field" path returns 0 hits. Both must now attach a visible `.warning` either way.

def test_lucene_unknown_field_returns_zero_hits_with_warning(lucene_eng):
    r = lucene_eng.search_indri("dog.bogusfield", k=10)
    assert r.error is None
    assert r.hits == []                        # Lucene's silent-0-hits half of the finding
    assert r.warning is not None and "bogusfield" in r.warning


def test_lucene_known_field_has_no_warning(lucene_eng):
    r = lucene_eng.search_indri("dog.title", k=10)
    assert r.warning is None


def test_lucene_and_python_warn_identically_for_the_same_unknown_field(indri_ex, lucene_eng):
    ref = indri_ex.search("dog.bogusfield", k=10)
    got = lucene_eng.search_indri("dog.bogusfield", k=10)
    assert ref.hits, "python engine should still return hits (never a hard zero)"
    assert got.hits == [], "lucene engine silently returns zero hits -- the finding"
    assert ref.warning == got.warning, "both engines must warn identically"


def test_isearch_tool_text_shows_zero_hits_and_warning_under_lucene_backend(lucene_eng, units):
    """End-to-end through the agent-facing tool layer (doc_indri.IndriFetchWorkspace):
    the Lucene backend's 0-hit result must render a VISIBLE warning line, not just
    silently report "(0 hits)" indistinguishable from a normal empty-result query."""
    from agent_search.agent.tools.doc_indri import IndriFetchWorkspace
    adapter = LuceneIndriAdapter(lucene_eng)
    ws = IndriFetchWorkspace(units, executor=adapter, op_nudge=False)
    out = ws.run("isearch", {"query": "dog.bogusfield"})
    assert "(0 hits)" in out
    assert "warning" in out.lower() and "bogusfield" in out
