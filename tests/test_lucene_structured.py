"""Known-answer tests for the Lucene structured engine (`agent_search/retrievers/lucene/`),
the engine behind the document BQL and Indri tools.

Every test asserts the match set (and, where the semantic demands it, the order) a
hand-built fixture corpus must produce for a query:
  1. Boolean and filter Indri queries (`#band`, `#filreq`/`#filrej`, date ranges).
  2. Span, field and wildcard Indri queries on their exact positional/field semantics.
  3. BQL queries over every document region the schema indexes (title, body, section,
     author, date ranges), plus the documented rejection of the code-AST regions.
  4. Date operators inside `#combine`/`#weight`/`#filreq` as hard gates.
  5. Index congruence: a stale index (doc count or fingerprint) is rebuilt, never reused.
  6. The unknown-field warning, through the engine and through the `isearch` tool.

Requires a real JVM (pyserini/pyjnius). The shared fixture index is built once per corpus
under `tests/lucene_support.py`'s session root; JVM boot plus the ~30-doc index build is the
expensive part, not the per-test queries.
"""
from __future__ import annotations

import json
import os

import pytest

from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lucene import index_builder
from agent_search.retrievers.lucene.adapters import LuceneIndriAdapter
from agent_search.retrievers.lucene.engine import LuceneStructuredEngine
from tests import lucene_support
from tests.lucene_support import require_jvm

require_jvm()


# --- shared fixture corpus: built via `units_from_documents`, the same path a real dataset
# takes (agent_search/evaluation/datasets.py), so title and body land in the same fields they
# do in production. -------------------------------------------------------------------

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
        # --- graded ranking with no full match ---
        {"_id": "nz1", "title": "nz1", "text": "kappa lambda filler filler filler"},
        {"_id": "nz2", "title": "nz2", "text": "mu nu filler filler filler"},
        {"_id": "nz3", "title": "nz3", "text": "xi filler filler filler filler"},
    ]


@pytest.fixture(scope="module")
def units():
    return units_from_documents(_docs())


@pytest.fixture(scope="module")
def lucene_eng(units) -> LuceneStructuredEngine:
    key = lucene_support.build_structured_index(units)
    return LuceneStructuredEngine(index_root=lucene_support.index_root(), dataset=key, mu=2500)


N = 40   # > corpus size, so search(k=N) always returns the full match set


def _indri_set(eng, q: str) -> set:
    r = eng.search_indri(q, k=N)
    assert r.error is None, f"indri compile/exec error for {q!r}: {r.error}"
    return {h.doc_id for h in r.hits}


def _indri_ids(eng, q: str) -> list:
    r = eng.search_indri(q, k=N)
    assert r.error is None, f"indri compile/exec error for {q!r}: {r.error}"
    return [h.doc_id for h in r.hits]


def _bql_set(eng, q: str) -> set:
    r = eng.search_bql(q, k=N)
    assert r.error is None, f"bql compile/exec error for {q!r}: {r.error}"
    return {h.doc_id for h in r.hits}


# --- (1) boolean / filter Indri queries -------------------------------------------
# Bare `#odN`/`#uwN`/`.field`/`#syn` are graded queries; wrapped in `#band` the child's
# "matches" test is a hard boolean gate on the exact (unstemmed) field.

@pytest.mark.parametrize("q,expected", [
    ("#band(dog train)", {"both"}),
    ("#band(red blue)", {"bandboth"}),
    ("#band(#od1(white house))", {"phrase"}),
    ("#band(#1(white house))", {"phrase"}),
    ("#band(#uw8(white house))", {"phrase", "scrambled", "gap1"}),
    ("#band(dog.title)", {"onlydog", "titlehit"}),          # exact field: "Dogs" is not "dog"
    ("#date:before(1985)", {"d1979", "d1980"}),
    ("#date:after(1985)", {"d1989", "d1990"}),
    ("#date:between(1980 1989)", {"d1980", "d1985", "d1989"}),
])
def test_indri_boolean_match_set(lucene_eng, q, expected):
    assert _indri_set(lucene_eng, q) == expected


def test_filreq_filrej_known_answer(lucene_eng):
    """`A` (sheep) filters the match set; `Q` (dolly) only ranks it (Occur.SHOULD, not MUST),
    so an A-matching doc lacking Q's terms is still included, just ranked last. `#filrej`
    keeps every doc without `sheep` and ranks the one carrying `dolly` first."""
    filreq = _indri_ids(lucene_eng, "#filreq(sheep dolly)")
    assert set(filreq) == {"sheepdolly", "sheepwool"}
    assert filreq[0] == "sheepdolly"
    filrej = _indri_ids(lucene_eng, "#filrej(sheep dolly)")
    assert "sheepdolly" not in filrej and "sheepwool" not in filrej
    assert filrej[0] == "dollynosheep"
    assert set(filrej) == {u.doc_id for u in units_from_documents(_docs())} - {"sheepdolly", "sheepwool"}


# --- (2) span, field and wildcard Indri queries -------------------------------------


def test_span_od_vs_uw_known_answer(lucene_eng):
    """#od1 (exact phrase) matches neither the scrambled nor the gapped doc; #od2 admits the
    one-word gap; #uw8 matches all three (unordered within the window)."""
    assert _indri_set(lucene_eng, "#od1(white house)") == {"phrase"}
    assert _indri_set(lucene_eng, "#od2(white house)") == {"phrase", "gap1"}
    assert _indri_set(lucene_eng, "#uw8(white house)") == {"phrase", "scrambled", "gap1"}


def test_syn_known_answer(lucene_eng):
    r = lucene_eng.search_indri("#syn(car automobile)", k=N)
    assert r.error is None
    scores = {h.doc_id: h.score for h in r.hits}
    assert set(scores) == {"hascar", "hasauto"}
    # one term for scoring purposes: equal-length docs with one occurrence each score alike
    assert scores["hascar"] == pytest.approx(scores["hasauto"], rel=1e-6)


def test_bare_term_scores_the_body_only(lucene_eng):
    assert _indri_set(lucene_eng, "dog") == {"both", "onlydog", "bodyhit"}
    assert _indri_set(lucene_eng, "#or(dog cat)") == {"both", "onlydog", "bodyhit", "neither"}


def test_field_restriction_known_answer(lucene_eng):
    """A bare `.title` restriction scores the stemmed `title` field (broader recall, standard
    IR practice), so the plural "Dogs" also matches; `#band(dog.title)` (above) is the exact
    unstemmed gate. Either way the body-only hit is out."""
    got = _indri_set(lucene_eng, "dog.title")
    assert got == {"both", "onlydog", "titlehit"}
    assert "bodyhit" not in got


def test_section_restriction_known_answer(lucene_eng):
    assert _indri_set(lucene_eng, "kangaroo.section") == {"sectionhit"}


def test_wildcard_known_answer(lucene_eng):
    assert _indri_set(lucene_eng, "wild*") == {"wild1"}


def test_graded_queries_rank_the_full_match_above_the_partial(lucene_eng):
    for q in ("#combine(dog train)", "#max(dog train)"):
        ids = _indri_ids(lucene_eng, q)
        assert set(ids) == {"both", "onlydog", "onlytrain", "bodyhit"}, q
        assert ids[0] == "both", q


def test_weight_order_follows_weights(lucene_eng):
    favour_dog = _indri_ids(lucene_eng, "#weight(5.0 dog 1.0 train)")
    favour_train = _indri_ids(lucene_eng, "#weight(1.0 dog 5.0 train)")
    assert favour_dog.index("onlydog") < favour_dog.index("onlytrain")
    assert favour_train.index("onlytrain") < favour_train.index("onlydog")


def test_combine_with_no_full_match_still_returns_hits(lucene_eng):
    got = _indri_set(lucene_eng, "#combine(kappa lambda mu nu xi)")
    assert got == {"nz1", "nz2", "nz3"}


# --- (3) BQL: every indexed region ---------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("dog", {"both", "onlydog", "titlehit", "bodyhit"}),          # unscoped: body or title
    ("AND(dog, train)", {"both"}),
    ("AND(dog, NOT(cat))", {"both", "onlydog", "titlehit", "bodyhit"}),
    ("OR(dog, cat)", {"both", "onlydog", "titlehit", "bodyhit", "neither"}),
    ("IN(title, dog)", {"onlydog", "titlehit"}),
    ("IN(section, kangaroo)", {"sectionhit"}),
    ("PREFIX(wild)", {"wild1", "wild2"}),                          # both titles start "Wild"
    ("NEAR/w5(dog, train)", {"both"}),
    ('PHRASE(white, house)', {"phrase"}),
    ("IN(author, jane)", {"authored"}),
    ("IN(author, smith)", {"authored"}),
    ("IN(date, __daterange__1980-01-01__1989-12-31)", {"d1980", "d1985", "d1989"}),
])
def test_bql_match_set(lucene_eng, q, expected):
    assert _bql_set(lucene_eng, q) == expected


def test_bql_unsupported_region_raises_documented_error(lucene_eng):
    r = lucene_eng.search_bql("IN(def, dog)", k=5)
    assert r.error is not None and "code-AST regions" in r.error


# --- (4) date operators inside #combine / #weight / #filreq are hard gates ------------
# A separate small corpus, so its dates cannot collide with the range assertions above.

def _date_combine_docs() -> list[dict]:
    return [
        {"_id": "in_range_plain", "title": "d1", "text": "alpha filler", "date": "2002-06-01"},
        {"_id": "in_range_univ", "title": "d2", "text": "alpha university", "date": "2002-09-01"},
        {"_id": "out_range_univ", "title": "d3", "text": "alpha university", "date": "1999-01-01"},
        {"_id": "out_range_plain", "title": "d4", "text": "alpha filler", "date": "1999-06-01"},
    ]


@pytest.fixture(scope="module")
def date_combine_lucene_eng() -> LuceneStructuredEngine:
    units = units_from_documents(_date_combine_docs())
    key = lucene_support.build_structured_index(units)
    return LuceneStructuredEngine(index_root=lucene_support.index_root(), dataset=key, mu=2500)


def test_date_filter_inside_combine_is_hard_gate_not_should(date_combine_lucene_eng):
    """Regression: a `#date:...` operator inside `#combine(...)` once compiled to a
    constant-score `Occur.SHOULD` clause, so any doc matching the other term leaked in
    regardless of date. The date child is a filter: "out_range_univ" (1999) matches
    "university" and must not appear; "in_range_plain" (in range, no "university") must
    still appear through the filter alone."""
    got = _indri_set(date_combine_lucene_eng,
                     "#combine(#date:between(2002-01-01 2002-12-31) university)")
    assert got == {"in_range_plain", "in_range_univ"}


def test_date_filter_inside_weight_is_hard_gate(date_combine_lucene_eng):
    got = _indri_set(date_combine_lucene_eng,
                     "#weight(1.0 #date:between(2002-01-01 2002-12-31) 2.0 university)")
    assert got == {"in_range_plain", "in_range_univ"}


def test_filreq_date_operator_accepted_as_filter(date_combine_lucene_eng):
    """A date operator is a valid `#filreq`/`#filrej` filter argument (the compiler's
    `compile_exact` accepts it), with the same hard-gate match set."""
    q = "#filreq(#date:between(2002-01-01 2002-12-31) university)"
    r = date_combine_lucene_eng.search_indri(q, k=N)
    assert r.error is None, f"lucene should accept #filreq(#date:...): {r.error}"
    assert {h.doc_id for h in r.hits} == {"in_range_plain", "in_range_univ"}


# --- result shape --------------------------------------------------------------------

def test_indri_search_returns_scored_hits_with_metadata(lucene_eng):
    r = lucene_eng.search_indri("#combine(dog train)", k=3)
    assert r.error is None
    assert r.hits
    assert all(isinstance(h.score, float) for h in r.hits)
    assert r.hits[0].doc_id is not None
    # matched_fields is a best-effort explain()-derived hint, not guaranteed non-empty for
    # every query shape, but must never raise (see engine.py's _matched_fields).
    assert isinstance(r.hits[0].matched_fields, tuple)


# --- (5) index congruence ----------------------------------------------------------------

def test_is_built_rejects_stale_doc_count_and_rebuilds(tmp_path):
    """An index left on disk under `<index_root>/lucene_structured/<dataset>/` whose corpus
    later changed (docs added or removed) must not be reused by `build()`'s skip-if-built
    path. `is_built` cross-checks the index's own `numDocs()` against the current corpus
    size, mirroring the checks in `retrievers/dense/base.py` and `lexical/pyserini.py`."""
    idx_root = str(tmp_path)
    dataset = "congruence_test"

    docs_a = [{"_id": f"a{i}", "title": f"a{i}", "text": f"filler {i}"} for i in range(5)]
    units_a = units_from_documents(docs_a)
    stats_a = index_builder.build(units_a, idx_root, dataset, rebuild=True, progress=False)
    assert stats_a["n_docs"] == 5 and not stats_a["skipped"]
    assert index_builder.is_built(idx_root, dataset, expected_n_docs=5)
    assert not index_builder.is_built(idx_root, dataset, expected_n_docs=8)

    # A different-sized corpus reusing the same dataset key, rebuild=False (the normal reuse
    # path): build() must detect the incongruence and rebuild.
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
            "stale corpus_a doc_ids leaked through: the doc-count congruence check did not "
            "trigger a rebuild")
    finally:
        eng.close()


def test_is_built_rejects_stale_fingerprint_same_doc_count_and_rebuilds(tmp_path):
    """Same doc count (and same doc_ids) but different content: the doc-count check cannot
    see a unit edited in place. `build()` also writes a `corpus_fingerprint` into a
    `meta.json` sidecar and `is_built` cross-checks it, so a same-count, different-content
    reuse of the same dataset key still triggers a rebuild."""
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

    # same doc_id, same count, different text: the doc-count check alone would trust this
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


# --- (6) unknown-field warning ----------------------------------------------------------
# An unrecognized `.field` name (e.g. `.bogusfield`) is a valid Indri QL query: the compiler
# does not raise, `_resolve_fields` takes its "unknown -> filter-only, no scoring field" path
# and the search returns zero hits. The result carries a visible `.warning` so the agent can
# tell this from a genuinely zero-hit query.

def test_lucene_unknown_field_returns_zero_hits_with_warning(lucene_eng):
    r = lucene_eng.search_indri("dog.bogusfield", k=10)
    assert r.error is None
    assert r.hits == []
    assert r.warning is not None and "bogusfield" in r.warning


def test_lucene_known_field_has_no_warning(lucene_eng):
    r = lucene_eng.search_indri("dog.title", k=10)
    assert r.warning is None


def test_isearch_tool_text_shows_zero_hits_and_warning(lucene_eng, units):
    """End-to-end through the agent-facing tool layer (`search_indri`/`fetch`): the 0-hit
    result renders a visible warning line, not a bare "(0 hits)" indistinguishable from a
    normal empty result."""
    from agent_search.tools.base import EpisodeState, ToolBox
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_indri.tool import SearchIndri

    adapter = LuceneIndriAdapter(lucene_eng)
    state = EpisodeState(question="q")
    ubyid = {u.doc_id: u for u in units}
    search = SearchIndri(name="isearch", op_nudge=False).bind(state, units, ubyid, {"indri": adapter})
    fetch = Fetch(name="fetch").bind(state, units, ubyid, {})
    ws = ToolBox([search, fetch], state)
    out = ws.run("isearch", {"query": "dog.bogusfield"})
    assert "(0 hits)" in out
    assert "warning" in out.lower() and "bogusfield" in out


def test_exact_field_keeps_numbers_and_one_member_window_compiles():
    """Schema 2: the exact fields tokenize numbers (a year is a term), and a window with one
    member is that member instead of a span query Lucene rejects."""
    pytest.importorskip("jnius")
    from agent_search.retrievers.lucene import jni_utils as J
    from agent_search.retrievers.indri import parser as P
    from agent_search.retrievers.lucene.indri_compiler import compile_score
    assert J.exact_tokens("Treaty of 1848") == ["treaty", "of", "1848"]
    parsed = P.parse("#combine(#uw5(treaty 1848) #od2(guadalupe))")
    node = getattr(parsed, "tree", None) or getattr(parsed, "root", None) or getattr(parsed, "expr", None) or parsed
    assert compile_score(node) is not None
