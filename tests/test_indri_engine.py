"""Tests for the Indri query language on the document engine (`LuceneIndriAdapter` over the
Lucene structured index, `agent_search/retrievers/lucene/`).

CPU-only; a small synthetic ~30-doc corpus with dates, titles, authors and sections exercises
the parser subset, the belief-combination operators, proximity windows, synonyms, field
restriction, filters, date operators, `#band`, the graded-ranking guarantee (a `#combine`
with no full match still returns hits) and the result shape. Every ranking assertion is a
semantic (which documents match, which ranks above which), never a score inherited from
another engine. Needs a JVM; the module skips without one.
"""
from __future__ import annotations

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.indri import parser as P
from agent_search.retrievers.indri.result import IndriResult
from tests.lucene_support import build_lucene_indri, require_jvm

require_jvm()

# --- fixtures -----------------------------------------------------------------


def _mk(doc_id: str, code: str, title: str | None = None, section: str | None = None,
        author: str | None = None, date: str | None = None) -> CodeUnit:
    meta = {}
    if author is not None:
        meta["author"] = author
    if date is not None:
        meta["date"] = date
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                     start_line=1, end_line=1, code=code, title=title, section=section,
                     metadata=meta)


def _corpus() -> list[CodeUnit]:
    units = [
        # --- combine/weight ranking fixtures ---
        _mk("both", "dog train dog train station", title="Trains and Dogs"),
        _mk("onlydog", "dog dog dog dog park walk", title="Dog Park"),
        _mk("onlytrain", "train station schedule arrival", title="Train Schedule"),
        _mk("neither", "cat mouse bird fish garden", title="Animals"),
        # --- phrase / window fixtures ---
        _mk("phrase", "the white house announced today", title="Phrase doc"),
        _mk("scrambled", "house white today announced the", title="Scrambled doc"),
        _mk("gap1", "white small house announced", title="Gap1 doc"),
        # --- syn fixtures ---
        _mk("hascar", "i bought a car yesterday", title="Car doc"),
        _mk("hasauto", "i bought an automobile yesterday", title="Auto doc"),
        _mk("hasneither", "i bought a bicycle yesterday", title="Bike doc"),
        # --- field restriction fixtures ---
        _mk("titlehit", "irrelevant filler body text here", title="dog show winners"),
        _mk("bodyhit", "dog show winners announced today", title="irrelevant filler"),
        # --- filreq/filrej fixtures ---
        _mk("sheepdolly", "scientists cloned a sheep named dolly", title="Dolly"),
        _mk("sheepwool", "the sheep produced fine wool", title="Wool"),
        _mk("dollynosheep", "dolly parton released a new album", title="Music"),
        # --- date fixtures ---
        _mk("d1979", "alpha before", date="1979-12-31"),
        _mk("d1980", "alpha instart", date="1980-01-01"),
        _mk("d1985", "alpha midyear", date="1985-06-15"),
        _mk("d1989", "alpha inend", date="1989-12-31"),
        _mk("d1990", "alpha after", date="1990-01-01"),
        _mk("dmissing", "alpha nodate"),
        _mk("dmalformed", "alpha baddate", date="13/25/2002"),
        # --- band fixture ---
        _mk("bandboth", "red blue green", title="RGB"),
        _mk("bandonered", "red only here", title="Red"),
        _mk("bandoneblue", "blue only here", title="Blue"),
        # --- graded-ranking fixture: 5 rare terms, no doc has all 5 ---
        _mk("nz1", "alpha bravo filler filler filler"),
        _mk("nz2", "charlie delta filler filler filler"),
        _mk("nz3", "echo filler filler filler filler"),
        _mk("nz4", "filler filler filler filler filler"),
        _mk("diagmiss", "quux appears here but the other term does not", title="Diag doc"),
    ]
    return units


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture(scope="module")
def ex(units):
    return build_lucene_indri(units)


def _ids(res) -> list:
    assert res.error is None, res.error
    return [d for d, _ in res.hits]


# --- (1) parser -----------------------------------------------------------------


def test_parser_round_trips_each_operator():
    cases = [
        "dog",
        '"NASA"',
        "term*",
        "#od1(white house)",
        "#1(white house)",
        "#uw8(a b c)",
        "#od(a b)",
        "#uw(a b)",
        "#syn(a b)",
        "{a b}",
        "<a b>",
        "#wsyn(1.0 car 0.5 automobile)",
        "#combine(dog train)",
        "#weight(1.0 dog 0.5 train)",
        "#or(dog cat)",
        "#not(dog)",
        "#max(dog train)",
        "#band(dog train)",
        "dog.title",
        "dog.(title)",
        "dog.title.(title)",
        "dog.title,section",
        "#filreq(sheep #combine(dolly cloning))",
        "#filrej(sheep #combine(dolly cloning))",
        "#date:before(2020-01-01)",
        "#date:after(2020)",
        "#date:between(2020-01-01 2021-01-01)",
        "#datebefore(2020-01-01)",
        "#dateafter(2020-01-01)",
        "#datebetween(2020 2021)",
        "#between(date 1980 1989)",
    ]
    for q in cases:
        r = P.parse(q)
        assert r.ok, f"{q!r} failed to parse: {r.error}"
        assert r.expr is not None


@pytest.mark.parametrize("q,op", [
    ("#prior(RECENT)", "prior"),
    ("#any:DATE", "any"),
    ("#wsum(1.0 dog 0.5 dog.(title))", "wsum"),
    ("#wand(dog train)", "wand"),
    ("#sum(dog train)", "sum"),
    ("#base64hash(abc)", None),
    ("#combine[passage200:100](query)", "combine"),
    ("#less(READINGLEVEL 10)", "less"),
])
def test_parser_structured_error_on_unsupported(q, op):
    r = P.parse(q)
    assert not r.ok
    assert r.error is not None
    assert r.error.pos is not None
    if op:
        assert r.error.op is not None and op in r.error.op.lower()


@pytest.mark.parametrize("q", [
    "((unterminated",
    "#combine(a b",
    "#foo(bar)",
    "#date:before(not-a-date)",
    "#weight(dog train)",     # missing weight numbers
])
def test_parser_error_position_on_malformed(q):
    r = P.parse(q)
    assert not r.ok
    assert isinstance(r.error.pos, int)
    assert r.error.message


# --- (2) #combine ranks the doc with both terms above the one-term docs ------------


def test_combine_ranks_both_above_one(ex):
    doc_ids = _ids(ex.search("#combine(dog train)", k=10))
    assert "both" in doc_ids
    assert doc_ids.index("both") < doc_ids.index("onlydog")
    assert doc_ids.index("both") < doc_ids.index("onlytrain")


# --- (3) #weight ordering follows the weights -----------------------------------


def test_weight_order_follows_weights(ex):
    ranks_dog = {d: i for i, d in enumerate(_ids(ex.search("#weight(5.0 dog 1.0 train)", k=10)))}
    ranks_train = {d: i for i, d in enumerate(_ids(ex.search("#weight(1.0 dog 5.0 train)", k=10)))}
    assert ranks_dog["onlydog"] < ranks_dog["onlytrain"]
    assert ranks_train["onlytrain"] < ranks_train["onlydog"]


# --- (4) proximity windows: a window matches only a real positional occurrence -----


def test_od1_matches_the_phrase_only(ex):
    got = set(_ids(ex.search("#od1(white house)", k=10)))
    assert "phrase" in got
    assert "scrambled" not in got
    assert "gap1" not in got


def test_uw_matches_both_orders(ex):
    uw = set(_ids(ex.search("#uw8(white house)", k=10)))
    assert {"phrase", "scrambled", "gap1"} <= uw


def test_od2_matches_the_one_word_gap(ex):
    od1 = set(_ids(ex.search("#od1(white house)", k=10)))
    od2 = set(_ids(ex.search("#od2(white house)", k=10)))
    assert "gap1" in od2 and "gap1" not in od1
    assert "scrambled" not in od2


# --- (5) #syn: the synonyms count as one term ----------------------------------------


def test_syn_union_counts(ex):
    res = ex.search("#syn(car automobile)", k=10)
    scores = dict(res.hits)
    assert "hascar" in scores and "hasauto" in scores
    assert "hasneither" not in scores
    # both docs have the same length and one synonym occurrence each: the same score
    assert scores["hascar"] == pytest.approx(scores["hasauto"], rel=1e-6)


# --- (6) field restriction ---------------------------------------------------------


def test_field_restriction_title_only(ex):
    got = set(_ids(ex.search("dog.title", k=10)))
    assert "titlehit" in got
    assert "bodyhit" not in got          # "dog" only in the body


# --- (6b) unknown field: a valid query on a field with no postings gets a warning ----

def test_unknown_field_gets_no_error_but_a_warning(ex):
    res = ex.search("dog.bogusfield", k=10)
    assert res.error is None
    assert res.hits == []
    assert res.warning is not None
    assert "bogusfield" in res.warning


def test_known_field_has_no_warning(ex):
    res = ex.search("dog.title", k=10)
    assert res.warning is None


def test_unknown_field_warning_names_every_unrecognized_field(ex):
    res = ex.search("#band(dog.bogusfield train.alsofake)", k=10)
    assert res.warning is not None
    assert "bogusfield" in res.warning and "alsofake" in res.warning


def test_unknown_field_warning_survives_a_zero_hit_error_free_query(ex):
    res = ex.search("zzz_not_a_real_term_zzz.bogusfield", k=10)
    assert res.error is None
    assert res.warning is not None


# --- (7) #filreq / #filrej -----------------------------------------------------------


def test_filreq_restricts_candidates(ex):
    doc_ids = set(_ids(ex.search("#filreq(sheep #combine(dolly cloning))", k=10)))
    assert "sheepdolly" in doc_ids
    assert "sheepwool" in doc_ids
    assert "dollynosheep" not in doc_ids       # has "dolly" but not "sheep"


def test_filrej_excludes_matching(ex):
    # candidates = docs NOT matching "sheep", ranked by "dolly": the sheep docs are out and
    # the only doc carrying "dolly" without "sheep" ranks first.
    doc_ids = _ids(ex.search("#filrej(sheep dolly)", k=40))
    assert "sheepdolly" not in doc_ids
    assert "sheepwool" not in doc_ids
    assert doc_ids[0] == "dollynosheep"


# --- (8) date filters ----------------------------------------------------------------


def test_date_between_filters_iso_dates(ex):
    doc_ids = set(_ids(ex.search("#date:between(1980-01-01 1989-12-31)", k=20)))
    assert doc_ids == {"d1980", "d1985", "d1989"}


def test_date_between_both_spellings_agree(ex):
    a = set(_ids(ex.search("#date:between(1980-01-01 1989-12-31)", k=20)))
    b = set(_ids(ex.search("#datebetween(1980-01-01 1989-12-31)", k=20)))
    assert a == b == {"d1980", "d1985", "d1989"}


def test_date_before_after(ex):
    before = set(_ids(ex.search("#date:before(1980-01-01)", k=20)))
    after = set(_ids(ex.search("#date:after(1989-12-31)", k=20)))
    assert "d1979" in before
    assert "d1980" not in before
    assert "dmissing" not in before
    assert after == {"d1990"}
    # a non-ISO date ("13/25/2002") is left out of the range field, so it matches no range
    assert "dmalformed" not in before


def test_between_date_numeric_form(ex):
    doc_ids = set(_ids(ex.search("#between(date 1980 1989)", k=20)))
    assert doc_ids == {"d1980", "d1985", "d1989"}


# --- (9) #band requires all children ------------------------------------------------


def test_band_requires_all_children(ex):
    doc_ids = set(_ids(ex.search("#band(red blue)", k=10)))
    assert doc_ids == {"bandboth"}


# --- (10) graded ranking: a #combine no document fully satisfies still returns hits ----


def test_combine_with_no_full_match_still_returns_hits(units, ex):
    res = ex.search("#combine(alpha bravo charlie delta echo)", k=10)
    assert res.error is None
    assert len(res.hits) > 0
    text = {u.doc_id: u.code for u in units}
    for doc_id, _ in res.hits:
        present = sum(t in text[doc_id] for t in ("alpha", "bravo", "charlie", "delta", "echo"))
        assert 0 < present < 5


# --- misc: public API surface / result shape ------------------------------------------


def test_search_returns_indri_result_with_expected_shape(ex):
    res = ex.search("dog", k=3)
    assert isinstance(res, IndriResult)
    assert res.error is None
    assert isinstance(res.hits, list)
    assert all(isinstance(h, tuple) and len(h) == 2 for h in res.hits)
    assert isinstance(res.diagnostics, list)


def test_search_syntax_error_surfaces_structured_message(ex):
    res = ex.search("#nosuchop(x)", k=3)
    assert res.hits == []
    assert res.error is not None
    assert "nosuchop" in res.error
