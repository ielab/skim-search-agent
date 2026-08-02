"""Tests for the `indri` retrieval backend (agent_search/retrievers/structural/indri/).

CPU-only; a small synthetic ~30-doc corpus with dates/titles/authors/sections
exercises the parser subset, the Dirichlet-smoothed belief math (including a
hand-computable 2-doc toy asserted to exact log values), the belief-combination
operators, proximity windows, synonyms, field restriction, filters, date
operators, #band, the "no zero-hit pathology" graded-ranking guarantee,
save/load/attach_units round-tripping, and diagnostics.
"""
from __future__ import annotations

import math
import os

import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.indri import parser as P
from agent_search.retrievers.structural.indri.model import (
    IndriExecutor, IndriResult, indri_index_path, load_or_build,
)

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
        # --- no-zero-hit-pathology fixture: 5 rare terms, no doc has all 5 ---
        _mk("nz1", "alpha bravo filler filler filler"),
        _mk("nz2", "charlie delta filler filler filler"),
        _mk("nz3", "echo filler filler filler filler"),
        _mk("nz4", "filler filler filler filler filler"),
        # --- diagnostics fixture ---
        _mk("diagmiss", "quux appears here but the other term does not", title="Diag doc"),
    ]
    return units


@pytest.fixture()
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture()
def ex(units) -> IndriExecutor:
    return IndriExecutor(units, mu=2500)


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


# --- (2) hand-computable Dirichlet math -----------------------------------------


def test_dirichlet_math_exact_2doc_toy():
    units2 = [
        CodeUnit(doc_id="d1", path="d1", qualname="d1", start_line=1, end_line=1,
                 code="dog dog cat"),
        CodeUnit(doc_id="d2", path="d2", qualname="d2", start_line=1, end_line=1,
                 code="cat cat cat"),
    ]
    mu = 2500.0
    ex2 = IndriExecutor(units2, mu=mu)
    total = 6          # 3 body tokens per doc * 2 docs
    # doc0: tf(dog)=2, doclen=3, cf(dog)=2
    expected_dog_d0 = math.log((2 + mu * (2 / total)) / (3 + mu))
    # doc1: tf(dog)=0, doclen=3, cf(dog)=2 (unseen in doc but present in collection)
    expected_dog_d1 = math.log((0 + mu * (2 / total)) / (3 + mu))
    # doc0: tf(cat)=1, doclen=3, cf(cat)=4
    expected_cat_d0 = math.log((1 + mu * (4 / total)) / (3 + mu))
    term_dog = P.Term("dog")
    term_cat = P.Term("cat")
    b_dog_d0 = ex2._belief(term_dog, 0, ("body",), ("body",))
    b_dog_d1 = ex2._belief(term_dog, 1, ("body",), ("body",))
    b_cat_d0 = ex2._belief(term_cat, 0, ("body",), ("body",))
    assert b_dog_d0 == pytest.approx(expected_dog_d0, abs=1e-6)
    assert b_dog_d1 == pytest.approx(expected_dog_d1, abs=1e-6)
    assert b_cat_d0 == pytest.approx(expected_cat_d0, abs=1e-6)


def test_dirichlet_unseen_term_epsilon():
    """A term that appears NOWHERE in the collection uses the cf=0 epsilon
    (0.5/total) rather than raising or returning -inf."""
    units2 = [
        CodeUnit(doc_id="d1", path="d1", qualname="d1", start_line=1, end_line=1,
                 code="dog dog cat"),
    ]
    mu = 2500.0
    ex2 = IndriExecutor(units2, mu=mu)
    total = 3
    expected = math.log((0 + mu * (0.5 / total)) / (3 + mu))
    b = ex2._belief(P.Term("zzzznever"), 0, ("body",), ("body",))
    assert b == pytest.approx(expected, abs=1e-9)
    assert b != float("-inf")


# --- (3) #combine ranks both-terms doc above one-term doc -----------------------


def test_combine_ranks_both_above_one(ex):
    res = ex.search("#combine(dog train)", k=10)
    assert res.error is None
    doc_ids = [d for d, _ in res.hits]
    assert "both" in doc_ids
    assert doc_ids.index("both") < doc_ids.index("onlydog")
    assert doc_ids.index("both") < doc_ids.index("onlytrain")


# --- (4) #weight ordering changes with weights -----------------------------------


def test_weight_order_follows_weights(ex):
    res_favor_dog = ex.search("#weight(5.0 dog 1.0 train)", k=10)
    res_favor_train = ex.search("#weight(1.0 dog 5.0 train)", k=10)
    ranks_dog = {d: i for i, (d, _) in enumerate(res_favor_dog.hits)}
    ranks_train = {d: i for i, (d, _) in enumerate(res_favor_train.hits)}
    # heavily favoring "dog" should rank onlydog at least as well relative to
    # onlytrain as the reverse weighting, and the orderings should differ.
    assert ranks_dog["onlydog"] < ranks_dog["onlytrain"]
    assert ranks_train["onlytrain"] < ranks_train["onlydog"]


# --- (5) proximity windows -------------------------------------------------------


def test_od1_phrase_beats_scrambled(ex):
    res = ex.search("#od1(white house)", k=10)
    scores = dict(res.hits)
    assert scores.get("phrase", float("-inf")) > scores.get("scrambled", float("-inf"))


def test_uw_matches_both_orders_od_does_not(ex):
    uw = ex.search("#uw8(white house)", k=10)
    od1 = ex.search("#od1(white house)", k=10)
    uw_scores = dict(uw.hits)
    od1_scores = dict(od1.hits)
    # unordered window matches phrase, scrambled AND gap1 (any order, within span)
    assert uw_scores.get("phrase", float("-inf")) > float("-inf")
    assert uw_scores.get("scrambled", float("-inf")) > float("-inf")
    # ordered #od1 (exact adjacency) must NOT match the scrambled doc as well as
    # it matches the true phrase doc.
    assert od1_scores.get("phrase", float("-inf")) > od1_scores.get("scrambled", float("-inf"))


def test_od2_matches_gapped_but_od1_does_not_as_well(ex):
    od1 = dict(ex.search("#od1(white house)", k=10).hits)
    od2 = dict(ex.search("#od2(white house)", k=10).hits)
    # #od2(white house) matches "white * house" (gap1: "white small house")
    assert od2.get("gap1", float("-inf")) > od1.get("gap1", float("-inf"))


# --- (6) #syn union counts --------------------------------------------------------


def test_syn_union_counts(ex):
    res = ex.search("#syn(car automobile)", k=10)
    scores = dict(res.hits)
    assert scores.get("hascar", float("-inf")) > float("-inf")
    assert scores.get("hasauto", float("-inf")) > float("-inf")
    assert scores.get("hascar") == pytest.approx(scores.get("hasauto"), abs=1e-9)
    assert scores.get("hascar", float("-inf")) > scores.get("hasneither", float("-inf"))


# --- (7) field restriction ---------------------------------------------------------


def test_field_restriction_title_only(ex):
    res = ex.search("dog.title", k=10)
    scores = dict(res.hits)
    assert scores.get("titlehit", float("-inf")) > float("-inf")
    # bodyhit has "dog" only in body, not title -> title-restricted tf=0 there
    assert scores.get("titlehit", float("-inf")) > scores.get("bodyhit", float("-inf"))


# --- (7b) unknown field: LOW finding (adversarial verification) ----------------------
# python "searches anyway" (Dirichlet smoothing never hard-zeros, so an unknown-field
# restriction still returns SOME belief) with no error and no signal that the field
# name wasn't real; the fix adds a `.warning` on the result (surfaced by doc_indri.py's
# `_search_impl`), behavior (hits/ranking) otherwise unchanged.

def test_unknown_field_gets_no_error_but_a_warning(ex):
    res = ex.search("dog.bogusfield", k=10)
    assert res.error is None                          # NOT a parse/type error
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
    # a query that structurally can't match anything (nonsense term) but still
    # parses/executes cleanly -- the warning must still surface (checked at the
    # `IndriResult` level; doc_indri.py's 0-hit text path is covered in
    # tests/test_indri_hybrids.py).
    res = ex.search("zzz_not_a_real_term_zzz.bogusfield", k=10)
    assert res.error is None
    assert res.warning is not None


# --- (8) #filreq / #filrej -----------------------------------------------------------


def test_filreq_restricts_candidates(ex):
    res = ex.search("#filreq(sheep #combine(dolly cloning))", k=10)
    doc_ids = {d for d, _ in res.hits}
    assert "sheepdolly" in doc_ids
    assert "sheepwool" in doc_ids
    assert "dollynosheep" not in doc_ids       # has "dolly" but not "sheep" -> excluded


def test_filrej_excludes_matching(ex):
    # #filrej(A Q): candidates = docs NOT matching A ("sheep"), ranked by Q ("dolly").
    # Both sheep-containing docs are excluded; "dollynosheep" (no sheep, has dolly)
    # is exactly the doc this should surface.
    res = ex.search("#filrej(sheep dolly)", k=10)
    doc_ids = {d for d, _ in res.hits}
    assert "sheepdolly" not in doc_ids
    assert "sheepwool" not in doc_ids
    assert "dollynosheep" in doc_ids


# --- (9) date filters ----------------------------------------------------------------


def test_date_between_filters_iso_dates(ex):
    res = ex.search("#date:between(1980-01-01 1989-12-31)", k=20)
    doc_ids = {d for d, _ in res.hits}
    assert "d1980" in doc_ids
    assert "d1985" in doc_ids
    assert "d1989" in doc_ids
    assert "d1979" not in doc_ids
    assert "d1990" not in doc_ids
    assert "dmissing" not in doc_ids
    assert "dmalformed" not in doc_ids


def test_date_between_both_spellings_agree(ex):
    a = {d for d, _ in ex.search("#date:between(1980-01-01 1989-12-31)", k=20).hits}
    b = {d for d, _ in ex.search("#datebetween(1980-01-01 1989-12-31)", k=20).hits}
    assert a == b == {"d1980", "d1985", "d1989"}


def test_date_before_after(ex):
    before = {d for d, _ in ex.search("#date:before(1980-01-01)", k=20).hits}
    after = {d for d, _ in ex.search("#date:after(1989-12-31)", k=20).hits}
    assert "d1979" in before
    assert "d1980" not in before
    assert "d1990" in after
    assert "d1989" not in after


def test_between_date_numeric_form(ex):
    res = {d for d, _ in ex.search("#between(date 1980 1989)", k=20).hits}
    assert res == {"d1980", "d1985", "d1989"}


# --- (10) #band requires all children ------------------------------------------------


def test_band_requires_all_children(ex):
    res = ex.search("#band(red blue)", k=10)
    doc_ids = {d for d, _ in res.hits}
    assert "bandboth" in doc_ids
    assert "bandonered" not in doc_ids
    assert "bandoneblue" not in doc_ids


# --- (11) NO ZERO-HIT PATHOLOGY --------------------------------------------------------


def test_no_zero_hit_pathology_five_term_combine(ex):
    q = "#combine(alpha bravo charlie delta echo)"
    res = ex.search(q, k=10)
    assert res.error is None
    assert len(res.hits) > 0
    # confirm the premise: no single doc actually has all five terms
    for doc_id, _ in res.hits:
        u = next(u for u in ex.units if u.doc_id == doc_id)
        present = sum(t in u.code for t in ("alpha", "bravo", "charlie", "delta", "echo"))
        assert present < 5 or doc_id not in ("nz1", "nz2", "nz3", "nz4")


# --- (12) save/load/attach_units gives identical ranking ------------------------------


def test_save_load_attach_units_identical_ranking(tmp_path, units):
    fresh = IndriExecutor(units, mu=2500)
    q = "#combine(dog train)"
    fresh_hits = fresh.search(q, k=10).hits

    path = str(tmp_path / "indri_test.pkl")
    fresh.save(path)
    loaded = IndriExecutor.load(path)
    loaded.attach_units(units)
    loaded_hits = loaded.search(q, k=10).hits

    assert fresh_hits == loaded_hits


def test_load_or_build_persists_and_reloads(tmp_path, units):
    index_root = str(tmp_path / "indexes")
    key = "unit-test-corpus"
    ex1 = load_or_build(units, index_root=index_root, key=key)
    path = indri_index_path(index_root, key)
    assert os.path.exists(path)
    ex2 = load_or_build(units, index_root=index_root, key=key)
    q = "#combine(dog train)"
    assert ex1.search(q, k=10).hits == ex2.search(q, k=10).hits


# --- (13) diagnostics name the weak child -----------------------------------------------


def test_diagnostics_name_weak_child(ex):
    res = ex.search("#combine(quux zulu)", k=5)
    assert res.error is None
    assert len(res.hits) > 0
    assert res.hits[0][0] == "diagmiss"
    assert len(res.diagnostics) == 2
    reprs = {r for r, _ in res.diagnostics}
    assert "quux" in reprs
    assert "zulu" in reprs
    by_repr = dict(res.diagnostics)
    # "quux" is present in diagmiss, "zulu" is not -> quux's belief must be higher
    assert by_repr["quux"] > by_repr["zulu"]


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


# --- (14) two-stage max-score rescoring ------------------------------------------


def test_two_stage_rescoring_top1_matches_exact(monkeypatch):
    """Force INDRI_RESCORE_M=2 (well below the pool size) so STAGE 1's
    position-free approximation + STAGE 2 rescoring definitely kicks in, then
    assert the #od2 window query still surfaces the one doc that TRULY contains
    the ordered/gapped phrase as the top-1 hit — i.e. the doc that clearly wins
    both STAGE 1 (approx tf = min(member unigram tf), which is >= its true window
    match count so it can't be pruned out of the STAGE 2 rescore set) and STAGE 2
    (exact window position matching)."""
    monkeypatch.setenv("INDRI_RESCORE_M", "2")
    units = [
        # true winner: THREE tight, real #od2 matches AND the highest per-term
        # unigram tf in the pool (tf=3 each) -> wins STAGE 1 (approx tf = min
        # member tf) outright, not just STAGE 2.
        _mk("winner", "white house white house white house"),
        # has both unigrams (so it's NOT pruned by the term pool, tf=1 each) but
        # far apart / wrong order -> should NOT satisfy #od2(white house), and
        # tf=1 < winner's tf=3 so it can't outrank winner even under STAGE 1.
        _mk("decoy_far", "house " + ("filler " * 20) + "white filler filler"),
        _mk("decoy_order", "house then much later white appears here too"),
        # extra filler docs (tf=1 each) so the pool is bigger than the forced
        # RESCORE_M=2, and to confirm they never outrank the tf=3 winner.
        _mk("filler1", "white filler house filler more unrelated filler words"),
        _mk("filler2", "house filler white filler more unrelated filler words"),
        _mk("filler3", "white filler filler house filler more filler words"),
    ]
    ex = IndriExecutor(units, mu=2500)
    res = ex.search("#od2(white house)", k=10)
    assert res.error is None
    assert res.hits, "expected at least one hit"
    assert res.hits[0][0] == "winner"


def test_two_stage_rescoring_matches_single_stage_exact_scores(monkeypatch):
    """The exact belief of the top doc under forced two-stage rescoring
    (INDRI_RESCORE_M=1) must equal the exact belief computed directly (bypassing
    search's pooling/staging entirely) — i.e. STAGE 2 really does exact scoring,
    not just an approximation relabeled."""
    units = _corpus()
    ex = IndriExecutor(units, mu=2500)
    q = "#od1(white house)"
    direct_best_idx = max(
        range(len(units)),
        key=lambda i: ex._belief(P.parse(q).expr, i, ("body",), ("body",), approx=False),
    )
    expected_score = ex._belief(P.parse(q).expr, direct_best_idx, ("body",), ("body",),
                                 approx=False)
    monkeypatch.setenv("INDRI_RESCORE_M", "1")
    res = ex.search(q, k=1)
    assert res.hits[0][0] == units[direct_best_idx].doc_id
    assert res.hits[0][1] == pytest.approx(expected_score, abs=1e-9)


def test_two_stage_rescoring_perf_smoke():
    """Synthetic 5000-doc x ~200-token corpus, one #od2 window query: before the
    two-stage fix this wedged on pure-Python position scanning across the whole
    pool (minutes); after the fix STAGE 1 (position-free) handles the pool and
    only the small STAGE 2 rescore set pays for real position matching, so this
    should comfortably finish in well under the generous 5s budget.

    Both query terms are inserted into EVERY doc (not just drawn from the random
    vocab) so the candidate pool is the full 5000 docs -- reproducing the
    real-world "common query terms -> huge pool" shape of the wedge, not just a
    rare-term case that would stay small (and skip STAGE 1) regardless of the fix.
    """
    import random
    import time

    rng = random.Random(0)
    vocab = [f"tok{i}" for i in range(200)]
    units = []
    for d in range(5000):
        toks = [rng.choice(vocab) for _ in range(198)]
        toks.insert(50, "alpha")
        if d % 137 == 0:             # sprinkle in genuine ordered-gap matches
            toks.insert(52, "bravo")
        else:                        # present, but far away -> no #od2 match
            toks.append("bravo")
        units.append(_mk(f"doc{d}", " ".join(toks)))
    ex = IndriExecutor(units, mu=2500)
    start = time.time()
    res = ex.search("#od2(alpha bravo)", k=10)
    elapsed = time.time() - start
    assert res.error is None
    assert elapsed < 5.0, f"two-stage rescoring perf smoke took {elapsed:.2f}s (budget 5s)"
