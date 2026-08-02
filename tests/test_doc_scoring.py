"""Deep-research doc-QA scoring: grounded EM/F1 + gold-doc coverage (evaluation.doc_scoring).

The answer counts only when it also appears verbatim in the actual tool evidence (grounding
gate) — a right answer from the model's memory does not score."""
from evaluation.doc_scoring import (answer_in_evidence, gold_doc_coverage, score_answer)


# --- the grounding gate -----------------------------------------------------

def test_answer_in_evidence_exact_span():
    obs = ["The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."]
    assert answer_in_evidence("Treaty of Guadalupe Hidalgo", obs) is True
    assert answer_in_evidence("Treaty of Paris", obs) is False


def test_answer_in_evidence_ignores_case_and_punctuation():
    obs = ["It was signed in the U.S. capital."]
    assert answer_in_evidence("us capital", obs) is True          # U.S. vs us


def test_answer_in_evidence_no_partial_token_match():
    obs = ["He was from Yorkshire."]
    assert answer_in_evidence("York", obs) is False               # York != Yorkshire


def test_answer_in_evidence_empty_answer():
    assert answer_in_evidence("", ["anything"]) is False


# --- gold-doc coverage ------------------------------------------------------

def test_gold_doc_coverage_fraction():
    assert gold_doc_coverage({"d1", "d2", "d3"}, {"d1", "d2"}) == 1.0
    assert gold_doc_coverage({"d1"}, {"d1", "d2"}) == 0.5
    assert gold_doc_coverage({"dX"}, {"d1"}) == 0.0
    assert gold_doc_coverage({"d1"}, set()) == 0.0                # empty gold -> 0


# --- score_answer: grounded EM/F1 -------------------------------------------

def test_score_answer_grounded_hits():
    obs = ["The Treaty of Guadalupe Hidalgo ended the war in 1848."]
    r = score_answer("Treaty of Guadalupe Hidalgo", "Treaty of Guadalupe Hidalgo", obs)
    assert r["grounded"] is True
    assert r["answer_em"] == 1.0 and r["grounded_em"] == 1.0


def test_score_answer_correct_but_ungrounded_is_zeroed():
    # right answer, but NOT present in any tool observation -> grounded_em zeroed.
    r = score_answer("Treaty of Guadalupe Hidalgo", "Treaty of Guadalupe Hidalgo",
                     ["some unrelated evidence about Paris"])
    assert r["grounded"] is False
    assert r["answer_em"] == 1.0            # raw EM still 1 (for diagnosis)
    assert r["grounded_em"] == 0.0         # but the grounded metric is 0


def test_score_answer_wrong_answer():
    r = score_answer("Treaty of Paris", "Treaty of Guadalupe Hidalgo",
                     ["The Treaty of Paris ended a different war."])
    assert r["answer_em"] == 0.0 and r["grounded_em"] == 0.0
