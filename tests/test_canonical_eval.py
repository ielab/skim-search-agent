"""Canonical per-dataset eval protocol tests.

Covers the switch from an invented "cover-EM" (substring containment, which no paper uses) to
each dataset's published metric:

  HotpotQA / 2WikiMultihopQA : Answer EM + F1, SQuAD-style normalization (agent_search.evaluation.metrics).
  MuSiQue                    : Answer F1 (already covered) + SUPPORT F1 over gold-doc-id sets
                                (agent_search.evaluation.metrics.support_f1).
  BrowseComp-Plus            : LLM-as-judge (agent_search.evaluation.llm_judge.judge_answer), mocked here so
                                the suite never makes a real network call.

Also checks that `score_answer` extracts the `<answer>...</answer>` span (the short span the
agent is now instructed to emit) before scoring.
"""
from agent_search.evaluation.metrics import answer_em, answer_f1, support_f1
from agent_search.evaluation.doc_scoring import extract_answer_span, score_answer
from agent_search.evaluation.llm_judge import judge_answer, judge_answer_detail


# --- HotpotQA / 2WikiMultihopQA canonical EM/F1 ------------------------------

def test_em_case_insensitive():
    # "Yes" vs "yes" -> normalized equal -> EM 1 (case is not a real difference).
    assert answer_em("Yes", "yes") == 1.0


def test_em_article_stripped():
    # Leading "the" must not break EM: the official normalize_answer() strips a/an/the.
    assert answer_em("the North Atlantic Conference", "North Atlantic Conference") == 1.0


def test_em_wrong_answer_is_zero():
    assert answer_em("Treaty of Paris", "Treaty of Guadalupe Hidalgo") == 0.0


def test_em_empty_gold_is_zero():
    assert answer_em("anything", "") == 0.0


def test_f1_full_overlap_different_order():
    # F1 is a bag-of-tokens metric (order-free): same 3 tokens, reordered and with an article
    # stripped -> precision == recall == 1.0 -> F1 == 1.0.
    assert answer_f1("the Conference, North Atlantic", "North Atlantic Conference") == 1.0


def test_f1_no_overlap_is_zero():
    assert answer_f1("Paris", "Guadalupe Hidalgo") == 0.0


def test_f1_partial_token_overlap_is_between_zero_and_one():
    f1 = answer_f1("Giuseppe Verdi was a composer", "Giuseppe Verdi")
    assert 0.0 < f1 < 1.0


# --- MuSiQue SUPPORT F1 -------------------------------------------------------

def test_support_f1_perfect_match():
    assert support_f1({"d1", "d2"}, {"d1", "d2"}) == 1.0


def test_support_f1_partial_precision_and_recall():
    # surfaced = {d1, d2, d3}, gold = {d1, d2} -> precision 2/3, recall 2/2=1
    # f1 = 2*P*R/(P+R) = 2*(2/3)*1 / (2/3 + 1) = (4/3) / (5/3) = 0.8
    f1 = support_f1(["d1", "d2", "d3"], {"d1", "d2"})
    assert abs(f1 - 0.8) < 1e-9


def test_support_f1_no_overlap_is_zero():
    assert support_f1({"dX"}, {"d1"}) == 0.0


def test_support_f1_empty_gold_or_empty_surfaced_is_zero():
    assert support_f1({"d1"}, set()) == 0.0
    assert support_f1(set(), {"d1"}) == 0.0


def test_score_answer_emits_support_f1_when_gold_doc_ids_given():
    obs = ["The Treaty of Guadalupe Hidalgo ended the war in 1848."]
    r = score_answer("Treaty of Guadalupe Hidalgo", "Treaty of Guadalupe Hidalgo", obs,
                     surfaced_docs=["d1", "d3"], gold_doc_ids={"d1", "d2"})
    assert "support_f1" in r
    # precision 1/2, recall 1/2 -> f1 0.5
    assert abs(r["support_f1"] - 0.5) < 1e-9


def test_score_answer_omits_support_f1_without_gold_doc_ids():
    r = score_answer("Treaty of Guadalupe Hidalgo", "Treaty of Guadalupe Hidalgo",
                     ["Treaty of Guadalupe Hidalgo ended the war."])
    assert "support_f1" not in r


# --- <answer> extraction ------------------------------------------------------

def test_extract_answer_span_with_tags():
    assert extract_answer_span("I think it is <answer>1848</answer>") == "1848"


def test_extract_answer_span_no_tags_returns_raw_stripped():
    assert extract_answer_span("  1848  ") == "1848"


def test_extract_answer_span_uses_last_tag_if_multiple():
    text = "<answer>draft one</answer> ... more thinking ... <answer>final</answer>"
    assert extract_answer_span(text) == "final"


def test_extract_answer_span_empty():
    assert extract_answer_span("") == ""


def test_score_answer_extracts_tagged_span_before_scoring():
    # The raw final answer still carries its tag and surrounding prose; score_answer must
    # extract "1848" and score THAT against gold, not the whole sentence containing it (which
    # would fail strict EM even though the tagged short answer is exactly right).
    raw = "Based on the evidence, the treaty was signed in <answer>1848</answer>."
    obs = ["The treaty was signed in 1848 in Guadalupe Hidalgo."]
    r = score_answer(raw, "1848", obs)
    assert r["answer_em"] == 1.0
    assert r["grounded"] is True


def test_score_answer_falls_back_to_raw_when_no_tags():
    obs = ["The treaty was signed in 1848."]
    r = score_answer("1848", "1848", obs)
    assert r["answer_em"] == 1.0


# --- BrowseComp-Plus LLM-as-judge ---------------------------------------------

def test_judge_answer_model_param_and_env_default_exist():
    # Configurability contract: `model` is a keyword param, and a module-level default exists
    # (overridable via LLM_JUDGE_MODEL) — without actually hitting the network.
    import inspect
    import agent_search.evaluation.llm_judge as llm_judge
    sig = inspect.signature(judge_answer)
    assert "model" in sig.parameters
    assert hasattr(llm_judge, "DEFAULT_JUDGE_MODEL")
    assert llm_judge.DEFAULT_JUDGE_MODEL  # non-empty default model name


def test_judge_answer_detail_still_available_for_batch_grading():
    # judge_run_dir relies on the detailed dict form (judge_correct/judge_extracted/judge_reasoning).
    def fake_generate(prompt: str) -> str:
        return '{"extracted_final_answer": "x", "reasoning": "r", "correct": "yes", "confidence": 50}'

    detail = judge_answer_detail("Q", "gold", "some response text", fake_generate)
    assert detail["judge_correct"] is True
    assert "judge_extracted" in detail and "judge_reasoning" in detail
