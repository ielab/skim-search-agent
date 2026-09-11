"""The LLM-as-judge (BrowseComp-Plus protocol), offline, with a stub grader."""
from types import SimpleNamespace

import pytest

from agent_search.evaluation.llm_judge import BCP_JUDGE_PROMPT, judge_answer_detail, make_judge


def _stub(verdict: str):
    """A fake judge that asserts the BCP prompt was used, then returns a fixed verdict."""
    def gen(prompt: str) -> str:
        assert "[question]" in prompt and "[correct_answer]" in prompt      # the real BCP prompt
        return ('{"extracted_final_answer": "x", "reasoning": "r", '
                f'"correct": "{verdict}", "confidence": 100}}')
    return gen


def test_exact_match_short_circuits_without_a_call():
    calls = []
    v = judge_answer_detail("q", "Mike Medavoy", "  mike medavoy.  ", lambda p: calls.append(1) or "{}")
    assert v["judge_correct"] is True and not calls          # normalized exact match, no LLM call


def test_verbose_but_correct_is_judged_yes():
    v = judge_answer_detail("Who founded the company?", "Mike Medavoy",
                     "Mike Medavoy, Arthur Krim, and others founded the company.", _stub("yes"))
    assert v["judge_correct"] is True                        # the exact-match scorer would give 0 here


def test_wrong_answer_is_judged_no():
    v = judge_answer_detail("What state?", "Tamaulipas", "Nuevo Laredo, Mexico", _stub("no"))
    assert v["judge_correct"] is False


def test_empty_answer_is_no_without_a_call():
    calls = []
    v = judge_answer_detail("q", "gold", "", lambda p: calls.append(1) or "{}")
    assert v["judge_correct"] is False and not calls


def test_malformed_json_is_forgiving():
    v = judge_answer_detail("q", "g", "some answer", lambda p: 'noise {"correct": "yes"} trailing')
    assert v["judge_correct"] is True


def test_prompt_is_verbatim_browsecomp():
    # guard the exact wording — paraphrasing it silently changes the benchmark comparison
    assert "extracted_final_answer" in BCP_JUDGE_PROMPT
    assert "within a small margin of error for numerical problems" in BCP_JUDGE_PROMPT


# --- make_judge: auto-routes by model name, exactly like backends.make_generate ------------

def test_make_judge_routes_gemini_model_to_gemini_endpoint(monkeypatch):
    """A `gemini-*` --judge-model must build its client against Gemini's OpenAI-compatible
    endpoint with GEMINI_API_KEY — not OPENAI_API_KEY, and with no `api_base` required (mirrors
    the OpenAI judge branch, which also needs no api_base)."""
    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            seen["base_url"] = base_url
            seen["api_key"] = api_key

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setenv("GEMINI_API_KEY", "the-gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    make_judge("gemini-2.5-flash-lite")
    assert seen["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert seen["api_key"] == "the-gemini-key"


def test_make_judge_routes_openai_model_to_openai_endpoint(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            seen["base_url"] = base_url
            seen["api_key"] = api_key

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "the-openai-key")
    make_judge("gpt-4o-mini")
    assert seen["base_url"] == "https://api.openai.com/v1"
    assert seen["api_key"] == "the-openai-key"


def test_make_judge_non_provider_model_without_api_base_raises():
    # a served (vLLM) judge model with no api_base can't be routed anywhere -> a clear error,
    # not a silent fallback to the wrong endpoint.
    with pytest.raises(ValueError, match="needs a served endpoint"):
        make_judge("Qwen/Qwen2.5-Coder-7B-Instruct")


def test_make_judge_gemini_generate_uses_injected_client():
    # the generate() callable itself is provider-agnostic once a client is built/injected —
    # this exercises the JSON-mode call path with a Gemini-shaped client.
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"correct": "yes"}'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen = make_judge("gemini-2.5-flash-lite", client=client)
    assert gen("judge this") == '{"correct": "yes"}'
    assert seen["model"] == "gemini-2.5-flash-lite" and seen["temperature"] == 0.0
