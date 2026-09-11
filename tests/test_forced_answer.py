"""Unit tests for the shared prefill-elicitation mechanism (agent_search/agent/forced_answer.py),
used by both agent_search/agent/loop.py's inline elicitation and the offline backfill script in
scripts/force_answer_backfill.py so the two share one implementation. No GPU/model/network
needed; the client is a stub recording call kwargs (same pattern as
tests/test_force_answer_backfill.py's `_fake_client`).
"""
from __future__ import annotations

from types import SimpleNamespace

from agent_search.agent.forced_answer import (
    DEFAULT_FALLBACK_MAX_TOKENS,
    DEFAULT_PREFILL_MAX_TOKENS,
    FALLBACK_MSG,
    FORCE_MSG,
    call_plain_ask,
    call_prefill,
    elicit_final_answer,
    prefill_messages_for,
)


def _fake_client(contents):
    calls = []

    def create(**kwargs):
        i = len(calls)
        calls.append(kwargs)
        text = contents[min(i, len(contents) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._calls = calls
    return client


# --- prefill_messages_for ------------------------------------------------------------------------

def test_prefill_messages_for_appends_open_answer_tag():
    base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    prefilled = prefill_messages_for(base)
    assert prefilled[:-1] == base
    assert prefilled[-1] == {"role": "assistant", "content": "<answer>"}
    assert base == [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]  # no mutation


# --- call_prefill: request shape -----------------------------------------------------------------

def test_call_prefill_request_shape():
    client = _fake_client(["Paris"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    text = call_prefill(client, "my-model", messages, max_tokens=200)
    assert text == "Paris"
    kwargs = client._calls[0]
    assert kwargs["model"] == "my-model"
    assert kwargs["messages"][-1] == {"role": "assistant", "content": "<answer>"}
    assert kwargs["messages"][:-1] == messages
    assert kwargs["stop"] == ["</answer>"]
    assert kwargs["extra_body"] == {"add_generation_prompt": False, "continue_final_message": True}
    assert kwargs["max_tokens"] == 200


def test_call_prefill_uses_default_tuning_knobs():
    client = _fake_client(["Paris"])
    call_prefill(client, "m", [{"role": "user", "content": "Q"}])
    kwargs = client._calls[0]
    assert kwargs["max_tokens"] == DEFAULT_PREFILL_MAX_TOKENS


# --- call_plain_ask: request shape (no prefill flags) ---------------------------------------------

def test_call_plain_ask_request_shape_has_no_prefill_flags():
    client = _fake_client(["<answer>Paris</answer>"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    text = call_plain_ask(client, "my-model", messages)
    assert text == "<answer>Paris</answer>"
    kwargs = client._calls[0]
    assert "extra_body" not in kwargs
    assert "stop" not in kwargs
    assert kwargs["messages"][-1]["role"] == "user"
    assert FALLBACK_MSG in kwargs["messages"][-1]["content"]
    assert kwargs["max_tokens"] == DEFAULT_FALLBACK_MAX_TOKENS


# --- elicit_final_answer: primary/fallback + the returned method_tag ------------------------------

def test_elicit_final_answer_uses_prefill_when_non_empty_single_call():
    client = _fake_client(["Paris"])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q?"}]
    answer, tag, raw = elicit_final_answer(messages, client, "my-model")
    assert (answer, tag, raw) == ("Paris", "prefill", "Paris")
    assert len(client._calls) == 1                # single call — no fallback needed


def test_elicit_final_answer_strips_trailing_answer_tag_if_echoed():
    # belt-and-braces: some server config might echo the stop string anyway.
    client = _fake_client(["Paris</answer>"])
    messages = [{"role": "user", "content": "Q?"}]
    answer, tag, _raw = elicit_final_answer(messages, client, "my-model")
    assert answer == "Paris" and tag == "prefill"


def test_elicit_final_answer_falls_back_once_when_prefill_is_empty():
    client = _fake_client(["   ", "<answer>Paris</answer>"])
    messages = [{"role": "user", "content": "Q?"}]
    answer, tag, raw = elicit_final_answer(messages, client, "my-model")
    assert answer == "Paris"
    assert tag == "plain_ask_fallback"
    assert raw == "<answer>Paris</answer>"
    assert len(client._calls) == 2
    assert "extra_body" not in client._calls[1]     # the SECOND call is the plain-ask shape


def test_elicit_final_answer_returns_empty_tag_when_both_calls_fail():
    client = _fake_client(["", "no tags here, sorry"])
    messages = [{"role": "user", "content": "Q?"}]
    answer, tag, _raw = elicit_final_answer(messages, client, "my-model")
    assert answer == ""
    assert tag == "empty"


def test_elicit_final_answer_uses_custom_extract_fn():
    """The fallback extraction is pluggable (`extract_fn`) — loop.py's inline caller and
    force_answer_backfill.py's `elicit_answer` wrapper both pass loop._extract_answer explicitly;
    the default (None) lazily imports the same function, exercised by the test above."""
    client = _fake_client(["", "<answer>whatever the real extractor would find</answer>"])
    messages = [{"role": "user", "content": "Q?"}]
    answer, tag, raw = elicit_final_answer(
        messages, client, "my-model", extract_fn=lambda _text: "custom-extracted")
    assert answer == "custom-extracted"
    assert tag == "plain_ask_fallback"
    assert raw == "<answer>whatever the real extractor would find</answer>"


# --- constants sanity (the inline loop.py caller and the offline script both key off these) -------

def test_force_msg_is_a_tool_response_shaped_instruction():
    assert "STEP BUDGET REACHED" in FORCE_MSG and "<answer>" in FORCE_MSG


# --- usage accounting: the forcing call is the LARGEST prompt of an episode, must not be
# missing from cost accounting (agent_search.agent.backbone' thread-local usage ledger) ---------

def _fake_client_with_usage(text, prompt_tokens=11, completion_tokens=3):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_call_prefill_records_usage_on_shared_ledger():
    import agent_search.agent.backbone as B
    B.reset_usage()
    client = _fake_client_with_usage("Paris", prompt_tokens=11, completion_tokens=3)
    call_prefill(client, "m", [{"role": "user", "content": "Q"}])
    totals = B.usage_totals()
    assert totals == {"llm_calls": 1, "prompt_tokens": 11, "completion_tokens": 3,
                      "cached_input_tokens": 0, "reasoning_tokens": 0}


def test_call_plain_ask_records_usage_on_shared_ledger():
    import agent_search.agent.backbone as B
    B.reset_usage()
    client = _fake_client_with_usage("<answer>Paris</answer>", prompt_tokens=20, completion_tokens=5)
    call_plain_ask(client, "m", [{"role": "user", "content": "Q"}])
    totals = B.usage_totals()
    assert totals["llm_calls"] == 1
    assert totals["prompt_tokens"] == 20 and totals["completion_tokens"] == 5


def test_call_prefill_tolerates_missing_usage():
    """A stub client with no `usage` attribute (the pre-existing `_fake_client` in this file)
    must not crash — usage recording degrades to a no-op, not an AttributeError."""
    client = _fake_client(["Paris"])
    assert call_prefill(client, "m", [{"role": "user", "content": "Q"}]) == "Paris"


# --- retries: call_prefill/call_plain_ask route through backends._with_retries -------------------

def test_call_prefill_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0.001")   # keep the backoff sleep negligible in tests

    class RateLimitError(Exception):
        pass

    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitError("429")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert call_prefill(client, "m", [{"role": "user", "content": "Q"}]) == "Paris"
    assert calls["n"] == 2


def test_call_plain_ask_does_not_retry_bad_request_error():
    class BadRequestError(Exception):
        pass

    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        raise BadRequestError("400 invalid request")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    try:
        call_plain_ask(client, "m", [{"role": "user", "content": "Q"}])
        assert False, "expected BadRequestError to propagate"
    except BadRequestError:
        pass
    assert calls["n"] == 1
