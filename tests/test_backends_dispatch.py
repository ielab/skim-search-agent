"""Provider dispatch in backends.make_generate: OpenAI models route to the OpenAI API with
the right param split (chat vs reasoning); Gemini models route to Gemini's OpenAI-compatible
endpoint; the trained backbone keeps its vLLM path. This is the "test anytime" path:
`--model gpt-4o-mini` (or `gemini-*`) plus a key runs the whole pipeline, no cluster."""
from types import SimpleNamespace

import agent_search.agent.backbone as B


def _fake_client(sink):
    def create(**kwargs):
        sink.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# --- provider / model classification ----------------------------------------

def test_openai_models_are_recognized():
    for m in ("gpt-4o-mini", "gpt-4o", "gpt-5-nano", "o1", "o3-mini", "chatgpt-4o-latest"):
        assert B.is_openai_model(m), m
    assert not B.is_openai_model("Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    assert not B.is_openai_model("Qwen/Qwen2.5-Coder-7B-Instruct")


def test_reasoning_models_are_recognized():
    for m in ("gpt-5-nano", "gpt-5", "o1", "o3-mini", "o4-mini"):
        assert B.is_reasoning_model(m), m
    for m in ("gpt-4o-mini", "gpt-4o"):
        assert not B.is_reasoning_model(m), m


def test_gemini_models_are_recognized():
    for m in ("gemini-2.5-flash-lite", "gemini-1.5-flash-8b", "gemini-2.0-flash", "Gemini-Pro"):
        assert B.is_gemini_model(m), m
    # must not collide with OpenAI or the trained backbone
    for m in ("gpt-4o-mini", "gpt-5-nano", "o1", "chatgpt-4o-latest",
              "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B", "Qwen/Qwen2.5-Coder-7B-Instruct"):
        assert not B.is_gemini_model(m), m
    for m in ("gpt-4o-mini", "gpt-5-nano", "chatgpt-4o-latest"):
        assert not B.is_openai_model("gemini-2.5-flash-lite")
        assert B.is_openai_model(m)


# --- make_generate: right client + param split ------------------------------

def test_chat_model_sends_sampling_params():
    sink = {}
    gen = B.make_generate("gpt-4o-mini", backend="vllm", client=_fake_client(sink))
    assert gen("hi") == "ok"
    assert sink["model"] == "gpt-4o-mini"
    assert "temperature" in sink and "max_tokens" in sink       # chat path
    assert "reasoning_effort" not in sink


def test_reasoning_model_uses_completion_budget_and_effort_no_sampling():
    sink = {}
    gen = B.make_generate("gpt-5-nano", backend="vllm", client=_fake_client(sink))
    assert gen("hi") == "ok"
    assert "max_completion_tokens" in sink and "reasoning_effort" in sink
    # reasoning models reject sampling params -> must NOT be sent
    assert "temperature" not in sink and "top_p" not in sink and "max_tokens" not in sink


def test_openai_routing_ignores_backend_flag():
    # an OpenAI model routes to the API even when backend='vllm' (no GPU needed).
    sink = {}
    B.make_generate("gpt-4o", backend="vllm", client=_fake_client(sink))("hi")
    assert sink["model"] == "gpt-4o"


def test_usage_is_recorded():
    B.reset_usage()
    B.make_generate("gpt-4o-mini", client=_fake_client({}))("hi")
    totals = B.usage_totals()
    assert totals["llm_calls"] == 1 and totals["prompt_tokens"] == 3


# --- Gemini: same "test anytime" routing, via the OpenAI-compatible endpoint ----------------

def test_gemini_routing_ignores_backend_flag():
    # a Gemini model routes to Gemini's endpoint even when backend='vllm' (no GPU needed) —
    # mirrors test_openai_routing_ignores_backend_flag above.
    sink = {}
    gen = B.make_generate("gemini-2.5-flash-lite", backend="vllm", client=_fake_client(sink))
    assert gen("hi") == "ok"
    assert sink["model"] == "gemini-2.5-flash-lite"
    assert "temperature" in sink and "max_tokens" in sink


def test_gemini_generate_builds_client_with_gemini_base_url_and_key(monkeypatch):
    # no `client` injected -> gemini_generate must construct one against Gemini's endpoint,
    # reading the key from GEMINI_API_KEY (never OPENAI_API_KEY).
    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, **_kw):
            seen["base_url"] = base_url
            seen["api_key"] = api_key

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setenv("GEMINI_API_KEY", "the-gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    B.gemini_generate(model="gemini-2.5-flash-lite")
    assert seen["base_url"] == B._GEMINI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert seen["api_key"] == "the-gemini-key"


def test_gemini_generate_sends_full_params_first():
    sink = {}
    gen = B.gemini_generate(model="gemini-2.5-flash-lite", client=_fake_client(sink))
    assert gen("hi") == "ok"
    # first attempt includes the full param set (seed/presence_penalty included)
    assert sink["seed"] == 42 and sink["presence_penalty"] == 1.1
    assert sink["temperature"] == 0.6 and sink["top_p"] == 0.95 and "stop" in sink


def test_gemini_generate_falls_back_to_minimal_params_when_rejected():
    """Reproduces the real Gemini behavior confirmed against the live API: the OpenAI-compat
    endpoint 400s on `seed` ("Unknown name 'seed'") and on `presence_penalty` on at least the
    flash-lite tier ("Penalty is not enabled for <model>"). gemini_generate must retry once with
    only the safe params (temperature/max_tokens/top_p/stop) instead of crashing the episode."""
    calls = []

    class _FakeBadRequest(Exception):
        # mirrors the real `openai.BadRequestError` shape (an HTTP 400 carries `status_code`) —
        # this is what `_is_param_error` keys on, distinguishing a genuine param rejection from
        # a transient 429/5xx that must NOT trigger the reduced-param fallback.
        status_code = 400

    def create(**kwargs):
        calls.append(kwargs)
        if "seed" in kwargs or "presence_penalty" in kwargs:
            raise _FakeBadRequest("400 Bad Request: Unknown name 'seed': Cannot find field.")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen = B.gemini_generate(model="gemini-2.5-flash-lite", client=client)
    assert gen("hi") == "ok"                     # does NOT raise — falls back and succeeds
    assert len(calls) == 2                       # one rejected attempt, one successful retry
    assert "seed" not in calls[1] and "presence_penalty" not in calls[1]
    assert "temperature" in calls[1] and "max_tokens" in calls[1] and "top_p" in calls[1]


def test_gemini_usage_is_recorded_after_fallback():
    B.reset_usage()

    def create(**kwargs):
        if "seed" in kwargs:
            raise Exception("400: unknown field seed")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=4))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    B.gemini_generate(model="gemini-2.5-flash-lite", client=client)("hi")
    totals = B.usage_totals()
    assert totals["llm_calls"] == 1 and totals["prompt_tokens"] == 5


# --- api_key passthrough (live-demo server: per-request user keys, not process env) -------------

def _capture_openai_ctor(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, **_kw):
            seen["base_url"] = base_url
            seen["api_key"] = api_key

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return seen


def test_make_generate_forwards_explicit_api_key_over_env(monkeypatch):
    seen = _capture_openai_ctor(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    B.make_generate("gpt-4o-mini", api_key="user-key")
    assert seen["api_key"] == "user-key"
    assert seen["base_url"] == B._OPENAI_BASE_URL


def test_make_generate_api_key_defaults_to_env(monkeypatch):
    seen = _capture_openai_ctor(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    B.make_generate("gpt-4o-mini")
    assert seen["api_key"] == "env-key"


# --- _with_retries: the transient-failure retry helper -----------------------------------------
# class names are matched by NAME (see backends._is_transient_error), not isinstance, so a plain
# local class named e.g. `RateLimitError` exercises the same path a real `openai.RateLimitError`
# would without needing network/the real openai exception hierarchy.

def test_with_retries_retries_rate_limit_twice_then_succeeds():
    class RateLimitError(Exception):
        pass

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429 too many requests")
        return "ok"

    assert B._with_retries(flaky, attempts=5, base=0.001) == "ok"
    assert calls["n"] == 3                        # two failures, then the succeeding 3rd call


def test_with_retries_does_not_retry_bad_request_error():
    class BadRequestError(Exception):
        pass

    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise BadRequestError("400 invalid request")

    try:
        B._with_retries(bad, attempts=5, base=0.001)
        assert False, "expected BadRequestError to propagate"
    except BadRequestError:
        pass
    assert calls["n"] == 1                        # never retried — re-raised on the first attempt


def test_with_retries_exhausts_attempts_and_raises_last_error():
    class RateLimitError(Exception):
        pass

    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise RateLimitError("still limited")

    try:
        B._with_retries(always_flaky, attempts=3, base=0.001)
        assert False, "expected RateLimitError to propagate after exhausting attempts"
    except RateLimitError:
        pass
    assert calls["n"] == 3


def test_with_retries_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0.001")

    class RateLimitError(Exception):
        pass

    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise RateLimitError("still limited")

    try:
        B._with_retries(always_flaky)              # attempts/base read from env, not passed in
        assert False, "expected RateLimitError to propagate after exhausting attempts"
    except RateLimitError:
        pass
    assert calls["n"] == 2                         # LLM_RETRY_ATTEMPTS=2, not the default 5


def test_openai_compat_generate_is_wired_through_with_retries():
    """Integration: `openai_compat_generate`'s own `client.chat.completions.create` call goes
    through `_with_retries`, not a bare call — a transient failure on the first attempt(s) must
    not crash the episode."""
    class RateLimitError(Exception):
        pass

    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitError("429")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen = B.openai_compat_generate(model="m", client=client)
    assert gen("hi") == "ok"
    assert calls["n"] == 2


def test_openai_client_gets_explicit_timeout_and_no_sdk_retries(monkeypatch):
    """Every constructed `OpenAI(...)` client must set `timeout` (env-overridable via
    LLM_TIMEOUT_S) and `max_retries=0` — our own `_with_retries` does the retrying, not the
    SDK's own silent retry loop."""
    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, timeout, max_retries):
            seen["timeout"] = timeout
            seen["max_retries"] = max_retries

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setenv("LLM_TIMEOUT_S", "42")
    B.make_generate("gpt-4o-mini")
    assert seen["timeout"] == 42.0
    assert seen["max_retries"] == 0
