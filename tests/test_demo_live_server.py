"""The live-demo SSE server (demo/server.py): /api/run streams step -> done events per
strategy (fake generate injected — NO real API calls), errors surface as error events (never a
hang), the api_key reaches make_generate and is never echoed back."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from demo import server  # noqa: E402


def _sse_events(text):
    return [json.loads(ln[len("data: "):])
            for ln in text.splitlines() if ln.startswith("data: ")]


def _scripted_generate(outputs):
    def factory(model, *, backend="vllm", api_key=None, **kw):
        factory.seen_api_key = api_key
        it = iter(outputs)

        def generate(messages):
            return next(it)
        return generate
    return factory


SIEVE_SCRIPT = [
    '<tool_call>{"name":"search_s","arguments":{"query":"lady shri ram college"}}</tool_call>',
    "<answer>Lady Shri Ram College for Women</answer>",
]


def _post(client, **overrides):
    body = {"question": "Which college?", "api_key": "sk-test", "model": "gpt-4o-mini",
            "strategies": ["sieve"], **overrides}
    return client.post("/api/run", json=body)


def test_run_streams_step_then_done(monkeypatch):
    monkeypatch.setattr(server, "_make_generate", _scripted_generate(SIEVE_SCRIPT))
    r = _post(TestClient(server.app))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    kinds = [e["event"] for e in events]
    assert "step" in kinds and kinds[-1] == "done"
    done = events[-1]
    assert done["strategy"] == "sieve"
    assert done["answer"] == "Lady Shri Ram College for Women"


def test_search_step_carries_parsed_hits_and_usage(monkeypatch):
    monkeypatch.setattr(server, "_make_generate", _scripted_generate(SIEVE_SCRIPT))
    events = _sse_events(_post(TestClient(server.app)).text)
    search_steps = [e for e in events
                    if e["event"] == "step" and e["step"]["type"] == "search"]
    assert search_steps, "no parsed search step in the stream"
    s = search_steps[0]
    assert "hits" in s["step"] and "query" in s["step"]
    assert set(s["usage"]) >= {"prompt_tokens", "completion_tokens", "cost_usd", "steps"}


def test_api_key_reaches_make_generate_and_is_never_echoed(monkeypatch):
    factory = _scripted_generate(SIEVE_SCRIPT)
    monkeypatch.setattr(server, "_make_generate", factory)
    r = _post(TestClient(server.app), api_key="sk-supersecret")
    assert factory.seen_api_key == "sk-supersecret"
    assert "sk-supersecret" not in r.text


def test_compare_mode_streams_both_strategies(monkeypatch):
    def factory(model, *, backend="vllm", api_key=None, **kw):
        def generate(messages):
            return "<answer>done</answer>"
        return generate
    monkeypatch.setattr(server, "_make_generate", factory)
    events = _sse_events(_post(TestClient(server.app),
                               strategies=["sieve", "search_visit"]).text)
    dones = {e["strategy"] for e in events if e["event"] == "done"}
    assert dones == {"sieve", "search_visit"}


def test_generate_failure_becomes_error_event_not_hang(monkeypatch):
    def factory(model, *, backend="vllm", api_key=None, **kw):
        def generate(messages):
            raise RuntimeError("Incorrect API key provided")
        return generate
    monkeypatch.setattr(server, "_make_generate", factory)
    events = _sse_events(_post(TestClient(server.app)).text)
    errs = [e for e in events if e["event"] == "error"]
    assert errs and "Incorrect API key" in errs[0]["message"]


def test_unknown_strategy_rejected():
    r = _post(TestClient(server.app), strategies=["quantum"])
    assert r.status_code == 422


def test_cost_table():
    assert server._cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    # unknown model: None, never a confidently wrong $0.00 — the meter shows tokens only
    assert server._cost("unknown-model", 1000, 1000) is None


def test_error_event_redacts_key_shaped_tokens(monkeypatch):
    """Provider auth errors quote a masked copy of the key — the error event must redact it."""
    def factory(model, *, backend="vllm", api_key=None, **kw):
        def generate(messages):
            raise RuntimeError("Incorrect API key provided: sk-supers***cret")
        return generate
    monkeypatch.setattr(server, "_make_generate", factory)
    r = _post(TestClient(server.app), api_key="sk-supersecret")
    assert "sk-supers" not in r.text
    events = _sse_events(r.text)
    errs = [e for e in events if e["event"] == "error"]
    assert errs and "sk-***" in errs[0]["message"]


def test_user_defined_model_accepted_and_endpoint_pinned(monkeypatch):
    """Any model name is allowed (it runs on the user's own key) — but the endpoint must be
    PINNED to OpenAI: an unrecognised name once fell through make_generate's backend="api"
    branch to its localhost default, which would have sent the key to whatever listens there."""
    factory = _scripted_generate(SIEVE_SCRIPT)
    seen = {}

    def spy(model, **kw):
        seen.update(kw, model=model)
        return factory(model, **{k: v for k, v in kw.items() if k in ("backend", "api_key")})
    monkeypatch.setattr(server, "_make_generate", spy)
    r = _post(TestClient(server.app), model="my-custom-ft-9000")
    assert r.status_code == 200
    assert seen["model"] == "my-custom-ft-9000"
    assert seen["api_base"] == "https://api.openai.com/v1"


def test_malformed_model_name_rejected():
    r = _post(TestClient(server.app), model="bad name with spaces!")
    assert r.status_code == 422


def test_cost_discounts_cached_input():
    """Cached input is a SUBSET of prompt_tokens billed at the cached rate; the remainder
    bills at the full input rate. 1M uncached @ $0.15 + 1M cached @ $0.075 + 1M out @ $0.60."""
    assert server._cost("gpt-4o-mini", 2_000_000, 1_000_000, 1_000_000) == pytest.approx(0.825)
    # fully cached input is exactly half the uncached price
    full = server._cost("gpt-4o-mini", 1_000_000, 0, 0)
    cached = server._cost("gpt-4o-mini", 1_000_000, 0, 1_000_000)
    assert cached == pytest.approx(full / 2)
    # a provider reporting more cached than prompt tokens must never produce a negative bill
    assert server._cost("gpt-4o-mini", 1000, 0, 5000) >= 0


def test_usage_snapshot_reports_cached_tokens(monkeypatch):
    monkeypatch.setattr(server, "_make_generate", _scripted_generate(SIEVE_SCRIPT))
    events = _sse_events(_post(TestClient(server.app)).text)
    steps = [e for e in events if e["event"] == "step"]
    assert steps and "cached_input_tokens" in steps[0]["usage"]
