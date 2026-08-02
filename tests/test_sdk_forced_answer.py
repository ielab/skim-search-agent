"""Offline coverage for run_episode_sdk's MaxTurnsExceeded ask-with-retry forcing sequence (see
agent_search/agent/sdk_driver.py): a hosted API model has no vLLM-style assistant-prefill
affordance (that mechanism, agent_search/agent/forced_answer.py, is loop-driver-only), so the SDK
driver instead re-asks up to 2 more times ("Output ONLY <answer>...</answer>") after its
pre-existing single closer call, tagging `SdkTrajectory.elicitation`.

No network / real Agents SDK Runner is exercised: `Runner.run_sync` is monkeypatched so this stays
CPU-only and deterministic, mirroring tests/test_sdk_new_tools.py's FakeWS pattern.
"""
from __future__ import annotations

from types import SimpleNamespace

from agents.exceptions import MaxTurnsExceeded

import agent_search.agent.sdk_driver as sdk_driver
from agent_search.agent.sdk_driver import run_episode_sdk


class FakeWS:
    tools = ()
    seen = set()

    def run(self, name, args):  # pragma: no cover - no tools registered in these tests
        return ""


class _FakeResult:
    def __init__(self, final_output):
        self.final_output = final_output
        self.context_wrapper = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  input_tokens_details=None, output_tokens_details=None))
        self.raw_responses = []


def _install_fake_runner(monkeypatch, outputs):
    """The FIRST call (the main episode run, agent.name == 'research') always raises
    MaxTurnsExceeded; every SUBSEQUENT call pops the next canned output off `outputs` (in order:
    the pre-existing closer call, then up to 2 retries)."""
    calls = []

    def fake_run_sync(agent, _input, max_turns=1):
        calls.append(agent.name)
        if agent.name == "research":
            raise MaxTurnsExceeded("too many turns")
        return _FakeResult(outputs.pop(0))

    monkeypatch.setattr(sdk_driver.Runner, "run_sync", fake_run_sync)
    return calls


def test_closer_call_success_tags_ask_retry_inline_no_extra_retries(monkeypatch):
    """The pre-existing single closer call (unchanged behavior) already produces a usable answer
    — a PURE ADDITION means this must NOT trigger any of the new retry calls."""
    calls = _install_fake_runner(monkeypatch, ["Paris is the capital of France."])
    traj = run_episode_sdk(FakeWS(), "What is the capital of France?", model="gpt-4o-mini", max_turns=3)
    assert traj.final_answer == "Paris is the capital of France."
    assert traj.elicitation == "ask_retry_inline"
    assert calls == ["research", "research-final"]      # no retry agents invoked


def test_retries_fire_when_closer_call_is_empty_and_first_retry_wins(monkeypatch):
    calls = _install_fake_runner(monkeypatch, ["", "<answer>Paris</answer>"])
    traj = run_episode_sdk(FakeWS(), "What is the capital of France?", model="gpt-4o-mini", max_turns=3)
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "ask_retry_inline"
    assert calls == ["research", "research-final", "research-final-retry"]


def test_second_retry_wins_when_first_retry_is_also_empty(monkeypatch):
    calls = _install_fake_runner(monkeypatch, ["", "", "<answer>Paris</answer>"])
    traj = run_episode_sdk(FakeWS(), "What is the capital of France?", model="gpt-4o-mini", max_turns=3)
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "ask_retry_inline"
    assert calls == ["research", "research-final", "research-final-retry", "research-final-retry"]


def test_all_attempts_empty_tags_ask_retry_failed_and_keeps_placeholder(monkeypatch):
    calls = _install_fake_runner(monkeypatch, ["", "", ""])
    traj = run_episode_sdk(FakeWS(), "What is the capital of France?", model="gpt-4o-mini", max_turns=3)
    assert traj.final_answer == "(max turns exceeded — no final answer)"
    assert traj.elicitation == "ask_retry_failed"
    # exactly 2 retry attempts beyond the closer call — never crashes, never loops forever.
    assert calls == ["research", "research-final", "research-final-retry", "research-final-retry"]


def test_organic_completion_has_no_elicitation_tag(monkeypatch):
    """max_turns never exceeded -> elicitation stays None (this row is not a forced answer at all,
    same 'absent for pre-change rows' semantics as loop.py's Trajectory.elicitation)."""
    def fake_run_sync(agent, _input, max_turns=1):
        return _FakeResult("Paris")

    monkeypatch.setattr(sdk_driver.Runner, "run_sync", fake_run_sync)
    traj = run_episode_sdk(FakeWS(), "What is the capital of France?", model="gpt-4o-mini", max_turns=3)
    assert traj.final_answer == "Paris"
    assert traj.elicitation is None
