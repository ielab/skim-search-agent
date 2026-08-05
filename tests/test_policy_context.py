"""AgentPolicy.build_messages: a total context-BUDGET window (ctx_chars), replacing the
old flat 2000-char per-observation cap that silently clipped whole-doc observations
(550-22k+ tokens) while sections (always <2000 chars) sailed through untouched — see
BUG note in policies.py. These tests pin the budget semantics directly."""
from agent_search.agent.loop import Step, Task
from agent_search.agent.policies import AgentPolicy
from agent_search.prompts import get_prompt_spec

PROMPT_PATH = get_prompt_spec("research_snip").path


def _policy(**kw):
    return AgentPolicy(generate=lambda m: "<answer></answer>", prompt_path=PROMPT_PATH, **kw)


def _step(i, obs_len, obs_char="x"):
    return Step(name="fetch", args={}, observation=obs_char * obs_len,
                raw_output=f"<tool_call>step{i}</tool_call>")


def _tool_responses(msgs):
    """The <tool_response> observation bodies, in message order."""
    return [m["content"] for m in msgs if m["role"] == "user"][1:]  # drop the task/user msg


# --- a big-but-under-budget observation is kept WHOLE, not clipped -----------

def test_full_observation_kept_when_under_budget():
    p = _policy(ctx_chars=450_000, max_history=40)
    obs = "y" * 20_000
    hist = [Step(name="fetch", args={}, observation=obs, raw_output="<tool_call>fetch</tool_call>")]
    msgs = p.build_messages(Task("t", "q"), hist)
    bodies = _tool_responses(msgs)
    assert len(bodies) == 1
    assert obs in bodies[0]                 # the full 20000 chars, untouched
    assert "...(truncated)" not in bodies[0]


# --- total kept content respects ctx_chars: oldest steps dropped when over ---

def test_ctx_chars_budget_drops_oldest_steps():
    # 5 steps of 1000 chars each (obs + raw combined ~1010); budget only fits the
    # ~2 most recent ones.
    budget = 2100
    p = _policy(ctx_chars=budget, max_history=40)
    hist = [_step(i, 1000) for i in range(5)]
    msgs = p.build_messages(Task("t", "q"), hist)
    bodies = _tool_responses(msgs)
    # oldest steps must have been dropped, not just the newest kept-and-truncated
    assert len(bodies) < len(hist)
    assert len(bodies) >= 1
    # every kept observation is intact (not truncated) since each individually fits
    assert all("...(truncated)" not in b for b in bodies)
    # the total char cost of everything kept (assistant + observation pairs) stays
    # under the configured budget
    total = 0
    for m in msgs[2:]:                      # skip system + initial user/task message
        total += len(m["content"])
    assert total <= budget + len(hist) * 200  # generous slack for <tool_response> wrapper text
    # and it must be the MOST RECENT steps that survived (highest step indices)
    kept_ids = [i for i in range(len(hist)) if f"step{i}" in "".join(m["content"] for m in msgs)]
    assert kept_ids == sorted(kept_ids)
    assert kept_ids[-1] == len(hist) - 1    # the newest step always survives


# --- a single observation that alone overflows the WHOLE budget gets truncated ---

def test_single_over_budget_observation_is_truncated_with_marker():
    budget = 5_000
    p = _policy(ctx_chars=budget, max_history=40)
    huge_obs = "z" * 50_000                 # far bigger than the whole budget (browsecomp-scale)
    hist = [_step(0, len(huge_obs), obs_char="z")]
    hist[0].observation = huge_obs
    msgs = p.build_messages(Task("t", "q"), hist)
    bodies = _tool_responses(msgs)
    assert len(bodies) == 1
    assert "...(truncated)" in bodies[0]
    assert len(bodies[0]) < len(huge_obs)   # it was cut down, not kept whole
    assert len(bodies[0]) <= budget + 64    # roughly bounded by the budget (+wrapper slack)


# --- kept pairs are emitted in chronological (oldest-kept -> newest) order ---

def test_kept_pairs_are_chronologically_ordered():
    p = _policy(ctx_chars=450_000, max_history=40)
    hist = [_step(i, 50) for i in range(4)]
    msgs = p.build_messages(Task("t", "q"), hist)
    roles = [m["role"] for m in msgs]
    # system, task-user, then alternating assistant/user per kept step
    assert roles[:2] == ["system", "user"]
    assert roles[2:] == ["assistant", "user"] * 4
    # assistant messages appear in the SAME order the steps were taken (step0..step3)
    assistant_bodies = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert assistant_bodies == [f"<tool_call>step{i}</tool_call>" for i in range(4)]


# --- context-overflow shrink-retry (vLLM 400 "maximum context length") -------

def test_propose_shrinks_window_on_context_overflow():
    """A generate() that 400s on over-long prompts triggers up to 3 shrink-retries
    (ctx_chars ×0.85 each); the eventual call succeeds with a smaller window and
    non-overflowing histories are built with the ORIGINAL budget untouched."""
    seen_lens = []

    def gen(msgs):
        total = sum(len(m["content"]) for m in msgs)
        seen_lens.append(total)
        if total > 60_000:
            raise RuntimeError(
                "Error code: 400 - This model's maximum context length is 131072 tokens...")
        return "<answer>ok</answer>"

    p = AgentPolicy(generate=gen, prompt_path=PROMPT_PATH, ctx_chars=80_000, max_history=40)
    hist = [_step(i, 10_000) for i in range(10)]
    out = p.propose(Task("t", "q"), hist)
    assert out == "<answer>ok</answer>"
    assert len(seen_lens) >= 2 and seen_lens[-1] <= 60_000   # shrank until it fit
    assert p.ctx_chars == 80_000                             # persistent budget unchanged
    seen_lens.clear()
    p.propose(Task("t", "q"), [_step(0, 1_000)])
    assert len(seen_lens) == 1                               # small history: no retry


def test_propose_reraises_non_overflow_errors():
    def gen(msgs):
        raise RuntimeError("connection reset")
    p = AgentPolicy(generate=gen, prompt_path=PROMPT_PATH, ctx_chars=80_000)
    import pytest
    with pytest.raises(RuntimeError, match="connection reset"):
        p.propose(Task("t", "q"), [_step(0, 100)])
