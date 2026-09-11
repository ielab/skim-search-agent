"""AgentPolicy.build_messages: a total context BUDGET measured in TOKENS (ctx_tokens).

There is no character cap anywhere in the prompt assembly: whole (assistant, observation)
pairs are kept newest-first while their token cost fits the budget, older pairs are dropped,
and only a single observation that alone overflows the whole budget is cut, by tokens.
Budgets in these tests are derived from the library's own ruler (`count_tokens`) so they hold
whether tiktoken is installed (o200k_base) or the whitespace fallback is in use."""
import pytest

from agent_search.agent.loop import Step, Task
from agent_search.agent.policies import AgentPolicy, default_ctx_tokens
from agent_search.tokens import count_tokens
from agent_search.strategies import CONDITIONS

SYSTEM = CONDITIONS["research_snip"].render()


def _policy(**kw):
    return AgentPolicy(generate=lambda m: "<answer></answer>", system=SYSTEM, **kw)


def _obs(n_words: int, word: str = "alpha") -> str:
    return " ".join(f"{word}{i}" for i in range(n_words))


def _step(i, n_words, word="alpha"):
    return Step(name="fetch", args={}, observation=_obs(n_words, word),
                raw_output=f"<tool_call>step{i}</tool_call>")


def _tool_responses(msgs):
    """The <tool_response> observation bodies, in message order."""
    return [m["content"] for m in msgs if m["role"] == "user"][1:]  # drop the task/user msg


def _pair_tokens(step: Step) -> int:
    return count_tokens(step.raw_output) + count_tokens(step.observation)


# --- a big-but-under-budget observation is kept WHOLE, not clipped -----------

def test_full_observation_kept_when_under_budget():
    p = _policy(ctx_tokens=default_ctx_tokens(), max_history=40)
    obs = _obs(5_000)
    hist = [Step(name="fetch", args={}, observation=obs, raw_output="<tool_call>fetch</tool_call>")]
    msgs = p.build_messages(Task("t", "q"), hist)
    bodies = _tool_responses(msgs)
    assert len(bodies) == 1
    assert obs in bodies[0]                 # the full observation, untouched
    assert "...(truncated)" not in bodies[0]


# --- total kept content respects ctx_tokens: oldest steps dropped when over ---

def test_ctx_tokens_budget_drops_oldest_steps():
    hist = [_step(i, 300) for i in range(5)]
    per_pair = _pair_tokens(hist[0])
    budget = int(per_pair * 2.5)            # fits the 2 most recent pairs, not 3
    p = _policy(ctx_tokens=budget, max_history=40)
    msgs = p.build_messages(Task("t", "q"), hist)
    bodies = _tool_responses(msgs)
    assert len(bodies) == 2
    # every kept observation is intact (not truncated) since each individually fits
    assert all("...(truncated)" not in b for b in bodies)
    # and it must be the MOST RECENT steps that survived
    joined = "".join(m["content"] for m in msgs)
    assert "step3" in joined and "step4" in joined and "step0" not in joined


# --- a single observation that alone overflows the WHOLE budget gets truncated ---

def test_single_over_budget_observation_is_truncated_with_marker():
    hist = [_step(0, 20_000, word="zeta")]
    budget = max(64, count_tokens(hist[0].observation) // 10)
    p = _policy(ctx_tokens=budget, max_history=40)
    msgs = p.build_messages(Task("t", "q"), hist)
    bodies = _tool_responses(msgs)
    assert len(bodies) == 1
    assert "...(truncated)" in bodies[0]
    kept = bodies[0].split("<tool_response>\n", 1)[1].split("\n...(truncated)")[0]
    assert count_tokens(kept) <= budget     # cut by TOKENS, bounded by the budget
    assert count_tokens(kept) < count_tokens(hist[0].observation)


# --- kept pairs are emitted in chronological (oldest-kept -> newest) order ---

def test_kept_pairs_are_chronologically_ordered():
    p = _policy(ctx_tokens=default_ctx_tokens(), max_history=40)
    hist = [_step(i, 20) for i in range(4)]
    msgs = p.build_messages(Task("t", "q"), hist)
    roles = [m["role"] for m in msgs]
    # system, task-user, then alternating assistant/user per kept step
    assert roles[:2] == ["system", "user"]
    assert roles[2:] == ["assistant", "user"] * 4
    # assistant messages appear in the SAME order the steps were taken (step0..step3)
    assistant_bodies = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert assistant_bodies == [f"<tool_call>step{i}</tool_call>" for i in range(4)]


# --- the budget comes from AGENT_CTX_TOKENS at construction, never from characters ---

def test_default_budget_reads_env_at_construction(monkeypatch):
    monkeypatch.setenv("AGENT_CTX_TOKENS", "1234")
    assert _policy().ctx_tokens == 1234
    monkeypatch.delenv("AGENT_CTX_TOKENS")
    assert _policy().ctx_tokens == default_ctx_tokens()
    with pytest.raises(TypeError):
        _policy(ctx_chars=1000)             # the character-based knob no longer exists


# --- context-overflow shrink-retry (vLLM 400 "maximum context length") -------

def test_propose_shrinks_window_on_context_overflow():
    """A generate() that 400s on over-long prompts triggers up to 3 shrink-retries
    (ctx_tokens x0.85 each); the eventual call succeeds with a smaller window and
    non-overflowing histories are built with the ORIGINAL budget untouched."""
    seen = []
    hist = [_step(i, 400) for i in range(10)]
    limit = int(sum(_pair_tokens(s) for s in hist) * 0.75)

    def gen(msgs):
        total = sum(count_tokens(m["content"]) for m in msgs if m["role"] != "system")
        seen.append(total)
        if total > limit:
            raise RuntimeError(
                "Error code: 400 - This model's maximum context length is 131072 tokens...")
        return "<answer>ok</answer>"

    budget = sum(_pair_tokens(s) for s in hist) + 10
    p = AgentPolicy(generate=gen, system=SYSTEM, ctx_tokens=budget, max_history=40)
    out = p.propose(Task("t", "q"), hist)
    assert out == "<answer>ok</answer>"
    assert len(seen) >= 2 and seen[-1] <= limit             # shrank until it fit
    assert p.ctx_tokens == budget                           # persistent budget unchanged
    seen.clear()
    p.propose(Task("t", "q"), [_step(0, 10)])
    assert len(seen) == 1                                   # small history: no retry


def test_propose_reraises_non_overflow_errors():
    def gen(msgs):
        raise RuntimeError("connection reset")
    p = AgentPolicy(generate=gen, system=SYSTEM, ctx_tokens=8_000)
    with pytest.raises(RuntimeError, match="connection reset"):
        p.propose(Task("t", "q"), [_step(0, 10)])
