"""The unified agent loop: drive a search -> fetch workspace, end on <fix>/<answer>/
max_steps; BQL execution errors come back as observations; the LLM policy builds the
Tongyi chat message shape."""
from types import SimpleNamespace

from agent_search.agent.loop import Step, Task, run_episode
from agent_search.agent.policies import AgentPolicy
from agent_search.tokens import count_tokens
from agent_search.retrievers.bql.executor import StructuralExecutor, execute_bql
from agent_search.corpus.units import units_from_python_source
from agent_search.strategies import CONDITIONS
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch_code.tool import FetchCode, SearchCode

RESEARCH_SNIP_SYSTEM = CONDITIONS["research_snip"].render()

SRC = ("def make_token(user):\n    return str(user)\n"
       "def create_session_token(user):\n    return make_token(user)\n")


def _units():
    return units_from_python_source("a.py", SRC)


def _ex_byid():
    units = _units()
    return StructuralExecutor(units), {u.doc_id: u for u in units}


# --- execute_bql: errors become observations, never raise -------------------

def test_execute_bql_parse_error_is_observation():
    ex, by_id = _ex_byid()
    obs = execute_bql("AND(make_token", ex, by_id)
    assert obs.error and "parse error" in obs.error and obs.hits == []


def test_execute_bql_type_error_is_observation():
    ex, by_id = _ex_byid()
    obs = execute_bql("NOT(make_token)", ex, by_id)      # NOT only allowed inside AND
    assert obs.error and "type error" in obs.error


def test_execute_bql_valid_returns_hits_with_provenance():
    ex, by_id = _ex_byid()
    obs = execute_bql("IN(call, make_token)", ex, by_id)
    assert any(h.doc_id == "a.py::create_session_token" for h in obs.hits)
    assert all(h.path for h in obs.hits)


def test_execute_bql_n_hits_is_untruncated_total():
    ex, by_id = _ex_byid()
    obs = execute_bql("user", ex, by_id, k=1)
    assert len(obs.hits) <= 1 and obs.n_hits >= 2     # k truncates hits, not n_hits


# --- the unified loop: search -> fetch -> <fix> -----------------------------

def _fixws():
    units = _units()
    files = {"a.py": SRC}
    ubyid = {u.doc_id: u for u in units}
    ex = StructuralExecutor(units)
    state = EpisodeState(question="q")
    sc = SearchCode(name="search").bind(state, units, ubyid, {"bql_plain": ex}, files=files)
    fc = FetchCode(name="fetch").bind(state, units, ubyid, {}, files=files)
    return ToolBox([sc, fc], state), units


def test_run_episode_search_fetch_fix_terminates():
    ws, units = _fixws()

    class P:
        def propose(self, task, history):
            if len(history) == 0:
                return '<tool_call>{"name":"search","arguments":{"query":"make_token[def]"}}</tool_call>'
            if len(history) == 1:
                return '<tool_call>{"name":"fetch","arguments":{"specs":[[1,"make_token"]]}}</tool_call>'
            return "<fix>\nfile: a.py\nfunction: make_token\nchange: coerce user\n</fix>"
    traj = run_episode(P(), Task("t", "make token for user"), ws, units, max_steps=6)
    assert traj.stopped_reason == "fix"
    assert traj.fix_text and "file: a.py" in traj.fix_text
    assert [s.name for s in traj.steps][:2] == ["search", "fetch"]


def test_fix_guard_bounces_an_ungrounded_fix():
    """A <fix> that names a file never fetched is a guess — the guard bounces it back
    instead of ending the episode (mirrors the code arm's grounding guard)."""
    ws, units = _fixws()

    def guard(fix_text, steps):
        return (False, "not grounded") if "z.py" in fix_text else (True, "")

    class P:
        def __init__(self):
            self.n = 0
        def propose(self, task, history):
            self.n += 1
            # first emit an ungrounded fix (bounced), then a grounded one
            return ("<fix>\nfile: z.py\nfunction: x\nchange: y\n</fix>" if self.n == 1
                    else "<fix>\nfile: a.py\nfunction: make_token\nchange: y\n</fix>")

    traj = run_episode(P(), Task("t", "q"), ws, units, max_steps=6, fix_guard=guard)
    assert traj.stopped_reason == "fix" and "file: a.py" in traj.fix_text
    # the bounced attempt is recorded as a rejection step before the accepted fix
    assert any(s.name == "fix_rejected" for s in traj.steps)


# --- AgentPolicy: prompt-profile-driven, model injected ----------------------

def test_agent_policy_uses_prompt_profile_and_fake_model():
    seen = {}

    def fake_generate(messages):
        seen["messages"] = messages
        return '<tool_call>{"name":"search","arguments":{"query":"make_token[def]"}}</tool_call>'

    p = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    out = p.propose(Task("t", "find the bug"), [])
    assert "search" in out
    sysmsg = seen["messages"][0]["content"].lower()
    assert seen["messages"][0]["role"] == "system" and "search" in sysmsg
    assert seen["messages"][1]["role"] == "user" and "find the bug" in seen["messages"][1]["content"]


def test_agent_policy_history_is_alternating_messages():
    p = AgentPolicy(generate=lambda m: "<answer></answer>", system=RESEARCH_SNIP_SYSTEM)
    hist = [Step(name="search", args={"query": "x[title]"}, observation="3 matches",
                 raw_output='<tool_call>{"name":"search","arguments":{"query":"x[title]"}}</tool_call>')]
    msgs = p.build_messages(Task("t", "q"), hist)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert "<tool_response>" in msgs[-1]["content"] and "3 matches" in msgs[-1]["content"]


# --- inline forced-answer elicitation (the budget-nudge terminal branch) ----------------------
# See agent_search/agent/forced_answer.py for the shared prefill mechanism these exercise
# end-to-end through run_episode's terminal branch (force_answer gated to non-code domains).

class _AnyToolWS:
    """A minimal WorkspaceLike: accepts any tool call, returns a canned observation. No
    location-ranking `surfaced` attribute — mirrors the doc-research arm's <answer> terminal."""

    def run(self, name: str, args: dict) -> str:
        return "(1 matches) some observation"


def _fake_client(contents):
    """Same stub shape as tests/test_forced_answer.py / test_force_answer_backfill.py: returns
    `contents[i]` on the i-th `create()` call, records every call's kwargs."""
    calls = []

    def create(**kwargs):
        i = len(calls)
        calls.append(kwargs)
        text = contents[min(i, len(contents) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._calls = calls
    return client


def test_run_episode_nudge_compliance_tags_elicitation_nudge():
    """The model complies with the inline budget nudge directly (answers on the forced turn) —
    no inline elicitation call is needed; tagged "nudge"."""
    calls = {"n": 0}

    def fake_generate(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
        return "<answer>Paris</answer>"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=3, domain="general")
    assert traj.stopped_reason == "answer"
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "nudge"


def test_run_episode_ignored_nudge_triggers_inline_prefill_and_fills_answer():
    """The model tool-calls instead of answering on the forced turn (the bug this task fixes) —
    the inline elicitation (agent_search.agent.forced_answer, reached via policy.generate's
    client/model attributes — see agent_search/agent/backbone/openai_chat.py::openai_compat_generate) fires
    and fills final_answer; tagged "prefill_inline"."""
    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["Paris"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=3, domain="general")
    assert traj.stopped_reason == "max_steps"        # the underlying episode outcome is unchanged
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "prefill_inline"
    assert len(fake_generate.client._calls) == 1      # the prefill call alone succeeded
    prefill_kwargs = fake_generate.client._calls[0]
    assert prefill_kwargs["messages"][-1] == {"role": "assistant", "content": "<answer>"}


def test_run_episode_inline_elicitation_message_list_matches_live_build_messages():
    """The messages the inline call sends are EXACTLY `policy.build_messages(task, steps)` (the
    episode's own live steps, already including the injected budget nudge) plus the FORCE_MSG
    tools-disabled turn — the same construction the offline backfill script reconstructs from a
    persisted row (scripts/force_answer_backfill.py::reconstruct_messages)."""
    from agent_search.agent.forced_answer import FORCE_MSG

    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["Paris"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
               max_steps=3, domain="general")
    sent = fake_generate.client._calls[0]["messages"][:-1]   # strip the "<answer>" prefill turn
    assert sent[-1] == {"role": "user",
                        "content": f"<tool_response>\n{FORCE_MSG}\n</tool_response>"}
    assert sent[0]["role"] == "system"


def test_run_episode_prefill_failure_is_tagged_not_crashed():
    """policy.generate carries no client/model (e.g. in-process vLLM) -> inline elicitation has
    nothing to call: tagged "prefill_failed", episode still completes normally (never raises)."""
    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "q"), _AnyToolWS(), units=[], max_steps=3, domain="general")
    assert traj.stopped_reason == "max_steps"
    assert traj.final_answer == ""
    assert traj.elicitation == "prefill_failed"


def test_run_episode_prefill_call_exception_is_tagged_not_crashed():
    """The inline elicitation's own call raises (e.g. a dropped connection) -> caught, tagged
    "prefill_failed" — never propagates and crashes the episode."""
    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'

    def _boom(**kwargs):
        raise RuntimeError("connection reset")
    fake_generate.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_boom)))
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "q"), _AnyToolWS(), units=[], max_steps=3, domain="general")
    assert traj.final_answer == ""
    assert traj.elicitation == "prefill_failed"


def test_run_episode_code_domain_never_tags_elicitation():
    """force_answer is domain-gated (non-code only, same gate as the pre-existing budget nudge) —
    a code-domain episode never injects the nudge, so elicitation stays absent (None) even at
    max_steps, exactly matching pre-change rows."""
    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["should never be called"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "q"), _AnyToolWS(), units=[], max_steps=3, domain="code")
    assert traj.elicitation is None
    assert not fake_generate.client._calls           # the inline elicitation never fired at all


def test_run_episode_organic_answer_before_budget_has_no_elicitation_tag():
    """An episode that answers well before the reserved final turn (the common case) never
    touches the nudge/elicitation machinery — elicitation stays None."""
    def fake_generate(messages):
        return "<answer>Paris</answer>"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "q"), _AnyToolWS(), units=[], max_steps=10, domain="general")
    assert traj.stopped_reason == "answer"
    assert traj.final_answer == "Paris"
    assert traj.elicitation is None


# --- inline elicitation context-overflow shrink-retry (the DCI live-job bug) --------------------
#
# The condition agent's live loop-driver wiring is exactly `fake_generate.client`/
# `fake_generate.model` attached to `AgentPolicy.generate`, same as `openai_compat_generate`
# (agent_search/agent/backbone/openai_chat.py) does for real. `_elicit_inline` (agent_search/agent/loop.py)
# used to call `elicit_final_answer` ONCE at the policy's full, unshrunk `ctx_tokens` and let its
# outer `except Exception` swallow a "maximum context length" 400 exactly like any other failure
# — so a served-vLLM episode whose final-turn history overflowed the window always resent the
# SAME maximum-budget prompt and always failed the same way (never got smaller, never recovered).
# These tests drive that through the REAL `run_episode` -> `_elicit_inline` -> `forced_answer.
# elicit_final_answer` -> `client.chat.completions.create` chain (the loop driver's actual
# closure contract), not a synthetic lambda standing in for `propose()` alone.

_OVERFLOW_MSG = ("This model's maximum context length is 131072 tokens. However, you requested "
                 "4000 output tokens and your prompt contains at least 127073 input tokens, for a "
                 "total of at least 131073 tokens. Please reduce the length of the input prompt or "
                 "the number of requested output tokens. (parameter=input_tokens, value=127073)")


class _BigObsWS:
    """A WorkspaceLike whose every observation is large, so a many-step episode's accumulated
    history alone is big enough to exercise the elicitation call's ctx_tokens shrink (mirrors a
    DCI episode's big bash/read observations)."""

    def run(self, name: str, args: dict) -> str:
        return "obs " * 2_000                 # ~2,000 tokens on either ruler


def _fake_overflow_client(overflow_above: int, final_content: str = "Paris"):
    """Same stub shape as `_fake_client`, except `create()` raises a REAL "maximum context
    length"-shaped exception (the exact text vLLM's server sends — see the module docstring)
    whenever the request's total message TOKENS exceed `overflow_above`, and only succeeds once a
    shrunk retry brings it under that line — so this proves the shrink loop actually reaches a
    SMALLER prompt, not just that it retries at all."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        # the history is what the shrink governs; the (fixed) system prompt is excluded
        total = sum(count_tokens(m["content"]) for m in kwargs["messages"] if m["role"] != "system")
        if total > overflow_above:
            raise RuntimeError(f"Error code: 400 - {_OVERFLOW_MSG}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=final_content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._calls = calls
    return client


def test_run_episode_inline_elicitation_shrinks_on_context_overflow_then_recovers():
    """The reserved-final-turn elicitation call 400s at the full ctx_tokens budget; the SAME
    15%-shrink-and-retry `AgentPolicy.propose()` uses (up to 3 shrinks) must apply here too, so
    the episode still recovers an answer instead of degrading straight to "prefill_failed"."""
    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    # ctx_tokens=20_000: the full-budget elicitation prompt (~10 kept pairs of ~2k tokens) sits
    # above 13_000 tokens but three 0.85 shrinks (20000 -> 17000 -> 14450 -> 12282) bring it under.
    client = _fake_overflow_client(overflow_above=13_000)
    fake_generate.client = client
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM,
                         ctx_tokens=20_000)
    traj = run_episode(policy, Task("t", "q"), _BigObsWS(), units=[], max_steps=15, domain="general")

    assert traj.final_answer == "Paris"
    assert traj.elicitation == "prefill_inline"
    # more than one attempt was needed (the first, full-budget attempt overflowed) ...
    assert len(client._calls) > 1
    # ... and each retry's prompt was STRICTLY SMALLER than the previous one — the shrink is
    # actually taking effect, not resending the identical maximum-budget prompt every time (the
    # live bug: vLLM's log showed the identical "127073 input tokens" 400 on every occurrence).
    sizes = [sum(count_tokens(m["content"]) for m in c["messages"] if m["role"] != "system")
             for c in client._calls]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) > 1
    assert sizes[-1] <= 13_000                        # the attempt that finally succeeded


def test_run_episode_inline_elicitation_all_shrinks_still_overflow_stays_prefill_failed():
    """A pathological episode whose final-turn prompt is still too big after all 3 shrinks
    degrades to "prefill_failed" (exactly the pre-existing degrade path) rather than crashing —
    but only after genuinely trying 4 progressively smaller prompts, never 4 identical ones."""
    def fake_generate(messages):
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    # threshold far below anything 3 shrinks of a 20_000 budget can reach (20000 * 0.85**3 ≈ 12282).
    client = _fake_overflow_client(overflow_above=1_000)
    fake_generate.client = client
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM,
                         ctx_tokens=20_000)
    traj = run_episode(policy, Task("t", "q"), _BigObsWS(), units=[], max_steps=15, domain="general")

    assert traj.final_answer == ""
    assert traj.elicitation == "prefill_failed"       # degrades cleanly, never raises
    assert len(client._calls) == 4                    # 1 + 3 shrinks, matching propose()'s cap
    sizes = [sum(count_tokens(m["content"]) for m in c["messages"] if m["role"] != "system")
             for c in client._calls]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == 4                        # every attempt strictly smaller — not 4 retries of the same prompt


# --- proactive context-budget early stop (AGENT_CTX_WINDOW / AGENT_CTX_STOP_FRAC) --------------
#
# The reactive reserved-final-turn nudge above only fires once max_steps runs out; on a deep
# episode whose accumulated history already sits near the model's window, ONE more observation
# on that final turn is what trips vLLM's "prompt + requested_output > max-model-len" 400 (see
# agent_search/agent/loop.py's module-level comment above `_last_prompt_tokens`). These tests
# drive the PROACTIVE early stop through the REAL run_episode -> the SAME nudge/elicitation
# machinery, just triggered by observed token budget instead of step count.

def _prompt_tokens_usage(sequence):
    """A stateful usage_fn double. `record()` appends the next value of `sequence` (clamped to the
    last element once exhausted) as a fake (prompt_tokens, completion_tokens, 0, 0) event; the
    returned `usage_fn` is a LIVE reader of everything recorded so far — mirrors
    `agent_search.agent.backbone.usage_events` being a thread-local reader of whatever the real
    backend has recorded up to THIS point in the episode (see loop.py::_last_prompt_tokens)."""
    events = []

    def record():
        i = min(len(events), len(sequence) - 1)
        events.append((sequence[i], 10, 0, 0))

    def usage_fn():
        return list(events)

    return record, usage_fn


def test_run_episode_ctx_budget_early_stop_triggers_elicitation_before_max_steps():
    """Default knobs (AGENT_CTX_WINDOW=131072, AGENT_CTX_STOP_FRAC=0.90 -> threshold ~117965
    tokens). The fake generate's reported prompt_tokens crosses that threshold on its 2nd call;
    the model keeps tool-calling on the forced turn (never complies with the nudge directly), so
    the inline elicitation fires -- EARLY, well before the 20-step budget -- and stopped_reason is
    "ctx_budget", not "max_steps"."""
    record, usage_fn = _prompt_tokens_usage([50_000, 125_000])

    def fake_generate(messages):
        record()
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["Paris"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=20, usage_fn=usage_fn, domain="general")

    assert traj.stopped_reason == "ctx_budget"        # distinct from "max_steps"
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "prefill_inline"
    # stopped well short of the 20-step budget -- proves this was PROACTIVE, not a coincidence
    # of running out of steps.
    assert len(traj.steps) < 20


def test_run_episode_ctx_budget_subthreshold_behaves_like_max_steps_today():
    """prompt_tokens reported by usage_fn stay far under the 0.90*131072 threshold the whole
    episode -> the proactive early stop never fires; the episode runs to max_steps exactly as it
    does with no usage_fn passed at all (same assertions as
    test_run_episode_ignored_nudge_triggers_inline_prefill_and_fills_answer above)."""
    record, usage_fn = _prompt_tokens_usage([500])   # constant, far below any threshold

    def fake_generate(messages):
        record()
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["Paris"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=3, usage_fn=usage_fn, domain="general")

    assert traj.stopped_reason == "max_steps"         # NOT "ctx_budget" -- threshold never crossed
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "prefill_inline"
    assert len(fake_generate.client._calls) == 1


def test_run_episode_ctx_stop_frac_ge_1_disables_early_stop(monkeypatch):
    """AGENT_CTX_STOP_FRAC >= 1.0 is the explicit opt-out sentinel: even a reported prompt_tokens
    value already "over" the window never triggers the proactive stop -- behavior is
    byte-identical to today's reactive-only max_steps nudge."""
    monkeypatch.setenv("AGENT_CTX_STOP_FRAC", "1.0")
    record, usage_fn = _prompt_tokens_usage([200_000])   # already over any real window

    def fake_generate(messages):
        record()
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["Paris"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=3, usage_fn=usage_fn, domain="general")

    assert traj.stopped_reason == "max_steps"         # not "ctx_budget" -- early stop is disabled
    assert traj.final_answer == "Paris"
    assert traj.elicitation == "prefill_inline"


def test_run_episode_ctx_threshold_scales_with_window(monkeypatch):
    """A SMALLER AGENT_CTX_WINDOW trips the early stop at a smaller absolute prompt_tokens value
    -- proving the threshold is FRAC*WINDOW (adaptive), not a hardcoded absolute: the exact same
    reported prompt_tokens sequence crosses a tiny window's threshold but does NOT cross the
    DEFAULT (unset) 131072 window's threshold within the same step budget."""
    monkeypatch.setenv("AGENT_CTX_WINDOW", "2000")     # threshold = 0.90 * 2000 = 1800
    record, usage_fn = _prompt_tokens_usage([1_000, 1_900])

    def fake_generate(messages):
        record()
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate.client = _fake_client(["Paris"])
    fake_generate.model = "m"

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=20, usage_fn=usage_fn, domain="general")
    assert traj.stopped_reason == "ctx_budget"
    assert traj.final_answer == "Paris"

    monkeypatch.delenv("AGENT_CTX_WINDOW", raising=False)   # back to the DEFAULT 131072 window
    record2, usage_fn2 = _prompt_tokens_usage([1_000, 1_900])

    def fake_generate2(messages):
        record2()
        return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
    fake_generate2.client = _fake_client(["Paris"])
    fake_generate2.model = "m"

    policy2 = AgentPolicy(generate=fake_generate2, system=RESEARCH_SNIP_SYSTEM)
    traj2 = run_episode(policy2, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                        max_steps=3, usage_fn=usage_fn2, domain="general")
    # the SAME reported values (1_000, 1_900) never approach 0.90*131072 -- no early trigger, so
    # the episode runs its normal reactive max_steps path instead.
    assert traj2.stopped_reason == "max_steps"


# --- on_step streaming callback (live demo) ----------------------------------------------------

def test_run_episode_on_step_fires_once_per_step_with_same_objects():
    calls = {"n": 0}

    def fake_generate(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            return '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
        return "<answer>Paris</answer>"

    seen = []
    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "capital of France?"), _AnyToolWS(), units=[],
                       max_steps=5, domain="general", on_step=seen.append)
    assert len(seen) == len(traj.steps)
    assert all(a is b for a, b in zip(seen, traj.steps))
    assert [s.name for s in seen][-1] == "answer"


def test_run_episode_broken_on_step_never_crashes_episode():
    def fake_generate(messages):
        return "<answer>Paris</answer>"

    def boom(step):
        raise RuntimeError("listener died")

    policy = AgentPolicy(generate=fake_generate, system=RESEARCH_SNIP_SYSTEM)
    traj = run_episode(policy, Task("t", "q"), _AnyToolWS(), units=[],
                       max_steps=3, domain="general", on_step=boom)
    assert traj.final_answer == "Paris"
