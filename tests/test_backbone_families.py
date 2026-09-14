"""The per-backbone protocol pieces DIVER's clients differ in: a text terminal (a reply without
a tool call is the answer), a turn cut off mid-thought, per-turn generation budgets, the thinking
switch, the sampling knobs a served model takes, and the text-mode forced prefill."""
from types import SimpleNamespace

import agent_search.agent.backbone.openai_chat as OC
from agent_search.agent.forced_answer import TEXT_PREFILL, elicit_final_answer, prefill_for
from agent_search.agent.loop import Task, run_episode
from agent_search.agent.policies import AgentPolicy
from agent_search.strategies import CONDITIONS

QWEN_SYSTEM = CONDITIONS["research_dedup_dense_qwen"].render()


class _WS:
    tools = ("search", "get_document")

    def run(self, name, args):
        return "DocID:7\n[A doc]\nsome text"


def _gen(replies, finish=None):
    """A generate callable that replays `replies` and reports `finish` reasons like the served backend."""
    calls = []

    def generate(messages, **opts):
        i = len(calls)
        calls.append((messages, opts))
        generate.last_finish_reason = (finish or [None] * len(replies))[min(i, len(replies) - 1)]
        return replies[min(i, len(replies) - 1)]
    generate.calls = calls
    generate.last_finish_reason = None
    return generate


def test_text_terminal_ends_on_the_first_reply_without_a_tool_call():
    gen = _gen(['<tool_call>{"name":"search","arguments":{"query":"treaty 1848"}}</tool_call>',
                "<think>done</think>The Treaty of Guadalupe Hidalgo."])
    traj = run_episode(AgentPolicy(generate=gen, system=QWEN_SYSTEM), Task("t", "which treaty?"), _WS(),
                       units=[], max_steps=10, domain="general", terminal="text")
    assert traj.stopped_reason == "answer"
    assert traj.final_answer == "The Treaty of Guadalupe Hidalgo."
    assert [s.name for s in traj.steps] == ["search", "answer"]


def test_answer_terminal_still_nudges_on_a_reply_without_a_tool_call():
    gen = _gen(["I think it is Guadalupe Hidalgo.", "<answer>Guadalupe Hidalgo</answer>"])
    traj = run_episode(AgentPolicy(generate=gen, system=QWEN_SYSTEM), Task("t", "which treaty?"), _WS(),
                       units=[], max_steps=10, domain="general")
    assert traj.steps[0].name == "none" and traj.steps[0].observation.startswith("ERROR: no tool call")
    assert traj.final_answer == "Guadalupe Hidalgo"


def test_a_turn_cut_off_mid_thought_is_discarded_and_the_next_turn_runs_without_thinking():
    gen = _gen(["<think>let me reason at great length about", '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>', "Final: 1848."],
               finish=["length", "stop", "stop"])
    policy = AgentPolicy(generate=gen, system=QWEN_SYSTEM)
    traj = run_episode(policy, Task("t", "when?"), _WS(), units=[], max_steps=10, domain="general", terminal="text")
    assert traj.steps[0].name == "none"
    assert "too long and has been discarded" in traj.steps[0].observation
    assert gen.calls[1][1] == {"thinking": False}     # the retry runs with thinking off
    assert gen.calls[2][1] == {}                       # and only that one
    assert traj.final_answer == "Final: 1848."


def test_per_turn_generation_budgets_follow_the_schedule(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS_SCHEDULE", "4096,2048,1024")
    gen = _gen(['<tool_call>{"name":"search","arguments":{"query":"a"}}</tool_call>'] * 4 + ["done"])
    run_episode(AgentPolicy(generate=gen, system=QWEN_SYSTEM), Task("t", "q"), _WS(), units=[], max_steps=10,
                domain="general", terminal="text")
    assert [c[1].get("max_tokens") for c in gen.calls] == [4096, 2048, 1024, 1024, 1024]


def test_served_backend_passes_sampling_knobs_and_reports_the_finish_reason(monkeypatch):
    monkeypatch.setenv("LLM_TOP_K", "20")
    monkeypatch.setenv("LLM_PRESENCE_PENALTY", "1.5")
    monkeypatch.setenv("LLM_THINKING", "true")
    sink = {}

    def create(**kw):
        sink.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="length")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen = OC.openai_compat_generate("m", client=client, temperature=1.0)
    assert gen([{"role": "user", "content": "q"}], max_tokens=2048, thinking=False) == "hi"
    assert sink["max_tokens"] == 2048 and sink["temperature"] == 1.0 and sink["presence_penalty"] == 1.5
    assert sink["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    assert gen.last_finish_reason == "length"
    gen([{"role": "user", "content": "q"}])
    assert sink["max_tokens"] == 4000 and sink["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_served_backend_sends_no_extra_body_by_default(monkeypatch):
    for k in ("LLM_TOP_K", "LLM_THINKING", "LLM_PRESENCE_PENALTY", "LLM_TOP_P", "LLM_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    sink = {}

    def create(**kw):
        sink.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")])
    gen = OC.openai_compat_generate("m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    gen("q")
    assert "extra_body" not in sink and sink["top_p"] == 0.95 and sink["presence_penalty"] == 1.1


def test_text_terminal_forced_prefill_is_divers_sentence_and_keeps_the_whole_continuation(monkeypatch):
    monkeypatch.delenv("FORCED_ANSWER_PREFILL", raising=False)
    assert prefill_for("text") == TEXT_PREFILL and prefill_for() == "<answer>"
    monkeypatch.setenv("FORCED_ANSWER_PREFILL", "My answer: ")
    assert prefill_for("text") == "My answer: "
    monkeypatch.delenv("FORCED_ANSWER_PREFILL", raising=False)
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="the treaty of 1848. <tool_call>x"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer, tag, raw = elicit_final_answer([{"role": "user", "content": "q"}], client, "m", terminal="text")
    assert answer == "the treaty of 1848." and tag == "prefill"
    assert calls[0]["messages"][-1] == {"role": "assistant", "content": TEXT_PREFILL}
    assert calls[0]["stop"] is None
    assert len(calls) == 1


def test_an_empty_turn_is_re_asked_with_backoff_and_every_attempt_is_metered(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "4")
    from agent_search.agent.backbone.usage import reset_usage, usage_events
    reset_usage()
    replies = iter(["", "   ", "finally a call"])

    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1))
    gen = OC.openai_compat_generate("m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert gen("q") == "finally a call"
    assert len(usage_events()) == 3                      # the two empty attempts cost tokens too


def test_an_always_empty_turn_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "3")
    n = {"calls": 0}

    def create(**kw):
        n["calls"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")])
    gen = OC.openai_compat_generate("m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert gen("q") == "" and n["calls"] == 3 and gen.last_finish_reason == "length"


def test_the_forced_prefill_re_asks_an_empty_continuation(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "5")
    monkeypatch.delenv("FORCED_ANSWER_PREFILL", raising=False)
    replies = iter(["", "1848"])

    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer, tag, _ = elicit_final_answer([{"role": "user", "content": "q"}], client, "m", terminal="text")
    assert answer == "1848" and tag == "prefill"


def test_every_step_records_the_turns_finish_reason():
    gen = _gen(['<tool_call>{"name":"search","arguments":{"query":"a"}}</tool_call>', "done"], finish=["stop", "stop"])
    traj = run_episode(AgentPolicy(generate=gen, system=QWEN_SYSTEM), Task("t", "q"), _WS(), units=[], max_steps=5,
                       domain="general", terminal="text")
    assert [s.finish_reason for s in traj.steps] == ["stop", "stop"]


def test_an_empty_turn_with_thinking_off_is_retried_with_thinking_on(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "3")
    seen = []

    def create(**kw):
        seen.append(kw.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking"))
        content = "" if len(seen) == 1 else '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=""), finish_reason="stop")])
    gen = OC.openai_compat_generate("m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    out = gen("q", thinking=False)
    assert out.startswith("<tool_call>") and seen == [False, True]


def test_text_terminal_forced_answer_falls_back_to_a_plain_ask_when_the_prefill_is_empty(monkeypatch):
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    monkeypatch.delenv("FORCED_ANSWER_PREFILL", raising=False)
    replies = iter(["", "My final answer is Paris."])
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer, tag, _ = elicit_final_answer([{"role": "user", "content": "q"}], client, "m", terminal="text")
    assert answer == "My final answer is Paris." and tag == "plain_ask_fallback"
    assert "extra_body" not in calls[1]


def test_a_reply_filed_entirely_as_reasoning_is_the_turn(monkeypatch):
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "5")
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    n = {"calls": 0}

    def create(**kw):
        n["calls"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", reasoning_content='Let me search.\n\n<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'),
            finish_reason="stop")])
    gen = OC.openai_compat_generate("m", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    out = gen("q")
    assert out.startswith("Let me search.") and out.rstrip().endswith("</tool_call>") and n["calls"] == 1


def test_text_terminal_does_not_take_a_malformed_tool_call_as_the_answer():
    gen = _gen(["<tool_call>\nsearch for Gugulethu schools please\n</tool_call>", "The answer is X."])
    traj = run_episode(AgentPolicy(generate=gen, system=QWEN_SYSTEM), Task("t", "q"), _WS(), units=[], max_steps=5,
                       domain="general", terminal="text")
    assert traj.steps[0].name == "none" and traj.steps[0].observation.startswith("ERROR")
    assert traj.final_answer == "The answer is X."


def test_the_forced_answer_reads_the_reasoning_channel_when_the_content_is_empty(monkeypatch):
    monkeypatch.setenv("LLM_EMPTY_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0")
    monkeypatch.delenv("FORCED_ANSWER_PREFILL", raising=False)
    n = {"calls": 0}

    def create(**kw):
        n["calls"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning_content="Vitali Hakko."), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer, tag, _ = elicit_final_answer([{"role": "user", "content": "q"}], client, "m", terminal="text")
    assert answer == "Vitali Hakko." and tag == "prefill" and n["calls"] == 1


def test_the_budget_nudge_is_configurable_and_the_cut_off_message_names_the_terminal(monkeypatch):
    from agent_search.agent.loop import budget_nudge
    monkeypatch.delenv("FORCED_ANSWER_NUDGE", raising=False)
    assert budget_nudge().startswith("STEP BUDGET REACHED")
    monkeypatch.setenv("FORCED_ANSWER_NUDGE", "Retrieval complete. You are forbidden to call any tools now.")
    gen = _gen(['<tool_call>{"name":"search","arguments":{"query":"a"}}</tool_call>', "<answer>x</answer>"])
    traj = run_episode(AgentPolicy(generate=gen, system=QWEN_SYSTEM), Task("t", "q"), _WS(), units=[], max_steps=2, domain="general")
    assert any("Retrieval complete" in s.observation for s in traj.steps if s.name == "budget")
    gen2 = _gen(["<think>too long", '<tool_call>{"name":"search","arguments":{"query":"a"}}</tool_call>', "<answer>x</answer>"], finish=["length", "stop", "stop"])
    traj2 = run_episode(AgentPolicy(generate=gen2, system=QWEN_SYSTEM), Task("t", "q"), _WS(), units=[], max_steps=6, domain="general")
    assert traj2.steps[0].observation.endswith("directly provide the <tool_call> or <answer>.")
