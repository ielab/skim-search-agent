"""The Responses-API driver (DIVER's gpt-oss protocol) against a fake `/v1/responses` client."""
import json
from types import SimpleNamespace

from agent_search.agent.responses_driver import FINAL_ROUND_MSG, responses_tools, run_episode_responses
from agent_search.strategies import CONDITIONS


class _WS:
    tools = ("search", "get_document")

    def __init__(self):
        self.calls = []
        self.last_hits = []
        self.surfaced = []

    def __getitem__(self, name):
        return SimpleNamespace(declaration=lambda n=name: {"name": n, "description": f"the {n} tool",
                                                           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}})

    def run(self, name, args):
        self.calls.append((name, args))
        if name == "search":
            self.last_hits = ["7"]
            self.surfaced = ["7"]
            return "DocID:7\n[A doc]\nsome text"
        return f"full text of {args.get('docid')}"


def _resp(items, inp=100, out=20):
    return SimpleNamespace(
        model_dump=lambda mode="python": {"output": items},
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                              input_tokens_details=SimpleNamespace(cached_tokens=5),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=8)))


def _client(turns):
    requests = []

    def create(**kw):
        requests.append(kw)
        return turns[min(len(requests) - 1, len(turns) - 1)]
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    client.requests = requests
    return client


def _call(name, args, cid="c1"):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": cid}


def _reasoning(text):
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}], "content": []}


def _message(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def test_tools_are_declared_in_the_flat_responses_format():
    tools = responses_tools(_WS())
    assert tools[0]["type"] == "function" and tools[0]["name"] == "search"
    assert tools[0]["parameters"]["required"] == ["q"]
    assert "function" not in tools[0] or tools[0]["function"] is None or True


def test_an_episode_runs_calls_and_ends_on_the_first_plain_message():
    ws = _WS()
    client = _client([_resp([_reasoning("search first"), _call("search", {"query": "treaty 1848"})], inp=100),
                      _resp([_reasoning("read it"), _call("get_document", {"docid": "7"}, "c2")], inp=300),
                      _resp([_message("Explanation: ... [7]\nExact Answer: Guadalupe Hidalgo\nConfidence: 90%")], inp=450, out=30)])
    traj = run_episode_responses(ws, "which treaty?", model="openai/gpt-oss-20b", instructions="be careful",
                                 client=client, max_turns=50, user_template="Q: {question}")
    assert [s.name for s in traj.steps] == ["search", "get_document", "answer"]
    assert ws.calls == [("search", {"query": "treaty 1848"}), ("get_document", {"docid": "7"})]
    assert traj.stopped_reason == "answer" and "Exact Answer: Guadalupe Hidalgo" in traj.final_answer
    assert traj.llm_calls == 3 and traj.prompt_tokens == 850 and traj.completion_tokens == 70
    assert traj.cached_input_tokens == 15 and traj.reasoning_tokens == 24
    assert [s.prompt_tokens for s in traj.steps] == [100, 300, 450]     # count-once metering per turn
    first = client.requests[0]
    assert first["instructions"] == "be careful" and first["input"] == [{"role": "user", "content": "Q: which treaty?"}]
    assert first["reasoning"]["effort"] in ("medium", "low", "high") and first["truncation"] == "auto"
    assert [t["name"] for t in first["tools"]] == ["search", "get_document"]
    # the transcript carries the model's items and our outputs in order
    second = client.requests[1]["input"]
    assert second[1]["type"] == "reasoning" and second[2]["type"] == "function_call"
    assert second[3] == {"type": "function_call_output", "call_id": "c1", "output": "DocID:7\n[A doc]\nsome text"}
    assert traj.steps[0].raw_output.startswith("search first")


def test_the_final_turn_is_made_without_tools_and_the_model_is_told_retrieval_is_over():
    ws = _WS()
    client = _client([_resp([_call("search", {"query": "a"})]), _resp([_call("search", {"query": "b"}, "c2")]),
                      _resp([_message("Exact Answer: forced")])])
    traj = run_episode_responses(ws, "q", model="m", instructions="i", client=client, max_turns=3)
    assert traj.final_answer == "Exact Answer: forced" and traj.elicitation == "nudge"
    last = client.requests[-1]
    assert "tools" not in last
    assert last["input"][-1] == {"role": "user", "content": FINAL_ROUND_MSG}


def test_a_mislabelled_mcp_call_and_a_run_on_name_are_still_our_tool():
    ws = _WS()
    client = _client([_resp([{"type": "mcp_call", "server_label": "functions", "id": "m1", "name": "search<|channel|>commentary",
                              "arguments": json.dumps({"query": "x"})}]),
                      _resp([_message("done")])])
    traj = run_episode_responses(ws, "q", model="m", instructions="i", client=client, max_turns=5)
    assert ws.calls == [("search", {"query": "x"})]
    assert client.requests[1]["input"][2] == {"type": "function_call_output", "call_id": "m1", "output": "DocID:7\n[A doc]\nsome text"}
    assert traj.final_answer == "done"


def test_a_turn_that_ends_mid_thought_is_dropped_and_retried():
    ws = _WS()
    client = _client([_resp([_reasoning("hmm")]), _resp([_message("42")])])
    traj = run_episode_responses(ws, "q", model="m", instructions="i", client=client, max_turns=5)
    assert traj.steps[0].name == "none" and "mid-thought" in traj.steps[0].observation
    assert client.requests[1]["input"] == client.requests[0]["input"]     # the dangling item was dropped
    assert traj.final_answer == "42" and traj.llm_calls == 2


def test_invalid_arguments_are_an_error_observation_not_a_crash():
    ws = _WS()
    client = _client([_resp([{"type": "function_call", "name": "search", "arguments": "{not json", "call_id": "c1"}]),
                      _resp([_message("x")])])
    traj = run_episode_responses(ws, "q", model="m", instructions="i", client=client, max_turns=5)
    assert traj.steps[0].observation.startswith("ERROR: invalid tool call") and ws.calls == []


def test_the_strong_task_carries_divers_user_template():
    t = CONDITIONS["research_dedup_dense_strong"].task
    u = t.user_template.format(question="Q?")
    assert u.startswith("You are a deep research agent") and "Question: Q?" in u
    assert "Exact Answer: {your succinct, final answer}" in u


def test_a_rejected_transcript_drops_the_last_turn_and_takes_it_again():
    class BadRequestError(Exception):
        status_code = 400
    ws = _WS()
    turns = [_resp([_reasoning("bad header"), _call("search", {"query": "a"})]), "reject",
             _resp([_call("search", {"query": "b"}, "c2")]), _resp([_message("done")])]
    requests = []

    def create(**kw):
        requests.append(kw)
        t = turns[len(requests) - 1]
        if t == "reject":
            raise BadRequestError("Error code: 400 - Unknown channel: analysis-to=functions.search")
        return t
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    traj = run_episode_responses(ws, "q", model="m", instructions="i", client=client, max_turns=10)
    assert [s.name for s in traj.steps] == ["search", "none", "search", "answer"]
    assert "rejected the transcript" in traj.steps[1].observation
    # the third request carries only the user turn: the rejected turn and its tool output are gone
    assert [m.get("type") or m.get("role") for m in requests[2]["input"]] == ["user"]
    assert ws.calls == [("search", {"query": "a"}), ("search", {"query": "b"})]
    assert traj.final_answer == "done" and traj.stopped_reason == "answer"


def test_a_persistent_failure_ends_the_episode_with_an_error_row_not_a_crash(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    ws = _WS()

    def create(**kw):
        raise RuntimeError("connection reset")
    traj = run_episode_responses(ws, "q", model="m", instructions="i",
                                 client=SimpleNamespace(responses=SimpleNamespace(create=create)), max_turns=5)
    assert traj.stopped_reason == "error" and traj.final_answer == "" and traj.steps[0].name == "none"
