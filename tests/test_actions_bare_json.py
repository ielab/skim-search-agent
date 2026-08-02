"""Regression: models often emit the tool call as bare JSON WITHOUT the
<tool_call> wrapper. parse_tool_call must still recognize it (otherwise the JSON
is fed to a tool parser -> garbage), and a <think> that quotes a call must not
preempt the real (last) one."""
from agent_search.agent.actions import parse_tool_call


def test_bare_json_search_bql():
    assert parse_tool_call('{"name":"search_bql","arguments":{"query":"IN(def, foo)"}}') == \
        ("search_bql", {"query": "IN(def, foo)"})


def test_bare_json_grep():
    assert parse_tool_call('{"name":"grep","arguments":{"query":"foo bar"}}') == \
        ("grep", {"query": "foo bar"})


def test_bare_json_with_leading_thought():
    out = parse_tool_call('<think>look for the def</think>\n'
                          '{"name":"search_bql","arguments":{"query":"IN(call, parse)"}}')
    assert out == ("search_bql", {"query": "IN(call, parse)"})


