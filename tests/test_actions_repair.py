"""A tool call with one closing bracket missing inside a nested list still parses. Tongyi
drops the inner "]" of fetch specs in a quarter of search_fetch's turns; without this repair
every such turn is a wasted step."""
from agent_search.agent.actions import parse_tool_call


def test_nested_list_missing_one_closer_parses():
    raw = ('<think>\nScrolling further.\n</think>\n\n<tool_call>\n'
           '{"name":"fetch","arguments":{"specs":[[1779,"Joint 49 match summaries"]}}\n</tool_call>')
    assert parse_tool_call(raw) == ("fetch", {"specs": [[1779, "Joint 49 match summaries"]]})


def test_two_pairs_missing_closer_parses():
    raw = '<tool_call>{"name":"fetch","arguments":{"specs":[[1,"A"],[2,"B"]}}</tool_call>'
    assert parse_tool_call(raw) == ("fetch", {"specs": [[1, "A"], [2, "B"]]})


def test_truncated_call_still_repairs():
    raw = '<tool_call>{"name":"bm25_search","arguments":{"query":"treaty 1848'
    assert parse_tool_call(raw) == ("bm25_search", {"query": "treaty 1848"})


def test_stray_closer_is_dropped():
    raw = '<tool_call>{"name":"visit","arguments":{"rank":2}}}</tool_call>'
    assert parse_tool_call(raw) == ("visit", {"rank": 2})
