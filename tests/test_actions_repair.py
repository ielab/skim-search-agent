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


def test_dropped_colon_after_the_key_is_repaired():
    """Tongyi drops the colon and the value's opening quote when a query starts with a quoted
    phrase; the key is also stripped of padding spaces."""
    raw = '<tool_call>\n{"name":"hybrid_search","arguments":{"query "\\"more than 28\\" roads built"}}\n</tool_call>'
    assert parse_tool_call(raw) == ("hybrid_search", {"query": '"more than 28" roads built'})
    assert parse_tool_call('<tool_call>{"name":"bm25_search","arguments":{"query" "treaty 1848"}}</tool_call>') == ("bm25_search", {"query": "treaty 1848"})
