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


def test_unescaped_quotes_inside_the_query_string():
    # OpenResearcher-30B-A3B writes the quoted phrases of a query unescaped
    raw = ('<tool_call>\n{"name":"bm25_search","arguments":{"query":""coordinator" "research group" '
           'founded in 2009"}}\n</tool_call>')
    assert parse_tool_call(raw) == ("bm25_search", {"query": '"coordinator" "research group" founded in 2009'})
    raw = '<tool_call>{"name":"bm25_search","arguments":{"query":""Routledge" 2018 "edited" book""}}</tool_call>'
    assert parse_tool_call(raw) == ("bm25_search", {"query": '"Routledge" 2018 "edited" book"'})


def test_unescaped_quotes_with_a_second_simple_argument():
    raw = '<tool_call>{"name":"search_s","arguments":{"query":""Event overview"[section] AND 2002","k":5}}</tool_call>'
    assert parse_tool_call(raw) == ("search_s", {"query": '"Event overview"[section] AND 2002', "k": 5})


def test_properly_escaped_quotes_are_untouched():
    raw = '<tool_call>{"name":"search_s","arguments":{"query":"\\"Event overview\\"[section]"}}</tool_call>'
    assert parse_tool_call(raw) == ("search_s", {"query": '"Event overview"[section]'})


def test_unquoted_string_value_is_quoted():
    raw = '<tool_call>\n{"name":"bm25_search","arguments":{"query":2009 research group literature}}\n</tool_call>'
    assert parse_tool_call(raw) == ("bm25_search", {"query": "2009 research group literature"})
    raw = '<tool_call>{"name":"visit","arguments":{"rank":3}}</tool_call>'
    assert parse_tool_call(raw) == ("visit", {"rank": 3})           # a bare number stays a number




def test_stray_quote_before_the_closers_is_dropped():
    # Indri queries from Tongyi: escaped inner quotes, then one stray quote before )}}
    raw = ('<tool_call>{"name":"isearch_s","arguments":{"query":"#combine(\\"24 Hours of Spa\\" '
           '\\"French\\" \\"winners\\"")}}</tool_call>')
    assert parse_tool_call(raw) == ("isearch_s", {"query": '#combine("24 Hours of Spa" "French" "winners")'})
