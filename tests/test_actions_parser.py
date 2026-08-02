"""Lenient tool-call parsing: the model frequently truncates a call (generation
cut off, or a dropped brace) so the JSON is near-valid but not quite. Measurement
on real trajectories: ~20% of ALL agent steps were being dropped as action "none"
because a strict json.loads / balanced-object scan can't recover a call with e.g.
one missing closing brace. ``_repair_load`` (string-literal-aware: closes brackets
left open, terminates an unclosed string, strips a dangling trailing comma, then
re-validates with json.loads) is the fallback that recovers these without ever
fabricating a call out of plain prose.
"""
from agent_search.agent.actions import _repair_load, parse_tool_call


# --- _repair_load directly ----------------------------------------------------

def test_repair_missing_closing_brace():
    # the exact failure mode from the bug report: one missing closing '}'.
    raw = '{"name":"fetch","arguments":{"specs":[[2,"History"]]}'
    assert _repair_load(raw) == {"name": "fetch", "arguments": {"specs": [[2, "History"]]}}


def test_repair_trailing_comma():
    raw = '{"name":"grep","arguments":{"query":"foo",}'
    assert _repair_load(raw) == {"name": "grep", "arguments": {"query": "foo"}}


def test_repair_unterminated_string():
    raw = '{"name":"grep","arguments":{"query":"unterminated'
    assert _repair_load(raw) == {"name": "grep", "arguments": {"query": "unterminated"}}


def test_repair_nested_arrays():
    raw = '{"name":"fetch","arguments":{"specs":[[2,"History"],[3,"Geo'
    assert _repair_load(raw) == \
        {"name": "fetch", "arguments": {"specs": [[2, "History"], [3, "Geo"]]}}


def test_repair_fully_valid_call_unchanged():
    raw = '{"name":"open","arguments":{"path":"a.py","line":5}}'
    assert _repair_load(raw) == {"name": "open", "arguments": {"path": "a.py", "line": 5}}


def test_repair_returns_none_for_garbage():
    assert _repair_load("just some prose, not json at all") is None
    assert _repair_load("") is None
    assert _repair_load(None) is None


def test_repair_returns_none_for_mid_key_truncation():
    # cut off before a colon/value ever appeared for the trailing key - nothing
    # sane to fabricate, so repair must not guess a value.
    assert _repair_load('{"name":"fetch","argum') is None


# --- end-to-end through parse_tool_call ---------------------------------------

def test_parse_recovers_missing_closing_brace_in_tool_call_block():
    raw = '<tool_call>{"name":"fetch","arguments":{"specs":[[2,"History"]]}</tool_call>'
    assert parse_tool_call(raw) == ("fetch", {"specs": [[2, "History"]]})


def test_parse_recovers_missing_closing_brace_and_missing_close_tag():
    # generation cut off entirely: no closing '}' AND no closing </tool_call>.
    raw = '<tool_call>{"name":"fetch","arguments":{"specs":[[2,"History"]]}'
    assert parse_tool_call(raw) == ("fetch", {"specs": [[2, "History"]]})


def test_parse_recovers_bare_json_missing_closing_brace_no_wrapper():
    raw = '{"name":"fetch","arguments":{"specs":[[2,"History"]]}'
    assert parse_tool_call(raw) == ("fetch", {"specs": [[2, "History"]]})


def test_parse_valid_call_still_parses_the_same():
    assert parse_tool_call(
        '<tool_call>{"name": "open", "arguments": {"path": "a.py", "line": 5}}'
        '</tool_call>') == ("open", {"path": "a.py", "line": 5})


def test_parse_quoted_decoy_in_think_does_not_preempt_a_valid_real_call():
    # regression guard: repair must not change the pre-existing "last (valid) call
    # wins over an earlier quoted decoy" behavior.
    raw = ('<think>maybe {"name":"grep","arguments":{"query":"x"}}</think>\n'
           '<tool_call>{"name":"open","arguments":{"path":"a.py"}}</tool_call>')
    assert parse_tool_call(raw) == ("open", {"path": "a.py"})


def test_parse_think_only_prose_mention_returns_none():
    # a <think> that merely narrates a call in prose (never valid/repairable JSON,
    # no <tool_call> ever opened) must still return None - repair must not
    # hallucinate a call out of a plan that was never actually emitted.
    raw = '<think>I should call fetch with specs=[[2,"History"]] next</think>'
    assert parse_tool_call(raw) is None


def test_parse_none_when_absent():
    assert parse_tool_call("just thinking, no call") is None
    assert parse_tool_call(None) is None
    assert parse_tool_call("") is None
