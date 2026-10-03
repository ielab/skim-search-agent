"""Backbones that write tool calls in their own text format (agent_search.agent.call_formats)."""
import pytest

from agent_search.agent.actions import parse_tool_call
from agent_search.agent.call_formats import call_format, parse_alternate


GLM = ('<think>Look for the ship first.</think>I will search.\n'
       '<tool_call>search<arg_key>query</arg_key><arg_value>Copacabana Belgian ship</arg_value></tool_call>')
GLM_LIST = '<tool_call>search<arg_key>query</arg_key><arg_value>["a b", "c"]</arg_value></tool_call>'
GLM_TWO = ('<tool_call>get_document<arg_key>docid</arg_key><arg_value>4021</arg_value>'
           '<arg_key>note</arg_key><arg_value>the 1952 book</arg_value></tool_call>')
QWEN = ('<think>x</think>\n<tool_call>\n<function=search>\n<parameter=query>\nCopacabana Belgian ship\n'
        '</parameter>\n</function>\n</tool_call>')
QWEN_OPEN = '<tool_call><function=get_document><parameter=docid>4021'      # cut off mid-call


def test_glm_calls():
    fmt = call_format("glm")
    assert fmt.parse(GLM) == ("search", {"query": "Copacabana Belgian ship"})
    assert fmt.parse(GLM_LIST) == ("search", {"query": ["a b", "c"]})
    assert fmt.parse(GLM_TWO) == ("get_document", {"docid": 4021, "note": "the 1952 book"})


def test_qwen_xml_calls():
    fmt = call_format("qwen_xml")
    assert fmt.parse(QWEN) == ("search", {"query": "Copacabana Belgian ship"})
    assert fmt.parse(QWEN_OPEN) == ("get_document", {"docid": 4021})


MCP = ('<think>plan</think>\n\n<tool_name>search</tool_name>\n<arguments>\n{"query": "Ka Hao 2021"}\n </>\n'
       '<tool_name>search</tool_name>\n<arguments>\n{"query": "\\"convenor\\" panel 2018"}\n </')
MCP_WRAPPED = ('<use_mcp_tool>\n<server_name>tool-search</server_name>\n<tool_name>get_document</tool_name>\n'
               '<arguments>\n{"docid": "4021"}\n</arguments>\n</use_mcp_tool>')


def test_mcp_calls_take_the_last_one_even_with_broken_closing_tags():
    fmt = call_format("mcp")
    assert fmt.parse(MCP) == ("search", {"query": '"convenor" panel 2018'})
    assert fmt.parse(MCP_WRAPPED) == ("get_document", {"docid": "4021"})


MIRO_BARE = ('<tool>\n {"name": "search", "arguments": {"query": "research group 2009"}}\n </>\n'
             '<tool>\n {"name": "search", "arguments": {"query": "Routledge 2018 co-edited"}}\n </>')
QWEN36 = '<tool_call>\n{"name": search", "arguments": {"query": "\\"hat shop\\"", "k": 10}}\n</tool_call>'


def test_mcp_reads_a_bare_named_object():
    assert call_format("mcp").parse(MIRO_BARE) == ("search", {"query": "Routledge 2018 co-edited"})
    # the closing brace of the object is often missing before " </>"
    cut = '<tool>\n {"name": "search", "arguments": {"query": "a"}\n </>\n<tool>\n {"name": "search", "arguments": {"query": "b"}\n </>'
    assert call_format("mcp").parse(cut) == ("search", {"query": "b"})


def test_lenient_json_repairs_an_unquoted_tool_name():
    assert parse_tool_call(QWEN36) is None
    assert call_format("lenient_json").parse(QWEN36) == ("search", {"query": '"hat shop"', "k": 10})


def test_a_format_never_reads_a_call_quoted_in_thinking():
    quoted = "<think><tool_call>search<arg_key>query</arg_key><arg_value>x</arg_value></tool_call></think>done"
    assert call_format("glm").parse(quoted) is None


def test_json_stays_the_default(monkeypatch):
    monkeypatch.delenv("AGENT_TOOL_CALL_FORMAT", raising=False)
    assert parse_alternate(GLM) is None and call_format("json") is None
    monkeypatch.setenv("AGENT_TOOL_CALL_FORMAT", "glm")
    assert parse_alternate(GLM) == ("search", {"query": "Copacabana Belgian ship"})
    # the JSON parser finds nothing in either format, so the loop's fallback is what reads them
    assert parse_tool_call(GLM) is None and parse_tool_call(QWEN) is None


def test_unknown_format_is_an_error():
    with pytest.raises(ValueError):
        call_format("hermes2")
