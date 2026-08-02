"""ADDITIVE-only coverage for sdk_driver's two new toolsets: `research_v2`'s ("search_v2",
"fetch_v2") and `research_indri`'s ("isearch", "fetch") — see agent_search/agent/tools/doc_research.py
(search_v2/fetch_v2 alias search/fetch on DocSearchFetch.run) and agent_search/agent/tools/doc_indri.py
(isearch on IndriFetchWorkspace.run). Before this test's corresponding fix, `_tools_for` didn't know
these tool names, so an OpenAI-backbone SDK run over either workspace got ZERO tools.

CPU-only, no network: a tiny fake workspace (tools attribute + run(name, args) recording calls,
returning a canned string) stands in for the real DocSearchFetch/IndriFetchWorkspace. Each SDK
`function_tool`'s underlying callable is invoked via its `.on_invoke_tool(ctx, json_args)` — the
`agents` library's own invocation path — never a real model/API call.
"""
from __future__ import annotations

import asyncio
import json

from agents.tool_context import ToolContext

from agent_search.agent.sdk_driver import (
    _INSTR_BQL, _INSTR_BQL_VISIT, _instructions_for, _tools_for)


class FakeWS:
    """Stands in for DocSearchFetch / IndriFetchWorkspace: `tools` selects which SDK function
    tools `_tools_for` builds; `run` records every (name, args) call and returns a canned
    string keyed by name (so a test can assert both the ws.run call AND the tool's return)."""

    def __init__(self, tools):
        self.tools = tools
        self.calls: list[tuple[str, dict]] = []

    def run(self, name: str, args: dict) -> str:
        self.calls.append((name, dict(args or {})))
        return f"OBS:{name}"


def _invoke(tool, **kwargs):
    """Call a `@function_tool`-wrapped function the way the Agents SDK Runner does: build a
    minimal ToolContext + JSON-encoded args, then run the async `on_invoke_tool`."""
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="t1",
                       tool_arguments=json.dumps(kwargs))
    return asyncio.run(tool.on_invoke_tool(ctx, json.dumps(kwargs)))


def _by_name(tools):
    return {t.name: t for t in tools}


# --- name-set 1: the pre-existing ("search", "fetch") — must be unaffected --------------

def test_search_fetch_tools_unaffected_by_v2_additions():
    ws = FakeWS(("search", "fetch"))
    tools, trace = _tools_for(ws)
    names = {t.name for t in tools}
    assert names == {"search", "fetch"}
    by = _by_name(tools)

    out = _invoke(by["search"], query="alpha[body]", k=2)
    assert ws.calls[-1] == ("search", {"query": "alpha[body]", "k": 2})
    assert out == "OBS:search"
    assert trace[-1][0] == "search"

    out2 = _invoke(by["fetch"], doc="1", section="History")
    assert ws.calls[-1] == ("fetch", {"specs": [["1", "History"]]})
    assert out2 == "OBS:fetch"
    assert trace[-1][0] == "fetch"


# --- name-set 2: research_v2's ("search_v2", "fetch_v2") --------------------------------

def test_search_v2_fetch_v2_tools():
    ws = FakeWS(("search_v2", "fetch_v2"))
    tools, trace = _tools_for(ws)
    names = {t.name for t in tools}
    assert names == {"search_v2", "fetch_v2"}
    by = _by_name(tools)

    out = _invoke(by["search_v2"], query="foo[title] AND date[1980..1989]", k=3)
    assert ws.calls[-1] == ("search_v2", {"query": "foo[title] AND date[1980..1989]", "k": 3})
    assert out == "OBS:search_v2"
    assert trace[-1][0] == "search_v2"
    assert trace[-1][1] == {"query": "foo[title] AND date[1980..1989]", "k": 3}

    out2 = _invoke(by["fetch_v2"], specs=[["1", "History"], ["2", "infobox"]])
    assert ws.calls[-1] == ("fetch_v2", {"specs": [["1", "History"], ["2", "infobox"]]})
    assert out2 == "OBS:fetch_v2"
    assert trace[-1][0] == "fetch_v2"


# --- name-set 3: research_indri's ("isearch", "fetch") ----------------------------------

def test_isearch_fetch_tools():
    ws = FakeWS(("isearch", "fetch"))
    tools, trace = _tools_for(ws)
    names = {t.name for t in tools}
    assert names == {"isearch", "fetch"}
    by = _by_name(tools)

    out = _invoke(by["isearch"], query="#combine(dog train)", k=4)
    assert ws.calls[-1] == ("isearch", {"query": "#combine(dog train)", "k": 4})
    assert out == "OBS:isearch"
    assert trace[-1][0] == "isearch"

    out2 = _invoke(by["fetch"], doc="1", section="History")
    assert ws.calls[-1] == ("fetch", {"specs": [["1", "History"]]})
    assert out2 == "OBS:fetch"
    assert trace[-1][0] == "fetch"


# --- name-set 4: research_bql_visit's ("search_bv", "visit_bv") — the {BQL search} x
# {whole-doc visit} factorial cell (doc_research.py's BqlVisitWorkspace) ------------------

def test_search_bv_visit_bv_tools():
    ws = FakeWS(("search_bv", "visit_bv"))
    tools, trace = _tools_for(ws)
    names = {t.name for t in tools}
    assert names == {"search_bv", "visit_bv"}
    by = _by_name(tools)

    out = _invoke(by["search_bv"], query="foo[title] AND date[1980..1989]", k=3)
    assert ws.calls[-1] == ("search_bv", {"query": "foo[title] AND date[1980..1989]", "k": 3})
    assert out == "OBS:search_bv"
    assert trace[-1][0] == "search_bv"
    assert trace[-1][1] == {"query": "foo[title] AND date[1980..1989]", "k": 3}

    out2 = _invoke(by["visit_bv"], doc="1")
    assert ws.calls[-1] == ("visit_bv", {"rank": "1"})
    assert out2 == "OBS:visit_bv"
    assert trace[-1][0] == "visit_bv"


# --- _instructions_for must tolerate the new tool names, still picking the BQL coaching ---

def test_instructions_for_tolerates_search_v2_fetch_v2():
    instr = _instructions_for(FakeWS(("search_v2", "fetch_v2")))
    assert _INSTR_BQL in instr


def test_instructions_for_tolerates_isearch():
    instr = _instructions_for(FakeWS(("isearch", "fetch")))
    assert _INSTR_BQL in instr


def test_instructions_for_picks_bql_visit_coaching_for_search_bv_visit_bv():
    instr = _instructions_for(FakeWS(("search_bv", "visit_bv")))
    assert _INSTR_BQL_VISIT in instr
    assert "search_bv" in instr and "visit_bv" in instr
