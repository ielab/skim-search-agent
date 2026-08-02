"""Declared-location resolution — the workspace-agnostic localization piece of the unified
agent loop (see agent_search/agent/loop.py). No model. `parse_tool_call` itself is specced
in tests/test_actions_parser.py; no parser tests are duplicated here.

The SWE-agent-style multi-tool ACI (find_file/search_dir/open/goto/scroll/search_bql, the
`Workspace` class in agent_search/agent/tools/workspace.py) this file used to exercise was
removed (~Jul 3-4); the code arm's tool surface is now CodeFixWorkspace (search -> fetch,
<fix> terminal) and the grep baseline is GrepReadWorkspace (grep -> read, <fix> terminal) —
see agent_search/agent/tools/code_fix.py and code_grep.py. `run_episode` over those
workspaces (search->fetch->fix, fix_guard rejection, etc.) is already covered end-to-end in
tests/test_agent_loop.py, and their own tool contracts in tests/test_code_fix_tools.py /
tests/test_code_grep_tools.py, so those episode-level tests are not re-created here."""
from agent_search.agent.loop import resolve_locations
from agent_search.corpus.units import units_from_python_source

SRC = (
    "def create_session_token(user):\n"
    "    token = make_token(user)\n"
    "    return token\n"
    "\n"
    "def make_token(user):\n"
    "    return str(user)\n"
)


def _units():
    return units_from_python_source("app/session.py", SRC)


# --- resolve_locations -------------------------------------------------------

def test_resolve_path_and_function():
    assert resolve_locations(["app/session.py:make_token"], _units()) == \
        ["app/session.py::make_token"]


def test_resolve_bare_symbol_suffix():
    assert resolve_locations(["make_token"], _units()) == ["app/session.py::make_token"]


def test_resolve_dedup_and_order():
    got = resolve_locations(["make_token", "create_session_token", "make_token"], _units())
    assert got == ["app/session.py::make_token", "app/session.py::create_session_token"]


def test_resolve_file_only_returns_units_in_file():
    got = resolve_locations(["app/session.py"], _units())
    assert set(got) == {"app/session.py::create_session_token", "app/session.py::make_token"}
