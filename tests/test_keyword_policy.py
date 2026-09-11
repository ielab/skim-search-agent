"""KeywordPolicy: the no-model stub that drives any toolset, never a hardcoded condition
name, so check_conditions.py / trace_agent_chat.py exercise a whole arm with no LLM.

Walks each arm's own script: search->fetch (the method), bm25_search->visit (retrieve-then-
visit), grep->read (the code baseline), or bash->read (the DCI baseline), then the arm's
terminal (<fix> for code, <answer> for docs)."""
from agent_search.agent.loop import Task, run_episode
from agent_search.agent.policies import KeywordPolicy
from agent_search.legacy.workspaces.code_fix import CodeFixWorkspace
from agent_search.legacy.workspaces.code_grep import GrepReadWorkspace
from agent_search.legacy.workspaces.doc_dci import DciWorkspace
from agent_search.legacy.workspaces.search_visit import Bm25Visit
from agent_search.legacy.workspaces.sieve import DocSearchFetch
from agent_search.corpus.units import units_from_documents, units_from_python_source

SRC = (
    "def create_session_token(user):\n"
    "    token = make_token(user)\n"
    "    return token\n"
    "\n"
    "def make_token(user):\n"
    "    return str(user)\n"
)
DOCS = [{"_id": "d1", "title": "Doc One",
        "text": "Harbor Festival is annual.\n\n## History\nFounded in 1897."}]


def _code_units():
    return units_from_python_source("auth/session.py", SRC)


def _doc_units():
    return units_from_documents(DOCS)


# --- code arms: search->fetch (method) and grep->read (baseline) walk to a <fix> --------

def test_search_fetch_stub_walks_to_a_fix():
    units = _code_units()
    ws = CodeFixWorkspace(units, {"auth/session.py": SRC})
    policy = KeywordPolicy(("search", "fetch"))
    traj = run_episode(policy, Task("t", "session token expiry"), ws, units, max_steps=5)
    assert traj.stopped_reason == "fix"
    assert [s.name for s in traj.steps][:2] == ["search", "fetch"]


def test_grep_read_stub_walks_to_a_fix():
    units = _code_units()
    ws = GrepReadWorkspace(units, {"auth/session.py": SRC})
    policy = KeywordPolicy(("grep", "read"))
    traj = run_episode(policy, Task("t", "session token expiry"), ws, units, max_steps=5)
    assert traj.stopped_reason == "fix"
    assert [s.name for s in traj.steps][:2] == ["grep", "read"]
    assert "auth/session.py" in traj.fix_text


def test_grep_read_stub_never_calls_search_or_fetch():
    """Regression: the FIRST move must be a grep (or a graceful <answer>), never a hardcoded
    'search' call the workspace doesn't understand."""
    units = _code_units()
    ws = GrepReadWorkspace(units, {"auth/session.py": SRC})
    policy = KeywordPolicy(("grep", "read"))
    raw = policy.propose(Task("t", "session token expiry"), [])
    assert '"name":"grep"' in raw or '"name": "grep"' in raw
    assert "search" not in raw


# --- doc arms: search->fetch (method), bm25_search->visit, bash->read (DCI) -------------

def test_doc_search_fetch_stub_walks_to_an_answer():
    units = _doc_units()
    ws = DocSearchFetch(units)
    policy = KeywordPolicy(("search", "fetch"))
    traj = run_episode(policy, Task("t", "harbor festival history"), ws, units,
                       max_steps=5, domain="general")
    assert traj.stopped_reason == "answer"


def test_bm25_visit_stub_walks_to_an_answer():
    units = _doc_units()
    ws = Bm25Visit(units)
    policy = KeywordPolicy(("bm25_search", "visit"))
    traj = run_episode(policy, Task("t", "harbor festival history"), ws, units,
                       max_steps=5, domain="general")
    assert traj.stopped_reason == "answer"
    assert [s.name for s in traj.steps][:2] == ["bm25_search", "visit"]


def test_bash_read_stub_walks_to_an_answer():
    units = _doc_units()
    ws = DciWorkspace(units)
    policy = KeywordPolicy(("bash", "read"))
    traj = run_episode(policy, Task("t", "harbor festival history"), ws, units,
                       max_steps=5, domain="general")
    assert traj.stopped_reason == "answer"
    assert [s.name for s in traj.steps][:2] == ["bash", "read"]


def test_bash_read_stub_never_calls_search():
    """Regression: the FIRST move must be a bash call, never a hardcoded 'search' the
    DciWorkspace rejects as unknown."""
    units = _doc_units()
    policy = KeywordPolicy(("bash", "read"))
    raw = policy.propose(Task("t", "harbor festival history"), [])
    assert '"name":"bash"' in raw or '"name": "bash"' in raw
    assert '"name":"search"' not in raw and '"name": "search"' not in raw
