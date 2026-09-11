"""KeywordPolicy: the no-model stub that drives any toolset, never a hardcoded condition
name, so check_conditions.py / trace_agent_chat.py exercise a whole arm with no LLM.

Walks each arm's own script: search->fetch (the method), bm25_search->visit (retrieve-then-
visit), grep->read (the code baseline), or bash->read (the DCI baseline), then the arm's
terminal (<fix> for code, <answer> for docs)."""
from agent_search.agent.loop import Task, run_episode
from agent_search.agent.policies import KeywordPolicy
from agent_search.corpus.units import units_from_documents, units_from_python_source
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.bash.tool import Bash
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.fetch_code.tool import FetchCode, SearchCode
from agent_search.tools.grep.tool import Grep
from agent_search.tools.read.tool import Read
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_bql.tool import SearchBql
from agent_search.tools.visit.tool import Visit
from agent_search.retrievers.bql.executor import StructuralExecutor
from tests.lucene_support import build_lucene_bql, build_pyserini, require_jvm

require_jvm()

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


def _toolbox(tools, units, engines=None, files=None):
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    bound = [t.bind(state, units, ubyid, engines or {}, files=files) for t in tools]
    return ToolBox(bound, state)


# --- code arms: search->fetch (method) and grep->read (baseline) walk to a <fix> --------

def test_search_fetch_stub_walks_to_a_fix():
    units = _code_units()
    files = {"auth/session.py": SRC}
    ex = StructuralExecutor(units)
    ws = _toolbox([SearchCode(name="search"), FetchCode(name="fetch")], units,
                 engines={"bql_plain": ex}, files=files)
    policy = KeywordPolicy(("search", "fetch"))
    traj = run_episode(policy, Task("t", "session token expiry"), ws, units, max_steps=5)
    assert traj.stopped_reason == "fix"
    assert [s.name for s in traj.steps][:2] == ["search", "fetch"]


def test_grep_read_stub_walks_to_a_fix():
    units = _code_units()
    files = {"auth/session.py": SRC}
    ws = _toolbox([Grep(name="grep"), Read(name="read", source="repo")], units, files=files)
    policy = KeywordPolicy(("grep", "read"))
    traj = run_episode(policy, Task("t", "session token expiry"), ws, units, max_steps=5)
    assert traj.stopped_reason == "fix"
    assert [s.name for s in traj.steps][:2] == ["grep", "read"]
    assert "auth/session.py" in traj.fix_text


def test_grep_read_stub_never_calls_search_or_fetch():
    """Regression: the FIRST move must be a grep (or a graceful <answer>), never a hardcoded
    'search' call the toolbox doesn't understand."""
    units = _code_units()
    policy = KeywordPolicy(("grep", "read"))
    raw = policy.propose(Task("t", "session token expiry"), [])
    assert '"name":"grep"' in raw or '"name": "grep"' in raw
    assert "search" not in raw


# --- doc arms: search->fetch (method), bm25_search->visit, bash->read (DCI) -------------

def test_doc_search_fetch_stub_walks_to_an_answer():
    units = _doc_units()
    ex = build_lucene_bql(units)
    ws = _toolbox([SearchBql(name="search"), Fetch(name="fetch")], units, engines={"bql": ex})
    policy = KeywordPolicy(("search", "fetch"))
    traj = run_episode(policy, Task("t", "harbor festival history"), ws, units,
                       max_steps=5, domain="general")
    assert traj.stopped_reason == "answer"


def test_bm25_visit_stub_walks_to_an_answer():
    units = _doc_units()
    engine = build_pyserini(units)
    ws = _toolbox([SearchBm25(name="bm25_search"), Visit(name="visit")], units,
                 engines={"bm25": engine})
    policy = KeywordPolicy(("bm25_search", "visit"))
    traj = run_episode(policy, Task("t", "harbor festival history"), ws, units,
                       max_steps=5, domain="general")
    assert traj.stopped_reason == "answer"
    assert [s.name for s in traj.steps][:2] == ["bm25_search", "visit"]


def test_bash_read_stub_walks_to_an_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SEARCH_DCI_CACHE", str(tmp_path))
    units = _doc_units()
    ws = _toolbox([Bash(), Read()], units)
    policy = KeywordPolicy(("bash", "read"))
    traj = run_episode(policy, Task("t", "harbor festival history"), ws, units,
                       max_steps=5, domain="general")
    assert traj.stopped_reason == "answer"
    assert [s.name for s in traj.steps][:2] == ["bash", "read"]


def test_bash_read_stub_never_calls_search():
    """Regression: the FIRST move must be a bash call, never a hardcoded 'search' the
    bash/read toolbox rejects as unknown."""
    policy = KeywordPolicy(("bash", "read"))
    raw = policy.propose(Task("t", "harbor festival history"), [])
    assert '"name":"bash"' in raw or '"name": "bash"' in raw
    assert '"name":"search"' not in raw and '"name": "search"' not in raw
