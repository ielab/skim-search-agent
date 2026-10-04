"""OpenSeeker's tools: a batched search and a visit that reads for a goal."""
from agent_search.corpus.units import units_from_documents
from agent_search.strategies.conditions import CONDITIONS
from agent_search.strategies.queries_visit import queries_visit_dense
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.search_queries.tool import SearchQueries
from agent_search.tools.visit_goal import tool as visit_goal
from agent_search.tools.visit_goal.tool import VisitGoal

DOCS = [
    {"_id": "1", "title": "Treaty of Guadalupe Hidalgo", "text": "Signed in 1848, it ended the Mexican-American War."},
    {"_id": "2", "title": "Rio Grande", "text": "The border river was named in the treaty."},
]


class _Pool:
    def top_k_doc_ids(self, query, k=None):
        return ["2", "1"] if "river" in query else ["1", "2"]


def _box():
    units = units_from_documents(DOCS)
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    tools = [SearchQueries(ranking="dense").bind(state, units, ubyid, {"dense": _Pool()}),
             VisitGoal(refusals={"run_command": "[ERROR]: E2B_API_KEY is not set."}).bind(state, units, ubyid, {})]
    return ToolBox(tools, state), state


def test_each_query_of_the_array_gets_its_own_block():
    box, state = _box()
    out = box.run("search", {"query": ["treaty 1848", "border river"]})
    first, second = out.split("\n---\n")
    assert first.startswith("### A Google search for 'treaty 1848' found 2 results:\n\n"
                            "1. [Treaty of Guadalupe Hidalgo](https://corpus/1)\n\n")
    assert second.startswith("### A Google search for 'border river' found 2 results:\n\n1. [Rio Grande](https://corpus/2)")
    assert state.last_hits == ["1", "2"] and state.previous_queries == ["treaty 1848", "border river"]
    assert box.run("search", {"query": "one string"}).startswith("### A Google search for 'one string'")
    assert box.run("search", {}).startswith("[Search] Invalid request format")


def test_visit_returns_what_the_reader_extracts_for_the_goal(monkeypatch):
    seen = {}

    def reader(content, goal):
        seen["content"], seen["goal"] = content, goal
        return {"rational": "r", "evidence": "It ended the war.", "summary": "The treaty ended the war in 1848."}
    monkeypatch.setattr(visit_goal, "read_with_goal", reader)
    box, state = _box()
    out = box.run("visit", {"url": "https://corpus/1", "goal": "when the war ended"})
    assert out == ("The useful information in https://corpus/1 for user goal when the war ended as follows: \n\n"
                   "Evidence in page: \nIt ended the war.\n\nSummary: \nThe treaty ended the war in 1848.")
    assert seen["goal"] == "when the war ended" and "Signed in 1848" in seen["content"]
    assert state.reads == ["1"]
    two = box.run("visit", {"url": ["https://corpus/1", "https://corpus/2"], "goal": "g"})
    assert two.count("\n---\n") == 1 and state.reads == ["1", "2"]


def test_visit_reports_a_page_it_cannot_read(monkeypatch):
    monkeypatch.setattr(visit_goal, "read_with_goal", lambda content, goal: {})
    box, _ = _box()
    for url in ("https://elsewhere/9", "https://corpus/1"):       # no such page; the reader failed
        out = box.run("visit", {"url": url, "goal": "g"})
        assert "The provided webpage content could not be accessed." in out
    assert box.run("visit", {"url": "https://corpus/1"}).startswith("[Visit] Invalid request format")


def test_a_sandbox_call_gets_the_official_answer_for_a_missing_key():
    box, _ = _box()
    assert box.run("run_command", {"command": "ls"}) == "[ERROR]: E2B_API_KEY is not set."


def test_the_condition_renders_the_official_system_prompt_and_user_turn():
    cond = CONDITIONS["research_openseeker_dense"]
    system = cond.render()
    assert system.startswith("You are a tool-augmented QA agent. Cleverly leverage appropriate tools")
    assert system.count('"type": "function"') == 7 and "{{" not in system
    assert cond.task.loop_user_template.endswith("wrap the final answer in \\\\boxed{}.")
    assert [t.name for t in queries_visit_dense.tools] == ["search", "visit"]
