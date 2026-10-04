"""QUEST's search tool and task."""
from agent_search.corpus.units import units_from_documents
from agent_search.strategies.conditions import CONDITIONS
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.search_passages.tool import SearchPassages

DOCS = [
    {"_id": "7", "title": "Treaty of Guadalupe Hidalgo", "text": "Signed in 1848, it ended the Mexican-American War."},
    {"_id": "9", "title": "Rio Grande", "text": "The border river was named in the treaty."},
]


class _Pool:
    def search_scored(self, query, k=None):
        return [("7", 0.61234), ("9", 0.5)][:k]


def _box():
    units = units_from_documents(DOCS)
    state = EpisodeState(question="q")
    tool = SearchPassages(ranking="dense").bind(state, units, {u.doc_id: u for u in units}, {"dense": _Pool()})
    return ToolBox([tool], state), state


def test_every_query_runs_and_returns_scored_passages():
    box, state = _box()
    out = box.run("search", {"query": ["treaty", "river"]})
    first = out.split("\n\nA search for 'river'")[0]
    assert first.startswith("A search for 'treaty' found 2 results:\n\n"
                            "Title: Document 7\nLink: bm25://7\nScore: 0.6123\nSnipptes: ")
    assert "Signed in 1848" in first and "Title: Document 9\nLink: bm25://9\nScore: 0.5000\n" in first
    assert out.count("A search for '") == 2
    assert state.last_hits == ["7", "9"] and state.previous_queries == ["treaty", "river"]
    assert box.run("search", {"query": []}) == "[Tool Error] Search query cannot be empty."
    assert box.run("search", {"query": "one string"}).startswith("A search for 'one string' found 2 results:")


def test_the_condition_renders_the_official_system_prompt():
    cond = CONDITIONS["research_quest_dense"]
    system = cond.render()
    assert system.startswith("You are a deep research assistant.")
    assert '{"type": "function", "function": {"name": "search"' in system
    assert "# CRITICAL - visit tool is DISABLED in this run" in system
    assert "Current date: 20" in system and "{{" not in system
    declared = cond.strategy.tools[0].declaration()
    import json
    line = next(l for l in system.splitlines() if l.startswith('{"type": "function", "function": {"name": "search"'))
    assert json.loads(line)["function"] == declared
    assert cond.task.loop_user_template == "{question}" and cond.task.terminal == "answer"
