"""agent_search.research / build_agent: the programmatic path through the same objects the
command line uses, with the caller's own model callable."""
import agent_search
from agent_search import research
from agent_search.api import build_agent
from lucene_support import index_root, require_jvm

require_jvm()

DOCS = [
    {"_id": "d_guadalupe", "title": "Treaty of Guadalupe Hidalgo",
     "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
    {"_id": "d_paris", "title": "Treaty of Paris (1898)",
     "text": "The 1898 Treaty of Paris ended the Spanish-American War."},
    {"_id": "d_adams", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
]
QUESTION = "Which treaty ended the Mexican-American War, and in what year?"


def test_research_with_a_scripted_model_callable():
    calls = []

    def generate(messages):
        calls.append(messages)
        if len(calls) == 1:
            return '<tool_call>{"name": "search_s", "arguments": {"query": "Guadalupe Hidalgo[title]"}}</tool_call>'
        if len(calls) == 2:
            return '<tool_call>{"name": "fetch_s", "arguments": {"specs": [[1, "(intro)"]]}}</tool_call>'
        return "<answer>Treaty of Guadalupe Hidalgo, 1848</answer>"

    result = research(QUESTION, DOCS, strategy="sieve_bm25", generate=generate, max_steps=6, index_root=index_root())
    assert result.answer == "Treaty of Guadalupe Hidalgo, 1848"
    assert result.stopped == "answer"
    assert result.ranking and result.ranking[0] == "d_guadalupe"
    assert [s["action"] for s in result.steps] == ["search_s", "fetch_s", "answer"]
    assert "Guadalupe" in result.observations[1]
    assert result.usage["llm_calls"] == 3
    assert calls[0][0]["role"] == "system" and QUESTION in calls[0][1]["content"]


def test_research_without_a_model_runs_the_scripted_policy():
    result = research(QUESTION, DOCS, strategy="search_visit", max_steps=4, index_root=index_root())
    assert result.condition == "agent_search_visit"
    assert result.steps and result.usage["llm_calls"] >= 1


def test_build_agent_rejects_retrieval_only_floors():
    import pytest
    with pytest.raises(ValueError):
        build_agent("bm25")


def test_lazy_package_exports():
    assert callable(agent_search.research) and callable(agent_search.build_agent)
    assert "sieve_bm25" in agent_search.STRATEGIES
    assert isinstance(agent_search.__version__, str)
