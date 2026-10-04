"""MiroThinker's tools, its history format and its task."""
import json

from agent_search.agent.loop import Step, Task
from agent_search.agent.policies import AgentPolicy, PlainResultsPolicy, policy_class
from agent_search.corpus.units import units_from_documents
from agent_search.strategies.conditions import CONDITIONS
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.google_search.tool import GoogleSearch
from agent_search.tools.scrape_extract import tool as scrape_extract
from agent_search.tools.scrape_extract.tool import ScrapeExtract

DOCS = [{"_id": str(i), "title": f"Page {i}", "text": f"Text of page {i} about treaties."} for i in range(1, 26)]


class _Pool:
    def top_k_doc_ids(self, query, k=None):
        return [str(i) for i in range(1, 26)][:k]


def _box():
    units = units_from_documents(DOCS)
    state = EpisodeState(question="q")
    ubyid = {u.doc_id: u for u in units}
    tools = [GoogleSearch(ranking="dense").bind(state, units, ubyid, {"dense": _Pool()}),
             ScrapeExtract(refusals={"run_python_code": "[ERROR]: E2B_API_KEY is not set."}).bind(state, units, ubyid, {})]
    return ToolBox(tools, state), state


def test_search_returns_the_organic_results_as_json_and_pages():
    box, state = _box()
    data = json.loads(box.run("google_search", {"q": "treaties"}))
    assert [r["position"] for r in data["organic"]] == list(range(1, 11))
    assert data["organic"][0] == {"title": "Page 1", "link": "https://corpus/1",
                                  "snippet": data["organic"][0]["snippet"], "position": 1}
    assert data["searchParameters"]["q"] == "treaties" and state.last_hits == [str(i) for i in range(1, 11)]
    second = json.loads(box.run("google_search", {"q": "treaties", "num": 5, "page": 3}))
    assert [r["link"] for r in second["organic"]] == [f"https://corpus/{i}" for i in range(11, 16)]
    assert json.loads(box.run("google_search", {}))["success"] is False


def test_scrape_returns_the_readers_extraction_in_the_official_wrapper(monkeypatch):
    seen = {}

    def reader(content, wanted):
        seen["content"], seen["wanted"] = content, wanted
        return "Page 3 is about treaties."
    monkeypatch.setattr(scrape_extract, "extract", reader)
    box, state = _box()
    out = json.loads(box.run("scrape_and_extract_info", {"url": "https://corpus/3", "info_to_extract": "what is it about"}))
    assert list(out) == ["success", "url", "extracted_info", "error", "scrape_stats", "model_used", "tokens_used"]
    assert out["success"] is True and out["extracted_info"] == "Page 3 is about treaties." and out["error"] == ""
    assert out["scrape_stats"]["all_content_displayed"] is True and seen["wanted"] == "what is it about"
    assert "Text of page 3" in seen["content"] and state.reads == ["3"]
    missing = json.loads(box.run("scrape_and_extract_info", {"url": "https://elsewhere/9", "info_to_extract": "x"}))
    assert missing["success"] is False and missing["extracted_info"] == "" and "Scraping failed" in missing["error"]


def test_a_failed_reader_is_reported_in_the_wrapper(monkeypatch):
    def broken(content, wanted):
        raise RuntimeError("down")
    monkeypatch.setattr(scrape_extract, "extract", broken)
    out = json.loads(_box()[0].run("scrape_and_extract_info", {"url": "https://corpus/3", "info_to_extract": "x"}))
    assert out["success"] is False and "Unexpected error during LLM API call: down" in out["error"]
    assert out["scrape_stats"]["line_count"] >= 1


def test_only_the_newest_five_results_stay_in_the_prompt():
    assert policy_class("plain_results") is PlainResultsPolicy and policy_class("deepresearch_tool_call") is AgentPolicy
    steps = [Step(name="google_search", args={}, observation=f"result {i}", raw_output=f"reply {i}") for i in range(8)]
    policy = PlainResultsPolicy(generate=None, system="SYSTEM", user_template="{question}", system_verbatim=True)
    msgs = policy.build_messages(Task(task_id="q", query="Who?"), steps)
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == [f"reply {i}" for i in range(8)]
    results = [m["content"] for m in msgs[2:] if m["role"] == "user"]
    assert results == ["Tool result is omitted to save tokens."] * 3 + [f"result {i}" for i in range(3, 8)]


def test_the_condition_renders_miroflows_system_prompt_and_user_turn():
    cond = CONDITIONS["research_mirothinker_dense"]
    system = cond.render()
    assert system.startswith("In this environment you have access to a set of tools")
    assert "## Server name: search_and_scrape_webpage\n### Tool name: google_search" in system
    assert "### Tool name: scrape_and_extract_info" in system and system.count("### Tool name:") == 7
    assert "Today is: 20" in system and "{{" not in system
    assert cond.task.loop_user_template.endswith("wrap the final answer in \\boxed{}.")
    assert cond.task.message_format == "plain_results" and cond.task.terminal == "text"
