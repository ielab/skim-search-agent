"""OpenResearcher's browser tools: search, open and find over one page stack per episode."""
from agent_search.corpus.units import units_from_documents
from agent_search.strategies.conditions import CONDITIONS
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.browser_find.tool import BrowserFind
from agent_search.tools.browser_open.tool import BrowserOpen
from agent_search.tools.browser_pages import VIEW_TOKENS
from agent_search.tools.browser_search.tool import BrowserSearch

LONG = "\n".join(f"Line {i} of the long report about rivers." for i in range(400))
DOCS = [
    {"_id": "1", "title": "Treaty of Guadalupe Hidalgo", "text": "Signed in 1848, it ended the Mexican-American War.\n\nNicholas Trist negotiated it."},
    {"_id": "2", "title": "Rio Grande", "text": LONG + "\nThe border river was named in the treaty."},
    {"_id": "3", "title": "Gadsden Purchase", "text": "A later 1853 land purchase from Mexico."},
]


class _Pool:
    def __init__(self, order):
        self.order = order

    def top_k_doc_ids(self, query, k=None):
        return self.order[:k]

    def search(self, query, k):
        return self.order[:k]


def _box(order=("1", "2", "3"), ranking="dense"):
    units = units_from_documents(DOCS)
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    tools = [t.bind(state, units, ubyid, {ranking: _Pool(list(order))})
             for t in (BrowserSearch(ranking=ranking), BrowserOpen(), BrowserFind())]
    return ToolBox(tools, state), state


def test_search_lists_results_as_links_on_a_numbered_page():
    box, state = _box()
    out = box.run("browser.search", {"query": "treaty 1848"})
    assert out.startswith("[0] treaty 1848 (web-search://ts=")
    assert "**viewing lines [0 - " in out and "L2: # Search Results" in out
    assert "【0†Treaty of Guadalupe Hidalgo†corpus】" in out and "【2†Gadsden Purchase†corpus】" in out
    assert state.last_hits == ["1", "2", "3"] and list(state.seen) == ["1", "2", "3"]


def test_the_model_may_call_the_tools_without_the_namespace():
    box, _ = _box()
    assert box.run("search", {"query": "treaty"}).startswith("[0] treaty")
    assert box.run("open", {"id": 0}).startswith("[1] Treaty of Guadalupe Hidalgo (https://corpus/1)")


def test_open_follows_a_link_and_shows_the_document_from_the_top():
    box, state = _box()
    box.run("browser.search", {"query": "treaty"})
    out = box.run("browser.open", {"id": 0})
    assert out.startswith("[1] Treaty of Guadalupe Hidalgo (https://corpus/1)\n**viewing lines [0 - ")
    assert "L1: URL: https://corpus/1" in out and "L2: Signed in 1848" in out
    assert state.reads == ["1"]


def test_a_long_document_is_shown_one_window_at_a_time():
    box, _ = _box()
    box.run("browser.search", {"query": "rivers"})
    first = box.run("browser.open", {"id": 1})
    last_line = int(first.split("**viewing lines [0 - ")[1].split("]")[0])
    assert 0 < last_line < 400                                    # the window, not the page
    from agent_search.tokens import count_tokens
    assert count_tokens(first) <= VIEW_TOKENS + 64
    more = box.run("browser.open", {"cursor": 1, "loc": last_line + 1})
    assert more.startswith("[2] Rio Grande") and f"L{last_line + 1}: " in more
    exact = box.run("browser.open", {"cursor": 1, "loc": 10, "num_lines": 3})
    assert "**viewing lines [10 - 12] of " in exact and "L13:" not in exact


def test_open_takes_a_page_address_and_reports_a_bad_link_or_line():
    box, state = _box()
    assert box.run("browser.open", {"id": "https://corpus/3"}).startswith("[0] Gadsden Purchase")
    assert state.reads == ["3"]
    assert box.run("browser.open", {"id": 7}) == "Invalid link id `7`."
    assert "Cannot exceed page maximum of" in box.run("browser.open", {"loc": 999})
    assert "URL not found in corpus" in box.run("browser.open", {"id": "https://elsewhere/9"})
    assert _box()[0].run("browser.open", {"id": 0}) == "No pages to access!"


def test_find_lists_the_matching_lines_and_open_goes_to_a_match():
    box, _ = _box()
    box.run("browser.search", {"query": "rivers"})
    box.run("browser.open", {"id": 1})
    out = box.run("browser.find", {"pattern": "Border River"})        # case folded
    assert out.startswith("[2] Find results for text: `border river` in `Rio Grande`")
    assert "L0: # 【0†match at L402】" in out and "The border river was named" in out
    at = box.run("browser.open", {"id": 0})                            # just above the match
    assert at.startswith("[3] Rio Grande") and "**viewing lines [398 - " in at
    assert box.run("browser.find", {"pattern": "nowhere", "cursor": 1}).endswith(
        "No `find` results for pattern: `nowhere`")


def test_find_refuses_a_search_page():
    box, _ = _box()
    box.run("browser.search", {"query": "treaty"})
    assert box.run("browser.find", {"pattern": "treaty"}) == \
        "Cannot run `find` on search results page or find results page"


def test_the_condition_renders_the_models_own_system_prompt():
    system = CONDITIONS["research_openresearcher_dense"].render()
    assert system.startswith("You are a helpful assistant and harmless assistant.")
    assert "<name>browser.search</name>" in system and "<name>browser.find</name>" in system
    assert "Today's date: 20" in system and "{{" not in system
    assert "<function=example_function_name>" in system


def test_tool_messages_policy_sends_reasoning_and_call_as_fields(monkeypatch):
    from agent_search.agent.loop import Step, Task
    from agent_search.agent.policies import AgentPolicy, ToolMessagesPolicy
    monkeypatch.setenv("AGENT_TOOL_CALL_FORMAT", "qwen_xml")
    raw = ("I should search.\n</think>\n<tool_call>\n<function=browser.search>\n<parameter=query>\n"
           "treaty 1848\n</parameter>\n</function>\n</tool_call>")
    steps = [Step(name="browser.search", args={"query": "treaty 1848"}, observation="[0] page", raw_output=raw),
             Step(name="none", args={}, observation="ERROR: no tool call found.", raw_output="Thinking.\n</think>\nNo call here")]
    policy = ToolMessagesPolicy(generate=None, system="SYSTEM\n", user_template="{question}", system_verbatim=True)
    msgs = policy.build_messages(Task(task_id="q", query="Who?"), steps)
    assert msgs[0] == {"role": "system", "content": "SYSTEM"} and msgs[1] == {"role": "user", "content": "Who?"}
    assert msgs[2]["content"] == "" and msgs[2]["reasoning"] == "I should search."
    call = msgs[2]["tool_calls"][0]
    assert call["function"] == {"name": "browser.search", "arguments": '{"query": "treaty 1848"}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": call["id"], "content": "[0] page"}
    assert msgs[4]["content"] == "No call here" and "tool_calls" not in msgs[4] and msgs[5]["role"] == "user"
    # the default policy is unchanged: the reply as written, the observation in a user turn
    plain = AgentPolicy(generate=None, system="SYSTEM\n").build_messages(Task(task_id="q", query="Who?"), steps[:1])
    assert plain[1]["content"].endswith("\n\nWho?") and plain[1]["content"].startswith("Current date: ")
    assert plain[2] == {"role": "assistant", "content": raw}
    assert plain[3] == {"role": "user", "content": "<tool_response>\n[0] page\n</tool_response>"}
