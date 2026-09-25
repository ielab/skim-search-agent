"""A search tool takes its query as a string, a list of strings or a number: the backbone
sometimes emits a list (Tongyi's own tool and DIVER's run its first query), and a crash there
would cost the step."""
from agent_search.tools.base import query_text


def test_query_text_accepts_string_list_and_number():
    assert query_text({"query": " treaty 1848 "}) == "treaty 1848"
    assert query_text({"query": ["treaty of 1848", "guadalupe hidalgo"]}) == "treaty of 1848"   # the first query, as Tongyi's tool runs it
    assert query_text({"query": 1848}) == "1848"
    assert query_text({"q": ["only"]}) == "only"
    assert query_text({}) == ""


def test_functions_prefix_on_a_tool_name_is_accepted():
    from agent_search.tools.base import Tool, ToolBox, EpisodeState

    class Echo(Tool):
        name = "echo"
        description = "echo"
        parameters = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

        def run(self, args):
            return "got " + str(args.get("x"))

    box = ToolBox([Echo(name="echo")], EpisodeState())
    assert box.run("functions.echo", {"x": "1"}) == "got 1"
    assert box.run("echo", {"x": "2"}) == "got 2"


def test_i8_is_agentir_two_field_form():
    """DIVER's i8, used with AgentIR-4B: the issuing turn's reasoning verbatim, 'Empty' when
    there is none, then the sub-query; no main question and no previous interactions."""
    from agent_search.training.queries import structured_query, STYLES
    assert "i8" in STYLES
    q = structured_query("i8", "main?", "sub q", [{"query": "old"}], pre_reasoning="I think\nso")
    assert q == "Reasoning: I think\nso\n\nQuery: sub q"
    assert structured_query("i8", "main?", "sub q", [], pre_reasoning="") == "Reasoning: Empty\n\nQuery: sub q"
