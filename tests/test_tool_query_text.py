"""A search tool takes its query as a string, a list of strings or a number: the backbone
sometimes emits a list (ITER's tool accepted one), and a crash there would cost the step."""
from agent_search.tools.base import query_text


def test_query_text_accepts_string_list_and_number():
    assert query_text({"query": " treaty 1848 "}) == "treaty 1848"
    assert query_text({"query": ["treaty", "1848"]}) == "treaty 1848"
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
