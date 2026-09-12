"""A search tool takes its query as a string, a list of strings or a number: the backbone
sometimes emits a list (ITER's tool accepted one), and a crash there would cost the step."""
from agent_search.tools.base import query_text


def test_query_text_accepts_string_list_and_number():
    assert query_text({"query": " treaty 1848 "}) == "treaty 1848"
    assert query_text({"query": ["treaty", "1848"]}) == "treaty 1848"
    assert query_text({"query": 1848}) == "1848"
    assert query_text({"q": ["only"]}) == "only"
    assert query_text({}) == ""
