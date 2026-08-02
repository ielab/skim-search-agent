"""Real AST structural scoping: IN(def/call/comment/string/sig) must distinguish
*where* a term occurs — the differentiator from plain grep. Before the fix these
were all no-ops returning the same set."""
from agent_search.retrievers.structural.bql.parser import parse
from agent_search.retrievers.structural.bql.executor import StructuralExecutor
from agent_search.corpus.units import units_from_python_source

SRC = (
    "def make_token(user_id):\n"
    '    note = "alpha_string_token"\n'
    "    return user_id  # bravo_comment marker\n"
    "\n"
    "def login(user_id):\n"
    "    return make_token(user_id)\n"
)


def _run(q):
    units = units_from_python_source("a.py", SRC)
    return [d for d, _ in StructuralExecutor(units).run(parse(q).expr)]


def test_in_def_matches_where_defined():
    assert _run("IN(def, make_token)") == ["a.py::make_token"]


def test_in_call_matches_where_called():
    assert _run("IN(call, make_token)") == ["a.py::login"]


def test_in_comment_matches_comment_text_only():
    assert _run("IN(comment, bravo)") == ["a.py::make_token"]
    assert _run("IN(comment, login)") == []          # 'login' is code, not a comment


def test_in_string_matches_string_literal_only():
    assert _run("IN(string, alpha)") == ["a.py::make_token"]
    assert _run("IN(string, return)") == []          # 'return' is code, not a string


def test_def_and_call_are_distinct():
    assert _run("IN(def, make_token)") != _run("IN(call, make_token)")


def test_in_sig_matches_signature_tokens():
    # user_id is in both signatures; both functions have it in their def line
    assert set(_run("IN(sig, user_id)")) == {"a.py::make_token", "a.py::login"}
