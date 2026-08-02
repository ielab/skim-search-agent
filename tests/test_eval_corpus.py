from agent_search.corpus.units import units_from_python_source

SRC = '''def foo(x):
    return x + 1


class A:
    def bar(self):
        return 2

    def baz(self):
        return 3
'''


def test_extracts_top_level_function_and_methods():
    quals = {u.qualname for u in units_from_python_source("m.py", SRC)}
    assert "foo" in quals
    assert "A.bar" in quals and "A.baz" in quals


def test_unit_has_line_range_code_and_docid():
    units = {u.qualname: u for u in units_from_python_source("m.py", SRC)}
    foo = units["foo"]
    assert foo.start_line == 1
    assert "return x + 1" in foo.code
    assert foo.doc_id == "m.py::foo"
    assert units["A.bar"].start_line == 6


def test_syntax_error_returns_empty_not_crash():
    assert units_from_python_source("bad.py", "def (:\n") == []
