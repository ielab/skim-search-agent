"""The code-fix ACI: `search` (SearchCode) -> `fetch` (FetchCode), the code arm's tool contract.

search(query) returns ranked files and their function/method names, no bodies, numbered for
fetch; fetch([rank, part]) pulls a named function/method (or an L-range) from a ranked file,
never the whole file. Malformed inputs recover instead of dead-ending."""
from agent_search.corpus.units import units_from_python_source
from agent_search.retrievers.bql.executor import StructuralExecutor
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.fetch_code.tool import FetchCode, SearchCode

SRC = (
    "class Command:\n"
    "    def handle(self, *args, **options):\n"
    "        return self.sqlmigrate(args)\n"
    "\n"
    "    def sqlmigrate(self, args):\n"
    "        # cannot rollback DDL on this backend\n"
    "        return atomic(args)\n"
    "\n"
    "def atomic(x):\n"
    "    return x\n"
)


def _toolbox():
    files = {"core/mgmt.py": SRC}
    units = units_from_python_source("core/mgmt.py", SRC)
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    executor = StructuralExecutor(units)
    sc = SearchCode(name="search").bind(state, units, ubyid, {"bql_plain": executor}, files=files)
    fc = FetchCode(name="fetch").bind(state, units, ubyid, {}, files=files)
    return ToolBox([sc, fc], state)


# --- search: ranked files + structure, no bodies ----------------------------

def test_search_returns_files_and_method_names_no_bodies():
    out = _toolbox().run("search", {"query": "sqlmigrate[def]", "k": 5})
    assert "core/mgmt.py" in out
    assert "Command.sqlmigrate" in out           # its structure (name)
    assert "return atomic" not in out            # NOT the body
    assert "-> IN(def, sqlmigrate)" in out       # lowered query is shown


def test_search_by_error_string_scopes_to_comment():
    out = _toolbox().run("search", {"query": '"cannot rollback DDL"[comment]', "k": 5})
    assert "core/mgmt.py" in out and "Command.sqlmigrate" in out


def test_search_unknown_field_is_a_readable_error():
    out = _toolbox().run("search", {"query": "x[module]", "k": 5})
    assert "error" in out.lower() or "region" in out.lower()


def test_search_zero_hits_bare_multiword_reruns_as_or():
    # several bare words parse as one adjacent phrase (0-hits in real code); the tool
    # silently reruns as an OR and shows real hits instead of a dead end.
    out = _toolbox().run("search", {"query": "sqlmigrate atomic handle", "k": 5})
    assert "core/mgmt.py" in out                  # recovered via OR
    assert "reran as" in out or "IN" in out


def test_search_strips_noise_keyword_before_lowering():
    # "class Foo[def]" would become a two-word phrase; the tool strips the noise word.
    out = _toolbox().run("search", {"query": "class Command[def]", "k": 5})
    assert "core/mgmt.py" in out and "(0 hits)" not in out


# --- fetch: named part / L-range, never the whole file ----------------------

def test_fetch_named_method_returns_its_source():
    box = _toolbox()
    box.run("search", {"query": "sqlmigrate[def]", "k": 5})
    out = box.run("fetch", {"specs": [[1, "Command.sqlmigrate"]]})
    assert "def sqlmigrate" in out and "cannot rollback DDL" in out
    assert "[1] core/mgmt.py :: Command.sqlmigrate" in out


def test_fetch_suffix_matches_bare_name():
    box = _toolbox()
    box.run("search", {"query": "sqlmigrate[def]", "k": 5})
    out = box.run("fetch", {"specs": [[1, "sqlmigrate"]]})          # bare name, not the dotted qualname
    assert "def sqlmigrate" in out


def test_fetch_line_range():
    box = _toolbox()
    box.run("search", {"query": "atomic[def]", "k": 5})
    out = box.run("fetch", {"specs": [[1, "L1-3"]]})
    assert "1: class Command:" in out


def test_fetch_bad_part_lists_available_names():
    box = _toolbox()
    box.run("search", {"query": "sqlmigrate[def]", "k": 5})
    out = box.run("fetch", {"specs": [[1, "nonexistent_method"]]})
    assert "no part named" in out and "Command.sqlmigrate" in out


def test_fetch_flat_single_pair_is_accepted():
    # a model often sends [1, "X"] instead of [[1, "X"]] for one part — recovered.
    box = _toolbox()
    box.run("search", {"query": "sqlmigrate[def]", "k": 5})
    out = box.run("fetch", {"specs": [1, "Command.sqlmigrate"]})
    assert "def sqlmigrate" in out and "bad spec" not in out


def test_fetch_flattened_dict_via_run():
    box = _toolbox()
    box.run("search", {"query": "sqlmigrate[def]", "k": 5})
    out = box.run("fetch", {"rank": 1, "part": "Command.handle"})
    assert "def handle" in out


def test_fetch_before_search_errors_cleanly():
    out = _toolbox().run("fetch", {"specs": [[1, "x"]]})
    assert "no search results yet" in out


def test_fetch_rank_out_of_range():
    box = _toolbox()
    box.run("search", {"query": "sqlmigrate[def]", "k": 5})
    out = box.run("fetch", {"specs": [[9, "x"]]})
    assert "out of range" in out


def test_unknown_tool_lists_available():
    out = _toolbox().run("open", {"path": "x"})
    assert "unknown tool" in out.lower() and "search" in out and "fetch" in out
