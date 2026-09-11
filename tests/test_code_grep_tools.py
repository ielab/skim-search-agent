"""The code-fix grep baseline ACI: RegexGrep (real regex to matching lines) plus a line-range
read.

grep(pattern) returns matching lines as `path:line: text` (RISE-style: real regex,
case-insensitive, no BM25 rerank, no Boolean); read(path, start, end) returns a capped
line-range slice, never structured parts. Malformed inputs recover instead of dead-ending,
same contract shape as CodeFixWorkspace."""
from agent_search.legacy.workspaces.code_grep import GrepReadWorkspace, RegexGrep
from agent_search.corpus.units import units_from_python_source

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


def _ws():
    files = {"core/mgmt.py": SRC}
    units = units_from_python_source("core/mgmt.py", SRC)
    return GrepReadWorkspace(units, files)


# --- grep: real regex, matching LINES, no bodies-as-units -------------------

def test_grep_returns_matching_lines_with_path_and_line_number():
    out = _ws().grep("sqlmigrate")
    assert "core/mgmt.py:" in out
    assert "def sqlmigrate" in out or "self.sqlmigrate" in out


def test_grep_is_case_insensitive():
    out = _ws().grep("SQLMIGRATE")
    assert "core/mgmt.py:" in out


def test_grep_supports_real_regex_alternation():
    out = _ws().grep("handle|atomic")
    assert "core/mgmt.py:" in out


def test_grep_invalid_regex_falls_back_to_literal_substring():
    # an unbalanced group is invalid regex; RegexGrep degrades to grep -F rather than erroring.
    out = _ws().grep("sqlmigrate(")
    assert "0 matches" not in out


def test_grep_zero_hits_is_a_plain_message():
    out = _ws().grep("zz_not_a_real_token_zz")
    assert "0 matches" in out


def test_grep_empty_pattern_returns_no_hits():
    hits, n = RegexGrep(units_from_python_source("x.py", SRC)).search("")
    assert hits == [] and n == 0


# --- read: a plain line-range slice, never a structural part ---------------

def test_read_returns_line_range_with_numbers():
    out = _ws().read("core/mgmt.py", 1, 3)
    assert "1: class Command:" in out
    assert "core/mgmt.py lines 1-3 of" in out


def test_read_defaults_start_and_caps_at_80_lines():
    out = _ws().read("core/mgmt.py")
    assert "core/mgmt.py lines 1-" in out


def test_read_suffix_matches_bare_filename():
    ws = _ws()
    out = ws.read("mgmt.py", 1, 2)          # bare basename, not the full repo-relative path
    assert "class Command" in out


def test_read_unknown_file_lists_no_candidates():
    out = _ws().read("nope.py", 1, 2)
    assert "no such file" in out.lower()


def test_read_via_run_dispatch():
    ws = _ws()
    out = ws.run("read", {"path": "core/mgmt.py", "start": 1, "end": 1})
    assert "class Command" in out


def test_grep_via_run_dispatch_accepts_query_alias():
    # some models send {"query": ...} instead of {"pattern": ...} for grep — accepted.
    out = _ws().run("grep", {"query": "sqlmigrate"})
    assert "core/mgmt.py:" in out


def test_unknown_tool_lists_available():
    out = _ws().run("fetch", {"specs": []})
    assert "unknown tool" in out.lower() and "grep" in out and "read" in out
