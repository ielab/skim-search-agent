"""Code-fix scoring: fix-file-ok + the grounding guard (agent_search.evaluation.fix_scoring).

The code arm's end-to-end metric is whether the <fix>'s `file:` line names a gold-patch file
(suffix-lenient). The grounding guard rejects a <fix> whose file was never fetched."""
from agent_search.evaluation.fix_scoring import (extract_fix, fix_file, is_grounded, score_fix)

# a minimal unified diff editing two files.
PATCH = (
    "--- a/django/db/migrations/executor.py\n"
    "+++ b/django/db/migrations/executor.py\n"
    "@@ -10,3 +10,4 @@\n"
    "-    old\n"
    "+    new\n"
    "--- a/django/core/management/commands/sqlmigrate.py\n"
    "+++ b/django/core/management/commands/sqlmigrate.py\n"
    "@@ -5,2 +5,3 @@\n"
    "-    x\n"
    "+    y\n"
)

FIX = ("<fix>\n"
       "file: django/core/management/commands/sqlmigrate.py\n"
       "function: Command.handle\n"
       "change: guard the atomic block\n"
       "</fix>")


def test_extract_fix_returns_inner_block():
    assert extract_fix(f"reasoning...\n{FIX}").startswith("file:")
    assert extract_fix("no fix here") is None


def test_extract_fix_takes_the_last_block():
    two = "<fix>\nfile: a.py\n</fix>\nthen\n<fix>\nfile: b.py\n</fix>"
    assert "b.py" in extract_fix(two) and "a.py" not in extract_fix(two)


def test_fix_file_parses_the_path():
    assert fix_file(FIX) == "django/core/management/commands/sqlmigrate.py"
    assert fix_file("file: `path/x.py`") == "path/x.py"        # strips markup
    assert fix_file("no file line") is None


def test_score_fix_hits_a_gold_file():
    ok, shown = score_fix(FIX, PATCH)
    assert ok is True and shown.endswith("sqlmigrate.py")


def test_score_fix_wrong_file_misses():
    wrong = "<fix>\nfile: some/other/module.py\nfunction: f\nchange: z\n</fix>"
    ok, _ = score_fix(wrong, PATCH)
    assert ok is False


def test_score_fix_suffix_lenient():
    # the fix may name a longer or shorter suffix of the gold path.
    ok, _ = score_fix("<fix>\nfile: commands/sqlmigrate.py\nfunction: f\nchange: z\n</fix>", PATCH)
    assert ok is True


def test_score_fix_no_fix_is_false():
    assert score_fix(None, PATCH) == (False, "(no fix emitted)")


# --- grounding guard --------------------------------------------------------

def test_is_grounded_requires_the_file_was_fetched():
    read = {"core/mgmt.py", "core/util.py"}
    assert is_grounded("core/mgmt.py", read) is True
    assert is_grounded("core/other.py", read) is False
    assert is_grounded(None, read) is False


def test_is_grounded_suffix_lenient():
    assert is_grounded("mgmt.py", {"core/mgmt.py"}) is True
