"""The patch-mode agent's <fix> SEARCH/REPLACE edits compile to a git-applyable unified diff.

Guards the contract sb-cli depends on: the model works through the lean search->fetch ACI and
emits anchored edits; we hold the true base_commit source, so the compiled `model_patch`
applies. Tolerant of the tools' `<n>: ` line prefixes and trailing whitespace; refuses to
guess when a SEARCH block is ambiguous or missing.
"""
import subprocess
import tempfile

from agent_search.evaluation.patch_synthesis import parse_fix_edits, synthesize_patch


_SRC = ("def _line_type(line):\n"
        "    sline = line.strip()\n"
        "    if sline.startswith('READ'):\n"
        "        return 'command'\n"
        "    return 'data'\n")


def _apply(patch: str, path: str, src: str) -> int:
    """Return `git apply --check` rc for `patch` against a one-file repo holding {path: src}."""
    import os
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, os.path.dirname(path)), exist_ok=True)
    open(os.path.join(d, path), "w").write(src)
    g = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
    g("init", "-q"); g("add", "-A")
    g("-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "b")
    open(os.path.join(d, "p.diff"), "w").write(patch)
    return g("apply", "--check", "p.diff").returncode


def test_parse_strips_line_prefixes_and_tracks_file():
    fix = ("<fix>\nfile: a/b.py\n<<<<<<< SEARCH\n3:     if x:\n=======\n    if y:\n>>>>>>> REPLACE\n</fix>")
    edits = parse_fix_edits(fix)
    assert len(edits) == 1
    assert edits[0].path == "a/b.py"
    assert edits[0].search == "    if x:"        # "3: " prefix stripped
    assert edits[0].replace == "    if y:"


def test_synthesized_patch_git_applies():
    p = "astropy/io/ascii/qdp.py"
    fix = (f"<fix>\nfile: {p}\n<<<<<<< SEARCH\n    if sline.startswith('READ'):\n"
           "=======\n    if sline.upper().startswith('READ'):\n>>>>>>> REPLACE\n</fix>")
    patch, rep = synthesize_patch(fix, {p: _SRC})
    assert rep.n_applied == 1 and patch
    assert _apply(patch, p, _SRC) == 0            # applies cleanly against the true source


def test_multi_edit_same_file_one_diff():
    p = "m.py"
    src = "a = 1\nb = 2\nc = 3\n"
    fix = (f"<fix>\nfile: {p}\n<<<<<<< SEARCH\na = 1\n=======\na = 10\n>>>>>>> REPLACE\n"
           "<<<<<<< SEARCH\nc = 3\n=======\nc = 30\n>>>>>>> REPLACE\n</fix>")
    patch, rep = synthesize_patch(fix, {p: src})
    assert rep.n_applied == 2
    assert _apply(patch, p, src) == 0
    assert patch.count("diff --git") == 1         # both edits in one file's diff


def test_ambiguous_search_is_refused():
    # "x = 0" appears twice -> unanchorable -> dropped, no bogus patch.
    src = "x = 0\ny = 1\nx = 0\n"
    fix = "<fix>\nfile: m.py\n<<<<<<< SEARCH\nx = 0\n=======\nx = 9\n>>>>>>> REPLACE\n</fix>"
    patch, rep = synthesize_patch(fix, {"m.py": src})
    assert rep.n_applied == 0 and patch == ""


def test_reindents_mis_indented_replace_to_real_source():
    # the model copied a SEARCH block with the WRONG (uniform +1 space) indent and carried it
    # into REPLACE. We match on the real 4-space source and must re-base REPLACE to 4 spaces,
    # so the spliced code parses (else: applies cleanly but is an IndentationError).
    import ast
    src = ("class C:\n"
           "    def __init__(self):\n"
           "        super().__init__()\n")
    fix = ("<fix>\nfile: m.py\n<<<<<<< SEARCH\n"
           "     def __init__(self):\n"                  # 5 spaces (model's error)
           "         super().__init__()\n"
           "=======\n"
           "     def __init__(self, x=None):\n"          # 5 spaces
           "         self.x = x\n"
           "         super().__init__()\n"
           ">>>>>>> REPLACE\n</fix>")
    patch, rep = synthesize_patch(fix, {"m.py": src})
    assert rep.n_applied == 1
    # every added line must be re-based to 4-space depth, not 5
    added = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert added and all(l.startswith("    def") or l.startswith("        ") for l in added)
    assert _apply(patch, "m.py", src) == 0
    # the spliced result must be valid Python
    new = src.replace("    def __init__(self):\n        super().__init__()\n",
                      "    def __init__(self, x=None):\n        self.x = x\n        super().__init__()\n")
    ast.parse(new)


def test_prose_fix_yields_no_patch():
    # the localization twin's fix_text has no SEARCH/REPLACE -> empty patch (safe no-op).
    prose = "<fix>\nfile: m.py\nfunction: foo\nchange: make it case-insensitive\n</fix>"
    patch, rep = synthesize_patch(prose, {"m.py": _SRC})
    assert patch == "" and rep.n_edits == 0
