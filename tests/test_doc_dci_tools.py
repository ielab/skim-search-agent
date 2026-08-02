"""The deep-research DCI baseline ACI: DciWorkspace (bash + read over a flat corpus export).

bash(command) runs a real shell command (grep/rg/ls/...) with cwd = the flat export dir;
output is tail-truncated. read(path, offset, limit) returns a 1-indexed line-range of one
exported file. NO retriever — the agent must grep for candidate files itself. `.seen`
accumulates every doc_id surfaced (a direct read, or a filename mentioned in a bash command
or its output), for gold-doc coverage."""
from agent_search.agent.tools.doc_dci import DciWorkspace
from agent_search.corpus.units import units_from_documents

DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\nFounded in 1897 by A. Smith."},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
]


def _ws(key=None):
    return DciWorkspace(units_from_documents(DOCS), corpus_key=key)


# --- bash: a real shell over the flat export dir ----------------------------

def test_bash_grep_finds_the_exported_file():
    out = _ws().bash("grep -rl 'Harbor Festival' .")
    assert ".txt" in out


def test_bash_ls_lists_exported_files():
    out = _ws().bash("ls")
    assert ".txt" in out


def test_bash_empty_command_errors_cleanly():
    out = _ws().bash("")
    assert "Error" in out


def test_bash_no_matches_is_annotated():
    out = _ws().bash("grep -rl 'zz_definitely_absent_zz' .")
    assert "no matches found" in out or "Error" not in out


def test_bash_grep_surfaces_the_matched_doc_as_seen():
    ws = _ws()
    out = ws.bash("grep -rl 'Harbor Festival' .")
    # whatever filename grep printed must resolve to d_harbor in .seen
    assert "d_harbor" in ws.seen, f"expected d_harbor in seen from: {out!r}"


# --- read: a 1-indexed line-range of one exported file ----------------------

def test_read_returns_file_contents():
    ws = _ws()
    # discover the exact filename via bash first (real DCI usage pattern)
    listing = ws.bash("grep -rl 'Adams-Onis' .")
    fname = [tok for tok in listing.split() if tok.endswith(".txt")][0].lstrip("./")
    out = ws.read(fname)
    assert "Adams-Onis Treaty of 1819" in out


def test_read_paginates_with_offset_and_limit():
    ws = _ws()
    fname = [tok for tok in ws.bash("ls").split() if tok.endswith(".txt")][0]
    out = ws.read(fname, offset=1, limit=1)
    assert out.split("\n")[0]                      # got at least one line back


def test_read_unknown_file_errors_cleanly():
    out = _ws().read("nonexistent_doc_id.txt")
    assert "Error" in out


def test_read_path_cannot_escape_corpus_dir():
    out = _ws().read("../../etc/passwd")
    assert "Error" in out


def test_read_marks_doc_as_seen():
    ws = _ws()
    fname = [tok for tok in ws.bash("ls").split() if tok.endswith(".txt")][0]
    ws.read(fname)
    assert ws.seen                                   # something got marked seen


# --- run() dispatch + repeat-call guard --------------------------------------

def test_run_dispatches_bash_and_read():
    ws = _ws()
    out = ws.run("bash", {"command": "ls"})
    assert ".txt" in out


def test_run_unknown_tool_errors():
    out = _ws().run("search", {"query": "x"})
    assert "unknown tool" in out.lower()


def test_repeated_identical_call_is_not_specially_flagged():
    # DCI is RISE's brute-force baseline: it must NOT get an anti-loop "REPEAT" nudge that the
    # method (research/research_bm25) never receives — that would understate its measured cost.
    # A repeated identical call just re-runs and returns the same result, like the RISE reference.
    ws = _ws()
    first = ws.run("bash", {"command": "ls"})
    second = ws.run("bash", {"command": "ls"})
    assert "REPEAT" not in second
    assert first == second


# --- corpus_key reuses the SAME export dir across workspaces (shared corpus) ---

def test_same_corpus_key_reuses_export_dir():
    a = _ws(key="__test_dci_shared_corpus__")
    b = _ws(key="__test_dci_shared_corpus__")
    assert a.corpus_dir == b.corpus_dir
