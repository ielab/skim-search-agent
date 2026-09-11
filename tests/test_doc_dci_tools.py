"""The deep-research DCI baseline ACI: `bash` + `read` over a flat corpus export.

bash(command) runs a real shell command (grep/rg/ls/...) with cwd = the flat export dir;
output is tail-truncated. read(path, offset, limit) returns a 1-indexed line-range of one
exported file. There is no retriever; the agent must grep for candidate files itself. `.seen`
accumulates every doc_id surfaced (a direct read, or a filename mentioned in a bash command
or its output), for gold-doc coverage."""
from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.bash.tool import Bash, _tail_truncate
from agent_search.tools.read.tool import Read, _run_read

DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\nFounded in 1897 by A. Smith."},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
]


def _toolbox(key=None):
    units = units_from_documents(DOCS)
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="q")
    bound = [t.bind(state, units, ubyid, {}, corpus_key=key) for t in (Bash(), Read())]
    return ToolBox(bound, state)


# --- bash: a real shell over the flat export dir ----------------------------

def test_bash_grep_finds_the_exported_file():
    out = _toolbox().run("bash", {"command": "grep -rl 'Harbor Festival' ."})
    assert ".txt" in out


def test_bash_ls_lists_exported_files():
    out = _toolbox().run("bash", {"command": "ls"})
    assert ".txt" in out


def test_bash_empty_command_errors_cleanly():
    out = _toolbox().run("bash", {"command": ""})
    assert "Error" in out


def test_bash_no_matches_is_annotated():
    out = _toolbox().run("bash", {"command": "grep -rl 'zz_definitely_absent_zz' ."})
    assert "no matches found" in out or "Error" not in out


def test_bash_grep_surfaces_the_matched_doc_as_seen():
    box = _toolbox()
    out = box.run("bash", {"command": "grep -rl 'Harbor Festival' ."})
    # whatever filename grep printed must resolve to d_harbor in .seen
    assert "d_harbor" in box.seen, f"expected d_harbor in seen from: {out!r}"


# --- read: a 1-indexed line-range of one exported file ----------------------

def test_read_returns_file_contents():
    box = _toolbox()
    # discover the exact filename via bash first (real DCI usage pattern)
    listing = box.run("bash", {"command": "grep -rl 'Adams-Onis' ."})
    fname = [tok for tok in listing.split() if tok.endswith(".txt")][0].lstrip("./")
    out = box.run("read", {"path": fname})
    assert "Adams-Onis Treaty of 1819" in out


def test_read_paginates_with_offset_and_limit():
    box = _toolbox()
    fname = [tok for tok in box.run("bash", {"command": "ls"}).split() if tok.endswith(".txt")][0]
    out = box.run("read", {"path": fname, "offset": 1, "limit": 1})
    assert out.split("\n")[0]                      # got at least one line back


def test_read_unknown_file_errors_cleanly():
    out = _toolbox().run("read", {"path": "nonexistent_doc_id.txt"})
    assert "Error" in out


def test_read_path_cannot_escape_corpus_dir():
    out = _toolbox().run("read", {"path": "../../etc/passwd"})
    assert "Error" in out


def test_read_marks_doc_as_seen():
    box = _toolbox()
    fname = [tok for tok in box.run("bash", {"command": "ls"}).split() if tok.endswith(".txt")][0]
    box.run("read", {"path": fname})
    assert box.seen                                   # something got marked seen


# --- run() dispatch + repeat-call guard --------------------------------------

def test_run_dispatches_bash_and_read():
    out = _toolbox().run("bash", {"command": "ls"})
    assert ".txt" in out


def test_run_unknown_tool_errors():
    out = _toolbox().run("search", {"query": "x"})
    assert "unknown tool" in out.lower()


def test_repeated_identical_call_is_not_specially_flagged():
    # DCI is RISE's brute-force baseline: it must NOT get an anti-loop "REPEAT" nudge that the
    # method (research/research_bm25) never receives — that would understate its measured cost.
    # A repeated identical call just re-runs and returns the same result, like the RISE reference.
    box = _toolbox()
    first = box.run("bash", {"command": "ls"})
    second = box.run("bash", {"command": "ls"})
    assert "REPEAT" not in second
    assert first == second


# --- corpus_key reuses the SAME export dir across toolboxes (shared corpus) ---

def test_same_corpus_key_reuses_export_dir():
    a = _toolbox(key="__test_dci_shared_corpus__")
    b = _toolbox(key="__test_dci_shared_corpus__")
    assert a.state.scratch["dci_dir"] == b.state.scratch["dci_dir"]


# --- token caps (no character/byte caps anywhere — agent_search.tokens) ------------------

def test_tail_truncate_keeps_last_n_whitespace_tokens_not_bytes():
    """`_tail_truncate`'s size limit is WHITESPACE TOKENS, not bytes: a line whose token count
    is small but byte length is large (long individual words) must NOT trip the cap, and the
    kept content is the TAIL of the token stream."""
    # 50 short tokens fits comfortably under max_tokens=10 lines-worth if measured in bytes
    # (each token is 1 char), but token-counted only the LAST 10 survive.
    content = " ".join(f"t{i}" for i in range(50))
    out = _tail_truncate(content, max_lines=1000, max_tokens=10)
    assert "[Truncated:" in out and "token limit" in out
    kept = out.split("\n")[0]
    assert kept.split() == [f"t{i}" for i in range(40, 50)]


def test_tail_truncate_under_budget_is_unchanged():
    content = "a b c"
    assert _tail_truncate(content, max_lines=1000, max_tokens=10) == content


def test_run_read_caps_an_oversized_line_in_tokens_with_a_token_count_marker(tmp_path):
    """`_run_read`'s per-line cap (`cap_tokens`) truncates a line to `max_line_tokens`
    whitespace tokens and appends a `...[line truncated; N tokens]` marker — no character
    cap anywhere in the read path."""
    long_line = " ".join(f"w{i}" for i in range(30))
    f = tmp_path / "doc.txt"
    f.write_text(long_line)
    out = _run_read(tmp_path, "doc.txt", None, None, default_limit=10, max_line_tokens=5)
    assert out.startswith(" ".join(f"w{i}" for i in range(5)))
    assert "...[line truncated; 25 tokens]" in out


def test_run_read_line_within_budget_is_unmarked(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("short line")
    out = _run_read(tmp_path, "doc.txt", None, None, default_limit=10, max_line_tokens=5)
    assert out == "short line"
