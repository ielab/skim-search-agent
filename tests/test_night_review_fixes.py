"""Regressions for the 2026-06-11 night-review findings (agent+eval reviewer)."""
import json
import os
import tempfile

from agent_search.agent.actions import parse_tool_call
from agent_search.agent.loop import _final_locations
from evaluation.ground_truth import changed_line_ranges
from evaluation.run_eval import _load_rows
from agent_search.corpus.units import units_from_documents


def test_new_file_diff_does_not_leak_into_previous_gold():
    diff = ("--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
            "@@ -10,3 +10,3 @@ def f():\n a\n-b\n+B\n c\n"
            "diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/pkg/new.py\n"
            "@@ -0,0 +1,3 @@\n+def g():\n+    return 1\n+\n")
    assert changed_line_ranges(diff) == {"pkg/mod.py": [(11, 11)]}


def test_answer_survives_toolcall_markers_in_think():
    """A <think> that quotes a tool call must not preempt the real <answer> — the
    loop treats the turn as terminal and records the answer."""
    raw = ('<think>call {"name": "search_bql", "arguments": {"query": "token"}}'
           ' found it</think>\n<answer>auth/session.py:make_token</answer>')
    call = parse_tool_call(raw)
    name = call[0] if call else ""
    assert _final_locations(raw, name, call[1] if call else {}) == ["auth/session.py:make_token"]


def test_tool_call_with_close_tag_inside_arg_value():
    """A literal </tool_call> inside an argument string used to truncate the regex
    match and parse to None; the balanced-object fallback now recovers it."""
    raw = '<tool_call>{"name":"search_bql","arguments":{"query":"a</tool_call>b"}}</tool_call>'
    assert parse_tool_call(raw) == ("search_bql", {"query": "a</tool_call>b"})
    # a real call after a quoted decoy still wins
    raw2 = ('<think>{"name":"decoy","arguments":{}}</think>'
            '<tool_call>{"name":"grep","arguments":{"query":"real"}}</tool_call>')
    assert parse_tool_call(raw2) == ("grep", {"query": "real"})


def test_pyserini_truncation_terminates_and_raises():
    from agent_search.retrievers.lexical.pyserini import BM25Pyserini
    r = BM25Pyserini()

    class Boom:
        calls = 0
        def search(self, q, k):
            Boom.calls += 1
            raise RuntimeError("lucene error")

    r._searcher = Boom()
    try:
        r.search("some query text", k=10)
        assert False, "should raise"
    except RuntimeError:
        pass
    assert Boom.calls < 100          # terminates (was: infinite at cutoff=2)


def test_resume_tolerates_truncated_trailing_line():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "rows.jsonl")
    with open(p, "w") as fh:
        fh.write(json.dumps({"instance_id": "a", "recall@5": 1.0}) + "\n")
        fh.write('{"instance_id": "b", "recall@5"')      # killed mid-append
    rows, done = _load_rows(p)
    assert done == {"a"}             # partial line skipped -> 'b' re-scores


def test_beir_gold_reachability_matches_chunking():
    docs = [{"_id": "d1", "title": "", "text": ""},
            {"_id": "d2", "title": "T", "text": "x"}]
    unit_ids = {u.doc_id for u in units_from_documents(docs)}
    assert unit_ids == {"d2"}
    reachable = {
        str(d.get("doc_id") or d.get("id") or d.get("_id") or f"doc{i}")
        for i, d in enumerate(docs)
        if (d.get("title") or d.get("section") or d.get("heading")
            or d.get("text") or d.get("body") or d.get("contents"))
    }
    assert reachable == unit_ids
