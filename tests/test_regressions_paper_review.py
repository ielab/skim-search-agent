"""Regression tests pinning specific correctness fixes across the BQL parser (structured errors
instead of crashes), Python source-unit line spans (decorators, duplicate qualnames), diff-based
ground-truth parsing (CRLF, top-of-file insertions), NEAR proximity semantics, and run_eval
aggregation. Each test pins one previously-wrong behavior.
"""
from agent_search.retrievers.structural.bql.parser import parse
from agent_search.evaluation.ground_truth import changed_line_ranges
from agent_search.evaluation.run_eval import _aggregate
from agent_search.retrievers.structural.bql.executor import StructuralExecutor
from agent_search.corpus.units import CodeUnit, units_from_python_source


# #22 — parser must never raise on a bad NEAR spec; return a structured error
def test_bad_near_spec_is_structured_error_not_crash():
    r = parse("NEAR/xyz(a, b)")
    assert not r.ok and "NEAR" in r.error


# #8 — a decorated function's span must include the decorator lines
def test_decorator_lines_included_in_function_span():
    u = units_from_python_source("m.py", "@property\ndef foo(self):\n    return 1\n")[0]
    assert u.start_line == 1
    assert u.code.splitlines()[0] == "@property"


# #10 — duplicate qualnames (@overload) must yield unique doc_ids
def test_duplicate_qualnames_get_unique_doc_ids():
    src = "@overload\ndef f(x): ...\n@overload\ndef f(y): ...\ndef f(z):\n    return z\n"
    ids = [u.doc_id for u in units_from_python_source("m.py", src)]
    assert len(ids) == len(set(ids)) == 3


# #1 — CRLF diffs must not leave a trailing \r in the parsed path
def test_crlf_diff_path_has_no_carriage_return():
    diff = "--- a/pkg/mod.py\r\n+++ b/pkg/mod.py\r\n@@ -5,2 +5,3 @@\r\n ctx\r\n-old\r\n+new\r\n"
    assert list(changed_line_ranges(diff).keys()) == ["pkg/mod.py"]


# #3 — a top-of-file insertion (@@ -0,0) clamps to line 1, never (0,0)
def test_top_insertion_clamps_to_line_one():
    add = "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
    assert changed_line_ranges(add)["x.py"] == [(1, 1)]


# #17 — NEAR/lineN is line distance, not a token window
def test_near_line_uses_line_distance():
    ex = StructuralExecutor([CodeUnit("u::n", "u", "n", 1, 3, "alpha\nbeta\ngamma")])
    assert [d for d, _ in ex.run(parse("NEAR/line1(alpha, beta)").expr)] == ["u::n"]
    assert ex.run(parse("NEAR/line1(alpha, gamma)").expr) == []   # 2 lines apart


# #20 — a multi-token term requires adjacency (not bag-of-words)
def test_multitoken_term_requires_adjacency():
    units = [
        CodeUnit("a::f", "a", "f", 1, 1, "open the file"),   # not adjacent
        CodeUnit("a::g", "a", "g", 1, 1, "open_file()"),     # adjacent
    ]
    r = [d for d, _ in StructuralExecutor(units).run(parse('"open file"').expr)]
    assert r == ["a::g"]


# #12 — aggregation is robust to a resumed file whose rows have different metric keys
def test_aggregate_robust_to_mixed_schemas():
    # _aggregate now takes the UNION of metric keys (not the intersection): each key is
    # averaged over only the rows that HAVE it — load-bearing for fix_file_ok, which exists
    # only on rows that committed a fix (the old all-rows-only filter silently dropped it).
    # `timeout_rate` is always added (fraction of rows that hit max_steps).
    scored = [
        {"instance_id": "a", "recall@1": 1.0, "recall@10": 1.0},
        {"instance_id": "b", "recall@1": 0.0},                 # missing recall@10
    ]
    agg = _aggregate(scored)
    assert agg == {"recall@1": 0.5, "recall@10": 1.0, "timeout_rate": 0.0}


# 2026-06-23 audit — three new correctness fixes:

# MAP must not double-credit a repeated doc_id (used to return > 1.0)
def test_map_dedups_repeated_doc_ids():
    from agent_search.evaluation.metrics import average_precision_at_k
    assert average_precision_at_k(["a", "a", "b"], {"a"}, 10) == 1.0
    assert average_precision_at_k(["x", "a", "b"], {"a", "b"}, 10) < 1.0   # normal unchanged


# dense flat-index tie-break stays deterministic even when far more than k docs tie
def test_flat_index_tie_break_deterministic_for_huge_tie_run():
    import numpy as np
    from agent_search.retrievers.dense.vector_index import FlatIndex
    e = np.ones((20, 4), dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)             # all identical -> 20-way tie
    idx = FlatIndex.build(e, [f"d{i:02d}" for i in range(20)])
    assert idx.search(e[0], 3) == ["d00", "d01", "d02"]       # lowest doc_ids, reproducible


# NEAR/lineN under file/region scope can't measure source lines -> co-occurrence,
# NOT a silent token-distance (which used to make IN(file, NEAR/lineN) wrong)
def test_near_line_under_file_scope_is_cooccurrence_not_token_distance():
    a = CodeUnit("f.py::a", "f.py", "a", 1, 9,
                 "def a():\n x=1\n y=2\n z=3\n foo=1\n m=1\n n=1\n bar=2\n")  # foo..bar >2 lines
    b = CodeUnit("f.py::b", "f.py", "b", 10, 11, "def b():\n pass\n")
    ex = StructuralExecutor([a, b])
    hits = {d for d, _ in ex.run(parse("IN(file, NEAR/line2(foo, bar))").expr)}
    assert "f.py::a" in hits                                  # both terms co-occur in the file


# file-level token bags are built LAZILY (a memory win for large shared-doc corpora) but
# remain exactly correct: untouched until a file-scope query, then identical to the eager
# build (path tokens + every sibling unit's tokens, so a term in a SIBLING unit matches).
def test_file_token_bag_is_lazy_and_spans_siblings():
    a = CodeUnit("mod/x.py::a", "mod/x.py", "a", 1, 1, "alpha beta")
    b = CodeUnit("mod/x.py::b", "mod/x.py", "b", 2, 2, "gamma delta")   # sibling, same file
    ex = StructuralExecutor([a, b])
    assert ex._ftoks == {}                                   # nothing built until a file query
    # `gamma` lives in sibling b; IN(file, AND(alpha, gamma)) still matches both units of x.py
    hits = {d for d, _ in ex.run(parse("IN(file, AND(alpha, gamma))").expr)}
    assert hits == {"mod/x.py::a", "mod/x.py::b"}
    assert "mod/x.py" in ex._ftoks                           # now cached for that path
    assert "delta" in ex._file_toks("mod/x.py") and "x" in ex._file_toks("mod/x.py")  # +path token


# a document corpus with a duplicate doc_id must not produce two units with the same
# id (would collide in dense/BM25 dict-keying and skew metrics)
def test_documents_with_duplicate_ids_are_deduped():
    from agent_search.corpus.units import units_from_documents
    units = units_from_documents([
        {"doc_id": "d1", "text": "first"},
        {"doc_id": "d1", "text": "duplicate id, dropped"},
        {"doc_id": "d2", "text": "second"},
    ])
    ids = [u.doc_id for u in units]
    assert ids == ["d1", "d2"]                               # the second d1 is skipped


# 2026-06-24 audit (parallel correctness agents):

# acc@k must be 0.0 at k=0 (no top-k), like every other @k metric (was 1.0)
def test_acc_at_k_is_zero_at_k0():
    from agent_search.evaluation.metrics import acc_at_k
    assert acc_at_k(["a", "b"], {"a"}, 0) == 0.0
    assert acc_at_k(["a", "b"], {"a"}, 1) == 1.0


# EVERY @k metric must return 0.0 at k<=0 — a negative slice (retrieved[:-1]) used to
# give wrong values; average_precision even went negative (denom = min(|gold|,-1))
def test_all_at_k_metrics_zero_for_nonpositive_k():
    import agent_search.evaluation.metrics as M
    ret, gold = ["a", "b", "c"], {"a", "b"}
    for fn in (M.recall_at_k, M.acc_at_k, M.precision_at_k, M.f1_at_k,
               M.mrr_at_k, M.ndcg_at_k, M.hit_at_k, M.average_precision_at_k):
        for k in (0, -1, -5):
            assert fn(ret, gold, k) == 0.0, (fn.__name__, k)


# Unicode-aware tokenizer: ASCII byte-identical (code numbers unchanged), non-ASCII kept
def test_code_tokenize_is_ascii_identical_and_unicode_safe():
    from agent_search.corpus.units import code_tokenize
    assert code_tokenize("worldToPixel make_token foo123") == \
        ["world", "to", "pixel", "make", "token", "foo123"]   # ASCII unchanged
    assert "café" in code_tokenize("the café münchen") and "münchen" in code_tokenize("a münchen")


def _episode(policy_replies, domain, toolset=("ev",)):
    """A minimal stand-in for `run_episode`'s workspace argument — just enough to
    satisfy `agent.loop.WorkspaceLike` (a `.run(name, args)` dispatcher + a `.surfaced`
    accumulator), the same registry-driven "engine" shape every real workspace
    (CodeFixWorkspace/GrepReadWorkspace/DocSearchFetch/DciWorkspace) supports for a
    NEW retrieval method. These two tests pin `run_episode`'s own branching (bare-STOP
    termination; a research <answer>'s PROSE not overriding the surfaced ranking), not
    any one workspace's behavior, so a duck-typed stub is the right level of coupling."""
    from agent_search.agent.loop import run_episode, Task, Policy
    units = [CodeUnit(f"p{i}.py::g{i}", f"p{i}.py", f"g{i}", 1, 1, "evidence") for i in range(3)]
    ids = [u.doc_id for u in units]

    class Eng:
        def search(self, q, k): return ids[:k]

    class Stub:
        """`.run` dispatches to a {tool_name: Eng()} registry, same contract as a real
        workspace's engines-dict tools; `.surfaced` accumulates every call's hits."""
        def __init__(self, engines):
            self.engines = engines
            self.surfaced = []
        def run(self, name, args):
            eng = self.engines.get(name)
            if eng is None:
                return f"ERROR: unknown tool {name!r}."
            hits = eng.search(args.get("query", ""), 5)
            self.surfaced.extend(d for d in hits if d not in self.surfaced)
            return f"{len(hits)} hits for {args.get('query', '')}"

    class P(Policy):
        def __init__(self): self.i = 0
        def propose(self, t, h):
            r = policy_replies[min(self.i, len(policy_replies) - 1)]; self.i += 1; return r

    ws = Stub(engines={t: Eng() for t in toolset})
    return run_episode(P(), Task("q", "x"), ws, units, max_steps=6, domain=domain), units


# a research <answer> is PROSE, not locations: ranking = surfaced evidence, answer kept
def test_research_answer_ranks_by_surfaced_evidence_not_prose():
    traj, units = _episode(
        ['<tool_call>{"name":"ev","arguments":{"query":"x"}}</tool_call>',
         "<answer>Treaty of Guadalupe Hidalgo</answer>"], domain="general")
    assert traj.stopped_reason == "answer"
    assert traj.final_answer == "Treaty of Guadalupe Hidalgo"
    assert traj.located == [u.doc_id for u in units]          # surfaced, not prose-parsed


# the isolation task terminates with a bare STOP — the loop must honor it (not burn max_steps)
def test_bare_stop_terminates_with_surfaced_ranking():
    traj, units = _episode(
        ['<tool_call>{"name":"ev","arguments":{"query":"x"}}</tool_call>', "STOP", "STOP"],
        domain="code")
    assert traj.stopped_reason == "stop" and traj.llm_calls == 2
    assert traj.located == [u.doc_id for u in units]          # ranking = surfaced hits
