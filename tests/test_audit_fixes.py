"""Regressions for issues the 2026-06 audit found (loop/parse/domain/qrels)."""
import os
import tempfile

from agent_search.agent.actions import parse_tool_call
from agent_search.agent.loop import Step, Trajectory, _final_locations
from agent_search.agent.retriever import _trajectory_meta


def test_answer_quoted_in_think_does_not_end_episode():
    """A model that reasons aloud about its answer before searching must NOT be
    terminated: an <answer> inside <think> is not a real answer."""
    raw = ('<think>The <answer>Foo.bar</answer> might be right but let me search.</think>'
           '<tool_call>{"name":"grep","arguments":{"query":"x"}}</tool_call>')
    call = parse_tool_call(raw)
    assert _final_locations(raw, call[0], call[1]) is None     # -> the grep runs


def test_real_answer_outside_think_still_terminal():
    raw = "<think>done searching</think>\n<answer>auth/session.py:make_token</answer>"
    assert _final_locations(raw, "", {}) == ["auth/session.py:make_token"]


def test_bare_json_with_brace_in_value_parses():
    assert parse_tool_call('{"name":"grep","arguments":{"query":"a } b"}}') == \
        ("grep", {"query": "a } b"})
    assert parse_tool_call('{"name":"search_bql","arguments":{"query":"IN(call, foo) }"}}') == \
        ("search_bql", {"query": "IN(call, foo) }"})


def test_hits_per_step_reads_search_headers():
    # the search -> fetch `search` observation header carries the match count:
    #   code: "search: 'q' -> BQL  (N units in M files ...)"
    #   docs: "search: q -> BQL   (N matches, top K):"
    t = Trajectory(task_id="q", steps=[
        Step(name="search", args={},
             observation="search: 'x[def]' -> IN(def, x)  (5 units in 2 files, top 2):\n  1  a.py"),
        Step(name="search", args={},
             observation="search: y[title] -> IN(title, y)   (42 matches, top 5):\n  1  d1"),
        Step(name="fetch", args={}, observation="[1] a.py :: f\n1: def f():"),   # not a search -> 0
    ])
    assert _trajectory_meta(t)["hits_per_step"] == [5, 42, 0]


def test_qrels_tolerates_space_delimited_trec():
    from evaluation.datasets import _qrels_from_tsv
    d = tempfile.mkdtemp()
    p = os.path.join(d, "qrels.tsv")
    with open(p, "w") as fh:
        fh.write("q1 Q0 d1 1\nq1 Q0 d2 0\nq2 Q0 d3 1\n")    # space-delimited TREC
    q = _qrels_from_tsv(p)
    assert q == {"q1": {"d1"}, "q2": {"d3"}}                 # rel=0 dropped, no crash


def test_research_domain_uses_text_embedder_default():
    """A document benchmark must not default to the CODE embedder."""
    from evaluation.config import DatasetArgs, RetrieverArgs, RunConfig
    from evaluation.datasets import available_datasets
    assert {"browsecomp_plus", "hotpotqa", "musique"} <= available_datasets()
    cfg = RunConfig(
        dataset=DatasetArgs(name="hotpotqa_fixture"),
        retriever=RetrieverArgs(name="agent_research_bql"),
    ).resolved()
    assert cfg.agent.domain == "general"
    assert cfg.retriever.dense_model == "BAAI/bge-base-en-v1.5"
