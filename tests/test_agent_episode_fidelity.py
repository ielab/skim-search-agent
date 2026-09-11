"""Tongyi-loop fidelity and episode semantics under the unified agent loop:
a quoted call or STOP marker inside <think> must not preempt the real tool
call; the last <tool_call> wins; <answer> is terminal; the accumulated ranking
is recency-then-rank; each retrieval engine is built only when its tool is in
the toolset.
"""
from agent_search.agent.actions import parse_tool_call
from agent_search.agent.loop import Task, _final_locations, run_episode
from agent_search.corpus.units import CodeUnit
from tests.lucene_support import require_jvm

require_jvm()

THINK_THEN_CALL = (
    "<think>Should I STOP here? No — let me search the session code.</think>\n"
    '<tool_call>\n{"name": "grep", "arguments": {"query": "session token"}}\n</tool_call>'
)


def test_stop_inside_think_does_not_preempt_the_call():
    assert parse_tool_call(THINK_THEN_CALL) == ("grep", {"query": "session token"})


def test_last_tool_call_wins():
    s = ('<tool_call>{"name": "search_bql", "arguments": {"query": "IN(def, a)"}}</tool_call>\n'
         '<tool_call>{"name": "search_bql", "arguments": {"query": "IN(call, b)"}}</tool_call>')
    assert parse_tool_call(s) == ("search_bql", {"query": "IN(call, b)"})


def test_answer_is_terminal_and_carries_the_answer():
    assert _final_locations("<answer>auth/session.py</answer>", "", {}) == ["auth/session.py"]
    raw = '<think>{"name":"grep","arguments":{"query":"x"}}</think><answer>auth/session.py</answer>'
    call = parse_tool_call(raw)
    assert _final_locations(raw, call[0] if call else "", {}) == ["auth/session.py"]


def test_episode_records_final_answer():
    from agent_search.tools.base import EpisodeState, ToolBox
    from agent_search.tools.fetch.tool import Fetch
    from agent_search.tools.search_bql.tool import SearchBql

    class NeverSearched:
        """The doc arm's BQL engine slot. The policy answers at once, so no tool runs."""

        def run_with_count(self, expr, k=100):
            raise AssertionError("the toolbox must not be called before the answer")

    # doc arm: <answer> is the terminal; the policy answers immediately, so the toolbox is
    # never actually called (its exact tools/engine don't matter here).
    state = EpisodeState(question="q")
    sb = SearchBql(name="search").bind(state, [], {}, {"bql": NeverSearched()})
    fe = Fetch(name="fetch").bind(state, [], {}, {})
    ws = ToolBox([sb, fe], state)

    class AnswerPolicy:
        def propose(self, task, history):
            return "<answer>Treaty of Guadalupe Hidalgo, 1848</answer>"

    traj = run_episode(AnswerPolicy(), Task("t", "q"), ws, [], max_steps=5)
    assert traj.stopped_reason == "answer"
    assert traj.final_answer == "Treaty of Guadalupe Hidalgo, 1848"


def test_answer_metrics():
    from agent_search.evaluation.metrics import answer_em, answer_f1
    assert answer_em("The Treaty of Guadalupe Hidalgo!", "treaty of guadalupe hidalgo") == 1.0
    assert answer_em("something else", "guadalupe") == 0.0
    assert 0.5 < answer_f1("It was the Treaty of Guadalupe Hidalgo, signed 1848",
                           "Treaty of Guadalupe Hidalgo") < 1.0
    assert answer_f1("", "gold") == 0.0


def test_engine_built_only_for_its_arm(tmp_path):
    """The search -> fetch arms build the BQL executor (their `search` lowers to it); the
    retrieve-then-visit baseline builds the Lucene BM25 engine instead, never both."""
    from agent_search.evaluation.agent_runner import ConditionAgent
    from agent_search.strategies import get_condition

    units = [CodeUnit(doc_id="a.py::f", path="a.py", qualname="f",
                      code="def f():\n    pass\n", start_line=1, end_line=2)]
    root = str(tmp_path / "idx")
    code = ConditionAgent(get_condition("codefix"), lambda: None, index_root=root).index(units)
    base = ConditionAgent(get_condition("research_bm25"), lambda: None, index_root=root).index(units)
    # every bql ranking (plain/fused/dense-only) shares one persisted artifact under the
    # "bql" key (agent_search/retrievers/engines.py); what matters here is that the code
    # arm builds only that family and the retrieve-then-visit baseline builds only bm25.
    assert set(code.engines.built) == {"bql"}
    assert set(base.engines.built) == {"bm25"}


def test_grep_quoted_query_is_literal_substring():
    """SWE-agent search_dir semantics: a quoted query greps as ONE literal
    substring; unquoted is GrepRAG keyword mode (ANY keyword matches)."""
    from agent_search.retrievers.lexical.grep import GrepBaseline

    units = [
        CodeUnit(doc_id="a.py::f", path="a.py", qualname="f", start_line=1, end_line=2,
                 code='def f():\n    raise ValueError("expected time as the first column")\n'),
        CodeUnit(doc_id="b.py::g", path="b.py", qualname="g", start_line=1, end_line=2,
                 code="def g():\n    first = column = time = None\n"),
    ]
    g = GrepBaseline().index(units)
    ranked, n = g.search_with_count('"expected time as the first column"', k=10)
    assert n == 1 and ranked[0][0] == "a.py::f"
    _, n_kw = g.search_with_count("expected time first column", k=10)
    assert n_kw == 2
