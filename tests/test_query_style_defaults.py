"""A dense model is served in its own query format and with its own instruction unless a run
says otherwise: ITER's trained format and ITER's instruction for the ITER checkpoints, the plain
sub-query and the plain Qwen3-Embedding instruction for the rest. The old name `i9` still
resolves to `iter`."""
from __future__ import annotations

import pytest

from agent_search.training import queries as Q

ITER_06 = "ielabgroup/ITER-Qwen3-Embedding-0.6B"
ITER_4 = "ielabgroup/ITER-Qwen3-Embedding-4B"


def test_i9_is_an_alias_of_iter_and_renders_the_same_bytes():
    inter = [{"query": "first try", "visits": [("d1", "Doc one text", "")]}]
    assert Q.canonical_style("i9") == "iter" and "i9" not in Q.STYLES
    assert Q.render_query("i9", "Q?", "sub", inter, pre_reasoning="think") == \
        Q.render_query("iter", "Q?", "sub", inter, pre_reasoning="think")
    with pytest.raises(ValueError):
        Q.canonical_style("i10")


@pytest.mark.parametrize("model, style", [(ITER_06, "iter"), (ITER_4, "iter"), ("Qwen/Qwen3-Embedding-0.6B", "plain"),
                                          ("Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B", "plain"), (None, "plain")])
def test_each_model_gets_its_own_query_format(model, style):
    assert Q.default_style_for(model) == style


def test_iter_checkpoints_get_iters_instruction_and_lrat_keeps_the_plain_one(monkeypatch):
    from agent_search.retrievers.dense.trained import PLAIN_INSTRUCTION, TrainedRetriever, instruction_for_style
    assert instruction_for_style("iter") == Q.INSTRUCTIONS["iter"] == instruction_for_style("i9")
    assert instruction_for_style("plain") == PLAIN_INSTRUCTION == instruction_for_style(None)
    monkeypatch.delenv("DENSE_QUERY_INSTRUCTION", raising=False)
    monkeypatch.setenv("DENSE_QUERY_STYLE", "iter")
    prefix = TrainedRetriever(ITER_06, encoder=object()).query_prefix_for()
    assert prefix == f"Instruct: {Q.INSTRUCTIONS['iter']}\nQuery: "
    assert "Given the main question, the agent's reasoning" in prefix
    monkeypatch.setenv("DENSE_QUERY_STYLE", "plain")
    assert TrainedRetriever("Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B", encoder=object()).query_prefix_for() == \
        f"Instruct: {PLAIN_INSTRUCTION}\nQuery: "
    monkeypatch.setenv("DENSE_QUERY_INSTRUCTION", "custom")              # a run's own instruction still wins
    assert TrainedRetriever(ITER_06, encoder=object()).query_prefix_for() == "Instruct: custom\nQuery: "


@pytest.mark.parametrize("model, style", [(ITER_4, "iter"), ("Qwen/Qwen3-Embedding-0.6B", "plain")])
def test_an_experiment_file_with_a_null_style_runs_the_models_own(model, style, tmp_path):
    from agent_search import experiment as X
    data = X.defaults(None, "iter_dense")
    data["retrieval"]["dense_model"] = model
    data["retrieval"]["dense_query_style"] = None
    X.validate(data, complete=True)
    f = tmp_path / "exp.yaml"; f.write_text(X.render(data))
    _, env = X.to_invocation(X.load(str(f)))
    assert env["DENSE_QUERY_STYLE"] == style
