"""The `skimsearchagent` / run.py launcher: key=value -> run_eval flags + env knobs."""
import subprocess
import sys
from pathlib import Path

import pytest

from agent_search import cli
from agent_search.retrievers.registry import available
from agent_search.strategies.names import DEFAULT_STRATEGY, STRATEGIES
from tests.lucene_support import require_jvm

require_jvm()

REPO = Path(__file__).resolve().parent.parent


def _prebuild_lucene(dataset: str, index_root: str) -> None:
    """A document run opens a prebuilt Lucene structured index under `index_root`; build
    the fixture's index there, as `skimsearchagent-build-indexes` does before a real run."""
    from agent_search.evaluation.build_indexes import build
    from agent_search.evaluation.datasets import load_dataset_by_name
    build(load_dataset_by_name(dataset), index_root=index_root, retriever="search_lucene",
          progress=False)


def test_every_strategy_alias_resolves_to_a_registered_retriever():
    reg = available()
    missing = {k: v for k, v in STRATEGIES.items() if v not in reg}
    assert not missing, f"strategy aliases pointing at unregistered retrievers: {missing}"
    assert DEFAULT_STRATEGY in STRATEGIES


def test_defaults_are_document_first():
    args, env = cli.build_run_eval_argv({})
    assert args[:4] == ["--dataset", "doc_fixture", "--retriever", STRATEGIES[DEFAULT_STRATEGY]]
    assert "--policy" in args and args[args.index("--policy") + 1] == "stub"
    assert env == {}


def test_model_implies_llm_policy_and_forwards_flags():
    args, env = cli.build_run_eval_argv(
        {"dataset": "hotpotqa_fixture", "strategy": "search_visit", "model": "gpt-4o-mini",
         "limit": "3", "max_steps": "7", "runs_dir": "runs/x"})
    assert args[:4] == ["--dataset", "hotpotqa_fixture", "--retriever", "agent_search_visit"]
    assert args[args.index("--model") + 1] == "gpt-4o-mini"
    assert args[args.index("--policy") + 1] == "llm"
    assert args[args.index("--limit") + 1] == "3"
    assert args[args.index("--max-steps") + 1] == "7"
    assert args[args.index("--runs-dir") + 1] == "runs/x"
    assert env == {}


def test_env_knobs_are_exported_not_forwarded():
    args, env = cli.build_run_eval_argv(
        {"snippet_tokens": "64", "max_visit_tokens": "12000", "bql_soft_fallback": "0"})
    assert env == {"SNIPPET_TOKENS": "64", "MAX_VISIT_TOKENS": "12000",
                   "BQL_SOFT_FALLBACK": "0"}
    assert "--snippet-tokens" not in args and "--bql-soft-fallback" not in args


def test_boolean_flags():
    args, _ = cli.build_run_eval_argv({"rebuild": "true", "rejudge": "false",
                                       "allow_config_drift": "1"})
    assert "--rebuild" in args and "--allow-config-drift" in args and "--rejudge" not in args
    with pytest.raises(SystemExit):
        cli.build_run_eval_argv({"rebuild": "maybe"})


def test_bad_argument_shapes():
    with pytest.raises(SystemExit):
        cli.parse_kv(["dataset"])
    assert cli.parse_kv(["a=b", "c='d'"]) == {"a": "b", "c": "d"}


def test_help_lists_strategies_and_knobs(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in STRATEGIES:
        assert name in out
    assert "snippet_tokens" in out and "max_visit_tokens" in out


def test_run_py_is_a_thin_wrapper():
    out = subprocess.run([sys.executable, "run.py", "--help"], cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "strategies:" in out.stdout


def test_end_to_end_stub_episode_on_the_document_fixture(tmp_path):
    """The README quickstart: no model, no keys, a full episode, rank metrics that mean
    something (the stub surfaces the gold doc, so recall@10 must be 1, not 0)."""
    import json
    _prebuild_lucene("doc_fixture", str(tmp_path / "idx"))
    rc = cli.main([f"runs_dir={tmp_path / 'runs'}", f"index_root={tmp_path / 'idx'}"])
    assert rc == 0
    rows = list((tmp_path / "runs").rglob("rows.jsonl"))
    assert len(rows) == 1
    row = json.loads(rows[0].read_text().splitlines()[0])
    assert row["retrieved"], "document runs must report the surfaced ranking"
    assert row["recall@10"] == 1.0
    summary = json.loads(next((tmp_path / "runs").rglob("results.json")).read_text())
    assert "rows" not in summary and summary["n"] == 1
    cfg = json.loads(next((tmp_path / "runs").rglob("config.json")).read_text())
    assert cfg["token_ruler"] in ("tiktoken:o200k_base", "whitespace")
    assert "env_knobs" in cfg and "prompt_sha256" in cfg
