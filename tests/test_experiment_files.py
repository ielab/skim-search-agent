"""Experiment files: one file = one complete setting, translated onto the single execution path."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agent_search import cli
from agent_search import experiment as X
from agent_search.evaluation.datasets import available_datasets
from agent_search.retrievers.registry import available
from agent_search.strategies.names import resolve_strategy
from tests.lucene_support import require_jvm

require_jvm()

REPO = Path(__file__).resolve().parent.parent
SHIPPED = sorted((REPO / "configs").rglob("*.yaml"))
# training files (skimsearchagent-train-retriever) share the directory but are not experiment files
SHIPPED = [p for p in SHIPPED if "strategy" in yaml.safe_load(p.read_text())]

# retrieval keys the library no longer has; a shipped file that still carries one fails to
# validate, and the fix is in configs/, not in this test.
def _prebuild_lucene(dataset: str, index_root: str) -> None:
    """A document run opens a prebuilt Lucene structured index under `index_root`; build
    the fixture's index there, as `skimsearchagent-build-indexes` does before a real run."""
    from agent_search.evaluation.build_indexes import build
    from agent_search.evaluation.datasets import load_dataset_by_name
    build(load_dataset_by_name(dataset), index_root=index_root, retriever="search_lucene",
          progress=False)


def test_template_is_complete_and_round_trips():
    text = X.template()
    data = yaml.safe_load(text)
    X.validate(data, complete=True)
    relevant = X.relevant_keys(data["strategy"])
    for section, keys in X.SCHEMA.items():
        assert set(data.get(section) or {}) == set(relevant[section])
    paper = yaml.safe_load(X.template("paper"))
    X.validate(paper, complete=True)
    assert paper["agent"]["max_steps"] == 100 and paper["budgets"]["max_section_tokens"] == 12000
    assert yaml.safe_load(X.template("paper", "search_visit"))["budgets"]["max_visit_tokens"] == 12000


def test_files_are_scoped_to_what_the_strategy_reads():
    sv = yaml.safe_load(X.template(None, "search_visit"))
    assert set(sv["listing"]) == {"bm25_visit_topk"}          # no bm25_fetch_topk in a search_visit file
    assert "dense_model" not in sv["retrieval"] and "bm25_index" in sv["retrieval"]
    X.validate(sv, complete=True)
    dd = yaml.safe_load(X.template(None, "dedup_dense"))
    assert set(dd["listing"]) == {"dedup_topk", "dedup_pool_k", "dedup_snippet_tokens"}
    assert "dense_index" in dd["retrieval"] and "bm25_index" not in dd["retrieval"]
    code = yaml.safe_load(X.template(None, "codefix"))
    assert "listing" not in code and "repo_cache" in code["output"]
    # a key the strategy does not read is accepted, reported, and never required
    sv["listing"]["bm25_fetch_topk"] = 10
    X.validate(sv, complete=True)
    assert X.unused_keys(sv) == ["listing.bm25_fetch_topk"]
    del sv["retrieval"]["bm25_index"]
    with pytest.raises(X.ExperimentError, match="bm25_index"):
        X.validate(sv, complete=True)
    # plugin strategies (unknown names) read everything
    assert X.applies("listing", "bm25_fetch_topk", "my_plugin_strategy")


@pytest.mark.parametrize("path", SHIPPED, ids=[p.relative_to(REPO).as_posix() for p in SHIPPED])
def test_every_shipped_file_is_complete_and_names_real_things(path):
    exp = X.load(path, complete=True)
    assert resolve_strategy(exp.strategy) in available()
    assert exp.get("dataset", "name") in available_datasets()
    assert exp.name


def test_unknown_keys_are_errors():
    with pytest.raises(X.ExperimentError, match="unknown keys"):
        X.validate({"budgets": {"snippet_chars": 160}})
    with pytest.raises(X.ExperimentError, match="unknown top-level"):
        X.validate({"budget": {}})
    with pytest.raises(X.ExperimentError, match="expected int"):
        X.validate({"agent": {"max_steps": "many"}})
    with pytest.raises(X.ExperimentError, match="not complete"):
        X.validate({"dataset": {"name": "doc_fixture"}}, complete=True)


def test_translation_to_flags_and_env():
    d = X.defaults("paper")
    d["model"]["name"] = "gpt-4o-mini"
    d["model"]["seeds"] = [0, 1]
    d["evaluation"]["judge_model"] = "gpt-4o-mini"
    d["output"]["rebuild"] = True
    d["env"] = {"MY_KNOB": "7"}
    exp = X.from_dict(d)
    args, env = X.to_invocation(exp)
    assert args[:4] == ["--dataset", "hotpotqa_structured", "--retriever", "agent_research_bql_dense_snip"]
    assert args[args.index("--model") + 1] == "gpt-4o-mini"
    assert args[args.index("--policy") + 1] == "llm"
    assert args[args.index("--max-steps") + 1] == "100"
    assert args[args.index("--seeds") + 1] == "0,1" and "--seed" not in args
    assert "--rebuild" in args and "--rejudge" not in args
    assert args[args.index("--k") + 1:args.index("--k") + 5] == ["1", "3", "5", "10"]
    assert env["MAX_VISIT_TOKENS"] == "12000" and "STRUCTURED_BACKEND" not in env
    assert env["BQL_SOFT_FALLBACK"] == "1" and env["BQL_DENSE"] == "0"
    assert env["MY_KNOB"] == "7"
    # the stub policy is chosen when no model is named
    d["model"]["name"] = None
    args, _ = X.to_invocation(X.from_dict(d))
    assert args[args.index("--policy") + 1] == "stub"


def test_overrides_dotted_and_unique_bare_keys():
    exp = X.from_dict(X.defaults())
    exp = X.apply_overrides(exp, {"model.name": "gpt-4o", "snippet_tokens": "64",
                                  "output.rebuild": "true", "strategy": "search_visit",
                                  "env.FOO": "bar"})
    assert exp.get("model", "name") == "gpt-4o" and exp.get("budgets", "snippet_tokens") == 64
    assert exp.get("output", "rebuild") is True and exp.strategy == "search_visit"
    assert exp.data["env"] == {"FOO": "bar"}
    with pytest.raises(X.ExperimentError, match="unknown override"):
        X.apply_overrides(exp, {"name_": "x"})
    with pytest.raises(X.ExperimentError):
        X.apply_overrides(exp, {"name": "ambiguous?"}) if False else X.apply_overrides(exp, {"limit_": "1"})


def test_requirements_are_spelled_out():
    exp = X.from_dict(X.defaults("paper"))            # the dense-fused sieve
    notes = " ".join(X.requirements(exp))
    assert "dense embedding cache" in notes and "Lucene structured index" in notes and "Java 21" in notes
    bm25 = X.from_dict(X.defaults("paper", "search_visit"))
    assert "Pyserini" in " ".join(X.requirements(bm25))


def test_run_from_file_end_to_end_records_the_experiment(tmp_path):
    _prebuild_lucene("doc_fixture", str(tmp_path / "idx"))
    rc = cli.main(["run", str(REPO / "configs" / "smoke_doc_fixture_sieve_bm25.yaml"),
                   f"output.runs_dir={tmp_path / 'runs'}", f"output.index_root={tmp_path / 'idx'}"])
    assert rc == 0
    cfg = json.loads(next((tmp_path / "runs").rglob("config.json")).read_text())
    assert cfg["experiment_file"].endswith("smoke_doc_fixture_sieve_bm25.yaml")
    assert cfg["experiment"]["strategy"] == "sieve_bm25"
    assert cfg["experiment"]["budgets"]["snippet_tokens"] == 32
    assert cfg["env_knobs"]["SNIPPET_TOKENS"] == 32
    assert cfg["experiment_sha256"]


def test_validate_and_template_commands(capsys):
    assert cli.main(["validate", str(REPO / "configs" / "paper" / "hotpotqa_structured_sieve.yaml")]) == 0
    out = capsys.readouterr().out
    assert "valid experiment" in out and "--max-steps 100" in out and "needs:" in out
    assert cli.main(["template", "paper"]) == 0
    out = capsys.readouterr().out
    assert yaml.safe_load(out)["agent"]["max_steps"] == 100
    assert cli.main(["run"]) == 2
    assert cli.main(["validate", str(tmp := REPO / "nonexistent.yaml")]) == 2


def test_console_script_form(tmp_path):
    out = subprocess.run([sys.executable, "-m", "agent_search.cli", "template"], cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "schema: 1" in out.stdout
