"""A run directory holds ONE experiment: resuming with a different configuration is refused,
multi-seed runs get one directory per seed, and results.json is a summary (no row copies)."""
import json
import os

import pytest

from agent_search.evaluation import run_eval as R


def _base_args(**over):
    from argparse import Namespace
    a = dict(dataset="doc_fixture", retriever="agent_research_snip", model=None, dense_model=None,
             policy="stub", backend="vllm", api_base="http://localhost:8000/v1", max_steps=5,
             temperature=0.6, seed=42, level="function", k=[1, 3, 5, 10], corpus_limit=None,
             limit=None, workers=1, runs_dir="runs", results_dir=None)
    a.update(over)
    return Namespace(**a)


def test_identity_view_ignores_how_much_but_not_what():
    a = R._run_config_dict(_base_args(), "general")
    b = R._run_config_dict(_base_args(limit=3, workers=4), "general")
    assert R._identity_view(a) == R._identity_view(b)
    c = R._run_config_dict(_base_args(max_steps=100), "general")
    assert R._identity_view(a) != R._identity_view(c)


def test_resume_into_a_different_experiment_is_refused(tmp_path, monkeypatch):
    rd = tmp_path / "run"
    cfg = R._run_config_dict(_base_args(), "general")
    R._write_run_config(str(rd), _base_args(), "general", cfg=cfg)
    # same experiment: fine
    R._check_run_identity(str(rd), R._run_config_dict(_base_args(limit=2), "general"))
    # a changed backend knob is a different experiment
    monkeypatch.setenv("STRUCTURED_BACKEND", "lucene")
    other = R._run_config_dict(_base_args(), "general")
    with pytest.raises(SystemExit) as ei:
        R._check_run_identity(str(rd), other)
    assert "env_knobs" in str(ei.value) and "--allow-config-drift" in str(ei.value)
    R._check_run_identity(str(rd), other, allow_drift=True)      # explicit override
    monkeypatch.delenv("STRUCTURED_BACKEND")
    with pytest.raises(SystemExit):
        R._check_run_identity(str(rd), R._run_config_dict(_base_args(seed=7), "general"))


def test_multi_seed_runs_land_in_separate_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = ["run_eval", "--dataset", "doc_fixture", "--retriever", "agent_research_snip",
            "--policy", "stub", "--seeds", "1,2", "--runs-dir", str(tmp_path / "runs"),
            "--index-root", str(tmp_path / "idx"), "--max-steps", "4"]
    monkeypatch.setattr("sys.argv", argv)
    R.main()
    dirs = sorted(p.name for p in (tmp_path / "runs").rglob("seed=*"))
    assert dirs == ["seed=1", "seed=2"]
    for d in (tmp_path / "runs").rglob("seed=*"):
        assert (d / "rows.jsonl").exists() and (d / "config.json").exists()
        cfg = json.loads((d / "config.json").read_text())
        assert cfg["seed"] == int(d.name.split("=")[1])
        summary = json.loads((d / "results.json").read_text())
        assert "rows" not in summary and summary["rows_file"] == "rows.jsonl"
        assert summary["n"] == 1


def test_budget_stops_count_toward_timeout_rate():
    rows = [{"stopped": s, "x": 1.0} for s in ("answer", "max_steps", "ctx_budget", "max_turns")]
    agg = R._aggregate(rows)
    assert agg["timeout_rate"] == pytest.approx(0.75)


def test_missing_artifact_aborts_the_run_instead_of_erroring_every_instance(tmp_path):
    from agent_search.errors import SetupError
    from agent_search.evaluation.datasets import load_dataset_by_name

    class Broken:
        name = "broken"

        def index(self, units, key=None):
            raise RuntimeError("research_x needs a persisted dense doc-embedding cache — none found")

        def search(self, q, k):
            return []

    inst = load_dataset_by_name("doc_fixture")
    with pytest.raises(SetupError):
        R.evaluate(inst, lambda: Broken(), results_dir=str(tmp_path / "r"))
    assert not os.path.exists(tmp_path / "r" / "results.json")
