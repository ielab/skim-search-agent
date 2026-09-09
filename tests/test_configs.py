"""RunConfig (the structured, programmatic run configuration) and the config.json it records."""
from pathlib import Path

import yaml

from agent_search.retrievers.registry import available
from agent_search.evaluation.config import (
    AgentArgs,
    DatasetArgs,
    EvaluationArgs,
    OutputArgs,
    RetrieverArgs,
    RunConfig,
    results_dir_for,
)
from agent_search.evaluation.datasets import available_datasets


CONFIG_DIR = Path("configs")
# Sourced from the dataset registry (like `retrievers = available()` below), NOT a hand-kept
# list: a new `register_dataset(...)` call plugs in with no test edit (parameterize, don't
# hardcode — see agent_search/evaluation/datasets.py's registry-as-extension-point docstring).


# The shipped experiment files under configs/ are validated by tests/test_experiment_files.py
# (complete schema, registered strategy and dataset). This module covers RunConfig itself.


def test_structured_run_config_maps_to_retriever_config():
    cfg = RunConfig(
        dataset=DatasetArgs(name="fixture", limit=3),
        retriever=RetrieverArgs(
            name="agent_codefix",
            model="m",
            dense_model="d",
            index_root="idx",
            rebuild=True,
        ),
        agent=AgentArgs(
            policy="llm",
            backend="api",
            api_base="http://x/v1",
            max_steps=7,
            prompt_profile="p.yaml",
        ),
        evaluation=EvaluationArgs(level="file", k=(1, 5), workers=4),
        output=OutputArgs(runs_dir="out"),
    )

    rcfg = cfg.retriever_config()

    assert rcfg.model == "m"
    assert rcfg.dense_model == "d"
    assert rcfg.index_root == "idx"
    assert rcfg.rebuild is True
    assert rcfg.policy == "llm"
    assert rcfg.backend == "api"
    assert rcfg.api_base == "http://x/v1"
    assert rcfg.max_steps == 7
    assert rcfg.prompt_override == "p.yaml"
    # NESTED run-dir convention: <runs_dir>/<kind>/<dataset>/<model>/<retriever>. Everything
    # else (level, k, max_steps, seed, ...) is recorded in that dir's config.json instead of
    # being flattened into the folder name (see results_dir_for's docstring).
    assert results_dir_for(cfg).endswith("agent/fixture/m/agent_codefix")
    assert results_dir_for(cfg) == "out/agent/fixture/m/agent_codefix"


def test_structured_run_config_auto_detects_document_domain_defaults():
    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research"),
    )

    resolved = cfg.resolved()
    rcfg = cfg.retriever_config()

    assert resolved.agent.domain == "general"
    assert resolved.retriever.dense_model == "BAAI/bge-base-en-v1.5"
    assert rcfg.domain == "general"
    assert rcfg.dense_model == "BAAI/bge-base-en-v1.5"


def test_search_fetch_conditions_compose_the_new_toolsets():
    """THE method: search -> fetch (research_snip's search_s/fetch_s); the doc baseline is
    (bm25_search, visit)."""
    from agent_search.prompts import load_condition

    assert load_condition("research_snip").tool_names == ("search_s", "fetch_s")
    assert set(load_condition("research_bm25").tool_names) == {"bm25_search", "visit"}
    # the retired localization/one-shot toolsets are gone
    names = load_condition("research_snip").tool_names
    assert "search_bql" not in names and "submit" not in names and "semantic_search" not in names


def test_run_config_fixture(tmp_path):
    from agent_search.evaluation.run_eval import run_config

    # the fixture is a DOC instance; agent_research_snip drives search -> fetch -> <answer>.
    # The no-model stub answers from what it fetched, so the run completes with 0 errors and
    # records the doc arm's headline metric column.
    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_snip", index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )

    res = run_config(cfg, progress=False)

    assert res["n"] == 1
    assert res["n_errors"] == 0
    assert "answer_em" in res["rows"][0]           # the doc arm's headline metric is recorded
    assert (tmp_path / "run" / "config.json").exists()


def test_config_json_records_env_knobs(tmp_path):
    """Provenance audit fix: MAX_VISIT_TOKENS/INDRI_DENSE/... are read at import/call time
    deep in agent_search but never used to differ config.json, so two run dirs could be
    indistinguishable on disk. config.json now carries a resolved `env_knobs` sub-dict —
    additive only, must not touch episode behavior (still 0 errors / same fixture result)."""
    import json

    from agent_search.evaluation.run_eval import run_config

    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_snip", index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )
    res = run_config(cfg, progress=False)
    assert res["n_errors"] == 0                  # additive change: behavior unaffected

    with open(tmp_path / "run" / "config.json") as fh:
        written = json.load(fh)

    assert "env_knobs" in written
    knobs = written["env_knobs"]
    # the knobs this task's audit named explicitly
    for key in ("MAX_VISIT_TOKENS", "INDRI_DENSE", "INDRI_DENSE_W", "INDRI_RESCORE_M",
                "BQL_DATE_RANGE", "BQL_SOFT_FALLBACK", "AGENT_DRIVER"):
        assert key in knobs
    assert knobs["MAX_VISIT_TOKENS"] == 12000     # the doc_research.py default, unset here


def test_config_json_env_knobs_reflect_set_env_var(tmp_path, monkeypatch):
    """A non-default env value must show up resolved in config.json, not the default.

    Uses INDRI_DENSE/AGENT_DRIVER (read live via `os.environ.get` on every call — see
    agent_search/agent/retriever.py:217,366) rather than MAX_VISIT_TOKENS, which
    doc_research.py reads once as a module-level constant AT IMPORT; reimporting that
    module mid-suite would mint a second `DocSearchFetch` class object and break
    `isinstance` checks in tests that already hold the first one.

    AGENT_DRIVER is set to "loop" (not "sdk") — "sdk" would actually reroute the stub
    run through the OpenAI Agents SDK / a real network call; "loop" is the effective
    default behavior but, unlike leaving the var unset, still exercises the "a SET env
    var is reflected" path (None -> "loop" is a real, observable change in config.json)."""
    import json

    monkeypatch.setenv("INDRI_DENSE", "1")
    monkeypatch.setenv("AGENT_DRIVER", "loop")

    from agent_search.evaluation.run_eval import run_config

    cfg = RunConfig(
        dataset=DatasetArgs(name="browsecomp_plus_fixture"),
        retriever=RetrieverArgs(name="agent_research_snip", index_root=str(tmp_path / "idx")),
        evaluation=EvaluationArgs(k=(1, 10)),
        output=OutputArgs(results_dir=str(tmp_path / "run")),
    )
    res = run_config(cfg, progress=False)
    assert res["n_errors"] == 0

    with open(tmp_path / "run" / "config.json") as fh:
        knobs = json.load(fh)["env_knobs"]

    assert knobs["INDRI_DENSE"] is True
    assert knobs["AGENT_DRIVER"] == "loop"
