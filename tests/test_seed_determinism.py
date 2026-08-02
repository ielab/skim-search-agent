"""Seed + temperature threading and multi-seed runs (offline, stub policy — no GPU).

(a) seed/temperature are first-class on RetrieverConfig + AgentArgs and reach the
    retriever-registry builder (the factory) untouched.
(b) a multi-seed run writes a SEPARATE seed=<N> run dir (each with results.json),
    while a deterministic floor runs ONCE with no seed segment.

The stub (KeywordPolicy) ignores the sampling params — that's fine: we only assert
they THREAD THROUGH (the params matter once a real LLM backend is wired in).
"""
import json
import os

from evaluation.config import (
    AgentArgs,
    DatasetArgs,
    EvaluationArgs,
    OutputArgs,
    RetrieverArgs,
    RunConfig,
    results_dir_for,
)


def test_retriever_config_carries_seed_and_temperature():
    cfg = RunConfig(
        dataset=DatasetArgs(name="fixture"),
        retriever=RetrieverArgs(name="agent_codefix"),
        agent=AgentArgs(policy="stub", temperature=0.3, seed=7),
    ).retriever_config()
    assert cfg.temperature == 0.3
    assert cfg.seed == 7


def test_seed_temperature_reach_the_registry_builder(monkeypatch):
    """RunConfig -> retriever_config() -> build_factory: the cfg the registry builder
    receives must carry the exact seed/temperature (stub policy: no model needed)."""
    import agent_search.retrievers.registry as reg

    seen = {}
    orig = reg.build_factory

    def spy(name, cfg=None):
        seen["cfg"] = cfg
        return orig(name, cfg)

    monkeypatch.setattr(reg, "build_factory", spy)
    # run_eval imports build_factory lazily from the module, so patching reg is enough
    from evaluation.run_eval import make_factory_from_config

    config = RunConfig(
        dataset=DatasetArgs(name="fixture"),
        retriever=RetrieverArgs(name="agent_codefix"),
        agent=AgentArgs(policy="stub", temperature=0.25, seed=11),
    )
    make_factory_from_config(config)
    assert seen["cfg"].temperature == 0.25
    assert seen["cfg"].seed == 11


def test_make_factory_threads_seed_temperature(monkeypatch):
    """The CLI's _make_factory passes seed/temperature into the RetrieverConfig."""
    import agent_search.retrievers.registry as reg
    from evaluation.run_eval import _make_factory

    seen = {}
    orig = reg.build_factory
    monkeypatch.setattr(reg, "build_factory",
                        lambda name, cfg=None: (seen.update(cfg=cfg), orig(name, cfg))[1])
    _make_factory("agent_codefix", policy="stub", temperature=0.9, seed=3)
    assert seen["cfg"].temperature == 0.9 and seen["cfg"].seed == 3


# NOTE: results_dir_for() no longer flattens seed/steps/etc. into the run-dir path — see
# its docstring: "NESTED run-directory convention: <runs_dir>/<kind>/<dataset>/<model>/
# <retriever>. Everything else (level, k, max_steps, seed, corpus_limit, git rev, …) is
# recorded in that dir's config.json — NOT flattened into a long folder name." This is a
# deliberate design change (short, browsable paths; summarize_runs reads seed etc. from
# config.json instead), so the old `seed=<N>` path-segment assertion no longer applies —
# a multi-seed ablation must be pointed at a distinct `runs_dir=` per seed by the caller
# (also documented on results_dir_for) rather than relying on automatic seed-keyed dirs.
def test_run_dir_has_no_seed_segment_seed_lives_in_config_json_instead():
    """Locks in the current (intentional) behavior: the run-dir path is seed-agnostic for
    both agent and floor retrievers — seed is provenance recorded in config.json, not the
    directory name."""
    agent = RunConfig(
        dataset=DatasetArgs(name="fixture"),
        retriever=RetrieverArgs(name="agent_codefix"),
        agent=AgentArgs(policy="stub", seed=2),
    )
    rd = results_dir_for(agent)
    assert not any(p.startswith("seed=") for p in os.path.basename(rd).split("__"))

    floor = RunConfig(
        dataset=DatasetArgs(name="fixture"),
        retriever=RetrieverArgs(name="bm25_local"),
    )
    assert not any(p.startswith("seed=") for p in os.path.basename(results_dir_for(floor)).split("__"))


def _run_seed(seed, runs_dir=None, results_dir=None):
    from evaluation.datasets import fixture_instances
    from evaluation.run_eval import evaluate, make_factory_from_config

    output = OutputArgs(runs_dir=runs_dir) if runs_dir else OutputArgs(results_dir=results_dir)
    config = RunConfig(
        dataset=DatasetArgs(name="fixture"),
        retriever=RetrieverArgs(name="agent_codefix"),
        agent=AgentArgs(policy="stub", domain="code", seed=seed),
        evaluation=EvaluationArgs(level="function", k=(1, 10)),
        output=output,
    ).resolved()
    rd = results_dir_for(config)
    evaluate(fixture_instances(), make_factory_from_config(config),
             ks=[1, 10], level="function", results_dir=rd)
    return rd


def test_same_runs_dir_collapses_seeds_into_one_dir(tmp_path):
    """Current (intentional) behavior: since the path no longer encodes seed, multiple
    seeds sharing one `runs_dir` land in the SAME directory — this documents that a
    multi-seed sweep needs a distinct `results_dir`/`runs_dir` per seed (see
    results_dir_for's NOTE), not automatic seed-keyed subdirs."""
    runs_dir = str(tmp_path / "runs")
    dirs = [_run_seed(s, runs_dir=runs_dir) for s in (0, 1, 2)]
    assert len(set(dirs)) == 1                            # all three collapse to one dir
    res = json.loads(open(os.path.join(dirs[0], "results.json")).read())
    assert res["n"] == 1                                  # still a valid, scored run


def test_distinct_results_dir_per_seed_still_isolates_runs(tmp_path):
    """The documented workaround still works: pointing each seed at its own
    `output.results_dir` (rather than relying on the old automatic seed segment) gives
    each seed its own results.json — the programmatic mirror of a multi-seed sweep
    scripted with `--results-dir runs/.../seed=<N>` per seed."""
    dirs = [_run_seed(s, results_dir=str(tmp_path / f"seed={s}")) for s in (0, 1, 2)]

    assert len(set(dirs)) == 3                            # three DISTINCT run dirs
    for s, rd in zip((0, 1, 2), dirs):
        res_path = os.path.join(rd, "results.json")
        assert os.path.isfile(res_path)
        res = json.loads(open(res_path).read())
        assert res["n"] == 1                             # the fixture scored
