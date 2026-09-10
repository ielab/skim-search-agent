"""``skimsearchagent``: run a complete research-agent experiment with key=value arguments.

    skimsearchagent dataset=doc_fixture strategy=sieve_bm25
    skimsearchagent dataset=hotpotqa_structured strategy=search_visit model=gpt-4o-mini limit=20
    skimsearchagent dataset=browsecomp_plus_structured_full strategy=sieve \\
        model=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B backend=api api_base=http://localhost:8000/v1

``strategy`` accepts an alias (``skimsearchagent --help`` lists them) or any registered
retriever name. ``dataset`` defaults to ``doc_fixture`` (three inline documents, one question).
If ``model`` is given the agent policy defaults to ``llm``; without it the dependency-free
scripted policy runs. Every other ``key=value`` is forwarded to
``agent_search.evaluation.run_eval`` as ``--key value``; boolean flags take ``true``/``false``.

Environment knobs (``snippet_tokens=64``, ``max_visit_tokens=12000``, ``structured_backend=lucene``,
...; the full list is ``ENV_KNOBS`` / docs/CONFIGURATION.md) are exported as environment
variables BEFORE the harness is imported, so a sweep is written like any other argument and every
knob lands in the run's ``config.json``.
"""
from __future__ import annotations

import os
import sys

from agent_search.strategies import DEFAULT_STRATEGY, STRATEGIES, resolve_strategy

# key=value names accepted as environment knobs. Several tool modules read these at import
# time, so they cannot be run_eval flags; they are exported here before the harness loads and
# recorded by run_eval in config.json (`env_knobs`).
ENV_KNOBS = (
    # length budgets — TOKENS only
    "snippet_tokens", "max_visit_tokens", "max_section_tokens",
    "bash_max_tokens", "read_max_line_tokens", "grep_line_tokens",
    "agent_ctx_tokens", "agent_ctx_window", "agent_ctx_stop_frac",
    # listing depths
    "bm25_visit_topk", "bm25_fetch_topk", "dense_visit_topk", "dense_fetch_topk",
    "hybrid_visit_topk", "hybrid_fetch_topk", "autoread_topk", "bm25_dci_topk", "hybrid_pool",
    "dedup_topk", "dedup_pool_k",
    # method switches
    "bql_soft_fallback", "bql_dense", "bql_dense_rrf_k", "bql_date_range", "rrf_k",
    "indri_dense", "indri_dense_w", "indri_dense_expand_k", "indri_mu", "indri_pool_cap",
    "indri_rescore_m", "lucene_mu",
    # engine selection / models
    "structured_backend", "bm25_backend", "dense_model", "dense_query_style", "dense_query_instruction", "dense_pooling", "dense_dtype",
    "dense_index", "ann_ef_search", "bm25_index",
    "agent_driver", "reasoning_effort",
    "agent_default_condition", "vllm_api_base", "skimsearchagent_plugins",
    # index building / search backends
    "agent_search_ann", "agent_search_ann_min", "agent_search_ann_pq_min",
    "agent_search_flat_faiss", "agent_search_dense_device", "agent_search_bql_prefilter_min",
    "agent_search_data", "agent_search_dci_cache", "bm25_pyserini_threads", "bm25_pyserini_store_raw",
    "lucene_index_threads", "lucene_index_ram_mb",
    # model client
    "llm_timeout_s", "llm_retry_attempts", "llm_retry_base_s", "llm_judge_model",
    # SDK driver closer evidence
    "closer_evidence_arg_tokens", "closer_evidence_obs_tokens",
)

# run_eval `store_true` flags: accept key=true/false instead of `--key value`.
BOOL_FLAGS = ("rebuild", "allow_clone", "check_complete", "rejudge", "allow_config_drift")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

DEFAULT_DATASET = "doc_fixture"
DEFAULT_RUNS_DIR = "runs/quick"


def _usage() -> str:
    lines = [__doc__.strip(), "",
             "experiment files (one file = one complete setting; see docs/CONFIGURATION.md):",
             "  skimsearchagent run FILE.yaml [section.key=value ...]",
             "  skimsearchagent validate FILE.yaml",
             "  skimsearchagent template [library|paper] [STRATEGY]   # a complete file: every key that strategy reads",
             "", "strategies:"]
    for name in sorted(STRATEGIES):
        lines.append(f"  {name:<22} -> {STRATEGIES[name]}")
    lines += ["", "boolean flags: " + ", ".join(BOOL_FLAGS),
              "environment knobs: " + ", ".join(ENV_KNOBS)]
    return "\n".join(lines)


def parse_kv(argv: list[str]) -> dict[str, str]:
    kv: dict[str, str] = {}
    for a in argv:
        if "=" not in a:
            print(f"error: expected key=value, got {a!r} (try: skimsearchagent --help)", file=sys.stderr)
            raise SystemExit(2)                 # usage error, like argparse
        k, v = a.split("=", 1)
        kv[k.strip()] = v.strip().strip("'\"")
    return kv


def build_run_eval_argv(kv: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """key=value pairs -> (run_eval argv, environment knobs to export).

    Pure: no environment is touched here, so the mapping is unit-testable."""
    kv = dict(kv)
    dataset = kv.pop("dataset", DEFAULT_DATASET)
    strategy = kv.pop("strategy", DEFAULT_STRATEGY)
    retriever = resolve_strategy(strategy)
    args = ["--dataset", dataset, "--retriever", retriever]
    if "model" in kv:
        args += ["--model", kv.pop("model")]
        args += ["--policy", kv.pop("policy", "llm")]
    elif "policy" in kv:
        args += ["--policy", kv.pop("policy")]
    elif retriever.startswith("agent"):
        args += ["--policy", "stub"]
    args += ["--runs-dir", kv.pop("runs_dir", DEFAULT_RUNS_DIR)]

    env = {k.upper(): kv.pop(k) for k in list(kv) if k in ENV_KNOBS}

    for k, v in kv.items():
        flag = f"--{k.replace('_', '-')}"
        if k in BOOL_FLAGS:
            if v.lower() in _TRUE:
                args.append(flag)
            elif v.lower() in _FALSE:
                continue
            else:
                print(f"error: {k}= expects true/false, got {v!r}", file=sys.stderr)
                raise SystemExit(2)             # usage error, like argparse
            continue
        args += [flag, v]
    return args, env


def _run_invocation(args: list[str], env: dict[str, str]) -> int:
    # export env knobs BEFORE the harness is imported: several tool modules resolve them at
    # import, so setting them afterwards would silently have no effect.
    for k, v in env.items():
        os.environ[k] = v
        print(f">> {k}={v}", file=sys.stderr)
    print(">> python -m agent_search.evaluation.run_eval " + " ".join(args), file=sys.stderr)
    from agent_search.evaluation.run_eval import main as run_eval_main
    sys.argv = ["run_eval"] + args
    try:
        run_eval_main()
    except SystemExit as e:                 # run_eval signals failures via SystemExit
        code = e.code
        if isinstance(code, str):
            print(code, file=sys.stderr)
            return 1
        return int(code or 0)
    return 0


def _experiment_command(argv: list[str]) -> int:
    """`run FILE [section.key=value ...]`, `validate FILE`, `template [library|paper]`."""
    from agent_search import experiment as X
    cmd, rest = argv[0], argv[1:]
    if cmd == "template":
        preset = next((a for a in rest if a in ("library", "paper")), None)
        strategy = next((a for a in rest if a not in ("library", "paper")), None)
        print(X.template(preset, strategy), end="")
        return 0
    if not rest:
        print(f"error: {cmd} needs an experiment file (skimsearchagent {cmd} FILE)", file=sys.stderr)
        return 2
    overrides: dict = {}
    try:
        exp = X.load(rest[0])
        if len(rest) > 1:
            overrides = parse_kv(rest[1:])
            exp = X.apply_overrides(exp, overrides)
            exp = X.Experiment(path=rest[0], data=exp.data, sha256=exp.sha256)
    except (X.ExperimentError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args, env = X.to_invocation(exp)
    for k, v in overrides.items():
        args += ["--experiment-override", f"{k}={v}"]      # recorded in config.json
    if cmd == "validate":
        print(f"{rest[0]}: valid experiment {exp.name!r} (strategy={exp.strategy})")
        unused = X.unused_keys(exp.data)
        if unused:
            print(f"  unused by this strategy (harmless, but not read): {', '.join(unused)}")
        for note in X.requirements(exp):
            print(f"  needs: {note}")
        print("  run_eval:", " ".join(args))
        print("  env:", " ".join(f"{k}={v}" for k, v in env.items()) or "(none)")
        return 0
    for note in X.requirements(exp):
        print(f">> needs: {note}", file=sys.stderr)
    return _run_invocation(args, env)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(_usage())
        return 0
    if argv and (argv[0] in ("run", "validate", "template")
                 or (argv[0].endswith((".yaml", ".yml")) and "=" not in argv[0])):
        if argv[0].endswith((".yaml", ".yml")):
            argv = ["run"] + argv
        return _experiment_command(argv)
    kv = parse_kv(argv)
    args, env = build_run_eval_argv(kv)
    return _run_invocation(args, env)


if __name__ == "__main__":
    sys.exit(main())
