"""Eval driver: dataset -> corpus -> retriever -> metrics.

Document datasets are the primary arm: a shared corpus with qrels and optional
answer EM/F1, graded by the agent's retrieval + answer over that corpus. Code
datasets are a retained, secondary arm that follows SweRank / LocAgent instead:
function-level (or file-level) Acc@k plus Recall@k and MRR over the locations
the gold patch edits.

Runnable end-to-end with `--dataset doc_fixture --retriever bm25_pyserini` (Java 21 and
Pyserini), and on the cluster with a staged document corpus such as `browsecomp_plus`, or with
`--dataset swebench_verified --retriever bm25_pyserini|dense` for the code arm.

This module is the CLI entry point (`main`) plus explicit re-exports of the
names that live in the sibling modules below: `corpus_units.py` (building
units for an instance), `scoring.py` (scoring one instance), `identity.py`
(the run record and its identity check), and `runner.py` (the evaluate loop
and aggregation). Nothing here imports the sibling modules the other way
around, so there is no import cycle.
"""
from __future__ import annotations

import argparse
import os

from agent_search.errors import SetupError

from .config import (
    AgentArgs,
    DatasetArgs,
    EvaluationArgs,
    OutputArgs,
    RetrieverArgs,
    RunConfig,
    results_dir_for,
)
from .datasets import available_datasets, load_dataset_by_name

from .corpus_units import (
    _build_corpus,
    _build_document_corpus,
    _by_file,
    _corpus_key,
    _to_file_ranking,
    _UNITS_CACHE_VERSION,
    _units_disk_path,
    _units_for_instance,
)
from .scoring import (
    RetrieverFactory,
    _build_indexed,
    _indexed_retriever,
    _obs_token_count,
    _retrieved_detail,
    _retriever_meta,
    _score_instance,
    _SET_K,
    _SETUP_MARKERS,
    _THINK_RE,
    _trajectory_meta,
)
from .identity import (
    RUN_IDENTITY_KEYS,
    _attach_experiment,
    _check_run_identity,
    _identity_view,
    _resolve_env_knobs,
    _run_config_dict,
    _write_run_config,
)
from .runner import (
    _aggregate,
    _BUDGET_STOPS,
    _load_rows,
    _META_KEYS,
    _pending_instances,
    _progress,
    _progress_done,
    _REUSABLE_INDEX_RETRIEVERS,
    _should_reuse_index,
    evaluate,
    make_factory_from_config,
    run_config,
)


def _make_factory(name: str, model: str | None = None,
                  index_root: str = "indexes", rebuild: bool = False,
                  policy: str = "stub", backend: str = "vllm", tp: int = 1,
                  api_base: str = "http://localhost:8000/v1",
                  dense_model: str | None = None,
                  domain: str = "code", max_steps: int = 50,
                  prompt_path_override: str | None = None,
                  temperature: float = 0.6, seed: int | None = None) -> RetrieverFactory:
    """Resolve a condition name to a retriever factory via the retriever registry.
    The harness has NO per-method knowledge: each retriever registers its own
    builder next to itself (see agent_search/retrievers/registry.py). Add a method
    there + a prompt YAML; this file never changes."""
    from agent_search.retrievers.registry import RetrieverConfig, build_factory
    cfg = RetrieverConfig(
        model=model, dense_model=dense_model, index_root=index_root, rebuild=rebuild,
        policy=policy, backend=backend, tp=tp, api_base=api_base, domain=domain,
        max_steps=max_steps, prompt_override=prompt_path_override,
        temperature=temperature, seed=seed)
    return build_factory(name, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent-search retrieval eval")
    ap.add_argument("--dataset", default="fixture",
                    choices=sorted(available_datasets()))
    from agent_search.retrievers.registry import available
    _RETRIEVERS = available()        # the registry is the single source of truth

    def _retriever_arg(v: str) -> str:
        # comma list allowed (only meaningful with --check-complete)
        bad = [x for x in v.split(",") if x not in _RETRIEVERS]
        if bad:
            raise argparse.ArgumentTypeError(
                f"unknown retriever(s) {bad}; choose from {sorted(_RETRIEVERS)}")
        return v

    ap.add_argument("--retriever", default="bm25_pyserini", type=_retriever_arg,
                    help="one of %s; a comma list is allowed with --check-complete"
                         % sorted(_RETRIEVERS))
    ap.add_argument("--model", default=None, help="dense model id, or agent LLM model id")
    ap.add_argument("--dense-model", default=None,
                    help="embedding model for dense / semantic_search (default depends on "
                         "domain: CodeRankEmbed for code, bge-base for general/documents)")
    ap.add_argument("--domain", default=None, choices=["code", "general"],
                    help="prompt/embedder domain. Default: 'general' for document "
                         "benchmarks (browsecomp_plus / hotpotqa / 2wiki / musique), else 'code'.")
    ap.add_argument("--policy", default="stub", choices=["stub", "llm"],
                    help="agent query-formulation policy (llm = prompt-profile-driven LLM)")
    ap.add_argument("--check-complete", action="store_true",
                    help="exit 0 if this run config is already fully scored "
                         "(nothing to do), 3 if instances are pending; runs nothing")
    ap.add_argument("--max-steps", type=int, default=50,
                    help="agent turn budget per instance (E6 ablation knob)")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="LLM sampling temperature for agent policies (default 0.6)")
    ap.add_argument("--seed", type=int, default=None,
                    help="single sampling seed for a reproducible agent run (overrides --seeds)")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated agent seeds (default '42' = ONE fixed reproducible run). "
                         "Each seed is a SEPARATE run dir (seed=<N>); pass e.g. '0,1,2' if you ever "
                         "want a seed-to-seed variance band. Ignored for deterministic floors.")
    ap.add_argument("--prompt-profile", dest="prompt_profile", default=None,
                    help="override the condition's task: a registered task name, a registered "
                         "condition's task, or a template file (front matter + body) "
                         "(operator-ablation variants)")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "api"],
                    help="LLM backend: in-process vLLM on the GPU node (no server), "
                         "or 'api' to connect to an OpenAI-compatible server")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel size (# GPUs) for in-process vLLM")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent instances (Tongyi-style): use with agent + "
                         "--backend api (vLLM server) so vLLM continuous-batches turns")
    ap.add_argument("--repo-cache", default="data/repos",
                    help="pre-staged repo clones (read offline via git archive); "
                         "populate with git clones on a node with internet (code datasets only)")
    ap.add_argument("--allow-clone", action="store_true",
                    help="allow cloning missing repos at eval time (needs internet; "
                         "off by default so the GPU node fails fast instead of hanging)")
    ap.add_argument("--api-base", default="http://localhost:8000/v1",
                    help="endpoint for --backend api (only used in server mode)")
    ap.add_argument("--judge-model", default=None,
                    help="if set, auto-grade doc answers after the run with this model (the "
                         "BrowseComp-Plus LLM-judge protocol). AUTO-routed by name like --model "
                         "(gpt-* -> OpenAI API, else a served endpoint) — no per-backend flag. "
                         "Independent of --model (use a different, fixed judge to avoid self-grading).")
    ap.add_argument("--judge-api-base", default=None,
                    help="served endpoint for a non-OpenAI --judge-model (default: reuse --api-base)")
    ap.add_argument("--rejudge", action="store_true",
                    help="re-grade rows that already carry a judge verdict (default: only "
                         "ungraded rows are sent to the judge)")
    ap.add_argument("--allow-config-drift", action="store_true", dest="allow_config_drift",
                    help="resume into a run dir even if its config.json describes a different "
                         "experiment (default: refuse — see RUN_IDENTITY_KEYS)")
    ap.add_argument("--experiment-override", action="append", default=None, dest="experiment_override",
                    metavar="SECTION.KEY=VALUE",
                    help="an override applied to --experiment-file, recorded in config.json (repeatable)")
    ap.add_argument("--experiment-file", default=None, dest="experiment_file",
                    help="the experiment file this invocation was derived from (recorded in "
                         "config.json together with its content; see agent_search/experiment.py)")
    ap.add_argument("--level", default="function", choices=["function", "file"])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10],
                    help="report cutoffs; covers LocAgent's file Acc@1/3/5 and "
                         "function Acc@5/10 for literature comparability")
    ap.add_argument("--limit", type=int, default=None, help="cap #instances")
    ap.add_argument("--only-instances", default=None,
                    help="path to a newline-separated file of instance_ids; restrict the "
                         "run to exactly those (INSTANCE-SHARDING entry point for "
                         "scripts/shard_cell.sh — one cell's remaining episodes split "
                         "across N parallel jobs). Applied AFTER --limit below, so pass "
                         "this WITHOUT --limit for predictable results: combining both "
                         "lets --limit's slice silently drop ids this file wanted.")
    ap.add_argument("--corpus-limit", type=int, default=None,
                    help="cap fixed-corpus documents for shared document datasets; "
                         "ignored by code datasets")
    ap.add_argument("--index-root", default="indexes",
                    help="where persistent indexes live (default: indexes/)")
    ap.add_argument("--rebuild", action="store_true", help="force index rebuild")
    ap.add_argument("--runs-dir", default="runs",
                    help="root for saved results (default: runs/)")
    ap.add_argument("--results-dir", default=None,
                    help="override the exact results dir (default: runs/<config>)")
    args = ap.parse_args()

    # keep Pyserini's in-process JVM/logging quiet (the noisy index build is the
    # subprocess, captured in bm25_pyserini; this quiets the search-side logger)
    import logging
    logging.getLogger("pyserini").setLevel(logging.WARNING)

    # domain = code (SWE-bench localization) vs general (deep-research over a document
    # corpus). Auto-detected from the dataset unless --domain overrides. The dense /
    # semantic_search embedder default follows the domain: a code embedder over prose
    # documents (or vice versa) is wrong, so general defaults to a text embedder.
    from agent_search.evaluation.datasets import dataset_domain
    domain = dataset_domain(args.dataset, args.domain)

    def _config_for(retriever: str, seed: int | None = None) -> RunConfig:
        return RunConfig(
            dataset=DatasetArgs(
                name=args.dataset,
                limit=args.limit,
                corpus_limit=args.corpus_limit,
                repo_cache=args.repo_cache,
                allow_clone=args.allow_clone,
            ),
            retriever=RetrieverArgs(
                name=retriever,
                model=args.model,
                dense_model=args.dense_model,
                index_root=args.index_root,
                rebuild=args.rebuild,
            ),
            agent=AgentArgs(
                policy=args.policy,
                backend=args.backend,
                tp=args.tp,
                api_base=args.api_base,
                domain=domain,
                max_steps=args.max_steps,
                prompt_profile=args.prompt_profile,
                temperature=args.temperature,
                seed=seed,
            ),
            evaluation=EvaluationArgs(level=args.level, k=tuple(args.k), workers=args.workers),
            output=OutputArgs(runs_dir=args.runs_dir, results_dir=args.results_dir),
        )

    # Agent runs are stochastic, so we pin a single fixed seed (--seeds default '42') for a
    # reproducible run. --seeds can list several (e.g. '0,1,2') to opt into a seed-to-seed
    # variance band (one separate run dir per seed); an explicit --seed forces one value.
    # Deterministic floors run once with no seed (seed=None -> no seed= dir segment).
    def _seeds_for(retriever: str) -> list:
        if not retriever.startswith("agent"):
            return [None]                          # floors are deterministic: run once
        if args.seed is not None:
            return [args.seed]                     # explicit single-seed run
        return [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    instances = load_dataset_by_name(args.dataset, limit=args.limit,
                                     corpus_limit=args.corpus_limit)
    if args.only_instances:
        if not os.path.exists(args.only_instances):
            raise SystemExit(f"--only-instances file not found: {args.only_instances}")
        with open(args.only_instances) as fh:
            wanted = {ln.strip() for ln in fh if ln.strip()}
        if not wanted:
            raise SystemExit(f"--only-instances file is empty: {args.only_instances}")
        instances = [i for i in instances if i.instance_id in wanted]  # dataset order preserved
    # `domain` / `dense_model` were resolved above (auto-detected from the dataset)
    if "," in args.retriever and not args.check_complete:
        raise SystemExit("--retriever comma lists are only valid with --check-complete")
    if args.check_complete:
        # --retriever may be a comma list: check all of them (x every seed) in this one
        # process, loading the dataset once instead of once per interpreter per retriever
        any_pending = False
        for name in args.retriever.split(","):
            for seed in _seeds_for(name):
                rd = results_dir_for(_config_for(name, seed))
                n_pend = len(_pending_instances(instances, rd))
                status = "DONE" if n_pend == 0 else "PENDING"
                any_pending |= n_pend > 0
                tag = name if seed is None else f"{name} seed={seed}"
                print(f"{tag} {status} {len(instances) - n_pend}/{len(instances)}")
        raise SystemExit(3 if any_pending else 0)

    def _maybe_judge(results_dir: str) -> None:
        """Auto-grade doc answers with the BrowseComp-Plus LLM judge when --judge-model is set.
        The judge model auto-routes by name (gpt-* -> OpenAI API, else the served endpoint), so it
        needs no per-backend flag, the same principle as the agent's --model. Doc runs only; never
        fails the run (grading is a bonus on top of the deterministic exact-match/coverage metrics)."""
        if not args.judge_model or domain != "general":
            return
        from agent_search.evaluation.llm_judge import judge_run_dir, make_judge
        try:
            gen = make_judge(args.judge_model, api_base=(args.judge_api_base or args.api_base))
            js = judge_run_dir(results_dir, gen, judge_model=args.judge_model, dataset=args.dataset,
                               force=args.rejudge)
            print(f"  LLM judge ({args.judge_model}, BrowseComp-Plus protocol): "
                  f"{js['judge_accuracy'] * 100:.1f}%  ({js['n_correct']}/{js['n_judged']}"
                  f"; {js['n_newly_judged']} newly graded, {js['n_judge_errors']} judge errors)")
        except Exception as e:                          # noqa: BLE001 - optional; never break the run
            print(f"  (LLM judge skipped: {type(e).__name__}: {e})")

    seeds = _seeds_for(args.retriever)
    failed: list[str] = []

    def _run_one(seed: int | None) -> None:
        run_cfg = _config_for(args.retriever, seed).resolved()
        results_dir = results_dir_for(run_cfg)
        if len(seeds) > 1 and seed is not None and not args.results_dir:
            # a multi-seed run is a variance band: one SEPARATE run dir per seed
            # (scripts/summarize_runs.py reads the `seed=<N>` segment).
            results_dir = os.path.join(results_dir, f"seed={seed}")
        label = args.retriever if seed is None else f"{args.retriever} (seed={seed})"
        cfg = _run_config_dict(run_cfg.as_namespace(), domain)
        _attach_experiment(cfg, getattr(args, "experiment_file", None),
                           getattr(args, "experiment_override", None))
        _check_run_identity(results_dir, cfg, allow_drift=args.allow_config_drift)
        pending = _pending_instances(instances, results_dir)
        if not pending:
            # finished setting: never rebuild backends or rescore, just re-report
            rows, _ = _load_rows(os.path.join(results_dir, "rows.jsonl"))
            scored = [r for r in rows if "skipped" not in r]
            print(f"already complete ({len(scored)} scored, "
                  f"{len(rows) - len(scored)} skipped) -> {results_dir}/")
            for metric, val in sorted(_aggregate(scored).items()):
                print(f"  {metric:<12} {val:.4f}")
            _maybe_judge(results_dir)
            return
        _write_run_config(results_dir, run_cfg.as_namespace(), domain, cfg=cfg)
        factory = make_factory_from_config(run_cfg)
        try:
            res = evaluate(instances, factory, ks=args.k, level=args.level,
                           progress=True, results_dir=results_dir, workers=args.workers,
                           cache_dir=args.repo_cache, allow_clone=args.allow_clone,
                           # Reuse one indexed retriever across queries when it is safe (any
                           # retriever over a shared fixed corpus, plus the read-only BM25
                           # floors): corpus tokenized/embedded once, not per episode.
                           # (ConditionAgent.last_trajectory is thread-local, so sharing
                           # under --workers cannot cross-contaminate rows.)
                           reuse_indexed_retriever=_should_reuse_index(args.retriever, instances))
        except SetupError as e:
            raise SystemExit(f"error: {label} on {args.dataset} cannot start — {e}") from e

        print(f"\n{label} on {args.dataset} ({args.level}-level): "
              f"n={res['n']} scored, {res['n_skipped']} skipped, {res['n_errors']} errors")
        if res["n"] == 0:
            print("  ERROR: 0 instances scored (every instance was skipped or errored) — "
                  "see the [error] lines above.")
            failed.append(label)
        for metric, val in res["metrics"].items():
            print(f"  {metric:<12} {val:.4f}")
        print(f"\nresults saved -> {results_dir}/  (results.json + rows.jsonl)")
        print("re-run the same command to resume; already-scored instances are skipped.")
        _maybe_judge(results_dir)

    for seed in seeds:
        _run_one(seed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
