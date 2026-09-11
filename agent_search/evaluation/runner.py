"""The evaluate loop: run every instance, record rows, aggregate metrics.

`evaluate` is the core loop (sequential or thread-pooled), resumable via
rows.jsonl. `_aggregate` turns scored rows into mean metrics. `run_config`
is the structured-config entry point that resolves a `RunConfig`, checks run
identity, and calls `evaluate`, the programmatic equivalent of the CLI.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Sequence

from agent_search.core.errors import SetupError

from .config import RunConfig, results_dir_for
from .datasets import Instance, load_dataset_by_name
from .identity import _check_run_identity, _run_config_dict, _write_run_config
from .scoring import RetrieverFactory, _score_instance


def _progress(instances, enabled: bool):
    """tqdm bar if available and enabled, else a plain iterator."""
    if not enabled:
        return instances
    try:
        from tqdm import tqdm
        return tqdm(instances, unit="inst", desc="eval")
    except ImportError:
        return instances


def _progress_done(futures, enabled: bool):
    """as_completed, wrapped in a tqdm bar (with a known total) if enabled."""
    if not enabled:
        return as_completed(futures)
    try:
        from tqdm import tqdm
        return tqdm(as_completed(futures), total=len(futures), unit="inst", desc="eval")
    except ImportError:
        return as_completed(futures)


_META_KEYS = ("instance_id", "n_gold", "n_retrieved", "skipped",
              "queries", "actions", "hits_per_step", "stopped", "errors_per_step",
              "trajectory", "gold_ids", "retrieved", "final_answer", "gold_answer",
              "retriever", "tool_condition",
              "tool_description", "domain", "prompt_profile_path", "max_steps",
              "action_calls", "raw_actions", "declared",
              # search->fetch arm fields (non-numeric; carried for inspection, not aggregated)
              "fix_text", "predicted_file", "surfaced_docs", "observations")


# Index-reuse is a perf hint: build a corpus's index once and reuse it across queries.
# Worthwhile for (a) any retriever over a shared corpus (one corpus, many queries) and
# (b) persistent-index code retrievers that may hit the same repo@commit. (a) is
# general (no name check); (b) is a small declared allowlist a new persistent retriever
# can join, not a correctness gate, just a speedup, so a new method still works without it.
_REUSABLE_INDEX_RETRIEVERS = {"bm25_local", "bm25_pyserini"}


def _should_reuse_index(retriever_name: str, instances: Sequence) -> bool:
    # Reuse one indexed retriever across queries when either reason holds:
    shared_corpus = bool(instances) and instances[0].docs is not None   # (a) one corpus, many queries
    persistent_floor = retriever_name in _REUSABLE_INDEX_RETRIEVERS      # (b) a repo@commit-keyed floor
    return shared_corpus or persistent_floor


def _aggregate(scored: list) -> dict:
    if not scored:
        return {}
    n = len(scored)
    # Union of numeric metric keys across all rows (not just scored[0], and not requiring a key
    # be in every row). This matters for fix_file_ok, which exists only on rows that committed
    # a fix: requiring the key in every row would drop it and make code runs report no accuracy.
    # Each key is averaged over the rows that have it: for all-rows metrics that's unchanged;
    # for fix_file_ok it's the conditional (when-committed) accuracy. Pair it with `timeout_rate`
    # to read the overall.
    keys: set = set()
    for r in scored:
        keys.update(k for k, v in r.items()
                    if k not in _META_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool))
    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in scored if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            agg[k] = sum(vals) / len(vals)
    # non-termination rate: the fraction whose episode was ended by a budget rather than by the
    # agent: the loop driver's step cap ("max_steps") or proactive context stop ("ctx_budget"),
    # or the SDK driver's turn cap ("max_turns"). How to read a budget-diluted EM/fix_ok.
    agg["timeout_rate"] = sum(1 for r in scored if r.get("stopped") in _BUDGET_STOPS) / n
    return agg


_BUDGET_STOPS = ("max_steps", "ctx_budget", "max_turns")


def _load_rows(rows_path: str) -> tuple[list, set]:
    rows, done = [], set()
    if rows_path and os.path.exists(rows_path):
        with open(rows_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue   # run killed mid-append: the partial line was never
                               # counted done, so that instance simply re-scores
                if r["instance_id"] in done:
                    continue
                rows.append(r)
                done.add(r["instance_id"])
    return rows, done


def _pending_instances(instances: Sequence[Instance], results_dir: str | None) -> list:
    """Instances not yet recorded in results_dir/rows.jsonl (scored or skipped).
    Used to short-circuit finished runs before building backends / starting servers."""
    if not results_dir:
        return list(instances)
    _, done = _load_rows(os.path.join(results_dir, "rows.jsonl"))
    return [i for i in instances if i.instance_id not in done]


def evaluate(instances: Sequence[Instance], retriever_factory: RetrieverFactory,
             ks: Sequence[int] = (1, 5, 10), level: str = "function",
             cache_dir: str = "data/repos", progress: bool = False,
             results_dir: Optional[str] = None, workers: int = 1,
             allow_clone: bool = False, reuse_indexed_retriever: bool = False) -> dict:
    """Run retrieval + metrics over instances. Incremental + resumable when
    `results_dir` is set: each finished instance is appended to rows.jsonl and
    skipped on re-runs; deterministic skips (no_units/no_gold) are cached, transient
    errors are not (so they retry next run). Aggregate is written to results.json.

    `workers > 1` scores instances concurrently (thread pool), the efficient pattern
    for the LLM agent against a vLLM server (`--backend api`): concurrent episodes let
    vLLM continuous-batch the per-turn requests. For bm25_pyserini, pre-build indexes
    first (`build_indexes`) so threads don't race on the same index dir.
    """
    assert level in ("function", "file")
    max_k = max(ks)

    rows_path = os.path.join(results_dir, "rows.jsonl") if results_dir else None
    rows, done = _load_rows(rows_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    sink = open(rows_path, "a") if rows_path else None
    lock = threading.Lock()
    units_cache: dict = {}
    units_lock = threading.Lock()
    retriever_cache: dict = {}
    retriever_lock = threading.Lock()
    counters = {"err": 0, "streak": 0, "ok": len(done)}   # rows already on disk count as successes
    todo = [inst for inst in instances if inst.instance_id not in done]
    # Fail fast when nothing works: N consecutive errors before a single success in this run
    # means the setting cannot run here (an unreachable model endpoint, a broken index), and
    # every further instance would burn its full retry budget for nothing. 0 disables.
    max_streak = int(os.environ.get("AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS", "3") or 0)

    def record(inst, row, err):
        with lock:
            if err is not None:
                counters["err"] += 1
                counters["streak"] += 1
                print(f"  [error] {inst.instance_id}: {type(err).__name__}: {err}", flush=True)
                if max_streak and counters["ok"] == 0 and counters["streak"] >= max_streak:
                    raise SetupError(
                        f"{counters['streak']} consecutive errors before any instance succeeded "
                        f"(last: {type(err).__name__}: {err}). The setting cannot run here: a model "
                        f"endpoint or an index is not reachable from this node. Fix it and rerun; "
                        f"finished instances are kept. AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS=0 disables "
                        f"this check.")
                return
            counters["streak"] = 0
            counters["ok"] += 1
            rows.append(row)
            done.add(inst.instance_id)
            if sink:
                sink.write(json.dumps(row) + "\n")
                sink.flush()

    def work(inst):
        try:
            return inst, _score_instance(inst, retriever_factory, ks, level, max_k,
                                         cache_dir, allow_clone, units_cache,
                                         units_lock, retriever_cache, retriever_lock,
                                         reuse_indexed_retriever), None
        except SetupError:
            raise                                  # a missing index/cache: abort the whole run
        except Exception as e:  # noqa: BLE001 - long runs must survive one bad instance
            return inst, None, e

    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, inst) for inst in todo]
                try:
                    for fut in _progress_done(futures, progress):
                        record(*fut.result())
                except SetupError:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        else:
            for inst in _progress(todo, progress):
                record(*work(inst))
    finally:
        if sink:
            sink.close()
    n_err = counters["err"]

    scored = [r for r in rows if "skipped" not in r]
    result = {
        "n": len(scored),
        "n_skipped": len(rows) - len(scored),
        "n_errors": n_err,
        "level": level,
        "metrics": _aggregate(scored),
        "rows": rows,
    }
    if results_dir:
        # results.json is the summary only. The per-instance rows live in rows.jsonl (the
        # durable, resumable record); embedding a second copy here would make results.json as
        # large as the multi-GB rows file for whole-document conditions.
        summary = {k: v for k, v in result.items() if k != "rows"}
        summary["rows_file"] = "rows.jsonl"
        tmp = os.path.join(results_dir, f"results.json.tmp.{os.getpid()}")
        with open(tmp, "w") as fh:
            json.dump(summary, fh, indent=2)
        os.replace(tmp, os.path.join(results_dir, "results.json"))
    return result


def make_factory_from_config(config: RunConfig) -> RetrieverFactory:
    """Build a retriever factory from structured Python arguments."""
    from agent_search.retrievers.registry import build_factory
    return build_factory(config.retriever.name, config.retriever_config())


def run_config(config: RunConfig, progress: bool = False) -> dict:
    """Run evaluation from a structured config object, the minimal programmatic
    entry point. It resolves the config, builds the factory, and calls ``evaluate``.
    ``agent_search.evaluation.run_eval.main()`` shares that same ``evaluate`` core but adds
    the run-dir layout, ``--check-complete`` short-circuit, the "already complete" re-report,
    and pending-instance skipping; this function deliberately has none of that.

    This is the public Python equivalent of ``python -m agent_search.evaluation.run_eval``.
    """
    config = config.resolved()
    instances = load_dataset_by_name(
        config.dataset.name,
        limit=config.dataset.limit,
        corpus_limit=config.dataset.corpus_limit,
    )
    rd = results_dir_for(config)
    ns = config.as_namespace()
    cfg = _run_config_dict(ns, config.agent.domain or "code")
    _check_run_identity(rd, cfg, allow_drift=bool(getattr(ns, "allow_config_drift", False)))
    _write_run_config(rd, ns, config.agent.domain or "code")
    factory = make_factory_from_config(config)
    return evaluate(
        instances,
        factory,
        ks=config.evaluation.k,
        level=config.evaluation.level,
        progress=progress,
        results_dir=rd,
        workers=config.evaluation.workers,
        cache_dir=config.dataset.repo_cache,
        allow_clone=config.dataset.allow_clone,
        reuse_indexed_retriever=_should_reuse_index(config.retriever.name, instances),
    )
