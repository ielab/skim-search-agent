"""The run record: provenance snapshot, and the identity check that guards resume.

`_run_config_dict` builds the full config.json payload for a run (CLI args, env
knobs, prompt hash, package/git version). `RUN_IDENTITY_KEYS` is the subset of
that payload which defines which experiment a run directory holds; a resume
whose invocation differs on any of those keys is refused by
`_check_run_identity` unless the caller passes `allow_drift`.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from agent_search.tokens import ruler_name


def _resolve_env_knobs() -> dict:
    """Snapshot of the env-var knobs that materially define a run's retrieval condition
    (dense/BQL/indri toggles, token caps, ANN backend, agent driver/condition) but live
    outside `args`, so two run dirs could otherwise differ only by env and be
    indistinguishable on disk. Read via the same resolved constants/helpers the consuming
    code itself uses (imported, not re-read with a second copy of the default) wherever
    one is importable; an identical inline `os.environ.get(..., default)` otherwise
    (private module state or a check with no default sentinel).
    Excludes secrets (OPENAI_API_KEY/GEMINI_API_KEY) and infra-only vars (server URLs,
    device selection, cache-dir paths, parallelism toggles) that don't change results.

    Called once at run start and merged into config.json; this function only records
    knob values, it never feeds back into episode code.
    """
    import os as _os
    knobs: dict = {}

    try:
        from agent_search.tools.budgets import MAX_SECTION_TOKENS, MAX_VISIT_TOKENS, SNIPPET_TOKENS
        knobs["MAX_VISIT_TOKENS"] = MAX_VISIT_TOKENS
        knobs["MAX_SECTION_TOKENS"] = MAX_SECTION_TOKENS
        # listing-snippet window, used both by the method conditions' query-biased excerpt
        # (research_snip / *_fetch_snip) and by the visit baselines' opening
        # window (research_bm25/dense/hybrid, research_bm25_dci); see
        # agent_search.snippets. Recorded because it is a sweep axis:
        # without it a SNIPPET_TOKENS=64 run is indistinguishable from a default one in
        # config.json.
        knobs["SNIPPET_TOKENS"] = SNIPPET_TOKENS
    except Exception:
        pass

    # agent_search/evaluation/datasets.py:default_dense_model: the raw knob (unset vs an explicit
    # override), not the per-run resolved value (that's already recorded at config.json's
    # top level as `dense_model`, via args.dense_model / RunConfig.resolved()). Overrides the
    # document-domain dense embedder default only (e.g. `Qwen/Qwen3-Embedding-0.6B`); the
    # code-domain default (CodeRankEmbed) never reads this var, see that function's
    # docstring. No single importable resolved constant exists (it's a function of domain),
    # so read inline like INDRI_DENSE below.
    knobs["DENSE_MODEL"] = _os.environ.get("DENSE_MODEL")

    # agent_search/retrievers/engines.py: no importable symbol (checked inline); same
    # membership test.
    knobs["INDRI_DENSE"] = _os.environ.get("INDRI_DENSE") in ("1", "true", "yes")
    from agent_search.retrievers.lucene.engine import default_mu
    from agent_search.retrievers.lucene.adapters import indri_dense_expand_k, indri_dense_weight
    knobs["INDRI_MU"] = default_mu()
    knobs["INDRI_DENSE_W"] = indri_dense_weight()
    knobs["INDRI_DENSE_EXPAND_K"] = indri_dense_expand_k()

    try:
        from agent_search.retrievers.bql.surface import _date_range_enabled
        knobs["BQL_DATE_RANGE"] = _date_range_enabled()
    except Exception:
        pass
    # agent_search/tools/search_bql/tool.py: inline check, no importable symbol; read the
    # same env var the same way (enabled unless explicitly turned off).
    knobs["BQL_SOFT_FALLBACK"] = _os.environ.get("BQL_SOFT_FALLBACK", "1") not in ("0", "false", "no")
    # BQL_DENSE dense-fused ranking (agent_search/retrievers/bql/dense_fuse.py): the env
    # knob for a BQL condition ranked with plain BM25 (the shared `bql` engine kind), read
    # the same way `INDRI_DENSE` is above. A condition whose strategy already fuses or uses
    # dense ranking unconditionally (the `bql_fused`/`bql_dense` engine kinds) does not read
    # this var, but its value is still recorded here for provenance.
    knobs["HYBRID_RETRIEVERS"] = _os.environ.get("HYBRID_RETRIEVERS", "bm25,dense")
    knobs["HYBRID_FUSION"] = _os.environ.get("HYBRID_FUSION", "rrf")
    knobs["HYBRID_WEIGHTS"] = _os.environ.get("HYBRID_WEIGHTS", "")
    # agent_search/retrievers/reranked.py: the base retriever, the reranker and its model.
    knobs["RERANK_BASE"] = _os.environ.get("RERANK_BASE", "bm25")
    knobs["RERANK_METHOD"] = _os.environ.get("RERANK_METHOD", "cross_encoder")
    knobs["RERANK_MODEL"] = _os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    knobs["RERANK_POOL"] = int(_os.environ.get("RERANK_POOL", "100"))
    try:
        from agent_search.retrievers.bql.dense_fuse import bql_dense_enabled, RRF_K
        knobs["BQL_DENSE"] = bql_dense_enabled()
        knobs["BQL_DENSE_RRF_K"] = RRF_K
    except Exception:
        pass

    # agent_search/evaluation/agent_runner.py:ConditionAgent.search: env override only; the
    # non-env fallback is per-instance driver state, not a second env default, so there's
    # nothing further to import.
    knobs["AGENT_DRIVER"] = _os.environ.get("AGENT_DRIVER")
    try:
        from agent_search.strategies.conditions import AGENT_DEFAULT_CONDITION
        knobs["AGENT_DEFAULT_CONDITION"] = AGENT_DEFAULT_CONDITION
    except Exception:
        pass

    # agent/loop.py::run_episode: proactive context-budget early-stop (reads these env vars
    # once per episode, not at import time; see that module's comment block above
    # _last_prompt_tokens for the full rationale). Recorded here so config.json shows what
    # fraction/window this run used and whether the early-stop was active for it.
    # AGENT_CTX_STOP_FRAC >= 1.0 means the early-stop was disabled for this run.
    try:
        from agent_search.agent.loop import DEFAULT_CTX_WINDOW, DEFAULT_CTX_STOP_FRAC
        knobs["AGENT_CTX_WINDOW"] = int(_os.environ.get("AGENT_CTX_WINDOW", str(DEFAULT_CTX_WINDOW)))
        knobs["AGENT_CTX_STOP_FRAC"] = float(
            _os.environ.get("AGENT_CTX_STOP_FRAC", str(DEFAULT_CTX_STOP_FRAC)))
    except Exception:
        pass

    try:
        from agent_search.tools.search_bm25_dci.tool import BM25_DCI_TOPK
        knobs["BM25_DCI_TOPK"] = BM25_DCI_TOPK
    except Exception:
        pass
    try:
        from agent_search.tools.budgets import BM25_FETCH_TOPK, DENSE_FETCH_TOPK
        knobs["BM25_FETCH_TOPK"] = BM25_FETCH_TOPK
        knobs["DENSE_FETCH_TOPK"] = DENSE_FETCH_TOPK
        from agent_search.tools.budgets import HYBRID_FETCH_TOPK
        knobs["HYBRID_FETCH_TOPK"] = HYBRID_FETCH_TOPK
    except Exception:
        pass

    # dense/vector_index.py:choose_backend: read inline (takes n_docs as an arg, so the
    # module exposes no standalone resolved constant); same defaults as that function.
    knobs["AGENT_SEARCH_ANN"] = _os.environ.get("AGENT_SEARCH_ANN", "auto").lower()
    knobs["AGENT_SEARCH_ANN_MIN"] = int(_os.environ.get("AGENT_SEARCH_ANN_MIN", "1000000"))
    knobs["AGENT_SEARCH_ANN_PQ_MIN"] = int(_os.environ.get("AGENT_SEARCH_ANN_PQ_MIN", "8000000"))

    # dense/vector_index.py:_flat_faiss_enabled: opt-in exact faiss.IndexFlatIP fast path
    # for the `flat` backend (same stored embeddings, fp16->fp32 cast, still exact search).
    # Default off: unset falls back to a numpy fp16 matmul. This only records the knob's
    # value for provenance; episode code doesn't read it back.
    try:
        from agent_search.retrievers.dense.vector_index import _flat_faiss_enabled
        knobs["AGENT_SEARCH_FLAT_FAISS"] = _flat_faiss_enabled()
    except Exception:
        pass


    # Every length budget in the prompt path is measured in tokens; there is no character cap.
    try:
        from agent_search.agent.policies import default_ctx_tokens
        knobs["AGENT_CTX_TOKENS"] = default_ctx_tokens()
    except Exception:
        pass
    try:
        from agent_search.tools.bash.tool import BASH_MAX_TOKENS
        from agent_search.tools.read.tool import READ_MAX_LINE_TOKENS
        knobs["BASH_MAX_TOKENS"] = BASH_MAX_TOKENS
        knobs["READ_MAX_LINE_TOKENS"] = READ_MAX_LINE_TOKENS
    except Exception:
        pass
    try:
        from agent_search.tools.budgets import HYBRID_VISIT_TOPK
        knobs["HYBRID_VISIT_TOPK"] = HYBRID_VISIT_TOPK
    except Exception:
        pass
    knobs["DENSE_QUERY_STYLE"] = _os.environ.get("DENSE_QUERY_STYLE", "plain")
    knobs["DENSE_POOLING"] = _os.environ.get("DENSE_POOLING")
    knobs["DENSE_DTYPE"] = _os.environ.get("DENSE_DTYPE")
    knobs["DENSE_SEQ_LENGTH"] = _os.environ.get("DENSE_SEQ_LENGTH")
    knobs["DENSE_QUERY_SEQ_LENGTH"] = _os.environ.get("DENSE_QUERY_SEQ_LENGTH")
    knobs["DENSE_INDEX_PATH"] = _os.environ.get("DENSE_INDEX_PATH")
    knobs["BM25_INDEX_PATH"] = _os.environ.get("BM25_INDEX_PATH")
    knobs["AGENT_SEARCH_ANN_EF_SEARCH"] = int(_os.environ.get("AGENT_SEARCH_ANN_EF_SEARCH", "0") or 0)
    knobs["DEDUP_TOPK"] = int(_os.environ.get("DEDUP_TOPK", "10"))
    knobs["DEDUP_POOL_K"] = int(_os.environ.get("DEDUP_POOL_K", "100"))
    knobs["DENSE_QUERY_INSTRUCTION"] = _os.environ.get("DENSE_QUERY_INSTRUCTION")
    knobs["CLOSER_EVIDENCE_ARG_TOKENS"] = int(_os.environ.get("CLOSER_EVIDENCE_ARG_TOKENS", "32"))
    knobs["CLOSER_EVIDENCE_OBS_TOKENS"] = int(_os.environ.get("CLOSER_EVIDENCE_OBS_TOKENS", "160"))

    return knobs


def _run_config_dict(args, domain: str) -> dict:
    """Full provenance for a run: exact CLI config, resolved env knobs, the prompt profile's
    content hash, the package version, the token ruler, and the code version."""
    import datetime
    import subprocess
    cfg = {k: v for k, v in vars(args).items()}
    cfg["resolved_domain"] = domain
    try:
        cfg["env_knobs"] = _resolve_env_knobs()
    except Exception:
        cfg["env_knobs"] = {}
    if args.retriever.startswith("agent"):
        try:
            from agent_search.strategies.conditions import get_condition
            c = get_condition(args.retriever)
            cfg["prompt_task"] = c.task.name
            cfg["prompt_toolset"] = c.strategy.toolset_name or c.strategy.name
            cfg["prompt_strategy"] = c.strategy.name
            cfg["prompt_profile"] = c.task.prompt_file                 # the task's template file
            # hash the composed system prompt (task + tool declarations + manuals) for
            # reproducibility
            from agent_search.evaluation.datasets import dataset_field_profile
            profile = dataset_field_profile(args.dataset, getattr(args, "field_profile", None) or None)
            cfg["prompt_field_profile"] = profile
            cfg["prompt_sha256"] = c.system_sha256(profile)
        except Exception:
            pass
    try:
        from importlib.metadata import version as _pkg_version
        cfg["package_version"] = _pkg_version("skimsearchagent")
    except Exception:
        cfg["package_version"] = None
    cfg["token_ruler"] = ruler_name()
    try:
        cfg["git_rev"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, timeout=5).stdout.strip() or None
    except Exception:
        cfg["git_rev"] = None
    cfg["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return cfg


# The keys that define which experiment a run directory holds. A resume whose invocation
# differs on any of them is a different experiment and is refused (see _check_run_identity);
# keys that only change how much of the same experiment runs (limit, only_instances, workers,
# runs_dir, progress flags, timestamps) are deliberately excluded.
# what makes two invocations the same experiment. The served endpoint's address (api_base) is
# not in it: a model served on another port or host is the same experiment.
RUN_IDENTITY_KEYS = (
    "dataset", "retriever", "model", "dense_model", "policy", "backend",
    "max_steps", "temperature", "seed", "level", "k", "corpus_limit", "prompt_profile",
    "prompt_task", "prompt_toolset", "prompt_sha256", "resolved_domain", "env_knobs",
)


def _attach_experiment(cfg: dict, exp_path: Optional[str], overrides: Optional[list] = None) -> None:
    """Record the experiment an invocation came from: the file's path and hash, the parsed
    content with any `section.key=value` overrides applied, and the overrides themselves, so
    the run directory carries the setting that actually ran."""
    cfg["experiment_file"] = exp_path
    if not exp_path:
        return
    try:
        import hashlib as _hashlib
        from agent_search import experiment as _X
        exp = _X.load(exp_path)
        cfg["experiment_file_sha256"] = exp.sha256
        kv = {}
        for item in overrides or []:
            k, _, v = str(item).partition("=")
            kv[k.strip()] = v
        if kv:
            exp = _X.apply_overrides(exp, kv)
        cfg["experiment"] = exp.data
        cfg["experiment_overrides"] = kv
        cfg["experiment_sha256"] = (_hashlib.sha256(_X.render(exp.data).encode("utf-8")).hexdigest()
                                    if kv else exp.sha256)
    except Exception as e:  # noqa: BLE001 - provenance must never block a run
        cfg["experiment_error"] = f"{type(e).__name__}: {e}"


def _identity_view(cfg: dict) -> dict:
    """The run-identity subset of a config, JSON-normalized so tuples/lists compare equal."""
    view = {k: cfg.get(k) for k in RUN_IDENTITY_KEYS}
    return json.loads(json.dumps(view, default=str, sort_keys=True))


def _check_run_identity(results_dir: str, cfg: dict, allow_drift: bool = False) -> None:
    """Refuse to resume into a directory whose recorded config describes a different
    experiment. Without this, changing a backend knob or a seed and re-running into the same
    directory silently reports the old run as "already complete"."""
    path = os.path.join(results_dir, "config.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as fh:
            old = json.load(fh)
    except Exception:
        return                                    # unreadable: nothing to compare against
    a, b = _identity_view(old), _identity_view(cfg)
    diffs = {k: (a.get(k), b.get(k)) for k in RUN_IDENTITY_KEYS if a.get(k) != b.get(k)}
    if not diffs:
        return
    rows = os.path.join(results_dir, "rows.jsonl")
    if not os.path.exists(rows) or os.path.getsize(rows) == 0:
        return                                    # nothing was scored under the old config: a fresh start
    lines = [f"  {k}: recorded={old_v!r}  now={new_v!r}" for k, (old_v, new_v) in diffs.items()]
    msg = (f"run directory {results_dir!r} holds a DIFFERENT experiment (config.json "
           f"disagrees with this invocation):\n" + "\n".join(lines) +
           "\n  -> use a fresh --runs-dir/--results-dir for the new setting, or pass "
           "--allow-config-drift to resume anyway (the recorded config is then overwritten).")
    if allow_drift:
        import sys
        print("WARNING: " + msg, file=sys.stderr, flush=True)
        return
    raise SystemExit("error: " + msg)


def _write_run_config(results_dir: str, args, domain: str, cfg: Optional[dict] = None) -> None:
    """Write config.json next to the results (once per run; a resume re-checks identity and
    rewrites it so `started_at`/`git_rev` describe the latest invocation)."""
    cfg = cfg if cfg is not None else _run_config_dict(args, domain)
    os.makedirs(results_dir, exist_ok=True)
    tmp = os.path.join(results_dir, f"config.json.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)
    os.replace(tmp, os.path.join(results_dir, "config.json"))
