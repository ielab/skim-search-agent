"""Eval driver: dataset -> corpus -> retriever -> metrics.

Code datasets follow SweRank / LocAgent: function-level (or file-level) Acc@k plus
Recall@k and MRR over the locations the gold patch edits. Document datasets use a
shared corpus with qrels and optional answer EM/F1.

Runnable end-to-end with `--dataset fixture --retriever bm25_local` (no deps), and
on the cluster with `--dataset swebench_verified --retriever bm25_pyserini|dense`
or a staged document corpus such as `browsecomp_plus`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

from agent_search.corpus.code_repo import get_files
from agent_search.corpus.units import (
    CodeUnit,
    is_test_path,
    units_from_documents,
    units_from_python_source,
)
from agent_search.core.interfaces import Retriever

from .config import (
    AgentArgs,
    DatasetArgs,
    EvaluationArgs,
    OutputArgs,
    RetrieverArgs,
    RunConfig,
    results_dir_for,
)
from . import metrics as M
from .datasets import Instance, available_datasets, load_dataset_by_name
from .ground_truth import changed_line_ranges, gold_files, gold_units

RetrieverFactory = Callable[[], Retriever]


def _build_corpus(files: dict) -> tuple[list[CodeUnit], dict]:
    units: list[CodeUnit] = []
    by_file: dict = {}
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        if is_test_path(path):       # SWE-bench/CoRNStack convention, all conditions
            continue
        file_units = units_from_python_source(path, source)
        units.extend(file_units)
        by_file[path] = file_units
    return units, by_file


def _build_document_corpus(docs: list[dict]) -> list[CodeUnit]:
    return units_from_documents(docs)


def _to_file_ranking(doc_ids: Sequence[str]) -> list[str]:
    """Collapse unit doc_ids ("path::qual") to a deduped file ranking."""
    seen, out = set(), []
    for d in doc_ids:
        path = d.split("::", 1)[0]
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _corpus_key(inst: Instance) -> str:
    """Stable identity of a corpus = repo @ base commit. Instances sharing this
    reuse the same persisted index."""
    if inst.corpus_id:
        return inst.corpus_id.replace("/", "__")
    return f"{inst.repo.replace('/', '__')}@{inst.base_commit[:12]}"


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


def _retriever_meta(retriever: Retriever) -> dict:
    out = {"retriever": getattr(retriever, "name", type(retriever).__name__)}
    for attr, key in (
        ("tool", "tool_condition"),
        ("tool_description", "tool_description"),
        ("domain", "domain"),
        ("prompt_path", "prompt_profile_path"),
        ("max_steps", "max_steps"),
    ):
        val = getattr(retriever, attr, None)
        if val is not None:
            out[key] = val
    return out


def _trajectory_meta(retriever: Retriever) -> dict:
    """The episode metadata an agent retriever pre-serializes (rows.jsonl shape).

    `AgentRetriever` builds this in `last_trajectory_meta` (see agent/retriever.py);
    non-agent retrievers have none, so this is empty for them. The numeric cost fields
    (llm_calls, n_steps, tokens) are mean-aggregated into results.json — the
    LocAgent-style cost axis (calls + tokens to reach the result)."""
    meta = getattr(retriever, "last_trajectory_meta", None)
    return meta if isinstance(meta, dict) else {}


def _resolve_env_knobs() -> dict:
    """Snapshot of the env-var knobs that materially define a run's RETRIEVAL condition
    (dense/BQL/indri toggles, token caps, ANN backend, agent driver/condition) but live
    outside `args` and were previously invisible in config.json — two dirs could differ
    only by env and be indistinguishable on disk. Read via the SAME resolved constants/
    helpers the consuming code itself uses (imported, not re-read with a second copy of
    the default) wherever one is importable; an identical inline `os.environ.get(...,
    default)` otherwise (private module state or a check with no default sentinel).
    Excludes secrets (OPENAI_API_KEY/GEMINI_API_KEY) and infra-only vars (server URLs,
    device selection, cache-dir paths, parallelism toggles) that don't change results.

    ADDITIVE ONLY: called once at run start and merged into config.json; never read by
    any episode code, so it cannot change behavior — see TASK A of the provenance audit.
    """
    import os as _os
    knobs: dict = {}

    try:
        from agent_search.agent.tools.doc_research import (MAX_SECTION_TOKENS, MAX_VISIT_TOKENS,
                                                            SNIPPET_MAX_CHARS, SNIPPET_TOKENS)
        knobs["MAX_VISIT_TOKENS"] = MAX_VISIT_TOKENS
        knobs["MAX_SECTION_TOKENS"] = MAX_SECTION_TOKENS
        # query-biased snippet window (research_snip / *_fetch_snip / research_bm25q). Recorded
        # because it is a sweep axis: without it a SNIPPET_TOKENS=64 run is indistinguishable
        # from a default one in config.json.
        knobs["SNIPPET_TOKENS"] = SNIPPET_TOKENS
        knobs["SNIPPET_MAX_CHARS"] = SNIPPET_MAX_CHARS
    except Exception:
        pass

    # evaluation/datasets.py:default_dense_model — the raw KNOB (unset vs an explicit
    # override), not the per-run RESOLVED value (that's already recorded at config.json's
    # top level as `dense_model`, via args.dense_model / RunConfig.resolved()). Overrides the
    # GENERAL-domain dense embedder default only (e.g. `Qwen/Qwen3-Embedding-0.6B`); the
    # code-domain default (CodeRankEmbed) never reads this var — see that function's
    # docstring. No single importable resolved constant exists (it's a function of domain),
    # so read inline like INDRI_DENSE below.
    knobs["DENSE_MODEL"] = _os.environ.get("DENSE_MODEL")

    # agent/retriever.py:217 — no importable symbol (checked inline); same membership test.
    knobs["INDRI_DENSE"] = _os.environ.get("INDRI_DENSE") in ("1", "true", "yes")
    try:
        from agent_search.retrievers.structural.indri.model import (
            DEFAULT_MU, POOL_CAP, _rescore_m, _dense_weight, _dense_expand_k)
        knobs["INDRI_MU"] = DEFAULT_MU
        knobs["INDRI_POOL_CAP"] = POOL_CAP
        knobs["INDRI_RESCORE_M"] = _rescore_m()
        knobs["INDRI_DENSE_W"] = _dense_weight()
        knobs["INDRI_DENSE_EXPAND_K"] = _dense_expand_k()
    except Exception:
        pass

    try:
        from agent_search.retrievers.structural.bql.surface import _date_range_enabled
        knobs["BQL_DATE_RANGE"] = _date_range_enabled()
    except Exception:
        pass
    # doc_research.py:413 — inline check, no importable symbol; mirror it (enabled unless
    # explicitly turned off).
    knobs["BQL_SOFT_FALLBACK"] = _os.environ.get("BQL_SOFT_FALLBACK", "1") not in ("0", "false", "no")
    try:
        from agent_search.retrievers.structural.bql.executor import _PREFILTER_MIN_UNITS
        knobs["AGENT_SEARCH_BQL_PREFILTER_MIN"] = _PREFILTER_MIN_UNITS
    except Exception:
        pass
    # BQL_DENSE dense-fused ranking (agent_search/retrievers/structural/bql/dense_fuse.py):
    # the RETROFIT knob for the plain doc/docv2/docsnip/bqlvisit arms (mirrors INDRI_DENSE
    # above); research_bql_dense_visit/research_bql_dense_snip attach dense unconditionally in
    # their own arm branch and don't read this var, but it's still recorded here for provenance
    # (whether the retrofit was ALSO active for whichever arm this run used).
    try:
        from agent_search.retrievers.structural.bql.dense_fuse import bql_dense_enabled, RRF_K
        knobs["BQL_DENSE"] = bql_dense_enabled()
        knobs["BQL_DENSE_RRF_K"] = RRF_K
    except Exception:
        pass

    # agent/retriever.py:366 — env override only; the non-env fallback is per-instance
    # driver state, not a second env default, so there's nothing further to import.
    knobs["AGENT_DRIVER"] = _os.environ.get("AGENT_DRIVER")
    try:
        from agent_search.agent.retriever import AGENT_DEFAULT_CONDITION
        knobs["AGENT_DEFAULT_CONDITION"] = AGENT_DEFAULT_CONDITION
    except Exception:
        pass

    # agent/loop.py::run_episode — proactive context-budget early-stop (reads these env vars
    # once per episode, not at import time; see that module's comment block above
    # _last_prompt_tokens for the full rationale). Additive provenance only: records what
    # fraction/window THIS run used so a "dead cell" audit can tell whether the early-stop was
    # active. AGENT_CTX_STOP_FRAC >= 1.0 means the early-stop was disabled for this run.
    try:
        from agent_search.agent.loop import DEFAULT_CTX_WINDOW, DEFAULT_CTX_STOP_FRAC
        knobs["AGENT_CTX_WINDOW"] = int(_os.environ.get("AGENT_CTX_WINDOW", str(DEFAULT_CTX_WINDOW)))
        knobs["AGENT_CTX_STOP_FRAC"] = float(
            _os.environ.get("AGENT_CTX_STOP_FRAC", str(DEFAULT_CTX_STOP_FRAC)))
    except Exception:
        pass

    try:
        from agent_search.agent.tools.doc_bm25_dci import BM25_DCI_TOPK
        knobs["BM25_DCI_TOPK"] = BM25_DCI_TOPK
    except Exception:
        pass
    try:
        from agent_search.agent.tools.doc_research import BM25_FETCH_TOPK, DENSE_FETCH_TOPK
        knobs["BM25_FETCH_TOPK"] = BM25_FETCH_TOPK
        knobs["DENSE_FETCH_TOPK"] = DENSE_FETCH_TOPK
        from agent_search.agent.tools.doc_research import HYBRID_FETCH_TOPK
        knobs["HYBRID_FETCH_TOPK"] = HYBRID_FETCH_TOPK
    except Exception:
        pass

    # dense/vector_index.py:choose_backend — read inline (takes n_docs as an arg, so the
    # module exposes no standalone resolved constant); same defaults as that function.
    knobs["AGENT_SEARCH_ANN"] = _os.environ.get("AGENT_SEARCH_ANN", "auto").lower()
    knobs["AGENT_SEARCH_ANN_MIN"] = int(_os.environ.get("AGENT_SEARCH_ANN_MIN", "1000000"))
    knobs["AGENT_SEARCH_ANN_PQ_MIN"] = int(_os.environ.get("AGENT_SEARCH_ANN_PQ_MIN", "8000000"))

    # dense/vector_index.py:_flat_faiss_enabled — opt-in exact faiss.IndexFlatIP fast path
    # for the `flat` backend (same stored embeddings, fp16->fp32 cast, still exact search).
    # DEFAULT OFF: unset behaves byte-identically to the numpy fp16 matmul that predates
    # this knob. Additive-only snapshot; never read by episode code.
    try:
        from agent_search.retrievers.dense.vector_index import _flat_faiss_enabled
        knobs["AGENT_SEARCH_FLAT_FAISS"] = _flat_faiss_enabled()
    except Exception:
        pass

    # agent_search/retrievers/lexical/__init__.py:build_bm25_engine — which BM25 ENGINE every
    # bm25-family arm (bm25/bm25dci/bm25fetch/bm25q/bm25fetchsnip) uses this run: 'local'
    # (default, BM25Local's dependency-free approximation) or 'pyserini' (canonical Lucene
    # BM25). Material to results (an empirical ~0.546 top-5 Jaccard divergence between the two
    # on browsecomp_plus — see pyserini.py's module docstring), so it belongs in provenance
    # exactly like the other retrieval-condition knobs above.
    knobs["BM25_BACKEND"] = (_os.environ.get("BM25_BACKEND") or "local").strip().lower()

    # agent_search/retrievers/structural/backend.py:structured_backend — which STRUCTURAL
    # engine every BQL-family arm (search/search_v2/search_s/search_bv -> DocSearchFetch/
    # BqlVisitWorkspace) and Indri-family arm (isearch/isearch_v/isearch_s ->
    # IndriFetchWorkspace/IndriVisitWorkspace) uses this run: 'python' (default, the
    # pure-Python reference engines) or 'lucene' (the real-Lucene LMDirichlet/BM25 backend,
    # indexes/lucene_structured/). Material to results (same rationale as BM25_BACKEND above).
    knobs["STRUCTURED_BACKEND"] = (_os.environ.get("STRUCTURED_BACKEND") or "python").strip().lower()

    return knobs


def _write_run_config(results_dir: str, args, domain: str) -> None:
    """Full provenance next to the results: the numbers in results.json must be
    reproducible/attributable without re-running — exact CLI config, the prompt
    profile's content hash, and the code version."""
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
            from agent_search.agent.retriever import AGENT_DEFAULT_CONDITION
            from agent_search.prompts import load_condition
            cond = (AGENT_DEFAULT_CONDITION if args.retriever == "agent"
                    else args.retriever.replace("agent_", "", 1))
            prof = load_condition(cond, domain)
            cfg["prompt_task"] = prof.task
            cfg["prompt_toolset"] = prof.toolset
            cfg["prompt_profile"] = os.path.basename(prof.path)   # the task .md
            # hash the COMPOSED system (task + toolset + manuals) for reproducibility
            cfg["prompt_sha256"] = prof.system_sha256
        except Exception:
            pass
    try:
        cfg["git_rev"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, timeout=5).stdout.strip() or None
    except Exception:
        cfg["git_rev"] = None
    cfg["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)


# Bump when chunking/filtering changes (is_test exclusion, decorator spans, ...)
# so stale disk caches are never silently reused.
_UNITS_CACHE_VERSION = "v2"
# pool size for cutoff-free set metrics on one-shot floors (the agent ignores
# this — it returns its own accumulated set). A tool returning >this is not
# usefully 'retrieving'; set_precision will correctly read ~0 for it.
_SET_K = 1000

# Index-reuse is a perf hint: build a corpus's index once and reuse it across queries.
# Worthwhile for (a) ANY retriever over a SHARED corpus (one corpus, many queries) and
# (b) persistent-index code retrievers that may hit the same repo@commit. (a) is
# general (no name check); (b) is a small declared allowlist a new persistent retriever
# can join — not a correctness gate, just a speedup, so a new method still works without it.
_REUSABLE_INDEX_RETRIEVERS = {"bm25_local", "bm25_pyserini"}


def _should_reuse_index(retriever_name: str, instances: Sequence) -> bool:
    # Reuse one indexed retriever across queries when EITHER reason holds:
    shared_corpus = bool(instances) and instances[0].docs is not None   # (a) one corpus, many queries
    persistent_floor = retriever_name in _REUSABLE_INDEX_RETRIEVERS      # (b) a repo@commit-keyed floor
    return shared_corpus or persistent_floor


def _units_disk_path(inst: Instance, cache_dir: str) -> str:
    """Visible sibling of the repo cache (default: data/units_cache/), one pickle
    per repo@commit — the build-once parsed-corpus artifact, NOT a retrieval
    index (the method stays index-free; this caches AST chunking only)."""
    key = re.sub(r"[^A-Za-z0-9_.@-]+", "__", f"{inst.repo}@{inst.base_commit}")
    parent = os.path.dirname(os.path.normpath(cache_dir)) or "."
    return os.path.join(parent, "units_cache", f"{key}-{_UNITS_CACHE_VERSION}.pkl")


def _by_file(units: Sequence[CodeUnit]) -> dict:
    by_file: dict = {}
    for u in units:
        by_file.setdefault(u.path, []).append(u)
    return by_file


def _units_for_instance(inst: Instance, cache_dir: str, allow_clone: bool,
                        units_cache: dict, units_lock: threading.Lock
                        ) -> tuple[list[CodeUnit], dict, Optional[dict]]:
    """Build corpus units, reusing shared fixed-corpus document chunks across queries.

    Returns (units, by_file, files) where `files` is the raw {path -> source} dict the
    multi-tool localization agent reads — or None when units came from the disk cache
    (which stores only the AST chunking, not the raw files)."""
    if inst.docs is not None:
        key = _corpus_key(inst)
        with units_lock:
            cached = units_cache.get(key)
            if cached is None:
                cached = (_build_document_corpus(inst.docs), {})
                units_cache[key] = cached
            return cached[0], cached[1], {}

    if inst.files is not None:                   # inline fixture: cheap, no disk cache
        files = get_files(inst, cache_dir, allow_clone=allow_clone)
        units, by_file = _build_corpus(files)
        return units, by_file, files

    # Disk cache per repo@commit: git-archive + AST-chunking a whole repo is the
    # per-instance bottleneck, repeated across the 4 agent conditions and every
    # resume. Best-effort: corrupt/missing cache just rebuilds.
    path = _units_disk_path(inst, cache_dir)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                units = pickle.load(fh)
            return units, _by_file(units), None   # raw files not kept in the units cache
        except Exception:
            pass
    files = get_files(inst, cache_dir, allow_clone=allow_clone)
    units, by_file = _build_corpus(files)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "wb") as fh:
            pickle.dump(units, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                    # atomic; concurrent writers race harmlessly
    except Exception:
        pass
    return units, by_file, files


def _indexed_retriever(retriever_factory: RetrieverFactory, units: Sequence[CodeUnit],
                       key: str, reuse: bool, retriever_cache: dict,
                       retriever_lock: threading.Lock) -> Retriever:
    if not reuse:
        return retriever_factory().index(units, key=key)
    with retriever_lock:
        retriever = retriever_cache.get(key)
        if retriever is None:
            retriever = retriever_factory().index(units, key=key)
            retriever_cache[key] = retriever
        return retriever


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _obs_token_count(text: str) -> int:
    """Token count for a tool observation, on ONE fixed ruler (tiktoken o200k_base) across every
    arm — so structured `fetch`-a-section vs flat `visit`-whole-doc vs bm25 doc tokens are compared
    on the same scale, independent of the run's billing tokenizer. Falls back to a chars/4 proxy
    if tiktoken is unavailable. Cached module-side so we encode once per process, not per row."""
    global _O200K
    try:
        enc = _O200K
    except NameError:
        try:
            import tiktoken
            enc = _O200K = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = _O200K = None
    if enc is None:
        return len(text or "") // 4
    return len(enc.encode(text or "", disallowed_special=()))


def _score_instance(inst: Instance, retriever_factory: RetrieverFactory,
                    ks: Sequence[int], level: str, max_k: int, cache_dir: str,
                    allow_clone: bool = False, units_cache: Optional[dict] = None,
                    units_lock: Optional[threading.Lock] = None,
                    retriever_cache: Optional[dict] = None,
                    retriever_lock: Optional[threading.Lock] = None,
                    reuse_indexed_retriever: bool = False) -> dict:
    """Compute one instance's row. Deterministic skips carry a 'skipped' reason."""
    generic_gold = inst.gold_doc_ids
    units, by_file, files = _units_for_instance(
        inst,
        cache_dir,
        allow_clone,
        units_cache if units_cache is not None else {},
        units_lock if units_lock is not None else threading.Lock(),
    )
    if not units:
        return {"instance_id": inst.instance_id, "skipped": "no_units"}

    corpus_key = _corpus_key(inst)
    retriever = _indexed_retriever(
        retriever_factory,
        units,
        corpus_key,
        reuse_indexed_retriever,
        retriever_cache if retriever_cache is not None else {},
        retriever_lock if retriever_lock is not None else threading.Lock(),
    )
    # the multi-tool localization agent reads the live files (grep/open); plumb the
    # raw {path -> source} in (re-fetch if units came from the disk cache, which
    # keeps only the chunking). Per-instance retriever (reuse off for repo datasets).
    if getattr(retriever, "needs_files", False):
        if files is None:
            files = get_files(inst, cache_dir, allow_clone=allow_clone)
        retriever.set_files(files)
    # Agent retrievers return their FULL accumulated set regardless of k; one-shot
    # floors return top-k, so fetch a large pool (SET_K) for the cutoff-free set
    # metrics. @k still slices [:k] below, so headline @k numbers are unchanged.
    # a retriever that emits its own complete ranking (the agent) declares it; others
    # are padded to _SET_K. Capability flag, not a name check — so a new agentic
    # retriever class just sets `returns_full_set = True`.
    is_agent = getattr(retriever, "returns_full_set", False)
    k_fetch = max_k if is_agent else max(max_k, _SET_K)
    retrieved = retriever.search(inst.problem_statement, k=k_fetch)
    if generic_gold is not None:
        gold = set(generic_gold)
        ranking = retrieved
    elif level == "function":
        gold = gold_units(changed_line_ranges(inst.patch), by_file)
        ranking = retrieved
    else:
        # restrict to .py, non-test: only those files are in the corpus, so other
        # gold files would be unreachable and unfairly deflate file-level Acc@k
        gold = {f for f in gold_files(inst.patch)
                if f.endswith(".py") and not is_test_path(f)}
        ranking = _to_file_ranking(retrieved)

    if not gold:                          # no parseable gold location -> skip (fair)
        return {"instance_id": inst.instance_id, "skipped": "no_gold"}
    units_by_id = {u.doc_id: u for u in units}
    row = {
        "instance_id": inst.instance_id,
        "n_gold": len(gold),
        "n_retrieved": len(ranking),
        **{f"recall@{k}": M.recall_at_k(ranking, gold, k) for k in ks},
        **{f"hit@{k}": M.hit_at_k(ranking, gold, k) for k in ks},
        **{f"acc@{k}": M.acc_at_k(ranking, gold, k) for k in ks},
        **{f"precision@{k}": M.precision_at_k(ranking, gold, k) for k in ks},
        **{f"f1@{k}": M.f1_at_k(ranking, gold, k) for k in ks},
        **{f"map@{k}": M.average_precision_at_k(ranking, gold, k) for k in ks},
        **{f"ndcg@{k}": M.ndcg_at_k(ranking, gold, k) for k in ks},
        "mrr@10": M.mrr_at_k(ranking, gold, 10),
        # cutoff-FREE set metrics over the whole returned set (native to BQL/grep)
        "set_size": len(ranking),
        "set_recall": M.set_recall(ranking, gold),
        "set_precision": M.set_precision(ranking, gold),
        "set_f1": M.set_f1(ranking, gold),
        "gold_ids": sorted(gold),
        "retrieved": _retrieved_detail(ranking, units_by_id, max_k),
    }
    row.update(_retriever_meta(retriever))
    meta = _trajectory_meta(retriever)
    row.update(meta)
    # TOKEN COST, cache-aware — count each prefix-cached span ONCE. The raw per-step prompt_tokens
    # re-sends the whole growing context every turn, so summing it (meta["prompt_tokens"]) triple-
    # counts the cached initial prompt AND every earlier observation — which makes the method look
    # expensive precisely where it's cheap. The meaningful decomposition, each part counted once:
    #   initial_prompt_tokens : the fixed system+task+manual+question — step 0's input, pre-retrieval,
    #                           re-sent (cached) every later turn, so counted ONCE here.
    #   retrieved_doc_tokens  : the accumulated observation/reasoning content that actually ENTERED
    #                           the model's context window, counted once. NOTE: this used to be
    #                           `sum(_obs_token_count(o) for o in observations)` over the RAW,
    #                           uncapped tool outputs — but the policy truncates/caps what actually
    #                           gets inserted into the prompt (a single bash "read" can be 85k-370k
    #                           raw tokens while the per-step vLLM prompt_tokens tops out at a few
    #                           thousand), so that summation overcounted by up to ~67x for whole-doc/
    #                           bash baselines. Instead, derive it from the REAL per-step prompt_tokens
    #                           vLLM reported: `max(step.prompt_tokens) - initial_prompt_tokens` is the
    #                           most context the model ever actually held at once (windows/caps mean
    #                           it isn't strictly monotonic — hence max, not the last step), i.e. the
    #                           count-once cost of everything that entered the window after step 0.
    #                           Renamed internally to context_once_tokens; the field name
    #                           retrieved_doc_tokens is KEPT for downstream back-compat
    #                           (runs/_summary/*.py, scripts/summarize_runs.py all key off it).
    #   output_tokens         : the model's generated tokens (never cached; naturally once).
    # total_tokens_once sums them — the real marginal work of an episode, using only REAL usage
    # numbers (no raw-observation-text summation). The raw cumulative prompt_tokens/completion_tokens/
    # cached_input_tokens still ride in `meta` for billing reality.
    if "observations" in meta:
        traj_steps = meta.get("trajectory") or []
        step_prompt_tokens = [s.get("prompt_tokens") for s in traj_steps if s.get("prompt_tokens") is not None]
        row["initial_prompt_tokens"] = int(traj_steps[0].get("prompt_tokens") or 0) if traj_steps else 0
        if step_prompt_tokens:
            context_once_tokens = max(0, int(max(step_prompt_tokens)) - row["initial_prompt_tokens"])
            row["token_source"] = "prompt_tokens"
        else:
            # no per-step prompt_tokens on this trajectory (older run / non-vLLM backend) — fall
            # back to the old raw-observation-text estimate and flag it as such, since it can
            # badly overcount relative to what the model actually saw.
            context_once_tokens = sum(_obs_token_count(str(o)) for o in (meta.get("observations") or []))
            row["token_source"] = "fallback"
        row["context_once_tokens"] = context_once_tokens
        row["retrieved_doc_tokens"] = context_once_tokens
        row["output_tokens"] = int(meta.get("completion_tokens") or 0)
        row["total_tokens_once"] = (row["initial_prompt_tokens"] + row["retrieved_doc_tokens"]
                                    + row["output_tokens"])
        row["read_tokens"] = row["retrieved_doc_tokens"]     # back-compat name, now a REAL token count
        # REASONING (generation) tokens — a SUBSET already inside output_tokens, itemized (NOT added
        # on top, so total_tokens_once stays correct). OpenAI reasoning models report it in usage
        # (meta["reasoning_tokens"]); Tongyi/vLLM instead emit it inline as <think>…</think> with no
        # usage field, so fall back to counting those spans on the same ruler. Prefer usage; else spans.
        usage_reason = int(meta.get("reasoning_tokens") or 0)
        think_reason = sum(_obs_token_count(m) for s in traj_steps
                           for m in _THINK_RE.findall(str(s.get("raw_output") or "")))
        row["reasoning_tokens"] = usage_reason or think_reason
    # CODE-FIX arm: the end-to-end metric is fix-file-ok (did the <fix>'s file: hit a gold-patch
    # file), scored on the fix the agent committed to — not the @k of an empty ranking. The
    # retrieval @k columns still compute above (all ~0 for this arm) so the row shape is uniform.
    if meta.get("fix_text"):
        from evaluation.fix_scoring import fix_file, score_fix
        ok, _ = score_fix(meta["fix_text"], inst.patch)
        row["fix_file_ok"] = 1.0 if ok else 0.0
        row["predicted_file"] = fix_file(meta["fix_text"])
        # PATCH mode (task taskfix_patch): compile the <fix> SEARCH/REPLACE edits into a
        # git-applyable unified diff against the base_commit `files`, carried in rows.jsonl as
        # `model_patch` for later sb-cli submission (real FAIL_TO_PASS/PASS_TO_PASS grading).
        # A no-op for the prose `codefix` twins (their fix_text has no SEARCH/REPLACE -> "").
        if files:
            from evaluation.patch_synthesis import synthesize_patch
            patch, prep = synthesize_patch(meta["fix_text"], files)
            if patch:
                row["model_patch"] = patch
                row["patch_n_edits"] = prep.n_edits
                row["patch_n_applied"] = prep.n_applied
    # DEEP-RESEARCH arm: grounded EM/F1 (the answer must also appear in the tool evidence) plus
    # gold-doc coverage. QA datasets carry a gold answer; the doc arm surfaces evidence.
    if inst.answer:
        from evaluation.doc_scoring import gold_doc_coverage, score_answer
        pred = row.get("final_answer", "")
        # Pass surfaced_docs + gold_doc_ids so score_answer emits MuSiQue's SUPPORT F1 (its paired
        # paper metric alongside answer-F1) whenever the dataset carries gold supporting-doc ids.
        row.update(score_answer(pred, inst.answer, meta.get("observations", []),
                                surfaced_docs=meta.get("surfaced_docs"),
                                gold_doc_ids=inst.gold_doc_ids))
        row["gold_answer"] = inst.answer
        row["question"] = inst.problem_statement   # carried for the optional LLM judge (evaluation/llm_judge.py)
        if inst.gold_doc_ids is not None and meta.get("surfaced_docs") is not None:
            row["gold_doc_coverage"] = gold_doc_coverage(meta["surfaced_docs"], inst.gold_doc_ids)
    return row


def _retrieved_detail(ranking: Sequence[str], units_by_id: dict, n: int) -> list[dict]:
    """What was actually retrieved, reviewable without re-running.

    Code units carry their exact source location (path + 1-based line span +
    first-line snippet) — the full code is reproducible from repo@base_commit, so
    rows.jsonl stays small. Non-code/fixed-corpus docs (and file-level rankings)
    carry the doc id, plus title when the unit has one.
    """
    out = []
    for rank, doc_id in enumerate(ranking[:n], start=1):
        u = units_by_id.get(doc_id)
        if u is None:                       # file-level rank entry or unknown id
            out.append({"rank": rank, "doc_id": doc_id})
            continue
        ent = {"rank": rank, "doc_id": doc_id, "path": u.path,
               "lines": [u.start_line, u.end_line]}
        first = next((ln for ln in (u.code or "").strip().splitlines() if ln.strip()), "")
        if first:
            ent["snippet"] = first.strip()[:120]
        if u.title:
            ent["title"] = str(u.title)[:120]
        out.append(ent)
    return out


def _aggregate(scored: list) -> dict:
    if not scored:
        return {}
    n = len(scored)
    # UNION of numeric metric keys across ALL rows (not just scored[0], and not requiring a key be
    # in EVERY row). This is load-bearing for fix_file_ok, which exists only on rows that COMMITTED a
    # fix — the old all-rows filter silently dropped it, so code runs reported no accuracy. Each key
    # is averaged over the rows that HAVE it: for all-rows metrics that's unchanged; for fix_file_ok
    # it's the conditional (when-committed) accuracy. Pair it with `timeout_rate` to read the overall.
    keys: set = set()
    for r in scored:
        keys.update(k for k, v in r.items()
                    if k not in _META_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool))
    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in scored if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            agg[k] = sum(vals) / len(vals)
    # non-termination rate: the fraction that hit max_steps (how to read a timeout-diluted EM/fix_ok)
    agg["timeout_rate"] = sum(1 for r in scored if r.get("stopped") == "max_steps") / n
    return agg


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
    """Instances NOT yet recorded in results_dir/rows.jsonl (scored or skipped).
    Used to short-circuit finished runs BEFORE building backends / starting servers."""
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
    errors are NOT (so they retry next run). Aggregate is written to results.json.

    `workers > 1` scores instances concurrently (thread pool) — the efficient pattern
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
    counters = {"err": 0}
    todo = [inst for inst in instances if inst.instance_id not in done]

    def record(inst, row, err):
        with lock:
            if err is not None:
                counters["err"] += 1
                print(f"  [error] {inst.instance_id}: {type(err).__name__}: {err}")
                return
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
        except Exception as e:  # noqa: BLE001 — long runs must survive one bad repo
            return inst, None, e

    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, inst) for inst in todo]
                for fut in _progress_done(futures, progress):
                    record(*fut.result())
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
        with open(os.path.join(results_dir, "results.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    return result


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


def make_factory_from_config(config: RunConfig) -> RetrieverFactory:
    """Build a retriever factory from structured Python arguments."""
    from agent_search.retrievers.registry import build_factory
    return build_factory(config.retriever.name, config.retriever_config())


def run_config(config: RunConfig, progress: bool = False) -> dict:
    """Run evaluation from a structured config object — the minimal programmatic
    entry point. It resolves the config, builds the factory, and calls ``evaluate``.
    The CLI ``main()`` below shares that same ``evaluate`` core but ADDS the run-dir
    layout, ``--check-complete`` short-circuit, the "already complete" re-report, and
    pending-instance skipping; this function deliberately has none of that.

    This is the public Python equivalent of ``python -m evaluation.run_eval``.
    """
    config = config.resolved()
    instances = load_dataset_by_name(
        config.dataset.name,
        limit=config.dataset.limit,
        corpus_limit=config.dataset.corpus_limit,
    )
    rd = results_dir_for(config)
    _write_run_config(rd, config.as_namespace(), config.agent.domain or "code")
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

    ap.add_argument("--retriever", default="bm25_local", type=_retriever_arg,
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
                    help="override the YAML prompt profile or legacy markdown prompt "
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
                         "populate with scripts/prefetch_repos.py on a node with internet")
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
    # semantic_search embedder default follows the domain: a CODE embedder over prose
    # documents (or vice versa) is wrong, so general defaults to a text embedder.
    from evaluation.datasets import dataset_domain
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

    # Agent runs are stochastic, so we pin a SINGLE fixed seed (--seeds default '42') for a
    # reproducible run. --seeds can list several (e.g. '0,1,2') to opt into a seed-to-seed
    # variance band (one SEPARATE run dir per seed); an explicit --seed forces one value.
    # Deterministic floors run ONCE with no seed (seed=None -> no seed= dir segment).
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
        # --retriever may be a comma list: check ALL of them (x every seed) in this one
        # process (the suite used to spawn 4 interpreters = 4 dataset loads = slow submit)
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
        The judge model AUTO-routes by name (gpt-* -> OpenAI API, else the served endpoint), so it
        needs no per-backend flag — same principle as the agent's --model. Doc runs only; never
        fails the run (grading is a bonus on top of the deterministic exact-match/coverage metrics)."""
        if not args.judge_model or domain != "general":
            return
        from evaluation.llm_judge import judge_run_dir, make_judge
        try:
            gen = make_judge(args.judge_model, api_base=(args.judge_api_base or args.api_base))
            js = judge_run_dir(results_dir, gen, judge_model=args.judge_model, dataset=args.dataset)
            print(f"  LLM judge ({args.judge_model}, BrowseComp-Plus protocol): "
                  f"{js['judge_accuracy'] * 100:.1f}%  ({js['n_correct']}/{js['n_judged']})")
        except Exception as e:                          # noqa: BLE001 — optional; never break the run
            print(f"  (LLM judge skipped: {type(e).__name__}: {str(e)[:120]})")

    def _run_one(seed: int | None) -> None:
        run_cfg = _config_for(args.retriever, seed).resolved()
        results_dir = results_dir_for(run_cfg)
        label = args.retriever if seed is None else f"{args.retriever} (seed={seed})"
        pending = _pending_instances(instances, results_dir)
        if not pending:
            # finished setting: never rebuild backends or rescore — just re-report
            rows, _ = _load_rows(os.path.join(results_dir, "rows.jsonl"))
            scored = [r for r in rows if "skipped" not in r]
            print(f"already complete ({len(scored)} scored, "
                  f"{len(rows) - len(scored)} skipped) -> {results_dir}/")
            for metric, val in sorted(_aggregate(scored).items()):
                print(f"  {metric:<12} {val:.4f}")
            _maybe_judge(results_dir)
            return
        _write_run_config(results_dir, run_cfg.as_namespace(), domain)
        factory = make_factory_from_config(run_cfg)
        res = evaluate(instances, factory, ks=args.k, level=args.level,
                       progress=True, results_dir=results_dir, workers=args.workers,
                       cache_dir=args.repo_cache, allow_clone=args.allow_clone,
                       # Reuse one indexed retriever across queries when it is safe (any
                       # retriever over a shared fixed corpus, plus the read-only BM25
                       # floors) — corpus tokenized/embedded once, not per episode.
                       # (AgentRetriever.last_trajectory is thread-local, so sharing
                       # under --workers cannot cross-contaminate rows.)
                       reuse_indexed_retriever=_should_reuse_index(args.retriever, instances))

        print(f"\n{label} on {args.dataset} ({args.level}-level): "
              f"n={res['n']} scored, {res['n_skipped']} skipped, {res['n_errors']} errors")
        if res["n"] == 0:
            print("  WARNING: 0 instances scored — check dataset/patch parsing "
                  "(every instance was skipped or errored).")
        for metric, val in res["metrics"].items():
            print(f"  {metric:<12} {val:.4f}")
        print(f"\nresults saved -> {results_dir}/  (results.json + rows.jsonl)")
        print("re-run the same command to resume; already-scored instances are skipped.")
        _maybe_judge(results_dir)

    for seed in _seeds_for(args.retriever):
        _run_one(seed)


if __name__ == "__main__":
    main()
