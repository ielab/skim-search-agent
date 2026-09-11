"""Experiment files: one file fully determines one setting.

    skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml
    skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml model.name=gpt-4o output.runs_dir=runs/x
    skimsearchagent template            # print a complete file with every key and its default
    skimsearchagent template paper      # ... with the paper's base configuration
    skimsearchagent validate FILE       # check a file without running it

An experiment file is YAML with a fixed schema (below). Every key that changes what runs or how
it is measured has a place in it. A file is *complete* when it names every key, and the shipped
files under `configs/` are all complete, so a reader sees the whole setting in one screen and two
settings differ exactly where their files differ. Unknown keys are errors, never ignored.

The file is translated into the same `run_eval` flags and environment knobs the key=value
launcher produces (`agent_search/cli.py`), so there is one execution path. The file's path,
content hash and parsed content are recorded in the run's `config.json`.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from agent_search.strategies.names import DEFAULT_STRATEGY, DENSE_STRATEGIES, STRATEGIES, resolve_strategy

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Key:
    default: Any
    kind: str          # "flag" (run_eval flag), "env" (environment knob), "special", "bool-flag"
    target: str = ""   # the flag name or the environment variable
    help: str = ""


# section -> key -> Key. Order is the order the template prints in.
SCHEMA: dict[str, dict[str, Key]] = {
    "dataset": {
        "name": Key("doc_fixture", "flag", "--dataset", "registered dataset (`skimsearchagent-eval --help` lists them)"),
        "limit": Key(None, "flag", "--limit", "cap the number of questions (null = all)"),
        "corpus_limit": Key(None, "flag", "--corpus-limit", "cap the shared corpus (null = all; queries whose gold docs fall outside are dropped with a warning)"),
        "only_instances": Key(None, "flag", "--only-instances", "path to a file of instance ids to run (null = all)"),
    },
    "model": {
        "name": Key(None, "special", "", "model id: gpt-* / gemini-* route to their APIs; anything else is served (backend api) or in-process (backend vllm); null = the scripted stub policy"),
        "policy": Key(None, "special", "", "llm | stub; null = llm when a model is named, else stub"),
        "backend": Key("api", "flag", "--backend", "api (an OpenAI-compatible server at api_base) | vllm (in-process)"),
        "api_base": Key("http://localhost:8000/v1", "flag", "--api-base", "OpenAI-compatible endpoint for backend=api"),
        "tp": Key(1, "flag", "--tp", "tensor parallelism for backend=vllm"),
        "temperature": Key(0.6, "flag", "--temperature", "sampling temperature"),
        "seed": Key(42, "flag", "--seed", "sampling seed (fixed for reproducibility)"),
        "seeds": Key(None, "special", "", "variance band: a list such as [0, 1, 2] -> one seed=N run directory per seed (overrides seed)"),
        "driver": Key(None, "env", "AGENT_DRIVER", "loop | sdk; null = loop, or sdk when openai-agents is installed and the model is API-served"),
        "reasoning_effort": Key("low", "env", "REASONING_EFFORT", "OpenAI reasoning models only"),
        "timeout_s": Key(600, "env", "LLM_TIMEOUT_S", "per-request timeout for API clients"),
        "retry_attempts": Key(5, "env", "LLM_RETRY_ATTEMPTS", "retries on rate limits / transient errors"),
    },
    "agent": {
        "max_steps": Key(50, "flag", "--max-steps", "tool calls before the forced final answer (paper: 100)"),
        "prompt_profile": Key(None, "flag", "--prompt-profile", "override the condition's prompt (a condition name or a .md path); null = the strategy's own"),
        "ctx_tokens": Key(115000, "env", "AGENT_CTX_TOKENS", "history budget in model tokens kept in the prompt"),
        "ctx_window": Key(131072, "env", "AGENT_CTX_WINDOW", "the serving model's context window (tokens)"),
        "ctx_stop_frac": Key(0.9, "env", "AGENT_CTX_STOP_FRAC", "force the answer once the observed prompt exceeds this fraction of ctx_window (>= 1 disables)"),
    },
    "budgets": {   # tokens only — there are no character limits anywhere
        "snippet_tokens": Key(32, "env", "SNIPPET_TOKENS", "query-biased snippet width shown per result (whitespace tokens)"),
        "max_visit_tokens": Key(12000, "env", "MAX_VISIT_TOKENS", "whole-document read cap (the paper setting)"),
        "max_section_tokens": Key(12000, "env", "MAX_SECTION_TOKENS", "named-section read cap (the paper setting)"),
        "bash_max_tokens": Key(12000, "env", "BASH_MAX_TOKENS", "shell output cap for the direct-corpus-interaction arms"),
        "read_max_line_tokens": Key(400, "env", "READ_MAX_LINE_TOKENS", "per-line cap of the DCI read tool"),
        "grep_line_tokens": Key(24, "env", "GREP_LINE_TOKENS", "per-line cap of the code grep tool"),
        "closer_evidence_arg_tokens": Key(32, "env", "CLOSER_EVIDENCE_ARG_TOKENS", "SDK driver: per-call argument excerpt in the forced-answer evidence"),
        "closer_evidence_obs_tokens": Key(160, "env", "CLOSER_EVIDENCE_OBS_TOKENS", "SDK driver: per-observation excerpt in the forced-answer evidence"),
    },
    "listing": {
        "bm25_visit_topk": Key(5, "env", "BM25_VISIT_TOPK", "results per search, search->visit arms"),
        "bm25_fetch_topk": Key(10, "env", "BM25_FETCH_TOPK", "results per search, bm25 search->fetch arms"),
        "dense_visit_topk": Key(5, "env", "DENSE_VISIT_TOPK", "results per search, dense search->visit arm"),
        "dense_fetch_topk": Key(10, "env", "DENSE_FETCH_TOPK", "results per search, dense search->fetch arm"),
        "hybrid_visit_topk": Key(5, "env", "HYBRID_VISIT_TOPK", "results per search, hybrid search->visit arm"),
        "hybrid_fetch_topk": Key(10, "env", "HYBRID_FETCH_TOPK", "results per search, hybrid search->fetch arm"),
        "hybrid_pool": Key(100, "env", "HYBRID_POOL", "candidates per side before reciprocal-rank fusion"),
        "rerank_visit_topk": Key(5, "env", "RERANK_VISIT_TOPK", "results per search, reranked search->visit arm"),
        "rerank_fetch_topk": Key(10, "env", "RERANK_FETCH_TOPK", "results per search, reranked search->fetch arm"),
        "autoread_topk": Key(5, "env", "AUTOREAD_TOPK", "documents returned in full per search, autoread arms"),
        "bm25_dci_topk": Key(10, "env", "BM25_DCI_TOPK", "documents staged per search, bounded DCI arm"),
        "dedup_topk": Key(10, "env", "DEDUP_TOPK", "results per search, dedup arms (ITER: 10)"),
        "dedup_pool_k": Key(100, "env", "DEDUP_POOL_K", "over-fetch pool before dropping already-seen docs, dedup arms (ITER: 100)"),
        "dedup_snippet_tokens": Key(64, "env", "DEDUP_SNIPPET_TOKENS", "model tokens of passage shown per hit, dedup arms (ITER: 64)"),
    },
    "retrieval": {
        "dense_model": Key(None, "flag", "--dense-model", "the ONE dense model for every dense arm (ranker, fallback, baselines); null = BAAI/bge-base-en-v1.5 for documents; a directory trained by skimsearchagent-train-retriever works too"),
        "dense_query_style": Key("plain", "env", "DENSE_QUERY_STYLE", "how the dense query is written from the agent's history: plain | mem | docs | i1..i7 (must match the trained retriever)"),
        "dense_query_instruction": Key(None, "env", "DENSE_QUERY_INSTRUCTION", "override the query instruction prefix (null = the checkpoint's serving note or the built-in table)"),
        "dense_pooling": Key(None, "env", "DENSE_POOLING", "pooling for a local checkpoint: last_token | mean | cls (null = serving note, else auto from config.json)"),
        "dense_dtype": Key(None, "env", "DENSE_DTYPE", "precision the dense encoder runs in: float32 | float16 | bfloat16 (null = the checkpoint's serving note, else float32)"),
        "dense_index": Key(None, "env", "DENSE_INDEX_PATH", "prebuilt vector index to serve (this library's cache dir, or ITER's index.faiss + index.lookup.pkl); null = the per-corpus cache under index_root"),
        "ann_ef_search": Key(0, "env", "AGENT_SEARCH_ANN_EF_SEARCH", "HNSW efSearch for a prebuilt index (0 = as built)"),
        "bm25_index": Key(None, "env", "BM25_INDEX_PATH", "prebuilt Lucene index directory for the pyserini backend; null = built under index_root"),
        "bql_soft_fallback": Key(True, "env", "BQL_SOFT_FALLBACK", "rank the closest docs (same ranker) when a Boolean query matches nothing; false = strict-Boolean ablation"),
        "bql_soft_pool": Key(100, "env", "BQL_SOFT_POOL", "size of that fallback pool"),
        "bql_date_range": Key(True, "env", "BQL_DATE_RANGE", "typed date[RANGE] scopes in BQL"),
        "bql_dense": Key(False, "env", "BQL_DENSE", "retrofit dense fusion onto the plain BQL arms (the sieve family attaches it regardless)"),
        "bql_dense_rrf_k": Key(60, "env", "BQL_DENSE_RRF_K", "RRF constant for BQL dense fusion"),
        "rrf_k": Key(60, "env", "RRF_K", "RRF constant for the hybrid baselines"),
        "hybrid_retrievers": Key("bm25,dense", "env", "HYBRID_RETRIEVERS", "the retrievers a hybrid fuses, comma-separated engine kinds (bm25, dense, bql, indri)"),
        "hybrid_fusion": Key("rrf", "env", "HYBRID_FUSION", "how a hybrid fuses them: rrf | interpolation"),
        "hybrid_weights": Key("", "env", "HYBRID_WEIGHTS", "interpolation weights, comma-separated floats, one per retriever (empty = equal)"),
        "rerank_base": Key("bm25", "env", "RERANK_BASE", "the retriever whose pool a reranker reorders (bm25, dense, hybrid, bql, indri)"),
        "rerank_method": Key("cross_encoder", "env", "RERANK_METHOD", "the reranker: cross_encoder (a sequence-classification model over (query, document) pairs)"),
        "rerank_model": Key("BAAI/bge-reranker-v2-m3", "env", "RERANK_MODEL", "the reranker model id or local directory"),
        "rerank_pool": Key(100, "env", "RERANK_POOL", "candidates taken from the base retriever before reranking"),
        "rerank_batch_size": Key(32, "env", "RERANK_BATCH_SIZE", "pairs scored per forward pass"),
        "rerank_max_length": Key(512, "env", "RERANK_MAX_LENGTH", "tokens per (query, document) pair the reranker reads"),
        "indri_dense": Key(False, "env", "INDRI_DENSE", "attach the dense belief to the Indri arm"),
        "indri_dense_w": Key(0.35, "env", "INDRI_DENSE_W", "Indri dense belief weight"),
        "indri_dense_expand_k": Key(50, "env", "INDRI_DENSE_EXPAND_K", "Indri dense pool expansion"),
        "lucene_mu": Key(2500, "env", "LUCENE_MU", "Dirichlet smoothing of the Indri scorer"),
        "ann": Key("auto", "env", "AGENT_SEARCH_ANN", "vector index backend: auto | flat | hnsw | ivfpq"),
        "ann_min": Key(1000000, "env", "AGENT_SEARCH_ANN_MIN", "corpus size above which an ANN index is used"),
        "ann_pq_min": Key(8000000, "env", "AGENT_SEARCH_ANN_PQ_MIN", "corpus size above which IVF-PQ is used"),
        "dense_device": Key(None, "env", "AGENT_SEARCH_DENSE_DEVICE", "device for the dense encoder (null = auto)"),
    },
    "evaluation": {
        "level": Key("function", "flag", "--level", "function | file (code arm only; documents ignore it)"),
        "k": Key([1, 3, 5, 10], "special", "", "rank cutoffs for @k metrics"),
        "workers": Key(1, "flag", "--workers", "concurrent episodes (use with a served model)"),
        "judge_model": Key(None, "flag", "--judge-model", "LLM judge for document answers (null = off)"),
        "judge_api_base": Key(None, "flag", "--judge-api-base", "endpoint for a served judge"),
        "rejudge": Key(False, "bool-flag", "--rejudge", "re-grade rows that already carry a verdict"),
    },
    "output": {
        "runs_dir": Key("runs", "flag", "--runs-dir", "root of the run record (runs_dir/<kind>/<dataset>/<model>/<retriever>)"),
        "results_dir": Key(None, "flag", "--results-dir", "explicit run directory (overrides runs_dir layout)"),
        "index_root": Key("indexes", "flag", "--index-root", "where persistent indexes and embedding caches live"),
        "rebuild": Key(False, "bool-flag", "--rebuild", "rebuild persistent indexes"),
        "allow_config_drift": Key(False, "bool-flag", "--allow-config-drift", "resume even if the directory holds a different experiment"),
        "repo_cache": Key("data/repos", "flag", "--repo-cache", "code arm only"),
        "allow_clone": Key(False, "bool-flag", "--allow-clone", "code arm only"),
    },
}

TOP_LEVEL_KEYS = ("schema", "name", "strategy", "env")

SECTION_HELP = {
    "dataset": "what is searched and asked",
    "model": "the language model driving the agent",
    "agent": "the episode loop",
    "budgets": "length limits — all in TOKENS (there are no character limits anywhere)",
    "listing": "how many results a search shows",
    "retrieval": "engines and ranking",
    "evaluation": "scoring",
    "output": "where the run record goes",
}

PAPER_PRESET = {
    "agent": {"max_steps": 100},
    "output": {"runs_dir": "runs/paper"},
}


@dataclass
class Experiment:
    path: Optional[str]
    data: dict
    sha256: str

    @property
    def name(self) -> str:
        return str(self.data.get("name") or (Path(self.path).stem if self.path else "experiment"))

    @property
    def strategy(self) -> str:
        return str(self.data.get("strategy") or DEFAULT_STRATEGY)

    def get(self, section: str, key: str) -> Any:
        sec = self.data.get(section) or {}
        return sec.get(key, SCHEMA[section][key].default)


class ExperimentError(ValueError):
    pass


# --- construction ---------------------------------------------------------------------------

# --- which keys a strategy reads ------------------------------------------------------------
# A file is complete when it names every key its strategy reads. Keys absent from this map apply
# to every strategy; a key mapped to a set applies to those friendly strategy names only.
_DOC_AGENTS = {"search_visit", "search_visit_dense", "search_visit_hybrid", "search_visit_snippets", "search_visit_reranked",
               "autoread", "autoread_dense", "autoread_hybrid",
               "dci", "bounded_dci", "search_fetch", "search_fetch_dense", "search_fetch_hybrid",
               "search_fetch_bm25_plain", "search_fetch_dense_plain",
               "sieve", "sieve_bm25", "sieve_dense", "sieve_nosnip", "sieve_plain", "sieve_v2",
               "sieve_visit", "sieve_visit_fused", "sieve_visit_dense",
               "indri", "indri_plain", "indri_visit", "dedup_bm25", "dedup_dense",
               "plan_and_search", "plan_and_search_visit"}
_CODE_AGENTS = {"codefix", "codefix_grep", "codefix_patch"}
_RAG = {"rag_bm25", "rag_dense", "rag_hybrid"}          # one model call, no loop
_TEAMS = {"plan_and_search", "plan_and_search_visit"}  # procedures whose members are agents
_AGENTS = _DOC_AGENTS | _CODE_AGENTS | _TEAMS
_MODEL_USERS = _AGENTS | _RAG
_DENSE = set(DENSE_STRATEGIES) | {"sieve_visit_fused", "sieve_visit_dense", "search_fetch_dense_plain",
                                  "autoread_hybrid", "rag_dense", "rag_hybrid"}
_BM25_USERS = {"search_visit", "search_visit_snippets", "search_fetch", "search_fetch_bm25_plain", "autoread", "plan_and_search_visit",
               "autoread_hybrid", "bounded_dci", "search_visit_hybrid", "search_fetch_hybrid", "dedup_bm25",
               "bm25", "rag_bm25", "rag_hybrid"}
_BQL = {"sieve", "sieve_bm25", "sieve_dense", "sieve_nosnip", "sieve_plain", "sieve_v2", "sieve_visit", "plan_and_search",
        "sieve_visit_fused", "sieve_visit_dense", "codefix", "codefix_patch"}
_HYBRID = {"search_visit_hybrid", "search_fetch_hybrid", "autoread_hybrid", "rag_hybrid", "hybrid"}
_RERANK = {"search_visit_reranked", "reranked"}
_VISIT = {"search_visit", "search_visit_dense", "search_visit_hybrid", "search_visit_snippets", "search_visit_reranked", "plan_and_search_visit", "autoread",
          "autoread_dense", "autoread_hybrid", "sieve_visit", "sieve_visit_fused", "sieve_visit_dense",
          "indri_visit", "dedup_bm25", "dedup_dense"}
_FETCH = {"search_fetch", "search_fetch_dense", "search_fetch_hybrid", "search_fetch_bm25_plain",
          "search_fetch_dense_plain", "sieve", "sieve_bm25", "sieve_dense", "sieve_nosnip", "sieve_plain",
          "sieve_v2", "indri", "indri_plain", "plan_and_search"}
_INDRI = {"indri", "indri_plain", "indri_visit"}
APPLIES: dict[str, set] = {
    "model.name": _MODEL_USERS, "model.policy": _MODEL_USERS, "model.backend": _MODEL_USERS,
    "model.api_base": _MODEL_USERS, "model.tp": _MODEL_USERS, "model.temperature": _MODEL_USERS,
    "model.seed": _MODEL_USERS, "model.seeds": _MODEL_USERS, "model.driver": _AGENTS,
    "model.reasoning_effort": _MODEL_USERS, "model.timeout_s": _MODEL_USERS, "model.retry_attempts": _MODEL_USERS,
    "agent.max_steps": _AGENTS, "agent.prompt_profile": _AGENTS, "agent.ctx_tokens": _AGENTS | _RAG,
    "agent.ctx_window": _AGENTS, "agent.ctx_stop_frac": _AGENTS,
    "budgets.snippet_tokens": _DOC_AGENTS - {"dci", "dedup_bm25", "dedup_dense"},
    "budgets.max_visit_tokens": _VISIT,
    "budgets.max_section_tokens": _FETCH,
    "budgets.bash_max_tokens": {"dci", "bounded_dci"}, "budgets.read_max_line_tokens": {"dci", "bounded_dci"},
    "budgets.grep_line_tokens": {"codefix_grep"},
    "budgets.closer_evidence_arg_tokens": _AGENTS, "budgets.closer_evidence_obs_tokens": _AGENTS,
    "listing.bm25_visit_topk": {"search_visit", "search_visit_snippets", "plan_and_search_visit"},
    "listing.bm25_fetch_topk": {"search_fetch", "search_fetch_bm25_plain"},
    "listing.dense_visit_topk": {"search_visit_dense"},
    "listing.dense_fetch_topk": {"search_fetch_dense", "search_fetch_dense_plain"},
    "listing.hybrid_visit_topk": {"search_visit_hybrid"}, "listing.hybrid_fetch_topk": {"search_fetch_hybrid"},
    "listing.rerank_visit_topk": {"search_visit_reranked"}, "listing.rerank_fetch_topk": set(),
    "retrieval.rerank_base": _RERANK, "retrieval.rerank_method": _RERANK, "retrieval.rerank_model": _RERANK,
    "retrieval.rerank_pool": _RERANK, "retrieval.rerank_batch_size": _RERANK, "retrieval.rerank_max_length": _RERANK,
    "listing.hybrid_pool": _HYBRID, "listing.autoread_topk": {"autoread", "autoread_dense", "autoread_hybrid"},
    "listing.bm25_dci_topk": {"bounded_dci"},
    "listing.dedup_snippet_tokens": {"dedup_bm25", "dedup_dense"},
    "listing.dedup_topk": {"dedup_bm25", "dedup_dense"}, "listing.dedup_pool_k": {"dedup_bm25", "dedup_dense"},
    "retrieval.dense_model": _DENSE, "retrieval.dense_query_style": _DENSE,
    "retrieval.dense_query_instruction": _DENSE, "retrieval.dense_pooling": _DENSE, "retrieval.dense_dtype": _DENSE,
    "retrieval.dense_index": _DENSE, "retrieval.ann_ef_search": _DENSE, "retrieval.bm25_index": _BM25_USERS,
    "retrieval.bql_soft_fallback": _BQL, "retrieval.bql_soft_pool": _BQL, "retrieval.bql_date_range": _BQL,
    "retrieval.bql_dense": {"sieve_bm25", "sieve_plain", "sieve_v2", "sieve_visit"},
    "retrieval.bql_dense_rrf_k": {"sieve", "sieve_nosnip", "sieve_bm25", "sieve_visit_fused"},
    "retrieval.rrf_k": _HYBRID, "retrieval.hybrid_retrievers": _HYBRID, "retrieval.hybrid_fusion": _HYBRID,
    "retrieval.hybrid_weights": _HYBRID,
    "retrieval.indri_dense": _INDRI, "retrieval.indri_dense_w": _INDRI, "retrieval.indri_dense_expand_k": _INDRI,
    "retrieval.lucene_mu": _INDRI,
    "retrieval.ann": _DENSE, "retrieval.ann_min": _DENSE, "retrieval.ann_pq_min": _DENSE, "retrieval.dense_device": _DENSE,
    "output.repo_cache": _CODE_AGENTS, "output.allow_clone": _CODE_AGENTS,
}


def _friendly(strategy: str) -> Optional[str]:
    """A friendly strategy name for `strategy` (friendly or registered), or None if unknown."""
    if strategy in STRATEGIES:
        return strategy
    for k, v in STRATEGIES.items():
        if v == strategy:
            return k
    from agent_search.strategies.conditions import CONDITIONS
    name = strategy[len("agent_"):] if strategy.startswith("agent_") else strategy
    return name if name in CONDITIONS else None


def applies(section: str, key: str, strategy: str) -> bool:
    """True when `strategy` reads `section.key`. Unknown (plugin) strategies read every key."""
    who = APPLIES.get(f"{section}.{key}")
    if who is None:
        return True
    name = _friendly(strategy)
    return True if name is None else name in who


def relevant_keys(strategy: str) -> dict[str, list[str]]:
    return {section: [k for k in keys if applies(section, k, strategy)]
            for section, keys in SCHEMA.items()}


def unused_keys(data: dict) -> list[str]:
    """Keys present in `data` that its strategy does not read."""
    strategy = str(data.get("strategy") or DEFAULT_STRATEGY)
    out = []
    for section, keys in SCHEMA.items():
        for k in (data.get(section) or {}):
            if k in keys and not applies(section, k, strategy):
                out.append(f"{section}.{k}")
    return out


def defaults(preset: Optional[str] = None, strategy: Optional[str] = None) -> dict:
    """A complete experiment dict with every key at its default (or the named preset's)."""
    data: dict = {"schema": SCHEMA_VERSION, "name": None, "strategy": strategy or DEFAULT_STRATEGY}
    for section, keys in SCHEMA.items():
        data[section] = {k: spec.default for k, spec in keys.items()}
    data["env"] = {}
    if preset == "paper":
        for section, over in PAPER_PRESET.items():
            data[section].update(over)
        data["strategy"] = "sieve"
        data["dataset"]["name"] = "hotpotqa_structured"
    elif preset not in (None, "library"):
        raise ExperimentError(f"unknown preset {preset!r} (choose: library, paper)")
    if strategy:
        data["strategy"] = strategy
    return data


def validate(data: Any, *, complete: bool = False) -> dict:
    """Check shape, types and names; return the data. `complete=True` additionally requires
    every schema key to be present (the contract for shipped files)."""
    if not isinstance(data, dict):
        raise ExperimentError("an experiment file must be a YAML mapping")
    unknown = [k for k in data if k not in TOP_LEVEL_KEYS and k not in SCHEMA]
    if unknown:
        raise ExperimentError(f"unknown top-level keys {unknown}; allowed: "
                              f"{list(TOP_LEVEL_KEYS) + list(SCHEMA)}")
    if data.get("schema", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ExperimentError(f"schema {data.get('schema')!r} is not supported (want {SCHEMA_VERSION})")
    missing: list[str] = []
    strategy = str(data.get("strategy") or DEFAULT_STRATEGY)
    for section, keys in SCHEMA.items():
        sec = data.get(section)
        needed = [k for k in keys if applies(section, k, strategy)]
        if sec is None:
            if complete and needed:
                missing.append(section)
            continue
        if not isinstance(sec, dict):
            raise ExperimentError(f"section {section!r} must be a mapping")
        bad = [k for k in sec if k not in keys]
        if bad:
            raise ExperimentError(f"unknown keys {bad} in section {section!r}; allowed: {list(keys)}")
        if complete:
            missing += [f"{section}.{k}" for k in needed if k not in sec]
        for k, v in sec.items():
            _check_type(section, k, v, keys[k].default)
    if complete and missing:
        raise ExperimentError(f"experiment file is not complete — missing: {missing}")
    env = data.get("env")
    if env is not None and (not isinstance(env, dict) or not all(isinstance(k, str) for k in env)):
        raise ExperimentError("`env` must be a mapping of environment variable names to values")
    if "strategy" in data and not isinstance(data["strategy"], str):
        raise ExperimentError("`strategy` must be a string")
    return data


def _check_type(section: str, key: str, value: Any, default: Any) -> None:
    if value is None or default is None:
        return
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, list):
        ok = isinstance(value, list)
    else:
        ok = isinstance(value, str)
    if not ok:
        raise ExperimentError(f"{section}.{key}: expected {type(default).__name__}, got {value!r}")


def load(path: str | os.PathLike, *, complete: bool = False) -> Experiment:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    validate(data, complete=complete)
    return Experiment(path=str(path), data=data,
                      sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])


def from_dict(data: dict, *, complete: bool = False) -> Experiment:
    validate(data, complete=complete)
    blob = yaml.safe_dump(data, sort_keys=True).encode("utf-8")
    return Experiment(path=None, data=data, sha256=hashlib.sha256(blob).hexdigest()[:16])


# --- overrides ------------------------------------------------------------------------------

def _key_index() -> dict[str, tuple[str, str]]:
    idx: dict[str, list[tuple[str, str]]] = {}
    for section, keys in SCHEMA.items():
        for k in keys:
            idx.setdefault(k, []).append((section, k))
    return {k: v[0] for k, v in idx.items() if len(v) == 1}


def _coerce(value: str, default: Any) -> Any:
    if value.lower() in ("null", "none", "~"):
        return None
    if isinstance(default, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ExperimentError(f"expected true/false, got {value!r}")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        return yaml.safe_load(value)
    if default is None:
        # no type to follow: numbers and booleans become numbers and booleans, anything else
        # stays the string it was typed as (a model id, a path)
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            return value
        return parsed if isinstance(parsed, (int, float, bool, list)) else value
    return value


def apply_overrides(exp: Experiment, overrides: dict[str, str]) -> Experiment:
    """`section.key=value` (or a bare key when it is unique across sections) overrides."""
    data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in exp.data.items()}
    unique = _key_index()
    for raw_key, value in overrides.items():
        if raw_key in ("strategy", "name"):
            data[raw_key] = value
            continue
        if raw_key.startswith("env."):
            data.setdefault("env", {})[raw_key[4:]] = value
            continue
        if "." in raw_key:
            section, key = raw_key.split(".", 1)
        elif raw_key in unique:
            section, key = unique[raw_key]
        else:
            raise ExperimentError(f"unknown override {raw_key!r}; use section.key (sections: {list(SCHEMA)})")
        if section not in SCHEMA or key not in SCHEMA[section]:
            raise ExperimentError(f"unknown override {raw_key!r}")
        data.setdefault(section, {})[key] = _coerce(str(value), SCHEMA[section][key].default)
    return from_dict(data)


# --- translation to one execution path ------------------------------------------------------

def to_invocation(exp: Experiment) -> tuple[list[str], dict[str, str]]:
    """The experiment as (run_eval argv, environment knobs): the same shapes the key=value
    launcher produces, so a file and a command line run through one code path."""
    retriever = resolve_strategy(exp.strategy)
    args = ["--dataset", str(exp.get("dataset", "name")), "--retriever", retriever]
    env: dict[str, str] = {}

    model = exp.get("model", "name")
    policy = exp.get("model", "policy")
    if model:
        args += ["--model", str(model)]
    if policy is None:
        policy = "llm" if model else ("stub" if retriever.startswith("agent") else None)
    if policy:
        args += ["--policy", str(policy)]
    seeds = exp.get("model", "seeds")

    for section, keys in SCHEMA.items():
        for key, spec in keys.items():
            if spec.kind == "special" or (section == "dataset" and key == "name"):
                continue
            value = exp.get(section, key)
            if spec.kind == "flag":
                if key == "seed" and seeds:
                    continue
                if value is not None:
                    args += [spec.target, str(value)]
            elif spec.kind == "bool-flag":
                if value:
                    args.append(spec.target)
            elif spec.kind == "env":
                if value is None:
                    continue
                env[spec.target] = ("1" if value else "0") if isinstance(value, bool) else str(value)
    if seeds:
        args += ["--seeds", ",".join(str(s) for s in seeds)]
    ks = exp.get("evaluation", "k")
    if ks:
        args += ["--k"] + [str(k) for k in ks]
    for k, v in (exp.data.get("env") or {}).items():
        env[str(k)] = str(v)
    if exp.path:
        args += ["--experiment-file", str(exp.path)]
    return args, env


def requirements(exp: Experiment) -> list[str]:
    """Human-readable prerequisites implied by the setting (printed before a run)."""
    notes = []
    if exp.strategy in DENSE_STRATEGIES:
        ext = exp.get("retrieval", "dense_index")
        if ext:
            notes.append(f"the prebuilt vector index at {ext} (retrieval.dense_index) and faiss")
        else:
            notes.append("a persisted dense embedding cache for this dataset and retrieval.dense_model "
                         "(skimsearchagent-build-indexes --retriever dense)")
    if exp.get("retrieval", "bm25_index"):
        notes.append(f"the prebuilt Lucene index at {exp.get('retrieval', 'bm25_index')} (retrieval.bm25_index)")
    strategy = _friendly(exp.strategy) or exp.strategy
    if strategy in (_BQL | _INDRI) and strategy not in _CODE_AGENTS:
        notes.append("the Lucene structured index (skimsearchagent-build-indexes --retriever "
                     "search_lucene --dataset <name>) and Java 21+")
    if strategy in _BM25_USERS:
        notes.append("Pyserini and Java 21+ (pip install -e '.[retrieval]')")
    m = exp.get("model", "name")
    if m and str(m).startswith("gpt-"):
        notes.append("OPENAI_API_KEY")
    if m and str(m).startswith("gemini"):
        notes.append("GEMINI_API_KEY")
    return notes


# --- rendering ------------------------------------------------------------------------------

def _yaml_scalar(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().rstrip(".").rstrip("\n") \
        if isinstance(value, list) else yaml.safe_dump(value).split("\n")[0].replace("...", "").strip()


def render(data: dict, *, comments: bool = True) -> str:
    """Render a complete experiment dict as commented YAML (stable key order)."""
    out = [f"# SkimSearchAgent experiment file (schema {SCHEMA_VERSION}). ONE file = ONE complete setting.",
           "# Run:      skimsearchagent run <this file> [section.key=value ...]",
           "# Validate: skimsearchagent validate <this file>",
           "# Every key that changes what runs or how it is measured is listed; unknown keys are errors.",
           f"schema: {SCHEMA_VERSION}",
           f"name: {_yaml_scalar(data.get('name'))}",
           f"strategy: {_yaml_scalar(data.get('strategy', DEFAULT_STRATEGY))}"
           + ("   # friendly name (skimsearchagent --help) or agent_<condition>" if comments else ""),
           ""]
    strategy = str(data.get("strategy") or DEFAULT_STRATEGY)
    for section, keys in SCHEMA.items():
        shown = [k for k in keys if applies(section, k, strategy)]
        if not shown:
            continue
        if comments:
            out.append(f"# --- {section}: {SECTION_HELP[section]}")
        out.append(f"{section}:")
        sec = data.get(section) or {}
        for key in shown:
            spec = keys[key]
            value = sec.get(key, spec.default)
            line = f"  {key}: {_yaml_scalar(value)}"
            if comments and spec.help:
                line = f"{line:<40} # {spec.help}"
            out.append(line)
        out.append("")
    env = data.get("env") or {}
    if comments:
        out.append("# --- env: any other environment knob, verbatim (escape hatch; recorded in config.json)")
    out.append("env: {}" if not env else "env:")
    for k, v in env.items():
        out.append(f"  {k}: {_yaml_scalar(v)}")
    out.append("")
    return "\n".join(out)


def template(preset: Optional[str] = None, strategy: Optional[str] = None) -> str:
    """A complete file for `strategy` (the default strategy when omitted, or the preset's own):
    only the keys that strategy reads are listed, so the file shows exactly what the setting
    depends on."""
    return render(defaults(preset, strategy))


__all__ = ["SCHEMA", "SCHEMA_VERSION", "Experiment", "ExperimentError", "defaults", "validate", "applies",
           "relevant_keys", "unused_keys", "APPLIES",
           "load", "from_dict", "apply_overrides", "to_invocation", "requirements", "render",
           "template", "PAPER_PRESET"]
