"""Friendly strategy names on the command line and in experiment files.

A friendly name (`sieve_bm25`, `search_visit`, `dci`, ...) maps to the registered retriever that
runs it: `agent_<condition>` for an agent, the retriever itself for a floor. The document
strategies run under the library's default prompt (task `research`, `strategies/defaults.py`);
the paper's conditions (`research_snip`, `research_bm25`, ... in `strategies/paper.py`) keep the
paper's prompt and are selectable by their own names. Every condition in
`agent_search.strategies.conditions` is also selectable by its own name, and any registered
retriever by its registered name. `DENSE_STRATEGIES` lists the names that need a dense embedding
cache (the launcher builds it first). `DEFAULT_STRATEGY` is what runs when none is named.
"""
from __future__ import annotations

STRATEGIES: dict[str, str] = {
    # conventional baselines
    "search_visit": "agent_search_visit",
    "search_visit_dense": "agent_search_visit_dense",
    "search_visit_hybrid": "agent_search_visit_hybrid",
    "search_visit_reranked": "agent_search_visit_reranked",
    "autoread": "agent_autoread",
    "autoread_dense": "agent_autoread_dense",
    "dci": "agent_dci",
    "bounded_dci": "agent_bounded_dci",
    "search_fetch": "agent_search_fetch",
    "search_fetch_dense": "agent_search_fetch_dense",
    "search_fetch_hybrid": "agent_search_fetch_hybrid",
    # Sieve family (the paper's method)
    "sieve": "agent_sieve",
    "sieve_bm25": "agent_sieve_bm25",
    "sieve_dense": "agent_sieve_dense",
    "sieve_nosnip": "agent_sieve_nosnip",
    # structured-retrieval control
    "indri": "agent_indri",
    # multi-agent teams (agent_search/procedures)
    "plan_and_search": "agent_plan_and_search",
    "plan_and_search_visit": "agent_plan_and_search_visit",
    # ITER's strategy: de-duplicated search + get_document (BM25 or the run's dense model)
    "dedup_bm25": "agent_research_dedup_bm25",
    "dedup_dense": "agent_research_dedup_dense",
    # the same tools without the de-duplication: ITER's own evaluation setting (Table 1)
    "iter_bm25": "agent_research_iter_bm25",
    "iter_dense": "agent_research_iter_dense",
    "iter_dense_qwen": "agent_research_iter_dense_qwen",
    "iter_dense_strong": "agent_research_iter_dense_strong",
    # code-localization arm (SWE-bench-style repositories; dataset=code_fixture, swebench_*)
    "codefix": "agent_codefix",
    "codefix_grep": "agent_codefix_grep",
    "codefix_patch": "agent_codefix_patch",
    # retrieval-only floors (no agent loop)
    "bm25": "bm25_pyserini",
    "hybrid": "hybrid",
    "reranked": "reranked",
}

#: Strategies that need a pre-built dense embedding cache (``skimsearchagent-build-indexes
#: --retriever dense``) before they can run.
DENSE_STRATEGIES = frozenset({
    "search_visit_dense", "search_visit_hybrid", "autoread_dense", "autoread_hybrid", "search_fetch_dense",
    "search_fetch_dense_plain", "search_fetch_hybrid", "sieve", "sieve_dense", "sieve_nosnip",
    "sieve_visit_fused", "sieve_visit_dense", "dedup_dense", "iter_dense", "iter_dense_qwen", "iter_dense_strong", "rag_dense", "rag_hybrid", "hybrid", "dense",
})

DEFAULT_STRATEGY = "sieve_bm25"


def resolve_strategy(name: str) -> str:
    """Friendly name, condition name or raw retriever name -> the registered retriever name."""
    if name in STRATEGIES:
        return STRATEGIES[name]
    from agent_search.strategies.conditions import CONDITIONS
    cond = CONDITIONS.get(name)
    if cond is not None:
        if cond.strategy.retriever:
            return cond.strategy.retriever
        return f"agent_{name}"
    return name


def condition_of(retriever_name: str) -> str | None:
    """``agent_<condition>`` -> ``condition``; ``None`` for retrieval-only names."""
    if retriever_name.startswith("agent_"):
        return retriever_name[len("agent_"):]
    return None


__all__ = ["STRATEGIES", "DENSE_STRATEGIES", "DEFAULT_STRATEGY", "resolve_strategy", "condition_of"]
