"""Friendly strategy names on the command line and in experiment files.

A friendly name (`sieve_bm25`, `search_visit`, `dci`, ...) maps to the registered retriever that
runs it: `agent_<condition>` for an agent, the retriever itself for a floor. Every condition in
`agent_search.strategies.conditions` is also selectable by its own name, and any registered
retriever by its registered name. `DENSE_STRATEGIES` lists the names that need a dense embedding
cache (the launcher builds it first). `DEFAULT_STRATEGY` is what runs when none is named.
"""
from __future__ import annotations

STRATEGIES: dict[str, str] = {
    # conventional baselines
    "search_visit": "agent_research_bm25",
    "search_visit_dense": "agent_research_dense",
    "search_visit_hybrid": "agent_research_hybrid",
    "autoread": "agent_research_bm25_autoread",
    "autoread_dense": "agent_research_dense_autoread",
    "dci": "agent_research_dci",
    "bounded_dci": "agent_research_bm25_dci",
    "search_fetch": "agent_research_bm25_fetch_snip",
    "search_fetch_dense": "agent_research_dense_fetch",
    "search_fetch_hybrid": "agent_research_hybrid_fetch_snip",
    # Sieve family (the paper's method)
    "sieve": "agent_research_bql_dense_snip",
    "sieve_bm25": "agent_research_snip",
    "sieve_dense": "agent_research_bql_donly_snip",
    "sieve_nosnip": "agent_research_bql_dense_fetch",
    # structured-retrieval control
    "indri": "agent_research_indri_snip",
    # ITER's strategy: de-duplicated search + get_document (BM25 or the run's dense model)
    "dedup_bm25": "agent_research_dedup_bm25",
    "dedup_dense": "agent_research_dedup_dense",
    # code-localization arm (SWE-bench-style repositories; dataset=code_fixture, swebench_*)
    "codefix": "agent_codefix",
    "codefix_grep": "agent_codefix_grep",
    "codefix_patch": "agent_codefix_patch",
    # retrieval-only floors (no agent loop)
    "bm25": "bm25_local",
    "bm25_lucene": "bm25_pyserini",
}

#: Strategies that need a pre-built dense embedding cache (``skimsearchagent-build-indexes
#: --retriever dense``) before they can run.
DENSE_STRATEGIES = frozenset({
    "search_visit_dense", "search_visit_hybrid", "autoread_dense", "autoread_hybrid", "search_fetch_dense",
    "search_fetch_dense_plain", "search_fetch_hybrid", "sieve", "sieve_dense", "sieve_nosnip",
    "sieve_visit_fused", "sieve_visit_dense", "dedup_dense", "rag_dense", "rag_hybrid",
})

DEFAULT_STRATEGY = "sieve_bm25"


def resolve_strategy(name: str) -> str:
    """Friendly name, condition name or raw retriever name -> the registered retriever name."""
    if name in STRATEGIES:
        return STRATEGIES[name]
    from agent_search.strategies.conditions import CONDITIONS
    cond = CONDITIONS.get(name)
    if cond is not None:
        if not cond.strategy.loop and cond.strategy.retriever:
            return cond.strategy.retriever
        return f"agent_{name}"
    return name


def condition_of(retriever_name: str) -> str | None:
    """``agent_<condition>`` -> ``condition``; ``None`` for retrieval-only names."""
    if retriever_name.startswith("agent_"):
        return retriever_name[len("agent_"):]
    return None


__all__ = ["STRATEGIES", "DENSE_STRATEGIES", "DEFAULT_STRATEGY", "resolve_strategy", "condition_of"]
