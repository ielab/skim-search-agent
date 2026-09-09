"""Friendly strategy names -> registered agent conditions.

A *strategy* is a complete research agent: a prompt condition (task template x toolset,
``agent_search/prompts/conditions.yaml``) plus the retrieval engines its tools need. Every
strategy is selectable by its friendly name here, by its registered retriever name
(``agent_<condition>``), or — for retrieval-only floors — by the retriever name itself.

Add a strategy: add a condition to ``conditions.yaml`` (and a toolset to ``tools.yaml`` if it
needs a new tool surface); it is registered automatically as ``agent_<name>``. Add a friendly
alias here only if you want a short name on the command line.
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
    "search_visit_dense", "search_visit_hybrid", "autoread_dense", "search_fetch_dense",
    "search_fetch_hybrid", "sieve", "sieve_dense", "sieve_nosnip", "dedup_dense",
})

DEFAULT_STRATEGY = "sieve_bm25"


def resolve_strategy(name: str) -> str:
    """Friendly name or raw retriever name -> the registered retriever name."""
    return STRATEGIES.get(name, name)


def condition_of(retriever_name: str) -> str | None:
    """``agent_<condition>`` -> ``condition``; ``None`` for retrieval-only names."""
    if retriever_name.startswith("agent_"):
        return retriever_name[len("agent_"):]
    return None


__all__ = ["STRATEGIES", "DENSE_STRATEGIES", "DEFAULT_STRATEGY", "resolve_strategy", "condition_of"]
