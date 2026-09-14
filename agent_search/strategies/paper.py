"""The paper's condition names, each a task with a strategy. Kept so configs, run records and
`--retriever agent_<name>` keep working; the registry grows here as strategies are ported."""
from agent_search.strategies import (  # noqa: F401  (registrations)
    autoread, codefix, dci, dedup, indri, rag, retrieval_only, search_fetch, search_visit, sieve, teams)
from agent_search.strategies.conditions import alias, condition
from agent_search.tasks import TASKS  # noqa: F401  (registrations)

alias("research_bm25", "research_paper", "search_visit")
alias("research_dense", "research_paper", "search_visit_dense")
alias("research_hybrid", "research_paper", "search_visit_hybrid")
alias("research_bm25_autoread", "research_paper", "autoread")
alias("research_dense_autoread", "research_paper", "autoread_dense")
alias("research_snip", "research_paper", "sieve_bm25")
alias("research_bql_dense_snip", "research_paper", "sieve")
alias("research_bql_donly_snip", "research_paper", "sieve_dense")
alias("research_bql_dense_fetch", "research_paper", "sieve_nosnip")
alias("research_dci", "research_paper", "dci")
alias("research_bm25_dci", "research_paper", "bounded_dci")
alias("research_dedup_bm25", "research_dedup", "dedup_bm25")
alias("research_dedup_dense", "research_dedup", "dedup_dense")
condition("research_dedup_bm25_strong", "research_dedup_strong", "dedup_bm25")     # DIVER --strong, general backbones
condition("research_dedup_dense_strong", "research_dedup_strong", "dedup_dense")
condition("research_iter_dense", "research_tongyi", "iter_dense")            # the ITER paper's evaluation setting (Table 1): no dedup
condition("research_iter_bm25", "research_tongyi", "iter_bm25")
condition("research_dedup_bm25_qwen", "research_dedup_qwen", "dedup_bm25")        # DIVER on Qwen3.5 / WebExplorer: no answer tags
condition("research_dedup_dense_qwen", "research_dedup_qwen", "dedup_dense")
alias("codefix", "codefix", "codefix")
alias("codefix_grep", "codefix", "codefix_grep")
alias("codefix_patch", "codefix_patch", "codefix")

# loop-free baselines: the floors (no model) and one-shot RAG (one model call)
alias("bm25", "research_paper", "bm25")
alias("dense", "research_paper", "dense")
alias("rag_bm25", "research_paper", "rag")
alias("rag_dense", "research_paper", "rag_dense")
alias("rag_hybrid", "research", "rag_hybrid")

alias("research_bm25_fetch_snip", "research_paper", "search_fetch")
alias("research_dense_fetch", "research_paper", "search_fetch_dense")
alias("research_hybrid_fetch_snip", "research_paper", "search_fetch_hybrid")
alias("research_indri_snip", "research_paper", "indri")

# every other document strategy runs under its own name with the research task, so a strategy
# never exists without a way to run it
from agent_search.strategies.base import STRATEGIES as _ALL  # noqa: E402
from agent_search.strategies.conditions import CONDITIONS as _CONDS, condition as _condition  # noqa: E402

for _name, _s in list(_ALL.items()):
    if _name not in _CONDS and _s.domain in (None, "general"):
        _condition(_name, "research", _name)

# Sieve with the listing filled to k, under the paper's prompt (a like-for-like variant of the three Sieve cells)
condition("research_bql_dense_snip_fill", "research_paper", "sieve_fill")
condition("research_snip_fill", "research_paper", "sieve_bm25_fill")
condition("research_bql_donly_snip_fill", "research_paper", "sieve_dense_fill")

# the headline Sieve under the paper's stale BrowseComp manual (a reproduction of the paper's prompt)
condition("research_bql_dense_snip_papermanual", "research_paper", "sieve_paper_manual")
