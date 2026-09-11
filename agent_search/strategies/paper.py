"""The paper's condition names, each a task with a strategy. Kept so configs, run records and
`--retriever agent_<name>` keep working; the registry grows here as strategies are ported."""
from agent_search.strategies import (  # noqa: F401  (registrations)
    autoread, codefix, dci, dedup, indri, rag, retrieval_only, search_fetch, search_visit, sieve, teams)
from agent_search.strategies.conditions import alias
from agent_search.tasks import TASKS  # noqa: F401  (registrations)

alias("research_bm25", "research", "search_visit")
alias("research_dense", "research", "search_visit_dense")
alias("research_hybrid", "research", "search_visit_hybrid")
alias("research_bm25_autoread", "research", "autoread")
alias("research_dense_autoread", "research", "autoread_dense")
alias("research_snip", "research", "sieve_bm25")
alias("research_bql_dense_snip", "research", "sieve")
alias("research_bql_donly_snip", "research", "sieve_dense")
alias("research_bql_dense_fetch", "research", "sieve_nosnip")
alias("research_dci", "research", "dci")
alias("research_bm25_dci", "research", "bounded_dci")
alias("research_dedup_bm25", "research_dedup", "dedup_bm25")
alias("research_dedup_dense", "research_dedup", "dedup_dense")
alias("codefix", "codefix", "codefix")
alias("codefix_grep", "codefix", "codefix_grep")
alias("codefix_patch", "codefix_patch", "codefix")

# loop-free baselines: the floors (no model) and one-shot RAG (one model call)
alias("bm25", "research", "bm25")
alias("dense", "research", "dense")
alias("rag_bm25", "research", "rag")
alias("rag_dense", "research", "rag_dense")
alias("rag_hybrid", "research", "rag_hybrid")

alias("research_bm25_fetch_snip", "research", "search_fetch")
alias("research_dense_fetch", "research", "search_fetch_dense")
alias("research_hybrid_fetch_snip", "research", "search_fetch_hybrid")
alias("research_indri_snip", "research", "indri")

# every other document strategy runs under its own name with the research task, so a strategy
# never exists without a way to run it
from agent_search.strategies.base import STRATEGIES as _ALL  # noqa: E402
from agent_search.strategies.conditions import CONDITIONS as _CONDS, condition as _condition  # noqa: E402

for _name, _s in list(_ALL.items()):
    if _name not in _CONDS and _s.domain in (None, "general"):
        _condition(_name, "research", _name)
