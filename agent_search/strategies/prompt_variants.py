"""Prompt variants: the same strategy under a different task prompt, for prompt ablations.

`research` is the paper's prompt, `research_dedup` DIVER's Tongyi prompt, `research_dedup_strong`
DIVER's strong prompt, `research_combined` the three combined (agent_search/tasks/). A condition
here changes only the prompt of a paper condition, so a run under it differs from the paper run
in nothing else."""
from agent_search.strategies.conditions import condition

condition("research_bm25_combined", "research_combined", "search_visit")            # search_visit, combined prompt
condition("research_snip_combined", "research_combined", "sieve_bm25")              # Sieve BM25, combined prompt
condition("research_bql_dense_snip_combined", "research_combined", "sieve")         # fused Sieve, combined prompt
condition("research_dedup_dense_combined", "research_combined", "dedup_dense")      # ITER's tools, combined prompt
condition("research_dedup_dense_paper", "research", "dedup_dense")                  # ITER's tools, the paper's prompt
