"""Prompt variants for the ITER tools, for the prompt ablation.

`research_dedup` is DIVER's Tongyi prompt (the ITER protocol), `research_dedup_strong` DIVER's
strong prompt, `research` the library's default and `research_paper` the Sieve paper's. The Sieve
strategies under the default prompt are the friendly-name conditions in `defaults.py`."""
from agent_search.strategies.conditions import condition

condition("research_dedup_dense_combined", "research", "dedup_dense")        # ITER's tools, the default prompt
condition("research_dedup_dense_paper", "research_paper", "dedup_dense")     # ITER's tools, the paper's prompt
