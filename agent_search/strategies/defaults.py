"""The friendly strategy names as conditions under the library's default prompt (`research`).

`strategy=sieve_bm25` runs the condition `sieve_bm25` = task `research` x strategy `sieve_bm25`,
registered as the retriever `agent_sieve_bm25`. The paper's own conditions (`research_snip` and the
other paper-era names in `paper.py`) keep the paper's prompt (`research_paper`) so the reproduction
stays byte for byte; the ITER, team and code conditions carry their own tasks and are named in
`names.py` directly."""
from agent_search.strategies.conditions import condition

DEFAULT_RESEARCH = ("search_visit", "search_visit_dense", "search_visit_hybrid", "autoread", "autoread_dense",
                    "dci", "bounded_dci", "search_fetch", "search_fetch_dense", "search_fetch_hybrid",
                    "sieve", "sieve_bm25", "sieve_dense", "sieve_nosnip", "indri")

for _name in DEFAULT_RESEARCH:
    condition(_name, "research", _name)

__all__ = ["DEFAULT_RESEARCH"]
