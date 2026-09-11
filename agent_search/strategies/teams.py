"""Multi-agent strategies: a harness whose members are conditions run as agents.

`plan_and_search` is the first team: a planner splits the question, one Sieve agent per
sub-question, a synthesizer answers (`agent_search.harness.plan_and_search`). The member is
a strategy name; a variant with another member is one more line here.
"""
from agent_search.harness.plan_and_search import PlanAndSearch
from agent_search.strategies.base import Strategy, register_strategy

plan_and_search = register_strategy(Strategy(
    name="plan_and_search",
    description="a planner splits the question, one Sieve (BM25) agent per sub-question, a synthesizer answers",
    harness=PlanAndSearch(searcher="sieve_bm25")))

plan_and_search_visit = register_strategy(Strategy(
    name="plan_and_search_visit",
    description="a planner splits the question, one search-visit (BM25) agent per sub-question, a synthesizer answers",
    harness=PlanAndSearch(searcher="search_visit")))


__all__ = ["plan_and_search", "plan_and_search_visit"]
