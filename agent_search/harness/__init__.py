"""Harnesses: how a condition is run on a question, one file each. `react.py` (the default:
the model picks each step from the strategy's tools), `rag.py` (rank once, one model call),
`plan_and_search.py` (a team: a planner, one member agent per sub-question, a synthesizer).
`base.py` holds the contract. A strategy names its harness with `harness=`."""
from agent_search.harness.base import Harness, HarnessContext, HarnessResult, trajectory_from_steps
from agent_search.harness.plan_and_search import PlanAndSearch
from agent_search.harness.rag import OneShotRag
from agent_search.harness.react import ReAct

__all__ = ["Harness", "HarnessContext", "HarnessResult", "trajectory_from_steps", "ReAct", "OneShotRag", "PlanAndSearch"]
