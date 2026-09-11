"""Procedures: strategies that are programs rather than tool loops, one file each.
`rag.py` (one-shot RAG: rank, one prompt, one model call) and `plan_and_search.py` (a team: a
planner, one agent per sub-question, a synthesizer). `base.py` holds the contract."""
from agent_search.procedures.base import Procedure, ProcedureContext, ProcedureResult
from agent_search.procedures.plan_and_search import PlanAndSearch
from agent_search.procedures.rag import OneShotRag

__all__ = ["Procedure", "ProcedureContext", "ProcedureResult", "OneShotRag", "PlanAndSearch"]
