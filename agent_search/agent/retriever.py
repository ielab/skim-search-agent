"""Compatibility module: `agent_search.agent.retriever` moved to `agent_search.legacy.retriever`
(the pre-0.3 `AgentRetriever`). Conditions run through `agent_search.evaluation.agent_runner`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("agent_search.legacy.retriever")
