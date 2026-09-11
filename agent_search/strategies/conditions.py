"""Conditions: a task with a strategy. What a run names.

`condition(task, strategy)` pairs a registered task with a registered strategy (the strategy's
domain, if any, must match the task's). The paper's condition names stay valid through
`ALIASES`, one small table from the old names to (task, strategy). A condition renders its
system prompt from the task template and the strategy's tools, and the render is pinned by
`tests/test_prompt_fidelity.py` for every paper condition.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from agent_search.strategies.base import STRATEGIES, Strategy
from agent_search.tasks.base import TASKS, Task

CONDITIONS: dict[str, "Condition"] = {}

# what the bare retriever name `agent` runs (an environment knob so no condition is hardwired)
AGENT_DEFAULT_CONDITION = os.environ.get("AGENT_DEFAULT_CONDITION", "research_snip")

# the paper's condition names -> (task, strategy)
ALIASES: dict[str, tuple[str, str]] = {}


@dataclass(frozen=True)
class Condition:
    name: str
    task: Task
    strategy: Strategy

    @property
    def domain(self) -> str:
        return self.task.domain

    @property
    def tool_names(self) -> tuple:
        return self.strategy.tool_names

    def render(self, profile: Optional[str] = None) -> str:
        return self.task.render(self.strategy.tools, profile=profile, toolset_name=self.strategy.toolset_name)

    def system_sha256(self, profile: Optional[str] = None) -> str:
        return hashlib.sha256(self.render(profile).encode("utf-8")).hexdigest()[:16]


def condition(name: str, task: str, strategy: str) -> Condition:
    """Register the condition `name` = `task` x `strategy`."""
    t = TASKS[task] if isinstance(task, str) else task
    s = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    if s.domain is not None and s.domain != t.domain:
        raise ValueError(f"strategy {s.name!r} is for the {s.domain} domain; task {t.name!r} is {t.domain}")
    c = Condition(name=name, task=t, strategy=s)
    CONDITIONS[name] = c
    _register_as_retriever(c)
    return c


def _register_as_retriever(c: Condition) -> None:
    """Every condition run through a harness is the retriever `agent_<name>`; a floor is the
    plain retriever its strategy names, so it is not registered twice."""
    if c.strategy.retriever:
        return
    from agent_search.retrievers.registry import _REGISTRY, register

    def _build(cfg, rname):
        from agent_search.evaluation.agent_runner import build_condition_agent
        return build_condition_agent(cfg, rname[len("agent_"):])

    rname = f"agent_{c.name}"
    if rname not in _REGISTRY:
        register(rname)(_build)


def alias(old_name: str, task: str, strategy: str) -> Condition:
    """A paper-era condition name for `task` x `strategy`."""
    ALIASES[old_name] = (task, strategy)
    return condition(old_name, task, strategy)


def get_condition(name: str) -> Condition:
    """A condition by its name, or by its retriever name (`agent_<name>`, or the bare `agent`)."""
    if name == "agent":
        name = AGENT_DEFAULT_CONDITION
    elif name.startswith("agent_") and name not in CONDITIONS:
        name = name[len("agent_"):]
    if name in CONDITIONS:
        return CONDITIONS[name]
    raise ValueError(f"unknown condition {name!r}; choose from {sorted(CONDITIONS)}")


__all__ = ["Condition", "CONDITIONS", "ALIASES", "AGENT_DEFAULT_CONDITION", "condition", "alias", "get_condition"]
