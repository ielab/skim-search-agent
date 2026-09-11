"""The strategy contract: a named combination of tools with their options, and the harness
that runs them.

A strategy file lists the tools it takes (instances carrying their options and the name the
model sees), which engines they need (the union of the tools' declarations), and its harness
(`agent_search.harness`: ReAct by default, the model picking each step; one-shot RAG; a team
whose members are conditions). A floor names a registered retriever instead (`retriever=`):
rank once, no model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from agent_search.harness.base import Harness
from agent_search.harness.react import ReAct
from agent_search.tools.base import EpisodeState, Tool, ToolBox

STRATEGIES: dict[str, "Strategy"] = {}


@dataclass
class Strategy:
    name: str
    description: str
    tools: Sequence[Tool] = ()
    toolset_name: Optional[str] = None     # the paper's toolset name (rendered by {{toolset}})
    domain: Optional[str] = None           # restrict to code / general; None = any task
    harness: Harness = field(default_factory=ReAct)   # how the model is put to work (agent_search.harness)
    sdk: bool = True                       # may run through the Agents-SDK driver
    extra_engines: tuple = ()              # engines a harness needs beyond its tools
    retriever: Optional[str] = None        # a floor: the registered retriever that IS the strategy

    @property
    def engines(self) -> tuple:
        out: list = []
        for t in self.tools:
            for e in t.engines:
                if e not in out:
                    out.append(e)
        for e in self.extra_engines:
            if e not in out:
                out.append(e)
        for e in self.harness.all_engines():
            if e not in out:
                out.append(e)
        return tuple(out)

    @property
    def needs_files(self) -> bool:
        return any(t.needs_files for t in self.tools)

    @property
    def tool_names(self) -> tuple:
        return tuple(t.name for t in self.tools)

    def toolbox(self, state: EpisodeState, units, ubyid, engines, files=None, corpus_key=None) -> ToolBox:
        """Fresh tool instances bound for one episode."""
        bound = [t.fresh().bind(state, units, ubyid, engines, files=files, corpus_key=corpus_key)
                 for t in self.tools]
        return ToolBox(bound, state)


def register_strategy(strategy: Strategy) -> Strategy:
    STRATEGIES[strategy.name] = strategy
    return strategy


__all__ = ["Strategy", "STRATEGIES", "register_strategy"]
