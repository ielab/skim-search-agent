"""The harness contract: how a condition (a task with a strategy) is run on one question.

A strategy says what the agent has (tools with options, or a program's members); the harness
says how the model is put to work. `ReAct` (`react.py`) is the default: a loop in which the
model picks each step from the strategy's tools. `OneShotRag` (`rag.py`) ranks once and asks
the model once. `PlanAndSearch` (`plan_and_search.py`) is a team: a planner, one member
condition run through its own harness per sub-question, a synthesizer. One file per harness;
a strategy names one with `harness=`.

Every harness gets a `HarnessContext` (the condition, the corpus, the run's engines, the
model, and how to make a policy for a condition) and returns a `HarnessResult`: a
`Trajectory` (the record every run writes), the documents surfaced, and the member results
of a team. The evaluation runner (`agent_search/evaluation/agent_runner.py`) turns that into
the run record, so every harness is judged with the same metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence


@dataclass
class HarnessContext:
    condition: object                       # agent_search.strategies.conditions.Condition
    units: Sequence                         # the corpus
    ubyid: Mapping                          # doc_id -> unit
    engines: Mapping[str, object]           # engine kind -> the run's engine
    generate: Optional[Callable] = None     # messages -> text; None under the stub policy
    policy_for: Optional[Callable] = None   # condition -> a policy that proposes the next step
    max_steps: int = 50                     # the step budget of one loop episode
    files: dict = field(default_factory=dict)
    corpus_key: Optional[str] = None
    driver: str = "loop"                    # loop (text-parsed) | sdk (native function calling)
    model: Optional[str] = None
    api_base: Optional[str] = None
    field_profile: Optional[str] = None

    def for_condition(self, condition) -> "HarnessContext":
        """The same context for another condition (a team member)."""
        return HarnessContext(condition=condition, units=self.units, ubyid=self.ubyid, engines=self.engines,
                              generate=self.generate, policy_for=self.policy_for, max_steps=self.max_steps,
                              files=self.files, corpus_key=self.corpus_key, driver=self.driver,
                              model=self.model, api_base=self.api_base, field_profile=self.field_profile)


@dataclass
class HarnessResult:
    trajectory: object                      # agent_search.agent.loop.Trajectory
    surfaced: list = field(default_factory=list)     # every document listed or read
    members: list = field(default_factory=list)      # HarnessResult per member episode (teams)

    @property
    def located(self) -> list:
        return list(self.trajectory.located)

    @property
    def answer(self) -> str:
        return self.trajectory.final_answer or ""


class Harness:
    name: str = ""

    @property
    def members(self) -> tuple:
        """Strategy names a team runs as agents (empty for a single agent or a program)."""
        return ()

    def engines(self) -> tuple:
        """Engine kinds the harness reads itself, beyond the strategy's tools."""
        return ()

    def all_engines(self) -> tuple:
        """The harness's own engine kinds plus its members' (the run builds every one)."""
        from agent_search.strategies.base import STRATEGIES
        out = list(self.engines())
        for name in self.members:
            for e in STRATEGIES[name].engines:
                if e not in out:
                    out.append(e)
        return tuple(out)

    def prompts(self) -> list:
        """The harness's own prompt texts, hashed into the run identity (a loop harness has
        none of its own: the condition's rendered prompt is hashed by the runner)."""
        return []

    def run(self, question: str, ctx: HarnessContext) -> HarnessResult:
        raise NotImplementedError


def trajectory_from_steps(steps: list, located: Sequence[str], raw: str, stopped: str = "answer"):
    """A `Trajectory` for a program harness: its steps, the ranking it ends with, the answer
    read from `raw`, and the model usage since the runner's last reset."""
    from agent_search.agent.loop import Trajectory, _extract_answer
    import agent_search.agent.backbone as backends
    totals = backends.usage_totals()
    steps = list(steps)
    if steps and not any(getattr(st, "prompt_tokens", 0) for st in steps):
        # a program harness that did not meter its calls: the usage since the reset belongs to
        # the steps it made. With one step (one-shot RAG) the step is the whole call, so the
        # count-once accounting reads the real prompt size from it instead of zero.
        last = steps[-1]
        last.prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
        last.completion_tokens = int(totals.get("completion_tokens", 0) or 0)
    return Trajectory(task_id="q", steps=steps, located=list(located), declared=[],
                      llm_calls=int(totals.get("llm_calls", 0) or 0),
                      prompt_tokens=int(totals.get("prompt_tokens", 0) or 0),
                      completion_tokens=int(totals.get("completion_tokens", 0) or 0),
                      cached_input_tokens=int(totals.get("cached_input_tokens", 0) or 0),
                      reasoning_tokens=int(totals.get("reasoning_tokens", 0) or 0),
                      stopped_reason=stopped if raw else "no_model",
                      final_answer=_extract_answer(raw or ""))


__all__ = ["Harness", "HarnessContext", "HarnessResult", "trajectory_from_steps"]
