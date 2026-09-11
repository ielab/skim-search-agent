"""The procedure contract: a fixed program over engines and agents, without being a tool loop.

A loop strategy hands the model a toolbox and lets it act step by step. A procedure is the
other kind of strategy: Python decides what happens. One-shot RAG is a procedure (rank, one
prompt, one model call). A multi-agent team is a procedure too: it runs member conditions
as agents (`agent_search.agent.episode.run_condition_episode`), one episode each, and
combines what they found. Procedures live one file each under this package, and a strategy
names one with `loop=False, procedure=...`.

A procedure gets a `ProcedureContext` (the engines the run built, the corpus, the model, and
how to make a policy for a member condition) and returns a `ProcedureResult` (a ranking, the
raw text the answer is read from, the steps for the run record, and the member episodes).
The harness (`agent_search/evaluation/agent_runner.py`, `ProcedureAgent`) turns the result
into the same trajectory record a single agent produces, so every strategy is judged alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence


@dataclass
class ProcedureContext:
    engines: Mapping[str, object]           # engine kind -> the run's engine
    ubyid: Mapping                          # doc_id -> unit
    units: Sequence                         # the corpus
    generate: Optional[Callable] = None     # messages -> text; None under the stub policy
    policy_for: Optional[Callable] = None   # condition -> a policy for one member episode
    max_steps: int = 50                     # step budget of each member episode
    files: dict = field(default_factory=dict)
    corpus_key: Optional[str] = None
    driver: str = "loop"
    model: Optional[str] = None
    api_base: Optional[str] = None
    field_profile: Optional[str] = None


@dataclass
class ProcedureResult:
    doc_ids: list                           # the ranking the procedure ends with
    raw: str = ""                           # the last generation; the answer is read from it
    steps: list = field(default_factory=list)      # agent_search.agent.loop.Step, for the record
    members: list = field(default_factory=list)    # EpisodeResult per member episode (teams)
    surfaced: list = field(default_factory=list)   # every document a member listed or read


class Procedure:
    name: str = ""
    engines: tuple = ()                     # engine kinds the procedure reads itself

    @property
    def members(self) -> tuple:
        """Strategy names the procedure runs as agents (empty for a loop-free program)."""
        return ()

    def run(self, question: str, ctx: ProcedureContext) -> ProcedureResult:
        raise NotImplementedError

    def all_engines(self) -> tuple:
        """The procedure's own engine kinds plus its members' (the run builds every one)."""
        from agent_search.strategies.base import STRATEGIES
        out = list(self.engines)
        for name in self.members:
            for e in STRATEGIES[name].engines:
                if e not in out:
                    out.append(e)
        return tuple(out)


__all__ = ["Procedure", "ProcedureContext", "ProcedureResult"]
