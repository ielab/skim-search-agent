"""Plan and search: a planner splits the question, one agent per sub-question, a synthesizer
answers. The first multi-agent harness, and the template for the next one.

Three roles, all served by the run's model:
  1. The planner reads the question and writes up to `max_subquestions` sub-questions, one
     per line (`PLANNER_PROMPT`). Without a model (the stub policy) the plan is the question.
  2. Each sub-question runs as one episode of the member condition: the research task with
     the `searcher` strategy (`sieve_bm25` by default), through that strategy's own harness
     (ReAct for a tool strategy), with the run's step budget. A member is an ordinary
     condition, so its trajectory is recorded in full.
  3. The synthesizer reads the question and every member's answer and cited documents and
     writes the final `<answer>` (`SYNTHESIZER_PROMPT`). Without a model the first member's
     answer is the answer.

The record: one `plan` step, one `member` step per sub-question, one `synthesize` step, the
member trajectories under `members`, the union of the members' surfaced documents for the
gold-coverage metric, and the union of their rankings (first member first) as the ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from agent_search.harness.base import Harness, HarnessContext, HarnessResult, trajectory_from_steps

PLANNER_PROMPT = (
    "You are the planner of a research team. Split the question into the smallest set of "
    "independent sub-questions whose answers together answer it, at most {n}. Each sub-question "
    "must be answerable on its own from a document collection. Write one sub-question per line, "
    "numbered, and nothing else.")

SYNTHESIZER_PROMPT = (
    "You are the synthesizer of a research team. You get the question, and for each sub-question "
    "a researcher's answer with the documents they cited. Combine them into the final answer. "
    "Answer with ONLY the short answer span inside <answer></answer> tags — no explanation, no "
    "extra text.")

_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$")


def parse_plan(text: str, n: int) -> list[str]:
    """The sub-questions in a planner generation: numbered or bulleted lines, at most `n`."""
    out = []
    for line in (text or "").splitlines():
        m = _NUMBERED.match(line)
        if m and m.group(1):
            out.append(m.group(1))
    return out[:n]


def member_condition(searcher: str):
    """The research task paired with the `searcher` strategy, as a condition."""
    from agent_search.strategies.conditions import CONDITIONS, Condition
    from agent_search.strategies.base import STRATEGIES
    from agent_search.tasks.base import TASKS
    cond = CONDITIONS.get(searcher)
    if cond is not None and cond.task.name == "research":
        return cond
    return Condition(name=searcher, task=TASKS["research"], strategy=STRATEGIES[searcher])


@dataclass(frozen=True)
class PlanAndSearch(Harness):
    searcher: str = "sieve_bm25"            # the member strategy (research task)
    max_subquestions: int = 3

    @property
    def name(self) -> str:                  # type: ignore[override]
        return f"plan_and_search_{self.searcher}"

    @property
    def members(self) -> tuple:             # type: ignore[override]
        return (self.searcher,)

    def prompts(self) -> list:
        return [PLANNER_PROMPT, SYNTHESIZER_PROMPT]

    # --- the three roles --------------------------------------------------------------------
    def plan(self, question: str, ctx: HarnessContext) -> tuple[list[str], str]:
        if ctx.generate is None:
            return [question], ""
        raw = ctx.generate([{"role": "system", "content": PLANNER_PROMPT.format(n=self.max_subquestions)},
                            {"role": "user", "content": question}]) or ""
        subqs = parse_plan(raw, self.max_subquestions)
        return (subqs or [question]), raw

    def search(self, subquestion: str, ctx: HarnessContext) -> HarnessResult:
        member = member_condition(self.searcher)
        return member.strategy.harness.run(subquestion, ctx.for_condition(member))

    def synthesize(self, question: str, subqs: Sequence[str], members: Sequence[HarnessResult],
                   ctx: HarnessContext) -> str:
        if ctx.generate is None:
            first = next((m.answer for m in members if m.answer), "")
            return f"<answer>{first}</answer>" if first else ""
        findings = []
        for i, (sq, m) in enumerate(zip(subqs, members), 1):
            cited = ", ".join(_titles(m.surfaced[:5], ctx.ubyid)) or "none"
            findings.append(f"[{i}] {sq}\n    answer: {m.answer or '(none)'}\n    cited: {cited}")
        user = f"Question: {question}\n\nFindings:\n" + "\n".join(findings)
        return ctx.generate([{"role": "system", "content": SYNTHESIZER_PROMPT},
                             {"role": "user", "content": user}]) or ""

    # --- the program --------------------------------------------------------------------------
    def run(self, question: str, ctx: HarnessContext) -> HarnessResult:
        from agent_search.agent.loop import Step
        subqs, plan_raw = self.plan(question, ctx)
        steps = [Step(name="plan", args={"question": question}, observation="\n".join(subqs), raw_output=plan_raw)]
        members = []
        for sq in subqs:
            res = self.search(sq, ctx)
            members.append(res)
            steps.append(Step(name="member", args={"strategy": self.searcher, "query": sq},
                              observation=f"answer: {res.answer or '(none)'}; {len(res.surfaced)} documents surfaced; "
                                          f"{len(res.trajectory.steps)} steps",
                              prompt_tokens=res.trajectory.prompt_tokens,
                              completion_tokens=res.trajectory.completion_tokens))
        raw = self.synthesize(question, subqs, members, ctx)
        steps.append(Step(name="synthesize", args={"question": question, "members": len(members)},
                          observation="(final answer written from the members' findings)", raw_output=raw))
        doc_ids = _union(m.located for m in members)
        surfaced = _union(m.surfaced for m in members)
        return HarnessResult(trajectory=trajectory_from_steps(steps, doc_ids, raw), surfaced=surfaced, members=members)


def _union(lists) -> list:
    out, seen = [], set()
    for lst in lists:
        for d in lst:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def _titles(doc_ids: Sequence[str], ubyid) -> list[str]:
    out = []
    for d in doc_ids:
        u = ubyid.get(d)
        out.append(repr(u.title or u.qualname or d) if u is not None else repr(d))
    return out


__all__ = ["PlanAndSearch", "PLANNER_PROMPT", "SYNTHESIZER_PROMPT", "parse_plan", "member_condition"]
