"""ReAct: the default harness. One loop in which the model reasons, picks one of the
strategy's tools, and reads the observation, until it answers or the step budget ends
(`agent_search/agent/loop.py`). The Agents-SDK driver (`ctx.driver == "sdk"`, native function
calling) runs the same tools through a hosted model's own tool-calling API; the Responses driver
(`ctx.driver == "responses"`, `agent_search/agent/responses_driver.py`) runs them through
`/v1/responses`, the protocol DIVER evaluated gpt-oss with.

The dense-query context (`agent_search.training.history`) is set for the episode so a trained
retriever sees the same history text it was trained on.
"""
from __future__ import annotations

import os
import re
from datetime import date

from agent_search.harness.base import Harness, HarnessContext, HarnessResult
from agent_search.tools.base import EpisodeState


class ReAct(Harness):
    name = "react"

    def run(self, question: str, ctx: HarnessContext) -> HarnessResult:
        condition = ctx.condition
        strategy, task = condition.strategy, condition.task
        state = EpisodeState(question=question)
        engine_map = {k: ctx.engines[k] for k in strategy.engines}
        ws = strategy.toolbox(state, ctx.units, ctx.ubyid, engine_map, files=ctx.files or {}, corpus_key=ctx.corpus_key)
        driver = os.environ.get("AGENT_DRIVER") or ctx.driver
        if driver == "sdk" and strategy.sdk:
            traj = self._run_sdk(ctx, ws, question)
            return HarnessResult(trajectory=traj, surfaced=sorted(getattr(ws, "seen", set()) or set()))
        if driver == "responses":
            traj = self._run_responses(ctx, ws, question)
            return HarnessResult(trajectory=traj, surfaced=sorted(getattr(ws, "seen", set()) or set()))
        from agent_search.agent.loop import Task as LoopTask, run_episode
        from agent_search.tasks.codefix.guards import fix_guard_for
        import agent_search.agent.backbone as backends
        from agent_search.training.history import CURRENT, QueryContext, dense_query_style
        from agent_search.training.triples import is_read_action, is_search_action, read_ids
        policy = ctx.policy_for(condition) if ctx.policy_for is not None else None
        if policy is None:
            from agent_search.agent.policies import KeywordPolicy
            policy = KeywordPolicy(condition.tool_names)
        guard = fix_guard_for(strategy.tool_names, ws) if task.terminal in ("fix", "patch") else None

        def _doc_text(doc_id: str):
            u = ctx.ubyid.get(doc_id)
            if u is None:
                return None
            return "\n".join(p for p in (u.title, u.body or u.code) if p)

        qctx = QueryContext(question=question, text_of=_doc_text, style=dense_query_style())

        def _on_step(step) -> None:
            hits = list(getattr(ws, "last_hits", []) or [])
            step.hit_ids = hits if is_search_action(step.name) else []
            step.read_ids = read_ids({"args": step.args}, hits) if is_read_action(step.name) else []
            qctx.observe(step, hits)

        def _before_tool(name, args, raw) -> None:
            qctx.note(raw)

        token = CURRENT.set(qctx)
        try:
            traj = run_episode(policy, LoopTask(task_id="q", query=question), ws, ctx.units,
                               max_steps=ctx.max_steps, usage_fn=backends.usage_events,
                               domain=task.domain, terminal=task.terminal, before_tool=_before_tool, fix_guard=guard, on_step=_on_step)
        finally:
            CURRENT.reset(token)
        return HarnessResult(trajectory=traj, surfaced=sorted(getattr(ws, "seen", set()) or set()))

    def _run_responses(self, ctx: HarnessContext, ws, question: str):
        """Native function calling through the Responses API (DIVER's gpt-oss protocol). The
        dense-query context is kept the way the loop keeps it: the turn's reasoning before a
        search is the pre-search reasoning, the hits and reads are observed per step."""
        from agent_search.agent.backbone import DEFAULT_MODEL
        from agent_search.agent.responses_driver import run_episode_responses
        from agent_search.training.history import CURRENT, QueryContext, dense_query_style
        from agent_search.training.triples import is_read_action, is_search_action, read_ids
        condition = ctx.condition
        task = condition.task
        system = re.sub(r"\n{3,}", "\n\n", condition.render(ctx.field_profile)).strip()
        system = system.replace("{{step_budget}}", str(ctx.max_steps))

        def _doc_text(doc_id: str):
            u = ctx.ubyid.get(doc_id)
            if u is None:
                return None
            return "\n".join(p for p in (u.title, u.body or u.code) if p)

        qctx = QueryContext(question=question, text_of=_doc_text, style=dense_query_style())

        def _on_step(step) -> None:
            hits = list(getattr(ws, "last_hits", []) or [])
            step.hit_ids = hits if is_search_action(step.name) else []
            step.read_ids = read_ids({"args": step.args}, hits) if is_read_action(step.name) else []
            qctx.observe(step, hits)

        def _before_tool(name, args, thinking) -> None:
            qctx.note(thinking)

        token = CURRENT.set(qctx)
        try:
            return run_episode_responses(ws, question, model=ctx.model or DEFAULT_MODEL, api_base=ctx.api_base,
                                         instructions=system, max_turns=ctx.max_steps,
                                         user_template=getattr(task, "user_template", None),
                                         on_step=_on_step, before_tool=_before_tool)
        finally:
            CURRENT.reset(token)

    def _run_sdk(self, ctx: HarnessContext, ws, question: str):
        from agent_search.agent.loop import Step, Trajectory, resolve_locations
        from agent_search.agent.sdk_driver import run_episode_sdk
        from agent_search.agent.backbone import DEFAULT_MODEL
        condition = ctx.condition
        system = re.sub(r"\n{3,}", "\n\n", condition.render(ctx.field_profile)).strip()
        system = system.replace("{{step_budget}}", str(ctx.max_steps))
        user_input = f"Current date: {date.today().isoformat()}\n\n{question}"
        sdk = run_episode_sdk(ws, user_input, model=ctx.model or DEFAULT_MODEL, api_base=ctx.api_base,
                              max_turns=ctx.max_steps, instructions=system)
        steps = [Step(name=n, args=a, observation=o) for (n, a, o) in sdk.steps]
        if steps:
            steps[0].prompt_tokens = sdk.first_input_tokens
        located = list(getattr(ws, "last_hits", []) or []) or list(getattr(ws, "seen", []) or [])
        if condition.task.terminal in ("fix", "patch") and sdk.fix_text:
            from agent_search.evaluation.fix_scoring import fix_file
            f = fix_file(sdk.fix_text)
            located = resolve_locations([f], ctx.units) if f else located
        return Trajectory(task_id="q", steps=steps, located=located, declared=[], llm_calls=sdk.requests,
                          prompt_tokens=sdk.input_tokens, completion_tokens=sdk.output_tokens,
                          cached_input_tokens=sdk.cached_input_tokens, reasoning_tokens=sdk.reasoning_tokens,
                          stopped_reason=("fix" if sdk.fix_text else getattr(sdk, "stopped_reason", "answer")),
                          final_answer=sdk.final_answer, fix_text=sdk.fix_text, elicitation=sdk.elicitation)


__all__ = ["ReAct"]
