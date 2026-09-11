"""Run one condition (a task with a strategy) on one question: the single-agent unit.

`run_condition_episode` binds the strategy's tools to a fresh episode state, drives the agent
loop (or the Agents-SDK driver) with the policy, and returns the trajectory, its serialized
record and the surfaced documents. The evaluation harness calls it once per question
(`agent_search/evaluation/agent_runner.py`); a multi-agent procedure calls it once per member
(`agent_search/procedures/`). Both get the same record shape, so a team is judged with the
same metrics as one agent.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.strategies.conditions import Condition
from agent_search.tools.base import EpisodeState


@dataclass
class EpisodeResult:
    trajectory: object                      # agent_search.agent.loop.Trajectory
    meta: dict                              # the rows.jsonl record (`trajectory_meta`)
    located: list = field(default_factory=list)      # the ranking the episode ends with
    surfaced: list = field(default_factory=list)     # every document listed or read
    answer: str = ""


def run_condition_episode(condition: Condition, question: str, *, units: Sequence[CodeUnit], ubyid,
                          engines, policy, max_steps: int = 50, files: Optional[dict] = None,
                          corpus_key: Optional[str] = None, driver: str = "loop",
                          model: Optional[str] = None, api_base: Optional[str] = None,
                          field_profile: Optional[str] = None) -> EpisodeResult:
    """One episode of `condition` on `question`.

    `engines` is the run's `Engines` registry (or a dict of the kinds the strategy needs);
    `policy` proposes the next generation (`agent_search.agent.policies`); `driver` is `loop`
    (the text-parsed loop) or `sdk` (native function calling, when the strategy allows it).
    The dense-query context (`agent_search.training.history`) is set for the episode so a
    trained retriever sees the same history text it was trained on."""
    strategy, task = condition.strategy, condition.task
    state = EpisodeState(question=question)
    engine_map = {k: engines.get(k) for k in strategy.engines} if hasattr(engines, "get") and not isinstance(engines, dict) else dict(engines)
    ws = strategy.toolbox(state, units, ubyid, engine_map, files=files or {}, corpus_key=corpus_key)
    driver = os.environ.get("AGENT_DRIVER") or driver
    if driver == "sdk" and strategy.sdk:
        traj = _run_sdk(condition, ws, question, units, max_steps=max_steps, model=model, api_base=api_base,
                        field_profile=field_profile)
        return _result(traj, ws)
    from agent_search.agent.loop import Task as LoopTask, run_episode
    from agent_search.tasks.codefix.guards import fix_guard_for
    import agent_search.agent.backbone as backends
    from agent_search.training.history import CURRENT, QueryContext, dense_query_style
    from agent_search.training.triples import is_read_action, is_search_action, read_ids
    guard = fix_guard_for(strategy.tool_names, ws) if task.terminal in ("fix", "patch") else None

    def _doc_text(doc_id: str):
        u = ubyid.get(doc_id)
        if u is None:
            return None
        return "\n".join(p for p in (u.title, u.body or u.code) if p)

    ctx = QueryContext(question=question, text_of=_doc_text, style=dense_query_style())

    def _on_step(step) -> None:
        hits = list(getattr(ws, "last_hits", []) or [])
        step.hit_ids = hits if is_search_action(step.name) else []
        step.read_ids = read_ids({"args": step.args}, hits) if is_read_action(step.name) else []
        ctx.observe(step, hits)

    def _before_tool(name, args, raw) -> None:
        ctx.note(raw)

    token = CURRENT.set(ctx)
    try:
        traj = run_episode(policy, LoopTask(task_id="q", query=question), ws, units,
                           max_steps=max_steps, usage_fn=backends.usage_events,
                           domain=task.domain, before_tool=_before_tool, fix_guard=guard, on_step=_on_step)
    finally:
        CURRENT.reset(token)
    return _result(traj, ws)


def _result(traj, ws) -> EpisodeResult:
    meta = trajectory_meta(traj, ws)
    return EpisodeResult(trajectory=traj, meta=meta, located=list(traj.located),
                         surfaced=list(meta.get("surfaced_docs") or []), answer=traj.final_answer or "")


def _run_sdk(condition: Condition, ws, question: str, units, *, max_steps: int, model: Optional[str],
             api_base: Optional[str], field_profile: Optional[str]):
    from agent_search.agent.loop import Step, Trajectory, resolve_locations
    from agent_search.agent.sdk_driver import run_episode_sdk
    from agent_search.agent.backbone import DEFAULT_MODEL
    system = re.sub(r"\n{3,}", "\n\n", condition.render(field_profile)).strip()
    system = system.replace("{{step_budget}}", str(max_steps))
    user_input = f"Current date: {date.today().isoformat()}\n\n{question}"
    sdk = run_episode_sdk(ws, user_input, model=model or DEFAULT_MODEL, api_base=api_base,
                          max_turns=max_steps, instructions=system)
    steps = [Step(name=n, args=a, observation=o) for (n, a, o) in sdk.steps]
    if steps:
        steps[0].prompt_tokens = sdk.first_input_tokens
    located = list(getattr(ws, "last_hits", []) or []) or list(getattr(ws, "seen", []) or [])
    if condition.task.terminal in ("fix", "patch") and sdk.fix_text:
        from agent_search.evaluation.fix_scoring import fix_file
        f = fix_file(sdk.fix_text)
        located = resolve_locations([f], units) if f else located
    return Trajectory(task_id="q", steps=steps, located=located, declared=[], llm_calls=sdk.requests,
                      prompt_tokens=sdk.input_tokens, completion_tokens=sdk.output_tokens,
                      cached_input_tokens=sdk.cached_input_tokens, reasoning_tokens=sdk.reasoning_tokens,
                      stopped_reason=("fix" if sdk.fix_text else getattr(sdk, "stopped_reason", "answer")),
                      final_answer=sdk.final_answer, fix_text=sdk.fix_text, elicitation=sdk.elicitation)


def trajectory_meta(traj, ws=None) -> dict:
    """Serialize an episode into the rows.jsonl shape the analysis tools expect.

    `ws` (the episode's toolbox) supplies the surfaced-document set used for gold-document
    coverage on the document domain; the code domain carries its <fix> text for fix scoring."""
    steps = []
    for s in traj.steps:
        # the count from a search-shaped observation header: "(N units in M files ...)"
        # (code search->fetch), "(N matches ...)" (doc), or "N units matched /pat/" (code
        # grep baseline); 0 otherwise (dci's bash/read have no such header). Advisory only.
        m = (re.search(r"\((\d+)\s+(?:units|matches)\b", s.observation)
             or re.match(r"(\d+)\s+units matched\b", s.observation))
        n_hits = int(m.group(1)) if m else 0
        steps.append({
            "action": s.name, "args": s.args,
            "query": (s.args.get("query") or s.args.get("q") or ""),
            "observation": s.observation, "raw_output": s.raw_output, "n_hits": n_hits,
            "hit_ids": list(getattr(s, "hit_ids", []) or []),
            "read_ids": list(getattr(s, "read_ids", []) or []),
            "t_llm_s": round(s.t_llm, 3), "t_tool_s": round(s.t_tool, 3),
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
        })
    return {
        "queries": [s["query"] for s in steps],
        "actions": [s["action"] for s in steps],
        "hits_per_step": [s["n_hits"] for s in steps],
        "stopped": traj.stopped_reason, "final_answer": traj.final_answer,
        # provenance of a non-empty final_answer on a force-answer-gated episode: None (the
        # episode never reached the reserved final turn), "nudge" (the model complied with the
        # inline budget nudge), "prefill_inline" (the forced-answer elicitation call filled it
        # in, agent_search/agent/forced_answer.py), "prefill_failed" (that call produced nothing
        # usable). SDK-driven episodes carry "ask_retry_inline"/"ask_retry_failed".
        "elicitation": getattr(traj, "elicitation", None),
        "declared": traj.declared, "llm_calls": traj.llm_calls, "n_steps": len(traj.steps),
        "prompt_tokens": traj.prompt_tokens, "completion_tokens": traj.completion_tokens,
        "cached_input_tokens": traj.cached_input_tokens,
        "reasoning_tokens": traj.reasoning_tokens,
        "fix_text": traj.fix_text,
        # documents surfaced this episode (search hits + fetch/visit targets), for gold-doc coverage.
        "surfaced_docs": sorted(getattr(ws, "seen", set()) or set()),
        # every tool response the agent saw lives, in full and uncapped, on
        # `trajectory[i]["observation"]` (see agent_search.evaluation.rows.observations_of).
        "trajectory": steps,
    }


__all__ = ["EpisodeResult", "run_condition_episode", "trajectory_meta"]
