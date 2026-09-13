"""The run record of one episode: the rows.jsonl shape every harness produces."""
from __future__ import annotations

import re


def trajectory_meta(traj, surfaced=None) -> dict:
    """Serialize a trajectory into the rows.jsonl shape the analysis tools expect.

    `surfaced` is every document the episode listed or read, used for gold-document coverage
    on the document domain; the code domain carries its <fix> text for fix scoring."""
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
            "finish_reason": getattr(s, "finish_reason", None),
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
        # tool calls whose observation is an error string (a tool exception becomes an
        # observation, never a crash); a cell with many of these has a serving problem
        "tool_errors": sum(1 for s in steps if str(s["observation"]).startswith("ERROR")),
        "prompt_tokens": traj.prompt_tokens, "completion_tokens": traj.completion_tokens,
        "cached_input_tokens": traj.cached_input_tokens,
        "reasoning_tokens": traj.reasoning_tokens,
        "fix_text": traj.fix_text,
        # documents surfaced this episode (search hits + fetch/visit targets), for gold-doc coverage.
        "surfaced_docs": sorted(set(surfaced or ())),
        # every tool response the agent saw lives, in full and uncapped, on
        # `trajectory[i]["observation"]` (see agent_search.evaluation.rows.observations_of).
        "trajectory": steps,
    }


__all__ = ["trajectory_meta"]
