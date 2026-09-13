"""Responses-API driver: native function calling through `/v1/responses`, the protocol DIVER
evaluated gpt-oss with (`gptoss_responses_client.py`).

The loop driver (`agent_search/agent/loop.py`) reads tool calls out of the model's text. A
gpt-oss model served by vLLM speaks the harmony format instead: the server parses the tool
calls and returns them as `function_call` output items next to `reasoning` items, and the
conversation is a list of those items plus `function_call_output` results. This driver mirrors
DIVER's client turn for turn:

- the system prompt goes in `instructions`, the tools as native function definitions, the
  question through the task's user template (DIVER's `QUERY_TEMPLATE` for the strong prompt);
- `reasoning` is `{"effort": REASONING_EFFORT, "summary": "detailed"}`, `truncation` `auto`,
  `max_output_tokens` the model's `max_tokens` key;
- every `function_call` item runs through the tool box and its result goes back as a
  `function_call_output` item; a `mcp_call` item aimed at our functions is a function call the
  server mislabelled; a name with run-on header text is cut at the first `<`;
- a turn that ends on a bare `reasoning` item (cut off mid-thought) is dropped and retried;
- a request the server rejects (400: an item of the last turn does not re-parse when echoed
  back) drops that turn and lets the model take it again, twice at most; any other failure
  ends the episode with what it has (stopped reason `error`), as DIVER's client does;
- on the second-to-last turn the model is told retrieval is complete, and the final turn is
  made with no tools at all, so it must answer;
- the first turn without a function call ends the episode; the answer is that turn's message
  text, whole (DIVER's user template asks for Explanation / Exact Answer / Confidence lines).

The result is the loop driver's `Trajectory`, with one `Step` per tool call, so rows, judging
and the count-once token accounting are the same as for every other driver. Per-step
`prompt_tokens` are the turn's `input_tokens`, so the context-once metering (the largest prompt
the model ever held, minus the first) applies unchanged.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from agent_search.agent.loop import Step, Trajectory

FINAL_ROUND_MSG = ("Retrieval complete. You are forbidden to call any tools now. "
                   "You must provide your final answer based on the above info.")
DEFAULT_MAX_OUTPUT_TOKENS = 10000     # DIVER's --max-tokens for gpt-oss


def responses_tools(ws) -> list:
    """The tool box's declarations in the Responses API's flat function format."""
    out = []
    for name in getattr(ws, "tools", ()) or ():
        d = ws[name].declaration()
        out.append({"type": "function", "name": d["name"], "description": d["description"],
                    "parameters": d.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _item_text(item: dict) -> str:
    """The text of a `message` or `reasoning` output item."""
    parts = item.get("content")
    if isinstance(parts, str):
        return parts
    texts = []
    for c in parts or []:
        if isinstance(c, dict) and c.get("type") in ("output_text", "text", "reasoning_text"):
            texts.append(str(c.get("text") or ""))
    if not texts and item.get("type") == "reasoning":
        for s in item.get("summary") or []:
            if isinstance(s, dict) and s.get("text"):
                texts.append(str(s["text"]))
    return "\n".join(t for t in texts if t)


def _clean_name(name: str, known) -> str:
    """A tool name with the run-on header text some turns append, resolved to a known tool."""
    name = (name or "").split("<")[0].strip()
    if name.startswith("functions."):
        name = name[len("functions."):]
    for k in known:
        if name == k or name.startswith(k):
            return k
    return name


def _is_bad_request(e: Exception) -> bool:
    """A 4xx the server answers with when it cannot parse the request (openai.BadRequestError, or
    anything carrying a 400 status), as opposed to a transient failure."""
    if type(e).__name__ == "BadRequestError":
        return True
    return getattr(e, "status_code", None) == 400 or "Error code: 400" in str(e)


def _usage(resp) -> tuple:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0, 0, 0
    inp = int(getattr(u, "input_tokens", 0) or 0)
    out = int(getattr(u, "output_tokens", 0) or 0)
    cached = int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0)
    reasoning = int(getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
    return inp, out, cached, reasoning


def run_episode_responses(ws, question: str, *, model: str, instructions: str,
                          api_base: Optional[str] = None, client=None, max_turns: int = 50,
                          reasoning_effort: Optional[str] = None,
                          max_output_tokens: Optional[int] = None,
                          user_template: Optional[str] = None,
                          on_step=None, before_tool=None) -> Trajectory:
    """One episode over `ws` through the Responses API. `client` is injectable for tests;
    otherwise an OpenAI client is opened on `api_base` (OPENAI_API_KEY, "EMPTY" for vLLM)."""
    if not instructions:
        raise ValueError("run_episode_responses requires a non-empty `instructions` prompt")
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=api_base or "http://localhost:8000/v1",
                        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        timeout=float(os.environ.get("LLM_TIMEOUT_S", "1800")), max_retries=0)
    from agent_search.agent.backbone import _record_usage, _with_retries
    effort = reasoning_effort or os.environ.get("REASONING_EFFORT") or "medium"
    max_out = int(max_output_tokens or os.environ.get("LLM_MAX_TOKENS") or DEFAULT_MAX_OUTPUT_TOKENS)
    user = (user_template or "{question}").replace("{Question}", "{question}").format(question=question)
    messages: list = [{"role": "user", "content": user}]
    tools = responses_tools(ws)
    known = tuple(getattr(ws, "tools", ()) or ())
    steps: list = []
    final_answer, reason, llm_calls = "", "max_steps", 0
    tot_in = tot_out = tot_cached = tot_reasoning = 0
    force_text_only = False

    def _push(step: Step) -> None:
        steps.append(step)
        if on_step is not None:
            try:
                on_step(step)
            except Exception:  # noqa: BLE001 - a spectator must never kill the episode
                pass

    turn_starts: list = []        # index into `messages` where each turn's output items begin
    rejected = 0
    for i in range(max_turns):
        is_last_round = (i == max_turns - 2)
        request = {"model": model, "max_output_tokens": max_out, "input": list(messages),
                   "truncation": "auto", "reasoning": {"effort": effort, "summary": "detailed"},
                   "instructions": instructions}
        if not force_text_only:
            request["tools"] = tools
        t0 = time.monotonic()
        try:
            resp = _with_retries(lambda: client.responses.create(**request))
        except Exception as e:  # noqa: BLE001
            if not _is_bad_request(e) or not turn_starts or rejected >= 2:
                # a transient failure already went through the retry helper; anything else ends
                # the episode with what it has, as DIVER's client does (status incomplete)
                reason = "error"
                _push(Step(name="none", args={}, observation=f"ERROR: the model call failed: {e}"[:400],
                           raw_output="", t_llm=time.monotonic() - t0))
                break
            # the server rejected the transcript: an output item of the last turn does not
            # re-parse when echoed back (a harmony header the model ran together, for one).
            # Drop that turn's items and its tool outputs and let the model take the turn again.
            rejected += 1
            del messages[turn_starts.pop():]
            _push(Step(name="none", args={},
                       observation="ERROR: the server rejected the transcript; the last turn was dropped and taken again",
                       raw_output=str(e)[:400], t_llm=time.monotonic() - t0))
            continue
        t_llm = time.monotonic() - t0
        llm_calls += 1
        inp, out, cached, reasoning = _usage(resp)
        tot_in += inp; tot_out += out; tot_cached += cached; tot_reasoning += reasoning
        _record_usage(inp, out, cached, reasoning)
        items = resp.model_dump(mode="python")["output"] if hasattr(resp, "model_dump") else list(resp.output)
        for it in items:
            # the server labels a call from the analysis channel as an MCP call; it is ours
            if it.get("type") == "mcp_call" and it.get("server_label") == "functions":
                it["type"] = "function_call"
                it["call_id"] = it.pop("id", None) or it.get("call_id")
        turn_starts.append(len(messages))
        messages.extend(items)
        if items and items[-1].get("type") == "reasoning" and not force_text_only:
            # cut off mid-thought with no message or call: drop the dangling item and retry
            messages.pop()
            turn_starts.pop()
            _push(Step(name="none", args={}, observation="ERROR: the turn ended mid-thought; retried",
                       raw_output=_item_text(items[-1]), t_llm=t_llm, prompt_tokens=inp, completion_tokens=out))
            continue
        thinking = "\n".join(_item_text(it) for it in items if it.get("type") == "reasoning")
        calls = [it for it in items if it.get("type") == "function_call"] if not force_text_only else []
        if not calls:
            text = "\n".join(_item_text(it) for it in items if it.get("type") == "message").strip()
            final_answer = text
            reason = "answer"
            _push(Step(name="answer", args={}, observation="(episode ended)",
                       raw_output=(thinking + "\n" + text).strip(), t_llm=t_llm,
                       prompt_tokens=inp, completion_tokens=out))
            break
        for n, fc in enumerate(calls):
            name = _clean_name(fc.get("name"), known)
            try:
                args = json.loads(fc.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except Exception as e:  # noqa: BLE001
                args, obs = {}, f'ERROR: invalid tool call ({e}). Provide a valid "name" and "arguments".'
                t_tool = 0.0
            else:
                if before_tool is not None:
                    before_tool(name, args, thinking)
                t1 = time.monotonic()
                obs = ws.run(name, args)
                t_tool = time.monotonic() - t1
            messages.append({"type": "function_call_output", "call_id": fc.get("call_id"), "output": obs})
            _push(Step(name=name, args=args, observation=obs,
                       raw_output=(thinking + "\n" + json.dumps({"name": name, "arguments": args})).strip() if n == 0
                       else json.dumps({"name": name, "arguments": args}),
                       t_llm=t_llm if n == 0 else 0.0, t_tool=t_tool,
                       prompt_tokens=inp if n == 0 else None, completion_tokens=out if n == 0 else None))
        if is_last_round:
            messages.append({"role": "user", "content": FINAL_ROUND_MSG})
            force_text_only = True

    surfaced = list(getattr(ws, "surfaced", []) or [])
    traj = Trajectory(task_id="q", steps=steps, located=surfaced, declared=[], llm_calls=llm_calls,
                      stopped_reason=reason, final_answer=final_answer,
                      elicitation=("nudge" if force_text_only and reason == "answer" else None))
    traj.prompt_tokens, traj.completion_tokens = tot_in, tot_out
    traj.cached_input_tokens, traj.reasoning_tokens = tot_cached, tot_reasoning
    return traj


__all__ = ["run_episode_responses", "responses_tools", "FINAL_ROUND_MSG", "DEFAULT_MAX_OUTPUT_TOKENS"]
