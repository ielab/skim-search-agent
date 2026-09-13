"""Shared forced-terminal-answer elicitation mechanism.

This module holds only the elicitation mechanism itself, not the offline-recovery-specific
prompt reconstruction or row I/O, which stays in `scripts/force_answer_backfill.py` (see that
module's docstring for the full design rationale). `agent_search/agent/loop.py`'s `run_episode`
reserves its last allowed turn to inject a "STEP BUDGET REACHED" nudge; on a served-vLLM policy
the model still tool-calls instead of answering 25-40% of the time on hard episodes, which would
otherwise leave `final_answer == ""`. This module is the one mechanism both the
offline backfill script and the live inline loop use to force an answer out of a model that
ignored that nudge:

  Primary, assistant prefill (`call_prefill`): append an `assistant`-role message whose content
  is the open tag `"<answer>"` and ask vLLM's OpenAI-compatible `/v1/chat/completions` to
  continue it (`continue_final_message=True`, `add_generation_prompt=False`, passed via
  `extra_body`, since these are not part of the OpenAI SDK's typed signature). The model has no
  token position left at which a tool call could start, since it is already mid-way through an
  open `<answer>` string, so this is deterministic and cannot be ignored the way a plain
  instruction can. It stops at `"</answer>"`; the returned text is the answer span, tag-free by
  construction.

  Fallback (`call_plain_ask`, used only if the prefill continuation is empty or whitespace): one
  plain, non-prefilled follow-up call asking the model to emit its own `<answer>...</answer>`,
  extracted with the rfind-last-`<answer>` extractor (see
  `agent_search.agent.loop._extract_answer`'s own docstring for why rfind, not the first match).

Asymmetry with the SDK driver: `continue_final_message` and `add_generation_prompt` are
vLLM-specific `ChatCompletionRequest` fields, part of vLLM's OpenAI-compatible server. A hosted
API model (OpenAI/Gemini via the Agents SDK) has no equivalent assistant-prefill affordance, so
`agent_search/agent/sdk_driver.py`'s `MaxTurnsExceeded` branch cannot use this module at all; it
instead asks and retries (up to 2 strict "Output ONLY <answer>...</answer>" retries), tagging
`elicitation="ask_retry_inline"`. This module's mechanism is used only by the loop driver (a
served-vLLM or OpenAI-compatible policy, which is what `AGENT_DRIVER=loop` runs live) and by the
offline backfill script.

Usage accounting and retries: `call_prefill` and `call_plain_ask` build the largest prompt of an
episode, since they re-send the whole conversation once more. Each records its `resp.usage` via
`agent_search.agent.backbone._record_usage` (the same `_cached_tokens`/`_reasoning_tokens` decomposition
`backends.py` itself uses), so this forcing call is never missing from cost accounting. Both
calls also go through `backends._with_retries` (exponential backoff on a transient
429/5xx/connection/timeout failure), the same helper `backends.py`'s own
`client.chat.completions.create(...)` calls use.
"""
from __future__ import annotations

import os
import time

from typing import Optional, Tuple

# --- tuning defaults (shared with force_answer_backfill.py) -------

# Generation budget of the forced final answer (`FORCED_ANSWER_TOKENS`, agent.forced_answer_tokens).
# The call stops at </answer>, so the budget only matters for a backbone that answers at length.
# Measured on 830-question runs: under the paper's short-span prompt the forced answer is 3 tokens
# at the median and 18 at the 90th percentile, and the old 200-token cap cut 2-6% of them; under
# Tongyi's own prompt the answers run to a few hundred words (DIVER: median 374, 90th percentile
# 694), and DIVER gives the call 10,000 tokens. 2,000 covers both with margin; the ITER files set
# DIVER's 10,000.
DEFAULT_PREFILL_MAX_TOKENS = int(os.environ.get("FORCED_ANSWER_TOKENS", "2000"))
DEFAULT_FALLBACK_MAX_TOKENS = 512        # the one plain-ask fallback, same cap the harness uses
DEFAULT_TEMPERATURE = 0.6                # matches agent_search.agent.backbone' Tongyi-native default
DEFAULT_SEED = 42

FORCE_MSG = (
    "STEP BUDGET REACHED. Tools are now DISABLED. You must output your best final answer "
    "NOW in the format <answer>...</answer>. If uncertain, give your best guess."
)
# the one plain-ask fallback's own prompt (used only if the forced prefill continuation is
# empty or whitespace). Asks for the tag itself, since there is no prefill to lean on here.
FALLBACK_MSG = (
    "You did NOT provide <answer> tags. Output ONLY <answer>your answer</answer>."
)

# elicitation tags recorded on a row's `elicitation` field (see loop.py::Trajectory and
# sdk_driver.py::SdkTrajectory): a single vocabulary shared by both drivers.
METHOD_NUDGE = "nudge"                       # the model complied with the inline budget nudge directly
METHOD_PREFILL_INLINE = "prefill_inline"     # loop driver: this module's forced call filled the answer
METHOD_PREFILL_FAILED = "prefill_failed"     # loop driver: forced call fired but produced nothing usable
METHOD_ASK_RETRY_INLINE = "ask_retry_inline"  # sdk driver: ask-and-retry filled the answer
METHOD_ASK_RETRY_FAILED = "ask_retry_failed"  # sdk driver: ask-and-retry exhausted its attempts


# the forced continuation of a task whose answer is plain text (terminal `text`, the Qwen3.5 and
# WebExplorer prompts): DIVER's qwen35_utils prefill, the whole continuation is the answer
TEXT_PREFILL = "Based on all the information gathered so far, my final answer is: "


def prefill_for(terminal: str = "answer") -> str:
    """The text the forced final answer continues from: FORCED_ANSWER_PREFILL when set, else an
    open `<answer>` tag, or `TEXT_PREFILL` for a text-terminal task."""
    override = os.environ.get("FORCED_ANSWER_PREFILL")
    if override:
        return override
    return TEXT_PREFILL if terminal == "text" else "<answer>"


def prefill_messages_for(messages: list, prefill: str = "<answer>") -> list:
    """`messages` plus the forced-continuation assistant turn (`prefill`, an open `<answer>` tag
    by default, with no closing tag) for vLLM's `continue_final_message` to extend."""
    return messages + [{"role": "assistant", "content": prefill}]


def _record_call_usage(resp) -> None:
    """Record `resp.usage` on the shared per-episode ledger (agent_search.agent.backbone). The
    largest prompt of an episode, the whole conversation re-sent for the forcing call, must not
    be missing from cost accounting just because it went through this module instead of
    backends.py's own generate() callables. Lazy import: avoids a module-load-time circular
    import (backends.py doesn't import this module, but keeping the import local mirrors
    `elicit_final_answer`'s own lazy import of `loop._extract_answer` just below)."""
    from agent_search.agent.backbone import _cached_tokens, _reasoning_tokens, _record_usage
    u = getattr(resp, "usage", None)
    if u is not None:
        _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                      _cached_tokens(u), _reasoning_tokens(u))


def call_prefill(client, model: str, messages: list, *, max_tokens: int = DEFAULT_PREFILL_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED,
                 prefill: str = "<answer>", stop: Optional[list] = None) -> str:
    """The primary forcing call: `messages` plus an open `<answer>` assistant turn, continued
    (not restarted) by vLLM via `continue_final_message=True` and `add_generation_prompt=False`
    (mutually exclusive per vLLM's `ChatCompletionRequest` validator; see the module docstring
    for the exact protocol fields and vLLM version). Stops at `"</answer>"`. Returns the raw
    continuation text: the answer span itself, tag-free by construction (never "" unless the
    model truly generated nothing, which the one plain-ask fallback in `elicit_final_answer`
    handles).

    Routed through `backends._with_retries` (transient 429/5xx/connection/timeout only; see
    module docstring) and records `resp.usage` on the shared per-episode ledger."""
    from agent_search.agent.backbone import _with_retries
    retries = max(1, int(os.environ.get("LLM_EMPTY_RETRIES", "10")))
    base = float(os.environ.get("LLM_RETRY_BASE_S", "1.0"))
    resp = None
    for attempt in range(retries):
        resp = _with_retries(lambda: client.chat.completions.create(
            model=model, messages=prefill_messages_for(messages, prefill), max_tokens=max_tokens,
            temperature=temperature, seed=seed, stop=(["</answer>"] if stop is None else stop) or None,
            extra_body={"add_generation_prompt": False, "continue_final_message": True}))
        _record_call_usage(resp)
        if (resp.choices[0].message.content or "").strip():
            break
        # an empty continuation is re-asked the way the served backend re-asks an empty turn
        # (LLM_EMPTY_RETRIES, backoff); see agent_search/agent/backbone/openai_chat.py
        time.sleep(min(base * (2 ** attempt), 30.0))
    return resp.choices[0].message.content or ""


def call_plain_ask(client, model: str, messages: list, *, max_tokens: int = DEFAULT_FALLBACK_MAX_TOKENS,
                   temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED) -> str:
    """The one fallback call (only when `call_prefill`'s continuation is empty or whitespace): a
    normal, non-prefilled completion asking the model to emit its own `<answer>...</answer>`
    (`FALLBACK_MSG` appended as the next user turn).

    Routed through `backends._with_retries` and records `resp.usage`, same as `call_prefill`."""
    from agent_search.agent.backbone import _with_retries
    msgs = messages + [{"role": "user", "content": f"<tool_response>\n{FALLBACK_MSG}\n</tool_response>"}]
    resp = _with_retries(lambda: client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature, seed=seed))
    _record_call_usage(resp)
    return resp.choices[0].message.content or ""


def elicit_final_answer(messages: list, client, model: str, *,
                        prefill_max_tokens: int = DEFAULT_PREFILL_MAX_TOKENS,
                        fallback_max_tokens: int = DEFAULT_FALLBACK_MAX_TOKENS,
                        temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED,
                        extract_fn=None, terminal: str = "answer") -> Tuple[str, str, str]:
    """Primary: forced assistant-prefill continuation (deterministic, single call,
    `call_prefill`). Fallback: exactly one plain-ask retry, only if the continuation is empty or
    whitespace. Returns `(answer_text, method_tag, raw_text)`:
      - `method_tag` is `"prefill"` (the primary call worked), `"plain_ask_fallback"` (the
        fallback call worked), or `"empty"` (both calls came back with no usable `<answer>`).
      - `raw_text` is the raw text of whichever call produced `answer_text` (kept for
        provenance, e.g. `scripts/force_answer_backfill.py`'s `recovered_answers.jsonl`
        `raw_continuation` field), or the fallback's raw text when both fail.

    `extract_fn` extracts an `<answer>...</answer>` span from the fallback's raw text; defaults
    to `agent_search.agent.loop._extract_answer` (imported lazily to avoid a module-load-time
    circular import, since `loop.py` itself calls into this module for the inline elicitation)."""
    prefill = prefill_for(terminal)
    if terminal == "text":
        # a plain-text answer: no closing tag to stop at, the whole continuation is the answer
        raw = call_prefill(client, model, messages, max_tokens=prefill_max_tokens,
                           temperature=temperature, seed=seed, prefill=prefill, stop=[])
        answer = raw.split("<tool_call>")[0].strip()
        return (answer, "prefill", raw) if answer else ("", "empty", raw)
    raw = call_prefill(client, model, messages, max_tokens=prefill_max_tokens,
                       temperature=temperature, seed=seed, prefill=prefill)
    # belt-and-braces strip: vLLM's default include_stop_str_in_output=False already excludes the
    # stop string from the returned text, but a differently-configured server or a model that
    # emits the closing tag anyway (max_tokens hit before the stop is seen, etc.) is handled too.
    answer = raw.split("</answer>")[0].strip()
    if answer:
        return answer, "prefill", raw
    raw2 = call_plain_ask(client, model, messages, max_tokens=fallback_max_tokens,
                          temperature=temperature, seed=seed)
    if extract_fn is None:
        from agent_search.agent.loop import _extract_answer as extract_fn
    answer2 = extract_fn(raw2) if raw2.rfind("<answer>") >= 0 else ""
    return answer2, ("plain_ask_fallback" if answer2 else "empty"), raw2
