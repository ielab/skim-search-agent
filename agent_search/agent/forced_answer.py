"""Shared forced-terminal-answer elicitation mechanism.

WHY / MECHANISM: extracted verbatim from `scripts/force_answer_backfill.py` (see that
module's docstring for the full design rationale — this file holds ONLY the mechanism,
not the offline-recovery-specific prompt reconstruction / row I/O, which stays in the
backfill script). In short: `agent_search/agent/loop.py`'s `run_episode` reserves its
LAST allowed turn to inject a "STEP BUDGET REACHED" nudge; on a served-vLLM policy the
model still tool-calls instead of answering 25-40% of the time on hard episodes, which
used to leave `final_answer == ""`. This module is the ONE mechanism both the OFFLINE
backfill script and (as of this change) the LIVE inline loop use to force an answer out
of a model that just ignored that nudge:

  PRIMARY — ASSISTANT PREFILL (`call_prefill`): append an `assistant`-role message whose
  content is the OPEN tag `"<answer>"` and ask vLLM's OpenAI-compatible
  `/v1/chat/completions` to CONTINUE it (`continue_final_message=True`,
  `add_generation_prompt=False`, passed via `extra_body` — not part of the OpenAI SDK's
  typed signature). The model has no token position left at which a tool call could
  start — it is already mid-way through an open `<answer>` string — so this is
  deterministic and cannot be "ignored" the way a plain instruction can. Stops at
  `"</answer>"`; the returned text IS the answer span, tag-free by construction.

  FALLBACK (`call_plain_ask`, used only if the prefill continuation is empty/whitespace):
  ONE plain, non-prefilled follow-up call asking the model to emit its own
  `<answer>...</answer>`, extracted with the rfind-LAST-`<answer>` extractor (see
  `agent_search.agent.loop._extract_answer`'s own docstring for why rfind, not the first
  match).

ASYMMETRY WITH THE SDK DRIVER: `continue_final_message`/`add_generation_prompt` are vLLM-
specific `ChatCompletionRequest` fields (see
`envs/lib/python3.10/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py`).
A hosted API model (OpenAI/Gemini via the Agents SDK) has no equivalent assistant-prefill
affordance, so `agent_search/agent/sdk_driver.py`'s `MaxTurnsExceeded` branch cannot use
this module at all — it instead asks-and-retries (up to 2 strict "Output ONLY
<answer>...</answer>" retries), tagging `elicitation="ask_retry_inline"`. This module's
mechanism is used ONLY by the loop driver (a served-vLLM / OpenAI-compatible policy,
which is what `AGENT_DRIVER=loop` actually runs live) and by the offline backfill script.
"""
from __future__ import annotations

from typing import Optional, Tuple

# --- tuning defaults (identical to the pre-extraction values in force_answer_backfill.py) -------

DEFAULT_PREFILL_MAX_TOKENS = 200         # the answer span only — no room for tool-call syntax
DEFAULT_FALLBACK_MAX_TOKENS = 512        # the ONE plain-ask fallback, same cap the harness uses
DEFAULT_TEMPERATURE = 0.6                # matches agent_search.models.backends' Tongyi-native default
DEFAULT_SEED = 42

FORCE_MSG = (
    "STEP BUDGET REACHED. Tools are now DISABLED. You must output your best final answer "
    "NOW in the format <answer>...</answer>. If uncertain, give your best guess."
)
# the ONE plain-ask fallback's own prompt (used only if the forced prefill continuation is
# empty/whitespace) — asks for the tag itself, since there is no prefill to lean on here.
FALLBACK_MSG = (
    "You did NOT provide <answer> tags. Output ONLY <answer>your answer</answer>."
)

# elicitation tags recorded on a row's `elicitation` field (see loop.py::Trajectory /
# sdk_driver.py::SdkTrajectory) — a single vocabulary shared by both drivers.
METHOD_NUDGE = "nudge"                       # the model complied with the inline budget nudge directly
METHOD_PREFILL_INLINE = "prefill_inline"     # loop driver: this module's forced call filled the answer
METHOD_PREFILL_FAILED = "prefill_failed"     # loop driver: forced call fired but produced nothing usable
METHOD_ASK_RETRY_INLINE = "ask_retry_inline"  # sdk driver: ask-and-retry filled the answer
METHOD_ASK_RETRY_FAILED = "ask_retry_failed"  # sdk driver: ask-and-retry exhausted its attempts


def prefill_messages_for(messages: list) -> list:
    """`messages` plus the forced-continuation assistant turn: an OPEN `<answer>` tag with no
    closing tag, for vLLM's `continue_final_message` to extend."""
    return messages + [{"role": "assistant", "content": "<answer>"}]


def call_prefill(client, model: str, messages: list, *, max_tokens: int = DEFAULT_PREFILL_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED) -> str:
    """The PRIMARY forcing call: `messages` + an open `<answer>` assistant turn, continued (not
    restarted) by vLLM via `continue_final_message=True` + `add_generation_prompt=False` (mutually
    exclusive per vLLM's `ChatCompletionRequest` validator — see module docstring for the exact
    protocol fields and vLLM version). Stops at `"</answer>"`. Returns the raw continuation text —
    the answer span itself, tag-free by construction (never "" unless the model truly generated
    nothing, which the ONE plain-ask fallback in `elicit_final_answer` handles)."""
    resp = client.chat.completions.create(
        model=model, messages=prefill_messages_for(messages), max_tokens=max_tokens,
        temperature=temperature, seed=seed, stop=["</answer>"],
        extra_body={"add_generation_prompt": False, "continue_final_message": True})
    return resp.choices[0].message.content or ""


def call_plain_ask(client, model: str, messages: list, *, max_tokens: int = DEFAULT_FALLBACK_MAX_TOKENS,
                   temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED) -> str:
    """The ONE fallback call (only when `call_prefill`'s continuation is empty/whitespace): a
    normal, non-prefilled completion asking the model to emit its own `<answer>...</answer>`
    (`FALLBACK_MSG` appended as the next user turn)."""
    msgs = messages + [{"role": "user", "content": f"<tool_response>\n{FALLBACK_MSG}\n</tool_response>"}]
    resp = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature, seed=seed)
    return resp.choices[0].message.content or ""


def elicit_final_answer(messages: list, client, model: str, *,
                        prefill_max_tokens: int = DEFAULT_PREFILL_MAX_TOKENS,
                        fallback_max_tokens: int = DEFAULT_FALLBACK_MAX_TOKENS,
                        temperature: float = DEFAULT_TEMPERATURE, seed: Optional[int] = DEFAULT_SEED,
                        extract_fn=None) -> Tuple[str, str, str]:
    """PRIMARY: forced assistant-prefill continuation (deterministic, single call —
    `call_prefill`). FALLBACK: exactly ONE plain-ask retry, only if the continuation is
    empty/whitespace. Returns `(answer_text, method_tag, raw_text)`:
      - `method_tag` is `"prefill"` (the primary call worked), `"plain_ask_fallback"` (the
        fallback call worked), or `"empty"` (both calls came back with no usable `<answer>`).
      - `raw_text` is the raw text of whichever call produced `answer_text` (kept for audit —
        e.g. `scripts/force_answer_backfill.py`'s `recovered_answers.jsonl` `raw_continuation`
        field), or the fallback's raw text when both fail.

    `extract_fn` extracts an `<answer>...</answer>` span from the fallback's raw text; defaults
    to `agent_search.agent.loop._extract_answer` (imported lazily to avoid a module-load-time
    circular import, since `loop.py` itself calls into this module for the inline elicitation)."""
    raw = call_prefill(client, model, messages, max_tokens=prefill_max_tokens,
                       temperature=temperature, seed=seed)
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
