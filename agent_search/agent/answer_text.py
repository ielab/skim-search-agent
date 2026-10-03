"""The visible part of a model turn and the answer text inside it.

Some chat templates (OpenSeeker, OpenResearcher, other Qwen3 fine-tunes) put the opening `<think>`
in the prompt, so the generation carries only the closing `</think>`. Everything before a lone
`</think>` is reasoning too. `visible_text` drops both kinds of reasoning; `clean_answer` also
drops tool calls, which a final answer never contains. The loop and the forced answer read answers
with `clean_answer` (an empty one is asked for again); the judges read `judge_text`, which never
passes a tool call on but falls back to the reasoning when that is all the model wrote.

A model that quotes the prompt's `<answer>...</answer>` format inside its reasoning hits the
`</answer>` stop string mid-thought. `in_open_think` tells such a cut turn from a finished one:
the backend continues it, and nothing in it is read as an answer.
"""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LONE_CLOSE = re.compile(r"^.*</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)


def in_open_think(text: str | None) -> bool:
    """True when the text ends inside a `<think>` block that was opened and never closed: the
    generation was cut while the model was still reasoning."""
    low = (text or "").lower()
    return low.rfind("<think>") > low.rfind("</think>")


def visible_text(text: str | None) -> str:
    """The turn without its reasoning: closed `<think>` blocks, everything up to a `</think>`
    whose opening tag the template supplied, and everything after a `<think>` that never closes
    (a turn cut mid-thought has no visible part)."""
    out = _THINK.sub("", text or "")
    if re.search(r"</think>", out, re.IGNORECASE):
        out = _LONE_CLOSE.sub("", out)
    if in_open_think(out):
        out = out[:out.lower().rfind("<think>")]
    return out


def clean_answer(text: str | None) -> str:
    """An answer as a judge should read it: no reasoning, no tool call."""
    return _TOOL_CALL.sub("", visible_text(text)).strip()


def judge_text(text: str | None) -> str:
    """What a judge reads: the answer without tool calls. When nothing is left outside the
    reasoning (an answer written only inside the model's reasoning, as a forced answer at the
    step budget sometimes is), the reasoning without tool calls, so the judge still sees what the
    model concluded instead of an empty response."""
    answer = clean_answer(text)
    if answer:
        return answer
    return _TOOL_CALL.sub("", re.sub(r"</?think>", " ", text or "", flags=re.IGNORECASE)).strip()


__all__ = ["in_open_think", "visible_text", "clean_answer", "judge_text"]
