"""Generation-text helpers shared by every backend's `generate()` callable."""
from __future__ import annotations


def _repair_open_tag(text: str) -> str:
    """Generation stops AT `</tool_call>`/`</answer>` (the stop string is removed
    from the output), so re-append the close tag for the parser's regex. This is
    what makes the stop-after-the-action fix safe: the Tongyi model is trained on
    web research, not BQL — on this out-of-distribution task it does NOT reliably
    stop after its tool call, so without an explicit stop it rambled to
    max_tokens (10000) every turn = the 6-10 min/instance blow-up."""
    for open_t, close_t in (("<tool_call>", "</tool_call>"), ("<answer>", "</answer>")):
        if open_t in text and close_t not in text:
            text += close_t
    return text


# stop after the model emits ONE complete action (tool call or answer), not at
# max_tokens. Includes the Tongyi observation markers as belt-and-braces.
_STOP = ["</tool_call>", "</answer>", "\n<tool_response>", "<tool_response>"]


def _truncate_at_tool_response(text: str) -> str:
    """Official Tongyi loop's belt-and-braces: even with stop sequences set, cut
    anything past the first '<tool_response>' so a fabricated observation can
    never leak back into the conversation as if the tool had produced it."""
    pos = text.find("<tool_response>")
    return text[:pos] if pos != -1 else text
