"""Generation-text helpers shared by every backend's `generate()` callable."""
from __future__ import annotations


def _repair_open_tag(text: str) -> str:
    """Generation stops at `</tool_call>`/`</answer>` and the stop string itself is removed
    from the output, so re-append the close tag for the parser's regex. An explicit stop list
    matters here because the Tongyi model is trained on web research, not BQL: on this
    out-of-distribution task it does not reliably stop after its tool call. Without an explicit
    stop it ran on to max_tokens (10000) every turn, costing 6-10 minutes per instance."""
    for open_t, close_t in (("<tool_call>", "</tool_call>"), ("<answer>", "</answer>")):
        if open_t in text and close_t not in text:
            text += close_t
    return text


# Stop after the model emits one complete action (tool call or answer), not at
# max_tokens. Includes the Tongyi observation markers as an extra safeguard.
_STOP = ["</tool_call>", "</answer>", "\n<tool_response>", "<tool_response>"]


def _truncate_at_tool_response(text: str) -> str:
    """Extra safeguard from the official Tongyi loop: even with stop sequences set, cut
    anything past the first '<tool_response>' so a fabricated observation can
    never leak back into the conversation as if the tool had produced it."""
    pos = text.find("<tool_response>")
    return text[:pos] if pos != -1 else text
