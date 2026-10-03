"""A model's own tool-call text, for a backbone that writes calls in another shape than the
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` JSON that `agent.actions` reads.

A format is chosen per run with `model.tool_call_format` (`AGENT_TOOL_CALL_FORMAT`). The loop
tries it only when the JSON parser finds no call, so a backbone that does write JSON is read the
same way with or without it.
"""
from __future__ import annotations

import json
import re
from typing import Optional

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)


def value_of(text: str):
    """An argument value: JSON when it parses (a list of queries, a number), else the text."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


class CallFormat:
    """Subclass per format. `parse_block` reads the inside of the last <tool_call> block."""
    name: str = ""

    def parse(self, text: Optional[str]) -> Optional[tuple[str, dict]]:
        if not text:
            return None
        blocks = _BLOCK.findall(_THINK.sub("", text))
        return self.parse_block(blocks[-1]) if blocks else None

    def parse_block(self, block: str) -> Optional[tuple[str, dict]]:
        raise NotImplementedError
