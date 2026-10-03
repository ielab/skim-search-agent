"""JSON tool calls with the slips some backbones make often enough to lose a tenth of their turns
(`Qwen/Qwen3.6-27B` under a 1.5 presence penalty drops a repeated quote): a tool name missing its
opening or closing quote.

    <tool_call>{"name": search", "arguments": {"query": "hat shop"}}</tool_call>
"""
from __future__ import annotations

import re
from typing import Optional

from agent_search.agent.actions import parse_tool_call
from agent_search.agent.call_formats.base import CallFormat

_BARE_NAME = re.compile(r'("name"\s*:\s*)"?([A-Za-z_][\w.-]*)"?(\s*[,}])')


class LenientJsonCallFormat(CallFormat):
    name = "lenient_json"

    def parse(self, text: Optional[str]) -> Optional[tuple[str, dict]]:
        if not text:
            return None
        fixed = _BARE_NAME.sub(lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', text)
        return parse_tool_call(fixed) if fixed != text else None

    def parse_block(self, block: str) -> Optional[tuple[str, dict]]:
        return self.parse(block)
