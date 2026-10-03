"""MiroThinker (`miromind-ai/MiroThinker-1.7-mini`) and other MiroFlow-trained agents: an MCP-style
call, the tool name then its arguments as a JSON object, with or without the `<use_mcp_tool>` and
`<server_name>` wrappers and the closing `</arguments>`.

    <tool_name>search</tool_name>
    <arguments>
    {"query": "Copacabana Belgian ship"}
    </arguments>
"""
from __future__ import annotations

import json
import re
from typing import Optional

from agent_search.agent.call_formats.base import _THINK, CallFormat

_NAME = re.compile(r"<tool_name>\s*([^<\s]+)\s*</tool_name>\s*(?:<arguments>)?", re.DOTALL)


class McpCallFormat(CallFormat):
    name = "mcp"

    def parse(self, text: Optional[str]) -> Optional[tuple[str, dict]]:
        if not text:
            return None
        body = _THINK.sub("", text)
        calls = list(_NAME.finditer(body))
        if not calls:
            return None
        last = calls[-1]
        rest = body[last.end():]
        start = rest.find("{")
        args = {}
        if start >= 0:
            try:
                data, _ = json.JSONDecoder().raw_decode(rest[start:])
                if isinstance(data, dict):
                    args = {str(k).strip(): v for k, v in data.items()}
            except json.JSONDecodeError:
                return None
        return last.group(1), args

    def parse_block(self, block: str) -> Optional[tuple[str, dict]]:
        return self.parse(block)
