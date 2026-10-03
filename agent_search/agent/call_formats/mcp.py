"""MiroThinker (`miromind-ai/MiroThinker-1.7-mini`) and other MiroFlow-trained agents: an MCP-style
call, the tool name then its arguments as a JSON object, with or without the `<use_mcp_tool>` and
`<server_name>` wrappers and the closing `</arguments>`. MiroThinker also writes the call as a bare
`{"name": ..., "arguments": {...}}` object under a `<tool>` tag; the last such object is read too.

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

_CALL_END = re.compile(r"\s*(?:</>|</tool>|<tool>|</tool_call>)")
_NAME = re.compile(r"<tool_name>\s*([^<\s]+)\s*</tool_name>\s*(?:<arguments>)?", re.DOTALL)


class McpCallFormat(CallFormat):
    name = "mcp"

    def parse(self, text: Optional[str]) -> Optional[tuple[str, dict]]:
        if not text:
            return None
        body = _THINK.sub("", text)
        calls = list(_NAME.finditer(body))
        if not calls:
            return self._last_named_object(body)
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

    @staticmethod
    def _last_named_object(body: str) -> Optional[tuple[str, dict]]:
        """The last `{"name": ..., "arguments": {...}}` object in the text. MiroThinker often
        drops the object's closing brace before its ` </>`, so each call is cut at the next
        `</>` or `<tool>` and an unbalanced one is closed (`actions._repair_load`)."""
        from agent_search.agent.actions import _repair_load
        found = None
        starts = [m.start() for m in re.finditer(r"\{\s*\"name\"", body)]
        for i, st in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(body)
            seg = _CALL_END.split(body[st:end], 1)[0].strip()
            try:
                data = json.loads(seg)
            except json.JSONDecodeError:
                data = _repair_load(seg)
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                args = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
                found = (data["name"].strip(), {str(k).strip(): v for k, v in args.items()})
        return found

    def parse_block(self, block: str) -> Optional[tuple[str, dict]]:
        return self.parse(block)
