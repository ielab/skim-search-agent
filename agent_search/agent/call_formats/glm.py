"""GLM-4.5 / 4.6 / 4.7 (`zai-org/GLM-4.7-Flash`): the function name, then key and value pairs.

    <tool_call>search<arg_key>query</arg_key><arg_value>Copacabana ship</arg_value></tool_call>
"""
from __future__ import annotations

import re
from typing import Optional

from agent_search.agent.call_formats.base import CallFormat, value_of

_PAIR = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


class GlmCallFormat(CallFormat):
    name = "glm"

    def parse_block(self, block: str) -> Optional[tuple[str, dict]]:
        name = block.split("<arg_key>", 1)[0].strip()
        if not name or "<" in name or "{" in name:
            return None
        return name, {k.strip(): value_of(v) for k, v in _PAIR.findall(block)}
