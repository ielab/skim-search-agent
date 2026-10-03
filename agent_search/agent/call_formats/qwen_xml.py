"""Qwen3.5 and Qwen3-Coder (`osunlp/QUEST-35B-RL` and other Qwen3.5 fine-tunes): an XML function
with one element per parameter.

    <tool_call><function=search><parameter=query>Copacabana ship</parameter></function></tool_call>
"""
from __future__ import annotations

import re
from typing import Optional

from agent_search.agent.call_formats.base import CallFormat, value_of

_FUNCTION = re.compile(r"<function=\s*([^>\s]+)\s*>", re.DOTALL)
_PARAM = re.compile(r"<parameter=\s*([^>\s]+)\s*>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)


class QwenXmlCallFormat(CallFormat):
    name = "qwen_xml"

    def parse_block(self, block: str) -> Optional[tuple[str, dict]]:
        found = _FUNCTION.search(block)
        if not found:
            return None
        body = block[found.end():]
        return found.group(1), {k: value_of(v) for k, v in _PARAM.findall(body)}
