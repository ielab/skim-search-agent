"""Tool-call text formats other than the JSON one (`agent.actions.parse_tool_call`).

`parse_alternate(text)` reads the run's `AGENT_TOOL_CALL_FORMAT` (`model.tool_call_format`) and
returns `(name, arguments)` or None. Unset, it returns None and the JSON parser stays the only one.
"""
from __future__ import annotations

import os
from typing import Optional

from agent_search.agent.call_formats.base import CallFormat
from agent_search.agent.call_formats.glm import GlmCallFormat
from agent_search.agent.call_formats.lenient_json import LenientJsonCallFormat
from agent_search.agent.call_formats.mcp import McpCallFormat
from agent_search.agent.call_formats.qwen_xml import QwenXmlCallFormat

FORMATS: dict[str, type[CallFormat]] = {c.name: c for c in (GlmCallFormat, LenientJsonCallFormat, McpCallFormat, QwenXmlCallFormat)}


def call_format(name: Optional[str]) -> Optional[CallFormat]:
    """The format called `name`; None for an empty name or `json` (the default parser)."""
    name = (name or "").strip().lower()
    if name in ("", "json"):
        return None
    if name not in FORMATS:
        raise ValueError(f"AGENT_TOOL_CALL_FORMAT={name!r}; choose from json, {', '.join(sorted(FORMATS))}")
    return FORMATS[name]()


def parse_alternate(text: Optional[str]) -> Optional[tuple[str, dict]]:
    fmt = call_format(os.environ.get("AGENT_TOOL_CALL_FORMAT"))
    return fmt.parse(text) if fmt else None


__all__ = ["CallFormat", "GlmCallFormat", "LenientJsonCallFormat", "McpCallFormat", "QwenXmlCallFormat", "FORMATS", "call_format", "parse_alternate"]
