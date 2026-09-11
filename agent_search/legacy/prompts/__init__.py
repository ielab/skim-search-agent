"""Pre-0.3 prompt-profile machinery: conditions.yaml bound a task template to a toolset and
rendered them into one system prompt. Kept so the parity tests can compare against it.
`agent_search.tasks` (prompt.md plus task.py) and each tool's manual.md replace this."""

from .loader import (PromptProfile, load_condition, load_prompt_profile,
                     load_prompt_text, render_manuals, render_tools)
from .registry import DOMAINS, PromptSpec, get_prompt_spec

__all__ = [
    "DOMAINS",
    "PromptProfile",
    "PromptSpec",
    "get_prompt_spec",
    "load_condition",
    "load_prompt_profile",
    "load_prompt_text",
    "render_manuals",
    "render_tools",
]
