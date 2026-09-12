"""Rendering the system prompt: a task template plus the tools' declarations and manuals.

The composition is the one the paper prompts were rendered with: the `<tools>` block is one
compact JSON function declaration per line, the manuals are the tools' markdown files in tool
order (deduplicated by file), and the template placeholders `{{tools}}`, `{{tool_manuals}}`,
`{{tool_names}}`, `{{tool_rules}}` and `{{toolset}}` are replaced textually. `{{tool_rules}}` is
generated from the declarations: one line per tool naming its exact argument structure, so a
task can state strict argument rules without naming the tools itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import yaml


def split_front_matter(text: str) -> tuple[dict, str]:
    """`---\\n...\\n---` YAML front matter, then the body (leading blank lines tolerated)."""
    text = text.lstrip("\n")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = yaml.safe_load(text[3:end]) or {}
            body = text[end + 4:].lstrip("\n")
            if not isinstance(fm, dict):
                raise ValueError("task front matter must be a mapping")
            return fm, body
    return {}, text


def render_declarations(declarations: Sequence[dict]) -> str:
    rendered = []
    for d in declarations:
        fn = {"type": "function",
              "function": {"name": d["name"], "description": d["description"],
                           "parameters": d.get("parameters") or {"type": "object", "properties": {}}}}
        rendered.append(json.dumps(fn, ensure_ascii=False, separators=(",", ":")))
    return "<tools>\n" + "\n".join(rendered) + "\n</tools>"


def render_manuals(manual_paths: Sequence[Optional[str]]) -> str:
    """The manuals in tool order, each file once."""
    blocks, seen = [], set()
    for p in manual_paths:
        if not p or p in seen:
            continue
        seen.add(p)
        blocks.append(Path(p).read_text(encoding="utf-8").rstrip())
    return "\n\n".join(blocks)


def render_tool_rules(declarations: Sequence[dict]) -> str:
    """One line per tool: its exact argument structure, from the declaration's JSON schema."""
    lines = []
    for d in declarations:
        params = d.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        parts = []
        for arg, spec in props.items():
            kind = (spec or {}).get("type", "value")
            parts.append(f'"{arg}": <{kind}{"" if arg in required else ", optional"}>')
        lines.append(f"- {d['name']}: {{{', '.join(parts)}}}")
    return "\n".join(lines)


def compose(body: str, declarations: Sequence[dict], manual_paths: Sequence[Optional[str]],
            toolset_name: Optional[str] = None) -> str:
    replacements = {
        "{{tools}}": render_declarations(declarations),
        "{{tool_manuals}}": render_manuals(manual_paths),
        "{{tool_names}}": ", ".join(d["name"] for d in declarations),
        "{{tool_rules}}": render_tool_rules(declarations),
        "{{toolset}}": str(toolset_name or ""),
    }
    out = body
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out.rstrip() + "\n"


__all__ = ["split_front_matter", "render_declarations", "render_manuals", "render_tool_rules", "compose"]
