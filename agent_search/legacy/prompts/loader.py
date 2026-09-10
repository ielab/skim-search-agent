"""YAML-backed prompt composition: condition = task template x toolset.

A condition (conditions.yaml) binds a tool-agnostic task template (tasks/*.md,
front-matter + body with {{tools}} and {{tool_manuals}} placeholders) to a
toolset (tools.yaml). This loader renders the concrete ``<tools>`` block from the
shared tool registry and concatenates the per-tool manuals (skills/*.md) of any
toolset tools that declare one — so a tool's "manual" loads only when the tool is
in the set and the tool declares one (standard tools need none). Plain .md files
are still accepted for local prompt experiments, but conditions are canonical.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROMPT_ROOT = Path(__file__).resolve().parent
TOOL_REGISTRY = PROMPT_ROOT / "tools.yaml"
CONDITIONS = PROMPT_ROOT / "conditions.yaml"
TASKS_DIR = PROMPT_ROOT / "tasks"

# Runtime additions (plugins): merged over the YAML files by `_registry()`/`_conditions()`.
# See `register_tool`, `register_toolset`, `register_runtime_condition` below and
# `agent_search.legacy.prompts.registry.register_condition` (the public entry point).
_RUNTIME: dict[str, dict] = {"tools": {}, "toolsets": {}, "conditions": {}}


def register_tool(name: str, *, description: str, parameters: dict | None = None,
                  manual: str | dict | None = None) -> None:
    """Declare a tool the agent may call: its name, the description and JSON-schema
    ``parameters`` rendered into the prompt's ``<tools>`` block, and an optional ``manual``
    (a markdown file path, absolute or relative to the prompts package; or a
    ``{domain: path}`` mapping) appended to the system prompt when the tool is in the toolset."""
    spec: dict[str, Any] = {"description": description,
                            "parameters": parameters or {"type": "object", "properties": {}}}
    if manual:
        spec["manual"] = manual
    _RUNTIME["tools"][name] = spec


def register_toolset(name: str, tools: list[str] | tuple[str, ...]) -> None:
    """Name a tool combination (the agent's tool surface for a condition)."""
    _RUNTIME["toolsets"][name] = list(tools)


def register_runtime_condition(name: str, *, task: str, toolset: str) -> None:
    """Bind a task template (a name under tasks/ or a path to a .md file) to a toolset."""
    _RUNTIME["conditions"][name] = {"task": task, "toolset": toolset}


@dataclass(frozen=True)
class PromptProfile:
    name: str
    action: str
    domain: str
    path: str                      # the task template (.md) — the editable artifact
    toolset: str | None
    tool_names: tuple[str, ...]
    system: str
    message_format: str = "deepresearch_tool_call"
    description: str = ""
    task: str | None = None
    system_sha256: str = ""        # hash of the COMPOSED system (provenance)


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"prompt config must be a mapping: {path}")
    return data


def _registry() -> dict[str, Any]:
    reg = _read_yaml(TOOL_REGISTRY)
    if _RUNTIME["tools"] or _RUNTIME["toolsets"]:
        reg = dict(reg)
        reg["tools"] = {**(reg.get("tools") or {}), **_RUNTIME["tools"]}
        reg["toolsets"] = {**(reg.get("toolsets") or {}), **_RUNTIME["toolsets"]}
    return reg


def _conditions() -> dict[str, dict[str, str]]:
    data = _read_yaml(CONDITIONS)
    conds = data.get("conditions") or {}
    if not isinstance(conds, dict):
        raise ValueError("conditions.yaml must map name -> {task, toolset}")
    if _RUNTIME["conditions"]:
        conds = {**conds, **_RUNTIME["conditions"]}
    return conds


# --- task templates (front-matter + markdown body) --------------------------

def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse an optional leading `---\\n...\\n---` YAML front-matter block (leading blank
    lines are tolerated)."""
    text = text.lstrip("\n")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = yaml.safe_load(text[3:end]) or {}
            body = text[end + 4:].lstrip("\n")
            if not isinstance(fm, dict):
                raise ValueError("task front-matter must be a mapping")
            return fm, body
    return {}, text


def _task_path(task: str) -> Path:
    """A task template: a bare name resolves under the package's tasks/; a path to an
    existing .md file (a plugin's own template) is used as is."""
    p = Path(task)
    if p.suffix.lower() == ".md" and p.exists():
        return p
    return TASKS_DIR / f"{task}.md"


def load_task(task: str) -> tuple[dict[str, Any], str]:
    """Return (front_matter, body) for tasks/<task>.md."""
    return _split_front_matter(_task_path(task).read_text(encoding="utf-8"))


# --- toolset resolution -----------------------------------------------------

def _resolve_toolset(toolset: str, registry: dict[str, Any]) -> tuple[str, ...]:
    names = (registry.get("toolsets") or {}).get(toolset)
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"unknown toolset {toolset!r}")
    missing = [n for n in names if n not in (registry.get("tools") or {})]
    if missing:
        raise ValueError(f"unknown tool(s) {missing} in toolset {toolset!r}")
    return tuple(names)


def render_tools(tool_names: tuple[str, ...], registry: dict[str, Any] | None = None) -> str:
    registry = registry or _registry()
    tools = registry.get("tools") or {}
    rendered = []
    for name in tool_names:
        spec = tools[name]
        fn = {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        rendered.append(json.dumps(fn, ensure_ascii=False, separators=(",", ":")))
    return "<tools>\n" + "\n".join(rendered) + "\n</tools>"


def _manual_path_for(spec, domain: str) -> str | None:
    """Resolve a tool's `manual` to a file path for `domain`. A plain string means the
    same manual for every domain; a mapping `{domain: path}` selects per domain (a novel
    DSL like BQL ships a CODE manual and a DOCUMENT manual), falling back to `code` then
    any entry. None when the tool declares no manual."""
    if isinstance(spec, dict):
        return spec.get(domain) or spec.get("code") or next(iter(spec.values()), None)
    return spec or None


def render_manuals(tool_names: tuple[str, ...],
                   registry: dict[str, Any] | None = None,
                   domain: str = "code") -> str:
    """Concatenate the `manual` markdown of toolset tools that declare one, in toolset
    order, deduped, choosing each tool's domain-appropriate manual. "" if none."""
    registry = registry or _registry()
    tools = registry.get("tools") or {}
    blocks: list[str] = []
    seen: set[str] = set()
    for name in tool_names:
        rel = _manual_path_for((tools.get(name) or {}).get("manual"), domain)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        p = Path(rel)
        if not p.is_absolute():
            p = PROMPT_ROOT / p
        blocks.append(p.read_text(encoding="utf-8").rstrip())
    return "\n\n".join(blocks)


def _compose(body: str, tool_names: tuple[str, ...], toolset: str | None,
             registry: dict[str, Any], domain: str = "code") -> str:
    replacements = {
        "{{tools}}": render_tools(tool_names, registry),
        "{{tool_manuals}}": render_manuals(tool_names, registry, domain),
        "{{tool_names}}": ", ".join(tool_names),
        "{{toolset}}": str(toolset or ""),
    }
    out = body
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out.rstrip() + "\n"


# --- canonical composer: condition -> PromptProfile -------------------------

def load_condition(name: str, domain: str | None = None,
                   profile: str | None = None) -> PromptProfile:
    """Compose the system prompt for a condition = task template x toolset.

    `domain` is the task domain (code / general) — what the agent's ranking logic keys
    on, recorded as PromptProfile.domain. `profile` is the per-dataset FIELD profile that
    selects which per-tool manual variant renders (e.g. a STRUCTURED wiki corpus uses
    profile="wiki" to get the title/section/infobox BQL manual while domain stays
    "general"). It defaults to the domain, so flat corpora need not declare one."""
    conds = _conditions()
    if name not in conds:
        raise ValueError(f"unknown condition {name!r}; choose from {sorted(conds)}")
    binding = conds[name]
    task, toolset = binding["task"], binding["toolset"]
    fm, body = load_task(task)
    registry = _registry()
    tool_names = _resolve_toolset(toolset, registry)
    dom = domain or str(fm.get("domain") or "code")
    man = profile or dom                               # selects the per-tool manual variant
    system = _compose(body, tool_names, toolset, registry, man)
    return PromptProfile(
        name=name,
        action=name,
        domain=dom,
        path=str(_task_path(task)),
        toolset=toolset,
        tool_names=tool_names,
        system=system,
        message_format=str(fm.get("message_format") or "deepresearch_tool_call"),
        description=str(fm.get("description") or ""),
        task=task,
        system_sha256=hashlib.sha256(system.encode("utf-8")).hexdigest()[:16],
    )


def load_prompt_profile(path: str, profile: str | None = None) -> PromptProfile:
    """Backward-compatible entry. Accepts a condition NAME or a file path.

    - A bare condition name (no path separator, not an existing file) composes via
      load_condition.
    - A tasks/*.md path renders the task body WITHOUT a toolset (used only for
      direct template inspection).
    - Any other .md path is treated as a legacy raw prompt.

    `profile` (per-dataset field profile) is forwarded to manual selection; see
    load_condition. It only affects the condition-name path (the canonical one).
    """
    p = Path(path)
    if not p.exists() and "/" not in path and "\\" not in path:
        return load_condition(path, profile=profile)
    if p.suffix.lower() == ".md" and p.parent.name == "tasks":
        fm, body = _split_front_matter(p.read_text(encoding="utf-8"))
        system = _compose(body, (), None, _registry(), profile or str(fm.get("domain") or "code"))
        return PromptProfile(
            name=str(fm.get("name") or p.stem), action=str(fm.get("name") or p.stem),
            domain=str(fm.get("domain") or "code"), path=str(p), toolset=None,
            tool_names=(), system=system,
            message_format=str(fm.get("message_format") or "deepresearch_tool_call"),
            description=str(fm.get("description") or ""), task=p.stem,
            system_sha256=hashlib.sha256(system.encode("utf-8")).hexdigest()[:16],
        )
    text = p.read_text(encoding="utf-8")
    return PromptProfile(
        name=p.stem, action=p.stem, domain="code", path=str(p), toolset=None,
        tool_names=(), system=text, description="legacy markdown prompt",
        system_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])


def load_prompt_text(path: str, profile: str | None = None) -> str:
    return load_prompt_profile(path, profile).system
