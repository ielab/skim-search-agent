"""Prompt condition registry — discovered from conditions.yaml.

A condition binds a task template (tasks/*.md, which carries the domain) to a
toolset (tools.yaml). This registry reads conditions.yaml at import, so adding a
condition is one binding — no code change here. The agent harness turns each
condition `name` into a runnable `agent_<name>` (with `agent` aliased to `bql`).
`get_prompt_spec(name)` returns the binding; `spec.path` is the condition NAME,
which the loader composes via load_condition.
"""
from __future__ import annotations

from dataclasses import dataclass

from .loader import _conditions, load_task


@dataclass(frozen=True)
class PromptSpec:
    name: str
    action: str
    path: str          # the condition name (loader.load_prompt_profile composes it)
    domain: str = "code"
    task: str = ""
    toolset: str = ""


def _discover() -> dict[str, dict[str, PromptSpec]]:
    """conditions.yaml -> {domain: {condition_name: PromptSpec}}."""
    out: dict[str, dict[str, PromptSpec]] = {}
    for name, binding in _conditions().items():
        task, toolset = binding["task"], binding["toolset"]
        fm, _ = load_task(task)
        domain = str(fm.get("domain") or "code")
        spec = PromptSpec(name=name, action=name, path=name, domain=domain,
                          task=task, toolset=toolset)
        line = out.setdefault(domain, {})
        if name in line:
            raise ValueError(f"duplicate condition {name!r} in domain {domain!r}")
        line[name] = spec
    return out


PROMPTS: dict[str, dict[str, PromptSpec]] = _discover()
DOMAINS = tuple(PROMPTS)


def get_prompt_spec(name: str, domain: str = "code") -> PromptSpec:
    """Look up a condition by name. Tries `domain` first, then any domain (names
    are unique), so a caller need not know a condition's domain."""
    if name in PROMPTS.get(domain, {}):
        return PROMPTS[domain][name]
    for profs in PROMPTS.values():
        if name in profs:
            return profs[name]
    raise ValueError(f"unknown prompt profile {name!r}")
