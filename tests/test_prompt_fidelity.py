"""Every paper condition renders the exact system prompt it rendered before the restructuring.

The pins in prompt_pins.json were taken from the YAML-backed loader. The task, tool and
strategy packages (`agent_search.strategies.CONDITIONS`) must reproduce them byte for byte,
and so must the YAML loader while it is still shipped."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PINS = json.loads((Path(__file__).parent / "prompt_pins.json").read_text())


@pytest.mark.parametrize("name", sorted(PINS))
def test_condition_renders_the_pinned_prompt(name):
    from agent_search.strategies import CONDITIONS
    cond, _, profile = name.partition("@")
    assert cond in CONDITIONS, f"{cond} is not registered in agent_search.strategies"
    c = CONDITIONS[cond]
    assert c.system_sha256(profile or None) == PINS[name]["sha"], f"{name}: the rendered system prompt changed"
    if "tools" in PINS[name]:
        assert list(c.tool_names) == PINS[name]["tools"]


@pytest.mark.parametrize("name", sorted(PINS))
def test_yaml_loader_still_renders_the_pinned_prompt(name):
    from agent_search.legacy.prompts import load_condition
    cond, _, profile = name.partition("@")
    p = load_condition(cond, profile=profile or None)
    assert p.system_sha256 == PINS[name]["sha"], f"{name}: the YAML render changed"
