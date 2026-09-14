"""Every paper condition renders exactly the system prompt pinned in prompt_pins.json.

The pins were first taken from the pre-0.3 YAML-backed loader (removed at commit 9dfa5b2 after a
parity check) and matched the paper's prompts byte for byte. Two deliberate changes moved the
fetch-based conditions since: the `fetch` declaration is flat (`rank` + `section`, no nested
list) and the BrowseComp manual describes the sectioned corpus. `research_bql_dense_snip_papermanual`
pins the paper's stale BrowseComp manual. Any other change to what the model sees fails here."""
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
