"""Helpers for reading the run record (``rows.jsonl`` rows and episode metadata).

The run record is the library's one trace format (see "The run record" in ``docs/ARCHITECTURE.md``). These
helpers absorb the two shapes a row can have:

* current rows carry every tool observation, in full, on ``trajectory[i]["observation"]``;
* rows written before 2026-09 carried a character-capped copy there and the full text in a
  separate ``observations`` list.
"""
from __future__ import annotations

from typing import Any, Mapping


def observations_of(row: Mapping[str, Any]) -> list[str]:
    """Every tool observation the agent saw during the episode, in step order, uncapped."""
    obs = row.get("observations")
    if obs is not None:
        return [str(o) for o in obs]
    return [str(s.get("observation") or "") for s in (row.get("trajectory") or [])]


__all__ = ["observations_of"]
