"""The model contract: `generate(messages) -> text`.

Every provider in this package returns a callable of this shape from `make_generate`; a
user-provided provider is any callable of this shape.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """An LLM provider: ``generate(messages) -> text``.

    ``messages`` is an OpenAI-style chat list (``[{"role": ..., "content": ...}, ...]``); the
    return value is the raw generation. Tool calls live in that text (``<tool_call>...``), so
    any provider plugs in as one callable: a served vLLM, the OpenAI or Gemini APIs, or a
    lambda in a test. ``agent_search.agent.backbone.make_generate`` returns one of these.

    Optional attributes the loop reads with ``getattr`` (never required): ``client`` and
    ``model`` (an OpenAI-compatible client + model id) enable the forced-answer elicitation
    call on budget exhaustion; without them the episode simply ends with what it has."""

    def __call__(self, messages: list[dict]) -> str: ...


__all__ = ["Model"]
