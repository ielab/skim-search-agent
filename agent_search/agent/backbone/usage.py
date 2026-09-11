"""Per-episode LLM usage accounting.

Thread-local: each eval worker thread runs one instance's episode at a time, so
reset_usage()/usage_totals() bracket an episode exactly (agent_search/evaluation/agent_runner.py
does this). Both backends record (prompt_tokens, completion_tokens) per call.
"""
from __future__ import annotations

import threading

_usage = threading.local()


def reset_usage() -> None:
    _usage.events = []


def _record_usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
                  reasoning_tokens: int = 0) -> None:
    if not hasattr(_usage, "events"):
        _usage.events = []
    # reasoning_tokens is the thinking subset already inside completion_tokens (OpenAI reasoning
    # models bill it at the output rate). It is kept as a 4th slot so cost analysis can itemize
    # the invisible generation without double-counting it (output cost stays = completion_tokens).
    _usage.events.append(
        (int(prompt_tokens or 0), int(completion_tokens or 0), int(cached_tokens or 0),
         int(reasoning_tokens or 0)))


def _cached_tokens(usage) -> int:
    """Cached (prefix-reused) input tokens the provider reports. OpenAI/Gemini put this in
    usage.prompt_tokens_details.cached_tokens; a served vLLM with prefix caching reports it the
    same way. A stateless chat API re-sends the whole conversation each turn, so prompt_tokens
    re-counts it, but on a prefix-cache hit those resent tokens are billed about 10x cheaper
    (and cost about 0 GPU), so effective input is roughly prompt_tokens minus
    cached_input_tokens."""
    d = getattr(usage, "prompt_tokens_details", None)
    return int(getattr(d, "cached_tokens", 0) or 0) if d is not None else 0


def _reasoning_tokens(usage) -> int:
    """Hidden thinking tokens a reasoning model generated (o-series / gpt-5*): OpenAI reports them
    in usage.completion_tokens_details.reasoning_tokens, already counted inside completion_tokens.
    This is 0 for chat models (no details or no reasoning) and for a served vLLM that emits no
    reasoning parser output; its thinking, if any, lands in completion_tokens undifferentiated."""
    d = getattr(usage, "completion_tokens_details", None)
    return int(getattr(d, "reasoning_tokens", 0) or 0) if d is not None else 0


def usage_events() -> list:
    """This episode's per-call (prompt_tokens, completion_tokens, cached_input_tokens), in order."""
    return list(getattr(_usage, "events", []))


def usage_totals() -> dict:
    events = getattr(_usage, "events", [])
    return {
        "llm_calls": len(events),
        # prompt_tokens sums the resent conversation across turns (a stateless chat API re-sends
        # the full history each call); cached_input_tokens is the subset that hit the provider's
        # prefix cache (billed about 10x cheaper, near-free GPU). Effective input is roughly
        # prompt_tokens minus cached_input_tokens.
        "prompt_tokens": sum(e[0] for e in events),
        "completion_tokens": sum(e[1] for e in events),
        "cached_input_tokens": sum((e[2] if len(e) > 2 else 0) for e in events),
        # thinking tokens (reasoning models), a subset already inside completion_tokens above.
        "reasoning_tokens": sum((e[3] if len(e) > 3 else 0) for e in events),
    }
