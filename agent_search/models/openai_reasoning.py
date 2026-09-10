"""OpenAI reasoning models (o-series / gpt-5*): no sampling params — token budget is
`max_completion_tokens` and thinking depth is `reasoning_effort`."""
from __future__ import annotations

import os
from typing import Callable

from .retry import _is_param_error, _with_retries
from .text import _STOP, _repair_open_tag, _truncate_at_tool_response
from .usage import _cached_tokens, _reasoning_tokens, _record_usage

# OpenAI REASONING models (o-series, gpt-5*): the /chat API rejects temperature/top_p/etc.
# and uses max_completion_tokens + reasoning_effort. Everything else is a normal chat model.
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_OPENAI_BASE_URL = "https://api.openai.com/v1"


def is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(_OPENAI_REASONING_PREFIXES)


def openai_reasoning_generate(model: str, *, base_url: str = _OPENAI_BASE_URL,
                              api_key: str | None = None, client=None,
                              max_completion_tokens: int = 8000,
                              reasoning_effort: str | None = None,
                              seed: int | None = 42) -> Callable[[str], str]:
    """OpenAI reasoning models (o-series / gpt-5*): NO sampling params (temperature/top_p/
    presence_penalty are rejected); token budget is `max_completion_tokens` and thinking depth
    is `reasoning_effort` (low/medium/high). Reads OPENAI_API_KEY. `client` injectable for tests."""
    effort = reasoning_effort or os.environ.get("REASONING_EFFORT", "low")
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url,
                        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        timeout=float(os.environ.get("LLM_TIMEOUT_S", "600")), max_retries=0)

    def generate(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        kw = dict(model=model, messages=messages,
                  max_completion_tokens=max_completion_tokens, reasoning_effort=effort)
        # newer reasoning models (gpt-5*) REJECT `stop` (and some reject `seed`) with a 400
        # "Unsupported parameter". Try the full set, then retry without them — same defensive
        # pattern as `gemini_generate`. Dropping `stop` is safe: the loop parses the FIRST tool
        # call and `_truncate_at_tool_response` still cuts any fabricated observation post-hoc.
        # Each attempt goes through `_with_retries` on its own, so a transient 429/5xx is retried
        # WITHOUT ever falling through to the reduced param set — only a genuine 400-shaped
        # "unsupported parameter" error triggers the fallback (see `_is_param_error`).
        try:
            resp = _with_retries(lambda: client.chat.completions.create(seed=seed, stop=_STOP, **kw))
        except Exception as e:  # noqa: BLE001 — re-raised below unless it's a param rejection
            if not _is_param_error(e):
                raise
            resp = _with_retries(lambda: client.chat.completions.create(**kw))
        u = getattr(resp, "usage", None)
        if u is not None:
            _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                          _cached_tokens(u), _reasoning_tokens(u))
        return _truncate_at_tool_response(_repair_open_tag(resp.choices[0].message.content or ""))

    return generate
