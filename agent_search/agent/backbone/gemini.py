"""Gemini via its OpenAI-compatible endpoint. Uses the same client class as
`openai_compat_generate`, pointed at Google's server instead."""
from __future__ import annotations

import os
from typing import Callable

from .retry import _is_param_error, _with_retries
from .text import _STOP, _repair_open_tag, _truncate_at_tool_response
from .usage import _cached_tokens, _reasoning_tokens, _record_usage

# Gemini exposes an OpenAI-compatible endpoint, so it reuses `openai_compat_generate`-style
# calls (via `gemini_generate` below) with this base_url and key. No separate SDK or client
# type is needed.
_GEMINI_PREFIXES = ("gemini",)
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def is_gemini_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(_GEMINI_PREFIXES)


def gemini_generate(model: str, *, base_url: str = _GEMINI_BASE_URL,
                    api_key: str | None = None, client=None,
                    max_tokens: int = 4000, temperature: float = 0.6,
                    seed: int | None = 42, top_p: float = 0.95,
                    presence_penalty: float = 1.1,
                    stop: list[str] | None = None) -> Callable[[str], str]:
    """Gemini via its OpenAI-compatible endpoint (base_url plus GEMINI_API_KEY). Uses the same
    client class as `openai_compat_generate`, pointed at Google's server. Reads GEMINI_API_KEY.

    Gemini's endpoint does not accept the full OpenAI param surface the agent loop sends by
    default: `seed` is an unknown field (400) on every Gemini model tried, and
    `presence_penalty`/`frequency_penalty` are rejected on at least the flash-lite tier
    ("Penalty is not enabled for <model>"). Rather than hardcode a per-model exception list,
    which would drift as Google adds or drops support, each call tries the full param set
    first and, only on a 400-shaped failure, retries once with the minimal safe set
    (temperature/max_tokens/top_p/stop, confirmed to work across the tried models). This keeps
    the common case identical to `openai_compat_generate` while turning a genuine param-support
    gap into a one-time retry instead of a crashed episode. A transient 429/5xx on the first
    attempt is retried unchanged by `_with_retries` and never triggers this fallback (see
    `_is_param_error`)."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("GEMINI_API_KEY", "EMPTY"),
                        timeout=float(os.environ.get("LLM_TIMEOUT_S", "600")), max_retries=0)

    def generate(prompt, **opts) -> str:   # per-call options are a served-model feature; ignored here
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        kw = dict(model=model, messages=messages, max_tokens=max_tokens,
                  temperature=temperature, top_p=top_p, stop=stop or _STOP)
        try:
            resp = _with_retries(lambda: client.chat.completions.create(
                seed=seed, presence_penalty=presence_penalty, **kw))
        except Exception as e:  # noqa: BLE001, re-raised below unless it's a param rejection
            if not _is_param_error(e):
                raise
            resp = _with_retries(lambda: client.chat.completions.create(**kw))
        u = getattr(resp, "usage", None)
        if u is not None:
            _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                          _cached_tokens(u), _reasoning_tokens(u))
        return _truncate_at_tool_response(_repair_open_tag(resp.choices[0].message.content or ""))

    return generate
