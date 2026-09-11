"""Retry helper for transient failures at the model boundary.

Every `client.chat.completions.create(...)` call (this package and
`agent_search/agent/forced_answer.py`) goes through `_with_retries`, which applies
exponential backoff with jitter on transient failures (HTTP 429, 5xx, connection
errors, timeouts) and re-raises anything else immediately. Attempts and base delay
are env-overridable (`LLM_RETRY_ATTEMPTS`, `LLM_RETRY_BASE_S`).
"""
from __future__ import annotations

import os
import random
import time

# --- retry helper: transient failures at the model boundary --------------------------------
# Detected by exception class name, not isinstance, so this also recognizes a test double that
# mimics the openai package's exception shape without importing it. This matches how the openai
# SDK itself names these classes (openai.RateLimitError, .APIConnectionError, .APITimeoutError,
# .InternalServerError) regardless of which module actually defines the raised instance.
_TRANSIENT_EXC_NAMES = frozenset({
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
})
# 400-shaped "the server rejected one of our parameters" errors: the only case the param-probe
# fallbacks in openai_reasoning_generate/gemini_generate may treat as "drop a param and retry".
_PARAM_ERROR_EXC_NAMES = frozenset({"BadRequestError"})
_PARAM_ERROR_PHRASES = ("unsupported parameter", "unknown field", "not enabled")


def _is_transient_error(exc: BaseException) -> bool:
    """429 / 5xx / connection / timeout: worth an exponential-backoff retry. Anything else
    (auth errors, malformed requests, a genuine param rejection) must propagate immediately."""
    if type(exc).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status >= 500 or status == 429)


def _is_param_error(exc: BaseException) -> bool:
    """400-shaped 'unsupported parameter' error: the narrow case where re-issuing the call with
    a reduced parameter set is safe. A transient failure must never match this (see
    `_is_transient_error`, checked first by callers), so a 429 never silently changes sampling
    parameters."""
    if type(exc).__name__ in _PARAM_ERROR_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    msg = str(exc).lower()
    return any(p in msg for p in _PARAM_ERROR_PHRASES)


def _with_retries(fn, *, attempts: int | None = None, base: float | None = None):
    """Call `fn()`, retrying on a transient failure (see `_is_transient_error`) with exponential
    backoff plus jitter; anything else re-raises immediately on the first attempt. `attempts`
    and `base` default to the `LLM_RETRY_ATTEMPTS`/`LLM_RETRY_BASE_S` env vars (5 attempts,
    1.0s base)."""
    if attempts is None:
        attempts = int(os.environ.get("LLM_RETRY_ATTEMPTS", "5"))
    if base is None:
        base = float(os.environ.get("LLM_RETRY_BASE_S", "1.0"))
    last_exc: BaseException | None = None
    for attempt in range(max(attempts, 1)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001, re-raised immediately unless transient
            if not _is_transient_error(e):
                raise
            last_exc = e
            if attempt == attempts - 1:
                raise
            delay = base * (2 ** attempt) + random.uniform(0, base)
            time.sleep(delay)
    raise last_exc  # pragma: no cover, unreachable: the loop above always returns or raises
