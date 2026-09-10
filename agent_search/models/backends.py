"""Shim: real LLM backends used to live in this one module; they now live in
`agent_search.models` and its submodules (usage, retry, text, openai_chat,
openai_reasoning, gemini, vllm_local). This module re-exports every name the
old single-file version exposed, so existing imports keep working.
"""
from __future__ import annotations

from agent_search.models import (  # noqa: F401
    DEFAULT_MODEL,
    _OPENAI_BASE_URL,
    _OPENAI_PREFIXES,
    is_gemini_model,
    is_openai_model,
    is_reasoning_model,
    make_generate,
)
from agent_search.models.gemini import (  # noqa: F401
    _GEMINI_BASE_URL,
    _GEMINI_PREFIXES,
    gemini_generate,
)
from agent_search.models.openai_chat import openai_compat_generate  # noqa: F401
from agent_search.models.openai_reasoning import (  # noqa: F401
    _OPENAI_REASONING_PREFIXES,
    openai_reasoning_generate,
)
from agent_search.models.retry import (  # noqa: F401
    _PARAM_ERROR_EXC_NAMES,
    _PARAM_ERROR_PHRASES,
    _TRANSIENT_EXC_NAMES,
    _is_param_error,
    _is_transient_error,
    _with_retries,
)
from agent_search.models.text import (  # noqa: F401
    _STOP,
    _repair_open_tag,
    _truncate_at_tool_response,
)
from agent_search.models.usage import (  # noqa: F401
    _cached_tokens,
    _reasoning_tokens,
    _record_usage,
    _usage,
    reset_usage,
    usage_events,
    usage_totals,
)
from agent_search.models.vllm_local import vllm_generate  # noqa: F401
