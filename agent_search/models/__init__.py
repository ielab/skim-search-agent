"""Model backend adapters used by the orchestrator.

`make_generate(model, backend=...)` is the single agent-backend entry: it routes an OpenAI
model to the OpenAI API, a Gemini model to Gemini's OpenAI-compatible endpoint, `backend="api"`
to a served OpenAI-compatible server, and `backend="vllm"` (default) to in-process vLLM on the
GPU. See `agent_search.models.backends` for the historical single-module home of this code;
each provider now lives in its own file (openai_chat, openai_reasoning, gemini, vllm_local).
"""
from __future__ import annotations

from typing import Callable

from .gemini import _GEMINI_BASE_URL, _GEMINI_PREFIXES, gemini_generate, is_gemini_model
from .openai_chat import openai_compat_generate
from .openai_reasoning import (
    _OPENAI_BASE_URL,
    _OPENAI_REASONING_PREFIXES,
    is_reasoning_model,
    openai_reasoning_generate,
)
from .retry import (
    _PARAM_ERROR_EXC_NAMES,
    _PARAM_ERROR_PHRASES,
    _TRANSIENT_EXC_NAMES,
    _is_param_error,
    _is_transient_error,
    _with_retries,
)
from .text import _STOP, _repair_open_tag, _truncate_at_tool_response
from .usage import (
    _cached_tokens,
    _reasoning_tokens,
    _record_usage,
    reset_usage,
    usage_events,
    usage_totals,
)
from .vllm_local import DEFAULT_MODEL, vllm_generate

# --- provider dispatch: run the WHOLE pipeline on an API key, no GPU/vLLM -----
# A "test anytime" path: pick an OpenAI model with --model and the pipeline talks to the
# OpenAI API; the trained backbone keeps its vLLM path. Routing is by model-name prefix
# (a small, extensible provider table — add a provider by adding a matcher + a builder),
# NOT hardcoded per model, so new OpenAI models and new providers plug in.

# model-name prefixes that identify the OpenAI API provider.
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt")


def is_openai_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(_OPENAI_PREFIXES)


def make_generate(model: str = DEFAULT_MODEL, *, backend: str = "vllm",
                  api_base: str = "http://localhost:8000/v1", tp: int = 1,
                  temperature: float = 0.6, seed: int | None = 42,
                  client=None, api_key: str | None = None) -> Callable[[str], str]:
    """The single agent-backend entry: route (backend, model) to the right generate callable.

    - An OpenAI model (gpt-*, o-series, chatgpt-*) ALWAYS routes to the OpenAI API
      (base_url=https://api.openai.com/v1, key from OPENAI_API_KEY), regardless of `backend` —
      so `--model gpt-4o-mini` + a key runs the whole pipeline with no cluster. Reasoning
      models (o-series/gpt-5*) use the reasoning path (max_completion_tokens + reasoning_effort);
      gpt-4o/4o-mini use the chat path (max_tokens/temperature).
    - A Gemini model (gemini-*) ALWAYS routes to Gemini's OpenAI-compatible endpoint (key from
      GEMINI_API_KEY), regardless of `backend` — same "test anytime" story as OpenAI above.
    - backend="api" routes to an OpenAI-COMPATIBLE server (a served vLLM) at `api_base`.
    - backend="vllm" (default) loads the model in-process on the GPU.

    `client` is injectable (offline tests). Extensible: add a provider by adding a matcher
    (is_<provider>_model) + a branch here.

    `api_key` (optional) is forwarded to whichever HTTP-client branch is chosen, overriding the
    env-var fallback — required by the live-demo server, where concurrent requests carry
    DIFFERENT users' keys (env vars are process-global). None preserves today's env fallback."""
    if is_openai_model(model):
        if is_reasoning_model(model):
            return openai_reasoning_generate(model=model, base_url=_OPENAI_BASE_URL,
                                             seed=seed, client=client, api_key=api_key)
        return openai_compat_generate(model=model, base_url=_OPENAI_BASE_URL,
                                      temperature=temperature, seed=seed, client=client,
                                      api_key=api_key)
    if is_gemini_model(model):
        return gemini_generate(model=model, base_url=_GEMINI_BASE_URL,
                               temperature=temperature, seed=seed, client=client,
                               api_key=api_key)
    if backend == "api":
        return openai_compat_generate(model=model, base_url=api_base,
                                      temperature=temperature, seed=seed, client=client,
                                      api_key=api_key)
    return vllm_generate(model=model, tensor_parallel_size=tp,
                         temperature=temperature, seed=seed)
