"""Model backend adapters used by the orchestrator.

`make_generate(model, backend=...)` is the single agent-backend entry: it routes an OpenAI
model to the OpenAI API, a Gemini model to Gemini's OpenAI-compatible endpoint, `backend="api"`
to a served OpenAI-compatible server, and `backend="vllm"` (default) to in-process vLLM on the
GPU. Each provider lives in its own file (openai_chat, openai_reasoning, gemini, vllm_local);
this package re-exports every name a caller needs.
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

# --- provider dispatch: run the whole pipeline on an API key, no GPU or vLLM needed --------
# Picking an OpenAI model with --model routes the whole pipeline to the OpenAI API; the
# trained backbone keeps its vLLM path. Routing is by model-name prefix, a small provider
# table you extend by adding a matcher and a builder, not by a per-model lookup, so new
# OpenAI models and new providers plug in without touching the dispatch logic.

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

    - An OpenAI model (gpt-*, o-series, chatgpt-*) always routes to the OpenAI API
      (base_url=https://api.openai.com/v1, key from OPENAI_API_KEY), regardless of `backend`.
      This lets `--model gpt-4o-mini` plus a key run the whole pipeline with no cluster.
      Reasoning models (o-series/gpt-5*) use the reasoning path (max_completion_tokens plus
      reasoning_effort); gpt-4o/4o-mini use the chat path (max_tokens/temperature).
    - A Gemini model (gemini-*) always routes to Gemini's OpenAI-compatible endpoint (key from
      GEMINI_API_KEY), regardless of `backend`, for the same no-cluster testing as OpenAI above.
    - backend="api" routes to an OpenAI-compatible server (a served vLLM) at `api_base`.
    - backend="vllm" (default) loads the model in-process on the GPU.

    `client` is injectable for offline tests. To add a provider, add a matcher
    (is_<provider>_model) and a branch here.

    `api_key`, when given, is forwarded to whichever HTTP-client branch is chosen and overrides
    the env-var fallback. This is needed by the live-demo server, where concurrent requests carry
    different users' keys and an env var, being process-global, cannot. None keeps the env
    fallback."""
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
