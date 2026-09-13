"""OpenAI-compatible chat backend: chat models (gpt-4o*) and any OpenAI-compatible server
(a served vLLM, or the real OpenAI API when `base_url` is pointed at it)."""
from __future__ import annotations

import os
import sys
import time
from typing import Callable

from .retry import _with_retries
from .text import _STOP, _repair_open_tag, _truncate_at_tool_response
from .usage import _cached_tokens, _reasoning_tokens, _record_usage
from .vllm_local import DEFAULT_MODEL


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_bool(name: str):
    v = (os.environ.get(name) or "").strip().lower()
    return None if v in ("", "null", "none") else v in ("1", "true", "yes", "on")


def openai_compat_generate(model: str = DEFAULT_MODEL, *,
                           base_url: str = "http://localhost:8000/v1",
                           api_key: str | None = None, client=None,
                           max_tokens: int | None = None, temperature: float = 0.6,
                           seed: int | None = 42,   # fixed default for reproducibility
                           top_p: float | None = None, presence_penalty: float | None = None,
                           top_k: int | None = None, thinking: bool | None = None,
                           stop: list[str] | None = None,
                           ) -> Callable[[str], str]:
    """Call an OpenAI-compatible chat endpoint (vLLM server or API). `client` is
    injectable for offline tests. Start a server with: `vllm serve <model>`.

    `api_key` defaults to the `OPENAI_API_KEY` env var, so pointing `base_url` at
    the real OpenAI API works for quick commercial-model testing with no other setup.
    It falls back to vLLM's placeholder `"EMPTY"` when that var is unset; a local vLLM
    server ignores the key entirely, so this default is a no-op for it.

    `seed` is passed through for reproducible sampling (vLLM/OpenAI honor it);
    leave it None to allow non-deterministic sampling.

    `max_tokens`, `top_p`, `presence_penalty`, `top_k` and `thinking` default to the environment
    knobs LLM_MAX_TOKENS (4000), LLM_TOP_P (0.95), LLM_PRESENCE_PENALTY (1.1), LLM_TOP_K (unset)
    and LLM_THINKING (unset), the `model.*` keys of an experiment file. `top_k` and `thinking` are
    vLLM extensions (`extra_body`: `top_k`, `chat_template_kwargs.enable_thinking`). The returned
    callable takes per-call overrides, `generate(messages, max_tokens=..., thinking=...)`, and
    exposes the last response's finish reason as `generate.last_finish_reason`."""
    max_tokens = _env_int("LLM_MAX_TOKENS", 4000) if max_tokens is None else max_tokens
    top_p = _env_float("LLM_TOP_P", 0.95) if top_p is None else top_p
    presence_penalty = _env_float("LLM_PRESENCE_PENALTY", 1.1) if presence_penalty is None else presence_penalty
    top_k = _env_int("LLM_TOP_K", None) if top_k is None else top_k
    thinking = _env_bool("LLM_THINKING") if thinking is None else thinking
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        timeout=float(os.environ.get("LLM_TIMEOUT_S", "600")), max_retries=0)

    def generate(prompt, **opts) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        extra = {}
        if top_k is not None:
            extra["top_k"] = top_k
        think = opts.get("thinking", thinking)
        if think is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        kw = dict(model=model, messages=messages,
                  max_tokens=int(opts.get("max_tokens") or max_tokens),
                  temperature=temperature, seed=seed, top_p=top_p,
                  presence_penalty=presence_penalty, stop=stop or _STOP)
        if extra:
            kw["extra_body"] = extra
        resp = None
        for attempt in range(max(1, _env_int("LLM_EMPTY_RETRIES", 10))):
            resp = _with_retries(lambda: client.chat.completions.create(**kw))
            u = getattr(resp, "usage", None)
            if u is not None:
                _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                              _cached_tokens(u), _reasoning_tokens(u))
            msg = resp.choices[0].message
            if (msg.content or "").strip():
                break
            # an empty turn: the model put everything into its reasoning channel, or produced
            # nothing. DIVER's Qwen3.5 client re-asks the same turn with backoff; so does this,
            # LLM_EMPTY_RETRIES times (10). Every attempt's usage is recorded above. The turn is
            # logged with what the reasoning channel held, so a systematic cause shows in the
            # shard log; when the call had thinking switched off, the retry alternates it back
            # on (a reasoning parser that expects a think block can file a think-less reply as
            # reasoning and leave the content empty).
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            print(f"[backbone] empty turn {attempt + 1}: finish={getattr(resp.choices[0], 'finish_reason', None)} "
                  f"thinking={extra.get('chat_template_kwargs')} reasoning_len={len(str(reasoning))} "
                  f"reasoning_head={str(reasoning)[:160]!r}", file=sys.stderr, flush=True)
            if "chat_template_kwargs" in extra:
                extra["chat_template_kwargs"]["enable_thinking"] = not extra["chat_template_kwargs"]["enable_thinking"]
            time.sleep(min(_env_float("LLM_RETRY_BASE_S", 1.0) * (2 ** attempt), 30.0))
        generate.last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
        return _truncate_at_tool_response(_repair_open_tag(resp.choices[0].message.content or ""))

    # Expose the underlying (client, model) as attributes on the closure. Setting these
    # attributes does not change how `generate(prompt)` itself behaves; it lets
    # `agent_search.agent.loop.run_episode`'s inline forced-answer elicitation
    # (`agent_search/agent/forced_answer.py`) reuse the same OpenAI-compatible client and model
    # the episode's own policy is already talking to, instead of opening a second connection or
    # threading client/model through every caller of `AgentPolicy`. In-process vLLM
    # (`vllm_generate`, no HTTP client) and any generate callable built without these attributes
    # has no inline elicitation available: `run_episode` checks `getattr(..., "client", None)`
    # and degrades to `elicitation="prefill_failed"` instead of crashing.
    generate.client = client
    generate.model = model
    generate.last_finish_reason = None
    return generate
