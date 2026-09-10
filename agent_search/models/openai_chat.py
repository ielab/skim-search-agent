"""OpenAI-compatible chat backend — chat models (gpt-4o*) and any OpenAI-compatible server
(a served vLLM, or the real OpenAI API when `base_url` is pointed at it)."""
from __future__ import annotations

import os
from typing import Callable

from .retry import _with_retries
from .text import _STOP, _repair_open_tag, _truncate_at_tool_response
from .usage import _cached_tokens, _reasoning_tokens, _record_usage
from .vllm_local import DEFAULT_MODEL


def openai_compat_generate(model: str = DEFAULT_MODEL, *,
                           base_url: str = "http://localhost:8000/v1",
                           api_key: str | None = None, client=None,
                           max_tokens: int = 4000, temperature: float = 0.6,
                           seed: int | None = 42,   # fixed default for reproducibility
                           top_p: float = 0.95, presence_penalty: float = 1.1,
                           stop: list[str] | None = None,
                           ) -> Callable[[str], str]:
    """Call an OpenAI-compatible chat endpoint (vLLM server or API). `client` is
    injectable for offline tests. Start a server with: `vllm serve <model>`.

    `api_key` defaults to the `OPENAI_API_KEY` env var (so pointing `base_url` at
    the real OpenAI API "just works" for quick commercial-model testing), falling
    back to vLLM's placeholder `"EMPTY"` when that var is unset (a local vLLM
    server ignores the key entirely, so this default is a no-op for it).

    `seed` is passed through for reproducible sampling (vLLM/OpenAI honor it);
    leave it None for the legacy non-deterministic behavior."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        timeout=float(os.environ.get("LLM_TIMEOUT_S", "600")), max_retries=0)

    def generate(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        resp = _with_retries(lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop or _STOP))
        u = getattr(resp, "usage", None)
        if u is not None:
            _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                          _cached_tokens(u), _reasoning_tokens(u))
        return _truncate_at_tool_response(_repair_open_tag(resp.choices[0].message.content or ""))

    # Expose the underlying (client, model) as attributes on the closure — a PURE ADDITION (function
    # attributes never change how `generate(prompt)` itself behaves) that lets
    # `agent_search.agent.loop.run_episode`'s inline forced-answer elicitation
    # (`agent_search/agent/forced_answer.py`) reuse the SAME OpenAI-compatible client + model the
    # episode's own policy is already talking to, rather than opening a second connection or (worse)
    # threading client/model through every caller of `AgentPolicy`. In-process vLLM (`vllm_generate`,
    # no HTTP client) and any generate callable built without these attributes simply has no inline
    # elicitation available — `run_episode` checks `getattr(..., "client", None)` and degrades to
    # `elicitation="prefill_failed"` (never crashes).
    generate.client = client
    generate.model = model
    return generate
