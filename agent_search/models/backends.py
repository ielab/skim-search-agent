"""Real LLM backends for `AgentPolicy(generate=...)` — cluster-served OR API, one entry.

`make_generate(model, backend=...)` is the single agent-backend entry: it routes by model
name + backend to the right callable, so the WHOLE pipeline runs either on the GPU cluster or
on an API key with no cluster:
  - an OpenAI model (`gpt-4o-mini`, `gpt-4o`, `gpt-5*`, `o1/o3/o4*`) -> the OpenAI API
    (`https://api.openai.com/v1`, key from `OPENAI_API_KEY`). This is the "test anytime" path:
    `--model gpt-4o-mini` + a key runs both arms end-to-end, no GPU. Reasoning models use
    `max_completion_tokens` + `reasoning_effort`; chat models (gpt-4o*) use `max_tokens`/sampling.
  - a Gemini model (`gemini-*`) -> Gemini's OpenAI-COMPATIBLE endpoint
    (`https://generativelanguage.googleapis.com/v1beta/openai/`, key from `GEMINI_API_KEY`),
    same "test anytime, no GPU" path as OpenAI. Gemini's endpoint rejects some sampling params
    (`seed`, `presence_penalty`, `frequency_penalty` — see `gemini_generate`); it retries once
    with a minimal param set so a param-support gap never crashes an episode.
  - `backend="api"` -> an OpenAI-COMPATIBLE server (a served vLLM) at `api_base`.
  - `backend="vllm"` (default) -> in-process vLLM on the GPU (the trained backbone).

The default model is **Tongyi DeepResearch** (the trained deep-research backbone). Every path
returns a `generate(prompt) -> str` callable that plugs straight into `AgentPolicy`. Provider
dispatch is a small extensible table (add a `is_<provider>_model` matcher + a branch), not a
hardcoded per-model list.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

# Headline agent model (open-weights MoE; confirm the exact HF id for your mirror).
# Lighter iteration default: "Qwen/Qwen2.5-Coder-7B-Instruct".
DEFAULT_MODEL = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"


# --- per-episode usage accounting --------------------------------------------
# Thread-local: each eval worker thread runs ONE instance's episode at a time, so
# reset_usage()/usage_totals() bracket an episode exactly (AgentRetriever does
# this). Both backends record (prompt_tokens, completion_tokens) per call.
_usage = threading.local()


def reset_usage() -> None:
    _usage.events = []


def _record_usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
                  reasoning_tokens: int = 0) -> None:
    if not hasattr(_usage, "events"):
        _usage.events = []
    # reasoning_tokens is the thinking SUBSET already inside completion_tokens (OpenAI reasoning
    # models bill it at the output rate) — kept as a 4th slot so cost analysis can itemize the
    # invisible generation without double-counting it (output cost stays = completion_tokens).
    _usage.events.append(
        (int(prompt_tokens or 0), int(completion_tokens or 0), int(cached_tokens or 0),
         int(reasoning_tokens or 0)))


def _cached_tokens(usage) -> int:
    """Cached (prefix-reused) input tokens the provider reports. OpenAI/Gemini put this in
    usage.prompt_tokens_details.cached_tokens; a served vLLM with prefix caching reports it the
    same way. A stateless chat API re-sends the WHOLE conversation each turn, so prompt_tokens
    re-counts it — but on a prefix-cache HIT those resent tokens are billed ~10x cheaper (and cost
    ~0 GPU), so effective input ≈ prompt_tokens - cached_input_tokens."""
    d = getattr(usage, "prompt_tokens_details", None)
    return int(getattr(d, "cached_tokens", 0) or 0) if d is not None else 0


def _reasoning_tokens(usage) -> int:
    """Hidden thinking tokens a reasoning model generated (o-series / gpt-5*): OpenAI reports them
    in usage.completion_tokens_details.reasoning_tokens, ALREADY counted inside completion_tokens.
    0 for chat models (no details / no reasoning) and for a served vLLM that emits no reasoning
    parser output (its thinking, if any, just lands in completion_tokens undifferentiated)."""
    d = getattr(usage, "completion_tokens_details", None)
    return int(getattr(d, "reasoning_tokens", 0) or 0) if d is not None else 0


def usage_events() -> list:
    """This episode's per-call (prompt_tokens, completion_tokens, cached_input_tokens), in order."""
    return list(getattr(_usage, "events", []))


def usage_totals() -> dict:
    events = getattr(_usage, "events", [])
    return {
        "llm_calls": len(events),
        # prompt_tokens SUMS the resent conversation across turns (a stateless chat API re-sends the
        # full history each call); cached_input_tokens is the subset that hit the provider's prefix
        # cache (billed ~10x cheaper, ~free GPU). Effective input ≈ prompt_tokens - cached_input_tokens.
        "prompt_tokens": sum(e[0] for e in events),
        "completion_tokens": sum(e[1] for e in events),
        "cached_input_tokens": sum((e[2] if len(e) > 2 else 0) for e in events),
        # thinking tokens (reasoning models), a SUBSET already inside completion_tokens above.
        "reasoning_tokens": sum((e[3] if len(e) > 3 else 0) for e in events),
    }


def _repair_open_tag(text: str) -> str:
    """Generation stops AT `</tool_call>`/`</answer>` (the stop string is removed
    from the output), so re-append the close tag for the parser's regex. This is
    what makes the stop-after-the-action fix safe: the Tongyi model is trained on
    web research, not BQL — on this out-of-distribution task it does NOT reliably
    stop after its tool call, so without an explicit stop it rambled to
    max_tokens (10000) every turn = the 6-10 min/instance blow-up."""
    for open_t, close_t in (("<tool_call>", "</tool_call>"), ("<answer>", "</answer>")):
        if open_t in text and close_t not in text:
            text += close_t
    return text


# stop after the model emits ONE complete action (tool call or answer), not at
# max_tokens. Includes the Tongyi observation markers as belt-and-braces.
_STOP = ["</tool_call>", "</answer>", "\n<tool_response>", "<tool_response>"]


def _truncate_at_tool_response(text: str) -> str:
    """Official Tongyi loop's belt-and-braces: even with stop sequences set, cut
    anything past the first '<tool_response>' so a fabricated observation can
    never leak back into the conversation as if the tool had produced it."""
    pos = text.find("<tool_response>")
    return text[:pos] if pos != -1 else text


def vllm_generate(model: str = DEFAULT_MODEL, *, llm=None, tokenizer=None,
                  max_tokens: int = 4000, temperature: float = 0.6,
                  seed: int | None = 42,   # fixed default for reproducibility
                  top_p: float = 0.95, presence_penalty: float = 1.1,
                  stop: list[str] | None = None,
                  tensor_parallel_size: int = 1, **llm_kwargs) -> Callable[[str], str]:
    """In-process vLLM. `llm`/`tokenizer` are injectable for offline tests.

    `seed` makes sampling reproducible (vLLM seeds the per-request RNG); leave it
    None for the legacy non-deterministic behavior."""
    from vllm import SamplingParams
    if llm is None:
        from vllm import LLM
        llm = LLM(model=model, trust_remote_code=True,
                  tensor_parallel_size=tensor_parallel_size, **llm_kwargs)
    tok = tokenizer or llm.get_tokenizer()
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        top_p=top_p,
        presence_penalty=presence_penalty,
        stop=stop or _STOP,
    )
    lock = threading.Lock()   # sync LLM.generate is not thread-safe; --workers>1
                              # shares this callable. (Use --backend api + a vLLM
                              # server for true concurrency/continuous batching.)

    def generate(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        with lock:
            out = llm.generate([text], params)
        completion = out[0].outputs[0]
        _record_usage(len(out[0].prompt_token_ids or []), len(completion.token_ids or []))
        return _truncate_at_tool_response(_repair_open_tag(completion.text))

    return generate


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
        client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def generate(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop=stop or _STOP)
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


# --- provider dispatch: run the WHOLE pipeline on an API key, no GPU/vLLM -----
# A "test anytime" path: pick an OpenAI model with --model and the pipeline talks to the
# OpenAI API; the trained backbone keeps its vLLM path. Routing is by model-name prefix
# (a small, extensible provider table — add a provider by adding a matcher + a builder),
# NOT hardcoded per model, so new OpenAI models and new providers plug in.

# model-name prefixes that identify the OpenAI API provider.
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt")
# OpenAI REASONING models (o-series, gpt-5*): the /chat API rejects temperature/top_p/etc.
# and uses max_completion_tokens + reasoning_effort. Everything else is a normal chat model.
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Gemini exposes an OpenAI-COMPATIBLE endpoint, so it reuses `openai_compat_generate` (via
# `gemini_generate` below) with this base_url + key — no separate SDK/client type needed.
_GEMINI_PREFIXES = ("gemini",)
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def is_openai_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(_OPENAI_PREFIXES)


def is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(_OPENAI_REASONING_PREFIXES)


def is_gemini_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(_GEMINI_PREFIXES)


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
                        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def generate(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        kw = dict(model=model, messages=messages,
                  max_completion_tokens=max_completion_tokens, reasoning_effort=effort)
        # newer reasoning models (gpt-5*) REJECT `stop` (and some reject `seed`) with a 400
        # "Unsupported parameter". Try the full set, then retry without them — same defensive
        # pattern as `gemini_generate`. Dropping `stop` is safe: the loop parses the FIRST tool
        # call and `_truncate_at_tool_response` still cuts any fabricated observation post-hoc.
        try:
            resp = client.chat.completions.create(seed=seed, stop=_STOP, **kw)
        except Exception:  # noqa: BLE001 — unsupported param -> retry with the minimal reasoning set
            resp = client.chat.completions.create(**kw)
        u = getattr(resp, "usage", None)
        if u is not None:
            _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                          _cached_tokens(u), _reasoning_tokens(u))
        return _truncate_at_tool_response(_repair_open_tag(resp.choices[0].message.content or ""))

    return generate


def gemini_generate(model: str, *, base_url: str = _GEMINI_BASE_URL,
                    api_key: str | None = None, client=None,
                    max_tokens: int = 4000, temperature: float = 0.6,
                    seed: int | None = 42, top_p: float = 0.95,
                    presence_penalty: float = 1.1,
                    stop: list[str] | None = None) -> Callable[[str], str]:
    """Gemini via its OpenAI-COMPATIBLE endpoint (base_url + GEMINI_API_KEY) — same client class
    as `openai_compat_generate`, just pointed at Google's server. Reads GEMINI_API_KEY.

    Gemini's endpoint does NOT accept the full OpenAI param surface the agent loop sends by
    default: `seed` is an unknown field (400) on every Gemini model tried, and `presence_penalty`
    /`frequency_penalty` are rejected on at least the flash-lite tier ("Penalty is not enabled for
    <model>"). Rather than hardcode a per-model exception list (which would drift as Google adds/
    drops support), each call tries the FULL param set first and, only on a 400-shaped failure,
    retries once with the minimal safe set (temperature/max_tokens/top_p/stop — confirmed to work
    across the tried models). This keeps the common case identical to `openai_compat_generate`
    while making a param-support gap a one-time retry instead of a crashed episode."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("GEMINI_API_KEY", "EMPTY"))

    def generate(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        kw = dict(model=model, messages=messages, max_tokens=max_tokens,
                  temperature=temperature, top_p=top_p, stop=stop or _STOP)
        try:
            resp = client.chat.completions.create(seed=seed, presence_penalty=presence_penalty, **kw)
        except Exception:  # noqa: BLE001 — a rejected param (seed/presence_penalty/...): drop to the safe set
            resp = client.chat.completions.create(**kw)
        u = getattr(resp, "usage", None)
        if u is not None:
            _record_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                          _cached_tokens(u), _reasoning_tokens(u))
        return _truncate_at_tool_response(_repair_open_tag(resp.choices[0].message.content or ""))

    return generate


def make_generate(model: str = DEFAULT_MODEL, *, backend: str = "vllm",
                  api_base: str = "http://localhost:8000/v1", tp: int = 1,
                  temperature: float = 0.6, seed: int | None = 42,
                  client=None) -> Callable[[str], str]:
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
    (is_<provider>_model) + a branch here."""
    if is_openai_model(model):
        if is_reasoning_model(model):
            return openai_reasoning_generate(model=model, base_url=_OPENAI_BASE_URL,
                                             seed=seed, client=client)
        return openai_compat_generate(model=model, base_url=_OPENAI_BASE_URL,
                                      temperature=temperature, seed=seed, client=client)
    if is_gemini_model(model):
        return gemini_generate(model=model, base_url=_GEMINI_BASE_URL,
                               temperature=temperature, seed=seed, client=client)
    if backend == "api":
        return openai_compat_generate(model=model, base_url=api_base,
                                      temperature=temperature, seed=seed, client=client)
    return vllm_generate(model=model, tensor_parallel_size=tp,
                         temperature=temperature, seed=seed)
