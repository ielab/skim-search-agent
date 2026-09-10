"""In-process vLLM backend (the trained backbone, run on the GPU)."""
from __future__ import annotations

import threading
from typing import Callable

from .text import _STOP, _repair_open_tag, _truncate_at_tool_response
from .usage import _record_usage

# Headline agent model (open-weights MoE; confirm the exact HF id for your mirror).
# Lighter iteration default: "Qwen/Qwen2.5-Coder-7B-Instruct".
DEFAULT_MODEL = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"


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
