"""A reader model for tools that return what a model extracts from a page, never the page.

Several agents were trained with a read tool of this kind (OpenSeeker's `visit`, MiroThinker's
`scrape_and_extract_info`): the page and what the agent wants from it go to a second model, and
the agent sees that model's reply. The reader is an OpenAI-compatible endpoint named by
`VISIT_READER_API_BASE` and `VISIT_READER_MODEL`; a launcher points both at the run's own
served model unless told otherwise.
"""
from __future__ import annotations

import os
from typing import Optional


def reader_model() -> str:
    return os.environ.get("VISIT_READER_MODEL") or ""


def ask_reader(prompt: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """The reader's reply to one user message, without its reasoning. Raises when the endpoint
    is not set or the call fails; the calling tool turns that into its own failure text."""
    from openai import OpenAI
    from agent_search.agent.answer_text import visible_text
    base, model = os.environ.get("VISIT_READER_API_BASE"), reader_model()
    if not base or not model:
        raise ValueError("a read-with-a-model tool needs VISIT_READER_API_BASE and VISIT_READER_MODEL")
    client = OpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=300)
    kw = {"max_tokens": max_tokens} if max_tokens else {}
    reply = client.chat.completions.create(model=model, temperature=temperature,
                                           messages=[{"role": "user", "content": prompt}], **kw)
    return visible_text(reply.choices[0].message.content or "").strip()


__all__ = ["ask_reader", "reader_model"]
