"""One token ruler for every length limit in the library.

SkimSearchAgent measures and caps text in **tokens, never characters**. Two rulers exist,
each with one job:

* **Whitespace tokens** (``ws_tokens`` / ``cap_tokens``): the budget ruler. Every read cap
  the agent experiences (``SNIPPET_TOKENS``, ``MAX_VISIT_TOKENS``, ``MAX_SECTION_TOKENS``,
  bash/read output caps, the context-history budget) is expressed in whitespace tokens.
  This is the ruler the paper's experiments used, it is tokenizer-independent, and it is
  cheap enough to apply on every tool call.
* **Model tokens** (``count_tokens`` / ``truncate_tokens``): the measurement ruler.
  Cost and context accounting use tiktoken's ``o200k_base`` when it is installed so that
  every condition is measured on one fixed scale; when tiktoken is unavailable the whitespace
  ruler is used instead (never a characters-divided-by-four proxy).

There is deliberately no character-based helper in this module, and none should be added
elsewhere: a character cap silently interacts with a token knob (a 160-character clip once
bounded a "25-token" snippet at ~25 tokens regardless of the knob), which is exactly the
kind of hidden confound a research harness must not carry.
"""
from __future__ import annotations

from typing import Optional

TRUNCATED = " …(truncated)"

_ENC = None          # tiktoken encoding, resolved lazily once per process
_ENC_TRIED = False


def ws_tokens(text: Optional[str]) -> list[str]:
    """Whitespace tokens of ``text`` (the budget ruler)."""
    return (text or "").split()


def count_ws_tokens(text: Optional[str]) -> int:
    return len(ws_tokens(text))


def cap_tokens(text: Optional[str], n: int, tail: str = TRUNCATED) -> str:
    """Keep the first ``n`` whitespace tokens of ``text``; append ``tail`` when anything was
    dropped. ``n <= 0`` returns ``tail`` alone for non-empty text (nothing shown) so that a
    zero budget is visibly zero rather than silently unbounded."""
    text = text or ""
    if n is None:
        return text
    toks = text.split()
    if len(toks) <= max(n, 0):
        return text
    kept = " ".join(toks[:max(n, 0)])
    return kept + tail if kept else tail.lstrip()


def _encoding():
    """tiktoken ``o200k_base`` if importable, else ``None`` (resolved once)."""
    global _ENC, _ENC_TRIED
    if not _ENC_TRIED:
        _ENC_TRIED = True
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 - optional dependency; whitespace ruler is the fallback
            _ENC = None
    return _ENC


def count_tokens(text: Optional[str]) -> int:
    """Model-token count on one fixed ruler (tiktoken ``o200k_base``); whitespace tokens when
    tiktoken is not installed. Used for measurement (context size, tokens read), never as
    the agent-facing read budget."""
    enc = _encoding()
    if enc is None:
        return count_ws_tokens(text)
    return len(enc.encode(text or "", disallowed_special=()))


def truncate_tokens(text: Optional[str], n: int, tail: str = TRUNCATED) -> str:
    """Prefix of ``text`` that fits in ``n`` model tokens (tiktoken when available, else
    whitespace tokens), with ``tail`` appended when anything was dropped."""
    text = text or ""
    if n is None:
        return text
    enc = _encoding()
    if enc is None:
        return cap_tokens(text, n, tail)
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max(n, 0):
        return text
    kept = enc.decode(ids[:max(n, 0)])
    return kept + tail if kept else tail.lstrip()


def ruler_name() -> str:
    """Which measurement ruler is active: recorded in run provenance."""
    return "tiktoken:o200k_base" if _encoding() is not None else "whitespace"


__all__ = ["TRUNCATED", "ws_tokens", "count_ws_tokens", "cap_tokens", "count_tokens",
           "truncate_tokens", "ruler_name"]
