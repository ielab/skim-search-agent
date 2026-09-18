"""One token ruler for every length limit in the library.

SkimSearchAgent measures and caps text in **tokens, never characters**. Two rulers exist,
each with one job:

* **Model tokens** (``count_tokens`` / ``truncate_tokens`` / ``cap_tokens``): the ruler for
  every cap the agent experiences (``SNIPPET_TOKENS``, ``MAX_VISIT_TOKENS``,
  ``MAX_SECTION_TOKENS``, bash/read output caps, the context-history budget) and for cost and
  context accounting. It is tiktoken's ``o200k_base`` when tiktoken is installed, so every
  condition is measured and cut on one fixed scale.
* **Whitespace tokens** (``ws_tokens``): the fallback when tiktoken is not installed, and a
  cheap word count for code that only needs a rough length. Never a characters-divided-by-four
  proxy.

There is deliberately no character-based helper in this module, and none should be added
elsewhere: a character cap silently interacts with a token knob (a 160-character clip once
bounded a "25-token" snippet at ~25 tokens regardless of the knob), which is exactly the
kind of hidden confound a research harness must not carry.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

TRUNCATED = " …(truncated)"

_ENC = None          # tiktoken encoding, resolved lazily once per process
_ENC_TRIED = False
# tiktoken fetches the o200k_base file over the network on first use and caches it under the
# process's temp directory. A compute node has an empty temp directory and no network, so the
# fetch fails and, before 2026-09-18, the library fell back to whitespace tokens without a
# word: every cap in such a run was words, not model tokens. The cache now lives at a shared
# path (INDEX_ROOT/tiktoken_cache, seeded by `python -m agent_search.tokens --seed`), and a run
# that must be on the model-token ruler sets AGENT_SEARCH_REQUIRE_TIKTOKEN=1 to fail instead.
_CACHE_DIR = os.path.join(os.environ.get("INDEX_ROOT", "indexes"), "tiktoken_cache")


def ws_tokens(text: Optional[str]) -> list[str]:
    """Whitespace tokens of ``text``: the fallback ruler and a rough word count."""
    return (text or "").split()


def count_ws_tokens(text: Optional[str]) -> int:
    return len(ws_tokens(text))


def cap_tokens(text: Optional[str], n: int, tail: str = TRUNCATED) -> str:
    """Keep the first ``n`` model tokens of ``text`` (the same ruler as ``count_tokens``);
    append ``tail`` when anything was dropped. ``n <= 0`` returns ``tail`` alone for non-empty
    text (nothing shown) so that a zero budget is visibly zero rather than silently unbounded."""
    text = text or ""
    if n is None:
        return text
    if n <= 0:
        return tail.lstrip() if text else text
    return truncate_tokens(text, n, tail)


def _encoding():
    """tiktoken ``o200k_base`` if importable and its file is available, else ``None`` (resolved
    once). The shared cache directory is used when it holds the file; a failure is printed
    once, and raises when AGENT_SEARCH_REQUIRE_TIKTOKEN=1."""
    global _ENC, _ENC_TRIED
    if not _ENC_TRIED:
        _ENC_TRIED = True
        if "TIKTOKEN_CACHE_DIR" not in os.environ and os.path.isdir(_CACHE_DIR):
            os.environ["TIKTOKEN_CACHE_DIR"] = os.path.abspath(_CACHE_DIR)
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception as e:  # noqa: BLE001 - optional dependency; whitespace ruler is the fallback
            _ENC = None
            msg = (f"[tokens] the model-token ruler is unavailable ({type(e).__name__}: {str(e)[:120]}); "
                   f"falling back to WHITESPACE tokens. Seed the cache with `python -m agent_search.tokens "
                   f"--seed` on a node with network access (cache dir {_CACHE_DIR}).")
            if os.environ.get("AGENT_SEARCH_REQUIRE_TIKTOKEN", "").strip() not in ("", "0"):
                raise RuntimeError(msg) from e
            print(msg, file=sys.stderr, flush=True)
    return _ENC


def count_tokens(text: Optional[str]) -> int:
    """Model-token count on one fixed ruler (tiktoken ``o200k_base``); whitespace tokens when
    tiktoken is not installed. The same ruler measures an episode and cuts every budget."""
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
    if enc is None:                       # no tiktoken: the whitespace fallback, cut inline
        toks = text.split()
        if len(toks) <= max(n, 0):
            return text
        kept = " ".join(toks[:max(n, 0)])
        return kept + tail if kept else tail.lstrip()
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max(n, 0):
        return text
    kept = enc.decode(ids[:max(n, 0)])
    return kept + tail if kept else tail.lstrip()


def ruler_name() -> str:
    """Which measurement ruler is active: recorded in run provenance."""
    return "tiktoken:o200k_base" if _encoding() is not None else "whitespace"


def seed_cache(cache_dir: Optional[str] = None) -> str:
    """Copy tiktoken's o200k_base file into the shared cache directory (needs network access
    the first time, or an existing per-user cache). Returns the directory."""
    import shutil
    import tiktoken
    target = os.path.abspath(cache_dir or _CACHE_DIR)
    os.makedirs(target, exist_ok=True)
    tiktoken.get_encoding("o200k_base")                 # fetches into the process's cache dir
    src_dir = os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
    if not src_dir:
        import tempfile
        src_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if os.path.abspath(src_dir) != target:
        for name in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(target, name))
    # openai_harmony (gpt-oss serving) wants the same vocab as a plain o200k_base.tiktoken under
    # TIKTOKEN_ENCODINGS_BASE; write it next to the cache so offline nodes can serve gpt-oss
    enc_dir = os.path.join(os.path.dirname(target), "tiktoken_encodings")
    os.makedirs(enc_dir, exist_ok=True)
    for name in os.listdir(target):
        with open(os.path.join(target, name), "rb") as fh:
            head = fh.read(8)
        if head.startswith(b"IQ== 0"):                      # o200k_base's first line
            shutil.copy2(os.path.join(target, name), os.path.join(enc_dir, "o200k_base.tiktoken"))
    return target


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="the token ruler: report it, or seed the shared tiktoken cache")
    ap.add_argument("--seed", action="store_true", help="copy the o200k_base file into the shared cache dir")
    a = ap.parse_args()
    if a.seed:
        print(f"seeded {seed_cache()}")
    print(f"ruler: {ruler_name()}")


__all__ = ["TRUNCATED", "ws_tokens", "count_ws_tokens", "cap_tokens", "count_tokens",
           "truncate_tokens", "ruler_name", "seed_cache"]
