"""agent_search.core.tokens — the single token ruler; there is no character-based limit."""
import pytest

from agent_search.core import tokens as T


def test_cap_tokens_is_whitespace_token_based():
    text = "one two three four five"
    assert T.cap_tokens(text, 5) == text
    assert T.cap_tokens(text, 2) == "one two" + T.TRUNCATED
    assert T.cap_tokens(text, 2, tail=" [cut]") == "one two [cut]"
    assert T.cap_tokens("", 3) == ""
    assert T.cap_tokens(None, 3) == ""


def test_cap_tokens_ignores_character_length():
    # 3 very long tokens fit a 3-token budget regardless of their character length
    text = " ".join("x" * 5000 for _ in range(3))
    assert T.cap_tokens(text, 3) == text
    assert T.cap_tokens(text, 1) == "x" * 5000 + T.TRUNCATED


def test_zero_budget_shows_only_the_marker():
    assert T.cap_tokens("some text", 0) == T.TRUNCATED.lstrip()


def test_count_tokens_never_uses_a_character_proxy(monkeypatch):
    # force the fallback ruler and check it counts whitespace tokens, not len/4
    monkeypatch.setattr(T, "_ENC", None)
    monkeypatch.setattr(T, "_ENC_TRIED", True)
    assert T.count_tokens("a b c") == 3
    assert T.count_tokens("x" * 400) == 1
    assert T.truncate_tokens("a b c d", 2) == "a b" + T.TRUNCATED
    assert T.ruler_name() == "whitespace"


def test_truncate_tokens_prefix_fits_budget():
    text = " ".join(f"w{i}" for i in range(500))
    out = T.truncate_tokens(text, 50)
    body = out[: -len(T.TRUNCATED)] if out.endswith(T.TRUNCATED) else out
    assert T.count_tokens(body) <= 50
    assert out.endswith(T.TRUNCATED)
    assert T.truncate_tokens("short", 50) == "short"


@pytest.mark.parametrize("n", [1, 7, 33])
def test_cap_and_count_agree_on_whitespace_ruler(n):
    text = " ".join(f"t{i}" for i in range(100))
    assert T.count_ws_tokens(T.cap_tokens(text, n, tail="")) == n
