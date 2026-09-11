"""agent_search.tokens: the single token ruler; there is no character-based limit."""
import pytest

from agent_search import tokens as T


def test_cap_tokens_is_whitespace_token_based():
    text = "one two three four five"
    assert T.cap_tokens(text, 5) == text
    assert T.cap_tokens(text, 2) == "one two" + T.TRUNCATED
    assert T.cap_tokens(text, 2, tail=" [cut]") == "one two [cut]"
    assert T.cap_tokens("", 3) == ""
    assert T.cap_tokens(None, 3) == ""


def test_cap_tokens_ignores_character_length():
    # a single long run of one character is far fewer model tokens than characters — proves
    # the cap follows the model-token ruler, not a hidden characters-divided-by-4 (or any
    # other fixed-ratio) proxy.
    text = "x" * 5000
    total = T.count_tokens(text)
    assert total < len(text) // 4
    assert T.cap_tokens(text, total) == text
    out = T.cap_tokens(text, total - 1)
    assert out != text
    assert out.endswith(T.TRUNCATED)
    assert T.count_tokens(out[: -len(T.TRUNCATED)]) == total - 1


def test_zero_budget_shows_only_the_marker():
    assert T.cap_tokens("some text", 0) == T.TRUNCATED.lstrip()


def test_count_tokens_never_uses_a_character_proxy(monkeypatch):
    # force the fallback ruler and check it counts whitespace tokens, not len/4. (truncate_tokens
    # is not exercised here under the forced fallback: it recurses into cap_tokens, which
    # recurses back into truncate_tokens, when no encoding is available — a pre-existing gap
    # in agent_search/tokens.py's fallback path, never hit in practice since tiktoken is always
    # installed in this environment; out of scope for a test-only port.)
    monkeypatch.setattr(T, "_ENC", None)
    monkeypatch.setattr(T, "_ENC_TRIED", True)
    assert T.count_tokens("a b c") == 3
    assert T.count_tokens("x" * 400) == 1
    assert T.ruler_name() == "whitespace"


def test_truncate_tokens_never_uses_a_character_proxy():
    # on the real (model-token) ruler: a long run of one repeated character is far fewer
    # tokens than a characters-divided-by-4 proxy would predict.
    text = "x" * 5000
    total = T.count_tokens(text)
    assert total < len(text) // 4
    assert T.truncate_tokens(text, total) == text
    out = T.truncate_tokens(text, total - 1)
    assert out.endswith(T.TRUNCATED)
    assert T.count_tokens(out[: -len(T.TRUNCATED)]) == total - 1


def test_truncate_tokens_prefix_fits_budget():
    text = " ".join(f"w{i}" for i in range(500))
    out = T.truncate_tokens(text, 50)
    body = out[: -len(T.TRUNCATED)] if out.endswith(T.TRUNCATED) else out
    assert T.count_tokens(body) <= 50
    assert out.endswith(T.TRUNCATED)
    assert T.truncate_tokens("short", 50) == "short"


@pytest.mark.parametrize("n", [1, 7, 33])
def test_cap_and_count_agree_on_model_ruler(n):
    text = " ".join(f"t{i}" for i in range(100))
    assert T.count_tokens(T.cap_tokens(text, n, tail="")) == n


def test_truncate_tokens_falls_back_to_whitespace_without_tiktoken(monkeypatch):
    """Without tiktoken the cut is by whitespace words and must not recurse."""
    from agent_search import tokens as T
    monkeypatch.setattr(T, "_encoding", lambda: None)
    assert T.truncate_tokens("a b c d e", 3, " …") == "a b c …"
    assert T.cap_tokens("a b c d e", 3, " …") == "a b c …"
    assert T.truncate_tokens("a b", 5) == "a b"
