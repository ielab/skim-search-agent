"""The token ruler never falls back to whitespace silently (agent_search/tokens.py): a run that
sets AGENT_SEARCH_REQUIRE_TIKTOKEN=1 fails when the o200k_base file cannot be loaded, and the
shared cache directory under INDEX_ROOT is offered to tiktoken."""
import importlib
import os

import pytest


def _fresh_tokens(monkeypatch, fail: bool):
    import tiktoken
    if fail:
        monkeypatch.setattr(tiktoken, "get_encoding", lambda name: (_ for _ in ()).throw(RuntimeError("offline")))
    from agent_search import tokens
    importlib.reload(tokens)
    return tokens


def test_required_ruler_raises_when_the_encoding_cannot_load(monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_REQUIRE_TIKTOKEN", "1")
    tokens = _fresh_tokens(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="model-token ruler is unavailable"):
        tokens.count_tokens("a b c")


def test_optional_ruler_warns_and_falls_back(monkeypatch, capsys):
    monkeypatch.delenv("AGENT_SEARCH_REQUIRE_TIKTOKEN", raising=False)
    tokens = _fresh_tokens(monkeypatch, fail=True)
    assert tokens.count_tokens("a b c") == 3 and tokens.ruler_name() == "whitespace"
    assert "WHITESPACE" in capsys.readouterr().err


def test_shared_cache_dir_is_offered(monkeypatch, tmp_path):
    cache = tmp_path / "indexes" / "tiktoken_cache"
    cache.mkdir(parents=True)
    monkeypatch.setenv("INDEX_ROOT", str(tmp_path / "indexes"))
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    tokens = _fresh_tokens(monkeypatch, fail=False)
    tokens.count_tokens("x")
    assert os.environ.get("TIKTOKEN_CACHE_DIR") == str(cache)


@pytest.fixture(autouse=True)
def _restore_tokens_module():
    yield
    from agent_search import tokens
    importlib.reload(tokens)
