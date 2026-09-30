"""`encoder_prefix`: the whitespace-bounded prefix an encoder cuts to the same tokens as the
whole page (`agent_search/retrievers/dense/base.py`). A whitespace tokenizer stands in for the
model's; the identity against the real Qwen3 tokenizer was checked on the BrowseComp-Plus corpus
(261 pages, including a 10 MB one) when the helper was written."""
from agent_search.retrievers.dense.base import MARGIN, encoder_prefix, encoder_prefixes


class _WordTokenizer:
    """One token per whitespace-delimited word; counts calls so the test sees the window grow."""

    def __init__(self):
        self.calls = 0

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        return {"input_ids": text.split()}


def test_short_text_is_returned_whole_without_tokenizing():
    tok = _WordTokenizer()
    assert encoder_prefix("a few words", tok, 512) == "a few words" and tok.calls == 0


def test_long_page_is_cut_at_whitespace_with_enough_tokens():
    page = " ".join(f"w{i}" for i in range(100_000))          # 100k words, ~690k characters
    tok = _WordTokenizer()
    prefix = encoder_prefix(page, tok, 512)
    words = prefix.split()
    assert len(words) >= 512 + MARGIN and words == page.split()[:len(words)]   # a true prefix, in whole words
    assert not prefix.endswith(" ") and len(prefix) <= 512 * 8
    assert tok.calls == 1


def test_window_grows_when_the_first_cut_is_too_short():
    page = ("x" * 5000 + " ") * 2000                          # 5,001-character words: 8 chars/token is far too few
    tok = _WordTokenizer()
    prefix = encoder_prefix(page, tok, 512)
    assert len(prefix.split()) >= 512 + MARGIN and tok.calls > 1


def test_text_without_enough_tokens_comes_back_whole():
    page = "x" * 100_000                                     # one word, no whitespace: nothing to cut
    assert encoder_prefix(page, _WordTokenizer(), 512) == page


def test_no_tokenizer_means_no_cut():
    page = "w " * 100_000
    assert encoder_prefix(page, None, 512) == page
    assert encoder_prefixes([page, "b"], None, 512) == [page, "b"]
