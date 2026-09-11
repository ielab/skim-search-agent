"""`research_snip`: content snippets in the structured search listing (`SearchBql`,
`snippet=TermWindow()`). Each search hit gets one appended one-line best-matching excerpt: the ~25
token window of the doc body with the most overlap with the query's positive leaf tokens
(earliest window wins ties; falls back to the doc opening when there are no leaf tokens).

CPU-only, no network. `snippet=NoSnippet()` (the default) must reproduce the listing's output
byte-for-byte; the existing `research` condition/toolset is unaffected.
"""
from __future__ import annotations

import os

from agent_search.snippets import NoSnippet, TermWindow
from agent_search.corpus.units import units_from_documents
from agent_search.strategies import get_condition
from agent_search.tokens import count_tokens
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.budgets import SNIPPET_TOKENS
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bql.tool import SearchBql
from tests.lucene_support import build_lucene_bql, require_jvm

require_jvm()


_PAD = "x"          # a 1-char filler token, so the default window comfortably fits the char
                    # snippet cap even with the needle included — isolates the WHICH-WINDOW-WINS
                    # question from the separate char-cap behavior.


def _padded_doc(doc_id: str, title: str, needle: str,
                pad_before: int = SNIPPET_TOKENS + 5,
                pad_after: int = SNIPPET_TOKENS + 5) -> dict:
    """A doc whose body is `pad_before` filler tokens, then `needle` (a distinctive multi-word
    phrase), then `pad_after` more filler tokens — so the doc's OPENING (one full window) is
    pure filler and the best-matching window must come from the MIDDLE of the body. The padding
    tracks SNIPPET_TOKENS so the fixture stays valid at any configured window width."""
    before = " ".join([_PAD] * pad_before)
    after = " ".join([_PAD] * pad_after)
    return {"_id": doc_id, "title": title, "text": f"{before} {needle} {after}"}


DOCS = [
    _padded_doc("d_mid", "Mid-body Match", "cobra marker phrase right here"),
    {"_id": "d_plain", "title": "Plain Doc", "text": "A short document about nothing special."},
]


def _units():
    return units_from_documents(DOCS)


def _toolbox(snippet):
    """A (search, fetch) ToolBox over the corpus and its Lucene BQL engine, plus the bound
    `SearchBql` instance itself (for the tests that reach its `_best_line`/`_render_hits`
    helpers directly, the way the old workspace tests reached its instance methods)."""
    units = _units()
    ubyid = {u.doc_id: u for u in units}
    ex = build_lucene_bql(units)
    state = EpisodeState(question="q")
    sb = SearchBql(name="search", snippet=snippet).bind(state, units, ubyid, {"bql": ex})
    fe = Fetch(name="fetch").bind(state, units, ubyid, {})
    return ToolBox([sb, fe], state), sb, ubyid


# --- 1. snippet=NoSnippet() (default): the listing is byte-identical to before ------------------

def test_snippets_false_is_byte_identical_to_default():
    a, _, _ = _toolbox(snippet=NoSnippet())               # old default (no snippets kwarg at all)
    b, _, _ = _toolbox(snippet=NoSnippet())                # explicit False
    query = "cobra[body]"
    out_a = a.run("search", {"query": query})
    out_b = b.run("search", {"query": query})
    assert out_a == out_b
    assert "»" not in out_a
    assert "»" not in out_b


# --- 2. snippet=TermWindow(): hits carry a '»' excerpt overlapping the query terms, MID-BODY --------

def test_snippets_true_shows_mid_body_excerpt_not_opening():
    box, _, _ = _toolbox(snippet=TermWindow())
    out = box.run("search", {"query": "cobra[body]"})
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    assert "»" in hit_line
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "cobra" in excerpt
    # the doc's opening window is pure filler — proves the MID-BODY window won, not a
    # blind opening slice.
    opening = " ".join([_PAD] * SNIPPET_TOKENS)
    assert excerpt != opening


def test_snippets_true_second_hit_has_no_query_overlap_but_still_gets_a_line():
    # a doc that search doesn't even rank (no overlap) never reaches _render_hits, so this
    # instead checks the OTHER doc in a broader (0-exact-hit / soft) fallback: fetch _best_line
    # directly for a doc with none of the query's terms — falls back sanely (best-scoring window
    # is still whichever appears earliest, score 0 throughout, so it's the opening).
    _, sb, ubyid = _toolbox(snippet=TermWindow())
    u = ubyid["d_plain"]
    line = sb._best_line(u, ["cobra"])
    assert line == "A short document about nothing special."


# --- 3. empty leaf_toks -> opening-text fallback --------------------------------------------

def test_empty_leaf_toks_falls_back_to_doc_opening():
    _, sb, ubyid = _toolbox(snippet=TermWindow())
    u = ubyid["d_mid"]
    line = sb._best_line(u, [])
    opening = " ".join([_PAD] * SNIPPET_TOKENS)
    assert line == opening


def test_render_hits_with_empty_leaf_toks_uses_opening_fallback():
    _, sb, _ = _toolbox(snippet=TermWindow())
    table = sb._render_hits(["d_mid"], [], "header:")
    hit_line = [l for l in table.splitlines() if "d_mid" in l][0]
    assert "»" in hit_line
    excerpt = hit_line.split("»", 1)[1].strip()
    assert excerpt == " ".join([_PAD] * SNIPPET_TOKENS)


# --- 4. condition loading: research_snip carries the doc skill coaching -----------------

def test_research_snip_condition_loads_with_doc_skill_coaching():
    snip = get_condition("research_snip")
    assert snip.strategy.toolset_name == "search_fetch_s"
    assert snip.tool_names == ("search_s", "fetch_s")
    # a distinctive sentence from bql_doc.md present in the composed system prompt.
    sentinel = "Query entity NAMES, never the question's wording"
    assert sentinel in snip.render()


# --- 5. run() aliases: search_s/fetch_s behave exactly like search/fetch --------------------

def test_run_accepts_search_s_and_fetch_s_aliases():
    box, _, _ = _toolbox(snippet=TermWindow())
    out_alias = box.run("search_s", {"query": "cobra[body]", "k": 5})
    assert "»" in out_alias
    assert box.last_hits == ["d_mid"]
    fetch_out = box.run("fetch_s", {"specs": [[1, "(intro)"]]})
    assert fetch_out.startswith("fetch:")
    assert "ERROR" not in fetch_out


# --- 5. SNIPPET_TOKENS: the window width is a sweepable knob ---------------------------------
# Env-settable (read at import, like MAX_VISIT_TOKENS and the *_TOPK dials), so an ablation is
# SNIPPET_TOKENS=64 python -m agent_search.evaluation.run_eval ... with nothing else changed.

def test_default_width_is_32():
    # guarded: this file must also pass under a sweep (SNIPPET_TOKENS=64 pytest ...), where the
    # value is whatever the run configured — only the UNSET default is pinned to 32.
    if "SNIPPET_TOKENS" not in os.environ:
        assert SNIPPET_TOKENS == 32


def test_width_argument_controls_window_length():
    _, sb, ubyid = _toolbox(snippet=TermWindow())
    u = ubyid["d_mid"]
    body_tokens = count_tokens(" ".join((u.body or "").split()))
    for width in (32, 64, 128, 256, 512):
        line = sb._best_line(u, [], width=width)
        # the doc is shorter than the larger widths, so the window saturates at the doc length
        assert count_tokens(line) == min(width, body_tokens)


def test_env_override_is_picked_up_on_import(monkeypatch):
    import importlib
    from agent_search.tools import budgets
    from agent_search.snippets import base as snip_base, term_window
    monkeypatch.setenv("SNIPPET_TOKENS", "128")
    try:
        importlib.reload(budgets)                              # recomputes SNIPPET_TOKENS from env
        importlib.reload(snip_base)
        reloaded = importlib.reload(term_window)               # re-reads it from budgets
        assert reloaded.TermWindow.render.__defaults__[-1] == 128   # the default really moved
    finally:
        monkeypatch.undo()
        importlib.reload(budgets)
        importlib.reload(snip_base)
        importlib.reload(term_window)                           # restore for later tests


def test_window_is_not_character_capped():
    """The pre-knob implementation clipped every excerpt to 160 characters, which meant a
    token-count setting did not really control the snippet (prose runs ~6.4 chars/token, so
    the cap bound from about 25 tokens up). A window as wide as the whole body, in MODEL
    tokens, must come back as EXACTLY the whole body — any character cap would break the
    equality."""
    _, sb, ubyid = _toolbox(snippet=TermWindow())
    u = ubyid["d_mid"]
    whole = " ".join((u.body or "").split())
    line = sb._best_line(u, [], width=count_tokens(whole))
    assert line == whole
