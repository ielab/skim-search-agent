"""`research_snip`: content snippets in the structured search listing (DocSearchFetch,
`snippets=True`). Each search hit gets one appended one-line best-matching excerpt: the ~25
token window of the doc body with the most overlap with the query's positive leaf tokens
(earliest window wins ties; falls back to the doc opening when there are no leaf tokens).

CPU-only, no network. `snippets=False` (the default) must reproduce `_render_hits`'s output
byte-for-byte; the existing `research` condition/toolset is unaffected.
"""
from __future__ import annotations

import os

from agent_search.legacy.workspaces.sieve import DocSearchFetch
from agent_search.legacy.workspaces.budgets import SNIPPET_TOKENS
from agent_search.corpus.units import units_from_documents
from agent_search.legacy.prompts import load_condition


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
    _padded_doc("d_mid", "Mid-body Match", "zephyrquokka marker phrase right here"),
    {"_id": "d_plain", "title": "Plain Doc", "text": "A short document about nothing special."},
]


def _units():
    return units_from_documents(DOCS)


# --- 1. snippets=False (default): _render_hits output byte-identical to before -------------

def test_snippets_false_is_byte_identical_to_default():
    a = DocSearchFetch(_units())                    # old default (no snippets kwarg at all)
    b = DocSearchFetch(_units(), snippets=False)     # explicit False
    query = "zephyrquokka[body]"
    out_a = a.search(query)
    out_b = b.search(query)
    assert out_a == out_b
    assert "»" not in out_a
    assert "»" not in out_b


# --- 2. snippets=True: hits carry a '»' excerpt overlapping the query terms, MID-BODY --------

def test_snippets_true_shows_mid_body_excerpt_not_opening():
    ws = DocSearchFetch(_units(), snippets=True)
    out = ws.search("zephyrquokka[body]")
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    assert "»" in hit_line
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "zephyrquokka" in excerpt
    # the doc's opening window is pure filler — proves the MID-BODY window won, not a
    # blind opening slice.
    opening = " ".join([_PAD] * SNIPPET_TOKENS)
    assert excerpt != opening


def test_snippets_true_second_hit_has_no_query_overlap_but_still_gets_a_line():
    # a doc that search doesn't even rank (no overlap) never reaches _render_hits, so this
    # instead checks the OTHER doc in a broader (0-exact-hit / soft) fallback: fetch _best_line
    # directly for a doc with none of the query's terms — falls back sanely (best-scoring window
    # is still whichever appears earliest, score 0 throughout, so it's the opening).
    ws = DocSearchFetch(_units(), snippets=True)
    u = ws.ubyid["d_plain"]
    line = ws._best_line(u, ["zephyrquokka"])
    assert line == "A short document about nothing special."


# --- 3. empty leaf_toks -> opening-text fallback --------------------------------------------

def test_empty_leaf_toks_falls_back_to_doc_opening():
    ws = DocSearchFetch(_units(), snippets=True)
    u = ws.ubyid["d_mid"]
    line = ws._best_line(u, [])
    opening = " ".join([_PAD] * SNIPPET_TOKENS)
    assert line == opening


def test_render_hits_with_empty_leaf_toks_uses_opening_fallback():
    ws = DocSearchFetch(_units(), snippets=True)
    table = ws._render_hits(["d_mid"], [], "header:")
    hit_line = [l for l in table.splitlines() if "d_mid" in l][0]
    assert "»" in hit_line
    excerpt = hit_line.split("»", 1)[1].strip()
    assert excerpt == " ".join([_PAD] * SNIPPET_TOKENS)


# --- 4. condition loading: research_snip carries the doc skill coaching -----------------

def test_research_snip_condition_loads_with_doc_skill_coaching():
    snip = load_condition("research_snip")
    assert snip.toolset == "search_fetch_s"
    assert snip.tool_names == ("search_s", "fetch_s")
    # a distinctive sentence from bql_doc.md present in the composed system prompt.
    sentinel = "Query entity NAMES, never the question's wording"
    assert sentinel in snip.system


# --- 5. run() aliases: search_s/fetch_s behave exactly like search/fetch --------------------

def test_run_accepts_search_s_and_fetch_s_aliases():
    ws = DocSearchFetch(_units(), snippets=True)
    out_alias = ws.run("search_s", {"query": "zephyrquokka[body]", "k": 5})
    assert "»" in out_alias
    assert ws.last_hits == ["d_mid"]
    fetch_out = ws.run("fetch_s", {"specs": [[1, "(intro)"]]})
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
    ws = DocSearchFetch(_units(), snippets=True)
    u = ws.ubyid["d_mid"]
    for width in (32, 64, 128, 256, 512):
        line = ws._best_line(u, [], width=width)
        # the doc is shorter than the larger widths, so the window saturates at the doc length
        assert len(line.split()) == min(width, len((u.body or "").split()))


def test_env_override_is_picked_up_on_import(monkeypatch):
    import importlib
    from agent_search.legacy.workspaces import budgets, common
    monkeypatch.setenv("SNIPPET_TOKENS", "128")
    try:
        importlib.reload(budgets)                              # recomputes SNIPPET_TOKENS from env
        reloaded = importlib.reload(common)                    # re-reads it from budgets
        assert reloaded.SNIPPET_TOKENS == 128
        assert reloaded.best_line.__defaults__[0] == 128       # the default really moved
    finally:
        monkeypatch.undo()
        importlib.reload(budgets)
        importlib.reload(common)                                # restore for later tests


def test_window_is_not_character_capped():
    """The pre-knob implementation clipped every excerpt to 160 characters, which meant a
    token-count setting did not really control the snippet (prose runs ~6.4 chars/token, so
    the cap bound from about 25 tokens up). A window as wide as the whole body must come back
    as EXACTLY the whole body — any character cap would break the equality."""
    ws = DocSearchFetch(_units(), snippets=True)
    u = ws.ubyid["d_mid"]
    whole = " ".join((u.body or "").split())
    line = ws._best_line(u, [], width=len(whole.split()))
    assert line == whole
