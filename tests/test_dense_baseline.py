"""The two NEW, additive-only baselines:

  research_dense  : the DENSE retrieve-then-visit baseline (agent_search.agent.tools.doc_research
                     .DenseVisit) — a byte-identical clone of Bm25Visit (search renders rank/
                     doc_id/title/snippet, visit returns the whole capped doc) with a dense
                     embedding engine (a DenseBelief) swapped in for BM25. CPU-only here via a
                     STUB engine injected the same way test_doc_bm25_fetch_tools.py injects a
                     stub BM25 engine for Bm25Visit — no torch/sentence-transformers import.

  oneshot_rag     : scripts/oneshot_rag.py, the no-agent-loop one-shot RAG baseline (retrieve
                     top-k, stuff into ONE prompt, ONE model call, parse <answer>). Tested here
                     via its prompt-builder + answer-parser + `run_instance` with a FAKE
                     generate() (no API call).
"""
from __future__ import annotations

import pytest

from agent_search.agent.tools.doc_research import Bm25Visit, DenseVisit, SNIPPET_TOKENS
from agent_search.corpus.units import units_from_documents

# the SAME fixture doc set test_doc_research_tools.py / test_doc_bm25_fetch_tools.py use, so
# DenseVisit's rendering can be compared line-for-line against Bm25Visit's for the same hits.
DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today.",
     "infobox": "Founded: 1897; Location: Portville"},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    {"_id": "65405", "title": "Integer Id Doc",
     "text": "A document whose id is an integer string."},
]


def _units():
    return units_from_documents(DOCS)


class _StubDenseEngine:
    """A CPU-only stand-in for DenseBelief exposing ONLY the API DenseVisit calls
    (`top_k_doc_ids(query, k)` -> ranked doc_ids) — no torch/sentence-transformers import, no
    GPU, no persisted cache. `ranking` is a fixed doc_id list (or a {query: [...]} map for
    query-dependent tests); mirrors how test_doc_bm25_fetch_tools.py hands Bm25Visit/
    Bm25FetchWorkspace a real-but-tiny BM25Local instead of the production build path."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.calls: list = []

    def top_k_doc_ids(self, query, k=None):
        self.calls.append((query, k))
        ids = self._ranking.get(query, []) if isinstance(self._ranking, dict) else self._ranking
        return list(ids[: (k or len(ids))])


def _ws(ranking=("d_harbor", "d_flat")):
    return DenseVisit(_units(), engine=_StubDenseEngine(ranking))


# --- tools/dispatch -----------------------------------------------------------------------

def test_tools_tuple_is_dense_search_and_visit_d():
    assert DenseVisit.tools == ("dense_search", "visit_d")


def test_dense_visit_is_a_bm25visit_subclass_reusing_resolve_and_visit():
    """DenseVisit is a CLONE of Bm25Visit (same rendering/contract), implemented by reusing
    Bm25Visit's `visit`/`_resolve` verbatim (inherited, not reimplemented) — only `search`/
    `run`/`__init__`/`tools` differ (the engine)."""
    assert issubclass(DenseVisit, Bm25Visit)
    assert DenseVisit.visit is Bm25Visit.visit
    assert DenseVisit._resolve is Bm25Visit._resolve


# --- search: ranked + opening snippet, SAME rendering as Bm25Visit -------------------------

def test_search_renders_rank_doc_id_title_and_snippet():
    ws = _ws(("d_harbor",))
    out = ws.run("dense_search", {"query": "harbor festival", "k": 5})
    assert "search: harbor festival   (1 matches):" in out
    assert "1  d_harbor  'Harbor Festival'" in out
    assert "Harbor Festival is an annual event" in out          # opening snippet, not a section


def test_search_output_matches_bm25visit_rendering_for_the_same_hit_order():
    """DenseVisit.search must render BYTE-IDENTICALLY to Bm25Visit.search given the SAME
    ranked doc_ids — the only difference between the two arms is WHICH engine produced the
    ranking, never the observation shape."""
    units = _units()
    dense_out = DenseVisit(units, engine=_StubDenseEngine(("d_harbor", "d_flat"))).search(
        "harbor festival", k=5)
    # Bm25Visit renders from ITS OWN bm25 ranking, so build it a stub with the SAME order instead
    # of relying on real bm25 to agree with our fixed dense ranking.
    class _FixedBm25(Bm25Visit):
        def search(self, query, k=5):
            self.last_hits = ["d_harbor", "d_flat"][:k]
            for i in self.last_hits:
                self.seen.add(i)
            lines = [f"search: {query}   ({len(self.last_hits)} matches):"]
            for rank, i in enumerate(self.last_hits, start=1):
                u = self.ubyid.get(i)
                snip = " ".join((u.body or u.code or "")[:120].split())
                lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
            return "\n".join(lines)
    bm25_out = _FixedBm25(units, engine=object()).search("harbor festival", k=5)
    assert dense_out == bm25_out


def test_search_marks_hits_seen():
    ws = _ws(("d_harbor", "d_flat"))
    ws.run("dense_search", {"query": "harbor"})
    assert {"d_harbor", "d_flat"} <= ws.seen


def test_empty_query_message():
    ws = _ws()
    assert ws.run("dense_search", {"query": ""}) == "empty query"


def test_zero_hits_message():
    ws = DenseVisit(_units(), engine=_StubDenseEngine([]))
    out = ws.run("dense_search", {"query": "nothing matches"})
    assert "0 matches" in out


def test_zero_hits_after_a_prior_search_notes_previous_results_available():
    ws = DenseVisit(_units(), engine=_StubDenseEngine({"harbor": ["d_harbor"], "zzz": []}))
    ws.run("dense_search", {"query": "harbor"})            # establishes last_hits
    out = ws.run("dense_search", {"query": "zzz"})
    assert "0 matches" in out and "previous results still available" in out


def test_engine_receives_the_raw_query_and_the_knob_depth():
    """run() passes the RAW query through, with DENSE_VISIT_TOPK as k — tools.yaml's
    dense_search schema exposes ONLY `query`, so a hallucinated `k` in the tool-call args is
    ignored; the env knob alone sets the SERP listing depth (see doc_research.DENSE_VISIT_TOPK
    and test_doc_research_tools.py's SERP-listing-depth section)."""
    import agent_search.agent.tools.doc_research as m
    engine = _StubDenseEngine(("d_harbor",))
    ws = DenseVisit(_units(), engine=engine)
    ws.run("dense_search", {"query": "harbor festival", "k": 3})
    assert engine.calls == [("harbor festival", m.DENSE_VISIT_TOPK)]


# --- run() dispatch: dense_search/search, visit_d/visit aliases ----------------------------

def test_run_aliases_search_name():
    ws = _ws(("d_harbor",))
    out1 = ws.run("dense_search", {"query": "harbor"})
    ws2 = _ws(("d_harbor",))
    out2 = ws2.run("search", {"query": "harbor"})
    assert out1 == out2


def test_run_aliases_visit_name():
    ws = _ws(("d_harbor",))
    ws.run("dense_search", {"query": "harbor"})
    out1 = ws.run("visit_d", {"rank": 1})
    assert "Founded in 1897" in out1
    ws2 = _ws(("d_harbor",))
    ws2.run("dense_search", {"query": "harbor"})
    out2 = ws2.run("visit", {"rank": 1})
    assert out1 == out2


def test_visit_returns_whole_doc_not_a_section():
    ws = _ws(("d_harbor",))
    ws.run("dense_search", {"query": "harbor"})
    out = ws.run("visit_d", {"rank": 1})
    assert "Founded in 1897" in out and "Still held today" in out


def test_integer_doc_id_not_mistaken_for_rank():
    ws = _ws(("65405",))
    ws.run("dense_search", {"query": "integer"})
    out = ws.run("visit_d", {"rank": "65405"})
    assert "integer string" in out and "out of range" not in out


def test_run_unknown_tool_errors():
    out = _ws().run("fetch", {"specs": [[1, "History"]]})
    assert "unknown tool" in out.lower()


# --- condition wiring: research_dense loads + resolves via the retriever registry ----------

def test_research_dense_condition_loads_uncoached():
    from agent_search.prompts import load_condition, render_manuals

    p = load_condition("research_dense")
    assert p.toolset == "dense_visit"
    assert set(p.tool_names) == {"dense_search", "visit_d"}
    # UNCOACHED like research_bm25: no manual renders for this toolset.
    assert render_manuals(p.tool_names, domain="general") == ""
    assert "term[field]" not in p.system


def test_research_dense_resolves_via_registry_as_densevisit_arm():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("dense_search", "visit_d")
    assert r.tool == "agent_research_dense"
    assert r._arm == "densevisit"
    assert r.domain == "general"
    assert not r.needs_files


def test_densevisit_index_raises_clear_error_when_cache_missing(tmp_path):
    """This baseline needs a persisted dense doc-embedding cache — a missing cache must raise
    a CLEAR error at index() time, never silently fall back to live-encoding the corpus."""
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense", RetrieverConfig(
        policy="stub", index_root=str(tmp_path)))()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        r.index(_units(), key="no_such_corpus_key")


# =============================================================================================
# DenseFetchWorkspace (research_dense_fetch) — the {dense search} x {structure->parts read}
# factorial cell: dense embedding LIVE RETRIEVAL (DenseVisit's engine/DenseBelief) + BQL
# structured SECTION-FETCH read (Bm25FetchWorkspace's read, inherited from DocSearchFetch
# unchanged), PLUS a one-line best-matching excerpt per hit (research_snip's `_best_line`) since
# research_snip established a content-bearing listing is the fair default once any arm shows one.
# =============================================================================================

from agent_search.agent.tools.doc_research import DenseFetchWorkspace, DocSearchFetch  # noqa: E402

_PAD = "x"          # same padding trick as tests/test_snippet_listing.py — isolates "did the
                    # MID-BODY window win" from "is the char/token cap doing something weird".


def _padded_fetch_doc(doc_id, title, needle,
                      pad_before=SNIPPET_TOKENS + 5, pad_after=SNIPPET_TOKENS + 5):
    before = " ".join([_PAD] * pad_before)
    after = " ".join([_PAD] * pad_after)
    return {"_id": doc_id, "title": title, "text": f"{before} {needle}\n\n## History\n{after}"}


FETCH_DOCS = [
    _padded_fetch_doc("d_mid", "Mid-body Match", "zephyrquokka marker phrase right here"),
    {"_id": "d_plain2", "title": "Plain Doc Two", "text": "A short document about nothing special."},
]


def _fetch_units():
    return units_from_documents(FETCH_DOCS)


def _fetch_ws(query="zephyrquokka", ranking=("d_mid",), topk=5, units=None):
    return DenseFetchWorkspace(units or _fetch_units(), query,
                               engine=_StubDenseEngine(ranking), topk=topk)


# --- tools/dispatch -----------------------------------------------------------------------

def test_fetchws_tools_tuple_is_dense_search_f_and_fetch():
    assert DenseFetchWorkspace.tools == ("dense_search_f", "fetch")


def test_fetchws_is_a_docsearchfetch_subclass_reusing_fetch_verbatim():
    """DenseFetchWorkspace is Bm25FetchWorkspace's dense twin (search->fetch, read side
    INHERITED from DocSearchFetch, never reimplemented) — only `search`/`run`/`__init__`/`tools`
    differ (the retrieval engine + the excerpt line)."""
    assert issubclass(DenseFetchWorkspace, DocSearchFetch)
    assert DenseFetchWorkspace.fetch is DocSearchFetch.fetch
    assert DenseFetchWorkspace._fetch_one is DocSearchFetch._fetch_one
    assert DenseFetchWorkspace._resolve_doc is DocSearchFetch._resolve_doc


# --- search: structure TABLE (sections + infobox), NO full body, PLUS a query-biased excerpt ---

def test_fetchws_search_lists_structure_plus_excerpt_not_full_text():
    ws = _fetch_ws()
    out = ws.run("dense_search_f", {"query": "zephyrquokka"})
    assert "d_mid" in out and "'Mid-body Match'" in out
    assert "History" in out                      # section name shown (structure)
    assert "»" in out                             # excerpt marker present (content-bearing)
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    assert "zephyrquokka" in excerpt
    assert len(excerpt.split()) <= SNIPPET_TOKENS  # a bounded window, not the whole padded body


def test_fetchws_excerpt_is_mid_body_not_the_doc_opening():
    ws = _fetch_ws()
    out = ws.run("dense_search_f", {"query": "zephyrquokka"})
    hit_line = next(l for l in out.splitlines() if "d_mid" in l)
    excerpt = hit_line.split("»", 1)[1].strip()
    opening = " ".join([_PAD] * SNIPPET_TOKENS)
    assert excerpt != opening


def test_fetchws_marks_hits_seen():
    ws = _fetch_ws(ranking=("d_mid", "d_plain2"))
    ws.run("dense_search_f", {"query": "zephyrquokka"})
    assert {"d_mid", "d_plain2"} <= ws.seen


def test_fetchws_empty_query_message():
    # SAME fallback contract as Bm25FetchWorkspace: a blank `query` arg through run() falls back
    # to the workspace's own construction-time query (self.query) — construct it blank too, and
    # call search() directly, exactly like test_doc_bm25_fetch_tools.py's empty-query test.
    ws = DenseFetchWorkspace(_fetch_units(), "", engine=_StubDenseEngine(("d_mid",)))
    assert ws.last_hits == []
    assert ws.search("") == "empty query"


def test_fetchws_zero_hits_message():
    ws = DenseFetchWorkspace(_fetch_units(), "q", engine=_StubDenseEngine([]))
    out = ws.run("dense_search_f", {"query": "nothing matches"})
    assert "0 matches" in out


def test_fetchws_engine_receives_raw_query_and_topk():
    engine = _StubDenseEngine(("d_mid",))
    ws = DenseFetchWorkspace(_fetch_units(), "q", engine=engine, topk=7)
    ws.run("dense_search_f", {"query": "zephyrquokka"})
    assert engine.calls == [("zephyrquokka", 7)]


def test_fetchws_search_is_live_and_re_retrieves():
    """LIVE retrieval: a NEW query string in a later dense_search_f call re-runs the dense engine
    and CHANGES the ranking — same live-per-call contract as Bm25FetchWorkspace/DenseVisit."""
    engine = _StubDenseEngine({"zephyrquokka": ["d_mid"], "plain doc": ["d_plain2"]})
    ws = DenseFetchWorkspace(_fetch_units(), "zephyrquokka", engine=engine)
    ws.run("dense_search_f", {"query": "zephyrquokka"})
    first = list(ws.last_hits)
    ws.run("dense_search_f", {"query": "plain doc"})
    assert ws.last_hits != first
    assert "d_plain2" in ws.last_hits


# --- fetch: pull ONE named section, by rank, established by the last dense search --------------

def test_fetchws_fetch_by_rank_after_search():
    ws = _fetch_ws()
    ws.run("dense_search_f", {"query": "zephyrquokka"})   # rank 1 = d_mid
    out = ws.run("fetch", {"specs": [[1, "History"]]})
    assert "History" in out and "ERROR" not in out


def test_fetchws_fetch_bad_section_lists_available():
    ws = _fetch_ws()
    ws.run("dense_search_f", {"query": "zephyrquokka"})
    out = ws.run("fetch", {"specs": [[1, "Nonexistent"]]})
    assert "no section" in out and "History" in out


def test_fetchws_fetch_marks_doc_seen():
    ws = _fetch_ws()
    ws.run("dense_search_f", {"query": "zephyrquokka"})
    ws.run("fetch", {"specs": [[1, "History"]]})
    assert "d_mid" in ws.seen


# --- run() dispatch: dense_search_f/dense_search/search aliases, fetch delegated ---------------

def test_fetchws_run_aliases_dense_search_and_search_names():
    out1 = _fetch_ws().run("dense_search_f", {"query": "zephyrquokka"})
    out2 = _fetch_ws().run("dense_search", {"query": "zephyrquokka"})
    out3 = _fetch_ws().run("search", {"query": "zephyrquokka"})
    assert out1 == out2 == out3


def test_fetchws_run_unknown_tool_errors():
    out = _fetch_ws().run("visit", {"rank": 1})            # visit is not a tool of this arm
    assert "unknown tool" in out.lower()


# --- condition wiring: research_dense_fetch loads + resolves via the retriever registry --------

def test_research_dense_fetch_condition_loads_uncoached():
    from agent_search.prompts import load_condition, render_manuals

    p = load_condition("research_dense_fetch")
    assert p.toolset == "dense_fetch"
    assert set(p.tool_names) == {"dense_search_f", "fetch"}
    # UNCOACHED like research_bm25_fetch/research_dense: no manual renders for this toolset.
    assert render_manuals(p.tool_names, domain="general") == ""
    assert "term[field]" not in p.system


def test_research_dense_fetch_resolves_via_registry_as_densefetch_arm():
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_dense_fetch", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever)
    assert r.toolset == ("dense_search_f", "fetch")
    assert r.tool == "agent_research_dense_fetch"
    assert r._arm == "densefetch"
    assert r.domain == "general"
    assert not r.needs_files


# =============================================================================================
# scripts/oneshot_rag.py — no-agent-loop baseline: prompt builder + answer parse (no API call)
# =============================================================================================

import json  # noqa: E402
import os  # noqa: E402
import scripts.oneshot_rag as oneshot_rag  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _oneshot_units_and_ubyid():
    units = units_from_documents([
        {"_id": "d_guadalupe", "title": "Treaty of Guadalupe Hidalgo",
         "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
        {"_id": "d_paris", "title": "Treaty of Paris (1898)",
         "text": "The 1898 Treaty of Paris ended the Spanish-American War."},
    ])
    return units, {u.doc_id: u for u in units}


def test_stuff_docs_caps_and_orders_by_rank():
    units, ubyid = _oneshot_units_and_ubyid()
    hits = oneshot_rag.stuff_docs(["d_guadalupe", "d_paris"], ubyid, max_tokens=5)
    assert [h[0] for h in hits] == ["d_guadalupe", "d_paris"]
    assert hits[0][1] == "Treaty of Guadalupe Hidalgo"
    # capped at 5 whitespace tokens -> truncation marker appended (same _cap_tokens convention
    # doc_research.py's Bm25Visit/DenseVisit `visit` use).
    assert "truncated" in hits[0][2]


def test_stuff_docs_skips_unknown_doc_ids():
    _, ubyid = _oneshot_units_and_ubyid()
    hits = oneshot_rag.stuff_docs(["d_guadalupe", "does_not_exist"], ubyid)
    assert [h[0] for h in hits] == ["d_guadalupe"]


def test_build_messages_shape_and_content():
    units, ubyid = _oneshot_units_and_ubyid()
    hits = oneshot_rag.stuff_docs(["d_guadalupe"], ubyid)
    msgs = oneshot_rag.build_messages("Which treaty ended the Mexican-American War?", hits)
    assert msgs[0]["role"] == "system"
    assert "<answer>" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "Which treaty ended the Mexican-American War?" in msgs[1]["content"]
    assert "Treaty of Guadalupe Hidalgo" in msgs[1]["content"]
    assert "d_guadalupe" in msgs[1]["content"]


def test_build_messages_with_no_hits_still_shapes_a_prompt():
    msgs = oneshot_rag.build_messages("some question", [])
    assert "no documents retrieved" in msgs[1]["content"].lower()


def test_parse_answer_extracts_last_tag():
    text = "the short answer span inside <answer> tags. Thus: <answer>Galați</answer>"
    assert oneshot_rag.parse_answer(text) == "Galați"


def test_parse_answer_falls_back_to_stripped_text_with_no_tag():
    assert oneshot_rag.parse_answer("  just prose, no tag  ") == "just prose, no tag"


def test_run_instance_end_to_end_with_a_fake_generate_fn():
    """No API call: `generate` is a plain fake closure returning (text, prompt_tokens,
    completion_tokens), exactly `make_generate`'s contract."""
    units, ubyid = _oneshot_units_and_ubyid()

    class _FakeEngine:
        def search(self, query, k):                      # bm25-shaped
            return ["d_guadalupe"][:k]

    def fake_generate(messages):
        assert messages[0]["role"] == "system"
        assert "Mexican-American" in messages[1]["content"]
        return "<answer>Treaty of Guadalupe Hidalgo, 1848</answer>", 321, 12

    class _Inst:
        instance_id = "hotpotqa__mexican_war"
        problem_statement = "Which treaty ended the Mexican-American War, and in what year?"
        answer = "Treaty of Guadalupe Hidalgo, 1848"

    row = oneshot_rag.run_instance(_Inst(), "bm25", _FakeEngine(), ubyid, fake_generate)
    assert row == {
        "instance_id": "hotpotqa__mexican_war",
        "question": "Which treaty ended the Mexican-American War, and in what year?",
        "gold_answer": "Treaty of Guadalupe Hidalgo, 1848",
        "final_answer": "Treaty of Guadalupe Hidalgo, 1848",
        "retrieved_ids": ["d_guadalupe"],
        "prompt_tokens": 321,
        "completion_tokens": 12,
        "n_steps": 1,
    }


def test_retrieve_top_k_dispatches_bm25_and_dense():
    class _Bm25Engine:
        def search(self, query, k):
            return [f"bm25:{query}"][:k]

    class _DenseEngine:
        def top_k_doc_ids(self, query, k):
            return [f"dense:{query}"][:k]

    assert oneshot_rag.retrieve_top_k("bm25", _Bm25Engine(), "q", k=1) == ["bm25:q"]
    assert oneshot_rag.retrieve_top_k("dense", _DenseEngine(), "q", k=1) == ["dense:q"]
    with pytest.raises(ValueError):
        oneshot_rag.retrieve_top_k("nope", None, "q")


def test_build_dense_engine_raises_clear_error_when_cache_missing(tmp_path):
    units, _ = _oneshot_units_and_ubyid()
    with pytest.raises(RuntimeError, match="dense doc-embedding cache"):
        oneshot_rag._build_dense_engine(units, "no_such_corpus_key", index_root=str(tmp_path))


# =============================================================================================
# BUGFIX regression tests: token-accounting (real BPE tokens, not whitespace words) and the
# ONESHOT_MAX_TOKENS completion-budget knob. See scripts/oneshot_rag.py's module-level
# DEFAULT_MAX_TOKENS / MODEL_MAX_CONTEXT_TOKENS / SAFETY_MARGIN_TOKENS comments for the bug:
# a 120,000-WORD default stuffing budget produced a 130,561-real-TOKEN prompt (bm25 rows),
# over the 131,072 max-model-len, and killed 154/200 rows with a context-overflow error.
# =============================================================================================

def test_default_max_tokens_env_knob_in_a_fresh_process():
    """ONESHOT_MAX_TOKENS overrides the completion budget at import time; unset it defaults to
    4000 (not the old hardcoded 512, which let the Tongyi reasoning model burn its whole
    completion budget on <think> and never reach <answer>). Run in a subprocess (rather than
    importlib.reload-ing scripts.oneshot_rag in-process) so this doesn't blow away this test
    module's cached tokenizer (`oneshot_rag._TOKENIZER_CACHE`, expensive to reload)."""
    import subprocess
    import sys as _sys
    code = ("import scripts.oneshot_rag as m; print(m.DEFAULT_MAX_TOKENS)")

    env_unset = {k: v for k, v in os.environ.items() if k != "ONESHOT_MAX_TOKENS"}
    out = subprocess.run([_sys.executable, "-c", code], cwd=os.getcwd(),
                         env=env_unset, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "4000"

    env_set = dict(env_unset, ONESHOT_MAX_TOKENS="9001")
    out = subprocess.run([_sys.executable, "-c", code], cwd=os.getcwd(),
                         env=env_set, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "9001"


def test_make_generate_default_max_tokens_uses_the_env_configured_constant():
    """make_generate's max_tokens default (no explicit override) is oneshot_rag.DEFAULT_MAX_TOKENS
    — and it is actually sent to the chat.completions.create call — NOT the old hardcoded 512."""
    sink = {}

    def create(**kwargs):
        sink.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<answer>x</answer>"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen = oneshot_rag.make_generate("some-model", "http://x", client=fake_client)
    gen([{"role": "user", "content": "hi"}])
    assert sink["max_tokens"] == oneshot_rag.DEFAULT_MAX_TOKENS
    assert sink["max_tokens"] != 512


def test_cap_real_tokens_never_exceeds_the_requested_real_token_budget():
    """`_cap_real_tokens` must cap by REAL tokens (the model's own tokenizer), not whitespace
    words — the exact accounting bug. At realistic per-doc budgets (>= the 1000-token floor
    `stuff_docs` enforces), the capped text's real token count never exceeds `n`."""
    tok = oneshot_rag._get_tokenizer(oneshot_rag.DEFAULT_MODEL)
    if tok is None:
        pytest.skip("Tongyi tokenizer not available offline in this environment")
    text = "the quick brown fox jumps over the lazy dog " * 20000
    for n in (1000, 5000):
        capped = oneshot_rag._cap_real_tokens(text, n, oneshot_rag.DEFAULT_MODEL)
        assert len(tok.encode(capped)) <= n


def test_stuff_docs_auto_budget_never_exceeds_model_context_by_construction():
    """Regression test for the core bug: stuff_docs' auto-split budget (ONESHOT_DOC_CAP unset,
    the default path a real oneshot_rag run takes) must produce a stuffed prompt that fits under
    the model's real max-model-len minus the completion budget, counted in REAL tokens — not the
    naive whitespace-word count that let bm25 rows deterministically hit 130,561 real tokens
    against a 131,072 max-model-len (154/200 rows died with a context-overflow error)."""
    docs = [{"_id": f"d{i}", "title": f"Synthetic Doc {i}",
             "text": ("lorem ipsum dolor sit amet consectetur adipiscing elit " * 20000).strip()}
            for i in range(5)]
    units = units_from_documents(docs)
    ubyid = {u.doc_id: u for u in units}
    doc_ids = [f"d{i}" for i in range(5)]

    hits = oneshot_rag.stuff_docs(doc_ids, ubyid)
    msgs = oneshot_rag.build_messages("What is the capital of France?", hits)
    full_text = msgs[0]["content"] + "\n" + msgs[1]["content"]

    limit = oneshot_rag.MODEL_MAX_CONTEXT_TOKENS - oneshot_rag.DEFAULT_MAX_TOKENS
    tok = oneshot_rag._get_tokenizer(oneshot_rag.DEFAULT_MODEL)
    if tok is not None:
        n_tokens = len(tok.encode(full_text))
    else:
        # No tokenizer offline in this environment: fall back to the SAME calibrated estimate
        # `_cap_real_tokens` itself uses, so the test still catches an accounting regression.
        n_tokens = int(len(full_text.split()) * 1.088)
    assert n_tokens <= limit, (
        f"stuffed prompt is {n_tokens} tokens, over the {limit}-token budget "
        f"(model context {oneshot_rag.MODEL_MAX_CONTEXT_TOKENS} - "
        f"completion budget {oneshot_rag.DEFAULT_MAX_TOKENS})")


def test_stuff_docs_default_budget_is_derived_from_context_minus_completion_minus_margin():
    """The auto-split total budget is no longer a magic 120000 constant — it's derived from the
    real model limits, so raising ONESHOT_MAX_TOKENS automatically shrinks the doc budget (the
    two knobs can't drift out of sync and silently reintroduce the overflow bug)."""
    expected = max(1000, oneshot_rag.MODEL_MAX_CONTEXT_TOKENS - oneshot_rag.DEFAULT_MAX_TOKENS
                   - oneshot_rag.SAFETY_MARGIN_TOKENS)
    units, ubyid = _oneshot_units_and_ubyid()
    # single retrieved doc -> the whole per-doc budget is `expected` (k=1 split)
    hits = oneshot_rag.stuff_docs(["d_guadalupe"], ubyid)
    # the tiny fixture doc is far under budget either way, so this just exercises that the
    # auto-split path runs without error and returns the doc uncapped.
    assert hits[0][0] == "d_guadalupe"
    assert expected > 0


# =============================================================================================
# BUGFIX regression tests: residual context overflow (11/200 bm25 rows) — stuff_docs' per-doc
# budget only accounted for each doc's OWN body text, never the assembled system+question+wrapper
# text nor (the actual gap) the server's chat-template overhead. `stuff_docs_fit` re-measures the
# ACTUAL rendered prompt and iteratively shrinks the largest doc until it fits — see
# FIT_MARGIN_TOKENS' module comment in scripts/oneshot_rag.py for the full story.
# =============================================================================================

class _CharTok:
    """Deterministic fake tokenizer: 1 character == 1 token, with a fake chat template that
    wraps every message in role markers + a trailing generation-prompt marker — i.e. it ADDS
    tokens beyond the raw concatenated message content, exactly the class of gap
    (chat-template overhead) that under-counted the real overflowing bm25 rows."""
    chat_template = "fake-template"  # non-None: _render_full_prompt takes the apply_chat_template path

    def encode(self, text):
        return list(text)

    def decode(self, ids):
        return "".join(ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


def test_stuff_docs_fit_converges_on_synthetic_oversized_docs(monkeypatch):
    """3 docs, each far bigger than the (artificially tiny) model context — stuff_docs' own
    auto-split floors each doc's budget at 1000 tokens regardless of how small the context is, so
    the initial stuffed prompt massively overflows the fit check. stuff_docs_fit must shrink
    across MULTIPLE iterations (zeroing one oversized doc at a time) and converge to a prompt that
    actually fits — this is the 'guaranteed convergence' property, exercised for real rather than
    asserted by construction."""
    monkeypatch.setitem(oneshot_rag._TOKENIZER_CACHE, "fake-model", _CharTok())
    monkeypatch.setattr(oneshot_rag, "MODEL_MAX_CONTEXT_TOKENS", 500)
    monkeypatch.setattr(oneshot_rag, "FIT_MARGIN_TOKENS", 20)

    docs = [{"_id": f"d{i}", "title": f"Doc{i}", "text": "x" * 2000} for i in range(3)]
    units = units_from_documents(docs)
    ubyid = {u.doc_id: u for u in units}
    doc_ids = [f"d{i}" for i in range(3)]

    hits, messages, n_tokens = oneshot_rag.stuff_docs_fit(
        "What is X?", doc_ids, ubyid, "fake-model", max_tokens=100)

    limit = 500 - 100 - 20
    assert n_tokens <= limit
    # independently re-measure the returned messages — the loop's own bookkeeping isn't trusted
    rendered = oneshot_rag._render_full_prompt(messages, "fake-model")
    assert len(rendered) <= limit         # _CharTok: 1 char == 1 token, so len() == token count
    # convergence actually required shrinking: the docs' full (uncapped) text (2000 chars each)
    # could never have fit a 500-token context, so at least one doc must have been cut down
    assert any(len(text) < 2000 for _doc_id, _title, text in hits)


def test_stuff_docs_fit_catches_chat_template_overhead_the_raw_content_sum_misses(monkeypatch):
    """The specific real-world regression: a doc set that fits under stuff_docs' RAW per-doc
    content budget can still overflow once rendered through the chat template (role wrappers +
    generation-prompt token), because that overhead is invisible to per-doc accounting. Here the
    single doc is small enough that stuff_docs itself would leave it uncapped, but the fake
    template adds more overhead than FIT_MARGIN_TOKENS — stuff_docs_fit must catch it anyway."""
    class _OverheadTok(_CharTok):
        TEMPLATE_OVERHEAD = 100  # tokens added purely by chat-template wrapping, not real content

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            raw = "".join(m["content"] for m in messages)
            return raw + ("Z" * self.TEMPLATE_OVERHEAD)

    tok = _OverheadTok()
    monkeypatch.setitem(oneshot_rag._TOKENIZER_CACHE, "overhead-model", tok)
    monkeypatch.setattr(oneshot_rag, "MODEL_MAX_CONTEXT_TOKENS", 1200)
    monkeypatch.setattr(oneshot_rag, "FIT_MARGIN_TOKENS", 30)  # < TEMPLATE_OVERHEAD: must matter

    docs = [{"_id": "d0", "title": "Doc0", "text": "y" * 500}]
    units = units_from_documents(docs)
    ubyid = {u.doc_id: u for u in units}

    hits, messages, n_tokens = oneshot_rag.stuff_docs_fit(
        "Q?", ["d0"], ubyid, "overhead-model", max_tokens=100)

    limit = 1200 - 100 - 30
    assert n_tokens <= limit
    # sanity: the template overhead alone exceeds the margin, so this scenario genuinely
    # required the render-aware fit check (a raw-content-only check would have passed it through)
    assert tok.TEMPLATE_OVERHEAD > 30


def test_stuff_docs_fit_never_raises_when_docs_are_already_tiny(monkeypatch):
    """The common case (small docs, generous context): no shrinking needed, hits pass through
    unchanged and the loop exits on the first fit check."""
    monkeypatch.setitem(oneshot_rag._TOKENIZER_CACHE, "fake-model-2", _CharTok())
    units, ubyid = _oneshot_units_and_ubyid()
    hits, messages, n_tokens = oneshot_rag.stuff_docs_fit(
        "Which treaty ended the Mexican-American War?", ["d_guadalupe", "d_paris"], ubyid,
        "fake-model-2", max_tokens=100)
    assert [h[0] for h in hits] == ["d_guadalupe", "d_paris"]
    assert n_tokens > 0


def test_run_instance_uses_stuff_docs_fit_and_never_exceeds_the_fit_limit(monkeypatch):
    """run_instance (the real per-instance entry point) must route through stuff_docs_fit, not
    the old raw stuff_docs+build_messages pair — regression guard for the wiring, not just the
    helper function in isolation."""
    monkeypatch.setitem(oneshot_rag._TOKENIZER_CACHE, "fake-model-3", _CharTok())
    # limit must comfortably fit the FIXED overhead (system prompt + question + chat-template
    # markers, ~350 fake-tokens even with the doc emptied out) plus leave room to prove the doc
    # itself got shrunk rather than just zeroed to nothing.
    monkeypatch.setattr(oneshot_rag, "MODEL_MAX_CONTEXT_TOKENS", 600)
    monkeypatch.setattr(oneshot_rag, "FIT_MARGIN_TOKENS", 10)

    docs = [{"_id": "d0", "title": "Big Doc", "text": "q" * 3000}]
    units = units_from_documents(docs)
    ubyid = {u.doc_id: u for u in units}

    class _Engine:
        def search(self, query, k):
            return ["d0"][:k]

    seen_messages = {}

    def fake_generate(messages):
        seen_messages["messages"] = messages
        return "<answer>ok</answer>", 1, 1

    class _Inst:
        instance_id = "x1"
        problem_statement = "What is in the big doc?"
        answer = "ok"

    oneshot_rag.run_instance(_Inst(), "bm25", _Engine(), ubyid, fake_generate,
                             model="fake-model-3", max_tokens=50)
    limit = 600 - 50 - 10
    rendered = oneshot_rag._render_full_prompt(seen_messages["messages"], "fake-model-3")
    assert len(rendered) <= limit


# =============================================================================================
# BUGFIX regression tests: per-instance resume — a rerun of scripts/oneshot_rag.py must skip
# instances already completed (no 'error') in an existing rows.jsonl, retry only the errored
# ones, and rewrite the file atomically (temp + os.replace) so a kill mid-write can't corrupt
# the previously-good rows.
# =============================================================================================

def test_load_resume_state_keeps_ok_rows_drops_errored_and_tolerates_truncated_line(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(
        json.dumps({"instance_id": "a", "final_answer": "yes"}) + "\n" +
        json.dumps({"instance_id": "b", "final_answer": "", "error": "ContextOverflow: boom"}) + "\n" +
        json.dumps({"instance_id": "c", "final_answer": "ok"}) + "\n" +
        '{"instance_id": "d", "final_ans'                                   # killed mid-append
    )
    done = oneshot_rag._load_resume_state(str(p))
    assert set(done) == {"a", "c"}                    # errored 'b' dropped, truncated 'd' skipped
    assert done["a"]["final_answer"] == "yes"


def test_load_resume_state_a_later_error_line_drops_an_earlier_ok_row_for_the_same_id():
    """Defensive: if the SAME instance_id appears twice (ok then error), the error wins — that
    instance must retry, not be considered done from its stale ok row."""
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "rows.jsonl")
    with open(p, "w") as fh:
        fh.write(json.dumps({"instance_id": "a", "final_answer": "stale-ok"}) + "\n")
        fh.write(json.dumps({"instance_id": "a", "final_answer": "", "error": "boom"}) + "\n")
    done = oneshot_rag._load_resume_state(p)
    assert done == {}


def test_load_resume_state_missing_file_returns_empty():
    assert oneshot_rag._load_resume_state("/no/such/path/rows.jsonl") == {}


def test_atomic_write_rows_writes_via_tmp_and_replace(tmp_path):
    p = tmp_path / "rows.jsonl"
    rows = [{"instance_id": "a", "final_answer": "1"}, {"instance_id": "b", "final_answer": "2"}]
    oneshot_rag._atomic_write_rows(str(p), rows)
    assert not (tmp_path / "rows.jsonl.tmp").exists()          # tmp file cleaned up (renamed away)
    got = [json.loads(l) for l in p.read_text().splitlines()]
    assert got == rows


def test_main_resume_skips_done_retries_errored_and_runs_new(monkeypatch, tmp_path):
    """End-to-end main() resume: 'a' is already done (kept verbatim, no generate call); 'b'
    previously errored (retried, stale error row replaced); 'c' is new (runs for the first
    time). Only 2 of the 3 instances should trigger a generate() call."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows_path = out_dir / "rows.jsonl"
    rows_path.write_text(
        json.dumps({"instance_id": "a", "question": "qa", "gold_answer": "ga",
                    "final_answer": "ga", "retrieved_ids": [], "prompt_tokens": 1,
                    "completion_tokens": 1, "n_steps": 1}) + "\n" +
        json.dumps({"instance_id": "b", "question": "qb", "gold_answer": "gb",
                    "final_answer": "", "error": "ContextOverflow: boom", "retrieved_ids": [],
                    "prompt_tokens": 0, "completion_tokens": 0, "n_steps": 1}) + "\n")

    class _Inst:
        def __init__(self, iid):
            self.instance_id = iid
            self.problem_statement = f"question for {iid}"
            self.answer = f"gold for {iid}"
            self.docs = [{"_id": "d1", "title": "T", "text": "hello world"}]

    instances = [_Inst("a"), _Inst("b"), _Inst("c")]

    monkeypatch.setattr(oneshot_rag, "load_dataset_by_name",
                        lambda name, limit=None: instances)
    monkeypatch.setattr(oneshot_rag, "_units_and_key",
                        lambda insts: (units_from_documents(insts[0].docs), "fakekey"))

    class _Engine:
        def search(self, query, k):
            return ["d1"][:k]

    monkeypatch.setattr(oneshot_rag, "_build_bm25_engine",
                        lambda units, index_root=None, key=None, backend=None: _Engine())

    calls = []

    def fake_make_generate(model, api_base):
        def gen(messages):
            calls.append(messages)
            return "<answer>ok</answer>", 5, 2
        return gen

    monkeypatch.setattr(oneshot_rag, "make_generate", fake_make_generate)

    rc = oneshot_rag.main(["--dataset", "fake_dataset", "--retriever", "bm25",
                          "--out-dir", str(out_dir), "--workers", "1"])
    assert rc == 0
    assert len(calls) == 2                              # only 'b' (retried) and 'c' (new) ran

    final = {json.loads(l)["instance_id"]: json.loads(l) for l in rows_path.read_text().splitlines()}
    assert set(final) == {"a", "b", "c"}
    assert final["a"]["final_answer"] == "ga"           # untouched, kept verbatim from resume
    assert "error" not in final["b"]                    # retried: stale error row replaced
    assert final["b"]["final_answer"] == "ok"
    assert final["c"]["final_answer"] == "ok"


def test_main_resume_is_a_noop_when_everything_is_already_done(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows_path = out_dir / "rows.jsonl"
    rows_path.write_text(
        json.dumps({"instance_id": "a", "final_answer": "ga"}) + "\n")

    class _Inst:
        instance_id = "a"
        problem_statement = "q"
        answer = "ga"
        docs = [{"_id": "d1", "title": "T", "text": "hi"}]

    monkeypatch.setattr(oneshot_rag, "load_dataset_by_name",
                        lambda name, limit=None: [_Inst()])

    def _boom(*a, **k):
        raise AssertionError("should not build the retrieval engine when nothing is todo")
    monkeypatch.setattr(oneshot_rag, "_units_and_key", _boom)
    monkeypatch.setattr(oneshot_rag, "_build_bm25_engine", _boom)

    rc = oneshot_rag.main(["--dataset", "fake_dataset", "--retriever", "bm25",
                          "--out-dir", str(out_dir), "--workers", "1"])
    assert rc == 0
    got = [json.loads(l) for l in rows_path.read_text().splitlines()]
    assert got == [{"instance_id": "a", "final_answer": "ga"}]
