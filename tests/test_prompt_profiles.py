"""Prompt conditions = task template x toolset (loader composition).

THE method, both arms: the BQL field-tagged Boolean surface + search -> fetch.
  code  : task=taskfix,  toolset=search_fetch      (files -> functions, <fix>)
  docs  : task=research, toolset=research           (articles -> sections, <answer>)
          task=research, toolset=research_bm25      (retrieve-then-visit baseline)
Per-tool teaching lives in a tool's manual (skills/*.md) and renders only when the tool is in
the toolset; the field-tagged surface lowers to the SAME executor AST via
retrievers/structural/bql/surface.to_bql, so the parser/typechecker/executor are reused.
"""
import re

import pytest

from agent_search.prompts import (DOMAINS, get_prompt_spec, load_condition,
                                   load_prompt_profile, load_prompt_text,
                                   render_manuals)
from agent_search.prompts.loader import load_task
from agent_search.retrievers.structural.bql.parser import parse
from agent_search.retrievers.structural.bql.surface import to_bql
from agent_search.retrievers.structural.bql.types import check

CODE_CONDS = ("codefix", "codefix_grep")
GEN_CONDS = ("research", "research_bm25", "research_dci")
ALL_CONDS = CODE_CONDS + GEN_CONDS


def test_domains():
    assert set(DOMAINS) == {"code", "general"}


@pytest.mark.parametrize("cond", ALL_CONDS)
def test_every_condition_resolves_and_composes(cond):
    spec = get_prompt_spec(cond)
    prof = load_condition(cond)
    assert prof.name == cond
    assert prof.domain == spec.domain
    assert "<tools>" in prof.system
    assert prof.system.strip()
    assert load_prompt_profile(spec.path).system == prof.system


def test_condition_domains_are_correct():
    for c in CODE_CONDS:
        assert get_prompt_spec(c).domain == "code"
    for c in GEN_CONDS:
        assert get_prompt_spec(c).domain == "general"


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        get_prompt_spec("nonsense")
    with pytest.raises(ValueError):
        load_condition("nonsense")


def test_retired_conditions_are_gone():
    """The old prefix-surface / one-shot-search / localization conditions are retired for
    BOTH arms — they must no longer resolve.

    NOTE: `research_dense` is NOT in this list — that name was reclaimed (additively) for the
    NEW dense retrieve-then-visit baseline (toolset `dense_visit`, DenseVisit in
    doc_research.py; see tests/test_dense_baseline.py), the modern-RAG-default analogue of
    `research_bm25`. It is a live condition now, not a retired one."""
    for gone in ("bql", "grep", "bm25", "dense", "tools", "tools_bql",
                 "tools_nodense", "research_bql", "research_tools"):
        with pytest.raises(ValueError):
            get_prompt_spec(gone)


# --- tasks are tool-agnostic; manuals carry the per-tool teaching -----------

def test_tasks_are_tool_agnostic():
    """Task templates carry the {{tools}}/{{tool_manuals}} slots and must not TEACH the query
    language (that lives in the manuals) — a single illustrative example call is fine, but the
    operator reference / field table constructs must not appear in the task body."""
    for task in ("taskfix", "research"):
        _, body = load_task(task)
        assert "{{tools}}" in body and "{{tool_manuals}}" in body
        for marker in ("IN(def", "term[field]", "| `def` |", "| `title` |",
                       "## Field", "## Combine"):
            assert marker not in body, f"{task} still teaches the surface: {marker}"


def test_taskfix_output_contract():
    _, body = load_task("taskfix")
    assert "<fix>" in body and "file:" in body and "function:" in body and "change:" in body
    flat = " ".join(body.split())
    # the template is TOOL-AGNOSTIC (shared by codefix's search/fetch and codefix_grep's
    # grep/read) — a <fix> follows an investigative tool call, not a specific tool name.
    assert "<tool_call>" in flat and "SEEN the code" in flat


def test_research_output_contract():
    _, body = load_task("research")
    assert "<answer>" in body


# --- the field-tagged manuals: exist, attach only via `search`, and lower+parse ----

def _worked_examples(manual: str) -> list[str]:
    """The `term[field]` / boolean queries from the manual's '## Worked examples' section
    (backtick-quoted forms that look like queries, not fetch-spec JSON)."""
    section = manual[manual.index("Worked examples"):]
    cut = section.find("\n## ")
    if cut != -1:
        section = section[:cut]
    out = []
    for q in re.findall(r"`([^`]+)`", section):
        if "{" in q or '"specs"' in q:                   # skip fetch-spec JSON
            continue
        if "[" in q or any(op in q for op in (" AND ", " OR ", " NOT ")):
            out.append(q)
    return out


def test_code_manual_worked_examples_lower_and_typecheck():
    manual = render_manuals(("search",), domain="code")
    assert manual and "term[field]" in manual, "code `search` must declare the field-tagged manual"
    ex = _worked_examples(manual)
    assert ex, "no worked term[field] examples found in skills/bql_code.md"
    for q in ex:
        bql = to_bql(q, domain="code")
        r = parse(bql)
        assert r.ok, f"parse failed for {q!r} -> {bql!r}: {r.error}"
        assert check(r.expr).ok, f"type-check failed for {q!r} -> {bql!r}"


def test_doc_manual_worked_examples_lower_and_typecheck():
    manual = render_manuals(("search",), domain="general")
    assert manual and "term[field]" in manual, "doc `search` must declare the field-tagged manual"
    ex = _worked_examples(manual)
    assert ex, "no worked term[field] examples found in skills/bql_doc.md"
    for q in ex:
        bql = to_bql(q, domain="doc")
        r = parse(bql)
        assert r.ok, f"parse failed for {q!r} -> {bql!r}: {r.error}"
        assert check(r.expr).ok, f"type-check failed for {q!r} -> {bql!r}"


def test_code_and_doc_manuals_are_distinct_and_advertise_their_fields():
    code = render_manuals(("search",), domain="code")
    docs = render_manuals(("search",), domain="general")
    assert code and docs and code != docs
    for f in ("def", "call", "string", "comment", "sig", "file"):
        assert f"`{f}`" in code, f"code manual missing field {f!r}"
    assert "[def]" not in docs and "[call]" not in docs
    for f in ("title", "section", "body", "infobox"):
        assert f"`{f}`" in docs or f"[{f}]" in docs, f"doc manual missing field {f!r}"
    assert "[title]" not in code and "[infobox]" not in code


def test_only_search_family_tools_have_a_manual():
    """`search` (the method's field-tagged surface) carries a manual; so do its NEW,
    additive-only siblings `search_v2` (research_v2's BQL v2 — typed date ranges +
    coverage feedback), `search_s` (research_snip's excerpt-listing surface — SAME manual
    VALUES as `search`, just its own registry entry so a distinct tool name resolves one),
    `isearch` (research_indri's Indri graded query language), and ITS additive-only siblings
    `isearch_v` (research_indri_visit's content-bearing search + whole-doc visit read) and
    `isearch_s` (research_indri_snip's snippet-listing search + section fetch) — each a
    distinct tool NAME specifically so it gets its OWN manual (manuals are keyed per-tool+domain,
    not per-toolset; see prompts/loader.py), all resolving to the SAME skills/indri_doc.md
    manual VALUE as `isearch`. `fetch_v2`/`fetch_s` are plain aliases of `fetch` (no new
    coaching), so they carry none, exactly like `fetch` itself; `visit_v` (research_indri_visit's
    read tool) likewise carries none, exactly like `visit` itself. `search_bv`
    (research_bql_visit's {BQL search} x {whole-doc visit} factorial-cell search — the LAST of
    these additive-only siblings) resolves to the SAME skills/bql_doc_v2.md/bql_browsecomp_v2.md
    manual VALUES as `search_v2` (same BQL v2 query language, just paired with `visit_bv` instead
    of `fetch_v2`); `visit_bv` carries none, exactly like `visit`/`visit_v`. `search_bqld`
    (research_bql_dense_visit's BQL_DENSE dense-fused-ranking twin of `search_bv` — SAME
    bql_doc_v2.md/bql_browsecomp_v2.md manual VALUES, only the ranking underneath differs, see
    agent_search/retrievers/structural/bql/dense_fuse.py) and `search_bqlds` (research_bql_
    dense_snip's dense-fused twin of `search_s` — SAME bql_doc.md/bql_browsecomp.md manual
    VALUES) are additive-only siblings; `visit_bqld`/`fetch_bqlds` carry none, exactly like
    their own mirror-cell counterparts. `search_bqldf` (research_bql_dense_fetch's dense-fused
    twin of plain `search` — SAME bql_doc.md/bql_browsecomp.md manual VALUES, only the ranking
    underneath differs; completes the BQL_DENSE read-interface family alongside `search_bqld`/
    `search_bqlds`) is the newest additive-only sibling; `fetch_bqldf` carries none, exactly
    like `fetch`/`fetch_bqlds`. `search_bqldo` (research_bql_donly_visit's DENSE-ONLY twin of
    `search_bqld` — SAME bql_doc_v2.md/bql_browsecomp_v2.md manual VALUES, only the ranking
    underneath differs, see dense_fuse.py's "dense-ONLY ordering" section) and `search_bqldos`
    (research_bql_donly_snip's dense-only twin of `search_bqlds` — SAME bql_doc.md/
    bql_browsecomp.md manual VALUES) are the newest additive-only siblings; `visit_bqldo`/
    `fetch_bqldos` carry none, exactly like their own mirror-cell counterparts."""
    from agent_search.prompts.loader import _registry
    tools = _registry()["tools"]
    with_manual = [n for n, s in tools.items() if s.get("manual")]
    assert with_manual == ["search", "search_v2", "search_s", "isearch", "isearch_v", "isearch_s",
                           "search_bv", "search_bqld", "search_bqlds", "search_bqldf",
                           "search_bqldo", "search_bqldos"]


def test_codefix_exposes_only_search_and_fetch():
    text = load_prompt_text(get_prompt_spec("codefix").path)
    flat = text.replace(" ", "")
    assert '"name":"search"' in flat and '"name":"fetch"' in flat
    assert '"name":"search_bql"' not in flat and '"name":"grep"' not in flat
    assert '"name":"submit"' not in flat            # localization submit is retired


def test_codefix_grep_exposes_only_grep_and_read():
    text = load_prompt_text(get_prompt_spec("codefix_grep").path)
    flat = text.replace(" ", "")
    assert '"name":"grep"' in flat and '"name":"read"' in flat
    assert '"name":"search"' not in flat and '"name":"fetch"' not in flat


def test_codefix_grep_is_uncoached():
    """codefix_grep is the code arm's honest baseline: regex grep + a line-range read, NO
    manual (models already know regex)."""
    base = load_condition("codefix_grep")
    assert set(base.tool_names) == {"grep", "read"}
    assert render_manuals(base.tool_names, domain="code") == ""
    assert base.system.count("term[field]") == 0


def test_codefix_method_carries_the_code_manual_baseline_does_not():
    method = load_condition("codefix").system
    base = load_condition("codefix_grep").system
    assert "term[field]" in method and "term[field]" not in base


def test_codefix_and_its_baseline_share_the_same_fix_contract():
    """Both code conditions must share the identical <fix> output contract — the whole point
    of a fair in-loop baseline is that only the SEARCH INSTRUMENT differs."""
    method = load_condition("codefix").system
    base = load_condition("codefix_grep").system
    for marker in ("<fix>", "file:", "function:", "change:"):
        assert marker in method and marker in base


def test_research_baseline_is_uncoached():
    """research_bm25 is the retrieve-then-visit baseline: BM25 search + visit, NO manual."""
    base = load_condition("research_bm25")
    assert set(base.tool_names) == {"bm25_search", "visit"}
    assert render_manuals(base.tool_names, domain="general") == ""
    assert base.system.count("term[field]") == 0


def test_research_dci_is_uncoached():
    """research_dci is the brute-force baseline: bash + read over the raw corpus filesystem,
    NO retriever, NO manual (a shell needs no teaching)."""
    base = load_condition("research_dci")
    assert set(base.tool_names) == {"bash", "read"}
    assert render_manuals(base.tool_names, domain="general") == ""
    assert base.system.count("term[field]") == 0


def test_research_dci_exposes_only_bash_and_read():
    text = load_prompt_text(get_prompt_spec("research_dci").path)
    flat = text.replace(" ", "")
    assert '"name":"bash"' in flat and '"name":"read"' in flat
    assert '"name":"search"' not in flat and '"name":"fetch"' not in flat
    assert '"name":"bm25_search"' not in flat and '"name":"visit"' not in flat


def test_research_method_carries_the_doc_manual_baselines_do_not():
    method = load_condition("research").system
    bm25_base = load_condition("research_bm25").system
    dci_base = load_condition("research_dci").system
    assert "term[field]" in method
    assert "term[field]" not in bm25_base and "term[field]" not in dci_base


def test_research_and_its_baselines_share_the_same_answer_contract():
    """All three doc conditions must share the identical <answer> output contract — only the
    search instrument (structured / retrieve-then-visit / brute-force shell) differs."""
    for cond in ("research", "research_bm25", "research_dci"):
        assert "<answer>" in load_condition(cond).system


# --- surface lowering: the translator agrees with the manuals' promises ------

def test_surface_lowers_field_tags_for_both_domains():
    for q in ("isnan[call]", "save[def] NOT test[file]", '"cannot rollback"[string]',
              "seri*[def]", "handle[def] OR connect[def]"):
        r = parse(to_bql(q, "code"))
        assert r.ok and check(r.expr).ok, f"code surface {q!r} failed"
    for q in ("treaty[title]", "founder[infobox]", "history[section]",
              "munoz[title] OR muñoz[title]", "film[title,body]"):
        r = parse(to_bql(q, "doc"))
        assert r.ok and check(r.expr).ok, f"doc surface {q!r} failed"


def test_unknown_field_lowers_to_a_rejected_query():
    """An invented field is passed through so the executor's type checker rejects it with a
    readable reason — it must NOT silently translate to a valid query."""
    r = parse(to_bql("x[module]", "code"))
    assert (not r.ok) or (not check(r.expr).ok)
