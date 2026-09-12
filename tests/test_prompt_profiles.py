"""Conditions = task template x strategy (tools + options) composition.

The method, both arms: the BQL field-tagged Boolean surface + search -> fetch.
  code  : task=codefix,  strategy=codefix          (files -> functions, <fix>)
  docs  : task=research, strategy=sieve_bm25       (articles -> sections, <answer>)
          task=research, strategy=search_visit     (retrieve-then-visit baseline)
Per-tool teaching lives in a tool's manual (a markdown file next to its tool.py) and renders
only when the tool is in the strategy; the field-tagged surface lowers to the same executor AST
via retrievers/bql/surface.to_bql, so the parser/typechecker/executor are reused.
"""
import re

import pytest

from agent_search.snippets import TermWindow
from agent_search.strategies import CONDITIONS, get_condition
from agent_search.tasks.base import TASKS
from agent_search.tasks.render import render_manuals
from agent_search.retrievers.bql.parser import parse
from agent_search.retrievers.bql.surface import to_bql
from agent_search.retrievers.bql.types import check

# Document conditions live in the "general" domain; the code-localization arm
# (codefix / codefix_grep / codefix_patch over the codefix templates) in "code".
GEN_CONDS = ("research_snip", "research_bm25", "research_dci")
CODE_CONDS = ("codefix", "codefix_grep", "codefix_patch")
ALL_CONDS = GEN_CONDS + CODE_CONDS


def test_domains():
    assert {c.domain for c in CONDITIONS.values()} == {"general", "code"}


def test_code_conditions_compose_the_code_toolsets():
    for name, tools in (("codefix", ("search", "fetch")), ("codefix_grep", ("grep", "read")),
                        ("codefix_patch", ("search", "fetch"))):
        cond = get_condition(name)
        assert cond.domain == "code" and cond.tool_names == tools
        assert '"name":"' + tools[0] + '"' in cond.render()
    assert "bql_code" not in get_condition("codefix_grep").render().lower()


@pytest.mark.parametrize("cond", ALL_CONDS)
def test_every_condition_resolves_and_composes(cond):
    c = get_condition(cond)
    assert c.name == cond
    assert c.domain in ("general", "code")
    assert "<tools>" in c.render()
    assert c.render().strip()


def test_condition_domains_are_correct():
    for c in GEN_CONDS:
        assert get_condition(c).domain == "general"


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        get_condition("nonsense")


def test_retired_conditions_are_gone():
    """The old prefix-surface / one-shot-search / localization conditions are retired.

    `bm25`, `dense`, `bql` and `grep` are not in this list: those names are the loop-free
    retrieval-only floors (`agent_search/strategies/retrieval_only.py`), live conditions."""
    for gone in ("tools", "tools_bql", "tools_nodense", "research_bql", "research_tools"):
        with pytest.raises(ValueError):
            get_condition(gone)


# --- tasks are tool-agnostic; manuals carry the per-tool teaching -----------

def test_tasks_are_tool_agnostic():
    """Task templates carry the {{tools}}/{{tool_manuals}} slots and must not TEACH the query
    language (that lives in the manuals) — a single illustrative example call is fine, but the
    operator reference / field table constructs must not appear in the task body."""
    for task in ("research", "research_paper"):
        _, body = TASKS[task].template()
        assert "{{tools}}" in body and "{{tool_manuals}}" in body
        for marker in ("IN(def", "term[field]", "| `def` |", "| `title` |",
                       "## Field", "## Combine"):
            assert marker not in body, f"{task} still teaches the surface: {marker}"


def test_research_output_contract():
    _, body = TASKS["research"].template()
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


def _manual_of(tool, domain: str = "general") -> str:
    return render_manuals([tool.manual_path(domain)])


def test_doc_manual_worked_examples_lower_and_typecheck():
    from agent_search.tools.search_bql.tool import SearchBql
    manual = _manual_of(SearchBql(name="search_s", snippet=TermWindow()))
    assert manual and "term[field]" in manual, "doc `search_s` must declare the field-tagged manual"
    ex = _worked_examples(manual)
    assert ex, "no worked term[field] examples found in the BQL doc manual"
    for q in ex:
        bql = to_bql(q, domain="doc")
        r = parse(bql)
        assert r.ok, f"parse failed for {q!r} -> {bql!r}: {r.error}"
        assert check(r.expr).ok, f"type-check failed for {q!r} -> {bql!r}"


def test_doc_manual_advertises_its_fields():
    from agent_search.tools.search_bql.tool import SearchBql
    docs = _manual_of(SearchBql(name="search_s", snippet=TermWindow()))
    assert docs
    for f in ("title", "section", "body", "infobox"):
        assert f"`{f}`" in docs or f"[{f}]" in docs, f"doc manual missing field {f!r}"
    assert "[def]" not in docs and "[call]" not in docs


def test_only_search_family_tools_have_a_manual():
    """`search_s` (research_snip's excerpt-listing field-tagged surface) carries a manual; so
    do its siblings `isearch_s` (research_indri_snip's Indri graded query language) and the
    BQL_DENSE dense-fused twins `search_bqld{f,os}`/`search_bqlds` (research_bql_dense_fetch/
    research_bql_donly_snip/research_bql_dense_snip — SAME BQL doc manual VALUES, only the
    ranking underneath differs, see agent_search/retrievers/bql/dense_fuse.py).
    `fetch`/`fetch_s`/`fetch_bqld{f,os,s}` carry none — they're plain reads, no new coaching.
    The code arm's `search` carries the code BQL manual (bql_code.md)."""
    with_manual = sorted({t.name for cond in CONDITIONS.values() for t in cond.strategy.tools if t.manual})
    assert with_manual, "no tool carries a manual"
    for name in with_manual:                    # only the query-language searches coach the agent
        assert name.startswith(("search", "isearch")), name
    without = {t.name for cond in CONDITIONS.values() for t in cond.strategy.tools if not t.manual}
    assert not any(n.startswith(("search_s", "search_bql", "isearch")) for n in without), sorted(without)
    assert not any(n.startswith(("fetch", "visit", "bash", "read", "get_document")) for n in with_manual)


def test_research_baseline_is_uncoached():
    """research_bm25 is the retrieve-then-visit baseline: BM25 search + visit, NO manual."""
    base = get_condition("research_bm25")
    assert set(base.tool_names) == {"bm25_search", "visit"}
    assert all(not t.manual for t in base.strategy.tools)
    assert base.render().count("term[field]") == 0


def test_research_dci_is_uncoached():
    """research_dci is the brute-force baseline: bash + read over the raw corpus filesystem,
    NO retriever, NO manual (a shell needs no teaching)."""
    base = get_condition("research_dci")
    assert set(base.tool_names) == {"bash", "read"}
    assert all(not t.manual for t in base.strategy.tools)
    assert base.render().count("term[field]") == 0


def test_research_dci_exposes_only_bash_and_read():
    text = get_condition("research_dci").render()
    flat = text.replace(" ", "")
    assert '"name":"bash"' in flat and '"name":"read"' in flat
    assert '"name":"search"' not in flat and '"name":"fetch"' not in flat
    assert '"name":"bm25_search"' not in flat and '"name":"visit"' not in flat


def test_research_method_carries_the_doc_manual_baselines_do_not():
    method = get_condition("research_snip").render()
    bm25_base = get_condition("research_bm25").render()
    dci_base = get_condition("research_dci").render()
    assert "term[field]" in method
    assert "term[field]" not in bm25_base and "term[field]" not in dci_base


def test_research_and_its_baselines_share_the_same_answer_contract():
    """All three doc conditions must share the identical <answer> output contract — only the
    search instrument (structured / retrieve-then-visit / brute-force shell) differs."""
    for cond in GEN_CONDS:
        assert "<answer>" in get_condition(cond).render()


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


def test_tool_rules_render_from_declarations():
    """`{{tool_rules}}` names each tool's exact argument structure, so the combined prompt's
    strict rules follow the toolset instead of hardcoding search and get_document."""
    from agent_search.strategies import CONDITIONS
    from agent_search.tasks.render import render_tool_rules
    assert render_tool_rules([{"name": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}]) == '- search: {"query": <string>, "k": <integer, optional>}'
    for name, tools in (("search_visit", ("bm25_search", "visit")), ("research_dedup_dense_combined", ("search", "get_document"))):
        rendered = CONDITIONS[name].render()
        assert "{{" not in rendered.replace("{{step_budget}}", "")   # the budget is filled when the loop starts
        for tool in tools:
            assert f"- {tool}: {{" in rendered
