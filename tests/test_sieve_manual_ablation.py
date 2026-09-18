"""The Sieve manual ablation: three manual sets on the same fused Sieve. Only the manual block
of the system prompt may differ between the paper condition and the three ablation conditions
(BoolAgent's control grid, applied to this library's corrected manuals)."""
from agent_search.strategies import CONDITIONS, paper  # noqa: F401  (registrations)
from agent_search.tools.search_bql.tool import SearchBql

FULL = "research_bql_dense_snip"
ABLATIONS = ("research_bql_dense_snip_nomanual", "research_bql_dense_snip_syntax",
             "research_bql_dense_snip_noconstruct")


def _before_manual(system: str) -> str:
    cuts = [i for i in (system.find("# Structured document search"), system.find("# Tool reference")) if i >= 0]
    return system[:min(cuts)] if cuts else system


def test_only_the_manual_differs():
    for profile in ("browsecomp", "wiki"):
        heads = {c: _before_manual(CONDITIONS[c].render(profile)) for c in (FULL,) + ABLATIONS}
        assert len(set(heads.values())) == 1, profile
        assert all(list(CONDITIONS[c].tool_names) == ["search_bqlds", "fetch_bqlds"] for c in ABLATIONS)


def test_nomanual_is_the_stub():
    s = CONDITIONS["research_bql_dense_snip_nomanual"].render("browsecomp")
    assert "No further guidance is given in this condition." in s
    assert "## The fields" not in s and "## How to search" not in s


def test_syntax_keeps_the_reference_and_drops_the_advice():
    for profile in ("browsecomp", "wiki"):
        s = CONDITIONS["research_bql_dense_snip_syntax"].render(profile)
        assert "## The fields" in s and "## Worked examples" in s and "{\"rank\": 1, \"section\":" in s
        for gone in ("## How to search", "## Hops", "## Common mistakes"):
            assert gone not in s, (profile, gone)


def test_noconstruct_keeps_the_sections_but_not_the_construction_advice():
    full = CONDITIONS[FULL].render("browsecomp")
    s = CONDITIONS["research_bql_dense_snip_noconstruct"].render("browsecomp")
    assert "## How to search" in s and "## Common mistakes" in s
    for gone in ("Field-scope tightly", "use sparingly", "try `word*`", "`AND` requires ALL clauses"):
        assert gone in full and gone not in s, gone
    assert len(s.split()) < len(full.split())


def test_manual_sets_resolve_to_files():
    for ms in ("nomanual", "syntax", "noconstruct"):
        t = SearchBql(name="search_bqlds", ranking="fused", manual_set=ms)
        for profile in ("browsecomp", "wiki", "general", "code"):
            assert t.manual_path(profile).endswith(".md"), (ms, profile)
