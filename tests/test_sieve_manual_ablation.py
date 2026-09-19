"""The Sieve manual and its ablation. The default manual is the reference (query language, fields,
fetch call, worked examples). Each variant swaps only the manual block of the system prompt; the
composed variants must match scripts/compose_manuals.py."""
import importlib.util
import re
from pathlib import Path

from agent_search.strategies import CONDITIONS, paper  # noqa: F401  (registrations)
from agent_search.tools.search_bql.tool import SearchBql

ROOT = Path(__file__).resolve().parent.parent
FULL = "research_bql_dense_snip"
VARIANTS = ("card", "nomanual", "reference_howto", "reference_hops", "reference_mistakes",
            "reference_howto_hops_mistakes", "reference_howto_hops_mistakes_noconstruct")


def _composer():
    spec = importlib.util.spec_from_file_location("compose_manuals", ROOT / "scripts" / "compose_manuals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _before_manual(system: str) -> str:
    cuts = [i for i in (system.find("# Structured document search"), system.find("# Tool reference")) if i >= 0]
    return system[:min(cuts)] if cuts else system


def _headings(text: str):
    return [m.group(1) for m in re.finditer(r"(?m)^## (.+)$", text)]


def test_default_is_the_reference_manual():
    assert SearchBql(name="search_bqlds", ranking="fused").manual_set == "reference"
    for profile in ("browsecomp", "wiki"):
        s = CONDITIONS[FULL].render(profile)
        assert "## The fields" in s and "## Worked examples" in s and "{\"rank\": 1, \"section\":" in s
        for gone in ("## How to search", "## Hops", "## Common mistakes"):
            assert gone not in s, (profile, gone)


def test_only_the_manual_differs():
    for profile in ("browsecomp", "wiki"):
        heads = {c: _before_manual(CONDITIONS[c].render(profile)) for c in (FULL,) + tuple(f"{FULL}_{v}" for v in VARIANTS)}
        assert len(set(heads.values())) == 1, profile
        assert all(list(CONDITIONS[f"{FULL}_{v}"].tool_names) == ["search_bqlds", "fetch_bqlds"] for v in VARIANTS)


def test_composed_manuals_match_the_composer():
    comp = _composer()
    for path, content in comp.composed_files().items():
        assert path.exists() and path.read_text() == content, f"{path.name} is stale: rerun scripts/compose_manuals.py"


def test_each_variant_carries_its_named_parts():
    comp = _composer()
    for stem in ("bql_browsecomp", "bql_doc"):
        ref = _headings((comp.MANUAL_DIR / f"{stem}.md").read_text())
        assert [h.split(":")[0].split(" —")[0] for h in ref] == ["The fields", "Fetch", "Worked examples"]
        for suffix, advice in comp.VARIANTS.items():
            got = [h.split(":")[0].split(" —")[0] for h in _headings((comp.MANUAL_DIR / f"{stem}_{suffix}.md").read_text())]
            assert got == [n for n in comp.ORDER if n in ("The fields", "Fetch", "Worked examples") or n in advice], (stem, suffix)


def test_card_and_stub():
    s = CONDITIONS[f"{FULL}_nomanual"].render("browsecomp")
    assert "No further guidance is given in this condition." in s and "## The fields" not in s
    for profile in ("browsecomp", "wiki"):
        card = CONDITIONS[f"{FULL}_card"].render(profile)
        card = card[card.find("# Structured document search"):]
        assert len(card.split()) < 200 and "## " not in card
        for want in ("`term[field]`", "`title`", "`body`", "{\"rank\": 1, \"section\": \"Career\"}"):
            assert want in card, (profile, want)


def test_noconstruct_drops_the_construction_advice():
    full = CONDITIONS[f"{FULL}_reference_howto_hops_mistakes"].render("browsecomp")
    s = CONDITIONS[f"{FULL}_reference_howto_hops_mistakes_noconstruct"].render("browsecomp")
    assert "## How to search" in s and "## Common mistakes" in s
    for gone in ("Field-scope tightly", "use sparingly", "try `word*`", "`AND` requires ALL clauses"):
        assert gone in full and gone not in s, gone


def test_manual_sets_resolve_to_files():
    for ms in ("reference",) + VARIANTS + ("v2",):
        t = SearchBql(name="search_bqlds", ranking="fused", manual_set=ms)
        for profile in ("browsecomp", "wiki", "general", "code"):
            assert t.manual_path(profile).endswith(".md"), (ms, profile)
