"""The leave-one-section-out manual cuts: six manual sets on the fused Sieve, each dropping one
`## ` section of the corrected manual. The derived files must match scripts/derive_manual_cuts.py
and only the manual block of the system prompt may differ from the headline condition."""
import importlib.util
import re
from pathlib import Path

from agent_search.strategies import CONDITIONS, paper  # noqa: F401  (registrations)

ROOT = Path(__file__).resolve().parent.parent
FULL = "research_bql_dense_snip"
CUTS = {"nohowto": "How to search", "nofields": "The fields", "nofetch": "Fetch",
        "nohops": "Hops", "noexamples": "Worked examples", "nomistakes": "Common mistakes"}


def _generator():
    spec = importlib.util.spec_from_file_location("derive_manual_cuts", ROOT / "scripts" / "derive_manual_cuts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _headings(text: str):
    return [m.group(1) for m in re.finditer(r"(?m)^## (.+)$", text)]


def _before_manual(system: str) -> str:
    i = system.find("# Structured document search")
    return system[:i] if i >= 0 else system


def test_derived_files_match_the_generator():
    gen = _generator()
    for path, content in gen.derived_files().items():
        assert path.exists(), path.name
        assert path.read_text() == content, f"{path.name} is stale: rerun scripts/derive_manual_cuts.py"


def test_each_cut_drops_exactly_its_section():
    gen = _generator()
    for stem in ("bql_browsecomp", "bql_doc"):
        full = _headings((gen.MANUAL_DIR / f"{stem}.md").read_text())
        for ms, heading in CUTS.items():
            got = _headings((gen.MANUAL_DIR / f"{stem}_{ms}.md").read_text())
            expect = [h for h in full if not h.startswith(heading)]
            assert got == expect and len(got) == len(full) - 1, (stem, ms)


def test_only_the_manual_differs_from_the_headline():
    for profile in ("browsecomp", "wiki"):
        full = CONDITIONS[FULL].render(profile)
        for ms, heading in CUTS.items():
            c = CONDITIONS[f"research_bql_dense_snip_no{ms[2:]}"]
            s = c.render(profile)
            assert _before_manual(s) == _before_manual(full), (profile, ms)
            assert list(c.tool_names) == ["search_bqlds", "fetch_bqlds"]
            assert f"## {heading}" not in s and f"## {heading}" in full, (profile, ms)
            assert len(s.split()) < len(full.split())
