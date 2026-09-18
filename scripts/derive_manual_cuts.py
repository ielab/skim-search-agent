"""Derive the leave-one-section-out manuals for the Sieve manual ablation.

The full manuals are `agent_search/tools/search_bql/bql_browsecomp.md` (BrowseComp profile) and
`bql_doc.md` (general and wiki profiles). Each cut drops exactly one `## ` section and keeps the
preamble and every other section unchanged. Each add-back keeps the preamble, the three
reference sections (The fields, Fetch, Worked examples) and exactly one advice section. The derived files are committed next to the full
manuals; `tests/test_manual_cuts.py` checks they still match this derivation, so edit the full
manual and rerun this script rather than editing a derived file.

    python scripts/derive_manual_cuts.py          # rewrite the derived files
    python scripts/derive_manual_cuts.py --check  # exit 1 if any derived file is stale
"""
import re
import sys
from pathlib import Path

MANUAL_DIR = Path(__file__).resolve().parent.parent / "agent_search" / "tools" / "search_bql"

# manual-set suffix -> the heading of the section that cut removes (prefix match on the `## ` line)
CUTS = {
    "nohowto": "How to search",
    "nofields": "The fields",
    "nofetch": "Fetch",
    "nohops": "Hops",
    "noexamples": "Worked examples",
    "nomistakes": "Common mistakes",
}
# manual-set suffix -> the one advice section added back on top of the reference-only manual
# (The fields, Fetch, Worked examples); the mirror of CUTS, starting from the best manual
ADDS = {
    "refhowto": "How to search",
    "refhops": "Hops",
    "refmistakes": "Common mistakes",
}
REFERENCE = ("The fields", "Fetch", "Worked examples")
FULL = {"bql_browsecomp": "bql_browsecomp.md", "bql_doc": "bql_doc.md"}


def split_sections(text: str):
    """The preamble, then (heading line, body) pairs for every `## ` section, in order."""
    parts = re.split(r"(?m)^(?=## )", text)
    preamble, sections = parts[0], []
    for p in parts[1:]:
        heading, _, body = p.partition("\n")
        sections.append((heading, body))
    return preamble, sections


def cut(text: str, heading_prefix: str) -> str:
    preamble, sections = split_sections(text)
    kept = [(h, b) for h, b in sections if not h[3:].startswith(heading_prefix)]
    if len(kept) != len(sections) - 1:
        raise ValueError(f"expected exactly one section starting with {heading_prefix!r}")
    return preamble + "".join(h + "\n" + b for h, b in kept)


def keep(text: str, heading_prefixes) -> str:
    """The preamble plus only the sections whose heading starts with one of the prefixes, in
    the full manual's order."""
    preamble, sections = split_sections(text)
    kept = [(h, b) for h, b in sections if any(h[3:].startswith(p) for p in heading_prefixes)]
    if len(kept) != len(heading_prefixes):
        raise ValueError(f"expected exactly one section for each of {heading_prefixes!r}")
    return preamble + "".join(h + "\n" + b for h, b in kept)


def derived_files():
    """{derived path: content} for every (full manual, cut) and (full manual, add-back)."""
    out = {}
    for stem, name in FULL.items():
        text = (MANUAL_DIR / name).read_text()
        for suffix, heading in CUTS.items():
            out[MANUAL_DIR / f"{stem}_{suffix}.md"] = cut(text, heading)
        for suffix, heading in ADDS.items():
            out[MANUAL_DIR / f"{stem}_{suffix}.md"] = keep(text, REFERENCE + (heading,))
    return out


def main(argv):
    check = "--check" in argv
    stale = []
    for path, content in derived_files().items():
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(path.name)
        else:
            path.write_text(content)
            print("wrote", path.name, len(content.split()), "words")
    if check:
        print("stale: " + ", ".join(stale) if stale else "all derived manuals match")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
