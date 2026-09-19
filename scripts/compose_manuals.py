"""Compose the Sieve manual variants from the reference manual and its advice sections.

The default manual is the reference: the query language, the fields, the fetch call and worked
examples (`agent_search/tools/search_bql/bql_browsecomp.md` for the BrowseComp profile,
`bql_doc.md` for the wiki and general profiles). The advice sections the ablation adds back
("How to search", "Hops", "Common mistakes") live in `bql_<profile>_advice.md`. Each composed
manual keeps the sections in one fixed order, so `reference_howto_hops_mistakes` is the full
manual the ablation started from.

    python scripts/compose_manuals.py          # rewrite the composed files
    python scripts/compose_manuals.py --check  # exit 1 if a composed file is stale
"""
import re
import sys
from pathlib import Path

MANUAL_DIR = Path(__file__).resolve().parent.parent / "agent_search" / "tools" / "search_bql"
STEMS = ("bql_browsecomp", "bql_doc")
ORDER = ("How to search", "The fields", "Fetch", "Hops", "Worked examples", "Common mistakes")
# manual-set suffix -> the advice sections added to the reference
VARIANTS = {
    "reference_howto": ("How to search",),
    "reference_hops": ("Hops",),
    "reference_mistakes": ("Common mistakes",),
    "reference_howto_hops_mistakes": ("How to search", "Hops", "Common mistakes"),
}


def sections(text: str):
    """The preamble and the (heading line, body) pairs of every `## ` section."""
    parts = re.split(r"(?m)^(?=## )", text)
    return parts[0], [(p.partition("\n")[0], p.partition("\n")[2]) for p in parts[1:]]


def compose(stem: str, advice: tuple) -> str:
    preamble, ref = sections((MANUAL_DIR / f"{stem}.md").read_text())
    _, adv = sections((MANUAL_DIR / f"{stem}_advice.md").read_text())
    pool = {h[3:].split(":")[0].split(" —")[0].strip(): (h, b) for h, b in ref + adv}
    wanted = [name for name in ORDER if name in ("The fields", "Fetch", "Worked examples") or name in advice]
    return preamble + "".join(pool[n][0] + "\n" + pool[n][1] for n in wanted)


def composed_files():
    return {MANUAL_DIR / f"{stem}_{suffix}.md": compose(stem, advice)
            for stem in STEMS for suffix, advice in VARIANTS.items()}


def main(argv):
    check = "--check" in argv
    stale = []
    for path, content in composed_files().items():
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(path.name)
        else:
            path.write_text(content)
            print("wrote", path.name, len(content.split()), "words")
    if check:
        print("stale: " + ", ".join(stale) if stale else "all composed manuals match")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
