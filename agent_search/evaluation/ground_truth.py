"""Derive localization ground truth from a SWE-bench gold patch.

The corpus is the repo at the BASE commit (pre-patch), so we map the patch's
*old-side* (a-side) line numbers onto base-commit code units. A unit is gold if
its line range overlaps any changed old-side range.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

from agent_search.corpus.units import CodeUnit

# stop at tab/CR/newline so CRLF diffs or `--- a/path\ttimestamp` don't corrupt the path
# matches BOTH `--- a/<path>` and `--- /dev/null` (new files): /dev/null sections
# must terminate the previous file's block, or a patch that ADDS a file after a
# modified one leaks the new file's hunks into the previous file's gold ranges
_OLD_PATH = re.compile(r"^--- (?:a/([^\t\r\n]+)|/dev/null)", re.MULTILINE)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _iter_file_blocks(diff: str):
    """Yield (path, block_text) for each '--- a/<path>' file section."""
    matches = list(_OLD_PATH.finditer(diff))
    for i, m in enumerate(matches):
        path = m.group(1)
        if path is None:          # `--- /dev/null` = newly added file: no old side
            continue              # (still consumed as a boundary above)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(diff)
        yield path, diff[m.end():end]


def changed_line_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Per file path, the inclusive old-side (base) line ranges the patch EDITS.

    Walks each hunk body and records only the actually-changed lines: '-' lines at
    their old-side line number; '+' insertions as a zero-width anchor at the line
    before the insertion point. Context lines do NOT count — using the hunk-header
    span would mark neighboring functions gold merely for appearing as diff
    context (the LocAgent/Agentless convention maps edited lines only).
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for path, block in _iter_file_blocks(diff):
        ranges: list[tuple[int, int]] = []
        hunks = list(_HUNK.finditer(block))
        for i, h in enumerate(hunks):
            old_start = int(h.group(1))
            old_len = int(h.group(2)) if h.group(2) is not None else 1
            # for old_len == 0 git sets old_start to the line BEFORE the
            # insertion, not the first line of a range — shift so `old_ln`
            # uniformly means "next unconsumed old-side line"
            old_ln = old_start if old_len > 0 else old_start + 1
            end = hunks[i + 1].start() if i + 1 < len(hunks) else len(block)
            body = block[h.end():end]
            nl = body.find("\n")                       # drop the header line's
            body = body[nl + 1:] if nl != -1 else ""   # trailing `@@ def foo():`
            for line in body.splitlines():
                if line.startswith("-"):
                    ranges.append((old_ln, old_ln))
                    old_ln += 1
                elif line.startswith("+"):
                    s = max(1, old_ln - 1)             # insertion: line before
                    ranges.append((s, s))
                elif line.startswith("\\"):            # "\ No newline at end of file"
                    continue
                else:                                  # context line
                    old_ln += 1
        if ranges:
            out[path] = list(dict.fromkeys(ranges))   # dedup, keep order
    return out


def gold_files(diff: str) -> set[str]:
    """The set of base-side file paths the patch modifies."""
    return set(changed_line_ranges(diff).keys())


def gold_units(
    file_ranges: Mapping[str, Sequence[tuple[int, int]]],
    units_by_file: Mapping[str, Sequence[CodeUnit]],
) -> set[str]:
    """doc_ids of units whose [start_line, end_line] overlaps any changed range."""
    gold: set[str] = set()
    for path, ranges in file_ranges.items():
        for unit in units_by_file.get(path, []):
            for (lo, hi) in ranges:
                if unit.start_line <= hi and lo <= unit.end_line:  # overlap
                    gold.add(unit.doc_id)
                    break
    return gold
