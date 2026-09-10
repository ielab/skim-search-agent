"""Code-FIX task scoring — the code arm's end-to-end metric.

The code agent ends an episode by proposing a concrete fix (a <fix> block; see
agent_search/tasks/codefix/prompt.md). We score it LOCALLY, no test execution:

  fix-file-ok : did the `file:` line name a file the gold patch edits (suffix-lenient)?

plus the efficiency axis the agent loop already records (llm_calls / steps / tokens). This
mirrors the retrieval floors' file-level Acc@1 but on the fix the agent actually committed to,
so the code and deep-research arms stay comparable in shape (a single success bit + cost),
while deep-research keeps its @k / answer-EM metrics unchanged.

Reuses agent_search.evaluation.ground_truth.gold_files (the same base-side patch parser the retrieval
metrics use), so gold is defined identically across arms.
"""
from __future__ import annotations

import re
from typing import Optional

from agent_search.evaluation.ground_truth import gold_files

_FIX = re.compile(r"<fix>(.*?)</fix>", re.DOTALL | re.IGNORECASE)
_FILE_LINE = re.compile(r"file:\s*(\S+)", re.IGNORECASE)


def extract_fix(text: str | None) -> Optional[str]:
    """The LAST <fix>...</fix> block's inner text, or None. (Last so a <fix> merely quoted
    earlier in reasoning never wins over the real trailing one.)"""
    if not text:
        return None
    blocks = _FIX.findall(text)
    return blocks[-1].strip() if blocks else None


def fix_file(fix_text: str | None) -> Optional[str]:
    """The `file:` path a <fix> block names (stripped of markup), or None."""
    if not fix_text:
        return None
    m = _FILE_LINE.search(fix_text)
    return m.group(1).strip().strip("`'\"*") if m else None


def _path_hits(file: str, gold: set[str]) -> bool:
    """Suffix-lenient path match (the same leniency the retrieval file-ranking uses)."""
    return any(file == g or file.endswith("/" + g) or g.endswith("/" + file) for g in gold)


def score_fix(fix_text: str | None, patch: str) -> tuple[bool, str]:
    """(fix_file_ok, shown_file). fix_file_ok is True iff the `file:` line hits a gold-patch
    file. `shown_file` is the file for reporting (or a short reason)."""
    gold = gold_files(patch)
    file = fix_file(fix_text)
    if not fix_text:
        return False, "(no fix emitted)"
    if not file:
        return False, "(no file line)"
    return _path_hits(file, gold), file


def is_grounded(file: str | None, read_paths) -> bool:
    """A <fix> is grounded iff its `file:` was actually fetched/read this episode (so a fix is
    never a blind guess). Suffix-lenient, matching the scorer. Empty file -> not grounded."""
    if not file:
        return False
    return any(file == p or file.endswith("/" + p) or p.endswith("/" + file)
               for p in read_paths)
