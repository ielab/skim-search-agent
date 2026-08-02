"""Compile the code-fix agent's <fix> SEARCH/REPLACE edits into a git-applyable
unified diff (`model_patch`) for the real SWE-bench harness.

The patch-mode agent (task `taskfix_patch`) never sees whole files — it works through the
same lean search->fetch ACI as `codefix` and ends with anchored edits:

    <fix>
    file: path/to/mod.py
    <<<<<<< SEARCH
    <exact current lines>
    =======
    <replacement lines>
    >>>>>>> REPLACE
    </fix>

We hold the TRUE source at base_commit (AgentRetriever._files), so we locate each SEARCH
block in the real file and splice the REPLACE — line numbers come from the real file, not
from the model, which is why the resulting diff applies. Matching is tolerant of the
`<n>: ` line-number prefixes the tools render and of trailing-whitespace noise; if an edit
cannot be anchored, it is dropped (its instance simply gets no/partial patch -> scored
unresolved, which is honest). Output is a standard multi-file unified diff.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_FIX = re.compile(r"<fix>(.*?)</fix>", re.DOTALL | re.IGNORECASE)
_FILE = re.compile(r"^\s*file\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SEARCH = re.compile(r"^\s*<{5,}\s*SEARCH\s*$", re.IGNORECASE)
_DIVIDER = re.compile(r"^\s*={5,}\s*$")
_REPLACE = re.compile(r"^\s*>{5,}\s*REPLACE\s*$", re.IGNORECASE)
_LN_PREFIX = re.compile(r"^\s*\d+:\s?")          # "  810: code" -> "code" (tool render prefix)


@dataclass
class Edit:
    path: str
    search: str
    replace: str


@dataclass
class SynthReport:
    n_edits: int = 0
    n_applied: int = 0
    files: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)   # human-readable per-edit misses


def _strip_ln(line: str) -> str:
    return _LN_PREFIX.sub("", line)


def parse_fix_edits(fix_text: str) -> list[Edit]:
    """Parse a <fix> body (or a raw SEARCH/REPLACE payload) into Edits.

    Tracks the current `file:` across edits so multiple edits to one file may omit repeats.
    Line-number prefixes the tools render are stripped from both sections."""
    if not fix_text:
        return []
    m = _FIX.search(fix_text)
    body = m.group(1) if m else fix_text
    edits: list[Edit] = []
    cur_file: str | None = None
    lines = body.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        fm = _FILE.match(line)
        if fm:
            cur_file = fm.group(1).strip().strip("`").strip()
            i += 1
            continue
        if _SEARCH.match(line):
            i += 1
            search: list[str] = []
            while i < n and not _DIVIDER.match(lines[i]):
                search.append(_strip_ln(lines[i]))
                i += 1
            i += 1  # skip divider
            replace: list[str] = []
            while i < n and not _REPLACE.match(lines[i]):
                replace.append(_strip_ln(lines[i]))
                i += 1
            i += 1  # skip REPLACE marker
            if cur_file:
                edits.append(Edit(path=cur_file,
                                  search="\n".join(search),
                                  replace="\n".join(replace)))
            continue
        i += 1
    return edits


def _locate_lines(src_lines: list[str], q_lines: list[str]) -> tuple[int, int] | None:
    """Find the contiguous window of `src_lines` (as split, no keepends) matching q_lines.
    Tier 1: exact. Tier 2: trailing-whitespace tolerant. Returns (start, end) or None if
    not found OR ambiguous (matches >1 place -> refuse rather than edit the wrong span)."""
    m = len(q_lines)
    if m == 0:
        return None
    n = len(src_lines)

    def matches(cmp) -> list[int]:
        hits = []
        for s in range(0, n - m + 1):
            if all(cmp(src_lines[s + j]) == cmp(q_lines[j]) for j in range(m)):
                hits.append(s)
        return hits

    for cmp in (lambda x: x, lambda x: x.rstrip(), lambda x: x.strip()):
        hits = matches(cmp)
        if len(hits) == 1:
            return hits[0], hits[0] + m
        if len(hits) > 1:
            return None            # ambiguous under this tier — do not guess
    return None


def _leading_ws(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


def _reindent(lines: list[str], delta: int) -> list[str]:
    """Shift every non-blank line's leading spaces by `delta` (re-base a mis-indented
    REPLACE block onto the real source indentation). Blank lines are left as-is."""
    if delta == 0:
        return lines
    out = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
        elif delta > 0:
            out.append(" " * delta + ln)
        else:
            out.append(ln[min(-delta, _leading_ws(ln)):])
    return out


def _apply_edit(source: str, edit: Edit) -> str | None:
    """Return source with the edit applied, or None if the SEARCH can't be uniquely located."""
    src_lines = source.splitlines()
    q_lines = edit.search.splitlines()
    span = _locate_lines(src_lines, q_lines)
    if span is None:
        return None
    s, e = span
    repl = edit.replace.splitlines()
    # RE-BASE indentation: when SEARCH matched the real source under a leading-whitespace-
    # tolerant tier, the model's REPLACE often carries the SAME wrong indent (it copied its own
    # mis-indented SEARCH). Shift REPLACE by the delta between the real matched line and the
    # model's SEARCH line, so the spliced code keeps the file's ACTUAL indentation — otherwise
    # the patch applies cleanly but is a Python IndentationError (silent wrong, worse than a miss).
    if q_lines:
        delta = _leading_ws(src_lines[s]) - _leading_ws(q_lines[0])
        repl = _reindent(repl, delta)
    new_lines = src_lines[:s] + repl + src_lines[e:]
    trailing = "\n" if source.endswith("\n") else ""
    return "\n".join(new_lines) + trailing


def _file_diff(path: str, old: str, new: str) -> str:
    """A git-style unified diff for one file (empty string if unchanged)."""
    if old == new:
        return ""
    old_l = old.splitlines(keepends=True)
    new_l = new.splitlines(keepends=True)
    # git apply keys off the ---/+++ a/ b/ headers; the `diff --git` line is conventional.
    diff = difflib.unified_diff(old_l, new_l, fromfile=f"a/{path}", tofile=f"b/{path}")
    body = "".join(diff)
    if body and not body.endswith("\n"):
        body += "\n"
    return f"diff --git a/{path} b/{path}\n{body}"


def synthesize_patch(fix_text: str, files: dict) -> tuple[str, SynthReport]:
    """Compile a <fix> body + the true base_commit `files` ({path: source}) into a unified
    diff string (git-applyable) plus a report. Empty diff -> no edit anchored."""
    report = SynthReport()
    edits = parse_fix_edits(fix_text)
    report.n_edits = len(edits)
    if not edits:
        return "", report
    # group edits per file, apply sequentially to a working copy, diff once per file.
    working: dict[str, str] = {}
    order: list[str] = []
    for ed in edits:
        if ed.path not in files:
            report.failures.append(f"{ed.path}: not in corpus files")
            continue
        base = working.get(ed.path, files[ed.path])
        applied = _apply_edit(base, ed)
        if applied is None:
            report.failures.append(f"{ed.path}: SEARCH block not uniquely located")
            continue
        if ed.path not in working:
            order.append(ed.path)
        working[ed.path] = applied
        report.n_applied += 1
    parts = []
    for path in order:
        d = _file_diff(path, files[path], working[path])
        if d:
            parts.append(d)
            report.files.append(path)
    return "".join(parts), report
