"""The code-fix GREP baseline ACI: real regex `grep` (matching LINES) + `read` (line range).

This is the honest, uncoached full-power baseline for the code arm (RISE-style: give the agent
`grep`/`rg` as it naturally uses them). It mirrors `CodeFixWorkspace`'s `<fix>` contract exactly
— same episode, same fix-file-ok scoring, same grounding guard — but the search primitive is a
real regular expression over live source (not the field-tagged Boolean surface), and the read is
a plain line-range slice (not a structural part fetch). No skill/manual: models already know
regex; the fair baseline is grep-as-used.

Ported from an internal prototype (RegexGrep + GrepWorkspace), not part of this release.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from agent_search.core.tokens import cap_tokens
from agent_search.corpus.units import CodeUnit

# per-line cap on a grep hit's shown text, in whitespace tokens (agent_search.core.tokens) —
# SkimSearchAgent caps text in tokens everywhere, never characters.
GREP_LINE_TOKENS = int(os.environ.get("GREP_LINE_TOKENS", "24"))


class RegexGrep:
    """search(pattern) -> (hits, n_units_matched); hits = (doc_id, abs_line, line_text).

    grep as intended: the pattern is a real regular expression (alternation, classes, anchors,
    word boundaries; case-insensitive like `rg -i`), and the return is line-granular. An invalid
    regex degrades to a literal substring search (`grep -F`), so a typo'd pattern still returns
    something. Match order is per-unit match count, then corpus order (real grep enumerates)."""

    def __init__(self, units: Sequence[CodeUnit]):
        self._units = list(units)
        self._lines = [(u, (u.code or "").splitlines()) for u in self._units]

    def search(self, pattern: str, max_lines: int = 10,
               max_per_unit: int = 2) -> tuple[list[tuple[str, int, str]], int]:
        pattern = (pattern or "").strip()
        if not pattern:
            return [], 0
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)   # grep -F fallback
        per_unit: list[tuple[object, list[tuple[int, str]]]] = []
        for u, lines in self._lines:
            m = [(u.start_line + i, cap_tokens(ln.strip(), GREP_LINE_TOKENS))
                 for i, ln in enumerate(lines) if rx.search(ln)]
            if m:
                per_unit.append((u, m))
        n_matched = len(per_unit)
        per_unit.sort(key=lambda t: -len(t[1]))          # most matches first (grep -c order)
        hits: list[tuple[str, int, str]] = []
        for u, m in per_unit:
            for line_no, text in m[:max_per_unit]:
                hits.append((u.doc_id, line_no, text))
                if len(hits) >= max_lines:
                    return hits, n_matched
        return hits, n_matched


class GrepReadWorkspace:
    """The code grep baseline: `grep(pattern)` -> matching lines; `read(path, start, end)` ->
    a line-range slice (capped ~80 lines). Same `run(name, args)` dispatch contract as
    `CodeFixWorkspace`, so the agent loop drives it identically; the agent still commits a
    `<fix>` (scored fix-file-ok). Uncoached — no manual."""

    tools = ("grep", "read")

    def __init__(self, units: Sequence[CodeUnit], files: dict):
        self.files = files or {}
        self.rg = RegexGrep(units)

    def grep(self, pattern: str) -> str:
        hits, n_units = self.rg.search(pattern, max_lines=10)
        if not hits:
            return f"0 matches for /{pattern}/"
        lines = [f"{n_units} units matched /{pattern}/, first lines:"]
        for doc_id, line_no, text in hits:
            path = doc_id.split("::")[0]
            lines.append(f"  {path}:{line_no}: {text}")
        return "\n".join(lines)

    def read(self, path: str, start=None, end=None) -> str:
        src = self.files.get(path)
        if src is None:
            cands = [p for p in self.files if p.endswith("/" + path.split("/")[-1]) or p == path]
            if len(cands) == 1:
                path, src = cands[0], self.files[cands[0]]
            else:
                extra = f" Did you mean: {', '.join(cands[:4])}" if cands else ""
                return f"ERROR: no such file: {path}.{extra}"
        lines = src.splitlines()
        s = max(1, int(start or 1))
        e = min(len(lines), int(end or s + 60))
        e = min(e, s + 79)                        # cap a read at 80 lines
        body = "\n".join(f"{i}: {ln}" for i, ln in enumerate(lines[s - 1:e], start=s))
        return f"{path} lines {s}-{e} of {len(lines)}:\n{body}"

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name == "grep":
                return self.grep(args.get("pattern") or args.get("query") or "")
            if name == "read":
                return self.read(args.get("path") or args.get("file") or "",
                                 args.get("start"), args.get("end"))
        except Exception as e:  # noqa: BLE001 — a tool error is an observation, not a crash
            return f"ERROR: {type(e).__name__}: {e}"
        return (f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}.")
