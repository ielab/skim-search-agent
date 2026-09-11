"""`grep`: real regex over the repository files, matching lines, used by the `codefix_grep` strategy.

The pattern is a real regular expression (alternation, classes, anchors, word boundaries;
case-insensitive like `rg -i`); an invalid regex degrades to a literal substring search
(`grep -F`). Match order is per-unit match count, then corpus order (real grep enumerates).
Paired with `agent_search.tools.read.tool.Read(source="repo")` to read the interesting file.
"""
from __future__ import annotations

import os
import re
from typing import Sequence

from agent_search.tokens import cap_tokens
from agent_search.corpus.units import CodeUnit
from agent_search.tools.base import Tool

# per-line cap on a grep hit's shown text, in model tokens (agent_search.tokens).
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
               max_per_unit: int = 2) -> "tuple[list[tuple[str, int, str]], int]":
        pattern = (pattern or "").strip()
        if not pattern:
            return [], 0
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)   # grep -F fallback
        per_unit: "list[tuple[object, list[tuple[int, str]]]]" = []
        for u, lines in self._lines:
            m = [(u.start_line + i, cap_tokens(ln.strip(), GREP_LINE_TOKENS))
                 for i, ln in enumerate(lines) if rx.search(ln)]
            if m:
                per_unit.append((u, m))
        n_matched = len(per_unit)
        per_unit.sort(key=lambda t: -len(t[1]))          # most matches first (grep -c order)
        hits: "list[tuple[str, int, str]]" = []
        for u, m in per_unit:
            for line_no, text in m[:max_per_unit]:
                hits.append((u.doc_id, line_no, text))
                if len(hits) >= max_lines:
                    return hits, n_matched
        return hits, n_matched


class Grep(Tool):
    name = "grep"
    description = ("Regex search over every source file in the repository (grep -rn); returns "
                   "matching lines as path:line: text. Follow up with read on the interesting "
                   "file.")
    parameters = {"type": "object",
                  "properties": {"pattern": {"type": "string",
                                             "description": "A regular expression, e.g. def create_session_token or expir."}},
                  "required": ["pattern"]}
    needs_files = True

    def on_bind(self) -> None:
        self._rg = RegexGrep(self.units)

    def run(self, args: dict) -> str:
        args = args or {}
        pattern = args.get("pattern") or args.get("query") or ""
        hits, n_units = self._rg.search(pattern, max_lines=10)
        if not hits:
            return f"0 matches for /{pattern}/"
        lines = [f"{n_units} units matched /{pattern}/, first lines:"]
        for doc_id, line_no, text in hits:
            path = doc_id.split("::")[0]
            lines.append(f"  {path}:{line_no}: {text}")
        return "\n".join(lines)


__all__ = ["Grep", "RegexGrep", "GREP_LINE_TOKENS"]
