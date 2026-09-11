"""`read`: a line-range read, either over the DCI export directory or the raw repo files.

`source="export"` (default) is a 1-indexed line-range of one file under the DCI directory
(`state.scratch["dci_dir"]`, shared with `bash` and, for the bounded strategy, `bm25_search`);
paths may not escape that directory. Every exported filename mentioned in the requested path
is surfaced (added to `state.seen`).

`source="repo"` is a plain line-range slice of a raw repository file (`self.files`), capped
at 80 lines, with a bare-basename fallback when the exact path isn't found. This mode does
not touch `state.seen`: the `codefix_grep` strategy is scored on a committed `<fix>`, not on
gold-document coverage.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agent_search.core.tokens import cap_tokens
from agent_search.tools.bash.tool import _surface_from_text
from agent_search.tools.base import Tool

READ_DEFAULT_LIMIT = 2000
READ_MAX_LINE_TOKENS = int(os.environ.get("READ_MAX_LINE_TOKENS", "400"))


def _run_read(corpus_dir: Path, rel: str, offset, limit,
             default_limit: int, max_line_tokens: int) -> str:
    """Read a 1-indexed line-range of one exported file; rejects paths escaping corpus_dir."""
    if not rel:
        return "Error: read called with empty path."
    while rel.startswith("./"):
        rel = rel[2:]
    root_resolved = corpus_dir.resolve()
    candidate = Path(rel)
    target = (candidate.resolve() if candidate.is_absolute()
              else (root_resolved / candidate).resolve())
    if not target.is_relative_to(root_resolved):
        return f"Error: path {rel!r} escapes the corpus root — use a relative path."
    if not target.exists():
        return f"Error: file not found: {rel!r}. Use `bash` (ls/rg -l) to find the exact filename first."
    if not target.is_file():
        return f"Error: not a regular file: {rel!r}"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"Error: read failed: {type(e).__name__}: {e}"

    all_lines = text.split("\n")
    total_lines = len(all_lines)
    try:
        offset = int(offset) if offset is not None else 1
    except (TypeError, ValueError):
        offset = 1
    try:
        limit = int(limit) if limit is not None else default_limit
    except (TypeError, ValueError):
        limit = default_limit
    offset, limit = max(1, offset), max(1, limit)

    start = offset - 1
    if start >= total_lines:
        return f"Error: offset {offset} is beyond end of file ({total_lines} lines total)."
    end = min(start + limit, total_lines)

    formatted = []
    for line in all_lines[start:end]:
        n_tok = len(line.split())
        if n_tok > max_line_tokens:
            cut = n_tok - max_line_tokens
            line = cap_tokens(line, max_line_tokens, f"...[line truncated; {cut} tokens]")
        formatted.append(line)
    out = "\n".join(formatted)
    if end < total_lines:
        out += f"\n\n[Showing lines {offset}-{end} of {total_lines}. Use offset={end + 1} to continue.]"
    return out


class Read(Tool):
    name = "read"
    description = ("Read a file by line range. Code: a repo file path, up to ~80 lines per "
                   "call. Docs (DCI baseline): a path relative to the corpus directory, offset "
                   "(1-indexed line, default 1) and limit (default 2000 lines) page through "
                   "long files.")
    parameters = {"type": "object",
                  "properties": {
                      "path": {"type": "string",
                               "description": "File path (code: repo-relative; docs: relative to the corpus directory)."},
                      "start": {"type": "integer", "description": "Code arm: first line to read."},
                      "end": {"type": "integer", "description": "Code arm: last line to read."},
                      "offset": {"type": "integer",
                                "description": "Doc DCI arm: 1-indexed line number to start reading from (default 1)."},
                      "limit": {"type": "integer",
                               "description": "Doc DCI arm: maximum number of lines to read (default 2000)."}},
                  "required": ["path"]}

    # "export": the DCI directory (state.scratch["dci_dir"]). "repo": the raw repository
    # files (self.files), used by the codefix_grep strategy.
    source: str = "export"

    def __init__(self, name: Optional[str] = None, **options):
        super().__init__(name=name, **options)
        self.needs_files = self.source == "repo"

    def run(self, args: dict) -> str:
        args = args or {}
        if self.source == "repo":
            path = args.get("path") or args.get("file") or ""
            return self._read_repo(path, args.get("start"), args.get("end"))
        path = args.get("path") or args.get("file_path") or ""
        return self._read_export(path, args.get("offset"), args.get("limit"))

    def _read_export(self, path: str, offset=None, limit=None) -> str:
        rel = (path or "").strip()
        corpus_dir = self.state.scratch.get("dci_dir")
        rel_to_doc = self.state.scratch.get("dci_rel_to_doc", {})
        _surface_from_text(self.state, rel_to_doc, rel)
        obs = _run_read(corpus_dir, rel, offset, limit, READ_DEFAULT_LIMIT, READ_MAX_LINE_TOKENS)
        if not obs.startswith("Error"):
            _surface_from_text(self.state, rel_to_doc, rel)
        return obs

    def _read_repo(self, path: str, start=None, end=None) -> str:
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


__all__ = ["Read", "_run_read", "READ_DEFAULT_LIMIT", "READ_MAX_LINE_TOKENS"]
