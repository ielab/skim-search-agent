"""The pre-0.3 deep-research DCI baseline: `bash` (grep/rg/ls/etc) plus `read` (a file line-range).

Kept so the parity tests can compare against it. The current equivalent is the `dci` strategy
in `agent_search/strategies/dci.py`, using the `bash` and `read` tools.

DCI, Direct Corpus Interaction (`chen2026dci`, RISE's brute-force reference arm), gives the
agent no retriever at all: a flat file tree (`agent_search.corpus.flat_export`) and two
shell-shaped tools. The agent must grep for candidate files itself, then read line-ranges,
then `<answer>`. This is the accurate but token-expensive reference the structured
(`research`) and retrieve-then-visit (`research_bm25`) arms are read against on cost, not just
accuracy (RISE's own headline: comparable accuracy to a trained structured agent at a
fraction of its cost).

Two tools, uncoached (no manual: a shell needs no teaching):
  bash(command)             : run a bash command (grep/rg/ls/find/wc/...) with cwd = the
                              export dir; output combines stdout+stderr, TAIL-truncated to
                              ~2000 lines / ~12000 whitespace tokens (RISE's `truncateTail`
                              default, adapted to SkimSearchAgent's one-token-ruler-for-every-cap
                              convention; see agent_search.core.tokens).
  read(path, offset, limit) : read a 1-indexed line-range of one exported file (default
                              limit 2000 lines); paths are resolved relative to the export
                              dir and may not escape it.

`seen` accumulates every doc_id the episode surfaced: a file `read` directly, or any
``<doc_id>.txt`` filename that appears in a bash command or its output (a `grep -l` /
`rg -l` hit), for the same gold-doc-coverage metric the other doc arms report.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from agent_search.legacy.workspaces.common import _SeenMixin
from agent_search.core.seen import OrderedSeen
from agent_search.core.tokens import cap_tokens, count_ws_tokens
from agent_search.corpus.flat_export import export_flat_corpus
from agent_search.corpus.units import CodeUnit

# RISE tools.py defaults (PI_BASH_DEFAULT_MAX_LINES / PI_BASH_DEFAULT_MAX_BYTES /
# PI_READ_DEFAULT_MAX_LINES) — kept identical in SPIRIT (same shape: a line cap plus a size cap),
# but the size caps are WHITESPACE TOKENS, not bytes/characters — SkimSearchAgent measures and
# caps text in tokens everywhere (agent_search.core.tokens), never characters.
BASH_MAX_LINES = 2000
BASH_MAX_TOKENS = int(os.environ.get("BASH_MAX_TOKENS", "12000"))
READ_DEFAULT_LIMIT = 2000
READ_MAX_LINE_TOKENS = int(os.environ.get("READ_MAX_LINE_TOKENS", "400"))
_HARD_TIMEOUT_S = 60.0     # per-subprocess safety ceiling (catastrophic-regex guard)


def _tail_truncate(content: str, *, max_lines: int = BASH_MAX_LINES,
                   max_tokens: int = BASH_MAX_TOKENS) -> str:
    """Keep the LAST N lines or M whitespace tokens (whichever limit hits first); append a
    one-line `[Truncated: showing X of Y lines]` marker when truncation fires. Adapted from
    an internal prototype's `_tail_truncate` (self-contained here — no cross-import),
    with the original byte-count size limit replaced by SkimSearchAgent's
    whitespace-token ruler (agent_search.core.tokens) — see BASH_MAX_TOKENS."""
    total_tokens = count_ws_tokens(content)
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_tokens <= max_tokens:
        return content

    out_lines: list[str] = []
    out_tokens = 0
    truncated_by = "lines"
    for i in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            truncated_by = "lines"
            break
        line = lines[i]
        line_tokens = count_ws_tokens(line)
        if out_tokens + line_tokens > max_tokens:
            truncated_by = "tokens"
            if not out_lines:
                # keep the TAIL `max_tokens` whitespace tokens of this single oversized line.
                toks = line.split()
                kept = toks[-max_tokens:] if max_tokens > 0 else []
                out_lines.insert(0, " ".join(kept))
                out_tokens = len(kept)
            break
        out_lines.insert(0, line)
        out_tokens += line_tokens

    out = "\n".join(out_lines)
    if truncated_by == "lines":
        warning = f"\n[Truncated: showing {len(out_lines)} of {total_lines} lines]"
    else:
        warning = f"\n[Truncated: {len(out_lines)} lines shown ({max_tokens} token limit)]"
    return out + warning


def _run_bash(corpus_dir: Path, command: str, timeout_s: float,
             max_lines: int, max_tokens: int) -> str:
    """`bash -c <command>` with cwd=corpus_dir; combined stdout+stderr, tail-truncated. Exit 1 +
    empty output is annotated "(no matches found)" (grep/rg convention) so the agent can tell a
    clean miss from a crash."""
    proc = None
    try:
        proc = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=str(corpus_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return (f"Error: bash subprocess timed out after {timeout_s:.0f}s (SIGKILL'd). "
                    f"Try a simpler search: avoid catastrophic regexes; prefer plain patterns "
                    f"or multiple narrower searches.")
    except FileNotFoundError as e:
        return f"Error: failed to spawn bash: {e}"
    except Exception as e:  # noqa: BLE001
        return f"Error: bash subprocess exception: {type(e).__name__}: {e}"
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    stdout = (stdout or "").rstrip("\n")
    stderr = (stderr or "").rstrip("\n")
    full = (stdout + "\n" + stderr) if (stdout and stderr) else (stdout or stderr)
    if not full:
        if proc.returncode == 0:
            full = "(command succeeded, no output)"
        elif proc.returncode == 1:
            full = "(no matches found)"
        else:
            full = f"(no output; exit={proc.returncode})"
    elif proc.returncode != 0:
        full += f"\n\nCommand exited with code {proc.returncode}"
    return _tail_truncate(full, max_lines=max_lines, max_tokens=max_tokens)


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


class DciWorkspace(_SeenMixin):
    """`bash(command)` -> shell over the flat export dir; `read(path, offset, limit)` -> a
    line-range. NO retriever, NO search/fetch — the agent finds candidate files itself.

    `seen` accumulates every doc_id surfaced: a file directly `read`, plus any exported
    filename mentioned in a bash COMMAND or its OUTPUT (an `rg -l` / `grep -rl` hit), for
    the gold-doc-coverage metric (`evaluation/doc_scoring.gold_doc_coverage`)."""

    tools = ("bash", "read")

    def __init__(self, units: Sequence[CodeUnit], corpus_key: Optional[str] = None,
                *, max_bash_lines: int = BASH_MAX_LINES, max_bash_tokens: int = BASH_MAX_TOKENS,
                read_default_limit: int = READ_DEFAULT_LIMIT,
                read_max_line_tokens: int = READ_MAX_LINE_TOKENS):
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.corpus_dir, doc_to_rel = export_flat_corpus(self.units, key=corpus_key)
        self._rel_to_doc = {rel: doc_id for doc_id, rel in doc_to_rel.items()}
        self._max_bash_lines = max_bash_lines
        self._max_bash_tokens = max_bash_tokens
        self._read_default_limit = read_default_limit
        self._read_max_line_tokens = read_max_line_tokens
        self.seen = OrderedSeen()
        self.n_bash = 0
        self.n_read = 0

    def _surface_from_text(self, blob: str) -> None:
        for rel, doc_id in self._rel_to_doc.items():
            if rel in blob:
                self.seen.add(doc_id)

    def bash(self, command: str, timeout=None) -> str:
        command = (command or "").strip()
        if not command:
            return "Error: bash called with empty command."
        try:
            timeout_s = float(timeout) if timeout is not None else _HARD_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = _HARD_TIMEOUT_S
        timeout_s = max(1.0, min(timeout_s, _HARD_TIMEOUT_S))
        obs = _run_bash(self.corpus_dir, command, timeout_s,
                        self._max_bash_lines, self._max_bash_tokens)
        self._surface_from_text(command)
        self._surface_from_text(obs)
        return obs

    def read(self, path: str, offset=None, limit=None) -> str:
        rel = (path or "").strip()
        self._surface_from_text(rel)
        obs = _run_read(self.corpus_dir, rel, offset, limit,
                        self._read_default_limit, self._read_max_line_tokens)
        if not obs.startswith("Error"):
            self._surface_from_text(rel)
        return obs

    def run(self, name: str, args: dict) -> str:
        # NO repeat-dedup nudge: DCI is RISE's brute-force baseline (chen2026dci) and must
        # behave like it. The method (research/research_bm25) gets no "you already made this
        # call" hint, so neither may the baseline it is measured against on cost.
        args = args or {}
        try:
            if name == "bash":
                self.n_bash += 1
                return self.bash(args.get("command") or "", args.get("timeout"))
            if name == "read":
                self.n_read += 1
                path = args.get("path") or args.get("file_path") or ""
                return self.read(path, args.get("offset"), args.get("limit"))
        except Exception as e:  # noqa: BLE001 — a tool error is an observation, not a crash
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r} for the dci arm (use bash or read)."


# --- arms
from agent_search.legacy.retriever import register_workspace  # noqa: E402

register_workspace("dci", tools=("bash", "read"), builder=lambda ctx: DciWorkspace(ctx.units, corpus_key=ctx.corpus_key))
