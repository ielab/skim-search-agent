"""`bash`: run a shell command over the exported flat corpus directory.

With `bounded=False` the command runs over a flat export of the whole corpus (one file per
doc, `agent_search.corpus.flat_export`), cached by `corpus_key` and never removed here. With
`bounded=True` the command runs over a per-episode staging directory that starts empty and is
filled incrementally by the `bm25_search` tool (`agent_search.tools.search_bm25_dci`) as it
stages new hits; that directory is removed once the episode state is garbage-collected.

Output combines stdout and stderr, tail-truncated to ~2000 lines or `BASH_MAX_TOKENS`
whitespace tokens, never a byte or character cap (see `agent_search.tokens`). Every
exported filename mentioned in the command or its output is surfaced (added to
`state.seen`), the gold-document coverage signal the `dci` and `bounded_dci` strategies
report.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import weakref
from pathlib import Path
from typing import Optional

from agent_search.tokens import count_ws_tokens
from agent_search.corpus.flat_export import export_flat_corpus
from agent_search.tools.base import Tool

# Defaults modeled on RISE's tools.py (PI_BASH_DEFAULT_MAX_LINES / PI_BASH_DEFAULT_MAX_BYTES):
# a line cap plus a size cap, except the size cap here counts whitespace tokens, not bytes.
BASH_MAX_LINES = 2000
BASH_MAX_TOKENS = int(os.environ.get("BASH_MAX_TOKENS", "12000"))
_HARD_TIMEOUT_S = 60.0     # per-subprocess safety ceiling (catastrophic-regex guard)


def _tail_truncate(content: str, *, max_lines: int = BASH_MAX_LINES,
                   max_tokens: int = BASH_MAX_TOKENS) -> str:
    """Keep the last N lines or M whitespace tokens, whichever limit hits first; append a
    one-line `[Truncated: showing X of Y lines]` marker when truncation fires."""
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
                # keep the last `max_tokens` whitespace tokens of this single oversized line.
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


def get_dci_dir(state, units, corpus_key: Optional[str], bounded: bool):
    """Get or create the episode's export/staging directory and its {rel: doc_id} map, once
    per episode, shared via `state.scratch["dci_dir"]` / `state.scratch["dci_rel_to_doc"]` so
    `bash`, `read` and (bounded) `bm25_search` all see the same directory regardless of bind
    order.

    Unbounded: the whole corpus, exported once and cached by `corpus_key`
    (`agent_search.corpus.flat_export.export_flat_corpus`), never removed here. Bounded: a
    fresh, empty per-episode tempdir that `bm25_search` fills incrementally; a
    `weakref.finalize` on the episode `state` removes it once the episode is
    garbage-collected.
    """
    if "dci_dir" in state.scratch:
        return state.scratch["dci_dir"], state.scratch["dci_rel_to_doc"]
    if bounded:
        d = Path(tempfile.mkdtemp(prefix="agent_search_bm25dci_"))
        rel_to_doc: dict = {}
        weakref.finalize(state, shutil.rmtree, str(d), True)
    else:
        d, doc_to_rel = export_flat_corpus(units, key=corpus_key)
        rel_to_doc = {rel: doc_id for doc_id, rel in doc_to_rel.items()}
    state.scratch["dci_dir"] = d
    state.scratch["dci_rel_to_doc"] = rel_to_doc
    return d, rel_to_doc


def _surface_from_text(state, rel_to_doc, blob: str) -> None:
    for rel, doc_id in rel_to_doc.items():
        if rel in blob:
            state.seen.add(doc_id)


class Bash(Tool):
    name = "bash"
    description = ("Run a bash command (grep/rg/ls/find/wc) in the corpus directory to search "
                   "for candidate files. The corpus directory contains only the documents "
                   "available to you this episode (with bm25_search: only its top-ranked hits) "
                   "— there is no broader search/retrieval tool. Output is truncated to the "
                   "last ~2000 lines or ~50KB, whichever is hit first.")
    parameters = {"type": "object",
                  "properties": {"command": {"type": "string",
                                             "description": "A bash command, e.g. rg -l 'term' . or grep -rn 'term' ."},
                                 "timeout": {"type": "number",
                                            "description": "Timeout in seconds (optional, default 60)."}},
                  "required": ["command"]}

    # False: bash runs over the whole corpus export. True: bash runs over a staging dir
    # grown by bm25_search that holds only what has been retrieved so far this episode.
    bounded: bool = False

    def on_bind(self) -> None:
        get_dci_dir(self.state, self.units, self.corpus_key, self.bounded)

    def run(self, args: dict) -> str:
        args = args or {}
        command = (args.get("command") or "").strip()
        if not command:
            return "Error: bash called with empty command."
        timeout = args.get("timeout")
        try:
            timeout_s = float(timeout) if timeout is not None else _HARD_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = _HARD_TIMEOUT_S
        timeout_s = max(1.0, min(timeout_s, _HARD_TIMEOUT_S))
        corpus_dir, rel_to_doc = get_dci_dir(self.state, self.units, self.corpus_key, self.bounded)
        obs = _run_bash(corpus_dir, command, timeout_s, BASH_MAX_LINES, BASH_MAX_TOKENS)
        _surface_from_text(self.state, rel_to_doc, command)
        _surface_from_text(self.state, rel_to_doc, obs)
        return obs


__all__ = ["Bash", "get_dci_dir", "_surface_from_text", "_run_bash", "_tail_truncate",
           "BASH_MAX_LINES", "BASH_MAX_TOKENS"]
