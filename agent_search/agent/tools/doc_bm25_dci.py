"""The bm25 -> DCI ACI: `bm25_search` is now a LIVE, per-call BM25 retrieval (identical in
shape to `Bm25Visit.search`), and `bash`/`read` (the DCI shell) are rooted at a staging dir
that GROWS INCREMENTALLY as new docs are surfaced by search.

This is the missing controlled comparison between the doc arm's three baselines:

  research       : structured search  -> fetch a named SECTION      (the method)
  research_bm25  : BM25 top-k search  -> visit the WHOLE doc          (retrieve-then-visit)
  research_dci   : NO retriever, bash/read over the WHOLE corpus       (brute-force, chen2026dci)
  research_bm25_dci (this file) : BM25 search (SAME engine/query semantics as `research_bm25`)
                    -> bash/read (SAME shell as research_dci), but bounded to whatever bm25 has
                    surfaced SO FAR this episode.

Holding retrieval identical to `research_bm25` (same `BM25Local`, same live per-query search)
and swapping only the READ strategy (shell-grep vs whole-doc-visit) isolates "given the same
retrieval, how do you read?" — the other three baselines each change BOTH retrieval and read
together.

BUGFIX (was one-shot): retrieval used to run ONCE in `__init__` on the raw episode question,
and `bm25_search` tool calls just replayed that fixed ranking — the agent's own queries were
silently ignored for retrieval, which starved this arm of documents a live searcher would find
(measured: ~5% gold-doc surfacing on browsecomp vs ~59% for `Bm25Visit`, which DOES re-run bm25
per call). Retrieval is now LIVE per call, exactly like `Bm25Visit.search`:

  1. `__init__` still runs ONE bm25 call up front — `self.bm.search(query, k=topk)` on the
     episode's question text — to seed a starting pool (some episodes' first bash/read call
     assumes a non-empty corpus_dir at t=0). This is not special-cased: it is implemented by
     calling `self.search(query)`, the SAME method a later tool call uses.
  2. Every `bm25_search` call (including that seed one) runs `self.bm.search(query, k)` LIVE
     against the agent's actual query text — the SAME `BM25Local` engine `Bm25Visit` uses, so
     retrieval is byte-identical to `research_bm25` for the same query/engine/k.
  3. Any hit not already staged gets written into the SAME staging dir immediately, via
     `flat_export.stage_units_into` (the exact `<safe_doc_id>.txt` / `title\\n\\nbody` writer
     `export_flat_corpus` uses — factored out so there is ONE write path, not two). A doc that
     drops out of a later ranking stays staged: once surfaced, always readable, same as `seen`.
  4. `bash`/`read` (imported from `doc_dci` — no duplicated subprocess/truncation logic) operate
     with cwd/root = the staging dir. What's on disk only ever GROWS across the episode; nothing
     is ever unstaged.

doc_id -> filename: identical to `doc_dci`/`flat_export` (`_safe_filename`), so no new mapping
scheme — `stage_units_into`'s returned `{doc_id: relpath}` is merged into `_rel_to_doc` directly,
exactly like `DciWorkspace._rel_to_doc`.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from agent_search.agent.tools.doc_dci import (
    BASH_MAX_BYTES, BASH_MAX_LINES, READ_DEFAULT_LIMIT, READ_MAX_LINE_CHARS,
    _HARD_TIMEOUT_S, _run_bash, _run_read)
from agent_search.corpus.flat_export import stage_units_into
from agent_search.agent.tools.doc_research import opening_line
from agent_search.corpus.units import CodeUnit

# The retrieval-stage cutoff: how many bm25 hits a `bm25_search` call surfaces/stages by
# default. A FIXED constant (not the CLI's --k sweep, which sizes the episode's location-ranking
# cutoff, not the per-call retrieval width this arm's search box uses) — mirrors the sandbox
# prototype's `BOTH_TOPK`.
BM25_DCI_TOPK = int(os.environ.get("BM25_DCI_TOPK", "10"))


class Bm25DciWorkspace:
    """`bm25_search` runs BM25 LIVE on every call (see module docstring) — a new query text
    genuinely re-retrieves, matching `Bm25Visit.search`'s observation shape exactly. `bash`/
    `read` are `DciWorkspace`'s shell, rooted at a staging dir that starts with the episode
    question's top-k and grows by one batch per subsequent search call.

    `.seen` accumulates every doc_id surfaced: every bm25 hit that gets rendered in a `search`
    observation (same moment `Bm25Visit.search` marks a hit seen), plus anything the shell
    actually greps/reads — same gold-doc-coverage contract as every other doc workspace.
    """

    # tools.yaml's `bm25_dci` toolset names the search tool `bm25_search` (matching
    # `research_bm25`'s spelling, since retrieval is the same operation); `run()` also
    # accepts the bare `search` alias for the same reason `Bm25Visit.run` does.
    tools = ("bm25_search", "bash", "read")

    def __init__(self, units: Sequence[CodeUnit], query: str, engine=None,
                ubyid: Optional[dict] = None, topk: int = BM25_DCI_TOPK,
                *, max_bash_lines: int = BASH_MAX_LINES, max_bash_bytes: int = BASH_MAX_BYTES,
                read_default_limit: int = READ_DEFAULT_LIMIT,
                read_max_line_chars: int = READ_MAX_LINE_CHARS):
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if engine is None:
            # env BM25_BACKEND-selectable (default 'local', unchanged) — SAME fallback
            # Bm25Visit/Bm25FetchWorkspace use (doc_research.py); see
            # agent_search.retrievers.lexical.build_bm25_engine. Production callers
            # (agent_search.agent.retriever) always pass `engine` explicitly.
            from agent_search.retrievers.lexical import build_bm25_engine
            engine = build_bm25_engine(self.units)
        self.bm = engine
        self.topk = topk
        self._max_bash_lines = max_bash_lines
        self._max_bash_bytes = max_bash_bytes
        self._read_default_limit = read_default_limit
        self._read_max_line_chars = read_max_line_chars
        self.seen: set = set()
        self.query = ""
        self.last_hits: list[str] = []

        # A fresh, EMPTY staging dir (no `key` -> always a brand-new tempdir per episode, same
        # scheme `export_flat_corpus`'s un-keyed path used to use — never the corpus-wide cached
        # export `DciWorkspace` reuses). Every search — this seed one AND every later live call —
        # stages into THIS SAME dir, so bash/read's view grows monotonically over the episode.
        self.corpus_dir = Path(tempfile.mkdtemp(prefix="agent_search_bm25dci_"))
        self._rel_to_doc: dict[str, str] = {}
        self._staged: set = set()

        # SEED: one live bm25 call on the episode's question text, via the SAME `search()` a
        # tool call uses — not special-cased staging logic. A blank question yields no seed hits
        # and an empty corpus_dir, exactly like a blank first `bm25_search` call would.
        self.search(query, k=topk)

    # -- bm25_search: LIVE per call, same engine/shape as Bm25Visit.search --------------

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        self.query = query
        ids = list(self.bm.search(query, k=k or self.topk))
        self.last_hits = ids
        if not ids:
            return f"search: {query}   (0 matches)"

        lines = [f"search: {query}   ({len(ids)} matches):"]
        new_units = []
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            if doc_id not in self._staged:
                new_units.append(u)
            snip = opening_line(u)
            lines.append(f"  {rank}  {doc_id}  {(u.title or u.qualname or '')!r}  {snip}…")

        # INCREMENTAL staging: only docs not already on disk get written, via the SAME writer
        # `export_flat_corpus` uses (`stage_units_into`) — a doc surfaced by an earlier call
        # stays staged (never rewritten, never removed), so bash/read's view only ever grows.
        if new_units:
            new_map = stage_units_into(self.corpus_dir, new_units)
            self._rel_to_doc.update({rel: doc_id for doc_id, rel in new_map.items()})
            self._staged.update(new_map)

        return "\n".join(lines)

    # -- bash / read: DciWorkspace's shell, rooted at the (growing) staging dir ---------

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
                        self._max_bash_lines, self._max_bash_bytes)
        self._surface_from_text(command)
        self._surface_from_text(obs)
        return obs

    def read(self, path: str, offset=None, limit=None) -> str:
        rel = (path or "").strip()
        self._surface_from_text(rel)
        obs = _run_read(self.corpus_dir, rel, offset, limit,
                        self._read_default_limit, self._read_max_line_chars)
        if not obs.startswith("Error"):
            self._surface_from_text(rel)
        return obs

    def run(self, name: str, args: dict) -> str:
        # NO repeat-dedup nudge, matching DciWorkspace: this arm's shell stage is the SAME
        # brute-force baseline (chen2026dci), just retrieval-bounded — it must not get an
        # anti-loop hint the pure-DCI arm doesn't also get, or the cost comparison is unfair.
        args = args or {}
        try:
            if name in ("bm25_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "bash":
                return self.bash(args.get("command") or "", args.get("timeout"))
            if name == "read":
                path = args.get("path") or args.get("file_path") or ""
                return self.read(path, args.get("offset"), args.get("limit"))
        except Exception as e:  # noqa: BLE001 — a tool error is an observation, not a crash
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."
