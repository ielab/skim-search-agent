"""The code-fix ACI: the search -> fetch instrument over a code corpus.

This is THE code agent's tool surface (the field-tagged Boolean search->fetch design). Index-free &
unchunked, exactly like the retrieval engines: a "document" is a FILE; its "parts"
(functions/methods/classes) are the AST units the corpus already carries (CodeUnit), so
nothing is persisted beyond the in-session executor/units.

Two moves:
  search(query, k) : field-tagged surface -> BQL (surface.to_bql, unmodified) ->
                            execute against the units -> matching UNITS, GROUPED BY FILE.
                            Returns a file-level candidate table: each file's matched function
                            names (its "structure"), NO bodies. Numbered for `fetch` reference.
  fetch(specs)            : specs is a LIST of (rank, part) pairs referencing the last search's
                            numbered files — grep-like: part is a function/class qualname
                            ("Command.handle") or a line range ("L810-840") IN THAT FILE. Never
                            the whole file. Returns those parts' source, aggregated across specs,
                            capped ~40 lines/part.

The agent then commits to a concrete fix (a <fix> block; see prompts/tasks/taskfix.md), scored
by agent_search/evaluation/fix_scoring.py. The retrieval engines / localization ACI are GONE for code — this
module is the code arm. (The deep-research arm keeps its own tools, in doc_research.py/doc_indri.py/doc_bm25_dci.py/doc_dci.py.)
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.bql.executor import StructuralExecutor, execute_bql
from agent_search.retrievers.structural.bql.surface import to_bql

_LINE_RANGE = re.compile(r"^\s*L?(\d+)\s*-\s*L?(\d+)\s*$", re.IGNORECASE)
_MAX_PART_LINES = 40                  # fetch cap per spec (never the whole file)
_MAX_FETCH_TOTAL = 160                # hard cap on aggregated fetch output (4 specs worth)
# the model sometimes prefixes a name with the keyword that describes it ("class Foo[def]"),
# which to_bql would turn into a two-word phrase that silently 0-hits; strip the noise word
# (tool-input normalization only; the translator itself is untouched).
_NOISE_KEYWORD = re.compile(r"(?<![\w.])(class|def|function|method)\s+(?=\S)", re.IGNORECASE)


class CodeFixWorkspace:
    """One instance's CodeUnits + raw files -> search(query) then fetch(specs).

    Constructed once per episode by AgentRetriever. `run(name, args)` is the dispatch the
    agent loop calls, mirroring the retrieval Workspace's contract (unknown tool -> a plain
    ERROR observation, never a crash)."""

    tools = ("search", "fetch")

    def __init__(self, units: Sequence[CodeUnit], files: dict,
                 executor: Optional[StructuralExecutor] = None,
                 ubyid: Optional[dict] = None):
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.files = files or {}                               # {path: source}, for L-range fetch
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.by_file: dict[str, list] = defaultdict(list)
        for u in self.units:
            self.by_file[u.path].append(u)
        self.ex = executor if executor is not None else StructuralExecutor(self.units)
        self._last_files: list[str] = []                       # rank (1-based) -> path

    # -- search: field-tagged surface -> bql -> hits grouped by file --------

    def search(self, query: str, k: int = 5) -> str:
        query = (query or "").strip()
        # strip a leading class/def/function/method noise word before to_bql (repeat until
        # stable — "def function foo[def]" has two adjacent noise words, seen in the wild).
        prev = None
        while prev != query:
            prev = query
            query = _NOISE_KEYWORD.sub("", query).strip()
        if not query:
            return "empty query"
        bql = to_bql(query, domain="code")
        obs = execute_bql(bql, self.ex, self.ubyid, k=100)
        if obs.error:
            hint = ""
            if "unknown region" in obs.error:
                hint = "\n  valid fields: def, call, string, comment, sig, file"
            return f"search: {query!r} -> {bql}\n  {obs.error}{hint}"
        if not obs.hits:
            # instrument-side recovery instead of a dead end: several UNSCOPED bare words
            # parse as one exact adjacent phrase, which real code almost never satisfies —
            # silently rerun as an OR of the words and show real hits instead of a guess.
            words = [w for w in re.findall(r"[A-Za-z_]\w{2,}", query)
                     if w.upper() not in ("AND", "OR", "NOT")]
            if len(words) > 1 and "[" not in query:
                alt = " OR ".join(words[:6])
                obs2 = execute_bql(to_bql(alt, domain="code"), self.ex, self.ubyid, k=100)
                if obs2.hits:
                    return (f"0 hits for the exact phrase {query!r}; reran as "
                            f"{alt!r}:\n" + self._render(obs2, alt, to_bql(alt, "code"), k))
            hint = ""
            try:
                hint = self.ex.suggest(bql) or ""
            except Exception:  # noqa: BLE001 — advisory only
                pass
            # keep the PRIOR non-empty file ranking fetchable: a 0-hit loosen must not wipe
            # the last good hits from under a fetch.
            prior = "  (previous results still fetchable)" if self._last_files else ""
            return (f"search: {query!r} -> {bql}  (0 hits){prior} — loosen: drop a clause, "
                     f"try a stem*, or OR alternate names."
                     + (f"\n{hint}" if hint else ""))
        return self._render(obs, query, bql, k)

    def _render(self, obs, query: str, bql: str, k: int) -> str:
        # group hits by file, preserving rank order (first-seen file = best-ranked file)
        by_file_hits: dict[str, list] = {}
        for h in obs.hits:
            by_file_hits.setdefault(h.path, []).append(h)
        top_files = list(by_file_hits.items())[:k]
        self._last_files = [path for path, _ in top_files]
        n_files_total = len(by_file_hits)
        lines = [f"search: {query!r} -> {bql}  "
                 f"({obs.n_hits} units in {n_files_total} files, top {len(top_files)}):"]
        for rank, (path, hits) in enumerate(top_files, start=1):
            all_qual = [u.qualname for u in self.by_file.get(path, [])]
            matched_qual = [self.ubyid[h.doc_id].qualname for h in hits if h.doc_id in self.ubyid]
            defs = " . ".join(matched_qual[:5]) or " . ".join(all_qual[:5])
            more = f" (+{len(all_qual) - 5} more in file)" if len(all_qual) > 5 else ""
            lines.append(f"  {rank}  {path}   defs:[{defs}]{more}")
        return "\n".join(lines)

    # -- fetch: specs = [(rank, "Qualname"), (rank, "L810-840"), ...] -------

    def fetch(self, specs) -> str:
        if not specs:
            return "ERROR: fetch needs at least one (rank, part) spec from the last search."
        if not self._last_files:
            return "ERROR: no search results yet — search() first, then fetch() a ranked file."
        # a model commonly sends ONE flat pair [1, "Command.handle"] instead of a LIST of
        # pairs [[1, "Command.handle"]] when it wants one part. Detect that shape (a 2-element
        # list whose first element is rank-like and second is NOT itself a list) and treat it
        # as a single spec rather than two malformed ones.
        if (isinstance(specs, (list, tuple)) and len(specs) == 2
                and not isinstance(specs[0], (list, tuple))
                and not isinstance(specs[1], (list, tuple))):
            try:
                int(specs[0])
                specs = [specs]
            except (TypeError, ValueError):
                pass
        parts, budget = [], _MAX_FETCH_TOTAL
        for spec in specs:
            if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                parts.append(f"ERROR: bad spec {spec!r}, expected [rank, \"part\"]")
                continue
            rank, part = spec
            try:
                rank = int(rank)
            except (TypeError, ValueError):
                parts.append(f"ERROR: rank {rank!r} is not a number")
                continue
            if not (1 <= rank <= len(self._last_files)):
                parts.append(f"ERROR: rank {rank} out of range "
                              f"(last search returned {len(self._last_files)} files)")
                continue
            path = self._last_files[rank - 1]
            text, used = self._fetch_one(path, str(part).strip(), budget)
            parts.append(f"[{rank}] {path} :: {part}\n{text}")
            budget = max(0, budget - used)
        return "\n\n".join(parts)

    def _fetch_one(self, path: str, part: str, budget: int) -> tuple[str, int]:
        m = _LINE_RANGE.match(part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            src = self.files.get(path, "")
            lines = src.splitlines()
            if not lines:
                return "ERROR: file not found or empty", 0
            s = max(1, start)
            e = min(len(lines), end, s + min(_MAX_PART_LINES, budget) - 1)
            if e < s:
                return "ERROR: empty range after cap", 0
            body = "\n".join(f"{i}: {ln}" for i, ln in enumerate(lines[s - 1:e], start=s))
            return body, e - s + 1
        # else: a function/class/method qualname — exact match first, then suffix match
        # (agent may write "handle" for "Command.handle")
        cands = [u for u in self.by_file.get(path, []) if u.qualname == part]
        if not cands:
            cands = [u for u in self.by_file.get(path, [])
                      if u.qualname == part or u.qualname.endswith("." + part)]
        if not cands:
            class_slice = self._fetch_class(path, part, budget)
            if class_slice is not None:
                return class_slice
            avail_units = self.by_file.get(path, [])
            avail = ", ".join(u.qualname for u in avail_units[:12])
            return f"ERROR: no part named {part!r} in {path}. Available: {avail}", 0
        u = cands[0]
        code_lines = (u.code or "").splitlines()
        cap = min(_MAX_PART_LINES, budget if budget > 0 else _MAX_PART_LINES)
        shown = code_lines[:cap]
        body = "\n".join(f"{u.start_line + i}: {ln}" for i, ln in enumerate(shown))
        if len(code_lines) > cap:
            body += f"\n... ({len(code_lines) - cap} more lines, {u.qualname} " \
                    f"spans {u.start_line}-{u.end_line}; fetch a narrower L-range for the rest)"
        return body, len(shown)

    def _fetch_class(self, path: str, part: str, budget: int) -> Optional[tuple[str, int]]:
        """Return an on-demand AST class slice when the unit corpus only holds methods."""
        src = self.files.get(path, "")
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            return None

        classes: list[tuple[str, ast.ClassDef]] = []

        def collect(node, parents: tuple[str, ...] = ()) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    qualname = ".".join((*parents, child.name))
                    classes.append((qualname, child))
                    collect(child, (*parents, child.name))
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    collect(child, (*parents, child.name))

        collect(tree)
        cands = [(qualname, node) for qualname, node in classes if qualname == part]
        if not cands:
            cands = [(qualname, node) for qualname, node in classes
                     if qualname.endswith("." + part) or node.name == part]
        if not cands:
            return None

        qualname, node = cands[0]
        lines = src.splitlines()
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        cap = min(_MAX_PART_LINES, budget if budget > 0 else _MAX_PART_LINES)
        shown = lines[start - 1:min(end, start + cap - 1)]
        body = "\n".join(f"{start + i}: {line}" for i, line in enumerate(shown))
        if end - start + 1 > cap:
            body += (f"\n... ({end - start + 1 - cap} more lines, class {qualname} "
                     f"spans {start}-{end}; fetch a narrower L-range for the rest)")
        return body, len(shown)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name == "search":
                q = args.get("query") or args.get("q") or ""
                k = int(args.get("k", 5) or 5)
                return self.search(q, k)
            if name == "fetch":
                specs = args.get("specs") or args.get("parts") or []
                # a model very commonly flattens a single spec's fields to the top level
                # ({"rank":2,"part":"X"}) instead of nesting it in `specs`. Accept that too.
                if not specs and ("rank" in args or "part" in args):
                    specs = [args]
                if isinstance(specs, dict):
                    specs = [specs]
                norm = []
                for s in specs:
                    if isinstance(s, dict):
                        norm.append((s.get("rank") or s.get("doc") or s.get("file"),
                                     s.get("part") or s.get("name") or s.get("range")))
                    else:
                        norm.append(s)
                return self.fetch(norm)
        except Exception as e:  # noqa: BLE001 — a tool error is an observation, not a crash
            return f"ERROR: {type(e).__name__}: {e}"
        return (f"ERROR: unknown tool {name!r}. Available tools: "
                f"{', '.join(self.tools)}.")
