"""The code-fix ACI: BQL search over code units, plus fetch of functions.

Ported from `agent_search.agent.tools.code_fix.CodeFixWorkspace`, logic unchanged. A
"document" is a FILE; its "parts" (functions/methods/classes) are the AST units the corpus
already carries (CodeUnit).

Two tools:
  `SearchCode` (name "search")  -- field-tagged surface -> BQL (surface.to_bql, domain=
                            "code") -> execute against the units -> matching UNITS, GROUPED
                            BY FILE. Returns a file-level candidate table: each file's
                            matched function names (its "structure"), NO bodies. Numbered
                            for `FetchCode` reference (`state.last_hits` holds file PATHS,
                            not doc_ids).
  `FetchCode` (name "fetch")    -- `specs` is a LIST of (rank, part) pairs referencing the
                            last search's numbered files: `part` is a function/class
                            qualname ("Command.handle") or a line range ("L810-840") IN THAT
                            FILE. Never the whole file. Returns those parts' source,
                            aggregated across specs, capped ~40 lines/part.

Neither tool tracks `state.seen`: the old workspace never did either (the code arm's
episodes are scored by fix correctness, not by a retrieval ranking metric), so porting it
byte for byte means `state.seen`/`surfaced` stay untouched here.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Optional

from agent_search.retrievers.bql.executor import execute_bql
from agent_search.retrievers.bql.surface import to_bql

from agent_search.tools.base import Tool

_LINE_RANGE = re.compile(r"^\s*L?(\d+)\s*-\s*L?(\d+)\s*$", re.IGNORECASE)
_MAX_PART_LINES = 40                  # fetch cap per spec (never the whole file)
_MAX_FETCH_TOTAL = 160                # hard cap on aggregated fetch output (4 specs worth)
# the model sometimes prefixes a name with the keyword that describes it ("class Foo[def]"),
# which to_bql would turn into a two-word phrase that silently 0-hits; strip the noise word.
_NOISE_KEYWORD = re.compile(r"(?<![\w.])(class|def|function|method)\s+(?=\S)", re.IGNORECASE)


class SearchCode(Tool):
    """search(query) -> BQL over the code units, hits grouped by file, numbered for
    `FetchCode`. `state.last_hits` holds the ranked file PATHS (the code arm's "doc_id" is a
    file, not a unit)."""

    name = "search"
    # the exact text the paper's code prompts showed for `search`; the manual teaches the syntax
    description = ("Search the collection and return the top results with their DocID, title and "
                   "snippet. Documents: a keyword query. Code repositories: a Boolean field-tagged "
                   "query over code units (def, call, sig, comment, string, path, body; AND/OR/NOT, "
                   "phrases, wildcards), returning ranked files with the matched function names.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "string",
                                           "description": "The search query (plain keywords, or a Boolean query for code)."},
                                 "k": {"type": "integer",
                                      "description": "Number of results to list (code search only; default 5)."}},
                  "required": ["query"]}
    manual = "bql_code.md"
    engines = ("bql_plain",)

    def on_bind(self) -> None:
        self.by_file: dict = defaultdict(list)
        for u in self.units:
            self.by_file[u.path].append(u)

    def search(self, query: str, k: int = 5) -> str:
        query = (query or "").strip()
        # strip a leading class/def/function/method noise word before to_bql (repeat until
        # stable -- "def function foo[def]" has two adjacent noise words, seen in the wild).
        prev = None
        while prev != query:
            prev = query
            query = _NOISE_KEYWORD.sub("", query).strip()
        if not query:
            return "empty query"
        bql = to_bql(query, domain="code")
        obs = execute_bql(bql, self.engine["bql_plain"], self.ubyid, k=100)
        if obs.error:
            hint = ""
            if "unknown region" in obs.error:
                hint = "\n  valid fields: def, call, string, comment, sig, file"
            return f"search: {query!r} -> {bql}\n  {obs.error}{hint}"
        if not obs.hits:
            # instrument-side recovery instead of a dead end: several UNSCOPED bare words
            # parse as one exact adjacent phrase, which real code almost never satisfies --
            # silently rerun as an OR of the words and show real hits instead of a guess.
            words = [w for w in re.findall(r"[A-Za-z_]\w{2,}", query)
                     if w.upper() not in ("AND", "OR", "NOT")]
            if len(words) > 1 and "[" not in query:
                alt = " OR ".join(words[:6])
                obs2 = execute_bql(to_bql(alt, domain="code"), self.engine["bql_plain"],
                                   self.ubyid, k=100)
                if obs2.hits:
                    return (f"0 hits for the exact phrase {query!r}; reran as "
                            f"{alt!r}:\n" + self._render(obs2, alt, to_bql(alt, "code"), k))
            hint = ""
            try:
                hint = self.engine["bql_plain"].suggest(bql) or ""
            except Exception:  # noqa: BLE001 -- advisory only
                pass
            # keep the PRIOR non-empty file ranking fetchable: a 0-hit loosen must not wipe
            # the last good hits from under a fetch.
            prior = "  (previous results still fetchable)" if self.state.last_hits else ""
            return (f"search: {query!r} -> {bql}  (0 hits){prior} — loosen: drop a clause, "
                     f"try a stem*, or OR alternate names."
                     + (f"\n{hint}" if hint else ""))
        return self._render(obs, query, bql, k)

    def _render(self, obs, query: str, bql: str, k: int) -> str:
        # group hits by file, preserving rank order (first-seen file = best-ranked file)
        by_file_hits: dict = {}
        for h in obs.hits:
            by_file_hits.setdefault(h.path, []).append(h)
        top_files = list(by_file_hits.items())[:k]
        self.state.last_hits = [path for path, _ in top_files]
        n_files_total = len(by_file_hits)
        lines = [f"search: {query!r} -> {bql}  "
                 f"({obs.n_hits} units in {n_files_total} files, top {len(top_files)}):"]
        for rank, (path, hits) in enumerate(top_files, start=1):
            all_qual = [u.qualname for u in self.by_file.get(path, [])]
            matched_qual = [self.ubyid[h.doc_id].qualname for h in hits if h.doc_id in self.ubyid]
            defs = " . ".join(matched_qual[:5]) or " . ".join(all_qual[:5])
            more = f" (+{len(all_qual) - 5} more in file)" if len(all_qual) > 5 else ""
            self.state.listing[path] = {"matched_qual": matched_qual, "all_qual": all_qual}
            lines.append(f"  {rank}  {path}   defs:[{defs}]{more}")
        return "\n".join(lines)

    def run(self, args: dict) -> str:
        q = args.get("query") or args.get("q") or ""
        k = int(args.get("k", 5) or 5)
        return self.search(q, k)


class FetchCode(Tool):
    """fetch(specs) -- specs = [(rank, "Qualname"), (rank, "L810-840"), ...], `rank`
    referencing `state.last_hits` (file paths from the last `SearchCode` call). Reads the
    raw repository files for line-range specs and for the on-demand AST class slice."""

    name = "fetch"
    description = ("Pull a specific part of a candidate a previous search ranked — code: a "
                   "function/method name or a line range like L1-40; docs: a named section or the "
                   "infobox. Never the whole file/document.")
    parameters = {"type": "object",
                  "properties": {
                      "specs": {"type": "array",
                                "description": "List of [rank, part] pairs referencing the last "
                                               "search's numbering; part is a name from that "
                                               "candidate's structure list (or an L-range for code).",
                                "items": {"type": "array"}}},
                  "required": ["specs"]}
    needs_files = True

    def on_bind(self) -> None:
        self.by_file: dict = defaultdict(list)
        for u in self.units:
            self.by_file[u.path].append(u)

    def fetch(self, specs) -> str:
        if not specs:
            return "ERROR: fetch needs at least one (rank, part) spec from the last search."
        if not self.state.last_hits:
            return "ERROR: no search results yet — search() first, then fetch() a ranked file."
        # a model commonly sends ONE flat pair [1, "Command.handle"] instead of a LIST of
        # pairs [[1, "Command.handle"]] when it wants one part.
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
            if not (1 <= rank <= len(self.state.last_hits)):
                parts.append(f"ERROR: rank {rank} out of range "
                              f"(last search returned {len(self.state.last_hits)} files)")
                continue
            path = self.state.last_hits[rank - 1]
            text, used = self._fetch_one(path, str(part).strip(), budget)
            parts.append(f"[{rank}] {path} :: {part}\n{text}")
            budget = max(0, budget - used)
        return "\n\n".join(parts)

    def _fetch_one(self, path: str, part: str, budget: int) -> tuple:
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
        # else: a function/class/method qualname -- exact match first, then suffix match
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

    def _fetch_class(self, path: str, part: str, budget: int) -> Optional[tuple]:
        """Return an on-demand AST class slice when the unit corpus only holds methods."""
        src = self.files.get(path, "")
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            return None

        classes: list = []

        def collect(node, parents: tuple = ()) -> None:
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

    def run(self, args: dict) -> str:
        specs = args.get("specs") or args.get("parts") or []
        # a model very commonly flattens a single spec's fields to the top level
        # ({"rank":2,"part":"X"}) instead of nesting it in `specs`.
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


__all__ = ["SearchCode", "FetchCode"]
