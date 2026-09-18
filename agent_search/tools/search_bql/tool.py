"""`search`: field-tagged Boolean/date search over a structure table (the BQL sieve).

search(query, k) translates the field-tagged surface (term[field], AND/OR/NOT, (), wildcard*,
"phrase") to BQL and returns a candidate table (rank, id, title, section names, infobox keys,
matched fields, an optional excerpt), no body content. Pairs with `fetch`
(agent_search.tools.fetch, the named-section read) or `visit` (agent_search.tools.visit, the
whole-doc read) depending on the toolset a strategy assembles.

Options:
  snippet     -- the excerpt under each hit (agent_search.snippets); `TermWindow()` for `research_snip`.
  coverage    -- on a 0-exact-hit AND, try constraint-coverage ranking before the soft-
                 relevance fallback (BQL v2 Feature 2).
  date_nudge  -- append a one-line hint when the raw query text carries a bare temporal
                 clue outside a date[...] scope.
  ranking     -- which engine this tool binds to: "bm25" (plain BQL, engine kind `bql`),
                 "fused" (BM25+dense RRF, engine kind `bql_fused`), "dense" (dense-only
                 ordering, engine kind `bql_dense`). Sets `self.engines` accordingly.
  manual_set  -- "v1", "v2", "paper", "nomanual", "syntax" or "noconstruct" manual files. The run's field profile (general, wiki,
                 browsecomp, code) picks which file of the set the agent reads.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from agent_search.corpus.units import code_tokenize
from agent_search.retrievers.bql.ast import And
from agent_search.retrievers.bql.ast import In, Not, Or, Phrase, Prefix, Term
from agent_search.retrievers.bql.executor import _rank_leaves, execute_bql
from agent_search.retrievers.bql.parser import parse as bql_parse
from agent_search.retrievers.bql.surface import to_bql

from agent_search.tools.base import Tool, query_text
from agent_search.tools.budgets import SNIPPET_TOKENS
from agent_search.snippets import NoSnippet, Snippet
from agent_search.tools.common import _infobox, _INTRO, sections_from_body, structure_str

_DATE_RANGE_PREFIX = "__daterange__"

# date-nudge (`date_nudge`, off by default): a mechanical nudge derived only from the agent's
# own raw query text, never from the corpus, so it can't leak retrieval information
# (fairness-critical). Detects a temporal clue (a standalone year, a decade like "1980s", or
# "Month YYYY" text) written as a plain keyword outside any date[...] scope, and coaches the
# agent toward the typed date[RANGE] surface the manual teaches.
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
_DECADE_RE = re.compile(r"\b(?:18|19|20)\d0s\b", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"(?:18|19|20)\d{2}\b", re.IGNORECASE)
_DATE_SCOPE_RE = re.compile(r"date\[[^\]]*\]", re.IGNORECASE)

_DATE_NUDGE_HINT = (
    "hint: temporal clues match documents by METADATA date — write date[2018] / "
    "date[1980..1989] / date[<=2023-12] instead of the year as a keyword.")
_DATE_NUDGE_CAP = 3


def _has_bare_temporal_clue(query: str) -> bool:
    """True if `query` (the raw surface text, pre-BQL) carries a year/decade/month-year clue
    outside any `date[...]` scope. Operates purely on the agent's own query string, no
    corpus lookups, so the nudge can never leak corpus information."""
    if not query:
        return False
    remainder = _DATE_SCOPE_RE.sub(" ", query)
    return bool(_YEAR_RE.search(remainder) or _DECADE_RE.search(remainder)
                or _MONTH_YEAR_RE.search(remainder))


def _child_repr(node) -> str:
    """A compact, surface-ish repr of a BQL AND-child for coverage-miss labels (BQL v2
    Feature 2, `coverage=True`) -- e.g. 'foo[body]', 'NOT(baz)',
    'date[1980-01-01..1989-12-31]'. Not a full unparse; good enough to tell the agent WHICH
    clause missed."""
    if isinstance(node, Term):
        return f'"{node.text}"' if node.quoted else node.text
    if isinstance(node, Phrase):
        return '"' + " ".join(t.text for t in node.terms) + '"'
    if isinstance(node, Prefix):
        return f"{node.stem}*"
    if isinstance(node, In):
        if (isinstance(node.child, Term)
                and node.child.text.startswith(_DATE_RANGE_PREFIX)):
            rest = node.child.text[len(_DATE_RANGE_PREFIX):]
            lo, _, hi = rest.partition("__")
            lo = "" if lo == "open" else lo
            hi = "" if hi == "open" else hi
            return f"date[{lo}..{hi}]"
        return f"{_child_repr(node.child)}[{node.region.value}]"
    if isinstance(node, Not):
        return f"NOT({_child_repr(node.child)})"
    if isinstance(node, And):
        return "AND(" + ", ".join(_child_repr(c) for c in node.children) + ")"
    if isinstance(node, Or):
        return "OR(" + ", ".join(_child_repr(c) for c in node.children) + ")"
    return repr(node)


class SearchBql(Tool):
    """search(query) -> candidate table (structure only). `specs` entries a paired `fetch`
    tool later resolves reference the rank from the last search (int, 1-based) or an
    explicit doc_id (str). Every doc_id surfaced (a search hit) is added to `state.seen`."""

    name = "search"
    # the exact text the paper prompts showed for search_s / search_bqlds / search_bqldos
    # (excerpts on) and search_bqldf (excerpts off): one shared sentence, plus one more when
    # each hit carries an excerpt. Set in __init__ from the `snippet` option.
    DESCRIPTION = ("Boolean/structural search over the corpus (field-tagged term[field] syntax); "
                   "returns ranked candidates plus their STRUCTURE — code: a file's function/method "
                   "names; docs: an article's section names + infobox keys — no bodies, numbered for "
                   "fetch.")
    SNIPPET_SENTENCE = " Each hit includes a one-line best-matching excerpt."
    description = DESCRIPTION
    parameters = {"type": "object",
                  "properties": {
                      "query": {"type": "string",
                                "description": "Field-tagged Boolean query: term[field], AND/OR/NOT, "
                                               "wildcard*, \"phrase\". Code fields: def, call, string, "
                                               "comment, sig, file. Doc fields: title, body, section, "
                                               "infobox."},
                      "k": {"type": "integer",
                            "description": "Max ranked candidates to return (default 5)."}},
                  "required": ["query"]}

    # every name the paper's Sieve conditions exposed this search under (the model, or the
    # keyword stub, may call any of them)
    aliases = ("search", "search_v2", "search_s", "search_bqlds", "search_bqldf", "search_bqldos",
               "search_bv", "search_bqld", "search_bqldo")

    _RANKING_TO_ENGINE = {"bm25": "bql", "fused": "bql_fused", "dense": "bql_dense"}
    # the manual per field profile (the run's dataset profile picks one; `code` has its own)
    _MANUALS = {
        "v1": {"general": "bql_doc.md", "wiki": "bql_doc.md", "browsecomp": "bql_browsecomp.md",
               "code": "bql_code.md"},
        "v2": {"general": "bql_doc_v2.md", "wiki": "bql_doc_v2.md", "browsecomp": "bql_browsecomp_v2.md",
               "code": "bql_code.md"},
        # the Sieve paper's manuals verbatim (only the fetch examples carry the rank-and-section call);
        # the BrowseComp one describes an unsegmented corpus and understates section access
        "paper": {"general": "bql_doc.md", "wiki": "bql_doc.md", "browsecomp": "bql_browsecomp_paper.md",
                  "code": "bql_code.md"},
        # the manual ablation (BoolAgent's control grid, 2026-09-18), each cut applied to this
        # library's corrected manuals: no manual (a 21-word stub), the reference only (mechanics,
        # fields, fetch, worked examples), and the full manual without the query-construction advice
        "nomanual": {"general": "bql_browsecomp_nomanual.md", "wiki": "bql_browsecomp_nomanual.md",
                     "browsecomp": "bql_browsecomp_nomanual.md", "code": "bql_code.md"},
        "syntax": {"general": "bql_doc_syntax.md", "wiki": "bql_doc_syntax.md",
                   "browsecomp": "bql_browsecomp_syntax.md", "code": "bql_code.md"},
        "noconstruct": {"general": "bql_doc_noconstruct.md", "wiki": "bql_doc_noconstruct.md",
                        "browsecomp": "bql_browsecomp_noconstruct.md", "code": "bql_code.md"},
        # a card: the query language, the fields and the fetch call in about a hundred words
        "card": {"general": "bql_doc_card.md", "wiki": "bql_doc_card.md",
                 "browsecomp": "bql_browsecomp_card.md", "code": "bql_code.md"},
        # leave-one-section-out cuts of the corrected manuals, derived by
        # scripts/derive_manual_cuts.py (each drops one `## ` section: How to search, The fields,
        # Fetch, Hops, Worked examples, Common mistakes)
        # the reference-only manual plus one advice section added back (same generator)
        **{f"ref{part}": {"general": f"bql_doc_ref{part}.md", "wiki": f"bql_doc_ref{part}.md",
                          "browsecomp": f"bql_browsecomp_ref{part}.md", "code": "bql_code.md"}
           for part in ("howto", "hops", "mistakes")},
        **{f"no{part}": {"general": f"bql_doc_no{part}.md", "wiki": f"bql_doc_no{part}.md",
                         "browsecomp": f"bql_browsecomp_no{part}.md", "code": "bql_code.md"}
           for part in ("howto", "fields", "fetch", "hops", "examples", "mistakes")},
    }

    # options a strategy sets
    snippet: Snippet = NoSnippet()   # the excerpt under each hit (agent_search.snippets)
    coverage: bool = False
    date_nudge: bool = False
    ranking: str = "bm25"          # bm25 | fused | dense -- selects self.engines
    manual_set: str = "v1"         # v1 | v2 | paper | nomanual | card | syntax | noconstruct | no{howto,fields,fetch,hops,examples,mistakes}
    # fill=True: a listing always has k rows. The exact Boolean matches come first, in the
    # ranker's order; when the filter admits fewer than k documents the remaining rows are the
    # ranker's closest documents over the query's own terms (the same ranking the zero-hit
    # fallback uses), marked `~` so the agent knows they did not pass the filter. The paper's
    # Sieve (fill=False) shows only the exact matches, so a tight filter leaves the agent one
    # or two candidates where a baseline lists five.
    fill: bool = False
    manual = _MANUALS["v1"]

    def __init__(self, name: Optional[str] = None, **options):
        super().__init__(name, **options)
        kind = self._RANKING_TO_ENGINE.get(self.ranking)
        if kind is None:
            raise ValueError(f"unknown ranking {self.ranking!r} — choose bm25, fused or dense.")
        self.engines = (kind,)
        self.manual = self._MANUALS.get(self.manual_set, self._MANUALS["v1"])
        self.description = self.DESCRIPTION + (self.SNIPPET_SENTENCE if self.snippet.shows_excerpt else "")
        self._date_nudge_emitted = 0

    def on_bind(self) -> None:
        self._date_nudge_emitted = 0

    @property
    def ex(self):
        return self.engine[self.engines[0]]

    # -- section/infobox cache, shared with a paired `fetch` tool via state.listing -------

    def _secs(self, doc_id: str) -> dict:
        entry = self.state.listing.get(doc_id)
        if entry is not None and "sections" in entry:
            return entry["sections"]
        u = self.ubyid.get(doc_id)
        explicit = getattr(u, "sections", None) if u is not None else None
        if explicit:
            # a structured corpus: the matched (heading, text) parts, shipped explicitly.
            out: dict = {}
            for h, t in explicit:
                name = h or _INTRO
                key, n = name, 2
                while key in out:
                    key, n = f"{name} ({n})", n + 1
                out[key] = t
            secs = out or {_INTRO: ""}
        else:
            secs = sections_from_body(u.body if (u and u.body is not None)
                                      else (u.code if u else ""))
        entry = self.state.listing.setdefault(doc_id, {})
        entry["sections"] = secs
        entry["infobox"] = _infobox(u) if u is not None else {}
        return secs

    def _matched_fields(self, u, leaf_toks: list) -> str:
        """Which fields the query's positive tokens appear in -- the 'why this hit' signal,
        no content shown."""
        meta = u.metadata or {}
        out = []
        checks = [("title", u.title or u.qualname or ""),
                  ("section", u.section or " ".join(self._secs(u.doc_id))),
                  ("infobox", " ".join(f"{k} {v}" for k, v in _infobox(u).items())),
                  ("author", str(meta.get("author", ""))),
                  ("date", str(meta.get("date", "")))]
        for name, blob in checks:
            bag = set(code_tokenize(blob))
            if any(t in bag for t in leaf_toks):
                out.append(name)
        if any(t in set(code_tokenize(u.body or u.code or "")) for t in leaf_toks):
            out.append("body")
        return ",".join(out) or "-"

    def _best_line(self, u, leaf_toks: list, width: int = SNIPPET_TOKENS) -> str:
        """The excerpt under one hit: the strategy's snippet method over the query's leaf tokens."""
        return self.snippet.render(u, leaf_toks, width=width)

    def _render_hits(self, hit_ids: list, leaf_toks: list, header: str, filled=()) -> str:
        """Render the structure table (rank, doc_id, title, §section names, ib[infobox
        keys], matched fields) for `hit_ids` under `header`. Marks every listed doc surfaced:
        the gold-doc-coverage metric must count a fallback hit exactly like an exact hit.
        `filled` names the rows that did not pass the Boolean filter (the fill-to-k rows)."""
        lines = [header]
        filled = set(filled or ())
        for rank, doc_id in enumerate(hit_ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.state.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str, ib_str = structure_str(named, list(_infobox(u)))
            title = u.title or u.qualname or doc_id
            matched = self._matched_fields(u, leaf_toks)
            line = (f"  {rank}{'~' if doc_id in filled else ' '} {doc_id}  {title!r}  "
                   f"§[{sec_str}]  ib[{ib_str}]  matched: {matched}")
            if self.snippet.shows_excerpt:
                snip = self._best_line(u, leaf_toks)
                if snip:
                    line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def _coverage_render(self, query: str, bql: str, expr, k: int) -> Optional[str]:
        """BQL v2 Feature 2: render `self.ex.coverage_topk(expr, k)` as the same structure
        table `_render_hits` renders, plus a per-hit `cov=n/total miss=[...]` suffix naming
        the unmatched AND children. Returns None if coverage_topk finds nothing, so the
        caller falls back to the soft_topk path."""
        rows = self.ex.coverage_topk(expr, k=k)
        if not rows:
            return None
        self.state.last_hits = [r[0] for r in rows]
        self.state.seen.update(self.state.last_hits)
        children = list(expr.children)
        total = len(children)
        try:
            leaf_toks = [t.lower() for t in _rank_leaves(expr)]
        except Exception:  # noqa: BLE001
            leaf_toks = []
        header = (f"search: {query}  ->  {bql}   (0 exact matches — closest by CONSTRAINT "
                  f"COVERAGE; 'miss' names the unmatched constraint):")
        table = self._render_hits(self.state.last_hits, leaf_toks, header)
        lines = table.split("\n")
        by_id = {r[0]: r for r in rows}
        out = [lines[0]]
        rendered_ids = [d for d in self.state.last_hits if d in self.ubyid]
        for line, doc_id in zip(lines[1:], rendered_ids):
            _, mask, n_matched, _score = by_id[doc_id]
            miss = [_child_repr(c) for c, matched in zip(children, mask) if not matched]
            out.append(f"{line}  cov={n_matched}/{total} miss=[{', '.join(miss)}]")
        return "\n".join(out)

    def _search(self, query: str, k: int = 5) -> str:
        """search(query) -> the structure table, plus, when the v2 date-nudge is enabled (off
        by default), one appended hint line whenever the raw query text carries a bare
        temporal clue outside a `date[...]` scope, capped at `_DATE_NUDGE_CAP` per
        instance."""
        result = self._search_impl(query, k)
        if (self.date_nudge and self._date_nudge_emitted < _DATE_NUDGE_CAP
                and _has_bare_temporal_clue(query)):
            self._date_nudge_emitted += 1
            result = f"{result}\n{_DATE_NUDGE_HINT}"
        return result

    def _search_impl(self, query: str, k: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        try:
            bql = to_bql(query, domain="doc")
        except Exception:  # noqa: BLE001
            return (f"ERROR: could not translate query {query!r}. Use "
                     f"term[field] / AND / OR / NOT / () / \"phrase\" / wildcard*.")
        obs = execute_bql(bql, self.ex, self.ubyid, k=k)
        if obs.error:
            return f"search: {query}  ->  {bql}\nBQL {obs.error}"
        if not obs.hits:
            # There is no corpus-token "did you mean" here: peeking at the corpus vocabulary
            # to spell-correct a 0-hit term is a retrieval-side advantage the bm25/dci
            # baselines (and real web search) do not get. The soft-AND fallback below is
            # corpus-fair: it is plain BM25 over the agent's own query terms.
            try:
                parsed_expr = bql_parse(bql).expr
                leaf_toks = [t.lower() for t in _rank_leaves(parsed_expr)]
            except Exception:  # noqa: BLE001
                parsed_expr = None
                leaf_toks = []
            # coverage=True: a 0-exact-hit AND with >=2 children gets constraint-coverage
            # ranking first, which pinpoints which clause failed.
            if self.coverage and isinstance(parsed_expr, And) and len(parsed_expr.children) >= 2:
                cov_render = self._coverage_render(query, bql, parsed_expr, k)
                if cov_render is not None:
                    return cov_render
            # BQL_SOFT_FALLBACK=0 disables the fallback (ablation knob); default on.
            if os.environ.get("BQL_SOFT_FALLBACK", "1") in ("0", "false", "no"):
                leaf_toks = []
            if leaf_toks:
                soft_hits = self.ex.soft_topk(leaf_toks, k=k)
                if soft_hits:
                    # exact AND is brittle under paraphrase/obfuscation: fall back to the
                    # whole-corpus BM25 relevance ranking over the query's own terms. These
                    # soft hits replace last_hits (not append).
                    self.state.last_hits = [doc_id for doc_id, _ in soft_hits]
                    self.state.seen.update(self.state.last_hits)
                    header = (f"search: {query}  ->  {bql}   (0 exact matches — showing top "
                              f"{len(self.state.last_hits)} CLOSEST docs by term relevance; "
                              f"fetch to verify, or pivot/loosen)")
                    return self._render_hits(self.state.last_hits, leaf_toks, header)
            # no soft hits either (or the leaf parse failed): keep the prior non-empty
            # ranking fetchable, a 0-hit pivot/loosen must not wipe the last good hits.
            prior = ("  (previous results still available to fetch)"
                     if self.state.last_hits else "")
            return (f"search: {query}  ->  {bql}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        exact = [h.doc_id for h in obs.hits[:k]]
        try:
            leaf_toks = [t.lower() for t in _rank_leaves(bql_parse(bql).expr)]
        except Exception:  # noqa: BLE001
            leaf_toks = []
        filled: list = []
        if (self.fill and len(exact) < k and leaf_toks
                and os.environ.get("BQL_SOFT_FALLBACK", "1") not in ("0", "false", "no")):
            try:
                soft = self.ex.soft_topk(leaf_toks, k=k + len(exact) + 5)
            except Exception:  # noqa: BLE001, a ranker failure leaves the exact rows alone
                soft = []
            have = set(exact)
            filled = [d for d, _ in soft if d not in have][: k - len(exact)]
        self.state.last_hits = exact + filled
        if filled:
            header = (f"search: {query}  ->  {bql}   ({obs.n_hits} matches, top {len(exact)}; "
                      f"rows marked ~ did not pass the filter and are the closest by relevance):")
        else:
            header = (f"search: {query}  ->  {bql}   "
                      f"({obs.n_hits} matches, top {len(exact)}):")
        return self._render_hits(self.state.last_hits, leaf_toks, header, filled=filled)

    def run(self, args: dict) -> str:
        query = query_text(args)
        k = int(args.get("k", 5) or 5)
        return self._search(query, k)


__all__ = ["SearchBql"]
