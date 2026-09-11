"""Pre-0.3 BQL sieve: field-tagged Boolean/date search over a structure table, plus fetch.

Kept so the parity tests can compare against it. The current equivalent is the `sieve` family
of strategies in `agent_search/strategies/sieve.py`, built from the `search_bql` and `fetch`/
`visit` tools in `agent_search/tools/`.

`DocSearchFetch` is the paper's method: search(query, k) translates the field-tagged
surface (term[field], AND/OR/NOT, (), wildcard*, "phrase") to BQL and returns a candidate
table (title, matched fields, section/infobox names) with no body content; fetch(specs)
pulls a named section or infobox for specific docs. `BqlVisitWorkspace` pairs the same BQL
search (with coverage ranking and the date-nudge forced on) with a whole-doc visit read
instead of fetch. `BqlDonlyVisitWorkspace`/`DocSearchFetchDonlySnip` are thin dispatch
subclasses for the dense-only-ranked executor variants.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from agent_search.core.seen import OrderedSeen
from agent_search.core.tokens import cap_tokens as _cap_tokens
from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.bql.ast import And, In, Not, Or, Phrase, Prefix, Term
from agent_search.retrievers.bql.executor import (
    StructuralExecutor, execute_bql, _rank_leaves)
from agent_search.retrievers.bql.parser import parse as bql_parse
from agent_search.retrievers.bql.surface import to_bql

from .budgets import MAX_SECTION_TOKENS, MAX_VISIT_TOKENS, SNIPPET_TOKENS
from .common import _INTRO, _SeenMixin, _infobox, best_line, sections_from_body

_DATE_RANGE_PREFIX = "__daterange__"

# --- date-nudge (DocSearchFetch's `date_nudge`, DEFAULT OFF): a mechanical, CORPUS-FREE
# mid-episode nudge — derived only from the agent's own raw query text, never from the corpus — so it can't
# leak retrieval information (fairness-critical). Detects a temporal clue (a standalone year, a
# decade like "1980s", or "Month YYYY" text) written as a plain keyword OUTSIDE any date[...]
# scope, and coaches the agent toward the typed date[RANGE] surface the skill teaches.
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
    OUTSIDE any `date[...]` scope. Operates purely on the agent's own query string — no corpus
    lookups — so the nudge can never leak corpus information."""
    if not query:
        return False
    remainder = _DATE_SCOPE_RE.sub(" ", query)
    return bool(_YEAR_RE.search(remainder) or _DECADE_RE.search(remainder)
                or _MONTH_YEAR_RE.search(remainder))


def _child_repr(node) -> str:
    """A compact, surface-ish repr of a BQL AND-child for coverage-miss labels (BQL v2
    Feature 2, DocSearchFetch(coverage=True)) — e.g. 'foo[body]', 'NOT(baz)',
    'date[1980-01-01..1989-12-31]'. Not a full unparse (there is no reverse-render in
    surface.py); good enough to tell the agent WHICH clause missed."""
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


class DocSearchFetch(_SeenMixin):
    """search(query) -> candidate TABLE (structure only); fetch(specs) -> named
    section(s)/infobox for specific docs, AGGREGATED. `specs` entries reference the RANK
    from the last search (int, 1-based) or an explicit doc_id (str). Tracks every doc_id
    surfaced (search hit or fetch target) for the gold-doc metric."""

    tools = ("search", "fetch")

    def __init__(self, units: Sequence[CodeUnit],
                 executor: Optional[StructuralExecutor] = None,
                 ubyid: Optional[dict] = None,
                 coverage: bool = False,
                 date_nudge: bool = False,
                 snippets: bool = False):
        self.units = units if getattr(units, "lazy", False) else list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if executor is not None:
            self.ex = executor
        else:
            # env STRUCTURED_BACKEND-selectable (default 'python', unchanged) — see
            # agent_search.retrievers.backend.build_bql_engine. Production
            # callers (agent_search.legacy.retriever) always pass `executor` explicitly (built
            # with a real index_root/key so a 'lucene' backend opens the prebuilt index); this
            # fallback is for direct/standalone construction (tests, ad-hoc scripts).
            from agent_search.retrievers.backend import build_bql_engine
            self.ex = build_bql_engine(self.units)
        self._sections: dict[str, dict] = {}                  # doc_id -> {heading: text} (lazy)
        self.seen = OrderedSeen()
        self.last_hits: list[str] = []
        # DEFAULT OFF: on a 0-exact-hit AND with >=2 children, try constraint-COVERAGE ranking
        # (executor.coverage_topk) before the undifferentiated soft_topk relevance fallback —
        # see _coverage_render below. The plain `research` condition never passes coverage=True,
        # so it always takes the soft_topk fallback.
        self.coverage = coverage
        # DEFAULT OFF: emit `_DATE_NUDGE_HINT` after a search whose raw query text carries a
        # bare temporal clue (see `_has_bare_temporal_clue`), capped at `_DATE_NUDGE_CAP` per
        # episode/instance. Wired True only by callers that opt in (legacy/retriever.py); the
        # plain `research` condition and Bm25FetchWorkspace leave it off.
        self.date_nudge = date_nudge
        self._date_nudge_emitted = 0
        # research_snip (DEFAULT OFF, `snippets=True`): `_render_hits` appends a one-line
        # best-matching excerpt (see `_best_line`) to every search hit. Callers other than the
        # docsnip arm (legacy/retriever.py) leave this off, so their listing carries no excerpt.
        self.snippets = snippets

    def _secs(self, doc_id: str) -> dict:
        s = self._sections.get(doc_id)
        if s is None:
            u = self.ubyid.get(doc_id)
            explicit = getattr(u, "sections", None) if u is not None else None
            if explicit:
                # STRUCTURED corpus: the matched (heading, text) parts, shipped explicitly. Rebuild
                # the {heading: text} view disambiguating repeats exactly as sections_from_body does,
                # so `fetch` and the search listing behave identically to the derived path.
                out: "dict[str, str]" = {}
                for h, t in explicit:
                    name = h or _INTRO
                    key, n = name, 2
                    while key in out:
                        key, n = f"{name} ({n})", n + 1
                    out[key] = t
                s = out or {_INTRO: ""}
            else:
                s = sections_from_body(u.body if (u and u.body is not None) else (u.code if u else ""))
            self._sections[doc_id] = s
        return s

    # -- search: field-tagged surface -> BQL -> structure table ---------------

    def _matched_fields(self, u: CodeUnit, leaf_toks: list) -> str:
        """Which fields the query's positive tokens appear in — the 'why this hit' signal,
        no content shown.

        Checks title/section/infobox/author/date/body: a browsecomp doc that matches PURELY on
        IN(author,·)/IN(date,·) (its only real structured fields; browsecomp has no
        section/infobox at all) still reports "matched: author"/"matched: date" instead of
        hiding the actual reason behind a generic "matched: body" or "matched: -"."""
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

    def _best_line(self, u: CodeUnit, leaf_toks: list, width: int = SNIPPET_TOKENS) -> str:
        """research_snip (`snippets=True`): the doc's best-matching ~`width`-token window for
        `leaf_toks`. Thin wrapper over the module-level `best_line` (shared verbatim with
        `Bm25Visit`'s query-biased excerpt mode) — kept as an instance method so callers (this
        class's `_render_hits`, doc_indri.py, tests) can call it uniformly."""
        return best_line(u, leaf_toks, width=width)

    def _render_hits(self, hit_ids: list, leaf_toks: list, header: str) -> str:
        """Render the structure TABLE (rank, doc_id, title, §section names, ib[infobox keys],
        matched fields) for `hit_ids` under `header`. Shared by the exact-BQL-hit path and the
        soft-topk fallback so the two renderings can never drift. Marks every listed doc `seen`
        — the gold-doc-coverage metric must count a fallback hit exactly like an exact hit.

        research_snip (`self.snippets`, DEFAULT OFF): appends a one-line best-matching excerpt
        (`_best_line`) to every hit line; the plain `research` condition leaves it off."""
        lines = [header]
        for rank, doc_id in enumerate(hit_ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            matched = self._matched_fields(u, leaf_toks)
            line = (f"  {rank}  {doc_id}  {title!r}  "
                   f"§[{sec_str}]  ib[{ib_str}]  matched: {matched}")
            if self.snippets:
                snip = self._best_line(u, leaf_toks)
                if snip:
                    line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def _coverage_render(self, query: str, bql: str, expr, k: int) -> Optional[str]:
        """BQL v2 Feature 2: render `self.ex.coverage_topk(expr, k)` as the SAME structure
        table `_render_hits` renders, plus a per-hit `cov=n/total miss=[...]` suffix naming
        the unmatched AND children. None if coverage_topk finds nothing (caller falls back to
        the unchanged soft_topk path)."""
        rows = self.ex.coverage_topk(expr, k=k)
        if not rows:
            return None
        self.last_hits = [r[0] for r in rows]
        self.seen.update(self.last_hits)
        children = list(expr.children)
        total = len(children)
        try:
            leaf_toks = [t.lower() for t in _rank_leaves(expr)]
        except Exception:  # noqa: BLE001
            leaf_toks = []
        header = (f"search: {query}  ->  {bql}   (0 exact matches — closest by CONSTRAINT "
                  f"COVERAGE; 'miss' names the unmatched constraint):")
        table = self._render_hits(self.last_hits, leaf_toks, header)
        lines = table.split("\n")
        by_id = {r[0]: r for r in rows}
        out = [lines[0]]
        # `_render_hits` skips any hit id missing from `self.ubyid` (one line fewer than
        # `self.last_hits` in that case) — mirror that exact filter here so `lines[1:]` still
        # pairs each rendered line with the doc_id that actually produced it, instead of
        # silently drifting out of alignment once a hit id is missing.
        rendered_ids = [d for d in self.last_hits if d in self.ubyid]
        for line, doc_id in zip(lines[1:], rendered_ids):
            _, mask, n_matched, _score = by_id[doc_id]
            miss = [_child_repr(c) for c, matched in zip(children, mask) if not matched]
            out.append(f"{line}  cov={n_matched}/{total} miss=[{', '.join(miss)}]")
        return "\n".join(out)

    def search(self, query: str, k: int = 5) -> str:
        """search(query) -> the structure TABLE (see `_search_impl`), plus — DocSearchFetch's
        v2 date-nudge (DEFAULT OFF, `date_nudge=True`) — ONE appended hint line whenever the
        RAW query text carries a bare temporal clue outside a `date[...]` scope, capped at
        `_DATE_NUDGE_CAP` per instance. This wrapper is the single choke point so every
        `_search_impl` return path (hits, coverage table, soft-fallback, 0-match) gets the same
        treatment — the nudge reads only the agent's own query string, never the corpus."""
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
            # NO corpus-token "did you mean" here: peeking at the corpus vocabulary to spell-correct
            # a 0-hit term is a retrieval-side advantage the bm25/dci baselines (and real web search)
            # do NOT get — an unfair confound that would inflate the BQL arm. Syntax feedback (the
            # to_bql translate error above) and field/type feedback (obs.error above) are KEPT — those
            # are query-language ergonomics, not corpus peeking. (self.ex.suggest / _region_vocab is
            # now unused here.) The SOFT-AND fallback below is corpus-fair: it's plain BM25 over the
            # agent's OWN query terms (same power the bm25 baseline arm has), not corpus peeking.
            try:
                parsed_expr = bql_parse(bql).expr
                leaf_toks = [t.lower() for t in _rank_leaves(parsed_expr)]
            except Exception:  # noqa: BLE001
                parsed_expr = None
                leaf_toks = []
            # coverage=True: a 0-exact-hit AND with >=2 children gets constraint-COVERAGE ranking
            # FIRST — it pinpoints WHICH clause failed, strictly more informative than
            # soft_topk's undifferentiated bag-of-terms relevance. Falls through to the
            # soft_topk path below if the expr isn't an And (a single constraint has nothing to
            # break down) or coverage_topk finds nothing.
            if self.coverage and isinstance(parsed_expr, And) and len(parsed_expr.children) >= 2:
                cov_render = self._coverage_render(query, bql, parsed_expr, k)
                if cov_render is not None:
                    return cov_render
            # BQL_SOFT_FALLBACK=0 disables the fallback (ablation knob: quantifies what graceful
            # degradation contributes vs the pure exact-Boolean surface); default ON.
            if os.environ.get("BQL_SOFT_FALLBACK", "1") in ("0", "false", "no"):
                leaf_toks = []
            if leaf_toks:
                soft_hits = self.ex.soft_topk(leaf_toks, k=k)
                if soft_hits:
                    # exact AND is brittle under paraphrase/obfuscation: fall back to the whole-corpus
                    # BM25 relevance ranking over the query's own terms. These soft hits REPLACE
                    # last_hits (not append) — they ARE the fetchable ranking now, so the "previous
                    # results still available" note doesn't apply here.
                    self.last_hits = [doc_id for doc_id, _ in soft_hits]
                    self.seen.update(self.last_hits)
                    header = (f"search: {query}  ->  {bql}   (0 exact matches — showing top "
                              f"{len(self.last_hits)} CLOSEST docs by term relevance; fetch to "
                              f"verify, or pivot/loosen)")
                    return self._render_hits(self.last_hits, leaf_toks, header)
            # no soft hits either (or the leaf parse failed): keep the PRIOR non-empty ranking
            # fetchable — a 0-hit pivot/loosen (which the skill explicitly coaches) must not wipe
            # the last good hits from under a fetch.
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}  ->  {bql}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = [h.doc_id for h in obs.hits[:k]]
        try:
            leaf_toks = [t.lower() for t in _rank_leaves(bql_parse(bql).expr)]
        except Exception:  # noqa: BLE001
            leaf_toks = []
        header = (f"search: {query}  ->  {bql}   "
                  f"({obs.n_hits} matches, top {len(self.last_hits)}):")
        return self._render_hits(self.last_hits, leaf_toks, header)

    # -- fetch: rank/doc_id + section name -> aggregated slices --------------

    def _resolve_doc(self, ref):
        if isinstance(ref, (int, float)) or (isinstance(ref, str) and ref.strip().isdigit()):
            rank = int(ref)
            if 1 <= rank <= len(self.last_hits):
                return self.ubyid[self.last_hits[rank - 1]], None
            # A digit that is NOT a valid rank may be a real doc_id (production/browsecomp
            # doc_ids ARE integers the agent sees in results) — resolve it as one before
            # erroring (do NOT treat every digit as a rank).
            if str(ref).strip() in self.ubyid:
                return self.ubyid[str(ref).strip()], None
            if not self.last_hits:
                return None, f"ERROR: no prior search — rank {rank} has nothing to refer to."
            return None, (f"ERROR: rank {rank} out of range "
                          f"(last search had {len(self.last_hits)} results).")
        ref = (ref or "").strip()
        if ref in self.ubyid:
            return self.ubyid[ref], None
        low = ref.lower()
        cands = [i for i, u in self.ubyid.items()
                 if low in i.lower() or low in (u.title or u.qualname or "").lower()]
        if len(cands) == 1:
            return self.ubyid[cands[0]], None
        if cands:
            opts = ", ".join(f"{i} ({self.ubyid[i].title or self.ubyid[i].qualname})"
                             for i in cands[:5])
            return None, f"ERROR: ambiguous doc {ref!r} — did you mean: {opts}"
        return None, f"ERROR: no such doc {ref!r} — use a rank from the last search or a doc_id."

    def _fetch_one(self, doc_ref, section_ref: str) -> tuple:
        u, err = self._resolve_doc(doc_ref)
        if err:
            return (str(doc_ref), err)
        self.seen.add(u.doc_id)
        named = self._secs(u.doc_id)
        infobox = _infobox(u)
        avail = "·".join(list(named)[:12]) + (",infobox" if infobox else "")
        s = (section_ref or "").strip()
        if re.sub(r"[\s_-]", "", s.lower()) in ("infobox", "info", "facts"):
            if not infobox:
                return (f"{u.doc_id} §infobox", f"(no infobox — sections: {avail})")
            body = "; ".join(f"{k}={v}" for k, v in infobox.items())
            return (f"{u.doc_id} §infobox", body)
        low = s.lower()
        names = list(named)
        if not s:                            # no section named -> the lead/opening content
            match = [_INTRO] if _INTRO in named else [names[0]] if names else []
        elif len(names) == 1:
            # A FLAT doc (browsecomp, no '##' markers) has exactly one section, always named
            # '(intro)' — never a name the agent could plausibly type. Any part name on a
            # single-section doc means "the body" (the skill's own field name for it), so honor
            # it instead of erroring "no section 'body'".
            match = names
        else:
            match = ([n for n in names if n.lower() == low]
                     or [n for n in names if n.lower().startswith(low)]
                     or [n for n in names if low in n.lower()])
        if not match:
            return (f"{u.doc_id} §{s}",
                    f"ERROR: no section {s!r} on {u.doc_id}. Available: {avail}")
        if len(match) > 1:
            return (f"{u.doc_id} §{s}",
                    f"ERROR: {s!r} is ambiguous: {'·'.join(match[:8])}")
        text = _cap_tokens(named[match[0]], MAX_SECTION_TOKENS,
                           " …(truncated — fetch a narrower section)")
        return (f"{u.doc_id} §{match[0]}", text)

    def fetch(self, specs: list) -> str:
        if not specs:
            return "ERROR: fetch needs at least one (doc, section) pair."
        # single flat pair [rank, "section"] -> one spec (same recovery as the code arm)
        if (isinstance(specs, (list, tuple)) and len(specs) == 2
                and not isinstance(specs[0], (list, tuple))
                and not isinstance(specs[1], (list, tuple))):
            try:
                int(specs[0])
                specs = [specs]
            except (TypeError, ValueError):
                pass
        lines = ["fetch:"]
        for spec in specs:
            if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                lines.append(f"  ERROR: bad spec {spec!r} — expected [doc, section].")
                continue
            doc_ref, section_ref = spec
            label, text = self._fetch_one(doc_ref, section_ref)
            lines.append(f"  [{label}]  {text}")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            # "search_v2"/"fetch_v2" are the docv2 arm's tool NAMES (an alias of search/fetch):
            # a distinct tool name is what gets it its own manual, but the workspace method is
            # the same.
            # "search_s"/"fetch_s" are research_snip's tool NAMES, same pattern — the
            # `snippets=True` behavior lives in `_render_hits`, not in this dispatch.
            # "search_bqlds"/"fetch_bqlds" are research_bql_dense_snip's tool NAMES (mirroring
            # research_snip's search_s/fetch_s exactly — see BQL_DENSE / bql/dense_fuse.py):
            # SAME DocSearchFetch(snippets=True) workspace, the only difference is the
            # DENSE-ATTACHED executor `legacy/retriever.py`'s 'bqldensesnip' arm builds.
            # "search_bqldf"/"fetch_bqldf" are research_bql_dense_fetch's tool NAMES (mirroring
            # the PLAIN `doc` arm's bare search/fetch exactly — see BQL_DENSE / bql/dense_fuse.py):
            # SAME DocSearchFetch() plain (snippets=False, the constructor default) workspace, the
            # only difference is the DENSE-ATTACHED executor `legacy/retriever.py`'s
            # 'bqldensefetch' arm builds — this is research_bql_dense_snip minus the snippet.
            if name in ("search", "search_v2", "search_s", "search_bqlds", "search_bqldf"):
                q = args.get("query") or args.get("q") or ""
                return self.search(q, int(args.get("k", 5) or 5))
            if name in ("fetch", "fetch_v2", "fetch_s", "fetch_bqlds", "fetch_bqldf"):
                specs = args.get("specs") or args.get("parts") or []
                if not specs and ("doc" in args or "section" in args or "rank" in args):
                    specs = [args]
                if isinstance(specs, dict):
                    specs = [specs]
                norm = []
                for s in specs:
                    if isinstance(s, dict):
                        norm.append((s.get("rank") or s.get("doc") or s.get("id"),
                                     s.get("section") or s.get("part") or s.get("name") or ""))
                    else:
                        norm.append(s)
                return self.fetch(norm)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class BqlVisitWorkspace(DocSearchFetch):
    """search_bv(query, k) -> the SAME field-tagged BQL search `research`/`research_snip` use,
    with `DocSearchFetch(coverage=True, date_nudge=True)` forced on (executor `coverage_topk`
    ranking on a 0-exact-hit AND, plus the typed date[RANGE] mid-episode nudge — see
    `DocSearchFetch._search_impl`/`search`), rendered WITH a per-hit content excerpt
    (`snippets=True`, FORCED — the SAME fairness-parity amendment `IndriVisitWorkspace` makes
    over `IndriFetchWorkspace`'s default-off snippets in doc_indri.py: a content-blind listing
    would handicap this cell vs the bm25/dense/indri visit baselines' content-bearing listings).
    visit_bv(rank_or_id) -> the FULL text of a ranked document (capped at MAX_VISIT_TOKENS),
    mirroring `Bm25Visit.visit`/`Bm25Visit._resolve` EXACTLY (same rank-or-doc_id resolution,
    same numeric-doc_id fallback, same whole-doc token cap) — the SAME copy `IndriVisitWorkspace`
    already carries for its own visit_v (no read logic is shared cross-module; each visit-family
    workspace keeps its own copy, matching the existing pattern).

    Completes the search x read factorial's last missing cell: {BQL search} x {whole-doc visit}.
    `research`/`research_snip` already fill {BQL search} x {structure->parts read};
    `research_bm25`/`research_dense`/the indrivisit arm fill {bm25/dense/indri search} x
    {whole-doc visit}. No fetch/structure tools — visit is the ONLY read here."""

    tools = ("search_bv", "visit_bv")

    def __init__(self, units: Sequence[CodeUnit],
                 executor: Optional[StructuralExecutor] = None,
                 ubyid: Optional[dict] = None,
                 date_nudge: bool = True,
                 tool_names: Optional[tuple] = None):
        super().__init__(units, executor=executor, ubyid=ubyid, coverage=True,
                         date_nudge=date_nudge, snippets=True)  # snippets ALWAYS on — not a caller knob
        # `tool_names` (DEFAULT None -> the class default above): the dense-attached sibling
        # this class also backs (legacy/retriever.py's 'bqldensevisit' arm, ranked by a
        # DENSE-ATTACHED `executor` — see BQL_DENSE / bql/dense_fuse.py) needs its OWN tool
        # NAMES (search_bqld/visit_bqld) so tools.yaml/sdk_driver.py can give it its own
        # <tools> listing entry, the SAME reason every other sibling condition in this module
        # gets a distinct tool name (search_s/fetch_s, search_bv/visit_bv, ...). `run()` below
        # accepts BOTH name sets unconditionally (harmless — a plain bqlvisit episode never sees
        # "search_bqld" advertised in its own toolset, so a real model never calls it).
        if tool_names is not None:
            self.tools = tuple(tool_names)

    # -- visit_bv: whole-doc read, mirroring Bm25Visit.visit/_resolve exactly ------------------

    def _resolve(self, ref):
        """Mirrors `Bm25Visit._resolve` (and `IndriVisitWorkspace._resolve`) exactly: a rank
        (int or numeric string) resolves against `self.last_hits`; a digit that is NOT a valid
        rank may itself be a real doc_id (numeric doc_ids do occur), so try that before
        erroring; otherwise fall back to a doc_id / title lookup."""
        if isinstance(ref, (int, float)) or (isinstance(ref, str) and ref.strip().isdigit()):
            rank = int(ref)
            if 1 <= rank <= len(self.last_hits):
                return self.last_hits[rank - 1], None
            if str(ref).strip() in self.ubyid:            # a real doc_id that looks numeric
                return str(ref).strip(), None
            if not self.last_hits:
                return None, f"ERROR: no prior search — rank {rank} has nothing to refer to."
            return None, (f"ERROR: rank {rank} out of range "
                          f"(last search had {len(self.last_hits)} results).")
        ref = (ref or "").strip()
        if ref in self.ubyid:
            return ref, None
        low = ref.lower()
        cands = [i for i, u in self.ubyid.items() if low in (u.title or u.qualname or "").lower()]
        if len(cands) == 1:
            return cands[0], None
        if cands:
            opts = ", ".join(f"{i} ({self.ubyid[i].title or self.ubyid[i].qualname})"
                             for i in cands[:5])
            return None, f"ERROR: ambiguous doc {ref!r} — did you mean: {opts}"
        return None, f"ERROR: no such doc {ref!r} — use a rank from the last search or a doc_id."

    def visit(self, rank_or_id) -> str:
        doc_id, err = self._resolve(rank_or_id)
        if err:
            return err
        self.seen.add(doc_id)
        u = self.ubyid[doc_id]
        text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                           " …(truncated — this is the whole-doc cap)")
        return f"{doc_id}  {(u.title or u.qualname or '')!r}:\n{text}"

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            # "search"/"search_v2" aliases: the docv2 arm's `search`/`search_v2` tool names also
            # dispatch here (this workspace's constructor is DocSearchFetch(coverage=True,
            # date_nudge=True), plus forced snippets) — a real episode's model, or the
            # KeywordPolicy stub (which opens with a bare `search` call), must reach it under
            # any of these names.
            # "search_bqld"/"visit_bqld" are the bqldensevisit arm's tool NAMES (BQL_DENSE —
            # see bql/dense_fuse.py); accepted here unconditionally like every other alias set
            # this dispatch already handles (search_v2, search_s, ...).
            if name in ("search_bv", "search_bqld", "search", "search_v2"):
                q = args.get("query") or args.get("q") or ""
                return self.search(q, int(args.get("k", 5) or 5))
            if name in ("visit_bv", "visit_bqld", "visit"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class BqlDonlyVisitWorkspace(BqlVisitWorkspace):
    """The dense-only-ranked sibling of `BqlVisitWorkspace`'s dense-attached variant (the
    bqldensevisit arm) — SAME `BqlVisitWorkspace` behavior (with its own `tool_names`) except the
    executor passed in by `legacy/retriever.py`'s 'bqldonlyvisit' arm is a
    `DenseOnlyStructuralExecutor` (dense-ONLY candidate ordering) instead of a plain
    `StructuralExecutor` with a `DenseBelief` attached (RRF). `search_bqldo`/`visit_bqldo` are
    this arm's own tool names, translated to `search_bv`/`visit_bv` before delegating to the
    inherited `BqlVisitWorkspace.run` (which already accepts `search_bv`/`search_bqld`/`search`/
    `search_v2` and `visit_bv`/`visit_bqld`/`visit`)."""

    _NAME_MAP = {"search_bqldo": "search_bv", "visit_bqldo": "visit_bv"}

    def run(self, name: str, args: dict) -> str:
        return super().run(self._NAME_MAP.get(name, name), args)


class DocSearchFetchDonlySnip(DocSearchFetch):
    """research_bql_donly_snip — the dense-only-ranked sibling of research_bql_dense_snip
    (`DocSearchFetch(snippets=True)`) — except the executor passed in by `legacy/retriever.py`'s
    'bqldonlysnip' arm is a `DenseOnlyStructuralExecutor` (dense-ONLY ordering WITHIN each
    coverage tier — the tier STRUCTURE itself, and the exact-hit filter, are untouched) instead
    of a plain `StructuralExecutor` with a `DenseBelief` attached (RRF-within-tier).
    `search_bqldos`/`fetch_bqldos` are this condition's own tool names, translated to
    `search_bqlds`/`fetch_bqlds` before delegating to the inherited `DocSearchFetch.run` (which
    already accepts `search_bqlds`/`fetch_bqlds` as research_bql_dense_snip's own aliases)."""

    tools = ("search_bqldos", "fetch_bqldos")

    _NAME_MAP = {"search_bqldos": "search_bqlds", "fetch_bqldos": "fetch_bqlds"}

    def run(self, name: str, args: dict) -> str:
        return super().run(self._NAME_MAP.get(name, name), args)


# --- arms. `doc` (search/fetch) is the fallback for a general-domain toolset no other arm claims.
from agent_search.legacy.retriever import register_workspace  # noqa: E402

register_workspace("doc", tools=("search", "fetch"), engines=("bql",), domain="general",
                   builder=lambda ctx: DocSearchFetch(ctx.units, executor=ctx.bql(), ubyid=ctx.ubyid))
register_workspace("docv2", tools=("search_v2", "fetch_v2"), engines=("bql",),
                   builder=lambda ctx: DocSearchFetch(ctx.units, executor=ctx.bql(), ubyid=ctx.ubyid,
                                                      coverage=True, date_nudge=True))
register_workspace("docsnip", tools=("search_s", "fetch_s"), engines=("bql",),
                   builder=lambda ctx: DocSearchFetch(ctx.units, executor=ctx.bql(), ubyid=ctx.ubyid, snippets=True))
register_workspace("bqlvisit", tools=("search_bv", "visit_bv"), engines=("bql",),
                   builder=lambda ctx: BqlVisitWorkspace(ctx.units, executor=ctx.bql(), ubyid=ctx.ubyid))
register_workspace("bqldensevisit", tools=("search_bqld", "visit_bqld"), engines=("bql_dense",),
                   builder=lambda ctx: BqlVisitWorkspace(ctx.units, executor=ctx.bql_dense(), ubyid=ctx.ubyid,
                                                         tool_names=("search_bqld", "visit_bqld")))
register_workspace("bqldensesnip", tools=("search_bqlds", "fetch_bqlds"), engines=("bql_dense",),
                   builder=lambda ctx: DocSearchFetch(ctx.units, executor=ctx.bql_dense(), ubyid=ctx.ubyid, snippets=True))
register_workspace("bqldensefetch", tools=("search_bqldf", "fetch_bqldf"), engines=("bql_dense",),
                   builder=lambda ctx: DocSearchFetch(ctx.units, executor=ctx.bql_dense(), ubyid=ctx.ubyid))
register_workspace("bqldonlyvisit", tools=("search_bqldo", "visit_bqldo"), engines=("bql_dense_only",),
                   builder=lambda ctx: BqlDonlyVisitWorkspace(ctx.units, executor=ctx.bql_dense_only(), ubyid=ctx.ubyid,
                                                              tool_names=("search_bqldo", "visit_bqldo")))
register_workspace("bqldonlysnip", tools=("search_bqldos", "fetch_bqldos"), engines=("bql_dense_only",),
                   builder=lambda ctx: DocSearchFetchDonlySnip(ctx.units, executor=ctx.bql_dense_only(), ubyid=ctx.ubyid,
                                                               snippets=True))
