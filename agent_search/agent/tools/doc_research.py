"""The deep-research doc ACI: search -> fetch (the BQL field-tagged Boolean surface) + the bm25 baseline.

The doc arm mirrors the code arm's design over a document corpus (DESIGN.md's unifying
abstraction): a "document" is an ARTICLE; its "parts" are SECTIONS (## headings) computed
LIVE from the body — no new corpus format, the executor is untouched.

Two workspaces over the SAME unit collection (comparability):

  DocSearchFetch (the method) — search(query, k) translates the field-tagged surface
    (term[field], AND/OR/NOT, (), wildcard*, "phrase"; DOC_FIELDS: title/body/section/
    infobox/date) to BQL and runs it over the executor. Returns a candidate TABLE —
    title, matched fields, available section names, infobox keys — NEVER body content.
    fetch(specs) pulls a named SECTION (or infobox) for specific docs (by rank or doc_id),
    aggregated side by side.

  Bm25Visit (the baseline, retrieve-then-visit / RUC-style) — search(query, k) is plain
    BM25 over the flat doc text; results show title + a short opening snippet. visit(rank_or_id)
    returns that doc's FULL text (capped). No structure, no coaching skill.

  DenseVisit (the DENSE baseline, the modern RAG default) — a byte-identical clone of
    Bm25Visit, just with dense embedding cosine similarity (BAAI/bge-base-en-v1.5, via
    DenseBelief's cached doc-embedding matrix) swapped in for the BM25 engine.

  BqlVisitWorkspace (research_bql_visit — the {BQL search} x {whole-doc visit} factorial cell)
    — the SAME BQL v2 field-tagged search research_v2 uses (coverage_topk ranking + the typed
    date[RANGE] nudge), rendered WITH a per-hit content excerpt (fairness parity, like
    IndriVisitWorkspace over IndriFetchWorkspace), paired with a whole-doc VISIT read identical
    to Bm25Visit.visit. No fetch/structure tools — completes the search x read factorial's last
    missing cross.

  Bm25AutoRead (research_bm25_autoread — the "retrieve-and-read" baseline) — a byte-identical
    clone of Bm25Visit's BM25 ranking (SAME engine, SAME fallback), but search(query) itself
    renders the FULL TEXT of every one of the top `AUTOREAD_TOPK` hits (capped at
    MAX_VISIT_TOKENS per doc, the SAME `_cap_tokens` truncation/marker `Bm25Visit.visit` uses) —
    there is NO visit/fetch tool at all in this condition; a search call IS the read. Sits
    between the one-shot RAG baseline (a single stuffed prompt, no agent loop at all) and the
    SERP retrieve-then-visit baseline (research_bm25 — a listing the agent must then CHOOSE to
    visit): isolates the value of the agent's choose-what-to-read step by removing it entirely.

  DenseAutoRead (research_dense_autoread — the DENSE "retrieve-and-read" baseline) — a
    byte-identical clone of DenseVisit's dense-embedding ranking (SAME engine, SAME
    DenseBelief.top_k_doc_ids), but search(query) itself renders the FULL TEXT of every one of
    the top `AUTOREAD_TOPK` hits (SAME shared env knob Bm25AutoRead uses, capped at
    MAX_VISIT_TOKENS per doc, the SAME `_cap_tokens` truncation/marker `Bm25AutoRead`/
    `Bm25Visit.visit` use) — there is NO visit/fetch tool at all in this condition; a search
    call IS the read. The dense analog of Bm25AutoRead exactly as `DenseVisit` is the dense
    analog of `Bm25Visit`: the ONLY difference from `research_bm25_autoread` is the retrieval
    engine (dense cosine similarity instead of BM25) — auto-read behavior, top-k, the per-doc
    cap, and the <answer> contract are all identical.

  HybridVisit / HybridFetchSnipWorkspace (research_hybrid / research_hybrid_fetch_snip — the
    BM25+DENSE HYBRID baseline) — the control that isolates the BQL method's contribution from
    the mere "sparse+dense fusion" effect: research_bm25 alone and research_dense alone each
    change ONE retrieval axis, but a real deployment would just fuse them, so neither is the
    right control for "does the QL method beat textbook hybrid search". Retrieval is Reciprocal
    Rank Fusion (RRF; Cormack, Clarke & Buettcher 2009) over TWO INDEPENDENTLY-ranked pools,
    each `HYBRID_POOL` (100) docs deep:
      (a) the canonical pyserini/Lucene BM25 ranking (BM25_BACKEND=pyserini path — the SAME
          `build_bm25_engine` every bm25-consuming arm uses);
      (b) the FAISS/dense ranking (the SAME BAAI/bge-base-en-v1.5 embedder + persisted
          `indexes/dense/<model>-sl<len>/<key>/` cache research_dense/research_dense_fetch use,
          via DenseBelief.top_k_doc_ids — see vector_index.py for the FAISS-backed VectorIndex).
    Fusion formula (`rrf_fuse`, standard RRF, k=`RRF_K`=60):

        score(d) = sum_i  1 / (RRF_K + rank_i(d))     for each pool i where d appears
                                                        (rank_i is 1-based; a doc absent
                                                        from a pool contributes 0 for it)

    ranked strictly by descending score(d), ties broken by doc_id ascending (deterministic).
    `HybridVisit` (research_hybrid) mirrors Bm25Visit's retrieve-then-visit shape EXACTLY (same
    listing: title + opening snippet; same visit/_resolve, inherited unchanged) — only the
    retrieval ranking differs. `HybridFetchSnipWorkspace` (research_hybrid_fetch_snip) is its
    fetch-mode twin, mirroring Bm25FetchSnipWorkspace: the SAME RRF-fused retrieval, rendered as
    a structure table (section names + infobox keys) plus a one-line best-matching excerpt per
    hit, paired with the SAME structured section-fetch `fetch` every method/fetch cell uses.
    Both UNCOACHED (no manual), like the bm25/dense baselines they're built from.

  HybridAutoRead (research_hybrid_autoread — the BM25+DENSE HYBRID analog of Bm25AutoRead/
    DenseAutoRead) — the SAME RRF-fused bm25+dense retrieval as `HybridVisit` (SAME
    `HYBRID_POOL`/`RRF_K`, same two engines), but `search(query)` itself renders the FULL TEXT of
    every one of the top `AUTOREAD_TOPK` hits (SAME shared env knob/cap/marker
    Bm25AutoRead/DenseAutoRead use) — there is NO visit/fetch tool at all in this condition; a
    search call IS the read. Sits alongside research_bm25_autoread/research_dense_autoread as the
    third "retrieve-and-read" cell, isolating the choose-what-to-read step's value under fused
    retrieval instead of a single engine. Uncoached (no skill), like the bm25/dense/hybrid
    baselines it's built from.

  BqlDonlyVisitWorkspace / DocSearchFetchDonlySnip (research_bql_donly_visit /
    research_bql_donly_snip — the DENSE-ONLY siblings of research_bql_dense_visit/
    research_bql_dense_snip) — BYTE-FOR-BYTE `BqlVisitWorkspace`/`DocSearchFetch(snippets=True)`
    (same manual, same listing, same read), except the attached executor is a
    `agent_search.retrievers.structural.bql.executor.DenseOnlyStructuralExecutor`: the
    boolean/field/date FILTER and (for the snip cell) the coverage-tier structure are completely
    unchanged, but the ORDER of filter-passing candidates is PURE dense rank
    (`dense_rank_for_candidates`, dense_fuse.py) instead of RRF(bm25, dense). Both classes are
    thin dispatch subclasses (their own tool names translate to the mirror cell's tool names,
    then delegate) — `BqlVisitWorkspace.run`/`DocSearchFetch.run` are never touched.

Both use the SAME top-k so the two arms see equally many candidates; they differ only in
ACCESS (structured slice vs whole page) — the token gap is the measured result. Scored by
grounded EM/F1 + gold-doc coverage (evaluation.doc_scoring).
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.structural.bql.ast import And, In, Not, Or, Phrase, Prefix, Term
from agent_search.retrievers.structural.bql.executor import (
    StructuralExecutor, execute_bql, _rank_leaves)
from agent_search.retrievers.structural.bql.parser import parse as bql_parse
from agent_search.retrievers.structural.bql.surface import to_bql

# Query-biased snippet width, in whitespace tokens — the size of the best-matching window
# `best_line` picks for each search hit (research_snip / research_indri_snip / research_bm25q
# and every *_fetch_snip cell). A sweepable knob: 32 (default), 64, 128, 256, 512 ... Larger
# windows show the agent more context per hit before it decides what to fetch, at a
# proportional cost in listing tokens.
# NOTE: this default is 32; the pre-knob hardcoded value was 25, so a run that must reproduce
# earlier numbers exactly needs SNIPPET_TOKENS=25.
SNIPPET_TOKENS = int(os.environ.get("SNIPPET_TOKENS", "32"))
# Belt-and-braces character cap on the same window. Scales with SNIPPET_TOKENS at the ratio the
# hardcoded pair used (160 chars / 25 tokens = 6.4), so widening the window is not silently
# undone by a cap sized for a small one. Independently overridable.
SNIPPET_MAX_CHARS = int(os.environ.get("SNIPPET_MAX_CHARS", str(round(SNIPPET_TOKENS * 6.4))))

MAX_VISIT_TOKENS = int(os.environ.get("MAX_VISIT_TOKENS", "1200"))
# Parity fix (docs/bql_failure_forensics.md): fetch's per-section read budget defaults to the
# SAME resolved value as visit's whole-doc budget — the factorial's READ axis is meant to differ
# in WHAT is read (named part vs whole doc), not HOW MUCH can be read. Tracks MAX_VISIT_TOKENS's
# own env override when MAX_SECTION_TOKENS isn't independently set; still independently
# overridable via its own env var exactly as before.
MAX_SECTION_TOKENS = int(os.environ.get("MAX_SECTION_TOKENS", str(MAX_VISIT_TOKENS)))
_INTRO = "(intro)"
# a markdown/wiki heading line: leading #'s (level) then the heading text. The structured
# doc corpora carry `## History` markers in the body; a flat doc has none -> one section.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _cap_tokens(text: str, n: int, tail: str = " …(truncated)") -> str:
    toks = text.split()
    return text if len(toks) <= n else " ".join(toks[:n]) + tail


_DATE_RANGE_PREFIX = "__daterange__"

# --- v2 date-nudge (research_v2 condition, DEFAULT OFF): a mechanical, CORPUS-FREE mid-episode
# nudge — derived only from the agent's own raw query text, never from the corpus — so it can't
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


def sections_from_body(body: str) -> "dict[str, str]":
    """Split a doc body into {heading: text} LIVE on `##` markers (DESIGN.md). Text before
    the first heading is the '(intro)'. A body with no markers is one '(intro)' section (a
    flat doc), so the same fetch contract works on flat and structured corpora alike.
    Duplicate headings are disambiguated ('History', 'History (2)')."""
    out: "dict[str, str]" = {}
    cur, buf = _INTRO, []

    def flush(name: str, lines: list) -> None:
        if not lines and name == _INTRO:
            return
        text = "\n".join(lines).strip()
        key, n = name, 2
        while key in out:                       # disambiguate a repeated heading
            key, n = f"{name} ({n})", n + 1
        out[key] = text

    for line in (body or "").splitlines():
        m = _HEADING.match(line)
        if m:
            flush(cur, buf)
            cur, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    flush(cur, buf)
    if not out:                                 # empty body -> a single empty intro
        out[_INTRO] = ""
    return out


def best_line(u: CodeUnit, terms: "list[str]", width: int = SNIPPET_TOKENS,
              max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """The doc's best-matching ~`width`-token window for `terms` (`width` defaults to
    `SNIPPET_TOKENS`, env-settable) — a single pass over the
    whitespace-tokenized body, incrementally tracking how many DISTINCT `terms` (`code_tokenize`
    'd, case-insensitive) the current window contains as it slides one token at a time (add the
    entering token, drop the leaving one); the highest-scoring window wins, ties -> earliest (a
    strict `>` keeps the first max). Empty `terms` (unparseable query) falls back to the doc's
    opening `width` tokens. Capped at `max_chars` characters — belt and suspenders; a 25-token
    window rarely needs it.

    MODULE-LEVEL (not a method) so it's shared verbatim by `DocSearchFetch._best_line`
    (research_snip/research_indri_snip's leaf-token-driven excerpt) and `Bm25Visit`'s
    query-biased excerpt (research_bm25q) — one best-matching-window implementation, two
    callers with different term sources."""
    toks = ((u.body if u.body is not None else u.code) or "").split()
    if not toks:
        return ""
    term_set = {t.lower() for t in (terms or [])}
    if not term_set:
        return " ".join(toks[:width])[:max_chars]
    tok_terms = [set(code_tokenize(t)) & term_set for t in toks]
    counts: dict = {}
    score = 0

    def _add(i: int) -> None:
        nonlocal score
        for t in tok_terms[i]:
            c = counts.get(t, 0)
            if c == 0:
                score += 1
            counts[t] = c + 1

    def _drop(i: int) -> None:
        nonlocal score
        for t in tok_terms[i]:
            c = counts[t] - 1
            counts[t] = c
            if c == 0:
                score -= 1

    n = len(toks)
    w = min(width, n)
    for i in range(w):
        _add(i)
    best_start, best_score = 0, score
    for start in range(1, n - w + 1):
        _drop(start - 1)
        _add(start + w - 1)
        if score > best_score:
            best_score, best_start = score, start
    return " ".join(toks[best_start:best_start + w])[:max_chars]


def _infobox(u: CodeUnit) -> "dict[str, str]":
    """The doc's infobox facts, if the corpus carried any (structured wiki). Stored in
    metadata['infobox'] as 'k: v; k: v' by the corpus builder, else absent."""
    meta = u.metadata or {}
    raw = str(meta.get("infobox") or "")
    facts: "dict[str, str]" = {}
    for part in raw.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            if k.strip():
                facts[k.strip()] = v.strip()
    return facts


class DocSearchFetch:
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
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if executor is not None:
            self.ex = executor
        else:
            # env STRUCTURED_BACKEND-selectable (default 'python', unchanged) — see
            # agent_search.retrievers.structural.backend.build_bql_engine. Production
            # callers (agent_search.agent.retriever) always pass `executor` explicitly (built
            # with a real index_root/key so a 'lucene' backend opens the prebuilt index); this
            # fallback is for direct/standalone construction (tests, ad-hoc scripts).
            from agent_search.retrievers.structural.backend import build_bql_engine
            self.ex = build_bql_engine(self.units)
        self._sections: dict[str, dict] = {}                  # doc_id -> {heading: text} (lazy)
        self.seen: set = set()
        self.last_hits: list[str] = []
        # BQL v2 (research_v2 condition, DEFAULT OFF): on a 0-exact-hit AND with >=2 children,
        # try constraint-COVERAGE ranking (executor.coverage_topk) before the undifferentiated
        # soft_topk relevance fallback — see _coverage_render below. False reproduces the OLD
        # fallback behavior byte-for-byte (existing `research` condition is unaffected: it never
        # passes coverage=True).
        self.coverage = coverage
        # DEFAULT OFF: emit `_DATE_NUDGE_HINT` after a search whose raw query text carries a
        # bare temporal clue (see `_has_bare_temporal_clue`), capped at `_DATE_NUDGE_CAP` per
        # episode/instance. Wired True ONLY by the docv2 arm (agent/retriever.py) — every other
        # caller of DocSearchFetch (the plain `research` condition, Bm25FetchWorkspace) keeps the
        # old default and is byte-identical.
        self.date_nudge = date_nudge
        self._date_nudge_emitted = 0
        # research_snip (DEFAULT OFF, `snippets=True`): `_render_hits` appends a one-line
        # best-matching excerpt (see `_best_line`) to every search hit. False (the default)
        # reproduces `_render_hits`'s output byte-for-byte — every caller of DocSearchFetch
        # other than the docsnip arm (agent/retriever.py) keeps the old default.
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

        BUGFIX: this used to check only title/section/infobox/body, silently omitting
        author/date — so a browsecomp doc that matched PURELY on IN(author,·)/IN(date,·)
        (its only real structured fields; browsecomp has no section/infobox at all) showed
        "matched: body" or "matched: -", hiding the actual reason from the agent."""
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

    def _best_line(self, u: CodeUnit, leaf_toks: list, width: int = SNIPPET_TOKENS,
                   max_chars: int = SNIPPET_MAX_CHARS) -> str:
        """research_snip (`snippets=True`): the doc's best-matching ~`width`-token window for
        `leaf_toks`. Thin wrapper over the module-level `best_line` (shared verbatim with
        `Bm25Visit`'s query-biased excerpt, research_bm25q) — kept as an instance method so
        every existing caller (this class's `_render_hits`, doc_indri.py, tests) is unaffected."""
        return best_line(u, leaf_toks, width=width, max_chars=max_chars)

    def _render_hits(self, hit_ids: list, leaf_toks: list, header: str) -> str:
        """Render the structure TABLE (rank, doc_id, title, §section names, ib[infobox keys],
        matched fields) for `hit_ids` under `header`. Shared by the exact-BQL-hit path and the
        soft-topk fallback so the two renderings can never drift. Marks every listed doc `seen`
        — the gold-doc-coverage metric must count a fallback hit exactly like an exact hit.

        research_snip (`self.snippets`, DEFAULT OFF): appends a one-line best-matching excerpt
        (`_best_line`) to every hit line. False reproduces this method's output byte-for-byte."""
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
        for line, doc_id in zip(lines[1:], self.last_hits):
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
            # BQL v2 (coverage=True, research_v2 only): a 0-exact-hit AND with >=2 children gets
            # constraint-COVERAGE ranking FIRST — it pinpoints WHICH clause failed, strictly more
            # informative than soft_topk's undifferentiated bag-of-terms relevance. Falls through
            # to the unchanged soft_topk path below if the expr isn't an And (a single constraint
            # has nothing to break down) or coverage_topk finds nothing.
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
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
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
            # BUGFIX: a FLAT doc (browsecomp, no '##' markers) has exactly one section,
            # always named '(intro)' — never a name the agent could plausibly type. Any
            # part name on a single-section doc means "the body" (the skill's own field
            # name for it), so honor it instead of erroring "no section 'body'".
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
            # "search_v2"/"fetch_v2" are the research_v2 condition's tool NAMES (an alias of
            # search/fetch — see tools.yaml's search_fetch_v2 toolset docstring): a distinct
            # tool name is what gets it its own manual, but the workspace method is the same.
            # "search_s"/"fetch_s" are research_snip's tool NAMES, same pattern — the
            # `snippets=True` behavior lives in `_render_hits`, not in this dispatch.
            # "search_bqlds"/"fetch_bqlds" are research_bql_dense_snip's tool NAMES (mirroring
            # research_snip's search_s/fetch_s exactly — see BQL_DENSE / bql/dense_fuse.py):
            # SAME DocSearchFetch(snippets=True) workspace, the only difference is the
            # DENSE-ATTACHED executor `agent/retriever.py`'s 'bqldensesnip' arm builds.
            # "search_bqldf"/"fetch_bqldf" are research_bql_dense_fetch's tool NAMES (mirroring
            # the PLAIN `doc` arm's bare search/fetch exactly — see BQL_DENSE / bql/dense_fuse.py):
            # SAME DocSearchFetch() plain (snippets=False, the constructor default) workspace, the
            # only difference is the DENSE-ATTACHED executor `agent/retriever.py`'s
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


# the SERP listing depth for the retrieve-then-visit baselines: Bm25Visit (research_bm25 /
# research_bm25q) below, and — via its own twin constant DENSE_VISIT_TOPK — DenseVisit
# (research_dense). Same env-knob shape as BM25_FETCH_TOPK/DENSE_FETCH_TOPK. The agent-facing
# tool schema (tools.yaml's bm25_search/bm25q_search) exposes ONLY `query` — no `k` — so this
# env knob is the ONLY control over how many results the SERP listing shows; Bm25Visit.run()
# deliberately ignores any `k` in the tool-call args (a hallucinated arg the schema never
# advertised must not resize the listing). Default 5 reproduces the old hardcoded listing
# byte-for-byte.
BM25_VISIT_TOPK = int(os.environ.get("BM25_VISIT_TOPK", "5"))


class Bm25Visit:
    """The retrieve-then-visit baseline: search(query, k) is plain BM25 over the flat doc
    text (title + body); results show title + a short snippet, NO structure. visit(rank_or_id)
    returns the doc's FULL text (capped). Uncoached (no skill), like the code grep baseline.

    `query_biased` (DEFAULT OFF, research_bm25q): the search listing's per-hit snippet becomes a
    QUERY-BIASED best-matching excerpt (the SAME `best_line` window-scoring the method cells'
    snippets use — see doc_research.py's module-level `best_line`) instead of the doc's fixed
    OPENING slice. Motivated by fairness: real search engines show query-biased snippets in
    their result listing, and our method cells (research_snip/research_indri_snip) already do
    via `_best_line` — the bm25 baseline's opening-snippet listing was a listing-axis advantage
    for OUR method that a hardened baseline must not concede. False (the default) reproduces
    `search`'s rendering byte-for-byte — every caller other than the bm25q arm
    (agent/retriever.py) is unaffected."""

    # tools.yaml's toolset (and therefore the rendered <tools> block the agent sees) names
    # this tool `bm25_search` (distinguishing it from the method's `search`); `run()` accepts
    # both spellings so the workspace matches what a real episode's model actually calls.
    # research_bm25q's toolset instead names it `bm25q_search`/`visit_q` — `__init__` swaps
    # `self.tools` to that pair when `query_biased=True` (an instance override, same pattern
    # DenseVisit uses at the class level for its own tool names).
    tools = ("bm25_search", "visit")

    def __init__(self, units: Sequence[CodeUnit], engine=None, ubyid: Optional[dict] = None,
                 query_biased: bool = False):
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if engine is None:
            # env BM25_BACKEND-selectable (default 'local', unchanged) — see
            # agent_search.retrievers.lexical.build_bm25_engine. Production callers
            # (agent_search.agent.retriever) always pass `engine` explicitly (built with a
            # real index_root/key so a 'pyserini' backend persists); this fallback is for
            # direct/standalone construction (tests, ad-hoc scripts).
            from agent_search.retrievers.lexical import build_bm25_engine
            engine = build_bm25_engine(self.units)
        self.bm = engine
        self.seen: set = set()
        self.last_hits: list[str] = []
        self.query_biased = query_biased
        if query_biased:
            self.tools = ("bm25q_search", "visit_q")

    def search(self, query: str, k: int = BM25_VISIT_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = self.bm.search(query, k=k)
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        # query_biased=False (the default) keeps the OLD opening-char-slice snippet, byte-for-byte.
        terms = code_tokenize(query) if self.query_biased else None
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            if self.query_biased:
                snip = best_line(u, terms)
            else:
                snip = " ".join((u.body or u.code or "")[:120].split())
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)

    def _resolve(self, ref):
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
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
            # BUGFIX: tools.yaml's toolset (and the rendered <tools> spec) names this
            # `bm25_search` — a real episode's model calls it by that name, so `run()` must
            # accept it (a prior version only matched the bare `search`, silently rejecting
            # every real bm25_search call with "unknown tool" and burning the whole episode).
            # `bm25q_search`/`visit_q` are research_bm25q's tool NAMES for this SAME class
            # (`query_biased=True` — see __init__/search above); accepted here too so a real
            # episode's model calling by that name works identically.
            if name in ("bm25_search", "bm25q_search", "search"):
                # listing depth is BM25_VISIT_TOPK (env knob) ONLY — the tool schema exposes
                # just `query`, so a hallucinated `k` arg must not resize the SERP listing.
                return self.search(args.get("query") or args.get("q") or "",
                                   BM25_VISIT_TOPK)
            if name in ("visit", "visit_q"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# the "retrieve-and-read" baseline's listing depth (research_bm25_autoread, Bm25AutoRead below) —
# same env-knob SHAPE as BM25_VISIT_TOPK (tools.yaml's bm25_read_search schema exposes ONLY
# `query`, so this is the ONLY control; Bm25AutoRead.run() ignores a hallucinated `k` arg exactly
# like Bm25Visit.run() does). Default 5 matches BM25_VISIT_TOPK's own default (the two baselines
# show the same-sized pool; they differ only in whether the top-k is a listing to choose from or
# full text handed over unconditionally).
AUTOREAD_TOPK = int(os.environ.get("AUTOREAD_TOPK", "5"))


class Bm25AutoRead(Bm25Visit):
    """The "retrieve-and-read" baseline (research_bm25_autoread): SAME plain BM25 ranking as
    `Bm25Visit` (over the flat doc title+body — same engine, same env `BM25_BACKEND` fallback),
    but `search(query)` itself renders the FULL TEXT of every one of the top `AUTOREAD_TOPK`
    hits — there is NO visit/fetch tool at all in this condition; a search call IS the read.
    Each hit's text is capped at MAX_VISIT_TOKENS via the SAME `_cap_tokens` helper (same cap,
    same " …(truncated — this is the whole-doc cap)" marker) `Bm25Visit.visit` uses — so a
    single doc's rendered length is identical to what a `research_bm25` episode gets FROM
    visiting that same rank, only here every top-k hit is rendered, unconditionally, on every
    search call.

    Sits between the one-shot RAG baseline (scripts/oneshot_rag.py — a single stuffed prompt,
    no agent loop, no per-query retrieval choice at all) and the SERP retrieve-then-visit
    baseline (research_bm25 — a listing the agent must then CHOOSE to visit): isolates the value
    of the agent's choose-what-to-read step by removing it entirely, while keeping everything
    else (BM25 ranking, the agent loop, the <answer> contract) identical to research_bm25.
    Uncoached (no skill), like the bm25/dense baselines it's built from.

    tools.yaml's `bm25_autoread` toolset names this tool `bm25_read_search` (a distinct name
    from `bm25_search`, since its description says results already include full document text)
    — `run()` accepts both `bm25_read_search` and the bare `search`. There is deliberately NO
    `visit`/`visit_q`/`visit_d`/`fetch` dispatch: a real episode's model attempting one of those
    (a hallucinated call, or a habit carried over from a sibling condition's manual/memory) gets
    a clear ERROR string explaining why, not a silent "unknown tool" or a reused baseline read."""

    # ONE tool only — no read/visit tool exists in this condition (search already returns full
    # text). Overrides Bm25Visit's class-level `tools = ("bm25_search", "visit")`.
    tools = ("bm25_read_search",)

    def __init__(self, units: Sequence[CodeUnit], engine=None, ubyid: Optional[dict] = None):
        # SAME construction as Bm25Visit (query_biased is irrelevant here — there is no listing
        # snippet to bias, the search response IS the full text) — pass the default explicitly
        # for clarity, then restore this class's own `tools` (Bm25Visit.__init__ only swaps
        # `self.tools` when `query_biased=True`, so this is a no-op today, but explicit is safer
        # than relying on that not changing).
        super().__init__(units, engine=engine, ubyid=ubyid, query_biased=False)
        self.tools = ("bm25_read_search",)

    def search(self, query: str, k: int = AUTOREAD_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = self.bm.search(query, k=k)
        if not ids:
            prior = ("  (previous results still available)" if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            # SAME cap/marker as Bm25Visit.visit — a doc read via autoread's search is truncated
            # identically to the same doc read via research_bm25's visit.
            text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                               " …(truncated — this is the whole-doc cap)")
            blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
        return "\n\n".join(blocks)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("bm25_read_search", "search"):
                # listing depth is AUTOREAD_TOPK (env knob) ONLY — the tool schema exposes just
                # `query`, so a hallucinated `k` arg must not resize how many docs are read.
                return self.search(args.get("query") or args.get("q") or "", AUTOREAD_TOPK)
            if name in ("visit", "visit_q", "visit_d", "visit_v", "visit_h", "visit_bv",
                       "visit_bqld", "fetch"):
                return ("ERROR: no visit tool in this condition — search already returns full "
                        "documents.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# the dense twin of BM25_VISIT_TOPK — research_dense's SERP listing depth (DenseVisit below).
# Same rules: tools.yaml's dense_search schema exposes ONLY `query`, so this env knob is the
# ONLY control (DenseVisit.run() ignores a hallucinated `k` arg); default 5 reproduces the old
# hardcoded listing byte-for-byte. A separate knob (not BM25_VISIT_TOPK reused) so the two
# retrieve-then-visit arms' listing depths stay independently settable.
DENSE_VISIT_TOPK = int(os.environ.get("DENSE_VISIT_TOPK", "5"))


class DenseVisit(Bm25Visit):
    """The DENSE retrieve-then-visit baseline (the modern RAG default): search(query, k) is
    dense embedding cosine similarity over the document corpus — otherwise BYTE-IDENTICAL to
    `Bm25Visit` (same rendering: rank, doc_id, title, opening snippet; same `visit`/`_resolve`,
    inherited unchanged). Uncoached (no skill), like the bm25 baseline.

    `engine` is a `DenseBelief` (agent_search.retrievers.structural.indri.dense_belief) already
    `build_or_load`ed over this corpus — reused, not reimplemented: `top_k_doc_ids(query, k)`
    is DenseBelief's memoized query-encode + cosine-top-k over its cached embedding matrix (the
    SAME persisted `indexes/dense/<model>-sl<len>/<corpus_key>/` cache the `dense` retriever
    condition and Indri's dense belief already build/persist — see DEFAULT_MODEL there,
    BAAI/bge-base-en-v1.5). retriever.py's 'densevisit' arm builds + validates that cache at
    index() time and raises a CLEAR error if it's missing, rather than silently falling back to
    live-encoding the whole corpus (which this baseline does NOT support). Tests inject a stub
    `engine` exposing `top_k_doc_ids(query, k)` the same way test_doc_bm25_fetch_tools injects a
    stub BM25 engine for `Bm25Visit`."""

    # tools.yaml's toolset names this tool `dense_search` (distinguishing it from the bm25 arm's
    # `bm25_search`); `run()` accepts both `dense_search` and the bare `search`. `visit_d` is
    # tools.yaml's name for the (byte-identical) whole-doc read; `run()` accepts `visit` too.
    tools = ("dense_search", "visit_d")

    def __init__(self, units: Sequence[CodeUnit], engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None):
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if engine is None:
            from agent_search.retrievers.structural.indri.dense_belief import DenseBelief
            engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.engine = engine
        self.seen: set = set()
        self.last_hits: list[str] = []

    def search(self, query: str, k: int = DENSE_VISIT_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k) or [])
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = ids
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            snip = " ".join((u.body or u.code or "")[:120].split())
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_search", "search"):
                # listing depth is DENSE_VISIT_TOPK (env knob) ONLY — the tool schema exposes
                # just `query`, so a hallucinated `k` arg must not resize the SERP listing.
                return self.search(args.get("query") or args.get("q") or "",
                                   DENSE_VISIT_TOPK)
            if name in ("visit_d", "visit"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class DenseAutoRead(DenseVisit):
    """The DENSE "retrieve-and-read" baseline (research_dense_autoread): SAME dense-embedding
    ranking as `DenseVisit` (cosine similarity over the SAME persisted
    `indexes/dense/<model>-sl<len>/<corpus_key>/` doc-embedding cache, via
    `DenseBelief.top_k_doc_ids` — same engine, same fail-loud missing-cache contract), but
    `search(query)` itself renders the FULL TEXT of every one of the top `AUTOREAD_TOPK` hits —
    there is NO visit/fetch tool at all in this condition; a search call IS the read. Each hit's
    text is capped at MAX_VISIT_TOKENS via the SAME `_cap_tokens` helper (same cap, same
    " …(truncated — this is the whole-doc cap)" marker) `DenseVisit.visit`/`Bm25AutoRead.search`
    use — so a single doc's rendered length is identical to what a `research_dense` episode gets
    FROM visiting that same rank, only here every top-k hit is rendered, unconditionally, on
    every search call.

    The dense analog of `Bm25AutoRead` exactly as `DenseVisit` is the dense analog of
    `Bm25Visit`: reuses the SAME `AUTOREAD_TOPK` env knob (not a second one — the two autoread
    baselines show the same-sized pool) and the SAME MAX_VISIT_TOKENS cap. The ONLY difference
    from `research_bm25_autoread` is the retrieval engine (dense cosine similarity instead of
    BM25 keyword search) — auto-read behavior, top-k, the per-doc cap, and the <answer> contract
    are all identical. Uncoached (no skill), like the bm25/dense baselines it's built from.

    tools.yaml's `dense_autoread` toolset names this tool `dense_read_search` (a distinct name
    from `dense_search`, since its description says results already include full document text)
    — `run()` accepts both `dense_read_search` and the bare `search`. There is deliberately NO
    `visit`/`visit_d`/`visit_q`/`visit_v`/`fetch` dispatch: a real episode's model attempting one
    of those (a hallucinated call, or a habit carried over from a sibling condition's manual/
    memory) gets a clear ERROR string explaining why, not a silent "unknown tool" or a reused
    baseline read."""

    # ONE tool only — no read/visit tool exists in this condition (search already returns full
    # text). Overrides DenseVisit's class-level `tools = ("dense_search", "visit_d")`.
    tools = ("dense_read_search",)

    def __init__(self, units: Sequence[CodeUnit], engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None):
        # SAME construction as DenseVisit (engine/corpus_key resolve the SAME persisted
        # doc-embedding cache) — `tools` is already this class's own value via the class
        # attribute above; DenseVisit.__init__ never overrides `self.tools` (unlike Bm25Visit's
        # query_biased branch), so no restore-after-super() is needed here.
        super().__init__(units, engine=engine, ubyid=ubyid, corpus_key=corpus_key)

    def search(self, query: str, k: int = AUTOREAD_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k) or [])
        if not ids:
            prior = ("  (previous results still available)" if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = ids
        blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            # SAME cap/marker as DenseVisit.visit (inherited from Bm25Visit) — a doc read via
            # autoread's search is truncated identically to the same doc read via
            # research_dense's visit_d.
            text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                               " …(truncated — this is the whole-doc cap)")
            blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
        return "\n\n".join(blocks)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_read_search", "search"):
                # listing depth is AUTOREAD_TOPK (env knob) ONLY — the tool schema exposes just
                # `query`, so a hallucinated `k` arg must not resize how many docs are read.
                return self.search(args.get("query") or args.get("q") or "", AUTOREAD_TOPK)
            if name in ("visit", "visit_d", "visit_q", "visit_v", "visit_h", "visit_bv",
                       "visit_bqld", "fetch"):
                return ("ERROR: no visit tool in this condition — search already returns full "
                        "documents.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# a fixed bm25 top-k (not the CLI's --k sweep, which sizes the episode's location-ranking
# cutoff, not the retrieval pool the fetch reads from) — SAME constant/role as doc_bm25_dci's
# BM25_DCI_TOPK, so the two bm25-retrieval baselines stage the same-sized pool.
BM25_FETCH_TOPK = int(os.environ.get("BM25_FETCH_TOPK", "10"))


class Bm25FetchWorkspace(DocSearchFetch):
    """The RISE-style hybrid: bm25 LIVE RETRIEVAL + BQL structured SECTION-FETCH read.

    This is `DocSearchFetch`'s READ side (search lists STRUCTURE — section names + infobox
    keys, no bodies; fetch pulls one named section) placed on `Bm25Visit`'s RETRIEVAL (plain
    BM25 over flat doc text). So the ONLY difference from `research` (the method) is the
    retrieval surface — bm25 instead of the BQL executor — and the ONLY difference from
    `research_bm25` is the read (a named section instead of the whole doc). Completing the
    controlled set with `research_bm25_dci` (same bm25 retrieval, shell read):

        research           : BQL search  -> fetch a section   (the method)
        research_bm25      : bm25 search  -> visit whole doc   (retrieve-then-visit)
        research_bm25_dci  : bm25 search  -> bash/read shell   (retrieval-bounded brute-force)
        research_bm25_fetch: bm25 search  -> fetch a section   (this — retrieval-bounded structured read)

    Retrieval is LIVE, per `bm25_search` call — byte-identical retrieval BEHAVIOR to
    research_bm25/Bm25Visit for the same query/engine/k (each call re-runs bm25 on the
    agent's own query string; nothing is fixed at construction). This makes the controlled
    comparison honest: the ONLY difference from `research_bm25` is the READ (named-section
    fetch vs whole-doc visit), and the ONLY difference from `research` is the RETRIEVAL
    surface (plain BM25 vs field-tagged Boolean) — not "one arm re-retrieves and the other
    doesn't". Uncoached (no BQL skill — retrieval is a plain search box)."""

    # tools.yaml's `bm25_fetch` toolset names the retrieval tool `bm25_search` (matching the
    # other bm25 arms, since retrieval is the same operation); `fetch` is DocSearchFetch's.
    tools = ("bm25_search", "fetch")

    def __init__(self, units: Sequence[CodeUnit], query: str, engine=None,
                 ubyid: Optional[dict] = None, topk: int = BM25_FETCH_TOPK):
        # DocSearchFetch.__init__ builds the section cache + resolves ubyid; we do NOT need its
        # BQL StructuralExecutor (retrieval is bm25 here), so pass a lightweight executor stand-in
        # is avoided — instead call super() and simply never use self.ex. But super() would build
        # a StructuralExecutor over all units (O(N) postings) we never query, so set it up by hand.
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.ex = None                                   # unused: retrieval is bm25, not BQL
        self._sections: dict[str, dict] = {}
        self.seen: set = set()
        if engine is None:
            # SAME env BM25_BACKEND fallback as Bm25Visit above — see that constructor's comment.
            from agent_search.retrievers.lexical import build_bm25_engine
            engine = build_bm25_engine(self.units)
        self.bm = engine
        self.topk = topk
        # No retrieval at construction: `self.query` is kept only as the `bm25_search` fallback
        # when a tool call omits `query` (mirrors the episode's opening question); every actual
        # `bm25_search` call retrieves LIVE against the query it was given (see search() below).
        self.query = (query or "").strip()
        self.last_hits: list[str] = []

    # -- bm25_search: LIVE bm25 retrieval, rendered as a STRUCTURE table (no bodies) -------

    def search(self, query: str, k: Optional[int] = None) -> str:
        # LIVE per call — exactly like Bm25Visit.search (SAME engine/call shape, so retrieval is
        # byte-identical to research_bm25 for the same query/k). Same STRUCTURE listing as
        # DocSearchFetch.search (title + §[section names] + ib[infobox keys], NO bodies), so the
        # agent knows what to fetch; just driven by bm25 hits, not BQL leaves (so no `matched:`
        # field — bm25 has no field-tagged leaves to attribute).
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.bm.search(query, k=k or self.topk))
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            lines.append(f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("bm25_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim to DocSearchFetch's fetch (arg-normalization +
                # section logic) — the whole point is that only retrieval differs from `research`.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class Bm25FetchSnipWorkspace(Bm25FetchWorkspace):
    """`Bm25FetchWorkspace` (research_bm25_fetch) with a CONTENT-BEARING search listing —
    the fair-listing sibling that completes the grid. Every other method fetch cell
    (research_snip, research_indri_snip, research_dense_fetch) shows each hit's best-matching
    excerpt (`best_line`) in its search listing; research_bm25_fetch's listing was
    CONTENT-BLIND (structure only — no excerpt), an inconsistency across the grid. This
    workspace fixes exactly that, ONE deliberate departure from `Bm25FetchWorkspace`'s bare
    structure table — the SAME amendment `DenseFetchWorkspace` makes over `Bm25FetchWorkspace`'s
    own bare table (see its docstring) and `IndriVisitWorkspace` makes over
    `IndriFetchWorkspace`'s default-off snippets: a content-blind listing handicaps a cell once
    ANY sibling cell shows content.

    Retrieval, fetch, and construction are ALL inherited from `Bm25FetchWorkspace` UNCHANGED
    (same live-per-call bm25 search, same `topk`/`engine`/`query` construction, same
    `fetch`-delegates-to-DocSearchFetch machinery) — `search` is the ONLY override, and it is
    byte-for-byte `Bm25FetchWorkspace.search` plus ONE appended `» excerpt` line per hit
    (`_best_line`, the SAME module-level `best_line` window-scoring research_snip/
    research_indri_snip/research_bm25q/research_dense_fetch already share). Excerpt terms come
    from the raw query text (`code_tokenize`) — bm25 has no BQL leaves to draw from, the SAME
    term source `Bm25Visit.query_biased`/`DenseFetchWorkspace` use. So the ranking is
    IDENTICAL to `Bm25FetchWorkspace` for the same query/engine/k (same retrieval call, same
    top-k doc ids) — the ONLY difference from `research_bm25_fetch` is the listing's content,
    matching this cell's factorial-grid position exactly: {bm25 search} x {structure->parts
    read}, now WITH the same fairness-parity excerpt every other fetch cell carries."""

    # tools.yaml's `bm25_fetch_snip` toolset names the retrieval tool `bm25_search_snip`
    # (distinguishing it from `bm25_search`, the content-blind sibling's tool, so each gets its
    # own manual/description entry — SAME naming move `dense_search_f` makes over `dense_search`);
    # `fetch` is DocSearchFetch's, reused unmodified.
    tools = ("bm25_search_snip", "fetch")

    # -- bm25_search_snip: LIVE bm25 retrieval, rendered as a STRUCTURE table + excerpt -------

    def search(self, query: str, k: Optional[int] = None) -> str:
        # IDENTICAL retrieval call to Bm25FetchWorkspace.search (same engine, same query, same
        # topk fallback) — ranking parity is a property of this line alone; only the rendering
        # below differs (an appended excerpt per hit).
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.bm.search(query, k=k or self.topk))
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self._best_line(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("bm25_search_snip", "bm25_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim, exactly like Bm25FetchWorkspace — the only
                # difference from `research_bm25_fetch` is the search listing's content.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# same role/constant shape as BM25_FETCH_TOPK, for the dense-retrieval twin below.
DENSE_FETCH_TOPK = int(os.environ.get("DENSE_FETCH_TOPK", "10"))


class DenseFetchWorkspace(DocSearchFetch):
    """The DENSE-retrieval structured-read ACI: dense embedding LIVE RETRIEVAL + BQL structured
    SECTION-FETCH read. This fills the search x read factorial cell {dense search} x
    {structure->parts read} — Bm25FetchWorkspace already fills {bm25 search} x {structure->parts
    read} (research_bm25_fetch), DenseVisit fills {dense search} x {whole-doc visit}
    (research_dense). Wired EXACTLY like Bm25FetchWorkspace (see its docstring): retrieval is
    LIVE per `dense_search_f` call (`engine.top_k_doc_ids(query, k)` re-run on the agent's OWN
    query text every time — nothing fixed at construction, same live-retrieval fix
    Bm25FetchWorkspace already made over the old one-shot design); the section-fetch machinery
    (`fetch`) is INHERITED from `DocSearchFetch` unchanged, never duplicated.

    ONE deliberate departure from Bm25FetchWorkspace's bare structure table: research_snip
    established that a content-bearing listing (a one-line best-matching excerpt per hit,
    `best_line` — the SAME window-scoring research_snip/research_indri_snip/research_bm25q use)
    is the fair default once ANY arm shows one, and `dense_search_f`'s tool description promises
    exactly that ("... plus a best-matching excerpt — NOT full text; fetch a named section to
    read"). So this workspace's listing is Bm25FetchWorkspace's STRUCTURE table (rank, doc_id,
    title, §[section names], ib[infobox keys]) with ONE appended `» excerpt` line per hit.
    Excerpt terms come from the raw query text (`code_tokenize`) — dense retrieval has no BQL
    leaves to draw from, the same term source Bm25Visit's `query_biased` (research_bm25q) excerpt
    uses.

    `engine` is a `DenseBelief` already `build_or_load`ed over this corpus (the SAME persisted
    `indexes/dense/<model>-sl<len>/<corpus_key>/` cache DenseVisit/the `dense` retriever
    condition already build) — retriever.py's 'densefetch' arm builds + validates that cache at
    index() time and raises the SAME clear RuntimeError as the 'densevisit' arm if it's missing
    (this baseline never live-encodes a whole corpus at eval time). Uncoached (no skill) — like
    Bm25FetchWorkspace, retrieval is a plain search box and section-fetch needs no BQL skill (its
    listing already names the sections)."""

    # tools.yaml's `dense_fetch` toolset names the retrieval tool `dense_search_f` (distinguishing
    # it from `dense_search`, DenseVisit's whole-doc-read tool, so each gets its own manual/entry
    # even though neither renders one — UNCOACHED); `fetch` is DocSearchFetch's.
    tools = ("dense_search_f", "fetch")

    def __init__(self, units: Sequence[CodeUnit], query: str, engine=None,
                 ubyid: Optional[dict] = None, topk: int = DENSE_FETCH_TOPK,
                 corpus_key: Optional[str] = None):
        # SAME construction shape as Bm25FetchWorkspace.__init__: DocSearchFetch.__init__ would
        # build a StructuralExecutor over all units (O(N) postings) we never query (retrieval is
        # dense, not BQL) — set the essentials up by hand instead of calling super().__init__.
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.ex = None                                   # unused: retrieval is dense, not BQL
        self._sections: dict[str, dict] = {}
        self.seen: set = set()
        if engine is None:
            from agent_search.retrievers.structural.indri.dense_belief import DenseBelief
            engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.engine = engine
        self.topk = topk
        # No retrieval at construction: `self.query` is kept only as the `dense_search_f`
        # fallback when a tool call omits `query` (mirrors Bm25FetchWorkspace.query); every
        # actual call retrieves LIVE against the query it was given (see search() below).
        self.query = (query or "").strip()
        self.last_hits: list[str] = []

    # -- dense_search_f: LIVE dense retrieval, rendered as a STRUCTURE table + excerpt -------

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k or self.topk) or [])
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self._best_line(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_search_f", "dense_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim to DocSearchFetch's fetch, exactly like
                # Bm25FetchWorkspace — the only difference from `research` is retrieval, and the
                # only difference from `research_bm25_fetch` is the retrieval ENGINE.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class DenseFetchPlainWorkspace(DenseFetchWorkspace):
    """`DenseFetchWorkspace` (research_dense_fetch, "dense+snip fetch") with the excerpt REMOVED
    — the missing PLAIN (no-excerpt) dense fetch cell. Every other query engine's fetch family
    already has both a plain and a snip sibling: Bm25FetchWorkspace (research_bm25_fetch, plain)
    vs Bm25FetchSnipWorkspace (research_bm25_fetch_snip, excerpt); IndriFetchWorkspace's own
    `snippets=False` default (research_indri) vs `snippets=True` (research_indri_snip); the BQL
    method's plain `search`/`fetch` (research) vs `DocSearchFetch(snippets=True)`
    (research_snip) — and the SAME split for the dense-fused BQL executor,
    `DocSearchFetch()`/'bqldensefetch' (research_bql_dense_fetch) vs
    `DocSearchFetch(snippets=True)`/'bqldensesnip' (research_bql_dense_snip). `DenseFetchWorkspace`
    alone never got the split — its `search()` has no `snippets` flag at all and appends the
    `» excerpt` line unconditionally — so dense was the one query engine (bm25/bql/indri/dense)
    lacking a plain fetch cell. This workspace supplies exactly that missing sibling, the same
    way `research_bql_dense_fetch` supplies `research_bql_dense_snip`'s missing plain sibling for
    the dense-fused BQL executor (see that condition's doc_research.py/conditions.yaml comments).

    Retrieval, construction, and the section-`fetch` read are ALL inherited from
    `DenseFetchWorkspace` UNCHANGED (same live-per-call `engine.top_k_doc_ids`, same
    `topk`/`engine`/`corpus_key` construction, same `fetch`-delegates-to-DocSearchFetch
    machinery) — `search` is the ONLY override, and it is byte-for-byte
    `DenseFetchWorkspace.search` with the `code_tokenize`/`_best_line` excerpt computation and its
    appended `» excerpt` line removed; everything else (rank/doc_id/title/§sections/ib[infobox])
    renders identically. Ranking is therefore IDENTICAL to `research_dense_fetch` for the same
    query/engine/k — the ONLY difference from `research_dense_fetch` is the listing's content (no
    per-hit excerpt), isolating exactly what the snippet contributes on top of pure dense
    retrieval + structured section-fetch, the SAME isolation `research_bql_dense_fetch` performs
    for the dense-fused BQL executor. `DenseFetchWorkspace` itself is left completely untouched —
    this is a new sibling class, not a modification."""

    # tools.yaml's `dense_fetch_plain` toolset names the retrieval tool `dense_search_fp`
    # (distinguishing it from `dense_search_f`, the excerpt-bearing sibling's tool, so each gets
    # its own manual/description entry — SAME naming move `dense_search_f` makes over
    # `dense_search`); `fetch` is DocSearchFetch's, reused unmodified.
    tools = ("dense_search_fp", "fetch")

    # -- dense_search_fp: LIVE dense retrieval, rendered as a PLAIN STRUCTURE table (no excerpt) --

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        ids = list(self.engine.top_k_doc_ids(query, k=k or self.topk) or [])
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            # NO `» excerpt` line — the ONLY rendering difference from DenseFetchWorkspace.search.
            lines.append(f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("dense_search_fp", "dense_search_f", "dense_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim, exactly like DenseFetchWorkspace/Bm25FetchWorkspace
                # — the only difference from `research_dense_fetch` is the listing's content.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


class BqlVisitWorkspace(DocSearchFetch):
    """search_bv(query, k) -> the SAME BQL v2 field-tagged search `research_v2` uses
    (`DocSearchFetch(coverage=True, date_nudge=True)`: executor `coverage_topk` ranking on a
    0-exact-hit AND, plus the typed date[RANGE] mid-episode nudge — see
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
    `research`/`research_v2` already fill {BQL search} x {structure->parts read};
    `research_bm25`/`research_dense`/`research_indri_visit` fill {bm25/dense/indri search} x
    {whole-doc visit}. No fetch/structure tools — visit is the ONLY read here."""

    tools = ("search_bv", "visit_bv")

    def __init__(self, units: Sequence[CodeUnit],
                 executor: Optional[StructuralExecutor] = None,
                 ubyid: Optional[dict] = None,
                 date_nudge: bool = True,
                 tool_names: Optional[tuple] = None):
        super().__init__(units, executor=executor, ubyid=ubyid, coverage=True,
                         date_nudge=date_nudge, snippets=True)  # snippets ALWAYS on — not a caller knob
        # `tool_names` (DEFAULT None -> the class default above, byte-identical to before this
        # param existed): research_bql_dense_visit's ONLY difference from research_bql_visit is
        # the ranking (a DENSE-ATTACHED `executor`, wired by agent/retriever.py's
        # 'bqldensevisit' arm — see BQL_DENSE / bql/dense_fuse.py) — but it needs its OWN tool
        # NAMES (search_bqld/visit_bqld) so tools.yaml/sdk_driver.py can give it its own
        # <tools> listing entry, the SAME reason every other sibling condition in this module
        # gets a distinct tool name (search_s/fetch_s, search_bv/visit_bv, ...). `run()` below
        # accepts BOTH name sets unconditionally (harmless — a plain research_bql_visit episode
        # never sees "search_bqld" advertised in its own toolset, so a real model never calls it).
        if tool_names is not None:
            self.tools = tuple(tool_names)

    # -- visit_bv: whole-doc read, mirroring Bm25Visit.visit/_resolve exactly ------------------

    def _resolve(self, ref):
        """Mirrors `Bm25Visit._resolve` (and `IndriVisitWorkspace._resolve`) exactly: a rank
        (int or numeric string) resolves against `self.last_hits`; a digit that is NOT a valid
        rank may itself be a real doc_id (numeric doc_ids do occur), so try that before
        erroring; otherwise fall back to a doc_id / title lookup."""
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
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
            # "search"/"search_v2" aliases: the SAME BQL search research_v2's `search`/`search_v2`
            # tool names dispatch to (this workspace's constructor is byte-for-byte research_v2's
            # DocSearchFetch(coverage=True, date_nudge=True), plus forced snippets) — a real
            # episode's model, or the KeywordPolicy stub (which opens with a bare `search` call),
            # must reach it under any of these names.
            # "search_bqld"/"visit_bqld" are research_bql_dense_visit's tool NAMES (BQL_DENSE —
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


# --- BM25+DENSE HYBRID baseline: Reciprocal Rank Fusion over two independent rankings -------
#
# The control that isolates the BQL method's contribution from the mere "sparse+dense fusion"
# effect (see module docstring's HybridVisit/HybridFetchSnipWorkspace paragraph for the full
# motivation). `RRF_K` is the standard RRF constant (Cormack, Clarke & Buettcher 2009 pin
# k=60 as the value that is robust across collections/query types — not tuned here);
# `HYBRID_POOL` is how deep EACH of the two rankers is queried before fusion.
RRF_K = int(os.environ.get("RRF_K", "60"))
HYBRID_POOL = int(os.environ.get("HYBRID_POOL", "100"))       # per-ranker pool depth, pre-fusion


def rrf_fuse(bm25_ids: "Sequence[str]", dense_ids: "Sequence[str]", k: int = RRF_K,
            topk: Optional[int] = None) -> list:
    """Reciprocal Rank Fusion over two ranked doc_id lists (a BM25 pool, a dense pool).

        score(d) = sum_i  1 / (k + rank_i(d))     for each list i in which d appears
                                                    (rank_i is 1-based; a doc absent from a
                                                    list contributes NOTHING for it — never
                                                    an infinite/penalized rank)

    Docs are ranked by DESCENDING score; ties are broken by doc_id ASCENDING (deterministic —
    matches FlatIndex.search's own tie-break convention, so results are reproducible even when
    two docs land on an identical fused score, e.g. both absent from one list and tied in the
    other). `topk` truncates the returned list (None = every doc_id appearing in EITHER list,
    i.e. the full union, still score-sorted).

    Pure function of the two id lists — no corpus/engine access — so it is unit-testable
    against a hand-computed fixture with no retrieval stack at all (see tests/test_hybrid.py)."""
    scores: dict = {}
    for ids in (bm25_ids, dense_ids):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return ranked[:topk] if topk is not None else ranked


class HybridVisit(Bm25Visit):
    """The BM25+DENSE HYBRID retrieve-then-visit baseline (research_hybrid) — see the module
    docstring's HybridVisit/HybridFetchSnipWorkspace paragraph for the full RRF formula/pool
    sizes. `search(query, k)` queries BOTH rankers to `HYBRID_POOL` (100) depth, fuses with
    `rrf_fuse` (RRF_K=60), and renders the fused top-`k` EXACTLY like `Bm25Visit.search`
    (title + a short opening snippet, no query bias) — mirroring Bm25Visit's listing format
    byte-for-byte, only the ranking differs. `visit`/`_resolve` are INHERITED from `Bm25Visit`
    unchanged (whole-doc read, capped at MAX_VISIT_TOKENS). Uncoached (no skill), like the
    bm25/dense baselines it's built from.

    `bm25_engine` falls back to `build_bm25_engine` (env `BM25_BACKEND`-selectable, same
    fallback every bm25-consuming workspace uses — production callers, agent/retriever.py,
    always pass a real engine). `dense_engine` falls back to a fresh `DenseBelief.build_or_load`
    (same persisted `indexes/dense/<model>-sl<len>/<key>/` cache DenseVisit/DenseFetchWorkspace
    use; production callers pass the SAME prebuilt-and-validated belief those arms use)."""

    tools = ("hybrid_search", "visit_h")

    def __init__(self, units: Sequence[CodeUnit], bm25_engine=None, dense_engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None,
                 pool: int = HYBRID_POOL):
        # SAME manual-construction shape as DenseVisit.__init__ (not Bm25Visit.__init__'s
        # super() call — this workspace owns TWO engines, neither is "the" engine a bare
        # super().__init__ would wire to self.bm alone).
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        if bm25_engine is None:
            from agent_search.retrievers.lexical import build_bm25_engine
            bm25_engine = build_bm25_engine(self.units)
        self.bm = bm25_engine
        if dense_engine is None:
            from agent_search.retrievers.structural.indri.dense_belief import DenseBelief
            dense_engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.dense_engine = dense_engine
        self.pool = pool
        self.seen: set = set()
        self.last_hits: list[str] = []

    def search(self, query: str, k: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        bm25_ids = list(self.bm.search(query, k=self.pool))
        dense_ids = list(self.dense_engine.top_k_doc_ids(query, k=self.pool) or [])
        ids = rrf_fuse(bm25_ids, dense_ids, topk=k)
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        # IDENTICAL rendering to Bm25Visit.search's query_biased=False branch (opening slice,
        # not query-biased) — mirrors Bm25Visit's listing format byte-for-byte.
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            snip = " ".join((u.body or u.code or "")[:120].split())
            lines.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}  {snip}…")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("hybrid_search", "search"):
                return self.search(args.get("query") or args.get("q") or "",
                                   int(args.get("k", 5) or 5))
            if name in ("visit_h", "visit"):
                return self.visit(args.get("rank") or args.get("id") or args.get("doc")
                                  or args.get("doc_id"))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# same role/default as BM25_FETCH_TOPK/DENSE_FETCH_TOPK, for the hybrid fetch-mode twin below —
# this is the POST-FUSION count returned per search call, distinct from HYBRID_POOL (the
# PRE-fusion depth each of the two rankers is queried to).
HYBRID_FETCH_TOPK = int(os.environ.get("HYBRID_FETCH_TOPK", "10"))


class HybridFetchSnipWorkspace(DocSearchFetch):
    """The BM25+DENSE HYBRID {search} x {structure->parts read} cell (research_hybrid_fetch_snip)
    — the fetch-mode twin of `HybridVisit`, mirroring `Bm25FetchSnipWorkspace`'s listing shape
    exactly: the SAME RRF-fused retrieval as `HybridVisit` (see its docstring / `rrf_fuse` for
    the formula and `HYBRID_POOL`), rendered as the bm25/dense fetch cells' STRUCTURE table
    (rank, doc_id, title, §[section names], ib[infobox keys]) plus a one-line query-biased
    best-matching excerpt per hit (`_best_line`, inherited from `DocSearchFetch`), paired with
    the SAME structured section-`fetch` read `research`/`research_bm25_fetch`/
    `research_dense_fetch` use (inherited from `DocSearchFetch`, unmodified — delegated
    verbatim in `run`, exactly like `Bm25FetchWorkspace`/`DenseFetchWorkspace`).

    Retrieval is LIVE per `hybrid_search_snip` call — BOTH pools are re-queried on the agent's
    own query text every call, nothing fixed at construction — mirroring `Bm25FetchWorkspace`/
    `DenseFetchWorkspace`'s live-per-call retrieval exactly. Uncoached (no skill) — like every
    other bm25/dense-family fetch arm; section-fetch needs no BQL skill (its listing already
    names the sections)."""

    tools = ("hybrid_search_snip", "fetch")

    def __init__(self, units: Sequence[CodeUnit], query: str, bm25_engine=None,
                 dense_engine=None, ubyid: Optional[dict] = None,
                 topk: int = HYBRID_FETCH_TOPK, pool: int = HYBRID_POOL,
                 corpus_key: Optional[str] = None):
        # SAME manual-construction shape as Bm25FetchWorkspace/DenseFetchWorkspace: skip
        # DocSearchFetch.__init__'s BQL StructuralExecutor build (retrieval here is bm25+dense
        # fusion, not BQL) and set the essentials up by hand.
        self.units = list(units)
        self.ubyid = ubyid if ubyid is not None else {u.doc_id: u for u in self.units}
        self.ex = None                                   # unused: retrieval is bm25+dense, not BQL
        self._sections: dict[str, dict] = {}
        self.seen: set = set()
        if bm25_engine is None:
            from agent_search.retrievers.lexical import build_bm25_engine
            bm25_engine = build_bm25_engine(self.units)
        self.bm = bm25_engine
        if dense_engine is None:
            from agent_search.retrievers.structural.indri.dense_belief import DenseBelief
            dense_engine = DenseBelief().build_or_load(self.units, key=corpus_key)
        self.dense_engine = dense_engine
        self.topk = topk
        self.pool = pool
        # No retrieval at construction: `self.query` is kept only as the `hybrid_search_snip`
        # fallback when a tool call omits `query` (mirrors Bm25FetchWorkspace.query); every
        # actual call retrieves LIVE against the query it was given (see search() below).
        self.query = (query or "").strip()
        self.last_hits: list[str] = []

    # -- hybrid_search_snip: LIVE bm25+dense retrieval, RRF-fused, rendered as a STRUCTURE
    # table + excerpt (IDENTICAL rendering shape to Bm25FetchSnipWorkspace.search/
    # DenseFetchWorkspace.search — only the ranking differs) -------------------------------

    def search(self, query: str, k: Optional[int] = None) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        bm25_ids = list(self.bm.search(query, k=self.pool))
        dense_ids = list(self.dense_engine.top_k_doc_ids(query, k=self.pool) or [])
        ids = rrf_fuse(bm25_ids, dense_ids, topk=k or self.topk)
        if not ids:
            prior = ("  (previous results still available to fetch)"
                     if self.last_hits else "")
            return (f"search: {query}   (0 matches){prior}"
                    "\nhint: loosen the query — fewer/shorter terms, drop a field "
                    "scope, or OR name variants.")
        self.last_hits = ids
        terms = code_tokenize(query)
        lines = [f"search: {query}   ({len(ids)} matches):"]
        for rank, doc_id in enumerate(ids, start=1):
            u = self.ubyid.get(doc_id)
            if u is None:
                continue
            self.seen.add(doc_id)
            named = [s for s in self._secs(doc_id) if s != _INTRO] or list(self._secs(doc_id))
            sec_str = "·".join(named[:8]) + (",…" if len(named) > 8 else "")
            keys = list(_infobox(u))
            ib_str = "·".join(keys[:6]) + (",…" if len(keys) > 6 else "")
            title = u.title or u.qualname or doc_id
            line = f"  {rank}  {doc_id}  {title!r}  §[{sec_str}]  ib[{ib_str}]"
            snip = self._best_line(u, terms)
            if snip:
                line += f"  » {snip}"
            lines.append(line)
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("hybrid_search_snip", "hybrid_search", "search"):
                k = args.get("k")
                return self.search(args.get("query") or args.get("q") or self.query,
                                   int(k) if k else None)
            if name == "fetch":
                # DELEGATE the read verbatim, exactly like Bm25FetchWorkspace/DenseFetchWorkspace
                # — the only difference from those cells is the retrieval RANKING, not the read.
                return super().run("fetch", args)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# --- NEW, additive-only BM25+DENSE HYBRID "retrieve-and-read" baseline (research_hybrid_autoread)
# — see module docstring's HybridAutoRead paragraph. Subclasses `HybridVisit` (reuses its
# two-engine __init__ verbatim), overriding ONLY `tools`/`search`/`run` — `HybridVisit.search`/
# `HybridVisit.run` (research_hybrid) are never touched, exactly mirroring how `DenseAutoRead`
# subclasses `DenseVisit` and `Bm25AutoRead` subclasses `Bm25Visit`.

class HybridAutoRead(HybridVisit):
    """research_hybrid_autoread — the AutoRead analog of research_hybrid: SAME RRF-fused
    bm25+dense retrieval as `HybridVisit` (SAME `HYBRID_POOL`/`RRF_K`/two engines — see
    `HybridVisit`/`rrf_fuse`), but `search(query)` itself renders the FULL TEXT of every one of
    the top `AUTOREAD_TOPK` hits — there is NO visit/fetch tool at all in this condition; a
    search call IS the read. Each hit's text is capped at MAX_VISIT_TOKENS via the SAME
    `_cap_tokens` helper (same cap, same " …(truncated — this is the whole-doc cap)" marker)
    `Bm25AutoRead`/`DenseAutoRead`/`HybridVisit.visit` use — so a single doc's rendered length is
    identical to what a `research_hybrid` episode gets FROM visiting that same rank, only here
    every top-k hit is rendered, unconditionally, on every search call.

    Reuses the SAME `AUTOREAD_TOPK` env knob the other two autoread baselines use (not a third
    one — all three "retrieve-and-read" cells show the same-sized pool) and the SAME
    MAX_VISIT_TOKENS cap. The ONLY difference from `research_hybrid` is the render (full text vs
    a listing to visit) — exactly like `Bm25AutoRead`/`DenseAutoRead` over their own visit
    siblings. Uncoached (no skill), like `HybridVisit`."""

    # ONE tool only — no read/visit tool exists in this condition (search already returns full
    # text). Overrides HybridVisit's class-level `tools = ("hybrid_search", "visit_h")`.
    tools = ("hybrid_read_search",)

    def __init__(self, units: Sequence[CodeUnit], bm25_engine=None, dense_engine=None,
                 ubyid: Optional[dict] = None, corpus_key: Optional[str] = None,
                 pool: int = HYBRID_POOL):
        # SAME construction as HybridVisit (two engines, SAME pool depth) — `tools` is already
        # this class's own value via the class attribute above; HybridVisit.__init__ never
        # overrides `self.tools`, so no restore-after-super() is needed (mirrors DenseAutoRead's
        # own comment on DenseVisit.__init__).
        super().__init__(units, bm25_engine=bm25_engine, dense_engine=dense_engine,
                         ubyid=ubyid, corpus_key=corpus_key, pool=pool)

    def search(self, query: str, k: int = AUTOREAD_TOPK) -> str:
        query = (query or "").strip()
        if not query:
            return "empty query"
        bm25_ids = list(self.bm.search(query, k=self.pool))
        dense_ids = list(self.dense_engine.top_k_doc_ids(query, k=self.pool) or [])
        ids = rrf_fuse(bm25_ids, dense_ids, topk=k)
        if not ids:
            prior = ("  (previous results still available)" if self.last_hits else "")
            return f"search: {query}   (0 matches){prior}"
        self.last_hits = list(ids)
        blocks = [f"search: {query}   ({len(ids)} matches, full text below):"]
        for rank, i in enumerate(ids, start=1):
            u = self.ubyid.get(i)
            if u is None:
                continue
            self.seen.add(i)
            # SAME cap/marker as HybridVisit.visit (inherited from Bm25Visit) — a doc read via
            # autoread's search is truncated identically to the same doc read via
            # research_hybrid's visit_h.
            text = _cap_tokens(u.body or u.code or "", MAX_VISIT_TOKENS,
                               " …(truncated — this is the whole-doc cap)")
            blocks.append(f"  {rank}  {i}  {(u.title or u.qualname or '')!r}:\n{text}")
        return "\n\n".join(blocks)

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name in ("hybrid_read_search", "search"):
                # listing depth is AUTOREAD_TOPK (env knob) ONLY — the tool schema exposes just
                # `query`, so a hallucinated `k` arg must not resize how many docs are read.
                return self.search(args.get("query") or args.get("q") or "", AUTOREAD_TOPK)
            if name in ("visit", "visit_q", "visit_d", "visit_v", "visit_h", "visit_bv",
                       "visit_bqld", "fetch"):
                return ("ERROR: no visit tool in this condition — search already returns full "
                        "documents.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."


# --- NEW, additive-only DENSE-ONLY BQL siblings (research_bql_donly_visit / research_bql_donly_
# snip — see module docstring's BqlDonlyVisitWorkspace/DocSearchFetchDonlySnip paragraph and
# agent_search/retrievers/structural/bql/dense_fuse.py's "dense-ONLY ordering" section). Both are
# thin DISPATCH subclasses: they add their OWN tool names, translate to the mirror cell's tool
# name, and delegate to the mirror cell's `run()` — `BqlVisitWorkspace.run`/`DocSearchFetch.run`
# (used by research_bql_visit/research_bql_dense_visit and research_snip/research_bql_dense_snip/
# research_bql_dense_fetch respectively) are NEVER touched. The only real behavioral difference —
# dense-only vs RRF candidate ordering — lives entirely in which EXECUTOR class
# (`StructuralExecutor` vs `DenseOnlyStructuralExecutor`) `agent/retriever.py` attaches at
# construction time; neither workspace class here knows or cares which one it was given.

class BqlDonlyVisitWorkspace(BqlVisitWorkspace):
    """research_bql_donly_visit — BYTE-FOR-BYTE research_bql_dense_visit (`BqlVisitWorkspace`
    with its own `tool_names`) except the executor passed in by `agent/retriever.py`'s
    'bqldonlyvisit' arm is a `DenseOnlyStructuralExecutor` (dense-ONLY candidate ordering)
    instead of a plain `StructuralExecutor` with a `DenseBelief` attached (RRF). `search_bqldo`/
    `visit_bqldo` are this condition's own tool names, translated to `search_bv`/`visit_bv`
    before delegating to the inherited `BqlVisitWorkspace.run` (which already accepts
    `search_bv`/`search_bqld`/`search`/`search_v2` and `visit_bv`/`visit_bqld`/`visit` — adding
    a translation step here is strictly additive, no existing branch of that `run()` changes)."""

    _NAME_MAP = {"search_bqldo": "search_bv", "visit_bqldo": "visit_bv"}

    def run(self, name: str, args: dict) -> str:
        return super().run(self._NAME_MAP.get(name, name), args)


class DocSearchFetchDonlySnip(DocSearchFetch):
    """research_bql_donly_snip — BYTE-FOR-BYTE research_bql_dense_snip (`DocSearchFetch(
    snippets=True)`) except the executor passed in by `agent/retriever.py`'s 'bqldonlysnip' arm
    is a `DenseOnlyStructuralExecutor` (dense-ONLY ordering WITHIN each coverage tier — the tier
    STRUCTURE itself, and the exact-hit filter, are untouched) instead of a plain
    `StructuralExecutor` with a `DenseBelief` attached (RRF-within-tier). `search_bqldos`/
    `fetch_bqldos` are this condition's own tool names, translated to `search_bqlds`/
    `fetch_bqlds` before delegating to the inherited `DocSearchFetch.run` (which already accepts
    `search_bqlds`/`fetch_bqlds` as research_bql_dense_snip's own aliases — adding a translation
    step here is strictly additive, no existing branch of that `run()` changes)."""

    tools = ("search_bqldos", "fetch_bqldos")

    _NAME_MAP = {"search_bqldos": "search_bqlds", "fetch_bqldos": "fetch_bqlds"}

    def run(self, name: str, args: dict) -> str:
        return super().run(self._NAME_MAP.get(name, name), args)
