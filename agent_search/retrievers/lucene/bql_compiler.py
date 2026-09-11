"""Compiles the existing BQL AST (`agent_search.retrievers.bql.ast`,
parsed by `bql.parser.parse`, reused rather than re-parsed) into a Lucene `Query`:
`Occur.FILTER`/`Occur.MUST_NOT` clauses reproduce the reference's exact boolean
match set, plus `Occur.SHOULD` clauses give a BM25-style ranking signal, run under
`BM25Similarity(0.9, 0.4)` (matching `bm25_pyserini`'s defaults, see `engine.py`).

## Scope

This compiler covers the document-corpus regions this package's Lucene schema
actually indexes (`TITLE`/`BODY`/`SECTION`/`AUTHOR`/`DATE`/`DOC`) and `And`/`Or`/
`Not`/`Near`/`In` composition over `Term`/`Phrase`/`Prefix`/`Expand` leaves. It
does not implement the code-AST regions (`DEF`/`CALL`/`SIG`/`COMMENT`/`STRING`),
`Region.INFOBOX`, or `Region.FILE`: those need either a real AST-role index (code
regions) or a cross-document join (FILE, siblings of one path) that this
backend's one-Lucene-doc-per-corpus-doc schema (`schema.py`) doesn't carry. Each
raises a clear `LuceneCompileError` rather than silently returning wrong results.
This module has less test coverage than `indri_compiler.py`; treat the document-
region subset covered here as functional but more lightly validated, and the
code regions and coverage-ranking parity below as open extension points.

## Ranking delta vs the Python reference's `coverage_topk`

`StructuralExecutor.coverage_topk` (see `bql/executor.py`) ranks a 0-hit AND's
corpus by `(# children matched DESC, BM25(positive terms) DESC)`, a genuinely
different sort key (integer coverage first, tie-broken by score) from anything a
single Lucene `Query` scores in one pass. This compiler does not reproduce
`coverage_topk`: `compile_bql` always compiles the query's exact boolean
semantics (a hit means every constraint held, same as `run_with_count`), and its
ranking signal is the reference's ordinary hit-ranking path (BM25 over positive
leaf terms), not the diagnostic near-miss coverage ranking. Reproducing
`coverage_topk` here would need N separate boolean sub-queries (one per AND
child) plus a manual `(sum of matched, score)` combination client-side, which is
out of scope for this compiler.

## Field mapping (BQL `Region` -> Lucene field; see `schema.py`)

| Region                    | Exact/boolean field                  | notes |
|----------------------------|---------------------------------------|-------|
| (unscoped leaf) / `DOC`    | `body_exact` or `title_exact`          | mirrors the reference's unscoped/DOC match, which checks the unit's own text: title (`qualname`) and body (`code`) are separate fields on the Python side too (see `units.py`'s `units_from_documents`) |
| `BODY`                     | `body_exact`                           | |
| `TITLE`                    | `title_exact`                          | |
| `SECTION`                  | `section_exact`                        | |
| `AUTHOR`                   | `author_text` (tokenized)               | see `schema.py`'s `author_text`/`date_text` deviation note |
| `DATE` (plain token)       | `date_text` (tokenized)                 | |
| `DATE` (`__daterange__LO__HI` encoded term, from `surface.py`'s `date[RANGE]`) | `TermRangeQuery` on the ISO `date` StringField | exact |
| `INFOBOX`/`COMMENT`/`STRING`/`DEF`/`CALL`/`SIG`/`FILE` | -- | unsupported, `LuceneCompileError` |

Ranking (`_score_leaves`) always scores against `body`+`title` (stemmed, SHOULD-
unioned) regardless of any enclosing `IN(region, ...)`. This matches the Python
reference exactly: `_rank_leaves` (bql/executor.py) walks straight through `In`
nodes ignoring `.region`, because the underlying `_corpus_bm` is built once over
each unit's `qualname` (title) plus `code` (body, not section, see `units.py`'s
fairness comment) and reused for every query regardless of which region the
boolean match targeted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_search.retrievers.bql.ast import (
    And, Expand, Expr, In, Near, Not, Or, Phrase, Prefix, Region, Term,
)
from agent_search.retrievers.lucene import jni_utils as J
from agent_search.retrievers.lucene.schema import (
    F_AUTHOR_TEXT, F_BODY, F_BODY_EXACT, F_DATE, F_DATE_TEXT, F_SECTION_EXACT,
    F_TITLE, F_TITLE_EXACT,
)

_DATE_RANGE_TERM_RE = re.compile(
    r"^__daterange__(open|\d{4}-\d{2}-\d{2})__(open|\d{4}-\d{2}-\d{2})$")

# NEAR specs meaning "co-occur anywhere in scope" rather than a token window,
# the same set bql/executor.py's `_COOCCUR_SPECS` uses. In this one-Lucene-doc-
# per-unit schema (no separate section/paragraph/sentence index, see schema.py's
# field-schema decision), func/file/block/para/sent all collapse to "co-occur in
# the same Lucene document". This is a documented approximation: the reference's
# func/block/para/sent granularities are distinguishable in code corpora via
# AST/paragraph boundaries this backend doesn't index for prose documents.
_COOCCUR_SPECS = ("func", "file", "block", "para", "sent")

# Region -> the field used for EXACT boolean matching. `None` for the two regions
# handled specially (AUTHOR/DATE use their own tokenized `_text` fields, wired in
# `_field_for_region`; the unscoped/DOC default is a body+title union, also wired
# there rather than a single field).
_REGION_EXACT_FIELD = {
    Region.BODY: F_BODY_EXACT,
    Region.TITLE: F_TITLE_EXACT,
    Region.SECTION: F_SECTION_EXACT,
}
_UNSUPPORTED_REGIONS = {
    Region.INFOBOX: "no infobox index in this document-only Lucene schema",
    Region.COMMENT: "code-AST regions are not indexed by this backend (document corpora only)",
    Region.STRING: "code-AST regions are not indexed by this backend (document corpora only)",
    Region.DEF: "code-AST regions are not indexed by this backend (document corpora only)",
    Region.CALL: "code-AST regions are not indexed by this backend (document corpora only)",
    Region.SIG: "code-AST regions are not indexed by this backend (document corpora only)",
    Region.FILE: "file-scope (sibling units of one path) has no cross-document join in this backend",
}


@dataclass
class LuceneCompileError(Exception):
    op: str
    message: str

    def __str__(self) -> str:                # pragma: no cover - cosmetic
        return f"{self.op}: {self.message}"


def _bool_builder():
    return J.J("BooleanQueryBuilder")()


def _occur(name: str):
    return getattr(J.J("Occur"), name)


def _should_of(queries):
    if len(queries) == 1:
        return queries[0]
    b = _bool_builder()
    for q in queries:
        b.add(q, _occur("SHOULD"))
    b.setMinimumNumberShouldMatch(1)
    return b.build()


def _term_query(field: str, token: str):
    Term = J.J("Term")
    return J.J("TermQuery")(Term(field, token))


def _phrase_query(field: str, tokens: list):
    if not tokens:
        return J.J("MatchNoDocsQuery")()
    if len(tokens) == 1:
        return _term_query(field, tokens[0])
    b = J.J("PhraseQueryBuilder")()
    b.setSlop(0)
    Term = J.J("Term")
    for t in tokens:
        b.add(Term(field, t))
    return b.build()


def _prefix_query(field: str, stem: str):
    Term = J.J("Term")
    return J.J("PrefixQuery")(Term(field, stem.lower()))


# --- field resolution --------------------------------------------------------

def _default_exact_query(build_leaf) -> "Query":
    """Unscoped/`Region.DOC` leaf: OR across body_exact and title_exact (mirrors the
    Python reference's own unscoped/DOC match over `qualname`+`code`; see module
    docstring's field-mapping table)."""
    return _should_of([build_leaf(F_BODY_EXACT), build_leaf(F_TITLE_EXACT)])


def _field_for_region(region: Optional[Region]) -> Optional[str]:
    """A single exact field name for `region`, or `None` for the two regions
    (unscoped/DOC, AUTHOR/DATE) with special multi-field/tokenized handling done
    at the call site instead."""
    if region is None or region is Region.DOC:
        return None
    if region in _REGION_EXACT_FIELD:
        return _REGION_EXACT_FIELD[region]
    if region in (Region.AUTHOR, Region.DATE):
        return None
    if region in _UNSUPPORTED_REGIONS:
        raise LuceneCompileError(op=f"IN({region.value})", message=_UNSUPPORTED_REGIONS[region])
    raise LuceneCompileError(op=f"IN({region.value})", message="unrecognized region")


# --- EXACT compile (the boolean match set) ------------------------------------

def compile_exact(expr: Expr, region: Optional[Region] = None):
    """Compile `expr` to a Lucene `Query` whose hit set == the Python reference's
    `_eval(expr, ...)` == True set, scoped to `region` (None = unscoped/whole-unit,
    matching a bare leaf outside any `IN(...)`)."""
    if isinstance(expr, Term):
        return _leaf_query(region, lambda field: _term_leaf(field, expr.text))
    if isinstance(expr, Phrase):
        toks = []
        for t in expr.terms:
            toks.extend(J.exact_tokens(t.text, "body_exact"))
        return _leaf_query(region, lambda field: _phrase_query(field, toks))
    if isinstance(expr, Prefix):
        return _leaf_query(region, lambda field: _prefix_query(field, expr.stem))
    if isinstance(expr, Expand):
        # reference `_eval`: `t == stem or t.startswith(stem)`; a PrefixQuery
        # already covers both (a term trivially starts with itself).
        return _leaf_query(region, lambda field: _prefix_query(field, expr.term.text.lower()))
    if isinstance(expr, And):
        b = _bool_builder()
        for c in expr.children:
            if isinstance(c, Not):
                q = compile_exact(c.child, region)
                b.add(q, _occur("MUST_NOT"))
            else:
                q = compile_exact(c, region)
                b.add(q, _occur("MUST"))
        return b.build()
    if isinstance(expr, Or):
        qs = [compile_exact(c, region) for c in expr.children]
        return _should_of(qs)
    if isinstance(expr, Not):
        b = _bool_builder()
        match_all = J.J("MatchAllDocsQuery")()
        b.add(match_all, _occur("MUST"))
        q = compile_exact(expr.child, region)
        b.add(q, _occur("MUST_NOT"))
        return b.build()
    if isinstance(expr, Near):
        return _compile_near(expr, region)
    if isinstance(expr, In):
        rng = _daterange_query(expr)
        if rng is not None:
            return rng
        return compile_exact(expr.child, expr.region)
    raise LuceneCompileError(op=type(expr).__name__, message="no compile rule for this node type")


def _leaf_query(region: Optional[Region], build_leaf):
    """Dispatch a leaf-query builder (`build_leaf(field) -> Query`) to the right
    field(s) for `region`."""
    if region in (Region.AUTHOR, Region.DATE):
        field = F_AUTHOR_TEXT if region is Region.AUTHOR else F_DATE_TEXT
        return build_leaf(field)
    field = _field_for_region(region)
    if field is not None:
        return build_leaf(field)
    return _default_exact_query(build_leaf)


def _term_leaf(field: str, text: str):
    toks = J.exact_tokens(text, field)
    return _phrase_query(field, toks) if toks else J.J("MatchNoDocsQuery")()


def _daterange_query(expr: In):
    """`In(Region.DATE, Term("__daterange__LO__HI"))` (the surface-lowered
    `date[RANGE]` encoding, see bql/surface.py and executor.py's
    `_parse_daterange_term`) -> exact `TermRangeQuery` on the ISO `date`
    StringField. Returns None if `expr` isn't this encoding (an ordinary
    IN(date, term) falls through to the tokenized `date_text` path)."""
    if expr.region is not Region.DATE or not isinstance(expr.child, Term):
        return None
    m = _DATE_RANGE_TERM_RE.match(expr.child.text)
    if not m:
        return None
    lo = None if m.group(1) == "open" else m.group(1)
    hi = None if m.group(2) == "open" else m.group(2)
    TermRangeQuery = J.J("TermRangeQuery")
    return TermRangeQuery.newStringRange(F_DATE, lo, hi, True, True)


# --- NEAR: span-window if possible, else co-occurrence fallback -----------------

def _compile_near(expr: Near, region: Optional[Region]):
    if expr.spec in _COOCCUR_SPECS:
        b = _bool_builder()
        ql = compile_exact(expr.left, region)
        b.add(ql, _occur("MUST"))
        qr = compile_exact(expr.right, region)
        b.add(qr, _occur("MUST"))
        return b.build()
    field = _field_for_region(region) if region not in (Region.AUTHOR, Region.DATE, None) \
        else F_BODY_EXACT       # windows only make sense on a real text field
    win = _int_suffix(expr.spec)
    try:
        left_span = _span_of(expr.left, field)
        right_span = _span_of(expr.right, field)
    except LuceneCompileError:
        # Not span-compilable (And/Not/In/Near operand, or a `Near`-of-`Or`-
        # containing-a-Prefix): degrade to co-occurrence, the same "complex
        # operands fall back to co-occurrence" behavior bql/executor.py's
        # `_near_window` uses (module Deviations note above).
        b = _bool_builder()
        ql = compile_exact(expr.left, region)
        b.add(ql, _occur("MUST"))
        qr = compile_exact(expr.right, region)
        b.add(qr, _occur("MUST"))
        return b.build()
    b = J.J("SpanNearQueryBuilder")(field, False)     # BQL windows are unordered
    b.addClause(left_span)
    b.addClause(right_span)
    b.setSlop(max(win, 0))
    return b.build()


def _span_of(expr: Expr, field: str):
    if isinstance(expr, Term):
        toks = J.exact_tokens(expr.text, field)
        if not toks:
            raise LuceneCompileError(op="Term", message="empty term in NEAR window")
        return _span_phrase(field, toks)
    if isinstance(expr, Phrase):
        toks = []
        for t in expr.terms:
            toks.extend(J.exact_tokens(t.text, field))
        return _span_phrase(field, toks)
    if isinstance(expr, Prefix):
        SpanMultiTermQueryWrapper = J.J("SpanMultiTermQueryWrapper")
        prefix_q = _prefix_query(field, expr.stem)
        return SpanMultiTermQueryWrapper(prefix_q)
    if isinstance(expr, Expand):
        SpanMultiTermQueryWrapper = J.J("SpanMultiTermQueryWrapper")
        prefix_q = _prefix_query(field, expr.term.text.lower())
        return SpanMultiTermQueryWrapper(prefix_q)
    if isinstance(expr, Or):
        SpanOrQuery = J.J("SpanOrQuery")
        member_spans = [_span_of(c, field) for c in expr.children]
        return SpanOrQuery(member_spans)
    raise LuceneCompileError(op=type(expr).__name__, message="not span-compilable "
                              "(And/Not/In/Near operands fall back to co-occurrence)")


def _span_phrase(field: str, tokens: list):
    if len(tokens) == 1:
        Term = J.J("Term")
        return J.J("SpanTermQuery")(Term(field, tokens[0]))
    b = J.J("SpanNearQueryBuilder")(field, True)
    Term = J.J("Term")
    for t in tokens:
        span_term = J.J("SpanTermQuery")(Term(field, t))
        b.addClause(span_term)
    b.setSlop(0)
    return b.build()


def _int_suffix(spec: str, default: int = 5) -> int:
    digits = "".join(c for c in spec if c.isdigit())
    return int(digits) if digits else default


# --- SCORED compile: BM25-style ranking over positive leaf terms -----------------
# Matches `bql/executor.py`'s `_rank_leaves` exactly: walks straight through `In`
# (ignoring region), skips `Not` subtrees, and collects Term/Prefix/Expand/Phrase
# leaves, always scored against body+title (see module docstring).

def _score_leaves(expr: Expr, out: list) -> None:
    if isinstance(expr, Term):
        out.append(("term", expr.text))
    elif isinstance(expr, Prefix):
        out.append(("prefix", expr.stem))
    elif isinstance(expr, Expand):
        out.append(("term", expr.term.text))
    elif isinstance(expr, Phrase):
        for t in expr.terms:
            out.append(("term", t.text))
    elif isinstance(expr, Not):
        return
    elif isinstance(expr, And):
        for c in expr.children:
            _score_leaves(c, out)
    elif isinstance(expr, Or):
        for c in expr.children:
            _score_leaves(c, out)
    elif isinstance(expr, Near):
        _score_leaves(expr.left, out)
        _score_leaves(expr.right, out)
    elif isinstance(expr, In):
        _score_leaves(expr.child, out)


def compile_score(expr: Expr):
    leaves: list = []
    _score_leaves(expr, leaves)
    queries = []
    for kind, text in leaves:
        for sf in (F_BODY, F_TITLE):
            if kind == "prefix":
                queries.append(_prefix_query(sf, text))
            else:
                toks = J.stemmed_tokens(text, sf)
                if toks:
                    queries.append(_phrase_query(sf, toks))
    if not queries:
        return J.J("MatchAllDocsQuery")()
    b = _bool_builder()
    for q in queries:
        b.add(q, _occur("SHOULD"))
    return b.build()


# --- top level: boolean FILTER (exact match set) + scored SHOULD (ranking) ------

def compile_bql(expr: Expr):
    b = _bool_builder()
    exact_q = compile_exact(expr, None)
    b.add(exact_q, _occur("FILTER"))
    score_q = compile_score(expr)
    b.add(score_q, _occur("SHOULD"))
    return b.build()
