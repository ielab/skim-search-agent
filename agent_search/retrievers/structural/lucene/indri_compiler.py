"""Compiles the EXISTING Indri QL AST (`agent_search.retrievers.structural.indri.parser`
-- reused, not re-parsed: this module never defines its own grammar) into Lucene
`Query` objects, runnable via `IndexSearcher.search(query, k)` under a global
`LMDirichletSimilarity(mu)` (the searcher's similarity; set by `engine.py`).

## Per-operator compile mapping (op -> Lucene construct; "approx" = documented deviation)

| Indri AST node          | Lucene compile                                             | fidelity |
|--------------------------|-------------------------------------------------------------|----------|
| `Term` (1 subtoken)      | `TermQuery` on the stemmed field                             | LMD formula is IDENTICAL (see below); collection stats bookkeeping differs slightly from the Python reference -> graded-ranking equivalent, not byte-identical |
| `Term` (multi-subtoken)  | `PhraseQuery` (slop 0) on the stemmed field                  | exact adjacency, same as reference's `_seq_in` |
| `Wildcard`                | `PrefixQuery` on the `_exact` (unstemmed) field (or `SpanMultiTermQueryWrapper(PrefixQuery)` inside a window) | exact prefix match (reference matches raw unstemmed token prefixes) |
| `Window` (`#odN`/`#uwN`)  | `SpanNearQuery` (ordered=True for od, False for uw) on the `_exact` field, `slop=n-1` (or a large slop for unlimited) | exact positional semantics on THIS field's tokenization (SimpleAnalyzer, see schema.py deviations) |
| `Syn` (single-token members, one field) | `SynonymQuery` (native Lucene combined-tf disjunction -- the closest built-in match to "occurrences of a OR b" belief math) | close/native fit |
| `Syn` (general)          | `SpanOrQuery` (if every member is span-compilable) else `BooleanQuery` SHOULD of each member's own scored query | **approx**: loses the reference's tf-SUM-then-smooth math; independently-scored-then-summed instead |
| `WSyn`                   | `SynonymQuery.Builder.addTerm(term, boost)` for single-token weighted members; else boosted `BooleanQuery` SHOULD | **approx**, same reasoning as `Syn` |
| `Combine`                 | `BooleanQuery` SHOULD of each child's scored query, EXCEPT `DateBefore`/`DateAfter`/`DateBetween` children, which become an `Occur.FILTER` clause instead (see `_boolean_should`'s `filter_dates`) | **approx** for the scored children: Lucene SUMS per-clause scores; reference takes the MEAN of log-beliefs. Monotonic-similar for ranking within one query (fixed child count), not numerically identical -- validated via rank correlation, not exact scores. The date-child FILTER carve-out IS exact: `mean()` collapses to `NEG_INF` (excluded) the instant ANY child is `NEG_INF`, so a date operator anywhere inside `#combine` is a hard per-doc gate, not a graded/optional clause |
| `Weight`                  | `BooleanQuery` SHOULD, each clause `BoostQuery(clause, normalized_weight)`; same date-child FILTER carve-out as `Combine` (a nonzero-weight date child still hard-gates -- weighted mean has the same `NEG_INF`-propagates property; a zero-weight date child is dropped entirely, matching the reference's "zero-weight child never contributes") | same SUM-vs-weighted-MEAN approx as `Combine` for scored children; date-child FILTER is exact |
| `Or`                      | `BooleanQuery` SHOULD, `minimumShouldMatch=1`                  | **approx**: true math is `log(1-prod(1-p))`; Lucene sums scores instead |
| `Not`                     | folded into the ENCLOSING boolean composition as an `Occur.MUST_NOT` clause (or, standalone, `MatchAllDocsQuery MUST` + `MUST_NOT`) | **approx**: hard exclusion, not the reference's graded "absence boosts score" probability |
| `Max`                     | `DisjunctionMaxQuery` (tie=0.0)                                | close/native fit -- both take "the single best child" |
| `Band`                    | per child: `Occur.FILTER` clause from the EXACT-field "matches" test (guarantees match-SET parity) + `Occur.SHOULD` clause from the scored query (for ranking) | match set exact; ranking approx (same SUM-vs-MEAN note) |
| `FieldExpr` (`.field`)    | narrows to that field's stemmed/exact pair (`schema.SCORED_FIELD_MAP`); multiple field names -> `BooleanQuery` SHOULD union across each field's compiled query | **approx** for multi-field union: independently-scored-then-summed rather than the reference's aggregated single Dirichlet over combined counts |
| `FieldExpr` (`.(context)`)| collapsed into the SAME field-restriction path as `.field` (context used as if it were the counting field too) | **approx**: Lucene ties a field's term stats to that SAME field always, so the reference's counting-vs-smoothing-field SPLIT has no clean Lucene equivalent |
| `.author` / `.date`       | `ConstantScoreQuery` around an exact `TermQuery` on the StringField (SHOULD clause, constant contribution) -- NOT Dirichlet-scored | **approx**: documented in `schema.py`; author/date are filter-only fields in this backend |
| `FilReq(A, Q)`             | `BooleanQuery`: `A` compiled EXACT as `Occur.FILTER` (no score contribution) + `Q` compiled SCORED as `Occur.MUST` | exact filter-set parity; Q's ranking has the same approx notes as its own node kind |
| `FilRej(A, Q)`             | same, `A` as `Occur.MUST_NOT`                                  | exact |
| `#date:before/after/between` | `TermRangeQuery.newStringRange` on the `date` StringField (ISO strings sort chronologically). ALSO now accepted as `compile_exact`'s node (a `#filreq`/`#filrej`/`Band` first-argument use) -- the reference's `_matches()` has no first-argument restriction of its own for date operators, it just falls through to `belief != NEG_INF`, which for a date operator IS the same hard gate | exact |

## Fundamental Lucene limitation: no "smoothed match" for absent terms

The Python reference's Dirichlet formula gives EVERY document in a query's
candidate pool a finite belief for EVERY leaf, even leaves whose terms don't
occur in that document at all (`P(t|C)` collection smoothing never yields a hard
zero -- see `indri/model.py`'s Deviations, "never a hard zero"). Lucene's
`TermQuery`/`PhraseQuery`/`SpanNearQuery` have no equivalent: they can ONLY
match documents that literally contain the queried term(s)/span, by design (this
is what makes Lucene fast at corpus scale -- scoring the whole corpus for every
query is the thing an inverted index exists to avoid). Two concrete
consequences, both intentional, both left as Lucene's native (narrower, faster)
behavior rather than worked around:
  - A bare (unwrapped) `#odN`/`#uwN`/`.field` query is graded in the reference
    (its candidate pool is every doc containing the window's/field's INDIVIDUAL
    terms, each scored -- including docs with zero actual positional/field
    match). Lucene's compiled `SpanNearQuery`/field `TermQuery` only match docs
    with a REAL positional/field occurrence -- a strictly narrower, exact-match
    result. Validated as "known-answer" tests on hand-built docs (span/field
    ops behave per spec), NOT as match-set-equality against the reference's
    graded pool (see `tests/test_lucene_structured.py`'s module docstring note).
  - `#filreq(A Q)`/`#filrej(A Q)` avoid this by construction: `Q` compiles to an
    `Occur.SHOULD` (optional/scoring-only) clause, not `Occur.MUST` -- since `A`
    (`Occur.FILTER`) already anchors the match set, a doc that passes the filter
    is NEVER excluded for lacking `Q`'s terms, matching the reference's "A
    filters, Q ranks every A-matching doc" semantics exactly (this required
    fixing an initial version that used `Occur.MUST` for `Q`, which wrongly
    forced `Q`'s terms to be literally present -- see git history for the
    match-set regression that caught it).

`compile_score(expr)` is the top-level entry (what `engine.py` runs for ranking).
`compile_exact(expr, fields)` is the boolean "matches" compiler used for `Band`/
`FilReq`/`FilRej` filter clauses and `Not` exclusions -- restricted to leaf-ish nodes
(`Term`/`Wildcard`/`Window`/`Syn`/`WSyn`) + `FieldExpr`/`Not` wrapping them, mirroring
the Indri QL spec's own restriction that `#filreq`/`#filrej`'s first argument "must be
a term/proximity expression" (`agent_search/prompts/skills/indri_doc.md`). A `Combine`/`Weight`/`Or`/
`Max` used as a filter argument (rare, out of spec) raises `LuceneCompileError` rather
than silently approximating -- unlike the Python reference's `_matches`, which defines
"matches" for ANY compound node as "belief != -inf" (a graded fallback with no Lucene
equivalent worth approximating quietly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from agent_search.retrievers.structural.indri.parser import (
    Band, Combine, DateAfter, DateBefore, DateBetween, FieldExpr, FilReq, FilRej,
    Max, Not, Or, Syn, Term, Weight, Wildcard, Window, WSyn, date_bounds,
)
from agent_search.retrievers.structural.lucene import jni_utils as J
from agent_search.retrievers.structural.lucene.schema import (
    DEFAULT_EXACT_FIELD, DEFAULT_SCORED_FIELD, F_AUTHOR, F_DATE, SCORED_FIELD_MAP,
)

_LEAFISH = (Term, Wildcard, Window, Syn, WSyn)


@dataclass
class LuceneCompileError(Exception):
    op: str
    message: str

    def __str__(self) -> str:                # pragma: no cover - cosmetic
        return f"{self.op}: {self.message}"


# --- field resolution ---------------------------------------------------------

def _resolve_fields(fields: Optional[tuple]) -> tuple:
    """Indri field-name tuple (may be None = default `body`) -> tuple of Lucene
    (scored_field, exact_field, is_filter_only) triples. Unknown / author / date
    field names resolve to `(None, None, True)` -- filter-only, no LMD scoring (see
    `schema.py` Deviations)."""
    if not fields:
        return ((DEFAULT_SCORED_FIELD, DEFAULT_EXACT_FIELD, False),)
    out = []
    for f in fields:
        pair = SCORED_FIELD_MAP.get(f.lower())
        if pair is not None:
            out.append((pair[0], pair[1], False))
        else:
            out.append((None, None, True))          # author/date/unknown -> filter-only
    return tuple(out)


# --- query builders (thin wrappers over jni_utils classes) ---------------------

def _term_query(field: str, token: str):
    Term = J.J("Term")
    TermQuery = J.J("TermQuery")
    return TermQuery(Term(field, token))


def _phrase_query(field: str, tokens: list):
    if len(tokens) == 1:
        return _term_query(field, tokens[0])
    PhraseQueryBuilder = J.J("PhraseQueryBuilder")
    b = PhraseQueryBuilder()
    b.setSlop(0)
    Term = J.J("Term")
    for t in tokens:
        b.add(Term(field, t))
    return b.build()


def _prefix_query(field: str, stem: str):
    Term = J.J("Term")
    PrefixQuery = J.J("PrefixQuery")
    return PrefixQuery(Term(field, stem.lower()))


def _span_term(field: str, token: str):
    Term = J.J("Term")
    SpanTermQuery = J.J("SpanTermQuery")
    return SpanTermQuery(Term(field, token))


def _span_phrase(field: str, tokens: list):
    """A contiguous multi-token match as a SPAN (ordered, slop 0)."""
    if len(tokens) == 1:
        return _span_term(field, tokens[0])
    b = J.J("SpanNearQueryBuilder")(field, True)
    for t in tokens:
        b.addClause(_span_term(field, t))
    b.setSlop(0)
    return b.build()


def _bool_builder():
    return J.J("BooleanQueryBuilder")()


def _occur(name: str):
    Occur = J.J("Occur")
    return getattr(Occur, name)


_LARGE_SLOP = 1_000_000  # "unlimited" window (#od/#uw with no N) -- whole-field span


# --- SCORED compile (the graded ranking query) ----------------------------------

def compile_score(node, fields: Optional[tuple] = None):
    """Compile `node` into a Lucene `Query` for GRADED ranking under whatever
    similarity the caller's `IndexSearcher` has set (see `engine.py` -- expected to
    be `LMDirichletSimilarity(mu)`). `fields` is the current Indri field-restriction
    scope (None = default `body`)."""
    if isinstance(node, FieldExpr):
        f2 = node.fields if node.fields else ((node.context,) if node.context else fields)
        return compile_score(node.child, f2)

    if isinstance(node, Term):
        return _term_scored(node, fields)
    if isinstance(node, Wildcard):
        return _wildcard_scored(node, fields)
    if isinstance(node, Window):
        return _window_scored(node, fields)
    if isinstance(node, Syn):
        return _syn_scored(node, fields)
    if isinstance(node, WSyn):
        return _wsyn_scored(node, fields)

    if isinstance(node, Combine):
        return _boolean_should(node.children, fields, weights=None, filter_dates=True)
    if isinstance(node, Weight):
        children = [c for _, c in node.pairs]
        wsum = sum(w for w, _ in node.pairs) or 1.0
        weights = [w / wsum for w, _ in node.pairs]
        return _boolean_should(children, fields, weights=weights, filter_dates=True)
    if isinstance(node, Or):
        # NOTE: date children under #or do NOT get `filter_dates` treatment.
        # `model.py`'s Or belief is a probabilistic union (`log(1-prod(1-p))`):
        # a FAILING date child contributes p=0 (a no-op factor, same as an
        # absent term) rather than excluding the doc, and a PASSING date child
        # forces the whole Or to certain-match (p=1) rather than merely
        # boosting it. Neither is a hard AND-style filter, so date operators
        # here stay on the existing scored-SHOULD path (already documented as
        # an approximation in the Deviations table above).
        return _boolean_should(node.children, fields, weights=None, min_should=1)
    if isinstance(node, Max):
        DisjunctionMaxQuery = J.J("DisjunctionMaxQuery")
        ArrayList = _java_list(compile_score(c, fields) for c in node.children)
        return DisjunctionMaxQuery(ArrayList, 0.0)
    if isinstance(node, Band):
        return _band_scored(node, fields)
    if isinstance(node, Not):
        # standalone #not(x): "match everything except x" -- a hard-exclusion
        # approximation of the reference's graded 1-P(x) belief (see module
        # Deviations table).
        b = _bool_builder()
        match_all = J.J("MatchAllDocsQuery")()
        b.add(match_all, _occur("MUST"))
        q = compile_exact(node.child, fields)
        b.add(q, _occur("MUST_NOT"))
        return b.build()
    if isinstance(node, FilReq):
        # `Q` is Occur.SHOULD (scoring only), NOT MUST: the Python reference ranks
        # EVERY doc that passes filter `A` by Q's Dirichlet-smoothed belief, which
        # is ALWAYS finite (never -inf) even when Q's terms don't literally occur
        # in a doc (collection-frequency smoothing -- see indri/model.py's
        # Deviations: "graceful ... never a hard zero"). A MUST clause would
        # instead REQUIRE Q's terms to literally appear, wrongly excluding an
        # A-matching doc with zero Q-term occurrences -- Lucene has no native
        # "smoothed match everything" query, so `Occur.FILTER(A)` alone anchors
        # the match set (satisfies "at least one positive clause") and SHOULD(Q)
        # becomes pure, optional scoring, which is the closest reproducible
        # approximation.
        b = _bool_builder()
        qa = compile_exact(node.a, fields)
        b.add(qa, _occur("FILTER"))
        qq = compile_score(node.q, fields)
        b.add(qq, _occur("SHOULD"))
        return b.build()
    if isinstance(node, FilRej):
        # Same SHOULD-not-MUST reasoning as FilReq above. `Occur.MUST_NOT` alone
        # is not a "positive" clause in Lucene's matching rule, so a
        # `MatchAllDocsQuery` MUST clause anchors "everything not matching A",
        # matching the reference's FilRej semantics (rank every NON-A doc by Q).
        b = _bool_builder()
        match_all = J.J("MatchAllDocsQuery")()
        b.add(match_all, _occur("MUST"))
        qa = compile_exact(node.a, fields)
        b.add(qa, _occur("MUST_NOT"))
        qq = compile_score(node.q, fields)
        b.add(qq, _occur("SHOULD"))
        return b.build()
    if isinstance(node, (DateBefore, DateAfter, DateBetween)):
        return _date_query(node)
    raise LuceneCompileError(op=type(node).__name__, message="no compile rule for this node type")


_DATE_NODES = (DateBefore, DateAfter, DateBetween)


def _date_query(node):
    """Compile a date-operator node into its exact `TermRangeQuery`. Used both
    as `compile_score`'s own return value (a bare/top-level `#date:...`) and as
    a FILTER clause when the operator is embedded inside `#combine`/`#weight`
    (see `_boolean_should`) or `#filreq`/`#filrej` (see `compile_exact`) --
    the query itself is identical either way; only WHERE it gets attached in
    the enclosing `BooleanQuery` (SHOULD vs FILTER) differs."""
    if isinstance(node, DateBefore):
        return _date_range_query(None, date_bounds(node.date)[0], hi_exclusive=True)
    if isinstance(node, DateAfter):
        return _date_range_query(date_bounds(node.date)[1], None, lo_exclusive=True)
    if isinstance(node, DateBetween):
        lo = date_bounds(node.lo)[0] if node.lo else None
        hi = date_bounds(node.hi)[1] if node.hi else None
        return _date_range_query(lo, hi)
    raise LuceneCompileError(op=type(node).__name__, message="not a date operator")


def _java_list(items):
    """Python iterable -> `java.util.ArrayList` (needed for `DisjunctionMaxQuery`'s
    `Collection<Query>` constructor)."""
    ArrayList = J.J("ArrayList")
    lst = ArrayList()
    for it in items:
        lst.add(it)
    return lst


def _boolean_should(children: Sequence, fields, weights: Optional[list], min_should: int = 0,
                     filter_dates: bool = False):
    """`filter_dates=True` (Combine/Weight only -- see call sites) folds any
    `DateBefore`/`DateAfter`/`DateBetween` child into an `Occur.FILTER` clause
    on THIS builder instead of a scored `Occur.SHOULD`, matching the reference's
    belief math for those two nodes: `Combine`/`Weight` take a (weighted) MEAN
    of children beliefs, and Python `float` arithmetic makes ANY `-inf` child
    (a failed date gate) propagate to `-inf` for the whole node regardless of
    the other children's scores -- i.e. a date operator anywhere inside a
    `#combine`/`#weight` is a HARD per-document gate, not an optional/graded
    clause. `#or` is deliberately excluded (see its call site's comment):
    its probabilistic-union math does not collapse the same way, so it keeps
    the prior scored-SHOULD approximation for date children."""
    b = _bool_builder()
    if min_should:
        b.setMinimumNumberShouldMatch(min_should)
    BoostQuery = J.J("BoostQuery")
    for i, c in enumerate(children):
        # Not-children fold into MUST_NOT on the enclosing builder (see module
        # Deviations: a Not inside Combine/Weight/Or is hard-excluded, not
        # probabilistically down-weighted).
        if isinstance(c, Not):
            qn = compile_exact(c.child, fields)
            b.add(qn, _occur("MUST_NOT"))
            continue
        if filter_dates and isinstance(c, _DATE_NODES):
            if weights is not None and weights[i] == 0.0:
                # a zero-weight child never contributes ANYTHING in the
                # reference (model.py: `if wn == 0: continue` -- skipped
                # before even being matched-tested), so it must not gate the
                # query either.
                continue
            qd = _date_query(c)
            b.add(qd, _occur("FILTER"))
            continue
        q = compile_score(c, fields)
        if weights is not None and weights[i] != 1.0:
            q = BoostQuery(q, float(weights[i]))
        b.add(q, _occur("SHOULD"))
    return b.build()


def _band_scored(node: Band, fields):
    b = _bool_builder()
    for c in node.children:
        qf = compile_exact(c, fields)
        b.add(qf, _occur("FILTER"))
        qs = compile_score(c, fields)
        b.add(qs, _occur("SHOULD"))
    return b.build()


# --- leaf scoring ----------------------------------------------------------------

def _field_disjunction(build_one, resolved_fields):
    """Union a per-field compiled query across `resolved_fields` (>1 field name ->
    `BooleanQuery` SHOULD; 1 field -> that field's query directly, no wrapper)."""
    scored = [(sf, ef) for sf, ef, filt in resolved_fields if not filt]
    if not scored:
        # every field in scope is filter-only (e.g. `.author`/`.date`) -- no LMD
        # scoring possible; approximate with a constant-score exact match (module
        # Deviations table).
        ConstantScoreQuery = J.J("ConstantScoreQuery")
        qs = [ConstantScoreQuery(build_one(_filter_field_name(f), exact=True))
              for f in resolved_fields]
        return qs[0] if len(qs) == 1 else _should_of(qs)
    qs = [build_one(sf, exact=False) for sf, ef in scored]
    return qs[0] if len(qs) == 1 else _should_of(qs)


def _should_of(queries):
    b = _bool_builder()
    for q in queries:
        b.add(q, _occur("SHOULD"))
    return b.build()


def _filter_field_name(resolved) -> str:
    # resolved is (None, None, True) for a filter-only field; the ORIGINAL field
    # name isn't threaded through _resolve_fields today, so filter-only .author/
    # .date scoring falls back to the well-known StringField names directly. This
    # is safe because SCORED_FIELD_MAP's only misses in this backend's supported
    # field set ARE author/date (anything truly unknown has no index field at all
    # and correctly finds nothing).
    return F_AUTHOR  # author is the common case; date-as-a-scored-leaf is unusual
    # (a bare `foo.date` term leaf, as opposed to `#date:before(...)`, is exotic --
    # documented here rather than silently guessing between author/date).


def _term_scored(node: Term, fields):
    resolved = _resolve_fields(fields)

    def build_one(field, exact):
        toks = J.exact_tokens(node.text, field) if exact else J.stemmed_tokens(node.text, field)
        if not toks:
            return J.J("MatchNoDocsQuery")()
        return _phrase_query(field, toks)         # handles len(toks) == 1 internally

    return _field_disjunction(build_one, resolved)


def _wildcard_scored(node: Wildcard, fields):
    # wildcards always use the EXACT field for a faithful literal prefix (module
    # Deviations table) -- resolve directly rather than going through
    # `_field_disjunction`'s stemmed-field default.
    resolved = _resolve_fields(fields)
    scored = [(sf, ef) for sf, ef, filt in resolved if not filt]
    if not scored:
        return J.J("MatchNoDocsQuery")()
    qs = [_prefix_query(ef, node.stem) for sf, ef in scored]
    return qs[0] if len(qs) == 1 else _should_of(qs)


def _window_scored(node: Window, fields):
    resolved = _resolve_fields(fields)
    scored = [(sf, ef) for sf, ef, filt in resolved if not filt]
    if not scored:
        return J.J("MatchNoDocsQuery")()
    qs = [_window_span(node, ef) for sf, ef in scored]
    return qs[0] if len(qs) == 1 else _should_of(qs)


def _span_of(node, field: str):
    """Compile a leaf-ish node into a Lucene SPAN query on `field` (the EXACT
    field), for use as a Window/Syn member. Raises if the member isn't
    span-compilable (e.g. a compound belief node, out of QL spec for this
    position)."""
    if isinstance(node, FieldExpr):
        # a nested field restriction inside a window is unusual; honor it if it
        # names the SAME field family, else fall through to the enclosing field.
        f2 = node.fields
        if f2:
            resolved = _resolve_fields(f2)
            scored = [ef for sf, ef, filt in resolved if not filt]
            if scored:
                field = scored[0]
        return _span_of(node.child, field)
    if isinstance(node, Term):
        toks = J.exact_tokens(node.text, field)
        if not toks:
            raise LuceneCompileError(op="Term", message=f"empty term {node.text!r} in window")
        return _span_phrase(field, toks)
    if isinstance(node, Wildcard):
        SpanMultiTermQueryWrapper = J.J("SpanMultiTermQueryWrapper")
        prefix_q = _prefix_query(field, node.stem)
        return SpanMultiTermQueryWrapper(prefix_q)
    if isinstance(node, Window):
        return _window_span(node, field)
    if isinstance(node, Syn):
        SpanOrQuery = J.J("SpanOrQuery")
        # SpanOrQuery's varargs constructor wants an array, not a List; pyjnius
        # accepts a Python list for a Java varargs `SpanQuery...` parameter.
        member_spans = [_span_of(c, field) for c in node.children]
        return SpanOrQuery(member_spans)
    if isinstance(node, WSyn):
        SpanOrQuery = J.J("SpanOrQuery")
        member_spans = [_span_of(c, field) for _, c in node.pairs]
        return SpanOrQuery(member_spans)
    raise LuceneCompileError(op=type(node).__name__, message="not span-compilable")


def _window_span(node: Window, field: str):
    ordered = node.kind == "od"
    slop = (node.n - 1) if node.n is not None else _LARGE_SLOP
    b = J.J("SpanNearQueryBuilder")(field, ordered)
    for c in node.children:
        span_q = _span_of(c, field)
        b.addClause(span_q)
    b.setSlop(max(slop, 0))
    return b.build()


def _syn_scored(node: Syn, fields):
    resolved = _resolve_fields(fields)
    scored = [(sf, ef) for sf, ef, filt in resolved if not filt]
    if not scored:
        return J.J("MatchNoDocsQuery")()

    def one_field(sf, ef):
        # Tier 1: every member a single stemmed token -> native SynonymQuery
        # (combined-tf disjunction, the closest built-in fit).
        single_toks = []
        ok = True
        for c in node.children:
            if isinstance(c, Term):
                toks = J.stemmed_tokens(c.text, sf)
                if len(toks) == 1:
                    single_toks.append(toks[0])
                    continue
            ok = False
            break
        if ok and single_toks:
            SynonymQueryBuilder = J.J("SynonymQueryBuilder")(sf)
            JTerm = J.J("Term")          # NOT named `Term` -- would shadow the AST
            for t in single_toks:        # class of that name used by `isinstance(c, Term)` above
                SynonymQueryBuilder.addTerm(JTerm(sf, t))
            return SynonymQueryBuilder.build()
        # Tier 2: every member span-compilable -> SpanOrQuery on the exact field.
        try:
            SpanOrQuery = J.J("SpanOrQuery")
            member_spans = [_span_of(c, ef) for c in node.children]
            return SpanOrQuery(member_spans)
        except LuceneCompileError:
            pass
        # Tier 3: independent scored disjunction (approx, see module Deviations).
        member_scores = [compile_score(c, fields) for c in node.children]
        return _should_of(member_scores)

    qs = [one_field(sf, ef) for sf, ef in scored]
    return qs[0] if len(qs) == 1 else _should_of(qs)


def _wsyn_scored(node: WSyn, fields):
    resolved = _resolve_fields(fields)
    scored = [(sf, ef) for sf, ef, filt in resolved if not filt]
    if not scored:
        return J.J("MatchNoDocsQuery")()

    def one_field(sf, ef):
        pairs = []
        ok = True
        for w, c in node.pairs:
            if isinstance(c, Term):
                toks = J.stemmed_tokens(c.text, sf)
                if len(toks) == 1:
                    pairs.append((w, toks[0]))
                    continue
            ok = False
            break
        if ok and pairs:
            SynonymQueryBuilder = J.J("SynonymQueryBuilder")(sf)
            JTerm = J.J("Term")          # NOT named `Term` -- see `_syn_scored`'s note
            for w, t in pairs:
                SynonymQueryBuilder.addTerm(JTerm(sf, t), float(w))
            return SynonymQueryBuilder.build()
        BoostQuery = J.J("BoostQuery")
        qs = [BoostQuery(compile_score(c, fields), float(w)) for w, c in node.pairs]
        return _should_of(qs)

    qs = [one_field(sf, ef) for sf, ef in scored]
    return qs[0] if len(qs) == 1 else _should_of(qs)


# --- EXACT compile (boolean "matches" test: Band/FilReq/FilRej/Not) -------------

def compile_exact(node, fields: Optional[tuple] = None):
    """Compile a leaf-ish node (`Term`/`Wildcard`/`Window`/`Syn`/`WSyn`), optionally
    wrapped in `FieldExpr`/`Not`, into a Lucene `Query` usable as a boolean
    FILTER/MUST_NOT clause -- no scoring, matches the Python reference's exact
    "matches" test on the `_exact` (unstemmed) field. See module docstring's scope
    note: compound belief nodes (`Combine`/`Weight`/`Or`/`Max`) are OUT of scope
    here (the Indri QL spec restricts `#filreq`/`#filrej`'s first argument to a
    term/proximity expression) and raise `LuceneCompileError`.

    `DateBefore`/`DateAfter`/`DateBetween` ARE accepted here (despite not being
    `_LEAFISH`): the reference's `_matches()` (model.py) has no restriction
    check at all -- for any node it doesn't special-case (which includes the
    date operators), it falls through to `belief != NEG_INF`, and the date
    operators' belief is already a hard 0.0-or-NEG_INF gate (see `_belief`'s
    `DateBefore`/`DateAfter`/`DateBetween` branches). So `#filreq(#date:...(...)
    Q)` / `#filrej(#date:...(...) Q)` are valid in the reference and must be
    ACCEPTED here too, not rejected."""
    if isinstance(node, FieldExpr):
        f2 = node.fields if node.fields else ((node.context,) if node.context else fields)
        return compile_exact(node.child, f2)
    if isinstance(node, Not):
        b = _bool_builder()
        match_all = J.J("MatchAllDocsQuery")()
        b.add(match_all, _occur("MUST"))
        q = compile_exact(node.child, fields)
        b.add(q, _occur("MUST_NOT"))
        return b.build()
    if isinstance(node, Band):
        b = _bool_builder()
        for c in node.children:
            q = compile_exact(c, fields)
            b.add(q, _occur("MUST"))
        return b.build()
    if isinstance(node, _DATE_NODES):
        return _date_query(node)
    if not isinstance(node, _LEAFISH):
        raise LuceneCompileError(
            op=type(node).__name__,
            message="compile_exact only supports term/proximity expressions "
                    "(Term/Wildcard/Window/Syn/WSyn) per the Indri QL spec's "
                    "#filreq/#filrej first-argument restriction")

    resolved = _resolve_fields(fields)
    # filter-only fields (author/date) have no `_exact` sibling -- use the
    # StringField's own exact-match TermQuery instead (still an exact boolean test).

    def query_for_one(sf, ef, filt):
        if filt:
            field = _filter_field_name((sf, ef, filt))
            return _exact_boolean_leaf_stringfield(node, field)
        return _exact_boolean_leaf(node, ef)

    qs = [query_for_one(sf, ef, filt) for sf, ef, filt in resolved]
    return qs[0] if len(qs) == 1 else _should_of(qs)


def _exact_boolean_leaf(node, field: str):
    if isinstance(node, Term):
        toks = J.exact_tokens(node.text, field)
        if not toks:
            return J.J("MatchNoDocsQuery")()
        return _phrase_query(field, toks)
    if isinstance(node, Wildcard):
        return _prefix_query(field, node.stem)
    if isinstance(node, Window):
        return _window_span(node, field)
    if isinstance(node, Syn):
        return _should_of([_exact_boolean_leaf(c, field) for c in node.children])
    if isinstance(node, WSyn):
        return _should_of([_exact_boolean_leaf(c, field) for _, c in node.pairs])
    raise LuceneCompileError(op=type(node).__name__, message="not a boolean-matchable leaf")


def _exact_boolean_leaf_stringfield(node, field: str):
    """`Term`-only exact match against a StringField (author/date scope) -- the
    whole field value must equal the term text (StringField has no sub-token
    concept, so windows/wildcards degrade to a whole-value contains-check
    approximation via a PrefixQuery, documented as low-fidelity/rare)."""
    if isinstance(node, Term):
        return _term_query(field, node.text)
    if isinstance(node, Wildcard):
        return _prefix_query(field, node.stem)
    raise LuceneCompileError(
        op=type(node).__name__,
        message=f"proximity/synonym operators are not supported scoped to the "
                f"StringField {field!r} (author/date have no token positions)")


# --- date range -------------------------------------------------------------------

def _date_range_query(lo: Optional[str], hi: Optional[str],
                       lo_exclusive: bool = False, hi_exclusive: bool = False):
    TermRangeQuery = J.J("TermRangeQuery")
    return TermRangeQuery.newStringRange(
        F_DATE, lo, hi, not lo_exclusive, not hi_exclusive)
