"""BQL field-tagged Boolean surface syntax -> BQL string (the existing executor AST).

This is the agent-facing query surface for both domains (documents and code): a field-tagged
Boolean form (`term[field]`, `AND`/`OR`/`NOT`, phrase, wildcard) that every LLM writes fluently
from pretraining, translated to the BQL the executor already runs. The parser, type checker and
executor, and all their tests, are reused untouched; only the surface changes. The structural
operators (AST region scope, NEAR, the structural fetch) are still reached via the field aliases
below.

Surface grammar:
  term[field]        -> IN(field, term)
  term[f1,f2]        -> OR(IN(f1, term), IN(f2, term))         (field combination)
  term[tiab]         -> OR(IN(title, term), IN(body, term))    (combo alias -> several fields)
  "a b"[field]       -> IN(field, "a b")                        (quoted phrase, field-scoped)
  term*  /  term*[f] -> PREFIX(term)  /  IN(f, PREFIX(term))    (wildcard)
  A AND B, A OR B    -> passthrough (BQL uses the same infix)
  A NOT B            -> A AND NOT(B)   (binary NOT is set-difference; BQL's NOT is unary-in-AND)
  A AND NOT B        -> A AND NOT(B)   (explicit AND already present, so it must not double up
                                         to the malformed "A AND AND NOT(B)"; see `_pass2`)
  -term / -"a b"     -> NOT(term)      (a `-`-prefixed atom, no space, negates just that one
                                         atom, Google-style; folds into the surrounding
                                         multi-word run's AND, e.g. `alpha -gamma` ->
                                         `AND(alpha, NOT(gamma))`, same "NOT is unary-in-AND"
                                         constraint as binary NOT above; see `_run_to_bql`. A
                                         negated atom with no positive sibling anywhere in its
                                         run/clause has no valid BQL form (the type checker
                                         requires an AND to have >=1 positive clause) and is
                                         left to raise a clear type-check error, not papered
                                         over silently.
  ( ... )            -> passthrough grouping
  bare term / "a b"  -> passthrough (matches anywhere)
  date[RANGE]        -> IN(date, __daterange__LO__HI)          (typed date range; see below)

Typed date ranges (BQL v2): `date[1980..1989]`, `date[2002]` (bare year = that year's whole
span), `date[<2023-12]`, `date[>=2019-06]`, `date[2019-06..2021]`. RANGE is `A..B`, `<X`,
`<=X`, `>X`, `>=X`, or a bare partial date; each bound is YYYY[-MM[-DD]] and a partial date
widens to its full span (1980 -> 1980-01-01..1980-12-31; `<2023-12` means before 2023-12-01;
`>=2019` means from 2019-01-01). This is lowered to `IN(date, term)` where the term's text is
a canonical `__daterange__LO__HI` encoding (LO/HI are normalized ISO bounds or the literal
`open`), which `executor.py` recognizes and evaluates against `unit.metadata['date']`. It
reuses the existing `In(Region.DATE, Term)` node rather than adding a new AST leaf, so the
parser, type checker, `_rank_leaves`, and every other BQL-string consumer need no changes and
the result round-trips through `bql_parse` untouched. `BQL_DATE_RANGE=0` disables the rewrite,
so `date[...]` falls through to plain passthrough-field lowering (exact-token behavior).

Field aliases are configurable per domain (the customizable block below). An unknown field is
passed through so the executor's type checker rejects it with a readable reason (useful agent
feedback).
"""
from __future__ import annotations

import calendar
import os
import re
from datetime import date, timedelta
from typing import Optional

# alias -> list of canonical BQL regions (a combo alias like `tiab` expands to several).
# CODE_FIELDS is the codefix task's surface; DOC_FIELDS is the research task's surface.
DOC_FIELDS = {
    "title": ["title"], "ti": ["title"],
    "body": ["body"], "text": ["body"], "ab": ["body"], "abstract": ["body"],
    "section": ["section"], "sec": ["section"], "heading": ["section"],
    "infobox": ["infobox"], "ib": ["infobox"], "fact": ["infobox"],
    "author": ["author"], "by": ["author"], "byline": ["author"],
    "date": ["date"],
    "tiab": ["title", "body"], "all": ["title", "body", "section", "infobox"],
}
CODE_FIELDS = {
    "def": ["def"], "definition": ["def"],
    "call": ["call"], "callsite": ["call"],
    "string": ["string"], "str": ["string"],
    "comment": ["comment"], "com": ["comment"],
    "sig": ["sig"], "signature": ["sig"],
    "file": ["file"],
}

_TOK = re.compile(r"""\s*(?:
      (?P<negquote>-"[^"]*")
    | (?P<quote>"[^"]*")
    | (?P<lbrack>\[) | (?P<rbrack>\])
    | (?P<lparen>\() | (?P<rparen>\))
    | (?P<comma>,)
    | (?P<negword>-\w[\w.+#-]*\*?)
    | (?P<word>\w[\w.+#-]*\*?)
    )""", re.VERBOSE | re.UNICODE)
_OPS = {"AND", "OR", "NOT"}
# Google-style `-`-prefixed negation ("alpha -gamma"): a `-` immediately (no space)
# before a word or quoted phrase -- distinguished from the `word` pattern above (which
# can never itself start with `-`, only contain one internally, e.g. "well-known"), so
# these two alternatives never compete for the same input. See `_pass1`'s run-collection
# loop and `_run_to_bql` for how a negated run element becomes `NOT(...)`.
_NEG_KINDS = ("negword", "negquote")


# --- typed date ranges: `date[RANGE]` -> IN(date, __daterange__LO__HI) ------
#
# Rewritten at the raw string level, before tokenization, into a single atom whose text
# already carries the canonical encoding, so the existing word tokenizer/run-collector
# (which knows nothing about date ranges) treats it exactly like any other bare word, and
# `_leaf` above unwraps it into the IN(date, ...) leaf. This is what lets `date[1980..1989]`
# combine naturally with neighboring bare terms (`treaty date[1980..1989]` ->
# `AND(treaty, IN(date, __daterange__...))`) via the same multi-word-run machinery that
# already ANDs `Command handle[def]`, with no separate case for "a leaf beside other leaves".
_DATE_RANGE_PREFIX = "__daterange__"
_DATE_FIELD_RE = re.compile(r"\bdate\[([^\[\]]*)\]", re.IGNORECASE)
_PARTIAL_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def _date_range_enabled() -> bool:
    return os.environ.get("BQL_DATE_RANGE", "1").strip().lower() not in ("0", "false", "no", "off")


def _partial_date(s: str) -> Optional[tuple]:
    """Parse a YYYY[-MM[-DD]] partial date -> (year, month|None, day|None), validating a
    real calendar date when day+month are both given. None on anything else."""
    m = _PARTIAL_DATE_RE.match(s)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2)) if m.group(2) else None
    d = int(m.group(3)) if m.group(3) else None
    if mo is not None and not (1 <= mo <= 12):
        return None
    if d is not None:
        try:
            date(y, mo, d)
        except ValueError:
            return None
    return y, mo, d


def _start_of(s: str) -> Optional[str]:
    """First day of the partial date's span: 1980 -> 1980-01-01, 2019-06 -> 2019-06-01."""
    p = _partial_date(s)
    if p is None:
        return None
    y, mo, d = p
    return f"{y:04d}-{mo or 1:02d}-{d or 1:02d}"


def _end_of(s: str) -> Optional[str]:
    """Last day of the partial date's span: 1980 -> 1980-12-31, 2019-06 -> 2019-06-30."""
    p = _partial_date(s)
    if p is None:
        return None
    y, mo, d = p
    if d is not None:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    if mo is not None:
        return f"{y:04d}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}"
    return f"{y:04d}-12-31"


def _shift_day(iso: str, delta: int) -> Optional[str]:
    try:
        y, mo, d = (int(x) for x in iso.split("-"))
        return (date(y, mo, d) + timedelta(days=delta)).isoformat()
    except Exception:  # noqa: BLE001, malformed bound -> caller treats as unparseable
        return None


def _parse_date_range_spec(spec: str) -> Optional[tuple]:
    """`date[...]`'s bracket content -> (lo, hi), either an ISO 'YYYY-MM-DD' string or None
    (open-ended). Returns None if `spec` isn't a recognizable range (caller then leaves the
    original `date[...]` text untouched, so an unrecognized spec falls back to plain
    term[field] lowering rather than erroring)."""
    if not spec:
        return None
    if ".." in spec:
        a, _, b = spec.partition("..")
        if not a or not b:
            return None
        lo, hi = _start_of(a), _end_of(b)
        return (lo, hi) if (lo is not None and hi is not None) else None
    for op in ("<=", ">=", "<", ">"):     # longer ops first: "<=" must win over "<"
        if not spec.startswith(op):
            continue
        rest = spec[len(op):]
        if op == "<=":
            hi = _end_of(rest)
            return (None, hi) if hi is not None else None
        if op == ">=":
            lo = _start_of(rest)
            return (lo, None) if lo is not None else None
        if op == "<":
            b = _start_of(rest)
            hi = _shift_day(b, -1) if b is not None else None
            return (None, hi) if hi is not None else None
        if op == ">":
            b = _end_of(rest)
            lo = _shift_day(b, 1) if b is not None else None
            return (lo, None) if lo is not None else None
    lo, hi = _start_of(spec), _end_of(spec)          # bare year / year-month / date
    return (lo, hi) if (lo is not None and hi is not None) else None


def _rewrite_date_ranges(query: str) -> str:
    """Replace every `date[RANGE]` span with its `__daterange__LO__HI` encoding. A span whose
    bracket content isn't a recognizable range is left exactly as written (old behavior)."""
    def repl(m: re.Match) -> str:
        spec = re.sub(r"\s+", "", m.group(1))
        rng = _parse_date_range_spec(spec)
        if rng is None:
            return m.group(0)
        lo, hi = rng
        return f"{_DATE_RANGE_PREFIX}{lo or 'open'}__{hi or 'open'}"
    return _DATE_FIELD_RE.sub(repl, query)


def _tokens(s: str):
    i, out = 0, []
    while i < len(s):
        m = _TOK.match(s, i)
        if not m or m.end() == i:
            i += 1
            continue
        i = m.end()
        out.append((m.lastgroup, m.group(m.lastgroup)))
    return out


def _leaf(atom: str) -> str:
    if atom.startswith('"'):
        return atom
    if atom.startswith(_DATE_RANGE_PREFIX):
        return f"IN(date, {atom})"      # already-encoded date-range leaf; see _rewrite_date_ranges
    return f"PREFIX({atom[:-1]})" if atom.endswith("*") else atom


def _scoped(atom_bql: str, fields: list[str], alias: dict) -> str:
    regions = []
    for f in fields:
        regions.extend(alias.get(f.lower(), [f]))           # unknown -> passthrough
    ins = [f"IN({r}, {atom_bql})" for r in regions]
    return ins[0] if len(ins) == 1 else "OR(" + ", ".join(ins) + ")"


def _run_to_bql(run: list[tuple[str, bool]], fields: list[str], alias: dict) -> str:
    """A run of consecutive bare words/phrases (+ optional field), each tagged with
    whether it was `-`-prefixed (negated), -> BQL.
    - single word/phrase, not negated -> scoped leaf
    - single word/phrase, negated (a lone `-term` with no positive sibling in this
      run) -> bare `NOT(...)`; BQL requires NOT to be a clause of AND (module
      docstring's "NOT is unary-in-AND"), so this is only valid when the caller
      combines it with a positive clause via an explicit AND/OR elsewhere in the
      query (see `_pass1`'s dispatch) -- a truly standalone negation has no valid
      BQL form and is correctly left to raise a clear type-check error
      ("AND needs at least one positive clause"), not papered over here.
    - unquoted multi-word (an entity name typed bare) -> AND of the field-scoped terms,
      each `NOT(...)`-wrapped if it was `-`-prefixed (all positive words present,
      order-independent: forgiving, since a strict phrase 0-hits too often)
    - a quoted span the user wrote -> stays a phrase (they asked for contiguity)
    So `Command handle[def]` -> AND(IN(def,Command), IN(def,handle));
       `"cannot rollback"[string]` -> IN(string, "cannot rollback");
       `alpha -gamma` -> AND(alpha, NOT(gamma))."""
    def emit(atom: str, neg: bool) -> str:
        leaf = _leaf(atom)
        scoped = _scoped(leaf, fields, alias) if fields else leaf
        return f"NOT({scoped})" if neg else scoped

    if len(run) == 1:
        atom, neg = run[0]
        return emit(atom, neg)
    return "AND(" + ", ".join(emit(atom, neg) for atom, neg in run) + ")"


def _pass1(query: str, alias: dict) -> list[str]:
    """Collapse each atom(+field) into one BQL-leaf token; keep AND/OR/NOT/( ) as control tokens.
    A run of consecutive bare words ending in [field] scopes the whole run, so a multi-word
    name like `Command handle[def]` becomes AND(IN(def,Command), IN(def,handle)). A `-`-prefixed
    word/quote (`negword`/`negquote`) is part of the same run as any adjacent word/quote (so
    `alpha -gamma` collects into one run, not two juxtaposed-with-no-operator units), just
    tagged negated -- `_run_to_bql` wraps it in `NOT(...)`."""
    toks = _tokens(query)
    out, i = [], 0
    while i < len(toks):
        kind, val = toks[i]
        if kind == "word" and val.upper() in _OPS:
            out.append(val.upper())
            i += 1
            continue
        if kind in ("word", "quote") or kind in _NEG_KINDS:
            run: list[tuple[str, bool]] = []
            scoped = False
            while i < len(toks):
                k, v = toks[i]
                if (k == "word" and v.upper() in _OPS) or (
                        k not in ("word", "quote") and k not in _NEG_KINDS):
                    break
                neg = k in _NEG_KINDS
                run.append((v[1:] if neg else v, neg))            # strip the leading "-"
                if i + 1 < len(toks) and toks[i + 1][0] == "lbrack":   # this word carries the field
                    j, fields = i + 2, []
                    while j < len(toks) and toks[j][0] != "rbrack":
                        if toks[j][0] == "word":
                            fields.append(toks[j][1])
                        j += 1
                    out.append(_run_to_bql(run, fields, alias))
                    i = j + 1
                    scoped = True
                    break
                i += 1
            if not scoped and run:                                     # trailing unscoped run
                out.append(_run_to_bql(run, [], alias))
        elif kind == "lparen":
            out.append("(")
            i += 1
        elif kind == "rparen":
            out.append(")")
            i += 1
        else:
            i += 1
    return out


def _unit(toks: list[str], i: int) -> tuple[str, int]:
    """The BQL string for the unit starting at i (a leaf token, or a balanced (...) group)."""
    if toks[i] == "(":
        depth, j = 0, i
        while j < len(toks):
            if toks[j] == "(":
                depth += 1
            elif toks[j] == ")":
                depth -= 1
                if depth == 0:
                    return " ".join(toks[i:j + 1]), j + 1
            j += 1
        return " ".join(toks[i:]), len(toks)
    return toks[i], i + 1


def _pass2(toks: list[str]) -> str:
    """Emit BQL, rewriting binary `NOT B` -> `AND NOT( B )`, but only when an infix
    operator isn't already immediately in front of it. `_pass1` emits a literal "NOT"
    control token whenever the original surface text had the literal keyword NOT
    (case-insensitive), whether or not the user also typed an explicit AND/OR right
    before it:
      - `A NOT B`     (no explicit AND) -> out ends with a leaf; still needs "AND "
                       prepended so the result is valid BQL (`A AND NOT(B)`).
      - `A AND NOT B` (explicit AND)    -> out already ends with "AND"; prepending a
                       second "AND" would produce the malformed, unparseable
                       `A AND AND NOT(B)`. Since the infix operator is already there,
                       just emit `NOT(B)` and let it attach.
    The same reasoning applies to `A OR NOT B` (would otherwise become the equally
    malformed `A OR AND NOT(B)`) -- checked generically via "does `out` already end in
    an infix operator", not AND-specific."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t == "NOT":
            unit, i = _unit(toks, i + 1)
            if out and out[-1] in ("AND", "OR"):
                out.append(f"NOT({unit})")
            else:
                out.append(f"AND NOT({unit})")
        else:
            out.append(t)
            i += 1
    s = " ".join(out)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s)).strip()


def to_bql(query: str, domain: str = "code") -> str:
    """Translate a field-tagged surface query to a BQL string. `domain` selects the field
    aliases (code = AST regions; doc = title/body/section/...).

    Run the round-trip check via the test suite (tests/test_surface.py),
    not as ``python surface.py``: executing this file as a script puts the bql/ dir on
    sys.path[0], shadowing the stdlib ``types`` module that ``bql/types`` is named after."""
    alias = DOC_FIELDS if domain == "doc" else CODE_FIELDS
    if _date_range_enabled():
        query = _rewrite_date_ranges(query)
    return _pass2(_pass1(query, alias))
