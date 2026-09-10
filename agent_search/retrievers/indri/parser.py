"""Indri Query Language parser: query string -> small AST of dataclasses.

Source of truth: `agent_search/tools/search_indri/indri_doc.md` (verbatim-where-possible from the
official Indri Query Language reference / quick reference / belief-operations wiki).
This module implements the SUBSET named in that task's scope; anything outside the
subset raises a structured `IndriParseError` naming the unsupported operator rather
than silently mis-parsing it.

Grammar (informal, character-level recursive descent):

    query      := atom+                          (bare atoms implicitly #combine'd)
    atom       := primary field-suffix*
    primary    := '#' opname '(' ... ')'          (belief/proximity/filter/date ops)
                | '{' atom* '}'                   (#syn shortcut)
                | '<' atom* '>'                   (#syn shortcut)
                | '"' ... '"' | "'" ... "'"        (quoted term)
                | word                            (term, or term* suffix wildcard)
    field-suffix := '.' fieldlist                 (field restriction, e.g. `dog.title`)
                  | '.' '(' field ')'              (field-context evaluation, `dog.(title)`)

Commas are treated as insignificant whitespace everywhere EXCEPT inside a field
list (`dog.title,header`), where they separate field names. This is looser than
official Indri (which doesn't require commas at all in most positions) but never
rejects a query the official grammar would accept within our subset.

## Deviations (from agent_search/tools/search_indri/indri_doc.md)
- No stemming/normalization distinction between `term` and `"term"`: this backend
  never stems, so both are tokenized identically via `code_tokenize`. `Term.quoted`
  is retained on the AST only for round-tripping/diagnostics, never affects matching.
- `#od`/`#uw` WITHOUT a trailing window number (unlimited window) ARE supported
  (`n=None` on the `Window` node) — the reference marks this optional; we implement it
  because it costs nothing extra and the belief math for it is well defined (any-order
  match count, or ordered-anywhere match count, over the whole field).
- `#between(FIELD N_low N_high)` is accepted ONLY for `FIELD == date` (per the task's
  scope note); any other field raises an unsupported-operator error. `#less`/`#greater`/
  `#equals` on arbitrary numeric fields are NOT implemented (out of scope) and raise
  unsupported-operator errors.
- `#wsum`, `#wand`, `#sum` are NOT implemented — their belief math (weighted SUM of
  raw beliefs / boolean AND / plain sum) differs materially from `#weight`'s weighted
  MEAN of log-beliefs, so silently aliasing them would misrepresent the query; they
  raise a structured unsupported-operator error naming the operator.
- `#prior`, `#any`/`#any:FIELD`, `#base64*`, extent/passage retrieval
  (`#op[field](...)`, `#op[passageW:INC](...)`), and parent/ancestor references are
  all explicitly out of scope per the task spec and raise unsupported-operator errors
  naming the operator (or, for `[...]` syntax, "extent/passage retrieval").
- Date literals are restricted to ISO `YYYY[-MM[-DD]]` (per the task's corpus format);
  the alternate formats Indri itself accepts ("11 january 2004", "11-JAN-04",
  "01/11/04") are NOT parsed and raise a structured error citing the date operator.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from typing import Optional, Union

# --- AST --------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    text: str
    quoted: bool = False


@dataclass(frozen=True)
class Wildcard:
    stem: str


@dataclass(frozen=True)
class Window:
    kind: str                    # "od" | "uw"
    n: Optional[int]             # None = unlimited
    children: tuple = ()


@dataclass(frozen=True)
class Syn:
    children: tuple = ()


@dataclass(frozen=True)
class WSyn:
    pairs: tuple = ()            # tuple[tuple[float, Expr], ...]


@dataclass(frozen=True)
class Combine:
    children: tuple = ()


@dataclass(frozen=True)
class Weight:
    pairs: tuple = ()            # tuple[tuple[float, Expr], ...]


@dataclass(frozen=True)
class Or:
    children: tuple = ()


@dataclass(frozen=True)
class Not:
    child: "Expr" = None


@dataclass(frozen=True)
class Max:
    children: tuple = ()


@dataclass(frozen=True)
class Band:
    children: tuple = ()


@dataclass(frozen=True)
class FieldExpr:
    """`expr.field` (restrict counts to `fields`), `expr.(context)` (smooth against
    `context`'s collection model), or the combined `expr.field.(context)`."""
    child: "Expr" = None
    fields: Optional[tuple] = None     # counting scope, e.g. ("title",) or ("title","section")
    context: Optional[str] = None      # smoothing/collection-model scope


@dataclass(frozen=True)
class FilReq:
    a: "Expr" = None
    q: "Expr" = None


@dataclass(frozen=True)
class FilRej:
    a: "Expr" = None
    q: "Expr" = None


@dataclass(frozen=True)
class DateBefore:
    date: str = ""


@dataclass(frozen=True)
class DateAfter:
    date: str = ""


@dataclass(frozen=True)
class DateBetween:
    lo: Optional[str] = None
    hi: Optional[str] = None


Expr = Union[Term, Wildcard, Window, Syn, WSyn, Combine, Weight, Or, Not, Max, Band,
             FieldExpr, FilReq, FilRej, DateBefore, DateAfter, DateBetween]


# --- structured errors --------------------------------------------------------

@dataclass
class IndriParseError:
    op: Optional[str]
    pos: int
    message: str

    def __str__(self) -> str:                       # pragma: no cover - cosmetic
        where = f"#{self.op}" if self.op else "<query>"
        return f"{where}@{self.pos}: {self.message}"


@dataclass
class ParseResult:
    ok: bool
    expr: Optional[Expr] = None
    error: Optional[IndriParseError] = None


class _PErr(Exception):
    def __init__(self, err: IndriParseError):
        self.err = err


# --- unsupported-operator tables --------------------------------------------

_UNSUPPORTED_NAMES = {
    "prior": "priors (#prior) are not supported",
    "any": "extent-type wildcards (#any/#any:FIELD) are not supported",
    "wsum": "#wsum is not supported (its weighted-SUM belief math differs from #weight)",
    "wand": "#wand is not supported (weighted boolean AND is out of scope)",
    "sum": "#sum is not supported (plain-sum belief math is out of scope)",
    "less": "#less is not supported (arbitrary numeric-field filters are out of scope)",
    "greater": "#greater is not supported (arbitrary numeric-field filters are out of scope)",
    "equals": "#equals is not supported (arbitrary numeric-field filters are out of scope)",
    "scoreif": "#scoreif is not supported (use #filreq)",
    "scoreifnot": "#scoreifnot is not supported (use #filrej)",
}

_DATE_ISO_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

# --- character classes --------------------------------------------------------

_OPNAME_RE = re.compile(r"[A-Za-z0-9:_]+")
_BAREWORD_RE = re.compile(r"""[^\s()#{}<>"'.,]+""")
_FIELDLIST_RE = re.compile(r"[A-Za-z0-9_]+(?:,[A-Za-z0-9_]+)*")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_DATEWORD_RE = re.compile(r"[^\s(),]+")


class _Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    # --- low-level cursor helpers ------------------------------------------

    def _skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t\n\r,":
            self.i += 1

    def _peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def _err(self, message: str, op: Optional[str] = None, pos: Optional[int] = None) -> None:
        raise _PErr(IndriParseError(op=op, pos=self.i if pos is None else pos, message=message))

    def _match(self, regex: re.Pattern, what: str, op: Optional[str] = None) -> str:
        m = regex.match(self.s, self.i)
        if not m or m.start() == m.end():
            self._err(f"expected {what}", op=op)
        self.i = m.end()
        return m.group(0)

    def _expect_char(self, ch: str, op: Optional[str] = None) -> None:
        self._skip_ws()
        if self._peek() != ch:
            self._err(f"expected {ch!r}", op=op)
        self.i += 1

    # --- top level -----------------------------------------------------------

    def parse_query(self) -> Expr:
        atoms = []
        self._skip_ws()
        while self.i < self.n:
            atoms.append(self.parse_atom())
            self._skip_ws()
        if not atoms:
            self._err("empty query")
        return atoms[0] if len(atoms) == 1 else Combine(tuple(atoms))

    # --- atom = primary + field suffixes -------------------------------------

    def parse_atom(self) -> Expr:
        node = self.parse_primary()
        fields: Optional[tuple] = None
        context: Optional[str] = None
        while self._peek() == ".":
            self.i += 1
            if self._peek() == "(":
                self.i += 1
                self._skip_ws()
                context = self._match(_FIELDLIST_RE, "field name in .(field)")
                self._skip_ws()
                self._expect_char(")")
            else:
                flds = self._match(_FIELDLIST_RE, "field name after '.'")
                fields = tuple(flds.split(","))
        if fields is not None or context is not None:
            node = FieldExpr(child=node, fields=fields, context=context)
        return node

    # --- primary ---------------------------------------------------------------

    def parse_primary(self) -> Expr:
        self._skip_ws()
        if self.i >= self.n:
            self._err("unexpected end of query")
        c = self._peek()
        if c == "#":
            return self.parse_operator()
        if c == "{":
            self.i += 1
            children = self.parse_expr_list_until("}")
            return Syn(tuple(children))
        if c == "<":
            self.i += 1
            children = self.parse_expr_list_until(">")
            return Syn(tuple(children))
        if c in ("\"", "'"):
            return self.parse_quoted(c)
        if c in ")}>":
            self._err(f"unexpected {c!r}")
        word = self._match(_BAREWORD_RE, "a term")
        if len(word) > 1 and word.endswith("*"):
            return Wildcard(stem=word[:-1])
        return Term(text=word)

    def parse_quoted(self, quote: str) -> Expr:
        start = self.i
        self.i += 1
        buf = []
        while self.i < self.n and self.s[self.i] != quote:
            if self.s[self.i] == "\\" and self.i + 1 < self.n:
                buf.append(self.s[self.i + 1])
                self.i += 2
            else:
                buf.append(self.s[self.i])
                self.i += 1
        if self.i >= self.n:
            self._err("unterminated quoted term", pos=start)
        self.i += 1                        # closing quote
        return Term(text="".join(buf), quoted=True)

    # --- '#op(...)' ------------------------------------------------------------

    def parse_operator(self) -> Expr:
        start = self.i
        self.i += 1                        # consume '#'
        name = self._match(_OPNAME_RE, "operator name")
        lname = name.lower()
        if self._peek() == "[":
            self._err("extent/passage retrieval (#op[field]/#op[passageW:INC]) is not "
                       "supported", op=name, pos=start)
        if lname == "any" or lname.startswith("any:"):
            self._err(_UNSUPPORTED_NAMES["any"], op=name, pos=start)
        self._expect_char("(", op=name)

        # --- proximity windows: #odN / #uwN / #N / #od / #uw -------------------
        m = re.match(r"^(od|uw)(\d*)$", lname)
        if m:
            kind, digits = m.group(1), m.group(2)
            nwin = int(digits) if digits else None
            children = self.parse_expr_list_until(")")
            if not children:
                self._err(f"#{name} requires at least one child", op=name)
            return Window(kind=kind, n=nwin, children=tuple(children))
        if lname.isdigit():
            children = self.parse_expr_list_until(")")
            if not children:
                self._err(f"#{name} requires at least one child", op=name)
            return Window(kind="od", n=int(lname), children=tuple(children))

        if lname == "syn":
            return Syn(tuple(self.parse_expr_list_until(")")))
        if lname == "wsyn":
            return WSyn(tuple(self.parse_weighted_pairs_until(")", op=name)))
        if lname == "combine":
            return Combine(tuple(self.parse_expr_list_until(")")))
        if lname == "weight":
            return Weight(tuple(self.parse_weighted_pairs_until(")", op=name)))
        if lname == "or":
            return Or(tuple(self.parse_expr_list_until(")")))
        if lname == "not":
            child = self.parse_atom()
            self._skip_ws()
            self._expect_char(")", op=name)
            return Not(child=child)
        if lname == "max":
            return Max(tuple(self.parse_expr_list_until(")")))
        if lname == "band":
            return Band(tuple(self.parse_expr_list_until(")")))
        if lname == "wildcard":
            self._skip_ws()
            word = self._match(_BAREWORD_RE, "a term", op=name)
            self._skip_ws()
            self._expect_char(")", op=name)
            return Wildcard(stem=word.rstrip("*"))
        if lname == "filreq":
            a = self.parse_atom()
            q = self.parse_atom()
            self._skip_ws()
            self._expect_char(")", op=name)
            return FilReq(a=a, q=q)
        if lname == "filrej":
            a = self.parse_atom()
            q = self.parse_atom()
            self._skip_ws()
            self._expect_char(")", op=name)
            return FilRej(a=a, q=q)
        if lname in ("date:before", "datebefore"):
            d = self.parse_date_literal(name)
            self._skip_ws()
            self._expect_char(")", op=name)
            return DateBefore(date=d)
        if lname in ("date:after", "dateafter"):
            d = self.parse_date_literal(name)
            self._skip_ws()
            self._expect_char(")", op=name)
            return DateAfter(date=d)
        if lname in ("date:between", "datebetween"):
            lo = self.parse_date_literal(name)
            hi = self.parse_date_literal(name)
            self._skip_ws()
            self._expect_char(")", op=name)
            return DateBetween(lo=lo, hi=hi)
        if lname == "between":
            self._skip_ws()
            fld = self._match(_BAREWORD_RE, "field name", op=name)
            if fld.lower() != "date":
                self._err(f"#between is only supported for field=date (got {fld!r})", op=name)
            lo = self.parse_date_literal(name)
            hi = self.parse_date_literal(name)
            self._skip_ws()
            self._expect_char(")", op=name)
            return DateBetween(lo=lo, hi=hi)

        base = lname.split(":")[0]
        if lname in _UNSUPPORTED_NAMES or base in _UNSUPPORTED_NAMES or lname.startswith("base64"):
            msg = _UNSUPPORTED_NAMES.get(lname) or _UNSUPPORTED_NAMES.get(base) \
                or f"#{name} (base64 operators) are not supported"
            self._err(msg, op=name, pos=start)
        self._err(f"unsupported operator #{name}", op=name, pos=start)

    def parse_date_literal(self, op: str) -> str:
        self._skip_ws()
        tok = self._match(_DATEWORD_RE, "a date (YYYY[-MM[-DD]])", op=op)
        if not _DATE_ISO_RE.match(tok):
            self._err(f"malformed date literal {tok!r}: expected ISO YYYY[-MM[-DD]]", op=op)
        return tok

    # --- shared list helpers -----------------------------------------------

    def parse_expr_list_until(self, close: str) -> list:
        children = []
        self._skip_ws()
        while self._peek() != close:
            if self.i >= self.n:
                self._err(f"unterminated expression list (expected {close!r})")
            children.append(self.parse_atom())
            self._skip_ws()
        self.i += 1                        # consume close char
        return children

    def parse_weighted_pairs_until(self, close: str, op: str) -> list:
        pairs = []
        self._skip_ws()
        while self._peek() != close:
            if self.i >= self.n:
                self._err(f"unterminated weighted list (expected {close!r})", op=op)
            numtxt = self._match(_NUM_RE, "a weight number", op=op)
            self._skip_ws()
            expr = self.parse_atom()
            pairs.append((float(numtxt), expr))
            self._skip_ws()
        self.i += 1
        return pairs


def parse(query: str) -> ParseResult:
    """Parse an Indri query language string into a `ParseResult`.

    Never raises: any structural problem (unknown operator, malformed date,
    unbalanced parens, etc.) comes back as `ParseResult(ok=False, error=...)`
    with a structured `IndriParseError` (op name + position + message) so the
    agent tool layer can surface it as syntax feedback."""
    p = _Parser(query)
    try:
        expr = p.parse_query()
        return ParseResult(ok=True, expr=expr)
    except _PErr as e:
        return ParseResult(ok=False, error=e.err)
    except Exception as e:                 # defensive: never crash the agent loop
        return ParseResult(ok=False, error=IndriParseError(op=None, pos=p.i, message=str(e)))


# --- date bound helpers (shared with model.py) -------------------------------

def date_bounds(d: str) -> tuple:
    """ISO partial date `YYYY[-MM[-DD]]` -> (start, end) full ISO 'YYYY-MM-DD' strings
    spanning the whole granularity (a bare year covers the whole year, etc.)."""
    parts = d.split("-")
    y = int(parts[0])
    if len(parts) == 1:
        return f"{y:04d}-01-01", f"{y:04d}-12-31"
    mo = int(parts[1])
    if len(parts) == 2:
        last = calendar.monthrange(y, mo)[1]
        return f"{y:04d}-{mo:02d}-01", f"{y:04d}-{mo:02d}-{last:02d}"
    dd = int(parts[2])
    full = f"{y:04d}-{mo:02d}-{dd:02d}"
    return full, full


# --- pretty-printer (for diagnostics: naming "which constraint is weak") -----

def render(expr: Expr) -> str:
    """Compact, approximately-round-tripping rendering of an AST node — used only
    for human-readable diagnostics (not re-parsed)."""
    if expr is None:
        return "<none>"
    if isinstance(expr, Term):
        return f'"{expr.text}"' if expr.quoted else expr.text
    if isinstance(expr, Wildcard):
        return f"{expr.stem}*"
    if isinstance(expr, Window):
        n = expr.n if expr.n is not None else ""
        return f"#{expr.kind}{n}(" + " ".join(render(c) for c in expr.children) + ")"
    if isinstance(expr, Syn):
        return "#syn(" + " ".join(render(c) for c in expr.children) + ")"
    if isinstance(expr, WSyn):
        return "#wsyn(" + " ".join(f"{w} {render(e)}" for w, e in expr.pairs) + ")"
    if isinstance(expr, Combine):
        return "#combine(" + " ".join(render(c) for c in expr.children) + ")"
    if isinstance(expr, Weight):
        return "#weight(" + " ".join(f"{w} {render(e)}" for w, e in expr.pairs) + ")"
    if isinstance(expr, Or):
        return "#or(" + " ".join(render(c) for c in expr.children) + ")"
    if isinstance(expr, Not):
        return f"#not({render(expr.child)})"
    if isinstance(expr, Max):
        return "#max(" + " ".join(render(c) for c in expr.children) + ")"
    if isinstance(expr, Band):
        return "#band(" + " ".join(render(c) for c in expr.children) + ")"
    if isinstance(expr, FieldExpr):
        out = render(expr.child)
        if expr.fields:
            out += "." + ",".join(expr.fields)
        if expr.context:
            out += f".({expr.context})"
        return out
    if isinstance(expr, FilReq):
        return f"#filreq({render(expr.a)} {render(expr.q)})"
    if isinstance(expr, FilRej):
        return f"#filrej({render(expr.a)} {render(expr.q)})"
    if isinstance(expr, DateBefore):
        return f"#date:before({expr.date})"
    if isinstance(expr, DateAfter):
        return f"#date:after({expr.date})"
    if isinstance(expr, DateBetween):
        return f"#date:between({expr.lo} {expr.hi})"
    return repr(expr)
