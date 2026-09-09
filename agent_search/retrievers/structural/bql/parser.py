"""BQL parser: prefix-form string -> typed AST.

Prefix/functional form is chosen for reliable LLM generation (bql_spec.md §1).
Parse errors are returned as structured results (fed back as agent observation),
never raised to crash the loop. NOT is accepted syntactically anywhere; its
placement is enforced by the type checker (bql/types.py), which yields a clearer
error than a parse failure would.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent_search.retrievers.structural.bql.ast import (
    And, Expand, Expr, In, Near, Not, Or, Phrase, Prefix, Region, Strategy, Term,
)


@dataclass
class ParseResult:
    ok: bool
    expr: Expr | None = None
    error: str | None = None


class _ParseError(Exception):
    pass


# --- tokenizer --------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      \s*(?:
        (?P<lparen>\() |
        (?P<rparen>\)) |
        (?P<comma>,) |
        (?P<near>(?i:NEAR)/(?:[A-Za-z]+[0-9]*|[0-9]+)) |
        (?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*') |
        (?P<ident>\w[\w./*+#-]*)
      )\s*
    """,
    re.VERBOSE | re.UNICODE,
)

_KEYWORDS = {"AND", "OR", "NOT", "IN", "EXPAND", "PHRASE", "PREFIX"}


def _tokenize(s: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(s, pos)
        if not m or m.start() == m.end():
            raise _ParseError(f"unexpected character at position {pos}: {s[pos]!r}")
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


# --- recursive-descent parser ----------------------------------------------

class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.toks = tokens
        self.i = 0

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def _next(self):
        tok = self._peek()
        self.i += 1
        return tok

    def _expect(self, kind: str):
        k, v = self._next()
        if k != kind:
            # Teaching hint: a bare word / quoted string where a ')' or ',' was
            # expected almost always means an unquoted multi-word operand.
            if kind in ("rparen", "comma") and k in ("ident", "string"):
                raise _ParseError(
                    f"expected {kind}, got {v!r} — looks like a multi-word "
                    f'operand; quote it ("a b") or use PHRASE(a, b)')
            raise _ParseError(f"expected {kind}, got {v!r}")
        return v

    def parse(self) -> Expr:
        if not self.toks:
            raise _ParseError("empty query")
        expr = self._or_expr()
        if self.i != len(self.toks):
            k, v = self._peek()
            if k in ("ident", "string"):
                raise _ParseError(
                    f"trailing tokens after a complete query: {v!r} — if this is "
                    f'part of a multi-word operand, quote it ("a b") or use '
                    f"PHRASE(a, b); to combine clauses use AND/OR")
            raise _ParseError(f"trailing tokens after a complete query: {v!r}")
        return expr

    def _is_kw(self, word: str) -> bool:
        k, v = self._peek()
        return k == "ident" and v is not None and v.upper() == word

    # Surface-syntax tolerance: accept infix `A AND B`, `A OR B`, comma lists
    # `A, B` (top level / inside grouping parens), and parenthesized grouping
    # `(A OR B) AND C`, in addition to the canonical prefix `AND(A, B)` / `OR(A, B)`.
    # Same Boolean structure; models reach for infix/comma/parens (the standard
    # Lucene-style surface form). Precedence: OR < AND. Inside operator argument
    # lists the comma stays an argument separator (comma_is_and=False) but infix
    # AND/OR still bind: `AND(foo, bar OR baz)` == AND(foo, OR(bar, baz)).
    def _or_expr(self, comma_is_and: bool = True) -> Expr:
        parts = [self._and_expr(comma_is_and)]
        while self._is_kw("OR"):
            self._next()
            parts.append(self._and_expr(comma_is_and))
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def _and_expr(self, comma_is_and: bool = True) -> Expr:
        parts = [self._expr()]
        while (comma_is_and and self._peek()[0] == "comma") or self._is_kw("AND"):
            self._next()
            parts.append(self._expr())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def _starts_call(self) -> bool:
        """True if the current ident is an operator keyword in CALL position
        (immediately followed by '('), e.g. NOT( / IN( / PHRASE(. Such a keyword
        must NOT be swallowed into an implicit phrase run."""
        k, v = self._peek()
        if k != "ident" or v is None or v.upper() not in _KEYWORDS:
            return False
        return self.i + 1 < len(self.toks) and self.toks[self.i + 1][0] == "lparen"

    def _string_term(self, val: str) -> Term:
        text = re.sub(r"\\(.)", r"\1", val[1:-1])   # unescape \" \' \\
        if not text.strip():
            raise _ParseError("empty quoted term")
        return Term(text, quoted=True)

    def _bare_run(self, first: Term) -> Expr:
        """Forgiveness: a run of consecutive bare operands (bare words and/or
        quoted strings) with no operator between them is read as an implicit
        PHRASE — the model's natural instinct is to type a phrase, and BQL used
        to hard-error on that. The run STOPS at an infix boolean (AND/OR), at a
        keyword in call position (NOT(/IN(/...), and at any structural token
        (comma/paren/near). A single operand is returned unchanged, so every
        currently-valid query parses identically."""
        parts: list[Term] = [first]
        while True:
            k, v = self._peek()
            if k == "ident":
                if v.upper() in ("AND", "OR"):
                    break                         # infix boolean stays an operator
                if self._starts_call():
                    break                         # a call form, not a phrase word
                self._next()
                parts.append(Term(v))
            elif k == "string":
                self._next()
                parts.append(self._string_term(v))
            else:
                break
        if len(parts) == 1:
            return parts[0]
        return Phrase(tuple(parts))

    def _expr(self) -> Expr:
        kind, val = self._peek()
        if kind is None:
            raise _ParseError("unexpected end of input")
        if kind == "lparen":                      # grouping: (a OR b) AND c
            self._next()
            expr = self._or_expr()
            self._expect("rparen")
            return expr
        if kind == "near":
            return self._near()
        if kind == "string":
            self._next()
            # a quoted string may begin an implicit phrase run too
            # (e.g. `"foo bar" baz` -> PHRASE("foo bar", baz))
            return self._bare_run(self._string_term(val))
        if kind == "ident":
            self._next()  # consume the identifier / operator keyword
            # operator keywords are case-insensitive, but only with a call form
            # following — so a bare `and` / `In` can still be a search term
            if val.upper() in _KEYWORDS and self._peek()[0] == "lparen":
                return self._call(val.upper())
            # bare term, greedily merging a run of bare words into a phrase
            # (also catches PREFIX-style wildcard idents as the single-token case)
            return self._bare_run(Term(val))
        raise _ParseError(f"unexpected token {val!r}")

    def _args(self) -> list[Expr]:
        self._expect("lparen")
        args = [self._or_expr(comma_is_and=False)]
        while self._peek()[0] == "comma":
            self._next()
            args.append(self._or_expr(comma_is_and=False))
        self._expect("rparen")
        return args

    def _call(self, op: str) -> Expr:
        if op == "AND":
            return And(tuple(self._args()))
        if op == "OR":
            return Or(tuple(self._args()))
        if op == "NOT":
            args = self._args()
            if len(args) != 1:
                raise _ParseError("NOT takes exactly one argument")
            return Not(args[0])
        if op == "PHRASE":
            args = self._args()
            terms = []
            for a in args:
                if not isinstance(a, Term):
                    raise _ParseError("PHRASE arguments must be terms")
                terms.append(a)
            return Phrase(tuple(terms))
        if op == "PREFIX":
            args = self._args()
            if len(args) != 1 or not isinstance(args[0], Term):
                raise _ParseError("PREFIX takes one term (the stem)")
            return Prefix(args[0].text)
        if op == "EXPAND":
            return self._expand()
        if op == "IN":
            return self._in()
        raise _ParseError(f"unknown operator {op!r}")

    def _expand(self) -> Expand:
        self._expect("lparen")
        term = self._expr()
        if not isinstance(term, Term):
            raise _ParseError("EXPAND's first argument must be a term")
        strategy = Strategy.SYNONYM
        if self._peek()[0] == "comma":
            self._next()
            name = self._expect("ident")
            try:
                strategy = Strategy(name.lower())
            except ValueError:
                raise _ParseError(f"unknown EXPAND strategy {name!r}")
        self._expect("rparen")
        return Expand(term, strategy)

    def _in(self) -> In:
        self._expect("lparen")
        region_name = self._expect("ident")
        try:
            region = Region(region_name.lower())
        except ValueError:
            raise _ParseError(f"unknown region {region_name!r}")
        self._expect("comma")
        child = self._or_expr(comma_is_and=False)
        self._expect("rparen")
        return In(region, child)

    def _near(self):
        _, val = self._next()                 # e.g. "NEAR/func", "NEAR/w5", "NEAR/5"
        spec = val.split("/", 1)[1].lower()
        if spec.isdigit():
            spec = "w" + spec                 # Lucene/ProQuest-style NEAR/5 = NEAR/w5
        args = self._args()
        from agent_search.retrievers.structural.bql.types import gran_level  # local import to avoid cycle
        try:
            gran = gran_level(spec)           # validate the spec here, as a parse error
        except Exception:
            raise _ParseError(f"unknown NEAR granularity {spec!r}")
        if len(args) == 1:
            # lenient: keep the granularity lift instead of silently dropping it —
            # NEAR/file(x) means "x, viewed at file scope", not bare Term(x)
            return Near(args[0], args[0], gran, spec)
        if len(args) != 2:
            raise _ParseError("NEAR takes one or two arguments")
        return Near(args[0], args[1], gran, spec)


def parse(query: str) -> ParseResult:
    """Parse a BQL string. Pure; does not type-check (see types.check)."""
    try:
        tokens = _tokenize(query)
        expr = _Parser(tokens).parse()
        return ParseResult(ok=True, expr=expr)
    except _ParseError as e:
        return ParseResult(ok=False, error=str(e))
    except RecursionError:
        # degenerate LLM output (e.g. 'NOT('*400 ...) must not crash the episode
        return ParseResult(ok=False, error="query too deeply nested")
