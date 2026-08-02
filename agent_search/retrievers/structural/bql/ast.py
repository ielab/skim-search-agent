"""BQL abstract syntax tree.

Typed prefix-form Boolean Query Language. See docs/bql_spec.md for the grammar,
the granularity type lattice, and the operator->polarity spine. This module
defines the node types only; parsing lives in `parser.py`, type-checking in
`types.py`, and execution in `agent_search.retrievers.structural.bql.executor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Granularity(Enum):
    """Lattice: TOKEN < LINE < BLOCK < FUNC < FILE < DOC (ordering = `level`)."""
    TOKEN = 0
    LINE = 1
    BLOCK = 2
    FUNC = 3
    FILE = 4
    DOC = 5


class Region(Enum):
    """Structural scope for IN(region, .). Text fields + code AST roles."""
    TITLE = "title"
    BODY = "body"
    SECTION = "section"
    AUTHOR = "author"     # document metadata fields (deep-research corpora)
    DATE = "date"
    INFOBOX = "infobox"
    COMMENT = "comment"
    STRING = "string"
    DEF = "def"
    CALL = "call"
    SIG = "sig"
    FILE = "file"
    DOC = "doc"


class Strategy(Enum):
    """How EXPAND resolves its variant set."""
    LEXICAL = "lexical"   # prefix/wildcard family
    SYMBOL = "symbol"     # identifier naming variants + symbol/call graph (code)
    SYNONYM = "synonym"   # corpus-validated synonym set (text)


# --- Expression nodes -------------------------------------------------------

class Expr:
    """Base class. Every Expr is assigned a granularity level by the type checker."""


@dataclass(frozen=True)
class Term(Expr):
    text: str
    quoted: bool = False


@dataclass(frozen=True)
class Phrase(Expr):
    terms: Sequence[Term]


@dataclass(frozen=True)
class Prefix(Expr):
    stem: str  # identifier-boundary-aware in the code backend


@dataclass(frozen=True)
class Expand(Expr):
    term: Term
    strategy: Strategy = Strategy.SYNONYM


@dataclass(frozen=True)
class And(Expr):
    children: Sequence[Expr]


@dataclass(frozen=True)
class Or(Expr):
    children: Sequence[Expr]


@dataclass(frozen=True)
class Not(Expr):
    """Only valid as the right child of an And (set-difference). Enforced by the
    parser/type-checker, never standalone. See bql_spec.md §1."""
    child: Expr


@dataclass(frozen=True)
class Near(Expr):
    left: Expr
    right: Expr
    gran: Granularity   # lifted type level (see bql/types.py gran-spec mapping)
    spec: str           # raw granularity token as written, e.g. "w5", "line3", "func"


@dataclass(frozen=True)
class In(Expr):
    region: Region
    child: Expr
