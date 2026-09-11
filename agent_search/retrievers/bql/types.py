"""Granularity type checking for BQL.

Rejects ill-formed queries before execution (bql_spec.md §3). Term/Phrase/Prefix/
Expand are TOKEN-level; NEAR/g and IN(region,.) lift to level g / the region's
level; AND/OR over children lift to the coarser shared level. The checker also
enforces that NOT only appears as a set-difference clause inside AND.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_search.retrievers.bql.ast import (
    And, Expand, Expr, Granularity, In, Near, Not, Or, Phrase, Prefix, Region, Term,
)


@dataclass
class TypeResult:
    ok: bool
    level: Granularity | None = None
    error: str | None = None         # human-readable; surfaced in the observation
    error_node: str | None = None    # which clause is ill-typed


class _TypeError(Exception):
    def __init__(self, msg: str, node: str | None = None):
        super().__init__(msg)
        self.node = node


# region -> binding-region granularity (bql_spec.md §3, §5)
_REGION_LEVEL: dict[Region, Granularity] = {
    Region.TITLE: Granularity.DOC,
    Region.BODY: Granularity.DOC,
    Region.SECTION: Granularity.DOC,
    Region.AUTHOR: Granularity.DOC,
    Region.DATE: Granularity.DOC,
    Region.INFOBOX: Granularity.DOC,
    Region.DOC: Granularity.DOC,
    Region.COMMENT: Granularity.BLOCK,
    Region.STRING: Granularity.BLOCK,
    Region.SIG: Granularity.BLOCK,
    Region.DEF: Granularity.FUNC,
    Region.CALL: Granularity.FUNC,
    Region.FILE: Granularity.FILE,
}

# NEAR/<spec> -> granularity level. `spec` is the raw token, e.g. "w5", "func".
_GRAN_PREFIX_LEVEL: dict[str, Granularity] = {
    "w": Granularity.LINE,      # token window: co-occurrence treated at line level
    "line": Granularity.LINE,
    "sent": Granularity.LINE,
    "para": Granularity.BLOCK,
    "block": Granularity.BLOCK,
    "func": Granularity.FUNC,
    "file": Granularity.FILE,
}


def gran_level(spec: str) -> Granularity:
    """Map a NEAR granularity spec ('w5', 'line3', 'func', ...) to a lattice level."""
    alpha = "".join(c for c in spec if c.isalpha()).lower()
    if not alpha:
        alpha = "w"                 # bare numeric spec (NEAR/5) = token window
    if alpha not in _GRAN_PREFIX_LEVEL:
        raise _TypeError(f"unknown NEAR granularity {spec!r}", node="NEAR")
    return _GRAN_PREFIX_LEVEL[alpha]


def _coarser(a: Granularity, b: Granularity) -> Granularity:
    return a if a.value >= b.value else b


def _infer(expr: Expr) -> Granularity:
    """Infer the granularity level; raise _TypeError on ill-formed queries.

    NOT is handled only as a child of AND (below); reaching _infer(Not) means a NOT
    appeared somewhere it isn't allowed.
    """
    if isinstance(expr, (Term, Phrase, Prefix, Expand)):
        return Granularity.TOKEN

    if isinstance(expr, Not):
        raise _TypeError("NOT is only allowed as a clause of AND", node="NOT")

    if isinstance(expr, And):
        positives = [c for c in expr.children if not isinstance(c, Not)]
        negatives = [c for c in expr.children if isinstance(c, Not)]
        if not positives:
            raise _TypeError("AND needs at least one positive clause", node="AND")
        for neg in negatives:
            _infer(neg.child)  # validate the negated sub-expression
        level = Granularity.TOKEN
        for c in positives:
            level = _coarser(level, _infer(c))
        return level

    if isinstance(expr, Or):
        level = Granularity.TOKEN
        for c in expr.children:
            level = _coarser(level, _infer(c))  # _infer(Not) raises -> NOT banned in OR
        return level

    if isinstance(expr, Near):
        bind = gran_level(expr.spec)
        child = _coarser(_infer(expr.left), _infer(expr.right))
        if child.value > bind.value:
            # lattice rule: a binder cannot scope a coarser pattern:
            # NEAR/w5(NEAR/file(a,b), c) is ill-formed, not silently LINE-level
            raise _TypeError(
                f"NEAR/{expr.spec} cannot bind a {child.name}-level operand",
                node="NEAR",
            )
        return bind

    if isinstance(expr, In):
        region_level = _REGION_LEVEL[expr.region]
        child_level = _infer(expr.child)
        if child_level.value > region_level.value:
            raise _TypeError(
                f"IN({expr.region.value}, ...) cannot bind a "
                f"{child_level.name}-level pattern",
                node="IN",
            )
        return region_level

    raise _TypeError(f"unknown expression node: {type(expr).__name__}")


def check(expr: Expr) -> TypeResult:
    """Assign a granularity level or return a structured error.

    Well-typed -> TypeResult(ok=True, level=...). Ill-typed -> ok=False with a
    reason and the offending node; such a query is never executed and its error
    becomes agent observation / RL signal.
    """
    try:
        level = _infer(expr)
        return TypeResult(ok=True, level=level)
    except _TypeError as e:
        return TypeResult(ok=False, error=str(e), error_node=e.node)
    except RecursionError:
        return TypeResult(ok=False, error="query too deeply nested")
