"""Real AST structural scoping for a code unit.

`IN(region, x)` is the operator that distinguishes SkimSearchAgent from plain grep: it
restricts a match to a *structural region* (a definition, a call site, a comment, a
string literal, a signature). This module extracts, per unit, the identifier tokens
that occur in each region, so `IN(call, foo)` matches only where `foo` is called,
not merely mentioned. Pure stdlib (`ast` + `tokenize`); the ast-grep backend is the
faster cross-language equivalent.
"""
from __future__ import annotations

import ast
import io
import textwrap
import tokenize as _tok

from agent_search.corpus.units import code_tokenize

REGIONS = ("def", "call", "string", "comment", "sig")


def _call_name(func: ast.AST):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _func_args(node):
    a = node.args
    out = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
    if a.vararg:
        out.append(a.vararg)
    if a.kwarg:
        out.append(a.kwarg)
    return out


def _joinedstr_text(node) -> str:
    """Concatenate every literal string part of an f-string (ast.JoinedStr),
    recursing into nested format-spec JoinedStr nodes. Explicit traversal so
    f-string literals are collected reliably on any CPython >= 3.8 rather than
    depending on ast.walk yielding the JoinedStr's nested ast.Constant children
    (their reachability/representation has shifted across versions, and on the
    cluster runtime f-string text was silently dropped from the `string` bag)."""
    parts: list = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.JoinedStr):          # nested (rare)
            parts.append(_joinedstr_text(value))
        elif isinstance(value, ast.FormattedValue):
            if value.format_spec is not None:           # literal text in `:spec`
                parts.append(_joinedstr_text(value.format_spec))
    return " ".join(parts)


def region_token_bags(code: str) -> dict:
    """Map each structural region -> the list of identifier tokens occurring there,
    within a single unit's source. Adjacency within a name is preserved (so
    multi-token terms like `make_token` can require contiguity)."""
    bags: dict = {r: [] for r in REGIONS}
    dedented = textwrap.dedent(code)

    tree = None
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        # dedent fails when the unit contains a multiline string with column-0
        # content (common: docstrings/templates): the def stays indented and
        # ast.parse raises. Re-parse inside a synthetic class wrapper, which
        # accepts any consistent leading indent; skip the wrapper in the walk.
        wrapped = "class _BoolagentWrap_:\n" + (
            code if code[:1] in (" ", "\t") else textwrap.indent(code, "    "))
        try:
            tree = ast.parse(wrapped)
        except SyntaxError:
            tree = None

    if tree is not None:
        # Pre-collect nodes that live inside an f-string so the generic branches
        # below don't double-count them: _joinedstr_text already harvests an
        # f-string's nested Constants and its nested (format-spec) JoinedStr text,
        # so only the outermost JoinedStr should be processed by the main walk.
        fstring_const_ids: set = set()
        nested_joinedstr_ids: set = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.JoinedStr):
                for sub in ast.walk(n):
                    if sub is n:
                        continue
                    if isinstance(sub, ast.Constant):
                        fstring_const_ids.add(id(sub))
                    elif isinstance(sub, ast.JoinedStr):
                        nested_joinedstr_ids.add(id(sub))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "_BoolagentWrap_":
                continue
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    bags["call"] += code_tokenize(name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bags["def"] += code_tokenize(node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = _func_args(node)
                    for arg in args:
                        bags["def"] += code_tokenize(arg.arg)
                    # signature = name + params (+ annotations) + return annotation
                    bags["sig"] += code_tokenize(node.name)
                    for arg in args:
                        bags["sig"] += code_tokenize(arg.arg)
                        if arg.annotation is not None:
                            bags["sig"] += code_tokenize(ast.unparse(arg.annotation))
                    if node.returns is not None:
                        bags["sig"] += code_tokenize(ast.unparse(node.returns))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bags["def"] += code_tokenize(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                bags["def"] += code_tokenize(node.attr)   # self.x = ... defines x
            elif isinstance(node, ast.JoinedStr):
                if id(node) not in nested_joinedstr_ids:    # outermost only
                    bags["string"] += code_tokenize(_joinedstr_text(node))
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and id(node) not in fstring_const_ids):
                bags["string"] += code_tokenize(node.value)

    try:
        for t in _tok.generate_tokens(io.StringIO(dedented).readline):
            if t.type == _tok.COMMENT:
                bags["comment"] += code_tokenize(t.string)
    except (_tok.TokenError, IndentationError, ValueError):
        pass

    return bags
