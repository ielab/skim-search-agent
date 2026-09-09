"""Core shared primitives: code units + tokenization.

Both the method (index-free structural executor) and the baselines (BM25/dense)
chunk live files into function-level units and tokenize them. This lives at the top
level so neither layer depends on the other (method and eval both depend on `units`).

Chunking is done on the *live* files at query time — there is no persisted index.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


def is_test_path(path: str) -> bool:
    """SWE-bench's test-file heuristic (bm25_retrieval.py is_test / CoRNStack):
    split the path on separators and drop it if any word is test/tests/testing.
    Applied uniformly to the corpus of EVERY condition so test exclusion can
    never confound the method-vs-baseline comparison."""
    words = set(re.split(r"[_\-/\.]", path.lower()))
    return bool(words & {"test", "tests", "testing"})


@dataclass(frozen=True)
class CodeUnit:
    doc_id: str        # f"{path}::{qualname}"
    path: str
    qualname: str      # dotted, e.g. "ClassName.method"
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    code: str
    title: str | None = None
    body: str | None = None
    section: str | None = None
    # doc arm: sections as a MATCHED, ordered list of (heading, text) parts. A structured corpus
    # ships them explicitly, so `fetch` reads a real slice and IN(section,·) scopes to a named part,
    # instead of re-deriving them from `##` markers in the body. None for code units and flat docs
    # (fetch then falls back to splitting the body on `##`). A tuple keeps CodeUnit hashable/frozen.
    sections: tuple | None = None
    metadata: Mapping[str, str] | None = None


def units_from_python_source(path: str, source: str) -> list[CodeUnit]:
    """Extract function/method units from a Python source string.

    Unparseable sources return [] (never raise) — real repos contain files that
    don't parse under our Python version.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    units: list[CodeUnit] = []
    seen: dict[str, int] = {}   # disambiguate duplicate qualnames (e.g. @overload)

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                start = child.lineno
                # include decorator lines in the span so a patch editing a
                # @decorator maps to its function (child.lineno is the `def` line)
                if child.decorator_list:
                    start = min(start, min(d.lineno for d in child.decorator_list))
                end = getattr(child, "end_lineno", child.lineno)
                code = "\n".join(lines[start - 1:end])
                base = f"{path}::{qual}"
                if base in seen:               # unique doc_id even on qualname clash
                    seen[base] += 1
                    doc_id = f"{base}#{seen[base]}"
                else:
                    seen[base] = 0
                    doc_id = base
                units.append(CodeUnit(doc_id, path, qual, start, end, code))
                visit(child, f"{qual}.")  # nested functions
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return units


def units_from_documents(docs: Sequence[Mapping[str, object]]) -> list[CodeUnit]:
    """Turn fixed-corpus document records into retrievable units.

    Expected keys are flexible: ``doc_id``/``id``/``_id`` for identity, plus optional
    ``title``, ``section`` and ``text``/``body``/``contents``. This supports
    BrowseComp-Plus-style corpora and BEIR/CoIR JSONL without introducing another
    retriever interface.
    """
    units: list[CodeUnit] = []
    seen_ids: set = set()
    for i, d in enumerate(docs):
        doc_id = str(d.get("doc_id") or d.get("id") or d.get("_id") or f"doc{i}")
        if doc_id in seen_ids:
            continue                     # a duplicate id would silently collide in the
        seen_ids.add(doc_id)             # dense/BM25 dict-keying (and skew ranking metrics)
        title = str(d.get("title") or "")
        section = str(d.get("section") or d.get("heading") or "")
        body = str(d.get("text") or d.get("body") or d.get("contents") or "")
        # A STRUCTURED corpus ships `sections` as a matched, ordered list of {heading, text} parts
        # (wikipedia's real `##` sections; browsecomp's LLM-inserted TOC sections). Keep them as
        # (heading, text) pairs on the unit; derive the joined-heading `section` field (for
        # IN(section,·) + the search listing) from them when a flat `section` string wasn't given.
        raw_sections = d.get("sections")
        sections = None
        if isinstance(raw_sections, (list, tuple)) and raw_sections:
            sections = tuple((str(s.get("heading") or ""), str(s.get("text") or ""))
                             for s in raw_sections if isinstance(s, dict))
            if not section:
                section = " ".join(h for h, _ in sections if h)
        if not (title or section or body or sections):
            continue
        # The bm25/dense searchable blob is `code` ALONE, and every engine indexes
        # `f"{qualname} {code}"` (qualname = title). So `code` is BODY ONLY here — NOT
        # title+body: title already reaches the index once via `qualname`, and folding it
        # into `code` too would count every title token TWICE (a TF advantage no other
        # unit kind gets — `units_from_python_source`'s `code` never repeats `qualname`
        # either). FAIRNESS for the flat-vs-structured pair: a structured doc carries
        # `section` (its joined headings) while its flat twin does not, but their
        # `text`/`body` is byte-identical and already contains those headings
        # (`## History` ...). Including the standalone `section` field here would
        # double-count heading tokens in the structured arm only, giving bm25/dense a TF
        # advantage the flat arm lacks — a confound. `section` stays on `u.section` for
        # BQL's IN(section,·) to scope; it just doesn't inflate the lexical/dense blob.
        code = body
        path = str(d.get("path") or d.get("url") or doc_id)
        units.append(CodeUnit(
            doc_id=doc_id,
            path=path,
            qualname=title or doc_id,
            start_line=1,
            end_line=max(1, len(code.splitlines())),
            code=code,
            title=title or None,
            body=body or None,
            section=section or None,
            sections=sections,
            metadata={str(k): str(v) for k, v in d.items()
                      if k not in {"doc_id", "id", "_id", "title", "section", "sections",
                                   "heading", "text", "body", "contents", "path", "url"}},
        ))
    return units


_WORD = re.compile(r"\w+", re.UNICODE)     # Unicode words (keeps café/münchen/δelta etc.)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+")


def code_tokenize(text: str) -> list[str]:
    """Lowercase identifier-aware tokenizer: splits snake_case and camelCase.

    ASCII words go through the camelCase splitter (byte-identical to before, so code
    tokenization is unchanged); non-ASCII words (document corpora in other languages)
    are kept WHOLE rather than fragmented by the ASCII-only camelCase regex."""
    out: list[str] = []
    for word in _WORD.findall(text):
        for part in word.split("_"):
            if part.isascii():
                out.extend(m.lower() for m in _CAMEL.findall(part))
            else:
                out.append(part.lower())   # keep a non-ASCII word intact
    return [t for t in out if t]
