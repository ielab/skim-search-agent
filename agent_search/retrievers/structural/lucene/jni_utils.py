"""Shared JNI/pyjnius plumbing for the `lucene` backend package.

Every module under `agent_search/retrievers/structural/lucene/` needs the SAME
handful of raw Lucene classes (`Document`, `IndexWriter`, `SpanNearQuery`, ...) via
pyjnius `autoclass` — pyserini's high-level `LuceneSearcher`/`LuceneIndexer` API
(used by `agent_search.retrievers.lexical.pyserini.BM25Pyserini`, which another
agent is concurrently hardening) only exposes plain-text indexing and a fixed set of
similarities; it has no fielded-document API and no span queries. This module is the
ONE place that imports pyjnius and constructs the JVM class handles, so
`index_builder.py` / `indri_compiler.py` / `bql_compiler.py` / `engine.py` all share
one lazily-initialized, cached set of class refs instead of five copies of the same
`autoclass(...)` calls (and five chances to typo a class path).

Deliberately does NOT touch `agent_search/retrievers/lexical/pyserini.py` (per task
scope: that module is being hardened concurrently by another agent) — this package
is purely ADDITIVE and only ever calls into pyserini for `pyserini.pyclass.autoclass`
(the same import path pyserini itself uses internally to reach pyjnius) and to start
the shared JVM (pyserini's autoclass call has the side effect of booting the JVM;
pyjnius allows exactly ONE JVM per process, so any module in this codebase that needs
raw Lucene classes MUST go through this same boot path rather than starting a second,
incompatible one).

**JVM-boot landmine** (found while validating this package under pytest): a bare
`import jnius` -- ANYWHERE in the process, before `pyserini.pyclass` gets a chance
to run its classpath configuration -- starts the JVM with NO classpath in this
environment (pyjnius auto-starts on module import here, it does not wait for the
first `autoclass()` call). Every subsequent `autoclass('org.apache.lucene...')`
then fails with `NoClassDefFoundError`, even though `java.lang.String`-style JDK
classes still resolve fine (confirmed by reproducing it outside pytest: see
`tests/test_lucene_structured.py`'s JVM-boot guard, which calls `_boot()` directly
rather than `pytest.importorskip("jnius", ...)` for exactly this reason). This
module's `_boot()` is classpath-safe (it goes through `pyserini.pyclass`, which
configures the classpath to pyserini's bundled fat jar); the risk is only a
DIFFERENT module bare-importing `jnius` first. No other module in this codebase
does that today (grep for `import jnius` before adding one) -- if a future one
needs to, it should call `agent_search.retrievers.structural.lucene.jni_utils._boot()`
(or import this module) BEFORE its own `import jnius`.

Lucene version note (see task prereqs): this environment ships Lucene 9, where the
span-query classes moved package from `org.apache.lucene.search.spans` (Lucene <9) to
`org.apache.lucene.queries.spans` (Lucene 9+) — get this wrong and every span class
404s with a ClassNotFoundException that looks like a missing dependency, not a wrong
import path. Nested static classes (`Field.Store`, `SpanNearQuery.Builder`, ...) need
the JVM's `$`-separated inner-class name (`Field$Store`), not the Python dotted form.
"""
from __future__ import annotations

import contextlib
import os
import threading
from typing import Optional

_lock = threading.Lock()
_classes: dict = {}
_booted = False


@contextlib.contextmanager
def silence_fd(fd: int = 2):
    """Temporarily redirect a file descriptor (default stderr) to /dev/null.

    Copied convention from `agent_search.retrievers.lexical.pyserini._silence_fd`
    (not imported — see module docstring: no dependency on that module) — used only
    around JVM boot to swallow the benign one-time 'WARNING: Using incubator modules:
    jdk.incubator.vector' the JVM prints. A real init failure still raises a Python
    exception (jnius surfaces it as an exception, not stderr noise), so nothing that
    matters is hidden."""
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def _boot() -> None:
    """Boot the (process-global, singleton) JVM via pyserini's own autoclass path,
    exactly once. Safe to call redundantly (idempotent, lock-guarded)."""
    global _booted
    if _booted:
        return
    with _lock:
        if _booted:
            return
        # See module docstring's "Offline safety" caveat carried over from
        # pyserini.py: pyserini.search.lucene transitively imports an OpenAI client
        # constructor at IMPORT time. We only import pyserini.pyclass here (not
        # pyserini.search.lucene), so that landmine doesn't apply to this module —
        # but set-and-restore the placeholder anyway in case a caller imports both in
        # one process. Only set if absent, and removed again in `finally` (same
        # set-and-restore contract as `pyserini.py`'s `_openai_placeholder_env` — a
        # JVM boot must never leave a fake key sitting in the host process's
        # environment after the one import that needed it returns).
        had_key = "OPENAI_API_KEY" in os.environ
        if not had_key:
            os.environ["OPENAI_API_KEY"] = "agent-search-unused-placeholder"
        try:
            with silence_fd(2):
                from pyserini.pyclass import autoclass  # noqa: F401  (side effect: boots JVM)
        finally:
            if not had_key:
                os.environ.pop("OPENAI_API_KEY", None)
        _booted = True


# --- the fixed set of Lucene/JDK classes this backend uses ------------------------
# name -> fully qualified JVM class path (nested classes use '$').
_CLASS_PATHS = {
    # java.nio / lucene store
    "Paths": "java.nio.file.Paths",
    "FSDirectory": "org.apache.lucene.store.FSDirectory",
    "MMapDirectory": "org.apache.lucene.store.MMapDirectory",
    # documents / fields
    "Document": "org.apache.lucene.document.Document",
    "Field": "org.apache.lucene.document.Field",
    "FieldStore": "org.apache.lucene.document.Field$Store",
    "FieldType": "org.apache.lucene.document.FieldType",
    "TextField": "org.apache.lucene.document.TextField",
    "StringField": "org.apache.lucene.document.StringField",
    "StoredField": "org.apache.lucene.document.StoredField",
    "IndexOptions": "org.apache.lucene.index.IndexOptions",
    # indexing
    "IndexWriter": "org.apache.lucene.index.IndexWriter",
    "IndexWriterConfig": "org.apache.lucene.index.IndexWriterConfig",
    "OpenMode": "org.apache.lucene.index.IndexWriterConfig$OpenMode",
    "DirectoryReader": "org.apache.lucene.index.DirectoryReader",
    "IndexReader": "org.apache.lucene.index.IndexReader",
    "Term": "org.apache.lucene.index.Term",
    # analysis
    "EnglishAnalyzer": "org.apache.lucene.analysis.en.EnglishAnalyzer",
    "SimpleAnalyzer": "org.apache.lucene.analysis.core.SimpleAnalyzer",
    "StandardAnalyzer": "org.apache.lucene.analysis.standard.StandardAnalyzer",
    "CharTermAttribute": "org.apache.lucene.analysis.tokenattributes.CharTermAttribute",
    # search
    "IndexSearcher": "org.apache.lucene.search.IndexSearcher",
    "TermQuery": "org.apache.lucene.search.TermQuery",
    "PrefixQuery": "org.apache.lucene.search.PrefixQuery",
    "PhraseQuery": "org.apache.lucene.search.PhraseQuery",
    "PhraseQueryBuilder": "org.apache.lucene.search.PhraseQuery$Builder",
    "BooleanQuery": "org.apache.lucene.search.BooleanQuery",
    "BooleanQueryBuilder": "org.apache.lucene.search.BooleanQuery$Builder",
    "BooleanClause": "org.apache.lucene.search.BooleanClause",
    "Occur": "org.apache.lucene.search.BooleanClause$Occur",
    "TermRangeQuery": "org.apache.lucene.search.TermRangeQuery",
    "BoostQuery": "org.apache.lucene.search.BoostQuery",
    "ConstantScoreQuery": "org.apache.lucene.search.ConstantScoreQuery",
    "DisjunctionMaxQuery": "org.apache.lucene.search.DisjunctionMaxQuery",
    "SynonymQuery": "org.apache.lucene.search.SynonymQuery",
    "SynonymQueryBuilder": "org.apache.lucene.search.SynonymQuery$Builder",
    "MatchNoDocsQuery": "org.apache.lucene.search.MatchNoDocsQuery",
    "MatchAllDocsQuery": "org.apache.lucene.search.MatchAllDocsQuery",
    # spans -- Lucene 9 package (see module docstring)
    "SpanTermQuery": "org.apache.lucene.queries.spans.SpanTermQuery",
    "SpanNearQuery": "org.apache.lucene.queries.spans.SpanNearQuery",
    "SpanNearQueryBuilder": "org.apache.lucene.queries.spans.SpanNearQuery$Builder",
    "SpanOrQuery": "org.apache.lucene.queries.spans.SpanOrQuery",
    "SpanQuery": "org.apache.lucene.queries.spans.SpanQuery",
    "SpanMultiTermQueryWrapper": "org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper",
    # similarities
    "LMDirichletSimilarity": "org.apache.lucene.search.similarities.LMDirichletSimilarity",
    "BM25Similarity": "org.apache.lucene.search.similarities.BM25Similarity",
    "PerFieldSimilarityWrapper": "org.apache.lucene.search.similarities.PerFieldSimilarityWrapper",
    "PerFieldAnalyzerWrapper": "org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper",
    "HashMap": "java.util.HashMap",
    "ArrayList": "java.util.ArrayList",
}


def J(name: str):
    """Return the (cached) JVM class handle for a name registered in `_CLASS_PATHS`."""
    _boot()
    cls = _classes.get(name)
    if cls is None:
        from pyserini.pyclass import autoclass
        path = _CLASS_PATHS[name]
        cls = autoclass(path)
        _classes[name] = cls
    return cls


def jcast(iface_name: str, obj):
    """`jnius.cast` to a named JVM interface/class (e.g. `DirectoryReader` ->
    `IndexReader` for `IndexSearcher`'s constructor)."""
    _boot()
    from jnius import cast
    return cast(_CLASS_PATHS[iface_name], obj)


# --- token extraction: run text through a Lucene Analyzer, Python-side -----------

def analyze(analyzer, field: str, text: str) -> list:
    """Tokenize `text` through a live Lucene `Analyzer` instance for `field`,
    returning the resulting term strings in order. Used identically at index time
    (implicitly, by IndexWriter) and at QUERY compile time (explicitly, here) so a
    query's term forms always match what got indexed for that field -- the single
    most common source of Lucene "0 hits, both trivially correct in isolation" bugs.
    """
    _boot()
    CharTermAttribute = J("CharTermAttribute")
    out: list = []
    ts = analyzer.tokenStream(field, text)
    try:
        attr = ts.addAttribute(CharTermAttribute)
        ts.reset()
        while ts.incrementToken():
            out.append(attr.toString())
        ts.end()
    finally:
        ts.close()
    return out


# --- shared analyzer instances (stateless, safe to reuse across threads) ---------
# EnglishAnalyzer: Porter stemming + English stopwords -- the SAME analyzer family
# `bm25_pyserini` indexes with (see agent_search/retrievers/lexical/pyserini.py's
# module docstring: "same analyzer (Porter stemming...)"). Used for `body`/`title`/
# `section` (the scored fields).
#
# SimpleAnalyzer: LetterTokenizer + LowerCaseFilter -- lowercases and splits on any
# non-letter run, NO stemming, NO stopword removal. Used for the `*_exact` fields
# (span/window ops + exact boolean "matches" tests) as the closest built-in Lucene
# analyzer to the Python reference's `code_tokenize` (word-splitting, unstemmed) --
# it does NOT split camelCase/snake_case identifiers the way `code_tokenize` does
# (a documented deviation; low-impact for this prose document corpus, see
# `indri_compiler.py`'s module docstring).
_analyzers: dict = {}


def stemmed_analyzer():
    a = _analyzers.get("stemmed")
    if a is None:
        a = J("EnglishAnalyzer")()
        _analyzers["stemmed"] = a
    return a


def exact_analyzer():
    a = _analyzers.get("exact")
    if a is None:
        a = J("SimpleAnalyzer")()
        _analyzers["exact"] = a
    return a


def stemmed_tokens(text: str, field: str = "body") -> list:
    return analyze(stemmed_analyzer(), field, text or "")


def exact_tokens(text: str, field: str = "body_exact") -> list:
    return analyze(exact_analyzer(), field, text or "")
