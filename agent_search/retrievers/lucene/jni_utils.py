"""Shared JNI/pyjnius plumbing for the `lucene` backend package.

Every module under `agent_search/retrievers/lucene/` needs the same handful of raw
Lucene classes (`Document`, `IndexWriter`, `SpanNearQuery`, ...) via pyjnius
`autoclass`. Pyserini's high-level `LuceneSearcher`/`LuceneIndexer` API (used by
`agent_search.retrievers.lexical.pyserini.BM25Pyserini`) only exposes plain-text
indexing and a fixed set of similarities; it has no fielded-document API and no span
queries. This module is the one place that imports pyjnius and constructs the JVM
class handles, so `index_builder.py`, `indri_compiler.py`, `bql_compiler.py` and
`engine.py` share one lazily initialized, cached set of class refs instead of
separate `autoclass(...)` calls with separate chances to typo a class path.

This module never imports `agent_search/retrievers/lexical/pyserini.py` directly; it
only calls into pyserini for `pyserini.pyclass.autoclass` (the same import path
pyserini itself uses internally to reach pyjnius) and to start the shared JVM.
Pyserini's autoclass call has the side effect of booting the JVM, and pyjnius
allows exactly one JVM per process, so any module in this codebase that needs raw
Lucene classes must go through this same boot path rather than starting a second,
incompatible one.

JVM boot order matters: a bare `import jnius` anywhere in the process, before
`pyserini.pyclass` gets a chance to run its classpath configuration, starts the JVM
with no classpath in this environment (pyjnius auto-starts on module import here, it
does not wait for the first `autoclass()` call). Every subsequent
`autoclass('org.apache.lucene...')` then fails with `NoClassDefFoundError`, even
though `java.lang.String`-style JDK classes still resolve fine
(`tests/test_lucene_structured.py`'s JVM-boot guard calls `_boot()` directly rather
than `pytest.importorskip("jnius", ...)` for this reason). This module's `_boot()`
is classpath-safe: it goes through `pyserini.pyclass`, which configures the
classpath to pyserini's bundled fat jar. The risk is only a different module
bare-importing `jnius` first; grep for `import jnius` before adding one. A future
module that needs raw Lucene classes should call
`agent_search.retrievers.lucene.jni_utils._boot()` (or import this module) before
its own `import jnius`.

Lucene version note: this environment ships Lucene 9, where the span-query classes
moved package from `org.apache.lucene.search.spans` (Lucene <9) to
`org.apache.lucene.queries.spans` (Lucene 9+); getting this wrong makes every span
class 404 with a `ClassNotFoundException` that looks like a missing dependency, not
a wrong import path. Nested static classes (`Field.Store`, `SpanNearQuery.Builder`,
...) need the JVM's `$`-separated inner-class name (`Field$Store`), not the Python
dotted form.
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

    Same convention as `agent_search.retrievers.lexical.pyserini._silence_fd`, defined
    separately here since this module has no dependency on that one. Used only around
    JVM boot to swallow the benign one-time 'WARNING: Using incubator modules:
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
        # pyserini.search.lucene transitively imports an OpenAI client constructor
        # at import time; this module only imports pyserini.pyclass (not
        # pyserini.search.lucene), so that import-time dependency does not apply
        # here, but the placeholder is set and restored anyway in case a caller
        # imports both in one process. Only set if absent, and removed again in
        # `finally`, the same set-and-restore contract as `pyserini.py`'s
        # `_openai_placeholder_env`: a JVM boot must never leave a fake key sitting
        # in the host process's environment after the one import that needed it
        # returns.
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
    "TermInSetQuery": "org.apache.lucene.search.TermInSetQuery",
    "BytesRef": "org.apache.lucene.util.BytesRef",
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
    (implicitly, by IndexWriter) and at query compile time (explicitly, here) so a
    query's term forms always match what got indexed for that field. A mismatch here
    is the most common source of Lucene "0 hits, both trivially correct in isolation"
    bugs.
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
# EnglishAnalyzer: Porter stemming plus English stopwords, the same analyzer family
# `bm25_pyserini` indexes with (see agent_search/retrievers/lexical/pyserini.py's
# module docstring). Used for `body`/`title`/`section` (the scored fields).
#
# StandardAnalyzer with no stop words: StandardTokenizer (Unicode word boundaries, so letters
# and digits both form tokens) plus LowerCaseFilter, no stemming. Used for the `*_exact` fields
# (span/window ops and exact boolean "matches" tests) as the closest built-in Lucene analyzer to
# the Python reference's word-splitting, unstemmed tokenization. Its predecessor here,
# SimpleAnalyzer, tokenizes letter runs only and silently dropped every number, so a window such
# as #uw5(treaty 1848) had no term for 1848 and every year-bearing window failed (index schema 2).
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
        a = J("StandardAnalyzer")()                  # Lucene 9: the default stop set is empty
        _analyzers["exact"] = a
    return a


def stemmed_tokens(text: str, field: str = "body") -> list:
    return analyze(stemmed_analyzer(), field, text or "")


def exact_tokens(text: str, field: str = "body_exact") -> list:
    return analyze(exact_analyzer(), field, text or "")
