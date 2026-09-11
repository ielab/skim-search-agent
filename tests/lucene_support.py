"""Builders for the engines a document corpus needs in a test: Lucene BM25 (Pyserini) and
the Lucene structured index behind BQL and Indri. Document corpora have no in-memory engine
in the library, so every document-side test goes through these, and skips without a JVM.

Indexes are built once per distinct corpus (keyed by the corpus fingerprint) under one
session-wide temporary root, so tests that share a corpus share an index.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import pytest

from agent_search.corpus.fingerprint import corpus_fingerprint

_ROOT: str | None = None


def jvm_available() -> bool:
    """True when pyserini's JVM boots with the Lucene classpath. Goes through
    `jni_utils._boot()` (never a bare `import jnius`, which starts a JVM without a classpath)."""
    try:
        from agent_search.retrievers.lucene import jni_utils
        jni_utils._boot()
        return True
    except Exception:  # noqa: BLE001 - environment-dependent
        return False


def require_jvm() -> None:
    """Skip when no JVM is available: the whole module when called at import time, the one
    test when called inside a test function."""
    if not jvm_available():
        import inspect
        at_import = any(f.function == "<module>" for f in inspect.stack()[1:3])
        pytest.skip("needs a JVM (pip install -e '.[retrieval]' and JAVA_HOME at a JDK 21)",
                    allow_module_level=at_import)


def index_root() -> str:
    global _ROOT
    if _ROOT is None:
        _ROOT = tempfile.mkdtemp(prefix="ssa_test_indexes_")
        atexit.register(shutil.rmtree, _ROOT, True)
    return _ROOT


def corpus_key(units) -> str:
    return "t_" + corpus_fingerprint(units)[:16]


def build_pyserini(units):
    """A `BM25Pyserini` over `units`, built once per corpus."""
    from agent_search.retrievers.lexical.pyserini import BM25Pyserini
    units = list(units)
    return BM25Pyserini(index_root=index_root()).index(units, key=corpus_key(units))


def build_structured_index(units) -> str:
    """Build (once) the Lucene structured index for `units`; returns its corpus key."""
    from agent_search.retrievers.lucene import index_builder
    units = list(units)
    key = corpus_key(units)
    index_builder.build(units, index_root(), key, rebuild=False, progress=False)
    return key


def build_lucene_bql(units, dense=None):
    """The document BQL engine (`LuceneBqlAdapter`) over `units`."""
    from agent_search.retrievers.backend import build_bql_engine
    return build_bql_engine(list(units), index_root(), build_structured_index(units), dense=dense,
                            domain="general")


def build_lucene_bql_dense_only(units, dense=None):
    from agent_search.retrievers.backend import build_bql_engine_dense_only
    return build_bql_engine_dense_only(list(units), index_root(), build_structured_index(units),
                                       dense=dense, domain="general")


def build_lucene_indri(units, dense=None):
    """The Indri engine (`LuceneIndriAdapter`) over `units`."""
    from agent_search.retrievers.backend import build_indri_engine
    return build_indri_engine(list(units), index_root(), build_structured_index(units), dense=dense,
                              domain="general")


def build_engines(units, key=None, **kw):
    """An `Engines` registry over a document corpus whose Lucene indexes are already built
    under the test root, so `engines.get("bm25")`, `("bql")` and `("indri")` all resolve."""
    from agent_search.retrievers.engines import Engines
    units = list(units)
    k = key or corpus_key(units)
    build_structured_index(units)
    from agent_search.retrievers.lexical.pyserini import BM25Pyserini
    BM25Pyserini(index_root=index_root()).index(units, key=k)
    if k != corpus_key(units):
        from agent_search.retrievers.lucene import index_builder
        index_builder.build(units, index_root(), k, rebuild=False, progress=False)
    return Engines(units, k, index_root=index_root(), domain=kw.pop("domain", "general"), **kw)


__all__ = ["jvm_available", "require_jvm", "index_root", "corpus_key", "build_pyserini",
           "build_structured_index", "build_lucene_bql", "build_lucene_bql_dense_only",
           "build_lucene_indri", "build_engines"]
