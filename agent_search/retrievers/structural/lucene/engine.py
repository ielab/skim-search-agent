"""The searcher class the tool layer talks to: opens a prebuilt fielded Lucene index
(`index_builder.py`) and exposes a small, stable, THREAD-SAFE query API for both
compiled query languages (`indri_compiler.py`, `bql_compiler.py`).

Thread safety: a single `IndexSearcher`/`DirectoryReader` pair is safe to share
across concurrent callers -- this is standard Lucene practice (both classes are
documented thread-safe for reads) and matches the empirical finding already
recorded for `BM25Pyserini` (`agent_search.retrievers.lexical.pyserini`'s module
docstring: "500-doc index, 8 threads x 20 concurrent .search() calls ... zero
exceptions"). One `LuceneStructuredEngine` instance is built once per corpus and
reused across the agent's worker threads (default 8).

Offline safety: every path here is a LOCAL directory (`MMapDirectory(Paths.get(...))`)
-- never a network fetch, mirroring `pyserini.py`'s own "Offline safety" contract.

Memory-mapped, explicitly: opened via `MMapDirectory` (not the platform-dependent
`FSDirectory.open` default) on both the write side (`index_builder.py`) and here on
the read side, per the coordinator's efficiency requirement -- the OS page cache
then serves hot index pages across all 8 worker threads without a second copy into
the JVM heap.

Env knobs:
    LUCENE_MU        - LMDirichletSimilarity mu (default: INDRI_MU if set, else 2500 --
                        same default the Python `indri` reference uses, so a
                        side-by-side comparison starts from the SAME smoothing prior).
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field as dc_field
from typing import Optional

from agent_search.retrievers.structural.indri.parser import parse as indri_parse
from agent_search.retrievers.structural.lucene import bql_compiler, indri_compiler
from agent_search.retrievers.structural.lucene import jni_utils as J
from agent_search.retrievers.structural.lucene.index_builder import has_segments, index_dir
from agent_search.retrievers.structural.lucene.schema import EXACT_FIELDS, F_ID, STEMMED_FIELDS


def default_mu() -> float:
    return float(os.environ.get("LUCENE_MU", os.environ.get("INDRI_MU", "2500")))


@dataclass
class LuceneHit:
    """`title`/`date`/`author` are deliberately NOT here: the index is lean (only
    `id` is stored -- see `schema.py`'s "Storage: lean by design" note), and the
    tool layer already holds a `doc_id -> CodeUnit` corpus mapping to enrich a hit
    from, so this engine doesn't pay to store (and re-serve) a second copy."""
    doc_id: str
    score: float
    matched_fields: tuple = ()


@dataclass
class LuceneResult:
    hits: list = dc_field(default_factory=list)     # list[LuceneHit], best-first
    error: Optional[str] = None
    # LOW finding (adversarial verification): see `indri.model.IndriResult.warning`'s
    # docstring -- an unrecognized `.field` name silently resolves to Lucene's
    # filter-only "no scoring field" path (`indri_compiler._resolve_fields`) and
    # just returns 0/fewer hits with no error, indistinguishable from a genuinely
    # zero-hit query. Populated by `search_indri` via the SAME
    # `indri.model.unknown_query_fields` walk the python engine uses (both compile
    # the identical parsed AST), so the two engines warn identically.
    warning: Optional[str] = None


_FIELD_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in (*STEMMED_FIELDS, *EXACT_FIELDS)) + r")\s*:")


class LuceneStructuredEngine:
    """Open (lazily, once) the fielded Lucene index for one dataset/index dir, and
    answer Indri-QL / BQL queries against it. Not a `Retriever` subclass (the
    existing `Retriever.search(query, k) -> list[str]` interface is too narrow to
    carry scores/matched-field info the tool layer's listings need) -- see
    `search_indri`/`search_bql` below for the actual shape callers get."""

    def __init__(self, index_root: str = "indexes", dataset: Optional[str] = None,
                 index_path: Optional[str] = None, mu: Optional[float] = None):
        if index_path is None:
            if dataset is None:
                raise ValueError("LuceneStructuredEngine needs `dataset` or `index_path`")
            index_path = index_dir(index_root, dataset)
        self.index_path = index_path
        self.mu = float(mu) if mu is not None else default_mu()
        self._lock = threading.Lock()
        self._reader = None
        self._searcher_indri = None      # LMDirichletSimilarity(mu) -- immutable once set
        self._searcher_bql = None        # BM25Similarity(k1, b) -- immutable once set
        self._fsdir = None

    # --- lifecycle -----------------------------------------------------------
    # TWO IndexSearcher instances share ONE DirectoryReader (a Searcher's
    # constructor is a cheap wrapper, no I/O) rather than one searcher whose
    # Similarity gets swapped per query kind: `IndexSearcher.setSimilarity` mutates
    # shared state, and this engine is explicitly meant to be shared across worker
    # threads (module docstring) -- concurrent indri/bql calls flipping the SAME
    # searcher's similarity mid-search would be a real race (one thread's LMD
    # search silently rescored under the other thread's BM25 params, or vice
    # versa). Each searcher's Similarity is set ONCE at construction and never
    # touched again, so both are independently thread-safe for concurrent
    # `.search()` calls (the property this whole design exists to preserve).

    def _ensure_open(self) -> None:
        if self._searcher_indri is not None:
            return
        with self._lock:
            if self._searcher_indri is not None:
                return
            if not os.path.isdir(self.index_path):
                raise FileNotFoundError(
                    f"no lucene_structured index at {self.index_path!r}; build it with "
                    f"`python -m agent_search.retrievers.structural.lucene."
                    f"index_builder --dataset <name>`")
            MMapDirectory = J.J("MMapDirectory")
            Paths = J.J("Paths")
            DirectoryReader = J.J("DirectoryReader")
            IndexSearcher = J.J("IndexSearcher")
            LMDirichletSimilarity = J.J("LMDirichletSimilarity")
            BM25Similarity = J.J("BM25Similarity")
            self._fsdir = MMapDirectory(Paths.get(self.index_path))
            self._reader = DirectoryReader.open(self._fsdir)
            reader_iface = J.jcast("IndexReader", self._reader)
            s_indri = IndexSearcher(reader_iface)
            s_indri.setSimilarity(LMDirichletSimilarity(float(self.mu)))
            s_bql = IndexSearcher(reader_iface)
            s_bql.setSimilarity(BM25Similarity(0.9, 0.4))   # matches bm25_pyserini's defaults
            self._searcher_indri = s_indri
            self._searcher_bql = s_bql

    def close(self) -> None:
        with self._lock:
            for obj in (self._reader, self._fsdir):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
            self._reader = None
            self._searcher_indri = None
            self._searcher_bql = None
            self._fsdir = None

    def is_cached(self) -> bool:
        return has_segments(self.index_path)

    # --- indri ---------------------------------------------------------------

    def search_indri(self, query: str, k: int = 10) -> LuceneResult:
        r = indri_parse(query)
        if not r.ok:
            return LuceneResult(hits=[], error=str(r.error))
        from agent_search.retrievers.structural.indri.model import (
            _field_warning, unknown_query_fields)
        warning = _field_warning(unknown_query_fields(r.expr))
        try:
            self._ensure_open()
            q = indri_compiler.compile_score(r.expr)
            result = self._run(self._searcher_indri, q, k)
        except indri_compiler.LuceneCompileError as e:
            return LuceneResult(hits=[], error=str(e), warning=warning)
        except Exception as e:                      # never crash the agent loop
            return LuceneResult(hits=[], error=f"execution error: {e}", warning=warning)
        result.warning = warning
        return result

    # --- bql -------------------------------------------------------------------

    def search_bql(self, bql: str, k: int = 100) -> LuceneResult:
        from agent_search.retrievers.structural.bql.parser import parse as bql_parse
        from agent_search.retrievers.structural.bql.types import check as bql_check
        r = bql_parse(bql)
        if not r.ok:
            return LuceneResult(hits=[], error=f"parse error: {r.error}")
        t = bql_check(r.expr)
        if not t.ok:
            return LuceneResult(hits=[], error=f"type error: {t.error}")
        return self.search_bql_expr(r.expr, k)

    def search_bql_expr(self, expr, k: int = 100) -> LuceneResult:
        """Like `search_bql`, but takes an ALREADY parsed+typechecked BQL `Expr` -- skips
        the string round-trip. Used by `agent_search.retrievers.structural.lucene.adapters.
        LuceneBqlAdapter` (the STRUCTURED_BACKEND=lucene BQL surface): its caller
        (`execute_bql`, bql/executor.py) already parsed+typechecked the query string itself
        before calling the executor, so re-parsing here would be redundant work on every
        `search` tool call."""
        try:
            self._ensure_open()
            q = bql_compiler.compile_bql(expr)
            return self._run(self._searcher_bql, q, k)
        except bql_compiler.LuceneCompileError as e:
            return LuceneResult(hits=[], error=str(e))
        except Exception as e:
            return LuceneResult(hits=[], error=f"execution error: {e}")

    def count_bql_expr(self, expr) -> int:
        """EXACT total-match count for an already-compiled BQL `Expr` (`IndexSearcher.count`
        -- evaluates the query without collecting/scoring a top-k), so the
        STRUCTURED_BACKEND=lucene BQL adapter's search listing can show the SAME accurate
        "(N matches, top K)" header the python engine's `run_with_count` provides, instead of
        a truncated `len(hits)`."""
        self._ensure_open()
        q = bql_compiler.compile_bql(expr)
        return int(self._searcher_bql.count(q))

    # --- shared run/hydrate ------------------------------------------------------

    def _run(self, searcher, query, k: int) -> LuceneResult:
        top = searcher.search(query, int(k))
        sf = searcher.storedFields()          # only `id` is stored -- see LuceneHit
        hits = []
        for sd in top.scoreDocs:
            doc = sf.document(sd.doc)
            doc_id = doc.get(F_ID)
            matched = self._matched_fields(searcher, query, sd.doc)
            hits.append(LuceneHit(doc_id=doc_id, score=float(sd.score), matched_fields=matched))
        return LuceneResult(hits=hits, error=None)

    def _matched_fields(self, searcher, query, doc: int) -> tuple:
        """Best-effort matched-field extraction for listings: run `explain()` for
        this one hit and regex-scan its description text for our known field names.
        NOT authoritative (an Explanation's text format is a Lucene implementation
        detail, not a stable API) -- a display hint, never used for ranking/logic."""
        try:
            expl = searcher.explain(query, doc)
            text = expl.toString()
            return tuple(sorted(set(_FIELD_RE.findall(text))))
        except Exception:
            return ()


# --- module-level cache: one engine per index path, reused across callers --------
_ENGINES: dict = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(index_root: str = "indexes", dataset: Optional[str] = None,
               mu: Optional[float] = None) -> LuceneStructuredEngine:
    """Process-wide cached engine per (index_root, dataset) -- so N worker threads
    share one open `IndexSearcher` (see module docstring's thread-safety note)
    instead of each reopening the index."""
    key = (index_root, dataset, mu)
    eng = _ENGINES.get(key)
    if eng is None:
        with _ENGINES_LOCK:
            eng = _ENGINES.get(key)
            if eng is None:
                eng = LuceneStructuredEngine(index_root=index_root, dataset=dataset, mu=mu)
                _ENGINES[key] = eng
    return eng
