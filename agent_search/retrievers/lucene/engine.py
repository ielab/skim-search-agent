"""The searcher class the tool layer talks to: opens a prebuilt fielded Lucene index
(`index_builder.py`) and exposes a small, stable, thread-safe query API for both
compiled query languages (`indri_compiler.py`, `bql_compiler.py`).

Thread safety: a single `IndexSearcher`/`DirectoryReader` pair is safe to share
across concurrent callers. This is standard Lucene practice (both classes are
documented thread-safe for reads) and matches the same result already recorded for
`BM25Pyserini` (`agent_search.retrievers.lexical.pyserini`'s module docstring: a
500-doc index under 8 threads times 20 concurrent `.search()` calls raised zero
exceptions). One `LuceneStructuredEngine` instance is built once per corpus and
reused across the agent's worker threads (default 8).

Offline safety: every path here is a local directory (`MMapDirectory(Paths.get(...))`),
never a network fetch, matching `pyserini.py`'s own offline contract.

Memory-mapped explicitly: opened via `MMapDirectory` (not the platform-dependent
`FSDirectory.open` default) on both the write side (`index_builder.py`) and here on
the read side, so the OS page cache serves hot index pages across all worker threads
without a second copy into the JVM heap.

Env knobs:
    LUCENE_MU        - LMDirichletSimilarity mu (default: INDRI_MU if set, else 2500,
                        the same default the Python `indri` reference uses, so a
                        side-by-side comparison starts from the same smoothing prior).
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field as dc_field
from typing import Optional

from agent_search.retrievers.indri.parser import parse as indri_parse
from agent_search.retrievers.lucene import bql_compiler, indri_compiler
from agent_search.retrievers.lucene import jni_utils as J
from agent_search.retrievers.lucene.index_builder import has_segments, index_dir
from agent_search.retrievers.lucene.schema import EXACT_FIELDS, F_ID, STEMMED_FIELDS


def default_mu() -> float:
    return float(os.environ.get("LUCENE_MU", os.environ.get("INDRI_MU", "2500")))


@dataclass
class LuceneHit:
    """`title`/`date`/`author` are deliberately not here: the index is lean (only
    `id` is stored, see `schema.py`'s "Storage" note), and the tool layer already
    holds a `doc_id -> CodeUnit` corpus mapping to enrich a hit from, so this engine
    does not store (and re-serve) a second copy."""
    doc_id: str
    score: float
    matched_fields: tuple = ()


@dataclass
class LuceneResult:
    hits: list = dc_field(default_factory=list)     # list[LuceneHit], best-first
    error: Optional[str] = None
    # See `indri.result.IndriResult.warning`: an unrecognized `.field`
    # name silently resolves to Lucene's filter-only "no scoring field" path
    # (`indri_compiler._resolve_fields`) and just returns zero or fewer hits with no
    # error, indistinguishable from a genuinely zero-hit query. Populated by
    # `search_indri` via `indri.result.unknown_query_fields`.
    warning: Optional[str] = None


_FIELD_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in (*STEMMED_FIELDS, *EXACT_FIELDS)) + r")\s*:")


class LuceneStructuredEngine:
    """Open (lazily, once) the fielded Lucene index for one dataset/index dir, and
    answer Indri-QL / BQL queries against it. Not a `Retriever` subclass: the
    `Retriever.search(query, k) -> list[str]` interface is too narrow to carry the
    scores and matched-field info the tool layer's listings need. See
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
    # Two IndexSearcher instances share one DirectoryReader (a Searcher's
    # constructor is a cheap wrapper, no I/O) rather than one searcher whose
    # Similarity gets swapped per query kind. `IndexSearcher.setSimilarity` mutates
    # shared state, and this engine is meant to be shared across worker threads
    # (module docstring), so concurrent indri/bql calls flipping the same searcher's
    # similarity mid-search would be a real race: one thread's LMD search could get
    # rescored under the other thread's BM25 params, or vice versa. Each searcher's
    # Similarity is set once at construction and never touched again, so both are
    # independently thread-safe for concurrent `.search()` calls.

    def _ensure_open(self) -> None:
        if self._searcher_indri is not None:
            return
        with self._lock:
            if self._searcher_indri is not None:
                return
            if not os.path.isdir(self.index_path):
                raise FileNotFoundError(
                    f"no lucene_structured index at {self.index_path!r}; build it with "
                    f"`python -m agent_search.retrievers.lucene."
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
        from agent_search.retrievers.indri.result import field_warning, unknown_query_fields
        warning = field_warning(unknown_query_fields(r.expr))
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
        from agent_search.retrievers.bql.parser import parse as bql_parse
        from agent_search.retrievers.bql.types import check as bql_check
        r = bql_parse(bql)
        if not r.ok:
            return LuceneResult(hits=[], error=f"parse error: {r.error}")
        t = bql_check(r.expr)
        if not t.ok:
            return LuceneResult(hits=[], error=f"type error: {t.error}")
        return self.search_bql_expr(r.expr, k)

    def search_bql_expr(self, expr, k: int = 100) -> LuceneResult:
        """Like `search_bql`, but takes an already parsed and typechecked BQL `Expr`,
        skipping the string round-trip. Used by `agent_search.retrievers.lucene.adapters.
        LuceneBqlAdapter` (the document BQL engine): its caller
        (`execute_bql`, bql/executor.py) already parsed and typechecked the query
        string before calling the executor, so re-parsing here would be redundant
        work on every `search` tool call."""
        try:
            self._ensure_open()
            q = bql_compiler.compile_bql(expr)
            return self._run(self._searcher_bql, q, k)
        except bql_compiler.LuceneCompileError as e:
            return LuceneResult(hits=[], error=str(e))
        except Exception as e:
            return LuceneResult(hits=[], error=f"execution error: {e}")

    def rank_by_terms(self, expr, k: int = 100) -> LuceneResult:
        """Rank the whole index by the BQL scoring clause alone (`bql_compiler.compile_score`:
        BM25 over the query's positive leaf terms on body and title), with no boolean filter.
        A document with none of the terms is not returned. This is the ranking behind the BQL
        adapter's zero-hit fallback (`LuceneBqlAdapter.soft_topk`) and the candidate pool of
        its coverage ranking."""
        try:
            self._ensure_open()
            q = bql_compiler.compile_score(expr)
            return self._run(self._searcher_bql, q, k)
        except bql_compiler.LuceneCompileError as e:
            return LuceneResult(hits=[], error=str(e))
        except Exception as e:
            return LuceneResult(hits=[], error=f"execution error: {e}")

    def match_ids(self, expr, doc_ids) -> set:
        """The subset of `doc_ids` whose documents match `expr` exactly
        (`bql_compiler.compile_exact`). One boolean query: the exact clause as MUST, the id
        set as a FILTER (`TermInSetQuery` on the stored `id` field). The BQL adapter's coverage
        ranking calls this once per AND child over its candidate pool, which is how it learns
        which constraints each candidate satisfies without a Python executor."""
        ids = [d for d in doc_ids if d]
        if not ids:
            return set()
        self._ensure_open()
        BytesRef, ArrayList = J.J("BytesRef"), J.J("ArrayList")
        refs = ArrayList()
        for d in ids:
            refs.add(BytesRef(d.encode("utf-8")))
        in_set = J.J("TermInSetQuery")(F_ID, refs)
        exact = bql_compiler.compile_exact(expr, None)
        b = J.J("BooleanQueryBuilder")()
        Occur = J.J("Occur")
        # `exact` and `in_set` stay bound to locals until the search returns: pyjnius drops
        # a clause whose Python handle was a temporary.
        b.add(exact, Occur.MUST)
        b.add(in_set, Occur.FILTER)
        q = b.build()
        top = self._searcher_bql.search(q, len(ids))
        sf = self._searcher_bql.storedFields()
        return {sf.document(sd.doc).get(F_ID) for sd in top.scoreDocs}

    def count_bql_expr(self, expr) -> int:
        """Exact total-match count for an already-compiled BQL `Expr`
        (`IndexSearcher.count` evaluates the query without collecting or scoring a
        top-k), so the BQL adapter's search listing can
        show the same accurate "(N matches, top K)" header the Python engine's
        `run_with_count` provides, instead of a truncated `len(hits)`."""
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
        this one hit and regex-scan its description text for the known field names.
        Not authoritative, since an Explanation's text format is a Lucene
        implementation detail rather than a stable API; a display hint only, never
        used for ranking or logic."""
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
    """Process-wide cached engine per (index_root, dataset), so worker threads
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
