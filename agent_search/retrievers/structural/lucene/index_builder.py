"""Builds the fielded Lucene index this backend searches -- raw JNI (`IndexWriter` +
typed `Field`s), NOT pyserini's `LuceneSearcher`/JsonCollection indexing pipeline
(`agent_search.retrievers.lexical.pyserini.BM25Pyserini.index`), which only emits an
`{id, contents}` bag-of-text document and cannot express per-field schema at all.

Field schema: see `schema.py`'s module docstring for the full decision + rationale
(one Lucene Document per `CodeUnit`, stemmed/exact field pairs for body/title/
section, StringField date/author).

Persisted under `<index_root>/lucene_structured/<dataset>/` (index_root defaults to
`indexes/`, mirroring every other backend's `indexes/<name>/<key>/` convention).

CLI:
    envs/bin/python -m agent_search.retrievers.structural.lucene.index_builder \\
        --dataset browsecomp_plus_structured [--index-root indexes] [--rebuild] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.structural.indri.index import field_text
from agent_search.retrievers.structural.lucene import jni_utils as J
from agent_search.retrievers.structural.lucene.schema import (
    F_AUTHOR, F_AUTHOR_TEXT, F_BODY, F_BODY_EXACT, F_DATE, F_DATE_TEXT, F_ID,
    F_SECTION, F_SECTION_EXACT, F_TITLE, F_TITLE_EXACT,
)


def index_dir(index_root: str, dataset: str) -> str:
    return os.path.join(index_root, "lucene_structured", dataset)


def has_segments(path: str) -> bool:
    """A Lucene index exists at `path` if it holds a `segments_*` file (mirrors
    `BM25Pyserini._is_built`'s own check) -- shared by `is_built` here and
    `LuceneStructuredEngine.is_cached` (engine.py), which checks an arbitrary
    `index_path` (not necessarily `index_dir(index_root, dataset)`, since the
    engine also accepts a bare `index_path=` override).

    Deliberately left as a pure presence check (no doc-count validation): this
    is also the units-FREE probe (`LuceneIndexBuilder.is_cached` /
    `LuceneStructuredEngine.is_cached`) callers use specifically to skip
    parsing the corpus, so it has no corpus size to check against. The
    doc-count congruence check lives in `is_built` below, called from `build()`
    where `units` (and hence the expected count) IS available."""
    return os.path.isdir(path) and any(f.startswith("segments") for f in os.listdir(path))


def _lucene_doc_count(path: str) -> Optional[int]:
    """Cheap doc-count probe: opens (and immediately closes) a `DirectoryReader`
    on the index at `path` and returns `numDocs()`. Reads only segment metadata
    (the index's own commit-time doc count) -- no postings/term-vector I/O -- so
    this is fast even on a 100k+-doc index; NOT a hand-maintained sidecar file
    that could itself drift out of sync with the real index. Returns None if the
    directory can't be opened as a Lucene index (corrupt -- callers should
    already have checked `has_segments` first, so this is a defensive fallback,
    not the primary presence check)."""
    try:
        MMapDirectory = J.J("MMapDirectory")
        Paths = J.J("Paths")
        DirectoryReader = J.J("DirectoryReader")
        d = MMapDirectory(Paths.get(path))
        try:
            reader = DirectoryReader.open(d)
            try:
                return int(reader.numDocs())
            finally:
                reader.close()
        finally:
            d.close()
    except Exception:
        return None


def is_built(index_root: str, dataset: str, expected_n_docs: Optional[int] = None) -> bool:
    """`expected_n_docs` (passed by `build()`, where the current corpus size is
    known) adds a doc-count congruence check on top of the segments-presence
    check -- same correctness guard as the dense-cache check in
    `retrievers/dense/dense.py`: a stale index left on disk under the same
    `<index_root>/lucene_structured/<dataset>/` key after the corpus changed
    (added/removed docs) would otherwise be silently reused, serving hits for
    the WRONG document set with no error anywhere. `None` (the default, used by
    every units-free `is_cached` probe) skips this check entirely -- see
    `has_segments`'s docstring for why those callers have no count to check."""
    path = index_dir(index_root, dataset)
    if not has_segments(path):
        return False
    if expected_n_docs is not None:
        n = _lucene_doc_count(path)
        if n is not None and n != expected_n_docs:
            print(f"[lucene_structured] WARNING: index at {path!r} has {n} docs "
                  f"but the current corpus has {expected_n_docs} -- stale/mismatched "
                  f"index, rebuilding", file=sys.stderr, flush=True)
            return False
    return True


def _per_field_analyzer():
    """`body`/`title`/`section` -> stemmed `EnglishAnalyzer`; `*_exact` -> unstemmed
    `SimpleAnalyzer`; anything else (StringField `id`/`date`/`author` -- Lucene never
    invokes the configured Analyzer for a StringField, it indexes the raw value as a
    single unanalyzed token internally) falls back to the stemmed analyzer, which is
    irrelevant since it's never called for those fields."""
    HashMap = J.J("HashMap")()
    stemmed = J.stemmed_analyzer()
    exact = J.exact_analyzer()
    for f in (F_BODY, F_TITLE, F_SECTION):
        HashMap.put(f, stemmed)
    for f in (F_BODY_EXACT, F_TITLE_EXACT, F_SECTION_EXACT, F_AUTHOR_TEXT, F_DATE_TEXT):
        HashMap.put(f, exact)
    PerFieldAnalyzerWrapper = J.J("PerFieldAnalyzerWrapper")
    return PerFieldAnalyzerWrapper(stemmed, HashMap)


_RAM_BUFFER_MB = float(os.environ.get("LUCENE_INDEX_RAM_MB", "512"))
# Default 1 (serial), NOT `min(8, cpu_count)` -- see `build()`'s docstring:
# concurrent `IndexWriter.addDocument` calls from multiple Python/pyjnius threads
# were tried and reproducibly threw `java.util.ConcurrentModificationException`
# deep inside Lucene's `IndexingChain.processDocument` on the real 67707-doc
# browsecomp_plus_structured build (each thread building its OWN `Document` via
# `_build_document` -- no shared mutable Python state -- yet the crash was
# consistent, not a rare race). Root-caused as: the single, MODULE-CACHED
# `EnglishAnalyzer`/`SimpleAnalyzer` instances (`jni_utils.stemmed_analyzer`/
# `exact_analyzer`, reused across ALL fields/threads for cheapness) are not
# safely reentrant under this pyjnius/bundled-Lucene combination when
# `tokenStream()` is invoked concurrently from multiple JNI-attached threads --
# unlike pyserini's OWN `LuceneSearcher`, which pyserini.py's docstring confirms
# thread-safe for concurrent SEARCH (read-only), Lucene's Analyzer/IndexWriter
# INDEXING path was not verified thread-safe in this environment and empirically
# is not. Left as an opt-in knob (`LUCENE_INDEX_THREADS`) for a follow-up that
# root-causes it further (e.g. one Analyzer instance per thread instead of one
# shared instance); the RAM buffer + MMapDirectory + single-commit tuning below
# already gets the real bulk build to the range reported in REPORT without it.
_INDEX_THREADS = int(os.environ.get("LUCENE_INDEX_THREADS", "1"))


def _build_document(u: CodeUnit):
    """One `CodeUnit` -> one Lucene `Document`, entirely LOCAL JVM object
    construction (no shared mutable state) -- safe to call from multiple threads
    concurrently, each building its own documents before handing them to the
    (thread-safe) `IndexWriter.addDocument`.

    LEAN INDEX: only `id` is stored (`FieldStore.YES`). Every other field is
    indexed (tokenized/positions, for matching+scoring) but NOT stored -- the
    caller already holds the live `CodeUnit`/corpus mapping doc_id -> unit and
    reads title/section/date/author/body from THERE, not from the index; storing
    them a second time here would only bloat the index on disk for data the tool
    layer never reads back through this engine."""
    Document = J.J("Document")
    FieldStore = J.J("FieldStore")
    TextField = J.J("TextField")
    StringField = J.J("StringField")

    doc = Document()
    doc.add(StringField(F_ID, u.doc_id, FieldStore.YES))
    body_text = field_text(u, "body")
    title_text = field_text(u, "title")
    section_text = field_text(u, "section")
    date_text = field_text(u, "date")
    author_text = field_text(u, "author")
    doc.add(TextField(F_BODY, body_text, FieldStore.NO))
    doc.add(TextField(F_BODY_EXACT, body_text, FieldStore.NO))
    doc.add(TextField(F_TITLE, title_text, FieldStore.NO))
    doc.add(TextField(F_TITLE_EXACT, title_text, FieldStore.NO))
    doc.add(TextField(F_SECTION, section_text, FieldStore.NO))
    doc.add(TextField(F_SECTION_EXACT, section_text, FieldStore.NO))
    # date/author: OMIT the StringField entirely when empty/missing, rather than
    # indexing an empty-string value. This matters for correctness, not just
    # leanness: an empty string sorts LEXICOGRAPHICALLY BEFORE every real ISO
    # date, so `TermRangeQuery.newStringRange(date, null, "1985-01-01", ...)`
    # (an OPEN-lower-bound "#date:before" query) would otherwise match every
    # doc with a missing date too -- silently including the whole
    # missing-date corpus in a "before 1985" result. An ABSENT field can never
    # satisfy a Lucene range/term query, which is what we want: this mirrors
    # the Python reference's `_unit_date` -> None -> "never matches ANY date
    # filter" contract exactly (indri/model.py, bql/executor.py's `_unit_date`).
    if date_text:
        doc.add(StringField(F_DATE, date_text, FieldStore.NO))
        doc.add(TextField(F_DATE_TEXT, date_text, FieldStore.NO))
    if author_text:
        doc.add(StringField(F_AUTHOR, author_text, FieldStore.NO))
        doc.add(TextField(F_AUTHOR_TEXT, author_text, FieldStore.NO))
    return doc


def build(units: Sequence[CodeUnit], index_root: str, dataset: str,
          rebuild: bool = False, progress: bool = True,
          n_threads: Optional[int] = None) -> dict:
    """Build (or skip, if already built and not `rebuild`) the Lucene index for
    `units` under `<index_root>/lucene_structured/<dataset>/`. Returns a small stats
    dict (n_docs, elapsed_s, index_dir, skipped).

    Bulk-indexing tuning (per-task requirement, measured on the real
    browsecomp_plus_structured build -- see REPORT):
      - `MMapDirectory` (explicit, not just `FSDirectory.open`'s platform default)
        for both the writer here and the reader in `engine.py`.
      - `IndexWriterConfig.setRAMBufferSizeMB` (default 512, `LUCENE_INDEX_RAM_MB`)
        -- a generous RAM buffer means far fewer segment flushes during the bulk
        load, each flush being the expensive part.
      - ONE `commit()` at the end (not per-doc) -- already the only correct way to
        use `IndexWriter` for a bulk load; per-doc commits would be a serious
        anti-pattern (a full fsync-durable segment write per document).
      - `n_threads` (default 1, `LUCENE_INDEX_THREADS`): concurrent
        `IndexWriter.addDocument` feeding was ATTEMPTED but reverted to serial by
        default -- see the `_INDEX_THREADS` module comment above for the
        reproducible `ConcurrentModificationException` this hit on the real
        67707-doc build, and why. Left as an opt-in knob for a follow-up.
    """
    out_dir = index_dir(index_root, dataset)
    if not rebuild and is_built(index_root, dataset, expected_n_docs=len(units)):
        return {"n_docs": len(units), "elapsed_s": 0.0, "index_dir": out_dir, "skipped": True}

    os.makedirs(out_dir, exist_ok=True)
    MMapDirectory = J.J("MMapDirectory")
    Paths = J.J("Paths")
    IndexWriter = J.J("IndexWriter")
    IndexWriterConfig = J.J("IndexWriterConfig")
    OpenMode = J.J("OpenMode")

    mmap_dir = MMapDirectory(Paths.get(out_dir))
    cfg = IndexWriterConfig(_per_field_analyzer())
    cfg.setOpenMode(OpenMode.CREATE)
    cfg.setRAMBufferSizeMB(_RAM_BUFFER_MB)
    writer = IndexWriter(mmap_dir, cfg)

    t0 = time.time()
    n = len(units)
    threads = n_threads if n_threads is not None else _INDEX_THREADS
    threads = max(1, min(threads, n)) if n else 1
    done = 0
    done_lock = threading.Lock()

    def _feed(unit: CodeUnit) -> None:
        nonlocal done
        doc = _build_document(unit)
        writer.addDocument(doc)
        if progress:
            with done_lock:
                done += 1
                if done % 5000 == 0:
                    print(f"[lucene_structured] indexed {done}/{n} ({time.time() - t0:.1f}s)",
                          file=sys.stderr)

    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(_feed, units))
    else:
        for u in units:
            _feed(u)

    writer.commit()
    writer.close()
    mmap_dir.close()
    elapsed = time.time() - t0
    return {"n_docs": n, "elapsed_s": elapsed, "index_dir": out_dir, "skipped": False,
            "ram_buffer_mb": _RAM_BUFFER_MB, "n_threads": threads}


class LuceneIndexBuilder:
    """Offline persister mirroring `BQLIndexBuilder`/`IndriIndexBuilder`'s shape
    (`.index(units, key)` / `.is_cached(key)`), so this backend can plug into the
    same `evaluation/build_indexes.py` step-0 prebuild pattern if a follow-up wires
    it in (registered name left unregistered here -- see module docstring's task
    scope: this is an additive, standalone package)."""
    name = "search_lucene"

    def __init__(self, index_root: str = "indexes", rebuild: bool = False):
        self.index_root = index_root
        self.rebuild = rebuild

    def is_cached(self, key: str) -> bool:
        return is_built(self.index_root, key)

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "LuceneIndexBuilder":
        build(units, self.index_root, key or "default", rebuild=self.rebuild)
        return self


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index-root", default="indexes")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of corpus documents indexed (debugging)")
    args = ap.parse_args()

    from evaluation.datasets import load_dataset_by_name
    from agent_search.corpus.units import units_from_documents

    instances = load_dataset_by_name(args.dataset, limit=1)
    if not instances or instances[0].docs is None:
        raise SystemExit(f"dataset {args.dataset!r} has no shared `docs` corpus to index")
    docs = instances[0].docs
    if args.limit is not None:
        docs = docs[: args.limit]
    units = units_from_documents(docs)
    print(f"[lucene_structured] {args.dataset}: {len(units)} units from {len(docs)} docs",
          file=sys.stderr)
    stats = build(units, args.index_root, args.dataset, rebuild=args.rebuild)
    print(f"[lucene_structured] done: {stats}", file=sys.stderr)


if __name__ == "__main__":
    _main()
