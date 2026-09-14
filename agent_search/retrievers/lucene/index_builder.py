"""Builds the fielded Lucene index this backend searches: raw JNI (`IndexWriter` plus
typed `Field`s), not pyserini's `LuceneSearcher`/JsonCollection indexing pipeline
(`agent_search.retrievers.lexical.pyserini.BM25Pyserini.index`), which only emits an
`{id, contents}` bag-of-text document and cannot express per-field schema at all.

Field schema: see `schema.py`'s module docstring for the full decision + rationale
(one Lucene Document per `CodeUnit`, stemmed/exact field pairs for body/title/
section, StringField date/author).

Persisted under `<index_root>/lucene_structured/<dataset>_v<schema>/` (index_root defaults to
`indexes/`, mirroring every other backend's `indexes/<name>/<key>/` convention).

CLI:
    python -m agent_search.retrievers.lucene.index_builder \\
        --dataset browsecomp_plus_structured [--index-root indexes] [--rebuild] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from agent_search.corpus.fingerprint import corpus_fingerprint
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.indri.fields import field_text
from agent_search.retrievers.lucene import jni_utils as J
from agent_search.retrievers.lucene.schema import (
    F_AUTHOR, F_AUTHOR_TEXT, F_BODY, F_BODY_EXACT, F_DATE, F_DATE_TEXT, F_ID,
    F_INFOBOX, F_INFOBOX_EXACT, F_SECTION, F_SECTION_EXACT, F_TITLE, F_TITLE_EXACT,
)

# meta.json: a sidecar written at build time, alongside (never inside) the Lucene segment
# files. Lucene never writes a file by this name, so there's no collision risk. Uses the
# same meta.json congruence-check pattern as `bm25_pyserini` (lexical/pyserini.py).
_META_FILE = "meta.json"


def index_dir(index_root: str, dataset: str) -> str:
    """`<index_root>/lucene_structured/<dataset>_v<schema>`; schema 1 had no suffix."""
    from agent_search.retrievers.lucene.schema import SCHEMA_VERSION
    suffix = "" if SCHEMA_VERSION == 1 else f"_v{SCHEMA_VERSION}"
    return os.path.join(index_root, "lucene_structured", dataset + suffix)


def has_segments(path: str) -> bool:
    """A Lucene index exists at `path` if it holds a `segments_*` file (the same check
    `BM25Pyserini._is_built` uses); shared by `is_built` here and
    `LuceneStructuredEngine.is_cached` (engine.py), which checks an arbitrary
    `index_path` (not necessarily `index_dir(index_root, dataset)`, since the
    engine also accepts a bare `index_path=` override).

    A pure presence check, with no doc-count validation: this is also the
    units-free probe (`LuceneIndexBuilder.is_cached` / `LuceneStructuredEngine.is_cached`)
    callers use specifically to skip parsing the corpus, so it has no corpus size
    to check against. The doc-count congruence check lives in `is_built` below,
    called from `build()` where `units` (and hence the expected count) is
    available."""
    return os.path.isdir(path) and any(f.startswith("segments") for f in os.listdir(path))


def _lucene_doc_count(path: str) -> Optional[int]:
    """Cheap doc-count probe: opens (and immediately closes) a `DirectoryReader`
    on the index at `path` and returns `numDocs()`. Reads only segment metadata
    (the index's own commit-time doc count), no postings or term-vector I/O, so
    this is fast even on a 100k-plus-doc index; not a hand-maintained sidecar file
    that could itself drift out of sync with the real index. Returns None if the
    directory can't be opened as a Lucene index (corrupt). Callers should already
    have checked `has_segments` first, so this is a defensive fallback, not the
    primary presence check."""
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


def is_built(index_root: str, dataset: str, expected_n_docs: Optional[int] = None,
             expected_fingerprint: Optional[str] = None) -> bool:
    """`expected_n_docs` (passed by `build()`, where the current corpus size is
    known) adds a doc-count congruence check on top of the segments-presence
    check: the same correctness guard the dense-cache check in
    `retrievers/dense/base.py` uses. A stale index left on disk under the same
    `<index_root>/lucene_structured/<dataset>/` key after the corpus changed
    (added or removed docs) would otherwise be silently reused, serving hits for
    the wrong document set with no error anywhere. `None` (the default, used by
    every units-free `is_cached` probe) skips this check entirely; see
    `has_segments`'s docstring for why those callers have no count to check.

    `expected_fingerprint` (`agent_search.corpus.fingerprint.corpus_fingerprint`, also
    passed only by `build()`) additionally catches the case the doc-count check alone
    can't see: same doc count, different content (a unit edited in place, same corpus
    size). Compared against the `corpus_fingerprint` key in the `meta.json` sidecar
    `build()` writes; a present-but-different value is treated as stale, same as a
    doc-count mismatch. `None` (the default) skips this check too; an index built
    before this check existed has no `meta.json` or fingerprint key, and is trusted
    as-is, like the doc-count check's own pre-existing-index fallback."""
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
    if expected_fingerprint is not None:
        meta_path = os.path.join(path, _META_FILE)
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    fp = json.load(fh).get("corpus_fingerprint")
            except Exception:
                fp = None
            if fp is not None and fp != expected_fingerprint:
                print(f"[lucene_structured] WARNING: index at {path!r} has a different "
                      f"corpus_fingerprint than the current corpus -- stale/mismatched "
                      f"index (content changed), rebuilding", file=sys.stderr, flush=True)
                return False
    return True


def _per_field_analyzer():
    """`body`/`title`/`section` map to the stemmed `EnglishAnalyzer`; `*_exact` maps to
    the unstemmed `SimpleAnalyzer`; anything else (StringField `id`/`date`/`author`,
    since Lucene never invokes the configured Analyzer for a StringField, it indexes
    the raw value as a single unanalyzed token internally) falls back to the stemmed
    analyzer, which has no effect since it's never called for those fields."""
    HashMap = J.J("HashMap")()
    stemmed = J.stemmed_analyzer()
    exact = J.exact_analyzer()
    for f in (F_BODY, F_TITLE, F_SECTION, F_INFOBOX):
        HashMap.put(f, stemmed)
    for f in (F_BODY_EXACT, F_TITLE_EXACT, F_SECTION_EXACT, F_INFOBOX_EXACT, F_AUTHOR_TEXT, F_DATE_TEXT):
        HashMap.put(f, exact)
    PerFieldAnalyzerWrapper = J.J("PerFieldAnalyzerWrapper")
    return PerFieldAnalyzerWrapper(stemmed, HashMap)


_RAM_BUFFER_MB = float(os.environ.get("LUCENE_INDEX_RAM_MB", "512"))
# Default 1 (serial), not `min(8, cpu_count)`: see `build()`'s docstring. Concurrent
# `IndexWriter.addDocument` calls from multiple Python/pyjnius threads raise a
# reproducible `java.util.ConcurrentModificationException` deep inside Lucene's
# `IndexingChain.processDocument` on a large real build, even though each thread
# builds its own `Document` via `_build_document` with no shared mutable Python
# state. The cause is the single, module-cached `EnglishAnalyzer`/`SimpleAnalyzer`
# instances (`jni_utils.stemmed_analyzer`/`exact_analyzer`, reused across all
# fields and threads for cheapness): they are not safely reentrant under this
# pyjnius/bundled-Lucene combination when `tokenStream()` is invoked concurrently
# from multiple JNI-attached threads. Pyserini's own `LuceneSearcher` is confirmed
# thread-safe for concurrent search (read-only, see pyserini.py's docstring), but
# Lucene's Analyzer/IndexWriter indexing path is not thread-safe in this
# environment. Left as an opt-in knob (`LUCENE_INDEX_THREADS`) for a follow-up
# that gives each thread its own Analyzer instance instead of a shared one; the
# RAM buffer, MMapDirectory and single-commit tuning below already keep a large
# bulk build fast without it.
_INDEX_THREADS = int(os.environ.get("LUCENE_INDEX_THREADS", "1"))


_ISO_DATE_PREFIX_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def iso_date(raw: str) -> Optional[str]:
    """`YYYY-MM-DD` when `raw` starts with a real calendar date, else None."""
    m = _ISO_DATE_PREFIX_RE.match(str(raw or ""))
    if not m:
        return None
    y, mo, d = (int(g) for g in m.groups())
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _build_document(u: CodeUnit):
    """One `CodeUnit` -> one Lucene `Document`, entirely local JVM object
    construction (no shared mutable state), safe to call from multiple threads
    concurrently, each building its own documents before handing them to the
    (thread-safe) `IndexWriter.addDocument`.

    Lean index: only `id` is stored (`FieldStore.YES`). Every other field is
    indexed (tokenized and positioned, for matching and scoring) but not stored:
    the caller already holds the live `CodeUnit`/corpus mapping doc_id -> unit and
    reads title, section, infobox, date, author and body from there, not from the index, so
    storing them a second time here would only bloat the index on disk for data
    the tool layer never reads back through this engine."""
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
    infobox_text = field_text(u, "infobox")
    if infobox_text:
        doc.add(TextField(F_INFOBOX, infobox_text, FieldStore.NO))
        doc.add(TextField(F_INFOBOX_EXACT, infobox_text, FieldStore.NO))
    # date/author: omit the StringField entirely when empty or missing, rather than
    # indexing an empty-string value. This matters for correctness, not just
    # leanness: an empty string sorts lexicographically before every real ISO
    # date, so `TermRangeQuery.newStringRange(date, null, "1985-01-01", ...)`
    # (an open-lower-bound "#date:before" query) would otherwise match every
    # doc with a missing date too, silently including the whole missing-date
    # corpus in a "before 1985" result. An absent field can never satisfy a
    # Lucene range/term query, which is the contract we want: it matches the
    # Python reference's `_unit_date` -> None -> "never matches any date
    # filter" contract. The range field takes only a real calendar date written as
    # `YYYY-MM-DD` (a longer timestamp keeps its date prefix); any other value would sort
    # below every ISO date and match every open-lower-bound range. The tokenized `date_text`
    # field keeps the raw value for `IN(date, x)` term matches.
    if date_text:
        iso = iso_date(date_text)
        if iso:
            doc.add(StringField(F_DATE, iso, FieldStore.NO))
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

    Bulk-indexing tuning:
      - `MMapDirectory` (explicit, not just `FSDirectory.open`'s platform default)
        for both the writer here and the reader in `engine.py`.
      - `IndexWriterConfig.setRAMBufferSizeMB` (default 512, `LUCENE_INDEX_RAM_MB`):
        a generous RAM buffer means far fewer segment flushes during the bulk
        load, each flush being the expensive part.
      - One `commit()` at the end, not per document: the only correct way to
        use `IndexWriter` for a bulk load. Per-doc commits would be a serious
        anti-pattern (a full fsync-durable segment write per document).
      - `n_threads` (default 1, `LUCENE_INDEX_THREADS`): serial by default, since
        concurrent `IndexWriter.addDocument` feeding is not safe here; see the
        `_INDEX_THREADS` module comment above for the reproducible
        `ConcurrentModificationException` and its cause. Left as an opt-in knob
        for a follow-up.
    """
    out_dir = index_dir(index_root, dataset)
    fp = corpus_fingerprint(units)
    if not rebuild and is_built(index_root, dataset, expected_n_docs=len(units),
                                expected_fingerprint=fp):
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
    # meta.json sidecar (same pattern as bm25_pyserini's own meta.json): carries the
    # doc-count and content fingerprint `is_built` checks on the next build or open, so
    # a corpus change (added or removed docs, or a unit edited in place) is caught even
    # though Lucene's own segment files carry no such application-level identity.
    try:
        with open(os.path.join(out_dir, _META_FILE), "w") as fh:
            json.dump({"n_docs": n, "corpus_fingerprint": fp}, fh)
    except Exception:
        pass                    # best-effort sidecar; a missing/corrupt one just skips the check
    elapsed = time.time() - t0
    return {"n_docs": n, "elapsed_s": elapsed, "index_dir": out_dir, "skipped": False,
            "ram_buffer_mb": _RAM_BUFFER_MB, "n_threads": threads}


class LuceneIndexBuilder:
    """Offline persister (`.index(units, key)` / `.is_cached(key)`) for
    `agent_search/evaluation/build_indexes.py`, under the retriever kind `search_lucene`."""
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

    from agent_search.evaluation.datasets import load_dataset_by_name
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
