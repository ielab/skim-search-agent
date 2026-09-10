"""Faithful BM25 via Pyserini/Lucene (cluster: needs `pyserini` + Java 11+).

Matches SWE-bench's own BM25 retrieval setup (Anserini/Lucene): same engine, same
k1=0.9/b=0.4 (their implicit Lucene defaults), same analyzer (Porter stemming, no
identifier subtoken splitting), same adaptive query truncation. One deliberate
difference, stated when citing their numbers: SWE-bench indexes WHOLE FILES
(`relpath\\n + contents`) for context retrieval; we index function-level units
(`qualname + code`) because the task is function localization and EVERY SkimSearchAgent
condition shares this corpus — uniformity across conditions is the controlled
variable. Indexes are
PERSISTED and structured under `index_root/`, keyed by corpus identity, so they are
built once and reused across runs (and across instances that share a base commit):

    indexes/bm25_pyserini/<key>/corpus/docs*.jsonl  # the indexed units, one shard per
                                                    #   indexing thread (see index())
    indexes/bm25_pyserini/<key>/lucene/             # the Lucene index

Efficiency: the build shards the corpus into one jsonl per thread and runs Anserini's
JsonCollection indexer with `--threads` = all cores (env `BM25_PYSERINI_THREADS` overrides);
stored fields (positions/docvectors/raw) are OFF by default (env `BM25_PYSERINI_STORE_RAW=1`
restores them) since SkimSearchAgent's tool layer renders from its own in-memory units, never from
the index — see `_index_threads`/`_store_raw`. The searcher memory-maps the index (Lucene
MMapDirectory) and sets BM25 params once at construction.

`bm25_local` is the dependency-free approximation; use this for headline numbers.

ALSO the selectable BM25 ENGINE for the doc (research) arm: `agent_search.agent.retriever`
constructs this class (instead of `BM25Local`) for the bm25/bm25dci/bm25fetch/bm25q/
bm25fetchsnip arms when env `BM25_BACKEND=pyserini` (default `local`, unchanged behavior —
see agent/retriever.py's `index()`). Wired in because BM25Local's dependency-free analyzer
(agent_search.corpus.units.code_tokenize — identifier-aware camelCase/snake_case splitting,
no stemming) diverges badly from canonical Lucene BM25 (Porter stemming + stopwords,
k1=0.9/b=0.4) on prose corpora: an empirical top-5 Jaccard of ~0.546 on browsecomp_plus
between the two engines' rankings for the SAME query/corpus (see
tests/test_pyserini_backend.py's `test_agreement_pinned_within_tolerance_band`, which pins
this divergence with a tolerance band so analyzer drift is caught). `BM25_BACKEND=pyserini`
makes every reported bm25 number canonical Lucene, not the approximation.

The `search(query, k) -> list[str]` interface is IDENTICAL to `BM25Local.search` (see bm25.py),
so this class is a drop-in `engine=` for `Bm25Visit`/`Bm25FetchWorkspace`/`Bm25FetchSnipWorkspace`/
the BM25 tools (agent_search/tools/search_bm25, search_bm25_dci) with NO change
to their listing/best_line rendering, which only ever consumes the returned doc_id list.

Thread safety: `LuceneSearcher` wraps Anserini's `SimpleSearcher` via pyjnius (`jnius`), which
attaches each calling Python thread to the JVM transparently — verified empirically (500-doc
index, 8 threads x 20 concurrent `.search()` calls each on ONE shared searcher instance: zero
exceptions, results identical to the same queries run sequentially). Safe to share one
`BM25Pyserini` instance's `_searcher` across the agent's worker threads (default 8).

Offline safety: `index()`'s subprocess (`python -m pyserini.index.lucene`) and `search()`'s
`LuceneSearcher(index_dir)` both operate on a LOCAL directory path — never
`LuceneSearcher.from_prebuilt_index(name)`, the only pyserini call that fetches over the
network. The one network-shaped landmine is transitive: `pyserini.search.lucene` imports
`pyserini.encode._openai`, which raises at IMPORT time (not at call time — it eagerly
constructs an `openai.OpenAI()` client) if `OPENAI_API_KEY` is unset. That constructor makes
no network call itself, so a PLACEHOLDER value is sufficient and no actual key is ever used —
but it must NOT be set at this MODULE's import time: `agent_search.retrievers.registry`'s
`available()` imports every retriever module (this one included) just to run its `@register`
decorators, so a module-level `os.environ.setdefault` here would inject a fake
`OPENAI_API_KEY` into the whole HOST PROCESS's environment merely for listing retrievers,
with no matching cleanup. Instead, `_openai_placeholder_env()` (below) sets the placeholder
ONLY around the two places that actually import pyserini (the `LuceneSearcher` construction
in `index()`/`search()`'s callers, and the `python -m pyserini.index.lucene` build
subprocess, which inherits `os.environ`) — and restores the environment exactly as found
immediately after, via try/finally, so nothing leaks past the call that needed it.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from typing import Optional, Sequence

from agent_search.corpus.fingerprint import corpus_fingerprint
from agent_search.corpus.units import CodeUnit
from agent_search.core.interfaces import Retriever


@contextlib.contextmanager
def _openai_placeholder_env():
    """See module docstring's "Offline safety" paragraph: pyserini's transitive
    `_openai.py` needs `OPENAI_API_KEY` present (any value) only at IMPORT time, to
    construct its never-used OpenAI client — no network call happens. Set a placeholder
    ONLY if the var is currently absent, and remove it again in `finally` so a caller
    that had no key set still has none afterward (never leave a fake key sitting in a
    host process's environment past the one call that needed it)."""
    had_key = "OPENAI_API_KEY" in os.environ
    if not had_key:
        os.environ["OPENAI_API_KEY"] = "agent-search-unused-placeholder"
    try:
        yield
    finally:
        if not had_key:
            os.environ.pop("OPENAI_API_KEY", None)


@contextlib.contextmanager
def _silence_fd(fd: int = 2):
    """Temporarily redirect a file descriptor (default stderr) to /dev/null.

    Used only around JVM startup to swallow the benign one-time
    'WARNING: Using incubator modules: jdk.incubator.vector' the JVM prints.
    A real init failure still raises a Python exception, so nothing is hidden.
    """
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)

def _index_threads() -> int:
    """Indexing parallelism for `pyserini.index.lucene` — DEFAULT: all available cores
    (Anserini's JsonCollection indexer parallelizes across input FILES; sharding the corpus
    into one jsonl per thread is what unlocks it — see index() below). Env
    `BM25_PYSERINI_THREADS` overrides (e.g. set a small value on a busy shared node)."""
    env = os.environ.get("BM25_PYSERINI_THREADS")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _store_raw() -> bool:
    """Whether to ALSO store positions/docvectors/raw contents in the Lucene index. DEFAULT
    OFF — SkimSearchAgent's tool layer maps returned doc_ids back to its own in-memory units for all
    rendering (listings, best_line excerpts, whole-doc visits), so nothing ever reads a stored
    field back out of this index; storing them roughly triples the index size for zero
    consumer and slows the build. Env `BM25_PYSERINI_STORE_RAW=1` restores the old
    full-fidelity flags (positions + docvectors + raw) for ad-hoc debugging (e.g. inspecting
    indexed text via `searcher.doc(id).raw()`) or downstream tooling that wants phrase
    queries/relevance feedback over the same artifact."""
    return os.environ.get("BM25_PYSERINI_STORE_RAW", "") in ("1", "true", "yes")


class BM25Pyserini(Retriever):
    name = "bm25_pyserini"

    def __init__(self, k1: float = 0.9, b: float = 0.4,
                 index_root: str = "indexes", rebuild: bool = False):
        self.k1, self.b = k1, b
        self.index_root = index_root
        self.rebuild = rebuild
        self._searcher = None

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "BM25Pyserini":
        external = (os.environ.get("BM25_INDEX_PATH") or "").strip()
        if external:
            # a prebuilt Lucene index (e.g. ITER's wiki index): open it as is, never rebuild
            if not self._is_built(external):
                raise RuntimeError(f"BM25_INDEX_PATH={external!r} is not a Lucene index directory")
            with _openai_placeholder_env(), _silence_fd(2):
                from pyserini.search.lucene import LuceneSearcher
                self._searcher = LuceneSearcher(external)
                self._searcher.set_bm25(self.k1, self.b)
            return self
        if getattr(units, "lazy", False):
            from agent_search.core.errors import SetupError
            raise SetupError(
                f"corpus key {key!r} is an on-disk document store; BM25 over it needs a prebuilt "
                f"Lucene index: set BM25_INDEX_PATH (retrieval.bm25_index) to its directory")
        base = os.path.join(self.index_root, self.name, key or "default")
        index_dir = os.path.join(base, "lucene")
        fp = corpus_fingerprint(units)

        if self.rebuild or not self._is_built(index_dir, expected_n_docs=len(units),
                                              expected_fingerprint=fp):
            corpus_dir = os.path.join(base, "corpus")
            os.makedirs(corpus_dir, exist_ok=True)
            # ONE jsonl shard per indexing thread (ceil-split), because Anserini's
            # JsonCollection parallelizes ACROSS FILES: a single docs.jsonl is consumed by one
            # worker no matter what --threads says, so a one-file corpus indexes effectively
            # single-threaded. Sharding is what actually unlocks the multithreaded build.
            n_threads = _index_threads()
            per_shard = max(1, -(-len(units) // n_threads))      # ceil division
            fh, shard_idx, written = None, -1, per_shard          # force open on first doc
            try:
                for u in units:
                    if written >= per_shard:
                        if fh is not None:
                            fh.close()
                        shard_idx += 1
                        fh = open(os.path.join(corpus_dir, f"docs{shard_idx:05d}.jsonl"), "w")
                        written = 0
                    fh.write(json.dumps({"id": u.doc_id,
                                         "contents": f"{u.qualname} {u.code}"}) + "\n")
                    written += 1
            finally:
                if fh is not None:
                    fh.close()
            # drop a stale single-file corpus left by a build under the OLD layout, so a
            # rebuild never indexes the same docs twice (old docs.jsonl + new shards).
            legacy = os.path.join(corpus_dir, "docs.jsonl")
            if os.path.exists(legacy):
                os.remove(legacy)
            os.makedirs(index_dir, exist_ok=True)
            # Capture Lucene's verbose INFO logging to a per-index log file instead
            # of flooding the terminal; surface it only if indexing fails.
            log_path = os.path.join(base, "index.log")
            cmd = [
                sys.executable, "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection", "--input", corpus_dir,
                "--index", index_dir, "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", str(n_threads),
            ]
            if _store_raw():          # see _store_raw: OFF by default — lean postings-only
                cmd += ["--storePositions", "--storeDocvectors", "--storeRaw"]
            with open(log_path, "w") as log, _openai_placeholder_env():
                # the subprocess INHERITS os.environ, and `pyserini.index.lucene` transitively
                # imports the same OpenAI-client-at-import-time module `search.lucene` does
                # (see module docstring) — so the placeholder must be present for THIS call too.
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                with open(log_path) as f:
                    tail = f.read()[-2000:]
                raise RuntimeError(
                    f"pyserini indexing failed (key={key}); full log: {log_path}\n{tail}")
            # Cheap doc-count sentinel for the congruence check in `_is_built` below --
            # pyserini's `LuceneSearcher` python wrapper exposes no `numDocs()`/reader
            # accessor (unlike the raw-JNI lucene_structured backend, which reads it
            # straight off the index -- see index_builder.py's `_lucene_doc_count`), so
            # a small sidecar file is the cheapest way to record what corpus size this
            # index was built for.
            with open(os.path.join(base, "meta.json"), "w") as fh:
                json.dump({"n_docs": len(units), "corpus_fingerprint": fp}, fh)

        with _openai_placeholder_env(), _silence_fd(2):  # swallow the JVM incubator-modules warning
            # LuceneSearcher opens the index through Lucene's FSDirectory.open, which resolves
            # to MMapDirectory on 64-bit JVMs — memory-mapped, page-cache-backed postings, no
            # heap copy. BM25 params are set ONCE here at construction; searches never re-set.
            from pyserini.search.lucene import LuceneSearcher
            self._searcher = LuceneSearcher(index_dir)
            self._searcher.set_bm25(self.k1, self.b)
        return self

    @staticmethod
    def _is_built(index_dir: str, expected_n_docs: Optional[int] = None,
                  expected_fingerprint: Optional[str] = None) -> bool:
        """A Lucene index exists if its dir holds a `segments_*` file. `expected_n_docs`
        (passed by `index()`, where the current corpus size is known) additionally
        cross-checks the `meta.json` doc-count sentinel written at build time (see
        `index()`) -- same correctness guard as the dense-cache check in
        `retrievers/dense/dense.py`: a stale index left under the same
        `<index_root>/bm25_pyserini/<key>/lucene/` path after the corpus changed would
        otherwise be silently reused, serving hits for the WRONG document set with no
        error anywhere. `None` (the default, used by the units-free `is_cached` probe)
        skips this check -- that probe exists specifically to skip parsing the corpus,
        so it has no count to check against. An index built before this check existed
        has no `meta.json`; that's trusted as-is (segments-only, prior behavior) rather
        than forced to rebuild.

        `expected_fingerprint` (`agent_search.corpus.fingerprint.corpus_fingerprint`, also
        passed only by `index()`) catches the same-doc-COUNT-different-CONTENT case (a unit
        edited in place, same corpus size) that the doc-count check alone can't see. `None`
        skips this check too, and a `meta.json` written before this key existed has none --
        trusted as-is, same as the doc-count check's own pre-existing-index fallback."""
        if not (os.path.isdir(index_dir) and any(
                f.startswith("segments") for f in os.listdir(index_dir))):
            return False
        meta_path = os.path.join(os.path.dirname(index_dir), "meta.json")
        meta = None
        if expected_n_docs is not None or expected_fingerprint is not None:
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                except Exception:
                    meta = None
        if expected_n_docs is not None and meta is not None:
            n = meta.get("n_docs")
            if n is not None and n != expected_n_docs:
                print(f"  [bm25_pyserini] WARNING: index at {index_dir!r} has "
                      f"n_docs={n} in its meta.json but the current corpus has "
                      f"{expected_n_docs} docs -- stale/mismatched index, "
                      f"rebuilding", file=sys.stderr, flush=True)
                return False
        if expected_fingerprint is not None and meta is not None:
            fp = meta.get("corpus_fingerprint")
            if fp is not None and fp != expected_fingerprint:
                print(f"  [bm25_pyserini] WARNING: index at {index_dir!r} has a different "
                      f"corpus_fingerprint than the current corpus -- stale/mismatched "
                      f"index (content changed), rebuilding", file=sys.stderr, flush=True)
                return False
        return True

    def is_cached(self, key: Optional[str] = None) -> bool:
        """True if the Lucene index for `key` is already built (no rebuild), so a
        pre-builder can skip re-parsing the corpus's units for it."""
        index_dir = os.path.join(self.index_root, self.name, key or "default", "lucene")
        return self._is_built(index_dir)

    def search(self, query: str, k: int) -> list[str]:
        # SWE-bench's adaptive truncation: very long issue texts can exceed
        # Lucene's clause limit; shrink by 20% until the query is accepted.
        cutoff = len(query)
        last_exc: Exception | None = None
        while cutoff > 0:
            try:
                hits = self._searcher.search(query[:cutoff], k=k)
                return [h.docid for h in hits]
            except Exception as e:           # noqa: BLE001 - JVM errors vary
                last_exc = e
                cutoff = int(cutoff * 0.8)   # floor: reaches 0 (round() fixates at 2)
        # not a query-length problem (corrupt index, JVM failure): surface it —
        # the eval counts it as a per-instance error instead of hanging a worker
        raise RuntimeError(f"pyserini search failed at every cutoff: {last_exc}")


# --- registry ---------------------------------------------------------------
from agent_search.retrievers.registry import RetrieverConfig, register  # noqa: E402


@register("bm25_pyserini")
def _build_bm25_pyserini(cfg: RetrieverConfig, name: str):
    return lambda: BM25Pyserini(index_root=cfg.index_root, rebuild=cfg.rebuild)
