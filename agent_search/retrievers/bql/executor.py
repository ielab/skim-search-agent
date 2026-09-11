"""Index-free structural executor: the method.

Evaluates a parsed BQL query directly against live code units (re-parsed from the
current files each call), with no persisted index and no embeddings. This is the
"complex grep": Boolean logic + AST scope + recall expansion, run over live text.

Semantics (reference implementation):
  - Boolean selects the candidate set; a deterministic score orders it.
  - Units are function defs, so IN(def, x) ≡ x matches within the unit.
  - IN(file, x) switches scope to the whole file (path + all its units).
  - NEAR/func|block|para|sent ≡ co-occurrence in the unit; NEAR/file ≡ co-occurrence
    anywhere in the unit's FILE; NEAR/wN ≡ within N tokens; NEAR/lineN ≡ within N
    source lines. Multi-token terms require adjacency.
  - Window NEAR honors its operands' full match semantics: Term/Phrase (distance to
    the nearest token of a contiguous match), Prefix/Expand (prefix matches), and
    OR (union of child positions). Only And-composed operands fall back to unit
    co-occurrence (a conjunction has no single position); the type checker already
    rejects In/Near operands inside windows.
  - IN(def/call/comment/string/sig) use real AST scope:
    IN(call,x) matches where x is called, IN(def,x) where it's defined, etc.: the
    differentiator from plain grep. (ast-grep is the faster cross-language backend.)
  - EXPAND is a bounded prefix/identifier-variant match in the current executor.
"""
from __future__ import annotations

import bisect
import os
import pickle
import re
import threading
from typing import Optional, Sequence

from agent_search.corpus import grounding
from agent_search.corpus.fingerprint import corpus_fingerprint
from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.lexical.scorer import BM25
from agent_search.retrievers.bql.ast import (
    And, Expand, Expr, Granularity, In, Near, Not, Or, Phrase, Prefix, Region, Term,
)
from agent_search.retrievers.bql.structure import region_token_bags
from agent_search.retrievers.bql.dense_fuse import (
    fuse_coverage_tiers, fuse_ranked)
# Kept as a separate import statement (not folded into the one above) for the dense-only
# executor (`DenseOnlyStructuralExecutor`/`load_or_build_dense_only` at the bottom of this
# file, used by the `sieve_dense`/`sieve_visit_dense` strategies), so it is obvious this
# import serves only that class.
from agent_search.retrievers.bql.dense_fuse import (
    fuse_coverage_tiers_dense_only, fuse_ranked_dense_only)

# NEAR specs that mean "co-occur in the same region" rather than a token window.
_COOCCUR_SPECS = ("func", "file", "block", "para", "sent")

# BQL region -> structural bag key (real AST scope); others fall back to unit text.
_REGION_KEY = {
    Region.DEF: "def", Region.CALL: "call", Region.STRING: "string",
    Region.COMMENT: "comment", Region.SIG: "sig",
}

# document field regions handled by _field_bags. AUTHOR/DATE/INFOBOX come from unit metadata.
_FIELD_REGIONS = {Region.TITLE, Region.BODY, Region.SECTION, Region.DOC,
                  Region.AUTHOR, Region.DATE, Region.INFOBOX}
# Every doc field is narrowable. BODY/DOC use the combined corpus postings (body text
# dominates that index, so it's already tight); the sparse fields below get their own inverted
# index (token -> units where the token appears in that field), so IN(title/section/infobox/
# author/date, x) narrows to the exact matching docs (or empty, instant) instead of a full
# O(N) scan that would build every doc's field bags on the query clock: a 30s+/query
# (infobox/author/date) tail without it. With fielded postings a field query is a dict lookup
# plus a set operation.
_PREFILTER_FIELDS = {Region.TITLE, Region.BODY, Region.SECTION, Region.DOC}
_INDEXED_FIELDS = frozenset({Region.TITLE, Region.SECTION, Region.INFOBOX,
                             Region.AUTHOR, Region.DATE})

# --- BQL v2 Feature 1: typed date-range queries (surface.py `date[RANGE]`) --------------
#
# The surface lowers `date[1980..1989]` to `IN(date, __daterange__1980-01-01__1989-12-31)`,
# an ordinary In(Region.DATE, Term) node whose term text carries a canonical range encoding,
# rather than a new AST leaf (see surface.py's module docstring for why: it keeps the
# parser/type-checker/_rank_leaves untouched and the string round-trips through bql_parse).
# `_DATE_RANGE_TERM_RE` is how this executor recognizes that encoding at eval/candidates time.
_DATE_RANGE_TERM_RE = re.compile(
    r"^__daterange__(open|\d{4}-\d{2}-\d{2})__(open|\d{4}-\d{2}-\d{2})$")
_ISO_DATE_PREFIX_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")

# --- BQL v2 Feature 2: constraint-coverage ranking (StructuralExecutor.coverage_topk) ----
#
# Cap on the per-(doc, child) exact-_eval pool: for an AND whose children are all unnarrowable
# (e.g. every clause a Prefix), the candidate union is the whole corpus, and testing N children
# against N units is the same O(N) work run_with_count already pays. coverage_topk pays it
# even on a hit-having query's near-miss, though, so cap it to the top BM25-scored slice like
# the spec's "top ~2000 docs" guidance, not literal N.
_COVERAGE_POOL_CAP = 2000
# Size of the 0-hit fallback pool (`soft_topk`): the lexically closest docs plus, when a dense
# belief is attached, the dense side's nearest neighbours, before the condition's ranking
# model orders them.
_SOFT_POOL = int(os.environ.get("BQL_SOFT_POOL", "100"))

# Above this corpus size, narrow the live scan with an inverted-index prefilter
# (filter-then-verify) instead of scanning every unit per query. Below it (the common
# per-query code repo), keep the plain O(N) live scan. Override with
# AGENT_SEARCH_BQL_PREFILTER_MIN; the prefilter is recall-safe, so results are
# identical either way, only the speed differs.
_PREFILTER_MIN_UNITS = int(os.environ.get("AGENT_SEARCH_BQL_PREFILTER_MIN", "5000"))

# Persistent-index format version: bump if the pickled executor layout changes, so old
# artifacts are ignored (a stale path just misses and rebuilds; it never loads wrong data).
_BQL_INDEX_VERSION = "v3"    # v3: slim pickle, persisting only postings/field-postings/BM25;
                             # units + per-unit token caches rebuild from the corpus on load


def bql_index_path(index_root: str, key: str) -> str:
    """Where a corpus's prebuilt BQL structural index lives: the same layout as dense/pyserini
    under `index_root`, keyed by corpus. The build step (build_indexes.py) writes here;
    `load_or_build` reads here, so a corpus is prewarmed once, offline."""
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "__", key or "corpus")
    return os.path.join(index_root, "bql", f"{safe}-{_BQL_INDEX_VERSION}.pkl")


class StructuralExecutor:
    def __init__(self, units: Sequence[CodeUnit]):
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the BQL structural engine", "STRUCTURED_BACKEND=lucene with a prebuilt index")
        self.units = list(units)
        self._ubyid: dict[str, CodeUnit] = {u.doc_id: u for u in self.units}
        # Guards the O(N) lazy region-vocab build so N concurrent episodes (shared executor,
        # workers>1) build it once, not N times GIL-serialized.
        self._vocab_lock = threading.Lock()
        # per-unit token list/set + per-token line number (for line-distance NEAR)
        self._utoks: dict[str, list[str]] = {}
        self._uset: dict[str, set] = {}
        self._ulines: dict[str, list[int]] = {}
        # AST region bags and field bags are built lazily per unit on the first
        # IN(...) query that needs them: an ast.parse per unit is the dominant
        # build cost, and many episodes never use structural scope at all.
        self._region: dict[str, dict] = {}
        self._fields: dict[str, dict[Region, list[str]]] = {}
        # File-level token bags (NEAR/file, IN(file, ·)) are built lazily per path on first
        # use: most queries never use file scope, and for a large shared-document corpus
        # (one unit per path) an eager concat would be a full second copy of every token,
        # the dominant memory cost. Only the cheap path->doc_ids map is built up front.
        self._ftoks: dict[str, list[str]] = {}            # lazy cache: path -> file tokens
        self._units_by_path: dict[str, list[str]] = {}    # path -> doc_ids (refs, cheap)
        self._bm: Optional[BM25] = None        # corpus-level rerank scorer, built once
        # token -> set of unit indices, built lazily for large corpora (the prefilter)
        self._postings: Optional[dict] = None
        # per-field inverted index: {Region -> {token -> set of unit indices}} for the sparse
        # doc fields (title/section/infobox/author/date), so IN(field, x) narrows exactly.
        self._field_postings: Optional[dict] = None
        self._prefilter_on = len(self.units) >= _PREFILTER_MIN_UNITS
        # region -> union of that region's tokens across all units, built lazily on the
        # first suggest() call (the grounding 'did you mean' vocabulary). Index-free: it
        # just aggregates the per-unit bags above, costs nothing unless suggest is called.
        self._region_vocab: Optional[dict] = None
        self._global_vocab: set = set()
        # Lazy per-corpus sorted (date, unit-index) index for date-range queries (BQL v2
        # Feature 1): built on the first date-range query, not here, since most corpora/episodes
        # never issue one. Guarded by its own lock, same double-checked pattern as _region_vocab.
        self._date_lock = threading.Lock()
        self._date_keys: Optional[list] = None
        self._date_units: Optional[list] = None
        # dense-fused ranking (BQL_DENSE, default off; see bql/dense_fuse.py): an
        # agent_search.retrievers.dense.belief.DenseBelief, or None. Not persisted (excluded from
        # `_PERSIST`, the same rationale as `units`/`dense` on IndriExecutor; see
        # indri/model.py's own Deviations), so a pickle-loaded executor has `dense=None` until
        # `attach_dense(...)` is called, the same as `attach_units`.
        self.dense = None
        for u in self.units:
            toks, lns = _tok_lines(u.qualname, u.code)
            self._utoks[u.doc_id] = toks
            self._uset[u.doc_id] = set(toks)
            self._ulines[u.doc_id] = lns
            self._units_by_path.setdefault(u.path, []).append(u.doc_id)

    def _ensure_toks(self, doc_id: str) -> None:
        """Lazily tokenize and cache one unit on first access. After a slim-pkl load, attach_units
        leaves _utoks/_uset/_ulines empty: the persisted postings and BM25 already carry everything
        a whole-corpus scan needs, so only the per-query `_eval` on the prefilter's candidate subset
        actually touches per-unit tokens. Tokenizing just those candidates (sub-ms each) avoids an
        eager re-tokenize of all N units on load, which would cost several minutes for a large
        corpus. Fresh builds populate eagerly in __init__, so this is a no-op there. Assign _utoks
        last so `doc_id in self._utoks` signals all three caches are ready to a concurrent reader
        (dict writes are atomic under the GIL)."""
        if doc_id in self._utoks:
            return
        u = self._ubyid.get(doc_id)
        if u is None:
            self._uset[doc_id] = set(); self._ulines[doc_id] = []; self._utoks[doc_id] = []
            return
        toks, lns = _tok_lines(u.qualname, u.code)
        self._uset[doc_id] = set(toks)
        self._ulines[doc_id] = lns
        self._utoks[doc_id] = toks

    def _file_toks(self, path: str) -> list:
        """All tokens in a unit's file: path tokens plus every sibling unit's tokens, built
        once per path on first use and cached lazily, so a corpus whose queries never use
        file scope never pays for it."""
        toks = self._ftoks.get(path)
        if toks is None:
            toks = list(code_tokenize(path))
            for doc_id in self._units_by_path.get(path, []):
                self._ensure_toks(doc_id)
                toks += self._utoks[doc_id]
            self._ftoks[path] = toks
        return toks

    def _region_bags(self, doc_id: str) -> dict:
        bags = self._region.get(doc_id)
        if bags is None:
            u = self._ubyid[doc_id]
            bags = dict(region_token_bags(u.code))
            # A unit's own qualified name IS a definition here: a method of
            # `class RST` must match IN(def, RST) even though the class name
            # never appears in the method body (observed miss: rst.py::RST.write).
            bags["def"] = list(bags.get("def", [])) + code_tokenize(u.qualname)
            self._region[doc_id] = bags
        return bags

    def _field_bag(self, doc_id: str, region: Region) -> list:
        """Tokens of one document field for `doc_id`, built lazily and cached per (doc, field).
        `_eval` only ever needs the single field a leaf is scoped to, so this builds just that
        field, not all seven: re-tokenizing the full body on every infobox/title/section check
        would mean a query matching many docs re-tokenizes hundreds of bodies (900ms+
        single-thread, tens of seconds under 8-worker GIL contention). Per-field keeps it sub-ms."""
        cache = self._fields.get(doc_id)
        if cache is None:
            cache = self._fields[doc_id] = {}
        toks = cache.get(region)
        if toks is None:
            u = self._ubyid[doc_id]
            if region is Region.DOC:
                self._ensure_toks(doc_id)
                toks = self._utoks[doc_id]               # tokenized on demand + cached
            elif region is Region.TITLE:
                toks = code_tokenize(u.title if u.title is not None else u.qualname)
            elif region is Region.BODY:
                toks = code_tokenize(u.body if u.body is not None else u.code)
            elif region is Region.SECTION:
                toks = code_tokenize(u.section if u.section is not None else u.path)
            else:                                        # AUTHOR / DATE / INFOBOX (unit metadata)
                meta = u.metadata or {}
                key = {Region.AUTHOR: "author", Region.DATE: "date",
                       Region.INFOBOX: "infobox"}[region]
                toks = code_tokenize(str(meta.get(key, "")))
            cache[region] = toks
        return toks

    # --- grounding: region-aware near-miss suggestions ----------------------

    def _build_region_vocab(self) -> dict:
        """Union of each region's tokens across all units: the vocabulary a 'did you
        mean' draws from, keyed by Region. Reuses the same per-unit bag builders as
        matching (_region_bags / _field_bags), so it's exact and adds no second index;
        built once and cached. Only called from suggest(), so a corpus that never asks
        for suggestions never pays for it.

        Region.FILE has no per-unit bag (it spans a path's units); we approximate it with
        the global union, which is the right fallback for a 'no token anywhere' check."""
        vocab: dict[Region, set] = {}
        for u in self.units:
            region_bags = self._region_bags(u.doc_id)   # AST roles: def/call/sig/string/comment
            for region, key in _REGION_KEY.items():
                vocab.setdefault(region, set()).update(region_bags.get(key, ()))
            for region in _FIELD_REGIONS:                # title/body/section/doc/author/date/infobox
                vocab.setdefault(region, set()).update(self._field_bag(u.doc_id, region))
        # global / FILE fallback: every token seen in any region.
        glob: set = set()
        for toks in vocab.values():
            glob |= toks
        vocab[Region.FILE] = set(glob)
        vocab.setdefault(Region.DOC, set()).update(glob)
        self._global_vocab = glob
        return vocab

    def _region_vocab_for(self, region: Optional[Region]) -> set:
        """The token set a leaf is checked against: its IN(region) vocab, or the global
        union for a bare (unscoped) leaf."""
        if self._region_vocab is None:
            # Build once across concurrent episodes: without this lock N workers each rebuild the
            # whole-corpus vocab (GIL-serialized -> N x the ~minutes cost on browsecomp's 67k docs).
            lock = getattr(self, "_vocab_lock", None) or threading.Lock()
            with lock:
                if self._region_vocab is None:              # double-checked under the lock
                    self._region_vocab = self._build_region_vocab()
        if region is None:
            return self._global_vocab
        return self._region_vocab.get(region, self._global_vocab)

    def suggest(self, query: str) -> str:
        """Region-aware near-miss grounding for a (typically empty) BQL result.

        Parses `query`, walks the AST, and for each leaf token that is absent from its
        target region's vocabulary emits a 'no `foo` in region `call`; did you mean: bar'
        hint (the region is whichever IN(...) the leaf sits under, else the global scope).
        Returns "" when the query doesn't parse, when every leaf token already exists (so
        the 0 result is real, not a typo), or when nothing close is found. Capped at the
        first ~3 missing terms. Purely advisory: never raises, any failure yields ""."""
        try:
            from agent_search.retrievers.bql.parser import parse
            r = parse(query)
            if not r.ok or r.expr is None:
                return ""
            leaves = _suggest_leaves(r.expr, region=None)
            hints: list[str] = []
            seen: set = set()
            for term, region in leaves:
                tl = (term or "").strip().lower()
                if not tl or (tl, region) in seen:
                    continue
                seen.add((tl, region))
                vocab = self._region_vocab_for(region)
                if tl in vocab:
                    continue                      # token exists in scope -> not a typo
                suggestions = grounding.nearest_tokens(tl, vocab)
                hint = grounding.format_did_you_mean(
                    tl, suggestions, region=(region.value if region else None))
                if hint:
                    hints.append(hint)
                if len(hints) >= 3:               # keep the message short
                    break
            return "; ".join(hints)
        except Exception:
            return ""                             # advisory only, never break a query

    # --- public API ---------------------------------------------------------

    def run(self, expr: Expr, k: int = 100) -> list[tuple[str, float]]:
        """Boolean-filter the units, then rank the survivors deterministically."""
        return self.run_with_count(expr, k=k)[0]

    # --- offline prebuild: pay the O(N) build once, off the clock -----------

    def prewarm(self) -> "StructuralExecutor":
        """Build the lazy structures now: the inverted postings (large corpus only)
        and the corpus-level BM25 rerank scorer, so the first search_bql call in an agent
        episode doesn't pay them on the clock. Idempotent (each build is cached)."""
        if self._prefilter_on:
            self._postings_index()
            self._field_postings_index()      # per-field inverted index (title/section/infobox/author/date)
        self._corpus_bm()
        return self

    # --- slim pickling: persist only the index; rebuild per-unit caches on load ------
    # Measured on a 68k-doc corpus: postings 231MB + field_postings 4MB + BM25 770MB = ~1GB of
    # genuine index, but units + token caches (_utoks/_uset/_ulines) = ~8GB of data that rebuilds
    # from the corpus in ~5s. So this pickles only the index and re-attaches units at load time.
    _PERSIST = ("_postings", "_field_postings", "_bm", "_prefilter_on")

    def __getstate__(self) -> dict:
        st = {k: getattr(self, k, None) for k in self._PERSIST}
        st["_doc_id_order"] = [u.doc_id for u in self.units]   # ~few MB; validates the re-attach
        # Content fingerprint (agent_search.corpus.fingerprint): same doc_ids in the same
        # order can still carry different text (a unit edited in place); the doc-id-order
        # check above can't see that. attach_units re-derives this from the units it's given
        # and raises on a mismatch, exactly like the order check.
        st["_corpus_fingerprint"] = corpus_fingerprint(self.units)
        return st

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # units-derived structures are (re)built from the corpus by attach_units() (load_or_build).
        self.units = []
        self._ubyid = {}
        self._utoks = {}; self._uset = {}; self._ulines = {}
        self._units_by_path = {}
        self._ftoks = {}; self._region = {}; self._fields = {}
        self._region_vocab = None; self._global_vocab = set()
        self._vocab_lock = threading.Lock()    # not pickled (not in _PERSIST); recreated on load
        self._date_lock = threading.Lock()     # not pickled; the date index rebuilds lazily too
        self._date_keys = None; self._date_units = None
        self._units_attached = False
        self.dense = None                      # not persisted; see `attach_dense`

    def attach_dense(self, dense) -> "StructuralExecutor":
        """Attach (or detach, via `dense=None`) a `DenseBelief` post-construction, the same
        pattern as `IndriExecutor.attach_dense` (indri/model.py). `dense` is excluded from
        `_PERSIST`, so a pickle-loaded executor needs this called again after `attach_units`.
        Purely a setter: `run_with_count`/`coverage_topk` check `self.dense is not None`
        themselves, so a corpus that never attaches dense gets the same behavior as without
        dense fusion (see bql/dense_fuse.py's module docstring)."""
        self.dense = dense
        return self

    def attach_units(self, units: Sequence["CodeUnit"]) -> "StructuralExecutor":
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the BQL structural engine", "STRUCTURED_BACKEND=lucene with a prebuilt index")
        """Rebuild units + doc-id map + token caches from the corpus after a slim-pkl load. The
        persisted postings reference unit indices, so `units` must be in the same order as at
        build time, validated against the stored doc-id order; a mismatch (corpus changed)
        raises so load_or_build falls back to a fresh build rather than returning wrong hits."""
        units = list(units)
        order = getattr(self, "_doc_id_order", None)
        if order is not None and [u.doc_id for u in units] != order:
            raise ValueError("units order != persisted index order (corpus changed) — rebuild")
        stored_fp = getattr(self, "_corpus_fingerprint", None)
        if stored_fp is not None and corpus_fingerprint(units) != stored_fp:
            # Same doc_ids, same order, but different content (a unit edited in place):
            # the order check above can't see this; the content fingerprint can.
            raise ValueError("units content != persisted index fingerprint (corpus changed) — rebuild")
        self.units = units
        self._ubyid = {u.doc_id: u for u in units}
        self._utoks = {}; self._uset = {}; self._ulines = {}; self._units_by_path = {}
        # Token caches stay empty here, built lazily per candidate by _ensure_toks (the persisted
        # postings and BM25 cover every whole-corpus need; only per-query _eval touches per-unit
        # tokens). This keeps load at sub-ms per candidate instead of eagerly re-tokenizing every
        # unit. Only the cheap path->doc_ids grouping (no tokenization) is built now.
        for u in units:
            self._units_by_path.setdefault(u.path, []).append(u.doc_id)
        self._units_attached = True
        return self

    def save(self, path: str) -> str:
        """Prewarm, then persist only the index (postings, field postings, BM25) via
        __getstate__ to `path`, written atomically. `load()` reads it back so a run skips the
        build. Sibling document corpora reuse one artifact (one corpus key)."""
        self.prewarm()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                     # atomic; concurrent builders race harmlessly
        return path

    @staticmethod
    def load(path: str) -> "StructuralExecutor":
        """Load a prewarmed executor written by `save()` (a ready-to-query index).

        GC is disabled across the unpickle: this index is a huge object graph (postings = tens of
        millions of int/set objects; ~34GB live). Python's cyclic GC fires every ~700 allocations
        and each pass scans the entire live heap, so when the caller already holds a big corpus in
        memory (run_eval: 100k docs), GC-during-unpickle degrades to O(n^2) and a 44s load balloons
        to 10min+. Disabling GC for the load (re-enabled after) keeps it at the ~seconds it takes in
        isolation. No leak risk: pickle builds a tree here, not cycles that need collecting mid-load."""
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        finally:
            if gc_was_enabled:
                gc.enable()

    def _corpus_bm(self) -> BM25:
        """BM25 over the whole corpus, built once and reused across queries. The
        index-free part is candidate selection (the live Boolean scan below);
        ranking the survivors is a shared scoring layer, and its corpus
        statistics don't change per query, so the index build is paid once, not
        once per turn (a broad query over a large corpus would otherwise re-tokenize
        hundreds of big docs every call: a 300s+ per-query tail)."""
        if self._bm is None:
            for u in self.units:                 # only on a fresh build (persisted on load); ensure
                self._ensure_toks(u.doc_id)      # tokens exist even if reached via the lazy load path
            self._bm = BM25().index_tokenized(
                {u.doc_id: self._utoks[u.doc_id] for u in self.units})
        return self._bm

    def _fuse_soft(self, terms, pool: list) -> list:
        """Order a fallback pool with the same ranker the exact path uses when a dense belief is
        attached (RRF of BM25 and dense here; dense-only in `DenseOnlyStructuralExecutor`)."""
        return fuse_ranked(self.dense, " ".join(terms), pool)

    def soft_topk(self, terms, k: int = 5) -> list:
        """The graceful-degradation fallback for a 0-hit Boolean query (exact AND is brittle
        under paraphrase/obfuscation: the right doc is often close but not an exact conjunctive
        match). Returns [(doc_id, score)] best-first.

        Invariant: the fallback ranks with the same ranker, over the same index, as the exact
        path. Boolean is for filtering only; ranking is the condition's model. Without a dense
        belief that is the persisted corpus-level BM25 (`_corpus_bm`, the same scorer
        `run_with_count` orders exact matches with). With a dense belief attached (the `sieve`
        family), the pool is the union of the lexically closest `BQL_SOFT_POOL` docs and the
        dense side's own nearest neighbours, read from the persisted embedding cache, never
        encoded online, ordered by the fusion rule (`_fuse_soft`). A fallback can therefore
        never rank by a model the exact path does not use."""
        terms = list(terms)
        pool_n = max(k, _SOFT_POOL)
        scored = self._corpus_bm().score_terms(terms)
        pool = [(d, s) for d, s in scored[:pool_n] if s > 0]
        if self.dense is not None:
            try:
                dense_top = list(self.dense.top_k_doc_ids(" ".join(terms), k=pool_n) or [])
            except Exception:  # noqa: BLE001, a dense-side failure degrades to the BM25 pool
                dense_top = []
            have = {d for d, _ in pool}
            pool += [(d, 0.0) for d in dense_top if d not in have and d in self._ubyid]
            if pool:
                pool = self._fuse_soft(terms, pool)
        return pool[:k]

    def coverage_topk(self, expr: Expr, k: int = 5) -> list:
        """Constraint-coverage ranking for a 0-hit AND (BQL v2 Feature 2): a hard 6-term
        conjunction that matches nothing tells the agent nothing about which clause is the
        problem. Rank the corpus by (# of the AND's children matched DESC, BM25 over the
        query's positive terms DESC) instead, so a doc satisfying 5 of 6 constraints outranks
        one satisfying 2, pinpointing which constraint to loosen or drop, versus `soft_topk`'s
        undifferentiated bag-of-terms relevance.

        Returns [(doc_id, matched_mask, n_matched, score)], mask aligned with the AND's
        children (True at index i iff unit._eval-matches children[i]; a Not(...) child counts
        as matched when the negation holds, since `_eval` already inverts it). A non-AND `expr`
        (single constraint) has nothing to break down, so it degrades to `soft_topk` with a
        1-element mask (True) per hit, since the whole point is showing some relevance
        ordering even where the exact clause doesn't literally match."""
        if not isinstance(expr, And):
            terms = _rank_leaves(expr)
            return [(d, (True,), 1, s) for d, s in self.soft_topk(terms, k=k)]
        children = list(expr.children)
        # Recall-safe per-child candidate supersets (the same machinery run_with_count uses),
        # unioned into a pool of "docs matching >=1 child", narrower than the whole corpus
        # whenever at least one child is narrowable (an unnarrowable child, e.g. Prefix, or a
        # Not, safely widens the pool to everything rather than dropping a possible match).
        pool = _union(self._candidates(c) for c in children)
        pool_idx = set(pool) if pool is not None else set(range(len(self.units)))
        terms = _rank_leaves(expr)
        if len(pool_idx) > _COVERAGE_POOL_CAP:
            # Huge pool (e.g. every child unnarrowable): bound it to the top-scoring docs by
            # BM25 over the query's own terms before paying the per-(doc, child) exact _eval.
            doc_ids = [self.units[i].doc_id for i in pool_idx]
            top = self._corpus_bm().score_subset(terms, doc_ids)[:_COVERAGE_POOL_CAP]
            keep = {d for d, _ in top}
            pool_idx = {i for i in pool_idx if self.units[i].doc_id in keep}
        rows = []                                # (doc_id, mask, n_matched)
        for i in pool_idx:
            u = self.units[i]
            self._ensure_toks(u.doc_id)
            toks, tset = self._utoks[u.doc_id], self._uset[u.doc_id]
            mask = tuple(self._eval(c, toks, tset, u) for c in children)
            n = sum(mask)
            if n:                                 # 0-matched docs are noise, not signal: drop
                rows.append((u.doc_id, mask, n))
        if not rows:
            return []
        scores = dict(self._corpus_bm().score_subset(terms, [r[0] for r in rows]))
        rows = [(d, mask, n, scores.get(d, 0.0)) for d, mask, n in rows]
        rows.sort(key=lambda r: (-r[2], -r[3], r[0]))        # coverage desc, then BM25, then id
        if self.dense is not None:
            # BQL_DENSE dense-fused ranking: RRF bm25 x dense within each coverage tier only,
            # so a doc can never be promoted past a doc with more matched constraints (see
            # bql/dense_fuse.py's "Coverage tiers" section). `terms` is the same positive leaf
            # bag the BM25 side above just scored with; the dense query text mirrors it (no
            # raw surface query text is threaded this far; see run_with_count's own comment).
            rows = fuse_coverage_tiers(self.dense, " ".join(terms), rows)
        return rows[:k]

    def run_with_count(self, expr: Expr, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """Return the top-k ranked hits plus the untruncated match count.

        Boolean retrieval returns a set; rank metrics need an order. The matched
        set is ordered by a BM25 scorer over the query's positive leaf terms
        (GrepRAG-style rerank): index-free selection (the live scan), shared
        scoring. Uses corpus-level idf (built once, reused) rather than rebuilding
        per-candidate-set stats each call; zero-score candidates (matched via file
        scope / qualname) stay ranked, after scored ones.
        """
        # Selection: scan the units and keep those that _eval-match. For a large
        # corpus this is the inverted-index prefilter (a recall-safe candidate subset),
        # so this only _evals candidates, not all N units (bql_spec.md §6, filter-then-
        # verify). Small corpora (per-query code repos) scan everything: same result,
        # the index-free path. _eval is the exact arbiter either way, so the prefilter
        # never changes which units match, only how many there are to test.
        leaves = _rank_leaves(expr)
        scan = self._candidate_units(expr)
        matched = []
        for u in scan:                          # tokenize only the candidates the prefilter kept
            self._ensure_toks(u.doc_id)
            if self._eval(expr, self._utoks[u.doc_id], self._uset[u.doc_id], u):
                matched.append(u.doc_id)
        if not matched:
            return [], 0
        ranked = self._corpus_bm().score_subset(leaves, matched)
        if self.dense is not None:
            # BQL_DENSE dense-fused ranking (default off; see bql/dense_fuse.py): `matched` is
            # the full exact-match filter-passing set (every doc here satisfied the whole
            # boolean/field/date filter, one coverage tier by construction), so this is a
            # single-tier RRF fuse of the bm25 ranking above with dense similarity restricted
            # to these same doc_ids (never a global dense search; see dense_fuse.py's "Why
            # dense scoring is restricted..." section: the filter's candidate set is the only
            # thing that can ever appear in `ranked`, before or after fusion). Dense query text
            # mirrors `leaves` (the same positive terms the BM25 side just scored with); no
            # raw surface query string is threaded down to the executor.
            ranked = fuse_ranked(self.dense, " ".join(leaves), ranked)
        return ranked[:k], len(matched)

    # --- prefilter (inverted-index candidate set; filter-then-verify) -------

    def _postings_index(self) -> dict:
        """token -> set of unit indices, built once and reused. ~tens of MB to a few
        hundred MB for a 100k-doc corpus (sets of ints); only built when the corpus is
        large enough that the prefilter is worth it."""
        if self._postings is None:
            post: dict = {}
            for i, u in enumerate(self.units):
                self._ensure_toks(u.doc_id)      # fresh-build path has eager tokens; no-op there
                toks = set(self._uset[u.doc_id])
                toks.update(code_tokenize(u.path))    # the `section` field falls back to path
                # Index the document field bags too, so IN(title/body/section, x) can
                # narrow safely even if a field carries a token not in qualname+code
                # (units_from_documents folds fields into code, but a custom CodeUnit
                # may not). DOC's bag is already _uset. _eval still verifies exactly.
                for field in (u.title, u.section, u.body):
                    if field:
                        toks.update(code_tokenize(field))
                for tok in toks:
                    post.setdefault(tok, set()).add(i)
            self._postings = post
        return self._postings

    def _field_postings_index(self) -> dict:
        """{Region -> {token -> set of unit indices}} for the sparse doc fields
        (_INDEXED_FIELDS). Built once (offline in prewarm, persisted in the pkl). Tokenizes only
        those fields directly from each unit, using the same source resolution as `_field_bags`,
        so a candidate from the postings is guaranteed to match `_eval`'s field check (no recall
        loss), and not the big body/doc bags (those stay on the combined index). This pays the
        O(N) cost that would otherwise run per query on every infobox/author/date scan once,
        here. At query time IN(field, x) is a dict lookup plus a set intersection (sub-ms); a
        miss is empty (instant) instead of a full O(N) scan that builds every doc's field bags
        on the clock."""
        if self._field_postings is None:
            fp: dict = {f: {} for f in _INDEXED_FIELDS}
            for i, u in enumerate(self.units):
                meta = u.metadata or {}
                src = {
                    Region.TITLE: u.title if u.title is not None else u.qualname,
                    Region.SECTION: u.section if u.section is not None else u.path,
                    Region.INFOBOX: str(meta.get("infobox", "")),
                    Region.AUTHOR: str(meta.get("author", "")),
                    Region.DATE: str(meta.get("date", "")),
                }
                for f in _INDEXED_FIELDS:
                    d = fp[f]
                    for tok in set(code_tokenize(src[f])):
                        d.setdefault(tok, set()).add(i)
            self._field_postings = fp
        return self._field_postings

    def _ensure_date_index(self) -> tuple:
        """Lazy per-corpus sorted (date-string, unit-index) index for date-range queries,
        built on first use and cached: ~sub-second for 100k docs (one regex + one sort).
        Units with a missing/malformed date are excluded (they can never match a range).
        Rebuilds correctly after a slim-pkl load: it isn't persisted (not in `_PERSIST`), so
        it's just rebuilt from `self.units` here, same as the other lazy per-unit caches."""
        if self._date_keys is None:
            with self._date_lock:
                if self._date_keys is None:                  # double-checked under the lock
                    pairs = []
                    for i, u in enumerate(self.units):
                        d = _unit_date(u)
                        if d is not None:
                            pairs.append((d, i))
                    pairs.sort(key=lambda p: p[0])            # ISO strings sort chronologically
                    self._date_keys = [d for d, _ in pairs]
                    self._date_units = [i for _, i in pairs]
        return self._date_keys, self._date_units

    def _date_range_indices(self, lo: Optional[str], hi: Optional[str]) -> set:
        """Exact unit-index set matching `[lo, hi]` (either bound may be open), via binary
        search over the lazy sorted date index: the range narrows instead of scanning."""
        keys, units_idx = self._ensure_date_index()
        lo_pos = bisect.bisect_left(keys, lo) if lo is not None else 0
        hi_pos = bisect.bisect_right(keys, hi) if hi is not None else len(keys)
        return set(units_idx[lo_pos:hi_pos]) if hi_pos > lo_pos else set()

    def _candidate_units(self, expr: Expr) -> list:
        """The units _eval must test: a recall-safe candidate subset from the inverted
        index for a large corpus, else all units (the plain live scan)."""
        if not self._prefilter_on:
            return self.units
        cand = self._candidates(expr)
        return self.units if cand is None else [self.units[i] for i in cand]

    def _candidates(self, expr: Expr, field: Optional[Region] = None) -> Optional[set]:
        """Recall-safe superset of unit indices that could match `expr`, from the inverted
        index. ``None`` means "can't narrow, treat as the whole corpus". `field` threads the
        enclosing ``IN(field, ...)`` scope down to the leaf terms, so a term resolves against
        that field's own postings (title/section/infobox/author/date) instead of the corpus-wide
        postings, turning an infobox/author/date query from a full O(N) scan into an exact
        postings lookup (empty -> instant). Looser than ``_eval`` (ignores token position and
        negation), so every true match is in the returned set; ``_eval`` verifies exactly."""
        # postings source for a leaf under the current field scope: the field's own inverted
        # index when it's one of the sparse indexed fields, else the combined corpus postings
        # (BODY/DOC/unscoped: the combined index already covers the doc's full text tightly).
        post = (self._field_postings_index()[field] if field in _INDEXED_FIELDS
                else self._postings_index())
        if isinstance(expr, Term):
            return _isect(post.get(t, set()) for t in code_tokenize(expr.text))
        if isinstance(expr, Phrase):
            return _isect(post.get(t, set())
                          for term in expr.terms for t in code_tokenize(term.text))
        if isinstance(expr, (Prefix, Expand)):
            return None                             # prefix family: don't narrow; siblings do
        if isinstance(expr, And):                   # all positive clauses must match -> intersect
            return _isect(self._candidates(c, field) for c in expr.children if not isinstance(c, Not))
        if isinstance(expr, Or):                    # any clause may match -> union
            return _union(self._candidates(c, field) for c in expr.children)
        if isinstance(expr, Not):
            return None                             # negation can't narrow (only the AND siblings do)
        if isinstance(expr, Near):
            if expr.spec == "file":
                return None                         # file scope spans the file's units
            return _isect((self._candidates(expr.left, field), self._candidates(expr.right, field)))
        if isinstance(expr, In):
            # typed date-range leaf (BQL v2 Feature 1): resolve via the lazy sorted date index
            # (binary search, the exact match set) instead of the generic field-token postings
            # path below. The range's encoded term text (e.g. "__daterange__1980-01-01__...")
            # tokenizes into sub-tokens (year/month/day) whose AND-intersection would almost
            # never hold for a real single-date field, silently narrowing to empty.
            if expr.region == Region.DATE and isinstance(expr.child, Term):
                rng = _parse_daterange_term(expr.child.text)
                if rng is not None:
                    return self._date_range_indices(*rng)
            # narrow via the field's inverted index for any doc field (incl. infobox/author/date).
            if expr.region in _FIELD_REGIONS:
                return self._candidates(expr.child, field=expr.region)
            # FILE (term may sit in a sibling unit) and the code AST regions (def/call/sig: lazy
            # ast.parse bags; `sig` holds ast.unparse-normalized tokens absent from source) can't
            # be narrowed from token postings, so this returns None (scan; small per-repo code
            # corpora).
            if expr.region == Region.FILE or expr.region in _REGION_KEY:
                return None
            # No other case exists: `_FIELD_REGIONS` (7) + `_REGION_KEY` (5) + FILE (1) is
            # every member of the 13-value `Region` enum, with no overlap, so this `In`
            # branch is exhaustive and always returns above.
        return None

    # --- evaluation ---------------------------------------------------------

    def _eval(self, expr: Expr, toks: list, tset: set, unit: CodeUnit) -> bool:
        if isinstance(expr, Term):
            return _term_in(expr.text, tset, toks)
        if isinstance(expr, Prefix):
            stem = expr.stem.lower()
            return any(t.startswith(stem) for t in tset)
        if isinstance(expr, Expand):
            stem = expr.term.text.lower()
            return any(t == stem or t.startswith(stem) for t in tset)
        if isinstance(expr, Phrase):
            return _phrase_in([t.text for t in expr.terms], toks)
        if isinstance(expr, And):
            pos = [c for c in expr.children if not isinstance(c, Not)]
            neg = [c for c in expr.children if isinstance(c, Not)]
            if not all(self._eval(c, toks, tset, unit) for c in pos):
                return False
            return not any(self._eval(n.child, toks, tset, unit) for n in neg)
        if isinstance(expr, Or):
            return any(self._eval(c, toks, tset, unit) for c in expr.children)
        if isinstance(expr, Not):              # top-level NOT (typecheck forbids it)
            return not self._eval(expr.child, toks, tset, unit)
        if isinstance(expr, Near):
            if expr.spec == "file":             # same FILE, not same unit
                ftoks = self._file_toks(unit.path)
                fset = set(ftoks)
                return (self._eval(expr.left, ftoks, fset, unit)
                        and self._eval(expr.right, ftoks, fset, unit))
            if expr.spec in _COOCCUR_SPECS or expr.gran.value >= Granularity.FUNC.value:
                return (self._eval(expr.left, toks, tset, unit)
                        and self._eval(expr.right, toks, tset, unit))
            return self._near_window(expr, toks, tset, unit)
        if isinstance(expr, In):
            if expr.region == Region.FILE:
                ftoks = self._file_toks(unit.path)
                return self._eval(expr.child, ftoks, set(ftoks), unit)
            if expr.region == Region.DATE and isinstance(expr.child, Term):
                rng = _parse_daterange_term(expr.child.text)
                if rng is not None:                   # typed date-range leaf (BQL v2 Feature 1)
                    lo, hi = rng
                    d = _unit_date(unit)
                    return d is not None and (lo is None or d >= lo) and (hi is None or d <= hi)
            key = _REGION_KEY.get(expr.region)
            if key is not None:                       # real AST structural scope
                rtoks = self._region_bags(unit.doc_id)[key]
                return self._eval(expr.child, rtoks, set(rtoks), unit)
            if expr.region in _FIELD_REGIONS:
                ftoks = self._field_bag(unit.doc_id, expr.region)
                return self._eval(expr.child, ftoks, set(ftoks), unit)
            return self._eval(expr.child, toks, tset, unit)  # title/body/... ~ unit text
        return False

    def _near_window(self, near: Near, toks: list, tset: set, unit: CodeUnit) -> bool:
        win = _int_suffix(near.spec)
        lpos = _positions(near.left, toks)
        rpos = _positions(near.right, toks)
        if lpos is None or rpos is None:        # complex operands -> co-occurrence
            return (self._eval(near.left, toks, tset, unit)
                    and self._eval(near.right, toks, tset, unit))
        if near.spec.startswith("line"):        # NEAR/lineN -> within N source lines
            if toks is self._utoks.get(unit.doc_id):
                lns = self._ulines[unit.doc_id]
                return any(abs(lns[i] - lns[j]) <= win for i in lpos for j in rpos)
            # No per-token line numbers at this scope (the operand was bound by
            # IN(file, ...) / IN(region, ...), so `toks` is file/region tokens, not the
            # unit's own stream). A line window can't be measured here -> degrade to
            # co-occurrence (both operands present in scope) rather than silently
            # measuring TOKEN distance, which would be wrong.
            return bool(lpos) and bool(rpos)
        return any(abs(i - j) <= win for i in lpos for j in rpos)  # NEAR/wN -> N tokens

# --- helpers ----------------------------------------------------------------

def _parse_daterange_term(text: str) -> Optional[tuple]:
    """`term.text` -> (lo, hi) if it's a surface-emitted `__daterange__LO__HI` encoding
    (either bound may be None = open), else None (an ordinary date-field term)."""
    m = _DATE_RANGE_TERM_RE.match(text)
    if not m:
        return None
    lo = None if m.group(1) == "open" else m.group(1)
    hi = None if m.group(2) == "open" else m.group(2)
    return lo, hi


def _unit_date(u: CodeUnit) -> Optional[str]:
    """`unit.metadata['date']` normalized to 'YYYY-MM-DD', or None if missing/malformed/not
    a real calendar date. A leading ISO date is enough (a full timestamp's time-of-day suffix
    is ignored); anything that doesn't start with one never matches a date-range query."""
    meta = u.metadata or {}
    raw = meta.get("date")
    if not raw:
        return None
    m = _ISO_DATE_PREFIX_RE.match(str(raw))
    if not m:
        return None
    y, mo, d = (int(g) for g in m.groups())
    try:
        import datetime as _dt
        _dt.date(y, mo, d)                  # reject calendar-invalid strings (e.g. 2023-02-30)
    except ValueError:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _isect(sets) -> Optional[set]:
    """Intersection for the prefilter, where None means 'the whole corpus' (no
    constraint, skipped). Returns None only if every input was None. An empty set is a
    real constraint (a required token that no unit has) -> empty result.

    Smallest-first: sort the postings by size and intersect from the smallest, so a
    rare term (tiny set) drives the work and a huge common postings list is never
    copied or iterated (the standard inverted-index intersection order). The returned set is
    always a fresh copy the caller owns, never a live reference into the postings index
    (with only one input, `&` never runs, so this must copy explicitly or a caller
    mutating the result would corrupt the shared postings set)."""
    present = [s for s in sets if s is not None]
    if not present:
        return None
    present.sort(key=len)
    out = present[0]
    copied = False
    for s in present[1:]:
        if not out:
            break                                 # already empty -> done
        out = out & s                             # a fresh, ever-smaller set
        copied = True
    return out if copied else set(out)            # single-input case: never alias postings


def _union(sets) -> Optional[set]:
    """Union for the prefilter, where None means 'the whole corpus' -> any None makes
    the whole union None (can't narrow)."""
    out: set = set()
    for s in sets:
        if s is None:
            return None
        out |= s
    return out


def _tok_lines(qualname: str, code: str) -> tuple:
    """Tokens of (qualname + code) with a parallel per-token source-line number
    (qualname tokens are line 0; code tokens are 1-based within the unit)."""
    toks: list = []
    lns: list = []
    for t in code_tokenize(qualname):
        toks.append(t)
        lns.append(0)
    for i, line in enumerate(code.splitlines(), start=1):
        for t in code_tokenize(line):
            toks.append(t)
            lns.append(i)
    return toks, lns


def _seq_in(seq: list, toks: list) -> bool:
    """True if `seq` appears as a contiguous subsequence of `toks`."""
    n = len(seq)
    if n == 0:
        return False
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == seq:
            return True
    return False


def _term_in(text: str, tset: set, toks: list) -> bool:
    sub = code_tokenize(text)
    if not sub:
        return False
    if len(sub) == 1:
        return sub[0] in tset
    return _seq_in(sub, toks)   # multi-token term (identifier/quoted) -> adjacency


def _phrase_in(terms: list, toks: list) -> bool:
    return _seq_in([s for t in terms for s in code_tokenize(t)], toks)


def _seq_positions(seq: list, toks: list) -> Optional[list]:
    """All indices covered by contiguous occurrences of `seq` in `toks`: window
    distance then measures to the nearest part of the match, not just
    its first token."""
    n = len(seq)
    if n == 0:
        return None
    out: list = []
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == seq:
            out.extend(range(i, i + n))
    return sorted(set(out)) or None


def _positions(expr: Expr, toks: list) -> Optional[list]:
    """Token positions where this operand matches; None = not positionable.

    Window NEAR (`/wN`, `/lineN`) must honor its operand's full match
    semantics, or the window silently degrades to unit co-occurrence (observed:
    NEAR/w2(OR(a,b), c) matched with every pair far apart). Positionable:
    Term (incl. multi-token: start index of each contiguous match), Phrase,
    Prefix, Expand (prefix-consistent with its _eval), and Or (union of child
    positions). And/Not/In/Near operands stay None -> co-occurrence fallback.
    """
    if isinstance(expr, Or):
        out: list = []
        for c in expr.children:
            p = _positions(c, toks)
            if p is None:
                # a child with no occurrences contributes nothing; a child that
                # is structurally unpositionable forces the fallback. Telling
                # those apart needs the child's own match test, so be exact:
                if isinstance(c, (Term, Phrase, Prefix, Expand)):
                    continue            # positionable kind, just absent here
                return None
            out.extend(p)
        return sorted(set(out)) or None
    if isinstance(expr, Prefix):
        stem = expr.stem.lower()
        return [i for i, t in enumerate(toks) if t.startswith(stem)] or None
    if isinstance(expr, Expand):
        sub = code_tokenize(expr.term.text)
        if len(sub) != 1:
            return None
        s = sub[0]
        # prefix-consistent with Expand's _eval (t == s or t.startswith(s))
        return [i for i, t in enumerate(toks) if t == s or t.startswith(s)] or None
    if isinstance(expr, Phrase):
        return _seq_positions([s for t in expr.terms for s in code_tokenize(t.text)], toks)
    if isinstance(expr, Term):
        sub = code_tokenize(expr.text)
        if not sub:
            return None
        if len(sub) == 1:
            return [i for i, t in enumerate(toks) if t == sub[0]] or None
        return _seq_positions(sub, toks)
    return None


def _int_suffix(spec: str, default: int = 5) -> int:
    digits = "".join(c for c in spec if c.isdigit())
    return int(digits) if digits else default


def _suggest_leaves(expr: Expr, region: Optional[Region]) -> list:
    """Collect (token, target_region) pairs for grounding: every constituent token of each
    leaf, tagged with the region it must exist in. A leaf under IN(R, ...) targets R; a bare
    leaf targets None (the global scope). NOT subtrees are skipped, since an absent excluded
    token is not a typo to suggest.

    Checking every token (not just the head) is what catches the common compound miss: the
    agent guesses `polygon_to_masks` but the symbol is `polygon_to_mask`. The head `polygon`
    exists, so only the trailing `masks`->`mask` reveals the typo. Tokens that do exist are
    dropped later in suggest(), so this only ever surfaces the genuinely-absent ones.

    `region` threads the current IN-scope down the walk; nested IN takes the innermost."""
    out: list = []
    if isinstance(expr, Term):
        out.extend((t, region) for t in code_tokenize(expr.text))
    elif isinstance(expr, Phrase):
        out.extend((t, region) for term in expr.terms for t in code_tokenize(term.text))
    elif isinstance(expr, Prefix):
        out.append((expr.stem, region))
    elif isinstance(expr, Expand):
        out.extend((t, region) for t in code_tokenize(expr.term.text))
    elif isinstance(expr, Not):
        return out                                # excluded terms aren't typos to ground
    elif isinstance(expr, In):
        out.extend(_suggest_leaves(expr.child, expr.region))   # innermost IN wins
    elif isinstance(expr, (And, Or)):
        for c in expr.children:
            out.extend(_suggest_leaves(c, region))
    elif isinstance(expr, Near):
        out.extend(_suggest_leaves(expr.left, region))
        out.extend(_suggest_leaves(expr.right, region))
    return out


def _rank_leaves(expr: Expr) -> list:
    """Collect positive ranking terms (skip NOT subtrees)."""
    out: list = []

    def walk(e: Expr) -> None:
        if isinstance(e, Term):
            out.extend(code_tokenize(e.text))
        elif isinstance(e, Prefix):
            out.append(e.stem.lower())
        elif isinstance(e, Expand):
            out.extend(code_tokenize(e.term.text))
        elif isinstance(e, Phrase):
            for t in e.terms:
                out.extend(code_tokenize(t.text))
        elif isinstance(e, Not):
            return                              # exclusions don't add to score
        elif isinstance(e, And):
            for c in e.children:
                walk(c)
        elif isinstance(e, Or):
            for c in e.children:
                walk(c)
        elif isinstance(e, Near):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, In):
            walk(e.child)

    walk(expr)
    return out


# --- agent-facing helpers: parse+execute a BQL string -> Observation --------

def hits_from_ranked(ranked, units_by_id: dict):
    from agent_search.retrievers.base import Hit
    from agent_search.tokens import cap_tokens
    hits = []
    for doc_id, score in ranked:
        u = units_by_id.get(doc_id)
        snippet = cap_tokens(u.code.splitlines()[0], 16) if (u and u.code) else ""
        hits.append(Hit(doc_id=doc_id, score=score,
                        path=(u.path if u else None),
                        line=(u.start_line if u else None), snippet=snippet))
    return hits


def execute_bql(bql: str, executor, units_by_id: dict, k: int = 100):
    """Parse -> typecheck -> execute. Errors come back as Observations."""
    from agent_search.retrievers.base import Observation
    from agent_search.retrievers.bql.parser import parse
    from agent_search.retrievers.bql.types import check
    r = parse(bql)
    if not r.ok:
        return Observation(n_hits=0, hits=[], typecheck_ok=False, error=f"parse error: {r.error}")
    t = check(r.expr)
    if not t.ok:
        return Observation(n_hits=0, hits=[], typecheck_ok=False, error=f"type error: {t.error}")
    if hasattr(executor, "run_with_count"):
        ranked, n_hits = executor.run_with_count(r.expr, k=k)
    else:
        ranked = executor.run(r.expr, k=k)
        n_hits = len(ranked)
    return Observation(n_hits=n_hits, hits=hits_from_ranked(ranked, units_by_id), typecheck_ok=True)


def load_or_build(units, index_root: Optional[str] = None, key: Optional[str] = None,
                  rebuild: bool = False, dense=None) -> "StructuralExecutor":
    """Return a prewarmed executor loaded from disk if `build_indexes.py` materialized one
    for this corpus, else build it in memory. This is what makes `search_bql` load a
    prebuilt index instead of building it inside the first episode. A missing, corrupt or
    stale artifact silently falls back to a fresh build, so correctness never depends on
    the cache, only speed does.

    `dense` (an `agent_search.retrievers.dense.belief.DenseBelief`, or None; BQL_DENSE,
    default off) is attached after load/build either way, the same as `build_indri_engine`'s
    own `dense=` forwarding: it's never persisted (excluded from `_PERSIST`), so a
    freshly-loaded pickle's `.dense` is None until this attaches it, same as a fresh
    in-memory build."""
    if index_root and key and not rebuild:
        path = bql_index_path(index_root, key)
        if os.path.exists(path):
            try:
                ex = StructuralExecutor.load(path)
                ex.attach_units(units)     # slim pkl stores only the index; rebuild per-unit caches (~5s)
                return ex.attach_dense(dense)
            except Exception:
                pass                       # corrupt/stale/order-mismatch pickle -> rebuild from units
    # No prebuilt artifact: build in memory, but persist it so this O(N) build (e.g. ~45 min over
    # browsecomp's 100k docs) is paid once, not re-paid by every condition or restart.
    # Best-effort and atomic: a failed or raced save just means the next run rebuilds, never wrong.
    ex = StructuralExecutor(units)
    if index_root and key:
        try:
            ex.save(bql_index_path(index_root, key))
        except Exception:
            pass
    return ex.attach_dense(dense)


class BQLIndexBuilder:
    """Offline persister for the BQL structural index, so `build_indexes.py` can pre-build
    it exactly like dense/pyserini: `.index(units, key)` writes a prewarmed executor to
    disk; `.is_cached(key)` reports whether it already exists (skip the corpus parse).
    `load_or_build` reads it back. Keyed by corpus, so a shared document corpus is built
    once and reused across all its queries."""
    name = "search_bql"

    def __init__(self, index_root: str = "indexes", rebuild: bool = False):
        self.index_root = index_root
        self.rebuild = rebuild

    def is_cached(self, key: str) -> bool:
        return os.path.exists(bql_index_path(self.index_root, key))

    def index(self, units, key: Optional[str] = None) -> "BQLIndexBuilder":
        StructuralExecutor(list(units)).save(bql_index_path(self.index_root, key))
        return self


# --- Dense-only executor (`sieve_dense` / `sieve_visit_dense`, aliased as
# --- `research_bql_donly_snip` / `research_bql_donly_visit`)
#
# The candidate-ordering twin of `StructuralExecutor` (every method, attribute, and call site
# above this class is untouched): `run_with_count`/`coverage_topk` are overridden (their bodies
# are duplicated, not edited in place, since the parent methods are used by every other
# BQL-family condition) so that when a `DenseBelief` is attached, the order of filter-passing
# candidates comes purely from `dense_rank_for_candidates`
# (`fuse_ranked_dense_only`/`fuse_coverage_tiers_dense_only`, dense_fuse.py) instead of RRF
# (`fuse_ranked`/`fuse_coverage_tiers`). Candidate selection, everything above the `if
# self.dense is not None:` line in each method, is copied verbatim, so the filter/coverage-tier
# semantics can never drift from the parent class.
class DenseOnlyStructuralExecutor(StructuralExecutor):
    """The executor `sieve_dense`/`sieve_visit_dense` use: identical boolean/field/date
    filter and coverage-tier structure to `StructuralExecutor`. A document that fails the filter
    can never appear, and a doc can never cross a coverage-tier boundary, exactly as in the
    parent class. The only difference is the order of filter-passing candidates once a
    `DenseBelief` is attached (`self.dense is not None`, set via the inherited `attach_dense`,
    never overridden here): purely `dense_rank_for_candidates`, not RRF(bm25, dense). See
    `load_or_build_dense_only` below for how a real (persisted-index-backed) instance of this
    class is constructed, and `dense_fuse.py`'s "Dense-only ordering" section for the fusion
    functions."""

    def _fuse_soft(self, terms, pool: list) -> list:
        """Dense-only ordering of the 0-hit fallback pool: the same ranker as this executor's
        exact path, so the fallback never ranks by a model the exact path does not use."""
        return fuse_ranked_dense_only(self.dense, " ".join(terms), pool)

    def run_with_count(self, expr: Expr, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """Dense-only sibling of `StructuralExecutor.run_with_count`: selection logic (leaves,
        `_candidate_units`, `_eval`, the bm25 rerank of the exact-match set) copied verbatim;
        only the final fuse call differs (`fuse_ranked_dense_only` instead of `fuse_ranked`)."""
        leaves = _rank_leaves(expr)
        scan = self._candidate_units(expr)
        matched = []
        for u in scan:
            self._ensure_toks(u.doc_id)
            if self._eval(expr, self._utoks[u.doc_id], self._uset[u.doc_id], u):
                matched.append(u.doc_id)
        if not matched:
            return [], 0
        ranked = self._corpus_bm().score_subset(leaves, matched)
        if self.dense is not None:
            # Dense-only: the same single-tier filter-passing candidate set `run_with_count`
            # always had; only the order is now pure dense rank, not RRF(bm25, dense).
            # See dense_fuse.py.
            ranked = fuse_ranked_dense_only(self.dense, " ".join(leaves), ranked)
        return ranked[:k], len(matched)

    def coverage_topk(self, expr: Expr, k: int = 5) -> list:
        """Dense-only sibling of `StructuralExecutor.coverage_topk`: pool/candidate/mask/
        coverage-tier-sort logic copied verbatim; only the final fuse call differs
        (`fuse_coverage_tiers_dense_only` instead of `fuse_coverage_tiers`)."""
        if not isinstance(expr, And):
            terms = _rank_leaves(expr)
            return [(d, (True,), 1, s) for d, s in self.soft_topk(terms, k=k)]
        children = list(expr.children)
        pool = _union(self._candidates(c) for c in children)
        pool_idx = set(pool) if pool is not None else set(range(len(self.units)))
        terms = _rank_leaves(expr)
        if len(pool_idx) > _COVERAGE_POOL_CAP:
            doc_ids = [self.units[i].doc_id for i in pool_idx]
            top = self._corpus_bm().score_subset(terms, doc_ids)[:_COVERAGE_POOL_CAP]
            keep = {d for d, _ in top}
            pool_idx = {i for i in pool_idx if self.units[i].doc_id in keep}
        rows = []                                # (doc_id, mask, n_matched)
        for i in pool_idx:
            u = self.units[i]
            self._ensure_toks(u.doc_id)
            toks, tset = self._utoks[u.doc_id], self._uset[u.doc_id]
            mask = tuple(self._eval(c, toks, tset, u) for c in children)
            n = sum(mask)
            if n:                                 # 0-matched docs are noise, not signal: drop
                rows.append((u.doc_id, mask, n))
        if not rows:
            return []
        scores = dict(self._corpus_bm().score_subset(terms, [r[0] for r in rows]))
        rows = [(d, mask, n, scores.get(d, 0.0)) for d, mask, n in rows]
        rows.sort(key=lambda r: (-r[2], -r[3], r[0]))        # coverage desc, then BM25, then id
        if self.dense is not None:
            # Dense-only: the same coverage-tier boundaries as `coverage_topk` (a doc can
            # never cross a tier); only the within-tier order is now pure dense rank, not
            # RRF(bm25, dense). See dense_fuse.py.
            rows = fuse_coverage_tiers_dense_only(self.dense, " ".join(terms), rows)
        return rows[:k]


def load_or_build_dense_only(units, index_root: Optional[str] = None, key: Optional[str] = None,
                             rebuild: bool = False, dense=None) -> "DenseOnlyStructuralExecutor":
    """Executor constructor for `sieve_dense`/`sieve_visit_dense`: reuses `load_or_build`
    verbatim (the same on-disk `indexes/bql/<key>...pkl` index, the same `attach_units`/
    `attach_dense` contract, the same in-memory-build-and-persist fallback; nothing about
    loading, building or persisting the index is reimplemented here), then swaps the
    already-constructed instance's class to `DenseOnlyStructuralExecutor`. This is a pure
    `__class__` reassignment on an existing object, not a second construction path, so the
    dense-fused BQL conditions (RRF) and the dense-only conditions share the exact same
    persisted index artifact and only differ in which class's `run_with_count`/
    `coverage_topk` answers a query."""
    ex = load_or_build(units, index_root=index_root, key=key, rebuild=rebuild, dense=dense)
    ex.__class__ = DenseOnlyStructuralExecutor
    return ex
