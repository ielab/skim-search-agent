"""Indri retrieval model: Dirichlet-smoothed scoring over the `indri` backend's index.

Source of truth: `agent_search/tools/search_indri/indri_doc.md` (belief-combination math, Dirichlet
formula, operator semantics). See `parser.py` for the AST and its own Deviations
section (parsing-only divergences). This module implements the scorer: leaf beliefs
via Dirichlet-smoothed query likelihood, belief operators combining child log-beliefs
per the reference's formulas, field-scoped counting/smoothing, filters (`#filreq`/
`#filrej`/date operators) as per-document hard gates, and a bounded candidate-pool
ranking pass.

## Deviations (from agent_search/tools/search_indri/indri_doc.md)
- **Unseen-term smoothing epsilon**: official Indri discounts collection probability
  for OOV terms via its own scheme. This backend uses `P(t|C) = max(cf, 0.5) / total`:
  an unseen term (`cf == 0`) is treated as if it occurred exactly 0.5 times in the
  collection. This keeps `log(...)` finite (never `-inf` from smoothing alone) without
  needing a real OOV model, at the cost of an arbitrary constant (0.5 is the standard
  "half occurrence" Laplace-style choice).
- **Window/synonym pseudo-term collection stats**: `#odN`/`#uwN` don't have a natural
  single `cf` (they're not a vocabulary term). We approximate
  `cf(window) = min(cf(member) for member in children)`: a window can occur at most
  as often as its rarest constituent, so this is a defensible (if loose) upper bound
  used purely for the smoothing denominator. `#syn`/`#wsyn` use `cf = sum` (resp.
  weighted sum) of member `cf`s, matching their "occurrences of a OR b" semantics
  (this can double-count docs where both members co-occur; accepted as a minor
  overcount, documented rather than hidden).
- **Multi-token `Term` collection frequency**: our index stores unigram (not
  positional/phrase) collection frequencies, so a quoted/hyphenated multi-token
  term's `cf` is approximated as `min(cf(subtoken) for subtoken in term)` (same
  "rarest constituent" reasoning as windows) rather than a true phrase count.
- **Field restriction vs. field-context evaluation collapse for compound
  expressions**: `expr.field` (count-restriction) and `expr.(field)` (context/
  smoothing-only) are both implemented as (counting_fields, smoothing_fields)
  overrides threaded down through the whole subtree they wrap (including belief
  operators, not just leaves), so `#combine(a b).title` restricts both `a` and `b`
  to the title field. Official Indri's field restriction is closer to an
  extent-retrieval operation typically applied to a single term/proximity
  expression; broadening it to arbitrary subtrees is a pragmatic simplification for
  a single-shot (non-extent-indexed) scorer, and is the one place this backend
  extends the reference syntax's usual usage rather than narrowing it.
- **`#band`/`#filreq`/`#filrej` "matches" test**: for a leaf-ish node
  (term/wildcard/window/syn/wsyn) "matches" means raw match-count > 0 in the current
  field scope; for a compound sub-expression (rare as a filter argument) it means
  "belief did not hard-fail" (`!= -inf`). Filters and `#band` failures are
  represented as `-inf` log-belief, which propagates through `#combine`/`#weight`
  (mean/weighted-mean of `-inf` stays `-inf`: the whole subtree is excluded) but is
  correctly treated as probability-0 (not "unscoreable") by `#or`/`#not`, matching
  the official probability-space definition of those two operators.
- **Candidate pool + cap**: only a candidate pool is scored, not the whole corpus:
  the union of postings-doc-sets for every positive leaf term/wildcard found
  anywhere in the tree (including inside `#not`, `#filreq`'s `A`, etc., deliberately
  over-inclusive/recall-safe, never used to exclude), plus the exact matching doc set
  of every `#filreq`/date-filter node (so a pure-filter query, e.g. a bare
  `#date:between(...)`, still gets real candidates). If this pool exceeds
  `INDRI_POOL_CAP` (default 5000), it's trimmed to the top-N by a cheap heuristic
  (sum of body-field term frequencies for the query's terms) before the full
  Dirichlet/belief scoring pass runs: an approximation of "the docs most likely to
  matter," not an exact top-k. If the pool is empty (no leaf term appears anywhere
  and no filter matched), it falls back to the whole corpus rather than returning a
  hard zero. This is the graded-semantics point of the backend: a multi-term
  `#combine` where no single document has every term should still rank *something*.
- **Dense-embedding belief (off by default)**: `IndriExecutor(..., dense=None)`: pass a
  `dense_belief.DenseBelief` instance to turn this on; with `dense=None` (the default)
  every code path below is skipped entirely and behavior is byte-identical to the
  pre-dense engine (no new pool members, no diagnostics entry, no combination). When
  attached, two independent things happen, both keyed off the raw query surface text
  (`_plain_terms`-stripped of `#operator`/`.field` syntax before it reaches the encoder,
  mirroring `agent_search.tools.search_indri`'s `_indri_query_terms`):
  (1) **Pool expansion (recall)**: the dense retriever's top-`INDRI_DENSE_EXPAND_K`
  (default 50) doc_ids are unioned into the candidate pool after the lexical
  `POOL_CAP` heuristic runs, so a paraphrased/obfuscated document with zero query-term
  overlap (which the lexical `_collect_term_pool` would never surface, and which a
  lexical-tf `_cap_pool` heuristic would rank last if it somehow got in) is still
  scored. Such a doc still gets a real (if low) Dirichlet belief from pure collection
  smoothing (`_dirichlet` never returns `-inf` for a `Combine`/`Term` query, see the
  unseen-term-smoothing bullet above), so pool membership alone doesn't fabricate a
  false hard-fail.
  (2) **Belief combination (precision/reranking)**: for every pooled doc with both a
  finite lexical log-belief and a dense cosine similarity, `final_log_belief =
  (1-w)*lex_log_belief + w*log(dense_norm)`, where `w` = `INDRI_DENSE_W` (env, default
  0.35, read live, same pattern as `_rescore_m`) and `dense_norm` is the doc's raw
  cosine similarity min-max normalized *within the scored pool* (not the whole corpus)
  to `[DENSE_NORM_EPS, 1.0]`. This is a **deviation from the rest of the engine's true
  log-probability semantics**: Dirichlet log-beliefs are calibrated log-probabilities,
  but `log(dense_norm)` is a pool-relative, arbitrarily-rescaled score with no
  probabilistic meaning outside this one ranking call. The combination is a pragmatic
  linear score-fusion (in log space, consistent with `#combine`'s additive-in-log-space
  shape), not a joint probability. A degenerate pool (every doc's similarity identical,
  `range <= 0`) maps every doc to `dense_norm = 1.0` (`log(1.0) = 0`), a no-op additive
  shift that preserves the pure-lexical relative ranking rather than dividing by zero
  or crashing. Any dense-side exception (model/index missing, offline, corrupt cache)
  degrades silently to lexical-only pool/scoring for that call, consistent with this
  module's "never hard-fail the agent loop" philosophy elsewhere. Diagnostics gain one
  extra `("#dense", log(dense_norm))` entry for the top hit whenever a dense similarity
  was computed for it (even at `INDRI_DENSE_W=0`, so the contribution is visible without
  necessarily affecting ranking). `dense` is not persisted (excluded from `_PERSIST`,
  same rationale as `units`): a pickle-loaded executor has `dense=None` until
  `attach_dense(...)` is called, exactly mirroring `attach_units`.
- **Proximity window matching is a bounded reimplementation, not Indri's internal
  algorithm**: `#odN` counts one match per distinct start position of the first
  child that can be greedily extended in order within the gap bound (not an
  exhaustive/optimal combinatorial count); `#uwN` counts one match per window start
  offset `s` such that `[s, s+N-1]` contains at least one occurrence of every child
  (an `O(doc_len * N)` sliding check). Both are correct up to potential double
  counting of overlapping windows; `#od`/`#uw` with no window number is "unlimited"
  (whole-field span) and is implemented (the reference marks it optional).
- **Filter pool exactness**: `#filreq(A Q)`'s candidate-pool contribution scans all
  documents to test `A` exactly (this index has no separate structural/boolean
  postings the way `bql` does); `#filrej(A Q)` conservatively adds the whole corpus
  to the pool rather than paying the same scan (the exclusion is enforced correctly
  during scoring via the `-inf` gate regardless: this only affects the size of the
  candidate pool, never correctness).
- **Date literals/comparisons**: only ISO `YYYY[-MM[-DD]]` is accepted (see
  `parser.py`); a partial bound widens to the full covered span
  (`date_bounds`: year -> Jan 1..Dec 31, etc.). A document with a missing or
  malformed `metadata['date']` never satisfies any date filter (excluded, not
  "unknown").
- **Two-stage max-score-style rescoring for large pools**: window/proximity
  operators (`#odN`/`#uwN`) and multi-token `Term` phrases need per-doc token
  positions, which are only derivable by tokenizing the doc's field text: too
  expensive to do for the whole (up to `INDRI_POOL_CAP`-sized) candidate pool.
  Instead, when `len(pool) > INDRI_RESCORE_M` (env, default 300) scoring runs in
  two stages, the standard top-k/max-score optimization pattern: stage 1 scores
  every pooled doc with a position-free approximation. Window/phrase operators
  use `min(member unigram tf)` as their pseudo-tf (the same "rarest constituent"
  reasoning already used for their `cf` approximation above; since a window's
  true match count can never exceed its rarest member's raw frequency, this is a
  safe upper bound, so stage 1 never under-ranks a doc relative to its true
  score); everything else (single-token terms, wildcards, Dirichlet smoothing,
  belief combinators, filters, date gates) is exact already (no positions
  needed). Stage 2 exactly re-scores (real position/window matching) only the
  top `INDRI_RESCORE_M` stage-1 docs, plus any doc a `#filreq`/date-filter node
  hard-requires if there are fewer than `INDRI_RESCORE_M` of those (so a tight
  filter's docs always get exact treatment even if stage 1 under-ranked them
  for an unrelated reason). Final ranking uses stage 2's exact score for
  rescored docs and stage 1's approximate score for the rest, so `k` larger
  than the rescore set still returns `k` results, just with approximate scores
  below the exact top-`INDRI_RESCORE_M`. If `len(pool) <= INDRI_RESCORE_M`,
  stage 1 is skipped entirely (exact-only, identical to the pre-optimization
  behavior; this is why the small hand-computable tests stay exact). As a
  second, orthogonal bound, `_match_positions` (and window position matching)
  caps counted matches at `MATCH_POSITIONS_CAP` (50) per doc per operator and
  early-exits once reached; Dirichlet-smoothed tf saturates in its effect on
  the score well before 50 raw occurrences matter (mu dominates), so this is a
  negligible-precision-loss ceiling on otherwise-unbounded per-doc work, not a
  correctness change.
"""
from __future__ import annotations

import bisect
import datetime
import math
import os
import pickle
import re
from dataclasses import dataclass, field as dc_field
from typing import Optional, Sequence, TYPE_CHECKING

from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.indri.index import FIELDS, IndriIndex, field_text
from agent_search.retrievers.indri.parser import (
    Band, Combine, DateAfter, DateBefore, DateBetween, FieldExpr, FilReq, FilRej,
    Max, Not, Or, Syn, Term, Weight, Wildcard, Window, WSyn,
    date_bounds, parse, render,
)

if TYPE_CHECKING:          # pragma: no cover - type-only, never imported at runtime
    from agent_search.retrievers.dense import DenseBelief

DEFAULT_MU = float(os.environ.get("INDRI_MU", "2500"))
POOL_CAP = int(os.environ.get("INDRI_POOL_CAP", "5000"))
NEG_INF = float("-inf")

# --- dense-embedding belief source (off by default; see module Deviations) ---
DENSE_NORM_EPS = 1e-6

# Per-doc, per-operator cap on counted position matches (`_match_positions` and
# window position matching). Read live (not cached) so tests can override via
# monkeypatch.setenv without reloading the module. See module Deviations.
MATCH_POSITIONS_CAP = 50


def _rescore_m() -> int:
    """Stage 2 rescore-set size (see Deviations: two-stage max-score rescoring).
    Read live (not a module-level constant) so `INDRI_RESCORE_M` can be
    overridden per-test via `monkeypatch.setenv`/`os.environ` without a module
    reload, matching how tests need to force small values."""
    return int(os.environ.get("INDRI_RESCORE_M", "300"))


def _dense_weight() -> float:
    """`INDRI_DENSE_W`: the dense belief's blend weight when a `DenseBelief` is
    attached (see module Deviations). Read live (not cached), same pattern as
    `_rescore_m`, so tests can override via `monkeypatch.setenv` without a reload.
    Irrelevant (never read) when `self.dense is None`: dense is off by default."""
    return float(os.environ.get("INDRI_DENSE_W", "0.35"))


def _dense_expand_k() -> int:
    """`INDRI_DENSE_EXPAND_K`: size of the dense top-K pool-expansion union (see
    module Deviations). Read live, same pattern as `_dense_weight`/`_rescore_m`."""
    return int(os.environ.get("INDRI_DENSE_EXPAND_K", "50"))

_INDEX_VERSION = "v1"
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")

_LEAFISH = (Term, Wildcard, Window, Syn, WSyn)


def indri_index_path(index_root: str, key: str) -> str:
    """Where a corpus's prebuilt indri index lives: `indexes/indri/<safe_key>-v1.pkl`
    (mirrors `bql_index_path`)."""
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "__", key or "corpus")
    return os.path.join(index_root, "indri", f"{safe}-{_INDEX_VERSION}.pkl")


@dataclass
class IndriResult:
    hits: list                                  # [(doc_id, score), ...] best-first
    error: Optional[str] = None
    diagnostics: list = dc_field(default_factory=list)   # [(child_repr, log_belief), ...]
    # An unrecognized `.field` name (e.g. `.foo` where `foo` isn't
    # body/title/section/author/date) is not a parse/type error in Indri QL: it's a
    # syntactically valid restriction to a field with zero postings, so both engines
    # "search anyway" and just return fewer/zero hits, with nothing distinguishing that
    # from a genuinely zero-hit query on a known field. `warning` surfaces that
    # distinction to the agent (see `unknown_query_fields` below and
    # `agent_search/tools/search_indri/tool.py`, which appends it to the rendered
    # result text) without changing the hits/ranking themselves.
    warning: Optional[str] = None


_KNOWN_FIELDS = frozenset(f.lower() for f in FIELDS)


def unknown_query_fields(expr) -> list:
    """Every field name referenced by a `.field` / `.field.(context)` restriction
    anywhere in `expr` that isn't one of `index.FIELDS` (body/title/section/author/
    date), in first-seen order, deduped. Shared by `IndriExecutor.search` (below)
    and `LuceneStructuredEngine.search_indri` (lucene/engine.py): both compile
    the same parsed AST (`indri.parser.parse`), so this one walk is the single
    source of truth for "did this query name a field neither engine knows about."
    Purely diagnostic: an unknown field name is a valid query (Indri QL has no
    field-name validity restriction, see `_resolve_fields`'s "unknown -> filter-
    only, no scoring" comment in `lucene/indri_compiler.py`), so this never raises
    or changes what a query matches, only whether a warning gets attached."""
    seen: list = []

    def _note(name: Optional[str]) -> None:
        if name and name.lower() not in _KNOWN_FIELDS and name not in seen:
            seen.append(name)

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, FieldExpr):
            for f in (node.fields or ()):
                _note(f)
            _note(node.context)
            walk(node.child)
        elif isinstance(node, Not):
            walk(node.child)
        elif isinstance(node, (Window, Syn, Combine, Or, Max, Band)):
            for c in node.children:
                walk(c)
        elif isinstance(node, (Weight, WSyn)):
            for _, c in node.pairs:
                walk(c)
        elif isinstance(node, FilReq) or isinstance(node, FilRej):
            walk(node.a)
            walk(node.q)
        # Term/Wildcard/DateBefore/DateAfter/DateBetween: no children, no fields.

    walk(expr)
    return seen


def _field_warning(unknown: list) -> Optional[str]:
    if not unknown:
        return None
    names = ", ".join(repr(f) for f in unknown)
    return (f"warning: unrecognized field {names} (known: "
            f"{', '.join(sorted(_KNOWN_FIELDS))}) -- this restricts to a field with "
            f"no postings, so it contributes nothing; check for a typo")


def _unit_date(u: CodeUnit) -> Optional[str]:
    """`unit.metadata['date']` normalized to 'YYYY-MM-DD', or None if missing/
    malformed/not a real calendar date. Mirrors bql's `_unit_date` semantics."""
    meta = u.metadata or {}
    raw = meta.get("date")
    if not raw:
        return None
    m = _ISO_DATE_RE.match(str(raw))
    if not m:
        return None
    y, mo, d = (int(g) for g in m.groups())
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else NEG_INF


class IndriExecutor:
    """Public API: see the package `__init__.py` docstring."""

    def __init__(self, units: Sequence[CodeUnit], mu: Optional[float] = None,
                 dense: Optional["DenseBelief"] = None):
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the Indri engine", "a strategy with prebuilt-index support")
        self.units = list(units)
        self._ubyid = {u.doc_id: u for u in self.units}
        self.mu = float(mu) if mu is not None else DEFAULT_MU
        self.index = IndriIndex.build(self.units)
        self._doc_id_order = [u.doc_id for u in self.units]
        # lazy per-(doc, field-tuple) tokenized field cache, for proximity positions.
        # Not persisted: rebuilt on demand from the live (attached) units.
        self._doctok_cache: dict = {}
        # lazy sorted (date, unit_idx) index for date-filter pooling. Not persisted.
        self._date_keys: Optional[list] = None
        self._date_units: Optional[list] = None
        # Off by default (see module Deviations: "Dense-embedding belief"). Not
        # persisted: mirrors `units` (see `_PERSIST`/`attach_units`/`attach_dense`).
        self.dense: Optional["DenseBelief"] = dense
        # lazy doc_id -> unit-index map, for dense pool-expansion doc_id lookups.
        # Not persisted (derivable from `_doc_id_order`, which is persisted).
        self._doc_id_to_idx: Optional[dict] = None

    # --- persistence -----------------------------------------------------------
    # Slim pickle: only the postings index + mu + doc-id order. Units and the
    # per-doc token/date caches are not persisted (same rationale as the BQL
    # executor's slim pickle: they rebuild cheaply from the live corpus and would
    # otherwise dominate the pickle size).
    _PERSIST = ("index", "mu", "_doc_id_order")

    def __getstate__(self) -> dict:
        return {k: getattr(self, k) for k in self._PERSIST}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.units = []
        self._ubyid = {}
        self._doctok_cache = {}
        self._date_keys = None
        self._date_units = None
        self.dense = None                  # not persisted, see `attach_dense`
        self._doc_id_to_idx = None

    def attach_units(self, units: Sequence[CodeUnit]) -> "IndriExecutor":
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the Indri engine", "a strategy with prebuilt-index support")
        """Rebuild units + per-doc caches from the corpus after a slim-pkl load.
        The persisted postings reference unit indices, so `units` must be in the
        same order as at build time: validated against the stored doc-id order."""
        units = list(units)
        order = getattr(self, "_doc_id_order", None)
        if order is not None and [u.doc_id for u in units] != order:
            raise ValueError("units order != persisted index order (corpus changed) — rebuild")
        self.units = units
        self._ubyid = {u.doc_id: u for u in units}
        self._doctok_cache = {}
        self._date_keys = None
        self._date_units = None
        return self

    def attach_dense(self, dense: Optional["DenseBelief"]) -> "IndriExecutor":
        """Attach (or detach, via `dense=None`) a `DenseBelief` post-construction:
        mirrors `attach_units`'s "reattach after a slim-pkl load" pattern, since
        `dense` is likewise excluded from `_PERSIST`. Purely a setter (no pool/
        scoring logic here); off by default, so a caller that never calls this
        gets byte-identical pre-dense behavior (see module Deviations)."""
        self.dense = dense
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str) -> "IndriExecutor":
        """GC disabled across the unpickle, same rationale as
        `StructuralExecutor.load`: a big object graph unpickled while the caller
        already holds a large corpus in memory degrades to O(n^2) GC otherwise."""
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        finally:
            if gc_was_enabled:
                gc.enable()

    # --- public search -----------------------------------------------------------

    def search(self, query: str, k: int = 5) -> IndriResult:
        r = parse(query)
        if not r.ok:
            return IndriResult(hits=[], error=str(r.error), diagnostics=[])
        warning = _field_warning(unknown_query_fields(r.expr))
        try:
            result = self._search_expr(r.expr, k, query_text=query)
        except Exception as e:                 # never crash the agent loop
            return IndriResult(hits=[], error=f"execution error: {e}", diagnostics=[],
                                warning=warning)
        result.warning = warning
        return result

    def _search_expr(self, root, k: int, query_text: Optional[str] = None) -> IndriResult:
        n = len(self.units)
        if n == 0:
            return IndriResult(hits=[], error=None, diagnostics=[])
        pool = self._collect_term_pool(root) | self._collect_filter_pool(root)
        if not pool:
            pool = set(range(n))              # graceful fallback: never a hard zero
        if len(pool) > POOL_CAP:
            pool = self._cap_pool(pool, root)

        # Dense pool expansion (off unless `self.dense` is attached): unioned in
        # after the lexical POOL_CAP heuristic so a lexical-tf cap never evicts a
        # dense-only recall candidate. See module Deviations.
        if self.dense is not None and query_text:
            pool = pool | self._dense_expand(query_text)

        m = _rescore_m()
        if len(pool) <= m:
            # Pool already small enough: exact-only, identical to pre-optimization
            # behavior (stage 1 approximation is skipped entirely). See Deviations.
            scored = self._score_pool(root, pool, approx=False)
        else:
            scored = self._two_stage_score(root, pool, m)

        # Dense belief combination (off unless `self.dense` is attached): rescales
        # `scored`'s log-beliefs in place per doc; see module Deviations.
        dense_log_by_idx: dict = {}
        if self.dense is not None and query_text and scored:
            scored, dense_log_by_idx = self._combine_dense(scored, query_text)

        scored.sort(key=lambda x: (-x[1], self.units[x[0]].doc_id))
        top = scored[:k]
        hits = [(self.units[i].doc_id, b) for i, b in top]
        diagnostics = []
        if top:
            top_idx = top[0][0]
            for child in self._root_children(root):
                diagnostics.append(
                    (render(child), self._belief(child, top_idx, ("body",), ("body",))))
            dense_log = dense_log_by_idx.get(top_idx)
            if dense_log is not None:
                diagnostics.append(("#dense", dense_log))
        return IndriResult(hits=hits, error=None, diagnostics=diagnostics)

    # --- dense-embedding belief source (off unless `self.dense` is attached) ------

    def _idx_of_doc_id(self, doc_id: str) -> Optional[int]:
        if self._doc_id_to_idx is None:
            self._doc_id_to_idx = {d: i for i, d in enumerate(self._doc_id_order)}
        return self._doc_id_to_idx.get(doc_id)

    def _dense_expand(self, query_text: str) -> set:
        """Dense top-`INDRI_DENSE_EXPAND_K` doc_ids for `query_text`, mapped to unit
        indices: the recall half of the dense belief source (see module
        Deviations). Any dense-side failure degrades to no expansion (never breaks
        the lexical search)."""
        try:
            doc_ids = self.dense.top_k_doc_ids(query_text, _dense_expand_k())
        except Exception:
            return set()
        out: set = set()
        for d in doc_ids:
            i = self._idx_of_doc_id(d)
            if i is not None:
                out.add(i)
        return out

    def _combine_dense(self, scored: list, query_text: str) -> tuple:
        """Precision half of the dense belief source (see module Deviations):
        one batched dense similarity call for every doc in `scored`, min-max
        normalized to `[DENSE_NORM_EPS, 1.0]` *within this pool*, log-combined
        with each doc's lexical log-belief via `INDRI_DENSE_W`. Returns
        `(new_scored, dense_log_by_idx)`; the latter feeds the `#dense`
        diagnostics entry (computed regardless of `INDRI_DENSE_W`, including 0,
        so the contribution is visible even when it doesn't affect ranking). Any
        dense-side failure returns `scored` unchanged (lexical-only)."""
        doc_ids = [self.units[i].doc_id for i, _ in scored]
        try:
            sims = self.dense.score(query_text, doc_ids)
        except Exception:
            return scored, {}
        if not sims:
            return scored, {}
        vals = list(sims.values())
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        dense_log_by_idx: dict = {}
        for doc_idx, _lex_b in scored:
            sim = sims.get(self.units[doc_idx].doc_id)
            if sim is None:
                continue
            norm = (1.0 if rng <= 0 else
                    DENSE_NORM_EPS + (1.0 - DENSE_NORM_EPS) * (sim - lo) / rng)
            dense_log_by_idx[doc_idx] = math.log(norm)
        w = _dense_weight()
        if w <= 0 or not dense_log_by_idx:
            return scored, dense_log_by_idx
        combined = []
        for doc_idx, lex_b in scored:
            dl = dense_log_by_idx.get(doc_idx)
            combined.append((doc_idx, (1.0 - w) * lex_b + w * dl) if dl is not None
                             else (doc_idx, lex_b))
        return combined, dense_log_by_idx

    def _score_pool(self, root, pool, approx: bool) -> list:
        """Score every doc in `pool` against `root`; returns `[(doc_idx, belief), ...]`
        for docs that don't hard-fail (belief != NEG_INF). `approx=True` uses the
        position-free stage 1 approximation (see Deviations); `approx=False` is the
        exact evaluation."""
        out = []
        for i in pool:
            b = self._belief(root, i, ("body",), ("body",), approx=approx)
            if b != NEG_INF:
                out.append((i, b))
        return out

    def _two_stage_score(self, root, pool: set, m: int) -> list:
        """Stage 1 (position-free approximate) plus stage 2 (exact) max-score-style
        rescoring, see module Deviations. Stage 1 scores the whole pool cheaply;
        stage 2 exactly re-scores only the top-`m` stage-1 docs (plus any
        filter-required doc, if there are fewer than `m` of those), so real
        position/window matching work is bounded to a small fixed-size set
        regardless of pool size. Docs outside the rescored set keep their stage 1
        approximate score/inclusion; docs inside it are governed by stage 2's exact
        score/inclusion (a rescored doc that exactly hard-fails is correctly
        dropped even if stage 1 approximated it as passing)."""
        approx = dict(self._score_pool(root, pool, approx=True))
        ranked = sorted(approx, key=lambda i: (-approx[i], self.units[i].doc_id))
        rescore_set = set(ranked[:m])
        required = self._required_filter_docs(root) & pool
        if len(required) < m:
            rescore_set |= required
        exact = dict(self._score_pool(root, rescore_set, approx=False))
        scored = []
        for i in pool:
            if i in exact:
                scored.append((i, exact[i]))
            elif i not in rescore_set and i in approx:
                scored.append((i, approx[i]))
            # else: rescored but exact-excluded (-inf), or never had a finite
            # stage 1 score and wasn't rescored -> correctly dropped from hits.
        return scored

    def _required_filter_docs(self, root) -> set:
        """Docs a `#filreq`/date-filter node hard-requires: exact, and cheap to
        compute (bisect range lookups / a single exact `_matches` scan, no token
        positions), already computed en route to the candidate pool in
        `_collect_filter_pool`. Used to guarantee stage 2 exact rescoring for
        filter-gated docs even if stage 1's approximation under-ranked them for
        an unrelated reason. `#filrej` is deliberately excluded: its "required"
        set is usually ~the whole corpus, which the `< m` guard in
        `_two_stage_score` naturally skips anyway (and computing it exactly would
        cost the same full scan `_collect_filter_pool` already avoids paying)."""
        req: set = set()
        n = len(self.units)
        for node in self._walk_filters(root):
            if isinstance(node, FilReq):
                req |= {i for i in range(n) if self._matches(node.a, i, ("body",))}
            elif isinstance(node, DateBefore):
                hi = date_bounds(node.date)[0]
                req |= self._date_range_indices(None, hi, hi_exclusive=True)
            elif isinstance(node, DateAfter):
                lo = date_bounds(node.date)[1]
                req |= self._date_range_indices(lo, None, lo_exclusive=True)
            elif isinstance(node, DateBetween):
                lo = date_bounds(node.lo)[0] if node.lo else None
                hi = date_bounds(node.hi)[1] if node.hi else None
                req |= self._date_range_indices(lo, hi)
        return req

    def _root_children(self, root) -> list:
        if isinstance(root, (Combine, Or, Max, Band)):
            return list(root.children)
        if isinstance(root, Weight):
            return [c for _, c in root.pairs]
        if isinstance(root, Not):
            return [root.child]
        if isinstance(root, (FilReq, FilRej)):
            return [root.a, root.q]
        return [root]

    # --- candidate pool -----------------------------------------------------------

    def _walk_leaves(self, node, fields=("body",)):
        if isinstance(node, FieldExpr):
            f2 = node.fields if node.fields else fields
            yield from self._walk_leaves(node.child, f2)
            return
        if isinstance(node, WSyn):
            yield (node, fields)
            for _, c in node.pairs:
                yield from self._walk_leaves(c, fields)
            return
        if isinstance(node, _LEAFISH):
            yield (node, fields)
            for c in getattr(node, "children", ()):
                yield from self._walk_leaves(c, fields)
            return
        if isinstance(node, Weight):
            for _, c in node.pairs:
                yield from self._walk_leaves(c, fields)
            return
        if isinstance(node, (Combine, Or, Max, Band)):
            for c in node.children:
                yield from self._walk_leaves(c, fields)
            return
        if isinstance(node, Not):
            yield from self._walk_leaves(node.child, fields)
            return
        if isinstance(node, (FilReq, FilRej)):
            yield from self._walk_leaves(node.a, fields)
            yield from self._walk_leaves(node.q, fields)
            return
        return                                # DateBefore/After/Between: no term leaves

    def _walk_filters(self, node):
        if isinstance(node, (DateBefore, DateAfter, DateBetween)):
            yield node
            return
        if isinstance(node, (FilReq, FilRej)):
            yield node
            yield from self._walk_filters(node.a)
            yield from self._walk_filters(node.q)
            return
        if isinstance(node, FieldExpr):
            yield from self._walk_filters(node.child)
            return
        if isinstance(node, Not):
            yield from self._walk_filters(node.child)
            return
        if isinstance(node, Weight):
            for _, c in node.pairs:
                yield from self._walk_filters(c)
            return
        if isinstance(node, WSyn):
            for _, c in node.pairs:
                yield from self._walk_filters(c)
            return
        for c in getattr(node, "children", ()):
            yield from self._walk_filters(c)

    def _collect_term_pool(self, root) -> set:
        pool: set = set()
        for node, fields in self._walk_leaves(root):
            pool |= self._leaf_doc_set(node, fields)
        return pool

    def _leaf_doc_set(self, node, fields) -> set:
        if isinstance(node, Term):
            subtoks = code_tokenize(node.text)
            if not subtoks:
                return set()
            sets = [self._union_field_term_docs(t, fields) for t in subtoks]
            return set.intersection(*sets) if sets else set()
        if isinstance(node, Wildcard):
            out: set = set()
            stem = node.stem.lower()
            for f in fields:
                out |= self.index.doc_ids_for_prefix(f, stem)
            return out
        return set()          # Window/Syn/WSyn members are separately walked/unioned

    def _union_field_term_docs(self, term: str, fields) -> set:
        out: set = set()
        for f in fields:
            out |= self.index.doc_ids_for_term(f, term)
        return out

    def _collect_filter_pool(self, root) -> set:
        pool: set = set()
        n = len(self.units)
        for node in self._walk_filters(root):
            if isinstance(node, FilReq):
                pool |= {i for i in range(n) if self._matches(node.a, i, ("body",))}
            elif isinstance(node, FilRej):
                pool |= set(range(n))          # conservative; -inf gate excludes at score time
            elif isinstance(node, DateBefore):
                hi = date_bounds(node.date)[0]
                pool |= self._date_range_indices(None, hi, hi_exclusive=True)
            elif isinstance(node, DateAfter):
                lo = date_bounds(node.date)[1]
                pool |= self._date_range_indices(lo, None, lo_exclusive=True)
            elif isinstance(node, DateBetween):
                lo = date_bounds(node.lo)[0] if node.lo else None
                hi = date_bounds(node.hi)[1] if node.hi else None
                pool |= self._date_range_indices(lo, hi)
        return pool

    def _cap_pool(self, pool: set, root) -> set:
        terms: set = set()
        for node, _fields in self._walk_leaves(root):
            if isinstance(node, Term):
                terms.update(code_tokenize(node.text))
            elif isinstance(node, Wildcard):
                terms.add(node.stem.lower())

        def heuristic(i: int) -> int:
            return sum(self.index.tf("body", t, i) for t in terms)

        ranked = sorted(pool, key=lambda i: (-heuristic(i), self.units[i].doc_id))
        return set(ranked[:POOL_CAP])

    # --- lazy per-doc token cache (positions; not persisted) ----------------------

    def _doc_tokens(self, doc_idx: int, fields) -> list:
        key = (doc_idx, fields)
        toks = self._doctok_cache.get(key)
        if toks is None:
            u = self.units[doc_idx]
            toks = []
            for f in fields:
                toks.extend(code_tokenize(field_text(u, f)))
            self._doctok_cache[key] = toks
        return toks

    def _match_positions(self, subtoks: list, doc_idx: int, fields) -> list:
        """Exact position matches, early-exiting once `MATCH_POSITIONS_CAP` are
        found (see module Deviations, a bound on otherwise-unbounded per-doc
        work; only used for the stage 2 exact rescore set, or when the pool never
        needed stage 1 at all)."""
        if not subtoks:
            return []
        toks = self._doc_tokens(doc_idx, fields)
        out = []
        if len(subtoks) == 1:
            target = subtoks[0]
            for i, t in enumerate(toks):
                if t == target:
                    out.append(i)
                    if len(out) >= MATCH_POSITIONS_CAP:
                        break
            return out
        m = len(subtoks)
        for i in range(len(toks) - m + 1):
            if toks[i:i + m] == subtoks:
                out.append(i)
                if len(out) >= MATCH_POSITIONS_CAP:
                    break
        return out

    # --- collection-stat aggregation across a (possibly multi-)field scope --------

    def _agg_doclen(self, doc_idx: int, fields) -> int:
        return sum(self.index.doclen(f, doc_idx) for f in fields)

    def _agg_total(self, fields) -> int:
        return sum(self.index.total_of(f) for f in fields)

    def _agg_cf(self, term: str, fields) -> int:
        return sum(self.index.cf_of(f, term) for f in fields)

    def _agg_tf(self, term: str, doc_idx: int, fields) -> int:
        """Per-doc unigram tf straight from postings: O(1) dict lookups, no
        tokenization/positions. Used by the stage 1 approximation (`_raw_tf_approx`)."""
        return sum(self.index.tf(f, term, doc_idx) for f in fields)

    # --- positions / raw match counts / collection freq per leaf-ish node ---------

    def _positions(self, node, doc_idx: int, fields) -> list:
        if isinstance(node, FieldExpr):
            f2 = node.fields if node.fields else fields
            return self._positions(node.child, doc_idx, f2)
        if isinstance(node, Term):
            return self._match_positions(code_tokenize(node.text), doc_idx, fields)
        if isinstance(node, Wildcard):
            stem = node.stem.lower()
            toks = self._doc_tokens(doc_idx, fields)
            return [i for i, t in enumerate(toks) if t.startswith(stem)]
        if isinstance(node, Window):
            return self._window_positions(node, doc_idx, fields)
        if isinstance(node, Syn):
            out: set = set()
            for c in node.children:
                out.update(self._positions(c, doc_idx, fields))
            return sorted(out)
        if isinstance(node, WSyn):
            out = set()
            for _, c in node.pairs:
                out.update(self._positions(c, doc_idx, fields))
            return sorted(out)
        return []

    def _window_positions(self, node: Window, doc_idx: int, fields) -> list:
        member_positions = [self._positions(c, doc_idx, fields) for c in node.children]
        if not member_positions or any(not p for p in member_positions):
            return []
        if node.kind == "od":
            return self._ordered_matches(member_positions, node.n)
        doclen = self._agg_doclen(doc_idx, fields)
        return self._unordered_matches(member_positions, node.n, doclen)

    @staticmethod
    def _ordered_matches(member_positions: list, n: Optional[int]) -> list:
        """One match per distinct start position of the first child, greedily
        extended in order with gap <= n-1 between consecutive terms (n=None =
        unbounded gap). See module Deviations note."""
        matches = []
        first, rest = member_positions[0], member_positions[1:]
        for p0 in first:
            if len(matches) >= MATCH_POSITIONS_CAP:
                break
            cur = p0
            ok = True
            for positions in rest:
                nxt = None
                for p in positions:
                    if p > cur and (n is None or (p - cur - 1) <= (n - 1)):
                        nxt = p
                        break
                if nxt is None:
                    ok = False
                    break
                cur = nxt
            if ok:
                matches.append(cur)
        return matches

    @staticmethod
    def _unordered_matches(member_positions: list, n: Optional[int], doclen: int) -> list:
        """One match per window-start offset `s` such that `[s, s+win-1]` contains
        >=1 occurrence of every child (n=None = the whole field is one window)."""
        if n is None:
            ok = all(bool(p) for p in member_positions)
            return [0] if ok else []
        win = max(n, 1)
        span = max(doclen - win + 1, 1)
        matches = []
        for s in range(span):
            end = s + win - 1
            if all(any(s <= p <= end for p in positions) for positions in member_positions):
                matches.append(s)
                if len(matches) >= MATCH_POSITIONS_CAP:
                    break
        return matches

    def _raw_tf(self, node, doc_idx: int, fields) -> float:
        if isinstance(node, FieldExpr):
            f2 = node.fields if node.fields else fields
            return self._raw_tf(node.child, doc_idx, f2)
        if isinstance(node, (Term, Wildcard, Window)):
            positions = self._positions(node, doc_idx, fields)
            if isinstance(node, Window):
                doclen = self._agg_doclen(doc_idx, fields)
                return float(min(len(positions), doclen)) if doclen > 0 else float(len(positions))
            return float(len(positions))
        if isinstance(node, Syn):
            return float(sum(self._raw_tf(c, doc_idx, fields) for c in node.children))
        if isinstance(node, WSyn):
            return float(sum(w * self._raw_tf(c, doc_idx, fields) for w, c in node.pairs))
        return 0.0

    def _raw_tf_approx(self, node, doc_idx: int, fields) -> float:
        """Stage 1 position-free approximation of `_raw_tf` (see module Deviations:
        two-stage max-score-style rescoring). Single-token terms and wildcards are
        computed exactly here (postings already give exact per-doc unigram counts,
        no position scan needed); only window/phrase operators are approximated,
        via `min(member unigram tf)` (mirrors the existing `_raw_cf` "rarest
        constituent" approximation, and is a safe upper bound on the true window
        match count, so stage 1 never under-ranks a doc relative to stage 2)."""
        if isinstance(node, FieldExpr):
            f2 = node.fields if node.fields else fields
            return self._raw_tf_approx(node.child, doc_idx, f2)
        if isinstance(node, Term):
            subtoks = code_tokenize(node.text)
            if not subtoks:
                return 0.0
            tfs = [float(self._agg_tf(t, doc_idx, fields)) for t in subtoks]
            return min(tfs) if tfs else 0.0
        if isinstance(node, Wildcard):
            stem = node.stem.lower()
            total = 0.0
            for f in fields:
                for t in self.index.terms_for_prefix(f, stem):
                    total += self.index.tf(f, t, doc_idx)
            return total
        if isinstance(node, Window):
            member_tfs = [self._raw_tf_approx(c, doc_idx, fields) for c in node.children]
            return min(member_tfs) if member_tfs else 0.0
        if isinstance(node, Syn):
            return float(sum(self._raw_tf_approx(c, doc_idx, fields) for c in node.children))
        if isinstance(node, WSyn):
            return float(sum(w * self._raw_tf_approx(c, doc_idx, fields) for w, c in node.pairs))
        return 0.0

    def _raw_cf(self, node, fields) -> float:
        if isinstance(node, FieldExpr):
            f2 = (node.context,) if node.context else (node.fields if node.fields else fields)
            return self._raw_cf(node.child, f2)
        if isinstance(node, Term):
            subtoks = code_tokenize(node.text)
            if not subtoks:
                return 0.0
            if len(subtoks) == 1:
                return float(self._agg_cf(subtoks[0], fields))
            return float(min(self._agg_cf(t, fields) for t in subtoks))
        if isinstance(node, Wildcard):
            stem = node.stem.lower()
            total = 0.0
            for f in fields:
                for t in self.index.terms_for_prefix(f, stem):
                    total += self.index.cf_of(f, t)
            return total
        if isinstance(node, Window):
            member_cfs = [self._raw_cf(c, fields) for c in node.children]
            return float(min(member_cfs)) if member_cfs else 0.0
        if isinstance(node, Syn):
            return float(sum(self._raw_cf(c, fields) for c in node.children))
        if isinstance(node, WSyn):
            return float(sum(w * self._raw_cf(c, fields) for w, c in node.pairs))
        return 0.0

    # --- Dirichlet -----------------------------------------------------------------

    def _dirichlet(self, tf: float, doclen: float, cf: float, total: float) -> float:
        total = total if total > 0 else 1.0
        p_c = (cf / total) if cf > 0 else (0.5 / total)     # unseen-term epsilon (see Deviations)
        num = tf + self.mu * p_c
        den = doclen + self.mu
        if den <= 0 or num <= 0:
            return NEG_INF
        return math.log(num / den)

    # --- boolean-style "matches" test (band/filreq/filrej) -------------------------

    def _matches(self, node, doc_idx: int, fields, approx: bool = False) -> bool:
        if isinstance(node, FieldExpr):
            f2 = node.fields if node.fields else fields
            return self._matches(node.child, doc_idx, f2, approx)
        if isinstance(node, _LEAFISH):
            tf = (self._raw_tf_approx(node, doc_idx, fields) if approx
                  else self._raw_tf(node, doc_idx, fields))
            return tf > 0
        if isinstance(node, Not):
            return not self._matches(node.child, doc_idx, fields, approx)
        return self._belief(node, doc_idx, fields, fields, approx) != NEG_INF

    # --- belief (log-belief) evaluation ---------------------------------------------

    def _belief(self, node, doc_idx: int, counting_fields, smoothing_fields,
                approx: bool = False) -> float:
        """`approx=True` selects the stage 1 position-free approximation for
        window/phrase leaves (`_raw_tf_approx` instead of `_raw_tf`); every other
        operator's math is identical between stages (see module Deviations)."""
        if isinstance(node, FieldExpr):
            new_counting = node.fields if node.fields else counting_fields
            new_smoothing = ((node.context,) if node.context
                              else (node.fields if node.fields else smoothing_fields))
            return self._belief(node.child, doc_idx, new_counting, new_smoothing, approx)
        if isinstance(node, _LEAFISH):
            tf = (self._raw_tf_approx(node, doc_idx, counting_fields) if approx
                  else self._raw_tf(node, doc_idx, counting_fields))
            doclen = self._agg_doclen(doc_idx, counting_fields)
            cf = self._raw_cf(node, smoothing_fields)
            total = self._agg_total(smoothing_fields)
            return self._dirichlet(tf, doclen, cf, total)
        if isinstance(node, Combine):
            if not node.children:
                return NEG_INF
            vals = [self._belief(c, doc_idx, counting_fields, smoothing_fields, approx)
                    for c in node.children]
            return sum(vals) / len(vals)
        if isinstance(node, Weight):
            if not node.pairs:
                return NEG_INF
            wsum = sum(w for w, _ in node.pairs) or 1.0
            total = 0.0
            for w, c in node.pairs:
                wn = w / wsum
                if wn == 0:
                    continue                    # a zero-weight child never contributes
                v = self._belief(c, doc_idx, counting_fields, smoothing_fields, approx)
                if v == NEG_INF:
                    return NEG_INF               # nonzero-weight hard-fail propagates
                total += wn * v
            return total
        if isinstance(node, Or):
            if not node.children:
                return NEG_INF
            prod = 1.0
            for c in node.children:
                v = self._belief(c, doc_idx, counting_fields, smoothing_fields, approx)
                p = math.exp(v) if v != NEG_INF else 0.0
                prod *= (1.0 - p)
            return _safe_log(1.0 - prod)
        if isinstance(node, Not):
            v = self._belief(node.child, doc_idx, counting_fields, smoothing_fields, approx)
            p = math.exp(v) if v != NEG_INF else 0.0
            return _safe_log(1.0 - p)
        if isinstance(node, Max):
            if not node.children:
                return NEG_INF
            return max(self._belief(c, doc_idx, counting_fields, smoothing_fields, approx)
                       for c in node.children)
        if isinstance(node, Band):
            if not node.children:
                return NEG_INF
            if not all(self._matches(c, doc_idx, counting_fields, approx) for c in node.children):
                return NEG_INF
            vals = [self._belief(c, doc_idx, counting_fields, smoothing_fields, approx)
                    for c in node.children]
            return sum(vals) / len(vals)
        if isinstance(node, FilReq):
            if self._matches(node.a, doc_idx, counting_fields, approx):
                return self._belief(node.q, doc_idx, counting_fields, smoothing_fields, approx)
            return NEG_INF
        if isinstance(node, FilRej):
            if not self._matches(node.a, doc_idx, counting_fields, approx):
                return self._belief(node.q, doc_idx, counting_fields, smoothing_fields, approx)
            return NEG_INF
        if isinstance(node, DateBefore):
            return self._date_before_belief(doc_idx, node.date)
        if isinstance(node, DateAfter):
            return self._date_after_belief(doc_idx, node.date)
        if isinstance(node, DateBetween):
            return self._date_between_belief(doc_idx, node.lo, node.hi)
        return NEG_INF

    # --- date filters: per-doc gate (0.0 = pass, -inf = excluded) -----------------

    def _date_before_belief(self, doc_idx: int, bound: str) -> float:
        d = _unit_date(self.units[doc_idx])
        if d is None:
            return NEG_INF
        start = date_bounds(bound)[0]
        return 0.0 if d < start else NEG_INF

    def _date_after_belief(self, doc_idx: int, bound: str) -> float:
        d = _unit_date(self.units[doc_idx])
        if d is None:
            return NEG_INF
        end = date_bounds(bound)[1]
        return 0.0 if d > end else NEG_INF

    def _date_between_belief(self, doc_idx: int, lo: Optional[str], hi: Optional[str]) -> float:
        d = _unit_date(self.units[doc_idx])
        if d is None:
            return NEG_INF
        lo_b = date_bounds(lo)[0] if lo else None
        hi_b = date_bounds(hi)[1] if hi else None
        if lo_b is not None and d < lo_b:
            return NEG_INF
        if hi_b is not None and d > hi_b:
            return NEG_INF
        return 0.0

    # --- lazy sorted date index (pooling only) -------------------------------------

    def _ensure_date_index(self) -> tuple:
        if self._date_keys is None:
            pairs = []
            for i, u in enumerate(self.units):
                d = _unit_date(u)
                if d is not None:
                    pairs.append((d, i))
            pairs.sort(key=lambda p: p[0])
            self._date_keys = [d for d, _ in pairs]
            self._date_units = [i for _, i in pairs]
        return self._date_keys, self._date_units

    def _date_range_indices(self, lo: Optional[str], hi: Optional[str],
                             lo_exclusive: bool = False, hi_exclusive: bool = False) -> set:
        keys, units_idx = self._ensure_date_index()
        lo_pos = (bisect.bisect_right(keys, lo) if lo_exclusive
                  else bisect.bisect_left(keys, lo)) if lo is not None else 0
        hi_pos = (bisect.bisect_left(keys, hi) if hi_exclusive
                  else bisect.bisect_right(keys, hi)) if hi is not None else len(keys)
        return set(units_idx[lo_pos:hi_pos]) if hi_pos > lo_pos else set()


def load_or_build(units: Sequence[CodeUnit], index_root: Optional[str] = None,
                  key: Optional[str] = None, rebuild: bool = False,
                  dense: Optional["DenseBelief"] = None) -> IndriExecutor:
    """Return a prewarmed executor loaded from disk if one was materialized for this
    corpus, else build it in memory and persist it (same pattern as bql's
    `load_or_build`) so the O(n) build is paid once, not once per episode.

    `dense` (off by default, `None`): a `DenseBelief` to attach via `attach_dense`
    after load/build, convenience only, since `dense` is never persisted (see
    module Deviations). Omitting it (the default) leaves the executor lexical-only."""
    if index_root and key and not rebuild:
        path = indri_index_path(index_root, key)
        if os.path.exists(path):
            try:
                ex = IndriExecutor.load(path)
                ex.attach_units(units)
                ex.attach_dense(dense)
                return ex
            except Exception:
                pass                          # corrupt/stale/order-mismatch -> rebuild
    ex = IndriExecutor(units, dense=dense)
    if index_root and key:
        try:
            ex.save(indri_index_path(index_root, key))
        except Exception:
            pass
    return ex


class IndriIndexBuilder:
    """Offline persister for the Indri index, so `agent_search/evaluation/build_indexes.py` can pre-build
    it exactly like BQL's `BQLIndexBuilder`: `.index(units, key)` writes a prewarmed executor
    to disk; `.is_cached(key)` reports whether it already exists (skip the corpus parse). The
    agent (research_indri) loads it back via `load_or_build`."""
    name = "search_indri"

    def __init__(self, index_root: str = "indexes", rebuild: bool = False):
        self.index_root = index_root
        self.rebuild = rebuild

    def is_cached(self, key: str) -> bool:
        return os.path.exists(indri_index_path(self.index_root, key))

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "IndriIndexBuilder":
        IndriExecutor(list(units)).save(indri_index_path(self.index_root, key))
        return self
