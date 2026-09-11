"""The in-memory Boolean executor for code repositories: the "complex grep".

Evaluates a parsed BQL query directly against the repository's code units, with no
persisted index and no embeddings. Boolean logic selects the candidate set, AST scope
narrows it, and the in-memory code scorer (`agent_search/retrievers/lexical/scorer.py`)
orders the survivors. Document corpora never come here: they run on the Lucene structured
index (`agent_search/retrievers/lucene/`), and `agent_search/retrievers/backend.py` decides
by corpus kind.

The executor follows the repository. `refresh(units)` re-indexes only the files whose
units changed and drops the files that disappeared, so a task that edits code calls it
with the fresh units and pays for the changed files only.

Semantics:
  - Boolean selects the candidate set; a deterministic BM25 score orders it.
  - Units are function defs, so IN(def, x) means x matches within the unit.
  - IN(file, x) switches scope to the whole file (path + all its units).
  - NEAR/func|block|para|sent means co-occurrence in the unit; NEAR/file co-occurrence
    anywhere in the unit's file; NEAR/wN within N tokens; NEAR/lineN within N source
    lines. Multi-token terms require adjacency.
  - Window NEAR honors its operands' full match semantics: Term/Phrase (distance to the
    nearest token of a contiguous match), Prefix/Expand (prefix matches), and OR (union
    of child positions). And-composed operands fall back to unit co-occurrence.
  - IN(def/call/comment/string/sig) use real AST scope: IN(call, x) matches where x is
    called, IN(def, x) where it is defined. This is the differentiator from plain grep.
  - The document regions (title/body/section/doc/author/date/infobox) map onto what a
    code unit has: title is the qualified name, body the code, section the path, and the
    metadata fields are empty.
  - EXPAND is a bounded prefix/identifier-variant match.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional, Sequence

from agent_search.corpus import grounding
from agent_search.corpus.units import CodeUnit, code_tokenize
from agent_search.retrievers.lexical.grep import file_fingerprints
from agent_search.retrievers.lexical.scorer import BM25
from agent_search.retrievers.bql.ast import (
    And, Expand, Expr, Granularity, In, Near, Not, Or, Phrase, Prefix, Region, Term,
)
from agent_search.retrievers.bql.structure import region_token_bags

# NEAR specs that mean "co-occur in the same region" rather than a token window.
_COOCCUR_SPECS = ("func", "file", "block", "para", "sent")

# BQL region -> structural bag key (real AST scope); others fall back to unit text.
_REGION_KEY = {
    Region.DEF: "def", Region.CALL: "call", Region.STRING: "string",
    Region.COMMENT: "comment", Region.SIG: "sig",
}

# document field regions, mapped onto a code unit's fields by `_field_bag`.
_FIELD_REGIONS = {Region.TITLE, Region.BODY, Region.SECTION, Region.DOC,
                  Region.AUTHOR, Region.DATE, Region.INFOBOX}

# The surface lowers `date[1980..1989]` to `IN(date, __daterange__1980-01-01__1989-12-31)`
# (see surface.py). Code units carry no dates, so the leaf never matches here; the regex is
# kept so the encoding is recognised and evaluated consistently instead of tokenised.
_DATE_RANGE_TERM_RE = re.compile(
    r"^__daterange__(open|\d{4}-\d{2}-\d{2})__(open|\d{4}-\d{2}-\d{2})$")

# Cap on the per-(unit, child) pool of the coverage ranking (`coverage_topk`): the top
# BM25-scored slice of the candidate union, so a query whose children are all
# unnarrowable does not test every child against every unit.
_COVERAGE_POOL_CAP = 2000
# Size of the zero-hit fallback pool (`soft_topk`), shared with the Lucene adapter.
_SOFT_POOL = int(os.environ.get("BQL_SOFT_POOL", "100"))


class StructuralExecutor:
    def __init__(self, units: Sequence[CodeUnit]):
        from agent_search.corpus.docstore import refuse_lazy
        refuse_lazy(units, "the code Boolean executor", "a document corpus runs on the Lucene structured index")
        self.units: list[CodeUnit] = []
        self._ubyid: dict[str, CodeUnit] = {}
        self._files: dict[str, tuple[str, list[CodeUnit]]] = {}   # path -> (fingerprint, units)
        # per-unit token list/set + per-token line number (for line-distance NEAR)
        self._utoks: dict[str, list[str]] = {}
        self._uset: dict[str, set] = {}
        self._ulines: dict[str, list[int]] = {}
        # AST region bags and field bags, built lazily per unit on the first IN(...) query
        # that needs them: an ast.parse per unit is the dominant build cost.
        self._region: dict[str, dict] = {}
        self._fields: dict[str, dict[Region, list[str]]] = {}
        # File-level token bags (NEAR/file, IN(file, ·)), built lazily per path.
        self._ftoks: dict[str, list[str]] = {}
        self._units_by_path: dict[str, list[str]] = {}
        self._bm = BM25()                      # the corpus-level rerank scorer, kept up to date
        # region -> union of that region's tokens across all units, built lazily on the
        # first suggest() call (the grounding "did you mean" vocabulary).
        self._vocab_lock = threading.Lock()
        self._region_vocab: Optional[dict] = None
        self._global_vocab: set = set()
        self.refresh(units)

    # --- following the repository ---------------------------------------------------------
    def refresh(self, units: Sequence[CodeUnit]) -> "StructuralExecutor":
        """Bring the executor up to date with `units`: files whose units changed are
        re-tokenised and re-scored, files that disappeared are dropped, everything else is
        left alone. The first call builds everything."""
        fresh = file_fingerprints(units)
        changed = [p for p, (fp, _) in fresh.items() if self._files.get(p, ("", None))[0] != fp]
        gone = [p for p in self._files if p not in fresh]
        for path in changed + gone:
            _, old = self._files.get(path, ("", []))
            self._bm.remove(u.doc_id for u in old)
            for u in old:
                for cache in (self._utoks, self._uset, self._ulines, self._region, self._fields, self._ubyid):
                    cache.pop(u.doc_id, None)
            self._ftoks.pop(path, None)
            self._units_by_path.pop(path, None)
        for path in changed:
            new = fresh[path][1]
            toks_by_id = {}
            for u in new:
                toks, lns = _tok_lines(u.qualname, u.code)
                self._utoks[u.doc_id] = toks
                self._uset[u.doc_id] = set(toks)
                self._ulines[u.doc_id] = lns
                self._ubyid[u.doc_id] = u
                toks_by_id[u.doc_id] = toks
            self._units_by_path[path] = [u.doc_id for u in new]
            self._bm.add_tokenized(toks_by_id)
        if changed or gone:
            with self._vocab_lock:
                self._region_vocab = None
                self._global_vocab = set()
        self._files = fresh
        self.units = list(units)
        return self

    # --- per-unit caches --------------------------------------------------------------------
    def _file_toks(self, path: str) -> list:
        """All tokens in a unit's file: path tokens plus every sibling unit's tokens."""
        toks = self._ftoks.get(path)
        if toks is None:
            toks = list(code_tokenize(path))
            for doc_id in self._units_by_path.get(path, []):
                toks += self._utoks[doc_id]
            self._ftoks[path] = toks
        return toks

    def _region_bags(self, doc_id: str) -> dict:
        bags = self._region.get(doc_id)
        if bags is None:
            u = self._ubyid[doc_id]
            bags = dict(region_token_bags(u.code))
            # A unit's own qualified name IS a definition here: a method of `class RST`
            # must match IN(def, RST) even though the class name never appears in the body.
            bags["def"] = list(bags.get("def", [])) + code_tokenize(u.qualname)
            self._region[doc_id] = bags
        return bags

    def _field_bag(self, doc_id: str, region: Region) -> list:
        """Tokens of one document field for `doc_id`, built lazily per (unit, field)."""
        cache = self._fields.setdefault(doc_id, {})
        toks = cache.get(region)
        if toks is None:
            u = self._ubyid[doc_id]
            if region is Region.DOC:
                toks = self._utoks[doc_id]
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

    # --- grounding: region-aware near-miss suggestions --------------------------------------
    def _build_region_vocab(self) -> dict:
        """Union of each region's tokens across all units: the vocabulary a "did you mean"
        draws from, keyed by Region. Region.FILE spans a path's units, so it takes the
        global union."""
        vocab: dict[Region, set] = {}
        for u in self.units:
            region_bags = self._region_bags(u.doc_id)
            for region, key in _REGION_KEY.items():
                vocab.setdefault(region, set()).update(region_bags.get(key, ()))
            for region in _FIELD_REGIONS:
                vocab.setdefault(region, set()).update(self._field_bag(u.doc_id, region))
        glob: set = set()
        for toks in vocab.values():
            glob |= toks
        vocab[Region.FILE] = set(glob)
        vocab.setdefault(Region.DOC, set()).update(glob)
        self._global_vocab = glob
        return vocab

    def _region_vocab_for(self, region: Optional[Region]) -> set:
        if self._region_vocab is None:
            with self._vocab_lock:
                if self._region_vocab is None:
                    self._region_vocab = self._build_region_vocab()
        if region is None:
            return self._global_vocab
        return self._region_vocab.get(region, self._global_vocab)

    def suggest(self, query: str) -> str:
        """Region-aware near-miss grounding for a (typically empty) BQL result: for each leaf
        token absent from its target region's vocabulary, a "no `foo` in region `call`; did
        you mean: bar" hint. Returns "" when the query does not parse, when every token
        exists, or when nothing close is found. Advisory only: never raises."""
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
                    continue
                suggestions = grounding.nearest_tokens(tl, vocab)
                hint = grounding.format_did_you_mean(
                    tl, suggestions, region=(region.value if region else None))
                if hint:
                    hints.append(hint)
                if len(hints) >= 3:
                    break
            return "; ".join(hints)
        except Exception:
            return ""

    # --- public API -----------------------------------------------------------------------
    def run(self, expr: Expr, k: int = 100) -> list[tuple[str, float]]:
        """Boolean-filter the units, then rank the survivors deterministically."""
        return self.run_with_count(expr, k=k)[0]

    def run_with_count(self, expr: Expr, k: int = 100) -> tuple[list[tuple[str, float]], int]:
        """The top-k ranked hits plus the untruncated match count. Every unit is tested with
        `_eval` (a repository is small); the matched set is ordered by BM25 over the query's
        positive leaf terms, and zero-score candidates (matched via file scope or the
        qualified name) stay ranked after the scored ones."""
        leaves = _rank_leaves(expr)
        matched = [u.doc_id for u in self.units
                   if self._eval(expr, self._utoks[u.doc_id], self._uset[u.doc_id], u)]
        if not matched:
            return [], 0
        return self._bm.score_subset(leaves, matched)[:k], len(matched)

    def soft_topk(self, terms, k: int = 5) -> list:
        """The fallback for a zero-hit Boolean query: BM25 over `terms` alone, best first.
        The same scorer that orders exact matches, so a fallback never ranks by a model the
        exact path does not use."""
        terms = list(terms)
        pool_n = max(k, _SOFT_POOL)
        scored = self._bm.score_terms(terms)
        return [(d, s) for d, s in scored[:pool_n] if s > 0][:k]

    def coverage_topk(self, expr: Expr, k: int = 5) -> list:
        """Constraint-coverage ranking for a zero-hit AND: `[(doc_id, matched_mask, n_matched,
        score)]`, mask aligned with the AND's children (a Not child counts as matched when the
        negation holds), ordered by children matched, then BM25 over the query's positive
        terms, then id. A non-AND query degrades to `soft_topk` with a one-element mask."""
        if not isinstance(expr, And):
            terms = _rank_leaves(expr)
            return [(d, (True,), 1, s) for d, s in self.soft_topk(terms, k=k)]
        children = list(expr.children)
        terms = _rank_leaves(expr)
        pool = [u.doc_id for u in self.units]
        if len(pool) > _COVERAGE_POOL_CAP:
            pool = [d for d, _ in self._bm.score_subset(terms, pool)[:_COVERAGE_POOL_CAP]]
        rows = []
        for doc_id in pool:
            u = self._ubyid[doc_id]
            toks, tset = self._utoks[doc_id], self._uset[doc_id]
            mask = tuple(self._eval(c, toks, tset, u) for c in children)
            n = sum(mask)
            if n:
                rows.append((doc_id, mask, n))
        if not rows:
            return []
        scores = dict(self._bm.score_subset(terms, [r[0] for r in rows]))
        out = [(d, mask, n, scores.get(d, 0.0)) for d, mask, n in rows]
        out.sort(key=lambda r: (-r[2], -r[3], r[0]))
        return out[:k]

    # --- evaluation ---------------------------------------------------------------------
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
            if expr.region == Region.DATE and isinstance(expr.child, Term) \
                    and _DATE_RANGE_TERM_RE.match(expr.child.text):
                return False                      # code units carry no dates
            key = _REGION_KEY.get(expr.region)
            if key is not None:                   # real AST structural scope
                rtoks = self._region_bags(unit.doc_id)[key]
                return self._eval(expr.child, rtoks, set(rtoks), unit)
            if expr.region in _FIELD_REGIONS:
                ftoks = self._field_bag(unit.doc_id, expr.region)
                return self._eval(expr.child, ftoks, set(ftoks), unit)
            return self._eval(expr.child, toks, tset, unit)
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
            # The operand was bound by IN(file, ...) or IN(region, ...), so `toks` is not the
            # unit's own stream and a line window cannot be measured: degrade to co-occurrence.
            return bool(lpos) and bool(rpos)
        return any(abs(i - j) <= win for i in lpos for j in rpos)  # NEAR/wN -> N tokens


# --- helpers ----------------------------------------------------------------

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
    """All indices covered by contiguous occurrences of `seq` in `toks`."""
    n = len(seq)
    if n == 0:
        return None
    out: list = []
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == seq:
            out.extend(range(i, i + n))
    return sorted(set(out)) or None


def _positions(expr: Expr, toks: list) -> Optional[list]:
    """Token positions where this operand matches; None = not positionable. Positionable:
    Term (including multi-token), Phrase, Prefix, Expand, and Or (union of child positions).
    And/Not/In/Near operands stay None, which makes the window fall back to co-occurrence."""
    if isinstance(expr, Or):
        out: list = []
        for c in expr.children:
            p = _positions(c, toks)
            if p is None:
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
    """(token, target_region) pairs for grounding: every constituent token of each leaf,
    tagged with the region it must exist in. NOT subtrees are skipped."""
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
        return out
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
    """Collect positive ranking terms (skip NOT subtrees). Shared with the Lucene adapter."""
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
            return
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
    """Parse -> typecheck -> execute on whichever executor (`StructuralExecutor` for a code
    repository, `LuceneBqlAdapter` for documents). Errors come back as Observations."""
    from agent_search.retrievers.base import Observation
    from agent_search.retrievers.bql.parser import parse
    from agent_search.retrievers.bql.types import check
    r = parse(bql)
    if not r.ok:
        return Observation(n_hits=0, hits=[], typecheck_ok=False, error=f"parse error: {r.error}")
    t = check(r.expr)
    if not t.ok:
        return Observation(n_hits=0, hits=[], typecheck_ok=False, error=f"type error: {t.error}")
    ranked, n_hits = executor.run_with_count(r.expr, k=k)
    return Observation(n_hits=n_hits, hits=hits_from_ranked(ranked, units_by_id), typecheck_ok=True)
