"""Corpus grounding for the exact-search tools — the index-free analog of a code-entity index.

Exact / structural search (search_dir, search_bql) fails SILENTLY when the agent guesses a token
that isn't in the corpus, so the agent queries blind (~half of search_bql calls returned nothing
in our runs). This turns an empty result into a GROUNDED one — "no `foo`; did you mean `bar`,
`baz`?" — drawn from a token set built ONCE from the units (not a persistent index, so the
index-free thesis holds).

FAIRNESS: this is applied UNIFORMLY to every exact-search tool (search_dir / search_file /
find_file / search_bql), so the grounding is a property of the ENVIRONMENT, not a search_bql-only
advantage. The structural tool's only intrinsic extra is that its suggestions are REGION-aware
(it can say "no `foo` in region `call`"), which a flat grep cannot express.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize


def corpus_token_vocab(units: Sequence[CodeUnit]) -> set[str]:
    """Every distinct search token across the units' text (title/section/body/code + qualname/
    path) — the vocabulary an exact-search 'did you mean' draws from. Lower-cased."""
    vocab: set[str] = set()
    for u in units:
        for part in (u.title, u.section, u.body, u.code, u.qualname, u.path):
            if part:
                vocab.update(code_tokenize(part))
    return {v.lower() for v in vocab if v}


def _edit_distance_le(a: str, b: str, k: int) -> bool:
    """True iff Levenshtein(a, b) <= k. Banded DP with early-exit (only called on length-/edge-near
    candidates, so the corpus scan stays cheap)."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > k:
            return False
        prev = cur
    return prev[-1] <= k


def nearest_tokens(term: str, vocab: Iterable[str], n: int = 5) -> list[str]:
    """Closest existing tokens to `term` from `vocab`: the exact-prefix family first (either
    direction), then small typos (edit distance <= 1 for short terms, <= 2 for len>=5),
    case-insensitive, excluding `term` itself. Best-first (prefix > distance > length), deduped,
    up to `n`. Returns [] when nothing is close (then the empty result is real, not a typo)."""
    t = (term or "").strip().lower()
    if not t:
        return []
    k = 2 if len(t) >= 5 else 1
    prefix: list[str] = []
    typo: list[str] = []
    for v in vocab:
        vl = v.lower()
        if vl == t:
            continue
        if vl.startswith(t) or t.startswith(vl):
            prefix.append(vl)
        elif (abs(len(vl) - len(t)) <= k and (vl[:1] == t[:1] or vl[-1:] == t[-1:])
              and _edit_distance_le(t, vl, k)):
            typo.append(vl)
    prefix = sorted(set(prefix), key=lambda v: (abs(len(v) - len(t)), v))
    typo = sorted(set(typo), key=lambda v: (abs(len(v) - len(t)), v))
    out: list[str] = []
    for v in prefix + typo:
        if v not in out:
            out.append(v)
        if len(out) >= n:
            break
    return out


def format_did_you_mean(term: str, suggestions: Sequence[str], region: str | None = None) -> str:
    """One-line grounding hint, or '' when there is nothing to suggest. Region-qualified for a
    structural (BQL) miss: 'no `foo` in region `call`; did you mean: bar, baz'."""
    if not suggestions:
        return ""
    where = f" in region `{region}`" if region else ""
    return f"no `{term}`{where} in the corpus; did you mean: " + ", ".join(suggestions)


def ground_text(text: str, vocab: Iterable[str], max_hints: int = 3,
                region: Optional[str] = None) -> str:
    """Per-segment 'did you mean' grounding for a literal-search miss — the flat-tool analog of
    the executor's region-aware suggest().

    TOKENIZES `text` (so a MULTI-SEGMENT identifier like ``polygon_to_mask`` is grounded
    segment-by-segment, not as one opaque string), and for each token ABSENT from `vocab`
    emits a ``format_did_you_mean`` hint, joined with '; ' and capped at `max_hints`. This is
    the same shape as ``StructuralExecutor.suggest`` so the literal tools and search_bql ground
    a typo UNIFORMLY; BQL's sole intrinsic extra is that it can pass a `region`, which the flat
    tools leave None. Returns '' when every segment already exists (the empty result is real) or
    nothing close is found. Never raises.

    `vocab` may be any iterable; it is materialized to a set once so repeated membership tests and
    the per-token nearest-token scan are cheap."""
    try:
        vocabset = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
        hints: list[str] = []
        seen: set = set()
        for tok in code_tokenize(text or ""):
            tl = tok.strip().lower()
            if not tl or tl in seen:
                continue
            seen.add(tl)
            if tl in vocabset:
                continue                      # segment exists -> not a typo
            hint = format_did_you_mean(tl, nearest_tokens(tl, vocabset), region=region)
            if hint:
                hints.append(hint)
            if len(hints) >= max_hints:
                break
        return "; ".join(hints)
    except Exception:                         # noqa: BLE001 — advisory only, never break a tool
        return ""


__all__ = ["corpus_token_vocab", "nearest_tokens", "format_did_you_mean", "ground_text"]
