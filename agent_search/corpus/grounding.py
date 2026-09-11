"""Corpus grounding for exact/structural search: the index-free analog of a code-entity index.

Exact search (`search_bql`, via `agent_search.retrievers.bql.executor.StructuralExecutor.suggest`)
fails silently when the agent guesses a token that isn't in the corpus, so the agent queries
blind. This turns an empty result into a grounded one: "no `foo`; did you mean `bar`, `baz`?",
drawn from a token set built once from the units (not a persistent index, so the index-free
thesis holds).

The structural tool's suggestions are region-aware (it can say "no `foo` in region `call`");
`ground_text` below gives a plain literal-search caller the same "did you mean" without a
region, but nothing in the current tool set calls it, so the grounding is live only on
`search_bql` today.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from agent_search.corpus.units import CodeUnit, code_tokenize


def corpus_token_vocab(units: Sequence[CodeUnit]) -> set[str]:
    """Every distinct search token across the units' text (title/section/body/code + qualname/
    path): the vocabulary an exact-search 'did you mean' draws from. Lower-cased."""
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
    """Per-segment 'did you mean' grounding for a literal-search miss: the flat-tool analog of
    the executor's region-aware `suggest()`, for a caller that wants the same behavior outside
    `search_bql`.

    Tokenizes `text` (so a multi-segment identifier like ``polygon_to_mask`` is grounded
    segment-by-segment, not as one opaque string), and for each token absent from `vocab`
    emits a ``format_did_you_mean`` hint, joined with '; ' and capped at `max_hints`. Shaped
    the same way as ``StructuralExecutor.suggest``, minus the `region` qualifier the flat tools
    have no use for. Returns '' when every segment already exists (the empty result is real) or
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
    except Exception:                         # noqa: BLE001 - advisory only, never break a tool
        return ""


__all__ = ["corpus_token_vocab", "nearest_tokens", "format_did_you_mean", "ground_text"]
