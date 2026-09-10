"""Retrieval metrics for code localization (grounded in SweRank / LocAgent).

All functions are per-query and operate on document ids:
  retrieved : ranked list of doc ids (best first)
  gold      : set of relevant doc ids (the gold-patch locations)

- recall_at_k : fraction of gold ids appearing in the top-k.
- hit_at_k    : 1.0 iff ANY gold id is in the top-k (classic IR Top-N / Hit@k;
                the soft companion to the strict all-gold acc_at_k).
- precision_at_k / f1_at_k : textbook precision@k (|gold∩topk|/k) and its F1 with
                recall@k. NOTE: gold sets here are tiny (often 1 function), so
                precision@k is mechanically capped at |gold|/k — report as a
                surgicality DIAGNOSTIC (tight Boolean query vs broad grep), not a
                headline; the SWE-bench loc standard is acc_at_k (LocAgent/SweRank).
- average_precision_at_k : AP (BugLocator/MAP lineage); MAP = mean over instances.
- acc_at_k    : 1.0 iff min(|gold|, k) gold ids are in the top-k, else 0.0 —
                exactly LocAgent's released acc_at_k (eval_metric.py): all-or-
                nothing per instance, R-Precision-style, NOT any-hit. The min()
                means an instance with more gold locations than k can still
                score 1.0 when the entire top-k is gold (strict `gold ⊆ top-k`
                would make Acc@1 impossible for every multi-gold instance).
- mrr_at_k    : reciprocal rank of the first gold id within top-k.
- ndcg_at_k   : binary-relevance nDCG over the top-k.
Aggregate across queries by averaging.
"""
from __future__ import annotations

import math
import re
import string
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold or k <= 0:                      # k<=0 has no top-k; guard the negative
        return 0.0                              # slice (retrieved[:-1] would drop the last)
    topk = set(retrieved[:k])
    return len(topk & gold) / len(gold)


def acc_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold or k <= 0:                  # k<=0 has no top-k -> 0.0 (like the other @k)
        return 0.0
    need = min(len(gold), k)
    return 1.0 if len(set(retrieved[:k]) & gold) >= need else 0.0


def mrr_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# --- end-to-end answer metrics (QA datasets; offline, no judge) ---------------

def _normalize_answer(text: str) -> str:
    """SQuAD-style normalization, matching the OFFICIAL HotpotQA / 2WikiMultihopQA eval scripts
    verbatim (lower -> remove punctuation -> remove articles a/an/the -> collapse whitespace):

        def normalize_answer(s):
            def remove_articles(text): return re.sub(r'\\b(a|an|the)\\b', ' ', text)
            def white_space_fix(text): return ' '.join(text.split())
            def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
            def lower(text): return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

    Do not reorder these steps — the published EM/F1 numbers depend on this exact pipeline."""
    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    """Exact match after normalization (the standard QA EM; HotpotQA/2WikiMultihopQA canonical
    metric). Requires the WHOLE normalized prediction to equal the whole normalized gold —
    per the published protocol, the answerer is expected to emit a short span, not a sentence
    (see agent_search/tasks/research/prompt.md, which instructs exactly that)."""
    if not gold:
        return 0.0
    p_norm, g_norm = _normalize_answer(prediction), _normalize_answer(gold)
    if not g_norm:                 # a gold that normalizes to nothing cannot be matched
        return 0.0
    return 1.0 if p_norm == g_norm else 0.0


_YES_NO = {"yes", "no", "noanswer"}


def answer_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 after normalization (the standard QA F1) — robust to the
    model answering in a sentence while gold is a short span.

    Follows the official HotpotQA scorer: a yes/no/noanswer prediction that does not equal
    the gold scores 0 (no partial credit for "yes" against "yes, in 1999"), and an empty
    token overlap — including both sides normalizing to nothing — scores 0."""
    if not gold:
        return 0.0
    p_norm, g_norm = _normalize_answer(prediction), _normalize_answer(gold)
    if p_norm in _YES_NO and p_norm != g_norm:
        return 0.0
    p_toks = p_norm.split()
    g_toks = g_norm.split()
    if not p_toks or not g_toks:
        return 0.0
    common = 0
    g_counts: dict = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_toks:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def support_f1(surfaced_doc_ids: Sequence[str], gold_doc_ids: set[str]) -> float:
    """MuSiQue SUPPORT F1: set-F1 between the agent's surfaced/retrieved supporting-doc ids and
    the gold supporting-doc ids (MuSiQue's paired metric alongside Answer F1 — it grades WHICH
    paragraphs the system used to answer, not just whether the answer string is right).

        precision = |surfaced ∩ gold| / |surfaced|
        recall    = |surfaced ∩ gold| / |gold|
        f1        = harmonic mean

    Order-free (a set, not a ranking) — unlike acc_at_k/recall_at_k above, which are for the
    code-localization arm's RANKED retrieval. Empty gold or empty surfaced set -> 0.0."""
    gold = set(gold_doc_ids or ())
    surfaced = set(surfaced_doc_ids or ())
    if not gold or not surfaced:
        return 0.0
    inter = len(surfaced & gold)
    if inter == 0:
        return 0.0
    precision = inter / len(surfaced)
    recall = inter / len(gold)
    return 2 * precision * recall / (precision + recall)


def hit_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Top-N / Hit@k: 1.0 if at least one gold id is in the top-k."""
    if not gold or k <= 0:
        return 0.0
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """|gold ∩ top-k| / k. Coverage-deflated for small gold sets (see module doc)."""
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & gold) / k


def f1_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Harmonic mean of precision_at_k and recall_at_k."""
    p = precision_at_k(retrieved, gold, k)
    r = recall_at_k(retrieved, gold, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Average precision over the gold positions in the top-k (MAP lineage,
    BugLocator/BLUiR). Normalized by min(|gold|, k) so a perfect ranking = 1.0.
    MAP = mean of this over instances."""
    if not gold or k <= 0:                      # k<=0: no top-k (and min(|gold|,k)<=0
        return 0.0                              # would make the denom negative -> AP<0)
    # Dedup the top-k, preserving order: a repeated doc_id must not double-credit
    # (that would push AP above 1.0). The set-based metrics above are already
    # dup-immune; this is the one rank-walking metric, so it needs the guard.
    seen: set = set()
    ranked: list = []
    for doc in retrieved[:k]:
        if doc not in seen:
            seen.add(doc)
            ranked.append(doc)
    hits = 0
    score = 0.0
    for rank, doc in enumerate(ranked, start=1):
        if doc in gold:
            hits += 1
            score += hits / rank
    denom = min(len(gold), k)
    return score / denom if denom else 0.0


# --- cutoff-FREE set metrics (native to Boolean/grep: a query returns a SET, not
# a ranking; these answer "how do you cut off" = you don't — you report the whole
# returned set). Degenerate for dense (it ranks the whole corpus, no set), so use
# them to characterize the index-free set tools, not as a cross-condition headline.

def set_recall(retrieved: Sequence[str], gold: set[str]) -> float:
    """Did the query's FULL returned set contain the gold? |set ∩ gold| / |gold|."""
    if not gold:
        return 0.0
    return len(set(retrieved) & gold) / len(gold)


def set_precision(retrieved: Sequence[str], gold: set[str]) -> float:
    """Purity / surgicality of the returned set: |set ∩ gold| / |set|.
    A tight IN(def,x) (2 hits) scores far higher than a broad grep (1426 hits)."""
    s = set(retrieved)
    return len(s & gold) / len(s) if s else 0.0


def set_f1(retrieved: Sequence[str], gold: set[str]) -> float:
    p = set_precision(retrieved, gold)
    r = set_recall(retrieved, gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
