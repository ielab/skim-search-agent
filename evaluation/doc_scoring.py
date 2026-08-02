"""Deep-research (doc-QA) scoring — the doc arm's end-to-end metric.

The research agent ends with an <answer>. We score it GROUNDED: the answer counts only when
it also appears verbatim (case/punctuation-insensitive) in the actual tool evidence the agent
saw — so a right answer pulled from the model's memory, not the corpus, does not score. On top
of the grounding gate, the metrics emitted are each dataset's PUBLISHED protocol:

  answer_em / answer_f1 : HotpotQA / 2WikiMultihopQA / MuSiQue canonical QA metrics
                           (SQuAD-style EM + token-F1; evaluation.metrics), reused.
  support_f1            : MuSiQue's paired SUPPORT F1 — set-F1 between the surfaced doc ids
                           and the gold supporting-doc ids (evaluation.metrics.support_f1).
  gold_doc_coverage      : did the episode surface the gold document(s)? (retrieval-quality axis,
                           recall-only; kept for diagnosis alongside support_f1's precision+recall)

BrowseComp-Plus is graded separately by the LLM judge (evaluation/llm_judge.py) as a post-hoc
pass over rows.jsonl — not here, since a judge call is not free/deterministic like the metrics
above.

There is no "cover-EM" / substring-containment metric here: no published protocol for any of
these datasets grades containment, so it is not computed. Instead, `score_answer` extracts the
`<answer>...</answer>` span (the short answer the agent was instructed to emit) from the raw
final answer before scoring, so a correctly-tagged short answer is scored fairly against the
strict published EM/F1 without inventing a laxer metric.

plus the efficiency axis the loop records (llm_calls / steps / tokens). Deep-research keeps its
QA shape (this is not localization); only the SURFACE + TOOLS changed (field-tagged search->fetch).

The grounding token logic is ported from bql_skill_construct/proto/grounding.py.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

from evaluation.metrics import answer_em, answer_f1, support_f1

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_span(text: str) -> str:
    """Return the content of the LAST-OPENED `<answer>` tag in `text` if present; otherwise return
    `text` unchanged (stripped). Anchors on the LAST `<answer>` (via rfind), NOT a `(.*?)` findall:
    models routinely name the tag in prose first ('...the short answer span inside <answer> tags...
    Thus: <answer>Galați</answer>'), and a non-greedy findall then pairs the PROSE `<answer>` with
    the REAL closing tag, capturing the junk in between (this is exactly the ~37%-mis-extracted bug).
    Robust to a raw answer that still carries tags (rescoring old rows.jsonl) AND to a mis-extracted
    final_answer that still ends in a nested `<answer>span` (the true span is recovered)."""
    if not text:
        return text or ""
    idx = text.rfind("<answer>")
    if idx < 0:
        return text.strip()
    tail = text[idx + len("<answer>"):]
    end = tail.find("</answer>")
    return (tail[:end] if end >= 0 else tail).strip()


def evidence_tokens(text: str) -> list[str]:
    """Casefolded alphanumeric runs; punctuation is a separator, never evidence."""
    cleaned = "".join(
        c.casefold() if (c.isalnum() or c.isspace()) else " "
        for c in unicodedata.normalize("NFKC", text)
    )
    return cleaned.split()


def answer_in_evidence(answer: str, observations: Sequence[str]) -> bool:
    """Whether the answer is a contiguous span in the actual tool evidence.

    Exact token order/boundaries are required, but case and punctuation are ignored. A compact
    comparison over nearby token spans handles punctuation variants (``U.S.`` vs ``US``, curly
    vs straight apostrophes) without letting ``York`` match the larger token ``Yorkshire``."""
    needle = evidence_tokens(answer)
    if not needle:
        return False
    compact = "".join(needle)
    for observation in observations:
        hay = evidence_tokens(observation)
        n = len(needle)
        if any(hay[i:i + n] == needle for i in range(len(hay) - n + 1)):
            return True
        for width in range(max(1, n - 2), n + 3):     # punctuation may split a token
            if any("".join(hay[i:i + width]) == compact
                   for i in range(len(hay) - width + 1)):
                return True
    return False


def gold_doc_coverage(surfaced: Sequence[str], gold_doc_ids) -> float:
    """Fraction of the gold document ids the episode surfaced (search hit or fetch/visit).
    1.0 when every gold doc was reached; 0.0 when none. Empty gold -> 0.0 (nothing to cover)."""
    gold = set(gold_doc_ids or ())
    if not gold:
        return 0.0
    seen = set(surfaced or ())
    return len(gold & seen) / len(gold)


def score_answer(answer: str, gold_answer: str, observations: Sequence[str], *,
                 surfaced_docs: Optional[Sequence[str]] = None,
                 gold_doc_ids: Optional[Sequence[str]] = None) -> dict:
    """The doc-QA row fragment: grounded EM/F1 (zeroed when the answer isn't in evidence) plus
    the ungrounded EM/F1 for diagnosis, and (when `gold_doc_ids` is given) MuSiQue's SUPPORT F1.
    `observations` is every tool response the agent saw this episode.

    `answer` is first passed through `extract_answer_span` — if it carries an `<answer>...</answer>`
    tag, ONLY the tagged span is scored (the short span the agent was instructed to emit); a raw
    answer with no tag is scored as-is."""
    pred = extract_answer_span(answer)
    grounded = answer_in_evidence(pred, observations) if pred else False
    em = answer_em(pred, gold_answer)
    f1 = answer_f1(pred, gold_answer)
    out = {
        "grounded": bool(grounded),
        "answer_em": em,
        "answer_f1": f1,
        "grounded_em": em if grounded else 0.0,
        "grounded_f1": f1 if grounded else 0.0,
    }
    if gold_doc_ids is not None:
        out["support_f1"] = support_f1(surfaced_docs or (), gold_doc_ids)
    return out
