"""How a retriever query is written from the agent's history (ITER's query styles).

A retriever trained from trajectories conditions on what the agent has already done: the main
question, the sub-query it is issuing now, and what it tried and read before. Both the training
data builder (`triples.py`) and inference (`agent_search.retrievers.dense.belief`
via `history.py`) call `render_query` so the string is byte-identical on both sides. That is ITER's
one rule: train, serve, and evaluate with the same query text and the same instruction.

Styles (ITER's names are kept so results line up with the paper):

* ``plain``  the bare sub-query (what an untrained retriever gets)
* ``mem``    ``[Q] question / [Now] sub-query / [Memory] cleaned notes from earlier reads``
* ``docs``   ``[Q] / [Now] / [Prev] earlier sub-queries with the docs read under each``
* ``i1``..``i7`` the structured "Main Question / Current Subquery / Previous Interactions"
  template; ``i2`` (sub-queries only) is the default training style, ``i3`` adds visited
  documents, ``i4`` adds notes too, ``i5`` notes only, ``i6``/``i7`` are ``i3``/``i4`` with the
  shorter agent-view snippets and no id tags.

Token truncation uses the library's token ruler (`agent_search.tokens`), so this module has
no tokenizer dependency; the numbers are the ones ITER used (64-token snippets, 256-token
budgets, 128-token document items).
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Optional, Sequence

from agent_search.tokens import count_tokens, truncate_tokens

STYLES = ("plain", "mem", "docs", "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9")
# i9 (the paper's ITER-i7): i2's fields plus the agent's pre-search reasoning, one line, before the
# sub-query. The released ielabgroup/ITER-Qwen3-Embedding checkpoints are trained on it (model card).
DEFAULT_STYLE = "i9"

# The instruction the trained model is served with. Training passes the same string as
# FlagEmbedding's --query_instruction_for_retrieval; inference prefixes queries with
# "Instruct: {instruction}\nQuery: " (see `instruction_prefix`).
INSTRUCTIONS = {
    "plain": "Given a web search query, retrieve relevant passages that answer the query",
    "mem": "Given the main question, the current sub-query, and notes from the documents already read, retrieve documents relevant to the current sub-query that provide NEW information.",
    "docs": "Given the main question, the current sub-query, and the documents already visited under previous sub-queries, retrieve documents relevant to the current sub-query that provide NEW information.",
    "i0": "Given a web search query, retrieve relevant passages that answer the query",
    "i1": "Given the main question and the current sub-query, retrieve documents relevant to the current sub-query.",
    "i2": "Given the main question, the current sub-query, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.",
    "i3": "Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i4": "Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i5": "Given the main question, the current sub-query, and previous interactions with notes taken on the documents already read, retrieve documents relevant to the current sub-query that provide NEW information beyond what the notes cover.",
    "i6": "Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i7": "Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i8": "Given the agent's reasoning that led to the current sub-query, retrieve documents relevant to the current sub-query that provide NEW information beyond what the reasoning already covers.",
    "i9": "Given the main question, the agent's reasoning and the current sub-query it led to, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.",
}

# per-variant rendering for the structured template: (with_docs, with_notes, id_tags, doc_tokens)
_VARIANT_CFG = {
    "i2": (False, False, True, 128),
    "i3": (True, False, True, 128),
    "i4": (True, True, True, 128),
    "i5": (False, True, True, 128),
    "i6": (True, False, False, 64),
    "i7": (True, True, False, 64),
    "i9": (False, False, True, 128),   # i2's fields plus the Current Reasoning line
}


def instruction_prefix(instruction: str) -> str:
    """The exact prefix a Qwen3-Embedding-style model is served with."""
    return f"Instruct: {instruction}\nQuery: "


# --- note cleaning (deterministic; same at train and inference) ------------------------

_NOISE = re.compile(
    r"\b(docid|doc id|doc ids|fetch|truncat|internal index|internal database|"
    r"search tool return|get those|those ids|try \d{4,}|not correct|limited fetch|"
    r"expects specific|different format|full path|the system|the tool)\b", re.I)
_QREST = re.compile(r"(the question[:\s]|the user (asks|mention|want)|question asks|asks[:\s])", re.I)
_WORD = re.compile(r"[a-z0-9]+")
_PLAN_VERBS = (r"(?:search|verify|check|open|look|try|examine|find|confirm|refine|visit|browse|"
               r"query|google|think|consider|explore|dig|investigate|scroll|request|ask|re-?get|"
               r"re-?read|retrieve)")
_PLAN = re.compile(
    r"(\blet'?s\s+(?:(?:also|now|then|first|next|just|instead|quickly)\s+)?" + _PLAN_VERBS + r"\b"
    r"|\b(?:we|i)\s+(?:need\s+to|should|will|may\s+need\s+to|might|could|can)\s+"
    r"(?:(?:also|now|then|first|next|just|instead|quickly)\s+)?" + _PLAN_VERBS + r"\b"
    r"|^(?:search|try\s+searching|use\s+search)\b"
    r"|\bnext,?\s+(?:search|step|we'?ll|let'?s)\b"
    r"|\bmaybe\s+(?:search|check|try|look)\b)", re.I)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _tokens(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def reasoning_text(raw_output: str) -> str:
    """What the model said after reading a document: the <think> block if there is one, else
    the generation with any tool call stripped."""
    raw = raw_output or ""
    m = _THINK.search(raw)
    if m:
        return m.group(1).strip()
    return re.sub(r"<tool_call>.*?</tool_call>", " ", raw, flags=re.DOTALL).strip()


def clean_reasoning(reasoning: str, question: str, max_overlap: float = 0.6) -> str:
    """Keep facts and gap statements; drop tool noise and restatements of the question."""
    q_tokens = _tokens(question)
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", (reasoning or "").strip()):
        s = sent.strip()
        if not s or _NOISE.search(s) or _QREST.search(s):
            continue
        st = _tokens(s)
        if st and q_tokens and len(st) > 6 and len(st & q_tokens) / len(st) > max_overlap:
            continue
        kept.append(s)
    return " ".join(kept)


def clean_note(reasoning: str, question: str, max_overlap: float = 0.6) -> str:
    """`clean_reasoning` plus dropping forward-planning sentences ("let's search X")."""
    base = clean_reasoning(reasoning, question, max_overlap)
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", base) if s.strip() and not _PLAN.search(s))


def _oneline(text: str) -> str:
    return " ".join((text or "").split())


def _cut(text: str, n: int) -> str:
    return truncate_tokens(text, n, tail="")


# --- the query styles ----------------------------------------------------------------

def memory_query(question: str, current: str, notes: Sequence[str], *, per_visit_tokens: int = 64,
                 budget_tokens: int = 256, min_words: int = 8) -> str:
    lines = [f"[Q] {question}", f"[Now] {current}"]
    kept, used = [], 0
    for raw in reversed(list(notes)):
        cleaned = clean_reasoning(raw, question)
        if len(cleaned.split()) < min_words:
            continue
        snippet = _cut(cleaned, per_visit_tokens)
        n = count_tokens(snippet)
        if used + n > budget_tokens:
            break
        kept.append(snippet)
        used += n
    if kept:
        lines.append("[Memory] " + " ; ".join(kept))
    return "\n".join(lines)


def _title_of(text: str, max_tokens: int = 10) -> str:
    first = (text or "").split("\n")[0].strip()
    return truncate_tokens(first, max_tokens, tail="...") if count_tokens(first) > max_tokens else first


def prevdoc_query(question: str, current: str, interactions: Sequence[dict], *,
                  per_doc_tokens: int = 48, budget_tokens: int = 256) -> str:
    lines = [f"[Q] {question}", f"[Now] {current}"]
    entries, used = [], 0
    for it in reversed(list(interactions)):
        docs = [(d, t) for d, t, _ in it.get("visits", []) if t]
        if not docs:
            continue
        doc_strs = [f"{_title_of(t)}: {_cut(t, per_doc_tokens)}" for _, t in docs]
        entry = f"{it['query']} -> " + " ; ".join(doc_strs)
        n = count_tokens(entry)
        if used + n > budget_tokens:
            break
        entries.append(entry)
        used += n
    if entries:
        lines.append("[Prev] " + " | ".join(entries))
    return "\n".join(lines)


def _oneline(text: str) -> str:
    return " ".join((text or "").split())


def structured_query(variant: str, question: str, current: str, interactions: Sequence[dict], *,
                     per_note_tokens: int = 128, pre_reasoning: str = "") -> str:
    if variant == "i0":
        return current
    if variant == "i8":
        # AgentIR's two fields, as DIVER renders them: the issuing turn's reasoning verbatim
        # (not one-lined), "Empty" when there is none, then the sub-query. No main question, no
        # previous interactions.
        return f"Reasoning: {pre_reasoning or 'Empty'}\n\nQuery: {current}"
    lines = [f"Main Question: {question}"]
    if variant == "i9":
        # the <think> of the turn that issued this search, one line, untruncated; "<empty>" when
        # there is none (the model card's exact rule)
        lines.append(f"Current Reasoning: {_oneline(pre_reasoning) or '<empty>'}")
    lines.append(f"Current Subquery: {current}")
    if variant == "i1":
        return "\n".join(lines)
    with_docs, with_notes, tags, doc_cap = _VARIANT_CFG[variant]
    if not interactions:
        lines.append("Previous Interactions: <empty>")
        return "\n".join(lines)
    lines.append("Previous Interactions:")
    for k, it in enumerate(interactions, 1):
        lines.append(f"Previous SubQuery {k}: {it['query']}")
        if with_docs:
            docs = []
            for d, text, _ in it.get("visits", []):
                if text:
                    snip = _cut(_oneline(text), doc_cap)
                    docs.append(f"[docs_id:{d}] {snip}" if tags else snip)
            lines.append("Visited Documents: " + (" ; ".join(docs) if docs else "<empty>"))
        if with_notes:
            notes = []
            for d, _, raw in it.get("visits", []):
                cleaned = clean_note(raw, question)
                if cleaned:
                    snip = _cut(_oneline(cleaned), per_note_tokens)
                    notes.append(f"[docs_id:{d}] {snip}" if tags else snip)
            lines.append("Visited Document Notes: " + (" ; ".join(notes) if notes else "<empty>"))
    return "\n".join(lines)


def render_query(style: str, question: str, current: str, interactions: Sequence[dict],
                 pre_reasoning: str = "") -> str:
    """One entry point for every style.

    ``interactions`` are the searches BEFORE this one, oldest first:
    ``[{"query": sub_query, "visits": [(doc_id, doc_text, reasoning_after_reading), ...]}]``.
    ``pre_reasoning`` is the agent's reasoning in the turn that issues this search (i9 only).
    """
    if style not in STYLES:
        raise ValueError(f"unknown query style {style!r}; choose from {STYLES}")
    if style == "plain" or style == "i0":
        return current
    if style == "mem":
        notes = [r for it in interactions for _, _, r in it.get("visits", []) if r]
        return memory_query(question, current, notes)
    if style == "docs":
        return prevdoc_query(question, current, interactions)
    return structured_query(style, question, current, interactions, pre_reasoning=pre_reasoning)


__all__ = ["STYLES", "DEFAULT_STYLE", "INSTRUCTIONS", "instruction_prefix", "render_query",
           "reasoning_text", "clean_reasoning", "clean_note", "memory_query", "prevdoc_query",
           "structured_query"]
