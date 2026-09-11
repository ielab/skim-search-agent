"""One-shot RAG: rank once, put the top documents in one prompt, one model call, read the answer.

No loop and no tools. The harness names a ranker (`bm25`, `dense`, `hybrid`) and a depth. It
is the loop-free baseline the Sieve paper compares the agents against.

Lengths: every document is cut to an equal share of the prompt budget, measured with the model
ruler (`agent_search.tokens.truncate_tokens`). The budget is `AGENT_CTX_TOKENS` (the run's
`agent.ctx_tokens`) minus the completion reserve (`RAG_MAX_TOKENS`, default 4000) minus a fixed
margin for the instruction and the question. The record is one step named `retrieve` and the
answer read from the one generation, scored like a loop episode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from agent_search.harness.base import Harness, HarnessContext, HarnessResult, trajectory_from_steps
from agent_search.tokens import count_tokens, truncate_tokens

SYSTEM_PROMPT = (
    "You are a careful research assistant. Read the question and the numbered documents below, "
    "then answer using ONLY the documents. Answer with ONLY the short answer span inside "
    "<answer></answer> tags — no explanation, no extra text.")

TOP_K = 5                       # the paper's depth
FIXED_MARGIN_TOKENS = 512       # instruction, question, document headers, chat template


def prompt_budget() -> int:
    """Model tokens available for the stuffed documents."""
    ctx = int(os.environ.get("AGENT_CTX_TOKENS", "115000"))
    reserve = int(os.environ.get("RAG_MAX_TOKENS", "4000"))
    return max(1000, ctx - reserve - FIXED_MARGIN_TOKENS)


def rank(ranker: str, engines, query: str, k: int) -> list:
    """Ranked doc ids from one engine; `hybrid` is the run's fused engine."""
    if ranker == "bm25":
        return list(engines["bm25"].search(query, k=k) or [])
    if ranker == "dense":
        return list(engines["dense"].top_k_doc_ids(query, k=k) or [])
    if ranker == "hybrid":
        return list(engines["hybrid"].search(query, k))
    raise ValueError(f"unknown ranker {ranker!r} (bm25, dense, hybrid)")


def stuff(doc_ids: Sequence[str], ubyid, budget: int) -> list:
    """[(doc_id, title, text)] in rank order, each text cut to an equal share of `budget`."""
    per_doc = max(200, budget // max(len(doc_ids), 1))
    out = []
    for doc_id in doc_ids:
        u = ubyid.get(doc_id)
        if u is None:
            continue
        title = u.title or u.qualname or doc_id
        body = u.body or u.code or ""
        out.append((doc_id, title, truncate_tokens(body, per_doc, " …(truncated — whole-document cap)")))
    return out


def messages(question: str, hits: Sequence[tuple]) -> list:
    """The one-turn prompt: the instruction, then the question and the numbered documents."""
    if hits:
        blocks = [f"[{i}] {title!r} (doc_id={doc_id})\n{text}" for i, (doc_id, title, text) in enumerate(hits, 1)]
        docs = "\n\n".join(blocks)
    else:
        docs = "(no documents retrieved)"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nDocuments:\n{docs}"}]


@dataclass(frozen=True)
class OneShotRag(Harness):
    ranker: str = "bm25"
    k: int = TOP_K

    @property
    def name(self) -> str:                  # type: ignore[override]
        return f"rag_{self.ranker}"

    def engines(self) -> tuple:
        return {"bm25": ("bm25",), "dense": ("dense",), "hybrid": ("hybrid",)}[self.ranker]

    def prompts(self) -> list:
        return [SYSTEM_PROMPT]

    def run(self, question: str, ctx: HarnessContext) -> HarnessResult:
        from agent_search.agent.loop import Step
        doc_ids = rank(self.ranker, ctx.engines, question, self.k)
        hits = stuff(doc_ids, ctx.ubyid, prompt_budget())
        msgs = messages(question, hits)
        raw = ctx.generate(msgs) if ctx.generate is not None else ""
        step = Step(name="retrieve", args={"query": question, "k": len(doc_ids)},
                    observation=f"({len(doc_ids)} matches stuffed into one prompt)", raw_output=raw or "")
        return HarnessResult(trajectory=trajectory_from_steps([step], doc_ids, raw), surfaced=list(doc_ids))

    def prompt_tokens(self, msgs: list) -> int:
        return sum(count_tokens(m["content"]) for m in msgs)


__all__ = ["OneShotRag", "SYSTEM_PROMPT", "TOP_K", "prompt_budget", "rank", "stuff", "messages"]
