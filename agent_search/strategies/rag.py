"""One-shot RAG: rank once, put the top documents in one prompt, one model call, read the answer.

No loop and no tools. The strategy names a ranker (`bm25`, `dense`, `hybrid`) and a depth; the
procedure below does the rest. It is the loop-free baseline the Sieve paper compares the agents
against. `scripts/oneshot_rag.py` is the paper's standalone runner with the same instruction,
prompt layout, depth and ranking; it keeps its own record shape under `runs/_oneshot` and fits
the prompt with the served model's tokenizer, where this module uses the library ruler.

Lengths: every document is cut to an equal share of the prompt budget, measured with the model
ruler (`agent_search.core.tokens.truncate_tokens`). The budget is `AGENT_CTX_TOKENS` (the run's
`agent.ctx_tokens`) minus the completion reserve (`RAG_MAX_TOKENS`, default 4000) minus a fixed
margin for the instruction and the question. The record is one step named `retrieve` and the
answer read from the one generation, scored like an agent row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from agent_search.core.tokens import count_tokens, truncate_tokens
from agent_search.strategies.base import Strategy, register_strategy

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


def rank(ranker: str, engines: dict, query: str, k: int) -> list:
    """Ranked doc ids from one engine, or the RRF fusion of BM25 and dense for `hybrid`."""
    if ranker == "bm25":
        return list(engines["bm25"].search(query, k=k) or [])
    if ranker == "dense":
        return list(engines["dense"].top_k_doc_ids(query, k=k) or [])
    if ranker == "hybrid":
        from agent_search.tools.budgets import HYBRID_POOL, RRF_K
        from agent_search.tools.common import rrf_fuse
        bm25_ids = list(engines["bm25"].search(query, k=HYBRID_POOL) or [])
        dense_ids = list(engines["dense"].top_k_doc_ids(query, k=HYBRID_POOL) or [])
        return rrf_fuse(bm25_ids, dense_ids, k=RRF_K, topk=k)
    raise ValueError(f"unknown ranker {ranker!r} (bm25, dense, hybrid)")


def stuff(doc_ids: Sequence[str], ubyid: dict, budget: int) -> list:
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
class OneShotRag:
    """The procedure: `run(question, engines, ubyid, generate)` -> (doc_ids, prompt, raw output)."""
    ranker: str = "bm25"
    k: int = TOP_K

    @property
    def engines(self) -> tuple:
        return {"bm25": ("bm25",), "dense": ("dense",), "hybrid": ("bm25", "dense")}[self.ranker]

    def run(self, question: str, engines: dict, ubyid: dict, generate: Callable) -> tuple:
        doc_ids = rank(self.ranker, engines, question, self.k)
        hits = stuff(doc_ids, ubyid, prompt_budget())
        msgs = messages(question, hits)
        raw = generate(msgs) if generate is not None else ""
        return doc_ids, msgs, raw

    def prompt_tokens(self, msgs: list) -> int:
        return sum(count_tokens(m["content"]) for m in msgs)


rag = register_strategy(Strategy(
    name="rag", description="one-shot RAG: top-5 by BM25 in one prompt, one model call",
    loop=False, procedure=OneShotRag("bm25"), extra_engines=("bm25",)))

rag_dense = register_strategy(Strategy(
    name="rag_dense", description="one-shot RAG: top-5 by the dense model in one prompt, one model call",
    loop=False, procedure=OneShotRag("dense"), extra_engines=("dense",)))

rag_hybrid = register_strategy(Strategy(
    name="rag_hybrid", description="one-shot RAG: top-5 by RRF of BM25 and dense in one prompt, one model call",
    loop=False, procedure=OneShotRag("hybrid"), extra_engines=("bm25", "dense")))


__all__ = ["OneShotRag", "SYSTEM_PROMPT", "TOP_K", "prompt_budget", "rank", "stuff", "messages",
           "rag", "rag_dense", "rag_hybrid"]
