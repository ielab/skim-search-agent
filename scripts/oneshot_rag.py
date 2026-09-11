"""One-shot RAG baseline: NO agent loop. Per instance, retrieve top-k docs, stuff them into
one prompt with the question, one model call, parse <answer>. The simplest possible deep-research
baseline: no search->fetch, no multi-turn tool use, just "retrieve then answer".

Three retrieval variants (--retriever {bm25,dense,hybrid}), sharing the same corpus + `Instance`
loading the agent harness uses (agent_search.evaluation.datasets.load_dataset_by_name), so results are
comparable to the agent_research*/agent_research_bm25/agent_research_dense/agent_research_hybrid
conditions run through agent_search/evaluation/run_eval.py:

  bm25   Same text blob (title+body) the `search_visit` strategy's `bm25_search` tool builds
         (agent_search/tools/search_bm25/tool.py), through the same env `BM25_BACKEND`-selectable
         engine the agent harness's bm25-family arms use
         (agent_search.retrievers.lexical.build_bm25_engine): 'local' (default)
         BM25Local, the dependency-free approximation; 'pyserini', canonical Lucene BM25,
         persisted under indexes/bm25_pyserini/<corpus_key>/lucene/ (built once, reused).
         Override with --bm25-backend {local,pyserini} (defaults from env BM25_BACKEND).
  dense  DenseBelief (agent_search.retrievers.dense.belief), the same
         persisted doc-embedding cache Baseline 1's `research_dense` condition (the
         `search_visit_dense` strategy's `dense_search` tool) uses, default BAAI/bge-base-en-v1.5,
         indexes/dense/BAAI__bge-base-en-v1.5-sl1024/<corpus_key>/ (env `DENSE_MODEL` overrides,
         e.g. Qwen/Qwen3-Embedding-0.6B -> indexes/dense/Qwen__Qwen3-Embedding-0.6B-sl1024/).
         Missing cache -> a clear error (this baseline does not live-encode a whole corpus at
         eval time either).
  hybrid Reciprocal Rank Fusion (RRF, k=`agent_search.tools.budgets.RRF_K`=60) of the same
         bm25 engine (above) and the same dense engine (above), each queried to
         `agent_search.tools.budgets.HYBRID_POOL`=100 depth before fusion, the same formula/pool
         depths the `search_visit_hybrid` strategy's `hybrid_search` tool/`rrf_fuse` use for the
         `research_hybrid` condition (reused via import, not reimplemented). Needs both the bm25
         engine and the persisted dense cache the two variants above need individually; a missing
         dense cache raises the same clear error as --retriever dense.

Each retrieved doc is capped at MAX_VISIT_TOKENS (the same whitespace-token cap
agent_search.tools.budgets uses for a whole-doc `visit`), so the k=5 stuffed docs are comparable
in size to what the agent arms see per fetch/visit call.

Usage (PYTHONPATH=${REPO_ROOT:-.}, the project's `python`):
  python scripts/oneshot_rag.py --dataset hotpotqa_structured --retriever bm25 \\
      --model gpt-4o-mini --api-base https://api.openai.com/v1 --limit 50

  python scripts/oneshot_rag.py --dataset browsecomp_plus_structured --retriever dense \\
      --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --api-base http://127.0.0.1:8101/v1

Writes runs/_oneshot/<dataset>/<retriever>/rows.jsonl, one row per instance:
  {instance_id, question, gold_answer, final_answer, retrieved_ids,
   prompt_tokens, completion_tokens, n_steps: 1}

Resumable: re-running with the same --out-dir skips instances whose row is already in
rows.jsonl (no 'error' key); rows WITH an 'error' (e.g. a context-overflow instance) are
retried and their stale row replaced. Only the missing/broken instances recompute, see
`_load_resume_state`/`_atomic_write_rows`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

from agent_search.tokens import cap_tokens as _cap_tokens
from agent_search.corpus.units import CodeUnit, units_from_documents
from agent_search.retrievers.dense.belief import default_model as _default_dense_model
from agent_search.evaluation.datasets import Instance, load_dataset_by_name
from agent_search.evaluation.doc_scoring import extract_answer_span
from agent_search.evaluation.run_eval import _corpus_key

DENSE_MODEL = _default_dense_model()
TOP_K = 5                          # fixed retrieval depth (not a CLI knob, see module docstring)
DEFAULT_MODEL = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
DEFAULT_API_BASE = "http://127.0.0.1:8101/v1"     # matches oneshot_rag.sbatch's served port
# DENSE_MODEL: resolved from dense.belief.default_model() (same model the `dense_search` tool/
# DenseBelief use, env `DENSE_MODEL`-overridable, see that module) rather than a second
# hardcoded literal here.

# The completion budget must be env-configurable: a hardcoded max_tokens=512 lets the Tongyi
# reasoning model burn its whole completion budget inside <think>...</think> and never reach
# <answer>, so the row scores empty. Default is 4000; override with ONESHOT_MAX_TOKENS.
DEFAULT_MAX_TOKENS = int(os.environ.get("ONESHOT_MAX_TOKENS", "4000"))
# The doc-stuffing budget below must be counted in the same unit
# vLLM's --max-model-len enforces (real BPE tokens), not whitespace words, see
# `_get_tokenizer`/`_cap_real_tokens`. MODEL_MAX_CONTEXT_TOKENS matches oneshot_rag.sbatch's
# `--max-model-len 131072`; SAFETY_MARGIN_TOKENS covers the system prompt, question, chat-template
# special tokens, and each doc's "[i] 'title' (doc_id=...)" wrapper text, none of which are
# counted against the per-doc split.
MODEL_MAX_CONTEXT_TOKENS = int(os.environ.get("ONESHOT_MODEL_MAX_CONTEXT", "131072"))
SAFETY_MARGIN_TOKENS = int(os.environ.get("ONESHOT_SAFETY_MARGIN", "2000"))

# SAFETY_MARGIN_TOKENS above only reserves room
# at the PER-DOC BUDGETING stage (stuff_docs splits MODEL_MAX_CONTEXT_TOKENS - DEFAULT_MAX_TOKENS
# - SAFETY_MARGIN_TOKENS across the k docs, each capped by its OWN body-text token count). It never
# re-measures the ASSEMBLED prompt: system + question + "[i] 'title' (doc_id=...)" wrappers, and
# the chat template vLLM applies server-side (role headers, BOS/generation-prompt special
# tokens, the actual gap), none of which are counted against any single doc's cap. On the longest instances
# that untracked overhead was 1-70 tokens, just enough to push the real prompt (127,073 tokens) over
# vLLM's rejection threshold (127,072 = 131,072 - 4000) despite every per-doc cap being respected.
# FIT_MARGIN_TOKENS is the hard floor `stuff_docs_fit` re-checks the FULL assembled/chat-templated
# prompt against: (MODEL_MAX_CONTEXT_TOKENS - max_tokens - FIT_MARGIN_TOKENS). 256 is a deliberately
# generous multiple of the observed 1-70 token gap, covers chat-template drift across model/tokenizer
# versions without needing to special-case it, and is re-verified by actually rendering the prompt
# through the model's own chat template when available (see `_render_full_prompt`), not estimated.
FIT_MARGIN_TOKENS = int(os.environ.get("ONESHOT_FIT_MARGIN", "256"))

SYSTEM_PROMPT = (
    "You are a careful research assistant. Read the question and the numbered documents below, "
    "then answer using ONLY the documents. Answer with ONLY the short answer span inside "
    "<answer></answer> tags — no explanation, no extra text.")


# --- retrieval: bm25 / dense, both over the same shared document corpus -----------------------

def _units_and_key(instances: Sequence[Instance]) -> tuple[list[CodeUnit], str]:
    """The shared document corpus (one build for every instance in a doc dataset, same
    `docs` list, same `corpus_id`) + its stable cache key (`agent_search.evaluation.run_eval._corpus_key`,
    the same key `agent_search/evaluation/run_eval.py` and `agent_search/evaluation/build_indexes.py` use, so the dense
    variant resolves the same persisted embedding cache)."""
    inst0 = instances[0]
    units = units_from_documents(inst0.docs or [])
    return units, _corpus_key(inst0)


def _build_bm25_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                       key: Optional[str] = None, backend: Optional[str] = None):
    """Exactly how the `bm25_search` tool builds its engine, same text blob
    (`qualname code`, i.e. title+body). `backend` (default: env BM25_BACKEND, else 'local')
    selects BM25Local (dependency-free approximation) vs BM25Pyserini (canonical Lucene BM25,
    persisted under `index_root/bm25_pyserini/<key>/lucene/`) via the same
    `agent_search.retrievers.lexical.build_bm25_engine` the agent harness's bm25-family arms
    use (agent_search/tools/search_bm25/tool.py), so a oneshot bm25 run and an agent_research_bm25
    run backed by the same BM25_BACKEND are the same retrieval engine."""
    from agent_search.retrievers.lexical import build_bm25_engine
    resolved = (backend or os.environ.get("BM25_BACKEND") or "local").strip().lower()
    with _env_override("BM25_BACKEND", resolved):
        return build_bm25_engine(units, index_root=index_root, key=key)


@contextlib.contextmanager
def _env_override(name: str, value: str):
    """Temporarily set an env var for the duration of the `with` block, restoring whatever
    was there before (or removing it) on exit, used so `--bm25-backend` can override
    BM25_BACKEND for exactly the one `build_bm25_engine` call above without mutating this
    process's environment for anything else that later reads the env var."""
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _build_dense_engine(units: Sequence[CodeUnit], key: str, index_root: str = "indexes"):
    """The same cached-embedding path Baseline 1's `research_dense` condition (the
    `search_visit_dense` strategy's `dense_search` tool) uses, a
    persisted `indexes/dense/BAAI__bge-base-en-v1.5-sl1024/<key>/` cache. FAILS LOUD (does
    not live-encode the corpus) if that cache is missing, mirroring the `search_visit_dense`
    strategy's own dense-cache check."""
    from agent_search.retrievers.dense import DenseRetriever
    from agent_search.retrievers.dense.belief import DenseBelief

    probe = DenseRetriever(model=DENSE_MODEL, index_root=index_root)
    if not probe.is_cached(key):
        raise RuntimeError(
            f"oneshot_rag --retriever dense needs a persisted dense doc-embedding cache for "
            f"corpus key {key!r} at {probe._cache_dir(key)!r} — none found. Prebuild it with: "
            f"python -m agent_search.evaluation.build_indexes --retriever dense --model {DENSE_MODEL} ... "
            f"(this baseline does not live-encode the corpus at eval time).")
    return DenseBelief(model=DENSE_MODEL, index_root=index_root).build_or_load(units, key=key)


def _build_hybrid_engine(units: Sequence[CodeUnit], index_root: str = "indexes",
                         key: Optional[str] = None, backend: Optional[str] = None) -> tuple:
    """`(bm25_engine, dense_engine)`, the two engines `retrieve_top_k`'s
    `hybrid` branch RRF-fuses. Reuses `_build_bm25_engine`/`_build_dense_engine` verbatim (same
    engines, same fail-loud missing-cache contract as --retriever bm25/dense individually);
    nothing bm25-only or dense-only is reimplemented here."""
    bm25_engine = _build_bm25_engine(units, index_root=index_root, key=key, backend=backend)
    dense_engine = _build_dense_engine(units, key, index_root=index_root)
    return bm25_engine, dense_engine


def retrieve_top_k(retriever: str, engine, query: str, k: int = TOP_K) -> list[str]:
    """doc_ids, ranked, for `query`, `bm25`: BM25Local.search; `dense`: DenseBelief's
    memoized-encode cosine top-k (same call the `dense_search` tool makes); `hybrid`
    RRF-fuses a bm25 pool and a dense pool, `engine` a `(bm25_engine,
    dense_engine)` pair (see `_build_hybrid_engine`), same `HYBRID_POOL`/`RRF_K`/`rrf_fuse`
    formula the `hybrid_search` tool uses for research_hybrid,
    imported (not reimplemented) so the two stay byte-identical by construction."""
    if retriever == "bm25":
        return list(engine.search(query, k=k) or [])
    if retriever == "dense":
        return list(engine.top_k_doc_ids(query, k=k) or [])
    if retriever == "hybrid":
        from agent_search.tools.budgets import HYBRID_POOL, RRF_K
        from agent_search.tools.common import rrf_fuse
        bm25_engine, dense_engine = engine
        bm25_ids = list(bm25_engine.search(query, k=HYBRID_POOL) or [])
        dense_ids = list(dense_engine.top_k_doc_ids(query, k=HYBRID_POOL) or [])
        return rrf_fuse(bm25_ids, dense_ids, k=RRF_K, topk=k)
    raise ValueError(f"unknown --retriever {retriever!r} (choose bm25, dense, or hybrid)")


# --- token accounting: real BPE tokens (via the model's own tokenizer), not whitespace words ---
# The stuffing budget must be counted in real BPE tokens, not whitespace words
# (`_cap_tokens`-style text.split()): the model's actual limit is BPE tokens, and a 120,000-word
# budget can deterministically produce a 130,561-BPE-token prompt on bm25 rows, over the
# 131,072-max_tokens limit, killing rows with a context-overflow error.

_TOKENIZER_CACHE: dict = {}


def _get_tokenizer(model: str):
    """The model's own HF tokenizer, loaded once offline (HF_HUB_OFFLINE=1) and cached in
    `_TOKENIZER_CACHE`, so doc-stuffing is budgeted in the same unit vLLM's --max-model-len
    enforces. Returns None (callers fall back to `_cap_real_tokens`'s calibrated word estimate)
    if the tokenizer can't be loaded offline, e.g. `model` isn't an HF repo id (a real OpenAI
    model name) or its files aren't in the local HF cache. Checked: the Tongyi-DeepResearch-30B-A3B
    tokenizer IS present in the local HF cache, so the real-token path is the one actually used
    in production runs of this script."""
    if model in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model]
    tok = None
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception:
        tok = None
    _TOKENIZER_CACHE[model] = tok
    return tok


# Fallback calibration (used only when `_get_tokenizer` can't load a real tokenizer): measured
# directly on the buggy bm25 run's stuffed blob, 120,000 whitespace words -> 130,561 real BPE
# tokens for the Tongyi tokenizer, i.e. ~1.088 tokens/word. A conservative 1.25 divisor (higher
# than the observed 1.088 ratio) plus a fixed word-budget haircut keeps the word-based estimate
# comfortably under the real token count even if the true ratio drifts across corpora.
_WORD_TO_TOKEN_DIVISOR = 1.25
_WORD_BUDGET_MARGIN = 200

# Same calibration run (buggy bm25 blob: 120,000 words -> 130,561 real tokens), but for
# ESTIMATING an actual token count (stuff_docs_fit's fit-check) rather than conservatively
# CAPPING one, so this uses the observed ratio directly, not the deliberately-padded
# _WORD_TO_TOKEN_DIVISOR above (which under-budgets on purpose to leave headroom).
_MEASURED_WORD_TO_TOKEN_RATIO = 1.088


def _cap_real_tokens(text: str, n: int, model: str, tail: str = " …(truncated)") -> str:
    """Cap `text` to at most `n` tokens, counted the same way the served model counts them (its
    own tokenizer, via `_get_tokenizer`), so a stuffed prompt built from these caps can never
    exceed the budget it was capped to, by construction. Falls back to a calibrated whitespace-word
    cap (`_WORD_TO_TOKEN_DIVISOR`) if the real tokenizer isn't available offline for `model`."""
    if n <= 0:
        return text
    tok = _get_tokenizer(model)
    if tok is not None:
        ids = tok.encode(text)
        if len(ids) <= n:
            return text
        tail_ids = tok.encode(tail)
        keep = max(0, n - len(tail_ids))
        return tok.decode(ids[:keep]) + tail
    word_budget = max(1, int(n / _WORD_TO_TOKEN_DIVISOR) - _WORD_BUDGET_MARGIN)
    return _cap_tokens(text, word_budget, tail)


# --- prompt: ONE user turn, question + k stuffed docs ------------------------------------------

def stuff_docs(doc_ids: Sequence[str], ubyid: dict,
               max_tokens: int = int(os.environ.get("ONESHOT_DOC_CAP", "0")),
               model: str = DEFAULT_MODEL) -> list[tuple]:
    """[(doc_id, title, capped_text), ...] for the retrieved `doc_ids`, in rank order. Each
    doc's body is capped at `max_tokens` REAL tokens (see `_cap_real_tokens`, counted with
    `model`'s own tokenizer when available); 0 (the default, env ONESHOT_DOC_CAP) means UNCAPPED
   , the classic one-shot RAG stuffs FULL documents (user-set design: the no-agent baseline must
    feed literal everything; only guard is the model's own context). Non-zero mirrors convention
    the `visit` tool's whole-doc read caps (`agent_search.tokens.cap_tokens`/MAX_VISIT_TOKENS),
    so a stuffed doc here is comparable in size to what an agent arm sees per fetch/visit."""
    out = []
    for doc_id in doc_ids:
        u = ubyid.get(doc_id)
        if u is None:
            continue
        title = u.title or u.qualname or doc_id
        body = u.body or u.code or ""
        # "full content" bounded by physics: with no per-doc cap, fit the model's context by
        # splitting a total stuffing budget (env ONESHOT_TOTAL_BUDGET, default derived from
        # MODEL_MAX_CONTEXT_TOKENS - DEFAULT_MAX_TOKENS - SAFETY_MARGIN_TOKENS, so the stuffed
        # prompt can never exceed the model's real max-model-len by construction) evenly across
        # the k docs. A 400-context-overflow taught us the unbounded version is impossible on
        # real corpora (doc p95 22k, max ~930k tokens).
        if not max_tokens:
            default_budget = max(1000, MODEL_MAX_CONTEXT_TOKENS - DEFAULT_MAX_TOKENS
                                 - SAFETY_MARGIN_TOKENS)
            total_budget = int(os.environ.get("ONESHOT_TOTAL_BUDGET", str(default_budget)))
            max_tokens = max(1000, total_budget // max(len(doc_ids), 1))
        text = body if not max_tokens else _cap_real_tokens(
            body, max_tokens, model, " …(truncated — this is the whole-doc cap)")
        out.append((doc_id, title, text))
    return out


def _real_token_len(text: str, model: str) -> int:
    """Actual real-token count of `text` (the model's own tokenizer when available), falling
    back to the calibrated `_MEASURED_WORD_TO_TOKEN_RATIO` estimate, same fallback contract as
    `_cap_real_tokens`/`_get_tokenizer`."""
    tok = _get_tokenizer(model)
    if tok is not None:
        return len(tok.encode(text))
    return int(len(text.split()) * _MEASURED_WORD_TO_TOKEN_RATIO)


def _render_full_prompt(messages: list[dict], model: str) -> str:
    """The prompt text AS THE SERVER ACTUALLY SEES IT: rendered through the model's own chat
    template (role wrappers, BOS/generation-prompt special tokens) when the tokenizer exposes
    one, not just the raw concatenated message content, which is what under-counted the 11/200
    overflowing bm25 rows (the chat template's own overhead tokens were invisible to the old
    per-doc-budget accounting)."""
    tok = _get_tokenizer(model)
    if tok is not None and getattr(tok, "chat_template", None) is not None:
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # No tokenizer, or it has no chat template registered: best-effort raw rendering (still a
    # real under-estimate of server-side overhead, but the only text we have to measure).
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def stuff_docs_fit(question: str, doc_ids: Sequence[str], ubyid: dict, model: str,
                   max_tokens: int = DEFAULT_MAX_TOKENS
                   ) -> tuple[list[tuple], list[dict], int]:
    """`stuff_docs` + `build_messages`, then VERIFY the fully assembled prompt (chat-template
    rendered, see `_render_full_prompt`) actually fits under
    `MODEL_MAX_CONTEXT_TOKENS - max_tokens - FIT_MARGIN_TOKENS` real tokens, re-measuring the
    ACTUAL text, not trusting the per-doc budget sum (see FIT_MARGIN_TOKENS' comment for why that
    sum can still under-count by 1-70 tokens on the longest instances).

    If it doesn't fit: shrink the CURRENTLY LARGEST doc's per-doc token cap by the measured
    overage (floor 1 token) and re-stuff every doc against its (possibly now-smaller) cap, then
    re-measure. Convergence is guaranteed: each iteration strictly decreases the sum of per-doc
    caps by at least 1 while the prompt's fixed overhead (system prompt, question, doc-index
    wrappers, chat-template tokens) is unchanged, so the assembled prompt's token count is
    bounded below and the loop can only run for a finite number of iterations before either the
    prompt fits or every doc's cap has been driven to 0 (in which case the loop stops, an
    empty-docs prompt cannot be shrunk further, and in practice is far under FIT_MARGIN_TOKENS
    since it is only the fixed system+question overhead the margin was sized to absorb).
    """
    hits = stuff_docs(doc_ids, ubyid, model=model)
    caps = {doc_id: _real_token_len(text, model) for doc_id, _title, text in hits}
    limit = MODEL_MAX_CONTEXT_TOKENS - max_tokens - FIT_MARGIN_TOKENS

    messages = build_messages(question, hits)
    n_tokens = _real_token_len(_render_full_prompt(messages, model), model)

    while n_tokens > limit and any(c > 0 for c in caps.values()):
        overage = n_tokens - limit
        largest_id = max(caps, key=lambda d: caps[d])
        caps[largest_id] = max(0, caps[largest_id] - max(1, overage))

        hits = []
        for doc_id in doc_ids:
            u = ubyid.get(doc_id)
            if u is None:
                continue
            title = u.title or u.qualname or doc_id
            body = u.body or u.code or ""
            cap = caps.get(doc_id, 0)
            text = (_cap_real_tokens(body, cap, model,
                    " …(truncated — this is the whole-doc cap)") if cap > 0 else "")
            hits.append((doc_id, title, text))

        messages = build_messages(question, hits)
        n_tokens = _real_token_len(_render_full_prompt(messages, model), model)

    return hits, messages, n_tokens


def build_messages(question: str, hits: Sequence[tuple]) -> list[dict]:
    """The one-turn prompt: system (short instruction) + user (question + numbered docs,
    each `[rank] 'title' (doc_id=...)` followed by its capped text)."""
    if hits:
        blocks = [f"[{i}] {title!r} (doc_id={doc_id})\n{text}"
                 for i, (doc_id, title, text) in enumerate(hits, start=1)]
        docs_block = "\n\n".join(blocks)
    else:
        docs_block = "(no documents retrieved)"
    user = f"Question: {question}\n\nDocuments:\n{docs_block}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_answer(text: str) -> str:
    """The model's short answer span, reuses the same `<answer>...</answer>` extraction the
    agent harness scores with (agent_search.evaluation.doc_scoring.extract_answer_span), so a oneshot row's
    `final_answer` is graded identically to an agent row's."""
    return extract_answer_span(text or "")


# --- model call: an OpenAI-compatible client (served vLLM or a real OpenAI/compatible key) -----

def make_generate(model: str, api_base: str, api_key: Optional[str] = None,
                  temperature: float = 0.6, seed: Optional[int] = 42,
                  max_tokens: int = DEFAULT_MAX_TOKENS, client=None) -> Callable:
    """generate(messages) -> (text, prompt_tokens, completion_tokens). A plain OpenAI client
    against `api_base` (a served vLLM, e.g. Tongyi on :8101, or the real OpenAI API), deliberately
    NOT `agent_search.agent.backbone.openai_compat_generate` (that helper's
    `_repair_open_tag`/`_truncate_at_tool_response` post-processing is tool-call-loop-specific;
    a one-shot completion has no tool-call stop string to repair). `temperature`/`seed` default
    to the harness's own deterministic settings (agent_search.agent.backbone). `max_tokens`
    defaults to DEFAULT_MAX_TOKENS (env ONESHOT_MAX_TOKENS, default 4000): a hardcoded
    512 lets the Tongyi reasoning model burn its whole completion budget inside <think>...</think>
    and never reach <answer>."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=api_base,
                        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def generate(messages: list) -> tuple:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            seed=seed, max_tokens=max_tokens)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return text, prompt_tokens, completion_tokens

    return generate


# --- one instance end to end ---------------------------------------------------------------

def run_instance(inst: Instance, retriever: str, engine, ubyid: dict, generate: Callable,
                 model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    doc_ids = retrieve_top_k(retriever, engine, inst.problem_statement, k=TOP_K)
    hits, messages, _n_tokens = stuff_docs_fit(
        inst.problem_statement, doc_ids, ubyid, model, max_tokens=max_tokens)
    text, prompt_tokens, completion_tokens = generate(messages)
    return {
        "instance_id": inst.instance_id,
        "question": inst.problem_statement,
        "gold_answer": inst.answer,
        "final_answer": parse_answer(text),
        "retrieved_ids": doc_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "n_steps": 1,
    }


# --- per-instance resume: skip done rows, retry errored ones, atomic rewrite -------------------

def _load_resume_state(out_path: str) -> dict[str, dict]:
    """Existing COMPLETED rows from a prior run of `out_path`, keyed by instance_id, tolerant of
    a truncated trailing line (a killed run, same convention as agent_search.evaluation.run_eval._load_rows).

    A row WITH an 'error' key is NOT considered done: it is dropped here, so its instance_id
    falls through to the caller's todo list and gets recomputed, this is what makes a resume
    surgical (e.g. only the 11/200 bm25 rows that overflowed context recompute, not all 200)."""
    done: dict[str, dict] = {}
    if not out_path or not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # run killed mid-append: partial line skipped, instance re-runs
            inst_id = row.get("instance_id")
            if not inst_id:
                continue
            if row.get("error"):
                done.pop(inst_id, None)  # a later error line must drop an earlier good one
            else:
                done[inst_id] = row
    return done


def _atomic_write_rows(out_path: str, rows: Sequence[dict]) -> None:
    """Write `rows` to `out_path` via temp-file + os.replace, a resume run that gets killed
    mid-write leaves the PREVIOUS complete rows.jsonl intact instead of a half-written file."""
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp_path, out_path)


# --- CLI -------------------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="a doc-domain dataset name (agent_search.evaluation.datasets.available_datasets()), "
                         "e.g. hotpotqa_structured, browsecomp_plus_structured.")
    ap.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], required=True)
    ap.add_argument("--bm25-backend", choices=["local", "pyserini"], default=None,
                    help="engine for --retriever bm25/hybrid: 'local' (BM25Local, dependency-free "
                         "approximation) or 'pyserini' (canonical Lucene BM25, persisted "
                         "index). Default: env BM25_BACKEND, else 'local'. Ignored for "
                         "--retriever dense.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/_oneshot/<dataset>/<retriever>/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--index-root", default="indexes")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join("runs", "_oneshot", args.dataset, args.retriever)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rows.jsonl")

    print(f">> loading dataset={args.dataset!r} limit={args.limit}", file=sys.stderr)
    instances = load_dataset_by_name(args.dataset, limit=args.limit)
    if not instances:
        print(f"ERROR: dataset {args.dataset!r} yielded 0 instances.", file=sys.stderr)
        return 1

    # per-instance resume: rows already completed (no 'error') in a prior run of out_path are
    # kept verbatim and skipped; rows that errored (e.g. a context overflow) are dropped and
    # their instance_ids re-run, a resume only recomputes what's actually missing/broken.
    done_rows = _load_resume_state(out_path)
    todo_instances = [inst for inst in instances if inst.instance_id not in done_rows]
    if done_rows:
        print(f">> resume: {len(done_rows)}/{len(instances)} instances already done in "
             f"{out_path!r}, {len(todo_instances)} to (re)run", file=sys.stderr)

    if not todo_instances:
        print(f">> nothing to do: all {len(instances)} instances already in {out_path}.",
             file=sys.stderr)
        return 0

    units, key = _units_and_key(instances)
    ubyid = {u.doc_id: u for u in units}
    print(f">> corpus: {len(units)} units, key={key!r}", file=sys.stderr)

    if args.retriever == "bm25":
        engine = _build_bm25_engine(units, index_root=args.index_root, key=key,
                                    backend=args.bm25_backend)
    elif args.retriever == "dense":
        engine = _build_dense_engine(units, key, index_root=args.index_root)
    else:                                     # "hybrid"
        engine = _build_hybrid_engine(units, index_root=args.index_root, key=key,
                                      backend=args.bm25_backend)

    generate = make_generate(args.model, args.api_base)

    print(f">> running {len(todo_instances)} instances, retriever={args.retriever} "
         f"model={args.model} workers={args.workers}", file=sys.stderr)
    new_rows: list = [None] * len(todo_instances)
    lock = threading.Lock()
    n_done = 0

    def _work(i: int, inst: Instance) -> tuple:
        try:
            return i, run_instance(inst, args.retriever, engine, ubyid, generate, model=args.model)
        except Exception as e:  # one bad instance (context overflow etc.) must not kill the run
            return i, {"instance_id": inst.instance_id, "question": inst.problem_statement,
                       "gold_answer": inst.answer, "final_answer": "",
                       "error": f"{type(e).__name__}: {e}"[:300], "retrieved_ids": [],
                       "prompt_tokens": 0, "completion_tokens": 0, "n_steps": 1}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_work, i, inst) for i, inst in enumerate(todo_instances)]
        for fut in as_completed(futs):
            i, row = fut.result()
            new_rows[i] = row
            with lock:
                n_done += 1
                if n_done % 10 == 0 or n_done == len(todo_instances):
                    print(f"   {n_done}/{len(todo_instances)} done", file=sys.stderr)

    # merge: kept-done rows + freshly (re)computed rows, in the ORIGINAL `instances` order.
    # written atomically so a kill mid-write can't corrupt the previously-good rows.jsonl.
    new_by_id = {row["instance_id"]: row for row in new_rows}
    final_rows = [done_rows.get(inst.instance_id) or new_by_id[inst.instance_id]
                 for inst in instances]
    _atomic_write_rows(out_path, final_rows)
    print(f">> wrote {len(final_rows)} rows to {out_path} "
         f"({len(new_rows)} (re)computed, {len(done_rows)} kept from resume)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
