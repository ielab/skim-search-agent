"""OpenAI Agents SDK driver: native function calling for any condition's toolset.

An alternative to agent/loop.py's text-parsed ReAct loop: the tool box's tools become native SDK
`@function_tool`s and `Runner` drives the loop with structured tool-calling (no `<tool_call>` text
scraping). The same tool box (`agent_search.tools.base.ToolBox`, built by the condition's strategy)
is reused as-is; only the driver and the tool surface it presents to the model change.
`run_episode_sdk` returns a plain `SdkTrajectory` (answer, steps, token usage) so it can be scored
exactly like the loop's `Trajectory`.

`make_agent_model` (below) routes any backbone, OpenAI, Gemini, or a served vLLM/Tongyi model,
through the SDK's own tool-calling, so all three providers share this one driver.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# code condition: the terminal is a <fix>...</fix> block in the model's final message, as in the
# loop driver. Native tools handle search/fetch, and the fix is parsed from the final output.
_FIX = re.compile(r"<fix>(.*?)</fix>", re.DOTALL | re.IGNORECASE)

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

from agent_search.agent.loop import _extract_answer
from agent_search.tokens import cap_tokens
from agent_search.agent.backbone import (
    _GEMINI_BASE_URL, is_gemini_model, is_openai_model)

# Returned as `final_output` (set as `final`) when the SDK's own `max_turns` is exhausted, before
# either forcing mechanism below has had a chance to fill in a real answer. This is for logging
# only; it is never persisted as `SdkTrajectory.final_answer` (that field gets "" instead when
# both forcing mechanisms come up empty; see run_episode_sdk's MaxTurnsExceeded branch).
_MAX_TURNS_PLACEHOLDER = "(max turns exceeded — no final answer)"

# Caps (whitespace tokens) on the per-step evidence string the closer and retry forcing calls
# see. Env-overridable so a run with unusually verbose tool args or observations can widen them
# without a code change. A character cap here would silently interact with the token-budget
# knobs elsewhere (see agent_search.tokens's module docstring), so this uses the same token
# ruler as the rest of the library instead of `str(...)[:N]`.
CLOSER_EVIDENCE_ARG_TOKENS = int(os.environ.get("CLOSER_EVIDENCE_ARG_TOKENS", "32"))
CLOSER_EVIDENCE_OBS_TOKENS = int(os.environ.get("CLOSER_EVIDENCE_OBS_TOKENS", "160"))

# Do not ship benchmark traces to OpenAI's hosted tracing platform: it is a privacy concern, and
# it also 400s on non-OpenAI usage payloads (e.g. Gemini). The step trace is kept in SdkTrajectory
# instead. Deferred to first use of `run_episode_sdk` (see `_ensure_tracing_disabled`) rather than
# run at import time, so importing this module has no side effect on the SDK's global tracing
# state.
_tracing_disabled_once = False


def _ensure_tracing_disabled() -> None:
    global _tracing_disabled_once
    if not _tracing_disabled_once:
        set_tracing_disabled(True)
        _tracing_disabled_once = True


# AsyncOpenAI clients are cached per (base_url, api_key) so a single MaxTurnsExceeded episode
# (main run, closer, and up to 2 retries, each calling `make_agent_model` again) reuses one
# connection pool per endpoint instead of opening a fresh one per call.
_async_client_cache: dict[tuple[str, str], "object"] = {}


def _cached_async_openai_client(base_url: str, api_key: str):
    key = (base_url, api_key)
    client = _async_client_cache.get(key)
    if client is None:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        _async_client_cache[key] = client
    return client


def make_agent_model(model: str, *, api_base: Optional[str] = None):
    """Route a model name to what `Agent(model=...)` needs: the SDK analogue of
    `backends.make_generate`, so all providers go through one SDK path (no separate
    chat-completions calls). Reuses the same `is_openai_model`/`is_gemini_model` matchers.

    - OpenAI (`gpt-*`/`o-*`): the bare model string, routed to the SDK's default OpenAI client.
    - Gemini (`gemini*`): an `OpenAIChatCompletionsModel` on Gemini's OpenAI-compatible endpoint
      (the same endpoint `backends.gemini_generate` uses), so Gemini is driven by the SDK's
      tool-calling, not a bespoke call.
    - else (a served vLLM or Tongyi backbone): an `OpenAIChatCompletionsModel` at `api_base` (the
      OpenAI-compatible `vllm serve` endpoint). In-process vLLM has no SDK path; it must be served.

    The underlying `AsyncOpenAI` client (Gemini and served branches) is cached per
    (base_url, api_key) (see `_cached_async_openai_client`), since this is called again for every
    forcing-sequence Agent (closer and retries) within one episode.
    """
    if is_openai_model(model):
        return model
    if is_gemini_model(model):
        client = _cached_async_openai_client(_GEMINI_BASE_URL, os.environ.get("GEMINI_API_KEY", ""))
        return OpenAIChatCompletionsModel(model=model, openai_client=client)
    base = api_base or os.environ.get("VLLM_API_BASE") or "http://localhost:8000/v1"
    client = _cached_async_openai_client(base, os.environ.get("OPENAI_API_KEY", "EMPTY"))
    return OpenAIChatCompletionsModel(model=model, openai_client=client)


@dataclass
class SdkTrajectory:
    final_answer: str
    steps: list            # [(tool_name, args_dict, observation_str), ...] in call order
    input_tokens: int      # total input across all model calls (re-sends the growing context)
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int   # of output_tokens, the reasoning subset (0 for non-reasoning models)
    first_input_tokens: int  # the first model call's input, roughly the fixed initial prompt
                              # (system instructions, tool schemas, question), counted once
    requests: int
    seen: set              # doc_ids the tool box surfaced (gold-doc-coverage metric)
    fix_text: str = ""     # code condition: the <fix> block parsed from the final output (else "")
    # provenance of a forced (non-organic) final_answer, mirroring agent_search.agent.loop.
    # Trajectory.elicitation: None (max_turns never hit, final_output is organic), "ask_retry_inline"
    # (the MaxTurnsExceeded forcing sequence below, the closer call and/or its strict retries,
    # produced the answer), "ask_retry_failed" (that whole sequence exhausted its attempts). API
    # models have no assistant-prefill affordance (that is the loop driver's vLLM-only mechanism,
    # agent_search/agent/forced_answer.py), so this driver's forcing mechanism is ask-and-retry,
    # not prefill. See run_episode_sdk's MaxTurnsExceeded branch for the detail.
    elicitation: Optional[str] = None
    # terminal reason, mirroring agent_search.agent.loop.Trajectory.stopped_reason: "answer" (the
    # SDK's own Runner loop ended organically; final_output was produced without hitting
    # max_turns) or "max_turns" (the SDK raised MaxTurnsExceeded and the forcing sequence above
    # ran, regardless of whether it recovered an answer). The harness needs this to compute
    # timeout rates independently of whether a forced answer happened to land.
    stopped_reason: str = "answer"


def _tools_for(ws):
    """Build native SDK function tools for whatever `ws.tools` exposes: search/fetch (BQL),
    search_v2/fetch_v2 (BQL v2 field-tagged search plus whole-doc read), search_s/fetch_s
    (excerpt listing plus whole-doc read, research_snip), isearch/fetch (Indri graded search
    plus whole-doc read), isearch_v/visit_v (Indri graded search with a content-bearing listing
    plus whole-doc read), isearch_s/fetch (Indri plus snippet listing plus section fetch,
    research_indri_snip),
    bm25_search/visit (bm25 baseline), bm25q_search/visit_q (a hardened bm25 baseline: the same
    retrieval and read as bm25_search/visit, with a query-biased listing snippet),
    search_bv/visit_bv (BQL v2 search plus snippet listing plus whole-doc read, the
    {BQL search} x {whole-doc visit} factorial cell), bm25_search_snip/fetch (bm25 retrieval plus
    snippet listing plus section fetch, the fair-listing sibling of bm25_search/fetch,
    research_bm25_fetch_snip), hybrid_search/visit_h (BM25+dense RRF-fused retrieval plus
    whole-doc read, the hybrid baseline, research_hybrid), hybrid_search_snip/fetch (the same
    RRF-fused retrieval with snippet listing plus section fetch, research_hybrid_fetch_snip),
    dense_search_fp/fetch (the same dense retrieval as dense_search_f, with a plain listing that
    has no per-hit excerpt, plus section fetch), bash/read (the DCI shell). Each delegates to
    `ws.run(name, args)` as-is, reusing the tool box's own argument normalization, and appends to
    a shared `trace`."""
    trace: list = []
    names = set(getattr(ws, "tools", ()))
    out = []

    def record(name, args):
        obs = ws.run(name, args)
        trace.append((name, args, obs))
        return obs

    if "search" in names:
        @function_tool
        def search(query: str, k: int = 5) -> str:
            """Search with a field-tagged Boolean query (term[field], AND/OR/NOT, (), "phrase",
            wildcard*; fields title/body/section/infobox/author/date). Returns ranked docs + their
            section headings + infobox keys — NOT body text."""
            return record("search", {"query": query, "k": k})
        out.append(search)
    if "search_v2" in names:
        @function_tool
        def search_v2(query: str, k: int = 5) -> str:
            """Search with a field-tagged Boolean query (term[field], AND/OR/NOT, "phrase",
            wildcard*; fields title/body/section/infobox/author/date; typed date ranges
            date[1980..1989] / date[<=2023-12]). Returns ranked docs + section names + infobox
            keys, with constraint-coverage feedback on near-misses — NOT body text."""
            return record("search_v2", {"query": query, "k": k})
        out.append(search_v2)
    if "fetch_v2" in names:
        @function_tool
        def fetch_v2(specs: list) -> str:
            """Fetch ONE named section (or 'infobox') of a single document; returns only that
            section's text. `doc` = rank from the last search or a doc_id; `section` = a heading."""
            return record("fetch_v2", {"specs": specs})
        out.append(fetch_v2)
    if "search_s" in names:
        @function_tool
        def search_s(query: str, k: int = 5) -> str:
            """Search with a field-tagged Boolean query (term[field], AND/OR/NOT, (), "phrase",
            wildcard*; fields title/body/section/infobox/author/date). Returns ranked docs + their
            section headings + infobox keys, each with a one-line best-matching excerpt — NOT
            full body text."""
            return record("search_s", {"query": query, "k": k})
        out.append(search_s)
    if "fetch_s" in names:
        @function_tool
        def fetch_s(specs: list) -> str:
            """Fetch ONE named section (or 'infobox') of a single document; returns only that
            section's text. `doc` = rank from the last search or a doc_id; `section` = a heading."""
            return record("fetch_s", {"specs": specs})
        out.append(fetch_s)
    if "isearch" in names:
        @function_tool
        def isearch(query: str, k: int = 5) -> str:
            """Indri structured query over the corpus: #combine/#weight/#band, #odN/#uwN windows,
            #syn, term.field (title/body/section/author/date), #filreq/#filrej,
            #date:between(YYYY-MM-DD YYYY-MM-DD). Graded matching — never returns zero docs.
            Returns ranked docs + section names + a weakest-constraint diagnostic."""
            return record("isearch", {"query": query, "k": k})
        out.append(isearch)
    if "isearch_v" in names:
        @function_tool
        def isearch_v(query: str, k: int = 5) -> str:
            """Indri structured query over the corpus: #combine/#weight/#band, #odN/#uwN windows,
            #syn, term.field (title/body/section/author/date), #filreq/#filrej,
            #date:between(YYYY-MM-DD YYYY-MM-DD). Graded matching — never returns zero docs.
            Returns ranked docs + section names + a weakest-constraint diagnostic + a one-line
            best-matching excerpt per hit."""
            return record("isearch_v", {"query": query, "k": k})
        out.append(isearch_v)
    if "visit_v" in names:
        @function_tool
        def visit_v(doc: str) -> str:
            """Read a ranked document's full text (capped). rank_or_id = rank from the last
            search or a doc_id."""
            return record("visit_v", {"rank": doc})
        out.append(visit_v)
    if "isearch_s" in names:
        @function_tool
        def isearch_s(query: str, k: int = 5) -> str:
            """Indri structured query over the corpus: #combine/#weight/#band, #odN/#uwN windows,
            #syn, term.field (title/body/section/author/date), #filreq/#filrej,
            #date:between(YYYY-MM-DD YYYY-MM-DD). Graded matching — never returns zero docs.
            Returns ranked docs + section names + a weakest-constraint diagnostic, each with a
            one-line best-matching excerpt."""
            return record("isearch_s", {"query": query, "k": k})
        out.append(isearch_s)
    if "search_bv" in names:
        @function_tool
        def search_bv(query: str, k: int = 5) -> str:
            """Search with a field-tagged Boolean query (term[field], AND/OR/NOT, (), "phrase",
            wildcard*, typed date[RANGE] ranges; fields title/body/section/infobox/author/date).
            Returns ranked docs + section names + infobox keys, each with a one-line
            best-matching excerpt — NOT full body text. Then `visit_bv` a ranked doc to read its
            full text."""
            return record("search_bv", {"query": query, "k": k})
        out.append(search_bv)
    if "visit_bv" in names:
        @function_tool
        def visit_bv(doc: str) -> str:
            """Read the FULL text of one document (by its rank from the last search_bv or its
            doc_id)."""
            return record("visit_bv", {"rank": doc})
        out.append(visit_bv)
    if "search_bqld" in names:
        @function_tool
        def search_bqld(query: str, k: int = 5) -> str:
            """Search with a field-tagged Boolean query (term[field], AND/OR/NOT, (), "phrase",
            wildcard*, typed date[RANGE] ranges; fields title/body/section/infobox/author/date).
            Returns ranked docs + section names + infobox keys, each with a one-line
            best-matching excerpt — NOT full body text. Then `visit_bqld` a ranked doc to read
            its full text."""
            return record("search_bqld", {"query": query, "k": k})
        out.append(search_bqld)
    if "visit_bqld" in names:
        @function_tool
        def visit_bqld(doc: str) -> str:
            """Read the FULL text of one document (by its rank from the last search_bqld or its
            doc_id)."""
            return record("visit_bqld", {"rank": doc})
        out.append(visit_bqld)
    if "bm25_search" in names:
        @function_tool
        def bm25_search(query: str, k: int = 5) -> str:
            """Search the corpus with plain keywords (BM25). Returns ranked documents with a short
            opening snippet of each."""
            return record("bm25_search", {"query": query, "k": k})
        out.append(bm25_search)
    if "bm25_search_snip" in names:
        @function_tool
        def bm25_search_snip(query: str, k: int = 10) -> str:
            """Search the corpus with plain keywords (BM25). Returns ranked docs plus their
            section names + infobox keys + a best-matching excerpt for your query — NOT full
            text. Then `fetch` a named section to read."""
            return record("bm25_search_snip", {"query": query, "k": k})
        out.append(bm25_search_snip)
    if "dense_search_f" in names:
        @function_tool
        def dense_search_f(query: str, k: int = 10) -> str:
            """Semantic similarity search over the corpus (dense embeddings, cosine similarity).
            Returns ranked docs plus their section names + infobox keys + a best-matching excerpt
            for your query — NOT full text. Then `fetch` a named section to read it."""
            return record("dense_search_f", {"query": query, "k": k})
        out.append(dense_search_f)
    if "dense_search_fp" in names:
        @function_tool
        def dense_search_fp(query: str, k: int = 10) -> str:
            """Semantic similarity search over the corpus (dense embeddings, cosine similarity).
            Returns ranked docs plus their section names + infobox keys — NOT full text, no
            excerpt. Then `fetch` a named section to read it."""
            return record("dense_search_fp", {"query": query, "k": k})
        out.append(dense_search_fp)
    if "bm25q_search" in names:
        @function_tool
        def bm25q_search(query: str, k: int = 5) -> str:
            """Search the corpus with plain keywords (BM25). Returns ranked documents, each with
            its best-matching excerpt for your query (not just the doc opening)."""
            return record("bm25q_search", {"query": query, "k": k})
        out.append(bm25q_search)
    if "visit_q" in names:
        @function_tool
        def visit_q(doc: str) -> str:
            """Read the FULL text of one document (by its rank from the last search or its
            doc_id)."""
            return record("visit_q", {"rank": doc})
        out.append(visit_q)
    if "hybrid_search" in names:
        @function_tool
        def hybrid_search(query: str, k: int = 5) -> str:
            """Hybrid keyword+semantic search over the corpus: a canonical BM25 (Lucene) ranking
            and a dense-embedding ranking are each computed over a top-100 pool and combined via
            Reciprocal Rank Fusion (RRF, k=60). Returns ranked documents with a short opening
            snippet of each."""
            return record("hybrid_search", {"query": query, "k": k})
        out.append(hybrid_search)
    if "visit_h" in names:
        @function_tool
        def visit_h(doc: str) -> str:
            """Read the FULL text of one document (by its rank from the last search or its
            doc_id)."""
            return record("visit_h", {"rank": doc})
        out.append(visit_h)
    if "hybrid_search_snip" in names:
        @function_tool
        def hybrid_search_snip(query: str, k: int = 10) -> str:
            """Hybrid keyword+semantic search over the corpus (Reciprocal Rank Fusion, RRF k=60,
            over a top-100 BM25 pool and a top-100 dense-embedding pool). Returns ranked docs
            plus their section names + infobox keys + a best-matching excerpt for your query —
            NOT full text. Then `fetch` a named section to read."""
            return record("hybrid_search_snip", {"query": query, "k": k})
        out.append(hybrid_search_snip)
    if "grep" in names:
        @function_tool
        def grep(pattern: str) -> str:
            """Regex-search the code corpus for a pattern; returns matching units/lines (code grep
            baseline)."""
            return record("grep", {"pattern": pattern})
        out.append(grep)
    if "fetch" in names:
        @function_tool
        def fetch(doc: str, section: str) -> str:
            """Fetch ONE named section (or 'infobox') of a single document; returns only that
            section's text. `doc` = rank from the last search or a doc_id; `section` = a heading."""
            return record("fetch", {"specs": [[doc, section]]})
        out.append(fetch)
    if "visit" in names:
        @function_tool
        def visit(doc: str) -> str:
            """Read the FULL text of one document (by its rank from the last search or its doc_id)."""
            return record("visit", {"rank": doc})
        out.append(visit)
    if "get_document" in names:
        @function_tool
        def get_document(docid: str) -> str:
            """Retrieve the full content of one document given its DocID from a search result."""
            return record("get_document", {"docid": docid})
        out.append(get_document)
    if "bash" in names:
        @function_tool
        def bash(command: str) -> str:
            """Run a bash command (grep/rg/ls/find/…) over the corpus file tree (cwd = the export
            dir). Combined stdout+stderr, tail-truncated."""
            return record("bash", {"command": command})
        out.append(bash)
    if "read" in names:
        @function_tool
        def read(path: str, offset: int = 1, limit: int = 40) -> str:
            """Read a 1-indexed line range of one corpus file (`<doc_id>.txt`)."""
            return record("read", {"path": path, "offset": offset, "limit": limit})
        out.append(read)
    return out, trace


def run_episode_sdk(ws, question: str, *, model: str = "gpt-4o-mini", api_base: Optional[str] = None,
                    max_turns: int = 12, instructions: str) -> SdkTrajectory:
    """Drive one episode over `ws` via the Agents SDK. Tools come from `ws.tools`; coaching is
    `instructions`, which is required (raises `ValueError` if empty or None), since the caller
    (`agent_search.evaluation.agent_runner.ConditionAgent._run_sdk`) always builds the condition's
    rendered system prompt and passes it in; there is no per-condition coaching fallback in this
    module. `model` routes through `make_agent_model` (OpenAI, Gemini, or served vLLM). Hitting
    `max_turns` returns whatever was produced (partial answer plus the tool trace), not a crash,
    so a hard question cannot abort a run."""
    if not instructions:
        raise ValueError("run_episode_sdk requires a non-empty `instructions` prompt")
    _ensure_tracing_disabled()
    from agents.exceptions import MaxTurnsExceeded
    tools, trace = _tools_for(ws)
    agent = Agent(name="research", instructions=instructions,
                  model=make_agent_model(model, api_base=api_base), tools=tools)
    final, usage, raws = "", None, []
    elicitation: Optional[str] = None
    stopped_reason = "answer"
    try:
        result = Runner.run_sync(agent, question, max_turns=max_turns)
        final = str(result.final_output or "")
        usage = result.context_wrapper.usage
        raws = getattr(result, "raw_responses", None) or []
    except MaxTurnsExceeded as e:
        # regardless of whether the forcing sequence below recovers an answer, the SDK's own loop
        # never finished organically. The harness needs this to compute timeout rates
        # independently of whether a forced answer happened to land (see SdkTrajectory docstring).
        stopped_reason = "max_turns"
        # `_MAX_TURNS_PLACEHOLDER` is for logging only. `final` (persisted as `final_answer`)
        # stays "" until, and unless, one of the forcing calls below actually produces text.
        final = ""
        # the accumulated usage still lives on the exception's run_data (do not report 0 tokens
        # for an episode that actually made several model calls before hitting the cap).
        rd = getattr(e, "run_data", None)
        usage = getattr(getattr(rd, "context_wrapper", None), "usage", None)
        raws = getattr(rd, "raw_responses", None) or []

        def _fold_usage(extra_result) -> None:
            nonlocal usage
            u = getattr(getattr(extra_result, "context_wrapper", None), "usage", None)
            if usage is not None and u is not None:      # fold the extra call into the totals
                usage.input_tokens = int(getattr(usage, "input_tokens", 0)) + int(getattr(u, "input_tokens", 0) or 0)
                usage.output_tokens = int(getattr(usage, "output_tokens", 0)) + int(getattr(u, "output_tokens", 0) or 0)
            elif usage is None:
                usage = u

        # the episode's own evidence trace, shared by both the closer call and the retries below.
        # Computed once, defensively (an empty trace is still valid input to either prompt). Token
        # (not character) caps: see agent_search.tokens's module docstring for why a
        # character cap must never stand in for a token budget in this library.
        try:
            ev = "\n".join(f"[{n}] {cap_tokens(str(a), CLOSER_EVIDENCE_ARG_TOKENS)} -> "
                           f"{cap_tokens(str(o), CLOSER_EVIDENCE_OBS_TOKENS)}"
                           for n, a, o in trace[-25:])
        except Exception:
            ev = ""

        # Forced final answer, for parity with the loop driver's budget nudge (loop.py): without it
        # an SDK-driven episode forfeits every hard question as "(max turns exceeded)" while a
        # loop-driven episode commits a best-effort <answer> on its last turn. One extra no-tools
        # call over the episode's own evidence; a failure here just keeps `final` empty (it never
        # crashes the row). Extracted with the same `_extract_answer` as the retry path below, so
        # `final_answer` never carries a literal `<answer>...</answer>` tag regardless of which
        # forcing call produced it.
        try:
            closer = Agent(name="research-final", instructions=instructions,
                           model=make_agent_model(model, api_base=api_base), tools=[])
            r2 = Runner.run_sync(
                closer,
                f"{question}\n\nSTEP BUDGET REACHED — this is your FINAL turn. Do NOT search again. "
                f"Based ONLY on the evidence below from your earlier tool calls, give your single "
                f"best-effort answer NOW as <answer>...</answer>. If unsure, commit your most likely "
                f"answer — a best guess scores better than an empty answer.\n\n"
                f"=== your earlier tool calls and observations ===\n{ev}",
                max_turns=1)
            raw2 = str(r2.final_output or "")
            cand = _extract_answer(raw2) if raw2.rfind("<answer>") >= 0 else raw2.strip()
            if cand:
                final = cand
            _fold_usage(r2)
        except Exception:
            pass

        # Ask-with-retry fallback (up to 2 strict retries): fires only when the closer call above
        # still left `final` empty. API models (OpenAI/Gemini, driven through this SDK) have no
        # assistant-prefill affordance; that mechanism (`continue_final_message`) is vLLM-specific
        # and lives in `agent_search/agent/forced_answer.py`, used by the loop driver's inline
        # elicitation instead. Here the only lever left is to ask again, more strictly, each a
        # fresh no-tools call over the same evidence. The first non-empty `<answer>` wins.
        if not final.strip():
            for _attempt in range(2):
                try:
                    retry = Agent(name="research-final-retry",
                                  instructions=instructions,
                                  model=make_agent_model(model, api_base=api_base), tools=[])
                    r3 = Runner.run_sync(
                        retry,
                        f"{question}\n\nBased ONLY on the evidence below from your earlier tool "
                        f"calls: Output ONLY <answer>your answer</answer>. If unsure, commit your "
                        f"most likely answer — a best guess scores better than an empty answer.\n\n"
                        f"=== your earlier tool calls and observations ===\n{ev}",
                        max_turns=1)
                    raw3 = str(r3.final_output or "")
                    cand = _extract_answer(raw3) if raw3.rfind("<answer>") >= 0 else raw3.strip()
                    _fold_usage(r3)
                    if cand:
                        final = cand
                        elicitation = "ask_retry_inline"
                        break
                except Exception:
                    continue
            if elicitation is None:
                elicitation = "ask_retry_failed"
        else:
            elicitation = "ask_retry_inline"   # the single closer call above already worked
    cached = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
    reasoning = int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
    # the first model call's input, roughly the fixed initial prompt (system instructions, tool
    # schemas, question) before any tool observation: the cache-aware "count once" initial-prompt
    # figure.
    first_in = 0
    if raws and getattr(raws[0], "usage", None) is not None:
        first_in = int(getattr(raws[0].usage, "input_tokens", 0) or 0)
    fm = _FIX.search(final or "")
    return SdkTrajectory(
        final_answer=final, steps=list(trace),
        input_tokens=int(getattr(usage, "input_tokens", 0)),
        output_tokens=int(getattr(usage, "output_tokens", 0)),
        cached_input_tokens=cached, reasoning_tokens=reasoning, first_input_tokens=first_in,
        requests=int(getattr(usage, "requests", 0)),
        seen=set(getattr(ws, "seen", set())),
        fix_text=fm.group(1).strip() if fm else "",
        elicitation=elicitation,
        stopped_reason=stopped_reason,
    )
