"""OpenAI Agents SDK driver for the doc-research arm — Slice 1 of docs/agents_sdk_migration.md.

An ALTERNATIVE to agent/loop.py's text-parsed ReAct loop: the workspace's tools become native SDK
`@function_tool`s and `Runner` drives the loop with structured tool-calling (no `<tool_call>` text
scraping). The SAME workspace (`DocSearchFetch`) is reused verbatim — only the driver + tool surface
change. `run_episode_sdk` returns a plain `SdkTrajectory` (answer + steps + token usage) so it can be
scored exactly like the loop's `Trajectory`.

Slice 1 targets `gpt-4o-mini` (native function-calling, no GPU). vLLM/Tongyi + Gemini plug in later
via a `make_agent_model` (see the migration plan); the KEY open question there is whether a
vLLM-served model emits well-formed tool calls — validate before committing.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# code arm: the terminal is a <fix>…</fix> block in the model's final message (same as the loop);
# native tools handle search/fetch, and we parse the fix from the final output.
_FIX = re.compile(r"<fix>(.*?)</fix>", re.DOTALL | re.IGNORECASE)

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

from agent_search.agent.loop import _extract_answer
from agent_search.models.backends import (
    _GEMINI_BASE_URL, is_gemini_model, is_openai_model)

# Returned as `final_output` (well, set as `final`) when the SDK's own `max_turns` is exhausted,
# before either forcing mechanism below has had a chance to fill in a real answer.
_MAX_TURNS_PLACEHOLDER = "(max turns exceeded — no final answer)"

# Don't ship benchmark traces to OpenAI's hosted tracing platform (privacy + it 400s on non-OpenAI
# usage payloads, e.g. Gemini). We keep our own step trace in SdkTrajectory instead.
set_tracing_disabled(True)


def make_agent_model(model: str, *, api_base: Optional[str] = None):
    """Route a model name to what `Agent(model=…)` needs — the SDK analogue of
    `backends.make_generate`, so ALL providers go through ONE SDK path (no separate chat-completions
    calls). Reuses the same `is_openai_model`/`is_gemini_model` matchers.

    - OpenAI (`gpt-*`/`o-*`): the bare model string → the SDK's default OpenAI client.
    - Gemini (`gemini*`): an `OpenAIChatCompletionsModel` on Gemini's OpenAI-compatible endpoint (the
      SAME endpoint `backends.gemini_generate` uses) — so Gemini is driven by the SDK's tool-calling,
      not a bespoke call.
    - else (a served vLLM / Tongyi backbone): an `OpenAIChatCompletionsModel` at `api_base` (the
      OpenAI-compatible `vllm serve` endpoint). NOTE: in-process vLLM has no SDK path — must be served.
    """
    if is_openai_model(model):
        return model
    if is_gemini_model(model):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=_GEMINI_BASE_URL,
                             api_key=os.environ.get("GEMINI_API_KEY", ""))
        return OpenAIChatCompletionsModel(model=model, openai_client=client)
    from openai import AsyncOpenAI
    base = api_base or os.environ.get("VLLM_API_BASE") or "http://localhost:8000/v1"
    client = AsyncOpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    return OpenAIChatCompletionsModel(model=model, openai_client=client)


@dataclass
class SdkTrajectory:
    final_answer: str
    steps: list            # [(tool_name, args_dict, observation_str), ...] in call order
    input_tokens: int      # TOTAL input across all model calls (re-sends the growing context)
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int   # of output_tokens, the reasoning subset (0 for non-reasoning models)
    first_input_tokens: int  # the FIRST model call's input ≈ the fixed initial prompt (counted once)
    requests: int
    seen: set              # doc_ids the workspace surfaced (gold-doc-coverage metric)
    fix_text: str = ""     # code arm: the <fix> block parsed from the final output (else "")
    # provenance of a forced (non-organic) final_answer, mirroring agent_search.agent.loop.
    # Trajectory.elicitation: None (max_turns never hit — final_output is organic), "ask_retry_inline"
    # (the MaxTurnsExceeded forcing sequence below — the closer call and/or its strict retries —
    # produced the answer), "ask_retry_failed" (that whole sequence exhausted its attempts). API
    # models have no assistant-prefill affordance (that's the loop driver's vLLM-only mechanism,
    # agent_search/agent/forced_answer.py), so this driver's forcing mechanism is ask-and-retry,
    # not prefill — see run_episode_sdk's MaxTurnsExceeded branch for the detail.
    elicitation: Optional[str] = None


# per-arm coaching. The tool SET a workspace exposes (ws.tools) selects the instructions, so the
# SAME driver runs the BQL method, the bm25 baseline, and the DCI shell baseline.
_INSTR_BQL = (
    "Use `search` with a FIELD-TAGGED BOOLEAN query (term[field], AND/OR/NOT, (), \"phrase\", "
    "wildcard*; fields: title/body/section/infobox/author/date, e.g. `university[title] AND "
    "cultural[body]`) to find candidate docs and see their section headings. Then `fetch` ONLY the "
    "section(s) that answer the question (by the doc's rank or id + a heading) — fetch narrowly, "
    "never whole docs.")
_INSTR_BM25 = (
    "Use `bm25_search` with plain keywords to find candidate documents (it returns ranked docs + a "
    "short snippet), then `visit` the most relevant document to read its full text.")
_INSTR_BM25Q = (
    "Use `bm25q_search` with plain keywords to find candidate documents (it returns ranked docs, "
    "each with its best-matching excerpt for your query, not just the opening), then `visit_q` "
    "the most relevant document to read its full text.")
_INSTR_HYBRID = (
    "Use `hybrid_search` with plain keywords or a natural-language query to find candidate "
    "documents (it fuses a keyword-search ranking and a semantic-embedding ranking via "
    "Reciprocal Rank Fusion, returning ranked docs + a short snippet), then `visit_h` the most "
    "relevant document to read its full text.")
_INSTR_HYBRID_FETCH = (
    "Use `hybrid_search_snip` with plain keywords or a natural-language query to find candidate "
    "documents (it fuses a keyword-search ranking and a semantic-embedding ranking via "
    "Reciprocal Rank Fusion; each hit shows its section names + infobox keys + a best-matching "
    "excerpt), then `fetch` ONLY the section(s) that answer the question (by the doc's rank or "
    "id + a heading) — fetch narrowly, never whole docs.")
_INSTR_INDRI_VISIT = (
    "Use `isearch_v` with an Indri structured query (#combine/#weight/#band, #odN/#uwN windows, "
    "#syn, term.field, #filreq/#filrej, #date:between; graded matching — never returns zero hits) "
    "to find candidate documents (each hit includes a best-matching excerpt), then `visit_v` the "
    "most relevant document (by rank or doc_id) to read its full text.")
_INSTR_BQL_VISIT = (
    "Use `search_bv` with a FIELD-TAGGED BOOLEAN query (term[field], AND/OR/NOT, (), \"phrase\", "
    "wildcard*, typed date[RANGE] ranges; fields: title/body/section/infobox/author/date, e.g. "
    "`university[title] AND cultural[body] AND date[1980..1989]`) to find candidate documents "
    "(each hit includes a best-matching excerpt), then `visit_bv` the most relevant document (by "
    "rank or doc_id) to read its full text.")
_INSTR_BQL_DENSE_VISIT = (
    "Use `search_bqld` with a FIELD-TAGGED BOOLEAN query (term[field], AND/OR/NOT, (), \"phrase\", "
    "wildcard*, typed date[RANGE] ranges; fields: title/body/section/infobox/author/date, e.g. "
    "`university[title] AND cultural[body] AND date[1980..1989]`) to find candidate documents "
    "(each hit includes a best-matching excerpt), then `visit_bqld` the most relevant document (by "
    "rank or doc_id) to read its full text.")
_INSTR_DCI = (
    "You have a SHELL over a flat file tree: one `<doc_id>.txt` per document, whose text contains "
    "`## Section` headings. Use `bash` (e.g. `grep -rl PATTERN .`, `grep -n \"## Heading\" file`) to "
    "find relevant files and section line numbers, then `read(path, offset, limit)` to read just "
    "those lines.")
_PREAMBLE = "You are a research agent answering a question from a document corpus.\n"
_CLOSER = "\nWhen you have the answer, state it concisely and name the supporting doc id(s)."


def _instructions_for(ws) -> str:
    names = set(getattr(ws, "tools", ()))
    body = (_INSTR_BQL if ("fetch" in names and "search" in names)
                        or ("fetch_v2" in names and "search_v2" in names)
                        or ("fetch_s" in names and "search_s" in names)
                        or "isearch" in names           # Indri graded-QL: same fetch-narrowly coaching
                        or "isearch_s" in names          # research_indri_snip: isearch + snippet listing, same fetch-narrowly coaching
            else _INSTR_INDRI_VISIT if ("isearch_v" in names and "visit_v" in names)
            else _INSTR_BQL_VISIT if ("search_bv" in names and "visit_bv" in names)
            else _INSTR_BQL_DENSE_VISIT if ("search_bqld" in names and "visit_bqld" in names)
            else _INSTR_DCI if "bash" in names
            else _INSTR_BM25Q if "bm25q_search" in names
            else _INSTR_HYBRID_FETCH if "hybrid_search_snip" in names
            else _INSTR_HYBRID if "hybrid_search" in names
            else _INSTR_BM25)
    return _PREAMBLE + body + _CLOSER


def _tools_for(ws):
    """Build native SDK function tools for whatever `ws.tools` exposes — search/fetch (BQL),
    search_v2/fetch_v2 (BQL v2, research_v2), search_s/fetch_s (excerpt listing, research_snip),
    isearch/fetch (Indri, research_indri), isearch_v/visit_v (Indri + snippet listing + whole-doc
    read, research_indri_visit), isearch_s/fetch (Indri + snippet listing + section fetch,
    research_indri_snip), bm25_search/visit (bm25 baseline), bm25q_search/visit_q (the
    HARDENED bm25 baseline — SAME retrieval/read as bm25_search/visit, query-biased listing
    snippet, research_bm25q), search_bv/visit_bv (BQL v2 search + snippet listing + whole-doc
    read, research_bql_visit — the {BQL search} x {whole-doc visit} factorial cell),
    bm25_search_snip/fetch (bm25 retrieval + snippet listing + section fetch, the fair-listing
    sibling of bm25_search/fetch, research_bm25_fetch_snip), hybrid_search/visit_h (BM25+dense
    RRF-fused retrieval + whole-doc read, the hybrid baseline, research_hybrid),
    hybrid_search_snip/fetch (SAME RRF-fused retrieval + snippet listing + section fetch,
    research_hybrid_fetch_snip), dense_search_fp/fetch (SAME dense retrieval as dense_search_f,
    PLAIN listing with no per-hit excerpt + section fetch, research_dense_fetch_plain), bash/read
    (DCI shell). Each delegates to `ws.run(name, args)`
    verbatim (reusing the workspace's arg-normalization) and appends to a shared `trace`."""
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
                    max_turns: int = 12, instructions: Optional[str] = None) -> SdkTrajectory:
    """Drive one episode over `ws` via the Agents SDK. Tools + coaching are chosen from `ws.tools`,
    so the SAME driver runs the BQL method and the bm25/DCI baselines. `model` routes through
    `make_agent_model` (OpenAI / Gemini / served-vLLM). Hitting `max_turns` returns whatever was
    produced (partial answer + the tool trace), not a crash — so a hard question can't abort a run."""
    from agents.exceptions import MaxTurnsExceeded
    tools, trace = _tools_for(ws)
    agent = Agent(name="research", instructions=instructions or _instructions_for(ws),
                  model=make_agent_model(model, api_base=api_base), tools=tools)
    final, usage, raws = "", None, []
    elicitation: Optional[str] = None
    try:
        result = Runner.run_sync(agent, question, max_turns=max_turns)
        final = str(result.final_output or "")
        usage = result.context_wrapper.usage
        raws = getattr(result, "raw_responses", None) or []
    except MaxTurnsExceeded as e:
        final = _MAX_TURNS_PLACEHOLDER
        # the accumulated usage still lives on the exception's run_data (don't report 0 tokens for
        # an episode that actually made several model calls before hitting the cap).
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

        # the episode's own evidence trace, shared by BOTH the closer call and the retries below —
        # computed once, defensively (an empty trace is still valid input to either prompt).
        try:
            ev = "\n".join(f"[{n}] {str(a)[:120]} -> {str(o)[:700]}" for n, a, o in trace[-25:])
        except Exception:
            ev = ""

        # FORCED FINAL ANSWER — parity with the loop driver's budget nudge (loop.py): without it the
        # SDK arm forfeits every hard question as "(max turns exceeded)" while the loop arm commits a
        # best-effort <answer> on its last turn. One extra NO-TOOLS call over the episode's own
        # evidence; a failure here just keeps the placeholder (never crashes the row).
        try:
            closer = Agent(name="research-final", instructions=instructions or _instructions_for(ws),
                           model=make_agent_model(model, api_base=api_base), tools=[])
            r2 = Runner.run_sync(
                closer,
                f"{question}\n\nSTEP BUDGET REACHED — this is your FINAL turn. Do NOT search again. "
                f"Based ONLY on the evidence below from your earlier tool calls, give your single "
                f"best-effort answer NOW as <answer>...</answer>. If unsure, commit your most likely "
                f"answer — a best guess scores better than an empty answer.\n\n"
                f"=== your earlier tool calls and observations ===\n{ev}",
                max_turns=1)
            forced = str(r2.final_output or "").strip()
            if forced:
                final = forced
            _fold_usage(r2)
        except Exception:
            pass

        # ASK-WITH-RETRY FALLBACK (up to 2 strict retries) — PURE ADDITION, fires ONLY when the
        # closer call above still left `final` empty/at the placeholder. API models (OpenAI/Gemini,
        # driven through this SDK) have no assistant-prefill affordance — that mechanism
        # (`continue_final_message`) is vLLM-specific and lives in
        # `agent_search/agent/forced_answer.py`, used by the LOOP driver's inline elicitation
        # instead. Here the only lever left is to ask again, more strictly, each a fresh no-tools
        # call over the same evidence; the FIRST non-empty `<answer>` wins.
        if not final.strip() or final == _MAX_TURNS_PLACEHOLDER:
            for _attempt in range(2):
                try:
                    retry = Agent(name="research-final-retry",
                                  instructions=instructions or _instructions_for(ws),
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
            elicitation = "ask_retry_inline"   # the pre-existing single closer call already worked
    cached = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
    reasoning = int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
    # the FIRST model call's input ≈ the fixed initial prompt (system instructions + tool schemas +
    # question), before any tool observation — the cache-aware "count once" initial-prompt figure.
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
    )
