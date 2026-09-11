"""The live demo server: bring-your-own-key, real-time agent runs over the curated corpus.

    pip install -e ".[demo-live]"
    python demo/server.py            # -> http://localhost:8008/

Serves the built demo page and exposes POST /api/run, a Server-Sent-Events stream that drives
one REAL `run_episode` (agent_search/agent/loop.py) per requested strategy, each in its own
worker thread, relaying every Step the moment the loop records it (the `on_step` hook) plus a
cumulative token/cost meter. The request's `api_key` is used to build that run's generate
callable and nothing else: never logged, never stored, never echoed back in the stream.

Per-strategy token accounting works BECAUSE of the thread-per-strategy design:
`agent_search.models`'s usage ledger is thread-local, so `reset_usage()` at worker
start + `usage_totals()` inside on_step reads exactly this episode's cumulative usage (the
loop only attaches per-step tokens after an episode ends, so streaming reads the ledger live)."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import sys
import threading
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

DEFAULT_SNIPPET_TOKENS = 32   # mirrors agent_search.tools.budgets.SNIPPET_TOKENS

# THE PAPER'S READ BUDGET. Both caps are read at IMPORT time by
# agent_search/tools/budgets.py, so they must be set BEFORE that import below.
# The library default is 1200 tokens; the paper runs 12,000. That difference is not cosmetic
# for this demo: it is the baseline's whole-document `visit` budget, so leaving it at 1200
# silently truncated Search-Visit's reads ~10x and made the expensive-baseline contrast, the
# whole point of the comparison, invisible (Sieve appeared to cost more per run).
os.environ.setdefault("MAX_VISIT_TOKENS", "12000")     # whole-doc read ceiling (Search-Visit)
os.environ.setdefault("MAX_SECTION_TOKENS", "12000")   # per-section read ceiling (Sieve)
# Query-biased snippet window. Left at the library default so the demo shows the same listing
# the harness produces; set SNIPPET_TOKENS in the environment to demo a sweep value
# (32 / 64 / 128 / 256 / 512) and the search cards widen accordingly.
os.environ.setdefault("SNIPPET_TOKENS", str(DEFAULT_SNIPPET_TOKENS))

from fastapi import FastAPI                                    # noqa: E402
from fastapi.middleware.cors import CORSMiddleware             # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field                          # noqa: E402

from agent_search.agent.loop import Task, run_episode          # noqa: E402
from agent_search.agent.policies import AgentPolicy            # noqa: E402
from agent_search.core.tokens import count_tokens               # noqa: E402
import agent_search.models as backends                       # noqa: E402
from agent_search.retrievers.engines import Engines             # noqa: E402
from agent_search.strategies.conditions import get_condition   # noqa: E402
from agent_search.tools.base import EpisodeState                # noqa: E402
from demo.corpus import CORPUS, QUESTIONS                      # noqa: E402
from demo.parse import (parse_bm25_search, parse_fetch,        # noqa: E402
                        parse_search, parse_visit)

# Indirection so tests inject a scripted generate (monkeypatch server._make_generate), the
# ONLY seam between this server and a real API call.
_make_generate = backends.make_generate

MAX_STEPS = 20                       # bounds every live run's cost and latency. 20, not 12:
                                     # a multi-hop BrowseComp question needs several
                                     # search->read hops, and the fixed cost of an episode
                                     # (the manual, re-sent each turn) only amortises over a
                                     # run long enough to actually do the hops.
# $ per 1M tokens (OpenAI published rates): (uncached input, cached input, output). The
# frontend's model choice is restricted to these keys so the meter never guesses.
# Cached input matters here and is not a rounding error: OpenAI caches any prompt over 1024
# tokens automatically, and an agent episode re-sends a growing conversation with a stable
# prefix (system manual + question) every turn, so most input tokens after turn 1 are cache
# hits billed at half price. Ignoring that overstates cost, and overstates it unevenly,
# since the two strategies carry very different fixed prefixes (Sieve's BQL manual is ~2K
# tokens larger than the BM25 baseline's, and it is exactly the part that caches).
# OpenAI's published per-1M rates, Standard tier. source (the official guideline the meter
# implements): https://developers.openai.com/api/docs/pricing, every entry below verified
# against that page on 2026-08-08. The billed-cost formula is OpenAI's own:
#
#   cost = (input_tokens - cached_input_tokens)/1e6 * input_rate
#        +  cached_input_tokens/1e6              * cached_input_rate
#        +  output_tokens/1e6                    * output_rate
#
# where cached_input_tokens is what the API itself reports in
# usage.prompt_tokens_details.cached_tokens (see backends._cached_tokens), the meter reads
# the provider's numbers, it does not estimate them. Rates drift: an entry here only feeds
# the meter, and a stale one is still closer than the "n/a" an unlisted model shows.
# Reasoning-family names (gpt-5*, o3, o4-mini) route through the library's reasoning path.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5":        (1.25, 0.125, 10.00),
    "gpt-5-mini":   (0.25, 0.025, 2.00),
    "gpt-5-nano":   (0.05, 0.005, 0.40),
    "gpt-4.1":      (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o":       (2.50, 1.25, 10.00),
    "gpt-4o-mini":  (0.15, 0.075, 0.60),
    "o3":           (2.00, 0.50, 8.00),
    "o4-mini":      (1.10, 0.275, 4.40),
}
# dropdown order: the sensible default first, then cheap -> capable
MODEL_MENU = ["gpt-4o-mini", "gpt-5-nano", "gpt-4.1-nano", "gpt-5-mini", "gpt-4.1-mini",
              "o4-mini", "gpt-4o", "gpt-4.1", "o3", "gpt-5"]

# Neither shipped manual variant matches this corpus exactly: `browsecomp` documents
# title/author/date/body but denies that sections exist (so every named-section fetch fails),
# while `wiki` documents title/section/infobox/body but omits author/date and advertises an
# infobox these documents don't have. We use `wiki` (sections are load-bearing for fetch) and
# correct the field list here rather than editing the library's manuals, which the paper's
# experiments share.
CORPUS_NOTE = """

## This corpus (demo)

Documents here have BOTH named sections AND bibliographic metadata. Usable fields:
`title`, `section`, `body`, `author`, `date`. There is NO `infobox` on this corpus — scoping
to it 0-hits every document; use `author`/`date` for who-wrote-it and when-published facts
(e.g. `2014[date]`, `wilkinson[author]`).

Documents are journal articles, encyclopedia entries and news pages. Near-duplicate titles
occur: when two hits look like the same work, fetch both and prefer the one whose text
matches every constraint in the question.

Search entity NAMES, never the question's wording. Relation words (established, founded,
authored, located) are what you look FOR in a fetched section — never what you search for.
`established between 1949 and 1959 AND notable alumni` searches a sentence and finds nothing;
`1956[body]` or `kamath[title]` finds the document. Start from the 1-2 tokens most likely to
appear VERBATIM in the target document — a proper name, a domain term, an exact number.

Each hop is its own search + fetch, not a longer query. A fetched section names the next
entity; search THAT next. If a query returns "0 exact matches", your surface is wrong, not
the document missing: drop to fewer, more distinctive words rather than rephrasing the
sentence.
"""

STRATEGIES = {
    # name -> the paper condition (a task with a strategy, agent_search/strategies/paper.py).
    # The condition fixes the tool names the model calls: search_s/fetch_s for sieve,
    # bm25_search/visit for search_visit.
    "sieve": "research_snip",
    "search_visit": "research_bm25",
}


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    api_key: str = Field(min_length=1)
    # Free text: users bring their own model (it runs on their key). Safety property kept a
    # different way: the server pins api_base to OpenAI's endpoint when building the generate
    # callable, so an arbitrary model name can never route the key to make_generate's
    # localhost backend="api" fallback. Cost display degrades to tokens-only for models the
    # prices table doesn't know.
    model: str = Field(min_length=1, max_length=100, pattern=r"^[\w][\w\.\:/-]*$")
    strategies: list[Literal["sieve", "search_visit"]] = Field(min_length=1, max_length=2)


app = FastAPI(title="SkimSearchAgent live demo")
# The server now serves its own page (GET /), so the browser's origin is always this server's,
# no more file://-opened page reaching across origins. Pin to the two localhost spellings a
# browser may use to reach this same server, not "*" (which would let ANY page on the web POST a
# visitor's typed-in API key to this endpoint via a background fetch).
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:8008", "http://127.0.0.1:8008"],
                   allow_methods=["*"], allow_headers=["*"])


def _cost(model: str, prompt_tokens: int, completion_tokens: int,
          cached_tokens: int = 0) -> float | None:
    """Billed cost, or None for a model the PRICES table doesn't know (the meter then shows
    tokens only, rather than a confidently wrong dollar figure). `cached_tokens` is the
    SUBSET of `prompt_tokens` that hit the provider's prefix cache, billed at the discounted
    rate; the remainder bills at the full input rate."""
    if model not in PRICES:
        return None
    p_in, p_cached, p_out = PRICES[model]
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached / 1e6 * p_in + cached_tokens / 1e6 * p_cached
            + completion_tokens / 1e6 * p_out)


def _usage_snapshot(model: str) -> dict:
    t = backends.usage_totals()      # thread-local: this worker's episode only
    cached = t.get("cached_input_tokens", 0)
    cost = _cost(model, t["prompt_tokens"], t["completion_tokens"], cached)
    return {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
            "cached_input_tokens": cached, "llm_calls": t["llm_calls"],
            "cost_usd": round(cost, 6) if cost is not None else None}


def _step_payload(step) -> dict:
    """One loop Step -> the card dict the player renders (shared parsers from demo/parse)."""
    name, obs = step.name, step.observation or ""
    if name in ("search", "search_s"):
        return {"type": "search", "query": (step.args or {}).get("query", ""),
                **parse_search(obs)}
    if name in ("fetch", "fetch_s"):
        specs = (step.args or {}).get("specs") or []
        try:
            ask = ", ".join(f"{d} § {s}" for d, s in specs)
        except (TypeError, ValueError):
            ask = str(specs)
        return {"type": "fetch", "ask": ask, **parse_fetch(obs)}
    if name == "bm25_search":
        return {"type": "search", "query": (step.args or {}).get("query", ""),
                **parse_bm25_search(obs)}
    if name == "visit":
        return {"type": "visit", **parse_visit(obs)}
    # submit/answer/stop/budget/none/... -> the frontend's generic fallback renderer. The full
    # observation is sent uncapped: the real read caps live in
    # agent_search/tools/budgets.py's token budgets, not here.
    return {"type": "generic", "name": name, "observation": obs}


def _run_strategy(strategy: str, req: RunRequest, out: queue.Queue) -> None:
    """Worker thread: one full episode, every event pushed to `out`. Always ends with a
    `_finished` sentinel so the SSE generator can never wait forever."""
    try:
        backends.reset_usage()
        cond = get_condition(STRATEGIES[strategy])
        # a fresh in-memory engine per request (no index_root persistence): the demo corpus
        # is ~250 documents, cheap to index on the spot.
        engines = Engines(CORPUS, None, index_root="indexes", domain=cond.domain)
        engine_map = engines.build(cond.strategy.engines)
        ubyid = {u.doc_id: u for u in CORPUS}
        workspace = cond.strategy.toolbox(EpisodeState(question=req.question), CORPUS, ubyid, engine_map)
        # api_base pinned: even a model name make_generate doesn't recognise (custom
        # fine-tunes, new releases) goes to OpenAI's endpoint with the user's key, never
        # to the localhost fallback that api_base would otherwise default to.
        generate = _make_generate(req.model.strip(), backend="api",
                                  api_base=backends._OPENAI_BASE_URL, api_key=req.api_key)
        # "wiki", not "browsecomp", despite this being a BrowseComp-Plus corpus. The profile
        # names a manual variant (an interface shape), not a dataset: the `browsecomp` manual
        # describes the flat build, it states "there are no named sections", tells the agent
        # `[section]` always 0-hits, and demonstrates fetch as `[[1, "body"]]`. Our corpus is
        # the SECTIONED build (doc 51481 has 37 named sections), so that manual made every
        # fetch fail with "no section 'body'". `wiki` (skills/bql_doc.md) is the named-section
        # variant: "fetch a named section ... a name from that doc's list".
        policy = AgentPolicy(generate, system=cond.render("wiki"))
        policy.system += CORPUS_NOTE
        steps_seen = {"n": 0}
        # Document text pulled into context, the axis the two strategies actually differ on (a
        # named section vs a whole document). Token/cost totals alone are confounded when one
        # strategy gives up early and the other keeps hopping. `read_chars` is a character count;
        # kept (never mixed with the token ruler) because the compiled React bundle
        # (demo/app/dist/index.html, which this change cannot rebuild here) reads this exact
        # field name. `read_tokens` is the token-ruler companion (agent_search.core.tokens.
        # count_tokens, tiktoken o200k when available) for anything reading the live event
        # stream directly rather than through the prebuilt page.
        read_chars = {"n": 0}
        read_tokens = {"n": 0}

        def on_step(step) -> None:
            steps_seen["n"] += 1
            payload = _step_payload(step)
            if payload["type"] in ("fetch", "visit") and not payload.get("error"):
                text = payload.get("text") or ""
                read_chars["n"] += len(text)
                read_tokens["n"] += count_tokens(text)
            out.put({"event": "step", "strategy": strategy, "step": payload,
                     "usage": {**_usage_snapshot(req.model), "steps": steps_seen["n"],
                               "read_chars": read_chars["n"], "read_tokens": read_tokens["n"]}})

        traj = run_episode(policy, Task("live", req.question), workspace, CORPUS,
                           max_steps=MAX_STEPS, usage_fn=backends.usage_events,
                           domain="general", on_step=on_step)
        out.put({"event": "done", "strategy": strategy,
                 "answer": traj.final_answer, "stopped": traj.stopped_reason,
                 "usage": {**_usage_snapshot(req.model), "steps": steps_seen["n"],
                           "read_chars": read_chars["n"], "read_tokens": read_tokens["n"]}})
    except Exception as e:  # noqa: BLE001, any failure becomes an error event, never a hang
        # provider auth errors quote a (masked) copy of the offending key, redact any
        # key-shaped token so the "never echoed back" invariant holds on error paths too.
        message = re.sub(r"sk-[\w*-]+", "sk-***", f"{type(e).__name__}: {e}")[:300]
        out.put({"event": "error", "strategy": strategy, "message": message})
    finally:
        out.put({"event": "_finished", "strategy": strategy})


@app.post("/api/run")
async def run(req: RunRequest) -> StreamingResponse:
    strategies = list(dict.fromkeys(req.strategies))          # dedupe, keep order
    out: queue.Queue = queue.Queue()
    threads = [threading.Thread(target=_run_strategy, args=(s, req, out), daemon=True)
               for s in strategies]

    async def stream():
        loop = asyncio.get_running_loop()
        for t in threads:
            t.start()
        finished = 0
        while finished < len(threads):
            msg = await loop.run_in_executor(None, out.get)
            if msg["event"] == "_finished":
                finished += 1
                continue
            yield f"event: {msg['event']}\ndata: {json.dumps(msg)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/meta")
def meta() -> dict:
    """Everything the page needs before a run: the curated example questions, the collection
    shelf (id/title/section-count per doc), and the run parameters the UI displays."""
    return {"questions": [{"question": q, "gold": gold, "label": label, "paper": paper}
                          for q, gold, label, paper in QUESTIONS],
            "corpus": [{"id": u.doc_id, "title": u.title,
                        "sections": len(u.sections or ())} for u in CORPUS],
            "model": "gpt-4o-mini", "priced_models": MODEL_MENU,
            "pricing_url": "https://developers.openai.com/api/docs/pricing",
            "max_steps": MAX_STEPS}


_DIST = HERE / "app" / "dist" / "index.html"
_STATIC = HERE / "index.html"


@app.get("/")
def root() -> FileResponse:
    return FileResponse(_DIST if _DIST.exists() else _STATIC)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)
