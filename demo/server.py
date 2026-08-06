"""The live demo server: bring-your-own-key, real-time agent runs over the curated corpus.

    pip install -e ".[demo-live]"
    python demo/server.py            # -> http://localhost:8008/

Serves the built demo page and exposes POST /api/run — a Server-Sent-Events stream that drives
one REAL `run_episode` (agent_search/agent/loop.py) per requested strategy, each in its own
worker thread, relaying every Step the moment the loop records it (the `on_step` hook) plus a
cumulative token/cost meter. The request's `api_key` is used to build that run's generate
callable and nothing else: never logged, never stored, never echoed back in the stream.

Per-strategy token accounting works BECAUSE of the thread-per-strategy design:
`agent_search.models.backends`'s usage ledger is thread-local, so `reset_usage()` at worker
start + `usage_totals()` inside on_step reads exactly this episode's cumulative usage (the
loop only attaches per-step tokens AFTER an episode ends, so streaming reads the ledger live)."""
from __future__ import annotations

import asyncio
import json
import queue
import re
import sys
import threading
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fastapi import FastAPI                                    # noqa: E402
from fastapi.middleware.cors import CORSMiddleware             # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field                          # noqa: E402

from agent_search.agent.loop import Task, run_episode          # noqa: E402
from agent_search.agent.policies import AgentPolicy            # noqa: E402
from agent_search.agent.tools.doc_research import Bm25Visit, DocSearchFetch  # noqa: E402
from agent_search.models import backends                       # noqa: E402
from agent_search.prompts import get_prompt_spec               # noqa: E402
from demo.corpus import CORPUS, QUESTIONS                      # noqa: E402
from demo.parse import (parse_bm25_search, parse_fetch,        # noqa: E402
                        parse_search, parse_visit)

# Indirection so tests inject a scripted generate (monkeypatch server._make_generate) — the
# ONLY seam between this server and a real API call.
_make_generate = backends.make_generate

MAX_STEPS = 12                       # bounds every live run's cost and latency
# $ per 1M prompt/completion tokens (OpenAI published rates). The frontend's model choice is
# restricted to these keys so the meter never guesses.
PRICES: dict[str, tuple[float, float]] = {"gpt-4o-mini": (0.15, 0.60)}

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
    # name -> (workspace builder, prompt condition). The condition fixes the tool NAMES the
    # model calls (search_s/fetch_s vs bm25_search/visit) — see conditions.yaml.
    "sieve": (lambda: DocSearchFetch(CORPUS, snippets=True), "research_snip"),
    "search_visit": (lambda: Bm25Visit(CORPUS), "research_bm25"),
}


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    api_key: str = Field(min_length=1)
    # restricted to PRICES' keys so the cost meter never guesses AND an arbitrary model string
    # can never route the user's key to the backend="api" localhost fallback.
    model: Literal["gpt-4o-mini"] = "gpt-4o-mini"
    strategies: list[Literal["sieve", "search_visit"]] = Field(min_length=1, max_length=2)


app = FastAPI(title="SkimSearchAgent live demo")
# a file://-opened demo page must still reach a locally-running server.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return prompt_tokens / 1e6 * p_in + completion_tokens / 1e6 * p_out


def _usage_snapshot(model: str) -> dict:
    t = backends.usage_totals()      # thread-local: this worker's episode only
    return {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
            "llm_calls": t["llm_calls"],
            "cost_usd": round(_cost(model, t["prompt_tokens"], t["completion_tokens"]), 6)}


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
    # submit/answer/stop/budget/none/... -> the frontend's generic fallback renderer
    return {"type": "generic", "name": name, "observation": obs[:2000]}


def _run_strategy(strategy: str, req: RunRequest, out: queue.Queue) -> None:
    """Worker thread: one full episode, every event pushed to `out`. Always ends with a
    `_finished` sentinel so the SSE generator can never wait forever."""
    try:
        backends.reset_usage()
        build_ws, condition = STRATEGIES[strategy]
        workspace = build_ws()
        generate = _make_generate(req.model, backend="api", api_key=req.api_key)
        # "wiki", NOT "browsecomp", despite this being a BrowseComp-Plus corpus. The profile
        # names a MANUAL VARIANT (an interface shape), not a dataset: the `browsecomp` manual
        # describes the FLAT build — it states "there are no named sections", tells the agent
        # `[section]` always 0-hits, and demonstrates fetch as `[[1, "body"]]`. Our corpus is
        # the SECTIONED build (doc 51481 has 37 named sections), so that manual made every
        # fetch fail with "no section 'body'". `wiki` (skills/bql_doc.md) is the named-section
        # variant: "fetch a named section ... a name from that doc's list".
        policy = AgentPolicy(generate, prompt_path=get_prompt_spec(condition).path,
                             field_profile="wiki")
        policy.system += CORPUS_NOTE
        steps_seen = {"n": 0}

        def on_step(step) -> None:
            steps_seen["n"] += 1
            out.put({"event": "step", "strategy": strategy,
                     "step": _step_payload(step),
                     "usage": {**_usage_snapshot(req.model), "steps": steps_seen["n"]}})

        traj = run_episode(policy, Task("live", req.question), workspace, CORPUS,
                           max_steps=MAX_STEPS, usage_fn=backends.usage_events,
                           domain="general", on_step=on_step)
        out.put({"event": "done", "strategy": strategy,
                 "answer": traj.final_answer, "stopped": traj.stopped_reason,
                 "usage": {**_usage_snapshot(req.model), "steps": steps_seen["n"]}})
    except Exception as e:  # noqa: BLE001 — any failure becomes an error event, never a hang
        # provider auth errors quote a (masked) copy of the offending key — redact any
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
    return {"questions": [{"question": q, "gold": gold, "label": label}
                          for q, gold, label in QUESTIONS],
            "corpus": [{"id": u.doc_id, "title": u.title,
                        "sections": len(u.sections or ())} for u in CORPUS],
            "model": "gpt-4o-mini", "max_steps": MAX_STEPS}


_DIST = HERE / "app" / "dist" / "index.html"
_STATIC = HERE / "index.html"


@app.get("/")
def root() -> FileResponse:
    return FileResponse(_DIST if _DIST.exists() else _STATIC)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)
