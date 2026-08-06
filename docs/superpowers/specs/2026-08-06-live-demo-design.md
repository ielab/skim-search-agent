# Live demo: bring-your-own-key, real-time agent runs

Status: approved for planning. Author: session design doc (Claude), owner: Shuai Wang.

## 1. Goal

The existing `demo/index.html` only replays 3 fixed, pre-recorded episodes — there is no way for
a visitor to ask their own question and watch the agent actually search. This adds a **Live**
mode: each user runs a small local server with their own OpenAI-compatible API key, types any
question (or picks a curated example), picks one or two strategies, and watches the real
`agent_search` library run in real time — same visual language as the existing replay (typewriter
agent lines, result cards, evidence panel), plus a live cost/token meter and an optional
side-by-side strategy comparison.

Out of scope for this pass (explicitly deferred, not because they're hard forever, just to keep
the first version shippable):
- Dense/embedding-based strategies (`DenseVisit`, `DenseAutoRead`) — need a local embedding model
  download, adds friction to "bring your own key and go."
- Indri and DCI (`bash`/`read`) strategies — different observation shapes (graded belief scores,
  shell output) that need their own renderers; can follow once the two MVP strategies are proven.
- Public hosting / server-side key custody, rate limiting — each user runs their own local server
  with their own key (per the earlier scoping conversation), so none of that applies here.
- Any change to the paper's evaluation harness (`evaluation/run_eval.py`) — this is a demo-only
  addition layered on top of the existing library, not a new eval condition.

## 2. Corpus

**Source: two datasets, cross-referenced by `docid`** (revised after the user asked specifically
for the published mirror instead of running a fresh sectioning pass):

- `wshuai190/browsecomp-plus-structured-full` (the paper's own published structured corpus) for
  the **document content** — `structured/corpus.jsonl`'s `{_id, title, author, date, sections:
  [{heading, text}], text}` records are the REAL, already-sectioned output of the paper's
  `gpt-5.4-nano` batch pass, so using this means the demo corpus is byte-faithful to what the
  paper's own experiments ran over — no new sectioning pass needed at all. Caveat found this
  session: this published repo, as it exists on the hub today, contains only
  `flat/corpus.jsonl` (3GB), `structured/corpus.jsonl` (6.1GB), and `sections.jsonl` (2GB) — no
  `queries.jsonl`/`qrels/` (confirmed via the HF tree API), despite `corpus_build/README.md`
  describing that fuller layout. So it supplies documents but not query/gold labels.
- `Tevatron/browsecomp-plus` (the raw, obfuscated source dataset feeding the builder above) for
  **query text + gold-doc labels** — `corpus_build/browsecomp_plus/build.py::to_record_sectioned`
  emits the structured corpus's `_id` as `str(docid)`, literally the same `docid` the raw dataset
  carries, so a query's `gold_docs`/`negative_docs`/`evidence_docs` ids (pulled from this raw
  dataset, decrypted) are valid lookup keys straight into `structured/corpus.jsonl` — confirmed
  by reading the exact line in `build.py` that sets `_id`.

Curation therefore has two steps: (1) stream `Tevatron/browsecomp-plus` (fast — confirmed working
in this session, no need to materialize its 2.8GB split) for the two chosen queries' full
gold+negative+evidence docid set — 102 unique ids for the pair below; (2) stream-filter
`structured/corpus.jsonl` over HTTP for just those 102 `_id`s, stopping as soon as all are found,
rather than downloading the full 6.1GB file (confirmed working in this session via chunked
`requests.get(..., stream=True)` + newline-buffered JSON parsing, no full download needed).

**Selected queries** (real, verified against the live dataset):
- `query_id=798` — "Which institute is currently located within 2 to 7 miles arial distance from
  the location at which it was established? ..." → **Lady Shri Ram College for Women** (3 gold
  docs, 41 negatives, 9 evidence docs).
- `query_id=792` — "A journal article was published between 2010 and 2016 ... about a
  supplement-based approach in the context of a disease ..." → **"Oral creatine supplementation:
  A potential adjunct therapy for rheumatoid arthritis patients"** (1 gold doc, 47 negatives, 5
  evidence docs).

Both are genuinely hard multi-hop BrowseComp-style questions unanswerable from an LLM's memory
alone, and both are benign/safe topics (a college's history, a nutrition journal article) —
verified by inspecting decrypted gold-doc text during this session.

**Curation run completed this session** (not just planned — actually executed against the live
datasets): the 102-id stream-filter over `structured/corpus.jsonl` ran to completion (3071s over a
slow/lossy VPN connection, stopped early once all 102 were found rather than reading the full
6.1GB), and all 4 known gold docids (`37133`/`39666`/`41817` for query 798, `51481` for query 792)
confirmed present in the result.

**Size, measured directly (not estimated):** 102 docs, assembled as `## heading` + section text
exactly like `body` will be built in step 4 below, total **3.50MB**, median doc ~15KB, max ~176KB.
That number alone answers the "small enough for GitHub" requirement — GitHub's soft/hard
thresholds are 50MB/100MB per file, so 3.5MB total needs **no truncation at all** to be
comfortably git-friendly. I initially planned an aggressive ~6-8K-char per-doc cap for this
reason and it was wrong on both counts: unnecessary (the real total is already 15-30x under any
GitHub concern) and actively harmful — doc `51481` (the creatine/rheumatoid-arthritis gold doc for
query 792) has a single section, "Conclusions and recommendations," that is **37,225 chars** by
itself; a 6-8K cap would have silently truncated the actual gold answer out of the corpus. Caught
by checking gold-doc section sizes directly before finalizing this doc, not left as an assumption.

Decision: **no cap driven by repo-size concerns.** A much higher, purely UX/cost-motivated cap
(e.g. ~80K chars, only affecting the ~2-3 most extreme outliers among the 30/102 docs over 50K
chars) MAY still be applied in step 4 below — using the existing `_cap_tokens`-style truncation
marker pattern `doc_research.py` already uses elsewhere — solely so `Bm25Visit`'s whole-document
"visit" doesn't return an ungainly wall of text in the live UI or blow an unnecessary chunk of a
live run's token budget. This is optional polish, not a hard requirement, and must never be set
below the largest known gold section (37,225 chars) confirmed above.

**Curation script** (`demo/recorder/build_corpus.py`, new, one-time/offline — **no OpenAI key
needed**, since we're reusing the paper's already-sectioned published output instead of running a
fresh sectioning pass):

1. Stream `Tevatron/browsecomp-plus` (`load_dataset(..., streaming=True)`) and pull rows `798` and
   `792`; decrypt via `corpus_build.browsecomp_plus.build.transform_decrypt` (reused, not
   reimplemented).
2. Union `gold_docs ∪ negative_docs ∪ evidence_docs` docids across both rows, dedup → 102 unique
   target ids (confirmed this session).
3. Stream-filter `wshuai190/browsecomp-plus-structured-full`'s `structured/corpus.jsonl` over
   HTTP for those 102 `_id`s (chunked read, stop once all are found — confirmed this session,
   avoids the 6.1GB full download).
4. Each matched record already has `title`/`author`/`date`/`sections`/`text` — no frontmatter
   parsing or sectioning needed; map straight to `CodeUnit(doc_id=_id, title=title,
   body=text_with_##_headings_from_sections, metadata={"author":author,"date":date}, ...)`, the
   same shape `demo/recorder/corpus.py` already hand-builds today (`## heading` markers inside
   `body`, matching how `DocSearchFetch`/`Bm25Visit` expect section boundaries). Build `body` from
   `sections` (not the separate, redundant `text` field — same content, already split). No
   size-driven truncation (measured total is 3.5MB across all 102 docs — see above); optionally
   cap only the handful of true outlier docs (>~80K chars) using the existing `_cap_tokens`
   truncation-marker pattern, never below 40K chars so no known gold section is ever at risk.
5. Emit a static `demo/recorder/browsecomp_corpus.py` (mirrors today's `corpus.py` shape:
   `CORPUS: list[CodeUnit]`, `QUESTIONS: list[(question, gold_answer)]` with the two curated
   queries) — checked into the repo, so **no visitor or live-server run ever needs internet access
   to Hugging Face**; only the one-time curation step does.

**Existing 3 replay episodes:** re-recorded against the new corpus (per the earlier decision to
keep one consistent collection), using `demo/recorder/record.py` unchanged except pointing at
`browsecomp_corpus.CORPUS`/`QUESTIONS` instead of the current hand-written 8-doc `corpus.py`. The
old `corpus.py` can be deleted once the new one is in place — nothing else imports it.

## 3. Core library change (one additive parameter)

`agent_search/agent/loop.py::run_episode` gains one optional parameter:

```python
def run_episode(policy, task, workspace, units, max_steps=50, usage_fn=None,
                domain="code", fix_guard=None,
                on_step: Optional[Callable[["Step"], None]] = None) -> Trajectory:
```

`on_step`, when given, is called with `steps[-1]` immediately after each of the existing
`steps.append(...)` call sites (five: the tool-call step, the `fix`/`fix_rejected` steps, the
terminal `submit`/`answer`/`stop` step, and the `budget` nudge step) — no new branches, no
reordering, purely an additional call after work that already happens. Every existing caller
(`evaluation/run_eval.py`, `demo/recorder/record.py`'s pattern, tests) passes nothing, so
`on_step` stays `None` and behavior is byte-identical to today.

`agent_search/models/backends.py::make_generate` gains one optional parameter, forwarded to
whichever sub-builder it dispatches to:

```python
def make_generate(model=DEFAULT_MODEL, *, backend="vllm", api_base=..., tp=1,
                  temperature=0.6, seed=42, client=None, api_key: str | None = None):
    ...
    return openai_compat_generate(model=model, base_url=_OPENAI_BASE_URL,
                                  api_key=api_key, temperature=temperature, seed=seed, client=client)
    # (same api_key= passthrough on the other two branches that call openai_compat_generate /
    # openai_reasoning_generate / gemini_generate, which already accept api_key today)
```

Today `make_generate` never forwards a key — `openai_compat_generate` already accepts `api_key`
but `make_generate` doesn't expose it, so every caller relies on the `OPENAI_API_KEY` process env
var. That's wrong for a server handling concurrent requests from different users with different
keys (env vars are process-global, not per-request). This is the one other required core change;
everything else is additive and backward compatible (default `None` preserves today's env-var
fallback).

## 4. Backend: `demo/live/server.py` (new)

A small FastAPI app (new dependency — see §7). Two responsibilities:

**a. Serve the built frontend.** `StaticFiles` mounts `demo/app/dist/` (or falls back to serving
`demo/index.html` directly) at `/`, so `python demo/live/server.py` is the one command that gets a
visitor a working page at `http://localhost:8008/` — no separate npm dev server, no CORS
juggling. Opening `demo/index.html` directly via `file://` still works too (pure replay,
unchanged); the Live tab there will fail its `fetch`/`EventSource` calls gracefully (see §6
error handling) with instructions to run the server, since permissive CORS (`allow_origins=["*"]`)
is enabled specifically so a `file://`-opened page can still reach a locally-running server.

**b. `POST /api/run` — SSE stream.** Request body:

```json
{
  "question": "free text, or one of the 2 curated examples",
  "api_key": "sk-...",
  "model": "gpt-4o-mini",
  "strategies": ["sieve"] ,           // 1 or 2 of: "sieve" | "search_visit"
}
```

Handling, per requested strategy (run concurrently in Compare mode — two independent episodes,
not one shared one):

1. Build the workspace over the curated `CORPUS` units: `DocSearchFetch(units, snippets=True)`
   for `"sieve"`, `Bm25Visit(units)` for `"search_visit"` — both construct directly from an
   in-memory unit list (confirmed: no index-build step, no external service).
2. `generate = make_generate(model=request.model, backend="api", api_key=request.api_key)`.
3. `policy = AgentPolicy(generate, prompt_path=<research task prompt>, field_profile="browsecomp")`.
4. Run `run_episode(policy, task, workspace, units, max_steps=12, domain="general",
   usage_fn=agent_search.models.backends.usage_events, on_step=q.put)` in a worker thread
   (`concurrent.futures.ThreadPoolExecutor` — `run_episode` is synchronous/blocking, so it cannot
   run directly on FastAPI's event loop). `q` is a plain `queue.Queue()`.
5. An async generator drains `q` (via `loop.run_in_executor(None, q.get)`) and yields one SSE
   `event: step` message per `Step` the moment `on_step` fires — i.e. genuinely real-time, not
   buffered until the episode ends. On thread completion, yields one final `event: done` message
   with the `Trajectory`'s totals.
6. **Step → render shape.** Reuses the existing regex parsers from `demo/recorder/build_web.py`
   (`parse_search`, `parse_fetch` — extracted into an importable `demo/recorder/parse.py` shared
   by both the offline `build_web.py` and this live server, not duplicated) for `"search"`/
   `"fetch"` steps. Adds one new small parser, `parse_visit`, for `Bm25Visit`'s `visit` observation
   format (`"{doc_id}  '{title}':\n{text}"` — confirmed simple, single split). Any step name that
   isn't `search`/`fetch`/`visit` (e.g. `submit`, `answer`, `budget`, malformed-tool-call) is sent
   as a generic `{name, observation}` pair for the frontend's fallback renderer (§5).
7. **Cost.** Each `step` SSE message includes `prompt_tokens`/`completion_tokens` (already on
   `Step`) and a `cost_usd` computed from a small hardcoded per-model price table (starting with
   `gpt-4o-mini`: $0.15/$0.60 per 1M prompt/completion tokens, OpenAI's published rate) —
   cumulative, so the frontend just displays the running total it's told rather than recomputing.

The user's `api_key` is used to build one `openai.OpenAI` client for that single request and
discarded afterward — never logged (the server's own logging must redact the `api_key` field),
never written to disk, never sent anywhere but straight to `https://api.openai.com` via the
existing `openai_compat_generate` call path.

## 5. Frontend: extend `demo/app/src/App.jsx`

One new tab, `"▶ Live"`, alongside the existing `Q1`/`Q2`/`Q3` tabs (unchanged — they still
replay `data.json`, just now built from the new corpus per §2). Selecting it reveals a form
(question box pre-filled with a placeholder + a small "try: <curated question>" chip for each of
the 2 examples, API key field — `type="password"`, stored in `sessionStorage` only so it doesn't
survive a browser restart, a model dropdown restricted to the models in the backend's price table
— just `gpt-4o-mini` initially, so the cost meter is never guessing — and strategy checkboxes for
Sieve / Search-Visit with a note that picking both runs Compare mode) and a Run button.

On submit: open the SSE stream (`fetch` + `ReadableStream`, since `EventSource` can't send a
POST body — a small manual SSE line-parser, a well-known ~20-line pattern) to `/api/run`. Each
incoming `step` event is pushed through the **same per-step-type expansion** `toEvents` already
does for replay (`search` → agent-line + result cards; `fetch`/`visit` → agent-line + a card) —
reused, not reimplemented — except events are appended to the visible list as they arrive rather
than pre-computed and paced by the fixed `dwell` timer, since real network/API latency already
provides the pacing. The query-line typewriter effect is kept for visual consistency with replay;
result cards appear immediately once their step lands (no artificial extra delay).

New card: **`VisitCard`** (doc id/title + capped full text) for `Bm25Visit`'s `visit` steps —
visually similar to the existing `SectionCard` but labeled "whole document" instead of a named
section, making the Sieve-vs-Search-Visit contrast (named section vs whole document) visible at a
glance, which is exactly the paper's point.

New always-visible strip per running strategy: a **cost meter** (`$0.00312 · 1,240 tokens · 4
steps · 3.2s`, updating on every `step` event) using the cumulative fields the backend already
sends. In Compare mode (2 strategies picked), the stream renders as two side-by-side columns, each
with its own agent-line/card feed and its own cost meter, converging into a small bar comparison
(tokens / $ / steps / wall-clock) once both `done` events have arrived.

Reused components: `Typewriter`, `HitCard`, `SectionCard`, `EvidenceCard`, the sidebar "THE
COLLECTION" shelf (now showing ~90 docs instead of 8 — still fine as a scrollable list, no new
UI needed there since it was already a scrollable div).

## 6. Error handling

- No/invalid API key, or the OpenAI call fails (bad key, rate limit, network): the SSE stream
  sends `event: error` with a message; the frontend shows it inline in the Live panel (not a raw
  crash) and re-enables the Run button.
- Frontend can't reach `/api/run` at all (server not running, e.g. someone opened `index.html` via
  `file://` with no live server up): the fetch fails immediately; show a small inline "start the
  local live server: `python demo/live/server.py`" message instead of a silent hang.
- Backend thread raises unexpectedly (bug in a workspace, etc.): caught around `run_episode`,
  surfaced as the same `event: error` path — the SSE stream always terminates cleanly one way or
  another, never leaves the frontend spinning forever.
- `max_steps=12` (matching the existing recordings' budget) bounds every live run's cost and
  latency regardless of how the model behaves.

## 7. New dependencies

Added as a new optional extra so installing the paper's reproduction stack is unaffected:

```toml
[project.optional-dependencies]
demo-live = ["fastapi", "uvicorn[standard]"]
```

(`openai` is already a dependency of the `api` extra `agent_search/models/backends.py` uses.)
`demo/live/README.md` documents: `pip install -e ".[demo-live]"` then
`python demo/live/server.py`, mirroring the existing `demo/README.md`'s tone.

## 8. Testing

- `tests/test_agent_loop.py` (existing): add a case asserting `on_step` is called once per
  `steps.append` with the right `Step`, and that omitting it changes nothing (byte-identical
  `Trajectory` to today, using the existing fixture policy).
- `tests/test_backends_dispatch.py` (existing): add a case asserting `make_generate(...,
  api_key=X)` reaches the underlying client construction (mock/inject `client`, assert the key
  used matches — the existing tests already inject `client` for offline testing).
- New `tests/test_demo_live_parse.py`: unit tests for `parse_visit` (new) and confirms
  `parse_search`/`parse_fetch` still import correctly from their new shared location.
- New `tests/test_demo_live_server.py`: FastAPI `TestClient`-based tests for `/api/run` — a fake
  `generate` client injected so no real API calls happen; asserts SSE framing, that `step` events
  arrive before `done`, and that a simulated tool failure produces `event: error` not a hang.
- Manual verification pass (this design doc's author will run this once implemented): start the
  server for real with a real key, run both strategies on both curated questions plus one
  free-text question, confirm Compare mode and the cost meter visually.

## 9. File-level summary of changes

| file | change |
|---|---|
| `agent_search/agent/loop.py` | add optional `on_step` param (additive) |
| `agent_search/models/backends.py` | add optional `api_key` param to `make_generate`, forwarded (additive) |
| `demo/recorder/build_corpus.py` | **new** — offline curation script (§2) |
| `demo/recorder/browsecomp_corpus.py` | **new** — generated output of the above, checked in |
| `demo/recorder/corpus.py` | deleted once the above is in place |
| `demo/recorder/parse.py` | **new** — `parse_search`/`parse_fetch` extracted from `build_web.py`, `+parse_visit` |
| `demo/recorder/build_web.py` | import parsers from `parse.py` instead of defining them locally |
| `demo/recorder/record.py` | point at `browsecomp_corpus` instead of `corpus` |
| `demo/live/server.py` | **new** — FastAPI app (§4) |
| `demo/live/README.md` | **new** — setup/run instructions |
| `demo/app/src/App.jsx` | add Live tab, Compare mode, cost meter, `VisitCard` |
| `pyproject.toml` | add `demo-live` extra |
| `tests/test_demo_live_parse.py`, `tests/test_demo_live_server.py` | **new** |
| `tests/test_agent_loop.py`, `tests/test_backends_dispatch.py` | extended (existing files) |
