# Live demo — ask your own questions

The replay page shows recorded episodes; this server makes the same page **live**: type any
question, watch the real `agent_search` loop search/fetch in real time with a running
token/cost meter — and optionally race two strategies side by side.

    pip install -e ".[demo-live]"
    python demo/live/server.py        # -> http://localhost:8008/  (open it, pick the ▶ Live tab)

You bring your own OpenAI API key (gpt-4o-mini; a run costs well under a cent — the step
budget is capped at 12). The key travels browser → this local server → OpenAI, is used for
that one request, and is never logged or written anywhere.

The collection is the curated BrowseComp-Plus subsample (~102 real docs — see
`demo/recorder/build_corpus.py`). The two example chips are real BrowseComp-Plus queries with
known gold answers; free-text questions about anything in the collection work too.

**Compare mode:** tick both strategies and one question runs twice, side by side —
Sieve (Boolean search + fetch named sections) vs Search-Visit (BM25 + read whole documents) —
each with its own live cost meter, ending in a head-to-head bar chart. That contrast is the
paper's headline claim, live.
