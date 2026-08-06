# Demo — watch a research agent think

One page, one command: ask any question over a curated collection of **102 real
BrowseComp-Plus documents** and watch the real `agent_search` loop search, read, and answer —
live, streamed step by step, with every token on a running cost meter.

```bash
pip install -e ".[demo-live]"
python demo/server.py          # -> http://localhost:8008/
```

Bring your own OpenAI API key (typed into the page; a run is capped at 12 steps and costs
well under a cent on gpt-4o-mini). The key travels browser → this local server → OpenAI, is
used for that one request only, and is never stored or logged.

**Compare mode** runs the same question through two strategies side by side — **Sieve**
(Boolean search + fetch named sections, the paper's method) vs **Search-Visit** (BM25 + read
whole documents, the classic baseline) — each with its own live meter, ending in a
head-to-head chart of tokens / cost / steps / time. That contrast is the paper's headline
claim, live on your screen.

```
demo/
├── server.py          the whole backend: FastAPI + SSE, wraps the real agent loop
├── app/               React frontend (Vite): npm install && npm run build
├── index.html         prebuilt single-file page (the server serves app/dist/ or this)
├── corpus.py          loads the collection for the agent workspaces
├── corpus_data.json   the curated 102-doc BrowseComp-Plus subsample (checked in)
├── build_corpus.py    one-time offline curation script (documents the provenance)
└── parse.py           tool-observation -> card parsers shared by server tests
```

The two example questions on the page are real BrowseComp-Plus queries with known gold
answers (the page checks your run against gold when you use them). Free-text questions about
anything in the collection work too.
