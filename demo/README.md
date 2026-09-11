# Demo: watch a research agent think

One page, one command. Ask a question about a curated collection of **250 real BrowseComp-Plus
documents**. The page runs the `agent_search` loop and streams each search, read and answer step by
step, with a running cost meter on every token.

```bash
pip install -e ".[demo-live]"
python demo/server.py          # -> http://localhost:8008/
```

The server ships a prebuilt page (`demo/index.html`), so Node is needed only to change the
frontend. To do that, run `cd demo/app && npm install && npm run build`. The server picks up
`app/dist/` the next time the page loads.

The OpenAI API key is typed into the page. A run is capped at 20 steps and costs well under a cent
on gpt-4o-mini. The key lives in the browser's `sessionStorage` and is sent per request straight to
OpenAI. The server never stores it and never logs it.

**Race** runs the same question through two strategies side by side: **Sieve** (Boolean search plus
fetching named sections, the paper's method) against **Search-Visit** (BM25 plus reading whole
documents, the baseline). Each gets its own live meter, and the run ends in a head-to-head chart of
tokens, cost, steps and time.

```
demo/
├── server.py          the whole backend: FastAPI + SSE, wraps the real agent loop
├── app/               React frontend (Vite): npm install && npm run build
├── index.html         prebuilt single-file page (the server serves app/dist/ or this)
├── corpus.py          loads the collection for the agent's tools
├── corpus_data.json   the curated 250-doc BrowseComp-Plus subsample (checked in)
├── build_corpus.py    one-time offline curation script (documents the provenance)
└── parse.py           tool-observation -> card parsers shared by server tests
```

The page ships 6 example questions (they live in `corpus_data.json`). Those are real
BrowseComp-Plus queries with known gold answers, so the page checks the run against gold when one
is selected. Free-text questions about anything in the collection also work.

The demo pins the paper's read budgets explicitly rather than relying on the library defaults:
`MAX_VISIT_TOKENS` and `MAX_SECTION_TOKENS` are both set to 12,000, and `SNIPPET_TOKENS` stays at
32. Set `SNIPPET_TOKENS` in the environment before starting the server (32, 64, 128, 256 or 512)
and the search cards widen to match, which shows the snippet-width sweep.

## Data provenance

`corpus_data.json` is a curated subsample of **BrowseComp-Plus** (Chen et al.): 250 documents and 6
questions picked by `build_corpus.py` from the `Tevatron/browsecomp-plus` release on Hugging Face,
joined against the paper's sectioned corpus
([`wshuai190/browsecomp-plus-structured-full`](https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full)),
with the benchmark's obfuscation removed so the page can show readable text.

Those documents are web pages crawled for that benchmark. They are redistributed here as demo
material and stay under the terms of the upstream dataset and their original publishers. This
repository's Apache-2.0 licence covers the code, not these documents. Reuse of the subsample
requires citing BrowseComp-Plus and following its data terms.
