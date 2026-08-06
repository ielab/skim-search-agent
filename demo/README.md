# Demo — watch a research agent think

**Open [`index.html`](index.html).** That's it — one file, no key, no GPU, no server.

A React player of real recorded gpt-4o-mini episodes: the agent **types out** each Boolean
search live, result cards animate in (title, § section chips, matched fields, query-highlighted
snippets), every fetched section slides into the **Evidence Collected** panel, the collection
shelf lights up as documents are surfaced and read, and the final answer is revealed and checked
against gold. Play/pause (spacebar), speed slider, one tab per question. All three episodes are
genuine unedited runs — including the step where the agent fumbles a section name and corrects
itself.

```
demo/
├── index.html    ← the demo (prebuilt, self-contained — just open it)
├── app/          the React source (Vite): npm install && npm run dev
└── recorder/     how episodes are made
    ├── browsecomp_corpus.py  loads the curated ~102-doc BrowseComp-Plus subsample + 2 questions
    ├── browsecomp_data.json  the checked-in corpus data (generated once by build_corpus.py)
    ├── build_corpus.py       one-time offline curation script (Hugging Face -> the JSON above)
    ├── record.py        gpt-4o-mini drives the library's DocSearchFetch workspace
    ├── episodes.jsonl   the recorded episodes (all correct, 3–6 steps)
    └── build_web.py     episodes.jsonl → app/src/data.json
```

Record fresh episodes (a few cents of gpt-4o-mini), then rebuild:

```bash
export OPENAI_API_KEY=...
python demo/recorder/record.py
python demo/recorder/build_web.py
cd demo/app && npm run build && cp dist/index.html ../index.html
```
