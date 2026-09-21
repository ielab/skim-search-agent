# Wikipedia structured rebuild (hotpotqa / 2wiki / musique)

These multi-hop datasets are all Wikipedia, keyed by article title, but they are staged as flat
intro paragraphs. This builder rebuilds each doc from the full structured article in
`wikimedia/structured-wikipedia`, which already has parsed sections, an infobox and an abstract.
The Wikipedia title *is* the article's identity, so a title match (after normalization) is trusted
directly, with no content check. The output is a flat + structured pair over the same docs, which
isolates BQL's benefit: only the scopeable fields differ.

Run everything from **inside this folder**: `cd corpus_build/wikipedia`

## Setup
```bash
pip install pyarrow                   # enough for the FAST local-scan path (--sw-path)
pip install datasets                  # only for the streaming engine (no pre-download)
pip install duckdb huggingface_hub    # only for --engine duckdb
hf auth login                         # if prompted (or: huggingface-cli login)
```

## 1. Verify the real schema first
```bash
python build.py inspect               # a few arbitrary articles: INSTANT (first rows)
python build.py inspect --title Sanaa # find ONE title: LINEAR scan (minutes); usually skip this
```
Compare the printed `>>> parsed: N sections [...]` against the raw JSON above it. The parser is
written defensively (it handles `value`-string versus structured runs, nested `list_item`s, tables,
images and unknown part types). Report a parse that looks wrong for a given slice of the dump as an
issue.

## 2. Build

**Prerequisite:** the flat staging must already exist at `../../data/<name>/corpus.jsonl`, along
with `queries.jsonl` and `qrels/`. If it is missing, stage it first with
`scripts/stage_multihop.py`.

**Fast path.** Download the parquet once, then scan all shards in parallel. This is the recommended
route, and the download is reused across all three datasets:
```bash
# NB: the config `enwiki_namespace_0` lives at `enwiki/data/*.parquet` in the repo (the
# config name is NOT the folder), so download that path:
hf download wikimedia/structured-wikipedia --repo-type dataset \
    --include "enwiki/data/*.parquet" --local-dir ./sw
python build.py build --dataset musique  --sw-path ./sw --dump-misses musique_misses.txt   # ~18k titles
python build.py build --dataset hotpotqa --sw-path ./sw --dump-misses hotpotqa_misses.txt  # ~66k titles
python build.py build --dataset 2wiki    --sw-path ./sw --dump-misses 2wiki_misses.txt
#   --workers N  : parallel shards (default min(8, cpus); each shard ~hundreds of MB RAM)
#   --dump-misses: write the dropped (renamed/deleted) titles + a .gold.txt of the gold subset
#   --no-strip   : exact-title match only (skip the strip-normalization fallback)
```
Every run prints the match rate, the **gold coverage**, and how many docs and queries were dropped.
The `--sw-path` scan reuses the exact same title matching (by `name` and by the url-derived title),
so the result is identical to streaming. It is I/O-parallel off local disk instead of over the
network.

**No pre-download** (serial network stream, roughly 10 minutes per dataset):
```bash
python build.py build --dataset musique                  # streaming (needs `datasets`)
python build.py build --dataset hotpotqa --engine duckdb # snapshot_download + duckdb
```

## What it does
- Reads `../../data/<name>/corpus.jsonl` (the flat staging) and pulls out the unique titles.
- Matches each title in structured-wikipedia by **`norm_title`** (NFC, HTML-entity unescape,
  underscores to spaces, casefold), against both `name` and the url-derived title. If that misses,
  it falls back to **`strip_key`**, which also strips diacritics and drops every non-alphanumeric,
  so `Foo, Bar`, `Foo-Bar`, `Foo–Bar` and `Fóo Bar` all collapse together. A title match is trusted
  with no containment check, because the title is the article's identity.
- Parses the article defensively and loses nothing: section hierarchy with levels, nested lists,
  tables with their rows preserved, image captions, link text inside paragraphs, and infoboxes
  (repeated fields accumulate, and structured or wikilink values are coerced to text).
- Drops any title with no article behind it (renamed or deleted), so the corpus is a subset of the
  original. Then it prunes queries and qrels: gold pointing at a dropped doc is removed, and a query
  left with no gold is dropped. `--dump-misses FILE` writes the dropped titles plus a
  `FILE.gold.txt` with the gold subset.
- Reports **gold coverage**, the fraction of answer-bearing docs that survived. That number
  determines how well-powered the structured arm is.

## Output: a flat + structured pair (same docs, same text, same queries)
```
data/<name>_structured/corpus.jsonl : {_id, title, sections, section, infobox, text}   ← scopeable fields + matched sections
data/<name>_flat/corpus.jsonl       : {_id, title,                            text}   ← same text, no fields
```
`sections` is a matched, ordered list of `{heading, text}` parts: the article's real `##` sections,
kept paired with their own bodies. That lets `fetch` read a real slice and `IN(section,·)` scope to
a named part, instead of re-deriving them from `##` markers. `section` is the joined
headings, kept for the search listing, and `text` is the full structured markdown body (infobox
key-values, `## Section` headers, paragraphs).

Both arms carry byte-identical `text`, and `units_from_documents` builds the bm25/dense blob from
`title` + `text` **only**. The `sections`, `section` and `infobox` fields are not in that blob; they
go to `u.sections`, `u.section` and metadata, which only BQL's `IN(section,·)` / `IN(infobox,·)` and
`fetch` read. The lexical and dense input is therefore identical across both arms (the section
headings are already inside `text` once, not double-counted), and the pair isolates the value of
structure with no bm25/dense confound. `tests/test_corpus_identity.py` guards this.

## Next
`<name>_flat` and `<name>_structured` are registered as general-domain datasets. The structured arm
gets the `wiki` BQL manual, the flat arm gets the title/body manual. Run the toolset conditions
(`agent_research`, `agent_research_bm25`, `agent_research_dci`) on **both** the `_flat` and
`_structured` datasets to get the paired result.
