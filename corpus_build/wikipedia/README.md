# Wikipedia structured rebuild (hotpotqa / 2wiki / musique)

These multi-hop datasets are **all Wikipedia, keyed by article title**, but staged as *flat*
intro paragraphs. This builder rebuilds each doc from the **full structured article** in
`wikimedia/structured-wikipedia` (pre-parsed sections + infobox + abstract). The Wikipedia title
**is** the article identity, so a title match (after normalization) is trusted directly — no
content check. It emits a **flat + structured PAIR over the same docs** so BQL's benefit can be
isolated: only the scopeable fields differ.

Run from **inside this folder**: `cd corpus_build/wikipedia`

## Setup
```bash
pip install pyarrow                   # enough for the FAST local-scan path (--sw-path)
pip install datasets                  # only for the streaming engine (no pre-download)
pip install duckdb huggingface_hub    # only for --engine duckdb
hf auth login                         # if prompted (was: huggingface-cli login)
```

## 1. Verify the real schema first (don't trust, check)
```bash
python build.py inspect               # a few arbitrary articles — INSTANT (first rows)
python build.py inspect --title Sanaa # find ONE title — LINEAR scan (minutes); usually skip this
```
If the printed `>>> parsed: N sections [...]` looks wrong vs the raw JSON above it, tell me and
I'll adjust the parser — but it's written defensively (handles `value`-string vs structured
runs, nested `list_item`s, tables, images, unknown part types).

## 2. Build

**Prerequisite:** the flat staging must already exist at `../../data/<name>/corpus.jsonl`
(+ `queries.jsonl` + `qrels/`). Stage it first with `scripts/stage_multihop.py` if it's missing.

**Fast path — download the parquet ONCE, then scan all shards in parallel** (recommended; the
download is reused across all three datasets):
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
Each run prints the match rate, **GOLD coverage**, and how many docs/queries were dropped.
The `--sw-path` scan reuses the EXACT title matching (by `name` AND url-derived title), so the
result is identical to streaming — just I/O-parallel from local disk instead of a network stream.

**No pre-download** (serial network stream, ~10 min/dataset):
```bash
python build.py build --dataset musique                  # streaming (needs `datasets`)
python build.py build --dataset hotpotqa --engine duckdb # snapshot_download + duckdb
```

## What it does
- Reads `../../data/<name>/corpus.jsonl` (the flat staging) → unique titles.
- Matches each title in structured-wikipedia by **`norm_title`** (NFC + HTML-entity unescape +
  underscores→spaces + casefold), against `name` **and** the url-derived title. If that misses,
  falls back to **`strip_key`** (also strip diacritics + drop all non-alphanumerics) so
  `Foo, Bar` / `Foo-Bar` / `Foo–Bar` / `Fóo Bar` collapse together. A title match is trusted —
  **no containment check** (the title is the article's identity).
- Parses the article **defensively, losing nothing**: section hierarchy (with levels), nested
  lists, tables (rows preserved), image captions, link-text inside paragraphs, and infoboxes
  (repeated fields accumulated, structured/wikilink values coerced to text).
- A title with **no article** (renamed / deleted) is **dropped**; the corpus is a SUBSET of the
  original. Then **queries/qrels are pruned**: gold pointing at a dropped doc is removed, and any
  query left with no gold is dropped. (`--dump-misses FILE` writes the dropped titles, plus a
  `FILE.gold.txt` with the gold subset.)
- **GOLD coverage** is reported — what fraction of answer-bearing docs survived (the number that
  decides how well-powered the structured arm is).

## Output — a flat + structured PAIR (same docs, same text, same queries)
```
data/<name>_structured/corpus.jsonl : {_id, title, sections, section, infobox, text}   ← scopeable fields + matched sections
data/<name>_flat/corpus.jsonl       : {_id, title,                            text}   ← same text, no fields
```
`sections` is a **matched, ordered list of `{heading, text}` parts** — the article's real `##`
sections kept paired (heading ↔ its own body), so `fetch` reads a real slice and `IN(section,·)`
scopes to a named part instead of re-deriving them from `##` markers. `section` is the joined
headings (kept for the search listing); `text` is the full structured markdown body (infobox kv +
`## Section` headers + paragraphs).

Both arms carry **byte-identical `text`**, and `units_from_documents` builds the bm25/dense blob
from **`title` + `text` ONLY** — the `sections`/`section`/`infobox` **fields** are NOT in that blob
(they go to `u.sections` / `u.section` / metadata, which only BQL's `IN(section/infobox,·)` +
`fetch` read). So the lexical/dense input is **identical across both arms** (the section headings
are already inside `text` once, not double-counted) — the pair isolates exactly the value of
structure, with no bm25/dense confound. (Guarded by `tests/test_corpus_identity.py`.)

## Next (in the main repo — already wired)
`<name>_flat` and `<name>_structured` are registered (general domain; structured gets the
`wiki` BQL manual, flat the title/body manual). Run the toolset conditions
(`agent_research` / `agent_research_bm25` / `agent_research_dci`) on **both** the
`_flat` and `_structured` datasets → the paired result.
