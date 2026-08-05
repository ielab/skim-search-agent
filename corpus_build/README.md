# corpus_build — recover document STRUCTURE for the deep-research corpora

The deep-research benchmarks ship their documents as **flat text** (the upstream extractors
flattened the original structure). BQL's edge is *structural* — `IN(title,·)`, `IN(section,·)`,
`IN(author,·)`, field scope — so on flat prose it degenerates to keyword search. These tools
rebuild **structured variants** of the corpora so we can run the **paired flat-vs-structured
experiment** and show BQL's benefit scales with available structure.

Neither builder scrapes the web — the structure already exists upstream (a structured-Wikipedia
mirror; the shipped frontmatter). Gold is keyed by doc id, so the flat and structured arms always
evaluate the **same** queries.

## Two builders — pick by corpus type

| folder | corpus | structure source | fields | output |
|---|---|---|---|---|
| [`wikipedia/`](wikipedia/) | `hotpotqa` / `2wiki` / `musique` | **`structured-wikipedia` lookup** (HF `wikimedia/structured-wikipedia`) — 100% Wikipedia keyed by title | `title` / `section` / `infobox` / `body` | a **flat + structured pair** over a doc SUBSET |
| [`browsecomp_plus/`](browsecomp_plus/) | `browsecomp_plus` (~68K diverse web docs) | **shipped frontmatter** + **LLM-inserted sections** (one-time `gpt-5.4-nano` Batch pass; no fetch) | `title` / `author` / `date` / `section` / `body` | a **flat + structured pair**; **all** docs kept |

Run **inside** the relevant folder; each has its own README with exact commands.

## Published datasets (Hugging Face) — pull instead of rebuild

The built corpora used in the paper are published on Hugging Face, so you can pull them instead
of rebuilding (the browsecomp structured corpus took a one-time ~$19 `gpt-5.4-nano` sectioning
batch — see `browsecomp_plus/README.md`):

| dataset | contents |
|---|---|
| [`wshuai190/browsecomp-plus-structured-full`](https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full) | the complete 100,195-doc collection: `structured/` + `flat/` (each `corpus.jsonl` + `queries.jsonl` + `qrels/test.tsv`) **+ `sections.jsonl`** (raw `{_id, sections}` map) |
| [`wshuai190/hotpotqa-structured`](https://huggingface.co/datasets/wshuai190/hotpotqa-structured) | HotpotQA flat + structured twin, same layout |
| [`wshuai190/musique-structured`](https://huggingface.co/datasets/wshuai190/musique-structured) | MuSiQue flat + structured twin, same layout |
| *2wiki* | not published yet — rebuild with [`wikipedia/`](wikipedia/) (sections are native to `structured-wikipedia`, so no paid batch) |

Pull + stage into `data/` (the `data/<dataset_name>/` BEIR layout the harness expects):
```bash
huggingface-cli download wshuai190/browsecomp-plus-structured-full --repo-type dataset --local-dir data/_hf/bcp
cp -r data/_hf/bcp/structured data/browsecomp_plus_structured_full
cp -r data/_hf/bcp/flat       data/browsecomp_plus_flat_full

huggingface-cli download wshuai190/hotpotqa-structured --repo-type dataset --local-dir data/_hf/hotpotqa
cp -r data/_hf/hotpotqa/structured data/hotpotqa_structured
cp -r data/_hf/hotpotqa/flat       data/hotpotqa_flat

huggingface-cli download wshuai190/musique-structured --repo-type dataset --local-dir data/_hf/musique
cp -r data/_hf/musique/structured data/musique_structured
cp -r data/_hf/musique/flat       data/musique_flat

# browsecomp only: sections.jsonl lets you re-assemble the pair WITHOUT the paid batch:
#   python browsecomp_plus/build.py corpus --sections data/_hf/bcp/sections.jsonl
```

## How the two differ (important — they are NOT symmetric)
- **`wikipedia/`** — the title *is* the article identity, so a title match (after norm/strip
  normalization) is trusted directly, **no content check**. A title with **no article**
  (renamed/deleted) is **dropped**, and queries left with no gold are pruned → the corpus is a
  **subset** of the original. It writes BOTH arms itself: `data/<name>_flat/` and
  `data/<name>_structured/` over the **same** kept docs and pruned queries.
- **`browsecomp_plus/`** — parses `title`/`author`/`date` from each doc's own frontmatter, and
  gets `sections` from a one-time `gpt-5.4-nano` Batch pass that proposes section boundaries which a
  **deterministic** step applies to the ORIGINAL body (text preserved, only `## Heading` markers
  inserted). **Keeps every doc** (no frontmatter → empty fields; no reply → intro-only). It writes
  BOTH arms — `data/browsecomp_plus_{flat,structured}/` — over the same sectionized `text` (the flat
  twin carries the `## headings` in `text` too, just no scopeable fields).

## Shared properties
- **Gold never moves.** Output is keyed by the same `_id`; gold always resolves.
- **No web fetch / no drift.** Both read from local/hub data only.
- **Fairness.** Extra fields are also folded into the searchable `text` (wiki: infobox; browsecomp:
  author/date), so `bm25`/`dense` see them too — BQL gets *precise* field access via `IN(field,·)`,
  not *exclusive* access. The fields additionally ride in unit metadata for BQL to scope.

## Output schema (drops into the existing harness with no code change)
Both builders emit `sections` as a **matched, ordered list of `{heading, text}` parts** (each `##`
section kept WITH its own body), so `fetch` reads a real slice and `IN(section,·)` scopes to a named
part. The joined-heading `section` string is also kept for the search listing.
```json
// wikipedia: a PAIR over the same docs/text — only the fields differ
//   data/<name>_structured/corpus.jsonl
{"_id":"...","title":"...","sections":[{"heading":"History","text":"..."},{"heading":"Geography","text":"..."}],
 "section":"History Geography","infobox":"country: Yemen ...","text":"<infobox kv>\n\n## History\n...\n## Geography\n..."}
//   data/<name>_flat/corpus.jsonl   (same _id, same text, NO fields)
{"_id":"...","title":"...","text":"<infobox kv>\n\n## History\n...\n## Geography\n..."}

// browsecomp_plus_structured (sections from the gpt-5.4-nano pass)
{"_id":"...","title":"...","author":"...","date":"...",
 "sections":[{"heading":"(intro)","text":"..."},{"heading":"Recording","text":"..."}],
 "text":"By <author> <date>\n\n<body with ## headings>"}
```
`units_from_documents` maps `title`/`section`/`sections`/`text` directly and carries any extra key
(`infobox`/`author`/`date`) into unit metadata — so `IN(title,·)`, `IN(section,·)`, `IN(infobox,·)`,
`IN(author,·)`, `IN(date,·)`, `IN(body,·)` and section `fetch` all work with no executor change. Whole-doc units are
fine — BM25 reads the full text and a long-context dense embedder isn't truncated, so **no
section-chunking is needed**.

## Dependencies (run on a node WITH internet; the GPU node stays offline)
- `wikipedia/`: `datasets` (+ optional `duckdb huggingface_hub` for the fast index).
- `browsecomp_plus/`: `datasets` (reads the hub). Both: optional `tqdm` for progress bars.

Copy the resulting `data/*_flat/` and `data/*_structured/` to the GPU node afterwards.

## Wiring (already done in the main repo)
1. Datasets registered in `evaluation/datasets.py`:
   `<name>_flat` + `<name>_structured` for `hotpotqa`/`2wiki`/`musique` (structured gets
   `field_profile=wiki`), and `browsecomp_plus_structured` (`field_profile=browsecomp`).
2. One shared doc BQL skill — `skills/bql_doc.md` (title/body/section/infobox) — covers both the
   flat arms (`field_profile` unset, falls back to `domain=general`; section/infobox simply never
   populate) and the `wiki` profile; `skills/bql_browsecomp.md` (title/author/date/body, no
   section/infobox) is the corpus-correct manual for the `browsecomp` profile. Selected per
   dataset by `field_profile` via `tools.yaml`'s `search.manual` map (a skill advertises EXACTLY
   the corpus's real fields).
3. Run the method vs its baselines on **both** arms of each pair — `agent_research_bql_dense_snip` (the method)
   vs `agent_research_bm25` (retrieve-then-visit) vs `agent_research_dci` (brute-force shell,
   `conditions.yaml`):
   - wikipedia: `<name>_flat` vs `<name>_structured`
   - browsecomp: `browsecomp_plus` (original flat) vs `browsecomp_plus_structured`
