# corpus_build: recovering document structure for the deep-research corpora

The deep-research benchmarks ship their documents as flat text, because the upstream extractors
flattened whatever structure was there. BQL's advantage is structural (`IN(title,·)`,
`IN(section,·)`, `IN(author,·)`, field scope), and on flat prose it collapses into plain keyword
search. These tools rebuild structured variants of the corpora, which supports the paired
flat-vs-structured experiment and measures how BQL's benefit scales with the available structure.

Neither builder scrapes the web. The structure already exists upstream, in a structured-Wikipedia
mirror or in the shipped frontmatter. Gold is keyed by doc id, so the flat and structured arms
always evaluate the same queries.

## Two builders, pick by corpus type

| folder | corpus | structure source | fields | output |
|---|---|---|---|---|
| [`wikipedia/`](wikipedia/) | `hotpotqa` / `2wiki` / `musique` | **`structured-wikipedia` lookup** (HF `wikimedia/structured-wikipedia`), all Wikipedia, keyed by title | `title` / `section` / `infobox` / `body` | a **flat + structured pair** over a doc SUBSET |
| [`browsecomp_plus/`](browsecomp_plus/) | `browsecomp_plus` (~68K pooled web docs) | **shipped frontmatter** + **LLM-inserted sections** (one `gpt-5.4-nano` Batch pass, no fetching) | `title` / `author` / `date` / `section` / `body` | a **flat + structured pair**, **all** docs kept |

Run each one from **inside** its own folder. Both have a README with the exact commands.

## Published datasets on Hugging Face: pull instead of rebuild

The corpora used in the paper are published and can be downloaded instead of rebuilt. That
matters most for browsecomp: its structured corpus came from a one-time `gpt-5.4-nano` sectioning
batch that a rebuild has to pay for again. Its `sectionize-validate` stage projects the cost
beforehand; see `browsecomp_plus/README.md`.

| dataset | contents |
|---|---|
| [`wshuai190/browsecomp-plus-structured-full`](https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full) | the complete 100,195-doc collection: `structured/` + `flat/` (each `corpus.jsonl` + `queries.jsonl` + `qrels/test.tsv`) **+ `sections.jsonl`** (raw `{_id, sections}` map) |
| [`wshuai190/hotpotqa-structured`](https://huggingface.co/datasets/wshuai190/hotpotqa-structured) | HotpotQA flat + structured twin, same layout |
| [`wshuai190/musique-structured`](https://huggingface.co/datasets/wshuai190/musique-structured) | MuSiQue flat + structured twin, same layout |
| *2wiki* | not published; rebuild it with [`wikipedia/`](wikipedia/) (sections are native to `structured-wikipedia`, so there's no paid batch) |

The published browsecomp dataset is the **full** 100,195-doc collection, which registers as
`browsecomp_plus_structured_full` / `browsecomp_plus_flat_full`.
The local builder in `browsecomp_plus/` produces the smaller 67,707-doc pooled pair
(`browsecomp_plus_structured` / `browsecomp_plus_flat`) from the benchmark's own `gold_docs`. Same
830 queries either way, different corpus size.

Pull and stage into `data/` (the `data/<dataset_name>/` BEIR layout the harness expects):
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
The `flat/` folder inside `wshuai190/browsecomp-plus-structured-full` predates the fix that made
flat text plain: it still carries the leading date line and the `## heading` lines from the
structured arm, so a fresh pull needs a rebuild with `python browsecomp_plus/build.py corpus
--sections data/_hf/bcp/sections.jsonl` before its flat corpus is correct.

## How the two builders differ (they're not symmetric)

**`wikipedia/`**: the title *is* the article's identity, so a title match (after norm and strip
normalization) is trusted directly, with no content check. A title with no article behind it
(renamed or deleted) gets **dropped**, and any query left without gold is pruned, so the corpus
ends up a **subset** of the original. It writes both arms itself, `data/<name>_flat/` and
`data/<name>_structured/`, over the same kept docs and the same pruned queries.

**`browsecomp_plus/`**: it parses `title`, `author` and `date` out of each doc's own frontmatter,
and gets `sections` from a one-time `gpt-5.4-nano` Batch pass. The model proposes section
boundaries; a deterministic step applies them to the original body, so the text is preserved and
only `## Heading` markers are inserted. **Every doc is kept**: no frontmatter means empty fields, no
model reply means intro-only. It writes both arms, `data/browsecomp_plus_{flat,structured}/`, over
the same doc ids and the same `title`. The two twins are NOT byte-identical: the structured arm
adds `author`, `date`, `sections`, and a `text` with a leading date line and the `## heading` lines
folded in; the flat arm's `text` is the plain original body, frontmatter stripped, nothing added.

## Shared properties
- **Gold never moves.** Output is keyed by the same `_id`, so gold always resolves.
- **No web fetch, no drift.** Both read from local or hub data only.
- **Fairness.** The extra fields also get folded into the searchable `text` (infobox for wiki,
  author and date for browsecomp), so `bm25` and `dense` see them too. BQL gets *precise* field
  access through `IN(field,·)`, not *exclusive* access. The fields additionally ride along in unit
  metadata for BQL to scope against.

## Output schema (drops into the existing harness with no code change)
Both builders emit `sections` as a matched, ordered list of `{heading, text}` parts, with each `##`
section kept together with its own body. That lets `fetch` read a real slice and `IN(section,·)`
scope to a named part. The joined-heading `section` string is kept as well, for the search
listing.
```json
// wikipedia: a PAIR over the same docs/text; only the fields differ
//   data/<name>_structured/corpus.jsonl
{"_id":"...","title":"...","sections":[{"heading":"History","text":"..."},{"heading":"Geography","text":"..."}],
 "section":"History Geography","infobox":"country: Yemen ...","text":"<infobox kv>\n\n## History\n...\n## Geography\n..."}
//   data/<name>_flat/corpus.jsonl   (same _id, same text, NO fields)
{"_id":"...","title":"...","text":"<infobox kv>\n\n## History\n...\n## Geography\n..."}

// browsecomp_plus_structured (sections from the gpt-5.4-nano pass)
{"_id":"...","title":"...","author":"...","date":"...",
 "sections":[{"heading":"(intro)","text":"..."},{"heading":"Recording","text":"..."}],
 "text":"By <author> <date>\n\n<body with ## headings>"}
// browsecomp_plus_flat (same _id, same title, NO other fields)
{"_id":"...","title":"...","text":"<the plain original body, frontmatter stripped>"}
```
`units_from_documents` maps `title`, `section`, `sections` and `text` straight across, and carries
any extra key (`infobox`, `author`, `date`) into unit metadata. `IN(title,·)`, `IN(section,·)`,
`IN(infobox,·)`, `IN(author,·)`, `IN(date,·)`, `IN(body,·)` and section `fetch` therefore all work
without touching the executor. Whole-doc units are adequate here: BM25 reads the full text and a
long-context dense embedder is not truncated, so there is no need to chunk by section.

## Dependencies (run these on a node with internet; the GPU node stays offline)
- `wikipedia/`: `datasets`, plus optional `duckdb huggingface_hub` for the fast index path.
- `browsecomp_plus/`: `datasets` (it reads the hub).
- Both: `tqdm` for progress bars.

Copy the resulting `data/*_flat/` and `data/*_structured/` over to the GPU node afterwards.

## Wiring
1. Datasets are registered in `agent_search/evaluation/datasets/beir.py`: `<name>_flat` and
   `<name>_structured` for `hotpotqa`, `2wiki` and `musique` (structured gets
   `field_profile=wiki`), plus `browsecomp_plus_structured` (`field_profile=browsecomp`).
2. One shared doc BQL manual, `agent_search/tools/search_bql/bql_doc.md` (title/body/section/infobox),
   covers both the flat arms (no `field_profile`, so it falls back to `domain=general` and
   section/infobox never populate) and the `wiki` profile. `bql_browsecomp.md` in the same folder
   (title/author/date/body, no section/infobox) is the corpus-correct manual for the `browsecomp`
   profile. The `search_bql` tool's manual map picks one per dataset by `field_profile`, so a
   manual only ever advertises the corpus's real fields.
3. Run the method against its baselines on **both** arms of each pair:
   `agent_research_bql_dense_snip` (the method) vs `agent_research_bm25` (retrieve-then-visit) vs
   `agent_research_dci` (brute-force shell; the names are in `agent_search/strategies/paper.py`).
   - wikipedia: `<name>_flat` vs `<name>_structured`
   - browsecomp: `browsecomp_plus` (original flat) vs `browsecomp_plus_structured`

## ITER layout: topics and qrels over a chunked corpus, served from disk

The third input shape needs no builder at all. Stage the files (symlinks are fine):

```
data/<dataset>/topics.tsv                 # id <TAB> question [<TAB> answer]
data/<dataset>/qrels.txt                  # TREC: qid Q0 docid rel   (optional; without it the set is answer-only)
data/<dataset>/answers.tsv                # id <TAB> answer         (optional, when the answer is not in topics.tsv)
data/corpora/<corpus>/corpus.jsonl        # {"docid": ..., "text": ...}; title = a `title` field, a `---\ntitle:` front matter, or the first line
```

and register the dataset in one line (`corpus=` lets several topic sets share one corpus and its indexes):

```python
from agent_search.evaluation.datasets import register_dataset, _topics_qrels_loader
register_dataset("my_topics", domain="general")(_topics_qrels_loader("my_topics", corpus="wiki25_512"))
```

A corpus above 1 GiB (`AGENT_SEARCH_DOCSTORE_MIN_BYTES`) is not loaded: the loader writes a
byte-offset index next to it once and every later lookup is one seek. Retrieval then needs
prebuilt indexes named in the experiment file, `retrieval.dense_index` (a FAISS index with its
docid lookup) and/or `retrieval.bm25_index` (a Lucene index); the in-memory engines refuse such
a corpus with a message that says which knob to set.
