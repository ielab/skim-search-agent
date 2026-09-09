# BrowseComp-Plus: structured corpus builder (frontmatter + LLM sections, no scraping)

BrowseComp-Plus ships its documents obfuscated (XOR'd with a canary). A de-obfuscated document is
a YAML **frontmatter** block with `title`, `author` and `date`, followed by the body. Those fields
are already in the data, so this builder fetches nothing: it de-obfuscates, parses the frontmatter,
and writes the corpus out.

It works over the documents the benchmark's own `gold_docs` point at, which is 67,707 unique docs.
The full 100,195-doc collection is separate, published on Hugging Face as
`wshuai190/browsecomp-plus-structured-full`.

**Sections need a model, because there is no prebuilt sectioned corpus on the hub.** Raw web pages
have no reliable section markup, so a **`gpt-5.4-nano`** pass proposes section boundaries and
headings for each page, and then a deterministic step splits the original body at those boundaries
into a matched `(heading, text)` list. The model only says *where* sections start. It never
rewrites text, so the corpus stays faithful and its `text` still matches the flat twin
byte-for-byte. The pass goes through the OpenAI **Batch API** (24h async, about 50% cheaper than
list price) with many docs per request, so it is one overnight job.

Gold is a label on the `docid`, so this never touches the qrels. The flat and structured arms
evaluate the same queries.

Run everything from **inside this folder**. It is self-contained.

## Setup (staging node, needs internet and `OPENAI_API_KEY`)
```bash
pip install datasets openai     # + tqdm for progress bars (optional)
huggingface-cli login           # if prompted
export OPENAI_API_KEY=...        # for the sectionize submit/collect/validate stages
cd corpus_build/browsecomp_plus
```

## Sectioning pipeline: validate, prepare, submit, collect, corpus
```bash
# 0. VALIDATE first: one cheap sync call on a few real docs; eyeball quality + project the cost.
#    ALWAYS run this before the paid batch so it can't spend money for nothing.
python build.py sectionize-validate --sample 8               # prints sections + a full-corpus cost estimate

# 1. PREPARE: hub docs -> SHARDED batch input files + a manifest (no API call). Shards roll over
#    at the API's per-file caps so every file is submittable.
python build.py sectionize-prepare --out-prefix batch_input --manifest manifest.jsonl
#    --max-input-tokens N : per-request input cap (default 250000); this is what chunks the docs
#    --max-docs N         : also cap docs packed per request (default 40)
#    --max-shard-mb X     : roll to a new shard past this (default 180; API cap ~200MB)
#    --max-requests N     : roll to a new shard past this (default 40000; API cap 50k)
#    --model M            : default gpt-5.4-nano

# 2. SUBMIT: upload every shard + create a batch per shard; ids saved to batch_ids.txt.
python build.py sectionize-submit --shards 'batch_input_*.jsonl' --save-id batch_ids.txt

# 3. COLLECT: after the batches finish (up to 24h), apply the boundaries -> sections.jsonl.
#    Idempotent; if any batch is still running it says so and exits (just re-run later).
python build.py sectionize-collect --batch-ids batch_ids.txt --manifest manifest.jsonl --out sections.jsonl

# 4. CORPUS: assemble the flat+structured PAIR with matched sections.
python build.py corpus --sections sections.jsonl            # -> ../../data/browsecomp_plus_{flat,structured}/
python build.py corpus                                      # (no --sections -> frontmatter-only, section-less)
```
Two other helpers: `python build.py diagnose --limit 2000` reports title/author/date coverage, and
`python build.py dump --out deobf.jsonl` de-obfuscates the text for inspection.

**What it costs.** Run `sectionize-validate` and read its projection rather than a number from
this README. It prints both list price and the Batch API's roughly 50% discount. Two notes on
reading it: the per-doc input estimate comes from a small sample and the corpus has a heavy tail of
very long articles, so the sample is noisy. The exact figure measured from the full prepared shards
is 556.8M input tokens. The `gpt-5.4-nano` rates baked into the script ($0.05/M in, $0.40/M out)
are assumptions, so pass `--price-in` and `--price-out` after checking the current pricing page.

**The prompt.** Each request shows the model a chunk of web pages, each with its title, author and
date, then its body as numbered lines, and asks it to segment the *body* into the sections a reader
would see on the rendered page (a lead, then topical sections). Section line numbers refer to body
lines only, so metadata never turns into a section. It has been validated on real pages: the
*Sanaa* Wikipedia article splits into 34 correctly-named sections (History, Ottoman era, Geography,
Economy, and so on), and a 1-line page stays a single intro.

> The prepare stage downloads the ~2.8 GB dataset once and writes ~1 GB of batch input, so run it
> on a **staging node**, not a laptop. Point `--out-prefix` and `--manifest` at a scratch dir
> outside the repo. Docs that no batch covers fall back to a frontmatter-only record, still
> emitted but section-less, and `corpus` prints the percentage that got matched sections.

## Output: a flat + structured pair (same docs, text and queries)
```
data/browsecomp_plus_structured/corpus.jsonl : {_id, title, author, date, sections, text}   ← scopeable fields + matched sections
data/browsecomp_plus_flat/corpus.jsonl       : {_id, title,                       text}    ← same text, no fields
```
`sections` is a matched, ordered list of `{heading, text}` parts, with each `## section` kept
together with its own body. The `text` is byte-identical across the pair; the structured arm
*also* exposes the fields and section parts. Each arm gets its own `queries.jsonl` and
`qrels/test.tsv`, generated from the dataset's own `gold_docs`, identical in both. The pair is
`browsecomp_plus_flat` vs `browsecomp_plus_structured`.

> The query and gold are read via the field names `query_id`, `query` and `gold_docs` in
> `iter_rows`. If a `--limit 50` run prints "0 queries", those names have changed on the hub. The
> fix is one line.

**Fairness.** `author` and `date` are emitted twice over: as their own fields (so
`units_from_documents` carries them into metadata and BQL's `IN(author,·)` / `IN(date,·)` can scope
them), and folded into `text` so `bm25` and `dense` see them too. BQL gets **precise** field
access, not **exclusive** access; otherwise the comparison would be confounded. The `## headings`
stay **inside** `text` the same way (byte-identical to the flat arm), so bm25 and dense match
section titles as well. BQL's `IN(section,·)` and `fetch` address them precisely.

## Notes
- The sectioning model never rewrites text. It only proposes boundary line numbers, and the split
  is deterministic, so the emitted body is the original text with `## headings` inserted at those
  boundaries. The result is reproducible from `sections.jsonl`.
- Docs with no frontmatter still get emitted (flat, with `title`/`author`/`date` empty and the full
  `text`). Docs the batch never sectioned get a single `(intro)` part. Nothing is ever lost.
- `browsecomp_plus_structured` is already registered (general domain, `field_profile=browsecomp`)
  and loads the structured BQL skill (title/author/date/section/body) on its own.
