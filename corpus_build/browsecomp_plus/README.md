# BrowseComp-Plus — structured corpus builder (frontmatter + LLM sections, no scraping)

BrowseComp-Plus ships ~100K docs OBFUSCATED (XOR'd with a canary). Each doc's de-obfuscated text
carries a YAML **frontmatter** block — `title` / `author` / `date` — followed by the body. Those
fields are already in the data, so this builder **does not fetch anything**: de-obfuscate, parse
the frontmatter, emit.

**Sections (built with a model — there is NO prebuilt sectioned corpus on the hub).** browsecomp's
raw web pages have no reliable section markup, so a cheap **`gpt-5.4-nano`** pass proposes section
**boundaries + headings** for each page, and a **deterministic** step splits the ORIGINAL body at
those boundaries into a **matched `(heading, text)` list**. The model only says *where* sections
start — it never rewrites text — so the corpus stays faithful and its `text` still matches the flat
twin byte-for-byte. It runs through the OpenAI **Batch API** (24h async, ~50% cheaper) with **many
docs per request**, so the whole corpus is one cheap overnight job (≈ **$9** for ~100K docs).

Gold is a label on the `docid`, so this never touches the qrels — the flat and structured arms
evaluate the **same** queries.

Run from **inside this folder**; it's self-contained.

## Setup (staging node, needs internet + `OPENAI_API_KEY`)
```bash
pip install datasets openai     # + tqdm for progress bars (optional)
huggingface-cli login           # if prompted
export OPENAI_API_KEY=...        # for the sectionize submit/collect/validate stages
cd corpus_build/browsecomp_plus
```

## Sectioning pipeline (validate → prepare → submit → collect → corpus)
```bash
# 0. VALIDATE first — one cheap sync call on a few real docs; eyeball quality + project the cost.
#    ALWAYS run this before the paid batch so it can't spend money for nothing.
python build.py sectionize-validate --sample 8               # prints sections + a full-corpus cost estimate

# 1. PREPARE — hub docs -> SHARDED batch input files + a manifest (no API call). Shards roll over
#    at the API's per-file caps so every file is submittable.
python build.py sectionize-prepare --out-prefix batch_input --manifest manifest.jsonl
#    --chunk-size N     : docs per request (default 20 — a big chunk, not one-at-a-time / all)
#    --max-shard-mb X   : roll to a new shard past this (default 180; API cap ~200MB)
#    --max-requests N   : roll to a new shard past this (default 40000; API cap 50k)
#    --model M          : default gpt-5.4-nano

# 2. SUBMIT — upload every shard + create a batch per shard; ids saved to batch_ids.txt.
python build.py sectionize-submit --shards 'batch_input_*.jsonl' --save-id batch_ids.txt

# 3. COLLECT — after the batches finish (up to 24h), apply the boundaries -> sections.jsonl.
#    Idempotent; if any batch is still running it says so and exits (just re-run later).
python build.py sectionize-collect --batch-ids batch_ids.txt --manifest manifest.jsonl --out sections.jsonl

# 4. CORPUS — assemble the flat+structured PAIR with matched sections.
python build.py corpus --sections sections.jsonl            # -> ../../data/browsecomp_plus_{flat,structured}/
python build.py corpus                                      # (no --sections -> frontmatter-only, section-less)
```
Other helpers: `python build.py diagnose --limit 2000` (title/author/date coverage),
`python build.py dump --out deobf.jsonl` (just de-obfuscate to inspect).

**The prompt.** Each request shows the model a chunk of **web pages** — each with its title / author
/ date, then its body as numbered lines — and asks it to segment the *body* into the sections a
reader would see on the rendered page (lead, then topical sections). Section line numbers refer to
body lines only, so metadata never becomes a section. Validated on real pages: e.g. the *Sanaa*
Wikipedia article splits into 34 correctly-named sections (History, Ottoman era, Geography,
Economy…), a 1-line page correctly stays a single intro.

> The prepare stage downloads the ~2.8 GB dataset once and writes ~1 GB of batch input — run it on
> a **staging node**, not a laptop. Point `--out-prefix`/`--manifest` at a scratch dir OUTSIDE the
> repo. Docs a batch never covers fall back to a frontmatter-only record (still emitted, just
> section-less); `corpus` prints the % that got matched sections.

## Output — a flat + structured PAIR (same docs/text/queries)
```
data/browsecomp_plus_structured/corpus.jsonl : {_id, title, author, date, sections, text}   ← scopeable fields + matched sections
data/browsecomp_plus_flat/corpus.jsonl       : {_id, title,                       text}    ← same text, no fields
```
`sections` is a matched, ordered list of `{heading, text}` parts (each `## section` kept WITH its
own body). The `text` is byte-identical across the pair — the structured arm just *also* exposes
the fields + section parts. Plus `queries.jsonl` + `qrels/test.tsv` in each, generated from the
dataset's own `gold_docs` (same in both arms). The pair is `browsecomp_plus_flat` vs
`browsecomp_plus_structured`.

> Note: the query/gold are read via field names `query_id` / `query` / `gold_docs` in `iter_rows`.
> If a `--limit 50` run prints "0 queries", those names differ on the hub — tell me and it's a
> one-line fix.

**Fairness.** `author`/`date` are emitted both as their own fields (units_from_documents carries
them into metadata, so BQL's `IN(author,·)`/`IN(date,·)` can scope them) AND folded into `text`,
so `bm25`/`dense` see them too — BQL gets **precise** field access, not **exclusive** access
(otherwise the comparison would be confounded). The `## headings` are likewise kept **inside**
`text` (byte-identical to the flat arm), so bm25/dense match section titles too; BQL's
`IN(section,·)` + `fetch` just address them precisely.

## Notes
- The sectioning model NEVER rewrites text — it only proposes boundary line numbers, and the split
  is deterministic, so the emitted body is the ORIGINAL text with `## headings` inserted at the
  boundaries. Faithful + reproducible from `sections.jsonl`.
- Docs with no frontmatter are still emitted (flat: `title`/`author`/`date` empty, full `text`);
  docs the batch never sectioned get a single `(intro)` part — **nothing is ever lost**.
- `browsecomp_plus_structured` is already registered (general, `field_profile=browsecomp`) and
  loads the structured BQL skill (title/author/date/section/body) automatically.
