# Changelog

## 0.2.0 (2026-09, library release)

Breaking changes are marked **[breaking]**.

### ITER integration (search strategy, datasets, on-disk corpora, evaluation)
- `strategy=dedup_bm25` / `dedup_dense`: ITER's tool setup (over-fetched search that hides
  documents surfaced earlier, "Already-seen" list, `get_document` by id) with its task template.
- Datasets in ITER's layout (`topics.tsv`, TREC `qrels.txt`, `corpus.jsonl`): `infoseek_eval`,
  `infoseek_train`, `browsecomp_plus_chunks`. Answer-only sets (no qrels) run end to end and are
  scored on the answer; rank metrics are omitted for those rows instead of skipping them.
- Corpora above 1 GiB are served from disk (`agent_search.corpus.docstore`), also by the triple
  builder and the retriever evaluation, and searched through prebuilt indexes: `retrieval.dense_index` (this library's cache or ITER's `index.faiss` +
  `index.lookup.pkl`), `retrieval.bm25_index` (Lucene), `retrieval.ann_ef_search`.
- Released decoder checkpoints without a sentence-transformers config (ITER, LRAT) load with
  last-token pooling automatically, whether given as a local directory or a hub id (the cached
  snapshot is inspected); `retrieval.dense_pooling` overrides.
- Prebuilt FAISS indexes are read into RAM by default (`AGENT_SEARCH_FAISS_MMAP=1` memory-maps
  instead): memory-mapping a 46 GB HNSW index from a network filesystem cost ~18 s per search.
- `skimsearchagent-eval-retriever`: recall and novelty of a retriever on trajectory triples,
  optionally over a subset corpus.
- `skimsearchagent-sample-dataset`: a small dataset cut from a big one (topics, gold documents,
  a BM25 pool, random chunks) in the topics layout; any `data/<name>/topics.tsv` folder is
  discovered as a dataset without code. `scripts/slurm/iter_sample.sbatch` indexes the samples
  with a retriever, serves the backbone and runs the ITER files on them.
- Experiment files are scoped to their strategy: `template [preset] [STRATEGY]` lists only the
  keys that strategy reads, `validate` requires only those and reports unused ones.
- `configs/iter/`: InfoSeek and BrowseComp-Plus settings with the released ITER retriever and
  Tongyi, trajectory generation, and the smoke pipeline; `scripts/slurm/iter_smoke_*.sbatch`.
- A run stops early (SetupError) after 3 consecutive errors before any success, so an
  unreachable model endpoint no longer burns the retry budget on every question
  (`AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS`).
- The lazily built shared engines behind `WorkspaceContext` (`bm25()`, `bql()`, `dense()`) are
  built under a per-retriever lock: with `--workers N` every concurrent episode used to load the
  same index at once (two copies of a 49 GB index in the first ITER run).
- Every SLURM launcher sources `_common.sh` through `SLURM_SUBMIT_DIR`; the old `dirname $0`
  form broke under sbatch, which copies the script to a spool directory.

### Packaging and entry points
- **[breaking]** `evaluation` now lives inside the package as `agent_search.evaluation`, so imports
  and `python -m` paths change: `python -m evaluation.run_eval` becomes
  `python -m agent_search.evaluation.run_eval`.
- Prompt profiles, tool declarations and skill manuals ship as package data, so a wheel install
  works.
- Console scripts: `skimsearchagent` (the key=value launcher), `skimsearchagent-eval`,
  `skimsearchagent-build-indexes`, `skimsearchagent-judge`. `run.py` is now a thin wrapper.
- `analysis/` is no longer installed; the paper tooling stays in the repository.
- New extras: `agents` (the OpenAI Agents SDK, kept as a separate lane because it moves `openai` to
  3.x) and `tiktoken`, which was added to `api`, `eval` and `dev`.
- Real author metadata, the Apache-2.0 copyright line filled in, and a CI workflow that runs the
  tests and checks the wheel.

### Length limits are tokens only
- **[breaking]** `AgentPolicy(ctx_chars=...)` is now `AgentPolicy(ctx_tokens=...)`
  (`AGENT_CTX_TOKENS`, default 115,000 model tokens). `scripts/force_answer_backfill.py --ctx-chars`
  is now `--ctx-tokens`.
- **[breaking]** `BASH_MAX_BYTES` is now `BASH_MAX_TOKENS` and `READ_MAX_LINE_CHARS` is now
  `READ_MAX_LINE_TOKENS` (both DCI workspaces), plus `GREP_LINE_TOKENS` and
  `CLOSER_EVIDENCE_{ARG,OBS}_TOKENS` (SDK driver).
- Every character slice left on agent-facing or recorded text is gone. Trajectory observations are
  stored in full, retrieved-detail snippets and titles are untouched, the dense encoder no longer
  pre-clips text, and BQL hit snippets and the demo stream are token-capped instead.
- `agent_search.core.tokens` is the one ruler: whitespace tokens for read caps, tiktoken
  `o200k_base` for measurement with a whitespace fallback, and never characters ÷ 4.

### Harness correctness
- Document-research runs report rank metrics. Workspaces expose `surfaced` (an `OrderedSeen` in
  first-seen order); before this, hit@k, recall@k and nDCG were always 0.
- A missing dense cache or Lucene index aborts the run before the first episode (`SetupError`)
  instead of erroring every instance or spending the step budget on tool errors. A run that scored
  nothing exits non-zero.
- Run identity: `config.json` records `package_version`, `token_ruler` and every knob, and resuming
  a different experiment into the same directory is refused (`--allow-config-drift` overrides it).
- `--seeds a,b,c` writes one `seed=N` directory per seed. Previously only the first seed ran.
- `results.json` is a summary and no longer carries a copy of `rows.jsonl`.
- `timeout_rate` counts every budget stop: `max_steps`, `ctx_budget`, and the SDK's `max_turns`.
- Per-step token attribution is no longer shifted by the budget nudge, and forced-answer calls
  are counted in usage.
- Answer F1 follows the HotpotQA yes/no guard, and empty normalizations score 0.
- The LLM judge is idempotent (already-graded rows are skipped unless `--rejudge` is passed),
  rewrites atomically, and records an unparseable reply as `judge_error` rather than as "wrong".
- Dataset loaders warn when queries get dropped because their gold documents are missing.
- Persistent caches (dense, BQL, Pyserini) are validated against a corpus content fingerprint.
- BQL accepts Unicode terms like `café`, `Zürich` and `C++`; the query surface no longer drops
  non-ASCII characters.
- Document titles are indexed once. They used to be counted twice in BM25, dense and BQL.
- Dense-only fusion no longer drops filter-passing candidates that are missing from the dense index.
- Sieve's 0-hit fallback ranks with the SAME model over the SAME index as the exact path (RRF of
  BM25 and dense for `sieve`, dense-only for `sieve_dense`). It used to fall back to BM25 whatever
  the arm's ranker was. Every dense arm uses the run's `dense_model`.
- Model clients retry transient errors with backoff and carry explicit timeouts.
- Importing the package no longer mutates the process environment or the working directory.
- `--retriever agent` defaults to `research_snip`; the previous default no longer exists.

### Training from the run record
- `agent_search.training`: build ITER-style retriever triples from `rows.jsonl`
  (`skimsearchagent-build-triples`; oracle / answer / LLM-judge labellers), train with the paper's
  recipe (`skimsearchagent-train-retriever`; FlagEmbedding 1.3.5 plus ITER's patch, shipped),
  and plug the checkpoint back in as the one dense model with the same query instruction and
  history-conditioned query (`retrieval.dense_query_style`). Rows now record `hit_ids` and
  `read_ids` per step. `scripts/slurm/train_retriever.sbatch`; docs/TRAINING.md.

### Code-localization arm wired
- `codefix`, `codefix_grep` and `codefix_patch` are registered conditions again: `tools.yaml`
  declares the code `search` and `grep` tools, and `conditions.yaml` binds them to the `taskfix`
  templates. `dataset=code_fixture` runs them with the scripted policy. The library's default read
  caps are now the paper's 12,000 tokens.
- `scripts/slurm/` holds site-neutral launchers for the smoke suite, index builds, and
  serve-and-run experiments (vLLM inside the job, never on a login node).

### Experiment files
- `skimsearchagent run FILE.yaml`: one YAML file fully determines one setting, with a strict
  schema, every knob spelled out, and the whole file recorded in `config.json`. There are also
  `validate` and `template [paper]` commands, plus complete files for the smoke, quick and paper
  settings under `configs/`. The old descriptive `configs/eval_*.yaml` manifests are gone.

### Extensibility
- Contracts in `agent_search.core.interfaces`: `Retriever`, `Model`, `Policy`, `Workspace`.
- A programmatic API: `agent_search.research()` and `build_agent()`.
- Plugin registration through `register_tool`, `register_toolset`, `register_condition` and
  `register_workspace`, with discovery via the `skimsearchagent.plugins` entry-point group and
  `SKIMSEARCHAGENT_PLUGINS`.
- New documentation: `docs/ARCHITECTURE.md`, `docs/EXTENDING.md`, `docs/RUN_RECORD.md`,
  `docs/CONFIGURATION.md`, `docs/TRAINING.md` and `CONTRIBUTING.md`. The README was rewritten
  around the document fixture (`doc_fixture`).

### Removed
- Stale tests and configuration for conditions that no longer exist, the SLURM shell test, and the
  internal planning documents under `docs/superpowers/`.

## 0.1.0

Paper release (Search, Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents).
