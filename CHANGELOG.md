# Changelog

## 0.3.0 (2026-09, tools, tasks, strategies)

The code now says what a tool is, what a task is and what a strategy is. Nothing the model sees
changed: every paper condition renders the same system prompt (`tests/test_prompt_fidelity.py`
pins all 24), every tool gives the same observations as the workspace it replaces
(`tests/test_tool_parity.py`) and a stub episode is identical end to end
(`tests/test_episode_parity.py`). Old import paths keep working for this release.

### Structure
- `agent_search/tools/`: one folder per atomic tool (`search_bm25`, `search_dense`, `search_hybrid`,
  `search_bql`, `search_indri`, `search_dedup`, `search_bm25_dci`, `visit`, `fetch`, `fetch_code`,
  `get_document`, `bash`, `read`, `grep`), each with its declaration, its code and its manual.
- `agent_search/tasks/`: one folder per task (`research`, `research_dedup`, `codefix`,
  `codefix_patch`) with the prompt template and the answer protocol.
- `agent_search/strategies/`: one file per family; a strategy is tools with options under the
  names the paper prompts used. `conditions.py` pairs a task with a strategy; `paper.py` keeps the
  paper's condition names. New strategies: one-shot RAG (`rag_bm25`, `rag_dense`, `rag_hybrid`,
  from `scripts/oneshot_rag.py`), `search_visit_snippets`, `autoread_hybrid`, `sieve_plain`,
  `sieve_v2`, `sieve_visit*`, `indri_plain`, `indri_visit`.
- `agent_search/evaluation/agent_runner.py`: `ConditionAgent` runs a condition; `ProcedureAgent`
  runs a loop-free strategy. Every condition is the retriever `agent_<name>`.
- `agent_search/retrievers/`: `structural/` flattened into `bql/`, `indri/`, `lucene/` and
  `backend.py`; dense retrievers are a base class and one file per encoder family.
- `agent_search/legacy/`: the pre-0.3 workspaces, `AgentRetriever` and the YAML prompt registry,
  reachable under their old names, removed next release.
- The code task's grounding guards live with the task (`tasks/codefix/guards.py`); the corpus
  vocabulary helper moved to `corpus/grounding.py`.

### Extending
- A tool is a `Tool` subclass; a strategy is `register_strategy(Strategy(...))`; a condition is
  `condition(name, task, strategy)`. `docs/EXTENDING.md` and `examples/plugin_strategy.py` show
  the whole path. `register_tool` / `register_toolset` / `register_condition` /
  `register_workspace` still work through `agent_search.legacy`.

### Verified on the cluster (2026-09-11)
- The smoke suite (`scripts/slurm/smoke_suite.sbatch`): the full test suite, every strategy on
  the fixtures with the stub policy, and the dense episode parity (nine conditions).
- Replay of the 20 recorded ITER trajectories on the InfoSeek-Eval sample
  (`scripts/replay_check.py`): the old code and the new code give the same observation on all
  486 steps. Against the record, with the fp32 index the run used, every step reproduces except
  the forced final answer of the episodes that used all 40 steps (the model was called there).
- Tongyi samples through the new runner, 20 questions each, judged by gpt-4o-mini: Sieve
  (`sieve_bm25`) 25% and Search-Visit with Lucene BM25 25% on the BrowseComp-Plus chunk sample;
  one-shot RAG (`rag_bm25`) 70% and ITER's loop (`dedup_dense`, released ITER-0.6B) 50% on the
  InfoSeek-Eval sample. The ITER run scored 70% before the restructure; that run used an fp32
  embedding index and this one a bfloat16 rebuild, and the replay above shows the tools
  unchanged, so the gap is index precision and sampling on 20 questions, not the code.
- ITER, held out: 99 InfoSeek training trajectories to a bf16 checkpoint; on 20 unseen
  InfoSeek-Eval questions the trained retriever behind the agent scored 65% judged against 60%
  for its base (`docs/ITER.md`).

## 0.2.0 (2026-09, library release)

Breaking changes are marked **[breaking]**.

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

### Experiment files
- `skimsearchagent run FILE.yaml`: one YAML file fully determines one setting, with a strict
  schema, every knob spelled out, and the whole file recorded in `config.json`. There are also
  `validate` and `template [paper]` commands, plus complete files for the smoke, quick and paper
  settings under `configs/`. The old descriptive eval manifests are gone.

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

### Extensibility
- Contracts in `agent_search.core.interfaces`: `Retriever`, `Model`, `Policy`, `Workspace`.
- A programmatic API: `agent_search.research()` and `build_agent()`.
- Plugin registration through `register_tool`, `register_toolset`, `register_condition` and
  `register_workspace`, with discovery via the `skimsearchagent.plugins` entry-point group and
  `SKIMSEARCHAGENT_PLUGINS`.
- New documentation: `docs/ARCHITECTURE.md`, `docs/EXTENDING.md`, `docs/RUN_RECORD.md`,
  `docs/CONFIGURATION.md`, `docs/TRAINING.md` and `CONTRIBUTING.md`. The README was rewritten
  around the document fixture (`doc_fixture`).

### Code-localization arm wired
- `codefix`, `codefix_grep` and `codefix_patch` are registered conditions again: `tools.yaml`
  declares the code `search` and `grep` tools, and `conditions.yaml` binds them to the `taskfix`
  templates. `dataset=code_fixture` runs them with the scripted policy. The library's default read
  caps are now the paper's 12,000 tokens.
- `scripts/slurm/` holds site-neutral launchers for the smoke suite, index builds, and
  serve-and-run experiments (vLLM inside the job, never on a login node).

### Training from the run record
- `agent_search.training`: build ITER-style retriever triples from `rows.jsonl`
  (`skimsearchagent-build-triples`; oracle / answer / LLM-judge labellers), train with the paper's
  recipe (`skimsearchagent-train-retriever`; FlagEmbedding 1.3.5 plus ITER's patch, shipped),
  and plug the checkpoint back in as the one dense model with the same query instruction and
  history-conditioned query (`retrieval.dense_query_style`). Rows now record `hit_ids` and
  `read_ids` per step. `scripts/slurm/train_retriever.sbatch`; docs/TRAINING.md.

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

### Documentation split by paper
- `docs/ITER.md` holds everything about the ITER paper (tools, backbones, retrievers, datasets,
  samples, verified runs); `docs/SIEVE.md` and `docs/REPRODUCING.md` hold the Sieve paper; the
  README only points at them.

### Fixes from the review pass
- `section.key=value` overrides on `skimsearchagent run FILE` are recorded in `config.json`
  (`experiment` is the setting that ran, `experiment_overrides` lists them, `experiment_file_sha256`
  is the file as written). Untyped overrides such as `dataset.limit=1` are stored as numbers.
- The Python entry point `run_config()` refuses a different setting in an existing run
  directory, like the command line does.
- The fail-fast rule counts rows already on disk as successes, so a resumed run is not stopped
  by a few transient errors.
- The on-disk document store keeps one file handle per thread; concurrent episodes used to
  interleave reads on a shared handle and could return the wrong document.
- Every in-memory engine (BQL, Indri, local BM25, grep) refuses an on-disk corpus with a
  message naming the prebuilt-index knob, and no workspace loads such a corpus per episode.
- A note-conditioned query style (`i4`, `i5`, `i7`, `mem`) sees the model's note on the last
  read at the moment the next search is encoded, exactly as the triple builder renders it. The
  serving note's `query_max_len` is applied when queries are encoded (documents keep their own
  length). `tiktoken` is part of the `retrieval` and `serve` extras so query truncation is the
  same at training and serving time.
- An unknown `DENSE_POOLING` value is an error instead of silently becoming last-token pooling;
  a persisted vector index records the model that built it and refuses to serve another one;
  corpus fingerprints include document metadata (BQL indexes author, date and infobox fields).
- The subset retriever evaluation keeps weak negatives in its corpus; the title shortener in
  the `docs` query style counts tokens; `is_patched` checks every file the FlagEmbedding patch
  touches; a float `rank` in a tool call gives a clean error; the code-fix guard judges each
  fetched block on its own.
- The Tongyi ITER files set `agent.ctx_window` to the served 98,304-token window so the loop
  stops before the server rejects an over-long prompt.
- Precision follows the checkpoint: the trainer records `dtype` in the serving note, the encoder
  loads with it (bf16-trained models are served in bf16), caches and index metadata carry it,
  `retrieval.dense_dtype` overrides it. The ITER files use bfloat16, as ITER did.

### Removed
- Stale tests and configuration for conditions that no longer exist, the SLURM shell test, and the
  internal planning documents.

## 0.1.0

Paper release (Search, Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents).
