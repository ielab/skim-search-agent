# Configuration

Everything that changes what a run does or how it is measured is a setting you can see, set and
find again in the run's record. This page covers, in order:

1. experiment files, the normal way to run something;
2. how a setting travels from the file to the code;
3. what the run records;
4. the `key=value` launcher for one-off runs;
5. the reference tables: `run_eval` flags, index-building flags, judge flags, and every
   environment knob with its default;
6. the paper's base setting;
7. training files.

## 1. Experiment files

One YAML file is one complete setting. Run it, check it, or print a fresh one:

```bash
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml            # run it (this file names no model: scripted policy)
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml model.name=gpt-4o   # run it with a model
skimsearchagent validate configs/paper/hotpotqa_structured_sieve.yaml       # what it needs, what it will run
skimsearchagent template paper sieve > configs/mine.yaml                    # a complete file to edit
```

A file lists exactly the keys its strategy reads. A `search_visit` file has the BM25 listing depth
and the BM25 backend and nothing about dense models or Sieve. A `sieve` file has the BQL and dense
keys and no listing depths. `validate` tells you when a file is missing a key the strategy needs,
and names any key the strategy does not read. Strategies added by a plugin read every key.

The `paper` preset fills in the paper's base setting (100 steps, Pyserini BM25, Lucene BQL,
12,000-token reads); the `library` preset (the default) uses the library defaults.

The sections of a file:

| section | what it holds |
|---|---|
| top level | `schema`, `name`, `strategy` |
| `dataset` | `name`, `limit`, `corpus_limit`, `only_instances` |
| `model` | `name`, `policy`, `backend`, `api_base`, `tp`, `temperature`, `seed`, `seeds`, `driver`, `reasoning_effort`, `timeout_s`, `retry_attempts` |
| `agent` | `max_steps`, `prompt_profile`, `ctx_tokens`, `ctx_window`, `ctx_stop_frac` |
| `budgets` | every length budget: `snippet_tokens`, `max_visit_tokens`, `max_section_tokens`, `bash_max_tokens`, `read_max_line_tokens`, `grep_line_tokens`, `closer_evidence_*_tokens` |
| `listing` | how many results a search shows: `*_topk`, `hybrid_pool`, `dedup_pool_k` |
| `retrieval` | which engines and models: `bm25_backend`, `bm25_index`, `structured_backend`, `dense_model`, `dense_query_style`, `dense_query_instruction`, `dense_pooling`, `dense_dtype`, `dense_index`, the `bql_*`, `rrf_k`, `indri_*`, `lucene_mu`, `ann*` and `dense_device` knobs |
| `evaluation` | `level`, `k`, `workers`, `judge_model`, `judge_api_base`, `rejudge` |
| `output` | `runs_dir`, `results_dir`, `index_root`, `rebuild`, `allow_config_drift`, `repo_cache`, `allow_clone` |
| `env` | any other environment variable, exported as is |

Every key has a one-line comment in the template. The full schema is
`agent_search/experiment.py` (`SCHEMA`; `APPLIES` says which strategies read which key).

## 2. How a setting reaches the code

A file is turned into two things: flags for the harness (`agent_search.evaluation.run_eval`) and
environment variables the tool modules read. The `key=value` launcher produces the same two
things, so both forms run through one path.

Some tool modules read their environment knob when they are imported, not when they are called
(`SNIPPET_TOKENS` at the top of `agent_search/tools/budgets.py`, for example). The
launcher exports every knob before it imports the harness, so this is invisible when you use
`skimsearchagent`. If you call `python -m agent_search.evaluation.run_eval` from your own script,
export the variables first. The tables below mark these knobs with "before import".

## 3. What a run records

Every run directory holds `config.json`, `rows.jsonl` and `results.json`
(see [RUN_RECORD.md](RUN_RECORD.md)). `config.json` has the experiment file's path and hash, the
setting that actually ran (the file with any `section.key=value` overrides applied, and the
overrides themselves), every harness flag, the composed prompt and its hash, the installed
package version, the git revision, the token ruler, and a snapshot of every environment knob.
Secrets (`OPENAI_API_KEY`, `GEMINI_API_KEY`) and infrastructure settings (server URLs, devices,
cache paths) are left out because they don't change results.

A subset of that record is the run's identity: dataset, retriever, model, dense model, policy,
backend, endpoint, step budget, temperature, seed, level, cutoffs, corpus limit, prompt profile
and prompt hash, and the whole knob snapshot. Rerunning the same setting into the same directory
resumes it. Rerunning a different setting into the same directory is refused, unless you pass
`output.allow_config_drift: true` (a warning is still printed). Keys that only change how much of
the same experiment runs (`limit`, `only_instances`, `workers`) are not part of the identity.

## 4. The `key=value` launcher

For a one-off run, every knob is also a command-line argument:

```bash
skimsearchagent dataset=doc_fixture strategy=sieve_bm25
skimsearchagent dataset=hotpotqa_structured strategy=search_visit model=gpt-4o-mini limit=20
skimsearchagent dataset=browsecomp_plus_structured_full strategy=sieve \
  model=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B backend=api api_base=http://localhost:8000/v1
```

| key | becomes |
|---|---|
| `dataset` | `--dataset` (default `doc_fixture`) |
| `strategy` | `--retriever`, through the friendly name (default `sieve_bm25`) |
| `model` | `--model`, and `policy=llm` unless you set `policy` yourself; without a model the scripted `stub` policy runs |
| `runs_dir` | `--runs-dir` (default `runs/quick`) |
| a boolean flag (`rebuild`, `allow_clone`, `rejudge`, `allow_config_drift`, `check_complete`) | the flag, when the value is `true`, `1`, `yes` or `on` |
| an environment knob (any name in the tables below) | exported as `NAME=value` before the harness is imported |
| anything else | passed through as `--key value` |

`skimsearchagent --help` prints the strategy table, the boolean flags and the knob names from the
same source as this page.

## 5. Reference

### Strategies

| strategy | registered retriever | needs a dense cache | what it is |
|---|---|---|---|
| `search_visit` | `agent_research_bm25` | no | search, then read whole documents; BM25 |
| `search_visit_dense` | `agent_research_dense` | yes | the same with a dense ranker |
| `search_visit_hybrid` | `agent_research_hybrid` | yes | the same with BM25 and dense fused by RRF |
| `autoread` | `agent_research_bm25_autoread` | no | every search returns full documents; BM25 |
| `autoread_dense` | `agent_research_dense_autoread` | yes | the same with a dense ranker |
| `dci` | `agent_research_dci` | no | shell commands over exported files, no retriever |
| `bounded_dci` | `agent_research_bm25_dci` | no | the same, scoped to what BM25 surfaced |
| `search_fetch` | `agent_research_bm25_fetch_snip` | no | result cards with snippets, then named sections; BM25 |
| `search_fetch_dense` | `agent_research_dense_fetch` | yes | the same with a dense ranker |
| `search_fetch_hybrid` | `agent_research_hybrid_fetch_snip` | yes | the same with RRF fusion |
| `sieve` | `agent_research_bql_dense_snip` | yes | the paper's method: BQL filter, BM25 and dense fused, cards, sections |
| `sieve_bm25` | `agent_research_snip` | no | Sieve with BM25 ranking only (the default strategy) |
| `sieve_dense` | `agent_research_bql_donly_snip` | yes | Sieve with dense ranking only |
| `sieve_nosnip` | `agent_research_bql_dense_fetch` | yes | Sieve without listing snippets |
| `indri` | `agent_research_indri_snip` | no | Indri query language, cards and sections |
| `dedup_bm25` | `agent_research_dedup_bm25` | no | ITER's tools: search that hides documents shown before, `get_document` by id; BM25 |
| `dedup_dense` | `agent_research_dedup_dense` | yes | the same with the run's dense model |
| `codefix`, `codefix_grep`, `codefix_patch` | `agent_codefix*` | no | code localization over a repository |
| `bm25`, `bm25_lucene` | `bm25_local`, `bm25_pyserini` | no | rank once, no agent |

"Needs a dense cache" means the corpus must be embedded once
(`skimsearchagent-build-indexes --dataset <name> --retriever dense --model <model>`) or a
prebuilt index named (`retrieval.dense_index`). Nothing is embedded during a run.

### `run_eval` flags

`python -m agent_search.evaluation.run_eval` is what every experiment file turns into.

| flag | default | what it does |
|---|---|---|
| `--dataset` | `fixture` | a registered dataset |
| `--retriever` | `bm25_local` | a registered retriever (the strategy's registered name) |
| `--model` | none | the agent's model, or the dense model for a retrieval-only run |
| `--dense-model` | by domain | the embedding model every dense arm uses |
| `--domain` | by dataset | `code` or `general`; picks the prompts and the default embedder |
| `--policy` | `stub` | `stub` (scripted, no model) or `llm` |
| `--max-steps` | 50 | tool calls per question |
| `--temperature` | 0.6 | sampling temperature |
| `--seed` / `--seeds` | `42` | one seed, or a comma list; each seed gets its own directory |
| `--prompt-profile` | none | a different prompt profile |
| `--backend` | `vllm` | `vllm` (in-process) or `api` (an OpenAI-compatible server at `--api-base`) |
| `--api-base` | `http://localhost:8000/v1` | the server for `--backend api` |
| `--tp` | 1 | GPUs for in-process vLLM |
| `--workers` | 1 | questions run concurrently |
| `--judge-model` | none | grade the answers after the run with this model |
| `--judge-api-base` | none | the judge's endpoint (default: `--api-base`) |
| `--rejudge` | off | grade rows that already have a verdict |
| `--allow-config-drift` | off | resume into a directory that holds a different setting |
| `--level` | `function` | `function` or `file` (code datasets) |
| `--k` | `1 3 5 10` | rank-metric cutoffs |
| `--limit` | none | at most this many questions |
| `--only-instances` | none | a file of instance ids to run (used for sharding) |
| `--corpus-limit` | none | at most this many documents of a shared corpus |
| `--index-root` | `indexes` | where persisted indexes live |
| `--rebuild` | off | rebuild indexes |
| `--runs-dir` | `runs` | where run directories go |
| `--results-dir` | none | the exact run directory |
| `--repo-cache` | `data/repos` | cloned repositories for code datasets |
| `--allow-clone` | off | clone missing repositories during the run (needs internet) |
| `--check-complete` | off | exit 0 if the run is finished, 3 if questions are pending; runs nothing |
| `--experiment-file` | none | the experiment file this invocation came from; `skimsearchagent run` sets it so the record carries the whole setting |
| `--experiment-override` | none | one `section.key=value` override applied to that file (repeatable); set by `skimsearchagent run` and recorded |

### `build_indexes` flags

`skimsearchagent-build-indexes` builds the persisted indexes a run needs, so the run never builds
them itself.

| flag | default | what it does |
|---|---|---|
| `--dataset` | `swebench_verified` | a registered dataset |
| `--retriever` | `bm25_pyserini` | which index: `bm25_pyserini`, `dense`, `search_bql`, `search_indri`, `search_lucene` |
| `--model` | none | the dense model (with `--retriever dense`) |
| `--index-root` | `indexes` | where indexes go |
| `--repo-cache` | `data/repos` | cloned repositories for code datasets |
| `--limit`, `--corpus-limit` | none | build for a subset |
| `--rebuild` | off | rebuild an existing index |
| `--shard`, `--nshards` | 0, 1 | split the work across a SLURM array |

It exits with 1 if any corpus failed to build.

### `judge` flags

`skimsearchagent-judge` grades a finished run's answers with the BrowseComp judge prompt and
writes `judge_summary.json` next to the rows. Rows that already have a verdict are skipped.

| flag | default | what it does |
|---|---|---|
| `--results-dir` | required | the run directory |
| `--judge-model` | `gpt-4o-mini` | the grader |
| `--judge-api-base` | none | an OpenAI-compatible endpoint for the grader |
| `--dataset` | none | where to load the questions from if the rows lack them |

### Environment knobs

Every length budget is a token count. The budgets below are counted on the library's whitespace
ruler (`agent_search/core/tokens.py`), except the context budgets `AGENT_CTX_TOKENS` and
`AGENT_CTX_WINDOW`, which are model tokens. There are no character limits.

"Before import" marks a knob the module reads when it loads; the `skimsearchagent` launcher
handles that for you.

#### Length budgets

| knob | default | what it does | read by |
|---|---|---|---|
| `SNIPPET_TOKENS` | 32 | width of a result card's snippet, and of the opening excerpt in search-visit listings (before import) | `agent_search/tools/budgets.py` |
| `MAX_VISIT_TOKENS` | 12000 | how much of a whole document a read returns (`visit`, `get_document`, autoread) (before import) | `agent_search/tools/budgets.py` |
| `MAX_SECTION_TOKENS` | same as `MAX_VISIT_TOKENS` | how much of a section a `fetch` returns (before import) | `agent_search/tools/budgets.py` |
| `BASH_MAX_TOKENS` | 12000 | the tail of a DCI shell command's output that is kept (before import) | `agent_search/tools/bash/tool.py` |
| `READ_MAX_LINE_TOKENS` | 400 | per-line cap in the DCI `read` tool (before import) | `agent_search/tools/read/tool.py` |
| `GREP_LINE_TOKENS` | 24 | per-line cap on a `grep` hit (before import) | `agent_search/tools/grep/tool.py` |
| `CLOSER_EVIDENCE_ARG_TOKENS` | 32 | how much of a tool call's arguments the forced-answer closer shows (before import) | `agent_search/agent/sdk_driver.py` |
| `CLOSER_EVIDENCE_OBS_TOKENS` | 160 | how much of each observation the forced-answer closer shows (before import) | `agent_search/agent/sdk_driver.py` |
| `AGENT_CTX_TOKENS` | 115000 | the model-token budget the agent keeps its context under | `agent_search/agent/policies.py` |
| `AGENT_CTX_WINDOW` | 131072 | the model's context window | `agent_search/agent/loop.py` |
| `AGENT_CTX_STOP_FRAC` | 0.9 | stop and answer when the context reaches this share of the window | `agent_search/agent/loop.py` |

#### Listing depths

| knob | default | what it does | read by |
|---|---|---|---|
| `BM25_VISIT_TOPK` | 5 | results per search, `search_visit` (before import) | `agent_search/tools/budgets.py` |
| `DENSE_VISIT_TOPK` | 5 | results per search, `search_visit_dense` (before import) | `agent_search/tools/budgets.py` |
| `HYBRID_VISIT_TOPK` | 5 | results per search after fusion, `search_visit_hybrid` (before import) | `agent_search/tools/budgets.py` |
| `BM25_FETCH_TOPK` | 10 | results per search, `search_fetch` (before import) | `agent_search/tools/budgets.py` |
| `DENSE_FETCH_TOPK` | 10 | results per search, `search_fetch_dense` (before import) | `agent_search/tools/budgets.py` |
| `HYBRID_FETCH_TOPK` | 10 | results per search after fusion, `search_fetch_hybrid` (before import) | `agent_search/tools/budgets.py` |
| `HYBRID_POOL` | 100 | how deep each ranker is queried before RRF fusion (before import) | `agent_search/tools/budgets.py` |
| `AUTOREAD_TOPK` | 5 | documents rendered in full per search, `autoread*` (before import) | `agent_search/tools/budgets.py` |
| `BM25_DCI_TOPK` | 10 | documents staged per search, `bounded_dci` (before import) | `agent_search/tools/search_bm25_dci/tool.py` |
| `DEDUP_TOPK` | 10 | results per search, `dedup_*` (before import) | `agent_search/tools/search_dedup/tool.py` |
| `DEDUP_POOL_K` | 100 | how many candidates a dedup search draws before dropping documents shown earlier (before import) | `agent_search/tools/search_dedup/tool.py` |

#### Retrieval and method switches

| knob | default | what it does | read by |
|---|---|---|---|
| `BQL_SOFT_FALLBACK` | 1 | when a Boolean query matches nothing, rank the corpus with the arm's own model instead of returning nothing; `0` is the strict-Boolean ablation | `agent_search/tools/search_bql/tool.py` |
| `BQL_SOFT_POOL` | 100 | how many candidates that fallback ranks (before import) | `agent_search/retrievers/bql/executor.py` |
| `BQL_DATE_RANGE` | 1 | allow `date[YYYY..YYYY]` in BQL | `agent_search/retrievers/bql/surface.py` |
| `BQL_DENSE` | 0 | attach the dense model to the BM25-only Sieve arms too | `agent_search/retrievers/bql/dense_fuse.py` |
| `BQL_DENSE_RRF_K` | 60 | RRF constant for Sieve's BM25 and dense fusion (before import) | `agent_search/retrievers/bql/dense_fuse.py` |
| `RRF_K` | 60 | RRF constant for the hybrid baselines (before import) | `agent_search/tools/budgets.py` |
| `AGENT_SEARCH_BQL_PREFILTER_MIN` | 5000 | corpus size above which BQL narrows a scan with an inverted index first; speed only (before import) | `agent_search/retrievers/bql/executor.py` |
| `INDRI_DENSE` | 0 | attach the dense model to the Indri arm | `agent_search/retrievers/engines.py` |
| `INDRI_DENSE_W` | 0.35 | weight of the dense score in Indri's ranking | `agent_search/retrievers/indri/model.py` |
| `INDRI_DENSE_EXPAND_K` | 50 | dense neighbours added to Indri's candidate pool | `agent_search/retrievers/indri/model.py` |
| `INDRI_MU` | 2500 | Dirichlet smoothing for Indri (before import) | `agent_search/retrievers/indri/model.py` |
| `LUCENE_MU` | `INDRI_MU` | the same for the Lucene structural engine | `agent_search/retrievers/lucene/engine.py` |
| `INDRI_POOL_CAP` | 5000 | Indri's candidate pool size (before import) | `agent_search/retrievers/indri/model.py` |
| `INDRI_RESCORE_M` | 300 | how many candidates Indri rescores in its second stage | `agent_search/retrievers/indri/model.py` |
| `DENSE_QUERY_STYLE` | `plain` | how the dense query is written from the agent's history (`plain`, `mem`, `docs`, `i1` to `i7`); must match the trained retriever | `agent_search/training/history.py` |
| `DENSE_QUERY_INSTRUCTION` | unset | the query instruction prefix; unset means the checkpoint's serving note or the built-in table | `agent_search/retrievers/dense/base.py` |
| `DENSE_POOLING` | unset | `last_token`, `mean` or `cls` for a checkpoint without a sentence-transformers config; unset means the serving note, then a guess from the model type | `agent_search/retrievers/dense/base.py` |
| `DENSE_DTYPE` | unset | the precision the dense encoder runs in: `float32`, `float16` or `bfloat16`. Unset means the checkpoint's serving note (a model trained with bf16 is served in bf16), else float32. Caches and index metadata carry it. | `agent_search/retrievers/dense/base.py` |
| `DENSE_INDEX_PATH` | unset | a prebuilt vector index to serve instead of the per-corpus cache (this library's cache directory, or ITER's `index.faiss` plus `index.lookup.pkl`); required for an on-disk corpus | `agent_search/retrievers/dense/base.py` |
| `AGENT_SEARCH_ANN_EF_SEARCH` | 0 | HNSW `efSearch` for a prebuilt index; 0 keeps the built-in value | `agent_search/retrievers/dense/vector_index.py` |
| `AGENT_SEARCH_FAISS_MMAP` | unset | `1` memory-maps a prebuilt FAISS index instead of reading it into RAM | `agent_search/retrievers/dense/vector_index.py` |
| `BM25_INDEX_PATH` | unset | a prebuilt Lucene index for the pyserini backend; required for an on-disk corpus | `agent_search/retrievers/lexical/pyserini.py` |
| `AGENT_SEARCH_DOCSTORE` | unset | `1` serves a topics-layout corpus from disk whatever its size | `agent_search/evaluation/datasets/topics.py` |
| `AGENT_SEARCH_DOCSTORE_MIN_BYTES` | 1 GiB | corpus size above which the loader serves it from disk (before import) | `agent_search/evaluation/datasets/topics.py` |

#### Engines and models

| knob | default | what it does | read by |
|---|---|---|---|
| `STRUCTURED_BACKEND` | `python` | the structural engine behind BQL and Indri: `python` or `lucene` (needs a prebuilt `indexes/lucene_structured/<key>/` and Java 21) | `agent_search/retrievers/backend.py` |
| `BM25_BACKEND` | `local` | the BM25 engine: `local` (in memory, no dependencies) or `pyserini` (Lucene, persisted; the paper's) | `agent_search/retrievers/lexical/__init__.py` |
| `DENSE_MODEL` | `BAAI/bge-base-en-v1.5` | the dense model for general-domain runs when the file names none; code datasets keep `nomic-ai/CodeRankEmbed` | `agent_search/evaluation/datasets/base.py` |
| `AGENT_SEARCH_ANN` | `auto` | vector index type: `flat`, `hnsw`, `ivfpq`, or `auto` by corpus size | `agent_search/retrievers/dense/vector_index.py` |
| `AGENT_SEARCH_ANN_MIN` | 1,000,000 | corpus size at which `auto` picks HNSW | `agent_search/retrievers/dense/vector_index.py` |
| `AGENT_SEARCH_ANN_PQ_MIN` | 8,000,000 | corpus size at which `auto` picks IVF-PQ | `agent_search/retrievers/dense/vector_index.py` |
| `AGENT_SEARCH_FLAT_FAISS` | off | use FAISS for the exact `flat` search (faster, same results) | `agent_search/retrievers/dense/vector_index.py` |
| `AGENT_SEARCH_DENSE_DEVICE` | auto | the device the dense encoder runs on (`cpu` when a vLLM server owns the GPU) | `agent_search/retrievers/dense/base.py` |
| `SKIMSEARCHAGENT_PLUGINS` | unset | comma-separated modules imported before the registries are read, so they can register things | `agent_search/retrievers/registry.py` |
| `AGENT_SEARCH_DATA` | `data` | where datasets are read from (before import) | `agent_search/evaluation/datasets/base.py` |
| `AGENT_SEARCH_DCI_CACHE` | `$TMPDIR/agent_search_dci` | where the DCI arms export their flat text files | `agent_search/corpus/flat_export.py` |

#### Index building

| knob | default | what it does | read by |
|---|---|---|---|
| `BM25_PYSERINI_THREADS` | all cores | indexing threads for Pyserini | `agent_search/retrievers/lexical/pyserini.py` |
| `BM25_PYSERINI_STORE_RAW` | off | also store raw text and positions in the Lucene index (bigger, slower) | `agent_search/retrievers/lexical/pyserini.py` |
| `LUCENE_INDEX_RAM_MB` | 512 | RAM buffer for the structural Lucene index builder (before import) | `agent_search/retrievers/lucene/index_builder.py` |
| `LUCENE_INDEX_THREADS` | 1 | threads for the structural Lucene index builder (before import) | `agent_search/retrievers/lucene/index_builder.py` |

#### Model client, driver, judge

| knob | default | what it does | read by |
|---|---|---|---|
| `LLM_TIMEOUT_S` | 600 | HTTP timeout for every model call | `agent_search/models/openai_chat.py` |
| `LLM_RETRY_ATTEMPTS` | 5 | attempts per model call before giving up | `agent_search/models/retry.py` |
| `LLM_RETRY_BASE_S` | 1.0 | base delay of the retry backoff | `agent_search/models/retry.py` |
| `REASONING_EFFORT` | `low` | reasoning effort for OpenAI reasoning models | `agent_search/models/openai_reasoning.py` |
| `AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS` | 3 | stop a run after this many consecutive failed questions before any success (an unreachable endpoint, a broken index); `0` disables | `agent_search/evaluation/run_eval.py` |
| `AGENT_DRIVER` | auto | `loop` (text-parsed tool calls) or `sdk` (OpenAI Agents SDK, native tool calling, document arms only) | `agent_search/evaluation/agent_runner.py` |
| `AGENT_DEFAULT_CONDITION` | `research_snip` | the condition the bare `agent` alias means (before import) | `agent_search/strategies/conditions.py` |
| `LLM_JUDGE_MODEL` | `gpt-4o-mini` | the judge module's default grader (before import) | `agent_search/evaluation/llm_judge.py` |
| `VLLM_API_BASE` | `http://localhost:8000/v1` | the served endpoint the SDK driver falls back to | `agent_search/agent/sdk_driver.py` |

#### Secrets

`OPENAI_API_KEY` and `GEMINI_API_KEY` are read when a client is built (`agent_search/models/openai_chat.py`,
`agent_search/models/openai_reasoning.py`, `agent_search/models/gemini.py`,
`agent_search/evaluation/llm_judge.py`, `agent_search/agent/sdk_driver.py`) and never written to
any record. A local vLLM server needs no key.

## 6. The paper's base setting

`skimsearchagent template paper <strategy>` prints it. Compared with the library defaults, the
paper sets:

| setting | paper | library default |
|---|---|---|
| `agent.max_steps` | 100 | 50 |
| `budgets.max_visit_tokens`, `budgets.max_section_tokens` | 12000 | 12000 |
| `budgets.snippet_tokens` | 32 | 32 |
| results per search | 5 | 5 |
| `retrieval.bm25_backend` | `pyserini` | `local` |
| `retrieval.structured_backend` | `lucene` | `python` |
| `retrieval.bql_soft_fallback` | 1 | 1 (the strict-Boolean ablation sets 0) |
| `model.temperature`, `model.seed` | 0.6, 42 | 0.6, 42 |

The shipped files under `configs/paper/` are these settings for each paper cell. Trust the run's
`config.json` over the file you meant to run, and check the first rows before scaling a sweep.

## 7. Training files

Retriever training has its own one-file-per-run format: `skimsearchagent-train-retriever template`
prints every knob with ITER's defaults, `skimsearchagent-train-retriever run train.yaml` runs it.
See [TRAINING.md](TRAINING.md).
