# Configuration reference

Every knob that changes what SkimSearchAgent runs or how it measures the run: CLI flags,
environment variables, and the `skimsearchagent` key=value launcher that sits on top of both.
Ground truth for every entry below is the source file cited next to it. Re-verify against the
cited `file:line` if the code has moved since.

Contents: [How configuration flows](#how-configuration-flows) ·
[Launcher](#launcher-skimsearchagent--runpy) · [Strategies](#strategies) ·
[`run_eval` flags](#run_eval-flags) · [`build_indexes` flags](#build_indexes-flags) ·
[`llm_judge` flags](#llm_judge-flags) · [Environment knobs](#environment-knobs) ·
[Paper base configuration](#paper-base-configuration) · [Open issues](#open-issues)

## Experiment files

The primary way to run an experiment is a YAML **experiment file**: one file fully determines
one setting. `skimsearchagent template [library|paper] [STRATEGY]` prints a complete file for
that strategy with every key at its library default and a comment per key; the `paper` preset
is the paper's base configuration (100 steps, Pyserini BM25, Lucene BQL). The shipped files under
[`configs/`](../configs/) are all complete.

A file lists exactly the keys its strategy reads (`agent_search/experiment.py::APPLIES`): a
`search_visit` file has `listing.bm25_visit_topk` and `retrieval.bm25_backend` and nothing about
dense models or Sieve; a `sieve` file has the BQL and dense keys and no listing depths. "Complete"
means every key the strategy reads is present. A key the strategy does not read is accepted,
never required, and `validate` names it as unused. A plugin strategy the map does not know reads
everything.

```bash
skimsearchagent run FILE.yaml [section.key=value ...]   # run (overrides are recorded too)
skimsearchagent validate FILE.yaml                       # prerequisites + the exact flags/env + unused keys
skimsearchagent template [library|paper] [STRATEGY]      # e.g. template paper dedup_dense
```

Schema (`agent_search/experiment.py::SCHEMA`), one section per concern:

| section | keys | becomes |
|---|---|---|
| top level | `schema`, `name`, `strategy` | the registered retriever (`agent_<condition>` or a floor) |
| `dataset` | `name`, `limit`, `corpus_limit`, `only_instances` | `run_eval` flags |
| `model` | `name`, `policy`, `backend`, `api_base`, `tp`, `temperature`, `seed`, `seeds`, `driver`, `reasoning_effort`, `timeout_s`, `retry_attempts` | flags (`--model`, `--policy`, `--seed`/`--seeds`, ...) and env (`AGENT_DRIVER`, `LLM_*`) |
| `agent` | `max_steps`, `prompt_profile`, `ctx_tokens`, `ctx_window`, `ctx_stop_frac` | `--max-steps`, `--prompt-profile`, `AGENT_CTX_*` |
| `budgets` | `snippet_tokens`, `max_visit_tokens`, `max_section_tokens`, `bash_max_tokens`, `read_max_line_tokens`, `grep_line_tokens`, `closer_evidence_*_tokens` | the token-budget env knobs (there are no character limits) |
| `listing` | `*_topk`, `hybrid_pool`, `dedup_pool_k` | the listing-depth env knobs |
| `retrieval` | `bm25_backend`, `bm25_index`, `structured_backend`, `dense_model`, `dense_query_style`, `dense_query_instruction`, `dense_pooling`, `dense_index`, `bql_*`, `rrf_k`, `indri_*`, `lucene_mu`, `ann*`, `ann_ef_search`, `dense_device` | `--dense-model` and the engine env knobs |
| `evaluation` | `level`, `k`, `workers`, `judge_model`, `judge_api_base`, `rejudge` | flags |
| `output` | `runs_dir`, `results_dir`, `index_root`, `rebuild`, `allow_config_drift`, `repo_cache`, `allow_clone` | flags |
| `env` | any other environment variable, verbatim | exported as is |

A file is translated into exactly the flags and environment knobs the key=value launcher
produces, so both forms share one execution path. The file's path, content hash and parsed
content are stored in `config.json` (`experiment_file`, `experiment_sha256`, `experiment`).
`skimsearchagent validate` prints the translation and the artifacts the setting needs (a dense
cache, a Lucene index, Java, an API key).

## How configuration flows

There are three layers, from lowest to highest level:

1. **`agent_search.evaluation.run_eval` CLI flags** (`--dataset`, `--retriever`, `--max-steps`,
   ...), parsed once by `argparse` in `main()`, stored on `args`, and threaded through
   `RunConfig` (`agent_search/evaluation/config.py`). These are the only knobs that can vary
   the run's SHAPE (which dataset, which retriever, how many steps).
2. **Environment-variable knobs** (`SNIPPET_TOKENS`, `MAX_VISIT_TOKENS`, `STRUCTURED_BACKEND`,
   ...), read directly with `os.environ.get(...)` by the module that needs them, usually with
   an inline default. Most of these govern *within-condition* behavior (a read cap, a listing
   depth, which engine backs a retriever) rather than which condition runs at all.
3. **The `skimsearchagent` / `run.py` key=value launcher** (`agent_search/cli.py`), a wrapper over
   layers 1 and 2: it splits `key=value` pairs into `run_eval` flags and env-var exports, then
   invokes `run_eval.main()` in-process.

**Import-time caveat.** Several tool modules resolve their env knob at *module import time*
(`SNIPPET_TOKENS = int(os.environ.get("SNIPPET_TOKENS", "32"))` at the top of
`agent_search/agent/tools/doc_research.py`, for example) rather than at call time. Setting the
variable after `agent_search.evaluation.run_eval` has already been imported has no effect: the
module-level constant is already frozen. `agent_search/cli.py::main` therefore exports every env
knob into `os.environ` **before** importing `agent_search.evaluation.run_eval`
(see `agent_search/cli.py:125-132`). When invoking `run_eval` directly from a shell script,
`export` the variable before the `python -m agent_search.evaluation.run_eval ...` line. The
"Read at" column in every table below says `import` or `call`. `import` means the variable must
be set before the module loads.

**What `config.json` records.** `agent_search/evaluation/run_eval.py::_run_config_dict`
(line 295) builds the full per-run record: every parsed CLI arg (`vars(args)`), the resolved
`resolved_domain`, the composed prompt's `prompt_task`/`prompt_toolset`/`prompt_sha256`, the
installed `package_version`, the active `token_ruler` (`agent_search/core/tokens.py::ruler_name`),
`git_rev`, `started_at`, and `env_knobs`, a snapshot of the environment-variable knobs produced by
`_resolve_env_knobs()` (line 146). `_resolve_env_knobs` is additive-only provenance: it is called
once at run start and merged into `config.json`. It is never read back by episode code, so
recording a knob cannot change behavior. It excludes secrets
(`OPENAI_API_KEY`/`GEMINI_API_KEY`) and infra-only vars (server URLs, device selection, cache-dir
paths, parallelism toggles) that don't change results. See the "API keys" and "Cluster /
SLURM-only" groups below for what that excludes.

**Run identity.** `RUN_IDENTITY_KEYS` (`run_eval.py:340`) is the subset of `config.json` that
defines *which experiment* a run directory holds: `dataset`, `retriever`, `model`, `dense_model`,
`policy`, `backend`, `api_base`, `max_steps`, `temperature`, `seed`, `level`, `k`,
`corpus_limit`, `prompt_profile`, `prompt_task`, `prompt_toolset`, `prompt_sha256`,
`resolved_domain`, and `env_knobs` (the whole resolved dict, so ANY env knob drifting counts).
Keys that only change *how much* of the same experiment runs (`limit`, `only_instances`,
`workers`, `runs_dir`, progress flags, timestamps) are excluded.
`_check_run_identity` (line 353) refuses to resume into a `results_dir` whose recorded
`config.json` disagrees on any identity key; pass `--allow-config-drift` to overwrite it anyway
(a warning is still printed to stderr).

## Launcher (`skimsearchagent` / `run.py`)

```bash
skimsearchagent dataset=doc_fixture strategy=sieve_bm25
skimsearchagent dataset=hotpotqa_structured strategy=search_visit model=gpt-4o-mini limit=20
skimsearchagent dataset=browsecomp_plus_structured_full strategy=sieve \
  model=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B backend=api api_base=http://localhost:8000/v1
```

`python run.py ...` is a repository-local alias for the same entry point
(`agent_search/cli.py::main`). Parsing (`build_run_eval_argv`, `agent_search/cli.py:83`) is pure:
it never touches the environment, it returns `(argv, env)`. `main()` then exports `env` and calls
`run_eval.main()` with `argv`.

| key | maps to | notes |
|---|---|---|
| `dataset` | `--dataset` | default `doc_fixture` (`DEFAULT_DATASET`) |
| `strategy` | `--retriever` (via `resolve_strategy`) | friendly name or a raw registered retriever name; default `sieve_bm25` (`DEFAULT_STRATEGY`) |
| `model` | `--model`, and defaults `policy=llm` unless `policy=` is also given | omitting `model` runs the dependency-free scripted (`stub`) policy |
| `policy` | `--policy` | explicit `policy=` (no `model=`) is honored as-is; with neither, an `agent_*` retriever defaults to `stub` |
| `runs_dir` | `--runs-dir` | default `runs/quick` (`DEFAULT_RUNS_DIR`, NOT `run_eval`'s own default of `runs/`) |
| any name in `BOOL_FLAGS` | `--<name>` (flag-only) | `rebuild`, `allow_clone`, `check_complete`, `rejudge`, `allow_config_drift`; value must be `true`/`1`/`yes`/`on` (flag added) or `false`/`0`/`no`/`off` (flag omitted), anything else is a hard error |
| any name in `ENV_KNOBS` | exported as `NAME=value` (uppercased) before `run_eval` is imported | the full list is enumerated in `agent_search/cli.py:29-53` and reproduced group-by-group in [Environment knobs](#environment-knobs) below |
| anything else | forwarded verbatim as `--key value` (`_` → `-`) | lets any current or future `run_eval` flag pass through without a `cli.py` change |

`skimsearchagent --help` (or no args match) prints the module docstring plus the live strategy
table, `BOOL_FLAGS`, and `ENV_KNOBS`. It is generated from the same source tables this document
is built from, so the two cannot drift out of sync in content, only in explanation.

## Strategies

`agent_search/strategies.py::STRATEGIES` maps a friendly name to a registered retriever name
(an `agent_<condition>` agent strategy, or a bare retrieval-only floor). `DENSE_STRATEGIES`
marks which ones need a pre-built dense embedding cache
(`skimsearchagent-build-indexes --retriever dense`, or `build_indexes.py --retriever dense`)
before they can run. Without it, `AgentRetriever` raises an error at `index()` time. It never
live-encodes.

| strategy | retriever | dense cache required | what it is |
|---|---|---|---|
| `search_visit` | `agent_research_bm25` | no | Search→Visit (whole-document read), BM25 ranker |
| `search_visit_dense` | `agent_research_dense` | **yes** | Search→Visit, dense ranker |
| `search_visit_hybrid` | `agent_research_hybrid` | **yes** | Search→Visit, BM25+dense RRF fusion |
| `autoread` | `agent_research_bm25_autoread` | no | Search→AutoRead (search renders full text), BM25 |
| `autoread_dense` | `agent_research_dense_autoread` | **yes** | Search→AutoRead, dense |
| `dci` | `agent_research_dci` | no | Direct Corpus Interaction, bash/read, no retriever at all |
| `bounded_dci` | `agent_research_bm25_dci` | no | BM25-bounded DCI (bash/read scoped to what BM25 has surfaced) |
| `search_fetch` | `agent_research_bm25_fetch_snip` | no | Search→Fetch (named-section read), BM25 |
| `search_fetch_dense` | `agent_research_dense_fetch` | **yes** | Search→Fetch, dense |
| `search_fetch_hybrid` | `agent_research_hybrid_fetch_snip` | **yes** | Search→Fetch, BM25+dense RRF |
| `sieve` | `agent_research_bql_dense_snip` | **yes** | **Sieve, the paper's method**, BQL filter, BM25+dense fusion, snippets |
| `sieve_bm25` | `agent_research_snip` | no | Sieve, BQL filter + BM25-only ranking (default strategy) |
| `sieve_dense` | `agent_research_bql_donly_snip` | **yes** | Sieve, BQL filter + dense-only ranking |
| `sieve_nosnip` | `agent_research_bql_dense_fetch` | **yes** | Sieve without listing snippets (ablation) |
| `indri` | `agent_research_indri_snip` | no | Indri query-language executor, structural-retrieval control |
| `bm25` | `bm25_local` | no | retrieval-only floor, no agent loop |
| `bm25_lucene` | `bm25_pyserini` | no | retrieval-only floor, canonical Lucene BM25 |

## `run_eval` flags

`python -m agent_search.evaluation.run_eval [flags]` (`agent_search/evaluation/run_eval.py::main`,
line 923).

| flag | default | help |
|---|---|---|
| `--dataset` | `fixture` | dataset registry name (choices from `available_datasets()`) |
| `--retriever` | `bm25_local` | one of the registered retrievers (`agent_search.retrievers.registry.available()`); a comma list is allowed only with `--check-complete` |
| `--model` | `None` | dense model id, or agent LLM model id |
| `--dense-model` | `None` | embedding model for dense / semantic_search (default depends on domain: CodeRankEmbed for code, bge-base for general/documents) |
| `--domain` | `None` | prompt/embedder domain (`code`\|`general`); default `general` for document benchmarks (browsecomp_plus/hotpotqa/2wiki/musique), else `code` |
| `--policy` | `stub` | agent query-formulation policy (`stub`\|`llm`; `llm` = prompt-profile-driven LLM) |
| `--check-complete` | off | exit 0 if this run config is already fully scored (nothing to do), 3 if instances are pending; runs nothing |
| `--max-steps` | `50` | agent turn budget per instance (E6 ablation knob) |
| `--temperature` | `0.6` | LLM sampling temperature for agent policies |
| `--seed` | `None` | single sampling seed for a reproducible agent run (overrides `--seeds`) |
| `--seeds` | `"42"` | comma-separated agent seeds; each is a SEPARATE run dir (`seed=<N>`); ignored for deterministic floors |
| `--prompt-profile` | `None` | override the YAML prompt profile or legacy markdown prompt (operator-ablation variants) |
| `--backend` | `vllm` | LLM backend (`vllm`\|`api`): in-process vLLM on the GPU node, or `api` for an OpenAI-compatible server |
| `--tp` | `1` | tensor-parallel size (# GPUs) for in-process vLLM |
| `--workers` | `1` | concurrent instances (Tongyi-style); use with agent + `--backend api` so vLLM continuous-batches turns |
| `--repo-cache` | `data/repos` | pre-staged repo clones (populate with `scripts/prefetch_repos.py`) |
| `--allow-clone` | off | allow cloning missing repos at eval time (needs internet; off by default so a GPU node fails fast) |
| `--api-base` | `http://localhost:8000/v1` | endpoint for `--backend api` |
| `--judge-model` | `None` | if set, auto-grade doc answers after the run (BrowseComp-Plus LLM-judge protocol); auto-routed by name like `--model`; independent of `--model` |
| `--judge-api-base` | `None` | served endpoint for a non-OpenAI `--judge-model` (default: reuse `--api-base`) |
| `--rejudge` | off | re-grade rows that already carry a judge verdict (default: only ungraded rows are sent) |
| `--allow-config-drift` | off | resume into a run dir even if `config.json` describes a different experiment |
| `--level` | `function` | `function`\|`file` |
| `--k` | `[1, 3, 5, 10]` | report cutoffs (nargs `+`); covers LocAgent's file Acc@1/3/5 and function Acc@5/10 |
| `--limit` | `None` | cap #instances |
| `--only-instances` | `None` | path to a newline-separated instance-id file; applied AFTER `--limit`, so pass it WITHOUT `--limit` (sharding entry point for `scripts/shard_cell.sh`) |
| `--corpus-limit` | `None` | cap fixed-corpus documents for shared document datasets; ignored by code datasets |
| `--index-root` | `indexes` | where persistent indexes live |
| `--rebuild` | off | force index rebuild |
| `--runs-dir` | `runs` | root for saved results |
| `--results-dir` | `None` | override the exact results dir (default: `runs/<config>`) |

## `build_indexes` flags

`python -m agent_search.evaluation.build_indexes [flags]`
(`agent_search/evaluation/build_indexes.py::main`, line 186). Pre-builds the persistent indexes a
run of `--retriever` will need (see `prebuildable_for`, line 41), so a later `run_eval` reuses
them instead of building them during the run.

| flag | default | help |
|---|---|---|
| `--dataset` | `swebench_verified` | dataset registry name |
| `--index-root` | `indexes` | (no help string) |
| `--repo-cache` | `data/repos` | pre-staged repo clones |
| `--retriever` | `bm25_pyserini` | which persistent index to pre-build (`bm25_pyserini`\|`dense`\|`search_bql`\|`search_indri`\|`search_lucene`) |
| `--model` | `None` | dense model id (only for `--retriever dense`) |
| `--limit` | `None` | cap #instances |
| `--corpus-limit` | `None` | cap fixed-corpus documents for shared document datasets; ignored by code datasets |
| `--rebuild` | off | (no help string) |
| `--shard` | `0` | this shard id (0-based) |
| `--nshards` | `1` | total shards (SLURM array) |

Exit code 1 if any corpus failed to build, or if the shard built and cached nothing at all.
Callers such as `scripts/run.sh`'s Phase 0 abort rather than serve with no index.

## `llm_judge` flags

`python -m agent_search.evaluation.llm_judge [flags]`
(`agent_search/evaluation/llm_judge.py::main`, line 226). Post-hoc and optional. It grades a
finished run's `rows.jsonl` with the verbatim BrowseComp Appendix F judge prompt and writes
`judge_summary.json`; idempotent (already-graded rows are skipped unless `--rejudge` on the
`run_eval` side, or re-run with `force=True` programmatically).

| flag | default | help |
|---|---|---|
| `--results-dir` | *(required)* | a run dir containing `rows.jsonl` |
| `--judge-model` | `gpt-4o-mini` | grader model (cheap is fine) |
| `--judge-api-base` | `None` | OpenAI-compatible endpoint for the judge, e.g. a served vLLM on the cluster (default: the OpenAI API via `OPENAI_API_KEY`) |
| `--dataset` | `None` | load questions from this dataset for rows that lack a `question` field |

## Environment knobs

Every length budget below is measured in **whitespace tokens** (`agent_search.core.tokens.ws_tokens`
/ `cap_tokens`) except `AGENT_CTX_TOKENS` and `AGENT_CTX_WINDOW`, which are **model tokens**
(`count_tokens`/`truncate_tokens`, tiktoken `o200k_base` when installed). There is no
character-based limit anywhere in the library (`agent_search/core/tokens.py`'s module docstring
states this as a design invariant); the grep behind this document found none. See
[Open issues](#open-issues) if that ever changes.

"Read at" is `import` (module load; the variable must be exported before
`agent_search.evaluation.run_eval` is imported, see the import-time caveat above) or `call` (read
inside a function or method each time it runs; an `export` any time before that call takes effect,
including mid-process for a `--workers`-parallel run's *next* call).

### Length budgets (tokens)

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `SNIPPET_TOKENS` | 32 | import | Width of the query-biased listing snippet shown by Sieve's search results (`research_snip`/`_bql_*_snip`/`_fetch_snip` arms) AND the retrieve-then-visit baselines' opening excerpt (`research_bm25`/`_dense`/`_hybrid`, `bm25_dci`), one knob, both arms of the paper's comparison; the snippet-width sweep axis (`32/64/128/256/512`). | `agent_search/agent/tools/doc_research.py:129` |
| `MAX_VISIT_TOKENS` | 12000 | import | Whole-document read cap for every `visit`-style tool (`Bm25Visit`/`DenseVisit`/`HybridVisit`/`BqlVisitWorkspace`) and the per-doc render cap for the `*_autoread` arms. | `agent_search/agent/tools/doc_research.py:130` |
| `MAX_SECTION_TOKENS` | tracks `MAX_VISIT_TOKENS`'s resolved value unless set independently | import | Per-section read cap for every `fetch`-style tool (`DocSearchFetch`, `Bm25FetchWorkspace`, `DenseFetchWorkspace`, `HybridFetchSnipWorkspace`), deliberately mirrors `MAX_VISIT_TOKENS` so the search-vs-fetch factorial's READ axis differs only in WHAT is read, never HOW MUCH. | `agent_search/agent/tools/doc_research.py:136` |
| `BASH_MAX_TOKENS` | 12000 | import | Tail-truncation cap on a DCI `bash` tool call's combined stdout+stderr (`research_dci`, `research_bm25_dci`). | `agent_search/agent/tools/doc_dci.py:44` |
| `READ_MAX_LINE_TOKENS` | 400 | import | Per-line token cap inside the DCI `read` tool's line-range output. | `agent_search/agent/tools/doc_dci.py:46` |
| `GREP_LINE_TOKENS` | 24 | import | Per-line cap on a `grep` tool hit's shown text (the code-fix grep baseline). | `agent_search/agent/tools/code_grep.py:23` |
| `AGENT_CTX_TOKENS` | 115,000 (model tokens) | call (`AgentPolicy` construction, not import) | History budget for the (assistant, observation) pairs kept in the prompt, headroom in a 131k-token window for system prompt + question + next generation. | `agent_search/agent/policies.py:53` |
| `AGENT_CTX_WINDOW` | 131,072 (model tokens) | call (once per episode, `run_episode`) | Assumed model context window used by the proactive early-stop check. | `agent_search/agent/loop.py:215` |
| `AGENT_CTX_STOP_FRAC` | 0.90 (fraction, not tokens) | call (once per episode) | Fraction of `AGENT_CTX_WINDOW` at which the loop injects a forced-answer nudge before another tool call; `>= 1.0` disables the proactive early-stop entirely (falls back to the pre-existing max-steps-only nudge). | `agent_search/agent/loop.py:216` |
| `CLOSER_EVIDENCE_ARG_TOKENS` | 32 | import | Cap on a tool call's argument text shown to the SDK driver's closer/retry forcing prompt. | `agent_search/agent/sdk_driver.py:42` |
| `CLOSER_EVIDENCE_OBS_TOKENS` | 160 | import | Cap on a tool observation's text shown to the same SDK closer/retry forcing prompt. | `agent_search/agent/sdk_driver.py:43` |

### Listing depths

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `BM25_VISIT_TOPK` | 5 | import | SERP listing depth for `Bm25Visit` (`research_bm25`/`research_bm25q`), the ONLY control, since the `bm25_search` tool schema exposes no `k` arg (a hallucinated `k` is ignored). | `agent_search/agent/tools/doc_research.py:715` |
| `AUTOREAD_TOPK` | 5 | import | Number of top hits `Bm25AutoRead`/`DenseAutoRead` render full text for (`research_bm25_autoread`/`research_dense_autoread`). | `agent_search/agent/tools/doc_research.py:849` |
| `DENSE_VISIT_TOPK` | 5 | import | SERP listing depth for `DenseVisit` (`research_dense`), dense twin of `BM25_VISIT_TOPK`, independently settable. | `agent_search/agent/tools/doc_research.py:933` |
| `BM25_FETCH_TOPK` | 10 | import | BM25 retrieval-pool depth staged for `Bm25FetchWorkspace` (`research_bm25_fetch_snip`); same role/constant shape as `BM25_DCI_TOPK`. | `agent_search/agent/tools/doc_research.py:1087` |
| `DENSE_FETCH_TOPK` | 10 | import | Dense retrieval-pool depth for `DenseFetchWorkspace` (`research_dense_fetch`). | `agent_search/agent/tools/doc_research.py:1281` |
| `HYBRID_POOL` | 100 | import | Per-ranker pool depth each of the BM25 and dense rankers is queried to BEFORE RRF fusion (`HybridVisit`/`HybridFetchSnipWorkspace`). | `agent_search/agent/tools/doc_research.py:1585` |
| `HYBRID_VISIT_TOPK` | 5 | import | Post-fusion SERP listing depth for `HybridVisit` (`research_hybrid`). | `agent_search/agent/tools/doc_research.py:1591` |
| `HYBRID_FETCH_TOPK` | 10 | import | Post-fusion result count per search call for `HybridFetchSnipWorkspace` (`research_hybrid_fetch_snip`), distinct from `HYBRID_POOL` (the pre-fusion depth). | `agent_search/agent/tools/doc_research.py:1700` |
| `BM25_DCI_TOPK` | 10 | import | How many BM25 hits one `bm25_search` call surfaces/stages in the BM25→DCI arm (`research_bm25_dci`), a fixed retrieval-stage cutoff, distinct from `--k`'s report cutoffs. | `agent_search/agent/tools/doc_bm25_dci.py:65` |
| `DEDUP_TOPK` | 10 | import | Results one `search` shows in the ITER dedup arms (`dedup_bm25`, `dedup_dense`). | `agent_search/agent/tools/doc_dedup.py` |
| `DEDUP_POOL_K` | 100 | import | Over-fetch pool a dedup search draws from before dropping documents surfaced earlier in the episode. | `agent_search/agent/tools/doc_dedup.py` |

### Retrieval / method switches

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `BQL_SOFT_FALLBACK` | `1` (on) | call (per BQL search call) | `0`/`false`/`no` disables the fallback from a 0-exact-hit Boolean query to whole-corpus BM25 term ranking, the paper's strict-Boolean ablation; ON is the paper's own default for every Sieve arm (the ablation quantifies what graceful degradation contributes). | `agent_search/agent/tools/doc_research.py:546` |
| `BQL_DENSE` | `0` (off) | call (`AgentRetriever.index()`) | Retrofits a `DenseBelief` onto the general BQL/doc-vN executor for the `doc`/`docv2`/`docsnip`/`bqlvisit` arms ONLY, the dedicated `research_bql_dense_visit`/`_snip` arms attach dense unconditionally in their own branch and never consult this. | `agent_search/retrievers/structural/bql/dense_fuse.py:78` |
| `BQL_DENSE_RRF_K` | 60 | import | RRF constant for `BQL_DENSE`'s bm25/dense fusion (a separate local implementation of the same constant `doc_research.RRF_K` uses). | `agent_search/retrievers/structural/bql/dense_fuse.py:66` |
| `BQL_DATE_RANGE` | `1` (on) | call | Enables the `date[YYYY..YYYY]` typed date-range surface in the BQL query grammar; `0`/`false`/`no`/`off` disables it. | `agent_search/retrievers/structural/bql/surface.py:110` |
| `RRF_K` | 60 | import | Reciprocal Rank Fusion constant (Cormack/Clarke/Buettcher 2009, k=60) for the Search-Visit/Search-Fetch HYBRID baseline's bm25+dense fusion. | `agent_search/agent/tools/doc_research.py:1584` |
| `INDRI_DENSE` | off | call (`AgentRetriever.index()`, `indri`/`indrivisit`/`indrisnip` arms) | Attaches a `DenseBelief` to the Indri executor; a missing dense cache degrades to lexical-only with a warning rather than failing. | `agent_search/agent/retriever.py:583` |
| `INDRI_DENSE_W` | 0.35 | call (read live, uncached, to allow test overrides) | Blend weight of the dense belief when `INDRI_DENSE` is attached. | `agent_search/retrievers/structural/indri/model.py:195` |
| `INDRI_DENSE_EXPAND_K` | 50 | call (read live) | Size of the dense top-K pool-expansion union in the Indri ranker. | `agent_search/retrievers/structural/indri/model.py:201` |
| `INDRI_MU` | 2500 | import | Dirichlet-smoothing μ for the Indri LM scorer, also `LUCENE_MU`'s own fallback default. | `agent_search/retrievers/structural/indri/model.py:169` |
| `INDRI_POOL_CAP` | 5000 | import | Candidate pool cap for the Indri executor. | `agent_search/retrievers/structural/indri/model.py:170` |
| `INDRI_RESCORE_M` | 300 | call (read live) | Stage-2 rescore-set size in the Indri two-stage max-score ranking. | `agent_search/retrievers/structural/indri/model.py:187` |
| `LUCENE_MU` | `INDRI_MU`'s resolved value (2500) | call | Dirichlet-smoothing μ for the real-Lucene LMDirichlet engine (`STRUCTURED_BACKEND=lucene`); falls back to `INDRI_MU` when unset so the Python and Lucene structural engines share one default. | `agent_search/retrievers/structural/lucene/engine.py:43` |
| `BQL_SOFT_POOL` | `100` | import | Size of the 0-hit fallback pool ordered by the arm's own ranker (lexically closest docs ∪ dense nearest neighbours when a dense belief is attached), see docs/SIEVE.md "Ranking invariant". | `agent_search/retrievers/structural/bql/executor.py` (`_SOFT_POOL`) |
| `DENSE_QUERY_STYLE` | `plain` | call | How the dense query is written from the agent's history (plain, mem, docs, i1..i7); must match the trained retriever. See docs/TRAINING.md. | `agent_search/training/history.py` |
| `DENSE_QUERY_INSTRUCTION` | unset | call | Override the query instruction prefix; unset = the checkpoint's `skimsearchagent_dense.json` or the built-in table. | `agent_search/retrievers/dense/dense.py` (`query_prefix_for`) |
| `DENSE_POOLING` | unset (auto) | call | Pooling for a local checkpoint directory without a sentence-transformers config: `last_token`, `mean` or `cls`. Unset = the serving note, else `last_token` for a decoder checkpoint (Qwen3-Embedding, ITER, LRAT), else sentence-transformers decides. | `agent_search/retrievers/dense/dense.py` (`resolve_pooling`) |
| `DENSE_INDEX_PATH` | unset | call | A prebuilt vector index to serve instead of the per-corpus cache: this library's cache directory, or ITER's `index.faiss` + `index.lookup.pkl`. Required for an on-disk corpus. | `agent_search/retrievers/dense/dense.py`, `vector_index.py` (`ExternalFaissIndex`) |
| `AGENT_SEARCH_ANN_EF_SEARCH` | 0 (as built) | call | HNSW efSearch for a prebuilt index. | `agent_search/retrievers/dense/vector_index.py` |
| `AGENT_SEARCH_FAISS_MMAP` | unset (load into RAM) | call | `1` memory-maps a prebuilt FAISS index instead of reading it into RAM: less memory, but every early search pages in from disk. | `agent_search/retrievers/dense/vector_index.py` (`ExternalFaissIndex.open`) |
| `BM25_INDEX_PATH` | unset | call | A prebuilt Lucene index directory for `BM25_BACKEND=pyserini`. Required for an on-disk corpus. | `agent_search/retrievers/lexical/pyserini.py` |
| `AGENT_SEARCH_DOCSTORE` | unset | call (dataset load) | `1` forces the topics+corpus loader to serve the corpus from disk regardless of size. | `agent_search/evaluation/datasets.py` |
| `AGENT_SEARCH_DOCSTORE_MIN_BYTES` | 1 GiB | import | Corpus size above which the loader serves it from disk (`agent_search.corpus.docstore`). | `agent_search/evaluation/datasets.py` |

### Engine selection

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `STRUCTURED_BACKEND` | `python` | call (`AgentRetriever.index()`, and every workspace's `executor=None` fallback) | Which structural engine every BQL-family and Indri-family arm uses: `python` (reference) or `lucene` (real-Lucene LMDirichlet/BM25Similarity, requires a prebuilt `indexes/lucene_structured/<key>/`). An unknown value raises loud. | `agent_search/retrievers/structural/backend.py:49` |
| `BM25_BACKEND` | `local` | call (`build_bm25_engine`, and every bm25/hybrid workspace's `engine=None` fallback) | Which BM25 engine every bm25-family and hybrid arm uses: `local` (`BM25Local`, dependency-free, built fresh in memory) or `pyserini` (canonical Lucene BM25, k1=0.9/b=0.4, persisted). Measured ~0.546 top-5 Jaccard divergence between the two on browsecomp_plus. An unknown value raises loud. | `agent_search/retrievers/lexical/__init__.py:43` |
| `DENSE_MODEL` | `BAAI/bge-base-en-v1.5` (general domain only) | call | Overrides the GENERAL-domain dense embedder default (e.g. a Qwen3-Embedding sweep); the CODE-domain default (`nomic-ai/CodeRankEmbed`) never reads this var, so it can't accidentally swap the SWE-bench embedder. | `agent_search/evaluation/datasets.py:373` (a separate, import-time copy of the same knob backs the Indri dense-belief default: `agent_search/retrievers/structural/indri/dense_belief.py:50`) |
| `AGENT_SEARCH_ANN` | `auto` | call (vector-index load) | ANN backend override: `flat`\|`hnsw`\|`ivfpq`\|`auto` (auto picks by corpus size). | `agent_search/retrievers/dense/vector_index.py:64` |
| `AGENT_SEARCH_ANN_MIN` | 1,000,000 | call | Corpus-size threshold to engage HNSW in `auto` mode. | `agent_search/retrievers/dense/vector_index.py:69` |
| `AGENT_SEARCH_ANN_PQ_MIN` | 8,000,000 | call | Corpus-size threshold to engage IVF-PQ in `auto` mode. | `agent_search/retrievers/dense/vector_index.py:70` |
| `AGENT_SEARCH_FLAT_FAISS` | off | call | Opt-in exact `faiss.IndexFlatIP` fast path for the `flat` backend, same stored fp16 embeddings, still exact search, just SIMD+threaded instead of a slow numpy fp16 matmul. | `agent_search/retrievers/dense/vector_index.py:54` |
| `SKIMSEARCHAGENT_PLUGINS` | unset | call (first registry access) | Comma-separated dotted module names imported at registry-discovery time so a plugin can `register()`/`register_dataset()` extra retrievers/datasets. Not in the launcher's `ENV_KNOBS`, see [Open issues](#open-issues). | `agent_search/retrievers/registry.py:142` |

### Index building

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `AGENT_SEARCH_BQL_PREFILTER_MIN` | 5000 (corpus units) | import | Above this corpus size, the BQL executor narrows a live scan with an inverted-index prefilter instead of scanning every unit per query; recall-identical either way, speed only. | `agent_search/retrievers/structural/bql/executor.py:97` |
| `AGENT_SEARCH_DATA` | `data` | import | Root directory datasets are pre-downloaded into / loaded from (`DATA_DIR`). | `agent_search/evaluation/datasets.py:18` |
| `BM25_PYSERINI_THREADS` | all available cores | call (index build) | Indexing parallelism for `pyserini.index.lucene` (Anserini's `JsonCollection` indexer parallelizes across sharded input files). | `agent_search/retrievers/lexical/pyserini.py:122` |
| `BM25_PYSERINI_STORE_RAW` | off | call (index build) | Whether the pyserini index also stores raw text/positions/docvectors (bigger index, slower build) vs lean postings-only. | `agent_search/retrievers/lexical/pyserini.py:137` |
| `LUCENE_INDEX_RAM_MB` | 512 | import | `IndexWriterConfig.setRAMBufferSizeMB` for the structural Lucene index builder. | `agent_search/retrievers/structural/lucene/index_builder.py:152` |
| `LUCENE_INDEX_THREADS` | 1 | import | Concurrent indexing threads for the structural Lucene index builder. | `agent_search/retrievers/structural/lucene/index_builder.py:171` |
| `AGENT_SEARCH_DENSE_DEVICE` | unset (sentence-transformers auto-selects) | call (lazy, first encode) | Forces the dense encoder's device (e.g. `cpu`), cluster launchers (`scripts/run.sh`, `scripts/shard_cell.sh`) use this to keep the encoder off a GPU a co-resident vLLM server owns. | `agent_search/retrievers/dense/dense.py:104` |
| `AGENT_SEARCH_DCI_CACHE` | `$TMPDIR/agent_search_dci` (process-lifetime export cache) | call | Root directory the DCI / BM25-DCI arms (`research_dci`, `research_bm25_dci`) export their flat per-doc `.txt` corpus into. Not in the launcher's `ENV_KNOBS`, see [Open issues](#open-issues). | `agent_search/corpus/flat_export.py:39,51` |

### Model client

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `LLM_TIMEOUT_S` | 600 (seconds) | call (client construction) | HTTP timeout for every OpenAI-compatible client, the agent backend and the LLM judge. | `agent_search/models/backends.py:258` (also `:338`, `:388`) |
| `LLM_RETRY_ATTEMPTS` | 5 | call (per retried call) | Max attempts in the transient-failure retry wrapper (`_with_retries`) around LLM calls. | `agent_search/models/backends.py:178` |
| `LLM_RETRY_BASE_S` | 1.0 (seconds) | call | Exponential-backoff base delay for the same retry wrapper. | `agent_search/models/backends.py:180` |
| `REASONING_EFFORT` | `low` | call | Thinking-depth parameter (`low`\|`medium`\|`high`) for OpenAI reasoning models (o-series / gpt-5\*), which reject `temperature`/`top_p`. | `agent_search/models/backends.py:333` |
| `AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS` | 3 | call | A run stops with a SetupError after this many consecutive instance errors before any instance succeeded (an unreachable endpoint, a broken index); finished instances are kept. `0` disables. | `agent_search/evaluation/run_eval.py` (`evaluate`) |

### Agent driver

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `AGENT_DRIVER` | unset (auto: SDK for OpenAI/Gemini LLM runs; loop for stub/tests + vLLM until Tongyi tool-calling is validated on served vLLM) | call (per `search()` call) | Forces the episode driver: `loop` (text-parsed ReAct, `agent/loop.py`) or `sdk` (OpenAI Agents SDK, native tool-calling; doc arms only, the code arm's `<fix>`-guard terminal isn't ported). | `agent_search/agent/retriever.py:869` |
| `AGENT_DEFAULT_CONDITION` | `research_snip` | import | Which condition the bare `agent`/`bql` retriever aliases resolve to (no single condition is hardwired as "the agent"). | `agent_search/agent/retriever.py:1088` |

### Judge

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `LLM_JUDGE_MODEL` | `gpt-4o-mini` | import | Module-level default grader model for `agent_search.evaluation.llm_judge` (independent of `run_eval --judge-model`'s own hardcoded `gpt-4o-mini` default, the two defaults happen to agree but are two separate literals). | `agent_search/evaluation/llm_judge.py:87` |

### API keys (secrets, excluded from `config.json`)

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `OPENAI_API_KEY` | none (`"EMPTY"` placeholder for a local vLLM server) | call | OpenAI API key for `--model gpt-*`, the default LLM judge, and any `api_base`-less OpenAI-compatible call. | `agent_search/models/backends.py:257`; `agent_search/evaluation/llm_judge.py:153,158`; `agent_search/agent/sdk_driver.py:97` |
| `GEMINI_API_KEY` | none (`""`/`"EMPTY"` placeholder) | call | Gemini API key for `--model gemini-*` via Gemini's OpenAI-compatible endpoint. | `agent_search/models/backends.py:387`; `agent_search/evaluation/llm_judge.py:156`; `agent_search/agent/sdk_driver.py:94` |

### Cluster / SLURM-only

Infra-only vars excluded from `_resolve_env_knobs`'s provenance snapshot (server URLs,
device selection, cache-dir paths, parallelism toggles that don't change results), plus a few
script-local knobs outside the core `run_eval`/`cli.py` path entirely.

| Name | Default | Read at | Controls | Where |
|---|---|---|---|---|
| `VLLM_API_BASE` | `http://localhost:8000/v1` (fallback) | call | Served-endpoint fallback for the SDK driver when no explicit `api_base` is passed. | `agent_search/agent/sdk_driver.py:96` |
| `SWEBENCH_APPT_TMPDIR` | `/tmp/<user>-apptainer-build` | call | Scratch dir for building the SWE-bench Apptainer sandbox image on a cluster node. | `agent_search/evaluation/swebench_apptainer.py:58` |
| `COMPARE_WORKERS` | `min(8, cpu_count())` | call | Thread-pool size for `scripts/compare_cells.py`'s paired-comparison statistics (analysis script, not the eval harness itself). | `scripts/compare_cells.py:886` |
| `MODEL`, `OPENAI_BASE_URL` / `OPENAI_API_BASE` | none | call | Model id / served endpoint for `scripts/force_answer_backfill.py`'s forced-answer recovery pass (script-local, it has no `--model`/`--api-base` flags of its own). | `scripts/force_answer_backfill.py:427-428` |
| `ONESHOT_MAX_TOKENS` / `ONESHOT_MODEL_MAX_CONTEXT` / `ONESHOT_SAFETY_MARGIN` / `ONESHOT_FIT_MARGIN` / `ONESHOT_DOC_CAP` / `ONESHOT_TOTAL_BUDGET` | 4000 / 131072 / 2000 / 256 / 0 / derived | import (first four) / call (last two) | Token budgets for the one-shot RAG floor baseline (`scripts/oneshot_rag.py`, launched via `scripts/oneshot_rag.sbatch`), a single stuffed-prompt baseline entirely outside the agent harness. | `scripts/oneshot_rag.py:78,85,86,101,269,294` |

## Paper base configuration

`docs/REPRODUCING.md` §5 and `docs/SIEVE.md`'s headline-results paragraph specify one fixed base
configuration every paper cell runs under. Comparing each against the library's own default
(above) shows which ones a paper-faithful run sets explicitly:

| knob | paper value | library default | differs? |
|---|---|---|---|
| `MAX_VISIT_TOKENS` | 12,000 | 1,200 | **yes, 10x** |
| `MAX_SECTION_TOKENS` | 12,000 | tracks `MAX_VISIT_TOKENS` (1,200 unless set) | **yes, 10x** |
| `--max-steps` (CLI flag, not an env var) | 100 | 50 | **yes, 2x** |
| `BM25_BACKEND` | `pyserini` | `local` | **yes** |
| `STRUCTURED_BACKEND` | `lucene` | `python` | **yes** |
| `BQL_SOFT_FALLBACK` | `1` (on) | `1` (on) | no, the strict-Boolean ablation sets `0` |
| `SNIPPET_TOKENS` | 32 | 32 | no |
| per-search result depth (`*_VISIT_TOPK`/`*_FETCH_TOPK` family) | 5 | 5 | no, REPRODUCING.md's "results per search (k)" row means these topk knobs, NOT the `--k` report-cutoffs flag (default `[1,3,5,10]`, a completely different axis: rank-metric cutoffs computed post-hoc over the episode's full surfaced set) |
| temperature / seed | 0.6 / 42 | 0.6 / `"42"` (`--seeds`) | no |
| default dense encoder | `BAAI/bge-base-en-v1.5` | `BAAI/bge-base-en-v1.5` (general domain) | no |

So a paper-faithful invocation needs, at minimum:

```bash
MAX_VISIT_TOKENS=12000 MAX_SECTION_TOKENS=12000 BM25_BACKEND=pyserini STRUCTURED_BACKEND=lucene \
python -m agent_search.evaluation.run_eval --max-steps 100 --k 1 3 5 10 \
  --dataset browsecomp_plus_structured_full --retriever agent_research_bql_dense_snip \
  --runs-dir runs/<tier>
```

Every run directory records what landed (`config.json`'s top-level `max_steps` and its
`env_knobs`). Trust that record over the intended invocation. Verify the first rows before scaling
a sweep.

## Open issues

- **`DENSE_MODEL` has two independent read sites with the same default.**
  `agent_search/evaluation/datasets.py::default_dense_model` and
  `agent_search/retrievers/structural/indri/dense_belief.py::DEFAULT_MODEL` both fall back to
  `BAAI/bge-base-en-v1.5`. They agree today; a change to one without the other would make the
  Indri dense-belief arm's embedder diverge silently. Prefer a single source before adding a third.
- **Several knobs are still read at import time** (marked "import" in the tables above). They
  are correctly exported before the harness loads by the `skimsearchagent` launcher and recorded
  in `config.json`, but a library caller who imports `agent_search` first and then sets the
  variable will not see the change in that process.

### Resolved in 0.2.0

- `MAX_STEPS` was documented as an environment variable but never read; the docs now say
  `--max-steps 100` / `max_steps=100`.
- `docs/REPRODUCING.md` now lists the snippet length as 32 whitespace tokens (`SNIPPET_TOKENS`).
- `AGENT_SEARCH_DCI_CACHE`, `SKIMSEARCHAGENT_PLUGINS`, and `VLLM_API_BASE` are accepted by the
  launcher's `ENV_KNOBS`.
- No character-based length limit exists anywhere in `agent_search/` or `scripts/`; every cap is
  a token count through `agent_search/core/tokens.py`. A change that adds one belongs here, not
  in the tables above.

## Training files

Retriever training has its own one-file-per-run format: `skimsearchagent-train-retriever template`
prints every knob with ITER's defaults and a comment each; `skimsearchagent-train-retriever run train.yaml`
runs it. See [docs/TRAINING.md](TRAINING.md).
