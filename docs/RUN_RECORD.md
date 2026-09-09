# The run record

Every experiment writes three files into its run directory. Together they are the library's
single trace and evidence format: enough to score, compare, audit and train from without
re-running anything.

```
runs/<agent|retrieval_only>/<dataset>/<model>/<retriever>[/seed=N]/
├── config.json      what was run (provenance; defines the run's identity)
├── rows.jsonl       one JSON object per instance: the full episode (append-only, resumable)
├── results.json     the aggregate summary (metrics averaged over scored rows)
└── judge_summary.json   present after an LLM-judge pass
```

## `config.json`

| key | meaning |
|---|---|
| every `run_eval` flag (`dataset`, `retriever`, `model`, `policy`, `backend`, `max_steps`, `temperature`, `seed`, `k`, `limit`, ...) | the resolved invocation |
| `resolved_domain` | `general` (documents) or `code` |
| `env_knobs` | every environment knob's resolved value (`SNIPPET_TOKENS`, `MAX_VISIT_TOKENS`, `STRUCTURED_BACKEND`, `BM25_BACKEND`, `AGENT_CTX_*`, ...). See [CONFIGURATION.md](CONFIGURATION.md) |
| `prompt_task`, `prompt_toolset`, `prompt_profile`, `prompt_sha256` | the condition, plus a hash of the *composed* system prompt (task × tools × manuals) |
| `package_version`, `token_ruler`, `git_rev`, `started_at` | code version, which token ruler measured this run, when it started |
| `experiment_file`, `experiment_file_sha256`, `experiment`, `experiment_overrides`, `experiment_sha256` | the experiment file the run came from, its hash as written, the setting that actually ran (overrides applied), the overrides, and the hash of that setting |

A run's **identity** is the subset of those keys listed in `RUN_IDENTITY_KEYS`
(`agent_search/evaluation/run_eval.py`): dataset, retriever, model, policy, backend, budgets,
seed, prompt hash and env knobs. The harness refuses to resume into a directory whose identity
differs.

## `rows.jsonl`: one row per instance

Rows are appended as instances finish, so a killed run picks up where it stopped. A row is either
a *skip* (`{"instance_id", "skipped": "no_units" | "no_gold"}`) or a full record.

### Identity and gold

| field | type | meaning |
|---|---|---|
| `instance_id` | str | `<dataset>__<query id>` |
| `question` | str | the information need (QA datasets) |
| `gold_answer` | str | the reference answer (QA datasets) |
| `gold_ids` | list[str] | gold document ids (`qrels`) |
| `n_gold` | int | number of gold ids |

### Retrieval outcome

| field | type | meaning |
|---|---|---|
| `retrieved` | list[dict] | the agent's ranking: surfaced documents in first-seen order, with `rank`, `doc_id`, `title`, `snippet` |
| `n_retrieved` | int | length of the ranking |
| `recall@k`, `hit@k`, `acc@k`, `precision@k`, `f1@k`, `map@k`, `ndcg@k`, `mrr@10` | float | rank metrics at every `--k`; absent on answer-only rows |
| `answer_only` | bool | the dataset has an answer but no document labels (e.g. InfoSeek): the row is scored on the answer, `n_gold` is 0 and the rank metrics are left out |
| `set_size`, `set_recall`, `set_precision`, `set_f1` | float | cutoff-free set metrics over the surfaced set |
| `surfaced_docs` | list[str] | every document the episode surfaced (search hits and reads), sorted |
| `gold_doc_coverage` | float | fraction of gold documents surfaced |

### Answer outcome (QA datasets)

| field | type | meaning |
|---|---|---|
| `final_answer` | str | the agent's answer span |
| `answer_em`, `answer_f1` | float | exact match and token F1 after SQuAD normalisation (HotpotQA protocol, including the yes/no guard) |
| `support_f1` | float | MuSiQue support F1, when the dataset carries supporting-document ids |
| `grounded_*` | float | the answer scored only if it also shows up in the tool evidence (`doc_scoring.score_answer`) |
| `judge_correct`, `judge_extracted`, `judge_reasoning` | bool / str | LLM-judge verdict (BrowseComp-Plus protocol). `judge_correct` is `null` with a `judge_error` when the judge's reply couldn't be parsed |

### The episode

| field | type | meaning |
|---|---|---|
| `retriever`, `tool_condition`, `tool_description`, `domain`, `prompt_profile_path`, `max_steps` | str/int | which strategy produced the row |
| `actions` | list[str] | tool name per step, then the terminal (`answer`, `submit`, `stop`) |
| `queries` | list[str] | the query argument per step (`""` for non-search tools) |
| `hits_per_step` | list[int] | how many results each search returned |
| `stopped` | str | `answer`, `submit`, `stop`, `fix`, or a budget stop: `max_steps`, `ctx_budget` (loop driver), `max_turns` (SDK driver) |
| `elicitation` | str or null | where a forced answer came from: `nudge`, `prefill_inline`, `prefill_failed`, `ask_retry_inline`, `ask_retry_failed` |
| `declared` | list[str] | what the terminal declared (the answer text, or locations for the code arm) |
| `trajectory` | list[dict] | one entry per step: `action`, `args`, `query`, `observation` (the **full** tool output, no cap), `raw_output` (the model's generation), `n_hits`, `t_llm_s`, `t_tool_s`, `prompt_tokens`, `completion_tokens` |

`trajectory[i]["observation"]` is the complete text the agent saw. To read observations from rows
of any age, use `agent_search.evaluation.rows.observations_of(row)`. Rows written before September
2026 kept a separate full `observations` list plus a capped copy in the trajectory. The helper
reads both layouts.

### Cost

These counts are model tokens as the provider reported them (`prompt_tokens`,
`completion_tokens`, `cached_input_tokens`, `reasoning_tokens`, `llm_calls`, `n_steps`). On top of
those the harness computes a count-once decomposition on one fixed ruler
(`config.json:token_ruler`):

| field | meaning |
|---|---|
| `initial_prompt_tokens` | the first call's prompt (system prompt plus question), counted once |
| `context_once_tokens` (= `retrieved_doc_tokens`, `read_tokens`) | the most context the model ever held beyond the initial prompt (`max(step prompt_tokens) − initial`) |
| `output_tokens` | generated tokens |
| `total_tokens_once` | `initial_prompt_tokens + context_once_tokens + output_tokens`, the marginal work of the episode |
| `token_source` | `prompt_tokens` (from real per-step usage) or `fallback` (summed observation text, for rows with no per-step usage) |

## `results.json`

A summary and nothing more: `n` scored, `n_skipped`, `n_errors`, `level`, `metrics` (every numeric
row field averaged over the rows that carry it, plus `timeout_rate`, the fraction of budget stops),
and `rows_file` pointing at `rows.jsonl`. It never embeds the rows themselves.

## Reading and comparing runs

* `python scripts/summarize_runs.py --help` prints tables across run directories and paired
  t-tests across seeds.
* `python scripts/compare_cells.py --help` does a paired comparison of two conditions on the same
  instances (McNemar), with cost columns and judge overlays.
* `python -m agent_search.evaluation.llm_judge --results-dir <dir> --judge-model gpt-4o-mini`
  grades answers. It is idempotent: already-graded rows are skipped unless `--rejudge` is passed.

## Using the record for training

Each row is a complete, replayable trajectory. The composed system prompt is reproducible from
`config.json:prompt_sha256` plus the prompt files at `git_rev`, the question is in `question`, and
`trajectory[i].raw_output` and `trajectory[i].observation` are the alternating assistant and tool
turns. For the exact rebuild, read `scripts/force_answer_backfill.py::reconstruct_messages`.
