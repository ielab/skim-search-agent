# Architecture

SkimSearchAgent is a research harness for **deep-search agents**: agents that answer a hard
question by searching a fixed collection over several steps. The design goal is to allow one
component to be swapped while the rest of the experiment stays fixed. The library is six modules
behind one set of contracts, one run record, and one evaluation layer.

```
                      ┌──────────────────────── experiment setup ────────────────────────┐
                      │ configuration (config.json) · corpus snapshot · trace schema      │
                      └───────────────────────────────────────────────────────────────────┘
 documents ──▶ 01 corpus ──▶ 02 retrieval & reranking ──▶ 03 strategy & tools ──▶ 04 runtime ──▶ 05 evaluation & evidence
              units          BM25 / dense / hybrid /       condition = task x     loop or SDK   metrics · judge ·
              sections       BQL / Indri                   toolset → workspace    driver        run record · stats
                                                                                        └──▶ 06 training & rollouts
                                                                                             (trajectories in rows.jsonl)
```

## The six modules and where they live

| # | module | what stays stable | what you swap | code |
|---|---|---|---|---|
| 01 | **Corpus & document representation** | `Unit` (`doc_id`, `title`, `body`, optional `sections`, `metadata`); one dataset registry; corpora too large for memory are served from an on-disk document store | corpus builders, dataset loaders, field profiles | `agent_search/corpus/`, `agent_search/evaluation/datasets.py`, `corpus_build/` |
| 02 | **Retrieval & reranking** | `Retriever.index / search`; persistent indexes keyed by corpus identity and content fingerprint | BM25 (local or Lucene), dense encoders, RRF fusion, BQL fielded retrieval, Indri-style structured retrieval, your ranker | `agent_search/retrievers/` |
| 03 | **Search strategy & tools** | a *condition* = task template x toolset; tools declared in YAML; a *workspace* answers tool calls | prompts, tools, toolsets, workspaces | `agent_search/prompts/`, `agent_search/agent/tools/` |
| 04 | **Agent runtime** | `run_episode` (reason, act, observe), budgets in tokens, forced-answer handling; the Agents-SDK driver as an alternative runtime | policies, models, drivers | `agent_search/agent/loop.py`, `policies.py`, `sdk_driver.py`, `agent_search/models/` |
| 05 | **Evaluation & evidence** | the run record (`rows.jsonl` + `config.json` + `results.json`), resume semantics, run identity | metrics, judges, paired statistics | `agent_search/evaluation/`, `scripts/summarize_runs.py`, `scripts/compare_cells.py` |
| 06 | **Training & rollouts** | every episode is a full trajectory (prompts, tool calls, observations, tokens, what each search listed and each read opened) | retriever training from trajectories (the ITER recipe, shipped); policy SFT/RL consumers | `agent_search/training/`, `rows.jsonl` (see [TRAINING.md](TRAINING.md), [RUN_RECORD.md](RUN_RECORD.md)) |

Modules 01 through 04 declare their contracts in
[`agent_search/core/interfaces.py`](../agent_search/core/interfaces.py). Every built-in implements
them the same way a plugin does. [EXTENDING.md](EXTENDING.md) covers each extension point with a
runnable example.

## One episode, end to end

1. **Dataset to instances.** A dataset loader returns `Instance`s: a question, the shared document
   list, gold document ids, and (for QA sets) a gold answer. The built-in `doc_fixture` is three
   inline documents and one question. The whole path runs with nothing staged.
2. **Documents to units.** `units_from_documents` turns dicts (`_id`, `title`, `text`, optional
   `sections`) into `Unit`s. The title is indexed once, as the unit's name; the body is the text.
3. **Strategy to condition to workspace.** `sieve_bm25` resolves to the registered retriever
   `agent_research_snip`, which is the condition `research_snip` in `conditions.yaml` (task
   `research` times toolset `search_fetch_s`). The toolset's marker tools select a workspace,
   `DocSearchFetch` in this case. The engines that workspace needs are built or loaded once per
   corpus.
4. **Policy drives the loop.** `AgentPolicy` renders the system prompt (task template, tool
   schemas, per-tool manuals) and keeps as much history as fits in `ctx_tokens`. Each turn the
   model emits one `<tool_call>`, the workspace answers with a text observation, and a terminal
   `<answer>` ends the episode. On the last allowed step, or once the observed prompt crosses
   `AGENT_CTX_STOP_FRAC` of `AGENT_CTX_WINDOW`, a budget nudge forces a best-effort answer.
5. **Record.** The trajectory, the surfaced documents in first-seen order (the agent's retrieval
   ranking), the answer, and token usage become one row of `rows.jsonl`. Metrics are computed from
   the row, `results.json` summarises them, and `config.json` records provenance.

## Length is measured in tokens, never characters

Every limit the agent runs into is a **token** count (`agent_search/core/tokens.py`): snippet
width, whole-document and section read budgets, shell-output and per-line caps, the prompt history
budget. Read caps count whitespace tokens, the paper's ruler, which stays
tokenizer-independent. Measurement and the history budget use tiktoken `o200k_base` when it is
installed, whitespace tokens otherwise. There is no character cap anywhere in the prompt path. Do
not add one: a character clip interacts silently with the token limit next to it.

## Run identity and resumption

An experiment is a YAML **experiment file** (`skimsearchagent run FILE`). One file fully
determines one setting, every knob is explicit, unknown keys are rejected, and the whole file is
recorded in `config.json`. The key=value launcher and the raw `run_eval` flags take the same
execution path. Behaviour does not depend on how the run was started.

A run directory is `runs_dir/<agent|retrieval_only>/<dataset>/<model>/<retriever>`, with a
`seed=N` segment appended when several seeds are requested. `config.json` records the exact
invocation, every environment knob, the composed prompt's hash, the package version, the token
ruler, and the git revision. Re-running the same command resumes: finished instances are skipped
and a torn last line is re-scored. Re-running a *different* experiment into the same directory
(another backend, seed, budget, or prompt) is refused. Use a new `runs_dir`, or pass
`--allow-config-drift`. Missing artifacts, a dense embedding cache or a Lucene index, abort the
run before the first episode instead of partway through.

## Ranking never changes model behind the agent's back

A structured search (BQL, Indri) picks candidates and one ranking model orders them. When a query
matches nothing, the fallback ranks a wider pool with that same model over that same index
(`docs/SIEVE.md`, "Ranking invariant"), so a zero-hit query cannot switch the ranker
mid-episode. Dense similarity always comes from a persisted embedding cache built for the run's
`dense_model`, or from a prebuilt index named in the file. A dense arm without either stops
before the first episode rather than encoding the corpus on the clock. A run whose first questions
all fail (an unreachable model endpoint, a broken index) stops after three of them instead of
burning the retry budget on every question.

## Two runtimes

* **Loop driver** (the default): text-parsed tool calls. Works with any `messages -> text`
  callable, in-process vLLM included, and supports the forced-answer prefill on served models.
* **Agents-SDK driver** (optional, `pip install ".[agents]"`): native function calling through the
  OpenAI Agents SDK for OpenAI, Gemini, and OpenAI-compatible servers. Same workspaces, same run
  record. Select one with `AGENT_DRIVER=loop|sdk`.

## The code-localization arm

The same loop, tools, and run record also drive a **code** task. The agent is given a bug report
and a repository at a commit. It searches or greps the code, reads the relevant functions, and
commits a `<fix>` naming the file and the change. Those are `codefix` and `codefix_grep`;
`codefix_patch` emits a unified diff instead. Units are functions parsed out of the repository,
the task template is `taskfix`, and the score is `fix_file_ok` plus the usual rank metrics over
the gold patch's functions. It runs on the inline `code_fixture` and on staged SWE-bench
repositories (`--repo-cache`).

## Cluster launchers

`scripts/slurm/` holds launchers for the smoke suite, index builds, serve-and-run experiments,
retriever training and the ITER smoke and sample pipelines. Run one with `bash` and it executes
in the current shell. Run it with `sbatch` and it submits. Model serving, corpus embedding, and
training never run on a login node.
