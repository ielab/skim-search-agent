# Architecture

SkimSearchAgent is a research framework for **deep-search agents**: agents that answer a hard
question by searching a fixed collection over several steps. The design goal is to let one
component be swapped while the rest of the experiment stays fixed. The library is six modules
behind one set of contracts, one run record, and one evaluation layer.

```
                      ┌──────────────────────── experiment setup ────────────────────────┐
                      │ configuration (config.json) · corpus snapshot · trace schema      │
                      └───────────────────────────────────────────────────────────────────┘
 documents ──▶ 01 corpus ──▶ 02 retrieval & reranking ──▶ 03 tools, tasks, strategies ──▶ 04 runtime ──▶ 05 evaluation & evidence
              units          BM25 / dense / hybrid /       condition = task x strategy      loop or SDK   metrics · judge ·
              sections       BQL / Indri (engines)         strategy = tools + options       driver        run record · stats
                                                                                                    └──▶ 06 training & rollouts
                                                                                                         (trajectories in rows.jsonl)
```

## The modules and where they live

| # | module | what stays stable | what you swap | code |
|---|---|---|---|---|
| 01 | **Corpus & document representation** | `Unit` (`doc_id`, `title`, `body`, optional `sections`, `metadata`); one dataset registry; corpora too large for memory are served from an on-disk document store | corpus builders, dataset loaders, field profiles | `agent_search/corpus/`, `agent_search/evaluation/datasets/`, `corpus_build/` |
| 02 | **Retrieval & reranking** | `Retriever.index / search`; persistent indexes keyed by corpus identity and content fingerprint; one engine registry per corpus shared by every tool; a hybrid is retrievers plus a fusion method, a reranked retriever is one retriever plus a reranker | Lucene BM25, dense encoders, fusion methods, rerankers, BQL fielded retrieval, Indri-style structured retrieval, your ranker | `agent_search/retrievers/` (`fusion/`, `rerankers/`, `hybrid.py`, `reranked.py`) |
| 03 | **Snippets** | `Snippet.render(unit, terms, width)`: the excerpt a listing shows under one hit, in tokens | the opening line, the query-term window, none, your excerpt method | `agent_search/snippets/` |
| 04 | **Tools, tasks, strategies** | a *tool* is one atomic action with its declaration, its code and its manual, and names its engine kind and its snippet; a *task* is the goal and the answer protocol; a *strategy* is a combination of tools with options, a procedure, or a floor; a *condition* is a task with a strategy | tools, tasks, strategies, conditions | `agent_search/tools/`, `agent_search/tasks/`, `agent_search/strategies/` |
| 05 | **Harness** | `Harness.run(question, ctx)`: how the model is put to work on a condition; every harness returns the same `Trajectory` record. ReAct is the default (the model picks each step), one-shot RAG asks once, a team runs member conditions through their own harnesses and records them all | ReAct, one-shot RAG, plan-and-search, your harness | `agent_search/harness/`, built from `agent_search/agent/` (the step loop, policies, forced answer, drivers, `backbone/` with the model providers, the run record) |
| 06 | **Evaluation & evidence** | the run record (`rows.jsonl` + `config.json` + `results.json`), resume semantics, run identity | metrics, judges, paired statistics | `agent_search/evaluation/`, `scripts/summarize_runs.py`, `scripts/compare_cells.py` |
| 07 | **Training & rollouts** | every episode is a full trajectory (prompts, tool calls, observations, tokens, what each search listed and each read opened) | retriever training from trajectories (the ITER recipe, shipped); policy SFT/RL consumers | `agent_search/training/`, `rows.jsonl` (see [TRAINING.md](TRAINING.md) and "The run record" below) |

Each family declares its contract in a base module next to its implementations:
`retrievers/base.py`, `retrievers/fusion/base.py`, `retrievers/rerankers/base.py`, `snippets/base.py`,
`tools/base.py`, `tasks/base.py`, `strategies/base.py`, `harness/base.py`, `agent/policies.py` and
`agent/backbone/base.py`. Every built-in implements them the same way
a plugin does. [EXTENDING.md](EXTENDING.md) covers each extension point with a
runnable example.

## The rule behind the layout

A family of components is a base class plus one file (or one folder) per concrete thing, and
each file carries that thing's own specifics, even when that repeats code from a sibling.
Selection is a registration or an explicit `matches()`, never a lookup table in a generic
module. You find a component by asking "what is it?" and opening the folder with that name.

## The nouns

**Corpus.** Documents turned into units (chunks, sections, functions), in memory or on disk,
fingerprinted so a persisted index is never served against a corpus it was not built for.

**Engine.** An index over a corpus plus a `query -> ranked ids` function. Each engine is built
once per corpus, persisted under `indexes/`, and shared by every tool that needs it
(`retrievers/engines.py`). There are four families: BM25 (Lucene, through Pyserini), dense
(one file per encoder family, plus the trained-checkpoint family), BQL (Boolean selection with
one ranking model), and Indri. Document corpora rank on Lucene only: BM25 and the structured
index behind BQL and Indri, including the zero-hit fallback and the coverage ranking. A code
repository is the one corpus kind with in-memory engines: the grep ranker and the Boolean
executor with its AST scopes. Those engines re-index only the files that changed.
`retrievers/backend.py` picks the engine by corpus kind, never by an environment variable. A
hybrid is not a family of its own. It is any retrievers the run names, fused by one method
(`rrf` over ranks, `interpolation` over normalised scores), so `search_hybrid` and the `hybrid`
floor use whatever `retrieval.hybrid_retrievers` and `retrieval.hybrid_fusion` say. The paper's
arms are the defaults: BM25 and the dense model, RRF. A reranked engine follows the same idea
with one retriever and a reranker (`retrievers/rerankers/`, one file per method): the base
retriever named in `retrieval.rerank_base` supplies a pool, and the reranker reads each (query,
document) pair and reorders it. Rerankers score online during a run. That is what reranking is.

**Harness.** How the model is put to work on a condition (`harness/`, one file each). ReAct
is the default: a loop in which the model picks each step from the strategy's tools. One-shot
RAG ranks, prompts once, and reads the answer. A team (plan-and-search) runs member conditions
through their own harnesses and combines what they found. The run record keeps every member
trajectory, sums their steps and tokens, and takes the union of their surfaced documents, so a
team is judged with the same metrics as one agent. A strategy names its harness with
`harness=`. A new harness is one new file, and the `agent/` package holds the machinery a
harness is built from.

**Snippet.** How one hit is excerpted in a listing (`snippets/`, one file each): the opening
line, the best window for the query terms, or nothing. A search tool takes one as its
`snippet=` option, so a strategy changes what the model reads under each hit without touching
the tool. Widths are token counts (`SNIPPET_TOKENS`).

**Tool.** One atomic action the agent can call. A tool owns three things: its declaration (the
name the model sees, the description, the JSON parameters), its code, and, when the agent has
to learn a syntax, its manual (the text rendered into the prompt). The code is `run(args)`,
which returns the observation text; an error is text too, never an exception. A tool reads and
writes the episode state (what was listed, what was surfaced, what was read) and names the
engines it needs. One folder per tool: `tools/<name>/tool.py`, plus `manual.md` files where
there is a manual.

The atomic tools: `search_bm25`, `search_dense`, `search_hybrid`, `search_bql`, `search_indri`,
`search_dedup`, `search_bm25_dci`, `visit`, `fetch`, `fetch_code` (with `search_code`),
`get_document`, `bash`, `read`, `grep`.

**Task.** The goal and the answer protocol: the prompt template, the domain, the message format,
the terminal (`<answer>`, `<fix>`, a patch). One folder per task: `tasks/<name>/prompt.md` and
`tasks/<name>/task.py`. The tasks: `research` (the default prompt), `research_paper` (the Sieve paper's), `research_dedup` (ITER's prompt), `research_dedup_strong`, `codefix`,
`codefix_patch`.

**Strategy.** A named combination of tools with their options, or a procedure that uses engines
without a loop. One file per family under `strategies/`:

| file | strategies | tools |
|---|---|---|
| `search_visit.py` | `search_visit`, `search_visit_dense`, `search_visit_hybrid`, `search_visit_reranked`, `search_visit_snippets` | a search that lists documents, then `visit` (whole document) |
| `autoread.py` | `autoread`, `autoread_dense`, `autoread_hybrid` | one search that returns the full text of its hits |
| `search_fetch.py` | `search_fetch`, `search_fetch_dense`, `search_fetch_hybrid`, `search_fetch_bm25_plain`, `search_fetch_dense_plain` | a search that lists structure, then `fetch` (one section); the `_plain` arms drop the excerpt from the listing |
| `sieve.py` | `sieve_bm25`, `sieve`, `sieve_dense`, `sieve_nosnip`, `sieve_nomanual`, `sieve_syntax`, `sieve_noconstruct`, `sieve_nohowto`, `sieve_nofields`, `sieve_nofetch`, `sieve_nohops`, `sieve_noexamples`, `sieve_nomistakes`, `sieve_paper_manual`, `sieve_plain`, `sieve_v2`, `sieve_visit`, `sieve_visit_fused`, `sieve_visit_dense` | `search_bql` (snippets, ranking model, manual set) then `fetch` or `visit` |
| `indri.py` | `indri`, `indri_plain`, `indri_visit` | `search_indri` then `fetch` or `visit` |
| `dci.py` | `dci`, `bounded_dci` | `bash` and `read` over the exported corpus; `bounded_dci` adds `bm25_search` |
| `dedup.py` | `dedup_dense`, `dedup_bm25` | ITER's `search` that drops already-listed documents, then `get_document` |
| `codefix.py` | `codefix`, `codefix_grep` | Boolean code search then `fetch`; or `grep` then `read` |
| `rag.py` | `rag`, `rag_dense`, `rag_hybrid` | no loop: rank once, one prompt, one model call |
| `teams.py` | `plan_and_search`, `plan_and_search_visit` | no loop: a planner, one member agent per sub-question, a synthesizer |
| `retrieval_only.py` | `bm25`, `dense`, `bql`, `grep`, `hybrid`, `reranked` | no loop, no model: the floor |

A strategy gives each tool the exposed name the paper prompt used (`search_s`, `fetch_s`,
`bm25_search`, `visit_d`, ...). The union of its tools also decides the engines a run must
build.

**Condition.** A task with a strategy. What a run names, what a config file's `strategy:` and
`dataset:` resolve to, what the record carries. `strategies/conditions.py` holds the registry.
`strategies/paper.py` keeps the paper's condition names (`research_snip`, `research_bm25`,
`codefix`, ...) as aliases, one line each. Every condition is a retriever the evaluation can run,
named `agent_<condition>`.

## Package map

```
agent_search/
  tokens.py        the token ruler every length limit uses; errors.py: SetupError (a run cannot start)
  corpus/          units, the on-disk document store, corpus fingerprints, code repositories, flat export, grounding ("did you mean")
  retrievers/
    base.py        the Retriever contract, Hit, Observation
    lexical/       pyserini.py (Lucene BM25, every corpus), grep.py and scorer.py (the code repository ranker and its in-memory scorer)
    dense/         base.py (DenseRetriever) + bge.py, coderank.py, qwen3_embedding.py, trained.py; decoder_encoder.py; belief.py; vector_index.py
    bql/           the Boolean structural method: parser, executor, dense fusion, the BQL retriever
    indri/         the Indri query language: parser, fields, result
    lucene/        both query languages compiled to Lucene: compilers, engine, adapters
    fusion/        base.py (the Fusion contract) + rrf.py, interpolation.py: how rankings are combined
    hybrid.py      the hybrid engine and retriever: any retrievers the run names, fused by one method
    rerankers/     base.py (the Reranker contract) + cross_encoder.py: how a candidate pool is reordered
    reranked.py    the reranked engine and retriever: one retriever's pool, one reranker
    backend.py     which engine serves BQL and Indri: Lucene for documents, the in-memory executor for a code repository
    engines.py     the per-corpus engine registry the tools share
    registry.py    name -> retriever builder; plugin discovery; conditions registered as agent_<name>
  snippets/        base.py (Snippet) + opening.py, term_window.py, none.py: the excerpt under a hit, in model tokens
  tools/           base.py (Tool, EpisodeState, ToolBox, the Workspace contract), seen.py (OrderedSeen), budgets.py (the token knobs), common.py (shared rendering),
                   then one folder per tool: search_bm25/, search_dense/, search_hybrid/, search_reranked/, search_bql/, search_indri/,
                   search_dedup/, search_bm25_dci/, visit/, fetch/, fetch_code/, get_document/, bash/, read/, grep/
  tasks/           base.py (Task), render.py (template + declarations + manuals), then research/, research_paper/, research_dedup/, research_dedup_strong/,
                   codefix/, codefix_patch/ (prompt.md + task.py each)
  strategies/      base.py (Strategy), names.py (friendly CLI names), conditions.py (the registry), defaults.py (friendly names under the default prompt), paper.py (the paper's conditions), prompt_variants.py
                   (the paper's names), then one file per family: search_visit.py, autoread.py, search_fetch.py,
                   sieve.py, indri.py, dci.py, dedup.py, codefix.py, rag.py, teams.py, retrieval_only.py
  harness/         base.py (Harness, HarnessContext, HarnessResult), then one file per harness:
                   react.py (the default loop), rag.py (one-shot RAG), plan_and_search.py (a team)
  agent/
    loop.py        the ReAct step loop: reason, act, observe
    record.py      the run record of one episode (trajectory_meta)
    policies.py    AgentPolicy (a model), ScriptPolicy and KeywordPolicy (scripted)
    actions.py     parsing tool calls and answers out of a generation
    forced_answer.py, sdk_driver.py, responses_driver.py
    backbone/      the model providers, one file each: openai_chat.py, openai_reasoning.py, gemini.py, vllm_local.py;
                   base.py (the Model contract), usage.py, retry.py, text.py
  evaluation/
    agent_runner.py  ConditionAgent (a condition run as a Retriever through its strategy's harness)
    datasets/      base.py (Instance, the registry), swebench.py, fixtures.py, beir.py, topics.py
    corpus_units.py, scoring.py, identity.py, runner.py, run_eval.py (the entry point)
    metrics.py, doc_scoring.py, fix_scoring.py, llm_judge.py, build_indexes.py, sample.py, rows.py, config.py
    ground_truth.py (SWE-bench gold patch -> localization ground truth), patch_synthesis.py (fix edits -> a
    unified diff), swebench_apptainer.py (self-hosted SWE-bench resolve-rate runner)
  training/        queries.py (query styles), triples.py (tiered negatives), build_triples.py (the entry point),
                   retriever.py (the trainer), history.py, retriever_eval.py, patches/ (the FlagEmbedding patch)
  experiment.py    the experiment-file schema; cli.py; api.py
```

## One episode, end to end

1. **Dataset to instances.** A dataset loader returns `Instance`s: a question, the shared document
   list, gold document ids, and (for QA sets) a gold answer. The built-in `doc_fixture` is three
   inline documents and one question. The whole path runs with nothing staged.
2. **Documents to units.** `units_from_documents` turns dicts (`_id`, `title`, `text`, optional
   `sections`) into `Unit`s. The title is indexed once, as the unit's name; the body is the text.
3. **Strategy to condition.** The experiment file names a strategy (`sieve_bm25`) and a dataset.
   The dataset's domain picks the task (`research` for documents, `codefix` for code), and the
   pair is the condition (`research_snip` in the paper's names). Its strategy lists its tools;
   their engines are built or loaded once per corpus before the first question. A missing
   artifact stops the run there.
4. **Tools over an episode state.** Per question the strategy binds fresh tool instances to an
   empty episode state. The task renders the system prompt from its template, the tools'
   declarations and their manuals. The policy (the model) keeps as much history as fits in
   `ctx_tokens`. Each turn the model emits one `<tool_call>`, the tool answers with a text
   observation, and the task's terminal (`<answer>`) ends the episode. On the last allowed step,
   or once the observed prompt crosses `AGENT_CTX_STOP_FRAC` of `AGENT_CTX_WINDOW`, a budget
   nudge forces a best-effort answer.
5. **Record.** The trajectory, the surfaced documents in first-seen order (the agent's retrieval
   ranking), the answer, and token usage become one row of `rows.jsonl`. Metrics are computed from
   the row, `results.json` summarises them, and `config.json` records provenance.

A loop-free strategy (`rag`, the floors) skips step 4: the procedure ranks once and either
prompts the model once or returns the ranking as is.

## What the model sees is pinned

The paper's task templates and manuals are files. Each tool carries the declaration text the
paper prompts showed under each exposed name, with two deliberate departures: the `fetch`
declaration names a rank and a section (no nested list) and the BrowseComp Sieve manual
describes the sectioned corpus (see REPRODUCING.md, "Two departures from the paper's prompts").
`tests/test_prompt_fidelity.py` pins the rendered system prompt of every paper condition to its
hash and fails if any other change moves it.

## Length is measured in tokens, never characters

Every limit the agent runs into is a **token** count (`agent_search/tokens.py`): snippet
width, whole-document and section read budgets, shell-output and per-line caps, the prompt history
budget. Every one of them, and the measurement of an episode, uses the same ruler: tiktoken
`o200k_base` when it is installed, whitespace words only as a fallback without it. The paper's
code cut by whitespace words; the library does not. There is no character cap anywhere in the
prompt path. Do not add one: a character clip interacts silently with the token limit next to it.

## The run record

Every experiment writes three files into `runs_dir/<agent|retrieval_only>/<dataset>/<model>/<retriever>`
(a `seed=N` segment is appended when several seeds are requested):

- `config.json`: what ran. Every flag, the resolved domain, every environment knob's value, the
  condition (`prompt_task`, `prompt_strategy`, `prompt_toolset`, `prompt_field_profile`) and the
  hash of the composed system prompt, the experiment file with its hash and overrides, the
  package version, the token ruler and the git revision.
- `rows.jsonl`: one JSON object per instance, appended as instances finish. A row carries the
  question and gold, the agent's ranking (`retrieved`, first-seen order), and its rank metrics.
  It also carries the answer and its scores (`answer_em`, `answer_f1`, the judge's verdict once
  graded), the episode (`actions`, `queries`, `stopped`, `trajectory` with every observation in
  full and the model's raw generation per step), and the cost (provider token counts plus a
  count-once decomposition on one fixed ruler). `agent_search.evaluation.rows.observations_of(row)`
  reads the observations of a row of any age.
- `results.json`: `n`, `n_skipped`, `n_errors` and every numeric row field averaged over the
  scored rows. `judge_summary.json` is added by `skimsearchagent-judge`.

`scripts/summarize_runs.py` tabulates runs and `scripts/compare_cells.py` compares two conditions
on the same instances. A row is a complete trajectory, so the retriever trainer
(`docs/TRAINING.md`) works from `rows.jsonl` alone.

An experiment is a YAML **experiment file** (`skimsearchagent run FILE`). One file fully
determines one setting, every knob is explicit, unknown keys are rejected, and the whole file is
recorded in `config.json`. The key=value launcher and the raw `run_eval` flags take the same
execution path. Behaviour does not depend on how the run was started.

A run's identity is the subset of `config.json` in `RUN_IDENTITY_KEYS`
(`agent_search/evaluation/identity.py`): dataset, retriever, model, dense model, policy,
backend, budgets, seed, cutoffs, the prompt hash and the environment knobs. The served
endpoint's address is not part of it. Re-running the same command resumes: finished instances
are skipped and a torn last line is re-scored. Re-running a *different* experiment into a
directory that has scored rows is refused. Use a new `runs_dir`, or pass
`--allow-config-drift`. A missing artifact, such as a dense embedding cache or a Lucene index,
aborts the run before the first episode instead of partway through.

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
  OpenAI Agents SDK for OpenAI, Gemini, and OpenAI-compatible servers. Same tools, same run
  record. Select one with `AGENT_DRIVER=loop|sdk`.

## The code task

The same loop, tools, and run record also drive a **code** task. The agent is given a bug report
and a repository at a commit. It searches or greps the code, reads the relevant functions, and
commits a `<fix>` naming the file and the change. Those are the `codefix` task with the
`codefix` and `codefix_grep` strategies. The `codefix_patch` task asks for a unified diff
instead. Units are functions parsed out of the repository, and the score is `fix_file_ok` plus
the usual rank metrics over the gold patch's functions. It runs on the inline `code_fixture` and
on staged SWE-bench repositories (`--repo-cache`).

## Cluster launchers

`scripts/slurm/` holds launchers for the smoke suite, index builds, serve-and-run experiments,
retriever training and the ITER smoke and sample pipelines. Run one with `bash` and it executes
in the current shell. Run it with `sbatch` and it submits. Model serving, corpus embedding, and
training never run on a login node.
