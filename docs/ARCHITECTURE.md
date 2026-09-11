# Architecture

SkimSearchAgent is a research harness for **deep-search agents**: agents that answer a hard
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

## The six modules and where they live

| # | module | what stays stable | what you swap | code |
|---|---|---|---|---|
| 01 | **Corpus & document representation** | `Unit` (`doc_id`, `title`, `body`, optional `sections`, `metadata`); one dataset registry; corpora too large for memory are served from an on-disk document store | corpus builders, dataset loaders, field profiles | `agent_search/corpus/`, `agent_search/evaluation/datasets/`, `corpus_build/` |
| 02 | **Retrieval & reranking** | `Retriever.index / search`; persistent indexes keyed by corpus identity and content fingerprint; one engine registry per corpus shared by every tool | BM25 (local or Lucene), dense encoders, RRF fusion, BQL fielded retrieval, Indri-style structured retrieval, your ranker | `agent_search/retrievers/` |
| 03 | **Tools, tasks, strategies** | a *tool* is one atomic action with its declaration, its code and its manual; a *task* is the goal and the answer protocol; a *strategy* is a combination of tools with options; a *condition* is a task with a strategy | tools, tasks, strategies, conditions | `agent_search/tools/`, `agent_search/tasks/`, `agent_search/strategies/` |
| 04 | **Agent runtime** | `run_episode` (reason, act, observe), budgets in tokens, forced-answer handling; the Agents-SDK driver as an alternative runtime | policies, models, drivers | `agent_search/agent/` (the loop, policies, drivers, and `backbone/` with the model providers) |
| 05 | **Evaluation & evidence** | the run record (`rows.jsonl` + `config.json` + `results.json`), resume semantics, run identity | metrics, judges, paired statistics | `agent_search/evaluation/`, `scripts/summarize_runs.py`, `scripts/compare_cells.py` |
| 06 | **Training & rollouts** | every episode is a full trajectory (prompts, tool calls, observations, tokens, what each search listed and each read opened) | retriever training from trajectories (the ITER recipe, shipped); policy SFT/RL consumers | `agent_search/training/`, `rows.jsonl` (see [TRAINING.md](TRAINING.md), [RUN_RECORD.md](RUN_RECORD.md)) |

Each family declares its contract in a base module next to its implementations:
`retrievers/base.py`, `tools/base.py`, `tasks/base.py`, `strategies/base.py`,
`agent/policies.py` and `agent/backbone/base.py`. Every built-in implements them the same way
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

**Engine.** An index over a corpus plus a `query -> ranked ids` function. Built once per corpus,
persisted under `indexes/`, shared by every tool that needs it (`retrievers/engines.py`).
Families: BM25 (in memory, Lucene), dense (one file per encoder family, plus the
trained-checkpoint family), BQL (Boolean selection with one ranking model), Indri.

**Tool.** One atomic action the agent can call. A tool owns its declaration (the name the model
sees, the description, the JSON parameters), its code (`run(args)` returns the observation text;
an error is text, never an exception) and, when the agent has to learn a syntax, its manual (the
text rendered into the prompt). A tool reads and writes the episode state (what was listed, what
was surfaced, what was read) and names the engines it needs. One folder per tool:
`tools/<name>/tool.py`, plus `manual.md` files where there is a manual.

The atomic tools: `search_bm25`, `search_dense`, `search_hybrid`, `search_bql`, `search_indri`,
`search_dedup`, `search_bm25_dci`, `visit`, `fetch`, `fetch_code` (with `search_code`),
`get_document`, `bash`, `read`, `grep`.

**Task.** The goal and the answer protocol: the prompt template, the domain, the message format,
the terminal (`<answer>`, `<fix>`, a patch). One folder per task: `tasks/<name>/prompt.md` and
`tasks/<name>/task.py`. The tasks: `research`, `research_dedup` (ITER's prompt), `codefix`,
`codefix_patch`.

**Strategy.** A named combination of tools with their options, or a procedure that uses engines
without a loop. One file per family under `strategies/`:

| file | strategies | tools |
|---|---|---|
| `search_visit.py` | `search_visit`, `search_visit_dense`, `search_visit_hybrid`, `search_visit_snippets` | a search that lists documents, then `visit` (whole document) |
| `autoread.py` | `autoread`, `autoread_dense`, `autoread_hybrid` | one search that returns the full text of its hits |
| `search_fetch.py` | `search_fetch`, `search_fetch_dense`, `search_fetch_hybrid`, `search_fetch_bm25_plain`, `search_fetch_dense_plain` | a search that lists structure, then `fetch` (one section); the `_plain` arms drop the excerpt from the listing |
| `sieve.py` | `sieve_bm25`, `sieve`, `sieve_dense`, `sieve_nosnip`, `sieve_plain`, `sieve_v2`, `sieve_visit`, `sieve_visit_fused`, `sieve_visit_dense` | `search_bql` (snippets, ranking model, manual set) then `fetch` or `visit` |
| `indri.py` | `indri`, `indri_plain`, `indri_visit` | `search_indri` then `fetch` or `visit` |
| `dci.py` | `dci`, `bounded_dci` | `bash` and `read` over the exported corpus; `bounded_dci` adds `bm25_search` |
| `dedup.py` | `dedup_dense`, `dedup_bm25` | ITER's `search` that drops already-listed documents, then `get_document` |
| `codefix.py` | `codefix`, `codefix_grep` | Boolean code search then `fetch`; or `grep` then `read` |
| `rag.py` | `rag`, `rag_dense`, `rag_hybrid` | no loop: rank once, one prompt, one model call |
| `retrieval_only.py` | `bm25`, `bm25_lucene`, `dense`, `bql`, `grep` | no loop, no model: the floor |

A strategy gives each tool the exposed name the paper prompt used (`search_s`, `fetch_s`,
`bm25_search`, `visit_d`, ...) and, by the union of its tools, the engines a run must build.

**Condition.** A task with a strategy. What a run names, what a config file's `strategy:` and
`dataset:` resolve to, what the record carries. `strategies/conditions.py` holds the registry;
`strategies/paper.py` keeps the paper's condition names (`research_snip`, `research_bm25`,
`codefix`, ...) as aliases, one line each. Every condition is a retriever the harness can run,
named `agent_<condition>`.

## Package map

```
agent_search/
  tokens.py        the token ruler every length limit uses; errors.py: SetupError (a run cannot start)
  corpus/          units, the on-disk document store, corpus fingerprints, code repositories, flat export, grounding ("did you mean")
  retrievers/
    base.py        the Retriever contract, Hit, Observation
    lexical/       scorer.py (the BM25 scorer), bm25.py (in memory), pyserini.py (Lucene), grep.py
    dense/         base.py (DenseRetriever) + bge.py, coderank.py, qwen3_embedding.py, trained.py; belief.py; vector_index.py
    bql/           the Boolean structural method: parser, executor, dense fusion, the BQL retriever
    indri/         the Indri query language: parser, index, model
    lucene/        both query languages compiled to Lucene: compilers, engine, adapters
    backend.py     which engine serves BQL and Indri (Python reference or Lucene)
    engines.py     the per-corpus engine registry the tools share
    registry.py    name -> retriever builder; plugin discovery; conditions registered as agent_<name>
  tools/           base.py (Tool, EpisodeState, ToolBox, the Workspace contract), seen.py (OrderedSeen), budgets.py (the token knobs), common.py (shared rendering),
                   then one folder per tool: search_bm25/, search_dense/, search_hybrid/, search_bql/, search_indri/,
                   search_dedup/, search_bm25_dci/, visit/, fetch/, fetch_code/, get_document/, bash/, read/, grep/
  tasks/           base.py (Task), render.py (template + declarations + manuals), then research/, research_dedup/,
                   codefix/, codefix_patch/ (prompt.md + task.py each)
  strategies/      base.py (Strategy), names.py (friendly CLI names), conditions.py (the registry), paper.py
                   (the paper's names), then one file per family: search_visit.py, autoread.py, search_fetch.py,
                   sieve.py, indri.py, dci.py, dedup.py, codefix.py, rag.py, retrieval_only.py
  agent/
    loop.py        one episode: reason, act, observe
    policies.py    AgentPolicy (a model), ScriptPolicy and KeywordPolicy (scripted)
    actions.py     parsing tool calls and answers out of a generation
    forced_answer.py, sdk_driver.py
    backbone/      the model providers, one file each: openai_chat.py, openai_reasoning.py, gemini.py, vllm_local.py;
                   base.py (the Model contract), usage.py, retry.py, text.py
  evaluation/
    agent_runner.py  ConditionAgent (a condition run as a Retriever), ProcedureAgent (loop-free strategies)
    datasets/      base.py (Instance, the registry), swebench.py, fixtures.py, beir.py, topics.py
    corpus_units.py, scoring.py, identity.py, runner.py, run_eval.py (the entry point)
    metrics.py, doc_scoring.py, fix_scoring.py, llm_judge.py, build_indexes.py, sample.py, rows.py, config.py
    ground_truth.py (SWE-bench gold patch -> localization ground truth), patch_synthesis.py (fix edits -> a
    unified diff), swebench_apptainer.py (self-hosted SWE-bench resolve-rate harness)
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

The paper's task templates and manuals are files, moved with no byte changed. Each tool carries
the exact declaration text the paper prompts showed under each exposed name.
`tests/test_prompt_fidelity.py` pins the rendered system prompt of every paper condition to its
hash and fails if any change moves it.

## Length is measured in tokens, never characters

Every limit the agent runs into is a **token** count (`agent_search/tokens.py`): snippet
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
  OpenAI Agents SDK for OpenAI, Gemini, and OpenAI-compatible servers. Same tools, same run
  record. Select one with `AGENT_DRIVER=loop|sdk`.

## The code task

The same loop, tools, and run record also drive a **code** task. The agent is given a bug report
and a repository at a commit. It searches or greps the code, reads the relevant functions, and
commits a `<fix>` naming the file and the change. Those are the `codefix` task with the
`codefix` and `codefix_grep` strategies; the `codefix_patch` task asks for a unified diff
instead. Units are functions parsed out of the repository, and the score is `fix_file_ok` plus
the usual rank metrics over the gold patch's functions. It runs on the inline `code_fixture` and
on staged SWE-bench repositories (`--repo-cache`).

## Cluster launchers

`scripts/slurm/` holds launchers for the smoke suite, index builds, serve-and-run experiments,
retriever training and the ITER smoke and sample pipelines. Run one with `bash` and it executes
in the current shell. Run it with `sbatch` and it submits. Model serving, corpus embedding, and
training never run on a login node.
