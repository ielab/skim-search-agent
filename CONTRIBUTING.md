# Contributing

Thanks for helping build SkimSearchAgent. This page covers setup, where things go, and how to
get a change merged.

## Setup

```bash
git clone https://github.com/ielab/skim-search-agent.git && cd skim-search-agent
python -m pip install -e ".[dev]"      # core plus pytest
python -m pytest -q                    # the full suite; no Java, GPU or API key needed
```

Optional: `.[retrieval]` for the Lucene and dense tests. It needs a JDK 21 or newer on
`JAVA_HOME` before you run the suite: Pyserini starts a JVM on import, and an older Java aborts
the process without a message. `.[api]` adds the OpenAI client tests. `python -m pytest -q -m slow` builds the wheel and checks
that the prompt files ship with it.

## Where things go

| adding | put it in | register with | example |
|---|---|---|---|
| a tool | `agent_search/tools/<name>/tool.py` (and `manual.md` files) | a `Tool` subclass | [EXTENDING.md §3](docs/EXTENDING.md#3-a-tool) |
| a task | `agent_search/tasks/<name>/prompt.md` + `task.py` | `register_task` | [EXTENDING.md §4](docs/EXTENDING.md#4-a-task) |
| a strategy | `agent_search/strategies/<family>.py` | `register_strategy` | [EXTENDING.md §5](docs/EXTENDING.md#5-a-strategy-and-a-condition) |
| a condition (task x strategy) | `agent_search/strategies/paper.py` or your module | `condition` / `alias` | [EXTENDING.md §5](docs/EXTENDING.md#5-a-strategy-and-a-condition) |
| a retriever or ranker | `agent_search/retrievers/` | `retrievers.registry.register` | [EXTENDING.md §2](docs/EXTENDING.md#2-a-retriever-or-ranker) |
| a dataset | `agent_search/evaluation/datasets/` or a plugin | `register_dataset` | [EXTENDING.md §1](docs/EXTENDING.md#1-a-corpus-or-dataset) |
| a model provider | `agent_search/models/` (one file per provider; `make_generate` in `__init__.py`) | a matcher and a branch in `make_generate` | [EXTENDING.md §6](docs/EXTENDING.md#6-a-model-provider) |
| a metric or judge | `agent_search/evaluation/metrics.py`, `doc_scoring.py`, `llm_judge.py` | a function over rows | [EXTENDING.md §7](docs/EXTENDING.md#7-a-metric-or-judge) |
| a knob | `agent_search/experiment.py::SCHEMA`, `cli.py::ENV_KNOBS`, `run_eval._resolve_env_knobs` | one entry in each | [CONFIGURATION.md](docs/CONFIGURATION.md) |

A component that lives outside this repository registers the same way from a plugin module
(entry-point group `skimsearchagent.plugins`, or `SKIMSEARCHAGENT_PLUGINS=my.module`).

## Conventions

- Length limits are token counts (`agent_search.core.tokens`). Do not add character-based caps.
- A tool returns an error as its observation text; it never raises.
- A tool adds the documents it surfaced to `self.state.seen` (an `OrderedSeen`); that order is
  the agent's retrieval ranking for the rank metrics.
- Anything that changes results goes into the experiment-file schema so it is recorded in
  `config.json` and counted in the run's identity.
- Tests use `tmp_path` and never write into the repository. Name test files by subject.
- Documentation is plain technical prose: short sentences, exact commands, no filler. It is
  checked against the code: `tests/test_docs_alignment.py` fails when a doc names a path, console
  script, strategy, dataset or environment knob that does not exist.

## Submitting a change

1. `python -m pytest -q` passes.
2. A new strategy runs with the scripted policy: `skimsearchagent dataset=doc_fixture strategy=<name>`.
3. New prompts or tools come with a scripted-model test (pattern: `tests/test_extension_points.py`).
4. A change that affects the paper's numbers says so in the pull request and in `CHANGELOG.md`.
5. Open the pull request against `main`. CI runs the suite on Python 3.10 and 3.12 and checks the wheel.
