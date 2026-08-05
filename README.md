<div align="center">

# SkimSearchAgent

**A composable library for building and evaluating deep-research agents over your own corpus.**

Bring the documents, model, and search strategy. SkimSearchAgent supplies the agent loop,
retrieval and reading primitives, reproducible evaluation, and the interfaces that connect them.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](#quickstart)
[![Research preview](https://img.shields.io/badge/status-research%20preview-7C3AED.svg)](#project-status)
[![Pluggable](https://img.shields.io/badge/corpora%20%7C%20models%20%7C%20methods-pluggable-00897B.svg)](#what-can-be-swapped)

[Quickstart](#quickstart) ·
[Concepts](#what-can-be-swapped) ·
[Strategies](#included-strategies) ·
[Extend](#build-your-own) ·
[Sieve](docs/SIEVE.md) ·
[Reproduce the paper](docs/REPRODUCING.md)

</div>

SkimSearchAgent is infrastructure for research agents that answer difficult questions by
interacting with a collection over multiple steps. It deliberately separates the parts that are
often entangled in an agent implementation: the corpus, model provider, agent policy, tool
surface, retrieval and ranking method, reading strategy, and evaluator. You can replace one layer
without rewriting the rest of the system.

The repository includes conventional Search–Visit agents, section-level Search–Fetch agents,
direct-corpus interaction, one-shot retrieval, lexical and dense retrieval, and structured search.
**Sieve is one included strategy built from these components; it is not the scope of the library.**

<p align="center">
  <img src="docs/assets/skimsearchagent-overview.png" width="100%" alt="SkimSearchAgent architecture: interchangeable corpora, models, strategies, retrievers, and evaluators around one research-agent loop"/>
</p>

## Why SkimSearchAgent?

A useful deep-research experiment changes more than a prompt. It may change how documents are
represented, what a search result exposes, how the agent reads evidence, which model drives the
loop, and how answer quality and cost are measured. SkimSearchAgent gives those decisions explicit
interfaces and runs every configuration through the same harness.

- **Build complete research agents.** Use the shared reason–act–observe loop, termination and
  answer handling, configurable budgets, prompt profiles, and tool registry.
- **Work over different corpora.** Document collections share one corpus and unit abstraction.
  Flat text, passages, sections, and metadata are all supported.
- **Mix search and reading strategies.** Combine whole-document visits, section fetches, result
  cards, snippets, shell-style corpus access, or your own tools.
- **Swap retrieval components.** Use local or Lucene BM25, dense retrieval, reciprocal-rank
  fusion, BQL fielded retrieval, Indri-style structured retrieval, or a custom ranker.
- **Bring the model you need.** Run an open model in-process with vLLM, call a local
  OpenAI-compatible server, use OpenAI or Gemini through their APIs, or implement the small model
  protocol.
- **Evaluate the whole system.** Save resumable run traces and measure answer accuracy, retrieval
  quality, model calls, tokens, latency, and paired statistical comparisons.

## Quickstart

SkimSearchAgent requires Python 3.10 or newer. The reference BQL engine and local BM25 smoke test
do not require Java, a GPU, or an API key.

```bash
git clone <repository-url>
cd skim-search-agent

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Try structured Boolean search over a tiny in-memory corpus:

```bash
python -m examples.quickstart_bql
```

Then run the dependency-free evaluation smoke test:

```bash
python -m evaluation.run_eval \
  --dataset fixture \
  --retriever bm25_local \
  --runs-dir runs/smoke
```

For document-scale retrieval, dense encoders, and local model serving, install the optional stack:

```bash
python -m pip install -e ".[all]"
```

The package extras use the same versions as [`requirements.txt`](requirements.txt), which records
the exact Python 3.10 environment used for the paper. Use the requirements file when reproducing
the released experiments end to end.

After staging a document collection as described in
[`corpus_build/`](corpus_build/README.md), a complete agent run against an OpenAI model looks like
the following. API keys and the common configuration knobs are documented in
[`.env.example`](.env.example) — copy it to `.env`, fill in what you need, and load it with
`set -a; source .env; set +a` (nothing auto-loads it).

```bash
export OPENAI_API_KEY=...

python -m evaluation.run_eval \
  --dataset hotpotqa_structured \
  --retriever agent_research_bm25 \
  --policy llm \
  --model gpt-4o-mini \
  --runs-dir runs/demo \
  --limit 20
```

For an open model served by vLLM, use the same command with `--backend api`, the served model ID,
and `--api-base http://localhost:8000/v1`. In-process vLLM is available with `--backend vllm`.

## What can be swapped?

| layer | interface or entry point | included examples |
|---|---|---|
| Corpus | [`CorpusSource`](agent_search/core/interfaces.py), [`Unit`](agent_search/core/units.py) | shared QA collections, structured documents |
| Model | [`Model`](agent_search/core/interfaces.py), [`models/backends.py`](agent_search/models/backends.py) | in-process vLLM, OpenAI-compatible servers, OpenAI, Gemini |
| Agent policy | [`Policy`](agent_search/agent/loop.py), prompt profiles | ReAct-style research policies |
| Tools and reading | [`Tool`](agent_search/core/interfaces.py), [`agent/tools/`](agent_search/agent/tools/) | search, result inspection, whole-page visit, section fetch, shell and file access |
| Retrieval and ranking | [`Retriever`](agent_search/core/interfaces.py), [`retrievers/`](agent_search/retrievers/) | grep, BM25, dense, hybrid, BQL, Indri-style retrieval |
| Evaluation | [`evaluation/`](evaluation/) | answer scoring, retrieval metrics, LLM judging, tokens, calls, traces |

The main command-line entry point is always `python -m evaluation.run_eval`. A configuration can
change several layers, but the execution, provenance, and scoring path remains the same.

## Included strategies

Strategies are presets over the shared agent loop and tool interfaces. They are selected with
`--retriever`; the name is historical and includes both ordinary retrievers and complete agent
configurations.

| family | representative values | interaction |
|---|---|---|
| Retrieval-only | `bm25_local`, `bm25_pyserini`, `dense`, `bql`, `grep` | return a ranked set without an agent loop |
| One-shot retrieve–read | [`scripts/oneshot_rag.py`](scripts/oneshot_rag.py) | retrieve once, then answer once |
| Search–AutoRead | `agent_research_bm25_autoread`, `agent_research_dense_autoread` | attach full text to each search result |
| Direct corpus interaction | `agent_research_dci`, `agent_research_bm25_dci` | search raw files directly, optionally inside a BM25-bounded working set |
| Search–Visit | `agent_research_bm25`, `agent_research_dense`, `agent_research_hybrid` | inspect ranked results, then open whole documents |
| Search–Fetch | `agent_research_bm25_fetch_snip`, `agent_research_dense_fetch`, `agent_research_hybrid_fetch_snip` | inspect result cards, then fetch named sections |
| Sieve | `agent_research_snip`, `agent_research_bql_donly_snip`, `agent_research_bql_dense_snip` | BQL candidate selection, pluggable ranking, result cards, and section fetch |
| Structured-retrieval controls | `agent_research_indri`, `agent_research_indri_snip`, `agent_research_indri_visit` | Indri-style structured retrieval with different reading surfaces |

Run `python -m evaluation.run_eval --help` to see every registered strategy and dataset. The Sieve
design, settings, and paper-specific ablations live in [`docs/SIEVE.md`](docs/SIEVE.md), separate
from the library overview.

## Retrieval and ranking components

| component | available implementations |
|---|---|
| Sparse | dependency-free local BM25 or Pyserini/Lucene BM25 |
| Dense | sentence-transformers-compatible encoders with model-specific persistent caches |
| Fusion | reciprocal-rank fusion of lexical and dense rankings |
| Fielded retrieval | BQL over `title`, `section`, `body`, dates, and corpus-specific metadata; reference Python and Lucene executors |
| Structured scoring | Indri-style operators and Dirichlet-smoothed belief scoring, with optional dense fusion |
| Vector indexing | exact NumPy/FAISS search, with HNSW and IVF-PQ options for larger collections |

The corpus determines which fields exist. A custom collection does not need to be structured: it
can expose only document text and still use the ordinary lexical, dense, and whole-document agent
strategies.

## Models and serving

The agent only requires a callable that accepts chat messages and returns generated text. Built-in
dispatch supports:

- OpenAI models via `OPENAI_API_KEY`;
- Gemini models through Gemini's OpenAI-compatible endpoint and `GEMINI_API_KEY`;
- any model exposed through an OpenAI-compatible server with `--backend api`;
- open-weight models loaded directly by vLLM with `--backend vllm`.

Model choice is independent of retrieval choice. The same search strategy can therefore be tested
across agent backbones without changing its tools, budget, corpus, or evaluator.

For OpenAI, Gemini, and served OpenAI-compatible models, episodes are driven through the OpenAI
Agents SDK when the optional `openai-agents` package is installed, and through the built-in
text-parsed loop driver otherwise (in-process vLLM always uses the loop driver). Set
`AGENT_DRIVER=loop|sdk` to override the choice explicitly.

## Build your own

### Add a retrieval method

Implement the [`Retriever`](agent_search/core/interfaces.py) contract and register a lazy factory.
Modules placed under `agent_search/retrievers/` are discovered automatically.

```python
# agent_search/retrievers/my_method.py
from agent_search.retrievers.registry import RetrieverConfig, register


@register("my_method")
def build_my_method(cfg: RetrieverConfig, name: str):
    from my_package import MyRetriever
    return lambda: MyRetriever(index_root=cfg.index_root)
```

It immediately becomes an evaluation option:

```bash
python -m evaluation.run_eval \
  --dataset fixture \
  --retriever my_method \
  --runs-dir runs/my-method
```

### Add something else

| extension | start here |
|---|---|
| Corpus or task | [`evaluation/datasets.py`](evaluation/datasets.py) and [`corpus_build/`](corpus_build/) |
| Search, inspect, or read tool | [`agent_search/agent/tools/`](agent_search/agent/tools/) and [`prompts/tools.yaml`](agent_search/prompts/tools.yaml) |
| Agent instructions | [`agent_search/prompts/`](agent_search/prompts/) |
| Model provider | [`agent_search/models/backends.py`](agent_search/models/backends.py) or the `Model` protocol |
| Metric or answer evaluator | [`evaluation/metrics.py`](evaluation/metrics.py) and [`evaluation/llm_judge.py`](evaluation/llm_judge.py) |

## Data and reproducibility

The repository contains builders for BrowseComp-Plus and Wikipedia-based QA collections, together
with paired flat and structured variants used by the Sieve study. The paper's built corpora are
published on Hugging Face —
[`wshuai190/browsecomp-plus-structured-full`](https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full),
[`wshuai190/hotpotqa-structured`](https://huggingface.co/datasets/wshuai190/hotpotqa-structured),
and [`wshuai190/musique-structured`](https://huggingface.co/datasets/wshuai190/musique-structured)
— so they can be pulled instead of rebuilt; staging commands and the builders are documented in
[`corpus_build/README.md`](corpus_build/README.md).

Every run writes its resolved configuration, per-question outputs, agent actions, observations,
and cost fields under `runs/`. Re-running the same configuration resumes completed work. The
cluster launchers, judging workflow, statistics, and paper-table pipeline are documented in
[`docs/REPRODUCING.md`](docs/REPRODUCING.md).

Useful checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m evaluation.run_eval --help
python scripts/summarize_runs.py --help
```

Two test modules exercise the optional OpenAI Agents SDK adapter and skip themselves when that
SDK is not installed. With the full retrieval stack installed, the Pyserini/Lucene tests require
Java 21+ (see [`docs/REPRODUCING.md`](docs/REPRODUCING.md)); note that once the JVM starts it
swallows pytest's terminal output — pass `--junitxml=report.xml` if you need a machine-readable
result.

## Sieve and the accompanying paper

Sieve is a Boolean-filtered search–inspect–fetch strategy implemented using the library's general
interfaces. It combines fielded candidate selection, an interchangeable ranker, compact result
cards, and section-level reading. The library also supports all baselines and controls used to
evaluate it. See [`docs/SIEVE.md`](docs/SIEVE.md) for the method and
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the experimental workflow.

If you use Sieve or its released evaluation resources, please cite:

```bibtex
@misc{wang2026sieve,
  title         = {Search, Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents},
  author        = {Wang, Shuai and Chen, Haodong and Yin, Yu and Zhuang, Shengyao and
                   Koopman, Bevan and Zuccon, Guido},
  year          = {2026},
  eprint        = {2608.02751},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2608.02751}
}
```

## Project status

SkimSearchAgent is a research preview extracted from a large experimental codebase. The core
interfaces, reference retrievers, evaluation path, and paper configurations are tested, but public
APIs may still change before the first stable release. Issues and focused pull requests are
welcome.
