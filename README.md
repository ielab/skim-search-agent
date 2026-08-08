<div align="center">

# SkimSearchAgent

**A composable library for building and evaluating deep-research agents over your own corpus.**

Bring the documents, model, and search strategy. SkimSearchAgent supplies the agent loop,
retrieval and reading primitives, reproducible evaluation, and the interfaces that connect them.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](#quickstart)
[![Pluggable](https://img.shields.io/badge/corpora%20%7C%20models%20%7C%20methods-pluggable-00897B.svg)](#what-can-be-swapped)

**[🌐 Project page](https://ielab.github.io/skim-search-agent/)** ·
[Paper](https://arxiv.org/abs/2608.02751) ·
[Quickstart](#quickstart) ·
[Demo](#demo) ·
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

SkimSearchAgent requires Python 3.10+. The commands below need no Java, GPU, staged dataset, or
API key until you bring a model.

```bash
git clone https://github.com/ielab/skim-search-agent.git && cd skim-search-agent
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

Prefer to *watch* one first? See the [Demo](#demo) below — a local server and your own
API key, no GPU.

Run a complete research-agent experiment as one command — every knob is a `key=value`:

```bash
# a full agent episode on the built-in toy corpus (no model, no keys: scripted stub policy)
python run.py dataset=fixture strategy=sieve_bm25

# the same episode driven by a real model
export OPENAI_API_KEY=...
python run.py dataset=fixture strategy=sieve_bm25 model=gpt-4o-mini limit=1

# swap ONE word to run a different strategy — same corpus, model, budgets, and scoring
python run.py dataset=fixture strategy=search_visit model=gpt-4o-mini limit=1

# or run an open-weight model in-process with vLLM (GPU required, no API key)
python run.py dataset=fixture strategy=sieve_bm25 \
    model=openai/gpt-oss-20b backend=vllm limit=1
```

Tool-surface knobs — the snippet window, read budgets, listing depths, engine choice — are
`key=value` too, and each is recorded in the run's `config.json`:

```bash
# widen the query-biased snippet the agent skims before choosing what to read
python run.py dataset=fixture strategy=sieve_bm25 snippet_tokens=64

# the paper's read budget, explicitly
python run.py dataset=fixture strategy=search_visit max_visit_tokens=12000
```

`strategy` accepts friendly names (`search_visit`, `search_fetch`, `autoread`, `dci`,
`bounded_dci`, `sieve`, `sieve_bm25`, `sieve_dense`, `indri`, ... — `python run.py --help`
lists them all) or any raw registered retriever name. Everything else
(`model=`, `limit=`, `backend=`, `runs_dir=`, ...) is forwarded to the underlying
`python -m evaluation.run_eval`, which remains the fully-flagged entry point.

Each run writes a resumable trace (`rows.jsonl` + `results.json`) with the answer, the agent's
searches and fetches, tokens, and calls. Strategies with a dense channel (`sieve`,
`search_visit_dense`, ...) additionally need a one-time embedding cache —
`bash scripts/build_indexes.sh` after staging data — and for document-scale corpora install the
full stack:

```bash
python -m pip install -e ".[all]"
```

The package extras use the same versions as [`requirements.txt`](requirements.txt), which records
the exact Python 3.10 environment used for the paper. After staging a real collection
([`corpus_build/`](corpus_build/README.md)), the same one-liner scales up. For open models,
`backend=vllm` loads small and mid-size models in-process; for large backbones (the paper's
30B-A3B models), serve them with `vllm serve <model>` and point the same command at the server
with `backend=api api_base=http://localhost:8000/v1` — this served path is how every paper
experiment ran. API keys and common knobs are documented in [`.env.example`](.env.example) — copy to
`.env`, fill in what you need, and load with `set -a; source .env; set +a`.

```bash
python run.py dataset=hotpotqa_structured strategy=sieve model=gpt-4o-mini limit=20
```

## Demo

**The live playground** — ask your own questions and watch the real agent loop work:

```bash
pip install -e ".[demo-live]"
python demo/server.py          # -> http://localhost:8008/
```

A little agent hops between its tool stations on a stage — and the two strategies have
different boards: Sieve runs the paper's own pipeline (Search → Inspect → Fetch § → Answer)
while the baseline has no inspect stage at all (Search → Visit doc → Answer), the
250-document collection wall lights up as documents are surfaced and read, and a live meter
counts every token, cached-token discount and fraction of a cent. Bring your own OpenAI key (a
run is capped at 20 steps — well under 1¢ on gpt-4o-mini; the key is used per-request and never
stored or logged).
**Race mode** runs Sieve vs the Search-Visit baseline side by side on the same question and ends
in a head-to-head chart — the paper's claim, live. The collection is a curated subsample of real
BrowseComp-Plus documents; see [`demo/`](demo/README.md) for how it was built.

There is also a **project page** at [`docs/index.html`](docs/index.html), published to
<https://ielab.github.io/skim-search-agent/> — see [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

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
| Structured-retrieval control | `agent_research_indri_snip` | Indri-QL structured retrieval with result cards and section fetch |

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

## The team

<p align="center">
  <img src="docs/assets/team.png" width="100%" alt="The six authors: Shuai Wang, Haodong Chen, Yu Yin, Shengyao Zhuang, Bevan Koopman and Guido Zuccon."/>
</p>

<div align="center">

[Shuai Wang](https://shuaiwang.io)<sup>1</sup> ·
[Haodong Chen](https://donovan0243.github.io/)<sup>1</sup> ·
[Yu Yin](https://yinyubb.github.io/)<sup>1</sup> ·
[Shengyao Zhuang](https://arvinzhuang.github.io/) ·
[Bevan Koopman](https://bevankoopman.github.io/)<sup>2,1</sup> ·
[Guido Zuccon](https://ielab.io/people/guido-zuccon.html)<sup>1</sup>

<sup>1</sup>[ielab](https://ielab.io), The University of Queensland ·
<sup>2</sup>Australian e-Health Research Centre, CSIRO

**[Project page](https://ielab.github.io/skim-search-agent/)** ·
**[Paper](https://arxiv.org/abs/2608.02751)**

</div>
