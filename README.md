<div align="center">

# SkimSearchAgent

**A research framework for deep-search agents. Change one component, keep the rest of the experiment fixed.**

[![CI](https://github.com/ielab/skim-search-agent/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/ielab/skim-search-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](#install)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-00897B.svg)](LICENSE)

[Project page](https://ielab.github.io/skim-search-agent/) ·
[Paper](https://arxiv.org/abs/2608.02751) ·
[The library](#the-library) · [Install](#install) · [Run an experiment](#run-an-experiment) · [Change one thing](#change-one-thing) ·
[Python API](#python-api) · [Strategies](#strategies) · [Train a retriever](#train-a-retriever) ·
[Docs](#documentation)

</div>

<p align="center">
  <img src="docs/assets/skimsearchagent-overview.png" width="100%" alt="Corpora, models, strategies, retrievers and evaluators around one agent loop"/>
</p>

A deep-search agent answers a question by searching a collection over several steps. Which
retriever it uses, what a search result shows, how it reads a document, which agent backbone drives it,
and how the answer is scored are separate decisions. SkimSearchAgent makes each one a component
with a fixed interface and runs every combination through the same evaluation, so two experiments
differ only where you changed them. Every run writes the same record: the full trajectory, the
configuration, and the metrics.

## The library

One package, `agent_search`, one folder per kind of component. Each folder is a base class plus
one file per concrete class, so a new method is one new file. Read them in the order a query
travels:

| module | what it holds | the files |
|---|---|---|
| `corpus/` | documents and functions as units; an on-disk store for corpora too large for memory; fingerprints so a persisted index is never served against a corpus it was not built for | `units.py`, `docstore.py`, `fingerprint.py`, `code_repo.py` |
| `retrievers/` | engines that rank ids for a query: Lucene BM25, dense encoders (one file per family), the Boolean method BQL, Indri; and two compositions, a hybrid (any retrievers fused by one method) and a reranked retriever (one retriever's pool reordered by a reranker) | `lexical/`, `dense/`, `bql/`, `indri/`, `lucene/`, `fusion/`, `hybrid.py`, `rerankers/`, `reranked.py`, `engines.py`, `backend.py` |
| `snippets/` | how one hit is excerpted in a listing: the opening line, the best window for the query terms, or nothing; widths in model tokens | `opening.py`, `term_window.py`, `none.py` |
| `tools/` | the actions the agent backbone can call, one folder each with its declaration, code and manual; a tool names its engine kind and its snippet | `search_bm25/`, `search_dense/`, `search_hybrid/`, `search_reranked/`, `search_bql/`, `search_indri/`, `search_dedup/`, `search_bm25_dci/`, `visit/`, `fetch/`, `fetch_code/`, `get_document/`, `bash/`, `read/`, `grep/` |
| `tasks/` | what the backbone is asked to produce: the prompt template and the answer protocol | `research/`, `research_dedup/`, `codefix/`, `codefix_patch/` |
| `strategies/` | the named combinations a run selects: tools with their options and a harness, or a retrieval-only floor; `conditions.py` pairs a strategy with a task | `search_visit.py`, `search_fetch.py`, `autoread.py`, `sieve.py`, `indri.py`, `dci.py`, `dedup.py`, `codefix.py`, `rag.py`, `teams.py`, `retrieval_only.py` |
| `harness/` | how the backbone is put to work on a condition, one file each: ReAct (the default loop, the backbone picks each step), one-shot RAG (rank once, one call), plan-and-search (a team: a planner, one member agent per sub-question, a synthesizer) | `react.py`, `rag.py`, `plan_and_search.py` |
| `agent/` | the machinery a harness is built from: the backbone providers, the policies, the step loop, the forced answer, the Agents-SDK and Responses drivers, the run record | `backbone/`, `policies.py`, `loop.py`, `forced_answer.py`, `sdk_driver.py`, `responses_driver.py`, `record.py` |
| `evaluation/` | the evaluation: datasets, the runner, the metrics, the judge, the run record and its identity, the index prebuild | `datasets/`, `runner.py`, `run_eval.py`, `llm_judge.py`, `identity.py`, `build_indexes.py` |
| `training/` | the ITER recipe: trajectories to triples, retriever training and evaluation | `build_triples.py`, `retriever.py`, `retriever_eval.py` |

Document corpora rank on Lucene (BM25 through Pyserini, BQL and Indri on a fielded Lucene
index); a code repository is indexed in memory and re-indexed only for the files that change.
Every length limit is a token count. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains each
module; [docs/EXTENDING.md](docs/EXTENDING.md) shows how to add one of each.

## Install

```bash
git clone https://github.com/ielab/skim-search-agent.git && cd skim-search-agent
python -m pip install -e .                 # core: PyYAML only
python -m pip install -e ".[api]"          # OpenAI, Gemini, OpenAI-compatible servers
python -m pip install -e ".[retrieval]"    # Lucene (BM25, BQL, Indri), dense and hybrid retrieval (Java 21+, torch)
```

Other extras: `eval` (dataset staging, statistics), `serve` (vLLM), `demo-live`, `dev`
(tests), `train` (retriever training, separate environment), `all`. Pins match
[`requirements.txt`](requirements.txt), the environment used for the paper.

Every document run needs the `retrieval` extra: document corpora rank on Lucene only (BM25
through Pyserini, and the structured index behind BQL and Indri). The core install runs code
repositories, which are indexed in memory. Point `JAVA_HOME` at a JDK 21 or newer before
running anything that touches Lucene, the test suite included. Pyserini starts a JVM on
import, and an older Java aborts the process without a message.

## Run an experiment

One YAML file is one complete setting. The file lists every knob its strategy reads: dataset,
agent backbone, token budgets, listing depths, retrieval engines, scoring, output. A `search_visit` file
has no dense-model keys; a `sieve` file has no listing depths.

```bash
skimsearchagent run configs/smoke_doc_fixture_sieve_bm25.yaml        # scripted policy, no keys
export OPENAI_API_KEY=...
skimsearchagent run configs/doc_fixture_sieve_bm25_gpt4omini.yaml    # a real backbone
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml     # the paper's setting, scripted policy (no backbone named)
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml model.name=gpt-4o output.runs_dir=runs/gpt4o   # the same with a backbone
skimsearchagent validate configs/paper/hotpotqa_structured_sieve.yaml   # what it needs, what it will run
skimsearchagent template paper sieve > configs/mine.yaml              # a complete file for one strategy, to edit
```

The run directory holds `config.json` (the file's content plus every resolved knob),
`rows.jsonl` (one full trajectory per question) and `results.json` (metrics). Rerunning the same
command resumes. Rerunning a different setting into the same directory is refused.

For one-off runs every knob is also a `key=value` argument:

```bash
skimsearchagent dataset=doc_fixture strategy=search_visit model=gpt-4o-mini snippet_tokens=64
```

## Change one thing

Each block changes one component and leaves the rest of the experiment as it was. The blocks
follow the module table above: the strategy, the agent backbone, the harness, the retriever, the
snippet, the tool, the task, the corpus, the dataset.

**The strategy** (one word in the experiment file, or on the command line):

```yaml
strategy: search_visit        # sieve, sieve_bm25, search_fetch, autoread, dci, indri, ... (see Strategies)
```

**The agent backbone**, the language model that drives the agent. The experiment file's section
is called `model`; it takes any OpenAI-compatible server, OpenAI, Gemini, or in-process vLLM:

```yaml
model:
  name: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
  backend: api                # a served backbone
  api_base: http://localhost:8000/v1
```

From Python the backbone is any callable that maps chat messages to text:

```python
def my_backbone(messages: list[dict]) -> str:
    return my_client.chat(messages)          # returns the generation, tool calls included

research(question, docs, strategy="sieve_bm25", generate=my_backbone)
```

**The harness**, how the backbone is put to work. ReAct is the default; a strategy names
another one, such as a team whose members are conditions:

```python
from agent_search.harness.plan_and_search import PlanAndSearch
from agent_search.strategies.base import Strategy, register_strategy

register_strategy(Strategy(name="plan_and_search_visit_dense", description="a planner, dense search-visit agents, a synthesizer",
                           harness=PlanAndSearch(searcher="search_visit_dense")))
# run it: strategy=plan_and_search_visit_dense
```

**The retriever**, as a new engine or as a retrieval-only floor:

```python
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.registry import register

class MyRetriever(Retriever):
    name = "my_method"
    def index(self, units, key=None): self._ids = [u.doc_id for u in units]; return self
    def search(self, query, k): return self._ids[:k]

register("my_method")(lambda cfg, name: (lambda: MyRetriever()))
# run it: strategy=my_method (retrieval-only), or give it an engine kind so a tool can use it
```

A hybrid is any retrievers fused by one method (`retrieval.hybrid_retrievers`,
`retrieval.hybrid_fusion`); a reranked retriever is one retriever's pool reordered by a reranker
(`retrieval.rerank_base`, `retrieval.rerank_model`).

**The dense retriever**, one setting for Sieve's ranker, its fallback, and every dense baseline:

```yaml
retrieval:
  dense_model: models/my-retriever     # a hub id or a checkpoint trained below
  dense_query_style: i9                # how the query is written from the agent's history
```

**The snippet**, what a listing shows under each hit. A tool takes a method from
`agent_search/snippets/`; widths are model tokens:

```python
from agent_search.snippets import TermWindow
from agent_search.tools.search_bm25.tool import SearchBm25

SearchBm25(name="bm25q_search", snippet=TermWindow())      # the best window for the query terms
```

**A tool** (its declaration and its code in one class), **a strategy** (which tools, under which
names, with which options) and **a condition** (a task with that strategy):

```python
from agent_search.tools.base import Tool
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.strategies.conditions import condition

class TitleLookup(Tool):
    name = "title_lookup"
    description = "Documents whose title contains the words."
    parameters = {"type": "object", "properties": {"words": {"type": "string"}}, "required": ["words"]}
    def run(self, args):                          # self.units, self.ubyid, self.state, self.engine[...]
        hits = [u for u in self.units if all(w in (u.title or "").lower() for w in args["words"].lower().split())]
        self.state.seen.update(u.doc_id for u in hits)   # first-seen order = the agent's ranking
        return "\n".join(f"{u.doc_id}  {u.title}: {u.body}" for u in hits) or "no match"

register_strategy(Strategy(name="title_only", description="look up by title", tools=(TitleLookup(name="title_lookup"),)))
condition("title_agent", task="research", strategy="title_only")
# run it: strategy=title_agent
```

**The task**, what the backbone is asked to produce: edit `agent_search/tasks/research/prompt.md`
(the default prompt; the Sieve paper's own prompt is `research_paper/`),
or write a task with your own template and pair it with an existing strategy:

```python
from agent_search.tasks.base import Task, register_task
from agent_search.strategies.conditions import condition

@register_task
class MyTask(Task):
    name, domain, terminal = "my_task", "general", "answer"
    prompt_file = "/path/to/my_task.md"        # front matter + body with {{tools}} and {{tool_manuals}}

condition("my_sieve", task="my_task", strategy="sieve_bm25")
# run it: strategy=my_sieve
```

**The corpus**, from Python, with your own documents:

```python
from agent_search import research

docs = [{"_id": "d1", "title": "Treaty of Guadalupe Hidalgo",
         "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."}]
result = research("Which treaty ended the Mexican-American War?", docs,
                  strategy="sieve_bm25", model="gpt-4o-mini")
print(result.answer, result.ranking, result.usage)
```

**A dataset**:

```python
from agent_search.evaluation.datasets import Instance, register_dataset

@register_dataset("my_qa", domain="general")
def load_my_qa(limit=None, corpus_limit=None):
    return [Instance(instance_id="my_qa__1", repo="local/my_qa", base_commit="0" * 40,
                     problem_statement="Which treaty ended the Mexican-American War?", patch="",
                     docs=docs, gold_doc_ids={"d1"}, answer="Treaty of Guadalupe Hidalgo", corpus_id="my_qa")]
# run it: dataset=my_qa
```

Registrations in an installed package load through the `skimsearchagent.plugins` entry-point
group or `SKIMSEARCHAGENT_PLUGINS=my_module`. [docs/EXTENDING.md](docs/EXTENDING.md) has the
complete version of each block, including metrics and judges.

## Python API

```python
from agent_search import research, build_agent

result = research(question, docs, strategy="sieve_bm25", model="gpt-4o-mini")
result.answer        # the answer span
result.ranking       # documents surfaced, in first-seen order
result.steps         # per step: tool, arguments, full observation, tokens, hit_ids, read_ids
result.usage         # llm_calls, prompt, completion and cached tokens

agent = build_agent("sieve_bm25", model="gpt-4o-mini")   # reuse one index for many questions
agent.index(units, key="my_corpus")
agent.search(question, k=10)
```

## Strategies

| family | `strategy=` | what the agent does |
|---|---|---|
| Retrieval-only | `bm25`, `dense`, `bql`, `grep`, `hybrid`, `reranked` | rank once, no agent loop, no backbone; `reranked` reorders the pool of the retriever named in `retrieval.rerank_base` with the reranker in `retrieval.rerank_model`; `hybrid` fuses the retrievers named in `retrieval.hybrid_retrievers` with `retrieval.hybrid_fusion` (`rrf` or `interpolation`) |
| One-shot RAG | `rag_bm25`, `rag_dense`, `rag_hybrid` | rank once, put the top five documents in one prompt, one backbone call |
| Search–Visit | `search_visit`, `search_visit_dense`, `search_visit_hybrid`, `search_visit_reranked`, `search_visit_snippets` | read a result list, open whole documents |
| Search–AutoRead | `autoread`, `autoread_dense`, `autoread_hybrid` | every search returns full text |
| Direct corpus interaction | `dci`, `bounded_dci` | shell commands over exported files, optionally within a BM25 working set |
| Search–Fetch | `search_fetch`, `search_fetch_dense`, `search_fetch_hybrid`, `search_fetch_bm25_plain`, `search_fetch_dense_plain` | result cards with snippets (or, for the plain arms, without), then named sections |
| **Sieve** | `sieve`, `sieve_bm25`, `sieve_dense`, `sieve_nosnip`, `sieve_nomanual`, `sieve_card`, `sieve_syntax`, `sieve_noconstruct`, `sieve_nohowto`, `sieve_nofields`, `sieve_nofetch`, `sieve_nohops`, `sieve_noexamples`, `sieve_nomistakes`, `sieve_plain`, `sieve_v2`, `sieve_visit`, `sieve_visit_fused`, `sieve_visit_dense` | BQL candidate filtering, one ranking model, result cards, section fetch (or whole documents) |
| Structured control | `indri`, `indri_plain`, `indri_visit` | Indri-QL retrieval with cards and section fetch (or whole documents) |
| Code localization | `codefix`, `codefix_grep`, `codefix_patch` | search or grep a repository, read functions, propose a fix (`dataset=code_fixture`) |
| ITER search | `dedup_bm25`, `dedup_dense` | ITER's tool setup; see [docs/ITER.md](docs/ITER.md) |
| Multi-agent | `plan_and_search`, `plan_and_search_visit` | a planner splits the question, one agent per sub-question (Sieve, or search-visit), a synthesizer answers; every member episode is recorded |

Every index is built once per dataset, before any run. Build the dense embedding cache with
`skimsearchagent-build-indexes --dataset <name> --retriever dense`, the Lucene BM25 index with
`--retriever bm25_pyserini`, and the Lucene structured index behind BQL and Indri with
`--retriever search_lucene`. A run builds what it needs in its first step; a missing dense
cache stops the run before the first episode and prints the build command. A code repository
is indexed in memory and re-indexed only for the files that change.

A corpus too large for memory is served from disk and searched through prebuilt indexes named
in the file; nothing is encoded during a run. See [docs/ITER.md](docs/ITER.md).

## Train a retriever

Every run is a trajectory, so it is also training data. Triples with tiered negatives come out
of any run directory. A patched FlagEmbedding trainer fits a dense retriever on them. The
checkpoint plugs back in as the dense model of any strategy, served with the query style,
instruction and precision it was trained with:

```bash
skimsearchagent-build-triples --runs runs/... --dataset hotpotqa_structured --out train_data/hotpotqa_i2.jsonl --query-style i2 --labeller oracle
skimsearchagent-train-retriever template > train.yaml
sbatch --export=ALL,TRAIN=train.yaml,TRAIN_ENV=$PWD/envs-train scripts/slurm/train_retriever.sbatch
skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml retrieval.dense_model=models/my-retriever retrieval.dense_query_style=i2
```

[docs/TRAINING.md](docs/TRAINING.md) is the recipe; [docs/ITER.md](docs/ITER.md) is the paper it
comes from, with its backbones, released retrievers, datasets and the verified runs.

## Reproducibility

- `config.json` records the experiment file, every knob, the composed prompt's hash, the package
  version, the token ruler and the git revision.
- A run directory holds one setting. Resuming a different one into it is refused.
- `model.seeds: [0, 1, 2]` writes one `seed=N` directory per seed.
- Persistent indexes are keyed by corpus identity and a content fingerprint.
- The paper's corpora are on Hugging Face
  ([`wshuai190/browsecomp-plus-structured-full`](https://huggingface.co/datasets/wshuai190/browsecomp-plus-structured-full),
  [`wshuai190/hotpotqa-structured`](https://huggingface.co/datasets/wshuai190/hotpotqa-structured),
  [`wshuai190/musique-structured`](https://huggingface.co/datasets/wshuai190/musique-structured)).
  Staging and building: [`corpus_build/`](corpus_build/README.md).
- Cluster jobs (serving, index builds, training): [`scripts/slurm/`](scripts/slurm/README.md).

```bash
python -m pip install -e ".[dev]"
python -m pytest -q                            # no GPU or API key needed; the document tests need the retrieval extra and Java 21, and skip without them
```

## Demo

```bash
pip install -e ".[demo-live]"
python demo/server.py          # http://localhost:8008/
```

Ask a question and watch Sieve and Search–Visit answer it side by side, over a 250-document
subsample of BrowseComp-Plus, with every token metered. Your OpenAI key stays in the browser's
session storage and is sent per request to OpenAI; the server does not store or log it. See
[`demo/`](demo/README.md).

## Documentation

| document | content |
|---|---|
| [docs/EXTENDING.md](docs/EXTENDING.md) | every extension point with a complete code example |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | the experiment-file schema, every flag and knob |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | modules, contracts, one episode end to end |
| [docs/TRAINING.md](docs/TRAINING.md) | retriever training from run records |
| [docs/SIEVE.md](docs/SIEVE.md) | the Sieve paper: the method and its ablations |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | the paper from a fresh clone: environment, data, indexes, cells, sharded runs, judging |
| [docs/ITER.md](docs/ITER.md) | the ITER paper: its search tools, backbones, retrievers, datasets, training and the verified runs |
| [CONTRIBUTING.md](CONTRIBUTING.md) | development setup and how to add components |

## Papers

Two papers run on this library. Each has its own page; the README only points at them.

### Sieve

[docs/SIEVE.md](docs/SIEVE.md). A Boolean-filtered
search, inspect, fetch strategy: fielded candidate selection (BQL), one ranking model, compact
result cards with query-biased snippets, and section-level reading. On BrowseComp-Plus, HotpotQA
and MuSiQue it matched or improved accuracy while reading 30 to 51% fewer tokens than
Search-Visit.

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

### ITER

[docs/ITER.md](docs/ITER.md), https://github.com/ielab/ITER. Interaction-aware retrieval for
agentic search: a dense retriever trained from search-agent trajectories, conditioned on the
agent's earlier searches and trained to return documents it has not read yet. The library
carries its search tools, its training recipe, its released checkpoints and its evaluation sets.

```bibtex
@misc{chen2026iter,
  title         = {ITER: Interaction-Aware Retrieval for Agentic Search},
  author        = {Chen, Haodong and Wang, Shuai and Yin, Yu and Zhuang, Shengyao and
                   Zuccon, Guido and Leelanupab, Teerapong},
  year          = {2026},
  eprint        = {2608.27912},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2608.27912}
}
```
