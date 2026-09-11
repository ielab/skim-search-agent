# Extending SkimSearchAgent

Every extension is a registration against a base class that lives next to its family:
`agent_search/retrievers/base.py` (retrievers), `agent_search/agent/backbone/base.py` (model
providers), `agent_search/agent/policies.py` (policies), `agent_search/tools/base.py` (tools)
and `agent_search/tasks/base.py` (tasks). The evaluation does not
special-case the built-ins, so a registered component gets the same run record, metrics and
resume behaviour. Each section below is a complete, runnable example.
[`examples/plugin_strategy.py`](../examples/plugin_strategy.py) and
[`tests/test_extension_points.py`](../tests/test_extension_points.py) contain the same code.

| to add | register with | contract |
|---|---|---|
| a corpus or dataset | `agent_search.evaluation.datasets.register_dataset` | a loader returning `Instance` objects |
| a retriever or ranker | `agent_search.retrievers.registry.register` | `Retriever.index(units, key)` / `search(query, k)` |
| a tool | a `Tool` subclass (`agent_search.tools.base`) | declaration + `run(args) -> str` + an optional manual |
| a task | a `Task` subclass (`agent_search.tasks.base`) with a `prompt.md` | the template, the domain, the terminal |
| a strategy | `agent_search.strategies.base.register_strategy` | tools with their exposed names and options |
| a condition | `agent_search.strategies.conditions.condition` | task x strategy; runnable by name |
| a model provider | any `messages -> text` callable | `Model` |
| a policy | any object with `propose(task, history) -> str` | `Policy` |
| a metric or judge | a function over rows | `evaluation/metrics.py`, `doc_scoring.py`, `llm_judge.py` |

## 1. A corpus or dataset

A document is a dict. `units_from_documents` reads these keys:

```python
{"_id": "d1",                        # or "id" / "doc_id"
 "title": "Treaty of Guadalupe Hidalgo",
 "text": "The Treaty ... in 1848.",  # or "body" / "contents"
 "sections": [{"heading": "(intro)", "text": "..."}, {"heading": "Aftermath", "text": "..."}],  # optional
 "url": "...", "author": "...", "date": "1848-02-02"}   # any other key becomes metadata (a BQL field)
```

A dataset is a loader `(limit, corpus_limit) -> list[Instance]`. Every instance of a shared-corpus
dataset carries the same `docs` list.

```python
from agent_search.evaluation.datasets import Instance, register_dataset

@register_dataset("my_qa", domain="general")            # field_profile="wiki" for sectioned documents
def load_my_qa(limit=None, corpus_limit=None):
    docs = [...]
    return [Instance(instance_id="my_qa__1", repo="local/my_qa", base_commit="0" * 40,
                     problem_statement="Which treaty ended the Mexican-American War?",
                     patch="", docs=docs, gold_doc_ids={"d1"},
                     answer="Treaty of Guadalupe Hidalgo", corpus_id="my_qa")]
```

`corpus_id` names the persistent index caches; keep it stable. For a BEIR-style directory
(`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) call `_load_beir_style` from
`agent_search/evaluation/datasets/beir.py`. The paper's corpus builders are in
[`corpus_build/`](../corpus_build/README.md).

```bash
skimsearchagent dataset=my_qa strategy=sieve_bm25 model=gpt-4o-mini
```

## 2. A retriever or ranker

```python
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.registry import RetrieverConfig, register

class MyRetriever(Retriever):
    name = "my_method"

    def index(self, units, key=None):         # key: corpus identity for on-disk caches
        self._ids = [u.doc_id for u in units]
        return self

    def search(self, query, k):
        return self._ids[:k]                  # doc ids, best first

@register("my_method")                        # heavy imports belong inside the builder
def build(cfg: RetrieverConfig, name: str):
    return lambda: MyRetriever()
```

`strategy=my_method` runs it as a retrieval-only condition with the same rank metrics as
`bm25`. To use it inside an agent, give it to a tool (section 3). A persistent index
should be keyed by `key` and by `agent_search.corpus.fingerprint.corpus_fingerprint(units)`, so an
edited document is never served from a stale cache; the built-in retrievers do this.

### A dense encoder family

Dense retrievers are one base class and one file per encoder family
(`agent_search/retrievers/dense/`: `bge.py`, `coderank.py`, `qwen3_embedding.py`, `trained.py`).
A family states its own query prefix, pooling, precision and lengths; nothing is looked up in a
table. To add one, write a file with a subclass, say which model ids it serves, and register it:

```python
from agent_search.retrievers.dense import DenseRetriever, register_family

@register_family
class E5Retriever(DenseRetriever):
    query_prefix = "query: "          # what the model was trained with; documents get no prefix
    pooling = "mean"                  # None when the checkpoint ships its own sentence-transformers config
    default_dtype = "float16"

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return model_id.startswith("intfloat/e5-")
```

`DenseRetriever("intfloat/e5-large-v2")` now returns an `E5Retriever`, and every dense arm
(Sieve's ranker and fallback, the dense and hybrid baselines, Indri's belief) uses it through
`retrieval.dense_model`. A checkpoint trained with `skimsearchagent-train-retriever` needs no
family: `TrainedRetriever` reads its serving note.

### A fusion method

A hybrid retriever is a list of retrievers and a fusion method. The methods live one file each
under `agent_search/retrievers/fusion/`; a new one is a `Fusion` subclass that turns ranked
`(doc_id, score)` lists into one ranking:

```python
from agent_search.retrievers.fusion import Fusion, register_fusion

@register_fusion
class CombSum(Fusion):
    name = "combsum"
    needs_scores = True                       # False when only ranks are read

    def fuse(self, rankings, k=None):
        total = {}
        for ranking in rankings:
            for doc_id, score in ranking:
                total[doc_id] = total.get(doc_id, 0.0) + score
        ranked = sorted(total, key=lambda d: (-total[d], d))
        return ranked[:k] if k is not None else ranked
```

Then, in an experiment file: `retrieval.hybrid_retrievers: bm25,dense` (any engine kinds),
`retrieval.hybrid_fusion: combsum`. Every hybrid strategy (`search_visit_hybrid`,
`search_fetch_hybrid`, `autoread_hybrid`, `rag_hybrid`) and the `hybrid` floor use them.

### A reranker

A reranked retriever is one retriever and one reranker. The rerankers live one file each under
`agent_search/retrievers/rerankers/`; a new one is a `Reranker` subclass that scores
`(doc_id, text)` candidates for a query. A listwise reranker that asks the run's model to order
the candidates would be the next file in that folder:

```python
from agent_search.retrievers.rerankers import Reranker, register_reranker

@register_reranker
class TitleLength(Reranker):
    name = "title_length"

    def rerank(self, query, candidates, k=None):
        ranked = sorted(((d, float(len(t))) for d, t in candidates), key=lambda x: (-x[1], x[0]))
        return ranked[:k] if k is not None else ranked
```

Then, in an experiment file: `retrieval.rerank_base: bm25` (any engine kind),
`retrieval.rerank_method: title_length`, `retrieval.rerank_pool: 100`. The `reranked` floor and
the `search_visit_reranked` strategy use them; the shipped method is `cross_encoder`
(`retrieval.rerank_model`, default `BAAI/bge-reranker-v2-m3`). A reranker reads the documents
during the run, so it costs a forward pass per candidate per search.

## 3. A tool

A tool is one action the agent can call. It owns three things: its declaration (the name the
model sees, the description, the JSON parameters), its code (`run(args)` returns the observation
text; an error is returned as text, never raised) and, when the agent has to learn a syntax, its
manual (a Markdown file rendered into the prompt after the declarations). The built-ins live one
folder each under `agent_search/tools/`; a plugin's tool is just the class.

```python
from agent_search.tokens import cap_tokens
from agent_search.tools.base import Tool

class TitleLookup(Tool):
    name = "title_lookup"
    description = "Return every document whose title contains ALL the given words."
    parameters = {"type": "object", "properties": {"words": {"type": "string"}},
                  "required": ["words"]}
    # engines = ("bm25",)          # engines to build for this tool; reachable as self.engine["bm25"]
    # manual = "title_lookup.md"   # a file next to the class; a dict maps field profiles to files

    def run(self, args: dict) -> str:
        words = (args or {}).get("words", "").lower().split()
        hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
        self.state.seen.update(u.doc_id for u in hits)     # first-seen order = the ranking
        return "\n".join(f"{u.doc_id}  {u.title}: {cap_tokens(u.body, 48)}" for u in hits) or "no match"
```

A bound tool sees `self.units`, `self.ubyid` (doc id to unit), `self.state` (the episode: `seen`,
`last_hits`, `listing`, `reads`, `scratch`), `self.engine` (the engines it declared, built once
per corpus and shared with the other tools), `self.files` (the repository files, when
`needs_files = True`) and `self.corpus_key`. Options are class attributes a strategy overrides by
keyword (`SearchBm25(name="bm25q_search", query_biased=True)`); `on_bind()` runs once per episode.
Text limits inside a tool are token caps (`cap_tokens`, `count_tokens` in `agent_search.tokens`).

The engines a tool can name are the kinds in `agent_search/retrievers/engines.py`: `bm25`,
`dense`, `bql`, `bql_fused`, `bql_dense`, `bql_plain`, `indri`. To use a retriever registered in
section 2 from a tool, add a kind for it there.

### A snippet method

The excerpt a listing shows under each hit is a `Snippet` from `agent_search/snippets/`, one
file per method: `OpeningLine` (the first tokens), `TermWindow` (the best window for the query
terms), `NoSnippet` (structure only). A search tool takes one as `snippet=`, so a strategy
decides the presentation:

```python
from agent_search.snippets import Snippet, register_snippet

@register_snippet
class LastLine(Snippet):
    name = "last_line"

    def render(self, u, terms=None, width=32):
        toks = ((u.body if u.body is not None else u.code) or "").split()
        return " ".join(toks[-width:])

register_strategy(Strategy(name="search_visit_tail", description="search (BM25), last lines as snippets",
                           tools=(SearchBm25(name="bm25_search", snippet=LastLine()), Visit(name="visit"))))
```

The width is always `SNIPPET_TOKENS` tokens; a method never caps by characters.

## 4. A task

A task is the goal and the answer protocol: a folder with `prompt.md` (front matter `name`,
`domain`, `message_format`, `terminal`; a body with the `{{tools}}` and `{{tool_manuals}}`
placeholders) and `task.py` (a `Task` subclass naming the folder). The four built-ins are under
`agent_search/tasks/`. A plugin task points `prompt_file` at its own file:

```python
from agent_search.tasks.base import Task, register_task

@register_task
class TitleTask(Task):
    name = "title_qa"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"                      # <answer> ends the episode; "fix" and "patch" are the code terminals
    prompt_file = "/abs/path/to/title_qa.md"
```

## 5. A strategy and a condition

A strategy is a combination of tools with their options, under the names the prompt should show.
A condition pairs a task with a strategy and is what a run names (`strategy=` in an experiment
file or on the command line); it is registered as the retriever `agent_<name>` at the same time.

```python
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.strategies.conditions import condition

register_strategy(Strategy(name="title_only", description="look documents up by title words",
                           tools=(TitleLookup(name="title_lookup"),)))
condition("title_agent", task="research", strategy="title_only")
```

A strategy with no tools is a floor (`retrieval_only.py`, `retriever=`: rank once, no model).
Every other strategy has a harness, ReAct unless it says otherwise.

### A harness

A harness is how the model is put to work on a condition (`agent_search/harness/`, one file
each). It gets a `HarnessContext` (the condition, the corpus, the run's engines, `generate`,
and `policy_for(condition)` to make a policy) and returns a `HarnessResult`: a `Trajectory`
(the record every run writes), the documents surfaced, and the member results of a team.
`react.py` is the default loop; `rag.py` ranks once and asks once; `plan_and_search.py` is a
team: a planner splits the question, one member condition runs per sub-question through its own
harness, a synthesizer answers. A team with a different member is one line:

```python
from agent_search.harness.plan_and_search import PlanAndSearch
from agent_search.strategies.base import Strategy, register_strategy

register_strategy(Strategy(name="plan_and_search_dense",
                           description="a planner, one dense search-visit agent per sub-question, a synthesizer",
                           harness=PlanAndSearch(searcher="search_visit_dense", max_subquestions=4)))
```

A harness with different roles is a new file: subclass `Harness`, list the member strategies in
`members` (the run builds their engines), put its own prompt texts in `prompts()` (they are
hashed into the run identity), and write `run(question, ctx)`, building the `Trajectory` with
`trajectory_from_steps`. The record then carries the member trajectories under `members`, the
union of their surfaced documents, and the steps you return; the evaluation judges it like any
agent. Members can differ in strategy; they share the run's model.

The paper's condition names are aliases in `agent_search/strategies/paper.py`
(`alias("research_snip", "research", "sieve_bm25")`). To change only the prompt of an existing
strategy, write a task with your template and pair it with the strategy:

```python
condition("my_sieve", task="title_qa", strategy="sieve_bm25")
```

Run it:

```bash
skimsearchagent dataset=doc_fixture strategy=title_agent model=gpt-4o-mini
```

## 6. A model provider

The model contract is a callable from OpenAI-style chat messages to the generated text. Tool
calls are part of the text.

```python
def my_generate(messages: list[dict]) -> str:
    return my_client.chat(messages)

from agent_search import research
research(question, docs, strategy="sieve_bm25", generate=my_generate)
```

To make a provider selectable by name on the command line (`model.name: my-model-x`), add a
matcher and a branch to `make_generate` in `agent_search/agent/backbone/__init__.py`; the OpenAI and
Gemini branches show the pattern. Set `.client` and `.model` attributes on the returned callable
if the forced-answer elicitation should reuse your client.

## 7. A metric or judge

Rank metrics are in `agent_search/evaluation/metrics.py` and are computed in
`run_eval._score_instance`; answer metrics are in `evaluation/doc_scoring.py::score_answer`. A new
metric is a function of `(prediction, gold, observations)` returning a float; add the call in
`score_answer` and the value is averaged into `results.json` (every numeric row field is). A judge
is a `generate(prompt) -> str` callable used by `llm_judge.judge_run_dir`.

## 8. A plugin package

Put the registrations in a module and expose it in one of two ways:

```toml
# pyproject.toml of your package
[project.entry-points."skimsearchagent.plugins"]
my_plugin = "my_pkg.skim_plugin"
```

```bash
SKIMSEARCHAGENT_PLUGINS=my_pkg.skim_plugin skimsearchagent dataset=my_qa strategy=title_agent
```

The registry imports plugin modules after the built-ins and before any name is resolved.

## 9. Testing an extension

Follow `tests/test_extension_points.py`: register, then call
`agent_search.research(question, docs, strategy="<condition>", generate=scripted)` where
`scripted` returns fixed `<tool_call>` and `<answer>` strings. The result carries the answer, the
ranking, the per-step record and the token usage. No model, network or on-disk index is needed.
