# Extending SkimSearchAgent

Every extension is a registration against a contract in
[`agent_search/core/interfaces.py`](../agent_search/core/interfaces.py). The harness does not
special-case the built-ins, so a registered component gets the same run record, metrics and
resume behaviour. Each section below is a complete, runnable example.
[`examples/plugin_strategy.py`](../examples/plugin_strategy.py) and
[`tests/test_extension_points.py`](../tests/test_extension_points.py) contain the same code.

| to add | register with | contract |
|---|---|---|
| a corpus or dataset | `agent_search.evaluation.datasets.register_dataset` | a loader returning `Instance` objects |
| a retriever or ranker | `agent_search.retrievers.registry.register` | `Retriever.index(units, key)` / `search(query, k)` |
| a tool | `agent_search.prompts.loader.register_tool` | a JSON-schema declaration rendered into the prompt |
| a toolset | `agent_search.prompts.loader.register_toolset` | a list of tool names |
| a strategy (condition) | `agent_search.prompts.registry.register_condition` | task template × toolset; registered as `agent_<name>` |
| a workspace (tool execution) | `agent_search.agent.retriever.register_workspace` | `run(name, args) -> str`, `tools`, `surfaced` |
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
`agent_search/evaluation/datasets.py`. The paper's corpus builders are in
[`corpus_build/`](../corpus_build/README.md).

```bash
skimsearchagent dataset=my_qa strategy=sieve_bm25 model=gpt-4o-mini
```

## 2. A retriever or ranker

```python
from agent_search.core import Retriever
from agent_search.retrievers.registry import RetrieverConfig, register

class MyRetriever(Retriever):
    name = "my_method"

    def index(self, units, key=None):         # key: corpus identity for on-disk caches; None = in memory
        self._ids = [u.doc_id for u in units]
        return self

    def search(self, query, k):
        return self._ids[:k]                  # doc ids, best first

@register("my_method")                        # heavy imports belong inside the builder
def build(cfg: RetrieverConfig, name: str):
    return lambda: MyRetriever()
```

`strategy=my_method` runs it as a retrieval-only condition with the same rank metrics as
`bm25_local`. To use it inside an agent, call it from a workspace (section 4). A persistent index
should be keyed by `key` and by `agent_search.corpus.fingerprint.corpus_fingerprint(units)`, so an
edited document is never served from a stale cache; the built-in retrievers do this.

## 3. A tool, a toolset, and a condition

A tool declaration is rendered into the system prompt. A toolset names the tools a condition
exposes. A condition binds a task template to a toolset and is registered as the strategy
`agent_<name>`.

```python
from agent_search.prompts.loader import register_tool, register_toolset
from agent_search.prompts.registry import register_condition

register_tool("title_lookup",
              description="Return every document whose title contains the given words.",
              parameters={"type": "object",
                          "properties": {"words": {"type": "string"}}, "required": ["words"]},
              manual="/abs/path/to/title_lookup_manual.md")           # optional per-tool manual
register_toolset("title_only", ["title_lookup"])
register_condition("title_agent", task="research", toolset="title_only")
```

`task` is a template name under `agent_search/prompts/tasks/` (`research` is the
document-research template; it contains the `{{tools}}` and `{{tool_manuals}}` placeholders) or
a path to your own `.md` file with the same front matter (`name`, `domain`, `message_format`).

Inside this repository the same three registrations are entries under `tools:` and `toolsets:` in
`agent_search/prompts/tools.yaml` and one line in `agent_search/prompts/conditions.yaml`.

To change only the prompt of an existing strategy, edit the template, or register a condition
that binds your template to the existing toolset:

```python
register_condition("my_sieve", task="/path/to/my_research.md", toolset="search_fetch_s")
```

## 4. A workspace: how the tools execute

A workspace answers the agent's tool calls. Requirements: a `tools` tuple, a `run(name, args)`
method that returns the observation text (an error is returned as text, never raised), a `seen`
attribute of type `OrderedSeen`, and a `surfaced` property returning the seen ids in order. The
`surfaced` list is the agent's retrieval ranking for the rank metrics.

```python
from agent_search.agent.retriever import WorkspaceContext, register_workspace
from agent_search.core import OrderedSeen, cap_tokens

class TitleWorkspace:
    tools = ("title_lookup",)

    def __init__(self, ctx: WorkspaceContext):
        self.units, self.ubyid = ctx.units, ctx.ubyid
        self.bm25 = ctx.bm25()                # shared engines, built once per corpus: bm25(), bql(), dense()
        self.seen = OrderedSeen()

    @property
    def surfaced(self):
        return list(self.seen)

    def run(self, name, args):
        if name != "title_lookup":
            return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."
        words = (args or {}).get("words", "").lower().split()
        hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
        self.seen.update(u.doc_id for u in hits)
        return "\n".join(f"{u.doc_id}  {u.title}: {cap_tokens(u.body, 64)}" for u in hits) or "no match"

register_workspace("title_arm", tools=("title_lookup",), builder=TitleWorkspace)
```

A condition whose toolset contains every tool in `tools=` uses this workspace; the registration
with the most marker tools wins. Text limits inside a tool are token caps (`cap_tokens`,
`count_tokens` in `agent_search.core.tokens`).

Run it:

```bash
skimsearchagent dataset=doc_fixture strategy=agent_title_agent model=gpt-4o-mini
```

## 5. A model provider

The model contract is a callable from OpenAI-style chat messages to the generated text. Tool
calls are part of the text.

```python
def my_generate(messages: list[dict]) -> str:
    return my_client.chat(messages)

from agent_search import research
research(question, docs, strategy="sieve_bm25", generate=my_generate)
```

To make a provider selectable by name on the command line (`model.name: my-model-x`), add a
matcher and a branch to `make_generate` in `agent_search/models/backends.py`; the OpenAI and
Gemini branches show the pattern. Set `.client` and `.model` attributes on the returned callable
if the forced-answer elicitation should reuse your client.

## 6. A metric or judge

Rank metrics are in `agent_search/evaluation/metrics.py` and are computed in
`run_eval._score_instance`; answer metrics are in `evaluation/doc_scoring.py::score_answer`. A new
metric is a function of `(prediction, gold, observations)` returning a float; add the call in
`score_answer` and the value is averaged into `results.json` (every numeric row field is). A judge
is a `generate(prompt) -> str` callable used by `llm_judge.judge_run_dir`.

## 7. A plugin package

Put the registrations in a module and expose it in one of two ways:

```toml
# pyproject.toml of your package
[project.entry-points."skimsearchagent.plugins"]
my_plugin = "my_pkg.skim_plugin"
```

```bash
SKIMSEARCHAGENT_PLUGINS=my_pkg.skim_plugin skimsearchagent dataset=my_qa strategy=agent_title_agent
```

The registry imports plugin modules after the built-ins and before any name is resolved.

## 8. Testing an extension

Follow `tests/test_extension_points.py`: register, then call
`agent_search.research(question, docs, strategy="agent_<name>", generate=scripted)` where
`scripted` returns fixed `<tool_call>` and `<answer>` strings. The result carries the answer, the
ranking, the per-step record and the token usage. No model, network or on-disk index is needed.
