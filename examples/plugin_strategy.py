#!/usr/bin/env python
"""A whole plugin strategy in one file: a new tool, a strategy that uses it, and a condition
that pairs it with the document-research task, driven through the library's own agent loop by
a scripted model. No index, no network.

    python examples/plugin_strategy.py

It prints the answer, the ranking, the actions the loop took, and the token usage. To run it
against a real model, drop the `generate=scripted_model` argument and pass `model="gpt-4o-mini"`
instead (OPENAI_API_KEY must be set). To run it over another collection, pass `research()` a
different list of documents in place of `DOCS`. docs/EXTENDING.md has the full contract.
"""
from __future__ import annotations

from agent_search import research
from agent_search.tokens import cap_tokens
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.strategies.conditions import condition
from agent_search.tools.base import Tool


# 1. the tool: its declaration (what the model sees) and its code (what a call returns)
class TitleLookup(Tool):
    name = "title_lookup"
    description = "Return every document whose title contains ALL the given words."
    parameters = {"type": "object", "properties": {"words": {"type": "string"}},
                  "required": ["words"]}

    def run(self, args: dict) -> str:
        words = (args or {}).get("words", "").lower().split()
        hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
        self.state.seen.update(u.doc_id for u in hits)     # first-seen order = the ranking
        return "\n".join(f"{u.doc_id}  {u.title}: {cap_tokens(u.body, 48)}" for u in hits) or "no match"


# 2. the strategy: which tools, under which names, with which options
register_strategy(Strategy(name="title_only", description="look documents up by title words",
                           tools=(TitleLookup(name="title_lookup"),)))

# 3. the condition: the research task with that strategy; runnable as `title_agent`
condition("title_agent", task="research", strategy="title_only")


DOCS = [
    {"_id": "curie", "title": "Marie Curie",
     "text": "Marie Curie discovered polonium and radium and won two Nobel Prizes."},
    {"_id": "meitner", "title": "Lise Meitner",
     "text": "Lise Meitner co-discovered nuclear fission in 1938."},
    {"_id": "hopper", "title": "Grace Hopper",
     "text": "Grace Hopper wrote the first compiler in 1952."},
]


def scripted_model(messages: list[dict]) -> str:
    """Stands in for an LLM: one tool call, then the answer."""
    if len(messages) <= 2:
        return '<tool_call>{"name": "title_lookup", "arguments": {"words": "Hopper"}}</tool_call>'
    return "<answer>1952</answer>"


if __name__ == "__main__":
    result = research("When did Grace Hopper write the first compiler?", DOCS,
                      strategy="title_agent", generate=scripted_model, max_steps=4)
    print("answer  :", result.answer)
    print("ranking :", result.ranking)
    print("steps   :", [s["action"] for s in result.steps])
    print("tokens  :", result.usage)
