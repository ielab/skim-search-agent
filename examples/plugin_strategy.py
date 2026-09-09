#!/usr/bin/env python
"""A whole plugin strategy in one file: a new tool, toolset, condition and workspace, driven
through the library's own agent loop by a scripted model. No index, no network.

    python examples/plugin_strategy.py

It prints the answer, the ranking, the actions the loop took, and the token usage. To run it
against a real model, drop the `generate=scripted_model` argument and pass `model="gpt-4o-mini"`
instead (OPENAI_API_KEY must be set). To run it over another collection, pass `research()` a
different list of documents in place of `DOCS`. docs/EXTENDING.md has the full contract.
"""
from __future__ import annotations

from agent_search import research
from agent_search.agent.retriever import WorkspaceContext, register_workspace
from agent_search.core import OrderedSeen, cap_tokens
from agent_search.prompts.loader import register_tool, register_toolset
from agent_search.prompts.registry import register_condition

# 1. declare the tool (rendered into the system prompt as a JSON schema)
register_tool("title_lookup",
              description="Return every document whose title contains ALL the given words.",
              parameters={"type": "object", "properties": {"words": {"type": "string"}},
                          "required": ["words"]})
# 2. name the tool surface, 3. bind it to the document-research task template
register_toolset("title_only", ["title_lookup"])
register_condition("title_agent", task="research", toolset="title_only")


# 4. how the tool executes
class TitleWorkspace:
    tools = ("title_lookup",)

    def __init__(self, ctx: WorkspaceContext):
        self.units = ctx.units
        self.seen = OrderedSeen()          # first-seen order = the agent's retrieval ranking

    @property
    def surfaced(self):
        return list(self.seen)

    def run(self, name: str, args: dict) -> str:
        if name != "title_lookup":
            return f"ERROR: unknown tool {name!r}. Available tools: title_lookup."
        words = (args or {}).get("words", "").lower().split()
        hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
        self.seen.update(u.doc_id for u in hits)
        return "\n".join(f"{u.doc_id}  {u.title}: {cap_tokens(u.body, 48)}" for u in hits) or "no match"


register_workspace("title_arm", tools=("title_lookup",), builder=TitleWorkspace)


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
                      strategy="agent_title_agent", generate=scripted_model, max_steps=4)
    print("answer  :", result.answer)
    print("ranking :", result.ranking)
    print("steps   :", [s["action"] for s in result.steps])
    print("tokens  :", result.usage)
