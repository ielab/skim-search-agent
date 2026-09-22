"""Tongyi calls its native `visit` tool inside the ITER and dedup strategies, which expose
`search` and `get_document` only. DIVER's client serves such a call as get_document; the
library answered "unknown tool" and burned the turn (868 times over the 830 published ITER
questions). The alias makes `visit` reach get_document without changing the rendered prompt."""
import json

from agent_search import research
from agent_search.strategies.base import STRATEGIES
from agent_search.strategies.conditions import CONDITIONS
from lucene_support import index_root, require_jvm

require_jvm()

DOCS = [
    {"_id": "d_guadalupe", "title": "Treaty of Guadalupe Hidalgo",
     "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
    {"_id": "d_paris", "title": "Treaty of Paris (1898)",
     "text": "The 1898 Treaty of Paris ended the Spanish-American War."},
]
QUESTION = "Which treaty ended the Mexican-American War?"


def test_visit_reaches_get_document_in_the_iter_strategies():
    for name in ("iter_bm25", "dedup_bm25"):
        calls = []

        def generate(messages):
            calls.append(messages)
            if len(calls) == 1:
                return '<tool_call>{"name": "bm25_search", "arguments": {"query": "Mexican-American War treaty"}}</tool_call>'
            if len(calls) == 2:
                return '<tool_call>{"name": "visit", "arguments": {"docid": "d_guadalupe", "goal": "check the year"}}</tool_call>'
            return "<answer>Treaty of Guadalupe Hidalgo</answer>"

        r = research(QUESTION, DOCS, strategy=name, generate=generate, max_steps=5, index_root=index_root())
        visit = next(s for s in r.steps if s["action"] == "visit")
        assert not visit["observation"].startswith("ERROR"), visit["observation"][:120]
        assert "1848" in visit["observation"]
        assert r.answer == "Treaty of Guadalupe Hidalgo"


def test_the_alias_is_not_rendered_in_the_prompt():
    for name in ("iter_bm25", "iter_dense", "dedup_bm25", "dedup_dense"):
        strategy = STRATEGIES[name]
        assert "visit" not in strategy.tool_names
        tool = next(t for t in strategy.tools if t.name == "get_document")
        assert "visit" in tool.aliases
    cond = CONDITIONS.get("research_iter_bm25")
    if cond is not None:
        assert '"visit"' not in json.dumps(cond.render())
