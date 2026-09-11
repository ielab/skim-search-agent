"""Procedures: one-shot RAG and the plan-and-search team run through `ProcedureAgent`, and
their record has the single-agent shape (steps, members, surfaced documents, an answer)."""
from __future__ import annotations

import pytest

from agent_search.corpus.units import units_from_documents
from agent_search.procedures.base import Procedure, ProcedureContext
from agent_search.procedures.plan_and_search import PlanAndSearch, member_condition, parse_plan
from agent_search.procedures.rag import OneShotRag
from agent_search.strategies import CONDITIONS
from agent_search.strategies.base import STRATEGIES
from lucene_support import build_engines, corpus_key, index_root, require_jvm

require_jvm()

DOCS = [
    {"_id": "d1", "title": "Treaty of Guadalupe Hidalgo",
     "text": "The treaty that ended the Mexican-American War was signed in 1848 in Guadalupe Hidalgo."},
    {"_id": "d2", "title": "Mexican-American War",
     "text": "The war lasted from 1846 to 1848 between the United States and Mexico."},
    {"_id": "d3", "title": "Bird song", "text": "birds sing at dawn"},
]
QUESTION = "Which treaty ended the Mexican-American War and when was it signed?"


@pytest.fixture(scope="module")
def corpus():
    units = units_from_documents(DOCS)
    engines = build_engines(units)
    return units, corpus_key(units), engines


def test_parse_plan_reads_numbered_and_bulleted_lines():
    text = "Here is the plan:\n1. Which treaty ended the war?\n2) When was it signed?\n- a third one\nfour"
    assert parse_plan(text, 5) == ["Which treaty ended the war?", "When was it signed?", "a third one"]
    assert parse_plan(text, 1) == ["Which treaty ended the war?"]
    assert parse_plan("", 3) == []


def test_team_strategy_unions_its_members_engines():
    assert STRATEGIES["plan_and_search"].engines == STRATEGIES["sieve_bm25"].engines
    assert STRATEGIES["plan_and_search_visit"].engines == STRATEGIES["search_visit"].engines
    assert PlanAndSearch(searcher="sieve_bm25").members == ("sieve_bm25",)
    assert member_condition("sieve_bm25").task.name == "research"
    assert "plan_and_search" in CONDITIONS


def test_stub_team_runs_members_and_records_them(corpus):
    units, key, _ = corpus
    from agent_search.retrievers.registry import RetrieverConfig, build_factory
    cfg = RetrieverConfig(index_root=index_root(), domain="general", policy="stub")
    agent = build_factory("agent_plan_and_search", cfg)().index(units, key=key)
    ranking = agent.search(QUESTION, 5)
    meta = agent.last_trajectory_meta
    assert meta["actions"] == ["plan", "member", "synthesize"]     # no model: the plan is the question
    assert len(meta["members"]) == 1 and meta["members"][0]["actions"]
    assert set(ranking) <= {"d1", "d2", "d3"} and meta["surfaced_docs"]
    assert set(ranking) <= set(meta["surfaced_docs"])
    assert agent.system_prompt()                                   # planner, synthesizer and member prompts


def test_scripted_team_plans_runs_two_members_and_synthesizes(corpus):
    units, key, _ = corpus
    from agent_search.api import build_agent
    seen = []

    def gen(msgs):
        system = msgs[0]["content"]
        seen.append(system[:40])
        if system.startswith("You are the planner"):
            return "1. Which treaty ended the Mexican-American War?\n2. When was it signed?"
        if system.startswith("You are the synthesizer"):
            assert "Findings:" in msgs[-1]["content"]
            return "<answer>Treaty of Guadalupe Hidalgo, 1848</answer>"
        turns = sum(1 for m in msgs if m["role"] == "assistant")
        if turns == 0:
            return '<tool_call>{"name":"search","arguments":{"query":"treaty Mexican-American War"}}</tool_call>'
        return "<answer>1848</answer>"

    agent = build_agent("plan_and_search", generate=gen, index_root=index_root()).index(units, key=key)
    ranking = agent.search(QUESTION, 5)
    meta = agent.last_trajectory_meta
    assert meta["final_answer"] == "Treaty of Guadalupe Hidalgo, 1848"
    assert meta["actions"] == ["plan", "member", "member", "synthesize"]
    assert [m["actions"] for m in meta["members"]] == [["search", "answer"], ["search", "answer"]]
    assert ranking and ranking[0] == "d1"
    assert seen[0].startswith("You are the planner") and seen[-1].startswith("You are the synthesizer")


def test_rag_procedure_returns_one_retrieve_step(corpus):
    units, key, engines = corpus
    ubyid = {u.doc_id: u for u in units}
    ctx = ProcedureContext(engines={"bm25": engines.get("bm25")}, ubyid=ubyid, units=units,
                           generate=lambda msgs: "<answer>1848</answer>")
    res = OneShotRag("bm25").run(QUESTION, ctx)
    assert [s.name for s in res.steps] == ["retrieve"] and res.doc_ids and res.raw == "<answer>1848</answer>"
    assert res.members == [] and res.surfaced == res.doc_ids


def test_procedure_base_contract():
    p = Procedure()
    assert p.members == () and p.all_engines() == ()
    with pytest.raises(NotImplementedError):
        p.run("q", ProcedureContext(engines={}, ubyid={}, units=[]))
