"""The retriever registry + prompt auto-discovery — the extension points.

These pin the contract that adding a method/condition needs NO harness edit: a
builder registered with @register becomes selectable, and a YAML prompt profile
dropped on disk becomes resolvable, without touching evaluation/ or this registry.
"""
import pytest

from agent_search.retrievers import registry as R
from agent_search.retrievers.registry import RetrieverConfig, available, build_factory, register


# standalone retriever floors (unchanged) + the agent conditions of the NEW design
# (the method's two arms, plus their in-loop access baselines).
BUILTINS = {"grep", "bm25_local", "bm25_pyserini", "dense", "bql",
            "agent", "agent_codefix", "agent_codefix_grep",
            "agent_research", "agent_research_bm25", "agent_research_dci"}


def test_all_builtins_registered():
    assert BUILTINS <= available()


def test_retired_agent_conditions_are_gone():
    for gone in ("agent_bql", "agent_grep", "agent_bm25", "agent_dense",
                 "agent_tools", "agent_tools_bql", "agent_research_bql"):
        assert gone not in available()


def test_build_baseline_factory_no_model():
    from agent_search.retrievers.lexical.grep import GrepBaseline
    r = build_factory("grep")()
    assert isinstance(r, GrepBaseline) and r.name == "grep"


def test_build_dense_factory_does_not_load_encoder():
    from agent_search.retrievers.dense.dense import DenseRetriever
    r = build_factory("dense")()
    assert isinstance(r, DenseRetriever) and r.name == "dense"
    assert r._model is None


def test_build_direct_bql_factory_no_model():
    from agent_search.retrievers.structural.bql.retriever import BQLRetriever
    r = build_factory("bql")()
    assert isinstance(r, BQLRetriever) and r.name == "bql"


def test_build_code_fix_agent_stub():
    from agent_search.agent.retriever import AgentRetriever
    r = build_factory("agent_codefix", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever) and r.toolset == ("search", "fetch")
    assert r._arm == "code" and r.domain == "code"


def test_build_code_grep_baseline_stub():
    from agent_search.agent.retriever import AgentRetriever
    r = build_factory("agent_codefix_grep", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever) and r.toolset == ("grep", "read")
    assert r._arm == "grep" and r.domain == "code" and r.needs_files


def test_build_research_dci_baseline_stub():
    from agent_search.agent.retriever import AgentRetriever
    r = build_factory("agent_research_dci", RetrieverConfig(policy="stub"))()
    assert isinstance(r, AgentRetriever) and r.toolset == ("bash", "read")
    assert r._arm == "dci" and r.domain == "general" and not r.needs_files


def test_one_builder_serves_several_names():
    # one AgentRetriever; the condition is entirely the toolset + domain (from the profile)
    for name, want, arm in [("agent_codefix", "fetch", "code"),
                            ("agent_codefix_grep", "read", "grep"),
                            ("agent_research", "fetch", "doc"),
                            ("agent_research_bm25", "visit", "bm25"),
                            ("agent_research_dci", "read", "dci")]:
        r = build_factory(name, RetrieverConfig(policy="stub"))()
        assert want in r.toolset and r.tool == name and r._arm == arm


def test_build_search_fetch_agents_no_model():
    from agent_search.agent.retriever import AgentRetriever
    for name in ("agent_codefix", "agent_research"):
        r = build_factory(name, RetrieverConfig(policy="stub"))()
        # both arms drive the search -> fetch instrument (no localization `submit`)
        assert isinstance(r, AgentRetriever) and "search" in r.toolset and "fetch" in r.toolset
        assert "submit" not in r.toolset


def test_doc_arm_and_its_two_baselines_share_the_answer_contract():
    # research (search->fetch), research_bm25 (bm25_search->visit), research_dci (bash->read)
    # are the three doc conditions; all general-domain, <answer>-terminal arms.
    from agent_search.agent.retriever import AgentRetriever
    for name in ("agent_research", "agent_research_bm25", "agent_research_dci"):
        r = build_factory(name, RetrieverConfig(policy="stub"))()
        assert isinstance(r, AgentRetriever) and r.domain == "general"


def test_unknown_name_raises_with_choices():
    with pytest.raises(ValueError) as e:
        build_factory("does_not_exist")
    assert "does_not_exist" in str(e.value) and "grep" in str(e.value)


def test_register_adds_method_without_harness_edit():
    name = "__test_dummy_method__"
    try:
        @register(name)
        def _b(cfg, n):
            return lambda: build_factory("grep")()
        assert name in available()
        assert build_factory(name)().name == "grep"
    finally:
        R._REGISTRY.pop(name, None)


def test_duplicate_registration_of_different_builder_raises():
    name = "__test_dup__"
    try:
        register(name)(lambda cfg, n: None)
        with pytest.raises(ValueError):
            register(name)(lambda cfg, n: None)   # different fn, same name
    finally:
        R._REGISTRY.pop(name, None)


# --- prompt auto-discovery ---------------------------------------------------

def test_prompt_profiles_discovered_from_disk():
    from agent_search.prompts import DOMAINS, get_prompt_spec
    assert {"code", "general"} <= set(DOMAINS)
    for name in ("codefix", "codefix_grep", "research", "research_bm25", "research_dci"):
        spec = get_prompt_spec(name)
        # path is now the condition name (loader composes task x toolset from it)
        assert spec.name == name and spec.path == name


def test_every_discovered_profile_loads_and_renders():
    from agent_search.prompts import load_prompt_profile
    from agent_search.prompts.registry import PROMPTS
    for domain_specs in PROMPTS.values():           # every domain, not just "code"
        for spec in domain_specs.values():
            prof = load_prompt_profile(spec.path)
            assert prof.system.strip(), f"{spec.name} renders an empty system prompt"


def test_unknown_prompt_profile_raises():
    from agent_search.prompts import get_prompt_spec
    with pytest.raises(ValueError):
        get_prompt_spec("__nope__")
