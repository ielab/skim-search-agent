"""The retriever registry + prompt auto-discovery — the extension points.

These pin the contract that adding a method/condition needs NO harness edit: a
builder registered with @register becomes selectable, and a YAML prompt profile
dropped on disk becomes resolvable, without touching evaluation/ or this registry.
"""
import sys

import pytest

from agent_search.retrievers import registry as R
from agent_search.retrievers.registry import RetrieverConfig, available, build_factory, register


# standalone retriever floors + a sample of the agent conditions: the paper's 15 document
# conditions and the code-localization arm (codefix / codefix_grep / codefix_patch).
BUILTINS = {"grep", "bm25_local", "bm25_pyserini", "dense", "bql",
            "agent_research_snip", "agent_research_bm25", "agent_research_dci",
            "agent_codefix", "agent_codefix_grep", "agent_codefix_patch"}


def test_all_builtins_registered():
    assert BUILTINS <= available()


def test_retired_agent_conditions_are_gone():
    for gone in ("agent_bql", "agent_grep", "agent_bm25", "agent_dense",
                 "agent_tools", "agent_tools_bql", "agent_research_bql",
                 "agent_research", "agent_research_v2"):
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
    from agent_search.retrievers.bql.retriever import BQLRetriever
    r = build_factory("bql")()
    assert isinstance(r, BQLRetriever) and r.name == "bql"


def test_build_research_snip_agent_stub():
    from agent_search.evaluation.agent_runner import ConditionAgent
    r = build_factory("agent_research_snip", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent) and r.toolset == ("search_s", "fetch_s")
    assert r.strategy.name == "sieve_bm25" and r.domain == "general"


def test_build_research_dci_baseline_stub():
    from agent_search.evaluation.agent_runner import ConditionAgent
    r = build_factory("agent_research_dci", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent) and r.toolset == ("bash", "read")
    assert r.strategy.name == "dci" and r.domain == "general" and not r.needs_files


def test_one_builder_serves_several_names():
    # one runner; the condition (task x strategy) is what differs
    for name, want, strategy in [("agent_research_snip", "fetch_s", "sieve_bm25"),
                                 ("agent_research_bm25", "visit", "search_visit"),
                                 ("agent_research_dci", "read", "dci")]:
        r = build_factory(name, RetrieverConfig(policy="stub"))()
        assert want in r.toolset and r.condition.name == name[len("agent_"):]
        assert r.strategy.name == strategy


def test_build_search_fetch_agents_no_model():
    from agent_search.evaluation.agent_runner import ConditionAgent
    for name in ("agent_research_snip",):
        r = build_factory(name, RetrieverConfig(policy="stub"))()
        # the doc arm drives the search -> fetch instrument (no localization `submit`)
        assert isinstance(r, ConditionAgent) and "search_s" in r.toolset and "fetch_s" in r.toolset
        assert "submit" not in r.toolset


def test_doc_arm_and_its_two_baselines_share_the_answer_contract():
    # research_snip (search_s->fetch_s), research_bm25 (bm25_search->visit), research_dci
    # (bash->read) are three of the doc conditions; all general-domain, <answer>-terminal.
    for name in ("agent_research_snip", "agent_research_bm25", "agent_research_dci"):
        r = build_factory(name, RetrieverConfig(policy="stub"))()
        assert r.domain == "general" and r.task.terminal == "answer"


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


# --- plugin discovery (SKIMSEARCHAGENT_PLUGINS + `skimsearchagent.plugins` entry points) -----

def test_plugin_env_var_registers_retriever(tmp_path, monkeypatch):
    """`SKIMSEARCHAGENT_PLUGINS` names a dotted module (put on `sys.path` here); the module
    self-registers by calling `register(...)` at its own import time, exactly like a
    built-in retriever module does — the out-of-tree extension point this module's
    docstring documents."""
    plugin_dir = tmp_path / "my_plugin_pkg"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("")
    (plugin_dir / "plugin_mod.py").write_text(
        "from agent_search.retrievers.registry import register\n\n"
        "@register('my_plugin_retriever')\n"
        "def _build(cfg, name):\n"
        "    from agent_search.retrievers.lexical.grep import GrepBaseline\n"
        "    return lambda: GrepBaseline()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("SKIMSEARCHAGENT_PLUGINS", "my_plugin_pkg.plugin_mod")
    monkeypatch.setattr(R, "_loaded", False)      # force a fresh discovery pass to pick it up
    try:
        assert "my_plugin_retriever" in available()
        assert build_factory("my_plugin_retriever")().name == "grep"
    finally:
        R._REGISTRY.pop("my_plugin_retriever", None)
        sys.modules.pop("my_plugin_pkg.plugin_mod", None)
        sys.modules.pop("my_plugin_pkg", None)


def test_plugin_env_var_unknown_module_warns_but_does_not_crash_discovery(monkeypatch, capsys):
    """A typo'd/missing plugin module name must not take down discovery of every OTHER
    (built-in) retriever — logged as a warning, not raised."""
    monkeypatch.setenv("SKIMSEARCHAGENT_PLUGINS", "no_such_plugin_module_at_all")
    monkeypatch.setattr(R, "_loaded", False)
    assert "grep" in available()                  # built-ins still discovered
    assert "no_such_plugin_module_at_all" in capsys.readouterr().err


# --- prompt auto-discovery ---------------------------------------------------

def test_prompt_profiles_discovered_from_disk():
    from agent_search.prompts import DOMAINS, get_prompt_spec
    assert {"general"} <= set(DOMAINS)
    for name in ("research_snip", "research_bm25", "research_dci"):
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
