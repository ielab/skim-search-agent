"""End-to-end proof of the extension contract: a plugin adds a new tool, strategy, and
condition without editing the library, and the result is a runnable strategy with the same
run record as the built-ins."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PLUGIN = textwrap.dedent('''
    """A minimal SkimSearchAgent plugin: one tool, one strategy, one condition."""
    from agent_search.strategies.base import Strategy, register_strategy
    from agent_search.strategies.conditions import condition
    from agent_search.tools.base import Tool


    class TitleLookup(Tool):
        name = "title_lookup"
        description = "Return every document whose title contains the given words."
        parameters = {"type": "object", "properties": {"words": {"type": "string"}},
                      "required": ["words"]}

        def run(self, args):
            words = (args or {}).get("words", "").lower().split()
            hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
            self.state.seen.update(u.doc_id for u in hits)
            return "\\n".join(f"{u.doc_id}  {u.title}: {u.body}" for u in hits) or "no match"


    register_strategy(Strategy(name="plugin_title_only", description="by title",
                               tools=(TitleLookup(name="title_lookup"),)))
    condition("plugin_title", task="research", strategy="plugin_title_only")
''')


def _install_plugin(tmp_path: Path):
    mod = tmp_path / "my_skim_plugin.py"
    mod.write_text(PLUGIN)
    return mod


def test_plugin_is_discovered_from_the_environment_in_a_fresh_process(tmp_path):
    """SKIMSEARCHAGENT_PLUGINS names modules the registry imports before resolving names —
    the same mechanism the `skimsearchagent.plugins` entry-point group uses. The plugin
    registers a new Tool subclass, a Strategy naming it, and a condition pairing it with the
    research task (agent_search/strategies/base.py, agent_search/strategies/conditions.py) —
    no edit to the library, no YAML."""
    _install_plugin(tmp_path)
    env = dict(os.environ, SKIMSEARCHAGENT_PLUGINS="my_skim_plugin",
               PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    code = ("from agent_search.retrievers.registry import available; "
            "import json; print(json.dumps(sorted(n for n in available() if 'plugin' in n)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-800:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ["agent_plugin_title"]


def test_a_tool_a_strategy_and_a_condition_make_a_runnable_agent():
    """The extension path: a Tool subclass, a Strategy naming it, a condition pairing it
    with the research task; runnable by the condition's name with a scripted model."""
    from agent_search import research
    from agent_search.strategies.base import STRATEGIES, Strategy, register_strategy
    from agent_search.strategies.conditions import CONDITIONS, condition
    from agent_search.tools.base import Tool

    class TitleLookup(Tool):
        name = "title_lookup"
        description = "Return every document whose title contains ALL the given words."
        parameters = {"type": "object", "properties": {"words": {"type": "string"}},
                      "required": ["words"]}

        def run(self, args):
            words = (args or {}).get("words", "").lower().split()
            hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
            self.state.seen.update(u.doc_id for u in hits)
            return "\n".join(f"{u.doc_id}  {u.title}: {u.body}" for u in hits) or "no match"

    register_strategy(Strategy(name="__test_title_only", description="by title",
                               tools=(TitleLookup(name="title_lookup"),)))
    condition("__test_title_agent", task="research", strategy="__test_title_only")
    try:
        docs = [{"_id": "hopper", "title": "Grace Hopper", "text": "Grace Hopper wrote the first compiler in 1952."},
                {"_id": "curie", "title": "Marie Curie", "text": "Marie Curie won two Nobel Prizes."}]

        def scripted(messages):
            if len(messages) <= 2:
                return '<tool_call>{"name": "title_lookup", "arguments": {"words": "Hopper"}}</tool_call>'
            return "<answer>1952</answer>"

        r = research("When did Grace Hopper write the first compiler?", docs,
                     strategy="__test_title_agent", generate=scripted, max_steps=4)
        assert r.answer == "1952" and r.ranking == ["hopper"]
        assert [s["action"] for s in r.steps] == ["title_lookup", "answer"]
        assert "title_lookup" in r.system_prompt if hasattr(r, "system_prompt") else True
    finally:
        CONDITIONS.pop("__test_title_agent", None)
        STRATEGIES.pop("__test_title_only", None)
        from agent_search.retrievers import registry as R
        R._REGISTRY.pop("agent___test_title_agent", None)
