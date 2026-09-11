"""End-to-end proof of the extension contract: a plugin adds a new tool, toolset, condition,
and workspace without editing the library, and the result is a runnable strategy with the same
run record as the built-ins."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PLUGIN = textwrap.dedent('''
    """A minimal SkimSearchAgent plugin: one tool, one toolset, one condition, one workspace."""
    from agent_search.legacy.retriever import register_workspace
    from agent_search.core.seen import OrderedSeen
    from agent_search.legacy.prompts.loader import register_tool, register_toolset
    from agent_search.legacy.prompts.registry import register_condition

    register_tool("title_lookup",
                  description="Return every document whose title contains the given words.",
                  parameters={"type": "object", "properties": {"words": {"type": "string"}},
                              "required": ["words"]})
    register_toolset("title_only", ["title_lookup"])
    register_condition("plugin_title", task=TASK_PATH, toolset="title_only")


    class TitleWorkspace:
        tools = ("title_lookup",)

        def __init__(self, ctx):
            self.units = ctx.units
            self.seen = OrderedSeen()

        @property
        def surfaced(self):
            return list(self.seen)

        def run(self, name, args):
            if name != "title_lookup":
                return f"ERROR: unknown tool {name!r}. Available tools: title_lookup."
            words = (args or {}).get("words", "").lower().split()
            hits = [u for u in self.units if all(w in (u.title or "").lower() for w in words)]
            for u in hits:
                self.seen.add(u.doc_id)
            return "\\n".join(f"{u.doc_id}  {u.title}: {u.body}" for u in hits) or "no match"


    register_workspace("plugin_title", tools=("title_lookup",), builder=TitleWorkspace)
''')

TASK_MD = textwrap.dedent('''
    ---
    name: plugin_title
    domain: general
    description: a plugin task template
    ---
    You answer questions using ONLY the tools below. Call one tool per turn as
    <tool_call>{"name": ..., "arguments": {...}}</tool_call>; finish with <answer>...</answer>.

    {{tools}}
''')


def _install_plugin(tmp_path: Path):
    task = tmp_path / "plugin_task.md"
    task.write_text(TASK_MD)
    mod = tmp_path / "my_skim_plugin.py"
    mod.write_text(f"TASK_PATH = {str(task)!r}\n" + PLUGIN)
    return mod


def test_plugin_registers_a_runnable_strategy(tmp_path, monkeypatch):
    mod = _install_plugin(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    import my_skim_plugin  # noqa: F401 — registration happens on import
    importlib.reload(my_skim_plugin)

    from agent_search import research
    from agent_search.legacy.prompts import load_condition
    from agent_search.retrievers.registry import available

    assert "agent_plugin_title" in available()
    prof = load_condition("plugin_title")
    assert prof.tool_names == ("title_lookup",)
    assert '"name":"title_lookup"' in prof.system      # rendered into the <tools> block

    def generate(messages):
        if len(messages) <= 2:
            return '<tool_call>{"name": "title_lookup", "arguments": {"words": "Guadalupe"}}</tool_call>'
        return "<answer>1848</answer>"

    docs = [{"_id": "g", "title": "Treaty of Guadalupe Hidalgo", "text": "Signed in 1848."},
            {"_id": "p", "title": "Treaty of Paris", "text": "Signed in 1898."}]
    res = research("When was the treaty signed?", docs, strategy="agent_plugin_title",
                   generate=generate, max_steps=4)
    assert res.answer == "1848"
    assert res.ranking == ["g"]
    assert [s["action"] for s in res.steps] == ["title_lookup", "answer"]
    assert "Guadalupe" in res.observations[0]


def test_plugin_is_discovered_from_the_environment_in_a_fresh_process(tmp_path):
    """SKIMSEARCHAGENT_PLUGINS names modules the registry imports before resolving names —
    the same mechanism the `skimsearchagent.plugins` entry-point group uses."""
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
    """The 0.3 extension path: a Tool subclass, a Strategy naming it, a condition pairing it
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
