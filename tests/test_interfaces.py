"""The contracts in agent_search.core.interfaces are real: the built-ins satisfy them and a
minimal user implementation of each is accepted by the loop and the harness."""
from agent_search.agent.loop import Task, run_episode
from agent_search.core import Model, OrderedSeen, Policy, Retriever, Workspace
from agent_search.corpus.units import units_from_documents

DOCS = [
    {"_id": "d1", "title": "Treaty of Guadalupe Hidalgo",
     "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
    {"_id": "d2", "title": "Treaty of Paris (1898)",
     "text": "The 1898 Treaty of Paris ended the Spanish-American War."},
]


def test_a_plain_callable_is_a_model():
    def generate(messages):
        return "<answer>ok</answer>"
    assert isinstance(generate, Model)
    assert isinstance(lambda m: "", Model)
    assert not isinstance(object(), Model)


def test_builtin_backends_satisfy_model():
    from agent_search.models import openai_compat_generate

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise AssertionError("not called")
    gen = openai_compat_generate(model="m", client=_Client())
    assert isinstance(gen, Model)


def test_a_minimal_retriever_plugs_into_the_registry(monkeypatch):
    from agent_search.retrievers import registry

    class Echo(Retriever):
        name = "echo"

        def index(self, units, key=None):
            self._ids = [u.doc_id for u in units]
            return self

        def search(self, query, k):
            return self._ids[:k]

    registry.register("echo_test_retriever")(lambda cfg, name: (lambda: Echo()))
    try:
        r = registry.build_factory("echo_test_retriever")()
        r.index(units_from_documents(DOCS))
        assert r.search("anything", 1) == ["d1"]
        assert "echo_test_retriever" in registry.available()
    finally:
        registry._REGISTRY.pop("echo_test_retriever", None)


def test_a_minimal_workspace_and_policy_run_an_episode():
    class Lookup:
        """A one-tool workspace: `lookup(q)` returns the matching doc's text."""
        tools = ("lookup",)

        def __init__(self, units):
            self.units = units
            self.seen = OrderedSeen()

        @property
        def surfaced(self):
            return list(self.seen)

        def run(self, name, args):
            if name != "lookup":
                return f"ERROR: unknown tool {name!r}"
            q = (args.get("q") or "").lower()
            hits = [u for u in self.units if q in (u.body or "").lower()]
            for u in hits:
                self.seen.add(u.doc_id)
            return "\n".join(f"{u.doc_id}: {u.body}" for u in hits) or "no match"

    class TwoStep:
        def propose(self, task, history):
            if not history:
                return '<tool_call>{"name": "lookup", "arguments": {"q": "mexican"}}</tool_call>'
            return "<answer>Treaty of Guadalupe Hidalgo</answer>"

    units = units_from_documents(DOCS)
    ws = Lookup(units)
    assert isinstance(ws, Workspace) and isinstance(TwoStep(), Policy)
    traj = run_episode(TwoStep(), Task("t", "Which treaty ended the Mexican-American War?"),
                       ws, units, max_steps=5, domain="general")
    assert traj.final_answer == "Treaty of Guadalupe Hidalgo"
    assert traj.stopped_reason == "answer"
    assert traj.located == ["d1"]          # the surfaced ranking, in first-seen order
