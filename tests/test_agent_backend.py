"""Offline wiring test for the real LLM backend (no GPU/model needed).

An injected fake OpenAI-compatible client verifies the full chain: backend ->
AgentPolicy -> run_episode -> the final answer, exactly as it runs against a
real vLLM server, without loading a model.
"""
import sys
from types import ModuleType, SimpleNamespace

from agent_search.agent.backbone import openai_compat_generate, vllm_generate
from agent_search.agent.loop import Task, run_episode
from agent_search.agent.policies import AgentPolicy
from agent_search.strategies import CONDITIONS


def _fake_client(responses):
    state = {"i": 0}

    def create(**kwargs):
        txt = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=txt))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_backend_parses_chat_response():
    gen = openai_compat_generate(model="x", client=_fake_client(["hello world"]))
    assert gen("any prompt") == "hello world"


def test_backend_sends_prompt_and_model():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    openai_compat_generate(model="my-model", client=client)("hello")
    assert seen["model"] == "my-model"
    assert seen["messages"][0]["content"] == "hello"
    # Tongyi-DeepResearch native sampling (Alibaba-NLP/DeepResearch react_agent.py).
    assert seen["temperature"] == 0.6 and seen["top_p"] == 0.95
    assert seen["presence_penalty"] == 1.1
    # the fabricated-observation guards must both be present in the stop set.
    assert "\n<tool_response>" in seen["stop"] and "<tool_response>" in seen["stop"]
    assert seen["seed"] == 42                    # default: fixed seed for reproducibility


def test_backend_passes_seed_and_temperature():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    openai_compat_generate(model="m", client=client, temperature=0.2, seed=42)("hi")
    assert seen["temperature"] == 0.2 and seen["seed"] == 42


def test_vllm_backend_sampling_params(monkeypatch):
    """The vLLM path must wire the same Tongyi-native sampling/stop config into
    SamplingParams (Alibaba-NLP/DeepResearch react_agent.py). vLLM isn't installed
    offline, so stub `from vllm import SamplingParams` with a recording fake and
    inject a fake llm/tokenizer."""
    seen = {}

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: seen.update(kw) or SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_out = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[SimpleNamespace(text="hi there", token_ids=[3])])
    fake_llm = SimpleNamespace(generate=lambda texts, params: [fake_out])
    fake_tok = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt-text")

    gen = vllm_generate(model="m", llm=fake_llm, tokenizer=fake_tok)
    assert gen("hello") == "hi there"
    assert seen["temperature"] == 0.6 and seen["top_p"] == 0.95
    assert seen["presence_penalty"] == 1.1 and seen["seed"] == 42
    assert "\n<tool_response>" in seen["stop"] and "<tool_response>" in seen["stop"]


# --- cluster-readiness: the NEW baselines drive the SAME backend dispatch ---------------
# Neither make_generate nor vllm_generate/AgentPolicy/run_episode branch on condition name —
# so the two new baselines (codefix_grep, research_dci) exercise the identical vLLM/API
# dispatch code path as the method's own conditions. These tests are the static (no-GPU)
# proof that the vLLM backbone path stays intact for them, mirroring
# test_vllm_backend_sampling_params / test_full_agent_episode_with_real_backend_shape above.

def test_make_generate_routes_backbone_through_vllm_for_any_condition(monkeypatch):
    """make_generate(backend='vllm') for a NON-OpenAI model (the trained backbone) constructs
    a real vllm.LLM and dispatches through vllm_generate — the SAME call regardless of which
    condition's registry builder makes it, so the codefix_grep/research_dci baselines get the
    identical vLLM path as codefix/research, not a condition-specific branch. `vllm` isn't
    installed offline, so stub the module (same pattern as test_vllm_backend_sampling_params)."""
    import agent_search.agent.backbone as B

    seen = {}
    fake_out = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[SimpleNamespace(text="hi there", token_ids=[3])])

    class FakeLLM:
        def __init__(self, **kw):
            seen["llm_kwargs"] = kw
        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt-text")
        def generate(self, texts, params):
            seen["sampling_params"] = params
            return [fake_out]

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    # backend='vllm' (the cluster default) + a non-OpenAI model id -> in-process vLLM,
    # exactly what scripts/run.sh does for RETRIEVER=agent_codefix_grep / agent_research_dci.
    gen = B.make_generate("Alibaba-NLP/Tongyi-DeepResearch-30B-A3B", backend="vllm", tp=2)
    assert gen("hello") == "hi there"
    assert seen["llm_kwargs"]["model"] == "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    assert seen["llm_kwargs"]["tensor_parallel_size"] == 2      # TP flows through unmodified
    assert seen["sampling_params"].temperature == 0.6           # unchanged Tongyi-native default


def test_research_dci_episode_drives_through_agentpolicy_and_run_episode():
    """The DCI baseline's full episode shape: bash -> read -> <answer>, through the SAME
    AgentPolicy/run_episode chain as the method (only the workspace + toolset differ)."""
    from agent_search.corpus.units import units_from_documents
    from agent_search.tools.base import EpisodeState, ToolBox
    from agent_search.tools.bash.tool import Bash
    from agent_search.tools.read.tool import Read

    docs = [{"_id": "d1", "title": "Doc One", "text": "The answer is forty-two."}]
    units = units_from_documents(docs)
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="What is the answer?")
    bash = Bash().bind(state, units, ubyid, {}, corpus_key="agent_backend_dci")
    read = Read().bind(state, units, ubyid, {}, corpus_key="agent_backend_dci")
    ws = ToolBox([bash, read], state)
    gen = openai_compat_generate(model="x", client=_fake_client([
        '<tool_call>{"name":"bash","arguments":{"command":"grep -rl \'answer\' ."}}</tool_call>',
        '<tool_call>{"name":"read","arguments":{"path":"d1.txt"}}</tool_call>',
        "<answer>forty-two</answer>",
    ]))
    traj = run_episode(AgentPolicy(generate=gen, system=CONDITIONS["research_dci"].render()),
                       Task("q", "What is the answer?"), ws, units, domain="general")
    assert traj.stopped_reason == "answer"
    assert traj.final_answer == "forty-two"
    assert [s.name for s in traj.steps][:2] == ["bash", "read"]
