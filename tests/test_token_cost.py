"""The count-once token cost (`total_tokens_once`) must reflect what the model
actually held in its context window (real per-step `prompt_tokens`), not a sum over
raw/uncapped tool-observation text — a bash "read" observation can be 10-100x bigger
than what the policy actually lets into the prompt. See agent_search/evaluation/run_eval.py, the
TOKEN COST block in `_score_instance`."""
from agent_search.evaluation.datasets import Instance
from agent_search.evaluation.run_eval import evaluate


class _StubAgentRetriever:
    """Minimal stand-in for AgentRetriever: returns a fixed doc set and pre-serializes
    `last_trajectory_meta` the way the real agent retriever does (rows.jsonl shape)."""
    returns_full_set = True
    needs_files = False

    def __init__(self, trajectory, observations, completion_tokens):
        self.doc_ids = ["d1"]
        self._meta = {
            "observations": observations,
            "trajectory": trajectory,
            "completion_tokens": completion_tokens,
        }

    def index(self, units, key=None):
        return self

    def search(self, query, k):
        return self.doc_ids

    @property
    def last_trajectory_meta(self):
        return self._meta


def _instance():
    docs = [{"doc_id": "d1", "title": "Alpha", "text": "shared document"}]
    return Instance("q1", "local/shared", "0" * 40, "alpha", "", docs=docs,
                    gold_doc_ids={"d1"}, corpus_id="shared")


def test_total_tokens_once_uses_real_prompt_tokens_not_raw_observation_text():
    # A huge raw tool observation (simulating an uncapped bash "read" of ~50k chars)
    # that the policy would truncate before it ever entered the prompt: the model's
    # actual per-step prompt_tokens never exceeds ~500.
    huge_observation = "x " * 50_000
    trajectory = [
        {"prompt_tokens": 100, "completion_tokens": 10},
        {"prompt_tokens": 300, "completion_tokens": 10},
        {"prompt_tokens": 500, "completion_tokens": 10},   # max
        {"prompt_tokens": 420, "completion_tokens": 10},   # windowed back down
    ]
    retriever = _StubAgentRetriever(trajectory, [huge_observation], completion_tokens=40)

    res = evaluate([_instance()], lambda: retriever, ks=[1], level="function")
    row = res["rows"][0]

    assert row["token_source"] == "prompt_tokens"
    assert row["initial_prompt_tokens"] == 100
    # context_once_tokens = max(prompt_tokens) - initial = 500 - 100 = 400, NOT a
    # multi-thousand-token count over the raw 50k-char observation.
    assert row["context_once_tokens"] == 400
    assert row["retrieved_doc_tokens"] == 400          # back-compat field mirrors it
    assert row["output_tokens"] == 40
    assert row["total_tokens_once"] == 100 + 400 + 40
    # sanity: the old buggy behavior would have been orders of magnitude bigger
    assert row["total_tokens_once"] < 2000


def test_total_tokens_once_falls_back_to_text_estimate_when_prompt_tokens_missing():
    # Older trajectories (or non-vLLM backends) may not report per-step prompt_tokens;
    # the fallback path should kick in and be flagged, not silently assumed accurate.
    trajectory = [
        {"completion_tokens": 10},
        {"completion_tokens": 10},
    ]
    observations = ["short doc one", "short doc two"]
    retriever = _StubAgentRetriever(trajectory, observations, completion_tokens=20)

    res = evaluate([_instance()], lambda: retriever, ks=[1], level="function")
    row = res["rows"][0]

    assert row["token_source"] == "fallback"
    assert row["initial_prompt_tokens"] == 0
    assert row["context_once_tokens"] > 0
    assert row["retrieved_doc_tokens"] == row["context_once_tokens"]
    assert row["total_tokens_once"] == row["initial_prompt_tokens"] + row["context_once_tokens"] + row["output_tokens"]
