"""RepLLaMA (dense/repllama.py) and monoT5 (rerankers/monot5.py) on small random models with the
real tokenizers: the end token RepLLaMA pools survives truncation, the card's input formats, and
monoT5's score is log P(true) with the `Relevant:` cue kept on long pages. Skipped where the
tokenizers are not in the local cache (CI)."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tokenizer(model_id):
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{model_id} tokenizer not cached")


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    tok = _tokenizer("meta-llama/Llama-2-7b-hf")
    path = str(tmp_path_factory.mktemp("repllama"))
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    transformers.LlamaModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_repllama_pools_the_end_token_even_when_a_page_is_cut(tiny_llama):
    from agent_search.retrievers.dense.decoder_encoder import DecoderEncoder
    enc = DecoderEncoder(tiny_llama, max_seq_length=12, device="cpu", append_eos=True)
    batch = enc._batch(["query: short", "passage: " + "word " * 500])
    eos = enc.tokenizer.eos_token_id
    assert batch["input_ids"].shape[1] <= 12
    assert batch["input_ids"][:, -1].tolist() == [eos, eos]           # left padding: the end token is last
    plain = DecoderEncoder(tiny_llama, max_seq_length=12, device="cpu")._batch(["passage: " + "word " * 500])
    assert plain["input_ids"][0, -1].item() != eos                     # without it, truncation drops the end token


def test_repllama_card_formats(tiny_llama):
    from agent_search.corpus.units import CodeUnit
    from agent_search.retrievers.dense import DenseRetriever
    from agent_search.retrievers.dense.repllama import RepLlamaRetriever
    r = DenseRetriever(model=os.path.join(tiny_llama, "repllama-merged"), encoder=object())
    assert isinstance(r, RepLlamaRetriever) and r.query_prefix_for() == "query: "
    u = CodeUnit(doc_id="d", path="d", qualname="Llama", start_line=1, end_line=1, code="The llama is a camelid.",
                 body="The llama is a camelid.", title="Llama")
    assert r.document_template.format(title=u.qualname, body=u.code) == "passage: Llama The llama is a camelid."
    tok = transformers.AutoTokenizer.from_pretrained(tiny_llama)
    card = tok("query: What is llama?</s>")["input_ids"]
    from agent_search.retrievers.dense.decoder_encoder import DecoderEncoder
    ours = DecoderEncoder(tiny_llama, max_seq_length=512, device="cpu", append_eos=True)._batch(["query: What is llama?"])
    assert ours["input_ids"][0].tolist() == card                       # byte-identical to the model card's input


@pytest.fixture(scope="module")
def tiny_t5(tmp_path_factory):
    tok = _tokenizer("castorini/monot5-3b-msmarco-10k")
    path = str(tmp_path_factory.mktemp("monot5"))
    cfg = transformers.T5Config(vocab_size=len(tok), d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2,
                                decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    torch.manual_seed(0)
    transformers.T5ForConditionalGeneration(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_monot5_score_is_log_p_true_and_keeps_the_cue(tiny_t5):
    from agent_search.retrievers.rerankers.monot5 import MonoT5Reranker
    r = MonoT5Reranker(model=tiny_t5, max_length=40, device="cpu")
    tok, model, false_id, true_id, *_ = r._model()
    docs = ["The treaty ended the war.", "word " * 500]
    ids = r._inputs(tok, "what ended the war", docs)
    assert ids[0] == tok(r.prompt("what ended the war", docs[0]))["input_ids"]      # pygaggle's string, short page
    assert len(ids[1]) == 40 and tok.decode(ids[1][-4:]).endswith("Relevant:</s>")   # long page: cue kept
    got = r.scores("what ended the war", docs)
    with torch.no_grad():
        for d, s in zip(ids, got):
            logits = model(input_ids=torch.tensor([d]), decoder_input_ids=torch.tensor([[0]])).logits[0, 0]
            want = torch.log_softmax(logits[[false_id, true_id]].float(), dim=-1)[1].item()
            assert s == pytest.approx(want, rel=1e-4, abs=1e-5)
    ranked = r.rerank("what ended the war", [("a", docs[0]), ("b", docs[1])])
    assert sorted(d for d, _ in ranked) == ["a", "b"]


def test_laya_batched_scores_equal_its_own_api():
    """Laya scores each candidate in a batch; each score must equal what its RLAgent.system_one
    returns for that page alone (same calibrated temperature, same P(yes))."""
    try:
        from huggingface_hub import snapshot_download
        from agent_search.retrievers.rerankers.laya import CODE_FILES, LAYA_FILES
        snapshot_download("convaiinnovations/laya", allow_patterns=CODE_FILES + LAYA_FILES, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("convaiinnovations/laya not cached")
    from agent_search.retrievers.rerankers.laya import LayaReranker
    from agent_search.training.history import CURRENT, QueryContext
    r = LayaReranker(device="cpu")
    main, search = "Which treaty ended the Mexican-American War?", "treaty ended war 1848"
    docs = ["Treaty of Guadalupe Hidalgo\nThe Treaty of Guadalupe Hidalgo, signed in 1848, ended the Mexican-American War.",
            "Photosynthesis\nPhotosynthesis converts carbon dioxide and water into glucose using sunlight."]
    agent = r._agent()[0]
    token = CURRENT.set(QueryContext(question=main, text_of=lambda d: "", style="plain"))
    try:
        batched = r.scores(search, docs)
    finally:
        CURRENT.reset(token)
    api = [agent.system_one({"question": main, "search": search, "candidate": d},
                            {"x": {"type": "noul", "instructions": r.question, "criteria": r.criteria}})["answers"]["x"]["noul"]
           for d in docs]
    assert [round(x, 4) for x in batched] == api                     # the original question is in the state
