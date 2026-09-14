"""The plain decoder encoder serves a decoder checkpoint the way Tevatron encodes it: the end
token is kept, the last position is pooled, vectors are unit length. The check against a real
checkpoint runs only where the ITER snapshot is on disk (the cluster); the surface checks always."""
import os

import numpy as np
import pytest

from agent_search.retrievers.dense.base import local_snapshot

ITER = "ielabgroup/ITER-Qwen3-Embedding-0.6B"


def test_decoder_encoder_rejects_unknown_pooling():
    from agent_search.retrievers.dense.decoder_encoder import DecoderEncoder
    with pytest.raises(ValueError):
        DecoderEncoder(path="unused", pooling="first_token")


@pytest.mark.skipif(not local_snapshot(ITER), reason="the ITER checkpoint is not on this machine")
def test_decoder_encoder_matches_tevatron_style_encoding():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from agent_search.retrievers.dense.decoder_encoder import DecoderEncoder
    snap = local_snapshot(ITER)
    texts = ["A short document about the Treaty of Guadalupe Hidalgo (1848).", "word " * 900]
    enc = DecoderEncoder(snap, max_seq_length=512, device="cpu", torch_dtype=torch.float32)
    ours = enc.encode(texts, batch_size=2)
    assert ours.shape == (2, enc.model.config.hidden_size)
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0, atol=1e-4)
    # Tevatron: tokenizer (end token kept under truncation), left padding, last position
    tok = AutoTokenizer.from_pretrained(snap)
    tok.padding_side = "left"
    batch = tok(texts, max_length=512, truncation=True, padding=True, return_tensors="pt")
    assert batch["input_ids"][0, -1].item() == tok.convert_tokens_to_ids("<|endoftext|>")
    model = AutoModel.from_pretrained(snap, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, -1], dim=-1).numpy()
    assert float((ours * ref).sum(1).min()) > 0.99


def test_dense_belief_reads_the_same_cache_the_probe_checks(monkeypatch, tmp_path):
    """DENSE_SEQ_LENGTH decides the cache directory for the belief exactly as for the probe."""
    from agent_search.retrievers.dense import DenseBelief, DenseRetriever
    monkeypatch.setenv("DENSE_SEQ_LENGTH", "512")
    probe = DenseRetriever(model="ielabgroup/ITER-Qwen3-Embedding-0.6B", index_root=str(tmp_path), encoder=object())
    belief = DenseBelief(model="ielabgroup/ITER-Qwen3-Embedding-0.6B", index_root=str(tmp_path), encoder=object())
    assert belief._retriever._cache_dir("k") == probe._cache_dir("k")
    assert "-sl512" in belief._retriever._cache_dir("k")
    monkeypatch.delenv("DENSE_SEQ_LENGTH")
    assert "-sl1024" in DenseBelief(model="ielabgroup/ITER-Qwen3-Embedding-0.6B", index_root=str(tmp_path), encoder=object())._retriever._cache_dir("k")
