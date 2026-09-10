"""A checkpoint is served in the precision it was trained in: the trainer writes `dtype` into
the serving note, the encoder loads with it, caches and index metadata carry it, and
`DENSE_DTYPE` overrides it."""
from __future__ import annotations

import json

import pytest

from agent_search.retrievers.dense import dense as D


def test_trainer_note_records_the_training_precision(tmp_path):
    from agent_search.training.retriever import TrainConfig, write_serving_note
    cfg = TrainConfig(train_data="x.jsonl", output_dir=str(tmp_path / "m"), bf16=True)
    note = json.loads(open(write_serving_note(cfg)).read())
    assert note["dtype"] == "bfloat16"
    cfg32 = TrainConfig(train_data="x.jsonl", output_dir=str(tmp_path / "m32"), bf16=False)
    assert json.loads(open(write_serving_note(cfg32)).read())["dtype"] == "float32"


def test_dtype_resolution_order(tmp_path, monkeypatch):
    ck = tmp_path / "ckpt"; ck.mkdir()
    (ck / "skimsearchagent_dense.json").write_text(json.dumps({"dtype": "bfloat16"}))
    monkeypatch.delenv("DENSE_DTYPE", raising=False)
    assert D.resolve_dtype(str(ck)) == "bfloat16"                  # the note
    assert D.resolve_dtype("BAAI/bge-base-en-v1.5") == "float32"   # no note: float32
    monkeypatch.setenv("DENSE_DTYPE", "fp16")
    assert D.resolve_dtype(str(ck)) == "float16"                   # the knob wins
    assert D.resolve_dtype(str(ck), device="cpu") == "bfloat16"    # no fp16 matmul on a CPU
    monkeypatch.setenv("DENSE_DTYPE", "int8")
    with pytest.raises(ValueError, match="DENSE_DTYPE"):
        D.resolve_dtype(str(ck))


def test_precision_separates_caches_and_is_recorded_on_the_index(tmp_path, monkeypatch):
    import numpy as np
    from agent_search.corpus.units import units_from_documents
    monkeypatch.delenv("DENSE_DTYPE", raising=False)
    monkeypatch.delenv("DENSE_INDEX_PATH", raising=False)
    ck = tmp_path / "ckpt"; ck.mkdir()
    (ck / "skimsearchagent_dense.json").write_text(json.dumps({"dtype": "bfloat16"}))

    class Enc:
        max_seq_length = 512

        def encode(self, texts, **kw):
            return np.ones((len(texts), 4), dtype="float32")

    r = D.DenseRetriever(str(ck), encoder=Enc(), index_root=str(tmp_path / "idx"))
    assert r.dtype == "bfloat16" and r._cache_dir("k").endswith("-bfloat16/k")
    r.index(units_from_documents([{"_id": "1", "title": "t", "text": "x"}]), key="k")
    meta = json.loads((tmp_path / "idx" / "dense" / r._cache_dir("k").split("/dense/")[1] / "meta.json").read_text())
    assert meta["dense_dtype"] == "bfloat16" and meta["dense_model"] == str(ck)
    r32 = D.DenseRetriever("BAAI/bge-base-en-v1.5", encoder=Enc(), index_root=str(tmp_path / "idx"))
    assert r32.dtype == "float32" and "-float32" not in r32._cache_dir("k")


def test_bf16_tensors_become_float32_numpy():
    torch = pytest.importorskip("torch")
    out = D._to_numpy(torch.ones(2, 3, dtype=torch.bfloat16))
    assert out.dtype.name == "float32" and out.shape == (2, 3)
