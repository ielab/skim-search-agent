"""Qwen/Qwen3-Embedding-0.6B as a selectable dense embedder (additive to bge-base-en-v1.5,
the existing general-domain default) — the env knob `DENSE_MODEL`, its cache-namespace
isolation, faithful model-card usage (last-token pooling is handled entirely by
sentence-transformers' bundled `modules.json`/`1_Pooling` config — nothing to test at this
layer; the query INSTRUCT prefix is this repo's own responsibility, tested below), and
dim-agnostic vector_index round-trip at Qwen3-Embedding's 1024-dim output.

No torch/sentence-transformers import in these tests (CPU/CI-safe) except the opt-in
integration check at the bottom, gated exactly like test_indri_dense.py's
`INDRI_DENSE_INTEGRATION` pattern.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.dense import vector_index as vi
from agent_search.retrievers.dense.dense import DenseRetriever, _QUERY_PREFIX
from agent_search.evaluation.datasets import default_dense_model


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- 1. DENSE_MODEL env knob -------------------------------------------------------------

def test_default_dense_model_unset_keeps_bge_compat_default(monkeypatch):
    monkeypatch.delenv("DENSE_MODEL", raising=False)
    assert default_dense_model("general") == "BAAI/bge-base-en-v1.5"
    assert default_dense_model("code") == "nomic-ai/CodeRankEmbed"


def test_default_dense_model_env_overrides_general_domain_only(monkeypatch):
    monkeypatch.setenv("DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    assert default_dense_model("general") == "Qwen/Qwen3-Embedding-0.6B"
    # code domain keeps its own code-trained embedder regardless of DENSE_MODEL — a general
    # text embedder over code is a domain mismatch this knob must never silently cause.
    assert default_dense_model("code") == "nomic-ai/CodeRankEmbed"


def test_dense_belief_default_model_env_knob_in_a_fresh_process():
    """DenseBelief.DEFAULT_MODEL is resolved at import time (same pattern as
    AGENT_DEFAULT_CONDITION / oneshot_rag.DEFAULT_MAX_TOKENS) — verified in a subprocess so
    this test doesn't reload the already-imported module in-process."""
    code = ("from agent_search.retrievers.indri.dense_belief import DEFAULT_MODEL; "
            "print(DEFAULT_MODEL)")

    env_unset = {k: v for k, v in os.environ.items() if k != "DENSE_MODEL"}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env_unset,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "BAAI/bge-base-en-v1.5"

    env_set = dict(env_unset, DENSE_MODEL="Qwen/Qwen3-Embedding-0.6B")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env_set,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "Qwen/Qwen3-Embedding-0.6B"


def test_resolve_env_knobs_records_dense_model(monkeypatch):
    """agent_search/evaluation/run_eval.py's provenance snapshot must capture the raw knob (so two run
    dirs that differ only by DENSE_MODEL are distinguishable in config.json)."""
    from agent_search.evaluation.run_eval import _resolve_env_knobs

    monkeypatch.delenv("DENSE_MODEL", raising=False)
    assert _resolve_env_knobs()["DENSE_MODEL"] is None

    monkeypatch.setenv("DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    assert _resolve_env_knobs()["DENSE_MODEL"] == "Qwen/Qwen3-Embedding-0.6B"


# --- 2. query instruction prefix (model-card fidelity) -----------------------------------

def test_qwen3_embedding_query_prefix_matches_model_card():
    """Per Qwen/Qwen3-Embedding-0.6B's README (`get_detailed_instruct`) and its
    `config_sentence_transformers.json` `prompts.query` string verbatim: NO space between
    'Query:' and the query text. Documents get no prefix (prompts.document == '')."""
    expected = ("Instruct: Given a web search query, retrieve relevant passages that answer "
                "the query\nQuery:")
    for model_id in ("Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B",
                      "Qwen/Qwen3-Embedding-8B"):
        assert _QUERY_PREFIX[model_id] == expected


class _CapturingEncoder:
    """Records exactly what text DenseRetriever.search hands to encode()."""

    def __init__(self):
        self.query_texts: list = []

    def encode(self, texts, **kwargs):
        self.query_texts.extend(texts)
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def test_dense_retriever_applies_qwen3_instruct_prefix_to_queries_not_docs(tmp_path):
    units = [CodeUnit("d1", "d1.txt", "t", 1, 1, "some document body")]
    enc = _CapturingEncoder()
    r = DenseRetriever("Qwen/Qwen3-Embedding-0.6B", index_root=str(tmp_path), encoder=enc)
    r.index(units, key="k")
    enc.query_texts.clear()
    r.search("what is the capital of France?", k=1)
    assert enc.query_texts == [
        "Instruct: Given a web search query, retrieve relevant passages that answer the "
        "query\nQuery:what is the capital of France?"
    ]


# --- 3. cache-namespace isolation ---------------------------------------------------------

def test_qwen3_and_bge_never_share_a_cache_namespace(tmp_path):
    units = [CodeUnit("d1", "d1.txt", "t", 1, 1, "x")]
    enc_qwen, enc_bge = _CapturingEncoder(), _CapturingEncoder()
    r_qwen = DenseRetriever("Qwen/Qwen3-Embedding-0.6B", index_root=str(tmp_path), encoder=enc_qwen)
    r_bge = DenseRetriever("BAAI/bge-base-en-v1.5", index_root=str(tmp_path), encoder=enc_bge)

    assert r_qwen._cache_dir("corpus") != r_bge._cache_dir("corpus")
    assert "Qwen__Qwen3-Embedding-0.6B" in r_qwen._cache_dir("corpus")
    assert "BAAI__bge-base-en-v1.5" in r_bge._cache_dir("corpus")

    r_qwen.index(units, key="corpus")
    # building the qwen cache must not create (or touch) the bge namespace at all
    assert not os.path.exists(r_bge._cache_dir("corpus"))
    assert os.path.exists(r_qwen._cache_dir("corpus"))
    assert r_bge.is_cached("corpus") is False
    assert r_qwen.is_cached("corpus") is True


# --- 4. dim-agnostic vector_index round-trip at Qwen3-Embedding's 1024-dim ----------------

QWEN3_DIM = 1024


def _corpus(n=200, d=QWEN3_DIM, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, [f"d{i}" for i in range(n)]


def test_flat_index_round_trip_at_qwen3_dim(tmp_path):
    emb, ids = _corpus()
    idx = vi.build_index(emb, ids, backend="flat")
    vi.save_index(idx, str(tmp_path))
    loaded = vi.load_index(str(tmp_path))
    assert loaded is not None and loaded.backend == "flat"
    qv = emb[17]
    assert loaded.search(qv, 1) == ["d17"]
    assert loaded.search(qv, 5) == idx.search(qv, 5)


faiss_available = vi._faiss() is not None
needs_faiss = pytest.mark.skipif(not faiss_available, reason="faiss not installed")


@needs_faiss
@pytest.mark.parametrize("backend", ["hnsw", "ivfpq"])
def test_faiss_backend_round_trip_at_qwen3_dim(tmp_path, backend):
    emb, ids = _corpus(n=4000, d=QWEN3_DIM)
    idx = vi.build_index(emb, ids, backend=backend)
    assert idx.backend == backend
    vi.save_index(idx, str(tmp_path))
    loaded = vi.load_index(str(tmp_path))
    assert loaded.backend == backend
    hits = loaded.search(emb[42], 10)
    assert "d42" in hits


# --- 5. opt-in real-model / real-corpus integration check --------------------------------

_BC_CORPUS = os.path.join(REPO, "data", "browsecomp_plus_structured", "corpus.jsonl")
_QWEN_SNAPSHOT_ROOT = os.path.join(
    os.environ.get("HF_HOME", ""), "hub", "models--Qwen--Qwen3-Embedding-0.6B")

integration = pytest.mark.skipif(
    not (os.environ.get("DENSE_QWEN_INTEGRATION")
         and os.path.exists(_BC_CORPUS) and os.path.isdir(_QWEN_SNAPSHOT_ROOT)),
    reason="opt-in real-model check: set DENSE_QWEN_INTEGRATION=1 with the local browsecomp "
           "corpus + Qwen3-Embedding-0.6B HF snapshot (HF_HOME) present")


@integration
def test_real_qwen3_embedding_surfaces_gold_doc_on_mini_corpus(tmp_path):
    """200-doc mini-check (mirrors test_indri_dense.py's bge equivalent) with the REAL
    Qwen3-Embedding-0.6B encoder, offline: last-token pooling + instruct prefix must rank the
    query's own gold doc highly, not just embed-without-crashing."""
    from agent_search.corpus.units import units_from_documents
    docs = []
    with open(_BC_CORPUS, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            docs.append(json.loads(line))
    units = units_from_documents(docs)
    gold = {"5412", "82002", "86190", "18639", "41759"}
    assert gold & {u.doc_id for u in units}

    r = DenseRetriever("Qwen/Qwen3-Embedding-0.6B", index_root=str(tmp_path), device="cpu")
    r.index(units, key="bc_mini_200")
    top = r.search("college festival supporting palestinians", k=10)
    assert gold & set(top)
